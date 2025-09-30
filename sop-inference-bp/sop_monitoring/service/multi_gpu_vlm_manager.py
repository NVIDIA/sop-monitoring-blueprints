# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import itertools
import time
import logging
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp

from ..multi_gpu_utils import init_mp_spawn
from ..vlm import CosmosReason1 as VlmModel

_LOGGER = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Represents a single inference request"""
    request_id: str
    system_prompt: str
    prompt: str
    # FIXME: video_filenames is a bit legacy.
    # When it is an empty array, the gpu worker process use video_filename, chunk_start_second, chunk_end_second
    video_filenames: list[str]

    # new approach
    video_filename: str | None = None
    chunk_start_second: float | None = None
    chunk_end_second: float | None = None


@dataclass
class InferenceResponse:
    """Represents an inference response"""
    request_id: str
    results: list
    error: str | None = None


# Sentinel object to signal worker shutdown
_SHUTDOWN_SENTINEL = "SHUTDOWN"


def gpu_worker_process(gpu_id: int, model_dir: str, request_queue: mp.Queue, response_queue: mp.Queue):
    """
    GPU worker process function - runs independently for each GPU.

    Args:
        gpu_id: GPU device ID to use
        model_dir: Path to VLM model directory
        request_queue: Queue to receive inference requests
        response_queue: Queue to send back responses
    """
    # Set up logging for this process
    logger = logging.getLogger(f"GPU-{gpu_id}")
    logger.info("Starting GPU worker process for GPU %s", gpu_id)

    # Handle shutdown gracefully
    def signal_handler(signum, frame):
        logger.info("GPU %s received shutdown signal", gpu_id)
        return

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Initialize VLM on this GPU
        logger.info("Initializing VLM on GPU %s", gpu_id)
        vlm_instance = VlmModel(model_dir, device=f"cuda:{gpu_id}")

        logger.info("=========================================")
        logger.info("VLM initialized successfully on GPU %s", gpu_id)
        logger.info("=========================================")

        processed_count = 0

        # Main processing loop
        while True:
            try:
                # Get next request (blocking)
                request = request_queue.get()

                # Check for shutdown signal
                if request == _SHUTDOWN_SENTINEL:
                    logger.info("GPU %s received shutdown sentinel", gpu_id)
                    break

                logger.debug("GPU %s processing request %s", gpu_id, request.request_id)

                # Process the request
                start_time = time.time()
                try:
                    if request.video_filenames:
                        results = vlm_instance.inference(request.prompt, request.video_filenames, system_prompt=request.system_prompt)
                    else:
                        result = vlm_instance.chunk_and_infer(
                            request.prompt,
                            request.video_filename,
                            request.chunk_start_second,
                            request.chunk_end_second,
                           system_prompt=request.system_prompt)
                        results = [result]

                    response = InferenceResponse(
                        request_id=request.request_id,
                        results=results,
                        error=None
                    )

                    inference_time = time.time() - start_time
                    processed_count += 1

                    logger.debug("GPU %s completed request %s in %.2fs (total: %s)",
                               gpu_id, request.request_id, inference_time, processed_count)

                except Exception as e:
                    logger.error("GPU %s failed processing request %s: %s", gpu_id, request.request_id, e)
                    response = InferenceResponse(
                        request_id=request.request_id,
                        results=[],
                        error=str(e)
                    )

                # Send response back
                response_queue.put(response)

            except KeyboardInterrupt:
                logger.info("GPU %s interrupted", gpu_id)
                break
            except Exception as e:
                logger.error("Unexpected error in GPU %s worker: %s", gpu_id, e)
                time.sleep(1)  # Brief pause before retrying

    except Exception as e:
        logger.error("Failed to initialize GPU %s worker: %s", gpu_id, e)
    finally:
        logger.info("GPU %s worker process ending", gpu_id)


@dataclass
class GPUWorker:
    """Represents a single GPU worker process"""
    gpu_id: int
    process: mp.Process
    request_queue: mp.Queue


class MultiGPUVLMManager:
    """
    Manages multiple NvilaPyTorch instances across all available GPUs
    with concurrent inference capabilities through threadpool.
    """

    def __init__(self, model_dir: str, max_queue_size: int = 1000):
        """
        Initialize the multi-GPU VLM manager.

        Args:
            model_dir: Path to the VLM model directory
            max_queue_size: Maximum number of requests in the queue
        """
        init_mp_spawn()

        self.model_dir = model_dir
        self.max_queue_size = max_queue_size
        self.gpu_workers: list[GPUWorker] = []
        self.response_queue = mp.Queue()  # Shared response queue for all workers
        self.is_running = False

        # Thread pool for handling responses
        self.response_thread = None
        self.pending_requests = {}  # request_id -> Future mapping
        self.pending_lock = threading.Lock()

        self._setup_gpu_workers()
        self._start_response_handler()

        # Round-robin counter for worker selection
        self.round_robin_lock = threading.Lock()
        self.gpu_workers_iterator_cycle = itertools.cycle(self.gpu_workers)

    def _detect_gpus(self) -> list[int]:
        """Detect available GPUs and return their IDs"""
        gpu_ids = []

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_ids = list(range(gpu_count))
            _LOGGER.info("Detected %s CUDA GPUs: %s", gpu_count, gpu_ids)
        else:
            _LOGGER.error("No CUDA GPUs detected or PyTorch not available, using CPU")
            raise RuntimeError("No CUDA GPUs detected or PyTorch not available. At least one GPU is required.")

        return gpu_ids

    def _init_single_gpu_worker(self, gpu_id: int) -> GPUWorker | None:
        """Initialize a single GPU worker process"""
        try:
            _LOGGER.info("Creating GPU worker process for GPU %s", gpu_id)

            # Create request queue for this GPU worker
            request_queue = mp.Queue(maxsize=self.max_queue_size)

            # Create the worker process
            process = mp.Process(
                target=gpu_worker_process,
                args=(gpu_id, self.model_dir, request_queue, self.response_queue),
                name=f"GPU-Worker-{gpu_id}"
            )

            worker = GPUWorker(
                gpu_id=gpu_id,
                process=process,
                request_queue=request_queue
            )

            # Start the process
            process.start()
            _LOGGER.info("Started GPU worker process for GPU %s (PID: %s)", gpu_id, process.pid)

            # Wait a bit to ensure process started successfully
            time.sleep(3)
            if not process.is_alive():
                _LOGGER.error("GPU worker process for GPU %s failed to start", gpu_id)
                return None

            return worker

        except Exception as e:
            _LOGGER.error("Failed to create GPU worker for GPU %s: %s", gpu_id, e)
            return None

    def _setup_gpu_workers(self):
        """Initialize GPU worker processes for all available GPUs"""
        gpu_ids = self._detect_gpus()

        _LOGGER.info("Creating %s GPU worker processes...", len(gpu_ids))

        # Create GPU workers sequentially to avoid race conditions during process startup
        workers = []
        for gpu_id in gpu_ids:
            worker = self._init_single_gpu_worker(gpu_id)
            if worker is not None:
                workers.append(worker)

        self.gpu_workers = workers

        if not self.gpu_workers:
            raise RuntimeError("Failed to initialize any GPU worker processes")

        successful_gpus = [w.gpu_id for w in self.gpu_workers]
        _LOGGER.info("Successfully created %s GPU worker processes on GPUs: %s",
                    len(self.gpu_workers), successful_gpus)

    def _start_response_handler(self):
        """Start the response handler thread"""
        self.is_running = True
        self.response_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Response-Handler")
        self.response_thread.submit(self._response_handler_loop)

    def _response_handler_loop(self):
        """Handle responses from GPU worker processes"""
        _LOGGER.info("Response handler started")

        while self.is_running:
            try:
                # Get response with timeout to allow periodic shutdown checks
                try:
                    response = self.response_queue.get(timeout=10)
                except:
                    continue

                # Find the corresponding future
                with self.pending_lock:
                    if response.request_id in self.pending_requests:
                        future = self.pending_requests.pop(response.request_id)

                        # Set result or exception on the future
                        if response.error:
                            future.set_exception(Exception(response.error))
                        else:
                            future.set_result(response.results)
                    else:
                        _LOGGER.warning("Received response for unknown request: %s", response.request_id)

            except Exception as e:
                _LOGGER.error("Error in response handler: %s", e)

        _LOGGER.info("Response handler ended")

    def _get_next_worker_round_robin(self) -> GPUWorker | None:
        """Get the next GPU worker using round-robin selection"""
        with self.round_robin_lock:
            return next(self.gpu_workers_iterator_cycle)

    def submit_chunk_and_infer(self,
                               request_id: str,
                               system_prompt: str,
                               prompt: str,
                               video_filename: str,
                               chunk_start_second: float,
                               chunk_end_second: float) -> Future:
        """
        Submit a chunk and infer request to a GPU worker process.
        """
        if not self.is_running:
            raise RuntimeError("VLM GPU Manager is not running")

        worker = self._get_next_worker_round_robin()
        if worker is None:
            raise RuntimeError("No GPU workers available")

        future = Future()

        with self.pending_lock:
            if request_id in self.pending_requests:
                raise ValueError(f"Request ID '{request_id}' already exists in pending requests. Request IDs must be unique.")
            self.pending_requests[request_id] = future

        request = InferenceRequest(
            request_id=request_id,
            system_prompt=system_prompt,
            prompt=prompt,
            video_filenames=[], # to trigger new approarch
            video_filename=video_filename,
            chunk_start_second=chunk_start_second,
            chunk_end_second=chunk_end_second
        )

        try:
            _LOGGER.debug("Sending chunk and infer request %s to GPU %s", request_id, worker.gpu_id)
            worker.request_queue.put(request)
            _LOGGER.debug("Chunk and infer request %s sent to GPU %s", request_id, worker.gpu_id)

        except Exception as e:
            with self.pending_lock:
                self.pending_requests.pop(request_id, None)
            raise RuntimeError(f"Failed to submit chunk and infer request to GPU {worker.gpu_id}: {e}")

        return future

    def submit_inference(self, request_id: str, system_prompt: str, prompt: str, video_filenames: list[str],
                        timeout: float | None = None) -> Future:
        """
        Submit an inference request to a GPU worker process.

        Args:
            request_id: Unique identifier for the request
            prompt: Text prompt for the VLM
            video_filenames: List of video file paths
            timeout: Optional timeout for the request

        Returns:
            Future object that will contain the inference results

        Raises:
            RuntimeError: If no GPU workers are available
        """
        if not self.is_running:
            raise RuntimeError("VLM Manager is not running")

        # Get an available worker
        worker = self._get_next_worker_round_robin()
        if worker is None:
            raise RuntimeError("No GPU workers available")

        # Create future for this request
        future = Future()

        # Store the future for response matching (check for duplicates)
        with self.pending_lock:
            if request_id in self.pending_requests:
                raise ValueError(f"Request ID '{request_id}' already exists in pending requests. Request IDs must be unique.")
            self.pending_requests[request_id] = future

        # Create the request
        request = InferenceRequest(
            request_id=request_id,
            system_prompt=system_prompt,
            prompt=prompt,
            video_filenames=video_filenames
        )

        try:
            # Send request to the selected worker process
            _LOGGER.debug("Sending request %s to GPU %s", request_id, worker.gpu_id)
            worker.request_queue.put(request)
            _LOGGER.debug("Request %s sent to GPU %s", request_id, worker.gpu_id)

        except Exception as e:
            # Remove from pending requests if submission failed
            with self.pending_lock:
                self.pending_requests.pop(request_id, None)
            raise RuntimeError(f"Failed to submit request to GPU {worker.gpu_id}: {e}")

        return future

    def shutdown(self, timeout: float = 30.0):
        """Shutdown the VLM manager and cleanup resources"""
        _LOGGER.info("Shutting down VLM Manager...")

        self.is_running = False

        # Send shutdown signals to all worker processes
        for worker in self.gpu_workers:
            try:
                if worker.process.is_alive():
                    worker.request_queue.put(_SHUTDOWN_SENTINEL, block=False)
                    _LOGGER.info("Sent shutdown signal to GPU %s", worker.gpu_id)
            except Exception as e:
                _LOGGER.warning("Failed to send shutdown signal to GPU %s: %s", worker.gpu_id, e)

        # Wait for processes to terminate gracefully
        _LOGGER.info("Waiting for GPU worker processes to terminate...")
        for worker in self.gpu_workers:
            try:
                worker.process.join(timeout=timeout/len(self.gpu_workers))
                if worker.process.is_alive():
                    _LOGGER.warning("Force terminating GPU %s process", worker.gpu_id)
                    worker.process.terminate()
                    worker.process.join(timeout=5)
                    if worker.process.is_alive():
                        _LOGGER.error("Force killing GPU %s process", worker.gpu_id)
                        worker.process.kill()
            except Exception as e:
                _LOGGER.error("Error shutting down GPU %s worker: %s", worker.gpu_id, e)

        # Shutdown response handler
        if self.response_thread:
            self.response_thread.shutdown(wait=True, timeout=10)

        # Cancel any remaining pending requests
        with self.pending_lock:
            for request_id, future in self.pending_requests.items():
                future.cancel()
            self.pending_requests.clear()

        _LOGGER.info("VLM Manager shutdown complete")

    def __del__(self):
        """Destructor - safety net for cleanup if shutdown() wasn't called"""
        try:
            if hasattr(self, 'is_running') and self.is_running:
                _LOGGER.warning("VLMManager being garbage collected without proper shutdown - cleaning up")
                self.shutdown(timeout=10)  # Shorter timeout for destructor
        except Exception as e:
            # Don't raise exceptions from destructor
            _LOGGER.error("Error during VLMManager cleanup in destructor: %s", e)
