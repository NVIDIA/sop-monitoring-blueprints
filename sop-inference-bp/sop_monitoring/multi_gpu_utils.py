# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from __future__ import annotations

import itertools
import logging
import threading
import traceback
import time
import uuid

from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.multiprocessing as mp

_LOGGER = logging.getLogger(__name__)

def get_gpu_ids() -> list[int]:
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_ids = list(range(gpu_count))
        _LOGGER.info("Detected %s CUDA GPUs: %s", gpu_count, gpu_ids)
    else:
        _LOGGER.error("No CUDA GPUs detected or PyTorch not available, using CPU")
        raise RuntimeError("No CUDA GPUs detected or PyTorch not available. At least one GPU is required.")
    return gpu_ids

def init_mp_spawn() -> None:

    mp_start_method = mp.get_start_method(allow_none=True)
    if mp_start_method is None:
        mp.set_start_method('spawn')
        _LOGGER.info("Set mp start method to spawn")
    elif mp_start_method != 'spawn':
        _LOGGER.error("mp start method is already set to %s, but it should be spawn", mp_start_method)
        raise RuntimeError(f"mp start method is already set to {mp_start_method}, but it should be spawn")
    else:
        _LOGGER.info("mp start method is already set to spawn")

class MultiWorkerManager:
    """
    [EXPERIMENTAL]
    This class abstracts multi-workers with request-response pattern.

    FIXME:
    Not sure if this can be useful now, but let's see.
    If it's useful, we can rewrite MultiGPUVLMManager and MultiGpuDdmNet to compose this class.
    """

    class Worker(ABC):
        """
        Abstract base class for worker.
        The class instance is used in a process like below:

        Process-1:
            # init code can be here.
            worker.initialize()

            # main event loop
            while True:
                # get request from request queue.
                response = worker.process_request(request)
                # put response to response queue
        """
        @abstractmethod
        def get_name(self) -> str:
            pass

        @abstractmethod
        def initialize(self) -> None:
            pass

        @abstractmethod
        def process_request(self, request: Any) -> Any:
            pass

    @dataclass
    class _Request:
        request_id: str
        request: Any

    @dataclass
    class _Response:
        request_id: str
        response: Any
        error: str

    @dataclass
    class _WorkerProcess:
        worker_name: str
        process: mp.Process
        request_queue: mp.Queue


    def __init__(self,
                 workers: list[Worker],
                 max_queue_size: int = 1000):

        init_mp_spawn()

        self._workers = workers
        self._max_queue_size = max_queue_size

        self._is_running = False
        self._response_thread = None
        self._response_queue = mp.Queue()

        self._pending_requests = {}  # request_id -> Future mapping
        self._pending_lock = threading.Lock()

        self._worker_processes = self._setup_worker_processes()
        self._worker_processes_iterator_cycle = itertools.cycle(self._worker_processes)
        self._worker_processes_lock = threading.Lock()

        self._start_response_handler()

    def submit_request(self, request: Any) -> Future:
        """Submit a request to the worker manager"""
        worker_process = self._get_next_worker_process()
        if worker_process is None:
            raise RuntimeError("No GPU workers available. "
                               "This might be caused by failures to initialize GPU works. "
                               "You might want to check logs of the service.")

        future = Future()
        request_id = uuid.uuid4().hex
        with self._pending_lock:
            self._pending_requests[request_id] = future

        _LOGGER.info("Submitting request %s to worker process %s", request_id, worker_process.worker_name)
        _request = self._Request(request_id=request_id, request=request)
        worker_process.request_queue.put(_request)

        return future

    def shutdown(self) -> None:
        self._is_running = False

        for worker_process in self._worker_processes:
            worker_process.process.terminate()
            worker_process.process.join(timeout=10)
            if worker_process.process.is_alive():
                _LOGGER.error("Worker process %s did not terminate within 10 seconds", worker_process.worker_name)
                worker_process.process.kill()

        with self._pending_lock:
            for future in self._pending_requests.values():
                future.set_exception(RuntimeError("Worker manager shutdown"))


    def _init_single_worker_process(self, worker: Worker) -> _WorkerProcess | None:

        worker_process = None
        _LOGGER.info("Create process for worker %s...", worker.get_name())
        try:
            request_queue = mp.Queue(maxsize=self._max_queue_size)
            process = mp.Process(target=self._worker_process_target,
                                 args=(worker, request_queue, self._response_queue))
            process.start()

            time.sleep(2)
            if not process.is_alive():
                _LOGGER.error("Worker process %s failed to start", worker.get_name())
                return None

            worker_process = self._WorkerProcess(worker.get_name(), process, request_queue)

        except Exception as e:
            error_msg = traceback.format_exc()
            _LOGGER.error("Failed to initialize worker process %s: %s", worker.get_name(), error_msg)

        return worker_process

    def _setup_worker_processes(self) -> list[_WorkerProcess]:

        _LOGGER.info("Creating %s processes...", len(self._workers))

        worker_processes = []
        for worker in self._workers:
            worker_process = self._init_single_worker_process(worker)
            if worker_process is not None:
                worker_processes.append(worker_process)

        return worker_processes

    def _get_next_worker_process(self) -> _WorkerProcess:
        with self._worker_processes_lock:
            ret = next(self._worker_processes_iterator_cycle)
            dead_gpus = set()
            while not ret.process.is_alive():
                dead_gpus.add(ret.gpu_id)
                _LOGGER.warning("GPU worker %s is not alive. Selecting next worker.", ret.gpu_id)
                ret = next(self._worker_processes_iterator_cycle)
                if ret.gpu_id in dead_gpus:
                    _LOGGER.error("Running out of GPU workers. All GPU workers are dead."
                                  "You might want to check service logs to see what went wrong.")
                    return None

            return ret

    def _start_response_handler(self) -> None:
        """Start the response handler thread"""
        self._is_running = True
        self._response_thread = threading.Thread(
            target=self._response_handler_loop,
            daemon=True
        )
        self._response_thread.start()

    def _response_handler_loop(self) -> None:
        """Handle responses from worker processes"""
        _LOGGER.info("Response handler started")
        while self._is_running:
            try:
                _response = self._response_queue.get(timeout=10)
            except:
                continue

            with self._pending_lock:
                if _response.request_id in self._pending_requests:
                    future = self._pending_requests.pop(_response.request_id)
                    if _response.error:
                        future.set_exception(Exception(_response.error))
                    else:
                        future.set_result(_response.response)
                else:
                    _LOGGER.warning("Received response for unknown request: %s", _response.request_id)

    @staticmethod
    def _worker_process_target(worker: Worker,
                               request_queue: mp.Queue,
                               response_queue: mp.Queue) -> None:

        _LOGGER.info("Starting worker process %s...", worker.get_name())
        worker.initialize()

        while True:
            _request = request_queue.get()

            try:
                response = worker.process_request(_request.request)
                error_msg = ""
            except Exception as e:
                error_msg = (
                    f"Worker {worker.get_name()} failed to process request {_request.request_id}. "
                    f"Error: {e}. Traceback: {traceback.format_exc()}"
                )
                _LOGGER.error(error_msg)
                response = None

            finally:
                _response = MultiWorkerManager._Response(request_id=_request.request_id, response=response, error=error_msg)
                response_queue.put(_response)

        _LOGGER.info("Worker process %s terminated", worker.get_name())
