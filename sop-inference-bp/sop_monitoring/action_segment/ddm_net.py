# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
Use DDM-Net to perform action segmentation.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import os
import time
import threading
import traceback
import signal
import uuid
import sys

from concurrent.futures import Future
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import torch.multiprocessing as mp
from torchcodec.decoders import VideoDecoder

from ..multi_gpu_utils import (
    init_mp_spawn,
)

from ..utils import setup_logging

_LOGGER = logging.getLogger(__name__)

# Sentinel object to signal worker shutdown
_SHUTDOWN_SENTINEL = "SHUTDOWN"

@dataclass
class DdmNetRequest:
    """Represents a single DDM-Net request"""
    request_id: str
    input_video_path: str
    start_sec: float
    end_sec: float
    batch_size: int


@dataclass
class DdmNetResponse:
    """Represents a single DDM-Net response"""
    request_id: str
    scores: list[float]
    error: str = ""


@dataclass
class DdmNetGpuWorker:
    gpu_id: int
    process: mp.Process
    request_queue: mp.Queue

@dataclass
class VideoMetaData:
    fps: float
    duration_sec: float
    total_frames: int

def gpu_worker_process(gpu_id: int,
                       checkpoint_path: str,
                       resolution: int,
                       frames_per_side: int,
                       request_queue: mp.Queue,
                       response_queue: mp.Queue):

    log_level_name = os.environ.get("ACTION_SEGMENT_SERVICE_LOG_LEVEL", "INFO")
    setup_logging(log_level_name)

    logger = logging.getLogger(f"DDM-Net-GPU-{gpu_id}")
    logger.info("Starting DDM-Net GPU worker process for GPU %s", gpu_id)

    try:
        model = _load_model(checkpoint_path, frames_per_side, gpu_id)
        preprocess_pipeline = T.Compose([
            T.Resize((resolution, resolution)),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        logger.info("DDM-Net model loaded successfully on GPU %s", gpu_id)
    except Exception as e:
        traceback_msg = traceback.format_exc()
        logger.error("Failed to load DDM-Net model on GPU %s: %s", gpu_id, traceback_msg)
        raise

    while True:
        try:
            request = request_queue.get()
            if request == _SHUTDOWN_SENTINEL:
                logger.info("DDM-Net GPU %s received shutdown signal", gpu_id)
                break

            logger.debug("DDM-Net GPU %s processing request %s, file: %s, start_sec: %s, end_sec: %s, batch_size: %s",
                         gpu_id, request.request_id, request.input_video_path, request.start_sec, request.end_sec, request.batch_size)

            # Use torchcodec to decode video frames for the specified time segment
            decoder = VideoDecoder(request.input_video_path)
            fps = decoder.metadata.average_fps

            # Convert time segment to frame indices
            start_frame = int(request.start_sec * fps)
            end_frame = int(request.end_sec * fps)

            # Extract frames for the time segment
            # torchcodec returns frames in TCHW format by default
            vframes = decoder[start_frame:end_frame]  # Shape: (T, C, H, W)

            vframes = vframes.to(f"cuda:{gpu_id}")

            with torch.no_grad():
                preprocessed_frames = preprocess_pipeline(vframes)

                # Free original vframes from GPU memory immediately after preprocessing
                del vframes
                torch.cuda.empty_cache()

                # preprocessed_frames is in TCHW format (T=time, C=channels, H=height, W=width)
                # We need to create sliding windows and batch them into NTCHW format
                batch_size = request.batch_size
                total_frames = preprocessed_frames.shape[0]
                window_size = 2 * frames_per_side + 1
                scores = []

                # Create sliding windows from preprocessed_frames
                batch_windows = []

                # Generate sliding windows
                for i in range(total_frames - window_size + 1):
                    # Extract window of frames: [i:i+window_size, :, :, :]
                    window = preprocessed_frames[i:i+window_size]  # Shape: (window_size, C, H, W)
                    batch_windows.append(window)

                    # Process batch when it's full
                    if len(batch_windows) == batch_size:
                        # Stack windows to form batch: (N, T, C, H, W)
                        batch_tensor = torch.stack(batch_windows, dim=0)

                        # Run inference
                        outputs, _, _ = model(batch_tensor)
                        if isinstance(outputs, (list, tuple)):
                            output = outputs[-1]

                        # Get boundary scores for this batch
                        window_scores = F.softmax(output, dim=1)[:, 1].cpu().numpy()
                        scores.extend(window_scores)

                        # Reset batch
                        batch_windows = []

                # Process remaining windows if batch size >= 2
                if len(batch_windows) >= 2:
                    batch_tensor = torch.stack(batch_windows, dim=0)

                    outputs, _, _ = model(batch_tensor)
                    if isinstance(outputs, (list, tuple)):
                        output = outputs[-1]

                    window_scores = F.softmax(output, dim=1)[:, 1].cpu().numpy()
                    scores.extend(window_scores)

                # Skip final batch if it has fewer than 2 windows
                elif len(batch_windows) > 0:
                    logger.debug("DDM-Net GPU %s: Skipping final batch with %d windows (< 2)",
                               gpu_id, len(batch_windows))

                # Pad scores for frames that couldn't be processed at beginning and end
                scores = list(map(float, scores))
                pad_start = frames_per_side
                pad_end = total_frames - len(scores) - pad_start
                padded_scores = [0.0] * pad_start + scores + [0.0] * max(0, pad_end)

                # Send response back
                response = DdmNetResponse(
                    request_id=request.request_id,
                    scores=padded_scores,
                    error=""
                )
                response_queue.put(response)

                logger.debug("DDM-Net GPU %s completed request %s, processed %d windows",
                           gpu_id, request.request_id, len(scores))

        except Exception as e:
            traceback_msg = traceback.format_exc()
            error_msg = f"DDM-Net GPU {gpu_id} failed to process request: {traceback_msg}"
            logger.error(error_msg)
            # Send error response
            error_response = DdmNetResponse(
                request_id=getattr(request, 'request_id', 'unknown'),
                scores=[],
                error=error_msg
            )
            response_queue.put(error_response)

    logger.info("DDM-Net GPU %s exiting", gpu_id)


class MultiGpuDdmNet:
    def __init__(self,
                 checkpoint_path: str,
                 resolution: int,
                 frames_per_side: int,
                 frames_per_segment_hint: int,
                 max_queue_size: int = 1000):

        init_mp_spawn()

        self.checkpoint_path = checkpoint_path
        self.resolution = resolution
        self.frames_per_side = frames_per_side
        self.frames_per_segment_hint = frames_per_segment_hint
        self.max_queue_size = max_queue_size

        # Response handler
        self.is_running = False
        self.response_thread = None
        self.response_queue = mp.Queue()

        # Pending requests
        self.pending_requests = {}  # request_id -> Future mapping
        self.pending_lock = threading.Lock()

        # GPU workers
        gpu_ids = self._detect_gpus()
        self.gpu_workers = self._setup_gpu_workers(gpu_ids)
        self.num_workers = len(self.gpu_workers)
        self.gpu_workers_iterator_cycle = itertools.cycle(self.gpu_workers)
        self.gpu_workers_lock = threading.Lock()

        try:
            self.time_out_sec = int(os.getenv("ACTION_SEGMENT_DDM_NET_TIMEOUT_SEC", 900))
        except ValueError:
            _LOGGER.error("Invalid value for ACTION_SEGMENT_DDM_NET_TIMEOUT_SEC: %s. Using default value of 900 seconds.",
                          os.getenv("ACTION_SEGMENT_DDM_NET_TIMEOUT_SEC"))
            self.time_out_sec = 900

        self._start_response_handler()

    def process_video(self,
                      input_video_path: str,
                      threshold: float,
                      nms_sec: float,
                      batch_size: int) -> tuple[list[float], list[float]]:
        """
        Complete pipeline to detect boundaries and calculate chunk time boundaries.
        This function must be thread-safe because the main event loop can call it in parallel for multiple requests.

        Args:
            input_video_path: Path to the input video file
            threshold: Threshold for boundary detection
            nms_sec: The half-length of the window of NMS filtering in seconds.
            batch_size: Number of windows per batch

        Returns:
            Tuple[List[float], List[float]]: (chunk_start_seconds, chunk_end_seconds)
        """

        _LOGGER.info("Processing video: %s", input_video_path)

        ddm_scores, video_metadata = self.get_ddm_scores(input_video_path, batch_size)

        if nms_sec == 0.0:
            nms_sec = 0.025 * video_metadata.duration_sec
            _LOGGER.debug("Using default nms_sec of 0.025 * video duration in seconds: %.2f", nms_sec)

        # Detect boundaries based on threshold
        nms_size = int(nms_sec * video_metadata.fps)
        boundaries = detect_boundaries(ddm_scores, threshold, nms_size)
        _LOGGER.debug("Detected %d boundaries in %d frames", len(boundaries), len(ddm_scores))

        # Calculate chunk boundaries based on detected boundaries
        chunk_start_seconds, chunk_end_seconds = calculate_chunk_boundaries(
            boundaries,
            video_metadata.fps,
            video_metadata.duration_sec,
            video_metadata.total_frames,
        )

        _LOGGER.info("Created %d time-based chunks", len(chunk_start_seconds))

        return chunk_start_seconds, chunk_end_seconds


    def get_ddm_scores(self, input_video_path: str, batch_size: int) -> tuple[list[float], VideoMetaData]:
        """
        Returns:
            (ddm scores, video_metadata)
        """

        _LOGGER.info("Running DDM inferencing for video: %s, batch_size: %d", input_video_path, batch_size)

        # Get video metadata using torchcodec.decoders.VideoDecoder
        decoder = VideoDecoder(input_video_path)
        metadata = decoder.metadata
        fps = metadata.average_fps
        duration_sec = metadata.duration_seconds
        total_frames = metadata.num_frames

        _LOGGER.info("Video metadata: duration=%.2fs, fps=%.2f, total_frames=%d",
                    duration_sec, fps, total_frames)

        target_frames_per_segment = self.frames_per_segment_hint

        # Calculate number of segments needed (minimum is number of GPUs)
        optimal_segments = max(self.num_workers, int(np.ceil(total_frames / target_frames_per_segment)))

        # Calculate minimal segment duration based on sliding window requirements
        # Need window_size + batch_size - 1 frames to create one full batch of windows
        # Minimum is window_size + 1 frames (for 2 windows, which is the minimum processable batch)
        window_size = self.frames_per_side * 2 + 1
        min_frames_for_processing = window_size + max(1, batch_size - 1)  # At least window_size + 1
        minimal_segment_duration = min_frames_for_processing / fps

        segment_duration = duration_sec / optimal_segments
        _LOGGER.debug("Initial segment duration: %.2fs, minimal segment duration: %.2fs", segment_duration, minimal_segment_duration)

        # Ensure optimal_segments doesn't go below 1 to prevent division by zero
        while segment_duration < minimal_segment_duration and optimal_segments > 1:
            optimal_segments -= 1
            segment_duration = duration_sec / optimal_segments
            _LOGGER.info("Adjusting number of segments to %d (%.2fs each)", optimal_segments, segment_duration)

        # Final check: if we still can't meet the minimum duration, warn but proceed
        if segment_duration < (window_size+1) / fps:
            _LOGGER.error("Cannot achieve minimal segment duration of %.2fs with single segment (%.2fs). "
                          "Processing is not allowed. Returning full video.", (window_size+1) / fps, segment_duration)
            return [0.0], [duration_sec]

        if segment_duration < minimal_segment_duration:
            _LOGGER.warning("Cannot achieve minimal segment duration of %.2fs with single segment (%.2fs). "
                            "Processing may be inefficient.", minimal_segment_duration, segment_duration)

        frames_per_segment = total_frames // optimal_segments
        _LOGGER.info("Splitting video into %d segments (~%d frames/%.2fs each) across %d GPUs",
                    optimal_segments, frames_per_segment, segment_duration, self.num_workers)

        # Calculate overlap in seconds based on frames_per_side
        # We need overlapping frames to handle boundaries at segment edges
        overlap_frames = self.frames_per_side
        overlap_sec = overlap_frames / fps

        # Create time segments with overlaps
        time_segments = []
        for i in range(optimal_segments):
            start_sec = i * segment_duration
            end_sec = (i + 1) * segment_duration

            # Add overlap for non-first segments (extend start backwards)
            if i > 0:
                start_sec = max(0, start_sec - overlap_sec)

            # Add overlap for non-last segments (extend end forwards)
            if i < optimal_segments - 1:
                end_sec = min(duration_sec, end_sec + overlap_sec)
            else:
                # For the last segment, ensure we process until the end
                end_sec = duration_sec

            time_segments.append((start_sec, end_sec))
            _LOGGER.debug("Segment %d: %.2fs to %.2fs (duration: %.2fs)",
                         i, start_sec, end_sec, end_sec - start_sec)

        # Submit inference requests to GPU workers in parallel
        futures = []
        request_ids = []

        request_id_prefix = uuid.uuid4().hex
        for i, (start_sec, end_sec) in enumerate(time_segments):
            request_id = f"{request_id_prefix}-{input_video_path}-segment{i}"
            request_ids.append(request_id)

            future = self._submit_inference(
                request_id=request_id,
                input_video_path=input_video_path,
                start_sec=start_sec,
                end_sec=end_sec,
                batch_size=batch_size
            )
            futures.append(future)

            _LOGGER.debug("Submitted request %s for segment %.2fs-%.2fs",
                         request_id, start_sec, end_sec)

        _LOGGER.info("Waiting for %d segments to be processed by %d GPU workers...",
                    len(time_segments), self.num_workers)

        segment_results = []
        for i, future in enumerate(futures):
            try:
                scores = future.result(timeout=self.time_out_sec)
                segment_results.append((time_segments[i], scores))
                _LOGGER.debug("Segment %d completed with %d scores", i, len(scores))
            except Exception as e:
                error_msg = f"Segment {i} failed to process: {e}"
                _LOGGER.error(error_msg)
                # FIXME: continue instead of raising error?
                raise RuntimeError(error_msg)

        # Merge boundary scores from all segments
        _LOGGER.debug("Merging boundary scores from %d segments", len(segment_results))
        merged_scores = self._merge_segment_scores(segment_results, fps, total_frames, overlap_frames)
        _LOGGER.debug("%d total scores for %d frames", len(merged_scores), total_frames)

        video_metadata = VideoMetaData(
            fps=fps, duration_sec=duration_sec, total_frames=total_frames)

        return merged_scores, video_metadata

    def shutdown(self, timeout: float = 30.0):
        """Shutdown the DDM-Net manager and cleanup resources"""
        _LOGGER.info("Shutting down DDM-Net manager...")

        # Stop response handler
        self.is_running = False

        # Send shutdown signals to all worker processes
        for worker in self.gpu_workers:
            try:
                worker.request_queue.put(_SHUTDOWN_SENTINEL, block=False)
                _LOGGER.info("Sent shutdown signal to GPU %s", worker.gpu_id)
            except Exception as e:
                _LOGGER.warning("Failed to send shutdown signal to GPU %s: %s", worker.gpu_id, e)

        _LOGGER.info("DDM-Net manager shutdown complete")

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

    def _init_single_gpu_workder(self, gpu_id: int) -> DdmNetGpuWorker | None:
        """Initialize a single GPU worker process"""
        try:
            _LOGGER.info("Creating DDM-Net GPU worker process for GPU %s", gpu_id)

            request_queue = mp.Queue(maxsize=self.max_queue_size)
            process = mp.Process(
                target=gpu_worker_process,
                args=(gpu_id,
                      self.checkpoint_path,
                      self.resolution,
                      self.frames_per_side,
                      request_queue,
                      self.response_queue),
            )
            process.daemon = True

            worker = DdmNetGpuWorker(
                gpu_id=gpu_id,
                process=process,
                request_queue=request_queue
            )

            process.start()
            _LOGGER.info("DDM-Net GPU worker process for GPU %s started", gpu_id)

            # Wait a bit to ensure process started successfully
            time.sleep(2)
            if not process.is_alive():
                _LOGGER.error("DDM-Net GPU worker process for GPU %s failed to start", gpu_id)
                return None

            return worker

        except Exception as e:
            _LOGGER.error("Failed to create DDM-Net GPU worker for GPU %s: %s", gpu_id, e)
            return None

    def _setup_gpu_workers(self, gpu_ids: list[int]) -> list[DdmNetGpuWorker]:

        _LOGGER.info("Creating %s GPU worker processes...", len(gpu_ids))

        workers = []
        for gpu_id in gpu_ids:
            worker = self._init_single_gpu_workder(gpu_id)
            if worker is not None:
                workers.append(worker)

        if not workers:
            raise RuntimeError("Failed to initialize any GPU worker processes")

        successful_gpus = [w.gpu_id for w in workers]
        _LOGGER.info("Successfully created %s GPU worker processes on GPUs: %s",
                    len(workers), successful_gpus)

        return workers

    def _start_response_handler(self):
        """Start the response handler thread"""
        self.is_running = True
        self.response_thread = threading.Thread(
            target=self._response_handler_loop,
            name="DDM-Net-Response-Handler",
            daemon=True
        )
        self.response_thread.start()

    def _response_handler_loop(self):
        """Handle responses from GPU worker processes"""
        _LOGGER.info("Response handler started")

        while self.is_running:
            try:
                try:
                    response = self.response_queue.get(timeout=10)
                except:
                    continue

                with self.pending_lock:
                    if response.request_id in self.pending_requests:
                        future = self.pending_requests.pop(response.request_id)

                        if response.error:
                            future.set_exception(Exception(response.error))
                        else:
                            future.set_result(response.scores)
                    else:
                        _LOGGER.warning("Received response for unknown request: %s", response.request_id)

            except Exception as e:
                error_msg = f"Error in response handler: {traceback.format_exc()}"
                _LOGGER.error(error_msg)

        _LOGGER.info("Response handler ended")

    def _get_next_worker_round_robin(self) -> DdmNetGpuWorker:
        """Get the next GPU worker in round-robin order"""
        with self.gpu_workers_lock:
            ret = next(self.gpu_workers_iterator_cycle)
            dead_gpus = set()
            while not ret.process.is_alive():
                dead_gpus.add(ret.gpu_id)
                _LOGGER.warning("GPU worker %s is not alive. Selecting next worker.", ret.gpu_id)
                ret = next(self.gpu_workers_iterator_cycle)
                if ret.gpu_id in dead_gpus:
                    _LOGGER.error("Running out of GPU workers. All GPU workers are dead."
                                  "You might want to check service logs to see what went wrong.")
                    return None

            return ret

    def _submit_inference(self, request_id: str, input_video_path: str, start_sec: float, end_sec: float, batch_size: int) -> Future:
        """Submit an inference request to the DDM-Net model"""

        if not self.is_running:
            raise RuntimeError("No GPU workers available. "
                               "This might be caused by failures to initialize GPU works. "
                               "You might want to check logs of the service.")

        worker = self._get_next_worker_round_robin()
        if worker is None:
            raise RuntimeError("No GPU workers available")

        future = Future()
        with self.pending_lock:
            if request_id in self.pending_requests:
                raise RuntimeError(f"Request {request_id} already in progress")
            self.pending_requests[request_id] = future

        request = DdmNetRequest(
            request_id=request_id,
            input_video_path=input_video_path,
            start_sec=start_sec,
            end_sec=end_sec,
            batch_size=batch_size
        )

        try:
            worker.request_queue.put(request)
            _LOGGER.debug("Submitted request %s to GPU %s", request_id, worker.gpu_id)

        except Exception as e:
            with self.pending_lock:
                self.pending_requests.pop(request_id, None)
            traceback_msg = traceback.format_exc()
            error_msg = f"Failed to submit request to GPU {worker.gpu_id}: {traceback_msg}"
            _LOGGER.error(error_msg)
            raise RuntimeError(error_msg)

        return future

    def _merge_segment_scores(self, segment_results: list[tuple[tuple[float, float], list[float]]],
                             fps: float, total_frames: int, overlap_frames: int) -> list[float]:
        """
        Merge boundary scores from multiple GPU segments, handling overlapping regions.

        Args:
            segment_results: List of (time_segment, scores) tuples
            fps: Video frame rate
            total_frames: Total number of frames in the video
            overlap_frames: Number of overlapping frames between segments

        Returns:
            List of merged boundary scores for the entire video
        """
        # Initialize the merged scores array
        merged_scores = [0.0] * total_frames
        overlap_counts = [0] * total_frames  # Track how many segments contribute to each frame

        for i, ((start_sec, end_sec), scores) in enumerate(segment_results):
            # Convert time to frame indices
            start_frame = int(start_sec * fps)
            segment_frame_count = len(scores)

            # Calculate the actual frame range this segment covers
            # Note: GPU worker already handles padding for frames_per_side at start/end
            for j, score in enumerate(scores):
                frame_idx = start_frame + j

                # Skip frames beyond video bounds
                if frame_idx >= total_frames:
                    break

                # Handle overlapping regions by averaging scores
                if overlap_counts[frame_idx] > 0:
                    if score == 0.0:
                        continue

                    new_score = max(merged_scores[frame_idx], score)
                    #_LOGGER.debug("Overlapping frame %d: score=%f, current_score=%f, current_overlap_count=%d -> new_score=%f",
                    #              frame_idx, score, merged_scores[frame_idx], overlap_counts[frame_idx], new_score)
                    merged_scores[frame_idx] = new_score
                else:
                    # First score for this frame
                    merged_scores[frame_idx] = score

                overlap_counts[frame_idx] += 1

            _LOGGER.debug("Merged segment %d: frames %d-%d",
                         i,
                         start_frame + overlap_frames,
                         start_frame + segment_frame_count - 1 - overlap_frames)

        return merged_scores

    def __del__(self):
        """Destructor - safety net for cleanup if shutdown() wasn't called"""
        if hasattr(self, 'is_running') and self.is_running:
            _LOGGER.warning("MultiGpuDdmNet being garbage collected without proper shutdown - cleaning up")
            try:
                self.shutdown(timeout=10)  # Shorter timeout for destructor
            except Exception as e:
                _LOGGER.error("Error during MultiGpuDdmNet cleanup in destructor: %s", e)


def detect_boundaries(scores: list[float], threshold: float, nms_size: int) -> list[int]:
    """
    Detect event boundaries based on scores and threshold.

    Args:
        scores: List of boundary scores for each frame
        threshold: Threshold for boundary detection
        nms_size: The peak in index i can filter out scores within [i-nms_size, i+nms_size]
                  even if the scores pass the threshold.

    Returns:
        List[int]: Frame indices where boundaries are detected
    """
    np_scores = np.array(scores)
    boundaries = []

    for i, score in enumerate(scores):
        if score > threshold:

            left_bdy = max(0, i - nms_size)
            right_bdy_plus_1 = min(len(scores), i + nms_size + 1)

            is_bdy = (np.argmax(np_scores[left_bdy:right_bdy_plus_1]) + left_bdy) == i

            if is_bdy:
                boundaries.append(i)

    return boundaries


def calculate_chunk_boundaries(boundaries: list[int],
                               fps: float,
                               duration_sec: float,
                               total_frames: int) -> tuple[list[float], list[float]]:
    """
    Calculate chunk start and end times based on detected boundaries index

    Args:
        boundaries: List of boundary frame indices from DDM-Net
        fps: Video frame rate
        duration_sec: Video duration in seconds
        total_frames: Total number of frames in video

    Returns:
        Tuple of (chunk_start_seconds, chunk_end_seconds) lists
    """
    _LOGGER.info(f"Calculating chunk boundaries")
    _LOGGER.info(f"Original boundaries: {len(boundaries)}")
    _LOGGER.info(f"Total frames: {total_frames}, FPS: {fps}, Duration: {duration_sec:.2f}s")

    boundaries_in_sec = [bdy / fps for bdy in boundaries]

    # Convert frame indices to time in seconds
    chunk_start_seconds = [0.0] + boundaries_in_sec
    chunk_end_seconds = boundaries_in_sec + [duration_sec]

    _LOGGER.info(f"Created {len(chunk_start_seconds)} time-based chunks")

    # Log chunk details
    for i, (start_sec, end_sec) in enumerate(zip(chunk_start_seconds, chunk_end_seconds)):
        duration = end_sec - start_sec
        _LOGGER.debug(f"Chunk {i}: {start_sec:.2f}s - {end_sec:.2f}s (duration: {duration:.2f}s)")

    return chunk_start_seconds, chunk_end_seconds


def _load_model(checkpoint_path: str, frames_per_side: int, gpu_id: int) -> torch.nn.Module:

    _LOGGER.info("Loading DDM-Net model...")
    _LOGGER.info("Checkpoint path: %s", checkpoint_path)
    _LOGGER.info("Frames per side: %d", frames_per_side)
    _LOGGER.info("GPU ID: %d", gpu_id)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found as '{checkpoint_path}'")

    # raise exception if DDM_BASE_PATH is not set
    ddm_base_path = os.environ["DDM_BASE_PATH"]
    modeling_package_parent_path = os.path.join(ddm_base_path, "DDM-Net")

    try:
        sys.path.insert(0, modeling_package_parent_path)
        from modeling.resnetGEBD import resnetGEBD
    finally:
        sys.path.pop(0)

    model = resnetGEBD(
        backbone="resnet50",
        pretrained=False,
        num_classes=2,
        frames_per_side=frames_per_side,
    )

    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")


    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        model_prefix = "model."
        module_prefix = "module."
        # clean key names
        for key, value in state_dict.items():
            if key.startswith(model_prefix):
                new_key = key[len(model_prefix):]
            elif key.startswith(module_prefix):
                new_key = key[len(module_prefix):]
            else:
                new_key = key
            new_state_dict[new_key] = value

    else:
        _LOGGER.info("state_dict doesn't exist. Attempting to load as PyTorch checkpoint...")
        new_state_dict = checkpoint

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        _LOGGER.warning("Missing keys in DDM checkppint: %s", missing_keys)
    if unexpected_keys:
        _LOGGER.warning("Unexpected keys in DDM checkpoint: %s", unexpected_keys)

    _LOGGER.info("DDM-Net model loaded.")

    device = f"cuda:{gpu_id}"
    _LOGGER.debug("Moving model to GPU %s", device)
    model = model.to(device)

    model.eval()

    return model
