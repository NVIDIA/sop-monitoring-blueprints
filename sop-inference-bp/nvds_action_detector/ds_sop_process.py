#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import asyncio
import atexit
import concurrent.futures
import concurrent.futures as futures
import copy
import json
import os
import signal
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from pyservicemaker import BufferRetriever, EOSMessage, PipelineState, StateTransitionMessage
from pyservicemaker.utils import MediaInfo, VideoStreamInfo

from . import ds_logger
from .ds_3d_action_pipeline import create_dummy_pipeline, create_inference_pipeline, encode_video, encode_video_gst
from .sop_step_checker import SopCheckerCache, SopCheckerRequest, SopCheckerResponse
from .utils import SafeThreadEventLoop, TimeMeasure, get_media_info_gst
from .vlm_inference_client import VLMInferenceClient

logger = ds_logger.get_logger(__name__)


BOUNDARY_DELAY_FRAME = int(os.getenv("BOUNDARY_DELAY_FRAME", "8"))
CV_THREAD_NUM = int(os.getenv("CV_THREAD_NUM", "32"))
CLIP_THREAD_NUM = int(os.getenv("CLIP_THREAD_NUM", "32"))
VLM_INFERENCE_THREAD_NUM = int(os.getenv("VLM_INFERENCE_THREAD_NUM", "64"))
VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "256"))

DISABLE_VLM_INFERENCE = os.getenv("DISABLE_VLM_INFERENCE", "false").lower() in ["true", "1", "yes", "y"]
if DISABLE_VLM_INFERENCE:
    logger.info("VLM inference is disabled")

VLLM_MODEL_PATH = os.getenv("VLLM_MODEL_PATH", "nvidia/Cosmos-Reason1-7B")
USE_VLLM_INFERENCE = os.getenv("USE_VLLM_INFERENCE", "1").lower() in ["true", "1", "yes", "y"]
if USE_VLLM_INFERENCE:
    logger.info("VLLM inference is enabled")
    logger.info(f"VLLM model path: {VLLM_MODEL_PATH}")
else:
    logger.info("VLM inference is enabled")
VLM_INFERENCE_ENDPOINT = os.getenv("VLM_INFERENCE_ENDPOINT", "http://localhost:8000")

ENABLE_ALERT_SOUND = os.getenv("ENABLE_ALERT_SOUND", "false").lower() in ["true", "1", "yes", "y"]
if ENABLE_ALERT_SOUND:
    logger.info("Alert sound is enabled")
else:
    logger.info("Alert sound is disabled")

ALERT_SOUND_FILE = os.getenv("ALERT_SOUND_FILE", "streams/alert.wav")
if not os.path.isfile(ALERT_SOUND_FILE):
    ENABLE_ALERT_SOUND = False
    logger.info(
        f"ALERT_SOUND_FILE {ALERT_SOUND_FILE} is not a file, alert sound is disabled, resetting ENABLE_ALERT_SOUND to False"
    )
else:
    logger.info(f"Alert sound file: {ALERT_SOUND_FILE}")

DISABLE_SOP_CHECKER = os.getenv("DISABLE_SOP_CHECKER", "false").lower() in ["true", "1", "yes", "y"]
if DISABLE_SOP_CHECKER:
    logger.info("SOP checker is disabled")
elif DISABLE_VLM_INFERENCE:
    DISABLE_SOP_CHECKER = True
    logger.info("SOP checker is disabled because VLM inference is disabled")
else:
    logger.info("SOP checker is enabled")

ENABLE_MESSAGING = os.getenv("ENABLE_MESSAGING", "false").lower() in ["true", "1", "yes", "y"]
if ENABLE_MESSAGING:
    logger.info("Messaging chunk_info is enabled")
else:
    logger.info("Messaging chunk_info is disabled")

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")

ENCODE_VIDEO = os.getenv("ENCODE_VIDEO", "false").lower() in ["true", "1", "yes", "y"]
ENCODE_VIDEO_OUTPUT_DIR = os.getenv("ENCODE_VIDEO_OUTPUT_DIR", "./chunks")
if ENCODE_VIDEO:
    logger.info("Video encoding is enabled")
    logger.info(f"Video encoding output directory: {ENCODE_VIDEO_OUTPUT_DIR}")
else:
    logger.info("Video encoding is disabled")

ACTION_CONFIG_PATH = os.getenv("ACTION_CONFIG_PATH", "configs/actions.json")

VLM_PROMPT_PATH = os.getenv("VLM_PROMPT_PATH", "configs/vlm_prompts.txt")

if not os.path.isfile(VLM_PROMPT_PATH):
    raise ValueError(f"VLM_PROMPT_PATH {VLM_PROMPT_PATH} is not a file")

with open(VLM_PROMPT_PATH, "r", encoding="utf-8") as fp:
    VLM_PROMPT = fp.read()

VLM_SYSTEM_PROMPT = "Answer the questions."


@dataclass
class ChunkParams:
    """Parameters for chunk boundary detection and segmentation."""

    threshold: float = 0.8
    """Confidence threshold for detecting action boundaries (0.0-1.0)"""

    min_length_sec: float = 1.0
    """Minimum chunk length in seconds (chunks shorter than this will be merged)"""

    max_length_sec: float = 10.0
    """Maximum chunk length in seconds (chunks longer than this will be split, None for no limit)"""

    duration_sec: Optional[float] = None
    """Total video duration in seconds"""

    fps: Optional[float] = None
    """Video frame rate"""


def chunk_info_func(chunk: Dict[str, Any]) -> str:
    res = f"chunk {chunk['chunk_idx']}: start={chunk['start_time']:.3f}s, end={chunk['end_time']:.3f}s"
    if "frame_number" in chunk:
        res += f", frame_number: {chunk['frame_number']}"
    if "vlm_execute_time" in chunk:
        res += f", vlm_execute_time: {chunk['vlm_execute_time']:.3f}s"
    if "checker_execute_time" in chunk:
        res += f", checker_execute_time: {chunk['checker_execute_time']:.3f}s"
    if "cv_execute_time" in chunk:
        res += f", cv_execute_time: {chunk['cv_execute_time']:.3f}s"

    pipeline_keys = [
        "pipeline_starting_timestamp",
        "pipeline_cv_ready_timestamp",
        "pipeline_vlm_starting_timestamp",
        "pipeline_vlm_ready_timestamp",
    ]
    for k in chunk.keys():
        if k in pipeline_keys:
            res += f", {k}: {chunk[k] - chunk['pipeline_starting_timestamp']:.3f}s"

    if "response" in chunk:
        res += f", response: {chunk['response']}"

    return res


def future_callback_profiling(tm: TimeMeasure):
    def callback(future):
        tm.update_execute_time()

    return callback


def run_in_executor_with_errors(loop, executor, func, *args, **kwargs):
    fut = loop.run_in_executor(executor, func, *args, **kwargs)

    def _cb(f):
        try:
            f.result()
        except Exception as e:
            logger.exception(f"Error in run_in_executor: {e}")

    fut.add_done_callback(_cb)
    return fut


def submit_in_executor(executor, func, *args, **kwargs):
    fut = executor.submit(func, *args, **kwargs)

    def _error_cb(f):
        try:
            f.result()
        except Exception as e:
            logger.exception(f"Error in submit_in_executor: {e}")

    fut.add_done_callback(_error_cb)
    return fut


class ModelInitalizer:
    def __init__(self):
        self._init_time = TimeMeasure()
        self._lock = threading.Lock()
        self._dummy_pipeline = create_dummy_pipeline()
        self._started_event_cv = threading.Event()
        self._warmup_ready_event_cv = threading.Event()

        self.vlm_thread = None
        self.vlm_ready_event = threading.Event()
        self._vlm_model = None
        if USE_VLLM_INFERENCE and not DISABLE_VLM_INFERENCE:
            self._initialize_vlm_model()
            pass

        def on_message_dummy_pipeline(message):
            if isinstance(message, StateTransitionMessage):
                if message.new_state == PipelineState.PLAYING:
                    with self._lock:
                        if not self._started_event_cv.is_set():
                            self._started_event_cv.set()
                            self._init_time.log_elapsed_time(f"Model initialized")
            if isinstance(message, EOSMessage):
                logger.info("End-of-Stream received! Pipeline has finished processing.")
                self._warmup_ready_event_cv.set()
                self._init_time.log_elapsed_time(f"Model warmed up")

        self._dummy_pipeline.start(on_message_dummy_pipeline)

    @property
    def vlm_model(self):
        return self._vlm_model

    def _initialize_vlm_model(self):
        from .vllm_inference import VLLMInference

        with TimeMeasure("VLLM model initialization") as tm:
            model_path = VLLM_MODEL_PATH
            self._vlm_model = VLLMInference(model_path=model_path, device="cuda:0")
            tm.log_elapsed_time(f"VLLM model: {model_path} initialized")

            def warmup_thread():
                with TimeMeasure("VLLM model warmup") as tm:
                    self.vlm_model.inference("Say Hi", [torch.zeros(224, 224, 3, dtype=torch.uint8)])
                    tm.log_elapsed_time(f"VLLM model: {model_path} warmed up")
                    self.vlm_ready_event.set()

            self.vlm_thread = threading.Thread(target=warmup_thread)
            self.vlm_thread.start()

    def is_model_ready(self):
        cv_ready = self._warmup_ready_event_cv.is_set()
        if not cv_ready:
            return False
        if self._vlm_model is not None:
            if not self.vlm_ready_event.is_set():
                return False
        return True

    def wait_for_model_ready(self):
        if not self._started_event_cv.is_set():
            self._started_event_cv.wait()

        if not self._warmup_ready_event_cv.is_set():
            self._warmup_ready_event_cv.wait()

        if self.vlm_model is not None:
            # self.vlm_thread.join()
            self.vlm_ready_event.wait()

    async def await_for_model_ready(self):
        while not self.is_model_ready():
            await asyncio.sleep(0.1)

    def close(self):
        if self._dummy_pipeline is not None:
            self._dummy_pipeline.stop()
            self._dummy_pipeline.wait()
        if self._vlm_model is not None:
            self._vlm_model.stop()


# atexit.register(initlize_model.close)


class SOPVideoProcessor:
    class DecodedFrameRetriever(BufferRetriever):
        """Retrieve frames from the buffer and put them into the queue"""

        def __init__(self, sop_video_processor: "SOPVideoProcessor"):
            super().__init__()
            self._sop_video_processor = sop_video_processor
            # Calculate max queue size based on video parameters
            fps = sop_video_processor._chunk_params.fps if sop_video_processor._chunk_params.fps < 1 else 30.0
            max_length_sec = max(sop_video_processor._chunk_params.max_length_sec, 1)
            max_queue_size = max(int(fps * max_length_sec + 10), 30)
            self.decoded_frame_queue = Queue(maxsize=max_queue_size)
            assert self.decoded_frame_queue is not None, "decoded frame queue is not set"
            logger.info(f"DecodedFrameRetriever initialized with max_queue_size: {max_queue_size}")
            self._count = 0
            self._lock = threading.Lock()
            self.new_frame_event = threading.Event()
            self._last_timestamp = 0
            self.end_of_stream = False
            self._start_sec = 0.0
            self._duration_sec = sop_video_processor._chunk_params.duration_sec

        def last_timestamp(self):
            with self._lock:
                return self._last_timestamp

        def end_sec(self):
            return self._start_sec + self._duration_sec

        def is_end_of_stream(self):
            with self._lock:
                return self.end_of_stream

        def wait_for_new_frame(self):
            self.new_frame_event.wait()
            self.new_frame_event.clear()

        def set_end_of_stream(self):
            self.decoded_frame_queue.put(None)
            with self._lock:
                self.end_of_stream = True
            self.new_frame_event.set()
            logger.info(f"DecodedFrameRetriever set end of stream, queue size: {self.decoded_frame_queue.qsize()}")

        def consume(self, buffer):
            logger.debug(f"receiving frame: {self._count}, buffer.timestamp: {buffer.timestamp}")
            try:
                tensor = buffer.extract(0).clone()
                torch_tensor = torch.utils.dlpack.from_dlpack(tensor)
                timestamp = buffer.timestamp / 1e9  # convert nanoseconds to seconds
                # if self._count == 0:
                #     self._sop_video_processor.first_timestamp = time.time()
                logger.debug(
                    f"received frame: {self._count}, timestamp: {timestamp}, tensor.shape: {torch_tensor.shape}"
                )
                try:
                    # video file will be blocked call,
                    # live stream will drop frames if inference speed is getting slower than streaming
                    block = not self._sop_video_processor._is_live
                    self.decoded_frame_queue.put((timestamp, torch_tensor), block=block)
                except Full:
                    # for live stream dropping only
                    try:
                        dropped_frame = self.decoded_frame_queue.get(block=False)
                        if dropped_frame is not None:
                            dropped_timestamp, _ = dropped_frame
                            logger.warning(
                                f"DecodedFrameRetriever queue is full (size: {self.decoded_frame_queue.qsize()}), "
                                f"dropping oldest frame with timestamp: {dropped_timestamp:.3f}s to make room for new frame at {timestamp:.3f}s"
                            )
                    except Empty:
                        pass
                    try:
                        self.decoded_frame_queue.put((timestamp, torch_tensor), block=False)
                    except Full:
                        # This should rarely happen, but handle it gracefully
                        logger.error(
                            f"DecodedFrameRetriever queue still full after dropping oldest frame, skipping frame at {timestamp:.3f}s"
                        )
                        return 0

                with self._lock:
                    self._last_timestamp = timestamp
                self.new_frame_event.set()
            except Exception as e:
                logger.exception(f"Error in DecodedFrameRetriever consume: {e}")
                return 0
            if self.end_sec() < timestamp:
                logger.info(
                    f"DecodedFrameRetriever force end of stream, timestamp: {timestamp}, end_sec: {self.end_sec()}"
                )
                self._sop_video_processor._boundary_queue.put(None)
                self.set_end_of_stream()
                return 0
            logger.debug(f"DecodedFrameRetriever continue, timestamp: {timestamp}, end_sec: {self.end_sec()}")
            self._count += 1
            return 1

    def __init__(
        self,
        file_path,
        id=None,
        manager=None,
        device="cuda:0",
        chunk_params: Optional[ChunkParams] = None,
        cv_process_pool=None,
        clip_process_pool=None,
        vlm_inference_pool=None,
        vlm_request_pool=None,
        vlm_model=None,
        messager=None,
        **kwargs,
    ):
        super().__init__()
        self._tm_e2e = TimeMeasure("SOPVideoProcessor")
        self._file_path = file_path
        self._manager = manager
        self._camera_serial_number = kwargs.get("camera_serial_number", None)
        self._camera_config = kwargs.get("camera_config", None)
        self._prompt = kwargs.get("prompt", "")
        if self._camera_serial_number is not None:
            self._is_camera = True
            self._is_live = True
        elif self._file_path and self._file_path.startswith("rtsp://"):
            self._is_live = True
            self._is_camera = False
        else:
            self._is_camera = False
            self._is_live = False
        self._cv_process_pool = (
            cv_process_pool if cv_process_pool is not None else futures.ThreadPoolExecutor(max_workers=8)
        )
        self._clip_process_pool = (
            clip_process_pool if clip_process_pool is not None else futures.ThreadPoolExecutor(max_workers=8)
        )
        self._vlm_inference_pool = (
            vlm_inference_pool if vlm_inference_pool is not None else futures.ThreadPoolExecutor(max_workers=8)
        )
        self._vlm_request_pool = (
            vlm_request_pool if vlm_request_pool is not None else futures.ThreadPoolExecutor(max_workers=8)
        )
        self._device = device
        gpu_id_str = device.split(":")[-1] if ":" in device else "0"
        self._gpu_id = int(gpu_id_str) if gpu_id_str else 0
        self._started_event = Event()
        self._boundary_queue = Queue()
        self._chunk_queue = Queue()
        self._vlm_response_future_queue = Queue()
        self._vlm_response_queue = Queue()
        self._chunk_params = chunk_params if chunk_params is not None else ChunkParams()
        self.original_width, self.original_height = 0, 0
        if self._file_path and not self._camera_serial_number:
            duration_sec, fps, self.original_width, self.original_height = self.get_media_info(self._file_path)
        else:
            duration_sec, fps = float("inf"), 30.0
        if self._chunk_params.duration_sec is None:
            self._chunk_params.duration_sec = duration_sec
        self._chunk_params.fps = fps
        self._id = id if id is not None else uuid.uuid4()
        self._pipeline_thread_future = None
        self._clip_process_future = None
        self._vlm_inference_future = None
        self._boundary_pipeline_future = None
        self._vlm_response_thread: Optional[threading.Thread] = None
        self._vlm_response_thread_future = None
        self._clip_start_sec = 0
        self._clip_cur_sec = 0
        self._vlm_inference = None
        self._decoded_frame_retriever = None
        self._sop_checker_future = None
        self._post_dispatch_thread_pool = futures.ThreadPoolExecutor(max_workers=1)
        self._post_dispatch_future = None
        self._final_queue = Queue()
        self._messager = messager
        self._message_pool = kwargs.get("message_pool", None)
        self.first_timestamp = 0.0

        if not self._is_camera:
            kwargs.update(
                {
                    "mux_width": self.original_width,
                    "mux_height": self.original_height,
                }
            )

        if not DISABLE_VLM_INFERENCE:
            if USE_VLLM_INFERENCE:
                self._vlm_inference = vlm_model
                self._decoded_frame_retriever = self.DecodedFrameRetriever(self)
            else:
                self._vlm_inference = VLMInferenceClient(VLM_INFERENCE_ENDPOINT)
            self._initialize_checker()

        # Store only serializable parameters from kwargs to avoid pickle errors
        serializable_kwargs = {}
        skip_keys = {
            "cv_process_pool",
            "clip_process_pool",
            "vlm_model",
            "vlm_inference_pool",
            "vlm_request_pool",
            "message_pool",
            "messager",
            "frame_retriever",
        }
        for key, value in kwargs.items():
            if key in skip_keys:
                continue
            try:
                # Try to deep copy the value to ensure it's serializable
                serializable_kwargs[key] = copy.deepcopy(value)
            except (TypeError, AttributeError):
                # Skip values that cannot be deep copied
                logger.debug(f"Skipping non-serializable kwarg: {key}")
                pass
        self._original_kwargs = serializable_kwargs
        # import pdb; pdb.set_trace()
        logger.info(
            f"Initializing SOP video Processor for stream: {self._file_path},"
            f"id: {self._id},"
            f"video_width: {self.original_width}, video_height: {self.original_height}, duration_sec: {self._chunk_params.duration_sec}, fps: {self._chunk_params.fps},"
            f"chunk_params: {self._chunk_params},"
            f"gpu_id: {self._gpu_id},"
            f"kwargs: {kwargs}",
        )
        with TimeMeasure("pipeline") as tm:
            try:
                update_args = {}
                self._inference_pipeline = create_inference_pipeline(
                    file_path,
                    self._boundary_queue,
                    self._gpu_id,
                    frame_retriever=self._decoded_frame_retriever,
                    update_args=update_args,
                    **kwargs,
                )
                tm.log_elapsed_time("inference pipeline creation")
                if update_args:
                    logger.info(f"######## update_args: {update_args}")
                    self._original_kwargs.update(update_args)
                    if "original_width" in update_args:
                        self.original_width = update_args["original_width"]
                    if "original_height" in update_args:
                        self.original_height = update_args["original_height"]
                    if "original_fps" in update_args:
                        self._chunk_params.fps = update_args["original_fps"]
            except Exception as e:
                logger.exception(f"Error in create_inference_pipeline: {e}")
                self._inference_pipeline = None
                raise e
        if self._inference_pipeline is None:
            logger.error(f"Failed to create inference pipeline, stopping video processor {self._id}")
            return

    @property
    def id(self):
        return self._id

    @property
    def final_queue(self):
        return self._final_queue

    @property
    def inference_last_queue(self):
        if DISABLE_VLM_INFERENCE:
            return self._chunk_queue
        elif DISABLE_SOP_CHECKER:
            return self._vlm_response_queue
        else:
            return self._sop_checker_result_queue

    @classmethod
    def get_media_info(cls, file_path):
        DISABLE_SERVICE_MAKER_INFO = True
        if DISABLE_SERVICE_MAKER_INFO:
            from pymediainfo import MediaInfo

            if file_path.startswith("rtsp://"):
                video_duration_sec, video_fps, video_width, video_height = get_media_info_gst(file_path)
                return video_duration_sec, video_fps, video_width, video_height

            if file_path.startswith("file://"):
                file = file_path[7:]
            else:
                file = file_path
            media_info = MediaInfo.parse(file)
            have_image_or_video = False
            for track in media_info.tracks:
                if track.track_type == "Video":
                    video_codec = track.format
                    video_duration_sec = float(track.duration) / 1e3
                    video_fps = track.frame_rate
                    video_width, video_height = track.width, track.height
                    have_image_or_video = True
                if track.track_type == "Image":
                    video_codec = track.format
                    video_duration_sec = 0
                    video_fps = 0
                    video_width, video_height = track.width, track.height
                    have_image_or_video = True

            if not have_image_or_video:
                raise Exception("MediaInfo.parse: Unsupported file type - " + file)
            return video_duration_sec, float(video_fps), video_width, video_height

        else:
            mediainfo = MediaInfo.discover(file_path)
            assert mediainfo.duration != 0
            v_streams = [s for s in mediainfo.streams if isinstance(s, VideoStreamInfo)]
            assert len(v_streams) == 1
            fps = float(v_streams[0].framerate[0]) / v_streams[0].framerate[1]
            return mediainfo.duration / 1e9, fps, v_streams[0].width, v_streams[0].height

    async def start(self):
        # import pdb; pdb.set_trace()
        logger.info(f"SOPVideoProcessor {self._id} starting")
        loop = asyncio.get_running_loop()
        self._pipeline_thread_future = submit_in_executor(self._cv_process_pool, self.run_pipeline)
        self._clip_process_future = submit_in_executor(self._clip_process_pool, self.clip_post_process)
        self._post_dispatch_future = submit_in_executor(self._post_dispatch_thread_pool, self.post_dispatch_process)
        if not DISABLE_VLM_INFERENCE:
            await self._start_vlm(loop)
        if not DISABLE_SOP_CHECKER:
            await self._start_sop_checker(loop)
        logger.info(f"SOPVideoProcessor {self._id} started (running in background)")

    async def _start_vlm(self, loop: asyncio.AbstractEventLoop):
        self._vlm_inference_future = submit_in_executor(self._vlm_inference_pool, self.vlm_inference_request_process)
        self._vlm_response_thread = threading.Thread(target=self.vlm_inference_response_process)
        self._vlm_response_thread.start()

    def _initialize_checker(self):
        with open(ACTION_CONFIG_PATH, "r") as f:
            action_config = json.load(f)
        self._action_config = action_config
        self._sop_checker = SopCheckerCache()
        self._sop_checker_pool = futures.ThreadPoolExecutor(max_workers=1)
        self._sop_checker_result_queue = Queue()
        self._sop_checker_future = None

    async def _start_sop_checker(self, loop: asyncio.AbstractEventLoop):
        self._sop_checker_future = submit_in_executor(self._sop_checker_pool, self.sop_checker_process)

    def sop_checker_process(self):
        checker_id = "*"
        cycle_completion_threshold = 0.6
        cycle_boundary_threshold_low = 0.3
        cycle_boundary_threshold_high = 0.8

        def process_sop_check_func(request_id, vlm_output, checker_id, keep_alive=True):
            nonlocal cycle_completion_threshold
            nonlocal cycle_boundary_threshold_low
            nonlocal cycle_boundary_threshold_high
            sop_checker_request = SopCheckerRequest(
                action_json=json.dumps(self._action_config),
                vlm_output=vlm_output,
                keep_alive=keep_alive,
                checker_id=checker_id,
                cycle_completion_threshold=cycle_completion_threshold,
                cycle_boundary_threshold_low=cycle_boundary_threshold_low,
                cycle_boundary_threshold_high=cycle_boundary_threshold_high,
            )
            tm = TimeMeasure("SOP checker")
            try:
                checker_result = self._sop_checker.process_sop_check(request_id, sop_checker_request)
            except Exception as e:
                error_message = traceback.format_exc()
                logger.exception(f"Error in process_sop_check: {e}")
                checker_result = SopCheckerResponse(
                    request_id=request_id,
                    checker_id="",
                    cycle=0,
                    missing_detected=[],
                    misordered_detected=[],
                    final_missing_detected=[],
                    final_misordered_detected=[],
                    cycle_completed=False,
                    summary_cycles_detected=[],
                    summary_cycle_analysis=[],
                    error_message=error_message,
                )
            finally:
                return checker_result.asdict(), tm.elapsed_time

        while True:
            chunk = self._vlm_response_queue.get(block=True)
            if chunk is None:
                if checker_id != "*":
                    request_id = str(uuid.uuid4())
                    checker_result, execute_time = process_sop_check_func(request_id, "", checker_id, keep_alive=False)
                    checker_id = checker_result.get("checker_id", "*")
                    logger.info(f"SOP checker, final result: {checker_result}")
                    final_chunk = {
                        "checker_result": checker_result,
                        "req_id": request_id,
                        "chunk_idx": -1,
                        "start_time": 0,
                        "end_time": 0,
                        "checker_execute_time": execute_time,
                    }
                    self._sop_checker_result_queue.put(final_chunk)
                break
            vlm_response = chunk.get("response", "")  # None is not valid for checker process
            request_id = chunk.get("req_id", None)
            chunk_idx = chunk.get("chunk_idx", 0)
            cycle_completion_threshold = chunk.get("cycle_completion_threshold", 0.6)
            cycle_boundary_threshold_low = chunk.get("cycle_boundary_threshold_low", 0.3)
            cycle_boundary_threshold_high = chunk.get("cycle_boundary_threshold_high", 0.8)

            checker_result, execute_time = process_sop_check_func(request_id, vlm_response, checker_id, keep_alive=True)
            checker_id = checker_result.get("checker_id", "*")
            chunk["checker_result"] = checker_result
            chunk["checker_execute_time"] = execute_time
            logger.info(f"SOP checker, chunk_idx:{chunk_idx} result: {checker_result}")
            self._sop_checker_result_queue.put(chunk)

        # end of stream
        self._sop_checker_result_queue.put(None)
        logger.info(f"SOPVIdeoProcessor: {self.id} sop_checker_process done")

    async def stop(self, force=False):
        loop = asyncio.get_running_loop()
        try:
            if self._inference_pipeline is not None:
                logger.info(f"SOPVideoProcessor {self._id} stopping inference pipeline")
                await loop.run_in_executor(None, self._inference_pipeline.stop)
                self._inference_pipeline = None
            if self._pipeline_thread_future is not None:
                logger.info(f"SOPVideoProcessor {self._id} waiting for pipeline thread ended")
                await asyncio.wrap_future(self._pipeline_thread_future, loop=loop)
                self._pipeline_thread_future = None
            if self._clip_process_future is not None:
                # Wrap the future so it can be awaited from a different loop
                await asyncio.wrap_future(self._clip_process_future, loop=loop)
            if self._vlm_inference_future is not None:
                await asyncio.wrap_future(self._vlm_inference_future, loop=loop)
            if self._vlm_response_thread is not None:
                await loop.run_in_executor(None, self._vlm_response_thread.join)

            # stop sop checker
            if self._sop_checker_future is not None:
                await asyncio.wrap_future(self._sop_checker_future, loop=loop)
            # if not force and self._inference_pipeline is not None:
            #     await loop.run_in_executor(None, self._inference_pipeline.stop)
            if self._manager is not None:
                self._manager.remove_video_processor(self._id)
        except BaseException as e:
            logger.exception(f"Error in SOPVideoProcessor {self._id} stop: {e}")
            # Optionally re-raise if needed
            # raise
        finally:
            logger.info(f"SOPVideoProcessor {self._id} stopped (force: {force})")

    def alert_sound(self, chunk):
        logger.info(f"Alert sound for chunk: {chunk}")
        from playsound import playsound

        playsound(ALERT_SOUND_FILE)

    def messaging_chunk(self, chunk):
        if self._message_pool is not None and self._messager is not None:
            logger.debug(f"Messaging chunk: {chunk.get('chunk_idx', None)}")
            if self._camera_serial_number:
                chunk["sensor_id"] = self._camera_serial_number
            else:
                chunk["sensor_id"] = self._file_path

            self._message_pool.submit(self._messager.produce, chunk)

    def post_dispatch_process(self):
        while True:
            logger.debug(f"========Post dispatch process started")
            inference_last_queue = self.inference_last_queue
            chunk = inference_last_queue.get(block=True)
            self._final_queue.put(chunk)
            if chunk is None:
                logger.debug(f"========Post dispatch process end of stream")
                break
            if ENABLE_ALERT_SOUND:
                self.alert_sound(chunk)
                logger.debug(f"Alert sound done")
            if ENABLE_MESSAGING:
                logger.debug(f"Alert messaging")
                self.messaging_chunk(chunk)

        self._final_queue.put(None)
        logger.info(f"SOPVIdeoProcessor: {self.id} post_dispatch_process done")

    def run_pipeline(self):
        tm = TimeMeasure()
        is_pipeline_ready = False
        is_pipeline_playing = False

        def on_message(message):
            nonlocal is_pipeline_ready
            nonlocal is_pipeline_playing
            if isinstance(message, StateTransitionMessage):
                logger.debug(f"StateTransitionMessage received: {message}")
                if message.new_state == PipelineState.PLAYING and not is_pipeline_playing:
                    is_pipeline_playing = True
                    self._started_event.set()
                    tm.log_elapsed_time(f"inference pipeline: {self.id} has started playing")
                if message.new_state == PipelineState.READY and not is_pipeline_ready:
                    is_pipeline_ready = True
                    tm.log_elapsed_time(f"inference pipeline: {self.id} is ready to start")
                elif message.new_state == PipelineState.INVALID:
                    self._started_event.set()
                    self._tm_e2e.log_elapsed_time(f"inference pipeline: {self.id} has entered INVALID state")

            if isinstance(message, EOSMessage):
                logger.info("End-of-Stream received! Pipeline has finished processing.")
                self._boundary_queue.put(None)
                if self._decoded_frame_retriever is not None:
                    self._decoded_frame_retriever.set_end_of_stream()
                tm.log_elapsed_time(
                    f"inference pipeline has finished w/ EOS queue size: {self._boundary_queue.qsize()}"
                )

        self._inference_pipeline.start(on_message)
        tm.log_elapsed_time("inference pipeline is starting in async mode")
        self._inference_pipeline.wait()
        self._started_event.set()
        self._boundary_queue.put(None)
        if self._decoded_frame_retriever is not None:
            self._decoded_frame_retriever.set_end_of_stream()
        tm.log_elapsed_time(f"SOPVIdeoProcessor: {self.id} DeepStream inference pipeline stopped.")

    def clip_post_process(self):
        self._started_event.wait()
        # items = []
        # import pdb; pdb.set_trace()
        tm = TimeMeasure("clip_post_process")
        logger.info("clip_post_process started")
        assert self._boundary_queue is not None
        is_last_item = False
        delayed_frames = 0
        # list of (frame_id, pts, score)
        boundaries = []
        need_check_delayed = False
        is_ready = False
        chunk_idx = 0
        while not is_last_item:
            item = self._boundary_queue.get(block=True)
            if item is None:
                logger.info("last item is None received")
                is_last_item = True
            else:
                # items.append(item)
                frame_id, pts, score = item
                self._clip_cur_sec = pts
                if score < 0:  # this is the first ready frame, need to init some base timestamp offsets
                    # the self.first_timestamp and start_time is a match for live stream processing
                    self.first_timestamp = self._tm_e2e.now()
                    self._clip_start_sec = pts
                    logger.info(f"SOP Processor: {self.id} first metadata found: {frame_id}, pts: {pts:.3f}s")
                    continue
                if score >= self._chunk_params.threshold:
                    boundaries.append(item)
                    need_check_delayed = True
                    logger.debug(f"New boundary found: {item}")
                    # scores.append(score)
                    # timestamps.append(pts)
                elif pts - self._clip_start_sec >= self._chunk_params.max_length_sec:
                    boundaries.append(item)
                    is_ready = True
                    logger.info(
                        f"boundary found: {item} max length reached, pts: {pts:.3f}s, max length: {self._chunk_params.max_length_sec}s"
                    )

                if need_check_delayed:
                    delayed_frames += 1
                    if delayed_frames >= BOUNDARY_DELAY_FRAME:
                        is_ready = True
            if is_ready or is_last_item:
                logger.debug(f"is_ready or is_last_item: {is_ready} {is_last_item}, item: {item}")
                end, score = self.calculate_next_chunk_boundary(boundaries, is_end=is_last_item)
                logger.debug(f"calculate_next_chunk_boundary: {end}")
                if end is not None:
                    tm.log_elapsed_time(f"calculated next chunk clip {self._clip_start_sec:.3f} - {end:.3f} video ")
                    chunk_info = {
                        "chunk_idx": chunk_idx,
                        "start_time": self._clip_start_sec,
                        "end_time": end,
                        "cv_boundary_score": score,  # this is the score of end_time boundary score
                        "cv_execute_time": tm.elapsed_time,
                        "first_timestamp": self.first_timestamp,
                        "pipeline_starting_timestamp": self._tm_e2e.start_time,
                        "pipeline_cv_ready_timestamp": self._tm_e2e.now(),
                    }
                    chunk_idx += 1
                    self._chunk_queue.put(chunk_info)
                    self._clip_start_sec = end
                    tm.reset()
                boundaries.clear()
                delayed_frames = 0
                need_check_delayed = False
                is_ready = False

        # this is the end of the clips post processing
        self._chunk_queue.put(None)
        logger.info(f"SOPVIdeoProcessor: {self.id} clip_post_process done")

    def calculate_next_chunk_boundary(
        self, boundaries: list[Tuple[int, float, float]], is_end: bool
    ) -> Optional[float]:
        """
        Calculate chunk start and end times based on detected boundaries with length constraints.

        Args:
            boundaries: List of boundary (frame_id, pts, score)
            params: ChunkParams object containing all configuration parameters

        Returns:
            Tuple of (chunk_start_seconds, chunk_end_seconds) lists
        """
        # Validate required parameters
        if self._chunk_params.duration_sec is None:
            raise ValueError("ChunkParams must have fps, total_frames, and duration_sec set")

        if is_end:
            if self._is_live:
                return self._clip_cur_sec, 0
            return self._chunk_params.duration_sec, 0

        # find largest score
        most_likely_boundary = max(boundaries, key=lambda x: x[2])
        idx, pts, score = most_likely_boundary
        if score < self._chunk_params.threshold:
            most_likely_boundary = boundaries[-1]
            idx, pts, score = most_likely_boundary

        # return None if the chunk is too short
        if pts - self._clip_start_sec <= self._chunk_params.min_length_sec:
            return None, None

        return pts, score

    def vlm_inference_request_process(self):
        while True:
            chunk = self._chunk_queue.get(block=True)
            if chunk is None:
                break
            chunk_info = chunk
            start_time, end_time = chunk_info["start_time"], chunk_info["end_time"]
            logger.info(f"VLM start inference on chunk {start_time:.3f} - {end_time:.3f} video")
            chunk_info["pipeline_vlm_starting_timestamp"] = self._tm_e2e.now()
            vlm_tm = TimeMeasure("VLM inference")
            prompt = self._prompt if self._prompt and len(self._prompt) > 0 else VLM_PROMPT
            system_prompt = VLM_SYSTEM_PROMPT
            if USE_VLLM_INFERENCE:
                response_future = self.submit_vllm_inference(prompt, system_prompt, start_time, end_time, chunk_info)
            else:
                response_future = self._vlm_request_pool.submit(
                    self._vlm_inference.inference,
                    prompt=prompt,
                    video_path=self._file_path,
                    start_time=start_time,
                    end_time=end_time,
                    max_tokens=VLM_MAX_TOKENS,
                    num_frames_per_chunk=int((end_time - start_time) * self._chunk_params.fps),
                    # system_prompt=VLM_SYSTEM_PROMPT,
                )

            if response_future is not None:
                response_future.add_done_callback(future_callback_profiling(vlm_tm))

            chunk_info["response_future"] = response_future
            chunk_info["file_path"] = self._file_path
            chunk_info["vlm_time_measure"] = vlm_tm
            self._vlm_response_future_queue.put(chunk_info)
            logger.debug(f"VLM inference request submitted for chunk {chunk_info_func(chunk_info)}")

        self._vlm_response_future_queue.put(None)
        logger.info(f"SOPVIdeoProcessor: {self.id} vlm_inference_request_process done")

    def submit_vllm_inference(self, prompt, system_prompt, start_time, end_time, chunk_info):
        decoded_frame_queue = self._decoded_frame_retriever.decoded_frame_queue
        assert decoded_frame_queue is not None, "decoded frame queue is not set"

        while (
            not self._decoded_frame_retriever.is_end_of_stream()
            and self._decoded_frame_retriever.last_timestamp() < end_time
        ):
            self._decoded_frame_retriever.wait_for_new_frame()

        frames = []
        encode_frames = Queue()
        while not decoded_frame_queue.empty():
            frame = decoded_frame_queue.get(block=True)
            if frame is None:
                break
            timestamp, tensor = frame
            logger.debug(
                f"before submit_vllm_inference: get decoded frame, timestamp: {timestamp}, tensor.shape: {tensor.shape}"
            )
            frames.append(tensor)
            encode_frames.put((timestamp, tensor))
            if timestamp + 1e-3 >= end_time:
                break
        encode_frames.put(None)
        chunk_info["frame_number"] = len(frames)

        if len(frames) == 0:
            logger.warning(f"submit_vllm_inference: no frames decoded, start: {start_time}, end: {end_time}")
            return None

        logger.info(
            f"submit_vllm_inference: VLM chunk start: {start_time}, end: {end_time} with decoded frame count: {len(frames)}"
        )
        logger.debug(f"prompt: {prompt}")
        chunk_idx = chunk_info.get("chunk_idx", 0)
        req_id = chunk_info.get("req_id", str(self.id if len(str(self.id)) > 16 else uuid.uuid4()))
        vlm_req_id = f"{chunk_idx:04d}-{req_id}"
        kwargs = {
            "req_id": vlm_req_id,
            "video_fps": self._chunk_params.fps,
            "temperature": self._original_kwargs.get("temperature", None),
            "max_completion_tokens": self._original_kwargs.get("max_completion_tokens", 256),  # keep short
            "seed": self._original_kwargs.get("seed", None),
            "top_p": self._original_kwargs.get("top_p", None),
        }
        response_future = self._vlm_request_pool.submit(
            self._vlm_inference.inference,
            prompt=prompt,
            video=frames,
            system_prompt=system_prompt,
            **kwargs,
        )

        if ENCODE_VIDEO:
            if self._camera_serial_number:
                video_file = (
                    f"{self._camera_serial_number}-{vlm_req_id}-chunk_{chunk_idx}-{start_time:.3f}-{end_time:.3f}.mp4"
                )
            elif self._is_live:
                video_file = os.path.basename(self._file_path)
                video_file += f"-{vlm_req_id}-chunk_{chunk_idx}-{start_time:.3f}-{end_time:.3f}.mp4"
            else:
                video_file = os.path.basename(self._file_path)
                # remove the last suffix of video_file (e.g. _00000.mp4)
                video_file = video_file.rsplit(".", 1)[0] + f"-chunk_{chunk_idx}-{start_time:.3f}-{end_time:.3f}.mp4"
            output_path = os.path.join(ENCODE_VIDEO_OUTPUT_DIR, video_file)
            logger.info(
                f"encoding video{self.original_width}x{self.original_height}@{self._chunk_params.fps} chunk to {output_path}"
            )
            try:
                encode_video_gst(
                    encode_frames, self.original_width, self.original_height, self._chunk_params.fps, output_path
                )
                if self._is_live:
                    chunk_info["file_path"] = video_file
            except Exception as e:
                logger.exception(f"encode video: {video_file} failed: {e}")
                response_future = None
        # logger.info(f"submit_vllm_inference: VLM inference result: {response_future.result()}")
        return response_future

    def vlm_inference_response_process(self):
        while True:
            chunk_info = self._vlm_response_future_queue.get(block=True)
            if chunk_info is None:
                break
            response_future = chunk_info.pop("response_future", None)
            response = {}
            try:
                if response_future is not None:
                    response = response_future.result()
                chunk_info.update(response)
                logger.debug(f"VLM inference result: {response}")
            except Exception as e:
                logger.exception(f"VLM inference failed: {e}")
                response = None
            tm = chunk_info.pop("vlm_time_measure", None)
            vlm_execute_time = None
            if tm is not None:
                tm.update_execute_time()
                vlm_execute_time = tm.first_execute_time
            chunk_info["vlm_execute_time"] = vlm_execute_time
            chunk_info["pipeline_vlm_ready_timestamp"] = self._tm_e2e.now()
            logger.info(f"VLM inference on chunk {chunk_info_func(chunk_info)} took {vlm_execute_time:.3f} seconds")
            self._vlm_response_queue.put(chunk_info)

        self._vlm_response_queue.put(None)
        logger.info(f"SOPVIdeoProcessor: {self.id} vlm_inference_response_process done")


class SOPProcessManager:
    def __init__(self):
        self._initlize_model = ModelInitalizer()
        self._cv_process_pool = futures.ThreadPoolExecutor(max_workers=CV_THREAD_NUM)
        self._clip_process_pool = futures.ThreadPoolExecutor(max_workers=CLIP_THREAD_NUM)
        self._vlm_inference_pool = futures.ThreadPoolExecutor(max_workers=VLM_INFERENCE_THREAD_NUM)
        self._vlm_request_pool = futures.ThreadPoolExecutor(max_workers=VLM_INFERENCE_THREAD_NUM)
        self.processor_list = {}
        self.processor_list_lock = threading.Lock()
        self._safe_thread_event_loop = SafeThreadEventLoop()

        self._messager = None
        self._message_pool = None
        if ENABLE_MESSAGING:
            from .messager import create_producer

            self._messager = create_producer(KAFKA_BROKER)
            self._message_pool = futures.ThreadPoolExecutor(max_workers=16)

    def trigger_stop_processors(self, processor: SOPVideoProcessor, force=False):
        return self._safe_thread_event_loop.run_coroutine_threadsafe(processor.stop(force=force))

    def wait_for_model_ready(self):
        self._initlize_model.wait_for_model_ready()

    async def await_for_model_ready(self):
        await self._initlize_model.await_for_model_ready()

    def is_model_ready(self):
        return self._initlize_model.is_model_ready()

    def close(self, force=False):
        # Get list of processors to stop and clear the dict to prevent deadlock
        # (stop() calls remove_video_processor which acquires the same lock)
        processors = []
        with self.processor_list_lock:
            processors = list(self.processor_list.values())
            self.processor_list.clear()  # Clear now so stop() won't try to remove

        # Call async stop() methods properly
        if len(processors) > 0:
            futures = [self.trigger_stop_processors(processor, force=force) for processor in processors]
            # Wait for all futures with a total timeout of 5 seconds
            done, not_done = concurrent.futures.wait(futures, timeout=5)

            # Check results of completed futures
            for future in done:
                try:
                    future.result()  # This will re-raise any exceptions
                except Exception as e:
                    logger.exception(f"Error stopping processor: {e}")

            # Log any futures that didn't complete
            if not_done:
                logger.error(f"Timeout: {len(not_done)} processor(s) did not stop within 5 seconds")

        self._initlize_model.close()
        self._cv_process_pool.shutdown(wait=True)
        self._clip_process_pool.shutdown(wait=True)
        self._safe_thread_event_loop.close()

        if self._messager:
            self._messager.close()

        if self._message_pool:
            self._message_pool.shutdown(wait=True)
            self._message_pool = None

    def create_video_processor(
        self, file_path, id=None, chunk_params: Optional[ChunkParams] = None, **kwargs
    ) -> SOPVideoProcessor:
        if id is None:
            id = uuid.uuid4()
        processor = SOPVideoProcessor(
            file_path,
            id=id,
            manager=self,
            chunk_params=chunk_params,
            cv_process_pool=self._cv_process_pool,
            clip_process_pool=self._clip_process_pool,
            vlm_model=self._initlize_model.vlm_model,
            vlm_inference_pool=self._vlm_inference_pool,
            vlm_request_pool=self._vlm_request_pool,
            message_pool=self._message_pool,
            messager=self._messager,
            **kwargs,
        )
        if processor:
            self.add_video_processor(processor)
        return processor

    def add_video_processor(self, processor: SOPVideoProcessor):
        with self.processor_list_lock:
            self.processor_list[processor._id] = processor

    def remove_video_processor(self, id: str):
        with self.processor_list_lock:
            self.processor_list.pop(id, None)

    def get_video_processor(self, id: str):
        with self.processor_list_lock:
            return self.processor_list.get(id, None)


async def process_single_video(processor: SOPVideoProcessor, tm: TimeMeasure) -> Tuple[str, List[Tuple[float, float]]]:
    """Process a single video and collect its chunks."""
    loop = asyncio.get_running_loop()
    processor_id = processor._id

    # Start processing (non-blocking)
    await processor.start()

    # Collect all chunks from the queue
    chunks = []
    while True:
        chunk = await loop.run_in_executor(None, lambda: processor.final_queue.get(block=True))
        if chunk is None:
            break
        chunks.append(chunk)
        tm.update_execute_time()
        logger.debug(f"Processor {processor_id} - Received chunk_info: {chunk}, time: {tm.elapsed_time:.3f}s")

    # Wait for processing to complete
    logger.info(f"Processor {processor_id} - Waiting for processing to complete")
    await processor.stop()
    logger.info(f"Processor {processor_id} - Processing completed")

    # logger.info(f"Processor {processor_id} - Total chunks: {len(chunks)}, elapsed: {tm.elapsed_time:.3f}s")
    return processor_id, chunks


# unit test
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepStream Action Detector Pipeline")
    parser.add_argument(
        "--video-path",
        type=str,
        default="",
        help="Path to the video file or RTSP stream URL",
    )
    # add a batch size
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the action detection")
    parser.add_argument("--camera-serial-number", type=str, default=None, help="Camera serial number")
    parser.add_argument("--camera-config", type=str, default=None, help="Camera configuration path")
    parser.add_argument("--vlm-model-path", type=str, default=None, help="Path to the VLM model")
    args = parser.parse_args()

    model_time = TimeMeasure("Model Initialization")
    sop_manager = SOPProcessManager()

    def handle_signal(signum, frame):
        logger.info(f"received signal: {signum} to force stop all processors")
        sop_manager.close(force=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    def dump_all_threads():
        for thread_id, frame in sys._current_frames().items():
            logger.info(f"Thread {threading._active.get(thread_id)} (id={thread_id}):")
            stack_trace = "".join(traceback.format_stack(frame))
            logger.info(f"Stack trace:\n{stack_trace}")

    signal.signal(signal.SIGUSR2, lambda *a: dump_all_threads())

    video_duration, video_fps = 0, 30.0
    video_path = args.video_path
    camera_serial_number = None
    camera_config = None
    if args.camera_serial_number is not None:
        camera_serial_number = args.camera_serial_number
        camera_config = args.camera_config
        video_path = "camera://" + camera_serial_number
        logger.info(f"######## using camera serial number: {camera_serial_number}, camera_config: {camera_config}")
    elif args.video_path:
        if not args.video_path.startswith("rtsp://"):
            video_path = os.path.abspath(args.video_path)
        logger.info(f"######## video_path: {video_path}")
        video_duration, video_fps, _, _ = SOPVideoProcessor.get_media_info(video_path)
        logger.info(f"######## video_duration: {video_duration:.3f}s, video_fps: {video_fps:.3f}")
    else:
        raise ValueError("Either video_path or camera_serial_number must be provided")

    sop_manager.wait_for_model_ready()
    model_time.update_execute_time()
    logger.info(f"######## model initialized in {model_time.elapsed_time:.3f}s")

    logger.info(f"######## video_duration: {video_duration:.3f}s, video_fps: {video_fps:.3f}")
    chunk_params = ChunkParams(min_length_sec=1, max_length_sec=10, threshold=0.8)
    sop_video_processors = [
        sop_manager.create_video_processor(
            video_path,
            id=idx,
            chunk_params=chunk_params,
            camera_serial_number=camera_serial_number,
            camera_config=camera_config,
        )
        for idx in range(args.batch_size)
    ]

    async def main():
        total_tms = [TimeMeasure("Total Execution") for _ in range(args.batch_size)]

        # Start all processors concurrently
        tasks = [
            asyncio.create_task(process_single_video(proc, total_tms[idx]))
            for idx, proc in enumerate(sop_video_processors)
        ]

        # Wait for all processors to complete
        results = await asyncio.gather(*tasks)

        # breakdowns
        e2e_cv_time = []
        e2e_vlm_time = []
        e2e_1st_chunk_time = [tm.first_execute_time for tm in total_tms]
        e2e_total_time = [tm.total_execute_time for tm in total_tms]
        for proc_id, chunks in results:
            pipeline_starting_timestamp = chunks[0].get("pipeline_starting_timestamp", 0)
            pipeline_vlm_starting_timestamp = min(
                chunk.get("pipeline_vlm_starting_timestamp", float("inf")) for chunk in chunks
            )
            pipeline_vlm_ready_timestamp = max(chunk.get("pipeline_vlm_ready_timestamp", 0) for chunk in chunks)
            pipeline_cv_ready_timestamp = max(chunk.get("pipeline_cv_ready_timestamp", 0) for chunk in chunks)
            e2e_cv_time.append(pipeline_cv_ready_timestamp - pipeline_starting_timestamp)
            e2e_vlm_time.append(pipeline_vlm_ready_timestamp - pipeline_vlm_starting_timestamp)

        # Log summary
        logger.info(f"All processors completed")
        for proc_id, chunks in results:
            logger.info(f"Processor {proc_id}: {len(chunks)} chunks")
            for i, chunks_info in enumerate(chunks):  # chunks_info is a tuple of (start_time, end_time, response)
                logger.info(f"chunk info, index: {i}, {chunk_info_func(chunks_info)}")

        # performance benchmark log
        total_chunks = sum(len(chunks) for _, chunks in results)

        avg_video_cv_execute_time = (
            sum(chunk.get("cv_execute_time", 0) for _, chunks in results for chunk in chunks) / args.batch_size
        )
        avg_video_vlm_execute_time = (
            sum(chunk.get("vlm_execute_time", 0) for _, chunks in results for chunk in chunks) / total_chunks
        )
        avg_video_checker_execute_time = (
            sum(chunk.get("checker_execute_time", 0) for _, chunks in results for chunk in chunks) / args.batch_size
        )
        avg_first_execute_time = sum(e2e_1st_chunk_time) / args.batch_size
        avg_total_execute_time = sum(e2e_total_time) / args.batch_size
        logger.info(f"Total chunks: {total_chunks}")
        logger.info(f"Average E2E CV execute time: {sum(e2e_cv_time) / args.batch_size:.3f}s, [overlapped with VLM]")
        logger.info(f"Average E2E VLM execute time: {sum(e2e_vlm_time) / args.batch_size:.3f}s, [overlapped with CV]")
        logger.info(f"Average video Checker execute time: {avg_video_checker_execute_time:.3f}s")
        # vlm_sequence_time = avg_total_execute_time - (avg_video_cv_execute_time + avg_video_checker_execute_time)
        # logger.info(f"Average video vlm sequencing inference time: {vlm_sequence_time:.3f}s")
        logger.info(f"Model initialization time: {model_time.total_execute_time:.3f}s")
        logger.info(f"Concurrent number: {args.batch_size}")
        logger.info(f"Video duration: {video_duration:.3f}s, video_fps: {video_fps:.3f}")
        logger.info(f"First Chunk execution time: {avg_first_execute_time:.3f}s")
        logger.info(f"Total execution time: {avg_total_execute_time:.3f}s")
        logger.info(
            f"Average throughput: {video_duration * video_fps * args.batch_size / avg_total_execute_time:.3f} frames/sec"
        )
        return results

    results = asyncio.run(main())
    sop_manager.close()
