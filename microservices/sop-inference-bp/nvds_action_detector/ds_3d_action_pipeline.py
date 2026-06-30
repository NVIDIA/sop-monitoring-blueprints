#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import asyncio
import os
import queue
import string
import socket
import sys
import threading
import time
from queue import Empty, Full, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

import gi
import numpy as np
import torch
import torchvision.transforms.v2 as T
from pyservicemaker import (
    BatchMetadataOperator,
    Buffer,
    BufferOperator,
    BufferProvider,
    BufferRetriever,
    ColorFormat,
    EOSMessage,
    Feeder,
    Pipeline,
    PipelineState,
    Probe,
    Receiver,
    StateTransitionMessage,
    as_tensor,
)

# from torch.multiprocessing import Queue
from torch.utils.dlpack import from_dlpack, to_dlpack

gi.require_version("Gst", "1.0")
try:
    gi.require_version("GstRtspServer", "1.0")
    from gi.repository import GLib, Gst, GstRtspServer  # noqa: E402
except ValueError:
    GstRtspServer = None
    print("WARNING: GstRtspServer not found, RTSP streaming will not be available")
    from gi.repository import GLib, Gst  # noqa: E402

from . import ds_logger

# ds_logger.set_log_level(ds_logger.get_logger(), "INFO")
logger = ds_logger.get_logger(__name__)

# Gst.init(None)


# FRAMES_PER_SIDE is DDM's temporal context and must match the checkpoint.
# SEQUENCE_BATCH is runtime grouping/stride, not the training batch size.
FRAMES_PER_SIDE = int(os.getenv("FRAMES_PER_SIDE", "5"))
SEQUENCE_BATCH = int(os.getenv("SEQUENCE_BATCH", "8"))
SLIDING_WINDOWS_SIZE = 2 * FRAMES_PER_SIDE + SEQUENCE_BATCH

# Optional Compose env values render as "", so treat empty strings as unset.
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH")) if os.getenv("CAMERA_WIDTH") else 1280
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT")) if os.getenv("CAMERA_HEIGHT") else 720
CAMERA_FORMAT = os.getenv("CAMERA_FORMAT", "RGB")
CAMERA_FPS_NUM = int(os.getenv("CAMERA_FPS_NUM")) if os.getenv("CAMERA_FPS_NUM") else None
CAMERA_FPS_DEN = int(os.getenv("CAMERA_FPS_DEN")) if os.getenv("CAMERA_FPS_DEN") else None
CAMERA_NUM_BUFFERS = int(os.getenv("CAMERA_NUM_BUFFERS")) if os.getenv("CAMERA_NUM_BUFFERS") else None
camera_format_cpu_conversion = {
    "RGB": False,
    "UYVY": False,
    "NV12": False,
    "YUY2": True,
}
if CAMERA_FORMAT not in camera_format_cpu_conversion:
    logger.exception(f"Invalid CAMERA_FORMAT: {CAMERA_FORMAT}")
    sys.exit(1)
else:
    need_cpu_conversion = camera_format_cpu_conversion[CAMERA_FORMAT]
    logger.info(f"Default CAMERA_FORMAT: {CAMERA_FORMAT}, need_cpu_conversion: {need_cpu_conversion}")

SW_ENCODER = os.getenv("SW_ENCODER", "false").lower() in ["true", "1", "yes", "y"]

DS_ACTION_IN_RESOLUTION = int(os.getenv("DS_ACTION_IN_RESOLUTION", "224"))
DS_ACTION_IN_RESIZE_METHOD = os.getenv("DS_ACTION_IN_RESIZE_METHOD", "nearest").lower()
DS_ACTION_IN_RESIZE_METHOD_MAP = {
    "nearest": 0,
    "bilinear": 1,
    "cubic": 2,
    "super": 3,
    "lanzos": 4,
}
if DS_ACTION_IN_RESIZE_METHOD not in DS_ACTION_IN_RESIZE_METHOD_MAP:
    logger.exception(
        f"DS_ACTION_IN_RESIZE_METHOD can only be {list(DS_ACTION_IN_RESIZE_METHOD_MAP.keys())}, but got {DS_ACTION_IN_RESIZE_METHOD}"
    )
    sys.exit(1)

DS_ACTION_IN_RESIZE_METHOD_ENUM = DS_ACTION_IN_RESIZE_METHOD_MAP[DS_ACTION_IN_RESIZE_METHOD]

PREPROCESS_CONFIG_TEMPLATE = "configs/nvds_preprocess_template.txt"
INFERENCE_CONFIG_TEMPLATE = "configs/nvds_inference_template.txt"

PREPROCESS_CONFIG = "configs/nvds_preprocess_rendered.txt"
INFERENCE_CONFIG = "configs/nvds_inference_rendered.txt"
PREPROCESS_WIDTH = DS_ACTION_IN_RESOLUTION
PREPROCESS_HEIGHT = DS_ACTION_IN_RESOLUTION

PIPELINE_NAME = "ds_action_detector"
USE_TRITON = 1

TRITON_CONFIG_TEMPLATE = "nvds_action_detector/triton_model_repo/ddm/config_template.pbtxt"
TRITON_CONFIG = "nvds_action_detector/triton_model_repo/ddm/config.pbtxt"

# Create configuration based on environment variable.
for _template_file, _config_file in [
    (PREPROCESS_CONFIG_TEMPLATE, PREPROCESS_CONFIG),
    (INFERENCE_CONFIG_TEMPLATE, INFERENCE_CONFIG),
    (TRITON_CONFIG_TEMPLATE, TRITON_CONFIG),
]:
    with open(_template_file, "r", encoding="utf-8") as _fp:
        _template = string.Template(_fp.read())
    with open(_config_file, "w", encoding="utf-8") as _fp:
        _fp.write(
            _template.safe_substitute(
                {
                    "DS_ACTION_IN_RESOLUTION": DS_ACTION_IN_RESOLUTION,
                    "DS_ACTION_IN_RESIZE_METHOD_ENUM": DS_ACTION_IN_RESIZE_METHOD_ENUM,
                    "FRAMES_PER_SIDE": FRAMES_PER_SIDE,
                    "SEQUENCE_BATCH": SEQUENCE_BATCH,
                    "SLIDING_WINDOWS_SIZE": SLIDING_WINDOWS_SIZE,
                }
            )
        )

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class RTSPStreamingServer:
    def __init__(self, port, udp_port, stream_path="/ds-test"):
        if GstRtspServer is None:
            return
        self.server = GstRtspServer.RTSPServer.new()
        self.server.set_service(str(port))
        mounts = self.server.get_mount_points()
        factory = GstRtspServer.RTSPMediaFactory.new()
        factory.set_launch(
            f'( udpsrc name=pay0 port={udp_port} caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" )'
        )
        factory.set_shared(True)
        mounts.add_factory(stream_path, factory)
        self.server.attach(None)
        logger.info(f"RTSP Server started at rtsp://127.0.0.1:{port}{stream_path}")

class MyBufferProvider(BufferProvider):
    def __init__(self, width, height, device="cpu", framerate=30, format="RGB"):
        super().__init__()
        self.width = width
        self.height = height
        self.format = format
        self.framerate = framerate
        self.device = device
        self.count = 0
        self.expected = 255

    def generate(self, size):
        data = [self.count] * (self.width * self.height * 3)
        if self.count < self.expected:
            self.count += 1
        return Buffer() if self.count == self.expected else Buffer(data)


class TensorInspector(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        collected = []
        for u_mata in batch_meta.preprocess_batch_items:
            preprocess_batch = u_mata.as_preprocess_batch()
            if not preprocess_batch:
                continue
            # preprocess_tensor = preprocess_batch.preprocess_tensor_meta
            # print(f"######## preprocess_tensor.name: {preprocess_tensor.name}")
            # if preprocess_tensor.name == "input_0":
            #     collected.append(preprocess_tensor)

        assert len(collected) == 1 or len(collected) == 0
        print(f"######## tensor inspector: collected tensor nums: {len(collected)}")
        # for tensor_info in collected:
        #     print(f" inspected tensor: {tensor_info.name}, tensor.shape: {tensor_info.tensor.shape}, dtype: {tensor_info.tensor.dtype}")
        #     # assert "input_1" in [collected[0].name, collected[1].name]
        #     # assert "input_2" in [collected[0].name, collected[1].name]
        #     # assert collected[0].tensor.shape == (3, 224, 224)
        #     # assert collected[1].tensor.shape == (3, 224, 224)


class InferenceOutputTensorParser(BatchMetadataOperator):
    def __init__(self, queue: Queue):
        super().__init__()
        self._queue = queue
        self._sliding_windows_size = SLIDING_WINDOWS_SIZE
        self._temporal_pts = []
        self._first_metadata = True

    def handle_metadata(self, batch_meta):
        logger.debug(f"######## batch_meta: {batch_meta}")
        for frame_meta in batch_meta.frame_items:
            logger.debug(f"######## frame_meta: {frame_meta}")
            frame_num = frame_meta.frame_number
            pts = float(frame_meta.buffer_pts) / 1e9
            self._temporal_pts.append((frame_num, pts))
            if self._first_metadata:
                self._first_metadata = False
                self._queue.put((frame_num, pts, -1))

            if len(self._temporal_pts) > self._sliding_windows_size:
                self._temporal_pts.pop(0)

            boundary_items = []
            for object_meta in frame_meta.object_items:
                class_id = object_meta.class_id
                frame_id = object_meta.object_id
                confidence = object_meta.confidence
                label = object_meta.label
                logger.debug(
                    f"handle_metadata Object: class_id={class_id},"
                    f"Frame Number={frame_id},"
                    f"Confidence={confidence},"
                    f"Label={label}"
                )
                assert class_id == 0
                boundary_items.append((frame_id, confidence))

            logger.debug(f"handle_metadata boundary_items size: {len(boundary_items)}")
            boundary_items.sort(key=lambda x: x[0])
            for i, (frame_id, confidence) in enumerate(boundary_items):
                assert FRAMES_PER_SIDE + i < len(self._temporal_pts)
                frame_i, pts_i = self._temporal_pts[FRAMES_PER_SIDE + i]
                if frame_i != frame_id:
                    logger.warning(f"handle_metadata frame_i != frame_id, frame_i: {frame_i}, frame_id: {frame_id}")
                self._queue.put((frame_i, pts_i, confidence))

            # tensor_items = list(frame_meta.tensor_items)
            # print(f"######## tensor_items, size: {len(tensor_items)}, tensor_items: {tensor_items}")

            # for user_meta in tensor_items:
            #     print(f"######## user_meta: {user_meta}")
            #     output_tensor = user_meta.as_tensor_output()
            #     print(f"######## output_tensor: {output_tensor}")
            #     if output_tensor:
            #         # dict of tensors
            #         tensor_pts = self._temporal_pts[FRAMES_PER_SIDE]
            #         tensor_layers = output_tensor.get_layers()
            #         print(f"######## tensor_layers: {tensor_layers}")
            #         score_tensor = tensor_layers.pop("output_0", None)
            #         if score_tensor:
            #             print(f"######## score_tensor: {score_tensor}")
            #             self._queue.put((tensor_pts, score_tensor))
            #     print(f"######## output_tensor: {output_tensor}")


class FrameBufferRetriever(BufferRetriever):
    def __init__(self, queue):
        super().__init__()
        self.count = 0
        self._queue = queue

    def consume(self, buffer):
        logger.info(f"receiving frame: {self.count}, buffer.timestamp: {buffer.timestamp}")
        try:
            tensor = buffer.extract(0).clone()
            torch_tensor = torch.utils.dlpack.from_dlpack(tensor)
            timestamp = buffer.timestamp
            logger.info(f"received frame: {self.count}, timestamp: {timestamp}, tensor.shape: {tensor.shape}")
            self._queue.put((timestamp, tensor))
        except Full:
            logger.error(f"FrameBufferRetriever queue is full, buffer dropped")
            return 0
        except Exception as e:
            logger.exception(f"Error in FrameBufferRetriever: {e}")
            return 0
        self.count += 1
        return 1


class BufferTemporalOperator(BufferOperator):
    def __init__(self, unique_id: int = 0, device="cuda:0"):
        super().__init__()
        self.device = device

    def handle_buffer(self, buffer: Buffer):
        tensor, pts = None, None
        batch_meta = buffer.batch_meta
        print(f"######## batch_meta.n_frames: {batch_meta.n_frames}")
        for idx in range(batch_meta.n_frames):
            tensor, pts = buffer.extract(idx).clone(), buffer.timestamp
            print(f"######## buffer.extract{idx}.clone().type: {type(tensor)}")


def create_dummy_pipeline(gpu_id=0):
    """
    create a dummy pipeline with a buffer provider that generates dummy frames
    """

    class TensorBufferProvider(BufferProvider):
        def __init__(self):
            super().__init__()
            self.format = "RGB"
            self.width = PREPROCESS_WIDTH
            self.height = PREPROCESS_HEIGHT
            self.framerate = 30
            self.device = "gpu"
            self.max_count = SLIDING_WINDOWS_SIZE
            self.frame_idx = 0

        def generate(self, size):
            if self.frame_idx >= self.max_count:
                return Buffer()
            # torch_tensor = torch.zeros(self.height, self.width, 3, dtype=torch.int8).to("cuda:0")
            # pyservicemaker failed on new torch 2.10 versions.
            # use CPU tensor as workaround for now.
            torch_tensor = torch.zeros(self.height, self.width, 3, dtype=torch.int8)
            # import cupy as cp
            # torch_tensor = cp.zeros((self.height, self.width, 3), dtype=cp.int8)
            logger.debug(f"torch_tensor.shape: {torch_tensor.shape}, dtype: {torch_tensor.dtype}")
            logger.debug(f"dummy tensor height: {self.height}, width: {self.width}")
            ds_tensor = as_tensor(torch_tensor, "HWC").to_gpu(0)

            logger.debug(f"dummy pipeline generated frame: {self.frame_idx}")
            self.frame_idx += 1
            return ds_tensor.wrap(ColorFormat.RGB)

    provider = TensorBufferProvider()
    caps1 = (
        f"video/x-raw, format=RGB, width={provider.width}, height={provider.height}, framerate={provider.framerate}/1"
    )
    caps2 = f"video/x-raw(memory:NVMM), format=NV12, width={provider.width}, height={provider.height}, framerate={provider.framerate}/1"
    pipeline = Pipeline("ds_dummy_pipeline")
    pipeline.add("appsrc", "appsrc", {"caps": caps1, "do-timestamp": True})
    pipeline.add("capsfilter", "caps2", {"caps": caps2})
    pipeline.add("nvvideoconvert", "convert_to_nv12", {"nvbuf-memory-type": 2, "gpu-id": 0, "compute-hw": 1})
    pipeline.add(
        "nvstreammux",
        "mux",
        {
            "batch-size": 1,
            "width": PREPROCESS_WIDTH,
            "height": PREPROCESS_HEIGHT,
            "batched-push-timeout": -1,
            "live-source": False,
            "buffer-pool-size": 4,
        },
    )
    pipeline.add("nvdspreprocess", "preprocess_3d", {"config-file": PREPROCESS_CONFIG})
    if USE_TRITON:
        pipeline.add("nvinferserver", "inferencer", {"config-file-path": INFERENCE_CONFIG})
    else:
        pipeline.add("nvinfer", "inferencer", {"config-file-path": INFERENCE_CONFIG})
    pipeline.add("queue", "queue1")
    pipeline.add("fakesink", "fakesink", {"sync": False, "qos": False})
    pipeline.attach("appsrc", Feeder("feeder", provider), tips="need-data/enough-data")

    pipeline.link("appsrc", "convert_to_nv12", "caps2")
    pipeline.link(("caps2", "mux"), ("", "sink_%u"))
    pipeline.link("mux", "preprocess_3d", "inferencer", "queue1", "fakesink")
    return pipeline


def create_inference_pipeline(
    file_path,
    score_queue: Queue,
    gpu_id: Optional[int] = 0,
    frame_queue: Optional[Queue] = None,
    update_args: Optional[Dict[str, Any]] = None,
    uniform_chunk: Optional[bool] = False,
    **kwargs,
):
    # update the config file with the new prompt if it exists else use
    # the default prompt
    pipeline = Pipeline("ds_action_detector")
    #    pipeline.add("filesrc", "src", {"location": file_path})
    #    pipeline.add("decodebin", "srcbin")
    frame_retriever = kwargs.get("frame_retriever", None)
    mux_width = kwargs.get("mux_width", CAMERA_WIDTH)
    mux_height = kwargs.get("mux_height", CAMERA_HEIGHT)
    is_camera = False
    camera_serial_number = kwargs.get("camera_serial_number", None)
    if camera_serial_number is not None and file_path.startswith("camera://"):
        is_camera = True
        camera_serial_number = file_path.split("camera://")[1]

    if is_camera:
        camera_format = kwargs.get("camera_format", None)
        if not camera_format:
            camera_format = CAMERA_FORMAT
        if camera_format not in camera_format_cpu_conversion:
            raise ValueError(f"Invalid camera_format: {camera_format}")
        camera_width = kwargs.get("camera_width", None)
        camera_height = kwargs.get("camera_height", None)
        if not camera_width:
            camera_width = CAMERA_WIDTH
        if not camera_height:
            camera_height = CAMERA_HEIGHT
        camera_fps_num = kwargs.get("camera_fps_num", CAMERA_FPS_NUM)
        camera_fps_den = kwargs.get("camera_fps_den", CAMERA_FPS_DEN)
        mux_height = camera_height
        mux_width = camera_width
        logger.info(
            f"######## camera_serial_number: {camera_serial_number}, camera_format: {camera_format}, camera_width: {camera_width}, camera_height: {camera_height}, camera_fps_num: {camera_fps_num}, camera_fps_den: {camera_fps_den}"
        )

        if update_args is not None:
            update_args["camera_format"] = camera_format
            update_args["original_width"] = camera_width
            update_args["original_height"] = camera_height
            if camera_fps_num and camera_fps_den:
                update_args["original_fps_num"] = camera_fps_num
                update_args["original_fps_den"] = camera_fps_den
                update_args["original_fps"] = camera_fps_num / camera_fps_den
            update_args["mux_width"] = mux_width
            update_args["mux_height"] = mux_height

    logger.info(
        f"######## create_inference_pipeline with stream: {file_path}, mux_width: {mux_width}, mux_height: {mux_height}"
    )

    if is_camera:
        if os.getenv("PYLON_CAMEMU") == "1":
            # Loud, once-only notice is emitted at server startup (see api_server
            # lifespan); keep this per-stream line at debug to avoid repetition.
            logger.debug("Camera emulation mode enabled (PYLON_CAMEMU=1)")
        else:
            os.environ.pop("PYLON_CAMEMU", None)
        pylonsrc_props = {"device-serial-number": camera_serial_number, "capture-error": 1}
        if CAMERA_NUM_BUFFERS and os.getenv("PYLON_CAMEMU") == "1":
            pylonsrc_props["num-buffers"] = CAMERA_NUM_BUFFERS
        pipeline.add("pylonsrc", "pylonsrc", pylonsrc_props)
        # pipeline.add("pylonsrc", "pylonsrc")
        logger.info(f"######## pylonsrc added")
        camera_caps_str = f"video/x-raw, format={camera_format}, width={camera_width}, height={camera_height}"
        if camera_fps_num and camera_fps_den:
            camera_caps_str += f", framerate={camera_fps_num}/{camera_fps_den}"
        pipeline.add(
            "capsfilter",
            "cam_caps1",
            {"caps": camera_caps_str},
        )
        need_cpu_conversion = camera_format_cpu_conversion.get(camera_format, False)
        if need_cpu_conversion:
            pipeline.add("videoconvert", "cpu_convert1")
        camera_config = kwargs.get("camera_config", "")
        if camera_config and camera_config.endswith(".pfs"):
            pipeline["pylonsrc"].set({"pfs-location": camera_config})
        pipeline.add("nvvideoconvert", "cam_convert1", {"nvbuf-memory-type": 2, "gpu-id": gpu_id, "compute-hw": 1})
        pipeline.add("capsfilter", "cam_caps2", {"caps": f"video/x-raw(memory:NVMM), format=NV12"})
        pipeline.add("queue", "last_src")
        if need_cpu_conversion:
            pipeline.link("pylonsrc", "cam_caps1", "cpu_convert1", "cam_convert1", "cam_caps2", "last_src")
        else:
            pipeline.link("pylonsrc", "cam_caps1", "cam_convert1", "cam_caps2", "last_src")

    elif file_path.startswith("rtsp://") or file_path.startswith("file://"):
        pipeline.add("nvurisrcbin", "srcbin", {"uri": file_path})
        logger.info(f"######## Using RTSP or file path file_path: {file_path}")
    else:
        assert file_path, "file_path is required when camera_serial_number is missing"
        pipeline.add("nvurisrcbin", "srcbin", {"uri": "file://" + file_path})

    if is_camera is False and file_path.startswith("rtsp://"):
        pipeline["srcbin"].set(
            {
                "latency": 100,
                "leaky": 2,
                "max-size-buffers": 2,
                "num-extra-surfaces": 10,
                "init-rtsp-reconnect-interval": 10,
            }
        )
        is_live = True
    elif is_camera:
        is_live = True  # camera is always a live source
    else:
        is_live = False

    pipeline.add(
        "nvstreammux",
        "mux",
        {
            "batch-size": 1,
            "width": mux_width,
            "height": mux_height,
            "batched-push-timeout": -1,
            "live-source": is_live,
            "buffer-pool-size": 16,
            "gpu-id": gpu_id,
        },
    )
    pipeline.link(("last_src" if is_camera else "srcbin", "mux"), ("", "sink_%u"))
    logger.info("######## linked source and mux")

    if not uniform_chunk:
        pipeline.add("tee", "tee1")
        pipeline.add("queue", "queue1")
        pipeline.add("queue", "queue2")
        pipeline.add("nvdspreprocess", "preprocess_3d", {"config-file": PREPROCESS_CONFIG, "gpu-id": gpu_id})
        if USE_TRITON:
            pipeline.add("nvinferserver", "inferencer", {"config-file-path": INFERENCE_CONFIG})
        else:
            pipeline.add("nvinfer", "inferencer", {"config-file-path": INFERENCE_CONFIG, "gpu-id": gpu_id})
        pipeline.add("fakesink", "fakesink", {"sync": False, "qos": False})
        meta_probe = Probe("probe", InferenceOutputTensorParser(queue=score_queue))
        pipeline.attach("queue2", meta_probe)
        pipeline.link("mux", "tee1", "queue1", "preprocess_3d", "inferencer", "queue2", "fakesink")
        logger.info("######## linked mux -> tee1 -> preprocess_3d -> inferencer -> fakesink")
        frame_branch_src = "tee1"
    else:
        frame_branch_src = "mux"

    rtsp_port = kwargs.get("rtsp_port", None)
    rtsp_path = kwargs.get("rtsp_path", "/ds-test")
    if rtsp_port is not None:
        if GstRtspServer is not None:
            udp_port = get_free_port()
            pipeline.rtsp_server = RTSPStreamingServer(rtsp_port, udp_port, stream_path=rtsp_path)
            pipeline.add("queue", "queue_rtsp")
            pipeline.add("nvvideoconvert", "convert_rtsp", {"gpu-id": gpu_id})

            if SW_ENCODER:
                pipeline.add("capsfilter", "caps_rtsp", {"caps": "video/x-raw, format=I420"})
                
                # Check for available software encoders
                if not Gst.is_initialized():
                    Gst.init(None)

                if Gst.ElementFactory.find("x264enc"):
                    logger.info("Using x264enc for software encoding")
                    pipeline.add("x264enc", "enc_rtsp", {"bitrate": 4000, "tune": "zerolatency", "speed-preset": "superfast", "bframes": 0})
                elif Gst.ElementFactory.find("avenc_h264"):
                    logger.info("Using avenc_h264 for software encoding")
                    pipeline.add("avenc_h264", "enc_rtsp", {"bitrate": 4000000})
                elif Gst.ElementFactory.find("openh264enc"):
                    logger.info("Using openh264enc for software encoding")
                    pipeline.add("openh264enc", "enc_rtsp", {"bitrate": 4000000})
                else:
                    factories = Gst.Registry.get().get_feature_list(Gst.ElementFactory)
                    available_encs = [f.get_name() for f in factories if "enc" in f.get_name()]
                    logger.warning(f"No suitable H264 software encoder found. Falling back to mjpeg encoding using jpegenc. Available encoders: {available_encs}")
                    
                    # Fallback to MJPEG if no H264 SW encoder is available
                    # We need to change the payloader as well
                    pipeline.add("jpegenc", "enc_rtsp", {})
                    # For MJPEG, the payloader is typically rtpjpegpay
                    # But we've already added rtph264pay later in the code. We need to handle this.
                    # We will set a flag to change the payloader later
                    use_mjpeg = True

            else:
                use_mjpeg = False
                pipeline.add("capsfilter", "caps_rtsp", {"caps": "video/x-raw(memory:NVMM), format=NV12"})
                pipeline.add(
                    "nvv4l2h264enc",
                    "enc_rtsp",
                    {"bitrate": 4000000, "preset-level": 1, "insert-sps-pps": 1, "bufapi-version": 1, "num-B-Frames": 0},
                )

            if SW_ENCODER and locals().get("use_mjpeg", False):
                 pipeline.add("rtpjpegpay", "pay_rtsp", {"pt": 96})
            else:
                 pipeline.add("rtph264pay", "pay_rtsp", {"pt": 96})
            
            pipeline.add("udpsink", "sink_rtsp", {"host": "127.0.0.1", "port": udp_port, "async": False, "sync": False})
            pipeline.link(frame_branch_src, "queue_rtsp", "convert_rtsp", "caps_rtsp", "enc_rtsp", "pay_rtsp", "sink_rtsp")
            logger.info(f"######## linked RTSP stream on port {rtsp_port}")
        else:
            logger.warning("RTSP streaming requested but GstRtspServer is not available")

    if frame_queue is not None or frame_retriever is not None:
        pipeline.add("queue", "queue3")
        pipeline.add("queue", "queue4")
        pipeline.add("nvvideoconvert", "frame_converter", {"nvbuf-memory-type": 2, "gpu-id": gpu_id, "compute-hw": 1})
        pipeline.add("capsfilter", "frame_capsfilter", {"caps": "video/x-raw(memory:NVMM), format=RGB"})
        pipeline.add("appsink", "frame_sink", {"emit-signals": True, "sync": False, "qos": False, "async": True})
        pipeline.link(frame_branch_src, "queue3", "frame_converter", "frame_capsfilter", "queue4", "frame_sink")
        retriever = frame_retriever if frame_retriever else FrameBufferRetriever(frame_queue)

        pipeline.attach("frame_sink", Receiver("receiver", retriever), tips="new-sample")
        logger.info(f"######## linked {frame_branch_src} -> frame_converter -> frame_capsfilter -> frame_sink")
    elif uniform_chunk:
        pipeline.add("fakesink", "fakesink", {"sync": False, "qos": False})
        pipeline.link("mux", "fakesink")
        logger.info("######## linked mux -> fakesink (uniform, no frame retriever)")

    return pipeline


def encode_video(frame_queue: Queue, width: int, height: int, fps: int, output_path: str):
    pipeline_started_event = threading.Event()

    class MyBufferProvider(BufferProvider):
        def __init__(self, frame_queue):
            super().__init__()
            self.count = 0
            self._queue = frame_queue

        def generate(self, size):
            if not pipeline_started_event.is_set():
                pipeline_started_event.wait(0.2)
                logger.info(f"encode_video: pipeline is ready to receive frames")
            data = self._queue.get(block=True)
            if data is None:
                logger.info(f"encode_video: EOS sent, returning Buffer()")
                return Buffer()
            timestamp, torch_tensor = data
            logger.info(
                f"encode_video: got frame, timestamp: {timestamp}, tensor.shape: {torch_tensor.shape}, tensor.device: {torch_tensor.device}, tensor.dtype: {torch_tensor.dtype}"
            )
            if not torch_tensor.is_contiguous():
                logger.warning(f"torch_tensor is not contiguous, re-contiguousing")
                torch_tensor = torch_tensor.contiguous()
            if torch_tensor.is_cuda:
                torch.cuda.synchronize()  # Ensure CUDA operations are complete before DLPack conversion
            ds_tensor = as_tensor(torch_tensor, "HWC")  # TODO: handle batch size > 1
            buffer = ds_tensor.wrap(ColorFormat.RGB)
            # buffer.timestamp = timestamp
            self.count += 1
            return buffer

    fps = int(fps)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    encode_pipeline = Pipeline("encode-pipeline")
    encode_pipeline.add(
        "appsrc",
        "src",
        {
            "caps": f"video/x-raw(memory:NVMM), format=RGB, width={width}, height={height}, framerate={fps}/1",
            "do-timestamp": True,
        },
    )
    encode_pipeline.add("nvvideoconvert", "convert", {"nvbuf-memory-type": 2, "compute-hw": 1})
    # encode_pipeline.add("nvstreammux", "mux", {"batch-size": 1, "width": width, "height": height})
    # encode_pipeline.add("nvvideoconvert", "convert1", {"nvbuf-memory-type": 2, "compute-hw": 1})
    encode_pipeline.add("h264parse", "parser")
    encode_pipeline.add("qtmux", "qtmux")
    encode_pipeline.add("filesink", "sink", {"location": output_path, "sync": False})

    if SW_ENCODER:
        encode_pipeline.add("x264enc", "encoder")
        encode_pipeline.add("capsfilter", "capsfilter", {"caps": "video/x-raw, format=I420"})
    else:
        encode_pipeline.add("nvv4l2h264enc", "encoder")
        encode_pipeline.add("capsfilter", "capsfilter", {"caps": "video/x-raw(memory:NVMM), format=NV12"})

    # encode_pipeline.link("src", "convert").link(("convert", "mux"), ("", "sink_%u")).link("mux", "encoder").link("encoder", "parser").link("parser", "qtmux").link("qtmux", "sink")
    # encode_pipeline.link("src", "convert").link(("convert", "mux"), ("", "sink_%u")).link("mux", "convert1").link("convert1", "capsfilter").link("capsfilter", "encoder").link("encoder", "parser").link("parser", "qtmux").link("qtmux", "sink")
    encode_pipeline.link("src", "convert", "capsfilter", "encoder", "parser")
    encode_pipeline.link(("parser", "qtmux"), ("", "video_%u")).link("qtmux", "sink")
    encode_pipeline.attach("src", Feeder("feeder", MyBufferProvider(frame_queue)), tips="need-data/enough-data")

    def on_message(message):
        nonlocal pipeline_started_event
        if isinstance(message, StateTransitionMessage):
            logger.info(f"Encoder pipeline StateTransitionMessage received: {message}")
            # if message.new_state == PipelineState.PLAYING:
            if message.new_state == PipelineState.READY and not pipeline_started_event.is_set():
                pipeline_started_event.set()
            # elif message.new_state == PipelineState.INVALID:
            #     pipeline_started_event.clear()

        if isinstance(message, EOSMessage):
            logger.info("Encoder pipeline End-of-Stream received! Pipeline has finished processing.")

    logger.info("Starting encoder pipeline")
    encode_pipeline.start(on_message)
    logger.info("Encoder pipeline started, waiting for EOS")
    encode_pipeline.wait()
    logger.info("Encoder pipeline EOS done, stopping pipeline")
    encode_pipeline.stop()
    logger.info("Encoder pipeline stopped")
    return output_path


def encode_video_gst(frame_queue: Queue, width: int, height: int, fps: int, output_path: str):
    Gst.init(None)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline_desc = (
        "appsrc name=src is-live=false block=true "
        f"caps=video/x-raw,format=RGB,width={width},height={height},framerate={(int)(fps)}/1 ! "
        "nvvideoconvert nvbuf-memory-type=2 compute-hw=1 ! "
        "video/x-raw(memory:NVMM),format=NV12 ! "
        "nvv4l2h264enc ! h264parse ! qtmux ! "
        f"filesink location={output_path} sync=false "
    )
    pipeline = Gst.parse_launch(pipeline_desc)
    appsrc = pipeline.get_by_name("src")
    # appsrc.set_property("caps", f"video/x-raw,format=RGB,width={width},height={height},framerate={fps}/1")
    appsrc.set_property("is-live", False)
    # appsrc.set_property("do-timestamp", True)
    # appsrc.set_property("max-buffers", 1)
    # appsrc.set_property("queue-size", 1)

    def tensor_to_gst_buffer(tensor: torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.detach().cpu()

        tensor = tensor.contiguous().to(torch.uint8)
        data = tensor.numpy().tobytes()

        logger.debug(f"encode_video_gst: data length: {len(data)}")
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        return buf

    pipeline.set_state(Gst.State.PLAYING)
    first_offset = None

    while True:
        data = frame_queue.get(block=True)
        if data is None:
            break
        timestamp, tensor = data
        if first_offset is None:
            first_offset = timestamp
        timestamp -= first_offset
        buf = tensor_to_gst_buffer(tensor)
        buf.pts = (int)(timestamp * 1e9)
        buf.duration = int(1e9 / fps)
        ret = appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            logger.error(f"encode_video_gst: push-buffer returned {ret}, timestamp: {timestamp}")
            break

    logger.info(f"encoder pipeline with output_path:{output_path} Sending EOS...")
    appsrc.emit("end-of-stream")

    bus = pipeline.get_bus()
    while True:
        msg = bus.timed_pop_filtered(
            Gst.SECOND,
            Gst.MessageType.ERROR | Gst.MessageType.EOS,
        )
        if msg:
            t = msg.type
            if t == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                logger.error(f"encode_video_gst: message error: {err}, {debug}")
                break
            elif t == Gst.MessageType.EOS:
                logger.info("encode_video_gst: EOS received — done.")
                break

    logger.info(f"encoder pipeline with output_path:{output_path} pipeline stopped")
    pipeline.set_state(Gst.State.NULL)

    return output_path


def ds_boundary_infernce(
    file_path, score_queue: Queue, threshold: float = 0.8, gpu_id: int = 0, frame_queue: Queue = None, **kwargs
):
    start_time = time.time()
    pipeline = create_inference_pipeline(file_path, score_queue, gpu_id, frame_queue=frame_queue, **kwargs)
    logger.info(f"inference pipeline created in {time.time() - start_time:.3f} seconds")

    def on_message(message):
        if isinstance(message, EOSMessage):
            logger.info("End-of-Stream received! Pipeline has finished processing.")
            score_queue.put(None)
            if frame_queue is not None:
                frame_queue.put(None)
            logger.info(
                f"inference pipeline has finished w/ EOS in {time.time() - start_time:.3f} seconds, queue size: {score_queue.qsize()}"
            )

    pipeline.start(on_message)
    pipeline.wait()

    # time.sleep(1)

    # wait for the score_queue to be filled
    timestamps, bounds_ids, scores = [], [], []
    while not score_queue.empty():
        item = score_queue.get()
        if item is not None:
            frame_id, pts, score = item
            if score >= threshold:
                scores.append(score)
                bounds_ids.append(frame_id)
                timestamps.append(pts)
    logger.info(f"Pipeline has stopped in {time.time() - start_time:.3f} seconds")
    if len(scores) == 0:
        logger.warning("No scores received! Pipeline has finished processing.")
        return
    logger.info(f"boundary nums detected: {len(scores)}")
    logger.info(f"boundary ids: {bounds_ids}")
    logger.info(f"timestamps: {timestamps}")
    logger.info(f"scores: {scores}")
    return timestamps, bounds_ids, scores


if __name__ == "__main__":
    import argparse
    import os
    import signal
    import sys
    import threading
    import traceback

    def dump_all_threads():
        for thread_id, frame in sys._current_frames().items():
            logger.info(f"Thread {threading._active.get(thread_id)} (id={thread_id}):")
            stack_trace = "".join(traceback.format_stack(frame))
            logger.info(f"Stack trace:\n{stack_trace}")

    signal.signal(signal.SIGUSR2, lambda *a: dump_all_threads())

    parser = argparse.ArgumentParser(description="DeepStream Action Detector Pipeline")
    parser.add_argument(
        "--video-path",
        type=str,
        default="/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4",
        help="Path to the video file or RTSP stream URL",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use for tensor operations (e.g., 0, 1, 2, 3)")
    parser.add_argument("--threshold", type=float, default=0.8, help="Threshold for the action detection")
    parser.add_argument("--rtsp-port", type=int, default=None, help="RTSP server port (e.g. 8554)")
    args = parser.parse_args()

    video_path = os.path.abspath(args.video_path)
    logger.info(f"######## video_path: {video_path}")

    dummy_pipeline = create_dummy_pipeline()
    dummy_pipeline.start()
    # exit(0)
    score_queue = Queue()
    frame_queue = Queue()
    # frame_queue = None
    ds_boundary_infernce(
        video_path, score_queue, args.threshold, args.gpu_id, frame_queue=frame_queue, rtsp_port=args.rtsp_port
    )
    logger.info(f"frame_queue size: {frame_queue.qsize()}")
    dummy_pipeline.wait()
    dummy_pipeline.stop()
