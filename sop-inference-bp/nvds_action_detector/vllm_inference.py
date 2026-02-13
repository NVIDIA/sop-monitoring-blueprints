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
import concurrent.futures
import copy
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import transformers
import vllm
from torch.multiprocessing import Queue
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from . import ds_logger
from .utils import TimeMeasure

logger = ds_logger.get_logger(__name__)

try:
    import qwen_vl_utils

    # from cosmos_reason1_utils.vision import overlay_text_on_tensor
except ImportError as e:
    qwen_vl_utils = None
    overlay_text_on_tensor = None
    logger.exception(f"Error importing dependencies for vllm: {e}")

VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.3"))
VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "20480"))
VLLM_MAX_NUM_SEQS = int(os.getenv("VLLM_MAX_NUM_SEQS", "16"))

VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "256"))
VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.2"))
VLM_FPS = float(os.getenv("VLM_FPS", "8.0"))
VLM_MAX_PIXELS = int(os.getenv("VLM_MAX_PIXELS", "0"))
VLM_MAX_FRAMES = int(os.getenv("VLM_MAX_FRAMES", "0"))
VLM_MAX_TOTAL_PIXELS = int(os.getenv("VLM_MAX_TOTAL_PIXELS", "0"))
VLM_RESIZED_HEIGHT = int(os.getenv("VLM_RESIZED_HEIGHT", "0"))
VLM_RESIZED_WIDTH = int(os.getenv("VLM_RESIZED_WIDTH", "0"))

VLM_PROFILING = bool(os.getenv("VLM_PROFILING", "0").lower() in ["yes", "y", "true", "1"])
VLM_DEBUG_DUMMP_FRAMES = bool(os.getenv("VLM_DEBUG_DUMMP_FRAMES", "0").lower() in ["yes", "y", "true", "1"])

FRAME_FACTOR = 2
IMAGE_FACTOR = 28
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768
VIDEO_FRAME_MIN_PIXELS = VIDEO_MIN_TOKEN_NUM * IMAGE_FACTOR * IMAGE_FACTOR
VIDEO_FRAME_MAX_PIXELS = VIDEO_MAX_TOKEN_NUM * IMAGE_FACTOR * IMAGE_FACTOR
MODEL_SEQ_LEN = VLLM_MAX_NUM_SEQS


VLM_PROMPT = """There are 10 possible steps for the SOP (Standard Operation Procedure) of the given video.
What step is the operator doing?
(1) installing the first fan by connecting the connector and then pressing the fan in place
(2) installing the second fan by connecting the connector and then pressing the fan in place
(3) installing the third fan by connecting the connector and then pressing the fan in place
(4) installing the forth fan by connecting the connector and then pressing the fan in place
(5) installing the fifth fan by connecting the connector and then pressing the fan in place
(6) installing the sixth fan by connecting the connector and then pressing the fan in place
(7) installing the first Power Supply by placing the power supply in its compartment and then firmly pressing it in its place until the latch clicks
(8) installing the second Power Supply by placing the power supply in its compartment and then firmly pressing it in its place until the latch clicks
(9) installing the server cover by placing it in its shown location on top of the server and then presses the black latch to lock the cover in place
(10) doing action not belong to the defined SOP
"""

SYSTEM_PROMPT = ""


class VLLMInference:
    def __init__(self, model_path, device: str = "cuda:0", **kwargs):
        self._event_loop = asyncio.new_event_loop()

        def exception_handler(loop, context):
            """Handle exceptions in the event loop."""
            exception = context.get("exception")
            message = context.get("message", "Unhandled exception in event loop")
            logger.error(f"Event loop exception: {message}", exc_info=exception)

        def loop_thread():
            asyncio.set_event_loop(self._event_loop)
            self._event_loop.set_exception_handler(exception_handler)
            self._event_loop.run_forever()

        self._inference_thread = threading.Thread(target=loop_thread, daemon=True)
        self._inference_thread.start()
        self._device = device
        self._gpu_memory_utilization = kwargs.pop("gpu_memory_utilization", VLLM_GPU_MEMORY_UTILIZATION)
        logger.info(
            f"VLLMInference initialized with device: {self._device}, gpu_memory_utilization: {self._gpu_memory_utilization}"
        )
        logger.info(f"VLLMInference Loading model from: {model_path}")
        self._llm, self._processor, self._sampling_params = self._load_model(model_path)
        logger.info(f"VLLMInference model {model_path} loaded")

    def __del__(self):
        self.stop()

    def stop(self):
        if self._event_loop is not None and self._event_loop.is_running():
            try:
                if hasattr(self._llm, "engine") and hasattr(self._llm.engine, "shutdown"):
                    self._llm.engine.shutdown()
                elif hasattr(self._llm, "shutdown"):
                    self._llm.shutdown()
            except Exception as e:
                logger.warning(f"Error during LLM shutdown: {e}")
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)
            self._inference_thread.join()
            # self._event_loop.close()
            self._event_loop = None
            self._inference_thread = None

    def _load_model(self, model_path: str):
        engine_args = AsyncEngineArgs(
            model=model_path,
            max_model_len=VLLM_MAX_MODEL_LEN,
            limit_mm_per_prompt={"image": 8, "video": 1},
            gpu_memory_utilization=self._gpu_memory_utilization,
            max_num_seqs=VLLM_MAX_NUM_SEQS,
        )
        llm = AsyncLLMEngine.from_engine_args(engine_args)
        processor = transformers.AutoProcessor.from_pretrained(model_path, use_fast=True)
        sampling_params = vllm.SamplingParams(temperature=VLM_TEMPERATURE, max_tokens=VLM_MAX_TOKENS)

        return llm, processor, sampling_params

    def inference(self, prompt: str, video: Any, **kwargs) -> str:
        user_content = []
        if isinstance(video, str):
            video_filename = video
        else:
            video_filename = "video.mp4"
        video_info = {
            "type": "video",
            "video": video_filename,
            "fps": VLM_FPS,
        }

        if VLM_MAX_PIXELS > 0:
            video_info["max_pixels"] = VLM_MAX_PIXELS
        if VLM_MAX_FRAMES > 0:
            video_info["max_frames"] = VLM_MAX_FRAMES
        if VLM_MAX_TOTAL_PIXELS > 0:
            video_info["total_pixels"] = VLM_MAX_TOTAL_PIXELS
        if VLM_RESIZED_HEIGHT > 0:
            video_info["resized_height"] = VLM_RESIZED_HEIGHT
        if VLM_RESIZED_WIDTH > 0:
            video_info["resized_width"] = VLM_RESIZED_WIDTH

        logger.debug(f"vllm inference, prompt: {prompt}, video_filename: {video_filename}")
        logger.debug(f"video_info={video_info}")
        user_content.append(video_info)
        user_content.append({"type": "text", "text": prompt})
        messages = []
        messages.append({"role": "user", "content": user_content})
        system_prompt = kwargs.pop("system_prompt", "")
        if system_prompt:
            logger.debug(f"Appending system prompt: {system_prompt}")
            messages.append({"role": "system", "content": system_prompt})

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs = None
        if isinstance(video, str):
            image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
                messages, return_video_kwargs=True
            )
            # video_inputs = torch.stack(video_inputs).float().to(self._device)
        else:  # list of tensors
            logger.info(f"vllm inference, video is a list of tensors, video size: {len(video)}")
            video_inputs, video_kwargs = self._preprocess_video(video_info, video, **kwargs)
            logger.info(
                f"vllm inference, video_inputs size: {video_inputs.shape}, dtype: {video_inputs.dtype}, device: {video_inputs.device}"
            )

        if "timestamp" in video_kwargs:
            video_inputs = self._add_timestamp_to_video(video_inputs, video_kwargs)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        # Construct the prompt with multi-modal data for AsyncLLMEngine
        engine_prompt = {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }
        # logger.info(f"=============== llm_inputs: prompt: {text}, mm_processor_kwargs: {video_kwargs}")
        logger.info(f"=============== video_inputs size: {len(video_inputs)}")
        logger.info(
            f"=============== mm_data video size: {video_inputs[0].shape}, dtype: {video_inputs[0].dtype}, device: {video_inputs[0].device}"
        )
        request_id = kwargs.pop("req_id", str(uuid.uuid4()))
        # extra_args = { "request_id": request_id, "use_tqdm": True }
        extra_args = {"request_id": request_id}

        logger.info(f"=============== extra_args: {extra_args}")
        temperature = kwargs.get("temperature", None)
        max_completion_tokens = kwargs.get("max_completion_tokens", None)
        seed = kwargs.get("seed", None)
        top_p = kwargs.get("top_p", None)
        sampling_params = copy.deepcopy(self._sampling_params)
        if temperature is not None:
            sampling_params.temperature = temperature
        if max_completion_tokens is not None:
            sampling_params.max_tokens = max_completion_tokens
        if seed is not None:
            sampling_params.seed = seed
        if top_p is not None:
            sampling_params.top_p = top_p

        async def generate_async():
            # Wait for the result
            final_output = None
            async for request_output in self._llm.generate(
                prompt=engine_prompt,
                sampling_params=sampling_params,
                **extra_args,
            ):
                final_output = request_output

            if not final_output:
                logger.error("Async generate returned no output")
                return {"req_id": request_id, "response": "Error: No response generated"}
            generated_text = final_output.outputs[0].text
            return {"req_id": request_id, "response": generated_text}

        output = asyncio.run_coroutine_threadsafe(generate_async(), self._event_loop)

        # output is a future object, need to get by
        # output.result() to get the result
        try:
            output = output.result(timeout=300)  # 5 minutes timeout
        except concurrent.futures.TimeoutError as e:
            logger.exception(f"Timeout waiting for async generation for request {request_id}")
            raise e
        except Exception as e:
            logger.exception(f"Error in async generation for request {request_id}: {e}", exc_info=True)
            raise e

        return output

    def _add_timestamp_to_video(self, video_inputs: list[torch.Tensor], video_kwargs: dict) -> list[torch.Tensor]:
        ret = []
        for i, video in enumerate(video_inputs):
            ret.append(overlay_text_on_tensor(video, fps=video_kwargs["fps"][i]))
        return ret

    def _preprocess_video(
        self,
        video_info: Dict[str, Any],
        video,
        **kwargs,
    ) -> tuple[list[torch.Tensor], dict]:
        if VLM_PROFILING:
            torch.cuda.synchronize()
            start_time = time.perf_counter()
        # Get frame info WITHOUT stacking all frames yet (memory optimization)
        if isinstance(video, list):
            total_frames = len(video)
            height, width, channels = video[0].shape
        elif isinstance(video, torch.Tensor):
            total_frames, _, height, width = video.shape
        else:
            raise ValueError(f"Invalid video type: {type(video)}")
        logger.debug("total_frames: %d", total_frames)
        original_video_fps = kwargs.get("video_fps", 25.0)

        # Calculate nframes FIRST before stacking (critical for memory optimization)
        if total_frames > 1:
            nframes = qwen_vl_utils.vision_process.smart_nframes(
                ele=video_info,
                total_frames=total_frames,
                video_fps=original_video_fps,
            )
        else:
            nframes = 1
        logger.debug(
            f"vllm inference sub-sampling n-frames: {nframes} from total frames: {total_frames} with original fps: {original_video_fps} and vlm_fps: {VLM_FPS}"
        )

        min_pixels = video_info.get("min_pixels", VIDEO_FRAME_MIN_PIXELS)
        total_pixels = video_info.get("total_pixels", MODEL_SEQ_LEN * IMAGE_FACTOR * IMAGE_FACTOR * 0.9)
        max_pixels = max(min(VIDEO_FRAME_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = video_info.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in video_info and "resized_width" in video_info:
            resized_height, resized_width = qwen_vl_utils.smart_resize(
                video_info["resized_height"],
                video_info["resized_width"],
                factor=IMAGE_FACTOR,
            )
        else:
            height, width, channels = video[0].shape
            resized_height, resized_width = qwen_vl_utils.smart_resize(
                height,
                width,
                factor=IMAGE_FACTOR,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        if isinstance(video, list):
            idx = [round(i) for i in torch.linspace(0, total_frames - 1, nframes).tolist()]
            video = [video[i].permute(2, 0, 1) for i in idx]
            total_frames = len(video)
            BATCH_SIZE = 16
            video_inputs = []
            for i in range(0, total_frames, BATCH_SIZE):
                batch_end = min(i + BATCH_SIZE, total_frames)
                batch = torch.stack([video[i + j] for j in range(batch_end - i)])
                batch = transforms.functional.resize(
                    batch,
                    [resized_height, resized_width],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ).cpu()
                video_inputs.extend(batch)
            video_inputs = torch.stack(video_inputs)
        elif isinstance(video, torch.Tensor):
            total_frames, _, height, width = video.shape
            video_inputs = video.permute(0, 2, 3, 1).float()
        else:
            raise ValueError(f"Invalid video type: {type(video)}")

        video_kwargs = {"do_sample_frames": False}
        video_kwargs.update({"fps": [10.0]})

        if VLM_PROFILING:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            logger.info(f"DEBUG: _preprocess_video time: {(end_time - start_time) * 1000} ms")
            if VLM_DEBUG_DUMMP_FRAMES:
                import cv2

                os.makedirs("debug_frames", exist_ok=True)
                for i in range(len(video_inputs)):
                    cv2.imwrite(f"debug_frames/video_input_{i}.png", video_inputs[i].cpu().numpy().transpose(1, 2, 0))
        return video_inputs, video_kwargs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepStream Action Detector Pipeline")
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to the video file or RTSP stream URL",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Path to the video directory",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for tensor operations (e.g., 'cuda:0', 'cpu')"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Path to the model",
    )
    args = parser.parse_args()

    video_list = []
    if args.video_path:
        video_list.append(os.path.abspath(args.video_path))

    if args.video_dir:
        video_extensions = (".mp4", ".mov")
        video_dir = os.path.abspath(args.video_dir)
        if os.path.isdir(video_dir):
            for filename in sorted(os.listdir(video_dir)):
                if filename.lower().endswith(video_extensions):
                    video_list.append(os.path.join(video_dir, filename))
        else:
            logger.warning(f"Video directory does not exist: {video_dir}")

    if not video_list:
        logger.error("No videos found. Please provide --video-path or --video-dir")
        exit(1)

    vlm = VLLMInference(args.model_path, device=args.device)

    def timed_inference(prompt, video_path, **kwargs):
        """Wrapper function to time each inference call."""
        video_file_name = os.path.basename(video_path)
        tm = TimeMeasure(f"inference-{video_file_name}")
        output = vlm.inference(prompt, video_path, **kwargs)
        tm.log_elapsed_time(f"inference-{video_file_name}")
        return output

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for i, video_path in enumerate(video_list):
            future = executor.submit(timed_inference, VLM_PROMPT, video_path, req_id=f"{i}")
            futures.append(future)

        # for future in concurrent.futures.as_completed(futures):
        for future in futures:
            result = future.result()
            logger.info(f"Result for {result['req_id']}: {result['response']}")
