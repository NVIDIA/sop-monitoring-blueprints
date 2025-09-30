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
Module to run VLM inferences
"""

import logging
import math

import torch
import transformers
import qwen_vl_utils
from torchcodec.decoders import VideoDecoder

from cosmos_reason1_utils.vision import (
    overlay_text_on_tensor,
)

_LOGGER = logging.getLogger(__name__)

class CosmosReason1:

    def __init__(self, model_path, device: str = "cuda"):
        self._device = device
        self._model, self._processor, self._sampling_params = self._load_model_transformers(model_path)

    @property
    def device(self):
        return self._device

    def chunk_and_infer(self,
                        prompt: str,
                        video_filename: str,
                        chunk_start_second: float | None,
                        chunk_end_second: float | None,
                        system_prompt: str | None = None,
                        timestamp: bool = False) -> str:

        # Check if video range can provide at least 2 frames
        if not self._check_video_range(video_filename, chunk_start_second, chunk_end_second):
            _LOGGER.error("Video range [%s, %s] does not provide sufficient frames (minimum 2 required)",
                          chunk_start_second, chunk_end_second)
            return ""

        user_content = []
        user_content.append({
            "type": "video",
            "video": video_filename,
            # FIXME: user-input?
            "fps": 10.0,
            "max_pixels": 81920,
            "video_start": chunk_start_second,
            "video_end": chunk_end_second,
        })

        user_content.append({"type": "text", "text": prompt})

        system_content = []
        if system_prompt:
            _LOGGER.debug("System prompt: %s", system_prompt)
            system_content.append({"type": "text", "text": system_prompt})

        messages = []
        messages.append({"role": "user", "content": user_content})
        if system_content:
            messages.append({"role": "system", "content": system_content})

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(messages, return_video_kwargs=True)

        if timestamp:
            video_inputs = self._add_timestamp_to_video(video_inputs, video_kwargs)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        generated_ids = self._model.generate(**inputs, generation_config=self._sampling_params)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        #import os
        #from cosmos_reason1_utils.vision import save_tensor
        #this_dir = os.path.dirname(os.path.abspath(__file__))
        #output_dir = os.getenv("VLM_DEBUG_OUTPUT_DIR", this_dir)
        #save_dir = os.path.join(output_dir, f"{video_filename}-{chunk_start_second}-{chunk_end_second}")
        #os.makedirs(save_dir, exist_ok=True)
        #for i, video in enumerate(video_inputs):
        #    save_tensor(video, f"{save_dir}")
        #with open(f"{save_dir}/text.txt", "w") as f:
        #    f.write(text)
        #with open(f"{save_dir}/output_text.txt", "w") as f:
        #    f.write(output_text[0])
        ###########
        #_LOGGER.info(f"Saved debug output to {save_dir}")

        return output_text[0]

    def inference(self, prompt: str, video_filenames: list[str], system_prompt: str | None = None) -> list[str]:
        return self._inference_transformers(prompt, video_filenames, system_prompt)

    def _add_timestamp_to_video(self, video_inputs: list[torch.Tensor], video_kwargs: dict) -> list[torch.Tensor]:
        ret = []
        for i, video in enumerate(video_inputs):
            ret.append(overlay_text_on_tensor(video, fps=video_kwargs["fps"][i]))
        return ret

    def _inference_transformers(self, prompt: str, video_filenames: list[str], system_prompt: str | None = None) -> list[str]:
        all_outputs = []

        for video_filename in video_filenames:
            output_text = self.chunk_and_infer(prompt, video_filename, None, None, system_prompt)
            all_outputs.append(output_text)

        return all_outputs

    def _load_model_transformers(self, model_path: str):
        model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map=self._device
        )

        processor = transformers.AutoProcessor.from_pretrained(model_path, use_fast=True)

        generation_config = transformers.GenerationConfig(max_new_tokens=4096)

        return model, processor, generation_config

    def _check_video_range(self, video_filename: str, chunk_start_second: float | None, chunk_end_second: float | None) -> bool:
        """
        Check if we can get at least 2 frames from the video given the time range.

        Args:
            video_filename: Path to the video file
            chunk_start_second: Start time in seconds (None means start from beginning)
            chunk_end_second: End time in seconds (None means until end)

        Returns:
            True if we can get at least 2 frames, False otherwise
        """
        try:
            # Get video metadata using torchcodec VideoDecoder
            decoder = VideoDecoder(video_filename)
            metadata = decoder.metadata
            video_fps = metadata.average_fps
            duration_sec = metadata.duration_seconds
            total_frames = int(duration_sec * video_fps)

            if total_frames < 2 or video_fps <= 0.0:
                _LOGGER.warning("Video %s has insufficient frames (%d) or invalid fps (%.2f)",
                               video_filename, total_frames, video_fps)
                return False

            max_duration = duration_sec

            # Process start frame
            if chunk_start_second is not None:
                video_start = max(0.0, min(chunk_start_second, max_duration))
                start_frame = math.ceil(video_start * video_fps)
            else:
                start_frame = 0

            # Process end frame
            if chunk_end_second is not None:
                video_end = max(0.0, min(chunk_end_second, max_duration))
                end_frame = math.floor(video_end * video_fps)
                end_frame = min(end_frame, total_frames - 1)
            else:
                end_frame = total_frames - 1

            # Ensure we have at least 2 frames
            available_frames = end_frame - start_frame + 1

            if available_frames < 2:
                _LOGGER.warning("Insufficient frames (%d) in range [%.2f, %.2f] for video %s",
                               available_frames,
                               chunk_start_second or 0.0,
                               chunk_end_second or max_duration,
                               video_filename)
                return False

            return True

        except Exception as e:
            _LOGGER.error("Error checking video range for %s: %s", video_filename, e)
            return False
