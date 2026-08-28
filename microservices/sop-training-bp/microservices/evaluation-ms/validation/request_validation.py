######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
######################################################################################################

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResolutionConfig(BaseModel):
    """
    Per-job vision-input resolution overrides for the VLM. Mirrors the
    qwen-vl-utils `process_vision_info` knobs. All fields are optional —
    `max_frames` and `total_pixels` default to the training config's
    `[custom.vision]` values so evaluation runs in-distribution unless the
    caller deliberately overrides.

    Field semantics (see qwen-vl-utils for full docs):
      max_frames        — cap on number of decoded video frames per chunk
      total_pixels      — target total pixel budget across all frames
                          (16572416 == 32*32*8092*2, i.e. 16k vision tokens)
      resized_height/   — explicit per-frame resize; if set, the parser
      resized_width       uses these instead of computing from total_pixels
      max_pixels        — upper bound on per-frame pixel count
      min_pixels        — lower bound on per-frame pixel count
    """
    model_config = ConfigDict(extra="forbid")

    max_frames: int = 40
    total_pixels: int = 16572416
    resized_height: Optional[int] = None
    resized_width: Optional[int] = None
    max_pixels: Optional[int] = None
    min_pixels: Optional[int] = None


class EvaluationRequest(BaseModel):
    training_job_id: str
    val_dataset_id: str
    fps: int = 8
    temperature: float = 0.0
    # Default 1.0 = no nucleus filtering. Irrelevant at temperature=0.
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    backend: str = "vllm"
    checkpoint_step: Optional[int] = None
    resolution_config: Optional[ResolutionConfig] = None
    # Pin the subprocess to a specific host GPU index. None = use all visible
    # GPUs. When set, app.py exports CUDA_VISIBLE_DEVICES=<gpu_id> so both
    # DDM's hardcoded cuda:0 and vLLM's auto-detected TP target it.
    gpu_id: Optional[int] = None
    # vLLM backend only. Context ceiling for the engine. Defaults to 32768
    # because deriving it from the checkpoint is a broken default: Qwen3-VL
    # declares 262144, which sizes the KV cache at ~36 GiB and the engine
    # refuses to start on an 80 GiB card. 32768 needs ~4.5 GiB and clears the
    # ~20.4k a default request actually uses. Raise it if a request is rejected
    # for length, lower it on a smaller card. Explicit null restores vLLM's own
    # derivation.
    max_model_len: Optional[int] = 32768


class EvaluationResponse(BaseModel):
    eval_job_id: str
    status: str
    message: str
    created_at: datetime


class EvaluationStatus(BaseModel):
    eval_job_id: str
    training_job_id: str
    val_dataset_id: str
    status: str
    overall_accuracy: Optional[float] = None
    checkpoint_step: Optional[int] = None
    created_at: datetime
    updated_at: datetime


class E2eEvaluationRequest(BaseModel):
    training_job_id: str
    # Required only when chunking_algorithm='ddm'; the cross-field rule
    # lives in _check_chunking_args below.
    ddm_training_job_id: Optional[str] = None
    val_dataset_id: str
    fps: int = 8
    temperature: float = 0.0
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    backend: str = "vllm"
    checkpoint_step: Optional[int] = None
    ddm_checkpoint: Optional[str] = None
    resolution_config: Optional[ResolutionConfig] = None
    score_threshold: float = 0.5
    nms_sec: float = 0.0
    ddm_batch_size: int = 8
    frames_per_segment_hint: int = 256
    # 'ddm' runs DDM-Net segmentation; 'uniform' uses fixed-length chunks
    # and skips DDM. Mirrors inference pipeline's chunking_options.algorithm.
    chunking_algorithm: Literal["ddm", "uniform"] = "ddm"
    chunk_length_sec: Optional[float] = None
    stride_sec: Optional[float] = None
    smooth_min_seg_sec: float = 2.0
    smooth_min_vote: int = 1
    non_sop_action: Optional[int] = None
    max_model_len: Optional[int] = 32768
    gpu_id: Optional[int] = None

    @model_validator(mode="after")
    def _check_chunking_args(self) -> "E2eEvaluationRequest":
        if self.chunking_algorithm == "ddm":
            if not self.ddm_training_job_id:
                raise ValueError(
                    "ddm_training_job_id is required when chunking_algorithm='ddm'"
                )
            if self.stride_sec is not None:
                raise ValueError(
                    "stride_sec applies to chunking_algorithm='uniform' only; DDM "
                    "segmentation produces its own non-overlapping boundaries"
                )
        else:  # uniform
            if self.chunk_length_sec is None:
                raise ValueError(
                    "chunk_length_sec is required when chunking_algorithm='uniform'"
                )
            if self.chunk_length_sec <= 0:
                raise ValueError(
                    f"chunk_length_sec must be > 0, got {self.chunk_length_sec}"
                )
            if self.stride_sec is not None:
                if self.stride_sec <= 0:
                    raise ValueError(f"stride_sec must be > 0, got {self.stride_sec}")
                # A stride wider than the window leaves stretches of the video inside no
                # window at all. Those instants collect no votes and are emitted as
                # non-SOP, so the run would silently report "nothing happening" over
                # footage it never actually evaluated. Equal is allowed: that is the
                # ordinary non-overlapping grid.
                if self.stride_sec > self.chunk_length_sec:
                    raise ValueError(
                        f"stride_sec ({self.stride_sec}) must be <= chunk_length_sec "
                        f"({self.chunk_length_sec}); a larger stride leaves gaps between "
                        f"windows that no window covers"
                    )
                if self.smooth_min_seg_sec < 0:
                    raise ValueError(
                        f"smooth_min_seg_sec must be >= 0, got {self.smooth_min_seg_sec}"
                    )
                if self.smooth_min_vote < 1:
                    raise ValueError(
                        f"smooth_min_vote must be >= 1, got {self.smooth_min_vote}"
                    )
                if self.non_sop_action is not None and self.non_sop_action < 1:
                    raise ValueError(
                        f"non_sop_action is a 1-based action index, got {self.non_sop_action}"
                    )
        return self


class E2eEvaluationResponse(BaseModel):
    eval_job_id: str
    status: str
    message: str
    created_at: datetime


class E2eEvaluationStatus(BaseModel):
    eval_job_id: str
    training_job_id: str
    # Optional, matching the request model: a uniform-chunking run has no DDM job.
    # Typed as a required str this rejected its own valid response, so every status
    # poll for a DDM-less run returned HTTP 500 and the job could never be observed
    # to finish -- the run succeeded while the caller recorded a timeout.
    ddm_training_job_id: Optional[str] = None
    val_dataset_id: str
    status: str
    overall_accuracy: Optional[float] = None
    avg_f1: Optional[float] = None
    checkpoint_step: Optional[int] = None
    created_at: datetime
    updated_at: datetime
