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
    gpu_id: Optional[int] = None

    @model_validator(mode="after")
    def _check_chunking_args(self) -> "E2eEvaluationRequest":
        if self.chunking_algorithm == "ddm":
            if not self.ddm_training_job_id:
                raise ValueError(
                    "ddm_training_job_id is required when chunking_algorithm='ddm'"
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
        return self


class E2eEvaluationResponse(BaseModel):
    eval_job_id: str
    status: str
    message: str
    created_at: datetime


class E2eEvaluationStatus(BaseModel):
    eval_job_id: str
    training_job_id: str
    ddm_training_job_id: str
    val_dataset_id: str
    status: str
    overall_accuracy: Optional[float] = None
    avg_f1: Optional[float] = None
    checkpoint_step: Optional[int] = None
    created_at: datetime
    updated_at: datetime
