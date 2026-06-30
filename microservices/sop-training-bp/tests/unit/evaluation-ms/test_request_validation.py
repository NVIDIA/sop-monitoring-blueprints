######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Unit tests for evaluation-ms/validation/request_validation.py — covers the
cross-field validation that lets E2eEvaluationRequest carry either a DDM job
reference (default) or a uniform chunk length.
"""

import pytest
from pydantic import ValidationError


class TestE2eEvaluationRequestDefault:
    """Existing e2e flow (no chunking_algorithm field) keeps working — defaults to DDM."""

    def test_default_chunking_algorithm_is_ddm(self):
        from validation.request_validation import E2eEvaluationRequest

        req = E2eEvaluationRequest(
            training_job_id="tj-1",
            ddm_training_job_id="ddm-1",
            val_dataset_id="ds-1",
        )
        assert req.chunking_algorithm == "ddm"
        assert req.ddm_training_job_id == "ddm-1"
        assert req.chunk_length_sec is None

    def test_legacy_request_without_chunking_field_validates(self):
        """Backwards compat: existing clients still POST without chunking_algorithm."""
        from validation.request_validation import E2eEvaluationRequest

        req = E2eEvaluationRequest(
            training_job_id="tj-1",
            ddm_training_job_id="ddm-1",
            val_dataset_id="ds-1",
            fps=8,
            temperature=0.0,
            score_threshold=0.5,
            nms_sec=0.0,
            ddm_batch_size=8,
        )
        assert req.chunking_algorithm == "ddm"


class TestE2eEvaluationRequestDdm:
    """When chunking_algorithm == 'ddm', ddm_training_job_id must be present."""

    def test_explicit_ddm_with_training_job_id_validates(self):
        from validation.request_validation import E2eEvaluationRequest

        req = E2eEvaluationRequest(
            training_job_id="tj-1",
            ddm_training_job_id="ddm-1",
            val_dataset_id="ds-1",
            chunking_algorithm="ddm",
        )
        assert req.chunking_algorithm == "ddm"

    def test_ddm_without_training_job_id_rejected(self):
        from validation.request_validation import E2eEvaluationRequest

        with pytest.raises(ValidationError) as exc_info:
            E2eEvaluationRequest(
                training_job_id="tj-1",
                val_dataset_id="ds-1",
                chunking_algorithm="ddm",
            )
        assert "ddm_training_job_id" in str(exc_info.value).lower()

    def test_ddm_with_empty_training_job_id_rejected(self):
        from validation.request_validation import E2eEvaluationRequest

        with pytest.raises(ValidationError):
            E2eEvaluationRequest(
                training_job_id="tj-1",
                ddm_training_job_id="",
                val_dataset_id="ds-1",
                chunking_algorithm="ddm",
            )


class TestE2eEvaluationRequestUniform:
    """When chunking_algorithm == 'uniform', chunk_length_sec is required and must be > 0."""

    def test_uniform_with_chunk_length_validates(self):
        from validation.request_validation import E2eEvaluationRequest

        req = E2eEvaluationRequest(
            training_job_id="tj-1",
            val_dataset_id="ds-1",
            chunking_algorithm="uniform",
            chunk_length_sec=10.0,
        )
        assert req.chunking_algorithm == "uniform"
        assert req.chunk_length_sec == 10.0
        # ddm_training_job_id is optional in uniform mode
        assert req.ddm_training_job_id is None

    def test_uniform_without_chunk_length_rejected(self):
        from validation.request_validation import E2eEvaluationRequest

        with pytest.raises(ValidationError) as exc_info:
            E2eEvaluationRequest(
                training_job_id="tj-1",
                val_dataset_id="ds-1",
                chunking_algorithm="uniform",
            )
        assert "chunk_length_sec" in str(exc_info.value).lower()

    def test_uniform_with_zero_chunk_length_rejected(self):
        from validation.request_validation import E2eEvaluationRequest

        with pytest.raises(ValidationError):
            E2eEvaluationRequest(
                training_job_id="tj-1",
                val_dataset_id="ds-1",
                chunking_algorithm="uniform",
                chunk_length_sec=0.0,
            )

    def test_uniform_with_negative_chunk_length_rejected(self):
        from validation.request_validation import E2eEvaluationRequest

        with pytest.raises(ValidationError):
            E2eEvaluationRequest(
                training_job_id="tj-1",
                val_dataset_id="ds-1",
                chunking_algorithm="uniform",
                chunk_length_sec=-1.0,
            )

    def test_uniform_does_not_require_ddm_training_job_id(self):
        """Uniform chunking must not crash if ddm_training_job_id is absent."""
        from validation.request_validation import E2eEvaluationRequest

        req = E2eEvaluationRequest(
            training_job_id="tj-1",
            val_dataset_id="ds-1",
            chunking_algorithm="uniform",
            chunk_length_sec=5.0,
        )
        assert req.ddm_training_job_id is None


class TestE2eEvaluationRequestInvalidAlgorithm:
    """Anything other than 'ddm' or 'uniform' must be rejected up front."""

    def test_unknown_algorithm_rejected(self):
        from validation.request_validation import E2eEvaluationRequest

        with pytest.raises(ValidationError):
            E2eEvaluationRequest(
                training_job_id="tj-1",
                val_dataset_id="ds-1",
                chunking_algorithm="kmeans",
                chunk_length_sec=5.0,
            )

    def test_empty_algorithm_rejected(self):
        from validation.request_validation import E2eEvaluationRequest

        with pytest.raises(ValidationError):
            E2eEvaluationRequest(
                training_job_id="tj-1",
                val_dataset_id="ds-1",
                chunking_algorithm="",
                chunk_length_sec=5.0,
            )


class TestResolutionConfig:
    """The resolution_config field used to be `Optional[dict]` which gave
    callers no clue about the schema (Leo flagged this on MR !36). Replace
    with a typed model that documents the shape and defaults."""

    def test_default_values_match_training_config(self):
        from validation.request_validation import ResolutionConfig

        rc = ResolutionConfig()
        # Mirrors assets/config/train_config.toml [custom.vision]: 16k vision tokens
        # at 40 frames. Same resolution as training keeps the VLM in-distribution.
        assert rc.max_frames == 40
        assert rc.total_pixels == 16572416

    def test_optional_pixel_overrides_default_to_none(self):
        from validation.request_validation import ResolutionConfig

        rc = ResolutionConfig()
        assert rc.resized_height is None
        assert rc.resized_width is None
        assert rc.max_pixels is None
        assert rc.min_pixels is None

    def test_accepts_user_overrides(self):
        from validation.request_validation import ResolutionConfig

        rc = ResolutionConfig(
            max_frames=30, total_pixels=12688256,
            resized_height=567, resized_width=1008,
            max_pixels=81920, min_pixels=1024,
        )
        assert rc.max_frames == 30
        assert rc.resized_height == 567
        assert rc.max_pixels == 81920
        assert rc.min_pixels == 1024

    def test_request_accepts_resolution_config_model(self):
        from validation.request_validation import EvaluationRequest, ResolutionConfig

        req = EvaluationRequest(
            training_job_id="tj-1",
            val_dataset_id="ds-1",
            resolution_config=ResolutionConfig(max_frames=20),
        )
        assert req.resolution_config.max_frames == 20
        assert req.resolution_config.total_pixels == 16572416  # default

    def test_request_accepts_resolution_config_dict(self):
        # Pydantic should coerce a plain dict into the typed model so
        # existing JSON clients keep working.
        from validation.request_validation import E2eEvaluationRequest

        req = E2eEvaluationRequest(
            training_job_id="tj-1",
            ddm_training_job_id="ddm-1",
            val_dataset_id="ds-1",
            resolution_config={"max_frames": 50, "max_pixels": 65536},
        )
        assert req.resolution_config.max_frames == 50
        assert req.resolution_config.max_pixels == 65536

    def test_unknown_field_rejected(self):
        # Typed model should reject typos so silent fallthroughs don't bite.
        from validation.request_validation import ResolutionConfig

        with pytest.raises(ValidationError):
            ResolutionConfig(maxx_frames=40)  # typo


class TestTopP:
    """`top_p` was hard-coded to 1.0 in the subprocess SamplingParams, which
    is irrelevant at temperature=0 (argmax) but starts mattering as soon as
    the user raises temperature. Expose as a request field so the caller
    controls both knobs together."""

    def test_per_chunk_top_p_defaults_to_one(self):
        from validation.request_validation import EvaluationRequest

        req = EvaluationRequest(training_job_id="tj-1", val_dataset_id="ds-1")
        assert req.top_p == 1.0

    def test_e2e_top_p_defaults_to_one(self):
        from validation.request_validation import E2eEvaluationRequest

        req = E2eEvaluationRequest(
            training_job_id="tj-1",
            ddm_training_job_id="ddm-1",
            val_dataset_id="ds-1",
        )
        assert req.top_p == 1.0

    def test_top_p_accepts_user_override(self):
        from validation.request_validation import EvaluationRequest, E2eEvaluationRequest

        a = EvaluationRequest(training_job_id="tj-1", val_dataset_id="ds-1", top_p=0.5)
        b = E2eEvaluationRequest(
            training_job_id="tj-1", ddm_training_job_id="ddm-1",
            val_dataset_id="ds-1", top_p=0.9,
        )
        assert a.top_p == 0.5
        assert b.top_p == 0.9

    def test_top_p_out_of_range_rejected(self):
        from validation.request_validation import EvaluationRequest

        with pytest.raises(ValidationError):
            EvaluationRequest(training_job_id="tj-1", val_dataset_id="ds-1", top_p=1.5)
        with pytest.raises(ValidationError):
            EvaluationRequest(training_job_id="tj-1", val_dataset_id="ds-1", top_p=-0.1)
