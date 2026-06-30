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

import json
import os

import pytest


class TestResolveDdmCheckpoint:
    """Tests for resolve_ddm_checkpoint()."""

    def test_finds_last_ckpt(self, temp_dir):
        job_dir = temp_dir / "job-ddm-1" / "train" / "job-ddm-1"
        job_dir.mkdir(parents=True)
        (job_dir / "last.ckpt").write_text("fake")
        (job_dir / "config.yaml").write_text(
            "dataset_config:\n  resolution: 224\n  frames_per_side: 5\n"
        )

        from utils.e2e_eval_utils import resolve_ddm_checkpoint

        ckpt_path, config_path = resolve_ddm_checkpoint(str(temp_dir), "job-ddm-1")
        assert os.path.basename(ckpt_path) == "last.ckpt"
        assert os.path.basename(config_path) == "config.yaml"

    def test_finds_named_ckpt(self, temp_dir):
        job_dir = temp_dir / "job-ddm-2" / "train" / "job-ddm-2"
        job_dir.mkdir(parents=True)
        (job_dir / "epoch_010-f1_score0.95.ckpt").write_text("fake")
        (job_dir / "last.ckpt").write_text("fake")
        (job_dir / "config.yaml").write_text("dataset_config:\n  resolution: 512\n")

        from utils.e2e_eval_utils import resolve_ddm_checkpoint

        ckpt_path, _ = resolve_ddm_checkpoint(
            str(temp_dir), "job-ddm-2", checkpoint_name="epoch_010-f1_score0.95.ckpt"
        )
        assert "epoch_010" in ckpt_path

    def test_raises_when_no_ckpt(self, temp_dir):
        job_dir = temp_dir / "job-ddm-3" / "train" / "job-ddm-3"
        job_dir.mkdir(parents=True)

        from utils.e2e_eval_utils import resolve_ddm_checkpoint

        with pytest.raises(FileNotFoundError):
            resolve_ddm_checkpoint(str(temp_dir), "job-ddm-3")

    def test_raises_when_no_train_dir(self, temp_dir):
        from utils.e2e_eval_utils import resolve_ddm_checkpoint

        with pytest.raises(FileNotFoundError):
            resolve_ddm_checkpoint(str(temp_dir), "nonexistent-job")

    def test_falls_back_to_job_level_config(self, temp_dir):
        job_dir = temp_dir / "job-ddm-4" / "train" / "job-ddm-4"
        job_dir.mkdir(parents=True)
        (job_dir / "last.ckpt").write_text("fake")
        # No config.yaml in train dir, but one at job level
        job_root = temp_dir / "job-ddm-4"
        (job_root / "job-ddm-4.yaml").write_text(
            "dataset_config:\n  resolution: 224\n"
        )

        from utils.e2e_eval_utils import resolve_ddm_checkpoint

        _, config_path = resolve_ddm_checkpoint(str(temp_dir), "job-ddm-4")
        assert "job-ddm-4.yaml" in config_path


class TestLoadDdmConfig:
    """Tests for load_ddm_config()."""

    def test_extracts_resolution_and_frames(self, temp_dir):
        cfg = temp_dir / "config.yaml"
        cfg.write_text(
            "dataset_config:\n  resolution: 224\n  frames_per_side: 5\n"
            "model_config:\n  backbone: resnet50\n"
        )

        from utils.e2e_eval_utils import load_ddm_config

        result = load_ddm_config(str(cfg))
        assert result["resolution"] == 224
        assert result["frames_per_side"] == 5

    def test_defaults_when_missing(self, temp_dir):
        cfg = temp_dir / "config.yaml"
        cfg.write_text("dataset_config:\n  resolution: 512\n")

        from utils.e2e_eval_utils import load_ddm_config

        result = load_ddm_config(str(cfg))
        assert result["resolution"] == 512
        assert result["frames_per_side"] == 5  # default


class TestExtractGoldenBoundaries:
    """Tests for extract_golden_boundaries()."""

    def test_extracts_from_annotation(self, temp_dir):
        anno = {
            "video1.mp4": [
                {"description": "idle", "start_timestamp": 0.0, "end_timestamp": 5.0},
                {"description": "step1", "start_timestamp": 5.5, "end_timestamp": 10.0},
                {"description": "step2", "start_timestamp": 10.2, "end_timestamp": 15.0},
            ]
        }
        anno_path = temp_dir / "anno.json"
        anno_path.write_text(json.dumps(anno))

        from utils.e2e_eval_utils import extract_golden_boundaries

        result = extract_golden_boundaries(str(anno_path))
        assert "video1.mp4" in result
        bdys = result["video1.mp4"]
        assert bdys[0] == 0.0  # start
        assert bdys[-1] == 15.0  # end
        assert len(bdys) == 4  # start + 2 midpoints + end

    def test_single_event(self, temp_dir):
        anno = {
            "vid.mp4": [
                {"description": "only", "start_timestamp": 0.0, "end_timestamp": 10.0},
            ]
        }
        anno_path = temp_dir / "anno.json"
        anno_path.write_text(json.dumps(anno))

        from utils.e2e_eval_utils import extract_golden_boundaries

        result = extract_golden_boundaries(str(anno_path))
        assert result["vid.mp4"] == [0.0, 10.0]

    def test_multiple_videos(self, temp_dir):
        anno = {
            "v1.mp4": [
                {"description": "a", "start_timestamp": 0.0, "end_timestamp": 5.0},
            ],
            "v2.mp4": [
                {"description": "b", "start_timestamp": 0.0, "end_timestamp": 8.0},
            ],
        }
        anno_path = temp_dir / "anno.json"
        anno_path.write_text(json.dumps(anno))

        from utils.e2e_eval_utils import extract_golden_boundaries

        result = extract_golden_boundaries(str(anno_path))
        assert len(result) == 2


class TestComputeTemporalMetrics:
    """Tests for compute_temporal_metrics()."""

    def test_perfect_match(self):
        from utils.e2e_eval_utils import compute_temporal_metrics

        golden = [0.0, 5.0, 10.0]
        pred = [0.0, 5.0, 10.0]
        m = compute_temporal_metrics(golden, pred, duration_sec=10.0)
        assert m["F1"] == 1.0
        assert m["Precision"] == 1.0
        assert m["Recall"] == 1.0

    def test_with_false_positives(self):
        from utils.e2e_eval_utils import compute_temporal_metrics

        golden = [0.0, 5.0, 10.0]
        pred = [0.0, 5.0, 7.5, 10.0]
        m = compute_temporal_metrics(golden, pred, duration_sec=10.0)
        assert m["True Positive"] == 3
        assert m["False Positive"] == 1

    def test_with_false_negatives(self):
        from utils.e2e_eval_utils import compute_temporal_metrics

        golden = [0.0, 5.0, 10.0, 15.0]
        pred = [0.0, 15.0]
        m = compute_temporal_metrics(golden, pred, duration_sec=15.0)
        assert m["True Positive"] == 2
        assert m["False Negative"] == 2

    def test_no_golden(self):
        from utils.e2e_eval_utils import compute_temporal_metrics

        m = compute_temporal_metrics(None, [0.0, 5.0], duration_sec=10.0)
        assert m["F1"] is None
        assert m["True Positive"] is None


class TestMapChunksToGroundTruth:
    """Tests for map_chunks_to_ground_truth()."""

    def test_simple_mapping(self):
        from utils.e2e_eval_utils import map_chunks_to_ground_truth

        golden_bdys = [0.0, 5.0, 10.0, 15.0]
        pred_bdys = [0.0, 4.8, 10.2, 15.0]
        result = map_chunks_to_ground_truth(pred_bdys, golden_bdys, action_count=3)
        assert len(result) == 3
        assert result[0] == 1
        assert result[1] == 2
        assert result[2] == 3

    def test_more_predicted_than_golden(self):
        from utils.e2e_eval_utils import map_chunks_to_ground_truth

        golden_bdys = [0.0, 5.0, 10.0]
        pred_bdys = [0.0, 2.5, 5.0, 7.5, 10.0]
        result = map_chunks_to_ground_truth(pred_bdys, golden_bdys, action_count=2)
        assert len(result) == 4
        assert result[0] == 1  # [0-2.5] → action 1
        assert result[1] == 1  # [2.5-5.0] → action 1
        assert result[2] == 2  # [5.0-7.5] → action 2
        assert result[3] == 2  # [7.5-10] → action 2

    def test_single_chunk(self):
        from utils.e2e_eval_utils import map_chunks_to_ground_truth

        golden_bdys = [0.0, 5.0, 10.0]
        pred_bdys = [0.0, 10.0]
        result = map_chunks_to_ground_truth(pred_bdys, golden_bdys, action_count=2)
        assert len(result) == 1
        # Single chunk [0-10] overlaps equally with both actions, picks first
        assert result[0] in [1, 2]


class TestUniformChunkBoundaries:
    """Tests for uniform_chunk_boundaries() — fixed-length chunking helper."""

    def test_exact_division(self):
        from utils.e2e_eval_utils import uniform_chunk_boundaries

        boundaries = uniform_chunk_boundaries(duration_sec=10.0, chunk_length_sec=5.0)
        assert boundaries == [0.0, 5.0, 10.0]

    def test_with_remainder(self):
        from utils.e2e_eval_utils import uniform_chunk_boundaries

        # 12s with 5s chunks → 0-5, 5-10, 10-12 (last one short)
        boundaries = uniform_chunk_boundaries(duration_sec=12.0, chunk_length_sec=5.0)
        assert boundaries == [0.0, 5.0, 10.0, 12.0]

    def test_duration_shorter_than_chunk(self):
        from utils.e2e_eval_utils import uniform_chunk_boundaries

        # 3s video with 5s chunks → single chunk 0-3
        boundaries = uniform_chunk_boundaries(duration_sec=3.0, chunk_length_sec=5.0)
        assert boundaries == [0.0, 3.0]

    def test_single_chunk_when_equal(self):
        from utils.e2e_eval_utils import uniform_chunk_boundaries

        # 5s video with 5s chunks → exactly one chunk
        boundaries = uniform_chunk_boundaries(duration_sec=5.0, chunk_length_sec=5.0)
        assert boundaries == [0.0, 5.0]

    def test_rejects_zero_chunk_length(self):
        from utils.e2e_eval_utils import uniform_chunk_boundaries

        with pytest.raises(ValueError):
            uniform_chunk_boundaries(duration_sec=10.0, chunk_length_sec=0.0)

    def test_rejects_negative_chunk_length(self):
        from utils.e2e_eval_utils import uniform_chunk_boundaries

        with pytest.raises(ValueError):
            uniform_chunk_boundaries(duration_sec=10.0, chunk_length_sec=-1.0)

    def test_rejects_zero_duration(self):
        from utils.e2e_eval_utils import uniform_chunk_boundaries

        # A 0-second video has no chunks; raise rather than silently return [0.0].
        with pytest.raises(ValueError):
            uniform_chunk_boundaries(duration_sec=0.0, chunk_length_sec=5.0)

    def test_fractional_chunk_length(self):
        from utils.e2e_eval_utils import uniform_chunk_boundaries

        # 10s with 2.5s chunks → 0, 2.5, 5, 7.5, 10
        boundaries = uniform_chunk_boundaries(duration_sec=10.0, chunk_length_sec=2.5)
        assert len(boundaries) == 5
        assert boundaries[0] == 0.0
        assert boundaries[-1] == 10.0

    def test_long_duration_short_chunks(self):
        from utils.e2e_eval_utils import uniform_chunk_boundaries

        # 60s video, 10s chunks → 7 boundaries (6 chunks)
        boundaries = uniform_chunk_boundaries(duration_sec=60.0, chunk_length_sec=10.0)
        assert len(boundaries) == 7
        assert boundaries == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
