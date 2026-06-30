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

"""Coverage for sop_e2e_eval.build_arg_parser and compute_e2e_accuracy edges.

build_arg_parser is pure argparse (no GPU/model deps); these tests pin its
required args, defaults, and choice validation. The compute_e2e_accuracy
cases cover the out-of-range-label, repeated-action, and
more-chunks-than-actions branches not hit by test_sop_e2e_eval.py.
"""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MS_PATH = str(PROJECT_ROOT / "microservices" / "evaluation-ms")
if MS_PATH not in sys.path:
    sys.path.insert(0, MS_PATH)


REQUIRED = [
    "--vlm-model-path", "/models/cr1",
    "--asset-root", "/assets",
    "--output-dir", "/out",
    "--video-dir", "/videos",
    "--anno-json-path", "/anno.json",
    "--actions-json-path", "/actions.json",
]


@pytest.mark.unit
class TestBuildArgParser:
    def test_parses_required_args_and_defaults(self):
        from sop.sop_e2e_eval import build_arg_parser

        args = build_arg_parser().parse_args(REQUIRED)

        assert args.vlm_model_path == "/models/cr1"
        assert args.anno_json_path == "/anno.json"
        assert args.actions_json_path == "/actions.json"
        # Defaults.
        assert args.video_ext == "mp4"
        assert args.fps == 8
        assert args.temperature == 0.0
        assert args.top_p == 1.0  # dest override of --top-p
        assert args.backend == "vllm"
        assert args.chunking_algorithm == "ddm"
        assert args.chunk_length_sec is None
        assert args.ddm_resolution == 224
        assert args.ddm_frames_per_side == 5
        assert args.score_threshold == 0.5
        assert args.nms_sec == 0.0
        assert args.ddm_batch_size == 8
        assert args.frames_per_segment_hint == 256
        assert args.tensor_parallel_size == 0
        assert args.resolution_config is None

    def test_missing_required_arg_exits(self):
        from sop.sop_e2e_eval import build_arg_parser

        # Drop the trailing --actions-json-path pair.
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(REQUIRED[:-2])

    def test_invalid_backend_choice_exits(self):
        from sop.sop_e2e_eval import build_arg_parser

        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(REQUIRED + ["--backend", "onnx"])

    def test_uniform_chunking_overrides(self):
        from sop.sop_e2e_eval import build_arg_parser

        args = build_arg_parser().parse_args(
            REQUIRED + ["--chunking-algorithm", "uniform", "--chunk-length-sec", "4.0"]
        )
        assert args.chunking_algorithm == "uniform"
        assert args.chunk_length_sec == pytest.approx(4.0)

    def test_transformers_backend_and_resolution_config(self):
        from sop.sop_e2e_eval import build_arg_parser

        args = build_arg_parser().parse_args(
            REQUIRED + ["--backend", "transformers",
                        "--resolution-config", '{"max_frames": 40}']
        )
        assert args.backend == "transformers"
        assert args.resolution_config == '{"max_frames": 40}'


@pytest.mark.unit
class TestComputeE2eAccuracyEdges:
    def test_out_of_range_action_index_labelled_unknown(self):
        from sop.sop_e2e_eval import compute_e2e_accuracy

        # action_idx 9 is past the 2-choice list -> "(?) unknown 9" label and
        # an empty expected_label (so verify_pred sees no match).
        vlm_outputs = {"v.mp4": {"c0": "anything"}}
        chunk_action_map = {"v.mp4": [9]}
        choices = ["(1) a", "(2) b"]

        result = compute_e2e_accuracy(vlm_outputs, chunk_action_map, choices)

        assert "9" in result["per_action"]
        assert result["per_action"]["9"]["label"].startswith("(?) unknown")
        assert result["per_action"]["9"]["total"] == 1

    def test_repeated_action_index_reuses_bucket(self):
        from sop.sop_e2e_eval import compute_e2e_accuracy

        # Same action_idx twice -> second iteration skips the label-init branch
        # (key already in per_action) and just accumulates totals.
        vlm_outputs = {"v.mp4": {"c0": "(1) a", "c1": "(1) a"}}
        chunk_action_map = {"v.mp4": [1, 1]}
        choices = ["(1) a", "(2) b"]

        result = compute_e2e_accuracy(vlm_outputs, chunk_action_map, choices)

        assert result["per_action"]["1"]["total"] == 2
        assert result["per_action"]["1"]["correct"] == 2
        assert result["per_action"]["1"]["accuracy"] == pytest.approx(1.0)

    def test_more_chunks_than_actions_breaks_early(self):
        from sop.sop_e2e_eval import compute_e2e_accuracy

        # Two chunks but only one action index -> the second chunk hits the
        # `i >= len(action_indices)` break and is not scored.
        vlm_outputs = {"v.mp4": {"c0": "(1) a", "c1": "(1) a"}}
        chunk_action_map = {"v.mp4": [1]}
        choices = ["(1) a", "(2) b"]

        result = compute_e2e_accuracy(vlm_outputs, chunk_action_map, choices)

        assert result["per_action"]["1"]["total"] == 1
