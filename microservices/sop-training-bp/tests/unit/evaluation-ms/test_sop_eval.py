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

"""Unit tests for sop_eval.py — parse_action_index, read_txt, and inference functions."""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MS_PATH = str(PROJECT_ROOT / "microservices" / "evaluation-ms")
if MS_PATH not in sys.path:
    sys.path.insert(0, MS_PATH)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_fake_qwen_vl_utils():
    """Return a MagicMock that stands in for qwen_vl_utils."""
    mod = types.ModuleType("qwen_vl_utils")
    mod.process_vision_info = MagicMock(return_value=(None, [MagicMock()], {"fps": 8.0}))
    return mod


def make_mock_args(tmp_path, use_fps_or_nframes="fps"):
    args = argparse.Namespace(
        val_videos_path=str(tmp_path / "val"),
        asset_root=str(tmp_path / "assets"),
        vlm_prompts_file="vlm_prompts.txt",
        fps=8,
        nframes=12,
        use_fps_or_nframes=use_fps_or_nframes,
    )
    return args


def setup_fake_val_dir(tmp_path):
    """Create a fake validation directory with one video dir and one .mp4 chunk."""
    val_dir = tmp_path / "val"
    video_dir = val_dir / "video1"
    video_dir.mkdir(parents=True)
    # Create a stub .mp4 file (content irrelevant — model calls are mocked)
    (video_dir / "01_assembly_v1_1_1.mp4").write_bytes(b"fake mp4 data")

    asset_dir = tmp_path / "assets"
    asset_dir.mkdir(parents=True)
    (asset_dir / "vlm_prompts.txt").write_text("What step is the operator doing?\n(1) step one\n(2) step two")

    return val_dir


class TestParseActionIndex:
    # parse_action_index now always returns a list[int] (length 1 for single-op,
    # length >= 2 for concurrent two-op chunks). See sop/sop_eval.py.
    def test_standard_annotation_ms_format(self):
        from sop.sop_eval import parse_action_index
        assert parse_action_index("01_assembly_video_1_1.mp4") == [1]

    def test_two_digit_action(self):
        from sop.sop_eval import parse_action_index
        assert parse_action_index("11_video_2_3.mp4") == [11]

    def test_single_digit_zero_padded(self):
        from sop.sop_eval import parse_action_index
        assert parse_action_index("07_some_video_1_1.mp4") == [7]

    def test_no_leading_zero(self):
        from sop.sop_eval import parse_action_index
        assert parse_action_index("3_video_1_1.mp4") == [3]

    def test_fallback_on_unparseable(self):
        from sop.sop_eval import parse_action_index
        assert parse_action_index("weird_name.mp4") == [1]

    def test_fallback_on_non_numeric_prefix(self):
        from sop.sop_eval import parse_action_index
        assert parse_action_index("abc_video_1_1.mp4") == [1]

    def test_two_op_concurrent_prefix(self):
        from sop.sop_eval import parse_action_index
        assert parse_action_index("01-04_video_1_1.mp4") == [1, 4]

    def test_two_op_sorted(self):
        from sop.sop_eval import parse_action_index
        # filename order may not be sorted; output always is
        assert parse_action_index("10-01_video_1_1.mp4") == [1, 10]

    def test_two_op_multi_action_prefix(self):
        from sop.sop_eval import parse_action_index
        # >2 concurrent ids are also supported (defensive — annotation MS
        # only emits up to 2 today but the parser is general)
        assert parse_action_index("01-03-05-07_video_1_1.mp4") == [1, 3, 5, 7]

    def test_fallback_on_partial_two_op_with_letters(self):
        from sop.sop_eval import parse_action_index
        # mixed non-numeric tokens in the dash group fall back to [1]
        assert parse_action_index("01-ab_video_1_1.mp4") == [1]


class TestReadTxt:
    def test_reads_file_content(self, tmp_path):
        from sop.sop_eval import read_txt
        f = tmp_path / "prompt.txt"
        f.write_text("hello world")
        assert read_txt(str(f)) == "hello world"


class TestActionInferenceTransformers:
    def test_fps_path_returns_inference_results(self, tmp_path):
        """action_inference_transformers with fps mode returns results dict with video key."""
        setup_fake_val_dir(tmp_path)

        fake_qvu = make_fake_qwen_vl_utils()
        sys.modules["qwen_vl_utils"] = fake_qvu

        try:
            from sop.sop_eval import action_inference_transformers

            mock_processor = MagicMock()
            mock_processor.apply_chat_template.return_value = "<chat>"
            mock_processor.return_value.to.return_value = MagicMock(input_ids=[[1, 2, 3]])
            mock_processor.batch_decode.return_value = ["(1) step one"]

            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.generate.return_value = [[1, 2, 3, 4]]

            args = make_mock_args(tmp_path, use_fps_or_nframes="fps")
            result = action_inference_transformers(
                args, mock_model, mock_processor, {},
                system_prompt="Answer.", resolution_config={"max_pixels": 81920},
            )
        finally:
            sys.modules.pop("qwen_vl_utils", None)

        assert "video1" in result
        assert len(result["video1"]) == 1
        assert result["video1"][0][0] == [1]  # action ids parsed from filename (list, single-op)

    def test_default_resolution_config_applied_when_none(self, tmp_path):
        """resolution_config=None defaults to {"max_pixels": 81920}."""
        setup_fake_val_dir(tmp_path)

        fake_qvu = make_fake_qwen_vl_utils()
        sys.modules["qwen_vl_utils"] = fake_qvu

        try:
            from sop.sop_eval import action_inference_transformers

            mock_processor = MagicMock()
            mock_processor.apply_chat_template.return_value = "<chat>"
            mock_processor.return_value.to.return_value = MagicMock(input_ids=[[1]])
            mock_processor.batch_decode.return_value = ["(1) step one"]
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.generate.return_value = [[1, 2]]

            args = make_mock_args(tmp_path, use_fps_or_nframes="fps")
            # Pass resolution_config=None — should not raise
            result = action_inference_transformers(args, mock_model, mock_processor, {})
        finally:
            sys.modules.pop("qwen_vl_utils", None)

        assert "video1" in result

    def test_fps_in_video_kwargs_is_flattened(self, tmp_path):
        """If process_vision_info returns fps as list, it's cast to float."""
        setup_fake_val_dir(tmp_path)

        fake_qvu = make_fake_qwen_vl_utils()
        # Return fps as a list (triggers the isinstance branch)
        fake_qvu.process_vision_info = MagicMock(
            return_value=(None, [MagicMock()], {"fps": [8.0, 8.0]})
        )
        sys.modules["qwen_vl_utils"] = fake_qvu

        try:
            from sop.sop_eval import action_inference_transformers

            mock_processor = MagicMock()
            mock_processor.apply_chat_template.return_value = "<chat>"
            mock_processor.return_value.to.return_value = MagicMock(input_ids=[[1]])
            mock_processor.batch_decode.return_value = ["(1) step one"]
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.generate.return_value = [[1, 2]]

            args = make_mock_args(tmp_path)
            result = action_inference_transformers(args, mock_model, mock_processor, {})
        finally:
            sys.modules.pop("qwen_vl_utils", None)

        # No exception means the fps list was correctly handled
        assert "video1" in result


class TestActionInferenceVllm:
    def test_fps_path_returns_inference_results(self, tmp_path):
        """action_inference_vllm with fps mode returns results dict."""
        setup_fake_val_dir(tmp_path)

        fake_qvu = make_fake_qwen_vl_utils()
        sys.modules["qwen_vl_utils"] = fake_qvu

        try:
            from sop.sop_eval import action_inference_vllm

            mock_output = MagicMock()
            mock_output.outputs[0].text = "(1) step one"

            mock_llm = MagicMock()
            mock_llm.generate.return_value = [mock_output]

            mock_processor = MagicMock()
            mock_processor.apply_chat_template.return_value = "<chat>"

            mock_sampling_params = MagicMock()

            args = make_mock_args(tmp_path, use_fps_or_nframes="fps")
            result = action_inference_vllm(
                args, mock_llm, mock_processor, mock_sampling_params,
                system_prompt="Answer.", resolution_config={"max_pixels": 81920},
            )
        finally:
            sys.modules.pop("qwen_vl_utils", None)

        assert "video1" in result
        assert result["video1"][0][1] == "(1) step one"

    def test_default_resolution_config_applied_when_none(self, tmp_path):
        """resolution_config=None defaults to {"max_pixels": 81920}."""
        setup_fake_val_dir(tmp_path)

        fake_qvu = make_fake_qwen_vl_utils()
        sys.modules["qwen_vl_utils"] = fake_qvu

        try:
            from sop.sop_eval import action_inference_vllm

            mock_output = MagicMock()
            mock_output.outputs[0].text = "(1) step one"
            mock_llm = MagicMock()
            mock_llm.generate.return_value = [mock_output]
            mock_processor = MagicMock()
            mock_processor.apply_chat_template.return_value = "<chat>"

            args = make_mock_args(tmp_path, use_fps_or_nframes="fps")
            result = action_inference_vllm(args, mock_llm, mock_processor, MagicMock())
        finally:
            sys.modules.pop("qwen_vl_utils", None)

        assert "video1" in result

    def test_image_inputs_added_to_mm_data(self, tmp_path):
        """image_inputs != None → mm_data["image"] is populated."""
        setup_fake_val_dir(tmp_path)

        fake_qvu = make_fake_qwen_vl_utils()
        fake_image = MagicMock()
        fake_qvu.process_vision_info = MagicMock(
            return_value=(fake_image, [MagicMock()], {"fps": 8.0})
        )
        sys.modules["qwen_vl_utils"] = fake_qvu

        try:
            from sop.sop_eval import action_inference_vllm

            captured_calls = []
            mock_output = MagicMock()
            mock_output.outputs[0].text = "(1) step one"

            def capture_generate(inputs_list, *a, **kw):
                captured_calls.extend(inputs_list)
                return [mock_output]

            mock_llm = MagicMock()
            mock_llm.generate.side_effect = capture_generate
            mock_processor = MagicMock()
            mock_processor.apply_chat_template.return_value = "<chat>"

            args = make_mock_args(tmp_path)
            action_inference_vllm(args, mock_llm, mock_processor, MagicMock())
        finally:
            sys.modules.pop("qwen_vl_utils", None)

        assert any("image" in call.get("multi_modal_data", {}) for call in captured_calls)


class TestRcaParityOutputFormat:
    """The standalone evaluation script's outputs feed the sop-rca-plugin
    parser (`analyze_by_action_confusion.py` on branch wind/dev/sop-ft-orchestrate).
    These regression tests pin the format contracts so a future change can't
    silently break the agentic FT flow downstream of evaluation.

    The parser expects:
      - inference_results.json values are 3-tuples: [gt_action, pred_text, chunk_path]
      - log file contains lines starting with `Action Chunk: <path>` followed
        by `(N)<response text>` on the next line
      - log file contains an `Args:` line (so fps_by_action can be parsed)
    """

    _SCRIPT = (
        Path(__file__).resolve().parents[3]
        / "microservices/evaluation-ms/sop/sop_eval.py"
    )

    def test_inference_results_emit_three_tuple(self):
        src = self._SCRIPT.read_text()
        assert "inference_results[video].append([cur_action, response])" not in src, (
            "sop_eval.py must emit 3-tuple [gt_action, pred_text, chunk_path] "
            "in inference_results.json so the RCA parser can unpack it. "
            "Found a 2-tuple append — see analyze_by_action_confusion.py:parse_json_results."
        )
        # Each backend (transformers + vllm) appends once.
        assert src.count("inference_results[video].append([cur_action, response, chunk])") >= 2

    def test_per_chunk_log_uses_action_chunk_prefix(self):
        src = self._SCRIPT.read_text()
        # Old format: `chunk=<basename>, action=<n>, response=<...>` —
        # the RCA parser silently produces 0 entries from this format.
        assert "chunk={os.path.basename(chunk)}, action=" not in src, (
            "sop_eval.py must log `Action Chunk: <path>` followed by the response "
            "on the next line (RCA parser's expected format)."
        )
        # New format: emits `Action Chunk: <chunk>` line.
        assert "Action Chunk:" in src, (
            "sop_eval.py must emit `Action Chunk: <path>` lines that the RCA parser "
            "(analyze_by_action_confusion.py:parse_log_results) keys off."
        )

    def test_main_logs_args_at_startup(self):
        src = self._SCRIPT.read_text()
        # The RCA SKILL.md Step 2 extracts fps_by_action from an `Args:` line
        # in the per-action-chunk eval log.
        assert 'logging.info("Args: %s"' in src or 'logging.info(f"Args:' in src, (
            "sop_eval.py __main__ must log `Args: <args>` at startup so the RCA "
            "skill can extract fps_by_action (SKILL.md Step 2 — see fps_by_action row)."
        )

    def test_no_orphan_sop_eval_log_filehandler(self):
        src = self._SCRIPT.read_text()
        # The combined log.txt (written by app.py from subprocess stdout/stderr)
        # is the single source of truth. The previous `sop_eval_log.txt` FileHandler
        # ended up 0 bytes in practice; drop it to avoid confusion.
        assert "sop_eval_log.txt" not in src, (
            "sop_eval.py should not write its own sop_eval_log.txt FileHandler — "
            "app.py already captures subprocess stdout/stderr into log.txt."
        )
