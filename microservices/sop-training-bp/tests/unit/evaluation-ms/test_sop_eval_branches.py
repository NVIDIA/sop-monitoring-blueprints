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

"""Branch-coverage tests for sop_eval inference functions.

Covers the ``--use-fps-or-nframes nframes`` branch (which imports cv2 and
counts frames) for both backends, and the vllm branch where
``build_vllm_video_mm_data`` returns ``None`` so no video is attached. The
``fps`` branches are already covered by test_sop_eval.py; these exercise the
remaining paths without requiring real cv2 / model dependencies.
"""
import argparse
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MS_PATH = str(PROJECT_ROOT / "microservices" / "evaluation-ms")
if MS_PATH not in sys.path:
    sys.path.insert(0, MS_PATH)


# ── Helpers (mirrors test_sop_eval.py) ──────────────────────────────────────────

def make_fake_qwen_vl_utils():
    mod = types.ModuleType("qwen_vl_utils")
    mod.process_vision_info = MagicMock(return_value=(None, [MagicMock()], {"fps": 8.0}))
    return mod


def make_fake_cv2(total_frames=30):
    """Fake cv2 whose VideoCapture reports ``total_frames`` frames."""
    mod = types.ModuleType("cv2")
    mod.CAP_PROP_FRAME_COUNT = 7  # the real enum value; only identity matters here
    cap = MagicMock()
    cap.get.return_value = total_frames
    mod.VideoCapture = MagicMock(return_value=cap)
    return mod, cap


def make_mock_args(tmp_path, use_fps_or_nframes="nframes", nframes=12):
    return argparse.Namespace(
        val_videos_path=str(tmp_path / "val"),
        asset_root=str(tmp_path / "assets"),
        vlm_prompts_file="vlm_prompts.txt",
        fps=8,
        nframes=nframes,
        use_fps_or_nframes=use_fps_or_nframes,
    )


def setup_fake_val_dir(tmp_path):
    video_dir = tmp_path / "val" / "video1"
    video_dir.mkdir(parents=True)
    (video_dir / "01_assembly_v1_1_1.mp4").write_bytes(b"fake mp4 data")
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir(parents=True)
    (asset_dir / "vlm_prompts.txt").write_text("What step?\n(1) a\n(2) b")
    return tmp_path / "val"


@pytest.mark.unit
class TestNframesBranchTransformers:
    def test_nframes_mode_counts_frames_via_cv2(self, tmp_path):
        setup_fake_val_dir(tmp_path)
        fake_qvu = make_fake_qwen_vl_utils()
        fake_cv2, cap = make_fake_cv2(total_frames=30)
        sys.modules["qwen_vl_utils"] = fake_qvu
        sys.modules["cv2"] = fake_cv2
        try:
            from sop.sop_eval import action_inference_transformers

            mock_processor = MagicMock()
            mock_processor.apply_chat_template.return_value = "<chat>"
            mock_processor.return_value.to.return_value = MagicMock(input_ids=[[1, 2, 3]])
            mock_processor.batch_decode.return_value = ["(1) a"]
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.generate.return_value = [[1, 2, 3, 4]]

            args = make_mock_args(tmp_path, use_fps_or_nframes="nframes", nframes=12)
            result = action_inference_transformers(
                args, mock_model, mock_processor, {}, resolution_config={"max_pixels": 81920}
            )
        finally:
            sys.modules.pop("qwen_vl_utils", None)
            sys.modules.pop("cv2", None)

        # cv2 path was taken: VideoCapture opened the chunk and was released.
        fake_cv2.VideoCapture.assert_called_once()
        cap.release.assert_called_once()
        assert "video1" in result


@pytest.mark.unit
class TestNframesBranchVllm:
    def test_nframes_mode_counts_frames_via_cv2(self, tmp_path):
        setup_fake_val_dir(tmp_path)
        fake_qvu = make_fake_qwen_vl_utils()
        fake_cv2, cap = make_fake_cv2(total_frames=20)
        sys.modules["qwen_vl_utils"] = fake_qvu
        sys.modules["cv2"] = fake_cv2
        try:
            from sop.sop_eval import action_inference_vllm

            mock_output = MagicMock()
            mock_output.outputs[0].text = "(1) a"
            mock_llm = MagicMock()
            mock_llm.generate.return_value = [mock_output]
            mock_processor = MagicMock()
            mock_processor.apply_chat_template.return_value = "<chat>"

            args = make_mock_args(tmp_path, use_fps_or_nframes="nframes", nframes=12)
            result = action_inference_vllm(
                args, mock_llm, mock_processor, MagicMock(), resolution_config={"max_pixels": 81920}
            )
        finally:
            sys.modules.pop("qwen_vl_utils", None)
            sys.modules.pop("cv2", None)

        fake_cv2.VideoCapture.assert_called_once()
        cap.release.assert_called_once()
        assert result["video1"][0][1] == "(1) a"

    def test_no_video_attached_when_mm_data_is_none(self, tmp_path):
        """build_vllm_video_mm_data → None means mm_data has no 'video' key."""
        setup_fake_val_dir(tmp_path)
        fake_qvu = make_fake_qwen_vl_utils()
        sys.modules["qwen_vl_utils"] = fake_qvu
        try:
            from sop.sop_eval import action_inference_vllm

            captured = []
            mock_output = MagicMock()
            mock_output.outputs[0].text = "(1) a"

            def capture_generate(inputs_list, *a, **kw):
                captured.extend(inputs_list)
                return [mock_output]

            mock_llm = MagicMock()
            mock_llm.generate.side_effect = capture_generate
            mock_processor = MagicMock()
            mock_processor.apply_chat_template.return_value = "<chat>"

            args = make_mock_args(tmp_path, use_fps_or_nframes="fps")
            with patch("utils.eval_utils.build_vllm_video_mm_data", return_value=None):
                action_inference_vllm(args, mock_llm, mock_processor, MagicMock())
        finally:
            sys.modules.pop("qwen_vl_utils", None)

        assert captured, "llm.generate should have been called"
        for call in captured:
            assert "video" not in call.get("multi_modal_data", {})
