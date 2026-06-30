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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


class TestTransformersBackendModelClass:
    """The transformers branch must use AutoModelForImageTextToText so it
    dispatches by config.architectures (Qwen2_5_VL for CR1 checkpoints,
    Qwen3VL for CR2). Hard-coding Qwen2_5_VLForConditionalGeneration would
    silently mis-init weights on a CR2 checkpoint and crash later in
    get_rope_index with `IndexError: index 1 is out of bounds for
    dimension 0 with size 1`.

    Regression net for both sop_eval.py (per-action-chunk) and
    sop_e2e_eval.py (e2e) transformers branches."""

    _SCRIPTS = [
        Path(__file__).resolve().parents[3]
        / "microservices/evaluation-ms/sop/sop_eval.py",
        Path(__file__).resolve().parents[3]
        / "microservices/evaluation-ms/sop/sop_e2e_eval.py",
    ]

    @pytest.mark.parametrize("script", _SCRIPTS, ids=lambda p: p.name)
    def test_uses_auto_dispatch_not_hardcoded_qwen2_5(self, script):
        src = script.read_text()
        assert "AutoModelForImageTextToText.from_pretrained" in src, (
            f"{script.name}: transformers branch must call "
            "AutoModelForImageTextToText.from_pretrained for CR1/CR2 dispatch."
        )
        assert "Qwen2_5_VLForConditionalGeneration.from_pretrained" not in src, (
            f"{script.name}: do not hard-code Qwen2_5_VLForConditionalGeneration; "
            "CR2 checkpoints (Qwen3VL architecture) will silently mis-init."
        )


class TestCollectAnnotations:
    """Test annotation collection from per-video subdirectories."""

    def test_collects_single_video(self, temp_dir):
        vid_dir = temp_dir / "video1"
        vid_dir.mkdir()
        anno = [{"description": "step1", "start_timestamp": 0.0, "end_timestamp": 5.0}]
        (vid_dir / "video1_annotation.json").write_text(json.dumps(anno))
        # Also create a video file so the key matches
        (temp_dir / "video1.mp4").write_text("fake")

        from sop.sop_e2e_eval import collect_annotations

        result = collect_annotations(str(temp_dir))
        assert "video1.mp4" in result
        assert len(result["video1.mp4"]) == 1
        assert result["video1.mp4"][0]["description"] == "step1"

    def test_collects_multiple_videos(self, temp_dir):
        for name in ["vid1", "vid2"]:
            d = temp_dir / name
            d.mkdir()
            anno = [{"description": "a", "start_timestamp": 0.0, "end_timestamp": 5.0}]
            (d / f"{name}_annotation.json").write_text(json.dumps(anno))
            (temp_dir / f"{name}.mp4").write_text("fake")

        from sop.sop_e2e_eval import collect_annotations

        result = collect_annotations(str(temp_dir))
        assert len(result) == 2

    def test_skips_dirs_without_json(self, temp_dir):
        (temp_dir / "empty_dir").mkdir()

        from sop.sop_e2e_eval import collect_annotations

        result = collect_annotations(str(temp_dir))
        assert len(result) == 0

    def test_skips_non_directories(self, temp_dir):
        (temp_dir / "some_file.txt").write_text("not a dir")

        from sop.sop_e2e_eval import collect_annotations

        result = collect_annotations(str(temp_dir))
        assert len(result) == 0

    def test_unrelated_json_no_longer_picked_up_as_annotation(self, temp_dir, caplog):
        """A subdir containing only non-annotation JSON (e.g. metadata.json)
        used to be silently loaded as the annotation. Now we require the
        *_annotation.json convention and log a warning."""
        import logging
        vid_dir = temp_dir / "videoX"
        vid_dir.mkdir()
        (vid_dir / "metadata.json").write_text('{"unrelated": true}')

        from sop.sop_e2e_eval import collect_annotations

        with caplog.at_level(logging.WARNING):
            result = collect_annotations(str(temp_dir))

        assert result == {}
        assert any(
            "videoX" in rec.message and "_annotation.json" in rec.message
            for rec in caplog.records
        )


class TestComputeE2eAccuracy:
    """Test the accuracy computation for e2e evaluation."""

    def test_perfect_accuracy(self):
        from sop.sop_e2e_eval import compute_e2e_accuracy

        vlm_outputs = {
            "video1.mp4": {
                "[0.00s-5.00s]": "(1) do step one",
                "[5.00s-10.00s]": "(2) do step two",
            }
        }
        chunk_action_map = {
            "video1.mp4": [1, 2],
        }
        choices = ["(1) do step one", "(2) do step two"]

        result = compute_e2e_accuracy(vlm_outputs, chunk_action_map, choices)
        assert result["overall_accuracy"] == 1.0
        assert result["per_action"]["1"]["correct"] == 1
        assert result["per_action"]["2"]["correct"] == 1

    def test_partial_accuracy(self):
        from sop.sop_e2e_eval import compute_e2e_accuracy

        vlm_outputs = {
            "video1.mp4": {
                "[0.00s-5.00s]": "(1) do step one",
                "[5.00s-10.00s]": "(1) do step one",  # wrong — should be action 2
            }
        }
        chunk_action_map = {
            "video1.mp4": [1, 2],
        }
        choices = ["(1) do step one", "(2) do step two"]

        result = compute_e2e_accuracy(vlm_outputs, chunk_action_map, choices)
        assert result["overall_accuracy"] == 0.5
        assert result["per_action"]["1"]["correct"] == 1
        assert result["per_action"]["2"]["correct"] == 0

    def test_empty_outputs(self):
        from sop.sop_e2e_eval import compute_e2e_accuracy

        result = compute_e2e_accuracy({}, {}, ["(1) a"])
        assert result["overall_accuracy"] == 0.0
        assert result["per_action"] == {}


class TestRunUniformStage:
    """Test the uniform-chunking branch of stage 1 — must produce the same
    output shape as run_ddm_stage so the VLM stage is reused unchanged."""

    def _make_args_and_anno(self, tmp_path, chunk_length_sec):
        # Fake video files (.mp4) to be picked up by glob.
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "v1.mp4").write_bytes(b"fake")
        (video_dir / "v2.mp4").write_bytes(b"fake")

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        anno = {
            "v1.mp4": [
                {"description": "(1) one", "start_timestamp": 0.0, "end_timestamp": 5.0},
                {"description": "(2) two", "start_timestamp": 5.0, "end_timestamp": 10.0},
            ],
            "v2.mp4": [
                {"description": "(1) one", "start_timestamp": 0.0, "end_timestamp": 6.0},
            ],
        }

        args = SimpleNamespace(
            video_dir=str(video_dir),
            video_ext="mp4",
            output_dir=str(output_dir),
            chunk_length_sec=chunk_length_sec,
        )
        return args, anno

    def test_uniform_stage_returns_same_shape_as_ddm(self, tmp_path):
        from sop.sop_e2e_eval import run_uniform_stage

        args, anno = self._make_args_and_anno(tmp_path, chunk_length_sec=5.0)

        # Mock the duration helper so we don't need real video files.
        durations = {"v1.mp4": 10.0, "v2.mp4": 6.0}
        with patch(
            "sop.sop_e2e_eval.get_video_duration_sec",
            side_effect=lambda p: durations[os.path.basename(p)],
        ):
            result = run_uniform_stage(args, anno)

        assert "v1.mp4" in result
        assert "v2.mp4" in result
        # Same shape as run_ddm_stage: each video has "boundaries" + "metric"
        assert result["v1.mp4"]["boundaries"] == [0.0, 5.0, 10.0]
        # 6s with 5s chunks -> [0, 5, 6]
        assert result["v2.mp4"]["boundaries"] == [0.0, 5.0, 6.0]
        assert "metric" in result["v1.mp4"]
        # avg_f1 / avg_precision / avg_recall populated like the DDM stage
        assert "avg_f1" in result
        assert "avg_precision" in result
        assert "avg_recall" in result

    def test_uniform_stage_writes_anno_and_results_json(self, tmp_path):
        from sop.sop_e2e_eval import run_uniform_stage

        args, anno = self._make_args_and_anno(tmp_path, chunk_length_sec=5.0)

        with patch(
            "sop.sop_e2e_eval.get_video_duration_sec",
            return_value=10.0,
        ):
            run_uniform_stage(args, anno)

        # Output should be in output_dir/outputs_temporal_segmentation/
        ts_dir = os.path.join(args.output_dir, "outputs_temporal_segmentation")
        assert os.path.isdir(ts_dir)
        # anno.json is written for the downstream sequence-accuracy comparison
        with open(os.path.join(ts_dir, "anno.json"), "r") as f:
            saved_anno = json.load(f)
        assert "v1.mp4" in saved_anno

    def test_uniform_stage_validates_chunk_length(self, tmp_path):
        from sop.sop_e2e_eval import run_uniform_stage

        args, anno = self._make_args_and_anno(tmp_path, chunk_length_sec=0)
        with patch(
            "sop.sop_e2e_eval.get_video_duration_sec",
            return_value=10.0,
        ):
            with pytest.raises(ValueError):
                run_uniform_stage(args, anno)

    def test_uniform_stage_no_videos_raises(self, tmp_path):
        from sop.sop_e2e_eval import run_uniform_stage

        # Empty video_dir
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        (tmp_path / "out").mkdir()

        args = SimpleNamespace(
            video_dir=str(empty_dir),
            video_ext="mp4",
            output_dir=str(tmp_path / "out"),
            chunk_length_sec=5.0,
        )

        with pytest.raises(FileNotFoundError):
            run_uniform_stage(args, {})


class TestGetVideoDurationSec:
    """get_video_duration_sec uses PyAV metadata (stream.duration / time_base)
    so it doesn't have to decode every frame just to learn the length."""

    def test_returns_duration_from_stream_metadata(self, monkeypatch):
        from utils import e2e_eval_utils

        class FakeTimeBase:
            def __mul__(self, other):
                # time_base * duration -> seconds
                return other * 0.04  # 25fps

        class FakeStream:
            duration = 250  # ticks
            time_base = FakeTimeBase()
            average_rate = 25.0

        class FakeContainer:
            duration = 10_000_000  # av-microseconds
            streams = SimpleNamespace(video=[FakeStream()])

            def close(self):
                pass

        def fake_open(path):
            return FakeContainer()

        with patch.object(e2e_eval_utils, "_av_open", fake_open):
            d = e2e_eval_utils.get_video_duration_sec("/fake/video.mp4")

        # 250 ticks * (1/25) = 10s
        assert d == pytest.approx(10.0)

    def test_falls_back_to_container_duration(self, monkeypatch):
        from utils import e2e_eval_utils

        class FakeStream:
            duration = None
            time_base = None
            average_rate = None

        class FakeContainer:
            # av container.duration is in AV_TIME_BASE units (1e6)
            duration = 7_500_000  # 7.5 seconds
            streams = SimpleNamespace(video=[FakeStream()])

            def close(self):
                pass

        def fake_open(path):
            return FakeContainer()

        with patch.object(e2e_eval_utils, "_av_open", fake_open):
            d = e2e_eval_utils.get_video_duration_sec("/fake/v.mp4")

        assert d == pytest.approx(7.5)


class TestReadTxt:
    """read_txt is a one-liner but it's part of the file's coverage surface."""

    def test_returns_file_contents(self, tmp_path):
        from sop.sop_e2e_eval import read_txt

        p = tmp_path / "prompt.txt"
        p.write_text("hello world")
        assert read_txt(str(p)) == "hello world"

    def test_returns_empty_string_for_empty_file(self, tmp_path):
        from sop.sop_e2e_eval import read_txt

        p = tmp_path / "empty.txt"
        p.write_text("")
        assert read_txt(str(p)) == ""


class TestMainDispatch:
    """The subprocess entry point routes Stage 1 to ddm or uniform based on
    --chunking-algorithm. We mock every heavy stage so the test stays a pure
    unit test — no torch / no vllm / no real videos.

    A `main(args)` function is extracted from `if __name__ == "__main__":`
    so the dispatch + result assembly is callable from a test."""

    def _write_inputs(self, tmp_path):
        """Lay down anno.json + actions.json on disk so main(args) can read them."""
        anno = {
            "vid1.mp4": [
                {"description": "(1) one", "start_timestamp": 0.0, "end_timestamp": 5.0},
                {"description": "(2) two", "start_timestamp": 5.0, "end_timestamp": 10.0},
            ],
        }
        anno_path = tmp_path / "anno.json"
        anno_path.write_text(json.dumps(anno))

        actions_path = tmp_path / "actions.json"
        actions_path.write_text(json.dumps({
            "actions": ["one.", "two."],
            "actions_can_be_skipped": [],
        }))

        # main() reads pred JSON via evaluate_action_sequences — we'll mock
        # that, but we still create the file path the function constructs
        # so the patch path is straightforward.
        return str(anno_path), str(actions_path)

    def _args(self, tmp_path, anno_path, actions_path, **overrides):
        defaults = dict(
            vlm_model_path="/fake/vlm",
            asset_root=str(tmp_path / "assets"),
            output_dir=str(tmp_path / "out"),
            video_dir=str(tmp_path / "videos"),
            video_ext="mp4",
            fps=8,
            temperature=0.0,
            backend="vllm",
            resolution_config=None,
            tensor_parallel_size=0,
            chunking_algorithm="ddm",
            chunk_length_sec=None,
            ddm_checkpoint_path="/fake/last.ckpt",
            ddm_resolution=224,
            ddm_frames_per_side=5,
            score_threshold=0.5,
            nms_sec=0.0,
            ddm_batch_size=8,
            frames_per_segment_hint=256,
            anno_json_path=anno_path,
            actions_json_path=actions_path,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _stage_mocks(self):
        """Common mocks for the heavy stages so main() runs end-to-end without GPU."""
        fake_temporal = {
            "vid1.mp4": {
                "boundaries": [0.0, 5.0, 10.0],
                "metric": {"F1": 0.9, "Precision": 0.9, "Recall": 0.9,
                           "True Positive": 2, "False Positive": 0, "False Negative": 0},
            },
            "avg_f1": 0.9, "avg_precision": 0.9, "avg_recall": 0.9,
        }
        fake_vlm = {"vid1.mp4": {"[0.00s-5.00s]": "(1) one", "[5.00s-10.00s]": "(2) two"}}
        fake_accuracy = {
            "overall_accuracy": 1.0,
            "per_action": {"1": {"label": "(1) one", "correct": 1, "total": 1, "accuracy": 1.0}},
        }
        fake_sequence = {
            "sequence_accuracy": 1.0, "action_accuracy": 1.0,
            "total_videos": 1, "total_videos_dist_0": 1,
            "total_actions": 2, "wrong": 0, "duplicate": 0, "missing": 0,
            "videos_with_error": [], "per_video": [],
        }
        return fake_temporal, fake_vlm, fake_accuracy, fake_sequence

    def test_uniform_path_calls_uniform_stage(self, tmp_path):
        from sop import sop_e2e_eval

        anno_path, actions_path = self._write_inputs(tmp_path)
        args = self._args(tmp_path, anno_path, actions_path,
                          chunking_algorithm="uniform", chunk_length_sec=5.0,
                          ddm_checkpoint_path=None)

        fake_temporal, fake_vlm, fake_accuracy, fake_sequence = self._stage_mocks()

        with patch.object(sop_e2e_eval, "run_uniform_stage", return_value=fake_temporal) as m_uniform, \
             patch.object(sop_e2e_eval, "run_ddm_stage") as m_ddm, \
             patch.object(sop_e2e_eval, "run_vlm_stage", return_value=fake_vlm), \
             patch.object(sop_e2e_eval, "compute_e2e_accuracy", return_value=fake_accuracy), \
             patch("utils.e2e_eval_utils.evaluate_action_sequences", return_value=fake_sequence), \
             patch.object(sop_e2e_eval, "extract_golden_boundaries",
                   return_value={"vid1.mp4": [0.0, 5.0, 10.0]}), \
             patch("utils.eval_utils.extract_mcq_data", return_value=("prompt", ["(1) one", "(2) two"])):
            sop_e2e_eval.main(args)

        m_uniform.assert_called_once()
        m_ddm.assert_not_called()

    def test_ddm_path_calls_ddm_stage(self, tmp_path):
        from sop import sop_e2e_eval

        anno_path, actions_path = self._write_inputs(tmp_path)
        args = self._args(tmp_path, anno_path, actions_path)  # default chunking_algorithm="ddm"

        fake_temporal, fake_vlm, fake_accuracy, fake_sequence = self._stage_mocks()

        with patch.object(sop_e2e_eval, "run_ddm_stage", return_value=fake_temporal) as m_ddm, \
             patch.object(sop_e2e_eval, "run_uniform_stage") as m_uniform, \
             patch.object(sop_e2e_eval, "run_vlm_stage", return_value=fake_vlm), \
             patch.object(sop_e2e_eval, "compute_e2e_accuracy", return_value=fake_accuracy), \
             patch("utils.e2e_eval_utils.evaluate_action_sequences", return_value=fake_sequence), \
             patch.object(sop_e2e_eval, "extract_golden_boundaries",
                   return_value={"vid1.mp4": [0.0, 5.0, 10.0]}), \
             patch("utils.eval_utils.extract_mcq_data", return_value=("prompt", ["(1) one", "(2) two"])):
            sop_e2e_eval.main(args)

        m_ddm.assert_called_once()
        m_uniform.assert_not_called()

    def test_uniform_without_chunk_length_raises(self, tmp_path):
        from sop import sop_e2e_eval

        anno_path, actions_path = self._write_inputs(tmp_path)
        args = self._args(tmp_path, anno_path, actions_path,
                          chunking_algorithm="uniform", chunk_length_sec=None,
                          ddm_checkpoint_path=None)

        with pytest.raises(ValueError, match="chunk-length-sec"):
            sop_e2e_eval.main(args)

    def test_ddm_without_checkpoint_path_raises(self, tmp_path):
        from sop import sop_e2e_eval

        anno_path, actions_path = self._write_inputs(tmp_path)
        args = self._args(tmp_path, anno_path, actions_path,
                          chunking_algorithm="ddm", ddm_checkpoint_path=None)

        with pytest.raises(ValueError, match="ddm-checkpoint-path"):
            sop_e2e_eval.main(args)

    def test_e2e_results_shape_persisted_to_disk(self, tmp_path):
        """The frontend keys off specific field names in e2e_results.json —
        guard the shape so a renamed key doesn't silently break the panel."""
        from sop import sop_e2e_eval

        anno_path, actions_path = self._write_inputs(tmp_path)
        args = self._args(tmp_path, anno_path, actions_path,
                          chunking_algorithm="uniform", chunk_length_sec=5.0,
                          ddm_checkpoint_path=None)

        fake_temporal, fake_vlm, fake_accuracy, fake_sequence = self._stage_mocks()

        with patch.object(sop_e2e_eval, "run_uniform_stage", return_value=fake_temporal), \
             patch.object(sop_e2e_eval, "run_vlm_stage", return_value=fake_vlm), \
             patch.object(sop_e2e_eval, "compute_e2e_accuracy", return_value=fake_accuracy), \
             patch("utils.e2e_eval_utils.evaluate_action_sequences", return_value=fake_sequence), \
             patch.object(sop_e2e_eval, "extract_golden_boundaries",
                   return_value={"vid1.mp4": [0.0, 5.0, 10.0]}), \
             patch("utils.eval_utils.extract_mcq_data", return_value=("prompt", ["(1) one", "(2) two"])):
            sop_e2e_eval.main(args)

        e2e_results_path = os.path.join(args.output_dir, "e2e_results.json")
        assert os.path.isfile(e2e_results_path)

        with open(e2e_results_path) as f:
            results = json.load(f)

        # Top-level keys the frontend reads
        assert "temporal_segmentation" in results
        assert "action_recognition" in results

        ts = results["temporal_segmentation"]
        assert ts["avg_f1"] == 0.9
        assert "per_video" in ts
        assert "vid1.mp4" in ts["per_video"]
        assert ts["per_video"]["vid1.mp4"]["f1"] == 0.9
        assert ts["per_video"]["vid1.mp4"]["boundaries"] == [0.0, 5.0, 10.0]

        ar = results["action_recognition"]
        # Chunk-level (legacy)
        assert ar["overall_accuracy"] == 1.0
        assert "per_action" in ar
        # Sequence-level (primary display)
        assert ar["sequence_accuracy"] == 1.0
        assert ar["action_accuracy"] == 1.0
        assert ar["total_videos"] == 1
        assert ar["wrong"] == 0
        assert ar["duplicate"] == 0
        assert ar["missing"] == 0
        assert ar["videos_with_error"] == []
        assert ar["per_video"] == []

    def test_accuracy_json_also_persisted(self, tmp_path):
        """accuracy.json is the stand-alone sequence-level report — also persisted."""
        from sop import sop_e2e_eval

        anno_path, actions_path = self._write_inputs(tmp_path)
        args = self._args(tmp_path, anno_path, actions_path,
                          chunking_algorithm="uniform", chunk_length_sec=5.0,
                          ddm_checkpoint_path=None)

        fake_temporal, fake_vlm, fake_accuracy, fake_sequence = self._stage_mocks()

        with patch.object(sop_e2e_eval, "run_uniform_stage", return_value=fake_temporal), \
             patch.object(sop_e2e_eval, "run_vlm_stage", return_value=fake_vlm), \
             patch.object(sop_e2e_eval, "compute_e2e_accuracy", return_value=fake_accuracy), \
             patch("utils.e2e_eval_utils.evaluate_action_sequences", return_value=fake_sequence), \
             patch.object(sop_e2e_eval, "extract_golden_boundaries",
                   return_value={"vid1.mp4": [0.0, 5.0, 10.0]}), \
             patch("utils.eval_utils.extract_mcq_data", return_value=("prompt", ["(1) one", "(2) two"])):
            sop_e2e_eval.main(args)

        accuracy_path = os.path.join(
            args.output_dir, "outputs_action_recognition", "accuracy.json"
        )
        assert os.path.isfile(accuracy_path)
        with open(accuracy_path) as f:
            payload = json.load(f)
        assert payload["sequence_accuracy"] == 1.0


class TestRcaParityShimLogs:
    """The sop-rca-plugin extracts hyperparameters from two named log files
    inside the e2e output dirs (see SKILL.md Step 2 table):
      outputs_temporal_segmentation/temporal_segmentation.log → resolution,
                                                                nms_sec,
                                                                score_threshold
      outputs_action_recognition/action_recognition_multi_gpu.log → max_frames

    Eval-ms historically only emitted a combined `log.txt` / `sop_e2e_eval_log.txt`
    at the job root with our prefixed field names (`ddm_resolution` instead
    of `resolution`). Write shim files in the right subdirectories with
    standalone-style bare field names.
    """

    def _make_args(self, tmp_path):
        ts_dir = tmp_path / "outputs_temporal_segmentation"
        ts_dir.mkdir()
        ar_dir = tmp_path / "outputs_action_recognition"
        ar_dir.mkdir()
        return SimpleNamespace(
            output_dir=str(tmp_path),
            ddm_resolution=384, ddm_batch_size=4, ddm_frames_per_side=7,
            score_threshold=0.6, nms_sec=0.5,
            frames_per_segment_hint=128,
            fps=8, temperature=0.0,
        )

    def test_dump_temporal_segmentation_args_writes_named_log(self, tmp_path):
        from sop.sop_e2e_eval import dump_temporal_segmentation_args

        args = self._make_args(tmp_path)
        dump_temporal_segmentation_args(args, str(tmp_path / "outputs_temporal_segmentation"))

        log_path = tmp_path / "outputs_temporal_segmentation" / "temporal_segmentation.log"
        assert log_path.is_file(), \
            "must write outputs_temporal_segmentation/temporal_segmentation.log"
        content = log_path.read_text()
        # standalone field names (no ddm_ prefix)
        assert "Args: Namespace(" in content
        assert "resolution=384" in content
        assert "score_threshold=0.6" in content
        assert "nms_sec=0.5" in content
        assert "frames_per_side=7" in content
        assert "batch_size=4" in content
        # must NOT use our internal prefixed field names
        assert "ddm_resolution" not in content
        assert "ddm_batch_size" not in content
        assert "ddm_frames_per_side" not in content

    def test_dump_action_recognition_args_writes_named_log(self, tmp_path):
        from sop.sop_e2e_eval import dump_action_recognition_args

        args = self._make_args(tmp_path)
        resolution_config = {"max_frames": 50, "total_pixels": 16572416}
        dump_action_recognition_args(
            args, str(tmp_path / "outputs_action_recognition"), resolution_config
        )

        log_path = tmp_path / "outputs_action_recognition" / "action_recognition_multi_gpu.log"
        assert log_path.is_file(), \
            "must write outputs_action_recognition/action_recognition_multi_gpu.log"
        content = log_path.read_text()
        assert "Args: Namespace(" in content
        assert "max_frames=50" in content
        assert "total_pixels=16572416" in content
        # optional fields render as None when missing
        assert "resized_height=None" in content
        assert "resized_width=None" in content

    def test_dump_action_recognition_args_includes_pixel_overrides(self, tmp_path):
        from sop.sop_e2e_eval import dump_action_recognition_args

        args = self._make_args(tmp_path)
        # When the user passes typed overrides, they must surface in the log
        # so the RCA skill sees the actual eval config.
        resolution_config = {
            "max_frames": 30,
            "total_pixels": 16572416,
            "resized_height": 567,
            "resized_width": 1008,
        }
        dump_action_recognition_args(
            args, str(tmp_path / "outputs_action_recognition"), resolution_config
        )

        log_path = tmp_path / "outputs_action_recognition" / "action_recognition_multi_gpu.log"
        content = log_path.read_text()
        assert "max_frames=30" in content
        assert "resized_height=567" in content
        assert "resized_width=1008" in content
