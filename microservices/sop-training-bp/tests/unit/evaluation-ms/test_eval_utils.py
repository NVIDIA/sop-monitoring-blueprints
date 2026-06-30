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

"""Unit tests for evaluation-related models and utilities."""
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MS_PATH = str(PROJECT_ROOT / "microservices" / "evaluation-ms")
if MS_PATH not in sys.path:
    sys.path.insert(0, MS_PATH)


class TestEvaluationJobModel:
    def test_evaluation_job_has_required_columns(self):
        from validation.postgres_validation import EvaluationJob
        required = [
            "id", "training_job_id", "val_dataset_id", "checkpoint_step",
            "status", "overall_accuracy", "results_json",
            "fps", "temperature", "backend", "created_at", "updated_at",
        ]
        for col in required:
            assert hasattr(EvaluationJob, col), f"Missing column: {col}"

    def test_evaluation_job_to_dict(self):
        from validation.postgres_validation import EvaluationJob, TrainingStatusEnum
        from datetime import datetime
        job = EvaluationJob(
            id="eval-1", training_job_id="train-1",
            val_dataset_id="val-ds-1", checkpoint_step=100,
            status=TrainingStatusEnum.completed, overall_accuracy=0.87,
            results_json={"per_action": {}}, fps=8, temperature=0.0,
            backend="vllm", created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )
        d = job.to_dict()
        assert d["id"] == "eval-1"
        assert d["overall_accuracy"] == pytest.approx(0.87)
        assert d["checkpoint_step"] == 100
        assert d["backend"] == "vllm"
        expected_keys = {
            "id", "training_job_id", "val_dataset_id", "checkpoint_step",
            "status", "overall_accuracy", "results_json", "fps",
            "temperature", "backend", "created_at", "updated_at",
        }
        assert set(d.keys()) == expected_keys


class TestEvalPydanticModels:
    def test_evaluation_request_defaults(self):
        from validation.request_validation import EvaluationRequest
        req = EvaluationRequest(training_job_id="tj-1", val_dataset_id="vd-1")
        assert req.fps == 8
        assert req.temperature == pytest.approx(0.0)
        assert req.backend == "vllm"
        assert req.checkpoint_step is None

    def test_evaluation_request_custom(self):
        from validation.request_validation import EvaluationRequest
        req = EvaluationRequest(
            training_job_id="tj-1", val_dataset_id="vd-1",
            fps=4, temperature=0.5, backend="transformers", checkpoint_step=200
        )
        assert req.fps == 4
        assert req.checkpoint_step == 200

    def test_evaluation_status_optional_fields(self):
        from validation.request_validation import EvaluationStatus
        from datetime import datetime
        status = EvaluationStatus(
            eval_job_id="ej-1", training_job_id="tj-1",
            val_dataset_id="vd-1", status="running",
            created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 1),
        )
        assert status.overall_accuracy is None
        assert status.checkpoint_step is None


class TestPostgresDBEvalMethods:
    @pytest.mark.asyncio
    async def test_insert_eval_job_calls_session_add_and_commit(self):
        from unittest.mock import AsyncMock, patch, MagicMock
        from datetime import datetime

        with patch("components.postgres_db.create_async_engine"), \
             patch("components.postgres_db.sessionmaker") as mock_sm:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_sm.return_value.return_value = mock_session

            from components.postgres_db import PostgresDB
            db = PostgresDB()
            await db.insert_evaluation_job(
                id="ej-1", training_job_id="tj-1", val_dataset_id="vd-1",
                status="queued", created_at=datetime.now(), updated_at=datetime.now(),
            )
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_eval_job_calls_execute_and_commit(self):
        from unittest.mock import AsyncMock, patch
        from datetime import datetime

        with patch("components.postgres_db.create_async_engine"), \
             patch("components.postgres_db.sessionmaker") as mock_sm:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_sm.return_value.return_value = mock_session

            from components.postgres_db import PostgresDB
            db = PostgresDB()
            await db.update_evaluation_job("ej-1", status="completed", overall_accuracy=0.9)
            mock_session.execute.assert_called_once()
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_eval_job_returns_scalar(self):
        from unittest.mock import AsyncMock, patch, MagicMock

        with patch("components.postgres_db.create_async_engine"), \
             patch("components.postgres_db.sessionmaker") as mock_sm:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            fake_result = MagicMock()
            fake_result.scalar_one_or_none.return_value = MagicMock(id="ej-1")
            mock_session.execute = AsyncMock(return_value=fake_result)
            mock_sm.return_value.return_value = mock_session

            from components.postgres_db import PostgresDB
            db = PostgresDB()
            result = await db.get_evaluation_job("ej-1")
            assert result.id == "ej-1"

    @pytest.mark.asyncio
    async def test_list_eval_jobs_returns_all(self):
        from unittest.mock import AsyncMock, patch, MagicMock

        with patch("components.postgres_db.create_async_engine"), \
             patch("components.postgres_db.sessionmaker") as mock_sm:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            fake_scalars = MagicMock()
            fake_scalars.all.return_value = [MagicMock(id="ej-1"), MagicMock(id="ej-2")]
            fake_result = MagicMock()
            fake_result.scalars.return_value = fake_scalars
            mock_session.execute = AsyncMock(return_value=fake_result)
            mock_sm.return_value.return_value = mock_session

            from components.postgres_db import PostgresDB
            db = PostgresDB()
            jobs = await db.list_evaluation_jobs()
            assert len(jobs) == 2


class TestResolveCheckpointPath:
    def test_finds_latest_step(self, tmp_path):
        from utils.eval_utils import resolve_checkpoint_path
        for step in [100, 200, 500]:
            (tmp_path / f"step_{step}").mkdir()
        path, step = resolve_checkpoint_path(str(tmp_path.parent), tmp_path.name)
        assert step == 500
        assert "step_500" in path

    def test_explicit_step_exists(self, tmp_path):
        from utils.eval_utils import resolve_checkpoint_path
        (tmp_path / "step_200").mkdir()
        path, step = resolve_checkpoint_path(str(tmp_path.parent), tmp_path.name, step=200)
        assert step == 200
        assert "step_200" in path

    def test_explicit_step_missing_raises(self, tmp_path):
        from utils.eval_utils import resolve_checkpoint_path
        (tmp_path / "step_100").mkdir()
        with pytest.raises(FileNotFoundError, match="step 999"):
            resolve_checkpoint_path(str(tmp_path.parent), tmp_path.name, step=999)

    def test_no_checkpoints_raises(self, tmp_path):
        from utils.eval_utils import resolve_checkpoint_path
        with pytest.raises(FileNotFoundError):
            resolve_checkpoint_path(str(tmp_path.parent), tmp_path.name)

    def test_finds_nested_cosmos_rl_v039_layout(self, tmp_path):
        """cosmos-rl v0.3.9 writes to <job>/<timestamp>/safetensors/step_<N>/."""
        from utils.eval_utils import resolve_checkpoint_path
        nested = tmp_path / "20260424081531" / "safetensors" / "step_164"
        nested.mkdir(parents=True)
        path, step = resolve_checkpoint_path(str(tmp_path.parent), tmp_path.name)
        assert step == 164
        assert "step_164" in path
        assert "safetensors" in path

    def test_picks_latest_mtime_when_step_number_collides(self, tmp_path):
        """Across multiple runs the same step number may exist twice; pick the newer."""
        import os
        import time
        from utils.eval_utils import resolve_checkpoint_path
        old_run = tmp_path / "20260101000000" / "safetensors" / "step_100"
        old_run.mkdir(parents=True)
        # Force a small mtime gap so the 'newer' comparison is deterministic.
        old_mtime = time.time() - 1000
        os.utime(old_run, (old_mtime, old_mtime))
        new_run = tmp_path / "20260401000000" / "safetensors" / "step_100"
        new_run.mkdir(parents=True)
        path, step = resolve_checkpoint_path(str(tmp_path.parent), tmp_path.name, step=100)
        assert step == 100
        assert "20260401000000" in path


class TestBuildVllmVideoMmData:
    """
    The helper's output goes directly into vLLM 0.11.0's MultiModalDataParser
    under mm_data["video"] (singular). Each item must be a (video_array,
    metadata_dict) tuple, and each metadata dict must contain only keys that
    transformers.video_utils.VideoMetadata accepts (so qwen3_vl.py's
    `VideoMetadata(**{k: m[k] for k in m if k != "do_sample_frames"})` call
    doesn't TypeError on an unknown kwarg).
    """

    # Fields VideoMetadata accepts; do_sample_frames is filtered out before
    # the VideoMetadata constructor call, so we keep it as a permitted extra.
    VIDEO_METADATA_FIELDS = {
        "total_num_frames", "fps", "width", "height",
        "duration", "video_backend", "frames_indices",
    }
    PERMITTED_KEYS = VIDEO_METADATA_FIELDS | {"do_sample_frames"}

    def _fake_video(self, num_frames):
        # Mimic a video tensor: shape[0] is the temporal axis.
        class _V:
            shape = (num_frames, 224, 224, 3)
        return _V()

    def test_returns_none_for_empty_video_inputs(self):
        from utils.eval_utils import build_vllm_video_mm_data
        assert build_vllm_video_mm_data(None, {}, 8) is None
        assert build_vllm_video_mm_data([], {}, 8) is None

    def test_pairs_each_video_with_metadata_tuple(self):
        from utils.eval_utils import build_vllm_video_mm_data
        videos = [self._fake_video(120), self._fake_video(64)]
        result = build_vllm_video_mm_data(videos, {"fps": [8.0, 4.0]}, default_fps=2)
        assert isinstance(result, list)
        assert len(result) == 2
        for entry in result:
            assert isinstance(entry, tuple) and len(entry) == 2

    def test_metadata_keys_subset_of_videometadata_fields(self):
        from utils.eval_utils import build_vllm_video_mm_data
        videos = [self._fake_video(80)]
        (_, metadata) = build_vllm_video_mm_data(videos, {"fps": [8.0]}, 8)[0]
        unknown = set(metadata) - self.PERMITTED_KEYS
        assert not unknown, (
            f"metadata has keys VideoMetadata does not accept: {unknown}. "
            f"qwen3_vl.py would TypeError on these."
        )
        # total_num_frames is mandatory in VideoMetadata
        assert "total_num_frames" in metadata
        assert metadata["total_num_frames"] == 80
        # do_sample_frames must be False — qwen_vl_utils has already sampled.
        assert metadata.get("do_sample_frames") is False

    def test_per_video_fps_unpacked_from_list(self):
        from utils.eval_utils import build_vllm_video_mm_data
        videos = [self._fake_video(60), self._fake_video(90)]
        result = build_vllm_video_mm_data(videos, {"fps": [4.0, 6.0]}, default_fps=2)
        assert result[0][1]["fps"] == 4.0
        assert result[1][1]["fps"] == 6.0

    def test_falls_back_to_default_fps_when_video_kwargs_missing(self):
        from utils.eval_utils import build_vllm_video_mm_data
        videos = [self._fake_video(50)]
        result = build_vllm_video_mm_data(videos, None, default_fps=8)
        assert result[0][1]["fps"] == 8.0

    def test_duration_computed_from_total_frames_and_fps(self):
        from utils.eval_utils import build_vllm_video_mm_data
        videos = [self._fake_video(96)]
        result = build_vllm_video_mm_data(videos, {"fps": [8.0]}, 8)
        assert result[0][1]["duration"] == 12.0  # 96 / 8

    def test_frames_indices_match_total_num_frames(self):
        from utils.eval_utils import build_vllm_video_mm_data
        videos = [self._fake_video(7)]
        result = build_vllm_video_mm_data(videos, {"fps": [8.0]}, 8)
        assert result[0][1]["frames_indices"] == [0, 1, 2, 3, 4, 5, 6]


class TestExtractMcqData:
    def test_generates_prompt_and_choices_from_actions_json(self, tmp_path):
        import json
        from utils.eval_utils import extract_mcq_data
        actions = {"actions": ["do step one.", "do step two.", "doing none of the above."]}
        actions_path = tmp_path / "actions.json"
        actions_path.write_text(json.dumps(actions))
        prompt, choices = extract_mcq_data(str(actions_path))
        assert "There are 3 possible steps" in prompt
        assert "operator" in prompt
        assert choices[0] == "(1) do step one"
        assert choices[1] == "(2) do step two"
        assert choices[2] == "(3) doing none of the above"

    def test_missing_actions_json_raises(self, tmp_path):
        from utils.eval_utils import extract_mcq_data
        with pytest.raises(FileNotFoundError):
            extract_mcq_data(str(tmp_path / "nonexistent.json"))

    def test_empty_actions_raises(self, tmp_path):
        import json
        from utils.eval_utils import extract_mcq_data
        actions_path = tmp_path / "actions.json"
        actions_path.write_text(json.dumps({"actions": []}))
        with pytest.raises(ValueError):
            extract_mcq_data(str(actions_path))


class TestPrepareEvalAssets:
    def test_creates_vlm_prompts_txt(self, tmp_path, monkeypatch):
        from utils import eval_utils
        monkeypatch.setattr(eval_utils.const, "RESULTS_ROOT", str(tmp_path))
        from utils.eval_utils import prepare_eval_assets
        asset_dir = prepare_eval_assets("eval-job-1", "There are 3 steps.\n(1) step one\n(2) step two")
        assert os.path.exists(asset_dir)
        prompts_path = os.path.join(asset_dir, "vlm_prompts.txt")
        assert os.path.exists(prompts_path)
        content = open(prompts_path).read()
        assert "There are 3 steps" in content

import os

class TestParseEvalResults:
    def test_perfect_accuracy(self):
        from utils.eval_utils import parse_eval_results
        choices = ["(1) do step one", "(2) do step two", "(3) doing none of the above"]
        inference = {"video1": [[1, "(1) do step one"], [2, "(2) do step two"]]}
        result = parse_eval_results(inference, choices)
        assert result["overall_accuracy"] == pytest.approx(1.0)
        assert result["per_action"]["1"]["correct"] == 1
        assert result["per_action"]["1"]["total"] == 1
        assert result["per_action"]["1"]["accuracy"] == pytest.approx(1.0)

    def test_zero_accuracy(self):
        from utils.eval_utils import parse_eval_results
        choices = ["(1) do step one", "(2) do step two"]
        inference = {"video1": [[1, "(2) do step two"], [2, "(1) do step one"]]}
        result = parse_eval_results(inference, choices)
        assert result["overall_accuracy"] == pytest.approx(0.0)

    def test_partial_accuracy_across_videos(self):
        from utils.eval_utils import parse_eval_results
        choices = ["(1) do step one", "(2) do step two"]
        inference = {
            "video1": [[1, "(1) do step one"]],
            "video2": [[1, "(2) wrong answer"]],
        }
        result = parse_eval_results(inference, choices)
        assert result["overall_accuracy"] == pytest.approx(0.5)
        assert result["per_action"]["1"]["correct"] == 1
        assert result["per_action"]["1"]["total"] == 2
        assert result["per_action"]["1"]["accuracy"] == pytest.approx(0.5)

    def test_per_action_label_from_choices(self):
        from utils.eval_utils import parse_eval_results
        choices = ["(1) install cable", "(2) install board"]
        inference = {"v": [[1, "(1) install cable"], [2, "(2) install board"]]}
        result = parse_eval_results(inference, choices)
        assert result["per_action"]["1"]["label"] == "(1) install cable"
        assert result["per_action"]["2"]["label"] == "(2) install board"

    def test_empty_inference_results(self):
        from utils.eval_utils import parse_eval_results
        result = parse_eval_results({}, ["(1) step one"])
        assert result["overall_accuracy"] == pytest.approx(0.0)
        assert result["per_action"] == {}

    def test_accepts_three_tuple_with_chunk_path(self):
        """sop_eval.py now emits 3-tuples [action, response, chunk_path] for
        RCA-parser parity. The downstream consumer must accept that shape too;
        otherwise the run_evaluation background task crashes with
        `ValueError: too many values to unpack (expected 2)`."""
        from utils.eval_utils import parse_eval_results
        choices = ["(1) do step one", "(2) do step two"]
        inference = {
            "video1": [
                [1, "(1) do step one", "/path/to/01_chunk.mp4"],
                [2, "(2) do step two", "/path/to/02_chunk.mp4"],
            ],
        }
        result = parse_eval_results(inference, choices)
        assert result["overall_accuracy"] == pytest.approx(1.0)
        assert result["per_action"]["1"]["correct"] == 1
        assert result["per_action"]["2"]["correct"] == 1

    def test_accepts_legacy_two_tuple(self):
        """Older inference_results.json files in the DB still use 2-tuples.
        Reading historical results via the API must keep working."""
        from utils.eval_utils import parse_eval_results
        choices = ["(1) do step one", "(2) do step two"]
        inference = {"video1": [[1, "(1) do step one"], [2, "(2) do step two"]]}
        result = parse_eval_results(inference, choices)
        assert result["overall_accuracy"] == pytest.approx(1.0)

    # ----- two-operator (concurrent action) grading -----
    def test_two_op_correct_full_set_match(self):
        """A two-op chunk with gt=[1,4] is correct iff pred mentions both
        (1) and (4). Both gt-id buckets get +1 total and +1 correct."""
        from utils.eval_utils import parse_eval_results
        choices = ["(1) step one", "(2) step two", "(3) step three", "(4) step four"]
        inference = {"v": [[[1, 4], "(1) step one (4) step four", "/x/01-04.mp4"]]}
        result = parse_eval_results(inference, choices)
        assert result["overall_accuracy"] == pytest.approx(1.0)
        assert result["per_action"]["1"]["total"] == 1
        assert result["per_action"]["1"]["correct"] == 1
        assert result["per_action"]["4"]["total"] == 1
        assert result["per_action"]["4"]["correct"] == 1

    def test_two_op_wrong_missing_one_id(self):
        """gt=[1,4] but pred only mentions (1): chunk is wrong; both buckets
        get +1 total, +0 correct."""
        from utils.eval_utils import parse_eval_results
        choices = ["(1) step one", "(2) step two", "(3) step three", "(4) step four"]
        inference = {"v": [[[1, 4], "(1) step one", "/x/01-04.mp4"]]}
        result = parse_eval_results(inference, choices)
        assert result["overall_accuracy"] == pytest.approx(0.0)
        assert result["per_action"]["1"]["total"] == 1
        assert result["per_action"]["1"]["correct"] == 0
        assert result["per_action"]["4"]["total"] == 1
        assert result["per_action"]["4"]["correct"] == 0

    def test_two_op_wrong_extra_id(self):
        """gt=[1,4], pred mentions (1)(4)(2): extra (2) breaks set equality."""
        from utils.eval_utils import parse_eval_results
        choices = ["(1) one", "(2) two", "(3) three", "(4) four"]
        inference = {"v": [[[1, 4], "(1) one (4) four (2) two", "/x/01-04.mp4"]]}
        result = parse_eval_results(inference, choices)
        assert result["overall_accuracy"] == pytest.approx(0.0)

    def test_two_op_order_invariant(self):
        """Order of (N) tokens in the response doesn't matter — set equality."""
        from utils.eval_utils import parse_eval_results
        choices = ["(1) one", "(2) two", "(3) three", "(4) four"]
        inference = {"v": [[[1, 4], "(4) four (1) one", "/x/01-04.mp4"]]}
        result = parse_eval_results(inference, choices)
        assert result["overall_accuracy"] == pytest.approx(1.0)

    def test_mixed_single_and_two_op(self):
        """Mixed dataset: single-op uses verify_pred; two-op uses set match.
        Per-action totals correctly attribute concurrent chunks to both buckets."""
        from utils.eval_utils import parse_eval_results
        choices = ["(1) one", "(2) two", "(3) three", "(4) four"]
        inference = {"v": [
            [1, "(1) one", "/x/01.mp4"],                      # single-op correct
            [[1, 4], "(1) one (4) four", "/x/01-04.mp4"],     # two-op correct
            [[1, 4], "(1) one", "/x/01-04_b.mp4"],            # two-op wrong (missing 4)
            [[2], "(3) three", "/x/02_x.mp4"],                # single-id-list, wrong
        ]}
        result = parse_eval_results(inference, choices)
        # 2 of 4 chunks correct
        assert result["overall_accuracy"] == pytest.approx(0.5)
        # bucket 1: 3 chunks contributed (single-op, two-op-correct, two-op-wrong); 2 correct
        assert result["per_action"]["1"]["total"] == 3
        assert result["per_action"]["1"]["correct"] == 2
        # bucket 4: 2 chunks (both two-op); 1 correct
        assert result["per_action"]["4"]["total"] == 2
        assert result["per_action"]["4"]["correct"] == 1
        # bucket 2: 1 chunk (single-id list); 0 correct
        assert result["per_action"]["2"]["total"] == 1
        assert result["per_action"]["2"]["correct"] == 0

    def test_two_op_list_with_single_element_falls_back_to_verify_pred(self):
        """A 1-element list (e.g. after dedup) grades exactly like a scalar
        int via verify_pred (not set-match)."""
        from utils.eval_utils import parse_eval_results
        choices = ["(1) one"]
        inference = {"v": [[[1], "(1) one with extra commentary", "/x/01.mp4"]]}
        # verify_pred matches by (N) prefix, so the trailing text is allowed
        result = parse_eval_results(inference, choices)
        assert result["overall_accuracy"] == pytest.approx(1.0)


class TestVerifyPred:
    def test_exact_match(self):
        from utils.eval_utils import verify_pred
        assert verify_pred("(1) do step one", "(1) do step one") is True

    def test_action_number_match(self):
        from utils.eval_utils import verify_pred
        assert verify_pred("The answer is (1) do step one", "(1) do step one") is True

    def test_case_insensitive_text_match(self):
        from utils.eval_utils import verify_pred
        assert verify_pred("DO STEP ONE", "(1) do step one") is True

    def test_answer_tag_match(self):
        from utils.eval_utils import verify_pred
        assert verify_pred("<answer>(1) do step one</answer>", "(1) do step one") is True

    def test_wrong_answer(self):
        from utils.eval_utils import verify_pred
        assert verify_pred("(2) do step two", "(1) do step one") is False

    def test_answer_tag_with_json_content(self):
        """Answer tag wrapping a JSON object — extracts .answer key."""
        from utils.eval_utils import verify_pred
        import json
        answer_json = json.dumps({"answer": "(1) do step one"})
        pred = f"<answer>{answer_json}</answer>"
        assert verify_pred(pred, "(1) do step one") is True

    def test_letter_answer_in_tag_matches_action_number(self):
        """<answer>A</answer> should map to action (1)."""
        from utils.eval_utils import verify_pred
        assert verify_pred("<answer>A</answer>", "(1) do step one") is True

    def test_letter_answer_b_matches_action_2(self):
        """<answer>B</answer> should map to action (2)."""
        from utils.eval_utils import verify_pred
        assert verify_pred("<answer>B</answer>", "(2) do step two") is True

    def test_letter_answer_wrong(self):
        """<answer>B</answer> should NOT match action (1)."""
        from utils.eval_utils import verify_pred
        assert verify_pred("<answer>B</answer>", "(1) do step one") is False


class TestResolveCheckpointPathEdgeCases:
    def test_malformed_dir_names_skipped_and_valid_one_used(self, tmp_path):
        """Dirs that can't be parsed as step_N are silently skipped; valid ones still work."""
        from utils.eval_utils import resolve_checkpoint_path
        (tmp_path / "step_abc").mkdir()   # unparseable — triggers IndexError/ValueError
        (tmp_path / "step_300").mkdir()   # valid
        path, step = resolve_checkpoint_path(str(tmp_path.parent), tmp_path.name)
        assert step == 300
        assert "step_300" in path

    def test_all_malformed_dirs_raises(self, tmp_path):
        """When all step dirs are malformed, step_map is empty → FileNotFoundError."""
        from utils.eval_utils import resolve_checkpoint_path
        (tmp_path / "step_xyz").mkdir()
        (tmp_path / "step_foo").mkdir()
        with pytest.raises(FileNotFoundError, match="Could not parse step numbers"):
            resolve_checkpoint_path(str(tmp_path.parent), tmp_path.name)


class TestEvaluateActionSequences:
    """
    Sequence-level accuracy ported from
    sop-monitoring-blueprints/.../print_out_pred_and_golden_action.py.
    These tests verify the Levenshtein-based error classification matches
    the reference: each non-matching position is Wrong (substitution),
    Duplicate (extra in pred), or Missing (omitted from pred).
    """

    @staticmethod
    def _write_anno(tmp_path, video_to_actions):
        """Write an anno.json with golden boundaries in 'description: (N) ...' format."""
        import json
        d = {}
        for v, action_nums in video_to_actions.items():
            d[v] = []
            for n in action_nums:
                d[v].append({
                    "description": f"({n}) action label {n}",
                    "start_timestamp": 0.0,
                    "end_timestamp": 1.0,
                })
        path = tmp_path / "anno.json"
        path.write_text(json.dumps(d))
        return str(path)

    @staticmethod
    def _write_pred(tmp_path, video_to_chunks):
        """
        Write a video_name_to_output_text.json. video_to_chunks maps each
        video name to a list of (chunk_key, response_text) pairs.
        """
        import json
        d = {v: dict(chunks) for v, chunks in video_to_chunks.items()}
        path = tmp_path / "pred.json"
        path.write_text(json.dumps(d))
        return str(path)

    def test_perfect_sequence_match_yields_zero_errors(self, tmp_path):
        from utils.e2e_eval_utils import evaluate_action_sequences
        anno = self._write_anno(tmp_path, {"v1.mp4": [1, 2, 3]})
        pred = self._write_pred(tmp_path, {"v1.mp4": [
            ("[0.0s-1.0s]", "(1) action one"),
            ("[1.0s-2.0s]", "(2) action two"),
            ("[2.0s-3.0s]", "(3) action three"),
        ]})
        r = evaluate_action_sequences(anno, pred)
        assert r["sequence_accuracy"] == 1.0
        assert r["action_accuracy"] == 1.0
        assert r["wrong"] == 0 and r["duplicate"] == 0 and r["missing"] == 0
        assert r["videos_with_error"] == []
        assert r["per_video"][0]["edit_distance"] == 0

    def test_substitution_counts_as_wrong(self, tmp_path):
        from utils.e2e_eval_utils import evaluate_action_sequences
        anno = self._write_anno(tmp_path, {"v1.mp4": [1, 2, 3]})
        pred = self._write_pred(tmp_path, {"v1.mp4": [
            ("[0.0s-1.0s]", "(1) action one"),
            ("[1.0s-2.0s]", "(5) wrong action"),  # should be (2)
            ("[2.0s-3.0s]", "(3) action three"),
        ]})
        r = evaluate_action_sequences(anno, pred)
        assert r["wrong"] == 1
        assert r["duplicate"] == 0
        assert r["missing"] == 0
        # action_accuracy = (3 - 1 - 0 - 0) / 3 = 0.6667
        assert abs(r["action_accuracy"] - 2 / 3) < 1e-9
        assert r["sequence_accuracy"] == 0.0  # not perfect

    def test_extra_pred_counts_as_duplicate(self, tmp_path):
        from utils.e2e_eval_utils import evaluate_action_sequences
        anno = self._write_anno(tmp_path, {"v1.mp4": [1, 2]})
        pred = self._write_pred(tmp_path, {"v1.mp4": [
            ("[0.0s-1.0s]", "(1) action one"),
            ("[1.0s-2.0s]", "(2) action two (3) action three"),  # extra (3)
        ]})
        r = evaluate_action_sequences(anno, pred)
        assert r["duplicate"] == 1
        assert r["wrong"] == 0
        assert r["missing"] == 0

    def test_missing_pred_counts_as_missing(self, tmp_path):
        from utils.e2e_eval_utils import evaluate_action_sequences
        anno = self._write_anno(tmp_path, {"v1.mp4": [1, 2, 3]})
        pred = self._write_pred(tmp_path, {"v1.mp4": [
            ("[0.0s-1.0s]", "(1) action one"),
            ("[2.0s-3.0s]", "(3) action three"),  # action 2 not predicted
        ]})
        r = evaluate_action_sequences(anno, pred)
        assert r["missing"] == 1
        assert r["wrong"] == 0
        assert r["duplicate"] == 0

    def test_consecutive_duplicates_in_pred_are_collapsed(self, tmp_path):
        """remove_continuous_rep — if the model says '(1)' twice in a row, count once."""
        from utils.e2e_eval_utils import evaluate_action_sequences
        anno = self._write_anno(tmp_path, {"v1.mp4": [1, 2]})
        pred = self._write_pred(tmp_path, {"v1.mp4": [
            ("[0.0s-1.0s]", "(1) action one"),
            ("[1.0s-2.0s]", "(1) action one again"),  # adjacent duplicate
            ("[2.0s-3.0s]", "(2) action two"),
        ]})
        r = evaluate_action_sequences(anno, pred)
        assert r["wrong"] == 0 and r["duplicate"] == 0 and r["missing"] == 0
        assert r["sequence_accuracy"] == 1.0

    def test_skippable_actions_excluded(self, tmp_path):
        """Action numbers under 'actions_can_be_skipped' don't count as missing."""
        import json
        from utils.e2e_eval_utils import evaluate_action_sequences

        anno = self._write_anno(tmp_path, {"v1.mp4": [1, 10, 2]})  # 10 is skippable
        pred = self._write_pred(tmp_path, {"v1.mp4": [
            ("[0.0s-1.0s]", "(1) action one"),
            ("[1.0s-2.0s]", "(2) action two"),  # skipped 10 — should be fine
        ]})
        actions_path = tmp_path / "actions.json"
        actions_path.write_text(json.dumps({
            "actions": [],
            "actions_can_be_skipped": ["(10) skippable label"],
        }))
        r = evaluate_action_sequences(anno, pred, str(actions_path))
        assert r["sequence_accuracy"] == 1.0
        assert r["missing"] == 0

    def test_per_video_table_records_diff_steps(self, tmp_path):
        from utils.e2e_eval_utils import evaluate_action_sequences
        anno = self._write_anno(tmp_path, {"v1.mp4": [1, 2, 3]})
        pred = self._write_pred(tmp_path, {"v1.mp4": [
            ("[0.0s-1.0s]", "(1) ok"),
            ("[1.0s-2.0s]", "(9) wrong"),
            ("[2.0s-3.0s]", "(3) ok"),
        ]})
        r = evaluate_action_sequences(anno, pred)
        v = r["per_video"][0]
        assert v["video"] == "v1.mp4"
        assert v["golden"] == [1, 2, 3]
        assert v["predicted"] == [1, 9, 3]
        assert v["edit_distance"] == 1
        assert any("Wrong" in s for s in v["steps"])

    def test_chunk_keys_sorted_by_start_time_not_lexicographic(self, tmp_path):
        """Chunk keys like [9.5s-12s] should sort before [12s-15s] (numerically)."""
        from utils.e2e_eval_utils import evaluate_action_sequences
        anno = self._write_anno(tmp_path, {"v1.mp4": [1, 2]})
        # Lexicographic order would put [12s...] before [9.5s...] — wrong.
        # Numeric (by start time) puts [9.5s...] first — what we want.
        pred = self._write_pred(tmp_path, {"v1.mp4": [
            ("[12.0s-15.0s]", "(2) second"),
            ("[9.5s-12.0s]", "(1) first"),
        ]})
        r = evaluate_action_sequences(anno, pred)
        assert r["per_video"][0]["predicted"] == [1, 2]
        assert r["sequence_accuracy"] == 1.0

    def test_action_accuracy_floored_at_zero_when_many_extras(self, tmp_path):
        """When predicted has many extras, raw (total - w - d - m) / total
        can go negative. We clamp at 0 so the frontend's percent formatter
        gets a sensible value."""
        from utils.e2e_eval_utils import evaluate_action_sequences
        # 1 golden action vs 5 predicted → 4 duplicates, raw = -3.0 → clamps to 0.
        anno = self._write_anno(tmp_path, {"v1.mp4": [1]})
        pred = self._write_pred(tmp_path, {"v1.mp4": [
            ("[0.0s-1.0s]", "(1) ok (2) extra (3) extra (4) extra (5) extra"),
        ]})
        r = evaluate_action_sequences(anno, pred)
        assert r["duplicate"] >= 4
        assert r["action_accuracy"] == 0.0

    # ----- two-operator (concurrent action) E2E flow -----
    @staticmethod
    def _write_two_op_anno(tmp_path, video_to_chunks):
        """video_to_chunks: {video: [(ids_list, descs_list_or_None), ...]} per chunk."""
        import json
        d = {}
        for v, chunks in video_to_chunks.items():
            d[v] = []
            for ids, descs in chunks:
                entry = {"start_timestamp": 0.0, "end_timestamp": 1.0}
                if len(ids) == 1:
                    entry["action"] = ids[0]
                    entry["description"] = descs[0] if descs else f"({ids[0]}) label {ids[0]}"
                else:
                    entry["actions"] = ids
                    entry["descriptions"] = descs or [f"({i}) label {i}" for i in ids]
                    entry["is_concurrent"] = True
                d[v].append(entry)
        path = tmp_path / "anno_two_op.json"
        path.write_text(json.dumps(d))
        return str(path)

    def test_two_op_golden_reads_plural_actions(self, tmp_path):
        """_get_golden_actions extracts ids from the plural `actions` array,
        appending them sorted-ascending per chunk."""
        from utils.e2e_eval_utils import _get_golden_actions
        anno = self._write_two_op_anno(tmp_path, {"v1.mp4": [
            ([1], None),
            ([4, 10], None),   # concurrent
            ([2], None),
        ]})
        g = _get_golden_actions(anno)
        assert g["v1.mp4"] == [1, 4, 10, 2]

    def test_two_op_golden_falls_back_to_descriptions(self, tmp_path):
        """When `actions` is absent, plural `descriptions` (N) prefixes are used."""
        import json
        anno = tmp_path / "anno.json"
        anno.write_text(json.dumps({"v1.mp4": [{
            "descriptions": ["(7) lower idle", "(1) upper install"],
            "is_concurrent": True,
            "start_timestamp": 0.0, "end_timestamp": 1.0,
        }]}))
        from utils.e2e_eval_utils import _get_golden_actions
        g = _get_golden_actions(str(anno))
        assert g["v1.mp4"] == [1, 7]  # sorted ascending

    def test_two_op_perfect_concurrent_match(self, tmp_path):
        """Concurrent chunk: VLM emits both (N) tokens; per-chunk sort
        makes order irrelevant. seq_acc = 1.0."""
        from utils.e2e_eval_utils import evaluate_action_sequences
        anno = self._write_two_op_anno(tmp_path, {"v1.mp4": [
            ([1], None),
            ([4, 10], None),
            ([2], None),
        ]})
        pred = self._write_pred(tmp_path, {"v1.mp4": [
            ("[0.0s-1.0s]", "(1) upper install"),
            ("[1.0s-2.0s]", "(10) idle (4) lower install"),  # reversed order
            ("[2.0s-3.0s]", "(2) upper secured"),
        ]})
        r = evaluate_action_sequences(anno, pred)
        assert r["per_video"][0]["golden"] == [1, 4, 10, 2]
        assert r["per_video"][0]["predicted"] == [1, 4, 10, 2]
        assert r["sequence_accuracy"] == 1.0
        assert r["wrong"] == 0 and r["duplicate"] == 0 and r["missing"] == 0

    def test_two_op_missing_one_concurrent_id_counts_as_missing(self, tmp_path):
        """If VLM emits only one of the two concurrent ids, the other is
        flagged as missing by the edit-distance backtrace."""
        from utils.e2e_eval_utils import evaluate_action_sequences
        anno = self._write_two_op_anno(tmp_path, {"v1.mp4": [
            ([4, 10], None),
        ]})
        pred = self._write_pred(tmp_path, {"v1.mp4": [
            ("[0.0s-1.0s]", "(4) lower install"),   # forgot (10)
        ]})
        r = evaluate_action_sequences(anno, pred)
        assert r["per_video"][0]["golden"] == [4, 10]
        assert r["per_video"][0]["predicted"] == [4]
        assert r["missing"] == 1
        assert r["sequence_accuracy"] == 0.0

    def test_mixed_single_and_two_op_dataset(self, tmp_path):
        """A dataset with both single-op and two-op chunks evaluates correctly
        when the model emits each chunk's full id set."""
        from utils.e2e_eval_utils import evaluate_action_sequences
        anno = self._write_two_op_anno(tmp_path, {
            "v1.mp4": [([1], None), ([4, 10], None)],
            "v2.mp4": [([2], None), ([3, 10], None), ([5], None)],
        })
        pred = self._write_pred(tmp_path, {
            "v1.mp4": [
                ("[0.0s-1.0s]", "(1) ok"),
                ("[1.0s-2.0s]", "(4) lower (10) idle"),
            ],
            "v2.mp4": [
                ("[0.0s-1.0s]", "(2) upper"),
                ("[1.0s-2.0s]", "(3) cooling (10) idle"),
                ("[2.0s-3.0s]", "(5) lower secured"),
            ],
        })
        r = evaluate_action_sequences(anno, pred)
        assert r["sequence_accuracy"] == 1.0
        assert r["wrong"] == 0 and r["duplicate"] == 0 and r["missing"] == 0

    def test_adjacent_dup_collapsed_symmetrically(self, tmp_path):
        """_remove_continuous_rep is applied to both golden and pred. A
        two-op chunk (3, 10) followed by a single-op chunk (10) produces
        golden=[3, 10, 10] -> [3, 10], and pred=[3, 10] stays [3, 10].
        No spurious 'missing'."""
        from utils.e2e_eval_utils import evaluate_action_sequences
        anno = self._write_two_op_anno(tmp_path, {"v.mp4": [([3, 10], None), ([10], None)]})
        pred = self._write_pred(tmp_path, {"v.mp4": [
            ("[0.0s-1.0s]", "(3) cooling (10) idle"),
            ("[1.0s-2.0s]", "(10) idle"),
        ]})
        r = evaluate_action_sequences(anno, pred)
        assert r["per_video"][0]["golden"] == [3, 10]
        assert r["per_video"][0]["predicted"] == [3, 10]
        assert r["sequence_accuracy"] == 1.0

    def test_skippable_filtered_before_continuous_dedup(self, tmp_path):
        """Order matters: skip skippable FIRST, then collapse adjacent dupes.
        Reverse order would leave phantom duplicates because the skippable
        id acts as a separator. pred [1, 2, 10, 2] with skippable={10}
        must become [1, 2], not [1, 2, 2]."""
        import json
        from utils.e2e_eval_utils import evaluate_action_sequences
        anno = self._write_anno(tmp_path, {"v1.mp4": [1, 2]})
        pred = self._write_pred(tmp_path, {"v1.mp4": [
            ("[0.0s-1.0s]", "(1) one"),
            ("[1.0s-2.0s]", "(2) two"),
            ("[2.0s-3.0s]", "(10) idle"),
            ("[3.0s-4.0s]", "(2) two"),
        ]})
        actions_path = tmp_path / "actions.json"
        actions_path.write_text(json.dumps({
            "actions": [],
            "actions_can_be_skipped": ["(10) idle"],
        }))
        r = evaluate_action_sequences(anno, pred, str(actions_path))
        assert r["per_video"][0]["predicted"] == [1, 2]  # not [1, 2, 2]
        assert r["sequence_accuracy"] == 1.0
        assert r["duplicate"] == 0


class TestDdmInferenceHelpers:
    """
    Pure-Python helpers in utils/ddm_inference.py that match the contract
    of upstream sop_monitoring.action_segment.ddm_net.detect_boundaries /
    calculate_chunk_boundaries — verifiable without torch/CUDA/PyAV.
    """

    def test_detect_boundaries_picks_local_maxima_above_threshold(self):
        from utils.ddm_inference import detect_boundaries
        # scores: peaks at index 2 (0.9) and 7 (0.8); index 5 (0.4) below threshold.
        scores = [0.1, 0.3, 0.9, 0.5, 0.2, 0.4, 0.3, 0.8, 0.6, 0.1]
        # threshold 0.6, nms_size 1 (so peak at i suppresses i-1 and i+1)
        result = detect_boundaries(scores, threshold=0.6, nms_size=1)
        assert result == [2, 7]

    def test_detect_boundaries_nms_suppresses_within_window(self):
        from utils.ddm_inference import detect_boundaries
        # Two adjacent high scores; only the larger one wins.
        scores = [0.0, 0.7, 0.9, 0.0, 0.0]
        result = detect_boundaries(scores, threshold=0.5, nms_size=2)
        assert result == [2]

    def test_detect_boundaries_empty_when_below_threshold(self):
        from utils.ddm_inference import detect_boundaries
        result = detect_boundaries([0.1, 0.2, 0.15], threshold=0.9, nms_size=1)
        assert result == []

    def test_calculate_chunk_boundaries_inserts_zero_and_duration(self):
        from utils.ddm_inference import calculate_chunk_boundaries
        # 3 boundary frames at 30/60/90 with fps=30 → 1.0/2.0/3.0 sec.
        starts, ends = calculate_chunk_boundaries(
            boundaries=[30, 60, 90], fps=30.0, duration_sec=4.5, total_frames=135,
        )
        assert starts == [0.0, 1.0, 2.0, 3.0]
        assert ends == [1.0, 2.0, 3.0, 4.5]

    def test_calculate_chunk_boundaries_no_boundaries_yields_single_chunk(self):
        from utils.ddm_inference import calculate_chunk_boundaries
        starts, ends = calculate_chunk_boundaries(
            boundaries=[], fps=30.0, duration_sec=10.0, total_frames=300,
        )
        assert starts == [0.0]
        assert ends == [10.0]


class TestExtractMcqDataEdgeCases:
    def test_missing_actions_key_raises(self, tmp_path):
        """JSON without 'actions' key raises ValueError."""
        import json
        from utils.eval_utils import extract_mcq_data
        actions_path = tmp_path / "actions.json"
        actions_path.write_text(json.dumps({"steps": ["a", "b"]}))
        with pytest.raises(ValueError, match="No actions found"):
            extract_mcq_data(str(actions_path))

    @pytest.mark.asyncio
    async def test_get_training_job_returns_scalar(self):
        from unittest.mock import AsyncMock, patch, MagicMock

        with patch("components.postgres_db.create_async_engine"), \
             patch("components.postgres_db.sessionmaker") as mock_sm:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            fake_result = MagicMock()
            fake_result.scalar_one_or_none.return_value = MagicMock(id="tj-1")
            mock_session.execute = AsyncMock(return_value=fake_result)
            mock_sm.return_value.return_value = mock_session

            from components.postgres_db import PostgresDB
            db = PostgresDB()
            result = await db.get_training_job("tj-1")
            assert result.id == "tj-1"

    @pytest.mark.asyncio
    async def test_list_training_jobs_returns_all(self):
        from unittest.mock import AsyncMock, patch, MagicMock

        with patch("components.postgres_db.create_async_engine"), \
             patch("components.postgres_db.sessionmaker") as mock_sm:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            fake_scalars = MagicMock()
            fake_scalars.all.return_value = [MagicMock(id="tj-1"), MagicMock(id="tj-2")]
            fake_result = MagicMock()
            fake_result.scalars.return_value = fake_scalars
            mock_session.execute = AsyncMock(return_value=fake_result)
            mock_sm.return_value.return_value = mock_session

            from components.postgres_db import PostgresDB
            db = PostgresDB()
            jobs = await db.list_training_jobs()
            assert len(jobs) == 2

    @pytest.mark.asyncio
    async def test_list_training_jobs_with_status_filter(self):
        """list_training_jobs(status=...) applies a WHERE filter (covers the status branch)."""
        from unittest.mock import AsyncMock, patch, MagicMock

        with patch("components.postgres_db.create_async_engine"), \
             patch("components.postgres_db.sessionmaker") as mock_sm:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            fake_scalars = MagicMock()
            fake_scalars.all.return_value = [MagicMock(id="tj-running")]
            fake_result = MagicMock()
            fake_result.scalars.return_value = fake_scalars
            mock_session.execute = AsyncMock(return_value=fake_result)
            mock_sm.return_value.return_value = mock_session

            from components.postgres_db import PostgresDB
            db = PostgresDB()
            jobs = await db.list_training_jobs(status="running")
            assert len(jobs) == 1

    @pytest.mark.asyncio
    async def test_insert_eval_job_no_valid_fields_returns_early(self):
        from unittest.mock import AsyncMock, patch

        with patch("components.postgres_db.create_async_engine"), \
             patch("components.postgres_db.sessionmaker") as mock_sm:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_sm.return_value.return_value = mock_session

            from components.postgres_db import PostgresDB
            db = PostgresDB()
            result = await db.insert_evaluation_job(nonexistent_field="x")
            assert result is None
            mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_eval_job_no_valid_fields_returns_early(self):
        from unittest.mock import AsyncMock, patch

        with patch("components.postgres_db.create_async_engine"), \
             patch("components.postgres_db.sessionmaker") as mock_sm:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_sm.return_value.return_value = mock_session

            from components.postgres_db import PostgresDB
            db = PostgresDB()
            result = await db.update_evaluation_job("ej-1", nonexistent_field="x")
            assert result is None
            mock_session.execute.assert_not_called()
