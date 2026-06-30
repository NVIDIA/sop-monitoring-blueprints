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

"""Unit tests for evaluation API endpoints and run_evaluation background task."""
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MS_PATH = str(PROJECT_ROOT / "microservices" / "evaluation-ms")
if MS_PATH not in sys.path:
    sys.path.insert(0, MS_PATH)

# Mock DB before importing app
with patch("components.postgres_db.postgres_db") as mock_db:
    mock_db.insert_training_job = AsyncMock(return_value=None)
    mock_db.get_training_job = AsyncMock(return_value=None)
    mock_db.update_training_job = AsyncMock(return_value=None)
    mock_db.list_training_jobs = AsyncMock(return_value=[])
    mock_db.insert_evaluation_job = AsyncMock(return_value=None)
    mock_db.get_evaluation_job = AsyncMock(return_value=None)
    mock_db.update_evaluation_job = AsyncMock(return_value=None)
    mock_db.list_evaluation_jobs = AsyncMock(return_value=[])
    from app import run_evaluation, run_e2e_evaluation


class TestRunEvaluationSuccess:
    @pytest.mark.asyncio
    async def test_run_evaluation_completes_and_stores_results(self, tmp_path):
        """run_evaluation writes results to DB on subprocess success (return_code=0)."""
        eval_job_id = "eval-success-1"
        output_dir = str(tmp_path / "eval-success-1")
        os.makedirs(output_dir, exist_ok=True)

        # Write fake inference results that sop_eval.py would produce
        fake_inference = {
            "video1": [[1, "(1) install cable"], [2, "(2) install board"]],
            "video2": [[1, "(1) install cable"]],
        }
        (tmp_path / "eval-success-1" / "inference_results.json").write_text(
            json.dumps(fake_inference)
        )

        fake_choices = ["(1) install cable", "(2) install board"]

        mock_process = MagicMock()
        mock_process.pid = 9001
        mock_process.wait = AsyncMock(return_value=0)

        mock_process.stdout = MagicMock()
        mock_process.stdout.readline = AsyncMock(return_value=b"")
        mock_process.stderr = MagicMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")

        cached_job = {
            "eval_job_id": eval_job_id,
            "training_job_id": "tj-1",
            "val_dataset_id": "vd-1",
            "checkpoint_step": 500,
            "status": "running",
            "overall_accuracy": None,
            "results_json": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "log_file_path": str(tmp_path / "eval-success-1" / "log.txt"),
            "process_pid": 9001,
        }
        Path(cached_job["log_file_path"]).write_text("")

        final_status = {}

        with patch("app.eval_jobs_cache") as mock_cache, \
             patch("app.postgres_db") as mock_pg, \
             patch("app.prepare_eval_assets", return_value=str(tmp_path / "assets")), \
             patch("app.extract_mcq_data", return_value=("prompt text", fake_choices)), \
             patch("app.parse_eval_results") as mock_parse, \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
             patch("utils.constant.RESULTS_ROOT", str(tmp_path)):

            mock_cache.get.return_value = cached_job
            mock_exec.return_value = mock_process
            mock_pg.update_evaluation_job = AsyncMock()

            mock_parse.return_value = {
                "overall_accuracy": 1.0,
                "per_action": {
                    "1": {"label": "(1) install cable", "correct": 2, "total": 2, "accuracy": 1.0},
                    "2": {"label": "(2) install board", "correct": 1, "total": 1, "accuracy": 1.0},
                }
            }

            mock_cache.update.side_effect = lambda jid, **kw: final_status.update(kw)

            await run_evaluation(
                eval_job_id=eval_job_id,
                training_job_id="tj-1",
                actions_json_path="/fake/actions.json",
                val_dataset_id="vd-1",
                checkpoint_path="/fake/step_500",
                checkpoint_step=500,
                fps=8,
                temperature=0.0,
                backend="vllm",
            )

        assert final_status.get("status") == "completed"
        assert final_status.get("overall_accuracy") == pytest.approx(1.0)
        mock_cache.delete.assert_called_with(eval_job_id)


class TestRunEvaluationFailure:
    @pytest.mark.asyncio
    async def test_run_evaluation_sets_failed_on_nonzero_return(self, tmp_path):
        """run_evaluation sets status=failed when subprocess returns non-zero."""
        eval_job_id = "eval-fail-1"
        os.makedirs(str(tmp_path / eval_job_id), exist_ok=True)
        log_path = str(tmp_path / eval_job_id / "log.txt")
        Path(log_path).write_text("")

        mock_process = MagicMock()
        mock_process.pid = 9002
        mock_process.wait = AsyncMock(return_value=1)

        mock_process.stdout = MagicMock()
        mock_process.stdout.readline = AsyncMock(return_value=b"")
        mock_process.stderr = MagicMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")

        cached_job = {
            "eval_job_id": eval_job_id,
            "status": "running",
            "overall_accuracy": None,
            "results_json": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "log_file_path": log_path,
            "process_pid": 9002,
        }

        final_status = {}

        with patch("app.eval_jobs_cache") as mock_cache, \
             patch("app.postgres_db") as mock_pg, \
             patch("app.prepare_eval_assets", return_value=str(tmp_path / "assets")), \
             patch("app.extract_mcq_data", return_value=("prompt", ["(1) step one"])), \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
             patch("utils.constant.RESULTS_ROOT", str(tmp_path)):

            mock_cache.get.return_value = cached_job
            mock_exec.return_value = mock_process
            mock_pg.update_evaluation_job = AsyncMock()
            mock_cache.update.side_effect = lambda jid, **kw: final_status.update(kw)

            await run_evaluation(
                eval_job_id=eval_job_id,
                training_job_id="tj-1",
                actions_json_path="/fake/actions.json",
                val_dataset_id="vd-1",
                checkpoint_path="/fake/step_100",
                checkpoint_step=100,
                fps=8,
                temperature=0.0,
                backend="vllm",
            )

        assert final_status.get("status") == "failed"
        mock_cache.delete.assert_called_with(eval_job_id)


class TestRunEvaluationWithLogOutput:
    @pytest.mark.asyncio
    async def test_run_evaluation_streams_stdout_to_log(self, tmp_path):
        """Lines from subprocess stdout are written to the log file (covers read_stream body)."""
        eval_job_id = "eval-log-1"
        os.makedirs(str(tmp_path / eval_job_id), exist_ok=True)
        log_path = str(tmp_path / eval_job_id / "log.txt")
        Path(log_path).write_text("")

        # Fake inference results for return_code=0 path
        import json
        fake_inference = {"video1": [[1, "(1) step one"]]}
        (tmp_path / eval_job_id / "inference_results.json").write_text(json.dumps(fake_inference))

        # Simulate two log lines then EOF
        mock_process = MagicMock()
        mock_process.pid = 9003
        mock_process.wait = AsyncMock(return_value=0)
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[b"Loading model...\n", b"Running inference...\n", b""]
        )
        mock_process.stderr = MagicMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")

        cached_job = {
            "eval_job_id": eval_job_id,
            "status": "running",
            "overall_accuracy": None,
            "results_json": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "log_file_path": log_path,
            "process_pid": 9003,
        }

        with patch("app.eval_jobs_cache") as mock_cache, \
             patch("app.postgres_db") as mock_pg, \
             patch("app.prepare_eval_assets", return_value=str(tmp_path / "assets")), \
             patch("app.extract_mcq_data", return_value=("prompt", ["(1) step one"])), \
             patch("app.parse_eval_results") as mock_parse, \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
             patch("utils.constant.RESULTS_ROOT", str(tmp_path)):

            mock_cache.get.return_value = cached_job
            mock_exec.return_value = mock_process
            mock_pg.update_evaluation_job = AsyncMock()
            mock_parse.return_value = {"overall_accuracy": 1.0, "per_action": {}}

            await run_evaluation(
                eval_job_id=eval_job_id,
                training_job_id="tj-1",
                actions_json_path="/fake/actions.json",
                val_dataset_id="vd-1",
                checkpoint_path="/fake/step_500",
                checkpoint_step=500,
                fps=8,
                temperature=0.0,
                backend="vllm",
            )

        log_content = Path(log_path).read_text()
        assert "Loading model" in log_content
        assert "Running inference" in log_content


class TestRunEvaluationExceptionPath:
    @pytest.mark.asyncio
    async def test_run_evaluation_exception_sets_failed_and_updates_db(self, tmp_path):
        """When an unexpected exception occurs inside run_evaluation, status→failed and DB updated."""
        eval_job_id = "eval-exc-1"
        os.makedirs(str(tmp_path / eval_job_id), exist_ok=True)
        log_path = str(tmp_path / eval_job_id / "log.txt")
        Path(log_path).write_text("")

        cached_job = {
            "eval_job_id": eval_job_id,
            "status": "running",
            "overall_accuracy": None,
            "results_json": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "log_file_path": log_path,
            "process_pid": None,
        }
        final_status = {}

        def fake_get(jid):
            return {**cached_job, **final_status}

        with patch("app.eval_jobs_cache") as mock_cache, \
             patch("app.postgres_db") as mock_pg, \
             patch("app.extract_mcq_data", side_effect=RuntimeError("boom")), \
             patch("utils.constant.RESULTS_ROOT", str(tmp_path)):

            mock_cache.get.side_effect = fake_get
            mock_cache.update.side_effect = lambda jid, **kw: final_status.update(kw)
            mock_pg.update_evaluation_job = AsyncMock()

            await run_evaluation(
                eval_job_id=eval_job_id,
                training_job_id="tj-1",
                actions_json_path="/fake/actions.json",
                val_dataset_id="vd-1",
                checkpoint_path="/fake/step_100",
                checkpoint_step=100,
                fps=8,
                temperature=0.0,
                backend="vllm",
            )

        assert final_status.get("status") == "failed"
        mock_pg.update_evaluation_job.assert_called()
        mock_cache.delete.assert_called_with(eval_job_id)
        log_content = Path(log_path).read_text()
        assert "[ERROR]" in log_content


class TestRunE2eEvaluationUniformChunking:
    """run_e2e_evaluation must pass --chunking-algorithm and --chunk-length-sec
    to the subprocess and skip DDM-only args when uniform."""

    @pytest.mark.asyncio
    async def test_uniform_passes_chunking_args_to_subprocess(self, tmp_path):
        eval_job_id = "e2e-uniform-1"
        os.makedirs(str(tmp_path / eval_job_id), exist_ok=True)
        log_path = str(tmp_path / eval_job_id / "log.txt")
        Path(log_path).write_text("")

        # Fake e2e_results.json that the subprocess would write
        fake_results = {
            "temporal_segmentation": {"avg_f1": 0.0},
            "action_recognition": {"overall_accuracy": 1.0},
        }
        (tmp_path / eval_job_id / "e2e_results.json").write_text(json.dumps(fake_results))

        mock_process = MagicMock()
        mock_process.pid = 9101
        mock_process.wait = AsyncMock(return_value=0)
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline = AsyncMock(return_value=b"")
        mock_process.stderr = MagicMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")

        cached_job = {
            "eval_job_id": eval_job_id,
            "status": "running",
            "overall_accuracy": None,
            "avg_f1": None,
            "results_json": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "log_file_path": log_path,
            "process_pid": None,
        }

        captured_cmd = []

        with patch("app.e2e_eval_jobs_cache") as mock_cache, \
             patch("app.postgres_db") as mock_pg, \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
             patch("utils.constant.RESULTS_ROOT", str(tmp_path)):

            mock_cache.get.return_value = cached_job
            mock_pg.update_e2e_evaluation_job = AsyncMock()

            async def fake_exec(*args, **kwargs):
                captured_cmd.extend(args)
                return mock_process

            mock_exec.side_effect = fake_exec

            await run_e2e_evaluation(
                eval_job_id=eval_job_id,
                actions_json_path="/fake/actions.json",
                anno_json_path="/fake/anno.json",
                val_dataset_id="vd-1",
                checkpoint_path="/fake/step_500",
                checkpoint_step=500,
                ddm_checkpoint_path=None,
                ddm_resolution=224,
                ddm_frames_per_side=5,
                fps=8,
                temperature=0.0,
                backend="vllm",
                score_threshold=0.5,
                nms_sec=0.0,
                ddm_batch_size=8,
                frames_per_segment_hint=256,
                chunking_algorithm="uniform",
                chunk_length_sec=10.0,
            )

        # Inspect the subprocess CLI that was launched
        cmd = list(captured_cmd)
        assert "--chunking-algorithm" in cmd
        assert cmd[cmd.index("--chunking-algorithm") + 1] == "uniform"
        assert "--chunk-length-sec" in cmd
        assert cmd[cmd.index("--chunk-length-sec") + 1] == "10.0"
        # DDM checkpoint path must NOT be on the CLI when uniform (no DDM run).
        assert "--ddm-checkpoint-path" not in cmd

    @pytest.mark.asyncio
    async def test_ddm_passes_ddm_args_to_subprocess(self, tmp_path):
        """Sanity check: DDM mode still passes --ddm-checkpoint-path (no regression)."""
        eval_job_id = "e2e-ddm-1"
        os.makedirs(str(tmp_path / eval_job_id), exist_ok=True)
        log_path = str(tmp_path / eval_job_id / "log.txt")
        Path(log_path).write_text("")

        fake_results = {
            "temporal_segmentation": {"avg_f1": 0.9},
            "action_recognition": {"overall_accuracy": 0.85},
        }
        (tmp_path / eval_job_id / "e2e_results.json").write_text(json.dumps(fake_results))

        mock_process = MagicMock()
        mock_process.pid = 9102
        mock_process.wait = AsyncMock(return_value=0)
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline = AsyncMock(return_value=b"")
        mock_process.stderr = MagicMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")

        cached_job = {
            "eval_job_id": eval_job_id,
            "status": "running",
            "overall_accuracy": None,
            "avg_f1": None,
            "results_json": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "log_file_path": log_path,
            "process_pid": None,
        }

        captured_cmd = []

        with patch("app.e2e_eval_jobs_cache") as mock_cache, \
             patch("app.postgres_db") as mock_pg, \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
             patch("utils.constant.RESULTS_ROOT", str(tmp_path)):

            mock_cache.get.return_value = cached_job
            mock_pg.update_e2e_evaluation_job = AsyncMock()

            async def fake_exec(*args, **kwargs):
                captured_cmd.extend(args)
                return mock_process

            mock_exec.side_effect = fake_exec

            await run_e2e_evaluation(
                eval_job_id=eval_job_id,
                actions_json_path="/fake/actions.json",
                anno_json_path="/fake/anno.json",
                val_dataset_id="vd-1",
                checkpoint_path="/fake/step_500",
                checkpoint_step=500,
                ddm_checkpoint_path="/fake/last.ckpt",
                ddm_resolution=224,
                ddm_frames_per_side=5,
                fps=8,
                temperature=0.0,
                backend="vllm",
                score_threshold=0.5,
                nms_sec=0.0,
                ddm_batch_size=8,
                frames_per_segment_hint=256,
                chunking_algorithm="ddm",
                chunk_length_sec=None,
            )

        cmd = list(captured_cmd)
        assert "--chunking-algorithm" in cmd
        assert cmd[cmd.index("--chunking-algorithm") + 1] == "ddm"
        assert "--ddm-checkpoint-path" in cmd
        assert cmd[cmd.index("--ddm-checkpoint-path") + 1] == "/fake/last.ckpt"
        # No --chunk-length-sec when ddm mode (it's None)
        assert "--chunk-length-sec" not in cmd
