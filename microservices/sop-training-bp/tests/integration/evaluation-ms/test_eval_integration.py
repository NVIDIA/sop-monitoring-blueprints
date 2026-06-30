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

"""
End-to-end integration tests for the CR1 evaluation API.

Uses FastAPI TestClient with mocked DB and subprocess.
Tests the complete evaluation flow:
  1. POST /api/v1/evaluation/start
  2. GET  /api/v1/evaluation/status/{id}
  3. GET  /api/v1/evaluation/results/{id}
  4. GET  /api/v1/evaluation/all_jobs
  5. POST /api/v1/evaluation/cancel/{id}

Mock data:
  - Training job: completed, aug_dataset_id="aug-foxconn-001"
  - Val dataset: "val-foxconn-001" with 3 action types
  - MCQ choices: 3 actions + "doing none of the above"
  - Inference results: video1 has 2 chunks (action1 correct, action2 correct)
  - Expected overall accuracy: 1.0 (both correct)
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MS_PATH = str(PROJECT_ROOT / "microservices" / "evaluation-ms")
if MS_PATH not in sys.path:
    sys.path.insert(0, MS_PATH)

# ── Mock data ──────────────────────────────────────────────────────────────────

TRAINING_JOB_ID = "train-foxconn-001"
AUG_DATASET_ID = "foxconn-001_augmented_0"
ORIGINAL_DATASET_ID = "foxconn-001"
VAL_DATASET_ID = "val-foxconn-001"
CHECKPOINT_STEP = 500

MCQ_CHOICES = [
    "(1) installing the cable to the server closer to the operator",
    "(2) installing the cable to the middle of the server",
    "(3) doing none of the above",
]
MCQ_PROMPT = (
    "There are 3 possible steps for the SOP.\n"
    "What step is the operator doing?\n"
    + "\n".join(MCQ_CHOICES)
)
ACTIONS_DATA = {"actions": [
    "installing the cable to the server closer to the operator.",
    "installing the cable to the middle of the server.",
    "doing none of the above.",
]}

# Inference results that sop_eval.py would produce
# video1: chunk 01_video_1_1.mp4 → action 1 correct, chunk 02_video_1_2.mp4 → action 2 correct
INFERENCE_RESULTS = {
    "video1": [
        [1, "(1) installing the cable to the server closer to the operator"],
        [2, "(2) installing the cable to the middle of the server"],
    ]
}
EXPECTED_OVERALL_ACCURACY = 1.0
EXPECTED_PER_ACTION = {
    "1": {"correct": 1, "total": 1, "accuracy": 1.0},
    "2": {"correct": 1, "total": 1, "accuracy": 1.0},
}

# ── Mock DB objects ────────────────────────────────────────────────────────────

def make_mock_training_job():
    job = MagicMock()
    job.id = TRAINING_JOB_ID
    job.aug_dataset_id = AUG_DATASET_ID
    job.status = "completed"
    job.to_dict.return_value = {
        "id": TRAINING_JOB_ID,
        "aug_dataset_id": AUG_DATASET_ID,
        "status": "completed",
        "progress": 100.0,
        "current_step": 500,
        "total_steps": 500,
        "loss": 0.1,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }
    return job


def make_mock_eval_job(eval_job_id, status="completed", overall_accuracy=1.0, results_json=None):
    job = MagicMock()
    job.id = eval_job_id
    job.training_job_id = TRAINING_JOB_ID
    job.val_dataset_id = VAL_DATASET_ID
    job.checkpoint_step = CHECKPOINT_STEP
    job.status = status
    job.overall_accuracy = overall_accuracy
    job.results_json = results_json or {
        "overall_accuracy": overall_accuracy,
        "per_action": {
            "1": {"label": MCQ_CHOICES[0], "correct": 1, "total": 1, "accuracy": 1.0},
            "2": {"label": MCQ_CHOICES[1], "correct": 1, "total": 1, "accuracy": 1.0},
        }
    }
    job.to_dict.return_value = {
        "id": eval_job_id,
        "training_job_id": TRAINING_JOB_ID,
        "val_dataset_id": VAL_DATASET_ID,
        "checkpoint_step": CHECKPOINT_STEP,
        "status": status,
        "overall_accuracy": overall_accuracy,
        "results_json": job.results_json,
        "fps": 8,
        "temperature": 0.0,
        "backend": "vllm",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }
    return job


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestEvalApiStartEndpoint:
    """POST /api/v1/evaluation/start"""

    def test_start_returns_queued_eval_job_id(self, tmp_path):
        from fastapi.testclient import TestClient
        from app import app

        mock_training_job = make_mock_training_job()
        # Create actions.json in original dataset
        orig_dir = tmp_path / ORIGINAL_DATASET_ID
        orig_dir.mkdir(parents=True, exist_ok=True)
        (orig_dir / "actions.json").write_text(json.dumps(ACTIONS_DATA))
        # Create fake checkpoint dir
        checkpoint_dir = tmp_path / TRAINING_JOB_ID / f"step_{CHECKPOINT_STEP}"
        checkpoint_dir.mkdir(parents=True)

        cached_job_state = {}

        def fake_set(job_id, data):
            cached_job_state[job_id] = data

        def fake_get(job_id):
            return cached_job_state.get(job_id)

        with patch("app.postgres_db") as mock_pg, \
             patch("app.const") as mock_const, \
             patch("app.eval_jobs_cache") as mock_eval_cache, \
             patch("app.run_evaluation", new_callable=AsyncMock):
            mock_const.RUNNING_STATUS = "running"
            mock_const.QUEUE_STATUS = "queued"
            mock_const.COMPLETED_STATUS = "completed"
            mock_const.DATASET_ROOT = str(tmp_path)
            mock_const.RESULTS_ROOT = str(tmp_path)
            mock_pg.get_training_job = AsyncMock(return_value=mock_training_job)
            mock_pg.get_original_dataset_id = AsyncMock(return_value=ORIGINAL_DATASET_ID)
            mock_pg.insert_evaluation_job = AsyncMock(return_value=None)
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[mock_training_job])
            mock_eval_cache.cache = {}
            mock_eval_cache.get = MagicMock(side_effect=fake_get)
            mock_eval_cache.set = MagicMock(side_effect=fake_set)
            mock_eval_cache.update = MagicMock()

            client = TestClient(app)
            resp = client.post("/api/v1/evaluation/start", json={
                "training_job_id": TRAINING_JOB_ID,
                "val_dataset_id": VAL_DATASET_ID,
                "fps": 8,
                "temperature": 0.0,
                "backend": "vllm",
            })

        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "eval_job_id" in data
        assert data["status"] == "queued"
        assert data["message"] != ""

    def test_start_404_on_unknown_training_job(self):
        from fastapi.testclient import TestClient
        from app import app

        with patch("app.postgres_db") as mock_pg, \
             patch("app.eval_jobs_cache") as mock_eval_cache:
            mock_pg.get_training_job = AsyncMock(return_value=None)
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])
            mock_eval_cache.cache = {}

            client = TestClient(app)
            resp = client.post("/api/v1/evaluation/start", json={
                "training_job_id": "nonexistent-job",
                "val_dataset_id": VAL_DATASET_ID,
            })

        assert resp.status_code == 404

    def test_start_400_on_incomplete_training_job(self):
        from fastapi.testclient import TestClient
        from app import app

        running_job = make_mock_training_job()
        running_job.status = "running"

        with patch("app.postgres_db") as mock_pg, \
             patch("app.eval_jobs_cache") as mock_eval_cache, \
             patch("app.const") as mock_const:
            mock_const.RUNNING_STATUS = "running"
            mock_const.COMPLETED_STATUS = "completed"
            mock_pg.get_training_job = AsyncMock(return_value=running_job)
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])
            mock_eval_cache.cache = {}

            client = TestClient(app)
            resp = client.post("/api/v1/evaluation/start", json={
                "training_job_id": TRAINING_JOB_ID,
                "val_dataset_id": VAL_DATASET_ID,
            })

        assert resp.status_code == 400
        assert "not completed" in resp.json()["detail"]


class TestEvalApiStatusEndpoint:
    """GET /api/v1/evaluation/status/{eval_job_id}"""

    def test_status_returns_job_info(self):
        from fastapi.testclient import TestClient
        from app import app, eval_jobs_cache

        eval_job_id = "eval-status-test-1"
        mock_eval_job = make_mock_eval_job(eval_job_id, status="completed", overall_accuracy=1.0)
        mock_eval_job.to_dict.return_value.update({
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        })

        with patch("app.postgres_db") as mock_pg:
            mock_pg.get_evaluation_job = AsyncMock(return_value=mock_eval_job)
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])

            # Clear cache so it falls back to DB
            eval_jobs_cache.cache.clear()

            client = TestClient(app)
            resp = client.get(f"/api/v1/evaluation/status/{eval_job_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["eval_job_id"] == eval_job_id
        assert data["status"] == "completed"
        assert data["checkpoint_step"] == CHECKPOINT_STEP

    def test_status_404_on_unknown_job(self):
        from fastapi.testclient import TestClient
        from app import app, eval_jobs_cache

        with patch("app.postgres_db") as mock_pg:
            mock_pg.get_evaluation_job = AsyncMock(return_value=None)
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])

            eval_jobs_cache.cache.clear()

            client = TestClient(app)
            resp = client.get("/api/v1/evaluation/status/no-such-job")

        assert resp.status_code == 404


class TestEvalApiResultsEndpoint:
    """GET /api/v1/evaluation/results/{eval_job_id}"""

    def test_results_returns_accuracy_and_per_action_breakdown(self):
        from fastapi.testclient import TestClient
        from app import app

        eval_job_id = "eval-results-test-1"
        expected_results = {
            "overall_accuracy": EXPECTED_OVERALL_ACCURACY,
            "per_action": {
                "1": {"label": MCQ_CHOICES[0], "correct": 1, "total": 1, "accuracy": 1.0},
                "2": {"label": MCQ_CHOICES[1], "correct": 1, "total": 1, "accuracy": 1.0},
            }
        }
        mock_eval_job = make_mock_eval_job(eval_job_id, status="completed",
                                           overall_accuracy=1.0, results_json=expected_results)

        with patch("app.postgres_db") as mock_pg, \
             patch("app.const") as mock_const:
            mock_const.COMPLETED_STATUS = "completed"
            mock_pg.get_evaluation_job = AsyncMock(return_value=mock_eval_job)
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])

            client = TestClient(app)
            resp = client.get(f"/api/v1/evaluation/results/{eval_job_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_accuracy"] == EXPECTED_OVERALL_ACCURACY
        assert "per_action" in data
        assert "1" in data["per_action"]
        assert data["per_action"]["1"]["correct"] == EXPECTED_PER_ACTION["1"]["correct"]
        assert data["per_action"]["1"]["accuracy"] == EXPECTED_PER_ACTION["1"]["accuracy"]
        assert data["per_action"]["2"]["accuracy"] == EXPECTED_PER_ACTION["2"]["accuracy"]

    def test_results_400_when_not_completed(self):
        from fastapi.testclient import TestClient
        from app import app

        eval_job_id = "eval-running-test"
        mock_eval_job = make_mock_eval_job(eval_job_id, status="running", overall_accuracy=None)

        with patch("app.postgres_db") as mock_pg, \
             patch("app.const") as mock_const:
            mock_const.COMPLETED_STATUS = "completed"
            mock_pg.get_evaluation_job = AsyncMock(return_value=mock_eval_job)
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])

            client = TestClient(app)
            resp = client.get(f"/api/v1/evaluation/results/{eval_job_id}")

        assert resp.status_code == 400
        assert "not completed" in resp.json()["detail"]

    def test_results_404_on_unknown_job(self):
        from fastapi.testclient import TestClient
        from app import app

        with patch("app.postgres_db") as mock_pg:
            mock_pg.get_evaluation_job = AsyncMock(return_value=None)
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])

            client = TestClient(app)
            resp = client.get("/api/v1/evaluation/results/no-such-job")

        assert resp.status_code == 404


class TestEvalApiAllJobsEndpoint:
    """GET /api/v1/evaluation/all_jobs"""

    def test_all_jobs_returns_dict_keyed_by_id(self):
        from fastapi.testclient import TestClient
        from app import app

        eval_job_1 = make_mock_eval_job("eval-all-1", status="completed", overall_accuracy=0.85)
        eval_job_2 = make_mock_eval_job("eval-all-2", status="failed", overall_accuracy=None)

        with patch("app.postgres_db") as mock_pg:
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[eval_job_1, eval_job_2])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])

            client = TestClient(app)
            resp = client.get("/api/v1/evaluation/all_jobs")

        assert resp.status_code == 200
        data = resp.json()
        assert "eval-all-1" in data
        assert "eval-all-2" in data
        assert data["eval-all-1"]["status"] == "completed"
        assert data["eval-all-2"]["status"] == "failed"

    def test_all_jobs_returns_empty_dict_when_no_jobs(self):
        from fastapi.testclient import TestClient
        from app import app

        with patch("app.postgres_db") as mock_pg:
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])

            client = TestClient(app)
            resp = client.get("/api/v1/evaluation/all_jobs")

        assert resp.status_code == 200
        assert resp.json() == {}


class TestEvalApiCancelEndpoint:
    """POST /api/v1/evaluation/cancel/{eval_job_id}"""

    def test_cancel_404_when_not_in_cache(self):
        from fastapi.testclient import TestClient
        from app import app, eval_jobs_cache

        with patch("app.postgres_db") as mock_pg:
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])

            eval_jobs_cache.cache.clear()

            client = TestClient(app)
            resp = client.post("/api/v1/evaluation/cancel/no-such-job")

        assert resp.status_code == 404

    def test_cancel_200_with_no_pid_yet(self):
        """Job in cache but PID not set yet returns 200 with 'not started' message."""
        from fastapi.testclient import TestClient
        from app import app

        eval_job_id = "eval-cancel-nopid"
        cached_job = {"eval_job_id": eval_job_id, "status": "queued", "process_pid": None}

        with patch("app.eval_jobs_cache") as mock_cache, \
             patch("app.postgres_db") as mock_pg:
            mock_cache.get.return_value = cached_job
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])

            client = TestClient(app)
            resp = client.post(f"/api/v1/evaluation/cancel/{eval_job_id}")

        assert resp.status_code == 200
        assert "not started" in resp.json()["message"].lower()

    def test_cancel_200_on_successful_kill(self):
        """Kill succeeds: status set to cancelled, DB updated, 200 returned."""
        from fastapi.testclient import TestClient
        from app import app

        eval_job_id = "eval-cancel-success"
        update_state = {}

        def fake_get(jid):
            return {**{"eval_job_id": jid, "status": "running", "process_pid": 12345}, **update_state}

        with patch("app.eval_jobs_cache") as mock_cache, \
             patch("app.postgres_db") as mock_pg, \
             patch("app.terminate_process_tree", return_value=True):
            mock_cache.get.side_effect = fake_get
            mock_cache.update.side_effect = lambda jid, **kw: update_state.update(kw)
            mock_pg.update_evaluation_job = AsyncMock()
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])

            client = TestClient(app)
            resp = client.post(f"/api/v1/evaluation/cancel/{eval_job_id}")

        assert resp.status_code == 200
        assert "cancelled" in resp.json()["message"].lower()
        mock_pg.update_evaluation_job.assert_called()

    def test_cancel_200_on_failed_kill(self):
        """Kill fails: returns 200 with 'failed to cancel' message."""
        from fastapi.testclient import TestClient
        from app import app

        eval_job_id = "eval-cancel-killfail"

        with patch("app.eval_jobs_cache") as mock_cache, \
             patch("app.postgres_db") as mock_pg, \
             patch("app.terminate_process_tree", return_value=False):
            mock_cache.get.return_value = {"eval_job_id": eval_job_id, "status": "running", "process_pid": 99999}
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])

            client = TestClient(app)
            resp = client.post(f"/api/v1/evaluation/cancel/{eval_job_id}")

        assert resp.status_code == 200
        assert "failed" in resp.json()["message"].lower()


class TestEvalApiStartEdgeCases:
    """Additional edge cases for POST /api/v1/evaluation/start."""

    def test_start_400_when_eval_already_running(self):
        """Guard: reject start if another eval is already running."""
        from fastapi.testclient import TestClient
        from app import app

        with patch("app.eval_jobs_cache") as mock_cache, \
             patch("app.postgres_db") as mock_pg, \
             patch("app.const") as mock_const:
            mock_const.RUNNING_STATUS = "running"
            mock_const.COMPLETED_STATUS = "completed"
            mock_const.QUEUE_STATUS = "queued"
            mock_cache.cache = {"existing-job": {"eval_job_id": "existing-job", "status": "running"}}
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])

            client = TestClient(app)
            resp = client.post("/api/v1/evaluation/start", json={
                "training_job_id": TRAINING_JOB_ID,
                "val_dataset_id": VAL_DATASET_ID,
            })

        assert resp.status_code == 400
        assert "already running" in resp.json()["detail"]

    def test_start_400_when_actions_json_missing(self, tmp_path):
        """Guard: actions.json not present in original dataset → 400."""
        from fastapi.testclient import TestClient
        from app import app

        mock_training_job = make_mock_training_job()
        # Create checkpoint dir but NOT the actions.json
        checkpoint_dir = tmp_path / TRAINING_JOB_ID / f"step_{CHECKPOINT_STEP}"
        checkpoint_dir.mkdir(parents=True)

        with patch("app.postgres_db") as mock_pg, \
             patch("app.const") as mock_const, \
             patch("app.eval_jobs_cache") as mock_cache:
            mock_const.RUNNING_STATUS = "running"
            mock_const.QUEUE_STATUS = "queued"
            mock_const.COMPLETED_STATUS = "completed"
            mock_const.DATASET_ROOT = str(tmp_path)
            mock_const.RESULTS_ROOT = str(tmp_path)
            mock_pg.get_training_job = AsyncMock(return_value=mock_training_job)
            mock_pg.get_original_dataset_id = AsyncMock(return_value=ORIGINAL_DATASET_ID)
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])
            mock_cache.cache = {}

            client = TestClient(app)
            resp = client.post("/api/v1/evaluation/start", json={
                "training_job_id": TRAINING_JOB_ID,
                "val_dataset_id": VAL_DATASET_ID,
            })

        assert resp.status_code == 400
        assert "actions.json not found" in resp.json()["detail"]

    def test_start_400_when_checkpoint_missing(self, tmp_path):
        """Guard: no checkpoint dir for the training job → 400."""
        from fastapi.testclient import TestClient
        from app import app

        mock_training_job = make_mock_training_job()
        # Create actions.json but no checkpoint dirs
        orig_dir = tmp_path / ORIGINAL_DATASET_ID
        orig_dir.mkdir(parents=True, exist_ok=True)
        (orig_dir / "actions.json").write_text(json.dumps(ACTIONS_DATA))

        with patch("app.postgres_db") as mock_pg, \
             patch("app.const") as mock_const, \
             patch("app.eval_jobs_cache") as mock_cache:
            mock_const.RUNNING_STATUS = "running"
            mock_const.QUEUE_STATUS = "queued"
            mock_const.COMPLETED_STATUS = "completed"
            mock_const.DATASET_ROOT = str(tmp_path)
            mock_const.RESULTS_ROOT = str(tmp_path)
            mock_pg.get_training_job = AsyncMock(return_value=mock_training_job)
            mock_pg.get_original_dataset_id = AsyncMock(return_value=ORIGINAL_DATASET_ID)
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])
            mock_cache.cache = {}

            client = TestClient(app)
            resp = client.post("/api/v1/evaluation/start", json={
                "training_job_id": TRAINING_JOB_ID,
                "val_dataset_id": VAL_DATASET_ID,
            })

        assert resp.status_code == 400


class TestEvalApiErrorHandling:
    """500 error paths for status and results endpoints."""

    def test_status_500_on_db_exception(self):
        from fastapi.testclient import TestClient
        from app import app, eval_jobs_cache

        eval_jobs_cache.cache.clear()

        with patch("app.postgres_db") as mock_pg:
            mock_pg.get_evaluation_job = AsyncMock(side_effect=RuntimeError("DB down"))
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])

            client = TestClient(app)
            resp = client.get("/api/v1/evaluation/status/some-job-id")

        assert resp.status_code == 500

    def test_results_500_on_db_exception(self):
        from fastapi.testclient import TestClient
        from app import app

        with patch("app.postgres_db") as mock_pg:
            mock_pg.get_evaluation_job = AsyncMock(side_effect=RuntimeError("DB down"))
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])

            client = TestClient(app)
            resp = client.get("/api/v1/evaluation/results/some-job-id")

        assert resp.status_code == 500

    def test_start_500_on_unexpected_exception(self):
        """Generic 500 path in start_evaluation when an unexpected error occurs after guards pass."""
        from fastapi.testclient import TestClient
        from app import app

        mock_training_job = make_mock_training_job()

        with patch("app.postgres_db") as mock_pg, \
             patch("app.const") as mock_const, \
             patch("app.eval_jobs_cache") as mock_cache, \
             patch("app.os.path.exists", return_value=True), \
             patch("app.resolve_checkpoint_path", return_value=("/fake/step_500", 500)), \
             patch("app.uuid") as mock_uuid:
            mock_const.RUNNING_STATUS = "running"
            mock_const.QUEUE_STATUS = "queued"
            mock_const.COMPLETED_STATUS = "completed"
            mock_const.DATASET_ROOT = "/fake"
            mock_const.RESULTS_ROOT = "/fake"
            mock_pg.get_training_job = AsyncMock(return_value=mock_training_job)
            mock_pg.get_original_dataset_id = AsyncMock(return_value=ORIGINAL_DATASET_ID)
            mock_pg.list_evaluation_jobs = AsyncMock(return_value=[])
            mock_pg.list_training_jobs = AsyncMock(return_value=[])
            mock_cache.cache = {}
            # Cause an unexpected non-HTTP exception after the guards pass
            mock_uuid.uuid4.side_effect = RuntimeError("unexpected")

            client = TestClient(app)
            resp = client.post("/api/v1/evaluation/start", json={
                "training_job_id": TRAINING_JOB_ID,
                "val_dataset_id": VAL_DATASET_ID,
            })

        assert resp.status_code == 500


class TestFullEvalFlowWithMockSubprocess:
    """
    Full flow: start → run_evaluation (with mock subprocess) → status → results.
    Verifies that the accuracy output matches expectations from mock inference data.
    """

    @pytest.mark.asyncio
    async def test_full_eval_flow_accuracy_matches_mock_data(self, tmp_path):
        """
        Given: mock training job, mock MCQ JSON, mock inference results
        When: run_evaluation is called with a mock subprocess that writes fake inference JSON
        Then: overall_accuracy == 1.0, per_action matches EXPECTED_PER_ACTION
        """
        import asyncio
        eval_job_id = "eval-full-flow-1"
        output_dir = str(tmp_path / eval_job_id)
        os.makedirs(output_dir)
        log_path = os.path.join(output_dir, "log.txt")
        Path(log_path).write_text("")

        # Write mock inference results that the fake subprocess will produce
        inference_json_path = os.path.join(output_dir, "inference_results.json")

        def write_fake_results(*args, **kwargs):
            """Called when subprocess 'runs' — writes fake inference JSON."""
            Path(inference_json_path).write_text(json.dumps(INFERENCE_RESULTS))

        mock_process = MagicMock()
        mock_process.pid = 9999
        mock_process.wait = AsyncMock(return_value=0)

        mock_process.stdout = MagicMock()
        mock_process.stdout.readline = AsyncMock(return_value=b"")
        mock_process.stderr = MagicMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")

        # Write actions.json to original dataset path
        orig_dir = tmp_path / ORIGINAL_DATASET_ID
        orig_dir.mkdir(parents=True, exist_ok=True)
        (orig_dir / "actions.json").write_text(json.dumps(ACTIONS_DATA))

        final_result = {}

        cached_job = {
            "eval_job_id": eval_job_id,
            "training_job_id": TRAINING_JOB_ID,
            "val_dataset_id": VAL_DATASET_ID,
            "checkpoint_step": CHECKPOINT_STEP,
            "status": "running",
            "overall_accuracy": None,
            "results_json": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "log_file_path": log_path,
            "process_pid": None,
        }

        with patch("app.eval_jobs_cache") as mock_cache, \
             patch("app.postgres_db") as mock_pg, \
             patch("utils.constant.DATASET_ROOT", str(tmp_path)), \
             patch("utils.constant.RESULTS_ROOT", str(tmp_path)), \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:

            mock_cache.get.return_value = cached_job
            mock_pg.update_evaluation_job = AsyncMock()

            def capture_update(job_id, **kwargs):
                final_result.update(kwargs)
                cached_job.update(kwargs)
            mock_cache.update.side_effect = lambda jid, **kw: capture_update(jid, **kw)

            def fake_exec_sync(*args, **kwargs):
                write_fake_results()
                return mock_process

            mock_exec.side_effect = fake_exec_sync

            from app import run_evaluation
            await run_evaluation(
                eval_job_id=eval_job_id,
                training_job_id=TRAINING_JOB_ID,
                actions_json_path=str(tmp_path / ORIGINAL_DATASET_ID / "actions.json"),
                val_dataset_id=VAL_DATASET_ID,
                checkpoint_path=f"/fake/step_{CHECKPOINT_STEP}",
                checkpoint_step=CHECKPOINT_STEP,
                fps=8,
                temperature=0.0,
                backend="vllm",
            )

        # Verify the accuracy matches the mock inference data
        assert final_result.get("status") == "completed", f"Got status: {final_result.get('status')}"
        assert final_result.get("overall_accuracy") == EXPECTED_OVERALL_ACCURACY
        results = final_result.get("results_json", {})
        assert results["overall_accuracy"] == EXPECTED_OVERALL_ACCURACY

        per_action = results["per_action"]
        for action_key, expected in EXPECTED_PER_ACTION.items():
            assert action_key in per_action, f"Missing action {action_key} in results"
            assert per_action[action_key]["correct"] == expected["correct"]
            assert per_action[action_key]["total"] == expected["total"]
            assert per_action[action_key]["accuracy"] == expected["accuracy"]

        mock_cache.delete.assert_called_with(eval_job_id)


class TestE2eApiUniformChunking:
    """POST /api/v1/e2e-evaluation/start with chunking_algorithm='uniform' must
    skip DDM job validation and succeed without ddm_training_job_id.

    Mirrors inference-bp's chunking_options.algorithm='uniform' path
    (sop-monitoring-blueprints @ ce71dde) but kept inside our evaluation-ms.
    """

    def test_uniform_start_skips_ddm_validation(self, tmp_path):
        from fastapi.testclient import TestClient
        from app import app

        # Set up the val dataset with one annotated video subdir + actions.json,
        # so collect_annotations finds something and the actions.json check passes.
        val_dir = tmp_path / VAL_DATASET_ID
        val_dir.mkdir(parents=True)
        (val_dir / "actions.json").write_text(json.dumps(ACTIONS_DATA))
        sub = val_dir / "vid1"
        sub.mkdir()
        (sub / "vid1_annotation.json").write_text(json.dumps([
            {"description": "(1) one", "start_timestamp": 0.0, "end_timestamp": 5.0}
        ]))

        # Create checkpoint dir for the VLM training job
        checkpoint_dir = tmp_path / TRAINING_JOB_ID / f"step_{CHECKPOINT_STEP}"
        checkpoint_dir.mkdir(parents=True)

        cached_job_state = {}

        def fake_set(job_id, data):
            cached_job_state[job_id] = data

        def fake_get(job_id):
            return cached_job_state.get(job_id)

        with patch("app.postgres_db") as mock_pg, \
             patch("app.const") as mock_const, \
             patch("app.e2e_eval_jobs_cache") as mock_cache, \
             patch("app.run_e2e_evaluation", new_callable=AsyncMock), \
             patch("utils.constant.RESULTS_ROOT", str(tmp_path)), \
             patch("utils.constant.DATASET_ROOT", str(tmp_path)):
            mock_const.RUNNING_STATUS = "running"
            mock_const.QUEUE_STATUS = "queued"
            mock_const.COMPLETED_STATUS = "completed"
            mock_const.DATASET_ROOT = str(tmp_path)
            mock_const.RESULTS_ROOT = str(tmp_path)
            mock_const.LOG_FILENAME = "log.txt"
            mock_pg.get_training_job = AsyncMock(return_value=make_mock_training_job())
            # CRITICAL: get_ddm_training_job MUST NOT be called for uniform.
            mock_pg.get_ddm_training_job = AsyncMock(side_effect=AssertionError(
                "DDM job validation should be skipped for uniform chunking"
            ))
            mock_pg.insert_e2e_evaluation_job = AsyncMock(return_value=None)
            mock_pg.list_e2e_evaluation_jobs = AsyncMock(return_value=[])
            mock_cache.cache = {}
            mock_cache.get = MagicMock(side_effect=fake_get)
            mock_cache.set = MagicMock(side_effect=fake_set)
            mock_cache.update = MagicMock()

            client = TestClient(app)
            resp = client.post("/api/v1/e2e-evaluation/start", json={
                "training_job_id": TRAINING_JOB_ID,
                "val_dataset_id": VAL_DATASET_ID,
                "chunking_algorithm": "uniform",
                "chunk_length_sec": 10.0,
            })

        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "eval_job_id" in data
        assert data["status"] == "queued"
        # The DDM job lookup must not have been invoked
        mock_pg.get_ddm_training_job.assert_not_called()

    def test_uniform_start_persists_chunking_columns(self, tmp_path):
        """The new DB columns (chunking_algorithm, chunk_length_sec) must land
        in the insert payload."""
        from fastapi.testclient import TestClient
        from app import app

        val_dir = tmp_path / VAL_DATASET_ID
        val_dir.mkdir(parents=True)
        (val_dir / "actions.json").write_text(json.dumps(ACTIONS_DATA))
        sub = val_dir / "vid1"
        sub.mkdir()
        (sub / "vid1_annotation.json").write_text(json.dumps([
            {"description": "(1) one", "start_timestamp": 0.0, "end_timestamp": 5.0}
        ]))

        checkpoint_dir = tmp_path / TRAINING_JOB_ID / f"step_{CHECKPOINT_STEP}"
        checkpoint_dir.mkdir(parents=True)

        cached_job_state = {}
        captured_insert_kwargs = {}

        def fake_set(job_id, data):
            cached_job_state[job_id] = data

        def fake_get(job_id):
            return cached_job_state.get(job_id)

        async def fake_insert(**kwargs):
            captured_insert_kwargs.update(kwargs)

        with patch("app.postgres_db") as mock_pg, \
             patch("app.const") as mock_const, \
             patch("app.e2e_eval_jobs_cache") as mock_cache, \
             patch("app.run_e2e_evaluation", new_callable=AsyncMock), \
             patch("utils.constant.RESULTS_ROOT", str(tmp_path)), \
             patch("utils.constant.DATASET_ROOT", str(tmp_path)):
            mock_const.RUNNING_STATUS = "running"
            mock_const.QUEUE_STATUS = "queued"
            mock_const.COMPLETED_STATUS = "completed"
            mock_const.DATASET_ROOT = str(tmp_path)
            mock_const.RESULTS_ROOT = str(tmp_path)
            mock_const.LOG_FILENAME = "log.txt"
            mock_pg.get_training_job = AsyncMock(return_value=make_mock_training_job())
            mock_pg.insert_e2e_evaluation_job = AsyncMock(side_effect=fake_insert)
            mock_pg.list_e2e_evaluation_jobs = AsyncMock(return_value=[])
            mock_cache.cache = {}
            mock_cache.get = MagicMock(side_effect=fake_get)
            mock_cache.set = MagicMock(side_effect=fake_set)
            mock_cache.update = MagicMock()

            client = TestClient(app)
            resp = client.post("/api/v1/e2e-evaluation/start", json={
                "training_job_id": TRAINING_JOB_ID,
                "val_dataset_id": VAL_DATASET_ID,
                "chunking_algorithm": "uniform",
                "chunk_length_sec": 7.5,
            })

        assert resp.status_code == 200, resp.text
        assert captured_insert_kwargs.get("chunking_algorithm") == "uniform"
        assert captured_insert_kwargs.get("chunk_length_sec") == 7.5

    def test_uniform_without_chunk_length_rejected_at_api(self, tmp_path):
        """Pydantic-level rejection (422): uniform requires chunk_length_sec."""
        from fastapi.testclient import TestClient
        from app import app

        with patch("app.e2e_eval_jobs_cache") as mock_cache, \
             patch("app.postgres_db") as mock_pg:
            mock_pg.list_e2e_evaluation_jobs = AsyncMock(return_value=[])
            mock_cache.cache = {}

            client = TestClient(app)
            resp = client.post("/api/v1/e2e-evaluation/start", json={
                "training_job_id": TRAINING_JOB_ID,
                "val_dataset_id": VAL_DATASET_ID,
                "chunking_algorithm": "uniform",
                # chunk_length_sec missing → 422
            })

        assert resp.status_code == 422

    def test_ddm_default_still_validates_ddm_job(self, tmp_path):
        """Backwards-compat: legacy ddm path keeps the existing DDM-validation
        branch (no regression)."""
        from fastapi.testclient import TestClient
        from app import app

        with patch("app.e2e_eval_jobs_cache") as mock_cache, \
             patch("app.postgres_db") as mock_pg, \
             patch("app.const") as mock_const:
            mock_const.RUNNING_STATUS = "running"
            mock_const.COMPLETED_STATUS = "completed"
            mock_pg.get_training_job = AsyncMock(return_value=make_mock_training_job())
            mock_pg.get_ddm_training_job = AsyncMock(return_value=None)
            mock_pg.list_e2e_evaluation_jobs = AsyncMock(return_value=[])
            mock_cache.cache = {}

            client = TestClient(app)
            resp = client.post("/api/v1/e2e-evaluation/start", json={
                "training_job_id": TRAINING_JOB_ID,
                "ddm_training_job_id": "no-such-ddm-job",
                "val_dataset_id": VAL_DATASET_ID,
            })

        # 404 because the DDM job lookup returned None
        assert resp.status_code == 404
        mock_pg.get_ddm_training_job.assert_called_once()
