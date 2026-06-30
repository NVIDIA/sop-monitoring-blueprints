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
Unit tests for evaluation-ms app.py REST endpoints and GPU helpers
(the pieces not exercised by test_app_eval.py, which covers the background tasks).
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MS_PATH = str(PROJECT_ROOT / "microservices" / "evaluation-ms")
if MS_PATH not in sys.path:
    sys.path.insert(0, MS_PATH)

with patch("components.postgres_db.postgres_db"):
    from fastapi.testclient import TestClient
    import app as appmod

client = TestClient(appmod.app)


# --------------------------------------------------------------------------- #
# GPU helper functions (pure logic)
# --------------------------------------------------------------------------- #
class TestValidateGpuId:
    def test_none_returns_none(self):
        assert appmod._validate_gpu_id(None) is None

    def test_out_of_range_raises_400(self):
        # torch is not installed in CI -> visible count is 0 -> any id is out of range.
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            appmod._validate_gpu_id(0)
        assert exc.value.status_code == 400

    def test_valid_id_passes_through(self):
        # Force a visible device count of 2 so id=1 validates.
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.device_count.return_value = 2
        with patch.dict(sys.modules, {"torch": fake_torch}):
            assert appmod._validate_gpu_id(1) == 1


class TestSubprocessEnvForGpu:
    def test_none_returns_none(self):
        assert appmod._subprocess_env_for_gpu(None) is None

    def test_sets_cuda_visible_devices(self):
        env = appmod._subprocess_env_for_gpu(3)
        assert env is not None
        assert env["CUDA_VISIBLE_DEVICES"] == "3"


# --------------------------------------------------------------------------- #
# /api/v1/gpus
# --------------------------------------------------------------------------- #
class TestListGpus:
    def test_returns_empty_when_torch_unavailable(self):
        # No torch in the test env -> the import-guard path returns an empty list.
        resp = client.get("/api/v1/gpus")
        assert resp.status_code == 200
        assert resp.json() == {"count": 0, "gpus": []}

    def test_enumerates_devices_with_fake_torch(self):
        """Inject a fake torch so the device-enumeration branch is exercised."""
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.device_count.return_value = 2
        prop = MagicMock()
        prop.name = "A100"
        prop.total_memory = 80 * 1024 * 1024 * 1024
        fake_torch.cuda.get_device_properties.return_value = prop

        # pynvml is not installed in CI, so the free-memory lookup is skipped
        # (free_memory_mb stays None) -- that's the realistic path.
        with patch.dict(sys.modules, {"torch": fake_torch}):
            resp = client.get("/api/v1/gpus")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["gpus"]) == 2
        assert data["gpus"][0]["name"] == "A100"
        assert data["gpus"][0]["total_memory_mb"] == 80 * 1024
        assert data["gpus"][0]["free_memory_mb"] is None


# --------------------------------------------------------------------------- #
# Evaluation status / results / all_jobs / cancel
# --------------------------------------------------------------------------- #
def _job(**over):
    base = {
        "training_job_id": "tj-1",
        "val_dataset_id": "vd-1",
        "status": "running",
        "overall_accuracy": None,
        "checkpoint_step": 100,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }
    base.update(over)
    return base


class TestGetEvaluationStatus:
    def test_cache_hit(self):
        with patch.object(appmod, "eval_jobs_cache") as cache:
            cache.get.return_value = _job(status="completed", overall_accuracy=0.9)
            resp = client.get("/api/v1/evaluation/status/e1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_db_fallback(self):
        db_job = MagicMock()
        db_job.to_dict.return_value = _job(status="failed")
        with patch.object(appmod, "eval_jobs_cache") as cache, \
             patch.object(appmod, "postgres_db") as pg:
            cache.get.return_value = None
            pg.get_evaluation_job = AsyncMock(return_value=db_job)
            resp = client.get("/api/v1/evaluation/status/e1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"

    def test_not_found_404(self):
        with patch.object(appmod, "eval_jobs_cache") as cache, \
             patch.object(appmod, "postgres_db") as pg:
            cache.get.return_value = None
            pg.get_evaluation_job = AsyncMock(return_value=None)
            resp = client.get("/api/v1/evaluation/status/missing")
        assert resp.status_code == 404


class TestGetEvaluationResults:
    def test_completed_returns_results(self):
        db_job = MagicMock()
        db_job.status = appmod.const.COMPLETED_STATUS
        db_job.results_json = {"overall_accuracy": 1.0}
        with patch.object(appmod, "postgres_db") as pg:
            pg.get_evaluation_job = AsyncMock(return_value=db_job)
            resp = client.get("/api/v1/evaluation/results/e1")
        assert resp.status_code == 200
        assert resp.json() == {"overall_accuracy": 1.0}

    def test_not_completed_400(self):
        db_job = MagicMock()
        db_job.status = "running"
        with patch.object(appmod, "postgres_db") as pg:
            pg.get_evaluation_job = AsyncMock(return_value=db_job)
            resp = client.get("/api/v1/evaluation/results/e1")
        assert resp.status_code == 400

    def test_not_found_404(self):
        with patch.object(appmod, "postgres_db") as pg:
            pg.get_evaluation_job = AsyncMock(return_value=None)
            resp = client.get("/api/v1/evaluation/results/missing")
        assert resp.status_code == 404


class TestGetAllEvaluationJobs:
    def test_lists_jobs_keyed_by_id(self):
        job = MagicMock()
        job.id = "e1"
        job.to_dict.return_value = {"status": "completed"}
        with patch.object(appmod, "postgres_db") as pg:
            pg.list_evaluation_jobs = AsyncMock(return_value=[job])
            resp = client.get("/api/v1/evaluation/all_jobs")
        assert resp.status_code == 200
        assert resp.json() == {"e1": {"status": "completed"}}


class TestCancelEvaluation:
    def test_not_found_404(self):
        with patch.object(appmod, "eval_jobs_cache") as cache:
            cache.get.return_value = None
            resp = client.post("/api/v1/evaluation/cancel/missing")
        assert resp.status_code == 404

    def test_not_started_yet(self):
        with patch.object(appmod, "eval_jobs_cache") as cache:
            cache.get.return_value = {"process_pid": None}
            resp = client.post("/api/v1/evaluation/cancel/e1")
        assert resp.status_code == 200
        assert "has not started yet" in resp.json()["message"]

    def test_cancel_success(self):
        with patch.object(appmod, "eval_jobs_cache") as cache, \
             patch.object(appmod, "postgres_db") as pg, \
             patch.object(appmod, "terminate_process_tree", return_value=True):
            cache.get.return_value = {"process_pid": 1234}
            pg.update_evaluation_job = AsyncMock()
            resp = client.post("/api/v1/evaluation/cancel/e1")
        assert resp.status_code == 200
        assert "cancelled successfully" in resp.json()["message"]


# --------------------------------------------------------------------------- #
# E2E evaluation status / results / all_jobs / cancel
# --------------------------------------------------------------------------- #
def _e2e_job(**over):
    base = _job()
    base.update({"ddm_training_job_id": "dj-1", "avg_f1": None})
    base.update(over)
    return base


class TestGetE2eEvaluationStatus:
    def test_cache_hit(self):
        with patch.object(appmod, "e2e_eval_jobs_cache") as cache:
            cache.get.return_value = _e2e_job(status="completed", overall_accuracy=0.8, avg_f1=0.9)
            resp = client.get("/api/v1/e2e-evaluation/status/e1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_not_found_404(self):
        with patch.object(appmod, "e2e_eval_jobs_cache") as cache, \
             patch.object(appmod, "postgres_db") as pg:
            cache.get.return_value = None
            pg.get_e2e_evaluation_job = AsyncMock(return_value=None)
            resp = client.get("/api/v1/e2e-evaluation/status/missing")
        assert resp.status_code == 404


class TestGetE2eEvaluationResults:
    def test_completed_returns_results(self):
        db_job = MagicMock()
        db_job.status = appmod.const.COMPLETED_STATUS
        db_job.results_json = {"action_recognition": {"overall_accuracy": 1.0}}
        with patch.object(appmod, "postgres_db") as pg:
            pg.get_e2e_evaluation_job = AsyncMock(return_value=db_job)
            resp = client.get("/api/v1/e2e-evaluation/results/e1")
        assert resp.status_code == 200

    def test_not_found_404(self):
        with patch.object(appmod, "postgres_db") as pg:
            pg.get_e2e_evaluation_job = AsyncMock(return_value=None)
            resp = client.get("/api/v1/e2e-evaluation/results/missing")
        assert resp.status_code == 404


class TestGetAllE2eEvaluationJobs:
    def test_lists_jobs(self):
        job = MagicMock()
        job.id = "e1"
        job.to_dict.return_value = {"status": "completed"}
        with patch.object(appmod, "postgres_db") as pg:
            pg.list_e2e_evaluation_jobs = AsyncMock(return_value=[job])
            resp = client.get("/api/v1/e2e-evaluation/all_jobs")
        assert resp.status_code == 200
        assert resp.json() == {"e1": {"status": "completed"}}


class TestCancelE2eEvaluation:
    def test_not_found_404(self):
        with patch.object(appmod, "e2e_eval_jobs_cache") as cache:
            cache.get.return_value = None
            resp = client.post("/api/v1/e2e-evaluation/cancel/missing")
        assert resp.status_code == 404

    def test_cancel_success(self):
        with patch.object(appmod, "e2e_eval_jobs_cache") as cache, \
             patch.object(appmod, "postgres_db") as pg, \
             patch.object(appmod, "terminate_process_tree", return_value=True):
            cache.get.return_value = {"process_pid": 4321}
            pg.update_e2e_evaluation_job = AsyncMock()
            resp = client.post("/api/v1/e2e-evaluation/cancel/e1")
        assert resp.status_code == 200
        assert "cancelled successfully" in resp.json()["message"]


# --------------------------------------------------------------------------- #
# start handlers (called directly with mocked deps, like test_app_eval.py)
# --------------------------------------------------------------------------- #
def _completed_training_job():
    job = MagicMock()
    job.status = appmod.const.COMPLETED_STATUS
    job.aug_dataset_id = "aug-1"
    return job


class TestStartEvaluationErrorBranches:
    @pytest.mark.asyncio
    async def test_missing_original_dataset_returns_400(self):
        """get_original_dataset_id returning falsy -> 400 (covers that branch)."""
        from fastapi import HTTPException
        req = MagicMock(gpu_id=None, training_job_id="tj-1")
        with patch.object(appmod, "eval_jobs_cache") as cache, \
             patch.object(appmod, "postgres_db") as pg:
            cache.cache = {}
            pg.get_training_job = AsyncMock(return_value=_completed_training_job())
            pg.get_original_dataset_id = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc:
                await appmod.start_evaluation(req, MagicMock())
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_path_traversal_returns_400(self):
        """safe_dataset_path raising ValueError -> 400 (covers the except branch)."""
        from fastapi import HTTPException
        req = MagicMock(gpu_id=None, training_job_id="tj-1", val_dataset_id="../evil")
        with patch.object(appmod, "eval_jobs_cache") as cache, \
             patch.object(appmod, "postgres_db") as pg, \
             patch.object(appmod, "safe_dataset_path", side_effect=ValueError("bad path")):
            cache.cache = {}
            pg.get_training_job = AsyncMock(return_value=_completed_training_job())
            pg.get_original_dataset_id = AsyncMock(return_value="orig-1")
            with pytest.raises(HTTPException) as exc:
                await appmod.start_evaluation(req, MagicMock())
        assert exc.value.status_code == 400


class TestStartE2eEvaluationSuccess:
    @pytest.mark.asyncio
    async def test_uniform_success_queues_job(self):
        """Uniform-chunking success path: skips DDM, queues the job, schedules the task."""
        req = MagicMock(
            gpu_id=None, training_job_id="tj-1", ddm_training_job_id=None,
            chunking_algorithm="uniform", val_dataset_id="vd-1", checkpoint_step=None,
            fps=8, temperature=0.0, top_p=1.0, backend="vllm", score_threshold=0.5,
            nms_sec=0.0, ddm_batch_size=8, frames_per_segment_hint=256,
            chunk_length_sec=10.0, resolution_config=None,
        )
        bg = MagicMock()

        with patch.object(appmod, "e2e_eval_jobs_cache") as cache, \
             patch.object(appmod, "postgres_db") as pg, \
             patch.object(appmod, "safe_dataset_path", return_value="/ds/vd-1"), \
             patch.object(appmod, "resolve_checkpoint_path", return_value=("/ckpt/step_500", 500)), \
             patch.object(appmod, "extract_mcq_data", return_value=("prompt text", ["(1) a"])), \
             patch.object(appmod, "prepare_eval_assets"), \
             patch.object(appmod, "create_file"), \
             patch("sop.sop_e2e_eval.collect_annotations", return_value={"v1": [{"start": 0}]}), \
             patch("os.path.exists", return_value=True), \
             patch("os.makedirs"), \
             patch("builtins.open", MagicMock()), \
             patch("json.dump"):
            cache.cache = {}
            pg.get_training_job = AsyncMock(return_value=_completed_training_job())
            pg.insert_e2e_evaluation_job = AsyncMock()

            resp = await appmod.start_e2e_evaluation(req, bg)

        assert resp.status == appmod.const.QUEUE_STATUS
        bg.add_task.assert_called_once()
        cache.set.assert_called_once()
