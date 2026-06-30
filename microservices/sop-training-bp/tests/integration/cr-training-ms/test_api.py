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
Integration tests for cr-training-ms API endpoints.

These tests verify the API contract and behavior using FastAPI's TestClient.
They use mocked dependencies to avoid requiring actual database/GPU resources.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# We need to mock the database before importing the app
with patch("components.postgres_db.postgres_db") as mock_db:
    mock_db.insert_training_job = AsyncMock(return_value=None)
    mock_db.get_training_job = AsyncMock(return_value=None)
    mock_db.list_training_jobs = AsyncMock(return_value=[])
    from app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check_returns_200(self, client):
        """Test that health endpoint returns 200 OK."""
        response = client.get("/health")

        assert response.status_code == 200
        assert "message" in response.json()
        assert "running" in response.json()["message"].lower()


class TestGetAllJobs:
    """Tests for the get all jobs endpoint."""

    def test_get_all_jobs_empty(self, client):
        """Test getting all jobs when none exist."""
        with patch("app.postgres_db.list_training_jobs", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []

            response = client.get("/api/v1/fine-tuning/all_jobs")

            assert response.status_code == 200
            assert response.json() == {}

    def test_get_all_jobs_with_records(self, client):
        """Test getting all jobs when jobs exist."""
        from datetime import datetime

        # Create mock job objects with id attribute and to_dict method
        mock_job1 = MagicMock()
        mock_job1.id = "job-123"
        mock_job1.to_dict.return_value = {
            "job_id": "job-123",
            "status": "completed",
            "progress": 100.0,
            "created_at": datetime(2026, 1, 15, 10, 0, 0).isoformat(),
        }

        mock_job2 = MagicMock()
        mock_job2.id = "job-456"
        mock_job2.to_dict.return_value = {
            "job_id": "job-456",
            "status": "running",
            "progress": 50.0,
            "created_at": datetime(2026, 1, 16, 12, 0, 0).isoformat(),
        }

        with patch("app.postgres_db.list_training_jobs", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = [mock_job1, mock_job2]

            response = client.get("/api/v1/fine-tuning/all_jobs")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert "job-123" in data
            assert "job-456" in data
            assert data["job-123"]["status"] == "completed"
            assert data["job-456"]["status"] == "running"
            assert data["job-456"]["progress"] == 50.0


class TestStartFineTuning:
    """Tests for the start fine-tuning endpoint."""

    def test_start_fine_tuning_success(self, client):
        """Test starting a fine-tuning job successfully."""
        with patch("app.training_jobs_cache") as mock_cache:
            # No running jobs
            mock_cache.cache.values.return_value = []
            mock_cache.get.return_value = {
                "job_id": "test-job-id",
                "status": "queued",
                "progress": 0.0,
                "created_at": "2026-01-15T10:00:00",
            }

            with patch("app.create_file"):
                with patch("app.postgres_db.insert_training_job", new_callable=AsyncMock):
                    # Mock run_fine_tuning to prevent background task from actually running
                    with patch("app.run_fine_tuning", new_callable=AsyncMock):
                        response = client.post(
                            "/api/v1/fine-tuning/start",
                            params={"dataset_id": "test-dataset"}
                        )

                        assert response.status_code == 200
                        data = response.json()
                        assert "job_id" in data
                        assert data["status"] == "queued"
                        assert "message" in data

    def test_start_fine_tuning_job_already_running(self, client):
        """Test starting a job when another is already running returns 400."""
        running_job = {
            "job_id": "existing-job-123",
            "status": "running",
        }

        with patch("app.training_jobs_cache") as mock_cache:
            mock_cache.cache.values.return_value = [running_job]

            response = client.post(
                "/api/v1/fine-tuning/start",
                params={"dataset_id": "test-dataset"}
            )

            assert response.status_code == 400
            assert "already running" in response.json()["detail"].lower()
            assert "existing-job-123" in response.json()["detail"]

    def test_start_fine_tuning_database_error(self, client):
        """Test starting a job when database insert fails returns 500."""
        with patch("app.training_jobs_cache") as mock_cache:
            mock_cache.cache.values.return_value = []
            mock_cache.get.return_value = {"status": "queued"}

            with patch("app.create_file"):
                with patch(
                    "app.postgres_db.insert_training_job",
                    new_callable=AsyncMock,
                    side_effect=Exception("Database connection failed")
                ):
                    response = client.post(
                        "/api/v1/fine-tuning/start",
                        params={"dataset_id": "test-dataset"}
                    )

                    assert response.status_code == 500
                    assert "Database connection failed" in response.json()["detail"]


class TestGetTrainingStatus:
    """Tests for the training status endpoint."""

    def test_get_status_job_not_found(self, client):
        """Test getting status for non-existent job returns 404."""
        with patch("app.training_jobs_cache") as mock_cache:
            mock_cache.get.return_value = None
            with patch("app.postgres_db.get_training_job", new_callable=AsyncMock) as mock_db_get:
                mock_db_get.return_value = None

                response = client.get("/api/v1/fine-tuning/status/nonexistent-job")

                assert response.status_code == 404
                assert "Training job not found" in response.json()["detail"]

    def test_get_status_from_cache(self, client):
        """Test getting status from cache."""
        from datetime import datetime

        cached_job = {
            "job_id": "test-job-123",
            "status": "running",
            "progress": 50.0,
            "current_step": 50,
            "total_steps": 100,
            "loss": 0.5,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        with patch("app.training_jobs_cache") as mock_cache:
            mock_cache.get.return_value = cached_job

            response = client.get("/api/v1/fine-tuning/status/test-job-123")

            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == "test-job-123"
            assert data["status"] == "running"
            assert data["progress"] == 50.0


class TestCancelFineTuning:
    """Tests for the cancel fine-tuning endpoint."""

    def test_cancel_nonexistent_job_returns_404(self, client):
        """Test cancelling non-existent job returns 404."""
        with patch("app.training_jobs_cache") as mock_cache:
            mock_cache.get.return_value = None

            response = client.post("/api/v1/fine-tuning/cancel/nonexistent-job")

            assert response.status_code == 404

    def test_cancel_job_without_pid_returns_wait_message(self, client):
        """Test cancelling job before process is assigned."""
        from datetime import datetime

        job_without_pid = {
            "job_id": "test-job",
            "status": "queued",
            "process_pid": None,
            "created_at": datetime.now(),
        }

        with patch("app.training_jobs_cache") as mock_cache:
            mock_cache.get.return_value = job_without_pid

            response = client.post("/api/v1/fine-tuning/cancel/test-job")

            assert response.status_code == 200
            assert "wait" in response.json()["message"].lower()

    def test_cancel_job_success(self, client):
        """Test successfully cancelling a running job."""
        from datetime import datetime

        job_with_pid = {
            "job_id": "running-job",
            "status": "running",
            "process_pid": 12345,
            "log_file_path": "/tmp/test_log.txt",
            "created_at": datetime.now(),
        }

        with patch("app.training_jobs_cache") as mock_cache:
            mock_cache.get.return_value = job_with_pid

            with patch("app.terminate_process_tree", return_value=True):
                with patch("os.path.exists", return_value=True):
                    with patch("builtins.open", MagicMock()):
                        response = client.post("/api/v1/fine-tuning/cancel/running-job")

                        assert response.status_code == 200
                        assert "cancelled successfully" in response.json()["message"]
                        mock_cache.update.assert_called()

    def test_cancel_job_failure(self, client):
        """Test failing to cancel a running job."""
        from datetime import datetime

        job_with_pid = {
            "job_id": "running-job",
            "status": "running",
            "process_pid": 12345,
            "created_at": datetime.now(),
        }

        with patch("app.training_jobs_cache") as mock_cache:
            mock_cache.get.return_value = job_with_pid

            with patch("app.terminate_process_tree", return_value=False):
                response = client.post("/api/v1/fine-tuning/cancel/running-job")

                assert response.status_code == 200
                assert "Failed to cancel" in response.json()["message"]

    def test_cancel_job_log_file_not_exists(self, client):
        """Test cancelling job when log file doesn't exist."""
        from datetime import datetime

        job_with_pid = {
            "job_id": "running-job",
            "status": "running",
            "process_pid": 12345,
            "log_file_path": "/nonexistent/log.txt",
            "created_at": datetime.now(),
        }

        with patch("app.training_jobs_cache") as mock_cache:
            mock_cache.get.return_value = job_with_pid

            with patch("app.terminate_process_tree", return_value=True):
                with patch("os.path.exists", return_value=False):
                    response = client.post("/api/v1/fine-tuning/cancel/running-job")

                    assert response.status_code == 200
                    assert "cancelled successfully" in response.json()["message"]


class TestGetTrainingStatusExceptions:
    """Tests for exception handling in training status endpoint."""

    def test_get_status_database_exception(self, client):
        """Test getting status when database throws exception returns 500."""
        with patch("app.training_jobs_cache") as mock_cache:
            mock_cache.get.return_value = None

            with patch("app.postgres_db.get_training_job", new_callable=AsyncMock) as mock_db:
                mock_db.side_effect = Exception("Database connection failed")

                response = client.get("/api/v1/fine-tuning/status/test-job")

                assert response.status_code == 500
                assert "Failed to get training status" in response.json()["detail"]

    def test_get_status_from_database(self, client):
        """Test getting status from database when not in cache."""
        from datetime import datetime

        mock_job = MagicMock()
        mock_job.to_dict.return_value = {
            "job_id": "db-job-123",
            "status": "completed",
            "progress": 100.0,
            "current_step": 100,
            "total_steps": 100,
            "loss": 0.1,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        with patch("app.training_jobs_cache") as mock_cache:
            mock_cache.get.return_value = None

            with patch("app.postgres_db.get_training_job", new_callable=AsyncMock) as mock_db:
                mock_db.return_value = mock_job

                response = client.get("/api/v1/fine-tuning/status/db-job-123")

                assert response.status_code == 200
                data = response.json()
                assert data["job_id"] == "db-job-123"
                assert data["status"] == "completed"
