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
Integration tests for data-generation-pipeline API endpoints.

These tests verify the API contract and behavior using FastAPI's TestClient.
They use mocked dependencies to avoid requiring actual database/filesystem resources.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# We need to mock the database and logger before importing the app
with patch("components.postgres_db.postgres_db") as mock_db:
    mock_db.insert_data = AsyncMock(return_value=None)
    mock_db.get_data = AsyncMock(return_value=None)
    mock_db.list_data = AsyncMock(return_value=[])
    mock_db.update_data = AsyncMock(return_value=None)
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
        assert response.json()["status"] == "healthy"


class TestGetAllAugmentedDatasets:
    """Tests for the get all augmented datasets endpoint."""

    def test_get_all_augmented_datasets_empty(self, client):
        """Test getting all datasets when none exist."""
        with patch("app.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []

            response = client.get("/api/v1/augmented_datasets")

            assert response.status_code == 200
            assert response.json() == {}

    def test_get_all_augmented_datasets_with_records(self, client):
        """Test getting all datasets when datasets exist."""
        # Create mock augmentation dataset
        mock_dataset = MagicMock()
        mock_dataset.id = "dataset-aug-001"
        mock_dataset.dataset_id = "original-dataset"
        mock_dataset.status = "completed"

        # Create mock video
        mock_video = MagicMock()
        mock_video.id = "video-001"

        # Create mock chunks
        mock_chunk1 = MagicMock()
        mock_chunk2 = MagicMock()

        with patch("app.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
            # Configure mock to return different values based on schema
            def list_data_side_effect(schema, condition=None):
                schema_name = schema.__name__ if hasattr(schema, "__name__") else str(schema)
                if "Augmentation" in schema_name:
                    return [mock_dataset]
                elif "Video" in schema_name:
                    return [mock_video]
                elif "Chunk" in schema_name:
                    return [mock_chunk1, mock_chunk2]
                return []

            mock_list.side_effect = list_data_side_effect

            response = client.get("/api/v1/augmented_datasets")

            assert response.status_code == 200
            data = response.json()
            assert "dataset-aug-001" in data
            assert data["dataset-aug-001"]["status"] == "completed"
            assert data["dataset-aug-001"]["video_count"] == 1
            assert data["dataset-aug-001"]["total_clips"] == 2


class TestGetAugmentationStatus:
    """Tests for the augmentation status endpoint."""

    def test_get_status_not_found(self, client):
        """Test getting status for non-existent dataset returns 404."""
        with patch("app.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = client.get("/api/v1/augmentation_status/nonexistent-dataset")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_get_status_success(self, client):
        """Test getting status for existing dataset."""
        # Create mock augmentation
        mock_augmentation = MagicMock()
        mock_augmentation.status = "running"

        # Create mock stages
        mock_stage1 = MagicMock()
        mock_stage1.status = "completed"
        mock_stage2 = MagicMock()
        mock_stage2.status = "running"

        with patch("app.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_augmentation

            with patch("app.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
                mock_list.return_value = [mock_stage1, mock_stage2]

                response = client.get("/api/v1/augmentation_status/test-dataset")

                assert response.status_code == 200
                data = response.json()
                assert data["dataset_id"] == "test-dataset"
                assert data["status"] == "running"
                assert data["progress"] == 50.0  # 1 of 2 stages completed

    def test_get_status_all_stages_completed(self, client):
        """Test getting status when all stages are completed."""
        mock_augmentation = MagicMock()
        mock_augmentation.status = "completed"

        mock_stage1 = MagicMock()
        mock_stage1.status = "completed"
        mock_stage2 = MagicMock()
        mock_stage2.status = "completed"
        mock_stage3 = MagicMock()
        mock_stage3.status = "completed"

        with patch("app.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_augmentation

            with patch("app.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
                mock_list.return_value = [mock_stage1, mock_stage2, mock_stage3]

                response = client.get("/api/v1/augmentation_status/completed-dataset")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "completed"
                assert data["progress"] == 100.0


class TestAugmentEndpoint:
    """Tests for the augment endpoint."""

    def test_augment_label_data_not_found(self, client):
        """Test augment when label data path doesn't exist returns 400."""
        with patch("app.load_config_yaml") as mock_config:
            mock_config.return_value = {"video_extention": "mp4"}

            with patch("app.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
                mock_list.return_value = []

                with patch("os.path.exists") as mock_exists:
                    mock_exists.return_value = False

                    response = client.post(
                        "/api/v1/augment",
                        params={"label_data_id": "nonexistent-data"}
                    )

                    assert response.status_code == 400
                    assert "not found" in response.json()["detail"].lower()

    def test_augment_success(self, client):
        """Test successful augmentation request."""
        mock_config = {
            "video_extention": "mp4",
            "bcq": {"enable": True, "subject": "operator"},
        }

        with patch("app.load_config_yaml", return_value=mock_config):
            with patch("app.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
                mock_list.return_value = []

                with patch("os.path.exists", return_value=True):
                    # Pre-flight video-folder validation passes (folders present).
                    with patch("app.vlm_service.find_video_folders", return_value=["/fake/video1"]):
                        with patch("app.clean_and_create_dir"):
                            with patch("app.postgres_db.insert_data", new_callable=AsyncMock):
                                with patch("app.postgres_db.update_data", new_callable=AsyncMock):
                                    with patch("asyncio.create_task"):
                                        response = client.post(
                                            "/api/v1/augment",
                                            params={"label_data_id": "test-label-data"}
                                        )

                                        assert response.status_code == 200
                                        data = response.json()
                                        assert "dataset_id" in data
                                        assert "test-label-data" in data["dataset_id"]

    def test_augment_no_video_folders_returns_400(self, client):
        """Pre-flight rejects a dataset with no video folders synchronously,
        before any DB row or async task is created."""
        mock_config = {"video_extention": "mp4"}

        with patch("app.load_config_yaml", return_value=mock_config):
            with patch("app.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
                mock_list.return_value = []

                with patch("os.path.exists", return_value=True):
                    # Label data + actions.json exist, but no video folders inside.
                    with patch("app.vlm_service.find_video_folders", return_value=[]):
                        with patch("app.postgres_db.insert_data", new_callable=AsyncMock) as mock_insert:
                            with patch("asyncio.create_task") as mock_task:
                                response = client.post(
                                    "/api/v1/augment",
                                    params={"label_data_id": "no-videos"}
                                )

        assert response.status_code == 400
        assert "no video folders" in response.json()["detail"].lower()
        # Fail-fast: no DB row inserted and no background task spawned.
        mock_insert.assert_not_called()
        mock_task.assert_not_called()

    def test_augment_actions_json_not_found(self, client):
        """Test augment when actions.json doesn't exist returns 400."""
        mock_config = {"video_extention": "mp4"}

        with patch("app.load_config_yaml", return_value=mock_config):
            with patch("app.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
                mock_list.return_value = []

                def exists_side_effect(path):
                    # Label data path exists, but actions.json doesn't
                    if "actions.json" in path or "sop_actions.json" in path:
                        return False
                    return True

                with patch("os.path.exists", side_effect=exists_side_effect):
                    response = client.post(
                        "/api/v1/augment",
                        params={"label_data_id": "data-without-actions"}
                    )

                    assert response.status_code == 400
                    assert "not found" in response.json()["detail"].lower()

    def test_augment_unexpected_error_returns_500(self, client):
        """Test that unexpected errors in augment endpoint return 500."""
        mock_config = {
            "video_extention": "mp4",
            "bcq": {"enable": True, "subject": "operator"},
        }

        with patch("app.load_config_yaml", return_value=mock_config):
            with patch("app.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
                mock_list.return_value = []

                with patch("os.path.exists", return_value=True):
                    with patch("app.clean_and_create_dir"):
                        # Make insert_data fail unexpectedly
                        with patch(
                            "app.postgres_db.insert_data",
                            new_callable=AsyncMock,
                            side_effect=Exception("Unexpected DB error"),
                        ):
                            with patch("app.postgres_db.update_data", new_callable=AsyncMock):
                                response = client.post(
                                    "/api/v1/augment",
                                    params={"label_data_id": "test-data"}
                                )

                                assert response.status_code == 500
                                assert "Internal server error" in response.json()["detail"]


class TestGetAllAugmentedDatasetsErrors:
    """Tests for error handling in get_all_augmented_datasets endpoint."""

    def test_get_all_augmented_datasets_unexpected_error_returns_500(self, client):
        """Test that unexpected errors return 500."""
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.id = "dataset-aug-001"
        mock_dataset.dataset_id = "original-dataset"
        mock_dataset.status = "completed"

        with patch("app.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
            # First call returns datasets, second call (for videos) raises an error
            def list_data_side_effect(schema, condition=None):
                schema_name = schema.__name__ if hasattr(schema, "__name__") else str(schema)
                if "Augmentation" in schema_name:
                    return [mock_dataset]
                raise Exception("Database connection error")

            mock_list.side_effect = list_data_side_effect

            response = client.get("/api/v1/augmented_datasets")

            assert response.status_code == 500
            assert "Internal server error" in response.json()["detail"]


class TestGetAugmentationStatusErrors:
    """Tests for error handling in get_augmentation_status endpoint."""

    def test_get_status_unexpected_error_returns_500(self, client):
        """Test that unexpected errors return 500."""
        with patch("app.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            # Return a valid augmentation first, but then list_data fails
            mock_augmentation = MagicMock()
            mock_augmentation.status = "running"
            mock_get.return_value = mock_augmentation

            with patch("app.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
                mock_list.side_effect = Exception("Database error")

                response = client.get("/api/v1/augmentation_status/test-dataset")

                assert response.status_code == 500
                assert "Internal server error" in response.json()["detail"]
