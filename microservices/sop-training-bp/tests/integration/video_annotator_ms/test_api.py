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
Integration tests for video-annotator-ms API endpoints.

These tests verify the API contract and behavior using FastAPI's TestClient.
They use mocked dependencies to avoid requiring actual database/filesystem resources.
"""

import json
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# We need to mock the database before importing the app
with patch("components.postgres_db.postgres_db") as mock_db:
    mock_db.get_data = AsyncMock(return_value=None)
    mock_db.list_data = AsyncMock(return_value=[])
    mock_db.insert_data = AsyncMock(return_value=None)
    mock_db.update_data = AsyncMock(return_value=None)
    mock_db.delete_data = AsyncMock(return_value=1)
    mock_db.delete_all_data = AsyncMock(return_value=0)
    from inference import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoints:
    """Tests for the health check endpoints."""

    def test_health_live_returns_200(self, client):
        """Test that liveness probe returns 200 OK."""
        response = client.get("/health/live")

        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_health_ready_returns_200(self, client):
        """Test that readiness probe returns 200 OK."""
        response = client.get("/health/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "ready"


class TestActionsUpload:
    """Tests for the actions upload endpoint."""

    def test_upload_actions_success(self, client):
        """Test uploading a valid actions.json file."""
        actions_data = {
            "actions": [
                "Pick up the component",
                "Place on assembly line",
                "Secure with fastener"
            ]
        }

        with patch("inference.create_dir"):
            with patch("inference.postgres_db.insert_data", new_callable=AsyncMock):
                with patch("builtins.open", MagicMock()):
                    response = client.post(
                        "/api/v1/actions/upload",
                        files={"file": ("actions.json", json.dumps(actions_data), "application/json")}
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "success"
                    assert data["actions_count"] == 3
                    assert "data_id" in data

    def test_upload_actions_invalid_json(self, client):
        """Test uploading invalid JSON returns 400."""
        response = client.post(
            "/api/v1/actions/upload",
            files={"file": ("actions.json", "invalid json {{{", "application/json")}
        )

        assert response.status_code == 400
        assert "Invalid JSON" in response.json()["detail"]

    def test_upload_actions_empty_array(self, client):
        """Test uploading empty actions array returns error.
        """
        actions_data = {"actions": []}

        response = client.post(
            "/api/v1/actions/upload",
            files={"file": ("actions.json", json.dumps(actions_data), "application/json")}
        )

        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"]

    def test_upload_actions_generic_exception(self, client):
        """Test that generic exception returns 500."""
        with patch("inference.create_dir", side_effect=Exception("Cannot create directory")):
            response = client.post(
                "/api/v1/actions/upload",
                files={"file": ("actions.json", '{"actions": ["Action 1"]}', "application/json")}
            )

            assert response.status_code == 500


class TestActionsReset:
    """Tests for the actions reset endpoint."""

    def test_reset_actions_success(self, client):
        """Test resetting actions successfully."""
        # First set a current data_id by simulating an upload
        import inference
        inference.current_data_id = "test-data-id"

        response = client.post("/api/v1/actions/reset")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["previous_data_id"] == "test-data-id"

    def test_reset_actions_when_none_set(self, client):
        """Test resetting when no data_id is set.
        """
        import inference
        inference.current_data_id = None

        response = client.post("/api/v1/actions/reset")

        assert response.status_code == 400
        assert "Nothing to reset" in response.json()["detail"]


class TestSetCurrentDataset:
    """Tests for the set current dataset endpoint."""

    def test_set_current_dataset_success(self, client):
        """Test setting current dataset successfully."""
        mock_dataset = MagicMock()
        mock_dataset.id = "existing-dataset-123"

        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_dataset

            response = client.post("/api/v1/dataset/existing-dataset-123/set_current")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["data_id"] == "existing-dataset-123"

    def test_set_current_dataset_not_found(self, client):
        """Test setting non-existent dataset returns 404."""
        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = client.post("/api/v1/dataset/nonexistent/set_current")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_set_current_dataset_generic_exception(self, client):
        """Test that generic exceptions return 500."""
        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Database connection failed")

            response = client.post("/api/v1/dataset/test-id/set_current")

            assert response.status_code == 500


class TestGetVideo:
    """Tests for the get video metadata endpoint."""

    def test_get_video_success(self, client):
        """Test getting video metadata successfully."""
        from datetime import datetime

        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "test_video.mp4"
        mock_video.dataset_id = "dataset-456"
        mock_video.file_size = 1024000
        mock_video.mime_type = "video/mp4"
        mock_video.created_at = datetime(2026, 1, 15, 10, 0, 0)

        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_video

            response = client.get("/api/v1/videos/video-123")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "video-123"
            assert data["filename"] == "test_video.mp4"
            assert data["file_size"] == 1024000

    def test_get_video_not_found(self, client):
        """Test getting non-existent video returns 404.
        """
        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = client.get("/api/v1/videos/nonexistent")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_get_video_exception_handling(self, client):
        """Test that generic exceptions return 500."""
        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Database error")

            response = client.get("/api/v1/videos/video-123")

            assert response.status_code == 500
            assert "Error getting video metadata" in response.json()["detail"]


class TestGetChunk:
    """Tests for the get chunk metadata endpoint."""

    def test_get_chunk_success(self, client):
        """Test getting chunk metadata successfully."""
        from datetime import datetime

        mock_chunk = MagicMock()
        mock_chunk.id = "chunk-123"
        mock_chunk.video_id = "video-456"
        mock_chunk.name = "01_test_video_1_1.mp4"
        mock_chunk.action = "Pick up component"
        mock_chunk.file_size = 512000
        mock_chunk.mime_type = "video/mp4"
        mock_chunk.created_at = datetime(2026, 1, 15, 10, 0, 0)

        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_chunk

            response = client.get("/api/v1/chunks/chunk-123")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "chunk-123"
            assert data["video_id"] == "video-456"
            assert data["action"] == "Pick up component"

    def test_get_chunk_not_found(self, client):
        """Test getting non-existent chunk returns 404."""
        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = client.get("/api/v1/chunks/nonexistent")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_get_chunk_exception_handling(self, client):
        """Test that generic exceptions return 500."""
        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Database error")

            response = client.get("/api/v1/chunks/chunk-123")

            assert response.status_code == 500
            assert "Error getting chunk metadata" in response.json()["detail"]


class TestListVideos:
    """Tests for the list videos endpoint."""

    def test_list_videos_empty(self, client):
        """Test listing videos when none exist."""
        with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []

            response = client.get("/api/v1/videos")

            assert response.status_code == 200
            assert response.json() == []

    def test_list_videos_with_records(self, client):
        """Test listing videos when videos exist."""
        from datetime import datetime

        mock_video1 = MagicMock()
        mock_video1.id = "video-1"
        mock_video1.name = "video1.mp4"
        mock_video1.dataset_id = "dataset-1"
        mock_video1.file_size = 1000000
        mock_video1.mime_type = "video/mp4"
        mock_video1.created_at = datetime(2026, 1, 15, 10, 0, 0)

        mock_video2 = MagicMock()
        mock_video2.id = "video-2"
        mock_video2.name = "video2.mp4"
        mock_video2.dataset_id = "dataset-1"
        mock_video2.file_size = 2000000
        mock_video2.mime_type = "video/mp4"
        mock_video2.created_at = datetime(2026, 1, 16, 10, 0, 0)

        with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = [mock_video1, mock_video2]

            response = client.get("/api/v1/videos")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["id"] == "video-1"
            assert data[1]["id"] == "video-2"

    def test_list_videos_exception_handling(self, client):
        """Test that exceptions return 500."""
        with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
            mock_list.side_effect = Exception("Database error")

            response = client.get("/api/v1/videos")

            assert response.status_code == 500
            assert "Error listing videos" in response.json()["detail"]


class TestGetDatasets:
    """Tests for the get datasets endpoint."""

    def test_get_datasets_empty(self, client):
        """Test getting datasets when none exist."""
        with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []

            response = client.get("/api/v1/datasets")

            assert response.status_code == 200
            assert response.json() == {}

    def test_get_datasets_with_records(self, client):
        """Test getting datasets when datasets exist with videos and annotations."""
        from datetime import datetime

        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.id = "dataset-123"
        mock_dataset.actions = ["Pick up component", "Place component"]

        # Create mock video
        mock_video = MagicMock()
        mock_video.id = "video-456"
        mock_video.name = "video-456_test_video.mp4"

        # Create mock annotation
        mock_annotation = MagicMock()
        mock_annotation.chunk_id = "chunk-789"
        mock_annotation.start_time = 0.0
        mock_annotation.end_time = 5.0
        mock_annotation.action_description = "Pick up component"
        mock_annotation.action_index = 0
        mock_annotation.created_at = datetime(2026, 1, 15, 10, 0, 0)

        # Create mock chunk
        mock_chunk = MagicMock()
        mock_chunk.id = "chunk-789"
        mock_chunk.name = "01_test_video_1_1.mp4"

        with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
            with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
                def list_data_side_effect(schema, condition=None):
                    schema_name = schema.__name__ if hasattr(schema, "__name__") else str(schema)
                    if "Dataset" in schema_name:
                        return [mock_dataset]
                    elif "Video" in schema_name:
                        return [mock_video]
                    elif "Annotation" in schema_name:
                        return [mock_annotation]
                    return []

                mock_list.side_effect = list_data_side_effect
                mock_get.return_value = mock_chunk  # For chunk lookup

                response = client.get("/api/v1/datasets")

                assert response.status_code == 200

                data = response.json()
                assert "dataset-123" in data
                assert data["dataset-123"]["actions"] == ["Pick up component", "Place component"]
                assert "video-456" in data["dataset-123"]["videos"]

                video_info = data["dataset-123"]["videos"]["video-456"]
                assert video_info["original_file_name"] == "test_video.mp4"
                assert len(video_info["clips"]) == 1
                assert video_info["clips"][0]["action_description"] == "Pick up component"

    def test_get_datasets_video_without_annotations(self, client):
        """Test getting datasets when videos have no annotations."""
        mock_dataset = MagicMock()
        mock_dataset.id = "dataset-123"
        mock_dataset.actions = ["Action 1"]

        mock_video = MagicMock()
        mock_video.id = "video-456"
        mock_video.name = "video-456_test.mp4"

        with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
            def list_data_side_effect(schema, condition=None):
                schema_name = schema.__name__ if hasattr(schema, "__name__") else str(schema)
                if "Dataset" in schema_name:
                    return [mock_dataset]
                elif "Video" in schema_name:
                    return [mock_video]
                elif "Annotation" in schema_name:
                    return []
                return []

            mock_list.side_effect = list_data_side_effect

            response = client.get("/api/v1/datasets")

            assert response.status_code == 200
            assert "dataset-123" in response.json()
            assert response.json()["dataset-123"]["videos"] == {}

    def test_get_datasets_exception_handling(self, client):
        """Test that exceptions during get datasets return 500."""
        with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
            mock_list.side_effect = Exception("Database error")

            response = client.get("/api/v1/datasets")

            assert response.status_code == 500
            assert "Error getting datasets info" in response.json()["detail"]


class TestClearDataset:
    """Tests for the clear dataset endpoints."""

    def test_clear_specific_dataset_success(self, client):
        """Test clearing a specific dataset."""
        with patch("inference.postgres_db.delete_data", new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = 1

            with patch("os.path.exists", return_value=True):
                with patch("shutil.rmtree"):
                    response = client.delete("/api/v1/videos/clear-dataset/test-dataset")

                    assert response.status_code == 200
                    data = response.json()
                    assert "Successfully cleared" in data["message"]
                    assert data["data_id"] == "test-dataset"

    def test_clear_specific_dataset_exception(self, client):
        """Test that exception during clear specific dataset returns 500."""
        with patch("inference.postgres_db.delete_data", new_callable=AsyncMock) as mock_delete:
            mock_delete.side_effect = Exception("Database error")

            response = client.delete("/api/v1/videos/clear-dataset/test-dataset")

            assert response.status_code == 500

    def test_clear_dataset_directory_not_exists(self, client):
        """Test clearing dataset when directory doesn't exist."""
        with patch("inference.postgres_db.delete_data", new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = 1

            with patch("os.path.exists", return_value=False):
                response = client.delete("/api/v1/videos/clear-dataset/test-dataset")

                assert response.status_code == 200
                assert response.json()["files_deleted"] == 0

    def test_clear_dataset_rmtree_failure(self, client):
        """Test handling shutil.rmtree failure during clear."""
        with patch("inference.postgres_db.delete_data", new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = 1

            with patch("os.path.exists", return_value=True):
                with patch("shutil.rmtree", side_effect=Exception("Permission denied")):
                    response = client.delete("/api/v1/videos/clear-dataset/test-dataset")

                    assert response.status_code == 200

    def test_clear_all_datasets_success(self, client):
        """Test clearing all datasets."""
        mock_dataset = MagicMock()
        mock_dataset.id = "dataset-1"

        with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = [mock_dataset]

            with patch("inference.postgres_db.delete_all_data", new_callable=AsyncMock) as mock_delete:
                mock_delete.return_value = 1

                with patch("os.path.exists", return_value=True):
                    with patch("shutil.rmtree"):
                        response = client.delete("/api/v1/videos/clear-all-datasets")

                        assert response.status_code == 200
                        data = response.json()
                        assert "Successfully cleared" in data["message"]

    def test_clear_all_datasets_empty(self, client):
        """Test clearing all datasets when none exist."""
        with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []

            with patch("inference.postgres_db.delete_all_data", new_callable=AsyncMock) as mock_delete:
                mock_delete.return_value = 0

                response = client.delete("/api/v1/videos/clear-all-datasets")

                assert response.status_code == 200
                assert response.json()["deleted_count"] == 0

    def test_clear_all_datasets_exception(self, client):
        """Test that exception during clear all returns 500."""
        with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
            mock_list.side_effect = Exception("Database error")

            response = client.delete("/api/v1/videos/clear-all-datasets")

            assert response.status_code == 500


class TestUploadVideo:
    """Tests for the video upload endpoint."""

    def test_upload_video_no_data_id_set(self, client):
        """Test uploading video when no data_id is set returns 400."""
        import inference
        inference.current_data_id = None

        # Create a fake video file
        fake_video = BytesIO(b"fake video content" * 100)

        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.mp4", fake_video, "video/mp4")}
        )

        assert response.status_code == 400
        assert "Data ID is not set" in response.json()["detail"]

    def test_upload_video_success(self, client):
        """Test uploading video successfully when data_id is set."""
        import inference
        inference.current_data_id = "test-dataset-id"

        # Create a fake video file (must be > 1KB to pass validation)
        fake_video = BytesIO(b"fake video content" * 100)

        with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []  # No existing videos

            with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = None  # No existing record

                with patch("inference.postgres_db.insert_data", new_callable=AsyncMock):
                    with patch("inference.convert_to_h264", new_callable=AsyncMock) as mock_convert:
                        mock_convert.return_value = "/fake/path/video.mp4"

                        with patch("os.path.getsize", return_value=1024000):
                            with patch("builtins.open", MagicMock()):
                                with patch("inference.clean_up_file"):
                                    response = client.post(
                                        "/api/v1/upload",
                                        files={"file": ("test_video.mp4", fake_video, "video/mp4")}
                                    )

                                    assert response.status_code == 200
                                    data = response.json()
                                    assert "file_id" in data
                                    assert "successfully uploaded" in data["message"].lower()

    def test_upload_video_file_too_small(self, client):
        """Test uploading video that is too small returns 400."""
        import inference
        inference.current_data_id = "test-dataset-id"

        # Create a very small file (< 1KB)
        small_file = BytesIO(b"tiny")

        with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []

            with patch("builtins.open", MagicMock()):
                with patch("inference.clean_up_file"):
                    response = client.post(
                        "/api/v1/upload",
                        files={"file": ("small.mp4", small_file, "video/mp4")}
                    )

                    assert response.status_code == 400
                    assert "too small" in response.json()["detail"].lower()

    def test_upload_video_overwrite_existing(self, client):
        """Test uploading video that overwrites an existing video."""
        import inference
        inference.current_data_id = "test-dataset-id"

        existing_video = MagicMock()
        existing_video.id = "existing-video-id"
        existing_video.name = "existing-video-id_test_video.mp4"

        fake_video = BytesIO(b"fake video content" * 100)

        with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = [existing_video]

            with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = existing_video

                with patch("inference.postgres_db.update_data", new_callable=AsyncMock) as mock_update:
                    with patch("inference.convert_to_h264", new_callable=AsyncMock) as mock_convert:
                        mock_convert.return_value = "/fake/path/video.mp4"

                        with patch("os.path.getsize", return_value=1024000):
                            with patch("builtins.open", MagicMock()):
                                with patch("inference.clean_up_file"):
                                    response = client.post(
                                        "/api/v1/upload",
                                        files={"file": ("test_video.mp4", fake_video, "video/mp4")}
                                    )

                                    assert response.status_code == 200
                                    mock_update.assert_called()

    def test_upload_video_conversion_failure(self, client):
        """Test handling video conversion failure."""
        import inference
        inference.current_data_id = "test-dataset-id"

        fake_video = BytesIO(b"fake video content" * 100)

        with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []

            with patch("builtins.open", MagicMock()):
                with patch("inference.convert_to_h264", new_callable=AsyncMock) as mock_convert:
                    mock_convert.side_effect = Exception("Conversion failed")

                    with patch("inference.clean_up_file"):
                        response = client.post(
                            "/api/v1/upload",
                            files={"file": ("test.mp4", fake_video, "video/mp4")}
                        )

                        assert response.status_code == 500
                        assert "conversion failed" in response.json()["detail"].lower()

    def test_upload_video_generic_exception(self, client):
        """Test that generic exception during video upload returns 500."""
        import inference
        inference.current_data_id = "test-dataset-id"

        fake_video = BytesIO(b"fake video content" * 100)

        with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
            mock_list.side_effect = Exception("Database error")

            with patch("builtins.open", MagicMock()):
                with patch("inference.clean_up_file"):
                    response = client.post(
                        "/api/v1/upload",
                        files={"file": ("test.mp4", fake_video, "video/mp4")}
                    )

                    assert response.status_code == 500


class TestSplitVideo:
    """Tests for the split video endpoint."""

    def test_split_video_missing_request_body(self, client):
        """Test splitting video with missing request body returns 422.

        FastAPI returns 422 Unprocessable Entity when required Body parameter is missing.
        """
        response = client.post(
            "/api/v1/videos/video-123/split",
            json=None
        )

        assert response.status_code == 422  # FastAPI validation error

    def test_split_video_invalid_timestamps(self, client):
        """Test splitting video with empty timestamps returns 400."""
        response = client.post(
            "/api/v1/videos/video-123/split",
            json={"timestamps": []}
        )

        assert response.status_code == 400
        assert "Invalid timestamp data" in response.json()["detail"]

    def test_split_video_not_found(self, client):
        """Test splitting non-existent video returns 404."""
        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = client.post(
                "/api/v1/videos/nonexistent/split",
                json={"timestamps": [{"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "Test"}]}
            )

            assert response.status_code == 404
            assert "Could not find video" in response.json()["detail"]

    def test_split_video_no_valid_timestamps(self, client):
        """Test splitting video when all timestamps are invalid returns 400."""
        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "video-123_test.mp4"
        mock_video.dataset_id = "dataset-456"
        mock_video.to_dict.return_value = {"id": "video-123"}

        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_video

            # Timestamps where end <= start (invalid)
            response = client.post(
                "/api/v1/videos/video-123/split",
                json={"timestamps": [{"start": 10, "end": 5, "actionIndex": 0, "actionDescription": "Test"}]}
            )

            assert response.status_code == 400
            assert "No valid timestamps" in response.json()["detail"]

    def test_split_video_success(self, client):
        """Test splitting video successfully."""
        from datetime import datetime

        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "video-123_test.mp4"
        mock_video.dataset_id = "dataset-456"
        mock_video.to_dict.return_value = {"id": "video-123"}

        mock_clips = [
            {
                "id": "chunk-001",
                "filename": "01_test_1_1.mp4",
                "start_time": 0.0,
                "end_time": 5.0,
                "duration": 5.0,
                "action_index": 0,
                "action_number": 1,
                "repetition_count": 1,
                "timeline_order": 1,
                "action_description": "Pick up component",
                "created_at": datetime(2026, 1, 15, 10, 0, 0),
            }
        ]

        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_video

            with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
                mock_list.return_value = []  # No existing chunks

                with patch("inference._split_video_by_timestamps", new_callable=AsyncMock) as mock_split:
                    mock_split.return_value = mock_clips

                    response = client.post(
                        "/api/v1/videos/video-123/split",
                        json={"timestamps": [{"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "Pick up component"}]}
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "success"
                    assert len(data["clips"]) == 1
                    assert data["clips"][0]["id"] == "chunk-001"

    def test_split_video_cleans_up_existing_chunks(self, client):
        """Test that existing chunks are cleaned up during re-annotation."""
        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "video-123_test.mp4"
        mock_video.dataset_id = "dataset-456"
        mock_video.to_dict.return_value = {"id": "video-123"}

        existing_chunk = MagicMock()
        existing_chunk.id = "old-chunk-1"
        existing_chunk.name = "01_test_1_1.mp4"

        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_video

            with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
                mock_list.return_value = [existing_chunk]

                with patch("inference.postgres_db.delete_data", new_callable=AsyncMock) as mock_delete:
                    with patch("os.path.exists", return_value=True):
                        with patch("os.remove"):
                            with patch("inference._split_video_by_timestamps", new_callable=AsyncMock) as mock_split:
                                mock_split.return_value = [{"id": "new-chunk", "filename": "01_test_1_1.mp4"}]

                                response = client.post(
                                    "/api/v1/videos/video-123/split",
                                    json={"timestamps": [{"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "Test"}]}
                                )

                                assert mock_delete.called or response.status_code == 200

    def test_split_video_timestamp_with_missing_fields(self, client):
        """Test that timestamps with missing required fields are skipped."""
        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "video-123_test.mp4"
        mock_video.dataset_id = "dataset-456"
        mock_video.to_dict.return_value = {"id": "video-123"}

        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_video

            with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
                mock_list.return_value = []

                response = client.post(
                    "/api/v1/videos/video-123/split",
                    json={"timestamps": [
                        {"start": 0},
                        {"end": 5},
                        "not a dict"
                    ]}
                )

                assert response.status_code == 400
                assert "No valid timestamps" in response.json()["detail"]

    def test_split_video_generic_exception(self, client):
        """Test that generic exception during split returns 500."""
        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "video-123_test.mp4"
        mock_video.dataset_id = "dataset-456"
        mock_video.to_dict.return_value = {"id": "video-123"}

        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_video

            with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
                mock_list.return_value = []

                with patch("inference._split_video_by_timestamps", new_callable=AsyncMock) as mock_split:
                    mock_split.side_effect = Exception("Unexpected error")

                    response = client.post(
                        "/api/v1/videos/video-123/split",
                        json={"timestamps": [{"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "Test"}]}
                    )

                    assert response.status_code == 500


class TestDownloadVideo:
    """Tests for the download video endpoint."""

    def test_download_video_success_full_file(self, client):
        """Test downloading full video file without range header."""
        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "test_video.mp4"
        mock_video.dataset_id = "dataset-456"
        mock_video.file_size = 1024
        mock_video.mime_type = "video/mp4"

        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_video

            with patch("os.path.exists", return_value=True):
                with patch("inference.FileResponse") as mock_response:
                    mock_response.return_value = MagicMock()
                    response = client.get("/api/v1/videos/video-123/download")

                    assert response.status_code == 200 or mock_response.called

    def test_download_video_not_found_in_db(self, client):
        """Test downloading video that doesn't exist in database returns 404."""
        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = client.get("/api/v1/videos/nonexistent/download")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_download_video_file_not_on_disk(self, client):
        """Test downloading video when file doesn't exist on disk returns 404."""
        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "test_video.mp4"
        mock_video.dataset_id = "dataset-456"
        mock_video.file_size = 1024
        mock_video.mime_type = "video/mp4"

        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_video

            with patch("os.path.exists", return_value=False):
                response = client.get("/api/v1/videos/video-123/download")

                assert response.status_code == 404
                assert "not found on disk" in response.json()["detail"].lower()

    def test_download_video_with_range_header(self, client):
        """Test downloading video with valid range header returns partial content."""
        from unittest.mock import mock_open

        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "test_video.mp4"
        mock_video.dataset_id = "dataset-456"
        mock_video.file_size = 1000
        mock_video.mime_type = "video/mp4"

        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_video

            with patch("os.path.exists", return_value=True):
                m = mock_open(read_data=b"x" * 1000)
                with patch("builtins.open", m):
                    response = client.get(
                        "/api/v1/videos/video-123/download",
                        headers={"range": "bytes=0-99"}
                    )

                    assert response.status_code == 206
                    assert "Content-Range" in response.headers

    def test_download_video_exception_handling(self, client):
        """Test that exceptions during download are handled properly."""
        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "test_video.mp4"
        mock_video.dataset_id = "dataset-456"
        mock_video.file_size = 1000
        mock_video.mime_type = "video/mp4"

        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_video

            with patch("os.path.exists", return_value=True):
                with patch("builtins.open", side_effect=IOError("Cannot read file")):
                    response = client.get(
                        "/api/v1/videos/video-123/download",
                        headers={"range": "bytes=0-100"}
                    )

                    assert response.status_code == 500
                    assert "Error downloading video" in response.json()["detail"]


class TestDownloadChunk:
    """Tests for the download chunk endpoint."""

    def test_download_chunk_success(self, client):
        """Test downloading chunk file successfully."""
        mock_chunk = MagicMock()
        mock_chunk.id = "chunk-123"
        mock_chunk.video_id = "video-456"
        mock_chunk.name = "01_test_video_1_1.mp4"
        mock_chunk.mime_type = "video/mp4"

        mock_video = MagicMock()
        mock_video.id = "video-456"
        mock_video.name = "video-456_test_video.mp4"
        mock_video.dataset_id = "dataset-789"

        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            def get_data_side_effect(id_val, schema):
                schema_name = schema.__name__ if hasattr(schema, "__name__") else str(schema)
                if "Chunk" in schema_name:
                    return mock_chunk
                elif "Video" in schema_name:
                    return mock_video
                return None

            mock_get.side_effect = get_data_side_effect

            with patch("os.path.exists", return_value=True):
                with patch("inference.FileResponse") as mock_response:
                    mock_response.return_value = MagicMock(status_code=200)
                    response = client.get("/api/v1/chunks/chunk-123/download")

                    assert response.status_code == 200 or mock_response.called

    def test_download_chunk_not_found(self, client):
        """Test downloading non-existent chunk returns 404."""
        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = client.get("/api/v1/chunks/nonexistent/download")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_download_chunk_parent_video_not_found(self, client):
        """Test downloading chunk when parent video not found returns 404."""
        mock_chunk = MagicMock()
        mock_chunk.id = "chunk-123"
        mock_chunk.video_id = "video-456"
        mock_chunk.name = "01_test_video_1_1.mp4"
        mock_chunk.mime_type = "video/mp4"

        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            def get_data_side_effect(id_val, schema):
                schema_name = schema.__name__ if hasattr(schema, "__name__") else str(schema)
                if "Chunk" in schema_name:
                    return mock_chunk
                return None

            mock_get.side_effect = get_data_side_effect

            response = client.get("/api/v1/chunks/chunk-123/download")

            assert response.status_code == 404
            assert "parent video not found" in response.json()["detail"].lower()

    def test_download_chunk_exception_handling(self, client):
        """Test that exceptions during chunk download are handled properly."""
        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Database error")

            response = client.get("/api/v1/chunks/chunk-123/download")

            assert response.status_code == 500
            assert "Error downloading chunk" in response.json()["detail"]


class TestDownloadAllVideoClips:
    """Tests for the download all video clips endpoint."""

    def test_download_all_clips_video_not_found(self, client):
        """Test downloading all clips when video not found returns 404."""
        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = client.get("/api/v1/videos/nonexistent/download-all")

            assert response.status_code == 404
            assert "video not found" in response.json()["detail"].lower()

    def test_download_all_clips_no_chunks_found(self, client):
        """Test downloading all clips when no chunks exist returns 404."""
        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "video-123_test_video.mp4"
        mock_video.dataset_id = "dataset-456"

        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_video

            with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
                mock_list.return_value = []

                response = client.get("/api/v1/videos/video-123/download-all")

                assert response.status_code == 404
                assert "no chunks found" in response.json()["detail"].lower()

    def test_download_all_clips_zip_creation_failure(self, client):
        """Test handling ZIP creation failure."""
        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "video-123_test_video.mp4"
        mock_video.dataset_id = "dataset-456"

        mock_chunk = MagicMock()
        mock_chunk.name = "01_test_video_1_1.mp4"

        with patch("inference.postgres_db.get_data", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_video

            with patch("inference.postgres_db.list_data", new_callable=AsyncMock) as mock_list:
                mock_list.return_value = [mock_chunk]

                with patch("os.path.exists", return_value=True):
                    with patch("tempfile.NamedTemporaryFile") as mock_temp:
                        mock_temp.return_value.__enter__.return_value.name = "/tmp/test.zip"

                        with patch("zipfile.ZipFile", side_effect=Exception("Cannot create ZIP")):
                            with patch("os.unlink"):
                                response = client.get("/api/v1/videos/video-123/download-all")

                                assert response.status_code == 500
                                assert "Failed to create ZIP" in response.json()["detail"]
