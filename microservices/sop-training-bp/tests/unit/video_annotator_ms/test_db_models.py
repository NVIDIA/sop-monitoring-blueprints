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
Unit tests for video-annotator-ms/annotation_backend/validations/db_models.py
"""

from datetime import datetime

from validations.db_models import Annotation, Chunk, Dataset, Video


class TestDataset:
    """Tests for Dataset model."""

    def test_to_dict_all_fields(self):
        """Test to_dict returns all fields correctly."""
        now = datetime.now()

        dataset = Dataset(
            id="dataset-123",
            actions=["action1", "action2", "action3"],
            created_at=now,
            updated_at=now,
        )

        result = dataset.to_dict()

        assert result["id"] == "dataset-123"
        assert result["actions"] == ["action1", "action2", "action3"]
        assert result["created_at"] == now
        assert result["updated_at"] == now

    def test_to_dict_with_none_values(self):
        """Test to_dict handles None values correctly."""
        dataset = Dataset(
            id="dataset-789",
            actions=None,
            created_at=None,
            updated_at=None,
        )

        result = dataset.to_dict()

        assert result["id"] == "dataset-789"
        assert result["actions"] is None
        assert result["created_at"] is None

    def test_to_dict_contains_all_keys(self):
        """Test to_dict contains all expected keys."""
        dataset = Dataset(id="dataset-test")

        result = dataset.to_dict()
        expected_keys = {"id", "actions", "two_operator_mode", "created_at", "updated_at"}

        assert set(result.keys()) == expected_keys

    def test_to_dict_with_empty_actions(self):
        """Test to_dict with empty actions list."""
        dataset = Dataset(
            id="dataset-empty",
            actions=[],
        )

        result = dataset.to_dict()

        assert result["actions"] == []


class TestVideo:
    """Tests for Video model."""

    def test_to_dict_all_fields(self):
        """Test to_dict returns all fields correctly."""
        now = datetime.now()

        video = Video(
            id="video-123",
            dataset_id="dataset-456",
            name="assembly_video.mp4",
            mime_type="video/mp4",
            file_size=1024000,
            created_at=now,
            updated_at=now,
        )

        result = video.to_dict()

        assert result["id"] == "video-123"
        assert result["dataset_id"] == "dataset-456"
        assert result["name"] == "assembly_video.mp4"
        assert result["mime_type"] == "video/mp4"
        assert result["file_size"] == 1024000
        assert result["created_at"] == now
        assert result["updated_at"] == now

    def test_to_dict_with_none_values(self):
        """Test to_dict handles None values correctly."""
        video = Video(
            id="video-789",
            dataset_id=None,
            name=None,
            mime_type=None,
            file_size=None,
        )

        result = video.to_dict()

        assert result["id"] == "video-789"
        assert result["dataset_id"] is None
        assert result["name"] is None
        assert result["file_size"] is None

    def test_to_dict_contains_all_keys(self):
        """Test to_dict contains all expected keys."""
        video = Video(id="video-test")

        result = video.to_dict()
        expected_keys = {"id", "dataset_id", "name", "mime_type", "file_size", "created_at", "updated_at"}

        assert set(result.keys()) == expected_keys


class TestChunk:
    """Tests for Chunk model."""

    def test_to_dict_all_fields(self):
        """Test to_dict returns all fields correctly."""
        now = datetime.now()

        chunk = Chunk(
            id="chunk-123",
            video_id="video-456",
            name="chunk_001.mp4",
            action="pick_component",
            mime_type="video/mp4",
            file_size=512000,
            created_at=now,
            updated_at=now,
        )

        result = chunk.to_dict()

        assert result["id"] == "chunk-123"
        assert result["video_id"] == "video-456"
        assert result["name"] == "chunk_001.mp4"
        assert result["action"] == "pick_component"
        assert result["mime_type"] == "video/mp4"
        assert result["file_size"] == 512000
        assert result["created_at"] == now
        assert result["updated_at"] == now

    def test_to_dict_with_none_values(self):
        """Test to_dict handles None values correctly."""
        chunk = Chunk(
            id="chunk-789",
            video_id=None,
            name=None,
            action=None,
            mime_type=None,
            file_size=None,
        )

        result = chunk.to_dict()

        assert result["id"] == "chunk-789"
        assert result["video_id"] is None
        assert result["action"] is None

    def test_to_dict_contains_all_keys(self):
        """Test to_dict contains all expected keys."""
        chunk = Chunk(id="chunk-test")

        result = chunk.to_dict()
        expected_keys = {"id", "video_id", "name", "action", "mime_type", "file_size", "created_at", "updated_at"}

        assert set(result.keys()) == expected_keys


class TestAnnotation:
    """Tests for Annotation model."""

    def test_to_dict_all_fields(self):
        """Test to_dict returns all fields correctly."""
        now = datetime.now()

        annotation = Annotation(
            id="ann-123",
            video_id="video-456",
            chunk_id="chunk-789",
            start_time=0.0,
            end_time=5.5,
            action_index=1,
            action_description="Pick up the component",
            created_at=now,
            updated_at=now,
        )

        result = annotation.to_dict()

        assert result["id"] == "ann-123"
        assert result["video_id"] == "video-456"
        assert result["chunk_id"] == "chunk-789"
        assert result["start_time"] == 0.0
        assert result["end_time"] == 5.5
        assert result["action_index"] == 1
        assert result["action_description"] == "Pick up the component"
        assert result["created_at"] == now
        assert result["updated_at"] == now

    def test_to_dict_with_none_values(self):
        """Test to_dict handles None values correctly."""
        annotation = Annotation(
            id="ann-789",
            video_id=None,
            chunk_id=None,
            start_time=None,
            end_time=None,
            action_index=None,
            action_description=None,
        )

        result = annotation.to_dict()

        assert result["id"] == "ann-789"
        assert result["video_id"] is None
        assert result["chunk_id"] is None
        assert result["start_time"] is None
        assert result["action_description"] is None

    def test_to_dict_contains_all_keys(self):
        """Test to_dict contains all expected keys."""
        annotation = Annotation(id="ann-test")

        result = annotation.to_dict()
        expected_keys = {
            "id",
            "video_id",
            "chunk_id",
            "start_time",
            "end_time",
            "action_index",
            "action_description",
            "created_at",
            "updated_at",
        }

        assert set(result.keys()) == expected_keys

    def test_to_dict_with_float_times(self):
        """Test to_dict with various float time values."""
        annotation = Annotation(
            id="ann-float",
            start_time=10.123,
            end_time=25.456,
        )

        result = annotation.to_dict()

        assert result["start_time"] == 10.123
        assert result["end_time"] == 25.456
