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
Unit tests for data-generation-pipeline/validation/postgres_validation.py
"""

from datetime import datetime

from validation.postgres_validation import (
    Augmentation,
    AugmentationStage,
    Chunk,
    StatusEnum,
    Video,
)


class TestStatusEnum:
    """Tests for StatusEnum."""

    def test_enum_values_exist(self):
        """Test all expected enum values exist."""
        assert StatusEnum.running == "running"
        assert StatusEnum.completed == "completed"
        assert StatusEnum.failed == "failed"
        assert StatusEnum.pending == "pending"

    def test_enum_is_string_subclass(self):
        """Test enum inherits from str."""
        assert isinstance(StatusEnum.running, str)

    def test_enum_string_comparison(self):
        """Test enum values can be compared with strings."""
        assert StatusEnum.pending == "pending"
        assert StatusEnum.running != "completed"

    def test_enum_member_count(self):
        """Test enum has exactly 4 members."""
        assert len(StatusEnum) == 4


class TestAugmentation:
    """Tests for Augmentation model."""

    def test_to_dict_all_fields(self):
        """Test to_dict returns all fields correctly."""
        now = datetime.now()

        aug = Augmentation(
            id="aug-123",
            dataset_id="dataset-456",
            parameters={"key": "value"},
            status=StatusEnum.running,
            created_at=now,
            updated_at=now,
        )

        result = aug.to_dict()

        assert result["id"] == "aug-123"
        assert result["dataset_id"] == "dataset-456"
        assert result["parameters"] == {"key": "value"}
        assert result["status"] == StatusEnum.running
        assert result["created_at"] == now
        assert result["updated_at"] == now

    def test_to_dict_with_none_values(self):
        """Test to_dict handles None values correctly."""
        aug = Augmentation(
            id="aug-789",
            dataset_id=None,
            parameters=None,
            status=StatusEnum.pending,
            created_at=None,
            updated_at=None,
        )

        result = aug.to_dict()

        assert result["id"] == "aug-789"
        assert result["dataset_id"] is None
        assert result["parameters"] is None

    def test_to_dict_contains_all_keys(self):
        """Test to_dict contains all expected keys."""
        aug = Augmentation(id="aug-test", status=StatusEnum.completed)

        result = aug.to_dict()
        expected_keys = {"id", "dataset_id", "parameters", "status", "created_at", "updated_at"}

        assert set(result.keys()) == expected_keys

    def test_repr(self):
        """Test __repr__ returns expected format."""
        aug = Augmentation(
            id="aug-123",
            dataset_id="dataset-456",
            status=StatusEnum.running,
        )

        repr_str = repr(aug)

        assert "aug-123" in repr_str
        assert "dataset-456" in repr_str
        assert "running" in repr_str


class TestAugmentationStage:
    """Tests for AugmentationStage model."""

    def test_to_dict_all_fields(self):
        """Test to_dict returns all fields correctly."""
        now = datetime.now()

        stage = AugmentationStage(
            id="stage-123",
            augmentation_id="aug-456",
            stage_name="bcq_generation",
            status=StatusEnum.completed,
            created_at=now,
            updated_at=now,
            error_message=None,
        )

        result = stage.to_dict()

        assert result["id"] == "stage-123"
        assert result["augmentation_id"] == "aug-456"
        assert result["stage_name"] == "bcq_generation"
        assert result["status"] == StatusEnum.completed
        assert result["created_at"] == now
        assert result["updated_at"] == now
        assert result["error_message"] is None

    def test_to_dict_with_error_message(self):
        """Test to_dict includes error message when present."""
        stage = AugmentationStage(
            id="stage-failed",
            augmentation_id="aug-123",
            stage_name="gqa_generation",
            status=StatusEnum.failed,
            error_message="Processing failed due to invalid input",
        )

        result = stage.to_dict()

        assert result["status"] == StatusEnum.failed
        assert result["error_message"] == "Processing failed due to invalid input"

    def test_to_dict_contains_all_keys(self):
        """Test to_dict contains all expected keys."""
        stage = AugmentationStage(id="stage-test", stage_name="test", status=StatusEnum.pending)

        result = stage.to_dict()
        expected_keys = {"id", "augmentation_id", "stage_name", "status", "created_at", "updated_at", "error_message"}

        assert set(result.keys()) == expected_keys

    def test_repr(self):
        """Test __repr__ returns expected format."""
        stage = AugmentationStage(
            id="stage-123",
            augmentation_id="aug-456",
            stage_name="mcq_generation",
            status=StatusEnum.running,
        )

        repr_str = repr(stage)

        assert "stage-123" in repr_str
        assert "aug-456" in repr_str
        assert "mcq_generation" in repr_str
        assert "running" in repr_str


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
