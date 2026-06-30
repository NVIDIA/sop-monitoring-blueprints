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
Unit tests for cr-training-ms/validation/postgres_validation.py
"""

from datetime import datetime

from validation.postgres_validation import TrainingJob, TrainingStatusEnum


class TestTrainingStatusEnum:
    """Tests for TrainingStatusEnum."""

    def test_enum_values_exist(self):
        """Test all expected enum values exist."""
        assert TrainingStatusEnum.queued == "queued"
        assert TrainingStatusEnum.running == "running"
        assert TrainingStatusEnum.completed == "completed"
        assert TrainingStatusEnum.cancelled == "cancelled"
        assert TrainingStatusEnum.failed == "failed"

    def test_enum_is_string_subclass(self):
        """Test enum inherits from str."""
        assert isinstance(TrainingStatusEnum.running, str)

    def test_enum_string_comparison(self):
        """Test enum values can be compared with strings."""
        assert TrainingStatusEnum.queued == "queued"
        assert TrainingStatusEnum.running != "queued"

    def test_enum_member_count(self):
        """Test enum has exactly 5 members."""
        assert len(TrainingStatusEnum) == 5


class TestTrainingJob:
    """Tests for TrainingJob model."""

    def test_to_dict_all_fields(self):
        """Test to_dict returns all fields correctly."""
        now = datetime.now()

        job = TrainingJob(
            id="job-123",
            aug_dataset_id="dataset-456",
            status=TrainingStatusEnum.running,
            total_steps=100,
            current_step=50,
            progress=50.0,
            loss=0.5,
            created_at=now,
            updated_at=now,
        )

        result = job.to_dict()

        assert result["id"] == "job-123"
        assert result["aug_dataset_id"] == "dataset-456"
        assert result["status"] == TrainingStatusEnum.running
        assert result["total_steps"] == 100
        assert result["current_step"] == 50
        assert result["progress"] == 50.0
        assert result["loss"] == 0.5
        assert result["created_at"] == now
        assert result["updated_at"] == now

    def test_to_dict_with_none_values(self):
        """Test to_dict handles None values correctly."""
        now = datetime.now()

        job = TrainingJob(
            id="job-789",
            aug_dataset_id=None,
            status=TrainingStatusEnum.queued,
            total_steps=None,
            current_step=None,
            progress=None,
            loss=None,
            created_at=now,
            updated_at=now,
        )

        result = job.to_dict()

        assert result["id"] == "job-789"
        assert result["aug_dataset_id"] is None
        assert result["total_steps"] is None
        assert result["current_step"] is None
        assert result["progress"] is None
        assert result["loss"] is None

    def test_to_dict_returns_dict_type(self):
        """Test to_dict returns a dictionary."""
        job = TrainingJob(
            id="job-test",
            status=TrainingStatusEnum.completed,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        result = job.to_dict()

        assert isinstance(result, dict)

    def test_to_dict_contains_all_keys(self):
        """Test to_dict contains all expected keys."""
        job = TrainingJob(
            id="job-test",
            status=TrainingStatusEnum.failed,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        result = job.to_dict()
        expected_keys = {
            "id",
            "aug_dataset_id",
            "status",
            "total_steps",
            "current_step",
            "progress",
            "loss",
            "created_at",
            "updated_at",
        }

        assert set(result.keys()) == expected_keys

    def test_to_dict_with_different_statuses(self):
        """Test to_dict works with all status values."""
        now = datetime.now()

        for status in TrainingStatusEnum:
            job = TrainingJob(
                id=f"job-{status.value}",
                status=status,
                created_at=now,
                updated_at=now,
            )

            result = job.to_dict()

            assert result["status"] == status
