######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Unit tests for cr-training-ms/components/cache.py
"""

from datetime import datetime

import pytest

from components.cache import TrainingJobCache


class TestTrainingJobCache:
    """Tests for TrainingJobCache class."""

    @pytest.fixture
    def cache(self):
        """Create a fresh cache instance for each test."""
        return TrainingJobCache()

    @pytest.fixture
    def sample_job(self):
        """Create a sample job record."""
        return {
            "job_id": "test-job-123",
            "status": "running",
            "progress": 0.0,
            "current_step": 0,
            "total_steps": 100,
            "loss": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

    def test_set_and_get_job(self, cache, sample_job):
        """Test setting and getting a job."""
        job_id = "test-job-123"

        cache.set(job_id, sample_job)
        result = cache.get(job_id)

        assert result == sample_job
        assert result["status"] == "running"

    def test_get_nonexistent_job_returns_none(self, cache):
        """Test that getting a nonexistent job returns None."""
        result = cache.get("nonexistent-job")
        assert result is None

    def test_update_job(self, cache, sample_job):
        """Test updating a job's properties."""
        job_id = "test-job-123"
        cache.set(job_id, sample_job)

        cache.update(job_id, status="completed", progress=100.0, current_step=100)

        result = cache.get(job_id)
        assert result["status"] == "completed"
        assert result["progress"] == 100.0
        assert result["current_step"] == 100
        # Other properties should be preserved
        assert result["total_steps"] == 100
        assert result["job_id"] == "test-job-123"

    def test_update_with_new_fields(self, cache, sample_job):
        """Test that update can add new fields."""
        job_id = "test-job-123"
        cache.set(job_id, sample_job)

        cache.update(job_id, process_pid=12345, custom_field="value")

        result = cache.get(job_id)
        assert result["process_pid"] == 12345
        assert result["custom_field"] == "value"

    def test_delete_job(self, cache, sample_job):
        """Test deleting a job."""
        job_id = "test-job-123"
        cache.set(job_id, sample_job)

        cache.delete(job_id)

        assert cache.get(job_id) is None

    def test_delete_nonexistent_job_no_error(self, cache):
        """Test that deleting a nonexistent job doesn't raise error."""
        # Should not raise
        cache.delete("nonexistent-job")

    def test_clear_cache(self, cache, sample_job):
        """Test clearing all jobs from cache."""
        cache.set("job-1", sample_job)
        cache.set("job-2", {**sample_job, "job_id": "job-2"})
        cache.set("job-3", {**sample_job, "job_id": "job-3"})

        cache.clear()

        assert cache.get("job-1") is None
        assert cache.get("job-2") is None
        assert cache.get("job-3") is None
        assert len(cache.cache) == 0

    def test_multiple_jobs_isolation(self, cache):
        """Test that multiple jobs are isolated from each other."""
        job1 = {"job_id": "job-1", "status": "running"}
        job2 = {"job_id": "job-2", "status": "queued"}

        cache.set("job-1", job1)
        cache.set("job-2", job2)

        # Update job-1 shouldn't affect job-2
        cache.update("job-1", status="completed")

        assert cache.get("job-1")["status"] == "completed"
        assert cache.get("job-2")["status"] == "queued"

    def test_cache_stores_reference(self, cache, sample_job):
        """Test cache behavior with mutable objects."""
        job_id = "test-job-123"
        cache.set(job_id, sample_job)

        # Modifying the original should affect cached value (reference)
        # This is expected Python behavior
        sample_job["status"] = "modified"
        assert cache.get(job_id)["status"] == "modified"
