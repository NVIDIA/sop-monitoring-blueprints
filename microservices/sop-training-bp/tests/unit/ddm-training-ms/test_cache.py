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
Unit tests for ddm-training-ms/components/cache.py
"""

from components.cache import TrainingJobCache


class TestTrainingJobCache:
    """Tests for TrainingJobCache class."""

    def test_set_and_get_job(self):
        """Test setting and getting a job."""
        cache = TrainingJobCache()
        job_record = {"status": "running", "progress": 50.0}

        cache.set("job-123", job_record)
        result = cache.get("job-123")

        assert result == job_record

    def test_get_nonexistent_job_returns_none(self):
        """Test that getting a nonexistent job returns None."""
        cache = TrainingJobCache()

        result = cache.get("nonexistent")

        assert result is None

    def test_update_job(self):
        """Test updating a job's properties."""
        cache = TrainingJobCache()
        cache.set("job-123", {"status": "queued", "progress": 0.0})

        cache.update("job-123", status="running", progress=25.0)
        result = cache.get("job-123")

        assert result["status"] == "running"
        assert result["progress"] == 25.0

    def test_update_with_new_fields(self):
        """Test that update can add new fields."""
        cache = TrainingJobCache()
        cache.set("job-123", {"status": "running"})

        cache.update("job-123", loss=0.5, current_step=100)
        result = cache.get("job-123")

        assert result["status"] == "running"
        assert result["loss"] == 0.5
        assert result["current_step"] == 100

    def test_delete_job(self):
        """Test deleting a job."""
        cache = TrainingJobCache()
        cache.set("job-123", {"status": "running"})

        cache.delete("job-123")
        result = cache.get("job-123")

        assert result is None

    def test_delete_nonexistent_job_no_error(self):
        """Test that deleting a nonexistent job doesn't raise error."""
        cache = TrainingJobCache()

        # Should not raise
        cache.delete("nonexistent")

    def test_clear_cache(self):
        """Test clearing all jobs from cache."""
        cache = TrainingJobCache()
        cache.set("job-1", {"status": "running"})
        cache.set("job-2", {"status": "queued"})
        cache.set("job-3", {"status": "completed"})

        cache.clear()

        assert cache.get("job-1") is None
        assert cache.get("job-2") is None
        assert cache.get("job-3") is None

    def test_multiple_jobs_isolation(self):
        """Test that multiple jobs are isolated from each other."""
        cache = TrainingJobCache()
        cache.set("job-1", {"status": "running", "progress": 50.0})
        cache.set("job-2", {"status": "queued", "progress": 0.0})

        cache.update("job-1", progress=75.0)

        assert cache.get("job-1")["progress"] == 75.0
        assert cache.get("job-2")["progress"] == 0.0

    def test_cache_stores_reference(self):
        """Test cache behavior with mutable objects."""
        cache = TrainingJobCache()
        job_record = {"status": "running", "steps": [1, 2, 3]}

        cache.set("job-123", job_record)

        # Modifying original affects cached value (reference semantics)
        job_record["status"] = "completed"
        assert cache.get("job-123")["status"] == "completed"
