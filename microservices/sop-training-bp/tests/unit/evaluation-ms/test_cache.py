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

"""Unit tests for components.cache.JobCache (in-memory job record store)."""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MS_PATH = str(PROJECT_ROOT / "microservices" / "evaluation-ms")
if MS_PATH not in sys.path:
    sys.path.insert(0, MS_PATH)


@pytest.mark.unit
class TestJobCache:
    def test_set_then_get_returns_record(self):
        from components.cache import JobCache

        cache = JobCache()
        record = {"status": "running", "progress": 0.0}
        cache.set("job-1", record)
        assert cache.get("job-1") == record

    def test_get_missing_returns_none(self):
        from components.cache import JobCache

        assert JobCache().get("nope") is None

    def test_update_merges_fields(self):
        from components.cache import JobCache

        cache = JobCache()
        cache.set("job-1", {"status": "running", "progress": 0.0})
        cache.update("job-1", status="completed", progress=1.0)
        assert cache.get("job-1") == {"status": "completed", "progress": 1.0}

    def test_delete_removes_record(self):
        from components.cache import JobCache

        cache = JobCache()
        cache.set("job-1", {"status": "running"})
        cache.delete("job-1")
        assert cache.get("job-1") is None

    def test_delete_missing_is_noop(self):
        from components.cache import JobCache

        cache = JobCache()
        # pop(..., None) means deleting an absent key must not raise.
        cache.delete("absent")
        assert cache.get("absent") is None

    def test_clear_empties_cache(self):
        from components.cache import JobCache

        cache = JobCache()
        cache.set("a", {"x": 1})
        cache.set("b", {"y": 2})
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.cache == {}
