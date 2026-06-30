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
Unit tests for the pure helper functions in annotation_backend/inference.py:
_emit_segment, _find_concurrent_time_segments, and _load_action_descriptions.
"""

import json
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

# Mock the database before importing the app module (mirrors test_inference.py).
with patch("components.postgres_db.postgres_db") as mock_db:
    mock_db.get_data = AsyncMock(return_value=None)
    from inference import (
        _emit_segment,
        _find_concurrent_time_segments,
        _load_action_descriptions,
    )


class TestEmitSegment:
    """Tests for _emit_segment."""

    def test_single_active_action_not_concurrent(self):
        timestamps = [
            {"actionIndex": 1, "actionDescription": "pick"},
            {"actionIndex": 2, "actionDescription": "place"},
        ]
        seg = _emit_segment(0.0, 5.0, {0}, timestamps)
        assert seg["start"] == 0.0
        assert seg["end"] == 5.0
        assert seg["is_concurrent"] is False
        assert seg["concurrent_actions"] == [{"actionIndex": 1, "actionDescription": "pick"}]

    def test_two_active_actions_is_concurrent(self):
        timestamps = [
            {"actionIndex": 1, "actionDescription": "pick"},
            {"actionIndex": 2, "actionDescription": "place"},
        ]
        seg = _emit_segment(1.0, 4.0, {0, 1}, timestamps)
        assert seg["is_concurrent"] is True
        assert len(seg["concurrent_actions"]) == 2
        assert {a["actionIndex"] for a in seg["concurrent_actions"]} == {1, 2}

    def test_duplicate_action_index_deduplicated(self):
        timestamps = [
            {"actionIndex": 1, "actionDescription": "pick"},
            {"actionIndex": 1, "actionDescription": "pick again"},
        ]
        seg = _emit_segment(0.0, 2.0, {0, 1}, timestamps)
        # Same actionIndex appears once; not concurrent.
        assert len(seg["concurrent_actions"]) == 1
        assert seg["is_concurrent"] is False


class TestFindConcurrentTimeSegments:
    """Tests for _find_concurrent_time_segments."""

    def test_empty_returns_empty(self):
        assert _find_concurrent_time_segments([]) == []

    def test_non_overlapping_actions(self):
        timestamps = [
            {"start": 0, "end": 5, "actionIndex": 1, "actionDescription": "a"},
            {"start": 5, "end": 10, "actionIndex": 2, "actionDescription": "b"},
        ]
        segs = _find_concurrent_time_segments(timestamps)
        assert all(not s["is_concurrent"] for s in segs)
        assert [s["concurrent_actions"][0]["actionIndex"] for s in segs] == [1, 2]

    def test_overlapping_actions_produce_concurrent_segment(self):
        timestamps = [
            {"start": 0, "end": 10, "actionIndex": 1, "actionDescription": "a"},
            {"start": 5, "end": 15, "actionIndex": 2, "actionDescription": "b"},
        ]
        segs = _find_concurrent_time_segments(timestamps)
        # 0-5 (a), 5-10 (a+b concurrent), 10-15 (b)
        assert len(segs) == 3
        assert segs[0]["is_concurrent"] is False
        assert segs[1]["is_concurrent"] is True
        assert {a["actionIndex"] for a in segs[1]["concurrent_actions"]} == {1, 2}
        assert segs[2]["is_concurrent"] is False


class TestLoadActionDescriptions:
    """Tests for _load_action_descriptions."""

    def test_loads_and_indexes_actions(self):
        video_metadata = MagicMock()
        video_metadata.dataset_id = "ds-1"
        payload = json.dumps({"actions": ["pick up", "place down", "inspect"]})

        with patch("builtins.open", mock_open(read_data=payload)):
            result = _load_action_descriptions(video_metadata)

        assert result == {0: "pick up", 1: "place down", 2: "inspect"}

    def test_missing_actions_key_returns_empty(self):
        video_metadata = MagicMock()
        video_metadata.dataset_id = "ds-1"

        with patch("builtins.open", mock_open(read_data=json.dumps({}))):
            result = _load_action_descriptions(video_metadata)

        assert result == {}
