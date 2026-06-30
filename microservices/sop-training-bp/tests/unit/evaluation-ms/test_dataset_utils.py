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

"""Unit tests for evaluation-ms ``utils/dataset_utils.py``."""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MS_PATH = str(PROJECT_ROOT / "microservices" / "evaluation-ms")
if MS_PATH not in sys.path:
    sys.path.insert(0, MS_PATH)


@pytest.mark.unit
class TestGetAllJsonPaths:
    def test_returns_empty_for_nonexistent_path(self, tmp_path):
        from utils.dataset_utils import get_all_json_paths

        missing = tmp_path / "does-not-exist"
        assert get_all_json_paths(str(missing)) == []

    def test_returns_empty_when_no_json_files(self, tmp_path):
        from utils.dataset_utils import get_all_json_paths

        (tmp_path / "notes.txt").write_text("not json")
        (tmp_path / "video.mp4").write_text("binary-ish")
        assert get_all_json_paths(str(tmp_path)) == []

    def test_finds_json_files_recursively(self, tmp_path):
        from utils.dataset_utils import get_all_json_paths

        top = tmp_path / "a.json"
        top.write_text("{}")
        nested_dir = tmp_path / "nested" / "deeper"
        nested_dir.mkdir(parents=True)
        nested = nested_dir / "b.json"
        nested.write_text("{}")
        # A non-json sibling that must be ignored.
        (tmp_path / "skip.yaml").write_text("k: v")

        result = get_all_json_paths(str(tmp_path))

        assert set(result) == {str(top), str(nested)}
        # Paths are returned as strings, not Path objects.
        assert all(isinstance(p, str) for p in result)

    def test_accepts_path_object_via_str(self, tmp_path):
        from utils.dataset_utils import get_all_json_paths

        (tmp_path / "only.json").write_text("{}")
        result = get_all_json_paths(str(tmp_path))
        assert len(result) == 1
        assert result[0].endswith("only.json")
