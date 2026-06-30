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
Unit tests for cr-training-ms/utils/dataset_utils.py
"""

from utils.dataset_utils import get_all_json_paths


class TestGetAllJsonPaths:
    """Tests for get_all_json_paths function."""

    def test_empty_directory(self, temp_dir):
        """Test with empty directory returns empty list."""
        result = get_all_json_paths(str(temp_dir))
        assert result == []

    def test_nonexistent_path(self, temp_dir):
        """Test with non-existent path returns empty list."""
        result = get_all_json_paths(str(temp_dir / "nonexistent"))
        assert result == []

    def test_finds_json_files(self, temp_dir):
        """Test finding JSON files in directory."""
        (temp_dir / "file1.json").write_text("{}")
        (temp_dir / "file2.json").write_text("{}")

        result = get_all_json_paths(str(temp_dir))

        assert len(result) == 2
        assert all(p.endswith(".json") for p in result)

    def test_recursive_search(self, temp_dir):
        """Test finding JSON files in nested directories."""
        (temp_dir / "subdir").mkdir()
        (temp_dir / "file1.json").write_text("{}")
        (temp_dir / "subdir" / "file2.json").write_text("{}")

        result = get_all_json_paths(str(temp_dir))

        assert len(result) == 2

    def test_deeply_nested_directories(self, temp_dir):
        """Test finding JSON files in deeply nested directories."""
        nested = temp_dir / "a" / "b" / "c"
        nested.mkdir(parents=True)
        (temp_dir / "root.json").write_text("{}")
        (temp_dir / "a" / "level1.json").write_text("{}")
        (nested / "deep.json").write_text("{}")

        result = get_all_json_paths(str(temp_dir))

        assert len(result) == 3

    def test_ignores_non_json_files(self, temp_dir):
        """Test that non-JSON files are ignored."""
        (temp_dir / "file.json").write_text("{}")
        (temp_dir / "file.txt").write_text("text")
        (temp_dir / "file.yaml").write_text("yaml: true")
        (temp_dir / "file.jsonl").write_text("{}")

        result = get_all_json_paths(str(temp_dir))

        assert len(result) == 1
        assert result[0].endswith(".json")

    def test_returns_absolute_paths(self, temp_dir):
        """Test that returned paths are absolute paths."""
        (temp_dir / "file.json").write_text("{}")

        result = get_all_json_paths(str(temp_dir))

        assert len(result) == 1
        assert result[0].startswith("/")
