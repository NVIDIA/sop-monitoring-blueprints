######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
######################################################################################################

"""
Unit tests for cr1_training_ms/utils/dataset_utils.py
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
