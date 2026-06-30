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
Unit tests for ddm-training-ms/utils/dataset_utils.py
"""

import json

import pytest

from utils.dataset_utils import generate_ddm_annotation, get_all_json_paths


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

        result = get_all_json_paths(str(temp_dir))

        assert len(result) == 1
        assert result[0].endswith(".json")


class TestGenerateDdmAnnotation:
    """Tests for generate_ddm_annotation function."""

    def test_generates_annotation_file(self, temp_dir):
        """Test generating annotation file from video and annotation."""
        # Create a mock video file
        video_file = temp_dir / "test_video.mp4"
        video_file.write_text("mock video content")

        # Create annotation directory and file
        annotation_dir = temp_dir / "test_video"
        annotation_dir.mkdir()
        annotation_file = annotation_dir / "test_video_annotation.json"
        annotation_data = {"actions": [{"start": 0, "end": 10, "label": "action1"}]}
        annotation_file.write_text(json.dumps(annotation_data))

        result = generate_ddm_annotation(str(temp_dir), "output.json")

        assert result == str(temp_dir / "output.json")
        assert (temp_dir / "output.json").exists()

        # Verify content
        with open(result) as f:
            output_data = json.load(f)
        assert "test_video" in output_data
        assert output_data["test_video"] == annotation_data

    def test_multiple_videos(self, temp_dir):
        """Test generating annotation from multiple videos."""
        # Create multiple videos and annotations
        for i in range(3):
            video_file = temp_dir / f"video{i}.mp4"
            video_file.write_text("mock video")

            annotation_dir = temp_dir / f"video{i}"
            annotation_dir.mkdir()
            annotation_file = annotation_dir / f"video{i}_annotation.json"
            annotation_file.write_text(json.dumps({"id": i}))

        result = generate_ddm_annotation(str(temp_dir), "combined.json")

        with open(result) as f:
            output_data = json.load(f)

        assert len(output_data) == 3
        assert "video0" in output_data
        assert "video1" in output_data
        assert "video2" in output_data

    def test_nonexistent_dataset_path_raises_error(self, temp_dir):
        """Test that non-existent dataset path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            generate_ddm_annotation(str(temp_dir / "nonexistent"), "output.json")

    def test_no_valid_annotations_raises_error(self, temp_dir):
        """Test that no valid video/annotation pairs raises ValueError."""
        # Create video without annotation
        video_file = temp_dir / "orphan_video.mp4"
        video_file.write_text("mock video")

        with pytest.raises(ValueError, match="No valid video/annotation pairs"):
            generate_ddm_annotation(str(temp_dir), "output.json")

    def test_invalid_json_raises_error(self, temp_dir):
        """Test that invalid JSON in annotation raises JSONDecodeError."""
        video_file = temp_dir / "test_video.mp4"
        video_file.write_text("mock video")

        annotation_dir = temp_dir / "test_video"
        annotation_dir.mkdir()
        annotation_file = annotation_dir / "test_video_annotation.json"
        annotation_file.write_text("invalid json {{{")

        with pytest.raises(json.JSONDecodeError):
            generate_ddm_annotation(str(temp_dir), "output.json")

    def test_skips_videos_without_annotation(self, temp_dir):
        """Test that videos without annotations are skipped but others processed."""
        # Video with annotation
        video1 = temp_dir / "video1.mp4"
        video1.write_text("mock video")
        ann_dir1 = temp_dir / "video1"
        ann_dir1.mkdir()
        (ann_dir1 / "video1_annotation.json").write_text(json.dumps({"id": 1}))

        # Video without annotation
        video2 = temp_dir / "video2.mp4"
        video2.write_text("mock video")

        result = generate_ddm_annotation(str(temp_dir), "output.json")

        with open(result) as f:
            output_data = json.load(f)

        assert len(output_data) == 1
        assert "video1" in output_data
        assert "video2" not in output_data

    def test_empty_directory_no_videos(self, temp_dir):
        """Test with directory containing no videos raises ValueError."""
        # Create some non-video files
        (temp_dir / "readme.txt").write_text("readme")

        with pytest.raises(ValueError, match="No valid video/annotation pairs"):
            generate_ddm_annotation(str(temp_dir), "output.json")
