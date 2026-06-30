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
Unit tests for vlm_aug/utils/helper.py

Tests cover pure utility functions and simple I/O operations.
Video processing functions are skipped as they require cv2/moviepy with actual video files.
"""

import argparse

import pytest

from unittest.mock import MagicMock, patch

import numpy as np

from vlm_aug.utils.helper import (
    clean_sentence,
    create_dir,
    custom_sort_key,
    dump_json,
    format_concurrent_actions,
    get_video_meta,
    parse_video_action_indices,
    read_json,
    read_txt,
    str2bool,
    unpack_annotation,
    write_frames,
    write_txt,
    write_video,
)


class TestCleanSentence:
    """Tests for clean_sentence function."""

    def test_removes_leading_numbers(self):
        """Test removing leading numbers and special chars."""
        assert clean_sentence("1. Pick up the item") == "Pick up the item"
        assert clean_sentence("123 Start here") == "Start here"

    def test_removes_trailing_special_chars(self):
        """Test removing trailing special chars."""
        assert clean_sentence("Pick up the item.") == "Pick up the item"
        assert clean_sentence("Do the action!") == "Do the action"
        assert clean_sentence("Question?") == "Question"

    def test_removes_leading_special_chars(self):
        """Test removing leading special chars."""
        assert clean_sentence("- Pick up item") == "Pick up item"
        assert clean_sentence("* Action item") == "Action item"
        assert clean_sentence("  Spaced text") == "Spaced text"

    def test_preserves_middle_content(self):
        """Test that middle content is preserved."""
        assert clean_sentence("1. Step 1: Do something.") == "Step 1: Do something"
        assert clean_sentence("2) Item-2 here!") == "Item-2 here"

    def test_already_clean_sentence(self):
        """Test sentence that's already clean."""
        assert clean_sentence("Already clean") == "Already clean"

    def test_empty_string(self):
        """Test with empty string."""
        assert clean_sentence("") == ""

    def test_only_special_chars(self):
        """Test string with only special chars."""
        assert clean_sentence("123...") == ""
        assert clean_sentence("!!!") == ""


class TestStr2Bool:
    """Tests for str2bool function."""

    def test_true_values(self):
        """Test various true string values."""
        assert str2bool("yes") is True
        assert str2bool("Yes") is True
        assert str2bool("YES") is True
        assert str2bool("true") is True
        assert str2bool("True") is True
        assert str2bool("t") is True
        assert str2bool("y") is True
        assert str2bool("1") is True

    def test_false_values(self):
        """Test various false string values."""
        assert str2bool("no") is False
        assert str2bool("No") is False
        assert str2bool("NO") is False
        assert str2bool("false") is False
        assert str2bool("False") is False
        assert str2bool("f") is False
        assert str2bool("n") is False
        assert str2bool("0") is False

    def test_bool_passthrough(self):
        """Test that bool values pass through unchanged."""
        assert str2bool(True) is True
        assert str2bool(False) is False

    def test_invalid_value_raises_error(self):
        """Test that invalid values raise ArgumentTypeError."""
        with pytest.raises(argparse.ArgumentTypeError):
            str2bool("invalid")
        with pytest.raises(argparse.ArgumentTypeError):
            str2bool("maybe")
        with pytest.raises(argparse.ArgumentTypeError):
            str2bool("2")


class TestUnpackAnnotation:
    """Tests for unpack_annotation function."""

    def test_unpack_single_list(self):
        """Test unpacking a single list of annotations."""
        annotations = [[{"id": 99, "text": "a"}, {"id": 88, "text": "b"}]]

        result = unpack_annotation(annotations)

        assert len(result) == 2
        assert result[0]["id"] == 0
        assert result[0]["text"] == "a"
        assert result[1]["id"] == 1
        assert result[1]["text"] == "b"

    def test_unpack_multiple_lists(self):
        """Test unpacking multiple lists of annotations."""
        annotations = [
            [{"id": 0, "text": "a"}],
            [{"id": 0, "text": "b"}, {"id": 1, "text": "c"}],
            [{"id": 0, "text": "d"}],
        ]

        result = unpack_annotation(annotations)

        assert len(result) == 4
        assert [r["id"] for r in result] == [0, 1, 2, 3]
        assert [r["text"] for r in result] == ["a", "b", "c", "d"]

    def test_unpack_empty_list(self):
        """Test unpacking empty list."""
        result = unpack_annotation([])
        assert result == []

    def test_unpack_preserves_other_fields(self):
        """Test that other fields are preserved."""
        annotations = [[{"id": 5, "text": "hello", "extra": "data"}]]

        result = unpack_annotation(annotations)

        assert result[0]["id"] == 0
        assert result[0]["text"] == "hello"
        assert result[0]["extra"] == "data"


class TestCustomSortKey:
    """Tests for custom_sort_key function."""

    def test_basic_sorting(self):
        """Test basic action sorting."""
        assert custom_sort_key("action1.json", "action", ".json") == (1, 0)
        assert custom_sort_key("action2.json", "action", ".json") == (2, 0)
        assert custom_sort_key("action10.json", "action", ".json") == (10, 0)

    def test_sorting_with_suffix(self):
        """Test sorting with underscore suffix."""
        assert custom_sort_key("action1_1.json", "action", ".json") == (1, 1)
        assert custom_sort_key("action1_2.json", "action", ".json") == (1, 2)
        assert custom_sort_key("action2_1.json", "action", ".json") == (2, 1)

    def test_sorting_order(self):
        """Test that sorting produces correct order."""
        files = ["action2.json", "action1_2.json", "action1.json", "action1_1.json", "action10.json"]

        sorted_files = sorted(files, key=lambda f: custom_sort_key(f, "action", ".json"))

        assert sorted_files == ["action1.json", "action1_1.json", "action1_2.json", "action2.json", "action10.json"]

    def test_different_keyword(self):
        """Test with different keyword."""
        assert custom_sort_key("step5.txt", "step", ".txt") == (5, 0)
        assert custom_sort_key("chunk3_2.mp4", "chunk", ".mp4") == (3, 2)


class TestCreateDir:
    """Tests for create_dir function."""

    def test_creates_new_directory(self, temp_dir):
        """Test creating a new directory."""
        new_dir = temp_dir / "new_folder"
        assert not new_dir.exists()

        create_dir(str(new_dir))

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_creates_nested_directories(self, temp_dir):
        """Test creating nested directories."""
        nested = temp_dir / "a" / "b" / "c"

        create_dir(str(nested))

        assert nested.exists()

    def test_existing_directory_no_error(self, temp_dir):
        """Test that existing directory doesn't raise error."""
        existing = temp_dir / "existing"
        existing.mkdir()

        # Should not raise
        create_dir(str(existing))

        assert existing.exists()


class TestReadWriteTxt:
    """Tests for read_txt and write_txt functions."""

    def test_write_and_read_txt(self, temp_dir):
        """Test writing and reading text file."""
        txt_path = temp_dir / "test.txt"
        content = "Hello, World!\nLine 2"

        write_txt(str(txt_path), content)
        result = read_txt(str(txt_path))

        assert result == content

    def test_write_empty_string(self, temp_dir):
        """Test writing empty string."""
        txt_path = temp_dir / "empty.txt"

        write_txt(str(txt_path), "")
        result = read_txt(str(txt_path))

        assert result == ""

    def test_write_unicode(self, temp_dir):
        """Test writing unicode content."""
        txt_path = temp_dir / "unicode.txt"
        content = "Hello 世界 🌍"

        write_txt(str(txt_path), content)
        result = read_txt(str(txt_path))

        assert result == content


class TestReadWriteJson:
    """Tests for read_json and dump_json functions."""

    def test_write_and_read_json(self, temp_dir):
        """Test writing and reading JSON file."""
        json_path = temp_dir / "test.json"
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}

        dump_json(str(json_path), data)
        result = read_json(str(json_path))

        assert result == data

    def test_write_nested_json(self, temp_dir):
        """Test writing nested JSON structure."""
        json_path = temp_dir / "nested.json"
        data = {
            "level1": {
                "level2": {
                    "level3": "deep"
                }
            }
        }

        dump_json(str(json_path), data)
        result = read_json(str(json_path))

        assert result == data

    def test_write_list_json(self, temp_dir):
        """Test writing JSON list."""
        json_path = temp_dir / "list.json"
        data = [{"id": 1}, {"id": 2}, {"id": 3}]

        dump_json(str(json_path), data)
        result = read_json(str(json_path))

        assert result == data

    def test_write_empty_json(self, temp_dir):
        """Test writing empty JSON object."""
        json_path = temp_dir / "empty.json"

        dump_json(str(json_path), {})
        result = read_json(str(json_path))

        assert result == {}


class TestWriteFrames:
    """Tests for write_frames function."""

    def test_write_frames_success(self):
        """Test writing frames from capture to writer."""
        mock_cap = MagicMock()
        mock_out = MagicMock()

        # Simulate 3 frames then end
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.ones((480, 640, 3), dtype=np.uint8)
        frame3 = np.full((480, 640, 3), 128, dtype=np.uint8)

        mock_cap.isOpened.side_effect = [True, True, True, True]
        mock_cap.read.side_effect = [
            (True, frame1),
            (True, frame2),
            (True, frame3),
            (False, None),
        ]

        write_frames(mock_cap, mock_out)

        assert mock_out.write.call_count == 3

    def test_write_frames_empty_video(self):
        """Test writing frames when video has no frames."""
        mock_cap = MagicMock()
        mock_out = MagicMock()

        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)

        write_frames(mock_cap, mock_out)

        mock_out.write.assert_not_called()

    def test_write_frames_video_not_opened(self):
        """Test writing frames when video is not opened."""
        mock_cap = MagicMock()
        mock_out = MagicMock()

        mock_cap.isOpened.return_value = False

        write_frames(mock_cap, mock_out)

        mock_out.write.assert_not_called()
        mock_cap.read.assert_not_called()


class TestGetVideoMeta:
    """Tests for get_video_meta function."""

    def test_get_video_meta_moviepy_success(self):
        """Test getting video metadata using MoviePy."""
        mock_clip = MagicMock()
        mock_clip.fps = 30.0
        mock_clip.duration = 10.0
        mock_clip.w = 1920
        mock_clip.h = 1080
        mock_clip.reader = MagicMock()
        mock_clip.reader.nframes = 300
        mock_clip.__enter__ = MagicMock(return_value=mock_clip)
        mock_clip.__exit__ = MagicMock(return_value=False)

        with patch("vlm_aug.utils.helper.VideoFileClip", return_value=mock_clip):
            frame_count, fps, size = get_video_meta("/fake/video.mp4")

            assert frame_count == 300  # 10.0 * 30.0
            assert fps == 30.0
            assert size == (1920, 1080)

    def test_get_video_meta_moviepy_no_duration(self):
        """Test getting video metadata when duration is 0."""
        mock_clip = MagicMock()
        mock_clip.fps = 24.0
        mock_clip.duration = 0.0
        mock_clip.w = 640
        mock_clip.h = 480
        mock_clip.reader = MagicMock()
        mock_clip.reader.nframes = 100
        mock_clip.__enter__ = MagicMock(return_value=mock_clip)
        mock_clip.__exit__ = MagicMock(return_value=False)

        with patch("vlm_aug.utils.helper.VideoFileClip", return_value=mock_clip):
            frame_count, fps, size = get_video_meta("/fake/video.mp4")

            assert frame_count == 100  # Uses nframes fallback
            assert fps == 24.0
            assert size == (640, 480)

    def test_get_video_meta_moviepy_no_fps(self):
        """Test getting video metadata when fps is None."""
        mock_clip = MagicMock()
        mock_clip.fps = None
        mock_clip.duration = 5.0
        mock_clip.w = 640
        mock_clip.h = 480
        mock_clip.reader = MagicMock()
        mock_clip.reader.nframes = 0
        mock_clip.__enter__ = MagicMock(return_value=mock_clip)
        mock_clip.__exit__ = MagicMock(return_value=False)

        with patch("vlm_aug.utils.helper.VideoFileClip", return_value=mock_clip):
            frame_count, fps, size = get_video_meta("/fake/video.mp4")

            assert fps == 30.0  # Default FPS
            assert frame_count == 150  # 5.0 * 30.0
            assert size == (640, 480)

    def test_get_video_meta_moviepy_no_reader(self):
        """Test getting video metadata when reader attribute is missing."""
        mock_clip = MagicMock()
        mock_clip.fps = 25.0
        mock_clip.duration = None
        mock_clip.w = 800
        mock_clip.h = 600
        mock_clip.reader = None
        mock_clip.__enter__ = MagicMock(return_value=mock_clip)
        mock_clip.__exit__ = MagicMock(return_value=False)

        with patch("vlm_aug.utils.helper.VideoFileClip", return_value=mock_clip):
            frame_count, fps, size = get_video_meta("/fake/video.mp4")

            assert frame_count == 0  # No duration, no nframes
            assert fps == 25.0
            assert size == (800, 600)

    def test_get_video_meta_opencv_fallback(self):
        """Test falling back to OpenCV when MoviePy fails."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 100,    # CAP_PROP_FRAME_COUNT = 7 (but we use index)
            5: 29.97,  # CAP_PROP_FPS
            3: 1280,   # CAP_PROP_FRAME_WIDTH
            4: 720,    # CAP_PROP_FRAME_HEIGHT
        }.get(prop, 0)

        with patch("vlm_aug.utils.helper.VideoFileClip", side_effect=Exception("MoviePy error")):
            with patch("vlm_aug.utils.helper.cv2.VideoCapture", return_value=mock_cap):
                with patch("vlm_aug.utils.helper.cv2.CAP_PROP_FRAME_COUNT", 0):
                    with patch("vlm_aug.utils.helper.cv2.CAP_PROP_FPS", 5):
                        with patch("vlm_aug.utils.helper.cv2.CAP_PROP_FRAME_WIDTH", 3):
                            with patch("vlm_aug.utils.helper.cv2.CAP_PROP_FRAME_HEIGHT", 4):
                                frame_count, fps, size = get_video_meta("/fake/video.mp4")

                                assert frame_count == 100
                                assert fps == 29.97
                                assert size == (1280, 720)
                                mock_cap.release.assert_called_once()

    def test_get_video_meta_opencv_no_fps(self):
        """Test OpenCV fallback when FPS is 0."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 50,    # FRAME_COUNT
            5: 0,     # FPS (0 - should default to 30)
            3: 640,   # WIDTH
            4: 480,   # HEIGHT
        }.get(prop, 0)

        with patch("vlm_aug.utils.helper.VideoFileClip", side_effect=Exception("MoviePy error")):
            with patch("vlm_aug.utils.helper.cv2.VideoCapture", return_value=mock_cap):
                with patch("vlm_aug.utils.helper.cv2.CAP_PROP_FRAME_COUNT", 0):
                    with patch("vlm_aug.utils.helper.cv2.CAP_PROP_FPS", 5):
                        with patch("vlm_aug.utils.helper.cv2.CAP_PROP_FRAME_WIDTH", 3):
                            with patch("vlm_aug.utils.helper.cv2.CAP_PROP_FRAME_HEIGHT", 4):
                                frame_count, fps, size = get_video_meta("/fake/video.mp4")

                                assert fps == 30.0  # Default FPS

    def test_get_video_meta_both_fail(self):
        """Test RuntimeError when both MoviePy and OpenCV fail."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("vlm_aug.utils.helper.VideoFileClip", side_effect=Exception("MoviePy error")):
            with patch("vlm_aug.utils.helper.cv2.VideoCapture", return_value=mock_cap):
                with pytest.raises(RuntimeError) as exc_info:
                    get_video_meta("/fake/video.mp4")

                assert "Failed to open video" in str(exc_info.value)


class TestWriteVideo:
    """Tests for write_video function."""

    def test_write_video_success(self):
        """Test writing video successfully."""
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.ones((480, 640, 3), dtype=np.uint8) * 255,
        ]
        mock_clip = MagicMock()

        with patch("vlm_aug.utils.helper.ImageSequenceClip", return_value=mock_clip) as mock_isc:
            write_video(frames, "/fake/output.mp4", fps=30, size=(640, 480))

            mock_isc.assert_called_once()
            mock_clip.write_videofile.assert_called_once_with(
                "/fake/output.mp4",
                fps=30,
                codec="libx264",
                audio=False,
                verbose=False,
                logger=None,
            )
            mock_clip.close.assert_called_once()

    def test_write_video_empty_frames_raises_error(self):
        """Test that empty frames raises RuntimeError."""
        with pytest.raises(RuntimeError) as exc_info:
            write_video([], "/fake/output.mp4", fps=30, size=(640, 480))

        assert "No frames to write" in str(exc_info.value)

    def test_write_video_resizes_frames(self):
        """Test that frames are resized if they don't match target size."""
        # Create a frame with wrong size
        wrong_size_frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # 1280x720
        frames = [wrong_size_frame]
        mock_clip = MagicMock()

        with patch("vlm_aug.utils.helper.ImageSequenceClip", return_value=mock_clip) as mock_isc:
            with patch("vlm_aug.utils.helper.cv2.resize") as mock_resize:
                mock_resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

                write_video(frames, "/fake/output.mp4", fps=30, size=(640, 480))

                mock_resize.assert_called_once()
                mock_isc.assert_called_once()

    def test_write_video_no_resize_when_size_matches(self):
        """Test that frames are not resized when size matches."""
        # Create a mock frame with correct shape attributes (height=480, width=640)
        mock_frame = MagicMock()
        mock_frame.shape = (480, 640, 3)  # shape[0]=height, shape[1]=width
        frames = [mock_frame]
        mock_clip = MagicMock()

        with patch("vlm_aug.utils.helper.ImageSequenceClip", return_value=mock_clip) as mock_isc:
            with patch("vlm_aug.utils.helper.cv2.resize") as mock_resize:
                write_video(frames, "/fake/output.mp4", fps=30, size=(640, 480))

                # When (shape[1], shape[0]) == (w, h), resize should not be called
                mock_resize.assert_not_called()
                # Verify the frame was passed to ImageSequenceClip unchanged
                call_args = mock_isc.call_args[0][0]
                assert mock_frame in call_args


class TestParseVideoActionIndices:
    """Tests for parse_video_action_indices function."""

    def test_single_action(self):
        """Single action format parses to a one-element list, not concurrent."""
        assert parse_video_action_indices("01_video") == ([1], False)
        assert parse_video_action_indices("07_video") == ([7], False)

    def test_single_action_two_operator_mode(self):
        """Single action format is unaffected by two-operator mode."""
        assert parse_video_action_indices("03_video", two_operator_mode=True) == ([3], False)

    def test_concurrent_skipped_when_two_operator_off(self):
        """Concurrent format is skipped (empty, not concurrent) when two-operator mode is off."""
        assert parse_video_action_indices("01-03_video") == ([], False)

    def test_concurrent_parsed_when_two_operator_on(self):
        """Concurrent format parses to all indices when two-operator mode is on."""
        assert parse_video_action_indices("01-03_video", two_operator_mode=True) == ([1, 3], True)

    def test_concurrent_three_actions(self):
        """Concurrent format handles more than two actions."""
        assert parse_video_action_indices("02-05-07_video", two_operator_mode=True) == ([2, 5, 7], True)

    def test_custom_action_sep(self):
        """A non-default action separator is honored when splitting off the keyword."""
        assert parse_video_action_indices("04|clip", action_sep="|") == ([4], False)


class TestFormatConcurrentActions:
    """Tests for format_concurrent_actions function."""

    def test_formats_indices_and_descriptions(self):
        """Descriptions are cleaned, first-letter-lowercased, and prefixed with their index."""
        result = format_concurrent_actions(["Picking up the item.", "Inspecting label"], [1, 3])
        assert result == "(1) picking up the item (3) inspecting label"

    def test_single_action(self):
        """A single description formats with its index."""
        assert format_concurrent_actions(["Tighten the screw"], [2]) == "(2) tighten the screw"

    def test_strips_leading_numbers_via_clean_sentence(self):
        """Leading numbering is stripped by clean_sentence before formatting."""
        assert format_concurrent_actions(["1. Open the door"], [5]) == "(5) open the door"

    def test_empty_inputs(self):
        """Empty inputs produce an empty string."""
        assert format_concurrent_actions([], []) == ""
