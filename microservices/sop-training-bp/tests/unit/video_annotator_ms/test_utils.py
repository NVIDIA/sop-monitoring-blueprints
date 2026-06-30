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
Unit tests for video-annotator-ms/annotation_backend/utils/utils.py
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.utils import (
    create_dir,
    clean_up_file,
    modify_directory_permission,
    convert_to_h264,
    check_if_h264_encoded,
)


class TestCreateDir:
    """Tests for create_dir function."""

    def test_create_new_directory(self, temp_dir):
        """Test creating a new directory."""
        new_dir = temp_dir / "new_folder"
        assert not new_dir.exists()

        result = create_dir(str(new_dir))

        assert result is True
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_existing_directory_returns_false(self, temp_dir):
        """Test that existing directory returns False."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()

        result = create_dir(str(existing_dir))

        assert result is False
        assert existing_dir.exists()

    def test_create_nested_directories(self, temp_dir):
        """Test creating nested directories."""
        nested_dir = temp_dir / "level1" / "level2" / "level3"

        result = create_dir(str(nested_dir))

        assert result is True
        assert nested_dir.exists()


class TestCleanUpFile:
    """Tests for clean_up_file function."""

    def test_clean_up_existing_file(self, temp_dir):
        """Test cleaning up an existing file."""
        test_file = temp_dir / "to_delete.txt"
        test_file.write_text("content")
        assert test_file.exists()

        result = clean_up_file(str(test_file))

        assert result is True
        assert not test_file.exists()

    def test_clean_up_nonexistent_file(self, temp_dir):
        """Test cleaning up a file that doesn't exist."""
        nonexistent = temp_dir / "does_not_exist.txt"

        result = clean_up_file(str(nonexistent))

        assert result is False


class TestModifyDirectoryPermission:
    """Tests for modify_directory_permission function."""

    def test_modify_permission_success(self, temp_dir):
        """Test modifying directory permissions successfully."""
        test_dir = temp_dir / "perm_test"
        test_dir.mkdir()

        result = modify_directory_permission(str(test_dir), 0o755)

        assert result is True
        # Verify permission was changed (masking with 0o777 to ignore special bits)
        assert (os.stat(str(test_dir)).st_mode & 0o777) == 0o755

    def test_modify_permission_default_777(self, temp_dir):
        """Test default permission is 0o777."""
        test_dir = temp_dir / "default_perm"
        test_dir.mkdir()

        result = modify_directory_permission(str(test_dir))

        assert result is True

    def test_modify_permission_invalid_path(self):
        """Test modifying permission on invalid path returns False."""
        result = modify_directory_permission("/nonexistent/path/12345")

        assert result is False


class TestCheckIfH264Encoded:
    """Tests for check_if_h264_encoded function."""

    @pytest.mark.asyncio
    async def test_check_h264_encoded_true(self):
        """Test detecting H264 encoded video."""
        # Create mock video stream with h264 codec
        mock_stream = MagicMock()
        mock_stream.codec.name.lower.return_value = "h264"

        mock_container = MagicMock()
        mock_container.streams.video = [mock_stream]
        mock_container.__enter__ = MagicMock(return_value=mock_container)
        mock_container.__exit__ = MagicMock(return_value=False)

        with patch("utils.utils.av.open", return_value=mock_container):
            result = await check_if_h264_encoded("/path/to/video.mp4")

        assert result is True

    @pytest.mark.asyncio
    async def test_check_h264_encoded_avc_codec(self):
        """Test detecting AVC (H264) encoded video."""
        mock_stream = MagicMock()
        mock_stream.codec.name.lower.return_value = "avc"

        mock_container = MagicMock()
        mock_container.streams.video = [mock_stream]
        mock_container.__enter__ = MagicMock(return_value=mock_container)
        mock_container.__exit__ = MagicMock(return_value=False)

        with patch("utils.utils.av.open", return_value=mock_container):
            result = await check_if_h264_encoded("/path/to/video.mp4")

        assert result is True

    @pytest.mark.asyncio
    async def test_check_h264_encoded_libx264_codec(self):
        """Test detecting libx264 (H264) encoded video."""
        mock_stream = MagicMock()
        mock_stream.codec.name.lower.return_value = "libx264"

        mock_container = MagicMock()
        mock_container.streams.video = [mock_stream]
        mock_container.__enter__ = MagicMock(return_value=mock_container)
        mock_container.__exit__ = MagicMock(return_value=False)

        with patch("utils.utils.av.open", return_value=mock_container):
            result = await check_if_h264_encoded("/path/to/video.mp4")

        assert result is True

    @pytest.mark.asyncio
    async def test_check_h264_encoded_false_other_codec(self):
        """Test detecting non-H264 encoded video."""
        mock_stream = MagicMock()
        mock_stream.codec.name.lower.return_value = "vp9"

        mock_container = MagicMock()
        mock_container.streams.video = [mock_stream]
        mock_container.__enter__ = MagicMock(return_value=mock_container)
        mock_container.__exit__ = MagicMock(return_value=False)

        with patch("utils.utils.av.open", return_value=mock_container):
            result = await check_if_h264_encoded("/path/to/video.mp4")

        assert result is False

    @pytest.mark.asyncio
    async def test_check_h264_encoded_no_video_streams(self):
        """Test handling video with no video streams."""
        mock_container = MagicMock()
        mock_container.streams.video = []
        mock_container.__enter__ = MagicMock(return_value=mock_container)
        mock_container.__exit__ = MagicMock(return_value=False)

        with patch("utils.utils.av.open", return_value=mock_container):
            result = await check_if_h264_encoded("/path/to/video.mp4")

        assert result is False

    @pytest.mark.asyncio
    async def test_check_h264_encoded_exception_returns_false(self):
        """Test that exception during check returns False."""
        with patch("utils.utils.av.open", side_effect=Exception("Cannot open file")):
            result = await check_if_h264_encoded("/path/to/invalid.mp4")

        assert result is False


class TestConvertToH264:
    """Tests for convert_to_h264 function."""

    @pytest.mark.asyncio
    async def test_convert_already_h264_copies_file(self, temp_dir):
        """Test that already H264 video is just copied."""
        input_path = temp_dir / "input.mp4"
        output_path = temp_dir / "output.mp4"
        input_path.write_bytes(b"fake video content")

        with patch("utils.utils.check_if_h264_encoded", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = True

            result = await convert_to_h264(str(input_path), str(output_path))

        assert result == str(output_path)
        assert output_path.exists()
        assert output_path.read_bytes() == b"fake video content"

    @pytest.mark.asyncio
    async def test_convert_non_h264_video(self, temp_dir):
        """Test converting non-H264 video."""
        input_path = temp_dir / "input.avi"
        output_path = temp_dir / "output.mp4"
        input_path.write_bytes(b"fake avi content")

        # Create mock clip
        mock_clip = MagicMock()
        mock_clip.write_videofile = MagicMock()
        mock_clip.close = MagicMock()

        with patch("utils.utils.check_if_h264_encoded", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = False

            with patch("utils.utils.VideoFileClip", return_value=mock_clip):
                # Mock the output file creation
                def create_output_file(*args, **kwargs):
                    output_path.write_bytes(b"converted h264 content" * 100)

                mock_clip.write_videofile.side_effect = create_output_file

                result = await convert_to_h264(str(input_path), str(output_path))

        assert result == str(output_path)
        mock_clip.write_videofile.assert_called_once()
        mock_clip.close.assert_called_once()

        # Verify write_videofile was called with correct codec
        call_kwargs = mock_clip.write_videofile.call_args[1]
        assert call_kwargs["codec"] == "libx264"
        assert call_kwargs["audio_codec"] == "aac"

    @pytest.mark.asyncio
    async def test_convert_output_file_not_found_raises(self, temp_dir):
        """Test that missing output file raises exception."""
        input_path = temp_dir / "input.avi"
        output_path = temp_dir / "output.mp4"
        input_path.write_bytes(b"fake avi content")

        mock_clip = MagicMock()
        mock_clip.write_videofile = MagicMock()  # Does nothing, file not created
        mock_clip.close = MagicMock()

        with patch("utils.utils.check_if_h264_encoded", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = False

            with patch("utils.utils.VideoFileClip", return_value=mock_clip):
                with pytest.raises(FileNotFoundError) as exc_info:
                    await convert_to_h264(str(input_path), str(output_path))

                assert "output file not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_convert_output_file_too_small_raises(self, temp_dir):
        """Test that too small output file raises exception."""
        input_path = temp_dir / "input.avi"
        output_path = temp_dir / "output.mp4"
        input_path.write_bytes(b"fake avi content")

        mock_clip = MagicMock()
        mock_clip.close = MagicMock()

        def create_small_file(*args, **kwargs):
            output_path.write_bytes(b"tiny")  # Less than 1KB

        mock_clip.write_videofile.side_effect = create_small_file

        with patch("utils.utils.check_if_h264_encoded", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = False

            with patch("utils.utils.VideoFileClip", return_value=mock_clip):
                with pytest.raises(ValueError) as exc_info:
                    await convert_to_h264(str(input_path), str(output_path))

                assert "too small" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_convert_cleans_up_on_error(self, temp_dir):
        """Test that output file is cleaned up on error."""
        input_path = temp_dir / "input.avi"
        output_path = temp_dir / "output.mp4"
        input_path.write_bytes(b"fake avi content")

        mock_clip = MagicMock()
        mock_clip.close = MagicMock()

        def create_and_fail(*args, **kwargs):
            output_path.write_bytes(b"partial content")
            raise Exception("Conversion failed mid-process")

        mock_clip.write_videofile.side_effect = create_and_fail

        with patch("utils.utils.check_if_h264_encoded", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = False

            with patch("utils.utils.VideoFileClip", return_value=mock_clip):
                with pytest.raises(Exception) as exc_info:
                    await convert_to_h264(str(input_path), str(output_path))

                assert "Conversion failed" in str(exc_info.value)

        # Output file should be cleaned up
        assert not output_path.exists()

    @pytest.mark.asyncio
    async def test_convert_videofileclip_exception_cleans_up(self, temp_dir):
        """Test that exception in VideoFileClip is handled and cleaned up."""
        input_path = temp_dir / "input.avi"
        output_path = temp_dir / "output.mp4"
        input_path.write_bytes(b"fake avi content")

        with patch("utils.utils.check_if_h264_encoded", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = False

            with patch("utils.utils.VideoFileClip", side_effect=Exception("Cannot read video")):
                with pytest.raises(Exception) as exc_info:
                    await convert_to_h264(str(input_path), str(output_path))

                assert "Cannot read video" in str(exc_info.value)
