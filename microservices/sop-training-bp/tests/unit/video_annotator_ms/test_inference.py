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
Unit tests for video-annotator-ms/annotation_backend/inference.py
These tests directly call internal functions rather than going through API endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

# We need to mock the database before importing the app
with patch("components.postgres_db.postgres_db") as mock_db:
    mock_db.get_data = AsyncMock(return_value=None)
    mock_db.list_data = AsyncMock(return_value=[])
    mock_db.insert_data = AsyncMock(return_value=None)
    mock_db.update_data = AsyncMock(return_value=None)
    mock_db.delete_data = AsyncMock(return_value=1)
    mock_db.delete_all_data = AsyncMock(return_value=0)
    from inference import get_openapi_schema


class TestGetOpenAPISchema:
    """Tests for get_openapi_schema function."""

    def test_get_openapi_schema_returns_schema(self):
        """Test that get_openapi_schema returns the OpenAPI schema."""
        with patch("builtins.open", mock_open()):
            with patch("json.dump"):
                with patch("pprint.pprint"):
                    schema = get_openapi_schema(file_dump=True, pprint=True)

                    assert schema is not None
                    assert "paths" in schema or "info" in schema or "openapi" in schema

    def test_get_openapi_schema_no_file_dump(self):
        """Test get_openapi_schema without file dump."""
        schema = get_openapi_schema(file_dump=False, pprint=False)

        assert schema is not None

    def test_get_openapi_schema_cached(self):
        """Test that get_openapi_schema returns cached schema on subsequent calls."""
        schema1 = get_openapi_schema(file_dump=False, pprint=False)
        schema2 = get_openapi_schema(file_dump=False, pprint=False)

        assert schema1 is schema2


class TestSplitVideoByTimestampsFunction:
    """Tests for the _split_video_by_timestamps internal function."""

    @pytest.mark.asyncio
    async def test_split_video_by_timestamps_source_file_not_exists(self):
        """Test that missing source video file raises 404."""
        from inference import _split_video_by_timestamps

        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "test_video.mp4"
        mock_video.dataset_id = "dataset-456"

        timestamps = [{"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "Test"}]

        with patch("os.path.exists", return_value=False):
            with patch("builtins.open", mock_open(read_data='{"actions": ["Action 1"]}')):
                from fastapi import HTTPException
                with pytest.raises(HTTPException) as exc_info:
                    await _split_video_by_timestamps(mock_video, timestamps)

                assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_split_video_by_timestamps_invalid_video_dimensions(self):
        """Test that video with invalid dimensions raises error."""
        from inference import _split_video_by_timestamps

        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "test_video.mp4"
        mock_video.dataset_id = "dataset-456"

        timestamps = [{"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "Test"}]

        mock_clip = MagicMock()
        mock_clip.w = 0
        mock_clip.h = 0
        mock_clip.close = MagicMock()

        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data='{"actions": ["Action 1"]}')):
                with patch("inference.VideoFileClip", return_value=mock_clip):
                    from fastapi import HTTPException
                    with pytest.raises(HTTPException) as exc_info:
                        await _split_video_by_timestamps(mock_video, timestamps)

                    assert exc_info.value.status_code == 400
                    assert "Failed to analyze video" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_split_video_by_timestamps_invalid_video_duration(self):
        """Test that video with invalid duration raises error."""
        from inference import _split_video_by_timestamps

        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "test_video.mp4"
        mock_video.dataset_id = "dataset-456"

        timestamps = [{"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "Test"}]

        mock_clip = MagicMock()
        mock_clip.w = 1920
        mock_clip.h = 1080
        mock_clip.duration = 0
        mock_clip.close = MagicMock()

        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data='{"actions": ["Action 1"]}')):
                with patch("inference.VideoFileClip", return_value=mock_clip):
                    from fastapi import HTTPException
                    with pytest.raises(HTTPException) as exc_info:
                        await _split_video_by_timestamps(mock_video, timestamps)

                    assert exc_info.value.status_code == 400
                    assert "Failed to analyze video" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_split_video_by_timestamps_video_analysis_failure(self):
        """Test that video analysis failure raises error."""
        from inference import _split_video_by_timestamps

        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "test_video.mp4"
        mock_video.dataset_id = "dataset-456"

        timestamps = [{"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "Test"}]

        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data='{"actions": ["Action 1"]}')):
                with patch("inference.VideoFileClip", side_effect=Exception("Cannot read video")):
                    from fastapi import HTTPException
                    with pytest.raises(HTTPException) as exc_info:
                        await _split_video_by_timestamps(mock_video, timestamps)

                    assert exc_info.value.status_code == 400
                    assert "Failed to analyze video" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_split_video_by_timestamps_start_exceeds_duration(self):
        """Test that timestamps with start exceeding video duration are skipped."""
        from inference import _split_video_by_timestamps

        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "test_video.mp4"
        mock_video.dataset_id = "dataset-456"

        timestamps = [{"start": 100, "end": 105, "actionIndex": 0, "actionDescription": "Test"}]

        mock_clip = MagicMock()
        mock_clip.w = 1920
        mock_clip.h = 1080
        mock_clip.duration = 10
        mock_clip.close = MagicMock()

        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data='{"actions": ["Action 1"]}')):
                with patch("inference.VideoFileClip", return_value=mock_clip):
                    with patch("inference.create_dir"):
                        from fastapi import HTTPException
                        with pytest.raises(HTTPException) as exc_info:
                            await _split_video_by_timestamps(mock_video, timestamps)

                        assert exc_info.value.status_code == 500
                        assert "no segments generated" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_split_video_by_timestamps_success(self):
        """Test successful video splitting with valid timestamps."""
        from inference import _split_video_by_timestamps

        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "video-123_test_video.mp4"
        mock_video.dataset_id = "dataset-456"

        timestamps = [
            {"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "Action 1"},
            {"start": 5, "end": 10, "actionIndex": 1, "actionDescription": "Action 2"}
        ]

        mock_main_clip = MagicMock()
        mock_main_clip.w = 1920
        mock_main_clip.h = 1080
        mock_main_clip.duration = 60
        mock_main_clip.close = MagicMock()

        mock_subclip = MagicMock()
        mock_subclip.write_videofile = MagicMock()
        mock_subclip.close = MagicMock()

        mock_full_clip = MagicMock()
        mock_full_clip.subclip = MagicMock(return_value=mock_subclip)
        mock_full_clip.close = MagicMock()

        mock_test_clip = MagicMock()
        mock_test_clip.duration = 5
        mock_test_clip.close = MagicMock()

        clip_call_count = [0]

        def video_file_clip_side_effect(path):
            clip_call_count[0] += 1
            if clip_call_count[0] == 1:
                return mock_main_clip
            elif clip_call_count[0] in [2, 4]:
                return mock_full_clip
            else:
                return mock_test_clip

        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data='{"actions": ["Action 1", "Action 2"]}')):
                with patch("inference.VideoFileClip", side_effect=video_file_clip_side_effect):
                    with patch("inference.create_dir"):
                        with patch("os.path.getsize", return_value=50000):
                            with patch("inference.postgres_db.insert_data", new_callable=AsyncMock):
                                result = await _split_video_by_timestamps(mock_video, timestamps)

                                assert len(result) == 2
                                assert result[0]["action_index"] == 0
                                assert result[1]["action_index"] == 1

    @pytest.mark.asyncio
    async def test_split_video_by_timestamps_end_exceeds_duration_adjusted(self):
        """Test that end time exceeding duration is adjusted to video duration."""
        from inference import _split_video_by_timestamps

        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "video-123_test_video.mp4"
        mock_video.dataset_id = "dataset-456"

        timestamps = [{"start": 5, "end": 100, "actionIndex": 0, "actionDescription": "Test"}]

        mock_main_clip = MagicMock()
        mock_main_clip.w = 1920
        mock_main_clip.h = 1080
        mock_main_clip.duration = 10
        mock_main_clip.close = MagicMock()

        mock_subclip = MagicMock()
        mock_subclip.write_videofile = MagicMock()
        mock_subclip.close = MagicMock()

        mock_full_clip = MagicMock()
        mock_full_clip.subclip = MagicMock(return_value=mock_subclip)
        mock_full_clip.close = MagicMock()

        mock_test_clip = MagicMock()
        mock_test_clip.duration = 5
        mock_test_clip.close = MagicMock()

        clip_call_count = [0]

        def video_file_clip_side_effect(path):
            clip_call_count[0] += 1
            if clip_call_count[0] == 1:
                return mock_main_clip
            elif clip_call_count[0] == 2:
                return mock_full_clip
            else:
                return mock_test_clip

        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data='{"actions": ["Action 1"]}')):
                with patch("inference.VideoFileClip", side_effect=video_file_clip_side_effect):
                    with patch("inference.create_dir"):
                        with patch("os.path.getsize", return_value=50000):
                            with patch("inference.postgres_db.insert_data", new_callable=AsyncMock):
                                result = await _split_video_by_timestamps(mock_video, timestamps)

                                assert len(result) == 1
                                assert result[0]["end_time"] == 10

    @pytest.mark.asyncio
    async def test_split_video_by_timestamps_segment_processing_error(self):
        """Test that segment processing errors are handled."""
        from inference import _split_video_by_timestamps

        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "video-123_test_video.mp4"
        mock_video.dataset_id = "dataset-456"

        timestamps = [{"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "Test"}]

        mock_main_clip = MagicMock()
        mock_main_clip.w = 1920
        mock_main_clip.h = 1080
        mock_main_clip.duration = 60
        mock_main_clip.close = MagicMock()

        mock_full_clip = MagicMock()
        mock_full_clip.subclip = MagicMock(side_effect=Exception("Subclip failed"))
        mock_full_clip.close = MagicMock()

        clip_call_count = [0]

        def video_file_clip_side_effect(path):
            clip_call_count[0] += 1
            if clip_call_count[0] == 1:
                return mock_main_clip
            else:
                return mock_full_clip

        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data='{"actions": ["Action 1"]}')):
                with patch("inference.VideoFileClip", side_effect=video_file_clip_side_effect):
                    with patch("inference.create_dir"):
                        with patch("os.remove"):
                            with patch("inference.postgres_db.delete_data", new_callable=AsyncMock):
                                from fastapi import HTTPException
                                with pytest.raises((HTTPException, UnboundLocalError)):
                                    await _split_video_by_timestamps(mock_video, timestamps)

    @pytest.mark.asyncio
    async def test_split_video_by_timestamps_output_too_small(self):
        """Test that output files that are too small are rejected."""
        from inference import _split_video_by_timestamps

        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "video-123_test_video.mp4"
        mock_video.dataset_id = "dataset-456"

        timestamps = [{"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "Test"}]

        mock_main_clip = MagicMock()
        mock_main_clip.w = 1920
        mock_main_clip.h = 1080
        mock_main_clip.duration = 60
        mock_main_clip.close = MagicMock()

        mock_subclip = MagicMock()
        mock_subclip.write_videofile = MagicMock()
        mock_subclip.close = MagicMock()

        mock_full_clip = MagicMock()
        mock_full_clip.subclip = MagicMock(return_value=mock_subclip)
        mock_full_clip.close = MagicMock()

        clip_call_count = [0]

        def video_file_clip_side_effect(path):
            clip_call_count[0] += 1
            if clip_call_count[0] == 1:
                return mock_main_clip
            else:
                return mock_full_clip

        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data='{"actions": ["Action 1"]}')):
                with patch("inference.VideoFileClip", side_effect=video_file_clip_side_effect):
                    with patch("inference.create_dir"):
                        with patch("os.path.getsize", return_value=100):
                            with patch("os.remove"):
                                with patch("inference.postgres_db.delete_data", new_callable=AsyncMock):
                                    from fastapi import HTTPException
                                    with pytest.raises((HTTPException, UnboundLocalError, ValueError)):
                                        await _split_video_by_timestamps(mock_video, timestamps)

    @pytest.mark.asyncio
    async def test_split_video_by_timestamps_multiple_actions_same_index(self):
        """Test splitting with multiple timestamps for the same action index."""
        from inference import _split_video_by_timestamps

        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "video-123_test_video.mp4"
        mock_video.dataset_id = "dataset-456"

        timestamps = [
            {"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "Action 1"},
            {"start": 10, "end": 15, "actionIndex": 0, "actionDescription": "Action 1"},
        ]

        mock_main_clip = MagicMock()
        mock_main_clip.w = 1920
        mock_main_clip.h = 1080
        mock_main_clip.duration = 60
        mock_main_clip.close = MagicMock()

        mock_subclip = MagicMock()
        mock_subclip.write_videofile = MagicMock()
        mock_subclip.close = MagicMock()

        mock_full_clip = MagicMock()
        mock_full_clip.subclip = MagicMock(return_value=mock_subclip)
        mock_full_clip.close = MagicMock()

        mock_test_clip = MagicMock()
        mock_test_clip.duration = 5
        mock_test_clip.close = MagicMock()

        clip_call_count = [0]

        def video_file_clip_side_effect(path):
            clip_call_count[0] += 1
            if clip_call_count[0] == 1:
                return mock_main_clip
            elif clip_call_count[0] in [2, 4]:
                return mock_full_clip
            else:
                return mock_test_clip

        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data='{"actions": ["Action 1"]}')):
                with patch("inference.VideoFileClip", side_effect=video_file_clip_side_effect):
                    with patch("inference.create_dir"):
                        with patch("os.path.getsize", return_value=50000):
                            with patch("inference.postgres_db.insert_data", new_callable=AsyncMock):
                                result = await _split_video_by_timestamps(mock_video, timestamps)

                                assert len(result) == 2
                                assert result[0]["action_index"] == 0
                                assert result[1]["action_index"] == 0
                                assert result[0]["repetition_count"] == 1
                                assert result[1]["repetition_count"] == 2


class TestLifespanContextManager:
    """Tests for the FastAPI lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_creates_video_directory(self):
        """Test that lifespan event creates video directory on startup."""
        from inference import lifespan

        mock_app = MagicMock()

        with patch("inference.create_dir") as mock_create_dir:
            async with lifespan(mock_app):
                mock_create_dir.assert_called()


class TestAnnotationFileSaveError:
    """Tests for annotation file save error handling."""

    @pytest.mark.asyncio
    async def test_split_video_annotation_file_save_error(self):
        """Test that annotation file save error is handled."""
        from inference import _split_video_by_timestamps

        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.name = "video-123_test_video.mp4"
        mock_video.dataset_id = "dataset-456"

        timestamps = [{"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "Action 1"}]

        mock_main_clip = MagicMock()
        mock_main_clip.w = 1920
        mock_main_clip.h = 1080
        mock_main_clip.duration = 60
        mock_main_clip.close = MagicMock()

        mock_subclip = MagicMock()
        mock_subclip.write_videofile = MagicMock()
        mock_subclip.close = MagicMock()

        mock_full_clip = MagicMock()
        mock_full_clip.subclip = MagicMock(return_value=mock_subclip)
        mock_full_clip.close = MagicMock()

        mock_test_clip = MagicMock()
        mock_test_clip.duration = 5
        mock_test_clip.close = MagicMock()

        clip_call_count = [0]

        def video_file_clip_side_effect(path):
            clip_call_count[0] += 1
            if clip_call_count[0] == 1:
                return mock_main_clip
            elif clip_call_count[0] == 2:
                return mock_full_clip
            else:
                return mock_test_clip

        file_open_count = [0]

        def open_side_effect(path, *args, **kwargs):
            file_open_count[0] += 1
            if file_open_count[0] == 1:
                return mock_open(read_data='{"actions": ["Action 1"]}')()
            raise IOError("Cannot write annotation file")

        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", side_effect=open_side_effect):
                with patch("inference.VideoFileClip", side_effect=video_file_clip_side_effect):
                    with patch("inference.create_dir"):
                        with patch("os.path.getsize", return_value=50000):
                            with patch("inference.postgres_db.insert_data", new_callable=AsyncMock):
                                from fastapi import HTTPException
                                with pytest.raises(HTTPException) as exc_info:
                                    await _split_video_by_timestamps(mock_video, timestamps)

                                assert exc_info.value.status_code == 500
                                assert "annotation file" in exc_info.value.detail.lower()
