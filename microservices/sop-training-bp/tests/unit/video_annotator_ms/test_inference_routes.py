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
Unit tests for the FastAPI routes and request/validation helpers in
annotation_backend/inference.py.

These tests call the async route handlers and internal helpers directly with
mocked filesystem, DB (AsyncMock) and moviepy/ffmpeg dependencies. They cover the
testable branches that were previously uncovered (config loading, two-operator
toggle, upload/download routes, dataset clearing, split-video validation, and the
concurrent/single segment processors).
"""

from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

# Mock the database before importing the app module (mirrors test_inference.py).
with patch("components.postgres_db.postgres_db") as mock_db:
    mock_db.get_data = AsyncMock(return_value=None)
    mock_db.list_data = AsyncMock(return_value=[])
    mock_db.insert_data = AsyncMock(return_value=None)
    mock_db.update_data = AsyncMock(return_value=None)
    mock_db.delete_data = AsyncMock(return_value=1)
    mock_db.delete_all_data = AsyncMock(return_value=0)
    import inference
    from inference import (
        _is_small_relative_to_neighbors,
        _clips_to_merge_format,
        _load_merge_threshold,
        _merge_small_chunks,
        _process_concurrent_segment,
        _process_single_segment,
        _split_video_by_timestamps,
        _try_merge_one,
        _validate_video_file,
        clear_all_videos,
        clear_dataset,
        download_all_video_clips,
        download_chunk,
        download_video,
        get_chunk,
        get_datasets_info,
        lifespan,
        reset_actions,
        set_two_operator_mode,
        split_video,
        upload_actions,
        upload_video,
    )

from fastapi import HTTPException


def _make_video(**kwargs):
    v = MagicMock()
    v.id = kwargs.get("id", "video-123")
    v.name = kwargs.get("name", "video-123_test.mp4")
    v.dataset_id = kwargs.get("dataset_id", "ds-1")
    v.file_size = kwargs.get("file_size", 100000)
    v.mime_type = kwargs.get("mime_type", "video/mp4")
    v.created_at = kwargs.get("created_at", MagicMock(isoformat=lambda: "2026-01-01T00:00:00"))
    v.to_dict = MagicMock(return_value={"id": v.id, "name": v.name})
    return v


# --------------------------------------------------------------------------- #
# _load_merge_threshold
# --------------------------------------------------------------------------- #
class TestLoadMergeThreshold:
    def test_file_missing_returns_zero(self):
        with patch("builtins.open", side_effect=FileNotFoundError()):
            assert _load_merge_threshold() == 0.0

    def test_disabled_returns_zero(self):
        cfg = {"merge_small_chunks": {"enable": False, "threshold": 0.5}}
        with patch("builtins.open", mock_open()):
            with patch("inference._yaml.safe_load", return_value=cfg):
                assert _load_merge_threshold() == 0.0

    def test_enabled_returns_threshold(self):
        cfg = {"merge_small_chunks": {"enable": True, "threshold": 0.35}}
        with patch("builtins.open", mock_open()):
            with patch("inference._yaml.safe_load", return_value=cfg):
                assert _load_merge_threshold() == 0.35

    def test_missing_key_defaults(self):
        with patch("builtins.open", mock_open()):
            with patch("inference._yaml.safe_load", return_value={}):
                # No merge_small_chunks key -> enable defaults True, threshold 0.2
                assert _load_merge_threshold() == 0.2

    def test_empty_yaml_uses_default_threshold(self):
        with patch("builtins.open", mock_open()):
            with patch("inference._yaml.safe_load", return_value=None):
                assert _load_merge_threshold() == 0.2


# --------------------------------------------------------------------------- #
# set_two_operator_mode
# --------------------------------------------------------------------------- #
class TestSetTwoOperatorMode:
    @pytest.mark.asyncio
    async def test_dataset_not_found(self):
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=None)):
            with pytest.raises(HTTPException) as exc:
                await set_two_operator_mode("ds-1", {"two_operator_mode": True})
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_missing_body_key(self):
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=MagicMock())):
            with pytest.raises(HTTPException) as exc:
                await set_two_operator_mode("ds-1", {})
            assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_none_body(self):
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=MagicMock())):
            with pytest.raises(HTTPException) as exc:
                await set_two_operator_mode("ds-1", None)
            assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_success(self):
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=MagicMock())):
            with patch("inference.postgres_db.update_data", new=AsyncMock()) as upd:
                result = await set_two_operator_mode("ds-1", {"two_operator_mode": True})
                assert result["two_operator_mode"] is True
                assert result["status"] == "success"
                upd.assert_awaited()

    @pytest.mark.asyncio
    async def test_db_error_raises_500(self):
        with patch("inference.postgres_db.get_data", new=AsyncMock(side_effect=RuntimeError("boom"))):
            with pytest.raises(HTTPException) as exc:
                await set_two_operator_mode("ds-1", {"two_operator_mode": True})
            assert exc.value.status_code == 500


# --------------------------------------------------------------------------- #
# upload_video (uncovered branches)
# --------------------------------------------------------------------------- #
class TestUploadVideo:
    @pytest.mark.asyncio
    async def test_no_current_data_id(self):
        inference.current_data_id = None
        upload = MagicMock()
        upload.file.read = MagicMock(return_value=b"x")
        with pytest.raises(HTTPException) as exc:
            await upload_video(upload)
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_filename_normalized_and_too_small(self):
        # Covers the .mp4 normalization branch (209-211) and the too-small branch (263).
        inference.current_data_id = "ds-1"
        upload = MagicMock()
        upload.filename = "clip.avi"  # no .mp4 -> triggers normalization
        upload.file.read = MagicMock(return_value=b"tiny")  # < 1024 bytes
        with patch("inference.postgres_db.list_data", new=AsyncMock(return_value=[])):
            with patch("builtins.open", mock_open()):
                with patch("inference.clean_up_file"):
                    with pytest.raises(HTTPException) as exc:
                        await upload_video(upload)
        assert exc.value.status_code == 400
        assert "too small" in exc.value.detail.lower()
        inference.current_data_id = None

    @pytest.mark.asyncio
    async def test_existing_video_reuse_and_conversion_failure(self):
        inference.current_data_id = "ds-1"
        upload = MagicMock()
        upload.filename = "clip.mp4"
        upload.file.read = MagicMock(return_value=b"x" * 2048)
        existing = _make_video(name="abc_clip.mp4", id="abc")
        with patch("inference.postgres_db.list_data", new=AsyncMock(return_value=[existing])):
            with patch("builtins.open", mock_open()):
                with patch("inference.clean_up_file"):
                    with patch("inference.convert_to_h264", new=AsyncMock(side_effect=Exception("bad"))):
                        with pytest.raises(HTTPException) as exc:
                            await upload_video(upload)
        assert exc.value.status_code == 500
        assert "conversion failed" in exc.value.detail.lower()
        inference.current_data_id = None

    @pytest.mark.asyncio
    async def test_new_video_success_insert(self):
        inference.current_data_id = "ds-1"
        upload = MagicMock()
        upload.filename = "clip.mp4"
        upload.file.read = MagicMock(return_value=b"x" * 2048)
        with patch("inference.postgres_db.list_data", new=AsyncMock(return_value=[])):
            with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=None)):
                with patch("inference.postgres_db.insert_data", new=AsyncMock()) as ins:
                    with patch("builtins.open", mock_open()):
                        with patch("inference.clean_up_file"):
                            with patch("inference.convert_to_h264", new=AsyncMock(return_value="/x/clip.mp4")):
                                with patch("os.path.getsize", return_value=99999):
                                    resp = await upload_video(upload)
        assert resp.file_id is not None
        ins.assert_awaited()
        inference.current_data_id = None


# --------------------------------------------------------------------------- #
# upload_actions
# --------------------------------------------------------------------------- #
class TestUploadActions:
    @pytest.mark.asyncio
    async def test_success(self):
        upload = MagicMock()
        upload.filename = "actions.json"
        upload.read = AsyncMock(return_value=b'{"actions": ["a", "b"]}')
        with patch("inference.create_dir"):
            with patch("builtins.open", mock_open()):
                with patch("inference.postgres_db.insert_data", new=AsyncMock()):
                    resp = await upload_actions(upload)
        assert resp.actions_count == 2
        assert resp.status == "success"
        inference.current_data_id = None

    @pytest.mark.asyncio
    async def test_empty_actions_array(self):
        upload = MagicMock()
        upload.filename = "actions.json"
        upload.read = AsyncMock(return_value=b'{"actions": []}')
        with pytest.raises(HTTPException) as exc:
            await upload_actions(upload)
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        upload = MagicMock()
        upload.filename = "actions.json"
        upload.read = AsyncMock(return_value=b"not-json")
        with pytest.raises(HTTPException) as exc:
            await upload_actions(upload)
        assert exc.value.status_code == 400


# --------------------------------------------------------------------------- #
# get_datasets_info
# --------------------------------------------------------------------------- #
class TestGetDatasetsInfo:
    @pytest.mark.asyncio
    async def test_aggregates_clips(self):
        dataset = MagicMock()
        dataset.id = "ds-1"
        dataset.actions = ["a"]
        dataset.two_operator_mode = True
        video = _make_video()
        annotation = MagicMock()
        annotation.start_time = 0
        annotation.end_time = 5
        annotation.created_at = MagicMock()
        annotation.chunk_id = "chunk-1"
        annotation.action_description = "a"
        annotation.action_index = 0
        chunk = MagicMock()
        chunk.id = "chunk-1"
        chunk.name = "01_clip.mp4"

        async def list_side(model, condition=None):
            name = getattr(model, "__name__", str(model))
            if name == "Dataset":
                return [dataset]
            if name == "Video":
                return [video]
            if name == "Annotation":
                return [annotation]
            return []

        with patch("inference.postgres_db.list_data", new=AsyncMock(side_effect=list_side)):
            with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=chunk)):
                result = await get_datasets_info()
        assert "ds-1" in result
        assert result["ds-1"]["two_operator_mode"] is True
        assert video.id in result["ds-1"]["videos"]

    @pytest.mark.asyncio
    async def test_video_without_annotations_skipped(self):
        dataset = MagicMock()
        dataset.id = "ds-1"
        dataset.actions = ["a"]
        dataset.two_operator_mode = None
        video = _make_video()

        async def list_side(model, condition=None):
            name = getattr(model, "__name__", str(model))
            if name == "Dataset":
                return [dataset]
            if name == "Video":
                return [video]
            return []  # no annotations

        with patch("inference.postgres_db.list_data", new=AsyncMock(side_effect=list_side)):
            result = await get_datasets_info()
        assert result["ds-1"]["videos"] == {}
        assert result["ds-1"]["two_operator_mode"] is False

    @pytest.mark.asyncio
    async def test_db_error(self):
        with patch("inference.postgres_db.list_data", new=AsyncMock(side_effect=RuntimeError("x"))):
            with pytest.raises(HTTPException) as exc:
                await get_datasets_info()
            assert exc.value.status_code == 500


# --------------------------------------------------------------------------- #
# download_video
# --------------------------------------------------------------------------- #
class TestDownloadVideo:
    @pytest.mark.asyncio
    async def test_not_found(self):
        req = MagicMock()
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=None)):
            with pytest.raises(HTTPException) as exc:
                await download_video(req, "video-1")
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_file_missing_on_disk(self):
        req = MagicMock()
        req.headers = {}
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=_make_video())):
            with patch("os.path.exists", return_value=False):
                with pytest.raises(HTTPException) as exc:
                    await download_video(req, "video-1")
                assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_range_request_partial_content(self):
        req = MagicMock()
        req.headers = {"range": "bytes=0-99"}
        video = _make_video(file_size=1000)
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=video)):
            with patch("os.path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data=b"x" * 100)):
                    resp = await download_video(req, "video-1")
        assert resp.status_code == 206
        assert resp.headers["Content-Range"] == "bytes 0-99/1000"

    @pytest.mark.asyncio
    async def test_invalid_range_falls_back_to_full_file(self):
        # Covers line 618: invalid range header parse error -> FileResponse fallback.
        req = MagicMock()
        req.headers = {"range": "bytes=abc-def"}
        video = _make_video(file_size=1000)
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=video)):
            with patch("os.path.exists", return_value=True):
                resp = await download_video(req, "video-1")
        # FileResponse fallback
        assert getattr(resp, "status_code", 200) in (200, None)

    @pytest.mark.asyncio
    async def test_no_range_full_file(self):
        req = MagicMock()
        req.headers = {}
        video = _make_video(file_size=1000)
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=video)):
            with patch("os.path.exists", return_value=True):
                resp = await download_video(req, "video-1")
        assert resp is not None


# --------------------------------------------------------------------------- #
# get_chunk / download_chunk
# --------------------------------------------------------------------------- #
class TestGetChunk:
    @pytest.mark.asyncio
    async def test_not_found(self):
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=None)):
            with pytest.raises(HTTPException) as exc:
                await get_chunk("chunk-1")
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_success(self):
        chunk = MagicMock()
        chunk.id = "chunk-1"
        chunk.video_id = "video-1"
        chunk.name = "01_clip.mp4"
        chunk.action = "a"
        chunk.file_size = 5000
        chunk.mime_type = "video/mp4"
        chunk.created_at = MagicMock(isoformat=lambda: "2026-01-01T00:00:00")
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=chunk)):
            result = await get_chunk("chunk-1")
        assert result["id"] == "chunk-1"


class TestDownloadChunk:
    @pytest.mark.asyncio
    async def test_chunk_not_found(self):
        req = MagicMock()
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=None)):
            with pytest.raises(HTTPException) as exc:
                await download_chunk(req, "chunk-1")
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_parent_video_not_found(self):
        req = MagicMock()
        chunk = MagicMock()
        chunk.video_id = "video-1"
        chunk.name = "01_clip.mp4"
        with patch("inference.postgres_db.get_data", new=AsyncMock(side_effect=[chunk, None])):
            with pytest.raises(HTTPException) as exc:
                await download_chunk(req, "chunk-1")
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_file_missing_on_disk(self):
        # Covers lines 705-706.
        req = MagicMock()
        chunk = MagicMock()
        chunk.video_id = "video-1"
        chunk.name = "01_clip.mp4"
        chunk.mime_type = "video/mp4"
        video = _make_video()
        with patch("inference.postgres_db.get_data", new=AsyncMock(side_effect=[chunk, video])):
            with patch("os.path.exists", return_value=False):
                with pytest.raises(HTTPException) as exc:
                    await download_chunk(req, "chunk-1")
                assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_success(self):
        req = MagicMock()
        chunk = MagicMock()
        chunk.video_id = "video-1"
        chunk.name = "01_clip.mp4"
        chunk.mime_type = "video/mp4"
        video = _make_video()
        with patch("inference.postgres_db.get_data", new=AsyncMock(side_effect=[chunk, video])):
            with patch("os.path.exists", return_value=True):
                resp = await download_chunk(req, "chunk-1")
        assert resp is not None


# --------------------------------------------------------------------------- #
# download_all_video_clips (ZIP)
# --------------------------------------------------------------------------- #
class TestDownloadAllVideoClips:
    @pytest.mark.asyncio
    async def test_video_not_found(self):
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=None)):
            with pytest.raises(HTTPException) as exc:
                await download_all_video_clips("video-1")
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_no_chunks(self):
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=_make_video())):
            with patch("inference.postgres_db.list_data", new=AsyncMock(return_value=[])):
                with pytest.raises(HTTPException) as exc:
                    await download_all_video_clips("video-1")
                assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_zip_created(self):
        # Covers the ZIP-building body (760-811) with mocked filesystem/zipfile.
        chunk = MagicMock()
        chunk.name = "01_clip.mp4"
        video = _make_video()
        mock_zip = MagicMock()
        mock_zip.__enter__ = MagicMock(return_value=mock_zip)
        mock_zip.__exit__ = MagicMock(return_value=False)
        mock_temp = MagicMock()
        mock_temp.name = "/tmp/out.zip"
        mock_temp.__enter__ = MagicMock(return_value=mock_temp)
        mock_temp.__exit__ = MagicMock(return_value=False)
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=video)):
            with patch("inference.postgres_db.list_data", new=AsyncMock(return_value=[chunk])):
                with patch("tempfile.NamedTemporaryFile", return_value=mock_temp):
                    with patch("zipfile.ZipFile", return_value=mock_zip):
                        with patch("os.path.exists", return_value=True):
                            with patch("os.path.getsize", return_value=12345):
                                resp = await download_all_video_clips("video-1")
        assert resp is not None
        mock_zip.write.assert_called()

    @pytest.mark.asyncio
    async def test_chunk_file_missing_skipped(self):
        # exercises the "else" warning branch (clip file not found).
        chunk = MagicMock()
        chunk.name = "01_clip.mp4"
        video = _make_video()
        mock_zip = MagicMock()
        mock_zip.__enter__ = MagicMock(return_value=mock_zip)
        mock_zip.__exit__ = MagicMock(return_value=False)
        mock_temp = MagicMock()
        mock_temp.name = "/tmp/out.zip"
        mock_temp.__enter__ = MagicMock(return_value=mock_temp)
        mock_temp.__exit__ = MagicMock(return_value=False)

        # First exists() -> False (chunk skipped), then True for the zip-created check.
        exists_vals = iter([False, True])

        def exists_side(_path):
            try:
                return next(exists_vals)
            except StopIteration:
                return True

        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=video)):
            with patch("inference.postgres_db.list_data", new=AsyncMock(return_value=[chunk])):
                with patch("tempfile.NamedTemporaryFile", return_value=mock_temp):
                    with patch("zipfile.ZipFile", return_value=mock_zip):
                        with patch("os.path.exists", side_effect=exists_side):
                            with patch("os.path.getsize", return_value=12345):
                                resp = await download_all_video_clips("video-1")
        assert resp is not None
        mock_zip.write.assert_not_called()


# --------------------------------------------------------------------------- #
# clear_dataset
# --------------------------------------------------------------------------- #
class TestClearDataset:
    @pytest.mark.asyncio
    async def test_invalid_path_raises_400(self):
        with patch("inference.safe_dataset_path", side_effect=ValueError("bad path")):
            with pytest.raises(HTTPException) as exc:
                await clear_dataset("../evil")
            assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_success_deletes_dir(self):
        with patch("inference.safe_dataset_path", return_value="/app/videos/ds-1"):
            with patch("os.path.exists", return_value=True):
                with patch("shutil.rmtree"):
                    with patch("inference.postgres_db.delete_data", new=AsyncMock(return_value=3)):
                        resp = await clear_dataset("ds-1")
        assert resp.files_deleted == 1
        assert resp.deleted_count == 3

    @pytest.mark.asyncio
    async def test_rmtree_failure_warned(self):
        # Covers the warning branch when rmtree fails (919-920 region equivalent).
        with patch("inference.safe_dataset_path", return_value="/app/videos/ds-1"):
            with patch("os.path.exists", return_value=True):
                with patch("shutil.rmtree", side_effect=OSError("locked")):
                    with patch("inference.postgres_db.delete_data", new=AsyncMock(return_value=1)):
                        resp = await clear_dataset("ds-1")
        # Directory deletion failed but DB cleanup proceeds.
        assert resp.files_deleted == 0


# --------------------------------------------------------------------------- #
# split_video (request handling / validation)
# --------------------------------------------------------------------------- #
class TestSplitVideo:
    @pytest.mark.asyncio
    async def test_invalid_request_body(self):
        with pytest.raises(HTTPException) as exc:
            await split_video("video-1", None)
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_no_timestamps(self):
        with pytest.raises(HTTPException) as exc:
            await split_video("video-1", {"timestamps": []})
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_video_not_found(self):
        inference.current_data_id = None
        body = {"timestamps": [{"start": 0, "end": 5}], "twoOperatorMode": False}
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=None)):
            with pytest.raises(HTTPException) as exc:
                await split_video("video-1", body)
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_no_valid_timestamps_after_filtering(self):
        # Timestamps present but all invalid (end<=start / wrong type) -> 400.
        inference.current_data_id = None
        body = {
            "timestamps": [
                {"start": 5, "end": 5},  # end<=start
                "not-a-dict",            # wrong type
                {"start": "x", "end": "y"},  # non-numeric
            ],
            "twoOperatorMode": False,
        }
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=_make_video())):
            with pytest.raises(HTTPException) as exc:
                await split_video("video-1", body)
            assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_two_operator_mode_from_dataset(self):
        # twoOperatorMode omitted -> falls back to dataset setting (covers 995-1000).
        inference.current_data_id = "ds-1"
        ds = MagicMock()
        ds.two_operator_mode = True
        video = _make_video()
        body = {"timestamps": [{"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "a"}]}

        async def get_side(_id, model):
            name = getattr(model, "__name__", str(model))
            if name == "Dataset":
                return ds
            return video

        with patch("inference.postgres_db.get_data", new=AsyncMock(side_effect=get_side)):
            with patch("inference.postgres_db.list_data", new=AsyncMock(return_value=[])):
                with patch("inference._split_video_by_timestamps", new=AsyncMock(return_value=[{"id": "c1"}])):
                    with patch("inference._load_merge_threshold", return_value=0.0):
                        resp = await split_video("video-1", body)
        assert resp["twoOperatorMode"] is True
        assert resp["status"] == "success"
        inference.current_data_id = None

    @pytest.mark.asyncio
    async def test_existing_chunks_cleaned_up(self):
        # Covers chunk-cleanup loop (1075-1098) including os.remove and DB delete.
        inference.current_data_id = None
        video = _make_video()
        old_chunk = MagicMock()
        old_chunk.id = "old-1"
        old_chunk.name = "01_old.mp4"
        body = {
            "timestamps": [{"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "a"}],
            "twoOperatorMode": False,
        }
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=video)):
            with patch("inference.postgres_db.list_data", new=AsyncMock(return_value=[old_chunk])):
                with patch("inference.postgres_db.delete_data", new=AsyncMock()) as dele:
                    with patch("os.path.exists", return_value=True):
                        with patch("os.remove"):
                            with patch("inference._split_video_by_timestamps", new=AsyncMock(return_value=[{"id": "c1"}])):
                                resp = await split_video("video-1", body)
        assert resp["status"] == "success"
        dele.assert_awaited()

    @pytest.mark.asyncio
    async def test_chunk_remove_failure_warned(self):
        # Covers the os.remove failure warning branch (1094-1095).
        inference.current_data_id = None
        video = _make_video()
        old_chunk = MagicMock()
        old_chunk.id = "old-1"
        old_chunk.name = "01_old.mp4"
        body = {
            "timestamps": [{"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "a"}],
            "twoOperatorMode": False,
        }
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=video)):
            with patch("inference.postgres_db.list_data", new=AsyncMock(return_value=[old_chunk])):
                with patch("inference.postgres_db.delete_data", new=AsyncMock()):
                    with patch("os.path.exists", return_value=True):
                        with patch("os.remove", side_effect=OSError("locked")):
                            with patch("inference._split_video_by_timestamps", new=AsyncMock(return_value=[{"id": "c1"}])):
                                resp = await split_video("video-1", body)
        assert resp["status"] == "success"

    @pytest.mark.asyncio
    async def test_two_operator_merge_path(self):
        # Covers post-split merge logic (1107-1151) including DB cleanup of merged chunks.
        inference.current_data_id = None
        video = _make_video()
        body = {
            "timestamps": [{"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "a"}],
            "twoOperatorMode": True,
        }
        merge_stats = {
            "original_count": 2,
            "final_count": 1,
            "merged_count": 1,
            "_removed_chunk_ids": ["removed-1"],
            "_updated_chunks": {"keep-1": {"start_time": 0, "end_time": 10}},
        }
        ann = MagicMock()
        ann.id = "ann-1"
        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=video)):
            with patch("inference.postgres_db.list_data", new=AsyncMock(return_value=[ann])):
                with patch("inference.postgres_db.delete_data", new=AsyncMock()):
                    with patch("inference.postgres_db.update_data", new=AsyncMock()):
                        with patch("inference._split_video_by_timestamps", new=AsyncMock(return_value=[{"id": "keep-1"}])):
                            with patch("inference._load_merge_threshold", return_value=0.2):
                                with patch("inference._merge_small_chunks", return_value=([{"id": "keep-1"}], dict(merge_stats))):
                                    resp = await split_video("video-1", body)
        assert resp["mergeStats"]["merged_count"] == 1
        assert resp["twoOperatorMode"] is True


# --------------------------------------------------------------------------- #
# _validate_video_file
# --------------------------------------------------------------------------- #
class TestValidateVideoFile:
    def test_missing_file(self):
        with patch("os.path.exists", return_value=False):
            with pytest.raises(HTTPException) as exc:
                _validate_video_file("/x/v.mp4")
            assert exc.value.status_code == 404

    def test_invalid_dimensions(self):
        clip = MagicMock(w=0, h=0)
        with patch("os.path.exists", return_value=True):
            with patch("inference.VideoFileClip", return_value=clip):
                with pytest.raises(HTTPException) as exc:
                    _validate_video_file("/x/v.mp4")
                assert exc.value.status_code == 400

    def test_invalid_duration(self):
        clip = MagicMock(w=1920, h=1080, duration=0)
        with patch("os.path.exists", return_value=True):
            with patch("inference.VideoFileClip", return_value=clip):
                with pytest.raises(HTTPException) as exc:
                    _validate_video_file("/x/v.mp4")
                assert exc.value.status_code == 400

    def test_success_returns_duration(self):
        clip = MagicMock(w=1920, h=1080, duration=42.0)
        with patch("os.path.exists", return_value=True):
            with patch("inference.VideoFileClip", return_value=clip):
                assert _validate_video_file("/x/v.mp4") == 42.0


# --------------------------------------------------------------------------- #
# _process_single_segment
# --------------------------------------------------------------------------- #
class TestProcessSingleSegment:
    def _patch_clip(self, write_ok=True, test_duration=5.0):
        main = MagicMock()
        sub = MagicMock()
        if not write_ok:
            sub.write_videofile = MagicMock(side_effect=Exception("ffmpeg failed"))
        main.subclip = MagicMock(return_value=sub)
        test_clip = MagicMock(duration=test_duration)
        calls = [0]

        def side(_path):
            calls[0] += 1
            return main if calls[0] == 1 else test_clip

        return side

    @pytest.mark.asyncio
    async def test_start_exceeds_duration_skipped(self):
        ts = {"start": 100, "end": 105, "actionIndex": 0, "actionDescription": "a"}
        result = await _process_single_segment(
            ts, "/x/v.mp4", "v", "/x/v", _make_video(), 10, {}, {}, 1
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_success(self):
        ts = {"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "a"}
        with patch("inference.VideoFileClip", side_effect=self._patch_clip()):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.getsize", return_value=50000):
                    with patch("inference.postgres_db.insert_data", new=AsyncMock()):
                        result = await _process_single_segment(
                            ts, "/x/v.mp4", "v", "/x/v", _make_video(), 60,
                            {0: "a"}, {}, 1,
                        )
        clip_result, annotation_entry = result
        assert clip_result["action_index"] == 0
        assert annotation_entry["action"] == 1

    @pytest.mark.asyncio
    async def test_end_adjusted_to_duration(self):
        ts = {"start": 0, "end": 999, "actionIndex": 0, "actionDescription": "a"}
        with patch("inference.VideoFileClip", side_effect=self._patch_clip()):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.getsize", return_value=50000):
                    with patch("inference.postgres_db.insert_data", new=AsyncMock()):
                        result = await _process_single_segment(
                            ts, "/x/v.mp4", "v", "/x/v", _make_video(), 10,
                            {0: "a"}, {}, 1,
                        )
        clip_result, _ = result
        assert clip_result["end_time"] == 10

    @pytest.mark.asyncio
    async def test_write_failure_cleans_up_and_raises(self):
        ts = {"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "a"}
        main = MagicMock()
        sub = MagicMock()
        sub.write_videofile = MagicMock(side_effect=Exception("ffmpeg failed"))
        main.subclip = MagicMock(return_value=sub)
        with patch("inference.VideoFileClip", return_value=main):
            with patch("os.path.exists", return_value=True):
                with patch("os.remove"):
                    with patch("inference.postgres_db.delete_data", new=AsyncMock()):
                        with pytest.raises(HTTPException) as exc:
                            await _process_single_segment(
                                ts, "/x/v.mp4", "v", "/x/v", _make_video(), 60,
                                {0: "a"}, {}, 1,
                            )
                        assert exc.value.status_code == 500


# --------------------------------------------------------------------------- #
# _process_concurrent_segment
# --------------------------------------------------------------------------- #
class TestProcessConcurrentSegment:
    def _segment(self, concurrent=False):
        actions = [{"actionIndex": 0, "actionDescription": "a"}]
        if concurrent:
            actions.append({"actionIndex": 1, "actionDescription": "b"})
        return {
            "start": 0,
            "end": 5,
            "concurrent_actions": actions,
            "is_concurrent": concurrent,
        }

    @pytest.mark.asyncio
    async def test_start_exceeds_duration_skipped(self):
        seg = {"start": 100, "end": 105, "concurrent_actions": [{"actionIndex": 0, "actionDescription": "a"}], "is_concurrent": False}
        result = await _process_concurrent_segment(
            seg, "/x/v.mp4", "v", "/x/v", _make_video(), 10, {}, 1
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_too_short_skipped(self):
        seg = {"start": 0, "end": 0.05, "concurrent_actions": [{"actionIndex": 0, "actionDescription": "a"}], "is_concurrent": False}
        result = await _process_concurrent_segment(
            seg, "/x/v.mp4", "v", "/x/v", _make_video(), 60, {}, 1
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_concurrent_success(self):
        seg = self._segment(concurrent=True)
        main = MagicMock()
        sub = MagicMock()
        main.subclip = MagicMock(return_value=sub)
        with patch("inference.VideoFileClip", return_value=main):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.getsize", return_value=50000):
                    with patch("inference.postgres_db.insert_data", new=AsyncMock()):
                        result = await _process_concurrent_segment(
                            seg, "/x/v.mp4", "v", "/x/v", _make_video(), 60, {}, 1
                        )
        clip_result, annotation_entry = result
        assert clip_result["is_concurrent"] is True
        assert "AND" in clip_result["action_description"]
        assert annotation_entry["actions"] == [1, 2]

    @pytest.mark.asyncio
    async def test_output_too_small_returns_none(self):
        seg = self._segment(concurrent=False)
        main = MagicMock()
        sub = MagicMock()
        main.subclip = MagicMock(return_value=sub)
        with patch("inference.VideoFileClip", return_value=main):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.getsize", return_value=100):  # < 10240
                    with patch("os.remove"):
                        result = await _process_concurrent_segment(
                            seg, "/x/v.mp4", "v", "/x/v", _make_video(), 60, {}, 1
                        )
        assert result is None

    @pytest.mark.asyncio
    async def test_write_failure_cleans_up_and_raises(self):
        seg = self._segment(concurrent=False)
        main = MagicMock()
        sub = MagicMock()
        sub.write_videofile = MagicMock(side_effect=Exception("ffmpeg failed"))
        main.subclip = MagicMock(return_value=sub)
        with patch("inference.VideoFileClip", return_value=main):
            with patch("os.path.exists", return_value=True):
                with patch("os.remove"):
                    with pytest.raises(Exception):
                        await _process_concurrent_segment(
                            seg, "/x/v.mp4", "v", "/x/v", _make_video(), 60, {}, 1
                        )

    @pytest.mark.asyncio
    async def test_end_adjusted_and_output_missing(self):
        # Covers end-time clamp to duration (1364) and "output not created" -> None (1416).
        seg = {"start": 0, "end": 999, "concurrent_actions": [{"actionIndex": 0, "actionDescription": "a"}], "is_concurrent": False}
        main = MagicMock()
        sub = MagicMock()
        main.subclip = MagicMock(return_value=sub)
        with patch("inference.VideoFileClip", return_value=main):
            with patch("os.path.exists", return_value=False):  # output never created
                result = await _process_concurrent_segment(
                    seg, "/x/v.mp4", "v", "/x/v", _make_video(), 10, {}, 1
                )
        assert result is None


# --------------------------------------------------------------------------- #
# lifespan migration error branch
# --------------------------------------------------------------------------- #
class TestLifespanMigration:
    @pytest.mark.asyncio
    async def test_migration_failure_is_warned(self):
        # Covers 105-106: auto-migration exception is swallowed/warned.
        app = MagicMock()
        with patch("inference.create_dir"):
            with patch("inference.postgres_db") as db:
                db.engine.begin = MagicMock(side_effect=RuntimeError("no engine"))
                async with lifespan(app):
                    pass  # should not raise


# --------------------------------------------------------------------------- #
# reset_actions
# --------------------------------------------------------------------------- #
class TestResetActions:
    @pytest.mark.asyncio
    async def test_no_data_id_set(self):
        inference.current_data_id = None
        with pytest.raises(HTTPException) as exc:
            await reset_actions()
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_success(self):
        inference.current_data_id = "ds-1"
        resp = await reset_actions()
        assert resp.status == "success"
        assert resp.previous_data_id == "ds-1"
        assert inference.current_data_id is None


# --------------------------------------------------------------------------- #
# upload_video large-file warning branch (line 263)
# --------------------------------------------------------------------------- #
class TestUploadVideoLargeFile:
    @pytest.mark.asyncio
    async def test_large_file_warns(self):
        inference.current_data_id = "ds-1"
        upload = MagicMock()
        upload.filename = "clip.mp4"
        # > 1GB triggers the large-file warning branch.
        big = MagicMock()
        big.__len__ = MagicMock(return_value=2 * 1024 * 1024 * 1024)
        upload.file.read = MagicMock(return_value=big)
        with patch("inference.postgres_db.list_data", new=AsyncMock(return_value=[])):
            with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=None)):
                with patch("inference.postgres_db.insert_data", new=AsyncMock()):
                    with patch("builtins.open", mock_open()):
                        with patch("inference.clean_up_file"):
                            with patch("inference.convert_to_h264", new=AsyncMock(return_value="/x/clip.mp4")):
                                with patch("os.path.getsize", return_value=99999):
                                    resp = await upload_video(upload)
        assert resp.file_id is not None
        inference.current_data_id = None


# --------------------------------------------------------------------------- #
# clear_all_videos
# --------------------------------------------------------------------------- #
class TestClearAllVideos:
    @pytest.mark.asyncio
    async def test_clears_existing_dirs(self):
        ds = MagicMock()
        ds.id = "ds-1"
        with patch("inference.postgres_db.list_data", new=AsyncMock(return_value=[ds])):
            with patch("inference.postgres_db.delete_all_data", new=AsyncMock(return_value=2)):
                with patch("os.path.exists", return_value=True):
                    with patch("shutil.rmtree"):
                        resp = await clear_all_videos()
        assert resp.files_deleted == 1
        assert resp.deleted_count == 2

    @pytest.mark.asyncio
    async def test_dataset_dir_missing_warns(self):
        # Covers line 881: dataset directory not found warning.
        ds = MagicMock()
        ds.id = "ds-1"
        # root exists, dataset dir does not.
        exists_vals = iter([True, False])
        with patch("inference.postgres_db.list_data", new=AsyncMock(return_value=[ds])):
            with patch("inference.postgres_db.delete_all_data", new=AsyncMock(return_value=0)):
                with patch("os.path.exists", side_effect=lambda p: next(exists_vals)):
                    resp = await clear_all_videos()
        assert resp.files_deleted == 0

    @pytest.mark.asyncio
    async def test_db_error(self):
        with patch("inference.postgres_db.list_data", new=AsyncMock(side_effect=RuntimeError("x"))):
            with pytest.raises(HTTPException) as exc:
                await clear_all_videos()
            assert exc.value.status_code == 500


# --------------------------------------------------------------------------- #
# split_video dataset-fetch fallback (999-1000) and merge DB-cleanup warnings
# --------------------------------------------------------------------------- #
class TestSplitVideoExtraBranches:
    @pytest.mark.asyncio
    async def test_dataset_fetch_exception_defaults_false(self):
        # Covers 999-1000: dataset lookup raises -> two_operator_mode falls back to False.
        inference.current_data_id = "ds-1"
        video = _make_video()
        body = {"timestamps": [{"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "a"}]}

        async def get_side(_id, model):
            name = getattr(model, "__name__", str(model))
            if name == "Dataset":
                raise RuntimeError("db down")
            return video

        with patch("inference.postgres_db.get_data", new=AsyncMock(side_effect=get_side)):
            with patch("inference.postgres_db.list_data", new=AsyncMock(return_value=[])):
                with patch("inference._split_video_by_timestamps", new=AsyncMock(return_value=[{"id": "c1"}])):
                    resp = await split_video("video-1", body)
        assert resp["twoOperatorMode"] is False
        inference.current_data_id = None

    @pytest.mark.asyncio
    async def test_merge_db_cleanup_warnings(self):
        # Covers 1135-1136 and 1150-1151: warnings when removing/updating merged chunk DB records fail.
        inference.current_data_id = None
        video = _make_video()
        body = {
            "timestamps": [{"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "a"}],
            "twoOperatorMode": True,
        }
        merge_stats = {
            "original_count": 2,
            "final_count": 1,
            "merged_count": 1,
            "_removed_chunk_ids": ["removed-1"],
            "_updated_chunks": {"keep-1": {"start_time": 0, "end_time": 10}},
        }
        async def list_side(model, condition=None):
            # Initial existing-chunk lookup (Chunk) succeeds; Annotation lookups during
            # merge cleanup raise to exercise the warning branches.
            name = getattr(model, "__name__", str(model))
            if name == "Annotation":
                raise RuntimeError("list fail")
            return []

        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=video)):
            with patch("inference.postgres_db.list_data", new=AsyncMock(side_effect=list_side)):
                with patch("inference.postgres_db.delete_data", new=AsyncMock()):
                    with patch("inference._split_video_by_timestamps", new=AsyncMock(return_value=[{"id": "keep-1"}])):
                        with patch("inference._load_merge_threshold", return_value=0.2):
                            with patch("inference._merge_small_chunks", return_value=([{"id": "keep-1"}], dict(merge_stats))):
                                resp = await split_video("video-1", body)
        assert resp["twoOperatorMode"] is True


# --------------------------------------------------------------------------- #
# _process_single_segment FileNotFound / invalid duration (1601, 1619-1621)
# --------------------------------------------------------------------------- #
class TestProcessSingleSegmentEdgeCases:
    @pytest.mark.asyncio
    async def test_output_not_created_raises(self):
        # Covers 1601: output file not created -> FileNotFoundError -> 500.
        ts = {"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "a"}
        main = MagicMock()
        sub = MagicMock()
        main.subclip = MagicMock(return_value=sub)
        with patch("inference.VideoFileClip", return_value=main):
            with patch("os.path.exists", return_value=False):  # output missing
                with patch("os.remove"):
                    with patch("inference.postgres_db.delete_data", new=AsyncMock()):
                        with pytest.raises(HTTPException) as exc:
                            await _process_single_segment(
                                ts, "/x/v.mp4", "v", "/x/v", _make_video(), 60,
                                {0: "a"}, {}, 1,
                            )
                        assert exc.value.status_code == 500

    @pytest.mark.asyncio
    async def test_invalid_split_duration_raises(self):
        # Covers 1619-1621: test_clip has invalid duration -> ValueError -> 500.
        ts = {"start": 0, "end": 5, "actionIndex": 0, "actionDescription": "a"}
        main = MagicMock()
        sub = MagicMock()
        main.subclip = MagicMock(return_value=sub)
        test_clip = MagicMock(duration=0)  # invalid
        calls = [0]

        def side(_path):
            calls[0] += 1
            return main if calls[0] == 1 else test_clip

        with patch("inference.VideoFileClip", side_effect=side):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.getsize", return_value=50000):
                    with patch("os.remove"):
                        with patch("inference.postgres_db.delete_data", new=AsyncMock()):
                            with pytest.raises(HTTPException) as exc:
                                await _process_single_segment(
                                    ts, "/x/v.mp4", "v", "/x/v", _make_video(), 60,
                                    {0: "a"}, {}, 1,
                                )
                            assert exc.value.status_code == 500


# --------------------------------------------------------------------------- #
# _split_video_by_timestamps two-operator branch (1734-1775)
# --------------------------------------------------------------------------- #
class TestSplitByTimestampsTwoOperator:
    @pytest.mark.asyncio
    async def test_two_operator_writes_annotation(self):
        video = _make_video()
        timestamps = [
            {"start": 0, "end": 10, "actionIndex": 0, "actionDescription": "a"},
            {"start": 5, "end": 15, "actionIndex": 1, "actionDescription": "b"},
        ]
        clip_result = {"id": "c1", "start_time": 0, "end_time": 5, "filename": "x.mp4"}
        annotation_entry = {"chunk": "chunk #1"}
        # Let the real asyncio.to_thread run the _write_annotation closure (covers 1769-1770)
        # with a mocked open so no file is actually written.
        with patch("inference._load_action_descriptions", return_value={0: "a", 1: "b"}):
            with patch("inference._validate_video_file", return_value=60):
                with patch("inference.create_dir"):
                    with patch("inference._process_concurrent_segment",
                               new=AsyncMock(return_value=(clip_result, annotation_entry))):
                        with patch("builtins.open", mock_open()):
                            result = await _split_video_by_timestamps(video, timestamps, two_operator_mode=True)
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_two_operator_segment_skipped(self):
        # _process_concurrent_segment returns None -> segment skipped, empty result.
        video = _make_video()
        timestamps = [{"start": 0, "end": 10, "actionIndex": 0, "actionDescription": "a"}]
        with patch("inference._load_action_descriptions", return_value={0: "a"}):
            with patch("inference._validate_video_file", return_value=60):
                with patch("inference.create_dir"):
                    with patch("inference._process_concurrent_segment", new=AsyncMock(return_value=None)):
                        result = await _split_video_by_timestamps(video, timestamps, two_operator_mode=True)
        assert result == []

    @pytest.mark.asyncio
    async def test_two_operator_processing_error_continues(self):
        # Exception inside _process_concurrent_segment is caught and the loop continues.
        video = _make_video()
        timestamps = [{"start": 0, "end": 10, "actionIndex": 0, "actionDescription": "a"}]
        with patch("inference._load_action_descriptions", return_value={0: "a"}):
            with patch("inference._validate_video_file", return_value=60):
                with patch("inference.create_dir"):
                    with patch("inference._process_concurrent_segment",
                               new=AsyncMock(side_effect=Exception("boom"))):
                        result = await _split_video_by_timestamps(video, timestamps, two_operator_mode=True)
        assert result == []


# --------------------------------------------------------------------------- #
# Merge helpers (pure logic, no ffmpeg)
# --------------------------------------------------------------------------- #
class TestMergeHelpers:
    def _clip(self, idx, start, end, action):
        return {
            "id": idx,
            "filename": f"{idx}.mp4",
            "start_time": start,
            "end_time": end,
            "action_indices": [action],
            "action_descriptions": ["a"],
            "is_concurrent": False,
            "timeline_order": 1,
        }

    def test_clips_to_merge_format(self):
        clips = [self._clip("c1", 0, 5, 1)]
        out = _clips_to_merge_format(clips)
        assert out[0]["start_timestamp"] == 0
        assert out[0]["chunk_name"] == "c1.mp4"
        assert out[0]["_clip_data"]["id"] == "c1"

    def test_is_small_no_candidates(self):
        chunks = [
            {"start_timestamp": 0, "end_timestamp": 5, "actions": [1]},
            {"start_timestamp": 5, "end_timestamp": 10, "actions": [2]},
        ]
        is_small, idx, direction = _is_small_relative_to_neighbors(0, chunks)
        assert is_small is False

    def test_is_small_with_prev_neighbor(self):
        chunks = [
            {"start_timestamp": 0, "end_timestamp": 10, "actions": [1]},
            {"start_timestamp": 10, "end_timestamp": 10.5, "actions": [1]},  # tiny, shares action
        ]
        is_small, idx, direction = _is_small_relative_to_neighbors(1, chunks)
        assert is_small is True
        assert idx == 0
        assert direction == "prev"

    def test_try_merge_one_not_small(self):
        chunks = [
            {"start_timestamp": 0, "end_timestamp": 5, "actions": [1], "chunk_name": "a.mp4"},
            {"start_timestamp": 5, "end_timestamp": 10, "actions": [2], "chunk_name": "b.mp4"},
        ]
        assert _try_merge_one(0, chunks, "/x", 0.2, [], {}) is False

    def test_try_merge_one_missing_files(self):
        chunks = [
            {"start_timestamp": 0, "end_timestamp": 10, "actions": [1], "chunk_name": "a.mp4"},
            {"start_timestamp": 10, "end_timestamp": 10.5, "actions": [1], "chunk_name": "b.mp4"},
        ]
        with patch("os.path.exists", return_value=False):
            assert _try_merge_one(1, chunks, "/x", 0.2, [], {}) is False

    def test_try_merge_one_concatenate_failure(self):
        chunks = [
            {"start_timestamp": 0, "end_timestamp": 10, "actions": [1], "chunk_name": "a.mp4",
             "_clip_data": {"id": "c0"}},
            {"start_timestamp": 10, "end_timestamp": 10.5, "actions": [1], "chunk_name": "b.mp4",
             "_clip_data": {"id": "c1"}},
        ]
        with patch("os.path.exists", return_value=True):
            with patch("inference._concatenate_videos", side_effect=Exception("ffmpeg")):
                with patch("os.remove"):
                    assert _try_merge_one(1, chunks, "/x", 0.2, [], {}) is False

    def test_try_merge_one_success(self):
        chunks = [
            {"start_timestamp": 0, "end_timestamp": 10, "actions": [1], "chunk_name": "a.mp4",
             "_clip_data": {"id": "c0"}},
            {"start_timestamp": 10, "end_timestamp": 10.5, "actions": [1], "chunk_name": "b.mp4",
             "_clip_data": {"id": "c1"}},
        ]
        removed, updated = [], {}
        with patch("os.path.exists", return_value=True):
            with patch("inference._concatenate_videos", return_value=True):
                with patch("os.remove"):
                    with patch("os.rename"):
                        merged = _try_merge_one(1, chunks, "/x", 0.2, removed, updated)
        assert merged is True
        assert "c1" in removed
        assert "c0" in updated
        assert len(chunks) == 1

    def test_merge_small_chunks_end_to_end(self):
        clips = [
            self._clip("c0", 0, 10, 1),
            self._clip("c1", 10, 10.5, 1),  # tiny -> merges into c0
        ]
        with patch("os.path.exists", return_value=True):
            with patch("inference._concatenate_videos", return_value=True):
                with patch("os.remove"):
                    with patch("os.rename"):
                        merged, stats = _merge_small_chunks(clips, "/x", threshold=0.2)
        assert stats["original_count"] == 2
        assert stats["merged_count"] == 1
        assert stats["final_count"] == 1

    def test_merge_small_chunks_no_merge(self):
        clips = [self._clip("c0", 0, 5, 1), self._clip("c1", 5, 10, 2)]
        merged, stats = _merge_small_chunks(clips, "/x", threshold=0.2)
        assert stats["merged_count"] == 0
        assert stats["final_count"] == 2

    def test_is_small_with_next_neighbor(self):
        # Covers 1861-1863: the "next" neighbor branch.
        chunks = [
            {"start_timestamp": 0, "end_timestamp": 0.5, "actions": [1]},  # tiny, shares action with next
            {"start_timestamp": 0.5, "end_timestamp": 10, "actions": [1]},
        ]
        is_small, idx, direction = _is_small_relative_to_neighbors(0, chunks)
        assert is_small is True
        assert idx == 1
        assert direction == "next"

    def test_try_merge_one_temp_not_created(self):
        # Covers 1955-1956: _concatenate_videos returns False -> warning, no merge.
        chunks = [
            {"start_timestamp": 0, "end_timestamp": 10, "actions": [1], "chunk_name": "a.mp4",
             "_clip_data": {"id": "c0"}},
            {"start_timestamp": 10, "end_timestamp": 10.5, "actions": [1], "chunk_name": "b.mp4",
             "_clip_data": {"id": "c1"}},
        ]
        with patch("os.path.exists", return_value=True):
            with patch("inference._concatenate_videos", return_value=False):
                assert _try_merge_one(1, chunks, "/x", 0.2, [], {}) is False

    def test_concatenate_videos_prev_and_next(self):
        # Covers _concatenate_videos with mocked moviepy (both direction branches).
        final = MagicMock()
        with patch("inference.VideoFileClip", return_value=MagicMock()):
            with patch("inference.concatenate_videoclips", return_value=final):
                with patch("os.path.exists", return_value=True):
                    from inference import _concatenate_videos
                    assert _concatenate_videos("/s.mp4", "/t.mp4", "/tmp/o.mp4", "prev") is True
                    assert _concatenate_videos("/s.mp4", "/t.mp4", "/tmp/o.mp4", "next") is True


# --------------------------------------------------------------------------- #
# download_all_video_clips: zip-create-failure (780) and HTTPException re-raise (811)
# --------------------------------------------------------------------------- #
class TestDownloadAllVideoClipsEdgeCases:
    @pytest.mark.asyncio
    async def test_zip_not_created_raises_500(self):
        # Covers line 780: ZIP file not present after creation -> 500.
        chunk = MagicMock()
        chunk.name = "01_clip.mp4"
        video = _make_video()
        mock_zip = MagicMock()
        mock_zip.__enter__ = MagicMock(return_value=mock_zip)
        mock_zip.__exit__ = MagicMock(return_value=False)
        mock_temp = MagicMock()
        mock_temp.name = "/tmp/out.zip"
        mock_temp.__enter__ = MagicMock(return_value=mock_temp)
        mock_temp.__exit__ = MagicMock(return_value=False)

        # exists() True while adding chunk, then False at the zip-created check, then
        # False in the cleanup branch.
        exists_vals = iter([True, False, False])

        with patch("inference.postgres_db.get_data", new=AsyncMock(return_value=video)):
            with patch("inference.postgres_db.list_data", new=AsyncMock(return_value=[chunk])):
                with patch("tempfile.NamedTemporaryFile", return_value=mock_temp):
                    with patch("zipfile.ZipFile", return_value=mock_zip):
                        with patch("os.path.exists", side_effect=lambda p: next(exists_vals, False)):
                            with pytest.raises(HTTPException) as exc:
                                await download_all_video_clips("video-1")
        assert exc.value.status_code == 500
