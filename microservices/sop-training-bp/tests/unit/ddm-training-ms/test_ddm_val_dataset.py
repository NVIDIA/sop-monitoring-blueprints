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
Unit tests for DDMValStreamingDataset and DecordStreamingReader.
Covers init parameter handling, chunking, load balancing, annotation parsing, and collate logic.
"""

import sys
import json
import math
from unittest.mock import MagicMock, patch, mock_open

# Mock heavy ML dependencies before any project imports.
_ML_MODULES = [
    "torch",
    "torch.nn",
    "torch.utils",
    "torch.utils.data",
    "torch.utils.data.dataloader",  # needed: ddm_val_dataset.py imports default_collate from here
    "torch.distributed",
    "torchvision",
    "torchvision.transforms",
    "av",
    "decord",
]
for _mod in _ML_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# decord.bridge.set_bridge is called at import time; ensure it is a proper mock.
_decord_mock = sys.modules["decord"]
_decord_mock.bridge = MagicMock()
_decord_mock.VideoReader = MagicMock()
_decord_mock.cpu = MagicMock()

# IterableDataset must be a real Python class so that DDMValStreamingDataset subclasses it properly.
sys.modules["torch.utils.data"].IterableDataset = type("IterableDataset", (), {})

# torch.LongTensor must support real index assignment and comparison used in _process_data.
# MagicMock.__setitem__ is a no-op, so label assertions would be unreliable without this override.
import numpy as np
import pytest


class _FakeLongTensor:
    """Minimal torch.LongTensor replacement supporting .numpy() and index assignment."""

    def __init__(self, data):
        self._data = np.array(list(data), dtype=np.int64)

    def numpy(self):
        return self._data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __len__(self):
        return len(self._data)


sys.modules["torch"].LongTensor = _FakeLongTensor

from datasets.ddm_val_dataset import DDMValStreamingDataset


class TestDDMValStreamingDatasetInit:
    """Tests for DDMValStreamingDataset.__init__() parameter handling."""

    def test_integer_resolution_converted_to_tuple(self):
        """Test that an integer resolution is converted to a (int, int) tuple."""
        with patch.object(DDMValStreamingDataset, "_process_data"):
            dataset = DDMValStreamingDataset(
                annotation_file="/fake/anno.json",
                video_root="/fake/videos",
                resolution=224,
            )
            assert dataset.resolution == (224, 224)

    def test_tuple_resolution_kept_as_is(self):
        """Test that a tuple resolution is stored unchanged."""
        with patch.object(DDMValStreamingDataset, "_process_data"):
            dataset = DDMValStreamingDataset(
                annotation_file="/fake/anno.json",
                video_root="/fake/videos",
                resolution=(128, 256),
            )
            assert dataset.resolution == (128, 256)

    def test_default_attributes_set_correctly(self):
        """Test that default parameter values are assigned to instance attributes."""
        with patch.object(DDMValStreamingDataset, "_process_data"):
            dataset = DDMValStreamingDataset(
                annotation_file="/fake/anno.json",
                video_root="/fake/videos",
            )
            assert dataset.frames_per_side == 5
            assert dataset.downsample == 1
            assert dataset.temporal_stride == 1
            assert dataset.min_change_dur == pytest.approx(0.3)
            assert dataset.chunk_duration is None
            assert dataset.enable_load_balancing is True
            assert dataset.verbose is False

    def test_custom_attributes_set_correctly(self):
        """Test that custom parameter values are assigned to instance attributes."""
        with patch.object(DDMValStreamingDataset, "_process_data"):
            dataset = DDMValStreamingDataset(
                annotation_file="/fake/anno.json",
                video_root="/fake/videos",
                frames_per_side=10,
                downsample=2,
                temporal_stride=3,
                min_change_dur=0.5,
                chunk_duration=30.0,
                enable_load_balancing=False,
                verbose=True,
            )
            assert dataset.frames_per_side == 10
            assert dataset.downsample == 2
            assert dataset.temporal_stride == 3
            assert dataset.min_change_dur == pytest.approx(0.5)
            assert dataset.chunk_duration == pytest.approx(30.0)
            assert dataset.enable_load_balancing is False
            assert dataset.verbose is True


class TestChunkVideos:
    """Tests for DDMValStreamingDataset._chunk_videos()."""

    def _make_dataset(self):
        with patch.object(DDMValStreamingDataset, "_process_data"):
            return DDMValStreamingDataset(
                annotation_file="/fake/anno.json",
                video_root="/fake/videos",
            )

    def test_single_chunk_when_duration_fits(self):
        """Test that a video shorter than chunk_duration produces a single chunk."""
        dataset = self._make_dataset()
        config = {"path": "/v.mp4", "video_id": "v1", "start": 0.0, "end": 10.0}

        result = dataset._chunk_videos(config, chunk_duration=20.0)

        assert len(result) == 1
        assert result[0]["start"] == pytest.approx(0.0)
        assert result[0]["end"] == pytest.approx(10.0)

    def test_exact_split_into_chunks(self):
        """Test that a 60-second video splits into exactly 3 chunks of 20 seconds."""
        dataset = self._make_dataset()
        config = {"path": "/v.mp4", "video_id": "v1", "start": 0.0, "end": 60.0}

        result = dataset._chunk_videos(config, chunk_duration=20.0)

        assert len(result) == 3
        assert result[0]["start"] == pytest.approx(0.0)
        assert result[0]["end"] == pytest.approx(20.0)
        assert result[1]["start"] == pytest.approx(20.0)
        assert result[1]["end"] == pytest.approx(40.0)
        assert result[2]["start"] == pytest.approx(40.0)
        assert result[2]["end"] == pytest.approx(60.0)

    def test_last_chunk_shorter_when_not_divisible(self):
        """Test that the last chunk is shorter when duration does not divide evenly."""
        dataset = self._make_dataset()
        config = {"path": "/v.mp4", "video_id": "v1", "start": 0.0, "end": 50.0}

        result = dataset._chunk_videos(config, chunk_duration=20.0)

        assert len(result) == 3
        assert result[2]["start"] == pytest.approx(40.0)
        assert result[2]["end"] == pytest.approx(50.0)

    def test_zero_duration_returns_original(self):
        """Test that a zero-duration video is returned as-is without splitting."""
        dataset = self._make_dataset()
        config = {"path": "/v.mp4", "video_id": "v1", "start": 5.0, "end": 5.0}

        result = dataset._chunk_videos(config, chunk_duration=10.0)

        assert len(result) == 1
        assert result[0]["start"] == pytest.approx(5.0)
        assert result[0]["end"] == pytest.approx(5.0)

    def test_non_zero_start_offset_preserved(self):
        """Test that chunk timestamps are correctly offset when start time is non-zero."""
        dataset = self._make_dataset()
        config = {"path": "/v.mp4", "video_id": "v1", "start": 10.0, "end": 40.0}

        result = dataset._chunk_videos(config, chunk_duration=15.0)

        assert len(result) == 2
        assert result[0]["start"] == pytest.approx(10.0)
        assert result[0]["end"] == pytest.approx(25.0)
        assert result[1]["start"] == pytest.approx(25.0)
        assert result[1]["end"] == pytest.approx(40.0)

    def test_video_id_preserved_in_all_chunks(self):
        """Test that each chunk retains the original video_id and path."""
        dataset = self._make_dataset()
        config = {"path": "/v.mp4", "video_id": "v1", "start": 0.0, "end": 30.0}

        result = dataset._chunk_videos(config, chunk_duration=10.0)

        for chunk in result:
            assert chunk["video_id"] == "v1"
            assert chunk["path"] == "/v.mp4"



class TestBalanceLoadByDuration:
    """Tests for DDMValStreamingDataset._balance_load_by_duration()."""

    def _make_dataset(self):
        with patch.object(DDMValStreamingDataset, "_process_data"):
            return DDMValStreamingDataset(
                annotation_file="/fake/anno.json",
                video_root="/fake/videos",
            )

    def test_single_partition_gets_everything(self):
        """Test that all clips are placed in a single partition when num_partitions=1."""
        dataset = self._make_dataset()
        configs = [
            {"path": "/v1.mp4", "video_id": "v1", "start": 0.0, "end": 10.0},
            {"path": "/v2.mp4", "video_id": "v2", "start": 0.0, "end": 20.0},
        ]

        result = dataset._balance_load_by_duration(configs, num_partitions=1)

        assert len(result) == 1
        total_dur = sum(c["end"] - c["start"] for c in result[0])
        assert total_dur == pytest.approx(30.0)

    def test_equal_duration_clips_distributed_evenly(self):
        """Test that 4 equal-duration clips are split evenly across 2 partitions."""
        dataset = self._make_dataset()
        configs = [
            {"path": f"/v{i}.mp4", "video_id": f"v{i}", "start": 0.0, "end": 10.0}
            for i in range(4)
        ]

        result = dataset._balance_load_by_duration(configs, num_partitions=2)

        assert len(result) == 2
        dur_0 = sum(c["end"] - c["start"] for c in result[0])
        dur_1 = sum(c["end"] - c["start"] for c in result[1])
        assert dur_0 == pytest.approx(20.0)
        assert dur_1 == pytest.approx(20.0)

    def test_empty_configs_returns_empty_partitions(self):
        """Test that an empty clip list returns the correct number of empty partitions."""
        dataset = self._make_dataset()

        result = dataset._balance_load_by_duration([], num_partitions=3)

        assert len(result) == 3
        for partition in result:
            assert partition == []

    def test_more_partitions_than_clips(self):
        """Test that some partitions are empty when num_partitions exceeds clip count."""
        dataset = self._make_dataset()
        configs = [
            {"path": "/v1.mp4", "video_id": "v1", "start": 0.0, "end": 10.0},
        ]

        result = dataset._balance_load_by_duration(configs, num_partitions=4)

        assert len(result) == 4
        non_empty = [p for p in result if len(p) > 0]
        assert len(non_empty) >= 1

    def test_all_clips_accounted_for(self):
        """Test that total duration is preserved across all partitions after distribution."""
        dataset = self._make_dataset()
        configs = [
            {"path": "/v1.mp4", "video_id": "v1", "start": 0.0, "end": 30.0},
            {"path": "/v2.mp4", "video_id": "v2", "start": 0.0, "end": 20.0},
            {"path": "/v3.mp4", "video_id": "v3", "start": 0.0, "end": 10.0},
        ]
        total_input = sum(c["end"] - c["start"] for c in configs)

        result = dataset._balance_load_by_duration(configs, num_partitions=2)

        total_output = sum(
            c["end"] - c["start"]
            for partition in result
            for c in partition
        )
        assert total_output == pytest.approx(total_input)

    def test_long_clip_can_be_split_across_partitions(self):
        """Test that a single long clip can be split across multiple partitions."""
        dataset = self._make_dataset()
        configs = [
            {"path": "/v1.mp4", "video_id": "v1", "start": 0.0, "end": 100.0},
        ]

        result = dataset._balance_load_by_duration(configs, num_partitions=2)

        assert len(result) == 2
        total = sum(c["end"] - c["start"] for p in result for c in p)
        assert total == pytest.approx(100.0)


class TestProcessData:
    """Tests for DDMValStreamingDataset._process_data() annotation parsing and label assignment."""

    def _make_dataset_no_process(self):
        """Build a dataset instance with _process_data patched out."""
        with patch.object(DDMValStreamingDataset, "_process_data"):
            dataset = DDMValStreamingDataset(
                annotation_file="/fake/anno.json",
                video_root="/fake/videos",
                min_change_dur=0.3,
            )
        return dataset

    def _make_mock_av_container(self, duration=10.0, fps=30.0, frames=300):
        """Build a minimal mock av container with the given video properties."""
        mock_stream = MagicMock()
        mock_stream.duration = int(duration / 0.001)
        mock_stream.time_base = 0.001
        mock_stream.average_rate = fps
        mock_stream.frames = frames

        mock_container = MagicMock()
        mock_container.__enter__ = MagicMock(return_value=mock_container)
        mock_container.__exit__ = MagicMock(return_value=False)
        mock_container.streams.video = [mock_stream]
        return mock_container

    def test_final_segment_excluded_from_boundary_calc(self):
        """Test that segments with description 'Final Segment' are excluded from boundary labels.

        With exclusion: only one boundary at frame 105 (Step1->Step2).
        Without exclusion: a second boundary would appear near frame 225 (Step2->FinalSeg).
        """
        dataset = self._make_dataset_no_process()
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 3.0},
                {"description": "Step 2", "start_timestamp": 4.0, "end_timestamp": 7.0},
                {"description": "Final Segment", "start_timestamp": 8.0, "end_timestamp": 10.0},
            ]
        }
        mock_container = self._make_mock_av_container(duration=10.0, fps=30.0, frames=300)

        with patch("builtins.open", mock_open(read_data=json.dumps(annotation))), \
             patch("os.path.exists", return_value=True), \
             patch("av.open", return_value=mock_container):
            dataset._process_data("/fake/anno.json", "/fake/videos")

        assert "v1" in dataset.video_info
        labels = dataset.video_info["v1"]["labels"]
        # boundary window around frame 105 should be labeled 1
        assert labels[105] == 1
        assert labels[101] == 1
        # frame 225 should remain 0 since Final Segment is excluded
        assert labels[225] == 0

    def test_missing_video_file_skipped(self):
        """Test that a video is skipped when its file does not exist on disk."""
        dataset = self._make_dataset_no_process()
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 3.0},
            ]
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(annotation))), \
             patch("os.path.exists", return_value=False):
            dataset._process_data("/fake/anno.json", "/fake/videos")

        assert "v1" not in dataset.video_info

    def test_video_clips_created_without_chunking(self):
        """Test that chunk_duration=None produces exactly one clip config per video."""
        dataset = self._make_dataset_no_process()
        dataset.chunk_duration = None
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 5.0},
            ]
        }
        mock_container = self._make_mock_av_container(duration=10.0, fps=30.0, frames=300)

        with patch("builtins.open", mock_open(read_data=json.dumps(annotation))), \
             patch("os.path.exists", return_value=True), \
             patch("av.open", return_value=mock_container):
            dataset._process_data("/fake/anno.json", "/fake/videos")

        assert len(dataset.video_clip_configs) == 1
        assert dataset.video_clip_configs[0]["video_id"] == "v1"

    def test_video_clips_created_with_chunking(self):
        """Test that a long video is split into multiple clip configs when chunk_duration is set."""
        dataset = self._make_dataset_no_process()
        dataset.chunk_duration = 5.0
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 5.0},
            ]
        }
        mock_container = self._make_mock_av_container(duration=10.0, fps=30.0, frames=300)

        with patch("builtins.open", mock_open(read_data=json.dumps(annotation))), \
             patch("os.path.exists", return_value=True), \
             patch("av.open", return_value=mock_container):
            dataset._process_data("/fake/anno.json", "/fake/videos")

        assert len(dataset.video_clip_configs) == 2

    def test_boundary_label_positions(self):
        """Test that frames within the boundary window are labeled 1 and others are labeled 0.

        boundary = floor((end_t + start_t) / 2 * fps), half_width = min_change_dur * fps / 2.
        """
        dataset = self._make_dataset_no_process()
        dataset.min_change_dur = 1.0
        fps = 10.0
        frames = 100
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 4.0},
                {"description": "Step 2", "start_timestamp": 6.0, "end_timestamp": 9.0},
            ]
        }
        mock_container = self._make_mock_av_container(duration=10.0, fps=fps, frames=frames)

        with patch("builtins.open", mock_open(read_data=json.dumps(annotation))), \
             patch("os.path.exists", return_value=True), \
             patch("av.open", return_value=mock_container):
            dataset._process_data("/fake/anno.json", "/fake/videos")

        # boundary: midpoint of (4.0, 6.0) scaled by fps=10 → frame 50; half_dur=5; window: frames 45–55
        labels = dataset.video_info["v1"]["labels"]
        assert labels[50] == 1   # boundary center
        assert labels[45] == 1   # left edge of labeled window
        assert labels[55] == 1   # right edge of labeled window
        assert labels[44] == 0   # just outside window
        assert labels[56] == 0   # just outside window
        assert labels[0] == 0    # far from boundary

    def test_empty_annotation_no_crash(self):
        """Test that an empty annotation JSON does not raise and leaves video_info empty."""
        dataset = self._make_dataset_no_process()

        with patch("builtins.open", mock_open(read_data=json.dumps({}))):
            dataset._process_data("/fake/anno.json", "/fake/videos")

        assert len(dataset.video_info) == 0
        assert len(dataset.video_clip_configs) == 0

    def test_annotation_file_load_failure_handled(self):
        """Test that a corrupt annotation file is silently skipped without crashing."""
        dataset = self._make_dataset_no_process()

        import json as json_module
        with patch("builtins.open", mock_open(read_data="not valid json")), \
             patch.object(json_module, "load", side_effect=ValueError("invalid JSON")):
            dataset._process_data("/fake/anno.json", "/fake/videos")

        assert len(dataset.video_info) == 0
        assert len(dataset.video_clip_configs) == 0

    def test_av_open_exception_silently_skips_video(self):
        """Test that a video is silently skipped when av.open raises an exception."""
        dataset = self._make_dataset_no_process()
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 5.0},
            ]
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(annotation))), \
             patch("os.path.exists", return_value=True), \
             patch("av.open", side_effect=Exception("corrupt video")):
            dataset._process_data("/fake/anno.json", "/fake/videos")

        assert "v1" not in dataset.video_info
        assert len(dataset.video_clip_configs) == 0

    def test_video_duration_stored_in_video_info(self):
        """Test that video duration and fps are stored in video_info after processing."""
        dataset = self._make_dataset_no_process()
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 5.0},
            ]
        }
        mock_container = self._make_mock_av_container(duration=30.0, fps=25.0, frames=750)

        with patch("builtins.open", mock_open(read_data=json.dumps(annotation))), \
             patch("os.path.exists", return_value=True), \
             patch("av.open", return_value=mock_container):
            dataset._process_data("/fake/anno.json", "/fake/videos")

        assert "v1" in dataset.video_info
        assert dataset.video_info["v1"]["duration"] == pytest.approx(30.0)
        assert dataset.video_info["v1"]["fps"] == pytest.approx(25.0)

    def test_multiple_videos_all_processed(self):
        """Test that all videos in the annotation are added to video_info and clip configs."""
        dataset = self._make_dataset_no_process()
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 5.0},
            ],
            "v2": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 3.0},
            ],
        }
        mock_container = self._make_mock_av_container(duration=10.0, fps=30.0, frames=300)

        with patch("builtins.open", mock_open(read_data=json.dumps(annotation))), \
             patch("os.path.exists", return_value=True), \
             patch("av.open", return_value=mock_container):
            dataset._process_data("/fake/anno.json", "/fake/videos")

        assert "v1" in dataset.video_info
        assert "v2" in dataset.video_info
        assert len(dataset.video_clip_configs) == 2



class TestDecordStreamingReaderInit:
    """Tests for DecordStreamingReader.__init__() parameter handling."""

    def test_integer_resolution_converted_to_tuple(self):
        """Test that an integer resolution is converted to a (int, int) tuple."""
        from datasets.ddm_val_dataset import DecordStreamingReader

        mock_vr = MagicMock()
        mock_vr.get_avg_fps.return_value = 30.0
        mock_vr.__len__ = MagicMock(return_value=300)
        mock_vr.get_batch.return_value = MagicMock()

        with patch("datasets.ddm_val_dataset.VideoReader", return_value=mock_vr), \
             patch.object(DecordStreamingReader, "_fill_buffer"):
            reader = DecordStreamingReader(path="/fake/v.mp4", resolution=224)

        assert reader.resolution == (224, 224)

    def test_tuple_resolution_kept(self):
        """Test that a tuple resolution is stored unchanged."""
        from datasets.ddm_val_dataset import DecordStreamingReader

        mock_vr = MagicMock()
        mock_vr.get_avg_fps.return_value = 30.0
        mock_vr.__len__ = MagicMock(return_value=300)

        with patch("datasets.ddm_val_dataset.VideoReader", return_value=mock_vr), \
             patch.object(DecordStreamingReader, "_fill_buffer"):
            reader = DecordStreamingReader(path="/fake/v.mp4", resolution=(128, 256))

        assert reader.resolution == (128, 256)

    def test_invalid_resolution_uses_default(self):
        """Test that an invalid resolution falls back to DEFAULT_RESOLUTION."""
        from datasets.ddm_val_dataset import DecordStreamingReader, DEFAULT_RESOLUTION

        mock_vr = MagicMock()
        mock_vr.get_avg_fps.return_value = 30.0
        mock_vr.__len__ = MagicMock(return_value=300)

        with patch("datasets.ddm_val_dataset.VideoReader", return_value=mock_vr), \
             patch.object(DecordStreamingReader, "_fill_buffer"):
            reader = DecordStreamingReader(path="/fake/v.mp4", resolution=(1, 2, 3))

        assert reader.resolution == DEFAULT_RESOLUTION

    def test_end_time_none_uses_total_frames(self):
        """Test that end_frame_idx equals total_frames when end_time is None."""
        from datasets.ddm_val_dataset import DecordStreamingReader

        mock_vr = MagicMock()
        mock_vr.get_avg_fps.return_value = 30.0
        mock_vr.__len__ = MagicMock(return_value=300)

        with patch("datasets.ddm_val_dataset.VideoReader", return_value=mock_vr), \
             patch.object(DecordStreamingReader, "_fill_buffer"):
            reader = DecordStreamingReader(path="/fake/v.mp4", end_time=None)

        assert reader.end_frame_idx == 300

    def test_end_time_specified_clamps_to_total(self):
        """Test that end_frame_idx is clamped to total_frames when end_time exceeds video length."""
        from datasets.ddm_val_dataset import DecordStreamingReader

        mock_vr = MagicMock()
        mock_vr.get_avg_fps.return_value = 10.0
        mock_vr.__len__ = MagicMock(return_value=100)

        with patch("datasets.ddm_val_dataset.VideoReader", return_value=mock_vr), \
             patch.object(DecordStreamingReader, "_fill_buffer"):
            reader = DecordStreamingReader(path="/fake/v.mp4", end_time=20.0)

        # end_time=20 at fps=10 gives frame 200, clamped to total_frames=100
        assert reader.end_frame_idx == 100

    def test_start_time_sets_initial_center_idx(self):
        """Test that current_center_idx is set to round(start_time * fps) when start_time > 0."""
        from datasets.ddm_val_dataset import DecordStreamingReader

        mock_vr = MagicMock()
        mock_vr.get_avg_fps.return_value = 10.0
        mock_vr.__len__ = MagicMock(return_value=100)

        with patch("datasets.ddm_val_dataset.VideoReader", return_value=mock_vr), \
             patch.object(DecordStreamingReader, "_fill_buffer"):
            reader = DecordStreamingReader(path="/fake/v.mp4", start_time=2.0)

        assert reader.current_center_idx == 20  # start_time=2.0 at fps=10 → frame 20

    def test_default_transform_set_when_none(self):
        """Test that a default transform pipeline is created when transform=None."""
        from datasets.ddm_val_dataset import DecordStreamingReader

        mock_vr = MagicMock()
        mock_vr.get_avg_fps.return_value = 30.0
        mock_vr.__len__ = MagicMock(return_value=300)

        with patch("datasets.ddm_val_dataset.VideoReader", return_value=mock_vr), \
             patch.object(DecordStreamingReader, "_fill_buffer"):
            reader = DecordStreamingReader(path="/fake/v.mp4", transform=None)

        assert reader.transform is not None

    def test_custom_transform_stored_when_provided(self):
        """Test that an explicit transform is stored directly without creating a default pipeline."""
        from datasets.ddm_val_dataset import DecordStreamingReader

        mock_vr = MagicMock()
        mock_vr.get_avg_fps.return_value = 30.0
        mock_vr.__len__ = MagicMock(return_value=300)

        custom_transform = MagicMock()

        with patch("datasets.ddm_val_dataset.VideoReader", return_value=mock_vr), \
             patch.object(DecordStreamingReader, "_fill_buffer"):
            reader = DecordStreamingReader(path="/fake/v.mp4", transform=custom_transform)

        assert reader.transform is custom_transform



class TestGetClippedIndices:
    """Tests for DecordStreamingReader._get_clipped_indices() boundary-aware index computation."""

    def _make_reader(self, frames_per_side=2, downsample=1, total_frames=100, center_idx=50):
        from datasets.ddm_val_dataset import DecordStreamingReader

        mock_vr = MagicMock()
        mock_vr.get_avg_fps.return_value = 30.0
        mock_vr.__len__ = MagicMock(return_value=total_frames)

        with patch("datasets.ddm_val_dataset.VideoReader", return_value=mock_vr), \
             patch.object(DecordStreamingReader, "_fill_buffer"):
            reader = DecordStreamingReader(
                path="/fake/v.mp4",
                frames_per_side=frames_per_side,
                downsample=downsample,
            )
        reader.total_frames = total_frames
        reader.current_center_idx = center_idx
        return reader

    def test_normal_indices(self):
        """Test that indices are symmetric around the center when no clipping is needed."""
        reader = self._make_reader(frames_per_side=2, downsample=1, center_idx=50)

        indices = reader._get_clipped_indices()

        assert indices == [48, 49, 50, 51, 52]

    def test_indices_with_downsample(self):
        """Test that downsample=2 produces indices spaced 2 frames apart."""
        reader = self._make_reader(frames_per_side=2, downsample=2, center_idx=50)

        indices = reader._get_clipped_indices()

        assert indices == [46, 48, 50, 52, 54]

    def test_clipped_at_start(self):
        """Test that negative indices are clipped to 0 near the video start."""
        reader = self._make_reader(frames_per_side=2, downsample=1, center_idx=1)

        indices = reader._get_clipped_indices()

        assert indices == [0, 0, 1, 2, 3]

    def test_clipped_at_end(self):
        """Test that out-of-bound indices are clipped to total_frames-1 near the video end."""
        reader = self._make_reader(
            frames_per_side=2, downsample=1, total_frames=100, center_idx=98
        )

        indices = reader._get_clipped_indices()

        assert indices == [96, 97, 98, 99, 99]

    def test_downsample_clipped_at_boundaries(self):
        """Test that large downsample values near the boundary clip multiple indices to 0."""
        reader = self._make_reader(
            frames_per_side=2, downsample=5, total_frames=100, center_idx=3
        )

        indices = reader._get_clipped_indices()

        # center=3, downsample=5 yields raw indices -7, -2, 3, 8, 13
        # negative values are clipped to 0
        assert indices == [0, 0, 3, 8, 13]



class TestDDMValStreamingDatasetCollate:
    """Tests for DDMValStreamingDataset.collate_fn()."""

    def test_collate_fn_calls_default_collate(self):
        """Test that collate_fn passes the batch to default_collate and returns its result."""
        with patch.object(DDMValStreamingDataset, "_process_data"):
            dataset = DDMValStreamingDataset(
                annotation_file="/fake/anno.json",
                video_root="/fake/videos",
            )

        fake_batch = [{"inp": MagicMock(), "label": 0}, {"inp": MagicMock(), "label": 1}]

        with patch("datasets.ddm_val_dataset.default_collate") as mock_dc:
            mock_dc.return_value = MagicMock()
            result = dataset.collate_fn(fake_batch)

        mock_dc.assert_called_once_with(fake_batch)
        assert result is mock_dc.return_value
