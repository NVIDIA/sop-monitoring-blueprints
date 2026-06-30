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
Unit tests for DDMDataset in datasets/ddm_dataset.py.
Covers __init__ validation, process_data label logic, collate_fn, and __len__ behavior.
Heavy ML dependencies (torch, torchvision, av, transformers) are mocked throughout.
"""

import sys
import math
import json
from unittest.mock import MagicMock, patch, mock_open

import numpy as np  # must come before mock setup so _FakeLongTensor can use it

# Mock heavy ML packages to avoid real imports during testing.
_ML_MODULES = [
    "torch",
    "torch.nn",
    "torch.utils",
    "torch.utils.data",
    "torch.utils.data.dataloader",
    "torchvision",
    "torchvision.transforms",
    "av",
    "torchcodec",
    "torchcodec.decoders",
    "qwen_vl_utils",
    "transformers",
    "tqdm",
    "timm",
]
for _mod in _ML_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.modules["torch.utils.data"].Dataset = type("Dataset", (), {})

sys.modules["torch"].Tensor = type("Tensor", (), {})

class _FakeLongTensor:
    """Minimal stand-in for torch.LongTensor that supports .numpy()."""
    def __init__(self, data):
        self._data = np.array(list(data), dtype=np.int64)
    def numpy(self):
        return self._data
sys.modules["torch"].LongTensor = _FakeLongTensor

import pytest

from datasets.ddm_dataset import DDMDataset


def _dummy_process_data(self, anno_path):
    """Stub for process_data used in __init__ validation tests.

    Provides one label=0 and one label=1 sample so the train-mode ratio
    calculation in __init__ does not raise ZeroDivisionError.
    """
    self.seqs = np.array([
        {"video_id": "v0", "label": np.float64(0.0), "current_idx": 10,
         "block_idx": np.array([8, 9, 10, 11, 12])},
        {"video_id": "v0", "label": np.float64(1.0), "current_idx": 20,
         "block_idx": np.array([18, 19, 20, 21, 22])},
    ], dtype=object)
    self.video_paths = {}
    self.video_info = {}


class TestDDMDatasetInitValidation:
    """Tests for DDMDataset.__init__ parameter validation."""

    def test_invalid_mode_raises_assertion_error(self):
        """Test that an invalid mode raises AssertionError immediately."""
        with pytest.raises(AssertionError, match="Wrong mode"):
            DDMDataset(
                mode="invalid_mode",
                anno_path="/fake/anno.json",
                data_root="/fake/data",
            )

    def test_mode_is_case_insensitive(self):
        """Test that mode string is normalized to lowercase."""
        with patch.object(DDMDataset, "process_data", _dummy_process_data):
            # Should not raise
            dataset = DDMDataset(
                mode="Train",  # uppercase T
                anno_path="/fake/anno.json",
                data_root="/fake/data",
            )
            assert dataset.mode == "train"

    def test_invalid_backend_raises_assertion_error(self):
        """Test that an unsupported video_backend raises AssertionError."""
        with pytest.raises(AssertionError):
            DDMDataset(
                mode="train",
                anno_path="/fake/anno.json",
                data_root="/fake/data",
                video_backend="ffmpeg",
            )

    def test_valid_modes_are_accepted(self):
        """Test that train, val, and test are all accepted as valid modes."""
        for mode in ["train", "val", "test"]:
            with patch.object(DDMDataset, "process_data", _dummy_process_data):
                DDMDataset(
                    mode=mode,
                    anno_path="/fake/anno.json",
                    data_root="/fake/data",
                )

    def test_valid_backends_are_accepted(self):
        """Test that pyav and torchcodec are accepted as valid video backends."""
        for backend in ["pyav", "torchcodec"]:
            with patch.object(DDMDataset, "process_data", _dummy_process_data):
                DDMDataset(
                    mode="train",
                    anno_path="/fake/anno.json",
                    data_root="/fake/data",
                    video_backend=backend,
                )

    def test_processor_forces_pyav_backend(self):
        """Test that specifying a processor forces video_backend to pyav."""
        with patch.object(DDMDataset, "process_data", _dummy_process_data):
            # torchcodec passed but processor present — should be overridden to pyav
            dataset = DDMDataset(
                mode="train",
                anno_path="/fake/anno.json",
                data_root="/fake/data",
                video_backend="torchcodec",
                processor_name_or_path="some/processor",
            )
            assert dataset.video_backend == "pyav"

        with patch.object(DDMDataset, "process_data", _dummy_process_data):
            dataset = DDMDataset(
                mode="train",
                anno_path="/fake/anno.json",
                data_root="/fake/data",
                video_backend="torchcodec",
                processor_name_or_path="",
            )
            assert dataset.video_backend == "torchcodec"
            assert dataset.processor is None


    def test_use_cache_disabled_in_train_mode(self):
        """Test that use_cache is forced to False in train mode."""
        with patch.object(DDMDataset, "process_data", _dummy_process_data):
            dataset = DDMDataset(
                mode="train",
                anno_path="/fake/anno.json",
                data_root="/fake/data",
                use_cache=True,
            )
            assert dataset.use_cache is False

    def test_integer_resolution_converted_to_tuple(self):
        """Test that an integer resolution is converted to a (h, w) tuple."""
        with patch.object(DDMDataset, "process_data", _dummy_process_data):
            dataset = DDMDataset(
                mode="train",
                anno_path="/fake/anno.json",
                data_root="/fake/data",
                resolution=224,
            )
            assert dataset.resolution == (224, 224)

    def test_tuple_resolution_kept_as_is(self):
        """Test that a tuple resolution is stored unchanged."""
        with patch.object(DDMDataset, "process_data", _dummy_process_data):
            target_resolution = (128, 256)
            dataset = DDMDataset(
                mode="train",
                anno_path="/fake/anno.json",
                data_root="/fake/data",
                resolution=target_resolution,
            )
            assert dataset.resolution == target_resolution


class TestDDMDatasetCollateFunction:
    """Tests for DDMDataset.collate_fn."""

    def _make_dataset(self):
        """Return a dataset instance with process_data patched out."""
        with patch.object(DDMDataset, "process_data", _dummy_process_data):
            dataset = DDMDataset(
                mode="val",
                anno_path="/fake/anno.json",
                data_root="/fake/data",
            )
        return dataset

    def _make_sample(self, video_id, current_ids, inp=MagicMock(), label=MagicMock()):
        """Return a minimal sample dict matching the __getitem__ output format."""
        return {
            "inp": inp,
            "label": label,
            "path": [video_id],
            "current_ids": [current_ids],
        }

    def test_paths_are_flattened_to_list(self):
        """Test that per-sample path lists are flattened into a single list."""
        dataset = self._make_dataset()
        batch = [
            self._make_sample("video_1", 10),
            self._make_sample("video_2", 20),
        ]

        result = dataset.collate_fn(batch)

        assert "video_1" in result["path"]
        assert "video_2" in result["path"]
        assert len(result["path"]) == 2

    def test_current_ids_are_flattened_to_list(self):
        """Test that current_ids are collected into a flat Python list, not a tensor."""
        dataset = self._make_dataset()
        batch = [
            self._make_sample("video_1", 42),
            self._make_sample("video_2", 99),
        ]

        result = dataset.collate_fn(batch)

        assert isinstance(result["current_ids"], list)
        assert 42 in result["current_ids"]
        assert 99 in result["current_ids"]


class TestDDMDatasetProcessData:
    """Tests for DDMDataset.process_data label computation and sample assembly."""

    def _make_dataset_no_process(self, **kwargs):
        """Return a dataset with process_data skipped, using val mode by default.

        val mode avoids the ZeroDivisionError that train mode triggers when
        seqs is empty after process_data is patched to a no-op.
        """
        defaults = dict(
            mode="val",
            anno_path="/fake/anno.json",
            data_root="/fake/data",
            frames_per_side=2,
            downsample=1,
            min_change_dur=0.3,
        )
        defaults.update(kwargs)
        with patch.object(DDMDataset, "process_data"):
            return DDMDataset(**defaults)

    def _make_mock_decoder(self, fps=30.0, duration=10.0, vlen=300):
        """Return a mock video decoder with the given metadata values."""
        mock_vd = MagicMock()
        mock_vd.metadata.average_fps = fps
        mock_vd.metadata.duration_seconds = duration
        mock_vd.__len__ = MagicMock(return_value=vlen)
        return mock_vd

    def _run_process_data(self, dataset, annotation, fps=30.0, duration=10.0, vlen=300):
        """Run process_data with mocked I/O and a passthrough tqdm.

        tqdm in sys.modules is a MagicMock, so calling tqdm(iterable) returns
        another MagicMock rather than the iterable itself. The side_effect here
        restores the expected identity behavior so the inner loop executes.
        """
        mock_vd = self._make_mock_decoder(fps=fps, duration=duration, vlen=vlen)

        with patch("builtins.open", mock_open(read_data=json.dumps(annotation))), \
             patch("os.path.exists", return_value=True), \
             patch("datasets.ddm_dataset.tqdm", side_effect=lambda x, *a, **kw: x), \
             patch("datasets.ddm_dataset.PyAVVideoDecoder", return_value=mock_vd):
            dataset.process_data("/fake/anno.json")

    def test_video_not_found_raises_error(self):
        """Test that a missing video file raises FileNotFoundError."""
        dataset = self._make_dataset_no_process()
        annotation = {"v1": [{"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 5.0}]}

        with patch("builtins.open", mock_open(read_data=json.dumps(annotation))), \
             patch("os.path.exists", return_value=False), \
             patch("datasets.ddm_dataset.tqdm", side_effect=lambda x, *a, **kw: x):
            with pytest.raises(FileNotFoundError, match="Video file"):
                dataset.process_data("/fake/anno.json")

    def test_empty_annotation_produces_no_samples(self):
        """Test that an empty annotation dict results in zero samples."""
        dataset = self._make_dataset_no_process()

        with patch("builtins.open", mock_open(read_data=json.dumps({}))):
            dataset.process_data("/fake/anno.json")

        assert len(dataset.seqs) == 0

    def test_final_segment_excluded_from_boundary(self):
        """Test that a segment described as 'Final segment' is excluded from boundary calculation."""
        dataset = self._make_dataset_no_process(frames_per_side=1, downsample=1)
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 5.0},
                {"description": "Final segment", "start_timestamp": 5.0, "end_timestamp": 5.0},
            ]
        }
        self._run_process_data(dataset, annotation, fps=10.0, duration=5.0, vlen=50)

        boundary_samples = [s for s in dataset.seqs if s["label"] != 0]
        assert boundary_samples == [], f"Expected no boundary, got {len(boundary_samples)} samples"

    def test_final_segment_case_and_punctuation_insensitive(self):
        """Test that 'final segment' is filtered regardless of case, punctuation, or prefix."""
        variants = [
            "Final Segment",
            "Final segment",
            "final segment",
            "FINAL SEGMENT",
            "Final segment.",
            "Final segment ",
            "step 10: Final segment",
        ]
        for description in variants:
            dataset = self._make_dataset_no_process(frames_per_side=1, downsample=1)
            annotation = {
                "v1": [
                    {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 3.0},
                    {"description": description, "start_timestamp": 5.0, "end_timestamp": 10.0},
                ]
            }
            self._run_process_data(dataset, annotation, fps=10.0, duration=5.0, vlen=50)

            boundary_samples = [s for s in dataset.seqs if s["label"] != 0]
            assert boundary_samples == [], (
                f"'{description}' should be filtered but got {len(boundary_samples)} boundary samples"
            )

    def test_boundary_label_positions_correct(self):
        """Test that boundary frames are labeled 1 and non-boundary frames are labeled 0."""
        dataset = self._make_dataset_no_process(
            frames_per_side=1, downsample=1, min_change_dur=1.0
        )
        fps = 10.0
        vlen = 100
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 4.0},
                {"description": "Step 2", "start_timestamp": 6.0, "end_timestamp": 9.0},
            ]
        }
        self._run_process_data(dataset, annotation, fps=fps, duration=10.0, vlen=vlen)

        # boundary: midpoint of (4.0, 6.0) scaled by fps=10 → frame 50
        # half_dur: min_change_dur * fps / 2 → 5 frames
        # labeled range: frames 45 through 55
        boundary_samples = [s for s in dataset.seqs if s["label"] == 1]
        non_boundary_samples = [s for s in dataset.seqs if s["label"] == 0]

        assert len(boundary_samples) > 0, "Should have boundary-labeled samples"
        assert len(non_boundary_samples) > 0, "Should have non-boundary samples"

        for s in boundary_samples:
            assert 45 <= s["current_idx"] <= 55


    def test_first_and_last_frame_skipped(self):
        """Test that frame 0 and frame vlen-1 are never included as samples."""
        dataset = self._make_dataset_no_process(frames_per_side=1, downsample=1)
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 5.0},
            ]
        }
        self._run_process_data(dataset, annotation, fps=10.0, duration=10.0, vlen=100)

        current_indices = [s["current_idx"] for s in dataset.seqs]
        assert 0 not in current_indices
        assert 99 not in current_indices

    def test_block_idx_clipped_to_valid_range(self):
        """Test that block_idx values are clipped to [0, vlen-1]."""
        dataset = self._make_dataset_no_process(frames_per_side=2, downsample=3)
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 1.0},
            ]
        }
        self._run_process_data(dataset, annotation, fps=10.0, duration=2.0, vlen=20)

        for sample in dataset.seqs:
            for idx in sample["block_idx"]:
                assert 0 <= idx <= 19

    def test_downsample_affects_sample_density(self):
        """Test that a higher downsample value produces fewer samples."""
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 5.0},
            ]
        }

        dataset_d1 = self._make_dataset_no_process(downsample=1)
        self._run_process_data(dataset_d1, annotation, fps=10.0, duration=10.0, vlen=100)

        dataset_d5 = self._make_dataset_no_process(downsample=5)
        self._run_process_data(dataset_d5, annotation, fps=10.0, duration=10.0, vlen=100)

        assert len(dataset_d1.seqs) > len(dataset_d5.seqs)

    def test_multiple_boundaries_all_labeled(self):
        """Test that all boundaries across multiple step transitions are labeled."""
        dataset = self._make_dataset_no_process(
            frames_per_side=1, downsample=1, min_change_dur=0.5
        )
        fps = 10.0
        vlen = 200
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 3.0},
                {"description": "Step 2", "start_timestamp": 5.0, "end_timestamp": 7.0},
                {"description": "Step 3", "start_timestamp": 10.0, "end_timestamp": 15.0},
            ]
        }
        self._run_process_data(dataset, annotation, fps=fps, duration=20.0, vlen=vlen)

        # boundary1: midpoint of (3.0, 5.0) scaled by fps=10 → frame 40
        # boundary2: midpoint of (7.0, 10.0) scaled by fps=10 → frame 85
        boundary_samples = [s for s in dataset.seqs if s["label"] == 1]
        boundary_indices = {s["current_idx"] for s in boundary_samples}

        has_near_40 = any(35 <= idx <= 45 for idx in boundary_indices)
        has_near_85 = any(80 <= idx <= 90 for idx in boundary_indices)
        assert has_near_40, "Should have boundary-labeled samples near frame 40"
        assert has_near_85, "Should have boundary-labeled samples near frame 85"

    def test_video_metadata_stored_correctly(self):
        """Test that fps, duration, and vlen are stored in video_info after process_data."""
        dataset = self._make_dataset_no_process()
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 5.0},
            ]
        }
        self._run_process_data(dataset, annotation, fps=24.0, duration=15.0, vlen=360)

        assert "v1" in dataset.video_info
        assert dataset.video_info["v1"]["fps"] == pytest.approx(24.0)
        assert dataset.video_info["v1"]["duration"] == pytest.approx(15.0)
        assert dataset.video_info["v1"]["vlen"] == 360

    def test_video_path_stored_correctly(self):
        """Test that the video file path is stored in video_paths after process_data."""
        dataset = self._make_dataset_no_process()
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 5.0},
            ]
        }
        self._run_process_data(dataset, annotation, fps=30.0, duration=10.0, vlen=300)

        assert "v1" in dataset.video_paths
        assert dataset.video_paths["v1"].endswith("v1.mp4")

    def test_decoder_exception_silently_skips_video(self):
        """Test that a corrupt video (decoder raises) is silently skipped with no samples produced.

        video_paths and video_info still contain the key because they are set before the
        try block, but video_info[k] will be an empty dict with no fps/duration/vlen.
        """
        dataset = self._make_dataset_no_process()
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 5.0},
            ]
        }
        # spec=[] causes any attribute access on metadata to raise AttributeError.
        mock_vd = MagicMock()
        mock_vd.metadata = MagicMock(spec=[])

        with patch("builtins.open", mock_open(read_data=json.dumps(annotation))), \
             patch("os.path.exists", return_value=True), \
             patch("datasets.ddm_dataset.tqdm", side_effect=lambda x, *a, **kw: x), \
             patch("datasets.ddm_dataset.PyAVVideoDecoder", return_value=mock_vd):
            dataset.process_data("/fake/anno.json")

        assert len(dataset.seqs) == 0
        assert "v1" in dataset.video_paths
        assert "v1" in dataset.video_info
        assert "fps" not in dataset.video_info["v1"]

    def test_multiple_videos_all_samples_collected(self):
        """Test that samples from all videos in an annotation are collected into seqs."""
        dataset = self._make_dataset_no_process(frames_per_side=1, downsample=1)
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 5.0},
            ],
            "v2": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 3.0},
            ],
        }
        self._run_process_data(dataset, annotation, fps=10.0, duration=5.0, vlen=50)

        video_ids_in_seqs = {s["video_id"] for s in dataset.seqs}
        assert "v1" in video_ids_in_seqs
        assert "v2" in video_ids_in_seqs

        assert "v1" in dataset.video_paths
        assert "v2" in dataset.video_paths
        assert "v1" in dataset.video_info
        assert "v2" in dataset.video_info

        dataset_single = self._make_dataset_no_process(frames_per_side=1, downsample=1)
        single_annotation = {"v1": annotation["v1"]}
        self._run_process_data(dataset_single, single_annotation, fps=10.0, duration=5.0, vlen=50)
        assert len(dataset.seqs) > len(dataset_single.seqs)

    def test_block_idx_length_correct(self):
        """Test that each sample's block_idx has length 2 * frames_per_side + 1."""
        frames_per_side = 3
        dataset = self._make_dataset_no_process(frames_per_side=frames_per_side, downsample=1)
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 5.0},
            ]
        }
        self._run_process_data(dataset, annotation, fps=10.0, duration=10.0, vlen=100)

        expected_len = 2 * frames_per_side + 1
        for sample in dataset.seqs:
            assert len(sample["block_idx"]) == expected_len

    def test_current_idx_is_center_of_block(self):
        """Test that current_idx equals the center element of block_idx."""
        dataset = self._make_dataset_no_process(frames_per_side=2, downsample=1)
        annotation = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 5.0},
            ]
        }
        self._run_process_data(dataset, annotation, fps=10.0, duration=10.0, vlen=100)

        for sample in dataset.seqs:
            block = sample["block_idx"]
            center = block[len(block) // 2]
            assert sample["current_idx"] == center


class TestDDMDatasetLen:
    """Tests for DDMDataset.__len__ in train vs. val/test modes."""

    def _make_dataset_with_label_indices(self, mode, label_to_indices):
        """Return a dataset with label_to_indices manually overridden after construction.

        Uses _dummy_process_data so train mode's ratio calculation (label=1 count / label=0 count)
        does not raise ZeroDivisionError. label_to_indices is then replaced to test __len__ directly.
        """
        with patch.object(DDMDataset, "process_data", _dummy_process_data):
            dataset = DDMDataset(
                mode=mode,
                anno_path="/fake/anno.json",
                data_root="/fake/data",
            )
        dataset.labels_set = [0, 1]
        dataset.label_to_indices = label_to_indices
        return dataset

    def test_len_train_mode_returns_positive_sample_count(self):
        """Test that __len__ in train mode returns only the count of label=1 samples."""
        dataset = self._make_dataset_with_label_indices(
            mode="train",
            label_to_indices={0: np.array([0, 1, 2, 3, 4, 5, 6, 7]), 1: np.array([8, 9])},
        )
        assert len(dataset) == 2

    def test_len_val_mode_returns_total_sample_count(self):
        """Test that __len__ in val mode returns the total number of samples across all labels."""
        dataset = self._make_dataset_with_label_indices(
            mode="val",
            label_to_indices={0: np.array([0, 1, 2, 3, 4, 5, 6, 7]), 1: np.array([8, 9])},
        )
        assert len(dataset) == 10  # 8 + 2

    def test_len_train_is_less_than_val_for_same_data(self):
        """Test that train __len__ is always less than val __len__ for the same label distribution."""
        indices = {0: np.array([0, 1, 2, 3, 4, 5, 6, 7]), 1: np.array([8, 9])}
        train_ds = self._make_dataset_with_label_indices("train", {k: v.copy() for k, v in indices.items()})
        val_ds = self._make_dataset_with_label_indices("val", {k: v.copy() for k, v in indices.items()})
        assert len(train_ds) < len(val_ds)
