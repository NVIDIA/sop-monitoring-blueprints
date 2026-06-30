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
Unit tests for DDMDataModule in pl_ddm_datamodule.py.
Covers __init__ config parsing, setup dataset construction, and dataloader parameter wiring.
All ML dependencies are mocked; no GPU or real video files are required.
"""

import sys
from unittest.mock import MagicMock, patch, call

# Mock heavy ML packages before any project imports.
_ML_MODULES = [
    "torch",
    "torch.nn",
    "torch.utils",
    "torch.utils.data",
    "torch.utils.data.dataloader",
    "torch.distributed",
    "torchvision",
    "torchvision.transforms",
    "torchvision.transforms.v2",
    "torchvision.transforms.functional",
    "lightning",
    "av",
    "decord",
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

# Configure the decord mock with the attributes the code expects.
_decord_mock = sys.modules["decord"]
_decord_mock.bridge = MagicMock()
_decord_mock.VideoReader = MagicMock()
_decord_mock.cpu = MagicMock()

# Replace MagicMock base classes with real plain classes so subclasses can be instantiated.
sys.modules["torch.utils.data"].Dataset = type("Dataset", (), {})
sys.modules["torch.utils.data"].IterableDataset = type("IterableDataset", (), {})
sys.modules["lightning"].LightningDataModule = type("LightningDataModule", (), {})

import pytest

from pl_ddm_datamodule import DDMDataModule


def _make_config(**overrides):
    """Return a minimal valid dataset_config dict, with optional field overrides."""
    config = {
        "batch_size": 4,
        "workers": 2,
        "resolution": 224,
        "dataset": "DDMDataset",
        "num_classes": 2,
        "frames_per_side": 5,
        "downsample": 1,
        "min_change_dur": 0.3,
        "seed": 42,
        "video_backend": "pyav",
        "train_config": {
            "mode": "train",
            "anno_path": "/fake/train_anno.json",
            "data_root": "/fake/train_data",
            "augmentation": {},
        },
        "val_config": {
            "anno_path": "/fake/val_anno.json",
            "data_root": "/fake/val_data",
            "augmentation": {},
            "temporal_stride": 1,
        },
    }
    config.update(overrides)
    return config


class TestDDMDataModuleInit:
    """Tests for DDMDataModule.__init__() config parsing."""

    def test_integer_resolution_converted_to_tuple(self):
        """Test that an integer resolution is converted to a (H, W) tuple."""
        dm = DDMDataModule(_make_config(resolution=224))
        assert dm.resolution == (224, 224)

    def test_tuple_resolution_kept(self):
        """Test that a tuple resolution is stored unchanged."""
        dm = DDMDataModule(_make_config(resolution=(128, 256)))
        assert dm.resolution == (128, 256)

    def test_batch_size_stored(self):
        """Test that batch_size is stored from config."""
        dm = DDMDataModule(_make_config(batch_size=8))
        assert dm.batch_size == 8

    def test_num_workers_stored(self):
        """Test that num_workers is stored from the workers config key."""
        dm = DDMDataModule(_make_config(workers=4))
        assert dm.num_workers == 4

    def test_dataset_name_stored(self):
        """Test that the dataset name string is stored from config."""
        dm = DDMDataModule(_make_config(dataset="DDMDataset"))
        assert dm.dataset == "DDMDataset"


class TestDDMDataModuleSetup:
    """Tests for DDMDataModule.setup() dataset construction."""

    @patch("pl_ddm_datamodule.compose_default_augmentations")
    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_setup_fit_creates_train_and_val_datasets(
        self, mock_ddm_dataset, mock_val_dataset, mock_compose_aug
    ):
        """Test that setup with stage='fit' creates both train and val datasets."""
        mock_compose_aug.return_value = MagicMock()
        dm = DDMDataModule(_make_config())

        dm.setup(stage="fit")

        mock_ddm_dataset.assert_called_once()
        mock_val_dataset.assert_called_once()

    @patch("pl_ddm_datamodule.compose_default_augmentations")
    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_setup_none_also_creates_datasets(
        self, mock_ddm_dataset, mock_val_dataset, mock_compose_aug
    ):
        """Test that setup with stage=None also creates both datasets."""
        mock_compose_aug.return_value = MagicMock()
        dm = DDMDataModule(_make_config())

        dm.setup(stage=None)

        mock_ddm_dataset.assert_called_once()
        mock_val_dataset.assert_called_once()

    @patch("pl_ddm_datamodule.compose_default_augmentations")
    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_train_dataset_receives_correct_params(
        self, mock_ddm_dataset, mock_val_dataset, mock_compose_aug
    ):
        """Test that DDMDataset is called with mode='train' from config."""
        mock_transform = MagicMock()
        mock_compose_aug.return_value = mock_transform
        config = _make_config(
            frames_per_side=10,
            downsample=2,
            num_classes=3,
            seed=123,
            video_backend="torchcodec",
        )
        dm = DDMDataModule(config)

        dm.setup(stage="fit")

        assert mock_ddm_dataset.call_args.kwargs["mode"] == "train"

    @patch("pl_ddm_datamodule.compose_default_augmentations")
    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_val_dataset_receives_correct_params(
        self, mock_ddm_dataset, mock_val_dataset, mock_compose_aug
    ):
        """Test that DDMValStreamingDataset is called with temporal_stride from config."""
        mock_transform = MagicMock()
        mock_compose_aug.return_value = mock_transform
        config = _make_config()
        config["val_config"]["temporal_stride"] = 3
        dm = DDMDataModule(config)

        dm.setup(stage="fit")

        assert mock_val_dataset.call_args.kwargs["temporal_stride"] == 3

    @patch("pl_ddm_datamodule.compose_default_augmentations")
    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_augmentation_config_passed_to_compose(
        self, mock_ddm_dataset, mock_val_dataset, mock_compose_aug
    ):
        """Test that compose_default_augmentations is called once for train and once for val."""
        config = _make_config()
        config["train_config"]["augmentation"] = {"color_jitter": True}
        config["val_config"]["augmentation"] = {"color_jitter": False}
        dm = DDMDataModule(config)

        dm.setup(stage="fit")

        assert mock_compose_aug.call_count == 2

    def test_unsupported_dataset_raises_not_implemented(self):
        """Test that an unrecognized dataset name raises NotImplementedError."""
        config = _make_config(dataset="FakeDataset")
        dm = DDMDataModule(config)

        with pytest.raises(NotImplementedError, match="Wrong dataset name"):
            dm.setup(stage="fit")

    @patch("pl_ddm_datamodule.compose_default_augmentations")
    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_setup_stores_dataset_instances_on_module(
        self, mock_ddm_dataset, mock_val_dataset, mock_compose_aug
    ):
        """Test that setup assigns train_dataset and val_dataset attributes on the module."""
        mock_compose_aug.return_value = MagicMock()
        dm = DDMDataModule(_make_config())
        dm.setup(stage="fit")

        assert dm.train_dataset is mock_ddm_dataset.return_value
        assert dm.val_dataset is mock_val_dataset.return_value

    @patch("pl_ddm_datamodule.compose_default_augmentations")
    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_stage_test_or_predict_does_not_create_datasets(
        self, mock_ddm_dataset, mock_val_dataset, mock_compose_aug
    ):
        """Test that setup with stage='test' or 'predict' does not construct any datasets."""
        dm = DDMDataModule(_make_config())

        dm.setup(stage="test")
        mock_ddm_dataset.assert_not_called()

        dm.setup(stage="predict")
        mock_ddm_dataset.assert_not_called()


class TestDDMDataModuleDataloaders:
    """Tests for DDMDataModule train_dataloader() and val_dataloader() wiring."""

    @patch("pl_ddm_datamodule.DataLoader")
    @patch("pl_ddm_datamodule.compose_default_augmentations")
    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_train_dataloader_params(
        self, mock_ddm_dataset, mock_val_dataset, mock_compose_aug, mock_data_loader
    ):
        """Test that train_dataloader uses shuffle=True and the configured batch_size."""
        mock_compose_aug.return_value = MagicMock()
        mock_data_loader.return_value = MagicMock()
        config = _make_config(batch_size=16, workers=4)
        dm = DDMDataModule(config)
        dm.setup(stage="fit")

        dm.train_dataloader()

        kw = mock_data_loader.call_args.kwargs
        assert kw["batch_size"] == 16
        assert kw["shuffle"] is True

    @patch("pl_ddm_datamodule.DataLoader")
    @patch("pl_ddm_datamodule.compose_default_augmentations")
    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_val_dataloader_no_shuffle(
        self, mock_ddm_dataset, mock_val_dataset, mock_compose_aug, mock_data_loader
    ):
        """Test that val_dataloader does not shuffle."""
        mock_compose_aug.return_value = MagicMock()
        mock_data_loader.return_value = MagicMock()
        dm = DDMDataModule(_make_config())
        dm.setup(stage="fit")

        dm.val_dataloader()

        assert mock_data_loader.call_args.kwargs["shuffle"] is False

    @patch("pl_ddm_datamodule.DataLoader")
    @patch("pl_ddm_datamodule.compose_default_augmentations")
    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_train_dataloader_uses_collate_fn(
        self, mock_ddm_dataset, mock_val_dataset, mock_compose_aug, mock_data_loader
    ):
        """Test that train_dataloader passes a collate_fn to DataLoader."""
        mock_compose_aug.return_value = MagicMock()
        mock_data_loader.return_value = MagicMock()
        dm = DDMDataModule(_make_config())
        dm.setup(stage="fit")

        dm.train_dataloader()

        assert "collate_fn" in mock_data_loader.call_args.kwargs

    @patch("pl_ddm_datamodule.DataLoader")
    @patch("pl_ddm_datamodule.compose_default_augmentations")
    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_val_dataloader_uses_collate_fn(
        self, mock_ddm_dataset, mock_val_dataset, mock_compose_aug, mock_data_loader
    ):
        """Test that val_dataloader passes a collate_fn to DataLoader."""
        mock_compose_aug.return_value = MagicMock()
        mock_data_loader.return_value = MagicMock()
        dm = DDMDataModule(_make_config())
        dm.setup(stage="fit")

        dm.val_dataloader()

        assert "collate_fn" in mock_data_loader.call_args.kwargs

    @patch("pl_ddm_datamodule.DataLoader")
    @patch("pl_ddm_datamodule.compose_default_augmentations")
    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_train_dataloader_num_workers(
        self, mock_ddm_dataset, mock_val_dataset, mock_compose_aug, mock_data_loader
    ):
        """Test that train_dataloader uses num_workers from config."""
        mock_compose_aug.return_value = MagicMock()
        mock_data_loader.return_value = MagicMock()
        config = _make_config(workers=8)
        dm = DDMDataModule(config)
        dm.setup(stage="fit")

        dm.train_dataloader()

        kw = mock_data_loader.call_args.kwargs
        assert kw["num_workers"] == 8

    @patch("pl_ddm_datamodule.DataLoader")
    @patch("pl_ddm_datamodule.compose_default_augmentations")
    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_val_dataloader_num_workers_and_batch_size(
        self, mock_ddm_dataset, mock_val_dataset, mock_compose_aug, mock_data_loader
    ):
        """Test that val_dataloader uses both num_workers and batch_size from config."""
        mock_compose_aug.return_value = MagicMock()
        mock_data_loader.return_value = MagicMock()
        config = _make_config(workers=6, batch_size=32)
        dm = DDMDataModule(config)
        dm.setup(stage="fit")

        dm.val_dataloader()

        kw = mock_data_loader.call_args.kwargs
        assert kw["num_workers"] == 6
        assert kw["batch_size"] == 32

    @patch("pl_ddm_datamodule.DataLoader")
    @patch("pl_ddm_datamodule.compose_default_augmentations")
    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_train_dataloader_pin_memory_and_drop_last(
        self, mock_ddm_dataset, mock_val_dataset, mock_compose_aug, mock_data_loader
    ):
        """Test that train_dataloader sets pin_memory=True and drop_last=False."""
        mock_compose_aug.return_value = MagicMock()
        mock_data_loader.return_value = MagicMock()
        dm = DDMDataModule(_make_config())
        dm.setup(stage="fit")

        dm.train_dataloader()

        kw = mock_data_loader.call_args.kwargs
        assert kw["pin_memory"] is True
        assert kw["drop_last"] is False

    @patch("pl_ddm_datamodule.DataLoader")
    @patch("pl_ddm_datamodule.compose_default_augmentations")
    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_train_dataset_collate_fn_is_from_train_dataset(
        self, mock_ddm_dataset, mock_val_dataset, mock_compose_aug, mock_data_loader
    ):
        """Test that train_dataloader uses collate_fn from train_dataset, not val_dataset."""
        mock_compose_aug.return_value = MagicMock()
        mock_data_loader.return_value = MagicMock()
        dm = DDMDataModule(_make_config())
        dm.setup(stage="fit")

        dm.train_dataloader()

        kw = mock_data_loader.call_args.kwargs
        assert kw["collate_fn"] is mock_ddm_dataset.return_value.collate_fn
