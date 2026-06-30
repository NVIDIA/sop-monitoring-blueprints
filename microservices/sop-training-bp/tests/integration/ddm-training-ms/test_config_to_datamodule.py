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
Integration tests: config.py → DDMDataModule → DDMDataset / DDMValStreamingDataset.

Verifies that config key names, dataset constructor signatures, and augmentation
composition are mutually compatible without requiring GPU or real video files.
"""

import sys
import argparse
from unittest.mock import MagicMock, patch

_ML_MODULES = [
    "torch",
    "torch.nn",
    "torch.utils",
    "torch.utils.data",
    "torch.utils.data.dataloader",
    "torch.distributed",
    "torchvision",
    "torchvision.transforms",
    "torchvision.transforms.functional",
    "av",
    "decord",
    "torchcodec",
    "torchcodec.decoders",
    "qwen_vl_utils",
    "transformers",
    "tqdm",
    "timm",
    "lightning",
]
for _mod in _ML_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

_decord_mock = sys.modules["decord"]
_decord_mock.bridge = MagicMock()
_decord_mock.VideoReader = MagicMock()
_decord_mock.cpu = MagicMock()

# LightningDataModule must be a real class so DDMDataModule can inherit and set instance attributes
sys.modules["lightning"].LightningDataModule = type("LightningDataModule", (), {})

# Only clear pl_ddm_datamodule if it was mocked (e.g., when running integration tests alone
# without unit tests, test_app_to_training_pipeline.py may have mocked it). Do NOT clear
# datasets modules — that would invalidate string-based patches in unit tests collected earlier.
from unittest.mock import MagicMock as _MagicMock
if isinstance(sys.modules.get("pl_ddm_datamodule"), _MagicMock):
    sys.modules.pop("pl_ddm_datamodule")

import pytest
from omegaconf import OmegaConf

from config.config import get_system_defaults, merge_configs
from datasets.ddm_dataset import DDMDataset
from datasets.ddm_val_dataset import DDMValStreamingDataset
from datasets.default_aug import compose_default_augmentations
from pl_ddm_datamodule import DDMDataModule


def _make_valid_config_dict(tmp_path, **yaml_overrides):
    """Build a real config dict via the full merge_configs() pipeline from a minimal YAML."""
    yaml_content = {
        "dataset_config": {
            "dataset": "DDMDataset",
            "resolution": 224,
            "frames_per_side": 5,
            "downsample": 1,
            "min_change_dur": 0.3,
            "num_classes": 2,
            "seed": 42,
            "video_backend": "pyav",
            "batch_size": 2,
            "workers": 2,
            "train_config": {
                "mode": "train",
                "anno_path": "/fake/train_anno.json",
                "data_root": "/fake/train_data",
                "augmentation": {},
            },
            "val_config": {
                "mode": "val",
                "anno_path": "/fake/val_anno.json",
                "data_root": "/fake/val_data",
                "temporal_stride": 1,
                "augmentation": {},
            },
        },
        "model_config": {
            "model_name": "multiframes_resnet",
            "backbone": "resnet50",
            "pretrained": None,
            "freeze_backbone": False,
            "num_classes": 2,
            "img_size": 224,
        },
        "training_config": {
            "optimizer": "adamw",
            "learning_rate": 0.01,
            "weight_decay": 0.0001,
            "momentum": 0.9,
            "scheduler": "step",
            "warmup_epochs": 0,
            "epochs": 1,
            "output": str(tmp_path / "output"),
            "exp_name": "test_integration",
        },
    }
    for key, val in yaml_overrides.items():
        parts = key.split(".")
        d = yaml_content
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = val

    import yaml
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(yaml_content, f)

    args = argparse.Namespace(config=str(config_path), overrides=[])
    cfg = merge_configs(args)
    OmegaConf.resolve(cfg)
    return OmegaConf.to_container(cfg, resolve=True)


class TestConfigToDDMDataModule:
    """Tests for config.py → DDMDataModule.__init__() interface."""

    def test_datamodule_init_with_real_config(self, tmp_path):
        """Test that DDMDataModule initializes from a real config dict without raising."""
        config = _make_valid_config_dict(tmp_path)
        dm = DDMDataModule(config["dataset_config"])

        assert dm.batch_size == 2
        assert dm.num_workers == 2
        assert dm.resolution == (224, 224)
        assert dm.dataset == "DDMDataset"

    def test_integer_resolution_from_config(self, tmp_path):
        """Test that an integer resolution in config is converted to a (H, W) tuple."""
        config = _make_valid_config_dict(tmp_path, **{"dataset_config.resolution": 384})
        dm = DDMDataModule(config["dataset_config"])

        assert dm.resolution == (384, 384)


class TestConfigToDatasets:
    """Tests for DDMDataModule.setup() → DDMDataset / DDMValStreamingDataset parameter passing.

    Patches DDMDataset and DDMValStreamingDataset at their use site in pl_ddm_datamodule so
    that setup() can run without real video files, and call args verify config key alignment.
    """

    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_setup_fit_constructs_both_datasets(
        self, mock_ddm_dataset, mock_val_dataset, tmp_path
    ):
        """Test that setup('fit') instantiates both DDMDataset and DDMValStreamingDataset."""
        config = _make_valid_config_dict(tmp_path)
        dm = DDMDataModule(config["dataset_config"])

        dm.setup(stage="fit")

        mock_ddm_dataset.assert_called_once()
        mock_val_dataset.assert_called_once()
        assert dm.train_dataset is mock_ddm_dataset.return_value
        assert dm.val_dataset is mock_val_dataset.return_value

    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_setup_stage_none_constructs_datasets(
        self, mock_ddm_dataset, mock_val_dataset, tmp_path
    ):
        """Test that setup(stage=None) also creates both datasets (same as 'fit')."""
        config = _make_valid_config_dict(tmp_path)
        dm = DDMDataModule(config["dataset_config"])

        dm.setup(stage=None)

        mock_ddm_dataset.assert_called_once()
        mock_val_dataset.assert_called_once()

    def test_setup_unknown_dataset_raises(self, tmp_path):
        """Test that setup raises NotImplementedError for an unrecognized dataset name."""
        config = _make_valid_config_dict(tmp_path, **{"dataset_config.dataset": "UnknownDataset"})
        dm = DDMDataModule(config["dataset_config"])

        with pytest.raises(NotImplementedError):
            dm.setup(stage="fit")

    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_train_dataset_constructor_kwargs_match_config(
        self, mock_ddm_dataset, mock_val_dataset, tmp_path
    ):
        """Test that DDMDataset is called with the correct kwargs from config."""
        config = _make_valid_config_dict(tmp_path)
        dm = DDMDataModule(config["dataset_config"])

        dm.setup(stage="fit")

        kwargs = mock_ddm_dataset.call_args.kwargs
        assert kwargs["mode"] == "train"
        assert kwargs["frames_per_side"] == 5
        assert kwargs["downsample"] == 1
        assert kwargs["resolution"] == (224, 224)
        assert kwargs["num_classes"] == 2
        assert kwargs["anno_path"] == "/fake/train_anno.json"
        assert kwargs["data_root"] == "/fake/train_data"

    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_val_dataset_constructor_kwargs_match_config(
        self, mock_ddm_dataset, mock_val_dataset, tmp_path
    ):
        """Test that DDMValStreamingDataset is called with the correct kwargs from config."""
        config = _make_valid_config_dict(tmp_path)
        dm = DDMDataModule(config["dataset_config"])

        dm.setup(stage="fit")

        kwargs = mock_val_dataset.call_args.kwargs
        assert kwargs["frames_per_side"] == 5
        assert kwargs["downsample"] == 1
        assert kwargs["temporal_stride"] == 1
        assert kwargs["resolution"] == (224, 224)
        assert kwargs["min_change_dur"] == pytest.approx(0.3)
        assert kwargs["annotation_file"] == "/fake/val_anno.json"
        assert kwargs["video_root"] == "/fake/val_data"

    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_custom_config_values_propagated(
        self, mock_ddm_dataset, mock_val_dataset, tmp_path
    ):
        """Test that non-default config values propagate to both dataset constructors."""
        config = _make_valid_config_dict(
            tmp_path,
            **{
                "dataset_config.frames_per_side": 10,
                "dataset_config.downsample": 3,
                "dataset_config.min_change_dur": 0.5,
                "dataset_config.resolution": 384,
            }
        )
        dm = DDMDataModule(config["dataset_config"])

        dm.setup(stage="fit")

        train_kwargs = mock_ddm_dataset.call_args.kwargs
        assert train_kwargs["frames_per_side"] == 10
        assert train_kwargs["downsample"] == 3
        assert train_kwargs["resolution"] == (384, 384)

        val_kwargs = mock_val_dataset.call_args.kwargs
        assert val_kwargs["frames_per_side"] == 10
        assert val_kwargs["downsample"] == 3
        assert val_kwargs["min_change_dur"] == pytest.approx(0.5)


class TestDataloaders:
    """Tests for train_dataloader() and val_dataloader() after setup."""

    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_train_dataloader_returns_dataloader(
        self, mock_ddm_dataset, mock_val_dataset, tmp_path
    ):
        """Test that train_dataloader() returns a DataLoader without raising."""
        config = _make_valid_config_dict(tmp_path)
        dm = DDMDataModule(config["dataset_config"])
        dm.setup(stage="fit")

        loader = dm.train_dataloader()

        assert loader is not None

    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_val_dataloader_returns_dataloader(
        self, mock_ddm_dataset, mock_val_dataset, tmp_path
    ):
        """Test that val_dataloader() returns a DataLoader without raising."""
        config = _make_valid_config_dict(tmp_path)
        dm = DDMDataModule(config["dataset_config"])
        dm.setup(stage="fit")

        loader = dm.val_dataloader()

        assert loader is not None


class TestConfigToAugmentation:
    """Tests for augmentation config → compose_default_augmentations → Dataset.transform."""

    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_empty_augmentation_config(
        self, mock_ddm_dataset, mock_val_dataset, tmp_path
    ):
        """Test that an empty augmentation dict does not crash setup."""
        config = _make_valid_config_dict(tmp_path)
        dm = DDMDataModule(config["dataset_config"])

        dm.setup(stage="fit")

        mock_ddm_dataset.assert_called_once()
        mock_val_dataset.assert_called_once()

    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_full_augmentation_config(
        self, mock_ddm_dataset, mock_val_dataset, tmp_path
    ):
        """Test that a full augmentation config with all supported fields does not crash setup."""
        import yaml

        yaml_content = {
            "dataset_config": {
                "dataset": "DDMDataset",
                "resolution": 224,
                "frames_per_side": 5,
                "downsample": 1,
                "min_change_dur": 0.3,
                "num_classes": 2,
                "seed": 42,
                "video_backend": "pyav",
                "batch_size": 2,
                "workers": 2,
                "train_config": {
                    "mode": "train",
                    "anno_path": "/fake/train_anno.json",
                    "data_root": "/fake/train_data",
                    "augmentation": {
                        "RandomResize": {
                            "enabled": True,
                            "interpolation": ["bilinear", "bicubic"],
                            "antialias_prob": 0.5,
                        },
                        "ColorJitter": {
                            "enabled": True,
                            "brightness": 0.25,
                            "contrast": 0.3,
                            "saturation": 0.15,
                            "hue": 0.02,
                        },
                        "GaussianBlur": {
                            "enabled": True,
                            "apply_prob": 0.5,
                            "kernel_size": 3,
                            "sigma": [0.1, 0.5],
                        },
                    },
                },
                "val_config": {
                    "mode": "val",
                    "anno_path": "/fake/val_anno.json",
                    "data_root": "/fake/val_data",
                    "temporal_stride": 1,
                    "augmentation": {},
                },
            },
        }
        config_path = tmp_path / "aug_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(yaml_content, f)

        args = argparse.Namespace(config=str(config_path), overrides=[])
        cfg = merge_configs(args)
        OmegaConf.resolve(cfg)
        config = OmegaConf.to_container(cfg, resolve=True)

        dm = DDMDataModule(config["dataset_config"])
        dm.setup(stage="fit")

        mock_ddm_dataset.assert_called_once()

    @patch("pl_ddm_datamodule.DDMValStreamingDataset")
    @patch("pl_ddm_datamodule.DDMDataset")
    def test_transform_is_passed_to_train_dataset(
        self, mock_ddm_dataset, mock_val_dataset, tmp_path
    ):
        """Test that a transform object is passed to the DDMDataset constructor."""
        config = _make_valid_config_dict(tmp_path)
        dm = DDMDataModule(config["dataset_config"])

        dm.setup(stage="fit")

        kwargs = mock_ddm_dataset.call_args.kwargs
        # transform is always passed (either real transforms or MagicMock from mocked torchvision)
        assert "transform" in kwargs


class TestConfigKeyConsistency:
    """Tests that system defaults contain all keys required by DDMDataModule and DDMDataset."""

    def test_system_defaults_has_required_dataset_keys(self):
        """Test that system defaults contain all keys DDMDataModule reads at init."""
        cfg = get_system_defaults()
        ds_cfg = cfg.dataset_config

        required_keys = ["batch_size", "workers", "resolution", "dataset"]
        for key in required_keys:
            assert key in ds_cfg, f"Missing key '{key}' in system defaults dataset_config"

    def test_system_defaults_has_train_config(self):
        """Test that dataset_config contains train_config."""
        cfg = get_system_defaults()
        assert "train_config" in cfg.dataset_config

    def test_system_defaults_has_val_config(self):
        """Test that dataset_config contains val_config."""
        cfg = get_system_defaults()
        assert "val_config" in cfg.dataset_config

    def test_train_config_has_required_keys(self):
        """Test that train_config contains mode, anno_path, and data_root."""
        cfg = get_system_defaults()
        train_cfg = cfg.dataset_config.train_config
        for key in ["mode", "anno_path", "data_root"]:
            assert key in train_cfg, f"Missing key '{key}' in train_config"

    def test_dataset_config_has_dataset_constructor_keys(self):
        """Test that dataset_config contains all keys needed by DDMDataset constructor."""
        cfg = get_system_defaults()
        ds_cfg = cfg.dataset_config
        constructor_keys = [
            "num_classes", "frames_per_side", "downsample",
            "min_change_dur", "seed", "video_backend",
        ]
        for key in constructor_keys:
            assert key in ds_cfg, f"Missing key '{key}' needed by DDMDataset constructor"
