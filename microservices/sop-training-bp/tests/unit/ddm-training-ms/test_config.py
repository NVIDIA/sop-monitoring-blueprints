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
Unit tests for DDM-Net config/config.py.

Covers required-field validation, yaml/toml loading, and CLI argument priority.
"""

import argparse
import os

import pytest
import yaml
import toml
from omegaconf import OmegaConf, DictConfig

from config.config import (
    get_system_defaults,
    load_file_to_omegaconf,
    load_config_file,
    validate_config,
    merge_configs,
    save_config,
)


class TestGetSystemDefaults:
    """Tests for get_system_defaults()."""

    def test_get_system_defaults(self):
        """Test that defaults return a DictConfig with anno_path set to None."""
        result = get_system_defaults()
        assert isinstance(result, DictConfig)
        # None is the required sentinel value; validate_config relies on it to catch unset paths
        assert result.dataset_config.train_config.anno_path is None
        assert result.dataset_config.val_config.anno_path is None


class TestLoadConfigFile:
    """Tests for load_config_file()."""

    @pytest.mark.parametrize("ext", [".yaml", ".yml"])
    def test_load_yaml_returns_plain_dict(self, tmp_path, ext):
        """Test that .yaml and .yml files are loaded as plain dicts."""
        config = {"training_config": {"epochs": 10, "optimizer": "adamw"}}
        yaml_file = tmp_path / f"config{ext}"
        yaml_file.write_text(yaml.dump(config))

        result = load_config_file(str(yaml_file))

        assert isinstance(result, dict)
        assert result["training_config"]["epochs"] == 10
        assert result["training_config"]["optimizer"] == "adamw"

    def test_load_toml_returns_plain_dict(self, tmp_path):
        """Test that a toml file is loaded as a plain dict."""
        config = {"dataset_config": {"batch_size": 8, "workers": 5}}
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml.dumps(config))

        result = load_config_file(str(toml_file))

        assert isinstance(result, dict)
        assert result["dataset_config"]["batch_size"] == 8
        assert result["dataset_config"]["workers"] == 5

    def test_unsupported_format_raises_value_error(self, tmp_path):
        """Test that an unsupported file format raises ValueError."""
        json_file = tmp_path / "config.json"
        json_file.write_text('{"key": "value"}')

        with pytest.raises(ValueError):
            load_config_file(str(json_file))

    def test_nonexistent_file_raises(self, tmp_path):
        """Test that a missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config_file(str(tmp_path / "nonexistent.yaml"))


class TestLoadFileToOmegaConf:
    """Tests for load_file_to_omegaconf()."""

    @pytest.mark.parametrize("ext", [".yaml", ".yml"])
    def test_load_yaml_file(self, tmp_path, ext):
        """Test that .yaml and .yml files are loaded into an OmegaConf object."""
        config = {"training_config": {"epochs": 5, "learning_rate": 0.001}}
        yaml_file = tmp_path / f"config{ext}"
        yaml_file.write_text(yaml.dump(config))

        result = load_file_to_omegaconf(str(yaml_file))

        assert result.training_config.epochs == 5
        assert result.training_config.learning_rate == pytest.approx(0.001)

    def test_load_toml_file(self, tmp_path):
        """Test that a toml file is loaded into an OmegaConf object."""
        config = {"dataset_config": {"batch_size": 4}}
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml.dumps(config))

        result = load_file_to_omegaconf(str(toml_file))

        assert result.dataset_config.batch_size == 4

    def test_nonexistent_file_raises_file_not_found(self, tmp_path):
        """Test that a missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_file_to_omegaconf(str(tmp_path / "nonexistent.yaml"))

    def test_unsupported_format_raises_value_error(self, tmp_path):
        """Test that an unsupported file format raises ValueError."""
        json_file = tmp_path / "config.json"
        json_file.write_text('{"key": "value"}')

        with pytest.raises(ValueError):
            load_file_to_omegaconf(str(json_file))


class TestValidateConfig:
    """Tests for validate_config()."""

    def _make_valid_config(self):
        """Return a fully populated valid config."""
        return OmegaConf.create({
            "dataset_config": {
                "train_config": {
                    "anno_path": "/data/train_anno.json",
                    "data_root": "/data/train_videos",
                },
                "val_config": {
                    "anno_path": "/data/val_anno.json",
                    "data_root": "/data/val_videos",
                },
            }
        })

    def test_valid_config_does_not_raise(self):
        """Test that a fully populated config passes validation without error."""
        config = self._make_valid_config()
        validate_config(config)  # should not raise

    @pytest.mark.parametrize("split", ["train_config", "val_config"])
    def test_missing_anno_path_raises(self, split):
        """Test that a None anno_path raises ValueError."""
        config = self._make_valid_config()
        getattr(config.dataset_config, split).anno_path = None

        with pytest.raises(ValueError, match="annotation path"):
            validate_config(config)

    @pytest.mark.parametrize("split", ["train_config", "val_config"])
    def test_missing_data_root_raises(self, split):
        """Test that a None data_root raises ValueError."""
        config = self._make_valid_config()
        getattr(config.dataset_config, split).data_root = None

        with pytest.raises(ValueError):
            validate_config(config)

    def test_empty_string_anno_path_raises(self):
        """Test that an empty string anno_path raises ValueError."""
        config = self._make_valid_config()
        config.dataset_config.train_config.anno_path = ""

        with pytest.raises(ValueError):
            validate_config(config)

    def test_whitespace_only_anno_path_raises_validation_error(self):
        """Test that a whitespace-only anno_path is rejected by validation."""
        config = self._make_valid_config()
        config.dataset_config.train_config.anno_path = "   "  # whitespace-only

        with pytest.raises(ValueError, match="Training annotation path not set"):
            validate_config(config)


class TestMergeConfigs:
    """Tests for merge_configs() three-layer priority: defaults < config file < CLI args."""

    def _make_args(self, config_path=None, overrides=None, **kwargs):
        """Return a fake argparse.Namespace simulating CLI input."""
        namespace = argparse.Namespace(
            config=config_path,
            overrides=overrides or [],
            **kwargs
        )
        return namespace

    def test_defaults_used_when_no_config_no_cli(self):
        """Test that system defaults are used when no config file or CLI args are provided."""
        args = self._make_args()
        result = merge_configs(args)

        assert result.dataset_config.video_backend == "pyav"
        assert result.training_config.optimizer == "adamw"

    def test_config_file_overrides_defaults(self, tmp_path):
        """Test that config file values override system defaults."""
        config = {"training_config": {"epochs": 99}}
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump(config))

        args = self._make_args(config_path=str(yaml_file))
        result = merge_configs(args)

        assert result.training_config.epochs == 99

    def test_cli_overrides_config_file(self, tmp_path):
        """Test that CLI args take priority over config file values."""
        config = {"training_config": {"epochs": 50}}
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump(config))

        # CLI epochs=10 should override config file epochs=50
        args = self._make_args(config_path=str(yaml_file), epochs=10)
        result = merge_configs(args)

        assert result.training_config.epochs == 10

    @pytest.mark.parametrize("cli_key,config_split,path", [
        ("train_anno_path", "train_config", "/data/train.json"),
        ("val_anno_path", "val_config", "/data/val.json"),
    ])
    def test_cli_anno_path_mapped_correctly(self, cli_key, config_split, path):
        """Test that --train-anno-path and --val-anno-path are mapped into the correct config fields."""
        args = self._make_args(**{cli_key: path})
        result = merge_configs(args)

        assert getattr(result.dataset_config, config_split).anno_path == path

    @pytest.mark.parametrize("cli_key,config_split,path", [
        ("train_dataroot", "train_config", "/data/train_videos"),
        ("test_dataroot", "val_config", "/data/test_videos"),
    ])
    def test_cli_dataroot_mapped_correctly(self, cli_key, config_split, path):
        """Should map --train-dataroot / --test-dataroot to the correct data_root field."""
        args = self._make_args(**{cli_key: path})
        result = merge_configs(args)

        assert getattr(result.dataset_config, config_split).data_root == path

    def test_omegaconf_generic_override(self):
        """Test that OmegaConf-style key=value overrides are applied correctly."""
        args = self._make_args(overrides=["training_config.learning_rate=0.0001"])
        result = merge_configs(args)

        assert result.training_config.learning_rate == pytest.approx(0.0001)

    def test_seed_propagates_to_both_dataset_and_training_config(self):
        """Test that --seed is written to both dataset_config.seed and training_config.seed."""
        args = self._make_args(seed=999)
        result = merge_configs(args)

        assert result.dataset_config.seed == 999
        assert result.training_config.seed == 999

    def test_img_size_propagates_to_resolution_and_model(self):
        """Test that --img-size is written to both dataset_config.resolution and model_config.img_size."""
        args = self._make_args(img_size=512)
        result = merge_configs(args)

        assert result.dataset_config.resolution == 512
        assert result.model_config.img_size == 512

    def test_num_classes_propagates_to_dataset_and_model(self):
        """Test that --num-classes is written to both dataset_config.num_classes and model_config.num_classes."""
        args = self._make_args(num_classes=5)
        result = merge_configs(args)

        assert result.dataset_config.num_classes == 5
        assert result.model_config.num_classes == 5

    def test_seed_in_config_file_does_not_propagate_to_both(self, tmp_path):
        """Setting seed in a config file should only update training_config.seed.

        Unlike CLI --seed which dual-writes to both dataset_config and training_config,
        the config file layer merges directly without the dual-write logic, so
        dataset_config.seed stays at the default value.
        """
        config = {"training_config": {"seed": 42}}
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump(config))

        args = self._make_args(config_path=str(yaml_file))
        result = merge_configs(args)

        defaults = get_system_defaults()
        assert result.training_config.seed == 42
        assert result.dataset_config.seed == defaults.dataset_config.seed


class TestSaveConfig:
    """Tests for save_config()."""

    def test_save_config_creates_yaml_file(self, tmp_path):
        """Test that save_config creates a file at the given path."""
        config = {"training_config": {"epochs": 10, "optimizer": "adamw"}}
        save_path = str(tmp_path / "saved.yaml")

        save_config(config, save_path)

        assert os.path.exists(save_path)

    def test_save_config_content_is_correct(self, tmp_path):
        """Test that the saved YAML file round-trips back to the original dict."""
        config = {
            "dataset_config": {"batch_size": 4, "resolution": 224},
            "training_config": {"epochs": 5, "optimizer": "adamw"},
        }
        save_path = str(tmp_path / "saved.yaml")

        save_config(config, save_path)

        with open(save_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["dataset_config"]["batch_size"] == 4
        assert loaded["training_config"]["optimizer"] == "adamw"

    def test_save_config_overwrites_existing_file(self, tmp_path):
        """Test that saving to an existing path overwrites rather than appends."""
        save_path = str(tmp_path / "saved.yaml")

        save_config({"key": "old"}, save_path)
        save_config({"key": "new"}, save_path)

        with open(save_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["key"] == "new"
