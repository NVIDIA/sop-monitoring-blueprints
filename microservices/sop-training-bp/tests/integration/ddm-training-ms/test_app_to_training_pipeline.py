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
Integration tests: app.py config modification → train_sop_lightning.py SOPLightningModule.

Verifies that app.py's config dict structure is compatible with SOPLightningModule's
constructor and that YAML round-trips preserve all required fields.
"""

import sys
import os
import copy
import argparse
from unittest.mock import MagicMock

_ML_MODULES = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.distributed",
    "torch.optim",
    "torch.optim.lr_scheduler",
    "torchvision",
    "torchvision.transforms",
    "lightning",
    "lightning.pytorch",
    "lightning.pytorch.callbacks",
    "lightning.pytorch.strategies",
    "lightning.pytorch.plugins",
    "lightning.pytorch.plugins.environments",
    "av",
    "decord",
    "torchcodec",
    "torchcodec.decoders",
    "qwen_vl_utils",
    "transformers",
    "tqdm",
    "timm",
    "utils.getter",
    "utils.metric",
    "utils.visualize",
    "utils.model_ema",
    "pl_ddm_datamodule",
]
for _mod in _ML_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

_decord_mock = sys.modules["decord"]
_decord_mock.bridge = MagicMock()
_decord_mock.VideoReader = MagicMock()
_decord_mock.cpu = MagicMock()

# LightningModule must be a real class so SOPLightningModule can set instance attributes
sys.modules["lightning"].LightningModule = type("LightningModule", (), {
    "__init__": lambda self, *a, **kw: None,
    "save_hyperparameters": lambda self, *a, **kw: None,
    "log": lambda self, *a, **kw: None,
})

# Re-import train_sop_lightning so its SOPLightningModule uses the real LightningModule above.
# If another test file was collected first (e.g., test_train_sop_lightning.py), the cached
# module's SOPLightningModule inherits from MagicMock, preventing instance attribute setting.
sys.modules.pop("train_sop_lightning", None)

import yaml
import pytest
from omegaconf import OmegaConf

from config.config import merge_configs
from train_sop_lightning import SOPLightningModule


def _simulate_app_config_modification(tmp_path):
    """Replicate app.py's run_fine_tuning() path/config modification and write the result to disk."""
    base_config = {
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
                "anno_path": "/placeholder/train_anno.json",
                "data_root": "/placeholder/train_data",
                "augmentation": {},
            },
            "val_config": {
                "mode": "val",
                "anno_path": "/placeholder/val_anno.json",
                "data_root": "/placeholder/val_data",
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
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "momentum": 0.9,
            "opt_eps": None,
            "opt_betas": None,
            "scheduler": "step",
            "warmup_epochs": 0,
            "warmup_lr": 0.0001,
            "decay_epochs": 2,
            "decay_rate": 0.5,
            "min_lr": 1e-10,
            "patience_epochs": 10,
            "epochs": 1,
            "eval_freq": 1,
            "seed": 42,
            "clip_grad": None,
            "clip_mode": "norm",
            "amp": False,
            "model_ema": False,
            "model_ema_decay": 0.999,
            "model_ema_start_epoch": 0,
            "eval_metric": "f1_score",
            "save_visualizations": False,
            "log_interval": 50,
            "output": str(tmp_path / "output"),
            "exp_name": "test_job",
            "resume": None,
            "checkpoint_top_k": 3,
            "num_gpus": 1,
            "num_nodes": 1,
            "strategy": "auto",
        },
    }

    dataset_path = "/fake/dataset/train"
    validation_dataset_path = "/fake/dataset/val"
    job_id = "test-job-123"

    # deep copy to avoid mutating base_config's nested dicts
    train_config = copy.deepcopy(base_config)
    train_config["dataset_config"]["train_config"]["anno_path"] = os.path.join(
        dataset_path, "ddm_train_annotation.json"
    )
    train_config["dataset_config"]["train_config"]["data_root"] = dataset_path
    train_config["dataset_config"]["val_config"]["anno_path"] = os.path.join(
        validation_dataset_path, "ddm_val_annotation.json"
    )
    train_config["dataset_config"]["val_config"]["data_root"] = validation_dataset_path
    train_config["training_config"]["output"] = str(tmp_path / "results" / job_id)
    train_config["training_config"]["exp_name"] = job_id

    job_config_path = tmp_path / f"{job_id}.yaml"
    with open(job_config_path, "w") as f:
        yaml.dump(train_config, f)

    return train_config, str(job_config_path)


class TestAppConfigToMainFunction:
    """Tests that app.py's modified config has all keys required by train_sop_lightning.main()."""

    def test_app_modified_config_has_all_keys_for_main(self, tmp_path):
        """Test that the modified config contains all keys accessed by main()."""
        config, _ = _simulate_app_config_modification(tmp_path)

        assert "dataset_config" in config
        assert "model_config" in config
        assert "training_config" in config

        dataset_cfg = config["dataset_config"]
        model_cfg = config["model_config"]
        train_cfg = config["training_config"]

        for key in ["batch_size", "workers", "resolution", "dataset"]:
            assert key in dataset_cfg, f"Missing '{key}' in dataset_config"

        for key in ["model_name", "backbone", "num_classes", "pretrained", "freeze_backbone"]:
            assert key in model_cfg, f"Missing '{key}' in model_config"

        for key in [
            "learning_rate", "weight_decay", "optimizer", "scheduler",
            "warmup_epochs", "decay_epochs", "decay_rate", "min_lr",
            "warmup_lr", "clip_grad", "clip_mode", "eval_metric",
            "save_visualizations", "momentum", "opt_eps", "opt_betas",
        ]:
            assert key in train_cfg, f"Missing '{key}' in training_config"

        assert "frames_per_side" in dataset_cfg
        assert "val_config" in dataset_cfg
        assert "anno_path" in dataset_cfg["val_config"]
        assert "resolution" in dataset_cfg

    def test_app_modified_config_anno_paths_are_set(self, tmp_path):
        """Test that anno_path and data_root are non-None after app modification."""
        config, _ = _simulate_app_config_modification(tmp_path)

        ds = config["dataset_config"]
        assert ds["train_config"]["anno_path"] is not None
        assert ds["train_config"]["data_root"] is not None
        assert ds["val_config"]["anno_path"] is not None
        assert ds["val_config"]["data_root"] is not None

    def test_app_modified_config_output_paths_are_set(self, tmp_path):
        """Test that output and exp_name are non-default after app modification."""
        config, _ = _simulate_app_config_modification(tmp_path)

        assert config["training_config"]["output"] is not None
        assert config["training_config"]["exp_name"] is not None
        assert config["training_config"]["exp_name"] != "exp"


class TestAppConfigToSOPLightningModule:
    """Tests for app.py config → SOPLightningModule constructor compatibility."""

    def _build_module(self, tmp_path):
        """Helper: build SOPLightningModule from simulated app config."""
        config, _ = _simulate_app_config_modification(tmp_path)
        dataset_cfg = config["dataset_config"]
        model_cfg = config["model_config"]
        train_cfg = config["training_config"]
        return SOPLightningModule(
            model_name=model_cfg["model_name"],
            backbone=model_cfg["backbone"],
            num_classes=model_cfg["num_classes"],
            frames_per_side=dataset_cfg["frames_per_side"],
            pretrained=model_cfg["pretrained"],
            freeze_backbone=model_cfg["freeze_backbone"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            optimizer=train_cfg["optimizer"],
            scheduler=train_cfg["scheduler"],
            warmup_epochs=train_cfg["warmup_epochs"],
            decay_epochs=train_cfg["decay_epochs"],
            decay_rate=train_cfg["decay_rate"],
            min_lr=train_cfg["min_lr"],
            warmup_lr=train_cfg["warmup_lr"],
            clip_grad=train_cfg["clip_grad"],
            clip_mode=train_cfg["clip_mode"],
            eval_metric=train_cfg["eval_metric"],
            val_anno_path=dataset_cfg["val_config"]["anno_path"],
            save_visualizations=train_cfg["save_visualizations"],
            momentum=train_cfg["momentum"],
            opt_eps=train_cfg["opt_eps"],
            opt_betas=train_cfg["opt_betas"],
            resolution=dataset_cfg["resolution"],
        )

    def test_sop_module_init_with_app_config(self, tmp_path):
        """Test that SOPLightningModule initializes from app config without raising."""
        module = self._build_module(tmp_path)

        assert module is not None

    def test_sop_module_stores_model_and_criterion(self, tmp_path):
        """Test that constructor assigns model, criterion, and validation_step_outputs."""
        # hparams is not checkable here because lightning is mocked; verify direct assignments instead
        module = self._build_module(tmp_path)

        assert module.model is not None
        assert module.criterion is not None
        assert module.validation_step_outputs == []


class TestYamlRoundTrip:
    """Tests for YAML serialization/deserialization fidelity across the app.py → main() path."""

    def test_yaml_roundtrip_preserves_structure(self, tmp_path):
        """Test that key values survive a YAML write/read cycle unchanged."""
        config, yaml_path = _simulate_app_config_modification(tmp_path)

        args = argparse.Namespace(config=yaml_path, overrides=[])
        loaded_cfg = merge_configs(args)
        OmegaConf.resolve(loaded_cfg)
        loaded_config = OmegaConf.to_container(loaded_cfg, resolve=True)

        assert loaded_config["dataset_config"]["train_config"]["anno_path"] == config["dataset_config"]["train_config"]["anno_path"]
        assert loaded_config["dataset_config"]["val_config"]["anno_path"] == config["dataset_config"]["val_config"]["anno_path"]
        assert loaded_config["training_config"]["output"] == config["training_config"]["output"]
        assert loaded_config["training_config"]["exp_name"] == config["training_config"]["exp_name"]

    def test_yaml_roundtrip_anno_paths_not_none(self, tmp_path):
        """Test that anno_path and data_root values survive YAML round-trip as non-None."""
        _, yaml_path = _simulate_app_config_modification(tmp_path)

        args = argparse.Namespace(config=yaml_path, overrides=[])
        loaded_cfg = merge_configs(args)
        OmegaConf.resolve(loaded_cfg)
        loaded_config = OmegaConf.to_container(loaded_cfg, resolve=True)

        ds = loaded_config["dataset_config"]
        assert ds["train_config"]["anno_path"] is not None
        assert ds["val_config"]["anno_path"] is not None
        assert ds["train_config"]["data_root"] is not None
        assert ds["val_config"]["data_root"] is not None

    def test_yaml_roundtrip_resolution_type_preserved(self, tmp_path):
        """Test that resolution retains a numeric type after YAML round-trip."""
        _, yaml_path = _simulate_app_config_modification(tmp_path)

        args = argparse.Namespace(config=yaml_path, overrides=[])
        loaded_cfg = merge_configs(args)
        OmegaConf.resolve(loaded_cfg)
        loaded_config = OmegaConf.to_container(loaded_cfg, resolve=True)

        resolution = loaded_config["dataset_config"]["resolution"]
        assert isinstance(resolution, (int, list, tuple))
