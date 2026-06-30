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
Unit tests for SOPLightningModule in train_sop_lightning.py.
Covers NMS boundary selection, F1 evaluation, GT preparation, and optimizer/scheduler configuration.
All heavy ML dependencies are mocked; numpy is used directly for real array computation.
"""

import sys
from unittest.mock import MagicMock

# Mock heavy ML packages so tests run without GPU or installed frameworks.
_ML_MODULES = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.distributed",
    "lightning",
    "lightning.pytorch",
    "lightning.pytorch.callbacks",
    "lightning.pytorch.strategies",
    "lightning.pytorch.plugins",
    "lightning.pytorch.plugins.environments",
    "tqdm",
    "utils.getter",
    "utils.metric",
    "utils.visualize",
    "utils.model_ema",
    "pl_ddm_datamodule",
]
for _mod in _ML_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# LightningModule must be a real Python class so SOPLightningModule can be instantiated via __new__.
sys.modules["lightning"].LightningModule = type("LightningModule", (), {})

import numpy as np
import pytest

from train_sop_lightning import SOPLightningModule


def _make_module():
    """Create a bare SOPLightningModule instance by bypassing __init__."""
    return SOPLightningModule.__new__(SOPLightningModule)


class TestGetIdxFromScoreByThreshold:
    """Tests for _get_idx_from_score_by_threshold."""

    def test_basic_local_maximum_selected(self):
        """Test that a clear local maximum above threshold is selected."""
        module = _make_module()
        indices = [0, 1, 2, 3, 4]
        scores = [0.1, 0.1, 0.9, 0.1, 0.1]

        result = module._get_idx_from_score_by_threshold(
            scope=1, threshold=0.5, seq_indices=indices, seq_scores=scores
        )

        assert 2 in result

    def test_score_below_threshold_not_selected(self):
        """Test that no index is selected when all scores are below threshold."""
        module = _make_module()
        indices = list(range(10))
        scores = [0.1] * 10

        result = module._get_idx_from_score_by_threshold(
            scope=2, threshold=0.5, seq_indices=indices, seq_scores=scores
        )

        assert result == []

    def test_not_local_maximum_not_selected(self):
        """Test that a score above threshold is not selected when a neighbor is higher."""
        module = _make_module()
        indices = [0, 1, 2, 3, 4]
        scores = [0.1, 0.1, 0.7, 0.9, 0.1]

        result = module._get_idx_from_score_by_threshold(
            scope=2, threshold=0.5, seq_indices=indices, seq_scores=scores
        )

        assert 2 not in result

    def test_tie_score_not_selected(self):
        """Test that neither frame is selected when two adjacent frames share the highest score."""
        module = _make_module()
        indices = [0, 1, 2, 3, 4]
        scores = [0.1, 0.1, 0.9, 0.9, 0.1]

        result = module._get_idx_from_score_by_threshold(
            scope=1, threshold=0.5, seq_indices=indices, seq_scores=scores
        )

        assert 2 not in result
        assert 3 not in result

    def test_too_short_sequence_returns_empty(self):
        """Test that a sequence of 4 or fewer frames yields no selected indices."""
        module = _make_module()
        result = module._get_idx_from_score_by_threshold(
            scope=1, threshold=0.5,
            seq_indices=[0, 1, 2, 3],
            seq_scores=[0.9, 0.9, 0.9, 0.9],
        )
        assert result == []

    def test_multiple_boundaries_all_selected(self):
        """Test that multiple well-separated local maxima are all selected."""
        module = _make_module()
        indices = list(range(10))
        scores = [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1]

        result = module._get_idx_from_score_by_threshold(
            scope=1, threshold=0.5, seq_indices=indices, seq_scores=scores
        )

        # index 1 falls outside range(2, 8), so only index 7 is eligible.
        assert 7 in result

    def test_scope_suppresses_nearby_secondary_peaks(self):
        """Test that a lower peak within scope of a higher peak is suppressed."""
        module = _make_module()
        indices = list(range(8))
        scores = [0.1, 0.1, 0.1, 0.9, 0.1, 0.6, 0.1, 0.1]

        result = module._get_idx_from_score_by_threshold(
            scope=3, threshold=0.5, seq_indices=indices, seq_scores=scores
        )

        assert 3 in result
        assert 5 not in result

    def test_scope_zero_selects_all_above_threshold(self):
        """Test that scope=0 disables NMS and selects all frames above threshold."""
        module = _make_module()
        indices = list(range(8))
        scores = [0.1, 0.1, 0.9, 0.1, 0.8, 0.1, 0.1, 0.1]

        result = module._get_idx_from_score_by_threshold(
            scope=0, threshold=0.5, seq_indices=indices, seq_scores=scores
        )

        assert 2 in result
        assert 4 in result


class TestEvalF1:
    """Tests for _eval_f1."""

    def _make_gt(self, video_id, boundaries, duration, quality=1.0):
        """Build a ground-truth dict for a single video."""
        return {
            video_id: {
                "f1_consis_avg": quality,
                "substages_timestamps": [boundaries],
                "video_duration": duration,
            }
        }

    def test_perfect_prediction_gives_f1_one(self):
        """Test that an exact boundary match yields F1, precision, and recall all equal to 1.0."""
        module = _make_module()
        gt = self._make_gt("v1", boundaries=[5.0], duration=10.0)
        pred = {"v1": [5.0]}

        prec, rec, f1 = module._eval_f1(gt, pred)

        assert f1 == pytest.approx(1.0)
        assert prec == pytest.approx(1.0)
        assert rec == pytest.approx(1.0)

    def test_no_prediction_gives_f1_zero(self):
        """Test that an empty prediction list yields F1 of zero."""
        module = _make_module()
        gt = self._make_gt("v1", boundaries=[5.0], duration=10.0)
        pred = {"v1": []}

        _, _, f1 = module._eval_f1(gt, pred)

        assert f1 == pytest.approx(0.0)

    def test_prediction_within_tolerance_counts_as_tp(self):
        """Test that a prediction within 2.5% of video duration is counted as a true positive."""
        module = _make_module()
        gt = self._make_gt("v1", boundaries=[50.0], duration=100.0)
        pred = {"v1": [51.0]}  # 1s offset, tolerance is 2.5s

        _, _, f1 = module._eval_f1(gt, pred)

        assert f1 == pytest.approx(1.0)

    def test_prediction_outside_tolerance_is_false_positive(self):
        """Test that a prediction outside the tolerance window is treated as a false positive."""
        module = _make_module()
        gt = self._make_gt("v1", boundaries=[5.0], duration=10.0)
        pred = {"v1": [9.0]}  # 4s offset, tolerance is 0.25s

        _, _, f1 = module._eval_f1(gt, pred)

        assert f1 == pytest.approx(0.0)

    def test_low_quality_annotation_is_skipped(self):
        """Test that videos with f1_consis_avg below 0.3 are excluded from evaluation."""
        module = _make_module()
        gt = self._make_gt("v1", boundaries=[5.0], duration=10.0, quality=0.2)
        pred = {"v1": []}

        prec, rec, f1 = module._eval_f1(gt, pred)

        # Skipped video leaves num_pos_all=0: recall defaults to 1, precision and F1 to 0.
        assert rec == pytest.approx(1.0)
        assert prec == pytest.approx(0.0)
        assert f1 == pytest.approx(0.0)

    def test_video_not_in_pred_dict_counts_as_miss(self):
        """Test that a GT video absent from the prediction dict causes all its boundaries to be missed."""
        module = _make_module()
        gt = self._make_gt("v1", boundaries=[5.0], duration=10.0)
        pred = {}

        _, _, f1 = module._eval_f1(gt, pred)

        assert f1 == pytest.approx(0.0)

    def test_empty_gt_dict(self):
        """Test that an empty GT dict yields recall=1, precision=0, and F1=0."""
        module = _make_module()
        prec, rec, f1 = module._eval_f1({}, {})

        assert rec == pytest.approx(1.0)
        assert prec == pytest.approx(0.0)
        assert f1 == pytest.approx(0.0)

    def test_multiple_videos_aggregated(self):
        """Test that TP counts are aggregated across videos before computing F1."""
        module = _make_module()
        gt = {}
        gt.update(self._make_gt("v1", [5.0], 10.0))
        gt.update(self._make_gt("v2", [3.0], 10.0))
        pred = {"v1": [5.0], "v2": [3.0]}

        _, _, f1 = module._eval_f1(gt, pred)

        assert f1 == pytest.approx(1.0)

    def test_multiple_boundaries_partial_match(self):
        """Test that hitting one of two GT boundaries gives precision=1.0, recall=0.5."""
        module = _make_module()
        gt = self._make_gt("v1", boundaries=[5.0, 8.0], duration=20.0)
        pred = {"v1": [5.0]}

        prec, rec, f1 = module._eval_f1(gt, pred)

        assert prec == pytest.approx(1.0)
        assert rec == pytest.approx(0.5)
        assert f1 == pytest.approx(2 * 1.0 * 0.5 / (1.0 + 0.5))

    def test_false_positives_reduce_precision(self):
        """Test that one TP and one FP yields precision=0.5 and recall=1.0."""
        module = _make_module()
        gt = self._make_gt("v1", boundaries=[5.0], duration=20.0)
        pred = {"v1": [5.0, 15.0]}

        prec, rec, f1 = module._eval_f1(gt, pred)

        assert prec == pytest.approx(0.5)
        assert rec == pytest.approx(1.0)
        assert f1 == pytest.approx(2 * 0.5 * 1.0 / (0.5 + 1.0))

    def test_prediction_outside_video_duration_excluded(self):
        """Test that predictions beyond video duration are filtered out before scoring."""
        module = _make_module()
        gt = self._make_gt("v1", boundaries=[5.0], duration=10.0)
        pred = {"v1": [5.0, 99.0]}  # 99.0 exceeds duration=10.0

        prec, _, f1 = module._eval_f1(gt, pred)

        assert f1 == pytest.approx(1.0)
        assert prec == pytest.approx(1.0)


class TestPrepareGtDict:
    """Tests for _prepare_gt_dict."""

    def _make_dataset_mock(self, video_ids_and_info):
        """Build a mock dataset with a video_info attribute."""
        mock_dataset = MagicMock()
        mock_dataset.video_info = video_ids_and_info
        return mock_dataset

    def test_boundary_calculated_as_midpoint(self):
        """Test that boundary timestamp is the midpoint between adjacent segment endpoints."""
        module = _make_module()
        dataset = self._make_dataset_mock({"v1": {"fps": 30, "duration": 10.0}})
        val_anno = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 3.0},
                {"description": "Step 2", "start_timestamp": 4.0, "end_timestamp": 10.0},
                {"description": "Final Segment", "start_timestamp": 10.0, "end_timestamp": 10.0},
            ]
        }

        result = module._prepare_gt_dict(dataset, val_anno)

        # boundary = (3.0 + 4.0) / 2 = 3.5 between Step 1 and Step 2
        assert "v1" in result
        assert pytest.approx(3.5) in result["v1"]["substages_timestamps"][0]

    def test_final_segment_excluded(self):
        """Test that segments with description 'Final Segment' are excluded from boundary calculation."""
        module = _make_module()
        dataset = self._make_dataset_mock({"v1": {"fps": 30, "duration": 10.0}})
        val_anno = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 10.0},
                {"description": "Final Segment", "start_timestamp": 10.0, "end_timestamp": 10.0},
            ]
        }

        result = module._prepare_gt_dict(dataset, val_anno)

        assert result["v1"]["substages_timestamps"][0] == []

    def test_video_not_in_dataset_excluded(self):
        """Test that a video present in annotations but absent from dataset.video_info is skipped."""
        module = _make_module()
        dataset = self._make_dataset_mock({})
        val_anno = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 3.0},
            ]
        }

        result = module._prepare_gt_dict(dataset, val_anno)

        assert "v1" not in result

    def test_multiple_boundaries_all_included(self):
        """Test that all boundaries between non-final segments are included in the result."""
        module = _make_module()
        dataset = self._make_dataset_mock({"v1": {"fps": 30, "duration": 20.0}})
        val_anno = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 3.0},
                {"description": "Step 2", "start_timestamp": 4.0, "end_timestamp": 7.0},
                {"description": "Step 3", "start_timestamp": 10.0, "end_timestamp": 20.0},
                {"description": "Final Segment", "start_timestamp": 20.0, "end_timestamp": 20.0},
            ]
        }

        result = module._prepare_gt_dict(dataset, val_anno)

        boundaries = result["v1"]["substages_timestamps"][0]
        assert len(boundaries) == 2
        assert pytest.approx(3.5) in boundaries   # (3.0 + 4.0) / 2
        assert pytest.approx(8.5) in boundaries   # (7.0 + 10.0) / 2

    def test_gt_dict_contains_required_fields(self):
        """Test that each gt_dict entry contains fps, video_duration, f1_consis_avg, and substages_timestamps."""
        module = _make_module()
        dataset = self._make_dataset_mock({"v1": {"fps": 25, "duration": 15.0}})
        val_anno = {
            "v1": [
                {"description": "Step 1", "start_timestamp": 0.0, "end_timestamp": 5.0},
            ]
        }

        result = module._prepare_gt_dict(dataset, val_anno)

        assert "fps" in result["v1"]
        assert "video_duration" in result["v1"]
        assert "f1_consis_avg" in result["v1"]
        assert "substages_timestamps" in result["v1"]
        assert result["v1"]["fps"] == 25
        assert result["v1"]["video_duration"] == pytest.approx(15.0)




class TestConfigureOptimizers:
    """Tests for configure_optimizers."""

    def _make_module_with_hparams(self, optimizer="adamw", scheduler="step", **kwargs):
        """Build a module with mocked model parameters and hparams."""
        module = _make_module()
        module.model = MagicMock()
        fake_param = MagicMock()
        fake_param.requires_grad = True
        fake_param.shape = [512, 256]  # 2D shape goes into the weight-decay group
        module.model.named_parameters.return_value = [("weight", fake_param)]

        module.hparams = MagicMock()
        module.hparams.optimizer = optimizer
        module.hparams.scheduler = scheduler
        module.hparams.learning_rate = 0.01
        module.hparams.weight_decay = 0.0001
        module.hparams.momentum = 0.9
        module.hparams.warmup_epochs = 0
        module.hparams.get.return_value = None
        for key, val in kwargs.items():
            setattr(module.hparams, key, val)
        return module

    def test_unsupported_optimizer_raises_value_error(self):
        """Test that an unrecognized optimizer name raises ValueError immediately."""
        module = self._make_module_with_hparams(optimizer="invalid_opt")
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            module.configure_optimizers()

    def test_adamw_optimizer_selected(self):
        """Test that optimizer='adamw' calls torch.optim.AdamW."""
        import torch
        module = self._make_module_with_hparams(optimizer="adamw")

        module.configure_optimizers()

        torch.optim.AdamW.assert_called_once()

    def test_sgd_optimizer_selected(self):
        """Test that optimizer='sgd' calls torch.optim.SGD."""
        import torch
        module = self._make_module_with_hparams(optimizer="sgd")

        module.configure_optimizers()

        torch.optim.SGD.assert_called_once()

    def test_unknown_scheduler_returns_optimizer_only(self):
        """Test that an unrecognized scheduler name returns the optimizer directly, not a dict."""
        module = self._make_module_with_hparams(
            optimizer="adamw",
            scheduler="unknown_scheduler",
        )
        result = module.configure_optimizers()

        assert not isinstance(result, dict)

    def test_adam_optimizer_selected(self):
        """Test that optimizer='adam' calls torch.optim.Adam."""
        import torch
        torch.optim.Adam.reset_mock()
        module = self._make_module_with_hparams(optimizer="adam")

        module.configure_optimizers()

        torch.optim.Adam.assert_called_once()

    def test_momentum_optimizer_is_sgd(self):
        """Test that optimizer='momentum' calls torch.optim.SGD."""
        import torch
        torch.optim.SGD.reset_mock()
        module = self._make_module_with_hparams(optimizer="momentum")

        module.configure_optimizers()

        torch.optim.SGD.assert_called_once()

    def test_case_insensitive_optimizer_name(self):
        """Test that optimizer names are matched case-insensitively."""
        import torch
        torch.optim.AdamW.reset_mock()
        module = self._make_module_with_hparams(optimizer="ADAMW")

        module.configure_optimizers()

        torch.optim.AdamW.assert_called_once()

    def test_1d_param_goes_to_no_decay_group(self):
        """Test that 1D parameters are placed in the no-decay group and 2D parameters in the decay group."""
        module = _make_module()
        module.model = MagicMock()

        fake_1d = MagicMock()
        fake_1d.requires_grad = True
        fake_1d.shape = [512]  # 1D (e.g. BN weight)

        fake_2d = MagicMock()
        fake_2d.requires_grad = True
        fake_2d.shape = [512, 256]  # 2D (e.g. linear weight)

        module.model.named_parameters.return_value = [
            ("bn.weight", fake_1d),
            ("fc.weight", fake_2d),
        ]
        module.hparams = MagicMock()
        module.hparams.optimizer = "adamw"
        module.hparams.scheduler = "unknown"
        module.hparams.learning_rate = 0.01
        module.hparams.weight_decay = 0.0001
        module.hparams.warmup_epochs = 0
        module.hparams.get.return_value = None

        import torch
        torch.optim.AdamW.reset_mock()
        module.configure_optimizers()

        call_args = torch.optim.AdamW.call_args
        param_groups = call_args.args[0] if call_args.args else call_args[0][0]

        no_decay_group = param_groups[0]
        decay_group = param_groups[1]

        assert no_decay_group["weight_decay"] == pytest.approx(0.0)
        assert fake_1d in no_decay_group["params"]

        assert decay_group["weight_decay"] == pytest.approx(0.0001)
        assert fake_2d in decay_group["params"]

    def test_bias_param_name_goes_to_no_decay_group(self):
        """Test that a parameter whose name ends with '.bias' is placed in the no-decay group regardless of shape."""
        module = _make_module()
        module.model = MagicMock()

        fake_bias = MagicMock()
        fake_bias.requires_grad = True
        fake_bias.shape = [512, 2]  # 2D but named .bias

        module.model.named_parameters.return_value = [
            ("layer.bias", fake_bias),
        ]
        module.hparams = MagicMock()
        module.hparams.optimizer = "adamw"
        module.hparams.scheduler = "unknown"
        module.hparams.learning_rate = 0.01
        module.hparams.weight_decay = 0.0001
        module.hparams.warmup_epochs = 0
        module.hparams.get.return_value = None

        import torch
        torch.optim.AdamW.reset_mock()
        module.configure_optimizers()

        call_args = torch.optim.AdamW.call_args
        param_groups = call_args.args[0] if call_args.args else call_args[0][0]
        no_decay_group = param_groups[0]

        assert fake_bias in no_decay_group["params"]
        assert no_decay_group["weight_decay"] == pytest.approx(0.0)

    def test_step_scheduler_returns_dict_with_scheduler_key(self):
        """Test that scheduler='step' returns a dict with 'optimizer' and 'lr_scheduler' keys."""
        module = self._make_module_with_hparams(optimizer="adamw", scheduler="step")
        module.hparams.decay_epochs = 10
        module.hparams.decay_rate = 0.1

        result = module.configure_optimizers()

        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert result["lr_scheduler"]["interval"] == "epoch"

    def test_cosine_scheduler_returns_dict(self):
        """Test that scheduler='cosine' returns a dict with an 'lr_scheduler' key."""
        module = self._make_module_with_hparams(optimizer="adamw", scheduler="cosine")
        module.hparams.min_lr = 1e-6
        module.trainer = MagicMock()
        module.trainer.max_epochs = 30

        result = module.configure_optimizers()

        assert isinstance(result, dict)
        assert "lr_scheduler" in result

    def test_plateau_scheduler_returns_dict_with_monitor_key(self):
        """Test that scheduler='plateau' returns a dict whose lr_scheduler config includes a monitor key."""
        module = self._make_module_with_hparams(optimizer="adamw", scheduler="plateau")
        module.hparams.eval_metric = "f1_score"
        module.hparams.decay_rate = 0.1
        module.hparams.min_lr = 1e-6
        module.hparams.get = MagicMock(side_effect=lambda key, *a: 10 if key == "patience_epochs" else None)

        result = module.configure_optimizers()

        assert isinstance(result, dict)
        assert result["lr_scheduler"]["monitor"] == "val/f1_score"
        assert "strict" in result["lr_scheduler"]

    def test_frozen_params_excluded_from_all_param_groups(self):
        """Test that parameters with requires_grad=False are not placed in any optimizer param group."""
        module = _make_module()
        module.model = MagicMock()

        frozen_param = MagicMock()
        frozen_param.requires_grad = False
        frozen_param.shape = [512, 512]

        trainable_param = MagicMock()
        trainable_param.requires_grad = True
        trainable_param.shape = [512, 512]

        module.model.named_parameters.return_value = [
            ("layer.frozen_weight", frozen_param),
            ("layer.trainable_weight", trainable_param),
        ]
        module.hparams = MagicMock()
        module.hparams.optimizer = "adamw"
        module.hparams.scheduler = "unknown"
        module.hparams.learning_rate = 0.01
        module.hparams.weight_decay = 0.0001
        module.hparams.warmup_epochs = 0
        module.hparams.get.return_value = None

        import torch
        torch.optim.AdamW.reset_mock()
        module.configure_optimizers()

        call_args = torch.optim.AdamW.call_args
        param_groups = call_args.args[0] if call_args.args else call_args[0][0]

        all_params = []
        for group in param_groups:
            all_params.extend(group["params"])

        assert frozen_param not in all_params
        assert trainable_param in all_params
