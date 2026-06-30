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
#
# This file is based on DDM-Net (https://github.com/MCG-NJU/DDM),
# Copyright (c) 2021 Mike Zheng Shou, licensed under the MIT License.
# Modifications Copyright (c) NVIDIA CORPORATION & AFFILIATES.
######################################################################################################

"""
PyTorch Lightning 2.x Training Script for DDM (Dense Temporal Boundary Detection)

This version integrates the improved DDMDataset and DDMDataModule implementation,
with support for both YAML config files and command-line arguments.

All configuration logic is in config/config.py for better organization.
"""

import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import json
import numpy as np
import argparse
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback, TQDMProgressBar
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.plugins.environments import SLURMEnvironment

# Import modules
from utils.getter import getModel
from utils.metric import accuracy
from utils.visualize import visualize_scores_with_boundaries
from utils.model_ema import ModelEmaV2
from pl_ddm_datamodule import DDMDataModule
from config.config import get_config_with_cli, save_config, validate_config, load_config_file


class SOPLightningModule(L.LightningModule):
    """Lightning Module for SOP (Sequence of Procedures) boundary detection."""

    def __init__(
        self,
        model_name: str = "multiframes_resnet",
        backbone: str = "resnet50",
        num_classes: int = 2,
        frames_per_side: int = 5,
        pretrained: Optional[str] = None,
        freeze_backbone: bool = False,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0001,
        optimizer: str = "adamw",
        scheduler: str = "step",
        warmup_epochs: int = 2,
        decay_epochs: int = 30,
        decay_rate: float = 0.1,
        min_lr: float = 1e-5,
        warmup_lr: float = 0.0001,
        clip_grad: Optional[float] = None,
        clip_mode: str = "norm",
        eval_metric: str = "F1_score",
        val_anno_path: Optional[str] = None,
        save_visualizations: bool = True,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create model
        self.model = getModel(
            model_name=model_name,
            args=argparse.Namespace(
                model=model_name,
                backbone=backbone,
                num_classes=num_classes,
                frames_per_side=frames_per_side,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                **kwargs
            )
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Store predictions for F1 calculation
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input = batch['inp']
        target = batch['label']
        # path and current_ids are already flattened by DDMDataset.collate_fn
        batch_size = target.size(0)

        # Reshape input for multi-frame processing
        input = input.view((-1,) + input.size()[2:])
        target = target.view((-1,))

        # Forward pass
        outputs, rgbs, ddms = self.model(input)

        # Calculate loss for all outputs
        loss = 0
        for output in outputs:
            loss += self.criterion(output, target)
        for rgb in rgbs:
            loss += self.criterion(rgb, target)
        for ddm in ddms:
            loss += self.criterion(ddm, target)

        # Logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], on_step=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        input = batch['inp']
        target = batch['label']
        path = batch['path']  # Already flattened by DDMDataset.collate_fn
        current_ids = batch['current_ids']  # Already flattened by DDMDataset.collate_fn
        batch_size = target.size(0)

        # Reshape input and target to flatten batch dimension
        if input.ndim == 6:
            input = input.view((-1,) + input.size()[2:])
            target = target.view((-1,))
        elif input.ndim != 5:
            raise ValueError(f"Input tensor shape must be (B, 1/2, num_frames, C, H, W) or (B, num_frames, C, H, W), but got {input.shape}")

        # Forward pass
        outputs, rgbs, ddms = self.model(input)

        # Calculate loss
        loss = 0
        for output in outputs:
            loss += self.criterion(output, target)
        for rgb in rgbs:
            loss += self.criterion(rgb, target)
        for ddm in ddms:
            loss += self.criterion(ddm, target)

        # Get final output for metrics
        if isinstance(outputs, (tuple, list)):
            output = outputs[-1]
        else:
            output = outputs

        # Calculate accuracy
        acc1 = accuracy(output, target, topk=(1,))
        if isinstance(acc1, (list, tuple)):
            acc1 = acc1[0]

        # Get boundary scores
        bdy_scores = F.softmax(output, dim=1)[:, 1].cpu().numpy()

        # Store results for F1 calculation
        results = {
            'path': path,
            'current_ids': current_ids,
            'scores': bdy_scores,
            'loss': loss.item(),
            'acc': acc1.item()
        }
        self.validation_step_outputs.append(results)

        # Logging
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('val/acc', acc1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return results

    def on_validation_epoch_end(self):
        """Calculate F1 score and other metrics at the end of validation epoch."""
        if not self.validation_step_outputs:
            return

        # Gather predictions from THIS rank
        local_predictions = {}

        for output in self.validation_step_outputs:
            path = output['path']
            current_ids = output['current_ids']
            scores = output['scores']

            # Process predictions (data already flattened by collate_fn)
            for idx, (p, curr_id) in enumerate(zip(path, current_ids)):
                vname = p
                frame_idx = int(curr_id)
                score = float(scores[idx])

                if vname not in local_predictions:
                    local_predictions[vname] = {}
                local_predictions[vname][frame_idx] = score

        # Clear outputs
        self.validation_step_outputs.clear()

        # In DDP mode, we need to gather predictions from ALL ranks
        # Use torch.distributed.all_gather_object which preserves Python dict structure perfectly
        if self.trainer.world_size > 1:
            # Prepare list to receive gathered objects
            gathered_predictions = [None for _ in range(self.trainer.world_size)]

            # Gather the complete dict from each rank
            # all_gather_object uses pickle, so it preserves the exact structure
            dist.all_gather_object(gathered_predictions, local_predictions)

            # Merge all predictions (only rank 0 processes)
            if self.trainer.is_global_zero:
                all_predictions = {}
                for rank_idx, rank_preds in enumerate(gathered_predictions):
                    if rank_preds and isinstance(rank_preds, dict):
                        for vname, frames in rank_preds.items():
                            if vname not in all_predictions:
                                all_predictions[vname] = {}
                            # Merge frames from this rank
                            if isinstance(frames, dict):
                                for frame_idx, score in frames.items():
                                    if frame_idx in all_predictions[vname]:
                                        continue
                                    all_predictions[vname][frame_idx] = score
            else:
                all_predictions = {}
        else:
            # Single GPU mode
            all_predictions = local_predictions

        # Calculate F1 score (only on rank 0 to avoid redundant computation)
        prec, rec, f1 = 0.0, 0.0, 0.0  # Default values for all ranks

        if self.trainer.is_global_zero and all_predictions:
            # Sort predictions by frame index
            all_predictions, tmp_all_predictions = {}, all_predictions
            for vname, content in tmp_all_predictions.items():
                frame_indices = np.array(list(content.keys()))
                scores = np.array(list(content.values()))
                sorted_indices = np.argsort(frame_indices)
                frame_indices = frame_indices[sorted_indices]
                scores = scores[sorted_indices]
                all_predictions[vname] = {int(frame_idx): float(score) for frame_idx, score in zip(frame_indices, scores)}

            # Get val_anno_path from hparams or datamodule
            val_anno_path = self.hparams.get('val_anno_path')
            if val_anno_path is None and hasattr(self.trainer, 'datamodule'):
                val_anno_path = self.trainer.datamodule.dataset_config['val_config']['anno_path']

            if val_anno_path is not None:
                val_anno = json.load(open(val_anno_path, 'r'))

                # Get dataset from datamodule
                if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'val_dataset'):
                    dataset = self.trainer.datamodule.val_dataset
                    nms_result = self._prepare_nms_result(all_predictions, dataset, val_anno)
                    gt_dict = self._prepare_gt_dict(dataset, val_anno)

                    # Calculate F1 score
                    prec, rec, f1 = self._eval_f1(gt_dict, nms_result)

                    # Save visualizations if needed
                    if self.hparams.get('save_visualizations', True) and not self.trainer.sanity_checking:
                        try:
                            self._save_visualizations(all_predictions, nms_result, gt_dict, dataset)
                        except Exception as e:
                            print(f"Warning: Visualization failed: {e}")

        # IMPORTANT: Log metrics OUTSIDE the if block
        # Use rank_zero_only=True and sync_dist=False to avoid deadlock
        # Only rank 0 logs the real values, other ranks are automatically handled
        self.log('val/precision', prec, on_epoch=True, sync_dist=True, rank_zero_only=False, reduce_fx="sum")
        self.log('val/recall', rec, on_epoch=True, sync_dist=True, rank_zero_only=False, reduce_fx="sum")
        self.log('val/f1_score', f1, on_epoch=True, sync_dist=True, rank_zero_only=False, reduce_fx="sum")
        
        # CRITICAL: Ensure all ranks wait for rank 0's visualization to complete
        # This barrier ensures DDP is properly synchronized before next epoch
        if self.trainer.world_size > 1:
            dist.barrier()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler using PyTorch standard methods."""

        # === Optimizer Setup ===
        # Filter parameters: separate bias/BN (no weight decay) from other params (with weight decay)
        no_decay = []
        decay = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen parameters
            # Bias and 1D parameters (like BN) don't need weight decay
            if len(param.shape) == 1 or name.endswith('.bias'):
                no_decay.append(param)
            else:
                decay.append(param)

        parameters = [
            {'params': no_decay, 'weight_decay': 0.0},
            {'params': decay, 'weight_decay': self.hparams.weight_decay}
        ]

        # Create optimizer
        opt_name = self.hparams.optimizer.lower()
        opt_kwargs = {}
        if self.hparams.get('opt_betas') is not None:
            opt_kwargs['betas'] = self.hparams.opt_betas
        if self.hparams.get('opt_eps') is not None:
            opt_kwargs['eps'] = self.hparams.opt_eps

        if opt_name == 'adamw':
            optimizer = torch.optim.AdamW(
                parameters,
                lr=self.hparams.learning_rate,
                **opt_kwargs
            )
        elif opt_name == 'adam':
            optimizer = torch.optim.Adam(
                parameters,
                lr=self.hparams.learning_rate,
                **opt_kwargs
            )
        elif opt_name in ['sgd', 'momentum']:
            optimizer = torch.optim.SGD(
                parameters,
                lr=self.hparams.learning_rate,
                momentum=self.hparams.get('momentum', 0.9),
                nesterov=True
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        # === Scheduler Setup ===
        scheduler_name = self.hparams.scheduler.lower()

        if scheduler_name == 'cosine':
            # Cosine Annealing
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs - self.hparams.warmup_epochs,
                eta_min=self.hparams.min_lr
            )
        elif scheduler_name == 'step':
            # Step LR
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.decay_epochs,
                gamma=self.hparams.decay_rate
            )
        elif scheduler_name == 'plateau':
            # Reduce on Plateau
            mode = 'min' if 'loss' in self.hparams.eval_metric.lower() else 'max'
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=self.hparams.decay_rate,
                patience=self.hparams.get('patience_epochs', 10),
                min_lr=self.hparams.min_lr
            )
        else:
            # No scheduler
            return optimizer

        # === Warmup Setup (if needed) ===
        if self.hparams.warmup_epochs > 0:
            # Use SequentialLR to combine warmup with main scheduler
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.hparams.warmup_lr / self.hparams.learning_rate,
                end_factor=1.0,
                total_iters=self.hparams.warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.hparams.warmup_epochs]
            )

        # Configure scheduler
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }

        # For ReduceLROnPlateau, we need to monitor a metric
        if scheduler_name == 'plateau':
            scheduler_config['monitor'] = f'val/{self.hparams.eval_metric}'
            scheduler_config['strict'] = False

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_config
        }


    def _prepare_nms_result(self, predictions, dataset, val_anno):
        """Prepare NMS results from predictions."""
        nms_result = {}
        result = {}

        for vid, info in predictions.items():
            result_dict = {
                "frame_idx": [],
                "scores": []
            }

            for key in sorted(info.keys()):
                result_dict["frame_idx"].append(key)
                result_dict["scores"].append(info[key])
            result[vid] = result_dict

        for vid in result:
            if vid in val_anno:
                fps = dataset.video_info[vid]["fps"]
                duration = dataset.video_info[vid]["duration"]
                det_t = (
                    np.array(
                        self._get_idx_from_score_by_threshold(
                            scope=max(3, int(duration * fps * 0.025 / 2)),
                            threshold=0.5,
                            seq_indices=result[vid]["frame_idx"],
                            seq_scores=result[vid]["scores"],
                        )
                    )
                    / fps
                )
                nms_result[vid] = det_t.tolist()
        return nms_result

    def _prepare_gt_dict(self, dataset, val_anno):
        """Prepare ground truth dictionary."""
        gt_dict = {}
        for vid, content in val_anno.items():
            if vid in dataset.video_info:
                boundary_list = []
                content = [ct for ct in content if ct["description"] != "Final Segment"]
                for s_sample, e_sample in zip(content[:-1], content[1:]):
                    s_time = s_sample["end_timestamp"]
                    e_time = e_sample["start_timestamp"]
                    boundary_list.append((s_time + e_time) / 2)
                gt_dict[vid] = {
                    "fps": dataset.video_info[vid]["fps"],
                    "video_duration": dataset.video_info[vid]["duration"],
                    "f1_consis_avg": 1.0,
                    "f1_consis": [1.0],
                    "substages_timestamps": [boundary_list],
                }
        return gt_dict

    def _get_idx_from_score_by_threshold(self, scope=5, threshold=0.5, seq_indices=None, seq_scores=None):
        """Get boundary indices from scores by threshold."""
        seq_indices = np.array(seq_indices)
        seq_scores = np.array(seq_scores)
        bdy_indices_in_video = []

        for i in range(2, len(seq_scores) - 2):
            if seq_scores[i] >= threshold:
                sign = 1

                for j in range(max(0, i - scope), min(i + scope + 1, len(seq_scores))):
                    if seq_scores[j] > seq_scores[i]:
                        sign = 0
                    if seq_scores[j] == seq_scores[i] and i != j:
                        sign = 0

                if sign == 1:
                    bdy_indices_in_video.append(seq_indices[i])

        return bdy_indices_in_video

    def _eval_f1(self, gt_dict, pred_dict):
        """Evaluate F1 score."""
        threshold = 0.025
        tp_all = 0
        num_pos_all = 0
        num_det_all = 0

        for vid_id in list(gt_dict.keys()):
            if gt_dict[vid_id]["f1_consis_avg"] < 0.3:
                continue

            if vid_id not in pred_dict.keys():
                num_pos_all += len(gt_dict[vid_id]["substages_timestamps"][0])
                continue

            bdy_timestamps_det = pred_dict[vid_id]
            my_dur = gt_dict[vid_id]["video_duration"]
            ins_start = 0
            ins_end = my_dur

            # Remove detected boundaries outside the action instance
            tmp = []
            for det in bdy_timestamps_det:
                tmpdet = det + ins_start
                if tmpdet >= ins_start and tmpdet <= ins_end:
                    tmp.append(tmpdet)
            bdy_timestamps_det = tmp

            if not bdy_timestamps_det:
                num_pos_all += len(gt_dict[vid_id]["substages_timestamps"][0])
                continue

            num_det = len(bdy_timestamps_det)
            num_det_all += num_det

            # Compare with ground truth
            bdy_timestamps_list_gt_allraters = gt_dict[vid_id]["substages_timestamps"]
            f1_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
            tp_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
            num_pos_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))

            for ann_idx in range(len(bdy_timestamps_list_gt_allraters)):
                bdy_timestamps_list_gt = bdy_timestamps_list_gt_allraters[ann_idx]
                num_pos = len(bdy_timestamps_list_gt)
                tp = 0
                offset_arr = np.zeros((len(bdy_timestamps_list_gt), len(bdy_timestamps_det)))

                for ann1_idx in range(len(bdy_timestamps_list_gt)):
                    for ann2_idx in range(len(bdy_timestamps_det)):
                        offset_arr[ann1_idx, ann2_idx] = abs(
                            bdy_timestamps_list_gt[ann1_idx] - bdy_timestamps_det[ann2_idx]
                        )

                for ann1_idx in range(len(bdy_timestamps_list_gt)):
                    if offset_arr.shape[1] == 0:
                        break
                    min_idx = np.argmin(offset_arr[ann1_idx, :])
                    if offset_arr[ann1_idx, min_idx] <= threshold * my_dur:
                        tp += 1
                        offset_arr = np.delete(offset_arr, min_idx, 1)

                num_pos_tmplist[ann_idx] = num_pos
                fn = num_pos - tp
                fp = num_det - tp

                rec = 1 if num_pos == 0 else tp / (tp + fn)
                prec = 0 if (tp + fp) == 0 else tp / (tp + fp)
                f1 = 0 if (rec + prec) == 0 else 2 * rec * prec / (rec + prec)

                tp_tmplist[ann_idx] = tp
                f1_tmplist[ann_idx] = f1

            ann_best = np.argmax(f1_tmplist)
            tp_all += tp_tmplist[ann_best]
            num_pos_all += num_pos_tmplist[ann_best]

        fn_all = num_pos_all - tp_all
        fp_all = num_det_all - tp_all

        rec = 1 if num_pos_all == 0 else tp_all / (tp_all + fn_all)
        prec = 0 if (tp_all + fp_all) == 0 else tp_all / (tp_all + fp_all)
        f1 = 0 if (rec + prec) == 0 else 2 * rec * prec / (rec + prec)

        return prec, rec, f1

    def _save_visualizations(self, predictions, nms_result, gt_dict, dataset):
        """Save visualization images."""
        save_path = os.path.join(self.trainer.visualizations_save_folder, f"epoch_{self.current_epoch}_visualizations")
        os.makedirs(save_path, exist_ok=True)

        for k in tqdm.tqdm(nms_result, desc="visualizations"):
            try:
                save_name = os.path.join(save_path, k + ".png")
                visualize_scores_with_boundaries(
                    video_name=k,
                    golden_bdy=gt_dict[k]["substages_timestamps"][0],
                    pred_bdy=nms_result[k],
                    scores_dict=predictions[k],
                    fps=gt_dict[k]["fps"],
                    output_path=save_name
                )
            except Exception as e:
                print(f"Error saving visualization for {k}: {e}")
                continue


class EMACallback(Callback):
    """Callback for Exponential Moving Average of model weights."""

    def __init__(
        self,
        decay: float = 0.9998,
        start_epoch: int = 0,
        device: Optional[str] = None
    ):
        self.decay = decay
        self.start_epoch = start_epoch
        self.device = device
        self.ema_model = None
        self.active = False

    def on_fit_start(self, trainer, pl_module):
        """Initialize EMA model."""
        self.ema_model = ModelEmaV2(
            pl_module.model,
            decay=self.decay,
            device=self.device,
            delayed_start=(self.start_epoch > 0)
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update EMA model after each training batch."""
        if trainer.current_epoch >= self.start_epoch:
            if not self.active:
                self.ema_model.activate(pl_module.model)
                self.active = True
                print(f"EMA activated at epoch {trainer.current_epoch}")
            self.ema_model.update(pl_module.model)

    def on_validation_start(self, trainer, pl_module):
        """Swap model with EMA model for validation."""
        if self.active and self.ema_model is not None:
            self.model_backup = pl_module.model
            pl_module.model = self.ema_model.module

    def on_validation_end(self, trainer, pl_module):
        """Restore original model after validation."""
        if self.active and hasattr(self, 'model_backup'):
            pl_module.model = self.model_backup
            del self.model_backup


def main():
    """Main training function."""
    # Get configuration (handles YAML + CLI args merging)
    config, args = get_config_with_cli()

    # Extract configs for clarity
    dataset_cfg = config['dataset_config']
    model_cfg = config['model_config']
    train_cfg = config['training_config']

    # Set seed
    L.seed_everything(train_cfg['seed'], workers=True)

    # Create output directory
    output_dir = os.path.join(train_cfg['output'], "train", train_cfg['exp_name'])
    os.makedirs(output_dir, exist_ok=True)

    # Save complete config to output directory
    config_save_path = os.path.join(output_dir, 'config.yaml')
    save_config(config, config_save_path)
    print(f"Configuration saved to: {config_save_path}")

    # Initialize DataModule
    datamodule = DDMDataModule(dataset_config=dataset_cfg)

    # Initialize model
    model = SOPLightningModule(
        model_name=model_cfg['model_name'],
        backbone=model_cfg['backbone'],
        num_classes=model_cfg['num_classes'],
        frames_per_side=dataset_cfg['frames_per_side'],
        pretrained=model_cfg['pretrained'],
        freeze_backbone=model_cfg['freeze_backbone'],
        learning_rate=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
        optimizer=train_cfg['optimizer'],
        scheduler=train_cfg['scheduler'],
        warmup_epochs=train_cfg['warmup_epochs'],
        decay_epochs=train_cfg['decay_epochs'],
        decay_rate=train_cfg['decay_rate'],
        min_lr=train_cfg['min_lr'],
        warmup_lr=train_cfg['warmup_lr'],
        clip_grad=train_cfg['clip_grad'],
        clip_mode=train_cfg['clip_mode'],
        eval_metric=train_cfg['eval_metric'],
        val_anno_path=dataset_cfg['val_config']['anno_path'],
        save_visualizations=train_cfg['save_visualizations'],
        momentum=train_cfg['momentum'],
        opt_eps=train_cfg['opt_eps'],
        opt_betas=train_cfg['opt_betas'],
        resolution=dataset_cfg['resolution'],
    )

    # Callbacks
    callbacks = []

    # Force the tqdm progress bar. When `rich` is installed (now a transitive
    # dep of the ML stack), Lightning auto-selects RichProgressBar, whose output
    # is not emitted to the non-TTY pipe the training runs under and is not
    # parseable by the service's log parser (parse_ddm_log expects tqdm
    # "Epoch N: NN%|...train/loss_step=" lines). Explicitly using TQDMProgressBar
    # restores the per-step progress lines so the training-status bar advances.
    callbacks.append(TQDMProgressBar())

    # Model checkpoint
    checkpoint_monitor_metric = f"val/{train_cfg['eval_metric']}"
    checkpoint_mode = 'min' if 'loss' in train_cfg['eval_metric'].lower() else 'max'
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=f'epoch_{{epoch:03d}}-{{{checkpoint_monitor_metric}:.3f}}',
        monitor=checkpoint_monitor_metric,
        mode=checkpoint_mode,
        save_top_k=train_cfg['checkpoint_top_k'],
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # EMA callback
    if train_cfg['model_ema']:
        ema_callback = EMACallback(
            decay=train_cfg['model_ema_decay'],
            start_epoch=train_cfg['model_ema_start_epoch'],
        )
        callbacks.append(ema_callback)

    # Configure trainer strategy
    # For multi-GPU training with models that have unused parameters,
    # we need to use DDPStrategy with find_unused_parameters=True
    if train_cfg['num_gpus'] > 1:
        strategy = DDPStrategy(
            find_unused_parameters=True,  # Required for models with unused params
            static_graph=False,
            process_group_backend='nccl',  # Ensure NCCL backend
        )
    else:
        strategy = 'auto'  # Single GPU

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=train_cfg['epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=train_cfg['num_gpus'] if torch.cuda.is_available() else 1,
        num_nodes=train_cfg['num_nodes'],
        strategy=strategy,
        callbacks=callbacks,
        default_root_dir=output_dir,
        precision='16-mixed' if train_cfg['amp'] else '32-true',
        gradient_clip_val=train_cfg['clip_grad'] if train_cfg['clip_grad'] else None,
        gradient_clip_algorithm=train_cfg['clip_mode'],
        check_val_every_n_epoch=train_cfg['eval_freq'],
        log_every_n_steps=train_cfg.get('log_interval', 50),
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        benchmark=False,
        sync_batchnorm=True,
    )
    # Provide a folder for saving visualizations
    if train_cfg['save_visualizations']:
        visualizations_save_folder = os.path.join(output_dir, "visualizations")
        os.makedirs(visualizations_save_folder, exist_ok=True)
        trainer.visualizations_save_folder = visualizations_save_folder

    resume_path = os.path.join(output_dir, "last.ckpt")
    if os.path.exists(resume_path):
        ckpt_path = resume_path
    elif train_cfg['resume']:
        ckpt_path = train_cfg['resume']
    else:
        ckpt_path = None

    # Train model
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best {train_cfg['eval_metric']}: {checkpoint_callback.best_model_score:.4f}")
    print(f"Configuration: {config_save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

