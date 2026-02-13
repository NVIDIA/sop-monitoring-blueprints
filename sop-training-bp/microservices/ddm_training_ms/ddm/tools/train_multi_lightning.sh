#!/bin/bash

######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
######################################################################################################

# Training with YAML configuration file
# This is the recommended way - cleaner and easier to manage

# Note: Using full parameter names (--learning-rate, --optimizer, --scheduler)
# that match YAML keys. Short aliases (--lr, --opt, --sched) also work.

python DDM-Net/train_sop_lightning.py \
--config DDM-Net/config/sample.yaml \
--exp-name your_experiment_name \
--backbone resnet50 \
--output lightning_output \
--pretrained True  \
--learning-rate 0.0001 \
--min-lr 1e-10 \
--warmup-epochs 0 \
--epochs 30 \
--decay-epochs 2 \
--decay-rate 0.5 \
--model-ema \
--model-ema-decay 0.999 \
--model-ema-start-epoch 10 \
--eval-metric f1_score \
--num-workers 4 \
--num-gpus 1 \
--save-visualizations \

