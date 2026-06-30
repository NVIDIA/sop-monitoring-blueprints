#!/bin/bash

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

