#!/bin/bash
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

# Watchdog for API-polled fine-tuning jobs (DDM and VLM).
# Usage: watch_api_job.sh <PREFIX> <BASE_URL> <JOB_ID> <LOG_PATH> [TIMEOUT] [SLEEP]
#
#   PREFIX  DDM — emits DDM_DONE/FAILED/HANG/TIMEOUT, F1_UPDATE
#           VLM — emits VLM_DONE/FAILED/HANG/TIMEOUT, status=running heartbeats
#   TIMEOUT default 7200 s; SLEEP default 60 s (pass 120 for VLM)
PREFIX=$1; BASE_URL=$2; JOB_ID=$3; LOG=$4
TIMEOUT=${5:-7200}; SLEEP=${6:-60}
START=$(date +%s); LAST_PROGRESS=""; STUCK=0

while true; do
  STATUS=$(curl -s "$BASE_URL/api/v1/fine-tuning/status/$JOB_ID" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null)

  if [[ "$STATUS" == "completed" ]]; then echo "${PREFIX}_DONE:$STATUS"; exit 0; fi
  if [[ "$STATUS" == "failed" || "$STATUS" == "cancelled" ]]; then echo "${PREFIX}_FAILED:$STATUS"; exit 1; fi

  GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | head -1 | tr -d ' %')

  if [[ "$PREFIX" == "DDM" ]]; then
    # Epoch-level loss from Lightning progress bar — NOT the per-step API loss field.
    # The API loss reflects a single mini-batch and fluctuates far below the epoch average.
    EPOCH_LOSS=$(grep -oP "Epoch [0-9]+: 100%.*train/loss_epoch=\K[0-9.]+" "$LOG" 2>/dev/null | tail -1)
    F1=$(grep -oP "Epoch [0-9]+, global step [0-9]+: 'val/f1_score' reached \K[0-9.]+" "$LOG" 2>/dev/null | tail -1)
    PROGRESS="$EPOCH_LOSS"
    if [[ -n "$EPOCH_LOSS" && "$EPOCH_LOSS" != "$LAST_PROGRESS" ]]; then
      echo "F1_UPDATE:epoch_f1=$F1 epoch_loss=$EPOCH_LOSS"
    fi
  else
    PROGRESS=$(grep -c "step" "$LOG" 2>/dev/null)
    if [[ "$PROGRESS" != "$LAST_PROGRESS" ]]; then echo "status=running step=$PROGRESS"; fi
  fi

  if [[ "$PROGRESS" == "$LAST_PROGRESS" ]]; then ((STUCK++)); else STUCK=0; fi
  LAST_PROGRESS="$PROGRESS"
  if [[ "$GPU" == "0" && "$STUCK" -ge 5 ]]; then echo "${PREFIX}_HANG:gpu_idle_no_progress"; exit 2; fi
  if [[ $(( $(date +%s) - START )) -gt $TIMEOUT ]]; then echo "${PREFIX}_TIMEOUT"; exit 3; fi
  sleep "$SLEEP"
done
