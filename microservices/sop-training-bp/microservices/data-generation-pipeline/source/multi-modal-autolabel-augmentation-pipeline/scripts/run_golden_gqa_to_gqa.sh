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

export PYTHONPATH=.

# Set default value
EXT="mp4"
HUMAN_SUFFIX="<video>\n"
OUTPUT_NAME=golden_gqa
MIN_FRAMES=5
MAX_FRAMES=6
FRAMES_UPPERBOUND=-1
DYNAMIC_SAMPE=true
ACTION_JSON=""
GOLDEN_QA_ROOT=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --golden-qa-root) GOLDEN_QA_ROOT="$2"; shift ;;
        --action-json) ACTION_JSON="$2"; shift ;;
        --video-root) VIDEO_ROOT="$2"; shift ;;
        --video-ext) EXT="$2"; shift ;;
        --human-suffix) HUMAN_SUFFIX="$2"; shift ;;
        --output-root) OUTPUT_ROOT="$2"; shift ;;
        --output-name) OUTPUT_NAME="$2"; shift ;;
        --min_frames) MIN_FRAMES="$2"; shift ;;
        --max_frames) MAX_FRAMES="$2"; shift ;;
        --frames_upperbound) FRAMES_UPPERBOUND="$2"; shift ;;
        --dynamic_sample) DYNAMIC_SAMPE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

VIDEO_ROOT=${VIDEO_ROOT:?"--video-root is not set"}
OUTPUT_ROOT=${OUTPUT_ROOT:?"--output-root is not set"}


python autolabel_augmenting/golden_gqa_to_gqa.py \
        --golden-qa-root "${GOLDEN_QA_ROOT:-}" \
        --action-json "${ACTION_JSON:-}" \
        --video-root $VIDEO_ROOT \
        --ext $EXT \
        --human-suffix $HUMAN_SUFFIX \
        --output-root $OUTPUT_ROOT \
        --output-name $OUTPUT_NAME \
        --min_frames $MIN_FRAMES \
        --max_frames $MAX_FRAMES \
        --frames_upperbound $FRAMES_UPPERBOUND \
        --dynamic_sample $DYNAMIC_SAMPE
