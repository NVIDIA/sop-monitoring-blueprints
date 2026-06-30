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
LLM="meta/llama-3.1-8b-instruct"
NV_API_KEY=$API_KEY
NUM_QA_LLM=5

EXT="mp4"
HUMAN_SUFFIX="<video>\n"
NUM_QA_PER_CHUNK=1
OUTPUT_NAME=gqa
REPLACEMENT=true
MIN_FRAMES=5
MAX_FRAMES=6
FRAMES_UPPERBOUND=-1
DYNAMIC_SAMPE=true
ACTION_JSON=""
SAMPLE_QA_ROOT=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --llm) LLM="$2"; shift ;;
        --api-key) NV_API_KEY="$2"; shift ;;
        --sample-qa-root) SAMPLE_QA_ROOT="$2"; shift ;;
        --action-json) ACTION_JSON="$2"; shift ;;
        --num-qa-llm) NUM_QA_LLM="$2"; shift ;;
        --video-root) VIDEO_ROOT="$2"; shift ;;
        --video-ext) EXT="$2"; shift ;;
        --human-suffix) HUMAN_SUFFIX="$2"; shift ;;
        --num-qa-per-chunk) NUM_QA_PER_CHUNK="$2"; shift ;;
        --output-root) OUTPUT_ROOT="$2"; shift ;;
        --output-name) OUTPUT_NAME="$2"; shift ;;
        --replace) REPLACEMENT="$2"; shift ;;
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


python autolabel_augmenting/gqa_to_gqas.py \
        --llm $LLM \
        --api-key "${NV_API_KEY:-}" \
        --action-json "${ACTION_JSON:-}" \
        --sample-qa-root "${SAMPLE_QA_ROOT:-}" \
        --num-qa-llm $NUM_QA_LLM \
        --video-root $VIDEO_ROOT \
        --ext $EXT \
        --human-suffix $HUMAN_SUFFIX \
        --num-qa-per-chunk $NUM_QA_PER_CHUNK \
        --output-root $OUTPUT_ROOT \
        --output-name $OUTPUT_NAME \
        --replace $REPLACEMENT \
        --min_frames $MIN_FRAMES \
        --max_frames $MAX_FRAMES \
        --frames_upperbound $FRAMES_UPPERBOUND \
        --dynamic_sample $DYNAMIC_SAMPE
