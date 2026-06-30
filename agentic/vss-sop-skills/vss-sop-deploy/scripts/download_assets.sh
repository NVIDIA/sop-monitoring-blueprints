#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Download models and assets for VSS SOP deployment

set -euo pipefail

# Shared helper for pre-creating bind-mounted data_log directories
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/lib/setup_datalog_dirs.sh"

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/lib/resolve_bp_repo.sh"
resolve_bp_repo "$@"
WORKSPACE="$BP_REPO"

echo "=== Starting Asset Download & Preparation ==="
echo "Blueprint Repo: $BP_REPO"
echo "Workspace:      $WORKSPACE"

# Ensure NGC API Key is exported
if [[ -s "$BP_REPO/.secret/ngc_api_key.txt" ]]; then
  export NGC_CLI_API_KEY=$(cat "$BP_REPO/.secret/ngc_api_key.txt")
else
  echo "❌ Error: NGC API key file not found at $BP_REPO/.secret/ngc_api_key.txt" >&2
  exit 1
fi

# Create resource directory
mkdir -p "$WORKSPACE/sop-resources"
cd "$WORKSPACE/sop-resources"

# 1. Verify models and configs exist under /opt
echo "Verifying that trained model and config files exist..."
MISSING=0

if [[ ! -f "/opt/models/vlm/checkpoint/config.json" ]]; then
  echo "⚠️ Missing Cosmos Reason model config under /opt/models/vlm/checkpoint/config.json"
  MISSING=1
fi

if [[ ! -f "/opt/models/gbed_models/ddm/checkpoint.pth.tar" ]]; then
  echo "⚠️ Missing DDM checkpoint file under /opt/models/gbed_models/ddm/checkpoint.pth.tar"
  MISSING=1
fi

if [[ ! -f "/opt/sop/configs/actions.json" ]] || [[ ! -f "/opt/sop/configs/vlm_prompts.txt" ]]; then
  echo "⚠️ Missing SOP configs under /opt/sop/configs/"
  MISSING=1
fi

if [[ "$MISSING" -eq 1 ]]; then
  echo "❌ Error: Model and/or config files do not exist." >&2
  echo "For optimal accuracy, you must retrain/fine-tune the models, which can be done using the SOP Training Blueprint. After training, move model and config to \`/opt/models/...\` and \`/opt/sop/...\` directories." >&2
  exit 1
else
  echo "✅ All required model and config files verified successfully under /opt/models and /opt/sop."
fi

# 2. Download sample RTSP video
echo "Downloading sample training video..."
if [[ ! -d "sop-server-fan-installation-data_v1.0-260213" ]]; then
  ngc registry resource download-version "nvidia/tao/sop-server-fan-installation-data:1.0-260213"
  cd sop-server-fan-installation-data_v1.0-260213
  tar -xzf sop-sample-training-data.tar.gz
  cd ..
  echo "✅ Sample video downloaded."
else
  echo "✅ Sample video already exists, skipping download."
fi

# 3. Convert sample video to H.264 30 FPS
RTSP_SAMPLE_DIR="$WORKSPACE/sop-resources/sop-server-fan-installation-data_v1.0-260213/server_fan/raw"
RAW_VIDEO="$RTSP_SAMPLE_DIR/Install_1.MP4"
CONVERTED_VIDEO="$RTSP_SAMPLE_DIR/Install_1_h264_30fps.mp4"

if [[ ! -f "$RAW_VIDEO" ]]; then
  echo "❌ Error: Raw video not found at $RAW_VIDEO" >&2
  exit 1
fi

echo "Checking if ffmpeg is installed..."
if ! command -v ffmpeg &> /dev/null; then
  echo "Installing ffmpeg..."
  sudo apt-get update && sudo apt-get install -y ffmpeg
fi

if [[ ! -f "$CONVERTED_VIDEO" ]]; then
  echo "Converting sample video to H.264 at 30 FPS..."
  ffmpeg -y -i "$RAW_VIDEO" -c:v libx264 -r 30 -an "$CONVERTED_VIDEO"
  echo "✅ Video converted successfully."
else
  echo "✅ Converted video already exists, skipping."
fi

echo "Verifying converted video format..."
if command -v ffprobe &> /dev/null; then
  VERIFICATION=$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,r_frame_rate,width,height \
    -of csv=p=0 "$CONVERTED_VIDEO")
  echo "ffprobe result: $VERIFICATION"
else
  echo "⚠️ Warning: ffprobe not found. Cannot verify format."
fi

# 4. Model and config directories verification
echo "✅ Verification complete. Models and configurations are active under /opt/models and /opt/sop/configs."

# 5. Fix cache permissions
echo "Setting up ds-sop cache directory..."
mkdir -p "$HOME/.cache/ds-sop"
sudo chown -R 1001:1001 "$HOME/.cache/ds-sop"

# 6. Prepare datalog directory
echo "Preparing data log directory..."
MDX_DATA_DIR="$BP_REPO/sop-data"
setup_datalog_dirs "$MDX_DATA_DIR"

echo "=== Asset Preparation Complete! ==="
echo "RTSP_SAMPLE_VIDEO is located at: $CONVERTED_VIDEO"

