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

# Start the RTSP server simulation for VSS SOP

set -euo pipefail

# Auto-detect and handle Docker permission issues if needed, but RTSP server runs natively on the host
BP_REPO="."
VIDEO_FILE=""

show_help() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -r, --bp-repo PATH         Path to the vss-sop repository (default: .)"
  echo "  -f, --video-file PATH      Path to the input video file (H.264, 30 FPS)"
  echo "  -h, --help                 Show this help message"
  return 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -r|--bp-repo)
      BP_REPO="$2"
      shift 2
      ;;
    -f|--video-file)
      VIDEO_FILE="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

BP_REPO=$(realpath "$BP_REPO")
WORKSPACE="$BP_REPO"

# Resolve default video file if not provided
if [[ -z "$VIDEO_FILE" ]]; then
  DEFAULT_VIDEO="$WORKSPACE/sop-resources/sop-server-fan-installation-data_v1.0-260213/server_fan/raw/Install_1_h264_30fps.mp4"
  if [[ -f "$DEFAULT_VIDEO" ]]; then
    VIDEO_FILE="$DEFAULT_VIDEO"
  else
    echo "⚠️ Warning: Default video file not found at $DEFAULT_VIDEO"
  fi
fi

if [[ -z "$VIDEO_FILE" ]] || [[ ! -f "$VIDEO_FILE" ]]; then
  echo "❌ Error: Video file not found. Please provide a valid file via --video-file PATH." >&2
  exit 1
fi

echo "=== Starting RTSP Server Simulation ==="
echo "Blueprint Repo: $BP_REPO"
echo "Video File:     $VIDEO_FILE"

# Start the RTSP Server
RTSP_TOOLS_DIR="$BP_REPO/deployments/sop/sop-app/helper-scripts/rtsp_tools"
if [[ ! -d "$RTSP_TOOLS_DIR" ]]; then
  echo "❌ Error: RTSP tools directory not found at $RTSP_TOOLS_DIR" >&2
  exit 1
fi

cd "$RTSP_TOOLS_DIR"

echo "Installing RTSP server prerequisites..."
./install.sh

echo "Starting RTSP server in background (nohup)..."
# Kill existing rtsp_server.py processes to avoid port conflicts
pkill -f rtsp_server.py || true
nohup python rtsp_server.py --filename "$VIDEO_FILE" --mount /sensor_0 > server.log 2>&1 &

echo "✅ RTSP server started. Logs are writing to $RTSP_TOOLS_DIR/server.log."

# Start the standalone RTSP output stream on :8554/ds-out/sensor_0 for VST to
# record. This replaces the in-pipeline RTSP output, which can hang the
# DeepStream/pyservicemaker inference pipeline on no-NVENC GPUs.
RELAY_SENSOR="${RELAY_SENSOR:-sensor_0}"
RELAY_PORT="${RELAY_PORT:-8554}"
sleep 2  # allow the prior relay process to release port ${RELAY_PORT} before rebinding
nohup python3 rtsp_server.py \
  --filename "$VIDEO_FILE" \
  --port "${RELAY_PORT}" \
  --mount "/ds-out/${RELAY_SENSOR}" \
  --mode overlay > relay.log 2>&1 &
echo "✅ RTSP output stream started on :${RELAY_PORT}/ds-out/${RELAY_SENSOR} (logs: $RTSP_TOOLS_DIR/relay.log)."
echo "=== RTSP Server Simulation Triggered Successfully! ==="

