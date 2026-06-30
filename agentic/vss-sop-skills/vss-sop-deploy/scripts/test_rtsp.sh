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

# Simulate an RTSP stream and run the VSS SOP RTSP test

set -euo pipefail

# Auto-detect and handle Docker permission issues (run under sg docker or sudo if needed)
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/lib/ensure_docker_access.sh"
ensure_docker_access "$@"

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

echo "=== Starting RTSP Stream Simulation ==="
echo "Blueprint Repo: $BP_REPO"

# Verify if RTSP Server is already running, start it if not
if ! pgrep -f rtsp_server.py >/dev/null; then
  echo "⚠️ RTSP server (rtsp_server.py) is not running. Starting it..."
  SCRIPT_DIR="$(dirname "$(realpath "$0")")"
  if [[ -f "$SCRIPT_DIR/start_rtsp_server.sh" ]]; then
    RTSP_ARGS=(--bp-repo "$BP_REPO")
    [[ -n "$VIDEO_FILE" ]] && RTSP_ARGS+=(--video-file "$VIDEO_FILE")
    bash "$SCRIPT_DIR/start_rtsp_server.sh" "${RTSP_ARGS[@]}"
  else
    echo "❌ Error: start_rtsp_server.sh not found at $SCRIPT_DIR/start_rtsp_server.sh" >&2
    exit 1
  fi
else
  echo "✅ RTSP server is already running."
fi

# 2. Wait for DS-SOP API server to be ready
echo "Waiting for mdx-ds-sop-1 API server readiness..."
MAX_ATTEMPTS=60
ATTEMPT=0
READY=false

while [[ $ATTEMPT -lt $MAX_ATTEMPTS ]]; do
  if docker logs mdx-ds-sop-1 2>&1 | grep -qE "Uvicorn running on|Application startup complete"; then
    echo "✅ API server is ready!"
    READY=true
    break
  fi
  echo "Waiting for API server... ($((ATTEMPT + 1))/$MAX_ATTEMPTS)"
  sleep 5
  ATTEMPT=$((ATTEMPT + 1))
done

if [[ "$READY" = "false" ]]; then
  echo "⚠️ Warning: Timeout waiting for API server readiness (300s)."
  echo "   We will attempt to trigger the test client anyway, but it may fail."
fi

# 3. Run the RTSP client test
HELPER_DIR="$BP_REPO/deployments/sop/sop-app/helper-scripts"
if [[ ! -d "$HELPER_DIR" ]]; then
  echo "❌ Error: Helper scripts directory not found at $HELPER_DIR" >&2
  exit 1
fi

cd "$HELPER_DIR"

echo "Starting RTSP client test in background (nohup)..."
# Kill existing run_rtsp_test.sh or similar test processes
pkill -f run_rtsp_test.sh || true
nohup ./run_rtsp_test.sh > client.log 2>&1 &

echo "RTSP client test started. Logs are writing to $HELPER_DIR/client.log."
echo "=== RTSP Simulation Triggered Successfully! ==="

