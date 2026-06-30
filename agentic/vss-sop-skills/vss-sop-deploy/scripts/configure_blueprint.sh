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

# Configure deployments/sop/.env for VSS SOP deployment

set -euo pipefail

DEFAULT_API_URL="https://integrate.api.nvidia.com"
BP_REPO="."
LLM_MODE="remote"
VLM_MODE="remote"
LLM_BASE_URL="$DEFAULT_API_URL"
VLM_BASE_URL="$DEFAULT_API_URL"
LLM_NAME="nvidia/llama-3.3-nemotron-super-49b-v1.5"
VLM_NAME="ds_sop_model"

show_help() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -r, --bp-repo PATH         Path to the vss-sop repository (default: .)"
  echo "  --llm-mode MODE            LLM Mode: 'local', 'local_shared', or 'remote' (default: remote)"
  echo "  --vlm-mode MODE            VLM Mode: 'local', 'local_shared', or 'remote' (default: remote)"
  echo "  --llm-base-url URL         LLM Base URL (default: $DEFAULT_API_URL for remote)"
  echo "  --vlm-base-url URL         VLM Base URL (default: $DEFAULT_API_URL for remote, http://localhost:8300 for local)"
  echo "  --llm-name NAME            LLM Model Name (default: nvidia/llama-3.3-nemotron-super-49b-v1.5)"
  echo "  --vlm-name NAME            VLM Model Name (default: ds_sop_model for local)"
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
    --llm-mode)
      LLM_MODE="$2"
      shift 2
      ;;
    --vlm-mode)
      VLM_MODE="$2"
      shift 2
      ;;
    --llm-base-url)
      LLM_BASE_URL="$2"
      shift 2
      ;;
    --vlm-base-url)
      VLM_BASE_URL="$2"
      shift 2
      ;;
    --llm-name)
      LLM_NAME="$2"
      shift 2
      ;;
    --vlm-name)
      VLM_NAME="$2"
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
ENV_FILE="$BP_REPO/deployments/sop/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "❌ Error: .env file not found at $ENV_FILE" >&2
  exit 1
fi

echo "=== Starting Blueprint Configuration ==="
echo "Blueprint Repo: $BP_REPO"
echo "Env File:       $ENV_FILE"

# 1. Detect HOST_IP
HOST_IP=$(hostname -I | tr ' ' '\n' | grep -v '^169\.254\.' | head -1 || echo "127.0.0.1")
echo "Detected HOST_IP: $HOST_IP"

# 2. Detect EXTERNAL_IP
EXTERNAL_IP=$(curl -s icanhazip.com || echo "")
if [[ -z "${EXTERNAL_IP}" ]]; then
  EXTERNAL_IP="${HOST_IP}"
else
  echo "Performing health check on EXTERNAL_IP: $EXTERNAL_IP using ping..."
  if ! ping -c 2 -W 2 "${EXTERNAL_IP}" >/dev/null 2>&1; then
    echo "⚠️ Ping health check failed for EXTERNAL_IP $EXTERNAL_IP. Falling back to HOST_IP $HOST_IP"
    EXTERNAL_IP="${HOST_IP}"
  else
    echo "✅ Ping health check succeeded for EXTERNAL_IP $EXTERNAL_IP"
  fi
fi
echo "Detected EXTERNAL_IP: $EXTERNAL_IP"

# 3. Update paths and network config
REAL_DEPLOY_DIR=$(realpath "$BP_REPO/deployments")
REAL_DATA_DIR=$(realpath "$BP_REPO/sop-data")

# Backup the original .env
cp "$ENV_FILE" "${ENV_FILE}.bak"
echo "Backup of .env created at ${ENV_FILE}.bak"

# Perform substitutions
sed -i "s|^MDX_SAMPLE_APPS_DIR=.*|MDX_SAMPLE_APPS_DIR=\"$REAL_DEPLOY_DIR\"|" "$ENV_FILE"
sed -i "s|^MDX_DATA_DIR=.*|MDX_DATA_DIR=\"$REAL_DATA_DIR\"|" "$ENV_FILE"
sed -i "s|^HOST_IP=.*|HOST_IP='$HOST_IP'|" "$ENV_FILE"
sed -i "s|^EXTERNAL_IP=.*|EXTERNAL_IP='$EXTERNAL_IP'|" "$ENV_FILE"

# 4. LLM / VLM mode configuration

# Resolve LLM defaults based on mode if not explicitly overridden by user
if [[ ( "$LLM_MODE" = "local" || "$LLM_MODE" = "local_shared" ) && "$LLM_BASE_URL" = "$DEFAULT_API_URL" ]]; then
  # For local LLM modes, if LLM_BASE_URL is still the default remote value, set it to empty
  # so that vss-agent docker compose falls back to local LLM port.
  LLM_BASE_URL=""
fi

sed -i "s|^LLM_MODE=.*|LLM_MODE=$LLM_MODE|" "$ENV_FILE"
sed -i "s|^LLM_BASE_URL=.*|LLM_BASE_URL='$LLM_BASE_URL'|" "$ENV_FILE"
sed -i "s|^LLM_NAME=.*|LLM_NAME=$LLM_NAME|" "$ENV_FILE"

LLM_NAME_SLUG=$(echo "$LLM_NAME" | awk -F/ '{print $NF}')
sed -i "s|^LLM_NAME_SLUG=.*|LLM_NAME_SLUG=$LLM_NAME_SLUG|" "$ENV_FILE"


# Resolve VLM defaults based on mode if not explicitly overridden by user
if [[ "$VLM_MODE" = "local" ]]; then
  # SOP Local VLM uses the local DS-SOP container's vLLM on port 8300
  if [[ "$VLM_BASE_URL" = "$DEFAULT_API_URL" ]]; then
    VLM_BASE_URL="http://localhost:8300"
  fi
  if [[ "$VLM_NAME" = "meta/llama-3.2-90b-vision-instruct" ]]; then
    VLM_NAME="ds_sop_model"
  fi
elif [[ "$VLM_MODE" = "local_shared" ]]; then
  # For local_shared VLM, if base URL is still remote default, set to empty for fallback
  if [[ "$VLM_BASE_URL" = "$DEFAULT_API_URL" ]]; then
    VLM_BASE_URL=""
  fi
fi

sed -i "s|^VLM_MODE=.*|VLM_MODE=$VLM_MODE|" "$ENV_FILE"
sed -i "s|^VLM_BASE_URL=.*|VLM_BASE_URL='$VLM_BASE_URL'|" "$ENV_FILE"
sed -i "s|^VLM_NAME=.*|VLM_NAME=$VLM_NAME|" "$ENV_FILE"

VLM_NAME_SLUG=$(echo "$VLM_NAME" | awk -F/ '{print $NF}')
sed -i "s|^VLM_NAME_SLUG=.*|VLM_NAME_SLUG=$VLM_NAME_SLUG|" "$ENV_FILE"

# 5. Ensure use_base64: true in vss-agent/configs/config.yml under video_understanding
CONFIG_YML="$BP_REPO/deployments/sop/vss-agent/configs/config.yml"
if [[ -f "$CONFIG_YML" ]]; then
  echo "Ensuring 'use_base64: true' is set under video_understanding in $CONFIG_YML"
  python3 -c "
import sys
path = '$CONFIG_YML'
with open(path, 'r') as f:
    lines = f.readlines()

in_video_understanding = False
has_use_base64 = False
indent = ''
insert_idx = -1

for i, line in enumerate(lines):
    stripped = line.strip()
    if line.startswith('  video_understanding:'):
        in_video_understanding = True
        continue
    if in_video_understanding:
        if line.strip() and not line.startswith(' ' * 3):
            in_video_understanding = False
            continue
        if stripped.startswith('use_base64:'):
            has_use_base64 = True
            if 'true' not in stripped.lower():
                leading_spaces = len(line) - len(line.lstrip())
                lines[i] = ' ' * leading_spaces + 'use_base64: true\n'
            break
        if stripped.startswith('vst_internal_url:'):
            leading_spaces = len(line) - len(line.lstrip())
            indent = ' ' * leading_spaces
            insert_idx = i + 1

if not has_use_base64 and insert_idx != -1:
    lines.insert(insert_idx, f'{indent}use_base64: true\n')

with open(path, 'w') as f:
    f.writelines(lines)
"
  echo "✅ config.yml updated"
else
  echo "⚠️ Warning: config.yml not found at $CONFIG_YML"
fi

# 6. Ensure SOP_MESSAGING_SCHEMA=JSON and ENABLE_MESSAGING=1 in deployments/ds/ds-sop/.env
DS_SOP_ENV="$BP_REPO/deployments/ds/ds-sop/.env"
if [[ -f "$DS_SOP_ENV" ]]; then
  echo "Ensuring 'SOP_MESSAGING_SCHEMA=JSON' and 'ENABLE_MESSAGING=1' are set in $DS_SOP_ENV"
  if grep -q "^SOP_MESSAGING_SCHEMA=" "$DS_SOP_ENV"; then
    sed -i "s|^SOP_MESSAGING_SCHEMA=.*|SOP_MESSAGING_SCHEMA=JSON|" "$DS_SOP_ENV"
  else
    echo "SOP_MESSAGING_SCHEMA=JSON" >> "$DS_SOP_ENV"
  fi
  
  if grep -q "^ENABLE_MESSAGING=" "$DS_SOP_ENV"; then
    sed -i "s|^ENABLE_MESSAGING=.*|ENABLE_MESSAGING=1|" "$DS_SOP_ENV"
  else
    echo "ENABLE_MESSAGING=1" >> "$DS_SOP_ENV"
  fi
  echo "✅ DS-SOP .env updated"
else
  echo "⚠️ Warning: DS-SOP .env not found at $DS_SOP_ENV"
fi

# Verification
echo -e "\nUpdated configurations in .env:"
grep -E '^(MDX_SAMPLE_APPS_DIR|MDX_DATA_DIR|HOST_IP|EXTERNAL_IP|LLM_MODE|VLM_MODE|LLM_BASE_URL|VLM_BASE_URL|LLM_NAME|VLM_NAME)=' "$ENV_FILE"

echo "=== Blueprint Configuration Complete! ==="

