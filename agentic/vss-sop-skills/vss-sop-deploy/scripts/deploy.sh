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

# Deploy the VSS SOP blueprint end-to-end

set -euo pipefail

# Auto-detect and handle Docker permission issues (run under sg docker or sudo if needed)
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/lib/ensure_docker_access.sh"
ensure_docker_access "$@"

# Shared helper for pre-creating bind-mounted data_log directories
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/lib/setup_datalog_dirs.sh"

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/lib/resolve_bp_repo.sh"
resolve_bp_repo "$@"
ENV_FILE="$BP_REPO/deployments/sop/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "❌ Error: .env file not found at $ENV_FILE" >&2
  exit 1
fi

echo "=== Starting VSS SOP Deployment ==="
echo "Blueprint Repo: $BP_REPO"

# 0. Auto-configure blueprint if HOST_IP is still a placeholder
if grep -q "^HOST_IP='<HOST_IP>'" "$ENV_FILE" 2>/dev/null; then
  CONFIGURE_SCRIPT="$(dirname "$(realpath "$0")")/configure_blueprint.sh"
  if [[ -x "$CONFIGURE_SCRIPT" ]] || [[ -f "$CONFIGURE_SCRIPT" ]]; then
    echo "⚙️  .env has unconfigured HOST_IP — running configure_blueprint.sh with defaults..."
    bash "$CONFIGURE_SCRIPT" --bp-repo "$BP_REPO"
  else
    echo "⚠️  .env has unconfigured HOST_IP but configure_blueprint.sh not found at $CONFIGURE_SCRIPT"
    echo "   Run configure_blueprint.sh manually before deploy, or set HOST_IP in $ENV_FILE"
  fi
fi

# 1. Docker login to NGC
NGC_KEY_FILE="$BP_REPO/.secret/ngc_api_key.txt"
if [[ -s "$NGC_KEY_FILE" ]]; then
  echo "Logging into NGC registry nvcr.io..."
  NGC_CLI_API_KEY=$(cat "$NGC_KEY_FILE")
  docker login --username '$oauthtoken' --password "${NGC_CLI_API_KEY}" nvcr.io
else
  echo "❌ Error: NGC API key file not found at $NGC_KEY_FILE" >&2
  exit 1
fi

# 2. Export NVIDIA API Key if exists (for remote NIM models)
NVIDIA_KEY_FILE="$BP_REPO/.secret/nvidia_build_api_key.txt"
if [[ -s "$NVIDIA_KEY_FILE" ]]; then
  echo "Found NVIDIA Build API Key. Exporting NVIDIA_API_KEY..."
  export NVIDIA_API_KEY="$(cat "$NVIDIA_KEY_FILE")"
  # VLM_MODEL_TYPE=openai uses OPENAI_API_KEY for auth even against NVIDIA endpoints
  export OPENAI_API_KEY="${OPENAI_API_KEY:-${NVIDIA_API_KEY}}"
  # Also persist into sop/.env so independently-started containers (e.g. vss-agent restart)
  # pick up the correct key without re-running deploy.sh
  SOP_ENV="$BP_REPO/deployments/sop/.env"
  if [[ -f "$SOP_ENV" ]]; then
    sed -i "s|^NVIDIA_API_KEY=.*|NVIDIA_API_KEY='${NVIDIA_API_KEY}'|" "$SOP_ENV"
    echo "  Wrote NVIDIA_API_KEY into deployments/sop/.env"
  fi
else
  echo "⚠️ Warning: NVIDIA Build API key file not found at $NVIDIA_KEY_FILE."
  echo "   If using remote NIMs, the deployment might fail or lose connection to the endpoint."
fi

# 2.5 Pre-create bind-mounted data directories with 777 permissions before starting containers
# This ensures services like Kafka (mdx-kafka) and Elasticsearch (mdx-elastic) do not crash
# with permission denied errors when writing logs or data inside the volumes.
MDX_DATA_DIR=""
if [[ -f "$ENV_FILE" ]]; then
  # Extract MDX_DATA_DIR from the env file
  MDX_DATA_DIR=$(grep -E '^MDX_DATA_DIR=' "$ENV_FILE" | cut -d= -f2- | tr -d '"'\' | xargs || true)
fi

if [[ -z "${MDX_DATA_DIR}" ]]; then
  MDX_DATA_DIR="$BP_REPO/sop-data"
fi

# Clean up old Elasticsearch and Kafka data from prior deployments to prevent index/mapping pollution (e.g. 1970-01-01 index)
if [[ -d "$MDX_DATA_DIR/data_log" ]]; then
  echo "🧹 Cleaning up old Elasticsearch and Kafka data from prior deployments to prevent index/mapping pollution..."
  if command -v sudo &>/dev/null && [[ "$EUID" -ne 0 ]]; then
    sudo rm -rf "$MDX_DATA_DIR/data_log/elastic/data"/* "$MDX_DATA_DIR/data_log/elastic/logs"/* "$MDX_DATA_DIR/data_log/kafka"/* || true
  else
    rm -rf "$MDX_DATA_DIR/data_log/elastic/data"/* "$MDX_DATA_DIR/data_log/elastic/logs"/* "$MDX_DATA_DIR/data_log/kafka"/* || true
  fi
fi

setup_datalog_dirs "$MDX_DATA_DIR"

# 2.6 Pre-create the DS-SOP host cache (HOST_CACHE) and make it world-writable.
# The ds-sop container runs as uid 1001 (USER_ID:-1001) but writes JIT build artifacts
# (flashinfer/vllm/tvm-ffi lock files) into /opt/nvidia/nvds_sop/.cache, which is bind-mounted
# from HOST_CACHE. If a prior root run (e.g. evaluation with USER_ID=0) created root-owned cache
# subdirs, uid 1001 cannot write its locks and the vLLM EngineCore crashes with
# "PermissionError: ... sampling.lock". Recursively 777 the cache so any UID can write it.
DS_SOP_ENV_FILE="$BP_REPO/deployments/ds/ds-sop/.env"
HOST_CACHE=""
if [[ -f "$DS_SOP_ENV_FILE" ]]; then
  HOST_CACHE=$(grep -E '^HOST_CACHE=' "$DS_SOP_ENV_FILE" | cut -d= -f2- | tr -d '"'\' | xargs || true)
fi
if [[ -z "${HOST_CACHE}" ]]; then
  HOST_CACHE="$HOME/.cache/ds-sop"
fi
echo "🗂️  Ensuring DS-SOP host cache is writable by the container (uid 1001): $HOST_CACHE"
mkdir -p "$HOST_CACHE" 2>/dev/null || sudo mkdir -p "$HOST_CACHE"
if command -v sudo &>/dev/null && [[ "$EUID" -ne 0 ]]; then
  sudo chmod -R 777 "$HOST_CACHE" || true
else
  chmod -R 777 "$HOST_CACHE" || true
fi

# 3. Start the blueprint
echo "Starting Docker Compose services with profile 'bp_sop_2d'..."
cd "$BP_REPO/deployments"
docker compose -f compose.yml --env-file sop/.env --profile bp_sop_2d up -d

echo -e "\nServices status check:"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

echo -e "\n=== Deployment Command Completed ==="
echo "Monitor logs using: docker logs mdx-ds-sop-1"

