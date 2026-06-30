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

# Tear down the VSS SOP deployment

set -euo pipefail

# Auto-detect and handle Docker permission issues (run under sg docker or sudo if needed)
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/lib/ensure_docker_access.sh"
ensure_docker_access "$@"

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/lib/resolve_bp_repo.sh"
resolve_bp_repo "$@"
ENV_FILE="$BP_REPO/deployments/sop/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "❌ Error: .env file not found at $ENV_FILE" >&2
  exit 1
fi

echo "=== Tearing Down VSS SOP Deployment ==="
echo "Blueprint Repo: $BP_REPO"

cd "$BP_REPO/deployments"

echo "Stopping Docker Compose services..."
docker compose -f compose.yml --env-file sop/.env --profile bp_sop_2d down

echo "Cleaning up all data logs..."
if [[ -f "./cleanup_all_datalog.sh" ]]; then
  chmod +x ./cleanup_all_datalog.sh
  ./cleanup_all_datalog.sh -e sop/.env
else
  echo "⚠️ Warning: cleanup_all_datalog.sh not found."
fi

echo "=== Teardown Complete ==="

