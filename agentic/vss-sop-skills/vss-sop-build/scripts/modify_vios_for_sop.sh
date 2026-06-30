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

# Modify the copied VIOS (VST) folder to work with SOP profile.
#
# Thin orchestration wrapper: all modification logic lives in
# modify_vios_for_sop.py, which transforms the copied upstream VST structure
# into the SOP VIOS layout:
#   - Copies reference SDR docker_cluster_config.json files
#   - Renames services from -dev → -sop, changes profiles → bp_sop_2d
#   - Splits monolith sdr-streamprocessing into 4 microservice SDRs
#   - Adds new services (storage-ms-sop, minio-server)
#   - Modifies .env / JSON configs, writes static nginx.conf + per-SDR envoy.yaml
#   - Writes the top-level vst/compose.yml and removes upstream leftovers
# Verification lives in verify_build.py. This script only wires them together.
#
# Prerequisites: copy_vios_from_upstream.sh must have been run first.
#
# Usage: modify_vios_for_sop.sh [-r|--bp-repo PATH] [BP_REPO]
#   BP_REPO  Path to the blueprint repo root (default: cwd)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/resolve_bp_repo.sh"
resolve_bp_repo "$@"

SOP_VST="$BP_REPO/deployments/vst/sop/vst"

echo "=== VIOS Step 2: Modify for SOP profile ==="

if [[ ! -d "$SOP_VST" ]]; then
  echo "Error: SOP VST directory not found at $SOP_VST" >&2
  echo "Run copy_vios_from_upstream.sh first."
  exit 1
fi

# Apply all SOP modifications (SDR configs, compose/.env/configs, top-level compose, cleanup).
python3 "$SCRIPT_DIR/modify_vios_for_sop.py" "$BP_REPO"

# Verify (single source of truth: verify_build.py).
echo ""
echo "=== VIOS Modification Complete ==="
python3 "$SCRIPT_DIR/verify_build.py" "$BP_REPO" --component vios || true

echo ""
echo "Next step: Run verify_build.sh to validate the full build."

