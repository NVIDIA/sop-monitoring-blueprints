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

# Modify the copied foundational folder to work with SOP profile.
#
# Thin orchestration wrapper: all modification logic lives in
# modify_foundational_for_sop.py (profiles, ES stock image, Logstash configs,
# Kafka topics, ES init scripts, and ES Dockerfile removal), and all
# verification lives in verify_build.py. This script only wires them together.
#
# Prerequisites: copy_foundational_from_upstream.sh must have been run first.
#
# Usage: modify_foundational_for_sop.sh [-r|--bp-repo PATH] [BP_REPO]
#   BP_REPO  Path to the blueprint repo root (default: cwd)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/resolve_bp_repo.sh"
resolve_bp_repo "$@"

TARGET="$BP_REPO/deployments/foundational"

echo "=== Foundational Step 2: Modify for SOP profile ==="

if [[ ! -d "$TARGET" ]]; then
  echo "Error: Foundational directory not found at $TARGET" >&2
  echo "Run copy_foundational_from_upstream.sh first."
  exit 1
fi

# Apply all SOP modifications (profiles, ES image, Logstash, topics, Dockerfile removal).
python3 "$SCRIPT_DIR/modify_foundational_for_sop.py" "$BP_REPO"

# Verify (single source of truth: verify_build.py).
echo ""
echo "=== Foundational Modification Complete ==="
python3 "$SCRIPT_DIR/verify_build.py" "$BP_REPO" --component foundational || true

echo ""
echo "Next step: Run verify_build.sh to validate the full build."

