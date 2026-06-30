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

# Copy all reference files into the target deployments directory.
#
# This copies top-level compose.yml, cleanup_all_datalog.sh, and the
# entire deployments/sop/ directory (including all subfolders and hidden files),
# as well as the DS-SOP references, verbatim from the skill references.
#
# Usage: copy_references.sh [-r|--bp-repo PATH] [BP_REPO]
#   BP_REPO  Optional path to the blueprint repo root (default: cwd)
set -euo pipefail

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/lib/resolve_bp_repo.sh"
resolve_bp_repo "$@"

DEPLOY_REFS="$BP_REPO/agentic/vss-sop-skills/vss-sop-build/references/deployments"
DS_REFS="$BP_REPO/agentic/vss-sop-skills/vss-sop-build/references/deployments/ds/ds-sop"

DEPLOY_DEST="$BP_REPO/deployments"
DS_DEST="$BP_REPO/deployments/ds/ds-sop"

if [[ ! -d "$DEPLOY_REFS" ]]; then
  echo "Error: Skill deployments reference directory not found at $DEPLOY_REFS" >&2
  exit 1
fi

if [[ ! -d "$DS_REFS" ]]; then
  echo "Error: Skill DS-SOP reference directory not found at $DS_REFS" >&2
  exit 1
fi

mkdir -p "$DEPLOY_DEST"
mkdir -p "$DS_DEST"

echo "=== Copying all reference files ==="
echo "Copying deployments references from: $DEPLOY_REFS"
echo "To destination: $DEPLOY_DEST"
cp -r "$DEPLOY_REFS"/. "$DEPLOY_DEST/"

echo "Copying DS-SOP references from: $DS_REFS"
echo "To destination: $DS_DEST"
cp "$DS_REFS/.env" "$DS_DEST/.env"
cp "$DS_REFS/ds-sop-docker-compose.yml" "$DS_DEST/ds-sop-docker-compose.yml"

# Count total copied files
FILE_COUNT="$(find "$DEPLOY_DEST" -type f | wc -l)"
echo "Success! Total files now in $DEPLOY_DEST/: $FILE_COUNT"
echo "=== Reference Copy Completed ==="

