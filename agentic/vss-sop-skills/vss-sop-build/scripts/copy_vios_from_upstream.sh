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

# Copy VIOS (VST) from upstream video-search-and-summarization into deployments/vst/.
#
# Workflow:
#   1. Find the vst (vios) folder in video-search-and-summarization/deployments/
#   2. Copy developer/vst/ contents to deployments/vst/sop/vst/ (renamed structure)
#   3. Create top-level compose.yml
#   4. Create additional SOP directories (minio, 4 SDR module dirs)
#   5. Remove upstream-only files that don't belong in SOP
#
# Usage: copy_vios_from_upstream.sh [-r|--bp-repo PATH] [BP_REPO]
#   BP_REPO  Path to the blueprint repo root (default: cwd)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/resolve_bp_repo.sh"
resolve_bp_repo "$@"

UPSTREAM_VST="$BP_REPO/video-search-and-summarization/deployments/vst"
TARGET_VST="$BP_REPO/deployments/vst"

echo "=== VIOS Step 1: Copy from upstream ==="

# Verify upstream exists
if [[ ! -d "$UPSTREAM_VST" ]]; then
  echo "Error: Upstream VST directory not found at $UPSTREAM_VST" >&2
  echo "Run clone_and_prepare.sh first to clone the upstream repository."
  exit 1
fi

# Find the developer/vst subfolder (upstream layout)
UPSTREAM_DEV="$UPSTREAM_VST/developer/vst"
if [[ ! -d "$UPSTREAM_DEV" ]]; then
  echo "Error: Upstream developer/vst/ not found at $UPSTREAM_DEV" >&2
  exit 1
fi

echo "Found upstream VST at: $UPSTREAM_DEV"

# Create target directory structure (SOP layout: developer/vst/ → sop/vst/)
mkdir -p "$TARGET_VST/sop/vst"

# Copy upstream developer/vst/ contents into deployments/vst/sop/vst/
echo "Copying $UPSTREAM_DEV/ → $TARGET_VST/sop/vst/"
cp -r "$UPSTREAM_DEV"/. "$TARGET_VST/sop/vst/"

# Create top-level compose.yml that points to the SOP structure
cat > "$TARGET_VST/compose.yml" << 'EOF'
include:
  - path: sop/vst/docker-compose.yaml
EOF
echo "Created top-level $TARGET_VST/compose.yml"

# Create additional SOP directories not in upstream
mkdir -p "$TARGET_VST/sop/vst/minio"
mkdir -p "$TARGET_VST/sop/vst/sdr-rtspserver-http/sdr-config"
mkdir -p "$TARGET_VST/sop/vst/sdr-recorder-http/sdr-config"
mkdir -p "$TARGET_VST/sop/vst/sdr-replaystream-http/sdr-config"
mkdir -p "$TARGET_VST/sop/vst/sdr-livestream-http/sdr-config"
echo "Created SOP directories (minio, 4 SDR module dirs)"

# Remove upstream-only files that don't belong in SOP
rm -f "$TARGET_VST/sop/vst/configs/nginx-mms.conf"
rm -f "$TARGET_VST/sop/vst/configs/nginx-mms.conf.template"
rm -f "$TARGET_VST/sop/vst/configs/nginx-vst.conf.template"
rm -f "$TARGET_VST/sop/vst/configs/nginx-vst.conf"
rm -rf "$TARGET_VST/sop/vst/sdr-streamprocessing/sdr-config/data_wl.yaml"
echo "Removed upstream-only files (nginx templates, data_wl.yaml)"

# Remove the upstream scripts/ directory if copied
rm -rf "$TARGET_VST/scripts" 2>/dev/null || true

# Summary
echo ""
echo "=== VIOS Copy Complete ==="
echo "Source:  $UPSTREAM_DEV"
echo "Target:  $TARGET_VST/sop/vst/"
echo ""
echo "Directory structure:"
find "$TARGET_VST" -type d | sort | head -20
echo ""
echo "Next step: Run modify_vios_for_sop.sh to apply SOP profile modifications."

