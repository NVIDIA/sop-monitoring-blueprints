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

# Copy the foundational folder from upstream video-search-and-summarization
# into deployments/foundational/.
#
# Workflow:
#   1. Find foundational/ in video-search-and-summarization/deployments/
#   2. Copy the entire folder to deployments/foundational/
#
# Usage: copy_foundational_from_upstream.sh [-r|--bp-repo PATH] [BP_REPO]
#   BP_REPO  Path to the blueprint repo root (default: cwd)
set -euo pipefail

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/lib/resolve_bp_repo.sh"
resolve_bp_repo "$@"

UPSTREAM="$BP_REPO/video-search-and-summarization/deployments/foundational"
TARGET="$BP_REPO/deployments/foundational"

echo "=== Foundational Step 1: Copy from upstream ==="

if [[ ! -d "$UPSTREAM" ]]; then
  echo "Error: Upstream foundational directory not found at $UPSTREAM" >&2
  echo "Run clone_and_prepare.sh first to clone the upstream repository."
  exit 1
fi

echo "Found upstream foundational at: $UPSTREAM"

mkdir -p "$TARGET"
cp -r "$UPSTREAM"/. "$TARGET/"

echo "Copied upstream foundational/ → $TARGET/"
echo ""
echo "Files copied: $(find "$TARGET" -type f | wc -l)"
echo ""
echo "Next step: Run modify_foundational_for_sop.sh to apply SOP profile modifications."

