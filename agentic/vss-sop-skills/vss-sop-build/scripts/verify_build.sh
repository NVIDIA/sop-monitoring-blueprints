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

# Build Verification Wrapper (Stage 5 Verification & Cleanup)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/resolve_bp_repo.sh"
resolve_bp_repo "$@"

cd "$BP_REPO"

echo "=== Starting Build Verification ==="
if python3 "$SCRIPT_DIR/verify_build.py" "$BP_REPO"; then
  echo ""
  echo "=== Cleanup: Removing upstream repository ==="
  if [[ -d "video-search-and-summarization" ]]; then
    rm -rf video-search-and-summarization
    echo "Upstream repository video-search-and-summarization successfully removed."
    echo "Only deployments/ remains."
  else
    echo "No upstream video-search-and-summarization repository to remove."
  fi
  echo "=== Build Verification and Cleanup Succeeded! ==="
  exit 0
else
  echo ""
  echo "❌ Build verification FAILED. Skipping upstream repository cleanup."
  exit 1
fi

