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

# Clone and Prepare Script for VSS SOP Build (Stage 0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/resolve_bp_repo.sh"
resolve_bp_repo "$@"

cd "$BP_REPO"

echo "=== Stage 0: Starting Clone & Prepare ==="

# 1. Clone the upstream repo (used as a read-only source)
if [[ ! -d "video-search-and-summarization" ]]; then
  echo "Cloning upstream video-search-and-summarization repo..."
  git clone https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization.git
  cd video-search-and-summarization
  git checkout 3.1.0
  cd ..
else
  echo "Upstream repository video-search-and-summarization already exists."
fi

# 2. Complete Stage 0
echo "=== Stage 0: Completed successfully ==="

