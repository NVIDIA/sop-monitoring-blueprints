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

# nim Step 3b — Remove unused env files and the FP8 nemotron variant.
# Usage: cleanup_unused.sh [BP_REPO]
set -euo pipefail

BP_REPO="${1:-${BP_REPO:-$(pwd)}}"
NIM_DIR="$BP_REPO/deployments/nim"

if [[ ! -d "$NIM_DIR" ]]; then
  echo "Error: NIM directory not found at $NIM_DIR" >&2
  exit 1
fi

echo "=== nim cleanup: removing unused env files ==="

# Per-model: drop hw-OTHER variants
find "$NIM_DIR" -maxdepth 2 -type f \( -name 'hw-OTHER.env' -o -name 'hw-OTHER-shared.env' \) -print -delete

# Drop hw-DGX-SPARK* and hw-L40S.env from models that don't need them
# (uncomment per-model targeted lines if desired):
# rm -f "$NIM_DIR/<model>/hw-DGX-SPARK*.env" "$NIM_DIR/<model>/hw-L40S.env"

# Special cases
rm -rf "$NIM_DIR/nvidia-nemotron-nano-9b-v2-fp8/"
rm -f  "$NIM_DIR/fallback-override.env"

echo "Cleanup done."

