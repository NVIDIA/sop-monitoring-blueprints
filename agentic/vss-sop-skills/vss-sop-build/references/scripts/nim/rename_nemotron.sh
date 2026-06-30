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

# nim Step 4 — Rename nvidia-nemotron-nano-9b-v2/ to nemotron-nano-v2/.
# Usage: rename_nemotron.sh [BP_REPO]
#
# After running, update the compose.yml inside to match SOP conventions
# (per-GPU profiles, short container name, *-full.env, adjusted device IDs).
set -euo pipefail

BP_REPO="${1:-${BP_REPO:-$(pwd)}}"
NIM_DIR="$BP_REPO/deployments/nim"

if [[ ! -d "$NIM_DIR" ]]; then
  echo "Error: NIM directory not found at $NIM_DIR" >&2
  exit 1
fi

if [[ -d "$NIM_DIR/nvidia-nemotron-nano-9b-v2" ]]; then
  mv "$NIM_DIR/nvidia-nemotron-nano-9b-v2" "$NIM_DIR/nemotron-nano-v2"
  echo "Renamed nvidia-nemotron-nano-9b-v2/ -> nemotron-nano-v2/"
else
  echo "Source directory $NIM_DIR/nvidia-nemotron-nano-9b-v2 not found (already renamed?)."
fi

