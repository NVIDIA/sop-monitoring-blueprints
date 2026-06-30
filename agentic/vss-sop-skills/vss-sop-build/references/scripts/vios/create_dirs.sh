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

# vios Step 0/2 — Create the VST SOP directory structure.
# SOP restructures the VST tree entirely (vst/developer/vst/ -> vst/sop/vst/),
# so we DO NOT bulk-copy upstream — only create the target dir layout.
# Usage: create_dirs.sh [BP_REPO]
set -euo pipefail

BP_REPO="${1:-${BP_REPO:-$(pwd)}}"
VST_DIR="$BP_REPO/deployments/vst"

if [[ ! -d "$BP_REPO/deployments" ]]; then
  echo "Error: deployments directory not found at $BP_REPO/deployments" >&2
  exit 1
fi



mkdir -p "$VST_DIR"

cd "$VST_DIR"
mkdir -p sop/vst/configs
mkdir -p sop/vst/sdr-rtspserver-http/sdr-config
mkdir -p sop/vst/sdr-recorder-http/sdr-config
mkdir -p sop/vst/sdr-replaystream-http/sdr-config
mkdir -p sop/vst/sdr-livestream-http/sdr-config
mkdir -p sop/vst/minio

echo "Created VST SOP directory tree under $(pwd)"

