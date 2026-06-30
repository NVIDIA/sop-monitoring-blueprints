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

# VSS Agent verification — thin wrapper.
#
# All agent checks (bp_sop_2d profiles in vss-agent + agent_ui composes, SOP patch
# volume mounts, commented-out ai-agents include, removed agent-eval volume) live in
# the single source of truth: scripts/verify_build.py. This wrapper just invokes that
# component.
#
# Usage: verify.sh [BP_REPO]   (BP_REPO defaults to the blueprint repo root)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_PY="$SCRIPT_DIR/../../../scripts/verify_build.py"
BP_REPO="${1:-$(cd "$SCRIPT_DIR/../../../../../.." && pwd)}"

exec python3 "$VERIFY_PY" "$BP_REPO" --component agents

