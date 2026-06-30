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

# Copy agents/ from upstream video-search-and-summarization and apply SOP modifications.
#
# Usage: copy_agents_from_upstream.sh [-r|--bp-repo PATH] [BP_REPO]
#   BP_REPO  Path to the blueprint repo root (default: cwd)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/resolve_bp_repo.sh"
resolve_bp_repo "$@"
UPSTREAM="${BP_REPO}/video-search-and-summarization/deployments/agents"
TARGET="${BP_REPO}/deployments/agents"
REFS="${BP_REPO}/agentic/vss-sop-skills/vss-sop-build/references/configs/vss-agent"

echo "=== Agents Step 1: Copy from upstream ==="

if [[ ! -d "${UPSTREAM}" ]]; then
  echo "ERROR: Upstream agents not found at ${UPSTREAM}" >&2
  echo "Run clone_and_prepare.sh first."
  exit 1
fi

rm -rf "${TARGET}"
cp -r "${UPSTREAM}" "${TARGET}"
echo "Copied upstream agents/ → ${TARGET}/"

# Step 1: Replace agents/compose.yml with reference version (ai-agents commented out)
cp "${REFS}/agents-compose-include.yml" "${TARGET}/compose.yml"
echo "Replaced agents/compose.yml from reference"

# Step 2: Patch bp_sop_2d profiles in vss-agent-docker-compose.yml
VSS_AGENT_COMPOSE="${TARGET}/vss-agent/vss-agent-docker-compose.yml"

# patch_profiles.py handles inline [profile] format; for block format, use sed
python3 "${SCRIPT_DIR}/patch_profiles.py" "${VSS_AGENT_COMPOSE}" 2>/dev/null || true

# Insert bp_sop_2d as first entry in block-format profiles: sections
python3 - "${VSS_AGENT_COMPOSE}" << 'PYEOF'
import re, sys
path = sys.argv[1]
content = open(path).read()
content = re.sub(r'(    profiles:\n)(    - )', r'\1    - bp_sop_2d\n\2', content)
open(path, "w").write(content)
assert content.count("- bp_sop_2d") >= 2, "bp_sop_2d not added to both services"
print("  Patched bp_sop_2d profiles in vss-agent-docker-compose.yml")
PYEOF

# Step 3: Add SOP patch volume mounts to vss-va-mcp
VOLUMES_REF="${REFS}/vss-va-mcp-volumes.yml"
python3 - "${VSS_AGENT_COMPOSE}" "${VOLUMES_REF}" << 'PYEOF'
import re, sys

vss_compose = sys.argv[1]
volumes_ref = sys.argv[2]

# Parse the 4 volume lines from the reference file
ref_lines = [l for l in open(volumes_ref).read().splitlines() if l.strip().startswith('-')]
new_volumes = "\n".join(f"    {l.strip()}" for l in ref_lines)

content = open(vss_compose).read()
# Replace only the first occurrence (vss-va-mcp, not vss-agent) of the single volume
content = content.replace(
    "    - ${MDX_SAMPLE_APPS_DIR}:/vss-agent/deployments:ro\n    environment:",
    f"{new_volumes}\n    environment:",
    1
)
open(vss_compose, "w").write(content)
print("  Added SOP patch volume mounts to vss-va-mcp")
PYEOF

# Step 4: Remove agent-eval volume and non-SOP depends_on entries from vss-agent
python3 - "${VSS_AGENT_COMPOSE}" << 'PYEOF'
import re, sys
path = sys.argv[1]
content = open(path).read()

# Remove agent-eval volume mount
content = content.replace("    - agent-eval:/vss-agent/agent_eval:rw\n", "")

# Remove top-level agent-eval volumes definition
content = re.sub(
    r'\nvolumes:\n  agent-eval:.*?device: \$MDX_DATA_DIR/agent_eval\n',
    '\n', content, flags=re.DOTALL
)

# Remove non-SOP depends_on entries
for entry in ["rtvi-vlm", "rtvi-embed", "lvs-server",
              "nvidia-nemotron-nano-9b-v2-fp8",
              "nvidia-nemotron-nano-9b-v2-fp8-shared-gpu"]:
    content = re.sub(
        rf'      {re.escape(entry)}:\n        condition: service_healthy\n        required: false\n',
        '', content
    )

open(path, "w").write(content)
assert "agent-eval" not in content, "agent-eval still present"
assert "rtvi-vlm" not in content, "rtvi-vlm still present"
assert "nvidia-nemotron-nano-9b-v2-fp8" not in content, "fp8 still present"
print("  Removed agent-eval volume and non-SOP depends_on from vss-agent")
PYEOF

# Step 5: Patch bp_sop_2d profiles in agent_ui/compose.yml (inline format)
python3 "${SCRIPT_DIR}/patch_profiles.py" "${TARGET}/agent_ui/compose.yml"
echo "  Patched bp_sop_2d profiles in agent_ui/compose.yml"

echo ""
echo "=== Agents Copy and Modify Complete ==="
# Verify via the single source of truth (verify_build.py), not duplicated grep checks.
python3 "${SCRIPT_DIR}/verify_build.py" "${BP_REPO}" --component agents || true
echo ""
echo "Next step: Run copy_nim_from_upstream.sh"

