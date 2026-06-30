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

# Copy nim/ from upstream video-search-and-summarization and apply SOP modifications.
#
# Usage: copy_nim_from_upstream.sh [-r|--bp-repo PATH] [BP_REPO]
#   BP_REPO  Path to the blueprint repo root (default: cwd)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/resolve_bp_repo.sh"
resolve_bp_repo "$@"
UPSTREAM="${BP_REPO}/video-search-and-summarization/deployments/nim"
export TARGET="${BP_REPO}/deployments/nim"

echo "=== NIM Step 1: Copy from upstream ==="

if [[ ! -d "${UPSTREAM}" ]]; then
  echo "ERROR: Upstream nim not found at ${UPSTREAM}" >&2
  echo "Run clone_and_prepare.sh first."
  exit 1
fi

rm -rf "${TARGET}"
cp -r "${UPSTREAM}" "${TARGET}"
echo "Copied upstream nim/ → ${TARGET}/"

python3 - << 'PYEOF'
import shutil, os, re

nim_dst = os.environ["TARGET"]

# Step 1: Update top-level nim/compose.yml
compose_path = f"{nim_dst}/compose.yml"
content = open(compose_path).read()
content = content.replace("  - path: nvidia-nemotron-nano-9b-v2/compose.yml\n", "  - path: nemotron-nano-v2/compose.yml\n")
content = content.replace("  - path: nvidia-nemotron-nano-9b-v2-fp8/compose.yml\n", "")
open(compose_path, "w").write(content)
print("  Updated nim/compose.yml (removed fp8, renamed nemotron-nano-v2)")

# Step 2: Rename nvidia-nemotron-nano-9b-v2/ -> nemotron-nano-v2/
nano_src = f"{nim_dst}/nvidia-nemotron-nano-9b-v2"
nano_dst_dir = f"{nim_dst}/nemotron-nano-v2"
if os.path.exists(nano_src):
    shutil.copytree(nano_src, nano_dst_dir)
    shutil.rmtree(nano_src)
    print("  Renamed nvidia-nemotron-nano-9b-v2/ -> nemotron-nano-v2/")

# Step 3: Remove fp8 directory and fallback-override.env
for path_to_rm in [f"{nim_dst}/nvidia-nemotron-nano-9b-v2-fp8", f"{nim_dst}/fallback-override.env"]:
    if os.path.exists(path_to_rm):
        if os.path.isdir(path_to_rm):
            shutil.rmtree(path_to_rm)
        else:
            os.remove(path_to_rm)
        print(f"  Removed {os.path.basename(path_to_rm)}")

# Step 4: Per-model env file renames and compose.yml updates
model_dirs = [d for d in os.listdir(nim_dst) if os.path.isdir(f"{nim_dst}/{d}")]
for model in sorted(model_dirs):
    model_path = f"{nim_dst}/{model}"
    # Rename *-shared.env -> *-full.env
    for f in list(os.listdir(model_path)):
        if f.endswith("-shared.env"):
            os.rename(f"{model_path}/{f}", f"{model_path}/{f.replace('-shared.env', '-full.env')}")
    # Rename RTXPRO6000BW -> RTX6000PROBW
    for f in list(os.listdir(model_path)):
        if "RTXPRO6000BW" in f:
            os.rename(f"{model_path}/{f}", f"{model_path}/{f.replace('RTXPRO6000BW', 'RTX6000PROBW')}")
    # Remove hw-OTHER*, hw-DGX-SPARK*, hw-L40S.env
    for f in list(os.listdir(model_path)):
        if f.startswith("hw-OTHER") or f.startswith("hw-DGX-SPARK") or f == "hw-L40S.env":
            os.remove(f"{model_path}/{f}")

    # Update compose.yml
    mc_path = f"{model_path}/compose.yml"
    if not os.path.exists(mc_path):
        continue
    mc = open(mc_path).read()

    # Add per-GPU profile variants
    def split_profile(m):
        p = m.group(1)
        return f"    - {p}_H100\n    - {p}_RTX6000PROBW"
    mc = re.sub(r'^    - ((?:vlm_local|llm_local|llm_vlm_local)\S+)$', split_profile, mc, flags=re.MULTILINE)

    # Shorten container names
    container_map = {
        "cosmos-reason2-8b": "cr2-8b", "cosmos-reason1-7b": "cr1-7b",
        "qwen3-vl-8b-instruct": "qwen3-vl-8b",
        "llama-3.3-nemotron-super-49b-v1.5": "nemotron-super-49b",
        "gpt-oss-20b": "gpt-oss-20b", "nemotron-3-nano": "nem3-nano",
        "nvidia-nemotron-nano-9b-v2": "nem-nano-v2",
    }
    for long_name, short_name in container_map.items():
        mc = re.sub(rf'^    container_name: {re.escape(long_name)}$',
                    f'    container_name: {short_name}', mc, flags=re.MULTILINE)
        mc = re.sub(rf'^    container_name: {re.escape(long_name)}-shared-gpu$',
                    f'    container_name: {short_name}-sg', mc, flags=re.MULTILINE)

    mc = mc.replace("-shared.env", "-full.env")
    mc = re.sub(r'.*fallback-override\.env.*\n', '', mc)
    mc = mc.replace("RTXPRO6000BW", "RTX6000PROBW")
    mc = mc.replace("${VLM_DEVICE_ID:-0}", "${VLM_DEVICE_ID:-2}")
    mc = mc.replace("${SHARED_LLM_VLM_DEVICE_ID:-${VLM_DEVICE_ID:-0}}", "${LLM_DEVICE_ID:-1}")
    mc = re.sub(r'.*NIM_DISABLE_MM_PREPROCESSOR_CACHE.*\n', '', mc)
    open(mc_path, "w").write(mc)

    # Clean env files
    for f in os.listdir(model_path):
        if f.endswith(".env"):
            fpath = f"{model_path}/{f}"
            c = open(fpath).read()
            c = re.sub(r'NIM_DISABLE_MM_PREPROCESSOR_CACHE=.*\n', '', c)
            open(fpath, "w").write(c)

# Verify
assert os.path.exists(f"{nim_dst}/nemotron-nano-v2"), "nemotron-nano-v2/ missing"
assert not os.path.exists(f"{nim_dst}/nvidia-nemotron-nano-9b-v2-fp8"), "fp8 dir still present"
nim_compose = open(f"{nim_dst}/compose.yml").read()
assert "nemotron-nano-v2" in nim_compose and "fp8" not in nim_compose
print("  All NIM modifications applied successfully.")
PYEOF

echo ""
echo "=== NIM Copy and Modify Complete ==="
# Verify via the single source of truth (verify_build.py), not duplicated grep checks.
python3 "${SCRIPT_DIR}/verify_build.py" "${BP_REPO}" --component nim || true
echo ""
echo "Next step: Run verify_build.sh"

