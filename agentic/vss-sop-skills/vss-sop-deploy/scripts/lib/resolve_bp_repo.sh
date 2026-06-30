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

# Shared helper: parse --bp-repo <path> from script arguments and export BP_REPO.
# Sourced by deploy.sh, teardown.sh, download_assets.sh, configure_blueprint.sh, etc.
#
# Usage (from a script):
#   source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/lib/resolve_bp_repo.sh"
#   resolve_bp_repo "$@"
#   # BP_REPO is now set and exported

resolve_bp_repo() {
  BP_REPO=""
  while [[ $# -gt 0 ]]; do
    local arg="$1"
    case "$arg" in
      --bp-repo)
        local val="$2"
        BP_REPO="$(cd "$val" && pwd)"
        shift 2
        ;;
      --bp-repo=*)
        BP_REPO="$(cd "${arg#*=}" && pwd)"
        shift
        ;;
      *)
        shift
        ;;
    esac
  done

  if [[ -z "$BP_REPO" ]]; then
    BP_REPO="$(pwd)"
  fi

  export BP_REPO
  return 0
}

