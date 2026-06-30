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

# Shared helper: ensure the current shell can talk to the Docker daemon.
# Sourced by deploy.sh, teardown.sh, and test_rtsp.sh.
#
# Usage (from a script):
#   source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/lib/ensure_docker_access.sh"
#   ensure_docker_access "$@"
#
# If the current user lacks Docker permissions, the calling script is
# re-executed under the 'docker' group (via sg) or with sudo, preserving its
# original arguments. Exits non-zero if the daemon is unreachable.

ensure_docker_access() {
  if docker ps &>/dev/null; then
    return 0
  fi

  if sg docker -c "docker ps" &>/dev/null; then
    echo "⚠️ Docker permission denied in current shell, but 'sg docker' is available."
    echo "Re-executing script under the 'docker' group..."
    local script_path args=""
    script_path=$(realpath "$0")
    for arg in "$@"; do
      args="$args \"$arg\""
    done
    exec sg docker -c "bash \"$script_path\" $args"
  elif sudo docker ps &>/dev/null; then
    echo "⚠️ Docker permission denied. Re-executing script with sudo..."
    exec sudo "$0" "$@"
  else
    echo "❌ Error: Cannot connect to Docker daemon. Please verify Docker is running." >&2
    exit 1
  fi
}

