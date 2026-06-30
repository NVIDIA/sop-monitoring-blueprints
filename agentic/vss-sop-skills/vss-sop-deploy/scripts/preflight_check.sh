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

# Pre-flight check script for VSS SOP deployment

set -euo pipefail

# Default path to blueprint repo is current directory
BP_REPO="."
FIX=false
AWK_PRINT_FIELD3='{print $3}'

show_help() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -r, --bp-repo PATH    Path to the vss-sop repository (default: .)"
  echo "  -f, --fix             Attempt to auto-install or fix missing prerequisites"
  echo "  -h, --help            Show this message"
  return 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -r|--bp-repo)
      BP_REPO="$2"
      shift 2
      ;;
    -f|--fix)
      FIX=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

BP_REPO=$(realpath "$BP_REPO")
echo "=== Starting Pre-flight Checks (Repo: $BP_REPO) ==="

# 1. Secret key files
echo -n "Checking NGC API key file... "
if [[ -s "$BP_REPO/.secret/ngc_api_key.txt" ]]; then
  echo "✅ Found at $BP_REPO/.secret/ngc_api_key.txt"
else
  if [[ "$FIX" = "true" ]] && [[ -t 0 ]]; then
    echo "❌ MISSING!"
    read -rsp "Enter your NVIDIA NGC API Key: " user_key
    echo ""
    if [[ -n "$user_key" ]]; then
      mkdir -p "$BP_REPO/.secret"
      chmod 700 "$BP_REPO/.secret"
      printf '%s' "$user_key" > "$BP_REPO/.secret/ngc_api_key.txt"
      chmod 600 "$BP_REPO/.secret/ngc_api_key.txt"
      echo "✅ Saved key to $BP_REPO/.secret/ngc_api_key.txt"
    else
      echo "❌ Invalid API key. Skipping."
      exit 1
    fi
  else
    echo "❌ MISSING!"
    echo "   Fix: Create the directory and save your key:"
    echo "   mkdir -p $BP_REPO/.secret && chmod 700 $BP_REPO/.secret"
    echo "   printf '%s' '<your_ngc_key>' > $BP_REPO/.secret/ngc_api_key.txt"
    echo "   chmod 600 $BP_REPO/.secret/ngc_api_key.txt"
    exit 1
  fi
fi

# 2. NVIDIA GPU Driver & CUDA Toolkit
echo -n "Checking NVIDIA GPU Driver... "
if command -v nvidia-smi &> /dev/null; then
  DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
  echo "✅ Found driver version $DRIVER_VER"
  # Extract major version
  MAJOR_VER=$(echo "$DRIVER_VER" | cut -d. -f1)
  if [[ "$MAJOR_VER" -lt 580 ]]; then
    echo "   ⚠️ Warning: Driver version is less than 580. Update recommended."
  fi
else
  if [[ "$FIX" = "true" ]]; then
    echo "❌ Missing. Installing NVIDIA driver 580..."
    sudo apt-get update
    
    # Auto-detect driver package
    DRIVER_PKG=""
    for pkg in nvidia-driver-580-open nvidia-driver-580 nvidia-open; do
      if apt-cache show "$pkg" &>/dev/null; then
        DRIVER_PKG="$pkg"
        break
      fi
    done
    
    if [[ -n "$DRIVER_PKG" ]]; then
      echo "Installing $DRIVER_PKG..."
      sudo apt-get install -y "$DRIVER_PKG"
      echo "Loading NVIDIA kernel modules..."
      sudo modprobe nvidia || true
      sudo modprobe nvidia-uvm || true
      sudo modprobe nvidia-modeset || true
      
      if command -v nvidia-smi &> /dev/null; then
        DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
        echo "✅ NVIDIA Driver installed successfully (version $DRIVER_VER)"
      else
        echo "❌ NVIDIA Driver installed but nvidia-smi is still not communicating. Please reboot."
        exit 1
      fi
    else
      echo "❌ Could not find a suitable NVIDIA 580 driver package in apt cache."
      exit 1
    fi
  else
    echo "❌ nvidia-smi not found. NVIDIA Driver is missing."
    echo "   Fix: Please install the NVIDIA GPU Driver (580.x + CUDA 13) or run with --fix."
    exit 1
  fi
fi

# Ensure CUDA Toolkit is installed if we are fixing the system
if [[ "$FIX" = "true" ]] && ! [[ -d "/usr/local/cuda-13.1" ]] && ! [[ -d "/usr/local/cuda-13.0" ]] && ! [[ -d "/usr/local/cuda" ]]; then
  echo "Checking for CUDA Toolkit..."
  CUDA_PKG=""
  for pkg in cuda-toolkit-13-1 cuda-toolkit-13-0 cuda-toolkit-13; do
    if apt-cache show "$pkg" &>/dev/null; then
      CUDA_PKG="$pkg"
      break
    fi
  done
  if [[ -n "$CUDA_PKG" ]]; then
    echo "Installing $CUDA_PKG..."
    sudo apt-get install -y "$CUDA_PKG"
  else
    echo "⚠️ Warning: Could not find cuda-toolkit-13 in apt cache."
  fi
fi

# 3. Docker & Docker Compose
echo -n "Checking Docker... "
if command -v docker &> /dev/null; then
  DOCKER_VER=$(docker --version | awk "$AWK_PRINT_FIELD3" | tr -d ',')
  echo "✅ Found Docker $DOCKER_VER"
else
  if [[ "$FIX" = "true" ]]; then
    echo "❌ Missing. Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    DOCKER_VER=$(docker --version | awk "$AWK_PRINT_FIELD3" | tr -d ',')
    echo "✅ Docker installed successfully ($DOCKER_VER)"
  else
    echo "❌ Docker not found."
    echo "   Fix: Follow agentic/vss-sop-skills/vss-sop-deploy/references/prerequisites.md to install Docker, or run with --fix."
    exit 1
  fi
fi

echo -n "Checking Docker Compose... "
if docker compose version &> /dev/null; then
  COMPOSE_VER=$(docker compose version | awk '{print $4}')
  echo "✅ Found Docker Compose $COMPOSE_VER"
else
  if [[ "$FIX" = "true" ]]; then
    echo "❌ Missing. Installing Docker Compose..."
    if apt-cache show docker-compose-v2 &>/dev/null; then
      sudo apt-get install -y docker-compose-v2
    else
      sudo apt-get install -y docker-compose
    fi
    if docker compose version &> /dev/null; then
      COMPOSE_VER=$(docker compose version | awk '{print $4}')
      echo "✅ Docker Compose installed successfully ($COMPOSE_VER)"
    else
      echo "❌ Failed to install Docker Compose."
      exit 1
    fi
  else
    echo "❌ Docker Compose not found."
    echo "   Fix: Follow agentic/vss-sop-skills/vss-sop-deploy/references/prerequisites.md to install Docker Compose, or run with --fix."
    exit 1
  fi
fi

echo -n "Checking Docker BuildKit / buildx... "
if docker buildx version &> /dev/null; then
  echo "✅ Found docker-buildx (BuildKit native)"
else
  if [[ "$FIX" = "true" ]]; then
    echo "❌ Missing. Installing docker-buildx..."
    sudo apt-get update
    if sudo apt-get install -y docker-buildx; then
      if docker buildx version &> /dev/null; then
        echo "✅ docker-buildx installed successfully"
      else
        echo "⚠️ Installed docker-buildx but 'docker buildx version' is failing. BuildKit fallback will be used."
      fi
    else
      echo "⚠️ Failed to install docker-buildx package. BuildKit fallback will be used."
    fi
  else
    echo "❌ Missing!"
    echo "   Fix: Install docker-buildx (required for building DS-SOP via BuildKit bind mount):"
    echo "   sudo apt-get update && sudo apt-get install -y docker-buildx"
    echo "   Or run with --fix to install automatically."
    exit 1
  fi
fi

echo -n "Checking Docker Daemon connection... "
DOCKER_CMD=""
if docker ps &> /dev/null; then
  echo "✅ Connected to Docker daemon"
elif sg docker -c "docker ps" &>/dev/null; then
  DOCKER_CMD="sg docker -c"
  echo "✅ Connected to Docker daemon (using sg docker)"
else
  # Check if permission issue (sudo docker ps works)
  if sudo docker ps &>/dev/null; then
    if [[ "$FIX" = "true" ]]; then
      echo "❌ Permission denied. Adding user $USER to docker group..."
      sudo usermod -aG docker "$USER"
      if sg docker -c "docker ps" &>/dev/null; then
        DOCKER_CMD="sg docker -c"
        echo "✅ Connected to Docker daemon (after adding to group via sg docker)"
      else
        echo "❌ Added user to docker group, but group membership is not active in this shell."
        echo "   Fix: To apply this change, please run this script using:"
        echo "   sg docker -c \"$0 $*\""
        exit 1
      fi
    else
      echo "❌ Permission denied."
      echo "   Fix: Add your user to the docker group and run again:"
      echo "   sudo usermod -aG docker \$USER"
      echo "   Then run with: sg docker -c \"$0 $*\""
      exit 1
    fi
  else
    # Docker daemon is not running at all
    if [[ "$FIX" = "true" ]]; then
      echo "❌ Daemon not running. Starting Docker service..."
      sudo systemctl start docker
      sudo systemctl enable docker
      if docker ps &>/dev/null; then
        echo "✅ Connected to Docker daemon"
      elif sg docker -c "docker ps" &>/dev/null; then
        DOCKER_CMD="sg docker -c"
        echo "✅ Connected to Docker daemon (using sg docker)"
      else
        echo "❌ Started docker service, but still cannot connect. Check permissions."
        exit 1
      fi
    else
      echo "❌ Cannot connect to Docker daemon. Is it running?"
      exit 1
    fi
  fi
fi

# Define helper to run docker commands with or without sg docker
run_docker() {
  if [[ -n "$DOCKER_CMD" ]]; then
    sg docker -c "$*"
  else
    "$@"
  fi
  return 0
}

# 4. NVIDIA Container Toolkit
echo -n "Checking NVIDIA Container Toolkit... "
if run_docker docker info 2>/dev/null | grep -q "nvidia"; then
  echo "✅ NVIDIA runtime registered in Docker"
else
  if [[ "$FIX" = "true" ]]; then
    echo "❌ Missing. Installing NVIDIA Container Toolkit..."
    sudo apt-get install -y nvidia-container-toolkit nvidia-container-toolkit-base
    echo "Configuring NVIDIA Container Toolkit in Docker..."
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    if run_docker docker info 2>/dev/null | grep -q "nvidia"; then
      echo "✅ NVIDIA runtime registered in Docker"
    else
      echo "❌ Failed to register NVIDIA runtime in Docker."
      exit 1
    fi
  else
    echo "❌ NVIDIA runtime NOT registered in Docker."
    echo "   Fix: Follow agentic/vss-sop-skills/vss-sop-deploy/references/prerequisites.md to configure NVIDIA Container Toolkit, or run with --fix."
    exit 1
  fi
fi

echo -n "Testing NVIDIA GPU access in container... "
if run_docker docker run --rm --gpus all ubuntu:22.04 nvidia-smi &> /dev/null; then
  echo "✅ GPU access verified"
else
  if [[ "$FIX" = "true" ]]; then
    echo "❌ GPU access test failed. Restarting Docker..."
    sudo systemctl restart docker
    if run_docker docker run --rm --gpus all ubuntu:22.04 nvidia-smi &> /dev/null; then
      echo "✅ GPU access verified"
    else
      echo "❌ GPU access test failed after restart."
      echo "   Fix: Verify NVIDIA Container Toolkit installation and driver communication."
      exit 1
    fi
  else
    echo "❌ GPU access test failed."
    echo "   Fix: Verify NVIDIA Container Toolkit installation and restart Docker, or run with --fix."
    exit 1
  fi
fi

# 5. NGC CLI
echo -n "Checking NGC CLI... "
if command -v ngc &> /dev/null; then
  NGC_VER=$(ngc --version 2>&1 | head -n1)
  echo "✅ Found NGC CLI ($NGC_VER)"
else
  if [[ "$FIX" = "true" ]]; then
    echo "❌ Missing. Installing NGC CLI..."
    # Install unzip if missing
    if ! command -v unzip &>/dev/null; then
      echo "unzip not found. Installing unzip..."
      sudo apt-get install -y unzip
    fi
    
    ARCH=$(uname -m)
    if [[ "$ARCH" = "x86_64" ]]; then
      URL="https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.10.0/files/ngccli_linux.zip"
    elif [[ "$ARCH" = "aarch64" ]]; then
      URL="https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.10.0/files/ngccli_arm64.zip"
    else
      echo "❌ Unsupported architecture: $ARCH"
      exit 1
    fi
    curl -sLo /tmp/ngccli.zip "$URL"
    sudo mkdir -p /usr/local/lib
    sudo unzip -qo /tmp/ngccli.zip -d /usr/local/lib
    sudo chmod +x /usr/local/lib/ngc-cli/ngc
    sudo ln -sfn /usr/local/lib/ngc-cli/ngc /usr/local/bin/ngc
    if command -v ngc &> /dev/null; then
      NGC_VER=$(ngc --version 2>&1 | head -n1)
      echo "✅ NGC CLI installed successfully ($NGC_VER)"
    else
      echo "❌ NGC CLI was installed but 'ngc' command is still not found in PATH."
      exit 1
    fi
  else
    echo "❌ NGC CLI not found."
    echo "   Fix: Follow agentic/vss-sop-skills/vss-sop-deploy/references/ngc.md to install the NGC CLI, or run with --fix."
    exit 1
  fi
fi

# 6. NGC CLI config
echo -n "Checking NGC CLI Configuration... "
if [[ -f "$HOME/.ngc/config" ]]; then
  ORG=$(ngc config current 2>/dev/null | grep -i 'org' | awk -F'|' "$AWK_PRINT_FIELD3" | tr -d ' ' || true)
  if [[ -z "$ORG" ]]; then
    ORG=$(ngc config current 2>/dev/null | grep -E '^org:' | awk '{print $2}' || true)
  fi
  TEAM=$(ngc config current 2>/dev/null | grep -i 'team' | awk -F'|' "$AWK_PRINT_FIELD3" | tr -d ' ' || true)
  if [[ -z "$TEAM" ]]; then
    TEAM=$(ngc config current 2>/dev/null | grep -E '^team:' | awk '{print $2}' || true)
  fi
  if [[ "$ORG" = "no-org" ]] && [[ "$TEAM" = "no-team" ]]; then
    echo "✅ Configured for no-org / no-team"
  else
    if [[ "$FIX" = "true" ]] && [[ -s "$BP_REPO/.secret/ngc_api_key.txt" ]]; then
      echo "⚠️ Mismatch detected (Org: '$ORG', Team: '$TEAM'). Reconfiguring..."
      export NGC_CLI_API_KEY=$(cat "$BP_REPO/.secret/ngc_api_key.txt")
      printf '%s\nascii\nno-org\nno-team\nno-ace\n' "${NGC_CLI_API_KEY}" | ngc config set &>/dev/null
      echo "✅ Configured for no-org / no-team"
    else
      echo "⚠️ Configured but organization/team mismatch (Org: '$ORG', Team: '$TEAM')"
      echo "   Re-run configuration using no-org / no-team."
    fi
  fi
else
  if [[ "$FIX" = "true" ]] && [[ -s "$BP_REPO/.secret/ngc_api_key.txt" ]]; then
    echo "❌ Missing. Configuring NGC CLI..."
    export NGC_CLI_API_KEY=$(cat "$BP_REPO/.secret/ngc_api_key.txt")
    printf '%s\nascii\nno-org\nno-team\nno-ace\n' "${NGC_CLI_API_KEY}" | ngc config set &>/dev/null
    echo "✅ Configured for no-org / no-team"
  else
    echo "❌ NGC configuration missing."
    echo "   Fix: Configure NGC CLI using: "
    echo "   export NGC_CLI_API_KEY=\$(cat $BP_REPO/.secret/ngc_api_key.txt)"
    echo "   printf '%s\nascii\nno-org\nno-team\nno-ace\n' \"\$NGC_CLI_API_KEY\" | ngc config set"
    exit 1
  fi
fi

echo "=== All Pre-flight Checks PASSED successfully! ==="

