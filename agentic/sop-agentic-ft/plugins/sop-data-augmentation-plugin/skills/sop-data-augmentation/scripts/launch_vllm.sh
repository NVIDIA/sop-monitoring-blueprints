#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#
# Launch a local vLLM server for GQA augmentation
#
# Usage:
#   ./scripts/launch_vllm.sh                          # Docker mode (recommended)
#   ./scripts/launch_vllm.sh --bare-metal              # Direct pip + vllm serve
#   ./scripts/launch_vllm.sh --model Qwen/Qwen2.5-7B  # Different model
#   ./scripts/launch_vllm.sh --port 8000               # Different port
#   ./scripts/launch_vllm.sh --tp 2                    # 2-GPU tensor parallel
#   ./scripts/launch_vllm.sh --max-len 65536           # Larger context window
#   ./scripts/launch_vllm.sh --max-num-seqs 256        # Lower batch concurrency
#   ./scripts/launch_vllm.sh --gpu-mem-util 0.95       # Tighter memory budget
#   ./scripts/launch_vllm.sh --stop                    # Stop the Docker container
#
# Why MAX_MODEL_LEN / MAX_NUM_SEQS defaults are conservative:
#   Hybrid models (e.g. Qwen3.5-27B) need both KV-cache slots and Mamba-cache
#   blocks per concurrent sequence. On a single 96 GB GPU the original
#   262144 / 1024 defaults reserved more memory than was available after
#   weights loaded, causing vLLM's CUDA-graph budget pre-flight to fail with:
#     "max_num_seqs (1024) exceeds available Mamba cache blocks (651)"
#   The defaults below stay within budget on a single 96 GB card. For longer
#   context or higher batch concurrency, raise these and/or use --tp 2.
#
# After launch, the script prints a ready-to-paste gqas: snippet to stdout
# (with the actual served model id, machine ip, and port). Use that snippet
# verbatim in assets/config/augment_config.yaml.
set -euo pipefail

# Default model — single source of truth. Override at the CLI with --model or
# via the MODEL env var. Anything else (docs, references) should defer here.
MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-9000}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
CONTAINER_NAME="sop-vllm"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai@sha256:2e08b462bb444a6da8a84a533f09024c61617574e67386efe4a723a0633fcc6a}"
MODE="docker"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bare-metal)    MODE="bare-metal"; shift ;;
        --model)         MODEL="$2"; shift 2 ;;
        --port)          PORT="$2"; shift 2 ;;
        --tp)            TP="$2"; shift 2 ;;
        --max-len)       MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-num-seqs)  MAX_NUM_SEQS="$2"; shift 2 ;;
        --gpu-mem-util)  GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --stop)          echo "Stopping ${CONTAINER_NAME}..."; docker rm -f "${CONTAINER_NAME}" 2>/dev/null; exit 0 ;;
        --help|-h)       head -25 "$0" | tail -23; exit 0 ;;
        *)               echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== vLLM Server Launch ==="
echo "  Model:        ${MODEL}"
echo "  Port:         ${PORT}"
echo "  TP:           ${TP}"
echo "  Max len:      ${MAX_MODEL_LEN}"
echo "  Max num seqs: ${MAX_NUM_SEQS}"
echo "  GPU mem util: ${GPU_MEMORY_UTILIZATION}"
echo "  Mode:         ${MODE}"
echo ""

if [[ "${MODE}" == "docker" ]]; then
    # Check prerequisites
    if ! command -v docker &>/dev/null; then
        echo "ERROR: docker not found. Install Docker or use --bare-metal mode." >&2
        exit 1
    fi
    if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia\|Default Runtime.*nvidia" && \
       ! docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &>/dev/null 2>&1; then
        echo "WARNING: NVIDIA Container Toolkit may not be installed."
        echo "  Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
        echo "  Continuing anyway..."
    fi

    # Stop existing container if running
    if docker ps -q -f name="${CONTAINER_NAME}" | grep -q .; then
        echo "Stopping existing ${CONTAINER_NAME} container..."
        docker rm -f "${CONTAINER_NAME}" >/dev/null
    fi

    echo "Pulling ${VLLM_IMAGE} (if needed)..."
    docker pull "${VLLM_IMAGE}" 2>/dev/null || true

    # Mount only the model-cache subdir (not the whole ~/.cache/huggingface,
    # which contains the auth `token` file). Pass HF_TOKEN as an env var so
    # vLLM can still authenticate for gated model downloads.
    mkdir -p "${HOME}/.cache/huggingface/hub"
    HF_TOKEN_ENV=()
    if [[ -r "${HOME}/.cache/huggingface/token" ]]; then
        HF_TOKEN_ENV=(-e "HF_TOKEN=$(cat "${HOME}/.cache/huggingface/token")")
    fi

    echo "Starting vLLM in Docker..."
    docker run -d \
        --name "${CONTAINER_NAME}" \
        --gpus "${DOCKER_GPUS:-all}" \
        --ipc host \
        -p "${PORT}:${PORT}" \
        -v "${HOME}/.cache/huggingface/hub:/root/.cache/huggingface/hub" \
        "${HF_TOKEN_ENV[@]}" \
        "${VLLM_IMAGE}" \
        --model "${MODEL}" \
        --port "${PORT}" \
        --tensor-parallel-size "${TP}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --reasoning-parser qwen3

    echo ""
    echo "Container '${CONTAINER_NAME}' started."
    echo "  Logs:  docker logs -f ${CONTAINER_NAME}"
    echo "  Stop:  ./scripts/launch_vllm.sh --stop"

elif [[ "${MODE}" == "bare-metal" ]]; then
    # Check/install vllm
    if ! python3 -c "import vllm" 2>/dev/null; then
        echo "Installing vllm..."
        pip install vllm
    fi

    echo "Starting vLLM server (foreground)..."
    exec vllm serve "${MODEL}" \
        --port "${PORT}" \
        --tensor-parallel-size "${TP}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --reasoning-parser qwen3
fi

# Wait for server to be ready
echo ""
echo "Waiting for server to be ready (up to 10 min for model loading + warmup)..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "vLLM server is ready at http://localhost:${PORT}"
        echo ""
        echo "Test:"
        echo "  curl http://localhost:${PORT}/v1/models"
        echo ""
        echo "Config for augment_config.yaml:"
        echo "  gqas:"
        echo "    llm_type: \"local\""
        echo "    local_llm_url: \"http://$(hostname -I | awk '{print $1}'):${PORT}/v1\""
        echo "    llm: ${MODEL}"
        echo "    enable_thinking: \"false\""
        exit 0
    fi
    sleep 5
done

echo "Server not ready after 10 minutes. Check logs:"
echo "  docker logs ${CONTAINER_NAME}"
