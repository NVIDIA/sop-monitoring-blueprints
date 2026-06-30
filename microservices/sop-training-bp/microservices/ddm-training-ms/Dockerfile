######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
######################################################################################################

ARG CUDA_VERSION=12.8.1
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS no-efa-base

ARG GDRCOPY_VERSION=v2.4.4
ARG NCCL_VERSION=2.26.2-1+cuda12.8
ARG PYTHON_VERSION=3.10
ARG FFMPEG_VERSION=6.1.1

ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

# ==============================================================================
# 1. System Dependencies & Setup
#    Includes: Build tools, FFmpeg deps (yasm), Python PPA, Redis Repo, NCCL Repo
# ==============================================================================
RUN apt-get update -y && apt-get upgrade -y && \
    # Install basic build tools and utilities
    apt-get install -y --no-install-recommends \
    curl git lsb-release tzdata wget ca-certificates \
    build-essential gcc g++ make pkg-config \
    software-properties-common gpg \
    yasm && \
    # Setup Python PPA
    add-apt-repository ppa:deadsnakes/ppa -y && \
    # Setup Redis Repository
    curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg && \
    chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list && \
    # Setup NCCL Repository (Remove old packages first to avoid conflicts)
    apt-get remove -y --purge --allow-change-held-packages libnccl2 libnccl-dev && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o cuda-keyring.deb && \
    dpkg -i cuda-keyring.deb && rm cuda-keyring.deb && \
    # Remove conflicting cuda-compat packages
    apt-get purge -y cuda-compat-* && \
    # Clean up apt cache to reduce image size
    rm -rf /var/lib/apt/lists/*

# ==============================================================================
# 2. Install Application Libraries
#    Install Python, Redis, and NCCL in a single layer
# ==============================================================================
RUN apt-get update -qq && \
    apt-get install -qq -y --allow-change-held-packages \
    libnccl2=${NCCL_VERSION} libnccl-dev=${NCCL_VERSION} \
    redis-server \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv && \
    rm -rf /var/lib/apt/lists/*

# ==============================================================================
# 3. Build & Install FFmpeg (and clean up source code)
# ==============================================================================
WORKDIR /tmp
RUN wget -q https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.bz2 \
    && tar xjvf ffmpeg-${FFMPEG_VERSION}.tar.bz2 \
    && cd ffmpeg-${FFMPEG_VERSION} \
    && ./configure \
      --prefix=/opt/ffmpeg \
      --disable-gpl \
      --disable-nonfree \
      --enable-shared \
      --disable-doc \
      --disable-static \
    && make -j$(nproc) \
    && make install \
    && cd /tmp \
    && rm -rf ffmpeg-${FFMPEG_VERSION} ffmpeg-${FFMPEG_VERSION}.tar.bz2

# ==============================================================================
# 4. Install GDRCopy (and clean up source code)
# ==============================================================================
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install \
    && rm -rf /tmp/gdrcopy

# ==============================================================================
# [FIX] Setup Environment Variables for FFmpeg & GDRCopy
# Crucial: LD_LIBRARY_PATH must include /opt/ffmpeg/lib for torchcodec to work
# ==============================================================================
ENV PATH=/opt/ffmpeg/bin:/opt/gdrcopy/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/ffmpeg/lib:/opt/gdrcopy/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/gdrcopy/lib:$LIBRARY_PATH
ENV PKG_CONFIG_PATH=/opt/ffmpeg/lib/pkgconfig:$PKG_CONFIG_PATH

# ==============================================================================
# 5. Python Environment & Basic Packages
# ==============================================================================
RUN python${PYTHON_VERSION} -m venv /opt/venv/ddm_net
ENV PATH="/opt/venv/ddm_net/bin:$PATH"

# Combined pip installs, removed duplicate psutil, and installed ninja early for extensions
RUN pip install -U pip setuptools==79.0.1 wheel==0.45.1 && \
    pip install \
    ninja==1.13.0 \
    psutil==7.1.2 \
    asyncpg==0.30.0 \
    sqlalchemy[asyncio]==2.0.41 \
    fastapi==0.124.0 \
    uvicorn[standard]==0.38.0 \
    pydantic==2.12.5 \
    toml==0.10.2

# Install PyTorch (Kept as a separate layer as it is large and changes infrequently)
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

###################################################
## Image target: ddm_ftms
FROM no-efa-base AS package

WORKDIR /workspace/sop-ddm-ftms

# Install DDM requirements
COPY ddm/requirements.txt /tmp/ddm_requirements.txt
# Removed numpy, torch, torchvision, torchcodec from requirements to avoid re-installation/conflicts
RUN sed -i '/^torch$/d' /tmp/ddm_requirements.txt && \
    sed -i '/^torchvision$/d' /tmp/ddm_requirements.txt && \
    sed -i '/^numpy$/d' /tmp/ddm_requirements.txt && \
    pip install --no-cache-dir -r /tmp/ddm_requirements.txt && \
    pip install torch==2.7.1 torchcodec==0.7.0 torchvision==0.22.1

RUN pip install "numpy<2.0" --force-reinstall

# Copy Source Code
COPY ddm /workspace/sop-ddm-ftms/ddm
COPY app.py run.sh ./
COPY utils /workspace/sop-ddm-ftms/utils
COPY validation /workspace/sop-ddm-ftms/validation
COPY components /workspace/sop-ddm-ftms/components

# Compile DDM-Net
# CUDA architecture and corresponding GPU versions:
# 7.0: V100
# 7.5: T4, RTX 2080, RTX 2070, RTX 2060
# 8.0: A100
# 8.6: RTX 3090, RTX 3080, RTX 3070, A4500, A6000
# 8.9: H100 Hopper Engineering Sample
# 9.0: H100 Hopper Production Version
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Cap parallel compile jobs: torch.utils.cpp_extension (which builds the
# DDM CUDA extension via `pip install -e .`) reads MAX_JOBS to control
# how many concurrent nvcc invocations it spawns. On a 64-core host the
# subprocesses' pgtables + rss can aggregate past 500GB of RAM and
# trigger global OOM. 8 jobs keeps peak under ~30GB.
#
# Configurable via build-arg for hosts with different memory headroom:
#   docker compose build --build-arg MAX_JOBS=16 ddm-training-microservice
# The ENV line is required because torch.utils.cpp_extension reads
# MAX_JOBS from os.environ during the pip install RUN.
ARG MAX_JOBS=8
ENV MAX_JOBS=${MAX_JOBS}

RUN cd /workspace/sop-ddm-ftms/ddm/DDM-Net && \
    pip install -e . --no-build-isolation -v

# Verify Installation
RUN echo "=== Verifying CUDA extension ===" && \
    ls -lh /workspace/sop-ddm-ftms/ddm/DDM-Net/modeling/tao_pytorch_backbone_v2/ops/*.so && \
    echo "=== CUDA extension successfully built and located ===" || \
    (echo "ERROR: CUDA extension .so file not found!" && exit 1)

EXPOSE 32100
ENV PROJECT_ROOT=/workspace/sop-ddm-ftms

CMD ["/bin/bash", "run.sh"]