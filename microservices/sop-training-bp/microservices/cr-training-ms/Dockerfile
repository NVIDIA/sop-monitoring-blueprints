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

ARG COSMOS_RL_BUILD_MODE=no-efa
ARG CUDA_VERSION=12.8.1
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS no-efa-base

ARG GDRCOPY_VERSION=v2.4.4
ARG EFA_INSTALLER_VERSION=1.42.0
ARG AWS_OFI_NCCL_VERSION=v1.16.0
# NCCL version, should be found at https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2204/x86_64/
ARG NCCL_VERSION=2.26.2-1+cuda12.8
ARG FLASH_ATTN_VERSION=2.8.3
ARG PYTHON_VERSION=3.12

ENV TZ=Etc/UTC

# Do not pass --allow-unauthenticated — that bypasses apt's signature verification.
# The base nvidia/cuda image already trusts the standard Ubuntu archives, and the
# CUDA repo's cuda-keyring is installed below for CUDA packages.
RUN apt-get update -y && apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        curl git lsb-release tzdata dnsutils \
    && apt-get purge -y cuda-compat-*


#################################################
## Install NVIDIA GDRCopy
##
## NOTE: if `nccl-tests` or `/opt/gdrcopy/bin/sanity -v` crashes with incompatible version, ensure
## that the cuda-compat-xx-x package is the latest.
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install

ENV LD_LIBRARY_PATH=/opt/gdrcopy/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/gdrcopy/lib:$LIBRARY_PATH
ENV PATH=/opt/gdrcopy/bin:$PATH

###################################################
## Install NCCL with specific version.
## Combined into a SINGLE RUN so the base's libnccl bytes don't persist in a
## lower layer (see the same pattern in evaluation-ms/Dockerfile).
RUN apt-get remove -y --purge --allow-change-held-packages libnccl2 libnccl-dev \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update -y \
    && apt-get install -y libnccl2=${NCCL_VERSION} libnccl-dev=${NCCL_VERSION}

###################################################
## Install cuDNN
RUN apt-get update -y && \
    apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12

###################################################
## Install redis
# Download and add Redis GPG key, Redis APT repository
RUN curl -fsSL https://packages.redis.io/gpg  | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg && \
    chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb  $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list

# Update package list
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -qq -y redis-server

###################################################
## Install python (add-apt-repository needs software-properties-common, so
## bundle these three steps into one RUN).
RUN apt-get install -qq -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -qq -y --allow-change-held-packages \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv
## Create a virtual environment
RUN python${PYTHON_VERSION} -m venv /opt/venv/cosmos_rl
ENV PATH="/opt/venv/cosmos_rl/bin:$PATH"

# Bootstrap pip + install torch in one RUN. Even though we don't depend on
# torchaudio, vllm does — pin the CUDA-matching torch here. This pytorch
# 2.8.0 will be overridden by the vllm installation below (to pytorch 2.9.1).
RUN pip install -U pip psutil==7.1.2 setuptools==79.0.1 wheel==0.45.1 \
    && pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Cap parallel compile jobs: flash-attn's setup.py + transformer_engine +
# (later) the upstream flash-attention git-clone build all respect MAX_JOBS
# and spawn that many concurrent nvcc invocations, each fanning out to
# cicc+cudafe+cc1plus+ptxas. On a 64-core host those subprocesses'
# pgtables + rss can aggregate past 500GB of RAM and trigger global OOM
# (the same failure mode that originally bit eval_ms). 8 jobs keeps peak
# under ~30GB with minimal wall-clock penalty.
#
# Configurable via build-arg for hosts with different memory headroom:
#   docker compose build --build-arg MAX_JOBS=16 cosmos-reason-microservice
# The ENV line is required because flash-attn / torch.utils.cpp_extension /
# ninja read MAX_JOBS from os.environ during the install RUN.
ARG MAX_JOBS=8
ENV MAX_JOBS=${MAX_JOBS}

# 1. Install other dependencies first
RUN pip install \
    torchao==0.16.0 \
    flash_attn==${FLASH_ATTN_VERSION} \
    vllm==0.11.0 \
    flashinfer-python==0.6.1 \
    transformer_engine[pytorch]==2.14.1 --no-build-isolation \
    wandb==0.24.2

# install apex
# RUN APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation git+https://github.com/NVIDIA/apex@bf903a2
RUN NVCC_APPEND_FLAGS="--threads ${MAX_JOBS}" APEX_PARALLEL_BUILD=${MAX_JOBS} APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation git+https://github.com/NVIDIA/apex@bf903a2

###################################################

# Install nvshmem + grouped_gemm for MoE (two pip ops, one layer)
RUN pip install nvidia-nvshmem-cu12==3.4.5 \
    && TORCH_CUDA_ARCH_LIST="8.0 9.0+PTX" pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4 --no-build-isolation
RUN apt-get update && apt-get install -y  libibverbs-dev
# Pin DeepEP to the last `main` commit before its NCCL-backend addition.
#
# The original line `git clone DeepEP.git` (no ref) was working when MR !34
# was authored on 2026-04-19 because at that point the latest main commit
# was 567632d (2026-02-03). On 2026-04-30, upstream merged b306af0 ("EPv2:
# faster EP, and Engram/PP/CP supports (#605)") which adds
# `csrc/kernels/backend/nccl.cu` that `#include`s `nccl_device/core.h` —
# a header introduced in NCCL 2.27+. The nvidia-nccl-cu12 wheel that
# torch/vllm pull into the venv predates that, so a fresh clone of main
# now fails to compile with:
#   fatal error: nccl_device/core.h: No such file or directory
#
# Pinning to 567632d freezes us at the commit Leo's build was tested with.
# Override via:  --build-arg DEEPEP_REF=<sha-or-tag>  if you ever need newer.
ARG DEEPEP_REF=567632d
RUN git clone https://github.com/deepseek-ai/DeepEP.git /tmp/deepep \
    && cd /tmp/deepep \
    && git checkout ${DEEPEP_REF} \
    && python setup.py build \
    && python setup.py install

# Phase for building any lib that we want to builf from source
FROM no-efa-base AS source-build

# install git
RUN apt-get update -y && apt-get install -y git

WORKDIR /workspace

RUN git clone --branch v${FLASH_ATTN_VERSION} --single-branch https://github.com/Dao-AILab/flash-attention.git

WORKDIR /workspace/flash-attention/hopper

RUN MAX_JOBS=${MAX_JOBS} NVCC_THREADS=1 python setup.py bdist_wheel

###################################################
## Image target: cosmos_rl
FROM no-efa-base AS pre-package

WORKDIR /workspace
# install fa3
COPY --from=source-build /workspace/flash-attention/hopper/dist/*.whl /workspace
RUN pip install /workspace/*.whl \
    && rm /workspace/*.whl

###################################################
## Image target: cosmos_rl
FROM pre-package AS package

WORKDIR /workspace/sop-cr-ftms

# Install cosmos-rl v0.3.9, then cosmos-reason2-utils, pin kernels, drop xformers.
# Pin kernels==0.12.3: 0.13.0 uses PEP 604 unions that huggingface_hub 0.36's
# strict dataclasses reject, and transformers 4.57 pins huggingface_hub<1.0.
RUN cd /workspace \
    && git clone --branch v0.3.9 --depth 1 https://github.com/nvidia-cosmos/cosmos-rl.git \
    && cd cosmos-rl \
    && pip install . \
    && cd .. \
    && rm -rf cosmos-rl \
    && pip install git+https://github.com/nvidia-cosmos/cosmos-reason2.git@c9301793ffa3830a2a2f061d96a991d5f7c9ac38#subdirectory=cosmos_reason2_utils \
    && pip install kernels==0.12.3 \
    && pip uninstall -y xformers

# Copy the microservice code
COPY app.py .
COPY run.sh .
COPY utils /workspace/sop-cr-ftms/utils
COPY validation /workspace/sop-cr-ftms/validation
COPY components /workspace/sop-cr-ftms/components

# Install dependencies
RUN pip install asyncpg==0.30.0 sqlalchemy[asyncio]==2.0.41

# Expose the API port
EXPOSE 32080

# Set environment variables
ENV PROJECT_ROOT=/workspace/sop-cr-ftms

# Run the microservice
CMD ["/bin/bash", "run.sh"]

