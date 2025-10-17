######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
######################################################################################################

ARG CUDA_VERSION=12.8.1
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS no-efa-base

ARG GDRCOPY_VERSION=v2.4.4
ARG EFA_INSTALLER_VERSION=1.42.0
ARG AWS_OFI_NCCL_VERSION=v1.16.0
# NCCL version, should be found at https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2204/x86_64/
ARG NCCL_VERSION=2.26.2-1+cuda12.8
ARG PYTHON_VERSION=3.12

ENV TZ=Etc/UTC

RUN apt-get update -y && apt-get upgrade -y

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    curl git lsb-release tzdata
RUN apt-get purge -y cuda-compat-*

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
## Install NCCL with specific version
RUN apt-get remove -y --purge --allow-change-held-packages \
    libnccl2 \
    libnccl-dev
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update -y \
    && apt-get install -y libnccl2=${NCCL_VERSION} libnccl-dev=${NCCL_VERSION}

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
RUN apt-get install -qq -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
## Install python
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -qq -y --allow-change-held-packages \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv
## Create a virtual environment
RUN python${PYTHON_VERSION} -m venv /opt/venv/cosmos_rl
ENV PATH="/opt/venv/cosmos_rl/bin:$PATH"

RUN pip install -U pip setuptools==79.0.1 wheel==0.45.1
# even though we don't depend on torchaudio, vllm does. in order to
# make sure the cuda version matches, we install it here.
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

RUN pip install \
    torchao==0.12.0 \
    vllm==0.10.0 \
    flash-attn==2.8.2 \
    https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.6.post1%2Bcu128torch2.7-cp39-abi3-linux_x86_64.whl


###################################################
## Image target: cosmos_rl
FROM no-efa-base AS package

WORKDIR /workspace/sop-cr1-ftms

# Install cosmos-rl to specific commit
RUN cd /workspace && \
    git clone https://github.com/nvidia-cosmos/cosmos-rl.git && \
    cd cosmos-rl && \
    git checkout b81170352d7425cdf026bac63307e78d65ae5339 && \
    pip install . && \
    cd .. && \
    rm -rf cosmos-rl

# Install cosmos-reason1-utils to specific commit
RUN pip install git+https://github.com/nvidia-cosmos/cosmos-reason1.git@3723d31ea7d8be1f1d8ba890784f24613c2831a2#subdirectory=cosmos_reason1_utils

# Copy the microservice code
COPY app.py .
COPY run.sh .
COPY utils /workspace/sop-cr1-ftms/utils
COPY validation /workspace/sop-cr1-ftms/validation
COPY components /workspace/sop-cr1-ftms/components

# Install dependencies
RUN pip install asyncpg==0.30.0 sqlalchemy[asyncio]==2.0.41

# Expose the API port
EXPOSE 32080

# Set environment variables
ENV PROJECT_ROOT=/workspace/sop-cr1-ftms

# Run the microservice
CMD ["/bin/bash", "run.sh"]
