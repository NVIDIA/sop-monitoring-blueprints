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
#
# Single-stage build on the CUDA 12.8.1 *devel* base. We previously tried a
# multi-stage refactor (commit b28df18) that put the final image on a CUDA
# *runtime* base to save ~10 GB. It worked through unit tests + parity
# checks but failed at e2e VLM inference because vLLM/Triton/flashinfer
# JIT-compile CUDA + C host code at runtime — they need nvcc + cuda
# headers + Python headers + multi-arch headers all present. The runtime
# base ships none of those, and adding each missing piece back surfaced
# the next one (gcc → Python.h → multi-arch pyconfig.h → nvcc → ...).
# At the point we'd need to COPY the entire cuda-toolkit (~6-8 GB), the
# multi-stage approach has effectively rebuilt the devel base inside the
# runtime stage, defeating the savings.
#
# The size optimisations that DON'T trade off against vLLM's runtime JIT
# expectations are kept in this single-stage Dockerfile:
#   - drop cosmos-rl (commit e7b0b4d, ~3-5 GB)
#   - libnccl install+purge in one RUN (commit 8c57d17, ~5 GB)
#   - drop libcudnn9-dev, consolidate cuDNN+python apt (commit ad4d538, ~100 MB)
#   - ARG-conditional DDM CUDA extension compile (commit 8627449)
# Result: ~36.4 GB image. ~12% smaller than the original 41.6 GB without
# the runtime-base risk.

ARG CUDA_VERSION=12.8.1
ARG FFMPEG_VERSION=6.1.1

###################################################
## Build FFmpeg from source (for DDM PyAV video decoding)
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS ffmpeg-builder
ARG FFMPEG_VERSION

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends yasm wget \
    && rm -rf /var/lib/apt/lists/*

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
    && make install

###################################################
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS no-efa-base

ARG GDRCOPY_VERSION=v2.4.4
# NCCL version, should be found at https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2204/x86_64/
ARG NCCL_VERSION=2.26.2-1+cuda12.8
ARG FLASH_ATTN_VERSION=2.8.3
ARG PYTHON_VERSION=3.12

ENV TZ=Etc/UTC

# Do not pass --allow-unauthenticated — that would bypass apt's signature
# verification. Modern apt with the CUDA repo's cuda-keyring (installed
# later) signs everything correctly, and the base nvidia/cuda image
# already trusts the standard Ubuntu archives.
# Add --no-install-recommends to keep the layer slim.
RUN apt-get update -y && apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl git lsb-release tzdata \
    && apt-get purge -y cuda-compat-* \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy FFmpeg from builder stage (required for DDM PyAV video decoding)
COPY --from=ffmpeg-builder /opt/ffmpeg /usr/local
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/ffmpeg.conf && ldconfig

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
##
## Combined into a SINGLE RUN: the base image (CUDA 12.8 devel) ships its own
## libnccl2 + libnccl-dev. We must (1) purge the base's version, (2) install
## our pinned version. If those happen in two separate RUNs, the base's bytes
## remain in the lower layer (whiteout doesn't free the original bytes on
## layered FS) — the image keeps ~756 MB of phantom libnccl. Doing
## remove+install in the same RUN means the layer-final state is just our
## pinned version, and the base's bytes are not committed to the new layer.
RUN apt-get remove -y --purge --allow-change-held-packages libnccl2 libnccl-dev \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update -y \
    && apt-get install -y libnccl2=${NCCL_VERSION} libnccl-dev=${NCCL_VERSION} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

###################################################
## Install cuDNN runtime + software-properties-common + python in a SINGLE
## apt RUN so we can clean up the apt cache once at the end without breaking
## downstream layers.
##
## We deliberately install libcudnn9-cuda-12 only (the runtime .so), NOT
## libcudnn9-dev-cuda-12. The dev package ships only headers + dev symlinks
## (18 files); our flash-attn build doesn't link against cudnn (verified via
## `ldd flash_attn_2_cuda.so | grep cudnn` -> empty), and torch/vllm dlopen
## the runtime libs that the non-dev package already provides. Saves ~600 MB.
## If a future build step actually needs cudnn headers, install the dev
## package in that RUN and `apt-get remove --purge` it in the same RUN so
## the bytes don't persist into a runtime layer.
RUN apt-get update -y \
    && apt-get install -y libcudnn9-cuda-12 software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -qq -y --allow-change-held-packages \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

## Create a virtual environment
RUN python${PYTHON_VERSION} -m venv /opt/venv/eval
ENV PATH="/opt/venv/eval/bin:$PATH"

RUN pip install -U pip psutil==7.1.2 setuptools==79.0.1 wheel==0.45.1
# Even though we don't depend on torchaudio, vllm does. Install torch here so the
# CUDA version is pinned before vllm is installed; vllm 0.11.0 requires torch 2.8.0
# so this base install is the version DDM-Net's CUDA extension will ultimately
# compile against (DDM recompiles in a later layer, binding to the final torch).
RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Cap parallel compile jobs: flash-attn's setup.py respects MAX_JOBS and spawns that
# many concurrent nvcc invocations, each of which fans out to cicc+cudafe+cc1plus+ptxas.
# On a 64-core host those subprocesses' pgtables + rss can aggregate past 500GB of RAM
# and trigger global OOM. 8 jobs keeps peak under ~30GB with minimal wall-clock penalty.
#
# Configurable via build-arg for hosts with different memory headroom:
#   docker compose build --build-arg MAX_JOBS=16 evaluation-microservice
# The ENV line is required because flash-attn / torch.utils.cpp_extension /
# ninja read MAX_JOBS from os.environ during the install RUN — ARG alone
# isn't visible to the subprocesses.
ARG MAX_JOBS=8
ENV MAX_JOBS=${MAX_JOBS}

# Inference stack aligned with cr-training-ms (demo branch): vllm 0.11.0 + flash-attn 2.8.3 +
# flashinfer-python. Skip training-only deps (apex, DeepEP, nvshmem, grouped_gemm,
# transformer_engine, wandb) — this is an inference-only service.
# --no-build-isolation lets flash_attn's setup.py see torch from the venv (PyPI ships
# a source tarball for flash-attn 2.8.3, not a Python-3.12 wheel).
RUN pip install --no-build-isolation \
    torchao==0.13.0 \
    flash_attn==${FLASH_ATTN_VERSION} \
    vllm==0.11.0 \
    flashinfer-python==0.6.1

###################################################
## Image target: package (microservice code + cosmos libs + DDM)
FROM no-efa-base AS package

WORKDIR /workspace/sop-eval-ms

# cosmos-rl is the cr-training-ms training framework. evaluation-ms imports
# zero symbols from cosmos_rl (verified: `grep -r 'cosmos_rl' . --include='*.py'`
# returns empty). The pure-inference reference VLM service skips it
# entirely. Dropping it saves ~3-5 GB of image size and several minutes of
# build time. The transitive deps cosmos-rl was pulling that we DO use
# (av, qwen-vl-utils, fastapi, pydantic, transformers, pyyaml, rich) all
# come in via vllm or cosmos-reason2-utils. Two deps that cosmos-rl was
# uniquely contributing — `toml` (used by DDM-Net) and `uvicorn` (used by
# run.sh to launch the API) — are now pinned explicitly below.

# Install cosmos-reason2-utils (pulls qwen-vl-utils → av, rich, transitive
# helpers), pin kernels, and drop xformers in one RUN.
# Pin kernels==0.12.3: 0.13.0 uses PEP 604 unions that huggingface_hub 0.36's
# strict dataclasses reject, and transformers 4.57 pins huggingface_hub<1.0.
RUN pip install git+https://github.com/nvidia-cosmos/cosmos-reason2.git@c9301793ffa3830a2a2f061d96a991d5f7c9ac38#subdirectory=cosmos_reason2_utils \
    && pip install kernels==0.12.3 \
    && pip uninstall -y xformers

# Copy the microservice code
COPY app.py .
COPY run.sh .
COPY utils /workspace/sop-eval-ms/utils
COPY sop /workspace/sop-eval-ms/sop
COPY validation /workspace/sop-eval-ms/validation
COPY components /workspace/sop-eval-ms/components

# Service dependencies (DB + video processing + API runtime) + DDM inference
# dependencies merged into one RUN. `uvicorn` was previously pulled
# transitively by cosmos-rl; pinned explicitly here now that cosmos-rl is
# gone — it's the ASGI server invoked by run.sh.
RUN pip install asyncpg==0.30.0 opencv-python-headless \
    qwen-vl-utils==0.0.14 sqlalchemy[asyncio]==2.0.41 \
    uvicorn==0.32.0 \
    && pip install --no-cache-dir \
    timm==1.0.19 \
    einops==0.8.1 \
    omegaconf==2.3.0 \
    matplotlib==3.10.9 \
    ninja==1.13.0
# Note: torchcodec not needed — DDM-Net video decoding uses PyAV backend
# (av package, pulled transitively by cosmos-reason2-utils → qwen-vl-utils).

# Install DDM-Net model code (source tree + optional CUDA extension).
#
# At runtime, evaluation-ms hardcodes resnetGEBD(backbone="resnet50") — see
# utils/ddm_inference.py:95. The MultiScaleDeformableAttention CUDA extension
# (.so) is loaded LAZILY inside MSDeformAttn.__init__(), and that class is
# only instantiated by the nvdinov2_* backbone code paths. So under the
# default resnet50 path, the .so is never touched; we can ship a smaller,
# faster-to-build image that omits it.
#
# Build modes:
#   default                                    -> slim build, skip .so compile
#   --build-arg WITH_DDM_CUDA=true             -> full build, compile .so
#                                                 (required ONLY if you plan
#                                                  to switch resnetGEBD to a
#                                                  nvdinov2_* backbone)
#
# Note: ddm-training-ms's Dockerfile keeps the unconditional compile, because
# its training pipeline DOES expose backbone selection (see
# assets/config/ddm_train_config.yaml: backbone). Source code is identical
# between the two services; only the build steps differ.
ARG WITH_DDM_CUDA=false
COPY ddm /workspace/ddm

# Target Ampere through Blackwell (only used when WITH_DDM_CUDA=true).
# 10.0 = datacenter Blackwell, 12.0 = workstation Blackwell.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"
ENV DDM_BASE_PATH=/workspace/ddm

# Conditional CUDA extension compile + verification.
RUN if [ "${WITH_DDM_CUDA}" = "true" ]; then \
        echo "=== Compiling DDM CUDA extension (WITH_DDM_CUDA=true) ===" && \
        cd /workspace/ddm/DDM-Net && pip install -e . --no-build-isolation -v && \
        echo "=== Verifying DDM CUDA extension ===" && \
        ls -lh /workspace/ddm/DDM-Net/modeling/tao_pytorch_backbone_v2/ops/*.so && \
        echo "=== DDM CUDA extension successfully built ==="; \
    else \
        echo "=== Skipping DDM CUDA extension (resnet50 path only) ===" && \
        echo "Re-build with --build-arg WITH_DDM_CUDA=true to enable nvdinov2_* backbones."; \
    fi

# Service runs as root (Docker default — no USER directive). Intentional
# for this internal dev tool: the bind-mounted host directories require
# the container to write as the host owner, and aligning a non-root
# container UID with each developer's host UID is friction we don't
# need. To switch to non-root later, wrap the build in a Makefile that
# injects APP_UID/APP_GID via --build-arg.

# Expose the API port
EXPOSE 32090

# Set environment variables
ENV PROJECT_ROOT=/workspace/sop-eval-ms

# Run the microservice
CMD ["/bin/bash", "run.sh"]
