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


# Multi-stage build for VLM Data Augmentation microservice
# Stage 1: build wheel
FROM nvcr.io/nvidia/pytorch:24.04-py3 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

# copy source (assumes source tree contains setup.cfg/pyproject.toml and requirements.txt)
COPY source/multi-modal-autolabel-augmentation-pipeline/ .

# install minimum system deps (clean apt lists to reduce image size)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsm6 libxext6 rclone ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# ensure pip/setuptools/wheel are recent
RUN python -m pip install --upgrade pip setuptools wheel

# Robustly remove any installed GUI OpenCV packages then install headless variant
RUN set -eux; \
    python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true; \
    find "$(python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || echo /usr/local/lib/python3.10/dist-packages)" -maxdepth 3 -type d -name "cv2" -exec rm -rf {} + || true; \
    python -m pip install --no-cache-dir opencv-python-headless==4.12.0.88

# Install build-time runtime requirements (builder may install runtime deps for build)
RUN python -m pip install --no-cache-dir -r requirements.txt

# packaging tools and build wheel (wheel output will be in /build/dist)
RUN python -m pip install --no-cache-dir hatchling==1.27.0 build==1.3.0 && \
    python -m build --wheel --outdir dist

# Stage 2: runtime image
FROM nvcr.io/nvidia/pytorch:24.04-py3 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/workspace:${PYTHONPATH:-}" \
    PROJECT_ROOT=/workspace \
    SERVICE_PORT=5487

WORKDIR /workspace

# Install runtime system libs and clean apt
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsm6 libxext6 rclone ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Copy the built wheel(s) from builder stage (wildcard so filename/version changes don't break)
COPY --from=builder /build/dist/*.whl /tmp/

# Install minimal runtime Python packages first (these are pinned in your requirements.txt)
RUN python -m pip install --no-cache-dir openai==1.55.3 moviepy==1.0.3

# Install the built package using wildcard; remove wheel after install
RUN set -eux; \
    ls -lah /tmp || true; \
    python -m pip install --no-cache-dir /tmp/vlm_aug-*.whl || python -m pip install --no-cache-dir /tmp/*.whl; \
    rm -f /tmp/*.whl

# Install web / DB dependencies for the microservice
RUN python -m pip install --no-cache-dir fastapi==0.116.1 \
    uvicorn[standard]==0.35.0 \
    asyncpg==0.30.0 \
    sqlalchemy[asyncio]==2.0.41

# Copy app files (preserve paths)
COPY app.py run.sh ./
COPY utils ./utils
COPY validation ./validation
COPY components ./components

# Make startup script executable and ensure ownership
RUN chmod +x /workspace/run.sh

EXPOSE 5487

# Start the microservice using the existing run.sh (it now creates directories and falls back)
CMD ["/bin/bash", "run.sh"]