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


# Use NVIDIA PyTorch container as base image
FROM nvcr.io/nvidia/pytorch:24.04-py3

# Set working directory
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt /workspace/

# Install dependencies
RUN apt-get update && apt-get install libsm6 libxext6 rclone -y
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y $(pip list --format=freeze | grep opencv) \
    && rm -rf /usr/local/lib/python3.10/dist-packages/cv2/ \
    && pip install opencv-python-headless

# need run `pip install hatchling && python -m build` to get `dist`
COPY dist/vlm_aug-0.0.1-py3-none-any.whl /tmp/
RUN cd /tmp/ && pip install vlm_aug-0.0.1-py3-none-any.whl && rm -f vlm_aug-0.0.1-py3-none-any.whl
# Default command
CMD ["/bin/bash"]
