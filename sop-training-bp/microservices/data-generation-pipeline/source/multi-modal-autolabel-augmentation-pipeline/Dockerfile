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


# Use NVIDIA PyTorch container as base image
FROM nvcr.io/nvidia/pytorch:24.04-py3

# Set working directory
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt /workspace/

# Install dependencies
RUN apt-get update && apt-get install libsm6 libxext6 rclone -y
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y $(pip list --format=freeze | grep opencv) && rm -rf /usr/local/lib/python3.10/dist-packages/cv2/ && pip install opencv-python-headless

# need run `pip install hatchling && python -m build` to get `dist`
COPY dist/vlm_aug-0.0.1-py3-none-any.whl /tmp/
RUN cd /tmp/ && pip install vlm_aug-0.0.1-py3-none-any.whl && rm -f vlm_aug-0.0.1-py3-none-any.whl
# Default command
CMD ["/bin/bash"]
