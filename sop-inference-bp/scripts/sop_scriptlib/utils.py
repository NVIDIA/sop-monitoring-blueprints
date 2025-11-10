#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
Utility functions for SOP scripts.
"""


import os
import logging

def setup_logging(log_filename: str, log_level: int):
    # Setup logging to both console and file
    log_format = "[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s"
    log_file_path = os.path.join(log_filename)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)