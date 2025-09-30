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


import os
import shutil
from typing import Any, Dict

import yaml
from vlm_aug.utils.logger import logging


def load_config_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        if not os.path.exists(config_path):
            logging.warning(f"Config file not found: {config_path}")
            return {}

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        logging.info(f"Loaded config from {config_path}: {config}")
        return config or {}

    except Exception as e:
        logging.error(f"Error loading config file {config_path}: {str(e)}")
        return {}

def clean_and_create_dir(dir_path: str) -> bool:
    """Create a directory if it doesn't exist"""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    
    os.makedirs(dir_path)
    return True
