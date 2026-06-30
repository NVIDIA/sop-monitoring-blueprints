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


import os
import re
import shutil
from typing import Any, Dict

import yaml
from vlm_aug.utils.logger import logging

# NGC/NVIDIA personal API keys are issued as ``nvapi-<token>``. Match the prefix
# plus any run of token characters so we can strip them from any text that may be
# logged or returned to a client (command lines, tracebacks, HTTP error bodies).
_NVAPI_KEY_RE = re.compile(r"nvapi-[A-Za-z0-9_\-]+")
_REDACTED = "nvapi-***REDACTED***"


def scrub_secrets(text: Any) -> str:
    """Redact NGC/NVIDIA API keys from ``text`` before logging or returning it.
    """
    return _NVAPI_KEY_RE.sub(_REDACTED, str(text))


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


def safe_dataset_path(root: str, dataset_id: str) -> str:
    """Join ``dataset_id`` onto ``root`` and verify the result stays under ``root``.

    Raises ``ValueError`` if ``dataset_id`` contains path separators or traversal
    segments, or if the resolved path escapes ``root``. Callers should translate
    that into an HTTP 400.
    """
    if not dataset_id or "/" in dataset_id or "\\" in dataset_id or dataset_id in (".", ".."):
        raise ValueError(f"Invalid dataset id: {dataset_id!r}")
    root_real = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root_real, dataset_id))
    if candidate != root_real and not candidate.startswith(root_real + os.sep):
        raise ValueError(f"Dataset id escapes root: {dataset_id!r}")
    return candidate
