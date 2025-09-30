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

import logging
from pathlib import Path
from typing import List


logger = logging.getLogger(__name__)


def get_all_json_paths(dataset_path: str) -> List[str]:
    """
    Get all JSON paths in a dataset folder.
    """
    json_paths = []
    dataset_path = Path(dataset_path)

    if dataset_path.exists():
        # Find all JSON files recursively
        for json_file in dataset_path.rglob("*.json"):
            json_paths.append(str(json_file))

    return json_paths
