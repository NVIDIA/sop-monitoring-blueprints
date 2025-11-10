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
Utility functions for sop_monitoring
"""
from __future__ import annotations

import logging

_LOGGER = logging.getLogger(__name__)

def setup_logging(log_level_name: str) -> None:
    try:
        log_level = getattr(logging, log_level_name.upper())
    except AttributeError:
        _LOGGER.error("Invalid log level: %s. Using INFO instead.", log_level_name)
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s"
    )

