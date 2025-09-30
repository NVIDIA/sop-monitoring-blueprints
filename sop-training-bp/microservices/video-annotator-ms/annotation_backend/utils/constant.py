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

DEFAULT_VIDEO_EXTENSION = ".mp4"
DEFAULT_ACTION_INDEX = 0
DEFAULT_ACTION_DESCRIPTION = "Unknown Action"
ID_NAME_SEPARATOR = "_"

VIDEO_ROOT = os.getenv("DATASET_ROOT", "/app/assets/videos")
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "/app/assets/logs/annotation_log")

# Postgres DB: postgresql+asyncpg://username:password@host:port/database_name
# host is the service name in docker-compose.yml
POSTGRES_USER = os.getenv("POSTGRES_USER", "sop")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "sop")
POSTGRES_DB = os.getenv("POSTGRES_DB", "sop_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "metadata_db")
POSTGRES_DB_URL = os.getenv(
    "POSTGRES_DB_URL",
    f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}",
)