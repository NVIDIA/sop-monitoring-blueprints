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


QUEUE_STATUS = "queued"
RUNNING_STATUS = "running"
COMPLETED_STATUS = "completed"
CANCELLED_STATUS = "cancelled"
FAILED_STATUS = "failed"

CUSTOM_DATASET_NAME = os.getenv("CUSTOM_DATASET_NAME", "cosmos_custom_dataset.py")
TRAIN_CONFIG_NAME = os.getenv("TRAIN_CONFIG_NAME", "train_config.toml")

DATASET_ROOT = os.getenv("DATASET_ROOT", "/workspace/sop-cr1-ftms/assets/data")
RESULTS_ROOT = os.getenv("RESULTS_ROOT", "/workspace/sop-cr1-ftms/assets/results")
PRETRAINED_MODEL_ROOT = os.getenv("PRETRAINED_MODEL_ROOT", "/workspace/sop-cr1-ftms/assets/weights")
CONFIG_PATH = os.getenv("CONFIG_PATH", "/workspace/sop-cr1-ftms/assets/config")
TOOL_PATH = os.getenv("TOOL_PATH", "/workspace/sop-cr1-ftms/assets/tools")


# Postgres DB: postgresql+asyncpg://username:password@host:port/database_name
# host is the service name in docker-compose.yml
POSTGRES_USER = os.getenv("POSTGRES_USER", "sop")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "sop")
POSTGRES_DB = os.getenv("POSTGRES_DB", "sop_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "metadata_db")
POSTGRES_DB_URL = os.getenv(
    "POSTGRES_DB_URL", f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}"
)
