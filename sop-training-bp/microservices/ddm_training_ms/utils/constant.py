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


TRAIN_CONFIG_NAME = os.getenv("TRAIN_CONFIG_NAME", "ddm_train_config.yaml")

DATASET_ROOT = os.getenv("DATASET_ROOT", "/workspace/sop-ddm-ftms/assets/data")
RESULTS_ROOT = os.getenv("RESULTS_ROOT", "/workspace/sop-ddm-ftms/assets/results")
PRETRAINED_MODEL_ROOT = os.getenv("PRETRAINED_MODEL_ROOT", "/workspace/sop-ddm-ftms/assets/weights")
CONFIG_PATH = os.getenv("CONFIG_PATH", "/workspace/sop-ddm-ftms/assets/config")
TOOL_PATH = os.getenv("TOOL_PATH", "/workspace/sop-ddm-ftms/assets/tools")

# file names
DDM_TRAIN_ANNOTATION_NAME = "ddm_train_annotation.json"
DDM_VAL_ANNOTATION_NAME = "ddm_val_annotation.json"


# Postgres DB: postgresql+asyncpg://username:password@host:port/database_name
# host is the service name in docker-compose.yml
POSTGRES_USER = os.getenv("POSTGRES_USER", "sop")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "sop")
POSTGRES_DB = os.getenv("POSTGRES_DB", "sop_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "metadata_db")
POSTGRES_DB_URL = os.getenv(
    "POSTGRES_DB_URL", f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}"
)

