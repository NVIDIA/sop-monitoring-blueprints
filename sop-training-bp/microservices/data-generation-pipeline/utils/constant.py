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

RUNNING_STATUS = "running"
COMPLETED_STATUS = "completed"
FAILED_STATUS = "failed"
PENDING_STATUS = "pending"

# Augmentation stage names
STAGE_CONFIG_TO_BCQ = "bcq"
STAGE_CONFIG_TO_MCQ = "sequential_mcq"
STAGE_GOLDEN_GQA_TO_GQA = "golden_gqa"
STAGE_GQA_TO_GQAS = "gqas"
STAGE_CONFIG_TO_DMCQ = "dynamic_mcq"
STAGE_CONFIG_TO_DS = "dynamic_shuffling"
STAGE_CONFIG_TO_EN = "extra_negative"


DEFAULT_VIDEO_EXTENSION = "mp4"
DEFAULT_SUBJECT = "operator"
DEFAULT_LLM = "meta/llama-3.1-70b-instruct"

AUGMENTATION_CONFIG_NAME = os.getenv("AUGMENTATION_CONFIG_NAME", "augment_config.yaml")
LOG_FILE_NAME = os.getenv("LOG_FILE_NAME", "data_augmentation_log")
SOP_ACTIONS_JSON_NAME = os.getenv("SOP_ACTIONS_JSON_NAME", "actions.json")

ID_SUFFIX = "_augmented"

CONFIG_PATH = os.getenv("CONFIG_PATH", "/workspace/assets/config")
DATASET_ROOT = os.getenv("DATASET_ROOT", "/workspace/assets/data")
LOG_FILE_ROOT = os.getenv("LOG_FILE_ROOT", "/workspace/assets/logs")


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
