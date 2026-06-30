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
STAGE_SPATIAL_LOCALIZATION = "spatial_localization"
STAGE_FRAME_DROP = "frame_drop"


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
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")  # no shipped default secret; must be set in .env (T07)
POSTGRES_DB = os.getenv("POSTGRES_DB", "sop_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "metadata_db")
POSTGRES_DB_URL = os.getenv(
    "POSTGRES_DB_URL",
    f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}",
)
