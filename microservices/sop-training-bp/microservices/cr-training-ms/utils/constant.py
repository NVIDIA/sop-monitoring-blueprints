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


QUEUE_STATUS = "queued"
RUNNING_STATUS = "running"
COMPLETED_STATUS = "completed"
CANCELLED_STATUS = "cancelled"
FAILED_STATUS = "failed"

CUSTOM_DATASET_NAME = os.getenv("CUSTOM_DATASET_NAME", "cosmos_custom_dataset.py")
TRAIN_CONFIG_NAME = os.getenv("TRAIN_CONFIG_NAME", "train_config.toml")

DATASET_ROOT = os.getenv("DATASET_ROOT", "/workspace/sop-cr-ftms/assets/data")
RESULTS_ROOT = os.getenv("RESULTS_ROOT", "/workspace/sop-cr-ftms/assets/results")
PRETRAINED_MODEL_ROOT = os.getenv("PRETRAINED_MODEL_ROOT", "/workspace/sop-cr-ftms/assets/weights")
CONFIG_PATH = os.getenv("CONFIG_PATH", "/workspace/sop-cr-ftms/assets/config")
TOOL_PATH = os.getenv("TOOL_PATH", "/workspace/sop-cr-ftms/assets/tools")


# Postgres DB: postgresql+asyncpg://username:password@host:port/database_name
# host is the service name in docker-compose.yml
POSTGRES_USER = os.getenv("POSTGRES_USER", "sop")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")  # no shipped default secret; must be set in .env (T07)
POSTGRES_DB = os.getenv("POSTGRES_DB", "sop_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "metadata_db")
POSTGRES_DB_URL = os.getenv(
    "POSTGRES_DB_URL", f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}"
)
