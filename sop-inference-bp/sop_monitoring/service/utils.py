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
Utility functions for the service
"""

import os
import logging
import socket
import uuid

import redis
import redis.asyncio

from .msg_types import STR_ENCODING

_LOGGER = logging.getLogger(__name__)

def create_redis_client(db_index: int) -> redis.Redis:
    return redis.Redis(
        host=os.environ["REDIS_MSG_BROKER_NAME"],
        port=os.environ["REDIS_MSG_BROKER_PORT"],
        encoding=STR_ENCODING,
        db=db_index,
    )

def create_async_redis_client(db_index: int) -> redis.asyncio.Redis:
    return redis.asyncio.Redis(
        host=os.environ["REDIS_MSG_BROKER_NAME"],
        port=os.environ["REDIS_MSG_BROKER_PORT"],
        encoding=STR_ENCODING,
        db=db_index,
    )

def generate_reply_stream_name() -> str:
    return uuid.uuid4().hex

def get_mongo_uri() -> str:
    MONGO_USER = os.environ["MONGO_INITDB_ROOT_USERNAME"]
    MONGO_PASS = os.environ["MONGO_INITDB_ROOT_PASSWORD"]
    MONGO_NAME = os.environ["MONGO_NAME"]
    MONGO_PORT = os.environ["MONGO_PORT"]
    uri = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_NAME}:{MONGO_PORT}/"
    return uri

def get_hostname() -> str:
    try:
        hostname = socket.gethostname()
    except Exception as e:
        hostname = uuid.uuid4().hex
        _LOGGER.warning("Error getting hostname. Using UUID: %s instead.", hostname)
        _LOGGER.warning("Error: %s", e)
    return hostname
