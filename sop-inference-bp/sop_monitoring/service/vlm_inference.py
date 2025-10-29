# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import os
import logging
import time

import minio

from .constants import (
    REDIS_STREAM_P_API_SERVER_C_VLM_INFERENCE_STREAM_NAME,
    REDIS_STREAM_C_VLM_INFERENCE_P_API_SERVER_GROUP_NAME,
    REDIS_STREAM_DB_INDEX,
    REDIS_CHUNK_VIDEO_DB_INDEX,
)
from .msg_types import (
    VlmInferenceRequest
)
from .vlm_inference_pool import VLMInferencePool
from .redis_stream import (
    RedisStream,
)
from .redis_database import RadisDatabase
from .utils import (
    create_redis_client,
    get_hostname,
    setup_logging
)

_LOGGER = logging.getLogger(__name__)

_TMP_DIR = "/dev/shm"

def get_model_dir() -> str:
    # raise exception if VLM_INFERENCE_MODEL_PATH_IN_CONTAINER is not set
    return os.environ["VLM_INFERENCE_MODEL_PATH_IN_CONTAINER"]

def get_video_filenames(redis_database: RadisDatabase, chunk_keys: list[str], chunk_dir: str) -> list[str]:
    video_filenames = []
    for chunk_key in chunk_keys:
        video_data = redis_database.get_blob(chunk_key)
        if not video_data:
            _LOGGER.error("No video data found for chunk key: %s", chunk_key)
            continue
        video_filename = os.path.join(chunk_dir, f"{chunk_key}.mp4")
        with open(video_filename, "wb") as f:
            f.write(video_data)
        video_filenames.append(video_filename)
    return video_filenames

def main():
    log_level_name = os.environ.get("VLM_INFERENCE_LOG_LEVEL", "INFO")
    setup_logging(log_level_name)

    redis_client_for_stream = create_redis_client(REDIS_STREAM_DB_INDEX)
    redis_client_for_database = create_redis_client(REDIS_CHUNK_VIDEO_DB_INDEX)

    _LOGGER.info("Testing Redis client at %s:%s",
                 redis_client_for_stream.connection_pool.connection_kwargs['host'],
                 redis_client_for_stream.connection_pool.connection_kwargs['port'])
    # this will raise exception if connection fails
    redis_client_for_stream.ping()

    redis_stream = RedisStream(redis_client_for_stream)
    redis_stream.create_consumer_group(REDIS_STREAM_P_API_SERVER_C_VLM_INFERENCE_STREAM_NAME,
                                       REDIS_STREAM_C_VLM_INFERENCE_P_API_SERVER_GROUP_NAME)

    redis_database = RadisDatabase(redis_client_for_database)

    # raise exception if any one of environment variables is not set or anything wrong.
    minio_client = minio.Minio(
        f"{os.environ['MINIO_NAME']}:{os.environ['MINIO_API_PORT']}",
        access_key=os.environ["MINIO_ROOT_USER"],
        secret_key=os.environ["MINIO_ROOT_PASSWORD"],
        secure=False
    )

    vlm_pool = VLMInferencePool(get_model_dir(), max_workers=10, max_gpu_queue_size=1000, tmp_dir=_TMP_DIR)

    _LOGGER.info("===========================================")
    _LOGGER.info("VLM inference service started.")
    _LOGGER.info("===========================================")

    # main event loop.
    while True:
        try:
            messages = redis_stream.xreadgroup(
                REDIS_STREAM_P_API_SERVER_C_VLM_INFERENCE_STREAM_NAME,
                REDIS_STREAM_C_VLM_INFERENCE_P_API_SERVER_GROUP_NAME,
                f"vlm_inference_worker_{get_hostname()}",  # consumer name
                VlmInferenceRequest,
                count=10,
                block=0  # blocking read
            )
        except Exception as e:
            _LOGGER.error("Error reading from Redis stream: %s", e)
            time.sleep(5)  # Wait before retrying to avoid tight error loop
            continue

        for message_id, vlm_request in messages:
            try:
                _LOGGER.info("Submitting VLM inference request %s to pool", message_id)
                vlm_pool.submit_request(message_id, vlm_request, redis_stream, redis_database, minio_client)
            except Exception as e:
                _LOGGER.error("Error submitting VLM inference request: %s", e)
                time.sleep(5)  # Wait before retrying to avoid tight error loop
                continue

if __name__ == "__main__":
    if not os.path.isdir(_TMP_DIR):
        _LOGGER.error("Temporary directory %s does not exist. Please check if the directory is mounted.", _TMP_DIR)
        raise RuntimeError(f"Temporary directory {_TMP_DIR} does not exist. Please check if the directory is mounted.")
    main()
