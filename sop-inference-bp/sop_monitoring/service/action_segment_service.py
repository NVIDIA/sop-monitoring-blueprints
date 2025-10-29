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
Module for action segment service.

This service is responsible for segmenting actions from a video.

It will:
- Get a request from the API server
- Download video files according to the request
- Segment the actions from the video using either uniform chunking or uboco intelligent chunking
- Send a message to the API server to indicate the completion of the action segmentation and the chunk_start/end_seconds
"""
from __future__ import annotations

import argparse
import os
import logging
import tempfile
import time
import traceback
import uuid

import minio
from pydantic import TypeAdapter

from .constants import (
    CHUNK_ALGO_UNIFORM_NAME,
    CHUNK_ALGO_UBOCO_NAME,
    CHUNK_ALGO_DDM_NET_NAME,
    CHUNK_ALGO_COSMOS_REASON_NAME,
    REDIS_STREAM_C_ACTION_SEGMENT_P_API_SERVER_GROUP_NAME,
    REDIS_CHUNK_VIDEO_DB_INDEX,
    REDIS_STREAM_DB_INDEX,
    MINIO_BUCKET,
    REDIS_STREAM_NAME_TO_AVAILABLE_ALGOS,
)
from .msg_types import (
    ActionSegmentRequest,
    ActionSegmentResponse,
)
from .pydantic_models import CHUNKING_OPTIONS_TYPE_HINT
from .redis_database import RadisDatabase
from .redis_stream import (
    RedisStream,
    send_response,
)
from .utils import (
    create_redis_client,
    get_hostname,
)

_LOGGER = logging.getLogger(__name__)

_TMP_DIR = "/dev/shm"


# raise exception if any one of environment variables is not set or anything wrong.
minio_client = minio.Minio(
    f"{os.environ['MINIO_NAME']}:{os.environ['MINIO_API_PORT']}",
    access_key=os.environ["MINIO_ROOT_USER"],
    secret_key=os.environ["MINIO_ROOT_PASSWORD"],
    secure=False
)


def is_algo_available(redis_stream_name: str, algo_name: str) -> bool:
    ret = False

    try:
        ret = algo_name in REDIS_STREAM_NAME_TO_AVAILABLE_ALGOS[redis_stream_name]
    except KeyError:
        _LOGGER.error("Unknown Redis stream name: %s. "
                      "If it is a stream for a new algo, please add it to the table in constants.py.",
                      redis_stream_name)
        raise

    return ret

def store_chunked_video_files(redis_database: RadisDatabase, chunk_filenames: list[str], key_prefix: str = "") -> list[str]:

    if not chunk_filenames:
        return []

    key_blob_pairs = {}
    for idx, chunk_filename in enumerate(chunk_filenames):
        with open(chunk_filename, "rb") as f:
            chunk_data = f.read()
        key = f"{key_prefix}_{idx}_{uuid.uuid4().hex}"
        key_blob_pairs[key] = chunk_data

    redis_database.store_blobs_batch(key_blob_pairs)
    keys = list(key_blob_pairs.keys())

    _LOGGER.debug("Stored %d chunked video files to RedisDB", len(keys))
    return keys

def _lazy_import_cosmos_reason():
    from ..action_segment.cosmos_reason import CosmosReasonActionSegmenter

    cosmos_reason_action_segmenter = None

    try:
        model_path = os.environ["ACTION_SEGMENT_CR_CHECKPOINT_PATH_IN_CONTAINER"]
        if not os.path.exists(model_path):
            _LOGGER.error("Cosmos Reason model path %s does not exist. Please check if the path is correct.", model_path)
            _LOGGER.error("Cosmos Reason will not be available for action segmentation.")
            return cosmos_reason_action_segmenter

        cosmos_reason_action_segmenter = CosmosReasonActionSegmenter(model_path=model_path)
    except Exception as e:
        error_msg = traceback.format_exc()
        _LOGGER.error("Error initializing Cosmos Reason: %s", error_msg)
        _LOGGER.error("Cosmos Reason will not be available for action segmentation.")
    return cosmos_reason_action_segmenter

def _lazy_import_ddm_net(ddm_frames_per_side):
    # This service might not be for ddm.
    # So we laze import it.
    # The caller should make sure the ddm should be imported or not.
    from ..action_segment.ddm_net import MultiGpuDdmNet

    ddm_multi_gpu_manager = None

    try:
        ddm_frames_per_segment_hint = int(os.getenv("ACTION_SEGMENT_DDM_NET_FRAMES_PER_SEGMENT_HINT", "256"))
    except Exception as e:
        error_msg = traceback.format_exc()
        _LOGGER.error("Error parsing ACTION_SEGMENT_DDM_NET_FRAMES_PER_SEGMENT_HINT: %s. Using default value 256.", error_msg)
        ddm_frames_per_segment_hint = 256

    try:
        ACTION_SEGMENT_DDM_NET_CHECKPOINT_PATH_IN_CONTAINER = os.environ["ACTION_SEGMENT_DDM_NET_CHECKPOINT_PATH_IN_CONTAINER"]
        # Check if the checkpoint file exists and is a file before proceeding
        if os.path.exists(ACTION_SEGMENT_DDM_NET_CHECKPOINT_PATH_IN_CONTAINER) and os.path.isfile(ACTION_SEGMENT_DDM_NET_CHECKPOINT_PATH_IN_CONTAINER):
            ddm_multi_gpu_manager = MultiGpuDdmNet(
                checkpoint_path=ACTION_SEGMENT_DDM_NET_CHECKPOINT_PATH_IN_CONTAINER,
                frames_per_side=ddm_frames_per_side,
                frames_per_segment_hint=ddm_frames_per_segment_hint
            )
        else:
            _LOGGER.error("DDM-Net checkpoint file does not exist or is not a file at path: %s", ACTION_SEGMENT_DDM_NET_CHECKPOINT_PATH_IN_CONTAINER)
            _LOGGER.error("DDM-Net will not be available for action segmentation.")
            ddm_multi_gpu_manager = None
    except Exception as e:
        error_msg = traceback.format_exc()
        _LOGGER.error("Error initializing DDM-Net: %s", error_msg)
        _LOGGER.error("DDM-Net will not be available for action segmentation.")
        ddm_multi_gpu_manager = None

    return ddm_multi_gpu_manager


# Service-level model caching
_cached_viclip_model = None
_cached_slowfast_model = None
def get_cached_viclip_model():
    """Get the cached ViCLIP model."""
    global _cached_viclip_model
    return _cached_viclip_model


def set_cached_viclip_model(model):
    """Set the cached ViCLIP model."""
    global _cached_viclip_model
    _cached_viclip_model = model
    _LOGGER.info("ViCLIP model cached for reuse")


def get_cached_slowfast_model():
    """Get the cached SlowFast model."""
    global _cached_slowfast_model
    return _cached_slowfast_model


def set_cached_slowfast_model(model):
    """Set the cached SlowFast model."""
    global _cached_slowfast_model
    _cached_slowfast_model = model
    _LOGGER.info("SlowFast model cached for reuse")


def main(args: argparse.Namespace):
    redis_stream_name = args.redis_stream
    worker_name = f"worker-{redis_stream_name}-{get_hostname()}"
    _LOGGER.debug("Initializing worker %s", worker_name)

    log_level_name = os.environ.get("ACTION_SEGMENT_SERVICE_LOG_LEVEL", "INFO")
    try:
        log_level = getattr(logging, log_level_name.upper())
    except AttributeError:
        _LOGGER.error("Invalid log level: %s. Using INFO instead.", log_level_name)
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s"
    )

    redis_client_for_stream = create_redis_client(REDIS_STREAM_DB_INDEX)
    _LOGGER.info("Testing Redis client at %s:%s",
                 redis_client_for_stream.connection_pool.connection_kwargs['host'],
                 redis_client_for_stream.connection_pool.connection_kwargs['port'])
    # this will raise exception if connection fails
    redis_client_for_stream.ping()
    redis_stream = RedisStream(redis_client_for_stream)
    redis_stream.create_consumer_group(redis_stream_name,
                                       REDIS_STREAM_C_ACTION_SEGMENT_P_API_SERVER_GROUP_NAME)

    redis_client_for_database = create_redis_client(REDIS_CHUNK_VIDEO_DB_INDEX)
    redis_database = RadisDatabase(redis_client_for_database)

    # FIXME: if this should be configurable?
    ddm_frames_per_side = 5
    ddm_multi_gpu_manager = None
    if is_algo_available(redis_stream_name, CHUNK_ALGO_DDM_NET_NAME):
        ddm_multi_gpu_manager = _lazy_import_ddm_net(ddm_frames_per_side)

    cosmos_reason_action_segmenter = None
    if is_algo_available(redis_stream_name, CHUNK_ALGO_COSMOS_REASON_NAME):
        cosmos_reason_action_segmenter = _lazy_import_cosmos_reason()

    _LOGGER.info("Action segment service started. Worker name: %s", worker_name)
    # main event loop.
    while True:

        try:
            messages = redis_stream.xreadgroup(
                redis_stream_name,
                REDIS_STREAM_C_ACTION_SEGMENT_P_API_SERVER_GROUP_NAME,
                worker_name,
                ActionSegmentRequest,
                count=10,
                block=0,
            )
        except Exception as e:
            _LOGGER.error("Error reading from Redis stream: %s", e)
            time.sleep(5)  # Wait before retrying to avoid tight error loop
            continue

        # TODO: can be parallelized.
        for message_id, action_segment_request in messages:
            try:
                _LOGGER.info("Received action segment request: %s", message_id)

                action_segment_options_type_adapter = TypeAdapter(CHUNKING_OPTIONS_TYPE_HINT)
                action_segment_options: CHUNKING_OPTIONS_TYPE_HINT = (
                    action_segment_options_type_adapter.validate_json(action_segment_request.options_json)
                )
                _LOGGER.debug("Received action segment options: %s", action_segment_options)
                action_segment_algo = action_segment_options.algorithm

                if not is_algo_available(redis_stream_name, action_segment_algo):
                    _LOGGER.error("Action segment algorithm %s is not available for Redis stream %s",
                                  action_segment_algo, redis_stream_name)
                    raise ValueError(f"Action segment algorithm {action_segment_algo} is not available for Redis stream {redis_stream_name}")

                # Get the object from MinIO
                minio_response = minio_client.get_object(MINIO_BUCKET, action_segment_request.video_id)

                # /dev/shm should be RAM-backed. Here hopefully a fast Memory I/O.
                with tempfile.TemporaryDirectory(dir=_TMP_DIR) as tmp_dir:

                    video_path = os.path.join(tmp_dir, action_segment_request.video_id)
                    with open(video_path, "wb") as video_fp:
                        piece_size = 50*1024*1024 # 50MB
                        for piece in minio_response.stream(piece_size):
                            video_fp.write(piece)

                    chunk_filenames = []
                    chunk_start_seconds = []
                    chunk_end_seconds = []
                    start_time = time.time()
                    if action_segment_algo == CHUNK_ALGO_UBOCO_NAME:
                        # Lazy import because this service might not be for uboco.
                        from ..action_segment.uboco import process_video_with_uboco
                        _LOGGER.info("Using uboco intelligent chunking")

                        # Cast to UboCoChunkingOptions to access uboco-specific parameters
                        uboco_options = action_segment_options

                        # Uboco paths - these should be configurable via environment variables
                        UBOCO_BASE_PATH = os.environ["UBOCO_BASE_PATH"]
                        ACTION_SEGMENT_UBOCO_SLOWFAST_PATH_IN_CONTAINER = os.environ["ACTION_SEGMENT_UBOCO_SLOWFAST_PATH_IN_CONTAINER"]
                        ACTION_SEGMENT_UBOCO_VICLIP_PATH_IN_CONTAINER = os.environ["ACTION_SEGMENT_UBOCO_VICLIP_PATH_IN_CONTAINER"]

                        # Use parameters directly from chunking options (no fallbacks)
                        _LOGGER.info("UboCo parameters: extracted_fps=%.1f (clip_len=%.3fs), is_deterministic=%s, min_segment_seconds=%.1fs, threshold=%.2f",
                                     uboco_options.extracted_fps, 1/uboco_options.extracted_fps, uboco_options.is_deterministic,
                                     uboco_options.min_segment_seconds, uboco_options.threshold)

                        # Get cached models
                        cached_viclip_model = get_cached_viclip_model()
                        cached_slowfast_model = get_cached_slowfast_model()

                        result = process_video_with_uboco(video_path,
                                                         tmp_dir,
                                                         UBOCO_BASE_PATH,
                                                         ACTION_SEGMENT_UBOCO_VICLIP_PATH_IN_CONTAINER,
                                                         ACTION_SEGMENT_UBOCO_SLOWFAST_PATH_IN_CONTAINER,
                                                         uboco_options.is_deterministic,
                                                         cached_viclip_model=cached_viclip_model,
                                                         cached_slowfast_model=cached_slowfast_model,
                                                         return_cached_model=uboco_options.return_cached_model,
                                                         extracted_fps=uboco_options.extracted_fps,
                                                         min_segment_seconds=uboco_options.min_segment_seconds,
                                                         threshold=uboco_options.threshold)

                        # Handle returned models for caching
                        if len(result) == 4:
                            # Return format: (chunk_start_seconds, chunk_end_seconds, returned_viclip_model, returned_slowfast_model)
                            chunk_start_seconds, chunk_end_seconds, returned_viclip_model, returned_slowfast_model = result
                            # Cache the returned models for next use
                            if returned_viclip_model is not None:
                                set_cached_viclip_model(returned_viclip_model)
                            if returned_slowfast_model is not None:
                                set_cached_slowfast_model(returned_slowfast_model)
                        elif len(result) == 3:
                            # Return format: (chunk_start_seconds, chunk_end_seconds, returned_viclip_model)
                            chunk_start_seconds, chunk_end_seconds, returned_viclip_model = result
                            # Cache the returned ViCLIP model for next use
                            if returned_viclip_model is not None:
                                set_cached_viclip_model(returned_viclip_model)
                        else:
                            # Return format: (chunk_start_seconds, chunk_end_seconds)
                            chunk_start_seconds, chunk_end_seconds = result

                    elif action_segment_algo == CHUNK_ALGO_DDM_NET_NAME:
                        _LOGGER.info("Using ddm-net action segmentation")

                        if ddm_multi_gpu_manager is None:
                            raise RuntimeError("DDM-Net is not available for action segmentation. "
                                               "Please check errors in the log when the service started.")

                        _LOGGER.debug("Using ddm-net action segmentation with options: "
                                      "threshold: %s, min_length_sec: %s, max_length_sec: %s, "
                                      "frames_per_side: %s, batch_size: %s",
                                      action_segment_options.threshold,
                                      action_segment_options.min_length_sec,
                                      action_segment_options.max_length_sec,
                                      ddm_frames_per_side,
                                      action_segment_options.batch_size)

                        chunk_start_seconds, chunk_end_seconds = ddm_multi_gpu_manager.process_video(
                            input_video_path=video_path,
                            threshold=action_segment_options.threshold,
                            min_length_sec=action_segment_options.min_length_sec,
                            max_length_sec=action_segment_options.max_length_sec,
                            batch_size=action_segment_options.batch_size
                        )

                    elif action_segment_algo == CHUNK_ALGO_UNIFORM_NAME:
                        # lazy import because this service might not be for uniform chunking.
                        from ..action_segment.fixed_length import fixed_length_split_start_end_time

                        _LOGGER.debug("Using uniform chunking with chunk length %d seconds",
                                      action_segment_options.chunk_length)
                        chunk_start_seconds, chunk_end_seconds = fixed_length_split_start_end_time(video_path, action_segment_options.chunk_length)

                    elif action_segment_algo == CHUNK_ALGO_COSMOS_REASON_NAME:
                        _LOGGER.info("Using cosmos-reason action segmentation")

                        if cosmos_reason_action_segmenter is None:
                            raise RuntimeError("Cosmos Reason is not available for action segmentation. "
                                               "Please check errors in the log when the service started.")

                        chunk_start_seconds, chunk_end_seconds, _ = cosmos_reason_action_segmenter.process_video(
                            video_filename=video_path,
                            prompt=action_segment_options.user_prompt,
                            system_prompt=action_segment_options.system_prompt,
                            chunk_duration_sec=action_segment_options.chunk_duration_sec,
                            min_length_sec=action_segment_options.min_length_sec
                        )

                    else:
                        raise ValueError(f"Unknown action segment algorithm: {action_segment_algo}")

                    end_time = time.time()
                    _LOGGER.info("Action segment service took %s seconds", end_time - start_time)

                    chunk_keys = store_chunked_video_files(redis_database, chunk_filenames)
                    # The tmp_dir will be deleted automatically.
                    # To prevent unusable filenames, we delete the list here.
                    del chunk_filenames

                action_segment_response = ActionSegmentResponse(
                    request_id=message_id,
                    chunk_start_seconds=chunk_start_seconds,
                    chunk_end_seconds=chunk_end_seconds,
                    chunk_keys=chunk_keys,
                    error_message="",
                )
            except Exception as e:
                error_msg = f"Error processing temporal segmentation request {message_id}, exception: {e}, {traceback.format_exc()}"
                _LOGGER.error(error_msg)
                action_segment_response = ActionSegmentResponse(
                    request_id=message_id,
                    chunk_start_seconds=[],
                    chunk_end_seconds=[],
                    chunk_keys=[],
                    error_message=error_msg,
                )
            finally:
                _LOGGER.debug("Acknowledged message %s", message_id)
                send_response(redis_stream, action_segment_request, action_segment_response)
                redis_stream.ack(redis_stream_name,
                                 REDIS_STREAM_C_ACTION_SEGMENT_P_API_SERVER_GROUP_NAME,
                                 message_id)

    if ddm_multi_gpu_manager is not None:
        ddm_multi_gpu_manager.shutdown()


if __name__ == "__main__":
    if not os.path.isdir(_TMP_DIR):
        _LOGGER.error("Temporary directory %s does not exist. Please check if the directory is mounted.", _TMP_DIR)
        raise RuntimeError(f"Temporary directory {_TMP_DIR} does not exist. Please check if the directory is mounted.")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--redis-stream",
                            type=str,
                            required=True,
                            choices=REDIS_STREAM_NAME_TO_AVAILABLE_ALGOS.keys(),
                            help="Redis stream name that this service will listen to.")
    args = arg_parser.parse_args()
    main(args)

