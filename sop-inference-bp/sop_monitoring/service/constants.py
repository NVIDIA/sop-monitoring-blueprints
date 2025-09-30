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
Constants for the service.
Constants here are not user-configurable. If we indeed want to make a value configurable, move it to the .env file or API.
"""

REDIS_STREAM_DB_INDEX = 0
REDIS_CHUNK_VIDEO_DB_INDEX = 1

# P_<producer>_C_<consumer>_stream_name
REDIS_STREAM_P_API_SERVER_C_VLM_INFERENCE_STREAM_NAME = "producer_api_server_consumer_vlm_inference_stream"
REDIS_STREAM_P_API_SERVER_C_ACTION_SEGMENT_STREAM_NAME = "producer_api_server_consumer_action_segment_stream"
REDIS_STREAM_P_API_SERVER_C_ACTION_SEGMENT_CR_STREAM_NAME = "producer_api_server_consumer_action_segment_cr_stream"
REDIS_STREAM_P_API_SERVER_C_SOP_CHECKER_STREAM_NAME = "producer_api_server_consumer_sop_checker_stream"

# C_<consumer>_P_<producer>_group_name
REDIS_STREAM_C_VLM_INFERENCE_P_API_SERVER_GROUP_NAME = "consumer_vlm_inference_producer_api_server_group"
REDIS_STREAM_C_ACTION_SEGMENT_P_API_SERVER_GROUP_NAME = "consumer_action_segment_producer_api_server_group"
REDIS_STREAM_C_SOP_CHECKER_P_API_SERVER_GROUP_NAME = "consumer_sop_checker_producer_api_server_group"

# NOTE: Cannot contain underscore.
MINIO_BUCKET = "user-files"

CHUNK_ALGO_UNIFORM_NAME = "uniform"
CHUNK_ALGO_UBOCO_NAME = "uboco"
CHUNK_ALGO_DDM_NET_NAME = "ddm-net"
CHUNK_ALGO_COSMOS_REASON_NAME = "cosmos-reason"


# Map for api server and action segment service to check available algorithms.
# The key is the redis stream name, and the value is a set of algorithm names.
REDIS_STREAM_NAME_TO_AVAILABLE_ALGOS = {
    REDIS_STREAM_P_API_SERVER_C_ACTION_SEGMENT_STREAM_NAME: set([
        CHUNK_ALGO_UNIFORM_NAME,
        CHUNK_ALGO_UBOCO_NAME,
        CHUNK_ALGO_DDM_NET_NAME,
    ]),
    # FIXME: enable this when we have the model weights.
    #REDIS_STREAM_P_API_SERVER_C_ACTION_SEGMENT_CR_STREAM_NAME: set([
    #    CHUNK_ALGO_COSMOS_REASON_NAME,
    #]),
}