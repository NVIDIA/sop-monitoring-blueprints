#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#

import json
import os
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, List, Tuple

import numpy as np
import requests
import torch
from torch.multiprocessing import Queue
from torch.utils.dlpack import from_dlpack, to_dlpack

from . import ds_logger

logger = ds_logger.get_logger(__name__)

BASE_URL = "https://localhost:8000/v1"


def vss_inference_post_request(
    api_url: str, req_id: str, api_key: str, video_path: str, start_time: float, end_time: float, args: dict
):
    """
    Perform VLM inference by sending a request to the review alert API.

    Args:
        req_id: Unique identifier for the alert
        timestamp: Timestamp of the alert
        args: Dictionary or object containing configuration parameters

    Returns:
        dict: Response from the API containing inference results
    """
    # Build VLM parameters
    vlm_params = {}

    if hasattr(args, "system_prompt") and args.system_prompt:
        vlm_params["system_prompt"] = args.system_prompt
    if hasattr(args, "prompt") and args.prompt:
        vlm_params["prompt"] = args.prompt
    if hasattr(args, "max_tokens") and args.max_tokens is not None:
        vlm_params["max_tokens"] = args.max_tokens
    if hasattr(args, "temperature") and args.temperature is not None:
        vlm_params["temperature"] = args.temperature
    if hasattr(args, "top_p") and args.top_p is not None:
        vlm_params["top_p"] = args.top_p
    if hasattr(args, "top_k") and args.top_k is not None:
        vlm_params["top_k"] = args.top_k
    if hasattr(args, "seed") and args.seed is not None:
        vlm_params["seed"] = args.seed

    # Build VSS parameters
    vss_params = {
        "vlm_params": vlm_params,
    }

    if hasattr(args, "chunk_duration") and args.chunk_duration is not None:
        vss_params["chunk_duration"] = args.chunk_duration
    if hasattr(args, "chunk_overlap_duration") and args.chunk_overlap_duration is not None:
        vss_params["chunk_overlap_duration"] = args.chunk_overlap_duration
    if hasattr(args, "num_frames_per_chunk") and args.num_frames_per_chunk is not None:
        vss_params["num_frames_per_chunk"] = args.num_frames_per_chunk
    if hasattr(args, "cv_metadata_overlay") and args.cv_metadata_overlay is not None:
        vss_params["cv_metadata_overlay"] = args.cv_metadata_overlay
    if hasattr(args, "enable_reasoning") and args.enable_reasoning is not None:
        vss_params["enable_reasoning"] = args.enable_reasoning
    # Build meta_labels if available (must be a list)
    meta_labels = []
    if hasattr(args, "meta_labels") and args.meta_labels is not None:
        meta_labels = args.meta_labels

    # Build alert information (all fields required)
    alert_info = {
        "severity": getattr(args, "alert_severity", "MEDIUM"),
        "status": "REVIEW_PENDING",
        "type": getattr(args, "alert_type", "SOP_DETECTION"),
        "description": getattr(args, "alert_description", "SOP violation detected"),
    }

    # Build event information (all fields required)
    event_info = {
        "type": getattr(args, "event_type", "sop_violation"),
        "description": getattr(args, "event_description", "Potential SOP violation detected in video"),
    }

    # Build the complete request
    req_json = {
        "version": "1.0",
        "sensor_id": getattr(args, "sensor_id", "sop_detector"),
        "video_path": video_path,
        "start_time": start_time,
        "end_time": end_time,
        "id": req_id,
        "@timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",  # Include milliseconds
        "confidence": getattr(args, "confidence", 1.0),
        "cv_metadata_path": getattr(args, "cv_metadata_path", ""),
        "alert": alert_info,
        "event": event_info,
        "vss_params": vss_params,
        "meta_labels": meta_labels,
    }
    if hasattr(args, "num_frames_per_chunk") and args.num_frames_per_chunk is not None:
        req_json["num_frames_per_chunk"] = args.num_frames_per_chunk

    # Add optional stream_name field
    if hasattr(args, "stream_name") and args.stream_name:
        req_json["stream_name"] = args.stream_name

    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    logger.debug(f"Sending VLM inference request for req_id: {req_id}, request JSON: {json.dumps(req_json, indent=2)}")

    # Send request to API
    try:
        logger.info(f"Sending VLM inference request for req_id: {req_id}")
        response = requests.post(api_url, json=req_json, headers=headers)
        response.raise_for_status()
        result = response.json()
        logger.info(f"VLM inference completed successfully for req_id: {req_id}, response: {response}")
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"VLM inference failed for req_id {req_id}: {str(e)}")
        raise


class VLMInferenceClient:
    def __init__(self, endpoint: str = BASE_URL):
        self.endpoint = endpoint
        self.api_key = os.environ.get("NVIDIA_API_KEY", "")

    def get_api_url(self, path: str):
        return f"{self.endpoint}{path}" if path.startswith("/") else f"{self.endpoint}/{path}"

    def inference(self, prompt: str, video_path: str, start_time: float, end_time: float, **kwargs):
        req_id = str(uuid.uuid4())
        args_dict = {"prompt": prompt, **kwargs}
        # Convert dict to object with attributes for the API function
        args = SimpleNamespace(**args_dict)
        api_url = self.get_api_url("/reviewAlert")
        response = {}
        try:
            result = vss_inference_post_request(api_url, req_id, self.api_key, video_path, start_time, end_time, args)
            logger.debug(f"VLM inference response for req_id {req_id}: {json.dumps(result, indent=2)}")
            response = {
                "req_id": req_id,
                "response": result["result"]["description"],
            }
        except Exception as e:
            logger.exception(f"Error performing VLM inference for req_id {req_id}: {e}")
            response = {
                "req_id": req_id,
                "response": "",
                "error": str(e),
            }
        return response
