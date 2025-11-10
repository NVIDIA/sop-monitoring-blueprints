#!/usr/bin/env python
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
A script to verify the blueprint by running the tests and checking the results.
"""
from __future__ import annotations

import argparse
import os
import re
import json
import pprint
import logging

from collections import defaultdict
from dataclasses import dataclass

import requests
from openai import OpenAI

from sop_scriptlib.utils import setup_logging

_LOGGER = logging.getLogger(__name__)

_ALGO_DDM_NET = "ddm-net"
_ALGO_UNIFORM = "uniform"

class InferenceRequestSender:
    """
    A class to send inference requests to the inference blueprint.
    """

    @dataclass
    class Request:
        file_id: str
        prompt_path: str
        action_json_path: str
        temporal_seg_algo: str
        uniform_chunk_length: float
        ddm_threshold: float
        ddm_nms_sec: float

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key="don't care.", max_retries=1)
        self.sop_detection_url = f"{base_url}/sop/detection"

    def send_sop_request(self, request: Request) -> tuple[dict | None, list[str] | None]:
        """
        Send an inference request to the inference blueprint.

        Returns
        """
        _LOGGER.info("Sending SOP request with file ID: %s", request.file_id)
        action_json_str = self._load_action_json(request.action_json_path)
        chat_response = self._create_chat_completion(request)

        if chat_response is None:
            return None, None

        vlm_outputs = []
        for chunk in chat_response:
            vlm_output = chunk.choices[0].delta.content
            if vlm_output is None:
                break
            vlm_outputs.append(vlm_output)

        sop_detection_response = self._run_sop_detection(
            action_json_str=action_json_str,
            vlm_output="\n".join(vlm_outputs),
            checker_id="*",
            keep_alive=False)

        return sop_detection_response, vlm_outputs

    def _load_prompt(self, prompt_path: str) -> str:
        """Load prompt from file."""
        with open(prompt_path, "r") as f:
            return f.read()

    def _load_action_json(self, action_json_path: str) -> str:
        """Load and serialize action JSON."""
        with open(action_json_path, "r") as f:
            action_json = json.load(f)
        return json.dumps(action_json)

    def _create_chat_completion(self, request: Request):
        """Create chat completion request."""
        chunking_options = {
            "algorithm": request.temporal_seg_algo,
        }
        if request.temporal_seg_algo == _ALGO_DDM_NET:
            chunking_options["threshold"] = request.ddm_threshold
            chunking_options["nms_sec"] = request.ddm_nms_sec
        elif request.temporal_seg_algo == _ALGO_UNIFORM:
            chunking_options["chunk_length"] = request.uniform_chunk_length

        prompt = self._load_prompt(request.prompt_path)

        ret = self.client.chat.completions.create(
            model="placeholder",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_file",
                         "image_file": {
                            "file_id": request.file_id,
                            "chunking_options": chunking_options,
                         }}
                    ]
                }
            ],
            stream=True,
        )

        return ret

    def _run_sop_detection(self,
                           action_json_str: str,
                           vlm_output: str,
                           checker_id: str,
                           keep_alive: bool) -> dict:
        """Run SOP detection request."""
        request_data = {
            "action_json": action_json_str,
            "vlm_output": vlm_output,
            "keep_alive": keep_alive,
            "checker_id": checker_id,
        }

        response = requests.post(self.sop_detection_url, json=request_data)

        if response.status_code == 200:
            result = response.json()
            _LOGGER.info("SOP Checker runs with vlm output\n%s", vlm_output)
            _LOGGER.info("SOP Checker result\n%s", pprint.pformat(result))
            return result

        _LOGGER.error("Error in SOP detection:[%d] %s", response.status_code, response.text)
        return {}


class Verifier:

    DETECT_CYCLE_NUM = "detect_cycle_num"
    DETECT_CYCLE_OK = "detect_cycle_ok"
    DETECT_CYCLE_ERROR = "detect_cycle_error"
    VIDEO_NUM = "video_num"
    VIDEO_TRUE_POSITIVE = "video_true_positive"
    VIDEO_TRUE_NEGATIVE = "video_true_negative"
    VIDEO_FALSE_POSITIVE = "video_false_positive"
    VIDEO_FALSE_NEGATIVE = "video_false_negative"
    ACCURACY = "accuracy"
    VIDEO_NAME_TO_FAILED_CYCLES = "video_name_to_failed_cycles"

    def __init__(self):
        self._reobj = re.compile(r"^Cycle.+->\s+(.+)$")

    def verify_results(self, inference_results: dict, video_name_to_info: dict) -> dict:
        """Verify the results."""
        cycle_count = 0
        cycle_ok = 0
        cycle_error = 0

        video_true_positive = 0
        video_true_negative = 0
        video_false_positive = 0
        video_false_negative = 0

        video_name_to_failed_cycles = defaultdict(list)

        for video_name, inference_result in inference_results.items():
            video_info = video_name_to_info[video_name]

            label = video_info["label"]

            cycle_analysis = inference_result["summary"]["cycle_analysis"]
            org_cycle_error = cycle_error
            for trace in cycle_analysis:

                cycle_count += 1

                match = self._reobj.match(trace)
                if match is None:
                    _LOGGER.error("Video_name: %s, invalid cycle analysis trace: %s", video_name, trace)
                    cycle_error += 1
                    video_name_to_failed_cycles[video_name].append(trace)
                    continue

                cause = match.group(1)
                if cause == "no issues":
                    cycle_ok += 1
                else:
                    cycle_error += 1
                    video_name_to_failed_cycles[video_name].append(trace)

            if org_cycle_error != cycle_error and label == "fail":
                video_true_negative += 1
            elif org_cycle_error == cycle_error and label == "pass":
                video_true_positive += 1
            elif org_cycle_error != cycle_error and label == "pass":
                video_false_negative += 1
            elif org_cycle_error == cycle_error and label == "fail":
                video_false_positive += 1

        ret = {
            self.DETECT_CYCLE_NUM: cycle_count,
            self.DETECT_CYCLE_OK: cycle_ok,
            self.DETECT_CYCLE_ERROR: cycle_error,
            self.VIDEO_NUM: len(inference_results),
            self.VIDEO_TRUE_POSITIVE: video_true_positive,
            self.VIDEO_TRUE_NEGATIVE: video_true_negative,
            self.VIDEO_FALSE_POSITIVE: video_false_positive,
            self.VIDEO_FALSE_NEGATIVE: video_false_negative,
            self.ACCURACY: (video_true_positive + video_true_negative) / len(inference_results),
            self.VIDEO_NAME_TO_FAILED_CYCLES: video_name_to_failed_cycles,
        }

        return ret

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Verify the blueprint by running the tests and checking the results."
    )

    parser.add_argument("-p", "--prompt_path",
                        type=str,
                        required=True,
                        help="Path to the VLM prompt file")

    parser.add_argument("-a", "--action_json_path",
                        type=str,
                        required=True,
                        help="Path to the action JSON file")

    parser.add_argument("-j", "--video_info_json",
                        type=str,
                        required=True,
                        help="Path to the video info JSON file containning file_id and label. "
                             "Please refer to upload_files.py for the format.")

    parser.add_argument("-o", "--output_dir",
                        type=str,
                        required=False,
                        default="outputs_verify_bp",
                        help="Path to the output directory. Default is outputs_verify_bp")

    parser.add_argument("-t", "--temporal_seg_algo",
                        type=str, required=False,
                        default=_ALGO_UNIFORM,
                        choices=[_ALGO_UNIFORM, _ALGO_DDM_NET],
                        help="Temporal segmentation algorithm. Default is uniform.")

    parser.add_argument("--uniform_chunk_length",
                        type=float, required=False,
                        default=2.0,
                        help="Chunk length for uniform temporal segmentation. Default is 2.0 seconds.")

    parser.add_argument("--ddm_threshold",
                        type=float, required=False,
                        default=0.5,
                        help="Threshold for DDM-Net temporal segmentation. Default is 0.5.")

    parser.add_argument("--ddm_nms_sec",
                        type=float,
                        required=False,
                        default=0.0,
                        help="NMS sec for DDM-Net temporal segmentation. Default is 0.0, "
                             "which would be converted to 0.025 * video duration in seconds.")

    parser.add_argument("-b", "--base_url",
                        type=str,
                        required=False,
                        default="http://localhost:8080/v1",
                        help="Base URL of the inference blueprint. Default is http://localhost:8080/v1.")

    parser.add_argument("--inference_results_json",
                        type=str,
                        required=False,
                        default="",
                        help="Path to the saved inference results JSON file. "
                             "If provided, the script would skip the inference and load the results from the file, "
                             "and then calculate accuracy..")

    args = parser.parse_args()
    prompt_path = args.prompt_path
    action_json_path = args.action_json_path
    video_info_json = args.video_info_json
    temporal_seg_algo = args.temporal_seg_algo
    uniform_chunk_length = args.uniform_chunk_length
    ddm_threshold = args.ddm_threshold
    ddm_nms_sec = args.ddm_nms_sec
    base_url = args.base_url
    output_dir = args.output_dir
    inference_results_json = args.inference_results_json

    error_msg = ""
    if os.path.isfile(video_info_json):
        pass
    else:
        error_msg = f"Video info JSON file not found: {video_info_json}. "
    if os.path.isfile(prompt_path):
        pass
    else:
        error_msg = f"Prompt file not found: {prompt_path}. "
    if os.path.isfile(action_json_path):
        pass
    else:
        error_msg = f"Action JSON file not found: {action_json_path}. "
    if error_msg:
        raise FileNotFoundError(error_msg)

    os.makedirs(output_dir, exist_ok=True)
    log_filename = os.path.join(output_dir, "verify_bp.log")
    setup_logging(log_filename, logging.INFO)

    inference_request_sender = InferenceRequestSender(base_url=base_url)

    with open(video_info_json, "r") as f:
        video_name_to_info = json.load(f)

    if inference_results_json:
        _LOGGER.info("Loading inference results from %s", inference_results_json)
        with open(inference_results_json, "r") as f:
            inference_results = json.load(f)
    else:
        try:
            inference_results = {}
            for video_name, video_info in video_name_to_info.items():
                file_id = video_info["file_id"]
                label = video_info["label"]

                request = InferenceRequestSender.Request(
                    file_id=file_id,
                    prompt_path=prompt_path,
                    action_json_path=action_json_path,
                    temporal_seg_algo=temporal_seg_algo,
                    uniform_chunk_length=uniform_chunk_length,
                    ddm_threshold=ddm_threshold,
                    ddm_nms_sec=ddm_nms_sec)

                sop_detection_response, vlm_outputs = inference_request_sender.send_sop_request(request)
                if sop_detection_response:
                    _LOGGER.info("Inference completed for %s", video_name)
                    inference_results[video_name] = {
                        "vlm_outputs": vlm_outputs,
                    }
                    inference_results[video_name].update(video_info)
                    inference_results[video_name].update(sop_detection_response)
                else:
                    _LOGGER.error("Inference failed for %s", video_name)

            output_json = os.path.join(output_dir, "inference_results.json")
            with open(output_json, "w") as f:
                json.dump(inference_results, f, indent=2)
            _LOGGER.info("Inference results saved to %s", output_json)

        except Exception as exc:
            _LOGGER.error("Error in main: %s", exc)
            raise

    verifier = Verifier()
    verification_results = verifier.verify_results(inference_results, video_name_to_info)
    verified_result_json = os.path.join(output_dir, "verification_results.json")
    with open(verified_result_json, "w") as f:
        json.dump(verification_results, f, indent=2)
    _LOGGER.info("Verification results saved to %s", verified_result_json)

if __name__ == "__main__":
    main()