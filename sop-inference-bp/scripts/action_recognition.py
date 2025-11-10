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
A script to run action recognition.
"""

import argparse
import os
import json
import logging

from collections import defaultdict

from sop_monitoring.vlm import CosmosReason1

from sop_scriptlib.utils import setup_logging

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="A script to run action recognition. Usually used after temporal segmentation for debugging purposes.")

    # interfaces for data
    parser.add_argument("--video",
                        type=str,
                        required=False,
                        default="",
                        help="Path to the video file to process. Exclusive with video_dir.")

    parser.add_argument("--video_dir",
                        type=str,
                        required=False,
                        default="",
                        help="Path to the directory containing video files. Exclusive with video_dir.")

    parser.add_argument("--video_ext",
                        type=str,
                        required=False,
                        default="mp4",
                        help="The expected video extension in video_dir. "
                             "Files with this ext in video_dir is considered as input. Default mp4")

    parser.add_argument("-p", "--prompt_path",
                        type=str,
                        required=True,
                        help="Path to the VLM prompt file")

    parser.add_argument("-m", "--vlm_model_dir",
                        type=str,
                        required=True,
                        help="Path to the directory containing the VLM model.")

    parser.add_argument("-t", "--temporal_seg_json",
                        type=str,
                        required=False,
                        default="",
                        help="Path to the f1_xxx.json output by temporal_segmentation.py. "
                             "If not provided, the script will just feed the whole video into the VLM model.")

    parser.add_argument("-o", "--output_dir",
                        type=str,
                        required=False,
                        default="outputs_action_recognition",
                        help="Path to the output directory. Default is outputs_action_recognition")


    args = parser.parse_args()
    video_path = args.video
    video_dir = args.video_dir
    video_ext = args.video_ext
    vlm_model_dir = args.vlm_model_dir
    prompt_path = args.prompt_path
    temporal_seg_json = args.temporal_seg_json
    output_dir = args.output_dir


    if video_path == "" and video_dir == "":
        raise ValueError("One of the arguments '--video' or '--video_dir' should be specified")
    if video_path != "" and video_dir != "":
        raise ValueError("Only one of the arguments '--video' or '--video_dir' should be specified, ")

    if temporal_seg_json:
        with open(temporal_seg_json, "r") as f:
            video_name_to_boundaries = json.load(f)
    else:
        video_name_to_boundaries = {}

    os.makedirs(output_dir, exist_ok=True)
    log_filename = os.path.join(output_dir, "action_recognition.log")
    setup_logging(log_filename, logging.INFO)

    logging.getLogger("qwen_vl_utils").setLevel(logging.WARNING)
    logging.getLogger("torchcodec").setLevel(logging.WARNING)

    _LOGGER.info(f"Args: {args}")

    videos = []
    if video_path:
        videos = [video_path]
    elif video_dir:
        for entry in os.scandir(video_dir):
            if entry.is_file() and entry.name.endswith(f".{video_ext}"):
                videos.append(entry.path)
    else:
        raise ValueError("Both --video and --video_dir are empty.")

    with open(prompt_path, "r") as f:
        prompt = f.read()

    vlm_model = CosmosReason1(model_path=vlm_model_dir)

    video_name_to_output_text = defaultdict(dict)

    for video in videos:
        video_name = os.path.basename(video)
        if video_name in video_name_to_boundaries:
            boundaries = video_name_to_boundaries[video_name]["boundaries"]
            chunk_start_seconds = boundaries[:-1]
            chunk_end_seconds = boundaries[1:]
            _LOGGER.info("Action recognition for %s with %d chunks", video_name, len(chunk_start_seconds))
            for chunk_start_second, chunk_end_second in zip(chunk_start_seconds, chunk_end_seconds):
                output_text = vlm_model.chunk_and_infer(prompt, video, chunk_start_second, chunk_end_second)
                key = f"[{chunk_start_second:.2f}s-{chunk_end_second:.2f}s]"
                _LOGGER.info("%s: %s", key, output_text)
                video_name_to_output_text[video_name][key] = output_text
        else:
            _LOGGER.info("Action recognition for %s with the whole video", video_name)
            output_text = vlm_model.chunk_and_infer(prompt, video, None, None)
            key = "whole"
            _LOGGER.info("%s: %s", key, output_text)
            video_name_to_output_text[video_name][key] = output_text

    output_json = os.path.join(output_dir, "video_name_to_output_text.json")
    _LOGGER.info("Saving output to %s", output_json)
    with open(output_json, "w") as f:
        json.dump(video_name_to_output_text, f, indent=2)


if __name__ == "__main__":
    main()