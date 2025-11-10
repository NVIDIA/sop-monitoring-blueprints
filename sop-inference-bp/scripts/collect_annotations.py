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
A helper script to collect annotation information from the annotation tool to a single json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import logging

_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""
A helper script to collect annotation information from the annotation tool to a single json.

The output json has the below format:
    {
        "video_name0.mp4": <contents from the video_name0_annotation.json>,
        "video_name1.mp4": <contents from the video_name1_annotation.json>,
        ...
    }""")


    parser.add_argument("--anno_dir",
                        type=str,
                        required=True,
                        help="The directory contains *_annotation.json. "
                             "Note that glob pattern <anno_dir>/**/*_annotation.json is used recursively.")

    parser.add_argument("--output_json",
                        type=str,
                        required=True,
                        help="The output json filename.")

    parser.add_argument("--video_ext",
                        type=str,
                        default="mp4",
                        help="The file extension of the video file. We need this "
                             "since the file extension in annotation json is stripped. Default mp4")

    args = parser.parse_args()
    anno_dir = args.anno_dir
    output_json = args.output_json
    video_ext = args.video_ext

    logging.basicConfig(level=logging.INFO)

    annotation_suffix = "_annotation.json"

    glob_pattern = os.path.join(anno_dir, "**", f"*{annotation_suffix}")


    video_name_to_events = {}
    for filename in glob.glob(glob_pattern):
        _LOGGER.info("Porcessing %s", filename)

        anno_json_parent_dir = os.path.dirname(filename)
        video_name = os.path.basename(anno_json_parent_dir)
        video_name = f"{video_name}.{video_ext}"

        with open(filename) as fp:
            events = json.load(fp)

        video_name_to_events[video_name] = events

    _LOGGER.info("Writing out %s", output_json)
    with open(output_json, "w", encoding="utf-8") as fp:
        json.dump(video_name_to_events, fp, indent=2)

