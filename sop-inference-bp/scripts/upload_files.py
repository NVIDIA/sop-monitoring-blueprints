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
A script to upload the files to the inference blueprint.
"""
from __future__ import annotations

import argparse
import json
import os
import logging

from openai import OpenAI

_LOGGER = logging.getLogger(__name__)

def main() -> None:

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Upload the files to the inference blueprint.")
    parser.add_argument("-i", "--video_dir", type=str, required=True,
                        help="Path to the directory containing the files to upload.")
    parser.add_argument("-o", "--output_json", type=str, required=True,
                        help="Name of the output JSON file")
    parser.add_argument("-l", "--label", type=str, required=True, choices=["pass", "fail"],
                        help="Label of the files to upload.")
    parser.add_argument("-e", "--video_ext", type=str, required=False,
                        default="mp4",
                        help="Extension of the files to upload. Default is mp4.")
    parser.add_argument("-b", "--base_url", type=str, required=False,
                        default="http://localhost:8080/v1",
                        help="Base URL of the inference blueprint. Default is http://localhost:8080/v1.")
    args = parser.parse_args()

    video_dir = args.video_dir
    output_json = args.output_json
    video_ext = args.video_ext
    label = args.label
    base_url = args.base_url

    client = OpenAI(base_url=base_url, api_key="don't care.", max_retries=1)

    video_infos = {}
    for entry in os.scandir(video_dir):
        if entry.is_file() and entry.name.endswith(f".{video_ext}"):
            _LOGGER.info(f"Uploading {entry.name}")
            with open(entry.path, "rb") as fp:
                uploaded_file = client.files.create(
                    file=fp,
                    purpose="vision",
                )
            video_infos[entry.name] = {
                "file_id": uploaded_file.id,
                "original_filepath": entry.path,
                "label": label,
            }

    _LOGGER.info(f"Uploaded {len(video_infos)} files")
    _LOGGER.info(f"Saving to {output_json}")
    with open(output_json, "w") as fp:
        json.dump(video_infos, fp, indent=2)

if __name__ == "__main__":
    main()
