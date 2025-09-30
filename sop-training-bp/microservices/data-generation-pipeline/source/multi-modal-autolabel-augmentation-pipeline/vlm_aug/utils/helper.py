######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
######################################################################################################


import argparse
import json
import os
import re
from typing import List


def create_dir(path: str):
    """Create directory if not exist

    Args:
        path (str): directory to be created
    """
    if not os.path.exists(path):
        print(f"Create {path}")
        os.makedirs(path, exist_ok=True)


def read_txt(path: str) -> str:
    """read txt file

    Args:
        path (str): txt file path

    Returns:
        str: text
    """
    with open(path, "r") as f:
        text = f.read()

    return text


def write_txt(path: str, content: str):
    """write txt file

    Args:
        path (str): txt file path
        content (str): content to be write
    """
    with open(path, "w") as f:
        f.write(content)


def write_frames(cap, out):
    """Write frames

    Args:
        cap (cv2.VideoCapture): cv2 VideoCapture object
        out (cv2.VideoWriter): cv2 VideoWriter object
    """
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)


def read_json(path: str) -> dict:
    """read json file

    Args:
        path (str): json file path

    Returns:
        dict: json content
    """

    with open(path, "r") as f:
        json_file = json.load(f)

    return json_file


def dump_json(output_path: str, obj):
    """dump json object

    Args:
        output_path (str): path to dump json
        obj: object to be dump
    """
    with open(output_path, "w") as fp:
        json.dump(obj, fp)


def clean_sentence(sentence: str) -> str:
    """Clear up sentence head and tail by removing any special symbols or numbers

    Args:
        sentence (str): input sentence

    Returns:
        str: cleaned sentence
    """

    return re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", sentence).strip()


def str2bool(v) -> bool:
    """convert string to boolean

    Args:
        v (Union[str | bool]): input option

    Raises:
        argparse.ArgumentTypeError

    Returns:
        bool: boolean value
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def unpack_annotation(annotations: List[List[dict]]) -> List[dict]:
    """unpack annotation and make sure id start from start_id

    Args:
        annotations (List[dict]): annotation in llava format

    Returns:
        List[dict]: re-id annotations
    """
    i = 0
    final_anns = []
    for ann in annotations:
        for qa in ann:
            qa["id"] = i
            i += 1
        final_anns.extend(ann)

    return final_anns


def custom_sort_key(filename: str, keyword: str, ext: str, sep: str = "_") -> tuple:
    """A custom sort key function

    Args:
        filename (str): file name to be parse
        keyword (str): keyword to be replaced with empty string
        ext (str): file extention
        sep (str): seperator

    Returns:
        tuple: sorting criteria
    """

    # Remove extension and split by seperator
    base, *suffix = filename.replace(ext, "").split(sep)

    # Extract the numeric part of the base (e.g., "action1" -> 1)
    action_number = int(base.replace(keyword, ""))

    # Handle suffix sorting (e.g., "_2" becomes 2)
    suffix_number = int(suffix[0]) if suffix else 0

    return (action_number, suffix_number)
