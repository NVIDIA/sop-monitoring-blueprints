######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
######################################################################################################


import argparse
import ipaddress
import json
import os
import re
import socket
from urllib.parse import urlparse, urlunparse
import cv2
import numpy as np
from typing import List, Tuple
from moviepy.editor import VideoFileClip, ImageSequenceClip


def llm_url_env_allows_private() -> bool:
    """Whether ALLOW_PRIVATE_LLM_URL opts in to private/loopback LLM endpoints."""
    return os.getenv("ALLOW_PRIVATE_LLM_URL", "").strip().lower() in ("1", "true", "yes", "on")


def validate_egress_url(url: str, allow_private: bool = False) -> str:
    """Validate an outbound LLM ``base_url`` to mitigate SSRF (FSR-NET-2 / T10).

    Enforces an ``http(s)`` scheme with a resolvable host, and rejects
    link-local (incl. the 169.254.169.254 cloud instance-metadata endpoint),
    multicast, reserved and unspecified addresses **in all cases**. Private and
    loopback addresses are rejected unless ``allow_private`` is set — the local
    LLM workflow opts in either by selecting ``llm_type='local'`` or by setting
    the ``ALLOW_PRIVATE_LLM_URL`` environment variable, so the agentic / local
    vLLM path keeps working while the metadata-SSRF crown jewel stays blocked.

    Anti-rebinding (TOCTOU): when the host is a DNS name, the validated IP is
    pinned into the returned ``http`` URL so the client connects to the address
    we actually checked rather than re-resolving (which a rebinding attacker
    could point elsewhere between this check and the request). IP-literal hosts
    are returned unchanged (nothing to re-resolve); ``https`` hosts are left
    as-is to preserve TLS SNI / certificate validation (residual, low — the
    local LLM path is plain http).

    Returns the (possibly IP-pinned) URL when valid; raises ``ValueError`` otherwise.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"local_llm_url must be an http(s) URL with a host: {url!r}")

    try:
        addrinfos = socket.getaddrinfo(parsed.hostname, parsed.port)
    except socket.gaierror as exc:
        raise ValueError(f"local_llm_url host could not be resolved: {parsed.hostname!r} ({exc})")

    validated_ips = []
    for *_, sockaddr in addrinfos:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            raise ValueError(
                f"local_llm_url resolves to a blocked address ({ip}); refusing for SSRF safety"
            )
        if (ip.is_private or ip.is_loopback) and not allow_private:
            raise ValueError(
                f"local_llm_url resolves to a private/loopback address ({ip}); "
                "use llm_type='local' or set ALLOW_PRIVATE_LLM_URL=true to permit this"
            )
        validated_ips.append(ip)

    # IP-literal host: nothing is re-resolved at connect, so no rebinding window.
    try:
        ipaddress.ip_address(parsed.hostname)
        return url
    except ValueError:
        pass

    # DNS-name host: pin the validated IP into http URLs so the client connects
    # to the checked address (closes the rebinding TOCTOU). Leave https as-is to
    # keep TLS SNI / cert validation working (documented residual).
    if parsed.scheme != "http":
        return url
    pinned = validated_ips[0]
    host = f"[{pinned}]" if pinned.version == 6 else str(pinned)
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    if parsed.username:
        userinfo = parsed.username + (f":{parsed.password}" if parsed.password else "")
        netloc = f"{userinfo}@{netloc}"
    return urlunparse(parsed._replace(netloc=netloc))


def resolve_llm_base_url(llm_type: str, local_llm_url: str, api_base_url: str) -> str:
    """Return the OpenAI-client ``base_url`` for the configured LLM backend.

    NVIDIA NIM uses the fixed public ``api_base_url``; the local path validates
    ``local_llm_url`` for SSRF (FSR-NET-2). Private/loopback endpoints are only
    permitted for ``llm_type == 'local'`` or via ``ALLOW_PRIVATE_LLM_URL`` — the
    agentic / local vLLM workflow — while link-local/metadata stays blocked.
    """
    if llm_type == "nvidia":
        return api_base_url
    allow_private = llm_type == "local" or llm_url_env_allows_private()
    return validate_egress_url(local_llm_url, allow_private=allow_private)


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


# ============================================================================
# Two-Operator Mode Helper Functions
# ============================================================================

# Separator for concurrent action indices in filename (e.g., "01-03" uses "-")
CONCURRENT_ACTION_SEP = "-"


def parse_video_action_indices(
    video_basename: str,
    action_sep: str = "_",
    two_operator_mode: bool = False
) -> Tuple[List[int], bool]:
    """Parse action indices from video filename, handling both single and concurrent formats.

    Args:
        video_basename (str): Video filename without extension (e.g., "01_video" or "01-03_video")
        action_sep (str): Separator between action part and "video" keyword (default "_")
        two_operator_mode (bool): Whether two-operator mode is enabled

    Returns:
        Tuple[List[int], bool]: (list of action indices, is_concurrent flag)

    Behavior:
        - two_operator_mode=False: Only parses single action format, skips concurrent
        - two_operator_mode=True: Parses both single and concurrent formats

    Examples:
        two_operator_mode=False:
            "01_video" -> ([1], False)
            "01-03_video" -> ([], False)  # Skipped - invalid in single-op mode

        two_operator_mode=True:
            "01_video" -> ([1], False)
            "01-03_video" -> ([1, 3], True)
            "02-05-07_video" -> ([2, 5, 7], True)
    """
    # Extract the action part before "_video"
    action_part = video_basename.split(action_sep)[0]

    # Check if it contains concurrent action separator
    if CONCURRENT_ACTION_SEP in action_part:
        if two_operator_mode:
            # Concurrent actions: "01-03" -> [1, 3]
            action_indices = [int(idx) for idx in action_part.split(CONCURRENT_ACTION_SEP)]
            return action_indices, True
        else:
            # Two-operator mode is OFF - skip concurrent videos
            return [], False
    else:
        # Single action: "01" -> [1]
        return [int(action_part)], False


def format_concurrent_actions(action_descriptions: List[str], action_indices: List[int]) -> str:
    """Format multiple action descriptions for concurrent actions.

    Args:
        action_descriptions (List[str]): List of action description strings
        action_indices (List[int]): List of action indices (1-based)

    Returns:
        str: Formatted string like "(1) picking up item (3) inspecting label"
    """
    formatted_parts = []
    for idx, desc in zip(action_indices, action_descriptions):
        # Clean and lowercase the description
        cleaned = clean_sentence(desc)
        cleaned = cleaned[0].lower() + cleaned[1:] if cleaned else cleaned
        formatted_parts.append(f"({idx}) {cleaned}")

    return " ".join(formatted_parts)


def get_video_meta(video_path: str) -> Tuple[int, float, Tuple[int, int]]:
    """
    Get video meta data using MoviePy for robust codec support; fallback to OpenCV if needed.

    Args:
        video_path (str): path to the video

    Returns:
        Tuple[int, float, Tuple[int, int]]: frame count, fps, and video size

    Raises:
        RuntimeError: if the video cannot be opened
    """
    # Try MoviePy first
    try:
        with VideoFileClip(video_path) as clip:
            fps = float(clip.fps) if clip.fps else 30.0
            # Some containers may report nframes as 0; compute from duration as a fallback.
            duration = float(clip.duration) if clip.duration else 0.0
            nframes_reader = getattr(getattr(clip, "reader", None), "nframes", 0) or 0
            frame_count = int(round(duration * fps)) if duration > 0 else int(nframes_reader)
            size = (int(clip.w), int(clip.h))
            return frame_count, fps, size
    except Exception:
        pass

    # Fallback to OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frame_count, fps, (width, height)


def write_video(frames: List[np.ndarray], output_path: str, fps: int, size: Tuple[int, int]) -> None:
    """
    Write video using MoviePy (libx264). Expects frames in RGB order.

    Args:
        frames (List[np.ndarray]): frames to be written
        output_path (str): path to write video
        fps (int): fps of the video
        size (Tuple[int, int]): size of the video

    Raises:
        RuntimeError: if no frames to write
    """
    if not frames:
        raise RuntimeError(f"No frames to write for: {output_path}")
    w, h = size
    normalized_frames: List[np.ndarray] = []
    for f in frames:
        if (f.shape[1], f.shape[0]) != (w, h):
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        normalized_frames.append(f)
    clip = ImageSequenceClip(normalized_frames, fps=fps)
    clip.write_videofile(output_path, fps=fps, codec="libx264", audio=False, verbose=False, logger=None)
    clip.close()
