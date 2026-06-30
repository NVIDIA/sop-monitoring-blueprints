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

"""
Frame Dropout Augmentation Module

Creates frame-dropout versions of videos for temporal diversity training.
Q&A content is preserved exactly -- only video files are modified.

Uses ffmpeg subprocess for video I/O to avoid OpenCV codec/numpy issues.
"""

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .utils.helper import create_dir, str2bool
from .utils.logger import logging

DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_ITERATIONS = 1

CORE_DATASETS = ["bcq", "mcq", "golden_gqa", "gqas", "spatial_localization"]


def _get_ffmpeg_exe() -> str:
    """Locate the ffmpeg binary (imageio_ffmpeg bundle or system PATH)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


_FFMPEG = _get_ffmpeg_exe()


def _get_video_info(video_path: str) -> Optional[Dict]:
    """Get video metadata using the ffmpeg binary (acts as ffprobe)."""
    try:
        cmd = [_FFMPEG, "-i", video_path, "-f", "null", "-"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        stderr = result.stderr

        duration = 0.0
        fps = 30.0

        dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", stderr)
        if dur_match:
            h, m, s, cs = dur_match.groups()
            duration = int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100.0

        fps_match = re.search(r"(\d{1,10}(?:\.\d{1,6})?)[ \t]*fps", stderr)
        if fps_match:
            fps = float(fps_match.group(1))

        total_frames = int(duration * fps) if duration > 0 and fps > 0 else 0

        if total_frames <= 0:
            logging.error(f"Could not parse video info from: {video_path}")
            return None

        return {
            "total_frames": total_frames,
            "fps": fps,
            "duration": duration,
        }
    except Exception as e:
        logging.error(f"Video probe failed for {video_path}: {e}")
        return None


def _build_keep_list(total_frames: int, effective_rate: float, seed: int = None) -> List[int]:
    """Select which frame indices to keep after dropout.

    Always retains the first and last frame.  Uses the ``random`` module
    for deterministic, non-security-sensitive augmentation.
    """
    if seed is not None:
        random.seed(seed)  # noqa: S311 nosec S2245 — deterministic augmentation, not security

    keep = [
        i for i in range(total_frames)
        if i == 0 or i == total_frames - 1 or random.random() > effective_rate  # noqa: S311 nosec S2245
    ]
    return keep if keep else [0]


def _run_ffmpeg_dropout(
    input_path: str, output_path: str, keep_indices: List[int],
) -> Tuple[bool, str]:
    """Execute ffmpeg with a ``select`` filter to retain only *keep_indices*.

    Returns ``(success, error_message)``.
    """
    select_parts = [f"eq(n\\,{n})" for n in keep_indices]
    select_expr = "+".join(select_parts)

    cmd = [
        _FFMPEG, "-y", "-i", input_path,
        "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
        "-an",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        return False, result.stderr[-500:]

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return False, "Output file empty or missing"

    return True, ""


def dropout_frames(
    input_path: str,
    output_path: str,
    dropout_rate: float,
    seed: int = None,
    adaptive: bool = True,
) -> Tuple[bool, Dict]:
    """Create a frame-dropped copy of a video using ffmpeg select filter.

    Generates a random keep/drop decision per frame, then uses ffmpeg's
    ``select`` filter to retain only the chosen frames.
    """
    try:
        info = _get_video_info(input_path)
        if info is None:
            logging.error(f"Cannot probe video: {input_path}")
            return False, {}

        total_frames = info["total_frames"]
        fps = info["fps"]

        if total_frames <= 0 or fps <= 0:
            logging.error(f"Invalid metadata ({total_frames} frames, {fps} fps): {input_path}")
            return False, {}

        effective_rate = dropout_rate
        if adaptive and info["duration"] < 1.5:
            effective_rate = dropout_rate * 0.3

        keep = _build_keep_list(total_frames, effective_rate, seed)

        success, err_msg = _run_ffmpeg_dropout(input_path, output_path, keep)
        if not success:
            logging.error(f"ffmpeg dropout failed for {input_path}: {err_msg}")
            return False, {}

        stats = {
            "original_frames": total_frames,
            "kept_frames": len(keep),
            "effective_dropout_rate": 1.0 - (len(keep) / total_frames) if total_frames > 0 else 0,
        }
        return True, stats

    except subprocess.TimeoutExpired:
        logging.error(f"ffmpeg timed out for {input_path}")
        return False, {}
    except Exception as e:
        logging.error(f"Frame dropout error for {input_path}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False, {}


def _resolve_json_file(input_dir: str) -> Optional[str]:
    """Return the first JSON file in *input_dir* (basename only), or None."""
    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    return json_files[0] if json_files else None


def _process_videos(
    processed_videos: Set[str],
    input_videos: str,
    output_videos: str,
    dropout_rate: float,
    seed: int,
) -> Tuple[Dict[str, str], Dict]:
    """Run frame dropout on each video and collect mapping + stats.

    Returns ``(video_mapping, stats)`` where *video_mapping* maps original
    filenames to their dropout counterparts.
    """
    video_mapping: Dict[str, str] = {}
    stats: Dict = {
        "unique_videos": len(processed_videos),
        "processed_videos": 0,
        "skipped_not_found": 0,
        "skipped_failed": 0,
        "avg_dropout_rate": [],
    }

    for video_name in sorted(processed_videos):
        input_video_path = os.path.join(input_videos, video_name)
        if not os.path.exists(input_video_path):
            stats["skipped_not_found"] += 1
            logging.warning(f"  Video not found: {input_video_path}")
            continue

        dropout_name = f"drop{int(dropout_rate * 100)}_{video_name}"
        output_video_path = os.path.join(output_videos, dropout_name)

        success, vstats = dropout_frames(
            input_video_path,
            output_video_path,
            dropout_rate,
            seed=seed + int(hashlib.sha256(video_name.encode()).hexdigest(), 16) % 10000,
            adaptive=True,
        )

        if success:
            video_mapping[video_name] = dropout_name
            stats["processed_videos"] += 1
            if vstats.get("effective_dropout_rate"):
                stats["avg_dropout_rate"].append(vstats["effective_dropout_rate"])
        else:
            stats["skipped_failed"] += 1

    return video_mapping, stats


def process_single_dataset(
    dataset_name: str,
    base_dir: str,
    output_suffix: str,
    dropout_rate: float,
    seed: int = 42,
) -> Dict:
    """Process one augmentation dataset folder, producing a frame-dropped copy."""
    input_dir = os.path.join(base_dir, dataset_name)
    output_dir = os.path.join(base_dir, f"{dataset_name}_{output_suffix}")

    logging.info(f"Frame drop: processing dataset '{dataset_name}' in {base_dir}")

    if not os.path.isdir(input_dir):
        logging.warning(f"Input dir not found, skipping: {input_dir}")
        return {}

    json_file = _resolve_json_file(input_dir)
    if json_file is None:
        logging.warning(f"No JSON file in {input_dir}")
        return {}

    input_json = os.path.join(input_dir, json_file)
    input_videos = os.path.join(input_dir, "videos")
    output_videos = os.path.join(output_dir, "videos")
    output_json = os.path.join(output_dir, f"{dataset_name}_{output_suffix}.json")

    logging.info(f"  input_json={input_json}, input_videos={input_videos}")

    if not os.path.exists(input_json):
        logging.warning(f"JSON not found: {input_json}")
        return {}
    if not os.path.isdir(input_videos):
        logging.warning(f"Videos dir not found: {input_videos}")
        return {}

    with open(input_json, "r") as f:
        data = json.load(f)

    logging.info(f"  Loaded {len(data)} entries from {json_file}")

    if not data:
        logging.warning(f"  JSON has 0 entries, skipping frame drop for {dataset_name}")
        create_dir(output_dir)
        with open(output_json, "w") as f:
            json.dump([], f)
        return {"dataset": dataset_name, "output_entries": 0}

    create_dir(output_videos)

    processed_videos: Set[str] = set()
    for entry in data:
        vname = Path(entry.get("video", "")).name
        if vname:
            processed_videos.add(vname)

    logging.info(f"  Found {len(processed_videos)} unique videos to process")

    video_mapping, stats = _process_videos(
        processed_videos, input_videos, output_videos, dropout_rate, seed,
    )
    stats["dataset"] = dataset_name

    new_data = []
    for i, entry in enumerate(data):
        vname = Path(entry.get("video", "")).name
        if vname in video_mapping:
            new_entry = entry.copy()
            new_entry["id"] = i
            new_entry["video"] = f"videos/{video_mapping[vname]}"
            new_data.append(new_entry)

    with open(output_json, "w") as f:
        json.dump(new_data, f, indent=2)

    avg = (
        sum(stats["avg_dropout_rate"]) / len(stats["avg_dropout_rate"])
        if stats["avg_dropout_rate"]
        else 0
    )
    stats["avg_dropout_rate"] = avg
    stats["output_entries"] = len(new_data)

    logging.info(
        f"Frame drop {dataset_name}: "
        f"{stats['processed_videos']}/{stats['unique_videos']} videos processed, "
        f"{stats['skipped_not_found']} not found, {stats['skipped_failed']} failed, "
        f"{stats['output_entries']} output entries"
    )
    return stats


def run_frame_drop_all(
    base_dir: str,
    datasets: List[str],
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = 42,
) -> List[Dict]:
    """Run frame drop for all specified datasets for N iterations."""
    all_stats: List[Dict] = []

    for iteration in range(1, iterations + 1):
        suffix = f"frame_drop_iter{iteration}"
        logging.info(f"Frame drop iteration {iteration}/{iterations}, suffix={suffix}")

        for dataset_name in datasets:
            stats = process_single_dataset(
                dataset_name, base_dir, suffix, dropout_rate, seed + iteration,
            )
            if stats:
                all_stats.append(stats)

    return all_stats


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frame dropout augmentation for dataset folders")
    parser.add_argument("--base-dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=CORE_DATASETS)
    parser.add_argument("--dropout-rate", type=float, default=DEFAULT_DROPOUT_RATE)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)  # noqa: S311 nosec S2245 — ML augmentation, not security
    stats = run_frame_drop_all(
        args.base_dir, args.datasets, args.dropout_rate, args.iterations, args.seed,
    )
    logging.info(f"Frame drop complete. Processed {len(stats)} dataset-iterations.")
