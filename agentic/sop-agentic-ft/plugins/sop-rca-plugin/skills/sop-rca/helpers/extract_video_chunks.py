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

"""Extract video chunks from full SOP videos using DDM boundary timestamps.

Usage:
    python extract_video_chunks.py \
        --video <video_path> \
        --boundaries <f1_json_or_boundaries_json> \
        --output-dir <output_dir> \
        [--video-name <name_in_json>]

This is a helper for visual inspection of DDM-segmented chunks.
Requires ffmpeg to be installed.
"""

from typing import Dict, List, Optional
import json
from typing import Dict, List, Optional
import os
from typing import Dict, List, Optional
import subprocess
import sys
from pathlib import Path


def extract_chunks(
    video_path: str,
    boundaries: List[float],
    output_dir: str,
    video_name: Optional[str] = None,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    stem = video_name or Path(video_path).stem
    output_paths = []

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        duration = end - start

        out_path = os.path.join(
            output_dir, f"chunk_{i:03d}_{start:.2f}s_{end:.2f}s_{stem}.mp4"
        )

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-loglevel", "warning",
            out_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            output_paths.append(out_path)
            print(f"  Extracted: {out_path} ({duration:.2f}s)")
        except subprocess.CalledProcessError as e:
            print(f"  ERROR extracting chunk {i}: {e.stderr.decode()[:200]}")
        except FileNotFoundError:
            print("ERROR: ffmpeg not found. Install ffmpeg to use this tool.")
            return []

    return output_paths


def load_boundaries_from_json(json_path: str, video_name: str) -> Optional[List[float]]:
    with open(json_path) as f:
        data = json.load(f)

    # Try f1_*.json format (boundaries nested under video name with metric)
    if video_name in data and isinstance(data[video_name], dict):
        return data[video_name].get("boundaries")

    # Try video_to_boundaries_debug.json format (direct list)
    if video_name in data and isinstance(data[video_name], list):
        return data[video_name]

    # Try without extension
    name_no_ext = Path(video_name).stem
    for key in data:
        if Path(key).stem == name_no_ext:
            if isinstance(data[key], dict):
                return data[key].get("boundaries")
            if isinstance(data[key], list):
                return data[key]

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract video chunks using DDM boundaries"
    )
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument(
        "--boundaries",
        required=True,
        help="Path to JSON file with boundaries (f1_*.json or boundaries_debug.json)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--video-name",
        help="Video name key in JSON (default: basename of video path)",
    )

    args = parser.parse_args()

    video_name = args.video_name or Path(args.video).name
    boundaries = load_boundaries_from_json(args.boundaries, video_name)

    if boundaries is None:
        print(f"ERROR: Could not find boundaries for '{video_name}' in {args.boundaries}")
        sys.exit(1)

    print(f"Video: {args.video}")
    print(f"Boundaries ({len(boundaries)}): {boundaries}")
    print(f"Chunks: {len(boundaries) - 1}")
    print()

    paths = extract_chunks(args.video, boundaries, args.output_dir, video_name)
    print(f"\nExtracted {len(paths)} chunks to {args.output_dir}")
