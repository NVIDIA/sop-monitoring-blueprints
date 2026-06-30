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

"""Analyze VLM output per chunk from end-to-end evaluation.

Usage:
    python analyze_vlm_output.py <video_name_to_output_text_json> \
        [--fps N] [--max-frames N] [--non-sop-action N] \
        [--golden-boundaries <path>] [--long-ratio N] [--short-ratio N]

Analyzes:
- Multi-action outputs (chunks where VLM predicted >1 action)
- Chunk duration distribution with data-driven thresholds
- Frame sampling analysis (effective frames per chunk at given fps/max_frames)
- Per-action prediction frequency

Thresholds:
- Long threshold: (max_frames / fps) * long_ratio. Chunks beyond this duration
  experience frame subsampling. Default long_ratio=3.0.
- Short threshold: min_golden_action_duration * short_ratio. Chunks shorter than
  this may lack sufficient information. Default short_ratio=0.8.
  Requires --golden-boundaries. If not provided, no short threshold is applied.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def compute_golden_min_action_duration(golden_boundaries_path):
    """Compute the minimum action duration from golden boundaries.

    Excludes the first and last segments of each video (typically idle periods).
    """
    with open(golden_boundaries_path) as f:
        golden = json.load(f)

    action_durations = []
    for _, boundaries in golden.items():
        if not isinstance(boundaries, list) or len(boundaries) < 3:
            continue
        # Exclude first segment (boundaries[0]-boundaries[1]) and
        # last segment (boundaries[-2]-boundaries[-1]) as they are
        # typically idle/transition periods
        for i in range(1, len(boundaries) - 2):
            dur = boundaries[i + 1] - boundaries[i]
            if dur > 0:
                action_durations.append(dur)

    if not action_durations:
        return None
    return min(action_durations)


def parse_vlm_output(vlm_output_path, fps=8, max_frames=40):
    with open(vlm_output_path) as f:
        vlm_output = json.load(f)

    all_chunks = []
    multi_action_chunks = []

    for video_name, chunks in vlm_output.items():
        for time_key, output_text in chunks.items():
            m = re.match(r"\[(\d+\.\d+)s-(\d+\.\d+)s\]", time_key)
            if not m:
                continue
            start = float(m.group(1))
            end = float(m.group(2))
            duration = end - start

            actions = re.findall(r"\((\d+)\)", output_text)
            num_actions = len(actions)
            total_frames = int(duration * fps)
            effective_frames = min(total_frames, max_frames)

            chunk = {
                "video": video_name,
                "time_range": time_key,
                "start": start,
                "end": end,
                "duration": duration,
                "num_actions": num_actions,
                "actions": [int(a) for a in actions],
                "total_frames": total_frames,
                "effective_frames": effective_frames,
                "output_text": output_text,
            }
            all_chunks.append(chunk)
            if num_actions > 1:
                multi_action_chunks.append(chunk)

    return all_chunks, multi_action_chunks


def analyze_chunks(all_chunks, multi_action_chunks, fps=8, max_frames=40,
                   non_sop_action=None, short_threshold=None, long_threshold=None):
    durations = [c["duration"] for c in all_chunks]
    action_freq = Counter()
    for c in all_chunks:
        for a in c["actions"]:
            action_freq[a] += 1

    # Classify chunks by duration thresholds
    short_chunks = []
    if short_threshold is not None:
        short_chunks = [c for c in all_chunks if c["duration"] < short_threshold]

    long_chunks = []
    if long_threshold is not None:
        long_chunks = [c for c in all_chunks if c["duration"] > long_threshold]

    # Identify problematic long chunks
    # If non_sop_action is provided, filter to those containing actual SOP actions
    # If not provided, report ALL long chunks and let the agent judge
    problematic_long = []
    for c in long_chunks:
        if non_sop_action is not None:
            has_sop_action = any(a != non_sop_action for a in c["actions"])
            if has_sop_action and c["num_actions"] >= 1:
                problematic_long.append(c)
        else:
            if c["num_actions"] >= 1:
                problematic_long.append(c)

    return {
        "total_chunks": len(all_chunks),
        "multi_action_chunks": len(multi_action_chunks),
        "multi_action_pct": len(multi_action_chunks) / len(all_chunks) * 100
        if all_chunks
        else 0,
        "short_threshold": round(short_threshold, 2) if short_threshold else None,
        "long_threshold": round(long_threshold, 2) if long_threshold else None,
        "short_chunks": len(short_chunks),
        "long_chunks": len(long_chunks),
        "problematic_long_chunks": len(problematic_long),
        "duration_stats": {
            "min": min(durations) if durations else 0,
            "max": max(durations) if durations else 0,
            "mean": sum(durations) / len(durations) if durations else 0,
        },
        "action_frequency": dict(sorted(action_freq.items())),
        "fps": fps,
        "max_frames": max_frames,
        "multi_action_details": [
            {
                "video": c["video"],
                "time_range": c["time_range"],
                "duration": round(c["duration"], 1),
                "actions": c["actions"],
                "effective_frames": c["effective_frames"],
            }
            for c in sorted(multi_action_chunks, key=lambda x: -x["duration"])
        ],
        "short_chunk_details": [
            {
                "video": c["video"],
                "time_range": c["time_range"],
                "duration": round(c["duration"], 2),
                "actions": c["actions"],
                "effective_frames": c["effective_frames"],
                "output_text": c["output_text"][:120],
            }
            for c in sorted(short_chunks, key=lambda x: x["duration"])
        ],
        "problematic_long_details": [
            {
                "video": c["video"],
                "time_range": c["time_range"],
                "duration": round(c["duration"], 1),
                "actions": c["actions"],
                "effective_frames": c["effective_frames"],
                "output_text": c["output_text"][:120],
            }
            for c in sorted(problematic_long, key=lambda x: -x["duration"])
        ],
    }


def print_report(analysis):
    print("=" * 70)
    print("VLM OUTPUT CHUNK ANALYSIS")
    print("=" * 70)
    print(f"Total chunks: {analysis['total_chunks']}")
    print(f"Multi-action chunks: {analysis['multi_action_chunks']} "
          f"({analysis['multi_action_pct']:.1f}%)")

    st = analysis["short_threshold"]
    lt = analysis["long_threshold"]
    if st is not None:
        print(f"Short chunks (<{st}s): {analysis['short_chunks']}")
    else:
        print("Short chunks: N/A (no --golden-boundaries provided)")
    if lt is not None:
        print(f"Long chunks (>{lt}s): {analysis['long_chunks']}")
        print(f"Problematic long chunks (non-idle, >{lt}s): "
              f"{analysis['problematic_long_chunks']}")
    else:
        print("Long chunks: N/A (thresholds not computed)")

    print(f"Duration: min={analysis['duration_stats']['min']:.1f}s, "
          f"max={analysis['duration_stats']['max']:.1f}s, "
          f"mean={analysis['duration_stats']['mean']:.1f}s")
    print(f"Config: fps={analysis['fps']}, max_frames={analysis['max_frames']}")
    if st is not None:
        print(f"Short threshold: {st}s (0.8 * min golden action duration)")
    if lt is not None:
        print(f"Long threshold: {lt}s (max_frames/fps * long_ratio)")
    print()

    print("Action Prediction Frequency:")
    for action_id, count in analysis["action_frequency"].items():
        print(f"  Action {action_id:2d}: {count}")
    print()

    if analysis["short_chunk_details"]:
        print(f"SHORT Chunks ({len(analysis['short_chunk_details'])}):")
        print("-" * 70)
        for c in analysis["short_chunk_details"][:20]:
            print(f"  {c['video']} {c['time_range']} "
                  f"({c['duration']}s, {c['effective_frames']} frames)")
            print(f"    Actions: {c['actions']}")
            print(f"    Output: {c['output_text']}")
        print()

    if analysis["multi_action_details"]:
        print(f"Multi-Action Chunks ({len(analysis['multi_action_details'])}):")
        print("-" * 70)
        for c in analysis["multi_action_details"][:20]:
            print(f"  {c['video']} {c['time_range']}")
            print(f"    Duration: {c['duration']}s, "
                  f"Effective frames: {c['effective_frames']}, "
                  f"Actions: {c['actions']}")
        print()

    if analysis["problematic_long_details"]:
        lt_str = f">{analysis['long_threshold']}s" if analysis["long_threshold"] else ""
        print(f"PROBLEMATIC Long Chunks ({lt_str}):")
        print("-" * 70)
        for c in analysis["problematic_long_details"]:
            print(f"  {c['video']} {c['time_range']} "
                  f"({c['duration']}s, eff={c['effective_frames']} frames)")
            print(f"    Actions: {c['actions']}")
            print(f"    Output: {c['output_text']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze VLM output per chunk from end-to-end evaluation."
    )
    parser.add_argument(
        "vlm_output_path",
        help="Path to video_name_to_output_text.json",
    )
    parser.add_argument(
        "--fps", type=int, default=8,
        help="Frame sampling rate (default: 8, matches the E2E pipeline's "
             "hard-coded value)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=40,
        help="Max frames per chunk (default: 40)",
    )
    parser.add_argument(
        "--non-sop-action", type=int, default=None,
        help="Action ID for the non-SOP catch-all action",
    )
    parser.add_argument(
        "--golden-boundaries", default=None,
        help="Path to golden boundaries JSON; required for short-threshold "
             "computation",
    )
    parser.add_argument(
        "--long-ratio", type=float, default=3.0,
        help="Long chunk threshold ratio applied to max_frames/fps "
             "(default: 3.0)",
    )
    parser.add_argument(
        "--short-ratio", type=float, default=0.8,
        help="Short chunk threshold ratio applied to min golden action "
             "duration (default: 0.8)",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save vlm_output_analysis.json "
             "(defaults to the input file's parent directory)",
    )
    args = parser.parse_args()

    long_threshold = (args.max_frames / args.fps) * args.long_ratio

    short_threshold = None
    if args.golden_boundaries:
        min_golden_dur = compute_golden_min_action_duration(
            args.golden_boundaries
        )
        if min_golden_dur is not None:
            short_threshold = min_golden_dur * args.short_ratio
            print(f"Golden min action duration: {min_golden_dur:.2f}s")
            print(f"Short threshold ({args.short_ratio}x): "
                  f"{short_threshold:.2f}s")
        else:
            print("WARNING: Could not compute min golden action duration")
    print(f"Long threshold (max_frames/fps * {args.long_ratio}): "
          f"{long_threshold:.2f}s")
    print()

    all_chunks, multi_action = parse_vlm_output(
        args.vlm_output_path, args.fps, args.max_frames
    )
    analysis = analyze_chunks(
        all_chunks, multi_action, args.fps, args.max_frames,
        args.non_sop_action, short_threshold, long_threshold,
    )
    print_report(analysis)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(args.vlm_output_path).parent
    out_path = out_dir / "vlm_output_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to: {out_path}")


if __name__ == "__main__":
    main()
