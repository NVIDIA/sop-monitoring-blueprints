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

"""Analyze by-action chunk evaluation results for confusion patterns.

Usage:
    python analyze_by_action_confusion.py <eval_log_or_json> [--actions-json <actions_json>]

Supports two input formats:
1. JSON results file: {video: [[gt_action, pred_text, chunk_path], ...]}
2. Text log file: Parses VLM responses from log with "Action Chunk:" markers

Outputs:
- Per-action error rates
- Confusion matrix (GT -> Predicted)
- Identification of commonly confused action pairs
- Per-video error counts
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional


def parse_json_results(json_path: str) -> List[dict]:
    with open(json_path) as f:
        results = json.load(f)

    entries = []
    for video, preds in results.items():
        for gt_action, pred_text, chunk_path in preds:
            pred_actions = re.findall(r"\((\d+)\)", pred_text)
            pred_action = int(pred_actions[0]) if pred_actions else -1
            entries.append(
                {
                    "video": video,
                    "gt_action": gt_action,
                    "pred_action": pred_action,
                    "pred_text": pred_text,
                    "chunk_path": chunk_path,
                    "correct": pred_action == gt_action,
                }
            )
    return entries


def parse_log_results(log_path: str) -> List[dict]:
    """Parse by-action evaluation log file.

    The log contains entries like:
    Action Chunk: /path/to/<NN>_<video>_<idx>_<action>.mp4
    (N)action description text

    Where NN is the ground truth action prefix in the filename.
    """
    with open(log_path) as f:
        content = f.read()

    # Remove ANSI color codes
    content = re.sub(r"\033\[[0-9;]*m", "", content)

    entries = []
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Action Chunk:"):
            chunk_path = line.split("Action Chunk:")[1].strip()
            # Extract GT action from filename prefix (e.g., "07_..." -> action 7)
            basename = Path(chunk_path).stem
            gt_match = re.match(r"(\d+)_", basename)
            if gt_match:
                gt_action = int(gt_match.group(1))
            else:
                i += 1
                continue

            # Next line(s) contain the prediction
            i += 1
            if i < len(lines):
                pred_text = lines[i].strip()
                pred_actions = re.findall(r"\((\d+)\)", pred_text)
                pred_action = int(pred_actions[0]) if pred_actions else -1

                # Extract video name from path
                video_dir = Path(chunk_path).parent.name
                entries.append(
                    {
                        "video": video_dir,
                        "gt_action": gt_action,
                        "pred_action": pred_action,
                        "pred_text": pred_text,
                        "chunk_path": chunk_path,
                        "correct": pred_action == gt_action,
                    }
                )
        i += 1
    return entries


def analyze_confusion(entries: List[dict], actions_json_path: Optional[str] = None):
    action_names = {}
    if actions_json_path:
        with open(actions_json_path) as f:
            actions_data = json.load(f)
        for action in actions_data.get("actions", []):
            if isinstance(action, dict):
                # Format: [{"id": 1, "name": "..."}, ...]
                action_names[action["id"]] = action.get("short_name", action.get("name", ""))
            elif isinstance(action, str):
                # Format: ["(1)description", "(2)description", ...]
                m = re.match(r"\((\d+)\)\s*(.*)", action)
                if m:
                    action_names[int(m.group(1))] = m.group(2).strip().rstrip(".")

    total = len(entries)
    correct = sum(1 for e in entries if e["correct"])
    incorrect = [e for e in entries if not e["correct"]]

    # Per-action stats
    gt_counts = Counter(e["gt_action"] for e in entries)
    gt_errors = Counter(e["gt_action"] for e in incorrect)

    per_action = []
    for action_id in sorted(gt_counts.keys()):
        total_samples = gt_counts[action_id]
        errors = gt_errors.get(action_id, 0)
        per_action.append(
            {
                "action_id": action_id,
                "name": action_names.get(action_id, ""),
                "total_samples": total_samples,
                "errors": errors,
                "error_rate": errors / total_samples if total_samples > 0 else 0,
            }
        )

    # Confusion pairs
    confusion = Counter((e["gt_action"], e["pred_action"]) for e in incorrect)
    confusion_pairs = [
        {
            "gt_action": gt,
            "pred_action": pred,
            "count": count,
            "pct_of_errors": count / len(incorrect) * 100 if incorrect else 0,
        }
        for (gt, pred), count in sorted(confusion.items(), key=lambda x: -x[1])
    ]

    # Per-video error counts
    video_errors = Counter(e["video"] for e in incorrect)
    per_video = [
        {"video": v, "errors": c}
        for v, c in sorted(video_errors.items(), key=lambda x: -x[1])
    ]

    # Identify dominant confusion pattern
    dominant = confusion_pairs[0] if confusion_pairs else None

    return {
        "total_chunks": total,
        "correct": correct,
        "incorrect": len(incorrect),
        "accuracy": correct / total if total > 0 else 0,
        "per_action": per_action,
        "confusion_pairs": confusion_pairs,
        "dominant_confusion": dominant,
        "per_video_errors": per_video,
        "error_details": [
            {
                "video": e["video"],
                "gt_action": e["gt_action"],
                "pred_action": e["pred_action"],
                "pred_text": e["pred_text"][:100],
            }
            for e in incorrect
        ],
    }


def print_report(analysis: dict) -> None:
    print("=" * 70)
    print("BY-ACTION CHUNK CONFUSION ANALYSIS")
    print("=" * 70)
    print(f"Accuracy: {analysis['accuracy']:.4f} "
          f"({analysis['correct']}/{analysis['total_chunks']})")
    print(f"Total errors: {analysis['incorrect']}")
    print()

    print("Per-Action Error Rates:")
    for a in analysis["per_action"]:
        if a["errors"] > 0:
            print(f"  Action {a['action_id']:2d}: {a['errors']}/{a['total_samples']} "
                  f"({a['error_rate']:.1%}) {a['name']}")
    print()

    zero_error_actions = [a for a in analysis["per_action"] if a["errors"] == 0]
    if zero_error_actions:
        ids = [str(a["action_id"]) for a in zero_error_actions]
        print(f"  Actions with ZERO errors: {', '.join(ids)}")
        print()

    print("Confusion Pairs (GT -> Predicted):")
    for cp in analysis["confusion_pairs"]:
        print(f"  Action {cp['gt_action']:2d} -> Action {cp['pred_action']:2d}: "
              f"{cp['count']} errors ({cp['pct_of_errors']:.1f}%)")
    print()

    if analysis["dominant_confusion"]:
        dc = analysis["dominant_confusion"]
        print(f"DOMINANT CONFUSION: Action {dc['gt_action']} -> Action {dc['pred_action']} "
              f"({dc['count']} errors, {dc['pct_of_errors']:.1f}% of all errors)")
    print()

    if analysis["per_video_errors"]:
        print("Per-Video Error Counts:")
        for v in analysis["per_video_errors"][:10]:
            print(f"  {v['video']}: {v['errors']} errors")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze by-action chunk evaluation results for "
                    "confusion patterns."
    )
    parser.add_argument(
        "input_path",
        help="Path to the by-action chunk evaluation log "
             "(.json or .txt — input format is auto-detected by extension)",
    )
    parser.add_argument(
        "--actions-json", default=None,
        help="Path to actions.json (provides action names for the report)",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save confusion_analysis.json "
             "(defaults to the input file's parent directory)",
    )
    args = parser.parse_args()

    if args.input_path.endswith(".json"):
        entries = parse_json_results(args.input_path)
    else:
        entries = parse_log_results(args.input_path)

    analysis = analyze_confusion(entries, args.actions_json)
    print_report(analysis)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(args.input_path).parent
    out_path = out_dir / "confusion_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to: {out_path}")


if __name__ == "__main__":
    main()
