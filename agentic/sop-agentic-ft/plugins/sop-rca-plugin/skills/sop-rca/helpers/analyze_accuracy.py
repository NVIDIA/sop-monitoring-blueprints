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

"""Analyze end-to-end accuracy.json from SOP action recognition evaluation.

Usage:
    python analyze_accuracy.py <accuracy_json_path> [--output-dir <dir>]

Outputs a structured summary of:
- Overall metrics (sequence accuracy, action accuracy)
- Per-video error details
- Error type breakdown (wrong, duplicate, missing)
- Specific action IDs that are frequently missed or extra/hallucinated
- Confusion pairs (golden -> predicted) from "Wrong detection" steps,
  i.e. VLM substitution errors useful for Pattern 2 (Action Pair Confusion)
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# Match both step-string wordings emitted by accuracy.json across BP
# versions: legacy ("Missing detection: N", "Extra wrong detection: N",
# "Wrong detection: G is mis-understood as P") and current ("Missing:
# golden N at golden idx X", "Duplicate (extra): predicted N at pred idx
# X", "Wrong: golden G predicted as P at golden idx X").
RE_MISSING = re.compile(
    r"(?:Missing detection:\s*(\d+))|(?:Missing:\s+golden\s+(\d+))"
)
RE_EXTRA = re.compile(
    r"(?:Extra wrong detection:\s*(\d+))"
    r"|(?:Duplicate(?:\s*\(extra\))?:\s+predicted\s+(\d+))"
)
RE_WRONG = re.compile(
    r"(?:Wrong detection:\s*(\d+)\s+is mis-understood as\s+(\d+))"
    r"|(?:Wrong:\s+golden\s+(\d+)\s+predicted as\s+(\d+))"
)


def _get(d: dict, *keys, required: bool = True, default=None):
    """Return the first key present in d, or default. Used to bridge
    legacy/current field-name pairs in accuracy.json (e.g. seq_accuracy
    vs sequence_accuracy)."""
    for k in keys:
        if k in d:
            return d[k]
    if required:
        raise KeyError(f"none of {keys!r} present in accuracy.json")
    return default


def analyze_accuracy(accuracy_path: str) -> dict:
    with open(accuracy_path) as f:
        data = json.load(f)

    summary = {
        "total_videos": data["total_videos"],
        "seq_accuracy": _get(data, "sequence_accuracy", "seq_accuracy"),
        "total_actions": data["total_actions"],
        "action_accuracy": _get(data, "action_accuracy", "accuracy"),
        "error_counts": {
            "wrong": data["wrong"],
            "duplicate": data["duplicate"],
            "missing": data["missing"],
        },
        "num_failing_videos": len(data["videos_with_error"]),
        "failing_videos": [],
    }

    missing_actions = Counter()
    duplicate_actions = Counter()
    confusion_pairs = Counter()

    for entry in data["per_video"]:
        if entry["edit_distance"] == 0:
            continue

        video_info = {
            "video": entry["video"],
            "edit_distance": entry["edit_distance"],
            "golden": entry["golden"],
            "predicted": entry["predicted"],
            "wrong": entry["wrong"],
            "duplicate": entry["duplicate"],
            "missing": entry["missing"],
            "error_details": entry["steps"],
        }
        summary["failing_videos"].append(video_info)

        for step in entry["steps"]:
            m = RE_MISSING.search(step)
            if m:
                missing_actions[int(next(g for g in m.groups() if g))] += 1
                continue
            m = RE_EXTRA.search(step)
            if m:
                duplicate_actions[int(next(g for g in m.groups() if g))] += 1
                continue
            m = RE_WRONG.search(step)
            if m:
                groups = [g for g in m.groups() if g is not None]
                # legacy or current wording both end up as [golden, predicted]
                golden_id = int(groups[0])
                pred_id = int(groups[1])
                confusion_pairs[(golden_id, pred_id)] += 1

    summary["missing_action_counts"] = dict(
        sorted(missing_actions.items(), key=lambda x: -x[1])
    )
    summary["duplicate_action_counts"] = dict(
        sorted(duplicate_actions.items(), key=lambda x: -x[1])
    )
    summary["confusion_pair_counts"] = [
        {"golden": g, "predicted": p, "count": c}
        for (g, p), c in sorted(confusion_pairs.items(), key=lambda x: -x[1])
    ]

    return summary


def print_report(summary: dict) -> None:
    print("=" * 70)
    print("END-TO-END ACCURACY ANALYSIS")
    print("=" * 70)
    print(f"Sequence Accuracy: {summary['seq_accuracy']:.4f} "
          f"({summary['total_videos'] - summary['num_failing_videos']}"
          f"/{summary['total_videos']} videos correct)")
    print(f"Action Accuracy:   {summary['action_accuracy']:.4f} "
          f"({summary['total_actions']} total actions)")
    print(f"Errors: {summary['error_counts']['wrong']} wrong, "
          f"{summary['error_counts']['duplicate']} duplicate, "
          f"{summary['error_counts']['missing']} missing")
    print()

    if summary["missing_action_counts"]:
        print("Missing Actions (action_id: count):")
        for action_id, count in summary["missing_action_counts"].items():
            print(f"  Action {action_id}: {count}x")
        print()

    if summary["duplicate_action_counts"]:
        print("Extra / Hallucinated Actions (action_id: count):")
        for action_id, count in summary["duplicate_action_counts"].items():
            print(f"  Action {action_id}: {count}x")
        print()

    if summary["confusion_pair_counts"]:
        print("Confusion Pairs (golden -> predicted: count):")
        for pair in summary["confusion_pair_counts"]:
            print(f"  {pair['golden']} -> {pair['predicted']}: "
                  f"{pair['count']}x")
        print()

    print(f"Failing Videos ({summary['num_failing_videos']}):")
    print("-" * 70)
    for v in summary["failing_videos"]:
        print(f"  {v['video']} (edit_dist={v['edit_distance']})")
        print(f"    Golden:    {v['golden']}")
        print(f"    Predicted: {v['predicted']}")
        for step in v["error_details"]:
            print(f"    -> {step}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze end-to-end accuracy.json from SOP action "
                    "recognition evaluation."
    )
    parser.add_argument("accuracy_path", help="Path to accuracy.json")
    parser.add_argument(
        "--output-dir",
        help="Directory to save accuracy_analysis.json "
             "(defaults to the input file's parent directory)",
    )
    args = parser.parse_args()

    summary = analyze_accuracy(args.accuracy_path)
    print_report(summary)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(args.accuracy_path).parent
    out_path = out_dir / "accuracy_analysis.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Analysis saved to: {out_path}")


if __name__ == "__main__":
    main()
