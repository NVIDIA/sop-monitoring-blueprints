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

"""Analyze VLM fine-tuning training data distribution and quality.

Usage:
    python analyze_training_data.py <augmented_data_root> [--actions-json <path>]

Where augmented_data_root contains subdirectories like:
    <video_set>_augmented_0/
        bcq/bcq.json
        mcq/mcq.json
        dmcq/dmcq_*.json
        ...

Analyzes:
- Total sample counts per QA type
- Action distribution across QA types
- DMCQ answer balance (non-SOP action vs SOP actions)
- MCQ max_chunk_len (multi-action coverage)
- Training data adequacy (number of SOP cycles)
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional


# No hardcoded action patterns — must be loaded from actions.json via --actions-json


def load_actions_from_json(actions_json_path: str) -> dict:
    """Load action name -> ID mapping from actions.json.

    Supports two formats:
    - List of dicts: [{"id": 1, "name": "..."}, ...]
    - List of strings: ["(1)description", "(2)description", ...]
    """
    with open(actions_json_path) as f:
        data = json.load(f)
    mapping = {}
    for action in data.get("actions", []):
        if isinstance(action, dict):
            name = action.get("name", "").lower().strip().rstrip(".")
            mapping[name] = action["id"]
        elif isinstance(action, str):
            m = re.match(r"\((\d+)\)\s*(.*)", action)
            if m:
                name = m.group(2).strip().rstrip(".").lower()
                mapping[name] = int(m.group(1))
    return mapping


def map_dmcq_answer_to_sop_action(
    answer_text: str, action_patterns: Optional[dict] = None
) -> Optional[int]:
    """Map DMCQ dynamic answer text to SOP action ID."""
    if not action_patterns:
        return None
    patterns = action_patterns
    m = re.match(r"\(\d+\)\s*(.*)", answer_text.strip())
    if not m:
        return None
    text = m.group(1).strip().rstrip(".")
    for pattern, action_id in patterns.items():
        if text.lower() == pattern.lower():
            return action_id
    return None


def find_action_ids_in_text(
    answer_text: str, action_patterns: Optional[dict] = None
) -> list:
    """Find SOP action IDs whose canonical description appears as a substring of the answer.

    Used for BCQ / golden_gqa / gqas answers where the action is described in
    natural language without a parenthesized ID prefix. May miss samples where
    the answer paraphrases the action (especially common in LLM-generated
    GQAs samples).
    """
    if not action_patterns or not answer_text:
        return []
    text_lower = answer_text.lower()
    return [
        action_id
        for pattern, action_id in action_patterns.items()
        if pattern and pattern.lower() in text_lower
    ]


def analyze_augmented_data(
    augmented_root: str, actions_json_path: Optional[str] = None,
    non_sop_action_id: Optional[int] = None
):
    action_patterns = {}
    if actions_json_path:
        action_patterns = load_actions_from_json(actions_json_path)

    augmented_root = Path(augmented_root)
    aug_sets = sorted(
        [d for d in augmented_root.iterdir() if d.is_dir() and "augmented" in d.name]
    )

    if not aug_sets:
        # Try treating root as a single augmented set
        aug_sets = [augmented_root]

    total_samples = 0
    qa_type_counts = Counter()
    action_dist_mcq = Counter()      # MCQ — direct (N) extraction (mirrors eval format)
    action_dist_bcq_gqa = Counter()  # BCQ/golden_gqa/gqas — text matched against actions.json
    action_dist_dmcq = Counter()     # DMCQ — text matched against actions.json
    mcq_multi_action = 0
    mcq_total = 0
    max_actions_in_mcq = 0
    num_sop_cycles = 0

    for aug_set in aug_sets:
        for qa_dir in sorted(aug_set.iterdir()):
            if not qa_dir.is_dir():
                continue
            qa_type = qa_dir.name
            for json_file in qa_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                if not isinstance(data, list):
                    continue

                count = len(data)
                total_samples += count
                qa_type_counts[f"{qa_type}/{json_file.name}"] += count

                for sample in data:
                    if not isinstance(sample, dict) or "conversations" not in sample:
                        continue
                    convs = sample["conversations"]
                    if len(convs) < 2:
                        continue
                    answer = convs[1].get("value", "")

                    if qa_type == "mcq":
                        # MCQ lists every option with its canonical SOP ID prefix
                        actions = [int(a) for a in re.findall(r"\((\d+)\)", answer)]
                        for a in actions:
                            action_dist_mcq[a] += 1
                        mcq_total += 1
                        if len(actions) >= 2:
                            mcq_multi_action += 1
                        max_actions_in_mcq = max(max_actions_in_mcq, len(actions))

                    elif qa_type in ("bcq", "golden_gqa", "gqas"):
                        # Free-form natural language; match against actions.json descriptions
                        for a in find_action_ids_in_text(answer, action_patterns):
                            action_dist_bcq_gqa[a] += 1

                    elif qa_type == "dmcq":
                        sop_action = map_dmcq_answer_to_sop_action(
                            answer, action_patterns
                        )
                        if sop_action is not None:
                            action_dist_dmcq[sop_action] += 1

    # Count SOP cycles from annotation directories
    annotation_root = augmented_root.parent
    for d in annotation_root.iterdir():
        if d.is_dir() and "annotation" in d.name.lower():
            for sub in d.rglob("*_annotation.json"):
                num_sop_cycles += 1

    return {
        "num_augmented_sets": len(aug_sets),
        "total_samples": total_samples,
        "qa_type_counts": dict(sorted(qa_type_counts.items())),
        "action_dist_mcq": dict(sorted(action_dist_mcq.items())),
        "action_dist_bcq_gqa": dict(sorted(action_dist_bcq_gqa.items())),
        "action_dist_dmcq": dict(sorted(action_dist_dmcq.items())),
        "mcq_multi_action_count": mcq_multi_action,
        "mcq_total": mcq_total,
        "mcq_multi_action_pct": mcq_multi_action / mcq_total * 100
        if mcq_total > 0
        else 0,
        "max_actions_in_mcq": max_actions_in_mcq,
        "non_sop_action_id": non_sop_action_id,
        "dmcq_non_sop_pct": (
            action_dist_dmcq.get(non_sop_action_id, 0) / sum(action_dist_dmcq.values()) * 100
            if action_dist_dmcq and non_sop_action_id is not None
            else 0
        ),
    }


def print_report(analysis: dict) -> None:
    print("=" * 70)
    print("TRAINING DATA ANALYSIS")
    print("=" * 70)
    print(f"Augmented sets: {analysis['num_augmented_sets']}")
    print(f"Total samples: {analysis['total_samples']}")
    print()

    print("QA Type Counts:")
    for qt, count in analysis["qa_type_counts"].items():
        print(f"  {qt}: {count}")
    print()

    print("Action Distribution (MCQ - direct (N) extraction, eval format):")
    for a, count in analysis["action_dist_mcq"].items():
        print(f"  Action {a:2d}: {count}")
    print()

    print("Action Distribution (BCQ/golden_gqa/gqas - text-matched against actions.json):")
    for a, count in analysis["action_dist_bcq_gqa"].items():
        print(f"  Action {a:2d}: {count}")
    print("  (Note: gqas samples may underreport — LLM-generated answers often paraphrase action descriptions.)")
    print()

    print("Action Distribution (DMCQ - text-matched):")
    for a, count in analysis["action_dist_dmcq"].items():
        print(f"  Action {a:2d}: {count}")
    non_sop_id = analysis.get("non_sop_action_id")
    non_sop_pct = analysis["dmcq_non_sop_pct"]
    if non_sop_pct > 0:
        print(f"  Action {non_sop_id} ('none') percentage: {non_sop_pct:.1f}%")
        if non_sop_pct > 50:
            print("  WARNING: DMCQ heavily imbalanced toward non-SOP action")
    print()

    print(f"MCQ multi-action coverage: {analysis['mcq_multi_action_count']}"
          f"/{analysis['mcq_total']} "
          f"({analysis['mcq_multi_action_pct']:.1f}%)")
    print(f"Max actions in single MCQ: {analysis['max_actions_in_mcq']}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze VLM fine-tuning training data distribution and "
                    "quality."
    )
    parser.add_argument(
        "augmented_data_root",
        help="Root directory containing augmented data subdirectories "
             "(e.g. <video_set>_augmented_0/{bcq,mcq,dmcq,...}/*.json)",
    )
    parser.add_argument(
        "--actions-json", default=None,
        help="Path to actions.json (required for DMCQ text-to-action mapping)",
    )
    parser.add_argument(
        "--non-sop-action", type=int, default=None,
        help="Action ID for the non-SOP catch-all action",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save training_data_analysis.json "
             "(defaults to the input data root)",
    )
    args = parser.parse_args()

    analysis = analyze_augmented_data(
        args.augmented_data_root, args.actions_json, args.non_sop_action
    )
    print_report(analysis)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(args.augmented_data_root)
    out_path = out_dir / "training_data_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to: {out_path}")


if __name__ == "__main__":
    main()
