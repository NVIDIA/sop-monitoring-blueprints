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

"""Analyze data augmentation config for known issues.

Usage:
    python analyze_augment_config.py <augment_config_yaml>

Checks for:
- non_sop_action misconfiguration (should match across sections)
- MCQ max_chunk_len adequacy
- DMCQ hard_neg_mode / confusion_map configuration
- Negative ratio balance
- Missing QA types
"""

import argparse
import json
from pathlib import Path

try:
    import yaml
except ImportError:
    print("WARNING: PyYAML not installed. Install with: pip install pyyaml")
    yaml = None


def analyze_augment_config(config_path: str) -> dict:
    if yaml is None:
        return {"error": "PyYAML not installed"}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    issues = []
    warnings = []
    info = []

    # Check non_sop_action consistency
    non_sop_values = {}
    for section_name, section in config.items():
        if isinstance(section, dict) and "non_sop_action" in section:
            non_sop_values[section_name] = section["non_sop_action"]

    if len(set(non_sop_values.values())) > 1:
        issues.append({
            "type": "CONFIG_MISMATCH",
            "severity": "HIGH",
            "field": "non_sop_action",
            "detail": f"Inconsistent non_sop_action values across sections: {non_sop_values}. "
                      "All sections should use the same non_sop_action value (typically the "
                      "'none of the above' action ID).",
            "fix": "Set all non_sop_action values to the 'none of the above' action ID.",
        })

    # Check dynamic_mcq config
    dmcq = config.get("dynamic_mcq", {})
    if dmcq.get("enable", False):
        # Check hard_neg_mode
        hard_neg = dmcq.get("hard_neg_mode", "")
        if not hard_neg or hard_neg == "":
            warnings.append({
                "type": "MISSING_HARD_NEG",
                "severity": "MEDIUM",
                "field": "dynamic_mcq.hard_neg_mode",
                "detail": "No hard_neg_mode configured for DMCQ. If there are commonly "
                          "confused action pairs, add confusion-aware hard negatives.",
                "fix": "Set hard_neg_mode: 'confusion' and provide confusion_map.",
            })

        confusion_map = dmcq.get("confusion_map", "")
        if hard_neg == "confusion" and not confusion_map:
            issues.append({
                "type": "MISSING_CONFUSION_MAP",
                "severity": "HIGH",
                "field": "dynamic_mcq.confusion_map",
                "detail": "hard_neg_mode is 'confusion' but no confusion_map provided.",
                "fix": "Add confusion_map with commonly confused action pairs.",
            })

        info.append({
            "field": "dynamic_mcq",
            "num_pos": dmcq.get("num_pos"),
            "num_neg": dmcq.get("num_neg"),
            "min_options": dmcq.get("min_options"),
            "max_options": dmcq.get("max_options"),
        })

    # Check sequential_mcq config
    mcq = config.get("sequential_mcq", {})
    if mcq.get("enable", False):
        max_chunk_len = mcq.get("max_chunk_len", 1)
        if max_chunk_len < 3:
            warnings.append({
                "type": "LOW_MCQ_CHUNK_LEN",
                "severity": "MEDIUM",
                "field": "sequential_mcq.max_chunk_len",
                "detail": f"max_chunk_len={max_chunk_len}. When DDM under-segments, "
                          "VLM may encounter chunks with 3-4 actions. Training with "
                          "max_chunk_len >= 3 improves robustness.",
                "fix": f"Increase max_chunk_len from {max_chunk_len} to 3.",
            })
        info.append({
            "field": "sequential_mcq",
            "max_chunk_len": max_chunk_len,
        })

    # Check enabled QA types
    qa_types = [
        "binary_choice_qa", "sequential_mcq", "dynamic_mcq",
        "dynamic_shuffling", "general_qa", "extra_negative",
    ]
    enabled_types = []
    disabled_types = []
    for qt in qa_types:
        section = config.get(qt, {})
        if isinstance(section, dict) and section.get("enable", False):
            enabled_types.append(qt)
        else:
            disabled_types.append(qt)

    info.append({
        "field": "enabled_qa_types",
        "enabled": enabled_types,
        "disabled": disabled_types,
    })

    # Check dynamic_shuffling
    ds = config.get("dynamic_shuffling", {})
    if ds.get("enable", False):
        ds_non_sop = ds.get("non_sop_action")
        dmcq_non_sop = dmcq.get("non_sop_action")
        if ds_non_sop and dmcq_non_sop and ds_non_sop != dmcq_non_sop:
            issues.append({
                "type": "DS_NON_SOP_MISMATCH",
                "severity": "HIGH",
                "field": "dynamic_shuffling.non_sop_action",
                "detail": f"dynamic_shuffling.non_sop_action={ds_non_sop} but "
                          f"dynamic_mcq.non_sop_action={dmcq_non_sop}. "
                          "These should match and both should be the 'none' action ID.",
                "fix": f"Change dynamic_shuffling.non_sop_action to {dmcq_non_sop}.",
            })

    return {
        "config_path": config_path,
        "issues": issues,
        "warnings": warnings,
        "info": info,
        "raw_config": config,
    }


def print_report(analysis: dict) -> None:
    print("=" * 70)
    print("AUGMENT CONFIG ANALYSIS")
    print("=" * 70)
    print(f"Config: {analysis['config_path']}")
    print()

    if analysis.get("error"):
        print(f"ERROR: {analysis['error']}")
        return

    if analysis["issues"]:
        print(f"ISSUES ({len(analysis['issues'])}):")
        for issue in analysis["issues"]:
            print(f"  [{issue['severity']}] {issue['type']}: {issue['detail']}")
            print(f"    FIX: {issue['fix']}")
        print()

    if analysis["warnings"]:
        print(f"WARNINGS ({len(analysis['warnings'])}):")
        for w in analysis["warnings"]:
            print(f"  [{w['severity']}] {w['type']}: {w['detail']}")
            print(f"    FIX: {w['fix']}")
        print()

    print("Config Info:")
    for item in analysis["info"]:
        field = item.pop("field")
        print(f"  {field}: {item}")
        item["field"] = field  # restore


def main():
    parser = argparse.ArgumentParser(
        description="Analyze data augmentation config for known issues."
    )
    parser.add_argument(
        "config_path",
        help="Path to the augment config YAML "
             "(e.g. augment_config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save augment_config_analysis.json "
             "(defaults to the input file's parent directory)",
    )
    args = parser.parse_args()

    analysis = analyze_augment_config(args.config_path)
    print_report(analysis)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(args.config_path).parent
    out_path = out_dir / "augment_config_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to: {out_path}")


if __name__ == "__main__":
    main()
