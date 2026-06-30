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

"""Analyze Cosmos Reason model fine-tuning training log.

Usage:
    python analyze_training_log.py <stderr_log_path>

Reports factual training metrics for the agent to interpret:
- Loss curve sampled at every 5% of training
- Learning rate schedule (initial, final, constant vs decay)
- Validation enabled or not
- Training completion status
- Errors and warnings
"""

import argparse
import json
import re
from pathlib import Path


def parse_training_log(log_path):
    with open(log_path) as f:
        content = f.read()

    lines = content.split("\n")

    # Extract loss values - support multiple log formats
    losses = []
    lr_values = []
    for line in lines:
        # Format 1: Step: N/M, Loss: X.XXX, ... Learning rate: X.XXXe-XX
        step_match = re.search(
            r"Step:\s*(\d+)/(\d+),\s*Loss:\s*([\d.e-]+).*?Learning rate:\s*([\d.e+-]+)",
            line,
        )
        if step_match:
            step = int(step_match.group(1))
            loss = float(step_match.group(3))
            lr = float(step_match.group(4))
            losses.append({"step": step, "loss": loss})
            lr_values.append({"step": step, "lr": lr})
            continue

        # Format 2: JSON-style: "step": N, ... "loss": X.XXX
        loss_match = re.search(r'"step":\s*(\d+).*?"loss":\s*([\d.e-]+)', line)
        if loss_match:
            step = int(loss_match.group(1))
            loss = float(loss_match.group(2))
            losses.append({"step": step, "loss": loss})

        lr_match = re.search(r'"lr":\s*\[([\d.e-]+)', line)
        if lr_match and losses:
            lr_values.append({"step": losses[-1]["step"], "lr": float(lr_match.group(1))})

    total_steps = losses[-1]["step"] if losses else 0

    # Build loss curve sampled at every 5% of training
    loss_curve = []
    if losses and total_steps > 0:
        for pct in range(0, 101, 5):
            target_step = int(total_steps * pct / 100)
            if target_step == 0:
                target_step = 1
            closest = min(losses, key=lambda l: abs(l["step"] - target_step))
            # Only include if reasonably close to the target
            if abs(closest["step"] - target_step) <= max(total_steps * 0.03, 5):
                loss_curve.append({
                    "pct": pct,
                    "step": closest["step"],
                    "loss": closest["loss"],
                })

    # Extract config dump values (done before LR schedule detection so
    # warmup/decay knobs are available there).
    epoch = None
    max_keep = None
    warmup_steps_config = None
    optm_decay_type = None
    optm_decay_ratio = None
    optm_min_lr_factor = None
    for line in lines:
        epoch_match = re.search(r"'epoch':\s*(\d+)", line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
        keep_match = re.search(r"'max_keep':\s*(\d+)", line)
        if keep_match:
            max_keep = int(keep_match.group(1))
        warmup_match = re.search(
            r"'optm_warmup_steps':\s*([\d.eE+-]+)", line
        )
        if warmup_match:
            warmup_steps_config = float(warmup_match.group(1))
        decay_type_match = re.search(
            r"'optm_decay_type':\s*('[^']*'|None|\w+)", line
        )
        if decay_type_match:
            val = decay_type_match.group(1)
            optm_decay_type = None if val == "None" else val.strip("'")
        decay_ratio_match = re.search(
            r"'optm_decay_ratio':\s*(None|[\d.eE+-]+)", line
        )
        if decay_ratio_match:
            val = decay_ratio_match.group(1)
            optm_decay_ratio = None if val == "None" else float(val)
        min_lr_match = re.search(
            r"'optm_min_lr_factor':\s*([\d.eE+-]+)", line
        )
        if min_lr_match:
            optm_min_lr_factor = float(min_lr_match.group(1))

    # Resolve warmup to a step count. Per Cosmos config:
    #   int → direct step count; float in [0.0, 1.0] → fraction of total_steps.
    if warmup_steps_config is not None:
        if warmup_steps_config < 1.0:
            warmup_steps = int(warmup_steps_config * total_steps)
        else:
            warmup_steps = int(warmup_steps_config)
    else:
        warmup_steps = None

    # Determine warmup boundary index used for LR schedule classification
    if warmup_steps and warmup_steps > 0:
        warmup_end = next(
            (i for i, lr in enumerate(lr_values)
             if lr["step"] >= warmup_steps),
            len(lr_values),
        )
        warmup_end = max(1, warmup_end)
    else:
        # Fallback when optm_warmup_steps is missing from the log
        warmup_end = max(1, len(lr_values) // 20)

    # LR schedule detection
    lr_schedule = "unknown"
    if lr_values:
        lr_post_warmup = lr_values[warmup_end:]
        if lr_post_warmup:
            lr_after_warmup = lr_post_warmup[0]["lr"]
            lr_last = lr_values[-1]["lr"]
            if abs(lr_after_warmup - lr_last) / max(lr_after_warmup, 1e-12) < 0.01:
                lr_schedule = "constant (no decay)"
            else:
                lr_schedule = "decaying"

    # Validation info
    has_validation = any(
        "validation" in line.lower() and "loss" in line.lower()
        for line in lines
    )

    # Check validation config
    val_enabled_in_config = False
    for line in lines:
        if "'enable': True" in line and "validation" in line:
            val_enabled_in_config = True
            break
        if "'enable': False" in line and "validation" in line:
            val_enabled_in_config = False
            break

    # Detect errors/warnings
    errors = []
    warnings = []
    for line in lines:
        if "error" in line.lower() and "traceback" not in line.lower():
            errors.append(line.strip()[:200])
        if "warning" in line.lower():
            warnings.append(line.strip()[:200])

    return {
        "total_steps": total_steps,
        "total_loss_entries": len(losses),
        "epoch": epoch,
        "max_keep": max_keep,
        "warmup_steps": warmup_steps,
        "warmup_ratio": (
            warmup_steps / total_steps
            if warmup_steps and total_steps > 0
            else None
        ),
        "initial_loss": losses[0]["loss"] if losses else None,
        "final_loss": losses[-1]["loss"] if losses else None,
        "loss_curve": loss_curve,
        "lr_initial": lr_values[0]["lr"] if lr_values else None,
        "lr_post_warmup": (
            lr_values[warmup_end]["lr"]
            if lr_values and warmup_end < len(lr_values)
            else None
        ),
        "lr_final": lr_values[-1]["lr"] if lr_values else None,
        "lr_schedule_observed": lr_schedule,
        "lr_schedule_configured": {
            "optm_decay_type": optm_decay_type,
            "optm_decay_ratio": optm_decay_ratio,
            "optm_min_lr_factor": optm_min_lr_factor,
        },
        "validation_enabled": val_enabled_in_config,
        "validation_loss_logged": has_validation,
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "errors_sample": errors[:5],
        "warnings_sample": warnings[:5],
    }


def print_report(analysis):
    print("=" * 70)
    print("TRAINING LOG ANALYSIS")
    print("=" * 70)
    print(f"Total steps: {analysis['total_steps']}")
    if analysis["epoch"]:
        print(f"Epochs: {analysis['epoch']}")
    if analysis.get("warmup_steps") is not None:
        ratio_str = (
            f"{analysis['warmup_ratio']:.1%}"
            if analysis.get("warmup_ratio") is not None
            else "n/a"
        )
        print(f"Warmup steps: {analysis['warmup_steps']} "
              f"({ratio_str} of total)")
    if analysis["max_keep"]:
        print(f"Checkpoint max_keep: {analysis['max_keep']}")
    print(f"Initial loss: {analysis['initial_loss']}")
    print(f"Final loss: {analysis['final_loss']}")
    print(f"Validation enabled: {analysis['validation_enabled']}")
    print(f"Validation loss logged: {analysis['validation_loss_logged']}")
    print()

    if analysis["loss_curve"]:
        print("Loss Curve (sampled every 5% of training):")
        for entry in analysis["loss_curve"]:
            print(f"  {entry['pct']:3d}% (step {entry['step']:5d}): "
                  f"loss = {entry['loss']:.6f}")
        print()

    if analysis["lr_initial"] is not None:
        print(f"LR: initial={analysis['lr_initial']}, "
              f"post_warmup={analysis['lr_post_warmup']}, "
              f"final={analysis['lr_final']}")
        cfg = analysis.get("lr_schedule_configured", {})
        print(f"LR schedule (configured): "
              f"decay_type={cfg.get('optm_decay_type')}, "
              f"decay_ratio={cfg.get('optm_decay_ratio')}, "
              f"min_lr_factor={cfg.get('optm_min_lr_factor')}")
        print(f"LR schedule (observed):   "
              f"{analysis['lr_schedule_observed']}")
        print()

    if analysis["errors_sample"]:
        print(f"Errors ({analysis['num_errors']} total, showing first 5):")
        for e in analysis["errors_sample"]:
            print(f"  {e[:100]}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Cosmos Reason model fine-tuning training log."
    )
    parser.add_argument(
        "log_path",
        help="Path to the VLM fine-tuning training log "
             "(e.g. vlm_train.log)",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save training_log_analysis.json "
             "(defaults to the input file's parent directory)",
    )
    args = parser.parse_args()

    analysis = parse_training_log(args.log_path)
    print_report(analysis)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(args.log_path).parent
    out_path = out_dir / "training_log_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to: {out_path}")


if __name__ == "__main__":
    main()
