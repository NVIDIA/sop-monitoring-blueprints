#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Harness-enforced bookkeeping gate for /sop-ft-orchestrate.

This is the *executable* form of SKILL.md Step 8a.0. The prose gate relies on the
orchestrating model remembering to run it every iteration; over a long run that
discipline drifts (skipped RCA, missing progress files, stale run_state.yaml).
This script moves the check into a process the Claude Code harness runs, so the
orchestrator physically cannot advance to the next phase — or finish the run —
while the previous iteration's artifacts are incomplete.

Wired via plugins/sop-ft-orchestrate-plugin/hooks/hooks.json:
  - PreToolUse (matcher "Skill"): blocks starting the NEXT phase delegation
    (augment / ddm / cr / by-action / e2e) until the last iteration is recorded.
  - Stop: blocks ending the run while the latest iteration is unrecorded or a
    failed eval lacks an RCA report.

Exit codes (Claude Code convention):
  0  -> allow (tool call proceeds / agent may stop)
  2  -> block; stderr is fed back to the model as the reason to self-correct

Design rule: NEVER brick a real run. Any validator error, missing dependency, or
absent run directory fails OPEN (exit 0). The cost is that a buggy validator lets
drift through; the benefit is a hook bug can't deadlock an expensive training run.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

# Skills that START new phase work. Invoking any of these means "advancing the
# pipeline", so the PREVIOUS iteration's bookkeeping must already be complete.
# /sop-rca is the corrective action and /sop-ft-orchestrate is the entry point /
# resume — both are deliberately ABSENT so the gate never blocks the very fix it
# is demanding, nor a fresh run, nor a resume.
ADVANCING_SKILLS = {
    "sop-data-augmentation",
    "sop-ddm-finetuning",
    "sop-cr-finetuning",
    "sop-by-action-eval",
    "sop-e2e-inference",
}


# Direct-driver phase-START signatures (used when the deployment has no eval-ms and
# the orchestrator advances phases via curl/bash instead of the Skill tool). NARROW on
# purpose — only the commands that START a new augment/train/eval phase, so ordinary
# bash (status polls, file ops, cleanup) is never gated. Fail-open everywhere else.
_ADVANCING_BASH = [
    re.compile(r"/api/v1/augment\b"),                       # data-gen augment start
    re.compile(r"/api/v1/fine-tuning/start\b"),             # DDM + VLM train start
    re.compile(r"sop-by-action-eval/scripts/run_eval\.sh\b"),   # direct by-action driver
    re.compile(r"sop-e2e-inference/scripts/run_e2e\.sh\b"),      # direct e2e driver
    re.compile(r"eval_api_client\.py\b"),                   # eval-ms client (if present)
]


def _is_advancing_bash(tool_name: str, command: str) -> bool:
    if tool_name != "Bash" or not command:
        return False
    # POST is what STARTS a job; a bare status GET to the same path must NOT gate.
    if "/api/v1/" in command and "fine-tuning/start" in command and "-X POST" not in command and "start?" not in command:
        return False
    return any(p.search(command) for p in _ADVANCING_BASH)


def allow(msg: str | None = None) -> "NoReturn":  # type: ignore[name-defined]
    if msg:
        print(f"[sop_iter_gate] {msg}", file=sys.stderr)
    sys.exit(0)


def block(violations: list[str]) -> "NoReturn":  # type: ignore[name-defined]
    print(
        "BOOKKEEPING GATE (Step 8a.0) — do this before proceeding:\n  - "
        + "\n  - ".join(violations),
        file=sys.stderr,
    )
    sys.exit(2)


def load_run_state(run_dir: str) -> dict:
    import yaml  # local import so a missing dep fails open, not at module load

    with open(os.path.join(run_dir, "run_state.yaml")) as f:
        return yaml.safe_load(f) or {}


def find_run_dir() -> str | None:
    """Locate the active run directory.

    Priority: explicit SOP_RUN_DIR, else the newest run_*/run_state.yaml under
    SOP_OUTPUT_ROOT / cwd / ./sop_fine_tune (the orchestrator's default output).
    """
    explicit = os.environ.get("SOP_RUN_DIR")
    if explicit and os.path.isdir(explicit):
        return explicit

    # Pointer file written by rs_update.py / gen_progress.py at every state update.
    # Lets the hook find a run_dir that lives OUTSIDE the harness cwd (the common case:
    # the user puts outputs in an arbitrary folder). Env var is unavailable to hooks,
    # so a file is the only reliable channel.
    ptr = os.path.join(os.path.expanduser("~"), ".cache", "sop-ft-orchestrate", "active_run")
    try:
        rd = open(ptr, encoding="utf-8").read().strip()
        if rd and os.path.isfile(os.path.join(rd, "run_state.yaml")):
            return rd
    except OSError:
        pass

    roots = [os.environ.get("SOP_OUTPUT_ROOT", ""), ".", "./sop_fine_tune"]
    candidates: list[str] = []
    for root in roots:
        if root:
            candidates += glob.glob(
                os.path.join(root, "**", "run_*", "run_state.yaml"), recursive=True
            )
    if not candidates:
        return None
    newest = max(candidates, key=os.path.getmtime)
    return os.path.dirname(newest)


def iter_dirs(run_dir: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for d in glob.glob(os.path.join(run_dir, "iter*")):
        m = re.search(r"iter(\d+)$", os.path.basename(d))
        if m and os.path.isdir(d):
            out[int(m.group(1))] = d
    return out


def snapshot_nonempty(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))


# A phase counts as EVALUATED only when its terminal RESULT file exists — not merely
# when the dir is non-empty. prepare_inputs.sh / a started-but-unfinished eval writes
# prompt/anno/log files into iter<N>/<phase>/ before any metrics exist; keying off
# "non-empty dir" wrongly flags an in-progress eval as done and blocks the next phase.
_PHASE_RESULT_FILES = {
    "by_action": ("results.json", "inference_results.json"),
    "e2e": ("outputs_action_recognition/accuracy.json", "e2e_results.json"),
}


def eval_completed(iter_dir: str, phase: str) -> bool:
    for rel in _PHASE_RESULT_FILES.get(phase, ()):
        if os.path.isfile(os.path.join(iter_dir, phase, rel)):
            return True
    return False


def _row_iter(row: dict):
    # run-state-schema.yaml uses `iteration`; older drafts used `iter`. Accept both.
    return row.get("iteration", row.get("iter"))


def _meets_criteria(row: dict, sc: dict) -> bool:
    """A row meets criteria when every ENABLED (non-null) success_criterion is satisfied
    by that row's metrics. Used only to exempt the final success iter from the RCA rule."""
    checks = [
        (sc.get("e2e_sequence_accuracy"), row.get("e2e_seq_acc")),
        (sc.get("e2e_action_accuracy"), row.get("e2e_action_acc")),
        (sc.get("by_action_accuracy"), row.get("by_action_acc")),
        (sc.get("ddm_f1"), row.get("ddm_f1")),
    ]
    enabled = [(t, v) for t, v in checks if t is not None]
    if not enabled:
        return False
    return all(v is not None and float(v) >= float(t) for t, v in enabled)


def check(run_dir: str) -> list[str]:
    """Return a list of violation strings. Empty list = the run is consistent."""
    violations: list[str] = []
    rs = load_run_state(run_dir)
    # Finished run -> nothing to enforce. The active_run pointer is never cleared, so
    # without this a completed run (esp. one that ended on a failed eval with no RCA)
    # would spuriously block the FIRST Stop of any later, unrelated session.
    status = str(rs.get("status") or "running").lower()
    if status not in ("running", "none", ""):
        return []
    run_rca = bool(rs.get("inputs", {}).get("run_rca", True)) and bool(rs.get("run_rca", True))
    sc = rs.get("success_criteria", {}) or {}
    eval_history = rs.get("eval_history", []) or []
    history_iters = {_row_iter(row) for row in eval_history}

    its = iter_dirs(run_dir)
    if not its:
        return violations  # nothing finished yet — never block the first launch

    # Which iterations have at least one non-empty eval snapshot on disk.
    evaluated: dict[int, list[str]] = {}
    for n, d in its.items():
        phases = [p for p in ("by_action", "e2e") if eval_completed(d, p)]
        if phases:
            evaluated[n] = phases

    # (1) Every snapshotted eval must have an eval_history row.
    for n in evaluated:
        if n not in history_iters:
            violations.append(
                f"iter{n}: eval snapshot exists on disk but there is no eval_history "
                f"row for it in run_state.yaml. Append the row now (Step 7c)."
            )

    # (2) run_state.iteration must not lag the iter dirs on disk.
    max_iter = max(its)
    if (rs.get("iteration") or 0) < max_iter:
        violations.append(
            f"run_state.yaml iteration={rs.get('iteration')} is behind iter{max_iter} "
            f"on disk. Update run_state.yaml to reflect iter{max_iter}."
        )

    # (3) Progress files: both exist; progress.md covers the latest iter.
    progress_md = os.path.join(run_dir, "progress.md")
    if not os.path.exists(progress_md):
        violations.append("progress.md is missing. Write it now (Step 2 / Step 7c).")
    else:
        try:
            text = open(progress_md, encoding="utf-8", errors="ignore").read()
        except OSError:
            text = ""
        # Anchor to the pipe-delimited table row gen_progress.py emits ("| N | ...").
        # A bare \b{max_iter}\b would false-pass on any metric/digit collision in the file.
        if not re.search(rf"^\|\s*{max_iter}\s*\|", text, re.MULTILINE):
            violations.append(
                f"progress.md has no row for iter{max_iter}. Regenerate it now "
                f"(python3 scripts/gen_progress.py <run_dir>)."
            )
    if not os.path.exists(os.path.join(run_dir, "progress.html")):
        violations.append("progress.html is missing. Regenerate it from the template (Step 7c).")

    # (4) Mandatory RCA: every evaluated iteration needs rca_report.md, unless that
    #     iteration met criteria (the final success iter) or RCA is disabled.
    if run_rca:
        success_iter = next(
            (_row_iter(row) for row in eval_history
             if row.get("criteria_met") or _meets_criteria(row, sc)),
            None,
        )
        for n in sorted(evaluated):
            if n == success_iter:
                continue
            if not os.path.exists(os.path.join(its[n], "rca_report.md")):
                violations.append(
                    f"iter{n}: a failed eval has no rca_report.md. Delegate to /sop-rca "
                    f"(Step 8b) — do NOT write your own diagnosis."
                )
    return violations


def main() -> None:
    event = "stop"
    if "--event" in sys.argv:
        try:
            event = sys.argv[sys.argv.index("--event") + 1]
        except IndexError:
            pass

    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}

    if event == "pretooluse":
        # Only gate the iteration-advancing phase delegations. Everything else —
        # the orchestrator itself, /sop-rca, Read/Edit/non-advancing Bash — passes through.
        tool_input = payload.get("tool_input") or {}
        tool_name = payload.get("tool_name") or ""
        skill = (tool_input.get("skill", "") or "").lstrip("/").split(":")[-1]
        cmd = tool_input.get("command", "") or ""
        # The orchestrator delegates phases via the Skill tool when sub-skills exist,
        # but in deployments WITHOUT eval-ms it advances phases via direct curl/bash
        # drivers. Gate BOTH paths, else the per-iteration check is silently bypassed.
        advancing = (skill in ADVANCING_SKILLS) or _is_advancing_bash(tool_name, cmd)
        if not advancing:
            allow()

    if event == "stop":
        # We've already forced one continuation; don't trap an autonomous run in a
        # loop if the model genuinely can't satisfy the gate. PreToolUse remains the
        # stronger, per-phase enforcement.
        if payload.get("stop_hook_active"):
            allow()

    run_dir = find_run_dir()
    if not run_dir:
        allow()  # no active run -> nothing to enforce

    try:
        violations = check(run_dir)
    except Exception as e:  # missing pyyaml, malformed run_state, etc. -> fail open
        allow(f"validator error, allowing (fix the gate, do not rely on it blocking): {e}")

    if violations:
        block(violations)
    allow()


if __name__ == "__main__":
    main()
