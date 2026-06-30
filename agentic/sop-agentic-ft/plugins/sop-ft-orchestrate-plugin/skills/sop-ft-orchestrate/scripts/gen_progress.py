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

"""Regenerate progress.md (and progress.html) from run_state.yaml.

WHY THIS EXISTS
---------------
progress.md used to be hand-appended row-by-row with the Edit tool. Over a long run
that discipline drifts and the file silently freezes several iterations behind the
real state (a documented bug). The fix: progress.md is no longer authored — it is a
PURE PROJECTION of run_state.yaml's eval_history. Regenerate it after every eval and
it can never disagree with the canonical state.

USAGE
  gen_progress.py <run_dir>                      # writes progress.md (+ progress.html if template found)
  gen_progress.py <run_dir> --template <tmpl>    # explicit progress-chart-template.html

Idempotent: same run_state -> same output. Run it in Step 7c instead of editing
progress.md by hand.
"""
from __future__ import annotations
import json, os, sys
try:
    import yaml
except ImportError:
    sys.exit("PyYAML required")

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TMPL = os.path.join(SKILL_DIR, "references", "progress-chart-template.html")


def pct(x):
    return None if x is None else round(float(x) * 100, 1)


# --- schema-tolerant accessors (run_state row conventions vary across writers) ---
def _errs(e: dict):
    """wrong/dup/miss — accept flat top-level keys OR a nested `errors` dict."""
    err = e.get("errors") or {}
    pick = lambda k: e.get(k, err.get(k, "-"))
    return pick("wrong"), pick("duplicate"), pick("missing")


def _note(e: dict) -> str:
    """Per-row note — accept `note` (singular) or `notes` (plural)."""
    return e.get("note") or e.get("notes") or ""


def _row_type(e: dict, rs: dict) -> str:
    """Row type — from the row if present, else cross-reference iteration_queue, else substantive."""
    if e.get("type"):
        return e["type"]
    for q in rs.get("iteration_queue", []) or []:
        if q.get("iter") == e.get("iteration"):
            return q.get("type", "substantive")
    return "substantive"


def _top_notes(rs: dict) -> list:
    """Top-level notes — accept a list OR a multiline string (YAML block scalar)."""
    n = rs.get("notes") or []
    if isinstance(n, str):
        return [ln for ln in n.splitlines() if ln.strip()]
    return list(n)


def gen_md(rs: dict) -> str:
    sc = rs.get("success_criteria", {}) or {}
    tgt = sc.get("e2e_sequence_accuracy")
    lines = [
        f"# SOP Fine-tuning Progress — {rs.get('run_id','?')}",
        "",
        f"**Dataset:** {rs.get('dataset_path','?')}",
        f"**Target:** E2E sequence_accuracy >= {tgt}",
        f"**Status:** {rs.get('status','?')}  |  **Current iteration:** {rs.get('iteration','?')}",
        "",
        "> GENERATED from run_state.yaml by gen_progress.py — do not hand-edit; re-run after each eval.",
        "",
        "## Iteration results (from eval_history)",
        "",
        "| Iter | Type | by-action | E2E action | **seq** | DDM F1 | thr | wrong/dup/miss | Change | Notes |",
        "|------|------|-----------|------------|---------|--------|-----|----------------|--------|-------|",
    ]
    def pctcell(x):
        return "-" if x is None else f"{pct(x)}%"

    for e in rs.get("eval_history", []) or []:
        w, d, m = _errs(e)
        wdm = f"{w}/{d}/{m}"
        ch = "; ".join(e.get("changes_applied", []) or [])
        lines.append(
            f"| {e.get('iteration','?')} | {_row_type(e, rs)} "
            f"| {pctcell(e.get('by_action_acc'))} "
            f"| {pctcell(e.get('e2e_action_acc'))} "
            f"| {pctcell(e.get('e2e_seq_acc'))} "
            f"| {e.get('ddm_f1','-')} | {e.get('ddm_threshold','-')} | {wdm} "
            f"| {ch} | {_note(e)} |"
        )
    b = rs.get("iteration_budget", {}) or {}
    lines += [
        "",
        f"**Budget:** substantive {b.get('iterations_substantive','?')}/{b.get('max_pipeline_iterations','?')}"
        f"  | eval-only {b.get('iterations_eval_only','?')}  | RCA runs {b.get('rca_runs_completed','?')}"
        f"  | remaining {b.get('iterations_remaining','?')}",
        "",
        "## Phase status (current iteration)",
        "",
    ]
    for k, v in (rs.get("phase_status", {}) or {}).items():
        lines.append(f"- {k}: {v}")
    notes = _top_notes(rs)
    if notes:
        lines += ["", "## Notes"] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def gen_html(rs: dict, tmpl_path: str) -> str | None:
    if not os.path.isfile(tmpl_path):
        return None
    tmpl = open(tmpl_path).read()
    sc = rs.get("success_criteria", {}) or {}
    tgt = sc.get("e2e_sequence_accuracy")
    iters = []
    for e in rs.get("eval_history", []) or []:
        w, d, m = _errs(e)
        to_int = lambda v: 0 if v in (None, "-") else v
        chart = e.get("chart", {}) or {}  # optional viz extras (ph,t,lr,qas,samp)
        row = {
            "n": str(e.get("iteration", "")),
            "ph": chart.get("ph", "III"),
            "t": chart.get("t", _row_type(e, rs)),
            "ba": pct(e.get("by_action_acc")),
            "ea": pct(e.get("e2e_action_acc")) or 0,
            "sq": pct(e.get("e2e_seq_acc")) or 0,
            "f1": e.get("ddm_f1") or 0,
            "th": e.get("ddm_threshold") or 0,
            "d": to_int(d), "w": to_int(w), "m": to_int(m),
            "lr": chart.get("lr", ""),
            "note": _note(e) or (e.get("changes_applied") or [""])[0],
        }
        if "samp" in chart: row["samp"] = chart["samp"]
        if "qas" in chart: row["qas"] = chart["qas"]
        iters.append(row)
    data = "const RUN=" + json.dumps({
        "id": rs.get("run_id", ""),
        "dataset": os.path.basename(rs.get("dataset_path", "") or ""),
        "target": pct(tgt) if tgt is not None else 90,
        "iters": iters,
    }) + ";"
    return tmpl.replace("/* ITER_DATA */", data)


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    run_dir = argv[1]
    tmpl = DEFAULT_TMPL
    if "--template" in argv:
        tmpl = argv[argv.index("--template") + 1]
    # keep the gate's active-run pointer current (harness hook can't read env vars)
    try:
        d = os.path.join(os.path.expanduser("~"), ".cache", "sop-ft-orchestrate")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "active_run"), "w").write(os.path.abspath(run_dir))
    except OSError:
        pass
    rs = yaml.safe_load(open(os.path.join(run_dir, "run_state.yaml"))) or {}
    md_path = os.path.join(run_dir, "progress.md")
    open(md_path, "w").write(gen_md(rs))
    print(f"=== PROGRESS (local file): {md_path} ===")
    html = gen_html(rs, tmpl)
    if html is not None:
        html_path = os.path.join(run_dir, "progress.html")
        open(html_path, "w").write(html)
        print(f"=== PROGRESS (local file): {html_path} ===")


if __name__ == "__main__":
    main(sys.argv)
