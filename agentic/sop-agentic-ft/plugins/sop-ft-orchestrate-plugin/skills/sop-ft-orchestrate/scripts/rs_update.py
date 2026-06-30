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

"""Structured, safe mutation of run_state.yaml for /sop-ft-orchestrate.

WHY THIS EXISTS
---------------
run_state.yaml is the single source of truth for an orchestration run. Editing it
with ad-hoc `sed`/`str.replace()`/from-memory Edit blocks is the documented cause of
two real production bugs:
  * notes/field CROSS-CONTAMINATION — appending an eval_history/rca_reports entry by
    copy-pasting a previous entry as a template carries the old entry's `notes` over.
  * SILENT NO-OP — str.replace() returns the string unchanged when the anchor does not
    byte-match (whitespace drift), so the "update" is lost with no error.

This tool eliminates both: it loads YAML -> mutates the Python object -> dumps YAML.
No string matching, no templates. Every write is round-trip validated.

USAGE (always pass the run_dir; the tool finds run_state.yaml inside it)
  # set scalar(s) (dotted path; ints/floats/bools/null auto-typed)
  rs_update.py <run_dir> set iteration=6 ddm_threshold=0.44 status=running
  rs_update.py <run_dir> set phase_status.vlm_train=done phase_status.rca=in_progress

  # append a row to a list, value is JSON
  rs_update.py <run_dir> append eval_history '{"iteration":5,"e2e_seq_acc":0.5,...}'
  rs_update.py <run_dir> append rca_reports  '{"iteration":5,"report_path":"...","typed_actions":[...]}'

  # bump iteration_budget counters atomically
  rs_update.py <run_dir> budget --substantive +1 --rca +1 --eval-only +0

Exit 0 on success (prints the changed keys); non-zero on error. NEVER silently no-ops:
appending requires the target to be a list; set requires the parent path to exist.
"""
from __future__ import annotations
import json, sys, os
try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")


def _write_active_pointer(run_dir: str) -> None:
    """Record this run_dir so the harness gate (sop_iter_gate.py) can locate a run
    that lives outside the harness cwd. Best-effort; never fail the update over it."""
    try:
        d = os.path.join(os.path.expanduser("~"), ".cache", "sop-ft-orchestrate")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "active_run"), "w", encoding="utf-8") as f:
            f.write(os.path.abspath(run_dir))
    except OSError:
        pass


def _rs_path(run_dir: str) -> str:
    p = os.path.join(run_dir, "run_state.yaml")
    if not os.path.isfile(p):
        sys.exit(f"run_state.yaml not found in {run_dir}")
    _write_active_pointer(run_dir)
    return p


def _load(p):
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _dump(p, data):
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True, width=4096)
    # round-trip validate before replacing the real file
    with open(tmp) as f:
        yaml.safe_load(f)
    os.replace(tmp, p)


def _coerce(v: str):
    low = v.lower()
    if low in ("null", "none", "~"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _set_dotted(root: dict, dotted: str, value):
    keys = dotted.split(".")
    node = root
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            sys.exit(f"set: parent path '{'.'.join(keys[:keys.index(k)+1])}' missing/not a mapping")
        node = node[k]
    node[keys[-1]] = value


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    run_dir, op = argv[1], argv[2]
    p = _rs_path(run_dir)
    data = _load(p)
    changed = []

    if op == "set":
        for pair in argv[3:]:
            if "=" not in pair:
                sys.exit(f"set: expected key=value, got '{pair}'")
            k, v = pair.split("=", 1)
            _set_dotted(data, k, _coerce(v))
            changed.append(k)

    elif op == "append":
        if len(argv) < 5:
            sys.exit("append: needs <list_key> <json_value>")
        list_key, raw = argv[3], argv[4]
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as e:
            sys.exit(f"append: value is not valid JSON: {e}")
        lst = data.get(list_key)
        if lst is None:
            lst = data[list_key] = []
        if not isinstance(lst, list):
            sys.exit(f"append: '{list_key}' is not a list")
        lst.append(value)
        changed.append(f"{list_key}[+1] (now {len(lst)})")

    elif op == "budget":
        b = data.setdefault("iteration_budget", {})
        import argparse
        ap = argparse.ArgumentParser(prog="rs_update budget")
        ap.add_argument("--substantive", type=int, default=0)
        ap.add_argument("--eval-only", type=int, default=0)
        ap.add_argument("--rca", type=int, default=0)
        a = ap.parse_args(argv[3:])
        b["iterations_substantive"] = int(b.get("iterations_substantive", 0)) + a.substantive
        b["iterations_eval_only"] = int(b.get("iterations_eval_only", 0)) + getattr(a, "eval_only")
        b["rca_runs_completed"] = int(b.get("rca_runs_completed", 0)) + a.rca
        mx = int(b.get("max_pipeline_iterations", 8))
        b["iterations_remaining"] = mx - int(b["iterations_substantive"])
        changed.append(f"budget: substantive={b['iterations_substantive']} eval_only={b['iterations_eval_only']} rca={b['rca_runs_completed']} remaining={b['iterations_remaining']}")
    else:
        sys.exit(f"unknown op '{op}' (set|append|budget)")

    _dump(p, data)
    print("[rs_update] OK: " + "; ".join(changed))


if __name__ == "__main__":
    main(sys.argv)
