#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Shared API client used by /sop-by-action-eval and /sop-e2e-inference.

Reads an inputs.yaml, POSTs the corresponding eval-ms endpoint, polls
status until terminal, resolves the host-side output directory, and
prints a structured JSON envelope to stdout so the calling skill (or
orchestrator) can consume it without parsing logs.

Usage:
    eval_api_client.py by-action <inputs.yaml> [--overrides <overrides.yaml>]
    eval_api_client.py e2e       <inputs.yaml> [--overrides <overrides.yaml>]

Stdout JSON envelope (single line, last line printed):
    {
        "mode": "by-action" | "e2e",
        "eval_job_id": str,
        "status": "completed" | "failed" | "cancelled",
        "host_output_dir": str | null,
        "container_output_dir": str,
        "artifacts": { ... mode-specific paths (None when missing) ... },
        "error": str | null
    }

Stderr carries progress logs (one line per poll tick + the request body
that was sent).

inputs.yaml schema (common):
    eval_host: localhost
    eval_port: 32090
    host_results_root: /abs/host/path/to/results   # maps to container /workspace/sop-eval-ms/assets/results
    training_job_id: <uuid>
    val_dataset_id: <uuid>
    backend: vllm                                  # transformers | vllm
    fps: 8
    temperature: 0.0
    top_p: 1.0
    checkpoint_step: null
    resolution_config: null                        # see ResolutionConfig in BP request_validation
    gpu_id: null
    poll_interval_sec: 20
    timeout_sec: 3600

e2e-only:
    ddm_training_job_id: <uuid>                    # required when chunking_algorithm=ddm
    ddm_checkpoint: null
    score_threshold: 0.5
    nms_sec: 0.0
    ddm_batch_size: 8
    frames_per_segment_hint: 256
    chunking_algorithm: ddm                        # ddm | uniform
    chunk_length_sec: null                         # required when chunking_algorithm=uniform
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _http_json(method: str, url: str, body: Optional[dict] = None, timeout: int = 30) -> tuple[int, Any]:
    """Minimal urllib JSON client — keeps the skill dependency-free."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode()
            return resp.status, (json.loads(payload) if payload else None)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"detail": body}
        return e.code, parsed


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        _eprint("ERROR: PyYAML is required. Install with: pip install pyyaml")
        sys.exit(2)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        _eprint(f"ERROR: {path} did not parse to a mapping")
        sys.exit(2)
    return data


def _merge(base: dict, override: dict) -> dict:
    """Shallow merge with override winning. Used to layer natural-language
    overrides onto inputs.yaml. None values in override are skipped so
    yaml `key:` (parsed as None) does not erase a non-None default."""
    out = dict(base)
    for k, v in override.items():
        if v is not None:
            out[k] = v
    return out


def _build_request(mode: str, cfg: dict) -> dict:
    """Build the eval-ms request body from the merged config.

    Strips keys that are None (Pydantic models with Optional fields treat
    `absent` and `None` equivalently, but the eval-ms ResolutionConfig
    uses extra='forbid' and would reject e.g. `resolution_config: {}`).
    """
    common_keys = (
        "training_job_id", "val_dataset_id", "fps", "temperature", "top_p",
        "backend", "checkpoint_step", "resolution_config", "gpu_id",
    )
    e2e_keys = (
        "ddm_training_job_id", "ddm_checkpoint", "score_threshold", "nms_sec",
        "ddm_batch_size", "frames_per_segment_hint", "chunking_algorithm",
        "chunk_length_sec",
    )

    body: dict = {}
    keys = common_keys + (e2e_keys if mode == "e2e" else ())
    for k in keys:
        if k in cfg and cfg[k] is not None:
            body[k] = cfg[k]

    # eval-ms requires training_job_id and val_dataset_id
    for required in ("training_job_id", "val_dataset_id"):
        if required not in body:
            _eprint(f"ERROR: '{required}' is required in inputs.yaml")
            sys.exit(2)

    if mode == "e2e":
        algo = body.get("chunking_algorithm", "ddm")
        if algo == "ddm" and "ddm_training_job_id" not in body:
            _eprint("ERROR: 'ddm_training_job_id' is required when chunking_algorithm='ddm'")
            sys.exit(2)
        if algo == "uniform" and "chunk_length_sec" not in body:
            _eprint("ERROR: 'chunk_length_sec' is required when chunking_algorithm='uniform'")
            sys.exit(2)

    return body


def _resolve_host_output_dir(cfg: dict, eval_job_id: str) -> Optional[str]:
    """The eval-ms always writes under <RESULTS_ROOT>/<eval_job_id>/ on
    the container; that path maps to <host_results_root>/<eval_job_id>/
    on the host via the docker-compose volume."""
    host_root = cfg.get("host_results_root")
    if not host_root:
        return None
    return str(Path(host_root).expanduser().resolve() / eval_job_id)


def _artifact_paths(mode: str, host_output_dir: Optional[str]) -> dict:
    if not host_output_dir:
        return {}
    root = Path(host_output_dir)
    if mode == "e2e":
        return {
            "e2e_results_json": str(root / "e2e_results.json"),
            "accuracy_json": str(root / "outputs_action_recognition" / "accuracy.json"),
            "video_name_to_output_text_json": str(
                root / "outputs_action_recognition" / "video_name_to_output_text.json"
            ),
            "action_recognition_log": str(
                root / "outputs_action_recognition" / "action_recognition_multi_gpu.log"
            ),
            "temporal_segmentation_dir": str(root / "outputs_temporal_segmentation"),
            "temporal_segmentation_log": str(
                root / "outputs_temporal_segmentation" / "temporal_segmentation.log"
            ),
            "sop_e2e_eval_log": str(root / "sop_e2e_eval_log.txt"),
            "log": str(root / "log.txt"),
        }
    return {
        "inference_results_json": str(root / "inference_results.json"),
        "log": str(root / "log.txt"),
    }


def _poll(base_url: str, mode: str, eval_job_id: str, interval: int, timeout: int) -> dict:
    """Returns the final status payload from eval-ms."""
    endpoint = "evaluation" if mode == "by-action" else "e2e-evaluation"
    deadline = time.time() + timeout
    terminal = {"completed", "failed", "cancelled"}
    last_status = None
    while time.time() < deadline:
        code, payload = _http_json("GET", f"{base_url}/api/v1/{endpoint}/status/{eval_job_id}")
        if code != 200 or not isinstance(payload, dict):
            _eprint(f"status check returned HTTP {code}: {payload}")
            time.sleep(interval)
            continue
        status = payload.get("status")
        if status != last_status:
            _eprint(f"[{eval_job_id}] status={status}")
            last_status = status
        if status in terminal:
            return payload
        time.sleep(interval)
    _eprint(f"ERROR: timed out after {timeout}s waiting for terminal status")
    return {"eval_job_id": eval_job_id, "status": "timeout"}


def run(mode: str, inputs_path: Path, overrides_path: Optional[Path]) -> int:
    cfg = _load_yaml(inputs_path)
    if overrides_path is not None:
        cfg = _merge(cfg, _load_yaml(overrides_path))

    host = cfg.get("eval_host", "localhost")
    port = cfg.get("eval_port", 32090)
    poll_interval = int(cfg.get("poll_interval_sec", 20))
    timeout = int(cfg.get("timeout_sec", 3600))
    base_url = f"http://{host}:{port}"
    endpoint = "evaluation" if mode == "by-action" else "e2e-evaluation"
    request_body = _build_request(mode, cfg)

    _eprint(f"POST {base_url}/api/v1/{endpoint}/start")
    _eprint(f"request body: {json.dumps(request_body, indent=2)}")

    code, payload = _http_json("POST", f"{base_url}/api/v1/{endpoint}/start", body=request_body, timeout=30)
    if code != 200 or not isinstance(payload, dict) or "eval_job_id" not in payload:
        envelope = {
            "mode": mode,
            "eval_job_id": None,
            "status": "failed",
            "host_output_dir": None,
            "container_output_dir": None,
            "artifacts": {},
            "error": f"start request failed (HTTP {code}): {payload}",
        }
        print(json.dumps(envelope))
        return 1

    eval_job_id = payload["eval_job_id"]
    _eprint(f"job started: {eval_job_id}")

    final = _poll(base_url, mode, eval_job_id, poll_interval, timeout)
    status = final.get("status", "unknown")

    container_root = "/workspace/sop-eval-ms/assets/results"
    container_output_dir = f"{container_root}/{eval_job_id}"
    host_output_dir = _resolve_host_output_dir(cfg, eval_job_id)
    artifacts = _artifact_paths(mode, host_output_dir)

    error = None
    if status != "completed":
        error = f"job ended in non-completed status: {status}"

    # Lift the terminal /status response's headline metrics into the envelope.
    headline: dict = {}
    for key in ("overall_accuracy", "avg_f1"):
        if key in final and final[key] is not None:
            headline[key] = final[key]

    envelope = {
        "mode": mode,
        "eval_job_id": eval_job_id,
        "status": status,
        "host_output_dir": host_output_dir,
        "container_output_dir": container_output_dir,
        "artifacts": artifacts,
        "headline_metrics": headline,
        "error": error,
    }
    print(json.dumps(envelope))
    return 0 if status == "completed" else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("by-action", "e2e"))
    p.add_argument("inputs_yaml", type=Path)
    p.add_argument("--overrides", type=Path, default=None,
                   help="Optional overrides yaml merged on top of inputs.yaml")
    args = p.parse_args()
    return run(args.mode, args.inputs_yaml, args.overrides)


if __name__ == "__main__":
    sys.exit(main())
