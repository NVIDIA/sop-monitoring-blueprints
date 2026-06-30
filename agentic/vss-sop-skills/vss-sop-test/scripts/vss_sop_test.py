#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
VSS SOP Post-Deployment Test Suite

Validates every layer of the VSS SOP stack after deployment:
  Phase 1 — Service health (Docker containers)
  Phase 2 — ELK data pipeline (Elasticsearch indices)
  Phase 3 — VIOS recording & livestream
  Phase 4 — VSS Agent (MCP, LLM, VLM, snapshot, video, report)

Exit codes:
  0  — all tests passed
  1  — one or more tests failed

Usage:
  python scripts/vss_sop_test.py [--bp-repo <path>] [--env-file <path>]

The script auto-detects HOST_IP/EXTERNAL_IP from the .env file when --env-file
is supplied, otherwise falls back to localhost.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

# Single source of truth for the SOP flat-JSON Kibana/ES field rules. It lives in the
# vss-sop-build skill (which this skill already depends on, e.g. verify_build.sh). Import
# it directly; fall back to an inline shim only if the build skill is not co-located.
_KIBANA_LIB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "vss-sop-build", "scripts", "lib",
)
sys.path.insert(0, _KIBANA_LIB)
try:
    import kibana_fields  # type: ignore
except Exception:  # pragma: no cover - standalone fallback
    class _KibanaFieldsShim:
        FLAT_FIELDS = {"response", "sensor_id", "cv_execute_time",
                       "vlm_execute_time", "chunk_idx", "frame_number", "@timestamp"}
        PROTOBUF_NESTED_ROOTS = {"llm", "sensor"}
        BAD_TIME_FIELD = "timestamp"
        TIME_FIELD = "@timestamp"

        def scan_mapping_fields(self, mapping_fields):
            return (self.FLAT_FIELDS & mapping_fields,
                    self.FLAT_FIELDS - mapping_fields,
                    self.PROTOBUF_NESTED_ROOTS & mapping_fields)

        def scan_runtime_field_map(self, runtime_map, time_field):
            errors = []
            for token in ("llm.queries.response", "sensor.id.keyword"):
                if token in runtime_map:
                    errors.append(f"runtime fields reference protobuf-style '{token}'")
            if time_field == self.BAD_TIME_FIELD:
                errors.append("timeFieldName is 'timestamp' instead of '@timestamp'")
            return errors

    kibana_fields = _KibanaFieldsShim()  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def log_pass(msg: str) -> None:
    print(f"{Colors.GREEN}[PASS]{Colors.RESET} {_ts()} {msg}", flush=True)


def log_fail(msg: str) -> None:
    print(f"{Colors.RED}[FAIL]{Colors.RESET} {_ts()} {msg}", flush=True)


def log_info(msg: str) -> None:
    print(f"{Colors.CYAN}[INFO]{Colors.RESET} {_ts()} {msg}", flush=True)


def log_warn(msg: str) -> None:
    print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {_ts()} {msg}", flush=True)


def log_phase(name: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}", flush=True)
    print(f"  {name}", flush=True)
    print(f"{'='*60}{Colors.RESET}\n", flush=True)


def load_env_file(path: str) -> Dict[str, str]:
    """Parse a docker-compose style .env file into a dict."""
    env = {}
    if not os.path.isfile(path):
        return env
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = val.strip().strip("'\"")
            env[key] = val
    return env


# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name: str, passed: bool, detail: str = "", auto_debug: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.auto_debug = auto_debug


results: List[TestResult] = []


def record(name: str, passed: bool, detail: str = "", auto_debug: str = "") -> bool:
    results.append(TestResult(name, passed, detail, auto_debug))
    if passed:
        log_pass(f"{name}: {detail}" if detail else name)
    else:
        log_fail(f"{name}: {detail}" if detail else name)
        if auto_debug:
            log_warn(f"  Auto-debug hint: {auto_debug}")
    return passed


# ---------------------------------------------------------------------------
# Phase 1 — Service health
# ---------------------------------------------------------------------------

EXPECTED_CONTAINERS = [
    "mdx-kafka",
    "mdx-redis",
    "mdx-elastic",
    "mdx-logstash",
    "mdx-kibana",
    "vss-agent",
    "vss-va-mcp",
    "mdx-ds-sop-1",
    "sensor-ms-sop",
    "recorder-ms-1-sop",
    "rtspserver-ms-1-sop",
    "storage-ms-sop",
    "sdr-http-recorder-sop",
    "sdr-http-rtspserver-sop",
]

OPTIONAL_CONTAINERS = [
    "mdx-prometheus",
    "mdx-grafana",
    "mdx-dcgm-exporter",
    "mdx-cadvisor",
    "mdx-node-exporter",
    "mdx-phoenix",
    "vss-ui",
]


def _detect_docker_prefix() -> List[str]:
    """Detect how to run docker commands based on permissions."""
    try:
        subprocess.run(["docker", "ps"], capture_output=True, check=True, timeout=5)
        return []
    except Exception:
        pass

    try:
        subprocess.run(["sg", "docker", "-c", "docker ps"], capture_output=True, check=True, timeout=5)
        return ["sg", "docker", "-c"]
    except Exception:
        pass

    try:
        subprocess.run(["sudo", "docker", "ps"], capture_output=True, check=True, timeout=5)
        return ["sudo"]
    except Exception:
        pass

    return []


DOCKER_PREFIX = _detect_docker_prefix()


def run_docker(args_list: List[str], **kwargs) -> str:
    """Execute a docker command with the appropriate prefix."""
    cmd = ["docker"] + args_list
    if DOCKER_PREFIX == ["sg", "docker", "-c"]:
        import shlex
        escaped_cmd = " ".join(shlex.quote(arg) for arg in cmd)
        full_cmd = ["sg", "docker", "-c", escaped_cmd]
    else:
        full_cmd = DOCKER_PREFIX + cmd
    
    if kwargs.get("text") is None and kwargs.get("universal_newlines") is None:
        kwargs["text"] = True
    return subprocess.check_output(full_cmd, **kwargs)


def _docker_ps() -> List[Dict[str, str]]:
    """Return a list of container dicts from docker ps."""
    try:
        out = run_docker(
            ["ps", "--format", "{{.Names}}\t{{.Status}}"],
            timeout=15,
        )
    except Exception as e:
        record("docker_ps", False, str(e), "Is Docker running?")
        return []
    containers = []
    for line in out.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            containers.append({"name": parts[0], "status": parts[1]})
    return containers


def phase1_service_health() -> None:
    log_phase("Phase 1 — Service Health Checks")
    containers = _docker_ps()
    if not containers:
        record("docker_ps", False, "No containers found", "Run: docker compose up -d")
        return

    running_names = {c["name"] for c in containers}

    for name in EXPECTED_CONTAINERS:
        found = name in running_names
        status = next((c["status"] for c in containers if c["name"] == name), "NOT FOUND")
        is_up = found and "Up" in status
        record(
            f"container_{name}",
            is_up,
            status if found else "container not running",
            f"docker logs {name} --tail 50" if not is_up else "",
        )

    for name in OPTIONAL_CONTAINERS:
        if name in running_names:
            status = next((c["status"] for c in containers if c["name"] == name), "")
            if "Up" in status:
                log_info(f"Optional container {name}: {status}")
            else:
                log_warn(f"Optional container {name} not healthy: {status}")


# ---------------------------------------------------------------------------
# Phase 2 — ELK data pipeline
# ---------------------------------------------------------------------------

def _es_health(es_url: str) -> Optional[Dict]:
    try:
        r = requests.get(f"{es_url}/_cluster/health", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _es_indices(es_url: str) -> List[Dict]:
    try:
        r = requests.get(f"{es_url}/_cat/indices?format=json", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def _es_count(es_url: str, index_pattern: str) -> int:
    try:
        r = requests.get(f"{es_url}/{index_pattern}/_count", timeout=10)
        if r.status_code == 200:
            return r.json().get("count", 0)
    except Exception:
        pass
    return 0


def _es_count_since(es_url: str, index_pattern: str, time_field: str = "@timestamp",
                    window: str = "now-1h") -> int:
    """Count docs whose time_field falls within [window, now] — i.e. what the
    time-filtered Kibana dashboard would actually display."""
    body = {"query": {"range": {time_field: {"gte": window}}}}
    try:
        r = requests.post(f"{es_url}/{index_pattern}/_count",
                          json=body, timeout=10)
        if r.status_code == 200:
            return r.json().get("count", 0)
    except Exception:
        pass
    return 0


def _es_count_field_positive(es_url: str, index_pattern: str, field: str) -> int:
    """Count docs whose numeric `field` exists and is > 0 — i.e. docs that
    actually carry real data for a dashboard metric (not just the field present
    in the mapping, and not a default 0)."""
    body = {"query": {"range": {field: {"gt": 0}}}}
    try:
        r = requests.post(f"{es_url}/{index_pattern}/_count",
                          json=body, timeout=10)
        if r.status_code == 200:
            return r.json().get("count", 0)
    except Exception:
        pass
    return 0


def _es_mapping_fields(es_url: str, index_pattern: str) -> set:
    """Return the set of top-level field names from the ES mapping."""
    try:
        r = requests.get(f"{es_url}/{index_pattern}/_mapping", timeout=10)
        if r.status_code == 200:
            data = r.json()
            fields = set()
            for idx_data in data.values():
                props = idx_data.get("mappings", {}).get("properties", {})
                fields.update(props.keys())
            return fields
    except Exception:
        pass
    return set()


def _es_sample_doc(es_url: str, index_pattern: str) -> Optional[Dict]:
    """Fetch one non-error document from the index."""
    try:
        r = requests.post(
            f"{es_url}/{index_pattern}/_search?size=1",
            json={"query": {"bool": {"must_not": [{"term": {"tags.keyword": "_jsonparsefailure"}}]}}},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if r.status_code == 200:
            hits = r.json().get("hits", {}).get("hits", [])
            if hits:
                return hits[0].get("_source", {})
    except Exception:
        pass
    return None


def _kibana_dashboard_field_check(es_url: str, kb_url: str) -> None:
    """Verify the Kibana dashboard's data view fields match the actual ES mapping.

    The SOP dashboard (sop-kibana-objects.ndjson) defines an index-pattern with
    runtime fields that reference specific ES field names. If the ndjson was built
    for the protobuf schema but DS-SOP sends flat JSON, the fields won't exist
    and every dashboard panel shows "No field found" errors.
    """
    mapping_fields = _es_mapping_fields(es_url, "mdx-vlm-captions-*")
    if not mapping_fields:
        record("kibana_dashboard_fields", False,
               "Cannot read mdx-vlm-captions-* mapping (index may not exist yet)",
               "Wait for DS-SOP to produce data, then re-run this test")
        return

    present, missing, bad_present = kibana_fields.scan_mapping_fields(mapping_fields)

    errors = []
    if missing:
        errors.append(f"missing flat fields: {sorted(missing)}")
    if bad_present:
        errors.append(
            f"found protobuf-style nested fields {sorted(bad_present)} — "
            "ndjson may reference wrong field paths"
        )

    sample = _es_sample_doc(es_url, "mdx-vlm-captions-*")
    if sample:
        if "response" not in sample and "llm" in sample:
            errors.append(
                "sample doc has 'llm' (protobuf) instead of 'response' (flat JSON) — "
                "check SOP_MESSAGING_SCHEMA=JSON in DS-SOP .env"
            )

    if errors:
        record("kibana_dashboard_fields", False,
               "; ".join(errors),
               "The sop-kibana-objects.ndjson data view must use flat JSON field names "
               "(response.keyword, sensor_id.keyword, @timestamp, cv_execute_time, etc.). "
               "See SKILL.md troubleshooting: 'Kibana dashboard No field found'")
    else:
        record("kibana_dashboard_fields", True,
               f"All {len(present)} expected flat fields present in mapping")

    # Also verify Kibana saved objects reference correct fields
    try:
        r = requests.get(
            f"{kb_url}/api/saved_objects/_find",
            params={"type": "index-pattern", "search": "mdx-vlm-captions*",
                    "search_fields": "title"},
            headers={"kbn-xsrf": "true"},
            timeout=10,
        )
        if r.status_code == 200:
            saved = r.json().get("saved_objects", [])
            if saved:
                attrs = saved[0].get("attributes", {})
                runtime_map = attrs.get("runtimeFieldMap", "")
                time_field = attrs.get("timeFieldName", "")
                ndjson_errors = kibana_fields.scan_runtime_field_map(runtime_map, time_field)
                if ndjson_errors:
                    record("kibana_ndjson_fields", False,
                           "; ".join(ndjson_errors),
                           "Re-import the corrected sop-kibana-objects.ndjson: "
                           "curl -X POST localhost:5601/api/saved_objects/_import?overwrite=true "
                           "-H 'kbn-xsrf: true' --form file=@sop-kibana-objects.ndjson")
                else:
                    record("kibana_ndjson_fields", True,
                           "Kibana data view uses correct flat JSON field references")
            else:
                log_warn("  No mdx-vlm-captions index-pattern found in Kibana saved objects")
    except Exception:
        log_warn("  Could not query Kibana saved objects API (Kibana may not be ready)")


def phase2_elk_data(es_url: str, kb_url: str = "http://localhost:5601") -> None:
    log_phase("Phase 2 — ELK Data Pipeline")

    health = _es_health(es_url)
    if not health:
        record("elasticsearch_reachable", False, f"Cannot reach {es_url}",
               "Check mdx-elastic container and port 9200")
        return
    record("elasticsearch_reachable", True, f"status={health.get('status')}")

    es_status = health.get("status", "")
    record(
        "elasticsearch_cluster_health",
        es_status in ("green", "yellow"),
        f"cluster status: {es_status}",
        "docker logs mdx-elastic --tail 50" if es_status == "red" else "",
    )

    indices = _es_indices(es_url)
    mdx_indices = [i for i in indices if i.get("index", "").startswith("mdx-")]
    record(
        "elk_indices_exist",
        len(mdx_indices) > 0,
        f"Found {len(mdx_indices)} mdx-vlm-captions indices",
        "Check Logstash config and Kafka topics. docker logs mdx-logstash --tail 50",
    )

    for idx_info in mdx_indices[:10]:
        idx_name = idx_info.get("index", "")
        doc_count = int(idx_info.get("docs.count") or 0)
        if doc_count > 0:
            log_info(f"  Index {idx_name}: {doc_count} docs")

    vlm_count = _es_count(es_url, "mdx-vlm-captions-*")
    record(
        "elk_vlm_messages",
        vlm_count > 0,
        f"{vlm_count} mdx-vlm-captions documents",
        "Ensure DS-SOP pipeline is running and producing VLM messages to Kafka",
    )

    # Documents can exist yet be invisible on the Kibana dashboard if their
    # @timestamp is wrong. The classic failure: Logstash derives @timestamp from
    # the RELATIVE stream offset (first_timestamp + start_time) instead of a
    # wall-clock field, so every doc lands on 1970-01-01 (index
    # mdx-vlm-captions-1970-01-01) and the dashboard's recent-time filter shows
    # NOTHING — even though elk_vlm_messages passes. This check asserts docs are
    # actually queryable in a recent window (what the dashboard displays).
    if vlm_count > 0:
        recent_count = _es_count_since(es_url, "mdx-vlm-captions-*",
                                       time_field="@timestamp", window="now-1h")
        stale_1970 = any(
            i.get("index", "").startswith("mdx-vlm-captions-1970")
            for i in mdx_indices
        )
        record(
            "elk_dashboard_recent_records",
            recent_count > 0 and not stale_1970,
            f"{recent_count} doc(s) with @timestamp within the last hour"
            + (" (FOUND stale mdx-vlm-captions-1970-* index!)" if stale_1970 else ""),
            "@timestamp is not a real wall-clock time (docs likely on 1970-01-01), so "
            "the Kibana dashboard shows no records. Fix the mdx-vlm-captions Logstash "
            "filter to set 'timestamp' from a wall-clock epoch field "
            "(pipeline_chunk_end_timestamp / pipeline_vlm_ready_timestamp), NOT from "
            "first_timestamp + start_time. See modify_foundational_for_sop.py "
            "(modify_kafka_logstash_conf) and deployments/foundational/elk/configs/"
            "mdx-kafka-logstash.conf, then restart mdx-logstash and delete the stale "
            "mdx-vlm-captions-1970-* index.",
        )

        # The dashboard's "CV/VLM execution time over chunk idx" panels plot
        # Average(cv_execute_time) / Average(vlm_execute_time). These fields can be
        # present in the index MAPPING (from the template) yet never populated by
        # the DS-SOP chunk messages — so kibana_dashboard_fields passes while the
        # panels stay empty. Assert the documents actually carry real (non-zero)
        # values for these metrics.
        for _field in ("cv_execute_time", "vlm_execute_time"):
            populated = _es_count_field_positive(es_url, "mdx-vlm-captions-*", _field)
            record(
                f"elk_metric_populated_{_field}",
                populated > 0,
                f"{populated} doc(s) with {_field} > 0",
                f"The '{_field}' field is in the ES mapping but no document populates "
                f"it, so the SOP dashboard panel that averages {_field} over chunk_idx "
                f"is empty. DS-SOP must emit {_field} on each chunk message: see "
                "ds_sop_process.py (submit_vllm_inference sets cv_execute_time; "
                "vlm_inference_response_process sets vlm_execute_time; _publish_message "
                "defaults both). Rebuild ds-sop and redeploy.",
            )

        _kibana_dashboard_field_check(es_url, kb_url)


# ---------------------------------------------------------------------------
# Phase 3 — VIOS Recording & Livestream
# ---------------------------------------------------------------------------

def _vst_sensor_list(vst_url: str) -> List[Dict]:
    try:
        r = requests.get(f"{vst_url}/api/v1/sensor/list", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def _vst_streams(vst_url: str) -> Any:
    try:
        r = requests.get(f"{vst_url}/api/v1/sensor/streams", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def phase3_vios(vst_url: str) -> None:
    log_phase("Phase 3 — VIOS Recording & Livestream")

    try:
        r = requests.get(f"{vst_url}/api/v1/sensor/list", timeout=10)
        record("vios_reachable", r.status_code == 200, f"HTTP {r.status_code}",
               "Check VST containers and port 30888")
    except Exception as e:
        record("vios_reachable", False, str(e), "docker logs sensor-ms-sop --tail 50")
        return

    # Check general Web UI proxy and root redirects
    base_vst_url = vst_url.rsplit("/vst", 1)[0]
    try:
        r_root = requests.get(base_vst_url, allow_redirects=False, timeout=5)
        is_redirect = r_root.status_code == 301 and r_root.headers.get("Location", "").endswith("/vst/")
        record("vst_ui_root_redirect", is_redirect, f"HTTP {r_root.status_code} -> Location: {r_root.headers.get('Location', '?')}",
               "nginx.conf needs: location = / { return 301 /vst/; }")
    except Exception as e:
        record("vst_ui_root_redirect", False, str(e), "Check nginx.conf root redirect rules")

    try:
        r_ui = requests.get(f"{vst_url}/", timeout=5)
        has_ui_content = r_ui.status_code == 200 and "<title>VST UI</title>" in r_ui.text
        record("vst_ui_index_accessible", has_ui_content, f"HTTP {r_ui.status_code} (has VST UI title: {has_ui_content})",
               "nginx.conf needs: location /vst/ { proxy_pass http://sensor-ms/; }")
    except Exception as e:
        record("vst_ui_index_accessible", False, str(e), "Check nginx.conf /vst/ proxy routing")

    sensors = _vst_sensor_list(vst_url)
    record(
        "vios_sensors_registered",
        len(sensors) > 0,
        f"{len(sensors)} sensor(s) registered",
        "Run run_rtsp_test.sh to register sensor_0 with VST",
    )

    for s in sensors:
        name = s.get("name", "?")
        state = s.get("state", "?")
        log_info(f"  Sensor '{name}': state={state}, ip={s.get('sensorIp', '?')}")

    streams = _vst_streams(vst_url)
    has_streams = bool(streams) and (isinstance(streams, list) and len(streams) > 0)
    record(
        "vios_streams_available",
        has_streams,
        f"{len(streams) if isinstance(streams, list) else 0} stream group(s)",
        "Ensure RTSP stream is active and sensor is registered",
    )

    # Recording check
    _ACTIVE_RECORD_STATES = {"on", "alwayson", "recording", "started", "start", "running", "active", "user"}
    recording_ok = False
    for s in sensors:
        sensor_id = s.get("sensorId", "")
        name = s.get("name", "?")
        try:
            r = requests.get(f"{vst_url}/api/v1/record/status",
                             params={"sensorId": sensor_id}, timeout=10)
            if r.status_code == 200:
                log_info(f"  Recording status for '{name}': {r.text[:200]}")
                payload = r.json() if r.text.strip() else {}
                rec_status = ""
                if isinstance(payload, dict):
                    for key, val in payload.items():
                        if isinstance(val, dict):
                            rec_status = val.get("recording_status", val.get("status", ""))
                            break
                        elif isinstance(val, str):
                            rec_status = val
                            break
                if rec_status.strip().lower() in _ACTIVE_RECORD_STATES:
                    recording_ok = True
        except Exception:
            pass

    if sensors:
        record(
            "vios_recording_active",
            recording_ok,
            "At least one sensor is recording" if recording_ok else "No sensor is recording",
            "VIOS recording failure has 3 common causes: "
            "(1) SDR recorder WDM_WL_ADD_URL must be /api/v1/record/stream/add (NOT /api/v1/proxy/stream/add); "
            "(2) SDR recorder WDM_WL_CHANGE_ID_ADD must be camera_streaming (NOT camera_proxy); "
            "(3) recorder-ms needs STORAGE_MODULE_ENDPOINT env var pointing to storage-ms. "
            "Check: docker logs sdr-http-recorder-sop --tail 50 ; "
            "docker logs recorder-ms-1-sop --tail 50"
            if not recording_ok else "",
        )

    # Livestream check
    for s in sensors:
        sensor_id = s.get("sensorId", "")
        name = s.get("name", "?")
        try:
            r = requests.get(f"{vst_url}/api/v1/live/streams",
                             params={"sensorId": sensor_id}, timeout=10)
            if r.status_code == 200:
                data = r.json() if r.text.strip() else {}
                if data:
                    log_info(f"  Livestream for '{name}': available")
                    record(f"vios_livestream_{name}", True, "livestream endpoint responding")
                else:
                    log_info(f"  Livestream for '{name}': no active streams")
        except Exception as e:
            log_warn(f"  Livestream check for '{name}': {e}")


# ---------------------------------------------------------------------------
# Phase 4 — VSS Agent
# ---------------------------------------------------------------------------

def _http_get_ok(url: str, label: str, timeout: int = 15, auto_debug: str = "") -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        return record(label, r.status_code == 200,
                      f"HTTP {r.status_code}", auto_debug)
    except Exception as e:
        return record(label, False, str(e), auto_debug)


def _check_openai_models(base_url: str, label: str) -> bool:
    """Hit /v1/models and verify at least one model is listed."""
    url = f"{base_url}/v1/models"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return record(label, False, f"HTTP {r.status_code} from {url}",
                          f"Check endpoint {base_url}")
        data = r.json()
        models = data.get("data", [])
        model_ids = [m.get("id", "?") for m in models[:5]]
        return record(label, len(models) > 0,
                      f"{len(models)} model(s): {model_ids}")
    except Exception as e:
        return record(label, False, str(e), f"Check endpoint {base_url}")


def _agent_chat(agent_url: str, query: str, timeout: int = 60) -> Optional[Dict]:
    """Send a query to the VSS agent chat endpoint."""
    payload = {
        "messages": [{"role": "user", "content": query}],
        "stream": False,
    }
    try:
        r = requests.post(f"{agent_url}/v1/chat/completions",
                          json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# Substrings that mean the agent surfaced an error or tool-call failure in
# what would otherwise look like a normal reply. The agent renders these
# inline (e.g. "Error generating incident report: ..."), so a plain keyword
# search on the reply would still match — we have to grep for them explicitly.
_AGENT_ERROR_PATTERNS = (
    "error generating",                         # template_report_gen surfaces this
    "validation error",                         # pydantic validation failure
    "1 validation error",
    "input should be a valid",                  # pydantic type mismatch
    "templatereportgeninput",
    "templatereportgen",
    "alert_to_timestamp",                       # mentioned only when validation failed on it
    "alert_from_timestamp",
    "traceback",
    "tool call failed",
    "an internal error occurred",
    "rate limit",
    "service unavailable",
)


def _agent_reply_text(response: dict | None) -> str:
    """Extract the assistant's textual reply from a /chat/completions response."""
    if not response:
        return ""
    choices = response.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "") or ""


def _find_agent_error(answer: str) -> str | None:
    """Return the first error sentinel found in the reply, or None.
    Matches are case-insensitive and against the full reply (errors can be
    rendered inline alongside otherwise-plausible content)."""
    if not answer:
        return None
    lower = answer.lower()
    for pat in _AGENT_ERROR_PATTERNS:
        if pat in lower:
            return pat
    return None


def phase4_vss_agent(
    agent_url: str,
    mcp_url: str,
    llm_base_url: str,
    vlm_base_url: str,
    vst_url: str,
    llm_mode: str,
    vlm_mode: str,
) -> None:
    log_phase("Phase 4 — VSS Agent End-to-End")

    # 4.1 MCP health
    _http_get_ok(f"{mcp_url}/health", "vss_agent_mcp_health",
                 auto_debug="docker logs vss-va-mcp --tail 50")

    # 4.2 LLM endpoint
    if llm_mode == "remote":
        log_info(f"LLM is remote ({llm_base_url}), checking /v1/models")
        _check_openai_models(llm_base_url, "vss_agent_llm_endpoint")
    else:
        _http_get_ok(f"{llm_base_url}/v1/models", "vss_agent_llm_endpoint",
                     auto_debug="Check NIM LLM container")

    # 4.3 VLM endpoint
    if vlm_mode == "remote":
        log_info(f"VLM is remote ({vlm_base_url}), checking /v1/models")
        _check_openai_models(vlm_base_url, "vss_agent_vlm_endpoint")
    elif vlm_mode == "local":
        _http_get_ok(f"{vlm_base_url}/v1/models", "vss_agent_vlm_endpoint",
                     auto_debug="Check DS-SOP or NIM VLM container. docker logs mdx-ds-sop-1 --tail 50")
    else:
        _check_openai_models(vlm_base_url, "vss_agent_vlm_endpoint")

    # 4.4 VSS Agent health
    _http_get_ok(f"{agent_url}/health", "vss_agent_health",
                 auto_debug="docker logs vss-agent --tail 50")

    # 4.5 Snapshot via VSS agent
    sensors = _vst_sensor_list(vst_url)
    sensor_id = sensors[0].get("name", "sensor_0") if sensors else "sensor_0"

    log_info(f"Testing agent snapshot for sensor '{sensor_id}'...")
    snap_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    snap_response = _agent_chat(
        agent_url,
        f"Take a snapshot of sensor {sensor_id} at {snap_now}",
        timeout=300,  # Remote LLM (Nemotron Super 49B) can be slow; 90s is insufficient
    )
    if snap_response:
        answer = _agent_reply_text(snap_response)
        err = _find_agent_error(answer)
        has_url = "http" in answer.lower() or "snapshot" in answer.lower()
        if err:
            record("vss_agent_snapshot", False,
                   f"Agent reply contains error marker '{err}' (response length: {len(answer)})",
                   f"Full reply: {answer[:300]}... ; docker logs vss-agent --tail 100")
        else:
            record("vss_agent_snapshot", has_url,
                   f"Response length: {len(answer)} chars",
                   "Agent may not have a sensor registered. Run run_rtsp_test.sh first.")
    else:
        record("vss_agent_snapshot", False, "No response from agent",
               "docker logs vss-agent --tail 100")

    # 4.6 Video from VIOS → VLM
    log_info(f"Testing agent video clip + VLM analysis for sensor '{sensor_id}'...")
    now = datetime.now(timezone.utc)
    # Use a 60-minute window to avoid transient "no clips found" issues during active/mid-segment recordings.
    start = (now - timedelta(minutes=60)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    video_response = _agent_chat(
        agent_url,
        f"What is the current SOP status for sensor {sensor_id}?",
        timeout=600,  # Remote LLM (Nemotron Super 49B) requires multiple calls; observed ~5m33s on busy API
    )
    if video_response:
        answer = _agent_reply_text(video_response)
        err = _find_agent_error(answer)
        has_sop = any(kw in answer.lower() for kw in ["sop", "action", "cycle", "sensor", "status"])
        if err:
            record("vss_agent_video_vlm", False,
                   f"Agent reply contains error marker '{err}' (response length: {len(answer)})",
                   f"Full reply: {answer[:300]}... ; docker logs vss-agent --tail 100")
        else:
            record("vss_agent_video_vlm", has_sop,
                   f"Response length: {len(answer)} chars",
                   "Check VLM endpoint, DS-SOP pipeline, and sensor registration")
    else:
        record("vss_agent_video_vlm", False, "No response from agent",
               "docker logs vss-agent --tail 100")

    # 4.7 Report generation
    # Use a 1-minute window ending now (now-1min to now). Wait at least 1 minute
    # after the RTSP stream registers before running this check.
    report_now = datetime.now(timezone.utc)
    report_start = (report_now - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    report_end = report_now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    log_info(f"Testing agent report generation for sensor '{sensor_id}'...")
    report_response = _agent_chat(
        agent_url,
        f"Generate an SOP compliance report for sensor {sensor_id} from {report_start} to {report_end}",
        timeout=600,  # Remote LLM (Nemotron Super 49B) report requires many calls; observed 8-12min on busy API
    )
    if report_response:
        answer = _agent_reply_text(report_response)
        err = _find_agent_error(answer)
        has_report = any(kw in answer.lower() for kw in ["report", "compliance", "download", ".md", ".pdf"])
        if err:
            # NOTE: the prior implementation only checked for the keyword "report",
            # but the agent's error string ("Error generating incident report: ...")
            # ALSO contains "report" — so a Pydantic validation failure on
            # alert_to_timestamp/alert_from_timestamp would silently PASS. The
            # error-pattern check now catches those.
            record("vss_agent_report", False,
                   f"Report generation FAILED: agent reply contains '{err}' (response length: {len(answer)})",
                   f"Full reply: {answer[:500]}... ; "
                   f"Common cause: SOP tools returning a float field where "
                   f"template_report_gen expects ISO string. "
                   f"docker logs vss-agent --tail 100 | grep -i 'validation\\|template_report_gen'")
        else:
            record("vss_agent_report", has_report,
                   f"Response length: {len(answer)} chars",
                   "Check LLM endpoint, VLM endpoint, and template configuration")
    else:
        record("vss_agent_report", False, "No response from agent",
               "docker logs vss-agent --tail 100")


# ---------------------------------------------------------------------------
# Auto-debug: collect logs for failures
# ---------------------------------------------------------------------------

def auto_debug_failures() -> None:
    failures = [r for r in results if not r.passed]
    if not failures:
        return

    log_phase("Auto-Debug — Collecting diagnostics for failures")

    container_names_to_check = set()
    for f in failures:
        name_lower = f.name.lower()
        if "kibana" in name_lower:
            container_names_to_check.add("mdx-kibana")
        if "elasticsearch" in name_lower or "elk" in name_lower:
            container_names_to_check.update(["mdx-elastic", "mdx-logstash"])
        elif "recording" in name_lower:
            container_names_to_check.update([
                "recorder-ms-1-sop", "sdr-http-recorder-sop",
                "sensor-ms-sop", "storage-ms-sop",
            ])
        elif "vios" in name_lower or "vst" in name_lower or "livestream" in name_lower:
            container_names_to_check.update([
                "sensor-ms-sop", "rtspserver-ms-1-sop",
                "recorder-ms-1-sop", "sdr-http-recorder-sop",
            ])
        elif "mcp" in name_lower:
            container_names_to_check.add("vss-va-mcp")
        elif "agent" in name_lower:
            container_names_to_check.add("vss-agent")
        elif "llm" in name_lower:
            container_names_to_check.add("vss-agent")
        elif "vlm" in name_lower:
            container_names_to_check.update(["mdx-ds-sop-1", "vss-agent"])
        elif "ds" in name_lower or "container_mdx-ds" in name_lower:
            container_names_to_check.add("mdx-ds-sop-1")

    for cname in sorted(container_names_to_check):
        print(f"\n{Colors.YELLOW}--- Last 30 lines of {cname} ---{Colors.RESET}", flush=True)
        try:
            out = run_docker(
                ["logs", cname, "--tail", "30"],
                stderr=subprocess.STDOUT, timeout=10,
            )
            print(out, flush=True)
        except subprocess.CalledProcessError as e:
            print(f"  (docker logs failed: {e.output[:200] if e.output else e})", flush=True)
        except Exception as e:
            print(f"  (could not collect logs: {e})", flush=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary() -> None:
    log_phase("Test Summary")
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)

    for r in results:
        icon = f"{Colors.GREEN}PASS{Colors.RESET}" if r.passed else f"{Colors.RED}FAIL{Colors.RESET}"
        detail = f" — {r.detail}" if r.detail else ""
        print(f"  [{icon}] {r.name}{detail}", flush=True)

    print(flush=True)
    color = Colors.GREEN if failed == 0 else Colors.RED
    print(f"{color}{Colors.BOLD}{passed}/{total} passed, {failed} failed{Colors.RESET}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="VSS SOP Post-Deployment Test Suite")
    parser.add_argument("--bp-repo", default=os.environ.get("BP_REPO", "."),
                        help="Path to vss-sop repo root")
    parser.add_argument("--env-file", default=None,
                        help="Path to deployments/sop/.env (auto-detected from --bp-repo if omitted)")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run only this phase (1-4). 0 = all.")
    args = parser.parse_args()

    bp_repo = os.path.abspath(args.bp_repo)
    env_file = args.env_file or os.path.join(bp_repo, "deployments", "sop", ".env")
    env = load_env_file(env_file)
    log_info(f"Loaded {len(env)} vars from {env_file}")

    host_ip = env.get("HOST_IP", "localhost").strip("'\"")
    external_ip = env.get("EXTERNAL_IP", host_ip).strip("'\"")
    if external_ip.startswith("${"):
        external_ip = host_ip

    es_port = env.get("ELASTIC_SEARCH_PORT", env.get("VSS_ES_PORT", "9200"))
    vst_port = env.get("VST_PORT", "30888")
    agent_port = env.get("VSS_AGENT_PORT", "8000")
    mcp_port = env.get("VSS_VA_MCP_PORT", "9901")
    llm_mode = env.get("LLM_MODE", "remote")
    vlm_mode = env.get("VLM_MODE", "local")
    llm_base_url = env.get("LLM_BASE_URL", f"http://localhost:30081").strip("'\"")
    vlm_base_url = env.get("VLM_BASE_URL", f"http://localhost:30082").strip("'\"")

    kb_port = env.get("KIBANA_PORT", "5601")

    es_url = f"http://localhost:{es_port}"
    kb_url = f"http://localhost:{kb_port}"
    vst_url = f"http://localhost:{vst_port}/vst"
    agent_url = f"http://localhost:{agent_port}"
    mcp_url = f"http://localhost:{mcp_port}"

    log_info(f"HOST_IP={host_ip}  EXTERNAL_IP={external_ip}")
    log_info(f"ES={es_url}  KB={kb_url}  VST={vst_url}  Agent={agent_url}  MCP={mcp_url}")
    log_info(f"LLM={llm_base_url} ({llm_mode})  VLM={vlm_base_url} ({vlm_mode})")

    run_all = args.phase == 0

    if run_all or args.phase == 1:
        phase1_service_health()
    if run_all or args.phase == 2:
        phase2_elk_data(es_url, kb_url)
    if run_all or args.phase == 3:
        phase3_vios(vst_url)
    if run_all or args.phase == 4:
        phase4_vss_agent(agent_url, mcp_url, llm_base_url, vlm_base_url,
                         vst_url, llm_mode, vlm_mode)

    auto_debug_failures()
    print_summary()

    failed = sum(1 for r in results if not r.passed)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

