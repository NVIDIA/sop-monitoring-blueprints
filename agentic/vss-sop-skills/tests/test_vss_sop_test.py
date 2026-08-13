######################################################################################################
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
######################################################################################################

"""Unit tests for vss_sop_test.py (vss-sop-test skill post-deploy harness)."""

import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "vss-sop-test" / "scripts" / "vss_sop_test.py"
_KIBANA_LIB = _REPO_ROOT / "vss-sop-build" / "scripts" / "lib"


def load_vss_sop_test():
    """Load vss_sop_test without triggering real docker at import."""
    lib = str(_KIBANA_LIB)
    if lib not in sys.path:
        sys.path.insert(0, lib)
    with mock.patch("subprocess.run", side_effect=Exception("no docker")):
        spec = importlib.util.spec_from_file_location("vss_sop_test", _SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["vss_sop_test"] = mod
        spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return load_vss_sop_test()


@pytest.fixture(autouse=True)
def clear_results(mod):
    mod.results.clear()
    yield
    mod.results.clear()


def _mock_response(status_code=200, json_data=None, text="", headers=None):
    r = mock.Mock()
    r.status_code = status_code
    if json_data is not None and not text:
        text = json.dumps(json_data)
    r.text = text
    r.headers = headers or {}
    if json_data is not None:
        r.json.return_value = json_data
    return r


# ---------------------------------------------------------------------------
# Helpers: Colors, logging, load_env_file
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHelpers:
    def test_colors_constants(self, mod):
        assert mod.Colors.GREEN
        assert mod.Colors.RESET

    def test_logging_helpers(self, mod, capsys):
        mod.log_pass("ok")
        mod.log_fail("bad")
        mod.log_info("info")
        mod.log_warn("warn")
        mod.log_phase("Phase X")
        out = capsys.readouterr().out
        assert "[PASS]" in out
        assert "[FAIL]" in out
        assert "[INFO]" in out
        assert "[WARN]" in out
        assert "Phase X" in out

    def test_load_env_file_missing(self, mod, tmp_path):
        assert mod.load_env_file(str(tmp_path / "missing.env")) == {}

    def test_load_env_file_parsing(self, mod, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# comment\n"
            "HOST_IP=127.0.0.1\n"
            'QUOTED="hello"\'\n'
            "EMPTY=\n"
            "noequals\n"
        )
        env = mod.load_env_file(str(env_file))
        assert env["HOST_IP"] == "127.0.0.1"
        assert env["QUOTED"] == "hello"
        assert env["EMPTY"] == ""


# ---------------------------------------------------------------------------
# TestResult / record
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecord:
    def test_record_pass(self, mod):
        assert mod.record("t1", True, "detail") is True
        assert len(mod.results) == 1
        assert mod.results[0].passed is True

    def test_record_fail_with_auto_debug(self, mod, capsys):
        assert mod.record("t2", False, "oops", "hint") is False
        assert mod.results[0].auto_debug == "hint"
        assert "Auto-debug hint" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Docker prefix / run_docker / _docker_ps
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDockerHelpers:
    def test_detect_docker_prefix_direct(self, mod):
        with mock.patch.object(mod.subprocess, "run", return_value=mock.Mock(returncode=0)):
            assert mod._detect_docker_prefix() == []

    def test_detect_docker_prefix_sg(self, mod):
        def fake_run(cmd, **kwargs):
            if cmd == ["docker", "ps"]:
                raise OSError("denied")
            if cmd[:3] == ["sg", "docker", "-c"]:
                return mock.Mock(returncode=0)
            raise OSError("fail")

        with mock.patch.object(mod.subprocess, "run", side_effect=fake_run):
            assert mod._detect_docker_prefix() == ["sg", "docker", "-c"]

    def test_detect_docker_prefix_sudo(self, mod):
        def fake_run(cmd, **kwargs):
            if cmd in (["docker", "ps"], ["sg", "docker", "-c", "docker ps"]):
                raise OSError("denied")
            if cmd == ["sudo", "docker", "ps"]:
                return mock.Mock(returncode=0)
            raise OSError("fail")

        with mock.patch.object(mod.subprocess, "run", side_effect=fake_run):
            assert mod._detect_docker_prefix() == ["sudo"]

    def test_detect_docker_prefix_total_failure(self, mod):
        with mock.patch.object(mod.subprocess, "run", side_effect=Exception("no docker")):
            assert mod._detect_docker_prefix() == []

    def test_run_docker_empty_prefix(self, mod):
        mod.DOCKER_PREFIX = []
        with mock.patch.object(mod.subprocess, "check_output", return_value="out\n") as co:
            assert mod.run_docker(["ps"]) == "out\n"
            co.assert_called_once()
            assert co.call_args[0][0][0] == "docker"

    def test_run_docker_sg_prefix(self, mod):
        mod.DOCKER_PREFIX = ["sg", "docker", "-c"]
        with mock.patch.object(mod.subprocess, "check_output", return_value="sg-out") as co:
            assert mod.run_docker(["ps"]) == "sg-out"
            assert co.call_args[0][0][:3] == ["sg", "docker", "-c"]

    def test_docker_ps_success(self, mod):
        lines = "mdx-kafka\tUp 1 hour\nvss-agent\tUp 2 hours"
        with mock.patch.object(mod, "run_docker", return_value=lines):
            containers = mod._docker_ps()
        assert len(containers) == 2
        assert containers[0]["name"] == "mdx-kafka"

    def test_docker_ps_failure(self, mod):
        with mock.patch.object(mod, "run_docker", side_effect=RuntimeError("docker down")):
            assert mod._docker_ps() == []
        assert any(r.name == "docker_ps" and not r.passed for r in mod.results)


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhase1:
    def _containers(self, names_status):
        return [{"name": n, "status": s} for n, s in names_status]

    def test_no_containers(self, mod):
        with mock.patch.object(mod, "_docker_ps", return_value=[]):
            mod.phase1_service_health()
        assert any(r.name == "docker_ps" and not r.passed for r in mod.results)

    def test_expected_up_and_missing(self, mod):
        running = self._containers([
            ("mdx-kafka", "Up 1 hour"),
            ("mdx-redis", "Exited"),
        ])
        with mock.patch.object(mod, "_docker_ps", return_value=running):
            mod.phase1_service_health()
        passed = {r.name: r.passed for r in mod.results}
        assert passed["container_mdx-kafka"] is True
        assert passed["container_mdx-redis"] is False
        assert passed["container_vss-agent"] is False

    def test_optional_containers(self, mod, capsys):
        running = self._containers([
            ("mdx-kafka", "Up"),
            ("mdx-redis", "Up"),
            ("mdx-elastic", "Up"),
            ("mdx-logstash", "Up"),
            ("mdx-kibana", "Up"),
            ("vss-agent", "Up"),
            ("vss-va-mcp", "Up"),
            ("mdx-ds-sop-1", "Up"),
            ("sensor-ms-sop", "Up"),
            ("recorder-ms-1-sop", "Up"),
            ("rtspserver-ms-1-sop", "Up"),
            ("storage-ms-sop", "Up"),
            ("sdr-http-recorder-sop", "Up"),
            ("sdr-http-rtspserver-sop", "Up"),
            ("mdx-prometheus", "Up"),
            ("vss-ui", "Down"),
        ])
        with mock.patch.object(mod, "_docker_ps", return_value=running):
            mod.phase1_service_health()
        out = capsys.readouterr().out
        assert "Optional container mdx-prometheus" in out
        assert "vss-ui not healthy" in out


# ---------------------------------------------------------------------------
# Phase 2 — ES helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEsHelpers:
    def test_es_health_success(self, mod):
        with mock.patch.object(mod.requests, "get", return_value=_mock_response(200, {"status": "green"})):
            assert mod._es_health("https://es:9200")["status"] == "green"

    def test_es_health_failure(self, mod):
        with mock.patch.object(mod.requests, "get", side_effect=ConnectionError("down")):
            assert mod._es_health("https://es:9200") is None

    def test_es_indices(self, mod):
        with mock.patch.object(mod.requests, "get", return_value=_mock_response(200, [{"index": "mdx-a"}])):
            assert len(mod._es_indices("https://es:9200")) == 1

    def test_es_count(self, mod):
        with mock.patch.object(mod.requests, "get", return_value=_mock_response(200, {"count": 42})):
            assert mod._es_count("https://es:9200", "mdx-*") == 42

    def test_es_count_since(self, mod):
        with mock.patch.object(mod.requests, "post", return_value=_mock_response(200, {"count": 5})):
            assert mod._es_count_since("https://es:9200", "mdx-*") == 5

    def test_es_count_field_positive(self, mod):
        with mock.patch.object(mod.requests, "post", return_value=_mock_response(200, {"count": 3})):
            assert mod._es_count_field_positive("https://es:9200", "mdx-*", "cv_execute_time") == 3

    def test_es_mapping_fields(self, mod):
        data = {"idx": {"mappings": {"properties": {"response": {}, "sensor_id": {}}}}}
        with mock.patch.object(mod.requests, "get", return_value=_mock_response(200, data)):
            fields = mod._es_mapping_fields("https://es:9200", "mdx-*")
        assert "response" in fields

    def test_es_sample_doc(self, mod):
        hits = {"hits": {"hits": [{"_source": {"response": "hi"}}]}}
        with mock.patch.object(mod.requests, "post", return_value=_mock_response(200, hits)):
            assert mod._es_sample_doc("https://es:9200", "mdx-*")["response"] == "hi"

    def test_es_helpers_non_200_and_exceptions(self, mod):
        with mock.patch.object(mod.requests, "get", return_value=_mock_response(500)):
            assert mod._es_indices("https://es:9200") == []
            assert mod._es_count("https://es:9200", "mdx-*") == 0
            assert mod._es_mapping_fields("https://es:9200", "mdx-*") == set()
        with mock.patch.object(mod.requests, "get", side_effect=ConnectionError("x")):
            assert mod._es_indices("https://es:9200") == []
            assert mod._es_count("https://es:9200", "mdx-*") == 0
            assert mod._es_mapping_fields("https://es:9200", "mdx-*") == set()
        with mock.patch.object(mod.requests, "post", side_effect=ConnectionError("x")):
            assert mod._es_count_since("https://es:9200", "mdx-*") == 0
            assert mod._es_count_field_positive("https://es:9200", "mdx-*", "f") == 0
            assert mod._es_sample_doc("https://es:9200", "mdx-*") is None


# ---------------------------------------------------------------------------
# Kibana dashboard field check
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKibanaDashboardFieldCheck:
    def test_no_mapping(self, mod):
        with mock.patch.object(mod, "_es_mapping_fields", return_value=set()):
            mod._kibana_dashboard_field_check("https://es:9200", "https://kb:5601")
        assert any(r.name == "kibana_dashboard_fields" and not r.passed for r in mod.results)

    def test_missing_flat_fields(self, mod):
        with mock.patch.object(mod, "_es_mapping_fields", return_value={"response"}), \
             mock.patch.object(mod, "_es_sample_doc", return_value=None), \
             mock.patch.object(mod.requests, "get", side_effect=ConnectionError("kb down")):
            mod._kibana_dashboard_field_check("https://es:9200", "https://kb:5601")
        assert any(r.name == "kibana_dashboard_fields" and not r.passed for r in mod.results)

    def test_protobuf_sample_doc(self, mod):
        flat = mod.kibana_fields.FLAT_FIELDS
        with mock.patch.object(mod, "_es_mapping_fields", return_value=flat), \
             mock.patch.object(mod, "_es_sample_doc", return_value={"llm": {}}), \
             mock.patch.object(mod.requests, "get", side_effect=ConnectionError("kb down")):
            mod._kibana_dashboard_field_check("https://es:9200", "https://kb:5601")
        assert any(r.name == "kibana_dashboard_fields" and not r.passed for r in mod.results)

    def test_success_with_kibana_saved_objects(self, mod):
        flat = mod.kibana_fields.FLAT_FIELDS
        saved = {"saved_objects": [{"attributes": {
            "runtimeFieldMap": "response.keyword",
            "timeFieldName": "@timestamp",
        }}]}
        with mock.patch.object(mod, "_es_mapping_fields", return_value=flat), \
             mock.patch.object(mod, "_es_sample_doc", return_value={"response": "ok"}), \
             mock.patch.object(mod.requests, "get", return_value=_mock_response(200, saved)):
            mod._kibana_dashboard_field_check("https://es:9200", "https://kb:5601")
        assert any(r.name == "kibana_dashboard_fields" and r.passed for r in mod.results)
        assert any(r.name == "kibana_ndjson_fields" and r.passed for r in mod.results)

    def test_ndjson_runtime_errors(self, mod):
        flat = mod.kibana_fields.FLAT_FIELDS
        saved = {"saved_objects": [{"attributes": {
            "runtimeFieldMap": "llm.queries.response",
            "timeFieldName": "timestamp",
        }}]}
        with mock.patch.object(mod, "_es_mapping_fields", return_value=flat), \
             mock.patch.object(mod, "_es_sample_doc", return_value={"response": "ok"}), \
             mock.patch.object(mod.requests, "get", return_value=_mock_response(200, saved)):
            mod._kibana_dashboard_field_check("https://es:9200", "https://kb:5601")
        assert any(r.name == "kibana_ndjson_fields" and not r.passed for r in mod.results)

    def test_no_saved_objects_warns(self, mod, capsys):
        flat = mod.kibana_fields.FLAT_FIELDS
        with mock.patch.object(mod, "_es_mapping_fields", return_value=flat), \
             mock.patch.object(mod, "_es_sample_doc", return_value={"response": "ok"}), \
             mock.patch.object(mod.requests, "get", return_value=_mock_response(200, {"saved_objects": []})):
            mod._kibana_dashboard_field_check("https://es:9200", "https://kb:5601")
        assert "No mdx-vlm-captions index-pattern" in capsys.readouterr().out

    def test_protobuf_nested_mapping_fields(self, mod):
        flat = mod.kibana_fields.FLAT_FIELDS
        bad = flat | {"llm"}
        with mock.patch.object(mod, "_es_mapping_fields", return_value=bad), \
             mock.patch.object(mod, "_es_sample_doc", return_value={"response": "ok"}), \
             mock.patch.object(mod.requests, "get", side_effect=ConnectionError("kb down")):
            mod._kibana_dashboard_field_check("https://es:9200", "https://kb:5601")
        assert any(r.name == "kibana_dashboard_fields" and not r.passed for r in mod.results)

    def test_kibana_api_exception_warns(self, mod, capsys):
        flat = mod.kibana_fields.FLAT_FIELDS
        with mock.patch.object(mod, "_es_mapping_fields", return_value=flat), \
             mock.patch.object(mod, "_es_sample_doc", return_value={"response": "ok"}), \
             mock.patch.object(mod.requests, "get", side_effect=ConnectionError("kb down")):
            mod._kibana_dashboard_field_check("https://es:9200", "https://kb:5601")
        assert "Could not query Kibana saved objects API" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# phase2_elk_data
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhase2ElkData:
    def test_unreachable_es(self, mod):
        with mock.patch.object(mod, "_es_health", return_value=None):
            mod.phase2_elk_data("https://es:9200")
        assert any(r.name == "elasticsearch_reachable" and not r.passed for r in mod.results)

    def test_healthy_with_docs_and_metrics(self, mod):
        indices = [
            {"index": "mdx-vlm-captions-2026", "docs.count": "10"},
        ]
        with mock.patch.object(mod, "_es_health", return_value={"status": "green"}), \
             mock.patch.object(mod, "_es_indices", return_value=indices), \
             mock.patch.object(mod, "_es_count", return_value=100), \
             mock.patch.object(mod, "_es_count_since", return_value=50), \
             mock.patch.object(mod, "_es_count_field_positive", return_value=10), \
             mock.patch.object(mod, "_kibana_dashboard_field_check"):
            mod.phase2_elk_data("https://es:9200", "https://kb:5601")
        names = {r.name: r.passed for r in mod.results}
        assert names["elasticsearch_reachable"] is True
        assert names["elk_vlm_messages"] is True
        assert names["elk_dashboard_recent_records"] is True

    def test_stale_1970_index(self, mod):
        indices = [{"index": "mdx-vlm-captions-1970-01-01", "docs.count": "5"}]
        with mock.patch.object(mod, "_es_health", return_value={"status": "yellow"}), \
             mock.patch.object(mod, "_es_indices", return_value=indices), \
             mock.patch.object(mod, "_es_count", return_value=5), \
             mock.patch.object(mod, "_es_count_since", return_value=0), \
             mock.patch.object(mod, "_es_count_field_positive", return_value=0), \
             mock.patch.object(mod, "_kibana_dashboard_field_check"):
            mod.phase2_elk_data("https://es:9200")
        assert any(r.name == "elk_dashboard_recent_records" and not r.passed for r in mod.results)

    def test_red_cluster_and_no_indices(self, mod):
        with mock.patch.object(mod, "_es_health", return_value={"status": "red"}), \
             mock.patch.object(mod, "_es_indices", return_value=[]), \
             mock.patch.object(mod, "_es_count", return_value=0):
            mod.phase2_elk_data("https://es:9200")
        names = {r.name: r.passed for r in mod.results}
        assert names["elasticsearch_cluster_health"] is False
        assert names["elk_indices_exist"] is False
        assert names["elk_vlm_messages"] is False


# ---------------------------------------------------------------------------
# Phase 3 — VIOS
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhase3Vios:
    def test_unreachable_vios(self, mod):
        with mock.patch.object(mod.requests, "get", side_effect=ConnectionError("down")):
            mod.phase3_vios("https://host/vst")
        assert any(r.name == "vios_reachable" and not r.passed for r in mod.results)

    def test_full_success_path(self, mod):
        sensor = {"name": "sensor_0", "sensorId": "sid0", "state": "on", "sensorIp": "1.2.3.4"}
        rec_payload = {"sid0": {"recording_status": "on"}}

        def fake_get(url, **kwargs):
            if url.endswith("/api/v1/sensor/list"):
                return _mock_response(200, [sensor])
            if url.endswith("/api/v1/sensor/streams"):
                return _mock_response(200, [{"streams": 1}])
            if "record/status" in url:
                return _mock_response(200, rec_payload, text=json.dumps(rec_payload))
            if "live/streams" in url:
                return _mock_response(200, {"live": True})
            base = url.rsplit("/vst", 1)[0]
            if url == base:
                return _mock_response(301, headers={"Location": "/vst/"})
            if url.endswith("/vst/"):
                return _mock_response(200, text="<title>VST UI</title>")
            return _mock_response(404)

        with mock.patch.object(mod.requests, "get", side_effect=fake_get):
            mod.phase3_vios("https://host:30888/vst")
        names = {r.name: r.passed for r in mod.results}
        assert names["vios_reachable"] is True
        assert names["vst_ui_root_redirect"] is True
        assert names["vst_ui_index_accessible"] is True
        assert names["vios_sensors_registered"] is True
        assert names["vios_streams_available"] is True
        assert names["vios_recording_active"] is True
        assert names["vios_livestream_sensor_0"] is True

    def test_redirect_and_ui_failures(self, mod):
        sensor = {"name": "s1", "sensorId": "1", "state": "off"}
        with mock.patch.object(mod.requests, "get") as mget:
            mget.side_effect = [
                _mock_response(200, [sensor]),          # sensor list (reachable)
                ConnectionError("root fail"),           # root redirect
                ConnectionError("ui fail"),               # ui index
            ]
            with mock.patch.object(mod, "_vst_sensor_list", return_value=[sensor]), \
                 mock.patch.object(mod, "_vst_streams", return_value=[]):
                mod.phase3_vios("https://host/vst")
        assert any(r.name == "vst_ui_root_redirect" and not r.passed for r in mod.results)
        assert any(r.name == "vst_ui_index_accessible" and not r.passed for r in mod.results)

    def test_vst_sensor_list_and_streams_helpers(self, mod):
        with mock.patch.object(mod.requests, "get", return_value=_mock_response(200, [{"a": 1}])):
            assert len(mod._vst_sensor_list("https://v")) == 1
            assert mod._vst_streams("https://v")
        with mock.patch.object(mod.requests, "get", side_effect=ConnectionError("x")):
            assert mod._vst_sensor_list("https://v") == []
            assert mod._vst_streams("https://v") == []

    def test_recording_string_status_and_livestream_exception(self, mod, capsys):
        sensor = {"name": "cam1", "sensorId": "1", "state": "on"}
        rec_payload = {"1": "recording"}

        def fake_get(url, **kwargs):
            if url.endswith("/api/v1/sensor/list"):
                return _mock_response(200, [sensor])
            if url.endswith("/api/v1/sensor/streams"):
                return _mock_response(200, [])
            if "record/status" in url:
                return _mock_response(200, rec_payload)
            if "live/streams" in url:
                raise ConnectionError("live down")
            base = url.rsplit("/vst", 1)[0]
            if url == base:
                return _mock_response(301, headers={"Location": "/vst/"})
            if url.endswith("/vst/"):
                return _mock_response(200, text="<title>VST UI</title>")
            return _mock_response(404)

        with mock.patch.object(mod.requests, "get", side_effect=fake_get):
            mod.phase3_vios("https://host/vst")
        assert any(r.name == "vios_recording_active" and r.passed for r in mod.results)
        assert "Livestream check" in capsys.readouterr().out

    def test_recording_nested_status_key(self, mod):
        sensor = {"name": "cam2", "sensorId": "2", "state": "on"}
        rec_payload = {"2": {"status": "active"}}

        def fake_get(url, **kwargs):
            if url.endswith("/api/v1/sensor/list"):
                return _mock_response(200, [sensor])
            if url.endswith("/api/v1/sensor/streams"):
                return _mock_response(200, [{"g": 1}])
            if "record/status" in url:
                return _mock_response(200, rec_payload)
            base = url.rsplit("/vst", 1)[0]
            if url == base:
                return _mock_response(301, headers={"Location": "/vst/"})
            if url.endswith("/vst/"):
                return _mock_response(200, text="<title>VST UI</title>")
            if "live/streams" in url:
                return _mock_response(200, {"s": 1})
            return _mock_response(404)

        with mock.patch.object(mod.requests, "get", side_effect=fake_get):
            mod.phase3_vios("https://host/vst")
        assert any(r.name == "vios_recording_active" and r.passed for r in mod.results)

    def test_livestream_empty_response(self, mod, capsys):
        sensor = {"name": "empty_cam", "sensorId": "9", "state": "on"}

        def fake_get(url, **kwargs):
            if url.endswith("/api/v1/sensor/list"):
                return _mock_response(200, [sensor])
            if url.endswith("/api/v1/sensor/streams"):
                return _mock_response(200, [])
            if "record/status" in url:
                return _mock_response(200, {})
            if "live/streams" in url:
                return _mock_response(200, {})
            base = url.rsplit("/vst", 1)[0]
            if url == base:
                return _mock_response(301, headers={"Location": "/vst/"})
            if url.endswith("/vst/"):
                return _mock_response(200, text="<title>VST UI</title>")
            return _mock_response(404)

        with mock.patch.object(mod.requests, "get", side_effect=fake_get):
            mod.phase3_vios("https://host/vst")
        assert "no active streams" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Phase 4 helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhase4Helpers:
    def test_http_get_ok(self, mod):
        with mock.patch.object(mod.requests, "get", return_value=_mock_response(200)):
            assert mod._http_get_ok("https://x", "lbl") is True
        with mock.patch.object(mod.requests, "get", return_value=_mock_response(500)):
            assert mod._http_get_ok("https://x", "lbl2") is False
        with mock.patch.object(mod.requests, "get", side_effect=TimeoutError("t")):
            assert mod._http_get_ok("https://x", "lbl3") is False

    def test_check_openai_models(self, mod):
        data = {"data": [{"id": "m1"}, {"id": "m2"}]}
        with mock.patch.object(mod.requests, "get", return_value=_mock_response(200, data)):
            assert mod._check_openai_models("https://llm", "models_ok") is True
        with mock.patch.object(mod.requests, "get", return_value=_mock_response(503)):
            assert mod._check_openai_models("https://llm", "models_bad") is False
        with mock.patch.object(mod.requests, "get", side_effect=ConnectionError("x")):
            assert mod._check_openai_models("https://llm", "models_err") is False

    def test_agent_chat(self, mod):
        with mock.patch.object(mod.requests, "post", return_value=_mock_response(200, {"choices": []})):
            assert mod._agent_chat("https://agent", "hi") is not None
        with mock.patch.object(mod.requests, "post", side_effect=ConnectionError("x")):
            assert mod._agent_chat("https://agent", "hi") is None

    def test_agent_reply_text(self, mod):
        assert mod._agent_reply_text(None) == ""
        assert mod._agent_reply_text({"choices": []}) == ""
        resp = {"choices": [{"message": {"content": "hello"}}]}
        assert mod._agent_reply_text(resp) == "hello"

    def test_find_agent_error(self, mod):
        assert mod._find_agent_error("") is None
        assert mod._find_agent_error("All good") is None
        assert mod._find_agent_error("Error generating incident report") == "error generating"
        assert mod._find_agent_error("TRACEBACK here") == "traceback"


# ---------------------------------------------------------------------------
# phase4_vss_agent
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhase4VssAgent:
    _URLS = dict(
        agent_url="https://agent:8000",
        mcp_url="https://mcp:9901",
        llm_base_url="https://llm:30081",
        vlm_base_url="https://vlm:30082",
        vst_url="https://vst:30888/vst",
    )

    def test_remote_llm_vlm_success(self, mod):
        sensors = [{"name": "sensor_0", "sensorId": "0"}]
        chat_replies = [
            {"choices": [{"message": {"content": "Snapshot at https://snap/url"}}]},
            {"choices": [{"message": {"content": "SOP status is active for sensor"}}]},
            {"choices": [{"message": {"content": "Compliance report download .pdf ready"}}]},
        ]

        def fake_get(url, **kwargs):
            if url.endswith("/health"):
                return _mock_response(200)
            if url.endswith("/v1/models"):
                return _mock_response(200, {"data": [{"id": "m"}]})
            return _mock_response(404)

        with mock.patch.object(mod.requests, "get", side_effect=fake_get), \
             mock.patch.object(mod, "_vst_sensor_list", return_value=sensors), \
             mock.patch.object(mod, "_agent_chat", side_effect=chat_replies):
            mod.phase4_vss_agent(llm_mode="remote", vlm_mode="remote", **self._URLS)
        names = {r.name: r.passed for r in mod.results}
        assert names["vss_agent_mcp_health"] is True
        assert names["vss_agent_llm_endpoint"] is True
        assert names["vss_agent_vlm_endpoint"] is True
        assert names["vss_agent_snapshot"] is True
        assert names["vss_agent_video_vlm"] is True
        assert names["vss_agent_report"] is True

    def test_local_vlm_and_error_replies(self, mod):
        with mock.patch.object(mod, "_http_get_ok", return_value=True), \
             mock.patch.object(mod, "_vst_sensor_list", return_value=[]), \
             mock.patch.object(mod, "_agent_chat", return_value=None):
            mod.phase4_vss_agent(llm_mode="local", vlm_mode="local", **self._URLS)
        assert any(r.name == "vss_agent_snapshot" and not r.passed for r in mod.results)

        mod.results.clear()
        err_reply = {"choices": [{"message": {"content": "Error generating incident report: bad"}}]}
        with mock.patch.object(mod, "_http_get_ok", return_value=True), \
             mock.patch.object(mod, "_vst_sensor_list", return_value=[{"name": "s0"}]), \
             mock.patch.object(mod, "_agent_chat", return_value=err_reply):
            mod.phase4_vss_agent(llm_mode="local", vlm_mode="other", **self._URLS)
        assert any(r.name == "vss_agent_snapshot" and not r.passed for r in mod.results)
        assert any(r.name == "vss_agent_video_vlm" and not r.passed for r in mod.results)
        assert any(r.name == "vss_agent_report" and not r.passed for r in mod.results)

    def test_weak_agent_answers_fail_keywords(self, mod):
        weak = {"choices": [{"message": {"content": "I cannot help with that."}}]}
        with mock.patch.object(mod, "_http_get_ok", return_value=True), \
             mock.patch.object(mod, "_check_openai_models", return_value=True), \
             mock.patch.object(mod, "_vst_sensor_list", return_value=[{"name": "s0"}]), \
             mock.patch.object(mod, "_agent_chat", return_value=weak):
            mod.phase4_vss_agent(llm_mode="remote", vlm_mode="remote", **self._URLS)
        assert any(r.name == "vss_agent_snapshot" and not r.passed for r in mod.results)
        assert any(r.name == "vss_agent_video_vlm" and not r.passed for r in mod.results)
        assert any(r.name == "vss_agent_report" and not r.passed for r in mod.results)


# ---------------------------------------------------------------------------
# auto_debug_failures, print_summary, main
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAutoDebugAndSummary:
    def test_auto_debug_no_failures(self, mod):
        mod.record("ok", True)
        mod.auto_debug_failures()  # no-op

    def test_auto_debug_collects_logs(self, mod, capsys):
        mod.record("elasticsearch_reachable", False, "down")
        mod.record("vss_agent_snapshot", False, "fail")
        with mock.patch.object(mod, "run_docker", return_value="log lines\n"):
            mod.auto_debug_failures()
        out = capsys.readouterr().out
        assert "mdx-elastic" in out
        assert "vss-agent" in out

    def test_auto_debug_called_process_error(self, mod, capsys):
        mod.record("kibana_dashboard_fields", False)
        err = mod.subprocess.CalledProcessError(1, "cmd", output=b"partial")
        with mock.patch.object(mod, "run_docker", side_effect=err):
            mod.auto_debug_failures()
        assert "docker logs failed" in capsys.readouterr().out

    def test_auto_debug_generic_exception(self, mod, capsys):
        mod.record("vios_reachable", False)
        with mock.patch.object(mod, "run_docker", side_effect=RuntimeError("boom")):
            mod.auto_debug_failures()
        assert "could not collect logs" in capsys.readouterr().out

    def test_auto_debug_branch_containers(self, mod, capsys):
        mod.record("vios_recording_active", False)
        mod.record("vss_agent_mcp_health", False)
        mod.record("vss_agent_llm_endpoint", False)
        mod.record("vss_agent_vlm_endpoint", False)
        mod.record("container_mdx-ds-sop-1", False)
        with mock.patch.object(mod, "run_docker", return_value="logs\n"):
            mod.auto_debug_failures()
        out = capsys.readouterr().out
        assert "recorder-ms-1-sop" in out
        assert "vss-va-mcp" in out
        assert "mdx-ds-sop-1" in out

    def test_auto_debug_ds_container_branch(self, mod, capsys):
        mod.record("container_mdx-ds-sop-1", False)
        with mock.patch.object(mod, "run_docker", return_value="ds logs\n"):
            mod.auto_debug_failures()
        assert "mdx-ds-sop-1" in capsys.readouterr().out

    def test_auto_debug_llm_vlm_only_branches(self, mod, capsys):
        mod.record("remote_llm_probe", False)
        mod.record("remote_vlm_probe", False)
        with mock.patch.object(mod, "run_docker", return_value="logs\n"):
            mod.auto_debug_failures()
        out = capsys.readouterr().out
        assert "vss-agent" in out
        assert "mdx-ds-sop-1" in out

    def test_print_summary(self, mod, capsys):
        mod.record("a", True, "ok")
        mod.record("b", False, "bad")
        mod.print_summary()
        out = capsys.readouterr().out
        assert "1/2 passed" in out
        assert "FAIL" in out


@pytest.mark.unit
class TestMain:
    def test_main_all_phases_pass(self, mod, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("HOST_IP=127.0.0.1\nELASTIC_SEARCH_PORT=9200\n")
        with mock.patch.object(mod, "phase1_service_health"), \
             mock.patch.object(mod, "phase2_elk_data"), \
             mock.patch.object(mod, "phase3_vios"), \
             mock.patch.object(mod, "phase4_vss_agent"), \
             mock.patch.object(mod, "auto_debug_failures"), \
             mock.patch.object(mod, "print_summary"), \
             mock.patch.object(mod.sys, "argv", [
                 "vss_sop_test.py", "--bp-repo", str(tmp_path), "--env-file", str(env_file),
             ]):
            assert mod.main() == 0

    def test_main_single_phase_and_failures(self, mod, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("EXTERNAL_IP=${HOST_IP}\n")
        mod.record("forced_fail", False)

        def phase1():
            mod.record("p1", False)

        with mock.patch.object(mod, "phase1_service_health", side_effect=phase1), \
             mock.patch.object(mod, "phase2_elk_data"), \
             mock.patch.object(mod, "phase3_vios"), \
             mock.patch.object(mod, "phase4_vss_agent"), \
             mock.patch.object(mod, "auto_debug_failures"), \
             mock.patch.object(mod, "print_summary"), \
             mock.patch.object(mod.sys, "argv", [
                 "vss_sop_test.py", "--bp-repo", str(tmp_path), "--phase", "1",
                 "--env-file", str(env_file),
             ]):
            assert mod.main() == 1

    def test_main_default_env_path(self, mod, tmp_path):
        deploy = tmp_path / "deployments" / "sop"
        deploy.mkdir(parents=True)
        (deploy / ".env").write_text("VSS_AGENT_PORT=9000\n")
        with mock.patch.object(mod, "phase1_service_health"), \
             mock.patch.object(mod, "phase2_elk_data"), \
             mock.patch.object(mod, "phase3_vios"), \
             mock.patch.object(mod, "phase4_vss_agent"), \
             mock.patch.object(mod, "auto_debug_failures"), \
             mock.patch.object(mod, "print_summary"), \
             mock.patch.object(mod.sys, "argv", ["vss_sop_test.py", "--bp-repo", str(tmp_path)]):
            assert mod.main() == 0

    def test_main_entrypoint_sys_exit(self, mod):
        with mock.patch.object(mod, "main", return_value=0) as main_fn, \
             mock.patch.object(mod, "sys") as mock_sys:
            mock_sys.exit.side_effect = SystemExit(0)
            mod.__dict__["__name__"] = "__main__"
            with pytest.raises(SystemExit):
                if mod.__name__ == "__main__":
                    mod.sys.exit(mod.main())
            main_fn.assert_called_once()


@pytest.mark.unit
def test_vss_sop_test_main_runpy(tmp_path):
    import runpy
    env_file = tmp_path / ".env"
    env_file.write_text("HOST_IP=127.0.0.1\n")
    with mock.patch("subprocess.run", side_effect=Exception("no docker")), \
         mock.patch("subprocess.check_output", return_value="mdx-kafka\tUp\n"), \
         mock.patch("requests.get", side_effect=ConnectionError("down")), \
         mock.patch("requests.post", side_effect=ConnectionError("down")), \
         mock.patch.object(sys, "argv", [
             "vss_sop_test.py", "--bp-repo", str(tmp_path),
             "--env-file", str(env_file), "--phase", "2",
         ]):
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(_SCRIPT), run_name="__main__")
        assert exc.value.code == 1


@pytest.mark.unit
class TestKibanaFieldsShim:
    def test_shim_when_import_fails(self, tmp_path, monkeypatch):
        """Exercise inline _KibanaFieldsShim fallback (lines 62-83)."""
        fake_lib = tmp_path / "lib"
        fake_lib.mkdir()
        monkeypatch.syspath_prepend(str(fake_lib))
        # Force kibana_fields import to fail inside a fresh load
        with mock.patch("subprocess.run", side_effect=Exception("no docker")):
            spec = importlib.util.spec_from_file_location("vss_sop_test_shim", _SCRIPT)
            m = importlib.util.module_from_spec(spec)
            # Block kibana_fields
            sys.modules["kibana_fields"] = None  # type: ignore
            try:
                spec.loader.exec_module(m)
            finally:
                sys.modules.pop("kibana_fields", None)
        shim = m.kibana_fields
        present, missing, bad = shim.scan_mapping_fields({"response", "llm"})
        assert "response" in present
        assert "llm" in bad
        errs = shim.scan_runtime_field_map("llm.queries.response", "timestamp")
        assert any("protobuf" in e for e in errs)
