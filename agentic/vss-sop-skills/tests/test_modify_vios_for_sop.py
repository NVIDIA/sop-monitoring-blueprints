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

"""Unit tests for modify_vios_for_sop.py."""
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

SKILLS = Path(__file__).resolve().parents[1]
REAL_REFS = SKILLS / "vss-sop-build" / "references"


def _vst_tree(base: Path) -> tuple[Path, Path]:
    vst = base / "deployments" / "vst"
    sop_vst = vst / "sop" / "vst"
    configs = sop_vst / "configs"
    configs.mkdir(parents=True)
    sop_vst.mkdir(parents=True, exist_ok=True)
    (configs / "adaptor_config.json").write_text(
        json.dumps({"adaptors": [{"name": "vst_rtsp", "media_adaptor_lib_path": "/lib.so"}]})
    )
    (configs / "rtsp_streams.json").write_text(json.dumps({"enabled": False, "max_stream_count": 1}))
    for name in ("vst_config.json", "vst_config_redis.json", "vst_config_kafka.json"):
        data = {
            "ai_bridge_endpoint": "https://old",
            "halo_safety_x": True,
            "data": {"nv_streamer_loop_playback": False},
            "network": {"rtsp_server_instances_count": 3},
        }
        (configs / name).write_text(json.dumps(data))
    (vst / "scripts").mkdir()
    (vst / "scripts" / "legacy.sh").write_text("# old\n")
    monolith = sop_vst / "sdr-streamprocessing"
    monolith.mkdir()
    (monolith / "old.yaml").write_text("x\n")
    return vst, sop_vst


def _stub_refs(base: Path) -> Path:
    refs = base / "references"
    for mod in ("rtspserver", "recorder", "replaystream", "livestream"):
        p = (
            refs / "deployments" / "vst" / "sop" / "vst"
            / f"sdr-{mod}-http" / "sdr-config" / "docker_cluster_config.json"
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"module": mod}))
    for name, rel in (
        ("sop-docker-compose.yaml", "configs/vios/sop-docker-compose.yaml"),
        ("vst-top-level-compose.yml", "configs/vios/vst-top-level-compose.yml"),
        ("minio-server.service.yml", "configs/vios/minio-server.service.yml"),
    ):
        dest = refs / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"# stub {name}\n")
    return refs


@pytest.mark.unit
class TestCopySdrClusterConfigs:
    def test_copies_all_modules(self, modify_vios, tmp_path):
        vst, sop_vst = _vst_tree(tmp_path)
        refs = _stub_refs(tmp_path)
        modify_vios.copy_sdr_cluster_configs(refs, sop_vst)
        for mod in ("rtspserver", "recorder", "replaystream", "livestream"):
            dest = sop_vst / f"sdr-{mod}-http" / "sdr-config" / "docker_cluster_config.json"
            assert dest.exists()
            assert json.loads(dest.read_text())["module"] == mod

    def test_warns_when_reference_missing(self, modify_vios, tmp_path, capsys):
        _, sop_vst = _vst_tree(tmp_path)
        modify_vios.copy_sdr_cluster_configs(tmp_path / "empty_refs", sop_vst)
        assert "WARNING" in capsys.readouterr().out


@pytest.mark.unit
class TestWriteTopLevelCompose:
    def test_copies_compose(self, modify_vios, tmp_path, capsys):
        vst, _ = _vst_tree(tmp_path)
        refs = _stub_refs(tmp_path)
        modify_vios.write_top_level_compose(refs, vst)
        assert (vst / "compose.yml").exists()
        assert "Wrote top-level" in capsys.readouterr().out

    def test_warns_when_missing(self, modify_vios, tmp_path, capsys):
        vst, _ = _vst_tree(tmp_path)
        modify_vios.write_top_level_compose(tmp_path / "nope", vst)
        assert "WARNING" in capsys.readouterr().out


@pytest.mark.unit
class TestRemoveUpstreamLeftovers:
    def test_removes_monolith_and_scripts(self, modify_vios, tmp_path, capsys):
        vst, sop_vst = _vst_tree(tmp_path)
        modify_vios.remove_upstream_leftovers(vst, sop_vst)
        assert not (sop_vst / "sdr-streamprocessing").exists()
        assert not (vst / "scripts").exists()
        assert "Removed" in capsys.readouterr().out


@pytest.mark.unit
class TestWriteSopEnv:
    def test_writes_env_file(self, modify_vios, tmp_path, capsys):
        _, sop_vst = _vst_tree(tmp_path)
        modify_vios.write_sop_env(sop_vst)
        env = (sop_vst / ".env").read_text()
        assert "VST_INGRESS_HTTP_PORT=30888" in env
        assert "bp_sop" not in env
        assert "Wrote .env" in capsys.readouterr().out


@pytest.mark.unit
class TestWriteSopDockerCompose:
    def test_copies_reference_compose(self, modify_vios, tmp_path, capsys):
        _, sop_vst = _vst_tree(tmp_path)
        refs = _stub_refs(tmp_path)
        modify_vios.write_sop_docker_compose(refs, sop_vst)
        assert (sop_vst / "docker-compose.yaml").exists()

    def test_warns_when_reference_missing(self, modify_vios, tmp_path, capsys):
        _, sop_vst = _vst_tree(tmp_path)
        modify_vios.write_sop_docker_compose(tmp_path / "nope", sop_vst)
        assert "WARNING" in capsys.readouterr().out


@pytest.mark.unit
class TestWriteNginxConf:
    def test_creates_nginx_with_routes(self, modify_vios, tmp_path):
        _, sop_vst = _vst_tree(tmp_path)
        configs = sop_vst / "configs"
        modify_vios.write_nginx_conf(configs)
        content = (configs / "nginx.conf").read_text()
        assert "return 301 /vst/;" in content
        assert "location /vst/" in content


@pytest.mark.unit
class TestModifyJsonConfigs:
    def test_modify_adaptor_config(self, modify_vios, tmp_path):
        _, sop_vst = _vst_tree(tmp_path)
        modify_vios.modify_adaptor_config(sop_vst / "configs")
        data = json.loads((sop_vst / "configs" / "adaptor_config.json").read_text())
        assert "media_adaptor_lib_path" not in data["adaptors"][0]

    def test_modify_adaptor_config_skip_missing(self, modify_vios, tmp_path, capsys):
        modify_vios.modify_adaptor_config(tmp_path / "nope")
        assert "SKIP" in capsys.readouterr().out

    def test_modify_rtsp_streams(self, modify_vios, tmp_path):
        _, sop_vst = _vst_tree(tmp_path)
        modify_vios.modify_rtsp_streams(sop_vst / "configs")
        data = json.loads((sop_vst / "configs" / "rtsp_streams.json").read_text())
        assert data["enabled"] is True
        assert data["max_stream_count"] == 100

    def test_modify_vst_configs_all_files(self, modify_vios, tmp_path):
        _, sop_vst = _vst_tree(tmp_path)
        modify_vios.modify_vst_configs(sop_vst / "configs")
        for name in ("vst_config.json", "vst_config_redis.json", "vst_config_kafka.json"):
            data = json.loads((sop_vst / "configs" / name).read_text())
            assert data["max_webrtc_out_connections"] == 8
            assert "ai_bridge_endpoint" not in data
            assert "halo_safety_x" not in data
            assert data["data"]["nv_streamer_loop_playback"] is True
            assert data["network"]["rtsp_server_instances_count"] == 1
        kafka = json.loads((sop_vst / "configs" / "vst_config_kafka.json").read_text())
        assert kafka["use_webrtc_hw_dec"] is False
        redis = json.loads((sop_vst / "configs" / "vst_config_redis.json").read_text())
        assert redis["observability"]["enable_telemetry"] is False
        main = json.loads((sop_vst / "configs" / "vst_config.json").read_text())
        assert main["webrtc_in_video_degradation_preference"] == "detail"

    def test_modify_vst_configs_without_nested_sections(self, modify_vios, tmp_path):
        configs = tmp_path / "configs"
        configs.mkdir()
        (configs / "vst_config.json").write_text("{}")
        modify_vios.modify_vst_configs(configs)
        data = json.loads((configs / "vst_config.json").read_text())
        assert data["nv_streamer_loop_playback"] is True
        assert data["rtsp_server_instances_count"] == 1


@pytest.mark.unit
class TestSdrYamlGenerators:
    def test_envoy_yaml_contains_ports(self, modify_vios):
        yaml_text = modify_vios._envoy_yaml("recorder", 10001, 4002, 30006)
        assert "port_value: 10001" in yaml_text
        assert "recorder-ms_route" in yaml_text

    def test_sdr_compose_recorder_has_storage_endpoint(self, modify_vios):
        compose = modify_vios._sdr_compose(
            module="recorder",
            image_var="VST_RECORDER_IMAGE",
            http_port_var="RECORDER_HTTP_PORT_1",
            sdr_port=4002,
            base_id_var="RECORDER_BASE_ID",
            wl_add_url="/api/v1/record/stream/add",
            wl_delete_url="/api/v1/record/stream/",
            wl_health_url="/api/v1/record/configuration",
            wl_change_id_add="camera_streaming",
        )
        assert "STORAGE_MODULE_ENDPOINT" in compose
        assert "camera_streaming" in compose

    def test_sdr_compose_rtspserver_has_rtsp_port(self, modify_vios):
        compose = modify_vios._sdr_compose(
            module="rtspserver",
            image_var="VST_RTSPSERVER_IMAGE",
            http_port_var="RTSP_SERVER_HTTP_PORT_1",
            sdr_port=4003,
            base_id_var="RTSPSERVER_BASE_ID",
            wl_add_url="/api/v1/proxy/stream/add",
            wl_delete_url="/api/v1/proxy/stream/",
            wl_health_url="/api/v1/proxy/configuration",
            wl_change_id_add="camera_proxy",
        )
        assert "RTSP_SERVER_PORT" in compose


@pytest.mark.unit
class TestWriteSdrModules:
    def test_creates_all_module_dirs(self, modify_vios, tmp_path, capsys):
        _, sop_vst = _vst_tree(tmp_path)
        modify_vios.write_sdr_modules(sop_vst)
        for mod in ("rtspserver", "recorder", "replaystream", "livestream"):
            d = sop_vst / f"sdr-{mod}-http"
            assert (d / "sdr-compose.yaml").exists()
            assert (d / "envoy.yaml").exists()
        assert "Created sdr-" in capsys.readouterr().out


@pytest.mark.unit
class TestWriteMinioCompose:
    def test_copies_minio_compose(self, modify_vios, tmp_path, capsys):
        _, sop_vst = _vst_tree(tmp_path)
        refs = _stub_refs(tmp_path)
        modify_vios.write_minio_compose(refs, sop_vst)
        assert (sop_vst / "minio" / "minio-compose.yaml").exists()

    def test_warns_when_missing(self, modify_vios, tmp_path, capsys):
        _, sop_vst = _vst_tree(tmp_path)
        modify_vios.write_minio_compose(tmp_path / "nope", sop_vst)
        assert "WARNING" in capsys.readouterr().out


@pytest.mark.unit
class TestModifyViosMain:
    def test_main_with_real_references(self, modify_vios, tmp_path, capsys):
        _vst_tree(tmp_path)
        with mock.patch.object(sys, "argv", ["modify_vios_for_sop.py", str(tmp_path)]):
            modify_vios.main()
        out = capsys.readouterr().out
        assert "All SOP modifications applied successfully" in out
        sop_vst = tmp_path / "deployments" / "vst" / "sop" / "vst"
        assert (sop_vst / "docker-compose.yaml").exists()
        assert (sop_vst / "configs" / "nginx.conf").exists()

    def test_main_exits_when_sop_vst_missing(self, modify_vios, tmp_path):
        with mock.patch.object(sys, "argv", ["modify_vios_for_sop.py", str(tmp_path)]):
            with pytest.raises(SystemExit) as exc:
                modify_vios.main()
            assert exc.value.code == 1

    def test_main_uses_legacy_refs_fallback(self, modify_vios, tmp_path, monkeypatch, capsys):
        _vst_tree(tmp_path)
        legacy = tmp_path / ".claude" / "skills" / "vss-sop-build" / "references"
        _stub_refs(legacy.parent.parent.parent)  # creates under tmp_path/.claude/...
        # Point __file__ parent.parent to a path with no references
        fake_script = tmp_path / "fake" / "scripts" / "modify_vios_for_sop.py"
        fake_script.parent.mkdir(parents=True)
        fake_script.write_text("# stub\n")
        monkeypatch.setattr(
            modify_vios,
            "__file__",
            str(fake_script),
        )
        legacy_root = tmp_path / ".claude" / "skills" / "vss-sop-build" / "references"
        _stub_refs(tmp_path)  # ensure stub exists at legacy path
        # Move stub refs to legacy location
        import shutil
        src = tmp_path / "references"
        if src.exists():
            shutil.rmtree(legacy_root, ignore_errors=True)
            shutil.copytree(src, legacy_root)
        with mock.patch.object(sys, "argv", ["modify_vios_for_sop.py", str(tmp_path)]):
            modify_vios.main()
        assert "All SOP modifications applied successfully" in capsys.readouterr().out
