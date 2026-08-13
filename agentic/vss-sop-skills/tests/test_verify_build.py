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

"""Unit tests for verify_build.py."""
import os
import sys
from unittest import mock

import pytest

GOOD_NDJSON = (
    '{"attributes":{"timeFieldName":"@timestamp",'
    '"fields":"response.keyword sensor_id.keyword @timestamp"}}'
)

LOGSTASH_GOOD = """
input {
  kafka {
    topics => ["mdx-vlm-captions"]
    codec => "json"
  }
}
filter {
  if [type] == "mdx-vlm-captions" {
    ruby { code => "first_timestamp start_time" }
    mutate { remove_field => ["kafka", "message", "@version"] }
  }
}
output {
  if [type] == "other" {
    elasticsearch { document_id => "x" }
  }
}
"""

SDR_RECORDER_GOOD = """
environment:
  - WDM_WL_ADD_URL=/api/v1/record/stream/add
  - WDM_WL_CHANGE_ID_ADD=camera_streaming
  - STORAGE_MODULE_ENDPOINT=http://localhost:30011
  - WDM_CLUSTER_CONTAINER_NAMES='["recorder-ms-1"]'
"""

SDR_MODULE_GOOD = """
environment:
  - WDM_CLUSTER_CONTAINER_NAMES='["module-ms-1"]'
"""

SDR_RTSP_GOOD = """
environment:
  - HTTP_PORT=${RTSP_SERVER_HTTP_PORT_1}
  - WDM_CLUSTER_CONTAINER_NAMES='["rtspserver-ms-1"]'
"""

NGINX_GOOD = """
location = / {
    return 301 /vst/;
}
location /vst/ {
    proxy_pass http://sensor-ms/;
}
"""


def _touch(base: os.PathLike, rel: str, content: str = "") -> None:
    path = os.path.join(base, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def build_passing_tree(base):
    """Minimal deployments tree that satisfies all verify_build checks."""
    paths = [
        "deployments/sop/vss-agent/configs/.keep",
        "deployments/sop/vss-agent/patches/.keep",
        "deployments/sop/vss-agent/templates/.keep",
        "deployments/ds/ds-sop/.keep",
        "deployments/ds/compose.yml",
        "deployments/agents/vss-agent/vss-agent-docker-compose.yml",
        "deployments/nim/compose.yml",
        "deployments/compose.yml",
        "deployments/foundational/mdx-foundational.yml",
        "deployments/foundational/elk/configs/mdx-kafka-logstash.conf",
        "deployments/sop/sop-app/kibana-dashboard/sop-kibana-objects.ndjson",
        "deployments/nim/nemotron-nano-v2/compose.yml",
        "deployments/agents/agent_ui/compose.yml",
        "deployments/agents/compose.yml",
        "deployments/vst/compose.yml",
        "deployments/vst/sop/vst/docker-compose.yaml",
        "deployments/vst/sop/vst/.env",
        "deployments/vst/sop/vst/configs/nginx.conf",
        "deployments/vst/sop/vst/minio/minio-compose.yaml",
    ]
    for p in paths:
        _touch(base, p)

    _touch(base, "deployments/compose.yml", "include:\n  - ./sop/compose.yml\n")
    _touch(
        base,
        "deployments/sop/compose.yml",
        'profiles: ["bp_sop_2d"]\nimage: nvcr.io/test/sop:1\n',
    )
    _touch(
        base,
        "deployments/foundational/mdx-foundational.yml",
        "bp_sop_2d\n"
        "docker.elastic.co/elasticsearch/elasticsearch:9.3.0\n"
        "mdx-vlm-captions\n",
    )
    _touch(base, "deployments/foundational/elk/configs/mdx-kafka-logstash.conf", LOGSTASH_GOOD)
    _touch(base, "deployments/sop/sop-app/kibana-dashboard/sop-kibana-objects.ndjson", GOOD_NDJSON)
    _touch(
        base,
        "deployments/nim/compose.yml",
        "services:\n  nemotron-nano-v2:\n    image: nemotron\n",
    )
    _touch(
        base,
        "deployments/agents/vss-agent/vss-agent-docker-compose.yml",
        'profiles: ["bp_sop_2d"]\npatches/tools.py\n',
    )
    _touch(base, "deployments/agents/agent_ui/compose.yml", 'profiles: ["bp_sop_2d"]\n')
    _touch(base, "deployments/agents/compose.yml", "# ai-agents disabled\n")
    _touch(base, "deployments/vst/sop/vst/configs/nginx.conf", NGINX_GOOD)

    for mod in ("rtspserver", "recorder", "replaystream", "livestream"):
        sdr_dir = f"deployments/vst/sop/vst/sdr-{mod}-http"
        if mod == "recorder":
            sdr_content = SDR_RECORDER_GOOD
        elif mod == "rtspserver":
            sdr_content = SDR_RTSP_GOOD
        else:
            sdr_content = SDR_MODULE_GOOD.replace("module", mod)
        _touch(base, f"{sdr_dir}/sdr-compose.yaml", sdr_content)
        _touch(base, f"{sdr_dir}/envoy.yaml", "node: {}\n")
        _touch(base, f"{sdr_dir}/sdr-config/docker_cluster_config.json", "{}")


@pytest.mark.unit
class TestReadHelper:
    def test_read_returns_none_for_missing(self, verify_build, tmp_path):
        assert verify_build._read(str(tmp_path / "nope")) is None

    def test_read_returns_content(self, verify_build, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello")
        assert verify_build._read(str(f)) == "hello"


@pytest.mark.unit
class TestVerifyStructure:
    def test_passes_with_full_tree(self, verify_build, tmp_path, capsys):
        build_passing_tree(tmp_path)
        assert verify_build.verify_structure(str(tmp_path)) is True

    def test_fails_when_paths_missing(self, verify_build, tmp_path, capsys):
        assert verify_build.verify_structure(str(tmp_path)) is False


@pytest.mark.unit
class TestVerifyComposeInclude:
    def test_pass(self, verify_build, tmp_path):
        _touch(tmp_path, "deployments/compose.yml", "include:\n  - ./sop/compose.yml\n")
        assert verify_build.verify_compose_include(str(tmp_path)) is True

    def test_fail_missing_file(self, verify_build, tmp_path):
        assert verify_build.verify_compose_include(str(tmp_path)) is False

    def test_fail_no_sop_include(self, verify_build, tmp_path):
        _touch(tmp_path, "deployments/compose.yml", "include: []\n")
        assert verify_build.verify_compose_include(str(tmp_path)) is False


@pytest.mark.unit
class TestVerifyProfiles:
    def test_pass_when_bp_sop_2d_found(self, verify_build, tmp_path):
        _touch(tmp_path, "deployments/sop/x.yml", 'profiles: ["bp_sop_2d"]\n')
        assert verify_build.verify_profiles(str(tmp_path)) is True

    def test_fail_when_no_profile(self, verify_build, tmp_path):
        os.makedirs(tmp_path / "deployments", exist_ok=True)
        assert verify_build.verify_profiles(str(tmp_path)) is False

    def test_fail_when_deployments_missing(self, verify_build, tmp_path):
        assert verify_build.verify_profiles(str(tmp_path)) is False


@pytest.mark.unit
class TestCheckContainerVersions:
    def test_prints_images_when_found(self, verify_build, tmp_path, capsys):
        _touch(tmp_path, "deployments/sop/svc.yml", "image: nvcr.io/test:1\n")
        verify_build.check_container_versions(str(tmp_path))
        assert "nvcr.io/test:1" in capsys.readouterr().out

    def test_warns_when_no_images(self, verify_build, tmp_path, capsys):
        os.makedirs(tmp_path / "deployments/sop", exist_ok=True)
        verify_build.check_container_versions(str(tmp_path))
        assert "No direct" in capsys.readouterr().out


@pytest.mark.unit
class TestVerifyFoundational:
    def test_profiles_pass(self, verify_build, tmp_path):
        _touch(tmp_path, "deployments/foundational/mdx-foundational.yml", "bp_sop_2d\n")
        assert verify_build.verify_foundational_profiles(str(tmp_path)) is True

    def test_profiles_fail_missing_file(self, verify_build, tmp_path):
        assert verify_build.verify_foundational_profiles(str(tmp_path)) is False

    def test_profiles_fail_minimal_profile(self, verify_build, tmp_path):
        _touch(tmp_path, "deployments/foundational/mdx-foundational.yml", "MINIMAL_PROFILE\n")
        assert verify_build.verify_foundational_profiles(str(tmp_path)) is False

    def test_es_image_stock_tag(self, verify_build, tmp_path):
        _touch(
            tmp_path,
            "deployments/foundational/mdx-foundational.yml",
            "docker.elastic.co/elasticsearch/elasticsearch:9.3.0\n",
        )
        assert verify_build.verify_foundational_es_image(str(tmp_path)) is True

    def test_es_image_other_elasticsearch_tag(self, verify_build, tmp_path):
        _touch(tmp_path, "deployments/foundational/mdx-foundational.yml", "image: my-elasticsearch:8\n")
        assert verify_build.verify_foundational_es_image(str(tmp_path)) is True

    def test_es_image_missing(self, verify_build, tmp_path):
        _touch(tmp_path, "deployments/foundational/mdx-foundational.yml", "no es here\n")
        assert verify_build.verify_foundational_es_image(str(tmp_path)) is False

    def test_es_custom_dockerfile_still_present(self, verify_build, tmp_path):
        _touch(
            tmp_path,
            "deployments/foundational/mdx-foundational.yml",
            "docker.elastic.co/elasticsearch/elasticsearch:9.3.0\n",
        )
        _touch(tmp_path, "deployments/foundational/Dockerfiles/elasticsearch.Dockerfile", "FROM es\n")
        assert verify_build.verify_foundational_es_image(str(tmp_path)) is False

    def test_kafka_topic_pass(self, verify_build, tmp_path):
        _touch(tmp_path, "deployments/foundational/mdx-foundational.yml", "mdx-vlm-captions\n")
        assert verify_build.verify_foundational_kafka_topic(str(tmp_path)) is True

    def test_kafka_topic_fail_embed_filtered(self, verify_build, tmp_path):
        _touch(tmp_path, "deployments/foundational/mdx-foundational.yml", "mdx-vlm-captions\n")
        _touch(tmp_path, "deployments/foundational/other.txt", "embed-filtered\n")
        assert verify_build.verify_foundational_kafka_topic(str(tmp_path)) is False

    def test_logstash_pass(self, verify_build, tmp_path):
        _touch(
            tmp_path,
            "deployments/foundational/elk/configs/mdx-kafka-logstash.conf",
            LOGSTASH_GOOD,
        )
        assert verify_build.verify_foundational_logstash(str(tmp_path)) is True

    def test_logstash_fail_missing_input(self, verify_build, tmp_path):
        _touch(tmp_path, "deployments/foundational/elk/configs/mdx-kafka-logstash.conf", "input {}\n")
        assert verify_build.verify_foundational_logstash(str(tmp_path)) is False


@pytest.mark.unit
class TestVerifyKibanaDashboard:
    def test_pass(self, verify_build, tmp_path):
        _touch(
            tmp_path,
            "deployments/sop/sop-app/kibana-dashboard/sop-kibana-objects.ndjson",
            GOOD_NDJSON,
        )
        assert verify_build.verify_kibana_dashboard(str(tmp_path)) is True

    def test_fail_bad_fields(self, verify_build, tmp_path):
        _touch(
            tmp_path,
            "deployments/sop/sop-app/kibana-dashboard/sop-kibana-objects.ndjson",
            "llm.queries.response",
        )
        assert verify_build.verify_kibana_dashboard(str(tmp_path)) is False

    def test_fail_missing_file(self, verify_build, tmp_path):
        assert verify_build.verify_kibana_dashboard(str(tmp_path)) is False


@pytest.mark.unit
class TestVerifyNim:
    def test_pass(self, verify_build, tmp_path):
        build_passing_tree(tmp_path)
        assert verify_build.verify_nim(str(tmp_path)) is True

    def test_fail_missing_nim_dir(self, verify_build, tmp_path):
        assert verify_build.verify_nim(str(tmp_path)) is False

    def test_fail_fallback_override_present(self, verify_build, tmp_path):
        build_passing_tree(tmp_path)
        _touch(tmp_path, "deployments/nim/fallback-override.env", "X=1\n")
        assert verify_build.verify_nim(str(tmp_path)) is False

    def test_fail_hw_other_files(self, verify_build, tmp_path):
        build_passing_tree(tmp_path)
        _touch(tmp_path, "deployments/nim/hw-OTHER-x.yml", "x\n")
        assert verify_build.verify_nim(str(tmp_path)) is False

    def test_fail_fp8_dir_present(self, verify_build, tmp_path):
        build_passing_tree(tmp_path)
        os.makedirs(tmp_path / "deployments/nim/nvidia-nemotron-nano-9b-v2-fp8")
        assert verify_build.verify_nim(str(tmp_path)) is False


@pytest.mark.unit
class TestVerifyAgents:
    def test_pass(self, verify_build, tmp_path):
        build_passing_tree(tmp_path)
        assert verify_build.verify_agents(str(tmp_path)) is True

    def test_fail_missing_compose(self, verify_build, tmp_path):
        assert verify_build.verify_agents(str(tmp_path)) is False

    def test_fail_agent_eval_present(self, verify_build, tmp_path):
        build_passing_tree(tmp_path)
        _touch(
            tmp_path,
            "deployments/agents/vss-agent/vss-agent-docker-compose.yml",
            "agent-eval\nbp_sop_2d\npatches/tools.py\n",
        )
        assert verify_build.verify_agents(str(tmp_path)) is False


@pytest.mark.unit
class TestVerifyVios:
    def test_structure_skips_when_sop_vst_missing(self, verify_build, tmp_path):
        assert verify_build.verify_vios_structure(str(tmp_path)) is True

    def test_structure_pass(self, verify_build, tmp_path):
        build_passing_tree(tmp_path)
        assert verify_build.verify_vios_structure(str(tmp_path)) is True

    def test_structure_fail_leftover(self, verify_build, tmp_path):
        build_passing_tree(tmp_path)
        os.makedirs(tmp_path / "deployments/vst/developer")
        assert verify_build.verify_vios_structure(str(tmp_path)) is False

    def test_sdr_recorder_pass(self, verify_build, tmp_path):
        build_passing_tree(tmp_path)
        assert verify_build.verify_sdr_recorder(str(tmp_path)) is True

    def test_sdr_recorder_skips_when_missing(self, verify_build, tmp_path):
        assert verify_build.verify_sdr_recorder(str(tmp_path)) is True

    def test_sdr_recorder_fail_proxy_url(self, verify_build, tmp_path):
        build_passing_tree(tmp_path)
        _touch(
            tmp_path,
            "deployments/vst/sop/vst/sdr-recorder-http/sdr-compose.yaml",
            "WDM_WL_ADD_URL=/api/v1/proxy/\nWDM_WL_CHANGE_ID_ADD=camera_streaming\n"
            "STORAGE_MODULE_ENDPOINT=x\n",
        )
        assert verify_build.verify_sdr_recorder(str(tmp_path)) is False

    def test_sdr_double_quotes_pass(self, verify_build, tmp_path):
        build_passing_tree(tmp_path)
        assert verify_build.verify_sdr_double_quotes(str(tmp_path)) is True

    def test_sdr_double_quotes_fail(self, verify_build, tmp_path):
        build_passing_tree(tmp_path)
        _touch(
            tmp_path,
            "deployments/vst/sop/vst/sdr-rtspserver-http/sdr-compose.yaml",
            'WDM_CLUSTER_CONTAINER_NAMES=""\n',
        )
        assert verify_build.verify_sdr_double_quotes(str(tmp_path)) is False

    def test_sdr_http_port_pass(self, verify_build, tmp_path):
        build_passing_tree(tmp_path)
        assert verify_build.verify_sdr_http_port_naming(str(tmp_path)) is True

    def test_sdr_http_port_fail_buggy_var(self, verify_build, tmp_path):
        build_passing_tree(tmp_path)
        _touch(
            tmp_path,
            "deployments/vst/sop/vst/sdr-rtspserver-http/sdr-compose.yaml",
            "HTTP_PORT=${RTSPSERVER_HTTP_PORT_1}\nWDM_CLUSTER_CONTAINER_NAMES='[]'\n",
        )
        assert verify_build.verify_sdr_http_port_naming(str(tmp_path)) is False

    def test_nginx_pass(self, verify_build, tmp_path):
        build_passing_tree(tmp_path)
        assert verify_build.verify_nginx_routing(str(tmp_path)) is True

    def test_nginx_fail(self, verify_build, tmp_path):
        build_passing_tree(tmp_path)
        _touch(tmp_path, "deployments/vst/sop/vst/configs/nginx.conf", "empty\n")
        assert verify_build.verify_nginx_routing(str(tmp_path)) is False


@pytest.mark.unit
class TestVerifyBuildMain:
    def test_main_all_passes(self, verify_build, tmp_path):
        build_passing_tree(tmp_path)
        with mock.patch.object(sys, "argv", ["verify_build.py", str(tmp_path)]):
            with pytest.raises(SystemExit) as exc:
                verify_build.main()
            assert exc.value.code == 0

    def test_main_all_fails(self, verify_build, tmp_path):
        with mock.patch.object(sys, "argv", ["verify_build.py", str(tmp_path)]):
            with pytest.raises(SystemExit) as exc:
                verify_build.main()
            assert exc.value.code == 1

    def test_main_single_component(self, verify_build, tmp_path):
        build_passing_tree(tmp_path)
        with mock.patch.object(
            sys,
            "argv",
            ["verify_build.py", str(tmp_path), "--component", "nim"],
        ):
            with pytest.raises(SystemExit) as exc:
                verify_build.main()
            assert exc.value.code == 0

    def test_print_section(self, verify_build, capsys):
        verify_build.print_section("Test Title")
        out = capsys.readouterr().out
        assert "Test Title" in out
