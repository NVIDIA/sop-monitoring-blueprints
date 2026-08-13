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

"""Unit tests for modify_foundational_for_sop.py."""
import sys
from pathlib import Path
from unittest import mock

import pytest

FOUNDATIONAL_YML = """\
services:
  elasticsearch:
    build:
      context: .
      dockerfile: Dockerfiles/elasticsearch.Dockerfile
      network: "host"
    image: elasticsearch
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9200/_cluster/health"]
      interval: 10s
      timeout: 10s
      retries: 15
  elasticsearch-init:
    container_name: mdx-elasticsearch-init
    environment:
      - BP_PROFILE=default
      - ELASTICSEARCH_CONNECTION_MAX_ATTEMPTS=5
    build:
      context: .
      network: "host"
  broker-health-check:
    environment:
      BOOTSTRAP_HOST: localhost
      KAFKA_PORT: 9092
      REDIS_PORT: 6379
    build:
      context: .
      network: "host"
  kibana:
    container_name: mdx-kibana
    healthcheck:
      retries: 30
      start_period: 60s
    environment:
      SERVER_PUBLICBASEURL: http://localhost
      SERVER_SECURITYRESPONSEHEADERS_DISABLEEMBEDDING: "true"
      CSP_STRICT: "false"
  kafka-topic-init:
    command: >
      [{"name": "mdx-vlm"},
        {"name": "mdx-embed-filtered"}]
"""

KAFKA_LOGSTASH = """\
input {
\tkafka {
\t\ttype => "mdx-embed-filtered"
\t\ttopics => ["mdx-embed-filtered"]
\t}
}
filter {
\tjson { source => "message" }
\t# Formatting timestamp
\truby {
\t\tcode => "event.set('timestamp',(((event.get('[timestamp][seconds]').to_f)*1000) +((event.get('[timestamp][nanos]').to_f) * (10 ** -6)).floor()))"
\t}
\tdate {
\t\tmatch => [ "timestamp","UNIX_MS" ]
\t\ttarget => "timestamp"
\t\ttimezone => "UTC"
\t}
\tgrok {
\t\tmatch => ["timestamp", "%{YEAR:[@metadata][year]}-%{MONTHNUM:[@metadata][month]}-%{MONTHDAY:[@metadata][day]}T%{GREEDYDATA}"]
\t}
\tmutate {
\t\tremove_field => ["kafka", "message", "@timestamp", "@version"]
\t}
}
output {}
"""

ES_TEMPLATE_SCRIPT = """\
#!/bin/bash
ELASTICSEARCH_CONNECTION_RETRY_ATTEMPTS=3
ELASTICSEARCH_CONNECTION_MAX_ATTEMPTS=5
ELASTICSEARCH_URL=http://es:9200
BP_PROFILE=bp_developer
echo "BP_PROFILE: $BP_PROFILE"
ELASTICSEARCH_RTVI_CV_EMBEDDINGS_DIM=512
ELASTICSEARCH_VISION_LLM_EMBEDDINGS_DIM=512
    if [[ "${BP_PROFILE:-}" == "bp_developer_search" ]]; then
        echo "search behavior"
        echo "Successfully created behavior search"
    else
        create_index_template "mdx_behavior_template"
    fi

    create_index_template "mdx_events_template"
    if [[ "${BP_PROFILE:-}" == "bp_developer_search" ]]; then
        echo "search raw"
        echo "Successfully created raw search"
    else
        create_index_template "mdx_raw_template"
    fi

    create_index_template "mdx_incidents_template"
    create_index_template "mdx_embed_filtered_template" '"mdx-embed-filtered-*"' mdx-embed-filtered-ilm-policy
    dims '"${ELASTICSEARCH_RTVI_CV_EMBEDDINGS_DIM}"' '"${ELASTICSEARCH_VISION_LLM_EMBEDDINGS_DIM}"'
    echo "Successfully created index templates."
"""

ES_ILM_SCRIPT = """\
ELASTICSEARCH_CONNECTION_RETRY_ATTEMPTS=1
ELASTICSEARCH_CONNECTION_MAX_ATTEMPTS=2
ELASTICSEARCH_URL=http://old:9200
ELASTICSEARCH_ILM_MIN_AGE=8h
    create_ilm_policy "mdx-embed-filtered-ilm-policy"
"""

ES_INGEST_SCRIPT = """\
ELASTICSEARCH_URL=http://old:9200
ELASTICSEARCH_CONNECTION_RETRY_ATTEMPTS=1
ELASTICSEARCH_CONNECTION_MAX_ATTEMPTS=2
"""

KAFKA_TOPICS_SCRIPT = """\
KAFKA_HOST=${BOOTSTRAP_HOST:-localhost}
KAFKA_PORT=${KAFKA_PORT:-9092}
kafka-topics --bootstrap-server $KAFKA_HOST:$KAFKA_PORT
"""

BROKER_KAFKA_SCRIPT = """\
connect ${KAFKA_HOST}:${KAFKA_PORT}
"""

BROKER_REDIS_SCRIPT = """\
connect ${REDIS_HOST} ${REDIS_PORT}
"""


def _foundational_tree(base: Path) -> Path:
    f = base / "deployments" / "foundational"
    (f / "elk" / "configs").mkdir(parents=True)
    (f / "elk" / "init-scripts").mkdir(parents=True)
    (f / "kafka" / "init-scripts").mkdir(parents=True)
    (f / "broker-health-check" / "scripts").mkdir(parents=True)
    (f / "Dockerfiles").mkdir(parents=True)
    (f / "mdx-foundational.yml").write_text(FOUNDATIONAL_YML)
    (f / "elk" / "configs" / "mdx-kafka-logstash.conf").write_text(KAFKA_LOGSTASH)
    (f / "elk" / "configs" / "mdx-redis-logstash.conf").write_text("mdx-embed-filtered\n")
    (f / "elk" / "init-scripts" / "elasticsearch-template-creation.sh").write_text(ES_TEMPLATE_SCRIPT)
    (f / "elk" / "init-scripts" / "elasticsearch-ilm-policy-creation.sh").write_text(ES_ILM_SCRIPT)
    (f / "elk" / "init-scripts" / "elasticsearch-ingest-pipeline-creation.sh").write_text(ES_INGEST_SCRIPT)
    (f / "kafka" / "init-scripts" / "create-kafka-topics.sh").write_text(KAFKA_TOPICS_SCRIPT)
    (f / "broker-health-check" / "scripts" / "check-kafka-health.sh").write_text(BROKER_KAFKA_SCRIPT)
    (f / "broker-health-check" / "scripts" / "check-redis-health.sh").write_text(BROKER_REDIS_SCRIPT)
    (f / "Dockerfiles" / "elasticsearch.Dockerfile").write_text("FROM es\n")
    (f / "Dockerfiles" / "elasticsearch-gpu.Dockerfile").write_text("FROM es-gpu\n")
    return f


@pytest.mark.unit
class TestValidatePath:
    def test_accepts_path_inside_base(self, modify_foundational, tmp_path):
        base = tmp_path / "foundational"
        base.mkdir()
        child = base / "file.txt"
        child.write_text("x")
        assert modify_foundational._validate_path(child, base) == child.resolve()

    def test_rejects_path_traversal(self, modify_foundational, tmp_path):
        base = tmp_path / "foundational"
        base.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("x")
        with pytest.raises(ValueError, match="Path traversal"):
            modify_foundational._validate_path(outside, base)


@pytest.mark.unit
class TestHardcodeEsConnectionVars:
    def test_rewrites_elasticsearch_vars(self, modify_foundational):
        content = (
            "ELASTICSEARCH_CONNECTION_RETRY_ATTEMPTS=5\n"
            "ELASTICSEARCH_CONNECTION_MAX_ATTEMPTS=99\n"
            "ELASTICSEARCH_URL=http://old\n"
            "curl $ELASTICSEARCH_URL ${ELASTICSEARCH_URL}\n"
        )
        out = modify_foundational.hardcode_es_connection_vars(content)
        assert "ES_CONNECTION_RETRY_ATTEMPTS=0" in out
        assert "ES_CONNECTION_MAX_ATTEMPTS=10" in out
        assert 'ES_URL="http://localhost:9200"' in out
        assert "ELASTICSEARCH_URL" not in out


@pytest.mark.unit
class TestModifyFoundationalYml:
    def test_applies_stock_es_and_topic_renames(self, modify_foundational, tmp_path):
        path = tmp_path / "mdx-foundational.yml"
        path.write_text(FOUNDATIONAL_YML)
        modify_foundational.modify_foundational_yml(path)
        content = path.read_text()
        assert "docker.elastic.co/elasticsearch/elasticsearch:9.3.0" in content
        assert "mdx-vlm-captions" in content
        assert "mdx-embed-filtered" not in content
        assert "BP_PROFILE" not in content
        assert "BOOTSTRAP_HOST" not in content
        assert "retries: 5" in content

    def test_skips_missing_file(self, modify_foundational, tmp_path, capsys):
        modify_foundational.modify_foundational_yml(tmp_path / "missing.yml")
        assert "SKIP" in capsys.readouterr().out


@pytest.mark.unit
class TestModifyLogstashConfigs:
    def test_modify_kafka_logstash_conf(self, modify_foundational, tmp_path):
        f = _foundational_tree(tmp_path)
        modify_foundational.modify_kafka_logstash_conf(f)
        content = (f / "elk" / "configs" / "mdx-kafka-logstash.conf").read_text()
        assert 'type => "mdx-vlm-captions"' in content
        assert "first_timestamp" in content
        assert "mdx-embed-filtered" not in content

    def test_kafka_logstash_skip_when_already_modified(self, modify_foundational, tmp_path, capsys):
        f = _foundational_tree(tmp_path)
        modify_foundational.modify_kafka_logstash_conf(f)
        modify_foundational.modify_kafka_logstash_conf(f)
        assert "already modified" in capsys.readouterr().out

    def test_kafka_logstash_skip_missing(self, modify_foundational, tmp_path, capsys):
        f = tmp_path / "foundational"
        f.mkdir()
        modify_foundational.modify_kafka_logstash_conf(f)
        assert "SKIP" in capsys.readouterr().out

    def test_modify_redis_logstash_conf(self, modify_foundational, tmp_path):
        f = _foundational_tree(tmp_path)
        modify_foundational.modify_redis_logstash_conf(f)
        content = (f / "elk" / "configs" / "mdx-redis-logstash.conf").read_text()
        assert "mdx-embed" in content
        assert "mdx-embed-filtered" not in content


@pytest.mark.unit
class TestModifyEsScripts:
    def test_modify_es_template_creation(self, modify_foundational, tmp_path):
        path = tmp_path / "elasticsearch-template-creation.sh"
        path.write_text(ES_TEMPLATE_SCRIPT)
        modify_foundational.modify_es_template_creation(path)
        content = path.read_text()
        assert "mdx_vlm_captions_template" in content
        assert "mdx_embed_template" in content
        assert "BP_PROFILE" not in content
        assert "ELASTICSEARCH_RTVI_CV_EMBEDDINGS_DIM" not in content

    def test_modify_es_ilm_policy_creation(self, modify_foundational, tmp_path):
        path = tmp_path / "elasticsearch-ilm-policy-creation.sh"
        path.write_text(ES_ILM_SCRIPT)
        modify_foundational.modify_es_ilm_policy_creation(path)
        content = path.read_text()
        assert "mdx-embed-ilm-policy" in content
        assert "mdx-vlm-captions-ilm-policy" in content
        assert "ELASTICSEARCH_ILM_MIN_AGE" not in content

    def test_modify_es_ingest_pipeline_creation(self, modify_foundational, tmp_path):
        path = tmp_path / "elasticsearch-ingest-pipeline-creation.sh"
        path.write_text(ES_INGEST_SCRIPT)
        modify_foundational.modify_es_ingest_pipeline_creation(path)
        assert "ES_URL" in path.read_text()

    def test_skip_missing_es_scripts(self, modify_foundational, tmp_path, capsys):
        missing = tmp_path / "nope.sh"
        modify_foundational.modify_es_template_creation(missing)
        modify_foundational.modify_es_ilm_policy_creation(missing)
        modify_foundational.modify_es_ingest_pipeline_creation(missing)
        assert capsys.readouterr().out.count("SKIP") == 3


@pytest.mark.unit
class TestModifyKafkaAndHealth:
    def test_modify_kafka_create_topics(self, modify_foundational, tmp_path):
        path = tmp_path / "create-kafka-topics.sh"
        path.write_text(KAFKA_TOPICS_SCRIPT)
        modify_foundational.modify_kafka_create_topics(path)
        content = path.read_text()
        assert "localhost:9092" in content
        assert "KAFKA_HOST=" not in content

    def test_modify_broker_health_kafka(self, modify_foundational, tmp_path):
        path = tmp_path / "check-kafka-health.sh"
        path.write_text(BROKER_KAFKA_SCRIPT)
        modify_foundational.modify_broker_health_kafka(path)
        content = path.read_text()
        assert "localhost:9092" in content
        assert "KAFKA_HOST=" not in content

    def test_modify_broker_health_redis(self, modify_foundational, tmp_path):
        path = tmp_path / "check-redis-health.sh"
        path.write_text(BROKER_REDIS_SCRIPT)
        modify_foundational.modify_broker_health_redis(path)
        content = path.read_text()
        assert "localhost 6379" in content
        assert "REDIS_HOST=" not in content


@pytest.mark.unit
class TestRemoveEsDockerfiles:
    def test_removes_existing_dockerfiles(self, modify_foundational, tmp_path, capsys):
        f = _foundational_tree(tmp_path)
        modify_foundational.remove_es_dockerfiles(f)
        assert not (f / "Dockerfiles" / "elasticsearch.Dockerfile").exists()
        assert "Removed custom ES Dockerfiles" in capsys.readouterr().out

    def test_no_dockerfiles_message(self, modify_foundational, tmp_path, capsys):
        f = tmp_path / "foundational"
        (f / "Dockerfiles").mkdir(parents=True)
        modify_foundational.remove_es_dockerfiles(f)
        assert "No custom ES Dockerfiles" in capsys.readouterr().out


@pytest.mark.unit
class TestModifyFoundationalMain:
    def test_main_success(self, modify_foundational, tmp_path):
        _foundational_tree(tmp_path)
        with mock.patch.object(sys, "argv", ["modify_foundational_for_sop.py", str(tmp_path)]):
            modify_foundational.main()

    def test_main_exits_when_foundational_missing(self, modify_foundational, tmp_path):
        with mock.patch.object(sys, "argv", ["modify_foundational_for_sop.py", str(tmp_path)]):
            with pytest.raises(SystemExit) as exc:
                modify_foundational.main()
            assert exc.value.code == 1
