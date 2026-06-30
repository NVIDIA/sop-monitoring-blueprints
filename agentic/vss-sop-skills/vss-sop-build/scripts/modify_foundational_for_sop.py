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

"""Modify the copied foundational folder to work with SOP profile.

Applies all SOP-specific modifications to foundational services:
- mdx-foundational.yml: stock ES image, remove env blocks, rename Kafka topics, tune health checks
- Logstash configs: rename embed-filtered, add mdx-vlm-captions input + filter
- ES init scripts: hardcode vars, remove BP_PROFILE branches, add vlm-captions template
- Kafka/health-check scripts: hardcode connection vars

Prerequisites: copy_foundational_from_upstream.sh must have been run first.

This is the single source of truth for foundational modifications: it patches
profiles, rewrites configs, and removes the custom ES Dockerfiles. The
modify_foundational_for_sop.sh wrapper only orchestrates this script and
verify_build.py.
"""
import re
import sys
from pathlib import Path

# patch_profiles lives alongside this script and is reused across components.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import patch_profiles  # noqa: E402


def _validate_path(file_path: Path, base_dir: Path) -> Path:
    """Resolve *file_path* and ensure it stays within *base_dir*.

    Raises ValueError if the resolved path escapes the base directory
    (e.g. via '..' traversal in user-supplied input).
    """
    resolved = file_path.resolve()
    base_resolved = base_dir.resolve()
    if not resolved.is_relative_to(base_resolved):
        raise ValueError(
            f"Path traversal detected: {file_path} resolves to "
            f"{resolved}, which is outside {base_resolved}"
        )
    return resolved


def main():
    bp_repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    foundational = bp_repo / "deployments" / "foundational"

    if not foundational.exists():
        print(f"Error: {foundational} does not exist.")
        sys.exit(1)

    # Validate all target paths upfront to guard against symlink traversal.
    target_paths = [
        foundational / "mdx-foundational.yml",
        foundational / "elk" / "configs" / "mdx-kafka-logstash.conf",
        foundational / "elk" / "configs" / "mdx-redis-logstash.conf",
        foundational / "elk" / "init-scripts" / "elasticsearch-template-creation.sh",
        foundational / "elk" / "init-scripts" / "elasticsearch-ilm-policy-creation.sh",
        foundational / "elk" / "init-scripts" / "elasticsearch-ingest-pipeline-creation.sh",
        foundational / "kafka" / "init-scripts" / "create-kafka-topics.sh",
        foundational / "broker-health-check" / "scripts" / "check-kafka-health.sh",
        foundational / "broker-health-check" / "scripts" / "check-redis-health.sh",
    ]
    for p in target_paths:
        if p.exists():
            _validate_path(p, foundational)

    # Step 1: add bp_sop_2d profile + strip MINIMAL_PROFILE suffixes.
    patch_profiles.patch_file_profiles(str(foundational / "mdx-foundational.yml"))

    modify_foundational_yml(foundational / "mdx-foundational.yml")
    modify_kafka_logstash_conf(foundational)
    modify_redis_logstash_conf(foundational)
    modify_es_template_creation(foundational / "elk" / "init-scripts" / "elasticsearch-template-creation.sh")
    modify_es_ilm_policy_creation(foundational / "elk" / "init-scripts" / "elasticsearch-ilm-policy-creation.sh")
    modify_es_ingest_pipeline_creation(foundational / "elk" / "init-scripts" / "elasticsearch-ingest-pipeline-creation.sh")
    modify_kafka_create_topics(foundational / "kafka" / "init-scripts" / "create-kafka-topics.sh")
    modify_broker_health_kafka(foundational / "broker-health-check" / "scripts" / "check-kafka-health.sh")
    modify_broker_health_redis(foundational / "broker-health-check" / "scripts" / "check-redis-health.sh")
    remove_es_dockerfiles(foundational)

    print("  All foundational SOP modifications applied.")


def remove_es_dockerfiles(foundational: Path):
    """Remove the custom Elasticsearch Dockerfiles (SOP uses the stock ES image)."""
    removed = []
    for name in ("elasticsearch.Dockerfile", "elasticsearch-gpu.Dockerfile"):
        df = foundational / "Dockerfiles" / name
        if df.exists():
            df.unlink()
            removed.append(name)
    if removed:
        print(f"  Removed custom ES Dockerfiles: {', '.join(removed)}")
    else:
        print("  No custom ES Dockerfiles to remove")


def hardcode_es_connection_vars(content: str) -> str:
    """Hardcode ES connection env vars (ELASTICSEARCH_* -> ES_*) and rewrite references.

    Shared by the three ES init-script modifiers (template, ILM policy, ingest pipeline),
    which all need the same connection-var rewrite before applying their own changes.
    """
    content = re.sub(
        r'ELASTICSEARCH_CONNECTION_RETRY_ATTEMPTS=.*\n',
        'ES_CONNECTION_RETRY_ATTEMPTS=0\n',
        content,
    )
    content = re.sub(
        r'ELASTICSEARCH_CONNECTION_MAX_ATTEMPTS=.*\n',
        'ES_CONNECTION_MAX_ATTEMPTS=10\n',
        content,
    )
    content = re.sub(
        r'ELASTICSEARCH_URL=.*\n',
        'ES_URL="http://localhost:9200"\n',
        content,
    )

    content = content.replace('$ELASTICSEARCH_URL', '$ES_URL')
    content = content.replace('${ELASTICSEARCH_URL}', '${ES_URL}')
    content = content.replace('${ELASTICSEARCH_CONNECTION_RETRY_ATTEMPTS}', '${ES_CONNECTION_RETRY_ATTEMPTS}')
    content = content.replace('${ELASTICSEARCH_CONNECTION_MAX_ATTEMPTS}', '${ES_CONNECTION_MAX_ATTEMPTS}')
    content = content.replace('$ELASTICSEARCH_CONNECTION_RETRY_ATTEMPTS', '$ES_CONNECTION_RETRY_ATTEMPTS')
    content = content.replace('$ELASTICSEARCH_CONNECTION_MAX_ATTEMPTS', '$ES_CONNECTION_MAX_ATTEMPTS')
    return content


def modify_foundational_yml(path: Path):
    """Step 1c-1f: stock ES image, remove env blocks, rename topics, tune health checks."""
    if not path.exists():
        print(f"  SKIP {path.name} (not found)")
        return

    content = path.read_text()

    # 1c: Replace ES build block with stock image
    content = re.sub(
        r'(  elasticsearch:\n)    build:\n.*?dockerfile: Dockerfiles/elasticsearch\.Dockerfile\n.*?network: "host"\n    image: elasticsearch\n',
        r'\1    image: docker.elastic.co/elasticsearch/elasticsearch:9.3.0\n',
        content,
        flags=re.DOTALL,
    )

    # 1d: Remove BP_PROFILE env var from elasticsearch-init-container (keep environment: key
    # and other vars like ELASTICSEARCH_CONNECTION_MAX_ATTEMPTS, ELASTICSEARCH_ILM_MIN_AGE, etc.)
    content = re.sub(
        r'(    container_name: mdx-elasticsearch-init\n)    environment:\n(      - BP_PROFILE.*?\n)+',
        r'\1    environment:\n',
        content,
    )

    # 1d: Remove broker-health-check environment vars (BOOTSTRAP_HOST, KAFKA_PORT, REDIS_PORT)
    content = re.sub(r'      BOOTSTRAP_HOST: localhost\n', '', content)
    content = re.sub(r'      KAFKA_PORT: 9092\n', '', content)
    content = re.sub(r'      REDIS_PORT: 6379\n', '', content)

    # 1d: Remove kibana environment block
    content = re.sub(
        r'(    container_name: mdx-kibana\n.*?)(    environment:\n      SERVER_PUBLICBASEURL:.*?\n      SERVER_SECURITYRESPONSEHEADERS_DISABLEEMBEDDING:.*?\n      CSP_STRICT:.*?\n)',
        r'\1',
        content,
        flags=re.DOTALL,
    )

    # 1e: Rename Kafka topics in kafka-topic-init-container
    content = content.replace('{"name": "mdx-vlm"},\n        ', '')
    content = content.replace('{"name": "mdx-embed-filtered"}', '{"name": "mdx-vlm-captions"}')

    # 1e: Rename in broker-health-check KAFKA_TOPICS
    content = content.replace('{"name": "mdx-vlm"},\n        ', '')

    # 1f: Tune ES health check retries
    content = re.sub(
        r'(    healthcheck:\n      test: \["CMD", "curl", "-f", "http://localhost:9200/_cluster/health"\]\n      interval: 10s\n      timeout: 10s\n      retries: )15',
        r'\g<1>5',
        content,
    )

    # 1f: Tune kibana health check
    content = re.sub(r'(      retries: )30(\n      start_period: )60s', r'\g<1>5\g<2>30s', content)

    # 1f: Remove network: "host" from build blocks of elasticsearch-init-container and broker-health-check
    # (but not from the elasticsearch service itself - that's already removed by stock image replacement)
    lines = content.split('\n')
    new_lines = []
    in_build_block = False
    build_indent = 0
    skip_network_in_build = False

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if stripped.startswith('build:') and indent >= 4:
            in_build_block = True
            build_indent = indent
            new_lines.append(line)
            continue

        if in_build_block:
            if stripped and indent <= build_indent and not stripped.startswith('#'):
                in_build_block = False
            elif stripped.startswith('network:') and 'host' in stripped:
                continue

        new_lines.append(line)

    content = '\n'.join(new_lines)

    path.write_text(content)
    print(f"  Modified {path.name}")


def modify_kafka_logstash_conf(foundational: Path):
    """Step 2a: rename embed type, add vlm-captions input, add filter branch."""
    rel = Path("elk") / "configs" / "mdx-kafka-logstash.conf"
    path = foundational / rel
    if not path.exists():
        print(f"  SKIP {path.name} (not found)")
        return
    path = path.resolve()
    if not path.is_relative_to(foundational.resolve()):
        raise ValueError(f"Refusing to modify path outside foundational dir: {path}")

    content = path.read_text()

    # Already modified check
    if 'type => "mdx-vlm-captions"' in content and 'first_timestamp' in content:
        print(f"  SKIP {path.name} (already modified)")
        return

    # 2a(1): Rename ALL mdx-embed-filtered → mdx-embed (type, topics, filter/output refs)
    content = content.replace('mdx-embed-filtered', 'mdx-embed')

    # 2a(2): Add mdx-vlm-captions Kafka input before the closing }
    vlm_input = """\tkafka {
\t\ttype => "mdx-vlm-captions"
\t\tconsumer_threads => 4
\t\ttopics => ["mdx-vlm-captions"]
\t\tauto_offset_reset => "earliest"
\t\tdecorate_events => true
\t\tgroup_id => "logstash"
\t\tkey_deserializer_class => "org.apache.kafka.common.serialization.StringDeserializer"
\t\tvalue_deserializer_class => "org.apache.kafka.common.serialization.StringDeserializer"
\t\tcodec => "json"
\t\tbootstrap_servers => "localhost:9092"
\t}"""

    # Insert before the closing } of input block
    content = content.replace(
        '\t}\n}\nfilter {',
        f'\t}}\n{vlm_input}\n}}\nfilter {{',
        1,
    )

    # 2a(3): Wrap protobuf timestamp in else block, add vlm-captions branch
    old_timestamp_block = """\tjson { source => "message" }
\t# Formatting timestamp
\truby {
\t\tcode => "event.set('timestamp',(((event.get('[timestamp][seconds]').to_f)*1000) +((event.get('[timestamp][nanos]').to_f) * (10 ** -6)).floor()))"
\t}
\tdate {
\t\tmatch => [ "timestamp","UNIX_MS" ]
\t\ttarget => "timestamp"
\t\ttimezone => "UTC"
\t}"""

    new_timestamp_block = """\tjson { source => "message" }
\tif [type] == "mdx-vlm-captions" {
\t\truby {
\t\t\t# Use the wall-clock pipeline timestamps (epoch seconds) for @timestamp.
\t\t\t# first_timestamp + start_time are RELATIVE stream seconds, which map to a
\t\t\t# ~1970 epoch and make the Kibana dashboard (recent-time filter) show no
\t\t\t# records. pipeline_chunk_end_timestamp / pipeline_vlm_ready_timestamp are
\t\t\t# real epoch seconds; fall back to ingest time if neither is present.
\t\t\tcode => "pe = event.get('pipeline_chunk_end_timestamp'); pv = event.get('pipeline_vlm_ready_timestamp'); if !pe.nil? && pe.to_f > 1000000000 then event.set('timestamp', (pe.to_f * 1000).to_i) elsif !pv.nil? && pv.to_f > 1000000000 then event.set('timestamp', (pv.to_f * 1000).to_i) else event.set('timestamp', (Time.now.to_f * 1000).to_i) end"
\t\t}
\t\tdate {
\t\t\tmatch => [ "timestamp","UNIX_MS" ]
\t\t\ttarget => "@timestamp"
\t\t\ttimezone => "UTC"
\t\t}
\t\tdate {
\t\t\tmatch => [ "timestamp","UNIX_MS" ]
\t\t\ttarget => "timestamp"
\t\t\ttimezone => "UTC"
\t\t}
\t} else {
\t# Formatting timestamp
\truby {
\t\tcode => "event.set('timestamp',(((event.get('[timestamp][seconds]').to_f)*1000) +((event.get('[timestamp][nanos]').to_f) * (10 ** -6)).floor()))"
\t}
\tdate {
\t\tmatch => [ "timestamp","UNIX_MS" ]
\t\ttarget => "timestamp"
\t\ttimezone => "UTC"
\t}
\t}"""

    content = content.replace(old_timestamp_block, new_timestamp_block)

    # Replace final grok + mutate with conditional version
    old_grok_mutate = """\tgrok {
\t\tmatch => ["timestamp", "%{YEAR:[@metadata][year]}-%{MONTHNUM:[@metadata][month]}-%{MONTHDAY:[@metadata][day]}T%{GREEDYDATA}"]
\t}
\tmutate {
\t\tremove_field => ["kafka", "message", "@timestamp", "@version"]
\t}"""

    new_grok_mutate = """\tif [type] == "mdx-vlm-captions" {
\t\truby { code => "t=event.get('@timestamp').to_s; event.set('[@metadata][year]',t[0,4]); event.set('[@metadata][month]',t[5,2]); event.set('[@metadata][day]',t[8,2])" }
\t\tmutate { remove_field => ["kafka", "message", "@version"] }
\t} else {
\tgrok {
\t\tmatch => ["timestamp", "%{YEAR:[@metadata][year]}-%{MONTHNUM:[@metadata][month]}-%{MONTHDAY:[@metadata][day]}T%{GREEDYDATA}"]
\t}
\t\tmutate { remove_field => ["kafka", "message", "@timestamp", "@version"] }
\t}"""

    content = content.replace(old_grok_mutate, new_grok_mutate)

    path.write_text(content)
    print(f"  Modified {path.name}")


def modify_redis_logstash_conf(foundational: Path):
    """Step 2b: rename embed-filtered → embed globally."""
    rel = Path("elk") / "configs" / "mdx-redis-logstash.conf"
    path = foundational / rel
    if not path.exists():
        print(f"  SKIP {path.name} (not found)")
        return
    path = path.resolve()
    if not path.is_relative_to(foundational.resolve()):
        raise ValueError(f"Refusing to modify path outside foundational dir: {path}")

    content = path.read_text()
    content = content.replace('mdx-embed-filtered', 'mdx-embed')
    path.write_text(content)
    print(f"  Modified {path.name}")


def modify_es_template_creation(path: Path):
    """Step 3c: hardcode vars, remove BP_PROFILE branches, add vlm-captions template."""
    if not path.exists():
        print(f"  SKIP {path.name} (not found)")
        return

    content = path.read_text()

    content = hardcode_es_connection_vars(content)

    # Remove BP_PROFILE and embedding dimension variables
    content = re.sub(r'BP_PROFILE=.*\n', '', content)
    content = re.sub(r'echo "BP_PROFILE:.*\n', '', content)
    content = re.sub(r'.*ELASTICSEARCH_RTVI_CV_EMBEDDINGS_DIM.*\n', '', content)
    content = re.sub(r'.*ELASTICSEARCH_VISION_LLM_EMBEDDINGS_DIM.*\n', '', content)

    # Remove BP_PROFILE conditional blocks for behavior template (keep the non-search version)
    content = re.sub(
        r'    if \[\[ "\$\{BP_PROFILE:-\}" == "bp_developer_search" \]\]; then\n.*?echo "Successfully created.*?"\n    else\n',
        '',
        content,
        flags=re.DOTALL,
        count=1,
    )
    # Remove the closing fi for behavior
    content = re.sub(r'    fi\n\n    create_index_template "mdx_events_template"', '    create_index_template "mdx_events_template"', content)

    # Remove BP_PROFILE conditional blocks for raw template (keep the non-search version)
    content = re.sub(
        r'    if \[\[ "\$\{BP_PROFILE:-\}" == "bp_developer_search" \]\]; then\n.*?echo "Successfully created.*?"\n    else\n',
        '',
        content,
        flags=re.DOTALL,
        count=1,
    )
    # Remove the closing fi for raw
    content = re.sub(r'    fi\n\n    create_index_template "mdx_incidents_template"', '    create_index_template "mdx_incidents_template"', content)

    # Rename mdx_embed_filtered_template → mdx_embed_template
    content = content.replace('mdx_embed_filtered_template', 'mdx_embed_template')
    content = content.replace('"mdx-embed-filtered-*"', '"mdx-embed-*"')
    content = content.replace('mdx-embed-filtered-ilm-policy', 'mdx-embed-ilm-policy')

    # Hardcode embedding dims
    content = content.replace(
        """'"${ELASTICSEARCH_RTVI_CV_EMBEDDINGS_DIM}"'""",
        '768',
    )
    content = content.replace(
        """'"${ELASTICSEARCH_VISION_LLM_EMBEDDINGS_DIM}"'""",
        '768',
    )

    # Add mdx_vlm_captions_template before the final echo
    vlm_template = '''
    create_index_template "mdx_vlm_captions_template" '{
        "index_patterns": ["mdx-vlm-captions-*"],
        "priority": 516,
        "template": {
          "settings": { "index.lifecycle.name": "mdx-vlm-captions-ilm-policy" },
          "mappings": {
            "dynamic": true,
            "properties": {
              "cv_execute_time":  { "type": "double" },
              "vlm_execute_time": { "type": "double" },
              "frame_number":     { "type": "integer" },
              "chunk_idx":        { "type": "integer" }
            }
          }
        }
      }'

'''
    content = content.replace(
        '    echo "Successfully created index templates."',
        f'{vlm_template}    echo "Successfully created index templates."',
    )

    path.write_text(content)
    print(f"  Modified {path.name}")


def modify_es_ilm_policy_creation(path: Path):
    """Step 3a: hardcode vars, rename embed-filtered, add vlm-captions policy."""
    if not path.exists():
        print(f"  SKIP {path.name} (not found)")
        return

    content = path.read_text()

    content = hardcode_es_connection_vars(content)

    # ILM-specific: drop ELASTICSEARCH_ILM_MIN_AGE and inline its default
    content = re.sub(
        r'ELASTICSEARCH_ILM_MIN_AGE=.*\n',
        '',
        content,
    )
    content = content.replace('${ELASTICSEARCH_ILM_MIN_AGE:-4h}', '4h')

    # Rename embed-filtered
    content = content.replace('mdx-embed-filtered-ilm-policy', 'mdx-embed-ilm-policy')

    # Add vlm-captions ILM policy (insert after embed policy creation)
    if 'mdx-vlm-captions-ilm-policy' not in content:
        content = content.replace(
            'mdx-embed-ilm-policy',
            'mdx-embed-ilm-policy',
        )
        # Find a good insertion point - after the last create_ilm_policy call
        # Add a vlm-captions ILM policy call
        vlm_policy_line = '    create_ilm_policy "mdx-vlm-captions-ilm-policy"\n'
        embed_policy_match = re.search(r'(    create_ilm_policy "mdx-embed-ilm-policy".*\n)', content)
        if embed_policy_match:
            content = content.replace(
                embed_policy_match.group(0),
                embed_policy_match.group(0) + vlm_policy_line,
            )

    path.write_text(content)
    print(f"  Modified {path.name}")


def modify_es_ingest_pipeline_creation(path: Path):
    """Step 3b: hardcode connection vars."""
    if not path.exists():
        print(f"  SKIP {path.name} (not found)")
        return

    content = path.read_text()

    content = hardcode_es_connection_vars(content)

    path.write_text(content)
    print(f"  Modified {path.name}")


def modify_kafka_create_topics(path: Path):
    """Step 4: remove parameterized KAFKA_HOST/KAFKA_PORT."""
    if not path.exists():
        print(f"  SKIP {path.name} (not found)")
        return

    content = path.read_text()
    content = re.sub(r'KAFKA_HOST=\$\{BOOTSTRAP_HOST:-localhost\}\n', '', content)
    content = re.sub(r'KAFKA_PORT=\$\{KAFKA_PORT:-9092\}\n', '', content)
    content = content.replace('$KAFKA_HOST:$KAFKA_PORT', 'localhost:9092')
    content = content.replace('${KAFKA_HOST}:${KAFKA_PORT}', 'localhost:9092')
    path.write_text(content)
    print(f"  Modified {path.name}")


def modify_broker_health_kafka(path: Path):
    """Step 4: hardcode localhost:9092 in kafka health check."""
    if not path.exists():
        print(f"  SKIP {path.name} (not found)")
        return

    content = path.read_text()
    content = re.sub(r'KAFKA_HOST=.*\n', '', content)
    content = re.sub(r'KAFKA_PORT=.*\n', '', content)
    content = re.sub(r'echo.*KAFKA_HOST.*\n', '', content)
    content = re.sub(r'echo.*KAFKA_PORT.*\n', '', content)
    content = content.replace('$KAFKA_HOST:$KAFKA_PORT', 'localhost:9092')
    content = content.replace('${KAFKA_HOST}:${KAFKA_PORT}', 'localhost:9092')
    path.write_text(content)
    print(f"  Modified {path.name}")


def modify_broker_health_redis(path: Path):
    """Step 4: hardcode localhost 6379 in redis health check."""
    if not path.exists():
        print(f"  SKIP {path.name} (not found)")
        return

    content = path.read_text()
    content = re.sub(r'REDIS_HOST=.*\n', '', content)
    content = re.sub(r'REDIS_PORT=.*\n', '', content)
    content = re.sub(r'echo.*REDIS_HOST.*\n', '', content)
    content = re.sub(r'echo.*REDIS_PORT.*\n', '', content)
    content = content.replace('$REDIS_HOST $REDIS_PORT', 'localhost 6379')
    content = content.replace('${REDIS_HOST} ${REDIS_PORT}', 'localhost 6379')
    path.write_text(content)
    print(f"  Modified {path.name}")


if __name__ == "__main__":
    main()

