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

# Deep Verification Script for VSS SOP Build.
#
# This is the SINGLE SOURCE OF TRUTH for build-time verification. The copy/modify
# scripts and the per-component references/scripts/*/verify.sh wrappers all delegate
# here instead of re-implementing grep checks, so a rule is defined exactly once.
#
# Usage:
#   verify_build.py [BASE_DIR] [--component {all,foundational,nim,agents,vios}]
#     BASE_DIR     Blueprint repo root (default: ".")
#     --component  Limit checks to one component (default: all)
import os
import sys
import re
import argparse

# Shared, single-source field-naming rules (also used by vss-sop-test).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import kibana_fields  # noqa: E402


def print_section(title):
    print("\n" + "=" * 50)
    print(f" {title} ")
    print("=" * 50)


def _read(path):
    """Read a file, returning None if it does not exist."""
    if not os.path.exists(path):
        return None
    with open(path, "r", errors="ignore") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Global / structural checks (run in 'all' mode)
# ---------------------------------------------------------------------------
def verify_structure(base_dir):
    print_section("SOP Blueprint Structure")
    paths_to_check = [
        "deployments/sop",
        "deployments/sop/vss-agent/configs",
        "deployments/sop/vss-agent/patches",
        "deployments/sop/vss-agent/templates",
        "deployments/ds/ds-sop",
        "deployments/ds/compose.yml",
        "deployments/agents/vss-agent/vss-agent-docker-compose.yml",
        "deployments/nim/compose.yml",
    ]

    all_ok = True
    for p in paths_to_check:
        full_path = os.path.join(base_dir, p)
        if os.path.exists(full_path):
            print(f"✅ FOUND: {p}")
        else:
            print(f"❌ MISSING: {p}")
            all_ok = False
    return all_ok


def verify_compose_include(base_dir):
    print_section("Compose Include Check")
    compose_path = os.path.join(base_dir, "deployments/compose.yml")
    content = _read(compose_path)
    if content is None:
        print(f"❌ {compose_path} does not exist!")
        return False

    if "./sop/compose.yml" in content or "sop/compose.yml" in content:
        print("✅ PASS: top-level compose.yml includes './sop/compose.yml'")
        return True
    print("❌ FAIL: top-level compose.yml is missing './sop/compose.yml' inclusion!")
    return False


def verify_profiles(base_dir):
    print_section("Profile Check")
    found_profiles = []
    deployments_dir = os.path.join(base_dir, "deployments")

    if not os.path.exists(deployments_dir):
        print("❌ deployments directory not found!")
        return False

    for root, _, files in os.walk(deployments_dir):
        for file in files:
            if file.endswith((".yml", ".yaml")):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", errors="ignore") as f:
                        for line_num, line in enumerate(f, 1):
                            if "bp_sop_2d" in line:
                                rel_path = os.path.relpath(file_path, base_dir)
                                found_profiles.append(f"{rel_path}:{line_num}")
                except Exception:
                    pass

    if found_profiles:
        print(f"✅ PASS: Found 'bp_sop_2d' profile usage in {len(found_profiles)} locations:")
        for loc in found_profiles[:10]:
            print(f"  - {loc}")
        if len(found_profiles) > 10:
            print(f"  - ... and {len(found_profiles) - 10} more.")
        return True
    print("❌ FAIL: No files in deployments/ reference the 'bp_sop_2d' profile!")
    return False


def check_container_versions(base_dir):
    print_section("Container Versions")
    sop_dir = os.path.join(base_dir, "deployments/sop")
    ds_dir = os.path.join(base_dir, "deployments/ds")
    images = set()

    for folder in [sop_dir, ds_dir]:
        if not os.path.exists(folder):
            continue
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith((".yml", ".yaml", ".env")):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", errors="ignore") as f:
                            for line in f:
                                if "image:" in line:
                                    img = line.split("image:")[-1].strip().strip('"').strip("'")
                                    images.add(img)
                    except Exception:
                        pass

    if images:
        print("SOP/DS image versions found in compose files:")
        for img in sorted(images):
            print(f"  - {img}")
    else:
        print("⚠️ No direct 'image:' keys found in SOP/DS compose files.")


# ---------------------------------------------------------------------------
# Foundational component checks
# ---------------------------------------------------------------------------
def verify_foundational_profiles(base_dir):
    print_section("Foundational: Profiles")
    fyml = os.path.join(base_dir, "deployments/foundational/mdx-foundational.yml")
    content = _read(fyml)
    if content is None:
        print(f"❌ FAIL: {fyml} not found (run modify_foundational_for_sop.sh first)")
        return False

    ok = True
    count = content.count("bp_sop_2d")
    if count > 0:
        print(f"✅ PASS: bp_sop_2d profile present ({count} occurrences)")
    else:
        print("❌ FAIL: bp_sop_2d profile missing from mdx-foundational.yml")
        ok = False

    if "MINIMAL_PROFILE" in content:
        print("❌ FAIL: MINIMAL_PROFILE references still present")
        ok = False
    else:
        print("✅ PASS: no MINIMAL_PROFILE references")
    return ok


def verify_foundational_es_image(base_dir):
    print_section("Foundational: Stock Elasticsearch Image")
    fyml = os.path.join(base_dir, "deployments/foundational/mdx-foundational.yml")
    content = _read(fyml)
    if content is None:
        print(f"❌ FAIL: {fyml} not found")
        return False

    if "docker.elastic.co/elasticsearch/elasticsearch:9.3.0" in content:
        print("✅ PASS: stock Elasticsearch image (elasticsearch:9.3.0) in use")
        es_image_ok = True
    elif re.search(r"image:.*elasticsearch", content):
        print("⚠️ WARNING: an elasticsearch image is set but not the expected 9.3.0 stock tag")
        es_image_ok = True
    else:
        print("❌ FAIL: no stock Elasticsearch image reference found")
        es_image_ok = False

    # Custom ES Dockerfiles must be removed.
    dockerfiles_ok = True
    for df in ("elasticsearch.Dockerfile", "elasticsearch-gpu.Dockerfile"):
        if os.path.exists(os.path.join(base_dir, "deployments/foundational/Dockerfiles", df)):
            print(f"❌ FAIL: custom ES Dockerfile still present: {df}")
            dockerfiles_ok = False
    if dockerfiles_ok:
        print("✅ PASS: custom Elasticsearch Dockerfiles removed")
    return es_image_ok and dockerfiles_ok


def verify_foundational_kafka_topic(base_dir):
    print_section("Foundational: Kafka Topics")
    fyml = os.path.join(base_dir, "deployments/foundational/mdx-foundational.yml")
    content = _read(fyml)
    if content is None:
        print(f"❌ FAIL: {fyml} not found")
        return False

    ok = True
    if "mdx-vlm-captions" in content:
        print("✅ PASS: mdx-vlm-captions topic present")
    else:
        print("❌ FAIL: mdx-vlm-captions topic missing")
        ok = False

    found_dir = os.path.join(base_dir, "deployments/foundational")
    embed_hit = False
    for root, _, files in os.walk(found_dir):
        for file in files:
            c = _read(os.path.join(root, file))
            if c and "embed-filtered" in c:
                embed_hit = True
                break
        if embed_hit:
            break
    if embed_hit:
        print("❌ FAIL: embed-filtered references still present under foundational/")
        ok = False
    else:
        print("✅ PASS: no embed-filtered references remain")
    return ok


def verify_foundational_logstash(base_dir):
    print_section("Foundational: Logstash Pipeline")
    lsconf = os.path.join(base_dir, "deployments/foundational/elk/configs/mdx-kafka-logstash.conf")
    content = _read(lsconf)
    if content is None:
        print(f"❌ FAIL: {lsconf} not found")
        return False

    ok = True

    if re.search(r'topics => \["mdx-vlm-captions"\]', content) and 'codec => "json"' in content:
        print("✅ PASS: mdx-vlm-captions Kafka input with json codec present")
    else:
        print("❌ FAIL: mdx-vlm-captions Kafka input/json codec missing (ES will report 0 indices)")
        ok = False

    if re.search(r"first_timestamp.*start_time", content):
        print("✅ PASS: timestamp wiring (first_timestamp/start_time) present")
    else:
        print("❌ FAIL: timestamp wiring missing (docs land in mdx-vlm-captions-1970-01-01)")
        ok = False

    # Final mutate must preserve @timestamp for mdx-vlm-captions.
    m = re.search(r'if \[type\] == "mdx-vlm-captions".{0,200}', content, re.DOTALL)
    if m and 'remove_field => ["kafka", "message", "@version"]' in m.group():
        print("✅ PASS: @timestamp preserved for mdx-vlm-captions")
    else:
        print("⚠️ WARNING: could not confirm @timestamp preservation mutate for mdx-vlm-captions")

    # mdx-vlm-captions must NOT be in the document_id output branch.
    output_section = content[content.rfind("output {"):]
    dm = re.search(r"if \[type\].*?document_id", output_section, re.DOTALL)
    if dm and "mdx-vlm-captions" in dm.group():
        print("❌ FAIL: mdx-vlm-captions in document_id branch — ES doc count will stay at 1")
        ok = False
    else:
        print("✅ PASS: mdx-vlm-captions excluded from document_id branch")
    return ok


def verify_kibana_dashboard(base_dir):
    print_section("Kibana Dashboard Field Validation")
    ndjson_path = os.path.join(
        base_dir, "deployments/sop/sop-app/kibana-dashboard/sop-kibana-objects.ndjson"
    )
    content = _read(ndjson_path)
    if content is None:
        print(f"❌ FAIL: Kibana dashboard file not found at {ndjson_path}")
        return False

    errors = kibana_fields.scan_ndjson_text(content)
    if not errors:
        print("✅ PASS: Kibana ndjson uses correct flat JSON field names and @timestamp timefield.")
        return True
    for err in errors:
        print(f"❌ FAIL: {err}")
    print(f"❌ FAIL: {len(errors)} Kibana field name validation errors detected.")
    return False


# ---------------------------------------------------------------------------
# NIM component checks
# ---------------------------------------------------------------------------
def verify_nim(base_dir):
    print_section("NIM: Cleanup & Rename")
    nim = os.path.join(base_dir, "deployments/nim")
    if not os.path.isdir(nim):
        print(f"❌ FAIL: {nim} not found (run copy_nim_from_upstream.sh first)")
        return False

    ok = True

    if os.path.exists(os.path.join(nim, "fallback-override.env")):
        print("❌ FAIL: fallback-override.env still present")
        ok = False
    else:
        print("✅ PASS: fallback-override.env removed")

    hw_other = []
    for root, _, files in os.walk(nim):
        hw_other += [f for f in files if f.startswith("hw-OTHER")]
    if hw_other:
        print(f"❌ FAIL: hw-OTHER* files still present: {hw_other[:5]}")
        ok = False
    else:
        print("✅ PASS: no hw-OTHER* files")

    if os.path.exists(os.path.join(nim, "nemotron-nano-v2/compose.yml")):
        print("✅ PASS: nemotron-nano-v2/compose.yml exists")
    else:
        print("❌ FAIL: nemotron-nano-v2/compose.yml missing")
        ok = False

    if os.path.isdir(os.path.join(nim, "nvidia-nemotron-nano-9b-v2-fp8")):
        print("❌ FAIL: fp8 directory still present")
        ok = False
    else:
        print("✅ PASS: fp8 directory removed")

    nim_compose = _read(os.path.join(nim, "compose.yml"))
    if nim_compose and "nemotron-nano-v2" in nim_compose and "fp8" not in nim_compose:
        print("✅ PASS: nim/compose.yml references nemotron-nano-v2 and drops fp8")
    else:
        print("❌ FAIL: nim/compose.yml not correctly updated for nemotron-nano-v2/fp8")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# VSS Agent component checks
# ---------------------------------------------------------------------------
def verify_agents(base_dir):
    print_section("VSS Agent: Profiles, Patches & Includes")
    agents = os.path.join(base_dir, "deployments/agents")
    vss_compose = _read(os.path.join(agents, "vss-agent/vss-agent-docker-compose.yml"))
    if vss_compose is None:
        print(f"❌ FAIL: vss-agent-docker-compose.yml not found under {agents}")
        return False

    ok = True

    if "bp_sop_2d" in vss_compose:
        print("✅ PASS: bp_sop_2d profile in vss-agent-docker-compose.yml")
    else:
        print("❌ FAIL: bp_sop_2d profile missing from vss-agent-docker-compose.yml")
        ok = False

    if "patches/tools.py" in vss_compose:
        print("✅ PASS: SOP patch volume mounts present in vss-va-mcp")
    else:
        print("❌ FAIL: SOP patch volume mounts (patches/tools.py) missing")
        ok = False

    if "agent-eval" in vss_compose:
        print("❌ FAIL: agent-eval references still present in vss-agent compose")
        ok = False
    else:
        print("✅ PASS: agent-eval volume removed")

    agent_ui = _read(os.path.join(agents, "agent_ui/compose.yml"))
    if agent_ui and "bp_sop_2d" in agent_ui:
        print("✅ PASS: bp_sop_2d profile in agent_ui/compose.yml")
    else:
        print("❌ FAIL: bp_sop_2d profile missing from agent_ui/compose.yml")
        ok = False

    agents_compose = _read(os.path.join(agents, "compose.yml"))
    if agents_compose and re.search(r"#.*ai-agents", agents_compose):
        print("✅ PASS: ai-agents include commented out in agents/compose.yml")
    elif agents_compose and "ai-agents" not in agents_compose:
        print("✅ PASS: ai-agents include absent in agents/compose.yml")
    else:
        print("⚠️ WARNING: could not confirm ai-agents include is commented out")
    return ok


# ---------------------------------------------------------------------------
# VIOS (VST) component checks
# ---------------------------------------------------------------------------
def verify_vios_structure(base_dir):
    print_section("VIOS: SOP Tree Structure")
    sop_vst = os.path.join(base_dir, "deployments/vst/sop/vst")
    if not os.path.isdir(sop_vst):
        print(f"⚠️ WARNING: {sop_vst} not found (VST split not copied yet). Skipping.")
        return True

    ok = True
    expected = [
        "compose.yml",  # top-level vst/compose.yml (one level up)
        "docker-compose.yaml",
        ".env",
        "configs/nginx.conf",
        "minio/minio-compose.yaml",
    ]
    # Top-level compose lives at deployments/vst/compose.yml
    if os.path.exists(os.path.join(base_dir, "deployments/vst/compose.yml")):
        print("✅ PASS: vst/compose.yml")
    else:
        print("❌ FAIL: vst/compose.yml missing")
        ok = False
    for rel in expected[1:]:
        if os.path.exists(os.path.join(sop_vst, rel)):
            print(f"✅ PASS: sop/vst/{rel}")
        else:
            print(f"❌ FAIL: sop/vst/{rel} missing")
            ok = False

    for module in ("rtspserver", "recorder", "replaystream", "livestream"):
        for rel in ("sdr-compose.yaml", "envoy.yaml", "sdr-config/docker_cluster_config.json"):
            p = os.path.join(sop_vst, f"sdr-{module}-http", rel)
            if os.path.exists(p):
                print(f"✅ PASS: sdr-{module}-http/{rel}")
            else:
                print(f"❌ FAIL: sdr-{module}-http/{rel} missing")
                ok = False

    # No upstream leftovers.
    for leftover in ("developer", "scripts", "sop/vst/sdr-streamprocessing"):
        if os.path.isdir(os.path.join(base_dir, "deployments/vst", leftover)):
            print(f"❌ FAIL: upstream leftover present: vst/{leftover}")
            ok = False
    return ok


def verify_sdr_recorder(base_dir):
    print_section("SDR Recorder API URL Validation")
    sdr_path = os.path.join(base_dir, "deployments/vst/sop/vst/sdr-recorder-http/sdr-compose.yaml")
    content = _read(sdr_path)
    if content is None:
        print(f"⚠️ WARNING: SDR recorder compose file not found at {sdr_path}. Skipping VST-SDR check.")
        return True

    errors = 0
    if "WDM_WL_ADD_URL" in content and "proxy" in content:
        print("❌ FAIL: SDR recorder WDM_WL_ADD_URL still uses /api/v1/proxy/ (should be /api/v1/record/)")
        errors += 1
    if "WDM_WL_CHANGE_ID_ADD" in content and "camera_proxy" in content:
        print("❌ FAIL: SDR recorder WDM_WL_CHANGE_ID_ADD is camera_proxy (should be camera_streaming)")
        errors += 1
    if "STORAGE_MODULE_ENDPOINT" not in content:
        print("❌ FAIL: recorder-ms missing STORAGE_MODULE_ENDPOINT env var")
        errors += 1

    if errors == 0:
        print("✅ PASS: SDR recorder uses correct recorder API endpoints.")
        return True
    print(f"❌ FAIL: {errors} SDR recorder configuration errors detected.")
    return False


def verify_sdr_double_quotes(base_dir):
    print_section("SDR Cluster Container Names Double-Quotes Check")
    sdr_modules = ["rtspserver", "recorder", "replaystream", "livestream"]
    all_ok = True

    for mod in sdr_modules:
        sdr_path = os.path.join(base_dir, f"deployments/vst/sop/vst/sdr-{mod}-http/sdr-compose.yaml")
        content = _read(sdr_path)
        if content is None:
            print(f"⚠️ WARNING: SDR {mod} compose file not found at {sdr_path}. Skipping check.")
            continue

        found_env = False
        for line in content.splitlines():
            if "WDM_CLUSTER_CONTAINER_NAMES" in line:
                found_env = True
                if '""' in line:
                    print(f"❌ FAIL: SDR {mod} WDM_CLUSTER_CONTAINER_NAMES has invalid double-double-quotes: {line.strip()}")
                    all_ok = False
                else:
                    print(f"✅ PASS: SDR {mod} WDM_CLUSTER_CONTAINER_NAMES is valid: {line.strip()}")
        if not found_env:
            print(f"❌ FAIL: SDR {mod} is missing WDM_CLUSTER_CONTAINER_NAMES env var!")
            all_ok = False
    return all_ok


def verify_sdr_http_port_naming(base_dir):
    print_section("SDR RTSP Server HTTP Port Variable Naming Check")
    sdr_path = os.path.join(base_dir, "deployments/vst/sop/vst/sdr-rtspserver-http/sdr-compose.yaml")
    content = _read(sdr_path)
    if content is None:
        print(f"⚠️ WARNING: SDR rtspserver compose file not found at {sdr_path}. Skipping check.")
        return True

    found_port = False
    all_ok = True
    for line in content.splitlines():
        if "HTTP_PORT=" in line or "HTTP_PORT:" in line:
            found_port = True
            if "RTSPSERVER_HTTP_PORT_1" in line:
                print(f"❌ FAIL: RTSP server HTTP_PORT has buggy RTSPSERVER_HTTP_PORT_1 instead of RTSP_SERVER_HTTP_PORT_1: {line.strip()}")
                all_ok = False
            elif "RTSP_SERVER_HTTP_PORT_1" in line:
                print(f"✅ PASS: RTSP server HTTP_PORT is using RTSP_SERVER_HTTP_PORT_1 correctly: {line.strip()}")
            else:
                print(f"⚠️ WARNING: RTSP server HTTP_PORT is using an unexpected variable: {line.strip()}")
    if not found_port:
        print("❌ FAIL: RTSP server compose is missing HTTP_PORT env var!")
        all_ok = False
    return all_ok


def verify_nginx_routing(base_dir):
    print_section("Nginx VST Ingress Routing Check")
    nginx_path = os.path.join(base_dir, "deployments/vst/sop/vst/configs/nginx.conf")
    content = _read(nginx_path)
    if content is None:
        print(f"⚠️ WARNING: Nginx configuration file not found at {nginx_path}. Skipping check.")
        return True

    errors = 0
    if "location = /" not in content or "return 301 /vst/;" not in content:
        print("❌ FAIL: nginx.conf is missing root '/' 301 redirect to '/vst/'")
        errors += 1
    else:
        print("✅ PASS: nginx.conf has root '/' 301 redirect to '/vst/'")

    has_vst_location = "location /vst/" in content
    has_sensor_proxy = (
        "proxy_pass http://sensor-ms/;" in content
        or "proxy_pass http://localhost:30000/;" in content
    )
    if not has_vst_location or not has_sensor_proxy:
        print("❌ FAIL: nginx.conf is missing general '/vst/' proxy to 'sensor-ms'")
        errors += 1
    else:
        print("✅ PASS: nginx.conf has general '/vst/' proxy to 'sensor-ms'")

    if errors == 0:
        return True
    print(f"❌ FAIL: {errors} Nginx ingress routing configuration errors detected.")
    return False


# ---------------------------------------------------------------------------
# Component registry + dispatch
# ---------------------------------------------------------------------------
COMPONENT_CHECKS = {
    "foundational": [
        verify_foundational_profiles,
        verify_foundational_es_image,
        verify_foundational_kafka_topic,
        verify_foundational_logstash,
        verify_kibana_dashboard,
    ],
    "nim": [
        verify_nim,
    ],
    "agents": [
        verify_agents,
    ],
    "vios": [
        verify_vios_structure,
        verify_sdr_recorder,
        verify_sdr_double_quotes,
        verify_sdr_http_port_naming,
        verify_nginx_routing,
    ],
}


def main():
    parser = argparse.ArgumentParser(description="Verify a VSS SOP build.")
    parser.add_argument("base_dir", nargs="?", default=".", help="Blueprint repo root (default: .)")
    parser.add_argument(
        "--component",
        choices=["all"] + list(COMPONENT_CHECKS.keys()),
        default="all",
        help="Limit checks to one component (default: all).",
    )
    args = parser.parse_args()
    base_dir = args.base_dir

    print(f"Verifying VSS SOP Build in: {os.path.abspath(base_dir)} (component: {args.component})")

    results = []
    if args.component == "all":
        results += [
            verify_structure(base_dir),
            verify_compose_include(base_dir),
            verify_profiles(base_dir),
        ]
        for component in COMPONENT_CHECKS:
            for check in COMPONENT_CHECKS[component]:
                results.append(check(base_dir))
        check_container_versions(base_dir)
    else:
        for check in COMPONENT_CHECKS[args.component]:
            results.append(check(base_dir))

    print_section("Verification Summary")
    if all(results):
        print("🎉 ALL VERIFICATION CHECKS PASSED!")
        sys.exit(0)
    else:
        print("❌ SOME VERIFICATION CHECKS FAILED! Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

