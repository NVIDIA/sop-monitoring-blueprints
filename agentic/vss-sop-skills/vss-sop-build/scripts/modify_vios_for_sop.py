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

"""Modify the copied VIOS (VST) folder to work with SOP profile.

Transforms the upstream VST structure (developer layout) into the SOP layout:
- Rewrites docker-compose.yaml for SOP services
- Rewrites .env with SOP-specific vars
- Modifies JSON config files
- Creates static nginx.conf
- Creates per-module SDR compose + envoy files
- Creates minio-compose.yaml

Prerequisites: copy_vios_from_upstream.sh must have been run first.

This script orchestrates all VIOS modifications: it copies the reference SDR
cluster configs, copies the SOP compose files from references/configs/vios/
(docker-compose.yaml, minio-compose.yaml, top-level vst/compose.yml), rewrites
the .env/config files, creates the SDR + minio modules, and removes the upstream
leftovers. The modify_vios_for_sop.sh wrapper only orchestrates this script and
verify_build.py.

The compose YAML lives in references/configs/vios/ (single source of truth),
not inline here, so reference docs and generated files cannot drift.
"""
import json
import shutil
import sys
from pathlib import Path

def main():
    bp_repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    vst = bp_repo / "deployments" / "vst"
    sop_vst = vst / "sop" / "vst"
    configs_dir = sop_vst / "configs"
    # References live alongside this script's skill dir
    # (.../vss-sop-build/references). Resolve from __file__ so the path is
    # correct regardless of how/where the skill is installed.
    refs = Path(__file__).resolve().parent.parent / "references"
    if not refs.exists():
        # Legacy install layout fallback.
        refs = bp_repo / ".claude" / "skills" / "vss-sop-build" / "references"

    if not sop_vst.exists():
        print(f"Error: {sop_vst} does not exist. Run copy_vios_from_upstream.sh first.")
        sys.exit(1)

    print("Applying SOP modifications...")

    # Step 1: Copy reference SDR docker_cluster_config.json files
    copy_sdr_cluster_configs(refs, sop_vst)

    # Step 2: Rewrite .env
    write_sop_env(sop_vst)

    # Step 3: Rewrite docker-compose.yaml
    write_sop_docker_compose(refs, sop_vst)

    # Step 4: Create static nginx.conf
    write_nginx_conf(configs_dir)

    # Step 5: Modify JSON config files
    modify_adaptor_config(configs_dir)
    modify_rtsp_streams(configs_dir)
    modify_vst_configs(configs_dir)

    # Step 6: Create SDR module directories
    write_sdr_modules(sop_vst)

    # Step 7: Create minio-compose.yaml
    write_minio_compose(refs, sop_vst)

    # Step 8: Write top-level vst/compose.yml and remove upstream leftovers
    write_top_level_compose(refs, vst)
    remove_upstream_leftovers(vst, sop_vst)

    print("All SOP modifications applied successfully.")


def copy_sdr_cluster_configs(refs: Path, sop_vst: Path):
    """Copy the reference SDR docker_cluster_config.json files into each SDR module."""
    for module in ("rtspserver", "recorder", "replaystream", "livestream"):
        src = (refs / "deployments" / "vst" / "sop" / "vst"
               / f"sdr-{module}-http" / "sdr-config" / "docker_cluster_config.json")
        dest = sop_vst / f"sdr-{module}-http" / "sdr-config" / "docker_cluster_config.json"
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest)
            print(f"  Copied: sdr-{module}-http/sdr-config/docker_cluster_config.json")
        else:
            print(f"  WARNING: Reference not found: {src}")


def write_top_level_compose(refs: Path, vst: Path):
    """Write the top-level vst/compose.yml from the reference config."""
    src = refs / "configs" / "vios" / "vst-top-level-compose.yml"
    dest = vst / "compose.yml"
    if src.exists():
        shutil.copy(src, dest)
        print("  Wrote top-level vst/compose.yml")
    else:
        print(f"  WARNING: Reference not found: {src}")


def remove_upstream_leftovers(vst: Path, sop_vst: Path):
    """Remove the upstream monolith SDR and stray scripts dir not used by SOP."""
    monolith = sop_vst / "sdr-streamprocessing"
    if monolith.exists():
        shutil.rmtree(monolith)
        print("  Removed upstream monolith sdr-streamprocessing/")
    scripts_dir = vst / "scripts"
    if scripts_dir.exists():
        shutil.rmtree(scripts_dir)
        print("  Removed upstream vst/scripts/")


def write_sop_env(sop_vst: Path):
    """Rewrite .env for SOP layout (Step 4 of vios-building.md)."""
    env_content = """\
VST_INSTALL_ADDITIONAL_PACKAGES=true
VST_BASE_PATH=${MDX_SAMPLE_APPS_DIR}/vst/sop/vst
VST_CONFIG_PATH=${VST_BASE_PATH}/configs
VST_VOLUME=${MDX_DATA_DIR}/data_log/vst
VST_DATA_PATH=${VST_VOLUME}/vst_data
VST_VIDEO_STORAGE_PATH=${VST_VOLUME}/vst_video
VST_TEMP_FILES_PATH=${VST_VOLUME}/temp_files
VST_LOGS=${VST_DATA_PATH}/logs
CLIP_STORAGE_PATH=${VST_VOLUME}/clip_storage
VST_INGRESS_HTTP_PORT=30888
VST_INGRESS_ENDPOINT=localhost:${VST_INGRESS_HTTP_PORT:-30888}/vst

SENSOR_HTTP_PORT=30000
RTSP_SERVER_HTTP_PORT_1=30001
RECORDER_HTTP_PORT_1=30006
STORAGE_HTTP_PORT=30011
REPLAYSTREAM_HTTP_PORT_1=30012
LIVESTREAM_HTTP_PORT_1=30017
RTSP_SERVER_PORT_1=30554

RTSP_SERVER_MODULE_ENDPOINT=http://localhost:10000
RECORDER_MODULE_ENDPOINT=http://localhost:10001
REPLAYSTREAM_MODULE_ENDPOINT=http://localhost:10002
LIVESTREAM_MODULE_ENDPOINT=http://localhost:10003
STORAGE_MODULE_ENDPOINT=http://${HOST_IP}:${STORAGE_HTTP_PORT}
SENSOR_MODULE_ENDPOINT=http://localhost:${SENSOR_HTTP_PORT}

RTSPSERVER_BASE_ID=1
RECORDER_BASE_ID=2
REPLAYSTREAM_BASE_ID=3
LIVESTREAM_BASE_ID=4

REDIS_HOSTADDR=${HOST_IP}
REDIS_PORT=6379
REDIS_MSG_KEY=vst.event
KAFKA_BOOTSTRAP_URL=${HOST_IP}:9092
KAFKA_MSG_KEY=sensor.id

CENTRALIZE_DB_NAME=nvcentralizedb
CENTRALIZE_DB_USERNAME=vst

MINIO_API_PORT=9000
MINIO_CONSOLE_PORT=9001
MINIO_BIND_IP=${HOST_IP}
MINIO_DATA_PATH=${VST_VOLUME}/minio/data
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=vsssop123?
MINIO_API_DELETE_CLEANUP_INTERVAL=20s
MINIO_API_STALE_UPLOADS_CLEANUP_INTERVAL=300s
MINIO_API_STALE_UPLOADS_EXPIRY=7d

MCP_GATEWAY_CPP_API_BASE_URL=http://${HOST_IP}:30888/vst
MCP_GATEWAY_CPP_API_TIMEOUT=30
MCP_GATEWAY_SERVER_NAME=vst-mcp-server
MCP_GATEWAY_SERVER_VERSION=1.0.0
MCP_GATEWAY_SERVER_HOST=${HOST_IP}
MCP_GATEWAY_SERVER_PORT=8001
MCP_GATEWAY_LOG_LEVEL=INFO
MCP_GATEWAY_ENABLE_JSONRPC_LOGGING=true

POSTGRES_IMAGE=postgres:17.6-alpine
NGINX_IMAGE=nvcr.io/nvidia/vss-core/vss-vios-ingress:3.1.0
SDR_IMAGE=nvcr.io/nvidia/vss-core/sdr:3.1.0
ENVOY_PROXY_IMAGE=nvcr.io/nvidia/vss-core/envoy-proxy:3.1.0

MINIO_IMAGE=quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z
VST_RTSPSERVER_IMAGE=nvcr.io/nvidia/vss-core/vss-vios-rtspserver:${VST_RTSPSERVER_IMAGE_TAG}
VST_RECORDER_IMAGE=nvcr.io/nvidia/vss-core/vss-vios-recorder:${VST_RECORDER_IMAGE_TAG}
VST_STORAGE_IMAGE=nvcr.io/nvidia/vss-core/vss-vios-storage:${VST_STORAGE_IMAGE_TAG}
VST_REPLAYSTREAM_IMAGE=nvcr.io/nvidia/vss-core/vss-vios-replaystream:${VST_REPLAYSTREAM_IMAGE_TAG}
VST_LIVESTREAM_IMAGE=nvcr.io/nvidia/vss-core/vss-vios-livestream:${VST_LIVESTREAM_IMAGE_TAG}

VST_SENSOR_IMAGE=nvcr.io/nvidia/vss-core/vss-vios-sensor:${VST_SENSOR_IMAGE_TAG}
VST_MCP_IMAGE=nvcr.io/nvidia/vss-core/vss-vios-mcp:${VST_MCP_IMAGE_TAG}
"""
    (sop_vst / ".env").write_text(env_content)
    print("  Wrote .env")


def write_sop_docker_compose(refs: Path, sop_vst: Path):
    """Copy the reference SOP docker-compose.yaml (Steps 3a-3f of vios-building.md).

    The full compose is the single source of truth at
    references/configs/vios/sop-docker-compose.yaml; this only copies it.
    """
    src = refs / "configs" / "vios" / "sop-docker-compose.yaml"
    dest = sop_vst / "docker-compose.yaml"
    if src.exists():
        shutil.copy(src, dest)
        print("  Wrote docker-compose.yaml")
    else:
        print(f"  WARNING: Reference not found: {src}")


def write_nginx_conf(configs_dir: Path):
    """Create static nginx.conf (Step 5 of vios-building.md)."""
    nginx_content = """\
events {
    worker_connections 1024;
}

http {
    server {
        listen 30888;

        location = / {
            return 301 /vst/;
        }

        location /vst/storage/ {
            proxy_pass http://localhost:30011/storage/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }

        location /vst/api/v1/proxy/ {
            proxy_pass http://localhost:10000/api/v1/proxy/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }

        location /vst/api/v1/record/ {
            proxy_pass http://localhost:30006/api/v1/record/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }

        location /vst/api/v1/replay/ {
            proxy_pass http://localhost:30012/api/v1/replay/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        location /vst/api/v1/live/ {
            proxy_pass http://localhost:30017/api/v1/live/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        location /vst/api/v1/storage/ {
            proxy_pass http://localhost:30011/api/v1/storage/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }

        location /vst/ {
            proxy_pass http://localhost:30000/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }
}
"""
    configs_dir.mkdir(parents=True, exist_ok=True)
    (configs_dir / "nginx.conf").write_text(nginx_content)
    print("  Wrote configs/nginx.conf")


def modify_adaptor_config(configs_dir: Path):
    """Remove media_adaptor_lib_path from vst_rtsp adaptor (Step 8a)."""
    path = configs_dir / "adaptor_config.json"
    if not path.exists():
        print("  SKIP adaptor_config.json (not found)")
        return
    data = json.loads(path.read_text())
    if "adaptors" in data:
        for adaptor in data["adaptors"]:
            if adaptor.get("name") == "vst_rtsp":
                adaptor.pop("media_adaptor_lib_path", None)
    path.write_text(json.dumps(data, indent=2) + "\n")
    print("  Modified configs/adaptor_config.json")


def modify_rtsp_streams(configs_dir: Path):
    """Set enabled=true, max_stream_count=100 (Step 8b)."""
    path = configs_dir / "rtsp_streams.json"
    if not path.exists():
        print("  SKIP rtsp_streams.json (not found)")
        return
    data = json.loads(path.read_text())
    data["enabled"] = True
    data["max_stream_count"] = 100
    path.write_text(json.dumps(data, indent=2) + "\n")
    print("  Modified configs/rtsp_streams.json")


def modify_vst_configs(configs_dir: Path):
    """Apply common changes to vst_config*.json (Step 8e)."""
    for filename in ["vst_config.json", "vst_config_redis.json", "vst_config_kafka.json"]:
        path = configs_dir / filename
        if not path.exists():
            print(f"  SKIP {filename} (not found)")
            continue
        data = json.loads(path.read_text())

        # Common changes
        data["max_webrtc_out_connections"] = 8
        data["max_webrtc_in_connections"] = 8
        data["rtp_udp_port_range"] = "31000-31100"
        data["websocket_keep_alive_ms"] = 5000
        data.pop("ai_bridge_endpoint", None)
        # nv_streamer_loop_playback lives under the "data" section; writing it at the
        # JSON root is a no-op (VST reads data.nv_streamer_loop_playback).
        if isinstance(data.get("data"), dict):
            data["data"]["nv_streamer_loop_playback"] = True
            data.pop("nv_streamer_loop_playback", None)
        else:
            data["nv_streamer_loop_playback"] = True
        data["nv_streamer_sync_playback"] = False
        data["nv_streamer_sync_file_count"] = 0
        data.pop("webrtc_out_default_resolution", None)
        data["download_files_timeout_secs"] = 300
        data["qos_logfile_path"] = ""
        # rtsp_server_instances_count lives under the "network" section; >1 makes the
        # rtspserver load-balancer round-robin proxy ports (30554/30555/...) but
        # recorder-ms always uses base 30554 -> 404 / no recording. Writing it at the
        # JSON root is a no-op (VST reads network.rtsp_server_instances_count).
        if isinstance(data.get("network"), dict):
            data["network"]["rtsp_server_instances_count"] = 1
            data.pop("rtsp_server_instances_count", None)
        else:
            data["rtsp_server_instances_count"] = 1

        # Remove halo safety block
        keys_to_remove = [k for k in data if k.startswith("halo_safety")]
        for k in keys_to_remove:
            del data[k]

        # Per-file specifics
        if filename == "vst_config_kafka.json":
            data["use_webrtc_hw_dec"] = False
        else:
            data["use_webrtc_hw_dec"] = True

        if filename == "vst_config.json":
            data["webrtc_in_video_degradation_preference"] = "detail"

        if filename == "vst_config_redis.json":
            data["observability"] = {
                "enable_telemetry": False,
                "otlp_endpoint": "http://localhost:4318/v1/traces"
            }

        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"  Modified configs/{filename}")


def _envoy_yaml(module: str, listen_port: int, sdr_port: int, http_port: int) -> str:
    """Generate envoy.yaml for a given module."""
    return f"""\
node:
  cluster: services
  id: ucs-svc-proxy
dynamic_resources:
  cds_config:
    api_config_source:
      transport_api_version: V3
      refresh_delay: 5s
      api_type: REST
      cluster_names: [xds_cluster]

static_resources:
  listeners:
  - name: svc_listener
    address:
      socket_address: {{ address: 0.0.0.0, port_value: {listen_port} }}
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          stat_prefix: ingress_http
          access_log:
          - name: envoy.access_loggers.file
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.access_loggers.file.v3.FileAccessLog
              path: /dev/stdout
              log_format:
                json_format:
                  custom_header: "%REQ(MY_CUSTOM_HEADER)%"
                  cluster_header: "%REQ(cluster_header)%"
                  authority: "%REQ(:AUTHORITY)%"
                  method: "%REQ(:METHOD)%"
                  path: "%REQ(X-ENVOY-ORIGINAL-PATH?:PATH)%"
                  protocol: "%PROTOCOL%"
                  response_code: "%RESPONSE_CODE%"
                  response_flags: "%RESPONSE_FLAGS%"
                  route_name: "%ROUTE_NAME%"
                  upstream_host: "%UPSTREAM_HOST%"
                  upstream_cluster: "%UPSTREAM_CLUSTER%"
                  upstream_time_ms: "%RESP(X-ENVOY-UPSTREAM-SERVICE-TIME)%"
                  total_time_ms: "%DURATION%"
                  user_agent: "%REQ(USER-AGENT)%"
                  x_client_namespace: "%REQ(X-CLIENT-NAMESPACE)%"
                  x_forwarded_for: "%REQ(X-FORWARDED-FOR)%"
                  request_id: "%REQ(X-REQUEST-ID)%"
                  grpc_status: "%GRPC_STATUS%"
          common_http_protocol_options:
            idle_timeout: 3600s  # 1 hour
          stream_idle_timeout: 300s  # 5 mins, must be disabled for long-lived and streaming requests
          upgrade_configs:
          - upgrade_type: websocket
          request_timeout: 300s  # 5 mins, must be disabled for long-lived and streaming requests
          stream_error_on_invalid_http_message: false
          rds:
            route_config_name: {module}-ms_route
            config_source:
              resource_api_version: V3
              api_config_source:
                request_timeout: 5s
                refresh_delay: 10s
                api_type: REST
                transport_api_version: V3
                cluster_names: [xds_cluster]
          codec_type: AUTO
          http_filters:
          - name: envoy.filters.http.lua
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.lua.v3.Lua
              default_source_code:
                inline_string: |
                      local redis = require 'redis'
                      local redisHost = os.getenv("WDM_WL_REDIS_SERVER")
                      local redisPort = os.getenv("WDM_WL_REDIS_PORT")
                      local client = redis.connect(redisHost, redisPort)
                      function envoy_on_request(request_handle)
                        local wlObj = os.getenv("WDM_WL_OBJECT_NAME")
                        local routeHeader = os.getenv("ENVOYROUTEHEADER")
                        local noHeaderTargetContainer = os.getenv("NOHEADERTARGETCONTAINER")
                        local noHeaderTargetport = os.getenv("NOHEADERTARGETPORT")
                        local containerName = nil
                        local containerHost = nil
                        id = request_handle:headers():get(routeHeader)
                        if id ~= nil then
                          request_handle:logInfo("id not nil " ..id)
                          containerName = client:hget(wlObj, id )
                          if containerName ~= nil then
                            request_handle:logInfo("containerName not nil :" ..containerName)
                            request_handle:logInfo(containerName .. containerName)
                            containerHost = client:hget(wlObj.."-pod", containerName)
                            if containerHost ~= nil then
                              request_handle:logInfo("routing stream id "..id.." to "..containerHost)
                            end
                          end

                          if containerName ~= nil then
                            request_handle:logInfo("containerName:" ..containerName)
                            if request_handle:headers():get("Sec-WebSocket-Key") ~= nil then
                              request_handle:logInfo("Websocket request")
                              request_handle:headers():replace (
                                    "upstream-cluster",
                                    containerName .. "-" .. containerName .."-websocket"
                                )
                            else
                              request_handle:logInfo("Not a websocket request")
                              message_encoding_header = request_handle:headers():get("content-type")
                              if message_encoding_header ~= 'application/grpc' then
                                request_handle:headers():replace (
                                    "upstream-cluster", containerName .. "-" .. containerName)
                              else
                                request_handle:headers():replace (
                                    "upstream-cluster",
                                    containerName..'-grpc'
                                )
                              end
                            end
                          end
                        else
                          request_handle:logInfo("Request has no stream header, routing directly")
                          request_handle:headers():replace (
                                  "upstream-cluster",
                                  "headerless_service")
                        end
                      end
          - name: envoy.filters.http.router
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router


  clusters:
  - type: STRICT_DNS
    connect_timeout: 1s
    typed_extension_protocol_options:
      envoy.extensions.upstreams.http.v3.HttpProtocolOptions:
        "@type": type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions
        explicit_http_config:
          http_protocol_options: {{}}
    name: xds_cluster
    load_assignment:
      cluster_name: xds_cluster
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: "127.0.0.1"
                port_value: {sdr_port}
  - type: STRICT_DNS
    connect_timeout: 0.25s
    name: headerless_service
    lb_policy: ROUND_ROBIN
    typed_extension_protocol_options:
      envoy.extensions.upstreams.http.v3.HttpProtocolOptions:
        "@type": type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions
        explicit_http_config:
          http_protocol_options: {{}}
    load_assignment:
      cluster_name: headerless_service
      endpoints:
        - lb_endpoints:
            - endpoint:
                address:
                  socket_address:
                    address: "127.0.0.1"
                    port_value: "{http_port}"
"""


def _sdr_compose(module: str, image_var: str, http_port_var: str,
                 sdr_port: int, base_id_var: str,
                 wl_add_url: str, wl_delete_url: str,
                 wl_health_url: str, wl_change_id_add: str,
                 extra_env: str = "") -> str:
    """Generate sdr-compose.yaml for a given module."""
    rtsp_line = ""
    if module == "rtspserver":
        rtsp_line = "\n      - RTSP_SERVER_PORT=${RTSP_SERVER_PORT_1}"

    storage_line = ""
    if module == "recorder":
        storage_line = "\n      - STORAGE_MODULE_ENDPOINT=${STORAGE_MODULE_ENDPOINT}"

    return f"""\
services:
  {module}-ms-1-sop:
    image: ${{{image_var}}}
    profiles: ["bp_sop_2d","bp_smc_2d","bp_developer_alerts_2d_cv","bp_developer_alerts_2d_vlm"]
    container_name: {module}-ms-1-sop
    network_mode: host
    runtime: nvidia
    user: "0:0"
    entrypoint: ["/bin/bash", "-c", "if [ \\"$$VST_INSTALL_ADDITIONAL_PACKAGES\\" = \\"true\\" ]; then /home/vst/vst_release/tools/user_additional_install.sh; fi && exec /home/vst/vst_release/launch_vst"]
    environment:
      - VST_INSTALL_ADDITIONAL_PACKAGES=${{VST_INSTALL_ADDITIONAL_PACKAGES}}
      - ADAPTOR=vst_rtsp
      - HTTP_PORT=${{{http_port_var}}}{rtsp_line}
      - CENTRALIZE_DB_NAME=${{CENTRALIZE_DB_NAME}}
      - CENTRALIZE_DB_USERNAME=${{CENTRALIZE_DB_USERNAME}}
      - SENSOR_MODULE_ENDPOINT=${{SENSOR_MODULE_ENDPOINT}}
      - VST_INGRESS_ENDPOINT=${{VST_INGRESS_ENDPOINT}}{storage_line}
    volumes:
      - ${{VST_CONFIG_PATH}}:/home/vst/vst_release/configs
      - ${{VST_DATA_PATH}}:/home/vst/vst_release/vst_data
      - ${{VST_VIDEO_STORAGE_PATH}}:/home/vst/vst_release/vst_video
      - ${{CLIP_STORAGE_PATH}}:/home/vst/vst_release/streamer_videos
      - ${{VST_TEMP_FILES_PATH}}:/home/vst/vst_release/webroot/temp_files
    deploy:
      restart_policy:
        condition: always
    depends_on:
      redis:
        condition: service_started

  sdr-http-{module}-sop:
    image: ${{SDR_IMAGE}}
    profiles: ["bp_sop_2d","bp_smc_2d","bp_developer_alerts_2d_cv","bp_developer_alerts_2d_vlm"]
    container_name: sdr-http-{module}-sop
    user: "0:0"
    network_mode: "host"
    logging:
      driver: "json-file"
      options:
        max-size: "8192m"
        max-file: "3"
    volumes:
      - ./sdr-config:/wdm-configs
      - /var/run/docker.sock:/var/run/docker.sock
      - ${{VST_DATA_PATH}}/sdr/{module}/log:/log
    environment:
      PORT: {sdr_port}
      WDM_CLUSTER_CONFIG_FILE: /wdm-configs/docker_cluster_config.json
      WDM_MSG_KEY: ${{REDIS_MSG_KEY}}
      WDM_WL_REDIS_SERVER: ${{REDIS_HOSTADDR}}
      WDM_WL_REDIS_PORT: ${{REDIS_PORT}}
      WDM_WL_REDIS_MSG_FIELD: sensor.id
      WDM_WL_ADD_URL: {wl_add_url}
      WDM_WL_DELETE_URL: {wl_delete_url}
      WDM_WL_HEALTH_CHECK_URL: {wl_health_url}
      WDM_WL_CHANGE_ID_ADD: {wl_change_id_add}
      WDM_WL_CHANGE_ID_DEL: camera_remove
      WDM_PRELOAD_WORKLOAD: ./tests/event_pre-roll.json
      WDM_CLEAR_DATA_WL: true
      WDM_KFK_ENABLE: false
      WDM_MSG_TOPIC: vst_events
      WDM_KFK_BOOTSTRAP_URL: ${{KAFKA_BOOTSTRAP_URL}}
      WDM_DS_SWAP_ID_NAME: false
      WDM_WL_THRESHOLD: 100
      WDM_ADD_REMOVE_RETRY_ATTEMPTS: 50
      WDM_CLUSTER_TYPE: docker
      WDM_POD_WATCH_DOCKER_DELAY: 0.5
      WDM_RESTART_DS_ON_ADD_FAIL: false
      WDM_DISABLE_WERKZEUG_LOGGING: true
      WDM_WL_OBJECT_NAME: {module}-ms
      WDM_CONSUMER_GRP_ID: sdr-http-{module}-cg
      WDM_CLUSTER_CONTAINER_NAMES: '["{module}-ms-1"]'
      VST_STREAMS_ENDPOINT: http://localhost:30000/api/v1/sensor/streams
      VST_STATUS_ENDPOINT: http://localhost:30000/api/v1/sensor/status
      OTEL_SDK_DISABLED: true
      WDM_INITIALIZE_FROM_VST: false
      ENVOY_REQUEST_TIMEOUT: 300
      WDM_TARGET_PORT_MAPPING: '{{"{module}-ms-1": ${{{http_port_var}}}}}'
      OTEL_SERVICE_NAME: SDR_AGENT
      WDM_REDIS_CACHE_OBJECT: "{module}-data"
      WDM_WL_NAME_IGNORE_REGEX: ""
    deploy:
      resources:
        limits:
          memory: 300M
      restart_policy:
        condition: always
    depends_on:
      redis:
        condition: service_started
      {module}-ms-1-sop:
        condition: service_started

  envoy-http-{module}-sop:
    image: ${{ENVOY_PROXY_IMAGE}}
    profiles: ["bp_sop_2d","bp_smc_2d","bp_developer_alerts_2d_cv","bp_developer_alerts_2d_vlm"]
    user: "0:0"
    command: /usr/local/bin/envoy -c /etc/envoy/envoy.yaml --concurrency 16 --base-id ${{{base_id_var}}}
    network_mode: "host"
    container_name: envoy-http-{module}-sop
    volumes:
      - ./envoy.yaml:/etc/envoy/envoy.yaml
    environment:
      WDM_WL_REDIS_SERVER: ${{REDIS_HOSTADDR}}
      WDM_WL_REDIS_PORT: ${{REDIS_PORT}}
      WDM_KFK_BOOTSTRAP_URL: ${{KAFKA_BOOTSTRAP_URL}}
      WDM_WL_OBJECT_NAME: {module}-ms
      ENVOYROUTEHEADER: "streamid"
      NOHEADERTARGETCONTAINER: "{module}-ms-1"
    depends_on:
      redis:
        condition: service_started
      {module}-ms-1-sop:
        condition: service_started
"""


# Module config: (module, image_var, http_port_var, envoy_listen, sdr_port, http_port, base_id_var, wl_add, wl_delete, wl_health, wl_change_id)
SDR_MODULES = [
    {
        "module": "rtspserver",
        "image_var": "VST_RTSPSERVER_IMAGE",
        "http_port_var": "RTSP_SERVER_HTTP_PORT_1",
        "envoy_listen": 10000,
        "sdr_port": 4003,
        "http_port": 30001,
        "base_id_var": "RTSPSERVER_BASE_ID",
        "wl_add_url": "/api/v1/proxy/stream/add",
        "wl_delete_url": "/api/v1/proxy/stream/",
        "wl_health_url": "/api/v1/proxy/configuration",
        "wl_change_id_add": "camera_proxy",
    },
    {
        "module": "recorder",
        "image_var": "VST_RECORDER_IMAGE",
        "http_port_var": "RECORDER_HTTP_PORT_1",
        "envoy_listen": 10001,
        "sdr_port": 4002,
        "http_port": 30006,
        "base_id_var": "RECORDER_BASE_ID",
        "wl_add_url": "/api/v1/record/stream/add",
        "wl_delete_url": "/api/v1/record/stream/",
        "wl_health_url": "/api/v1/record/configuration",
        "wl_change_id_add": "camera_streaming",
    },
    {
        "module": "replaystream",
        "image_var": "VST_REPLAYSTREAM_IMAGE",
        "http_port_var": "REPLAYSTREAM_HTTP_PORT_1",
        "envoy_listen": 10002,
        "sdr_port": 4004,
        "http_port": 30012,
        "base_id_var": "REPLAYSTREAM_BASE_ID",
        "wl_add_url": "/api/v1/replay/stream/add",
        "wl_delete_url": "/api/v1/replay/stream/",
        "wl_health_url": "/api/v1/replay/configuration",
        "wl_change_id_add": "camera_streaming",
    },
    {
        "module": "livestream",
        "image_var": "VST_LIVESTREAM_IMAGE",
        "http_port_var": "LIVESTREAM_HTTP_PORT_1",
        "envoy_listen": 10003,
        "sdr_port": 4005,
        "http_port": 30017,
        "base_id_var": "LIVESTREAM_BASE_ID",
        "wl_add_url": "/api/v1/live/stream/add",
        "wl_delete_url": "/api/v1/live/stream/",
        "wl_health_url": "/api/v1/live/configuration",
        "wl_change_id_add": "camera_streaming",
    },
]


def write_sdr_modules(sop_vst: Path):
    """Create all 4 SDR module directories with compose + envoy (Step 6)."""
    for cfg in SDR_MODULES:
        module = cfg["module"]
        sdr_dir = sop_vst / f"sdr-{module}-http"
        sdr_dir.mkdir(parents=True, exist_ok=True)
        (sdr_dir / "sdr-config").mkdir(exist_ok=True)

        # Write sdr-compose.yaml
        compose = _sdr_compose(
            module=module,
            image_var=cfg["image_var"],
            http_port_var=cfg["http_port_var"],
            sdr_port=cfg["sdr_port"],
            base_id_var=cfg["base_id_var"],
            wl_add_url=cfg["wl_add_url"],
            wl_delete_url=cfg["wl_delete_url"],
            wl_health_url=cfg["wl_health_url"],
            wl_change_id_add=cfg["wl_change_id_add"],
        )
        (sdr_dir / "sdr-compose.yaml").write_text(compose)

        # Write envoy.yaml
        envoy = _envoy_yaml(
            module=module,
            listen_port=cfg["envoy_listen"],
            sdr_port=cfg["sdr_port"],
            http_port=cfg["http_port"],
        )
        (sdr_dir / "envoy.yaml").write_text(envoy)

        print(f"  Created sdr-{module}-http/ (sdr-compose.yaml + envoy.yaml)")


def write_minio_compose(refs: Path, sop_vst: Path):
    """Copy the reference minio compose to minio/minio-compose.yaml (Step 7).

    The minio compose is the single source of truth at
    references/configs/vios/minio-server.service.yml; this only copies it.
    """
    minio_dir = sop_vst / "minio"
    minio_dir.mkdir(parents=True, exist_ok=True)
    src = refs / "configs" / "vios" / "minio-server.service.yml"
    dest = minio_dir / "minio-compose.yaml"
    if src.exists():
        shutil.copy(src, dest)
        print("  Created minio/minio-compose.yaml")
    else:
        print(f"  WARNING: Reference not found: {src}")


if __name__ == "__main__":
    main()

