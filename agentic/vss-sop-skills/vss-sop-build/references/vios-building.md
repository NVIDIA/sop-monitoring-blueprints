# Convert VSS VIOS (VST) → VSS SOP VIOS

> **Scope:** modifications inside `deployments/vst/`. For top-level compose change (`./vst/developer/vst/docker-compose.yaml` → `./vst/compose.yml`), see Step 0.1 in [`../SKILL.md`](../SKILL.md).

## Step 0 — Copy from Upstream and Restructure

**Approach:** First find the VIOS (VST) folder in `video-search-and-summarization/deployments/`, copy it to `deployments/vst/` (renaming `developer/vst/` → `sop/vst/` to follow SOP folder structure), then modify the copied folder to work with the SOP profile.

Run these two scripts in order:

1. **Copy & restructure:** `./agentic/vss-sop-skills/vss-sop-build/scripts/copy_vios_from_upstream.sh`
   - Finds `video-search-and-summarization/deployments/vst/developer/vst/`
   - Copies contents to `deployments/vst/sop/vst/`
   - Creates top-level `vst/compose.yml`
   - Creates additional SOP directories (minio, 4 SDR module dirs)
   - Removes upstream-only files (nginx templates, `data_wl.yaml`)

2. **Modify for SOP:** `./agentic/vss-sop-skills/vss-sop-build/scripts/modify_vios_for_sop.sh`
   - Rewrites `docker-compose.yaml` (services renamed -dev → -sop, profiles → `bp_sop_2d`, monolith split into microservices)
   - Rewrites `.env` (SOP paths, microservice ports, MinIO, MCP vars, new images)
   - Modifies JSON config files (adaptor, rtsp_streams, vst_config*)
   - Creates static `nginx.conf` (replaces template)
   - Creates 4 SDR module directories (sdr-compose.yaml + envoy.yaml each)
   - Creates `minio/minio-compose.yaml`
   - Copies reference `docker_cluster_config.json` files

## Overview of Changes

| # | Change |
|---|---|
| 1 | **Restructure:** `vst/developer/vst/` → `vst/sop/vst/` + new top-level `vst/compose.yml` |
| 2 | **Monolith → 5 microservices:** `streamprocessing-ms-dev` → `rtspserver-ms-1-sop`, `recorder-ms-1-sop`, `storage-ms-sop`, `replaystream-ms-1-sop`, `livestream-ms-1-sop` |
| 3 | **SDR split:** `sdr-streamprocessing` → 4 SDRs (`sdr-http-{rtspserver,recorder,replaystream,livestream}-sop`) with unique ports + envoy base-IDs |
| 4 | **New services:** `storage-ms-sop`, `minio-server` |
| 5 | **Profiles:** developer profiles → `bp_sop_2d` |
| 6 | **Naming:** `-dev` suffix → `-sop` |
| 7 | **Nginx:** template-based (`nginx-vst.conf.template`) → static `nginx.conf` |
| 8 | **Envoy Lua:** standard (rtspserver, recorder) + WebSocket-aware (replaystream, livestream) |
| 9 | **Dependencies:** `service_healthy` → `service_started`; cross-service deps |
| 10 | **Config tuning:** WebRTC limits, loop playback, software path, timeouts; halo-safety removed |
| 11 | **MCP Gateway:** hardcoded → `${...}` env refs |
| 12 | **Removed:** `scripts/user_additional_install.sh`, `nginx-mms*`, `nginx-vst.conf*`, `sdr-streamprocessing/sdr-config/data_wl.yaml` |

### Files Created (SOP)

| File | Purpose |
|---|---|
| `vst/compose.yml` | Top-level include → `sop/vst/docker-compose.yaml` |
| `vst/sop/vst/docker-compose.yaml` | Main: centralizedb, nginx, sensor-ms, storage-ms, vst-mcp |
| `vst/sop/vst/.env` | All VST env vars (microservice ports, MinIO, MCP Gateway) |
| `vst/sop/vst/configs/nginx.conf` | Static per-service routing (replaces template) |
| `vst/sop/vst/configs/{adaptor,vst,vst_kafka,vst_redis,vst_storage,rtsp_streams}_*.json` | Tuned configs |
| `vst/sop/vst/configs/postgresql.conf` | Postgres (unchanged minus license header) |
| `vst/sop/vst/sdr-{rtspserver,recorder,replaystream,livestream}-http/` | Per-module SDR + envoy |
| `vst/sop/vst/minio/minio-compose.yaml` | MinIO object storage |

---

## Step 1 — Top-Level `vst/compose.yml` (NEW)

See `./configs/vios/vst-top-level-compose.yml` — a single include: `sop/vst/docker-compose.yaml`. Replaces the upstream `./vst/developer/vst/docker-compose.yaml` reference at the build root.

## Step 2 — Create Directory Structure

> **Note:** When using the automated scripts (`copy_vios_from_upstream.sh` + `modify_vios_for_sop.sh`), this step is handled automatically. The copy script creates the SOP directory tree including minio and all 4 SDR module directories.

For manual builds, run `./scripts/vios/create_dirs.sh`. It creates `vss-sop/deployments/vst/sop/vst/{configs, minio}` plus four SDR directories (`sdr-rtspserver-http/sdr-config`, `sdr-recorder-http/sdr-config`, `sdr-replaystream-http/sdr-config`, `sdr-livestream-http/sdr-config`).

## Step 3 — `sop/vst/docker-compose.yaml`

Start from upstream `developer/vst/docker-compose.yaml`, then apply:

### 3a. Replace include section

See the `include:` section of `./configs/vios/sop-docker-compose.yaml` — five `$MDX_SAMPLE_APPS_DIR/vst/sop/vst/...` includes covering each VST microservice (`sdr-{rtspserver,recorder,replaystream,livestream}-http/sdr-compose.yaml`) plus `minio/minio-compose.yaml`.

### 3b. `centralizedb-dev` → `centralizedb-sop`

| Property | Upstream → SOP |
|---|---|
| Service / container name | `centralizedb-dev` → `centralizedb-sop` |
| Profiles | dev → `["bp_sop_2d"]` |
| Extra env | add `VST_INGRESS_ENDPOINT=${VST_INGRESS_ENDPOINT}` |
| Health check | **removed** |

### 3c. `vst-ingress-dev` → `vst-ingress-sop`

| Property | Upstream → SOP |
|---|---|
| Service / container name | `vst-ingress-dev` → `vst-ingress-sop` |
| Profiles | dev → `["bp_sop_2d"]` |
| Volumes | `nginx-${NGINX_MODE:-vst}.conf.template` + rendered conf → single `nginx.conf:/etc/nginx/nginx.conf` |
| `environment` | drop `HOST_IP`, `EXTERNAL_IP`, `VST_INGRESS_HTTP_PORT` (no template rendering) |
| `command` | drop `sed` template + `exec nginx` (static config) |
| Health check, `depends_on: sensor-ms-dev: service_healthy` | **removed** |

### 3d. `sensor-ms-dev` → `sensor-ms-sop`

| Property | Upstream → SOP |
|---|---|
| Service / container name | `sensor-ms-dev` → `sensor-ms-sop` |
| Profiles | dev → `["bp_sop_2d"]` |
| `HTTP_PORT` | `${SENSOR_HTTP_PORT:-30000}` → `${SENSOR_HTTP_PORT}` (no default) |
| Module endpoints | `STREAM_PROCESSOR_MODULE_ENDPOINT` → `RTSP_SERVER_MODULE_ENDPOINT`, `RECORDER_MODULE_ENDPOINT`, `STORAGE_MODULE_ENDPOINT` |
| Health check | **removed** |
| `depends_on` | `redis`, `vst-ingress-sop`, `centralizedb-sop`, all 4 SDR + envoy services (all `service_started`) |

### 3e. Add `storage-ms-sop` (NEW — handles clip downloads, temp files)

See the `storage-ms-sop` service in `./configs/vios/sop-docker-compose.yaml` (under `services:`). Key wiring:

- **Image:** `${VST_STORAGE_IMAGE}`; `profiles: ["bp_sop_2d"]`; `network_mode: host`; `runtime: nvidia`; `restart: on-failure`.
- **Entrypoint:** conditionally runs `tools/user_additional_install.sh` (when `VST_INSTALL_ADDITIONAL_PACKAGES=true`), then `exec /home/vst/vst_release/launch_vst`.
- **Adaptor flags:** `ADAPTOR=storage`, `NEED_STORAGE=true`, all other `NEED_*` flags `false` (this is a storage-only worker).
- **Ports/endpoints:** `HTTP_PORT=${STORAGE_HTTP_PORT}` (30011), `VST_INGRESS_ENDPOINT`, `CENTRALIZE_DB_*`.
- **Volumes:** the SOP `vst_config.json` (`vst_config_${STREAM_TYPE:-redis}.json`), plus `${VST_VIDEO_STORAGE_PATH}`, `${CLIP_STORAGE_PATH}`, `${VST_TEMP_FILES_PATH}`.
- **Depends on:** `redis`, `vst-ingress-sop`, `centralizedb-sop` (all `service_started` — no `service_healthy`).

### 3f. `vst-mcp-dev` → `vst-mcp-sop`

Service / container name → `-sop`; profiles → `["bp_sop_2d"]`. Convert hardcoded MCP gateway values to env-var references:

| Var | Upstream | SOP |
|---|---|---|
| `MCP_GATEWAY_CPP_API_TIMEOUT` | `30` | `${MCP_GATEWAY_CPP_API_TIMEOUT}` |
| `MCP_GATEWAY_SERVER_NAME` | `vst-mcp-server` | `${MCP_GATEWAY_SERVER_NAME}` |
| `MCP_GATEWAY_SERVER_VERSION` | `1.0.0` | `${MCP_GATEWAY_SERVER_VERSION}` |
| `MCP_GATEWAY_LOG_LEVEL` | `INFO` | `${MCP_GATEWAY_LOG_LEVEL}` |
| `MCP_GATEWAY_ENABLE_JSONRPC_LOGGING` | `true` | `${MCP_GATEWAY_ENABLE_JSONRPC_LOGGING}` |

`depends_on`: `sensor-ms-sop`, `vst-ingress-sop`, `centralizedb-sop` (all `service_started`).

---

## Step 4 — `.env` File

Start from upstream `developer/vst/.env`, then apply:

### 4a. Path

```diff
- VST_BASE_PATH=${MDX_SAMPLE_APPS_DIR}/vst/developer/vst
+ VST_BASE_PATH=${MDX_SAMPLE_APPS_DIR}/vst/sop/vst
```

### 4b. Remove upstream-only vars

Drop: `CLIP_STORAGE_PATH` (moved), `STREAM_PROCESSOR_HTTP_PORT`, `STREAM_PROCESSOR_MODULE_ENDPOINT`, `RTSP_SERVER_PORT`, `VST_ADAPTOR`, section headers/comments.

**Keep and set** `VST_INGRESS_HTTP_PORT` and `VST_INGRESS_ENDPOINT` — storage-ms and replaystream-ms use them to build download URLs. **Critical:** `VST_INGRESS_ENDPOINT` must NOT include `http://` (the microservice prepends it) and MUST include the `/vst` nginx path prefix:

```env
VST_INGRESS_HTTP_PORT=30888
VST_INGRESS_ENDPOINT=localhost:${VST_INGRESS_HTTP_PORT:-30888}/vst
```

If `VST_INGRESS_ENDPOINT` is set to `http://localhost:30888` (the upstream default), storage-ms and replaystream-ms produce double-scheme URLs like `http://http://localhost:30888/storage/...`, which both fail URL validation and have the wrong path (missing `/vst/`).

### 4c. Add microservice ports + endpoints

```env
CENTRALIZE_DB_NAME=nvcentralizedb
CENTRALIZE_DB_USERNAME=vst
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
```

### 4d. Envoy base IDs

```env
RTSPSERVER_BASE_ID=1
RECORDER_BASE_ID=2
REPLAYSTREAM_BASE_ID=3
LIVESTREAM_BASE_ID=4
```

### 4e. MinIO

```env
MINIO_API_PORT=9000
MINIO_CONSOLE_PORT=9001
MINIO_BIND_IP=${HOST_IP}
MINIO_DATA_PATH=${VST_VOLUME}/minio/data
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=vsssop123?
MINIO_API_DELETE_CLEANUP_INTERVAL=20s
MINIO_API_STALE_UPLOADS_CLEANUP_INTERVAL=300s
MINIO_API_STALE_UPLOADS_EXPIRY=7d
```

### 4f. MCP Gateway (moved from hardcoded compose values)

```env
MCP_GATEWAY_CPP_API_BASE_URL=http://${HOST_IP}:30888/vst
MCP_GATEWAY_CPP_API_TIMEOUT=30
MCP_GATEWAY_SERVER_NAME=vst-mcp-server
MCP_GATEWAY_SERVER_VERSION=1.0.0
MCP_GATEWAY_SERVER_HOST=${HOST_IP}
MCP_GATEWAY_SERVER_PORT=8001
MCP_GATEWAY_LOG_LEVEL=INFO
MCP_GATEWAY_ENABLE_JSONRPC_LOGGING=true
```

> **Required:** `MCP_GATEWAY_SERVER_HOST` and `MCP_GATEWAY_SERVER_PORT` must both be present. Without them, `vst-mcp-sop` fails to start with a pydantic `ValidationError: server_port — field required`. The upstream compose hardcodes these values; SOP's env-var approach requires them to be explicit in `.env`.

### 4g. Image variables

Remove `VST_STREAM_PROCESSOR_IMAGE`. Add:

```env
MINIO_IMAGE=quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z
VST_RTSPSERVER_IMAGE=nvcr.io/nvidia/vss-core/vss-vios-rtspserver:${VST_RTSPSERVER_IMAGE_TAG}
VST_RECORDER_IMAGE=nvcr.io/nvidia/vss-core/vss-vios-recorder:${VST_RECORDER_IMAGE_TAG}
VST_STORAGE_IMAGE=nvcr.io/nvidia/vss-core/vss-vios-storage:${VST_STORAGE_IMAGE_TAG}
VST_REPLAYSTREAM_IMAGE=nvcr.io/nvidia/vss-core/vss-vios-replaystream:${VST_REPLAYSTREAM_IMAGE_TAG}
VST_LIVESTREAM_IMAGE=nvcr.io/nvidia/vss-core/vss-vios-livestream:${VST_LIVESTREAM_IMAGE_TAG}
```

---

## Step 5 — Static `configs/nginx.conf` (replaces template)

Per-service routing (no more catch-all to streamprocessing envoy):

| Path | Upstream | SOP |
|---|---|---|
| `/vst/api/v1/replay/stream/.../picture` | regex → `localhost:10000` | **removed** |
| `/vst/api/v1/` (catch-all) | → `localhost:10000` | **removed** |
| `/vst/storage/` | → `localhost:10000` | → `localhost:30011` (storage-ms) |
| `/vst/api/v1/proxy/` | — | → `localhost:10000` (rtspserver envoy) |
| `/vst/api/v1/record/` | — | → `localhost:30006` (recorder) |
| `/vst/api/v1/replay/` | — | → `localhost:30012` (replaystream) + WebSocket |
| `/vst/api/v1/live/` | — | → `localhost:30017` (livestream) + WebSocket |
| `/vst/api/v1/storage/` | — | → `localhost:30011` |
| `/vst/` | — | → `localhost:30000` (sensor-ms UI) |
| `/` | — | → Redirect 301 to `/vst/` |
| `proxy_hide_header Cache-Control` (storage + replay) | present | **removed** |

> **Use `localhost`, not a Docker service name:** All `proxy_pass` targets must be `http://localhost:<PORT>/` — never `http://sensor-ms/` or `http://recorder-ms/`. Because all VST services use `network_mode: host`, Docker's internal DNS for service names is unavailable; nginx would log `host not found in upstream "sensor-ms"` and refuse to start.

---

## Step 6 — SDR Microservice Directories

Each `sdr-{module}-http/` contains: `envoy.yaml`, `sdr-compose.yaml`, `sdr-config/docker_cluster_config.json`.

### 6a. Port + base-id mapping

| Module | Envoy listen | SDR | HTTP | RTSP | Envoy base-id |
|---|---|---|---|---|---|
| **rtspserver** | 10000 | 4003 | 30001 | 30554 | 1 |
| **recorder** | 10001 | 4002 | 30006 | — | 2 |
| **replaystream** | 10002 | 4004 | 30012 | — | 3 |
| **livestream** | 10003 | 4005 | 30017 | — | 4 |

### 6b. Per-module changes (from upstream `sdr-streamprocessing/`)

**`sdr-compose.yaml`:**

| Field | Upstream `streamprocessing` → SOP per-module |
|---|---|
| Service image | `VST_STREAM_PROCESSOR_IMAGE` → `VST_{RTSPSERVER,RECORDER,REPLAYSTREAM,LIVESTREAM}_IMAGE` |
| Profiles | dev → `["bp_sop_2d","bp_smc_2d","bp_developer_alerts_2d_cv","bp_developer_alerts_2d_vlm"]` |
| Container names | `streamprocessing-ms-dev`, `sdr-streamprocessing`, `envoy-streamprocessing` → `{module}-ms-1-sop`, `sdr-http-{module}-sop`, `envoy-http-{module}-sop` |
| `container_name:` (VMS ms + SDR) | **Add explicit `container_name:` to both the VMS ms service and the SDR service** (e.g. `container_name: recorder-ms-1-sop`, `container_name: sdr-http-recorder-sop`). Without this, Docker Compose's project name (`COMPOSE_PROJECT_NAME=mdx`) prefixes the name and appends a replica suffix, making the container unreachable by the expected name in health checks and `vss-sop-test` Phase 1. |
| SDR `PORT` | 4003 → module-specific (table above) |
| SDR env vars | per-module object names, URLs, consumer groups (see 6d) |
| `WDM_CLUSTER_CONTAINER_NAMES` | `'["streamprocessing-ms"]'` → `'["{module}-ms-1"]'` (Crucial: use single double-quotes; do NOT use double-double-quotes `'[""{module}-ms-1""]'` which will fail to parse inside the container) |
| SDR `CONTAINER_NAME`, health check | **removed** |
| Envoy `--base-id` | `1` → `${{MODULE}_BASE_ID}` |
| Envoy `CONTAINER_NAME`, `restart: unless-stopped` | **removed** |
| VMS ms `restart: unless-stopped` | removed (deploy `restart_policy` only) |
| VMS ms health check | **removed** |
| `depends_on` | `centralizedb-dev: healthy`, `redis: started` → module-specific (e.g. rtspserver depends on recorder SDR) |

**`envoy.yaml`** per module:
- Listener `port_value`: 10000 / 10001 / 10002 / 10003
- `route_config_name`: `{module}-ms_route`
- xds_cluster `port_value`: 4003 / 4002 / 4004 / 4005
- headerless_service `port_value`: 30001 / 30006 / 30012 / 30017
- License header removed
- **replaystream / livestream:** replace stock Lua with WebSocket-aware Lua (see 6c)

**`sdr-config/docker_cluster_config.json`** per module:
- Key name: `{rtspserver,recorder,replaystream,livestream}-ms-1`
- `provisioning_address` port: 30001 / 30006 / 30012 / 30017
- **`"process_type": "docker"`** — must be present; without it the SDR cannot attach to the containerised microservice and fails with `KeyError: 'process_type'` on startup, causing all stream provisioning to fail

### 6c. Enhanced WebSocket Lua (replaystream + livestream only)

The Lua script:
1. Detects WebSocket via `Sec-WebSocket-Key` and `Connection: upgrade` headers
2. WebSocket: extracts `streamId` from URL query parameters
3. Non-WebSocket: reads `streamId` from the standard route header

Replaces upstream's simpler "read header only" Lua.

### 6d. Per-Module SDR WDM Env Vars (CRITICAL)

The upstream monolith uses `/api/v1/proxy/stream/add` everywhere. In SOP, each microservice owns its own API prefix — get this wrong and recording fails with "Stream not present in recorder".

| SDR Module | `WDM_WL_ADD_URL` | `WDM_WL_DELETE_URL` | `WDM_WL_HEALTH_CHECK_URL` | `WDM_WL_CHANGE_ID_ADD` |
|---|---|---|---|---|
| **sdr-http-rtspserver-sop** | `/api/v1/proxy/stream/add` | `/api/v1/proxy/stream/` | `/api/v1/proxy/configuration` | `camera_proxy` |
| **sdr-http-recorder-sop** | `/api/v1/record/stream/add` | `/api/v1/record/stream/` | `/api/v1/record/configuration` | `camera_streaming` |
| **sdr-http-replaystream-sop** | `/api/v1/replay/stream/add` | `/api/v1/replay/stream/` | `/api/v1/replay/configuration` | `camera_streaming` |
| **sdr-http-livestream-sop** | `/api/v1/live/stream/add` | `/api/v1/live/stream/` | `/api/v1/live/configuration` | `camera_streaming` |

**Why:** `WDM_WL_ADD_URL` provisions streams into the target microservice. Recorder SDR using rtspserver's `/api/v1/proxy/stream/add` returns 404 (recorder has no proxy route) → streams never provisioned.

**Event ordering:** `WDM_WL_CHANGE_ID_ADD` selects the Redis event. `camera_proxy` (sensor added → needs RTSP proxy) for rtspserver. `camera_streaming` (rtspserver proxy live → downstream can connect) for recorder/replaystream/livestream.

**Also:** `recorder-ms` needs `STORAGE_MODULE_ENDPOINT` in its environment (checks disk capacity before adding streams). Without it, defaults to `http://localhost:30000/api/v1/storage/capacity` (sensor-ms port → 404 → "insufficient disk capacity").

---

## Step 7 — `sop/vst/minio/minio-compose.yaml`

See `./configs/vios/minio-server.service.yml`. Key wiring:

- **Image / network:** `${MINIO_IMAGE}` running as `user: "0:0"` on `network_mode: host`. Standalone profile `[minio]` (separate from `bp_sop_2d` — MinIO is opt-in).
- **Command:** `server /data --address "${MINIO_BIND_IP}:${MINIO_API_PORT}" --console-address "${MINIO_BIND_IP}:${MINIO_CONSOLE_PORT}"`.
- **Storage:** single bind mount `${MINIO_DATA_PATH}:/data`.
- **Auth + cleanup:** `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`, plus the three stale-upload cleanup intervals (`MINIO_API_DELETE_CLEANUP_INTERVAL`, `MINIO_API_STALE_UPLOADS_CLEANUP_INTERVAL`, `MINIO_API_STALE_UPLOADS_EXPIRY`) all sourced from the SOP `.env` (Step 4e).
- **Healthcheck:** `curl -f http://${MINIO_BIND_IP}:${MINIO_API_PORT}/minio/health/live`, 30s interval, 3 retries, 40s start period.

> MinIO uses its own `minio` profile (not `bp_sop_2d`) — must be explicitly activated.

---

## Step 8 — Config Files

### 8a. `configs/adaptor_config.json`

Remove `media_adaptor_lib_path` from the `vst_rtsp` adaptor section.

### 8b. `configs/rtsp_streams.json`

| Field | Upstream → SOP |
|---|---|
| `enabled` | `false` → `true` |
| `max_stream_count` | `4` → `100` |

### 8c. `configs/vst_storage.json`

| Field | Note |
|---|---|
| `total_video_storage_size_MB` | Keep `100000` (100 GB). **Must NOT exceed available disk** — recorder rejects with "insufficient disk capacity" |

### 8d. `configs/postgresql.conf`

Copy verbatim from the upstream repo at `video-search-and-summarization/deployments/vst/developer/vst/configs/postgresql.conf`. This file is NOT in the skill's `references/deployments/vst/` directory (only the SDR `docker_cluster_config.json` files are there), so it must be copied from upstream during build — it is NOT created by `copy_references.sh` or `create_dirs.sh`.

> **Important:** `create_dirs.sh` creates `deployments/vst/sop/vst/configs/` as an empty directory. If you then run Docker Compose without `postgresql.conf` present as a file, Docker's bind-mount creates it as an **empty directory** instead of a file. `centralizedb-sop` will then fail to start with `input in flex scanner failed at file "/etc/postgresql/postgresql.conf" line 1` (PostgreSQL cannot parse a directory). Fix: remove the directory (`rm -rf deployments/vst/sop/vst/configs/postgresql.conf`) and copy the real file from upstream before starting the stack.

### 8e. `vst_config.json`, `vst_config_redis.json`, `vst_config_kafka.json` — common changes

| Field | Upstream → SOP |
|---|---|
| `max_webrtc_out_connections` | `40` → `8` |
| `max_webrtc_in_connections` | `1` → `8` |
| `rtp_udp_port_range` | (absent) → `"31000-31100"` |
| `websocket_keep_alive_ms` | `5000` then `"ai_bridge_endpoint": ""` → `5000` (drop `ai_bridge_endpoint`) |
| `nv_streamer_loop_playback` | `false` → `true` |
| `nv_streamer_sync_playback` | (absent) → `false` |
| `nv_streamer_sync_file_count` | (absent) → `0` |
| `webrtc_out_default_resolution` | `"1920x1080"` → **remove** |
| `use_webrtc_hw_dec` | (absent) → `true` (in `vst_config.json`, `vst_config_redis.json`); `false` (in `vst_config_kafka.json`) |
| `download_files_timeout_secs` | `120` → `300` |
| `qos_logfile_path` | `"./webroot/log/"` → `""` |
| Halo safety block | 10 lines (`halo_safety_*`) → **remove entirely** |
| `rtsp_server_instances_count` | (upstream value) → **`1`** |

**`rtsp_server_instances_count` must be `1`**: With multiple instances the load balancer assigns streams to ports 30554, 30555, 30556, … in round-robin. recorder-ms always uses the base port (`rtsp_server_port: 30554`) from `vst_config.json` — if a stream lands on port 30556, recorder-ms connects to the wrong port and recording fails silently. Setting `rtsp_server_instances_count: 1` ensures all streams stay on port 30554.

`vst_config.json` only: `webrtc_in_video_degradation_preference`: `"resolution"` → `"detail"`.

`vst_config_redis.json` only: add observability:

```json
"observability": {
    "enable_telemetry": false,
    "otlp_endpoint": "http://localhost:4318/v1/traces"
}
```

---

## Step 9 — Remove Upstream-Only Files

| File | Reason |
|---|---|
| `scripts/user_additional_install.sh` | not used in SOP layout |
| `configs/nginx-mms.conf` + `.template` | MMS mode unused |
| `configs/nginx-vst.conf.template` | template rendering removed |
| `configs/nginx-vst.conf` | replaced by static `nginx.conf` |
| `sdr-streamprocessing/sdr-config/data_wl.yaml` | unused |

---

## Verification

Run `./scripts/vios/verify.sh` (a thin wrapper that delegates to `scripts/verify_build.py --component vios`, the single source of truth) to inspect:

1. SOP directory structure (`sop/vst/configs/` and the 4 SDR dirs)
2. `bp_sop_2d` profile coverage
3. `container_name:` entries in `docker-compose.yaml` and SDR composes
4. Top-level `compose.yml` content
5. No upstream leftovers (`developer/`, `scripts/`)

---

## Troubleshooting

- **video clip URL `http://http://localhost:30888/storage/...` (double scheme)** — `VST_INGRESS_ENDPOINT` has `http://` prefix; storage-ms and replaystream-ms prepend another `http://`. Fix: `VST_INGRESS_ENDPOINT=localhost:${VST_INGRESS_HTTP_PORT:-30888}/vst` (no scheme, with `/vst` nginx prefix). See Step 4b.
- **UI page shows "Welcome to nginx!" on port 30888, or /vst/ returns 404** — The Nginx routing configuration is missing the general `/vst/` proxy to `sensor-ms` (which serves the Web UI on port 30000) and the root `/` redirect. Fix: Add `location = /` with a redirect to `/vst/` and `location /vst/` proxying to `sensor-ms` in `nginx.conf`, then reload Nginx (`docker exec vst-ingress-sop nginx -s reload`). See Step 5.
- **recorder-ms connects to wrong RTSP port (e.g. 30556 instead of 30554)** — `rtsp_server_instances_count > 1` causes the rtspserver-ms load balancer to assign streams to ports beyond 30554; recorder-ms always uses `rtsp_server_port: 30554`. Fix: set `rtsp_server_instances_count: 1` in all `vst_config*.json` files. See Step 8e.
- **recorder/replaystream/livestream SDRs never provision streams** — `WDM_WL_CHANGE_ID_ADD` must be `camera_streaming` for all three. If it is `camera_proxy`, the SDR waits for the rtspserver event instead of the streaming-ready event. See Step 6d.
- **recorder/livestream "stream not found" after sensor add** — `camera_streaming` Redis event is not published. `add-ds-sop-to-vst.py` must publish this event after sensor add and proxy URL confirmation. The reference script handles this — ensure you are using it, not the upstream version.
- **recording_status `"user"` not recognized as active** — this is a valid active recording state returned by recorder-ms for user-triggered recordings. Ensure test scripts and health checks treat `"user"` as equivalent to `"on"`/`"recording"`.
- **SDR `KeyError: 'process_type'` on startup** — each `sdr-config/docker_cluster_config.json` must include `"process_type": "docker"`. Without this field the SDR process model code raises a `KeyError` at start and cannot provision any streams. The reference files under `references/deployments/vst/sop/vst/sdr-*/sdr-config/docker_cluster_config.json` already contain this field and are copied verbatim by `copy_references.sh`. See Step 6b.
- **nginx `host not found in upstream "sensor-ms"` (or other service name)** — all VST containers use `network_mode: host`; Docker's internal DNS for Compose service names is unavailable. All `proxy_pass` directives in `configs/nginx.conf` must use `http://localhost:<PORT>/`, never a service name. See Step 5.
- **`vst-mcp-sop` pydantic `ValidationError: server_port field required`** — `MCP_GATEWAY_SERVER_HOST` and `MCP_GATEWAY_SERVER_PORT` are missing from `vst/sop/vst/.env`. Add both with values `${HOST_IP}` and `8001` respectively. See Step 4f.
- **`centralizedb-sop` won't start: `input in flex scanner failed at file "/etc/postgresql/postgresql.conf" line 1`** — `postgresql.conf` was created as an empty directory instead of a file. This happens when Docker Compose bind-mounts a path that does not yet exist as a file — Docker creates it as a directory. Fix: `rm -rf deployments/vst/sop/vst/configs/postgresql.conf` then copy the real file from upstream: `cp video-search-and-summarization/deployments/vst/developer/vst/configs/postgresql.conf deployments/vst/sop/vst/configs/postgresql.conf`. See Step 8d. Note: this file is NOT in the skill's `references/deployments/vst/` directory (it is not copied by `copy_references.sh`) — it must be sourced from upstream.
- **SDR fails to call rtspserver-ms/recorder-ms with `host='http', port=80` DNS error** — the `provisioning_address` in `docker_cluster_config.json` must NOT include the `http://` scheme prefix. Use `"provisioning_address": "localhost:<PORT>"` (not `"http://localhost:<PORT>"`). The SDR binary prepends `http://` itself; if the config already includes `http://`, the URL becomes `http://http://localhost/...` and DNS resolution fails. Reference files already use the correct format; verify deployed `sdr-*/sdr-config/docker_cluster_config.json` files match.
- **VIOS recording never starts despite SDR running (`camera_streaming` goes to wrong SDR)** — all 4 SDR services share consumer group `consumer-grp-id-3` (hardcoded in the binary), so Redis `vst.event` messages are round-robined across all SDRs. A `camera_proxy` event needed by rtspserver-SDR may arrive at recorder-SDR instead (which ignores it). Similarly, `camera_streaming` may miss the recorder-SDR. Workaround: call rtspserver-ms `POST /api/v1/proxy/stream/add` directly with `{"url": "<rtsp_url>", "id": "<sensor_uuid>", "name": "<name>"}` — rtspserver-ms will publish `camera_streaming` automatically once SDP is ready. Then recorder-ms is provisioned via the always_recording flag in its configuration. The `add-ds-sop-to-vst.py` script includes a `get_proxied_url_with_retry` path that handles this fallback.

---

## License

Use of this skill is governed by the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en) and the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
