# Create the SOP App Layer

> **Scope:** all files within `deployments/sop/`. The entire `sop/` directory is **new** — not in upstream VSS 3.1.0.
> For agent compose changes (`agents/compose.yml`, `agents/vss-agent/`), see [`vss-agent-building.md`](vss-agent-building.md).

## Step 0 — Copy Reference Code

Use the build orchestrator script `./agentic/vss-sop-skills/vss-sop-build/scripts/copy_references.sh` from the blueprint repository root to copy all customized deployments reference folders and files (including the entire `sop/` directory).

The reference code lives at `references/deployments/sop/` (relative to the skill root). It will be copied verbatim to `deployments/sop/` at the repository root.

### Resulting tree

```
deployments/sop/
├── compose.yml                                          ← top-level SOP compose
├── .env                                                 ← central config (MUST customize)
├── sop-app/
│   ├── compose.yml                                      ← Kibana init + Video Analytics API
│   ├── Dockerfiles/kibana-dashboard.Dockerfile         ← Alpine init container
│   ├── helper-scripts/
│   │   ├── add-ds-sop-to-vst.py                       ← VST sensor registration
│   │   ├── api_client_test.py                           ← DS-SOP test client
│   │   ├── run_rtsp_test.sh                             ← end-to-end wiring
│   │   └── rtsp_tools/{install.sh, rtsp_server.py}      ← GStreamer deps + RTSP loopback
│   ├── kibana-dashboard/
│   │   ├── sop-kibana-objects.ndjson                    ← saved objects
│   │   └── init-scripts/kibana-import-dashboard.sh
│   └── vss-video-analytics-api/configs/
│       └── vss-video-analytics-api-config.json          ← targets mdx-vlm-captions-*
└── vss-agent/
    ├── configs/{config.yml, va_mcp_server_config.yml}
    ├── patches/{es_client.py, tools.py, utils.py}        ← 234/1430/361 lines
    └── templates/{sop_compliance_report_template.md}
```

---

## Step 1 — Customize `sop/.env`

`.env` is the **only file requiring per-deployment changes**.

### 1a — Deployment paths

```env
MDX_SAMPLE_APPS_DIR="<absolute-path-to-your-deployments-dir>"
MDX_DATA_DIR="<absolute-path-to-your-data-dir>"
```

### 1b — Host IP

```env
HOST_IP='<your-actual-host-ip>'
```

### 1c — API keys

```env
NGC_CLI_API_KEY='<your-ngc-api-key>'      # local NIM images
NVIDIA_API_KEY='<your-nvidia-api-key>'    # remote LLM endpoints (build.nvidia.com)
```

### 1d — Hardware profile

```env
# Valid values: H100, L40S, RTX6000PROBW, DGX-THOR, DGX-SPARK
HARDWARE_PROFILE='H100'
```

### 1e — LLM and VLM Customization

Configure the LLM and VLM model names, slugs, and endpoints in `sop/.env`. By default, the LLM runs in remote mode using Llama 3.3 Nemotron Super, and the VLM runs in local mode using the built-in model.

#### Default Reference Settings:

```env
# LLM Configuration (Default: Remote Llama 3.3 Nemotron Super)
LLM_MODE=remote
LLM_BASE_URL='https://integrate.api.nvidia.com'
LLM_MODEL_TYPE=nim
LLM_NAME=nvidia/llama-3.3-nemotron-super-49b-v1.5
LLM_NAME_SLUG=llama-3.3-nemotron-super-49b-v1.5

# VLM Configuration (Default: Local DS-SOP Model)
VLM_MODE=local
VLM_BASE_URL='http://localhost:8300'
VLM_MODEL_TYPE=openai
VLM_NAME=ds_sop_model
VLM_NAME_SLUG=ds_sop_model
```

#### Customizing LLM:
- **Switch to Local NIM**: Set `LLM_MODE=local`, `LLM_BASE_URL='http://localhost:${LLM_PORT}'`, and `LLM_NAME` / `LLM_NAME_SLUG` to your local model name (e.g., `nvidia/nvidia-nemotron-nano-9b-v2`).
- **Use other remote LLMs**: Change `LLM_NAME` and `LLM_NAME_SLUG` to other supported endpoints (e.g., `openai/gpt-oss-20b`).

#### Customizing VLM:
- **Custom VLM Models**: You can customize `VLM_NAME` and `VLM_NAME_SLUG` to other models supported by your endpoint (e.g., `nvidia/cosmos-reason2-8b`, `nvidia/cosmos-reason1-7b` on build.nvidia.com, or `Qwen/Qwen3-VL-8B-Instruct` for internal NIMs).

### Variables that MUST NOT change

| Variable | Fixed Value | Reason |
|---|---|---|
| `VLM_MODE` | `local` | SOP uses DS-SOP's built-in vLLM |
| `VLM_BASE_URL` | `http://localhost:8300` | DS-SOP API server |
| `VLM_MODEL_TYPE` | `openai` | OpenAI-compatible API |
| `MODE` | `2d` | SOP is 2D-only |
| `BP_PROFILE` | `bp_sop` | SOP blueprint profile |
| `COMPOSE_PROFILES` | `${BP_PROFILE}_${MODE}` | computes to `bp_sop_2d` |
| `STREAM_TYPE` | `kafka` | SOP uses Kafka |
| `COMPOSE_PROJECT_NAME` | `mdx` | volume naming convention |
| `VSS_AGENT_CONFIG_FILE` | `./deployments/sop/vss-agent/configs/config.yml` | — |
| `VSS_VA_MCP_CONFIG_FILE` | `./deployments/sop/vss-agent/configs/va_mcp_server_config.yml` | — |

---

## File Reference

### `sop/compose.yml`

See `../configs/sop-compose.yml` (canonical copy at the skill root). Two includes: `./sop-app/compose.yml` and `../ds/ds-sop/ds-sop-docker-compose.yml`.

### `sop/sop-app/compose.yml` — services under `bp_sop_2d`

| Service | Image | Purpose | Depends on |
|---|---|---|---|
| `kibana-init-container-sop` | from `kibana-dashboard.Dockerfile` | Imports SOP Kibana dashboards | `kibana: service_healthy` |
| `vss-video-analytics-api-sop` | `nvcr.io/nvidia/vss-core/vss-video-analytics-api:3.1.0` | API with SOP config | `broker-health-check`, `elasticsearch-init-container` |

### `sop-app/vss-video-analytics-api/configs/vss-video-analytics-api-config.json`

`indexPrefix` and `rawIndex` target `mdx-vlm-captions-*` (JSON from DS-SOP via Kafka), not the standard `mdx-incidents-*`.

### `sop-app/Dockerfiles/kibana-dashboard.Dockerfile`

Alpine 3.21.3-based init container that copies the dashboard NDJSON + import script.

### `sop-app/kibana-dashboard/`

| File | Purpose |
|---|---|
| `sop-kibana-objects.ndjson` | Exported saved objects (dashboards, visualizations, index patterns) |
| `init-scripts/kibana-import-dashboard.sh` | Waits for ES + Kibana (10 retries each), imports via `curl -X POST /api/saved_objects/_import?overwrite=true` |

### `sop-app/helper-scripts/`

| File | Purpose |
|---|---|
| `rtsp_tools/install.sh` | GStreamer RTSP server deps (ffmpeg, PyGObject, GIR) |
| `rtsp_tools/rtsp_server.py` | RTSP server that loops a video file. Modes: **passthrough** (preserves codec/framerate) and **overlay** (decode + timestamp + re-encode) |
| `add-ds-sop-to-vst.py` | Registers DS-SOP RTSP output in VST. Recreates stale sensors when loopback IPs need real host IPs. Starts recording with retry |
| `api_client_test.py` | Test client for DS-SOP `/v1/chat/completions` (base64 video, RTSP, Basler) |
| `run_rtsp_test.sh` | End-to-end: rtsp_server.py → DS-SOP → register sensor → start recording |

### `vss-agent/configs/config.yml` (~366 lines)

| Section | Purpose |
|---|---|
| `general` | FastAPI frontend, CORS, Phoenix telemetry |
| `function_groups.video_analytics_mcp` | MCP client with SOP tools (`get_sop_status`, `get_sop_report`, `get_sop_as_incidents`, `get_sop_as_incident`) |
| `functions.video_understanding` | VLM video analysis via local DS-SOP |
| `functions.template_report_gen` | Report generator with SOP compliance template |
| `functions.report_agent` | Sub-agent: incident retrieval + template report gen |
| `llms` | LLM (remote NIM) + VLM (local DS-SOP at port 8300) |
| `workflow` | `top_agent` with SOP compliance persona, tool rules, report routing |
| `eval` | Trajectory evaluator |

### `vss-agent/configs/va_mcp_server_config.yml`

VA MCP server for the `vss-va-mcp` container. Includes SOP tools:
- `get_sop_status` — current compliance status, latest actions, cycle info
- `get_sop_report` — full report with violations, cycles, statistics
- `get_sop_as_incidents` / `get_sop_as_incident` — SOP events as incidents

### `vss-agent/patches/` — Python bind-mounts (~2000 lines)

Mounted into `vss-va-mcp` at `/vss-agent/.venv/lib/python3.13/site-packages/vss_agents/video_analytics/`:

| File | Lines | Purpose |
|---|---|---|
| `es_client.py` | 234 | Extended `ESClient` with `vision_llm_messages` index → `mdx-vlm-captions-*` |
| `tools.py` | 1430 | `GetSopStatus`, `GetSopReport`, `GetSopAsIncidents`, `GetSopAsIncident` |
| `utils.py` | 361 | Parse VisionLLM messages, cycle progression, violation detection |

### `vss-agent/templates/`

| File | Lines | Sections |
|---|---|---|
| `sop_compliance_report_template.md` | 65 | identifier, status, executive summary, sensor info, SOP observations, cycle analysis, recommendations, video analysis with snapshots |

---

## Verification

Run `./scripts/sop-app/verify.sh` from the skill root to confirm:

1. File count = 19 in `vss-sop/deployments/sop/`
2. `.env` has been customized (no `<PLACEHOLDER>` for `HOST_IP`)
3. Video Analytics API config targets `vlm-captions` (`mdx-vlm-captions-*` index)
4. Agent config includes SOP tools (`get_sop_*`)
5. Patch files (`vss-agent/patches/*.py`) at expected line counts
6. Diff against the reference tree — only `.env` should differ

---

## License

Use of this skill is governed by the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en) and the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
