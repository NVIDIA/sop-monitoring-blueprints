---
name: vss-sop-build
description: >-
  Build a custom VSS SOP blueprint from the VSS 3.1 base, then deploy and test
  in a loop until fully operational. Use when asked to create the SOP blueprint
  structure, customize VSS compose for SOP, configure SOP services, set up the
  VSS agent for SOP, or scaffold the SOP app layer on top of met-blueprints 3.1.
owner: NVIDIA
service: vss-sop
version: 1.0.0
license: CC-BY-4.0 AND Apache-2.0
reviewed: 2026-06-23
metadata:
  openclaw: { "emoji": "🔧", "os": ["linux"] }
  author: "Quan Vu <qvu@nvidia.com>"
  tags: ["vss", "sop", "build", "blueprint"]
---

# VSS SOP — Build a Custom Blueprint from VSS 3.1

Scaffold the SOP blueprint layer on top of `met-blueprints` 3.1. **This SKILL.md is a roadmap** — it points to detailed references for each stage. Read references only when working on that stage.

## Overview

Use this skill when asked to create, customize, or scaffold the VSS SOP blueprint on top of standard VSS 3.1 base components.

Key scenarios:
- Creating a new SOP blueprint from scratch.
- Merging upstream VSS changes with custom SOP layers.
- Integrating custom DS-SOP deepstream microservices.

## Prerequisites

- **Host Base System:** standard Linux OS, Docker with buildkit, Nvidia GPU drivers 580+.
- **Upstream Codebase:** Access to VSS 3.1 repository.

## Instructions

Run these scripts from the blueprint repo root, in order:

0. **Prerequisites Check & Auto-Install (run first)** — `./agentic/vss-sop-skills/vss-sop-deploy/scripts/preflight_check.sh --bp-repo . --fix` (see [Stage 0 — Prerequisites Check & Auto-Install](#stage-0--prerequisites-check--auto-install))
1. **Clone upstream repository (Stage 0.1)** — `./agentic/vss-sop-skills/vss-sop-build/scripts/clone_and_prepare.sh`
2. **Copy all reference files verbatim (Stage 1)** — `./agentic/vss-sop-skills/vss-sop-build/scripts/copy_references.sh`
3. **Copy foundational from upstream (Stage 3.1a)** — `./agentic/vss-sop-skills/vss-sop-build/scripts/copy_foundational_from_upstream.sh`
4. **Modify foundational for SOP profile (Stage 3.1b)** — `./agentic/vss-sop-skills/vss-sop-build/scripts/modify_foundational_for_sop.sh`
5. **Copy VIOS from upstream (Stage 3.4a)** — `./agentic/vss-sop-skills/vss-sop-build/scripts/copy_vios_from_upstream.sh`
6. **Modify VIOS for SOP profile (Stage 3.4b)** — `./agentic/vss-sop-skills/vss-sop-build/scripts/modify_vios_for_sop.sh`
7. **Copy agents from upstream + apply SOP profile (Stage 2)** — `./agentic/vss-sop-skills/vss-sop-build/scripts/copy_agents_from_upstream.sh`
8. **Copy NIM from upstream + apply SOP conventions (Stage 3.3)** — `./agentic/vss-sop-skills/vss-sop-build/scripts/copy_nim_from_upstream.sh`
9. **Generate DS-SOP source via ds-sop-skills (RTSP output is native), then build the image** (when needed) — first generate `../ds_sop_microservice` with the ds-sop-skills skill (`../ds-sop-skills/example_sop_prompt.md`), then run `./agentic/vss-sop-skills/vss-sop-build/scripts/build_ds_sop.sh <ds_sop_microservice_dir> <bp_repo_path>`
10. **Verify the build** — `./agentic/vss-sop-skills/vss-sop-build/scripts/verify_build.sh`

## Cross-Reference with Other Skills

| Need | Skill |
|---|---|
| Prerequisites, build DS-SOP image, models, deploy, RTSP test | **vss-sop-deploy** (Phase 0–5) |
| Post-deployment validation: health, ELK, VIOS, Agent | **vss-sop-test** (Phase 1–4) |
| Manage cameras/sensors/streams via VIOS | [`vss-call-vios-api`](references/vss/vss-call-vios-api/SKILL.md) |
| Query SOP data from Elasticsearch via VA-MCP | [`vss-query-analytics`](references/vss/vss-query-analytics/SKILL.md) |
| Generate SOP compliance reports | [`vss-generate-video-report`](references/vss/vss-generate-video-report/SKILL.md) |
| Search video archives with natural language | [`vss-search-archive`](references/vss/vss-search-archive/SKILL.md) |
| Summarize long videos / generate shift reports | [`vss-summarize-video`](references/vss/vss-summarize-video/SKILL.md) |

---

## Examples

### Run the Scaffolding Process (End-to-End)

```bash
# Prerequisites check & auto-install (run first)
./agentic/vss-sop-skills/vss-sop-deploy/scripts/preflight_check.sh --bp-repo . --fix

# Clone and prepare directories
./agentic/vss-sop-skills/vss-sop-build/scripts/clone_and_prepare.sh

# Copy references and foundations
./agentic/vss-sop-skills/vss-sop-build/scripts/copy_references.sh
./agentic/vss-sop-skills/vss-sop-build/scripts/copy_foundational_from_upstream.sh

# Modify foundational components for the SOP profile
./agentic/vss-sop-skills/vss-sop-build/scripts/modify_foundational_for_sop.sh
```

### Verify Scaffolding Integrity

```bash
# Execute local linter and integrity verification script
./agentic/vss-sop-skills/vss-sop-build/scripts/verify_build.sh
```

## What Changes from Upstream 3.1.0

The upstream `3.1.0` branch (`video-search-and-summarization/deployments/`) ships services for many blueprint variants. SOP keeps a slimmer subset and adds the `sop/` app layer.

| Category | What to do | Reference |
|---|---|---|
| Top-level compose | Strip `vlm-as-verifier`, `lvs`, `developer-workflow`, `proxy`; switch VST to `./vst/compose.yml`; add `./sop/compose.yml` | Stage 1 |
| Foundational | Add `bp_sop_2d` profiles, stock ES image, SOP Kafka topics + ES templates | [`foundational-building.md`](references/foundational-building.md) |
| VST | Replace `vst/developer/` with `vst/compose.yml` + `vst/sop/`; split monolith into 5 microservices; add MinIO | [`vios-building.md`](references/vios-building.md) |
| Agent | Add `bp_sop_2d` profile to `vss-va-mcp`, `vss-agent`, `vss-ui`; add SOP patch volume mounts | [`vss-agent-building.md`](references/vss-agent-building.md) |
| DS | Add `ds-sop/` subdirectory under `deployments/ds/` with compose + `.env` | [`ds-sop-building.md`](references/ds-sop-building.md) |
| New `sop/` directory | Entire SOP app layer: compose, .env, vss-agent configs/patches/templates, sop-app services | [`sop-app-building.md`](references/sop-app-building.md) |
| NIM | Per-GPU profiles, rename env files, tune NIM params | [`nim-building.md`](references/nim-building.md) |
| Remove | Delete `developer-workflow/`, `vlm-as-verifier/`, `lvs/`, `proxy/` from `deployments/` | — |

---

## Architecture Overview

### System Data Flow

```
Input Sources    Perception                      Messaging           Storage & Analytics
─────────────    ──────────                      ─────────           ───────────────────
 Video file  ─┐
 Basler cam  ─┼──► DS SOP ──► Metadata ──► Message Broker (Kafka) ──► Video Analytics DB
 RTSP stream ─┘        │                                                 (Elasticsearch)
                       │                                                        │
              RTSP output ──► VMS (VIOS/VST) ──► Video IO & Storage             │
                                                   │                            │
                                                   ▼                            ▼
                                          ┌─────────────────────────────────────────┐
                                          │              VSS Agent                  │
                                          │  Video IO │ Report Gen │ Search         │
                                          │       LLM (Nemotron Nano v2)            │
                                          │       VLM (Cosmos Reason 2)             │
                                          └─────────────┬───────────────────────────┘
                                                        ▼
                                                      User
```

### DS SOP Internal Pipeline

```
streaming ──► Decode ──► Create Video Chunks ──fix chunks──► DDM-Net (CV) ──action chunks──► Cosmos Reason (VLM) ──► Metadata
```

Metadata is published to Kafka (`mdx-vlm-captions`), ingested by Elasticsearch, and queried by the VSS Agent.

### Component → Service Mapping (key bindings)

| Component | Service / Image | Where configured |
|---|---|---|
| Agent | `vss-agent` + `vss-va-mcp` | `agents/vss-agent/`, `sop/vss-agent/` (Stage 2) |
| Video IO & Storage / VMS | VST 3.1.0 services | `vst/compose.yml` (Stage 3.4) |
| Search / Report Gen | Elasticsearch + `template_report_gen` | `sop/vss-agent/configs/config.yml` (Stage 2) |
| Nemotron Super (LLM) | Remote NIM endpoint (or local NIM) | `LLM_BASE_URL`, `LLM_NAME` in `sop/.env` (default remote Llama 3.3 Nemotron Super, customizable) |
| DS SOP Model (VLM) | Local — inside DS-SOP (or remote VLM) | `VLM_BASE_URL`, `VLM_NAME` in `sop/.env` (default local DS-SOP model, customizable) |
| DS SOP | `ds-sop:1.0.0` (local build) | `ds/ds-sop/` (Stage 3.6) |
| Message Broker | Kafka | `foundational/`, topic `mdx-vlm-captions` |
| Analytics DB | Elasticsearch | `foundational/`, index `mdx-vlm-captions-*` |

### Pinned Container Versions

| Component | Version |
|---|---|
| Elasticsearch / Kibana / Logstash | 9.3.0 (stock) |
| VSS Agent / VST / Video Analytics API / Agent UI | 3.1.0 |
| DS-SOP | 1.0.0 (local build) |

### Directory Structure

During build, all directories coexist under the current working directory (CWD). The upstream `video-search-and-summarization/` is removed after success.

```
<current working directory>/
├── deployments/                    ← starts empty; populated by Stages 0–4
│   ├── compose.yml                 ← top-level (Stage 1)
│   ├── cleanup_all_datalog.sh      ← cleanup data script
│   ├── foundational/               ← ELK 9.3.0, Kafka, Redis, Phoenix (Stage 3.1)
│   ├── vst/                        ← VST 3.1.0 microservices (Stage 3.4)
│   ├── agents/                     ← +bp_sop_2d profiles (Stage 2.1)
│   ├── nim/                        ← per-GPU profiles (Stage 3.3)
│   ├── monitoring/                 ← NEW: Prometheus/Grafana (Stage 3.2)
│   ├── ds/ds-sop/                  ← NEW: DS-SOP microservice (Stage 3.6)
│   └── sop/                        ← NEW: full SOP app layer (Stage 4)
├── sop-resources/                  ← NEW: resources downloaded in CWD (sample video, configs)
├── sop-data/                       ← NEW: datalog folders created in CWD (ES data, logs, Kafka, etc.)
└── video-search-and-summarization/ ← upstream (cloned to CWD, read-only, removed after build)
```

---

# Stage 0 — Prerequisites Check & Auto-Install

Run this **first**, before cloning or any build stage. It checks secret key files, GPU drivers, Docker, NVIDIA Container Toolkit, and NGC configuration, and can automatically install or configure missing prerequisites (NVIDIA Driver 580, CUDA Toolkit 13, Docker, Docker Compose, NVIDIA Container Toolkit, NGC CLI, and NGC CLI configuration).

To run the checks and automatically fix/install any missing components:

```bash
cd <bp-repo>
./agentic/vss-sop-skills/vss-sop-deploy/scripts/preflight_check.sh --bp-repo . --fix
```

To run the checks only, without modifying your system:

```bash
cd <bp-repo>
./agentic/vss-sop-skills/vss-sop-deploy/scripts/preflight_check.sh --bp-repo .
```

If any manual check fails, refer to the printed advice or the **vss-sop-deploy** Phase 0 references (secret keys, NVIDIA driver, Docker & Toolkit).

---

# Stage 0.1 — Clone and Prepare

All output goes into `deployments/`. Upstream `video-search-and-summarization/` is read-only and is cloned directly into the CWD. All paths are relative to CWD.

## Step 0.1.1 — Clone Upstream Repository
Run `./agentic/vss-sop-skills/vss-sop-build/scripts/clone_and_prepare.sh` to clone the upstream 3.1.0 branch of `video-search-and-summarization` (which serves as a read-only source for the build process).

---

# Stage 1 — Reference → Target File Mapping

Copy these files **verbatim** from `references/` (do NOT regenerate — they contain tuned defaults):

| Reference file | Copy to |
|---|---|
| `references/deployments/compose.yml` | `deployments/compose.yml` |
| `references/deployments/cleanup_all_datalog.sh` | `deployments/cleanup_all_datalog.sh` |
| `references/deployments/ds/compose.yml` | `deployments/ds/compose.yml` |
| `references/deployments/ds/ds-sop/.env` | `deployments/ds/ds-sop/.env` |
| `references/deployments/ds/ds-sop/ds-sop-docker-compose.yml` | `deployments/ds/ds-sop/ds-sop-docker-compose.yml` |
| `references/deployments/sop/.env` | `deployments/sop/.env` |

These references are all copied to their correct locations by running the unified copy script:
`./agentic/vss-sop-skills/vss-sop-build/scripts/copy_references.sh`

---

# Stage 2 — Convert VSS Agent → VSS SOP Agent

> Full details (YAML diffs, profiles, depends_on, agent-eval volume removal): [`references/vss-agent-building.md`](references/vss-agent-building.md)
> Full details (config.yml, va_mcp_server_config.yml, patches, templates): [`references/sop-app-building.md`](references/sop-app-building.md)

Transforms the upstream VSS Agent into the VSS SOP Agent (SOP compliance monitoring). **No image rebuilds** — done via configs, bind-mounts, env vars.

| # | Step | Action |
|---|---|---|
| 2.1 | Add `bp_sop_2d` profile + patch volumes | Add profile to `vss-ui`, `vss-va-mcp`, `vss-agent`, all foundational services. Bind-mount SOP patches into `vss-va-mcp`. Comment out `ai-agents` in `agents/compose.yml` |
| 2.2 | Configure VSS Agent for SOP | Create `sop/vss-agent/configs/config.yml` (workflow, SOP tools, prompt) and `va_mcp_server_config.yml` (SOP tools list) |
| 2.3 | Create Python patches | `sop/vss-agent/patches/{es_client.py, tools.py, utils.py}` — overlay video-analytics module |
| 2.4 | Create report templates | `sop/vss-agent/templates/{sop_compliance_report_template.md}` |
| 2.5 | Set agent `.env` vars | `VSS_AGENT_CONFIG_FILE`, `VSS_VA_MCP_CONFIG_FILE`, `VSS_AGENT_TEMPLATE_*`, `VLM_MODE=local`, `VLM_BASE_URL=http://localhost:8300`, `VLM_NAME=ds_sop_model` |

### Key SOP-specific tools (configured via `va_mcp_server_config.yml`)

- `get_sop_status` — current compliance status, latest actions, cycle info
- `get_sop_report` — full compliance report with violations, cycles, statistics
- `get_sop_as_incidents` / `get_sop_as_incident` — SOP events as incidents

### Patch mount target (verify Python version matches the VSS Agent image)

```
/vss-agent/.venv/lib/python3.13/site-packages/vss_agents/video_analytics/{tools,utils,es_client}.py
```

---

# Stage 3 — Build Foundational, Monitoring, NIM, VIOS & DS-SOP Layer

| # | Step | Reference |
|---|---|---|
| 3.1a | Copy foundational from upstream to deployments | `scripts/copy_foundational_from_upstream.sh` |
| 3.1b | Modify foundational for SOP: stock ES, Kafka topics, vlm-captions, init scripts | `scripts/modify_foundational_for_sop.sh` + [`foundational-building.md`](references/foundational-building.md) |
| 3.2 | Create monitoring stack (Prometheus, Grafana, DCGM, node-exporter, cAdvisor) | *(not yet documented)* |
| 3.3 | Convert NIM model services (per-GPU profiles, rename envs, tune params) | [`nim-building.md`](references/nim-building.md) |
| 3.4a | Copy VIOS from upstream to deployments (restructure `developer/vst/` → `sop/vst/`) | `scripts/copy_vios_from_upstream.sh` |
| 3.4b | Modify VIOS for SOP: split monolith → 5 microservices + MinIO | `scripts/modify_vios_for_sop.sh` + [`vios-building.md`](references/vios-building.md) |
| 3.5 | Wire SOP compose | See below |
| 3.6 | Configure DS-SOP | Copy reference files verbatim (see below) |

> DS-SOP source generation (ds-sop-skills, RTSP output native per § 18) + image build (via `build_ds_sop.sh`, no patch): see the Generate + Build section below + [`ds-sop-building.md`](references/ds-sop-building.md).

## Step 3.5 — Wire the SOP Compose

The compose files are wired here; both are kept in `configs/`:

- **`deployments/ds/compose.yml` (REPLACES upstream content)** — see `references/deployments/ds/compose.yml`. Single include: `ds-sop/ds-sop-docker-compose.yml`.

## Step 3.6 — Configure the DS-SOP Microservice

Copy both files verbatim from references — they encode tuned defaults, exact volume mounts, and env-var pass-through. This is done by the copy references script (`./agentic/vss-sop-skills/vss-sop-build/scripts/copy_references.sh`), which copies:

- `agentic/vss-sop-skills/vss-sop-build/references/deployments/ds/ds-sop/.env` → `deployments/ds/ds-sop/.env`
- `agentic/vss-sop-skills/vss-sop-build/references/deployments/ds/ds-sop/ds-sop-docker-compose.yml` → `deployments/ds/ds-sop/ds-sop-docker-compose.yml`

Critical defaults to be aware of:

| Variable | Default | Note |
|---|---|---|
| `API_SERVER_PORT` | 8300 | DS-SOP API server (also `VLM_BASE_URL` target) |
| `DS_SOP_KAFKA_TOPIC` | `mdx-vlm-captions` | Must match foundational Kafka topic |
| `SOP_MESSAGING_SCHEMA` | `JSON` | **Required for ELK pipeline** (not `NvProtoSchema`) |
| `ENABLE_MESSAGING` | `1` | Enable chunk → Kafka publication |
| `MODEL_ROOT_DIR` | `/opt/models` | Mounts `${MODEL_ROOT_DIR}:${MODEL_ROOT_DIR}` |
| `VLLM_MODEL_PATH` / `DDM_MODEL_PATH` | per `.env` | Models verified under `/opt/models` (vss-sop-deploy Phase 1) |
| `SW_ENCODER` | `true` | CPU `x264enc`; set `false` for GPU `nvv4l2h264enc` |

Container properties: image `ds-sop:1.0.0`, profile `bp_sop_2d`, runtime `nvidia`, `shm_size: 16gb`, `network_mode: host`, depends on `kafka`.


---

## Generate + Build: ds-sop-skills source → DS-SOP

> Full breakdown (build flow, 8 post-build checks, auto-fix recipes; RTSP is an opt-in ds-sop-skills feature, requested at generation — § 18): [`references/ds-sop-building.md`](references/ds-sop-building.md)
> Build script: [`scripts/build_ds_sop.sh`](scripts/build_ds_sop.sh)

The DS-SOP source is **not cloned** from the public NVIDIA SOP Inference Blueprint. It is **generated by the ds-sop-skills skill** into `ds_sop_microservice` and built as-is into the VSS blueprint's `ds-sop:1.0.0` image. RTSP streaming output is an **optional, opt-in** ds-sop-skills feature (§ 18) that the VSS-SOP blueprint requires, so the generation prompt must request it explicitly.

**Step A — Generate the source** (if `ds_sop_microservice` does not already exist) by following the ds-sop-skills prompt — note the **required** *"with rtsp streaming output feature"* phrase:

```
Please follow instructions in ../ds-sop-skills/example_sop_prompt.md to generate a SOP microservice with rtsp streaming output feature in folder ../ds_sop_microservice
```

This emits the standard DeepStream SOP microservice (DeepStream GEBD pipeline, VLM classification, SOP checker, Kafka/JSON messaging, Basler camera support) **with RTSP streaming output built in** (ds-sop-skills § 18). Omitting the RTSP phrase produces a source without RTSP and the RTSP post-build checks will fail.

**Step B — Build:** Run `./agentic/vss-sop-skills/vss-sop-build/scripts/build_ds_sop.sh <ds_sop_microservice_dir> <bp_repo_path>` (`<ds_sop_microservice_dir>` defaults to `../ds_sop_microservice`). It validates the generated layout, ensures the `binaries/` bind-mount target exists, then builds `ds-sop:1.0.0`. **No source modification.**

Notes:

- **RTSP is owned by ds-sop-skills (§ 18)** — it is an **opt-in** ds-sop-skills feature, so the generation prompt must request it (*"with rtsp streaming output feature"*). When requested, the generated source contains the GStreamer RTSP libs + codec plugins, `RTSPStreamingServer` + `tee1`-tapped encode branch, `SW_ENCODER`/`ENABLE_RTSP_OUTPUT`/`RTSP_PORT`, `--rtsp-port` CLI, api-server auto-injection, `/ds-out/{sensor_id}` paths, and `key-int-max=30`. vss-sop-build does not add or patch any of it.
- **Image vs. workdir** — the image is tagged `ds-sop:1.0.0` at build time, but the container workdir stays `/opt/nvidia/nvds_sop` (as generated). The VSS `ds-sop-docker-compose.yml` points `PYTHONPATH`, `working_dir`, and the cache/chunks/alert mounts at `/opt/nvidia/nvds_sop` accordingly.
- If a post-build check (§ checks in `ds-sop-building.md`) fails, fix it in the **ds-sop-skills generation** and regenerate — not by patching here.

---

# Stage 4 — Add SOP Infrastructure & App Layer

> Full details: [`references/sop-app-building.md`](references/sop-app-building.md)

The entire `sop/` directory is **new** — not in the upstream repo. This is copied as part of the copy references script (`./agentic/vss-sop-skills/vss-sop-build/scripts/copy_references.sh`) from the skill's reference directory `agentic/vss-sop-skills/vss-sop-build/references/deployments` (including the top-level `compose.yml`, `cleanup_all_datalog.sh`, and the full `sop/` directory with its `.env` file) into the target `deployments/` folder.

Then customize `deployments/sop/.env`:
1. Set `MDX_SAMPLE_APPS_DIR`, `MDX_DATA_DIR`, `HOST_IP`, API keys (`NGC_CLI_API_KEY`, `NVIDIA_API_KEY`), `HARDWARE_PROFILE`.
2. Customize the LLM and VLM settings if needed (defaults to Llama 3.3 Nemotron Super for remote LLM and `ds_sop_model` for local DS-SOP VLM):
   - **LLM Configuration (remote or local NIM)**: `LLM_MODE`, `LLM_BASE_URL`, `LLM_NAME` / `LLM_NAME_SLUG`.
   - **VLM Configuration (local or remote)**: `VLM_MODE`, `VLM_BASE_URL`, `VLM_NAME` / `VLM_NAME_SLUG`.

All other files are used as-is.

---

# Stage 5 — Build Verification + Deploy → Test Loop

## Step 5.0 — Verify the build

Run `./agentic/vss-sop-skills/vss-sop-build/scripts/verify_build.sh` — it invokes `verify_build.py` to check directory structure, top-level compose include, `bp_sop_2d` profile coverage, container image versions, Kibana dashboard flat-field mapping, SDR recorder API URLs, and SDR container cluster name syntax (checks for invalid double-double-quotes), then runs the post-verification cleanup of the upstream repo.

## Step 5.1 — Deploy (invoke `vss-sop-deploy`)

Phase 0: prerequisites & API keys → Phase 1: verify models & download assets → Phase 2: start RTSP simulator → Phase 3: build DS-SOP image & verify components → Phase 4: configure & deploy (`docker compose … --profile bp_sop_2d up -d`) → Phase 5: test RTSP stream & register to VIOS.

## Step 5.2 — Test (invoke `vss-sop-test`)

Phase 1: service health → Phase 2: ELK pipeline (ES reachable, data ingested, Kibana fields) → Phase 3: VIOS recording + livestream → Phase 4: VSS Agent (MCP, LLM, VLM, snapshot, video, report).

## Step 5.3 — Fix and Re-deploy (loop)

If a phase fails: read auto-debug output → apply fix (env, compose, patches, image rebuild) → re-deploy → re-test. Repeat until ALL phases pass with 100% check pass rate.

**MANDATORY: ALL tests must pass. Do NOT classify any failure as "environmental", "intermittent", or "acceptable". Do NOT proceed with any failing test. If stuck after 3 attempts on the same failure, STOP and ask the user for help.**

The pipeline is complete ONLY when the output matches:

```
✅ Phase 1 — Service Health: PASS (all checks)
✅ Phase 2 — ELK Data Pipeline: PASS (all checks)
✅ Phase 3 — VIOS Recording & Livestream: PASS (all checks, including recording)
✅ Phase 4 — VSS Agent: PASS (all checks, including report generation)
```

Any partial result (e.g., 36/38) means the pipeline is NOT complete — keep debugging.

---

## Error Handling

Build / structure issues (resolved during the build stages above):

- **Missing `bp_sop_2d` profile** — must be added manually to ALL shared services in `foundational/`, `agents/`, `vst/`. Upstream 3.1.0 doesn't include it.
- **Patch mount failures** — verify the Python version (`python3.13`) matches the VSS Agent image; mount target is `/vss-agent/.venv/lib/python3.13/site-packages/vss_agents/video_analytics/`.
- **`vss-agent` and `vss-va-mcp` containers never start with `bp_sop_2d` profile** — both services in `agents/vss-agent/vss-agent-docker-compose.yml` must have `bp_sop_2d` in their `profiles:` list. The `patch_profiles.py` script handles this automatically. Manual fallback: add `- bp_sop_2d` as the first entry in the `profiles:` list for both `vss-va-mcp` and `vss-agent`. Also add the SOP patch volume mounts (`tools.py`, `utils.py`, `es_client.py`) to both services so they both use the SOP-specific video analytics module.
- **Video analytics API empty** — `vss-video-analytics-api-config.json` must point to `mdx-vlm-captions-*` (not `mdx-incidents-*`).
- **Upstream includes fail** — clean up top-level `compose.yml` per Stage 1 (drop `developer-workflow`, etc.).
- **`sop/.env` not found** — must be at `deployments/sop/.env` (Docker Compose reads it from beside `sop/compose.yml`). Copied during Stage 4.
- **DS-SOP container won't start** — confirm `ds-sop:1.0.0` image exists; verify `VLLM_MODEL_PATH` / `DDM_MODEL_PATH` point to verified models.
- **Agent can't reach VLM** — SOP uses `VLM_BASE_URL=http://localhost:8300` (DS-SOP API), not a remote NIM endpoint.
- **Report generation fails with "error generating" / VST clip HTTP 500 ("Could not multiplex stream")** — the report agent extracts a short (~2s) incident video clip via `vst_video_url`. With `overlay_config: true`, VST's OSD/bbox transcode pipeline fails to multiplex such short clips (`VMSInternalError: Synchronous video generation failed`). SOP incidents carry no object bounding boxes (`objectId` is always empty), so set `overlay_config: false` for the `vst_video_url` tool in `sop/vss-agent/configs/config.yml` (leave the `vst_picture_url` snapshot tool as-is). The plain no-overlay clip path muxes reliably.
- **VSS agent [401] Unauthorized calling remote LLM** — `NVIDIA_API_KEY=''` in `deployments/sop/.env` is empty. `deploy.sh` now auto-writes the key from `.secret/nvidia_build_api_key.txt` into `sop/.env` on each run. If starting agent containers independently (not via `deploy.sh`), ensure `NVIDIA_API_KEY` is set in the shell or in `sop/.env` before running `docker compose up`.

Component-specific issues — the detailed symptoms and fixes live in the reference docs (single source of truth), so they don't drift out of sync with this skill:

- **Foundational / ELK pipeline** — Logstash `ConfigurationError` at `date` blocks, `mdx-vlm-captions` index never appears, `mdx-vlm-captions` doc count stuck at 1 (`document_id` branch), Kibana "No field found for [...]" / flat-JSON data view, `SOP_MESSAGING_SCHEMA`/`ENABLE_MESSAGING`/Kafka topic issues → see [`foundational-building.md`](references/foundational-building.md) Troubleshooting (validated by `verify_build.py --component foundational`).
- **VIOS / VST recording & streaming** — double-scheme clip URLs (`http://http://...`), recorder on wrong RTSP port (30556 vs 30554), "Stream not present in recorder", `camera_streaming` not published, SDR `KeyError: 'process_type'`, SDR `provisioning_address` scheme error, SDR consumer-group round-robin, nginx `host not found in upstream`, `vst-mcp-sop` `server_port` `ValidationError`, recorder "insufficient disk capacity" → see [`vios-building.md`](references/vios-building.md) Troubleshooting (validated by `verify_build.py --component vios`).
- **DS-SOP image** — empty (623-byte) MP4 report clips / `key-int-max` keyframe interval, RTSP streaming output (ds-sop-skills § 18; opt-in at generation, built, not patched) → see [`ds-sop-building.md`](references/ds-sop-building.md).
- **VSS Agent compose** — `vss-agent` refuses to start with `depends on undefined service "nvidia-nemotron-nano-9b-v2-fp8"` (or `-fp8-shared-gpu`) → see [`vss-agent-building.md`](references/vss-agent-building.md) Step 2d.
- **Prerequisites & runtime issues** → see **vss-sop-deploy** Phase 0 and Troubleshooting.

---

## License

Use of this skill is governed by the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en) and the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
