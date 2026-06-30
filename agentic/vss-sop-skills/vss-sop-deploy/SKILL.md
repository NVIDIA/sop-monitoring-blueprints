---
name: vss-sop-deploy
description: >-
  Build the DS-SOP Docker image and deploy the VSS SOP blueprint end-to-end.
  Use when asked to deploy SOP, build DS SOP, install SOP, set up the SOP
  pipeline, verify SOP models, start the SOP blueprint, simulate RTSP for SOP,
  run the SOP API test, or tear down SOP.
owner: NVIDIA
service: vss-sop
version: 1.0.0
license: CC-BY-4.0 AND Apache-2.0
reviewed: 2026-06-23
metadata:
  openclaw: { "emoji": "🏗️", "os": ["linux"] }
  author: "Quan Vu <qvu@nvidia.com>"
  tags: ["vss", "sop", "deploy"]
---

# VSS SOP — Build DS-SOP & Deploy

Complete end-to-end skill: build the DS-SOP Docker image, verify models, configure the blueprint, deploy, test, and tear down using structured helper scripts.

## Overview

Use this skill when deploying or tearing down the VSS SOP blueprint, starting RTSP simulations, or validating prerequisites.

Key operations:
- Verifying host requirements and Docker setups.
- Downloading assets and checkpoints under `/opt`.
- Creating RTSP simulation streams.
- Configuring host environments and deploying compose services.

## Prerequisites

- **Hardware:** H100 / H200 / A100 GPU (>= 80 GB VRAM).
- **Driver:** NVIDIA Driver 580 with CUDA 13.
- **NGC Keys:** Registered with subscription access to Metropolis data bundles.

## Instructions

## Requirements

| Item | Value |
|---|---|
| CUDA | 13 |
| Driver | 580 |
| Hardware | H100 / H200 / A100 |
| GPU | >= 80 GB VRAM, >= 1 GPU |

Software: Docker, NVIDIA Container Toolkit 1.18.1, NVIDIA Driver 580, `ngc` CLI.

## Examples

### Run Deploy Setup

```bash
# Configure blueprint environment variables
./agentic/vss-sop-skills/vss-sop-deploy/scripts/configure_blueprint.sh --bp-repo . --vlm-mode local

# Start the full deployment stack
./agentic/vss-sop-skills/vss-sop-deploy/scripts/deploy.sh --bp-repo .
```

### Manage Simulation and Teardown

```bash
# Start simulation stream on rtsp://localhost:8552/sensor_0
./agentic/vss-sop-skills/vss-sop-deploy/scripts/start_rtsp_server.sh --bp-repo .

# Tear down the deployment and clean volumes
./agentic/vss-sop-skills/vss-sop-deploy/scripts/teardown.sh --bp-repo .
```

## Phase 0 — Prerequisites Check & Auto-Install

Run this phase before every deploy. It checks secret key files, GPU drivers, Docker, NVIDIA Container Toolkit, and NGC configuration, and can automatically install or configure missing prerequisites.

### Run Pre-flight Checks and Auto-Install

To run the checks and automatically fix/install any missing components (such as NVIDIA Driver 580, CUDA Toolkit 13, Docker, Docker Compose, NVIDIA Container Toolkit, NGC CLI, and NGC CLI configuration):

```bash
cd <bp-repo>
./agentic/vss-sop-skills/vss-sop-deploy/scripts/preflight_check.sh --bp-repo . --fix
```

If you only want to run the checks without modifying your system:

```bash
cd <bp-repo>
./agentic/vss-sop-skills/vss-sop-deploy/scripts/preflight_check.sh --bp-repo .
```

If any manual check fails, refer to the printed advice or the following documentation to resolve the issue:
- Secret keys: See [`references/ngc.md`](references/ngc.md) § Configure NGC API Key
- NVIDIA driver: See [`references/nvidia_driver.md`](references/nvidia_driver.md)
- Docker & Toolkit: See [`references/prerequisites.md`](references/prerequisites.md)

---

## Phase 1 — Verify Models & Download Assets

Verify that the trained model and config files exist under `/opt/models/...` and `/opt/sop/...` directories. For optimal accuracy, you must retrain/fine-tune the models, which can be done using the SOP Training Blueprint. After training, move model and config to `/opt/models/...` and `/opt/sop/...` directories. Download the sample video, re-encode it to H.264 at 30 FPS, and prepare cache/datalog directories.

### Run Asset Download and Preparation

To execute this entire pipeline:

```bash
cd <bp-repo>
./agentic/vss-sop-skills/vss-sop-deploy/scripts/download_assets.sh --bp-repo .
```

This automates the following actions:
- Checks that the required model and config files exist under `/opt/models` and `/opt/sop`. If not, prompts the user to retrain/fine-tune the models using the SOP Training Blueprint.
- Downloads the sample RTSP video.
- Installs `ffmpeg` (if missing) and re-encodes the sample video to H.264 at 30 FPS.
- Configures local cache permissions and pre-creates all bind-mounted data log directories (e.g., Kafka, Elasticsearch) with universal read/write permissions to prevent container startup failures.

---

## Phase 2 — Start RTSP Server Simulation

The RTSP stream simulator is started in this phase — ahead of the build, verification, and deployment steps that follow — so a live stream (`rtsp://localhost:8552/sensor_0`) is available for the end-to-end ingestion testing in later phases.

```bash
cd <bp-repo>
./agentic/vss-sop-skills/vss-sop-deploy/scripts/start_rtsp_server.sh --bp-repo .
```

This installs RTSP server prerequisites and starts `rtsp_server.py` serving on `rtsp://localhost:8552/sensor_0` in the background (which wraps raw MP4 footage and streams it in real time).

---

## Phase 3 — Build DS-SOP Docker Image & Verify Components

To build the `ds-sop:1.0.0` Docker image and verify its RTSP components, refer to the detailed reference documentation:

- See [`references/build_ds_sop_image.md`](references/build_ds_sop_image.md)

This reference covers:
1. Generating the DeepStream SOP source code with the ds-sop-skills skill (into `ds_sop_microservice`) — RTSP streaming output is opt-in (ds-sop-skills § 18) and must be requested in the generation prompt (*"with rtsp streaming output feature"*).
2. Building the container image with BuildKit support (no patch step — the generated source is built as-is).
3. Verifying the RTSP components (the 6 pre-flight container checks run by `verify_rtsp_components.py`).

---

## Phase 4 — Configure & Deploy the Blueprint

### Configure the Blueprint

Configure host IPs, deployment paths, and LLM/VLM endpoints in the `.env` file. With no extra flags this auto-detects HOST_IP/EXTERNAL_IP and defaults to remote NIM endpoints:

```bash
cd <bp-repo>
./agentic/vss-sop-skills/vss-sop-deploy/scripts/configure_blueprint.sh --bp-repo .
```

For local VLM (served by the DS-SOP container on port 8300):

```bash
./agentic/vss-sop-skills/vss-sop-deploy/scripts/configure_blueprint.sh --bp-repo . --vlm-mode local
```

Run `configure_blueprint.sh --help` for the full set of options (`--llm-mode`, `--vlm-mode`, `--llm-base-url`, `--vlm-base-url`, `--llm-name`, `--vlm-name`).

This script also enforces `use_base64: true` in `config.yml` and `SOP_MESSAGING_SCHEMA=JSON` in the DS-SOP `.env` — both are required for correct Kibana field mapping and video understanding.

> **Auto-configuration safety net:** If you skip this step, `deploy.sh` will detect the unconfigured placeholder and run `configure_blueprint.sh` automatically with default settings before starting services.

### Run Deployment

Log into NGC registry and start the blueprint Docker compose services:

```bash
cd <bp-repo>
./agentic/vss-sop-skills/vss-sop-deploy/scripts/deploy.sh --bp-repo .
```

> **Note on Volume Permissions & Automatic Index Cleanup:** `deploy.sh` automatically pre-creates and configures all required bind-mounted data log directories (such as `kafka`, `elastic/data`, `redis/data`, etc.) with universal `777` permissions (using `sudo` if needed). This completely avoids container-level "Permission denied" errors.
>
> In addition, `deploy.sh` automatically cleans up old Elasticsearch (`elastic/data` and `elastic/logs`) and Kafka data from prior runs before launching the containers. This prevents index pattern mapping pollution (such as stale `1970-01-01` date indices with old protobuf schemas) from interfering with wildcard queries or breaking Kibana visualization fields.

### Monitor

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
```

Deploy is complete when all `mdx-*` containers show status `Up`.

### Endpoints

| Service | Endpoint |
|---|---|
| VSS-UI | `http://<EXTERNAL_IP>:3000` |
| Grafana-UI | `http://<EXTERNAL_IP>:35000/` |
| Kibana-UI | `http://<EXTERNAL_IP>:5601/app/home#/` |
| VIOS-UI | `http://<EXTERNAL_IP>:30888/vst/#/dashboard` |
| Phoenix-UI (Telemetry) | `http://<EXTERNAL_IP>:6006/projects` |

---

## Phase 5 — Test with RTSP Stream

### Trigger Client Test

With the RTSP streaming simulator already running, trigger the test client and monitor the integration:

```bash
cd <bp-repo>
./agentic/vss-sop-skills/vss-sop-deploy/scripts/test_rtsp.sh --bp-repo .
```

This automates:
1. Polls `mdx-ds-sop-1` logs until the API server is fully up.
2. Automatically triggers `./run_rtsp_test.sh` in the background to register streams to VIOS.

> **Self-Healing Integration Loop:** To completely avoid the stale RTSP factory bug (where subsequent client runs on `mdx-ds-sop-1` allocate random UDP ports but retain the old RTSP factory on port 8554, yielding 0 RTP bytes), `run_rtsp_test.sh` runs a continuous self-healing loop. If the ds-sop client exits or the connection is lost, the script automatically restarts the `mdx-ds-sop-1` container, waits for readiness, resets the VST sensors, and starts a fresh streaming session to resume recording automatically. This ensures continuous livestreaming and recording stability.

---

## Tear Down

To stop the services and clean up all data logs:

```bash
cd <bp-repo>
./agentic/vss-sop-skills/vss-sop-deploy/scripts/teardown.sh --bp-repo .
```

---

## Error Handling

- **Docker Socket Permissions / Permission Denied:** All scripts (including `deploy.sh`, `teardown.sh`, `test_rtsp.sh`, `build_ds_sop.sh`, and `vss_sop_test.py`) are fully docker-group aware. If the current shell has inactive docker group membership but the user belongs to the `docker` group, the scripts will automatically re-execute themselves under the `sg docker` command wrapper to bypass socket permission errors. If sudo is required, they will prompt/re-run via sudo. You do not need to prefix commands with `sg docker` manually.
- `unknown or invalid runtime name: nvidia` — NVIDIA Container Toolkit not installed or Docker not restarted. See [`references/prerequisites.md`](references/prerequisites.md).
- NGC auth error — verify `ngc_api_key.txt` and ensure it is exported correctly.
- `mdx-ds-sop-1` not starting — check model paths match between Phase 1 and `ds-sop/.env`.
- API server not ready — check `docker logs mdx-ds-sop-1` for errors or progress.
- `MediaInfo.parse: Unsupported file type` error in `mdx-ds-sop-1` when downloading video from VST directly — Ensure that `use_base64: true` is set under `video_understanding` in `deployments/sop/vss-agent/configs/config.yml`. This forces the VSS Agent to send base64-encoded video instead of VST links, avoiding VST download parsing errors.
- Kibana "No field found for [llm.queries.response.keyword]" or missing messages in ELK — Ensure `SOP_MESSAGING_SCHEMA=JSON` and `ENABLE_MESSAGING=1` are set in `deployments/ds/ds-sop/.env` (automatically enforced by `configure_blueprint.sh`). See **vss-sop-build** SKILL.md § Error Handling for the full field name reference and validation steps.
- UI shows `Failed to load image with src: http://<internal-ip>:30888/...` — `EXTERNAL_IP` in `.env` is a host-only address. Set it to the public/browser-reachable IP (see Phase 4) and recreate the agent container.

---

## License

Use of this skill is governed by the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en) and the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
