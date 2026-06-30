---
name: sop-build
description: >-
  Orchestrate the end-to-end SOP pipeline, including preflight prerequisite checks, verifying models and downloading assets,
  generating the DeepStream SOP microservice with RTSP output, evaluating the microservice, and building,
  deploying, and testing the VSS SOP blueprint. Use when asked to run the full SOP pipeline, set up the SOP
  pipeline from scratch, execute preflight checks, verify models, download assets, generate the SOP microservice,
  evaluate the microservice, build the VSS blueprint, deploy the VSS blueprint, test the VSS blueprint, or
  manage the complete build-evaluate-deploy-test cycle.
owner: NVIDIA
service: vss-sop
version: 1.0.0
license: CC-BY-4.0 AND Apache-2.0
reviewed: 2026-06-23
metadata:
  openclaw: { "emoji": "🚀", "os": ["linux"] }
  author: "Quan Vu <qvu@nvidia.com>"
  tags: ["sop", "build", "pipeline"]
---

# Full SOP Pipeline — Build, Evaluate, Deploy, Test

Orchestrates the complete SOP pipeline from asset download through deployment validation. Each phase must succeed before proceeding to the next.

## Overview

- Set up the entire SOP pipeline from scratch on a new machine
- Re-run the full pipeline after upstream changes
- Rebuild and redeploy after ds-sop-skills or vss-sop-build changes

## Prerequisites

| Requirement | Detail |
|---|---|
| GPU | NVIDIA H100 / H200 / A100 with >= 80 GB VRAM |
| Driver | NVIDIA 580+ with CUDA 13 |
| Software | Docker with BuildKit, NVIDIA Container Toolkit 1.18.1, `ngc` CLI |
| Secret key | `<bp-repo>/.secret/ngc_api_key.txt` must contain a valid NGC API key |
| ds-sop-skills | `../ds-sop-skills/` directory with `deepstream-sop/SKILL.md` and prompt files |

## Usage

| Phase | What | Key Script / Skill |
|---|---|---|
| Pre | Prerequisites Check & Auto-Install (run first) | `vss-sop-deploy/scripts/preflight_check.sh --fix` |
| 0 | Verify Models & Download Assets | `vss-sop-deploy/scripts/download_assets.sh` |
| 1 | Generate SOP Microservice | `../ds-sop-skills/example_sop_prompt.md` |
| 2 | Evaluate SOP Microservice | `../ds-sop-skills/eval_sop_prompt.md` |
| 3 | Build VSS-SOP Blueprint | **vss-sop-build** skill |
| 4 | Deploy VSS-SOP Blueprint | **vss-sop-deploy** skill |
| 5 | Test VSS-SOP Blueprint | **vss-sop-test** skill |

---

### Phase Pre — Prerequisites Check & Auto-Install

Run this **first**, before any other phase. It checks secret key files, GPU drivers, Docker, NVIDIA Container Toolkit, and NGC configuration, and can automatically install or configure missing prerequisites (NVIDIA Driver 580, CUDA Toolkit 13, Docker, Docker Compose, NVIDIA Container Toolkit, NGC CLI, and NGC CLI configuration).

```bash
cd <bp-repo>
# Run checks and auto-fix/install any missing components
./agentic/vss-sop-skills/vss-sop-deploy/scripts/preflight_check.sh --bp-repo . --fix
```

To run the checks only, without modifying your system:

```bash
cd <bp-repo>
./agentic/vss-sop-skills/vss-sop-deploy/scripts/preflight_check.sh --bp-repo .
```

**Verify:** the script reports all prerequisites satisfied (secret keys, NVIDIA driver, Docker, NVIDIA Container Toolkit, NGC CLI & config). If any manual check fails, resolve it per the printed advice before proceeding. See **vss-sop-deploy** Phase 0 for reference docs.

---

### Phase 0 — Verify Models & Download Assets

For optimal accuracy, you must retrain/fine-tune the models, which can be done using the SOP Training Blueprint. After training, move model and config to `/opt/models/...` and `/opt/sop/...` directories.

```bash
cd <bp-repo>
./agentic/vss-sop-skills/vss-sop-deploy/scripts/download_assets.sh --bp-repo .
```

This performs:
1. Checks that the required trained model and config files exist under `/opt/models` and `/opt/sop`. If not, prompts the user to retrain/fine-tune the models using the SOP Training Blueprint.
2. Downloads and re-encodes the sample RTSP video to H.264 at 30 FPS.
3. Sets up the `ds-sop` cache directory and data log directories.

**Verify:** `/opt/models/vlm/checkpoint/` contains VLM weights, `/opt/models/gbed_models/ddm/checkpoint.pth.tar` exists, `/opt/sop/configs/actions.json` and `/opt/sop/configs/vlm_prompts.txt` exist.

---

### Phase 1 — Generate SOP Microservice

Generate the DeepStream SOP microservice with RTSP streaming output into `../ds_sop_microservice`. Skip if `../ds_sop_microservice/` already exists with the expected layout.

The generation prompt **must** include the RTSP streaming output feature request (../ds-sop-skills § 18), which the VSS-SOP blueprint requires:

```
Please follow instructions in ../ds-sop-skills/example_sop_prompt.md to generate a SOP microservice with rtsp streaming output feature ($18) in folder ../ds_sop_microservice
```

This follows `../ds-sop-skills/deepstream-sop/SKILL.md` to generate:
- FastAPI microservice with `/v1/chat/completions` SSE endpoint
- DeepStream GEBD pipeline with DDM-Net via Triton CAPI
- VLM inference (Cosmos Reason) for action classification
- SOP sequence checker with compliance monitoring
- Kafka messaging (NvProto + JSON schema)
- Docker build file and deploy compose
- **RTSP streaming output** (`RTSPStreamingServer`, `tee1`-tapped encode branch, `SW_ENCODER`/`ENABLE_RTSP_OUTPUT`/`RTSP_PORT`)

**Verify:** `../ds_sop_microservice/` contains `docker/Docker.build`, `deploy/compose.yaml`, `nvds_action_detector/{api_server,ds_3d_action_pipeline,ds_sop_process,vllm_inference}.py`.

---

### Phase 2 — Evaluate SOP Microservice

Evaluate the generated microservice to catch issues before the full blueprint build.

```
Please follow ../ds-sop-skills/eval_sop_prompt.md to evaluate ../ds_sop_microservice with rtsp streaming output feature ($18)
```

Use the following environment settings:

```
VLLM_MODEL_PATH="/opt/models/vlm/checkpoint/"
DDM_MODEL_PATH="/opt/models/gbed_models/ddm/checkpoint.pth.tar"
MODEL_ROOT_DIR=/opt/models
ACTION_CONFIG_PATH=/opt/sop/configs/actions.json
VLM_PROMPT_PATH=/opt/sop/configs/vlm_prompts.txt
HOST_CACHE=$HOME/.cache/ds-sop
TEST_VIDEO_PATH=./sop-resources/sop-server-fan-installation-data_v1.0-260213/server_fan/raw/Install_1_h264_30fps.mp4
USER_ID=0
GROUP_ID=0
```

The Pylon SDK binary is at `../ds_sop_microservice/binaries/pylon-25.10.2_linux-x86_64_setup.tar.gz`.

The evaluation follows `../ds-sop-skills/deepstream-sop/references/skill_12_evaluation_workflow.md` and performs:
1. Static validation of generated source.
2. Docker image build.
3. Service launch with `deploy/.env`.
4. API unit tests.
5. File-video API evaluation.
6. Latency, camera, or Kafka evaluation as applicable.

**Fix any issues found during evaluation before proceeding.** Record commands, results, timings, and fixes in the evaluation report.

---

### Phase 3 — Build VSS-SOP Blueprint (vss-sop-build)

Run the **vss-sop-build** skill from the blueprint repo root. This scaffolds the SOP blueprint layer on top of VSS 3.1:

```bash
cd <bp-repo>
# Follow vss-sop-build SKILL.md Quick Start steps 1–10
```

Key stages:
1. Clone upstream VSS 3.1.0 repository.
2. Copy all reference files verbatim.
3. Copy and modify foundational services for SOP.
4. Copy and modify VIOS for SOP.
5. Copy agents from upstream and apply SOP profile.
6. Copy NIM from upstream with SOP conventions.
7. Build `ds-sop:1.0.0` image from the generated `../ds_sop_microservice`.
8. Verify the build with `verify_build.sh`.

See [vss-sop-build SKILL.md](../vss-sop-build/SKILL.md) for full details.

---

### Phase 4 — Deploy VSS-SOP Blueprint (vss-sop-deploy)

Run the **vss-sop-deploy** skill from the blueprint repo root:

```bash
cd <bp-repo>
# Follow vss-sop-deploy SKILL.md Phases 0–5
```

Deployment phases:
- **Phase 0:** Prerequisites check and auto-install (`preflight_check.sh --bp-repo . --fix`) — already run in **Phase Pre** above; re-runs as an idempotent safety check.
- **Phase 1:** Download models & assets (already done in Phase 0 above — will skip if present).
- **Phase 2:** Start RTSP server simulation (`start_rtsp_server.sh --bp-repo .`).
- **Phase 3:** Build DS-SOP Docker image & verify components (already built in Phase 3 above — verify only).
- **Phase 4:** Configure & deploy the blueprint (`configure_blueprint.sh`, `deploy.sh`).
- **Phase 5:** Test with RTSP stream (`test_rtsp.sh --bp-repo .`).

See [vss-sop-deploy SKILL.md](../vss-sop-deploy/SKILL.md) for full details.

---

### Phase 5 — Test VSS-SOP Blueprint (vss-sop-test)

Run the **vss-sop-test** skill from the blueprint repo root:

```bash
cd <bp-repo>
python agentic/vss-sop-skills/vss-sop-test/scripts/vss_sop_test.py \
  --bp-repo . \
  --env-file deployments/sop/.env
```

Test phases:
- **Phase 1:** Service health — all required containers running.
- **Phase 2:** ELK data pipeline — Elasticsearch, indices, VLM messages, Kibana fields.
- **Phase 3:** VIOS recording & livestream — sensors, streams, recording, livestream.
- **Phase 4:** VSS Agent end-to-end — MCP, LLM, VLM, snapshot, video, report generation.

See [vss-sop-test SKILL.md](../vss-sop-test/SKILL.md) for full details.

---

## Error Handling

### Pass Criteria and Verification

Ensure all unit tests, build checks, deploy checks, and test phases pass with zero failures.
- Do not classify any failure as environmental, intermittent, infrastructure, or acceptable. Treat every failure as a bug to be fixed.
- Do not proceed to the next phase if the current phase has any failing test. Fix all failures first.
- Do not declare completion if any test is failing. A result like "36/38 checks pass" is a failure, not a success.
- If a test times out, fix the timeout or the underlying latency rather than dismissing it.
- If stuck on the same failure after three fix attempts, stop and ask the user for help. Do not skip the failure.

### Auto-Debug and Fix Loop

When any phase fails:
1. Read the failure output and auto-debug hints.
2. Collect container logs (`docker logs <container> --tail 50`).
3. Apply the fix (env, compose, patches, image rebuild, source regeneration).
4. Re-run the failing phase to confirm the fix.
5. Repeat until all checks in that phase pass.

If the root cause is in the generated DS-SOP source code, update `ds-sop-skills` references and regenerate. If the root cause is in the build/deploy config, update `vss-sop-build` references.

---

## Examples

After all phases complete with 100% pass rate, present a summary table:

| Category | Status | Details |
|---|---|---|
| Phase Pre: Prerequisites | Pass/Fail | Driver, Docker, Toolkit, NGC config |
| Phase 0: Assets | Pass/Fail | Models and configs at /opt |
| Phase 1: Generate | Pass/Fail | ../ds_sop_microservice with RTSP |
| Phase 2: Evaluate | Pass/Fail | Unit tests + API evaluation |
| Phase 3: Build | Pass/Fail | Per-stage build status |
| Phase 4: Deploy | Pass/Fail | Per-phase deploy status |
| Phase 5: Test | Pass/Fail | Per-phase test status (must be 100%) |

Include:
- Any `vss-sop-build` or `ds-sop-skills` files updated during debugging.
- Service endpoints: VSS-UI, Kibana, VIOS-UI, Grafana.

If any test is still failing, the report must clearly state **"Pipeline Incomplete — X failures remain"** and list them. Do not present a partial success as completion.

### Service Endpoints (after successful deploy)

| Service | Endpoint |
|---|---|
| VSS-UI | `http://<EXTERNAL_IP>:3000` |
| Kibana | `http://<EXTERNAL_IP>:5601/app/home#/` |
| VIOS-UI | `http://<EXTERNAL_IP>:30888/vst/#/dashboard` |
| Grafana | `http://<EXTERNAL_IP>:35000/` |
| Phoenix (Telemetry) | `http://<EXTERNAL_IP>:6006/projects` |

---

## License

Use of this skill is governed by the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en) and the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
