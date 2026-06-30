---
name: vss-sop-test
description: >-
  Run post-deployment tests for the VSS SOP blueprint. Checks service health,
  ELK data pipeline, VIOS recording/livestream, and VSS agent (MCP, LLM, VLM,
  snapshot, video, report). Auto-debugs failures. Use when asked to test SOP,
  verify SOP deployment, check SOP services, validate SOP, run SOP health checks,
  or troubleshoot SOP after deploy.
owner: NVIDIA
service: vss-sop
version: 1.0.0
license: CC-BY-4.0 AND Apache-2.0
reviewed: 2026-06-23
metadata:
  openclaw: { "emoji": "🧪", "os": ["linux"] }
  author: "Quan Vu <qvu@nvidia.com>"
  tags: ["vss", "sop", "test"]
---

# VSS SOP — Post-Deployment Test Suite

Run this skill after the VSS SOP blueprint has been deployed (see `vss-sop-deploy`). It validates every layer of the stack, and when a test fails it auto-debugs by collecting container logs and printing actionable fix hints.

## Overview

- Verify a fresh SOP deployment is healthy
- Check that ELK is receiving VLM data
- Validate VIOS recording and livestream
- Test the VSS agent: MCP server, LLM endpoint, VLM endpoint, snapshot, video, and report generation
- Troubleshoot a broken deployment

## Instructions

### Quick Start

```bash
cd <bp-repo>
python agentic/vss-sop-skills/vss-sop-test/scripts/vss_sop_test.py \
  --bp-repo . \
  --env-file deployments/sop/.env
```

Run a single phase:

```bash
python agentic/vss-sop-skills/vss-sop-test/scripts/vss_sop_test.py --phase 1   # Service health only
python agentic/vss-sop-skills/vss-sop-test/scripts/vss_sop_test.py --phase 2   # ELK only
python agentic/vss-sop-skills/vss-sop-test/scripts/vss_sop_test.py --phase 3   # VIOS only
python agentic/vss-sop-skills/vss-sop-test/scripts/vss_sop_test.py --phase 4   # VSS Agent only
```

### Test Phases

### Phase 1 — Service Health

Checks that all required Docker containers are running (`Up`).

| Required containers | Optional |
|---|---|
| `mdx-kafka`, `mdx-redis`, `mdx-elastic`, `mdx-logstash`, `mdx-kibana`, `vss-agent`, `vss-va-mcp`, `mdx-ds-sop-1`, `sensor-ms-sop`, `recorder-ms-1-sop`, `rtspserver-ms-1-sop`, `storage-ms-sop`, `sdr-http-recorder-sop`, `sdr-http-rtspserver-sop` | `mdx-prometheus`, `mdx-grafana`, `mdx-dcgm-exporter`, `mdx-cadvisor`, `mdx-node-exporter`, `mdx-phoenix`, `vss-ui` |

**Auto-debug:** Prints `docker logs <container> --tail 30` for any missing or unhealthy container.

**Manual fix:** Redeploy with `docker compose -f compose.yml --env-file sop/.env --profile bp_sop_2d up -d`.

### Phase 2 — ELK Data Pipeline

1. **Elasticsearch reachable** — `GET localhost:9200/_cluster/health`
2. **Cluster health** — must be `green` or `yellow`
3. **Indices exist** — at least one `mdx-vlm-captions-*` index
4. **VLM messages** — `mdx-vlm-captions-*` index has `> 0` documents
5. **Kibana dashboard fields** — verifies the actual ES mapping has the flat JSON fields the dashboard expects (`response`, `sensor_id`, `cv_execute_time`, `vlm_execute_time`, `chunk_idx`, `frame_number`, `@timestamp`), and that no protobuf-style nested fields (`llm`, `sensor`) are present instead
6. **Kibana ndjson fields** — queries Kibana's saved objects API to confirm the imported data view's runtime fields reference the correct flat field names (not `llm.queries.response.keyword` or `sensor.id.keyword`), and that `timeFieldName` is `@timestamp`

**Auto-debug:** Tails `mdx-logstash`, `mdx-elastic`, and `mdx-kibana` logs.

**Manual fix if no VLM data:**

```bash
# Verify Kafka topic has messages
docker exec mdx-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 --topic mdx-vlm-captions \
  --from-beginning --max-messages 3 --timeout-ms 10000

# Check Logstash pipeline
docker logs mdx-logstash --tail 50
```

**Manual fix if Kibana dashboard field mismatch:**

```bash
# Check the actual field structure in Elasticsearch
curl localhost:9200/mdx-vlm-captions-*/_mapping?pretty

# If fields are flat (response, sensor_id) but ndjson uses nested (llm.queries.response):
# Re-import the corrected ndjson
curl -X POST localhost:5601/api/saved_objects/_import?overwrite=true \
  -H 'kbn-xsrf: true' \
  --form file=@deployments/sop/sop-app/kibana-dashboard/sop-kibana-objects.ndjson
```

### Phase 3 — VIOS Recording & Livestream

1. **VST reachable** — `GET localhost:30888/vst/api/v1/sensor/list`
2. **Sensors registered** — at least one sensor (typically `sensor_0`)
3. **Streams available** — `/sensor/streams` returns data
4. **Recording** — checks `/record/status` per sensor
5. **Livestream** — checks `/live/streams` per sensor

**Auto-debug:** Tails `sensor-ms-sop`, `rtspserver-ms-1-sop`, `recorder-ms-1-sop`, and `sdr-http-recorder-sop` logs.

**Manual fix if no sensor:**

```bash
cd <bp-repo>/deployments/sop/sop-app/helper-scripts
nohup ./run_rtsp_test.sh > client.log &
```

### Phase 4 — VSS Agent End-to-End

| # | Test | Method |
|---|---|---|
| 4.1 | MCP health | `GET localhost:9901/health` |
| 4.2 | LLM endpoint | `GET <LLM_BASE_URL>/v1/models` |
| 4.3 | VLM endpoint | `GET <VLM_BASE_URL>/v1/models` |
| 4.4 | Agent health | `GET localhost:8000/health` |
| 4.5 | Snapshot from VIOS | Agent chat: *"Take a snapshot of sensor sensor_0"* |
| 4.6 | Video → VLM | Agent chat: *"What is the current SOP status for sensor sensor_0?"* |
| 4.7 | Report generation | Agent chat: *"Generate an SOP compliance report for sensor sensor_0 from … to …"* |

**Auto-debug:** Tails `vss-agent`, `vss-va-mcp`, and `mdx-ds-sop-1` logs for any failing sub-test.

**Manual fix for common agent failures:**

```bash
# Agent won't start — check LLM/VLM config
docker inspect vss-agent --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '(LLM|VLM)'

# MCP not healthy — restart
docker restart vss-va-mcp

# NVIDIA API key missing for remote NIM
export NVIDIA_API_KEY="$(cat <bp-repo>/.secret/nvidia_build_api_key.txt)"
docker compose -f compose.yml --env-file sop/.env --profile bp_sop_2d up -d vss-agent
```

**Phase 4.7 "no streams found" on fresh deployment:**
If the test runs immediately after deploy, the report time window (`now-1min to now`) may
precede the first recording. Wait at least 1 minute after the RTSP stream registers before running
Phase 4.

## Examples

### Pass Criteria — MANDATORY

**ALL tests must pass (100%). There is NO acceptable failure rate.**

Rules the agent MUST follow:

1. **Zero tolerance for failures.** Every single test check across all phases MUST pass. A result of 36/38 or 17/19 is a FAIL — not a pass with caveats.
2. **Do NOT classify failures as "environmental", "infrastructure", "test harness", or "intermittent".** Every failure is a bug until you have fixed it and the test passes on re-run.
3. **Do NOT lower the pass threshold.** Never say "good enough", "residual", "non-blocking", or "acceptable failure". If a test fails, fix it.
4. **Do NOT skip or move on.** If a test fails after 3 fix attempts, STOP and ask the user for help. Do NOT proceed to the next phase or declare completion.
5. **Do NOT mark a phase as PASS if any check within it failed.** Use FAIL until every check is green.
6. **Timeout failures are bugs too.** If a test times out (e.g., 30s client read-timeout), increase the timeout or fix the underlying latency — do NOT dismiss it as a test configuration issue.
7. **Re-run after every fix.** After applying a fix, re-run the failing phase to confirm the fix works. A fix is not verified until the test passes.

**Completion criteria:** The pipeline is complete ONLY when the final report shows `38/38` (or equivalent total) with zero failures across all phases. Any other result means more work is needed.

## Workflow for the Agent

Copy this checklist and track progress:

```
Test Progress:
- [ ] Phase 1: Service health — ALL checks PASS
- [ ] Phase 2: ELK data pipeline — ALL checks PASS
- [ ] Phase 3: VIOS recording & livestream — ALL checks PASS (7/7)
- [ ] Phase 4: VSS agent (MCP, LLM, VLM, snapshot, video, report) — ALL checks PASS (7/7)
- [ ] Auto-debug any failures (do NOT skip — every failure must be fixed)
- [ ] Auto-update vss-sop-build skills/scripts if root cause found
- [ ] Confirm 100% pass rate before reporting to user
- [ ] Report summary to user
```

**Step 1 — Determine `<bp-repo>`**

```bash
# Usually the vss-sop checkout
BP_REPO=$(pwd)   # or wherever the user cloned vss-sop
```

**Step 2 — Run the full test suite**

```bash
cd "$BP_REPO"
python agentic/vss-sop-skills/vss-sop-test/scripts/vss_sop_test.py \
  --bp-repo "$BP_REPO" \
  --env-file "$BP_REPO/deployments/sop/.env"
```

## Error Handling

### Auto-Debugging Failures (MANDATORY — do NOT skip)

**CRITICAL: Every failing test MUST be fixed. Do NOT classify any failure as "environmental" or "acceptable" and move on.**

The script already collects container logs for failures. Additionally:

1. Read the `[FAIL]` output and the auto-debug hints printed by the script.
2. Apply the suggested fix (restart container, re-export key, re-run RTSP test, etc.).
3. Re-run the failing phase only:

```bash
python agentic/vss-sop-skills/vss-sop-test/scripts/vss_sop_test.py --phase <N>
```

4. Repeat until ALL tests in that phase pass (100%).
5. If a test still fails after 3 fix attempts, **STOP and ask the user for help**. Do NOT declare completion or proceed to the next step.
6. **Never rationalize a failure.** Statements like "this is an environmental issue, not a code defect" are NOT acceptable reasons to skip a fix. Either fix the environment or fix the test.

### Auto-Updating the Upstream Skill After Root-Cause Fix

When a test failure is debugged and the root cause is identified, update the corresponding upstream skill so the bug cannot recur. First determine **where** the root cause lives, then update the correct skill:

| Root cause lives in… | Upstream skill to update |
|---|---|
| Build/deploy config (`.env`, YAML, compose, copy scripts, build steps) | **`vss-sop-build`** (`agentic/vss-sop-skills/vss-sop-build/`) |
| DS-SOP source code (Python, DeepStream pipeline, Triton, VLM inference, API, Dockerfile, protos, SOP checker, RTSP streaming, etc.) | **`ds-sop-skills`** (`../ds-sop-skills/deepstream-sop/`) |

Keep every change **simple and general** — fix the pattern, not just the instance.

#### 4a. Root cause in build/deploy → update `vss-sop-build`

| Root-cause category | What to update in `vss-sop-build` |
|---|---|
| Missing/wrong config value (`.env`, YAML) | Fix the default in the matching reference file under `agentic/vss-sop-skills/vss-sop-build/references/` |
| Missing build step or wrong ordering | Add/reorder step in the relevant `*-building.md` reference doc |
| Compose issue (profile, volume, depends_on) | Patch the reference compose file(s) under `agentic/vss-sop-skills/vss-sop-build/references/deployments/` or `agentic/vss-sop-skills/vss-sop-build/references/configs/` |
| Script bug (copy, patch, verify) | Fix the script under `agentic/vss-sop-skills/vss-sop-build/scripts/` |
| Undocumented failure mode | Add entry to `vss-sop-build` SKILL.md `## Error Handling` section |

#### 4b. Root cause in DS-SOP source code → update `ds-sop-skills`

| Root-cause category | What to update in `ds-sop-skills` |
|---|---|
| API endpoint / schema bug | Fix the reference in `../ds-sop-skills/deepstream-sop/references/skill_01_fastapi_endpoints.md` or `../ds-sop-skills/deepstream-sop/references/skill_02_pydantic_schemas.md` |
| DeepStream pipeline issue | Fix the reference in `../ds-sop-skills/deepstream-sop/references/skill_03_deepstream_pipeline.md` or `../ds-sop-skills/deepstream-sop/references/skill_04_config_templates.md` |
| Triton / DDM model issue | Fix the reference in `../ds-sop-skills/deepstream-sop/references/skill_05_triton_ddm_model.md` or `../ds-sop-skills/deepstream-sop/references/skill_05b_custom_postprocess.md` |
| SOP process / checker logic | Fix the reference in `../ds-sop-skills/deepstream-sop/references/skill_06_sop_process_manager.md` or `../ds-sop-skills/deepstream-sop/references/skill_06b_sop_checker.md` |
| VLM inference bug | Fix `../ds-sop-skills/deepstream-sop/references/vllm_inference_reference.py` or the corresponding skill doc |
| RTSP streaming output | Fix `../ds-sop-skills/deepstream-sop/references/skill_18_rtsp_streaming_output.md` or the reference code |
| Messaging / Kafka schema | Fix `../ds-sop-skills/deepstream-sop/references/skill_16_message_schema.md` or `../ds-sop-skills/deepstream-sop/references/messager_reference.py` |
| Dockerfile / build | Fix `../ds-sop-skills/deepstream-sop/references/Dockerfile_reference` or `../ds-sop-skills/deepstream-sop/references/skill_09_docker_build_deploy.md` |
| Reference source file bug | Fix the corresponding `*_reference.py` / `*_reference.cpp` file under `../ds-sop-skills/deepstream-sop/references/` |
| Undocumented failure mode | Add entry to `../ds-sop-skills/deepstream-sop/SKILL.md` error handling or the relevant skill doc |

#### Procedure (applies to both 4a and 4b):

1. **Classify the root cause** — determine whether it belongs to build/deploy config (`vss-sop-build`) or DS-SOP source code (`ds-sop-skills`).
2. **Identify the file** — map the root cause to the corresponding source file in the chosen skill's directory.
3. **Make the minimal fix** — change only what is needed; prefer updating a default value, adding a guard/check, or appending an entry under Error Handling over larger rewrites.
4. **Verify consistency**:
   - If a `vss-sop-build` reference was changed, re-run `agentic/vss-sop-skills/vss-sop-build/scripts/verify_build.sh` to confirm the build still passes.
   - If a `ds-sop-skills` reference was changed, regenerate the source and rebuild `ds-sop:1.0.0` to confirm the fix.
5. **Log the change** — include in the final report which skill and file was updated and why.

> **Rule of thumb:** if the root cause is in the generated DS-SOP source code (Python logic, pipeline config, API, Dockerfile, etc.), the fix belongs in `ds-sop-skills` — so the next generation produces correct code. If it is in the VSS blueprint build process (compose wiring, `.env` defaults, copy scripts, build ordering), the fix belongs in `vss-sop-build`. If it is environment-specific (wrong IP, expired key), just document it in the report — don't change either skill.

**Step 5 — Report results**

Present the final summary table to the user. The report MUST show 100% pass rate across all phases.

- If all tests pass: report the green summary.
- If any test still fails: report the failure with full debug context and **explicitly state that the pipeline is NOT complete**. Do NOT use language suggesting partial success (e.g., "36/38 checks pass" is a failure report, not a success report).

If any `vss-sop-build` or `ds-sop-skills` files were updated in Step 4, list them with a one-line summary of each change.

## Service Endpoints Reference

| Service | URL |
|---|---|
| Elasticsearch | `http://localhost:9200` |
| Kibana | `http://<EXTERNAL_IP>:5601` |
| VST (VIOS) | `http://localhost:30888/vst` |
| VSS Agent | `http://localhost:8000` |
| VSS VA MCP | `http://localhost:9901` |
| LLM NIM | Configured via `LLM_BASE_URL` in `.env` |
| VLM NIM | Configured via `VLM_BASE_URL` in `.env` |
| VSS UI | `http://<EXTERNAL_IP>:3000` |
| Phoenix | `http://<EXTERNAL_IP>:6006` |

## Dependencies

The test script requires only `requests` (installed in most Python environments). No additional `pip install` needed if you have a standard Python 3.8+ environment.

---

## License

Use of this skill is governed by the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en) and the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
