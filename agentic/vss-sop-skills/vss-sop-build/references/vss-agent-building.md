# VSS Agent → VSS SOP Agent: Modification Guide

Modify upstream `video-search-and-summarization` (branch `3.1.0`) compose files under `deployments/agents/` for the `bp_sop_2d` profile.

> **Scope:** only `deployments/agents/`. Files under `deployments/sop/vss-agent/` (configs, patches, templates, `.env`) are covered in [`sop-app-building.md`](sop-app-building.md) and the main SKILL.md (Stage 2).

**Source:** `../video-search-and-summarization/deployments/agents/`
**Target:** `../vss-sop/deployments/agents/`

## Step 0 — Copy from Upstream and Modify for SOP

Run `./agentic/vss-sop-skills/vss-sop-build/scripts/copy_agents_from_upstream.sh` from the blueprint repo root. It copies the upstream `deployments/agents/` tree into `deployments/agents/` and applies all SOP modifications in one step (`bp_sop_2d` profiles, SOP patch volume mounts on `vss-va-mcp`, and removal of non-SOP includes/`depends_on` entries).

The sections below describe the individual changes for reference/manual builds.

## Overview

The VSS Agent is shared across blueprints (same image `nvcr.io/nvidia/vss-core/vss-agent:3.1.0`). For SOP:

1. Add `bp_sop_2d` to compose service profiles
2. Bind-mount SOP Python patches into `vss-va-mcp` (no image rebuild)
3. Comment out unused `ai-agents` includes
4. Drop `agent-eval` volume + dev-only `depends_on` from `vss-agent`

| File | Change Type |
|---|---|
| `agents/compose.yml` | Comment out `ai-agents` includes |
| `agents/vss-agent/vss-agent-docker-compose.yml` | Add `bp_sop_2d` to `vss-va-mcp` + `vss-agent`; add SOP patch volumes; remove `agent-eval` volume + extra `depends_on` |
| `agents/agent_ui/compose.yml` | Add `bp_sop_2d` to `vss-ui` profiles |

---

## Step 1 — `agents/compose.yml`

Comment out the two `ai-agents/*` includes (SOP doesn't deploy those services); keep `vss-agent/vss-agent-docker-compose.yml` and `agent_ui/compose.yml` active. Reference: `./configs/vss-agent/agents-compose-include.yml`.

## Step 2 — `agents/vss-agent/vss-agent-docker-compose.yml`

### 2a & 2c. Add `bp_sop_2d` to `vss-va-mcp` and `vss-agent` profiles

You can perform this profile insertion automatically using the Python patching script:

```bash
python3 agentic/vss-sop-skills/vss-sop-build/scripts/patch_profiles.py deployments/agents/vss-agent/vss-agent-docker-compose.yml
```

This automates robustly inserting `"bp_sop_2d"` into the `profiles:` list for all services in `vss-agent-docker-compose.yml`, preserving exact YAML structure and indentation.

*(Manual fallback: Insert `bp_sop_2d` into `vss-va-mcp`'s profile list and `vss-agent`'s profile list).*

### 2b. Add SOP patch volume mounts to `vss-va-mcp`

Replace the existing `vss-va-mcp.volumes:` list with the four-entry list in `./configs/vss-agent/vss-va-mcp-volumes.yml`. The first entry is the standard read-only `${MDX_SAMPLE_APPS_DIR}:/vss-agent/deployments:ro`; the next three overlay `tools.py`, `utils.py`, and `es_client.py` from `${MDX_SAMPLE_APPS_DIR}/sop/vss-agent/patches/` onto `/vss-agent/.venv/lib/python3.13/site-packages/vss_agents/video_analytics/` (read-only). No image rebuild needed.

> Verify the Python version (`python3.13`) matches the VSS Agent image. Wrong version = silent mount path mismatch.

### 2d. Remove `agent-eval` volume + extra `depends_on`

Drop these from `vss-agent` (they belong to other blueprints, not SOP):

- The `agent-eval:/vss-agent/agent_eval` line under `volumes:`
- The top-level `volumes:` block defining `agent-eval` (the entire definition)
- `depends_on` entries: `rtvi-vlm`, `rtvi-embed`, `lvs-server`

Keep only NIM model entries and `vss-va-mcp` in `depends_on`. Resulting `vss-agent.volumes:` is a single bind-mount: `${MDX_SAMPLE_APPS_DIR}:/vss-agent/deployments:ro`.

> **Also remove any fp8 NIM service entries.** The upstream compose may include `nvidia-nemotron-nano-9b-v2-fp8` and `nvidia-nemotron-nano-9b-v2-fp8-shared-gpu` entries in `vss-agent.depends_on`. These services are not defined in the SOP NIM profile, so Docker Compose will refuse to start `vss-agent` with `depends on undefined service`. Remove them along with the other non-SOP entries above.

## Step 3 — `agents/agent_ui/compose.yml`

You can perform this profile insertion automatically using the Python patching script:

```bash
python3 agentic/vss-sop-skills/vss-sop-build/scripts/patch_profiles.py deployments/agents/agent_ui/compose.yml
```

*(Manual fallback: Insert `"bp_sop_2d"` after `"bp_ps_2d"` in `vss-ui.profiles`. Resulting list: `["bp_wh_2d", "bp_smc_2d", "bp_ps_2d", "bp_sop_2d", "bp_developer_base_2d", "bp_developer_search_2d", "bp_developer_lvs_2d", "bp_developer_alerts_2d_cv", "bp_developer_alerts_2d_vlm"]`).*

---

## Summary

```
agents/compose.yml
  └─ Comment out ai-agents includes

agents/vss-agent/vss-agent-docker-compose.yml
  ├─ vss-va-mcp:  +bp_sop_2d, +3 SOP patch volume mounts
  ├─ vss-agent:   +bp_sop_2d, -agent-eval volume, -depends_on (rtvi-vlm, rtvi-embed, lvs-server)
  └─ Top-level:   -agent-eval volume definition

agents/agent_ui/compose.yml
  └─ vss-ui: +bp_sop_2d
```

## Verification

Run `./scripts/vss-agent/verify.sh` (a thin wrapper that delegates to `scripts/verify_build.py --component agents`, the single source of truth) to confirm:

1. `bp_sop_2d` is in `vss-agent-docker-compose.yml`
2. `bp_sop_2d` is in `agent_ui/compose.yml`
3. `ai-agents` includes are commented out in `agents/compose.yml`
4. Three SOP patch volume mounts present in `vss-va-mcp`
5. No `agent-eval` references remain

---

## License

Use of this skill is governed by the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en) and the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
