# VSS NIM → VSS SOP NIM

Modify upstream `video-search-and-summarization` (branch `3.1.0`) `deployments/nim/` for SOP.

**Source:** `../video-search-and-summarization/deployments/nim/`
**Target:** `../vss-sop/deployments/nim/`

## Step 0 — Copy from Upstream and Modify for SOP

Run `./agentic/vss-sop-skills/vss-sop-build/scripts/copy_nim_from_upstream.sh` from the blueprint repo root. It copies the upstream `deployments/nim/` tree into `deployments/nim/` and applies all SOP modifications in one step (per-GPU profile variants, env-file renames, `hw-OTHER*`/`fp8`/`fallback-override.env` cleanup, the `nvidia-nemotron-nano-9b-v2/` → `nemotron-nano-v2/` rename, and compose tuning).

The sections below describe the individual changes for reference/manual builds.

## Overview of Changes

| # | Change |
|---|---|
| 1 | Add per-GPU profile variants (split generic into `_H100`, `_RTX6000PROBW`, etc.) |
| 2 | Rename env files: `hw-*-shared.env` → `hw-*-full.env`; `hw-RTXPRO6000BW*` → `hw-RTX6000PROBW*` |
| 3 | Remove `hw-OTHER*` env files, `fallback-override.env`, entire `nvidia-nemotron-nano-9b-v2-fp8/` |
| 4 | Rename `nvidia-nemotron-nano-9b-v2/` → `nemotron-nano-v2/` (and update its compose) |
| 5 | Adjust default GPU device IDs for SOP multi-GPU layout |
| 6 | Remove `NIM_DISABLE_MM_PREPROCESSOR_CACHE` from env files |
| 7 | Tune `NIM_KVCACHE_PERCENT`, `NIM_MAX_MODEL_LEN` per workload |

---

## Step 1 — Top-Level `nim/compose.yml`

```diff
- - path: nvidia-nemotron-nano-9b-v2/compose.yml
- - path: nvidia-nemotron-nano-9b-v2-fp8/compose.yml
+ - path: nemotron-nano-v2/compose.yml
```

## Step 2 — Per-Model `compose.yml` Changes

For each model, apply these patterns:

### 2a. Per-GPU profile variants

```diff
  profiles:
- - vlm_local_cosmos-reason2-8b
+ - vlm_local_cosmos-reason2-8b_H100
+ - vlm_local_cosmos-reason2-8b_RTX6000PROBW
```

### 2b. Shorten container names

`container_name: cosmos-reason2-8b` → `cr2-8b`

### 2c. Reference `hw-*-full.env` (drop the fallback override)

```diff
  env_file:
-   - ${MDX_SAMPLE_APPS_DIR}/nim/cosmos-reason2-8b/hw-${HARDWARE_PROFILE}-shared.env
-   - ${VLM_ENV_FILE:-${MDX_SAMPLE_APPS_DIR}/nim/fallback-override.env}
+   - ${MDX_SAMPLE_APPS_DIR}/nim/cosmos-reason2-8b/hw-${HARDWARE_PROFILE}-full.env
```

### 2d. Adjust default GPU device IDs

| Variant | Upstream → SOP |
|---|---|
| Dedicated VLM | `${VLM_DEVICE_ID:-0}` → `${VLM_DEVICE_ID:-2}` |
| Shared LLM+VLM | `${SHARED_LLM_VLM_DEVICE_ID:-${VLM_DEVICE_ID:-0}}` → `${LLM_DEVICE_ID:-1}` |

---

## Step 3 — Env File Cleanup (per model)

### 3a. Rename

| Upstream | SOP |
|---|---|
| `hw-H100-shared.env` | `hw-H100-full.env` |
| `hw-RTXPRO6000BW.env` | `hw-RTX6000PROBW.env` |
| `hw-RTXPRO6000BW-shared.env` | `hw-RTX6000PROBW-full.env` |

### 3b. Remove

Run `./scripts/nim/cleanup_unused.sh` — it deletes per-model `hw-OTHER*` env files, the entire `nvidia-nemotron-nano-9b-v2-fp8/` directory, and `fallback-override.env`.

Also remove `hw-DGX-SPARK*.env` and `hw-L40S.env` from models that don't use those profiles in SOP (uncomment the targeted lines in the script or remove manually per model).

### 3c. Drop `NIM_DISABLE_MM_PREPROCESSOR_CACHE=1`

Remove from all VLM env files.

### 3d. Tune NIM params (example: `hw-RTX6000PROBW`)

| Param | Upstream → SOP |
|---|---|
| `NIM_KVCACHE_PERCENT` | `0.4` → `0.35` |
| `NIM_MAX_MODEL_LEN` | `32768` → `16384` |

---

## Step 4 — Rename `nvidia-nemotron-nano-9b-v2/` → `nemotron-nano-v2/`

Run `./scripts/nim/rename_nemotron.sh` to perform the directory rename.

Then update its `compose.yml` to match SOP conventions (per-GPU profiles, short container name, `-full.env`, adjusted device IDs).

---

## Result

```
nim/
├── compose.yml                         ← fp8 removed; nemotron-nano renamed
├── cosmos-reason1-7b/                  ← per-GPU profiles + env renames
├── cosmos-reason2-8b/
├── gpt-oss-20b/
├── llama-3.3-nemotron-super-49b-v1.5/
├── nemotron-3-nano/
├── nemotron-nano-v2/                   ← renamed
└── qwen3-vl-8b-instruct/

Removed: nvidia-nemotron-nano-9b-v2-fp8/, fallback-override.env, hw-OTHER*.env (per model)
```

| Naming | Upstream → SOP |
|---|---|
| `*-shared.env` | `*-full.env` |
| `hw-RTXPRO6000BW*` | `hw-RTX6000PROBW*` |
| `hw-OTHER*` | removed |
| `nvidia-nemotron-nano-9b-v2/` | `nemotron-nano-v2/` |
| `nvidia-nemotron-nano-9b-v2-fp8/` | removed |

## Verification

Run `./scripts/nim/verify.sh` (a thin wrapper that delegates to `scripts/verify_build.py --component nim`, the single source of truth) to confirm:

1. `fallback-override.env` removed
2. No `hw-OTHER*` files remain
3. `nemotron-nano-v2/compose.yml` exists
4. `nvidia-nemotron-nano-9b-v2-fp8/` directory removed
5. Sample of `-full.env` files (5 entries)

---

## License

Use of this skill is governed by the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en) and the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
