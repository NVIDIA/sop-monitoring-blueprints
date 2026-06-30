---
name: sop-data-augmentation
description: Use when the user wants to run data augmentation on an annotated dataset, configure augmentation parameters, check augmentation status, or understand what each QA augmentation type does (BCQ, MCQ, GQA, DMCQ, DSQA, ENQA)
license: "CC-BY-4.0 AND Apache-2.0"
---

# SOP Data Augmentation

## Overview

The Data and QA Augmentation microservice transforms annotated video data into structured question-answer (QA) formats used to train a Vision-Language Model (VLM) for SOP monitoring. It takes an annotated dataset (video action chunks with timestamps) and generates up to 7 types of QA training data — from simple binary yes/no questions to complex hard-negative mining with shuffled video frames.

This skill covers how to configure, trigger, monitor, and troubleshoot the augmentation pipeline.

**Bundled resources** (paths relative to this skill's directory):
- `scripts/launch_vllm.sh` — Helper to start/stop a local vLLM server for GQA generation
- `references/augmentation-types.md` — Detailed config parameters and guidance for all 7 augmentation types
- `references/config-template.md` — Full annotated `augment_config.yaml` template
- `references/launch-vllm-usage.md` — vLLM script options and Docker details

## Prerequisites

Before running augmentation, ensure:

1. **Service is running:** The `sop-data-gen` container must be up on port **5487**
   ```bash
   # Verify service health
   curl http://<server_ip>:5487/health
   # Expected: {"message": "VLM Data Augmentation API is running", "status": "healthy"}
   ```

2. **Annotated dataset exists:** The dataset must be in `assets/data/<dataset_id>/` with:
   - `actions.json` — action definitions (e.g., `{"actions": ["(1) standing by...", "(2) installing..."]}`)
   - One or more video folders (e.g., `video-1-pass/`) each containing:
     - Action video chunks (mp4 files named `<action_number>_<video_name>_<duplication>_<timeline>.mp4`)
     - `<video_name>_annotation.json` — frame-by-frame annotations

3. **Config file prepared:** `assets/config/augment_config.yaml` configured with desired parameters

4. **LLM access (for GQAs stage only):** The GQAs stage requires an LLM. Two options:
   - **NVIDIA NIM API:** Set `NGC_API_KEY` in `.env` and configure `llm_type: "nvidia"` in the config
   - **Local LLM (no API key needed):** Use the helper script `scripts/launch_vllm.sh` to start a local vLLM server, then configure `llm_type: "local"` in the config. See the workflow below for details.

## Workflow

If the user has an NVIDIA API key (`NGC_API_KEY` set in `.env`), use the NVIDIA NIM path. If not, use the local LLM path — start vLLM before augmentation and stop it after to free GPU memory.
**Note: The `NGC_API_KEY` might set to a placeholder, do check if the value is a valid NVIDIA API key before deciding to use NVIDIA NIM**

### Configuration policy (read before editing the config)

Treat the user's existing `assets/config/augment_config.yaml` as authoritative for *which stages* are enabled. Only modify the *minimum* settings required to satisfy the request.

```
                                     ┌─ Has NGC_API_KEY? ─┐
                                     │                     │
                                    yes                    no
                                     │                     │
                                     │         1a. Start local vLLM
                                     │             scripts/launch_vllm.sh
                                     │             (wait for ready)
                                     │                     │
                                     └──────────┬──────────┘
                                                │
                              1b. Edit augment_config.yaml
                                                │
                              2.  POST /api/v1/augment
                                                │
                              3.  GET /api/v1/augmentation_status
                                  (poll until 100%)
                                                │
                              4.  Verify output
                                                │
                                     ┌─ Used local LLM? ──┐
                                     │                     │
                                    no                    yes
                                     │                     │
                                     │         5. Stop vLLM
                                     │            scripts/launch_vllm.sh --stop
                                     │            (frees GPU for training)
                                     │                     │
                                     └──────────┬──────────┘
                                              Done
```

### Step 1a: Start Local LLM (skip if using NVIDIA NIM API)

If the user does not have an NVIDIA API key, start a local vLLM server **before** editing the config:

```bash
# Default model on port 9000 (takes ~6 min to load)
scripts/launch_vllm.sh

# Or with custom model/port:
scripts/launch_vllm.sh --model Qwen/Qwen2.5-7B --port 8000 --tp 2
```

When ready, the script prints a ready-to-paste `gqas:` config snippet (with the actual served model id, machine IP, and port). Capture that snippet — Step 1b uses it verbatim.

For full script options, see `references/launch-vllm-usage.md`.

### Step 1b: Configure

Edit `assets/config/augment_config.yaml` on the host machine. The file is mounted into the container — no rebuild needed.

**Per the Configuration policy above:** only change fields that are required for the request.

For local LLM, ensure the `gqas` section has:
```yaml
gqas:
  llm_type: "local"
  local_llm_url: "http://<vllm_machine_ip>:<port>/v1"   # from launch_vllm.sh output
  llm: <served_model_id>                                # exact id from /v1/models — do not guess
  enable_thinking: "false"                              # required for thinking-mode models
```

### Step 2: Trigger

```bash
curl -X POST \
  'http://<server_ip>:5487/api/v1/augment?label_data_id=<dataset_id>' \
  -H 'accept: application/json' \
  -d ''
```

The `label_data_id` is the dataset folder name under `assets/data/`. This is the original annotated dataset ID (not an augmented one).

### Step 3: Poll Status

```bash
curl 'http://<server_ip>:5487/api/v1/augmentation_status/<augmented_dataset_id>'
```

The `augmented_dataset_id` is returned by the POST request (format: `<label_data_id>_augmented_<N>`).

### Step 4: Verify Output

Check the output directory exists and contains expected subdirectories:
```bash
ls assets/data/<augmented_dataset_id>/
# Expected: bcq/ mcq/ golden_gqa/ gqas/ (and dmcq/ ds/ en/ if those stages were enabled)
```

### Step 5: Stop Local LLM (skip if using NVIDIA NIM API)

After augmentation completes, stop the vLLM server to free GPU memory for downstream tasks (e.g., VLM fine-tuning, DDM training):

```bash
scripts/launch_vllm.sh --stop
```

## API Reference

### POST /api/v1/augment

Trigger a data augmentation job. Runs asynchronously — returns immediately.

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `label_data_id` | query | yes | ID of the annotated dataset to augment |

**Response (200):**
```json
{
  "dataset_id": "<label_data_id>_augmented_<N>",
  "message": "Augmentation actions submitted successfully"
}
```

**Errors:**
- `400` — Missing `label_data_id`, dataset path not found, `actions.json` missing, or no video folders
- `500` — Internal server error

### GET /api/v1/augmentation_status/{dataset_id}

Poll augmentation progress.

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `dataset_id` | path | yes | The augmented dataset ID (returned by POST) |

**Response (200):**
```json
{
  "dataset_id": "<augmented_dataset_id>",
  "status": "running",
  "progress": 50.0
}
```

**Status values:** `pending` | `running` | `completed` | `failed`

**Progress:** Float 0.0–100.0 (percentage of stages that finished or failed). With default 4 stages: 25% = 1 done (BCQ), 50% = 2 done (+MCQ), 75% = 3 done (+Golden GQA), 100% = all done (+GQAs).

### GET /api/v1/augmented_datasets

List all completed augmented datasets with statistics.

**Response (200):**
```json
{
  "<augmented_dataset_id>": {
    "status": "completed",
    "video_count": 3,
    "total_clips": 12
  }
}
```

### GET /health

Health check.

**Response (200):**
```json
{
  "message": "VLM Data Augmentation API is running",
  "status": "healthy"
}
```

## Augmentation Types

7 types, executed sequentially. Each independently enabled/disabled in config. Default: first 4 enabled.

| # | Type | Default | Purpose | Key Params |
|---|------|---------|---------|------------|
| 1 | **BCQ** (Binary Choice QA) | Enabled | Yes/no questions for action presence/absence | `negative_ratio`, `subject` |
| 2 | **Sequential MCQ** | Enabled | Multi-choice from consecutive action sequences | `max_chunk_len` |
| 3 | **Golden GQA** | Enabled | Template-based Q&A (one per action) | — |
| 4 | **GQAs** (LLM-expanded) | Enabled | LLM generates multiple Q&A variations per action | `llm_type`, `llm`, `num_qa_llm` |
| 5 | **Dynamic MCQ** | Disabled | Hard negative mining with confusable/adjacent actions | `non_sop_action`, `num_pos/neg`, hard modes |
| 6 | **Dynamic Shuffling** | Disabled | Frame-shuffled noise videos as negatives | `non_sop_action`, `num_runs`, `num_hard_neg` |
| 7 | **Extra Negative** | Disabled | Cross-SOP negatives from different datasets | `non_sop_action`, `extra_negative_data_id` |

Stages 5-7 require `non_sop_action` (the "none of the above" action index from your `actions.json`).

For detailed config parameters, examples, hard mode explanations, and guidance on when to enable each type, read `references/augmentation-types.md`.

## Config File Reference

The config file is at `assets/config/augment_config.yaml` (mounted volume — edit on host, no rebuild needed).

Key concepts:
- **`non_sop_action`:** Action index for "none of the above." Required for dynamic_mcq, dynamic_shuffling, extra_negative.
- **`exclude_action`:** Underscore-separated indices to skip (e.g., `"1_2"` excludes actions 1 and 2).
- **`extra_negative_data_id`:** Must be a different annotated dataset ID already in `assets/data/`.

For the full annotated config template with all parameters, read `references/config-template.md`.

## Reference Configurations

These are templates for **first-time setup**, not actions to take when a config already exists. If `assets/config/augment_config.yaml` is already populated, see the Configuration policy in the Workflow section.

### Template: 4-stage starter (BCQ + sequential MCQ + golden GQA + GQAs)

Produces well-rounded training data without requiring `non_sop_action` setup. Use this only when initializing a fresh config.

```yaml
bcq:
  enable: true
sequential_mcq:
  enable: true
golden_gqa:
  enable: true
gqas:
  enable: true
```

### Template: All 7 stages

Enable all stages. Requires `non_sop_action` and (for extra_negative) a second annotated dataset.

Set `non_sop_action` in dynamic_mcq, dynamic_shuffling, and extra_negative to your dataset's "none of the above" action index. Set `extra_negative_data_id` to a different dataset's ID.

Start conservative: `num_pos: 1, num_neg: 2` for dynamic_mcq. `num_runs: 1` for dynamic_shuffling and extra_negative.

### Template: Local LLM for GQAs

To use a self-hosted vllm server instead of NVIDIA NIM. Replace `<served_model_id>` with the exact id reported by `curl <local_llm_url>/models`:

```yaml
gqas:
  enable: true
  llm_type: "local"
  local_llm_url: "http://<llm_server_ip>:<port>/v1"    # Must include /v1
  llm: <served_model_id>                                # exact id from /v1/models — do not guess
  enable_thinking: "false"                              # Set for thinking-mode models
```

## Output Structure

After successful augmentation, a new directory is created:

```
assets/data/<dataset_id>_augmented_<N>/
  bcq/                  # Binary choice QA
    videos/             # Copied video files
    bcq.json            # LLaVA-format annotations
  mcq/                  # Sequential multiple choice QA
    videos/             # Includes merged multi-action videos
    mcq.json
  golden_gqa/           # Golden grounded QA
    videos/
    golden_gqa.json
  gqas/                 # LLM-expanded QA
    videos/
    gqas.json
    GQA2GQAs/           # Raw LLM outputs per action per video
  dmcq/                 # Dynamic MCQ (if enabled)
    videos/
    dmcq.json
  ds/                   # Dynamic shuffling (if enabled)
    videos/             # Contains generated shuffled videos
    ds.json
  en/                   # Extra negative (if enabled)
    videos/
    en.json
```

Each annotation JSON follows LLaVA format:
```json
[
  {
    "id": 0,
    "conversations": [
      {"from": "human", "value": "<video>\n<question>"},
      {"from": "gpt", "value": "<answer>"}
    ],
    "video": "videos/<filename>.mp4"
  }
]
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| 400: "label_data_id is required" | Missing query parameter | Add `?label_data_id=<id>` to URL |
| 400: "Label data path not found" | Dataset folder doesn't exist | Check `assets/data/<id>/` exists |
| 400: "actions.json not found" | Missing action definitions | Ensure `actions.json` is in dataset folder |
| 400: "No video folders found" | No annotated videos | Complete video annotation first |
| All stages FAILED | One stage failed, causing cascade | Check Docker logs for the specific stage error: `docker compose logs sop-data-gen` |
| Output directory missing after failure | Automatic cleanup on failure | The pipeline deletes the output directory when any stage fails |
| GQAs: LLM rate limit (429) | Too many concurrent LLM calls | Switch to local LLM, or retry later |
| GQAs: Empty LLM response | Thinking-mode model (Qwen3 / Qwen3.5 etc.) | Set `enable_thinking: "false"` in config |
| Progress stuck at 0% | Augmentation still initializing | Wait and re-poll; check Docker logs for activity |

**Check Docker logs:**
```bash
docker compose logs sop-data-gen --tail 100 -f
```

## Important Notes

- **Augmentation is async.** The POST API returns immediately with the augmented dataset ID. You must poll `augmentation_status` to track completion.
- **Config is hot-reloadable.** The config YAML is a mounted volume — edit on host, changes take effect on next augmentation run without rebuilding the container.
- **One failure = all fail.** If any enabled stage fails, all stages are marked as failed and the entire output directory is deleted. Check logs for the root cause.
- **Output ID auto-increments.** The augmented dataset ID is `<label_data_id>_augmented_<N>` where N starts at 0 and increments with each new augmentation of the same source dataset.
- **Advanced stages need `non_sop_action`.** Dynamic MCQ, Dynamic Shuffling, and Extra Negative all require the `non_sop_action` index to be correctly set to your dataset's "none of the above" action.
- **Extra Negative needs a second dataset.** The `extra_negative_data_id` must point to a different annotated dataset that is already in `assets/data/`.
- **GQAs is the only LLM-dependent stage.** All other stages are deterministic and run locally without external API calls.
- **Adjust `min_options`/`max_options` to your action count.** For a dataset with N actions, `max_options` should not exceed N.
