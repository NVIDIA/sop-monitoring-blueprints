---
name: sop-ddm-finetuning
description: Fine-tune DDM-Net temporal boundary detector for SOP monitoring. Use when you need to launch and monitor a DDM-Net training run with a given dataset ID.
argument-hint: <dataset_id>
license: "CC-BY-4.0 AND Apache-2.0"
---

# SOP DDM-Net Fine-tuning

You are performing DDM-Net fine-tuning for an SOP (Standard Operating Procedure) monitoring system. DDM-Net is a dual-domain matching temporal boundary detector built on ResNet-50 that learns to segment SOP video into procedural steps.

Your job is to validate the environment, launch training, monitor it to completion, and write a training report. Training logs are persisted by the service at `assets/results/<job_id>/log.txt`.

## Input

The user provides a `dataset_id` as `$ARGUMENTS`. This is the subdirectory name under `assets/data/` containing the training videos and annotations.

If no argument is provided, list available datasets and ask the user which to use:
```bash
ls assets/data/
```

**IMPORTANT — also ask for a `validation_dataset_id`.** The DDM service's `/api/v1/fine-tuning/start` endpoint accepts a separate `validation_dataset_id` query parameter. If omitted, the service **silently defaults `validation_dataset_id = dataset_id`**, which causes training-time `val/f1_score` to be measured on the literal training set (the service generates byte-identical `ddm_train_annotation.json` and `ddm_val_annotation.json` from the same source). The resulting val/F1 number does not reflect generalization and is unreliable for early-stop decisions. Always pass a truly held-out dataset as `validation_dataset_id`. If the user has no held-out dataset prepared, **warn loudly** and continue only after the user confirms they understand the limitation (see "Known Limitation" below).

For reference on model architecture, log format, API endpoints, and config parameters, see `${CLAUDE_SKILL_DIR}/reference.md`.

## Known Limitation

**The DDM service silently uses the training set as the validation set when `validation_dataset_id` is not provided.** This is a service-side default in `microservices/ddm_training_ms/app.py`:

```python
if validation_dataset_id is None or validation_dataset_id == "":
    validation_dataset_id = dataset_id   # silent fallback
```

Both `generate_ddm_annotation` calls then resolve to the same dataset path and produce identical `ddm_train_annotation.json` and `ddm_val_annotation.json`. The reported `val/f1_score` will therefore be inflated and you cannot judge whether the resulting DDM checkpoint generalizes to unseen videos until E2E evaluation runs.

**Mitigation:** always pass a separate held-out `validation_dataset_id` (typically the same dataset you intend to use for E2E evaluation). Import that dataset first via `scripts/import_dataset.sh` so its per-video annotations are available to the service.

## Defaults

- **BASE_URL:** `http://localhost:${DDM_TRAINING_BACKEND_PORT:-32100}` (resolve `DDM_TRAINING_BACKEND_PORT` from `.env` if present; fall back to 32100)
- **RESULTS_ROOT:** `assets/results/` (host) ↔ `/workspace/sop-ddm-ftms/assets/results/` (container)
- **DATA_ROOT:** `assets/data/` (host) ↔ `/workspace/sop-ddm-ftms/assets/data/` (container)
- **CONFIG:** `assets/config/ddm_train_config.yaml` — users edit this directly to tune `batch_size`, `resolution`, `epochs`, `num_gpus`, `workers`. Do NOT edit `anno_path`, `data_root`, `output`, `exp_name` (auto-set by the service).

## Fine-tuning Procedure

### Phase 1: Pre-flight Checks

Before starting training, verify environment readiness:

1. **Start the training service** from the training blueprint root (the session's working directory — do NOT cd into any subdirectory):
   ```bash
   docker compose up -d ddm-training-microservice
   ```
   The top-level `docker-compose.yml` includes the database dependency (`metadata_db`). Running from a subdirectory (e.g. `microservices/ddm_training_ms/`) will miss it.

   Then verify health by polling until ready (retry a few times with short delays):
   ```bash
   curl -s <base_url>/health
   ```
   If the service fails to become healthy after retries, report the error and stop.

2. **Check for running jobs**:
   ```bash
   curl -s <base_url>/api/v1/fine-tuning/all_jobs
   ```
   If any job has status `running` or `queued`, warn the user and ask whether to wait or cancel before proceeding.

3. **Verify the dataset directory** at `assets/data/<dataset_id>/`:

   Expected structure:
   ```
   assets/data/<dataset_id>/
   ├── <video_id>.mp4
   ├── <video_id>/
   │   └── <video_id>_annotation.json
   ├── <video_id2>.mp4
   ├── <video_id2>/
   │   └── <video_id2>_annotation.json
   └── ...
   ```

   Check video and annotation counts:
   ```bash
   ls assets/data/<dataset_id>/*.mp4 2>/dev/null | wc -l
   find assets/data/<dataset_id> -name "*_annotation.json" | wc -l
   ```
   Expected: video count == annotation count, both > 0.

   **Auto-fix policy:** for obvious, non-destructive, reversible issues, fix them in place and announce what you did — do NOT prompt for confirmation. Hard failures (empty directory, missing directory) still stop with an error message.

   - **Uppercase `.MP4` files** — the dataset generator only globs `*.mp4`; uppercase files are silently skipped. Auto-rename to lowercase:
     ```bash
     UPPER=$(ls assets/data/<dataset_id>/*.MP4 2>/dev/null | wc -l)
     if [ "$UPPER" -gt 0 ]; then
       for f in assets/data/<dataset_id>/*.MP4; do mv "$f" "${f%.MP4}.mp4"; done
       echo "auto-fixed: renamed $UPPER files .MP4 → .mp4"
     fi
     ```
   - **Annotation filename stem mismatch** — each video `<stem>.mp4` expects a sibling `<stem>/<stem>_annotation.json`. If an annotation file exists in the video's subdirectory but its stem doesn't match (e.g. `Install 1_annotation.json` next to `Install_1.mp4`), rename the annotation to match the video stem and announce the change. Only do this when exactly one annotation file per subdirectory and one video stem per directory — otherwise stop with an error.
   - **Count mismatch (videos ≠ annotations) that is NOT fixable by renaming** — report the specific missing annotations and stop.
   - **Empty or missing directory** — stop and report the path is wrong. Suggest `ls assets/data/` to show available dataset IDs.

4. **Check GPU availability**:
   ```bash
   nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
   ```
   Read `num_gpus` from `assets/config/ddm_train_config.yaml` (`training_config.num_gpus`). Verify GPU count ≥ `num_gpus`. Warn if free GPU memory looks low — suggest the user reduce `batch_size` in the config if OOM is likely.

Report all pre-flight results before proceeding. **For auto-fixable issues** (uppercase `.MP4`, annotation stem mismatches), apply the fix in place and announce what you did — do NOT prompt for confirmation. **For hard failures** (service unreachable, insufficient GPUs, empty dataset directory), stop and tell the user what to fix.

### Phase 2: Training and Monitoring

1. **Print a one-block effective summary** for traceability (no confirmation prompt):
   ```
   === DDM Training — Effective Config ===
   dataset_id : <dataset_id>
   host path  : assets/data/<dataset_id>    (videos: N, annotations: N)
   batch_size : N    resolution: N
   num_gpus   : N    workers: N
   epochs     : N
   ```
   Values come from `assets/config/ddm_train_config.yaml`. Flag any parameter outside the normal ranges in `reference.md` as a one-line note underneath (e.g. `note: epochs=200 is above typical 10–50 range`).

2. **Start training**:
   ```bash
   curl -s -X POST "<base_url>/api/v1/fine-tuning/start?dataset_id=<dataset_id>&validation_dataset_id=<validation_dataset_id>"
   ```
   **Always include `validation_dataset_id`** — omitting it triggers the silent fallback described in "Known Limitation" and produces an unreliable `val/f1_score`. The held-out validation dataset must be a different dataset that has been imported into the BP (run `scripts/import_dataset.sh <val_dataset_path>` first). Record the `job_id` from the response.

3. **Locate the log file**: `assets/results/<job_id>/log.txt`.

4. **Monitor training progress** via the status endpoint:
   ```bash
   curl -s "<base_url>/api/v1/fine-tuning/status/<job_id>"
   ```
   Extract: `status`, `progress`, `current_step`, `total_steps`, `loss`.

5. **Parse val/f1_score from the log** alongside status polling (the status API may not include this metric):
   ```bash
   grep -E "Epoch [0-9]+, global step [0-9]+: 'val/f1_score'" assets/results/<job_id>/log.txt | tail -5
   ```

6. **Report progress every 10%** with:
   - Current epoch / total epochs and progress percentage
   - Latest `val/f1_score` and best `val/f1_score` seen so far
   - Any anomalies detected

7. **Monitor for anomalies** periodically:
   ```bash
   # OOM check
   grep -i "out of memory\|CUDA error\|OOM" assets/results/<job_id>/log.txt | tail -3
   # Error check
   grep -iE "^ERROR|exception|traceback" assets/results/<job_id>/log.txt | tail -5
   ```

8. **Wait for completion**. When status becomes `completed`, `failed`, or `cancelled`, proceed to Phase 3.

### Phase 3: Summary

After training completes, collect metrics and write a training report.

#### 3a: Collect Final Metrics

1. **Parse val/f1_score history** from the log:
   ```bash
   grep -E "Epoch [0-9]+, global step [0-9]+: 'val/f1_score'" assets/results/<job_id>/log.txt
   ```
   Extract: best `val/f1_score`, the epoch it was achieved, and the final epoch's score.

2. **Collect checkpoint paths** saved during training:
   ```bash
   grep "saving model to" assets/results/<job_id>/log.txt | tail -5
   ```

#### 3b: Write Training Report

Save a formatted training report to `assets/results/<job_id>/training_report.md`:

```markdown
# DDM-Net Training Report

## Overview
| Field | Value |
|-------|-------|
| Model | DDM-Net (ResNet-50, dual-domain matching) |
| Dataset | <dataset_id> |
| Job ID | <job_id> |
| Status | completed / failed / cancelled |
| Duration | Xh Xm |

## Training Metrics
| Metric | Value |
|--------|-------|
| Best val/f1_score | X.XXXXX (epoch N) |
| Final val/f1_score | X.XXXXX |
| Total Epochs | N |

## Config
| Parameter | Value |
|-----------|-------|
| Batch Size | N |
| Resolution | N |
| Num GPUs | N |
| Workers | N |
| Epochs | N |

## Checkpoints
- <list checkpoint paths from log>

## Anomalies
- List any OOM events, CUDA errors, or exceptions detected during training.
- "None" if no anomalies detected.

## Artifacts
- **Log:** `assets/results/<job_id>/log.txt`
- **Report:** `assets/results/<job_id>/training_report.md`
```

#### 3c: Output Summary

Print the training report content to the user. If training failed, include the relevant error lines from the log.
