---
name: sop-e2e-inference
description: Use when running the e2e evaluation pipeline (temporal segmentation + action recognition + accuracy) against the BP evaluation-ms HTTP API. Invoked as /sop-e2e-inference <inputs.yaml> [natural language parameter overrides]
license: "CC-BY-4.0 AND Apache-2.0"
---

# E2E Inference Pipeline (API-driven)

Run the end-to-end DDM + VLM pipeline by POSTing to the BP `evaluation-ms` HTTP service. Eval-ms runs DDM temporal segmentation, then VLM action recognition, then scores accuracy — and writes everything to a uuid-named directory under `<RESULTS_ROOT>/<eval_job_id>/`. This skill submits the request, polls until terminal, resolves the host-side output directory, and emits a structured JSON envelope on stdout.

When this skill is invoked, follow the steps below in order. Stop immediately if any step fails — show the full error output to the user, explain what likely went wrong and how to fix it, but do NOT attempt to fix it automatically.

**Bundled resources** (paths relative to this skill's directory):
- `scripts/eval_api_client.py` — single Python helper that POSTs, polls, resolves paths, prints the envelope.
- `references/inputs-template.yaml` — configuration template with all parameters documented.

**Path conventions**:
- `SKILL_DIR` = this skill's `scripts` directory (absolute path resolved at invocation time).

## Step 0: Parse Overrides (if any)

If the user provided natural language parameter overrides alongside the inputs.yaml:

1. Parse the overrides. If any part of the input cannot be clearly mapped to a yaml field, list the ambiguous parts and ask the user to clarify before proceeding. Map to the yaml structure (see `references/inputs-template.yaml`):
   - Required: `training_job_id`, `val_dataset_id`, and (when `chunking_algorithm=ddm`) `ddm_training_job_id`
   - Required (uniform chunking): `chunk_length_sec` when `chunking_algorithm=uniform`
   - Optional: `eval_host`, `eval_port`, `host_results_root`, `backend`, `fps`, `temperature`, `top_p`, `checkpoint_step`, `ddm_checkpoint`, `resolution_config`, `gpu_id`, `score_threshold`, `nms_sec`, `ddm_batch_size`, `frames_per_segment_hint`, `chunking_algorithm`, `poll_interval_sec`, `timeout_sec`

2. Write the overrides to `/tmp/e2e_overrides.yaml`. The api client merges them on top of inputs.yaml at invocation time.

3. Show the user what was overridden.

If no overrides, skip the overrides file.

## Step 1: Preflight — eval-ms reachability

Probe the health endpoint before submitting:

```bash
curl -fsS http://${EVAL_HOST:-localhost}:${EVAL_PORT:-32090}/health
```

- **Non-2xx** → stop and tell the user to bring eval-ms up (`docker compose up evaluation-ms`).
- **2xx** → continue.

## Step 2: Submit + Poll

`eval_api_client.py` is a **blocking** call. It POSTs the request, polls `/api/v1/e2e-evaluation/status/{eval_job_id}` every `poll_interval_sec` (default 20s) until terminal status (`completed` / `failed` / `cancelled`) or `timeout_sec` (default 3600), prints progress to stderr, and emits a single JSON envelope on the last stdout line. It also validates two cross-field rules client-side before sending: `ddm_training_job_id` is required when `chunking_algorithm=ddm`; `chunk_length_sec` is required when `chunking_algorithm=uniform`.

```bash
python3 SKILL_DIR/eval_api_client.py e2e <inputs.yaml> [--overrides /tmp/e2e_overrides.yaml]
```

### Choosing run_in_background vs synchronous

The client is blocking either way — `run_in_background` only governs how *you* (the caller) wait for it.

- **Interactive Claude Code (main agent)** — use `run_in_background: true`; you receive a completion notification carrying the script's stdout (the envelope).
- **Subagent or any non-interactive context** — call **synchronously** (do NOT pass `run_in_background: true`). A subagent has no completion-notification channel; a backgrounded process is killed when the subagent's bash session ends and the envelope is lost. Block until the script returns. Set the Bash `timeout` parameter generously (e.g. `1800000` ms = 30 min) for an e2e run with DDM + VLM stages.

In both cases the envelope appears on stdout once the script exits; that is the only thing the next step needs.

Tell the user the pipeline is running and you will report when it finishes.

**IMPORTANT: Do NOT poll or read intermediate logs while waiting.** The script's own poll loop is the only one needed.

## Step 3: Report Summary

When the background command completes:

- **Failure** (non-zero exit or envelope `"status": "failed"`):
  - Parse the envelope JSON. If `artifacts.sop_e2e_eval_log` is reachable, read its last 30 lines and show the error.
  - Surface the envelope's `error` field.

- **Success** (`"status": "completed"`):
  - Parse the envelope from stdout. The envelope's `headline_metrics` carries `overall_accuracy` (= chunk-level action accuracy) and `avg_f1` (= DDM temporal-segmentation F1) directly — no extra reads needed for the top-line numbers.
  - For the richer per-video / per-action breakdown, read `artifacts.e2e_results_json` (small — top-level keys: `temporal_segmentation.{avg_f1, avg_precision, avg_recall, per_video}` + `action_recognition.{sequence_accuracy, action_accuracy, total_videos, wrong, duplicate, missing, per_video, per_action}`).
  - Optionally tail `artifacts.sop_e2e_eval_log` (~20 lines) for the final stage summary.
  - Cite `host_output_dir`, `artifacts.accuracy_json`, `artifacts.temporal_segmentation_dir` so the user (or RCA) can navigate to deeper artifacts.

Do NOT load `video_name_to_output_text.json` into context — it can be large.

## Reference

### inputs.yaml Format

See `references/inputs-template.yaml`. Minimum:

```yaml
training_job_id: <training_job_uuid>
val_dataset_id: <val_dataset_uuid>
ddm_training_job_id: <ddm_training_job_uuid>   # required for chunking_algorithm=ddm
host_results_root: /abs/path/to/results
```

Set `chunking_algorithm: uniform` + `chunk_length_sec: <float>` to skip DDM entirely and use fixed-length time slices.

### Eval-ms request body

| inputs.yaml field | Mapped to request body | Default |
|---|---|---|
| `training_job_id` | `training_job_id` | required |
| `val_dataset_id` | `val_dataset_id` | required |
| `ddm_training_job_id` | `ddm_training_job_id` | required when `chunking_algorithm=ddm` |
| `ddm_checkpoint` | `ddm_checkpoint` | latest under `ddm_training_job_id` |
| `chunking_algorithm` | `chunking_algorithm` | `ddm` |
| `chunk_length_sec` | `chunk_length_sec` | required when `chunking_algorithm=uniform` |
| `score_threshold` | `score_threshold` | 0.5 |
| `nms_sec` | `nms_sec` | 0.0 |
| `ddm_batch_size` | `ddm_batch_size` | 8 |
| `frames_per_segment_hint` | `frames_per_segment_hint` | 256 |
| `fps` | `fps` | 8 |
| `temperature` | `temperature` | 0.0 |
| `top_p` | `top_p` | 1.0 |
| `backend` | `backend` | `vllm` |
| `checkpoint_step` | `checkpoint_step` | latest |
| `resolution_config` | `resolution_config` | training-mirror defaults |
| `gpu_id` | `gpu_id` | all visible GPUs |

### Outputs

The eval-ms service writes everything to `<host_results_root>/<eval_job_id>/`:

| File | Description |
|------|-------------|
| `e2e_results.json` | Combined summary (frontend-facing); `temporal_segmentation.*` + `action_recognition.*` blocks |
| `outputs_temporal_segmentation/f1_<thr>.json` | DDM predicted boundaries with F1/precision/recall per video. `<thr>` is a fixed evaluation-side tolerance (typically `0.95`); glob with `f1_*.json` for a stable filename. |
| `outputs_temporal_segmentation/video_to_boundaries_debug.json` | Golden boundaries |
| `outputs_temporal_segmentation/video_to_ddm_info_debug.json` | DDM per-frame scores, video fps, duration |
| `outputs_temporal_segmentation/<video>.png` | DDM boundary visualization plots |
| `outputs_temporal_segmentation/temporal_segmentation.log` | DDM stage log with `Args: Namespace(...)` |
| `outputs_action_recognition/accuracy.json` | Per-video errors, sequence/action accuracy, wrong/duplicate/missing breakdown |
| `outputs_action_recognition/video_name_to_output_text.json` | VLM output text per DDM-segmented chunk |
| `outputs_action_recognition/action_recognition_multi_gpu.log` | VLM stage log with `Args: Namespace(...)` |
| `sop_e2e_eval_log.txt` | Driver log spanning both stages |
| `log.txt` | Combined log (DDM + VLM) |

### Structured JSON envelope (stdout)

```json
{
  "mode": "e2e",
  "eval_job_id": "...",
  "status": "completed",
  "host_output_dir": "/abs/host/path/to/<eval_job_id>",
  "container_output_dir": "/workspace/sop-eval-ms/assets/results/<eval_job_id>",
  "artifacts": {
    "e2e_results_json": "<host_output_dir>/e2e_results.json",
    "accuracy_json": "<host_output_dir>/outputs_action_recognition/accuracy.json",
    "video_name_to_output_text_json": "<host_output_dir>/outputs_action_recognition/video_name_to_output_text.json",
    "action_recognition_log": "<host_output_dir>/outputs_action_recognition/action_recognition_multi_gpu.log",
    "temporal_segmentation_dir": "<host_output_dir>/outputs_temporal_segmentation",
    "temporal_segmentation_log": "<host_output_dir>/outputs_temporal_segmentation/temporal_segmentation.log",
    "sop_e2e_eval_log": "<host_output_dir>/sop_e2e_eval_log.txt",
    "log": "<host_output_dir>/log.txt"
  },
  "headline_metrics": {
    "overall_accuracy": 0.93,
    "avg_f1": 0.95
  },
  "error": null
}
```

`headline_metrics`:
- `overall_accuracy` — chunk-level VLM match rate over DDM-segmented chunks (informational; inflates the denominator when DDM over-segments, so not the same as action-level accuracy).
- `avg_f1` — DDM temporal-segmentation F1 averaged across videos.

For the action-level / sequence-level numbers the orchestrator and RCA care about (`action_accuracy`, `sequence_accuracy`, `wrong/duplicate/missing`), read `e2e_results.json.action_recognition.*` — it's a small file.

### Troubleshooting

- **HTTP 400 "ddm_training_job_id is required"**: provide a DDM training job UUID, or switch to `chunking_algorithm: uniform`.
- **HTTP 400 "chunk_length_sec is required"**: set `chunk_length_sec` (e.g. 12.0) when using `chunking_algorithm: uniform`.
- **HTTP 400 "Training/DDM job not found / not completed"**: confirm both training and DDM training jobs are in `completed` status before submitting.
- **HTTP 400 "An e2e evaluation is already running"**: eval-ms allows one e2e job at a time. Cancel the running one (`POST /api/v1/e2e-evaluation/cancel/{eval_job_id}`) or wait.
- **DDM under-segmentation** (low `temporal_segmentation.avg_f1`): try lowering `score_threshold` (e.g. 0.5 → 0.4). Re-evaluate before recommending DDM retraining.
- **Sequence accuracy collapse but per-action OK**: VLM-side issue (look in `outputs_action_recognition/video_name_to_output_text.json`); rerun by-action on the same checkpoint for confirmation.
- **CUDA OOM** during VLM stage: lower `resolution_config.total_pixels` or `resolution_config.max_frames`, or reduce `fps`.
- **Timeout** (envelope `"status": "timeout"`): the job may still be running on eval-ms. Check `/status` manually before retrying.
