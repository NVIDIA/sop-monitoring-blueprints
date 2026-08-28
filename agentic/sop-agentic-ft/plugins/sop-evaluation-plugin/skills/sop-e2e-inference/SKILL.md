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
   - Optional: `eval_host`, `eval_port`, `host_results_root`, `backend`, `fps`, `temperature`, `top_p`, `checkpoint_step`, `ddm_checkpoint`, `resolution_config`, `gpu_id`, `score_threshold`, `nms_sec`, `ddm_batch_size`, `frames_per_segment_hint`, `chunking_algorithm`, `stride_sec`, `smooth_min_seg_sec`, `smooth_min_vote`, `non_sop_action`, `poll_interval_sec`, `timeout_sec`

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

Add `stride_sec: <float>` on top of that for **overlapping** windows: one window of
`chunk_length_sec` every `stride_sec` instead of a non-overlapping grid. Reach for it when
actions are short relative to the chunk length — on a fixed grid an action straddling a
boundary lands half in each chunk and is diluted in both, while overlapping windows guarantee
at least one window contains it whole. The cost is `chunk_length_sec / stride_sec` times more
VLM calls, so 3s/1s is 3x the inference of 3s alone.

**Prefer `backend: vllm` for an overlapping run.** It is materially faster per window than
transformers, and an overlapping run is dominated by window count. Note the service runs one
e2e job at a time — a global guard, not per-GPU — so a single run cannot be sharded across
GPUs and is bounded by one device.

**Check `chunk_length_sec` against `resolution_config.max_frames`.** A chunk needs
`chunk_length_sec x fps` frames; anything above `max_frames` is subsampled, so the model
sees a thinned window rather than the window. At the common `fps: 8, max_frames: 40` that
ceiling is 5 seconds — fine for a 3s window, silently lossy at 6s. The stage warns when it
happens.

`max_model_len` defaults to 32768 so vLLM starts out of the box. Left to its own derivation,
vLLM sizes the KV cache from the checkpoint's declared context -- 262144 on Qwen3-VL, needing
~36 GiB, more than an 80 GiB card has free -- and the engine fails before any inference. 32768
needs ~4.5 GiB and clears the ~20.4k a default request uses.

Adjust it when the default does not fit: raise it if requests are rejected for length (vision
tokens are `total_pixels / (patch_size * spatial_merge_size)^2`, so raising `total_pixels`
raises the requirement), lower it if the engine cannot start on a smaller card.

Overlapping windows predict the same instant several times, so the service votes them back
down to a non-overlapping sequence before scoring — per time bin, ties going to the non-SOP
action. That action comes from `non_sop_action` if you set it, otherwise from
`actions_can_be_skipped` in `actions.json`. **Many datasets do not declare that field** — without a non-SOP action the windows cannot be collapsed at all, so the run is refused rather than returning metrics computed on raw overlapping windows. If you hit that error, pass `non_sop_action`; it is the same 1-based index the augmentation config uses. `smooth_min_vote` sets how many
overlapping windows must agree before an action is accepted, and `smooth_min_seg_sec` drops
action segments shorter than the given duration. Both raw and smoothed predictions are written
(`video_name_to_output_text.json` and `video_name_to_output_text_smoothed.json`), so the
windows remain inspectable. `stride_sec` is rejected with `chunking_algorithm: ddm`, and must be `<= chunk_length_sec` — a wider stride would leave gaps that no window covers, which would be scored as non-SOP.

### Eval-ms request body

| inputs.yaml field | Mapped to request body | Default |
|---|---|---|
| `training_job_id` | `training_job_id` | required |
| `val_dataset_id` | `val_dataset_id` | required |
| `ddm_training_job_id` | `ddm_training_job_id` | required when `chunking_algorithm=ddm` |
| `ddm_checkpoint` | `ddm_checkpoint` | latest under `ddm_training_job_id` |
| `chunking_algorithm` | `chunking_algorithm` | `ddm` |
| `chunk_length_sec` | `chunk_length_sec` | required when `chunking_algorithm=uniform` |
| `stride_sec` | `stride_sec` | unset (non-overlapping); uniform chunking only |
| `smooth_min_seg_sec` | `smooth_min_seg_sec` | `2.0`; used only with `stride_sec` |
| `smooth_min_vote` | `smooth_min_vote` | `1`; used only with `stride_sec` |
| `non_sop_action` | `non_sop_action` | inferred from `actions_can_be_skipped`; **required** with `stride_sec` when the dataset omits it |
| `max_model_len` | `max_model_len` | `32768`; vLLM backend only — raise if requests are rejected for length, lower if the engine cannot start |
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
- **HTTP 422 "stride_sec applies to chunking_algorithm='uniform' only"**: drop `stride_sec`, or switch to uniform chunking.
- **HTTP 422 "stride_sec must be <= chunk_length_sec"**: the stride is the step between window starts, so it has to be no larger than the window itself or the windows stop overlapping and start leaving gaps.
- **Overlapping run scores worse than expected**: check the log line reporting how many windows collapsed to how many segments. Everything collapsing to non-SOP usually means `smooth_min_vote` is too high for the stride, or `smooth_min_seg_sec` exceeds the actions' real duration.
- **HTTP 400 "Training/DDM job not found / not completed"**: confirm both training and DDM training jobs are in `completed` status before submitting.
- **HTTP 400 "An e2e evaluation is already running"**: eval-ms allows one e2e job at a time. Cancel the running one (`POST /api/v1/e2e-evaluation/cancel/{eval_job_id}`) or wait.
- **DDM under-segmentation** (low `temporal_segmentation.avg_f1`): try lowering `score_threshold` (e.g. 0.5 → 0.4). Re-evaluate before recommending DDM retraining.
- **Sequence accuracy collapse but per-action OK**: VLM-side issue (look in `outputs_action_recognition/video_name_to_output_text.json`); rerun by-action on the same checkpoint for confirmation.
- **CUDA OOM** during VLM stage: lower `resolution_config.total_pixels` or `resolution_config.max_frames`, or reduce `fps`.
- **Timeout** (envelope `"status": "timeout"`): the job may still be running on eval-ms. Check `/status` manually before retrying.
