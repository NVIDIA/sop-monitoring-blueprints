---
name: sop-by-action-eval
description: Use when running by-action VLM evaluation (per-action-clip inference + accuracy metrics) against the BP evaluation-ms HTTP API. Invoked as /sop-by-action-eval <inputs.yaml> [natural language parameter overrides]
license: "CC-BY-4.0 AND Apache-2.0"
---

# By-Action Evaluation (API-driven)

Run per-action VLM evaluation by POSTing to the BP `evaluation-ms` HTTP service. The service decides on its own where to write output (a uuid-named directory under `<RESULTS_ROOT>/<eval_job_id>/`), runs the inference subprocess, and updates Postgres + a job cache. This skill submits the request, polls until terminal, resolves the host-side output directory, and emits a structured JSON envelope on stdout.

When this skill is invoked, follow the steps below in order. Stop immediately if any step fails — show the full error output to the user, but do NOT attempt to fix it automatically.

**Bundled resources** (paths relative to this skill's directory):
- `scripts/eval_api_client.py` — single Python helper that POSTs, polls, resolves paths, prints the envelope.
- `references/inputs-template.yaml` — configuration template with all parameters documented.

**Path conventions**:
- `SKILL_DIR` = this skill's `scripts` directory (absolute path resolved at invocation time).

## Step 0: Parse Overrides (if any)

If the user provided natural language parameter overrides alongside the inputs.yaml:

1. Parse the overrides. If any part of the input cannot be clearly mapped to a yaml field, list the ambiguous parts and ask the user to clarify before proceeding. Map to the yaml structure (see `references/inputs-template.yaml` for the full list):
   - Required: `training_job_id`, `val_dataset_id`
   - Optional: `eval_host`, `eval_port`, `host_results_root`, `backend`, `fps`, `temperature`, `top_p`, `checkpoint_step`, `resolution_config`, `gpu_id`, `poll_interval_sec`, `timeout_sec`

2. Write the overrides to `/tmp/by_action_overrides.yaml` using the Write tool. The api client merges them onto the base inputs.yaml at invocation time — no separate merge step.

3. Show the user what was overridden.

If no overrides, skip the overrides file.

## Step 1: Preflight — eval-ms reachability

Before submitting, make sure the eval-ms service is up. Read `eval_host` and `eval_port` from inputs.yaml (defaults: `localhost:32090`) and probe the health endpoint:

```bash
curl -fsS http://${EVAL_HOST:-localhost}:${EVAL_PORT:-32090}/health
```

- **Connection refused / non-2xx** → stop and tell the user to bring eval-ms up (`docker compose up evaluation-ms` from the BP deployment root) before retrying.
- **2xx** → continue.

## Step 2: Submit + Poll

`eval_api_client.py` is a **blocking** call. It POSTs the request, polls `/api/v1/evaluation/status/{eval_job_id}` every `poll_interval_sec` (default 20s) until terminal status (`completed` / `failed` / `cancelled`) or `timeout_sec` (default 3600), prints progress to stderr, and emits a single JSON envelope on the last stdout line. The script returns only when eval-ms reports a terminal state.

```bash
python3 SKILL_DIR/eval_api_client.py by-action <inputs.yaml> [--overrides /tmp/by_action_overrides.yaml]
```

### Choosing run_in_background vs synchronous

The client is blocking either way — `run_in_background` only governs how *you* (the caller) wait for it.

- **Interactive Claude Code (main agent)** — use `run_in_background: true` so the main loop is free for other work; you receive a completion notification carrying the script's stdout (the envelope).
- **Subagent or any non-interactive context** — call **synchronously** (do NOT pass `run_in_background: true`). A subagent has no completion-notification channel; a backgrounded process is killed when the subagent's bash session ends and the envelope is lost. Block the agent's bash call until the script returns. The Bash tool's default timeout may need to be raised via the `timeout` parameter (e.g. `600000` ms = 10 min) for typical eval runs.

In both cases the envelope appears on stdout once the script exits; that is the only thing the next step needs.

Tell the user the evaluation is running and you will report when it finishes.

**IMPORTANT: Do NOT poll, monitor, or periodically read the log file while waiting.** The script's own poll loop is the only one needed. Do not use `Read` or `Bash` to check progress — wait for the script to return (or, in the interactive case, for the background-completion notification).

## Step 3: Report Results

When the background command completes:

- **Failure** (non-zero exit, or stdout envelope has `"status": "failed"`):
  - Parse the last stdout line as JSON. If `host_output_dir` is present, read `<host_output_dir>/log.txt` (last 50 lines) and show the error.
  - Also surface the `error` field from the envelope.

- **Success** (`"status": "completed"`):
  - Parse the JSON envelope from stdout. The envelope carries `headline_metrics.overall_accuracy` (0.0–1.0) directly — that is the by-action accuracy without any extra parsing.
  - Cite `host_output_dir`, `artifacts.inference_results_json`, and `artifacts.log` so the user can navigate to the raw outputs.
  - Tell the user that any downstream RCA invocation should pass `<host_output_dir>/log.txt` (or `inference_results.json`) as the by-action input.

Do NOT load `inference_results.json` directly into the context — it can be large.

## Reference

### inputs.yaml Format

See `references/inputs-template.yaml` for the canonical template with comments. Minimum required fields:

```yaml
training_job_id: <uuid>     # registered training job (eval-ms validates it's completed)
val_dataset_id: <uuid>      # validation dataset id (actions.json is resolved by eval-ms)
host_results_root: /abs/path/to/results   # docker-compose maps this -> /workspace/sop-eval-ms/assets/results
```

All other fields have sensible defaults; see template for details.

### Eval-ms request body

| inputs.yaml field | Mapped to request body | Default |
|---|---|---|
| `training_job_id` | `training_job_id` | required |
| `val_dataset_id` | `val_dataset_id` | required |
| `fps` | `fps` | 8 |
| `temperature` | `temperature` | 0.0 |
| `top_p` | `top_p` | 1.0 |
| `backend` | `backend` | `vllm` |
| `checkpoint_step` | `checkpoint_step` | latest |
| `resolution_config` | `resolution_config` | training-mirror defaults |
| `gpu_id` | `gpu_id` | all visible GPUs |

`backend: transformers` is required when evaluating LoRA-only checkpoints — vLLM's tokenizer/config compatibility breaks on some Qwen3-VL releases when `flash_attn` is not installed.

### Outputs

The eval-ms service writes everything to `<host_results_root>/<eval_job_id>/` (host-side path, resolved by this skill from the docker-compose volume mapping):

| File | Description |
|------|-------------|
| `inference_results.json` | `{video: [[gt_action, vlm_response, chunk_path], ...]}` — 3-tuple format consumed by `analyze_by_action_confusion.py` |
| `log.txt` | Full inference log with `Args: {...}` line, per-chunk `Action Chunk: <path>` markers, and the final accuracy summary |

### Structured JSON envelope (stdout)

```json
{
  "mode": "by-action",
  "eval_job_id": "...",
  "status": "completed",
  "host_output_dir": "/abs/host/path/to/<eval_job_id>",
  "container_output_dir": "/workspace/sop-eval-ms/assets/results/<eval_job_id>",
  "artifacts": {
    "inference_results_json": "<host_output_dir>/inference_results.json",
    "log": "<host_output_dir>/log.txt"
  },
  "headline_metrics": {
    "overall_accuracy": 0.95
  },
  "error": null
}
```

`headline_metrics.overall_accuracy` is the fraction of chunks the VLM classified correctly across the dataset, mirroring `evaluation_job.overall_accuracy` in the DB. The eval-ms grader is strict (requires the full `(N)` token list to match); when the VLM emits multi-action responses the helper `sop-rca-plugin/.../analyze_by_action_confusion.py` (first-`(N)` match) may report higher accuracy — prefer it for orchestrator success-criteria.

The orchestrator (or any caller) reads this envelope to record `host_output_dir` and downstream artifact paths.

### Troubleshooting

- **HTTP 400 "Training job not found"**: `training_job_id` does not exist in the eval-ms Postgres. Confirm the ID via the BP UI or via `GET /api/v1/training/all_jobs` on the training service.
- **HTTP 400 "Training job ... is not completed"**: wait for training to finish.
- **HTTP 400 "actions.json not found"**: the registered dataset directory is missing `actions.json`. Fix the dataset; eval-ms cannot proceed without it.
- **HTTP 400 "An evaluation is already running"**: eval-ms allows one by-action eval at a time. Cancel the running one (`POST /api/v1/evaluation/cancel/{eval_job_id}`) or wait.
- **Timeout** (envelope `"status": "timeout"`): the job has not reached a terminal state within `timeout_sec`. The job may still be running on eval-ms — check `GET /api/v1/evaluation/status/{eval_job_id}` manually before retrying.
- **CUDA OOM** (eval-ms subprocess log): lower `fps` or pass a `resolution_config` with lower `total_pixels` / `max_frames`.
