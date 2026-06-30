# Evaluation Microservice

Self-contained evaluation service for the SOP Training Service. Exposes two evaluation modes via REST:

- **Per-action-chunk evaluation** (`/api/v1/evaluation/*`) — runs the VLM on each pre-segmented action chunk against the golden action ID derived from the chunk's filename prefix. Isolates VLM accuracy from DDM segmentation quality.
- **End-to-end evaluation** (`/api/v1/e2e-evaluation/*`) — runs the full pipeline: DDM temporal segmentation (or uniform chunking) → VLM action recognition → sequence-level accuracy. Measures real-world performance.

See `api_spec/openapi.json` for the machine-readable contract.

## Service info

| | |
| --- | --- |
| **Port** | `32090` (override with `SERVICE_PORT` env var) |
| **Base URL** | `http://localhost:32090` (direct) or `http://<frontend>/api/evaluation/` (via nginx proxy) |
| **Status** | `GET /health` returns `{"message": "Evaluation Microservice is running"}` |
| **Backends supported** | `vllm` (default, fast, multi-GPU TP), `transformers` (HuggingFace generate, single-GPU fallback) |
| **Models supported** | Cosmos-Reason1 (Qwen2.5-VL) and Cosmos-Reason2 (Qwen3-VL). Backend auto-dispatches by `config.architectures`. |
| **Chunkers (E2E)** | `ddm` (learned segmentation via DDM-Net) and `uniform` (fixed-length time slices) |

## Build

```bash
make build

# Or via docker compose with a custom parallel-compile cap (default MAX_JOBS=8):
docker compose build --build-arg MAX_JOBS=16 evaluation-microservice
```

`MAX_JOBS` caps parallel `nvcc` invocations for `flash-attn` and the DDM-Net CUDA extension during build. The default 8 keeps peak RAM under ~30 GB; tune up only if you have memory headroom.

## Run

From the repository root:

```bash
docker compose up evaluation-microservice
```

The service expects four bind mounts (set in `docker-compose.yml`):
- `${DATASET_ROOT}` → `/workspace/sop-eval-ms/assets/data` — input datasets
- `${PRETRAINED_MODEL_ROOT}` → `/workspace/sop-eval-ms/assets/weights` — base model weights
- `${RESULTS_ROOT}` → `/workspace/sop-eval-ms/assets/results` — output dir (eval job results land in `<RESULTS_ROOT>/<eval_job_id>/`)
- `${TOOL_PATH}` → `/workspace/sop-eval-ms/assets/tools`

---

## Endpoints

### Status / discovery

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness check |
| GET | `/api/v1/gpus` | List GPUs visible to the container (index, name, total/free memory) |

### Per-action-chunk evaluation

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/v1/evaluation/start` | Queue a new per-chunk eval job |
| GET | `/api/v1/evaluation/status/{eval_job_id}` | Get current status / accuracy |
| GET | `/api/v1/evaluation/results/{eval_job_id}` | Get full per-action breakdown (after completion) |
| GET | `/api/v1/evaluation/all_jobs` | List all per-chunk eval jobs |
| POST | `/api/v1/evaluation/cancel/{eval_job_id}` | Cancel a running job |

### End-to-end evaluation

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/v1/e2e-evaluation/start` | Queue a new e2e eval job |
| GET | `/api/v1/e2e-evaluation/status/{eval_job_id}` | Get current status / sequence accuracy / temporal F1 |
| GET | `/api/v1/e2e-evaluation/results/{eval_job_id}` | Get full combined results JSON |
| GET | `/api/v1/e2e-evaluation/all_jobs` | List all e2e eval jobs |
| POST | `/api/v1/e2e-evaluation/cancel/{eval_job_id}` | Cancel a running job |

---

## Request schemas

All request fields are documented below with **types**, **defaults**, and **constraints**. Pydantic rejects unknown fields and out-of-range values at the API boundary, so a malformed request gets a `422 Unprocessable Entity` immediately.

### `ResolutionConfig` (used by both endpoints)

Vision-input resolution overrides for the VLM, mirroring the `qwen-vl-utils.process_vision_info` knobs. **All fields are optional**; omit fields you want to keep at default.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `max_frames` | `int` | `40` | Cap on number of decoded video frames per chunk. |
| `total_pixels` | `int` | `16572416` | Target total pixel budget across all frames (= `32 × 32 × 8092 × 2`, i.e. 16k vision tokens). Mirrors the training config's `[custom.vision]` setting so eval runs in-distribution by default. |
| `resized_height` | `int \| null` | `null` | Explicit per-frame resized height. If both `resized_height` and `resized_width` are set, they override `total_pixels`. |
| `resized_width` | `int \| null` | `null` | Explicit per-frame resized width. |
| `max_pixels` | `int \| null` | `null` | Upper bound on per-frame pixel count. |
| `min_pixels` | `int \| null` | `null` | Lower bound on per-frame pixel count. |

**Example values:**
```json
{ "max_frames": 40, "total_pixels": 16572416 }                       // training-aligned default
{ "max_frames": 30, "resized_height": 567, "resized_width": 1008 }   // fixed-resolution override
{ "max_frames": 40, "max_pixels": 81920, "min_pixels": 1024 }        // pixel-budget bounds
```

Unknown fields are rejected.

### `POST /api/v1/evaluation/start` — Per-action-chunk evaluation

```jsonc
{
  "training_job_id":   "string",          // REQUIRED. The completed VLM training job whose checkpoint to evaluate.
  "val_dataset_id":    "string",          // REQUIRED. Dataset directory under DATASET_ROOT containing per-action chunk subdirs.
  "fps":               8,                 // int,    default 8.    Frames-per-second the VLM samples from each chunk.
  "temperature":       0.0,               // float,  default 0.0.  vLLM/transformers sampling temperature.
  "top_p":             1.0,               // float,  default 1.0.  Range [0, 1]. Irrelevant at temperature=0; matters when temperature>0.
  "backend":           "vllm",            // string, default "vllm".  Choices: "vllm" | "transformers".
  "checkpoint_step":   null,              // int    | null. Specific training step to load; null = latest.
  "resolution_config": null,              // ResolutionConfig | null. See above. null = use defaults.
  "gpu_id":            null               // int    | null. Pin the subprocess to a specific GPU index; null = use all visible GPUs.
}
```

**Response (200):**
```json
{
  "eval_job_id": "uuid-string",
  "status": "queued",
  "message": "Evaluation job has been queued and will start shortly",
  "created_at": "2026-05-21T03:52:00.000000"
}
```

**Errors:**
- `400` if another evaluation is already running, the training job isn't completed, the checkpoint can't be found, or `actions.json` is missing.
- `404` if the training job ID doesn't exist.
- `422` if the request body fails Pydantic validation.

### `POST /api/v1/e2e-evaluation/start` — End-to-end evaluation

```jsonc
{
  "training_job_id":         "string",            // REQUIRED.  Completed VLM training job.
  "val_dataset_id":          "string",            // REQUIRED.  Dataset directory under DATASET_ROOT containing full videos + per-video annotation subdirs.
  "chunking_algorithm":      "ddm",               // string, default "ddm".  Choices: "ddm" | "uniform".  Controls Stage 1.

  // Required when chunking_algorithm == "ddm":
  "ddm_training_job_id":     "string | null",     // The completed DDM-Net training job whose checkpoint to use for temporal segmentation.

  // Required when chunking_algorithm == "uniform":
  "chunk_length_sec":        null,                // float | null.  Length of each uniform chunk in seconds.  Must be > 0.

  // DDM-specific (ignored when chunking_algorithm == "uniform"):
  "ddm_checkpoint":          null,                // str  | null.  Filename override (e.g. "best.ckpt"); null = use last.ckpt.
  "score_threshold":         0.5,                 // float, default 0.5.  Min DDM score to accept as a boundary.
  "nms_sec":                 0.0,                 // float, default 0.0.  NMS window in seconds.
  "ddm_batch_size":          8,                   // int,   default 8.
  "frames_per_segment_hint": 256,                 // int,   default 256.  DDM window stride hint.

  // VLM (common to both chunkers):
  "fps":                     8,                   // int,   default 8.
  "temperature":             0.0,                 // float, default 0.0.
  "top_p":                   1.0,                 // float, default 1.0.  Range [0, 1].
  "backend":                 "vllm",              // string, default "vllm".
  "checkpoint_step":         null,                // int    | null.  null = latest.
  "resolution_config":       null,                // ResolutionConfig | null.  null = use defaults.
  "gpu_id":                  null                 // int    | null.
}
```

**Cross-field rules (enforced by Pydantic):**
- If `chunking_algorithm == "ddm"`, `ddm_training_job_id` is required (non-empty).
- If `chunking_algorithm == "uniform"`, `chunk_length_sec` is required and must be `> 0`.

**Response (200):**
```json
{
  "eval_job_id": "uuid-string",
  "status": "queued",
  "message": "E2E evaluation job has been queued and will start shortly",
  "created_at": "2026-05-21T03:58:00.000000"
}
```

**Errors:**
- `400` if another e2e evaluation is already running, the VLM/DDM training job isn't completed, a checkpoint isn't found, or `actions.json` / annotation JSONs are missing.
- `404` if the VLM or DDM training job ID doesn't exist.
- `422` if the request body fails Pydantic validation (including the cross-field rules above).

---

## Choosing the backend

| Backend | When to use | Notes |
| --- | --- | --- |
| `vllm` (default) | Production runs. Faster (KV cache, batched generation), supports tensor parallelism — `tensor_parallel_size` is auto-detected from visible GPUs. Works for both CR1 (Qwen2.5-VL) and CR2 (Qwen3-VL). | Engine reads `config.architectures` from the checkpoint and dispatches the right model class. `gpu_memory_utilization=0.7` and `disable_custom_all_reduce=True` are hard-coded for stability on shared hosts. |
| `transformers` | Debugging or hosts where vLLM init fails (CUDA driver mismatch, custom-all-reduce kernel issues, etc.). | Uses `AutoModelForImageTextToText.from_pretrained` which dispatches by `config.architectures` — same CR1 / CR2 auto-detection as vLLM. Single-GPU only. |

The backend choice is **runtime-only** — both CR1 and CR2 checkpoints work with either backend without code changes. The only version-dependent piece is the default `resolution_config`, which is set to the value that aligns with the training config (`max_frames=40, total_pixels=16572416`); override per-request if needed.

## Choosing the E2E chunking strategy

| Strategy | When to use | What you need to provide |
| --- | --- | --- |
| `ddm` (default) | You have a trained DDM-Net checkpoint. Best accuracy when DDM is well-trained. | A completed DDM training job (`ddm_training_job_id`). |
| `uniform` | You don't have a DDM checkpoint, or want a fast sanity-check baseline. Splits each video into fixed-length slices regardless of action boundaries — F1 will be low but VLM-side metrics (action_accuracy, sequence_accuracy) are still meaningful. | `chunk_length_sec` only. No DDM job needed. |

The downstream VLM stage and accuracy computation are **identical** between strategies; only Stage 1 differs.

---

## Output files

When a job completes, all artefacts land in `${RESULTS_ROOT}/<eval_job_id>/`. Schemas below.

### Per-action-chunk evaluation

```
<RESULTS_ROOT>/<eval_job_id>/
├── inference_results.json       # VLM predictions per chunk
├── log.txt                      # Full subprocess stdout/stderr (Args + per-chunk lines)
└── assets/                      # Per-job VLM prompt (vlm_prompts.txt) generated from actions.json
```

**`inference_results.json` schema:**

```jsonc
{
  "<video_subdir_name>": [
    // One entry per chunk in the video subdir.
    [
      <gt_action_id: int>,      // Ground truth, parsed from chunk filename prefix (e.g. "01_..." → 1).
      "<vlm_response: str>",    // Raw VLM response text, e.g. "(1) installing the first fan...".
      "<chunk_path: str>"       // Absolute path to the chunk video file. Used by the RCA skill's parser.
    ],
    ...
  ],
  ...
}
```

> **Note**: Older results (predating this version) stored 2-tuples `[action, response]`. The current parser accepts both shapes; the API surface in `GET /results/{id}` returns whichever shape was stored at job time.

**`log.txt` (per-chunk) — key markers:**

```
<timestamp> Args: {'model_path': '...', 'fps': 8, 'top_p': 1.0, ...}      # one line at startup
...
Action Chunk: /workspace/sop-eval-ms/assets/data/<dataset>/<video>/01_<video>_1_2.mp4
(1) installing the first fan by connecting the connector...
Action Chunk: /workspace/sop-eval-ms/assets/data/<dataset>/<video>/02_<video>_1_3.mp4
(2) installing the second fan...
```

The downstream sop-rca-plugin parser keys off the `Action Chunk:` prefix and extracts the ground-truth action ID from the filename. See `helpers/analyze_by_action_confusion.py` in that skill.

### End-to-end evaluation

```
<RESULTS_ROOT>/<eval_job_id>/
├── anno.json                                                  # Collected per-video annotations
├── e2e_results.json                                           # Combined results — primary frontend display
├── log.txt                                                    # Full subprocess stdout/stderr
├── sop_e2e_eval_log.txt                                       # Python-logging-only stream
├── outputs_temporal_segmentation/
│   ├── temporal_segmentation.log                              # Args line for RCA hyperparameter extraction
│   ├── anno.json                                              # (per-stage copy)
│   ├── f1_<value>.json                                        # Per-video boundaries + F1/precision/recall + avg metrics
│   ├── video_to_boundaries_debug.json                         # Golden boundaries per video
│   ├── video_to_dmm_info_debug.json                           # DDM raw scores per video
│   └── <video_stem>.png                                       # DDM score visualisation per video (golden + predicted boundaries)
├── outputs_action_recognition/
│   ├── action_recognition_multi_gpu.log                       # Args line for RCA hyperparameter extraction
│   ├── video_name_to_output_text.json                         # VLM response per chunk per video
│   └── accuracy.json                                          # Stand-alone sequence-accuracy report
└── assets/                                                    # Per-job VLM prompt
```

**`e2e_results.json` schema:**

```jsonc
{
  "temporal_segmentation": {
    "avg_f1":        <float>,
    "avg_precision": <float>,
    "avg_recall":    <float>,
    "per_video": {
      "<video_name.mp4>": {
        "f1":         <float | null>,    // null if no golden boundaries available for this video
        "precision":  <float | null>,
        "recall":     <float | null>,
        "boundaries": [0.0, <boundary_sec>, ..., <duration_sec>]
      },
      ...
    }
  },
  "action_recognition": {
    // ─── Chunk-level (legacy, kept for backwards-compat) ─────────────────
    "overall_accuracy": <float>,           // Per-chunk classification accuracy (chunks mapped to golden action via boundary overlap).
    "per_action": {
      "<action_id_str>": {
        "label":    "<choice text>",
        "correct":  <int>,
        "total":    <int>,
        "accuracy": <float>
      },
      ...
    },
    // ─── Sequence-level (primary display) ────────────────────────────────
    "sequence_accuracy":     <float>,       // Fraction of videos whose predicted sequence == golden sequence (edit_distance == 0).
    "action_accuracy":       <float>,       // (total - wrong - duplicate - missing) / total, clamped to [0, 1].
    "total_videos":          <int>,
    "total_videos_dist_0":   <int>,         // Count of videos with edit_distance == 0.
    "total_actions":         <int>,
    "wrong":                 <int>,         // Substitutions in the Levenshtein backtrace.
    "duplicate":             <int>,         // Insertions (predicted but not in golden).
    "missing":               <int>,         // Deletions (in golden but not predicted).
    "videos_with_error":     ["<video_name>", ...],
    "per_video": [
      {
        "video":         "<video_name.mp4>",
        "golden":        [<action_id>, ...],
        "predicted":     [<action_id>, ...],
        "edit_distance": <int>,
        "wrong":         <int>,
        "duplicate":     <int>,
        "missing":       <int>,
        "steps":         ["Wrong: golden 2 predicted as 5 ...", ...]
      },
      ...
    ]
  }
}
```

`outputs_action_recognition/accuracy.json` contains the sequence-level subset (`sequence_accuracy`, `action_accuracy`, per-video diff, error step trace).

**E2E log files — Args markers (consumed by sop-rca-plugin):**

```
# outputs_temporal_segmentation/temporal_segmentation.log
Args: Namespace(resolution=224, nms_sec=0.0, score_threshold=0.5, batch_size=8, frames_per_segment_hint=256, frames_per_side=5)

# outputs_action_recognition/action_recognition_multi_gpu.log
Args: Namespace(max_frames=40, total_pixels=16572416, resized_height=None, resized_width=None, max_pixels=None, min_pixels=None, temperature=0.0, top_p=1.0, fps=8)
```

The field names match the upstream inference pipeline's standalone scripts so downstream tooling parses them identically.

---

## Examples

### Per-chunk evaluation (curl)

```bash
curl -X POST http://localhost:32090/api/v1/evaluation/start \
  -H 'Content-Type: application/json' \
  -d '{
    "training_job_id":  "d88bef4d-828a-4814-85dd-7dd2397a56a9",
    "val_dataset_id":   "server_assemble_test",
    "fps":              8,
    "temperature":      0.0,
    "backend":          "vllm",
    "resolution_config": {"max_frames": 40, "total_pixels": 16572416}
  }'
```

Returns `{"eval_job_id": "<uuid>", "status": "queued", ...}`. Poll:

```bash
curl http://localhost:32090/api/v1/evaluation/status/<eval_job_id>
```

When `status` becomes `"completed"`:

```bash
curl http://localhost:32090/api/v1/evaluation/results/<eval_job_id>
```

### E2E evaluation with DDM chunking

```bash
curl -X POST http://localhost:32090/api/v1/e2e-evaluation/start \
  -H 'Content-Type: application/json' \
  -d '{
    "training_job_id":      "d88bef4d-828a-4814-85dd-7dd2397a56a9",
    "ddm_training_job_id":  "db04f6ec-f4ec-4813-83e0-4486a929ff68",
    "val_dataset_id":       "server_assemble_test",
    "chunking_algorithm":   "ddm",
    "score_threshold":      0.5,
    "nms_sec":              0.0,
    "fps":                  8,
    "temperature":          0.0,
    "backend":              "vllm"
  }'
```

### E2E evaluation with uniform chunking

```bash
curl -X POST http://localhost:32090/api/v1/e2e-evaluation/start \
  -H 'Content-Type: application/json' \
  -d '{
    "training_job_id":     "d88bef4d-828a-4814-85dd-7dd2397a56a9",
    "val_dataset_id":      "server_assemble_test",
    "chunking_algorithm":  "uniform",
    "chunk_length_sec":    10.0,
    "fps":                 8,
    "temperature":         0.0,
    "backend":             "vllm"
  }'
```

Note: `ddm_training_job_id`, `score_threshold`, `nms_sec`, and DDM-related fields are ignored in uniform mode.

### Pinning a specific GPU

Add `"gpu_id": 1` to any start request to pin the subprocess to GPU 1. Without it the subprocess sees all GPUs the container sees (used by vLLM tensor parallelism). Discover what's available:

```bash
curl http://localhost:32090/api/v1/gpus
```

### Resolution-config override examples

```bash
# Higher frame budget, training-default pixels
"resolution_config": {"max_frames": 60, "total_pixels": 16572416}

# Fixed resize, smaller frame budget
"resolution_config": {"max_frames": 30, "resized_height": 384, "resized_width": 384}

# Pixel-budget bounds (let qwen-vl-utils decide the actual size between min and max)
"resolution_config": {"max_frames": 40, "max_pixels": 200704, "min_pixels": 50176}
```

---

## Operational notes

- **Only one evaluation runs at a time per kind.** The service rejects new per-chunk starts while another per-chunk eval is running (and the same for e2e). Start endpoints return `400 — An evaluation is already running` in that case.
- **Cancel cleans up the subprocess tree.** `POST /cancel/<id>` sends `SIGTERM` to the subprocess and any children; the job row's status flips to `cancelled`.
- **All visible CUDA devices are used by default.** `tensor_parallel_size` is auto-detected. Pin with `gpu_id` if other workloads share the host.
- **Job persistence.** Job rows are stored in the shared Postgres metadata DB (`evaluation_job` / `e2e_evaluation_job` tables). Results are persisted to disk under `<RESULTS_ROOT>/<eval_job_id>/` and also serialised into the `results_json` column once complete — so the API can serve historical results even if the on-disk artefacts are pruned.
- **Logs are persisted under the job dir.** `log.txt` carries everything; the rest is structured per-stage artefacts (see Output files above).

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `400 An evaluation is already running` | Another job in `running` state | Wait or cancel via `POST /cancel/<id>` |
| `400 actions.json not found for dataset <id>` | The validation dataset directory is missing `actions.json` | Place an actions JSON at `<DATASET_ROOT>/<val_dataset_id>/actions.json` |
| `400 No checkpoint dir for training job <id>` | The VLM training job didn't produce a checkpoint at the expected path | Verify `<RESULTS_ROOT>/<training_job_id>/.../safetensors/step_<N>` exists |
| `400 DDM training job <id> is not completed` (e2e ddm mode) | DDM job hasn't finished or failed | Train DDM to completion, or switch to `chunking_algorithm: "uniform"` |
| `422 chunking_algorithm='uniform' requires chunk_length_sec > 0` | Uniform mode without `chunk_length_sec` | Add `"chunk_length_sec": <seconds>` to the request |
| Eval crashes with `ValueError: too many values to unpack` | Downstream consumer expects different `inference_results.json` shape | This is fixed in the current version — rebuild the container to pick up the fix |
| Job stuck in `running` after subprocess exited | Subprocess died without writing the expected output file (`inference_results.json` / `e2e_results.json`) | Check `log.txt` for the underlying error; cancel and re-run |
