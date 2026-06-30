---
name: sop-cr-finetuning
description: Fine-tune Cosmos-Reason2 (CR2) VLM for SOP monitoring. Use when you need to launch and monitor a VLM training run with a given dataset ID.
argument-hint: <qa_augmented_dataset_id>
license: "CC-BY-4.0 AND Apache-2.0"
---

# SOP CR VLM Fine-tuning

You are performing Cosmos-Reason2 (CR2) VLM fine-tuning for an SOP (Standard Operating Procedure) monitoring system. The training pipeline fine-tunes a Qwen3VL-based vision-language model on augmented SOP video QA data so the model can classify operator actions from video segments.

Your job is to launch training and monitor it to completion. Training logs are persisted by the training service at `results/<job_id>/log.txt`.

## Input

The user provides a `qa_augmented_dataset_id` as `$ARGUMENTS`. This is the identifier for the augmented dataset to train on.

If no argument is provided, ask the user for the qa_augmented_dataset_id.

For reference on API endpoints, anomaly detection, and monitoring commands, see `${CLAUDE_SKILL_DIR}/reference.md`.

## Defaults

- **BASE_URL:** `http://localhost:32080/api/v1`
- **POLL_INTERVAL:** 60 seconds
- **RESULTS_ROOT:** `./assets/results`

## Fine-tuning Procedure

### Phase 1: Pre-flight Checks

Before starting any training, verify readiness:

1. **Check service health**:
   ```bash
   curl -s http://localhost:32080/health
   ```
   If the service is healthy, skip to step 2. If unreachable, start the container:
   ```bash
   docker compose up -d cosmos-reason-microservice
   ```
   Then poll `/health` with retries until ready. If the service fails to become healthy after retries, report the error and stop.

2. **Check GPU availability**:
   ```bash
   nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
   ```
   Warn if free memory is below 60GB per GPU.

3. **Check for running jobs**:
   ```bash
   curl -s http://localhost:32080/api/v1/fine-tuning/all_jobs
   ```
   If a job is already running, warn the user and ask whether to wait or cancel it.

Report all pre-flight results before proceeding.

### Phase 2: Training and Monitoring

1. **Start training**:
   ```bash
   curl -s -X POST "http://localhost:32080/api/v1/fine-tuning/start?dataset_id=<qa_augmented_dataset_id>" -H "Content-Type: application/json"
   ```
   Use the `qa_augmented_dataset_id` from `$ARGUMENTS`. Record the `job_id` from the response.

2. **Monitor training progress** by polling the status endpoint every 60 seconds:
   ```bash
   curl -s "http://localhost:32080/api/v1/fine-tuning/status/<job_id>"
   ```

   On each poll, extract and report:
   - `status` — queued / running / completed / failed / cancelled
   - `progress` — percentage complete
   - `current_step` / `total_steps`
   - `loss` — current training loss

3. **Monitor for anomalies** by checking container logs periodically:
   ```bash
   # Check for OOM
   docker compose logs --since 5m cosmos-reason-microservice 2>&1 | grep -i "out of memory\|OOM\|CUDA error"

   # Check for NaN
   docker compose logs --since 5m cosmos-reason-microservice 2>&1 | grep -iE "loss.*nan|nan.*loss" | head -5

   # Check GPU status
   docker compose exec cosmos-reason-microservice nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
   ```

4. **Report progress** every 10% (10%, 20%, …, 100%). Include:
   - Current loss and trend (decreasing / plateau / spiking)
   - Steps per second
   - Any warnings or anomalies detected (OOM, NaN, loss spikes, GPU temperature)

5. **Wait for completion**. When status becomes `completed`, `failed`, or `cancelled`, proceed to Phase 3.

### Phase 3: Summary

After training completes (regardless of outcome), collect metrics, write a training report to `./assets/results/<job_id>/training_report.md`, and report to the user.

#### 3a: Collect Final Metrics

1. **Gather metrics** from the final status response:
   - Final loss, best loss (lowest seen during training)
   - Total steps completed, duration
   - Final status (completed / failed / cancelled)

2. **Collect anomaly summary** from monitoring:
   - Any OOM events and recovery actions
   - Loss spikes or NaN occurrences
   - GPU memory peak usage

#### 3b: Merge LoRA Adapter (if present)

If the training run used LoRA, the checkpoint is a small adapter (delta weights) that cannot be used for inference on its own and must be merged with the base model. Full fine-tuning runs skip this step — the checkpoint is already a standalone full-weight model.

After this phase, the inference-ready model **always** lives at the same path: `assets/results/<job_id>/<timestamp>/safetensors/step_<N>/`. For LoRA runs, the merged full-weight model replaces the adapter at that path; the original adapter is preserved as a sibling `step_<N>_lora_adapter/`. Downstream evaluation reads `step_<N>/` via the standard `training_job_id` + `checkpoint_step` resolution and does not need to know whether the run was LoRA or full.

**Path layout.** Cosmos-RL writes checkpoints to:
```
assets/results/<job_id>/<timestamp>/safetensors/step_<N>/
```
`<timestamp>` is a UTC datetime subdir (one per training run); `<N>` is the global step. Pick the final step (largest `<N>`) for inference.

>IMPORTANT: **Run all merge commands inside the same `cosmos-reason-microservice` container that produced the adapter.**

- The base model is already on disk at `/workspace/sop-cr-ftms/assets/weights/<base_model_dir>/`.
- The adapter is already on disk at `/workspace/sop-cr-ftms/assets/results/<job_id>/...`.


Container path prefix mapping (host ↔ container) is a fixed swap:
```
host:      <bp_root>/assets/...
container: /workspace/sop-cr-ftms/assets/...
```

**Detect LoRA vs full fine-tune.** Resolve the final-step checkpoint dir, then probe for `adapter_config.json`:

```bash
CR_CONT=$(docker ps --format '{{.Names}}' | grep cosmos-reason-microservice | head -1)
JOB_ID=<job_id>
CKPT_DIR=$(docker exec $CR_CONT bash -c \
  "ls -d /workspace/sop-cr-ftms/assets/results/$JOB_ID/*/safetensors/step_* 2>/dev/null | sort -V | tail -1")
echo "checkpoint dir: $CKPT_DIR"

if docker exec $CR_CONT test -f "$CKPT_DIR/adapter_config.json"; then
  echo "LoRA adapter detected — run steps 3b.1 → 3b.3."
else
  echo "Full fine-tune detected — skip 3b.1 → 3b.3. step_<N>/ is already an inference-ready model."
fi
```

If full fine-tune: skip the rest of this section.

##### 3b.1 — Normalise `adapter_config.json` for forward-compat

Cosmos-RL's training stack may serialise LoRA config fields the inference-side peft doesn't accept (e.g. `r_pattern`), or write `null` where a dict is expected (e.g. `alpha_pattern: null`). Normalise defensively before loading:

```bash
docker exec $CR_CONT python3 -c "
import json
p='$CKPT_DIR/adapter_config.json'
d=json.load(open(p))
d.pop('r_pattern', None)                            # drop unknown fields
for k in ('alpha_pattern','rank_pattern'):
    if d.get(k) is None: d[k] = {}                  # null -> {} (peft expects dict)
for k,v in dict(bias='none', task_type='CAUSAL_LM', inference_mode=True,
                fan_in_fan_out=False, layers_to_transform=None,
                layers_pattern=None, megatron_config=None,
                megatron_core='megatron.core', loftq_config={},
                use_dora=False, init_lora_weights=True).items():
    d.setdefault(k, v)                              # backfill defaults
json.dump(d, open(p,'w'), indent=2)
print('adapter_config.json normalised')
"
```

##### 3b.2 — Merge adaptor with base model

Write to a sibling `step_<N>_merged_tmp/` so the final atomic swap (3b.3) is crash-safe:

The base model path and the base model class are both **derived from the adapter** at run-time:
- **Base path** comes from `adapter_config.json["base_model_name_or_path"]`, which peft writes at training time. It is typically a path relative to the training process's working directory (`/workspace/sop-cr-ftms`), e.g. `./assets/weights/Cosmos-Reason2-2B`. Resolve it against that working dir if it isn't already absolute.
- **Model class** comes from `AutoModelForImageTextToText`, which reads `architectures` from the base model's `config.json` and instantiates the right class (e.g. `Qwen3VLForConditionalGeneration` for Cosmos-Reason2-2B, `Qwen2_5_VLForConditionalGeneration` for older variants).

```bash
docker exec -w /workspace/sop-cr-ftms $CR_CONT python3 -c "
import json, os, torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

adapter_path = '$CKPT_DIR'
tmp_path     = adapter_path + '_merged_tmp'

# Resolve the base model path from the adapter's own metadata.
with open(os.path.join(adapter_path, 'adapter_config.json')) as f:
    base_rel = json.load(f)['base_model_name_or_path']
base_path = (base_rel if os.path.isabs(base_rel)
             else os.path.normpath(os.path.join('/workspace/sop-cr-ftms', base_rel)))
print(f'base model: {base_path}', flush=True)

print('loading base...',    flush=True)
base   = AutoModelForImageTextToText.from_pretrained(
            base_path, dtype=torch.bfloat16, device_map='cpu')
print('loading adapter...', flush=True)
model  = PeftModel.from_pretrained(base, adapter_path)
print('merging...',         flush=True)
merged = model.merge_and_unload()

os.makedirs(tmp_path, exist_ok=True)
merged.save_pretrained(tmp_path, safe_serialization=True)
# Processor/preprocessor configs live with the base model, not the adapter.
# The adapter dir only carries tokenizer files.
AutoProcessor.from_pretrained(base_path).save_pretrained(tmp_path)
print('merged ->', tmp_path)
"
```

Notes on the kwargs:
- `device_map='cpu'` — merge is memory-bandwidth-bound and uses ~6 GB CPU RAM. Keeps GPUs free for concurrent training/eval.
- `dtype=torch.bfloat16` — matches the precision the LoRA was trained against.
- The processor (`AutoProcessor`) is loaded from the **base** model path, not the adapter path. Vision/video preprocessor configs live with the base; the adapter dir only carries tokenizer + adapter files.

##### 3b.3 — Atomic swap into `step_<N>/`

Evaluation-ms resolves checkpoints by `safetensors/step_<N>/` and calls `from_pretrained` on that exact path; it has no knowledge of `merged_model/` or other subdirs. Swap the merged result in and keep the original adapter as a sibling:

```bash
docker exec $CR_CONT bash -c "
  STEP_DIR='$CKPT_DIR'
  PARENT=\$(dirname \$STEP_DIR)
  STEP=\$(basename \$STEP_DIR)
  mv \$STEP_DIR \${PARENT}/\${STEP}_lora_adapter
  mv \${PARENT}/\${STEP}_merged_tmp \$STEP_DIR
"
```

##### 3b.4 — Verify

Listing the new `step_<N>/` should show full-model files, not adapter files:

```bash
docker exec $CR_CONT ls "$CKPT_DIR"
```
Expect `config.json`, `model.safetensors` (or sharded `model-00001-of-NNNNN.safetensors`), `tokenizer.json`, `preprocessor_config.json`, `video_preprocessor_config.json`, generation/special-token configs. The presence of `adapter_model.safetensors` here means the merge did not land — recheck 3b.3.


#### 3c: Write Training Report

Save a formatted training report to `./assets/results/<job_id>/training_report.md`:

```markdown
# Training Report

## Overview
| Field | Value |
|-------|-------|
| Dataset | <qa_augmented_dataset_id> |
| Job ID | <job_id> |
| Status | completed / failed / cancelled |
| Duration | Xh Xm |

## Training Metrics
| Metric | Value |
|--------|-------|
| Final Loss | X.XX |
| Best Loss | X.XX (step N) |
| Total Steps | N / N |
| Steps/sec | X.XX |
| GPU Memory Peak | XX% |

## Anomalies
- List any OOM events, loss spikes, NaN occurrences, or GPU warnings detected during training.
- "None" if no anomalies were detected.

## Artifacts
- **Inference-ready checkpoint:** `assets/results/<job_id>/<timestamp>/safetensors/step_<N>/` — always, regardless of training mode (Phase 3b merges LoRA into the base in place; full fine-tune writes here directly).
- **LoRA adapter backup (LoRA runs only):** `assets/results/<job_id>/<timestamp>/safetensors/step_<N>_lora_adapter/` — the unmerged adapter, kept for reproducibility/audit.
- **Training log:** `results/<job_id>/log.txt`
```

#### 3d: Output Summary

Print the training report content to the user as well. If training failed, include the error details.
