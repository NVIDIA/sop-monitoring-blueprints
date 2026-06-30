---
name: sop-ft-orchestrate
description: Autonomous end-to-end orchestrator for SOP fine-tuning. Runs the full Import → Augment → DDM Train → VLM Train → Evaluate → RCA loop. Interprets RCA findings across DDM, VLM and augment axes, applies config fixes autonomously, and iterates until success criteria are met or max_pipeline_iterations reached. Call with a path to an inputs.yaml or with natural language.
argument-hint: <inputs.yaml> | "fine-tune on /path/to/dataset targeting seq_accuracy >= 0.70"
license: "CC-BY-4.0 AND Apache-2.0"
---

# SOP Fine-tuning Orchestrator

## Prime Directive: Never Stop Until the Goal Is Reached

>IMPORTANT: **Never stop unless iterations_substantive >= budget.max_pipeline_iterations, accuracy criteria met, the failure requires locked external systems, or hardware changes.**

This means:

- Phase fail → diagnose, fix, restart. Do not report and wait.
- `/sop-rca` delegation FAILED (skill missing or errored) → use the fallback diagnostics in Step 8b for that one iteration; document the failure in `run_state.yaml`; retry `/sop-rca` next iteration.
- max_pipeline_iterations reached → write PARTIAL report (Step 9) with remaining gaps and recommended fixes.
- Hang/crash → watchdog diagnoses and restarts autonomously, including code-level fixes. **All code fixes follow Override Policy — never modify `plugins/`.**

### Iteration Budget — ENFORCED

The budget is `max_pipeline_iterations` (default **8**, set in `inputs.yaml`). Track it explicitly under `run_state.yaml`:

```yaml
iteration_budget:
  max_pipeline_iterations: 8        # from inputs.yaml
  iterations_substantive: 0         # cumulative count of retraining iters (categories below)
  iterations_eval_only: 0           # threshold sweeps, etc — DO NOT count against budget
  iterations_remaining: 8           # = max - substantive
  rca_runs_completed: 0             # invariant: must equal count(eval_history entries that miss success_criteria) when run_rca: true
  attempts:                         # one entry per substantive iter — for the pre-PARTIAL gate
    augment_config_change: []       # list of distinct configs tried (e.g., ["DMCQ confusion", "DMCQ adjacent"])
    training_config_change: []      # e.g., ["LR corrected to 5e-6"]
    ddm_training_config_change: []  # e.g., ["epochs 30", "epochs 30 + RandomResize bilinear"]
    code_change: []                 # infra fixes — free, do not count against budget
```

**Iteration types — only "substantive" counts against the budget:**

| Type | Budget Counts? | Example |
|------|---------|---------|
| `substantive` | ✅ counts | augment-config-change, training-config-change, ddm-training-config-change |
| `eval-only` | ❌ not counts | DDM threshold sweep with same VLM checkpoint, eval-config-change with same model |
| `infrastructure` | ❌ not counts | code-change for an auto-fixable bug, container restart |

Only substantive counter matters — 2 retrains + 3 threshold sweeps = **2/8** budget used. Update `iteration_budget` after EVERY iteration.

(Diagnostic heuristics for the rare case where `/sop-rca` is **unavailable** have moved to **Step 8b-fallback**. Do not consult them as a routine substitute for RCA)

## Watchdog Policy

After every job launch: run with `run_in_background=true`; do not poll or sleep.

| Phase | Mechanism | Terminal signals / outputs |
|-------|-----------|---------------------------|
| DDM train | `watch_api_job.sh` watchdog | `DDM_DONE`, `DDM_FAILED`, `DDM_HANG` (in `watchdog_ddm.log`) |
| VLM train | `watch_api_job.sh` watchdog | `VLM_DONE`, `VLM_FAILED`, `VLM_HANG`, `VLM_TIMEOUT` (in `watchdog_vlm.log`) |
| By-action eval | `/sop-by-action-eval` skill (blocks on API poll) | JSON envelope on stdout — `status: completed \| failed \| timeout` |
| E2E eval | `/sop-e2e-inference` skill (blocks on API poll) | JSON envelope on stdout — `status: completed \| failed \| timeout` |

Training watchdogs remain file-based. Eval uses the blocking skill envelope (Step 7a/7b) — no watchdog log to monitor; the skill returns when eval-ms reports a terminal status.

If Monitor silent >10 min (training): check watchdog process alive; restart if dead.

---

## Override Policy: Never Modify Plugin Files During a Run

**Files inside `plugins/` are read-only.** Copy any file needing changes to `<run_dir>/overrides/` first, then modify the copy. Pass the override via env vars

Record in `run_state.yaml` under `overrides:`. Never use `plugins/` paths.

---

## Pipeline Overview

DDM and VLM are **independent retry axes** — each can be retrained without touching the other.
The table shows which steps run for each action type:

| Action type | Step 4 Augment | Step 5 DDM | Step 6 VLM | Step 7 Eval |
|---|---|---|---|---|
| eval-config-change | ❌ | ❌ | ❌ | E2E only |
| DDM training-config-change | ❌ | ✅ retrain | ❌ | E2E only |
| VLM training-config-change | ❌ | ❌ | ✅ retrain | Both |
| augment-config-change | ✅ re-augment | ❌ | ✅ retrain | Both |
| code-change | depends on phase | depends | depends | depends |

```
[Step 1] Prerequisites
      │
      ▼
[Step 2] Initialize run_state.yaml
      │
      ▼
[Step 3] Import dataset ── run ONCE, never repeated
      │
      ▼
[Step 4] Data Augmentation  ◄── augment-config-change (DMCQ, confusion_map, MCQ max_chunk_len)
         BCQ · MCQ · GoldenGQA · GQAs · DMCQ · DS            [DDM unaffected — skip Step 5]
      │
      ▼
[Step 5] DDM-Net Training  ◄── DDM training-config-change (resolution, RandomResize, epochs)
         ResNet-50 boundary detector                         [skip Step 4 + Step 6]
      │
      ▼
[Step 6] VLM Training  ◄────── VLM training-config-change (LR, fps/max_frames)
         Cosmos-Reason2-2B SFT + LoRA merge if applicable   [skip Step 4 + Step 5]
      │
      ▼
[Step 7] Evaluate
      ├── 7a By-Action  (VLM in isolation, perfect segmentation)
      └── 7b E2E        (DDM + VLM full pipeline)
            │  [DDM-only retrain → Step 7b only; VLM or augment → both 7a + 7b]
      ▼
[Step 8] RCA + Decision Loop
      │
      ├── ALL criteria met ─────────────────────────────────► DONE ✅
      │
      ├── eval-config-change ──────────────────────────────► Step 7 → Step 8
      │
      ├── DDM training-config-change ─────────────────────► Step 5 → Step 7b → Step 8
      │   resolution · RandomResize · epochs                 (augment + VLM unchanged)
      │
      ├── VLM training-config-change ─────────────────────► Step 6 → Step 7 → Step 8
      │   LR · fps/max_frames mismatch                       (augment + DDM unchanged)
      │
      ├── augment-config-change ───────────────────────────► Step 4 → Step 6 → Step 7 → Step 8
      │   DMCQ · confusion_map · MCQ max_chunk_len           (DDM unchanged — skip Step 5)
      │
      ├── code-change ──────────────────────────────────────► override fix → retry phase
      │
      └── max iterations or manual ──────────────────────────► Step 9 PARTIAL + hand off
```

---

## Reference Documents

- `${SKILL_DIR}/references/inputs-template.yaml` — all config fields
- `${SKILL_DIR}/references/decision-logic.md` — RCA action type → pipeline response mapping
- `${SKILL_DIR}/references/prerequisites.md` — pre-flight checklist
- `${SKILL_DIR}/references/run-state-schema.yaml` — full run_state.yaml template with field annotations
- `${SKILL_DIR}/references/gqas-preflight.md` — local vLLM launch, probe-validate, and output check (Steps 1–3)
- `${SKILL_DIR}/references/claude-gqas-setup.md` — Claude API backend setup: patch, restart, verify
- `${SKILL_DIR}/scripts/watch_api_job.sh` — watchdog for API-polled jobs (DDM, VLM training)
- Eval jobs (`/sop-by-action-eval`, `/sop-e2e-inference`) are NOT watchdogged — the eval skills block until the eval-ms HTTP job reaches a terminal state and return a JSON envelope.
- `${SKILL_DIR}/scripts/auto_detect_splits.py` — auto-detect train/test split subdirectories
- `${SKILL_DIR}/scripts/import_dataset.sh` - import annotated dataset
- `${SKILL_DIR}/references/augment-config-guide.md` — annotated augment_config.yaml template (DMCQ, MCQ, GQAs)

---

## Resume Protocol

**Run at the very start of every session:**

```bash
ls <output_dir>/run_*/run_state.yaml 2>/dev/null | sort | tail -1
```

If a `run_state.yaml` is found:

1. Print full contents with `=== RESUMING RUN: <run_id> ===`
2. `done` → skip. `in_progress` → verify: DDM/VLM: check API status; eval: check output file exists; augment: check augmented dir exists.
3. Continue from first non-`done` phase.

If no `run_state.yaml` exists — proceed to Step 0.

---

## Step 0: Parse Inputs

Read the inputs.yaml (or parse natural language). Resolve all paths. Set defaults. Print config summary and confirm with user before starting:

### Output Directory Contract (mandatory)

There are **two trees** in an orchestrated run, owned by different things:

1. `<run_dir>/` — owned by the orchestrator. Contains run-state, per-iter configs, RCA reports, progress, watchdogs, and a per-iter snapshot of each eval job's output.
2. `<host_results_root>/<eval_job_id>/` — owned by **eval-ms**. Each `/sop-by-action-eval` and `/sop-e2e-inference` invocation produces one such directory; eval-ms decides the uuid. The orchestrator never *writes* here, but it does two things after each eval: (a) records pointers in `run_state.eval_outputs`, **and** (b) snapshots the job's output into `<run_dir>/iter<N>/{by_action,e2e}/` so the run dir is self-contained and survives any later cleanup of the eval-ms store.

```
<output_dir>/                              # from inputs.yaml; default ./sop_fine_tune
└── run_<YYYYMMDD_HHMMSS>/                 # = <run_dir>; created in Step 2
    ├── run_state.yaml                     # full run state — incl. eval_outputs pointers (KEPT)
    ├── progress.md, progress.html         # phase logs + live progress chart
    ├── orchestrator_report.md             # final SUCCESS / PARTIAL summary
    ├── overrides/                         # code overrides per Override Policy
    ├── watchdog_*.log                     # DDM/VLM training watchdog tails
    └── iter<N>/
        ├── augment_config.yaml            # snapshot of augment config used
        ├── ddm_train_config.yaml          # snapshot of DDM training config
        ├── train_config.toml              # snapshot of VLM training config
        ├── training.log                   # copy of VLM training log
        ├── inputs_by_action_iter<N>.yaml  # generated per-iter eval input (moved here, off top level)
        ├── inputs_e2e_iter<N>.yaml        # generated per-iter eval input
        ├── by_action/                     # snapshot of the by-action eval job output
        ├── e2e/                           # snapshot of the e2e eval job output
        ├── rca_analysis/                  # /sop-rca helper JSONs
        └── rca_report.md                  # /sop-rca formal report (REQUIRED on failure)
```

```
<host_results_root>/                       # docker-compose volume
├── <by_action_eval_job_id>/               # /sop-by-action-eval job output
│   ├── inference_results.json
│   ├── log.txt
│   └── assets/
└── <e2e_eval_job_id>/                     # /sop-e2e-inference job output
    ├── e2e_results.json
    ├── log.txt
    ├── sop_e2e_eval_log.txt
    ├── outputs_action_recognition/
    │   ├── accuracy.json
    │   ├── video_name_to_output_text.json
    │   └── action_recognition_multi_gpu.log
    └── outputs_temporal_segmentation/
        ├── f1_<thr>.json
        ├── video_to_boundaries_debug.json
        ├── video_to_ddm_info_debug.json
        ├── temporal_segmentation.log
        └── <video>.png
```

**Hard rules:**

- The orchestrator MUST pass `output_dir=<run_dir>/iter<N>` when delegating to `/sop-rca` in Step 8b (see "RCA delegation contract" below). Never let `/sop-rca` default to `<cwd>/rca_reports/` — that splits artifacts.
- The orchestrator MUST NOT pass `output_dir` to `/sop-by-action-eval` or `/sop-e2e-inference`. Those skills do not accept an output-dir override — eval-ms decides where the job writes. The orchestrator captures the returned `host_output_dir` from each skill's JSON envelope and (a) persists it to `run_state.eval_outputs.<phase>.host_output_dir`, then (b) snapshots that directory into `<run_dir>/iter<N>/<phase>/` (Step 7c).
- The eval snapshot is a full recursive copy of the eval-ms job dir (e.g. `cp -a <host_output_dir>/. <run_dir>/iter<N>/by_action/`). Copy everything the job wrote; do not hand-pick files.
- All ad-hoc analyses produced by the orchestrator (helper-script outputs, debug dumps, scratch yamls) MUST live under `<run_dir>/` — never under `<cwd>/` and never under any plugin path.
- One run = one `<run_dir>`. Resumes write into the same `<run_dir>` discovered by the Resume Protocol.
- Eval-ms-owned directories under `<host_results_root>/` are not garbage-collected by the orchestrator.

### RCA delegation contract (Step 8b call site)

`/sop-rca` reads paths it was told about — it does NOT construct paths from conventions. The orchestrator builds the delegation payload from `run_state.eval_outputs` (populated by Step 7a/7b) plus the snapshots it placed under `<run_dir>/iter<N>/`:

```
/sop-rca \
  e2e_outputs_dir=<run_dir>/iter<N>/e2e/outputs_action_recognition \
  ddm_outputs_dir=<run_dir>/iter<N>/e2e/outputs_temporal_segmentation \
  by_action_results=<run_dir>/iter<N>/by_action/inference_results.json \   # JSON is preferred; log.txt also accepted
  actions_json=<dataset_path>/actions.json \
  augment_config=<run_dir>/iter<N>/augment_config.yaml \
  ddm_train_config=<run_dir>/iter<N>/ddm_train_config.yaml \
  vlm_train_config=<run_dir>/iter<N>/train_config.toml \
  vlm_train_log=<run_dir>/iter<N>/training.log \
  output_dir=<run_dir>/iter<N> \
  iter=<N> \
  success_criteria="e2e_seq_acc>=1.0,e2e_action_acc>=0.95,by_action_acc>=0.95,ddm_f1>=0.6"
```

**Dispatch this payload via a sub-agent** (Agent tool) — see Step 8b for why this is mandatory (keeps the heavy RCA out of the orchestrator's context so RCA is always affordable). The sub-agent writes analysis JSONs to `<output_dir>/rca_analysis/` and the report to `<output_dir>/rca_report.md`, and returns ONLY the compact `RCA_RESULT:` block (its skill's return contract). The orchestrator consumes that block — verifying `RCA_RESULT.report_path` exists and copying `RCA_RESULT.typed_actions` into `rca_reports[]` — and never needs to load the report prose.


### Dataset Split Auto-Detection

If the user provides a single `dataset_path` (no explicit `eval_dataset_path`), run:

```bash
python3 ${SKILL_DIR}/scripts/auto_detect_splits.py <dataset_path>
# Success: prints TRAIN=<abs_path> and EVAL=<abs_path>
# Failure: prints diagnostic and exits 1 — set paths explicitly in inputs.yaml
```

Matches `<stem>_train` ↔ `<stem>_test` pairs; fails on zero or ambiguous matches.

---

## Step 1: Prerequisites

Run all checks from `references/prerequisites.md`. Auto-fix soft issues (start services, install packages). Block on hard failures. Do not proceed until all pass.

---

## Step 2: Initialize Run State

Create `<output_dir>/run_<YYYYMMDD_HHMMSS>/run_state.yaml`. Print path as first output line: `=== RUN STATE: <full/path/to/run_state.yaml> ===`

### State + progress files (MANDATORY — generated, never hand-edited)

**`run_state.yaml` is edited ONLY via `scripts/rs_update.py` — never with `sed`, `str.replace()`,
or a from-memory Edit block.** Hand-editing a growing YAML file caused two real production bugs
(field cross-contamination from copy-pasted entry templates, and silent no-ops when a replace
anchor didn't byte-match). `rs_update.py` does load→mutate→dump with round-trip validation:

```bash
python3 ${SKILL_DIR}/scripts/rs_update.py <run_dir> set iteration=N phase_status.rca=done
python3 ${SKILL_DIR}/scripts/rs_update.py <run_dir> append eval_history '<json row>'
python3 ${SKILL_DIR}/scripts/rs_update.py <run_dir> append rca_reports  '<json entry>'
python3 ${SKILL_DIR}/scripts/rs_update.py <run_dir> budget --substantive +1 --rca +1
```

**`progress.md` and `progress.html` are GENERATED from `run_state.yaml` — never authored by hand.**
They are a pure projection of `eval_history` + `phase_status`, so they cannot drift behind the real
state. Regenerate BOTH after every eval (Step 7c) and at Step 2 init:

```bash
python3 ${SKILL_DIR}/scripts/gen_progress.py <run_dir>
# prints: === PROGRESS (local file): <run_dir>/progress.md ===  (and progress.html)
```

(For richer chart annotations, eval_history rows may carry an optional `chart` sub-dict:
`{ph, t, lr, samp, qas}` — gen_progress.py reads it; absence is fine.)

Both files + the canonical run_state are what the Step 8a.0 gate asserts are current — not optional.
`gen_progress.py`/`rs_update.py` also refresh `~/.cache/sop-ft-orchestrate/active_run` so the
harness gate can locate this run_dir even though it lives outside the harness cwd.

Initialize from `${SKILL_DIR}/references/run-state-schema.yaml`. Invariants:
- **`eval_vision_params`** — re-populate after every VLM training run; must match `train_config.toml [custom.vision]`.
- **`phase_status`** — `in_progress` BEFORE delegating; `done` AFTER.
- **`rca_reports`** — every failed iteration must have an entry.

---

## Step 3: Phase 1 — Import Dataset (run once)

Check: `curl -s http://localhost:8100/api/v1/datasets | python3 -c "import json,sys; d=json.load(sys.stdin); print([x['id'] for x in d.get('datasets',[])])"`

If not imported: copy dataset to `<training_bp_root>/assets/data/`, then `bash ${SKILL_DIR}/scripts/import_dataset.sh <dataset_id>`. Verify counts.
- Do NOT re-import on subsequent iterations.
- Do import every dataset provided, including training and validation

---

## Step 4: Phase 2 — Data Augmentation _(run iter 1 + on augment-config-change)_

Increment `augmented_dataset_id` suffix; apply RCA augmentation config changes.

GQAs backend (priority order):

**Check for `ANTHROPIC_API_KEY` (HIGHEST PRIORITY):** if set → use Claude (preferred: no GPU, ~10× faster). Set in `augment_config.yaml`:

```yaml
gqas:
  enable: true
  llm_type: "local"         # argparse only accepts "local" or "nvidia"; routing is by model name
  llm: claude-haiku-4-5-20251001   # detected as Claude by the override; use haiku for speed/cost
  local_llm_url: ""         # not used for Claude routing
  enable_thinking: "false"
  num_qa_llm: 8
  num_qa_per_chunk: 2
```

  Setup: `${SKILL_DIR}/references/claude-gqas-setup.md`.

**Fallback: NIM:** valid `NGC_API_KEY` → `llm_type: "nvidia"`. After 429 for >5 min → local.

**Fallback: local:** vLLM container `vllm/vllm-openai@sha256:2e08b462bb444a6da8a84a533f09024c61617574e67386efe4a723a0633fcc6a` with `Qwen/Qwen3-8B` (**no** `--reasoning-parser qwen3`). Stop server after augment. Alt: `Qwen/Qwen3.5-27B` (requires `--reasoning-parser qwen3`).

- **Never disable GQAs.** Fix start failures; do not set `enable: false`.

### GQAs Pre-flight (local vLLM only)

Complete all three steps in `${SKILL_DIR}/references/gqas-preflight.md` before augmenting: launch, probe-validate, post-augmentation output check.

Write `phase_status.augment: in_progress` to `run_state.yaml`, then **delegate to `/sop-data-augmentation`.**

On success: run Step 3 output check from `${SKILL_DIR}/references/gqas-preflight.md`; write `augmented_dataset_id` and `phase_status.augment: done`.

On failure: diagnose and fix; Do not skip GQAs unless genuinely unresolvable.

Keep 2 newest `<label_data_id>_augmented_*/` dirs. Via `docker exec <data_gen_container>`:
```bash
docker exec <data_gen_container> bash -c \
  "ls -dt /workspace/assets/data/<label_data_id>_augmented_*/ 2>/dev/null | tail -n +3 | xargs -r rm -rf"
```

---

## Step 5: Phase 3 — DDM Training _(run iter 1 + on DDM training-config-change)_

Set: `num_gpus` = GPUs detected; `batch_size`: < 25 GB → 4, < 50 GB → 16, else → 32.

Re-run: apply RCA changes; increment `ddm_config_version`.

**DDM resize augmentation policy (overrides `/sop-rca-plugin:sop-rca` Pattern 10):**

When RandomResize is recommended: **never use `[bilinear, bicubic, nearest]`** on datasets < 20 videos. Use one option at a time:
- `interpolation: [nearest]` — try first; strongest encoding-robustness signal
- `interpolation: [bilinear]` — fallback if convergence too slow (loss not decreasing after 10 epochs)

If neither improves E2E F1, disable RandomResize.

Write `phase_status.ddm_train: in_progress`, then **delegate to `/sop-ddm-finetuning` with `<dataset_id>` and `<validation_dataset_id>` (if available)**.

On completion: record `ddm_job_id`, `ddm_checkpoint`, `ddm_best_f1`, `ddm_best_loss`. Write `phase_status.ddm_train: done`.

Keep top-3 epoch ckpts by F1; delete rest + `last.ckpt`; delete stale `ddm_inference*.ckpt`. Via `docker exec <ddm_container>`:
```bash
docker exec <ddm_container> bash -c "
  ls /workspace/sop-ddm-ftms/assets/results/<job_id>/train/<job_id>/epoch_epoch=*-val*.ckpt 2>/dev/null \
    | sort -t= -k2 -rn | tail -n +4 | xargs -r rm -f
  ls /workspace/sop-ddm-ftms/assets/results/<job_id>/train/<job_id>/epoch_epoch=*-val*.ckpt 2>/dev/null \
    | grep -q . && rm -f /workspace/sop-ddm-ftms/assets/results/<job_id>/train/<job_id>/last.ckpt
  find /workspace/sop-ddm-ftms/assets/results/ -name 'ddm_inference*.ckpt' | sort | head -n -1 | xargs -r rm -f
"
```

Note: If best val/F1 < 0.5, log a `manual` note. Do NOT stop — continue evaluating.

### DDM Training Watchdog

```bash
bash ${SKILL_DIR}/scripts/watch_api_job.sh \
  DDM <DDM_BASE_URL> <JOB_ID> <LOG_PATH> <TIMEOUT> 60 \
  >> <run_dir>/watchdog_ddm.log 2>&1
```

```
tail -f <run_dir>/watchdog_ddm.log | grep --line-buffered -E "DDM_DONE|DDM_FAILED|DDM_HANG|DDM_TIMEOUT"
tail -f <run_dir>/watchdog_ddm.log | grep --line-buffered "F1_UPDATE"
```

Monitor 1: `DDM_DONE` → mark done; `DDM_FAILED` → read log tail, fix, restart; `DDM_HANG`/`DDM_TIMEOUT` → kill container, fix, restart.
Monitor 2: `F1_UPDATE` → log F1 and loss.

If Monitor silent >5 min: check `ps aux | grep watch_ddm`; restart watchdog if dead.

| Signal | Auto-fix |
|--------|----------|
| `failed` immediately (< 2 min) | Read `results/<job_id>/log.txt` last 30 lines; fix config/dataset/DB; restart |
| Loss NaN after first epoch | Halve `optm_lr`; restart |
| Loss flat for 5 epochs (delta < 5%) | Check `optm_warmup_steps`; if > 30% of total steps, reduce; restart |
| GPU = 0% for >5 min | Check `docker logs <ddm_container>`; if OOM reduce `batch_size`; if GPU lost restart container + job |
| `running` after `epochs × expected_epoch_time × 3` | Kill container, read log tail, restart |

Write `<run_dir>/progress.md` snapshot.

---

## Step 6: Phase 4 — VLM Training _(run iter 1 + on VLM training-config-change or augment-config-change)_

### Batch Size and Learning Rate

Set `train_batch_per_replica` in `train_config.toml`:
- Sufficient VRAM → 4; Low VRAM → 1. If OOM at batch=4, retry with batch=1.

**Learning rate — MODE-AWARE (LoRA vs full fine-tune). Check whether `[policy.lora]` is present in `train_config.toml`:**
- **LoRA run** (`[policy.lora]` present): small datasets (< 20 videos) use `optm_lr = [1.5e-5, 1.5e-5, 1.5e-5]`.
- **Full fine-tune run** (no `[policy.lora]`): small datasets (< 20 videos) use `optm_lr = [5e-6, 5e-6, 5e-6]`
- Larger datasets (≥ 50 videos): service default may be acceptable; validate after iteration 1.
- Repeated action pair confusion despite correct DMCQ → check LR first.

### VLM Training Watchdog

```bash
bash ${SKILL_DIR}/scripts/watch_api_job.sh \
  VLM <CR_BASE_URL> <JOB_ID> <LOG_PATH> <TIMEOUT> 120 \
  >> <run_dir>/watchdog_vlm.log 2>&1
```

```
tail -f <run_dir>/watchdog_vlm.log | grep --line-buffered -E "VLM_DONE|VLM_FAILED|VLM_HANG|VLM_TIMEOUT"
tail -f <run_dir>/watchdog_vlm.log | grep --line-buffered "status=running"
```

Monitor 1: `VLM_DONE` → proceed to 6a; `VLM_FAILED` → read log tail, fix; `VLM_HANG`/`VLM_TIMEOUT` → kill CR container, fix, restart.

If Monitor silent >10 min: check `ps aux | grep watch_vlm`; restart watchdog if dead.

| Signal | Auto-fix |
|--------|----------|
| `failed` immediately (< 5 min) | Read service container logs (dataset mount, DB, config parse); fix and restart |
| Loss NaN at any step | Cancel; halve all `optm_lr`; restart |
| Loss drops >90% early AND by-action shows collapse | Note: reduce LR next iteration; do not cancel unless NaN |
| GPU = 0% for >10 min | Check `docker logs <cr_container>`; if OOM reduce `train_batch_per_replica` to 1; if GPU reset restart container + job |
| `running` for > `epochs × ~2h` with no progress | Kill container, read log tail, restart |
| `failed` after partial training | Check `results/<job_id>/safetensors/` for valid checkpoint; eval before restarting training |

When re-running: apply RCA config changes; increment `vlm_config_version`.

Write `phase_status.vlm_train: in_progress`, then **delegate to `/sop-cr-finetuning` with `augmented_dataset_id`.**

### 6a. Record training mode

The Step 6 delegation to `/sop-cr-finetuning` runs the whole training procedure end-to-end — including its own LoRA detection and merge. On return, `<results_dir>/<timestamp>/safetensors/step_<N>/` is **always** a self-contained, inference-ready HF model.

Record training mode by checking whether cr-finetuning's merge step left a `step_<N>_lora_adapter/` sibling next to `step_<N>/`:

- **Sibling exists** → LoRA run. cr-finetuning Phase 3b created the backup during the in-place merge.
- **Sibling absent** → Full fine-tune. No merge was needed; `step_<N>/` was written as a full model directly by training.

Set in `run_state.yaml`:

| Field | LoRA run | Full fine-tune run |
|-------|----------|--------------------|
| `vlm_training_mode` | `"lora"` | `"full"` |
| `vlm_adapter_path`  | `<results_dir>/<ts>/safetensors/step_<N>_lora_adapter` | `null` |
| `vlm_inference_path`| `<results_dir>/<ts>/safetensors/step_<N>` | `<results_dir>/<ts>/safetensors/step_<N>` |

`vlm_inference_path` is the same path in both modes. No LoRA-vs-full branching is needed in Step 7.

Write `phase_status.vlm_train: done` to `run_state.yaml`.

Keep top-3 job_ids by `eval_history.e2e_seq_acc`; delete `*.safetensors` and `adapter_model.*` from others. Never delete `log.txt`, configs, or tokenizer files. Via `docker exec <cr2_container>`:
```bash
KEEP="<job_id_1> <job_id_2> <job_id_3>"  # from run_state.yaml eval_history
docker exec <cr2_container> bash -c "
  for d in /workspace/sop-cr-ftms/assets/results/*/; do
    j=\$(basename \"\$d\")
    echo '$KEEP' | grep -q \"\$j\" && continue
    find \"\$d\" -name '*.safetensors' -o -name 'adapter_model.*' | xargs -r rm -f
  done
"
```

---

## Step 7: Phase 5 — Evaluation

Run both evaluations in parallel if possible, otherwise sequentially.

**Eval is API-driven.** Both `/sop-by-action-eval` and `/sop-e2e-inference` POST to the BP `evaluation-ms` HTTP service (default `localhost:32090`) and return a structured JSON envelope on stdout. Eval-ms decides the output directory (uuid under its `RESULTS_ROOT`); the orchestrator only records the host-side paths from the envelope.

**Pre-eval requirements:**
- `run_state.training_job_id` and `run_state.val_dataset_id` must be set (populated when the training and dataset registration steps complete).
- For E2E: `run_state.ddm_training_job_id` must be set.
- Read `fps`, `max_frames`, `total_pixels` from `train_config.toml [custom.vision]` and write them into the eval inputs.yaml. Never use eval-script defaults.
- `run_state.host_results_root` must point at the host directory that docker-compose maps to `/workspace/sop-eval-ms/assets/results`. Default: `<bp_deployment_root>/assets/results`.

### 7-pre. eval-ms reachability check

```bash
curl -fsS http://${EVAL_HOST:-localhost}:${EVAL_PORT:-32090}/health
```

Non-2xx → stop and tell the user to bring eval-ms up (`docker compose up evaluation-ms` from the BP deployment root) before retrying.

### 7a. By-Action Evaluation

Generate `inputs_by_action_iter<N>.yaml` **into `<run_dir>/iter<N>/`** from `${SKILL_DIR}/references/by-action-eval-template.yaml`. Required fields: `training_job_id` (from `run_state`), `val_dataset_id`, `host_results_root`. Set `backend: transformers` for LoRA evaluations; `vllm` for full-FT. Pass `fps` from `train_config.toml`.

Write `phase_status.eval_by_action: in_progress`, then **delegate to `/sop-by-action-eval`** with the generated inputs.yaml.

**Capture the JSON envelope.** The skill emits a single JSON line on its last stdout. Parse it and persist into `run_state.eval_outputs.by_action`:

```yaml
eval_outputs:
  by_action:
    eval_job_id: <envelope.eval_job_id>
    host_output_dir: <envelope.host_output_dir>
    snapshot_dir: <run_dir>/iter<N>/by_action       # full copy made in Step 7c
    inference_results_json: <envelope.artifacts.inference_results_json>
    log: <envelope.artifacts.log>
    overall_accuracy_evalms: <envelope.headline_metrics.overall_accuracy>
    overall_accuracy_authoritative: null   # filled by analyze_by_action_confusion in Step 8b
```

| Envelope `status` | Action |
|--------|----------|
| `completed` | Record `envelope.headline_metrics.overall_accuracy` into `overall_accuracy_evalms`. Snapshot the job dir in Step 7c. Step 8b's `analyze_by_action_confusion.py` populates `overall_accuracy_authoritative`; Step 8a's success-criteria gate uses the authoritative value. Mark `phase_status.eval_by_action: done`. |
| `failed` | Read `<host_output_dir>/log.txt` tail; fix root cause; rerun `/sop-by-action-eval`. |
| `timeout` | Eval-ms job may still be running. `curl http://<eval_host>:<eval_port>/api/v1/evaluation/status/<eval_job_id>` to check before retrying. |

### 7b. E2E Evaluation

Generate `inputs_e2e_iter<N>.yaml` **into `<run_dir>/iter<N>/`** from `${SKILL_DIR}/references/e2e-eval-template.yaml`. Required fields: `training_job_id`, `val_dataset_id`, `ddm_training_job_id`, `host_results_root`. Set `score_threshold = run_state.ddm_threshold`. Pass `max_frames`/`total_pixels` through `resolution_config` and `fps` at the top level.

Write `phase_status.eval_e2e: in_progress`, then **delegate to `/sop-e2e-inference`**.

**Capture the JSON envelope** and persist into `run_state.eval_outputs.e2e`:

```yaml
eval_outputs:
  e2e:
    eval_job_id: <envelope.eval_job_id>
    host_output_dir: <envelope.host_output_dir>
    snapshot_dir: <run_dir>/iter<N>/e2e             # full copy made in Step 7c
    e2e_results_json: <envelope.artifacts.e2e_results_json>
    accuracy_json: <envelope.artifacts.accuracy_json>
    video_name_to_output_text_json: <envelope.artifacts.video_name_to_output_text_json>
    action_recognition_log: <envelope.artifacts.action_recognition_log>
    temporal_segmentation_dir: <envelope.artifacts.temporal_segmentation_dir>
    temporal_segmentation_log: <envelope.artifacts.temporal_segmentation_log>
    sop_e2e_eval_log: <envelope.artifacts.sop_e2e_eval_log>
    log: <envelope.artifacts.log>
```

| Envelope `status` | Action |
|--------|----------|
| `completed` | Read `<accuracy_json>` for `sequence_accuracy`/`action_accuracy`/`wrong`/`duplicate`/`missing`; read `<e2e_results_json>.temporal_segmentation.avg_f1` for `ddm_f1`. Mark `phase_status.eval_e2e: done`. |
| `failed` | Read `<sop_e2e_eval_log>` tail; fix root cause; rerun. |
| `timeout` | Same as 7a — query `/api/v1/e2e-evaluation/status/<eval_job_id>` before retry. |

**No file-existence polling, no watchdog scripts.** The eval skill blocks until terminal status and returns the envelope.

### 7c. Record Results

**Snapshot the eval job output(s) into `<run_dir>/iter<N>/`**. For each eval run this iteration, full-copy its job dir from the envelope's
`host_output_dir`:

```bash
# by-action (if run this iter)
mkdir -p <run_dir>/iter<N>/by_action && cp -a <eval_outputs.by_action.host_output_dir>/. <run_dir>/iter<N>/by_action/
# e2e (if run this iter)
mkdir -p <run_dir>/iter<N>/e2e        && cp -a <eval_outputs.e2e.host_output_dir>/.        <run_dir>/iter<N>/e2e/
```

**Append `eval_history` entry** in `run_state.yaml`:

```yaml
eval_history:
  - iteration: N
    by_action_acc: 0.XX
    e2e_action_acc: 0.XX
    e2e_seq_acc: 0.XX
    ddm_f1: 0.XX
    ddm_threshold: 0.XX
    changes_applied: ["description of what changed vs prev iteration"]
```

**Write progress chart** — every eval, generate `<run_dir>/progress.html`:
```python
import pathlib, json
tmpl = pathlib.Path('${SKILL_DIR}/references/progress-chart-template.html').read_text()
data = 'const RUN=' + json.dumps({
  'id': '<run_id>', 'dataset': '<dataset_name>', 'target': <seq_target_as_pct_0to100>,
  'iters': [
    # one entry per completed eval — all accuracy fields are 0-100 (not 0-1)
    # {'n':'1','ph':'I','t':'first','ba':18.2,'ea':33.3,'sq':0,'f1':0.848,'th':0.60,'d':7,'w':11,'m':6,'lr':'1.5e-5','note':'Baseline'},
  ]
}) + ';'
pathlib.Path('<run_dir>/progress.html').write_text(tmpl.replace('/* ITER_DATA */', data))
```
Schema: `n`=label, `ph`=I/II/III (I=pre-key-fix, II=key-fix-iter, III=post-fix), `t`=first/augVLM/vlm/ddm/eval, `ba`=by-action%(null if not run), `ea`=E2E-act%, `sq`=seq%, `f1`=DDM-F1, `th`=DDM-thr, `d/w/m`=dup/wrong/miss(int), `lr`=LR-string, `note`=one-line change, `samp`=total QA samples (training iters only; omit or 0 for eval-only), `qas`=object with per-type counts `{bcq,mcq,gqa,gqas,dmcq}` (omit for eval-only iters).

This is a **mandatory local-file write**. One row per eval (eval-only sweeps included — same as
`eval_history`). After writing, confirm with `=== PROGRESS (local file): <run_dir>/progress.html ===`.

---

## Step 8: Phase 6 — RCA + Decision Loop

>**IMPORTANT**: **RCA is mandatory after every failed iteration.**
> - Do Not skip RCA.
> - Do NOT read summary files and write your own RCA.
> - Do NOT use Step 8b-fallback's heuristics as a routine substitute — they exist ONLY for the case where `/sop-rca` delegation actually fails (skill missing, error, no report after retry). 
> - Always delegate to `/sop-rca`. The only sanctioned skip is `run_rca: false` in `inputs.yaml`. Step 8a's pre-flight gate (below) blocks iterations that try to skip RCA.

### 8a. Check Success Criteria

#### 8a.0 — Bookkeeping gate (RUN FIRST, before anything else in Step 8)

This is the executable form of the invariant promised above. It is a hard **STOP**, not advice.
You may not check success criteria, select an action, or declare SUCCESS/PARTIAL until ALL FOUR
assertions below pass. They are cheap; run them on entry to Step 8 every single iteration.

```python
bud = run_state.iteration_budget

# (1) Every eval that has been RUN must have an eval_history[] row. One row PER eval —
#     a 3-point threshold sweep is 3 rows (disambiguated by ddm_threshold / nms_sec / changes_applied),
#     not 1. This is what was missing when the run drifted: evals happened, rows didn't.
n_evals_run = count_evals_actually_run()   # = number of /sop-by-action-eval + /sop-e2e-inference completions this run
assert len(run_state.eval_history) == n_evals_run, (
    f"BOOKKEEPING GATE: {n_evals_run} evals run but eval_history has "
    f"{len(run_state.eval_history)} rows. Append the missing row(s) NOW — including for "
    f"eval-only threshold/NMS sweeps — before proceeding. Do not batch 'consolidate later'."
)

# (2) Every FAILED eval must have a completed /sop-rca run behind it (unless run_rca: false).
#     rca_runs_completed is incremented ONLY in Step 8b after rca_report.md is verified on disk —
#     never hand-set it. eval-only iterations are NOT exempt: a failed re-eval still needs RCA.
if inputs.run_rca:
    n_failed = len([e for e in run_state.eval_history if not meets_criteria(e, inputs.success_criteria)])
    assert bud.rca_runs_completed == n_failed, (
        f"RCA GATE: {n_failed} failed evals but only {bud.rca_runs_completed} RCA runs. "
        f"Run /sop-rca (Step 8b) on the un-analysed failed eval(s) before selecting any action. "
        f"Writing your own diagnosis instead of delegating to /sop-rca is the exact violation "
        f"this gate exists to catch."
    )

# (3) Progress files must exist and be current for THIS iteration (local artifacts, Step 2).
#     progress.md gets this iteration's row; progress.html is regenerated.
assert progress_md_exists() and last_logged_iter(progress_md) == run_state.iteration, (
    "PROGRESS GATE: progress.md missing or stale (no row for current iteration). "
    "Append this iteration's row NOW (Step 7c / Step 9)."
)
assert progress_html_exists(), (
    "PROGRESS GATE: progress.html missing. Regenerate it from the template NOW (Step 7c)."
)

# (4) Each eval run THIS iteration must be snapshotted into iter<N>/<phase>/ (Step 7c).
#     This is what gives eval-only iters a folder and pins which eval_job_id was this iter.
itdir = f"{run_dir}/iter{run_state.iteration}"
for phase in evals_run_this_iter():          # subset of {"by_action", "e2e"}
    assert snapshot_nonempty(f"{itdir}/{phase}"), (
        f"SNAPSHOT GATE: {phase} eval ran this iter but {itdir}/{phase}/ is missing or empty. "
        f"cp -a <host_output_dir>/. {itdir}/{phase}/ NOW (Step 7c) before proceeding."
    )
```

If any assertion fires, the ONLY valid next move is to do the missing work (snapshot eval output /
append rows / run RCA / write progress files), not to continue. There is no path through Step 8 that
leaves these unsatisfied.

#### 8a.1 — Success check

```python
r = latest_eval
c = inputs.success_criteria

all_met = all([
    c.ddm_f1 is None            or r.ddm_f1            >= c.ddm_f1,
    c.by_action_accuracy is None or r.by_action_acc     >= c.by_action_accuracy,
    c.e2e_action_accuracy is None or r.e2e_action_acc   >= c.e2e_action_accuracy,
    c.e2e_sequence_accuracy is None or r.e2e_seq_acc    >= c.e2e_sequence_accuracy,
])
```

If `all_met` → **SUCCESS**. Go to Step 9.

**To trigger PARTIAL, ALL of these must be true:**

```python
budget = run_state.iteration_budget
attempts = budget.attempts

partial_gate = all([
    budget.iterations_substantive >= budget.max_pipeline_iterations,
    len(attempts.augment_config_change)    >= 2,    # at least 2 distinct augment strategies tried
    len(attempts.training_config_change)   >= 1,    # at least 1 training-config-change tried
    len(attempts.ddm_training_config_change) >= 1,  # at least 1 ddm-training-config-change tried
])
```

If `partial_gate` is True → PARTIAL. Run Step 8b (RCA report is mandatory handoff), then Step 9.

**If `partial_gate` is False — KEEP ITERATING.** Judgment calls ("more data needed", "won't work") are NOT grounds for PARTIAL without exhausting the substantive budget across all fix categories.

**Common premature-stop antipatterns:**
- DDM threshold sweeps at 0.3/0.4/0.5/0.6 → eval-only, budget unused.
- DMCQ retrain regressed → alternative DMCQ mode (adjacent vs confusion) not tried yet.
- LR not verified correct → fix LR first before concluding training is stalled.

You **must** run Step 8b before anything in Step 8c.

### 8b. Run RCA (required gate before action selection)

Write `phase_status.rca: in_progress`, then **dispatch `/sop-rca` as a sub-agent** (Agent tool, e.g. `subagent_type: general-purpose`) — instruct the sub-agent to invoke the `sop-rca-plugin:sop-rca` skill with the payload below and to return exactly what that skill's "Invocation & Return Contract" specifies.

**Why a sub-agent (this is the rule, not an option):** the `/sop-rca` skill body, its helper-script output, and per-video analysis are large. Running it in a sub-agent keeps ALL of that in the sub-agent's context; the orchestrator receives only the compact `RCA_RESULT:` block. This makes RCA **always affordable no matter how many iterations run** — so there is **never** a context/token reason to skip it or hand-author the RCA yourself. Authoring your own RCA, or reading the eval files and writing your own diagnosis, is a violation (Step 8a's gate catches it); the sub-agent exists precisely to remove the temptation. The `/sop-rca` skill is path-driven and self-contained, so the fresh sub-agent has everything it needs from the payload.

The sub-agent reads from the durable `iter<N>/` snapshot (Step 7c) plus the augment/DDM/VLM config snapshots and training log — everything the payload references lives under `<run_dir>/iter<N>/`:

```
/sop-rca
  e2e_outputs_dir:       <run_dir>/iter<N>/e2e/outputs_action_recognition
  ddm_outputs_dir:       <run_dir>/iter<N>/e2e/outputs_temporal_segmentation
  by_action_results:     <run_dir>/iter<N>/by_action/inference_results.json   # JSON preferred; log.txt also accepted by analyze_by_action_confusion.py
  actions_json:          <dataset_path>/actions.json
  augment_config:        <run_dir>/iter<N>/augment_config.yaml
  ddm_training_config:   <run_dir>/iter<N>/ddm_train_config.yaml
  vlm_training_config:   <run_dir>/iter<N>/train_config.toml
  vlm_training_log:      <run_dir>/iter<N>/training.log
  output_dir:            <run_dir>/iter<N>/                                # RCA writes rca_report.md here
```

No path-discovery, no copy-of-`actions.json`. The orchestrator already knows `dataset_path`; RCA reads `actions.json` from that location directly. If it's missing, that is a setup bug — surface and stop, do not silently work around.

The sub-agent's final message is the compact **`RCA_RESULT:` block** (`status`, `report_path`, `rca_analysis_dir`, `headline_metrics`, `typed_actions`, `one_line_verdict`) — NOT the report prose. Consume it directly: read `report_path` (the report + `rca_analysis/` are on disk under `<run_dir>/iter<N>/`) and copy `typed_actions` verbatim into `run_state.yaml`'s `rca_reports:`:

```yaml
rca_reports:
  - iteration: N
    report_path: <run_dir>/iter<N>/rca_report.md      # from RCA_RESULT.report_path
    typed_actions:    # verbatim from RCA_RESULT.typed_actions
      - {action_type: eval-config-change,    pattern: 1, summary: "..."}
      - {action_type: augment-config-change, pattern: 2, summary: "..."}
```

Verify the file at `RCA_RESULT.report_path` exists. If the sub-agent returns `status: failed`, no `RCA_RESULT:` block, or the file is missing — that is a delegation failure: retry the sub-agent once, then fall to Step 8b-fallback only if it fails again (do NOT hand-author the RCA in lieu of delegating). On success, write `phase_status.rca: done` and increment `iteration_budget.rca_runs_completed`.

#### 8b-fallback: When `/sop-rca` is unavailable

**Use ONLY when delegation to `/sop-rca` actually fails** — the skill is missing, errors out, or returns no `rca_report.md` after a retry. **Not a substitute for routine RCA.** Recognising one of these patterns from metrics alone is *not* grounds to skip Step 8b — `/sop-rca` is expected to surface the same patterns plus diagnostics the heuristic table cannot replicate (per-video DDM analysis, residual error budget, signal audit).

When delegation has demonstrably failed for the current iteration:

- by-action ≈ 0% across all actions → model collapse (Pattern 9): lower LR, check non-SOP over-weighting, increase number of epochs
- by-action 0% on a specific action subset only → DMCQ coverage gap (Pattern 2): add confusion_map for that subset
- E2E seq_acc = 0 but by-action > 95% → DDM threshold too high or DDM under-segmentation: tune threshold first
- 33+ duplicates in E2E → MCQ max_chunk_len mismatch: reduce to 2
- Missing actions in E2E → DDM missed boundaries: lower threshold or retrain DDM with RandomResize
- Loss near zero but accuracy still low → format mismatch or evaluation pipeline bug: check eval container and inference params
- One duplicate at end of one video while by-action is high → un-annotated tail (mp4 longer than golden). The BP eval-ms handles trim-to-`max(annotation.end_timestamp)` internally; if the artifact still surfaces, the dataset's annotations may be incomplete — verify the val_dataset registration before iterating.

After applying a fallback diagnosis: write a brief `<run_dir>/iter<N>/rca_report.md` summarising the heuristic chosen and *why /sop-rca was unavailable*; add an `rca_reports[]` entry with `report_path` and `typed_actions`; set `phase_status.rca: done`; **do not** increment `rca_runs_completed` (it tracks successful RCA runs only); flag the failure under `notes:` so the next iteration retries `/sop-rca`.

### 8c. Interpret RCA and Select Next Action

Priority applies **only to typed actions in `rca_report.md`** — if a pattern is absent, RCA was incomplete: rerun 8b.

Apply **one action per iteration**. Emit full queue to `run_state.yaml` (diversity > depth):

```yaml
iteration_queue:
  - { iter: N+1, type: substantive, action: training-config-change, hypothesis: "Correct LR to 5e-6 (small dataset — see LR guidance) before any DMCQ tuning" }
  - { iter: N+2, type: substantive, action: augment-config-change, hypothesis: "DMCQ confusion mode addresses action pair confusion observed in by-action" }
  - { iter: N+3, type: substantive, action: augment-config-change, hypothesis: "DMCQ adjacent mode (alternative signal) if confusion mode regresses or collapses" }
  - { iter: N+4, type: substantive, action: ddm-training-config-change, hypothesis: "RandomResize bilinear to improve DDM generalization" }
```

Each remaining slot must propose a **distinct** action category or value. Repeating the same category+value does not advance the budget.

#### Priority 1: eval-config-change

**1a. DDM threshold tuning:** Under-seg (missing actions) → lower by 0.05–0.10 (min 0.35). Over-seg (duplicates) → raise by 0.05. Update `ddm_threshold` to the single RCA-recommended value. → Step 7b → back to Step 8a.

**1b. Wrong VLM inference path:** Fix `vlm_inference_path` (and `vlm_adapter_path` if LoRA). → Step 7 (both evals) → back to Step 8a.

#### Priority 2: DDM training-config-change

Apply RCA changes (epochs, resolution, batch_size) to `ddm_train_config.yaml`. Increment `ddm_config_version`. → Step 5. DDM retraining with more data is `manual`.

#### Priority 3: VLM training-config-change (medium cost — retrain VLM only)

**When both LR and confusion_map (augment-config-change) are flagged by RCA:** apply the LR fix first (this iteration); only add/extend confusion_map next iteration if confusion persists.

Apply RCA-recommended config changes to `train_config.toml`. Table = pattern → key path only; RCA value is authoritative:

| RCA pattern | Config key path(s) in `train_config.toml` |
|-------------|-------------------------------------------|
| Pattern 8 — fps/max_frames mismatch | `custom.vision.fps`, `custom.vision.max_frames`, `custom.vision.total_pixels`, `policy.model_max_length` |
| Pattern 5a — overfitting | Full FT: `train.epoch`, `train.optm_lr`; enable `train.validation_step`. LoRA: reduce `policy.lora.r` or raise `policy.lora.lora_dropout` (do NOT lower `policy.lora.lora_alpha` for this pattern). |
| Pattern 5b — underfitting | Full FT: `train.epoch`, `train.optm_lr`. LoRA priority order: 1) if `optm_lr ≤ 5e-6` (e.g. the full-FT small-data value mis-applied to LoRA), raise to ~1.5e-5 FIRST — this is the dominant underfit lever; 2) raise `policy.lora.lora_alpha` to reach `effective_scaling = lora_alpha / r ≥ 32`; 3) raise `train.epoch` (count is dataset-specific — extend until the loss curve plateaus); 4) raise `policy.lora.r` only as a last resort. Never bundle an alpha change with an `r` change in the same iteration. |
| Pattern 9 — model collapse | Non-SOP collapse → reduce `dynamic_mcq.num_neg` (and DS / EN non-SOP weights); LR-aggressive collapse → reduce `optm_lr`, raise `optm_warmup_steps`; LoRA over-scaling collapse → reduce `policy.lora.lora_alpha`. |

Increment `vlm_config_version`.
→ **Go to Step 6** (retrain VLM + merge if LoRA, then re-eval both)

#### Priority 4: augment-config-change (expensive — re-augment AND retrain VLM)

Apply config fixes to `augment_config.yaml`. See `${SKILL_DIR}/references/augment-config-guide.md`. Key rules:

- **DMCQ `num_hard_neg`**: 1 for ≤ 3 pairs; 0 for > 3 pairs.
- **MCQ `max_chunk_len`**: keep at 2; raise to 3 only if DDM under-segments 3+ chunks AND E2E shows MISSING errors.
- **GQAs**: always `enable: true`; carry Step 4 backend settings forward.

**Building the confusion_map:** Run `analyze_by_action_confusion.py` on by-action results; use only pairs from `Confusion Pairs` output. Start with 1–2 dominant pairs; expand only if they persist.

Increment `augment_config_version` and generate a new `augmented_dataset_id`.
→ **Go to Step 4** (re-augment, then retrain VLM + merge, then re-eval both)

augment-config-change does NOT require DDM retraining.

#### Priority 5: code-change

Copy file to `<run_dir>/overrides/`, fix it, wire via env var (`SOP_MONITORING_PATH`, `DDM_BASE_PATH_HOST`, etc.) or by mounting the override into the docker container. Re-run affected phase. **Never modify `plugins/`.**

#### Priority 6: manual

Log in final report. **Do NOT block. Continue with other findings.**

### 8d. Worked example — the canonical iteration loop

```
iter1 (substantive):
  Phase 1–5 (import → augment → DDM train → VLM train → eval)
  Step 7c: cp -a eval job dirs → iter1/by_action/ + iter1/e2e/   # ← snapshot, immediately
           append eval_history[0]; keep eval_outputs pointer; write progress.md/html
  Step 8a.0: bookkeeping gate passes (1 eval, 1 row, snapshots present, 0 failed-without-RCA yet)
  Step 8a.1: criteria not met
  Step 8b: delegate /sop-rca → write iter1/rca_report.md
           append rca_reports[0]; phase_status.rca: done
           rca_runs_completed: 0 → 1
  Step 8c: pick typed_action from rca_reports[0].typed_actions
           e.g. {action_type: augment-config-change, pattern: 2}
  → iter2

iter2 (substantive):
  Phase 2 (re-augment) → Phase 4 (retrain VLM) → Phase 5 (re-eval)
  append eval_history[1]
  Step 8a.0: gate passes (2 evals, 2 rows, 1 failed eval == 1 RCA run)
  Step 8a.1: criteria not met
  Step 8b: delegate /sop-rca → iter2/rca_report.md ; rca_runs_completed: 1 → 2
  Step 8c: typed_action = {eval-config-change, lower ddm_threshold to 0.50}
  → iter3 (eval-only)

iter3 (eval-only — threshold change, budget NOT incremented):
  Phase 7b only (re-eval E2E at thr=0.50; no retrain, no config snapshot)
  Step 7c: cp -a e2e job dir → iter3/e2e/   # ← eval-only STILL gets a folder + snapshot
           append eval_history[2] (ddm_threshold=0.50 in the row); write progress.md/html
  Step 8a.0: gate passes (3 evals, 3 rows, iter3/e2e snapshot present, 2 failed == 2 RCA)
  Step 8a.1: criteria not met
  Step 8b: delegate /sop-rca on the new result → rca_runs_completed: 2 → 3   # ← NOT skipped
  Step 8c: pick next typed_action
  → iter4

...

iterN (criteria met):
  Step 7c: snapshot eval dir(s) → iterN/{by_action,e2e}/; append eval_history[...]; write progress
  Step 8a.0: gate passes (snapshots present, rows current)
  Step 8a.1: all_met → SUCCESS → Step 9
```

---

## Step 9: Final Report

### Checkpoint Disk Cleanup (run after each VLM training phase)

After each VLM training phase, delete results directories from non-current jobs:

```bash
KEEP_JOBS=("${VLM_JOB_ID}" "${PREV_BEST_JOB_ID}")
for DIR in <run_dir>/results/*/; do
  JOB=$(basename "$DIR")
  if [[ ! " ${KEEP_JOBS[@]} " =~ " ${JOB} " ]] && [[ "$JOB" != "${DDM_JOB_ID}" ]]; then
    rm -rf "$DIR"
  fi
done
```

### Mid-run Progress File

Maintain `<run_dir>/progress.md` after every phase: status, timestamp, key metric, checkpoint path, any auto-fix applied, iteration number, criteria gap remaining. This is a **mandatory local-file write** (created in Step 2). The Step 8a.0 gate refuses to advance if it is missing or stale.

### Final Orchestration Report

Write `<run_dir>/orchestrator_report.md`:

```markdown
# SOP Fine-tuning Orchestration Report

**Run ID:** <run_id>
**Dataset:** <dataset_path>
**Status:** SUCCESS ✅ | PARTIAL ⚠️ | NEEDS_HUMAN 🔴
**Total iterations:** N
**Wall-clock time:** X hr Y min

## Results per Iteration

| Iter | DDM F1 | By-action | E2E action | E2E seq | DDM threshold | Changes |
|------|--------|-----------|------------|---------|---------------|---------|
| 1    | 0.412  | 26.7%     | 67.1%      | 0.0%    | 0.60          | baseline |
| 2    | 0.412  | 94.4%     | 94.3%      | 75.0%   | 0.45          | fps=4/frames=16, apply LoRA, threshold↓ |

## Final Model Paths

- DDM checkpoint: <path/to/best_ddm.ckpt>
- VLM training mode: <lora | full>
- VLM inference path: <vlm_inference_path>          # what to pass to evaluation
- VLM LoRA adapter: <vlm_adapter_path>              # omit this line for full fine-tune runs

## Criteria Status

| Criterion | Target | Achieved | Met? |
|-----------|--------|----------|------|
| DDM F1    | 0.60   | 0.412    | ❌   |
| By-action | 0.90   | 94.4%    | ✅   |
| E2E action| 0.90   | 94.3%    | ✅   |
| E2E seq   | 0.70   | 75.0%    | ✅   |

## Remaining Issues (manual action required)

- DDM F1 = 0.412 (target 0.60): DDM trained on only 10 videos; recommend annotating 10+ more
- Action 5 (black antenna) at 50%: black vs white antenna visually similar; add DMCQ confusion_map

## What Was Fixed Automatically

- Iteration 2: Applied fine-tuned LoRA (was using BASE model)
- Iteration 2: Fixed fps=1/max_frames=2 → fps=4/max_frames=16
- Iteration 2: Lowered DDM threshold 0.60 → 0.45 (reduced missing actions 46 → 8)
```