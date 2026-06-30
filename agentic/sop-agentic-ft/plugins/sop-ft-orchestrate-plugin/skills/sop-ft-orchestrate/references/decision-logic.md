# Orchestrator Decision Logic

## The Two Axes

The pipeline has two independent components that fail differently and require different fix scopes:

```
DDM axis:   threshold tuning → DDM config retrain (epochs/resolution)

VLM axis:   VLM config retrain → augment-config-change + VLM retrain
```

**Key diagnostic split:**
- `by_action_accuracy` measures VLM quality in isolation (DDM not involved)
- `e2e_seq_accuracy - by_action_accuracy` gap reveals DDM contribution

---

## RCA Action Type → Pipeline Re-run Scope

| RCA Action Type | Component | Phases to re-run | Cost |
|----------------|-----------|------------------|------|
| `eval-config-change` (threshold) | DDM | re-eval E2E only | Cheapest (~2 min) |
| `eval-config-change` (VLM ckpt) | VLM | re-eval both | Cheap (~4 min) |
| `training-config-change` (DDM) | DDM | retrain DDM → re-eval E2E | Medium (~15 min) |
| `training-config-change` (VLM) | VLM | retrain VLM + merge → re-eval both | Medium (~75 min) |
| `augment-config-change` | VLM only | re-augment → retrain VLM + merge → re-eval both | Expensive (~2+ hr) |
| `code-change` | varies | fix → re-run affected phase | varies |
| `manual` | human | nothing (log and continue) | none |

**Important:** `augment-config-change` does NOT require DDM retraining. Augmentation only produces VLM training data.

---

## Priority Order (apply one per iteration)

When multiple RCA findings exist, apply in this order to minimize wasted compute:

1. **eval-config-change** (no retraining) — cheapest possible fix
   - DDM threshold tuning (re-eval E2E only)
   - VLM checkpoint fix (re-eval both)

2. **DDM training-config-change** (retrain DDM, then re-eval E2E)

3. **VLM training-config-change** (retrain VLM, then re-eval both)

4. **augment-config-change** (re-augment + retrain VLM, then re-eval both)

5. **code-change** (implement fix, retry affected phase)

6. **manual** (log only, never blocks)

---

## State Machine

```
iteration N:
     │
     ▼
┌──────────────────────────────────────────────────┐
│  Phase 5: E2E eval + By-Action eval              │
│    → ddm_f1, by_action_acc, e2e_seq_acc recorded │
└──────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────┐
│  Check success criteria                          │
│  all met? ──────────────────────────────► DONE ✅│
│  max iterations? ───────────────────────► PARTIAL│
└──────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────┐
│  Quick triage (before full RCA):                 │
│  by_action < 0.75 → VLM bottleneck               │
│  by_action >= 0.95, seq < 0.60 → DDM bottleneck  │
└──────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────┐
│  Run /sop-rca → produce iter<N>/rca_report.md    │
│  Append entry to run_state.yaml rca_reports[]    │
└──────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────┐
│  Gate: rca_report.md exists AND has ≥1 typed     │
│  action?                                         │
│    no  → STOP (rerun /sop-rca, do NOT guess)     │
│    yes → pick highest-priority typed action      │
│          from report (never author your own)     │
└──────────────────────────────────────────────────┘
     │
     ├─ eval-config-change (DDM threshold) ─────────────────────────────────┐
     │  Update ddm_threshold in run_state                                   │
     │                                                                      ▼
     │                                                          re-eval E2E only → iter N+1
     │
     ├─ eval-config-change (wrong VLM inference path) ──────────────────────┐
     │  Set vlm_inference_path (+ vlm_adapter_path if LoRA) to correct path │
     │                                                                      ▼
     │                                                          re-eval both → iter N+1
     │
     ├─ DDM training-config-change ─────────────────────────────────────────┐
     │  Apply to ddm_train_config.yaml (more epochs, resolution, etc.)      │
     │  Increment ddm_config_version                                        │
     │                                                                      ▼
     │                                                             [Phase 3] Retrain DDM
     │                                                                      │
     │                                                             Update ddm_checkpoint
     │                                                                      │
     │                                                             re-eval E2E only → iter N+1
     │
     ├─ VLM training-config-change ─────────────────────────────────────────┐
     │  Apply to train_config.toml (fps, max_frames, LR, epochs, etc.)      │
     │  Increment vlm_config_version                                        │
     │                                                                      ▼
     │                                              [Phase 4] Retrain VLM (+ merge LoRA if applicable)
     │                                                                      │
     │                                                   Update vlm_inference_path (+ vlm_adapter_path if LoRA)
     │                                                                      │
     │                                                           re-eval both → iter N+1
     │
     ├─ augment-config-change ───────────────────────────────────────────────┐
     │  Apply to augment_config.yaml (DMCQ, GQAs, max_chunk_len, etc.)       │
     │  Increment augment_config_version                                     │
     │  New augmented_dataset_id (e.g., _v2)                                 │
     │                                                                       ▼
     │                                                          [Phase 2] Re-augment dataset
     │                                                                       │
     │                                                          [Phase 4] Retrain VLM (+ merge LoRA if applicable)
     │                                                                       │
     │                                                   Update vlm_inference_path (+ vlm_adapter_path if LoRA)
     │                                                                       │
     │                                                              re-eval both → iter N+1
     │
     ├─ code-change ─────────────────────────────── implement fix → retry phase → iter N+1
     │
     └─ manual only ─────────────────────────────── log → continue if other actions exist
                                                     OR stop → NEEDS_HUMAN
```

---

## RCA Pattern → Action Mapping (from known_failure_patterns.md)

| Pattern | Fix Type | Config Change | Re-run scope |
|---------|----------|--------------|--------------|
| 1: DDM under-segmentation | `eval-config-change` | lower `score_threshold` 0.6→0.45 | E2E only |
| 1: DDM under-segmentation (persistent) | DDM `training-config-change` | more `epochs` (e.g., 5→15) | retrain DDM |
| 1: DDM under-segmentation (VLM side) | `augment-config-change` | `sequential_mcq.max_chunk_len` 2→3 | re-augment + retrain VLM |
| 2: VLM action pair confusion | `augment-config-change` | `dynamic_mcq.enable: true`, add `confusion_map` | re-augment + retrain VLM |
| 3: DDM over-segmentation | `eval-config-change` | raise `score_threshold` | E2E only |
| 4: VLM hallucination (DDM cause) | `eval-config-change` | `nms_sec` increase | E2E only |
| 4: VLM hallucination (VLM cause) | `augment-config-change` | balance non_sop_action training data | re-augment + retrain VLM |
| 5a: VLM overfitting | VLM `training-config-change` | enable validation, reduce LR; LoRA: reduce `r`; full FT: reduce epochs | retrain VLM |
| 5b: VLM underfitting | VLM `training-config-change` | more epochs, higher LR | retrain VLM |
| 6: Insufficient training data | `manual` | annotate more videos | human needed |
| 7: Augment config issues | `augment-config-change` | fix `non_sop_action`, `max_chunk_len` | re-augment + retrain VLM |
| 8: Eval parameter mismatch | VLM `training-config-change` | fix fps/max_frames in train_config.toml | retrain VLM |

---

## Boundary Cases

**When DDM and VLM both need fixing in the same iteration:**
- Apply DDM threshold first (cheap, no retrain)
- Then evaluate: if seq_acc improves above threshold, VLM fix may not be needed
- If VLM fix still needed, apply it next iteration

**When augment-config-change and VLM training-config-change both recommended:**
- Combine both into a single re-augment + retrain VLM cycle
- Apply both config changes simultaneously before re-augmenting

**When DDM training-config-change and augment-config-change both recommended:**
- Apply DDM retrain first (DDM improvement may reduce the VLM's need for multi-chunk training)
- Re-evaluate after DDM retrain; if augment change still needed, apply next iteration

**Stopping heuristic:**
- If the same action type is recommended 2 iterations in a row with <2% improvement → escalate to manual
- If `manual` items from RCA indicate fundamental data quality issues, stop and report
