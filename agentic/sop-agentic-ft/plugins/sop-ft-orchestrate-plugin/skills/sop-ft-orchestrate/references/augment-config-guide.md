# augment_config.yaml — Annotated Template for augment-config-change

Use this when applying `augment-config-change` fixes (Priority 4). Copy and adapt
relevant sections into your `augment_config.yaml`.

---

## DMCQ (Pattern 2 — action confusion)

```yaml
# num_hard_pos is PRIMARY: mirrors eval format — all SOP actions always in options.
# num_hard_neg is SECONDARY: trains "predict non-SOP" when correct action is absent.
# Enabling DMCQ without setting num_hard_pos ≥ 1 is a no-op for Pattern 2.
dynamic_mcq:
  enable: true
  non_sop_action: <correct ID>
  confusion_map: "{A: [B], B: [A]}"   # derive from by-action confusion matrix — never assume adjacency
  num_hard_pos: 1
  hard_pos_mode: "confusion"
  num_hard_neg: 1   # safe when confusion_map has ≤ 3 pairs; set 0 for > 3 pairs (see below)
```

**num_neg / num_hard_neg scaling rule:**

- ≤ 3 confusion pairs → `num_neg=1, num_hard_neg=1` are safe
- \> 3 confusion pairs → set `num_neg=0, num_hard_neg=0`

  Why: each pair × `num_neg=1` generates one non-SOP example per clip. With 10 pairs ×
  `num_neg=1` = 10× more non-SOP than positive examples → model collapses to predicting
  non-SOP for everything. `BCQ negative_ratio=2.0` already provides non-SOP signal;
  DMCQ negatives are redundant when the map is large.

---

## MCQ max_chunk_len

```yaml
sequential_mcq:
  max_chunk_len: 2  # default=2, min=2, NEVER 1; raise to 3 only per SKILL.md Priority 4 rule (last resort)
```

**Constraints (see SKILL.md line 738 for authoritative rule):**
- `min=2`. Setting to 1 disables multi-action chunk handling → model collapses on DDM-imperfect E2E chunks.
- Combined outputs `"(N)+(N+1)"` in by-action eval → DDM threshold fix (eval-config-change), NOT a max_chunk_len change.
- Direction: 2→3 only. Only when DDM consistently under-segments into 3-action chunks AND E2E shows MISSING errors.

---

## GQAs backend

```yaml
# Always enabled. Backend is selected in Step 4 of SKILL.md; carry those settings forward here.
# Qwen/Qwen3-8B:    launch WITHOUT --reasoning-parser qwen3
# Qwen/Qwen3.5-27B: launch WITH    --reasoning-parser qwen3 (via scripts/launch_vllm.sh)
gqas:
  enable: true
  llm_type: "local"
  local_llm_url: "http://<vllm_ip>:9000/v1"
  llm: Qwen/Qwen3-8B        # or Qwen/Qwen3.5-27B
  enable_thinking: "false"
```
