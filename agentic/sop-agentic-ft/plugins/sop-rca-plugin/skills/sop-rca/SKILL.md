---
name: sop-rca
description: Root cause analysis for SOP monitoring pipeline failures. Analyzes end-to-end evaluation logs, DDM temporal segmentation, VLM action recognition, training data, and fine-tuning configs to identify failure patterns and produce an evidence-driven RCA report with actionable improvement recommendations.
license: "CC-BY-4.0 AND Apache-2.0"
---

# SOP Monitoring Root Cause Analysis Skill

You are an expert SOP (Standard Operating Procedure) Monitoring RCA agent. Your job is to analyze failure patterns in the SOP monitoring pipeline and produce an evidence-driven RCA report with actionable improvement recommendations.

## Invocation & Return Contract

**This skill is designed to be delegated as a sub-agent** (e.g. the orchestrator's Step 8b dispatches it via the Agent tool). All heavy work — the ~600-line skill body, helper-script output, per-video analysis — stays inside the sub-agent's own context. The caller pays almost nothing, so **RCA is always affordable: a calling agent must delegate this skill, never hand-author an RCA to "save context."** That tradeoff does not exist here.

To make that true, your **final message** (what the caller receives) MUST be **EXACTLY the fenced `RCA_RESULT:` block below and NOTHING ELSE** — no report title, no headings, no "Verdict:", no root-cause prose, no recommendations text, no "Report path: …" line. All narrative lives in `rca_report.md` on disk; the caller reads that file via `report_path` if it wants detail. The final message is a machine-parsed handoff, not a presentation.

```
RCA_RESULT:
  status: ok | failed            # failed only if you could not produce rca_report.md
  report_path: <output_dir>/rca_report.md
  rca_analysis_dir: <output_dir>/rca_analysis
  headline_metrics: {e2e_seq_acc, e2e_action_acc, by_action_acc, ddm_f1}   # nulls allowed
  typed_actions:                 # verbatim from report Section 5, priority order, one per line
    - {action_type, priority, confidence, summary}   # action_type ∈ {augment-config-change|training-config-change|ddm-training-config-change|eval-config-change|code-change|manual}
  one_line_verdict: "<single sentence>"
```

**Red flag — STOP if you catch yourself doing any of these in the FINAL message:** writing a `#` heading, a "Verdict"/"Root cause"/"Recommendation" sentence, a metrics table, or anything that isn't a field of the `RCA_RESULT:` block. That content belongs in `rca_report.md`, not the return. The caller parses `typed_actions` to choose the next pipeline action and checks that `report_path` exists — it does not read prose. Everything else in this document is for YOU (the sub-agent) while producing the report on disk.

## Pipeline Overview

The SOP monitoring pipeline is:
```
SOP Video (full cycle) -> DDMNet (temporal segmentation) -> segmented action chunks -> Cosmos Reason VLM -> predicted actions per chunk -> assembled action sequence -> evaluation
```

Two evaluation modes exist:
1. **End-to-end (E2E):** Full pipeline including DDM segmentation. Measures real-world performance.
2. **By-action chunk:** Perfectly segmented chunks (from annotations). Isolates VLM accuracy from DDM quality.

## Inputs

These will be provided by the user (standalone mode) or by the workflow coordinator (pipeline mode)

### Required
The upstream (user or workflow coordinator) **MUST** provide these items. Ask if any are missing.

#### Evaluation Logs

When the eval was produced by the BP `evaluation-ms`, these all live under a
single `<eval_job_id>/` directory on the host (resolved by the eval skill at
the end of a `/sop-by-action-eval` or `/sop-e2e-inference` run and passed in via
`host_output_dir`):

```
<host_output_dir>/                       # eval-ms job output (E2E)
├── e2e_results.json                     # combined summary (panel-facing)
├── outputs_action_recognition/
│   ├── accuracy.json
│   ├── video_name_to_output_text.json
│   └── action_recognition_multi_gpu.log
└── outputs_temporal_segmentation/
    ├── f1_<value>.json
    ├── video_to_boundaries_debug.json
    ├── video_to_ddm_info_debug.json
    ├── temporal_segmentation.log
    └── <video>.png
```

By-action job output:
```
<host_output_dir>/
├── inference_results.json   # {video: [[gt_action, pred_text, chunk_path], ...]}
└── log.txt                  # contains Args: {...} and "Action Chunk:" markers
```

1. **E2E action recognition logs directory** = `<host_output_dir>/outputs_action_recognition/`
   - `accuracy.json`: Overall sequence accuracy, action accuracy, per-video errors. The current BP writes `sequence_accuracy` / `action_accuracy` (legacy: `seq_accuracy` / `accuracy`) — `analyze_accuracy.py` accepts both.
   - `video_name_to_output_text.json`: VLM output text per DDM-segmented chunk per video
   - `action_recognition_multi_gpu.log`: Detailed VLM inference log (contains the `Args: Namespace(...)` line — see Step 2)
2. **DDM temporal segmentation logs directory** = `<host_output_dir>/outputs_temporal_segmentation/`
   - `f1_<value>.json`: DDM predicted boundaries with F1/precision/recall per video (`<value>` is a fixed evaluation-side tolerance, typically `0.95`).
   - `video_to_boundaries_debug.json`: Golden boundaries (midpoints of annotated segments)
   - `<video_name>.png`: Boundary visualization plots (DDM scores with golden/predicted boundaries)
   - `temporal_segmentation.log`: DDM inference log (contains its own `Args:` line)
   - `video_to_ddm_info_debug.json`: DDM predicted scores, video fps and video duration
3. **By-action chunk evaluation log** = `<host_output_dir>/log.txt` OR `<host_output_dir>/inference_results.json` — per-chunk VLM predictions and accuracy
4. **Actions definition** (`actions.json`) — provided as a path to `<eval_dataset_path>/actions.json` (orchestrator passes this; standalone callers point at the dataset). Required — stop with a clear error if missing. `actions`: list of action strings prefixed with `(N)` (the ID lives in the prefix, no separate fields); `actions_can_be_skipped`: list of skippable action strings (typically the non-SOP catch-all).


#### Fine-tuning Configs and Logs
5. **QA augmentation config** (`augment_config*.yaml`) — controls training data generation
6. **VLM fine-tuning config** (`.toml`) — VLM training hyperparameters
7. **VLM fine-tuning log or training report** — either a raw training log file containing step-by-step loss entries, or a structured training report markdown (from the upstream `sop-cr-finetuning` skill). If a training report is provided, use it directly instead of running `analyze_training_log.py`.
8. **DDM fine-tuning config** (`.yaml`) — DDM training hyperparameters

### Optional
These are NOT required, but can provide additional diagnostic value if available:
- **E2E evaluation videos** — raw SOP videos for visual inspection
- **By-action chunk evaluation videos** — per-action video chunks
- **VLM fine-tuning QAs and video chunks** — augmented training data
- **Annotated videos** — original annotated SOP cycle videos

## Analysis Procedure

Execute these steps in order. Use the helper scripts in `sop_skills/sop_rca/helpers/` for automated analysis. Run multiple independent analyses in parallel using subagents.

### Step 1: Determine Non-SOP Action ID

Before running helper scripts, read `actions.json` and identify which action is the **non-SOP action** (the catch-all action for idle/transition/not-doing-SOP periods). This is a semantic judgment — the non-SOP action is typically described as "none of the above", "not belong to the defined SOP", "non of the above", or similar. It is always a single action.

Record this action ID as `<NON_SOP_ID>` and pass it to helper scripts that accept `--non-sop-action`.

### Step 2: Extract Evaluation Parameters

Gather the parameters used during the E2E pipeline run, the by-action evaluation, and training. These feed Step 3 helpers and the Step 7 Failure Pattern 8 (Evaluation Parameter Mismatch) comparison.

| Parameter | Source | How to extract | Used in |
|---|---|---|---|
| `score_threshold`, `nms_sec`, `resolution`, `frames_per_side`, `batch_size`, `frames_per_segment_hint` | `temporal_segmentation.log` (DDM logs dir) — single `Args: Namespace(...)` line | parse log | Step 7 (Failure Pattern 8) |
| `max_frames`, `total_pixels`, `resized_height`, `resized_width`, `max_pixels`, `min_pixels`, `temperature`, `top_p`, `fps` | `action_recognition_multi_gpu.log` (E2E logs dir) — single `Args: Namespace(...)` line | parse log | Step 3b/3c (`<max_frames>`); Step 7 (Failure Pattern 8) |
| `fps_e2e` | `action_recognition_multi_gpu.log` `Args: Namespace(..., fps=N)` (BP eval-ms now reflects the request body — no longer hard-coded to 8). Confirm by parsing the log; if absent, fall back to `8` for legacy E2E runs. | parse log | Step 3b/3c (`<fps_e2e>`) |
| `chunking_algorithm`, `chunk_length_sec` | E2E driver log `sop_e2e_eval_log.txt` (top-of-file `Args: Namespace(...)` line). Tells you whether segmentation was DDM-driven or uniform-time. | parse log | Step 5a (skip DDM analysis when `chunking_algorithm=uniform`) |
| `fps_by_action`, `top_p_by_action`, `resolution_config` | By-action evaluation log (`<host_output_dir>/log.txt`) — top line `Args: {...}` (Python dict, not Namespace) | parse log | Step 7 (Failure Pattern 8 — compare against E2E fps and training fps) |
| `fps_train` | `train_config.toml` `[custom.vision]` | parse TOML | Step 7 (Failure Pattern 8 — compare against eval fps) |
| `lora_r`, `lora_alpha`, `lora_dropout`, `target_modules`, `use_rslora`, computed `effective_scaling` | `train_config.toml` `[policy.lora]` — section absent → full fine-tune run, skip LoRA branches in Step 6e and Pattern 5/9 LoRA fixes | parse TOML; `effective_scaling = lora_alpha / r` (or `lora_alpha / sqrt(r)` if `use_rslora=true`) | Step 6e (LoRA capacity audit); Patterns 5a/5b/9 LoRA-specific fix branches |

**Note:** The DDM and VLM inference logs can also be referred back to during deeper investigation if video loading errors, inference timeouts, or other runtime issues are suspected.

### Step 3: Parse and Summarize Metrics

Run the helper scripts to get structured analysis. All scripts that save JSON output accept `--output-dir` to control where files are saved.

Save analysis outputs to `<output_dir>/rca_analysis/` when an `output_dir` argument was supplied by the orchestrator (typical: `<run_dir>/iter<N>/rca_analysis/`). When invoked stand-alone with no `output_dir`, fall back to `<cwd>/rca_reports/<dataset_name>/analysis/`.

```bash
# Orchestrator-driven (preferred):
ANALYSIS_DIR=<output_dir>/rca_analysis    # e.g. <run_dir>/iter<N>/rca_analysis

# Stand-alone fallback:
# ANALYSIS_DIR=rca_reports/<dataset_name>/analysis

# 3a. End-to-end accuracy analysis
python helpers/analyze_accuracy.py <E2E_logs_dir>/accuracy.json --output-dir $ANALYSIS_DIR

# 3b. VLM output chunk analysis (use fps/max_frames extracted in Step 2)
python helpers/analyze_vlm_output.py <E2E_logs_dir>/video_name_to_output_text.json --fps <fps_e2e> --max-frames <max_frames> --non-sop-action <NON_SOP_ID> --golden-boundaries <DDM_logs_dir>/video_to_boundaries_debug.json --output-dir $ANALYSIS_DIR

# 3c. DDM boundary analysis
python helpers/analyze_ddm_boundaries.py <DDM_logs_dir>/f1_<value>.json --golden-boundaries <DDM_logs_dir>/video_to_boundaries_debug.json --ddm-info <DDM_logs_dir>/video_to_ddm_info_debug.json --fps <fps_e2e> --max-frames <max_frames> --output-dir $ANALYSIS_DIR

# 3d. By-action confusion analysis
python helpers/analyze_by_action_confusion.py <by-action chunk evaluation log> --actions-json <actions.json> --output-dir $ANALYSIS_DIR

# 3e. Training log analysis
python helpers/analyze_training_log.py <vlm fine-tuning log> --output-dir $ANALYSIS_DIR

# 3f. Training data distribution analysis (ONLY if user provided augmented data path)
python helpers/analyze_training_data.py <augmented_data_root_dir> --actions-json <actions.json> --non-sop-action <NON_SOP_ID> --output-dir $ANALYSIS_DIR
# If augmented data path was NOT provided, skip this step. Note in the report:
# "Training data distribution not analyzed (augmented data path not provided)."
# The training config dataset paths are container paths and cannot be resolved on the host.
```

### Step 3.1: Authoritative data sources for DDM vs VLM diagnosis

**`accuracy.json` `predicted` is POST-PROCESSED.** It has already had:
- Non-SOP action removed (action_id == `<NON_SOP_ID>` stripped)
- Consecutive duplicates collapsed (e.g. `[1,1,3,3,3,5]` → `[1,3,5]`)

This post-processing also applies to `accuracy_analysis.json` (Step 3a
output), because `analyze_accuracy.py` copies `predicted` and `golden`
straight from `accuracy.json` without re-deriving them. Treat both files'
`predicted` field as already-post-processed.

When diagnosing whether a failure is DDM-side or VLM-side, use these
authoritative sources first:

| To diagnose | Authoritative source | What NOT to use |
|---|---|---|
| DDM under-segmentation | `ddm_analysis.json` per-video `FN > 0` | length of `accuracy.json` / `accuracy_analysis.json` `predicted` |
| DDM over-segmentation  | `ddm_analysis.json` per-video `FP > 0` | length of `accuracy.json` / `accuracy_analysis.json` `predicted` |
| VLM per-chunk behavior | `video_name_to_output_text.json` (raw text per chunk — for case-by-case inspection) **or** `vlm_output_analysis.json` (Step 3b aggregates: `action_frequency`, multi-action chunks, problematic short/long chunks) | `accuracy.json` / `accuracy_analysis.json` `predicted` |
| Sequence correctness   | `accuracy.json` `seq_accuracy` + per-video `steps[]` **or** `accuracy_analysis.json` `seq_accuracy` + `failing_videos[].error_details` (Step 3a aggregates: `missing_action_counts`, `duplicate_action_counts`, `confusion_pair_counts`) | — |


### Step 3.2: Severity Triage

Before diving into per-video analysis, assess the overall severity from the Step 3 outputs to determine the right diagnostic path:

**Check the by-action prediction distribution** from `confusion_analysis.json` (`analyze_by_action_confusion.py` output):
- How many unique actions does the model actually predict? If 1-2 actions account for the vast majority of predictions, this is **model collapse** — a fundamentally different problem from specific action confusion.
- Compare by-action accuracy against random chance (`1 / num_actions`). If accuracy is near or below random, something fundamental is broken (e.g., LR too aggressive, data format mismatch, model loading error) — investigate the training pipeline (Step 6) before spending time on per-video DDM/VLM analysis.

**Severity levels:**
- **By-action accuracy near or below random chance + predictions dominated by 1-2 classes:** Model collapse. Prioritize Step 6 (training pipeline analysis), especially per-component LR and warmup. Per-video analysis (Step 5) will show the same failure everywhere — keep it brief.
- **By-action accuracy moderate but E2E accuracy low:** VLM is partially working but DDM or pipeline issues degrade E2E. Proceed with full per-video analysis (Step 5).
- **By-action accuracy high but E2E accuracy low:** DO NOT assume DDM is the bottleneck. First compare DDM **val/F1** (from training log) against the **E2E DDM F1** from `avg_f1` in `ddm_analysis.json` (`analyze_ddm_boundaries` output) or `Temporal Segmentation F1` in `summary.txt` (E2E evaluation output):
  - Both high and similar (gap < 0.1) → DDM generalizes fine. E2E collapse is a VLM issue (likely hallucination/duplicates — Failure Pattern 4 — or sequence-level confusion VLM didn't face on isolated by-action chunks). Focus Step 5 on VLM output patterns, not DDM boundaries.
  - Val/F1 high but E2E DDM F1 much lower (gap > 0.2) → DDM overfit to the validation split (Failure Pattern 10). Focus Step 5 on DDM quality and Step 6d on augmentation.

### Step 4: Identify Failing Videos

**Primary source:** `accuracy_analysis.json` `failing_videos[]` (Step 3a output — already pre-filtered to videos with `edit_distance > 0`).
**Fallback:** `accuracy.json` `per_video[]`, filtering to entries where `edit_distance > 0`.

For each failing video, capture the following into a working table that will populate Section 2 (Failure Inventory) of the final report:

| What to record | Source field in `accuracy_analysis.json` |
|---|---|
| Video name | `video` |
| Edit distance | `edit_distance` |
| `wrong` / `duplicate` / `missing` counts | three separate fields — a video can have non-zero values in more than one |
| Golden sequence | `golden` |
| Predicted sequence | `predicted` (post-processed — see Step 3.1) |
| Per-step error descriptions | `error_details` — strings like `"Missing detection: 6, around index: 6"` or `"Wrong detection: 6 is mis-understood as 5..."` |

For dataset-wide views across all failing videos, also note the top-level `missing_action_counts`, `duplicate_action_counts`, and `confusion_pair_counts` in `accuracy_analysis.json` — these surface which action IDs and pairs are most affected without per-video iteration.

### Step 5: Root Cause Analysis for Each Failing Video

For EACH failing video, determine which failure pattern applies by cross-referencing:

#### 5a. Check DDM segmentation quality

**Primary source:** `ddm_analysis.json` (Step 3c output) — per-video metrics, pre-computed chunk durations (DDM and golden), and pre-filtered lists of under-/over-segmented videos.
**Fallback:** `f1_*.json` (raw per-video boundaries, F1/precision/recall) — also use this when you need actual boundary positions, not just durations (e.g., for boundary-offset diagnosis in Step 7).

- Read `ddm_analysis.json` `all_videos[<this_video>]` → per-video `f1`, `precision`, `recall`, `tp`, `fp`, `fn`, plus `chunk_durations` (from DDM) and `golden_chunk_durations`.
- Compare DDM-derived chunk durations against golden chunk durations to spot mismatches in chunk count or duration distribution.
- Visually inspect the DDM boundary PNG (`<video_name>.png` in the DDM logs dir) if available.
- Look for:
  - **Under-segmentation:** video appears in `ddm_analysis.json` `videos_with_huge_chunks` (max chunk > `long_threshold`) and/or has high FN / low recall. Chunks longer than the long threshold typically contain multiple actions.
  - **Over-segmentation:** video appears in `ddm_analysis.json` `videos_with_tiny_chunks` (min chunk < `short_threshold`) and/or has high FP / low precision.

##### Score-level threshold tuning analysis

`score_threshold` and `nms_sec` are **global** evaluation parameters — applied uniformly to all videos. Tuning them affects the entire dataset, not one video. Before recommending a value, read `ddm_analysis.json` `score_threshold_summary` (populated when Step 3c is run with `--ddm-info`):

| Field | Meaning |
|---|---|
| `min_tp_score` | Smallest score among current TP boundaries across all videos. Raising threshold above this drops a real boundary somewhere. |
| `max_fp_score` | Largest score among current FP boundaries. Raising threshold above this eliminates ALL current FPs. |
| `max_missed_golden_peak_score` | Largest peak score among missed golden boundaries (FNs). If high (e.g., > current threshold), the FN was filtered by NMS, not the threshold — suggests `nms_sec` tuning, not threshold tuning. |
| `clean_fix_available_by_raising_threshold` | True if `max_fp_score < min_tp_score` — there exists a threshold range that removes all FPs without dropping any TP. |
| `fp_scores_sorted_asc` | Each value is a candidate threshold; raising the global threshold above value `v` eliminates every FP with score ≤ `v`. |
| `missed_golden_peaks_sorted_desc` | Each value is a candidate threshold; lowering the global threshold to `v` recovers the highest-confidence FNs first. |

**Pattern 3 (over-segmentation) — `score_threshold` route:** if `score_threshold_summary.clean_fix_available_by_raising_threshold == true`, recommend raising `score_threshold` to a value in `(max_fp_score, min_tp_score)`. Action type: `eval-config-change` (re-evaluate only).

**Pattern 1 (under-segmentation) — `score_threshold` route:** sort missed peaks descending. Lowering threshold to a peak value catches FNs with `peak_score ≥ that peak`. *Caveat:* lowering may admit new FP candidates not currently visible in the post-NMS pred list, so the F1 impact is approximate — re-evaluate after the change to confirm.

##### NMS sensitivity (bidirectional)

`nms_sec` tuning has **two-sided risk**, captured in `ddm_analysis.json.nms_sensitivity`. Read both halves before recommending a value.

**Raising `nms_sec`** suppresses any boundary that has a higher-score boundary within the new window. Effects:
- Eliminates FPs in `fps_suppressible_by_raising_nms_sec` whose `dist_to_higher_score_pred` is *less than* the new `nms_sec`.
- **Also** suppresses real TPs in `tps_at_risk_if_nms_sec_raised` whose `dist_to_higher_score_pred` is less than the new `nms_sec`.
- **Clean-fix test:** a clean Pattern-3 fix from raising `nms_sec` exists when `min_fp_dist_to_higher_score_pred < min_tp_dist_to_higher_score_pred` — pick a value strictly between the two. Otherwise raising `nms_sec` always trades FPs for TPs.

**Lowering `nms_sec`** admits previously-suppressed candidates. Effects:
- Recovers FNs in `fns_admittable_by_lowering_nms_sec` whose `dist_to_nearest_pred` is *greater than* the new `nms_sec` AND `peak_score ≥ current score_threshold` (so the candidate would have passed the threshold filter — not threshold-suppressed). Filter the list by these two conditions before counting.
- *Unmeasurable risk:* may also admit new FP candidates that aren't in the current pred list. The helper cannot quantify this without re-running detection — re-evaluate to confirm.

**Greedy-matcher artifacts are already partitioned out.** The helper writes a separate list, `nms_sensitivity.fns_greedy_matcher_artifacts`, containing FNs whose nearest pred is *within the F1 threshold of the golden but paired to a neighbor golden* by the greedy F1 matcher. These entries are NOT `nms_sec`-tunable — the pred is already in the output and was just claimed by a neighbor; lowering `nms_sec` will not bring back the missing golden. Treat `fns_admittable_by_lowering_nms_sec` as the canonical `nms_sec`-tunable list and do not re-apply a manual `dist_to_nearest_pred` heuristic on top.

**When to fall back to DDM retraining:** only when both `score_threshold` and `nms_sec` tuning analyses show no acceptable parameter values (e.g., overlapping score distributions, every FP very close to a TP). Action type: `ddm-training-config-change` is significantly more expensive than `eval-config-change`; recommend tuning first.

#### 5b. Check VLM predictions for the video

Use these sources together — each plays a distinct role:
- `vlm_output_analysis.json` (Step 3b output) — filter `multi_action_details[]`, `short_chunk_details[]`, and `problematic_long_details[]` to entries where `video == <this_video>` to identify which chunks are worth investigating first.
- `video_name_to_output_text.json` — full raw VLM output text per chunk; needed for any chunk-level claim ("VLM said X when expected Y") and for chunks not flagged by the analysis.
- `accuracy_analysis.json` `confusion_pair_counts` — cross-video aggregate of confused action pairs; cross-reference to see whether a confusion observed in this video is systematic or video-specific.

- For each chunk, compare the VLM output against the expected action (derived from golden boundaries and the action sequence for this video).
- Look for:
  - **Multi-action outputs in chunks that should be single-action** — start from `multi_action_details[]` for this video.
  - **Action hallucination in short/ambiguous chunks** — start from `short_chunk_details[]` or `problematic_long_details[]` for this video.
  - **Hallucination on over-segmented fragments (Pattern 3 + 4 combined).** When a chunk in `short_chunk_details[]` (this video) shows the VLM confidently predicting a SOP action — particularly when the fragment contains only the *beginning* or *ending* of a real action that resembles the start or end of several similar actions — that's Failure Pattern 4 evidence even when DDM over-segmentation (Failure Pattern 3) is the upstream cause. Note both for Step 7: DDM-side fix is primary, but DS `num_hard_neg` is a complementary VLM-side robustness fix worth surfacing when the over-segmentation can't be fully eliminated.
  - **Specific action pair confusion** — compare VLM-predicted action ID against the expected one from golden boundaries; if the same pair shows up in `confusion_pair_counts`, it's systematic, not just this video.

#### 5c. Cross-reference with by-action confusion

**Source:** `confusion_analysis.json` (Step 3d output). The by-action evaluation uses perfectly-segmented chunks, so any confusion observed there is pure-VLM (DDM-isolated). Use this to validate confusion patterns from 5b:

- Look up the confused pair (from 5b) in `confusion_pairs[]` — high `count` / `pct_of_errors` → systematic VLM confusion (Pattern 2 evidence).
- `dominant_confusion` — if a single pair dominates errors, that's strong Pattern 2 evidence.
- `per_video_errors[]` — errors concentrated in a few videos (video-specific) vs spread across many (systematic VLM issue).
- `per_action[].error_rate` — if a specific action has high error rate even on perfectly-segmented chunks, that points at a VLM training-data gap for that action, not a DDM problem.

#### 5d. Check DDM visualization (if PNGs available)
- Read the DDM boundary PNG for the failing video
- Verify whether DDM score signal exists at the missed boundary locations
- Check for noisy signal causing false positive boundaries

### Step 6: Analyze Training Pipeline

Before diving into per-knob analysis, classify each observed failure along two axes — the fix type depends on which axis the failure falls on:

- **Capability gap** — the model had the training signal but did not converge on it. Diagnostic evidence: loss curve still trending down at end of training (underfitting), loss collapsed early with collapsed predictions (LR-aggressive), or specific actions learned later than others in the loss trace. Fix type: `training-config-change`.
- **Coverage gap** — the model could not have learned this from its training data because the required discrimination signal is absent or under-weighted. Diagnostic evidence: the failure pattern maps to a specific augmentation parameter that is INACTIVE (see Step 6b' Augmentation Signal Audit), or a confusion pair never appears in hard-negative / hard-positive samples. Fix type: `augment-config-change`.

Both can coexist. When both apply, address the capability gap first (cheaper — no data regeneration) and re-evaluate; the coverage gap may disappear once the model has had enough training time to exploit the signal it already has. Only commit to data regeneration after the capability-side fix has been ruled out.

#### 6a. Training convergence and learning rate analysis
- Examine the loss curve from `training_log_analysis.json` (`analyze_training_log.py` output) (sampled at every 5% of training)
- **Per-component LR analysis:** Extract `optm_lr` from the training config as `[llm_lr, vlm_lr, linear_lr]`. Assess the magnitude of each component's LR and the warmup steps (`optm_warmup_steps`). If model collapse was detected in Step 3.2, assess LR aggressiveness by examining how quickly the loss converged (loss dropping 80-90% in the first 10-15% of training suggests too-aggressive LR) and computing the warmup ratio (`optm_warmup_steps / total_steps`).
- **Distinguish convergence issues:** Use both the loss curve AND the by-action prediction distribution (from Step 3.2) to diagnose:
  - If loss converges early AND model retains discriminative ability (diverse predictions, moderate by-action accuracy) → **overfitting** (Pattern 5a)
  - If loss converges early AND model collapses to 1-2 predictions (by-action near random) → **model collapse** (Pattern 9) — investigate LR aggressiveness and warmup
  - If loss is still decreasing at end of training → **underfitting** (Pattern 5b)
  - **If by-action validation accuracy < 95% → likely underfitting (Pattern 5b)**, even when the loss curve looks flat. Pursue 100% by-action accuracy as the training target: by-action eval runs on *golden* (perfectly-segmented) chunks, which is the easiest possible inference setting for the VLM. In real E2E, DDM-segmented chunks are noisier, so every percentage point of by-action accuracy below 100% magnifies into a larger E2E accuracy drop. Do not declare a VLM "trained enough" while by-action is below ~0.95 — see Pattern 5b for fixes.
- Check LR schedule (constant vs decay)
- Check if epochs are appropriate for dataset size. **Empirical SOP fine-tuning range: 5–20 epochs.** Use this as a sanity-check anchor only, not as a hard rule — confirm against the loss curve and by-action accuracy before recommending an epoch change.

#### 6b. Training data quality
- **MCQ is the evaluation format.** The MCQ prompt format (all actions listed with original SOP action IDs) is exactly what the model sees during inference. Prioritize MCQ-specific analysis:
  - Check per-action MCQ sample counts — if a confused action pair has low MCQ coverage, that directly explains weak discrimination during evaluation
  - Check MCQ `max_chunk_len` — this must cover the number of actions per chunk the model encounters during E2E evaluation (typically 1-2, but 3-4 when DDM under-segments)
  - Check MCQ multi-action answer distribution — are all action combinations represented?
- DMCQ uses dynamic option numbering (NOT SOP action IDs), so it trains action understanding but with a format mismatch vs evaluation. Still useful but less directly relevant than MCQ.
- Identify DMCQ answer imbalance (non-SOP action > 50%)
- Check action distribution across all QA types
- Verify `non_sop_action` consistency across config sections
- **If augmented data was not provided**, assess training data quality indirectly from:
  - By-action confusion results: high error rates on specific actions may indicate insufficient training samples for those actions
  - Augment config: which QA types are enabled/disabled, MCQ `max_chunk_len` setting

#### 6b. Augmentation Signal Audit (parameter-level)

Beyond checking whether each QA type is enabled/disabled, audit the **parameter-level dials** of each enabled QA type to determine what discrimination signal is actually being produced. An augmentation type that is `enable: true` may still contribute zero signal for a given failure pattern if its sub-parameters leave the relevant sub-feature dormant.

Produce an inventory of each enabled QA type's parameters, marking each dial as ACTIVE (producing signal) or INACTIVE (dormant despite the QA type being enabled). Refer to `references/data_generation_logic.md` for the precise semantics of each parameter.

**BCQ:** `negative_ratio` (always active when BCQ is enabled — controls Yes/No balance). Note: "No" answers still reveal the correct action in the sentence, so BCQ contributes positive-action grounding even on negative samples.

**Sequential MCQ (evaluation format):**
- `max_chunk_len = 1` → only single-action chunks trained; multi-action chunk handling INACTIVE
- `max_chunk_len ≥ 2` → single-action and multi-action chunks both active

**DMCQ** — four independent sample-type dials, each governing one axis:
- `num_pos > 0` → correct-in-options training (always worth having when DMCQ is on)
- `num_hard_pos > 0` AND `hard_pos_mode ∈ {"adjacent", "confusion", "adjacent,confusion"}` → confusion-aware "pick correct from confusable options" training. If mode is `"confusion"`, `confusion_map` must be set.
- `num_neg > 0` → correct-not-in-options → non-SOP answering (contributes to non-SOP answer weight)
- `num_hard_neg > 0` AND `hard_neg_mode ∈ {"adjacent", "confusion", "adjacent,confusion"}` → confusion-aware "reject tempting wrong option when correct is missing" training.
- **Defaults: `num_hard_pos = 0`, `num_hard_neg = 0`.** DMCQ enabled with defaults provides NO confusion-aware training — `num_pos`/`num_neg` alone.

**DS (Dynamic Shuffling):**
- `num_runs > 0` → shuffled-frame incoherent-video non-SOP training (active when DS enabled)
- `num_hard_neg > 0` → coherent-but-mixed video non-SOP training (the harder case). Default: 0.

**EN (Extra Negative):** `num_runs > 0` → cross-SOP real-video non-SOP training.

**Golden GQA / GQAs:** Golden GQA has a single enable flag; GQAs additionally has `num_qa_llm` and `num_qa_per_chunk` governing diversity and chunk coverage.

**Cross-reference the inventory against failure patterns identified in Step 7:**
- Pattern 2 (action pair confusion) → `num_hard_pos` with `hard_pos_mode: "confusion"` is PRIMARY (mirrors eval format — all actions always in options), `num_hard_neg` with `hard_neg_mode: "confusion"` is SECONDARY. If either is INACTIVE despite DMCQ being enabled, this is a coverage gap and the recommendation must specify the parameter, not just the QA type.
- Pattern 4 (VLM hallucination on transitions) → DS `num_hard_neg` and EN. If both are INACTIVE, the model has never seen coherent-but-wrong examples.
- Pattern 9 (model collapse to non-SOP specifically) → high DMCQ `num_neg` / heavy DS or EN weight relative to `num_pos + num_hard_pos`. Non-SOP over-weighting, not LR, is the root cause when the collapsed class is specifically non-SOP.
- Pattern 1 Fix 2 (DDM under-segmentation, VLM-side) → `sequential_mcq.max_chunk_len = 1` leaves multi-action training INACTIVE.

When a parameter-level dial is INACTIVE and directly maps to an observed failure pattern, the recommendation in Section 5 of the report must cite the specific parameter(s) to change, not just the QA type. Emitting "enable DMCQ" when the real fix is "set `num_hard_pos ≥ 1` with `hard_pos_mode: confusion`" is a no-op recommendation.

#### 6c. Augment config issues
- Check for `non_sop_action` mismatches
- Check for missing `confusion_map` when confusion pairs are identified

#### 6d. DDM augmentation analysis (only when val/F1 and E2E DDM F1 diverge)

This step is about DDM generalization. Before recommending augmentation, rule out the case where DDM is fine and VLM is the real problem.

Compare the DDM training val/F1 against the E2E test DDM F1 from `avg_f1` in `ddm_analysis.json` (`analyze_ddm_boundaries` output) or `Temporal Segmentation F1` in `summary.txt` (E2E evaluation output):
- **Gap > 0.2 (val high, E2E low):** DDM overfitting — proceed with this step, see Pattern 10.
- **Gap < 0.1 (both high, e.g. val 0.95 and E2E 0.97):** DDM generalizes correctly. Do NOT recommend augmentation. The E2E sequence-accuracy drop is a VLM problem (check duplicate count in `summary.txt`; if high → Pattern 4 hallucination). Skip to VLM-side fixes.
- **Gap between 0.1 and 0.2:** Ambiguous. Examine per-video F1 in `f1_*.json` — a handful of bad videos dragging the average points at VLM; uniform degradation points at DDM generalization.


Read the DDM training config to check current augmentation settings:
```bash
grep -A 20 "augmentation:" <ddm_train_config.yaml>
```

If all augmentations are disabled and the val→test gap is large, recommend trying augmentation:
- **`RandomResize` first** — most effective for E2E quality; reduces encoding/resize artifact sensitivity
- **`ColorJitter`** — try separately if lighting varies across cameras; can raise DDM F1 but may reduce E2E accuracy; always validate with full E2E evaluation, not just val/F1
- **`GaussianBlur`: do NOT recommend** — blurs frame-to-frame differences that DDM relies on for boundary detection; consistently hurts performance
- **Never recommend all three at once for datasets < 20 videos** — stacking augmentations is too aggressive for small datasets and degrades val/F1 and E2E simultaneously

#### 6e. LoRA capacity audit (only when `[policy.lora]` is present in train_config.toml)

>**IMPORTANT**: This step is about LoRA-specific hyperparameter levers. **Skip entirely when `train_config.toml` has no `[policy.lora]` section** — those runs are full fine-tuning and the existing Pattern 5a/5b/9 full-FT diagnostics already cover them.

Trigger conditions (all must hold for 6e to produce LoRA recommendations):
- `[policy.lora]` is present in the training config.
- By-action authoritative accuracy (from `confusion_analysis.json`) is below 95%.

When triggered, run three sub-checks. Each may emit a typed-action recommendation. Emit a separate recommendation per sub-check that fires — do not bundle multiple LoRA-knob changes into one iteration.

##### 6e.1 — Effective scaling check (single strongest LoRA signal)

Compute `effective_scaling = lora_alpha / r` (or `lora_alpha / sqrt(r)` when `use_rslora=true`).

- `effective_scaling < 16` → **strong underfit signal.** The LoRA delta magnitude is small relative to the rank; the model has the representational dimensions but cannot apply learned corrections with enough magnitude at inference. Emit a Pattern 5b LoRA-fix recommendation: raise `policy.lora.lora_alpha` so scaling reaches ≥ 32 (keep `r` fixed; doubling alpha is a safe first step).
- `16 ≤ effective_scaling < 32` → **soft underfit signal.** Only recommend the alpha bump when by-action is well below target (e.g., < 95%) and no cheaper augmentation lever is available.
- `effective_scaling ≥ 32` → scaling is in the productive range. Do NOT recommend raising alpha further as a first-line fix. If `effective_scaling > 128` AND collapse signals are present (see Step 3.2), route to Pattern 9 LoRA over-scaling branch instead.

##### 6e.2 — LoRA epoch budget check

Read `train.epoch` and the loss-curve sampling from `training_log_analysis.json`.

- If the loss curve was still descending at end of training (loss at 95-100% noticeably lower than at 85-90%) → LoRA was under-trained. Emit a Pattern 5b LoRA-fix recommendation: raise `train.epoch`. The right epoch count is **dataset-specific**. Observe the prior run's loss trajectory and extend until convergence. As a soft rule of thumb only, LoRA on small datasets typically needs more epochs than full fine-tuning to reach equivalent loss because only the LoRA path receives gradient.
- If the loss has clearly plateaued AND by-action is still < 95% → epoch budget is not the binding constraint. Do NOT emit an epoch recommendation; the bottleneck is elsewhere (scaling, augmentation coverage, or hard capacity).

##### 6e.3 — r-tuning churn detection

Examine prior `rca_reports[]` entries and any `iteration_queue` history in `run_state.yaml` to determine whether an earlier iteration already raised `policy.lora.r` without raising `policy.lora.lora_alpha`.

- If yes AND by-action did not improve by > 2 pp after that change → flag as **wrong-lever churn**. Emit a Pattern 5b LoRA-fix recommendation: raise `policy.lora.lora_alpha` instead. The LoRA update rule `(alpha/r) · B·A` means raising r alone *reduces* effective scaling, often hurting rather than helping by-action accuracy.

##### Output priority

When 6e fires alongside Pattern 2 (action pair confusion), emit recommendations from both. The orchestrator's Step 8c selects priority via the `typed_actions` list. General ordering guidance:
- If 6e.3 (wrong-lever churn) fires → highest priority.
- If LoRA `effective_scaling < 16` → list the LoRA-alpha fix first regardless of other signals. Augmentation improvements are unlikely to land while the LoRA delta cannot be applied with enough magnitude.
- If a dominant confusion pair exists AND a DMCQ-confusion / DMCQ-adjacent treatment for that pair has NOT been tried yet → list the augment-config-change first (cheaper, more targeted).
- Otherwise → list augment-config-change recommendations before LoRA-knob recommendations (augment changes typically have larger expected impact and don't risk numerical instability).

Action type for DDM augmentation recommendations: `ddm-training-config-change`

### Step 7: Map Failures to Patterns

Refer to `sop_skills/sop_rca/references/known_failure_patterns.md` for the catalog of known patterns. Match each failing video to one or more patterns:

**Known patterns:**
1. DDM Under-segmentation
2. VLM Action Pair Confusion
3. DDM Over-segmentation
4. VLM Hallucination
5. Training Convergence Issues (Overfitting / Underfitting)
6. Insufficient Training Data
7. Data Generation Config Issues
8. Evaluation Parameter Mismatch
9. Model Collapse / Catastrophic Forgetting
10. DDM Overfitting — High Val F1 but Poor E2E Sequence Accuracy

**Important:** Not every failure will fit a known pattern. If a failure does not match any known pattern, investigate independently and document it as a **novel failure pattern**. Common uncovered failure types include:

- **VLM output parsing failure** — VLM produces malformed text, empty responses, or free-form text that doesn't match the expected `(N)action` regex. Check `video_name_to_output_text.json` for responses that lack any `(N)` pattern.
- **DDM boundary offset** — DDM detects the correct number of boundaries but places them shifted in time, causing chunks to straddle two actions. Diagnose by comparing DDM boundary positions with golden boundaries — look for consistent time offsets rather than missing/extra boundaries.
- **Action ordering errors** — VLM identifies all correct actions within a multi-action chunk but in the wrong temporal order. Check multi-action VLM outputs where the action sequence doesn't match the golden order.
- **Video quality / encoding issues** — Corrupted frames, codec errors, or resolution changes mid-video. Check VLM inference logs for video loading warnings or unusually short frame counts.
- **Prompt / action definition mismatch** — The evaluation MCQ prompt differs from the training prompt (different wording, option ordering, or action count). Compare the VLM prompt in the evaluation log against the training MCQ format.
- **Token budget overflow** — Very long video chunks exceeding `model_max_length` token limit, causing visual token truncation. Calculate: `total_frames * pixels_per_frame / PIXELS_PER_TOKEN` and compare against `model_max_length`.

When a novel pattern is found, document it with the same structure as known patterns (symptoms, evidence, mechanism, recommended fix) and flag it for addition to `known_failure_patterns.md`.

### Step 7.1: Residual Error Budget

Before generating the report, reconcile the total observed error budget against the pattern matches from Step 7:

1. **Enumerate observed failures** — from Step 4 (failing videos, edit distance > 0) and from by-action confusion (failing chunks per action).
2. **For each Step 7 pattern match, estimate the error budget share** it explains — the number and fraction of the observed errors attributable to that pattern. Example: *"Pattern 2 (Action 7↔8 confusion) explains 4 of 5 by-action errors (80%) and 6 of 6 e2e action errors (100%)."*
3. **Identify residual errors** — failures not explained by any matched pattern. List each one explicitly:
   - The specific video / chunk / action involved
   - Whether it might fit a novel pattern (see Step 7 novel-pattern guidance) or is genuinely unexplained given current evidence
4. **Project post-fix impact** — if all high-confidence fixes (see Step 8 Confidence Tiers) were applied, what fraction of the error budget would remain? This determines whether the next iteration is likely to reach the orchestrator's success criteria or will need exploratory or human action.

Record the residual error budget accounting in Section 4 (Contributing Factors) of the report under a dedicated "Residual Error Budget" subsection.

### Step 8: Generate RCA Report

**Output location — orchestrator-driven path takes precedence.**

If the caller passed an explicit `output_dir` argument (the orchestrator does this from Step 8b's delegation contract — see `sop-ft-orchestrate-plugin` SKILL.md "Output Directory Contract"), save the report and analysis JSONs under that path:

```
<output_dir>/
├── rca_report.md          # this skill's formal report
└── rca_analysis/          # Step 3 helper-script JSONs
    ├── accuracy_analysis.json
    ├── confusion_analysis.json
    ├── ddm_analysis.json
    └── ...
```

Typical orchestrator-supplied `output_dir` is `<run_dir>/iter<N>/`. This keeps every artifact for one run inside one tree and makes resumes / archives a single `mv`.

**Stand-alone fallback (no `output_dir` argument).** When invoked directly by a user with no `output_dir`, fall back to the legacy path so existing workflows still work:

```
<cwd>/rca_reports/<dataset_name>/<report_name>_run_<n>.md
```

Auto-detect `<n>` by counting existing `<report_name>_run_*.md` files (add 1; start at 1). Create the `rca_reports/<dataset_name>/` directory if missing.

**Your final message is NOT the report.** After writing `rca_report.md` to disk, your final message MUST be ONLY the `RCA_RESULT:` block from "Invocation & Return Contract" (top of this document), populated from this report's headline metrics and Section 5 typed actions. Do NOT restate the report, the verdict, or the recommendations as prose — re-read the Red flag in that section. The full report already lives at `report_path`.


#### Action Types

Each recommendation must include an action type that tells the orchestrator what downstream workflow to trigger:

| Action type | Triggers |
|---|---|
| `augment-config-change` | data regen → VLM retrain → eval |
| `training-config-change` | VLM retrain → eval |
| `ddm-training-config-change` | DDM retrain -> eval |
| `eval-config-change` | re-evaluate only |
| `code-change` | implement → re-evaluate |
| `manual` | flag for human, always lowest priority |

#### Confidence Tiers

Each recommendation additionally carries a confidence tag that governs how the orchestrator applies it:

- **high-confidence** — emitted when a specific pattern from `known_failure_patterns.md` matches the observed evidence and the fix directly addresses that pattern's diagnostic criteria. Orchestrator applies autonomously in priority order.
- **exploratory** — emitted when the Residual Error Budget (Step 7.1) shows errors not explained by any matched pattern, AND a currently-INACTIVE augmentation parameter (identified via the Signal Audit in Step 6b) provides a discrimination signal relevant to the failure class. Exploratory recommendations are best-guess coverage-gap fills — they are not backed by a pattern match, only by an evidence-consistent theory. The orchestrator applies these **one at a time per iteration** so impact is attributable. Never bundle two exploratory fixes in the same iteration — if the combined result improves, you cannot tell which fix helped.

Label each Section 5 recommendation with its confidence tier. Within a confidence tier, order recommendations by expected impact (HIGH / MEDIUM / LOW priority).

Produce a structured Markdown report with these sections:

```markdown
# SOP Monitoring RCA Report

**Model:** <model name>
**Config:** <training config name>
**Checkpoint:** <checkpoint info>
**Date:** <analysis date>

## 1. Overall Metrics Summary
<Table with seq_accuracy, action_accuracy, by_action_accuracy, DDM F1>

## 2. Failure Inventory
<Table listing each failing video with error type and root cause category>

## 3. Root Cause Analysis
<For each failure pattern found, include:>
### 3.N Failure Pattern: <name>
**Affected videos:** <list>
**Evidence:** <specific numbers, file references, visual evidence>
**Mechanism:** <how this pattern causes the observed failures>

## 4. Contributing Factors
### 4.1 Training Data
<Table of data adequacy factors>
### 4.2 Training Convergence
<Overfitting or underfitting assessment>
### 4.3 Configuration
<Config parameter comparison table>
### 4.4 Config Issues
<Any detected config issues>

## 5. Recommended Improvements
<Ordered by expected impact. Only include automatable actions (not manual). Each with:>
### 5.N [PRIORITY] [CONFIDENCE] Fix Description
**Action type:** <augment-config-change | training-config-change | eval-config-change | code-change>
**Confidence:** <high-confidence | exploratory>
**Target file:** <specific file to modify>
**Action:** <specific config change or command — cite individual parameters, not just QA type toggles>
**Rationale:** <evidence-based reasoning, linking to specific Step 6b' Signal Audit findings and Step 7 pattern matches>
**Expected impact:** <which failures this addresses>
**Error budget share:** <fraction of residual errors from Step 7.5 this fix addresses>

## 6. Summary Table
<Failure pattern -> videos -> root cause -> primary fix -> secondary fix>

## 7. Future / Human-Required
<Only include recommendations with action type "manual". These require human intervention and are not actionable by the orchestrator. Each with:>
### 7.N [PRIORITY] Description
**Action type:** manual
**Rationale:** <why this requires human action>
**Suggested action:** <what the human should do>
```

## Key Principles

1. **Evidence-driven:** Every conclusion must trace to specific data (numbers, log entries, visualizations)
2. **Specific and actionable:** Recommendations must include exact config changes, not generic advice
3. **Prioritized:** Recommendations ordered by expected impact (HIGH/MEDIUM/LOW)
4. **Pattern-based:** Map each failure to a known pattern from the reference catalog
5. **Cross-validated:** Check the same failure from multiple data sources (E2E + by-action + DDM + training)

## Reference Documents

Read these embedded references during analysis:
- `sop_skills/sop_rca/references/known_failure_patterns.md` — Catalog of failure patterns with diagnostic criteria and fixes
- `sop_skills/sop_rca/references/e2e_evaluation_logic.md` — How evaluation pipeline works
- `sop_skills/sop_rca/references/data_generation_logic.md` — How training QA data is generated
- `sop_skills/sop_rca/references/available_training_params.md` — VLM training config parameters
- `sop_skills/sop_rca/references/vision_config_params.md` — Vision config parameters

## Helper Scripts

Located in `sop_skills/sop_rca/helpers/`:
- `analyze_accuracy.py` — Parse accuracy.json, identify failing videos and error types
- `analyze_vlm_output.py` — Analyze VLM output chunks (durations, multi-action, frame sampling)
- `analyze_ddm_boundaries.py` — Analyze DDM segmentation quality (F1, under/over-segmentation)
- `analyze_by_action_confusion.py` — Build confusion matrix from by-action evaluation
- `analyze_training_log.py` — Parse training log for loss curve (sampled every 5%), LR schedule, convergence assessment
- `analyze_training_data.py` — Analyze training data distribution and adequacy
- `analyze_augment_config.py` — Detect known config issues in augment YAML
- `extract_video_chunks.py` — Extract video chunks from full videos using DDM boundaries (requires ffmpeg)
