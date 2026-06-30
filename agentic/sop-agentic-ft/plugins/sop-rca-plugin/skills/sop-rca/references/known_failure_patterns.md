# Known SOP Monitoring Failure Patterns

This document catalogs known failure patterns in the SOP monitoring pipeline
(DDMNet temporal segmentation + Cosmos Reason VLM action recognition).
Each pattern includes diagnostic criteria and common fixes. The listed fixes are not exhaustive — consider additional context-specific solutions based on the evidence in the current analysis.

## Pattern 1: DDM Under-segmentation

**Symptoms:**
- Missing actions in end-to-end evaluation (edit distance > 0, missing > 0)
- DDM F1 < 0.85 and/or Recall < 0.80 for affected videos
- Chunks exceeding the long threshold (computed as `(max_frames / fps) * long_ratio`) containing multiple SOP actions
- VLM predicts fewer actions than golden for large chunks

**Diagnostic Criteria:**
- `f1_*.json`: Check for videos with high FN (False Negative) counts
- `video_name_to_output_text.json`: Look for chunks exceeding the long threshold with non-idle VLM predictions. Use the long threshold from `analyze_vlm_output.py` / `analyze_ddm_boundaries.py` output.
- DDM boundary plots: Look for regions with flat/zero DDM score signal where golden boundaries exist
- Score-level diagnosis (when `--ddm-info` was passed to Step 3c): read `ddm_analysis.json` `score_threshold_summary.missed_golden_peaks_sorted_desc` AND `nms_sensitivity.fns_admittable_by_lowering_nms_sec`. Each missed-golden entry has both `peak_score` and `dist_to_nearest_pred`. Interpretation: peak below `score_threshold` → threshold-filtered (lower threshold); peak above threshold AND `dist_to_nearest_pred` close to the current per-video `nms_sec` → NMS-suppressed (lower `nms_sec`); peak very low (e.g., < 0.3) → DDM signal genuinely weak, retraining is needed. The helper partitions greedy-matching artifacts (pred in output but paired to a neighbor golden) into `nms_sensitivity.fns_greedy_matcher_artifacts` — those are NOT `nms_sec`-tunable; fix requires upstream DDM detection diversity or an optimal-assignment F1 matcher, not `eval-config-change`. **Both threshold and NMS tuning have asymmetric risks** — see SKILL.md Step 5a "NMS sensitivity (bidirectional)" before recommending a value.

**Root Causes:**
- DDM `score_threshold` too high (default 0.6) — boundary signals in 0.3-0.5 range get filtered. Note: `score_threshold` is the boundary filtering parameter; `ddm_threshold` in `f1_*.json` is a different parameter used only for F1 metric calculation.
- DDM model lacks sensitivity for certain action transitions (especially at video start/end)
- Long idle periods before SOP cycles suppress DDM signal

**Common Fixes:**
1. Lower DDM `score_threshold` (e.g., from 0.6 to 0.5 or 0.45) — re-evaluate with the new threshold
2. Increase `sequential_mcq.max_chunk_len` > 1 so VLM can handle multi-action chunks
3. If DDM consistently fails for certain video segments, consider retraining DDM with more data
4. **If val/F1 looks good but E2E sequence accuracy is still low:** do NOT immediately blame DDM. First compare val/F1 against the E2E `Temporal Segmentation F1` from `summary.txt`. If both are similar and high (gap < 0.1), DDM generalizes fine — the E2E drop is a VLM issue (Pattern 4 hallucination is common when duplicates are high). Only when val/F1 is much higher than E2E F1 (gap > 0.2) should you suspect DDM overfitting and enable image augmentation — see Pattern 10.

---

## Pattern 2: VLM Action Pair Confusion

**Symptoms:**
- By-action accuracy < 97% with dominant confusion between two specific actions
- One action pair accounts for >50% of by-action errors
- Confusion is asymmetric (A->B much more common than B->A)
- Confusion occurs across many videos (not video-specific)

**Diagnostic Criteria:**
- By-action confusion matrix: Look for a single (GT, Pred) pair dominating errors
- Check if confused actions are visually similar (same component type, nearby locations)
- Check if confused actions involve the farther/smaller part of the workspace

**Root Causes:**
- Visually similar actions with subtle spatial differences (e.g., left vs right, near vs far)
- No confusion-aware hard negatives in DMCQ training data
- Camera angle makes spatial distinctions difficult for farther objects
- Training data may not emphasize the distinguishing features

**Common Fixes:**
1. Enable DMCQ confusion-aware training for the confused pair. Set **both** hard-positive and hard-negative dials — they train different but complementary skills:

   - **Hard positive** (PRIMARY for Pattern 2): correct action is in the options alongside a confusable distractor — model must pick the correct one. This directly mirrors the evaluation format (all SOP actions are always present in the option list during eval), so hard positive is the closest training analog to the eval-time discrimination task.
   - **Hard negative** (SECONDARY): correct action is NOT in the options but the confusable distractor IS — model must answer non-SOP instead of jumping to the plausible-but-wrong option. Prevents hallucination to the confusion twin when the option set is trimmed.

   ```yaml
   dynamic_mcq:
     hard_pos_mode: "confusion"
     hard_neg_mode: "confusion"
     confusion_map: "{A: [B], B: [A]}"
     num_hard_pos: 1
     num_hard_neg: 1
   ```

   Important: DMCQ defaults `num_hard_pos = 0`, `num_hard_neg = 0`. Simply setting `dynamic_mcq.enable: true` without raising these counts is a no-op for this pattern — the agent must recommend specific parameter values, not just toggle the QA type.

2. Add more training examples specifically for the confused actions
3. Consider camera angle adjustment if spatial resolution is the limiting factor

---

## Pattern 3: DDM Over-segmentation

**Symptoms:**
- DDM has high FP (False Positive) count for affected videos
- Very short chunks created by extra false-positive boundaries
- More predicted boundaries than golden boundaries

**Diagnostic Criteria:**
- `f1_*.json`: Check for videos with high FP counts
- Chunk duration analysis: Look for chunks shorter than the short threshold
- DDM boundary plots: Look for noisy score signal with multiple overlapping peaks
- Score-level diagnosis (when `--ddm-info` was passed to Step 3c): read `ddm_analysis.json` `score_threshold_summary` AND `nms_sensitivity`. **`score_threshold` route:** if `clean_fix_available_by_raising_threshold == true`, raising global `score_threshold` to a value in `(max_fp_score, min_tp_score)` eliminates all FPs without dropping any TP. Otherwise `fp_scores_sorted_asc` is the candidate-threshold ladder. **`nms_sec` route:** read `nms_sensitivity.fps_suppressible_by_raising_nms_sec` and `tps_at_risk_if_nms_sec_raised`. A clean NMS-route fix exists when `min_fp_dist_to_higher_score_pred < min_tp_dist_to_higher_score_pred` — pick `nms_sec` between them. Otherwise raising `nms_sec` always co-suppresses some TPs. See SKILL.md Step 5a "NMS sensitivity (bidirectional)" for the full procedure.

**Root Causes:**
- Noisy DDM score signal causing false positive boundary detections
- DDM `score_threshold` too low for noisy videos (threshold-precision tradeoff)

**Common Fixes:**
1. Adjust DDM `score_threshold` per-video or globally (balance with Pattern 1 — raising threshold reduces FP but may increase FN)
2. Post-process DDM boundaries: merge chunks shorter than a minimum duration
3. Apply stronger NMS (increase `nms_sec` parameter which default is 0.025 * video_duration when `nms_sec` is set to 0)
4. **Complementary VLM-side robustness fix.** When over-segmentation cannot be fully eliminated (small datasets, hard-to-detect boundaries), also strengthen the VLM's fallback-to-non-SOP behavior on short fragments via DS `num_hard_neg ≥ 1`. The model then learns to output non-SOP on coherent-but-ambiguous fragments instead of confidently hallucinating one of several similar-looking actions. This is a complement to fixes 1–3, not a replacement — DDM-side fixes are cheaper and more direct. See Pattern 4 fix #5 for the DS `num_hard_neg` rationale.

**Note:** DDM over-segmentation may or may not cause end-to-end errors. It is harmless when:
- The extra chunks are in idle/transition regions and VLM predicts non-SOP action for them (filtered out during evaluation)
- A single SOP action is split into multiple chunks but VLM predicts the same correct action for all of them (consecutive duplicates are removed by `remove_continuous_rep` during evaluation, e.g., [1, 1, 1] → [1])

It causes errors when:
- VLM predicts inconsistently across split chunks of the same action, e.g., action 1 split into 3 chunks but VLM predicts [1, X, 1] → after dedup this becomes [1, X, 1] with a spurious action inserted
- VLM hallucinates a wrong SOP action for an extra chunk in a transition region (Pattern 4)

---

## Pattern 4: VLM Hallucination

**Symptoms:**
- Duplicate/extra actions in end-to-end evaluation (duplicate > 0)
- VLM predicts real SOP actions for chunks that should be transitions/idle/non-SOP
- VLM confidently predicts a specific SOP action on chunks that are too short or partial to support that decision (e.g., small fragments produced by DDM over-segmentation, where only the beginning or ending of an action is visible — and the beginning or ending may look similar to multiple actions, leading to a confident-but-wrong prediction instead of a defensive non-SOP fallback)
- Can occur on any chunk size, but more common on very short or very long chunks

**Diagnostic Criteria:**
- `video_name_to_output_text.json`: Check chunks where VLM predicts SOP actions but no golden action exists at that time range
- By-action confusion: Check for non-SOP action being misclassified as specific SOP actions
- Cross-reference with DDM: If the hallucinated chunk is very short (from DDM FP boundary), the root cause is DDM over-segmentation triggering VLM hallucination. If the chunk is normal-sized, the root cause is pure VLM confusion.

**Root Causes:**
- VLM encounters ambiguous video content (transition between actions, partial action visibility)
- Very short chunks from DDM over-segmentation lack sufficient visual information
- Training data imbalance: model biased toward predicting certain SOP actions over the non-SOP action
- Very long chunks from DDM under-segmentation cause frame subsampling that confuses the VLM

**Common Fixes:**
1. If triggered by DDM over-segmentation: fix DDM boundaries first (Pattern 3)
2. If triggered by DDM under-segmentation: fix DDM boundaries first (Pattern 1)
3. Train VLM with varied chunk lengths including very short and very long ones
   - Increase `sequential_mcq.max_chunk_len` for handling very long ones (DDM under-segmentation)
4. Check training data balance between SOP actions and non-SOP action
5. If hallucination occurs on transition / ambiguous chunks, strengthen the "when visual content is unclear, answer non-SOP" signal directly on the VLM side. Two augmentation levers target this specifically:
   - Enable DS `num_hard_neg > 0` — produces coherent-but-mixed videos (frames sampled without shuffling) that teach the model to reject plausible-looking but incomplete video. DS defaults to `num_hard_neg = 0` — i.e., only fully shuffled-incoherent videos — which can leave the model under-trained on the harder ambiguous-but-coherent case.
   - Enable `extra_negative` pointing at a cross-SOP dataset — produces real coherent videos whose content matches none of the base SOP actions; answer is always non-SOP. Complements DS (synthetic frame-mixed nonsense) with real-but-wrong-domain video.

   Both fixes emit `augment-config-change`. Check their current state in the Augmentation Signal Audit (SKILL.md Step 6b') before recommending — if `num_hard_neg = 0` on DS or `extra_negative` is disabled, those are coverage gaps for this pattern.

---

## Pattern 5: Training Convergence Issues

Use the loss curve from `analyze_training_log.py` (sampled at every 5% of training) and the per-component learning rates from the training config to diagnose convergence issues.

**IMPORTANT — Check for model collapse first (Pattern 9).** Both overfitting and model collapse can produce "loss converges early then stays flat." Before diagnosing overfitting, examine the by-action prediction distribution. If the model predicts overwhelmingly 1-2 action classes for all inputs and by-action accuracy is near or below random chance (`1 / num_actions`), this is model collapse (Pattern 9), not overfitting. The fixes are different — overfitting calls for regularization/fewer epochs, while model collapse calls for lower learning rates and potentially more training.

### 5a. Overfitting

**Symptoms:**
- Training loss converges very early and remains flat for the majority of training
- Model performs well on training-similar videos but poorly on diverse validation
- Action confusion on actions with fewer training samples

**Diagnostic Criteria:**
- **First, rule out model collapse (Pattern 9):** Check the by-action prediction distribution. Overfitting means the model learned the training data but doesn't generalize — it should still show discriminative ability (different predictions for different inputs, moderate by-action accuracy). If the model has collapsed to predicting only 1-2 classes, see Pattern 9 instead.
- Training log: Examine the loss curve. Look for loss that stabilizes early and shows no meaningful decrease for a large portion of remaining training.
- Check LR schedule — constant LR with small dataset may lead to overfitting
- Cross-reference with by-action confusion: if specific actions have high error rates, overfitting may have caused the model to memorize training patterns rather than generalize

**Root Causes:**
- Too many epochs for the dataset size
- No learning rate decay schedule
- Full fine-tuning instead of LoRA for small datasets

**Common Fixes:**
1. Add LR decay: set `optm_decay_type` and `optm_decay_ratio`
2. Keep multiple checkpoints: increase `max_keep` to 3-5
3. Reduce epochs (only after confirming this is overfitting, not model collapse) — **and only when by-action accuracy is already saturated (≥ 0.97) across most classes.** A near-zero loss by itself is not proof of overfitting on small datasets; the augmented QA set contains heavy repetition and will bottom out on loss long before the model has actually generalized. Check `per-action` accuracy from `analyze_by_action_confusion.py` — if any action is below 0.90, prefer Pattern 5b (underfit / coverage gap) over Pattern 5a.
4. Consider LoRA fine-tuning for small datasets

**LoRA-specific fixes** (apply only when `[policy.lora]` is present in `train_config.toml`):
- Reduce `policy.lora.r` (e.g., 64 → 32 → 16). Lower rank means fewer trainable parameters and less capacity to memorise training quirks.
- Raise `policy.lora.lora_dropout` (e.g., 0.05 → 0.10). Acts directly on the LoRA path during training and is a cheap regularisation knob.
- Do NOT lower `policy.lora.lora_alpha` to fight overfitting. Alpha scales the magnitude of *all* learned deltas — correct and incorrect alike — so reducing it typically harms by-action accuracy more than it helps overfitting. Use the dropout and r levers instead.

### 5b. Underfitting

**Symptoms:**
- Training loss is still decreasing when training ends (hasn't converged)
- High error rates across many or all actions (not concentrated on a specific pair)
- Poor by-action accuracy AND poor end-to-end accuracy together
- Model frequently predicts the non-SOP action for actual SOP actions (underconfident)
- **By-action accuracy on the validation set is below 95%** — see diagnostic criterion below

**Diagnostic Criteria:**
- **By-action validation accuracy < 95%** is a strong primary signal for underfitting. The by-action evaluation runs the VLM on *golden* per-action chunks (perfectly-segmented), which is the easiest possible inference setting for the VLM. **Pursue 100% by-action accuracy** as the training target — in real E2E evaluation the chunks come from DDM segmentation which is never as clean as golden, so every percentage point of by-action accuracy below 100% will magnify into a larger E2E accuracy drop. Treat any by-action accuracy under ~0.95 as evidence that the VLM has not finished learning, even if the loss curve looks flat. Only declare a model "trained enough" once by-action accuracy is at or near 100%.
- Training log: Examine the loss curve. Look for loss that is still trending downward at the end of training (e.g., loss at 95%-100% is noticeably lower than at 85%-90%, indicating the model was still learning).
- Check total number of epochs — may be too few
- Check LR — may be too low for the model to learn effectively
- Cross-reference with by-action confusion: if errors are spread across many actions (not just one confused pair), the model may not have learned enough

**Root Causes:**
- Too few epochs for the dataset size
- Learning rate too low
- Training terminated prematurely (job failure, timeout)
- Dataset too large relative to training steps

**Common Fixes:**
1. Increase epochs — Empirically for SOP fine-tuning on small datasets **5–20 epochs is the useful range** (Not always true, decide number of epochs based on the information collected, but epoch < 5 usually is not sufficient for the model to be trained)
   - **Anti-pattern — avoid premature capping:** If a prior iteration raised `epoch` (e.g., 1 → 3) and the loss curve shows rapid convergence to near-zero in the first 30–40% of training, the reflexive conclusion is "overfit, do not raise epochs further." That conclusion is only correct when by-action accuracy also saturates near 100%. When by-action accuracy is still moderate (e.g., < 0.90) AND the loss is near-zero, the bottleneck is typically data *diversity*, not epoch count, and the next epoch bump may still help — do not emit a "do NOT raise beyond N" recommendation on the basis of loss-curve shape alone.
2. Increase learning rate (e.g., from 1e-6 to 5e-6)
3. Check training log for errors/crashes that may have caused early termination
4. If dataset is very large, increase `train_batch_per_replica` or `max_num_steps`
5. If steps-per-epoch × epochs is still only a few hundred, the LR schedule (warmup + decay) never reaches its intended plateau — either reduce `optm_warmup_steps` or raise epochs so the ratio `optm_warmup_steps / total_steps` lands in the 10–20% range rather than the 50–90% range.

**LoRA-specific fixes** (apply only when `[policy.lora]` is present in `train_config.toml`). Apply in priority order — never bundle alpha and r changes in the same iteration:

1. **PRIMARY: raise `policy.lora.lora_alpha`** to achieve `effective_scaling = lora_alpha / r ≥ 32`. The LoRA update applied at inference is `(alpha/r) · B·A`. When the model has converged on training data but by-action accuracy remains below target and no dominant confusion pair drives the errors, the binding constraint is typically the magnitude of the learned delta, not its representational capacity. Doubling alpha (e.g., 256 → 512, or 512 → 1024) is a single-parameter retrain.

2. **SECONDARY: raise `train.epoch`.** LoRA converges more slowly than full fine-tuning because only the LoRA path receives gradient. **The right epoch count is dataset-specific**. Observe the prior run's loss curve: if it was still descending at the end (loss at 95–100% noticeably lower than at 85–90%), extend training until the curve plateaus. As a soft rule of thumb, LoRA on small datasets typically requires more epochs than full fine-tuning to reach an equivalent training-set loss.

3. **LAST RESORT: raise `policy.lora.r`.** Increases trainable parameter count, memory, and merge time. Raising `r` alone (without raising alpha proportionally) reduces effective scaling and is unlikely to produce a monotonic by-action accuracy improvement. Attempt this only after steps 1 and 2 have been exhausted.

---

## Pattern 6: Insufficient Training Data

**Symptoms:**
- Low accuracy across all or most actions (not just specific confused pairs)
- Validation accuracy significantly lower than training accuracy
- Model frequently predicts non-SOP action for actual SOP actions

**Diagnostic Criteria:**
- Count annotated SOP cycle videos: should be >= 10-20
- Count total QA training samples: should be >= 2000
- Check action distribution: each action should have >= 50 samples
- Check if augmented data (DMCQ, BCQ, etc.) covers all actions

**Root Causes:**
- Too few annotated SOP cycle videos (e.g., only 3)
- Augmentation generates more QAs but doesn't add visual diversity
- Some actions may be underrepresented in training data

**Common Fixes:**
1. Annotate more SOP cycle videos (target 10-20 cycles minimum)
2. Ensure annotations cover different operators, speeds, and conditions
3. Increase QA augmentation for underrepresented actions

---

## Pattern 7: Data Generation Config Issues

**Symptoms:**
- Specific actions never appear as DMCQ or MCQ answers
- `non_sop_action` set to a real SOP action ID instead of "none" action
- DMCQ answer distribution heavily skewed (>50% non-SOP action)
- MCQ doesn't cover multi-action scenarios

**Diagnostic Criteria:**
- Analyze augment config YAML for consistency
- Check DMCQ answer distribution: should not be >50% for any single action
- Check `non_sop_action` matches across all config sections
- Check MCQ `max_chunk_len` value

**Root Causes:**
- Config copy-paste errors (e.g., `non_sop_action: 8` instead of `11`)
- DMCQ generation algorithm biases
- MCQ `max_chunk_len` too small for the expected evaluation chunk sizes

**Common Fixes:**
1. Fix `non_sop_action` to match the actual "none of the above" action ID
2. Ensure `non_sop_action` is consistent across all config sections
3. Increase MCQ `max_chunk_len` > 1
4. Review DMCQ negative sampling to ensure all actions appear as answers

---

## Pattern 8: Evaluation Parameter Mismatch

**Symptoms:**
- Good by-action accuracy but poor end-to-end accuracy
- Inference fps/max_frames differ from training values
- DDM `score_threshold` not optimized for the specific dataset

**Diagnostic Criteria:**
- Compare training config (fps, max_frames, total_pixels) with evaluation config
- Check if DDM `score_threshold` was tuned or using defaults (default 0.6)
- Check if `max_frames` is sufficient for the typical chunk durations

**Root Causes:**
- fps/max_frames mismatch between training and evaluation
- DDM `score_threshold` not calibrated for the specific camera/SOP
- `max_frames` too low for long chunks (causes severe frame subsampling)

**Common Fixes:**
1. Ensure fps, max_frames, total_pixels match between training and evaluation
2. Tune DDM `score_threshold` on a validation set (try 0.4, 0.5, 0.6)
3. Consider increasing max_frames if typical chunks exceed max_frames/fps seconds

---

## Pattern 9: Model Collapse / Catastrophic Forgetting

**Symptoms:**
- VLM predictions are dominated by 1-2 action classes across virtually all inputs
- By-action accuracy is near or below random chance (`1 / num_actions`)
- Training loss converges very early and stays flat (same loss curve shape as overfitting — must differentiate by checking prediction distribution)
- E2E predictions show extremely skewed action frequency (one action predicted far more than others)

**Diagnostic Criteria:**
- **Prediction distribution:** From by-action confusion analysis, check how many unique actions the model actually predicts. If 1-2 actions account for the vast majority of predictions, this is model collapse. This is the key differentiator from overfitting, where the model retains broader discriminative ability.
- **Per-component learning rates:** Extract `optm_lr` from the training config — this is a list `[llm_lr, vlm_lr, linear_lr]` controlling each model component. Assess whether any component LR is unusually aggressive. Consider the relationship between LR magnitude and warmup steps: high LR with very short warmup causes a gradient shock at the start of training that can destroy pretrained features before the model has a chance to adapt gradually.
- **Warmup adequacy:** Compare `optm_warmup_steps` against the LR magnitude. A very short warmup (e.g., 3 steps) combined with an aggressive LR is a strong indicator — the model is hit with a large learning rate almost immediately, with no gradual ramp-up.
- **Assess LR aggressiveness from the training run itself:** Look at how quickly the loss drops. If the loss drops by 80-90% within the first 10-15% of training steps, the LR may be too aggressive — the model is learning (or memorizing) too fast for the pretrained features to adapt gracefully. Also compute the warmup ratio (`optm_warmup_steps / total_steps`): if warmup covers only a tiny fraction of the first epoch, the model reaches full LR before it has seen enough diverse samples to stabilize.

**Root Causes:**
- Learning rate too high for one or more model components, destroying pretrained features
- Insufficient warmup — the model is hit with an aggressive LR before it can adapt gradually
- The combination of aggressive LR + short warmup + small dataset is particularly destructive

**Common Fixes:**
1. Reduce the per-component learning rates — compare against known-good configs for similar tasks to calibrate what values work
2. Increase warmup steps to allow gradual adaptation (known-good configs typically use ~20 warmup steps)
3. Increase training epochs — with a properly calibrated LR, the model needs more time to learn gradually (this is the opposite of the overfitting fix)
4. If no reference configs exist, try reducing the most aggressive component LR by 10-50x and increasing warmup proportionally

**Diagnostic Criteria (additional — non-SOP-specific collapse branch):**

Two different root causes can both produce model collapse, and they require different fixes. Before prescribing the LR/warmup fix, check which branch applies:

1. **LR-aggressive collapse:** the collapsed class is arbitrary (whichever class the model latched onto early; often a visually common class, not specifically non-SOP). LR/warmup fix applies.
2. **Non-SOP-specific collapse:** the collapsed class is specifically the non-SOP action. This is a **coverage / augmentation-weighting** root cause, not an LR root cause. Check training data balance:
   - DMCQ negative samples always answer non-SOP. If `num_neg` greatly exceeds `num_pos + num_hard_pos`, the model sees far more "answer non-SOP" examples than "answer specific action" examples.
   - DS answers are always non-SOP. Heavy DS with large `num_runs` + `num_hard_neg` adds to non-SOP weight.
   - EN answers are always non-SOP. Heavy `num_runs` in EN adds to non-SOP weight.
   - Aggregate check: if total DMCQ-negative + DS + EN sample counts exceed correct-action samples from BCQ positive + MCQ + Golden GQA + GQAs + DMCQ positive, non-SOP is over-weighted and the model's rational decision is to always predict non-SOP.
3. **LoRA over-scaling collapse (LoRA runs only):** when `[policy.lora]` is present and `effective_scaling = lora_alpha / r` is very high (e.g., > 128 or 256), gradient updates through the LoRA path can be amplified enough to destabilise pretrained features within the first epoch. The collapsed class can be arbitrary or skew toward non-SOP depending on augmentation balance. **Distinguishing signal:** collapse appears even when per-component LRs are conservative AND non-SOP augmentation is not over-weighted.

**Additional Fixes (non-SOP-specific collapse branch):**

5. Rebalance non-SOP answer share via `augment-config-change`:
   - Reduce `dynamic_mcq.num_neg`
   - Reduce `dynamic_shuffling.num_runs` or `dynamic_shuffling.num_hard_neg`
   - Reduce `extra_negative.num_runs`
   - Increase `dynamic_mcq.num_hard_pos` to push correct-action training signal up
6. If BCQ is disabled, enable it — BCQ positive samples produce "Yes, the operator is {action}" answers that ground the model in correct SOP actions.

**Additional Fixes (LoRA over-scaling collapse branch — LoRA runs only):**

7. Reduce `policy.lora.lora_alpha` so `effective_scaling = lora_alpha / r` lands in the 16–32 range. This restores numerical stability without reducing representational capacity (`r` unchanged). A halving of alpha is a reasonable first step.
8. Optionally raise `policy.lora.lora_dropout` modestly (0.05 → 0.10) to add direct regularisation on the LoRA path during the next training run.

**Key Distinction from Overfitting:**
- Overfitting: model learned the training data but doesn't generalize. Predictions are diverse but wrong on new data. Fix: regularize, reduce epochs.
- LR-aggressive model collapse: model's pretrained features were destroyed by gradient shock. Predictions collapse to 1-2 classes regardless of input; collapsed class is often arbitrary. Fix: lower LR, increase warmup, train longer.
- Non-SOP-specific model collapse: model is rationally predicting the majority class given a skewed training distribution; its features are intact but the reward signal pushed it to always predict non-SOP. Fix: rebalance augmentation (`augment-config-change`), not LR.
- LoRA over-scaling collapse: LoRA delta magnitude exceeded the model's stability budget, destabilising pretrained features through the alpha-scaled LoRA path specifically. Fix: reduce `policy.lora.lora_alpha`.

---

## Pattern 10: DDM Overfitting — High Val F1 but Poor E2E Sequence Accuracy

**Symptoms:**
- **Primary signature:** DDM val/F1 is much higher than E2E DDM F1 (gap > 0.2). Val/F1 alone looking "acceptable" is NOT the signature — it must diverge from E2E F1.
- Missing actions in E2E despite a seemingly well-trained DDM on the training split
- Problem worsens as the number of training videos is small (< 20)

**Diagnostic Criteria (must hold together):**
- `val/f1_score` from the training log vs `Temporal Segmentation F1` in `summary.txt` from E2E evaluation — **gap > 0.2** indicates poor generalization. A gap < 0.1 (e.g. val 0.95, E2E 0.97) rules this pattern out — DDM is fine, look at VLM (Pattern 4).
- By-action accuracy is high (VLM is fine in isolation) **AND** the val→E2E F1 gap is large — both conditions required. High by-action + high E2E DDM F1 + low sequence accuracy points at VLM, not this pattern.

**Root Causes:**
- DDM memorizes the specific lighting, encoding, and motion patterns of the small training set
- No image augmentation during training → model is brittle to any visual variation in the test videos
- Val split is too small (e.g. 1 video from a 10-video set) to reliably measure generalization

**Common Fixes — try in this order:**

1. **Try `RandomResize` augmentation first** (`action type: ddm-training-config-change`):
   - Randomly selects interpolation method (bilinear / bicubic / nearest) per training sample
   - Makes the model robust to encoding and resize artifacts in unseen test videos
   - Shown to reduce the val→test generalization gap without hurting convergence significantly
   - Config: `RandomResize.enabled: true`, `interpolation: [bilinear, bicubic, nearest]`, `antialias_prob: 0.5`
   - Run E2E after retraining — check if sequence accuracy improves

2. **Try `ColorJitter` augmentation if lighting/exposure varies** (`action type: ddm-training-config-change`):
   - Randomly shifts brightness, contrast, saturation, hue during training
   - Helps when test videos are captured under different lighting conditions than training
   - May improve raw DDM F1 but can occasionally reduce E2E accuracy if boundaries become less sharp
   - Keep values low: `brightness: 0.1, contrast: 0.1, saturation: 0.05, hue: 0.01`
   - Always run E2E (not just val/F1) to confirm improvement — ColorJitter and RandomResize should be tested **separately** first to isolate which augmentation helps

3. **Do NOT use `GaussianBlur`** for DDM:
   - DDM detects temporal boundaries by frame-to-frame differences; blur deliberately smears those differences
   - Consistently hurts DDM performance — avoid it

4. **Combining augmentations on small datasets:** each augmentation independently increases sample difficulty; combining them makes training too hard with too little data. Test one at a time, then combine only if both help independently.

*Key insight:* RandomResize improves E2E most despite potentially lower val F1 — val/F1 alone is not a reliable proxy for E2E quality when training data is small. ColorJitter can raise test DDM F1 but may worsen E2E (sharper boundaries but slightly misaligned, confusing VLM). All-3 together is too aggressive for small datasets.
