# End-to-End Evaluation Pipeline Logic

This document describes the logic of the SOP monitoring end-to-end evaluation
pipeline. This code is NOT provided during RCA — this reference captures the
key algorithms and data flows.

## Pipeline Overview

```
Full SOP Video
    |
    v
[Temporal Segmentation (DDMNet)] --> boundaries, scores per video
    |
    v
[Video Chunking] --> split video into action chunks at DDM boundaries
    |
    v
[VLM Action Recognition (Cosmos Reason)] --> predicted action(s) per chunk
    |
    v
[Action Sequence Assembly] --> ordered list of predicted actions per video
    |
    v
[Evaluation] --> edit distance, accuracy metrics
```

## 1. Temporal Segmentation

**Input:** Full SOP video files, DDMNet model checkpoint  
**Output:** `video_to_ddm_info_debug.json`, `video_to_boundaries_debug.json`, `f1_*.json`, boundary visualization PNGs

**Algorithm:**
1. Run DDMNet on each video to get per-frame boundary probability scores
2. Apply score threshold filtering (configurable, default ~0.6)
3. Apply Non-Maximum Suppression (NMS) with `nms_sec` parameter (if `nms_sec=0`, would use default: 2.5% of video duration)
4. Select peaks as predicted boundaries
5. Compute golden boundaries as midpoints between annotated action segment endpoints
6. Calculate TP, FP, FN, Precision, Recall, F1 per video
7. Generate visualization PNG overlaying DDM scores, predicted boundaries (green), and golden boundaries (red)

**Key Parameters (boundary filtering):**
- `score_threshold` (default 0.6): Minimum DDM score for a frame to be considered a boundary candidate. This is the parameter that controls how many boundaries are detected. Higher = fewer boundaries (higher precision, lower recall). **This is the parameter to tune when DDM under-segments or over-segments.**
- `nms_sec` (default 0.0, which triggers per-video adaptive NMS of 0.025 * video_duration): NMS half-window in seconds. Suppresses nearby duplicate peaks within this window. When `nms_sec=0.0` in the logs, NMS is NOT disabled — it means the default percentage-based window is used.

**Key Parameters (F1 metric calculation only — NOT used for filtering):**
- `ddm_threshold` (stored in `f1_*.json` per video): Computed as `video_duration * 0.025`. This is the tolerance window for matching predicted boundaries to golden boundaries when calculating TP/FP/FN. A predicted boundary within `ddm_threshold` seconds of a golden boundary counts as a True Positive. **This does NOT affect which boundaries are detected — it only affects how F1/precision/recall are computed.**

## 2. VLM Action Recognition

**Input:** Chunked video segments (from DDM boundaries), VLM model, MCQ prompt  
**Output:** `video_name_to_output_text.json`, `action_recognition_multi_gpu.log`

**Algorithm:**
1. For each video, get DDM-predicted boundaries
2. Extract video chunks between consecutive boundaries
3. For each chunk, construct MCQ prompt with all possible SOP actions
4. Run VLM inference (vLLM or transformers backend) with the video chunk + MCQ prompt
5. Parse VLM response to extract predicted action(s) using regex: `\((\d+)\)`
6. Store response text keyed by `[start_s-end_s]` time range

**Key Parameters:**
- `fps`: Frame sampling rate (must match training)
- `max_frames`: Maximum frames to sample per chunk (must match training)
- `total_pixels`, `max_pixels`, `min_pixels`: Vision resolution constraints
- `temperature`: 0.0 for deterministic inference
- `top_p`: 0.2 typically

**VLM Prompt Format:**
```
There are N possible steps for the SOP (Standard Operation Procedure) of the given video.
What step is the operator doing?
(1) action description 1
(2) action description 2
...
(N) doing action not belong to the SOP
```

## 3. Evaluation

**Input:** `video_name_to_output_text.json`, golden annotations (`<dataset>_anno.json`), `actions.json`
**Output:** `accuracy.json`

**Algorithm:**
1. For each video, extract predicted action sequence:
   a. Parse VLM output text for each time range
   b. Extract action numbers using regex `\((\d+)\)`
   c. Filter out `non_sop_action` (the non-SOP action identified in Step 1)
   d. Remove consecutive duplicates (`remove_continuous_rep`)
   e. Filter out `actions_can_be_skipped` (if defined in actions.json)

2. For each video, extract golden action sequence:
   a. Load annotations for the video
   b. Filter out non_sop_action and actions_can_be_skipped
   c. Remove consecutive duplicates

3. Compute edit distance between golden and predicted sequences

4. Classify errors:
   - **Wrong**: An action at position i doesn't match (substitution)
   - **Duplicate**: An extra action appears that is already in the predicted sequence
   - **Missing**: A golden action is absent from the predicted sequence

5. Aggregate metrics:
   - `seq_accuracy`: % of videos with edit_distance == 0
   - `accuracy`: 1 - (total_errors / total_actions)

**Key Behaviors:**
- `actions_can_be_skipped`: List of action IDs to ignore during evaluation. These actions are removed from both golden and predicted sequences before comparison.
- Multi-action VLM outputs: If VLM predicts "(5)(6)" for one chunk, both actions 5 and 6 are added to the sequence.
- `remove_continuous_rep`: If the same action appears consecutively (e.g., [5, 5, 6]), it's deduplicated to [5, 6].

## 4. By-Action Chunk Evaluation

**Input:** Pre-segmented action chunk videos (one action per chunk), VLM model  
**Output:** Per-chunk predictions, accuracy, confusion matrix

**Algorithm:**
1. Load pre-segmented video chunks where each chunk corresponds to exactly one annotated action
2. For each chunk, run VLM with same MCQ prompt as end-to-end evaluation
3. Extract first predicted action number from VLM response
4. Compare with ground truth action (from filename prefix: `NN_video_name.mp4` where NN = action ID)
5. Compute accuracy and confusion matrix

**Difference from E2E:** This evaluation isolates VLM accuracy from DDM segmentation quality. Chunks are perfectly segmented from annotations, not from DDM.

**Prediction Verification (`verify_pred`):**
Compares VLM prediction against ground truth label. Returns True if ANY of these match:
1. Exact string match: `pred == gt`
2. Numeric class match: Extracts numbers from `(\d+)` in both pred and gt; compares lists
3. Answer tag match: Extracts text from `<answer>...</answer>` tags, converts letters (A=1, B=2...) to numbers
4. Text match (case-insensitive): After stripping `(\d+)\s*` prefix, compares lowercase text

**Video Processing:**
- `total_pixels`: Hardcoded to 16572416 in the message payload
- `max_frames`: Hardcoded to 40
- Default `fps`: 8, `temperature`: 0.0 (greedy), `max_new_tokens`: 1024
- System prompt: "Answer the questions."

**LoRA Support:**
- `--model-base`: When set, `--model-path` is treated as LoRA adapter path
- For transformers: loads base model, merges LoRA in-place (single GPU due to Qwen3-VL bug)
- For vLLM: pre-merges LoRA to disk (`--merged-model-dir`), then loads merged model

**Output Format:**
- JSON: `{video_name: [[gt_action, pred_text, chunk_path], ...]}` per video
- Log: `{output_name}_log.txt` with per-chunk VLM responses and timing
