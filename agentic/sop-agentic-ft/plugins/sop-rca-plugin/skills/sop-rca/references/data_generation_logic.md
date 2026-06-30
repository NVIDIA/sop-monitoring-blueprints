# Data Generation Pipeline Logic

This document describes the QA data augmentation pipeline that generates
training data for VLM fine-tuning. The pipeline code is NOT provided during
RCA — this reference captures the key logic.

## Overview

The pipeline takes annotated SOP cycle videos and generates multiple types of
Question-Answer training data. Input is an `augment_config.yaml` that controls
what QA types to generate and their parameters.

## QA Types

### 1. Binary Choice QA (BCQ)
- **Format:** Yes/No question about whether a specific action is being performed in a video chunk. The question names one action; the model answers Yes or No.
- **Mechanism:** For each action chunk:
  1. The correct action is identified from the chunk's filename
  2. Generate `pos_cnt` positive samples: question asks about the correct action, answer is "Yes, the {subject} is {correct_action}."
  3. Generate `neg_cnt` negative samples: question asks about a randomly chosen wrong action, answer is "No, the {subject} is {correct_action}." — the "No" answer **still reveals the correct action**, teaching the model what is actually happening
  4. Counts are derived from `negative_ratio`: `pos_cnt = max(1, int(1 / negative_ratio))`, `neg_cnt = int(pos_cnt * negative_ratio)`
- **Config section:** `binary_choice_qa`
- **Key params:**
  - `negative_ratio`: Ratio of "No" to "Yes" answers (default: 2.0, meaning 2 "No" per 1 "Yes")
  - `subject`: Who performs the action (default: "operator")
  - `exclude_action`: Actions to skip during generation
- **Example:**
  - Positive: Q: "Is the operator installing the cable to the server closer to the operator?" A: "Yes, the operator is installing the cable to the server closer to the operator."
  - Negative: Q: "Is the operator installing the paddle board to far server, LEFT side?" A: "No, the operator is installing the cable to the server closer to the operator."
- **RCA relevance:**
  - High `negative_ratio` means model sees more "No" answers, potentially making it conservative about confirming actions
  - Since "No" answers reveal the correct action, BCQ teaches the model both what an action looks like AND what it doesn't look like

### 2. Sequential MCQ (MCQ) — **THIS IS THE EVALUATION FORMAT**
- **Format:** Standard MCQ with all action options listed. Option numbers match the original SOP action IDs (1, 2, 3, ..., N). This is **exactly the same format** used during VLM inference/evaluation — the model receives a video chunk and the full list of possible actions, and must output which action(s) the video contains.
- **Mechanism:** For each annotated SOP cycle video:
  1. **Single-action chunks:** Each individual action chunk is used as-is. Answer is the single action choice (e.g., `(5) installing the paddle board...`).
  2. **Multi-action chunks:** A sliding window of length 2 to `max_chunk_len` moves over consecutive action chunks (sorted by timeline). For each window, the video clips are concatenated using moviepy into a single combined chunk. Answer is the space-joined action choices in order (e.g., `(5) action5 (6) action6`).
  3. Both single-action and multi-action chunks are always generated.
- **Config section:** `sequential_mcq`
- **Key params:**
  - `max_chunk_len`: Maximum number of consecutive actions combined in one video chunk (default: 4 in code, commonly set to 2 in configs)
  - `exclude_action`: Actions to skip during multi-action chunk generation (excluded chunks are still included as single-action samples)
- **Example:**
  - Single-action: Q: "What step is the operator doing? (1)action1 (2)action2 ... (N)none" A: "(5) installing the paddle board..."
  - Multi-action: Q: same as above, A: "(5) installing the paddle board... (6) installing the paddle board..."
- **RCA relevance:** MCQ is the **highest-priority QA type** for RCA because:
  1. It matches the evaluation format exactly (same prompt structure, same option numbering)
  2. Per-action MCQ sample counts directly predict evaluation performance for that action
  3. `max_chunk_len` determines how many actions the model can handle per chunk — if E2E DDM creates chunks with 3-4 actions but training only covers max_chunk_len=2, the model will struggle
  4. When diagnosing action confusion (e.g., 7 vs 8), check MCQ samples first — this is where the model learned to distinguish them in the exact evaluation context

### 3. Dynamic MCQ (DMCQ)
- **Format:** MCQ with dynamically generated option lists (only a subset of actions shown as options). **Important:** Unlike MCQ, option numbers are dynamically assigned (1, 2, 3, ...) and do NOT correspond to the original SOP action IDs. This means DMCQ trains the model to understand action descriptions, but with a different option numbering scheme than what it sees during evaluation.
- **Mechanism:** For each action chunk, generates 4 types of QA samples:
  1. **Positive** (`num_pos` per chunk): Options include the correct action + random others. Answer = correct action. Teaches the model to identify the action.
  2. **Hard positive** (`num_hard_pos` per chunk, for each `hard_pos_mode`): Same as positive, but options include confusing/adjacent actions as distractors. Forces the model to discriminate between similar actions.
  3. **Negative** (`num_neg` per chunk): Options do NOT include the correct action. Answer = non-SOP action. Teaches the model to say "none of the above" when the correct answer isn't listed.
  4. **Hard negative** (`num_hard_neg` per chunk, for each `hard_neg_mode`): Same as negative, but options include confusing/adjacent actions. Harder because the options look plausible.
- **Hard modes:** Two modes are supported (can be combined, comma-separated):
  - `adjacent`: Uses actions with adjacent index numbers (e.g., for action 5, distractors are actions 4 and 6)
  - `confusion`: Uses `confusion_map` to select specific confused action pairs (e.g., `{7: [8], 8: [7]}`)
- **Option ordering:** Options are shuffled randomly, but the non-SOP action is always placed last.
- **Config section:** `dynamic_mcq`
- **Key params:**
  - `num_pos`: Number of positive samples per chunk (default: 2)
  - `num_neg`: Number of negative samples per chunk (default: 2)
  - `num_hard_pos`: Number of hard positive samples per chunk (default: 0)
  - `num_hard_neg`: Number of hard negative samples per chunk (default: 0)
  - `hard_pos_mode`: Hard positive mode — `"adjacent"`, `"confusion"`, or both comma-separated (default: "adjacent")
  - `hard_neg_mode`: Hard negative mode — `"adjacent"`, `"confusion"`, or both comma-separated (default: "adjacent")
  - `min_options` / `max_options`: Range of total options per question (default: 4 / 6)
  - `non_sop_action`: Action ID for non-SOP action (MUST match actual non-SOP action ID)
  - `confusion_map`: JSON string mapping action IDs to their commonly confused counterparts (required when mode includes "confusion")
  - `exclude_action`: Actions to exclude from DMCQ generation
- **Example:**
  - Positive: Q: "Which step? (1)cable closer (2)paddle left (3)I/O middle (4)none" A: "(2) paddle left"
  - Negative: Q: "Which step? (1)cable closer (2)I/O middle (3)paddle right (4)none" A: "(4) none"
- **RCA relevance:**
  - Options are dynamically numbered (1, 2, 3...), NOT SOP action IDs
  - `non_sop_action` misconfiguration is a common bug (e.g., set to a real SOP action ID instead of the actual non-SOP action ID)
  - `confusion_map` is critical for addressing action pair confusion — without it, hard negatives only use adjacent actions which may not be the confused pair
  - `num_hard_pos = 0` AND `num_hard_neg = 0` (defaults) means no confusion-aware training at all
  - Non-SOP action often dominates DMCQ answers (can be >50%) because negative samples always answer non-SOP

### 4. Dynamic Shuffling (DS)
- **Format:** MCQ with the same prompt as Sequential MCQ, but the video is synthetically constructed by mixing frames from multiple action chunks. The answer is **always the non-SOP action**, because the mixed video does not show any coherent SOP action.
- **Mechanism:** For each action chunk:
  1. Sample `num_distractor` distractor chunks from other actions (random between `min_distractor` and `max_distractor`)
  2. Sample frames from the target chunk and all distractor chunks
  3. Shuffle all sampled frames randomly and assemble into a new video
  4. The resulting video is visually incoherent — no meaningful SOP action is recognizable
  5. The answer is always the non-SOP action
- **Hard negatives:** A separate path generates harder training examples:
  1. Same frame-mixing from target + distractor chunks
  2. Only a portion of frames are sampled (controlled by `hard_neg_frames_ratio`)
  3. Frames are sampled using one of 3 modes: `front` (first N frames), `end` (last N frames), or `random`
  4. **Frames are NOT shuffled** — the video is more visually coherent, making it harder for the model
  5. Teaches the model that even somewhat coherent mixed frames are still non-SOP
- **Config section:** `dynamic_shuffling`
- **Key params:**
  - `non_sop_action`: Action ID for "none of the above" (MUST match actual none action ID)
  - `num_runs`: Number of shuffled variants to generate per chunk
  - `min_distractor` / `max_distractor`: Range of distractor chunks to mix in per sample
  - `num_hard_neg`: Number of hard negative samples per chunk (default: 0)
  - `hard_neg_frames_ratio`: Ratio of frames to sample for hard negatives (default: 0.1)
- **RCA relevance:**
  - `non_sop_action` misconfiguration causes the model to learn wrong "none" answers
  - `num_hard_neg = 0` (default) means no hard negatives — model may struggle with ambiguous/transition chunks during inference and hallucinate real SOP actions instead of predicting non-SOP
  - Low `min/max_distractor` reduces diversity of mixed videos
  - `hard_neg_frames_ratio` too high makes hard negatives too easy (too many mixed frames = too noisy to be confused with real actions)

### 5. General QA (GQA / Golden GQA)
- **Format:** Open-ended QA about the action in a video chunk. The question asks what the operator is doing; the answer describes the action in natural language (no option numbers).
- **Golden GQA mechanism:** For each action chunk:
  1. A golden QA template is created from `actions.json` — the question uses a fixed template, the answer uses the exact action description
  2. One QA is generated per chunk — the answer is the verbatim action description
  3. No LLM involved — purely template-based
- **GQA mechanism:** For each action chunk:
  1. A sample QA template is created per action (same as Golden GQA)
  2. The template is sent to an LLM (via OpenAI-compatible API) with few-shot examples to generate `num_qa_llm` paraphrased QA pairs per action
  3. From the LLM-generated pool, `num_qa_per_chunk` QAs are sampled (with or without replacement) for each chunk
  4. Supports both cloud LLM (Nvidia API) and local LLM endpoints
- **Config sections:** `general_qa`, `golden_general_qa`
- **Key params (GQA):**
  - `llm`: LLM model name for paraphrase generation (default: "meta/llama-3.1-70b-instruct")
  - `num_qa_llm`: Number of QA pairs to generate via LLM per action (default: 5)
  - `num_qa_per_chunk`: Number of QA pairs to sample per video chunk
  - `subject`: Who performs the action (default: "operator")
  - `exclude_action`: Actions to skip
- **RCA relevance:** Less directly relevant to MCQ-style evaluation since the format differs, but helps the model build general action understanding. GQA quality depends on the LLM — poor paraphrases could confuse the model.

### 6. Extra Negative (EN)
- **Format:** Dynamic MCQ-style questions using video chunks from a **different SOP**. The options are from the base SOP's action list (not the source SOP's). Since the video shows a different SOP entirely, none of the base SOP actions apply — the answer is **always the non-SOP action**.
- **Mechanism:** For each chunk from the external SOP:
  1. Generate `num_runs` MCQ questions with a random subset of the base SOP's actions as options (between `min_options` and `max_options`), plus the non-SOP action
  2. Answer is always the non-SOP action (because the video is from a different SOP)
  3. If `generate_all_options` is enabled: also generates one additional MCQ with ALL base SOP actions listed (golden question format), answer still non-SOP
  4. Option numbering is dynamic (1, 2, 3...) like DMCQ, not SOP action IDs
  5. Non-SOP action is always placed last in options
- **Config section:** `extra_negative`
- **Key params:**
  - `source_sop`: Path to video chunks from another SOP (configured in augment config YAML, passed as `--video-root` to the script)
  - `non_sop_action`: Non-SOP action ID of the base SOP (MUST match actual non-SOP action ID)
  - `num_runs`: Number of random-option MCQ variants per chunk (default: 2)
  - `min_options` / `max_options`: Range of total options per question (default: 4 / 6)
  - `generate_all_options`: If enabled, also generates one MCQ with all actions listed
  - `exclude_action`: Actions from the base SOP to exclude from option lists
- **RCA relevance:**
  - Teaches the model to say "none of the above" when viewing unfamiliar video content — complements DS (which uses frame-mixed nonsensical videos) with real coherent videos from a different SOP
  - `non_sop_action` misconfiguration here has the same impact as in DMCQ/DS
  - If the model hallucinates real SOP actions for transition/idle chunks, insufficient EN training data could be a contributing factor

## Common Config Issues

### non_sop_action Mismatch
All config sections that have `non_sop_action` MUST use the same value, and it MUST be the
non-SOP action ID (the catch-all action for idle/transition periods — identified in Step 1 of the analysis procedure).
Setting it to a real SOP action ID causes that action to be incorrectly treated as non-SOP.

### DMCQ Non-SOP Action Imbalance
Because many video chunks in training contain idle/transition periods, DMCQ
answers tend to be heavily skewed toward the non-SOP action (often 50-60%). This can make
the model conservative about predicting actual SOP actions.

### MCQ max_chunk_len
If DDM under-segments during end-to-end evaluation, the VLM may encounter chunks with
3-4 actions. If MCQ training only covers max_chunk_len=2, the model has never seen 3+ actions
in a single chunk during training and may struggle.

### Missing confusion_map
When by-action evaluation reveals a dominant confusion pair (e.g., action A frequently mistaken for action B),
adding `hard_neg_mode: "confusion"` with the appropriate `confusion_map` to DMCQ config is
the most effective targeted fix.
