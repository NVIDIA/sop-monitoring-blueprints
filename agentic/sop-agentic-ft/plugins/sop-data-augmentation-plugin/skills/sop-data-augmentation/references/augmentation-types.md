# Augmentation Types Reference

There are 7 augmentation types, executed sequentially in the order listed below. Each can be independently enabled/disabled in the config.

## 1. BCQ (Binary Choice QA)

**Purpose:** Generates yes/no questions that ask whether the operator is performing a specific action. Teaches the VLM to confirm or deny action presence.

**Default:** Enabled

**Example QA:**
- Positive: "Is the operator installing the first fan?" → "Yes, the operator is installing the first fan."
- Negative: "Is the operator installing the server cover?" → "No, the operator is installing the first fan."

**Config parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `true` | Enable this stage |
| `negative_ratio` | float | `2.0` | Ratio of negative to positive samples (2.0 = 1 yes + 2 no per video) |
| `subject` | string | `"operator"` | Who performs the action (used in question templates) |
| `exclude_action` | string | `""` | Action indices to exclude, underscore-separated (e.g., `"1_2"` excludes actions 1 and 2) |

**When to use:** Always. BCQ is the foundational augmentation type — it provides basic action recognition training.

---

## 2. Sequential MCQ (Multiple Choice QA)

**Purpose:** Creates multiple-choice questions from consecutive action sequences. Merges adjacent video chunks into multi-action clips and asks "what steps is the operator doing?" Teaches the VLM to recognize ordered action sequences.

**Default:** Enabled

**Example QA:**
- "There are 11 possible steps. What step is the operator doing? (1) standing by... (2) installing the first fan..."
- Answer: "(2) installing the first fan by connecting the connector and then pressing the fan in place (3) installing the second fan..."

**Config parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `true` | Enable this stage |
| `max_chunk_len` | int | `2` | Max actions per chunk. `2` generates chunks of 1-action and 2-action sequences; `3` adds 3-action sequences, etc. |
| `exclude_action` | string | `""` | Action indices to exclude (e.g., `"1_2"`) |

**When to use:** Always. Essential for teaching the VLM to understand multi-step sequences.

---

## 3. Golden GQA (Grounded Question-Answer)

**Purpose:** Uses pre-written (golden) question-answer pairs per action as direct training data. Each action gets one canonical Q&A pair from template files. Provides high-quality, human-verified training examples.

**Default:** Enabled

**Config parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `true` | Enable this stage |

**When to use:** Always. Golden QA pairs are the highest-quality training signal.

**Note:** Golden QA templates are auto-generated from `actions.json` if not manually provided. Manual golden QA files go in `assets/data/<dataset_id>/golden_gqa_to_gqas/action<N>.txt`.

---

## 4. GQAs (LLM-Expanded GQA)

**Purpose:** Uses an LLM to generate multiple question-answer variations from each golden QA pair. Dramatically increases QA diversity per action. This is the only stage that calls an external LLM.

**Default:** Enabled

**Example:** From one golden pair "What is the operator holding?" / "A black HMC", the LLM generates 8 variations like "What object does the worker have in their hands?" / "The worker is holding a dark-colored HMC device."

**Config parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `true` | Enable this stage |
| `llm_type` | string | `"nvidia"` | LLM backend: `"nvidia"` (NIM API) or `"local"` (self-hosted vllm) |
| `local_llm_url` | string | `""` | Local LLM endpoint URL (e.g., `"http://10.18.44.75:9000/v1"`). Required if `llm_type` is `"local"` |
| `llm` | string | `"meta/llama-3.1-70b-instruct"` | Model name. For NVIDIA NIM: use NIM model ID. For local: use model name served by vllm |
| `num_qa_llm` | int | `8` | Number of QA pairs the LLM generates per action |
| `num_qa_per_chunk` | int | `2` | Number of QA pairs to sample from LLM output per video chunk |
| `exclude_action` | string | `""` | Action indices to exclude (e.g., `"1_2"`) |
| `enable_thinking` | string | `""` | For thinking-capable models (e.g., Qwen3 / Qwen3.5): `"true"` enables thinking, `"false"` disables it. Empty string = auto-detect with fallback |

> The NGC API key is read only from the `NGC_API_KEY` environment variable (the deployment `.env`); it is not configurable here.

**When to use:** Always recommended. Provides the most diverse training data. If NVIDIA NIM rate limits are an issue, switch to a local LLM.

**LLM configuration guide:**
- **NVIDIA NIM API (default):** Set `llm_type: "nvidia"`, ensure `NGC_API_KEY` is in `.env`
- **Local vllm server:** Set `llm_type: "local"`, set `local_llm_url` to the vllm endpoint (must include `/v1` path)
- **Thinking-mode models (Qwen3, Qwen3.5, etc.):** Set `enable_thinking: "false"` to get direct content. If left empty, the system auto-retries with thinking disabled when the model returns empty content

---

## 5. Dynamic MCQ (Hard Negative Mining)

**Purpose:** Generates multiple-choice questions with carefully constructed positive and negative samples, including hard negatives from adjacent or easily confused actions. Forces the VLM to make fine-grained distinctions.

**Default:** Disabled

**Sample types generated:**
- **Positive (pos):** Correct action is among the choices, options are randomly sampled from all actions
- **Negative (neg):** Correct action is NOT among the choices (answer = non-SOP action), options randomly sampled
- **Hard Positive (hp):** Correct action is the answer, but the options deliberately include actions that are easy to confuse with the correct one — making the question harder because the VLM must pick the right action from very similar alternatives
- **Hard Negative (hn):** Correct action is NOT the answer (answer = non-SOP action), but the options include actions similar to what the video actually shows — making the question harder because the VLM must still reject the video even when plausible-looking actions are listed

**Hard modes explained:**
- **`"adjacent"`:** Includes actions that are sequentially adjacent in the action list (action N-1 and N+1). Useful when nearby steps in the SOP look similar (e.g., "install fan 1" is next to "install fan 2").
- **`"confusion"`:** Includes actions from a user-provided `confusion_map` that maps each action to its most commonly confused counterparts. Typically built from evaluation results where you identify which actions the VLM confuses most often.
- **`"adjacent,confusion"`:** Combines both — includes adjacent actions AND confusion-mapped actions in the same question's options.

**Config parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `false` | Enable this stage |
| `exclude_action` | string | `""` | Action indices to exclude |
| `non_sop_action` | int | **REQUIRED** | Action index of "none of the above" action (see note below) |
| `min_options` | int | `3` | Minimum number of answer options |
| `max_options` | int | `6` | Maximum number of answer options |
| `num_pos` | int | `1` | Positive samples per video |
| `num_neg` | int | `2` | Negative samples per video |
| `num_hard_pos` | int | `0` | Hard positive samples per video. Requires `hard_pos_mode` to be set |
| `num_hard_neg` | int | `0` | Hard negative samples per video. Requires `hard_neg_mode` to be set |
| `hard_neg_mode` | string | `""` | Hard negative modes: `"adjacent"`, `"confusion"`, or `"adjacent,confusion"` |
| `hard_pos_mode` | string | `""` | Hard positive modes: `"adjacent"`, `"confusion"`, or `"adjacent,confusion"` |
| `confusion_map` | string | `""` | JSON dict mapping action indices (1-based) to confusable actions. Format: `"{2: [1, 3], 4: [3, 5]}"` means action 2 is confused with 1 and 3 |

**When to use:** Enable when the VLM struggles to distinguish between similar actions (e.g., "installing fan 1" vs "installing fan 2"). Start with basic mode (`num_pos: 1, num_neg: 2`) without hard samples. Once you have evaluation results showing which actions confuse the model, build a `confusion_map` and enable hard modes to specifically train against those weaknesses.

**Recommended starting config:** `num_pos: 1, num_neg: 2, num_hard_pos: 0, num_hard_neg: 0`

**Advanced config with hard modes:**
```yaml
num_hard_pos: 1
num_hard_neg: 1
hard_pos_mode: "adjacent"          # or "confusion" or "adjacent,confusion"
hard_neg_mode: "adjacent"
confusion_map: "{2: [1, 3], 5: [4, 6]}"   # optional, needed for "confusion" mode
```

---

## 6. Dynamic Shuffling (DSQA)

**Purpose:** Creates noise videos by combining frames from multiple different action chunks, then asks the VLM to identify the action. The correct answer is always "non-SOP action" because the shuffled video doesn't represent any real action. Teaches the VLM to reject incoherent video sequences.

**Default:** Disabled

**Normal vs hard-negative shuffled videos:**
- **Normal (`num_runs`):** Randomly samples frames from the source video and several distractor videos, then **fully shuffles** all frames into random order. The result is visually chaotic and relatively easy for the VLM to reject as "not a real action."
- **Hard negative (`num_hard_neg`):** Samples frames from a constrained temporal region (front, end, or random subset controlled by `hard_neg_frames_ratio`) and crucially does **NOT shuffle** the frame order — preserving temporal coherence. This produces a more deceptive video that looks plausible at first glance but actually combines content from multiple sources, forcing the VLM to develop deeper understanding rather than relying on visual chaos as a rejection signal.

**Config parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `false` | Enable this stage |
| `exclude_action` | string | `""` | Action indices to exclude |
| `non_sop_action` | int | **REQUIRED** | Action index of "none of the above" action |
| `min_distractor` | int | `3` | Minimum distractor videos to sample frames from |
| `max_distractor` | int | `6` | Maximum distractor videos to sample frames from |
| `num_runs` | int | `1` | Normal shuffled videos per chunk (fully randomized frame order) |
| `num_hard_neg` | int | `0` | Hard negative videos per chunk (temporally coherent, more deceptive) |
| `hard_neg_frames_ratio` | float | `0.1` | Controls frame pool size for hard negatives. E.g., 0.1 means only the first/last 10% of frames (or a random 10%) are sampled, making the temporal region narrow and the resulting video more focused |

**When to use:** Enable when the VLM tends to make false positive identifications — seeing actions that aren't there. Start with `num_runs: 1` or `2` for basic shuffling. Add `num_hard_neg: 1` once the model handles basic shuffling well but still makes false positives on temporally coherent distractors.

---

## 7. Extra Negative (ENQA)

**Purpose:** Uses videos from a completely different SOP dataset as negative examples. The model must recognize these videos as "not part of this SOP." Teaches cross-domain negative recognition.

**Default:** Disabled

**Config parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `false` | Enable this stage |
| `exclude_action` | string | `""` | Actions from the extra negative source dataset to exclude |
| `extra_negative_data_id` | string | **REQUIRED** | Dataset ID of the other annotated dataset to use as negatives |
| `non_sop_action` | int | **REQUIRED** | Action index of "none of the above" action in the base dataset |
| `min_options` | int | `3` | Minimum answer options |
| `max_options` | int | `6` | Maximum answer options |
| `num_runs` | int | `1` | Number of negative samples per external video |
| `generate_all_options` | bool | `true` | Also generate a gold-standard sample with all action options |

**When to use:** Enable when you have multiple annotated SOP datasets and want to prevent cross-SOP confusion. The `extra_negative_data_id` must be a separate annotated dataset that has already been processed through the annotation pipeline.
