# Augmentation Types Reference

There are 8 augmentation types, executed sequentially in the order listed below. Each can be independently enabled/disabled in the config.

Types 1-7 consume the **pre-cut action chunks** produced by the video split, so every training clip is an exactly-trimmed action segment. Type 8 (WMCQ) is the exception: it goes back to the **source video** and cuts its own fixed-length windows.

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
| `llm` | string | `"meta/llama-3.3-70b-instruct"` | Model name. For NVIDIA NIM: use NIM model ID. For local: use model name served by vllm |
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

---

## 8. WMCQ (Window-Matched MCQ)

**Purpose:** Cuts training clips as **real fixed-length windows taken straight from the source video**, positioned so a key-step falls at varying offsets inside them, with the genuine surrounding footage as padding. Negatives are windows of the same length taken from regions containing no key-step at all — `neg_ratio` decides *how many*, and their positions are a random draw from every window that clears every key-step by `neg_margin`.

**Default:** Disabled

**When to use:** When **DDM temporal chunking performs badly and inference falls back to uniform chunking.**

This is the specific problem WMCQ solves. Every other augmentation type trains the VLM on exactly-trimmed action chunks — clean segments that start when the action starts and end when it ends. That is the right shape only when inference *also* sees exactly-trimmed segments, i.e. when DDM chunking is accurate. When it is not, and the pipeline falls back to uniform chunking, inference feeds the VLM fixed-length sliding windows instead: a brief key-step buried in surrounding non-SOP footage. The model has never been trained on anything that looks like that, so it is being asked a question in a format it has never seen.

WMCQ removes that mismatch by making the training clips have the same geometry as the windows the evaluation actually produces.

Do **not** reach for it first. If DDM chunking is good, the chunk-consuming types already match inference and WMCQ only adds redundant data. Diagnose the chunking first (see the RCA skill's DDM boundary analysis); WMCQ is the response to a *confirmed* chunking problem, not a general-purpose booster.

### What WMCQ assumes about your data

WMCQ was built for **long videos in which brief key-steps are separated by long stretches of non-SOP activity** — an operator performs a few seconds of procedure, then minutes of something else.

That is not a coincidence: it is the same shape that makes DDM fail, which is why WMCQ exists at all. DDM learns boundaries from transition movement, and this shape gives it few transitions, buried in continuous operator motion that looks much the same on either side. Poor chunking follows, uniform chunking is the fallback, and the train/eval geometry mismatch WMCQ fixes appears. On a normally-paced dataset — actions following one another, transitions being the main thing on screen — DDM should learn those transitions, provided the action definitions correspond to a visible change and the annotation boundaries are placed consistently. A DDM failing *there* is a signal about the action list or the annotations, not a reason to reach for WMCQ.

Two properties of the sparse shape are load-bearing:

**1. Most of the video is non-SOP.** Negatives are windows that clear *every* key-step by `neg_margin`. The denser the key-steps, the fewer such positions exist. Once the supply runs out you get fewer negatives than `neg_ratio` asked for — the stage warns, but it still succeeds, and a WMCQ set with too few negatives pushes the model toward firing an action on almost every window. On a densely annotated video the candidate pool can fall short by 3-4x.

**2. Key-steps are further apart than the window's slack.** A positive window must *contain* its key-step, so it can slide by `window - keystep_length` — call that the slack. If a neighbouring key-step is closer than the slack, the window swallows it, and the clip is then labelled with one action while showing two. With a 3s window: 2s actions separated by 1s gaps are fine (slack 1s, gap 1s), but 1s actions separated by 1s gaps put a second action inside **more than 90%** of the positive windows. Short key-steps are the exposed case — the shorter the action, the more room the window has to wander into its neighbour.

The stage counts these and warns at the end of the run, and each affected sample carries `overlaps_other_keystep: true` in its `meta` block so the set can be audited afterwards. A large fraction means the dataset is densely annotated and this is the wrong stage for it.

**Warnings the stage emits, and what each means:**

| Warning | Meaning |
|---|---|
| `N distinct window position(s) for the M passes requested` | The key-step is exactly `window` long, or sits against the video end, so only one window position contains it. All `M` passes take that position, so the same clip is cut `M` times. That repetition is deliberate: `tile_passes` gives every key-step the same number of passes regardless of length, and dropping the duplicates would under-expose exactly the key-steps whose geometry already matches an eval window. Expect this on any dataset with key-steps the same length as the window — it is informational, not a fault |
| `N of M positive window(s) also contain a NEIGHBOURING key-step` | Key-steps are closer together than the window's slack, so the single-action labels are incomplete. See the assumptions above |
| `N of M clip(s) are longer than the Ws window` | `tile_long` is off and a key-step did not fit, so clip duration now correlates with the action |
| `N of M clip(s) are SHORTER than the Ws window` | A window ran off the end of its source video |
| `only N non-SOP window(s) available but M requested` | Not enough footage clear of every key-step; the set will be short of negatives |
| `key-step ... has action N, outside the action list` | An annotation names an action the dataset does not define; that key-step is skipped |

**So WMCQ is not a general action-recognition augmentation.** For densely-packed actions, or any task where several actions occur inside one window, the single-label MCQ form it emits is the wrong shape and the chunk-consuming types (BCQ, sequential MCQ, DMCQ) are the right ones. Making WMCQ work there would mean a different labelling scheme — multi-label windows, or a dominant-action rule — not a parameter change.

**`window` must equal the evaluation sliding-window length.** This is the single most important setting. If the two diverge, the clips silently stop matching eval geometry — the augmentation still runs, still produces plausible-looking output, and still reports sample counts, but it no longer does the one thing it exists to do. There is no automatic check for this, because the augmentation service cannot see the evaluation config.

**This is a different stage from types 1-7.** WMCQ reads the source videos and their annotations, not the pre-cut chunks. It is therefore unaffected by `merge_small_chunks`, which runs after the video split and before augmentation — do not expect merged chunks to show up in WMCQ output.

**Key-steps longer than the window — and why `tile_long` matters:**

A key-step can be longer than one window, and there are two ways to handle it.

- **`tile_long: true` (default) — tile.** The key-step is covered by several windows of exactly `window`, each lying inside it.
- **`tile_long: false` — enlarge.** The window grows to `keystep_length + enlarge_pad` so it still contains the whole key-step.

Enlarging is simpler but introduces a subtle failure: **clip duration starts to correlate with the class.** If only one action is ever longer than the window, then every long clip is that action, and the model can score well on the training set by reading duration alone — without looking at the video at all. That shortcut does not exist at inference, where every window is exactly the same length, so the model finds nothing resembling what it learned. Tiling keeps every clip the same length, so duration carries no class information whatsoever.

The stage logs the clip-duration spread per action at the end of the run, and warns explicitly when it produced any clip longer than the window. Read that warning — it is the only visible symptom, and the sample counts look perfectly healthy either way.

**How the window count is chosen when tiling:**

| Setting | Windows per key-step | Meaning |
|---------|---------------------|---------|
| `tile_long`, `tile_passes: false` | `variants` | Fixed count regardless of key-step length |
| `tile_long` + `tile_passes` (default) | `variants * ceil(L / window)` | `variants` counts full **passes** over the key-step |

`tile_passes` exists because "N crops" otherwise means very different things for different classes: N crops shows a 1-second action N times over, but barely covers a 20-second action once. With `tile_passes`, one pass means 100% coverage for every class. Long actions do contribute proportionally more samples — that is the intent, not a side effect.

**Config parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `false` | Enable this stage |
| `exclude_action` | string | `""` | Action indices to exclude from positives (e.g. `"1_2"`) |
| `non_sop_action` | int | **REQUIRED** | Action index of "none of the above". Used as the label for every negative. **1-based**, matching the `(N)` prefix in `actions.json` — for a 4-action list whose last entry is `"(4) doing none of the above"`, this is `4`. The shipped `17` is a placeholder from another dataset and is almost certainly wrong for yours |
| `window` | float | `3.0` | Clip length in seconds. **Must equal the evaluation sliding-window length** |
| `variants` | int | `4` | Windows cut per key-step (see the table above for long key-steps) |
| `variants_per_action` | string | `""` | Per-action override, e.g. `"2:8"` cuts 8 windows per action-2 key-step |
| `tile_long` | bool | `true` | Tile over-long key-steps instead of enlarging the window |
| `tile_passes` | bool | `true` | With `tile_long`: read `variants` as full passes over the key-step |
| `enlarge_pad` | float | `1.0` | Padding added when a key-step exceeds `window` and `tile_long` is off |
| `neg_ratio` | float | `1.5` | Non-SOP window negatives per positive, drawn per source video |
| `neg_margin` | float | `0.5` | Keep negative windows this many seconds clear of any key-step |
| `seed` | int | unset | Optional: fix the random seed for reproducible negative sampling |

**`neg_ratio` sets the class prior the model learns, and it is not the same as the ratio the
evaluation presents.** A sliding-window evaluation sweeps every offset, so the great majority of
the windows it produces contain no key-step at all — on one real SOP test split, a 3s/1s sweep
gave roughly 20 non-SOP windows for every window touching a key-step, against the 1.5 that
`neg_ratio` trains. If the model over-fires — predicting an action on idle footage, showing up as
duplicate detections in the sequence — that gap is the first thing to look at, and raising
`neg_ratio` is the lever. Measure your own ratio rather than assuming: it depends on how much of
your footage is non-SOP.

**What WMCQ does not teach: partial overlap.** A positive window fully contains its key-step and a
negative clears every key-step by `neg_margin`, so the model only ever sees a key-step whole or
absent. The evaluation slides a window across every offset, so it also produces windows that
overlap a key-step *partially* — the onset and offset cases. Those are exactly the moments where
detections start and stop, and nothing in the training set resembles them. Expect boundary
behaviour to be the weakest part of a WMCQ-trained model, and read a low recall on short actions
in that light.

**`variants_per_action`:** raising overall negative pressure tends to kill the visually subtlest class first — it is the one with the least margin to lose. `variants_per_action` counterweights that by cutting more windows for the class that is being crowded out, without changing anything for the others.

**Recommended starting config.** The shipped defaults are already the configuration our own
runs converged on, so the only field you must set is `non_sop_action`:

```yaml
wmcq:
  enable: true
  non_sop_action: <your "none of the above" index>
  window: 3.0          # = evaluation sliding-window length
```

A note on what is *not* here: mining the model's own false positives and feeding them back as
hard negatives was tried and is **not** offered. Across every scored run it came out below its
own uniform-negative control — precision rose but recall collapsed, and in the worst case a
whole class stopped being predicted. Uniform negatives at `neg_ratio: 1.5` were better every
time. If you want to revisit it, treat it as a research question rather than a tuning knob.

**Output:** `wmcq/wmcq.json` plus `wmcq/videos/`. Each record carries a `meta` block so the generated set is auditable after the fact:

| Field | |
|---|---|
| `window_start`, `window_len` | where the clip was cut and how long it actually is |
| `source_keystep` | `[start, end]` of the key-step it covers (positives only) |
| `geometry` | `matched` = exactly `window` long · `enlarged` = stretched past it to fit an over-long key-step · `truncated` = ran off the end of the source video |
| `overlaps_other_keystep` | the window also contains a neighbouring key-step, so its single-action label is incomplete |
| `gt_action`, `pos_or_neg` | the label and whether the sample is a positive or a negative |
