# VLM Data Augmentation FastAPI Service

A FastAPI-based microservice for VLM (Vision Language Model) SOP monitoring data augmentation that automatically generates BCQA (Binary Choice QA), MCQA (Multiple Choice QA), GQA (General QA), DMCQA (Dynamic Multiple Choice QA), DSQA (Dynamic Shuffling QA), ENQA (Extra Negative QA) data from given SOP actions definition

## Prerequisites
- Docker 28.2.2 or later
- Docker Compose v2.36.2 or later
- NVIDIA API key to make request to NVIDIA NIM API
   - Refer to [NVIDIA NIM](https://build.nvidia.com/explore/discover) for how to get the API key

## Installation and Setup
1. Clone the repo
```
git clone <repo-url>
cd data-generation-pipeline
```

2. Docker login
```
docker login nvcr.io
```

3. Update .env
```
# replace NGC_API_KEY value
NGC_API_KEY=<your nvidia api key that can access LLM NIM API>
```

4. Create assets folders to be mounted
```
mkdir assets/data assets/logs assets/metadata_db
```

5. Run data / QA augment MS
```
docker compose up --build
```

## Services
After setting up, there would be 1 microservice
2. **Data / QA Generation** (`sop-data-gen`)

   * Port: 5487 (configurable via `DATA_GEN_PORT`)

   * Generates GQAs, BCQAs, MCQAs, DSQA, DMCQA, ENQA data for VLM fine-tuning

      * The GQAs generation would utilize NVIDIA LLM NIM for generation

      * Requires NVIDIA API Key


## Microservice API
1. **Data / QA generation**: [api spec](api_spec/openapi_spec.json)


## Quick Guideline

### 1. Prepare Your Data

Create the following structure:
```
assets/
  |── data/
        |
      your_label_data_id/
        ├── video_folder_1/
        │   ├── chunk_1.MP4
        │   ├── chunk_2.MP4
        │   └── annotation.json (optional, can be omitted)
        ├── video_folder_2/
        │   ├── chunk_1.MP4
        │   └── chunk_2.MP4
        └── actions.json (required)
```

### 2. Modify Augmentation Parameters
The parameters can be modified via `augment_config.yaml`. There's a template config inside `assets/config` folder.

All augmentation stage can be disabled by setting `enable` to `false`.


Below are the explaination for each parameter

* **General Config**
  * `video_extention`: Video extension to be used (recommend using mp4)

* **BCQ (Binary QA - Yes/No question) Config**
  * `enable`: Whether to enable BCQ augmentation stage (default `true`)
  * `negative_ratio`: The positive and negative QA ratio (2.0 means there would be 1 yes QA and 2 no QA)
  * `subject`: Who conduct the SOP action
  * `exclude_action`: Action to be excluded from the BCQ generation

* **Sequential MCQ (Multiple Choices QA) Config**
  * `enable`: Whether to enable MCQA augmentation stage (default `true`)
  * `max_chunk_len`: The maximum number of actions to be included into the generated MCQ chunk (2 means the generated MCQ chunk QA would include chunk containing action 1, chunk containing action 1 + 2, but not chunk containing action 1 + 2 + 3 or more)
  * `exclude_action`: Action to be excluded from the MCQ generation

* **GQAs Config**
  * `enable`: Whether to enable GQA augmentation stage (default `true`)
  * `llm_type`: LLM type, local or nvidia (local deploy nim or build.nvidia.com API)
  * `local_llm_url`: Local LLM URL to be used for GQA augmentation
  * `llm`: NVIDIA NIM API LLM Model to be used for GQA augmentation
  * `num_qa_llm`: Number of QA pairs to be genrerated by LLM
  * `num_qa_per_chunk`: Number of QA pairs to sample from num_qa_llm to be the final GQA pairs
  * `exclude_action`: Action to be excluded from the GQA to GQAs generation

  > The NVIDIA API key is read only from the `NGC_API_KEY` environment variable (your `.env`); it is not configurable in `augment_config.yaml`.

* **Golden GQA**
  * `enable`: Whether to enable golden GQA augmentation stage (default `true`)

* **Dynamic MCQ**
  * `enable`: Whether to enable DMCQA augmentation stage (default: `false`)
  * `exclude_action`: Action to be excluded from the dynamic MCQ generation
  * `non_sop_action`: Action index of non-SOP action option (This must be set)
    * non-SOP action option is the action option like "none of the above", "doing action not belong to the defined SOP", etc.
  * `min_options`: Minimum number of options (need to adjust according to the number of actions)
  * `max_options`: Maximum number of options (need to adjust according to the number of actions)
  * `num_pos`: Number of positive samples
  * `num_neg`: Number of negative samples

* **Dynamic Shuffling QA**
  * `enable`: Whether to enable DSQA augmentation stage (default: `false`)
  * `exclude_action`: Action to be excluded from the dynamic shuffling QA generation
  * `non_sop_action`: Action index of non-SOP action option (This must be set)
    * non-SOP action option is the action option like "none of the above", "doing action not belong to the defined SOP", etc.
  * `min_distractor`: Minimum number of distractor videos
  * `max_distractor`: Maximum number of distractor videos
  * `num_runs`: Number of runs for dynamic shuffling

* **Extra Negative Data QA**
  * `enable`: Whether to enable ENQA augmentation stage (default: `false`)
  * `exclude_action`: Extra negative source data action to be excluded from the ENQA generation
  * `extra_negative_data_id`: ID of the other labeled data to be used as extra negative data (This must be set)
  * `non_sop_action`: Base data action index of non-SOP action option (This must be set)
    * non-SOP action option is the action option like "none of the above", "doing action not belong to the defined SOP", etc.
  * `min_options`: Minimum number of options (need to adjust according to the number of actions)
  * `max_options`: Maximum number of options (need to adjust according to the number of actions)
  * `num_runs`: Number of runs for ENQA generation
  * `generate_all_options`: Generate all options QA for extra negative

### 3. Conduct Data / QA Generation

**HTTP Request:**
```bash
curl -X POST "http://localhost:5487/api/v1/augment?label_data_id=your_label_data_id"
```

## Input/Output Structure

### Input Structure (Annotation MS Output)
```
<label_data_id>/
├── <video_folder>/
│   ├── <video_chunk_1>
│   ├── <video_chunk_2>
│   └── annotation.json (optional)
├── <video2_folder>/
│   ├── <video2_chunk_1>
│   ├── <video2_chunk_2>
│   └── annotation.json (optional)
└── action.json (required)
```

### Output Structure (Data/QA Generation MS Output)
```
<augmented_dataset_id>/
├── bcq/
│   ├── videos/
│   └── bcq.json
├── mcq/
│   ├── videos/
│   └── mcq.json
├── golden_gqa/
│   ├── videos/
│   └── golden_gqa.json
├── gqas/
│   ├── videos/
│   └── gqas.json
├── dmcq/
│   ├── videos/
│   └── dmcq.json
├── ds/
│   ├── videos/
│   └── ds.json
└── en/
    ├── videos/
    └── en.json
```

## Response Format

```json
{
  "dataset_id": "dataset_12345678",
  "message": "All actions completed successfully"
}
```

## Two-operator (concurrent action) mode

The pipeline supports a "two-operator" mode in which two SOP actions occur concurrently in the same video chunk (e.g. one operator performs action 1 while another performs action 3). Single-operator data and two-operator data can co-exist in the same dataset.

### Filename convention

The action index encoded in the chunk filename tells the pipeline which actions are present:

* Single-action chunk: `<idx>_<base>.<ext>` (e.g. `01_video.mp4` means action 1).
* Concurrent two-action chunk: `<idx1>-<idx2>_<base>.<ext>` (e.g. `01-03_video.mp4` means actions 1 and 3 happen concurrently).

Indices are parsed by `vlm_aug/utils/helper.parse_video_action_indices`.

### Toggle

Two-operator mode is controlled by a single canonical flag in `assets/config/augment_config.yaml`:

```yaml
gqas:
  two_operator_mode: true
```

The flag lives under `gqas:` and is forwarded to the other augmentation stages, so you only need to set it once.

### Per-stage behavior

* `bcq` and `mcq`: use one unified question template across both modes. In two-operator mode the answers use the plural subject form (e.g. "the operators are X1, X2").
* `gqa`: keeps separate templates per mode (singular vs plural) - see the comment in `vlm_aug/cfg/gqa.py`.
* `spatial_localization`: runs in **either** mode. It auto-detects spatially-confusable action pairs from the action list, so it no longer requires two-operator mode to be enabled.
* `frame_drop`: runs in **either** mode whenever `frame_drop.iterations > 0`.
* `merge_small_chunks`: runs **only** in two-operator mode. It merges short adjacent clips that result from a two-operator split.

### Configuring `merge_small_chunks`

`merge_small_chunks` is configured **only** through `assets/config/augment_config.yaml`. The annotator no longer accepts a `mergeThreshold` override in the request body.

```yaml
merge_small_chunks:
  enable: true
  threshold: 0.2
```

* `enable`: turn the merge step on or off.
* `threshold`: minimum chunk length (in seconds) below which adjacent chunks are merged.

### Local LLM for `spatial_localization`

`spatial_localization` mirrors the LLM knobs already used by `gqa_to_gqas`. They are read from the `gqas:` block of `augment_config.yaml` and can also be passed as CLI flags:

* `--llm-type {nvidia,local}`: choose between the NVIDIA NIM API (`nvidia`) and a locally deployed NIM (`local`).
* `--local-llm-url`: URL of the local LLM endpoint (used when `--llm-type local`).
* `--api-key`: API key for the chosen endpoint (overrides `NGC_API_KEY` from `.env`).
* `--enable-thinking`: enable "thinking" mode for Qwen-style models (e.g. Qwen3-27B, Qwen2.5-8B). Leave this off for models that do not support a separate thinking phase.

A typical local-LLM run uses `--llm-type local` together with `--local-llm-url` pointing at the deployed Qwen3-27B or Qwen2.5-8B endpoint, and `--enable-thinking` when the served model supports it.

## License
This project is dual-licensed under the `CC-BY-4.0 AND Apache-2.0` terms in the top-level [`LICENSE`](../../../../LICENSE) file: source code under Apache-2.0, documentation under CC-BY-4.0. Bundled third-party software is listed in [`THIRD_PARTY_NOTICES.md`](../../../../THIRD_PARTY_NOTICES.md).