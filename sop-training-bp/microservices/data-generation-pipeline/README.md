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
  * `ngc_personal_key`: The NVIDIA API key. Please noted that this would override the API Key set in `.env`

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

**SOP Training BP:**
* We can also do it through [SOP training BP](https://gitlab-master.nvidia.com/sop-training-bp/sop-training-bp-deployment)


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

## License
The software and materials in this repository are governed by the [NVIDIA Software and Model Evaluation License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-and-model-evaluation-license/)