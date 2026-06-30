# Cosmos-Reason Fine-tuning Microservice

A FastAPI-based microservice for managing Cosmos-Reason fine-tuning jobs with real-time monitoring and job cancellation capabilities.


## Prerequisites
* Ubuntu 22.04 or later
* 4 * A100 (For full-finetuning with cosmos-reason at reasonable batch size)
* CUDA Version 12.8.1 or above
* NVIDIA Driver 550.144.03 for A100


## Installation and Setup
1. Clone the repo
```bash
git clone <repo-url>
cd cr-training-ms
```

2. Build docker image
```bash
# By make
make build DOCKER_TAG=<your-image-tag>

# Or via docker compose with a custom parallel-compile cap (default MAX_JOBS=8):
docker compose build --build-arg MAX_JOBS=16 cosmos-reason-microservice
```

> `MAX_JOBS` controls how many parallel `nvcc` invocations `flash-attn`,
> `transformer_engine`, and `apex` use during build. 8 is conservative
> (peak ~30 GB RAM); raise it on hosts with more headroom, lower it if
> the build OOMs.

4. create assets folders to be mounted if not existed
```bash
mkdir assets/data assets/logs assets/metadata_db assets/results assets/weights
```

5. Run the SOP training BP
```bash
export VLM_IMAGE=<your docker image>
docker compose up
```

## Service
After setting up, there would be 1 microservice

1. **Cosmos-Reason Fine-tuning** (`cosmos-reason-microservice`)

   * Port: 32080 (configurable via `VLM_PORT`)

   * Performs Cosmos-Reason fine-tuning using generated data

   * Requires GPU access

## Microservice API
1. **Cosmos-Reason fine-tuning**: [api spec](api_spec/openapi.json)

## Quick Guideline

### 1. Prepare Your Data

Create the following structure:
```
assets/
  |── data/
      |
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
      └── gqas/
          ├── videos/
          └── gqas.json
```

### 2. Modify Training Parameters
The parameters can be modified via `assets/config/train_config.toml`. If you want to use a different config naming, you can export the environment variable `TRAIN_CONFIG_NAME` to the name you want.

The parameters listed below are handled by the microservice under the hood. No need to modify this manually.
   * `train.output_dir`
   * `logging.experiment_name`
   * `train.train_policy.dataset.name`
   * `train.train_policy.dataset.split`


### 3. Send Training Request

**HTTP Request:**

```bash
curl -X POST "http://localhost:32080/api/v1/fine-tuning/start?dataset_id=augmented_dataset_id" \
  -H "Content-Type: application/json"
```

The response format would be
```json
{
  "job_id": "job_id",
  "status": "queued",
  "message": "string",
  "created_at": "2025-07-24T10:14:34.036Z"
}
```

### 4. Check training status

```bash
curl "http://localhost:32080/api/v1/fine-tuning/status/{job_id}"
```

The response format would be
```json
{
  "job_id": "job_id",
  "status": "running",
  "progress": 45.2,
  "current_step": 150,
  "total_steps": 300,
  "loss": 2.34,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T11:15:00Z"
}
```

### 5. Cancel training job
```bash
curl -X POST "http://localhost:32080/api/v1/fine-tuning/cancel/{job_id}"
```

## License
This project is dual-licensed under the `CC-BY-4.0 AND Apache-2.0` terms in the top-level [`LICENSE`](../../../../LICENSE) file: source code under Apache-2.0, documentation under CC-BY-4.0. Bundled third-party software is listed in [`THIRD_PARTY_NOTICES.md`](../../../../THIRD_PARTY_NOTICES.md).
