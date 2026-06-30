# DDM Fine-tuning Microservice

A FastAPI-based microservice for managing DDM-Net (Action Segmentation) fine-tuning jobs with real-time monitoring and job cancellation capabilities.


## Prerequisites
* Ubuntu 22.04 or later
* 2*A100 or more (To ensure reasonable batch size for training)
* CUDA Version 12.8.1 or above
* NVIDIA Driver 550.144.03 for A100


## Installation and Setup

1. Go to the repo
```bash
cd ddm-training-ms
```

2. Build docker image
```bash
# By make
make build DOCKER_TAG=<your-image-tag>
# e.g. make build DOCKER_TAG=latest

# Or via docker compose with a custom parallel-compile cap (default MAX_JOBS=8):
docker compose build --build-arg MAX_JOBS=16 ddm-training-microservice
```

> `MAX_JOBS` caps parallel `nvcc` invocations for the DDM-Net CUDA
> extension during build. The default 8 keeps peak under ~30 GB RAM;
> tune up only if you have memory headroom.

3. Create assets folders to be mounted if not existed
```bash
mkdir -p assets/data assets/logs assets/metadata_db assets/results assets/weights assets/config assets/tools
```

4. Prepare training configuration
* Modify ddm_train_config.yaml in assets/config/
* You can find the explanation of each config [here](ddm/DDM-Net/config/config_guide.md)

5. Run the DDM-Net training microservice
```bash
export DDM_IMAGE=<your docker image>
# e.g. export DDM_IMAGE=ddm_ftms
docker compose up
```

## Service

After setting up, there would be 1 microservice:

1. **DDM Fine-tuning** (`ddm-training-microservice`)
   * Port: 32100 (configurable via `DDM_PORT`)
   * Performs DDM-Net fine-tuning using provided data
   * Requires GPU access

## Microservice API

1. **DDM-Net fine-tuning**: [api spec](api_spec/openapi.json)

## Quick Guideline

### 1. Prepare Your Data

Create the following structure. You can use the same dataset for both training and validation, or use separate datasets:

```
assets/
└── data/
    ├── <dataset_id_0>/
    │   ├── <video_0>/
    │   │   └── video_0_annotation.json
    │   ├── <video_1>/
    │   │   └── video_1_annotation.json
    │   └── ...
    ├── <dataset_id_1>/
    │   └── ...
    └── ...
```

**Note:** 
- If you have previously completed the VLM Training pipeline, you can directly reuse your VLM Training dataset here for DDM-Net fine-tuning without any modifications.
- For DDM-Net fine-tuning, use the original uploaded dataset, not the augmented version. Please use the dataset ID without the `augmented` prefix for the following training pipeline.

### 2. Modify Training Parameters

The parameters can be modified via `assets/config/ddm_train_config.yaml`. If you want to use a different config naming, you can export the environment variable `TRAIN_CONFIG_NAME` to the name you want. You can visit DDM-Net's [repo](ddm/README.md) for more information about the parameters.

The parameters listed below are handled by the microservice under the hood. No need to modify this manually:
   * `dataset_config.train_config.anno_path`
   * `dataset_config.train_config.data_root`
   * `dataset_config.val_config.anno_path`
   * `dataset_config.val_config.data_root`
   * `raining_config.output`
   * `raining_config.exp_name`

### 3. Send Training Request

**HTTP Request:**

**Option 1: Using separate datasets for training and validation**
```bash
curl -X POST "http://localhost:32100/api/v1/fine-tuning/start?dataset_id=<train_dataset_id>&validation_dataset_id=<val_dataset_id>" \
  -H "Content-Type: application/json"
```

**Option 2: Using the same dataset for both training and validation**
```bash
curl -X POST "http://localhost:32100/api/v1/fine-tuning/start?dataset_id=<your_dataset_id>" \
  -H "Content-Type: application/json"
```

**Parameters:**
- `dataset_id` (required): Training dataset ID
- `validation_dataset_id` (optional): Validation dataset ID. If not provided, will use the same dataset as training
- **Note**: The `dataset_id` should be the original uploaded dataset, not the augmented version. Do not use dataset IDs with the `augmented` prefix

The response format would be:
```json
{
   "job_id":"job_id",
   "status":"queued",
   "message":"Fine-tuning job has been queued and will start shortly",
   "created_at":"2025-12-03T15:34:28.642798"
}
```

### 4. Check Training Status

```bash
curl "http://localhost:32100/api/v1/fine-tuning/status/{job_id}"
```

The response format would be:
```json
{
   "job_id": "job_id",
   "status": "running",
   "progress": 0.0,
   "current_step": 0,
   "total_steps": 1530,
   "loss": 12.7,
   "created_at":"2025-12-03T15:34:28.642798",
   "updated_at":"2025-12-03T15:38:16.054531"
}
```

### 5. Cancel Training Job

```bash
curl -X POST "http://localhost:32100/api/v1/fine-tuning/cancel/{job_id}"
```

### 6. Tensorboard

To monitor training progress with Tensorboard:

```bash
tensorboard --logdir /path/to/your/lightning_logs
```

For example, to view a specific training job's tensorboard logs:

```bash
tensorboard --logdir /<your result folder>/<job_id>
```

## Development

### Running in Development Mode
To run DDM-Net in development mode, you can refer to DDM-Net's repo. We provide 

This will start an interactive bash session in the container for debugging.

## Troubleshooting

1. **Training job fails immediately**
   - Check if the model weights are correctly placed in `assets/weights/` (only for nvdino backbone)
   - Verify the dataset format matches DDM's expectations
   - Check logs in `assets/results/<job_id>/log.txt`

2. **GPU memory issues**
   - Adjust batch size in training config
   - Ensure no other processes are using the GPU

3. **Database connection errors**
   - Ensure the Postgres container is running
   - Check environment variables for database credentials

## License

This project is dual-licensed under the `CC-BY-4.0 AND Apache-2.0` terms in the top-level [`LICENSE`](../../../../LICENSE) file: source code under Apache-2.0, documentation under CC-BY-4.0. Bundled third-party software is listed in [`THIRD_PARTY_NOTICES.md`](../../../../THIRD_PARTY_NOTICES.md).


## Acknowledgments

This project incorporates code from the following open-source repositories:

- **DDM-Net**: [MCG-NJU/DDM](https://github.com/MCG-NJU/DDM) - Generic event boundary detection for action segmentation

We thank the authors for their excellent work.


