# DDM-Net Fine-tuning Reference

## Architecture

DDM-Net (Dual-Domain Matching) is a temporal boundary detector:
- **Backbone**: ResNet-50 pretrained on ImageNet
- **Task**: Frame-level binary classification (boundary vs. non-boundary)
- **Primary metric**: `val/f1_score` (higher is better; >0.95 is excellent)
- **Training framework**: PyTorch Lightning with plain-text logging (`enable_progress_bar=False`)

## Log Format

Training logs use plain-text format. The key lines to parse for the primary metric:

```
Epoch N, global step M: 'val/f1_score' reached X.XXXXX (best Y.YYYYY), saving model to '<path>' as top 3
```

Parse with:
```bash
grep -E "Epoch [0-9]+, global step [0-9]+: 'val/f1_score'" results/<job_id>/log.txt
```

## Microservice API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/api/v1/fine-tuning/start` | POST | Start a training job |
| `/api/v1/fine-tuning/status/<job_id>` | GET | Job status: `status`, `progress`, `current_step`, `total_steps`, `loss` |
| `/api/v1/fine-tuning/cancel/<job_id>` | POST | Cancel a running job |
| `/api/v1/fine-tuning/all_jobs` | GET | List all jobs |

**Auto-set fields**: The `/start` endpoint automatically sets `anno_path`, `data_root`, `output`, and `exp_name`. Do NOT include these in the config YAML.

## Config Parameters

| Parameter | Tunable | Typical Range | Notes |
|-----------|---------|---------------|-------|
| `batch_size` | Yes | 4–32 | Use default; reduce if CUDA OOM |
| `resolution` | Yes | 224–512 | Higher = better quality, more memory |
| `num_gpus` | Yes | 1–8 | Must match available GPUs |
| `workers` | Yes | 4–16 | DataLoader workers |
| `epochs` | Yes | 10–50 | 30 is a good default |
| `anno_path` | No | — | Auto-set by service |
| `data_root` | No | — | Auto-set by service |
| `output` | No | — | Auto-set by service |
| `exp_name` | No | — | Auto-set by service |

## Dataset Requirements

Expected directory structure under `assets/data/<dataset_id>/`:
```
assets/data/<dataset_id>/
├── <video_id>.mp4                        ← video file (must be lowercase .mp4)
├── <video_id>/
│   └── <video_id>_annotation.json        ← annotation for that video
├── <video_id2>.mp4
├── <video_id2>/
│   └── <video_id2>_annotation.json
└── ...
```

- Videos must be lowercase `.mp4` — the dataset generator uses `glob("*.mp4")` which is case-sensitive on Linux. Uppercase `.MP4` files are silently skipped and cause an empty dataset error.
- `anno_path`, `data_root`, `output`, and `exp_name` are fully managed by the service. Do not set them in the config YAML.
- The service auto-generates the combined annotation JSON when `/start` is called.

## Docker Rebuild Note

If changes are made to DDM-Net source code (`ddm/`), the Docker image must be rebuilt — the top-level `docker-compose.yml` does NOT bind-mount the `ddm/` source directory:
```bash
docker compose build ddm-training-microservice
docker compose up -d ddm-training-microservice
```
