# Reference: CR2 VLM Fine-tuning

## Anomaly Detection

This skill detects and logs the following anomalies during training. It does **not** interpret them or suggest fixes — that is the responsibility of the tuning/analysis agents.

| Anomaly | Detection Method |
|---------|-----------------|
| **CUDA OOM** | "CUDA out of memory" or "OOM" in container logs |
| **NaN/Inf loss** | NaN in loss output |
| **Loss spike** | Sudden loss jump (>3× previous value) |
| **GPU temperature** | Temperature readings from nvidia-smi |
| **Low GPU utilization** | Utilization < 50% from nvidia-smi |

## API Endpoints

**Base URL:** `http://localhost:32080/api/v1` (fine-tuning endpoints). Health check is at the root: `http://localhost:32080/health`.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/health` (root, not under `/api/v1`) | Service health check |
| `POST` | `/fine-tuning/start?dataset_id=...` | Start training job |
| `GET` | `/fine-tuning/status/{job_id}` | Job progress (loss, step, GPU) |
| `POST` | `/fine-tuning/cancel/{job_id}` | Cancel running job |
| `GET` | `/fine-tuning/all_jobs` | List all jobs |

### Status Response Fields

```json
{
  "job_id": "uuid",
  "status": "running",       // queued | running | completed | failed | cancelled
  "progress": 45.2,          // percentage
  "current_step": 150,
  "total_steps": 300,
  "loss": 2.34,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T11:15:00Z"
}
```

## Monitoring Commands

```bash
# Live training logs
docker compose logs --since 5m cosmos-reason-microservice 2>&1 | grep -E "step|loss|epoch"

# GPU status
docker compose exec cosmos-reason-microservice nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader

# Check for OOM
docker compose logs --since 10m cosmos-reason-microservice 2>&1 | grep -i "out of memory\|OOM\|CUDA error"

# Check for NaN
docker compose logs cosmos-reason-microservice 2>&1 | grep -iE "loss.*nan|nan.*loss" | head -5

# Loss history
docker compose logs cosmos-reason-microservice 2>&1 | grep -oP "loss[:\s]+\K[0-9.]+" | tail -20

# Checkpoints
ls -lah assets/results/<job_id>/checkpoint-*/

# Database job status
docker compose exec metadata_db psql -U sop -d sop_db -c \
  "SELECT id, status, progress, loss, current_step, total_steps FROM training_job ORDER BY created_at DESC LIMIT 5;"
```
