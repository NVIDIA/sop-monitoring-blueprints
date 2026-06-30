# Prerequisites Checklist

The orchestrator runs these checks before starting any pipeline work.
Hard failures block execution. Soft warnings are logged but don't block.

## Hard Prerequisites (block if missing)

### GPU & Memory
```bash
# GPU free memory — need ≥ 60 GB for VLM training, ≥ 10 GB for DDM training
nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader

# RAM — need ≥ 40 GB available for CPU-side LoRA merge (LoRA runs only; skip check for full fine-tune)
awk '/MemAvailable/ {print $2/1024/1024}' /proc/meminfo
```
- Block if GPU memory < 10 GB free (another job may be running)
- Block if RAM < 20 GB available

### No running training jobs
```bash
# Check CR2 service
curl -s http://localhost:32080/api/v1/fine-tuning/all_jobs | python3 -c \
  "import json,sys; jobs=json.load(sys.stdin); \
   running=[j for j in jobs.values() if j['status'] in ('running','queued')]; \
   print('RUNNING:', running)"

# Check DDM service
curl -s http://localhost:32100/api/v1/fine-tuning/all_jobs 2>/dev/null
```

### Required Docker images
```bash
docker images --format "{{.Repository}}:{{.Tag}}" | grep -E \
  "ddm_ftms|cr_ftms|vlm_inference_service|sop-annotation-backend|sop-data-generation"
```
Required: `ddm_ftms:latest`, `cr_ftms:latest`, `sop_inference_bp:vlm_inference_service_*`, `sop-annotation-backend:latest`, `sop-data-generation:latest`


If `sop-annotation-backend` or `sop-data-generation` are missing, they must be built first:
```bash
cd <training_bp_root>
docker compose build annotation-backend sop-data-gen
```

### Training BP services running
```bash
curl -s http://localhost:5487/health   # sop-data-gen
curl -s http://localhost:8100/health   # annotation-backend
```
If not running: `docker compose up -d` from `training_bp_root`

### Dataset structure valid
```bash
# actions.json exists
ls <dataset_path>/actions.json

# At least 5 video directories with annotations
find <dataset_path> -name "*_annotation.json" | wc -l

# actions.json has correct format
python3 -c "
import json
d = json.load(open('<dataset_path>/actions.json'))
assert 'actions' in d and len(d['actions']) >= 2, 'actions.json must have >=2 actions'
print('Actions:', len(d['actions']))
"
```

### Test dataset structure valid
Same checks as training dataset.

### VLM base model weights present
```bash
ls <vlm_weights_dir>/config.json   # e.g. fine_tune/cr2/weights/Cosmos-Reason2-2B/
```
If missing, download from HuggingFace:
```bash
huggingface-cli download nvidia/Cosmos-Reason2-2B --local-dir fine_tune/cr2/weights/Cosmos-Reason2-2B
```

### DDM base model weights present
```bash
ls <ddm_config_data_root>   # dataset pointed to by ddm_train_config.yaml
```

---

## Soft Prerequisites (warn but continue)

### API keys for GQAs
```bash
# Check if NIM API key is available (for GQAs augmentation)
echo "${NGC_API_KEY:-not set}" | head -c 20
echo "${NV_DEVELOPER_API_KEY:-not set}" | head -c 20
```
If no key: warn that GQAs will be skipped or use local vLLM.

### pyyaml installed on host
```bash
python3 -c "import yaml; print('pyyaml OK')" 2>/dev/null || echo "WARNING: pip install pyyaml"
```

### peft installed on host (LoRA runs only — skip for full fine-tune)
```bash
python3 -c "import peft; print('peft OK')" 2>/dev/null || echo "WARNING: pip install peft"
```
Only required if the VLM training run produces a LoRA adapter (checkpoint contains `adapter_config.json`). Full fine-tune runs don't need peft because there is no adapter to merge.

---

## Remediation Guidance

| Missing item | Auto-fixable? | Command |
|-------------|--------------|---------|
| sop-annotation-backend image | Yes | `docker compose build annotation-backend` |
| sop-data-generation image | Yes | `docker compose build sop-data-gen` |
| Services not running | Yes | `docker compose up -d` |
| pyyaml missing | Yes | `pip install pyyaml` |
| peft missing | Yes | `pip install peft` |
| GPU in use | No | Wait or cancel existing job |
| ddm_ftms/cr_ftms missing | No | Must be pre-built by administrator |
| vlm_inference_service missing | No | `make -C docker build_vlm_inference_service` |
| Dataset missing | No | User must provide annotated dataset |
