# Local vLLM Server (launch_vllm.sh)

Helper script to deploy a local vLLM server for GQA augmentation. Uses Docker by default (no pip install needed). The server provides an OpenAI-compatible API that the GQAs augmentation stage calls.

## Location

Bundled with this skill at `scripts/launch_vllm.sh` (relative to the skill directory). Also available at `scripts/launch_vllm.sh` in the BP repo root.

## Usage

```bash
# Default model on port 9000 (Docker mode). For the current default, see the MODEL variable in scripts/launch_vllm.sh.
scripts/launch_vllm.sh

# Custom model and port
scripts/launch_vllm.sh --model Qwen/Qwen2.5-7B --port 8000

# Multi-GPU tensor parallel
scripts/launch_vllm.sh --tp 2

# Stop the server (frees GPU memory)
scripts/launch_vllm.sh --stop

# Bare-metal mode (pip install + foreground, no Docker)
scripts/launch_vllm.sh --bare-metal
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | see `MODEL` in `scripts/launch_vllm.sh` | HuggingFace model name |
| `--port` | `9000` | Port to serve on |
| `--tp` | `1` | Tensor parallel size (number of GPUs) |
| `--max-len` | `262144` | Max model context length |
| `--bare-metal` | — | Skip Docker, use pip + vllm directly (foreground) |
| `--stop` | — | Stop the running vLLM Docker container |

## Startup Time

Model loading + warmup typically takes a few minutes (longer for larger models). The script waits up to 10 minutes and prints a ready message with the server URL.

## After Launch

When ready, the script prints a config snippet to paste into `augment_config.yaml`. **Use the snippet exactly as printed** — do not edit the model name based on this document. The shape looks like:

```yaml
gqas:
  llm_type: "local"
  local_llm_url: "http://<machine_ip>:9000/v1"
  llm: <served_model_id>      # comes from the script's output / /v1/models — not from this doc
  enable_thinking: "false"
```

## Verify

```bash
curl http://localhost:9000/health
curl http://localhost:9000/v1/models
```

## Docker Details

- Container name: `sop-vllm`
- Image: `vllm/vllm-openai:latest`
- GPU access: `--gpus all`
- HuggingFace cache: mounts `~/.cache/huggingface` so model weights persist between runs
- Logs: `docker logs -f sop-vllm`
