# GQAs Pre-flight — Local vLLM Launch and Validation

Run these steps when using the local vLLM backend (Qwen3-8B or Qwen3.5-27B) for GQAs
generation. Complete all three steps before delegating to `/sop-data-augmentation`.

---

## Why this matters

Qwen3-series models have a built-in thinking mode that causes `content=null` in the API
response when misconfigured — silently producing ALL-empty GQAs samples. The augmentation
pipeline runs without error but generates zero usable training signal. The correct launch
flags differ by model:

- **Qwen/Qwen3-8B** — launch **without** `--reasoning-parser qwen3`. The parser causes
  `content=null`; omitting it keeps the full response in `content`.
- **Qwen/Qwen3.5-27B** — launch **with** `--reasoning-parser qwen3` (hybrid model,
  required). The parser strips `<think>` tokens so `content` contains only the answer.

---

## Step 1 — Launch vLLM

**Qwen/Qwen3-8B** (default, no `--reasoning-parser`):

```bash
docker run -d --name sop-vllm --gpus all --ipc host -p 9000:9000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai@sha256:2e08b462bb444a6da8a84a533f09024c61617574e67386efe4a723a0633fcc6a \
  --model Qwen/Qwen3-8B --port 9000 \
  --max-model-len 40960 --gpu-memory-utilization 0.65
  # Do NOT add --reasoning-parser qwen3 for this model
```

**Qwen/Qwen3.5-27B** (higher-capacity; use the script which sets the correct flags):

```bash
scripts/launch_vllm.sh  # defaults: Qwen/Qwen3.5-27B, --reasoning-parser qwen3, max-len 32768
```

Set in `augment_config.yaml`:

```yaml
gqas:
  llm_type: "local"
  enable_thinking: "false"   # sends /no_think as system message → skips thinking, ~100 tokens/response
```

---

## Step 2 — Probe-validate before triggering augmentation

Send a test request and confirm the response content is non-empty:

```bash
MODEL="Qwen/Qwen3-8B"   # match the model you launched

RESPONSE=$(curl -s http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"system\",\"content\":\"/no_think\"},{\"role\":\"user\",\"content\":\"Rephrase: Q: What is the operator doing? A: Installing a fan. Output 1 JSON object: {\\\"question\\\":\\\"...\\\",\\\"answer\\\":\\\"...\\\"}\"}],\"max_tokens\":256,\"temperature\":0}")

CONTENT=$(echo $RESPONSE | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'] or '')")

if [ -z "$CONTENT" ]; then
  echo "ERROR: vLLM returned empty content — thinking-mode misconfiguration"
  echo "  Qwen/Qwen3-8B:    restart WITHOUT --reasoning-parser qwen3"
  echo "  Qwen/Qwen3.5-27B: restart WITH    --reasoning-parser qwen3 (via scripts/launch_vllm.sh)"
else
  echo "OK: response is non-empty, safe to proceed"
fi
```

If the probe returns empty content: stop, kill the container, apply the correct flag for the
model, relaunch, and re-run this probe before proceeding.

---

## Step 3 — Verify GQAs output after augmentation completes

After `/sop-data-augmentation` finishes, always run this check before proceeding to VLM training:

```bash
python3 -c "
import json
data = json.load(open('<augmented_dataset_path>/gqas/gqas.json'))
empty = [d for d in data if not d['conversations'][1].get('value','').strip()]
print(f'GQAs: {len(data)} total, {len(empty)} empty')
if len(empty) == len(data):
    exit(1)  # All empty — abort, apply Step 2 fix, re-augment
"
```

If all GQAs are empty: delete the augmented dataset, fix the launch flags (Step 2), and
re-augment. Do not proceed to VLM training with empty GQAs.

After augmentation completes, stop the vLLM container to free GPU memory:

```bash
docker rm -f sop-vllm
```
