# Claude GQAs Backend — Setup Reference

Run these steps once before the first augmentation when `ANTHROPIC_API_KEY` is available.
This enables Claude (Haiku) for GQAs generation: no GPU required, ~10× faster than local vLLM.

---

## Step A — Patch gqa_to_gqas.py

Copy the `vlm_aug` package from the running data-gen container and apply two patches:
1. **Claude routing** with specificity preservation and optional thinking mode
2. **Empty-QA guard**: skip clips where the LLM returns no valid QA pairs

```bash
# 1. Copy the package
mkdir -p <run_dir>/overrides/vlm_aug
docker cp <data-gen-container>:/usr/local/lib/python3.10/dist-packages/vlm_aug/. \
  <run_dir>/overrides/vlm_aug/

# 2. Apply the patch
python3 - << 'PATCH'
import pathlib

f = pathlib.Path('<run_dir>/overrides/vlm_aug/gqa_to_gqas.py')
src = f.read_text()

CLAUDE_BLOCK = '''
    if str(args.llm).startswith("claude"):
        import anthropic as _anthropic, os as _os
        _client = _anthropic.Anthropic(api_key=_os.environ.get("ANTHROPIC_API_KEY", ""))
        _system = next((m["content"] for m in cur_messages if m["role"] == "system"), "")
        _system += (
            "\\n\\nCRITICAL — Specificity Preservation Rule: "
            "If the golden Q&A contains specific identifiers (ordinals, colors, part names, "
            "or any other distinguishing words that differentiate this action from similar ones), "
            "you MUST preserve those EXACT identifiers in BOTH the question AND the answer "
            "of every generated pair. Do NOT replace specific identifiers with generic terms. "
            "Dropping these identifiers is a critical error."
        )
        _msgs = [m for m in cur_messages if m["role"] != "system"]
        _use_thinking = (
            hasattr(args, "enable_thinking")
            and str(args.enable_thinking).lower() == "true"
        )
        _max_tokens = 8000 if _use_thinking else llm_cfg.get("max_tokens", 1024)
        _kw = dict(model=args.llm, max_tokens=_max_tokens, system=_system, messages=_msgs)
        if _use_thinking:
            _kw["thinking"] = {"type": "enabled", "budget_tokens": 5000}
        _resp = _client.messages.create(**_kw)
        llm_output = next((b.text for b in _resp.content if b.type == "text"), "")
    else:
'''
src = src.replace("    client = OpenAI(", CLAUDE_BLOCK + "        client = OpenAI(", 1)

src = src.replace(
    "        total_qa_req = num_qa_per_chunk",
    "        if not all_qa:\\n"
    "            logging.warning(f'No valid QA pairs for {video}. Skipping.')\\n"
    "            continue\\n"
    "        total_qa_req = num_qa_per_chunk"
)

f.write_text(src)
print("Patch applied successfully.")
PATCH
```

Record in `run_state.yaml`:
```yaml
overrides:
  vlm_aug_override: <run_dir>/overrides/vlm_aug
```

---

## Step B — Restart data-gen container with the key and override

```bash
docker rm -f <data-gen-container>

docker compose run -d \
  --name <data-gen-container> \
  --service-ports \
  -e "ANTHROPIC_API_KEY=$(printenv ANTHROPIC_API_KEY)" \
  -v "<run_dir>/overrides/vlm_aug:/usr/local/lib/python3.10/dist-packages/vlm_aug" \
  -v "<configs_dir>:/workspace/assets/config" \
  -v "<data_root>:/workspace/assets/data" \
  -v "<logs_dir>:/workspace/assets/logs" \
  sop-data-gen

docker exec <data-gen-container> pip install anthropic -q
```

---

## Step C — Set augment_config.yaml

```yaml
gqas:
  enable: true
  llm_type: "local"               # argparse only accepts "local" or "nvidia"
  llm: claude-haiku-4-5-20251001  # routed to Anthropic SDK by "claude-" prefix
  local_llm_url: ""               # unused for Claude
  enable_thinking: "false"        # "true" enables extended thinking — slower, more token cost
  num_qa_llm: 8
  num_qa_per_chunk: 2
```

---

## Verification

```bash
docker logs <data-gen-container> --since 1m 2>&1 | grep "api.anthropic.com"
# Expected: HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"
```

```python
import json
d = json.load(open('<augmented_dataset_path>/gqas/gqas.json'))
empty = [x for x in d if not x['conversations'][-1]['value'].strip()]
print(f'GQAs: {len(d)} total, {len(empty)} empty')
# Spot-check: answers should preserve action-specific identifiers from the golden GQA
for item in d[:5]:
    print(item['conversations'][-1]['value'][:100])
```
