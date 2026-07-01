# DeepStream SOP SKILL

AI coding assistant skill and reference materials for building the **NVIDIA DeepStream SOP
Inference Microservice** — GPU-accelerated temporal action detection + VLM-based
SOP compliance monitoring.

## Architecture

![DeepStream SOP Inference Agentic Workflow](assets/DeepStream-SOP-Inference-Agentic-Workflow.png)

**Key data flow:** Input sources → FastAPI `/v1/chat/completions` → `SOPProcessManager` → per-request `SOPVideoProcessor` with 4 stages: DeepStream GEBD pipeline (e.g. DDM, GPU) → Clip post-process → VLM inference (Cosmos Reason 1/2) → SOP Checker → SSE stream / Kafka output.

## Key Features

- **OpenAI-compatible API**: chat completions format with SSE streaming
- **Multi-source video input**: files, HTTP URLs, RTSP, Basler cameras
- **RTSP streaming output** *(optional, opt-in — § 18)*: re-stream live inputs over RTSP at `/ds-out/{sensor_id}` (CPU `x264enc` or GPU `nvv4l2h264enc` via `SW_ENCODER`). Generated only when explicitly requested in the generation prompt
- **GPU-accelerated event boundary detection**: any **GEBD** model via DeepStream + Triton CAPI (default: DDM)
- **VLM analysis**: Cosmos Reason 1 or 2 for action classification (`VLLM_MODEL_PATH` is required — no default)
- **SOP compliance**: missing/misordered step detection per cycle
- **Basler camera support**: pylonsrc GStreamer plugin (gst-plugin-pylon v1.0.0)
- **Kafka messaging**: optional event publishing per chunk
- **Prometheus metrics**: request counts, latency, GPU utilization
- **Docker deployment**: multi-stage build with kafka service

## Optional features (generate only when explicitly requested)

- **RTSP streaming output (§ 18)** — re-stream live inputs over RTSP at `/ds-out/{sensor_id}` with a `SW_ENCODER` CPU/GPU toggle. This is an **opt-in** feature.

---

## Install as Claude / Codex Skill

The `deepstream-sop` skill is agent-agnostic — a `SKILL.md` plus `references/` — and works with both **Claude Code** and **OpenAI Codex**. Each auto-discovers any directory containing a `SKILL.md`:

- **Claude Code** → project-local `.claude/skills/` (or global `~/.claude/skills/`)
- **Codex** → global `~/.codex/skills/`

Clone the repo and copy the `deepstream-sop` skill folder into the skills directory of whichever agent you use:

```bash
git clone https://github.com/NVIDIA/sop-monitoring-blueprints.git
SKILL_SRC="$(pwd)/sop-monitoring-blueprints/agentic/ds-sop-skills/deepstream-sop"

# Claude Code (project-local)
mkdir -p .claude/skills && cp -r "$SKILL_SRC" .claude/skills/

# Codex (global)
mkdir -p ~/.codex/skills && cp -r "$SKILL_SRC" ~/.codex/skills/
```

> Start a new agent session after installing — skills load at startup. To update, run `git pull` in the clone (or re-clone) and re-run the copy command.

---

## Quick Start

### 1. DeepStream SOP Microservice Generation

```
Please follow instructions in @example_sop_prompt.md to generate a SOP microservice in folder @ds_sop_microservice
```

To additionally generate the **optional RTSP streaming output** feature (§ 18), add the *"with rtsp streaming output feature"* phrase:

```
Please follow instructions in @example_sop_prompt.md to generate a SOP microservice with rtsp streaming output feature in folder @ds_sop_microservice
```

### 2. DeepStream SOP Microservice Evaluation

```
Please follow @eval_sop_prompt.md to evaluate @ds_sop_microservice
with the following env settings:

VLLM_MODEL_PATH="/models/cosmos-reason1.1-7b/"
DDM_MODEL_PATH="/models/ddm/checkpoint.pth.tar"
MODEL_ROOT_DIR=/models
ACTION_CONFIG_PATH=/opt/sop/configs/actions.json
VLM_PROMPT_PATH=/opt/sop/configs/vlm_prompts.txt
VLM_FPS=8.0
VLM_MAX_PIXELS=81920
HOST_CACHE=$HOME/.cache/ds_sop
TEST_VIDEO_PATH=/path/to/test_video_whole_sop_h264.mp4
USER_ID=0
GROUP_ID=0

The pylon SDK binary is at @ds_sop_microservice/binaries/pylon-25.10.2_linux-x86_64_setup.tar.gz.
Fix any issues found during evaluation.
```

### 3. SOP with Basler Camera Evaluation

```
Please evaluate @ds_sop_microservice with physical Basler camera serial 40748152,
using max_length_sec=2.0. Follow @deepstream-sop/skill.md for rules.

Additional env settings:
ACTION_CONFIG_PATH=/path/to/configs/assy17_actions.json
VLM_PROMPT_PATH=/path/to/configs/assy17_vlm_prompts.txt
VLM_FPS=8.0
```

### 4. Video File Input Latency Measurement

```
Please measure TTFC and C2C latency for file input using /path/to/test_video.mp4.
Follow @deepstream-sop/skill.md for the file latency measurement method.
```

### 5. Troubleshooting & Debug Prompts

**Inspect chunk results:**
```
Show me the chunk metadata, SOP checker results, and VLM responses for the test file.
```

**Chunk result mismatch with reference:**
```
The chunk results from nvds-sop:latest differ from the reference at
https://github.com/NVIDIA/sop-monitoring-blueprints/tree/main/microservices/sop-inference-bp.
Compare the code, identify the differences, and fix them.
```

**Server hang:**
```
The server may be hung. Please diagnose — check logs and active threads.
For /v1/chat/completions, use a 10s timeout to detect hangs.
```

---

## Customizing the SOP Checker

The SOP compliance checker (`missing_number_detector.py`) can be generated from your own
`actions.json` to bake in SOP-specific logic such as **unordered action groups** (steps that can
be performed in any order without triggering a mis-order violation). See
`deepstream-sop/reference/skill_06b_sop_checker.md § 6b-G` for full details.

### Unit-testing skill_06b in isolation

Use this when you want to generate and verify the detector for a specific `actions.json` without
running the full microservice generation.

**Prompt to Claude Code:**
```
Read deepstream-sop/reference/skill_06b_sop_checker.md and generate
missing_number_detector.py for /path/to/your/actions.json
```

If you omit the path, Claude defaults to `configs/actions.json` relative to the current working
directory. Claude will:

1. Parse your `actions.json` (required and skippable actions)
2. Auto-deduce unordered groups from action descriptions (no input needed)
3. Write `nvds_action_detector/missing_number_detector.py` — standalone, same interface as the
   reference file
4. Write `tests/test_missing_number_detector.py` with concrete test sequences

Run the generated tests to verify:
```bash
python3 -m pytest tests/test_missing_number_detector.py -v
```

**`actions.json` format:**
```json
{
  "actions": ["(1) first step", "(2) second step", ...],
  "actions_can_be_skipped": ["(N) optional step"]
}
```

The original `actions.json` is never modified.

### Real run — full microservice generation (top-level skill)

Use this to generate the complete DeepStream SOP microservice. `skill.md` automatically triggers
the § 6b-G generation workflow when it reaches the SOP checker step, reading
`configs/actions.json` from the project root.

**Prompt to Claude Code (see `example_sop_prompt.md` for the full version):**
```
Please follow instructions in @ds-sop-skills/example_sop_prompt.md to generate a SOP microservice
in folder @ds_sop_microservice
```

Place your custom `actions.json` at `configs/actions.json` inside the target project before
generation, or set `ACTION_CONFIG_PATH` to point to it at runtime. If `configs/actions.json` is
absent or invalid, Claude falls back to copying the generic reference detector.

---

## Key Reference

| Topic | File |
|-------|------|
| Architecture diagram | `deepstream-sop/references/sop_architecture.svg` |
| Generation prompt | `example_sop_prompt.md` |
| Evaluation prompt | `eval_sop_prompt.md` |
| Skill (all code patterns) | `deepstream-sop/skill.md` |
| RTSP streaming output *(optional, opt-in)* | `deepstream-sop/references/skill_18_rtsp_streaming_output.md` |

## Reference Implementation

- **GitHub**: https://github.com/NVIDIA/sop-monitoring-blueprints/tree/main/microservices/sop-inference-bp
- **Local**: `sop-monitoring-blueprints/microservices/sop-inference-bp/` directory (from a local clone of the repository)

## 3rdparty License
- Refer to `deepstream-sop/references/Dockerfile_reference` for a complete list of third-party dependencies included in this project.
- This project will download and install additional third-party open source software projects. Review the
license terms of these open source projects before use.
- Building the final container from `deepstream-sop/references/Dockerfile_reference` requires Basler Pylon SDK, which is subject to separate license terms. Users must independently download and accept the [Pylon SDK license terms](https://docs.baslerweb.com/pylonapi/cpp/licensing) before proceeding. The [Pylon-SDK-25.10](https://www.baslerweb.com/en/downloads/software/1932603569/) can be obtained from the official Basler website after completing the required registration form.
