---
name: vss-summarize-video
description: >-
  Summarize long videos, generate shift reports, and analyze extended recordings.
  Use when asked to summarize a video, generate a shift summary, analyze a long
  recording, or create a daily activity report. Requires the LVS profile to be deployed.
owner: NVIDIA
service: vss-sop
version: 1.0.0
license: CC-BY-4.0 AND Apache-2.0
reviewed: 2026-06-23
metadata:
  openclaw: { "emoji": "📹", "os": ["linux"] }
  author: "nvidia <info@nvidia.com>"
  tags: ["vss", "summarize", "video", "lvs"]
---

# Video Summarization Workflows

Summarize long-form video content that exceeds standard VLM context limits. Requires the LVS profile — deploy with the `deploy` skill (`-p lvs`).

## Overview

Use this skill when asked to summarize extended recordings, analyze activity during full shifts (such as a night shift), or compile daily narrative activity summaries.

Common triggers:
- "Summarize this video"
- "Generate a shift summary"
- "What happened during the night shift?"
- "Create a daily activity report"

## Prerequisites

- **LVS Profile Deployed:** Ensure that the LVS profile is running (`vss-sop-deploy` with `-p lvs`).
- **Video Storage Ingest:** Target videos must first be uploaded/stored in VIOS before the agent can request a summary.

## Instructions

### 1. High-Level LVS Pipeline

Long-form video processing is orchestrated across several steps:
1. **Segment:** The video is partitioned into smaller, digestible chunks.
2. **Analyze:** Each segment chunk is sent to the VLM independently.
3. **Synthesize:** Chunk analysis summaries are aggregated by the LLM.
4. **Report:** A final, cohesive narrative report containing timestamped highlights is output.

### 2. Execution via REST API

Upload files using VIOS storage APIs, then invoke the VSS Agent `/generate` endpoint with the summarization command.

## Examples

### Upload Video & Summarize

```bash
# Step 1: Upload a video via VIOS Storage API
curl -s -X PUT http://localhost:30888/vst/api/v1/storage/file/night-shift-entrance.mp4 \
  --upload-file /path/to/video.mp4 | jq .

# Step 2: Request the VSS Agent to summarize
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_message": "Generate a summary report for night-shift-entrance.mp4"}' | jq .
```

### Camera / Time-bounded Summaries

**Summarize for a Specific Sensor:**
```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_message": "Can you generate a summary report for sensor_0?"}' | jq .
```

**Summarize a Specific Timeframe:**
```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_message": "Summarize the activity between 8am and 12pm"}' | jq .
```

## Error Handling

### VST Video Upload Failures (HTTP 500 / 404)
If uploading videos to storage-ms fails:
1. Verify the storage service is running:
   ```bash
   docker ps -a --filter name=storage-ms-sop
   ```
2. Verify available disk space:
   ```bash
   df -h
   ```
   If full, clean up prior logs and recordings using `cleanup_all_datalog.sh`.

### Summarization Task Timeout or Hang
If the VLM summarization hangs or times out:
1. Check the VSS Agent logs to locate the active segment chunk:
   ```bash
   docker logs vss-agent --tail 50
   ```
2. Ensure local VLM and remote NIM endpoints are online and responsive.

---

## License

Use of this skill is governed by the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en) and the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
