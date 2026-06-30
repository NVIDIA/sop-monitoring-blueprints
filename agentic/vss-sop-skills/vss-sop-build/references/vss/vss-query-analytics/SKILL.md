---
name: vss-query-analytics
description: >-
  Query video analytics incidents, alerts, sensor data, and metrics from Elasticsearch
  via the VA-MCP server (port 9901). Use for any question about what happened in video —
  PPE violations, alerts, incidents, object counts, speeds, occupancy, or anything that
  requires looking up recorded events. This is the primary way to answer "what happened",
  "show me alerts", "any violations", "how many people", etc.
owner: NVIDIA
service: vss-sop
version: 1.0.0
license: CC-BY-4.0 AND Apache-2.0
reviewed: 2026-06-23
metadata:
  openclaw: { "emoji": "🔎", "os": ["linux"] }
  author: "nvidia <info@nvidia.com>"
  tags: ["vss", "analytics", "elasticsearch", "mcp"]
---

# Video Analytics (VA-MCP)

Queries incidents, alerts, and metrics stored in Elasticsearch via MCP JSON-RPC at port `9901`.

## Overview

Use this skill to fetch real-time or historical telemetry and compliance records stored in Elasticsearch. Always run queries programmatically via the VA-MCP server. 

Key capabilities:
- Getting incident details by unique identifier.
- Aggregating statistical trends (e.g. average people count over a time window).
- Searching for safety/PPE alerts across active video streams.

## Prerequisites

- **VA-MCP Running:** Ensure the `vss-va-mcp` service is running on port `9901`.
- **Session Identification:** All JSON-RPC queries require an active `mcp-session-id` fetched from an initialization header.

## Instructions

### 1. Two-Step JSON-RPC Pattern

To make queries, execute two consecutive shell/curl commands:
1. **Initialize Session:** Perform a POST call to `http://localhost:9901/mcp` with an `initialize` JSON payload, and grep for the `mcp-session-id` header.
2. **Execute Tool Call:** Send a POST call to `http://localhost:9901/mcp` with the `tools/call` method, passing the session header and the desired parameters.

### 2. Available Tool Options

Replace the parameters under Step 2 to target specific services:
- `video_analytics__get_incidents` — list recent incidents.
- `video_analytics__get_incident` — inspect detailed object lists / metadata.
- `video_analytics__get_sensor_ids` — find registered sensor identifiers.
- `video_analytics__get_places` — find configured spatial tags.
- `video_analytics__get_fov_histogram` — plot density curves.
- `video_analytics__analyze` — calculate metrics like speed or occupancies.

## Examples

### Get Incidents (Standard Query)

```bash
# Step 1: initialize — get session ID from response HEADER
SESSION_ID=$(curl -si -X POST http://localhost:9901/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"cli","version":"1.0"}},"id":0}' \
  | grep -i "mcp-session-id" | awk '{print $2}' | tr -d '\r')

# Step 2: call the tool using the session ID in the header
curl -s -X POST http://localhost:9901/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__get_incidents","arguments":{"max_count":10}},"id":1}' \
  | grep '^data:' | sed 's/^data: //' | jq -r '.result.content[0].text'
```

### Parameter Configurations

**Query a Specific Camera:**
```bash
-d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__get_incidents","arguments":{"source":"sensor_0","source_type":"sensor","max_count":20}},"id":1}'
```

**Filter to Confirmed Events Only:**
```bash
-d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__get_incidents","arguments":{"vlm_verdict":"confirmed","max_count":10}},"id":1}'
```

**Analyze Average Occupancy:**
```bash
-d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__analyze","arguments":{"source":"sensor_0","source_type":"sensor","start_time":"2025-09-11T00:00:00Z","end_time":"2025-09-11T23:59:59Z","analysis_type":"avg_num_people"}},"id":1}'
```

## Error Handling

### Connection Refused / Service Unreachable (port 9901)
If the connection is refused, the `vss-va-mcp` service might be offline:
1. Check the status of the container:
   ```bash
   docker ps -a --filter name=vss-va-mcp
   ```
2. Restart the service if it is down:
   ```bash
   docker compose -f deployments/compose.yml --profile bp_sop_2d restart vss-va-mcp
   ```

### Missing Session ID
If the command fails with `Bad Request: Missing session ID`, ensure:
1. Step 1 (the initialization call) ran successfully.
2. The `SESSION_ID` variable was correctly parsed from response headers (not the response body).

---

## License

Use of this skill is governed by the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en) and the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
