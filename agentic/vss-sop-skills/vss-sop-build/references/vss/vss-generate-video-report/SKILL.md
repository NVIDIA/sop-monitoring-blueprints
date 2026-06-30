---
name: vss-generate-video-report
description: >-
  Generate and query incident reports from VSS — look up incidents in Elasticsearch,
  analyze incident patterns, generate narrative reports. Use when asked about incidents,
  incident reports, PPE violations, safety events, or "what happened". Requires the alerts
  profile to be deployed.
owner: NVIDIA
service: vss-sop
version: 1.0.0
license: CC-BY-4.0 AND Apache-2.0
reviewed: 2026-06-23
metadata:
  openclaw: { "emoji": "📋", "os": ["linux"] }
  author: "nvidia <info@nvidia.com>"
  tags: ["vss", "incident", "report"]
---

# Incident Report Workflows

Query incidents from Elasticsearch, analyze patterns, and generate reports. Requires the alerts profile — deploy with the `deploy` skill (`-p alerts`).

## Overview

Use this skill when you need to query incidents, evaluate safety violations (such as PPE compliance), or compile visual/narrative incident reports. This covers requests like:
- "Show me incidents from today"
- "Generate an incident report"
- "How many PPE violations this week?"
- "Summarize incidents for sensor X"
- "What happened at the loading dock?"

## Prerequisites

- **Alerts Profile Deployed:** Ensure that the alerts profile has been successfully configured and started (`vss-sop-deploy` with `-p alerts`).
- **Elasticsearch Access:** Port `9200` must be accessible.
- **VA-MCP Access:** Port `9901` must be reachable.

## Instructions

### 1. Querying Incidents via VA-MCP

To query incidents, you must follow the two-step MCP JSON-RPC pattern on port `9901`:
1. **Initialize Session:** Fetch a session ID from the response header.
2. **Call Get Incidents:** Invoke `video_analytics__get_incidents` with desired filters (sensor source, time range, VLM verdict).

### 2. Pattern Analysis

To identify peaks or compute metrics, invoke `video_analytics__analyze` with parameters like `max_min_incidents` or `avg_num_people`.

### 3. Report Generation

To generate a full narrative report, make a POST request to the VSS Agent at port `8000`. This triggers a Human-in-the-Loop (HITL) prompt-editing workflow.

## Examples

### Query Incidents (Two-Step Pattern)

```bash
# Step 1: initialize — get session ID
SESSION_ID=$(curl -si -X POST http://localhost:9901/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"cli","version":"1.0"}},"id":0}' \
  | grep -i "mcp-session-id" | awk '{print $2}' | tr -d '\r')

# Step 2: query incidents
curl -s -X POST http://localhost:9901/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__get_incidents","arguments":{"max_count":10}},"id":1}' \
  | grep '^data:' | sed 's/^data: //' | jq -r '.result.content[0].text'
```

### Filtering Examples

**Filter by Sensor:**
```bash
-d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__get_incidents","arguments":{"source":"sensor_0","source_type":"sensor","max_count":20}},"id":1}'
```

**Filter by Time Range:**
```bash
-d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__get_incidents","arguments":{"start_time":"2025-09-11T00:00:00.000Z","end_time":"2025-09-11T23:59:59.999Z","max_count":50}},"id":1}'
```

**Filter by VLM Verdict (Confirmed only):**
```bash
-d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__get_incidents","arguments":{"vlm_verdict":"confirmed","max_count":10}},"id":1}'
```

**Get Incident Details:**
```bash
-d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__get_incident","arguments":{"id":"<incident-id>","includes":["objectIds","info"]}},"id":1}'
```

### Pattern Analysis Examples

**Peak Incident Times:**
```bash
-d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"video_analytics__analyze","arguments":{"source":"sensor_0","source_type":"sensor","start_time":"2025-09-11T00:00:00Z","end_time":"2025-09-11T23:59:59Z","analysis_type":"max_min_incidents"}},"id":1}'
```

### Generate Narrative Report

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_message": "Generate an incident report for sensor sensor_0"}' | jq .
```

The VSS Agent will pause for Human-in-the-Loop input with options:
- Submit (empty) -> Approve and generate
- New text -> Replace prompt manually
- `/generate <desc>` -> Let LLM write a new prompt
- `/refine <instructions>` -> Refine current prompt
- `/cancel` -> Cancel

Generated reports are located at: `http://<HOST_IP>:8000/static/agent_report_<DATE>.md` (or `.pdf`)

## Error Handling

### Elasticsearch Connection / Index Failure
If querying incidents via MCP returns an index-not-found error or empty results:
1. Confirm Elasticsearch is healthy:
   ```bash
   curl -s http://localhost:9200/_cluster/health | jq .
   ```
2. Verify Logstash is processing Kafka events:
   ```bash
   docker logs mdx-logstash --tail 50
   ```

### Report Generation Failures (port 8000)
If narrative report requests fail:
1. Verify the VSS Agent container is running:
   ```bash
   docker ps -a --filter name=vss-agent
   ```
2. Check the container logs for prompt compilation or inference errors:
   ```bash
   docker logs vss-agent --tail 100
   ```

---

## License

Use of this skill is governed by the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en) and the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
