---
name: vss-search-archive
description: >-
  Search video archives using natural language — find events, objects, actions, and people
  across recorded video using Cosmos Embed1 semantic search. Use when asked to search for
  something in video, find events, locate objects, or query video archives. Requires the
  search profile to be deployed.
owner: NVIDIA
service: vss-sop
version: 1.0.0
license: CC-BY-4.0 AND Apache-2.0
reviewed: 2026-06-23
metadata:
  openclaw: { "os": ["linux"] }
  author: "nvidia <info@nvidia.com>"
  tags: ["vss", "search", "cosmos-embed", "archive"]
---

# Video Search Workflows

Search video archives by natural language using Cosmos Embed1 embeddings. Requires the search profile — deploy with the `deploy` skill (`-p search`).

> **Alpha Feature** — not recommended for production use.

## Overview

Use this skill when a user asks to search historical footage or locate visual entities inside the recorded video database via natural language.

Key use cases:
- "Find all instances of forklifts"
- "When did someone enter the restricted area?"
- "Show me people near the loading dock"
- "Search for vehicles between 8am and noon"

## Prerequisites

- **Search Profile Deployed:** Ensure that the search profile has been started (`vss-sop-deploy` with `-p search`).
- **Embedding Ingestion Active:** Video uploads/streams must be active so the `rtvi-embed` service can write vectors to Elasticsearch.

## Instructions

### 1. Ingestion & Indexing

Videos uploaded or streamed via VIOS are processed by the `rtvi-embed` service. Cosmos Embed1 is used to generate 1024-dimension embeddings, which are automatically indexed into Elasticsearch via Kafka. No manual indexing steps are required.

### 2. Formulating Natural Language Queries

Users query using visual concepts (objects, colors, actions). The queries are mapped to similarity vectors to search the Elasticsearch index.

### 3. Execution via VSS Agent

Execute the natural-language search query by passing it to the VSS Agent's `/generate` endpoint.

## Examples

### Search via VSS Agent API

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_message": "find all instances of forklifts"}' | jq .
```

### Visual Description Search

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_message": "find someone wearing a red jacket near entrance"}' | jq .
```

### Action-Based Search

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_message": "show me people running in the parking lot"}' | jq .
```

## Error Handling

### Agent API Connection Refused (port 8000)
If requests fail with connection refused:
1. Check the VSS Agent container status:
   ```bash
   docker ps -a --filter name=vss-agent
   ```
2. Restart the agent container if needed:
   ```bash
   docker compose -f deployments/compose.yml --profile bp_sop_2d restart vss-agent
   ```

### Search Results Empty / No Matches
If semantic search queries return empty lists:
1. Verify the `search` profile is deployed and `rtvi-embed` is running:
   ```bash
   docker ps --format '{{.Names}}' | grep rtvi-embed
   ```
2. Verify Elasticsearch indices exist and are populated:
   ```bash
   curl -s http://localhost:9200/_cat/indices?v
   ```

---

## License

Use of this skill is governed by the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en) and the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
