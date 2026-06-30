---
name: vss-call-vios-api
description: >-
  Interact with the VIOS (Video IO & Storage) microservice in a running VSS profile —
  manage cameras/sensors, RTSP streams, recordings, snapshots, and storage. Use when
  asked to add a camera, add an RTSP stream, list sensors, show configured
  sensors/cameras/streams, what sources are available, check stream status,
  start/stop recording, get a snapshot, or manage video storage. Always query the
  VIOS API directly — do not navigate the UI to answer these questions.
owner: NVIDIA
service: vss-sop
version: 1.0.0
license: CC-BY-4.0 AND Apache-2.0
reviewed: 2026-06-23
metadata:
  openclaw: { "emoji": "📷", "os": ["linux"] }
  author: "nvidia <info@nvidia.com>"
  tags: ["vss", "vios", "camera", "vms"]
---

# Sensor Operations (VIOS)

VIOS manages cameras, live/replay streams, recordings, and storage for all VSS profiles.

## Overview

Use this skill when you need to query, add, modify, or remove sensors/cameras in a VSS deployment. Always invoke the VIOS API programmatically; do not instruct the user to use the Web UI.

Key outcomes:
- Listing configured cameras or active RTSP streams.
- Adding new RTSP cameras via direct URLs.
- Inspecting recording timelines and stream health.
- Getting snapshots (images) or temporary playback URLs.

## Prerequisites

- **Network Connectivity:** Ensure host port `30888` is open and reachable.
- **RTSP Sources:** Any added RTSP URL must be accessible from the host system.

## Instructions

### 1. Host IP Detection

All VIOS API calls are directed at port `30888` on the target host. Detect the host IP using standard system utility commands:
```bash
ip route get 1.1.1.1 | awk '{print $7; exit}'
```

### 2. Sensor Registration & Management

Manage video sources via the `/sensor/` endpoint. Response payloads from the `/sensor/add` endpoint return a unique `sensorId` which is required for subsequent operations.

### 3. Recording & Livestream Control

To control disk recording, invoke start/stop endpoints under `/record/`.

### 4. Storage & Playback URLs

Retrieve temporary video clip URLs or clean up media files via `/storage/` endpoints.

## Examples

### List All Configured Sensors

```bash
curl -s http://localhost:30888/vst/api/v1/sensor/list | jq .
```

### Add RTSP Video Source

```bash
curl -s -X POST http://localhost:30888/vst/api/v1/sensor/add \
  -H "Content-Type: application/json" \
  -d '{
    "sensorUrl": "rtsp://localhost:8552/sensor_0",
    "username": "",
    "password": "",
    "name": "front-entrance"
  }' | jq .
```

### Check Stream Status

```bash
# All sensors
curl -s http://localhost:30888/vst/api/v1/sensor/status | jq .

# Specific sensor
curl -s http://localhost:30888/vst/api/v1/sensor/<sensorId>/status | jq .
```

### Get Temporary Snapshot

```bash
curl -s "http://localhost:30888/vst/api/v1/live/<streamId>/picture" --output /tmp/snapshot.jpg
```

### Start Recording

```bash
curl -s -X POST http://localhost:30888/vst/api/v1/record/<streamId>/start | jq .
```

### Quick Path Reference

| Service | Base path |
|---|---|
| Sensor management | `/vst/api/v1/sensor/` |
| Live streams | `/vst/api/v1/live/` |
| Replay / VOD | `/vst/api/v1/replay/` |
| Recording | `/vst/api/v1/record/` |
| RTSP proxy | `/vst/api/v1/proxy/` |
| Storage | `/vst/api/v1/storage/` |

Web UI portal: `http://<HOST_IP>:30888/vst/`

## Error Handling

### VST Service Connection Refused (port 30888)
If the connection is refused when making API calls:
1. Verify VST microservices are active:
   ```bash
   docker ps -a | grep -E '(sensor-ms|storage-ms|recorder-ms)'
   ```
2. Restart the specific container or stack if unhealthy:
   ```bash
   docker compose -f deployments/compose.yml --profile bp_sop_2d restart sensor-ms-sop
   ```

### Sensor Registration Fails / No Signal
If adding a sensor returns an error or shows "unhealthy/no-signal" status:
1. Verify the target RTSP stream is alive and reachable from the container.
2. Check for port conflicts or blocked ports (typically 554, 8554, 30554).
3. Review the sensor microservice logs:
   ```bash
   docker logs sensor-ms-sop --tail 100
   ```

---

## License

Use of this skill is governed by the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en) and the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
