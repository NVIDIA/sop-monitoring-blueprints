<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
-->

# API Client Performance Test - Usage Guide

## Table of Contents

- [Overview](#overview)
- [Running the Tests](#running-the-tests)
- [Environment Variables](#environment-variables)
- [Usage Examples](#usage-examples)
- [Output Files](#output-files)
- [Chunk Data Format](#chunk-data-format)
- [Key Components](#key-components)

## Overview
The `api_client_perf.py` file implements performance testing for the NVDS Action Detector API server, focusing on stream latency and throughput metrics.

## Running the Tests

### Prerequisites
API server must be started with the following environment variables:
```bash
export ENCODE_VIDEO=0
export ENABLE_ALERT_SOUND=0
export ENABLE_MESSAGING=0
export DISABLE_SOP_CHECKER=0
export ENABLE_PROFILING=1 # Profiling must be enabled, otherwise all perf tests failed
./start_server.sh
```

### Camera Stream Test
```bash
export TEST_STREAM_MODE="camera"
export PHYSICAL_CAMERA_ID="40748152"  # Camera serial number(s)
export STREAM_TIMEOUT_SECONDS=50.0
export STREAM_MAX_CHUNKS=10
export STREAM_ITERATIONS=3
export CHUNK_MAX_LENGTH_SEC=2.0

python tests/api_client_perf.py
```

### RTSP Stream Test
```bash
export TEST_STREAM_MODE="rtsp"
export TEST_RTSP_VIDEO_URL="rtsp://0.0.0.0:8554/file-stream"
export STREAM_TIMEOUT_SECONDS=50.0
export STREAM_MAX_CHUNKS=10
export STREAM_ITERATIONS=3
export CHUNK_MAX_LENGTH_SEC=2.0

# need skip 1st chunk for RTSP jitter issues
# the skip will also cause a bit higher stream_startup_time
SKIP_FIRST_CHUNK=1 python tests/api_client_perf.py
```

### File Stream Test
```bash
export TEST_STREAM_MODE="file"
export TEST_VIDEO_PATH="path/to/test_video_whole_sop_h264.mp4"
export STREAM_MAX_CHUNKS=10
export STREAM_ITERATIONS=2
export CHUNK_MAX_LENGTH_SEC=2.0

python tests/api_client_perf.py
```

### With Custom Environment
```bash
BASE_URL=http://localhost:8300 python tests/api_client_perf.py
```

## Environment Variables

### API Server Settings
- `BASE_URL`: API server URL (default: http://localhost:8300)

### Test Configuration
- `TEST_STREAM_MODE`: Input source type - "camera", "rtsp", or "file" (default: "camera")
- `STREAM_TIMEOUT_SECONDS`: Maximum time to wait for streaming in seconds (default: -1, disabled)
- `STREAM_MAX_CHUNKS`: Maximum number of chunks to process (default: 10)
- `STREAM_ITERATIONS`: Number of test iterations for single stream test (default: 3)
- `CONVERT_CSV_TO_STREAMING_TIME`: Convert detailed CSV times to streaming time (default: true)

### Camera Settings (when TEST_STREAM_MODE="camera")
- `PHYSICAL_CAMERA_ID`: Camera serial number(s), comma-separated for multiple cameras (default: "40748152,40748151")
- `CAMERA_WIDTH`: Camera width in pixels (default: 1280)
- `CAMERA_HEIGHT`: Camera height in pixels (default: 720)
- `CAMERA_FORMAT`: Camera format (default: "RGB")
- `CAMERA_FPS_NUM`: Camera FPS numerator (optional)
- `CAMERA_FPS_DEN`: Camera FPS denominator (optional)
- `PYLON_CAMEMU`: Enable Basler pylon camera emulation (compose default: 1)
- `CAMERA_EMULATION_DIR`: Host directory containing PNG frames; mounted to `/opt/nvidia/nvds_sop/streams/simulation`
- `CAMERA_NUM_BUFFERS`: Optional `pylonsrc` frame limit for emulation. Unset by default; set to the PNG count when you want EOS after one pass through the image set.

### RTSP Settings (when TEST_STREAM_MODE="rtsp")
- `TEST_RTSP_VIDEO_URL`: RTSP stream URL (default: "rtsp://0.0.0.0:8554/file-stream")

### File Settings (when TEST_STREAM_MODE="file")
- `TEST_VIDEO_PATH`: Path to video file (default: "test_video_whole_sop_h264.mp4")

### Chunking Settings
- `CHUNK_MAX_LENGTH_SEC`: Maximum chunk length in seconds (default: 2.0)
- `CHUNK_BOUNDARY_THRESHOLD`: Threshold for chunk boundary detection (default: 1.1, fallback to CHUNK_MAX_LENGTH_SEC)

## Usage Examples

### Single Stream Test - Camera Input
```python
from api_client_perf import StreamClient

client = StreamClient()

# Test with camera input
content = [{
    "type": "input_camera",
    "input_camera": {
        "camera_id": "40748152",
        "camera_vendor": "Basler",
        "camera_format": "RGB",
        "camera_width": 1280,
        "camera_height": 720,
        "camera_fps_num": 30,
        "camera_fps_den": 1,
    },
}]

metrics = client.test_single_stream(
    content=content,
    timeout_seconds=50.0,
    max_chunks=10,
    iterations=3,
    output_csv="single_stream_perf.csv",
)
```

### Single Stream Test - RTSP Input
```python
# Test with RTSP stream
rtsp_content = [{
    "type": "video_url",
    "video_url": {
        "url": "rtsp://0.0.0.0:8554/file-stream",
    },
}]

metrics = client.test_single_stream(
    content=rtsp_content,
    timeout_seconds=50.0,
    max_chunks=10,
    iterations=2,
    output_csv="rtsp_stream_perf.csv",
)
```

### Single Stream Test - File Input
```python
import base64

# Test with base64-encoded video file
with open("path/to/test_video.mp4", "rb") as f:
    video_base64 = base64.b64encode(f.read()).decode("utf-8")

file_content = [{
    "type": "video_url",
    "video_url": {
        "url": f"data:video/mp4;base64,{video_base64}",
    },
}]

metrics = client.test_single_stream(
    content=file_content,
    max_chunks=10,
    iterations=2,
    output_csv="file_stream_perf.csv",
)
```

## Output Files

### Summary CSV (`*_metrics.csv`)
Columns:
- stream_id
- total_chunks
- stream_startup_time
- first_chunk_inference_time
- average_chunk_inference_time
- average_chunk_delay
- total_duration

### Detailed CSV (`*_metrics_detailed.csv`)
Columns:
- stream_id
- chunk_idx
- chunk_boundary_time
- chunk_inference_timestamp
- chunk_inference_delay



## Chunk Data Format
Each chunk from the API contains metadata in the format:
```json
{
  "choices": [{
    "delta": {
      "chunk_metadata": {
        "start_time": 0.228150258,
        "end_time": 2.228549936,
        "first_timestamp": 1768948480.7441804,
        "pipeline_starting_timestamp": 1768948474.4552605,
        "pipeline_vlm_ready_timestamp": 1768948483.5438457,
        "frame_number": 68,
        "req_id": "...",
        "response": "..."
      }
    }
  }]
}
```

All timing calculations are based on these metadata fields to provide accurate latency measurements.


## Key Components

### StreamClient Class
Main class for testing `/v1/chat/completions` streaming requests with performance metrics collection.

### StreamMetrics Dataclass
Container for all performance metrics:
- `stream_startup_time`: Time from pipeline start to first chunk's first_timestamp
- `first_chunk_duration`: First chunk's end_time - first chunk's start_time
- `first_chunk_inference_time`: Time from first_timestamp to VLM ready for first chunk
- `average_chunk_inference_time`: Average chunk inference time across all chunks
- `average_chunk_delay`: Average delay from chunking to VLM ready time across all chunks
- `total_duration`: Total test duration
- `chunk_boundary_times`: List of chunking end times relative to first chunk start
- `chunk_inference_timestamps`: List of VLM ready times relative to first chunk's first_timestamp
- `chunk_inference_delays`: List of delays (VLM ready - chunk end) for each chunk. [live streaming only]

### Key Methods

#### `_test_stream_request()`
Private function that:
- Sends streaming request with `"stream": True`
- Processes response chunks in real-time
- Extracts timing metadata from each chunk
- Calculates all latency metrics as specified:
  1. **stream_startup_time** = 1st chunk's `first_timestamp` - `pipeline_starting_timestamp`
  2. **first_chunk_duration** = 1st chunk's `end_time` - 1st chunk's `start_time`
  3. **first_chunk_inference_time** = 1st chunk's `pipeline_vlm_ready_timestamp` - `first_timestamp`
  4. **average_chunk_inference_time** = (last chunk's `pipeline_vlm_ready_timestamp` - 1st chunk's `first_timestamp`) / len(chunks)
  5. **average_chunk_delay** = average of (current chunk's `pipeline_vlm_ready_timestamp` - current chunk's `end_time`)
  6. **chunk_boundary_times** = list of (current chunk's `end_time` - 1st chunk's `start_time`)
  7. **chunk_inference_timestamps** = list of (current chunk's `pipeline_vlm_ready_timestamp` - 1st chunk's `first_timestamp`)
  8. **chunk_inference_delays** = list of (current chunk's `pipeline_vlm_ready_timestamp` - current chunk's `end_time`)

#### `test_single_stream()`
Tests a single stream multiple times (iterations) and outputs metrics to CSV files:
- Main CSV: Summary metrics per stream iteration
- Detailed CSV: Per-chunk timing data
- Supports `max_chunks` parameter to limit number of chunks processed
- Supports `timeout_seconds` parameter to limit stream duration

#### `test_multiple_streams()`
Tests multiple streams (sequential or concurrent) and provides:
- Individual stream metrics
- Aggregate statistics
- Overall throughput calculations
