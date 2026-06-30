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

# Nvidia DeepStream SOP Documentation

## Introduction

The DeepStream-SOP project implements a highly optimized computer vision inference for temporal action detection and VLM-based action evaluation pipeline, designed for low-latency processing of both video files and live Basler camera streams. The system operates as a real-time, accelerated microservice that produces operational insights for SOP-focused industry applications.

## 🚀 Recommended: Build & Customize with the Agentic Skill

**The fastest way to stand up and tailor this service is the agentic workflow — not the manual steps in this guide.** The [**DeepStream SOP skill**](../../agentic/ds-sop-skills/README.md) drives an AI coding agent (Claude Code / Codex) to deliver an **equivalent low-latency SOP inference microservice** end-to-end: describe your SOP and it handles

- **Code generation** — scaffold the full microservice (FastAPI + DeepStream/Triton + VLM pipeline) from a prompt
- **Customization** — adapt actions / SOP logic, VLM prompts, models, and the SOP checker to your workflow
- **Bug fixing & debugging** — diagnose server hangs and reconcile chunk-result mismatches against this reference
- **Container build & deployment** — Docker image build and `docker compose` bring-up
- **Benchmarking** — file-input TTFC and chunk-to-chunk (C2C) latency, plus camera/live chunk end-to-end (E2E) latency

👉 **Start here:** [agentic/ds-sop-skills/README.md](../../agentic/ds-sop-skills/README.md)

> **This `sop-inference-bp` directory is the canonical reference implementation** the skill generates against, debugs, evaluates, and benchmarks against — it is not deprecated. Follow the manual steps below when you want the reference code directly.

## Table of Contents

- [System Architecture](#architecture)
- [API Schema](#api-schema)
- [Getting started](#getting-started)
  - [Prepare Docker Container and Deploy Environments](#prepare-docker-container-and-deploy-environments)
  - [Download Required Model Checkpoints](#download-required-model-checkpoints)
  - [Launch SOP Microservice](#launch-sop-microservice)
- [Kafka Messaging Consumer](#kafka-messaging-consumer)
- [API Tests](#api-tests)
  - [Run API endpoints unit tests](#run-api-endpoints-unit-tests)
  - [Run API tests for video stream & camera](#run-api-tests-for-video-file-rtsp-live-stream-and-basler-camera-inputs)
    - [Test 1: Video File](#test-1-video-file-base64-encoded)
    - [Test 2: RTSP Live Stream](#test-2-rtsp-live-stream)
    - [Test 3: Basler Camera Live Streaming](#test-3-basler-camera-live-streaming)
- [Performance Profiling](#performance-profiling)
  - [API Client Performance Measurement](#api-client-performance-measurement)
  - [[Optional] [Developer] Profiling: Run the performance tests for the pipeline](#optional-developer-profiling-run-the-performance-tests-for-the-pipeline)
- [3rdparty License](#3rdparty-license)
- [Citation](#citation)

## System Architecture

The DeepStream-SOP microservice architecture integrates multiple components to deliver real-time temporal action detection and VLM-based evaluation:

![DeepStream SOP Architecture](docs/deepstream-sop-architecture.png)

**Key Components:**

- **Input Sources**: Supports video files, RTSP streams, and Basler camera live feeds
- **DeepStream Pipeline**: GPU-accelerated video processing using NVIDIA DeepStream SDK
- **Temporal Action Detection**: Real-time action recognition with [DDM](https://github.com/MCG-NJU/DDM) inference via Nvidia Triton acceleration
- **VLM Inference Evaluation**: Vision Language Model integration for intelligent action assessment using [Cosmos Reason Models](https://huggingface.co/nvidia/Cosmos-Reason2-2B) via vllm acceleration
- **API Server**: OpenAI-compatible REST interface for stream management and status monitoring
- **Output & Messaging**: Kafka messaging for event distribution, optional alert sounds, and video encoding capabilities

The architecture is designed for low-latency, high-throughput processing with configurable GPU memory utilization and flexible deployment options.

## API Schema

The DeepStream-SOP microservice exposes a RESTful API following OpenAI-compatible conventions. The complete API specification is available in OpenAPI 3.1.0 format.

**API Documentation:**
- **OpenAPI Spec**: [`docs/openapi.json`](docs/openapi.json)
- **Swagger UI**: Once the service is running, access the interactive API documentation at `http://localhost:8300/openapi.json`

**Main API Endpoints:**

- **Chat Completions** (`/v1/chat/completions`): Submit video streams for temporal action detection and VLM evaluation
- **File Management** (`/v1/files`): Upload, list, and manage video files
- **Health Checks** (`/v1/live`, `/v1/ready`, `/v1/startup`): Monitor service health and readiness
- **Models** (`/v1/models`): List available models
- **Metadata** (`/v1/metadata`): Retrieve service version and configuration information
- **Metrics** (`/v1/metrics`): Access Prometheus-compatible metrics

The API supports multiple input types including video files (base64 encoded), RTSP streams, and live Basler camera feeds.


## Getting started

### Prepare Docker Container and Deploy Environments

- **Pull source code and compose for deployment**

```
SOP_REPO=https://github.com/NVIDIA/sop-monitoring-blueprints.git
git clone https://github.com/NVIDIA/sop-monitoring-blueprints.git sop-monitoring-blueprints
cd sop-monitoring-blueprints/microservices/sop-inference-bp
```

- **Download Basler Pylon SDK (Required)**

  The build requires the Basler Pylon SDK, which is subject to separate license terms. Before building:

  1. Visit the official Basler website: [Pylon SDK 25.10.2](https://www.baslerweb.com/en/downloads/software/1932603569/)
  2. Complete the required registration form and agree to Basler's license terms
  3. Download `pylon-25.10.2_linux-x86_64_setup.tar.gz`
  4. Place the downloaded file in the `binaries/` directory:
     ```bash
     mkdir -p binaries
     mv ~/Downloads/pylon-25.10.2_linux-x86_64_setup.tar.gz binaries/
     ```

  **Note**: The Pylon SDK is a commercial, license-gated download. By default, if
  the file is not present in `binaries/`, the Docker build **stops with
  instructions** rather than downloading it silently. To accept the Basler license
  terms and let the build fetch it from the Basler CDN automatically, build with
  `ALLOW_PYLON_CDN_DOWNLOAD=1` (see below). The `binaries/` directory itself is
  tracked (via `binaries/.gitkeep`) so the build's bind mount always resolves.

- **Build the container**

```bash
docker compose -f deploy/compose.yaml build

# Or, to accept the Basler license terms and auto-download the Pylon SDK when it
# is not already in binaries/:
ALLOW_PYLON_CDN_DOWNLOAD=1 docker compose -f deploy/compose.yaml build
```

- **Download Required Model Checkpoints**

  This DeepStream-SOP microservice requires users to download the VLM and temporal action detection models. For optimal accuracy, you must retrain/fine-tune the models, which can be done using the SOP Training Blueprint.

  - **VLM Model**: Use retrained models compatible with NVIDIA Cosmos Reason. `VLLM_MODEL_PATH` is **required** and has no default — point it at your fine-tuned checkpoint, either a local directory under `MODEL_ROOT_DIR` or a Hugging Face repo id. Startup fails fast with a clear message if it is unset.

  - **Temporal Action Detection Models**: Use retrained models compatible with DDM-Net. For more information, refer to the [DDM repository](https://github.com/MCG-NJU/DDM).

- **Configure deployment settings**

  Create and configure `deploy/.env` file:

```
# vim deploy/.env

# Specify the model folder on host, e.g. /opt/models
# Make sure all the testing models are placed in this folder
MODEL_ROOT_DIR="/opt/models"

# Required (no default): the VLM checkpoint. Either a local checkpoint folder
# under $MODEL_ROOT_DIR, or a Hugging Face repo id (e.g. "your-org/your-model").
# The service fails fast at startup if this is empty.
VLLM_MODEL_PATH="/opt/models/cosmos-reason1.1-7b/checkpoint"

# Specify DDM temporal action detection model path
# It must be under folder $MODEL_ROOT_DIR
DDM_MODEL_PATH="/opt/models/gbed_models/ddm/checkpoint.pth.tar"

# Optional: enable the TensorRT path for the Triton DDM backend. False (default)
# = PyTorch only. True = at Triton init the backend loads the engine at
# DDM_TRT_ENGINE_OUTPUT_PATH if present, otherwise builds one on the fly from
# DDM_MODEL_PATH (5–15 min depending on GPU). PyTorch is loaded only as a
# last-resort fallback if both load and build fail. See the "DDM TensorRT
# optimization" section below.
# DDM_TRT_OPTIMIZATION=false

# Optional: where the TRT engine is loaded from and built to. Default
# /tmp/trt_opt/ddm.engine is per-container and rebuilds on every restart;
# point at a path under $MODEL_ROOT_DIR (bind-mounted persistently) to
# survive restarts.
# DDM_TRT_ENGINE_OUTPUT_PATH=/opt/models/gbed_models/ddm/ddm.engine

# Optional: engine precision. Default fp32.
# Only consulted when a build actually fires. If a cached engine already
# exists at DDM_TRT_ENGINE_OUTPUT_PATH and its shape+batch match, it is
# reused as-is and this setting is ignored. Delete the engine file to
# force a precision change to take effect.
# DDM_TRT_PRECISION=fp32

# Optional: GPU memory (GB) given to TensorRT for engine optimization
# during an on-the-fly build. Default 4. Raise on roomy GPUs for
# faster/better-tuned builds; lower if init shares the GPU with other
# services. No effect when reusing a cached engine.
# DDM_TRT_BUILD_WORKSPACE_GB=4

# Specify DDM model input resolution, select from [512, 384, 224], default value: 224
DS_ACTION_IN_RESOLUTION=224

# Specify DDM temporal settings.
# FRAMES_PER_SIDE must match the DDM checkpoint's temporal context.
# SEQUENCE_BATCH controls runtime grouping, not the training batch size.
# SLIDING_WINDOWS_SIZE is derived as 2 * FRAMES_PER_SIDE + SEQUENCE_BATCH.
FRAMES_PER_SIDE=5
SEQUENCE_BATCH=8

# Specify DDM model resize interpolation method, select from [nearest, bilinear], default value: nearest
DS_ACTION_IN_RESIZE_METHOD=nearest

# Specify cache path for vllm. Note that this path should be writable for the user in the microservice.
# The default value of HOST_CACHE is $HOME/.cache/ds_sop, which might not be writable for the nvds_sop
# In this case, we can just remove the HOST_CACHE volumes mount in compose.yaml
# HOST_CACHE=/path/to/writable/by/nvds_sop

# Specify the video subsample framerate for vllm input
VLM_FPS=8.0

# VLM video preprocessing parameters. These control how video is sampled and
# resized and MUST match the values used when the model was trained — otherwise
# the model falls back to its own implicit defaults and accuracy can drop
# significantly. Your training config may use only a subset; set the subset it
# used (at least one). If all are left at 0, the service logs a loud warning.
# The values below are an example for a 16:9 training setup — replace with yours.
# VLM_MAX_PIXELS=81920
# VLM_MAX_FRAMES=40
# VLM_MAX_TOTAL_PIXELS=12688256
# VLM_RESIZED_HEIGHT=567
# VLM_RESIZED_WIDTH=1008

# Specify whether to messaging chunk metadata through Kafka, disabled by default
#ENABLE_MESSAGING=1
# Optional: Kafka broker address. Defaults to localhost:9092, which targets the
# bundled single-host kafka service in deploy/compose.yaml. For an external or
# distributed broker, set this explicitly.
#KAFKA_BROKER=my-broker.example.com:9092
# Optional: Kafka topic for published chunk metadata. Default: mdx-vlm-captions
#DEFAULT_TOPIC=mdx-vlm-captions
# Optional: messaging schema, "JSON" (default) or "NvProtoSchema"
#SOP_MESSAGING_SCHEMA=JSON

# Optional: enable RTSP streaming output (only effective if the microservice was
# built with the optional RTSP feature). Disabled by default.
#ENABLE_RTSP_OUTPUT=true
#RTSP_PORT=8554

# Specify whether to sound alert when a chunk is ready, disabled by default
# Users need to specify ALERT_SOUND_FILE from host
#ENABLE_ALERT_SOUND=1
# Specify a host wav file which will be mount into container's
# alert file path: /opt/nvidia/nvds_sop/stream/alert.wav
#ALERT_SOUND_FILE="/host/system/alert.wav"

# Optional: specify the default action config path on the host. The file
# will be mounted into container's $ACTION_CONFIG_PATH, The file must be JSON format.
# Check configs/actions.json for example
#ACTION_CONFIG_PATH=/host/path/to/actions/config.json


# Optional: specify the default VLM prompts path on the host. The file
# will be mounted into container's $VLM_PROMPT_PATH
# Check configs/vlm_prompts.txt for example
#VLM_PROMPT_PATH=/host/path/to/configs/vlm_prompts.txt

# Optional: specify the default CAMERA format for Basler devices. default value: RGB
# supported format up to the camera [RGB, YUY2, UYVY]
CAMERA_FORMAT=RGB

# Optional: default Basler camera caps used when request payload omits them.
# Leave unset to use runtime defaults: width=1280, height=720, FPS unset.
#CAMERA_WIDTH=1280
#CAMERA_HEIGHT=720
#CAMERA_FPS_NUM=30
#CAMERA_FPS_DEN=1

# Optional: Basler pylon camera emulation. CAMERA_EMULATION_DIR should point
# to the host directory containing PNG frames. It is mounted to
# /opt/nvidia/nvds_sop/streams/simulation so configs/Emulation_0815-0000.pfs
# ImageFilename ./streams/simulation resolves inside the container.
# PYLON_CAMEMU defaults to 1 in deploy/compose.yaml. CAMERA_NUM_BUFFERS is
# unset by default; set it only when you want pylonsrc to emit EOS after N frames
# (for example, set it to the number of PNGs in CAMERA_EMULATION_DIR).
#PYLON_CAMEMU=1
#CAMERA_NUM_BUFFERS=<the number of PNGs>
#CAMERA_EMULATION_DIR=/host/path/to/streams/simulation

# Optional: specify GPU memory for the KV cache for performance
# VLLM_GPU_MEMORY_UTILIZATION=0.6

# Optional: specify max vllm request concurrency  for performance
# VLLM_MAX_NUM_SEQS=8

# Optional: specify the maximum total token sequence length of vlm
# VLLM_MAX_MODEL_LEN=50000

# Optional: Enable encoding and saving chunk files. Disabled by default
# ENCODE_VIDEO=0
# Optional: Specify host folder for file chunks, it will be mounted into
# container's /opt/nvidia/nvds_sop/chunks folder for chunks storage
# If specified, make sure any users have write/delete permission
# ENCODE_VIDEO_OUTPUT_DIR=/host/chunk/folder"

# Optional: Specify the host folder, it will be mounted into
# container's /tmp/nvds_sop_storage for file management.
# If specified, make sure any users have write/delete permission
# MEDIA_STORAGE_DIR=/host/media/folder

# Recommended on first run: run the container as your host user so it can write
# to bind-mounted host directories (media storage, chunks, model cache). The
# container defaults to uid/gid 1001; if that differs from the host user that
# owns the mounted directories, startup fails with a "No write permission"
# error. Set these to your host user to avoid that:
#   USER_ID=$(id -u)
#   GROUP_ID=$(id -g)
# (Use USER_ID=0 / GROUP_ID=0 to run as root for debugging only.)
# USER_ID=1001
# GROUP_ID=1001

# Optional: for debug purpose only
# WORK_DIR_PATH="/opt/nvidia/nvds_sop"

# Optional: for debug purpose only
# PYTHONPATH="/opt/nvidia/nvds_sop"

# Optional: for debug purpose only
# API_DUMMY_TEST=0

```

### (Optional) DDM TensorRT optimization

The Triton DDM backend can run the inner forward through TensorRT instead of
PyTorch. Enable it by setting `DDM_TRT_OPTIMIZATION=true` in `deploy/.env`;
the engine is loaded (or built on the fly) at Triton init. PyTorch and TRT
are never loaded together — exactly one runtime is chosen per process.

**Init workflow when `DDM_TRT_OPTIMIZATION=true`:**

1. Load the engine from `DDM_TRT_ENGINE_OUTPUT_PATH` if the file exists. If
   the engine's batch size and input dims match `SEQUENCE_BATCH`,
   `FRAMES_PER_SIDE`, and `DS_ACTION_IN_RESOLUTION`, TRT is ready and init
   finishes in seconds.
2. Otherwise (missing file, shape mismatch, or load failure), the backend
   builds an engine on the fly by running the export pipeline against
   `DDM_MODEL_PATH`. Build time is typically 5–15 minutes depending on GPU;
   the engine is written atomically to `DDM_TRT_ENGINE_OUTPUT_PATH` and then
   loaded.
3. If both load and build fail (e.g. the output path is not writable, or the
   checkpoint is missing), the backend falls back to PyTorch so the service
   still comes up.

**Persistence:** the default
`DDM_TRT_ENGINE_OUTPUT_PATH=/tmp/trt_opt/ddm.engine` is per-container and
rebuilds on every restart. For production, set the path to somewhere under
`${MODEL_ROOT_DIR}` (bind-mounted at the same path inside the container) so
the engine survives restarts:

```
DDM_TRT_ENGINE_OUTPUT_PATH=/opt/models/gbed_models/ddm/ddm.engine
```

**Cached engine wins — when a rebuild does (and doesn't) fire:** if the file
at `DDM_TRT_ENGINE_OUTPUT_PATH` exists and its embedded input shape and
batch size match the current env, the cached engine is reused as-is. The
backend re-checks:

- frames per window (from `FRAMES_PER_SIDE`)
- spatial resolution (from `DS_ACTION_IN_RESOLUTION`)
- batch dimension (from `SEQUENCE_BATCH`)

Changes to any of those three trigger an automatic rebuild on next start.
**`DDM_TRT_PRECISION` is NOT checked** — the engine doesn't expose its
built-in precision through a stable API, so switching `fp16` ↔ `fp32` ↔
`bf16` silently keeps using the existing engine. To actually switch
precision (or force a rebuild for any other reason), delete the cached
engine file or point `DDM_TRT_ENGINE_OUTPUT_PATH` at a different path:

```
rm /opt/models/gbed_models/ddm/ddm.engine   # then restart the service
```

**Tuning knobs (all optional, all in `deploy/.env`):**

| Env var | Default | Notes |
|---|---|---|
| `DDM_TRT_OPTIMIZATION` | `false` | Master switch. |
| `DDM_TRT_ENGINE_OUTPUT_PATH` | `/tmp/trt_opt/ddm.engine` | Load + cache path. |
| `DDM_TRT_PRECISION` | `fp32` | `fp32` / `fp16` / `bf16`. `fp32` is the safe default; `fp16` is the verified faster path. |
| `DDM_TRT_BUILD_WORKSPACE_GB` | `4` | GPU memory (GB) given to TensorRT for engine optimization during an on-the-fly build. No effect when reusing a cached engine. |

The build also reads `DS_ACTION_IN_RESOLUTION`, `FRAMES_PER_SIDE`,
`SEQUENCE_BATCH` (= engine batch dim), and `DDM_MODEL_PATH` from the
standard SOP config.

**Watching the build:**

```
docker compose -f deploy/compose.yaml logs -f nvds-action-sop
```

Look for `Building DDM TRT engine on the fly` to confirm a build is in
progress, and `TRT engine ready` once it completes. `Created TRT
thread-local state for thread <id>` appears on the first inference per
worker thread.

### Launch SOP Microservice

- **Launch the microservice**

```
# Launch microservice
docker compose -f deploy/compose.yaml up -d
```

The microservice will launch 2 containers: `nvds-action-sop` and `kafka`.

- **Check microservice status**

```
# check the last 200 lines of logs
docker compose -f deploy/compose.yaml logs -f --tail=200 nvds-action-sop
```

When the server is started, you will see logs like
```
...
INFO:     Started server process [3469]
INFO:     Waiting for application startup.
2026-01-16 22:54:34,814 [INFO] [DS_ACTION_DETECTOR.__main__]: Application started
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8300 (Press CTRL+C to quit)
```


- **Shutdown the microservice**

```
# After all the tests, to shutdown the microservice
docker compose -f deploy/compose.yaml down
```

## Kafka Messaging Consumer

When `ENABLE_MESSAGING=1` is enabled in `deploy/.env`, the microservice will publish
chunk metadata to the Kafka server after each video chunk is processed. This allows
real-time monitoring and integration with downstream systems.

**Start the Kafka consumer to view messages:**
```bash
docker compose -f deploy/compose.yaml exec nvds-action-sop python3 -m nvds_action_detector.messager --consumer
```

During each `/v1/chat/completions` request, you will see chunk metadata with nvprotobuf schema:
- Chunk ID and timestamp
- Video segment information (start/end time)
- VLM results
- SOP Checker results

## API Tests

### Run API endpoints unit tests

API schema could be found in Swagger [docs/openapi.json](docs/openapi.json)

Start the unittest for all MS playbook compliance tests
```
docker compose -f deploy/compose.yaml exec nvds-action-sop bash -c "TEST_VIDEO_PATH=/path/to/video.mp4 python3 tests/test_api_endpoints.py"
```

The unittest cover the following endpoints:
- Health check endpoints:
  - `GET /v1/live` - Service liveness check
  - `GET /v1/startup` - Service startup status
  - `GET /v1/ready` - Service readiness check
- Model endpoints:
  - `GET /v1/models` - List available models
- Metadata endpoint:
  - `GET /v1/metadata` - Show service metadata and version info
- Metrics endpoint:
  - `GET /v1/metrics` - Prometheus metrics for monitoring
- File management endpoints:
  - `POST /v1/files` - Upload a file
  - `GET /v1/files` - List all files
  - `GET /v1/files/{file_id}/content` - Download file content
  - `DELETE /v1/files/{file_id}` - Delete a file
- Chat completion endpoint:
  - `POST /v1/chat/completions` - Process video with AI model (supports streaming)


### Run API tests for video stream & camera

This test suite covers video file, RTSP stream, and Basler camera inputs.

```bash
# Basic test with video file
TEST_VIDEO_PATH=/path/to/video.mp4 python3 tests/api_client_test.py

# If running in container
docker compose -f deploy/compose.yaml exec nvds-action-sop bash -c "TEST_VIDEO_PATH=/path/to/video.mp4 python3 tests/api_client_test.py"
```

#### Test 1: Video File
Uses `test_chat_completion_basic()` - sends video file as base64 encoded data.

**Payload example:**
```json
{
  "model": "ds_sop_model",
  "messages": [{
    "role": "user",
    "content": [{
      "type": "video_url",
      "video_url": {
        "url": "data:video/mp4;base64,<base64_encoded_video>"
      }
    }]
  }],
  "stream": false,
  "chunking_options": {
    "algorithm": "ddm-net",
    "threshold": 0.8,
    "min_length_sec": 1.0,
    "max_length_sec": 10.0
  }
}
```

#### Test 2: RTSP Live Stream
Uses `test_video_rtsp_live_streaming()` - processes continuous RTSP video stream.

**Setup RTSP stream with VLC:**
```bash
# video.mp4 must use H.264/H.265 codec
cvlc --loop video.mp4 ":sout=#gather:rtp{sdp=rtsp://:8554/file-stream}" \
    :network-caching=1500 :sout-all :sout-keep
```

**Environment variable:**
```bash
export TEST_RTSP_VIDEO_URL="rtsp://0.0.0.0:8554/file-stream"
```

**Payload example:**
```json
{
  "model": "ds_sop_model",
  "messages": [{
    "role": "user",
    "content": [{
      "type": "video_url",
      "video_url": {
        "url": "rtsp://0.0.0.0:8554/file-stream"
      }
    }]
  }],
  "stream": true,
  "chunking_options": {
    "algorithm": "ddm-net",
    "threshold": 0.8,
    "min_length_sec": 1.0,
    "max_length_sec": 2.0
  }
}
```

#### Test 3: Basler Camera Live Streaming
Uses `test_physical_camera_live()` - processes live camera feed from Basler camera.

**Setup:**
- Install Pylon SDK 25.10.2 to get camera serial number via Pylon Viewer
- Find camera serial number (e.g., "40748152"), supported camera type: a2A2048-37gcPRO
- Optional: tune a Basler setting and save as `configs/Basler_camera_settings.pfs`, copy into the docker container.

**Environment variables:**
```bash
export PHYSICAL_CAMERA_ID="40748152" # camera serial number
export PHYSICAL_CAMERA_FORMAT="RGB"  # Options: RGB, UYVY, YUY2
```

**Payload example:**
```json
{
  "model": "ds_sop_model",
  "messages": [{
    "role": "user",
    "content": [{
      "type": "input_camera",
      "input_camera": {
        "camera_id": "40748152",
        "camera_vendor": "Basler",
        "camera_format": "RGB",
        "camera_width": 1280,
        "camera_height": 720,
        "camera_fps_num": 30,
        "camera_fps_den": 1
      }
    }]
  }],
  "stream": true,
  "chunking_options": {
    "algorithm": "ddm-net",
    "threshold": 0.8,
    "min_length_sec": 1.0,
    "max_length_sec": 2.0
  }
}
```

**Note:** For Basler cameras config file, add `"config": "configs/Basler_camera_settings.pfs"` to the `input_camera` object.

**Enable specific tests in code:**
Uncomment desired tests in `tests/api_client_test.py`:
```python
# test_instance.test_basler_camera_streaming_enumeration()
# test_instance.test_video_rtsp_live_streaming()
# test_instance.test_physical_camera_live(PHYSICAL_CAMERA_ID, "RGB", timeout_seconds=36)
```

## Performance Profiling

#### API Client Performance Measurement

For comprehensive performance testing of stream latency and throughput metrics using the `/v1/chat/completions` API endpoint with camera, RTSP, and file inputs, please refer to:

**[API Client Performance Test - Usage Guide](tests/README_perf.md)**

This guide provides detailed instructions for:
- Running performance tests with different stream types (camera/RTSP/file)
- Configuring environment variables and test parameters
- Understanding output metrics (stream startup time, chunk inference time, delays)
- Using the `StreamClient` class for automated performance measurement

### [Optional] [Developer] Profiling: Run the performance tests for the pipeline

- Running the pipeline for performance profiling
```
# update deploy/.env
vim deploy/.env

# update entrypoint to bash
ENTRYPOINT="/bin/bash"

# start container and run into terminal
docker compose -f deploy/compose.yaml up -d
docker compose -f deploy/compose.yaml attach nvds-action-sop

# make sure you are in the folder of nvds_action_detector
# check the model exist
ls $DDM_MODEL_PATH
ls $VLLM_MODEL_PATH

# start the benchmark test for E2E latency and throughput without API
# Disable sop checker for performance tests
DISABLE_SOP_CHECKER=1 python3 -m nvds_action_detector.ds_sop_process --video-path /path/to/test_video_whole_sop_h264.mp4 --batch-size 1

```

- Batch Size 1 is for single-stream, 8/16 for large concurrency.

```

# Run batch-size 8 test
DISABLE_SOP_CHECKER=1 python3 -m nvds_action_detector.ds_sop_process --video-path test_video_whole_sop_h264.mp4 --batch-size 8
```

## 3rdparty License
- Refer to `docker/Docker.build` for a complete list of third-party dependencies included in this project.
- This project will download and install additional third-party open source software projects. Review the
license terms of these open source projects before use.
- Building the final container from `docker/Docker.build` requires Basler Pylon SDK, which is subject to separate license terms. Users must independently download and accept the [Pylon SDK license terms](https://docs.baslerweb.com/pylonapi/cpp/licensing) before proceeding. The [Pylon-SDK-25.10](https://www.baslerweb.com/en/downloads/software/1932603569/) can be obtained from the official Basler website after completing the required registration form.

## Citation

This project utilizes [DDM-Net](https://github.com/MCG-NJU/DDM) for temporal action detection. If you use this DeepStream-SOP system in your research, please acknowledge the DDM-Net contribution by citing:

```bibtex
@InProceedings{Tang_2022_CVPR,
    author    = {Tang, Jiaqi and Liu, Zhaoyang and Qian, Chen and Wu, Wayne and Wang, Limin},
    title     = {Progressive Attention on Multi-Level Dense Difference Maps for Generic Event Boundary Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {3355-3364}
}
```
