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

"""
Performance test for chat completion streaming endpoint in api_server.py

This test suite focuses on:
- Single stream latency metrics
- Multiple concurrent stream throughput
- Detailed chunk-level timing analysis

Tests required API server start with following environment variables:
export ENCODE_VIDEO=0
export ENABLE_ALERT_SOUND=0
export ENABLE_MESSAGING=0
export DISABLE_SOP_CHECKER=0
export ENABLE_PROFILING=1
./start_server.sh

### Single Stream Test
- To start api_client_perf for single stream test
```bash
export TEST_STREAM_MODE="camera"
export PHYSICAL_CAMERA_ID="40748152" # camera serial numbers
export STREAM_TIMEOUT_SECONDS=50.0
export STREAM_MAX_CHUNKS=10
export STREAM_ITERATIONS=3
export CHUNK_MAX_LENGTH_SEC=2.0

python tests/api_client_perf.py
```

- to test rtsp stream, set TEST_STREAM_MODE to "rtsp" and set TEST_RTSP_VIDEO_URL to the rtsp url
```bash
export TEST_STREAM_MODE="rtsp"
export TEST_RTSP_VIDEO_URL="rtsp://0.0.0.0:8554/file-stream"
export STREAM_TIMEOUT_SECONDS=50.0
export STREAM_MAX_CHUNKS=10
export STREAM_ITERATIONS=2
export CHUNK_MAX_LENGTH_SEC=2.0

python tests/api_client_perf.py

```

- to test file stream, set TEST_STREAM_MODE to "file" and set TEST_FILE_PATH to the file path
```bash
export TEST_STREAM_MODE="file"
export TEST_FILE_PATH="path/to/test_video_whole_sop_h264.mp4"
#export STREAM_TIMEOUT_SECONDS=50.0
export STREAM_MAX_CHUNKS=10
export STREAM_ITERATIONS=2
export CHUNK_MAX_LENGTH_SEC=2.0

python tests/api_client_perf.py

```

"""
import argparse
import base64
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import requests


def print_log(msg: str):
    """Print with flush=True to ensure immediate output"""
    print(msg, flush=True)


# API server base URL
BASE_URL = os.getenv("BASE_URL", "http://localhost:8300")
MODEL_ID = "ds_sop_model"

TEST_STREAM_MODE = os.getenv("TEST_STREAM_MODE", "camera")
if TEST_STREAM_MODE not in ["camera", "rtsp", "file"]:
    raise ValueError(f"TEST_STREAM_MODE must be 'camera' or 'rtsp' or 'file', but got {TEST_STREAM_MODE}")
print_log(f"TEST_STREAM_MODE: {TEST_STREAM_MODE}")

TEST_VIDEO_PATH = os.getenv("TEST_VIDEO_PATH", "test_video_whole_sop_h264.mp4")


def _get_test_video_base64(video_path: str):
    try:
        with open(video_path, "rb") as f:
            encoded_base64 = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Test video file not found at {video_path}")
    return encoded_base64


STREAM_MAX_CHUNKS = int(os.getenv("STREAM_MAX_CHUNKS", "10"))
STREAM_TIMEOUT_SECONDS = float(os.getenv("STREAM_TIMEOUT_SECONDS", "-1"))
STREAM_ITERATIONS = int(os.getenv("STREAM_ITERATIONS", "3"))
CONVERT_CSV_TO_STREAMING_TIME = os.getenv("CONVERT_CSV_TO_STREAMING_TIME", "true").lower() in ["true", "1"]
print_log(f"CSV convert to streaming time: {CONVERT_CSV_TO_STREAMING_TIME}")

CHUNK_MAX_LENGTH_SEC = float(os.getenv("CHUNK_MAX_LENGTH_SEC", "2.0"))
# set threshold >1.0 to make the chunk always use the max length
# this is for performance measurement only, the value >1.1 could not be used in production
CHUNK_BOUNDARY_THRESHOLD = float(os.getenv("CHUNK_BOUNDARY_THRESHOLD", "1.1"))
TEST_RTSP_VIDEO_URL = os.getenv("TEST_RTSP_VIDEO_URL", "rtsp://0.0.0.0:8554/file-stream")
SKIP_FIRST_CHUNK = os.getenv("SKIP_FIRST_CHUNK", "false").lower() in ["true", "1"]

# Basler camera ID(s)
# Multiple cameras: comma-separated list (e.g., "40748152,40748153,40748154")
PHYSICAL_CAMERA_IDS_STR = os.getenv("PHYSICAL_CAMERA_ID", "40748152,40748151")
PHYSICAL_CAMERA_IDS = (
    [cid.strip() for cid in PHYSICAL_CAMERA_IDS_STR.split(",") if cid.strip()] if PHYSICAL_CAMERA_IDS_STR else None
)
if PHYSICAL_CAMERA_IDS is None or len(PHYSICAL_CAMERA_IDS) == 0:
    raise ValueError("PHYSICAL_CAMERA_ID is not set")
if TEST_STREAM_MODE == "camera":
    print_log(f"PHYSICAL_CAMERA_IDS list: {PHYSICAL_CAMERA_IDS}")

CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
CAMERA_FORMAT = os.getenv("CAMERA_FORMAT", "RGB")
CAMERA_FPS_NUM = int(os.getenv("CAMERA_FPS_NUM")) if os.getenv("CAMERA_FPS_NUM") is not None else None
CAMERA_FPS_DEN = int(os.getenv("CAMERA_FPS_DEN")) if os.getenv("CAMERA_FPS_DEN") is not None else None


@dataclass
class StreamMetrics:
    """Container for stream performance metrics"""

    stream_id: str
    total_chunks: int
    stream_startup_time: float  # 1st chunk's first_timestamp - pipeline_starting_timestamp
    first_chunk_duration: float  # 1st chunk's end_time - 1st chunk's start_time
    first_chunk_inference_time: float  # 1st chunk's pipeline_vlm_ready_timestamp - first_timestamp
    average_chunk_inference_time: float  # (last chunk's vlm_ready - 1st chunk's first_timestamp) / len(chunks)
    average_chunk_delay: float  # (sum of (current chunk's vlm_ready - current chunk's end_time)) / len(chunks)
    total_duration: float  # Total test duration
    chunk_boundary_times: List[float]  # List of (current chunk's end_time - 1st chunk's start_time)
    chunk_inference_timestamps: List[float]  # List of (current chunk's vlm_ready - 1st chunk's first_timestamp)
    chunk_inference_delays: List[float]  # List of (current chunk's vlm_ready - current chunk's end_time)
    cv_inference_times: List[float]  # List of each chunk's cv_execute_time
    vlm_inference_times: List[float]  # List of each chunk's vlm_execute_time
    chunk_e2e_latencies: List[float]  # List of (pipeline_vlm_ready - pipeline_chunk_end_timestamp) per chunk

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class StreamClient:
    """Client for testing /chat/completions streaming requests"""

    def __init__(self, base_url: str = BASE_URL, model_id: str = MODEL_ID):
        self.base_url = base_url
        self.model_id = model_id
        self.session = requests.Session()

    def _test_stream_request(
        self,
        content: List[Dict[str, Any]],
        timeout_seconds: float = -1.0,
        max_chunks: int = -1,
        chunking_options: Optional[Dict[str, Any]] = None,
        stream_id: Optional[str] = None,
    ) -> StreamMetrics:
        """
        Private function to test a single streaming request with focus on real-time response.

        Args:
            content: List of content items (e.g., video_url, input_camera)
            timeout_seconds: Maximum time to wait for streaming
            chunking_options: Optional chunking configuration
            stream_id: Optional identifier for this stream

        Returns:
            StreamMetrics object with calculated latency metrics
        """
        if chunking_options is None:
            chunking_options = {
                "algorithm": "ddm-net",
                "threshold": CHUNK_BOUNDARY_THRESHOLD,
                "min_length_sec": 1.0,
                "max_length_sec": CHUNK_MAX_LENGTH_SEC,
            }

        if stream_id is None:
            stream_id = f"stream-{int(time.time() * 1000)}"

        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "stream": True,
            "chunking_options": chunking_options,
        }

        headers = {"Content-Type": "application/json"}

        chunks = []
        start_time = time.time()

        # Metrics placeholders
        stream_startup_time = None
        first_chunk_inference_time = None
        average_chunk_inference_time = None
        chunk_boundary_times = []
        chunk_inference_timestamps = []
        cv_inference_times = []
        vlm_inference_times = []
        chunk_e2e_latencies = []
        # First chunk metadata
        first_chunk_start_time = None
        first_chunk_first_timestamp = None
        first_chunk_pipeline_starting_timestamp = None

        try:
            # Use None to disable timeout when timeout_seconds <= 0
            request_timeout = None if timeout_seconds <= 0 else (timeout_seconds, timeout_seconds * 2)
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
                timeout=request_timeout,
            )

            assert response.status_code == 200, (
                f"Status: {response.status_code}, "
                f"Body: {response.text[:500] if hasattr(response, 'text') else 'N/A'}"
            )

            content_type = response.headers.get("content-type", "")
            assert "text/event-stream" in content_type, f"Content-Type: {content_type}"

            # Read streaming response
            for line in response.iter_lines():
                elapsed_time = time.time() - start_time
                # Check timeout (only if timeout is enabled)
                if timeout_seconds > 0 and elapsed_time >= timeout_seconds:
                    print_log(f"[{stream_id}] Manually stopping after {elapsed_time:.2f}s (timeout)")
                    response.close()
                    break
                # Check max chunks (only if max_chunks is enabled)
                if max_chunks > 0 and len(chunk_boundary_times) >= max_chunks:
                    print_log(
                        f"[{stream_id}] Manually stopping after {len(chunk_boundary_times)} chunks (max_chunks: {max_chunks})"
                    )
                    response.close()
                    break

                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        if data_str == "[DONE]":
                            print_log(f"[{stream_id}] [DONE] received after {elapsed_time:.2f}s")
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            chunks.append(chunk_data)

                            # Extract chunk metadata
                            if len(chunk_data.get("choices", [])) > 0:
                                choice = chunk_data["choices"][0]
                                chunk_metadata = choice.get("delta", {}).get("chunk_metadata") or choice.get(
                                    "chunk_metadata", {}
                                )

                                if chunk_metadata:
                                    if SKIP_FIRST_CHUNK and len(chunks) == 1:
                                        print_log(f"[{stream_id}] Skipping first chunk")
                                        continue

                                    # Process first chunk
                                    if first_chunk_start_time is None:  # or len(chunks) == 1
                                        first_chunk_start_time = chunk_metadata.get("start_time", None)
                                        first_chunk_pipeline_starting_timestamp = chunk_metadata.get(
                                            "pipeline_starting_timestamp", None
                                        )
                                        if not SKIP_FIRST_CHUNK:
                                            first_chunk_first_timestamp = chunk_metadata.get("first_timestamp", None)
                                        else:
                                            first_chunk_first_timestamp = chunk_metadata.get(
                                                "pipeline_cv_ready_timestamp", None
                                            ) - chunk_metadata.get("cv_execute_time", None)

                                        if first_chunk_first_timestamp and first_chunk_pipeline_starting_timestamp:
                                            stream_startup_time = (
                                                first_chunk_first_timestamp - first_chunk_pipeline_starting_timestamp
                                            )
                                            print_log(f"[{stream_id}] Stream startup time: {stream_startup_time:.4f}s")

                                    # Calculate chunk boundary time (current chunk's end_time - 1st chunk's start_time)
                                    if first_chunk_start_time is not None and "cv_execute_time" in chunk_metadata:
                                        current_end_time = chunk_metadata.get("end_time", None)
                                        if current_end_time is not None:
                                            boundary_time = current_end_time - first_chunk_start_time
                                            chunk_boundary_times.append(boundary_time)

                                        # Calculate chunk inference timestamp (current chunk's vlm_ready - 1st chunk's first_timestamp)
                                        # if first_chunk_first_timestamp is not None:
                                        assert (
                                            first_chunk_first_timestamp is not None
                                        ), "first_chunk_first_timestamp is None"
                                        current_vlm_ready = chunk_metadata.get("pipeline_vlm_ready_timestamp", None)
                                        if current_vlm_ready is not None:
                                            inference_timestamp = current_vlm_ready - first_chunk_first_timestamp
                                            chunk_inference_timestamps.append(inference_timestamp)

                                        cv_execute_time = chunk_metadata.get("cv_execute_time", None)
                                        if cv_execute_time is not None:
                                            cv_inference_times.append(cv_execute_time)

                                        vlm_execute_time = chunk_metadata.get("vlm_execute_time", None)
                                        if vlm_execute_time is not None:
                                            vlm_inference_times.append(vlm_execute_time)

                                        # Camera E2E latency: wall-clock time from last decoded frame to VLM ready
                                        chunk_end_ts = chunk_metadata.get("pipeline_chunk_end_timestamp", None)
                                        if current_vlm_ready is not None and chunk_end_ts is not None:
                                            chunk_e2e_latencies.append(current_vlm_ready - chunk_end_ts)

                            # Print progress for first few chunks
                            if len(chunks) <= 3:
                                print_log(f"[{stream_id}] Chunk {len(chunks)}: {json.dumps(chunk_data)[:100]}...")

                        except json.JSONDecodeError as e:
                            print_log(f"[{stream_id}] Failed to parse chunk: {data_str[:100]} - {e}")

            response.close()

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            elapsed_time = time.time() - start_time
            if "Read timed out" in str(e):
                print_log(f"[{stream_id}] Read timeout after {elapsed_time:.2f}s")
            else:
                print_log(f"[{stream_id}] Connection error after {elapsed_time:.2f}s: {e}")
                raise
        assert len(chunk_inference_timestamps) == len(
            chunk_boundary_times
        ), f"chunk_inference_timestamps: {chunk_inference_timestamps}, chunk_boundary_times: {chunk_boundary_times}"
        total_chunks = len(chunk_inference_timestamps)
        chunk_inference_delays = [chunk_inference_timestamps[i] - chunk_boundary_times[i] for i in range(total_chunks)]
        # Calculate average chunk inference time
        # total_chunks should the the chunks with valid metadata
        if total_chunks > 0 and first_chunk_first_timestamp is not None:
            if len(chunk_inference_timestamps) > 0:
                # Use last chunk's vlm_ready timestamp
                last_chunk_vlm_ready = first_chunk_first_timestamp + chunk_inference_timestamps[-1]
                average_chunk_inference_time = (last_chunk_vlm_ready - first_chunk_first_timestamp) / total_chunks
            else:
                average_chunk_inference_time = 0.0
        else:
            average_chunk_inference_time = 0.0

        # total_duration = time.time() - start_time
        total_duration = chunk_boundary_times[-1] if len(chunk_boundary_times) > 0 else 0.0
        first_chunk_duration = chunk_boundary_times[0] if len(chunk_boundary_times) > 0 else 0.0
        first_chunk_inference_time = chunk_inference_timestamps[0] if len(chunk_inference_timestamps) > 0 else 0.0

        # Create metrics object
        metrics = StreamMetrics(
            stream_id=stream_id,
            total_chunks=total_chunks,
            stream_startup_time=stream_startup_time or 0.0,
            first_chunk_duration=first_chunk_duration,
            first_chunk_inference_time=first_chunk_inference_time or 0.0,
            average_chunk_inference_time=average_chunk_inference_time,
            average_chunk_delay=sum(chunk_inference_delays) / total_chunks,
            total_duration=total_duration,
            chunk_boundary_times=chunk_boundary_times,
            chunk_inference_timestamps=chunk_inference_timestamps,
            chunk_inference_delays=chunk_inference_delays,
            cv_inference_times=cv_inference_times,
            vlm_inference_times=vlm_inference_times,
            chunk_e2e_latencies=chunk_e2e_latencies,
        )

        print_log(f"[{stream_id}] Stream completed:")
        print_log(f"  - Total chunks: {metrics.total_chunks}")
        print_log(f"  - Stream startup time: {metrics.stream_startup_time:.4f}s")
        print_log(f"  - First chunk duration: {metrics.first_chunk_duration:.4f}s")
        print_log(f"  - First chunk inference time: {metrics.first_chunk_inference_time:.4f}s")
        print_log(f"  - Average chunk inference time: {metrics.average_chunk_inference_time:.4f}s")
        print_log(f"  - Average chunk stream latency: {metrics.average_chunk_delay:.4f}s")
        print_log(f"  - Total duration: {metrics.total_duration:.2f}s")
        print_log(f"  - Chunk boundary times: [{', '.join(f'{t:.4f}' for t in metrics.chunk_boundary_times)}]")
        print_log(
            f"  - Chunk inference timestamps: [{', '.join(f'{t:.4f}' for t in metrics.chunk_inference_timestamps)}]"
        )
        print_log(f"  - Chunk inference delays: [{', '.join(f'{t:.4f}' for t in metrics.chunk_inference_delays)}]")
        print_log(f"  - CV inference times: [{', '.join(f'{t:.4f}' for t in metrics.cv_inference_times)}]")
        print_log(f"  - VLM inference times: [{', '.join(f'{t:.4f}' for t in metrics.vlm_inference_times)}]")
        if metrics.chunk_e2e_latencies:
            avg_e2e = sum(metrics.chunk_e2e_latencies) / len(metrics.chunk_e2e_latencies)
            print_log(
                f"  - Chunk E2E latencies (vlm_ready - chunk_end): [{', '.join(f'{t:.4f}' for t in metrics.chunk_e2e_latencies)}]"
            )
            print_log(f"  - Average chunk E2E latency: {avg_e2e:.4f}s")
        return metrics

    def test_single_stream(
        self,
        content: List[Dict[str, Any]],
        timeout_seconds: float = STREAM_TIMEOUT_SECONDS,
        max_chunks: int = STREAM_MAX_CHUNKS,
        iterations: int = STREAM_ITERATIONS,
        chunking_options: Optional[Dict[str, Any]] = None,
        output_csv: str = "single_stream_metrics.csv",
    ) -> List[StreamMetrics]:
        """
        Test a single stream multiple times and save metrics to CSV.

        Args:
            content: List of content items for the request
            timeout_seconds: Maximum time for streaming
            chunking_options: Optional chunking configuration
            iterations: Number of times to run the test
            output_csv: Output CSV file path

        Returns:
            List of StreamMetrics objects for all iterations
        """
        print_log("=" * 80)
        print_log(f"SINGLE STREAM LATENCY TEST ({iterations} iteration(s))")
        print_log("=" * 80)

        all_metrics = []
        overall_start = time.time()

        for i in range(iterations):
            stream_id = f"single-stream-iter-{i+1}"
            print_log(f"\n--- Iteration {i+1}/{iterations} ---")

            metrics = self._test_stream_request(
                content=content,
                timeout_seconds=timeout_seconds,
                max_chunks=max_chunks,
                chunking_options=chunking_options,
                stream_id=stream_id,
            )
            all_metrics.append(metrics)

        overall_duration = time.time() - overall_start

        # Write all metrics to CSV
        self._write_metrics_to_csv(all_metrics, output_csv)
        print_log(f"\nMetrics saved to: {output_csv}")

        # Print summary with averages and list of all results
        self._print_iteration_summary(all_metrics, overall_duration)

        return all_metrics

    def _print_iteration_summary(self, all_metrics: List[StreamMetrics], overall_duration: float):
        """Print summary of all iteration results with averages."""
        if not all_metrics:
            print_log("No metrics to summarize")
            return

        print_log("\n" + "=" * 80)
        print_log("ITERATION SUMMARY")
        print_log("=" * 80)

        # Print list of all test results
        print_log("\nAll Test Results:")
        print_log("-" * 180)
        print_log(
            f"{'Iteration':<22} "
            f"{'Startup(s)':<12} "
            f"{'1stChunkDur(s)':<16} "
            f"{'1stChunkInf(s)':<16} "
            f"{'1stChunkDly(s)':<16} "
            f"{'AvgChunkDur(s)':<16} "
            f"{'AvgChunkInf(s)':<16} "
            f"{'AvgDelay(s)':<12} "
            f"{'AvgCV(s)':<12} "
            f"{'AvgVLM(s)':<12} "
            f"{'Chunks':<8}"
        )
        print_log("-" * 180)

        for metrics in all_metrics:
            avg_chunk_duration = metrics.total_duration / metrics.total_chunks
            first_chunk_delay = metrics.first_chunk_inference_time - metrics.first_chunk_duration
            avg_cv = (
                sum(metrics.cv_inference_times) / len(metrics.cv_inference_times) if metrics.cv_inference_times else 0
            )
            avg_vlm = (
                sum(metrics.vlm_inference_times) / len(metrics.vlm_inference_times)
                if metrics.vlm_inference_times
                else 0
            )

            print_log(
                f"{metrics.stream_id:<22} "
                f"{metrics.stream_startup_time:<12.4f} "
                f"{metrics.first_chunk_duration:<16.4f} "
                f"{metrics.first_chunk_inference_time:<16.4f} "
                f"{first_chunk_delay:<16.4f} "
                f"{avg_chunk_duration:<16.4f} "
                f"{metrics.average_chunk_inference_time:<16.4f} "
                f"{metrics.average_chunk_delay:<12.4f} "
                f"{avg_cv:<12.4f} "
                f"{avg_vlm:<12.4f} "
                f"{metrics.total_chunks:<8}"
            )

        print_log("-" * 180)

        # Calculate and print averages
        num_iterations = len(all_metrics)
        avg_total_chunks = sum(m.total_chunks for m in all_metrics) / num_iterations
        avg_startup_time = sum(m.stream_startup_time for m in all_metrics) / num_iterations
        avg_first_chunk_duration = sum(m.first_chunk_duration for m in all_metrics) / num_iterations
        avg_first_chunk_inference_time = sum(m.first_chunk_inference_time for m in all_metrics) / num_iterations
        avg_chunk_inference_time = sum(m.average_chunk_inference_time for m in all_metrics) / num_iterations
        avg_chunk_delay = sum(m.average_chunk_delay for m in all_metrics) / num_iterations
        avg_chunk_duration = sum(m.total_duration / m.total_chunks for m in all_metrics) / num_iterations
        avg_first_chunk_delay = (
            sum(m.first_chunk_inference_time - m.first_chunk_duration for m in all_metrics) / num_iterations
        )
        # Calculate average CV and VLM inference times
        avg_cv_inference_time = (
            sum(
                sum(m.cv_inference_times) / len(m.cv_inference_times) if m.cv_inference_times else 0
                for m in all_metrics
            )
            / num_iterations
        )
        avg_vlm_inference_time = (
            sum(
                sum(m.vlm_inference_times) / len(m.vlm_inference_times) if m.vlm_inference_times else 0
                for m in all_metrics
            )
            / num_iterations
        )

        print_log("\nAverage Metrics:")
        print_log("-" * 80)
        print_log(f"  Total iterations: {num_iterations}")
        print_log(f"  Overall test duration: {overall_duration:.2f}s")
        print_log(f"  Average total chunks: {avg_total_chunks:.2f}")
        print_log(f"  Average stream startup time: {avg_startup_time:.4f}s")
        print_log(f"  Average first chunk duration: {avg_first_chunk_duration:.4f}s")
        print_log(f"  Average first chunk inference time: {avg_first_chunk_inference_time:.4f}s")
        print_log(f"  Average first chunk stream latency: {avg_first_chunk_delay:.4f}s")
        print_log(f"  Average chunk duration: {avg_chunk_duration:.2f}s")
        print_log(f"  Average chunk inference time: {avg_chunk_inference_time:.4f}s")
        print_log(f"  Average chunk inference delay: {avg_chunk_delay:.4f}s")
        print_log(f"  Average CV inference time: {avg_cv_inference_time:.4f}s")
        print_log(f"  Average VLM inference time: {avg_vlm_inference_time:.4f}s")
        avg_e2e_latency = (
            sum(
                sum(m.chunk_e2e_latencies) / len(m.chunk_e2e_latencies) if m.chunk_e2e_latencies else 0
                for m in all_metrics
            )
            / num_iterations
        )
        if avg_e2e_latency > 0:
            print_log(f"  Average chunk E2E latency (vlm_ready - chunk_end): {avg_e2e_latency:.4f}s")
        print_log("=" * 80)

    def test_multiple_streams(
        self,
        content_list: List[List[Dict[str, Any]]],
        timeout_seconds: float = STREAM_TIMEOUT_SECONDS,
        chunking_options: Optional[Dict[str, Any]] = None,
        output_csv: str = "multiple_streams_metrics.csv",
        concurrent: bool = False,
    ) -> List[StreamMetrics]:
        """
        Test multiple streams sequentially or concurrently.

        Args:
            content_list: List of content items for each stream
            timeout_seconds: Maximum time for each stream
            chunking_options: Optional chunking configuration
            output_csv: Output CSV file path
            concurrent: If True, run streams concurrently (not yet implemented)

        Returns:
            List of StreamMetrics objects
        """
        print_log("=" * 80)
        print_log(f"MULTIPLE STREAMS THROUGHPUT TEST ({len(content_list)} streams)")
        print_log("=" * 80)

        if concurrent:
            print_log("WARNING: Concurrent testing not yet implemented. Running sequentially.")

        all_metrics = []
        overall_start = time.time()

        for idx, content in enumerate(content_list):
            stream_id = f"stream-{idx+1}"
            print_log(f"\n--- Testing {stream_id} ---")

            metrics = self._test_stream_request(
                content=content,
                timeout_seconds=timeout_seconds,
                chunking_options=chunking_options,
                stream_id=stream_id,
            )
            all_metrics.append(metrics)

        overall_duration = time.time() - overall_start

        # Print summary
        print_log("\n" + "=" * 80)
        print_log("SUMMARY")
        print_log("=" * 80)
        print_log(f"Total streams: {len(all_metrics)}")
        print_log(f"Overall duration: {overall_duration:.2f}s")

        if all_metrics:
            avg_startup = sum(m.stream_startup_time for m in all_metrics) / len(all_metrics)
            avg_first_chunk = sum(m.first_chunk_inference_time for m in all_metrics) / len(all_metrics)
            avg_chunk_time = sum(m.average_chunk_inference_time for m in all_metrics) / len(all_metrics)
            total_chunks = sum(m.total_chunks for m in all_metrics)

            print_log(f"Average stream startup time: {avg_startup:.4f}s")
            print_log(f"Average first chunk inference time: {avg_first_chunk:.4f}s")
            print_log(f"Average chunk inference time: {avg_chunk_time:.4f}s")
            print_log(f"Total chunks processed: {total_chunks}")
            print_log(f"Overall throughput: {total_chunks / overall_duration:.2f} chunks/sec")

        # Write metrics to CSV
        self._write_metrics_to_csv(all_metrics, output_csv)
        print_log(f"\nMetrics saved to: {output_csv}")

        return all_metrics

    def _write_metrics_to_csv(self, metrics_list: List[StreamMetrics], output_csv: str):
        """Write metrics to CSV file"""
        if not metrics_list:
            print_log("No metrics to write")
            return

        # Write summary metrics
        # check output_csv's parent dir and mkdir for it
        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print_log(f"Created output directory: {output_dir}")

        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")

            # Header
            writer.writerow(
                [
                    "stream_id",
                    "stream_startup_time",
                    "first_chunk_duration",
                    "first_chunk_inference_time",
                    "first_chunk_delay",
                    "average_chunk_duration",
                    "average_chunk_inference_time",
                    "average_chunk_delay",
                    "average_cv_inference_time",
                    "average_vlm_inference_time",
                    "average_chunk_e2e_latency",
                    "total_chunks",
                ]
            )

            # Data rows
            for metrics in metrics_list:
                # Calculate average CV and VLM inference times for this stream
                avg_cv = (
                    sum(metrics.cv_inference_times) / len(metrics.cv_inference_times)
                    if metrics.cv_inference_times
                    else 0
                )
                avg_vlm = (
                    sum(metrics.vlm_inference_times) / len(metrics.vlm_inference_times)
                    if metrics.vlm_inference_times
                    else 0
                )
                avg_e2e = (
                    sum(metrics.chunk_e2e_latencies) / len(metrics.chunk_e2e_latencies)
                    if metrics.chunk_e2e_latencies
                    else 0
                )

                writer.writerow(
                    [
                        metrics.stream_id,
                        f"{metrics.stream_startup_time:.6f}",
                        f"{metrics.first_chunk_duration:.6f}",
                        f"{metrics.first_chunk_inference_time:.6f}",
                        f"{metrics.first_chunk_inference_time - metrics.first_chunk_duration:.6f}",
                        f"{metrics.total_duration / metrics.total_chunks if metrics.total_chunks > 0 else 0:.6f}",
                        f"{metrics.average_chunk_inference_time:.6f}",
                        f"{metrics.average_chunk_delay:.6f}",
                        f"{avg_cv:.6f}",
                        f"{avg_vlm:.6f}",
                        f"{avg_e2e:.6f}",
                        metrics.total_chunks,
                    ]
                )

        # Write detailed chunk timing to separate file (transposed format)
        detailed_csv = output_csv.replace(".csv", "_detailed.csv")
        with open(detailed_csv, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")

            # Data rows (transposed: metrics as rows, chunks as columns)
            for metrics in metrics_list:
                max_len = max(
                    len(metrics.chunk_boundary_times),
                    len(metrics.chunk_inference_timestamps),
                    len(metrics.cv_inference_times),
                    len(metrics.vlm_inference_times),
                    len(metrics.chunk_e2e_latencies),
                )

                # Write stream identifier
                writer.writerow([f"Stream: {metrics.stream_id}"])

                # Header row with chunk indices
                header = ["metric"]
                if CONVERT_CSV_TO_STREAMING_TIME:
                    header.append("chunk_0")
                header.extend([f"chunk_{i+1}" for i in range(max_len)])
                writer.writerow(header)

                # Prepare data for each metric
                if CONVERT_CSV_TO_STREAMING_TIME:
                    startup_value = f"{metrics.stream_startup_time:.6f}"
                    chunk_boundary_row = ["chunk_boundary_time", startup_value]
                    chunk_inference_row = ["chunk_inference_timestamp", startup_value]
                    chunk_delay_row = ["chunk_inference_delay", "0"]
                    cv_inference_row = ["cv_inference_time", ""]
                    vlm_inference_row = ["vlm_inference_time", ""]
                    chunk_e2e_row = ["chunk_e2e_latency", ""]
                else:
                    chunk_boundary_row = ["chunk_boundary_time"]
                    chunk_inference_row = ["chunk_inference_timestamp"]
                    chunk_delay_row = ["chunk_inference_delay"]
                    cv_inference_row = ["cv_inference_time"]
                    vlm_inference_row = ["vlm_inference_time"]
                    chunk_e2e_row = ["chunk_e2e_latency"]

                # Fill in values for each chunk
                for i in range(max_len):
                    boundary_time = metrics.chunk_boundary_times[i] if i < len(metrics.chunk_boundary_times) else ""
                    inference_ts = (
                        metrics.chunk_inference_timestamps[i] if i < len(metrics.chunk_inference_timestamps) else ""
                    )

                    if CONVERT_CSV_TO_STREAMING_TIME and boundary_time != "":
                        boundary_time += metrics.stream_startup_time
                    if CONVERT_CSV_TO_STREAMING_TIME and inference_ts != "":
                        inference_ts += metrics.stream_startup_time

                    delay = (
                        metrics.chunk_inference_delays[i]
                        if i < len(metrics.chunk_inference_delays) and metrics.chunk_inference_delays[i] != ""
                        else ""
                    )
                    cv_time = metrics.cv_inference_times[i] if i < len(metrics.cv_inference_times) else ""
                    vlm_time = metrics.vlm_inference_times[i] if i < len(metrics.vlm_inference_times) else ""
                    e2e_time = metrics.chunk_e2e_latencies[i] if i < len(metrics.chunk_e2e_latencies) else ""

                    chunk_boundary_row.append(f"{boundary_time:.6f}" if boundary_time != "" else "")
                    chunk_inference_row.append(f"{inference_ts:.6f}" if inference_ts != "" else "")
                    chunk_delay_row.append(f"{delay:.6f}" if delay != "" else "")
                    cv_inference_row.append(f"{cv_time:.6f}" if cv_time != "" else "")
                    vlm_inference_row.append(f"{vlm_time:.6f}" if vlm_time != "" else "")
                    chunk_e2e_row.append(f"{e2e_time:.6f}" if e2e_time != "" else "")

                # Write all metric rows
                writer.writerow(chunk_boundary_row)
                writer.writerow(chunk_inference_row)
                writer.writerow(chunk_delay_row)
                writer.writerow(cv_inference_row)
                writer.writerow(vlm_inference_row)
                writer.writerow(chunk_e2e_row)

                # Add blank row between streams
                writer.writerow([])

        print_log(f"Detailed chunk metrics saved to: {detailed_csv}")


def run_example_tests(args: argparse.Namespace):
    """Run example performance tests"""
    # Example: Test with physical camera
    csv_dir = args.csv_dir
    client = StreamClient()

    # Single stream test
    camera_content = [
        {
            "type": "input_camera",
            "input_camera": {
                "camera_id": PHYSICAL_CAMERA_IDS[0],
                "camera_vendor": "Basler",
                "camera_format": CAMERA_FORMAT,
                "camera_width": CAMERA_WIDTH,
                "camera_height": CAMERA_HEIGHT,
                "camera_fps_num": CAMERA_FPS_NUM,
                "camera_fps_den": CAMERA_FPS_DEN,
            },
        },
    ]
    rtsp_content = [
        {
            "type": "video_url",
            "video_url": {
                "url": TEST_RTSP_VIDEO_URL,
            },
        },
    ]
    if TEST_STREAM_MODE == "camera":
        content = camera_content
    elif TEST_STREAM_MODE == "rtsp":
        content = rtsp_content
    elif TEST_STREAM_MODE == "file":
        test_video_base64 = _get_test_video_base64(TEST_VIDEO_PATH)
        content = [
            {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{test_video_base64}"},
            },
        ]
    else:
        raise ValueError(f"Invalid TEST_STREAM_MODE: {TEST_STREAM_MODE}")

    client.test_single_stream(
        content=content,
        timeout_seconds=STREAM_TIMEOUT_SECONDS,
        max_chunks=STREAM_MAX_CHUNKS,
        iterations=STREAM_ITERATIONS,
        output_csv=os.path.join(csv_dir, "single_stream_perf.csv"),
    )

    # Uncomment to test multiple streams
    # content_list = [content] * 3  # Test 3 streams with same content
    # all_metrics = client.test_multiple_streams(
    #     content_list=content_list,
    #     timeout_seconds=20.0,
    #     output_csv="multiple_streams_perf.csv",
    # )


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="API Client Performance Test", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--test_stream_mode", type=str, default="camera", help="Test stream mode: camera, rtsp, file")
    parser.add_argument("--csv_dir", type=str, default=".", help="CSV output directory")
    args = parser.parse_args()
    run_example_tests(args)
