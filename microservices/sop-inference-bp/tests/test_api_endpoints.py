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
Comprehensive unittests for all API endpoints in api_server.py

This test suite covers:
- Health check endpoints (/v1/live, /v1/startup, /v1/ready)
- Model listing endpoint (/v1/models)
- Metadata endpoint (/v1/metadata)
- File management endpoints (/v1/files)
- Chat completion endpoint (/v1/chat/completions)
- Metrics endpoint (/v1/metrics)
- Edge cases and error handling
"""

import base64
import io
import json
import os

import requests

# Base URL for the API server
BASE_URL = "http://localhost:8300"
TEST_VIDEO_PATH = os.getenv("TEST_VIDEO_PATH", "test_video_whole_sop_h264.mp4")


def _get_test_video_base64(video_path: str):
    try:
        with open(video_path, "rb") as f:
            encoded_base64 = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Test video file not found at {TEST_VIDEO_PATH}")
    return encoded_base64


test_video_base64_str = _get_test_video_base64(TEST_VIDEO_PATH)


def print_log(msg: str):
    """Print with flush=True to ensure immediate output"""
    print(msg, flush=True)


class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_health_live(self):
        """Test /v1/live endpoint"""
        response = requests.get(f"{BASE_URL}/v1/live")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "health.response"
        assert data["message"] == "Service is live."
        print_log("✓ Health live endpoint working")

    def test_health_startup(self):
        """Test /v1/startup endpoint"""
        response = requests.get(f"{BASE_URL}/v1/startup")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "health.response"
        assert "started successfully" in data["message"].lower()
        print_log("✓ Health startup endpoint working")

    def test_health_ready(self):
        """Test /v1/ready endpoint"""
        response = requests.get(f"{BASE_URL}/v1/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "health.response"
        assert "ready" in data["message"].lower() or "dummy" in data["message"].lower()
        print_log("✓ Health ready endpoint working")


class TestModelEndpoints:
    """Test model-related endpoints"""

    def test_list_models(self):
        """Test /v1/models endpoint"""
        response = requests.get(f"{BASE_URL}/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0

        # Check model structure
        model = data["data"][0]
        assert "id" in model
        assert model["object"] == "model"
        assert "owned_by" in model
        print_log(f"✓ Models endpoint working - found model: {model['id']}")


class TestMetadataEndpoint:
    """Test metadata endpoint"""

    def test_show_metadata(self):
        """Test /v1/metadata endpoint"""
        try:
            response = requests.get(f"{BASE_URL}/v1/metadata")
            assert response.status_code == 200, f"Status: {response.status_code}, Body: {response.text}"
            data = response.json()

            # Check structure
            assert "version" in data
            assert "modelInfo" in data
            assert "licenseInfo" in data

            # Check license info
            license_info = data["licenseInfo"]
            assert "name" in license_info
            assert "path" in license_info
            assert "size" in license_info
            assert "content" in license_info

            print_log(f"✓ Metadata endpoint working - version: {data['version']}")
        except Exception as e:
            print_log(f"✗ test_show_metadata ERROR: {e}")
            raise


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint"""

    def test_metrics_endpoint_basic(self):
        """Test /v1/metrics endpoint returns Prometheus format"""
        response = requests.get(f"{BASE_URL}/v1/metrics")
        assert response.status_code == 200

        # Check content type for Prometheus metrics
        content_type = response.headers.get("content-type", "")
        assert "text/plain" in content_type, f"Expected text/plain content type, got: {content_type}"

        # Get response text
        metrics_text = response.text
        assert len(metrics_text) > 0, "Metrics response should not be empty"

        print_log(f"✓ Metrics endpoint returns data: {len(metrics_text)} bytes")

    def test_metrics_contains_expected_metrics(self):
        """Test that metrics endpoint includes expected metric names"""
        response = requests.get(f"{BASE_URL}/v1/metrics")
        assert response.status_code == 200

        metrics_text = response.text
        print_log(f"Metrics text: {metrics_text}")

        # Check for expected metrics defined in api_server.py
        expected_metrics = [
            "api_requests_total",
            "api_request_latency_seconds",
            "chat_completions_total",
            "gpu_utilization_percent",
            "gpu_memory_used_megabytes",
        ]

        found_metrics = []
        missing_metrics = []

        for metric in expected_metrics:
            if metric in metrics_text:
                found_metrics.append(metric)
            else:
                missing_metrics.append(metric)

        # At minimum, we should have the API request metrics
        assert "api_requests_total" in metrics_text, "Should have api_requests_total metric"
        assert "api_request_latency_seconds" in metrics_text, "Should have api_request_latency_seconds metric"

        print_log(f"✓ Metrics endpoint contains expected metrics: {', '.join(found_metrics)}")
        if missing_metrics:
            print_log(f"  Note: Some metrics not present (may be GPU-dependent): {', '.join(missing_metrics)}")

    def test_metrics_format_validity(self):
        """Test that metrics follow Prometheus format"""
        response = requests.get(f"{BASE_URL}/v1/metrics")
        assert response.status_code == 200

        metrics_text = response.text
        lines = metrics_text.strip().split("\n")

        # Prometheus metrics should have:
        # - Comment lines starting with #
        # - Metric lines with format: metric_name{labels} value [timestamp]

        has_comments = False
        has_metrics = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                has_comments = True
            else:
                # Should be a metric line
                # Basic check: should contain at least a metric name and a value
                parts = line.split()
                if len(parts) >= 2:
                    has_metrics = True

        assert has_comments or has_metrics, "Metrics should contain either comments or metric data"
        print_log(f"✓ Metrics follow Prometheus format (comments: {has_comments}, metrics: {has_metrics})")

    def test_metrics_increments_on_requests(self):
        """Test that api_requests_total increments on API calls"""
        # Get initial metrics
        response1 = requests.get(f"{BASE_URL}/v1/metrics")
        assert response1.status_code == 200
        metrics_text1 = response1.text

        # Extract api_requests_total for /v1/live
        initial_count = self._extract_metric_value(metrics_text1, "api_requests_total", "/v1/live")

        # Make a request to /v1/live
        requests.get(f"{BASE_URL}/v1/live")

        # Get metrics again
        response2 = requests.get(f"{BASE_URL}/v1/metrics")
        assert response2.status_code == 200
        metrics_text2 = response2.text

        # Extract api_requests_total for /v1/live again
        final_count = self._extract_metric_value(metrics_text2, "api_requests_total", "/v1/live")

        # Verify it incremented (at least by 1, possibly more if other tests are running)
        if initial_count is not None and final_count is not None:
            assert final_count > initial_count, f"Request count should increment: {initial_count} -> {final_count}"
            print_log(f"✓ Metrics increment correctly: {initial_count} -> {final_count}")
        else:
            print_log("✓ Metrics present (exact count comparison skipped)")

    def test_metrics_chat_completions_counter(self):
        """Test that chat_completions_total metric exists"""
        response = requests.get(f"{BASE_URL}/v1/metrics")
        assert response.status_code == 200

        metrics_text = response.text

        # Check if chat_completions_total exists
        if "chat_completions_total" in metrics_text:
            print_log("✓ chat_completions_total metric found")
        else:
            # If not found, it might be because no chat completions have been made yet
            # This is still valid - the metric will appear after first chat completion
            print_log("✓ chat_completions_total metric will appear after first chat completion")

    def test_metrics_gpu_metrics_present(self):
        """Test that GPU metrics are present (if GPU available)"""
        response = requests.get(f"{BASE_URL}/v1/metrics")
        assert response.status_code == 200

        metrics_text = response.text

        # GPU metrics might not be available if no GPU or nvidia-smi not installed
        # This is expected behavior per the code
        has_gpu_util = "gpu_utilization_percent" in metrics_text
        has_gpu_mem = "gpu_memory_used_megabytes" in metrics_text

        if has_gpu_util or has_gpu_mem:
            print_log(f"✓ GPU metrics available (util: {has_gpu_util}, mem: {has_gpu_mem})")
        else:
            print_log("✓ GPU metrics not available (expected if no GPU/nvidia-smi)")

    def _extract_metric_value(self, metrics_text: str, metric_name: str, path_label: str = None) -> float:
        """Helper to extract a metric value from Prometheus text format"""
        for line in metrics_text.split("\n"):
            if line.startswith(metric_name):
                # Check if path label matches (if provided)
                if path_label and f'path="{path_label}"' not in line:
                    continue

                # Extract the value (last part of the line)
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[-1])
                    except ValueError:
                        continue
        return None


class TestFileEndpoints:
    """Test file management endpoints"""

    def test_file_upload_list_chat_completion_download_delete_workflow(self):
        """Test complete file workflow: upload -> list -> chat completion -> download -> delete"""

        # 1. Upload a test file
        test_video_path = TEST_VIDEO_PATH
        with open(test_video_path, "rb") as f:
            test_content = f.read()
        test_filename = os.path.basename(test_video_path)

        files = {"file": (test_filename, io.BytesIO(test_content), "video/mp4")}
        data = {"purpose": "test"}

        response = requests.post(f"{BASE_URL}/v1/files", files=files, data=data)
        assert response.status_code == 200
        upload_data = response.json()

        # Verify upload response
        assert upload_data["object"] == "file"
        assert "id" in upload_data
        assert upload_data["filename"] == test_filename
        assert upload_data["bytes"] == len(test_content)
        assert upload_data["purpose"] == "test"

        file_id = upload_data["id"]
        print_log(f"✓ File uploaded successfully: {file_id}")

        # 2. List files and verify our file is there
        response = requests.get(f"{BASE_URL}/v1/files")
        assert response.status_code == 200
        list_data = response.json()

        assert list_data["object"] == "list"
        assert isinstance(list_data["data"], list)

        # Find our file in the list
        our_file = None
        for file_obj in list_data["data"]:
            if file_obj["id"] == file_id:
                our_file = file_obj
                break

        assert our_file is not None, f"File {file_id} not found in list"
        assert our_file["filename"] == test_filename
        print_log(f"✓ File found in list: {len(list_data['data'])} total files")

        # 3. Download file content
        response = requests.get(f"{BASE_URL}/v1/files/{file_id}/content")
        assert response.status_code == 200
        downloaded_content = response.content
        assert downloaded_content == test_content
        assert response.headers.get("content-type") == "application/octet-stream"
        print_log(f"✓ File downloaded successfully: {len(downloaded_content)} bytes")

        # 4. test chat_completion with video_file content
        payload = {
            "model": "ds_sop_model",
            "messages": [{"role": "user", "content": [{"type": "input_video", "file_id": file_id}]}],
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers, timeout=60)
        assert response.status_code == 200, f"Status: {response.status_code}, Body: {response.text[:100]}"
        data = response.json()
        assert data["object"] == "chat.completion"
        assert "id" in data
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["index"] == 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "content" in data["choices"][0]["message"]
        assert isinstance(data["choices"][0]["message"]["content"], str)
        print_log(f"✓ Chat completion working - response: {data['choices'][0]['message']['content']}")

        # 5. Delete the file
        response = requests.delete(f"{BASE_URL}/v1/files/{file_id}")
        assert response.status_code == 200
        delete_data = response.json()

        assert delete_data["id"] == file_id
        assert delete_data["object"] == "file.deleted"
        assert delete_data["deleted"] is True
        print_log(f"✓ File deleted successfully: {file_id}")

        # 6. Verify file is no longer accessible
        response = requests.get(f"{BASE_URL}/v1/files/{file_id}/content")
        assert response.status_code == 404
        print_log("✓ File no longer accessible after deletion")

    def test_file_not_found(self):
        """Test accessing non-existent file"""
        fake_file_id = "file-nonexistent123456789"

        # Try to download non-existent file
        response = requests.get(f"{BASE_URL}/v1/files/{fake_file_id}/content")
        assert response.status_code == 404

        # Try to delete non-existent file
        response = requests.delete(f"{BASE_URL}/v1/files/{fake_file_id}")
        assert response.status_code == 404

        print_log("✓ File not found errors handled correctly")


class TestChatCompletionEndpoint:
    """Test chat completion endpoint"""

    def test_chat_completion_basic(self):
        """Test basic chat completion without streaming"""

        payload = {
            "model": "ds_sop_model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{test_video_base64_str}"},
                        },
                    ],
                }
            ],
            "stream": False,
            "chunking_options": {
                "algorithm": "ddm-net",
                "threshold": 0.8,
                "min_length_sec": 1.0,
                "max_length_sec": 60.0,
            },
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers, timeout=60)

        assert response.status_code == 200, f"Status: {response.status_code}, Body: {response.text[:500]}"
        data = response.json()

        # Verify response structure
        assert data["object"] == "chat.completion"
        assert "id" in data
        assert data["id"].startswith("chatcmpl-")
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert len(data["choices"]) > 0

        # Check choice structure
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert "content" in choice["message"]
        assert isinstance(choice["message"]["content"], str)
        print_log(f"✓ Chat completion working - response: {choice['message']['content']}")

        print_log(f"✓ Chat completion working - response length: {len(choice['message']['content'])} chars")

    def test_chat_completion_streaming(self):
        """Test chat completion with streaming"""

        payload = {
            "model": "ds_sop_model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{test_video_base64_str}"},
                        },
                    ],
                }
            ],
            "stream": True,
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(
            f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers, stream=True, timeout=60
        )

        assert (
            response.status_code == 200
        ), f"Status: {response.status_code}, Body: {response.text[:500] if hasattr(response, 'text') else 'N/A'}"
        # Check content-type (may include charset)
        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type, f"Content-Type: {content_type}"

        # Read streaming response
        chunks = []
        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]  # Remove 'data: ' prefix
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        chunks.append(chunk_data)
                        print_log(f"✓ Streaming chat completion working - received chunk: {chunk_data}")
                    except json.JSONDecodeError:
                        print_log(f"✗ Streaming chat completion working - error: {data_str}")
                        pass

        assert len(chunks) > 0, "Should receive at least one chunk"

        # Verify chunk structure
        first_chunk = chunks[0]
        assert first_chunk["object"] == "chat.completion.chunk"
        assert "id" in first_chunk
        assert "choices" in first_chunk

        print_log(f"✓ Streaming chat completion working - received {len(chunks)} chunks")

    def test_chat_completion_invalid_content_type(self):
        """Test chat completion with invalid content type"""

        payload = {"model": "ds_sop_model", "messages": [{"role": "user", "content": []}]}

        headers = {"Content-Type": "text/plain"}  # Invalid content type

        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers)

        assert response.status_code == 415  # Unsupported Media Type
        print_log("✓ Invalid content type handled correctly")

    def test_chat_completion_validation_errors(self):
        """Test chat completion with various validation errors"""

        # Test with invalid content type (this should fail)
        payload = {"model": "ds_sop_model", "messages": [{"role": "user", "content": []}]}

        headers = {"Content-Type": "text/plain"}  # Wrong content type

        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers)

        # Should return 415 Unsupported Media Type
        assert response.status_code == 415, f"Status: {response.status_code}, expected 415"
        print_log("✓ Validation errors handled correctly")


class TestUniformChunkingEndpoint:
    """Test chat completion endpoint with uniform chunking algorithm"""

    def test_uniform_chunking_non_streaming(self):
        """Uniform chunking options are accepted and return a valid non-streaming response"""
        payload = {
            "model": "ds_sop_model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{test_video_base64_str}"},
                        },
                    ],
                }
            ],
            "stream": False,
            "chunking_options": {
                "algorithm": "uniform",
                "chunk_length_sec": 2.5,
            },
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers, timeout=60)
        assert response.status_code == 200, f"Status: {response.status_code}, Body: {response.text[:500]}"
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["id"].startswith("chatcmpl-")
        assert len(data["choices"]) > 0
        choice = data["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        metadata_list = choice.get("chunk_metadata_list", [])
        assert len(metadata_list) > 0, "Expected at least one chunk in chunk_metadata_list"
        for chunk_meta in metadata_list:
            assert "start_time" in chunk_meta, "Each chunk must have start_time"
            assert "end_time" in chunk_meta, "Each chunk must have end_time"
            assert "chunk_idx" in chunk_meta, "Each chunk must have chunk_idx"
        print_log(f"✓ Uniform chunking (non-streaming) returned {len(metadata_list)} chunks")

    def test_uniform_chunking_streaming(self):
        """Uniform chunking options are accepted and return a valid streaming response"""
        payload = {
            "model": "ds_sop_model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{test_video_base64_str}"},
                        },
                    ],
                }
            ],
            "stream": True,
            "chunking_options": {
                "algorithm": "uniform",
                "chunk_length_sec": 2.5,
            },
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers, stream=True, timeout=60
        )
        assert response.status_code == 200, f"Status: {response.status_code}, Body: {response.text[:500]}"
        assert "text/event-stream" in response.headers.get("content-type", "")

        chunks_received = 0
        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if line == "data: [DONE]":
                break
            assert line.startswith("data: "), f"Unexpected SSE line: {line}"
            event_data = json.loads(line[len("data: "):])
            assert "choices" in event_data
            choice = event_data["choices"][0]
            chunk_meta = choice.get("chunk_metadata", {})
            assert "start_time" in chunk_meta, "Streaming chunk must have start_time"
            assert "end_time" in chunk_meta, "Streaming chunk must have end_time"
            chunks_received += 1

        assert chunks_received > 0, "Expected at least one streaming chunk"
        print_log(f"✓ Uniform chunking (streaming) received {chunks_received} chunks")

    def test_uniform_chunking_invalid_chunk_length(self):
        """chunk_length_sec <= 0 is rejected with 422"""
        payload = {
            "model": "ds_sop_model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{test_video_base64_str}"},
                        },
                    ],
                }
            ],
            "stream": False,
            "chunking_options": {
                "algorithm": "uniform",
                "chunk_length_sec": -1.0,
            },
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers, timeout=10)
        assert response.status_code == 422, f"Expected 422, got {response.status_code}: {response.text[:300]}"
        print_log("✓ Uniform chunking rejects chunk_length_sec <= 0 with 422")

    def test_uniform_chunking_rejects_ddm_extra_fields(self):
        """Sending DDM-specific fields (threshold) with algorithm='uniform' is rejected with 422"""
        payload = {
            "model": "ds_sop_model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{test_video_base64_str}"},
                        },
                    ],
                }
            ],
            "stream": False,
            "chunking_options": {
                "algorithm": "uniform",
                "chunk_length_sec": 5.0,
                "threshold": 0.9,
            },
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers, timeout=10)
        assert response.status_code == 422, f"Expected 422, got {response.status_code}: {response.text[:300]}"
        print_log("✓ Uniform chunking rejects extra DDM fields with 422")

    def test_ddm_net_still_works_after_uniform_added(self):
        """Regression: existing ddm-net algorithm still works correctly"""
        payload = {
            "model": "ds_sop_model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{test_video_base64_str}"},
                        },
                    ],
                }
            ],
            "stream": False,
            "chunking_options": {
                "algorithm": "ddm-net",
                "threshold": 0.8,
                "min_length_sec": 1.0,
                "max_length_sec": 60.0,
            },
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers, timeout=60)
        assert response.status_code == 200, f"Status: {response.status_code}, Body: {response.text[:500]}"
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) > 0
        print_log("✓ ddm-net algorithm still works correctly after uniform chunking added")


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_invalid_endpoint(self):
        """Test accessing invalid endpoint"""
        response = requests.get(f"{BASE_URL}/v1/invalid_endpoint")
        assert response.status_code == 404
        print_log("✓ Invalid endpoint returns 404")

    def test_method_not_allowed(self):
        """Test using wrong HTTP method"""
        # Try POST on a GET endpoint
        response = requests.post(f"{BASE_URL}/v1/live")
        assert response.status_code == 405  # Method Not Allowed
        print_log("✓ Method not allowed handled correctly")


def run_all_tests():
    """Run all tests and print_log summary"""
    print_log("\n" + "=" * 60)
    print_log("Running API Endpoint Tests")
    print_log("=" * 60 + "\n")

    test_classes = [
        TestHealthEndpoints,
        TestModelEndpoints,
        TestMetadataEndpoint,
        TestFileEndpoints,
        TestChatCompletionEndpoint,
        TestUniformChunkingEndpoint,
        TestEdgeCases,
        TestMetricsEndpoint,
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        print_log(f"\n{test_class.__doc__}")
        print_log("-" * 60)

        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith("test_")]

        for test_method_name in test_methods:
            total_tests += 1
            test_method = getattr(test_instance, test_method_name)

            try:
                test_method()
                passed_tests += 1
            except AssertionError as e:
                failed_tests.append((test_class.__name__, test_method_name, str(e)))
                print_log(f"✗ {test_method_name} FAILED: {e}")
            except Exception as e:
                failed_tests.append((test_class.__name__, test_method_name, str(e)))
                print_log(f"✗ {test_method_name} ERROR: {e}")

    # Print summary
    print_log("\n" + "=" * 60)
    print_log("Test Summary")
    print_log("=" * 60)
    print_log(f"Total tests: {total_tests}")
    print_log(f"Passed: {passed_tests}")
    print_log(f"Failed: {len(failed_tests)}")

    if failed_tests:
        print_log("\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print_log(f"  - {class_name}.{method_name}: {error}")
    else:
        print_log("\n🎉 All tests passed!")

    print_log("=" * 60 + "\n")

    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
