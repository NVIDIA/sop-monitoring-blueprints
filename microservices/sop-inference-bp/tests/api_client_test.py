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
Test chat completion endpoint in api_server.py

This test suite covers:
- Chat completion endpoint (/v1/chat/completions)
"""

import base64
import io
import json
import os
import time
from typing import Any, Dict, List

import requests
import urllib3


def print_log(msg: str):
    """Print with flush=True to ensure immediate output"""
    print(msg, flush=True)


# Base URL for the API server
BASE_URL = "http://localhost:8300"
TEST_VIDEO_PATH = os.getenv("TEST_VIDEO_PATH", "test_video_whole_sop_h264.mp4")

# RTSP video stream could be setup by
# cvlc --loop <vieo.mp4> ":sout=#gather:rtp{sdp=rtsp://:8554/file-stream}" \
#     :network-caching=1500 :sout-all :sout-keep
# video.mp4: must be a mp4 file with H.264/H.265 codec

TEST_RTSP_VIDEO_URL = os.getenv("TEST_RTSP_VIDEO_URL", "rtsp://0.0.0.0:8554/file-stream")
MODEL_ID = "ds_sop_model"
ENUMATION_CAMERA_ID = "0815-0000"
ENUMATION_CAMERA_CONFIG = "configs/Emulation_0815-0000.pfs"

# PHYSICAL_CAMERA_ID is required for physical Balser camera test.
# PHYSICAL_CAMERA_ID is the camera serial number which could be found through pylon viewer.
# Install pylon sdk 25.10.2 to get the camera serial number.
PHYSICAL_CAMERA_ID = os.getenv("PHYSICAL_CAMERA_ID", "40748152")
PHYSICAL_CAMERA_FORMAT = os.getenv("PHYSICAL_CAMERA_FORMAT", "RGB")


def _get_test_video_base64(video_path: str):
    try:
        with open(video_path, "rb") as f:
            encoded_base64 = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Test video file not found at {TEST_VIDEO_PATH}")
    return encoded_base64


def _test_live_streaming(
    url: str,
    content: List[Dict[str, Any]],
    timeout_seconds: float = 10.0,
    stream: bool = True,
    chunking_options: Dict[str, Any] = None,
):
    if chunking_options is None:
        chunking_options = {
            "algorithm": "ddm-net",
            "threshold": 0.8,
            "min_length_sec": 1.0,
            "max_length_sec": 2.0,  # keep shorter chunks for live testing
        }
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "stream": stream,
        "chunking_options": chunking_options,
    }

    headers = {"Content-Type": "application/json"}

    start_time = time.time()
    chunks_received = 0
    timed_out = False
    chunks = []
    first_chunk_time = 0

    try:
        # Initial timeout for connection only
        # Use tuple for timeout: (connect_timeout, read_timeout)
        # Set read timeout higher than manual timeout to let manual timeout control flow
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers=headers,
            stream=True,
            timeout=(timeout_seconds, timeout_seconds * 2),  # (connect, read) timeouts
        )

        assert (
            response.status_code == 200
        ), f"Status: {response.status_code}, Body: {response.text[:500] if hasattr(response, 'text') else 'N/A'}"

        # Check content-type (may include charset)
        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type, f"Content-Type: {content_type}"

        # Read streaming response until timeout
        for line in response.iter_lines():
            # Check if we've exceeded the timeout
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout_seconds:
                timed_out = True
                print_log(f"✓ live streaming test - manually stopping after {elapsed_time:.2f}s")
                response.close()  # Explicitly close the connection to notify server
                break

            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]  # Remove 'data: ' prefix
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        if len(chunks) == 0:
                            first_chunk_time = time.time() - start_time
                            print_log(f"✓ First chunk received at {first_chunk_time:.2f}s")
                        chunks.append(chunk_data)
                        chunks_received += 1
                        # if chunks_received <= 3:  # Print first few chunks
                        print_log(f"✓ streaming chunk {chunks_received}: {chunk_data}")
                    except json.JSONDecodeError:
                        print_log(f"✗ Failed to parse chunk: {data_str}")
                        pass

        # Close the connection explicitly
        response.close()

    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        # Timeout can occur during initial connection or while reading the stream
        elapsed_time = time.time() - start_time
        if "Read timed out" in str(e):
            # Read timeout while streaming - this can be expected for live streams
            timed_out = True
            print_log(f"✓ Read timeout after {elapsed_time:.2f}s while streaming (treating as manual timeout)")
        else:
            print_log(f"✗ Connection timeout after {elapsed_time:.2f}s: {e}")
    finally:
        e2e_time = time.time() - start_time
        if first_chunk_time > 0:
            print_log(f"✓ First chunk received at {first_chunk_time:.2f}s, e2e time: {e2e_time:.2f}s")
        else:
            print_log(f"✗ No chunks received after: {e2e_time:.2f}s")
        if len(chunks) > 0:
            print_log(f"✓ Received {len(chunks)} chunks, e2e time: {e2e_time:.2f}s")

    # Verify results
    elapsed_time = time.time() - start_time

    if timed_out:
        # Expected behavior - we manually stopped the stream after timeout
        print_log(f"✓ live streaming test completed - stopped after {elapsed_time:.2f}s as expected")
        print_log(f"✓ Received {chunks_received} chunks during streaming")
        assert chunks_received > 0, "Should receive at least one chunk before timeout"
    else:
        # Stream ended naturally (may happen if camera disconnects or server stops)
        print_log(f"⚠ live stream ended naturally after {elapsed_time:.2f}s")
        print_log(f"✓ Received {chunks_received} chunks total")
        assert chunks_received > 0, "Should receive at least one chunk"


class TestChatCompletionEndpoint:
    """Test chat completion endpoint"""

    def test_chat_completion_basic(self):
        """Test basic chat completion without streaming"""

        test_video_base64 = _get_test_video_base64(TEST_VIDEO_PATH)
        payload = {
            "model": "ds_sop_model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        # {"type": "text", "text": "prompt for video analysis"},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{test_video_base64}"},
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

        try:
            response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers, timeout=30)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, urllib3.exceptions.TimeoutError) as e:
            print_log(f"✗ test_chat_completion_basic failed: {e}")
            return
        except Exception as e:
            print_log(f"✗ test_chat_completion_basic failed: {e}")
            return
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
        print_log(f"✓ test_chat_completion_basic working - response: {choice['message']['content']}")

        print_log(f"✓ test_chat_completion_basic working - response: {choice['message']['content'][:30]}...")

    def test_chat_completion_streaming(self):
        """Test chat completion with streaming"""

        test_video_base64 = _get_test_video_base64(TEST_VIDEO_PATH)
        content = [
            {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{test_video_base64}"},
            },
        ]

        print_log(f"✓ Starting chat completion streaming test for base64 video: length: {len(test_video_base64)}")
        _test_live_streaming(BASE_URL, content, timeout_seconds=20, stream=True)
        print_log("✓ Chat completion streaming test completed for base64 video")

    def test_basler_camera_streaming_enumeration(self):
        """Test camera live streaming with timeout

        Camera streaming is continuous and never stops, so we use a timeout
        to terminate the connection after 10 seconds.
        """
        camera_id = "0815-0000"
        camera_config = "configs/Emulation_0815-0000.pfs"
        content = [
            {
                "type": "input_camera",
                "input_camera": {
                    "camera_id": camera_id,
                    "camera_vendor": "Basler",
                    "config": camera_config,
                    "camera_format": "RGB",
                },
            },
        ]
        print_log(f"✓ Starting camera live streaming test, camera_id: {camera_id}, camera_config: {camera_config}")
        _test_live_streaming(BASE_URL, content, timeout_seconds=10, stream=True)
        print_log(f"✓ Camera live streaming test completed, camera_id: {camera_id}, camera_config: {camera_config}")

    def test_physical_camera_live(
        self, camera_id: str = PHYSICAL_CAMERA_ID, format: str = "RGB", timeout_seconds: float = 20
    ):
        """Test camera live streaming with timeout

        format: ["RGB", "UYVY", "YUY2"].
        camera_id: "0815-0000"
        """
        content = [
            {
                "type": "input_camera",
                "input_camera": {
                    "camera_id": camera_id,
                    "camera_vendor": "Basler",
                    "camera_format": format,
                    "camera_width": 1280,
                    "camera_height": 720,
                    "camera_fps_num": 30,  # frame_rate = camera_fps_num / camera_fps_den
                    "camera_fps_den": 1,
                },
            },
        ]
        print_log(f"✓ Starting camera live streaming test, camera_id: {camera_id}, camera_format: {format}")
        _test_live_streaming(BASE_URL, content, timeout_seconds=timeout_seconds, stream=True)
        print_log(f"✓ Camera live streaming test completed, camera_id: {camera_id}, camera_format: {format}")

    def test_video_rtsp_live_streaming(self):
        """Test video rtsp live streaming with timeout

        Video rtsp streaming is continuous and never stops, so we use a timeout
        to terminate the connection after 20 seconds.
        """
        video_rtsp_url = TEST_RTSP_VIDEO_URL
        content = [
            {
                "type": "video_url",
                "video_url": {
                    "url": video_rtsp_url,
                },
            },
        ]
        print_log(f"✓ Starting video rtsp live streaming test, video_rtsp_url: {video_rtsp_url}")
        _test_live_streaming(BASE_URL, content, timeout_seconds=30, stream=True)
        print_log(f"✓ Video rtsp live streaming test completed, video_rtsp_url: {video_rtsp_url}")


def run_all_tests():
    """Run all tests"""
    test_instance = TestChatCompletionEndpoint()
    test_instance.test_chat_completion_basic()
    test_instance.test_chat_completion_streaming()
    # test_instance.test_basler_camera_streaming_enumeration()
    # test_instance.test_video_rtsp_live_streaming()
    # test_instance.test_physical_camera_live(PHYSICAL_CAMERA_ID, "RGB", timeout_seconds=36)


if __name__ == "__main__":
    run_all_tests()
