# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import time
import requests
import pprint

from openai import OpenAI
from termcolor import colored
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Create OpenAI client pointing to your API server
base_url = "http://localhost:8080/v1"
client = OpenAI(base_url=base_url, api_key="don't care.", max_retries=1)

video_path = os.path.join(_THIS_DIR, "test_video.mp4")
action_json_path = os.path.join(_THIS_DIR, "actions.json")
vlm_prompt_path = os.path.join(_THIS_DIR, "vlm_prompts.txt")
cr_segment_prompt_path = os.path.join(_THIS_DIR, "cr_act_seg_prompt.txt")

# Step 1: Upload a file
print(colored("\n=== Step 1: File Upload ===", "cyan"))
upload_start_time = time.time()
with open(video_path, "rb") as f:
    uploaded_file = client.files.create(
        file=f,
        purpose="vision"
    )
upload_end_time = time.time()
upload_duration = upload_end_time - upload_start_time

file_id = uploaded_file.id

with open(vlm_prompt_path, "r") as f:
    prompt = f.read()

with open(action_json_path, "r") as f:
    action_json = f.read()

if os.path.isfile(cr_segment_prompt_path):
    with open(cr_segment_prompt_path, "r") as fp:
        cr_segment_prompt = fp.read()

cr_segment_sys_prompt = """You are a helpful video analyzer."""

sop_detection_request = {
    "action_json": action_json,
    "vlm_output": "",
}

# Step 2: Chat completions with uploaded file
print(colored("\n=== Step 2 : Chat completions with uploaded file ===", "cyan"))

chat_start_time = time.time()
chat_response = client.chat.completions.create(
    model="placeholder", # temporary model name
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_file",
                    "image_file": {
                        "file_id": file_id,
                    }
                }
            ]
        }
    ],
)
chat_end_time = time.time()
chat_duration = chat_end_time - chat_start_time
print(colored(f"Chat completion time: {chat_duration:.2f} seconds", "green"))

# Step 3: Check the sop sequence output by VLM
print(colored("\n=== Step 3 : Check the sop sequence output by VLM ===", "cyan"))
vlm_output = chat_response.choices[0].message.content
sop_detection_request["vlm_output"] = vlm_output
# POST to /v1/sop/detection
sop_detection_url = f"{base_url}/sop/detection"
sop_detection_start_time = time.time()
sop_detection_response = requests.post(sop_detection_url, json=sop_detection_request)
sop_detection_end_time = time.time()
sop_detection_duration = sop_detection_end_time - sop_detection_start_time
wall_clock_time = sop_detection_end_time - chat_start_time

print(colored(f"SOP detection time: {sop_detection_duration:.2f} seconds", "green"))
print(colored(f"Wall clock time: {wall_clock_time:.2f} seconds", "green"))

if sop_detection_response.status_code == 200:
    sop_detection_result = sop_detection_response.json()
    print(colored(f"Checker ID: {sop_detection_result.get('checker_id')}", "yellow"), flush=True)
    print(colored(f"{vlm_output}", "yellow"), flush=True)
    print(colored(f"{pprint.pformat(sop_detection_result)}", "yellow"), flush=True)
else:
    print(colored(f"Error in SOP detection: {sop_detection_response.status_code} {sop_detection_response.text}", "red"))

print(colored("\n=== Test Complete ===", "magenta"))
