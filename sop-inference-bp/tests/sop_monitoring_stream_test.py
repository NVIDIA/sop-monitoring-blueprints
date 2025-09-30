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

video_path = os.path.join(_THIS_DIR, "test_video_whole_sop_h264.mp4")
action_json_path = os.path.join(_THIS_DIR, "actions.json")
vlm_prompt_path = os.path.join(_THIS_DIR, "vlm_prompts.txt")

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

# Step 2: Chat completions with uploaded file
print(colored("\n=== Step 2: Chat completions with uploaded file ===", "cyan"))
with open(vlm_prompt_path, "r") as f:
    prompt = f.read()

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
                        "chunking_options": {
                            "algorithm": "uniform",
                        }
                    }
                }
            ]
        }
    ],
    stream=True,
)

with open(action_json_path, "r") as f:
    action_json = f.read()

sop_detection_url = f"{base_url}/sop/detection"

inference_start_time = time.time()
checker_id = "*"
print(colored("Starting SOP detection", "yellow"))
for chunk in chat_response:
    vlm_output = chunk.choices[0].delta.content

    if vlm_output is None:
        break

    sop_detection_request = {
        "action_json": action_json,
        "vlm_output": vlm_output,
        "keep_alive": True,
        "checker_id": checker_id,
    }
    sop_detection_start_time = time.time()
    sop_detection_response = requests.post(sop_detection_url, json=sop_detection_request)
    sop_detection_end_time = time.time()

    sop_detection_duration = sop_detection_end_time - sop_detection_start_time
    #print(colored(f"SOP detection duration: {sop_detection_duration:.2f} seconds", "yellow"))

    print(colored(f"SOP Checker runs", "cyan"))
    if sop_detection_response.status_code == 200:
        sop_detection_result = sop_detection_response.json()
        print(colored(f"Checker ID: {sop_detection_result.get('checker_id')}", "yellow"), flush=True)
        print(colored(f"{vlm_output}", "yellow"), flush=True)
        print(colored(f"{pprint.pformat(sop_detection_result)}", "yellow"), flush=True)
        checker_id = sop_detection_result.get("checker_id")
    else:
        print(colored(f"Error in SOP detection: {sop_detection_response.status_code} {sop_detection_response.text}", "red"))

print(colored(f"Sending final request to clear the checker: {checker_id}", "yellow"))
# send the last request to clear the checker
sop_detection_request = {
    "action_json": action_json,
    "vlm_output": "",
    "keep_alive": False,
    "checker_id": checker_id,
}
sop_detection_response = requests.post(sop_detection_url, json=sop_detection_request)
if sop_detection_response.status_code == 200:
    sop_detection_result = sop_detection_response.json()
    print(colored(f"Checker ID: {sop_detection_result.get('checker_id')}", "yellow"), flush=True)
    print(colored(f"{vlm_output}", "yellow"), flush=True)
    print(colored(f"{pprint.pformat(sop_detection_result)}", "yellow"), flush=True)
else:
    print(colored(f"Error in SOP detection: {sop_detection_response.status_code} {sop_detection_response.text}", "red"))

print(colored("\n=== Test Complete ===", "magenta"))
