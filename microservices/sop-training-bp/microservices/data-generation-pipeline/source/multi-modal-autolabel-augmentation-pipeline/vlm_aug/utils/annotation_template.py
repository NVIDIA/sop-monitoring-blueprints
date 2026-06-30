######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
######################################################################################################



llava_video = {
        "id": 0,
        "conversations": [
            {
                "from": "human",
                "value": "human_prompt"
            },
            {
                "from": "gpt",
                "value": "gpt_response"
            }],
        "video": "video_path"
    }

dynamic_meta = {
    "frame_cnts": "",
    "min_frames": "",
    "max_frames": "",
    "dynamic_sample": ""
}

dmcq_meta = {
    "gt_action": "",
    "pos_or_neg": "",
    "hard_mode": "",
    "num_options": ""
}

ds_meta = {
    "frame_cnts": [],
    "source_videos": [],
    "total_frames": "",
    "is_hard": ""
}

en_meta = {
    "num_options": ""
}