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


SPATIAL_KEYWORDS = {
    "top", "bottom", "upper", "lower",
    "left", "right", "front", "back",
    "inner", "outer", "inside", "outside",
    "near", "far", "above", "below",
    "north", "south", "east", "west",
    "vertical", "horizontal",
    "forward", "backward",
    "high", "low",
}

spatial_llm_cfg = {
    "temperature": 0.2,
    "top_p": 0.8,
    "max_tokens": 512,
}

MAX_QUESTION_TOKENS = 60
MAX_QA_PER_GROUP = 3

# Fallback templates used when LLM is unavailable or returns invalid output.
# Placeholders: {action_a}, {action_b}, {idx_a}, {idx_b},
#               {shared_context}, {spatial_a}, {spatial_b}

FALLBACK_REGION_QUESTION = (
    "<video>\nFocus on the {shared_context}. "
    "Which position is being acted on?\n"
    "(A) The {spatial_a} position\n"
    "(B) The {spatial_b} position"
)

FALLBACK_CONTRAST_QUESTION = (
    "<video>\nWhich action is being performed?\n"
    "({idx_a}) {action_a}\n"
    "({idx_b}) {action_b}"
)

FALLBACK_REGION_ANSWER_MAP = {"a": "(A)", "b": "(B)"}
FALLBACK_CONTRAST_ANSWER_MAP_TEMPLATE = "({idx})"
