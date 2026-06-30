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


# NOTE: Unlike bcq/mcq, GQA templates are NOT unified across modes because the
# answer grammar differs between modes:
#   single-operator -> singular subject ("The operator is X")
#   two-operator    -> plural ("The workers are X1, X2")
# Both templates are actively used (see gqa_to_gqas.prepare_concurrent_sample_qa
# and golden_gqa_to_gqa.prepare_concurrent_sample_qa).
# Single-operator templates (original)
QUESTION_TEMPLATE = "Question: What step is the [SUBJECT] doing in the video?"
ANSWERS_TEMPLATE = "Answer: The [SUBJECT] is [STEP]."

# Two-operator / Concurrent action templates (only used when two-operator mode is ON)
QUESTION_TEMPLATE_CONCURRENT = "Question: What steps are being performed in the video?"
ANSWERS_TEMPLATE_CONCURRENT = "Answer: The workers are [STEPS]."