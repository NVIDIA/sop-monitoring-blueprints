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


# Window-Matched MCQ question template.
#
# Deliberately identical in wording to the sequential-MCQ / dynamic-shuffling
# templates: WMCQ changes the *geometry* of the training clip, not the task the
# model is asked to perform. Keeping the wording identical means a WMCQ sample
# and a plain MCQ sample differ only in what the video shows, which is the whole
# point of the augmentation.
#
# The "<video>" token is NOT part of this template. It is prepended at assembly
# time via the --human-suffix argument (default "<video>\n"), the same way
# golden_gqa_to_gqa.py and spatial_localization.py do it, so the separator is a
# real newline.
QUESTION_TEMPLATE = """There are [STEP] possible steps for the SOP (Standard Operation Procedure) of the given video.
What step is the [SUBJECT] doing?
"""
