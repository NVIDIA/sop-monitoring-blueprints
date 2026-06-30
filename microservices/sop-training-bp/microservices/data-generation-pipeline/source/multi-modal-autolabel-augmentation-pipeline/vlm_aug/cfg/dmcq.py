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


QUESTION_TEMPLATE = [
"""There are [STEP] possible steps for the SOP (Standard Operation Procedure) of the given video.
What step is the [SUBJECT] doing?
""",
"""There are [STEP] possible steps for the SOP (Standard Operation Procedure) of the given video.
What step does the [SUBJECT] take?
""",
"""There are [STEP] possible steps for the SOP (Standard Operation Procedure) of the given video.
What is the [SUBJECT] doing?
""",
"""There are [STEP] possible steps for the SOP (Standard Operation Procedure) of the given video.
Which step is the [SUBJECT] performing?
""",
"What actions does the [SUBJECT] take?",
"What is the [SUBJECT] doing?",
"Which action is the [SUBJECT] performing?"
]