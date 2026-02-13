######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
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