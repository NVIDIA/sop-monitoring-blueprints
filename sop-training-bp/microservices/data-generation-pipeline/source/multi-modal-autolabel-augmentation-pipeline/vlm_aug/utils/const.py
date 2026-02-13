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


import os


_THIS_SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
_THIS_SCRIPT_ROOT = os.path.abspath(_THIS_SCRIPT_ROOT)

API_BASE_URL = "https://integrate.api.nvidia.com/v1"
API_LOCAL_URL = "http://0.0.0.0:8000/v1"
PROMPTS_ROOT = os.path.join(_THIS_SCRIPT_ROOT, "..", "prompts")
LINE_BREAK = "\n"
VIDEO_ACTION_SEP = "_"
CONV = "conversations"
VALUE = "value"
VIDEO = "video"
ACTION_JSON_KEY = "actions"
DEFAULT_SUBJECT = "operator"
STEP_TOKEN = "[STEP]"
SUBJECT_TOKEN = "[SUBJECT]"
QUESTION = "question"
CHOICES = "choices"

# GQA
ACTION = "action"
GQA2GQAS = "gqa_to_gqas"
GOLDEN_GQA2GQAS = "golden_gqa_to_gqas"

# BCQ
YES = "y"
NO = "n"
BCQ = "config_to_bcq"

# MCQ
MCQ = "config_to_sequential_mcq"

# DMCQ
DMCQ = "config_to_dynamic_mcq"
ADJACENT = "adjacent"
CONFUSION = "confusion"
HARD_MODES = [ADJACENT, CONFUSION]
GT_ACTION = "gt_action"
POS_OR_NEG = "pos_or_neg"
HARD_MODE = "hard_mode"
NUM_OPTIONS = "num_options"

# DS
DS = "config_to_dynamic_shuffling"
DS_HARD_SAMPLING_MODE = ["front", "end", "random"]
SOURCE_VIDEOS = "source_videos"
TOTAL_FRAMES = "total_frames"
IS_HARD = "is_hard"

# EN
EN = "config_to_extra_negative"

# dynamic sampling
META = "meta"
MIN_FRAMES = "min_frames"
MAX_FRAMES = "max_frames"
FRAME_COUNTS = "frame_cnts"
DYNAMIC_SAMPLE = "dynamic_sample"
