######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Unit tests for vlm_aug/config_to_dynamic_mcq.py multi-operator (concurrent-action) handling.

Concurrent chunks are named with a hyphen-joined action prefix (e.g. "12-13_video_1_2.mp4").
For a concurrent chunk the correct answer is the combined action string (all co-occurring
actions listed together, matching config_to_sequential_mcq), and the individual single actions
are valid incomplete distractors. Single-operator chunks are handled by the unchanged
assemble_anns path.
"""

import argparse
import re

import pytest

import vlm_aug.config_to_dynamic_mcq as dmcq
from vlm_aug.utils import const

ACTIONS = [f"action {i}" for i in range(1, 15)]
NON_SOP = 11


def _make_video_dir(root, chunk_names):
    vid = "clipdir"
    d = root / vid
    d.mkdir(parents=True)
    for name in chunk_names:
        (d / name).write_bytes(b"")
    return vid


def _run(root, vid, out, exclude_actions=None, num_pos=1, num_neg=1,
         num_hard_pos=0, num_hard_neg=0, hard_pos_mode=None, hard_neg_mode=None, confusion_map=None):
    args = argparse.Namespace(exclude_actions=exclude_actions or [], choices=ACTIONS)
    return dmcq.process_chunk(
        str(root), vid, "mp4", 3, 5, NON_SOP,
        num_pos, num_neg, num_hard_pos, num_hard_neg,
        hard_pos_mode or [], hard_neg_mode or [], confusion_map, 42, str(out), args,
    )


def _options(qa):
    return re.findall(r"\(\d+\) (.+)", qa[const.CONV][0][const.VALUE])


def _answer_text(qa):
    return re.match(r"\(\d+\) (.+)", qa[const.CONV][1][const.VALUE]).group(1)


def test_concurrent_chunk_does_not_crash(tmp_path):
    vid = _make_video_dir(tmp_path / "vroot",
                          ["01_clipdir_1_1.mp4", "12-13_clipdir_1_2.mp4", "13_clipdir_1_3.mp4"])
    assert _run(tmp_path / "vroot", vid, tmp_path / "out")


def test_concurrent_positive_answer_is_combined(tmp_path):
    vid = _make_video_dir(tmp_path / "vroot", ["12-13_clipdir_1_2.mp4"])
    anns = _run(tmp_path / "vroot", vid, tmp_path / "out", num_pos=2)
    combined = f"{ACTIONS[11]} {ACTIONS[12]}"
    positives = [a for a in anns if a[const.META][const.POS_OR_NEG] == "pos"]
    assert positives
    for a in positives:
        assert _answer_text(a) == combined
        assert a[const.META][const.GT_ACTION] == combined


def test_concurrent_hard_positive_offers_single_actions_as_distractors(tmp_path):
    vid = _make_video_dir(tmp_path / "vroot", ["12-13_clipdir_1_2.mp4"])
    anns = _run(tmp_path / "vroot", vid, tmp_path / "out",
                num_pos=0, num_neg=0, num_hard_pos=1, hard_pos_mode=["confusion"],
                confusion_map={12: [5], 13: [6]})
    combined = f"{ACTIONS[11]} {ACTIONS[12]}"
    hard = [a for a in anns if a[const.META][const.POS_OR_NEG] == "hp"]
    assert hard
    for a in hard:
        opts = _options(a)
        assert _answer_text(a) == combined            # correct answer = the full combination
        assert ACTIONS[11] in opts and ACTIONS[12] in opts   # each single action is an (incomplete) distractor


def test_concurrent_non_hard_excludes_component_actions_as_distractors(tmp_path):
    vid = _make_video_dir(tmp_path / "vroot", ["12-13_clipdir_1_2.mp4"])
    anns = _run(tmp_path / "vroot", vid, tmp_path / "out", num_pos=3, num_neg=3)
    regular = [a for a in anns if a[const.META][const.POS_OR_NEG] in ("pos", "neg")]
    assert regular
    for a in regular:
        opts = _options(a)
        assert ACTIONS[11] not in opts and ACTIONS[12] not in opts


def test_exclude_action_filters_concurrent_chunk(tmp_path):
    vid = _make_video_dir(tmp_path / "vroot", ["01_clipdir_1_1.mp4", "12-13_clipdir_1_2.mp4"])
    anns = _run(tmp_path / "vroot", vid, tmp_path / "out", exclude_actions=[13])
    gts = {a[const.META][const.GT_ACTION] for a in anns}
    assert f"{ACTIONS[11]} {ACTIONS[12]}" not in gts
    assert ACTIONS[0] in gts


def test_concurrent_question_uses_concurrent_subject(tmp_path):
    vid = _make_video_dir(tmp_path / "vroot", ["12-13_clipdir_1_2.mp4"])
    anns = _run(tmp_path / "vroot", vid, tmp_path / "out", num_pos=2)
    assert anns
    subject = const.DEFAULT_SUBJECT_CONCURRENT
    for a in anns:
        question = a[const.CONV][0][const.VALUE]
        assert subject in question                            # "workers", not the single-op "operator"
        assert f"the {const.DEFAULT_SUBJECT} " not in question
        assert f"is the {subject} " not in question           # plural template: no singular-verb agreement
        assert f"does the {subject} " not in question


def test_single_action_chunk_unchanged(tmp_path):
    vid = _make_video_dir(tmp_path / "vroot", ["03_clipdir_1_1.mp4"])
    anns = _run(tmp_path / "vroot", vid, tmp_path / "out")
    assert {a[const.META][const.GT_ACTION] for a in anns} == {ACTIONS[2]}
    for a in anns:
        if a[const.META][const.POS_OR_NEG] == "pos":
            assert _answer_text(a) == ACTIONS[2]       # single-op answer is the single action
        if a[const.META][const.POS_OR_NEG] == "neg":
            assert ACTIONS[2] not in _options(a)       # true action is never offered in a negative
        assert const.DEFAULT_SUBJECT in a[const.CONV][0][const.VALUE]   # single-op keeps "operator"
