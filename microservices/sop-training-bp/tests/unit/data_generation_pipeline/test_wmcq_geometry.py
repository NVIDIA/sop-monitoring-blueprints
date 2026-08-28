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
Unit tests for vlm_aug/config_to_wmcq.py window geometry.

WMCQ exists to make training clips have the same geometry as the fixed-length windows a
sliding-window evaluation produces. Everything worth testing here is a geometric invariant:

  * a positive window fully CONTAINS its key-step (so the padding is real surrounding footage)
  * a tiled window lies INSIDE its key-step (so the clip stays exactly one window long)
  * a negative window touches NO key-step
  * under tiling, clip duration carries no information about the class

That last one is the whole reason the tiling options exist: if only one action is ever longer
than a window, the enlarge path makes every long clip that action, and the model can separate
the class on duration alone -- a shortcut that does not exist at inference time.
"""

import argparse
import math

import pytest

import vlm_aug.config_to_wmcq as wmcq
from vlm_aug.utils import const

WINDOW = 3.0
DURATION = 300.0


def _args(**overrides):
    base = dict(window=WINDOW, tile_long=False, tile_passes=False, enlarge_pad=1.0)
    base.update(overrides)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------------------
# window_offsets: the window must contain the key-step
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("start,end", [(100.0, 101.0), (100.0, 103.0), (50.0, 52.5)])
@pytest.mark.parametrize("k", [1, 2, 4, 7])
def test_window_offsets_contain_the_keystep(start, end, k):
    offsets = wmcq.window_offsets(start, end, WINDOW, DURATION, k)
    assert offsets, "at least one window must always be produced"
    # Exactly k passes, always. tile_passes defines one exposure as 100% coverage and
    # sets k so every key-step gets the same number of passes whatever its length; a
    # pinned interval repeats the one legal position rather than returning fewer.
    assert len(offsets) == k, f"expected {k} passes, got {len(offsets)}: {offsets}"
    for ws in offsets:
        assert ws <= start + 1e-6, f"window starts after the key-step: {ws} > {start}"
        assert ws + WINDOW >= end - 1e-6, f"window ends before the key-step: {ws + WINDOW} < {end}"
        assert 0.0 <= ws <= DURATION - WINDOW


@pytest.mark.parametrize("k", [2, 4, 7])
def test_window_offsets_give_k_distinct_windows_when_the_keystep_can_move(k):
    """The point of --variants: the key-step sits at a DIFFERENT offset in each window."""
    offsets = wmcq.window_offsets(100.0, 101.0, WINDOW, DURATION, k)
    assert len(offsets) == len(set(offsets)) == k


@pytest.mark.parametrize("fn,start,end,duration", [
    (wmcq.window_offsets, 100.0, 103.0, 300.0),  # key-step is exactly one window long
    (wmcq.window_offsets, 297.0, 300.0, 300.0),  # window-long key-step ending at the video end
    (wmcq.window_offsets, 299.0, 300.0, 300.0),  # short trailing key-step ending at the video end
    (wmcq.tiled_offsets,  298.0, 310.0, 300.0),  # tiling an annotation that runs past the end
])
def test_a_pinned_interval_still_yields_k_passes(fn, start, end, duration):
    """Only one window position is legal, and all k passes take it -- k clips, not one.

    An earlier version deduplicated here, reasoning that identical clips add no view
    diversity. That silently broke the invariant tile_passes exists to hold: every
    key-step gets the same number of passes regardless of length. Deduplicating removed
    that for one subset only -- key-steps exactly `window` long, whose geometry matches
    an eval window exactly and which are therefore the best-matched examples the stage
    cuts. On the 11-video training split it cost 21 of 272 positives, and the resulting
    630-sample dataset no longer matched the 680 the published recipe was validated on.

    The second and third cases are the trailing-action shape: a key-step starting within a
    window of the video end whose annotated end is clamped to the video duration, which is
    what annotation tools that clamp to video length produce routinely.
    """
    offsets = fn(start, end, WINDOW, duration, 4)
    assert len(offsets) == 4, f"expected 4 passes, got {offsets}"
    assert len(set(offsets)) == 1, f"expected one distinct position, got {set(offsets)}"


def test_offsets_report_when_passes_land_on_the_same_clip(caplog):
    """Informational, not a warning: repeating the clip is the intended behaviour."""
    with caplog.at_level("INFO"):
        wmcq.window_offsets(100.0, 103.0, WINDOW, DURATION, 4)
    assert "1 distinct window position(s) for the 4 passes requested" in caplog.text


def test_exactly_window_length_keystep_is_not_underexposed():
    """Regression guard for the dedup defect, stated as the recipe invariant.

    A key-step exactly `window` long must get the same number of passes as one that is
    shorter, otherwise `variants=N` silently means different things per key-step.
    """
    pinned = wmcq.window_offsets(100.0, 103.0, WINDOW, DURATION, 4)   # L == window
    movable = wmcq.window_offsets(100.0, 101.0, WINDOW, DURATION, 4)  # L <  window
    assert len(pinned) == len(movable) == 4


def test_window_offsets_clamps_at_the_start_of_the_video():
    offsets = wmcq.window_offsets(0.5, 1.5, WINDOW, DURATION, 4)
    assert all(ws >= 0.0 for ws in offsets)


def test_window_offsets_clamps_at_the_end_of_the_video():
    offsets = wmcq.window_offsets(DURATION - 1.5, DURATION - 0.5, WINDOW, DURATION, 4)
    assert all(ws + WINDOW <= DURATION + 1e-6 for ws in offsets)


def test_window_offsets_survives_a_video_shorter_than_the_window():
    assert wmcq.window_offsets(0.2, 0.8, WINDOW, 2.0, 4) == [0.0]


@pytest.mark.parametrize("start,end,duration", [
    (299.0, 301.0, 300.0),   # annotation end runs past the video duration
    (10.0, 20.0, 300.0),     # key-step longer than the window it is asked to fit in
])
def test_window_offsets_falls_back_to_one_window_when_containment_is_impossible(
        start, end, duration):
    """No window position contains the key-step, so exactly one best-effort clip is cut.

    Returning `k` copies instead would duplicate the same clip and add no view diversity,
    which is the only reason to cut more than one -- so one is right. What matters is that
    it is not silent: the sample count drops below --variants for this key-step and nothing
    else in the output would reveal why.
    """
    offsets = wmcq.window_offsets(start, end, WINDOW, duration, 4)
    assert len(offsets) == 1
    assert 0.0 <= offsets[0] <= duration - WINDOW


def test_window_offsets_warns_when_it_undercuts_the_requested_count(caplog):
    with caplog.at_level("WARNING"):
        wmcq.window_offsets(299.0, 301.0, WINDOW, 300.0, 4)
    assert "cutting 1 window instead of 4" in caplog.text


def test_window_offsets_does_not_warn_when_only_one_window_was_asked_for(caplog):
    with caplog.at_level("WARNING"):
        wmcq.window_offsets(299.0, 301.0, WINDOW, 300.0, 1)
    assert "instead of" not in caplog.text


# --------------------------------------------------------------------------------------
# tiled_offsets: the window must lie inside the key-step
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 5, 9])
def test_tiled_offsets_lie_inside_the_keystep(k):
    start, end = 200.0, 220.0
    for ws in wmcq.tiled_offsets(start, end, WINDOW, DURATION, k):
        assert ws >= start - 1e-6
        assert ws + WINDOW <= end + 1e-6


def test_tiled_offsets_fall_back_when_the_keystep_is_not_longer_than_the_window():
    """A key-step that does not exceed the window cannot be tiled; containment is used."""
    start, end = 100.0, 102.0
    assert wmcq.tiled_offsets(start, end, WINDOW, DURATION, 3) == \
        wmcq.window_offsets(start, end, WINDOW, DURATION, 3)


# --------------------------------------------------------------------------------------
# plan_windows: which geometry is chosen, and how many windows
# --------------------------------------------------------------------------------------

def test_short_keystep_uses_one_window_length():
    length, starts, geometry = wmcq.plan_windows(100.0, 102.0, DURATION, 4, _args())
    assert length == WINDOW
    assert geometry == const.GEOMETRY_MATCHED
    assert len(starts) == 4


def test_long_keystep_without_tiling_enlarges_the_window():
    length, starts, geometry = wmcq.plan_windows(200.0, 220.0, DURATION, 4, _args())
    assert abs(length - (20.0 + 1.0)) < 1e-6  # keystep length + enlarge_pad
    assert geometry == const.GEOMETRY_ENLARGED


def test_long_keystep_with_tiling_keeps_the_window_length():
    length, starts, geometry = wmcq.plan_windows(200.0, 220.0, DURATION, 4, _args(tile_long=True))
    assert length == WINDOW
    assert geometry == const.GEOMETRY_MATCHED
    assert len(starts) == 4  # plain tile_long does not scale the count


def test_tile_passes_reads_variants_as_full_passes():
    variants, start, end = 2, 200.0, 220.0
    expected = variants * int(math.ceil((end - start) / WINDOW))
    _, starts, _ = wmcq.plan_windows(start, end, DURATION, variants,
                                     _args(tile_long=True, tile_passes=True))
    assert len(starts) == expected == 14


def test_tile_passes_gives_every_class_the_same_number_of_exposures():
    """One "pass" must mean 100%% coverage whether the action is short or long.

    Without --tile-passes, N crops shows a 1s action N times over but barely covers a 20s
    action once, so the same --variants means very different amounts of exposure per class.
    """
    args = _args(tile_long=True, tile_passes=True)
    passes = 3

    _, short_starts, _ = wmcq.plan_windows(100.0, 101.0, DURATION, passes, args)
    _, long_starts, _ = wmcq.plan_windows(200.0, 220.0, DURATION, passes, args)

    short_coverage = len(short_starts) * WINDOW / WINDOW          # window >= key-step
    long_coverage = len(long_starts) * WINDOW / (220.0 - 200.0)
    assert abs(short_coverage - passes) < 1e-6
    assert long_coverage >= passes


# --------------------------------------------------------------------------------------
# the invariant the whole augmentation rests on
# --------------------------------------------------------------------------------------

def test_tiling_removes_the_duration_to_class_correlation():
    """With tiling on, every clip is the same length, so duration cannot identify the class."""
    keysteps = {1: (100.0, 101.0), 2: (150.0, 152.0), 3: (200.0, 220.0)}

    tiled = _args(tile_long=True, tile_passes=True)
    lengths = {a: wmcq.plan_windows(s, e, DURATION, 4, tiled)[0] for a, (s, e) in keysteps.items()}
    assert set(lengths.values()) == {WINDOW}

    enlarged = {a: wmcq.plan_windows(s, e, DURATION, 4, _args())[0]
                for a, (s, e) in keysteps.items()}
    # Without tiling the long class is separable on duration alone -- this is the bug.
    assert len(set(enlarged.values())) > 1
    assert enlarged[3] > enlarged[1]


# --------------------------------------------------------------------------------------
# overlaps_other_keystep: a positive window that swallowed its neighbour
# --------------------------------------------------------------------------------------

def test_sparse_keysteps_do_not_overlap_each_other():
    """The shape WMCQ is for: brief key-steps separated by long non-SOP stretches."""
    ks = [(92.0, 95.0), (220.0, 222.0), (400.0, 401.0)]
    for i, (s, e) in enumerate(ks):
        for ws in wmcq.window_offsets(s, e, WINDOW, 516.0, 4):
            assert not wmcq.overlaps_other_keystep(ws, WINDOW, i, ks)


def test_a_neighbour_further_away_than_the_slack_is_never_swallowed():
    """Slack is `window - keystep_length`; a gap at least that wide is safe.

    2s key-steps at 1s spacing: slack is 1s and the gap is 1s, so the window cannot
    reach past its own key-step's neighbourhood.
    """
    ks = [(10.0 + 3 * i, 12.0 + 3 * i) for i in range(6)]
    hits = sum(wmcq.overlaps_other_keystep(ws, WINDOW, i, ks)
               for i, (s, e) in enumerate(ks)
               for ws in wmcq.window_offsets(s, e, WINDOW, 60.0, 4))
    assert hits == 0


def test_a_neighbour_closer_than_the_slack_is_swallowed():
    """1s key-steps at 1s spacing: slack is 2s against a 1s gap, so windows overlap.

    This is the case the single-action label silently stops being true, and it is
    driven by SHORT key-steps -- a shorter action leaves the window more room to
    wander into its neighbour.
    """
    ks = [(10.0 + 2 * i, 11.0 + 2 * i) for i in range(6)]
    windows = [(i, ws) for i, (s, e) in enumerate(ks)
               for ws in wmcq.window_offsets(s, e, WINDOW, 60.0, 4)]
    hits = sum(wmcq.overlaps_other_keystep(ws, WINDOW, i, ks) for i, ws in windows)
    assert hits > 0.8 * len(windows)


def test_a_window_is_not_flagged_against_its_own_keystep():
    ks = [(100.0, 101.0)]
    for ws in wmcq.window_offsets(100.0, 101.0, WINDOW, DURATION, 4):
        assert not wmcq.overlaps_other_keystep(ws, WINDOW, 0, ks)


# --------------------------------------------------------------------------------------
# negative_slots: a negative must contain no key-step
# --------------------------------------------------------------------------------------

def test_negative_slots_avoid_every_keystep_and_its_margin():
    keysteps = [(50.0, 53.0), (120.0, 121.0), (200.0, 220.0)]
    margin = 0.5
    slots = wmcq.negative_slots(DURATION, WINDOW, keysteps, margin, 1.0)
    assert slots
    for ws in slots:
        for start, end in keysteps:
            overlaps = ws < end + margin and ws + WINDOW > start - margin
            assert not overlaps, f"negative window {ws} overlaps key-step {start}-{end}"


def test_negative_slots_use_the_requested_stride():
    slots = wmcq.negative_slots(60.0, WINDOW, [], 0.5, 2.0)
    assert slots == [round(x * 2.0, 3) for x in range(len(slots))]
    assert all(ws + WINDOW <= 60.0 + 1e-6 for ws in slots)


def test_negative_slots_empty_when_keysteps_cover_everything():
    assert wmcq.negative_slots(10.0, WINDOW, [(0.0, 10.0)], 0.5, 1.0) == []


@pytest.mark.parametrize("stride", [0.0, -1.0])
def test_negative_slots_rejects_a_non_advancing_stride(stride):
    """A non-positive stride never advances the scan position.

    Without this guard the loop spins forever appending slots until the worker runs out
    of memory. Nothing upstream times the subprocess out, so the augmentation would sit
    at "running" indefinitely rather than failing.
    """
    with pytest.raises(ValueError, match="stride must be positive"):
        wmcq.negative_slots(DURATION, WINDOW, [], 0.5, stride)


# --------------------------------------------------------------------------------------
# annotation parsing
# --------------------------------------------------------------------------------------

def test_parse_action_index_prefers_the_explicit_field():
    assert wmcq.parse_action_index({"action": 2, "description": "(3) mismatched"}) == 2


def test_parse_action_index_falls_back_to_the_description_prefix():
    assert wmcq.parse_action_index({"description": "(3) the worker checks the gasket"}) == 3


def test_parse_action_index_returns_none_when_unknowable():
    assert wmcq.parse_action_index({"description": "no index here"}) is None
