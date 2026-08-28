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
Unit tests for overlapping-window chunking and its smoothing.

A short action landing on a fixed chunk grid gets split across two chunks and diluted in
both. Overlapping windows fix that by guaranteeing at least one window contains the action
whole -- at the cost that the same instant is now predicted several times, so the raw
per-window labels are no longer a sequence. Smoothing votes them back down.

The properties worth holding:
  * with stride == window the windows are exactly the non-overlapping grid, so the new
    code path is a strict generalisation of the old one;
  * every instant of the video is covered;
  * smoothing returns non-overlapping, time-ordered, gap-free segments the existing
    sequence scorer can read;
  * each of the three filters (tie, min_vote, min_seg) actually changes the output.
"""

import pytest

from utils.e2e_eval_utils import (
    chunk_key_start_sec,
    get_golden_actions,
    map_chunks_to_ground_truth,
    chunk_keys_to_boundaries,
    get_non_sop_action,
    overlapping_chunk_windows,
    smooth_overlapping_windows,
    uniform_chunk_boundaries,
)

NON_SOP = 4


def _segments(smoothed):
    """[(action, start, end), ...] ordered by start."""
    out = []
    for key, text in smoothed.items():
        start, end = key.strip("[]").split("s-")
        out.append((int(text[1:text.index(")")]), float(start), float(end.rstrip("s"))))
    return sorted(out, key=lambda seg: seg[1])


def _windows_saying(spans, duration, window, stride, action_for):
    """Per-window VLM output built by a caller-supplied action_for(start, end)."""
    return {f"[{s:.2f}s-{e:.2f}s]": f"({action_for(s, e)}) text"
            for s, e in overlapping_chunk_windows(duration, window, stride)}


# ---------------------------------------------------------------------------------
# windowing
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("duration", [12.0, 10.0, 7.5, 33.3])
@pytest.mark.parametrize("window", [2.0, 3.0, 5.0])
def test_stride_equal_to_window_is_the_non_overlapping_grid(duration, window):
    """The new path must reduce to the old one, or it is not a generalisation.

    Durations that are NOT a whole number of windows are the ones that matter: the
    non-overlapping path keeps the short final chunk, and an overlapping path that
    dropped it would quietly stop evaluating the end of every such video.
    """
    windows = overlapping_chunk_windows(duration, window, window)
    boundaries = uniform_chunk_boundaries(duration, window)
    assert [tuple(w) for w in windows] == list(zip(boundaries[:-1], boundaries[1:]))


def test_the_short_trailing_chunk_is_kept():
    windows = overlapping_chunk_windows(10.0, 3.0, 3.0)
    assert windows[-1] == [9.0, 10.0], "the final partial window must not be dropped"


def test_windows_advance_by_exactly_one_stride():
    windows = overlapping_chunk_windows(10.0, 3.0, 1.0)
    starts = [w[0] for w in windows]
    assert starts == [round(float(i), 3) for i in range(len(starts))]


@pytest.mark.parametrize("stride", [0.5, 1.0, 1.5, 3.0])
def test_every_instant_is_covered_by_at_least_one_window(stride):
    """Includes stride == window, the boundary case: still gap-free.

    Beyond it there ARE gaps, which is why a stride wider than the window is
    rejected upstream -- an uncovered instant collects no votes and is emitted as
    non-SOP, reporting "nothing happening" over footage never evaluated.
    """
    duration, window = 10.0, 3.0
    windows = overlapping_chunk_windows(duration, window, stride)
    for t in [x * 0.25 for x in range(int(duration / 0.25))]:
        assert any(s <= t < e for s, e in windows), f"t={t} uncovered at stride={stride}"


def test_a_stride_wider_than_the_window_would_leave_gaps():
    """Documents why the guard exists, by showing the damage it prevents."""
    windows = overlapping_chunk_windows(20.0, 3.0, 5.0)
    uncovered = [t for t in [x * 0.5 for x in range(40)]
                 if not any(s <= t < e for s, e in windows)]
    assert uncovered, "expected gaps; if this ever passes the guard can be relaxed"


def test_stride_wider_than_the_window_is_rejected_by_main(tmp_path):
    anno, actions = _inputs(tmp_path)
    args = _args(tmp_path, anno, actions, chunk_length_sec=3.0, stride_sec=5.0)
    with pytest.raises(ValueError, match="leaves gaps"):
        _run_main(args)


def test_no_window_runs_past_the_end_of_the_video():
    for stride in (0.5, 1.0, 2.5, 3.0):
        for s, e in overlapping_chunk_windows(10.0, 3.0, stride):
            assert 0.0 <= s < e <= 10.0


def test_a_short_action_is_contained_whole_by_some_window():
    """The reason overlapping windows exist: a fixed grid can split a brief action."""
    action_start, action_end = 4.4, 6.1          # straddles the 3s grid boundary at 6.0
    grid = uniform_chunk_boundaries(12.0, 3.0)
    grid_pairs = list(zip(grid[:-1], grid[1:]))
    assert not any(s <= action_start and e >= action_end for s, e in grid_pairs)
    windows = overlapping_chunk_windows(12.0, 3.0, 1.0)
    assert any(s <= action_start and e >= action_end for s, e in windows)


@pytest.mark.parametrize("window,stride,duration", [(0, 1, 10), (3, 0, 10), (3, 1, 0), (3, -1, 10)])
def test_invalid_geometry_is_rejected(window, stride, duration):
    with pytest.raises(ValueError):
        overlapping_chunk_windows(duration, window, stride)


# ---------------------------------------------------------------------------------
# smoothing
# ---------------------------------------------------------------------------------

def test_smoothing_returns_contiguous_non_overlapping_segments():
    """The scorer reads these as ordinary chunks, so they must tile the timeline."""
    windows = _windows_saying(None, 12.0, 3.0, 1.0,
                              lambda s, e: 2 if (s < 8 and e > 5) else NON_SOP)
    segments = _segments(smooth_overlapping_windows(windows, NON_SOP))
    assert segments[0][1] == 0.0
    for (_, _, prev_end), (_, next_start, _) in zip(segments, segments[1:]):
        assert prev_end == next_start, "segments must be gap-free and non-overlapping"


def test_an_action_seen_by_most_windows_survives():
    windows = _windows_saying(None, 12.0, 3.0, 1.0,
                              lambda s, e: 2 if (s < 8 and e > 5) else NON_SOP)
    actions = {a for a, _, _ in _segments(smooth_overlapping_windows(windows, NON_SOP))}
    assert actions == {NON_SOP, 2}


def test_a_tie_resolves_to_non_sop():
    """An instant the windows disagree about is not evidence for any action."""
    windows = {"[0.00s-4.00s]": "(2) a", "[0.00s-4.00s] ": f"({NON_SOP}) n"}
    assert _segments(smooth_overlapping_windows(windows, NON_SOP)) == [(NON_SOP, 0.0, 4.0)]


def test_min_vote_suppresses_an_action_backed_by_too_few_windows():
    windows = {"[0.00s-6.00s]": f"({NON_SOP}) n", "[6.00s-12.00s]": f"({NON_SOP}) n",
               "[10.00s-16.00s]": "(3) a", "[16.00s-22.00s]": f"({NON_SOP}) n"}
    kept = _segments(smooth_overlapping_windows(windows, NON_SOP, min_seg_sec=0.5, min_vote=1))
    dropped = _segments(smooth_overlapping_windows(windows, NON_SOP, min_seg_sec=0.5, min_vote=2))
    assert 3 in {a for a, _, _ in kept}
    assert 3 not in {a for a, _, _ in dropped}


def test_min_seg_sec_drops_a_brief_blip():
    windows = {"[0.00s-5.00s]": f"({NON_SOP}) n", "[5.00s-6.00s]": "(1) a",
               "[6.00s-12.00s]": f"({NON_SOP}) n"}
    kept = _segments(smooth_overlapping_windows(windows, NON_SOP, min_seg_sec=0.5))
    dropped = _segments(smooth_overlapping_windows(windows, NON_SOP, min_seg_sec=2.0))
    assert 1 in {a for a, _, _ in kept}
    assert 1 not in {a for a, _, _ in dropped}


def test_min_seg_sec_never_drops_non_sop():
    """Only action segments are length-filtered; non-SOP is what they collapse into."""
    windows = {"[0.00s-6.00s]": "(2) a", "[6.00s-6.50s]": f"({NON_SOP}) n",
               "[6.50s-12.00s]": "(2) a"}
    assert {a for a, _, _ in _segments(
        smooth_overlapping_windows(windows, NON_SOP, min_seg_sec=5.0))} <= {NON_SOP, 2}


def test_windows_with_no_parsable_action_are_ignored():
    windows = {"[0.00s-3.00s]": "no action id here", "[1.00s-4.00s]": "(2) a"}
    assert 2 in {a for a, _, _ in _segments(smooth_overlapping_windows(windows, NON_SOP))}


def test_empty_input_yields_no_segments():
    assert smooth_overlapping_windows({}, NON_SOP) == {}


def test_without_a_non_sop_action_the_input_is_returned_untouched():
    """The primitive itself does not invent an id; main() refuses the run instead."""
    windows = {"[0.00s-3.00s]": "(2) a"}
    assert smooth_overlapping_windows(windows, None) == windows


@pytest.mark.parametrize("window,stride,expected_bin", [
    (3.0, 1.0, 1.0), (3.0, 0.5, 0.5), (2.5, 1.0, 0.5), (3.0, 0.4, 0.2), (4.0, 1.5, 0.5),
])
def test_bin_size_is_derived_from_the_window_geometry(window, stride, expected_bin):
    """Not a tunable: a window covers [iS, iS+W), so the set of windows covering an
    instant changes only at a multiple of gcd(W, S). Bins of that size are exact --
    finer ones are identical copies, coarser ones straddle a change point and mis-vote.

    Asserted through the output: every emitted boundary must land on the derived bin.
    """
    windows = _windows_saying(None, 20.0, window, stride,
                              lambda s, e: 2 if (s < 8 and e > 5) else NON_SOP)
    for _, start, end in _segments(smooth_overlapping_windows(windows, NON_SOP)):
        for edge in (start, end):
            assert abs(round(edge / expected_bin) * expected_bin - edge) < 1e-9, (
                f"{edge} is not a multiple of the derived bin {expected_bin}")


def test_binning_does_not_drift_on_float_boundaries():
    """int(0.3 / 0.1) is 2, not 3. Binning in floating point shifts a window's votes
    by one bin at ordinary timestamps; the implementation works in integer milliseconds."""
    windows = {"[0.00s-0.30s]": "(2) a", "[0.10s-0.40s]": "(2) a", "[0.20s-0.50s]": f"({NON_SOP}) n"}
    segments = _segments(smooth_overlapping_windows(windows, NON_SOP, min_seg_sec=0.0))
    assert segments == [(2, 0.0, 0.3), (NON_SOP, 0.3, 0.5)]


def test_a_zero_length_window_does_not_break_binning():
    windows = {"[1.00s-1.00s]": "(2) a", "[0.00s-2.00s]": f"({NON_SOP}) n"}
    assert _segments(smooth_overlapping_windows(windows, NON_SOP, min_seg_sec=0.0)) == [(NON_SOP, 0.0, 2.0)]


# ---------------------------------------------------------------------------------
# non-SOP discovery
# ---------------------------------------------------------------------------------

def test_non_sop_action_is_read_from_actions_json(tmp_path):
    import json
    p = tmp_path / "actions.json"
    p.write_text(json.dumps({"actions": ["(1) a", "(2) b", "(4) none"],
                             "actions_can_be_skipped": ["(4) none"]}))
    assert get_non_sop_action(str(p)) == 4


def test_non_sop_action_is_none_when_undeclared(tmp_path):
    import json
    p = tmp_path / "actions.json"
    p.write_text(json.dumps({"actions": ["(1) a"]}))
    assert get_non_sop_action(str(p)) is None
    assert get_non_sop_action(None) is None


# ---------------------------------------------------------------------------------
# main() dispatch
#
# These exist because the first version of this change spliced the stride validation
# into the middle of the uniform branch, orphaning the run_uniform_stage call after a
# raise. Uniform chunking -- the pre-existing path, not just the new one -- then died
# with NameError at stage 2. Nothing in this file exercised main(), so only the
# repository's own dispatch tests caught it.
# ---------------------------------------------------------------------------------

import json
from types import SimpleNamespace
from unittest.mock import patch


def _inputs(tmp_path):
    anno = {"vid1.mp4": [
        {"description": "(1) one", "start_timestamp": 0.0, "end_timestamp": 5.0},
        {"description": "(2) two", "start_timestamp": 5.0, "end_timestamp": 10.0},
    ]}
    (tmp_path / "anno.json").write_text(json.dumps(anno))
    (tmp_path / "actions.json").write_text(json.dumps(
        {"actions": ["one.", "two.", "none."], "actions_can_be_skipped": ["(3) none."]}))
    return str(tmp_path / "anno.json"), str(tmp_path / "actions.json")


def _args(tmp_path, anno_path, actions_path, **overrides):
    defaults = dict(
        vlm_model_path="/fake/vlm", asset_root=str(tmp_path / "assets"),
        output_dir=str(tmp_path / "out"), video_dir=str(tmp_path / "videos"),
        video_ext="mp4", fps=8, temperature=0.0, backend="vllm", resolution_config=None,
        tensor_parallel_size=0, chunking_algorithm="uniform", chunk_length_sec=5.0,
        ddm_checkpoint_path=None, ddm_resolution=224, ddm_frames_per_side=5,
        score_threshold=0.5, nms_sec=0.0, ddm_batch_size=8, frames_per_segment_hint=256,
        anno_json_path=anno_path, actions_json_path=actions_path,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _run_main(args, accuracy_spy=None):
    from sop import sop_e2e_eval
    fake_temporal = {"vid1.mp4": {"boundaries": [0.0, 5.0, 10.0],
                                  "metric": {"F1": 0.9, "Precision": 0.9, "Recall": 0.9,
                                             "True Positive": 2, "False Positive": 0,
                                             "False Negative": 0}},
                     "avg_f1": 0.9, "avg_precision": 0.9, "avg_recall": 0.9}
    fake_vlm = {"vid1.mp4": {"[0.00s-3.00s]": "(1) one", "[1.00s-4.00s]": "(1) one",
                             "[5.00s-8.00s]": "(2) two", "[6.00s-9.00s]": "(2) two"}}
    fake_seq = {"sequence_accuracy": 1.0, "action_accuracy": 1.0, "total_videos": 1,
                "total_videos_dist_0": 1, "total_actions": 2, "wrong": 0, "duplicate": 0,
                "missing": 0, "videos_with_error": [], "per_video": []}
    with patch.object(sop_e2e_eval, "run_uniform_stage", return_value=fake_temporal) as m_uniform, \
         patch.object(sop_e2e_eval, "run_ddm_stage", return_value=fake_temporal) as m_ddm, \
         patch.object(sop_e2e_eval, "run_vlm_stage", return_value=fake_vlm), \
         patch.object(sop_e2e_eval, "compute_e2e_accuracy",
                      side_effect=accuracy_spy) if accuracy_spy else \
         patch.object(sop_e2e_eval, "compute_e2e_accuracy",
                      return_value={"overall_accuracy": 1.0, "per_action": {}}), \
         patch("utils.e2e_eval_utils.evaluate_action_sequences", return_value=fake_seq), \
         patch.object(sop_e2e_eval, "extract_golden_boundaries", return_value={"vid1.mp4": [0.0, 5.0, 10.0]}), \
         patch("utils.eval_utils.extract_mcq_data", return_value=("prompt", ["(1) one", "(2) two"])):
        sop_e2e_eval.main(args)
    return m_uniform, m_ddm


def test_uniform_without_stride_still_runs_the_uniform_stage(tmp_path):
    """The pre-existing path. This is the one the first version of this change broke."""
    anno, actions = _inputs(tmp_path)
    m_uniform, m_ddm = _run_main(_args(tmp_path, anno, actions))
    m_uniform.assert_called_once()
    m_ddm.assert_not_called()


def test_uniform_with_stride_runs_the_uniform_stage_and_writes_smoothed_output(tmp_path):
    anno, actions = _inputs(tmp_path)
    args = _args(tmp_path, anno, actions, stride_sec=1.0, smooth_min_seg_sec=2.0,
                 smooth_min_vote=1)
    m_uniform, m_ddm = _run_main(args)
    m_uniform.assert_called_once()
    m_ddm.assert_not_called()
    smoothed = tmp_path / "out" / "outputs_action_recognition" / "video_name_to_output_text_smoothed.json"
    assert smoothed.exists(), "overlapping mode must write the smoothed predictions"
    assert json.loads(smoothed.read_text())["vid1.mp4"], "smoothed output must be non-empty"


def test_args_object_without_stride_sec_at_all_still_works(tmp_path):
    """main() is also driven by hand-built arg objects that predate these options."""
    anno, actions = _inputs(tmp_path)
    args = _args(tmp_path, anno, actions)
    assert not hasattr(args, "stride_sec")
    m_uniform, _ = _run_main(args)
    m_uniform.assert_called_once()


def test_stride_with_ddm_chunking_is_rejected(tmp_path):
    anno, actions = _inputs(tmp_path)
    args = _args(tmp_path, anno, actions, chunking_algorithm="ddm",
                 ddm_checkpoint_path="/fake/last.ckpt", chunk_length_sec=None, stride_sec=1.0)
    with pytest.raises(ValueError, match="uniform"):
        _run_main(args)


def test_non_positive_stride_is_rejected_before_any_stage_runs(tmp_path):
    anno, actions = _inputs(tmp_path)
    args = _args(tmp_path, anno, actions, stride_sec=0.0)
    with pytest.raises(ValueError, match="stride-sec"):
        _run_main(args)


# ---------------------------------------------------------------------------------
# chunk ordering and chunk-level accuracy
# ---------------------------------------------------------------------------------

def test_chunk_keys_sort_by_start_time_not_lexicographically():
    """"[10.00s-13.00s]" sorts before "[2.00s-5.00s]" as a string.

    compute_e2e_accuracy pairs the i-th chunk with the i-th ground-truth action, so a
    lexicographic order pairs predictions against entirely different moments once a
    video passes ten chunks. Overlapping windows hit this constantly.
    """
    keys = [f"[{s:.2f}s-{s + 3:.2f}s]" for s in range(0, 40, 3)]
    assert sorted(keys) != keys, "the failure mode this guards must actually exist"
    assert sorted(keys, key=chunk_key_start_sec) == keys


def test_boundaries_reconstructed_from_contiguous_chunk_keys():
    smoothed = {"[0.00s-4.00s]": "(4) ", "[4.00s-9.00s]": "(2) ", "[9.00s-20.00s]": "(4) "}
    assert chunk_keys_to_boundaries(smoothed) == [0.0, 4.0, 9.0, 20.0]


def test_boundaries_from_unordered_or_unparsable_keys():
    assert chunk_keys_to_boundaries({"[4.00s-9.00s]": "x", "[0.00s-4.00s]": "y"}) == [0.0, 4.0, 9.0]
    assert chunk_keys_to_boundaries({"nonsense": "x"}) == []
    assert chunk_keys_to_boundaries({}) == []


def test_chunk_accuracy_is_computed_on_the_smoothed_segments(tmp_path):
    """With a stride there are many more windows than stage-1 chunk slots, so scoring the
    raw windows pairs a prefix of them against unrelated ground truth. The run must score
    the smoothed segments instead."""
    anno, actions = _inputs(tmp_path)
    args = _args(tmp_path, anno, actions, stride_sec=1.0)

    calls = []

    def spy(outputs, chunk_action_map, choices):
        calls.append(dict(outputs))
        return {"overall_accuracy": 1.0, "per_action": {}}

    _run_main(args, accuracy_spy=spy)

    assert calls, "compute_e2e_accuracy was never called"
    scored = calls[-1]                      # the overlapping path recomputes; last wins
    keys = list(next(iter(scored.values())).keys())
    raw_keys = {"[0.00s-3.00s]", "[1.00s-4.00s]", "[5.00s-8.00s]", "[6.00s-9.00s]"}
    assert not (set(keys) & raw_keys), (
        f"accuracy scored the raw overlapping windows: {keys}")
    starts = [chunk_key_start_sec(k) for k in keys]
    assert starts == sorted(starts), "smoothed segments must be time-ordered"


def test_compute_e2e_accuracy_pairs_chunks_in_time_order(tmp_path):
    """Guards the CALL SITE, not just the sort helper.

    compute_e2e_accuracy pairs the i-th chunk with the i-th ground-truth action. Build a
    14-chunk video where every chunk's response is exactly right for its own slot: any
    ordering other than by start time mis-pairs them and drops the accuracy below 1.0.
    Fourteen chunks is past the point where lexicographic order diverges ("[12.00s..."
    sorts before "[3.00s...").
    """
    from sop import sop_e2e_eval

    choices = [f"({i}) action {i}" for i in range(1, 15)]
    keys = [f"[{s:.2f}s-{s + 3:.2f}s]" for s in range(0, 42, 3)]
    assert sorted(keys) != keys, "need enough chunks for lexicographic order to diverge"

    vlm_outputs = {"vid1.mp4": {k: choices[i] for i, k in enumerate(keys)}}
    chunk_action_map = {"vid1.mp4": list(range(1, 15))}

    result = sop_e2e_eval.compute_e2e_accuracy(vlm_outputs, chunk_action_map, choices)
    assert result["overall_accuracy"] == 1.0, (
        "chunks were paired against the wrong ground-truth slots; "
        f"got {result['overall_accuracy']}"
    )


# ---------------------------------------------------------------------------------
# resolving the non-SOP action
# ---------------------------------------------------------------------------------

def _actions_json(tmp_path, skippable=True):
    import json
    p = tmp_path / "actions.json"
    body = {"actions": ["(1) a", "(2) b", "(3) c", "(4) none"]}
    if skippable:
        body["actions_can_be_skipped"] = ["(4) none"]
    p.write_text(json.dumps(body))
    return str(p)


def test_overlap_run_is_refused_when_no_non_sop_action_can_be_determined(tmp_path):
    """An evaluation that cannot be scored must not return numbers that look scored.

    Without a non-SOP action the windows cannot be voted down, so BOTH the sequence
    metrics and the chunk accuracy would be computed on raw overlapping windows. The
    real Version-D dataset declares no `actions_can_be_skipped`, so this is the
    default path there, not an exotic one.
    """
    anno, actions = _inputs(tmp_path)
    import json
    json.dump({"actions": ["(1) one", "(2) two", "(3) none"]}, open(actions, "w"))
    args = _args(tmp_path, anno, actions, stride_sec=1.0)
    with pytest.raises(ValueError, match="no non-SOP action could be determined"):
        _run_main(args)


def test_an_explicit_non_sop_action_overrides_the_actions_json(tmp_path):
    anno, actions = _inputs(tmp_path)
    args = _args(tmp_path, anno, actions, stride_sec=1.0, non_sop_action=3)
    m_uniform, _ = _run_main(args)
    m_uniform.assert_called_once()
    smoothed = tmp_path / "out" / "outputs_action_recognition" / "video_name_to_output_text_smoothed.json"
    assert smoothed.exists()


def test_an_explicit_non_sop_action_works_without_actions_can_be_skipped(tmp_path):
    """The case that matters: a dataset that never declares skippable actions."""
    anno, actions = _inputs(tmp_path)
    import json
    json.dump({"actions": ["(1) one", "(2) two", "(3) none"]}, open(actions, "w"))
    args = _args(tmp_path, anno, actions, stride_sec=1.0, non_sop_action=3)
    m_uniform, _ = _run_main(args)
    m_uniform.assert_called_once()


def test_cli_rejects_a_smooth_min_vote_below_one(tmp_path):
    anno, actions = _inputs(tmp_path)
    args = _args(tmp_path, anno, actions, stride_sec=1.0, non_sop_action=3, smooth_min_vote=0)
    with pytest.raises(ValueError, match="smooth-min-vote"):
        _run_main(args)


def test_max_model_len_is_omitted_unless_given():
    """Unset must build the exact vLLM kwargs the service has always used."""
    import inspect, re
    from sop import sop_e2e_eval
    src = inspect.getsource(sop_e2e_eval)
    block = src[src.index("llm_kwargs = dict("):src.index("llm = LLM(**llm_kwargs)")]
    assert "max_model_len" not in block.split("max_model_len = getattr")[0], (
        "max_model_len must not be in the base kwargs; it is added only when set")
    assert 'if max_model_len:' in block, "must be conditional"


def test_max_model_len_is_accepted_by_the_request_model():
    from validation.request_validation import E2eEvaluationRequest as R
    base = dict(training_job_id="t", val_dataset_id="d",
                chunking_algorithm="uniform", chunk_length_sec=3.0)
    # Defaults to a working value: deriving from the checkpoint is what fails.
    assert R(**base).max_model_len == 32768
    assert R(**base, max_model_len=8192).max_model_len == 8192
    # Explicit null restores vLLM's own derivation for anyone who wants it.
    assert R(**base, max_model_len=None).max_model_len is None


# ---------------------------------------------------------------------------------
# status reporting for a run with no DDM job
# ---------------------------------------------------------------------------------

def test_status_model_accepts_a_run_with_no_ddm_job():
    """A uniform-chunking run has no DDM job, and must still be observable.

    Typed as a required str, the status model rejected its own valid response: every
    poll for a DDM-less run returned HTTP 500, so a job that succeeded could never be
    seen to finish and the caller recorded a timeout instead.
    """
    from datetime import datetime
    from validation.request_validation import E2eEvaluationStatus
    now = datetime.now()
    s = E2eEvaluationStatus(
        eval_job_id="e", training_job_id="t", ddm_training_job_id=None,
        val_dataset_id="d", status="completed", created_at=now, updated_at=now,
    )
    assert s.ddm_training_job_id is None


def test_status_model_still_accepts_a_ddm_run():
    from datetime import datetime
    from validation.request_validation import E2eEvaluationStatus
    now = datetime.now()
    s = E2eEvaluationStatus(
        eval_job_id="e", training_job_id="t", ddm_training_job_id="ddm-123",
        val_dataset_id="d", status="completed", created_at=now, updated_at=now,
    )
    assert s.ddm_training_job_id == "ddm-123"


def test_status_survives_the_apps_own_construction_path():
    """app.py builds it with job.get("ddm_training_job_id", ""), but the key EXISTS with
    value None for a uniform run, so the "" default never fires and None reaches the model."""
    from datetime import datetime
    from validation.request_validation import E2eEvaluationStatus
    job = {"ddm_training_job_id": None}
    now = datetime.now()
    s = E2eEvaluationStatus(
        eval_job_id="e", training_job_id="t",
        ddm_training_job_id=job.get("ddm_training_job_id", ""),
        val_dataset_id="d", status="completed", created_at=now, updated_at=now,
    )
    assert s.ddm_training_job_id is None


# ---------------------------------------------------------------------------------
# chunk -> ground-truth ACTION ID mapping
# ---------------------------------------------------------------------------------

def test_chunks_map_to_action_ids_not_chunk_positions():
    """A real SOP video repeats its non-SOP action between key-steps, so the number of
    golden CHUNKS far exceeds the number of distinct ACTIONS. Returning the interval's
    position made per_action key on chunk number and index off the end of the action
    list -- "(?) unknown 9" on a four-action dataset."""
    # 9 chunks, actions alternating non-SOP(4) and a key-step
    golden_bdy = [0.0, 10.0, 12.0, 20.0, 22.0, 30.0, 32.0, 40.0, 42.0, 50.0]
    golden_actions = [4, 1, 4, 2, 4, 3, 4, 1, 4]
    pred_bdy = list(golden_bdy)

    mapped = map_chunks_to_ground_truth(
        pred_bdy, golden_bdy, len(golden_bdy) - 1, golden_actions=golden_actions)
    assert set(mapped) <= {1, 2, 3, 4}, f"ids must stay within the action list, got {set(mapped)}"
    assert mapped == golden_actions

    legacy = map_chunks_to_ground_truth(pred_bdy, golden_bdy, len(golden_bdy) - 1)
    assert max(legacy) == 9, "the positional fallback is what produced out-of-range ids"


def test_action_ids_are_read_from_the_annotation(tmp_path):
    import json
    anno = tmp_path / "anno.json"
    anno.write_text(json.dumps({"v.mp4": [
        {"description": "(4) none", "start_timestamp": 0.0, "end_timestamp": 10.0},
        {"description": "(2) two",  "start_timestamp": 10.0, "end_timestamp": 12.0},
        {"description": "(4) none", "start_timestamp": 12.0, "end_timestamp": 20.0},
    ]}))
    assert get_golden_actions(str(anno))["v.mp4"] == [4, 2, 4]


def test_a_predicted_chunk_takes_the_action_it_overlaps_most():
    golden_bdy = [0.0, 10.0, 12.0, 20.0]
    golden_actions = [4, 2, 4]
    # one predicted chunk sitting mostly inside the action-2 interval
    mapped = map_chunks_to_ground_truth(
        [10.5, 11.8], golden_bdy, 3, golden_actions=golden_actions)
    assert mapped == [2]


# ---------------------------------------------------------------------------------
# max_model_len on the by-action path
#
# The e2e path gained max_model_len so vLLM could start; the by-action path did not,
# which left backend="vllm" unusable there for exactly the same reason -- a checkpoint
# declaring a 262144 context sizes the KV cache past the card and the engine refuses to
# start before any inference runs. These tests pin the two halves of that plumbing.
# ---------------------------------------------------------------------------------

def test_by_action_max_model_len_is_omitted_unless_given():
    """Unset must build the exact vLLM kwargs the by-action path has always used."""
    import inspect
    from sop import sop_eval
    src = inspect.getsource(sop_eval)
    block = src[src.index("llm_kwargs = dict("):src.index("llm = LLM(**llm_kwargs)")]
    assert "max_model_len" not in block.split("max_model_len = getattr")[0], (
        "max_model_len must not be in the base kwargs; it is added only when set")
    assert "if max_model_len:" in block, "must be conditional"


def test_by_action_max_model_len_is_accepted_by_the_request_model():
    from validation.request_validation import EvaluationRequest as R
    base = dict(training_job_id="t", val_dataset_id="d")
    assert R(**base).max_model_len == 32768
    assert R(**base, max_model_len=8192).max_model_len == 8192
    assert R(**base, max_model_len=None).max_model_len is None


def test_default_max_model_len_is_the_one_the_docs_derive():
    """The shipped default must clear the measured ~20.4k requirement."""
    from validation.request_validation import EvaluationRequest, E2eEvaluationRequest
    for R, extra in ((EvaluationRequest, {}),
                     (E2eEvaluationRequest,
                      dict(chunking_algorithm="uniform", chunk_length_sec=3.0))):
        d = R(training_job_id="t", val_dataset_id="d", **extra).max_model_len
        assert d > 20359, f"{R.__name__} default {d} cannot fit a default request"


@pytest.mark.parametrize("skill", ["sop-by-action-eval", "sop-e2e-inference"])
def test_skill_client_forwards_max_model_len_in_both_modes(skill):
    """It lived in e2e_keys, so a by-action user's value was silently dropped.

    Silently: the client strips unknown keys without warning, so the request
    went out without it and vLLM failed with a KV-cache error that names no
    setting the user had touched.
    """
    import re, pathlib
    here = pathlib.Path(__file__).resolve()
    root = next((p for p in here.parents if (p / "agentic").is_dir()), None)
    assert root is not None, f"repo root not found above {here}"
    src = (root / "agentic/sop-agentic-ft/plugins/sop-evaluation-plugin/skills"
           / skill / "scripts/eval_api_client.py").read_text()
    common = re.search(r"common_keys = \((.*?)\)", src, re.S).group(1)
    e2e = re.search(r"e2e_keys = \((.*?)\)", src, re.S).group(1)
    assert "max_model_len" in common, f"{skill}: must be sent in BOTH modes"
    assert "max_model_len" not in e2e, f"{skill}: must not be duplicated in e2e_keys"


def test_by_action_cli_accepts_max_model_len():
    """The flag app.py appends must actually parse, and default to None."""
    import subprocess, sys, os
    eval_ms = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                           "microservices", "evaluation-ms")
    script = os.path.abspath(os.path.join(eval_ms, "sop", "sop_eval.py"))
    out = subprocess.run([sys.executable, script, "--help"],
                         capture_output=True, text=True, timeout=120)
    assert "--max-model-len" in out.stdout, out.stdout[-2000:]


@pytest.mark.asyncio
@pytest.mark.parametrize("given,expected", [(32768, True), (None, False)])
async def test_by_action_cmd_carries_max_model_len(tmp_path, given, expected):
    """The value on the request must reach sop_eval.py's argv -- and only when set."""
    import json as _json
    from datetime import datetime
    from unittest.mock import patch, MagicMock, AsyncMock
    from pathlib import Path
    from app import run_evaluation

    job = "eval-mml"
    outdir = tmp_path / job
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "inference_results.json").write_text(_json.dumps({"v": [[1, "(1) a"]]}))
    log = outdir / "log.txt"
    log.write_text("")

    proc = MagicMock()
    proc.pid = 1
    proc.wait = AsyncMock(return_value=0)
    proc.stdout = MagicMock(); proc.stdout.readline = AsyncMock(return_value=b"")
    proc.stderr = MagicMock(); proc.stderr.readline = AsyncMock(return_value=b"")

    cached = {"eval_job_id": job, "log_file_path": str(log), "process_pid": 1,
              "created_at": datetime.now(), "updated_at": datetime.now()}

    with patch("app.eval_jobs_cache") as cache, \
         patch("app.postgres_db") as pg, \
         patch("app.prepare_eval_assets", return_value=str(tmp_path / "assets")), \
         patch("app.extract_mcq_data", return_value=("p", ["(1) a"])), \
         patch("app.parse_eval_results", return_value={"overall_accuracy": 1.0, "per_action": {}}), \
         patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as exec_, \
         patch("utils.constant.RESULTS_ROOT", str(tmp_path)):
        cache.get.return_value = cached
        exec_.return_value = proc
        pg.update_evaluation_job = AsyncMock()
        await run_evaluation(
            eval_job_id=job, training_job_id="t", actions_json_path="/fake/actions.json",
            val_dataset_id="d", checkpoint_path="/fake/step_504", checkpoint_step=504,
            fps=8, temperature=0.0, backend="vllm", max_model_len=given,
        )

    argv = list(exec_.call_args[0])
    assert ("--max-model-len" in argv) is expected, argv
    if expected:
        assert argv[argv.index("--max-model-len") + 1] == str(given), argv


def test_documented_max_model_len_actually_fits_a_default_request():
    """32768 is documented as the workable ceiling -- hold the arithmetic behind it.

    Measured on the shipped checkpoint (Qwen3VLForConditionalGeneration): vision
    patch_size=16, spatial_merge_size=2, so a token covers (16*2)^2 = 1024 pixels.
    The prompt text for a 4-action dataset tokenises to 79. max_new_tokens is 4096.
    An earlier version of this help text called that "a few thousand tokens", which
    is wrong by ~5x and would lead a reader to pick a value that starts the engine
    and then rejects every request.
    """
    from validation.request_validation import ResolutionConfig

    px_per_token = (16 * 2) ** 2
    vision_tokens = ResolutionConfig().total_pixels / px_per_token
    assert vision_tokens == 16184, vision_tokens

    required = vision_tokens + 79 + 4096          # vision + prompt + generation
    assert required == pytest.approx(20359)

    assert 32768 > required, "the documented value must fit a default request"
    assert 8192 < required, (
        "a 'few thousand token' ceiling does NOT fit -- this is the mistake the "
        "help text used to invite")


def test_help_text_does_not_understate_the_context_requirement():
    """The old wording under-specified by 5x; keep it from coming back."""
    import inspect
    from sop import sop_eval, sop_e2e_eval

    for mod in (sop_eval, sop_e2e_eval):
        src = inspect.getsource(mod)
        start = src.index('"--max-model-len"')
        block = src[start:start + 1200]
        assert "16184" in block, f"{mod.__name__}: help must state the derived token count"
        assert "is ample" not in block, (
            f"{mod.__name__}: 'ample' asserts sufficiency without showing the arithmetic")
