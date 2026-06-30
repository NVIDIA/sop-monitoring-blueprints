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

"""Analyze DDM temporal segmentation quality.

Usage:
    python analyze_ddm_boundaries.py <f1_json_path> \
        [--golden-boundaries <golden_json>] \
        [--fps N] [--max-frames N] [--long-ratio N] [--short-ratio N]

Analyzes:
- Per-video F1, precision, recall, TP, FP, FN
- Videos sorted by F1 (worst first) for severity assessment
- Boundary count mismatches (predicted vs golden)
- Chunk duration analysis with data-driven thresholds

Thresholds (same logic as analyze_vlm_output.py):
- Long: (max_frames / fps) * long_ratio. Default long_ratio=3.0.
- Short: min_golden_action_duration * short_ratio. Default short_ratio=0.8.
  Requires --golden-boundaries.
"""

import argparse
import json
import sys
from pathlib import Path

_VIDEO_EXTS = (".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV")


def _stem(name):
    """Strip a single trailing video extension. Idempotent for stem-form keys."""
    for ext in _VIDEO_EXTS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def _align_keys_to(source, target_keys, label):
    """Re-key `source` to match `target_keys`, joining on basename (stem).

    The DDM pipeline emits the three JSONs (f1_*.json, video_to_ddm_info_debug.json,
    video_to_boundaries_debug.json) with inconsistent key conventions — some keep
    the `.mp4` extension, the golden boundaries file does not. Without alignment
    the helper's per-video score analysis silently skips every video and leaves
    `score_threshold_summary` / `nms_sensitivity` empty, which agents read as
    "no signal" rather than "couldn't compute."

    This wrapper rebuilds `source` so its keys match `target_keys`, joining
    extension-with vs extension-without variants by their common stem. Emits a
    stderr warning when the join is partial or empty so the failure is visible
    instead of silent.
    """
    if not source:
        return source

    source_by_stem = {}
    for k, v in source.items():
        source_by_stem.setdefault(_stem(k), v)

    aligned = {}
    misses = []
    for tk in target_keys:
        if tk in source:
            aligned[tk] = source[tk]
        elif _stem(tk) in source_by_stem:
            aligned[tk] = source_by_stem[_stem(tk)]
        else:
            misses.append(tk)

    n_total = len(target_keys)
    n_hit = n_total - len(misses)
    if n_total > 0 and n_hit == 0:
        sample_t = next(iter(target_keys)) if target_keys else "(none)"
        sample_s = next(iter(source)) if source else "(none)"
        sys.stderr.write(
            f"WARN: {label} has 0/{n_total} videos matching the f1.json keys "
            f"even after stem normalization. Per-video score analysis "
            f"(score_threshold_summary, nms_sensitivity) will be skipped.\n"
            f"  f1.json sample key:  {sample_t!r}\n"
            f"  {label} sample key:  {sample_s!r}\n"
        )
    elif misses:
        sys.stderr.write(
            f"WARN: {label} is missing {len(misses)}/{n_total} videos after "
            f"stem normalization. First miss: {misses[0]!r}\n"
        )

    return aligned


def compute_golden_min_action_duration(golden_data):
    """Compute the minimum action duration from golden boundaries dict.

    Excludes the first and last segments of each video (typically idle periods).
    """
    action_durations = []
    for video_name, boundaries in golden_data.items():
        if not isinstance(boundaries, list) or len(boundaries) < 3:
            continue
        for i in range(1, len(boundaries) - 2):
            dur = boundaries[i + 1] - boundaries[i]
            if dur > 0:
                action_durations.append(dur)

    if not action_durations:
        return None
    return min(action_durations)


def lookup_score_at_time(scores, fps, t):
    """DDM frame score at time t (seconds). 0-indexed: scores[round(t*fps) - 1]."""
    if not scores or fps <= 0:
        return None
    idx = round(t * fps) - 1
    if idx < 0 or idx >= len(scores):
        return None
    return float(scores[idx])


def peak_score_in_window(scores, fps, t, window_sec):
    """Peak score in [t - window_sec, t + window_sec]."""
    if not scores or fps <= 0:
        return None
    n = len(scores)
    lo = max(0, round((t - window_sec) * fps) - 1)
    hi = min(n, round((t + window_sec) * fps))
    if lo >= hi:
        return None
    return float(max(scores[lo:hi]))


def classify_detected_boundaries(golden_bdy, pred_bdy, threshold_sec):
    """Match predicted boundaries to golden boundaries.

    Per-golden in order, pick the NEAREST unmatched pred within threshold_sec.
    The algorithm agrees when preds and goldens are both time-sorted and each golden has at
    most one pred within threshold, but diverge when a golden has multiple
    preds within threshold — only argmin matches the canonical F1.

    The first and last entries of both lists are post-processing endpoints
    (0.0 and duration_sec, added by the inference pipeline) and are excluded
    from this analysis — they aren't DDM-detected and their "scores" are
    meaningless for threshold tuning.

    Returns: (matched_pred_times, fp_pred_times, matched_golden_idx_set,
              detected_golden_list).
    """
    if (golden_bdy is None or len(golden_bdy) < 2 or len(pred_bdy) < 2
            or threshold_sec is None):
        return [], [], set(), []
    detected_pred = pred_bdy[1:-1]
    detected_golden = golden_bdy[1:-1]
    used_pred_idx = set()
    matched_golden_idx = set()
    matched_pred = []
    for gi, golden in enumerate(detected_golden):
        best_pi = None
        best_dist = None
        for pi, pred in enumerate(detected_pred):
            if pi in used_pred_idx:
                continue
            d = abs(golden - pred)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_pi = pi
        if best_pi is not None and best_dist <= threshold_sec:
            matched_pred.append(detected_pred[best_pi])
            used_pred_idx.add(best_pi)
            matched_golden_idx.add(gi)
    fp_pred = [p for pi, p in enumerate(detected_pred) if pi not in used_pred_idx]
    return matched_pred, fp_pred, matched_golden_idx, detected_golden


def analyze_ddm(f1_path, golden_path=None, ddm_info_path=None,
                fps=8, max_frames=40,
                long_ratio=3.0, short_ratio=0.8):
    with open(f1_path) as f:
        ddm_data = json.load(f)

    golden_data = None
    if golden_path:
        with open(golden_path) as f:
            golden_data = json.load(f)

    ddm_info_data = None
    if ddm_info_path:
        with open(ddm_info_path) as f:
            ddm_info_data = json.load(f)

    # Align golden/ddm_info keys to ddm_data keys (stem-based join).
    # The three JSONs are emitted with inconsistent extension conventions;
    # without this, per-video score analysis is silently skipped.
    target_keys = {
        k for k, v in ddm_data.items()
        if not k.startswith("avg_") and isinstance(v, dict)
    }
    if golden_data is not None:
        golden_data = _align_keys_to(golden_data, target_keys, "golden_boundaries")
    if ddm_info_data is not None:
        ddm_info_data = _align_keys_to(ddm_info_data, target_keys, "ddm_info")

    # Compute thresholds
    long_threshold = (max_frames / fps) * long_ratio

    short_threshold = None
    if golden_data:
        min_golden_dur = compute_golden_min_action_duration(golden_data)
        if min_golden_dur is not None:
            short_threshold = min_golden_dur * short_ratio

    videos = []
    avg_keys = ["avg_f1", "avg_precision", "avg_recall", "avg_tp", "avg_fp", "avg_fn"]
    summary_metrics = {k: ddm_data.get(k) for k in avg_keys if k in ddm_data}

    for video_name, info in ddm_data.items():
        if video_name.startswith("avg_") or not isinstance(info, dict):
            continue

        boundaries = info.get("boundaries", [])
        metric = info.get("metric", {})
        # ddm_threshold = duration * 0.025, used for F1 metric matching only
        # (NOT the boundary filtering threshold — that is score_threshold)
        threshold = info.get("ddm_threshold")

        num_chunks = max(len(boundaries) - 1, 0)

        # Calculate chunk durations from DDM boundaries
        chunk_durations = []
        for i in range(len(boundaries) - 1):
            dur = boundaries[i + 1] - boundaries[i]
            chunk_durations.append(round(dur, 2))

        # Get golden boundary count if available
        golden_boundary_count = None
        golden_chunk_durations = []
        if golden_data and video_name in golden_data:
            golden_bounds = golden_data[video_name]
            golden_boundary_count = len(golden_bounds)
            for i in range(len(golden_bounds) - 1):
                golden_chunk_durations.append(
                    round(golden_bounds[i + 1] - golden_bounds[i], 2)
                )

        video_info = {
            "video": video_name,
            "num_predicted_boundaries": len(boundaries),
            "num_chunks": num_chunks,
            "ddm_metric_cal_threshold": threshold,
            "f1": metric.get("F1"),
            "precision": metric.get("Precision"),
            "recall": metric.get("Recall"),
            "tp": metric.get("True Positive"),
            "fp": metric.get("False Positive"),
            "fn": metric.get("False Negative"),
            "chunk_durations": chunk_durations,
            "min_chunk_dur": min(chunk_durations) if chunk_durations else None,
            "max_chunk_dur": max(chunk_durations) if chunk_durations else None,
            "golden_boundary_count": golden_boundary_count,
            "golden_chunk_durations": golden_chunk_durations,
        }

        # Score-level boundary analysis (when ddm_info is available)
        if (ddm_info_data and video_name in ddm_info_data
                and golden_data and video_name in golden_data
                and threshold is not None):
            info_entry = ddm_info_data[video_name]
            scores = info_entry.get("scores")
            video_fps = info_entry.get("fps")
            if scores and video_fps:
                matched_pred, fp_pred, matched_gi, detected_golden = (
                    classify_detected_boundaries(
                        golden_data[video_name], boundaries, threshold
                    )
                )
                video_info["boundaries_with_scores"] = (
                    [
                        {"time": round(t, 3),
                         "score": lookup_score_at_time(scores, video_fps, t),
                         "is_tp": True}
                        for t in matched_pred
                    ]
                    + [
                        {"time": round(t, 3),
                         "score": lookup_score_at_time(scores, video_fps, t),
                         "is_tp": False}
                        for t in fp_pred
                    ]
                )
                video_info["golden_peak_scores"] = [
                    {
                        "time": round(g, 3),
                        "peak_score": peak_score_in_window(
                            scores, video_fps, g, threshold
                        ),
                        "matched": gi in matched_gi,
                    }
                    for gi, g in enumerate(detected_golden)
                ]

                # Annotate each pred boundary with distance to nearest
                # higher-score pred (used for NMS-sensitivity analysis).
                for b in video_info["boundaries_with_scores"]:
                    if b["score"] is None:
                        b["dist_to_higher_score_pred"] = None
                        continue
                    higher_dists = [
                        abs(other["time"] - b["time"])
                        for other in video_info["boundaries_with_scores"]
                        if (other is not b
                            and other["score"] is not None
                            and other["score"] > b["score"])
                    ]
                    b["dist_to_higher_score_pred"] = (
                        round(min(higher_dists), 3) if higher_dists else None
                    )

                # Annotate each missed golden with distance to the nearest
                # detected pred boundary (proxy for the NMS window that
                # suppressed it). Also flag whether that pred is a
                # "greedy-matcher artifact": the nearest pred sits WITHIN
                # the F1 threshold of this golden but was paired to a
                # neighbor by the greedy matcher. Lowering nms_sec does NOT
                # help these cases — the pred is already in the output.
                # The threshold gate rules out genuine detection misses
                # whose nearest pred happens to be far away and matched
                # to its own (valid) golden.
                pred_times = [
                    b["time"] for b in video_info["boundaries_with_scores"]
                ]
                matched_pred_time_set = {
                    b["time"] for b in video_info["boundaries_with_scores"]
                    if b["is_tp"]
                }
                for g in video_info["golden_peak_scores"]:
                    if g["matched"] or not pred_times:
                        g["dist_to_nearest_pred"] = None
                        g["is_greedy_matcher_artifact"] = None
                        continue
                    nearest_pred_time = min(
                        pred_times, key=lambda p: abs(p - g["time"])
                    )
                    dist = abs(nearest_pred_time - g["time"])
                    g["dist_to_nearest_pred"] = round(dist, 3)
                    g["is_greedy_matcher_artifact"] = (
                        dist <= threshold
                        and nearest_pred_time in matched_pred_time_set
                    )

        videos.append(video_info)

    # Classify by chunk duration thresholds
    tiny_chunks = []
    if short_threshold is not None:
        tiny_chunks = [
            v for v in videos
            if v["min_chunk_dur"] is not None and v["min_chunk_dur"] < short_threshold
        ]

    huge_chunks = [
        v for v in videos
        if v["max_chunk_dur"] is not None and v["max_chunk_dur"] > long_threshold
    ]

    # Sort all videos by F1 ascending (worst first) for agent to assess severity
    videos_by_f1 = sorted(
        [v for v in videos if v["f1"] is not None],
        key=lambda x: x["f1"]
    )

    # Videos with any boundary errors (FN > 0 or FP > 0), sorted by F1
    videos_with_errors = [
        v for v in videos_by_f1
        if (v["fn"] and v["fn"] > 0) or (v["fp"] and v["fp"] > 0)
    ]

    # Aggregate score-level summary across all videos (only meaningful when
    # ddm_info was provided so per-boundary scores were computed).
    all_tp_scores = []
    all_fp_scores = []
    all_missed_peaks = []
    for v in videos:
        for b in v.get("boundaries_with_scores", []):
            if b["score"] is None:
                continue
            (all_tp_scores if b["is_tp"] else all_fp_scores).append(b["score"])
        for g in v.get("golden_peak_scores", []):
            if g["peak_score"] is None or g["matched"]:
                continue
            all_missed_peaks.append(g["peak_score"])

    # NMS sensitivity: bidirectional analysis of nms_sec changes
    nms_sensitivity = None
    fps_suppressible = []
    tps_at_risk = []
    fns_admittable = []
    fns_greedy_artifacts = []
    for v in videos:
        for b in v.get("boundaries_with_scores", []):
            d = b.get("dist_to_higher_score_pred")
            if d is None:
                continue
            entry = {
                "video": v["video"],
                "time": b["time"],
                "score": b["score"],
                "dist_to_higher_score_pred": d,
            }
            (tps_at_risk if b["is_tp"] else fps_suppressible).append(entry)
        for g in v.get("golden_peak_scores", []):
            if g.get("matched") or g.get("dist_to_nearest_pred") is None:
                continue
            entry = {
                "video": v["video"],
                "golden_time": g["time"],
                "peak_score": g["peak_score"],
                "dist_to_nearest_pred": g["dist_to_nearest_pred"],
            }
            # Greedy-matcher artifact: nearest pred is within threshold of
            # this golden but paired to a neighbor. Lowering nms_sec won't
            # help — report separately so the agent doesn't act on it.
            if g.get("is_greedy_matcher_artifact"):
                fns_greedy_artifacts.append(entry)
            else:
                fns_admittable.append(entry)

    if fps_suppressible or tps_at_risk or fns_admittable or fns_greedy_artifacts:
        fps_suppressible.sort(key=lambda x: x["dist_to_higher_score_pred"])
        tps_at_risk.sort(key=lambda x: x["dist_to_higher_score_pred"])
        fns_admittable.sort(key=lambda x: x["dist_to_nearest_pred"])
        fns_greedy_artifacts.sort(key=lambda x: x["dist_to_nearest_pred"])
        fp_d = [e["dist_to_higher_score_pred"] for e in fps_suppressible]
        tp_d = [e["dist_to_higher_score_pred"] for e in tps_at_risk]
        fn_d = [e["dist_to_nearest_pred"] for e in fns_admittable]
        nms_sensitivity = {
            "note": (
                "nms_sec is GLOBAL and bidirectional. RAISING suppresses FPs "
                "near a higher-score boundary but may also suppress real TPs "
                "near higher-score TPs. LOWERING admits NMS-suppressed FNs but "
                "may also admit FP candidates currently hidden from the "
                "post-NMS output (cannot quantify without re-running detection)."
            ),
            "fps_suppressible_by_raising_nms_sec": fps_suppressible,
            "tps_at_risk_if_nms_sec_raised": tps_at_risk,
            "fns_admittable_by_lowering_nms_sec": fns_admittable,
            "fns_greedy_matcher_artifacts": fns_greedy_artifacts,
            "min_fp_dist_to_higher_score_pred": (
                min(fp_d) if fp_d else None
            ),
            "min_tp_dist_to_higher_score_pred": (
                min(tp_d) if tp_d else None
            ),
            "min_fn_dist_to_nearest_pred": min(fn_d) if fn_d else None,
            "caveats": [
                "Lowering nms_sec may admit new FP candidates that were "
                "previously suppressed but are not in the current detected "
                "boundary list. Cannot quantify without re-running detection "
                "— re-evaluate after the change.",
                "Raising nms_sec may suppress real TPs (see "
                "tps_at_risk_if_nms_sec_raised). Pick nms_sec strictly LESS "
                "than min_tp_dist_to_higher_score_pred to keep all current TPs.",
                "Filter fns_admittable_by_lowering_nms_sec by peak_score >= "
                "current score_threshold (from Step 2) to identify FNs that "
                "are genuinely NMS-suppressed (not just below threshold).",
                "fns_greedy_matcher_artifacts are FNs whose nearest pred is "
                "already in the detected output but was paired to a neighbor "
                "golden by the greedy F1 matcher. Lowering nms_sec does NOT "
                "help these — the pred isn't suppressed. Fixing them requires "
                "either an additional DDM detection near the missed golden or "
                "an optimal (Hungarian) F1 matcher upstream.",
            ],
        }

    score_threshold_summary = None
    if all_tp_scores or all_fp_scores or all_missed_peaks:
        min_tp = min(all_tp_scores) if all_tp_scores else None
        max_fp = max(all_fp_scores) if all_fp_scores else None
        clean_fix_available = (
            min_tp is not None and max_fp is not None and max_fp < min_tp
        )
        score_threshold_summary = {
            "note": (
                "score_threshold and nms_sec are GLOBAL evaluation parameters "
                "applied to all videos uniformly. The aggregates below are "
                "dataset-wide; tuning shifts FP/FN tradeoffs for the entire "
                "evaluation, not just one video."
            ),
            "min_tp_score": min_tp,
            "max_fp_score": max_fp,
            "max_missed_golden_peak_score": (
                max(all_missed_peaks) if all_missed_peaks else None
            ),
            "num_tp_boundaries": len(all_tp_scores),
            "num_fp_boundaries": len(all_fp_scores),
            "num_missed_golden": len(all_missed_peaks),
            "clean_fix_available_by_raising_threshold": clean_fix_available,
            "fp_scores_sorted_asc": sorted(all_fp_scores),
            "missed_golden_peaks_sorted_desc": sorted(
                all_missed_peaks, reverse=True
            ),
        }

    return {
        "summary_metrics": summary_metrics,
        "total_videos": len(videos),
        "short_threshold": round(short_threshold, 2) if short_threshold else None,
        "long_threshold": round(long_threshold, 2),
        "fps": fps,
        "max_frames": max_frames,
        "score_threshold_summary": score_threshold_summary,
        "nms_sensitivity": nms_sensitivity,
        "videos_with_boundary_errors": [
            {
                "video": v["video"],
                "f1": v["f1"],
                "precision": v["precision"],
                "recall": v["recall"],
                "tp": v["tp"],
                "fp": v["fp"],
                "fn": v["fn"],
                "num_predicted": v["num_predicted_boundaries"],
                "num_golden": v["golden_boundary_count"],
                "min_chunk_dur": v["min_chunk_dur"],
                "max_chunk_dur": v["max_chunk_dur"],
            }
            for v in videos_with_errors
        ],
        "videos_with_tiny_chunks": [
            {
                "video": v["video"],
                "min_chunk": v["min_chunk_dur"],
                "fp": v["fp"],
                "f1": v["f1"],
            }
            for v in sorted(tiny_chunks, key=lambda x: x["min_chunk_dur"] or 0)
        ],
        "videos_with_huge_chunks": [
            {
                "video": v["video"],
                "max_chunk": v["max_chunk_dur"],
                "fn": v["fn"],
                "f1": v["f1"],
            }
            for v in sorted(huge_chunks, key=lambda x: -(x["max_chunk_dur"] or 0))
        ],
        "all_videos": videos,
    }


def print_report(analysis):
    print("=" * 70)
    print("DDM TEMPORAL SEGMENTATION ANALYSIS")
    print("=" * 70)
    m = analysis["summary_metrics"]
    print(f"Total videos: {analysis['total_videos']}")
    print(f"Avg F1: {m.get('avg_f1', 'N/A'):.4f}, "
          f"Precision: {m.get('avg_precision', 'N/A'):.4f}, "
          f"Recall: {m.get('avg_recall', 'N/A'):.4f}")

    st = analysis["short_threshold"]
    lt = analysis["long_threshold"]
    if st is not None:
        print(f"Short threshold: {st}s (0.8 * min golden action duration)")
    else:
        print("Short threshold: N/A (no --golden-boundaries provided)")
    print(f"Long threshold: {lt}s (max_frames/fps * long_ratio)")
    print()

    errors = analysis["videos_with_boundary_errors"]
    if errors:
        print(f"Videos with Boundary Errors ({len(errors)}, sorted by F1 ascending):")
        print("-" * 70)
        for v in errors:
            golden_str = f", golden={v['num_golden']}" if v["num_golden"] else ""
            print(f"  {v['video']}: F1={v['f1']:.3f}, "
                  f"TP={v['tp']}, FP={v['fp']}, FN={v['fn']}, "
                  f"pred={v['num_predicted']}{golden_str}, "
                  f"chunks=[{v['min_chunk_dur']}s - {v['max_chunk_dur']}s]")
        print()

    if analysis["videos_with_huge_chunks"]:
        print(f"Videos with HUGE chunks (>{lt}s): "
              f"{len(analysis['videos_with_huge_chunks'])}")
        for v in analysis["videos_with_huge_chunks"]:
            print(f"  {v['video']}: max_chunk={v['max_chunk']}s, "
                  f"FN={v['fn']}, F1={v['f1']}")
        print()

    if analysis["videos_with_tiny_chunks"]:
        print(f"Videos with TINY chunks (<{st}s): "
              f"{len(analysis['videos_with_tiny_chunks'])}")
        for v in analysis["videos_with_tiny_chunks"]:
            print(f"  {v['video']}: min_chunk={v['min_chunk']}s, "
                  f"FP={v['fp']}, F1={v['f1']}")
        print()

    n = analysis.get("nms_sensitivity")
    if n:
        print("NMS Sensitivity (bidirectional — raising and lowering both have risks):")
        print("-" * 70)
        print(f"  FPs suppressible by raising nms_sec: "
              f"{len(n['fps_suppressible_by_raising_nms_sec'])} "
              f"(min dist to higher-score pred = "
              f"{n['min_fp_dist_to_higher_score_pred']})")
        print(f"  TPs at risk if nms_sec raised: "
              f"{len(n['tps_at_risk_if_nms_sec_raised'])} "
              f"(min dist to higher-score pred = "
              f"{n['min_tp_dist_to_higher_score_pred']})")
        print(f"  FNs admittable by lowering nms_sec: "
              f"{len(n['fns_admittable_by_lowering_nms_sec'])} "
              f"(min dist to nearest pred = "
              f"{n['min_fn_dist_to_nearest_pred']})")
        print(f"  FNs from greedy-matcher artifacts (pred in output, paired "
              f"to neighbor — NOT nms_sec-tunable): "
              f"{len(n.get('fns_greedy_matcher_artifacts', []))}")
        print(f"  Note: filter fns_admittable_* by peak_score >= current "
              f"score_threshold to identify NMS-suppressed (vs threshold-filtered) FNs.")
        print()

    s = analysis.get("score_threshold_summary")
    if s:
        print("Score-Level Threshold Analysis "
              "(GLOBAL — affects all evaluation videos):")
        print("-" * 70)
        print(f"  TP boundaries: {s['num_tp_boundaries']}, "
              f"FP boundaries: {s['num_fp_boundaries']}, "
              f"missed golden: {s['num_missed_golden']}")
        if s["min_tp_score"] is not None:
            print(f"  min TP score:  {s['min_tp_score']:.4f}  "
                  f"(raising threshold above this drops a TP somewhere)")
        if s["max_fp_score"] is not None:
            print(f"  max FP score:  {s['max_fp_score']:.4f}  "
                  f"(raising threshold above this eliminates ALL current FPs)")
        if s["max_missed_golden_peak_score"] is not None:
            print(f"  max missed-golden peak: "
                  f"{s['max_missed_golden_peak_score']:.4f}  "
                  f"(lowering threshold to this catches the highest-confidence FN)")
        if s["clean_fix_available_by_raising_threshold"]:
            print(f"  CLEAN FIX (Pattern 3): raise score_threshold to a value in "
                  f"({s['max_fp_score']:.4f}, {s['min_tp_score']:.4f}) — "
                  f"eliminates all FPs without dropping any TP.")
        elif s["min_tp_score"] is not None and s["max_fp_score"] is not None:
            print(f"  No clean Pattern-3 fix from threshold tuning alone: "
                  f"max_fp_score ({s['max_fp_score']:.4f}) "
                  f">= min_tp_score ({s['min_tp_score']:.4f}). "
                  f"Threshold tuning trades FPs for FNs.")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze DDM temporal segmentation quality."
    )
    parser.add_argument("f1_path", help="Path to f1_<value>.json")
    parser.add_argument(
        "--golden-boundaries", default=None,
        help="Path to golden boundaries JSON; required for short-threshold "
             "computation",
    )
    parser.add_argument(
        "--ddm-info", default=None,
        help="Path to video_to_ddm_info_debug.json (per-frame DDM scores). "
             "When provided alongside --golden-boundaries, enables the "
             "score-level threshold-tuning analysis "
             "(score_threshold_summary in the output).",
    )
    parser.add_argument(
        "--fps", type=int, default=8,
        help="Frame sampling rate (default: 8, matches the E2E pipeline's "
             "hard-coded value)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=40,
        help="Max frames per chunk (default: 40)",
    )
    parser.add_argument(
        "--long-ratio", type=float, default=3.0,
        help="Long chunk threshold ratio applied to max_frames/fps "
             "(default: 3.0)",
    )
    parser.add_argument(
        "--short-ratio", type=float, default=0.8,
        help="Short chunk threshold ratio applied to min golden action "
             "duration (default: 0.8)",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save ddm_analysis.json "
             "(defaults to the input file's parent directory)",
    )
    args = parser.parse_args()

    analysis = analyze_ddm(
        args.f1_path, args.golden_boundaries, args.ddm_info,
        args.fps, args.max_frames,
        args.long_ratio, args.short_ratio,
    )
    print_report(analysis)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(args.f1_path).parent
    out_path = out_dir / "ddm_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"Analysis saved to: {out_path}")


if __name__ == "__main__":
    main()
