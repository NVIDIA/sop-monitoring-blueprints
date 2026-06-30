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
"""Utilities for end-to-end evaluation (DDM temporal segmentation + VLM action recognition)."""

import json
import logging
import os
import re
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


# Sequence-level accuracy. Public e2e metrics:
#   sequence_accuracy = fraction of videos with edit-distance 0 against golden.
#   action_accuracy   = (total - wrong - duplicate - missing) / total,
#                       errors classified by Levenshtein backtrace.

_ACTION_PREFIX_RE = re.compile(r"^\((\d+)\).*")


def _remove_continuous_rep(seq: list) -> list:
    """Collapse adjacent duplicates: [1, 1, 2, 2, 3] → [1, 2, 3]."""
    out, prev = [], None
    for n in seq:
        if n != prev:
            out.append(n)
        prev = n
    return out


def _get_skippable_actions(actions_json_path: Optional[str]) -> set:
    """Return action numbers listed under 'actions_can_be_skipped' in the actions JSON."""
    if not actions_json_path or not os.path.exists(actions_json_path):
        return set()
    with open(actions_json_path, "r") as f:
        data = json.load(f)
    skippable = set()
    for entry in data.get("actions_can_be_skipped", []):
        m = _ACTION_PREFIX_RE.match(entry)
        if m:
            skippable.add(int(m.group(1)))
    return skippable


def _get_golden_actions(anno_json_path: str) -> dict:
    """
    Extract golden action sequences from an annotation JSON.

    Supported per-chunk shapes:
      single-op: {description: "(N) ...", ...}                     -> [N]
      single-op fallback: {action: N, ...}                          -> [N]
      two-op:    {actions: [N, M, ...], descriptions: ["(N) ...", "(M) ...", ...]}
                                                                    -> sorted [N, M]

    For two-op concurrent chunks the ids are sorted ascending and
    appended to the per-video sequence in that order, matching the
    convention used on the predicted side (see evaluate_action_sequences).
    """
    with open(anno_json_path, "r") as f:
        actions = json.load(f)

    out = {}
    for video_name in actions:
        out[video_name] = []
        for action in actions[video_name]:
            # two-op concurrent: plural arrays present
            if isinstance(action.get("actions"), list) and action["actions"]:
                ids = []
                for a in action["actions"]:
                    try:
                        ids.append(int(a))
                    except (TypeError, ValueError):
                        # skip malformed entries (None, "N/A", etc.) instead
                        # of aborting the whole eval — matches the resilience
                        # of the descriptions/single-op branches below
                        continue
                if ids:
                    out[video_name].extend(sorted(ids))
                    continue
            if isinstance(action.get("descriptions"), list) and action["descriptions"]:
                ids = []
                for d in action["descriptions"]:
                    m = _ACTION_PREFIX_RE.match(d or "")
                    if m:
                        ids.append(int(m.group(1)))
                if ids:
                    out[video_name].extend(sorted(ids))
                    continue
            # single-op
            desc = action.get("description", "")
            if desc == "Final segment":
                continue
            m = _ACTION_PREFIX_RE.match(desc)
            if m:
                out[video_name].append(int(m.group(1)))
            elif "action" in action:
                out[video_name].append(int(action["action"]))
    return out


def _read_sop_steps(content: str) -> list:
    """
    Parse "(N) ..." prefixed steps from a VLM response string.
    Returns list of "(N)<text>" strings in order.
    """
    if not content:
        return []
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    steps = []
    pos = 0
    while True:
        open_p = content.find("(", pos)
        if open_p == -1:
            break
        close_p = content.find(")", open_p)
        if close_p == -1:
            break
        step_num = content[open_p + 1: close_p]
        if not step_num.isdigit():
            pos = close_p + 1
            continue
        next_p = content.find("(", close_p + 1)
        body = content[close_p + 1: next_p] if next_p != -1 else content[close_p + 1:]
        body = body.replace("\n", " ").rstrip()
        steps.append(f"({step_num}){body}")
        pos = close_p + 1
    return steps


def _calculate_edit_distance(seq1: list, seq2: list) -> tuple:
    """
    Levenshtein distance with a backtrace that classifies each non-match step
    as Wrong (substitution), Extra/Duplicate (insertion in pred), or Missing
    (deletion in pred).

    Returns (distance, steps[], wrong, duplicate, missing).
    seq1 = predicted sequence (rows), seq2 = golden sequence (cols).
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,        # deletion (pred has extra)
                    dp[i][j - 1] + 1,        # insertion (pred is missing)
                    dp[i - 1][j - 1] + 1,    # substitution (wrong)
                )

    wrong = duplicate = missing = 0
    i, j = m, n
    steps = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq1[i - 1] == seq2[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            steps.append(
                f"Wrong: golden {seq2[j - 1]} predicted as {seq1[i - 1]} "
                f"(golden idx {j - 1}, pred idx {i - 1})"
            )
            wrong += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            steps.append(f"Duplicate (extra): predicted {seq1[i - 1]} at pred idx {i - 1}")
            duplicate += 1
            i -= 1
        else:
            steps.append(f"Missing: golden {seq2[j - 1]} at golden idx {j - 1}")
            missing += 1
            j -= 1
    return dp[m][n], steps[::-1], wrong, duplicate, missing


def evaluate_action_sequences(
    anno_json_path: str,
    pred_json_path: str,
    actions_json_path: Optional[str] = None,
) -> dict:
    """
    Sequence-level accuracy evaluation, matching the accuracy.json format
    produced by the upstream inference pipeline's reference comparison script.

    Args:
        anno_json_path:   path to anno.json (golden boundaries with "(N) <desc>" format).
        pred_json_path:   path to video_name_to_output_text.json from VLM stage.
        actions_json_path: optional; if present, action numbers listed under
                          "actions_can_be_skipped" are removed from both golden
                          and predicted sequences before comparison.

    Returns a dict with the same shape as the reference's accuracy.json.
    """
    skippable = _get_skippable_actions(actions_json_path)
    if skippable:
        logger.info("Skippable actions (excluded from comparison): %s", sorted(skippable))

    video_to_golden = _get_golden_actions(anno_json_path)
    with open(pred_json_path, "r") as f:
        pred_data = json.load(f)

    video_to_pred = {}
    for video_name, chunk_to_text in pred_data.items():
        video_name = os.path.basename(video_name)
        actions_in_order = []
        # Sort by chunk start-time, not lexicographically.
        sorted_keys = sorted(chunk_to_text.keys(), key=_chunk_key_start_sec)
        for k in sorted_keys:
            # Collect every (N) the VLM emitted for this chunk, then sort
            # ascending so concurrent two-op outputs align with golden's
            # sorted-ascending order (see _get_golden_actions). Single-op
            # chunks degenerate to a 1-element list, preserving original
            # behavior.
            chunk_ids = []
            for ss in _read_sop_steps(chunk_to_text[k]):
                m = _ACTION_PREFIX_RE.match(ss)
                if m:
                    chunk_ids.append(int(m.group(1)))
            actions_in_order.extend(sorted(chunk_ids))
        video_to_pred[video_name] = actions_in_order

    total_actions = total_wrong = total_duplicate = total_missing = 0
    total_videos = total_videos_dist_0 = 0
    videos_with_error: list = []
    per_video: list = []

    for video_name in sorted(video_to_pred):
        # Filter skippable ids first, then collapse adjacent duplicates.
        # The reverse order would let a skippable id act as a separator
        # that hides the dedup (e.g. [1, 2, 10, 2] with skippable={10}
        # would become [1, 2, 2] instead of [1, 2]).
        golden = _remove_continuous_rep(
            [a for a in video_to_golden.get(video_name, []) if a not in skippable]
        )
        pred = _remove_continuous_rep(
            [a for a in video_to_pred[video_name] if a not in skippable]
        )

        dist, steps, wrong, duplicate, missing = _calculate_edit_distance(pred, golden)
        total_videos += 1
        if dist == 0:
            total_videos_dist_0 += 1
        else:
            videos_with_error.append(video_name)

        total_actions += len(golden)
        total_wrong += wrong
        total_duplicate += duplicate
        total_missing += missing

        per_video.append({
            "video": video_name,
            "golden": golden,
            "predicted": pred,
            "edit_distance": dist,
            "wrong": wrong,
            "duplicate": duplicate,
            "missing": missing,
            "steps": steps,
        })

    # Floor at 0: with many extras, wrong+dup+missing can exceed total
    # and produce a negative numerator that would break percent formatters.
    action_accuracy = (
        max(0.0, (total_actions - total_wrong - total_duplicate - total_missing) / total_actions)
        if total_actions > 0 else 0.0
    )
    sequence_accuracy = total_videos_dist_0 / total_videos if total_videos > 0 else 0.0

    return {
        "total_videos": total_videos,
        "total_videos_dist_0": total_videos_dist_0,
        "sequence_accuracy": sequence_accuracy,
        "total_actions": total_actions,
        "wrong": total_wrong,
        "duplicate": total_duplicate,
        "missing": total_missing,
        "action_accuracy": action_accuracy,
        "videos_with_error": videos_with_error,
        "per_video": per_video,
    }


def _chunk_key_start_sec(key: str) -> float:
    """Extract the start-second from a chunk key like '[12.34s-56.78s]' for sorting."""
    m = re.match(r"\[(-?\d+(?:\.\d+)?)s", key.strip())
    return float(m.group(1)) if m else 0.0


def resolve_ddm_checkpoint(
    results_root: str, ddm_job_id: str, checkpoint_name: Optional[str] = None
) -> tuple[str, str]:
    """
    Find DDM checkpoint file and training config for a DDM training job.

    DDM training outputs: {results_root}/{job_id}/train/{job_id}/*.ckpt + config.yaml

    Returns:
        (checkpoint_path, config_path)

    Raises:
        FileNotFoundError: if job dir, checkpoint, or config not found
    """
    job_train_dir = os.path.join(results_root, ddm_job_id, "train", ddm_job_id)
    if not os.path.isdir(job_train_dir):
        raise FileNotFoundError(f"DDM training output not found: {job_train_dir}")

    if checkpoint_name:
        ckpt_path = os.path.join(job_train_dir, checkpoint_name)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"DDM checkpoint not found: {ckpt_path}")
    else:
        ckpt_path = os.path.join(job_train_dir, "last.ckpt")
        if not os.path.isfile(ckpt_path):
            ckpts = [f for f in os.listdir(job_train_dir) if f.endswith(".ckpt")]
            if not ckpts:
                raise FileNotFoundError(f"No .ckpt files found in {job_train_dir}")
            ckpt_path = os.path.join(job_train_dir, sorted(ckpts)[-1])

    config_path = os.path.join(job_train_dir, "config.yaml")
    if not os.path.isfile(config_path):
        # Fall back to the job-level config.
        config_path = os.path.join(results_root, ddm_job_id, f"{ddm_job_id}.yaml")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"DDM training config not found for job {ddm_job_id}"
            )

    return ckpt_path, config_path


def load_ddm_config(config_path: str) -> dict:
    """
    Extract DDM inference parameters from training config YAML.

    Returns:
        {"resolution": int, "frames_per_side": int}
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg.get("dataset_config", {})
    return {
        "resolution": dataset_cfg.get("resolution", 224),
        "frames_per_side": dataset_cfg.get("frames_per_side", 5),
    }


def extract_golden_boundaries(anno_json_path: str) -> dict[str, list[float]]:
    """
    Extract golden action boundaries from annotation JSON.

    Annotation format (from inference BP's collect_annotations.py):
    {
        "video.mp4": [
            {"description": "event0", "start_timestamp": 0.0, "end_timestamp": 5.0},
            {"description": "event1", "start_timestamp": 5.5, "end_timestamp": 10.0},
            ...
        ]
    }

    Boundaries are computed as midpoints between consecutive event end->start times,
    with 0.0 prepended and last end_timestamp appended.

    Returns:
        {video_name: [boundary_0, boundary_1, ...]}
    """
    with open(anno_json_path, "r") as f:
        video_to_anno = json.load(f)

    video_to_boundaries = {}
    for video, anno in video_to_anno.items():
        bdys = [0.0]
        for event1, event2 in zip(anno[:-1], anno[1:]):
            s_time = event1["end_timestamp"]
            e_time = event2["start_timestamp"]
            bdys.append((s_time + e_time) / 2)
        bdys.append(anno[-1]["end_timestamp"])
        video_to_boundaries[video] = bdys

    return video_to_boundaries


def compute_temporal_metrics(
    golden_bdy: Optional[list[float]],
    pred_bdy: list[float],
    duration_sec: float,
) -> dict:
    """
    Compute F1/precision/recall for boundary detection.

    Threshold = 2.5% of video duration (matches inference BP).
    Each golden boundary matches at most one predicted boundary.

    Returns:
        {"True Positive": int, "False Positive": int, "False Negative": int,
         "Precision": float, "Recall": float, "F1": float}
        (all None if golden_bdy is None)
    """
    if golden_bdy is None:
        return {
            "True Positive": None,
            "False Positive": None,
            "False Negative": None,
            "Precision": None,
            "Recall": None,
            "F1": None,
        }

    threshold = duration_sec * 0.025
    tp = 0
    used_pred = set()

    for golden in golden_bdy:
        for pred in pred_bdy:
            if abs(golden - pred) <= threshold and pred not in used_pred:
                tp += 1
                used_pred.add(pred)
                break

    fp = len(pred_bdy) - tp
    fn = len(golden_bdy) - tp
    precision = tp / len(pred_bdy) if pred_bdy else 0.0
    recall = tp / len(golden_bdy) if golden_bdy else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "True Positive": tp,
        "False Positive": fp,
        "False Negative": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def visualize_ddm_scores(
    video_name: str,
    golden_bdy: Optional[list[float]],
    pred_bdy: list[float],
    scores: list[float],
    fps: float,
    output_path: str,
):
    """
    Generate DDM boundary score visualization PNG.

    Ported from inference BP: scripts/temporal_segmentation.py:visualize_scores_with_boundaries
    """
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    if not scores:
        logger.warning("No scores to visualize for %s", video_name)
        return

    frame_times = [i / fps for i in range(len(scores))]
    max_score = max(scores) if scores else 1.0

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.fill_between(frame_times, scores, alpha=0.6, color="skyblue", label="Scores")
    ax.plot(frame_times, scores, color="blue", linewidth=1.5, alpha=0.8)

    if golden_bdy:
        for i, bdy in enumerate(golden_bdy):
            if 0 <= bdy <= max(frame_times):
                ax.axvline(
                    x=bdy,
                    color="red",
                    linewidth=2,
                    linestyle="--",
                    alpha=0.8,
                    label="Golden Boundaries" if i == 0 else "",
                )
                ax.text(
                    bdy,
                    max_score * 0.9,
                    str(i + 1),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor="red",
                        alpha=0.8,
                    ),
                )

    for i, bdy in enumerate(pred_bdy):
        if 0 <= bdy <= max(frame_times):
            ax.axvline(
                x=bdy,
                color="green",
                linewidth=2,
                linestyle="-",
                alpha=0.7,
                label="Predicted Boundaries" if i == 0 else "",
            )
            ax.text(
                bdy,
                max_score * 0.8,
                str(i + 1),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="lightgreen",
                    edgecolor="green",
                    alpha=0.9,
                ),
            )

    ax.set_xlim(0, max(frame_times) if frame_times else 1)
    ax.set_ylim(0, max_score * 1.1 if scores else 1)
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title(
        f"{video_name}\nScores over Time with Golden and Predicted Boundaries",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Indirection hook so unit tests can patch PyAV without real video files.
def _av_open(path: str):
    import av
    return av.open(path)


def get_video_duration_sec(video_path: str) -> float:
    """
    Read a video's duration in seconds from PyAV stream metadata only — no
    full decode. Used by the uniform-chunking branch of stage 1 where we
    don't need pixel data, just the runtime length to slice into fixed
    chunks.

    Falls back to ``container.duration`` (in AV_TIME_BASE = 1e6 microseconds)
    if the video stream's own ``duration`` and ``time_base`` aren't both
    populated. Raises if neither yields a positive duration.
    """
    container = _av_open(video_path)
    try:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.time_base * stream.duration)
            if duration > 0:
                return duration
        if container.duration:
            return container.duration / 1_000_000.0
    finally:
        container.close()
    raise RuntimeError(f"Could not determine duration for {video_path}")


def uniform_chunk_boundaries(duration_sec: float, chunk_length_sec: float) -> list[float]:
    """
    Compute fixed-length chunk boundaries for a video of length `duration_sec`.

    The last chunk is truncated to `duration_sec` if the duration isn't an exact
    multiple of `chunk_length_sec`. Returns a single boundary list with the same
    shape as DDM's stage-1 output ([0.0, b1, b2, ..., duration_sec]) so the VLM
    stage can be reused unchanged.

    Mirrors the "uniform" branch of the inference pipeline's
    chunking_options — no learned segmentation, just even time slices.
    """
    if chunk_length_sec <= 0:
        raise ValueError(f"chunk_length_sec must be > 0, got {chunk_length_sec}")
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0, got {duration_sec}")

    boundaries = [0.0]
    t = chunk_length_sec
    while t < duration_sec:
        boundaries.append(t)
        t += chunk_length_sec
    boundaries.append(duration_sec)
    return boundaries


def map_chunks_to_ground_truth(
    pred_boundaries: list[float],
    golden_boundaries: list[float],
    action_count: int,
) -> list[int]:
    """
    For each predicted chunk, determine the ground-truth action index (1-based)
    by computing which golden action interval has the maximum time overlap.

    Args:
        pred_boundaries: [start, b1, b2, ..., end] from DDM
        golden_boundaries: [start, b1, b2, ..., end] from annotation
        action_count: number of actions

    Returns:
        List of 1-based action indices, one per predicted chunk.
    """
    # action i (1-based) spans [golden[i-1], golden[i]]
    golden_intervals = list(zip(golden_boundaries[:-1], golden_boundaries[1:]))
    pred_chunks = list(zip(pred_boundaries[:-1], pred_boundaries[1:]))

    result = []
    for p_start, p_end in pred_chunks:
        best_action = 1
        best_overlap = 0.0
        for idx, (g_start, g_end) in enumerate(golden_intervals):
            overlap_start = max(p_start, g_start)
            overlap_end = min(p_end, g_end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_action = idx + 1  # 1-based

        result.append(min(best_action, action_count))

    return result
