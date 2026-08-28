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


"""Window-Matched MCQ (WMCQ) augmentation.

Every other augmentation stage consumes the *pre-cut* action chunks produced by the
video split (``<action>_<video>_<dup>_<timeline>.mp4``), so each training clip is a
clean, exactly-trimmed action segment. That is the right shape when inference also
sees exactly-trimmed segments -- i.e. when DDM temporal chunking is accurate.

It is the wrong shape when it is not. If DDM chunking performs badly and the
pipeline falls back to uniform chunking, inference sees fixed-length sliding
windows: a window contains a brief key-step surrounded by whatever came before and
after it, and the model has never been trained on anything that looks like that.
The model is being asked a question in a format it has not seen.

WMCQ closes that gap. It goes back to the *source* video and cuts training clips
with the same geometry the sliding-window evaluation uses: real windows of exactly
``--window`` seconds, positioned so a key-step falls at varying offsets inside them,
with the padding being the genuine surrounding footage. Negatives are windows of the
same length taken from regions that contain no key-step at all.

Because it cuts its own windows from the source video, WMCQ is a different *stage*
from the chunk-consuming augmentations: it does not read the pre-cut chunks and is
therefore unaffected by ``merge_small_chunks``, which runs after the video split and
before augmentation.

Key-steps longer than the window
--------------------------------
A key-step can be longer than one window, and there are two ways to handle it:

``--tile-long`` on (default)
    The key-step is *tiled* by several windows of exactly ``--window``, each lying
    inside it. Every clip is the same length, so duration carries no class
    information at all, and every clip matches eval geometry.

``--tile-long`` off
    The window is *enlarged* to ``(end - start + --enlarge-pad)`` so it still
    contains the whole key-step. Simpler, but it makes clip duration correlate with
    the class: if only one action is ever long, every long clip is that action and
    the model can score well on the training set without looking at the video --
    then finds nothing resembling it at inference, where every window is the same
    length. Duration becomes a shortcut feature that does not exist at test time.

    ``--tile-passes`` reads ``--variants`` as the number of full *passes* over the
                      key-step: ``k = variants * ceil(L / window)``. This makes
                      "one exposure" mean "100%% coverage" for every class. Without
                      it, N crops shows a short action N times over but barely
                      covers a long one once, so the same ``--variants`` means very
                      different things for different classes.

The defaults (``--tile-long`` and ``--tile-passes`` on, ``--variants 4``,
``--neg-ratio 1.5``) are the configuration our own runs converged on. Turn tiling off
only if you specifically want the enlarge behaviour.

Output: ``<output-root>/videos/*.mp4`` + ``<output-root>/<output-name>.json``.
"""

import argparse
import copy
import glob
import math
import os
import random
import re
import zlib
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from moviepy.editor import VideoFileClip

from .cfg.wmcq import QUESTION_TEMPLATE
from .utils import const
from .utils.annotation_template import llava_video, wmcq_meta
from .utils.helper import (
    clean_sentence,
    create_dir,
    dump_json,
    get_video_meta,
    read_json,
    read_txt,
    str2bool,
    unpack_annotation,
    write_txt,
    write_video,
)
from .utils.logger import logging

# Action index as written in actions.json descriptions, e.g. "(2) the worker ...".
ACTION_INDEX_RE = re.compile(r"\((\d+)\)")

# Resolution of the grid that negative-window candidates are laid out on. Not a tunable:
# the grid only decides how finely candidates are spaced, and the pool it produces is
# oversupplied by one to two orders of magnitude against the number of negatives actually
# drawn from it, so its value has no measurable effect on the sample. One second matches
# the granularity SOP annotations are written at.
NEGATIVE_GRID_SEC = 1.0


def prepare_sample_choices(action_json: str) -> str:
    """Write question.txt / choices.txt for WMCQ, mirroring the other stages."""
    all_actions = read_json(action_json)[const.ACTION_JSON_KEY]
    config_root = os.path.join(str(Path(action_json).parent), const.WMCQ)
    create_dir(config_root)

    choices = ""
    for i, cur_action in enumerate(all_actions, 1):
        cleaned_action = f"({i}) " + clean_sentence(cur_action)
        choices += cleaned_action
        if i != len(all_actions):
            choices += const.LINE_BREAK
    choices = choices.strip()

    question = (
        QUESTION_TEMPLATE.replace(const.STEP_TOKEN, f"{len(all_actions)}").replace(
            const.SUBJECT_TOKEN, const.DEFAULT_SUBJECT
        )
        + choices
    )

    logging.info(f"Create WMCQ config question: {const.QUESTION}.txt\n{question}")
    write_txt(os.path.join(config_root, const.QUESTION + ".txt"), question)

    logging.info(f"Create WMCQ config choices: {const.CHOICES}.txt\n{choices}")
    write_txt(os.path.join(config_root, const.CHOICES + ".txt"), choices)

    return config_root


def parse_action_index(event: dict) -> Optional[int]:
    """1-based action index of an annotation record.

    Prefers the explicit ``action`` field and falls back to the ``(N)`` prefix of
    ``description``, so annotations written by either convention are accepted.
    """
    action = event.get(const.ACTION)
    if action is not None:
        return int(action)
    match = ACTION_INDEX_RE.search(event.get("description", ""))
    return int(match.group(1)) if match else None


def _spread(low: float, high: float, k: int, start: float, end: float) -> List[float]:
    """``k`` window starts evenly spread over ``[low, high]``. Duplicates are KEPT.

    Starts are rounded to the millisecond, so a degenerate interval collapses to a
    single distinct position -- every requested pass lands on the same clip. This
    returns it ``k`` times anyway, and that is deliberate.

    An earlier version deduplicated, on the reasoning that byte-identical clips add no
    view diversity and silently upweight one key-step ``k``-fold. That reasoning is
    wrong for this stage. ``tile_passes`` defines one exposure as 100% coverage and
    sets ``k = variants * ceil(L / window)`` precisely so that every key-step gets the
    SAME number of passes regardless of its length -- uniform ``k``-fold weighting is
    the invariant, not an accident. Deduplicating removed it for one subset only: the
    key-steps whose duration is exactly ``window``, which are the best-matched training
    examples the stage produces, since their geometry is identical to an eval window.

    Measured on the 11-video training split: 7 of 38 key-steps are exactly 3.000s, and
    deduplicating cut them from 4 clips to 1 -- 251 positives instead of 272, and 630
    WMCQ samples instead of the 680 the published recipe was validated with. Keeping
    the duplicates reproduces that dataset exactly.

    The degenerate interval is pinned whenever the key-step is exactly ``window`` long,
    or starts within a window of the video end with its annotated end at the video
    duration. That second shape is what annotation tools which clamp to video length
    produce for a trailing action, so this is systematic, not exotic.
    """
    if k <= 1:
        return [round((low + high) / 2.0, 3)]

    starts = [round(low + (high - low) * i / (k - 1), 3) for i in range(k)]
    distinct = len(set(starts))
    if distinct < k:
        logging.info(
            f"key-step {start:.3f}-{end:.3f}s: {distinct} distinct window position(s) for the "
            f"{k} passes requested; the remaining passes repeat the same clip, which is how "
            f"tile_passes keeps exposure equal across key-step lengths"
        )
    return sorted(starts)


def window_offsets(start: float, end: float, window: float, duration: float, k: int) -> List[float]:
    """``k`` window starts ``ws`` such that ``[ws, ws + window]`` fully contains ``[start, end]``.

    The key-step therefore sits at a different offset inside each window and the
    padding around it is the genuine surrounding footage -- exactly what a sliding
    window at inference time produces.
    """
    if duration - window <= 0:
        # Source video is shorter than one window; nothing to position.
        return [0.0]

    low, high = max(0.0, end - window), min(start, duration - window)
    if low > high:
        # No window position contains the whole key-step: it either runs past the end of
        # the video (a rounding-off or over-long annotation) or sits too close to an edge
        # to be padded on both sides. Centre it as far as the video allows.
        #
        # There is exactly ONE such position, so this returns a single window whatever k
        # was asked for -- returning k copies would just duplicate the same clip and add
        # no view diversity, which is the only reason to cut more than one. Warn, because
        # it silently undercuts --variants for this key-step and the sample counts alone
        # will not show it.
        if k > 1:
            logging.warning(
                f"key-step {start:.3f}-{end:.3f}s cannot be contained by a {window}s window "
                f"(video is {duration:.3f}s); cutting 1 window instead of {k}"
            )
        return [round(max(0.0, min(start - (window - (end - start)) / 2.0, duration - window)), 3)]
    return _spread(low, high, k, start, end)


def tiled_offsets(start: float, end: float, window: float, duration: float, k: int) -> List[float]:
    """``k`` window starts that tile a key-step longer than ``window``.

    Each window lies *inside* ``[start, end]`` rather than containing it, so every
    clip stays exactly ``window`` long and duration cannot leak the class.
    """
    low, high = start, end - window
    if high <= low:
        # Not actually longer than the window; fall back to containment.
        return window_offsets(start, end, window, duration, k)

    low = max(0.0, min(low, duration - window))
    high = max(0.0, min(high, duration - window))
    return _spread(low, high, k, start, end)


def overlaps_other_keystep(window_start: float, window_len: float, own: int,
                          keysteps: List[Tuple[float, float]]) -> bool:
    """Does this window also contain some key-step other than the one it is labelled with?

    A positive window has to *contain* its key-step, so it can only slide by
    ``window - keystep_length`` -- the slack. Any neighbouring key-step closer than that
    slack ends up inside the window, which still carries a single-action label. Short
    key-steps are the exposed case: the shorter the action, the more slack the window has
    to wander into its neighbour.
    """
    return any(i != own and window_start < end and window_start + window_len > start
               for i, (start, end) in enumerate(keysteps))


def plan_windows(
    start: float,
    end: float,
    duration: float,
    variants: int,
    args,
) -> Tuple[float, List[float], str]:
    """Decide the clip length, the window starts, and which geometry was used."""
    length = end - start

    if length <= args.window:
        return args.window, window_offsets(start, end, args.window, duration, variants), \
            const.GEOMETRY_MATCHED

    if args.tile_long:
        # Windows needed to cover the key-step exactly once.
        if args.tile_passes:
            variants = variants * int(math.ceil(length / args.window))
        return args.window, tiled_offsets(start, end, args.window, duration, variants), \
            const.GEOMETRY_MATCHED

    window = length + args.enlarge_pad
    return window, window_offsets(start, end, window, duration, variants), const.GEOMETRY_ENLARGED


def cut_window(clip, window_start: float, window_len: float, fps: float, size, output_path: str):
    """Cut ``[window_start, window_start + window_len]`` out of an open clip and write it.

    Returns the length actually written, which is shorter than ``window_len`` when the
    window runs off the end of the source video, or ``None`` if nothing was written.
    The caller records the returned value rather than the requested one: the metadata is
    the only audit trail this stage has for geometry drift, so it must not claim a clip
    is a full window when it is not.

    Logs instead of raising, so one unreadable window does not abort the whole stage.
    """
    try:
        window_end = min(window_start + window_len, clip.duration)
        sub = clip.subclip(window_start, window_end)
        frames = list(sub.iter_frames(fps=fps, dtype="uint8"))
        if not frames:
            logging.warning(f"No frames decoded for {output_path} at {window_start:.3f}s -- skipping")
            return None
        write_video(frames, output_path, fps, size)
    except Exception as exc:  # noqa: BLE001 - one bad window must not kill the stage
        logging.warning(f"Failed to cut {output_path} at {window_start:.3f}s: {exc}")
        return None
    return window_end - window_start


def assemble_anns(
    video_name: str,
    action_index: int,
    pos_or_neg: str,
    window_start: float,
    window_len: float,
    source_keystep: Optional[List[float]],
    geometry: str,
    overlaps_other: bool,
    question: str,
    choices: List[str],
    human_suffix: str,
) -> dict:
    """Build one LLaVA-format QA record for a cut window."""
    qa_label = copy.deepcopy(llava_video)
    qa_label[const.CONV][0][const.VALUE] = f"{human_suffix}{question}"
    qa_label[const.CONV][1][const.VALUE] = choices[action_index]
    qa_label[const.VIDEO] = f"videos/{video_name}"

    qa_meta = copy.deepcopy(wmcq_meta)
    qa_meta[const.GT_ACTION] = choices[action_index]
    qa_meta[const.POS_OR_NEG] = pos_or_neg
    qa_meta[const.WINDOW_START] = round(window_start, 3)
    qa_meta[const.WINDOW_LEN] = round(window_len, 3)
    qa_meta[const.SOURCE_KEYSTEP] = source_keystep
    qa_meta[const.GEOMETRY] = geometry
    qa_meta[const.OVERLAPS_OTHER] = overlaps_other
    qa_label[const.META] = qa_meta

    return qa_label


def negative_slots(
    duration: float,
    window: float,
    keysteps: List[Tuple[float, float]],
    margin: float,
    stride: float = NEGATIVE_GRID_SEC,
) -> List[float]:
    """Window starts whose whole window avoids every key-step by at least ``margin``.

    Candidates are laid out on a ``stride``-second grid and the caller samples the number
    it needs from the result at random, so ``stride`` sets the spacing of the candidates,
    not how many negatives are produced -- that is ``neg_ratio`` times the positive count.
    """
    if stride <= 0:
        # Not reachable from config -- the stride is fixed at NEGATIVE_GRID_SEC -- but this
        # function is a callable primitive and a non-positive stride never advances
        # `position`: the loop below would spin forever appending slots until the worker
        # exhausted memory, with nothing upstream timing the subprocess out.
        raise ValueError(f"negative-window grid stride must be positive, got {stride}")

    blocked = [(s - margin, e + margin) for s, e in keysteps]
    slots, position = [], 0.0
    while position <= duration - window:
        if all(not (position < b_end and position + window > b_start) for b_start, b_end in blocked):
            slots.append(round(position, 3))
        position += stride
    return slots


def process_chunk(video_root, video, output_root, seed, args):
    """Generate every WMCQ sample for a single source video."""
    if seed is not None:
        # Offset by the video name, not just the seed. Seeding every worker identically
        # makes two videos with equal-length slot lists draw the SAME shuffle permutation,
        # so their negatives land at the same relative offsets along the candidate grid --
        # deterministic as intended, but correlated across videos. Keying on the name keeps
        # the run reproducible and independent of how the videos happen to be ordered.
        random.seed(seed + zlib.crc32(video.encode("utf-8")))
    else:
        random.seed(os.getpid())

    source_video = os.path.join(video_root, f"{video}.{args.ext}")
    annotation_path = os.path.join(video_root, video, f"{video}_annotation.json")

    if not os.path.isfile(source_video):
        logging.warning(f"No source video for '{video}' at {source_video} -- skipping")
        return []
    if not os.path.isfile(annotation_path):
        logging.warning(f"No annotation for '{video}' at {annotation_path} -- skipping")
        return []

    frame_count, fps, size = get_video_meta(source_video)
    duration = frame_count / fps if fps else 0.0
    if duration <= 0:
        logging.warning(f"Could not determine duration of {source_video} -- skipping")
        return []

    video_out_root = os.path.join(output_root, video)
    create_dir(video_out_root)

    events = read_json(annotation_path)
    if not isinstance(events, list):
        events = events.get("annotations", [])

    anns = []
    per_action_count: Dict[int, int] = {}
    timeline_index = 0

    def emit(clip, action, pos_or_neg, window_start, window_len, source_keystep, geometry,
             overlaps_other=False):
        nonlocal timeline_index
        per_action_count[action] = per_action_count.get(action, 0) + 1
        timeline_index += 1
        # Same naming convention as the video split:
        # <action_number>_<video_name>_<duplication_cnt>_<timeline_index>.<ext>
        video_name = (
            f"{action}{const.VIDEO_ACTION_SEP}{video}{const.VIDEO_ACTION_SEP}"
            f"{per_action_count[action]}{const.VIDEO_ACTION_SEP}{timeline_index}.{args.ext}"
        )
        actual_len = cut_window(clip, window_start, window_len, fps, size,
                                os.path.join(video_out_root, video_name))
        if actual_len is None:
            per_action_count[action] -= 1
            timeline_index -= 1
            return
        if actual_len < window_len - 1e-3:
            # The window ran off the end of the source video. Record what was written,
            # not what was asked for, and say so -- report_geometry reads only this
            # metadata, so recording the requested length would report a uniform window
            # everywhere and leave the audit blind exactly when geometry has drifted.
            logging.warning(
                f"{video}: window at {window_start:.3f}s was truncated to {actual_len:.3f}s "
                f"by the end of the video ({args.ext} is {duration:.3f}s, asked for "
                f"{window_len:.3f}s)"
            )
            geometry = const.GEOMETRY_TRUNCATED
        anns.append(assemble_anns(
            video_name, action - 1, pos_or_neg, window_start, actual_len,
            source_keystep, geometry, overlaps_other, args.question, args.choices,
            args.human_suffix,
        ))

    with VideoFileClip(source_video) as clip:
        # ---- positives: real windows covering each key-step -------------------------
        # Two passes: every key-step has to be known before any window is planned, so a
        # window can be checked against the key-steps that follow it as well as those
        # before. Pass one also records key-steps we cannot label, because those regions
        # are still not non-SOP and must not be offered up as negatives.
        keysteps: List[Tuple[float, float]] = []
        labelled = []
        for event in events:
            action = parse_action_index(event)
            if action is None or action == args.non_sop_action:
                continue

            start = float(event["start_timestamp"])
            end = float(event["end_timestamp"])
            index = len(keysteps)
            keysteps.append((start, end))

            # Bounds-check before the action index reaches assemble_anns, where it indexes
            # into choices. An out-of-range action in a source annotation would otherwise
            # raise a bare IndexError inside a worker process, which ProcessPoolExecutor
            # re-raises from future.result() and which fails the whole stage with a trace
            # pointing at assemble_anns rather than at the video and event responsible.
            if not 1 <= action <= len(args.choices):
                logging.warning(
                    f"{video}: key-step {start:.3f}-{end:.3f}s has action {action}, outside "
                    f"the action list (1..{len(args.choices)}) -- skipping it"
                )
                continue

            if action in args.exclude_actions:
                continue
            labelled.append((index, action, start, end))

        for index, action, start, end in labelled:
            variants = args.variants_per_action_map.get(action, args.variants)
            window_len, starts, geometry = plan_windows(start, end, duration, variants, args)
            for window_start in starts:
                emit(clip, action, const.POS, window_start, window_len, [start, end], geometry,
                     overlaps_other_keystep(window_start, window_len, index, keysteps))

        num_positive = len(anns)

        # ---- negatives: same-length windows containing no key-step ------------------
        # Negatives are drawn per source video, so the ratio holds within each video and
        # therefore across the dataset. Per-video is what makes the stage parallelisable:
        # a dataset-wide count would need a barrier across workers.
        num_neg = int(round(args.neg_ratio * num_positive))
        slots = negative_slots(duration, args.window, sorted(keysteps), args.neg_margin)
        random.shuffle(slots)
        if num_neg > len(slots):
            logging.warning(
                f"{video}: only {len(slots)} non-SOP window(s) available but {num_neg} "
                f"requested; the video may be mostly key-steps or neg_margin may be too large"
            )
        for window_start in slots[:num_neg]:
            emit(clip, args.non_sop_action, const.NEG, window_start, args.window, None,
                 const.GEOMETRY_MATCHED)

    logging.info(
        f"{video}: {len(anns)} WMCQ sample(s) "
        f"({num_positive} positive, {len(anns) - num_positive} negative)"
    )
    return anns


def process_video(args_tuple):
    video_root, video, output_root, seed, args = args_tuple
    return process_chunk(video_root, video, output_root, seed, args)


def report_geometry(annotations: List[dict], window: float) -> None:
    """Log the clip-duration spread per action, and warn when duration leaks the class.

    This is the failure the tiling options exist to prevent, and it is invisible in
    the sample counts: if one action's clips are systematically longer than the rest,
    the model can separate that class on duration alone -- a feature that does not
    exist at inference, where every window is the same length.
    """
    by_action: Dict[str, List[float]] = {}
    enlarged = truncated = overlapping = positives = 0
    for ann in annotations:
        meta = ann.get(const.META, {})
        by_action.setdefault(meta.get(const.GT_ACTION, "?"), []).append(meta.get(const.WINDOW_LEN, 0))
        if meta.get(const.GEOMETRY) == const.GEOMETRY_ENLARGED:
            enlarged += 1
        elif meta.get(const.GEOMETRY) == const.GEOMETRY_TRUNCATED:
            truncated += 1
        if meta.get(const.POS_OR_NEG) == const.POS:
            positives += 1
            if meta.get(const.OVERLAPS_OTHER):
                overlapping += 1

    for action, lengths in sorted(by_action.items()):
        logging.info(
            f"clip duration [{action}]: n={len(lengths)} "
            f"min={min(lengths):.2f}s max={max(lengths):.2f}s"
        )

    if overlapping:
        logging.warning(
            f"{overlapping} of {positives} positive window(s) also contain a NEIGHBOURING "
            f"key-step, but carry a single-action label -- those labels are incomplete. A "
            f"positive window can slide by (window - key-step length), so any key-step closer "
            f"than that ends up inside it; short key-steps are the exposed case. WMCQ assumes "
            f"long videos with key-steps spaced further apart than that slack. If this fraction "
            f"is large the dataset is densely annotated, which is not the shape this stage is "
            f"for -- prefer the chunk-consuming augmentations."
        )

    if truncated:
        logging.warning(
            f"{truncated} of {len(annotations)} clip(s) are SHORTER than the {window}s window "
            f"because they ran off the end of their source video. They do not match eval "
            f"geometry either -- check that the source videos are longer than the window and "
            f"that no annotation runs past the end of its video."
        )

    if enlarged:
        logging.warning(
            f"{enlarged} of {len(annotations)} clip(s) are longer than the {window}s window "
            f"because a key-step did not fit and --tile-long is off. Clip duration now "
            f"correlates with the action, which is a shortcut the model cannot use at "
            f"inference (every eval window is the same length). Enable tile_long "
            f"(optionally with tile_passes) to keep every clip exactly {window}s."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-root", type=str, default="", help="root path for storing question.txt and choices.txt"
    )
    parser.add_argument("--action-json", type=str, default="", help="action json file from labelling tool")
    parser.add_argument("--video-root", type=str, required=True,
                        help="label data root: full source videos plus per-video annotation folders")
    parser.add_argument("--ext", type=str, default="mp4", help="video extension format")
    parser.add_argument("--exclude-action", type=str, default="",
                        help="actions to exclude from WMCQ positives, underscore separated")
    parser.add_argument("--non-sop-action", type=int, required=True,
                        help="action index of the non-SOP action; it is the label for every negative")
    parser.add_argument("--window", type=float, default=3.0,
                        help="clip length in seconds. MUST equal the evaluation sliding-window "
                             "length -- matching it is the entire point of this augmentation")
    parser.add_argument("--variants", type=int, default=4,
                        help="windows cut per key-step (see --tile-passes for how this is read "
                             "when a key-step is longer than the window)")
    parser.add_argument("--variants-per-action", type=str, default="",
                        help="per-action override for --variants, e.g. '2:8' cuts 8 windows per "
                             "action-2 key-step. Use to counterweight a class the model keeps "
                             "missing: raising overall negative pressure tends to kill the "
                             "visually subtlest class first")
    parser.add_argument("--enlarge-pad", type=float, default=1.0,
                        help="padding added when a key-step is longer than the window and "
                             "--tile-long is off")
    parser.add_argument("--tile-long", type=str2bool, nargs="?", const=True, default=True,
                        help="tile key-steps longer than --window into several windows of exactly "
                             "--window instead of enlarging the window, so clip duration cannot "
                             "leak the class")
    parser.add_argument("--tile-passes", type=str2bool, nargs="?", const=True, default=True,
                        help="with --tile-long, read --variants as full passes over the key-step: "
                             "k = variants * ceil(L / window), so one pass means 100%% coverage "
                             "for every class regardless of its duration")
    parser.add_argument("--neg-ratio", type=float, default=1.5,
                        help="non-SOP window negatives per positive, drawn per source video")
    parser.add_argument("--neg-margin", type=float, default=0.5,
                        help="keep negative windows at least this many seconds clear of any key-step")
    parser.add_argument("--human-suffix", type=str, default=const.DEFAULT_HUMAN_SUFFIX)
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--output-root", type=str, required=True, help="output root for storing json file")
    # required, unlike the sibling stages: omitted, the f-string below writes "None.json",
    # which the importer then skips as a stage folder with no matching <folder>.json.
    parser.add_argument("--output-name", type=str, required=True, help="file name to be saved")
    args = parser.parse_args()

    # check if config or action json is provided
    if args.config_root == "" and args.action_json == "":
        raise ValueError("Must provide either 'config-root' or 'action-json'.")
    elif args.config_root:
        logging.info("Config root provided. Use config for generation.")
    else:
        logging.info("Use action json file for generation.")
        args.config_root = prepare_sample_choices(args.action_json)

    # split exclude action into a list of int
    if args.exclude_action != "":
        args.exclude_actions = [int(action) for action in args.exclude_action.split(const.VIDEO_ACTION_SEP)]
    else:
        args.exclude_actions = []

    # per-action variant overrides, "2:8,3:2"
    args.variants_per_action_map = {}
    if args.variants_per_action:
        for part in args.variants_per_action.split(","):
            action, count = part.split(":")
            args.variants_per_action_map[int(action)] = int(count)

    if args.window <= 0:
        raise ValueError("'window' must be positive.")
    if args.neg_ratio < 0:
        raise ValueError("'neg-ratio' must not be negative.")
    if args.tile_passes and not args.tile_long:
        raise ValueError("'tile-passes' only applies when 'tile-long' is enabled.")
    logging.info(
        f"WMCQ window = {args.window}s. This must equal the evaluation sliding-window length; "
        f"if the two diverge the clips stop matching eval geometry and the augmentation loses "
        f"its purpose while still producing plausible-looking output."
    )

    if args.seed is not None:
        random.seed(args.seed)

    create_dir(args.output_root)

    # load question and choices
    args.question = read_txt(os.path.join(args.config_root, f"{const.QUESTION}.txt"))
    args.choices = read_txt(os.path.join(args.config_root, f"{const.CHOICES}.txt")).split(const.LINE_BREAK)

    if not 1 <= args.non_sop_action <= len(args.choices):
        raise ValueError(
            f"'non-sop-action' {args.non_sop_action} is outside the action list "
            f"(1..{len(args.choices)})."
        )

    # Load all available videos: one folder per annotated source video. Identified by
    # the annotation file rather than by "is a directory", because prepare_sample_choices
    # writes its question/choices config into this same root when --action-json points
    # at the dataset's actions.json, and that config folder is not a video.
    all_videos = [
        entry for entry in sorted(os.listdir(args.video_root))
        if os.path.isfile(os.path.join(args.video_root, entry, f"{entry}_annotation.json"))
    ]
    if not all_videos:
        raise ValueError(f"No annotated video folders found in {args.video_root}")


    args_list = [
        (
            args.video_root,
            video,
            args.output_root,
            args.seed,
            args,
        )
        for video in all_videos
    ]

    # Use multiprocessing Pool
    # -1 to leave one core for the main process
    with ProcessPoolExecutor(max_workers=max(1, cpu_count() - 1)) as executor:
        futures = [executor.submit(process_video, arg) for arg in args_list]
        annotations = [future.result() for future in futures]

    # unpack annotations
    final_annotations = unpack_annotation(annotations)

    report_geometry(final_annotations, args.window)
    logging.info(f"WMCQ: {len(final_annotations)} sample(s) -> {args.output_root}")

    # save annotations
    dump_json(os.path.join(args.output_root, f"{args.output_name}.json"), final_annotations)

    # copy all videos to videos folder
    create_dir(os.path.join(args.output_root, "videos"))
    all_prc_videos = glob.glob(os.path.join(args.output_root, "*", f"*.{args.ext}"))

    for cur_video in all_prc_videos:
        os.replace(cur_video, os.path.join(args.output_root, "videos", os.path.basename(cur_video)))
