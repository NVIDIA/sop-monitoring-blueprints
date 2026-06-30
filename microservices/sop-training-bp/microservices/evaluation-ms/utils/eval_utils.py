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
"""Utility helpers for the per-action-chunk VLM evaluation pipeline."""

import glob
import json
import logging
import os
import re
import string
from typing import Optional

import utils.constant as const

logger = logging.getLogger(__name__)


def build_vllm_video_mm_data(video_inputs, video_kwargs, default_fps):
    """
    Pair each video tensor with a metadata dict for vLLM 0.11.0+.

    vLLM's Qwen3VLMultiModalProcessor (cosmos-reason2 / Qwen3-VL) declares
    ``video_needs_metadata=True`` and expects:

        mm_data["videos"] = [(video_array, metadata_dict), ...]

    where ``metadata_dict`` has at minimum:
        fps, total_num_frames, frames_indices, video_backend, do_sample_frames.

    ``qwen_vl_utils.process_vision_info`` returns ``video_kwargs`` as a dict of
    list-keyed values (one entry per video). We unpack the per-video slice
    here. ``do_sample_frames=False`` because qwen_vl_utils has already sampled
    frames at the requested fps.
    """
    if not video_inputs:
        return None

    fps_list = (video_kwargs or {}).get("fps", [default_fps])
    if not isinstance(fps_list, list):
        fps_list = [fps_list]

    items = []
    for i, video_arr in enumerate(video_inputs):
        fps_val = float(fps_list[i] if i < len(fps_list) else fps_list[0])
        try:
            total_frames = int(video_arr.shape[0])
        except (AttributeError, IndexError):
            total_frames = len(video_arr)
        items.append((
            video_arr,
            {
                "fps": fps_val,
                "duration": (total_frames / fps_val) if fps_val > 0 else 0.0,
                "total_num_frames": total_frames,
                "frames_indices": list(range(total_frames)),
                "video_backend": "torchvision",
                "do_sample_frames": False,
            },
        ))
    return items


def resolve_checkpoint_path(results_root: str, job_id: str, step: Optional[int] = None) -> tuple:
    """
    Find the checkpoint directory for a training job.

    Args:
        results_root: Root results directory (const.RESULTS_ROOT)
        job_id: Training job UUID
        step: If provided, validate and return that specific step; otherwise use latest.

    Returns:
        (absolute_checkpoint_path, step_number)

    Raises:
        FileNotFoundError: if no checkpoints exist or requested step not found
    """
    job_output_dir = os.path.join(results_root, job_id)
    # cosmos-rl v0.3.9 writes <job>/<run_ts>/safetensors/step_<N>/;
    # older versions wrote <job>/step_<N>/. Recursive glob handles both.
    step_dirs = [
        d for d in glob.glob(os.path.join(job_output_dir, "**", "step_*"), recursive=True)
        if os.path.isdir(d)
    ]

    if not step_dirs:
        raise FileNotFoundError(
            f"No checkpoints found for job {job_id} in {job_output_dir}"
        )

    step_map = {}
    for d in step_dirs:
        try:
            n = int(os.path.basename(d.rstrip("/")).split("_")[1])
            # On training resume the same step can appear under multiple
            # run timestamps; keep the most recently written.
            existing = step_map.get(n)
            if existing is None or os.path.getmtime(d) > os.path.getmtime(existing):
                step_map[n] = d.rstrip("/")
        except (IndexError, ValueError):
            pass

    if not step_map:
        raise FileNotFoundError(f"Could not parse step numbers from dirs in {job_output_dir}")

    if step is not None:
        if step not in step_map:
            raise FileNotFoundError(
                f"Requested step {step} not found for job {job_id}. Available: {sorted(step_map)}"
            )
        return step_map[step], step

    latest = max(step_map)
    return step_map[latest], latest


_NON_LETTERS = string.printable.translate(str.maketrans("", "", string.ascii_letters))


def _clean_action(sentence: str) -> str:
    """Remove leading/trailing non-letter chars (mirrors vlm_aug clean_sentence)."""
    return sentence.strip(_NON_LETTERS)


_MCQ_TEMPLATE = (
    "There are {step_count} possible steps for the SOP "
    "(Standard Operation Procedure) of the given video.\n"
    "What step is the operator doing?\n"
)


def extract_mcq_data(actions_json_path: str) -> tuple:
    """
    Generate MCQ prompt and ordered choices from an actions.json file.

    Uses the same mechanism as the data-augmentation pipeline
    (config_to_sequential_mcq): read actions, clean, number, and
    apply the standard MCQ template.  This removes the dependency on
    a pre-existing sequential_mcq.json augmentation output.

    Args:
        actions_json_path: Path to the actions.json file

    Returns:
        (prompt_text, choices_list) where choices_list is ["(1) label", "(2) label", ...]

    Raises:
        FileNotFoundError: if actions.json does not exist
        ValueError: if no actions are found
    """
    if not os.path.exists(actions_json_path):
        raise FileNotFoundError(f"actions.json not found: {actions_json_path}")

    with open(actions_json_path, "r") as f:
        data = json.load(f)

    actions = data.get("actions", [])
    if not actions:
        raise ValueError(f"No actions found in {actions_json_path}")

    choices = [f"({i}) {_clean_action(a)}" for i, a in enumerate(actions, 1)]

    prompt = _MCQ_TEMPLATE.format(step_count=len(actions)) + "\n".join(choices)
    return prompt, choices


def prepare_eval_assets(eval_job_id: str, prompt_text: str) -> str:
    """
    Write vlm_prompts.txt for the eval subprocess into a per-job assets directory.

    Returns:
        Path to the asset directory (pass as --asset-root to sop_eval.py)
    """
    asset_dir = os.path.join(const.RESULTS_ROOT, eval_job_id, "assets")
    os.makedirs(asset_dir, exist_ok=True)

    prompts_path = os.path.join(asset_dir, "vlm_prompts.txt")
    with open(prompts_path, "w") as f:
        f.write(prompt_text)

    logger.info(f"Wrote vlm_prompts.txt to {asset_dir}")
    return asset_dir


def _letter_to_number(letters: list) -> list:
    """Convert letter answers ['A','B',...] to 1-based number strings ['1','2',...]."""
    result = []
    for letter in letters:
        if len(letter) == 1 and letter.isalpha():
            result.append(str(ord(letter.upper()) - ord("A") + 1))
    return result


def _get_choice_label(action_idx: int, choices: list, fallback: str = "") -> str:
    """Return the MCQ choice string for a 1-based action index, or fallback if out of range."""
    if 0 < action_idx <= len(choices):
        return choices[action_idx - 1]
    return fallback


def verify_pred(pred: str, gt: str) -> bool:
    """
    Check if VLM response matches ground truth MCQ choice.

    Tries multiple matching strategies:
    1. Exact string equality
    2. Matching action number: (N) in both pred and gt
    3. Letter answer A→1, B→2 matching action number
    4. Case-insensitive text match after stripping leading (N)
    5. Answer tag extraction: <answer>...</answer>
    """
    answer_in_tags = [m.strip() for m in re.findall(r"<answer>([^<]*)</answer>", pred)]
    try:
        if answer_in_tags:
            answer_json = json.loads(answer_in_tags[0])
            answer_in_tags = [answer_json.get("answer", answer_in_tags[0])]
    except Exception:
        pass

    pred_cls = re.findall(r"\((\d+)\)", pred)
    label_cls = re.findall(r"\((\d+)\)", gt)
    pred_text = re.sub(r"^\(\d+\)\s*", "", pred).strip()
    gt_text = re.sub(r"^\(\d+\)\s*", "", gt).strip()

    try:
        if pred == gt:
            return True
        if pred_cls and label_cls and pred_cls == label_cls:
            return True
        if answer_in_tags and _letter_to_number(answer_in_tags) == label_cls:
            return True
        if pred_text.lower() == gt_text.lower() and gt_text:
            return True
    except Exception:
        pass
    return False


def parse_eval_results(inference_results: dict, choices: list) -> dict:
    """
    Compute overall and per-action accuracy from raw VLM inference results.

    Args:
        inference_results: {video_name: [[action_ids_1based, vlm_response, chunk_path?], ...]}
                           Accepts 2-tuple (legacy) or 3-tuple (current); the
                           optional third element is the chunk's on-disk path
                           consumed by the RCA parser only.
                           action_ids may be an int (single-op, legacy) or a
                           list of ints (concurrent two-op chunk).
        choices: ordered MCQ choice strings ["(1) do step one", "(2) do step two", ...]
                 index 0 = action 1, index 1 = action 2, etc.

    Grading:
      - single-op chunk: correct iff `verify_pred` matches the single gt label.
      - two-op chunk (multiple gt ids): correct iff the set of `(N)` tokens
        in the VLM response equals the gt set. Each gt id gets a "total"
        increment, and a "correct" increment when the chunk passes.

    Returns:
        {
            "overall_accuracy": float,
            "per_action": {
                "1": {"label": str, "correct": int, "total": int, "accuracy": float},
                ...
            }
        }
    """
    per_action: dict = {}
    total_correct = 0
    total_samples = 0

    for video, preds in inference_results.items():
        for entry in preds:
            # Accept legacy 2-tuple and current 3-tuple [action, response, chunk_path].
            # action_ids may be int (legacy single-op) or list[int] (two-op).
            raw_ids, vlm_response = entry[0], entry[1]
            gt_ids = [raw_ids] if isinstance(raw_ids, int) else sorted(int(i) for i in raw_ids)

            if len(gt_ids) == 1:
                expected_label = _get_choice_label(gt_ids[0], choices)
                is_correct = verify_pred(vlm_response, expected_label)
            else:
                # Two-op grading is intentionally stricter than verify_pred —
                # the model must enumerate each concurrent action as a "(N) <text>"
                # token; letter-format and <answer>-tag fallbacks don't apply.
                pred_ids = sorted(int(n) for n in re.findall(r"\((\d+)\)", vlm_response))
                is_correct = pred_ids == gt_ids

            for aid in gt_ids:
                key = str(aid)
                if key not in per_action:
                    label = _get_choice_label(aid, choices, f"(?) unknown action {aid}")
                    per_action[key] = {"label": label, "correct": 0, "total": 0, "accuracy": 0.0}
                per_action[key]["total"] += 1
                if is_correct:
                    per_action[key]["correct"] += 1

            # one chunk = one sample for overall accuracy
            total_correct += int(is_correct)
            total_samples += 1

    for key in per_action:
        t = per_action[key]["total"]
        per_action[key]["accuracy"] = per_action[key]["correct"] / t if t > 0 else 0.0

    overall = total_correct / total_samples if total_samples > 0 else 0.0
    return {"overall_accuracy": overall, "per_action": per_action}
