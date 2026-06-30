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

"""
Spatial Localization Augmentation Module

Generates spatial-discrimination Q&A pairs for visually similar actions that
differ only in spatial location.  Confusion pairs are derived automatically
from the user-provided action list (Phase 1) and question templates are
optionally refined by the same LLM used in gqa_to_gqas (Phase 2).
"""

import argparse
import copy
import glob
import os
import re
import random
import shutil
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
from openai import OpenAI

from .cfg.spatial import (
    FALLBACK_CONTRAST_ANSWER_MAP_TEMPLATE,
    FALLBACK_CONTRAST_QUESTION,
    FALLBACK_REGION_ANSWER_MAP,
    FALLBACK_REGION_QUESTION,
    MAX_QA_PER_GROUP,
    MAX_QUESTION_TOKENS,
    SPATIAL_KEYWORDS,
    spatial_llm_cfg,
)
from .utils import const
from .utils.annotation_template import llava_video
from .utils.helper import (
    create_dir,
    dump_json,
    parse_video_action_indices,
    read_json,
    read_txt,
    resolve_llm_base_url,
    str2bool,
    unpack_annotation,
    write_txt,
)
from .utils.logger import logging
from .gqa_to_gqas import normalize_llm_output as _normalize_llm_output


# ============================================================================
# Phase 1 – Deterministic confusion pair extraction
# ============================================================================

def extract_action_components(action: str) -> Tuple[str, str, str]:
    """Parse an action description into (verb, object, spatial_modifier).

    Strips leading action index prefixes like ``(1)``, ``(10)``.
    Uses the **last** spatial keyword as the modifier (the distinguishing
    region, e.g. "upper" in "Remove the upper tube from the **lower** chassis"
    should yield spatial="lower" because "lower chassis" is the region).
    """
    # Strip leading index prefix like "(1) ", "(10) "
    cleaned_action = re.sub(r"^\(\d+\)\s*", "", action.strip())
    tokens = re.split(r"\s+", cleaned_action.lower())
    if not tokens:
        return ("", "", "")

    verb = tokens[0]
    spatial = ""
    obj_tokens = []

    for tok in tokens[1:]:
        cleaned = re.sub(r"[^a-z]", "", tok)
        if cleaned in SPATIAL_KEYWORDS:
            # Always update -- take the LAST spatial keyword as the
            # region-level modifier (closest to the location noun).
            spatial = cleaned
        else:
            obj_tokens.append(cleaned)

    obj = " ".join(t for t in obj_tokens if t)
    return (verb, obj, spatial)


def auto_generate_confusion_pairs(actions: List[str]) -> Dict:
    """Group actions that share (verb, object) but differ in spatial modifier.

    Returns dict keyed by group label with structure:
      {
        "group_label": {
            "actions": [idx_a, idx_b, ...],   # 1-based
            "spatial_dimension": "bottom vs upper",
            "shared_context": "tube removal from chassis",
            "descriptions": {idx: "full description", ...}
        }
      }
    """
    groups: Dict[Tuple[str, str], List[Tuple[int, str, str]]] = defaultdict(list)

    for i, action in enumerate(actions, 1):
        verb, obj, spatial = extract_action_components(action)
        if spatial:
            groups[(verb, obj)].append((i, spatial, action))

    confusion_pairs: Dict[str, Dict] = {}
    label_counter = 0

    for (verb, obj), members in groups.items():
        if len(members) < 2:
            continue

        spatials = {m[1] for m in members}
        if len(spatials) < 2:
            continue

        label_counter += 1
        label = f"group_{label_counter}_{verb}_{obj.replace(' ', '_')[:20]}"

        action_indices = [m[0] for m in members]
        descriptions = {m[0]: m[2] for m in members}
        spatial_list = sorted({m[1] for m in members})
        spatial_dim = " vs ".join(spatial_list)
        shared = f"{verb} {obj}" if obj else verb

        confusion_pairs[label] = {
            "actions": action_indices,
            "spatial_dimension": spatial_dim,
            "shared_context": shared,
            "descriptions": descriptions,
        }

    return confusion_pairs


# ============================================================================
# Phase 2 – LLM-assisted template generation (one call per dataset)
# ============================================================================

QA_SEP = "===\n"
Q_START = "Question:\n"
A_START = "Answer:\n"


def validate_spatial_qa(question: str, answer: str, max_q_tokens: int = MAX_QUESTION_TOKENS) -> bool:
    """Return True if the Q&A passes training-stability constraints."""
    q_tokens = len(question.split())
    if q_tokens > max_q_tokens:
        logging.warning(f"Question exceeds {max_q_tokens} tokens ({q_tokens}): {question[:80]}...")
        return False
    if not re.match(r"^\([A-Za-z0-9]+\)$", answer.strip()):
        logging.warning(f"Answer format invalid: '{answer.strip()}'")
        return False
    return True


def _parse_llm_qa_pairs(raw: str) -> List[Tuple[str, str]]:
    """Parse LLM output into (question, answer) tuples."""
    raw = _normalize_llm_output(raw)
    blocks = raw.split(QA_SEP)
    pairs: List[Tuple[str, str]] = []
    q, a = None, None
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if block.startswith(Q_START.strip()):
            q = block[len(Q_START.strip()):].strip()
        elif block.startswith(A_START.strip()):
            a = block[len(A_START.strip()):].strip()
        if q is not None and a is not None:
            pairs.append((q, a))
            q, a = None, None
    return pairs


def _prep_spatial_sys_prompts() -> List[dict]:
    """Load spatial localization system prompt and few-shot examples."""
    prompt_dir = os.path.join(const.PROMPTS_ROOT, "spatial_localization")
    system_msg = read_txt(os.path.join(prompt_dir, "system_message.txt"))
    messages = [{"role": "system", "content": system_msg}]

    all_caps = sorted(glob.glob(os.path.join(prompt_dir, "*_caps.txt")))
    all_convs = sorted(glob.glob(os.path.join(prompt_dir, "*_conv.txt")))

    for cap, conv in zip(all_caps, all_convs):
        messages.append({"role": "user", "content": read_txt(cap)})
        messages.append({"role": "assistant", "content": read_txt(conv)})

    return messages


def _build_llm_user_prompt(group: Dict, max_qa: int) -> str:
    """Build the user prompt for one confusion group."""
    descs = group["descriptions"]
    indices = group["actions"]
    lines = [
        f"Confusion group: {group.get('shared_context', 'unknown')}",
        f"Shared context: {group['shared_context']}",
    ]
    for i, idx in enumerate(indices):
        label = chr(ord("A") + i)
        lines.append(f"Action {label} (index {idx}): {descs[idx]}")
    lines.append(f"Spatial dimension: {group['spatial_dimension']}")
    lines.append("")
    lines.append(
        f"Generate {max_qa} question-answer pairs for when Action A (index {indices[0]}) is the ground truth."
    )
    return "\n".join(lines)


def _build_openai_client(args) -> OpenAI:
    """Create an OpenAI client from CLI/pipeline arguments."""
    base_url = resolve_llm_base_url(args.llm_type, args.local_llm_url, const.API_BASE_URL)
    return OpenAI(
        base_url=base_url,
        api_key=args.api_key if args.llm_type == "nvidia" else "not-used",
    )


def _call_llm_with_retry(
    client: OpenAI,
    messages: List[dict],
    args,
    llm_model: str,
    max_retries: int = 5,
    base_delay: int = 10,
) -> Optional[str]:
    """Call the LLM and return the raw text output, retrying on 429 rate limits.

    Returns ``None`` if all attempts fail.
    """
    call_kwargs = dict(
        model=llm_model,
        messages=messages,
        temperature=spatial_llm_cfg["temperature"],
        top_p=spatial_llm_cfg["top_p"],
        max_tokens=spatial_llm_cfg["max_tokens"],
        stream=False,
    )
    if hasattr(args, "enable_thinking") and args.enable_thinking != "":
        call_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": args.enable_thinking.lower() == "true"}
        }

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(**call_kwargs)
            content = completion.choices[0].message.content
            if not content and "extra_body" not in call_kwargs:
                logging.info("Empty content from LLM, retrying with enable_thinking=false")
                call_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
                completion = client.chat.completions.create(**call_kwargs)
                content = completion.choices[0].message.content
            return content
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logging.warning(f"Rate limited (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                time.sleep(delay)
            else:
                logging.error(f"LLM call failed: {e}")
                return None
    return None


def _derive_answer_map(answer: str, group_indices: List[int]) -> Dict:
    """Build an answer_map for all group indices from the ground-truth answer."""
    a_stripped = answer.strip()
    answer_map = {group_indices[0]: a_stripped}
    if re.match(r"^\(\d+\)$", a_stripped):
        for idx in group_indices:
            answer_map[idx] = f"({idx})"
    else:
        opposite = "(B)" if a_stripped == "(A)" else "(A)"
        for idx in group_indices[1:]:
            answer_map[idx] = opposite
    return answer_map


def _parse_and_validate_qa(
    raw_output: str,
    group_indices: List[int],
    max_qa_per_group: int,
) -> List[Dict]:
    """Parse raw LLM output into validated Q&A templates with answer maps.

    Returns a list of ``{"question": str, "answer_map": dict}`` dicts,
    capped at *max_qa_per_group* entries.
    """
    pairs = _parse_llm_qa_pairs(raw_output)
    templates: List[Dict] = []

    for q, a in pairs:
        if not validate_spatial_qa(q, a):
            continue
        answer_map = _derive_answer_map(a, group_indices)
        templates.append({"question": q, "answer_map": answer_map})
        if len(templates) >= max_qa_per_group:
            break

    return templates


def llm_generate_qa_templates(
    confusion_pairs: Dict,
    args,
    max_qa_per_group: int = MAX_QA_PER_GROUP,
) -> Dict:
    """Call LLM once per confusion group to produce Q&A templates.

    Adds a ``qa_templates`` list to each group in *confusion_pairs* (in-place)
    and returns the updated dict.
    """
    if not confusion_pairs:
        return confusion_pairs

    messages = _prep_spatial_sys_prompts()
    client = _build_openai_client(args)

    for label, group in confusion_pairs.items():
        user_prompt = _build_llm_user_prompt(group, max_qa_per_group)
        cur_messages = copy.deepcopy(messages)
        cur_messages.append({"role": "user", "content": user_prompt})

        logging.info(f"LLM spatial QA request for group '{label}'")

        raw_output = _call_llm_with_retry(client, cur_messages, args, args.llm)

        if raw_output is None:
            logging.warning(f"No LLM output for group '{label}', using fallback templates")
            group["qa_templates"] = _build_fallback_templates(group)
            continue

        templates = _parse_and_validate_qa(raw_output, group["actions"], max_qa_per_group)

        if not templates:
            logging.warning(f"All LLM Q&A rejected for group '{label}', using fallback")
            templates = _build_fallback_templates(group)

        group["qa_templates"] = templates

    return confusion_pairs


# ============================================================================
# Fallback templates (deterministic, no LLM)
# ============================================================================

def _build_fallback_templates(group: Dict) -> List[Dict]:
    """Build deterministic fallback Q&A templates from cfg/spatial.py."""
    indices = group["actions"]
    descs = group["descriptions"]
    templates = []

    if len(indices) < 2:
        return templates

    idx_a, idx_b = indices[0], indices[1]
    comp = extract_action_components(descs[idx_a])
    comp_b = extract_action_components(descs[idx_b])
    spatial_a = comp[2] or "position A"
    spatial_b = comp_b[2] or "position B"
    shared = group["shared_context"]

    region_q = FALLBACK_REGION_QUESTION.format(
        shared_context=shared, spatial_a=spatial_a, spatial_b=spatial_b,
    )
    templates.append({
        "question": region_q,
        "answer_map": {idx_a: "(A)", idx_b: "(B)"},
    })

    contrast_q = FALLBACK_CONTRAST_QUESTION.format(
        idx_a=idx_a, action_a=descs[idx_a],
        idx_b=idx_b, action_b=descs[idx_b],
    )
    templates.append({
        "question": contrast_q,
        "answer_map": {
            idx_a: FALLBACK_CONTRAST_ANSWER_MAP_TEMPLATE.format(idx=idx_a),
            idx_b: FALLBACK_CONTRAST_ANSWER_MAP_TEMPLATE.format(idx=idx_b),
        },
    })

    return templates


# ============================================================================
# Per-video processing
# ============================================================================

def assemble_ann(video_name: str, question: str, answer: str, human_suffix: str = "<video>\n"):
    """Build a single LLaVA-style annotation entry."""
    qa_label = copy.deepcopy(llava_video)
    human = question if question.lstrip().startswith("<video>") else f"{human_suffix}{question}"
    qa_label[const.CONV][0][const.VALUE] = human
    qa_label[const.CONV][1][const.VALUE] = answer
    qa_label[const.VIDEO] = f"videos/{video_name}"
    return qa_label


def _generate_spatial_anns_for_video(
    cur_video: str,
    confusion_pairs: Dict,
    args,
) -> List:
    """Generate spatial Q&A annotations for a single video clip.

    Parses action indices from the filename, checks exclusions, and iterates
    over confusion pairs/templates to build annotation entries.
    """
    vid_basename = Path(cur_video).stem
    action_indices, is_concurrent = parse_video_action_indices(
        vid_basename, two_operator_mode=args.two_operator_mode,
    )

    if not action_indices:
        return []

    if any(idx in args.exclude_actions for idx in action_indices):
        return []

    action_set: Set[int] = set(action_indices)
    anns = []

    for _label, group in confusion_pairs.items():
        templates = group.get("qa_templates", [])
        group_actions = set(group["actions"])
        matching = action_set & group_actions

        if not matching:
            continue

        gt_action = action_indices[0] if not is_concurrent else min(matching)

        for tmpl in templates:
            answer = tmpl["answer_map"].get(gt_action)
            if answer is not None:
                anns.append(assemble_ann(os.path.basename(cur_video), tmpl["question"], answer))

    return anns


def process_spatial(
    video_root: str,
    video: str,
    video_ext: str,
    output_root: str,
    confusion_pairs: Dict,
    args,
):
    """Process one video session folder, applying spatial templates to clips."""
    random.seed(os.getpid())

    anns = []
    all_videos = sorted(glob.glob(os.path.join(video_root, f"{video}/*.{video_ext}")))
    video_out_root = os.path.join(output_root, video)
    create_dir(video_out_root)

    for cur_video in all_videos:
        anns.extend(_generate_spatial_anns_for_video(cur_video, confusion_pairs, args))

    for cur_video in all_videos:
        shutil.copyfile(cur_video, os.path.join(video_out_root, os.path.basename(cur_video)))

    return anns


def process_video(args_tuple):
    (video_root, video, video_ext, output_root, confusion_pairs, args) = args_tuple
    return process_spatial(video_root, video, video_ext, output_root, confusion_pairs, args)


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spatial Localization Augmentation with auto confusion pair generation"
    )
    # LLM args (reuse from gqa_to_gqas)
    parser.add_argument("--llm-type", type=str, default="nvidia", choices=["nvidia", "local"])
    parser.add_argument("--local-llm-url", type=str, default="")
    parser.add_argument("--llm", type=str, default="meta/llama-3.1-70b-instruct")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--enable-thinking", type=str, default="")
    # Data args
    parser.add_argument("--action-json", type=str, required=True)
    parser.add_argument("--video-root", type=str, required=True)
    parser.add_argument("--ext", type=str, default="mp4")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--exclude-action", type=str, default="")
    parser.add_argument("--two-operator-mode", type=str2bool, default=False)
    parser.add_argument("--max-qa-per-group", type=int, default=MAX_QA_PER_GROUP)
    parser.add_argument("--max-question-tokens", type=int, default=MAX_QUESTION_TOKENS)
    args = parser.parse_args()

    # API key via NGC_API_KEY env var;
    # fall back to it when --api-key is not given on the command line.
    if not args.api_key:
        args.api_key = os.environ.get("NGC_API_KEY", "")

    if args.exclude_action:
        args.exclude_actions = [int(a) for a in args.exclude_action.split(const.VIDEO_ACTION_SEP)]
    else:
        args.exclude_actions = []

    # Validate LLM configuration (mirrors gqa_to_gqas)
    assert not (args.local_llm_url == "" and args.api_key == ""), (
        "Must provide 'local-llm-url' or 'api-key' (from Nvidia API)."
    )
    if args.llm_type == "nvidia":
        assert args.api_key != "", "Must provide 'api-key' (from Nvidia API)."
        logging.info("Use Nvidia API LLM.")
    elif args.llm_type == "local":
        assert args.local_llm_url != "", "Must provide 'local-llm-url'."
        logging.info(f"Use local LLM on {args.local_llm_url}.")
    else:
        raise ValueError(f"Invalid LLM type: {args.llm_type}")

    # Phase 1: auto-generate confusion pairs
    all_actions = read_json(args.action_json)[const.ACTION_JSON_KEY]
    confusion_pairs = auto_generate_confusion_pairs(all_actions)

    if not confusion_pairs:
        logging.info("No spatially confusable action pairs found. Skipping spatial localization.")
        create_dir(args.output_root)
        dump_json(os.path.join(args.output_root, f"{args.output_name}.json"), [])
    else:
        logging.info(f"Found {len(confusion_pairs)} confusion group(s): "
                     f"{list(confusion_pairs.keys())}")

        # Phase 2: LLM-assisted template generation
        has_llm = (args.api_key != "" or args.local_llm_url != "")
        if has_llm:
            logging.info("Using LLM to generate spatial Q&A templates")
            confusion_pairs = llm_generate_qa_templates(
                confusion_pairs, args, max_qa_per_group=args.max_qa_per_group,
            )
        else:
            logging.info("No LLM configured, using fallback templates")
            for group in confusion_pairs.values():
                group["qa_templates"] = _build_fallback_templates(group)

        create_dir(args.output_root)

        all_videos = os.listdir(args.video_root)

        args_list = [
            (args.video_root, video, args.ext, args.output_root, confusion_pairs, args)
            for video in all_videos
        ]

        try:
            with ProcessPoolExecutor(max_workers=max(1, cpu_count() - 1)) as executor:
                futures = [executor.submit(process_video, a) for a in args_list]
                annotations = [f.result() for f in futures]
        except Exception as e:
            logging.warning(f"Multiprocessing failed ({e}), falling back to single process")
            annotations = [process_video(a) for a in args_list]

        final_annotations = unpack_annotation(annotations)
        dump_json(os.path.join(args.output_root, f"{args.output_name}.json"), final_annotations)

        create_dir(os.path.join(args.output_root, "videos"))
        all_prc_videos = glob.glob(os.path.join(args.output_root, "*", f"*.{args.ext}"))
        for cur_video in all_prc_videos:
            shutil.copyfile(
                cur_video,
                os.path.join(args.output_root, "videos", os.path.basename(cur_video)),
            )

    logging.info("Spatial localization augmentation complete.")
