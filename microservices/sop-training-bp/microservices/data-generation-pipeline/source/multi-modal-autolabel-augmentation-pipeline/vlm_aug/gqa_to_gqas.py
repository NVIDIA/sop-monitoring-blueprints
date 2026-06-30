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


import argparse
import copy
import glob
import os
import random
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import cv2
from openai import OpenAI

from .cfg.gqa import (
    ANSWERS_TEMPLATE,
    QUESTION_TEMPLATE,
    QUESTION_TEMPLATE_CONCURRENT,
    ANSWERS_TEMPLATE_CONCURRENT,
)
from .cfg.llm import llm_cfg
from .utils import const
from .utils.annotation_template import dynamic_meta, llava_video
from .utils.helper import (
    clean_sentence,
    create_dir,
    dump_json,
    format_concurrent_actions,
    parse_video_action_indices,
    read_json,
    read_txt,
    resolve_llm_base_url,
    str2bool,
    unpack_annotation,
    write_txt,
)
from .utils.logger import logging


QA_SEP = "===\n"
Q_START = "Question:\n"
A_START = "Answer:\n"
INSTRUCT = "\n\nPlease generate {num_qa} question-answer pairs. Only output question-answer pairs."
LLM_OUT_ROOT = "GQA2GQAs"


def normalize_llm_output(s):
    """Normalize LLM output to the expected multi-line QA format.

    Handles common LLM deviations:
    1. Inline format:      Question: <text> === Answer: <text> === ...
    2. No separators:      Question: <text>\\nAnswer: <text>\\n\\nQuestion: ...
    3. Separators but no   Question: <text>\\n===\\nAnswer: <text>\\n===\\n
       newline after Q/A:  (=== present but "Question: " uses space not newline)
    4. Leading preamble:   "Sure! Here are 8 pairs:\\n\\nQuestion:\\n..."
                           (smaller models like to chat)
    5. Trailing separator: "...Answer:\\n<text>\\n===\\n" — symmetric closure
                           the few-shot doesn't teach but smaller models add

    Target format:
        Question:\\n<text>\\n===\\nAnswer:\\n<text>\\n===\\n...Question:\\n<text>\\n===\\nAnswer:\\n<text>\\n
    (no trailing "===" after the last Answer)
    """
    # Step 1: Normalize any inline "===" onto its own line (only if === exists
    # but not yet in "===\n" form).  Uses split/join instead of regex to avoid
    # ReDoS scanner warnings.
    if '===' in s and QA_SEP not in s:
        parts = s.split('===')
        s = '\n===\n'.join(part.strip() for part in parts)

    # Step 2: Always normalize "Question: text" -> "Question:\ntext"
    # (anchored to line start to avoid matching inside answer content)
    s = re.sub(r'(?:^|(?<=\n))Question:[ \t]+', 'Question:\n', s)
    s = re.sub(r'(?:^|(?<=\n))Answer:[ \t]+', 'Answer:\n', s)

    # Step 3: Insert "===\n" before any Question:/Answer: block that isn't
    # already preceded by one (handles partial separators & no separators).
    s = re.sub(r'(?<!\n===)\n+(?=(?:Answer|Question):\n)', '\n===\n', s)

    # Step 3.5: Strip any preamble before the first "Question:\n" block.
    # Smaller models often add chatter like "Sure! Here are 8 pairs:" which
    # makes the first split-block fail validation.
    m = re.search(r'(?:^|\n)Question:\n', s)
    if m:
        # Keep starting from "Question:\n" itself (drop everything before).
        s = s[m.start():].lstrip('\n')

    # Step 3.6: Drop a trailing "===" separator (with optional whitespace
    # before/after) so split(QA_SEP) does not produce an empty trailing block.
    s = re.sub(r'\n===\s*$', '\n', s)

    # Step 4: Ensure trailing newline
    if not s.endswith('\n'):
        s += '\n'

    return s


def validate_format(s):
    # Filter empty/whitespace-only blocks: tolerant of leftover separators
    # that normalize_llm_output may not have caught (e.g. "===\n\n===\n").
    blocks = [b for b in s.split(QA_SEP) if b.strip()]

    for block in blocks:
        if block.startswith(Q_START) and block.endswith(const.LINE_BREAK):
            continue
        elif block.startswith(A_START):
            continue
        else:
            raise ValueError(f"LLM output invalid QA format: {s}")


def prepare_sample_qas(action_json):
    all_actions = read_json(action_json)[const.ACTION_JSON_KEY]
    sample_qa_root = os.path.join(str(Path(action_json).parent), const.GQA2GQAS)
    create_dir(sample_qa_root)

    for i, cur_action in enumerate(all_actions, 1):
        cleaned_action = clean_sentence(cur_action)
        cleaned_action = cleaned_action[0].lower() + cleaned_action[1:]

        question = QUESTION_TEMPLATE.replace(const.SUBJECT_TOKEN, const.DEFAULT_SUBJECT)
        answer = ANSWERS_TEMPLATE.replace(const.SUBJECT_TOKEN, const.DEFAULT_SUBJECT).replace(
            const.STEP_TOKEN, cleaned_action
        )

        content = question + const.LINE_BREAK + answer

        logging.info(f"Create sample qa: {const.ACTION}{i}.txt\nQ: {question}\nA: {answer}\n")
        write_txt(os.path.join(sample_qa_root, const.ACTION + f"{i}.txt"), content)

    return sample_qa_root


def get_action_descriptions(action_indices, action_json_path):
    """Get action descriptions for given indices from action.json.

    Args:
        action_indices (List[int]): List of action indices (1-based)
        action_json_path (str): Path to action.json file

    Returns:
        List[str]: List of action description strings
    """
    all_actions = read_json(action_json_path)[const.ACTION_JSON_KEY]
    descriptions = []
    for idx in action_indices:
        if 1 <= idx <= len(all_actions):
            descriptions.append(all_actions[idx - 1])  # Convert to 0-based
        else:
            logging.warning(f"Action index {idx} out of range")
            descriptions.append(f"action {idx}")
    return descriptions


def prepare_concurrent_sample_qa(action_indices, action_descriptions, sample_qa_root):
    """Prepare sample QA file for concurrent actions.

    Args:
        action_indices (List[int]): List of action indices (e.g., [1, 3])
        action_descriptions (List[str]): List of action description strings
        sample_qa_root (str): Root directory for sample QA files

    Returns:
        str: Path to the created sample QA file
    """
    # Format concurrent steps
    formatted_steps = format_concurrent_actions(action_descriptions, action_indices)

    question = QUESTION_TEMPLATE_CONCURRENT
    answer = ANSWERS_TEMPLATE_CONCURRENT.replace(const.STEPS_TOKEN, formatted_steps)

    content = question + const.LINE_BREAK + answer

    # Create filename for concurrent actions (e.g., "action1-3.txt")
    concurrent_filename = f"{const.ACTION}{'-'.join(map(str, action_indices))}.txt"
    concurrent_qa_path = os.path.join(sample_qa_root, concurrent_filename)

    write_txt(concurrent_qa_path, content)
    logging.info(f"Create concurrent sample qa: {concurrent_filename}\nQ: {question}\nA: {answer}\n")

    return concurrent_qa_path


def prep_sys_prompts():
    system_prompt = read_txt(os.path.join(const.PROMPTS_ROOT, "gqa_to_gqas", "system_message.txt"))
    messages = [{"role": "system", "content": system_prompt}]

    # load examples
    all_caps = sorted(glob.glob(os.path.join(const.PROMPTS_ROOT, "gqa_to_gqas", "*_caps.txt")))
    all_convs = sorted(glob.glob(os.path.join(const.PROMPTS_ROOT, "gqa_to_gqas", "*_conv.txt")))

    for cap, conv in zip(all_caps, all_convs):
        cur_caps = read_txt(cap)
        cur_conv = read_txt(conv)
        num_qa = len(cur_conv.split(QA_SEP)) // 2

        messages.append({"role": "user", "content": cur_caps + INSTRUCT.format(num_qa=num_qa)})
        messages.append({"role": "assistant", "content": cur_conv})

    return messages


def llm_gen(sample_qa_file, num_qa_llm, output_root, cur_video_name, args, messages):
    qa_file_basename = os.path.basename(sample_qa_file)
    cur_messages = copy.deepcopy(messages)
    captions = read_txt(sample_qa_file)

    cur_messages.append({"role": "user", "content": captions + INSTRUCT.format(num_qa=num_qa_llm)})

    for message in cur_messages:
        logging.info(f"{message['role']}\n{message['content']}\n")

    base_url = resolve_llm_base_url(args.llm_type, args.local_llm_url, const.API_BASE_URL)
    client = OpenAI(
        base_url=base_url,
        api_key=args.api_key if args.llm_type == "nvidia" else "not-used",
    )

    logging.info("Start Inference")

    call_kwargs = dict(
        model=args.llm,
        messages=cur_messages,
        temperature=llm_cfg["temperature"],
        top_p=llm_cfg["top_p"],
        max_tokens=llm_cfg["max_tokens"],
        stream=False,
    )

    # When enable_thinking is explicitly configured, pass it to the API.
    # Models like Qwen3.5 default to "thinking" mode which returns content=None;
    # setting enable_thinking=false produces direct content instead.
    if hasattr(args, "enable_thinking") and args.enable_thinking != "":
        call_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": args.enable_thinking.lower() == "true"}
        }

    completion = client.chat.completions.create(**call_kwargs)
    llm_output = completion.choices[0].message.content

    # Fallback: if content is empty and enable_thinking was not configured,
    # retry once with thinking disabled (auto-detect thinking-mode models).
    if not llm_output and "extra_body" not in call_kwargs:
        logging.info("Empty content from LLM, retrying with enable_thinking=false")
        call_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        completion = client.chat.completions.create(**call_kwargs)
        llm_output = completion.choices[0].message.content

    if not llm_output:
        raise ValueError("LLM returned empty content. Check model compatibility.")
    llm_output = normalize_llm_output(llm_output)
    validate_format(llm_output)

    # dump llm output
    create_dir(os.path.join(output_root, cur_video_name))
    write_txt(os.path.join(output_root, cur_video_name, qa_file_basename), llm_output)

    # post process llm output
    all_qa = llm_output.split(QA_SEP)
    all_qa = [qa.replace(Q_START, "").replace(A_START, "").replace(const.LINE_BREAK, "") for qa in all_qa]

    return list(zip(all_qa[::2], all_qa[1::2]))


def assemble_anns(
    video_name, gpt, human, suffix, min_frames, max_frames, frame_cnts, frames_upperbound, dynamic_sample
):
    if suffix:
        human = f"{suffix}{human.strip()}"
        human = human.replace("\\n", "\n")

    # assemble output qa label
    qa_label = copy.deepcopy(llava_video)
    qa_label[const.CONV][0][const.VALUE] = human
    qa_label[const.CONV][1][const.VALUE] = gpt.strip()
    qa_label[const.VIDEO] = f"videos/{video_name}"

    if frames_upperbound > 0:
        max_frames = frames_upperbound

    if dynamic_sample:
        # append dynamic sample metadata
        qa_meta = copy.deepcopy(dynamic_meta)
        qa_meta[const.FRAME_COUNTS] = frame_cnts
        qa_meta[const.MIN_FRAMES] = int(min(frame_cnts[0], min_frames))
        qa_meta[const.MAX_FRAMES] = int(min(frame_cnts[0], max_frames))
        qa_meta[const.DYNAMIC_SAMPLE] = dynamic_sample
        qa_label[const.META] = qa_meta

    return qa_label


def _resolve_sample_qa_file(vid_basename, args):
    """Resolve the sample QA file path for a video, handling concurrent mode.

    Returns (sample_qa_file, action_indices) or (None, []) if the video should be skipped.
    """
    action_indices, is_concurrent = parse_video_action_indices(
        vid_basename, two_operator_mode=args.two_operator_mode
    )

    if not action_indices:
        logging.info(f"Skip video {vid_basename} - concurrent format not supported in single-operator mode")
        return None, []

    if any(idx in args.exclude_actions for idx in action_indices):
        logging.info(f"Skip excluded action(s) in GQA to GQAs: {action_indices}")
        return None, []

    if is_concurrent and args.two_operator_mode:
        action_descriptions = get_action_descriptions(action_indices, args.action_json)
        sample_qa_file = prepare_concurrent_sample_qa(
            action_indices, action_descriptions, args.sample_qa_root
        )
        logging.info(f"Processing concurrent video: {vid_basename} with actions {action_indices}")
    else:
        cur_action = action_indices[0]
        sample_qa_file = os.path.join(args.sample_qa_root, f"{const.ACTION}{cur_action}.txt")

    if not os.path.exists(sample_qa_file):
        logging.warning(f"Sample qa file: {sample_qa_file} not exist. Skip this sample qa file.")
        return None, []

    return sample_qa_file, action_indices


def _generate_chunk_anns(cur_video, all_qa, num_qa_per_chunk, human_suffix,
                         min_frames, max_frames, frames_upperbound, dynamic_sample, args):
    """Generate annotations for a single video chunk from sampled QAs."""
    do_replacement = args.replace if len(all_qa) >= num_qa_per_chunk else True
    frame_cnts = [int(cv2.VideoCapture(cur_video).get(cv2.CAP_PROP_FRAME_COUNT))]

    picked_qas = (random.choices(all_qa, k=num_qa_per_chunk) if do_replacement
                  else random.sample(all_qa, k=num_qa_per_chunk))

    anns = []
    for qst, ans in picked_qas:
        anns.append(
            assemble_anns(
                os.path.basename(cur_video), ans, qst, human_suffix,
                min_frames, max_frames, frame_cnts, frames_upperbound, dynamic_sample,
            )
        )
    return anns


def process_gqa(
    num_qa_llm,
    video_root,
    video,
    video_ext,
    output_root,
    min_frames,
    max_frames,
    frames_upperbound,
    dynamic_sample,
    num_qa_per_chunk,
    human_suffix,
    args,
    messages,
):
    random.seed(os.getpid())

    anns = []
    all_videos = sorted(glob.glob(os.path.join(video_root, f"{video}/*.{video_ext}")))

    video_out_root = os.path.join(output_root, video)
    create_dir(video_out_root)

    for cur_video in all_videos:
        sample_qa_file, action_indices = _resolve_sample_qa_file(Path(cur_video).stem, args)
        if sample_qa_file is None:
            continue

        all_qa = llm_gen(sample_qa_file, num_qa_llm, os.path.join(output_root, LLM_OUT_ROOT), video, args, messages)

        anns.extend(_generate_chunk_anns(
            cur_video, all_qa, num_qa_per_chunk, human_suffix,
            min_frames, max_frames, frames_upperbound, dynamic_sample, args,
        ))

    for cur_video in all_videos:
        shutil.copyfile(cur_video, os.path.join(video_out_root, os.path.basename(cur_video)))

    return anns


def process_video(args_tuple):
    (
        num_qa_llm,
        video_root,
        video,
        video_ext,
        output_root,
        min_frames,
        max_frames,
        frames_upperbound,
        dynamic_sample,
        num_qa_per_chunk,
        human_suffix,
        args,
        messages,
    ) = args_tuple

    return process_gqa(
        num_qa_llm,
        video_root,
        video,
        video_ext,
        output_root,
        min_frames,
        max_frames,
        frames_upperbound,
        dynamic_sample,
        num_qa_per_chunk,
        human_suffix,
        args,
        messages,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # LLM related args
    parser.add_argument(
        "--llm-type", type=str, default="nvidia", choices=["nvidia", "local"], help="llm type, local or nvidia"
    )
    parser.add_argument(
        "--local-llm-url",
        type=str,
        default="",
        help="local LLM URL, if using nim on local machine, then it could be http://0.0.0.0:8000/v1",
    )
    parser.add_argument("--llm", type=str, default="meta/llama-3.1-70b-instruct")
    parser.add_argument(
        "--api-key", type=str, default="", help="Nvidia API key, if using local LLM, then it must be empty"
    )
    parser.add_argument(
        "--enable-thinking", type=str, default="",
        help="Set to 'true' or 'false' for models with thinking mode (e.g. Qwen3.5). Leave empty to auto-detect.",
    )
    parser.add_argument("--sample-qa-root", type=str, default="", help="directory that store sample qa txt files")
    parser.add_argument("--action-json", type=str, default="", help="action json file from labelling tool")
    parser.add_argument("--num-qa-llm", type=int, default=5, help="number of QA pairs generate by LLM")

    # annotation related args
    parser.add_argument("--video-root", type=str, required=True, help="video root path")
    parser.add_argument("--ext", type=str, default="mp4", help="video extension format")
    parser.add_argument("--subject", type=str, default=None, help="subject who take the action")
    parser.add_argument("--human-suffix", type=str, default="<video>\n")
    parser.add_argument("--num-qa-per-chunk", type=int, help="Number of qa per chunk")
    parser.add_argument("--output-root", type=str, help="output root")
    parser.add_argument("--output-name", type=str, help="file name to be saved")
    parser.add_argument("--replace", type=str2bool, default=False, help="Replacement or not")
    parser.add_argument("--min_frames", type=int, default=2, help="minimum frame for dyanmic sample metadata")
    parser.add_argument("--max_frames", type=int, default=3, help="maximum frame for dynamic sample metadata")
    parser.add_argument(
        "--frames_upperbound",
        type=int,
        default=-1,
        help="maximum number of frames can be sampled. If provided, max_frames would be override by frames_upperbound // num_chunks",
    )
    parser.add_argument("--dynamic_sample", type=str2bool, default=False, help="wether to enable dynamic sample flag")
    parser.add_argument(
        "--exclude-action", type=str, default="", help="actions to exclude from sequential actions chunks"
    )
    parser.add_argument(
        "--two-operator-mode",
        type=str2bool,
        default=False,
        help="Enable two-operator mode for concurrent action handling"
    )
    args = parser.parse_args()

    # The API key is supplied via the NGC_API_KEY environment variable. Fall back to it whenever
    # --api-key is not passed explicitly on the command line.
    if not args.api_key:
        args.api_key = os.environ.get("NGC_API_KEY", "")

    # split exclude action into a list of int
    if args.exclude_action != "":
        args.exclude_actions = [int(action) for action in args.exclude_action.split(const.VIDEO_ACTION_SEP)]
    else:
        args.exclude_actions = []

    # Validate: action-json required when two-operator mode is ON
    if args.two_operator_mode and args.action_json == "":
        raise ValueError("Must provide 'action-json' when two-operator-mode is enabled.")

    # assert if api-key is provided, if not, then must provide local-llm-url
    assert not (args.local_llm_url == "" and args.api_key == ""), (
        "Must provide 'local-llm-url' or 'api-key' (from Nvidia API)."
    )
    if args.llm_type == "nvidia":
        assert args.api_key != "", "Must provide 'api-key' (from Nvidia API)."
        logging.info("Use Nvidia API for LLM.")
    elif args.llm_type == "local":
        assert args.local_llm_url != "", "Must provide 'local-llm-url'."
        logging.info("Use local LLM on {args.local_llm_url}.")
    else:
        raise ValueError(f"Invalid LLM type: {args.llm_type}")

    # check if any of sample-qa-root or action json is provided
    if args.sample_qa_root == "" and args.action_json == "":
        raise ValueError("Must provide either 'sample-qa-root' or 'action-json'.")
    elif args.sample_qa_root:
        logging.info("Sample qa root provided. Use sample qas for generation.")
    else:
        logging.info("Use action json file for generation.")

        # process action json into sample qa files format
        created_sample_qa_root = prepare_sample_qas(args.action_json)
        args.sample_qa_root = created_sample_qa_root

    # Log two-operator mode status
    if args.two_operator_mode:
        logging.info("Two-operator mode ENABLED - concurrent actions will be processed.")
    else:
        logging.info("Two-operator mode DISABLED - only single actions will be processed.")

    create_dir(args.output_root)
    create_dir(os.path.join(args.output_root, LLM_OUT_ROOT))

    # prepare LLM system prompt and user prompt
    messages = prep_sys_prompts()

    # load all available videos
    all_videos = os.listdir(args.video_root)

    # start augmenting
    # Prepare arguments for multiprocessing
    args_list = [
        (
            args.num_qa_llm,
            args.video_root,
            video,
            args.ext,
            args.output_root,
            args.min_frames,
            args.max_frames,
            args.frames_upperbound,
            args.dynamic_sample,
            args.num_qa_per_chunk,
            args.human_suffix,
            args,
            messages,
        )
        for video in all_videos
    ]

    # Try multiprocessing first, fallback to single process if error occurs
    try:
        # -1 to leave one core for the main process
        with ProcessPoolExecutor(max_workers=max(1, cpu_count() - 1)) as executor:
            futures = [executor.submit(process_video, args) for args in args_list]
            annotations = [future.result() for future in futures]
    except Exception as e:
        logging.warning(f"Multiprocessing failed. Error: {e}. Fallback to single process.")
        annotations = [process_video(args) for args in args_list]

    # unpack annotations
    final_annotations = unpack_annotation(annotations)

    # save annotations
    dump_json(os.path.join(args.output_root, f"{args.output_name}.json"), final_annotations)

    # copy all videos to videos folder
    create_dir(os.path.join(args.output_root, "videos"))
    all_prc_videos = glob.glob(os.path.join(args.output_root, "*", f"*.{args.ext}"))

    for i, cur_video in enumerate(all_prc_videos):
        shutil.copyfile(cur_video, os.path.join(args.output_root, "videos", os.path.basename(cur_video)))
