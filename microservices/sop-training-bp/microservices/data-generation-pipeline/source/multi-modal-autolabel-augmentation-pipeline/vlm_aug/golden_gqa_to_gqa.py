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
import shutil
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import cv2

from .cfg.gqa import (
    ANSWERS_TEMPLATE,
    QUESTION_TEMPLATE,
    QUESTION_TEMPLATE_CONCURRENT,
    ANSWERS_TEMPLATE_CONCURRENT,
)
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
    str2bool,
    unpack_annotation,
    write_txt,
)
from .utils.logger import logging


GOLDEN_QA_SEP = "\n"
Q_START = "Question:"
A_START = "Answer:"


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


def generate_concurrent_qa(action_indices, action_descriptions):
    """Generate QA pair for concurrent actions.

    Args:
        action_indices (List[int]): List of action indices
        action_descriptions (List[str]): List of action descriptions

    Returns:
        Tuple[str, str]: (question, answer)
    """
    # Format: "(1) picking up item (3) inspecting label"
    formatted_steps = format_concurrent_actions(action_descriptions, action_indices)

    question = QUESTION_TEMPLATE_CONCURRENT
    answer = ANSWERS_TEMPLATE_CONCURRENT.replace(const.STEPS_TOKEN, formatted_steps)

    return question, answer


def prepare_golden_qas(action_json):
    all_actions = read_json(action_json)[const.ACTION_JSON_KEY]
    golden_qa_root = os.path.join(str(Path(action_json).parent), const.GOLDEN_GQA2GQAS)
    create_dir(golden_qa_root)

    for i, cur_action in enumerate(all_actions, 1):
        cleaned_action = clean_sentence(cur_action)
        cleaned_action = cleaned_action[0].lower() + cleaned_action[1:]

        question = QUESTION_TEMPLATE.replace(const.SUBJECT_TOKEN, const.DEFAULT_SUBJECT)
        answer = ANSWERS_TEMPLATE.replace(const.SUBJECT_TOKEN, const.DEFAULT_SUBJECT).replace(
            const.STEP_TOKEN, cleaned_action
        )

        content = question + const.LINE_BREAK + answer

        logging.info(f"Create sample qa: {const.ACTION}{i}.txt\nQ: {question}\nA: {answer}\n")
        write_txt(os.path.join(golden_qa_root, const.ACTION + f"{i}.txt"), content)

    return golden_qa_root


def process_qa(sample_qa_file):
    golden_qa = read_txt(sample_qa_file)

    all_qa = golden_qa.split(GOLDEN_QA_SEP)
    all_qa = [qa.replace(Q_START, "").replace(A_START, "").replace("\n", "") for qa in all_qa]

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


def process_gqa(
    video_root,
    video,
    video_ext,
    output_root,
    min_frames,
    max_frames,
    frames_upperbound,
    dynamic_sample,
    human_suffix,
    args,
):
    random.seed(os.getpid())

    anns = []
    all_videos = sorted(glob.glob(os.path.join(video_root, f"{video}/*.{video_ext}")))

    video_out_root = os.path.join(output_root, video)
    create_dir(video_out_root)

    for cur_video in all_videos:
        vid_basename = Path(cur_video).stem

        # Parse action indices - respects two_operator_mode toggle
        action_indices, is_concurrent = parse_video_action_indices(
            vid_basename,
            two_operator_mode=args.two_operator_mode
        )

        # Skip if no valid actions (e.g., concurrent video when toggle is OFF)
        if not action_indices:
            logging.info(f"Skip video {vid_basename} - concurrent format not supported in single-operator mode")
            continue

        # Skip if any action is in exclude list
        if any(idx in args.exclude_actions for idx in action_indices):
            logging.info(f"Skip excluded action(s) in Golden GQA to GQA: {action_indices}")
            continue

        frame_cnts = [int(cv2.VideoCapture(cur_video).get(cv2.CAP_PROP_FRAME_COUNT))]

        if is_concurrent and args.two_operator_mode:
            # === TWO-OPERATOR MODE: CONCURRENT ACTIONS ===
            action_descriptions = get_action_descriptions(action_indices, args.action_json)
            qst, ans = generate_concurrent_qa(action_indices, action_descriptions)
            logging.info(f"Concurrent QA for {vid_basename}: Q={qst}, A={ans}")
        else:
            # === ORIGINAL SINGLE-OPERATOR MODE ===
            cur_action = action_indices[0]

            # get corresponding sample qa file
            sample_qa_file = os.path.join(args.golden_qa_root, f"{const.ACTION}{cur_action}.txt")
            if not os.path.exists(sample_qa_file):
                logging.warning(f"Sample qa file: {sample_qa_file} not exist. Skip this sample qa file.")
                continue

            all_qa = process_qa(sample_qa_file)[0]
            logging.info(all_qa)

            # only support one golden qa set for now
            qst = all_qa[0]
            ans = all_qa[1]

        anns.append(
            assemble_anns(
                os.path.basename(cur_video),
                ans,
                qst,
                human_suffix,
                min_frames,
                max_frames,
                frame_cnts,
                frames_upperbound,
                dynamic_sample,
            )
        )

    for cur_video in all_videos:
        shutil.copyfile(cur_video, os.path.join(video_out_root, os.path.basename(cur_video)))

    return anns


def process_video(args_tuple):
    (
        video_root,
        video,
        video_ext,
        output_root,
        min_frames,
        max_frames,
        frames_upperbound,
        dynamic_sample,
        human_suffix,
        args,
    ) = args_tuple

    return process_gqa(
        video_root,
        video,
        video_ext,
        output_root,
        min_frames,
        max_frames,
        frames_upperbound,
        dynamic_sample,
        human_suffix,
        args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=str, required=True, help="video root path")
    parser.add_argument("--ext", type=str, default="mp4", help="video extension format")
    parser.add_argument("--golden-qa-root", type=str, default="", help="directory that store golden qa txt files")
    parser.add_argument("--action-json", type=str, default="", help="action json file from labelling tool")
    parser.add_argument("--human-suffix", type=str, default="<video>\n")
    parser.add_argument("--output-root", type=str, help="output root")
    parser.add_argument("--output-name", type=str, help="file name to be saved")
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

    # split exclude action into a list of int
    if args.exclude_action != "":
        args.exclude_actions = [int(action) for action in args.exclude_action.split(const.VIDEO_ACTION_SEP)]
    else:
        args.exclude_actions = []

    # Validate: action-json required when two-operator mode is ON
    if args.two_operator_mode and args.action_json == "":
        raise ValueError("Must provide 'action-json' when two-operator-mode is enabled.")

    # check if any of sample-qa-root or action json is provided
    if args.golden_qa_root == "" and args.action_json == "":
        raise ValueError("Must provide either 'golden-qa-root' or 'action-json'.")
    elif args.golden_qa_root:
        logging.info("Golden qa root provided. Use golden qas for generation.")
    else:
        logging.info("Use action json file for generation.")

        # process action json into sample qa files format
        created_golden_qa_root = prepare_golden_qas(args.action_json)
        args.golden_qa_root = created_golden_qa_root

    # Log two-operator mode status
    if args.two_operator_mode:
        logging.info("Two-operator mode ENABLED - concurrent actions will be processed.")
    else:
        logging.info("Two-operator mode DISABLED - only single actions will be processed.")

    create_dir(args.output_root)

    # load all available videos
    all_videos = os.listdir(args.video_root)

    # start augmenting
    # Prepare arguments for multiprocessing
    args_list = [
        (
            args.video_root,
            video,
            args.ext,
            args.output_root,
            args.min_frames,
            args.max_frames,
            args.frames_upperbound,
            args.dynamic_sample,
            args.human_suffix,
            args,
        )
        for video in all_videos
    ]

    # Use multiprocessing Pool
    # -1 to leave one core for the main process
    with ProcessPoolExecutor(max_workers=cpu_count() - 1) as executor:
        futures = [executor.submit(process_video, args) for args in args_list]
        annotations = [future.result() for future in futures]

    # unpack annotations
    final_annotations = unpack_annotation(annotations)

    # save annotations
    dump_json(os.path.join(args.output_root, f"{args.output_name}.json"), final_annotations)

    # copy all videos to videos folder
    create_dir(os.path.join(args.output_root, "videos"))
    all_prc_videos = glob.glob(os.path.join(args.output_root, "*", f"*.{args.ext}"))

    for i, cur_video in enumerate(all_prc_videos):
        shutil.copyfile(cur_video, os.path.join(args.output_root, "videos", os.path.basename(cur_video)))
