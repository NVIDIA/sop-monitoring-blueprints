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

from .cfg.bcq import QUESTION_TEMPLATE
from .utils import const
from .utils.annotation_template import dynamic_meta, llava_video
from .utils.helper import (
    clean_sentence,
    create_dir,
    dump_json,
    read_json,
    read_txt,
    str2bool,
    unpack_annotation,
    write_txt,
)
from .utils.logger import logging


def prepare_sample_qas(action_json):
    all_actions = read_json(action_json)[const.ACTION_JSON_KEY]
    config_root = os.path.join(str(Path(action_json).parent), const.BCQ)
    create_dir(config_root)

    choices = ""

    for i, cur_action in enumerate(all_actions, 1):
        cleaned_action = clean_sentence(cur_action)
        cleaned_action = cleaned_action[0].lower() + cleaned_action[1:]

        choices += cleaned_action

        if i != len(all_actions):
            choices += const.LINE_BREAK

    # Unified question template (action-focused, operator-agnostic)
    logging.info(f"Create BCQ config question: {const.QUESTION}.txt\n{QUESTION_TEMPLATE}")
    write_txt(os.path.join(config_root, const.QUESTION + ".txt"), QUESTION_TEMPLATE)

    choices = choices.strip()
    logging.info(f"Create BCQ config choices: {const.CHOICES}.txt\n{choices}")
    write_txt(os.path.join(config_root, const.CHOICES + ".txt"), choices)

    return config_root


def assemble_anns(video_name, gpt, human, min_frames, max_frames, frame_cnts, frames_upperbound, dynamic_sample):
    # assemble output qa label
    qa_label = copy.deepcopy(llava_video)
    qa_label[const.CONV][0][const.VALUE] = human
    qa_label[const.CONV][1][const.VALUE] = gpt
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


def process_trunks(
    video_root, video, video_ext, output_root, min_frames, max_frames, frames_upperbound, dynamic_sample, args
):
    random.seed(os.getpid())

    anns = []
    all_videos = sorted(glob.glob(os.path.join(video_root, f"{video}/*.{video_ext}")))

    video_out_root = os.path.join(output_root, video)

    create_dir(video_out_root)
    logging.info(args.choices)

    for cur_video in all_videos:
        video_basename = os.path.basename(cur_video)
        action_part = video_basename.split(".")[0].split(const.VIDEO_ACTION_SEP)[0]

        # Parse action indices from video filename
        # Single action: "01_video.mp4" -> [0]
        # Concurrent: "01-03_video.mp4" -> [0, 2]
        if "-" in action_part:
            action_indices = [int(a) - 1 for a in action_part.split("-")]
            logging.info(f"Concurrent action video: {video_basename}, actions: {[i+1 for i in action_indices]}")
        else:
            action_indices = [int(action_part) - 1]

        # skip if any action is excluded
        if any((i + 1) in args.exclude_actions for i in action_indices):
            logging.info(f"Skip excluded action in Config to BCQ: {[i+1 for i in action_indices]}")
            continue

        frame_cnts = [int(cv2.VideoCapture(cur_video).get(cv2.CAP_PROP_FRAME_COUNT))]

        # UNIFIED FORMAT: Same logic for single-action and concurrent videos
        # Positive choices: actions present in the video
        # Negative choices: actions NOT present in the video
        pos_choices = [args.choices[i] for i in action_indices]
        neg_choices = [c for j, c in enumerate(args.choices) if j not in action_indices]

        # Subject phrasing aligned with the original BCQ style; pluralized when a video
        # contains multiple concurrent actions so the answer mentions all of them.
        if len(pos_choices) == 1:
            subject_phrase = f"the {args.subject} is"
            actions_phrase = pos_choices[0]
        else:
            subject_phrase = f"the {args.subject}s are"
            actions_phrase = ", ".join(pos_choices)

        # Generate one positive sample per action in this video
        for pos_choice in pos_choices:
            human = args.human_prompts.replace(const.STEP_TOKEN, pos_choice) + "?"
            gpt = f"Yes, {subject_phrase} {actions_phrase}."
            anns.append(
                assemble_anns(
                    video_basename, gpt, human, min_frames, max_frames,
                    frame_cnts, frames_upperbound, dynamic_sample,
                )
            )

        # Generate negative samples (scaled by neg_cnt per positive action).
        # The negative answer reveals the actual action(s), matching the original
        # `f"No, the {args.subject} is {pos_choice}."` semantics on main.
        num_neg_samples = args.neg_cnt * len(pos_choices)
        for _ in range(num_neg_samples):
            if neg_choices:
                neg_choice = random.choice(neg_choices)  # nosec S2245 — ML sampling, not security
                human = args.human_prompts.replace(const.STEP_TOKEN, neg_choice) + "?"
                gpt = f"No, {subject_phrase} {actions_phrase}."
                anns.append(
                    assemble_anns(
                        video_basename, gpt, human, min_frames, max_frames,
                        frame_cnts, frames_upperbound, dynamic_sample,
                    )
                )

    # Copy all videos to output directory
    for video in all_videos:
        shutil.copyfile(video, os.path.join(video_out_root, os.path.basename(video)))

    return anns


def process_video(args_tuple):
    video_root, video, video_ext, output_root, min_frames, max_frames, frames_upperbound, dynamic_sample, args = (
        args_tuple
    )
    return process_trunks(
        video_root, video, video_ext, output_root, min_frames, max_frames, frames_upperbound, dynamic_sample, args
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-root", type=str, default="", help="root path for storing question.txt and choices.txt"
    )
    parser.add_argument("--action-json", type=str, default="", help="action json file from labelling tool")
    parser.add_argument("--subject", type=str, default="operator", help="subject who take the action")
    parser.add_argument("--video-root", type=str, required=True, help="video root path")
    parser.add_argument("--ext", type=str, default="mp4", help="video extension format")
    parser.add_argument("--negative-ratio", type=float, default=2.0, help="negative sample ratio")
    parser.add_argument("--output-root", type=str, required=True, help="output root for storing json file")
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
    args = parser.parse_args()

    # split exclude action into a list of int
    if args.exclude_action != "":
        args.exclude_actions = [int(action) for action in args.exclude_action.split(const.VIDEO_ACTION_SEP)]
    else:
        args.exclude_actions = []

    # check if any of sample-qa-root or action json is provided
    if args.config_root == "" and args.action_json == "":
        raise ValueError("Must provide either 'config-root' or 'action-json'.")
    elif args.config_root:
        logging.info("Config root provided. Use config for generation.")
    else:
        logging.info("Use action json file for generation.")

        # process action json into sample qa files format
        created_config_root = prepare_sample_qas(args.action_json)
        args.config_root = created_config_root

    create_dir(args.output_root)

    # load question and choices (unified format for both single and two-operator modes)
    question_txt = os.path.join(args.config_root, f"{const.QUESTION}.txt")
    choices_txt = os.path.join(args.config_root, f"{const.CHOICES}.txt")

    args.human_prompts = read_txt(question_txt).replace("\\n", "\n")  # not sure why \n would become \\n
    args.choices = read_txt(choices_txt).split(const.LINE_BREAK)

    # load all available videos
    all_videos = os.listdir(args.video_root)

    args.pos_cnt = max(1, int(1 / args.negative_ratio))
    args.neg_cnt = int(args.pos_cnt * args.negative_ratio)

    logging.info(f"pos_cnt: {args.pos_cnt}\nneg_cnt: {args.neg_cnt}")

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
