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
import ast
import copy
import glob
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

from .cfg.dmcq import QUESTION_TEMPLATE, CONCURRENT_QUESTION_TEMPLATE
from .utils import const
from .utils.annotation_template import dmcq_meta, llava_video
from .utils.helper import (
    clean_sentence,
    create_dir,
    dump_json,
    read_json,
    read_txt,
    unpack_annotation,
    write_txt,
)
from .utils.logger import logging


def prepare_sample_choices(action_json):
    all_actions = read_json(action_json)[const.ACTION_JSON_KEY]
    config_root = os.path.join(str(Path(action_json).parent), const.DMCQ)
    create_dir(config_root)

    choices = ""

    for i, cur_action in enumerate(all_actions, 1):
        cleaned_action = clean_sentence(cur_action)

        choices += cleaned_action

        if i != len(all_actions):
            choices += const.LINE_BREAK

    choices = choices.strip()

    logging.info(f"Create DMCQ config choices: {const.CHOICES}.txt\n{choices}")
    write_txt(os.path.join(config_root, const.CHOICES + ".txt"), choices)

    return config_root


def _get_action_indices(video_path):
    action_part = os.path.basename(video_path).split(const.VIDEO_ACTION_SEP)[0]
    return [int(a) for a in action_part.split(const.CONCURRENT_ACTION_SEP)]


def assemble_anns(
    video_name, 
    action_index, 
    is_positive, 
    min_options, 
    max_options,
    action_list,
    non_sop_action_index,
    confusion_map=None, 
    is_hard=False, 
    hard_mode="adjacent"):

    # random pick a number between min_options and max_options
    num_options = random.randint(min_options, max_options)

    cur_action = action_list[action_index]

    # construct adjacent first for hard samples (positive or negative)
    if is_hard and hard_mode == "adjacent":
        # hard samples are the options that are adjacent to the correct action
        if action_index == 0:
            hard_options = [action_list[action_index + 1]]
        elif action_index == len(action_list) - 1:
            hard_options = [action_list[action_index - 1]]
        else:
            hard_options = [action_list[action_index - 1], action_list[action_index + 1]]
    elif is_hard and hard_mode == "confusion":
        if action_index + 1 in confusion_map:
            hard_options = [action_list[index - 1] for index in confusion_map[action_index + 1]]
        else:
            hard_options = []
    
    if is_positive:
        correct_option = cur_action
        if is_hard:
            other_options = [option for option in action_list if option != correct_option and option not in hard_options]
            other_options = random.sample(other_options, num_options - 1 - len(hard_options))
            cur_options = hard_options + other_options + [correct_option]
            pos_or_neg = "hp" if len(hard_options) > 0 else "pos"
        else:
            other_options = [option for option in action_list if option != correct_option]
            other_options = random.sample(other_options, num_options - 1)
            cur_options = [correct_option] + other_options
            pos_or_neg = "pos"
    else:
        correct_option = action_list[non_sop_action_index]
        if is_hard:
            other_options = [option for option in action_list if option != cur_action and option != correct_option and option not in hard_options]
            other_options = random.sample(other_options, num_options - 1 - len(hard_options))
            cur_options = hard_options + other_options + [correct_option]
            pos_or_neg = "hn" if len(hard_options) > 0 else "neg"
        else:
            other_options = [option for option in action_list if option != correct_option and option != cur_action]
            other_options = random.sample(other_options, num_options - 1)
            cur_options = other_options + [correct_option]
            pos_or_neg = "neg"

    random.shuffle(cur_options)
    if action_list[non_sop_action_index] in cur_options:
        cur_options.remove(action_list[non_sop_action_index])
        cur_options.append(action_list[non_sop_action_index])

    # add option number to each option and get the correct option index
    correct_option_index = cur_options.index(correct_option)
    cur_options = [f"({i+1}) {option}" for i, option in enumerate(cur_options)]
    cur_options_str = "\n".join(cur_options)

    # question
    question = random.choice(QUESTION_TEMPLATE).replace(const.STEP_TOKEN, str(num_options)).replace(
        const.SUBJECT_TOKEN, const.DEFAULT_SUBJECT
    )
    question = question + "\n" + cur_options_str

    # answer
    answer = f"({correct_option_index+1}) {correct_option}"

    # assemble output qa label
    qa_label = copy.deepcopy(llava_video)
    qa_label[const.CONV][0][const.VALUE] = question
    qa_label[const.CONV][1][const.VALUE] = answer
    qa_label[const.VIDEO] = f"videos/{video_name}"

    qa_meta = copy.deepcopy(dmcq_meta)
    qa_meta[const.GT_ACTION] = cur_action
    qa_meta[const.POS_OR_NEG] = pos_or_neg
    qa_meta[const.HARD_MODE] = hard_mode if pos_or_neg == "hp" or pos_or_neg == "hn" else ""
    qa_meta[const.NUM_OPTIONS] = num_options
    qa_label[const.META] = qa_meta

    return qa_label


def assemble_concurrent_anns(
    video_name,
    chunk_action_indices,
    is_positive,
    min_options,
    max_options,
    action_list,
    non_sop_action_index,
    confusion_map=None,
    is_hard=False,
    hard_mode="adjacent"):

    # random pick a number between min_options and max_options
    num_options = random.randint(min_options, max_options)

    chunk_actions = [action_list[index] for index in chunk_action_indices]
    combined_action = " ".join(chunk_actions)
    non_sop_action = action_list[non_sop_action_index]

    # each single action of a concurrent chunk is an incomplete (hence hard)
    # distractor; add the union of adjacent / confusion targets of every action
    hard_options = []
    if is_hard:
        hard_options = list(chunk_actions)
        for action_index in chunk_action_indices:
            if hard_mode == "adjacent":
                if action_index == 0:
                    targets = [action_list[action_index + 1]]
                elif action_index == len(action_list) - 1:
                    targets = [action_list[action_index - 1]]
                else:
                    targets = [action_list[action_index - 1], action_list[action_index + 1]]
            elif hard_mode == "confusion":
                targets = [action_list[index - 1] for index in confusion_map.get(action_index + 1, [])]
            else:
                targets = []
            for target in targets:
                if target not in hard_options:
                    hard_options.append(target)
        hard_options = [option for option in hard_options if option != non_sop_action][:num_options - 1]

    if is_positive:
        correct_option = combined_action
        pos_or_neg = "hp" if len(hard_options) > 0 else "pos"
    else:
        correct_option = non_sop_action
        pos_or_neg = "hn" if len(hard_options) > 0 else "neg"

    # random distractors never include the chunk's own actions: as complete
    # single actions they belong only in hard_options, mirroring how the
    # single-operator path keeps the video's true action out of the pool
    other_options = [
        option for option in action_list
        if option != correct_option and option not in hard_options and option not in chunk_actions
    ]
    num_random = max(0, num_options - 1 - len(hard_options))
    other_options = random.sample(other_options, min(num_random, len(other_options)))
    cur_options = hard_options + other_options + [correct_option]

    random.shuffle(cur_options)
    if non_sop_action in cur_options:
        cur_options.remove(non_sop_action)
        cur_options.append(non_sop_action)

    correct_option_index = cur_options.index(correct_option)
    cur_options = [f"({i+1}) {option}" for i, option in enumerate(cur_options)]
    cur_options_str = "\n".join(cur_options)

    question = random.choice(CONCURRENT_QUESTION_TEMPLATE).replace(const.STEP_TOKEN, str(len(cur_options))).replace(
        const.SUBJECT_TOKEN, const.DEFAULT_SUBJECT_CONCURRENT
    )
    question = question + "\n" + cur_options_str

    answer = f"({correct_option_index+1}) {correct_option}"

    qa_label = copy.deepcopy(llava_video)
    qa_label[const.CONV][0][const.VALUE] = question
    qa_label[const.CONV][1][const.VALUE] = answer
    qa_label[const.VIDEO] = f"videos/{video_name}"

    qa_meta = copy.deepcopy(dmcq_meta)
    qa_meta[const.GT_ACTION] = combined_action
    qa_meta[const.POS_OR_NEG] = pos_or_neg
    qa_meta[const.HARD_MODE] = hard_mode if pos_or_neg == "hp" or pos_or_neg == "hn" else ""
    qa_meta[const.NUM_OPTIONS] = len(cur_options)
    qa_label[const.META] = qa_meta

    return qa_label


def _assemble_sample(
    video_name,
    chunk_action_indices,
    is_positive,
    min_options,
    max_options,
    action_list,
    non_sop_action_index,
    confusion_map,
    is_hard,
    hard_mode,
):
    if len(chunk_action_indices) == 1:
        return assemble_anns(
            video_name=video_name,
            action_index=chunk_action_indices[0],
            is_positive=is_positive,
            min_options=min_options,
            max_options=max_options,
            action_list=action_list,
            non_sop_action_index=non_sop_action_index,
            confusion_map=confusion_map,
            is_hard=is_hard,
            hard_mode=hard_mode,
        )
    return assemble_concurrent_anns(
        video_name=video_name,
        chunk_action_indices=chunk_action_indices,
        is_positive=is_positive,
        min_options=min_options,
        max_options=max_options,
        action_list=action_list,
        non_sop_action_index=non_sop_action_index,
        confusion_map=confusion_map,
        is_hard=is_hard,
        hard_mode=hard_mode,
    )


def process_chunk(
    video_root,
    video,
    video_ext,
    min_options,
    max_options,
    non_sop_action,
    num_pos,
    num_neg,
    num_hard_pos,
    num_hard_neg,
    hard_pos_mode,
    hard_neg_mode,
    confusion_map,
    seed,
    output_root,
    args,
):
    if seed is not None:
        random.seed(seed)
    else:
        random.seed(os.getpid())

    anns = []

    # sort by the name.split(".")[-2].split("_")[-1], which is the timeline index
    # video file naming convention: <action_number>_<video_name>_<duplication_cnt>_<timeline_index>.<video_ext>
    all_videos = sorted(
        glob.glob(os.path.join(video_root, f"{video}/*.{video_ext}")),
        key=lambda x: int(os.path.basename(x).split(".")[-2].split("_")[-1]),
    )
    filtered_all_videos = [
        vid
        for vid in all_videos
        if not any(idx in args.exclude_actions for idx in _get_action_indices(vid))
    ]

    video_out_root = os.path.join(output_root, video)
    create_dir(video_out_root)

    non_sop_action_index = non_sop_action - 1

    for cur_video in filtered_all_videos:
        video_basename = os.path.basename(cur_video)
        chunk_action_indices = [action - 1 for action in _get_action_indices(cur_video)]
        logging.info(f"Processing video: {cur_video}, action indices: {chunk_action_indices}")

        # construct postive samples (with correct action option)
        for _ in range(num_pos):
            anns.append(_assemble_sample(
                video_basename, chunk_action_indices, True,
                min_options, max_options, args.choices, non_sop_action_index,
                confusion_map, False, "",
            ))

        # construct hard positive samples (with incorrect action option but similar action)
        for _ in range(num_hard_pos):
            for mode in hard_pos_mode:
                anns.append(_assemble_sample(
                    video_basename, chunk_action_indices, True,
                    min_options, max_options, args.choices, non_sop_action_index,
                    confusion_map, True, mode,
                ))

        # construct negative samples (with incorrect action option)
        for _ in range(num_neg):
            anns.append(_assemble_sample(
                video_basename, chunk_action_indices, False,
                min_options, max_options, args.choices, non_sop_action_index,
                confusion_map, False, "",
            ))

        # construct hard negative samples (with incorrect action option but similar action)
        for _ in range(num_hard_neg):
            for mode in hard_neg_mode:
                anns.append(_assemble_sample(
                    video_basename, chunk_action_indices, False,
                    min_options, max_options, args.choices, non_sop_action_index,
                    confusion_map, True, mode,
                ))

        # copy video to output video root
        shutil.copyfile(cur_video, os.path.join(video_out_root, video_basename))

    return anns



def process_video(args_tuple):
    (
        video_root,
        video,
        video_ext,
        min_options,
        max_options,
        non_sop_action,
        num_pos,
        num_neg,
        num_hard_pos,
        num_hard_neg,
        hard_pos_mode,
        hard_neg_mode,
        confusion_map,
        seed,
        output_root,
        args,
    ) = args_tuple
    return process_chunk(
        video_root,
        video,
        video_ext,
        min_options,
        max_options,
        non_sop_action,
        num_pos,
        num_neg,
        num_hard_pos,
        num_hard_neg,
        hard_pos_mode,
        hard_neg_mode,
        confusion_map,
        seed,
        output_root,
        args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-root", type=str, default="", help="root path for storing question.txt and choices.txt"
    )
    parser.add_argument("--action-json", type=str, default="", help="action json file from labelling tool")
    parser.add_argument("--video-root", type=str, required=True, help="video root path")
    parser.add_argument(
        "--exclude-action", type=str, default="", help="actions to exclude from sequential actions chunks"
    )
    parser.add_argument("--ext", type=str, default="mp4", help="video extension format")
    parser.add_argument("--min-options", type=int, default=4, help="minimum number of options")
    parser.add_argument("--max-options", type=int, default=6, help="maximum number of options")
    parser.add_argument("--non-sop-action", type=int, required=True, help="action index of non-SOP action.")
    parser.add_argument("--num-pos", type=int, default=2, help="number of positive samples")
    parser.add_argument("--num-neg", type=int, default=2, help="number of negative samples")
    parser.add_argument("--num-hard-pos", type=int, default=0, help="number of hard positive samples")
    parser.add_argument("--num-hard-neg", type=int, default=0, help="number of hard negative samples")
    parser.add_argument("--hard-neg-mode", type=str, default="adjacent", help="hard negative mode. support 'adjacent' and 'confusion'. separated by comma.")
    parser.add_argument("--hard-pos-mode", type=str, default="adjacent", help="hard positive mode. support 'adjacent' and 'confusion'. separated by comma.")
    parser.add_argument("--confusion-map", type=str, default="", help="confusion map. format: {action_index: [action_index_1, action_index_2, ...]}")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--output-root", type=str, required=True, help="output root for storing json file")
    parser.add_argument("--output-name", type=str, help="file name to be saved")
    args = parser.parse_args()

    # check if config or action json is provided
    if args.config_root == "" and args.action_json == "":
        raise ValueError("Must provide either 'config-root' or 'action-json'.")
    elif args.config_root:
        logging.info("Config root provided. Use config for generation.")
    else:
        logging.info("Use action json file for generation.")

        # process action json into sample qa files format
        created_config_root = prepare_sample_choices(args.action_json)
        args.config_root = created_config_root

    # split exclude action into a list of int
    if args.exclude_action != "":
        args.exclude_actions = [int(action) for action in args.exclude_action.split(const.VIDEO_ACTION_SEP)]
    else:
        args.exclude_actions = []

    # validate args
    args.hard_neg_mode = args.hard_neg_mode.split(",") if args.hard_neg_mode != "" else []
    for mode in args.hard_neg_mode:
        if mode not in const.HARD_MODES:
            raise ValueError(f"Invalid hard negative mode: {mode}. Must be one of {const.HARD_MODES}.")
    args.hard_pos_mode = args.hard_pos_mode.split(",") if args.hard_pos_mode != "" else []
    for mode in args.hard_pos_mode:
        if mode not in const.HARD_MODES:
            raise ValueError(f"Invalid hard positive mode: {mode}. Must be one of {const.HARD_MODES}.")

    if const.CONFUSION in args.hard_neg_mode or const.CONFUSION in args.hard_pos_mode:
        if args.confusion_map == "":
            raise ValueError("Must provide 'confusion-map' when 'confusion' is in 'hard-neg-mode'.")
        args.confusion_map = ast.literal_eval(args.confusion_map)

    create_dir(args.output_root)

    # load question and choices
    choices_txt = os.path.join(args.config_root, f"{const.CHOICES}.txt")
    args.choices = read_txt(choices_txt).split(const.LINE_BREAK) # list of choices

    # load all available videos
    all_videos = os.listdir(args.video_root)

    # start augmenting
    # Prepare arguments for multiprocessing
    args_list = [
        (
            args.video_root,
            video,
            args.ext,
            args.min_options,
            args.max_options,
            args.non_sop_action,
            args.num_pos,
            args.num_neg,
            args.num_hard_pos,
            args.num_hard_neg,
            args.hard_pos_mode,
            args.hard_neg_mode,
            args.confusion_map,
            args.seed,
            args.output_root,
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