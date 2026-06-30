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

from .cfg.en import QUESTION_TEMPLATE, GOLDEN_QUESTION
from .utils import const
from .utils.annotation_template import en_meta, llava_video
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
    config_root = os.path.join(str(Path(action_json).parent), const.EN)
    create_dir(config_root)

    choices = ""

    for i, cur_action in enumerate(all_actions, 1):
        cleaned_action = clean_sentence(cur_action)

        choices += cleaned_action

        if i != len(all_actions):
            choices += const.LINE_BREAK

    choices = choices.strip()

    logging.info(f"Create EN config choices: {const.CHOICES}.txt\n{choices}")
    write_txt(os.path.join(config_root, const.CHOICES + ".txt"), choices)

    return config_root


def assemble_anns(
    chunk, 
    num_options,
    options,
    answer,
    question_template=None):

    # add option number to each option and get the correct option index
    correct_option_index = options.index(answer)
    options = [f"({i+1}) {option}" for i, option in enumerate(options)]
    options_str = "\n".join(options)

    # construct question
    question = random.choice(QUESTION_TEMPLATE) if question_template is None else question_template
    question = question.replace(const.STEP_TOKEN, str(num_options)).replace(
        const.SUBJECT_TOKEN, const.DEFAULT_SUBJECT
    )
    question = question + "\n" + options_str

    # answer
    answer = f"({correct_option_index+1}) {answer}"

    # construct annotation
    annotation = copy.deepcopy(llava_video)
    annotation["video"] = f"videos/{os.path.basename(chunk)}"
    annotation[const.CONV][0][const.VALUE] = question
    annotation[const.CONV][1][const.VALUE] = answer

    qa_meta = copy.deepcopy(en_meta)
    qa_meta[const.NUM_OPTIONS] = num_options
    annotation[const.META] = qa_meta

    return annotation


def process_chunk(
    video_root,
    video,
    video_ext,
    min_options,
    max_options,
    non_sop_action,
    num_runs,
    generate_all_options,
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
    all_chunks = sorted(
        glob.glob(os.path.join(video_root, f"{video}/*.{video_ext}")),
        key=lambda x: int(os.path.basename(x).split(".")[-2].split("_")[-1]),
    )
    filtered_all_chunks = [
        vid
        for vid in all_chunks
        if int(os.path.basename(vid).split(const.VIDEO_ACTION_SEP)[0]) not in args.exclude_actions
    ]

    video_out_root = os.path.join(output_root, video)
    create_dir(video_out_root)

    # process chunks with multiple actions
    for i, cur_chunk in enumerate(filtered_all_chunks):
        # process each run
        for _ in range(num_runs):
            # do option construction here, we can use assemble_anns for partial and all options generation
            # random pick a number between min_options and max_options
            num_options = random.randint(min_options, max_options)

            # construct options
            options = [option for option in args.choices if option != args.choices[non_sop_action - 1]]
            options = random.sample(options, num_options - 1)
            options = options + [args.choices[non_sop_action - 1]]
            
            annotation = assemble_anns(
                chunk=cur_chunk,
                num_options=num_options,
                options=options,
                answer=args.choices[non_sop_action - 1],
                question_template=None
            )
            anns.append(annotation)
        
        if generate_all_options:
            # construct all options
            options = args.choices
            annotation = assemble_anns(
                chunk=cur_chunk,
                num_options=len(options),
                options=options,
                answer=args.choices[non_sop_action - 1],
                question_template=GOLDEN_QUESTION
            )
            anns.append(annotation)
        
        # copy video to output video root
        shutil.copyfile(cur_chunk, os.path.join(video_out_root, os.path.basename(cur_chunk)))

    return anns


def process_video(args_tuple):
    (
        video_root,
        video,
        video_ext,
        min_options,
        max_options,
        non_sop_action,
        num_runs,
        generate_all_options,
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
        num_runs,
        generate_all_options,
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
        "--exclude-action", type=str, default="", help="actions to exclude from extra data source"
    )
    parser.add_argument("--ext", type=str, default="mp4", help="video extension format")
    parser.add_argument("--min-options", type=int, default=4, help="minimum number of options")
    parser.add_argument("--max-options", type=int, default=6, help="maximum number of options")
    parser.add_argument("--non-sop-action", type=int, required=True, help="action index of non-SOP action of base dataset.")
    parser.add_argument("--num-runs", type=int, default=2, help="number of runs for extra negative samples")
    parser.add_argument("--generate-all-options", action="store_true", help="generate all options for extra negative samples")
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

    create_dir(args.output_root)

    # load choices
    choices_txt = os.path.join(args.config_root, f"{const.CHOICES}.txt")
    args.choices = read_txt(choices_txt).split(const.LINE_BREAK)

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
            args.num_runs,
            args.generate_all_options,
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