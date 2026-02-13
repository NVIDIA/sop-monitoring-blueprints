######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
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
from moviepy.editor import VideoFileClip, concatenate_videoclips

from .cfg.mcq import QUESTION_TEMPLATE
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
    config_root = os.path.join(str(Path(action_json).parent), const.MCQ)
    create_dir(config_root)

    choices = ""

    for i, cur_action in enumerate(all_actions, 1):
        cleaned_action = clean_sentence(cur_action)
        cleaned_action = f"({i}) " + cleaned_action

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

    logging.info(f"Create MCQ config question: {const.QUESTION}.txt\n{question}")
    write_txt(os.path.join(config_root, const.QUESTION + ".txt"), question)

    logging.info(f"Create MCQ config choices: {const.CHOICES}.txt\n{choices}")
    write_txt(os.path.join(config_root, const.CHOICES + ".txt"), choices)

    return config_root


def assemble_anns(index, video_name, min_frames, max_frames, frame_cnts, frames_upperbound, dynamic_sample, args):
    if frames_upperbound > 0:
        max_frames = frames_upperbound // len(frame_cnts)

    smallest_frame_cnts = min(frame_cnts)
    selected_choices = [args.choices[i - 1] for i in index]
    # assemble gpt response
    gpt = " ".join(selected_choices)

    # assemble output qa label
    qa_label = copy.deepcopy(llava_video)
    qa_label[const.CONV][0][const.VALUE] = args.human_prompts
    qa_label[const.CONV][1][const.VALUE] = gpt
    qa_label[const.VIDEO] = f"videos/{video_name}"

    if dynamic_sample:
        # append dynamic sample metadata
        qa_meta = copy.deepcopy(dynamic_meta)
        qa_meta[const.FRAME_COUNTS] = frame_cnts
        qa_meta[const.MIN_FRAMES] = int(min(smallest_frame_cnts, min_frames))
        qa_meta[const.MAX_FRAMES] = int(min(smallest_frame_cnts, max_frames))
        qa_meta[const.DYNAMIC_SAMPLE] = dynamic_sample
        qa_label[const.META] = qa_meta

    return qa_label


def trunk_merging(
    video_root,
    video,
    video_ext,
    max_length,
    output_root,
    min_frames,
    max_frames,
    frames_upperbound,
    dynamic_sample,
    args,
):
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
        if int(os.path.basename(vid).split(const.VIDEO_ACTION_SEP)[0]) not in args.exclude_actions
    ]
    seen = {}

    video_basename = f"{video}.{video_ext}"
    video_out_root = os.path.join(output_root, video)
    create_dir(video_out_root)

    # process chunks with multiple actions
    for i, cur_video in enumerate(filtered_all_videos):
        for chunk_len in range(2, max_length + 1):
            if i + chunk_len > len(filtered_all_videos):
                # break if the end_idx exceed the last action
                break

            video_files = filtered_all_videos[i : i + chunk_len]
            index = [int(os.path.basename(video).split(const.VIDEO_ACTION_SEP)[0]) for video in video_files]
            prefix = "-".join(map(str, index))

            frame_cnts = [int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT)) for video in video_files]
            clips = [VideoFileClip(video) for video in video_files]
            final_clip = concatenate_videoclips(clips, method="compose")
            final_clip.write_videofile(os.path.join(video_out_root, f"{prefix}_{video_basename}"), logger=None, verbose=False)

            # generate labels
            anns.append(
                assemble_anns(
                    index,
                    f"{prefix}_{video_basename}",
                    min_frames,
                    max_frames,
                    frame_cnts,
                    frames_upperbound,
                    dynamic_sample,
                    args,
                )
            )
            seen[f"{prefix}"] = True

    # process chunks with 1 action only
    for i, video in enumerate(all_videos):
        cur_action = int(os.path.basename(video).split(const.VIDEO_ACTION_SEP)[0])

        shutil.copyfile(video, os.path.join(video_out_root, os.path.basename(video)))
        cur_frame_cnt = cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT)
        anns.append(
            assemble_anns(
                [cur_action],
                os.path.basename(video),
                min_frames,
                max_frames,
                [cur_frame_cnt],
                frames_upperbound,
                dynamic_sample,
                args,
            )
        )

    logging.info(f"PID: {os.getpid()}")
    logging.info(f"Seen: {seen}")
    logging.info(f"All source videos: {all_videos}")
    return anns


def process_video(args_tuple):
    (
        video_root,
        video,
        video_ext,
        max_length,
        output_root,
        min_frames,
        max_frames,
        frames_upperbound,
        dynamic_sample,
        args,
    ) = args_tuple
    return trunk_merging(
        video_root,
        video,
        video_ext,
        max_length,
        output_root,
        min_frames,
        max_frames,
        frames_upperbound,
        dynamic_sample,
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
    parser.add_argument("--max-chunk-len", type=int, default=4, help="maximum of actions in a chunk")
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
    args = parser.parse_args()

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

    # split exclude action into a list of int
    if args.exclude_action != "":
        args.exclude_actions = [int(action) for action in args.exclude_action.split(const.VIDEO_ACTION_SEP)]
    else:
        args.exclude_actions = []

    create_dir(args.output_root)

    # load question and choices
    question_txt = os.path.join(args.config_root, f"{const.QUESTION}.txt")
    choices_txt = os.path.join(args.config_root, f"{const.CHOICES}.txt")

    args.human_prompts = read_txt(question_txt).replace("\\n", "\n")  # not sure why \n would become \\n
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
            args.max_chunk_len,
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
