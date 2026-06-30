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
import cv2
import copy
import glob
import os
import random
import shutil
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Tuple
from moviepy.editor import VideoFileClip
from math import ceil

from .cfg.ds import QUESTION_TEMPLATE
from .utils import const
from .utils.annotation_template import ds_meta, llava_video
from .utils.helper import (
    get_video_meta,
    clean_sentence,
    create_dir,
    dump_json,
    read_json,
    read_txt,
    unpack_annotation,
    write_txt,
    write_video,
)
from .utils.logger import logging


def prepare_sample_qas(action_json):
    all_actions = read_json(action_json)[const.ACTION_JSON_KEY]
    config_root = os.path.join(str(Path(action_json).parent), const.DS)
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

    logging.info(f"Create DS config question: {const.QUESTION}.txt\n{question}")
    write_txt(os.path.join(config_root, const.QUESTION + ".txt"), question)

    logging.info(f"Create DS config choices: {const.CHOICES}.txt\n{choices}")
    write_txt(os.path.join(config_root, const.CHOICES + ".txt"), choices)

    return config_root


def read_random_frames(
    video_path: str,
    num_samples: int,
    target_size: Tuple[int, int],
    hard_neg_frames_ratio: float = 1.0,
    is_hard: bool = False
) -> Tuple[List[np.ndarray], int]:
    """
    Use MoviePy to sample frames at random times (mapped from random indices).
    Returns frames in RGB order.
    """
    frames: List[np.ndarray] = []
    with VideoFileClip(video_path) as clip:
        fps = float(clip.fps) if clip.fps else 30.0
        duration = float(clip.duration) if clip.duration else 0.0
        nframes_reader = getattr(getattr(clip, "reader", None), "nframes", 0) or 0
        # Prefer the actual frame count from ffmpeg reader; fall back to floor(duration * fps)
        # to avoid overestimating frames (round can exceed the real count).
        frame_count = int(nframes_reader) if nframes_reader > 0 else int(duration * fps)
        if frame_count <= 0:
            return [], 0

        if is_hard:
            num_hard_neg_samples = ceil(frame_count * hard_neg_frames_ratio) # take upperbound
            num_samples = min(num_samples, num_hard_neg_samples)
            mode = random.choice(const.DS_HARD_SAMPLING_MODE)
            if mode == "front":
                # pick the first num_hard_neg_samples frames
                indices = list(range(num_hard_neg_samples))
            elif mode == "end":
                # pick the last num_hard_negative_samples frames
                indices = list(range(frame_count - num_hard_neg_samples, frame_count))
            else: # ramdom mode
                indices = random.sample(range(frame_count), k=num_hard_neg_samples)
            logging.info(f"Hard negative sampling mode: {mode}")
        else:
            # random sample for non-hard negative samples
            num_samples = min(num_samples, frame_count)
            indices = random.sample(range(frame_count), k=num_samples)


        # Sort indices so we read frames sequentially — MoviePy uses a
        # sequential ffmpeg reader, so out-of-order access causes costly
        # seeks/restarts and can corrupt the internal pipe.
        sorted_indices = sorted(indices)
        logging.info(f"Sampled indices: {sorted_indices}")

        for i, frame in enumerate(clip.iter_frames(fps=fps, dtype="uint8")):
            if i in sorted_indices:
                if (frame.shape[1], frame.shape[0]) != target_size:
                    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                frames.append(frame)
            if len(frames) == len(sorted_indices):
                break  # early exit once all needed frames are collected

    return frames, num_samples


def construct_dynamic_shuffling_data(
    cur_chunk, 
    all_video_chunks,
    action_index,
    video_shape,
    fps,
    num_distractor,
    samples_per_video,
    video_out_root,
    run,
    hard_neg_frames_ratio,
    human_prompts,
    answer,
    is_hard=False):

    # sample distractor videos, and if the prefix is the same as the base video, skip and resample
    distractor_videos = []
    while len(distractor_videos) < num_distractor:
        distractor_video = random.choice(all_video_chunks)
        distractor_action = int(os.path.basename(distractor_video).split(const.VIDEO_ACTION_SEP)[0])
        if distractor_action == action_index + 1:
            continue
        distractor_videos.append(distractor_video)
    
    # construct dynamic shuffled video
    all_source_videos = [cur_chunk] + distractor_videos

    # sample frames from all source videos
    all_frames = []
    all_num_sample_frames = []
    for source_video in all_source_videos:
        frames, num_sample_frames = read_random_frames(
            source_video, 
            samples_per_video, 
            video_shape, 
            is_hard=is_hard, 
            hard_neg_frames_ratio=hard_neg_frames_ratio)

        all_frames.extend(frames)
        all_num_sample_frames.append(num_sample_frames)

    # shuffle frames
    if not is_hard:
        random.shuffle(all_frames)

    # write to video
    output_video = cur_chunk.replace(".mp4", f"_dynamic_shuffled_{run}.mp4") if not is_hard \
        else cur_chunk.replace(".mp4", f"_dynamic_shuffled_{run}_hard.mp4")
    output_video_name = os.path.basename(output_video)
    write_video(all_frames, os.path.join(video_out_root, output_video_name), fps, video_shape)

    
    # construct annotation json
    qa_label = copy.deepcopy(llava_video)

    # assemble output qa label
    qa_label[const.CONV][0][const.VALUE] = human_prompts
    qa_label[const.CONV][1][const.VALUE] = answer
    qa_label[const.VIDEO] = f"videos/{output_video_name}"

    qa_meta = copy.deepcopy(ds_meta)
    qa_meta[const.FRAME_COUNTS] = all_num_sample_frames
    qa_meta[const.SOURCE_VIDEOS] = all_source_videos
    qa_meta[const.TOTAL_FRAMES] = sum(all_num_sample_frames)
    qa_meta[const.IS_HARD] = is_hard
    qa_label[const.META] = qa_meta

    return qa_label


def process_chunk(
    cur_chunk,
    all_video_chunks,
    min_distractor,
    max_distractor,
    non_sop_action,
    num_runs,
    num_hard_neg,
    hard_neg_frames_ratio,
    seed,
    output_root,
    args,
):
    if seed is not None:
        random.seed(seed)
    else:
        random.seed(os.getpid())

    anns = []

    video = os.path.basename(os.path.dirname(cur_chunk))
    video_out_root = os.path.join(output_root, video)
    create_dir(video_out_root)

    # process chunk
    cur_action = int(os.path.basename(cur_chunk).split(const.VIDEO_ACTION_SEP)[0])
    action_index = cur_action - 1
    non_sop_action_index = non_sop_action - 1
    frame_count, fps, video_shape = get_video_meta(cur_chunk)

    logging.info(f"Processing chunk: {cur_chunk}, action index: {action_index}")

    # random pick a number between min_distractor and max_distractor
    num_distractor = random.randint(min_distractor, max_distractor)
    samples_per_video = frame_count // (num_distractor + 1) + 1

    # construct dynamic shuffled video
    for run in range(num_runs):
        annotation = construct_dynamic_shuffling_data(
            cur_chunk,
            all_video_chunks,
            action_index,
            video_shape,
            fps,
            num_distractor,
            samples_per_video,
            video_out_root,
            run,
            hard_neg_frames_ratio,
            args.human_prompts,
            args.choices[non_sop_action_index],
            is_hard=False
        )
        anns.append(annotation)
    
    # construct hard negative samples
    for run in range(num_hard_neg):
        annotation = construct_dynamic_shuffling_data(
            cur_chunk,
            all_video_chunks,
            action_index,
            video_shape,
            fps,
            num_distractor,
            samples_per_video,
            video_out_root,
            run,
            hard_neg_frames_ratio,
            args.human_prompts,
            args.choices[non_sop_action_index],
            is_hard=True
        )
        anns.append(annotation)

    return anns


def process_video(args_tuple):
    (
        cur_chunk,
        all_video_chunks,
        min_distractor,
        max_distractor,
        non_sop_action,
        num_runs,
        num_hard_neg,
        hard_neg_frames_ratio,
        seed,
        output_root,
        args,
    ) = args_tuple
    return process_chunk(
        cur_chunk,
        all_video_chunks,
        min_distractor,
        max_distractor,
        non_sop_action,
        num_runs,
        num_hard_neg,
        hard_neg_frames_ratio,
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
    parser.add_argument("--min-distractor", type=int, default=3, help="minimum number of distractor videos")
    parser.add_argument("--max-distractor", type=int, default=6, help="maximum number of distractor videos")
    parser.add_argument("--non-sop-action", type=int, required=True, help="action index of non-SOP action.")
    parser.add_argument("--num-runs", type=int, default=1, help="number of runs for dynamic shuffling")
    parser.add_argument("--num-hard-neg", type=int, default=0, help="number of hard negative samples")
    parser.add_argument("--hard-neg-frames-ratio", type=float, default=0.1, help="ratio of frames to be sampled as hard negative samples")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
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

    # load all available video chunks
    # sort by the name.split(".")[-2].split("_")[-1], which is the timeline index
    # video file naming convention: <action_number>_<video_name>_<duplication_cnt>_<timeline_index>.<video_ext>
    all_video_chunks = sorted(
        glob.glob(os.path.join(args.video_root, f"*/*.{args.ext}")),
        key=lambda x: int(os.path.basename(x).split(".")[-2].split("_")[-1]),
    )

    filtered_all_video_chunks = [
        chunk
        for chunk in all_video_chunks
        if int(os.path.basename(chunk).split(const.VIDEO_ACTION_SEP)[0]) not in args.exclude_actions
    ]


    # start augmenting
    # Prepare arguments for multiprocessing
    args_list = [
        (
            chunk,
            filtered_all_video_chunks,
            args.min_distractor,
            args.max_distractor,
            args.non_sop_action,
            args.num_runs,
            args.num_hard_neg,
            args.hard_neg_frames_ratio,
            args.seed,
            args.output_root,
            args,
        )
        for chunk in filtered_all_video_chunks
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