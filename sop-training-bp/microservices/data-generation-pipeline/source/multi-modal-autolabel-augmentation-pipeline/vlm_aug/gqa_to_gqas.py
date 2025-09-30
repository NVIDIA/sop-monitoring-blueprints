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
from openai import OpenAI

from .cfg.gqa import ANSWERS_TEMPLATE, QUESTION_TEMPLATE
from .cfg.llm import llm_cfg
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


QA_SEP = "===\n"
Q_START = "Question:\n"
A_START = "Answer:\n"
INSTRUCT = "\n\nPlease generate {num_qa} question-answer pairs. Only output question-answer pairs."
LLM_OUT_ROOT = "GQA2GQAs"


def validate_format(s):
    blocks = s.split(QA_SEP)

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

    client = OpenAI(
        base_url=const.API_BASE_URL if args.api_key != "" else args.local_llm_url,
        api_key=args.api_key if args.api_key != "" else "not-used",
    )

    logging.info("Start Inference")
    completion = client.chat.completions.create(
        model=args.llm,
        messages=cur_messages,
        temperature=llm_cfg["temperature"],
        top_p=llm_cfg["top_p"],
        max_tokens=llm_cfg["max_tokens"],
        stream=False,
    )

    llm_output = completion.choices[0].message.content
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
        vid_basename = Path(cur_video).stem
        cur_action = int(vid_basename.split(const.VIDEO_ACTION_SEP)[0])

        # skip excluded actions
        if cur_action in args.exclude_actions:
            logging.info(f"Skip excluded action in GQA to GQAs: {cur_action}")
            continue

        # get corresponding sample qa file
        sample_qa_file = os.path.join(args.sample_qa_root, f"{const.ACTION}{cur_action}.txt")
        if not os.path.exists(sample_qa_file):
            logging.warning(f"Sample qa file: {sample_qa_file} not exist. Skip this sample qa file.")
            continue

        all_qa = llm_gen(sample_qa_file, num_qa_llm, os.path.join(output_root, LLM_OUT_ROOT), video, args, messages)

        total_qa_req = num_qa_per_chunk
        do_replacement = args.replace if len(all_qa) >= total_qa_req else True
        frame_cnts = [int(cv2.VideoCapture(cur_video).get(cv2.CAP_PROP_FRAME_COUNT))]

        if do_replacement:
            # Random Sample QAs with Replacement
            picked_qas = random.choices(all_qa, k=total_qa_req)
        else:
            # Random Sample QAs without Replacement
            picked_qas = random.sample(all_qa, k=total_qa_req)

        for picked_qa in picked_qas:
            qst = picked_qa[0]
            ans = picked_qa[1]

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

    for i, cur_video in enumerate(all_videos):
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
    args = parser.parse_args()

    # split exclude action into a list of int
    if args.exclude_action != "":
        args.exclude_actions = [int(action) for action in args.exclude_action.split(const.VIDEO_ACTION_SEP)]
    else:
        args.exclude_actions = []

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
        with ProcessPoolExecutor(max_workers=cpu_count() - 1) as executor:
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
