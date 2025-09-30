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



from .utils.helper import create_dir, read_txt
from .utils.const import API_BASE_URL, PROMPTS_ROOT

import os
import argparse
import glob
from openai import OpenAI

QA_SEP = "===\n"
INSTRUCT = "\n\nPlease generate {num_qa} question-answer pairs."
CAPTION_PREFIX = "Caption:\n"
DESCRIPTION_PREFIX = "Description:\n"

def process(args):

    system_prompt = read_txt(os.path.join(PROMPTS_ROOT, f"{args.task}_captions_to_gqa", "system_message.txt"))
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # load examples
    all_caps = sorted(glob.glob(os.path.join(PROMPTS_ROOT, f"{args.task}_captions_to_gqa", "*_caps.txt")))
    all_convs = sorted(glob.glob(os.path.join(PROMPTS_ROOT, f"{args.task}_captions_to_gqa", "*_conv.txt")))

    for cap, conv in zip(all_caps, all_convs):
        print(cap)
        print(conv)
        cur_caps = read_txt(cap)
        cur_conv = read_txt(conv)
        num_qa = len(cur_conv.split(QA_SEP)) // 2

        messages.append({"role": "user", "content": cur_caps + INSTRUCT.format(num_qa=num_qa)})
        messages.append({"role": "assistant", "content": cur_conv})

    captions = read_txt(args.caption_file)
    captions = CAPTION_PREFIX + captions + "\n" + DESCRIPTION_PREFIX + args.prior_knowledge

    messages.append(
        {"role": "user", "content": captions + INSTRUCT.format(num_qa=args.num_qa)}
    )

    if args.verbose:
        for message in messages:
            print(f"{message['role']}\n{message['content']}\n")


    client = OpenAI(
        base_url = API_BASE_URL,
        api_key = args.api_key
    )

    print("Start Inference")
    completion = client.chat.completions.create(
    model=args.model,
    messages=messages,
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=args.max_tokens,
    stream=False
    )

    llm_output = completion.choices[0].message.content

    # dump output
    with open(os.path.join(args.output_root, f"{args.output_name}.txt"), "w") as f:
        f.write(llm_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta/llama-3.1-70b-instruct")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--api-key", type=str, help="Nvidia API key")
    parser.add_argument("--task", type=str, help="Task type")
    parser.add_argument("--prior-knowledge", type=str, help="prior knowledge prompt")
    parser.add_argument("--caption-file", type=str, help="file that store captions")
    parser.add_argument("--num_qa", type=int, default=5, help="number of QA pairs")
    parser.add_argument("--output-name", type=str, help="file name to be saved")
    parser.add_argument("--output-root", type=str, help="output root")
    parser.add_argument("--verbose", type=bool, default=True)
    args = parser.parse_args()

    create_dir(args.output_root)
    process(args)