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
SOP per-action-chunk VLM inference script.
Adapted from feature_to_add/cr_per_action_chunk_evaluation/mcq_eval.py.

Key differences:
  - parse_action_index uses annotation MS filename format: {NN}_{video}_{rep}_{timeline}.mp4
  - No labels.txt dependency — outputs raw inference JSON only
  - --output-dir controls where inference_results.json is written
"""

import argparse
import glob
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

# MUST precede the torch import (chained from vllm). expandable_segments
# lets the caching allocator release segments so vLLM's KV-cache profiling
# isn't blocked by fragmentation. Mirrors sop_e2e_eval.py.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Script-mode entry: put the microservice root on sys.path so `utils.*`
# resolves. Mirrors sop_e2e_eval.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_SYSTEM_PROMPT = "Answer the questions."


def parse_action_index(filename: str) -> list:
    """
    Extract 1-based action index(es) from annotation MS chunk filename.

    Annotation MS formats:
      single-op:   {NN}_{video_name}_{rep}_{timeline}.mp4       -> [NN]
      two-op:      {NN}-{MM}_{video_name}_{rep}_{timeline}.mp4  -> sorted([NN, MM])

    Always returns a 1-based action-id list (length 1 for single-op,
    length >= 2 for concurrent two-op chunks).
    """
    try:
        stem = Path(filename).stem                  # "01_..." or "01-04_..."
        prefix = stem.split("_")[0]                 # "01" or "01-04"
        ids = [int(p) for p in prefix.split("-")]   # [1] or [1, 4]
        return sorted(ids)
    except (ValueError, IndexError):
        logging.warning(f"Cannot parse action index from filename: {filename}, using [1] as default")
        return [1]


def read_txt(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def action_inference_transformers(args, model, processor, generation_config,
                                   system_prompt=DEFAULT_SYSTEM_PROMPT,
                                   resolution_config=None):
    """Run inference using transformers backend. Returns raw inference results dict."""
    from qwen_vl_utils import process_vision_info
    if resolution_config is None:
        # Default mirrors training config (max_frames=40, 16k vision
        # tokens) — eval-at-training-resolution keeps the VLM in-distribution.
        resolution_config = {"max_frames": 40, "total_pixels": 16572416}

    qs = read_txt(os.path.join(args.asset_root, args.vlm_prompts_file))
    logging.info(f"VLM prompt loaded from {args.asset_root}/{args.vlm_prompts_file}")

    all_video_dirs = sorted([
        d for d in os.listdir(args.val_videos_path)
        if os.path.isdir(os.path.join(args.val_videos_path, d))
    ])
    inference_results = defaultdict(list)
    logging.info(f"Found {len(all_video_dirs)} video directories to process")

    for video in all_video_dirs:
        logging.info(f"Processing video: {video}")
        chunk_paths = sorted(
            glob.glob(os.path.join(args.val_videos_path, video, "*.mp4"))
            + glob.glob(os.path.join(args.val_videos_path, video, "*.MP4"))
        )
        for chunk in chunk_paths:
            cur_action = parse_action_index(os.path.basename(chunk))

            if args.use_fps_or_nframes == "fps":
                video_kwargs_extra = {"fps": args.fps}
            else:
                import cv2
                cap = cv2.VideoCapture(chunk)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                video_kwargs_extra = {"nframes": min(total_frames - 1, args.nframes)}

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "video", "video": chunk, **resolution_config, **video_kwargs_extra},
                    {"type": "text", "text": qs},
                ]},
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            if isinstance(video_kwargs.get("fps"), list):
                video_kwargs["fps"] = float(video_kwargs["fps"][0])

            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt", **video_kwargs,
            ).to(model.device)

            generated_ids = model.generate(**inputs, **generation_config)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            response = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # Format contract for sop-rca-plugin/analyze_by_action_confusion.py:
            # 'Action Chunk: <full_path>' + next-line response. Emitted via
            # print to avoid a logging timestamp that would break the parser.
            print(f"Action Chunk: {chunk}", flush=True)
            print(response, flush=True)
            inference_results[video].append([cur_action, response, chunk])

    return inference_results


def action_inference_vllm(args, llm, processor, sampling_params,
                           system_prompt=DEFAULT_SYSTEM_PROMPT,
                           resolution_config=None):
    """Run inference using vllm backend. Returns raw inference results dict."""
    from qwen_vl_utils import process_vision_info
    from utils.eval_utils import build_vllm_video_mm_data
    if resolution_config is None:
        # Default mirrors training config (max_frames=40, 16k vision
        # tokens) — eval-at-training-resolution keeps the VLM in-distribution.
        resolution_config = {"max_frames": 40, "total_pixels": 16572416}

    qs = read_txt(os.path.join(args.asset_root, args.vlm_prompts_file))
    logging.info("VLM prompt loaded")

    all_video_dirs = sorted([
        d for d in os.listdir(args.val_videos_path)
        if os.path.isdir(os.path.join(args.val_videos_path, d))
    ])
    inference_results = defaultdict(list)
    logging.info(f"Found {len(all_video_dirs)} video directories")

    for video in all_video_dirs:
        logging.info(f"Processing video: {video}")
        chunk_paths = sorted(
            glob.glob(os.path.join(args.val_videos_path, video, "*.mp4"))
            + glob.glob(os.path.join(args.val_videos_path, video, "*.MP4"))
        )
        for chunk in chunk_paths:
            cur_action = parse_action_index(os.path.basename(chunk))

            if args.use_fps_or_nframes == "fps":
                video_kwargs_extra = {"fps": args.fps}
            else:
                import cv2
                cap = cv2.VideoCapture(chunk)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                video_kwargs_extra = {"nframes": min(total_frames - 1, args.nframes)}

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "video", "video": chunk, **resolution_config, **video_kwargs_extra},
                    {"type": "text", "text": qs},
                ]},
            ]

            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            videos_with_metadata = build_vllm_video_mm_data(video_inputs, video_kwargs, args.fps)
            if videos_with_metadata is not None:
                # MultiModalDataParser expects the singular "video" key with
                # a list of (array, metadata) tuples. The plural "videos" key
                # in qwen3_vl.py is the post-parse internal name only.
                mm_data["video"] = videos_with_metadata

            outputs = llm.generate(
                [{"prompt": prompt, "multi_modal_data": mm_data}],
                sampling_params, use_tqdm=False,
            )
            response = outputs[0].outputs[0].text
            # Same parser contract as the transformers branch.
            print(f"Action Chunk: {chunk}", flush=True)
            print(response, flush=True)
            inference_results[video].append([cur_action, response, chunk])

    return inference_results


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="SOP per-action-chunk VLM evaluation inference")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--val-videos-path", type=str, required=True)
    parser.add_argument("--asset-root", type=str, default="./assets")
    parser.add_argument("--vlm-prompts-file", type=str, default="vlm_prompts.txt")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write inference_results.json")
    parser.add_argument("--output-name", type=str, default="inference_results")
    parser.add_argument("--use-fps-or-nframes", type=str, default="fps", choices=["fps", "nframes"])
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--nframes", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--backend", type=str, default="vllm", choices=["transformers", "vllm"])
    parser.add_argument("--system-prompt", type=str, default="Answer the questions.")
    parser.add_argument(
        "--resolution-config", type=str, default='{"max_pixels": 81920}',
        help=(
            "JSON-encoded ResolutionConfig (see validation/request_validation.py). "
            'Fields: max_frames, total_pixels, resized_height, resized_width, '
            'max_pixels, min_pixels. Example: '
            '\'{"max_frames": 40, "total_pixels": 16572416}\'.'
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=0,
        help="vLLM tensor-parallel size. 0 = auto-detect from torch.cuda.device_count().",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Stdout-only; app.py merges stdout+stderr into <output_dir>/log.txt.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )

    # sop-rca-plugin parses this Args: line to extract fps_by_action.
    logging.info("Args: %s", vars(args))

    try:
        resolution_config = json.loads(args.resolution_config)
    except json.JSONDecodeError:
        resolution_config = {"max_frames": 40, "total_pixels": 16572416}

    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor
        import torch
        tp_size = args.tensor_parallel_size
        if tp_size <= 0:
            tp_size = max(1, torch.cuda.device_count())
        # gpu_memory_utilization=0.7 (vs vLLM's 0.9 default): leaves headroom
        # for concurrent GPU consumers (other trainings/inference) on the same
        # host. Mirrored in the e2e VLM stage.
        # disable_custom_all_reduce=True: vLLM's custom AR kernel is fragile
        # across GPU topologies; NCCL fallback is fine for TP=1/TP=2.
        logging.info(
            f"vLLM tensor_parallel_size={tp_size} (visible CUDA devices: {torch.cuda.device_count()}), "
            f"gpu_memory_utilization=0.7, disable_custom_all_reduce=True"
        )
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=0.7,
            disable_custom_all_reduce=True,
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
        sampling_params = SamplingParams(
            temperature=args.temperature, max_tokens=args.max_new_tokens, top_p=args.top_p,
        )
        inference_results = action_inference_vllm(
            args, llm, processor, sampling_params,
            system_prompt=args.system_prompt, resolution_config=resolution_config,
        )
    else:
        # Auto-dispatch by `config.architectures[0]`: Qwen2_5_VL for CR1,
        # Qwen3VL for CR2. Hard-coding Qwen2_5 silently mis-initialised
        # CR2 weights and crashed in get_rope_index.
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
        generation_config = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_beams": args.num_beams,
        }
        inference_results = action_inference_transformers(
            args, model, processor, generation_config,
            system_prompt=args.system_prompt, resolution_config=resolution_config,
        )

    out_path = os.path.join(args.output_dir, f"{args.output_name}.json")
    with open(out_path, "w") as f:
        json.dump(dict(inference_results), f, indent=2)
    logging.info(f"Inference results written to: {out_path}")
