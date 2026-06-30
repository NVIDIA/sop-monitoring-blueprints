######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
SOP end-to-end evaluation script.

Invoked as a subprocess by the evaluation microservice.
Pipeline: DDM temporal segmentation → VLM action recognition → accuracy metrics.

Outputs (to --output-dir):
  - outputs_temporal_segmentation/f1_X.XX.json  (per-video boundaries + F1 metrics)
  - outputs_temporal_segmentation/{video}.png   (DDM score visualization)
  - outputs_temporal_segmentation/video_to_boundaries_debug.json
  - outputs_temporal_segmentation/video_to_ddm_info_debug.json
  - outputs_action_recognition/video_name_to_output_text.json
  - e2e_results.json  (combined temporal + action accuracy)
"""

import argparse
import glob
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

# expandable_segments lets DDM-stage memory return to the driver before
# vLLM claims its KV cache. MUST precede the torch import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.eval_utils import verify_pred
from utils.e2e_eval_utils import (
    compute_temporal_metrics,
    extract_golden_boundaries,
    get_video_duration_sec,
    map_chunks_to_ground_truth,
    uniform_chunk_boundaries,
    visualize_ddm_scores,
)

DEFAULT_SYSTEM_PROMPT = "Answer the questions."


def collect_annotations(video_dir: str) -> dict:
    """
    Collect per-video annotation JSONs from subdirectories.

    Expects structure:
        video_dir/{video_name}/{video_name}_annotation.json

    Returns:
        {video_name.mp4: [events_list]}
    """
    result = {}
    for entry in sorted(os.scandir(video_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        # Strict *_annotation.json — falling back to *.json silently
        # ingests metadata/results files and corrupts golden annotations.
        json_files = [
            f for f in os.listdir(entry.path)
            if f.endswith("_annotation.json")
        ]
        if not json_files:
            logging.warning(
                "Skipping %s: no *_annotation.json file found in %s",
                entry.name, entry.path,
            )
            continue

        anno_file = os.path.join(entry.path, json_files[0])
        with open(anno_file, "r") as f:
            anno_data = json.load(f)

        video_key = f"{entry.name}.mp4"
        result[video_key] = anno_data

    return result


def compute_e2e_accuracy(
    vlm_outputs: dict[str, dict[str, str]],
    chunk_action_map: dict[str, list[int]],
    choices: list[str],
) -> dict:
    """
    Compute per-action accuracy from VLM outputs and chunk-to-action mapping.

    Args:
        vlm_outputs: {video_name: {chunk_key: vlm_response}}
        chunk_action_map: {video_name: [action_idx_1based, ...]}
        choices: ["(1) label", "(2) label", ...]

    Returns:
        {"overall_accuracy": float, "per_action": {action_idx: {label, correct, total, accuracy}}}
    """
    per_action = {}
    total_correct = 0
    total_samples = 0

    for video_name, chunk_texts in vlm_outputs.items():
        action_indices = chunk_action_map.get(video_name, [])
        chunk_keys = sorted(chunk_texts.keys())

        for i, chunk_key in enumerate(chunk_keys):
            if i >= len(action_indices):
                break

            action_idx = action_indices[i]
            vlm_response = chunk_texts[chunk_key]
            key = str(action_idx)

            if key not in per_action:
                label = choices[action_idx - 1] if 0 < action_idx <= len(choices) else f"(?) unknown {action_idx}"
                per_action[key] = {"label": label, "correct": 0, "total": 0, "accuracy": 0.0}

            expected_label = choices[action_idx - 1] if 0 < action_idx <= len(choices) else ""
            is_correct = verify_pred(vlm_response, expected_label)

            per_action[key]["total"] += 1
            if is_correct:
                per_action[key]["correct"] += 1
            total_correct += int(is_correct)
            total_samples += 1

    for key in per_action:
        t = per_action[key]["total"]
        per_action[key]["accuracy"] = per_action[key]["correct"] / t if t > 0 else 0.0

    overall = total_correct / total_samples if total_samples > 0 else 0.0
    return {"overall_accuracy": overall, "per_action": per_action}


def read_txt(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def dump_temporal_segmentation_args(args, ts_output_dir: str) -> None:
    """
    Write `outputs_temporal_segmentation/temporal_segmentation.log` with a
    single Args line in the same shape the standalone DDM script produces.

    The sop-rca-plugin (SKILL.md Step 2 — `resolution`, `nms_sec`,
    `score_threshold`) parses this file. We must translate our internal
    `ddm_*` prefixed field names back to the bare names the parser expects.
    """
    os.makedirs(ts_output_dir, exist_ok=True)
    log_path = os.path.join(ts_output_dir, "temporal_segmentation.log")
    fields = (
        f"resolution={getattr(args, 'ddm_resolution', None)}, "
        f"nms_sec={getattr(args, 'nms_sec', None)}, "
        f"score_threshold={getattr(args, 'score_threshold', None)}, "
        f"batch_size={getattr(args, 'ddm_batch_size', None)}, "
        f"frames_per_segment_hint={getattr(args, 'frames_per_segment_hint', None)}, "
        f"frames_per_side={getattr(args, 'ddm_frames_per_side', None)}"
    )
    with open(log_path, "w") as f:
        f.write(f"Args: Namespace({fields})\n")


def dump_action_recognition_args(args, ar_output_dir: str, resolution_config: dict) -> None:
    """
    Write `outputs_action_recognition/action_recognition_multi_gpu.log` with
    a single Args line. The sop-rca-plugin (SKILL.md Step 2 — `max_frames`)
    parses this file.
    """
    os.makedirs(ar_output_dir, exist_ok=True)
    log_path = os.path.join(ar_output_dir, "action_recognition_multi_gpu.log")
    rc = resolution_config or {}
    fields = (
        f"max_frames={rc.get('max_frames')}, "
        f"total_pixels={rc.get('total_pixels')}, "
        f"resized_height={rc.get('resized_height')}, "
        f"resized_width={rc.get('resized_width')}, "
        f"max_pixels={rc.get('max_pixels')}, "
        f"min_pixels={rc.get('min_pixels')}, "
        f"temperature={getattr(args, 'temperature', None)}, "
        f"top_p={getattr(args, 'top_p', None)}, "
        f"fps={getattr(args, 'fps', None)}"
    )
    with open(log_path, "w") as f:
        f.write(f"Args: Namespace({fields})\n")


# =============================================================================
# DDM Stage
# =============================================================================

def run_ddm_stage(args, anno_json: dict) -> dict:  # pragma: no cover
    """
    Run DDM temporal segmentation on all videos.

    Returns:
        {video_name: {"boundaries": [...], "metric": {...}}}
    """
    import torch
    from utils.e2e_eval_utils import extract_golden_boundaries

    ts_output_dir = os.path.join(args.output_dir, "outputs_temporal_segmentation")
    os.makedirs(ts_output_dir, exist_ok=True)

    # Shim Args line for the sop-rca-plugin parser (SKILL.md Step 2).
    dump_temporal_segmentation_args(args, ts_output_dir)

    anno_json_path = os.path.join(ts_output_dir, "anno.json")
    with open(anno_json_path, "w") as f:
        json.dump(anno_json, f, indent=2)

    golden_boundaries = extract_golden_boundaries(anno_json_path)
    with open(os.path.join(ts_output_dir, "video_to_boundaries_debug.json"), "w") as f:
        json.dump(golden_boundaries, f, indent=2)

    videos = sorted(glob.glob(os.path.join(args.video_dir, f"*.{args.video_ext}")))
    if not videos:
        raise FileNotFoundError(f"No .{args.video_ext} files found in {args.video_dir}")
    logging.info("Found %d videos for DDM inference", len(videos))

    # Single-GPU, in-process DDM-Net inference (vendored resnetGEBD at
    # $DDM_BASE_PATH/DDM-Net; PyAV for decoding).
    from utils.ddm_inference import load_ddm_model, run_ddm_inference

    ddm_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ddm_model = load_ddm_model(
        checkpoint_path=args.ddm_checkpoint_path,
        frames_per_side=args.ddm_frames_per_side,
        device=ddm_device,
    )

    # Tail-trim policy: when a video's mp4 extends past its annotated range,
    # DDM-Net would emit a phantom boundary on the unannotated tail and the
    # VLM would classify the fragment as non-SOP. That inserted (10) sits at
    # the end of the predicted sequence, surviving remove_continuous_rep,
    # and costs one duplicate per affected video — pure noise from the
    # eval grader's perspective. We compute a per-video decode cap from the
    # annotation's max end_timestamp; a small grace (DDM_TRIM_GRACE_SEC)
    # avoids cutting frames that fall within encoding/fps rounding error
    # of the last annotated event.
    DDM_TRIM_GRACE_SEC = 0.5

    video_to_ddm_info = {}
    for video_path in videos:
        video_name = os.path.basename(video_path)
        chunks = anno_json.get(video_name, [])
        end_cap = None
        if chunks:
            try:
                max_end = max(float(c["end_timestamp"]) for c in chunks)
                end_cap = max_end + DDM_TRIM_GRACE_SEC
            except (KeyError, TypeError, ValueError):
                # Missing or malformed end_timestamp — decode full video.
                end_cap = None
        logging.info(
            "DDM inference for: %s%s",
            video_name,
            f" (tail cap = {end_cap:.2f}s)" if end_cap is not None else "",
        )
        scores, metadata = run_ddm_inference(
            ddm_model,
            video_path,
            resolution=args.ddm_resolution,
            frames_per_side=args.ddm_frames_per_side,
            batch_size=args.ddm_batch_size,
            device=ddm_device,
            frames_per_segment_hint=args.frames_per_segment_hint,  # accepted for parity, unused
            end_timestamp_sec=end_cap,
        )
        video_to_ddm_info[video_name] = {
            "scores": scores,
            "fps": metadata.fps,
            "duration_sec": metadata.duration_sec,
        }

    with open(os.path.join(ts_output_dir, "video_to_ddm_info_debug.json"), "w") as f:
        json.dump(video_to_ddm_info, f, indent=2)

    # Full teardown before vLLM takes the device: del + empty_cache alone
    # leaves CUDA context that vLLM's custom AR kernel rejects.
    import gc
    logging.info("Freeing DDM model from GPU memory")
    del ddm_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    video_names = list(video_to_ddm_info.keys())
    nms_sec_list = []
    for vn in video_names:
        if args.nms_sec == 0.0:
            nms_sec_list.append(0.025 * video_to_ddm_info[vn]["duration_sec"])
        else:
            nms_sec_list.append(args.nms_sec)

    logging.info("Post-processing DDM results for %d videos...", len(video_names))

    from utils.ddm_inference import detect_boundaries

    video_to_metric = {}
    f1_list, precision_list, recall_list = [], [], []
    tp_list, fp_list, fn_list = [], [], []

    for i, video_name in enumerate(video_names):
        info = video_to_ddm_info[video_name]
        scores = info["scores"]
        fps = info["fps"]
        duration_sec = info["duration_sec"]
        nms_sec = nms_sec_list[i]

        nms_size = int(nms_sec * fps)
        boundary_frames = detect_boundaries(scores, args.score_threshold, nms_size)
        boundaries = [0.0] + [bdy / fps for bdy in boundary_frames] + [duration_sec]

        golden_bdy = golden_boundaries.get(video_name)
        metric = compute_temporal_metrics(golden_bdy, boundaries, duration_sec)

        ret = {"boundaries": boundaries, "ddm_threshold": duration_sec * 0.025, "metric": metric}
        video_to_metric[video_name] = ret

        png_path = os.path.join(ts_output_dir, f"{Path(video_name).stem}.png")
        visualize_ddm_scores(video_name, golden_bdy, boundaries, scores, fps, png_path)

        if metric["F1"] is not None:
            f1_list.append(metric["F1"])
            precision_list.append(metric["Precision"])
            recall_list.append(metric["Recall"])
            tp_list.append(metric["True Positive"])
            fp_list.append(metric["False Positive"])
            fn_list.append(metric["False Negative"])

    video_to_metric["avg_f1"] = float(np.mean(f1_list)) if f1_list else 0.0
    video_to_metric["avg_precision"] = float(np.mean(precision_list)) if precision_list else 0.0
    video_to_metric["avg_recall"] = float(np.mean(recall_list)) if recall_list else 0.0
    video_to_metric["avg_tp"] = float(np.mean(tp_list)) if tp_list else 0.0
    video_to_metric["avg_fp"] = float(np.mean(fp_list)) if fp_list else 0.0
    video_to_metric["avg_fn"] = float(np.mean(fn_list)) if fn_list else 0.0

    avg_f1 = video_to_metric["avg_f1"]
    result_json = os.path.join(ts_output_dir, f"f1_{avg_f1:.2f}.json")
    with open(result_json, "w") as f:
        json.dump(video_to_metric, f, indent=2)
    logging.info("DDM stage complete. F1 results: %s", result_json)

    return video_to_metric


# =============================================================================
# Uniform Stage (alternative to DDM — fixed-length time slices)
# =============================================================================

def run_uniform_stage(args, anno_json: dict) -> dict:
    """
    Stage 1 alternative: split each video into fixed-length chunks of
    `args.chunk_length_sec` seconds. Output shape is identical to
    run_ddm_stage so the VLM stage downstream is unchanged.

    Mirrors the "uniform" branch of the inference pipeline's
    chunking_options. No learned segmentation: read total duration from
    PyAV stream metadata (no decode) and split into fixed-length chunks
    of args.chunk_length_sec.

    Returns:
        {video_name: {"boundaries": [...], "metric": {...}}, "avg_f1": ..., ...}
    """
    if args.chunk_length_sec is None or args.chunk_length_sec <= 0:
        raise ValueError(
            f"chunk_length_sec must be > 0 for uniform chunking, got {args.chunk_length_sec}"
        )

    ts_output_dir = os.path.join(args.output_dir, "outputs_temporal_segmentation")
    os.makedirs(ts_output_dir, exist_ok=True)

    # Persist anno.json so stage 3 sees the same golden file as the DDM branch.
    anno_json_path = os.path.join(ts_output_dir, "anno.json")
    with open(anno_json_path, "w") as f:
        json.dump(anno_json, f, indent=2)

    golden_boundaries = extract_golden_boundaries(anno_json_path)
    with open(os.path.join(ts_output_dir, "video_to_boundaries_debug.json"), "w") as f:
        json.dump(golden_boundaries, f, indent=2)

    videos = sorted(glob.glob(os.path.join(args.video_dir, f"*.{args.video_ext}")))
    if not videos:
        raise FileNotFoundError(f"No .{args.video_ext} files found in {args.video_dir}")
    logging.info(
        "Uniform chunking: %d videos, chunk_length_sec=%.3f",
        len(videos), args.chunk_length_sec,
    )

    video_to_metric: dict = {}
    f1_list, precision_list, recall_list = [], [], []
    tp_list, fp_list, fn_list = [], [], []

    for video_path in videos:
        video_name = os.path.basename(video_path)
        duration = get_video_duration_sec(video_path)
        boundaries = uniform_chunk_boundaries(duration, args.chunk_length_sec)

        golden_bdy = golden_boundaries.get(video_name)
        metric = compute_temporal_metrics(golden_bdy, boundaries, duration)

        video_to_metric[video_name] = {
            "boundaries": boundaries,
            "duration_sec": duration,
            "metric": metric,
        }

        if metric["F1"] is not None:
            f1_list.append(metric["F1"])
            precision_list.append(metric["Precision"])
            recall_list.append(metric["Recall"])
            tp_list.append(metric["True Positive"])
            fp_list.append(metric["False Positive"])
            fn_list.append(metric["False Negative"])

    video_to_metric["avg_f1"] = float(np.mean(f1_list)) if f1_list else 0.0
    video_to_metric["avg_precision"] = float(np.mean(precision_list)) if precision_list else 0.0
    video_to_metric["avg_recall"] = float(np.mean(recall_list)) if recall_list else 0.0
    video_to_metric["avg_tp"] = float(np.mean(tp_list)) if tp_list else 0.0
    video_to_metric["avg_fp"] = float(np.mean(fp_list)) if fp_list else 0.0
    video_to_metric["avg_fn"] = float(np.mean(fn_list)) if fn_list else 0.0

    avg_f1 = video_to_metric["avg_f1"]
    result_json = os.path.join(ts_output_dir, f"f1_{avg_f1:.2f}.json")
    with open(result_json, "w") as f:
        json.dump(video_to_metric, f, indent=2)
    logging.info("Uniform stage complete. F1 results: %s", result_json)

    return video_to_metric


# =============================================================================
# VLM Stage
# =============================================================================

def run_vlm_stage(args, temporal_results: dict) -> dict:  # pragma: no cover
    """
    Run VLM action recognition on chunks defined by DDM boundaries.

    Returns:
        {video_name: {chunk_key: vlm_response}}
    """
    ar_output_dir = os.path.join(args.output_dir, "outputs_action_recognition")
    os.makedirs(ar_output_dir, exist_ok=True)

    prompt = read_txt(os.path.join(args.asset_root, "vlm_prompts.txt"))
    logging.info("VLM prompt loaded from %s", args.asset_root)

    # Default mirrors training config: max_frames=40, 16k vision tokens.
    # Eval-at-training-resolution keeps the VLM in-distribution.
    resolution_config = json.loads(args.resolution_config) if args.resolution_config else {"max_frames": 40, "total_pixels": 16572416}

    # Shim Args line for the sop-rca-plugin parser (SKILL.md Step 2 — `max_frames`).
    dump_action_recognition_args(args, ar_output_dir, resolution_config)

    video_name_to_output_text = defaultdict(dict)

    videos = sorted(glob.glob(os.path.join(args.video_dir, f"*.{args.video_ext}")))

    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor
        from qwen_vl_utils import process_vision_info
        from utils.eval_utils import build_vllm_video_mm_data
        import torch

        tp_size = getattr(args, "tensor_parallel_size", 0)
        if tp_size <= 0:
            tp_size = max(1, torch.cuda.device_count())
        # gpu_memory_utilization=0.7 (vs vLLM's 0.9 default): DDM stage
        # leaves ~50 GB reserved in PyTorch's caching allocator after
        # release; 0.9 OOMs, 0.7 leaves headroom for the residue.
        # disable_custom_all_reduce=True: the custom AR kernel rejects
        # the post-DDM CUDA context with "invalid argument"; NCCL fallback
        # is fine.
        logging.info(
            "vLLM tensor_parallel_size=%d (visible CUDA devices: %d), "
            "gpu_memory_utilization=0.7, disable_custom_all_reduce=True",
            tp_size, torch.cuda.device_count(),
        )
        llm = LLM(
            model=args.vlm_model_path,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=0.7,
            disable_custom_all_reduce=True,
        )
        processor = AutoProcessor.from_pretrained(args.vlm_model_path)
        sampling_params = SamplingParams(
            temperature=args.temperature, max_tokens=4096, top_p=args.top_p,
        )

        for video_path in videos:
            video_name = os.path.basename(video_path)
            if video_name not in temporal_results or "boundaries" not in temporal_results.get(video_name, {}):
                logging.warning("No temporal results for %s, skipping", video_name)
                continue

            boundaries = temporal_results[video_name]["boundaries"]
            chunk_starts = boundaries[:-1]
            chunk_ends = boundaries[1:]
            logging.info("VLM inference for %s: %d chunks", video_name, len(chunk_starts))

            for cs, ce in zip(chunk_starts, chunk_ends):
                messages = [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "video", "video": video_path,
                         "video_start": cs, "video_end": ce,
                         "fps": args.fps, **resolution_config},
                        {"type": "text", "text": prompt},
                    ]},
                ]

                prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                videos_with_metadata = build_vllm_video_mm_data(video_inputs, video_kwargs, args.fps)
                if videos_with_metadata is not None:
                    # vLLM parser expects the singular "video" key (see sop_eval.py).
                    mm_data["video"] = videos_with_metadata

                outputs = llm.generate(
                    [{"prompt": prompt_text, "multi_modal_data": mm_data}],
                    sampling_params, use_tqdm=False,
                )
                response = outputs[0].outputs[0].text

                key = f"[{cs:.2f}s-{ce:.2f}s]"
                logging.info("  %s: %s", key, response[:80])
                video_name_to_output_text[video_name][key] = response

    else:  # transformers backend
        # Auto-dispatch by `config.architectures[0]`: Qwen2_5_VL for CR1,
        # Qwen3VL for CR2. Hard-coding Qwen2_5 silently mis-initialised
        # CR2 weights and crashed in get_rope_index.
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from qwen_vl_utils import process_vision_info

        model = AutoModelForImageTextToText.from_pretrained(
            args.vlm_model_path, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.vlm_model_path)
        generation_config = {
            "max_new_tokens": 4096,
            "do_sample": args.temperature > 0,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }

        for video_path in videos:
            video_name = os.path.basename(video_path)
            if video_name not in temporal_results or "boundaries" not in temporal_results.get(video_name, {}):
                logging.warning("No temporal results for %s, skipping", video_name)
                continue

            boundaries = temporal_results[video_name]["boundaries"]
            chunk_starts = boundaries[:-1]
            chunk_ends = boundaries[1:]
            logging.info("VLM inference for %s: %d chunks", video_name, len(chunk_starts))

            for cs, ce in zip(chunk_starts, chunk_ends):
                messages = [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "video", "video": video_path,
                         "video_start": cs, "video_end": ce,
                         "fps": args.fps, **resolution_config},
                        {"type": "text", "text": prompt},
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

                key = f"[{cs:.2f}s-{ce:.2f}s]"
                logging.info("  %s: %s", key, response[:80])
                video_name_to_output_text[video_name][key] = response

    output_json = os.path.join(ar_output_dir, "video_name_to_output_text.json")
    with open(output_json, "w") as f:
        json.dump(dict(video_name_to_output_text), f, indent=2)
    logging.info("VLM stage complete. Results: %s", output_json)

    return dict(video_name_to_output_text)


# =============================================================================
# Main
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser. Split out so tests can introspect the args list."""
    parser = argparse.ArgumentParser(description="SOP end-to-end evaluation (DDM + VLM)")

    parser.add_argument("--vlm-model-path", type=str, required=True)
    parser.add_argument("--asset-root", type=str, required=True, help="Dir containing vlm_prompts.txt")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--video-dir", type=str, required=True, help="Dataset dir with full videos + annotation subdirs")
    parser.add_argument("--video-ext", type=str, default="mp4")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0, dest="top_p",
                        help="vLLM nucleus-sampling param. Irrelevant at temperature=0.")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "transformers"])
    parser.add_argument(
        "--resolution-config", type=str, default=None,
        help=(
            "JSON-encoded ResolutionConfig (see validation/request_validation.py). "
            'Fields: max_frames, total_pixels, resized_height, resized_width, '
            'max_pixels, min_pixels. Example: '
            '\'{"max_frames": 40, "total_pixels": 16572416}\'. '
            "Unset = mirror training defaults."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=0,
        help="vLLM tensor-parallel size. 0 = auto-detect from torch.cuda.device_count().",
    )

    parser.add_argument(
        "--chunking-algorithm", type=str, default="ddm",
        choices=["ddm", "uniform"],
        help="Stage-1 algorithm: 'ddm' (DDM-Net segmentation) or 'uniform' (fixed-length chunks)",
    )
    parser.add_argument(
        "--chunk-length-sec", type=float, default=None,
        help="Required when --chunking-algorithm=uniform. Length of each chunk in seconds.",
    )

    # DDM args — used only when --chunking-algorithm=ddm.
    parser.add_argument("--ddm-checkpoint-path", type=str, default=None)
    parser.add_argument("--ddm-resolution", type=int, default=224)
    parser.add_argument("--ddm-frames-per-side", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-sec", type=float, default=0.0)
    parser.add_argument("--ddm-batch-size", type=int, default=8)
    parser.add_argument("--frames-per-segment-hint", type=int, default=256)

    # Precomputed annotation JSON, written by the API endpoint.
    parser.add_argument("--anno-json-path", type=str, required=True)

    parser.add_argument("--actions-json-path", type=str, required=True)

    return parser


def _build_e2e_results(temporal_results: dict, accuracy_results: dict,
                       sequence_results: dict) -> dict:
    """
    Assemble the e2e_results.json payload. Field names here are the
    frontend's public contract — renaming a key here will silently break
    the EvaluationPanel results view.
    """
    return {
        "temporal_segmentation": {
            "avg_f1": temporal_results.get("avg_f1", 0.0),
            "avg_precision": temporal_results.get("avg_precision", 0.0),
            "avg_recall": temporal_results.get("avg_recall", 0.0),
            "per_video": {
                k: {
                    "f1": v.get("metric", {}).get("F1"),
                    "precision": v.get("metric", {}).get("Precision"),
                    "recall": v.get("metric", {}).get("Recall"),
                    "boundaries": v.get("boundaries"),
                }
                for k, v in temporal_results.items()
                if isinstance(v, dict) and "boundaries" in v
            },
        },
        "action_recognition": {
            # chunk-level — kept for backwards compat.
            "overall_accuracy": accuracy_results["overall_accuracy"],
            "per_action": accuracy_results["per_action"],
            # sequence-level — primary frontend display, matches reference accuracy.json.
            "sequence_accuracy": sequence_results["sequence_accuracy"],
            "action_accuracy": sequence_results["action_accuracy"],
            "total_videos": sequence_results["total_videos"],
            "total_videos_dist_0": sequence_results["total_videos_dist_0"],
            "total_actions": sequence_results["total_actions"],
            "wrong": sequence_results["wrong"],
            "duplicate": sequence_results["duplicate"],
            "missing": sequence_results["missing"],
            "videos_with_error": sequence_results["videos_with_error"],
            "per_video": sequence_results["per_video"],
        },
    }


def main(args) -> None:
    """
    Entry point invoked by the subprocess. Extracted from the
    `if __name__ == "__main__":` block so the dispatch + result-assembly
    logic is unit-testable (see TestMainDispatch).
    """
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(args.output_dir, "sop_e2e_eval_log.txt"),
                mode="w",
                encoding="utf-8",
            ),
        ],
        force=True,
    )

    logging.info("E2E evaluation starting with args: %s", args)

    with open(args.anno_json_path, "r") as f:
        anno_json = json.load(f)

    from utils.eval_utils import extract_mcq_data
    _, choices = extract_mcq_data(args.actions_json_path)

    # Stage 1: temporal segmentation. Uniform chunking is a drop-in
    # replacement (same output shape) for the DDM-Net branch.
    if args.chunking_algorithm == "uniform":
        if args.chunk_length_sec is None or args.chunk_length_sec <= 0:
            raise ValueError(
                "--chunk-length-sec must be > 0 when --chunking-algorithm=uniform"
            )
        logging.info(
            "=== Stage 1: Uniform Chunking (chunk_length_sec=%.3f) ===",
            args.chunk_length_sec,
        )
        temporal_results = run_uniform_stage(args, anno_json)
    else:
        if not args.ddm_checkpoint_path:
            raise ValueError(
                "--ddm-checkpoint-path is required when --chunking-algorithm=ddm"
            )
        logging.info("=== Stage 1: DDM Temporal Segmentation ===")
        temporal_results = run_ddm_stage(args, anno_json)

    logging.info("=== Stage 2: VLM Action Recognition ===")
    vlm_outputs = run_vlm_stage(args, temporal_results)

    logging.info("=== Stage 3: Accuracy Computation ===")

    anno_json_path = os.path.join(args.output_dir, "outputs_temporal_segmentation", "anno.json")
    golden_boundaries = extract_golden_boundaries(anno_json_path)

    chunk_action_map = {}
    for video_name, video_result in temporal_results.items():
        if not isinstance(video_result, dict) or "boundaries" not in video_result:
            continue
        golden_bdy = golden_boundaries.get(video_name)
        if golden_bdy is None:
            continue
        pred_boundaries = video_result["boundaries"]
        action_count = len(golden_bdy) - 1  # N boundaries → N-1 actions
        chunk_action_map[video_name] = map_chunks_to_ground_truth(
            pred_boundaries, golden_bdy, action_count
        )

    accuracy_results = compute_e2e_accuracy(vlm_outputs, chunk_action_map, choices)

    # Sequence-level evaluation via Levenshtein edit distance: classifies each
    # error as Wrong/Duplicate/Missing — the SOP-compliance metrics.
    from utils.e2e_eval_utils import evaluate_action_sequences
    pred_json_path = os.path.join(
        args.output_dir, "outputs_action_recognition", "video_name_to_output_text.json"
    )
    sequence_results = evaluate_action_sequences(
        anno_json_path=anno_json_path,
        pred_json_path=pred_json_path,
        actions_json_path=args.actions_json_path,
    )

    # Persist a stand-alone accuracy.json matching the reference layout.
    accuracy_json_path = os.path.join(
        args.output_dir, "outputs_action_recognition", "accuracy.json"
    )
    os.makedirs(os.path.dirname(accuracy_json_path), exist_ok=True)
    with open(accuracy_json_path, "w") as f:
        json.dump(sequence_results, f, indent=2)
    logging.info("Sequence-level accuracy report: %s", accuracy_json_path)

    e2e_results = _build_e2e_results(temporal_results, accuracy_results, sequence_results)

    e2e_results_path = os.path.join(args.output_dir, "e2e_results.json")
    with open(e2e_results_path, "w") as f:
        json.dump(e2e_results, f, indent=2)
    logging.info("E2E evaluation complete. Results: %s", e2e_results_path)
    logging.info(
        "Temporal F1=%.4f | Sequence accuracy=%.4f (%d/%d videos) | "
        "Action accuracy=%.4f (wrong=%d, duplicate=%d, missing=%d / %d total)",
        temporal_results.get("avg_f1", 0.0),
        sequence_results["sequence_accuracy"],
        sequence_results["total_videos_dist_0"],
        sequence_results["total_videos"],
        sequence_results["action_accuracy"],
        sequence_results["wrong"],
        sequence_results["duplicate"],
        sequence_results["missing"],
        sequence_results["total_actions"],
    )


if __name__ == "__main__":  # pragma: no cover
    main(build_arg_parser().parse_args())
