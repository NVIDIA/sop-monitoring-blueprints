#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
A script to trigger DDM for temporal segmentation.
"""

import argparse
import os
import logging
import json
import numpy as np
import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt

from sop_monitoring.action_segment.ddm_net import MultiGpuDdmNet, detect_boundaries

from sop_scriptlib.utils import setup_logging

_LOGGER = logging.getLogger(__name__)

plt.switch_backend('agg')

_ALGO_DDM_NET = "ddm-net"

def process_one_video(video_name: str,
                      score_threshold: float,
                      nms_sec: float,
                      ddm_info: dict,
                      golden_boundaries: list[float] | None,
                      output_dir: str) -> tuple[str, dict]:

    print(f"video_name: {video_name}, score_threshold: {score_threshold}, "
          f"nms_sec: {nms_sec}, output_dir: {output_dir}")

    ret = {}

    scores, fps = ddm_info["scores"], ddm_info["fps"]
    nms_size = int(nms_sec * fps)
    boundaries = detect_boundaries(scores, score_threshold, nms_size)
    # usually we also consider the starting and the ending are boundaries.
    boundaries = [0.0] + [bdy / fps for bdy in boundaries] + [ddm_info["duration_sec"]]
    ret["boundaries"] = boundaries
    ret["ddm_threshold"] = ddm_info["duration_sec"] * 0.025
    metric = Metric(golden_boundaries, boundaries, ret["ddm_threshold"])
    ret["metric"] = metric.to_dict()

    png_name = os.path.join(output_dir, f"{video_name}.png")
    visualize_scores_with_boundaries(video_name, golden_boundaries, boundaries, scores, fps, png_name)
    return video_name, ret


def visualize_scores_with_boundaries(video_name: str, golden_bdy: list[float] | None, pred_bdy: list[float],
                                     scores: list[float], fps: float, output_path: str):

    if not scores:
        print("No scores to visualize")
        return

    frame_times = [i / fps for i in range(len(scores))]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot scores as area plot
    ax.fill_between(frame_times, scores, alpha=0.6, color='skyblue', label='Scores')
    ax.plot(frame_times, scores, color='blue', linewidth=1.5, alpha=0.8)

    # Draw vertical lines for golden boundaries
    if golden_bdy:
        for i, bdy in enumerate(golden_bdy):
            if 0 <= bdy <= max(frame_times):
                ax.axvline(x=bdy, color='red', linewidth=2, linestyle='--', alpha=0.8,
                          label='Golden Boundaries' if i == 0 else "")

                # Add boundary index label (starting from 1)
                ax.text(bdy, max(scores) * 0.9, str(i + 1),
                       ha='center', va='bottom', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))

    # Draw vertical lines for predicted boundaries
    for i, bdy in enumerate(pred_bdy):
        if 0 <= bdy <= max(frame_times):
            ax.axvline(x=bdy, color='green', linewidth=2, linestyle='-', alpha=0.7,
                      label='Predicted Boundaries' if i == 0 else "")

            # Add boundary index label (starting from 1)
            ax.text(bdy, max(scores) * 0.8, str(i + 1),
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', edgecolor='green', alpha=0.9))

    # Set axis properties
    ax.set_xlim(0, max(frame_times) if frame_times else 1)
    ax.set_ylim(0, max(scores) * 1.1 if scores else 1)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc='upper right', fontsize=10)

    # Add title with video info
    ax.set_title(f'{video_name}\nScores over Time with Golden and Predicted Boundaries',
                fontsize=14, fontweight='bold')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory


class Metric:
    def __init__(self, golden_bdy: list[float] | None, pred_bdy: list[float], threshold: float):
        self.golden_bdy = golden_bdy
        self.pred_bdy = pred_bdy
        self.threshold = threshold

        self._calc_f1()

    def to_dict(self):
        return {
            "True Positive": self.tp,
            "False Positive": self.fp,
            "False Negative": self.fn,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1": self.f1,
        }

    @property
    def tp(self):
        return self._tp


    @property
    def fp(self):
        return self._fp

    @property
    def fn(self):
        return self._fn

    @property
    def precision(self):
        return self._precision

    @property
    def recall(self):
        return self._recall

    @property
    def f1(self):
        return self._f1

    def _calc_f1(self):

        if self.golden_bdy is None:
            tp = None
            fp = None
            fn = None
            precision = None
            recall = None
            f1 = None
        else:
            num_golden = len(self.golden_bdy)
            num_pred = len(self.pred_bdy)
            tp = 0
            used_pred = set()

            for golden in self.golden_bdy:
                for pred in self.pred_bdy:
                    if abs(golden - pred) <= self.threshold and pred not in used_pred:
                        tp += 1
                        used_pred.add(pred)
                        break

            fp = num_pred - tp
            fn = num_golden - tp
            assert fp >=0 and fn >=0, f"Something wrong. We should not get negative values, but fp: {fp}, fn: {fn}"
            precision = tp / num_pred
            recall = tp / num_golden
            f1 = 2 * precision * recall / (precision + recall)

        self._tp = tp
        self._fp = fp
        self._fn = fn
        self._precision = precision
        self._recall = recall
        self._f1 = f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to trigger DDM for temporal segmentation. "
                                                 "Note that this script should be executed in the image tagged as action_segment")

    # interfaces for data
    parser.add_argument("--video",
                        type=str,
                        required=False,
                        default="",
                        help="Path to the video file to process. Exclusive with video_dir.")

    parser.add_argument("--video_dir",
                        type=str,
                        required=False,
                        default="",
                        help="Path to the directory containing video files. Exclusive with video_dir.")

    parser.add_argument("--video_ext",
                        type=str,
                        required=False,
                        default="mp4",
                        help="The expected video extension in video_dir. "
                             "Files with this ext in video_dir is considered as input. Default mp4")

    parser.add_argument("--anno_json",
                        type=str,
                        required=True,
                        help="Path to a json containing annotation for each video. "
                             "The format should be "
                             "{ <video_filename_0>: [ {'description': 'event0', 'start_timestamp': 0.0, 'end_timestamp': 2.3}, ...], "
                             "  <video_filename_1>: [ {'description': 'event0', 'start_timestamp': 0.0, 'end_timestamp': 4.5}, ...], "
                             "  ...}"
                        )
    # required ddm-net arguments
    parser.add_argument("--checkpoint",
                        type=str,
                        required=False,
                        default="",
                        help="Checkpoint path to DDM-Net. Required if ddm-net is chosen.")
    parser.add_argument("--resolution",
                        type=int,
                        required=False,
                        default=0,
                        help="Resolution of the input image to the  model. It should be same as the one used in DDM finetuning.")
    ## optional ddm arguments
    parser.add_argument("--nms_sec",
                        type=float,
                        default=0.0,
                        required=False,
                        help="The hafl length of the window to perform non-maximun suppression. "
                             "The default value is roughly 0.025 * video length in seconds")
    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        required=False,
                        help="Batch size of the DDM-Net model. Default 8.")
    parser.add_argument("--frames_per_segment_hint",
                        type=int,
                        default=256,
                        required=False,
                        help="Frames per segment hint of the DDM-Net model. Default 256.")

    # optional
    parser.add_argument("--output_dir",
                        type=str,
                        required=False,
                        default="outputs_temporal_segmentation",
                        help="Path to the output directory. Default is outputs_temporal_segmentation")

    args = parser.parse_args()

    video_path = args.video
    video_dir = args.video_dir
    video_ext = args.video_ext
    anno_json = args.anno_json
    checkpoint = args.checkpoint
    resolution = args.resolution
    nms_sec = args.nms_sec
    batch_size = args.batch_size
    frames_per_segment_hint = args.frames_per_segment_hint
    output_dir = args.output_dir

    # sanity check arguments
    if video_path == "" and video_dir == "":
        raise ValueError("One of the arguments '--video' or '--video_dir' should be specified")
    if video_path != "" and video_dir != "":
        raise ValueError("Only one of the arguments '--video' or '--video_dir' should be specified, "
                         f"but got --video={video_path} and --video_dir={video_dir}.")

    error_msg = ""
    if not os.path.isfile(checkpoint):
        error_msg += f"checkpoint {checkpoint} not found.\n"
    if resolution == 0:
        error_msg += f"resolution is required and it should be same as the one used in finetuning DDM.\n"
    if error_msg:
        raise ValueError(error_msg)

    # Create output directory first
    os.makedirs(output_dir, exist_ok=True)
    log_filename = os.path.join(output_dir, "temporal_segmentation.log")
    setup_logging(log_filename, logging.DEBUG)
    _LOGGER.info(f"Args: {args}")

    videos = []
    if video_path:
        videos = [video_path]
    elif video_dir:
        for entry in os.scandir(video_dir):
            if entry.is_file() and entry.name.endswith(f".{video_ext}"):
                videos.append(entry.path)
    else:
        raise ValueError("Both --video and --video_dir are empty.")

    with open(anno_json) as fp:
        video_to_anno = json.load(fp)

    video_to_boundaries = {}
    for video, anno in video_to_anno.items():
        video_to_boundaries[video] = [0.0]

        for event1, event2 in zip(anno[:-1], anno[1:]):
            s_time = event1["end_timestamp"]
            e_time = event2["start_timestamp"]
            video_to_boundaries[video].append((s_time + e_time) / 2)
        video_to_boundaries[video].append(anno[-1]["end_timestamp"])

    with open(os.path.join(output_dir, "video_to_boundaries_debug.json"), "w") as f:
        json.dump(video_to_boundaries, f, indent=2)

    segmenter = MultiGpuDdmNet(checkpoint_path=checkpoint,
                               resolution=resolution,
                               frames_per_side=5,
                               frames_per_segment_hint=frames_per_segment_hint)

    video_to_ddm_info = {}
    for video_path in videos:
        video_name = os.path.basename(video_path)
        _LOGGER.info("Inferencing video: %s", video_name)
        video_to_ddm_info[video_name] = {}
        video_to_ddm_info[video_name]["scores"], video_metadata = segmenter.get_ddm_scores(video_path, batch_size)
        video_to_ddm_info[video_name]["fps"] = video_metadata.fps
        video_to_ddm_info[video_name]["duration_sec"] = video_metadata.duration_sec

    segmenter.shutdown()

    with open(os.path.join(args.output_dir, "video_to_dmm_info_debug.json"), "w") as f:
        json.dump(video_to_ddm_info, f, indent=2)

    video_names = list(video_to_ddm_info.keys())
    score_threshold_list = [0.5] * len(video_names)
    if nms_sec == 0.0:
        nms_sec_list = [0.025 * video_to_ddm_info[video_name]["duration_sec"]
                         for video_name in video_names]
    else:
        nms_sec_list = [nms_sec] * len(video_names)
    ddm_info_list = [video_to_ddm_info[video_name] for video_name in video_names]
    golden_boundaries_list = [video_to_boundaries.get(video_name, None) for video_name in video_names]
    output_dir_list = [output_dir] * len(video_names)

    _LOGGER.info("Post-processing results for %d videos...", len(video_names))
    with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), len(videos))) as executor:
        results = executor.map(process_one_video,
                               video_names,
                               score_threshold_list,
                               nms_sec_list,
                               ddm_info_list,
                               golden_boundaries_list,
                               output_dir_list)

    f1_list = []
    precision_list = []
    recall_list = []
    tp_list = []
    fp_list = []
    fn_list = []
    video_to_metric = {}

    for video_name, ret in results:
        f1_score = ret["metric"]["F1"]
        if f1_score is None:
            pass
        else:
            f1_list.append(f1_score)
            precision_list.append(ret["metric"]["Precision"])
            recall_list.append(ret["metric"]["Recall"])
            tp_list.append(ret["metric"]["True Positive"])
            fp_list.append(ret["metric"]["False Positive"])
            fn_list.append(ret["metric"]["False Negative"])
        video_to_metric[video_name] = ret

    video_to_metric["avg_f1"] = float(np.mean(f1_list))
    video_to_metric["avg_precision"] = float(np.mean(precision_list))
    video_to_metric["avg_recall"] = float(np.mean(recall_list))
    video_to_metric["avg_tp"] = float(np.mean(tp_list))
    video_to_metric["avg_fp"] = float(np.mean(fp_list))
    video_to_metric["avg_fn"] = float(np.mean(fn_list))

    result_json = os.path.join(output_dir, f"f1_{video_to_metric['avg_f1']:.2f}.json")
    _LOGGER.info("Writing out results to %s", result_json)
    with open(result_json, "w") as f:
        json.dump(video_to_metric, f, indent=2)
