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
Single-process DDM-Net inference for e2e evaluation.

Replaces ``sop_monitoring.action_segment.ddm_net.MultiGpuDdmNet`` with a
streamlined single-GPU implementation that uses DDM-Net's own ``resnetGEBD``
model code (already vendored at ``$DDM_BASE_PATH/DDM-Net``) and PyAV for
video decoding.

Functional parity with the upstream inference pipeline's reference
``temporal_segmentation`` script:

  * Same model loading flow (``resnetGEBD`` + checkpoint state_dict cleanup
    that strips ``model.`` / ``module.`` prefixes).
  * Same sliding-window inference: ``window_size = 2 * frames_per_side + 1``.
  * Same per-window score: ``F.softmax(output, dim=1)[:, 1]``, where
    ``output`` is the last element of the first item in the model's
    ``(outputs, _, _)`` return tuple.
  * Same end-padding: ``[0.0] * frames_per_side`` prepended and appended
    so that ``len(scores) == total_frames``.
  * Same ``(per_frame_scores, video_metadata)`` return shape.

Differences vs ``MultiGpuDdmNet``:

  * Single-process, single-GPU. The multi-process worker pool is dropped
    because (a) the user's cross-NUMA Blackwell host has NCCL P2P
    fragility that breaks multi-GPU collective ops, and (b) for SOP-scale
    eval (a handful of short videos), single-GPU comfortably finishes
    in under a minute per video.
  * PyAV ('av' package) for decode instead of ``torchcodec``. torchcodec
    is not installed in the eval_ms image (was deliberately excluded for
    the cosmos-rl stack).
  * Whole-video decode + windowed forward pass. ``frames_per_segment_hint``
    is accepted for API compatibility but unused — full-video decode fits
    comfortably in 96 GB VRAM for SOP-scale clips.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Heavy deps (torch, torchvision, av) are imported lazily so the pure-Python
# boundary helpers stay importable in test environments without GPU/PyAV.

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Public types
# ----------------------------------------------------------------------

@dataclass
class VideoMetadata:
    """Subset of fields downstream code reads from sop_monitoring's VideoMetaData."""
    fps: float
    duration_sec: float
    total_frames: int


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

def load_ddm_model(checkpoint_path: str, frames_per_side: int, device: str):  # pragma: no cover
    """
    Load ``resnetGEBD`` + checkpoint onto ``device``. Returns model in eval mode.

    Mirrors the upstream ``_load_model`` private helper in
    ``sop_monitoring.action_segment.ddm_net``.
    """
    import argparse as _argparse
    import torch

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"DDM checkpoint not found: {checkpoint_path}")

    ddm_base_path = os.environ.get("DDM_BASE_PATH", "/workspace/ddm")
    ddm_net_path = os.path.join(ddm_base_path, "DDM-Net")
    if ddm_net_path not in sys.path:
        sys.path.insert(0, ddm_net_path)

    from modeling.resnetGEBD import resnetGEBD  # noqa: E402  (path-dependent import)

    model = resnetGEBD(
        backbone="resnet50",
        pretrained=False,
        num_classes=2,
        frames_per_side=frames_per_side,
    )

    with torch.serialization.safe_globals([_argparse.Namespace]):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw = ckpt["state_dict"]
        cleaned = {}
        for k, v in raw.items():
            if k.startswith("model."):
                cleaned[k[len("model."):]] = v
            elif k.startswith("module."):
                cleaned[k[len("module."):]] = v
            else:
                cleaned[k] = v
    else:
        cleaned = ckpt

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning("Missing keys in DDM checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys in DDM checkpoint: %s", unexpected)

    model = model.to(device).eval()
    logger.info("DDM-Net loaded on %s (frames_per_side=%d)", device, frames_per_side)
    return model


# ----------------------------------------------------------------------
# Video decode (PyAV)
# ----------------------------------------------------------------------

def _decode_video_pyav(  # pragma: no cover
    video_path: str,
    target_resolution: Optional[int] = None,
    end_timestamp_sec: Optional[float] = None,
):
    """
    Decode an entire video to a uint8 ``[T, C, H, W]`` CPU tensor.

    If ``target_resolution`` is set, each frame is resized in PyAV's swscaler
    to ``(target_resolution, target_resolution)`` before being collected.
    Resizing at decode time keeps the resulting GPU tensor small (the model
    only needs 224x224 input anyway), which avoids OOM on long native-resolution
    videos: e.g. a 4000-frame 1920x1080 clip is ~92 GiB as float32 at native
    resolution, but only ~2 GiB at 224x224. Bilinear scaling in libswscale is
    functionally equivalent to ``T.Resize((R, R), antialias=True)`` for this
    pipeline (both are non-aspect-preserving, the model expects square inputs).

    If ``end_timestamp_sec`` is set, the decode loop stops as soon as it sees
    a frame whose presentation timestamp exceeds the cap. Used by
    ``run_ddm_stage`` to stop DDM from scoring unannotated tail footage that
    would otherwise produce a phantom tail boundary and inflate the duplicate
    count in the action-sequence comparison. When the cap is ``None`` (default)
    or the video naturally ends before the cap, every frame is decoded — i.e.
    behavior is identical to the pre-cap implementation. Frames whose PTS or
    the stream time_base is unavailable fall through without the cap (decode
    everything), so corrupt or unusual mp4s never error out due to this path.

    Returns the tensor + VideoMetadata. Raises if the video can't be opened
    or contains no decodable frames.
    """
    import av
    import torch

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        time_base = stream.time_base  # may be None on pathological streams
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        declared_nb_frames = stream.frames or 0  # 0 for many MP4s; we'll count

        frames = []
        for frame in container.decode(video=0):
            # Early-stop at the annotation tail when caller supplied a cap.
            # Guarded against frame.pts is None and stream.time_base is None
            # so corrupt/unusual mp4s decode normally instead of erroring.
            if (
                end_timestamp_sec is not None
                and frame.pts is not None
                and time_base is not None
                and float(frame.pts * time_base) > end_timestamp_sec
            ):
                break
            if target_resolution is not None:
                frame = frame.reformat(
                    width=target_resolution,
                    height=target_resolution,
                    format="rgb24",
                )
                frames.append(frame.to_ndarray())  # already rgb24 [R, R, 3]
            else:
                frames.append(frame.to_ndarray(format="rgb24"))  # [H, W, 3] uint8
    finally:
        container.close()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    if end_timestamp_sec is not None:
        # When a cap was set, ``declared_nb_frames`` reflects the full mp4
        # stream count — which includes frames past the cap that we
        # deliberately did NOT decode. Use the actual decoded count so
        # downstream sizing (window count, padding, chunk boundaries) lines
        # up with the tensor we return.
        total_frames = len(frames)
    else:
        total_frames = declared_nb_frames if declared_nb_frames > 0 else len(frames)
    duration_sec = (total_frames / fps) if fps > 0 else 0.0

    arr = np.stack(frames, axis=0)  # [T, H, W, C]
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]

    return tensor, VideoMetadata(fps=fps, duration_sec=duration_sec, total_frames=total_frames)


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------

def _model_window_scores(model, batch) -> np.ndarray:  # pragma: no cover
    """
    Run one forward pass and return softmax(output, dim=1)[:, 1] as numpy 1D.

    Matches the upstream destructuring:
        outputs, _, _ = model(batch_tensor)
        if isinstance(outputs, (list, tuple)):
            output = outputs[-1]
    """
    import torch.nn.functional as F

    result = model(batch)
    if isinstance(result, tuple) and len(result) >= 1:
        outputs = result[0]
    else:
        outputs = result

    if isinstance(outputs, (list, tuple)):
        output = outputs[-1]
    else:
        output = outputs

    return F.softmax(output, dim=1)[:, 1].detach().float().cpu().numpy()


def run_ddm_inference(
    model,
    video_path: str,
    *,
    resolution: int,
    frames_per_side: int,
    batch_size: int,
    device: str,
    frames_per_segment_hint: Optional[int] = None,  # accepted for API parity, unused
    end_timestamp_sec: Optional[float] = None,
) -> tuple[list[float], VideoMetadata]:
    """
    Run DDM-Net inference on a single video.

    Parameters
    ----------
    end_timestamp_sec : Optional[float]
        If set, decode stops at the first frame whose PTS exceeds this cap
        (see ``_decode_video_pyav``). Used by ``run_ddm_stage`` to clip
        unannotated tail footage so DDM doesn't emit a phantom tail
        boundary. ``None`` (default) decodes the entire video.

    Returns
    -------
    scores : list[float]
        Per-frame boundary scores; ``len(scores) == metadata.total_frames``.
        Frames within ``frames_per_side`` of either end are padded with 0.0.
    metadata : VideoMetadata
    """
    import torch
    import torchvision.transforms.v2 as T

    logger.info("Running DDM inference on %s", video_path)

    # Resize at decode time (libswscale): torchvision's Resize upcasts to
    # float32 first, which OOMs on long native-resolution videos.
    vframes_cpu, meta = _decode_video_pyav(
        video_path,
        target_resolution=resolution,
        end_timestamp_sec=end_timestamp_sec,
    )
    total_frames = meta.total_frames

    transform = T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    window_size = 2 * frames_per_side + 1

    if total_frames < window_size:
        logger.warning(
            "Video has %d frames but window_size=%d; cannot form any windows. Returning all zeros.",
            total_frames, window_size,
        )
        return [0.0] * total_frames, meta

    with torch.no_grad():
        vframes_gpu = vframes_cpu.to(device, non_blocking=True)
        del vframes_cpu
        preproc = transform(vframes_gpu)  # [T, C, R, R] float32, normalized
        del vframes_gpu
        if "cuda" in str(device) and torch.cuda.is_available():
            torch.cuda.empty_cache()

        window_scores: list[float] = []
        batch_windows: list[torch.Tensor] = []

        for i in range(total_frames - window_size + 1):
            batch_windows.append(preproc[i:i + window_size])
            if len(batch_windows) == batch_size:
                bt = torch.stack(batch_windows, dim=0)  # [B, T, C, H, W]
                window_scores.extend(_model_window_scores(model, bt).tolist())
                batch_windows = []

        # Final partial batch: only run if >= 2 windows (matches upstream).
        if len(batch_windows) >= 2:
            bt = torch.stack(batch_windows, dim=0)
            window_scores.extend(_model_window_scores(model, bt).tolist())
        elif len(batch_windows) > 0:
            logger.debug(
                "Skipping final batch of %d window(s) (< 2; matches upstream).",
                len(batch_windows),
            )

    # Zero-pad the edges where frames can't be window-centred.
    pad_start = frames_per_side
    pad_end = total_frames - len(window_scores) - pad_start
    padded = [0.0] * pad_start + [float(s) for s in window_scores] + [0.0] * max(0, pad_end)

    # Defensive trim/pad against off-by-one from container metadata.
    if len(padded) > total_frames:
        padded = padded[:total_frames]
    elif len(padded) < total_frames:
        padded = padded + [0.0] * (total_frames - len(padded))

    logger.info(
        "DDM inference done: %d windows scored, %d total frames",
        len(window_scores), total_frames,
    )
    return padded, meta


# ----------------------------------------------------------------------
# Score → boundary helpers (verbatim port from sop_monitoring)
# ----------------------------------------------------------------------

def detect_boundaries(scores: list[float], threshold: float, nms_size: int) -> list[int]:
    """
    Detect event boundaries: frames where score > threshold and the score
    is the local max within a [-nms_size, +nms_size] window.

    Verbatim port of ``sop_monitoring.action_segment.ddm_net.detect_boundaries``.
    """
    np_scores = np.array(scores)
    boundaries: list[int] = []

    for i, score in enumerate(scores):
        if score > threshold:
            left = max(0, i - nms_size)
            right_plus_1 = min(len(scores), i + nms_size + 1)
            is_local_max = (np.argmax(np_scores[left:right_plus_1]) + left) == i
            if is_local_max:
                boundaries.append(i)

    return boundaries


def calculate_chunk_boundaries(
    boundaries: list[int],
    fps: float,
    duration_sec: float,
    total_frames: int,
) -> tuple[list[float], list[float]]:
    """
    Convert detected boundary frame indices into chunk start/end seconds.

    Verbatim port of ``sop_monitoring.action_segment.ddm_net.calculate_chunk_boundaries``.
    Argument ``total_frames`` is accepted for API parity though it is not
    used by the upstream implementation either.
    """
    _ = total_frames  # API parity
    boundaries_in_sec = [b / fps for b in boundaries]
    chunk_start_seconds = [0.0] + boundaries_in_sec
    chunk_end_seconds = boundaries_in_sec + [duration_sec]
    return chunk_start_seconds, chunk_end_seconds
