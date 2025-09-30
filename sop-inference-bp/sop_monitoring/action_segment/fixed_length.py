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
Fixed-length video chunking module.

This module provides functionality to split videos into chunks of fixed duration.
"""

import logging

from torchcodec.decoders import VideoDecoder

_LOGGER = logging.getLogger(__name__)

def fixed_length_split_start_end_time(input_video_path: str, chunk_duration: float) -> tuple[list[float], list[float]]:
    """
    Split a video into chunks of fixed duration and return the start and end times of each chunk in seconds.

    Args:
        input_video_path: Path to the input video file
        chunk_duration: Duration of each chunk in seconds

    Returns:
        Tuple of (start_times, end_times) where each list contains the start/end times
        in seconds for each chunk

    Raises:
        ValueError: If the video file cannot be opened or read, or if chunk_duration is invalid
    """
    _LOGGER.debug("Reading video metadata: %s", input_video_path)

    try:
        decoder = VideoDecoder(input_video_path)
        metadata = decoder.metadata
        fps = metadata.average_fps
        duration_sec = metadata.duration_seconds
    except Exception as e:
        raise ValueError(f"Cannot read video metadata from {input_video_path}: {str(e)}")

    _LOGGER.info("Video metadata: duration=%.2fs, fps=%.2f", duration_sec, fps)

    if chunk_duration > duration_sec:
        _LOGGER.warning("Chunk duration (%.2fs) is longer than video duration (%.2fs). "
                       "Will create a single chunk covering the entire video.",
                       chunk_duration, duration_sec)

    start_times = []
    end_times = []

    current_time = 0.0
    chunk_index = 0

    while current_time < duration_sec:
        start_time = current_time
        end_time = min(current_time + chunk_duration, duration_sec)

        start_times.append(start_time)
        end_times.append(end_time)

        _LOGGER.debug("Chunk %d: %.2fs - %.2fs (duration: %.2fs)",
                     chunk_index, start_time, end_time, end_time - start_time)

        current_time += chunk_duration
        chunk_index += 1

    if not start_times:
        _LOGGER.warning("No chunks created for video %s", input_video_path)
        return [], []

    _LOGGER.info("Created %d time segments with duration %d seconds each",
                len(start_times), chunk_duration)

    return start_times, end_times
