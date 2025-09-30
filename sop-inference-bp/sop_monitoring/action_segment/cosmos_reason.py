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
Use Cosmos-Reason to perform action segmentation.
"""

import os
import re
import logging

from dataclasses import dataclass

from .fixed_length import fixed_length_split_start_end_time
from ..multi_gpu_utils import (
    get_gpu_ids,
    MultiWorkerManager,
)
from ..vlm import CosmosReason1

_LOGGER = logging.getLogger(__name__)

@dataclass
class _CrRequest:
    video_filename: str
    chunk_start_second: float
    chunk_end_second: float
    prompt: str
    system_prompt: str

@dataclass
class _CrResponse:
    result: str

class _CosmosReasonWorker(MultiWorkerManager.Worker):
    def __init__(self, model_path: str, gpu_id: int):
        self._device = f"cuda:{gpu_id}"
        self._model_path = model_path
        self._vlm = None
        self._re_timetoken = None

    def get_name(self) -> str:
        return f"cosmos_reason_worker_{self._device}"

    def initialize(self) -> None:
        self._vlm = CosmosReason1(self._model_path, device=self._device)
        self._re_timetoken = re.compile(r"<(\d+\.\d+)>")

    def process_request(self, request: _CrRequest) -> _CrResponse:

        result = self._vlm.chunk_and_infer(
            request.prompt,
            request.video_filename,
            request.chunk_start_second,
            request.chunk_end_second,
            system_prompt=request.system_prompt,
            timestamp=True
        )

        result = self._add_offset_to_timestamp(result, request.chunk_start_second, request.chunk_end_second)

        return _CrResponse(result=result)

    def _add_offset_to_timestamp(self, response: str, offset: float, max_timestamp: float) -> str:
        def replace_timestamp(match):
            timestamp = float(match.group(1))
            new_timestamp = min(timestamp + offset, max_timestamp)
            return f"<{new_timestamp:.2f}>"
        return self._re_timetoken.sub(replace_timestamp, response)


class CosmosReasonActionSegmenter:

    def __init__(self, model_path: str):
        self._gpu_ids = get_gpu_ids()
        self._workers: list[_CosmosReasonWorker] = []
        for gpu_id in self._gpu_ids:
            self._workers.append(_CosmosReasonWorker(model_path, gpu_id))
        self._multi_worker_manager = MultiWorkerManager(self._workers)

        self._re_start_end_time = re.compile(r"<(\d+\.\d+)>\s<(\d+\.\d+)>")

        try:
            self._time_out_sec = int(os.getenv("ACTION_SEGMENT_CR_TIMEOUT_SEC", 900))
        except ValueError:
            _LOGGER.error("Invalid value for ACTION_SEGMENT_CR_TIMEOUT_SEC: %s. Using default value of 900 seconds.",
                          os.getenv("ACTION_SEGMENT_CR_TIMEOUT_SEC"))
            self._time_out_sec = 900

    def process_video(self,
                      video_filename: str,
                      prompt: str,
                      system_prompt: str,
                      chunk_duration_sec: float,
                      min_length_sec: float) -> tuple[list[float], list[float]]:
        """
        Process the video and return the start and end times of the chunks.

        Returns:
            tuple[list[float], list[float], list[str]]:
            The start and end times of the chunks and the response from the model.
        """

        start_times, end_times = fixed_length_split_start_end_time(video_filename, chunk_duration_sec)

        futures = []
        for start_time, end_time in zip(start_times, end_times):
            cr_request = _CrRequest(
                video_filename=video_filename,
                chunk_start_second=start_time,
                chunk_end_second=end_time,
                prompt=prompt,
                system_prompt=system_prompt
            )
            future = self._multi_worker_manager.submit_request(cr_request)
            futures.append(future)

        chunk_start_times = []
        chunk_end_times = []
        chunk_responses = []
        for idx, future in enumerate(futures):
            try:
                response: _CrResponse = future.result(timeout=self._time_out_sec)
            except Exception as e:
                _LOGGER.error("Error processing request: %s", e)
                continue

            chunk_responses.append(response.result)

            matches = self._re_start_end_time.findall(response.result)
            if not matches:
                _LOGGER.warning("No chunks were detected for the response %s. Outputting the whole [%.2f, %.2f] seconds.", response.result, start_times[idx], end_times[idx])
                chunk_start_times.append(start_times[idx])
                chunk_end_times.append(end_times[idx])
                continue

            for start_time, end_time in matches:
                start_time = float(start_time)
                end_time = float(end_time)
                chunk_start_times.append(start_time)
                chunk_end_times.append(end_time)

        chunk_start_times, chunk_end_times = self._merge_short_chunks(chunk_start_times, chunk_end_times, min_length_sec)

        _LOGGER.debug("CosmosReason chunked start-end times:")
        for idx, (start, end) in enumerate(zip(chunk_start_times, chunk_end_times)):
            _LOGGER.debug("  Chunk %d: %.2f - %.2f (%.2f sec)", idx, start, end, end - start)

        return chunk_start_times, chunk_end_times, chunk_responses

    def shutdown(self) -> None:
        self._multi_worker_manager.shutdown()

    def _merge_short_chunks(self, chunk_start_times: list[float], chunk_end_times: list[float], min_length_sec: float) -> tuple[list[float], list[float]]:
        """
        Merge chunks that are shorter than min_length_sec with adjacent chunks.

        Args:
            chunk_start_times: List of chunk start times
            chunk_end_times: List of chunk end times
            min_length_sec: Minimum duration for chunks in seconds

        Returns:
            Tuple of merged start times and end times
        """
        if not chunk_start_times or not chunk_end_times:
            return chunk_start_times, chunk_end_times

        if len(chunk_start_times) != len(chunk_end_times):
            _LOGGER.warning("Mismatched start and end times lengths")
            return chunk_start_times, chunk_end_times

        chunks = list(zip(chunk_start_times, chunk_end_times))

        # Sort by start time to ensure proper ordering
        chunks.sort(key=lambda x: x[0])

        merged_chunks = []
        i = 0

        while i < len(chunks):
            current_start, current_end = chunks[i]
            current_duration = current_end - current_start

            # If current chunk is long enough, add it as is
            if current_duration >= min_length_sec:
                merged_chunks.append((current_start, current_end))
                i += 1
            else:
                # Try to merge with next chunk
                if i + 1 < len(chunks):
                    _, next_end = chunks[i + 1]
                    # Merge current and next chunk
                    merged_end = next_end
                    merged_duration = merged_end - current_start

                    # If merged chunk is still too short and there are more chunks, continue merging
                    j = i + 2
                    while merged_duration < min_length_sec and j < len(chunks):
                        _, next_next_end = chunks[j]
                        merged_end = next_next_end
                        merged_duration = merged_end - current_start
                        j += 1

                    merged_chunks.append((current_start, merged_end))
                    i = j  # Skip all merged chunks
                else:
                    # Last chunk and it's short - either merge with previous or keep as is
                    if merged_chunks and current_duration < min_length_sec:
                        # Merge with the last chunk in merged_chunks
                        last_start, _ = merged_chunks[-1]
                        merged_chunks[-1] = (last_start, current_end)
                    else:
                        # Keep the short chunk as is (no choice)
                        merged_chunks.append((current_start, current_end))
                    i += 1

        # Extract start and end times from merged chunks
        if merged_chunks:
            merged_start_times, merged_end_times = zip(*merged_chunks)
            return list(merged_start_times), list(merged_end_times)
        else:
            _LOGGER.error("Something wrong. No chunks were detected.")
            return chunk_start_times, chunk_end_times
