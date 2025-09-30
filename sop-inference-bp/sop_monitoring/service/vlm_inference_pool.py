# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import os
import logging
import traceback
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future

import minio

from .constants import (
    REDIS_STREAM_P_API_SERVER_C_VLM_INFERENCE_STREAM_NAME,
    REDIS_STREAM_C_VLM_INFERENCE_P_API_SERVER_GROUP_NAME,
    MINIO_BUCKET,
)

from .msg_types import VlmInferenceRequest, VlmInferenceResponse
from .redis_stream import RedisStream, send_response
from .redis_database import RadisDatabase
from .multi_gpu_vlm_manager import MultiGPUVLMManager

_LOGGER = logging.getLogger(__name__)

class VLMInferencePool:
    """
    Thread pool wrapper for handling VLM inference requests concurrently.
    Integrates MultiGPUVLMManager with the existing Redis stream processing.
    """

    def __init__(self, model_dir: str, max_workers: int, max_gpu_queue_size: int, tmp_dir: str):
        """
        Initialize the VLM inference pool.

        Args:
            model_dir: Path to the VLM model directory
            max_workers: Maximum number of worker threads for handling Redis messages
            max_gpu_queue_size: Maximum queue size for the GPU manager
            tmp_dir: Temporary directory for storing video files
        """

        try:
            self.time_out_in_seconds = int(os.environ["VLM_INFERENCE_TIME_OUT_SECONDS"])
        except Exception as e:
            _LOGGER.warning("Error getting VLM_INFERENCE_TIME_OUT_SECONDS: %s", e)
            self.time_out_in_seconds = 900
            _LOGGER.warning("VLM_INFERENCE_TIME_OUT_SECONDS is not set. Using default value of %s seconds", self.time_out_in_seconds)

        self.model_dir = model_dir
        self.max_workers = max_workers
        self.tmp_dir = tmp_dir
        # Initialize GPU manager
        self.gpu_manager = MultiGPUVLMManager(model_dir, max_queue_size=max_gpu_queue_size)

        # Thread pool for handling Redis message processing
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="VLM-Request-Worker"
        )

        self.is_running = True

        _LOGGER.info(f"VLM Inference Pool initialized with {max_workers} workers")

    def get_video_filenames(self, redis_database: RadisDatabase, chunk_keys: list[str], chunk_dir: str) -> list[str]:
        """Extract video files from Redis database to local directory"""
        video_filenames = []
        for chunk_key in chunk_keys:
            video_data = redis_database.get_blob(chunk_key)
            if not video_data:
                _LOGGER.error("No video data found for chunk key: %s", chunk_key)
                continue
            video_filename = os.path.join(chunk_dir, f"{chunk_key}.mp4")
            with open(video_filename, "wb") as f:
                f.write(video_data)
            video_filenames.append(video_filename)
        return video_filenames

    def _process_chunk_and_infer_request(self,
                                         message_id: str,
                                         vlm_request: VlmInferenceRequest,
                                         redis_stream: RedisStream,
                                         minio_client: minio.Minio):
        """
        Process a single VLM chunk and infer request in a worker thread.
        """
        try:
            _LOGGER.info(f"Processing VLM chunk and infer request: {message_id}")

            # download video file from minio
            minio_response = minio_client.get_object(MINIO_BUCKET, vlm_request.video_id)
            with tempfile.TemporaryDirectory(dir=self.tmp_dir) as tmp_dir:
                video_path = os.path.join(tmp_dir, vlm_request.video_id)
                with open(video_path, "wb") as video_fp:
                    piece_size = 50*1024*1024 # 50MB
                    for piece in minio_response.stream(piece_size):
                        video_fp.write(piece)

                futures = []
                start_to_submit_request = time.time()
                for chunk_start_second, chunk_end_second in zip(vlm_request.chunk_start_seconds, vlm_request.chunk_end_seconds):
                    request_id = uuid.uuid4().hex
                    _LOGGER.debug("Submitting chunk and infer request %s for message_id %s and chunk %.2fs-%.2fs", request_id, message_id, chunk_start_second, chunk_end_second)
                    future = self.gpu_manager.submit_chunk_and_infer(
                        request_id=request_id,
                        system_prompt=vlm_request.system_prompt,
                        prompt=vlm_request.prompt,
                        video_filename=video_path,
                        chunk_start_second=chunk_start_second,
                        chunk_end_second=chunk_end_second)
                    futures.append(future)
                after_submit_request = time.time()

                if vlm_request.stream_response:
                    # Handle streaming response
                    _LOGGER.info("Streaming response for message %s", message_id)

                    # send response once future is available
                    for future in futures[:-1]:
                        results = future.result(timeout=self.time_out_in_seconds)
                        vlm_response = VlmInferenceResponse(
                            request_id=message_id,
                            contents=results,
                            final_response=False,
                            error_message="",
                        )
                        send_response(redis_stream=redis_stream,
                                      received_request=vlm_request,
                                      response=vlm_response)

                    # send final future result
                    results = futures[-1].result(timeout=self.time_out_in_seconds)
                    after_get_result = time.time()

                    vlm_response = VlmInferenceResponse(
                        request_id=message_id,
                        contents=results,
                        final_response=True,
                        error_message="",
                    )
                else:
                    # Wait for all results
                    results = []
                    for future in futures:
                        results.extend(future.result(timeout=self.time_out_in_seconds))
                    after_get_result = time.time()

                    vlm_response = VlmInferenceResponse(
                        request_id=message_id,
                        contents=results,
                        final_response=True,
                        error_message="",
                    )

                _LOGGER.info("VLM chunk and infer submission took %s seconds", after_submit_request - start_to_submit_request)
                _LOGGER.info("VLM chunk and infer get result took %s seconds", after_get_result - after_submit_request)

        except Exception as e:
            error_msg = f"Error processing VLM chunk and infer request {message_id}, error: {traceback.format_exc()}"
            _LOGGER.error(error_msg)
            vlm_response = VlmInferenceResponse(
                request_id=message_id,
                contents=[],
                final_response=True,
                error_message=error_msg,
            )
        finally:
            # always send final response
            send_response(redis_stream=redis_stream,
                          received_request=vlm_request,
                          response=vlm_response)

            _LOGGER.info("Processed and acknowledged message %s", message_id)
            redis_stream.ack(REDIS_STREAM_P_API_SERVER_C_VLM_INFERENCE_STREAM_NAME,
                             REDIS_STREAM_C_VLM_INFERENCE_P_API_SERVER_GROUP_NAME,
                             message_id)

    def _process_inference_request(self, message_id: str, vlm_request: VlmInferenceRequest,
                                 redis_stream: RedisStream, redis_database: RadisDatabase):
        """
        Process a single VLM inference request in a worker thread.

        Args:
            message_id: Redis message ID
            vlm_request: The inference request
            redis_stream: Redis stream for sending responses
            redis_database: Redis database for video data
        """
        try:
            _LOGGER.info(f"Processing VLM inference request: {message_id}")

            with tempfile.TemporaryDirectory(dir=self.tmp_dir) as tmp_dir:
                # Extract video files
                video_filenames = self.get_video_filenames(redis_database, vlm_request.chunk_keys, tmp_dir)

                if not video_filenames:
                    raise ValueError("No video files found for the request")

                futures = []
                start_to_submit_request = time.time()
                for video_filename in video_filenames:
                    video_filename_basename = os.path.basename(video_filename)
                    future = self.gpu_manager.submit_inference(
                        request_id=f"{message_id}_{video_filename_basename}",
                        system_prompt=vlm_request.system_prompt,
                        prompt=vlm_request.prompt,
                        video_filenames=[video_filename]
                    )
                    futures.append(future)
                after_submit_request = time.time()

                if vlm_request.stream_response:
                    # Handle streaming response
                    _LOGGER.info("Streaming response for message %s", message_id)

                    # send response once future is available
                    for future in futures[:-1]:
                        results = future.result(timeout=self.time_out_in_seconds)

                        vlm_response = VlmInferenceResponse(
                            request_id=message_id,
                            contents=results,
                            final_response=False,
                            error_message="",
                        )

                        send_response(redis_stream=redis_stream,
                                    received_request=vlm_request,
                                    response=vlm_response)

                    # send final future result
                    results = futures[-1].result(timeout=self.time_out_in_seconds)
                    after_get_result = time.time()

                    vlm_response = VlmInferenceResponse(
                        request_id=message_id,
                        contents=results,
                        final_response=True,
                        error_message="",
                    )

                else:

                    # Wait for all results
                    results = []
                    for future in futures:
                        results.extend(future.result(timeout=self.time_out_in_seconds))
                    after_get_result = time.time()

                    vlm_response = VlmInferenceResponse(
                        request_id=message_id,
                        contents=results,
                        final_response=True,
                        error_message="",
                    )

                _LOGGER.info("VLM inference submission took %s seconds", after_submit_request - start_to_submit_request)
                _LOGGER.info("VLM inference get result took %s seconds", after_get_result - after_submit_request)

        except Exception as e:
            _LOGGER.error(f"Error processing message {message_id}: {e}", exc_info=True)

            vlm_response = VlmInferenceResponse(
                request_id=message_id,
                contents=[],
                final_response=True,
                error_message=traceback.format_exc(),
            )

        finally:
            # Always send final response
            send_response(redis_stream=redis_stream,
                        received_request=vlm_request,
                        response=vlm_response)

            _LOGGER.info("Processed and acknowledged message %s", message_id)
            redis_stream.ack(REDIS_STREAM_P_API_SERVER_C_VLM_INFERENCE_STREAM_NAME,
                             REDIS_STREAM_C_VLM_INFERENCE_P_API_SERVER_GROUP_NAME,
                             message_id)

    def submit_request(self,
                       message_id: str,
                       vlm_request: VlmInferenceRequest,
                       redis_stream: RedisStream,
                       redis_database: RadisDatabase,
                       minio_client: minio.Minio):
        """
        Submit a VLM inference request to the thread pool.

        Args:
            message_id: Redis message ID
            vlm_request: The inference request
            redis_stream: Redis stream for sending responses
            redis_database: Redis database for video data
            minio_client: Minio client for video data

        Returns:
            Future representing the submitted task
        """
        if not self.is_running:
            raise RuntimeError("VLM Inference Pool is not running")

        def logging_callback(future: Future):
            try:
                _LOGGER.debug("Calling result for request %s", message_id)
                future.result(timeout=self.time_out_in_seconds)
                _LOGGER.info("Request %s completed", message_id)
            except TimeoutError:
                _LOGGER.error("Request %s timed out", message_id)
            except Exception as e:
                _LOGGER.error("Error processing request %s: %s", message_id, e)

        if vlm_request.chunk_keys:
            future = self.thread_pool.submit(
                self._process_inference_request,
                message_id,
                vlm_request,
                redis_stream,
                redis_database
            )
        else:
            future = self.thread_pool.submit(
                self._process_chunk_and_infer_request,
                message_id,
                vlm_request,
                redis_stream,
                minio_client
            )

        future.add_done_callback(logging_callback)

        _LOGGER.debug(f"Submitted request {message_id} to thread pool")

    def shutdown(self, timeout: float = 30.0):
        """Shutdown the inference pool and cleanup resources"""
        _LOGGER.info("Shutting down VLM Inference Pool...")

        self.is_running = False

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True, timeout=timeout)

        # Shutdown GPU manager
        self.gpu_manager.shutdown(timeout=timeout)

        _LOGGER.info("VLM Inference Pool shutdown complete")

