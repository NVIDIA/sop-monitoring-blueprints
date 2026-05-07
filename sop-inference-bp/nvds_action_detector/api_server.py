# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
API Server for NVDS Action Detector
"""

import asyncio
import base64
import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import AsyncGenerator, Callable, Dict, Optional, Union

import aiohttp
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from . import ds_logger
from .api_types import (
    CameraInputContent,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DdmNetChunkingOptions,
    DeletionStatus,
    DeltaMessage,
    DSSOPMetadataResponse,
    ErrorInfo,
    ErrorResponse,
    FileList,
    FileObject,
    HealthSuccessResponse,
    LicenseInfoResponse,
    TextContent,
    UniformChunkingOptions,
    VideoFileContent,
    VideoURLContent,
)
from .ds_sop_process import ChunkParams, SOPProcessManager
from .utils import TimeMeasure

logger = ds_logger.get_logger(__name__)

API_SERVER_PORT = int(os.environ.get("API_SERVER_PORT", 8300))
STORAGE_DIR = os.environ.get("MEDIA_STORAGE_DIR", "/tmp/nvds_sop_storage")
METADATA_PATH = os.path.join(STORAGE_DIR, "metadata.json")
DS_SOP_LICENSE_PATH = os.environ.get("DS_SOP_LICENSE_PATH", "/opt/mm/LICENSE")

API_DUMMY_TEST = os.environ.get("API_DUMMY_TEST", "false").lower() in ["true", "1", "yes", "y"]
DS_SOP_VERSION = os.environ.get("DS_SOP_VERSION", "1.0.0")

global_metadata: dict = None
global_metadata_lock: asyncio.Lock = None

global_sop_manager: SOPProcessManager = None


MODEL_REGISTRY = {
    "ddm_cv_model": {
        "id": "ddm_finetune",
        "modalities": ["video"],
        "description": "DDM model for action segmentation",
    },
    "vlm_model": {
        "id": "cosmos_reason1_finetune",
        "modalities": ["image", "text"],
        "description": "Cosmos Reason1 finetune model",
    },
}

# ---------------------------
# Metrics Generation
# ---------------------------

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["path", "method"])

REQUEST_LATENCY = Histogram("api_request_latency_seconds", "API request latency", ["path", "method"])

CHAT_COMPLETIONS_COUNT = Counter("chat_completions_total", "Number of chat completion requests")
CHAT_COMPLETIONS_LATENCY = Histogram(
    "chat_completions_latency_seconds",
    "Latency of /v1/chat/completions in seconds",
    buckets=[0.5, 1, 5, 10],  # customize as needed
)

GPU_UTILIZATION = Gauge("gpu_utilization_percent", "GPU utilization percent", ["gpu"])
GPU_MEMORY = Gauge("gpu_memory_used_megabytes", "GPU memory used in MB", ["gpu"])


# ---------------------------
# API Server app Lifespan
# ---------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI app
    """
    global global_metadata, global_metadata_lock, global_sop_manager

    # Initialize storage and metadata on startup
    os.makedirs(STORAGE_DIR, exist_ok=True)
    # Check $STORAGE_DIR write permission
    if not os.access(STORAGE_DIR, os.W_OK):
        logger.error(f"No write permission for storage directory: {STORAGE_DIR}")
        logger.error(f"Current user: {os.getuid()}, group: {os.getgid()}")
        raise Exception(f"No write permission for storage directory: {STORAGE_DIR}")

    global_metadata = load_metadata()
    global_metadata_lock = asyncio.Lock()
    logger.info("Application started")

    yield

    logger.info("Closing SOP process manager...")
    if global_sop_manager is not None:
        global_sop_manager.close()
    global_sop_manager = None
    logger.info("SOP process manager closed")
    logger.info("Deleting storage directory...")
    # os.rmdir(STORAGE_DIR)
    logger.info("Storage directory deleted")


openapi_tags = [
    {
        "name": "Chat Completions",
        "description": "Operations to generate chat completions for a video stream",
    },
    {
        "name": "File and Stream Management",
        "description": "Files are used to upload and manage media files.",
    },
    {"name": "Health Check", "description": "Operations to check system health."},
    {"name": "Metrics", "description": "Operations to get metrics."},
    {
        "name": "Models",
        "description": "List and describe the various models available in the API.",
    },
    {"name": "Metadata", "description": "Operations to get service metadata."},
]

app = FastAPI(
    title="DeepStream SOP API",
    description="APIs for NVDS SOP CV and VLM cycle detection",
    contact={"name": "NVIDIA", "url": "https://nvidia.com"},
    version=DS_SOP_VERSION,
    lifespan=lifespan,
    openapi_tags=openapi_tags,
)

# ---------------------------
# Middleware to count all requests
# ---------------------------


@app.middleware("http")
async def add_metrics(request: Request, call_next: Callable):
    REQUEST_COUNT.labels(path=request.url.path, method=request.method).inc()

    with REQUEST_LATENCY.labels(path=request.url.path, method=request.method).time():
        response = await call_next(request)

    return response


def load_metadata() -> dict:
    if not os.path.exists(METADATA_PATH):
        return {}
    try:
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Error loading metadata: %s", e)
        return {}


def save_metadata(metadata: dict):
    try:
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f)
    except Exception as e:
        logger.error("Error saving metadata: %s", e)


def _create_error_response(
    message: str, err_type: str = "BadRequestError", status_code: HTTPStatus = HTTPStatus.BAD_REQUEST
) -> ErrorResponse:
    return ErrorResponse(error=ErrorInfo(message=message, type=err_type, code=status_code.value))


async def download_video_from_url(video_url: str, storage_dir: str) -> str:
    """
    Download video from URL or decode from base64 data URI.

    Args:
        video_url: Either HTTP(S) URL or data URI (data:video/mp4;base64,...)
        storage_dir: Directory to save the video file

    Returns:
        Path to the downloaded/saved video file

    Raises:
        HTTPException: If download fails or invalid format
    """
    # Generate unique filename
    file_id = f"video-{uuid.uuid4().hex}.mp4"
    file_path = os.path.join(storage_dir, file_id)

    try:
        # Check if it's a data URI (base64 encoded)
        if video_url.startswith("data:"):
            logger.info("Processing base64 data URI")

            # Parse data URI format: data:video/mp4;base64,{base64data}
            match = re.match(r"data:video/[^;]+;base64,(.+)", video_url)
            if not match:
                raise HTTPException(
                    status_code=400, detail="Invalid data URI format. Expected: data:video/mp4;base64,{base64data}"
                )

            base64_data = match.group(1)

            # Decode base64 data
            try:
                video_data = base64.b64decode(base64_data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode base64 data: {str(e)}")

            # Write to file (use thread pool to avoid blocking)
            def write_file():
                with open(file_path, "wb") as f:
                    f.write(video_data)

            await asyncio.to_thread(write_file)
            logger.info(f"Saved base64 video to {file_path} ({len(video_data)} bytes)")

        # Check if it's HTTP(S) URL
        elif video_url.startswith(("http://", "https://")):
            logger.info("Downloading video from HTTP(S) URL")

            # Download using aiohttp for async HTTP
            async with aiohttp.ClientSession() as session:
                async with session.get(video_url, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400, detail=f"Failed to download video. HTTP status: {response.status}"
                        )

                    # Stream download to file
                    total_size = 0

                    def write_chunk(chunk):
                        nonlocal total_size
                        total_size += len(chunk)

                    with open(file_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(1024 * 1024):  # 1MB chunks
                            await asyncio.to_thread(f.write, chunk)
                            total_size += len(chunk)

                    logger.info(f"Downloaded video to {file_path} ({total_size} bytes)")
        elif video_url.startswith(("rtsp://")):
            logger.info(f"received rtsp url {video_url[:40]}...")
            return video_url
        else:
            raise HTTPException(
                status_code=400, detail="Invalid video URL format. Must start with 'http://', 'https://', or 'data:'"
            )

        return file_path

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Clean up partial file if it exists
        if os.path.exists(file_path):
            try:
                await asyncio.to_thread(os.remove, file_path)
            except Exception:
                pass

        logger.error(f"Error downloading video from URL: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}")


def _json_error_response(original: ErrorResponse, *, status_code: Optional[int] = None) -> JSONResponse:
    content = original.model_dump()

    if status_code is None:
        status_code = original.code

    return JSONResponse(content=content, status_code=status_code)


async def ds_sop_generate_chunks_dummy(
    request: ChatCompletionRequest, raw_request: Request
) -> AsyncGenerator[str, None]:
    """
    Dummy implementation of ds_sop_generate_chunks for unittest
    """
    req_id = str(uuid.uuid4().hex)
    checker_result = {
        "request_id": req_id,
        "error_message": "",
        "checker_id": "sopchecker-32810b9f-0f5e-4f5d-af58-de5be56c0159",
        "cycle": 0,
        "missing_detected": [],
        "misordered_detected": [],
        "final_missing_detected": [],
        "final_misordered_detected": [],
        "cycle_completed": False,
        "summary_cycles_detected": [],
        "summary_cycle_analysis": [],
    }
    for i in range(1, 9):
        chunk = {
            "chunk_idx": i,
            "response": f"Hello, world! {i}",
            "req_id": req_id,
            "start_time": i * 10,
            "end_time": (i + 1) * 10,
            "frame_number": 10,
            "vlm_execute_time": 0.1,
            "checker_execute_time": 0.1,
            "cv_execute_time": 0.1,
            "checker_result": checker_result,
        }
        yield chunk
    # return final chunk
    checker_result["cycle_completed"] = True
    checker_result["summary_cycles_detected"] = ["Cycle 0: [2, 3, 4, 4, 5, 5, 6, 7, 8]"]
    checker_result["summary_cycle_analysis"] = ["Cycle 0: [2, 3, 4, 5, 6, 7, 8] -> no issues"]
    chunk = {
        "chunk_idx": -1,
        "response": "Hello, world! final chunk",
        "req_id": req_id,
        "start_time": 0,
        "end_time": 0,
        "frame_number": 0,
        "vlm_execute_time": 0.1,
        "checker_result": checker_result,
    }
    yield chunk
    yield None


async def ds_sop_generate_chunks(request: ChatCompletionRequest, raw_request: Request) -> AsyncGenerator[str, None]:
    """
    Create a chat completion
    """
    text_content = ""
    video_file_path = ""
    is_tmp_file = False
    is_live_stream = False
    camera_serial_number = None
    camera_config = None
    camera_extra_args = {
        "camera_format": None,
        "camera_width": None,
        "camera_height": None,
        "camera_fps_num": None,
        "camera_fps_den": None,
    }
    try:
        logger.info(f"Received request with model: {request.model}")
        logger.info(f"Number of messages: {len(request.messages)}")

        if len(request.messages) != 1:
            raise HTTPException(status_code=400, detail="Only one message with role 'user' is supported.")

        user_message = request.messages[0]
        if user_message.role != "user":
            raise HTTPException(status_code=400, detail=f"Only support role 'user', but got {user_message.role}")

        msg_content = request.messages[0].content
        if not isinstance(msg_content, list):
            raise HTTPException(
                status_code=400,
                detail=f"Content must be a list with type 'text' and 'image_file', but got {msg_content}",
            )

        for content in msg_content:
            if isinstance(content, TextContent):
                if text_content:
                    logger.error(f"Found one text content {text_content} but got another text contents: {content.text}")
                    raise HTTPException(
                        status_code=400, detail="Only one text content is supported, but got multiple text contents."
                    )
                text_content = content.text

            elif isinstance(content, VideoFileContent):
                if video_file_path:
                    raise HTTPException(
                        status_code=400,
                        detail="Only one video file is supported, but got multiple video files in a request.",
                    )

                video_file_path = content.file_id
                video_file_path = os.path.join(STORAGE_DIR, video_file_path)
                if not os.path.exists(video_file_path):
                    raise HTTPException(status_code=404, detail=f"Video file {video_file_path} not found")
            elif isinstance(content, VideoURLContent):
                if video_file_path:
                    raise HTTPException(
                        status_code=400,
                        detail="Only one video file is supported, but got multiple video files in a request.",
                    )
                video_url = content.video_url.url.strip()
                logger.info(f"Downloading video url {video_url[:40]}...")
                video_file_path = await download_video_from_url(video_url, STORAGE_DIR)
                if video_url.startswith("rtsp://"):
                    is_live_stream = True
                    is_tmp_file = False
                else:
                    is_tmp_file = True
                    is_live_stream = False
            elif isinstance(content, CameraInputContent):
                camera_serial_number = content.input_camera.camera_id
                if content.input_camera.camera_vendor != "Basler":
                    raise HTTPException(
                        status_code=400,
                        detail=f"Only support camera vendor 'Basler', but got {content.input_camera.camera_vendor}",
                    )
                camera_config = content.input_camera.config
                logger.info(
                    f"Received camera input with camera_id: {camera_serial_number}, camera_config: {camera_config}"
                )
                # TODO: Implement camera input processing
                video_file_path = "camera://" + camera_serial_number
                is_live_stream = True
                camera_extra_args = {
                    "camera_format": content.input_camera.camera_format,
                    "camera_width": content.input_camera.camera_width,
                    "camera_height": content.input_camera.camera_height,
                    "camera_fps_num": content.input_camera.camera_fps_num,
                    "camera_fps_den": content.input_camera.camera_fps_den,
                }
            else:
                raise HTTPException(status_code=400, detail=f"content type {content.type} is not supported")

        if not video_file_path:
            raise HTTPException(
                status_code=400,
                detail="video file path must be provided.",
            )
        if is_live_stream and not request.stream:
            raise HTTPException(
                status_code=400,
                detail="Live stream requests must enable streaming response, but got non-streaming request",
            )

        chunking_options = request.chunking_options
        if chunking_options.algorithm == "ddm-net":
            chunk_params = ChunkParams(
                algorithm="ddm-net",
                min_length_sec=chunking_options.min_length_sec,
                max_length_sec=chunking_options.max_length_sec,
                threshold=chunking_options.threshold,
            )
        elif chunking_options.algorithm == "uniform":
            chunk_params = ChunkParams(
                algorithm="uniform",
                chunk_length_sec=chunking_options.chunk_length_sec,
                max_length_sec=chunking_options.chunk_length_sec,
            )
        else:
            chunk_params = ChunkParams(min_length_sec=1, max_length_sec=10, threshold=0.8)

        global global_sop_manager
        loop = asyncio.get_running_loop()
        processor = None
        graceful_stop = False
        try:
            processor = global_sop_manager.create_video_processor(
                file_path=video_file_path,
                chunk_params=chunk_params,
                camera_serial_number=camera_serial_number,
                camera_config=camera_config,
                prompt=text_content,
                temperature=request.temperature,
                max_completion_tokens=request.max_completion_tokens,
                seed=request.seed,
                top_p=request.top_p,
                **camera_extra_args,
            )
            await processor.start()

            while True:
                # Wait for next chunk with periodic disconnection checks
                # Using timeout on queue.get() to allow checking client connection status
                chunk = None
                client_disconnected = False
                while True:
                    # check if client disconnected, request must close gracefully by itself
                    chunk = None
                    try:
                        disconnect_check = await asyncio.wait_for(raw_request.receive(), timeout=0.01)
                        if disconnect_check["type"] == "http.disconnect":
                            logger.info(f"Client: {processor._id} disconnected (detected via receive)")
                            client_disconnected = True
                            break
                    except asyncio.TimeoutError:
                        # No disconnect message - client still connected
                        pass

                    # Try to get chunk with timeout to avoid blocking indefinitely
                    try:
                        chunk = await loop.run_in_executor(
                            None, lambda: processor.final_queue.get(block=True, timeout=0.3)
                        )
                    except queue.Empty:
                        # No chunk yet, loop will check for disconnection again
                        continue
                    # New chunk is ready, break the loop
                    break
                if client_disconnected:
                    break
                need_to_break = chunk is None or chunk.get("is_last_item", False)
                # logger.info(f"Yielding chunk: {chunk.get('req_id', None)}, need_to_break: {need_to_break}")

                try:
                    yield chunk
                except Exception as e:
                    # When client disconnects, yield can raise:
                    # - ConnectionResetError: client closed connection
                    # - BrokenPipeError: write to closed socket
                    # - asyncio.CancelledError: task was cancelled
                    # - RuntimeError: generator already executing / closed
                    logger.info(f"Error during yield (likely client disconnected): {type(e).__name__}: {e}")
                    raise HTTPException(status_code=400, detail="Client disconnected during write")

                if need_to_break:
                    graceful_stop = True
                    break
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.exception(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # stop the processor in background
            if processor:
                logger.info(f"Stopping processor: {processor._id} in background, graceful_stop: {graceful_stop}")
                future = global_sop_manager.trigger_stop_processors(processor, force=not graceful_stop)

                def log_errors(fut):
                    try:
                        fut.result()  # Will raise if there was an exception
                    except Exception as e:
                        logger.exception(f"Error stopping processor in background: {e}")

                future.add_done_callback(log_errors)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if is_tmp_file:
            try:
                await asyncio.to_thread(os.remove, video_file_path)
            except Exception:
                pass


async def _run_stream_chunks_response(
    chunks_generator: AsyncGenerator[Dict, None], request: ChatCompletionRequest, raw_request: Request
) -> Union[ErrorResponse, AsyncGenerator[str, None]]:
    """
    Run stream response for chunks
    """

    try:
        chat_id = None

        async for chunk in chunks_generator:
            if chunk is None:
                break
            assert isinstance(chunk, dict)
            delta_message = DeltaMessage(content=chunk.get("response", "").strip())
            choice = ChatCompletionResponseStreamChoice(
                index=0, delta=delta_message, finish_reason=None, chunk_metadata=chunk
            )
            if chat_id is None:
                req_id = chunk.get("req_id", str(uuid.uuid4().hex))
                chat_id = f"chatcmpl-{req_id}"
            chat_completion_stream_response = ChatCompletionStreamResponse(
                id=chat_id,
                object="chat.completion.chunk",
                created=int(time.time()),
                model=request.model,
                choices=[choice],
            )
            yield f"data: {chat_completion_stream_response.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_chat_completion_response(
    chunks_generator: AsyncGenerator[Dict, None], request: ChatCompletionRequest, raw_request: Request
) -> Union[ErrorResponse, ChatCompletionResponse]:
    """
    Run chat completion response
    """
    try:
        chat_id = None
        responses = []
        metadata = []
        async for chunk in chunks_generator:
            if chunk is None:
                logger.info(f"{chat_id} received last chunk, breaking")
                break
            if chat_id is None:
                chat_id = chunk.get("req_id", str(uuid.uuid4().hex))
                chat_id = f"chatcmpl-{chat_id}"
            logger.info(f"{chat_id} Received new chunk, response: {chunk.get('response', '').strip()[:30]}...")
            responses.append(chunk.get("response", "").strip())
            metadata.append(chunk)
        logger.info(f"{chat_id} received {len(responses)} chunks, responses: {responses[:30]}...")
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatCompletionMessage(role="assistant", content="\n".join(responses)),
            finish_reason="stop",
            chunk_metadata_list=metadata,
        )
        logger.info(
            f"chat_id: {chat_id}, Chat completion response_length: {len(responses)}, response: {choice.message.content[:30]}..."
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-{chat_id}", created=int(time.time()), model=request.model, choices=[choice], usage={}
        )

    except Exception as e:
        logger.exception(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def validate_json_request(raw_request: Request):
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise HTTPException(status_code=415, detail="Unsupported Media Type: Only 'application/json' is allowed")


@app.post(
    "/v1/chat/completions",
    summary="OpenAI-compatible chat endpoint",
    response_model=Union[ChatCompletionResponse, ChatCompletionStreamResponse],
    responses={
        400: {
            "description": "Received an invalid request possibly containing unsupported or out-of-range parameter values.",
            "model": ErrorResponse,
        },
        404: {
            "description": "The requested model does not exist.",
            "model": ErrorResponse,
        },
        500: {"description": "", "model": ErrorResponse},
        415: {"description": "Unsupported Media Type", "model": ErrorResponse},
    },
    dependencies=[Depends(validate_json_request)],
    tags=["Chat Completions"],
)
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    logger.info(f"Received chat completion request at timestamp: {time.time()}, steam_option: {request.stream}")
    tm = TimeMeasure()

    try:
        if API_DUMMY_TEST:
            chunks_generator = ds_sop_generate_chunks_dummy(request, raw_request)
        else:
            chunks_generator = ds_sop_generate_chunks(request, raw_request)
        if isinstance(chunks_generator, ErrorResponse):
            return _json_error_response(chunks_generator, status_code=chunks_generator.code)
        if request.stream:
            stream_generator = _run_stream_chunks_response(chunks_generator, request, raw_request)
            return StreamingResponse(content=stream_generator, media_type="text/event-stream")
        else:
            response = await _run_chat_completion_response(chunks_generator, request, raw_request)
            assert isinstance(response, ChatCompletionResponse)
            return response
    finally:
        tm.log_elapsed_time("Chat completions processing latency")
        latency = tm.elapsed_time
        CHAT_COMPLETIONS_LATENCY.observe(latency)
        CHAT_COMPLETIONS_COUNT.inc()


@app.post(
    "/v1/files",
    response_model=FileObject,
    tags=["File and Stream Management"],
)
async def upload_file(file: UploadFile = File(...), purpose: str = Form(...)):
    """
    Uploads a file and returns the file object.
    """
    file_size = file.size
    file_id = f"file-{uuid.uuid4().hex}"
    created_at = int(time.time())
    file_path = os.path.join(STORAGE_DIR, file_id)
    with open(file_path, "wb") as out:
        while content := await file.read(1024 * 1024):
            out.write(content)

    file_object = FileObject(
        id=file_id,
        bytes=file_size,
        created_at=created_at,
        filename=file.filename,
        purpose=purpose,
    )

    async with global_metadata_lock:
        global_metadata[file_id] = file_object.model_dump()
        save_metadata(global_metadata)

    return file_object


@app.get("/v1/files/{file_id}/content", tags=["File and Stream Management"])
async def get_file_content(file_id: str):
    """
    Downloads the content of a specific file.
    """
    # Quickly get metadata with lock
    async with global_metadata_lock:
        file_object = global_metadata.get(file_id)

    if file_object is None:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    file_path = os.path.join(STORAGE_DIR, file_id)

    # Check if file exists on disk
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {file_id} not found on disk")

    # FileResponse handles efficient streaming automatically
    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=file_object["filename"],
        headers={"Content-Length": str(file_object["bytes"])},
    )


@app.delete("/v1/files/{file_id}", response_model=DeletionStatus, tags=["File and Stream Management"])
async def delete_file(file_id: str):
    """
    Deletes a file and its metadata.
    """
    # Check if file exists in metadata
    async with global_metadata_lock:
        if file_id not in global_metadata:
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    file_path = os.path.join(STORAGE_DIR, file_id)

    # Delete physical file first (non-blocking)
    try:
        await asyncio.to_thread(os.remove, file_path)
    except FileNotFoundError:
        # File already deleted or never existed - log but continue to clean metadata
        logger.warning(f"File {file_id} not found on disk, cleaning up metadata")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

    # Remove from metadata after successful file deletion
    async with global_metadata_lock:
        if file_id in global_metadata:  # Double-check in case of race condition
            del global_metadata[file_id]
            save_metadata(global_metadata)

    return DeletionStatus(id=file_id, deleted=True)


@app.get("/v1/files", response_model=FileList, tags=["File and Stream Management"])
async def list_files():
    """List all files in the medatadata storage"""
    async with global_metadata_lock:
        return FileList(data=[FileObject.model_validate(file_doc) for file_doc in global_metadata.values()])


@app.get("/v1/models", tags=["Models"])
async def list_models() -> Response:
    """List available models endpoint"""
    model_id = os.environ.get("DS_SOP_MODEL_NAME", "ds_sop_model")
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model", "created": 0, "owned_by": "nvds-sop-action-detector"}],
    }


@app.get(
    "/v1/metadata",
    summary="Provide DS SOP Metadata",
    description=("Show DS SOP Metadata").replace("\n", " ").strip(),
    response_model=DSSOPMetadataResponse,
    tags=["Metadata"],
)
async def show_metadata() -> Response:
    ver = DS_SOP_VERSION
    license_info = _get_license_info()
    model_info = MODEL_REGISTRY
    metadata = DSSOPMetadataResponse(version=ver, modelInfo=model_info, licenseInfo=license_info)
    return JSONResponse(content=metadata.model_dump())


@app.get(
    "/v1/live",
    summary="Service liveness check",
    description=("Indicates if the service is alive").strip(),
    response_model=Union[HealthSuccessResponse],
    tags=["Health Check"],
)
async def health_live() -> Response:
    return JSONResponse(content=HealthSuccessResponse(message="Service is live.").model_dump())


@app.get(
    "/v1/startup",
    summary="Microservice startup status",
    description=("Indicates if the service is alive").strip(),
    response_model=Union[HealthSuccessResponse],
    tags=["Health Check"],
)
async def health_startup() -> Response:
    return JSONResponse(content=HealthSuccessResponse(message="Service started successfully.").model_dump())


@app.get(
    "/v1/ready",
    response_model=HealthSuccessResponse,
    summary="Service ready check",
    description=("Indicates if the service is ready to receive inference requests").replace("\n", " ").strip(),
    responses={
        503: {
            "description": "Service is not ready to receive requests.",
            "model": ErrorResponse,
        },
    },
    tags=["Health Check"],
)
async def health_ready() -> Response:
    """Health check.
    Ensures all backend services are ready to serve inference requests.
    """
    try:
        if API_DUMMY_TEST:
            return JSONResponse(content=HealthSuccessResponse(message="Dummy test mode,Service is ready.").model_dump())
        global global_sop_manager
        if global_sop_manager is None or not global_sop_manager.is_model_ready():
            return _json_error_response(
                _create_error_response(
                    message="Service is not ready to receive requests.",
                    err_type="ServiceUnavailableError",
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                )
            )
    except Exception as e:
        return _json_error_response(
            _create_error_response(
                message=f"Service is not ready to receive requests, raised error: {str(e)}",
                err_type="ServiceUnavailableError",
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        )
    return JSONResponse(content=HealthSuccessResponse(message="Service is ready.").model_dump())


def _get_license_info():
    # Currently, License URL is not implemented
    with open(DS_SOP_LICENSE_PATH, "r") as f:
        content = f.read()
    license_info = LicenseInfoResponse(
        name=os.path.basename(DS_SOP_LICENSE_PATH),
        path=DS_SOP_LICENSE_PATH,
        size=os.path.getsize(DS_SOP_LICENSE_PATH),
        url="",
        type="file",
        content=content,
    )
    return license_info


def _update_gpu_metrics():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
        ).decode()

        for idx, line in enumerate(out.strip().split("\n")):
            util, mem = line.split(", ")
            GPU_UTILIZATION.labels(gpu=str(idx)).set(float(util))
            GPU_MEMORY.labels(gpu=str(idx)).set(float(mem))
    except Exception as e:
        logger.exception(f"Error updating GPU metrics: {e}")
        pass  # no GPU or nvidia-smi not installed


@app.get(
    "/v1/metrics",
    summary="Get Prometheus metrics",
    description="Get Prometheus metrics",
    responses={
        200: {"description": "Successful Response."},
        500: {"description": "Internal Server Error.", "model": ErrorResponse},
    },
    tags=["Metrics"],
)
async def metrics() -> Response:
    """Prometheus metrics endpoint"""
    try:
        _update_gpu_metrics()  # Update GPU metrics
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.exception(f"Error getting Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    global global_sop_manager
    if not API_DUMMY_TEST:
        global_sop_manager = SOPProcessManager()
        logger.info("Initializing SOP process manager...")
        global_sop_manager.wait_for_model_ready()
        logger.info("SOP process manager models initialized")
    else:
        logger.info("Running in DUMMY TEST mode - skipping SOP manager initialization")

    # Configure uvicorn to use proper log level from environment
    import logging

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    uvicorn_log_level = log_level_mapping.get(log_level, logging.INFO)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=API_SERVER_PORT,
        log_level=logging.getLevelName(uvicorn_log_level).lower(),  # uvicorn expects lowercase string
        access_log=True,
    )


if __name__ == "__main__":
    main()
