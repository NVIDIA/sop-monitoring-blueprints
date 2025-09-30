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
SOP Monitoring Inference API
"""

import asyncio
import os
import logging
import time
import uuid
from typing import AsyncGenerator
from contextlib import asynccontextmanager

import minio
import uvicorn

from fastapi import FastAPI, UploadFile,  File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pymongo import AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.errors import ConnectionFailure

from .pydantic_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse,
    CHUNKING_OPTIONS_TYPE_HINT,
    FileObject,
    FileList,
    DeletionStatus,
    SopDetectionRequest,
    SopDetectionResponse,
    SopDetectionSummary,
    Message,
)

from .msg_types import (
    VlmInferenceRequest,
    VlmInferenceResponse,
    ActionSegmentRequest,
    ActionSegmentResponse,
    SopCheckerRequest,
    SopCheckerResponse,
)
from .constants import (
    REDIS_STREAM_P_API_SERVER_C_VLM_INFERENCE_STREAM_NAME,
    REDIS_STREAM_P_API_SERVER_C_SOP_CHECKER_STREAM_NAME,
    REDIS_STREAM_DB_INDEX,
    REDIS_CHUNK_VIDEO_DB_INDEX,
    MINIO_BUCKET,
    REDIS_STREAM_NAME_TO_AVAILABLE_ALGOS,
)
from .redis_stream import (
    AsyncRedisStream,
    async_send_request_and_wait_for_response,
    async_send_request_and_get_response_generator,
)
from .redis_database import (
    AsyncRadisDatabase,
)
from .utils import (
    create_async_redis_client,
    generate_reply_stream_name,
    get_mongo_uri,
)

_LOGGER = logging.getLogger(__name__)

OPENAI_FILE_ID_PREFIX = "file-"

redis_client_for_stream = None
redis_client_for_database = None
minio_client = None
mongo_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI app
    """
    global redis_client_for_stream
    global redis_client_for_database
    global minio_client
    global mongo_client

    redis_client_for_stream = create_async_redis_client(REDIS_STREAM_DB_INDEX)
    redis_client_for_database = create_async_redis_client(REDIS_CHUNK_VIDEO_DB_INDEX)

    log_level_name = os.environ.get("API_SERVER_LOG_LEVEL", "INFO")
    try:
        log_level = getattr(logging, log_level_name.upper())
    except AttributeError:
        _LOGGER.error("Invalid log level: %s. Using INFO instead.", log_level_name)
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s"
    )

    _LOGGER.info("Testing Redis client at %s:%s",
                 redis_client_for_stream.connection_pool.connection_kwargs['host'],
                 redis_client_for_stream.connection_pool.connection_kwargs['port'])
    # this will raise exception if connection fails
    await redis_client_for_stream.ping()

    # raise exception if any one of environment variables is not set or anything wrong.
    minio_client = minio.Minio(
        f"{os.environ['MINIO_NAME']}:{os.environ['MINIO_API_PORT']}",
        access_key=os.environ["MINIO_ROOT_USER"],
        secret_key=os.environ["MINIO_ROOT_PASSWORD"],
        secure=False
    )

    await _ensure_minio_bucket_exists()

    mongo_uri = get_mongo_uri()
    mongo_client = AsyncMongoClient(mongo_uri)
    try:
        await mongo_client.aconnect()
        _LOGGER.info("MongoDB connection successful!")
    except ConnectionFailure as e:
        _LOGGER.error("Could not connect to MongoDB: %s", e)
        raise

    # set MongoDB log level
    mongo_log_level_name = os.environ.get("MONGO_LOG_LEVEL", "INFO")
    try:
        mongo_log_level = getattr(logging, mongo_log_level_name.upper())
    except AttributeError:
        _LOGGER.error("Invalid MongoDB log level: %s. Using INFO instead.", mongo_log_level_name)
        mongo_log_level = logging.INFO
    logging.getLogger('pymongo').setLevel(mongo_log_level)

    await _get_mongo_collection(mongo_client).create_index("id", unique=True)

    # application starts
    yield

    _LOGGER.info("Closing various resources...")
    await redis_client_for_stream.aclose()
    await redis_client_for_database.aclose()
    await mongo_client.close()


app = FastAPI(
    title="SOP Monitoring Inference API",
    description="APIs for SOP monitoring inference",
    version="1.0.0",
    lifespan=lifespan,
)

@asynccontextmanager
async def auto_delete_chunk_keys(redis_database: AsyncRadisDatabase, chunk_keys: list[str]) -> AsyncGenerator[None, None]:
    """Async context manager for automatic cleanup of chunk keys"""
    try:
        yield
    finally:
        await redis_database.delete_blobs_batch(chunk_keys)


async def _ensure_minio_bucket_exists():
    """
    Ensure that the required MinIO bucket exists, create it if it doesn't
    """
    def _inner():
        try:
            bucket_exists = minio_client.bucket_exists(MINIO_BUCKET)
            if not bucket_exists:
                _LOGGER.info("Creating MinIO bucket: %s", MINIO_BUCKET)
                minio_client.make_bucket(MINIO_BUCKET)
                _LOGGER.info("MinIO bucket '%s' created successfully", MINIO_BUCKET)
            else:
                _LOGGER.info("MinIO bucket '%s' already exists", MINIO_BUCKET)
        except Exception as e:
            _LOGGER.error("Failed to create MinIO bucket '%s': %s", MINIO_BUCKET, e)
            raise
    await asyncio.to_thread(_inner)


def _get_minio_object_name(file_object: FileObject) -> str:
    """
    Get the MinIO object name for a given file ID
    """
    return f"{file_object.id}-{file_object.filename}"


def _get_mongo_collection(mongo_client: AsyncMongoClient) -> AsyncCollection:
    return mongo_client["sop_monitoring"]["user_uploaded_files"]


async def _store_file_object(mongo_client: AsyncMongoClient, file_object: FileObject) -> bool:
    """
    Store a file object in the database
    """
    file_object_dict = file_object.model_dump()
    return await _get_mongo_collection(mongo_client).insert_one(file_object_dict)


async def _fetch_file_object(mongo_client: AsyncMongoClient, file_id: str) -> FileObject:
    """
    Fetch a file object from the database
    """
    file_object_dict = await _get_mongo_collection(mongo_client).find_one({"id": file_id})
    if not file_object_dict:
        raise HTTPException(status_code=404, detail="File not found")
    return FileObject.model_validate(file_object_dict)


async def _delete_file_object(mongo_client: AsyncMongoClient, file_id: str):
    """
    Delete a file object from the database
    """
    await _get_mongo_collection(mongo_client).delete_one({"id": file_id})

async def _run_vlm_inference(redis_stream: AsyncRedisStream, vlm_inference_request: VlmInferenceRequest) -> VlmInferenceResponse:
    """
    Run VLM inference
    """
    vlm_response = await async_send_request_and_wait_for_response(
        redis_stream,
        REDIS_STREAM_P_API_SERVER_C_VLM_INFERENCE_STREAM_NAME,
        vlm_inference_request,
        VlmInferenceResponse,
    )
    return vlm_response


async def _run_vlm_inference_stream(redis_stream: AsyncRedisStream, redis_database: AsyncRadisDatabase, model_name: str, vlm_inference_request: VlmInferenceRequest) -> AsyncGenerator[str, None]:
    """
    Run VLM inference and stream the response
    """

    response_generator = async_send_request_and_get_response_generator(
        redis_stream,
        REDIS_STREAM_P_API_SERVER_C_VLM_INFERENCE_STREAM_NAME,
        vlm_inference_request,
        VlmInferenceResponse,
    )

    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    should_continue = True
    responses = await response_generator.asend(None)
    try:
        while should_continue:

            last_response_is_final = False
            for idx, response in enumerate(responses):
                delta = ChatCompletionStreamResponseDelta(content="\n".join(response.contents))
                choices = [{
                    "delta": delta,
                    "finish_reason": None,
                    "index": idx,
                }]

                chat_completion_stream_response = CreateChatCompletionStreamResponse(
                    id=chat_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model_name,
                    choices=choices,
                )

                yield f"data: {chat_completion_stream_response.model_dump_json()}\n\n"

                last_response_is_final = response.final_response

            if last_response_is_final:
                should_continue = False
                chat_completion_stream_response = CreateChatCompletionStreamResponse(
                    id=chat_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model_name,
                    choices=[{
                        "index": 0,
                        "delta": ChatCompletionStreamResponseDelta(content=None),
                        "finish_reason": "stop",
                    }],
                )
                yield f"data: {chat_completion_stream_response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            responses = await response_generator.asend(should_continue)

    except StopAsyncIteration:
        _LOGGER.info("VLM inference stream finished")
    finally:
        await redis_database.delete_blobs_batch(vlm_inference_request.chunk_keys)

def _create_chat_completion_response(text_content: str, model_name: str, vlm_response: VlmInferenceResponse) -> ChatCompletionResponse:
    """
    Create a chat completion response
    """
    chat_response_content = "\n".join(vlm_response.contents)
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model_name,
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": chat_response_content
                },
                "finish_reason": "stop"
            }
        ],
        usage={
            "prompt_tokens": len(text_content),
            "completion_tokens": len(chat_response_content.split()),
            "total_tokens": len(text_content) + len(chat_response_content.split())
        }
    )


async def _run_action_segment(redis_stream: AsyncRedisStream,
                              video_id: str,
                              action_segment_options: CHUNKING_OPTIONS_TYPE_HINT) -> ActionSegmentResponse:
    """
    Run action segment
    """

    algorithm_name = action_segment_options.algorithm
    redis_stream_name = None
    for stream_name, algo_names in REDIS_STREAM_NAME_TO_AVAILABLE_ALGOS.items():
        if algorithm_name in algo_names:
            redis_stream_name = stream_name
            break
    if redis_stream_name is None:
        raise HTTPException(status_code=400, detail=f"Cannot find an image supporting algorithm {algorithm_name}.")

    _LOGGER.debug("Sending action segment request to %s with algorithm %s", redis_stream_name, algorithm_name)

    action_segment_request = ActionSegmentRequest(
        reply_stream_name=generate_reply_stream_name(),
        video_id=video_id,
        options_json=action_segment_options.model_dump_json(),
    )

    action_segment_response = await async_send_request_and_wait_for_response(
        redis_stream,
        redis_stream_name,
        action_segment_request,
        ActionSegmentResponse,
    )

    if action_segment_response.error_message:
        _LOGGER.error("Error processing message %s: %s",
                      action_segment_response.request_id, action_segment_response.error_message)
        raise HTTPException(status_code=500, detail=action_segment_response.error_message)

    return action_segment_response


async def chat_completions_impl(request: ChatCompletionRequest):
    """
    Chat completions endpoint conforming to OpenAI API

    Currently just echoes back the user's message
    """
    try:
        _LOGGER.info(f"Received request with model: {request.model}")
        _LOGGER.info(f"Number of messages: {len(request.messages)}")

        parser = _ChatCompletionRequestParser(request)
        user_prompt = parser.get_user_prompt()
        image_file_id = parser.get_image_file_id()
        system_prompt = parser.get_system_prompt()
        action_segment_options = parser.get_action_segment_options()

        redis_stream = AsyncRedisStream(redis_client_for_stream)
        redis_database = AsyncRadisDatabase(redis_client_for_database)

        # Action segment service need minio object name to download the video.
        file_object = await _fetch_file_object(mongo_client, image_file_id)
        minio_object_name = _get_minio_object_name(file_object)

        action_segment_response = await _run_action_segment(redis_stream, minio_object_name, action_segment_options)

        if action_segment_response.error_message:
            _LOGGER.error("Error processing message %s: %s",
                          action_segment_response.request_id, action_segment_response.error_message)
            raise HTTPException(status_code=500, detail=action_segment_response.error_message)

        chunk_start_seconds = action_segment_response.chunk_start_seconds
        chunk_end_seconds = action_segment_response.chunk_end_seconds
        chunk_keys = action_segment_response.chunk_keys

        vlm_inference_request = VlmInferenceRequest(
            system_prompt=system_prompt,
            prompt=user_prompt,
            video_id=minio_object_name,
            chunk_start_seconds=chunk_start_seconds,
            chunk_end_seconds=chunk_end_seconds,
            chunk_keys=chunk_keys,
            stream_response=request.stream,
            reply_stream_name=generate_reply_stream_name(),
        )

        if request.stream:
            response = StreamingResponse(
                _run_vlm_inference_stream(redis_stream, redis_database, request.model, vlm_inference_request),
                media_type="text/event-stream",
            )
        else:
            async with auto_delete_chunk_keys(redis_database, chunk_keys):
                vlm_response = await _run_vlm_inference(redis_stream, vlm_inference_request)
                if vlm_response.error_message:
                    _LOGGER.error("Error processing message %s: %s",
                                  vlm_response.request_id, vlm_response.error_message)
                    raise HTTPException(status_code=500, detail=vlm_response.error_message)
                response = _create_chat_completion_response(user_prompt, request.model, vlm_response)
            del chunk_keys

        return response

    except Exception as e:
        _LOGGER.error("Error processing request: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def sop_detection_impl(request: SopDetectionRequest):
    redis_stream = AsyncRedisStream(redis_client_for_stream)
    sop_checker_request = SopCheckerRequest(
        reply_stream_name=generate_reply_stream_name(),
        action_json=request.action_json,
        vlm_output=request.vlm_output,
        keep_alive=request.keep_alive,
        checker_id=request.checker_id,
        cycle_completion_threshold=request.options.cycle_completion_threshold,
        cycle_boundary_threshold_low=request.options.cycle_boundary_threshold_low,
        cycle_boundary_threshold_high=request.options.cycle_boundary_threshold_high,
    )

    _LOGGER.debug("Sending SopCheckerRequest to SopChecker service")
    sop_checker_response = await async_send_request_and_wait_for_response(
        redis_stream,
        REDIS_STREAM_P_API_SERVER_C_SOP_CHECKER_STREAM_NAME,
        sop_checker_request,
        SopCheckerResponse,
    )
    _LOGGER.debug(
        "SopCheckerResponse: checker_id=%s, "
        "cycle=%s, missing_detected=%s, misordered_detected=%s, "
        "final_missing_detected=%s, final_misordered_detected=%s, "
        "cycle_completed=%s, summary_cycles_detected=%s, summary_cycle_analysis=%s",
        sop_checker_response.checker_id,
        sop_checker_response.cycle,
        sop_checker_response.missing_detected,
        sop_checker_response.misordered_detected,
        sop_checker_response.final_missing_detected,
        sop_checker_response.final_misordered_detected,
        sop_checker_response.cycle_completed,
        sop_checker_response.summary_cycles_detected,
        sop_checker_response.summary_cycle_analysis,
    )

    if sop_checker_response.error_message:
        _LOGGER.error("Error processing message %s: %s",
                      sop_checker_response.request_id, sop_checker_response.error_message)
        raise HTTPException(status_code=400, detail=sop_checker_response.error_message)


    return SopDetectionResponse(
        checker_id=sop_checker_response.checker_id,
        cycle=sop_checker_response.cycle,
        missing_detected=sop_checker_response.missing_detected,
        misordered_detected=sop_checker_response.misordered_detected,
        final_missing_detected=sop_checker_response.final_missing_detected,
        final_misordered_detected=sop_checker_response.final_misordered_detected,
        cycle_completed=sop_checker_response.cycle_completed,
        summary=SopDetectionSummary(
            cycles_detected=sop_checker_response.summary_cycles_detected,
            cycle_analysis=sop_checker_response.summary_cycle_analysis,
        ),
    )


#@app.get("/")
#async def root():
#    """Health check endpoint"""
#    return {"message": "OpenAI-compatible API is running", "status": "healthy"}


CHAT_COMPLETION_API_RESPONSES = {
    200: {
        "content": {
            # The 'application/json' response for when stream=False
            "application/json": {
                "schema": ChatCompletionResponse.model_json_schema(),
                "example": {
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": "gpt-4",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "\n\nHello there! How can I assist you today?",
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 12,
                        "total_tokens": 21
                    }
                }
            },
            # The 'text/event-stream' response for when stream=True
            "text/event-stream": {
                "schema": {
                    "type": "string"
                },
                "example": "data: {\"id\":\"chatcmpl-123\",\"object\":\"chat.completion.chunk\",\"created\":1677652288,\"model\":\"gpt-4\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n"
            }
        }
    }
}


@app.post("/v1/chat/completions",
          responses=CHAT_COMPLETION_API_RESPONSES,
          response_model=None)
async def chat_completions(request: ChatCompletionRequest):
    return await chat_completions_impl(request)


@app.post("/v1/sop/detection", response_model=SopDetectionResponse)
async def sop_detection(request: SopDetectionRequest):
    return await sop_detection_impl(request)


@app.post("/v1/files", response_model=FileObject)
async def upload_file(
    file: UploadFile = File(...),
    purpose: str = Form(...)
):
    """
    Uploads a file and mimics the OpenAI file upload response.
    """

    file_size = file.size
    # Generate a unique ID for OpenAI and a separate one for MinIO
    file_uuid = uuid.uuid4()
    openai_file_id = f"{OPENAI_FILE_ID_PREFIX}{file_uuid.hex}"
    created_at = int(time.time())
    file_object = FileObject(
        id=openai_file_id,
        bytes=file_size,
        created_at=created_at,
        filename=file.filename,
        purpose=purpose,
    )

    minio_object_name = _get_minio_object_name(file_object)

    def _put_object():
        try:
            # Use put_object to upload the file from memory
            minio_client.put_object(
                MINIO_BUCKET,
                minio_object_name,
                data=file.file,
                length=file_size,
                content_type=file.content_type
            )
        except minio.S3Error as exc:
            raise HTTPException(status_code=500, detail=f"Error uploading to MinIO: {exc}")

    await asyncio.to_thread(_put_object)
    await _store_file_object(mongo_client, file_object)

    return file_object


@app.get("/v1/files/{file_id}/content")
async def get_file_content(file_id: str):
    """
    Downloads the content of a specific file.
    """

    file_object = await _fetch_file_object(mongo_client, file_id)

    try:
        # Get the object from MinIO
        minio_object_name = _get_minio_object_name(file_object)
        response = await asyncio.to_thread(minio_client.get_object, MINIO_BUCKET, minio_object_name)

        async def generate_file_content():
            try:
                for chunk in response.stream(32*1024):
                    yield chunk
            finally:
                # Close the response after streaming is complete
                response.close()
                response.release_conn()

        return StreamingResponse(
            generate_file_content(),
            media_type='application/octet-stream',
            headers={
                'Content-Disposition': f'attachment; filename="{file_object.filename}"',
                'Content-Length': str(file_object.bytes)
            }
        )

    except minio.S3Error as exc:
        raise HTTPException(status_code=500, detail=f"Error retrieving file from MinIO: {exc}")


@app.delete("/v1/files/{file_id}", response_model=DeletionStatus)
async def delete_file(file_id: str):
    """
    Deletes a file from MinIO and its metadata.
    """
    file_object = await _fetch_file_object(mongo_client, file_id)

    minio_object_name = _get_minio_object_name(file_object)

    try:
        await asyncio.to_thread(minio_client.remove_object, MINIO_BUCKET, minio_object_name)
    except minio.S3Error as exc:
        raise HTTPException(status_code=500, detail=f"Error deleting file from MinIO: {exc}")

    # Remove from our metadata store
    await _delete_file_object(mongo_client, file_id)

    return DeletionStatus(id=file_id, deleted=True)


@app.get("/v1/files", response_model=FileList)
async def list_files():
    """List all files in the database"""
    all_file_docs = _get_mongo_collection(mongo_client).find({"id": {"$regex": f"{OPENAI_FILE_ID_PREFIX}.*"}})
    file_objects = [FileObject.model_validate(file_doc) async for file_doc in all_file_docs]
    return FileList(data=file_objects)


@app.get("/v1/models")
async def list_models():
    """List available models endpoint"""
    model_id = os.environ.get("VLM_INFERENCE_MODEL_PATH_ON_HOST", "VLM_INFERENCE_MODEL_PATH_ON_HOST not set. Something wrong.")
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "sop-monitoring"
            }
        ]
    }

class _ChatCompletionRequestParser:
    def __init__(self, chat_completion_request: ChatCompletionRequest):
        if len(chat_completion_request.messages) > 2:
            raise HTTPException(
                status_code=400,
                detail=("Only one message with role 'user', one message with role 'system' are supported."
                        f"But got {len(chat_completion_request.messages)} messages: {chat_completion_request.messages}"),
            )

        user_message = None
        system_message = None

        self._image_file_id = ""
        self._user_prompt = ""
        self._system_prompt = ""
        self._action_segment_options = ""

        user_message, system_message = self._get_user_and_system_message(chat_completion_request)

        user_message_contents = user_message.content

        for content in user_message_contents:
            if content.type == "text":
                if self._user_prompt:
                    raise HTTPException(status_code=400,
                    detail=("Only one text content in user message is supported, but got at least two text contents: "
                            f"{self._user_prompt}\n"
                            f"{content.text}"),
                    )
                self._user_prompt = content.text

            elif content.type == "image_file":
                if self._image_file_id:
                    raise HTTPException(status_code=400,
                    detail=("Only one image_file content in user message is supported, but got at least two image_file contents: "
                            f"{self._image_file_id}\n"
                            f"{content.image_file.file_id}"),
                    )
                self._image_file_id = content.image_file.file_id
                self._action_segment_options = content.image_file.chunking_options
            else:
                raise HTTPException(status_code=400, detail=f"Only support type 'text' and 'image_file', but got {content.type}")

        if not self._user_prompt:
            raise HTTPException(status_code=400, detail="User prompt is required.")

        if not self._image_file_id:
            raise HTTPException(status_code=400, detail="Image file ID is required.")

        self._system_prompt = self._parse_system_prompt(system_message)
        _LOGGER.debug("Successfully parsed chat completion request:\n")

    def get_user_prompt(self) -> str:
        return self._user_prompt

    def get_image_file_id(self) -> str:
        return self._image_file_id

    def get_system_prompt(self) -> str:
        return self._system_prompt

    def get_action_segment_options(self) -> CHUNKING_OPTIONS_TYPE_HINT:
        return self._action_segment_options

    @staticmethod
    def _parse_system_prompt(system_message: Message | None) -> str:

        if not system_message:
            return ""

        if len(system_message.content) != 1:
            raise HTTPException(
                status_code=400,
                detail=("Only one system message with type 'text' is supported."
                        f"But got {len(system_message.content)} system message contents: {system_message.content}"),
            )
        system_message_content = system_message.content[0]
        if system_message_content.type != "text":
            raise HTTPException(
                status_code=400,
                detail=("Only one system message with type 'text' is supported."
                        f"But got tyep={system_message_content.type} system message contents: {system_message_content}"),
            )
        _LOGGER.debug("System prompt: %s", system_message_content.text)
        return system_message_content.text

    @staticmethod
    def _get_user_and_system_message(chat_completion_request: ChatCompletionRequest) -> tuple[Message, Message | None]:
        """
        Return:
        The tuple contains:
        [0]: user_message: Message with role 'user'
        [1]: system_message: Message with role 'system'
             None if there is no message with role 'system'
        """

        user_message = None
        system_message = None

        for message in chat_completion_request.messages:
            if message.role == "user":
                if user_message:
                    raise HTTPException(
                        status_code=400,
                        detail=("Only one message with role 'user' is supported."
                                "But got at least two messages with role 'user':\n"
                                f"{user_message}\n"
                                f"{message}"),
                    )

                user_message = message

            elif message.role == "system":
                if system_message:
                    raise HTTPException(
                        status_code=400,
                        detail=("Only one message with role 'system' is supported."
                                "But got at least two messages with role 'system':\n"
                                f"{system_message}\n"
                                f"{message}"),
                    )

                system_message = message
            else:
                raise HTTPException(status_code=400, detail=f"Only support role 'user' and 'system', but got {message.role}")

        if not user_message:
            raise HTTPException(
                status_code=400,
                detail=("Message with role 'user' is required. "
                        "Also it must contain text prompt and image_file with valid file_id."),
            )

        return user_message, system_message


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ["API_SERVER_PORT"]), # raise exception if this is not set
    )
