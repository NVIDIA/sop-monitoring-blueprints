# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
API schema types for FastAPI.

"""
import os
import time
from enum import Enum
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Annotated, Required, TypeAlias

from . import ds_logger

logger = ds_logger.get_logger(__name__)

MAX_DDM_THRESHOLD = 1.0
ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", "false").lower() in ["true", "1", "yes", "y"]
if ENABLE_PROFILING:
    MAX_DDM_THRESHOLD = 2.0
    logger.info("Profiling is enabled")
else:
    logger.info("Profiling is disabled")


class APIBaseModel(BaseModel):
    # OpenAI API does allow extra fields
    model_config = ConfigDict(extra="allow")

    # Cache class field names
    field_names: ClassVar[Optional[set[str]]] = None

    @model_validator(mode="wrap")
    @classmethod
    def __log_extra_fields__(cls, data, handler):
        result = handler(data)
        if not isinstance(data, dict):
            return result
        field_names = cls.field_names
        if field_names is None:
            # Get all class field names and their potential aliases
            field_names = set()
            for field_name, field in cls.model_fields.items():
                field_names.add(field_name)
                if alias := getattr(field, "alias", None):
                    field_names.add(alias)
            cls.field_names = field_names

        # Compare against both field names and aliases
        if any(k not in field_names for k in data):
            logger.warning(
                "The following fields were present in the request " "but ignored: %s",
                data.keys() - field_names,
            )
        return result


class ErrorInfo(APIBaseModel):
    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    param: Optional[str] = Field(None, description="Error parameter")
    code: int = Field(..., description="Error code")


class ErrorResponse(APIBaseModel):
    error: ErrorInfo = Field(..., description="Error information")


class DdmNetChunkingOptions(BaseModel):
    model_config = {"extra": "forbid"}
    algorithm: Literal["ddm-net"] = Field("ddm-net", description="Algorithm for DdmNet chunking.")
    threshold: float = Field(
        0.8,
        gt=0,
        lt=MAX_DDM_THRESHOLD,
        description="Threshold for DdmNet chunking. "
        "The lower, the more sensitive the chunking is. The value should be in (0.1, 1.0). "
        "Default is 0.8.",
    )
    min_length_sec: float = Field(
        1.0, gt=0, description="Minimum length of a chunk in seconds. " "Default is 1.0 seconds."
    )
    max_length_sec: float = Field(
        60.0,
        gt=0,
        description="Hint for maximum length of a chunk in seconds. "
        "It is not guaranteed that the chunk will be smaller than this value. "
        "But the algorithm will try to keep the chunk length below this value "
        "and not cut at the event boundaries. "
        "Default is 60 seconds.",
    )
    # batch_size: int = Field(8, ge=2, description="Batch size for DdmNet chunking. "
    #                               "Default is 8. The larger the batch size, the more memory is used. "
    #                               "But the larger the batch size, the faster the chunking is. "
    #                               "The batch size must be a power of 2. "
    #                               "The batch size must be greater than or equal to 2.")

    @model_validator(mode="after")
    def check_values(self):
        if self.max_length_sec <= self.min_length_sec:
            raise ValueError("max_length_sec must be greater than min_length_sec")
        return self


class UniformChunkingOptions(BaseModel):
    model_config = {"extra": "forbid"}
    algorithm: Literal["uniform"] = Field("uniform", description="Algorithm for uniform chunking.")
    chunk_length_sec: float = Field(
        5.0,
        gt=0,
        description="Length of each chunk in seconds. Default is 5.0 seconds.",
    )


class FileObject(APIBaseModel):
    id: str = Field(..., description="Unique file identifier with prefix 'file-'")
    object: Literal["file"] = Field("file", description="Object type, always 'file'")
    bytes: int = Field(..., description="Size of the file in bytes")
    created_at: int = Field(..., description="Unix timestamp when the file was created")
    filename: str = Field(..., description="Original filename of the uploaded file")
    purpose: str = Field(
        "",
        description="Purpose of the file upload. This field has not been well-defined yet. "
        "It's just a placeholder for now.",
    )


class FileList(APIBaseModel):
    object: Literal["list"] = Field("list", description="Object type, always 'list'")
    data: list[FileObject] = Field(..., description="List of file objects")


class DeletionStatus(APIBaseModel):
    id: str = Field(..., description="ID of the deleted file")
    object: Literal["file.deleted"] = Field("file.deleted", description="Object type, always 'file.deleted'")
    deleted: bool = Field(True, description="Whether the file was successfully deleted")


class TextContent(APIBaseModel):
    """Text content model"""

    type: Literal["text"] = Field("text", description="Content type, always 'text'")
    text: str = Field(..., description="The text content")


class VideoURL(APIBaseModel):
    url: str = Field(
        description="""Video file data in any of the three forms:
        <br>    - A URL of the video file.
        <br>    - Base64 encoded video data in form of `data:video/{format};base64,{base64encodedvideo}`.
        """
    )


class VideoURLContent(APIBaseModel):
    """Video content model"""

    type: Literal["video_url"] = Field("video_url", description="The type of the content part.")
    video_url: VideoURL = Field(
        ...,
        description="Reference to the video URL or base64 encoded video data in the form of `data:video/mp4;base64,{base64encodedvideo}`",
    )


class VideoFileContent(APIBaseModel):
    """Video content model"""

    type: Literal["input_video"] = Field("input_video", description="The type of the video data content in base64")
    file_id: str = Field(..., description="Input video file ID")


class InputCamera(APIBaseModel):
    """Input camera model"""

    camera_id: str = Field(..., description="ID of the camera")
    camera_vendor: Literal["Basler"] = Field("Basler", description="Vendor of the camera. Default is 'Basler'.")
    config: Optional[str] = Field(None, description="Configuration of the camera. This is the camera setting path")
    camera_format: Optional[Literal["RGB", "UYVY", "YUY2"]] = Field(
        None, description="Format of the camera. Default is None, which means the format is determined by the service."
    )
    camera_width: Optional[int] = Field(
        None, description="Width of the camera. Default is None, which means the width is determined by the service."
    )
    camera_height: Optional[int] = Field(
        None, description="Height of the camera. Default is None, which means the height is determined by the service."
    )
    camera_fps_num: Optional[int] = Field(
        None,
        ge=1,
        le=1e8,
        description="FPS numerator of the camera. Default is None, which means the FPS numerator is determined by the service.",
    )
    camera_fps_den: Optional[int] = Field(
        None,
        ge=1,
        le=1e8,
        description="FPS denominator of the camera. Default is None, which means the FPS denominator is determined by the service.",
    )


class CameraInputContent(APIBaseModel):
    """Camera content model"""

    type: Literal["input_camera"] = Field("input_camera", description="The type of the input_camera content.")
    input_camera: InputCamera = Field(..., description="Reference to the camera URL")


ChatMessageContent = Annotated[
    Union[
        TextContent,
        VideoURLContent,
        VideoFileContent,
        CameraInputContent,
    ],
    Field(discriminator="type"),
]


class ChatCompletionMessage(APIBaseModel):
    """Message model"""

    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'system')")
    content: Union[list[Union[ChatMessageContent, str]], str] = Field(
        None, description="List of content items (text and/or images) or a string for responses"
    )


ChunkingOptions = Annotated[Union[DdmNetChunkingOptions, UniformChunkingOptions], Field(discriminator="algorithm")]


class ChatCompletionRequest(APIBaseModel):
    """Chat completion request model"""

    messages: list[ChatCompletionMessage] = Field(..., description="List of messages in the conversation")
    model: Optional[str] = Field(
        None, description="ID of the model to use for completion. This is a placeholder for now."
    )
    max_completion_tokens: Optional[int] = Field(
        None, description="Maximum number of tokens to generate. This is a placeholder for now."
    )
    seed: Optional[int] = Field(None, ge=0, le=2**64 - 1, description="Seed for the random number generator.")
    stream: Optional[bool] = Field(False, description="Whether to stream the response.")
    temperature: Optional[float] = Field(None, description="Temperature for the random number generator.")
    top_p: Optional[float] = Field(None, description="Top-p for the random number generator.")
    chunking_options: Optional[ChunkingOptions] = Field(
        DdmNetChunkingOptions(), description="Options for the chunking algorithm. Default is DdmNetChunkingOptions."
    )


class UsageInfo(APIBaseModel):
    prompt_tokens: int = Field(0, description="The number of prompt tokens")
    total_tokens: int = Field(0, description="The number of total tokens")
    completion_tokens: Optional[int] = Field(None, description="The number of completion tokens")


class ChatCompletionResponseChoice(APIBaseModel):
    index: int = Field(..., description="Index of the choice")
    message: ChatCompletionMessage = Field(..., description="The message")
    finish_reason: Optional[str] = Field(None, description="The reason for the completion to finish")
    stop_reason: Optional[Union[int, str]] = Field(None, description="The reason for the completion to stop")
    token_ids: Optional[list[int]] = Field(None, description="The token IDs")
    chunk_metadata_list: Optional[List[Dict]] = Field(None, description="The metadata of the chunk")


class ChatCompletionResponse(APIBaseModel):
    """Chat completion response model"""

    id: str = Field(..., description="Unique identifier for the chat completion")
    object: Literal["chat.completion"] = Field("chat.completion", description="Object type, always 'chat.completion'")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field(..., description="ID of the model used for completion. This is a placeholder for now.")
    choices: list[ChatCompletionResponseChoice] = Field(..., description="List of completion choices")
    usage: Optional[UsageInfo] = Field(default=None, description="Usage statistics for the request.")


class DeltaMessage(APIBaseModel):
    """Delta message model"""

    role: Literal["user", "assistant", "system"] = Field(
        "assistant", description="The role of the message sender (e.g., 'user', 'assistant', 'system')"
    )
    content: Optional[str] = Field(None, description="The text content")


class ChatCompletionResponseStreamChoice(APIBaseModel):
    index: int = Field(0, description="Index of the choice")
    delta: DeltaMessage = Field(..., description="The delta message")
    finish_reason: Optional[str] = Field(None, description="The reason for the completion to finish")
    stop_reason: Optional[Union[int, str]] = Field(None, description="The reason for the completion to stop")
    # not part of the OpenAI spec but for tracing the tokens
    token_ids: Optional[list[int]] = Field(None, description="The token IDs")
    chunk_metadata: Optional[dict] = Field(None, description="The metadata of the chunk")


class ChatCompletionStreamResponse(APIBaseModel):
    id: str = Field(..., description="Unique identifier for the chat completion")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field(..., description="ID of the model used for completion")
    choices: list[ChatCompletionResponseStreamChoice] = Field(..., description="List of completion choices")
    usage: Optional[UsageInfo] = Field(default=None, description="Usage statistics for the request.")


class SopDetectionOptions(BaseModel):
    """SOP detection options model"""

    cycle_completion_threshold: float = Field(
        0.6,
        gt=0.1,
        le=1.0,
        description="Threshold for cycle completion. "
        "The lower, the more sensitive the cycle completion is. "
        "The value should be in (0.1, 1.0]. "
        "Default is 0.6.",
    )
    cycle_boundary_threshold_low: float = Field(
        0.3,
        gt=0.1,
        le=1.0,
        description="Threshold for cycle boundary. "
        "The higher, the more sensitive the cycle boundary is. "
        "The value should be in (0.1, 1.0]. "
        "Must be lower than `cycle_boundary_threshold_high`."
        "Default is 0.3.",
    )
    cycle_boundary_threshold_high: float = Field(
        0.8,
        gt=0.1,
        le=1.0,
        description="Threshold for cycle boundary. "
        "The lower, the more sensitive the cycle boundary is. "
        "The value should be in (0.1, 1.0]. "
        "Must be higher than `cycle_boundary_threshold_low`."
        "Default is 0.8.",
    )

    @model_validator(mode="after")
    def check_values(self):
        if self.cycle_boundary_threshold_low >= self.cycle_boundary_threshold_high:
            raise ValueError("cycle_boundary_threshold_low must be less than cycle_boundary_threshold_high")
        return self


class SopDetectionRequest(BaseModel):
    """SOP detection request model"""

    action_json: str = Field(..., description="JSON string with actions definitions")
    vlm_output: str = Field(..., description="Output from the VLM inference service")
    keep_alive: bool = Field(False, description="Whether to keep the checker and use it for the next request")
    checker_id: str = Field(
        "*",
        description="ID of the checker to use for the next request. "
        "The default value '*' means a new checker would be created. ",
    )
    options: SopDetectionOptions = Field(SopDetectionOptions(), description="Options for the SOP detection. ")


class SopDetectionSummary(BaseModel):
    """SOP detection summary model"""

    cycles_detected: list[str] = Field(..., description="List of detected action indexes for each cycle")
    cycle_analysis: list[str] = Field(..., description="Analysis of the cycles")


class SopDetectionResponse(BaseModel):
    """SOP detection response model"""

    checker_id: str = Field(
        "*",
        description="ID of the checker used for the next request. "
        "This field is only available if `keep_alive` is True.",
    )
    cycle: int = Field(..., description="Current cycle of the SOP")
    missing_detected: list[int] = Field(
        ...,
        description="List of missing actions. "
        "This field is accumulated from the vlm_output in the corresponding request."
        "Please note that if it's possible that results from multiple cycles are appended to this field.",
    )
    misordered_detected: list[int] = Field(
        ...,
        description="List of misordered actions. "
        "This field is accumulated from the vlm_output in the corresponding request."
        "Please note that if it's possible that results from multiple cycles are appended to this field.",
    )
    final_missing_detected: list[int] = Field(
        ...,
        description="List of missing actions. "
        "This field is only available when the `keep_alive` is False in the corresponding request. "
        "The list is the missing actions in the final cycle.",
    )
    final_misordered_detected: list[int] = Field(
        ...,
        description="List of misordered actions. "
        "This field is only available when the `keep_alive` is False in the corresponding request. "
        "The list is the misordered actions in the final cycle.",
    )
    cycle_completed: bool = Field(
        ...,
        description="Whether the current cycle is completed. "
        "Note that this field indicate the status of the final cycle in the vlm output.",
    )

    summary: SopDetectionSummary = Field(
        ...,
        description="Summary of the SOP detection. "
        "Only available when the `keep_alive` is False in the corresponding request. ",
    )


class LicenseInfoResponse(BaseModel):
    name: str = Field(description="The name of the NIMs license.")
    path: str = Field(description="The path of the NIMs license file inside the container.")
    size: int = Field(description="The size, in bytes, of the license file.")
    url: str = Field(description="The url of the NIMs license.")
    type: str = Field(description="The content type of the license.")
    content: str = Field(description="The content inside the NIMs license file.")


class DSSOPMetadataResponse(APIBaseModel):
    """DS SOP metadata response model"""

    model_config = ConfigDict(protected_namespaces=())

    version: str = Field("1.0.0", description="Version of the DS SOP")
    modelInfo: Dict[str, Any] = Field(..., description="List of model information")
    licenseInfo: LicenseInfoResponse = Field(..., description="License information")


class HealthSuccessResponse(APIBaseModel):
    object: Literal["health.response"] = Field("health.response", description="Object type, always 'health.response'")
    message: str = Field(..., description="Message of the health response")
