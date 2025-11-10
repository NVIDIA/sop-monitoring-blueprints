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
Pydantic models for FastAPI.

These models are used to validate the request and response bodies for the API endpoints.

The models are defined in this file, and used in other files.
"""
from enum import Enum
from typing import Literal
from pydantic import (
    BaseModel,
    Field,
    model_validator,
)

from .constants import (
    CHUNK_ALGO_UNIFORM_NAME,
    CHUNK_ALGO_UBOCO_NAME,
    CHUNK_ALGO_DDM_NET_NAME,
    CHUNK_ALGO_COSMOS_REASON_NAME,
)

class FileObject(BaseModel):
    id: str = Field(..., description="Unique file identifier with prefix 'file-'")
    object: str = Field("file", description="Object type, always 'file'")
    bytes: int = Field(..., description="Size of the file in bytes")
    created_at: int = Field(..., description="Unix timestamp when the file was created")
    filename: str = Field(..., description="Original filename of the uploaded file")
    purpose: str = Field("", description="Purpose of the file upload. This field has not been well-defined yet. "
                                           "It's just a placeholder for now.")

class FileList(BaseModel):
    object: str = Field("list", description="Object type, always 'list'")
    data: list[FileObject] = Field(..., description="List of file objects")

class DeletionStatus(BaseModel):
    id: str = Field(..., description="ID of the deleted file")
    object: str = Field("file.deleted", description="Object type, always 'file.deleted'")
    deleted: bool = Field(..., description="Whether the file was successfully deleted")

#
# begin of chunking algorithm options
#
class ChunkingAlgoName(str, Enum):
    UNIFORM = CHUNK_ALGO_UNIFORM_NAME
    UBOCO = CHUNK_ALGO_UBOCO_NAME
    DDM_NET = CHUNK_ALGO_DDM_NET_NAME
    COSMOS_REASON = CHUNK_ALGO_COSMOS_REASON_NAME

_UNIFORM_CHUNK_LENGTH_DEFAULT = 2.0
class UniformChunkingOptions(BaseModel):
    model_config = {
        "extra": "forbid"
    }
    algorithm: Literal[ChunkingAlgoName.UNIFORM]
    chunk_length: float = Field(_UNIFORM_CHUNK_LENGTH_DEFAULT,
                                ge=0.5,
                                description="Length of each chunk in seconds. "
                                            "Must be larger than 0.5 seconds. "
                                            f"Default is {_UNIFORM_CHUNK_LENGTH_DEFAULT} seconds.")

class UboCoChunkingOptions(BaseModel):
    model_config = {
        "extra": "forbid"
    }
    algorithm: Literal[ChunkingAlgoName.UBOCO]
    is_deterministic: bool = Field(True, description="Whether to use deterministic behavior for reproducible results. Default is True.")
    return_cached_model: bool = Field(False, description="Whether to return cached models for reuse. Default is False.")
    extracted_fps: float = Field(3.0, gt=0, description="Frame rate for feature extraction. This controls clip_len (clip_len = 1/extracted_fps). Default is 3.0 fps.")
    min_segment_seconds: float = Field(2.0, gt=0, description="Minimum segment duration in seconds for RTP algorithm. Controls the minimum duration of segments that can be further divided. Default is 2.0 seconds.")
    threshold: float = Field(0.2, ge=0.0, le=1.0, description="Score threshold for boundary detection in RTP algorithm. Higher values result in fewer boundaries. Default is 0.2.")

class DdmNetChunkingOptions(BaseModel):
    model_config = {
        "extra": "forbid"
    }
    algorithm: Literal[ChunkingAlgoName.DDM_NET]
    threshold: float = Field(0.6, gt=0.1, lt=1.0, description="Threshold for DdmNet chunking. "
                             "The lower, the more sensitive the chunking is.The value should be in (0.1, 1.0)"
                             "Default is 0.6.")
    min_length_sec: float = Field(1.0,
                                  gt=0,
                                  deprecated=True,
                                  description="This field is deprecated, no longer effective, "
                                              "and will be removed in the next release. "
                                              "Please use `nms_sec` instead.")
    max_length_sec: float = Field(60.0,
                                  gt=0,
                                  deprecated=True,
                                  description="This field is deprecated, no longer effective, "
                                              "and will be removed in the next release. "
                                              "Please use `nms_sec` instead.")
    nms_sec: float = Field(0.0,
                           ge=0.0,
                           description="The half-length of the window to perform non-maximun suppression. "
                                       "The default value is roughly 0.025 * video length in seconds")
    batch_size: int = Field(8, ge=2, description="Batch size for DdmNet chunking. "
                                  "Default is 8. The larger the batch size, the more memory is used. "
                                  "But the larger the batch size, the faster the chunking is. "
                                  "The batch size must be a power of 2. "
                                  "The batch size must be greater than or equal to 2.")

class CosmosReasonChunkingOptions(BaseModel):
    model_config = {
        "extra": "forbid"
    }
    algorithm: Literal[ChunkingAlgoName.COSMOS_REASON]
    user_prompt: str = Field(..., description="User prompt for the Cosmos Reason model.")
    system_prompt: str = Field(..., description="System prompt for the Cosmos Reason model.")
    chunk_duration_sec: float = Field(5.0, ge=2.0, description="Duration of each chunk in seconds. "
                                  "Default is 5.0 seconds. "
                                  "The duration should be equal to or larger than 2.0 seconds.")
    min_length_sec: float = Field(0.1, gt=0.0, description="Minimum length of a chunk in seconds. "
                                  "Default is 0.1 seconds.")

    @model_validator(mode="after")
    def check_values(self):
        if self.min_length_sec >= self.chunk_duration_sec:
            raise ValueError("min_length_sec must be less than chunk_duration_sec")
        return self
#
# end of chunking algorithm options
#

CHUNKING_OPTIONS_TYPE_HINT = (
    UniformChunkingOptions
    | UboCoChunkingOptions
    | DdmNetChunkingOptions
    #| CosmosReasonChunkingOptions # FIXME: enable this when we have the model weights.
)
class ImageFile(BaseModel):
    """Image file model"""
    file_id: str = Field(..., description="ID of the uploaded image file")
    chunking_options: CHUNKING_OPTIONS_TYPE_HINT = Field(
        UniformChunkingOptions(algorithm=ChunkingAlgoName.UNIFORM),
        description=(
            "Options for the chunking algorithm. "
            "Options are algorithm-dependent, but must contain the 'algorithm' field. "
            f"'algorithm' must be one of {ChunkingAlgoName.UNIFORM.value}, {ChunkingAlgoName.UBOCO.value}, "
            f"{ChunkingAlgoName.DDM_NET.value}. Default is {ChunkingAlgoName.UNIFORM.value}."
        ),
        discriminator="algorithm",
    )

class TextContent(BaseModel):
    """Text content model"""
    type: str = Field("text", description="Content type, always 'text'")
    text: str = Field(..., description="The text content")


class ImageContent(BaseModel):
    """Image content model"""
    type: str = Field("image_file", description="Content type, always 'image_file'")
    image_file: ImageFile = Field(..., description="Reference to the image file")


class Message(BaseModel):
    """Message model"""
    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'system')")
    #content: list[TextContent | ImageContent] | str
    content: list[TextContent | ImageContent] = Field(..., description="List of content items (text and/or images)")


class ChatCompletionRequest(BaseModel):
    """Chat completion request model"""
    model: str = Field(..., description="ID of the model to use for completion. This is a placeholder for now.")
    messages: list[Message] = Field(..., description="List of messages in the conversation")
    stream: bool = Field(False, description="Whether to stream the response")


class ChatCompletionResponse(BaseModel):
    """Chat completion response model"""
    id: str = Field(..., description="Unique identifier for the chat completion")
    object: str = Field("chat.completion", description="Object type, always 'chat.completion'")
    created: int = Field(..., description="Unix timestamp when the response was created")
    model: str = Field(..., description="ID of the model used for completion. This is a placeholder for now.")
    choices: list[dict] = Field(..., description="List of completion choices")
    usage: dict = Field(..., description="Usage statistics for the request. This field is not well-defined yet.")


class ChatCompletionStreamResponseDelta(BaseModel):
    """Chat completion stream response delta model"""
    content: str | None = Field(..., description="The text content")


class CreateChatCompletionStreamResponse(BaseModel):
    """Create chat completion stream response model"""
    id: str = Field(..., description="Unique identifier for the chat completion")
    object: str = Field("chat.completion.chunk", description="Object type, always 'chat.completion.chunk'")
    created: int = Field(..., description="Unix timestamp when the response was created")
    model: str = Field(..., description="ID of the model used for completion")
    choices: list[dict] = Field(..., description="List of completion choices")

class SopDetectionOptions(BaseModel):
    """SOP detection options model"""
    cycle_completion_threshold: float = Field(
        0.6,
        gt=0.1,
        le=1.0,
        description="Threshold for cycle completion. "
                    "The lower, the more sensitive the cycle completion is. "
                    "The value should be in (0.1, 1.0]. "
                    "Default is 0.6.")
    cycle_boundary_threshold_low: float = Field(
        0.3,
        gt=0.1,
        le=1.0,
        description="Threshold for cycle boundary. "
                    "The higher, the more sensitive the cycle boundary is. "
                    "The value should be in (0.1, 1.0]. "
                    "Must be lower than `cycle_boundary_threshold_high`."
                    "Default is 0.3.")
    cycle_boundary_threshold_high: float = Field(
        0.8,
        gt=0.1,
        le=1.0,
        description="Threshold for cycle boundary. "
                    "The lower, the more sensitive the cycle boundary is. "
                    "The value should be in (0.1, 1.0]. "
                    "Must be higher than `cycle_boundary_threshold_low`."
                    "Default is 0.8.")

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
    checker_id: str = Field("*", description="ID of the checker to use for the next request. "
                                             "The default value '*' means a new checker would be created. ")
    options: SopDetectionOptions = Field(
        SopDetectionOptions(),
        description="Options for the SOP detection. ")

class SopDetectionSummary(BaseModel):
    """SOP detection summary model"""
    cycles_detected: list[str] = Field(
        ...,
        description="List of detected action indexes for each cycle")
    cycle_analysis: list[str] = Field(
        ...,
        description="Analysis of the cycles")

class SopDetectionResponse(BaseModel):
    """SOP detection response model"""
    checker_id: str = Field("*", description="ID of the checker used for the next request. "
                                             "This field is only available if `keep_alive` is True.")
    cycle: int = Field(..., description="Current cycle of the SOP")
    missing_detected: list[int] = Field(
        ...,
        description="List of missing actions. "
                    "This field is accumulated from the vlm_output in the corresponding request."
                    "Please note that if it's possible that results from multiple cycles are appended to this field.")
    misordered_detected: list[int] = Field(
        ...,
        description="List of misordered actions. "
                    "This field is accumulated from the vlm_output in the corresponding request."
                    "Please note that if it's possible that results from multiple cycles are appended to this field.")
    final_missing_detected: list[int] = Field(
        ...,
        description="List of missing actions. "
                    "This field is only available when the `keep_alive` is False in the corresponding request. "
                    "The list is the missing actions in the final cycle.")
    final_misordered_detected: list[int] = Field(
        ...,
        description="List of misordered actions. "
                    "This field is only available when the `keep_alive` is False in the corresponding request. "
                    "The list is the misordered actions in the final cycle.")
    cycle_completed: bool = Field(
        ...,
        description="Whether the current cycle is completed. "
                    "Note that this field indicate the status of the final cycle in the vlm output.")

    summary: SopDetectionSummary = Field(
        ...,
        description="Summary of the SOP detection. "
                    "Only available when the `keep_alive` is False in the corresponding request. ")

