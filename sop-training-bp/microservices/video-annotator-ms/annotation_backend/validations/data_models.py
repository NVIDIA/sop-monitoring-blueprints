######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
######################################################################################################

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class JpegReq(BaseModel):
    file: Optional[bytes] = None


class VideoUploadResponse(BaseModel):
    message: Optional[str] = Field(
        None,
        description="Upload response message",
        examples=["Video file has been successfully uploaded and saved"],
    )
    file_id: Optional[str] = Field(
        None,
        description="ID of the saved file",
        examples=["f8a7b6c5-1234-5678-90ab-cdef01234567"],
    )


class ActionsUploadResponse(BaseModel):
    status: str = Field(..., description="Status of the upload")
    message: str = Field(..., description="Message of the upload")
    actions_count: int = Field(..., description="Number of actions")
    actions: list[str] = Field(..., description="List of actions")
    data_id: str = Field(..., description="ID of the dataset")


class ResetActionsResponse(BaseModel):
    status: str = Field(..., description="Status of the reset")
    message: str = Field(..., description="Message of the reset")
    previous_data_id: str = Field(..., description="ID of the previous dataset")


class ClearDatasetResponse(BaseModel):
    message: str = Field(..., description="Message of the clear")
    data_id: str = Field(..., description="ID of the dataset")
    deleted_count: int = Field(..., description="Number of deleted records")
    files_deleted: int = Field(..., description="Number of deleted files")

class VideoMetadata(BaseModel):
    id: str = Field(..., description="Unique ID of the video file")
    filename: str = Field(..., description="Original file name")
    file_path: str = Field(..., description="Path where file is saved on server")
    file_size: int = Field(..., description="File size in bytes")
    upload_time: datetime = Field(..., description="Upload time")
    mime_type: str = Field(default="video/mp4", description="File MIME type")
