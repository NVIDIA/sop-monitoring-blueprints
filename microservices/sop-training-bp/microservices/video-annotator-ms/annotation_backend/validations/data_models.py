######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
