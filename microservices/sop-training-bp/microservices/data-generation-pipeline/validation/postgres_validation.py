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


import enum

from sqlalchemy import JSON, TIMESTAMP, Column, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class StatusEnum(str, enum.Enum):
    running = "running"
    completed = "completed"
    failed = "failed"
    pending = "pending"


class Augmentation(Base):
    __tablename__ = "augmented_data"

    id = Column(String, primary_key=True)
    dataset_id = Column(String)
    parameters = Column(JSON)
    status = Column(Enum(StatusEnum, name="status_enum"))
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

    def __repr__(self):
        return f"<Augmentation(id={self.id}, dataset_id={self.dataset_id}, status={self.status})>"

    def to_dict(self):
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "parameters": self.parameters,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class AugmentationStage(Base):
    __tablename__ = "augmentation_stages"

    id = Column(String, primary_key=True)
    augmentation_id = Column(
        String, ForeignKey("augmented_data.id", ondelete="CASCADE")
    )
    stage_name = Column(String, nullable=False)
    status = Column(Enum(StatusEnum, name="status_enum"))
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)
    error_message = Column(Text)

    def __repr__(self):
        return f"<AugmentationStage(id={self.id}, augmentation_id={self.augmentation_id}, stage_name={self.stage_name}, status={self.status})>"

    def to_dict(self):
        return {
            "id": self.id,
            "augmentation_id": self.augmentation_id,
            "stage_name": self.stage_name,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error_message": self.error_message,
        }


class Video(Base):
    __tablename__ = "video"
    id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("dataset.id"))
    name = Column(String)
    mime_type = Column(String)
    file_size = Column(Integer)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

    def to_dict(self):
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "mime_type": self.mime_type,
            "file_size": self.file_size,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class Chunk(Base):
    __tablename__ = "chunk"
    id = Column(String, primary_key=True)
    video_id = Column(String, ForeignKey("video.id"))
    name = Column(String)
    action = Column(String)
    mime_type = Column(String)
    file_size = Column(Integer)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

    def to_dict(self):
        return {
            "id": self.id,
            "video_id": self.video_id,
            "name": self.name,
            "action": self.action,
            "mime_type": self.mime_type,
            "file_size": self.file_size,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
