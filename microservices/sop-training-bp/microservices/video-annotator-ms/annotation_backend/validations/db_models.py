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

from sqlalchemy import ARRAY, TIMESTAMP, Boolean, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Dataset(Base):
    __tablename__ = "dataset"
    id = Column(String, primary_key=True)
    actions = Column(ARRAY(String))
    two_operator_mode = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

    def to_dict(self):
        return {
            "id": self.id,
            "actions": self.actions,
            "two_operator_mode": self.two_operator_mode,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
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


class Annotation(Base):
    __tablename__ = "annotation"
    id = Column(String, primary_key=True)
    video_id = Column(String, ForeignKey("video.id"))
    chunk_id = Column(String, ForeignKey("chunk.id"))
    start_time = Column(Float)
    end_time = Column(Float)
    action_index = Column(Integer)
    action_description = Column(String)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

    def to_dict(self):
        return {
            "id": self.id,
            "video_id": self.video_id,
            "chunk_id": self.chunk_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "action_index": self.action_index,
            "action_description": self.action_description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
