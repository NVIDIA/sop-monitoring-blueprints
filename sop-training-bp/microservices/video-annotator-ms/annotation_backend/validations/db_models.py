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

from sqlalchemy import ARRAY, TIMESTAMP, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Dataset(Base):
    __tablename__ = "dataset"
    id = Column(String, primary_key=True)
    actions = Column(ARRAY(String))
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

    def to_dict(self):
        return {
            "id": self.id,
            "actions": self.actions,
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
