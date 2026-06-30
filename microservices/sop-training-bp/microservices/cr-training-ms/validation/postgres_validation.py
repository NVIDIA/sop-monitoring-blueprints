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

from sqlalchemy import JSON, TIMESTAMP, Column, Enum, Float, Integer, String
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class TrainingStatusEnum(str, enum.Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    cancelled = "cancelled"
    failed = "failed"


class TrainingJob(Base):
    __tablename__ = "training_job"

    id = Column(String, primary_key=True, index=True)
    aug_dataset_id = Column(String)
    status = Column(Enum(TrainingStatusEnum, name="training_status_enum"))
    total_steps = Column(Integer)
    current_step = Column(Integer)
    progress = Column(Float)
    loss = Column(Float)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

    def to_dict(self):
        return {
            "id": self.id,
            "aug_dataset_id": self.aug_dataset_id,
            "status": self.status,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "progress": self.progress,
            "loss": self.loss,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
