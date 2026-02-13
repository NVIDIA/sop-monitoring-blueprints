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
    __tablename__ = "ddm_training_job"

    id = Column(String, primary_key=True, index=True)
    aug_dataset_id = Column(String)
    status = Column(Enum(TrainingStatusEnum, name="training_status_enum"))
    total_steps = Column(Integer)
    current_step = Column(Integer)
    progress = Column(Float)
    loss = Column(Float)
    process_pid = Column(Integer)
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
            "process_pid": self.process_pid,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

