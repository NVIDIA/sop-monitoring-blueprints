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

# Mirrors tables owned by OTHER microservices (TrainingJob, DdmTrainingJob,
# Augmentation) for read-only ORM access. NEVER call
# Base.metadata.create_all(engine) here — the authoritative DDL lives in
# db-init-scripts/01-init-tables.sql and the owning services.

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
    """READ-ONLY mirror: the `training_job` table is owned by cr-training-ms.

    Defined here only for convenience cross-service reads (resolving a
    VLM checkpoint by training_job_id). Never INSERT/UPDATE through this
    class from evaluation-ms.
    """
    __tablename__ = "training_job"

    id = Column(String, primary_key=True, index=True)
    aug_dataset_id = Column(String)
    status = Column(Enum(TrainingStatusEnum, name="training_status_enum", create_constraint=False))
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


class DdmTrainingJob(Base):
    """READ-ONLY mirror: `ddm_training_job` is owned by ddm-training-ms.

    Defined here only for convenience cross-service reads (validating that
    the user-selected DDM training job exists and is completed before
    starting an e2e eval). Never INSERT/UPDATE through this class from
    evaluation-ms.

    The full column list mirrors db-init-scripts/01-init-tables.sql so an
    accidental `obj.process_pid` read returns ``None`` (the SQL value)
    instead of crashing with AttributeError. If ddm-training-ms grows the
    table later, this declaration needs the same update to stay accurate.
    """
    __tablename__ = "ddm_training_job"

    id = Column(String, primary_key=True, index=True)
    aug_dataset_id = Column(String)
    status = Column(Enum(TrainingStatusEnum, name="training_status_enum", create_constraint=False))
    total_steps = Column(Integer)
    current_step = Column(Integer)
    progress = Column(Float)
    process_pid = Column(Integer)
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
            "process_pid": self.process_pid,
            "loss": self.loss,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class Augmentation(Base):
    """READ-ONLY mirror: `augmented_data` is owned by data-generation-pipeline.

    Defined here only to resolve the original (pre-augmentation) dataset_id
    when a user selects an augmented dataset for evaluation. Mirrors only
    the two columns we read; the source-of-truth ORM with the full schema
    lives in the data-generation-pipeline service.
    """
    __tablename__ = "augmented_data"

    id = Column(String, primary_key=True, index=True)
    dataset_id = Column(String)


class EvaluationJob(Base):
    __tablename__ = "evaluation_job"

    id = Column(String, primary_key=True, index=True)
    training_job_id = Column(String)
    val_dataset_id = Column(String)
    checkpoint_step = Column(Integer)
    status = Column(Enum(TrainingStatusEnum, name="training_status_enum", create_constraint=False))
    overall_accuracy = Column(Float)
    results_json = Column(JSON)
    fps = Column(Integer)
    temperature = Column(Float)
    backend = Column(String)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

    def to_dict(self):
        return {
            "id": self.id,
            "training_job_id": self.training_job_id,
            "val_dataset_id": self.val_dataset_id,
            "checkpoint_step": self.checkpoint_step,
            "status": self.status,
            "overall_accuracy": self.overall_accuracy,
            "results_json": self.results_json,
            "fps": self.fps,
            "temperature": self.temperature,
            "backend": self.backend,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class E2eEvaluationJob(Base):
    __tablename__ = "e2e_evaluation_job"

    id = Column(String, primary_key=True, index=True)
    training_job_id = Column(String)
    ddm_training_job_id = Column(String)
    val_dataset_id = Column(String)
    checkpoint_step = Column(Integer)
    ddm_checkpoint = Column(String)
    status = Column(Enum(TrainingStatusEnum, name="training_status_enum", create_constraint=False))
    overall_accuracy = Column(Float)
    avg_f1 = Column(Float)
    results_json = Column(JSON)
    fps = Column(Integer)
    temperature = Column(Float)
    backend = Column(String)
    score_threshold = Column(Float)
    nms_sec = Column(Float)
    ddm_batch_size = Column(Integer)
    # Columns (not just results_json) so list_jobs renders chunking
    # metadata without parsing JSON per row. NULL chunk_length_sec for ddm.
    chunking_algorithm = Column(String)
    chunk_length_sec = Column(Float)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

    def to_dict(self):
        return {
            "id": self.id,
            "training_job_id": self.training_job_id,
            "ddm_training_job_id": self.ddm_training_job_id,
            "val_dataset_id": self.val_dataset_id,
            "checkpoint_step": self.checkpoint_step,
            "ddm_checkpoint": self.ddm_checkpoint,
            "status": self.status,
            "overall_accuracy": self.overall_accuracy,
            "avg_f1": self.avg_f1,
            "results_json": self.results_json,
            "fps": self.fps,
            "temperature": self.temperature,
            "backend": self.backend,
            "score_threshold": self.score_threshold,
            "nms_sec": self.nms_sec,
            "ddm_batch_size": self.ddm_batch_size,
            "chunking_algorithm": self.chunking_algorithm,
            "chunk_length_sec": self.chunk_length_sec,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
