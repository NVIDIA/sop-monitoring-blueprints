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

from sqlalchemy import select
from sqlalchemy import update as sqlalchemy_update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from utils.constant import POSTGRES_DB_URL
from validation.postgres_validation import (
    Augmentation,
    DdmTrainingJob,
    E2eEvaluationJob,
    EvaluationJob,
    TrainingJob,
)


class PostgresDB:
    def __init__(self):
        self.engine = create_async_engine(POSTGRES_DB_URL, echo=False)
        self.session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async def get_training_job(self, job_id):
        """
        Query a training job by job_id.
        """
        async with self.session() as session:
            result = await session.execute(select(TrainingJob).where(TrainingJob.id == job_id))
            return result.scalar_one_or_none()

    async def list_training_jobs(self, status=None):
        """
        List all training jobs, optionally filtered by status.
        """
        async with self.session() as session:
            stmt = select(TrainingJob)
            if status:
                stmt = stmt.where(TrainingJob.status == status)
            result = await session.execute(stmt)
            return result.scalars().all()

    async def insert_evaluation_job(self, **kwargs):
        """
        Insert a new evaluation job into the database.
        kwargs should match the EvaluationJob model fields.
        """
        valid_fields = {k: v for k, v in kwargs.items() if hasattr(EvaluationJob, k)}
        if not valid_fields:
            return
        async with self.session() as session:
            db_job = EvaluationJob(**valid_fields)
            session.add(db_job)
            await session.commit()
            return db_job

    async def update_evaluation_job(self, eval_job_id, **kwargs):
        """
        Update fields of an evaluation job by eval_job_id.
        kwargs: fields to update.
        """
        valid_fields = {k: v for k, v in kwargs.items() if hasattr(EvaluationJob, k)}
        if not valid_fields:
            return
        async with self.session() as session:
            await session.execute(
                sqlalchemy_update(EvaluationJob)
                .where(EvaluationJob.id == eval_job_id)
                .values(**valid_fields)
            )
            await session.commit()

    async def get_evaluation_job(self, eval_job_id):
        """
        Query an evaluation job by eval_job_id.
        """
        async with self.session() as session:
            result = await session.execute(
                select(EvaluationJob).where(EvaluationJob.id == eval_job_id)
            )
            return result.scalar_one_or_none()

    async def list_evaluation_jobs(self):
        """
        List all evaluation jobs.
        """
        async with self.session() as session:
            result = await session.execute(select(EvaluationJob))
            return result.scalars().all()

    async def get_original_dataset_id(self, aug_dataset_id: str):
        """
        Look up the original (pre-augmentation) dataset_id from the
        augmented_data table (owned by data-generation-pipeline).
        """
        async with self.session() as session:
            result = await session.execute(
                select(Augmentation.dataset_id).where(Augmentation.id == aug_dataset_id)
            )
            return result.scalar_one_or_none()

    # --- E2E Evaluation CRUD ---

    async def insert_e2e_evaluation_job(self, **kwargs):
        valid_fields = {k: v for k, v in kwargs.items() if hasattr(E2eEvaluationJob, k)}
        if not valid_fields:
            return
        async with self.session() as session:
            db_job = E2eEvaluationJob(**valid_fields)
            session.add(db_job)
            await session.commit()
            return db_job

    async def update_e2e_evaluation_job(self, eval_job_id, **kwargs):
        valid_fields = {k: v for k, v in kwargs.items() if hasattr(E2eEvaluationJob, k)}
        if not valid_fields:
            return
        async with self.session() as session:
            await session.execute(
                sqlalchemy_update(E2eEvaluationJob)
                .where(E2eEvaluationJob.id == eval_job_id)
                .values(**valid_fields)
            )
            await session.commit()

    async def get_e2e_evaluation_job(self, eval_job_id):
        async with self.session() as session:
            result = await session.execute(
                select(E2eEvaluationJob).where(E2eEvaluationJob.id == eval_job_id)
            )
            return result.scalar_one_or_none()

    async def list_e2e_evaluation_jobs(self):
        async with self.session() as session:
            result = await session.execute(select(E2eEvaluationJob))
            return result.scalars().all()

    async def get_ddm_training_job(self, job_id):
        """
        Query the ddm_training_job table (owned by ddm-training-ms) to
        validate the DDM job exists and to read its status.
        """
        async with self.session() as session:
            result = await session.execute(
                select(DdmTrainingJob).where(DdmTrainingJob.id == job_id)
            )
            return result.scalar_one_or_none()

    async def list_ddm_training_jobs(self, status=None):
        """List DDM training jobs, optionally filtered by status."""
        async with self.session() as session:
            stmt = select(DdmTrainingJob)
            if status:
                stmt = stmt.where(DdmTrainingJob.status == status)
            result = await session.execute(stmt)
            return result.scalars().all()


postgres_db = PostgresDB()
