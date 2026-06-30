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
from validation.postgres_validation import TrainingJob


class PostgresDB:
    def __init__(self):
        self.engine = create_async_engine(POSTGRES_DB_URL, echo=False)
        self.session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async def insert_training_job(self, **kwargs):
        """
        Insert a new training job into the database.
        kwargs should match the TrainingJob model fields.
        """
        # Filter out invalid fields
        valid_fields = {k: v for k, v in kwargs.items() if hasattr(TrainingJob, k)}

        if not valid_fields:
            return  # No valid fields to insert

        async with self.session() as session:
            db_job = TrainingJob(**valid_fields)
            session.add(db_job)
            await session.commit()
            return db_job

    async def update_training_job(self, job_id, **kwargs):
        """
        Update fields of a training job by job_id.
        kwargs: fields to update.
        """
        # Filter out invalid fields
        valid_fields = {k: v for k, v in kwargs.items() if hasattr(TrainingJob, k)}

        if not valid_fields:
            return  # No valid fields to update

        async with self.session() as session:
            await session.execute(
                sqlalchemy_update(TrainingJob).where(TrainingJob.id == job_id).values(**valid_fields)
            )
            await session.commit()

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


postgres_db = PostgresDB()

