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
from validation.postgres_validation import Augmentation


class PostgresDB:
    def __init__(self):
        self.engine = create_async_engine(POSTGRES_DB_URL, echo=False)
        self.session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    def _format_condition(self, schema, conditions):
        """
        Format condition for SQLAlchemy.
        """
        # Build where conditions properly for SQLAlchemy
        formated_conditions = []
        for key, value in conditions.items():
            if hasattr(schema, key):
                formated_conditions.append(getattr(schema, key) == value)
        return formated_conditions

    async def insert_data(self, schema=None, **kwargs):
        """
        Insert a new record into the database.
        """
        if schema is None:
            schema = Augmentation

        # Filter out invalid fields
        valid_fields = {k: v for k, v in kwargs.items() if hasattr(schema, k)}

        if not valid_fields:
            return  # No valid fields to insert

        async with self.session() as session:
            db_job = schema(**valid_fields)
            session.add(db_job)
            await session.commit()
            return db_job

    async def update_data(self, id: str, schema=None, condition=None, **kwargs):
        """
        Update fields of a record by id.
        kwargs: fields to update.
        """
        if schema is None:
            schema = Augmentation

        # Filter out invalid fields
        valid_fields = {k: v for k, v in kwargs.items() if hasattr(schema, k)}

        if not valid_fields:
            return  # No valid fields to update

        async with self.session() as session:
            if condition:
                condition = self._format_condition(schema, condition)
                await session.execute(
                    sqlalchemy_update(schema)
                    .where(schema.id == id, *condition)
                    .values(**valid_fields)
                )
            else:
                await session.execute(
                    sqlalchemy_update(schema)
                    .where(schema.id == id)
                    .values(**valid_fields)
                )
            await session.commit()

    async def get_data(self, id: str, schema=None, condition=None):
        """
        Query a record by id and condition.
        """
        if schema is None:
            schema = Augmentation

        async with self.session() as session:
            if condition:
                conditions = self._format_condition(schema, condition)
                result = await session.execute(
                    select(schema).where(schema.id == id, *conditions)
                )
            else:
                result = await session.execute(select(schema).where(schema.id == id))
            return result.scalar_one_or_none()

    async def list_data(self, schema=None, condition=None):
        """
        List all records, optionally filtered by status.
        """
        if schema is None:
            schema = Augmentation

        async with self.session() as session:
            stmt = select(schema)
            if condition:
                conditions = self._format_condition(schema, condition)
                stmt = stmt.where(*conditions)

            result = await session.execute(stmt)
            return result.scalars().all()


postgres_db = PostgresDB()
