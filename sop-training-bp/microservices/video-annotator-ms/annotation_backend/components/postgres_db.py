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

from sqlalchemy import delete, select
from sqlalchemy import update as sqlalchemy_update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from utils.constant import POSTGRES_DB_URL


class PostgresDB:
    def __init__(self):
        self.engine = create_async_engine(POSTGRES_DB_URL, echo=True)
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

    async def insert_data(self, schema, **kwargs):
        """
        Insert a new record into the database.
        """
        # Filter out invalid fields
        valid_fields = {k: v for k, v in kwargs.items() if hasattr(schema, k)}

        if not valid_fields:
            return  # No valid fields to insert

        async with self.session() as session:
            db_job = schema(**valid_fields)
            session.add(db_job)
            await session.commit()
            return db_job

    async def update_data(self, id: str, schema, **kwargs):
        """
        Update fields of a record by id.
        kwargs: fields to update.
        """
        # Filter out invalid fields
        valid_fields = {k: v for k, v in kwargs.items() if hasattr(schema, k)}

        if not valid_fields:
            return  # No valid fields to update

        async with self.session() as session:
            await session.execute(
                sqlalchemy_update(schema).where(schema.id == id).values(**valid_fields)
            )
            await session.commit()

    async def get_data(self, id: str, schema, condition=None):
        """
        Query a record by id and condition.
        """
        async with self.session() as session:
            if condition:
                conditions = self._format_condition(schema, condition)
                result = await session.execute(
                    select(schema).where(schema.id == id, *conditions)
                )
            else:
                result = await session.execute(select(schema).where(schema.id == id))
            return result.scalar_one_or_none()

    async def list_data(self, schema, condition=None):
        """
        List all records, optionally filtered by status.
        """
        async with self.session() as session:
            stmt = select(schema)
            if condition:
                # Build where conditions properly for SQLAlchemy
                conditions = self._format_condition(schema, condition)
                stmt = stmt.where(*conditions)

            result = await session.execute(stmt)
            return result.scalars().all()

    async def delete_data(self, id, schema, condition=None):
        """
        Delete a record by id.
        Returns the number of deleted rows.
        """
        async with self.session() as session:
            if condition:
                conditions = self._format_condition(schema, condition)
                result = await session.execute(
                    delete(schema).where(schema.id == id, *conditions)
                )
            else:
                result = await session.execute(delete(schema).where(schema.id == id))
            await session.commit()
            return result.rowcount

    async def delete_all_data(self, schema, condition=None):
        """
        Delete all records from a table, optionally filtered by condition.
        Returns the number of deleted rows.
        """
        async with self.session() as session:
            if condition:
                conditions = self._format_condition(schema, condition)
                result = await session.execute(delete(schema).where(*conditions))
            else:
                result = await session.execute(delete(schema))
            await session.commit()
            return result.rowcount


postgres_db = PostgresDB()
