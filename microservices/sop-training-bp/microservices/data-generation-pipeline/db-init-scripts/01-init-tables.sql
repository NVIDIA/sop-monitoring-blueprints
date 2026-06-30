-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
-- http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.


-- init-scripts/init-tables.sql

CREATE TYPE status_enum AS ENUM ('running', 'completed', 'failed', 'pending');

CREATE TABLE augmented_data (
    id VARCHAR PRIMARY KEY,
    dataset_id VARCHAR,
    parameters JSON,
    status status_enum,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE augmentation_stages (
    id VARCHAR PRIMARY KEY,
    augmentation_id VARCHAR REFERENCES augmented_data(id) ON DELETE CASCADE,
    stage_name VARCHAR NOT NULL,
    status status_enum,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    error_message TEXT
);

-- Create index for faster lookups
CREATE INDEX idx_augmentation_stages_augmentation_id ON augmentation_stages(augmentation_id);
CREATE INDEX idx_augmentation_stages_stage_name ON augmentation_stages(stage_name);

-- Add unique constraint to prevent duplicate stages per augmentation
ALTER TABLE augmentation_stages ADD CONSTRAINT unique_augmentation_stage UNIQUE (augmentation_id, stage_name);