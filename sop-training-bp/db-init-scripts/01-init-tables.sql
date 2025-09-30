-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: LicenseRef-NvidiaProprietary
--
-- NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
-- property and proprietary rights in and to this material, related
-- documentation and any modifications thereto. Any use, reproduction,
-- disclosure or distribution of this material and related documentation
-- without an express license agreement from NVIDIA CORPORATION or
-- its affiliates is strictly prohibited.

-- init-scripts/init-tables.sql
CREATE TYPE status_enum AS ENUM ('running', 'completed', 'failed', 'pending');
CREATE TYPE training_status_enum AS ENUM ('queued', 'running', 'completed', 'cancelled', 'failed');

-- Annotation MS Tables

CREATE TABLE dataset (
    id VARCHAR PRIMARY KEY,
    actions VARCHAR[], -- array of actions
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE video (
    id VARCHAR PRIMARY KEY,
    dataset_id VARCHAR REFERENCES dataset(id) ON DELETE CASCADE,
    name VARCHAR,
    mime_type VARCHAR,
    file_size INT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE chunk (
    id VARCHAR PRIMARY KEY,
    video_id VARCHAR REFERENCES video(id) ON DELETE CASCADE,
    name VARCHAR,
    action VARCHAR,
    mime_type VARCHAR,
    file_size INT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE annotation (
    id VARCHAR PRIMARY KEY,
    video_id VARCHAR REFERENCES video(id) ON DELETE CASCADE,
    chunk_id VARCHAR REFERENCES chunk(id) ON DELETE CASCADE,
    start_time FLOAT,
    end_time FLOAT,
    action_index INT,
    action_description VARCHAR,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);


-- Augmentation MS Tables

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


-- Training MS Tables
CREATE TABLE training_job (
    id VARCHAR PRIMARY KEY,
    aug_dataset_id VARCHAR,
    status training_status_enum,
    total_steps INT,
    current_step INT,
    progress FLOAT,
    loss FLOAT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);