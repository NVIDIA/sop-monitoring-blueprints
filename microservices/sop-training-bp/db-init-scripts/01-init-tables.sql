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
CREATE TYPE training_status_enum AS ENUM ('queued', 'running', 'completed', 'cancelled', 'failed');

-- Annotation MS Tables

CREATE TABLE dataset (
    id VARCHAR PRIMARY KEY,
    actions VARCHAR[], -- array of actions
    two_operator_mode BOOLEAN DEFAULT FALSE,
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


CREATE TABLE ddm_training_job (
    id VARCHAR PRIMARY KEY,
    aug_dataset_id VARCHAR,
    status training_status_enum,
    total_steps INT,
    current_step INT,
    progress FLOAT,
    process_pid INT,
    loss FLOAT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE evaluation_job (
    id VARCHAR(36) PRIMARY KEY,
    training_job_id VARCHAR(36),
    val_dataset_id VARCHAR(255),
    checkpoint_step INT,
    status training_status_enum,
    overall_accuracy FLOAT,
    results_json JSON,
    fps INT,
    temperature FLOAT,
    backend VARCHAR(50),
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- E2E Evaluation Tables
-- chunking_algorithm: 'ddm' (default — uses DDM-Net for temporal segmentation)
-- or 'uniform' (fixed-length chunks of chunk_length_sec). chunk_length_sec is
-- NULL for ddm jobs and required for uniform jobs (enforced at the API layer).
CREATE TABLE e2e_evaluation_job (
    id VARCHAR(36) PRIMARY KEY,
    training_job_id VARCHAR(36),
    ddm_training_job_id VARCHAR(36),
    val_dataset_id VARCHAR(255),
    checkpoint_step INT,
    ddm_checkpoint VARCHAR(255),
    status training_status_enum,
    overall_accuracy FLOAT,
    avg_f1 FLOAT,
    results_json JSON,
    fps INT,
    temperature FLOAT,
    backend VARCHAR(50),
    score_threshold FLOAT,
    nms_sec FLOAT,
    ddm_batch_size INT,
    chunking_algorithm VARCHAR(16) DEFAULT 'ddm',
    chunk_length_sec FLOAT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);