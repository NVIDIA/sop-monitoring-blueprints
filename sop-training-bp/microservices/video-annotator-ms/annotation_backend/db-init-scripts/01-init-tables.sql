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