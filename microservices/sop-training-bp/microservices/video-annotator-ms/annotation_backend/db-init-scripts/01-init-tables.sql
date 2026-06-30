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