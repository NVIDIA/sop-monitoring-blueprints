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

-- Create enum type for training status
CREATE TYPE training_status_enum AS ENUM ('queued', 'running', 'completed', 'cancelled', 'failed');


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