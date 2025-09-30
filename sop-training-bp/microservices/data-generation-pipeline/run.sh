#!/bin/bash
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



SERVICE_PORT=${SERVICE_PORT:-5487}
RELOAD_FLAG=${RELOAD_FLAG:-""}

echo "Starting the microservice on port $SERVICE_PORT"

if [ "$RELOAD_FLAG" = "true" ]; then
    echo "Running in development mode with auto-reload"
    uvicorn app:app --host 0.0.0.0 --port $SERVICE_PORT --reload
else
    echo "Running in production mode"
    uvicorn app:app --host 0.0.0.0 --port $SERVICE_PORT
fi