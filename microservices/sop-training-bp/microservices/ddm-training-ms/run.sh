#!/bin/bash
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

SERVICE_PORT=${SERVICE_PORT:-32100}
RELOAD_FLAG=${RELOAD_FLAG:-""}

echo "Starting the microservice on port $SERVICE_PORT"

if [[ "$RELOAD_FLAG" == "true" ]]; then
    echo "Running in development mode with auto-reload"
    uvicorn app:app --host 0.0.0.0 --port $SERVICE_PORT --reload
else
    echo "Running in production mode"
    uvicorn app:app --host 0.0.0.0 --port $SERVICE_PORT
fi

