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

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class FineTuningResponse(BaseModel):
    job_id: str
    status: str
    message: str
    created_at: datetime


class TrainingStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    loss: Optional[float] = None
    created_at: datetime
    updated_at: datetime

