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
