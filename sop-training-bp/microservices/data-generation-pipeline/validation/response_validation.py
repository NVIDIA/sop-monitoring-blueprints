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

from pydantic import BaseModel


class AugResponse(BaseModel):
    """Request model for VLM data augmentation"""

    dataset_id: str
    message: Optional[str] = "Augmentation actions submitted successfully"


class StageStatus(BaseModel):
    """Model for individual stage status"""

    stage_name: str
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class AugmentationStatusResponse(BaseModel):
    """Response model for augmentation status endpoint"""

    dataset_id: str
    status: str
    progress: float
