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

# TODO: Use Redis to replace this in-memory cache

from typing import Any, Dict


class TrainingJobCache:
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}

    def get(self, job_id: str) -> Dict[str, Any]:
        return self.cache.get(job_id, None)

    def set(self, job_id: str, job_record: Dict[str, Any]):
        self.cache[job_id] = job_record

    def update(self, job_id: str, **kwargs):
        self.cache[job_id].update(kwargs)

    def delete(self, job_id: str):
        self.cache.pop(job_id, None)

    def clear(self):
        self.cache.clear()
