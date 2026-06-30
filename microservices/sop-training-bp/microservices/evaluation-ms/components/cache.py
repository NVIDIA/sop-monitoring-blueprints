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

# TODO: Use Redis to replace this in-memory cache

from typing import Any, Dict


class JobCache:
    """In-memory cache of evaluation/e2e-evaluation job records keyed by job_id."""

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
