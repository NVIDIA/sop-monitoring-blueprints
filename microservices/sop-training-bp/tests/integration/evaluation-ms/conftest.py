######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Conftest for evaluation-ms unit tests — ensures correct module imports."""

import sys
from pathlib import Path

# Drop any modules cached under the top-level names we are about to shadow,
# so imports resolve against evaluation-ms and not a previously-tested MS.
modules_to_clear = [
    key for key in sys.modules.keys()
    if key.startswith(("utils", "components", "validation", "sop"))
]
for mod in modules_to_clear:
    del sys.modules[mod]

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MS_PATH = str(PROJECT_ROOT / "microservices" / "evaluation-ms")

# Remove any other microservice paths that might still be on sys.path.
paths_to_remove = [
    p for p in sys.path
    if "microservices" in p and "evaluation-ms" not in p
]
for p in paths_to_remove:
    if p in sys.path:
        sys.path.remove(p)

# Ensure evaluation-ms is at the front of sys.path.
if MS_PATH in sys.path:
    sys.path.remove(MS_PATH)
sys.path.insert(0, MS_PATH)
