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

"""
Conftest for ddm-training-ms integration tests - ensures correct module imports.
"""

import sys
from pathlib import Path

# Clear any cached modules to ensure correct imports
modules_to_clear = [key for key in sys.modules.keys() if key.startswith(("utils", "components", "app", "validation"))]
for mod in modules_to_clear:
    del sys.modules[mod]

# Ensure ddm-training-ms is at the front of sys.path for this test directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MS_PATH = str(PROJECT_ROOT / "microservices" / "ddm-training-ms")

# Remove any other microservice paths that might conflict
paths_to_remove = [p for p in sys.path if "microservices" in p and "ddm-training-ms" not in p]
for p in paths_to_remove:
    if p in sys.path:
        sys.path.remove(p)

# Ensure ddm-training-ms is at the front
if MS_PATH in sys.path:
    sys.path.remove(MS_PATH)
sys.path.insert(0, MS_PATH)

# Add DDM-Net to sys.path so tests can use the same relative imports as
# train_sop_lightning.py (e.g. `from config.config import ...`)
DDM_NET_PATH = str(PROJECT_ROOT / "microservices" / "ddm-training-ms" / "ddm" / "DDM-Net")
if DDM_NET_PATH in sys.path:
    sys.path.remove(DDM_NET_PATH)
sys.path.insert(0, DDM_NET_PATH)
