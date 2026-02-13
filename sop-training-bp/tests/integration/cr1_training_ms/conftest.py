######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
######################################################################################################

"""
Conftest for cr1_training_ms integration tests - ensures correct module imports.
"""

import sys
from pathlib import Path

# Clear any cached modules to ensure correct imports
modules_to_clear = [key for key in sys.modules.keys() if key.startswith(("utils", "components", "app", "validation"))]
for mod in modules_to_clear:
    del sys.modules[mod]

# Ensure cr1_training_ms is at the front of sys.path for this test directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MS_PATH = str(PROJECT_ROOT / "microservices" / "cr1_training_ms")

# Remove any other microservice paths that might conflict
paths_to_remove = [p for p in sys.path if "microservices" in p and "cr1_training_ms" not in p]
for p in paths_to_remove:
    if p in sys.path:
        sys.path.remove(p)

# Ensure cr1_training_ms is at the front
if MS_PATH in sys.path:
    sys.path.remove(MS_PATH)
sys.path.insert(0, MS_PATH)
