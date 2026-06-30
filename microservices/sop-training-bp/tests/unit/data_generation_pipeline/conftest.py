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
Conftest for data_generation_pipeline unit tests - ensures correct module imports.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure data-generation-pipeline is at the front of sys.path for this test directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MS_PATH = str(PROJECT_ROOT / "microservices" / "data-generation-pipeline")
VLM_AUG_PATH = str(PROJECT_ROOT / "microservices" / "data-generation-pipeline" / "source" / "multi-modal-autolabel-augmentation-pipeline")

# Remove any other microservice utils paths that might conflict
paths_to_remove = [p for p in sys.path if "microservices" in p and "data-generation-pipeline" not in p]
for p in paths_to_remove:
    if p in sys.path:
        sys.path.remove(p)

# Ensure data-generation-pipeline and vlm_aug are at the front
for path in [VLM_AUG_PATH, MS_PATH]:
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

# Mock heavy dependencies used by vlm_aug before they're imported
sys.modules["cv2"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["moviepy"] = MagicMock()
sys.modules["moviepy.editor"] = MagicMock()

# Mock the logger module to avoid file creation issues in tests
# This MUST happen before clearing other modules and before any app imports
mock_logger = MagicMock()
mock_logger.info = MagicMock()
mock_logger.error = MagicMock()
mock_logger.warning = MagicMock()
mock_logger.debug = MagicMock()

# Create a mock utils.logger module
mock_logger_module = MagicMock()
mock_logger_module.app_logger = mock_logger
mock_logger_module.get_logger = MagicMock(return_value=mock_logger)
mock_logger_module.setup_logger = MagicMock(return_value=mock_logger)

# Install mock logger FIRST before clearing any modules
sys.modules["utils.logger"] = mock_logger_module

# Now clear any cached modules to ensure correct imports (except our logger mock)
modules_to_clear = [
    key for key in list(sys.modules.keys())
    if key.startswith(("utils", "vlm_aug", "validation", "components", "app")) and key != "utils.logger"
]
for mod in modules_to_clear:
    del sys.modules[mod]
