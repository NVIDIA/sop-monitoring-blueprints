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
