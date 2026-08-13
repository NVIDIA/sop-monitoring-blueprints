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

"""Shared fixtures for vss-sop-build script unit tests."""
import importlib.util
import sys
from pathlib import Path

import pytest

SKILLS = Path(__file__).resolve().parents[1]
BUILD_SCRIPTS = SKILLS / "vss-sop-build" / "scripts"
BUILD_LIB = BUILD_SCRIPTS / "lib"


def load_module(name: str, path: Path):
    """Load a script module by file path (scripts are not packages)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def kibana_fields():
    if str(BUILD_LIB) not in sys.path:
        sys.path.insert(0, str(BUILD_LIB))
    return load_module("kibana_fields", BUILD_LIB / "kibana_fields.py")


@pytest.fixture(scope="session")
def patch_profiles():
    return load_module("patch_profiles", BUILD_SCRIPTS / "patch_profiles.py")


@pytest.fixture(scope="session")
def verify_build():
    if str(BUILD_LIB) not in sys.path:
        sys.path.insert(0, str(BUILD_LIB))
    return load_module("verify_build", BUILD_SCRIPTS / "verify_build.py")


@pytest.fixture(scope="session")
def modify_foundational():
    if str(BUILD_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(BUILD_SCRIPTS))
    return load_module(
        "modify_foundational_for_sop",
        BUILD_SCRIPTS / "modify_foundational_for_sop.py",
    )


@pytest.fixture(scope="session")
def modify_vios():
    return load_module(
        "modify_vios_for_sop",
        BUILD_SCRIPTS / "modify_vios_for_sop.py",
    )
