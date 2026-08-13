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

"""Smoke test for sop-build skill (docs-only)."""

from pathlib import Path

import pytest

_SKILL_ROOT = Path(__file__).resolve().parents[1] / "sop-build"


@pytest.mark.unit
def test_sop_build_skill_md_exists():
    skill_md = _SKILL_ROOT / "SKILL.md"
    assert skill_md.is_file(), f"Missing {skill_md}"
    assert skill_md.stat().st_size > 0


@pytest.mark.unit
def test_sop_build_evals_json_exists():
    evals_json = _SKILL_ROOT / "evals" / "evals.json"
    assert evals_json.is_file(), f"Missing {evals_json}"
    assert evals_json.stat().st_size > 0
