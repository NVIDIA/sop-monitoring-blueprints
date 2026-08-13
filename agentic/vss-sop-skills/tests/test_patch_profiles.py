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

"""Unit tests for patch_profiles.py."""
import runpy
import sys
from unittest import mock

import pytest

SAMPLE_COMPOSE = """\
services:
  elasticsearch:
    profiles: ["default", "other"]
    image: es
  kibana:
    profiles: ["default"]
    image: kibana
networks:
  host:
"""


@pytest.mark.unit
class TestPatchFileProfiles:
    def test_returns_false_when_file_missing(self, patch_profiles, tmp_path):
        missing = tmp_path / "missing.yml"
        assert patch_profiles.patch_file_profiles(str(missing)) is False

    def test_adds_profile_to_all_services(self, patch_profiles, tmp_path):
        yml = tmp_path / "compose.yml"
        yml.write_text(SAMPLE_COMPOSE)
        assert patch_profiles.patch_file_profiles(str(yml)) is True
        content = yml.read_text()
        assert 'profiles: ["bp_sop_2d"' in content
        assert "bp_sop_2d" in content.split("elasticsearch")[1].split("kibana")[0]

    def test_reorders_existing_profile_to_first(self, patch_profiles, tmp_path):
        yml = tmp_path / "compose.yml"
        yml.write_text(
            "services:\n  svc:\n    profiles: [\"other\", \"bp_sop_2d\"]\n"
        )
        patch_profiles.patch_file_profiles(str(yml))
        assert 'profiles: ["bp_sop_2d", "other"]' in yml.read_text()

    def test_filters_by_service_name(self, patch_profiles, tmp_path):
        yml = tmp_path / "compose.yml"
        yml.write_text(SAMPLE_COMPOSE)
        patch_profiles.patch_file_profiles(str(yml), services_to_patch=["kibana"])
        content = yml.read_text()
        es_block = content.split("elasticsearch:")[1].split("kibana:")[0]
        kb_block = content.split("kibana:")[1].split("networks:")[0]
        assert "bp_sop_2d" not in es_block
        assert "bp_sop_2d" in kb_block

    def test_strips_minimal_profile_suffix(self, patch_profiles, tmp_path):
        yml = tmp_path / "compose.yml"
        yml.write_text(
            "services:\n  svc:\n    profiles: [\"a\"${MINIMAL_PROFILE:+_extended}]\n"
        )
        patch_profiles.patch_file_profiles(str(yml))
        assert "${MINIMAL_PROFILE" not in yml.read_text()

    def test_custom_profile_name(self, patch_profiles, tmp_path):
        yml = tmp_path / "compose.yml"
        yml.write_text("services:\n  svc:\n    profiles: [\"x\"]\n")
        patch_profiles.patch_file_profiles(str(yml), profile_to_add="custom_prof")
        assert 'profiles: ["custom_prof", "x"]' in yml.read_text()


@pytest.mark.unit
class TestPatchProfilesMain:
    def test_main_usage_exits_when_no_args(self, patch_profiles):
        with mock.patch.object(sys, "argv", ["patch_profiles.py"]):
            with pytest.raises(SystemExit) as exc:
                patch_profiles.__name__  # ensure module loaded
                runpy.run_path(
                    str(patch_profiles.__file__),
                    run_name="__main__",
                )
            assert exc.value.code == 1

    def test_main_patches_file(self, patch_profiles, tmp_path):
        yml = tmp_path / "svc.yml"
        yml.write_text("services:\n  svc:\n    profiles: [\"a\"]\n")
        with mock.patch.object(
            sys,
            "argv",
            ["patch_profiles.py", str(yml), "all", "my_profile"],
        ):
            runpy.run_path(str(patch_profiles.__file__), run_name="__main__")
        assert "my_profile" in yml.read_text()

    def test_main_parses_service_list(self, patch_profiles, tmp_path):
        yml = tmp_path / "svc.yml"
        yml.write_text(SAMPLE_COMPOSE)
        with mock.patch.object(
            sys,
            "argv",
            ["patch_profiles.py", str(yml), "kibana,elasticsearch"],
        ):
            runpy.run_path(str(patch_profiles.__file__), run_name="__main__")
        content = yml.read_text()
        assert content.count("bp_sop_2d") == 2
