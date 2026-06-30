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

"""
Unit tests for data-generation-pipeline/utils/utils.py
"""

import os

import pytest

from utils.utils import load_config_yaml, clean_and_create_dir, scrub_secrets


class TestScrubSecrets:
    """Tests for scrub_secrets — redacts NGC/NVIDIA API keys from log/error text."""

    def test_redacts_nvapi_token(self):
        text = "Command failed: --api-key nvapi-abc123DEF456_-xyz blah"
        scrubbed = scrub_secrets(text)
        assert "nvapi-abc123DEF456_-xyz" not in scrubbed
        assert "REDACTED" in scrubbed

    def test_redacts_multiple_tokens(self):
        text = "first nvapi-AAA111 then nvapi-BBB222 done"
        scrubbed = scrub_secrets(text)
        assert "nvapi-AAA111" not in scrubbed
        assert "nvapi-BBB222" not in scrubbed

    def test_passthrough_without_secret(self):
        text = "Return code: 1\nTraceback ...\nValueError: boom"
        assert scrub_secrets(text) == text

    def test_handles_non_string(self):
        # Should coerce non-str input rather than raise.
        assert "1" in scrub_secrets(1)


class TestLoadConfigYaml:
    """Tests for load_config_yaml function."""

    def test_load_valid_yaml(self, temp_dir):
        """Test loading a valid YAML config file."""
        config_file = temp_dir / "config.yaml"
        config_file.write_text("""
augmentation:
  enabled: true
  types:
    - GQA
    - BCQA
model:
  name: test_model
  batch_size: 32
""")

        result = load_config_yaml(str(config_file))

        assert result["augmentation"]["enabled"] is True
        assert result["augmentation"]["types"] == ["GQA", "BCQA"]
        assert result["model"]["name"] == "test_model"
        assert result["model"]["batch_size"] == 32

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading a non-existent config file returns empty dict."""
        nonexistent = temp_dir / "does_not_exist.yaml"

        result = load_config_yaml(str(nonexistent))

        assert result == {}

    def test_load_empty_yaml(self, temp_dir):
        """Test loading an empty YAML file returns empty dict."""
        empty_file = temp_dir / "empty.yaml"
        empty_file.write_text("")

        result = load_config_yaml(str(empty_file))

        assert result == {}

    def test_load_yaml_preserves_types(self, temp_dir):
        """Test that YAML loading preserves data types."""
        config_file = temp_dir / "types.yaml"
        config_file.write_text("""
string_val: hello
int_val: 42
float_val: 3.14
bool_val: true
list_val:
  - 1
  - 2
  - 3
nested:
  key: value
""")

        result = load_config_yaml(str(config_file))

        assert result["string_val"] == "hello"
        assert result["int_val"] == 42
        assert result["float_val"] == 3.14
        assert result["bool_val"] is True
        assert result["list_val"] == [1, 2, 3]
        assert result["nested"]["key"] == "value"


class TestCleanAndCreateDir:
    """Tests for clean_and_create_dir function."""

    def test_create_new_directory(self, temp_dir):
        """Test creating a new directory."""
        new_dir = temp_dir / "new_folder"
        assert not new_dir.exists()

        result = clean_and_create_dir(str(new_dir))

        assert result is True
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_clean_and_recreate_existing_directory(self, temp_dir):
        """Test cleaning and recreating an existing directory with contents."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()
        # Create some files in the directory
        (existing_dir / "file1.txt").write_text("content1")
        (existing_dir / "file2.txt").write_text("content2")
        subdir = existing_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("content3")

        result = clean_and_create_dir(str(existing_dir))

        assert result is True
        assert existing_dir.exists()
        # Directory should be empty after clean
        assert list(existing_dir.iterdir()) == []

    def test_create_nested_directories(self, temp_dir):
        """Test creating nested directories."""
        nested_dir = temp_dir / "a" / "b" / "c"

        result = clean_and_create_dir(str(nested_dir))

        assert result is True
        assert nested_dir.exists()
