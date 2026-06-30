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
Unit tests for cr-training-ms/utils/utils.py
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
from utils.utils import create_dir, create_file, dump_toml, parse_cr_log, read_toml, terminate_process_tree


class TestCreateDir:
    """Tests for create_dir function."""

    def test_create_new_directory(self, temp_dir):
        """Test creating a new directory."""
        new_dir = temp_dir / "new_folder"
        assert not new_dir.exists()

        result = create_dir(str(new_dir))

        assert result is True
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_existing_directory_returns_false(self, temp_dir):
        """Test that existing directory returns False."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()

        result = create_dir(str(existing_dir))

        assert result is False
        assert existing_dir.exists()

    def test_create_nested_directories(self, temp_dir):
        """Test creating nested directories."""
        nested_dir = temp_dir / "level1" / "level2" / "level3"

        result = create_dir(str(nested_dir))

        assert result is True
        assert nested_dir.exists()


class TestCreateFile:
    """Tests for create_file function."""

    def test_create_new_file(self, temp_dir):
        """Test creating a new file."""
        new_file = temp_dir / "new_file.txt"
        assert not new_file.exists()

        result = create_file(str(new_file))

        assert result is True
        assert new_file.exists()
        assert new_file.is_file()

    def test_existing_file_returns_false(self, temp_dir):
        """Test that existing file returns False."""
        existing_file = temp_dir / "existing.txt"
        existing_file.write_text("content")

        result = create_file(str(existing_file))

        assert result is False
        # Content should be preserved
        assert existing_file.read_text() == "content"

    def test_create_file_with_nested_dirs(self, temp_dir):
        """Test creating file with nested parent directories."""
        nested_file = temp_dir / "a" / "b" / "c" / "file.txt"

        result = create_file(str(nested_file))

        assert result is True
        assert nested_file.exists()


class TestTomlOperations:
    """Tests for TOML read/write operations."""

    def test_dump_and_read_toml(self, temp_dir):
        """Test dumping and reading TOML files."""
        toml_path = temp_dir / "config.toml"
        config = {
            "train": {
                "output_dir": "/path/to/output",
                "epochs": 10,
            },
            "model": {
                "name": "test_model",
            },
        }

        # Dump to file
        result = dump_toml(config, str(toml_path))
        assert result is True
        assert toml_path.exists()

        # Read back
        loaded_config = read_toml(str(toml_path))
        assert loaded_config == config

    def test_read_toml_preserves_types(self, temp_dir):
        """Test that TOML preserves various data types."""
        toml_path = temp_dir / "types.toml"
        config = {
            "string_val": "hello",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "list_val": [1, 2, 3],
        }

        dump_toml(config, str(toml_path))
        loaded = read_toml(str(toml_path))

        assert loaded["string_val"] == "hello"
        assert loaded["int_val"] == 42
        assert loaded["float_val"] == 3.14
        assert loaded["bool_val"] is True
        assert loaded["list_val"] == [1, 2, 3]


class TestParseCrLog:
    """Tests for parse_cr_log function."""

    def test_parse_valid_log_line(self):
        """Test parsing a valid Cosmos-Reason log line."""
        log_line = "Step: 100/1000, Loss: 0.5234, Grad norm: 1.23e-02, Learning rate: 1e-05, Iteration time: 2.5s"

        result = parse_cr_log(log_line)

        assert result["current_step"] == 100
        assert result["total_steps"] == 1000
        assert result["loss"] == 0.5234
        assert result["grad_norm"] == 0.0123
        assert result["learning_rate"] == 1e-05
        assert result["iteration_time"] == 2.5

    def test_parse_log_with_scientific_notation(self):
        """Test parsing log with scientific notation values."""
        log_line = "Step: 1/100, Loss: 1.5e-03, Grad norm: 2.0e+00, Learning rate: 5e-06, Iteration time: 1.0s"

        result = parse_cr_log(log_line)

        assert result["current_step"] == 1
        assert result["total_steps"] == 100
        assert result["loss"] == 0.0015
        assert result["grad_norm"] == 2.0
        assert result["learning_rate"] == 5e-06

    def test_parse_invalid_log_returns_empty(self):
        """Test that invalid log lines return empty dict."""
        invalid_lines = [
            "Some random log message",
            "Training started...",
            "",
            "Step: 100",  # Incomplete
        ]

        for line in invalid_lines:
            result = parse_cr_log(line)
            assert result == {}

    def test_parse_log_with_trailing_period(self):
        """Test parsing log line with optional trailing period."""
        log_line = "Step: 50/500, Loss: 0.123, Grad norm: 0.5, Learning rate: 1e-04, Iteration time: 3.2s."

        result = parse_cr_log(log_line)

        assert result["current_step"] == 50
        assert result["total_steps"] == 500

    def test_parse_log_embedded_in_longer_string(self):
        """Test parsing when log pattern is embedded in longer string."""
        log_line = "[INFO] 2025-01-22 Step: 10/100, Loss: 0.8, Grad norm: 1.0, Learning rate: 1e-05, Iteration time: 1.5s - done"

        result = parse_cr_log(log_line)

        assert result["current_step"] == 10
        assert result["total_steps"] == 100

    def test_parse_log_exception_returns_empty(self):
        """Test that exceptions during parsing return empty dict."""
        # Mock re.search to raise an exception
        with patch("re.search", side_effect=Exception("Regex error")):
            result = parse_cr_log("Step: 1/100, Loss: 0.5, Grad norm: 1.0, Learning rate: 1e-05, Iteration time: 1.0s")
            assert result == {}


class TestTerminateProcessTree:
    """Tests for terminate_process_tree function."""

    def test_terminate_process_tree_success(self):
        """Test successful termination of process tree."""
        # Create mock processes
        mock_child1 = MagicMock()
        mock_child1.pid = 1001
        mock_child1.is_running.return_value = False

        mock_child2 = MagicMock()
        mock_child2.pid = 1002
        mock_child2.is_running.return_value = False

        mock_parent = MagicMock()
        mock_parent.pid = 1000
        mock_parent.children.return_value = [mock_child1, mock_child2]
        mock_parent.is_running.return_value = False

        with patch("psutil.Process", return_value=mock_parent):
            with patch("psutil.wait_procs", return_value=([], [])):
                result = terminate_process_tree(1000)

                assert result is True
                mock_parent.terminate.assert_called_once()
                mock_child1.terminate.assert_called_once()
                mock_child2.terminate.assert_called_once()

    def test_terminate_process_already_dead(self):
        """Test handling when process is already dead."""
        import psutil

        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(1000)):
            result = terminate_process_tree(1000)

            # Should return True since process is already gone
            assert result is True

    def test_terminate_process_child_already_dead(self):
        """Test handling when child process is already dead during termination."""
        import psutil

        mock_child = MagicMock()
        mock_child.pid = 1001
        mock_child.terminate.side_effect = psutil.NoSuchProcess(1001)
        mock_child.is_running.side_effect = psutil.NoSuchProcess(1001)

        mock_parent = MagicMock()
        mock_parent.pid = 1000
        mock_parent.children.return_value = [mock_child]
        mock_parent.is_running.return_value = False

        with patch("psutil.Process", return_value=mock_parent):
            with patch("psutil.wait_procs", return_value=([], [])):
                result = terminate_process_tree(1000)

                assert result is True

    def test_terminate_process_parent_already_dead_during_terminate(self):
        """Test handling when parent process dies during termination."""
        import psutil

        mock_parent = MagicMock()
        mock_parent.pid = 1000
        mock_parent.children.return_value = []
        mock_parent.terminate.side_effect = psutil.NoSuchProcess(1000)
        mock_parent.is_running.side_effect = psutil.NoSuchProcess(1000)

        with patch("psutil.Process", return_value=mock_parent):
            with patch("psutil.wait_procs", return_value=([], [])):
                result = terminate_process_tree(1000)

                assert result is True

    def test_terminate_process_force_kill_needed(self):
        """Test force killing processes that don't terminate gracefully."""
        # Process that stays alive after SIGTERM
        mock_stubborn = MagicMock()
        mock_stubborn.pid = 1001
        mock_stubborn.is_running.side_effect = [True, False]  # First check: alive, second: dead

        mock_parent = MagicMock()
        mock_parent.pid = 1000
        mock_parent.children.return_value = []
        mock_parent.is_running.return_value = False

        with patch("psutil.Process", return_value=mock_parent):
            # First wait_procs returns stubborn process still alive, second returns empty
            with patch("psutil.wait_procs", side_effect=[([], [mock_stubborn]), ([], [])]):
                result = terminate_process_tree(1000)

                assert result is True
                mock_stubborn.kill.assert_called_once()

    def test_terminate_process_force_kill_already_dead(self):
        """Test force kill when process dies before kill."""
        import psutil

        mock_stubborn = MagicMock()
        mock_stubborn.pid = 1001
        mock_stubborn.kill.side_effect = psutil.NoSuchProcess(1001)
        mock_stubborn.is_running.side_effect = psutil.NoSuchProcess(1001)

        mock_parent = MagicMock()
        mock_parent.pid = 1000
        mock_parent.children.return_value = []
        mock_parent.is_running.return_value = False

        with patch("psutil.Process", return_value=mock_parent):
            with patch("psutil.wait_procs", side_effect=[([], [mock_stubborn]), ([], [])]):
                result = terminate_process_tree(1000)

                assert result is True

    def test_terminate_process_still_running_after_kill(self):
        """Test handling when process is still running after kill attempt."""
        mock_parent = MagicMock()
        mock_parent.pid = 1000
        mock_parent.children.return_value = []
        mock_parent.is_running.return_value = True  # Still running!

        with patch("psutil.Process", return_value=mock_parent):
            with patch("psutil.wait_procs", return_value=([], [])):
                result = terminate_process_tree(1000)

                # Should return False since process is still running
                assert result is False

    def test_terminate_process_general_exception(self):
        """Test handling of general exceptions."""
        with patch("psutil.Process", side_effect=Exception("Unexpected error")):
            result = terminate_process_tree(1000)

            # Should return False on exception
            assert result is False

    def test_terminate_process_with_timeout(self):
        """Test that custom timeout is passed to wait_procs."""

        mock_parent = MagicMock()
        mock_parent.pid = 1000
        mock_parent.children.return_value = []
        mock_parent.is_running.return_value = False

        with patch("psutil.Process", return_value=mock_parent):
            with patch("psutil.wait_procs", return_value=([], [])) as mock_wait:
                result = terminate_process_tree(1000, timeout=60)

                assert result is True
                # Check that timeout was passed to wait_procs
                mock_wait.assert_called()
                call_args = mock_wait.call_args_list[0]
                assert call_args[1]["timeout"] == 60

    def test_terminate_process_no_children(self):
        """Test terminating a process with no children."""

        mock_parent = MagicMock()
        mock_parent.pid = 1000
        mock_parent.children.return_value = []  # No children
        mock_parent.is_running.return_value = False

        with patch("psutil.Process", return_value=mock_parent):
            with patch("psutil.wait_procs", return_value=([], [])):
                result = terminate_process_tree(1000)

                assert result is True
                mock_parent.terminate.assert_called_once()
