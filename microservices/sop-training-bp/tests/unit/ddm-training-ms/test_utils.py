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
Unit tests for ddm-training-ms/utils/utils.py
"""

from unittest.mock import MagicMock, patch

from utils.utils import (
    create_dir,
    create_file,
    read_toml,
    dump_toml,
    read_yaml,
    dump_yaml,
    parse_ddm_log,
    terminate_process_tree,
)


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
                "name": "ddm_net",
            },
        }

        result = dump_toml(config, str(toml_path))
        assert result is True
        assert toml_path.exists()

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


class TestYamlOperations:
    """Tests for YAML read/write operations."""

    def test_dump_and_read_yaml(self, temp_dir):
        """Test dumping and reading YAML files."""
        yaml_path = temp_dir / "config.yaml"
        config = {
            "training": {
                "epochs": 100,
                "batch_size": 16,
            },
            "data": {
                "path": "/data/videos",
            },
        }

        result = dump_yaml(config, str(yaml_path))
        assert result is True
        assert yaml_path.exists()

        loaded_config = read_yaml(str(yaml_path))
        assert loaded_config == config

    def test_read_yaml_preserves_types(self, temp_dir):
        """Test that YAML preserves various data types."""
        yaml_path = temp_dir / "types.yaml"
        config = {
            "string_val": "world",
            "int_val": 100,
            "float_val": 2.718,
            "bool_val": False,
            "list_val": ["a", "b", "c"],
        }

        dump_yaml(config, str(yaml_path))
        loaded = read_yaml(str(yaml_path))

        assert loaded["string_val"] == "world"
        assert loaded["int_val"] == 100
        assert loaded["float_val"] == 2.718
        assert loaded["bool_val"] is False
        assert loaded["list_val"] == ["a", "b", "c"]


class TestParseDdmLog:
    """Tests for parse_ddm_log function.
    """

    def test_parse_log_with_loss_outside_brackets(self):
        """Test parsing log where loss is outside brackets (current regex expectation)."""
        # This format matches the current regex pattern
        log_line = "Epoch 0:   1%|          | 1/115 [00:02<05:40] train/loss_step=12.90"

        result = parse_ddm_log(log_line)

        assert result["epoch"] == 0
        assert result["current_step"] == 1  # 0 * 115 + 1
        assert result["total_steps"] == 115
        assert result["loss"] == 12.90

    def test_parse_log_mid_epoch_outside_brackets(self):
        """Test parsing log line in the middle of an epoch with loss outside brackets."""
        log_line = "Epoch 2:  50%|#####     | 58/115 [01:30<01:30] train/loss_step=5.23"

        result = parse_ddm_log(log_line)

        assert result["epoch"] == 2
        assert result["current_step"] == 2 * 115 + 58  # global step
        assert result["total_steps"] == 115
        assert result["loss"] == 5.23

    def test_parse_log_with_loss_epoch_outside_brackets(self):
        """Test parsing log with train/loss_epoch outside brackets."""
        log_line = "Epoch 5: 100%|##########| 115/115 [03:00<00:00] train/loss_epoch=2.15"

        result = parse_ddm_log(log_line)

        assert result["epoch"] == 5
        assert result["loss"] == 2.15

    def test_parse_log_with_nan_loss_outside_brackets(self):
        """Test parsing log with NaN loss value outside brackets."""
        log_line = "Epoch 0:   1%|          | 1/115 [00:02<05:40] train/loss_step=nan"

        result = parse_ddm_log(log_line)

        assert result["epoch"] == 0
        assert result["loss"] is None

    def test_parse_invalid_log_returns_empty(self):
        """Test that invalid log lines return empty dict."""
        invalid_lines = [
            "Some random log message",
            "Training started...",
            "",
            "Epoch 0: Loading data...",
            "[INFO] Model initialized",
            # Standard PyTorch Lightning format with loss INSIDE brackets
            # doesn't match the current regex
            "Epoch 0:   1%|          | 1/115 [00:02<05:40,  0.33it/s, v_num=0, train/loss_step=12.90]",
        ]

        for line in invalid_lines:
            result = parse_ddm_log(line)
            assert result == {}

    def test_parse_log_high_epoch(self):
        """Test parsing log with high epoch number."""
        log_line = "Epoch 99:  75%|#######   | 86/115 [02:15<00:45] train/loss_step=0.45"

        result = parse_ddm_log(log_line)

        assert result["epoch"] == 99
        assert result["current_step"] == 99 * 115 + 86
        assert result["loss"] == 0.45

    def test_parse_log_exception_handling(self):
        """Test that exceptions in log parsing return empty dict."""
        # Patch re.search to raise an exception
        with patch("re.search", side_effect=Exception("Regex error")):
            result = parse_ddm_log("Epoch 0: 1%| | 1/115 [00:02<05:40] train/loss_step=12.90")

            assert result == {}

    def test_parse_log_invalid_loss_value(self):
        """Test parsing log with invalid loss value that raises ValueError."""
        # Mock the regex match to return an invalid loss string
        mock_match = MagicMock()
        mock_match.group.side_effect = lambda x: {
            1: "0",       # epoch
            2: "1",       # current_step
            3: "115",     # total_steps
            4: "invalid", # loss_str - can't be converted to float
        }[x]

        with patch("re.search", return_value=mock_match):
            result = parse_ddm_log("Epoch 0: 1%| | 1/115 [00:02<05:40] train/loss_step=invalid")

            assert result["epoch"] == 0
            assert result["current_step"] == 1
            assert result["total_steps"] == 115
            assert result["loss"] is None  # Should be None due to ValueError


class TestTerminateProcessTree:
    """Tests for terminate_process_tree function."""

    def test_terminate_process_tree_success(self):
        """Test successful termination of process tree."""
        mock_child = MagicMock()
        mock_child.pid = 12346
        mock_child.is_running.return_value = False

        mock_parent = MagicMock()
        mock_parent.pid = 12345
        mock_parent.children.return_value = [mock_child]
        mock_parent.is_running.return_value = False

        with patch("psutil.Process", return_value=mock_parent):
            with patch("psutil.wait_procs", return_value=([], [])):
                result = terminate_process_tree(12345)

                assert result is True
                mock_parent.terminate.assert_called_once()
                mock_child.terminate.assert_called_once()

    def test_terminate_process_already_dead(self):
        """Test terminating a process that is already dead."""
        import psutil

        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(12345)):
            result = terminate_process_tree(12345)

            assert result is True

    def test_terminate_process_child_already_dead(self):
        """Test terminating when child process is already dead."""
        import psutil

        mock_child = MagicMock()
        mock_child.pid = 12346
        mock_child.terminate.side_effect = psutil.NoSuchProcess(12346)
        mock_child.is_running.side_effect = psutil.NoSuchProcess(12346)

        mock_parent = MagicMock()
        mock_parent.pid = 12345
        mock_parent.children.return_value = [mock_child]
        mock_parent.is_running.return_value = False

        with patch("psutil.Process", return_value=mock_parent):
            with patch("psutil.wait_procs", return_value=([], [])):
                result = terminate_process_tree(12345)

                assert result is True

    def test_terminate_process_parent_already_dead_during_terminate(self):
        """Test when parent process dies during termination."""
        import psutil

        mock_parent = MagicMock()
        mock_parent.pid = 12345
        mock_parent.children.return_value = []
        mock_parent.terminate.side_effect = psutil.NoSuchProcess(12345)
        mock_parent.is_running.side_effect = psutil.NoSuchProcess(12345)

        with patch("psutil.Process", return_value=mock_parent):
            with patch("psutil.wait_procs", return_value=([], [])):
                result = terminate_process_tree(12345)

                assert result is True

    def test_terminate_process_force_kill_needed(self):
        """Test that processes are force killed if they don't terminate gracefully."""
        mock_child = MagicMock()
        mock_child.pid = 12346
        mock_child.is_running.return_value = False

        mock_parent = MagicMock()
        mock_parent.pid = 12345
        mock_parent.children.return_value = [mock_child]
        mock_parent.is_running.return_value = False

        # First wait_procs returns alive processes, second returns empty
        alive_processes = [mock_child]

        with patch("psutil.Process", return_value=mock_parent):
            with patch("psutil.wait_procs", side_effect=[([], alive_processes), ([], [])]):
                result = terminate_process_tree(12345)

                assert result is True
                mock_child.kill.assert_called_once()

    def test_terminate_process_force_kill_already_dead(self):
        """Test force kill when process is already dead."""
        import psutil

        mock_child = MagicMock()
        mock_child.pid = 12346
        mock_child.kill.side_effect = psutil.NoSuchProcess(12346)
        mock_child.is_running.side_effect = psutil.NoSuchProcess(12346)

        mock_parent = MagicMock()
        mock_parent.pid = 12345
        mock_parent.children.return_value = []
        mock_parent.is_running.return_value = False

        alive_processes = [mock_child]

        with patch("psutil.Process", return_value=mock_parent):
            with patch("psutil.wait_procs", side_effect=[([], alive_processes), ([], [])]):
                result = terminate_process_tree(12345)

                assert result is True

    def test_terminate_process_still_running_after_kill(self):
        """Test when process is still running after kill attempt."""
        mock_parent = MagicMock()
        mock_parent.pid = 12345
        mock_parent.children.return_value = []
        mock_parent.is_running.return_value = True  # Still running

        with patch("psutil.Process", return_value=mock_parent):
            with patch("psutil.wait_procs", return_value=([], [])):
                result = terminate_process_tree(12345)

                assert result is False

    def test_terminate_process_general_exception(self):
        """Test handling of general exceptions."""
        with patch("psutil.Process", side_effect=Exception("Unexpected error")):
            result = terminate_process_tree(12345)

            assert result is False

    def test_terminate_process_with_timeout(self):
        """Test termination with custom timeout."""
        mock_parent = MagicMock()
        mock_parent.pid = 12345
        mock_parent.children.return_value = []
        mock_parent.is_running.return_value = False

        with patch("psutil.Process", return_value=mock_parent):
            with patch("psutil.wait_procs", return_value=([], [])) as mock_wait:
                result = terminate_process_tree(12345, timeout=60)

                assert result is True
                # Verify timeout was passed to wait_procs
                mock_wait.assert_called()
                call_args = mock_wait.call_args_list[0]
                assert call_args[1]["timeout"] == 60

    def test_terminate_process_no_children(self):
        """Test terminating a process with no children."""
        mock_parent = MagicMock()
        mock_parent.pid = 12345
        mock_parent.children.return_value = []
        mock_parent.is_running.return_value = False

        with patch("psutil.Process", return_value=mock_parent):
            with patch("psutil.wait_procs", return_value=([], [])):
                result = terminate_process_tree(12345)

                assert result is True
                mock_parent.terminate.assert_called_once()
