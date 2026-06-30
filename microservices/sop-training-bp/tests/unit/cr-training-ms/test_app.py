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
Unit tests for cr-training-ms/app.py internal functions.

These tests directly call internal functions rather than going through API endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# We need to mock the database before importing the app
with patch("components.postgres_db.postgres_db") as mock_db:
    mock_db.insert_training_job = AsyncMock(return_value=None)
    mock_db.get_training_job = AsyncMock(return_value=None)
    mock_db.update_training_job = AsyncMock(return_value=None)
    mock_db.list_training_jobs = AsyncMock(return_value=[])
    from app import run_fine_tuning


class TestRunFineTuning:
    """Tests for the run_fine_tuning background task."""

    @pytest.mark.asyncio
    async def test_run_fine_tuning_success(self):
        """Test successful fine-tuning run."""
        from datetime import datetime

        job_id = "test-job-123"
        dataset_path = "/fake/dataset"
        train_config_path = "/fake/config.toml"
        custom_dataset_path = "/fake/custom_dataset.py"

        mock_config = {
            "train": {
                "train_policy": {"dataset": {"name": "", "split": []}},
                "output_dir": "",
            },
            "logging": {"experiment_name": ""},
            "policy": {"model_name_or_path": "/fake/model"},
        }

        # Mock process that returns success
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.wait = AsyncMock(return_value=0)
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()

        # Create async iterator for stdout/stderr
        async def mock_readline():
            return b""

        mock_process.stdout.readline = mock_readline
        mock_process.stderr.readline = mock_readline

        cached_job = {
            "job_id": job_id,
            "status": "running",
            "progress": 0.0,
            "current_step": 0,
            "total_steps": 0,
            "loss": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "log_file_path": "/tmp/test_log.txt",
            "process_pid": 12345,
        }

        with patch("app.get_all_json_paths", return_value=["/fake/dataset/data.json"]):
            with patch("app.read_toml", return_value=mock_config):
                with patch("app.dump_toml"):
                    with patch("app.training_jobs_cache") as mock_cache:
                        mock_cache.get.return_value = cached_job

                        with patch("app.postgres_db.update_training_job", new_callable=AsyncMock):
                            with patch("os.path.exists", return_value=True):
                                with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
                                    mock_exec.return_value = mock_process

                                    with patch("builtins.open", MagicMock()):
                                        await run_fine_tuning(
                                            job_id, dataset_path, train_config_path, custom_dataset_path
                                        )

                                        # Verify cache was updated
                                        mock_cache.update.assert_called()
                                        mock_cache.delete.assert_called_with(job_id)

    @pytest.mark.asyncio
    async def test_run_fine_tuning_model_not_found(self):
        """Test fine-tuning fails when model path doesn't exist."""
        from datetime import datetime

        from fastapi import HTTPException

        job_id = "test-job-123"
        dataset_path = "/fake/dataset"
        train_config_path = "/fake/config.toml"
        custom_dataset_path = "/fake/custom_dataset.py"

        mock_config = {
            "train": {
                "train_policy": {"dataset": {"name": "", "split": []}},
                "output_dir": "",
            },
            "logging": {"experiment_name": ""},
            "policy": {"model_name_or_path": "/nonexistent/model"},
        }

        cached_job = {
            "job_id": job_id,
            "status": "running",
            "progress": 0.0,
            "current_step": 0,
            "total_steps": 0,
            "loss": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "log_file_path": "/tmp/test_log.txt",
            "process_pid": None,
        }

        with patch("app.get_all_json_paths", return_value=["/fake/dataset/data.json"]):
            with patch("app.read_toml", return_value=mock_config):
                with patch("app.dump_toml"):
                    with patch("app.training_jobs_cache") as mock_cache:
                        mock_cache.get.return_value = cached_job

                        with patch("app.postgres_db.update_training_job", new_callable=AsyncMock):
                            with patch("os.path.exists", return_value=False):
                                with patch("builtins.open", MagicMock()):
                                    with pytest.raises(HTTPException):
                                        await run_fine_tuning(
                                            job_id, dataset_path, train_config_path, custom_dataset_path
                                        )

                                    # Verify job was marked as failed
                                    mock_cache.update.assert_called()

    @pytest.mark.asyncio
    async def test_run_fine_tuning_process_failure(self):
        """Test fine-tuning when process returns non-zero exit code."""
        from datetime import datetime

        job_id = "test-job-123"
        dataset_path = "/fake/dataset"
        train_config_path = "/fake/config.toml"
        custom_dataset_path = "/fake/custom_dataset.py"

        mock_config = {
            "train": {
                "train_policy": {"dataset": {"name": "", "split": []}},
                "output_dir": "",
            },
            "logging": {"experiment_name": ""},
            "policy": {"model_name_or_path": "/fake/model"},
        }

        # Mock process that returns failure
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.wait = AsyncMock(return_value=1)  # Non-zero exit code
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()

        async def mock_readline():
            return b""

        mock_process.stdout.readline = mock_readline
        mock_process.stderr.readline = mock_readline

        cached_job = {
            "job_id": job_id,
            "status": "running",
            "progress": 0.0,
            "current_step": 0,
            "total_steps": 0,
            "loss": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "log_file_path": "/tmp/test_log.txt",
            "process_pid": 12345,
        }

        with patch("app.get_all_json_paths", return_value=["/fake/dataset/data.json"]):
            with patch("app.read_toml", return_value=mock_config):
                with patch("app.dump_toml"):
                    with patch("app.training_jobs_cache") as mock_cache:
                        mock_cache.get.return_value = cached_job

                        with patch("app.postgres_db.update_training_job", new_callable=AsyncMock):
                            with patch("os.path.exists", return_value=True):
                                with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
                                    mock_exec.return_value = mock_process

                                    with patch("builtins.open", MagicMock()):
                                        await run_fine_tuning(
                                            job_id, dataset_path, train_config_path, custom_dataset_path
                                        )

                                        # Verify job status was updated to failed
                                        update_calls = mock_cache.update.call_args_list
                                        # Check if any call set status to failed
                                        status_updates = [
                                            call for call in update_calls
                                            if call.kwargs.get("status") == "failed"
                                        ]
                                        assert len(status_updates) > 0

    @pytest.mark.asyncio
    async def test_run_fine_tuning_cancelled_job_not_marked_failed(self):
        """Test that cancelled job is not marked as failed after process exit."""
        from datetime import datetime

        job_id = "test-job-123"
        dataset_path = "/fake/dataset"
        train_config_path = "/fake/config.toml"
        custom_dataset_path = "/fake/custom_dataset.py"

        mock_config = {
            "train": {
                "train_policy": {"dataset": {"name": "", "split": []}},
                "output_dir": "",
            },
            "logging": {"experiment_name": ""},
            "policy": {"model_name_or_path": "/fake/model"},
        }

        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.wait = AsyncMock(return_value=-9)  # Killed by signal
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()

        async def mock_readline():
            return b""

        mock_process.stdout.readline = mock_readline
        mock_process.stderr.readline = mock_readline

        # Job is already marked as cancelled
        cached_job = {
            "job_id": job_id,
            "status": "cancelled",
            "progress": 0.0,
            "current_step": 0,
            "total_steps": 0,
            "loss": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "log_file_path": "/tmp/test_log.txt",
            "process_pid": 12345,
        }

        with patch("app.get_all_json_paths", return_value=["/fake/dataset/data.json"]):
            with patch("app.read_toml", return_value=mock_config):
                with patch("app.dump_toml"):
                    with patch("app.training_jobs_cache") as mock_cache:
                        mock_cache.get.return_value = cached_job

                        with patch("app.postgres_db.update_training_job", new_callable=AsyncMock):
                            with patch("os.path.exists", return_value=True):
                                with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
                                    mock_exec.return_value = mock_process

                                    with patch("builtins.open", MagicMock()):
                                        await run_fine_tuning(
                                            job_id, dataset_path, train_config_path, custom_dataset_path
                                        )

                                        # Verify status was NOT changed to failed
                                        # (cancelled status should be preserved)
                                        mock_cache.delete.assert_called_with(job_id)

    @pytest.mark.asyncio
    async def test_run_fine_tuning_with_log_parsing(self):
        """Test fine-tuning with progress log parsing."""
        from datetime import datetime

        job_id = "test-job-123"
        dataset_path = "/fake/dataset"
        train_config_path = "/fake/config.toml"
        custom_dataset_path = "/fake/custom_dataset.py"

        mock_config = {
            "train": {
                "train_policy": {"dataset": {"name": "", "split": []}},
                "output_dir": "",
            },
            "logging": {"experiment_name": ""},
            "policy": {"model_name_or_path": "/fake/model"},
        }

        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.wait = AsyncMock(return_value=0)

        # Simulate log output with progress information
        log_lines = [
            b"Step 1/100, Loss: 2.5\n",
            b"Step 50/100, Loss: 1.0\n",
            b"",
        ]
        log_index = [0]

        async def mock_stdout_readline():
            if log_index[0] < len(log_lines):
                line = log_lines[log_index[0]]
                log_index[0] += 1
                return line
            return b""

        async def mock_stderr_readline():
            return b""

        mock_process.stdout = MagicMock()
        mock_process.stdout.readline = mock_stdout_readline
        mock_process.stderr = MagicMock()
        mock_process.stderr.readline = mock_stderr_readline

        cached_job = {
            "job_id": job_id,
            "status": "running",
            "progress": 0.0,
            "current_step": 0,
            "total_steps": 0,
            "loss": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "log_file_path": "/tmp/test_log.txt",
            "process_pid": 12345,
        }

        with patch("app.get_all_json_paths", return_value=["/fake/dataset/data.json"]):
            with patch("app.read_toml", return_value=mock_config):
                with patch("app.dump_toml"):
                    with patch("app.training_jobs_cache") as mock_cache:
                        mock_cache.get.return_value = cached_job

                        with patch("app.postgres_db.update_training_job", new_callable=AsyncMock):
                            with patch("os.path.exists", return_value=True):
                                with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
                                    mock_exec.return_value = mock_process

                                    with patch("builtins.open", MagicMock()):
                                        with patch("app.parse_cr_log") as mock_parse:
                                            mock_parse.return_value = {
                                                "current_step": 50,
                                                "total_steps": 100,
                                                "loss": 1.0,
                                            }

                                            await run_fine_tuning(
                                                job_id, dataset_path, train_config_path, custom_dataset_path
                                            )

                                            # Verify job completed successfully
                                            mock_cache.delete.assert_called_with(job_id)
