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
Unit tests for ddm-training-ms/app.py internal functions.

These tests cover internal functions that are not exposed via API endpoints:
- cancel_orphaned_jobs_on_startup: Startup cleanup logic
- run_fine_tuning: Background task for training execution
"""

import json
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

# Mock database before importing app module
with patch("components.postgres_db.postgres_db") as mock_db:
    mock_db.insert_training_job = AsyncMock(return_value=None)
    mock_db.get_training_job = AsyncMock(return_value=None)
    mock_db.list_training_jobs = AsyncMock(return_value=[])
    mock_db.update_training_job = AsyncMock(return_value=None)
    from app import (
        cancel_orphaned_jobs_on_startup,
        run_fine_tuning,
        training_jobs_cache,
    )
    import utils.constant as const


class TestCancelOrphanedJobsOnStartup:
    """Tests for the cancel_orphaned_jobs_on_startup function."""

    @pytest.mark.asyncio
    async def test_no_orphaned_jobs(self):
        """Test startup with no jobs in the database."""
        with patch("app.postgres_db.list_training_jobs", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []

            await cancel_orphaned_jobs_on_startup()

            mock_list.assert_called_once()

    @pytest.mark.asyncio
    async def test_jobs_in_terminal_states_not_cancelled(self):
        """Test that jobs in terminal states (completed, cancelled, failed) are not modified."""
        mock_completed = MagicMock()
        mock_completed.id = "job-completed"
        mock_completed.status = const.COMPLETED_STATUS

        mock_cancelled = MagicMock()
        mock_cancelled.id = "job-cancelled"
        mock_cancelled.status = const.CANCELLED_STATUS

        mock_failed = MagicMock()
        mock_failed.id = "job-failed"
        mock_failed.status = const.FAILED_STATUS

        with patch("app.postgres_db.list_training_jobs", new_callable=AsyncMock) as mock_list:
            with patch("app.postgres_db.update_training_job", new_callable=AsyncMock) as mock_update:
                mock_list.return_value = [mock_completed, mock_cancelled, mock_failed]

                await cancel_orphaned_jobs_on_startup()

                # No updates should be made for terminal state jobs
                mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_running_job_without_pid_cancelled(self):
        """Test that running job without PID is marked as cancelled."""
        mock_job = MagicMock()
        mock_job.id = "orphaned-job"
        mock_job.status = const.RUNNING_STATUS
        mock_job.process_pid = None  # No PID stored

        with patch("app.postgres_db.list_training_jobs", new_callable=AsyncMock) as mock_list:
            with patch("app.postgres_db.update_training_job", new_callable=AsyncMock) as mock_update:
                mock_list.return_value = [mock_job]

                await cancel_orphaned_jobs_on_startup()

                mock_update.assert_called_once()
                call_kwargs = mock_update.call_args[1]
                assert call_kwargs["status"] == const.CANCELLED_STATUS
                assert call_kwargs["process_pid"] is None

    @pytest.mark.asyncio
    async def test_running_job_with_pid_terminated_and_cancelled(self):
        """Test that running job with PID is terminated and marked as cancelled."""
        mock_job = MagicMock()
        mock_job.id = "orphaned-job-with-pid"
        mock_job.status = const.RUNNING_STATUS
        mock_job.process_pid = 12345

        with patch("app.postgres_db.list_training_jobs", new_callable=AsyncMock) as mock_list:
            with patch("app.postgres_db.update_training_job", new_callable=AsyncMock) as mock_update:
                with patch("app.terminate_process_tree", return_value=True) as mock_terminate:
                    mock_list.return_value = [mock_job]

                    await cancel_orphaned_jobs_on_startup()

                    mock_terminate.assert_called_once_with(12345)
                    mock_update.assert_called_once()
                    call_kwargs = mock_update.call_args[1]
                    assert call_kwargs["status"] == const.CANCELLED_STATUS

    @pytest.mark.asyncio
    async def test_running_job_with_pid_termination_fails(self):
        """Test handling when process termination fails (process already exited)."""
        mock_job = MagicMock()
        mock_job.id = "orphaned-job-pid-gone"
        mock_job.status = const.RUNNING_STATUS
        mock_job.process_pid = 99999

        with patch("app.postgres_db.list_training_jobs", new_callable=AsyncMock) as mock_list:
            with patch("app.postgres_db.update_training_job", new_callable=AsyncMock) as mock_update:
                with patch("app.terminate_process_tree", return_value=False) as mock_terminate:
                    mock_list.return_value = [mock_job]

                    await cancel_orphaned_jobs_on_startup()

                    # Should still mark as cancelled even if termination fails
                    mock_terminate.assert_called_once_with(99999)
                    mock_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_queued_job_cancelled(self):
        """Test that queued jobs are marked as cancelled on startup."""
        mock_job = MagicMock()
        mock_job.id = "queued-job"
        mock_job.status = const.QUEUE_STATUS
        mock_job.process_pid = None

        with patch("app.postgres_db.list_training_jobs", new_callable=AsyncMock) as mock_list:
            with patch("app.postgres_db.update_training_job", new_callable=AsyncMock) as mock_update:
                mock_list.return_value = [mock_job]

                await cancel_orphaned_jobs_on_startup()

                mock_update.assert_called_once()
                call_kwargs = mock_update.call_args[1]
                assert call_kwargs["status"] == const.CANCELLED_STATUS

    @pytest.mark.asyncio
    async def test_exception_handling_during_cleanup(self):
        """Test that exceptions during cleanup are caught and logged."""
        with patch("app.postgres_db.list_training_jobs", new_callable=AsyncMock) as mock_list:
            mock_list.side_effect = Exception("Database connection error")

            # Should not raise, just log the error
            await cancel_orphaned_jobs_on_startup()

    @pytest.mark.asyncio
    async def test_multiple_orphaned_jobs(self):
        """Test handling multiple orphaned jobs of different types."""
        mock_running = MagicMock()
        mock_running.id = "running-1"
        mock_running.status = const.RUNNING_STATUS
        mock_running.process_pid = 1111

        mock_queued = MagicMock()
        mock_queued.id = "queued-1"
        mock_queued.status = const.QUEUE_STATUS
        mock_queued.process_pid = None

        mock_completed = MagicMock()
        mock_completed.id = "completed-1"
        mock_completed.status = const.COMPLETED_STATUS

        with patch("app.postgres_db.list_training_jobs", new_callable=AsyncMock) as mock_list:
            with patch("app.postgres_db.update_training_job", new_callable=AsyncMock) as mock_update:
                with patch("app.terminate_process_tree", return_value=True):
                    mock_list.return_value = [mock_running, mock_queued, mock_completed]

                    await cancel_orphaned_jobs_on_startup()

                    # Should update only running and queued jobs (2 calls)
                    assert mock_update.call_count == 2


class TestRunFineTuning:
    """Tests for the run_fine_tuning async function."""

    @pytest.fixture
    def mock_training_config_yaml(self):
        """Sample YAML training configuration."""
        return {
            "dataset_config": {
                "train_config": {
                    "anno_path": "",
                    "data_root": "",
                },
                "val_config": {
                    "anno_path": "",
                    "data_root": "",
                },
            },
            "training_config": {
                "output": "",
                "exp_name": "",
                "epochs": 10,
            },
            "model_config": {
                "pretrained": False,
            },
        }

    @pytest.fixture
    def mock_training_config_toml(self):
        """Sample TOML training configuration."""
        return {
            "dataset_config": {
                "train_config": {
                    "anno_path": "",
                    "data_root": "",
                },
                "val_config": {
                    "anno_path": "",
                    "data_root": "",
                },
            },
            "training_config": {
                "output": "",
                "exp_name": "",
                "epochs": 5,
            },
            "model_config": {
                "pretrained": "/path/to/model.pth",
            },
        }

    @pytest.mark.asyncio
    async def test_unsupported_config_format(self):
        """Test that unsupported config file format raises ValueError."""
        job_id = "test-job-unsupported-format"

        # Setup cache
        training_jobs_cache.set(
            job_id,
            {
                "job_id": job_id,
                "status": const.QUEUE_STATUS,
                "log_file_path": "/tmp/test.log",
            },
        )

        with patch("app.postgres_db.update_training_job", new_callable=AsyncMock):
            with patch("builtins.open", mock_open()):
                with pytest.raises(Exception):
                    await run_fine_tuning(
                        job_id,
                        "/path/to/dataset",
                        "/path/to/val_dataset",
                        "/path/to/config.json",  # Unsupported format
                    )

        # Cleanup
        training_jobs_cache.delete(job_id)

    @pytest.mark.asyncio
    async def test_yaml_config_loading(self, mock_training_config_yaml):
        """Test training with YAML configuration file."""
        job_id = "test-job-yaml"

        training_jobs_cache.set(
            job_id,
            {
                "job_id": job_id,
                "status": const.QUEUE_STATUS,
                "progress": 0.0,
                "log_file_path": "/tmp/test.log",
            },
        )

        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock(return_value=0)

        # Create async iterators for stdout/stderr
        async def empty_readuntil(separator=b"\n"):
            return b""

        mock_process.stdout.readuntil = empty_readuntil
        mock_process.stderr.readuntil = empty_readuntil

        with patch("app.read_yaml", return_value=mock_training_config_yaml):
            with patch("app.dump_yaml"):
                with patch("app.postgres_db.update_training_job", new_callable=AsyncMock):
                    with patch("os.path.exists", return_value=False):  # No validation annotation
                        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                            with patch("builtins.open", mock_open()):
                                await run_fine_tuning(
                                    job_id,
                                    "/path/to/dataset",
                                    "/path/to/val_dataset",
                                    "/path/to/config.yaml",
                                )

        # Cleanup
        if training_jobs_cache.get(job_id):
            training_jobs_cache.delete(job_id)

    @pytest.mark.asyncio
    async def test_toml_config_loading(self, mock_training_config_toml):
        """Test training with TOML configuration file."""
        job_id = "test-job-toml"

        training_jobs_cache.set(
            job_id,
            {
                "job_id": job_id,
                "status": const.QUEUE_STATUS,
                "progress": 0.0,
                "log_file_path": "/tmp/test.log",
            },
        )

        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock(return_value=0)

        async def empty_readuntil(separator=b"\n"):
            return b""

        mock_process.stdout.readuntil = empty_readuntil
        mock_process.stderr.readuntil = empty_readuntil

        with patch("app.read_toml", return_value=mock_training_config_toml):
            with patch("app.dump_toml"):
                with patch("app.postgres_db.update_training_job", new_callable=AsyncMock):
                    with patch("os.path.exists", return_value=False):
                        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                            with patch("builtins.open", mock_open()):
                                await run_fine_tuning(
                                    job_id,
                                    "/path/to/dataset",
                                    "/path/to/val_dataset",
                                    "/path/to/config.toml",
                                )

        if training_jobs_cache.get(job_id):
            training_jobs_cache.delete(job_id)

    @pytest.mark.asyncio
    async def test_validation_annotation_exists_and_valid(self, mock_training_config_yaml):
        """Test using validation annotation when it exists and is valid."""
        job_id = "test-job-valid-val"

        training_jobs_cache.set(
            job_id,
            {
                "job_id": job_id,
                "status": const.QUEUE_STATUS,
                "progress": 0.0,
                "log_file_path": "/tmp/test.log",
            },
        )

        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock(return_value=0)

        async def empty_readuntil(separator=b"\n"):
            return b""

        mock_process.stdout.readuntil = empty_readuntil
        mock_process.stderr.readuntil = empty_readuntil

        valid_annotation = {"video1": {"actions": []}}

        with patch("app.read_yaml", return_value=mock_training_config_yaml):
            with patch("app.dump_yaml"):
                with patch("app.postgres_db.update_training_job", new_callable=AsyncMock):
                    with patch("os.path.exists", return_value=True):
                        with patch("builtins.open", mock_open(read_data=json.dumps(valid_annotation))):
                            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                                await run_fine_tuning(
                                    job_id,
                                    "/path/to/dataset",
                                    "/path/to/val_dataset",
                                    "/path/to/config.yaml",
                                )

        if training_jobs_cache.get(job_id):
            training_jobs_cache.delete(job_id)

    @pytest.mark.asyncio
    async def test_validation_annotation_exists_but_empty(self, mock_training_config_yaml):
        """Test fallback when validation annotation exists but is empty."""
        job_id = "test-job-empty-val"

        training_jobs_cache.set(
            job_id,
            {
                "job_id": job_id,
                "status": const.QUEUE_STATUS,
                "progress": 0.0,
                "log_file_path": "/tmp/test.log",
            },
        )

        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock(return_value=0)

        async def empty_readuntil(separator=b"\n"):
            return b""

        mock_process.stdout.readuntil = empty_readuntil
        mock_process.stderr.readuntil = empty_readuntil

        # Empty annotation (invalid)
        empty_annotation = {}

        with patch("app.read_yaml", return_value=mock_training_config_yaml):
            with patch("app.dump_yaml"):
                with patch("app.postgres_db.update_training_job", new_callable=AsyncMock):
                    with patch("os.path.exists", return_value=True):
                        with patch("builtins.open", mock_open(read_data=json.dumps(empty_annotation))):
                            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                                await run_fine_tuning(
                                    job_id,
                                    "/path/to/dataset",
                                    "/path/to/val_dataset",
                                    "/path/to/config.yaml",
                                )

        if training_jobs_cache.get(job_id):
            training_jobs_cache.delete(job_id)

    @pytest.mark.asyncio
    async def test_pretrained_model_boolean_true(self, mock_training_config_yaml):
        """Test when pretrained is set to boolean True."""
        job_id = "test-job-pretrained-bool"
        config = mock_training_config_yaml.copy()
        config["model_config"] = {"pretrained": True}

        training_jobs_cache.set(
            job_id,
            {
                "job_id": job_id,
                "status": const.QUEUE_STATUS,
                "progress": 0.0,
                "log_file_path": "/tmp/test.log",
            },
        )

        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock(return_value=0)

        async def empty_readuntil(separator=b"\n"):
            return b""

        mock_process.stdout.readuntil = empty_readuntil
        mock_process.stderr.readuntil = empty_readuntil

        with patch("app.read_yaml", return_value=config):
            with patch("app.dump_yaml"):
                with patch("app.postgres_db.update_training_job", new_callable=AsyncMock):
                    with patch("os.path.exists", return_value=False):
                        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                            with patch("builtins.open", mock_open()):
                                await run_fine_tuning(
                                    job_id,
                                    "/path/to/dataset",
                                    "/path/to/val_dataset",
                                    "/path/to/config.yaml",
                                )

        if training_jobs_cache.get(job_id):
            training_jobs_cache.delete(job_id)

    @pytest.mark.asyncio
    async def test_pretrained_model_relative_path_not_found(self, mock_training_config_yaml):
        """Test warning when pretrained model at relative path is not found."""
        job_id = "test-job-pretrained-rel-not-found"
        config = mock_training_config_yaml.copy()
        config["model_config"] = {"pretrained": "models/checkpoint.pth"}  # Relative path

        training_jobs_cache.set(
            job_id,
            {
                "job_id": job_id,
                "status": const.QUEUE_STATUS,
                "progress": 0.0,
                "log_file_path": "/tmp/test.log",
            },
        )

        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock(return_value=0)

        async def empty_readuntil(separator=b"\n"):
            return b""

        mock_process.stdout.readuntil = empty_readuntil
        mock_process.stderr.readuntil = empty_readuntil

        with patch("app.read_yaml", return_value=config):
            with patch("app.dump_yaml"):
                with patch("app.postgres_db.update_training_job", new_callable=AsyncMock):
                    with patch("os.path.exists", return_value=False):
                        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                            with patch("builtins.open", mock_open()):
                                await run_fine_tuning(
                                    job_id,
                                    "/path/to/dataset",
                                    "/path/to/val_dataset",
                                    "/path/to/config.yaml",
                                )

        if training_jobs_cache.get(job_id):
            training_jobs_cache.delete(job_id)

    @pytest.mark.asyncio
    async def test_pretrained_model_absolute_path_not_found(self, mock_training_config_yaml):
        """Test warning when pretrained model at absolute path is not found."""
        job_id = "test-job-pretrained-abs-not-found"
        config = mock_training_config_yaml.copy()
        config["model_config"] = {"pretrained": "/absolute/path/to/model.pth"}

        training_jobs_cache.set(
            job_id,
            {
                "job_id": job_id,
                "status": const.QUEUE_STATUS,
                "progress": 0.0,
                "log_file_path": "/tmp/test.log",
            },
        )

        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock(return_value=0)

        async def empty_readuntil(separator=b"\n"):
            return b""

        mock_process.stdout.readuntil = empty_readuntil
        mock_process.stderr.readuntil = empty_readuntil

        with patch("app.read_yaml", return_value=config):
            with patch("app.dump_yaml"):
                with patch("app.postgres_db.update_training_job", new_callable=AsyncMock):
                    with patch("os.path.exists", return_value=False):
                        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                            with patch("builtins.open", mock_open()):
                                await run_fine_tuning(
                                    job_id,
                                    "/path/to/dataset",
                                    "/path/to/val_dataset",
                                    "/path/to/config.yaml",
                                )

        if training_jobs_cache.get(job_id):
            training_jobs_cache.delete(job_id)

    @pytest.mark.asyncio
    async def test_training_process_failure(self, mock_training_config_yaml):
        """Test handling when training process returns non-zero exit code."""
        job_id = "test-job-process-fail"

        training_jobs_cache.set(
            job_id,
            {
                "job_id": job_id,
                "status": const.QUEUE_STATUS,
                "progress": 0.0,
                "log_file_path": "/tmp/test.log",
            },
        )

        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock(return_value=1)  # Non-zero exit code

        async def empty_readuntil(separator=b"\n"):
            return b""

        mock_process.stdout.readuntil = empty_readuntil
        mock_process.stderr.readuntil = empty_readuntil

        with patch("app.read_yaml", return_value=mock_training_config_yaml):
            with patch("app.dump_yaml"):
                with patch("app.postgres_db.update_training_job", new_callable=AsyncMock) as mock_update:
                    with patch("os.path.exists", return_value=False):
                        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                            with patch("builtins.open", mock_open()):
                                await run_fine_tuning(
                                    job_id,
                                    "/path/to/dataset",
                                    "/path/to/val_dataset",
                                    "/path/to/config.yaml",
                                )

                    # Verify the job was marked as failed
                    # Find the call that sets status to failed
                    failed_call = None
                    for call in mock_update.call_args_list:
                        kwargs = call[1]
                        if kwargs.get("status") == const.FAILED_STATUS:
                            failed_call = call
                            break
                    assert failed_call is not None

        if training_jobs_cache.get(job_id):
            training_jobs_cache.delete(job_id)

    @pytest.mark.asyncio
    async def test_training_already_cancelled_status(self, mock_training_config_yaml):
        """Test that failed status is not set if job was cancelled during training."""
        job_id = "test-job-already-cancelled"

        training_jobs_cache.set(
            job_id,
            {
                "job_id": job_id,
                "status": const.QUEUE_STATUS,
                "progress": 0.0,
                "log_file_path": "/tmp/test.log",
            },
        )

        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()

        # Simulate cancellation happening during process.wait()
        async def mock_wait_with_cancellation():
            # Simulate the cancel endpoint setting status to CANCELLED during training
            training_jobs_cache.update(job_id, status=const.CANCELLED_STATUS)
            return 1  # Non-zero due to cancellation (SIGTERM)

        mock_process.wait = mock_wait_with_cancellation

        async def empty_readuntil(separator=b"\n"):
            return b""

        mock_process.stdout.readuntil = empty_readuntil
        mock_process.stderr.readuntil = empty_readuntil

        with patch("app.read_yaml", return_value=mock_training_config_yaml):
            with patch("app.dump_yaml"):
                with patch("app.postgres_db.update_training_job", new_callable=AsyncMock) as mock_update:
                    with patch("os.path.exists", return_value=False):
                        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                            with patch("builtins.open", mock_open()):
                                await run_fine_tuning(
                                    job_id,
                                    "/path/to/dataset",
                                    "/path/to/val_dataset",
                                    "/path/to/config.yaml",
                                )

                    # Verify final database update did NOT set status to failed
                    # The last call should preserve the CANCELLED status
                    final_call = mock_update.call_args_list[-1]
                    final_kwargs = final_call[1]
                    assert final_kwargs.get("status") != const.FAILED_STATUS

        if training_jobs_cache.get(job_id):
            training_jobs_cache.delete(job_id)

    @pytest.mark.asyncio
    async def test_training_process_with_progress_logging(self, mock_training_config_yaml):
        """Test that progress is parsed from training output."""
        job_id = "test-job-progress"

        training_jobs_cache.set(
            job_id,
            {
                "job_id": job_id,
                "status": const.QUEUE_STATUS,
                "progress": 0.0,
                "total_epochs": 10,
                "log_file_path": "/tmp/test.log",
            },
        )

        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock(return_value=0)

        # Simulate training output
        output_lines = [
            b"Epoch 0: 100%|##########| 10/10 [00:10<00:00, loss=0.5]\n",
            b"",
        ]
        line_index = [0]

        async def mock_readuntil(separator=b"\n"):
            if line_index[0] < len(output_lines):
                line = output_lines[line_index[0]]
                line_index[0] += 1
                return line
            return b""

        mock_process.stdout = AsyncMock()
        mock_process.stdout.readuntil = mock_readuntil

        async def empty_readuntil(separator=b"\n"):
            return b""

        mock_process.stderr.readuntil = empty_readuntil

        with patch("app.read_yaml", return_value=mock_training_config_yaml):
            with patch("app.dump_yaml"):
                with patch("app.postgres_db.update_training_job", new_callable=AsyncMock):
                    with patch("os.path.exists", return_value=False):
                        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                            with patch("builtins.open", mock_open()):
                                with patch(
                                    "app.parse_ddm_log",
                                    return_value={"current_step": 10, "total_steps": 100, "loss": 0.5},
                                ):
                                    await run_fine_tuning(
                                        job_id,
                                        "/path/to/dataset",
                                        "/path/to/val_dataset",
                                        "/path/to/config.yaml",
                                    )

        if training_jobs_cache.get(job_id):
            training_jobs_cache.delete(job_id)

    @pytest.mark.asyncio
    async def test_exception_during_training_setup(self):
        """Test exception handling during training setup."""
        job_id = "test-job-setup-error"

        training_jobs_cache.set(
            job_id,
            {
                "job_id": job_id,
                "status": const.QUEUE_STATUS,
                "progress": 0.0,
                "log_file_path": "/tmp/test.log",
            },
        )

        with patch("app.read_yaml", side_effect=Exception("Config read error")):
            with patch("app.postgres_db.update_training_job", new_callable=AsyncMock):
                with patch("builtins.open", mock_open()):
                    with pytest.raises(Exception):
                        await run_fine_tuning(
                            job_id,
                            "/path/to/dataset",
                            "/path/to/val_dataset",
                            "/path/to/config.yaml",
                        )

        if training_jobs_cache.get(job_id):
            training_jobs_cache.delete(job_id)
