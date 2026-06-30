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
Unit tests for data-generation-pipeline/app.py VLMAugmentationService class.
"""

import asyncio
import os
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestVLMAugmentationServiceFindVideoFolders:
    """Tests for VLMAugmentationService.find_video_folders method."""

    def test_find_video_folders_empty_directory(self, tmp_path):
        """Test finding video folders in an empty directory."""
        from app import VLMAugmentationService

        service = VLMAugmentationService()
        result = service.find_video_folders(str(tmp_path), "mp4")

        assert result == []

    def test_find_video_folders_nonexistent_path(self, tmp_path):
        """Test finding video folders when path doesn't exist."""
        from app import VLMAugmentationService

        service = VLMAugmentationService()
        nonexistent_path = str(tmp_path / "nonexistent")
        result = service.find_video_folders(nonexistent_path, "mp4")

        assert result == []

    def test_find_video_folders_with_videos(self, tmp_path):
        """Test finding video folders that contain video files."""
        from app import VLMAugmentationService

        # Create video folder with video file
        video_folder = tmp_path / "video1"
        video_folder.mkdir()
        (video_folder / "test.mp4").touch()

        service = VLMAugmentationService()
        result = service.find_video_folders(str(tmp_path), "mp4")

        assert len(result) == 1
        assert str(video_folder) in result

    def test_find_video_folders_ignores_non_video_folders(self, tmp_path):
        """Test that folders without videos are ignored."""
        from app import VLMAugmentationService

        # Create folder with non-video files
        non_video_folder = tmp_path / "docs"
        non_video_folder.mkdir()
        (non_video_folder / "readme.txt").touch()

        service = VLMAugmentationService()
        result = service.find_video_folders(str(tmp_path), "mp4")

        assert result == []

    def test_find_video_folders_ignores_sop_actions_json_folder(self, tmp_path):
        """Test that sop_actions.json folder is ignored."""
        from app import VLMAugmentationService
        import utils.constant as const

        # Create sop_actions.json folder (should be ignored)
        sop_folder = tmp_path / const.SOP_ACTIONS_JSON_NAME
        sop_folder.mkdir()
        (sop_folder / "test.mp4").touch()

        service = VLMAugmentationService()
        result = service.find_video_folders(str(tmp_path), "mp4")

        assert result == []

    def test_find_video_folders_multiple_folders(self, tmp_path):
        """Test finding multiple video folders."""
        from app import VLMAugmentationService

        # Create multiple video folders
        for i in range(3):
            video_folder = tmp_path / f"video{i}"
            video_folder.mkdir()
            (video_folder / f"test{i}.mp4").touch()

        service = VLMAugmentationService()
        result = service.find_video_folders(str(tmp_path), "mp4")

        assert len(result) == 3

    def test_find_video_folders_different_extension(self, tmp_path):
        """Test finding video folders with different extensions."""
        from app import VLMAugmentationService

        # Create folder with avi files
        video_folder = tmp_path / "video1"
        video_folder.mkdir()
        (video_folder / "test.avi").touch()

        service = VLMAugmentationService()

        # Should not find mp4 videos
        result_mp4 = service.find_video_folders(str(tmp_path), "mp4")
        assert result_mp4 == []

        # Should find avi videos
        result_avi = service.find_video_folders(str(tmp_path), "avi")
        assert len(result_avi) == 1


class TestVLMAugmentationServiceRunCmd:
    """Tests for VLMAugmentationService._run_cmd method."""

    @pytest.mark.asyncio
    async def test_run_cmd_success(self):
        """Test running a successful command."""
        from app import VLMAugmentationService

        service = VLMAugmentationService()
        # Run a simple echo command
        return_code = await service._run_cmd(["echo", "hello"])

        assert return_code == 0

    @pytest.mark.asyncio
    async def test_run_cmd_failure_raises_exception(self):
        """Test that failed command raises CalledProcessError."""
        from app import VLMAugmentationService

        service = VLMAugmentationService()

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            await service._run_cmd(["false"])  # 'false' always returns exit code 1

        assert exc_info.value.returncode == 1

    @pytest.mark.asyncio
    async def test_run_cmd_captures_output(self):
        """Test that command output is captured and logged."""
        from app import VLMAugmentationService

        service = VLMAugmentationService()

        with patch("app.app_logger") as mock_logger:
            await service._run_cmd(["echo", "test output"])
            # Verify logger was called with output
            mock_logger.info.assert_called()


class TestVLMAugmentationServiceCleanUp:
    """Tests for VLMAugmentationService.clean_up method."""

    def test_clean_up_removes_non_video_directories(self, tmp_path):
        """Test that non-video directories are removed."""
        from app import VLMAugmentationService

        # Create directories
        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        (videos_dir / "test.mp4").touch()

        other_dir = tmp_path / "other"
        other_dir.mkdir()
        (other_dir / "file.txt").touch()

        bcq_dir = tmp_path / "bcq"
        bcq_dir.mkdir()

        service = VLMAugmentationService()
        service.clean_up(str(tmp_path))

        # 'videos' should remain
        assert videos_dir.exists()
        # Other directories should be removed
        assert not other_dir.exists()
        assert not bcq_dir.exists()

    def test_clean_up_preserves_videos_folder(self, tmp_path):
        """Test that videos folder is preserved during cleanup."""
        from app import VLMAugmentationService

        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        (videos_dir / "important.mp4").touch()

        service = VLMAugmentationService()
        service.clean_up(str(tmp_path))

        assert videos_dir.exists()
        assert (videos_dir / "important.mp4").exists()

    def test_clean_up_handles_empty_directory(self, tmp_path):
        """Test cleanup on empty directory doesn't raise."""
        from app import VLMAugmentationService

        service = VLMAugmentationService()
        # Should not raise
        service.clean_up(str(tmp_path))

    def test_clean_up_logs_errors_on_failure(self, tmp_path):
        """Test that errors during cleanup are logged."""
        from app import VLMAugmentationService

        # Create a directory
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        service = VLMAugmentationService()

        with patch("shutil.rmtree", side_effect=OSError("Permission denied")):
            with patch("app.app_logger") as mock_logger:
                service.clean_up(str(tmp_path))
                mock_logger.error.assert_called()


class TestVLMAugmentationServiceConfigToBcq:
    """Tests for VLMAugmentationService._config_to_bcq method."""

    @pytest.mark.asyncio
    async def test_config_to_bcq_builds_correct_command(self):
        """Test that BCQ generation builds correct command."""
        from app import VLMAugmentationService

        service = VLMAugmentationService()

        augment_config = {
            "video_extention": "mp4",
            "bcq": {
                "subject": "operator",
                "negative_ratio": "2.5",
                "exclude_action": "idle",
            },
        }

        with patch.object(service, "_run_cmd", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = 0

            result = await service._config_to_bcq(
                video_root="/data/videos",
                output_root="/output",
                output_name="bcq",
                actions_json="/data/actions.json",
                augment_config=augment_config,
            )

            assert result is True
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "vlm_aug.config_to_bcq" in cmd
            assert "--subject" in cmd
            assert "operator" in cmd
            assert "--negative-ratio" in cmd
            assert "2.5" in cmd

    @pytest.mark.asyncio
    async def test_config_to_bcq_uses_defaults(self):
        """Test that BCQ generation uses default values when not specified."""
        from app import VLMAugmentationService
        import utils.constant as const

        service = VLMAugmentationService()

        augment_config = {
            "bcq": {},  # Empty config, should use defaults
        }

        with patch.object(service, "_run_cmd", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = 0

            await service._config_to_bcq(
                video_root="/data/videos",
                output_root="/output",
                output_name="bcq",
                actions_json="/data/actions.json",
                augment_config=augment_config,
            )

            cmd = mock_run.call_args[0][0]
            assert const.DEFAULT_SUBJECT in cmd
            assert const.DEFAULT_VIDEO_EXTENSION in cmd


class TestVLMAugmentationServiceConfigToMcq:
    """Tests for VLMAugmentationService._config_to_mcq method."""

    @pytest.mark.asyncio
    async def test_config_to_mcq_builds_correct_command(self):
        """Test that MCQ generation builds correct command."""
        from app import VLMAugmentationService

        service = VLMAugmentationService()

        augment_config = {
            "video_extention": "mp4",
            "sequential_mcq": {
                "exclude_action": "idle",
                "max_chunk_len": "3",
            },
        }

        with patch.object(service, "_run_cmd", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = 0

            result = await service._config_to_mcq(
                video_root="/data/videos",
                output_root="/output",
                output_name="mcq",
                actions_json="/data/actions.json",
                augment_config=augment_config,
            )

            assert result is True
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "vlm_aug.config_to_sequential_mcq" in cmd
            assert "--max-chunk-len" in cmd
            assert "3" in cmd


class TestVLMAugmentationServiceGoldenGqaToGqa:
    """Tests for VLMAugmentationService._golden_gqa_to_gqa method."""

    @pytest.mark.asyncio
    async def test_golden_gqa_to_gqa_builds_correct_command(self):
        """Test that GQA generation builds correct command."""
        from app import VLMAugmentationService

        service = VLMAugmentationService()

        augment_config = {
            "video_extention": "mp4",
            "gqas": {
                "exclude_action": "idle",
            },
        }

        with patch.object(service, "_run_cmd", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = 0

            result = await service._golden_gqa_to_gqa(
                video_root="/data/videos",
                output_root="/output",
                output_name="golden_gqa",
                actions_json="/data/actions.json",
                augment_config=augment_config,
            )

            assert result is True
            cmd = mock_run.call_args[0][0]
            assert "vlm_aug.golden_gqa_to_gqa" in cmd


class TestVLMAugmentationServiceGqaToGqas:
    """Tests for VLMAugmentationService._gqa_to_gqas method."""

    @pytest.mark.asyncio
    async def test_gqa_to_gqas_builds_correct_command(self):
        """Test that GQA to GQAs generation builds correct command."""
        from app import VLMAugmentationService

        service = VLMAugmentationService()

        augment_config = {
            "video_extention": "mp4",
            "gqas": {
                "llm_type": "nvidia",
                "llm": "meta/llama3-70b-instruct",
                "num_qa_llm": "8",
                "num_qa_per_chunk": "2",
                "exclude_action": "",
                "local_llm_url": "",
            },
        }

        with patch.object(service, "_run_cmd", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = 0

            with patch.dict(os.environ, {"NGC_API_KEY": "test-key"}):
                result = await service._gqa_to_gqas(
                    video_root="/data/videos",
                    output_root="/output",
                    output_name="gqas",
                    actions_json="/data/actions.json",
                    augment_config=augment_config,
                )

            assert result is True
            cmd = mock_run.call_args[0][0]
            assert "vlm_aug.gqa_to_gqas" in cmd
            assert "--llm-type" in cmd
            assert "--api-key" not in cmd
            assert "test-key" not in cmd
            subprocess_env = mock_run.call_args.kwargs["env"]
            # NGC key is sourced only from the NGC_API_KEY env var, never from config
            assert subprocess_env["NGC_API_KEY"] == "test-key"

    @pytest.mark.asyncio
    async def test_gqa_to_gqas_uses_env_var_fallback(self):
        """Test that GQA to GQAs uses environment variable as fallback for API key."""
        from app import VLMAugmentationService

        service = VLMAugmentationService()

        augment_config = {
            "video_extention": "mp4",
            "gqas": {
                # NGC key is read from the NGC_API_KEY env var (operator .env)
                "llm_type": "nvidia",
            },
        }

        with patch.object(service, "_run_cmd", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = 0

            with patch.dict(os.environ, {"NGC_API_KEY": "env-key"}):
                await service._gqa_to_gqas(
                    video_root="/data/videos",
                    output_root="/output",
                    output_name="gqas",
                    actions_json="/data/actions.json",
                    augment_config=augment_config,
                )

                cmd = mock_run.call_args[0][0]
                # Key rides in the subprocess env, never in argv.
                assert "--api-key" not in cmd
                assert "env-key" not in cmd
                subprocess_env = mock_run.call_args.kwargs["env"]
                assert subprocess_env["NGC_API_KEY"] == "env-key"

    @pytest.mark.asyncio
    async def test_spatial_localization_keeps_key_out_of_argv(self):
        """Spatial localization must also pass the API key via env, not argv."""
        from app import VLMAugmentationService

        service = VLMAugmentationService()

        augment_config = {
            "video_extention": "mp4",
            "gqas": {
                "llm_type": "nvidia",
                "local_llm_url": "",
            },
            "spatial_localization": {},
        }

        with patch.object(service, "_run_cmd", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = 0

            with patch.dict(os.environ, {"NGC_API_KEY": "test-key"}):
                await service._spatial_localization(
                    video_root="/data/videos",
                    output_root="/output",
                    output_name="spatial_localization",
                    actions_json="/data/actions.json",
                    augment_config=augment_config,
                )

            cmd = mock_run.call_args[0][0]
            assert "vlm_aug.spatial_localization" in cmd
            assert "--api-key" not in cmd
            assert "test-key" not in cmd
            subprocess_env = mock_run.call_args.kwargs["env"]
            # NGC key is sourced only from the NGC_API_KEY env var, never from config
            assert subprocess_env["NGC_API_KEY"] == "test-key"


class TestVLMAugmentationServiceConfigToDmcq:
    """Tests for VLMAugmentationService._config_to_dmcq method."""

    @pytest.mark.asyncio
    async def test_config_to_dmcq_builds_correct_command(self):
        """Test that Dynamic MCQ generation builds correct command."""
        from app import VLMAugmentationService

        service = VLMAugmentationService()

        augment_config = {
            "video_extention": "mp4",
            "dynamic_mcq": {
                "exclude_action": "",
                "min_options": "3",
                "max_options": "6",
                "non_sop_action": "other_action",
                "num_pos": "1",
                "num_neg": "2",
                "num_hard_pos": "0",
                "num_hard_neg": "0",
                "hard_neg_mode": "",
                "hard_pos_mode": "",
                "confusion_map": "",
            },
        }

        with patch.object(service, "_run_cmd", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = 0

            result = await service._config_to_dmcq(
                video_root="/data/videos",
                output_root="/output",
                output_name="dmcq",
                actions_json="/data/actions.json",
                augment_config=augment_config,
            )

            assert result is True
            cmd = mock_run.call_args[0][0]
            assert "vlm_aug.config_to_dynamic_mcq" in cmd
            assert "--non-sop-action" in cmd

    @pytest.mark.asyncio
    async def test_config_to_dmcq_raises_without_non_sop_action(self):
        """Test that Dynamic MCQ raises HTTPException when non_sop_action is not set."""
        from app import VLMAugmentationService
        from fastapi import HTTPException

        service = VLMAugmentationService()

        augment_config = {
            "video_extention": "mp4",
            "dynamic_mcq": {
                # non_sop_action not set
            },
        }

        with pytest.raises(HTTPException) as exc_info:
            await service._config_to_dmcq(
                video_root="/data/videos",
                output_root="/output",
                output_name="dmcq",
                actions_json="/data/actions.json",
                augment_config=augment_config,
            )

        assert exc_info.value.status_code == 400
        assert "non_sop_action" in exc_info.value.detail


class TestVLMAugmentationServiceConfigToDs:
    """Tests for VLMAugmentationService._config_to_ds method."""

    @pytest.mark.asyncio
    async def test_config_to_ds_builds_correct_command(self):
        """Test that Dynamic Shuffling generation builds correct command."""
        from app import VLMAugmentationService

        service = VLMAugmentationService()

        augment_config = {
            "video_extention": "mp4",
            "dynamic_shuffling": {
                "exclude_action": "",
                "min_distractor": "3",
                "max_distractor": "6",
                "non_sop_action": "other_action",
                "num_runs": "1",
                "num_hard_neg": "0",
                "hard_neg_frames_ratio": "0.1",
            },
        }

        with patch.object(service, "_run_cmd", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = 0

            result = await service._config_to_ds(
                video_root="/data/videos",
                output_root="/output",
                output_name="ds",
                actions_json="/data/actions.json",
                augment_config=augment_config,
            )

            assert result is True
            cmd = mock_run.call_args[0][0]
            assert "vlm_aug.config_to_dynamic_shuffling" in cmd
            assert "--min-distractor" in cmd

    @pytest.mark.asyncio
    async def test_config_to_ds_raises_without_non_sop_action(self):
        """Test that Dynamic Shuffling raises HTTPException when non_sop_action is not set."""
        from app import VLMAugmentationService
        from fastapi import HTTPException

        service = VLMAugmentationService()

        augment_config = {
            "video_extention": "mp4",
            "dynamic_shuffling": {
                # non_sop_action not set
            },
        }

        with pytest.raises(HTTPException) as exc_info:
            await service._config_to_ds(
                video_root="/data/videos",
                output_root="/output",
                output_name="ds",
                actions_json="/data/actions.json",
                augment_config=augment_config,
            )

        assert exc_info.value.status_code == 400
        assert "non_sop_action" in exc_info.value.detail


class TestVLMAugmentationServiceConfigToEn:
    """Tests for VLMAugmentationService._config_to_en method."""

    @pytest.mark.asyncio
    async def test_config_to_en_builds_correct_command(self):
        """Test that Extra Negative generation builds correct command."""
        from app import VLMAugmentationService

        service = VLMAugmentationService()

        augment_config = {
            "video_extention": "mp4",
            "extra_negative": {
                "exclude_action": "",
                "min_options": "3",
                "max_options": "6",
                "non_sop_action": "other_action",
                "extra_negative_data_id": "extra_data_001",
                "num_runs": "1",
                "generate_all_options": True,
            },
        }

        with patch.object(service, "_run_cmd", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = 0

            result = await service._config_to_en(
                video_root="/data/videos",
                output_root="/output",
                output_name="en",
                actions_json="/data/actions.json",
                augment_config=augment_config,
            )

            assert result is True
            cmd = mock_run.call_args[0][0]
            assert "vlm_aug.config_to_extra_negative" in cmd
            assert "--generate-all-options" in cmd

    @pytest.mark.asyncio
    async def test_config_to_en_raises_without_non_sop_action(self):
        """Test that Extra Negative raises HTTPException when non_sop_action is not set."""
        from app import VLMAugmentationService
        from fastapi import HTTPException

        service = VLMAugmentationService()

        augment_config = {
            "video_extention": "mp4",
            "extra_negative": {
                "extra_negative_data_id": "extra_data_001",
                # non_sop_action not set
            },
        }

        with pytest.raises(HTTPException) as exc_info:
            await service._config_to_en(
                video_root="/data/videos",
                output_root="/output",
                output_name="en",
                actions_json="/data/actions.json",
                augment_config=augment_config,
            )

        assert exc_info.value.status_code == 400
        assert "non_sop_action" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_config_to_en_raises_without_extra_negative_data_id(self):
        """Test that Extra Negative raises HTTPException when extra_negative_data_id is not set."""
        from app import VLMAugmentationService
        from fastapi import HTTPException

        service = VLMAugmentationService()

        augment_config = {
            "video_extention": "mp4",
            "extra_negative": {
                "non_sop_action": "other_action",
                # extra_negative_data_id not set
            },
        }

        with pytest.raises(HTTPException) as exc_info:
            await service._config_to_en(
                video_root="/data/videos",
                output_root="/output",
                output_name="en",
                actions_json="/data/actions.json",
                augment_config=augment_config,
            )

        assert exc_info.value.status_code == 400
        assert "extra_negative_data_id" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_config_to_en_without_generate_all_options(self):
        """Test Extra Negative without generate_all_options flag."""
        from app import VLMAugmentationService

        service = VLMAugmentationService()

        augment_config = {
            "video_extention": "mp4",
            "extra_negative": {
                "non_sop_action": "other_action",
                "extra_negative_data_id": "extra_data_001",
                "generate_all_options": False,
            },
        }

        with patch.object(service, "_run_cmd", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = 0

            await service._config_to_en(
                video_root="/data/videos",
                output_root="/output",
                output_name="en",
                actions_json="/data/actions.json",
                augment_config=augment_config,
            )

            cmd = mock_run.call_args[0][0]
            # Should have empty string instead of --generate-all-options
            assert "--generate-all-options" not in cmd


class TestVLMAugmentationServiceProcessAllActions:
    """Tests for VLMAugmentationService.process_all_actions method."""

    @pytest.mark.asyncio
    async def test_process_all_actions_no_video_folders_marks_failed(self, tmp_path):
        """Validate-time failure must mark the augmentation row FAILED and NOT
        leak an exception out of the fire-and-forget task.

        Previously this raised an HTTPException that escaped the unawaited task,
        leaving the row stuck at "running" forever.
        """
        from app import VLMAugmentationService
        import utils.constant as const

        service = VLMAugmentationService()

        with patch("app.postgres_db") as mock_db:
            mock_db.insert_data = AsyncMock()
            mock_db.update_data = AsyncMock()

            with patch("shutil.rmtree"):
                # Empty tmp_path => no video folders => _validate_video_folders
                # raises internally; the backstop must catch it and resolve the
                # row to FAILED without re-raising.
                await service.process_all_actions(
                    label_data_path=str(tmp_path),
                    dataset_path=str(tmp_path / "output"),
                    actions_json=str(tmp_path / "actions.json"),
                    augment_config={"video_extention": "mp4"},
                    dataset_id="test-dataset",
                )

        failed_calls = [
            c for c in mock_db.update_data.call_args_list
            if c.kwargs.get("id") == "test-dataset"
            and c.kwargs.get("status") == const.FAILED_STATUS
        ]
        assert failed_calls, "augmentation row must be marked FAILED on validate failure"

    @pytest.mark.asyncio
    async def test_process_all_actions_runs_enabled_stages(self, tmp_path):
        """Test that process_all_actions runs only enabled stages."""
        from app import VLMAugmentationService
        import utils.constant as const

        service = VLMAugmentationService()

        # Create video folder
        video_folder = tmp_path / "video1"
        video_folder.mkdir()
        (video_folder / "test.mp4").touch()

        # Create output folder
        output_path = tmp_path / "output"
        output_path.mkdir()

        augment_config = {
            "video_extention": "mp4",
            const.STAGE_CONFIG_TO_BCQ: {"enable": True, "subject": "operator"},
        }

        with patch("app.postgres_db") as mock_db:
            mock_db.insert_data = AsyncMock()
            mock_db.update_data = AsyncMock()

            with patch.object(service, "_config_to_bcq", new_callable=AsyncMock) as mock_bcq:
                mock_bcq.return_value = True

                with patch.object(service, "clean_up"):
                    await service.process_all_actions(
                        label_data_path=str(tmp_path),
                        dataset_path=str(output_path),
                        actions_json=str(tmp_path / "actions.json"),
                        augment_config=augment_config,
                        dataset_id="test-dataset",
                    )

                    mock_bcq.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_all_actions_handles_failure(self, tmp_path):
        """Stage failure must resolve the row to FAILED and NOT leak an exception
        out of the fire-and-forget task."""
        from app import VLMAugmentationService
        import utils.constant as const

        service = VLMAugmentationService()

        # Create video folder
        video_folder = tmp_path / "video1"
        video_folder.mkdir()
        (video_folder / "test.mp4").touch()

        # Create output folder
        output_path = tmp_path / "output"
        output_path.mkdir()

        augment_config = {
            "video_extention": "mp4",
            const.STAGE_CONFIG_TO_BCQ: {"enable": True},
        }

        with patch("app.postgres_db") as mock_db:
            mock_db.insert_data = AsyncMock()
            mock_db.update_data = AsyncMock()

            with patch.object(
                service, "_config_to_bcq", new_callable=AsyncMock, side_effect=Exception("BCQ failed")
            ):
                with patch("shutil.rmtree"):
                    # No raise: the task resolves to a terminal FAILED state.
                    await service.process_all_actions(
                        label_data_path=str(tmp_path),
                        dataset_path=str(output_path),
                        actions_json=str(tmp_path / "actions.json"),
                        augment_config=augment_config,
                        dataset_id="test-dataset",
                    )

        failed_calls = [
            c for c in mock_db.update_data.call_args_list
            if c.kwargs.get("id") == "test-dataset"
            and c.kwargs.get("status") == const.FAILED_STATUS
        ]
        assert failed_calls, "augmentation row must be marked FAILED on stage failure"
