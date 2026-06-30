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


import asyncio
import os
import shutil
import subprocess
import traceback
from datetime import datetime
from pprint import pformat
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import utils.constant as const
from components.postgres_db import postgres_db
from utils.logger import app_logger
from utils.utils import (
    clean_and_create_dir,
    load_config_yaml,
    safe_dataset_path,
    scrub_secrets,
)
from validation.postgres_validation import Augmentation, AugmentationStage, Chunk, Video
from validation.response_validation import (
    AugmentationStatusResponse,
    AugResponse,
)

app = FastAPI(
    title="VLM Data Augmentation API",
    description="FastAPI service for VLM data augmentation - processes all four actions automatically",
)

# allow_credentials=True is deliberately omitted: combined with
# allow_origins=["*"] modern browsers reject the request. Frontend talks
# through nginx (same-origin), so no credentialed cross-origin is needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reject oversized request bodies up front to mitigate resource exhaustion (T17 / FSR-AVA-1).
# Configurable via MAX_REQUEST_BODY_MB (default 2048 MB, matching the nginx proxy cap).
MAX_REQUEST_BODY_BYTES = int(os.getenv("MAX_REQUEST_BODY_MB", "2048")) * 1024 * 1024


@app.middleware("http")
async def limit_request_body_size(request, call_next):
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            oversized = int(content_length) > MAX_REQUEST_BODY_BYTES
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})
        if oversized:
            return JSONResponse(status_code=413, content={"detail": "Request body too large"})
    return await call_next(request)


class VLMAugmentationService:
    def __init__(self):
        # Intentionally left blank: no initialization logic is required at this time.
        pass

    async def _run_cmd(self, cmd, env=None):
        """Run asynchronous subprocess with basic environment setup.

        Secrets (e.g. the NGC API key) are passed via *env*, never in *cmd*, so
        they cannot leak through the logged command line, process listings, or a
        CalledProcessError carrying the argv. scrub_secrets is applied to the log
        line as defense-in-depth.
        """
        app_logger.info(f"Command: {scrub_secrets(' '.join(cmd))}")

        # Create async subprocess. env=None inherits the parent environment.
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        async def read_stream(stream, prefix):
            """Read and log stream output in real-time"""
            while True:
                line_bytes = await stream.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8").strip()
                if line:
                    app_logger.info(f"{prefix}: {scrub_secrets(line)}")

        # Read stdout and stderr in real-time
        await asyncio.gather(
            read_stream(proc.stdout, "Stdout"),
            read_stream(proc.stderr, "Stderr"),
        )

        # Wait for process to complete
        return_code = await proc.wait()

        app_logger.info(f"Return code: {return_code}")

        # Raise an exception if the command failed
        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code, cmd, proc.stdout, proc.stderr
            )

        return return_code

    def find_video_folders(
        self, label_data_path: str, video_extension: str
    ) -> List[str]:
        """Find all video folders within the label_data_id directory"""
        video_folders = []
        if not os.path.exists(label_data_path):
            return video_folders

        for item in os.listdir(label_data_path):
            item_path = os.path.join(label_data_path, item)
            if os.path.isdir(item_path) and item != const.SOP_ACTIONS_JSON_NAME:
                # Check if this folder contains video files
                has_videos = any(
                    f.lower().endswith(f".{video_extension}")
                    for f in os.listdir(item_path)
                    if os.path.isfile(os.path.join(item_path, f))
                )
                if has_videos:
                    video_folders.append(item_path)
        return video_folders

    async def _config_to_bcq(
        self,
        video_root: str,
        output_root: str,
        output_name: str,
        actions_json: str,
        augment_config: Dict[str, Any],
    ) -> bool:
        """Binary Choice Questions generation"""

        subject = augment_config["bcq"].get("subject", const.DEFAULT_SUBJECT)
        negative_ratio = augment_config["bcq"].get("negative_ratio", "2.0")
        ext = augment_config.get("video_extention", const.DEFAULT_VIDEO_EXTENSION)
        exclude_action = augment_config["bcq"].get("exclude_action", "")

        cmd = [
            "python",
            "-m",
            "vlm_aug.config_to_bcq",
            "--action-json",
            actions_json,
            "--subject",
            subject,
            "--video-root",
            video_root,
            "--ext",
            ext,
            "--exclude-action",
            exclude_action,
            "--negative-ratio",
            str(negative_ratio),
            "--output-root",
            output_root,
            "--output-name",
            output_name,
        ]

        # run command
        await self._run_cmd(cmd)

        return True

    async def _config_to_mcq(
        self,
        video_root: str,
        output_root: str,
        output_name: str,
        actions_json: str,
        augment_config: Dict[str, Any],
    ) -> bool:
        """Multiple Choice Questions generation"""

        exclude_action = augment_config["sequential_mcq"].get("exclude_action", "")
        ext = augment_config.get("video_extention", const.DEFAULT_VIDEO_EXTENSION)
        max_chunk_len = augment_config["sequential_mcq"].get("max_chunk_len", "2")

        cmd = [
            "python",
            "-m",
            "vlm_aug.config_to_sequential_mcq",
            "--action-json",
            actions_json,
            "--video-root",
            video_root,
            "--exclude-action",
            exclude_action,
            "--ext",
            ext,
            "--max-chunk-len",
            str(max_chunk_len),
            "--output-root",
            output_root,
            "--output-name",
            output_name,
        ]

        # run command
        await self._run_cmd(cmd)

        return True

    async def _golden_gqa_to_gqa(
        self,
        video_root: str,
        output_root: str,
        output_name: str,
        actions_json: str,
        augment_config: Dict[str, Any],
    ) -> bool:
        """Golden GQA to GQA conversion"""

        ext = augment_config.get("video_extention", const.DEFAULT_VIDEO_EXTENSION)
        exclude_action = augment_config["gqas"].get("exclude_action", "")
        two_operator_mode = augment_config["gqas"].get("two_operator_mode", "false")

        cmd = [
            "python",
            "-m",
            "vlm_aug.golden_gqa_to_gqa",
            "--action-json",
            actions_json,
            "--video-root",
            video_root,
            "--exclude-action",
            exclude_action,
            "--ext",
            ext,
            "--two-operator-mode",
            str(two_operator_mode),
            "--output-root",
            output_root,
            "--output-name",
            output_name,
        ]

        # run command
        await self._run_cmd(cmd)

        return True

    async def _gqa_to_gqas(
        self,
        video_root: str,
        output_root: str,
        output_name: str,
        actions_json: str,
        augment_config: Dict[str, Any],
    ) -> bool:
        """GQA to multiple GQAs using LLM"""

        # NGC key comes only from the NGC_API_KEY environment variable (operator .env)
        ngc_api_key = os.getenv("NGC_API_KEY", "")
        llm_type = augment_config["gqas"].get("llm_type", "nvidia")
        local_llm_url = augment_config["gqas"].get("local_llm_url", "")
        llm = augment_config["gqas"].get("llm", const.DEFAULT_LLM)
        num_qa_llm = augment_config["gqas"].get("num_qa_llm", "8")
        num_qa_per_chunk = augment_config["gqas"].get("num_qa_per_chunk", "2")
        ext = augment_config.get("video_extention", const.DEFAULT_VIDEO_EXTENSION)
        exclude_action = augment_config["gqas"].get("exclude_action", "")
        enable_thinking = str(augment_config["gqas"].get("enable_thinking", ""))
        two_operator_mode = augment_config["gqas"].get("two_operator_mode", "false")

        cmd = [
            "python",
            "-m",
            "vlm_aug.gqa_to_gqas",
            "--llm-type",
            llm_type,
            "--llm",
            llm,
            "--local-llm-url",
            local_llm_url,
            "--enable-thinking",
            enable_thinking,
            "--action-json",
            actions_json,
            "--num-qa-llm",
            str(num_qa_llm),
            "--num-qa-per-chunk",
            str(num_qa_per_chunk),
            "--video-root",
            video_root,
            "--ext",
            ext,
            "--exclude-action",
            exclude_action,
            "--two-operator-mode",
            str(two_operator_mode),
            "--output-root",
            output_root,
            "--output-name",
            output_name,
        ]

        await self._run_cmd(cmd, env={**os.environ, "NGC_API_KEY": ngc_api_key})

        return True

    async def _config_to_dmcq(
        self,
        video_root: str,
        output_root: str,
        output_name: str,
        actions_json: str,
        augment_config: Dict[str, Any],
    ) -> bool:
        """Dynamic MCQ generation"""

        # Check request first, then environment variables as fallback
        ext = augment_config.get("video_extention", const.DEFAULT_VIDEO_EXTENSION)
        exclude_action = augment_config["dynamic_mcq"].get("exclude_action", "")
        min_options = augment_config["dynamic_mcq"].get("min_options", "3")
        max_options = augment_config["dynamic_mcq"].get("max_options", "6")
        num_pos = augment_config["dynamic_mcq"].get("num_pos", "1")
        num_neg = augment_config["dynamic_mcq"].get("num_neg", "2")
        num_hard_pos = augment_config["dynamic_mcq"].get("num_hard_pos", "0")
        num_hard_neg = augment_config["dynamic_mcq"].get("num_hard_neg", "0")
        hard_neg_mode = augment_config["dynamic_mcq"].get("hard_neg_mode", "")
        hard_pos_mode = augment_config["dynamic_mcq"].get("hard_pos_mode", "")
        confusion_map = augment_config["dynamic_mcq"].get("confusion_map", "")

        # make sure non-sop-action is set
        non_sop_action = augment_config["dynamic_mcq"].get("non_sop_action", None)
        if not non_sop_action:
            raise HTTPException(
                status_code=400, detail="non_sop_action action must be set for Dynamic MCQ generation"
            )

        cmd = [
            "python",
            "-m",
            "vlm_aug.config_to_dynamic_mcq",
            "--action-json",
            actions_json,
            "--video-root",
            video_root,
            "--ext",
            ext,
            "--exclude-action",
            exclude_action,
            "--min-options",
            str(min_options),
            "--max-options",
            str(max_options),
            "--non-sop-action",
            str(non_sop_action),
            "--num-pos",
            str(num_pos),
            "--num-neg",
            str(num_neg),
            "--num-hard-pos",
            str(num_hard_pos),
            "--num-hard-neg",
            str(num_hard_neg),
            "--hard-neg-mode",
            hard_neg_mode,
            "--hard-pos-mode",
            hard_pos_mode,
            "--confusion-map",
            confusion_map,
            "--output-root",
            output_root,
            "--output-name",
            output_name,
        ]

        # run command
        await self._run_cmd(cmd)

        return True

    async def _config_to_ds(
        self,
        video_root: str,
        output_root: str,
        output_name: str,
        actions_json: str,
        augment_config: Dict[str, Any],
    ) -> bool:
        """Dynamic Shuffling generation"""

        # Check request first, then environment variables as fallback
        ext = augment_config.get("video_extention", const.DEFAULT_VIDEO_EXTENSION)
        exclude_action = augment_config["dynamic_shuffling"].get("exclude_action", "")
        min_distractor = augment_config["dynamic_shuffling"].get("min_distractor", "3")
        max_distractor = augment_config["dynamic_shuffling"].get("max_distractor", "6")
        num_runs = augment_config["dynamic_shuffling"].get("num_runs", "1")
        num_hard_neg = augment_config["dynamic_shuffling"].get("num_hard_neg", "0")
        hard_neg_frames_ratio = augment_config["dynamic_shuffling"].get("hard_neg_frames_ratio", "0.1")

        # make sure non-sop-action is set
        non_sop_action = augment_config["dynamic_shuffling"].get("non_sop_action", None)
        if not non_sop_action:
            raise HTTPException(
                status_code=400, detail="non_sop_action action must be set for Dynamic Shuffling generation"
            )

        cmd = [
            "python",
            "-m",
            "vlm_aug.config_to_dynamic_shuffling",
            "--action-json",
            actions_json,
            "--video-root",
            video_root,
            "--ext",
            ext,
            "--exclude-action",
            exclude_action,
            "--min-distractor",
            str(min_distractor),
            "--max-distractor",
            str(max_distractor),
            "--non-sop-action",
            str(non_sop_action),
            "--num-runs",
            str(num_runs),
            "--num-hard-neg",
            str(num_hard_neg),
            "--hard-neg-frames-ratio",
            str(hard_neg_frames_ratio),
            "--output-root",
            output_root,
            "--output-name",
            output_name,
        ]

        # run command
        await self._run_cmd(cmd)

        return True

    async def _config_to_en(
        self,
        video_root: str,
        output_root: str,
        output_name: str,
        actions_json: str,
        augment_config: Dict[str, Any],
    ) -> bool:
        """Extra Negative generation"""

        # Check request first, then environment variables as fallback
        ext = augment_config.get("video_extention", const.DEFAULT_VIDEO_EXTENSION)
        exclude_action = augment_config["extra_negative"].get("exclude_action", "")
        min_options = augment_config["extra_negative"].get("min_options", "3")
        max_options = augment_config["extra_negative"].get("max_options", "6")
        num_runs = augment_config["extra_negative"].get("num_runs", "1")
        generate_all_options = augment_config["extra_negative"].get("generate_all_options", True)

        # make sure non-sop-action and extra data id    are set
        non_sop_action = augment_config["extra_negative"].get("non_sop_action", None)
        if not non_sop_action:
            raise HTTPException(
                status_code=400, detail="non_sop_action action must be set for Extra Negative generation"
            )
        extra_negative_data_id = augment_config["extra_negative"].get("extra_negative_data_id", None)
        if not extra_negative_data_id:
            raise HTTPException(
                status_code=400, detail="extra_negative_data_id must be set for Extra Negative generation"
            )

        # Confine the cross-dataset reference to DATASET_ROOT (reject separators / traversal). (T14)
        try:
            extra_negative_video_root = safe_dataset_path(const.DATASET_ROOT, extra_negative_data_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        cmd = [
            "python",
            "-m",
            "vlm_aug.config_to_extra_negative",
            "--action-json",
            actions_json,
            "--video-root",
            extra_negative_video_root,
            "--exclude-action",
            exclude_action,
            "--ext",
            ext,
            "--min-options",
            str(min_options),
            "--max-options",
            str(max_options),
            "--non-sop-action",
            str(non_sop_action),
            "--num-runs",
            str(num_runs),
            "--generate-all-options" if generate_all_options else "",
            "--output-root",
            output_root,
            "--output-name",
            output_name,
        ]

        # run command
        await self._run_cmd(cmd)

        return True

    async def _spatial_localization(
        self,
        video_root: str,
        output_root: str,
        output_name: str,
        actions_json: str,
        augment_config: Dict[str, Any],
    ) -> bool:
        """Spatial localization augmentation with auto confusion pair generation"""

        # NGC key comes only from the NGC_API_KEY environment variable (operator .env)
        ngc_api_key = os.getenv("NGC_API_KEY", "")
        llm_type = augment_config["gqas"].get("llm_type", "nvidia")
        local_llm_url = augment_config["gqas"].get("local_llm_url", "")
        llm = augment_config["gqas"].get("llm", const.DEFAULT_LLM)
        enable_thinking = str(augment_config["gqas"].get("enable_thinking", ""))
        ext = augment_config.get("video_extention", const.DEFAULT_VIDEO_EXTENSION)
        two_operator_mode = augment_config["gqas"].get("two_operator_mode", "false")

        sl_cfg = augment_config.get("spatial_localization", {})
        exclude_action = sl_cfg.get("exclude_action", "")
        max_qa_per_group = sl_cfg.get("max_qa_per_group", "3")
        max_question_tokens = sl_cfg.get("max_question_tokens", "60")

        cmd = [
            "python",
            "-m",
            "vlm_aug.spatial_localization",
            "--action-json",
            actions_json,
            "--video-root",
            video_root,
            "--ext",
            ext,
            "--output-root",
            output_root,
            "--output-name",
            output_name,
            "--exclude-action",
            exclude_action,
            "--two-operator-mode",
            str(two_operator_mode),
            "--max-qa-per-group",
            str(max_qa_per_group),
            "--max-question-tokens",
            str(max_question_tokens),
            "--llm-type",
            llm_type,
            "--llm",
            llm,
            "--local-llm-url",
            local_llm_url,
            "--enable-thinking",
            enable_thinking,
        ]

        await self._run_cmd(cmd, env={**os.environ, "NGC_API_KEY": ngc_api_key})
        return True

    async def _frame_drop(
        self,
        dataset_path: str,
        augment_config: Dict[str, Any],
        datasets: List[str],
        iterations: int = 1,
    ) -> bool:
        """Run frame drop post-processing on all augmentation outputs"""

        fd_cfg = augment_config.get("frame_drop", {})
        dropout_rate = fd_cfg.get("dropout_rate", "0.2")
        seed = fd_cfg.get("seed", "42")

        cmd = [
            "python",
            "-m",
            "vlm_aug.frame_drop",
            "--base-dir",
            dataset_path,
            "--datasets",
        ] + datasets + [
            "--dropout-rate",
            str(dropout_rate),
            "--iterations",
            str(iterations),
            "--seed",
            str(seed),
        ]

        await self._run_cmd(cmd)
        return True

    def clean_up(self, output_root: str):
        """Clean up by remove all subdirectories except 'videos'"""

        # Remove ALL subdirectories except 'videos'
        directories_to_remove = []

        for item in os.listdir(output_root):
            item_path = os.path.join(output_root, item)
            if os.path.isdir(item_path) and item != "videos":
                directories_to_remove.append(item_path)

        # Remove directories (with contents)
        for dir_path in directories_to_remove:
            try:
                shutil.rmtree(dir_path)
                app_logger.info(f"Removed directory: {dir_path}")
            except OSError as e:
                app_logger.error(f"Failed to remove directory {dir_path}: {e}")

    def _validate_video_folders(
        self, label_data_path: str, augment_config: Dict[str, Any]
    ) -> List[str]:
        """Validate that video folders exist in label data path."""
        video_folders = self.find_video_folders(
            label_data_path,
            augment_config.get("video_extention", const.DEFAULT_VIDEO_EXTENSION),
        )
        if not video_folders:
            raise HTTPException(
                status_code=400, detail=f"No video folders found in {label_data_path}"
            )
        return video_folders

    def _build_stage_config(
        self, augment_config: Dict[str, Any], two_operator_mode: bool
    ) -> Dict[str, Dict[str, Any]]:
        """Build the map of enabled augmentation stages."""
        all_actions = {
            const.STAGE_CONFIG_TO_BCQ: {"method": self._config_to_bcq, "output_folder": "bcq"},
            const.STAGE_CONFIG_TO_MCQ: {"method": self._config_to_mcq, "output_folder": "mcq"},
            const.STAGE_GOLDEN_GQA_TO_GQA: {"method": self._golden_gqa_to_gqa, "output_folder": "golden_gqa"},
            const.STAGE_GQA_TO_GQAS: {"method": self._gqa_to_gqas, "output_folder": "gqas"},
            const.STAGE_CONFIG_TO_DMCQ: {"method": self._config_to_dmcq, "output_folder": "dmcq"},
            const.STAGE_CONFIG_TO_DS: {"method": self._config_to_ds, "output_folder": "ds"},
            const.STAGE_CONFIG_TO_EN: {"method": self._config_to_en, "output_folder": "en"},
        }

        all_actions[const.STAGE_SPATIAL_LOCALIZATION] = {
            "method": self._spatial_localization,
            "output_folder": "spatial_localization",
        }

        actions = {}
        for action_name, action_config in all_actions.items():
            if augment_config.get(action_name, {}).get("enable", False):
                actions[action_name] = action_config
                app_logger.info(f"Augmentation stage {action_name} enabled")
        return actions

    async def _register_stages(
        self,
        actions: Dict[str, Any],
        augment_config: Dict[str, Any],
        dataset_id: str,
        frame_drop_iterations: int,
    ) -> None:
        """Insert stage records into the database."""
        for action_name in actions:
            await postgres_db.insert_data(
                schema=AugmentationStage,
                id=f"{dataset_id}_{action_name}",
                augmentation_id=dataset_id,
                stage_name=action_name,
                status=const.PENDING_STATUS,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

        fd_enabled = augment_config.get("frame_drop", {}).get("enable", False)
        if fd_enabled and frame_drop_iterations > 0:
            await postgres_db.insert_data(
                schema=AugmentationStage,
                id=f"{dataset_id}_{const.STAGE_FRAME_DROP}",
                augmentation_id=dataset_id,
                stage_name=const.STAGE_FRAME_DROP,
                status=const.PENDING_STATUS,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

    async def _run_stage(
        self,
        action_name: str,
        action_config: Dict[str, Any],
        label_data_path: str,
        dataset_path: str,
        actions_json: str,
        augment_config: Dict[str, Any],
        dataset_id: str,
    ) -> None:
        """Execute a single augmentation stage with status tracking."""
        await postgres_db.update_data(
            schema=AugmentationStage,
            id=f"{dataset_id}_{action_name}",
            condition={"stage_name": action_name},
            status=const.RUNNING_STATUS,
            updated_at=datetime.now(),
        )

        action_output_path = os.path.join(dataset_path, action_config["output_folder"])

        await action_config["method"](
            label_data_path,
            action_output_path,
            action_config["output_folder"],
            actions_json,
            augment_config,
        )

        self.clean_up(action_output_path)

        await postgres_db.update_data(
            schema=AugmentationStage,
            id=f"{dataset_id}_{action_name}",
            condition={"stage_name": action_name},
            status=const.COMPLETED_STATUS,
            updated_at=datetime.now(),
        )
        app_logger.info(f"Action {action_name} completed successfully")

    async def _fail_all_stages(
        self,
        actions: Dict[str, Any],
        augment_config: Dict[str, Any],
        dataset_id: str,
        dataset_path: str,
        error_msg: str,
        frame_drop_iterations: int,
    ) -> None:
        """Mark all stages as failed, clean up, and raise."""
        all_stage_names = list(actions.keys())
        fd_enabled = augment_config.get("frame_drop", {}).get("enable", False)
        if fd_enabled and frame_drop_iterations > 0:
            all_stage_names.append(const.STAGE_FRAME_DROP)
        for sname in all_stage_names:
            await postgres_db.update_data(
                schema=AugmentationStage,
                id=f"{dataset_id}_{sname}",
                condition={"stage_name": sname},
                status=const.FAILED_STATUS,
                updated_at=datetime.now(),
                error_message=error_msg,
            )

        app_logger.info(f"Cleaning up {dataset_path}")
        shutil.rmtree(dataset_path, ignore_errors=True)

        await postgres_db.update_data(
            id=dataset_id,
            status=const.FAILED_STATUS,
            updated_at=datetime.now(),
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        )

    async def _handle_frame_drop(
        self,
        actions: Dict[str, Any],
        augment_config: Dict[str, Any],
        dataset_path: str,
        dataset_id: str,
        frame_drop_iterations: int,
    ) -> None:
        """Run frame drop post-processing (non-fatal on failure)."""
        fd_cfg = augment_config.get("frame_drop", {})
        if not fd_cfg.get("enable", False) or frame_drop_iterations <= 0:
            return

        app_logger.info(
            f"Running frame drop post-processing ({frame_drop_iterations} iteration(s))"
        )
        try:
            await postgres_db.update_data(
                schema=AugmentationStage,
                id=f"{dataset_id}_{const.STAGE_FRAME_DROP}",
                condition={"stage_name": const.STAGE_FRAME_DROP},
                status=const.RUNNING_STATUS,
                updated_at=datetime.now(),
            )

            completed_datasets = [cfg["output_folder"] for cfg in actions.values()]

            await self._frame_drop(
                dataset_path, augment_config, completed_datasets,
                iterations=frame_drop_iterations,
            )

            await postgres_db.update_data(
                schema=AugmentationStage,
                id=f"{dataset_id}_{const.STAGE_FRAME_DROP}",
                condition={"stage_name": const.STAGE_FRAME_DROP},
                status=const.COMPLETED_STATUS,
                updated_at=datetime.now(),
            )
            app_logger.info("Frame drop post-processing completed")

        except Exception as e:
            error_msg = scrub_secrets(str(e))
            app_logger.error(f"Frame drop failed: {error_msg}")
            app_logger.error(scrub_secrets(traceback.format_exc()))

            await postgres_db.update_data(
                schema=AugmentationStage,
                id=f"{dataset_id}_{const.STAGE_FRAME_DROP}",
                condition={"stage_name": const.STAGE_FRAME_DROP},
                status=const.FAILED_STATUS,
                updated_at=datetime.now(),
                error_message=error_msg,
            )
            app_logger.warning("Frame drop failed but augmentation data is preserved")

    async def process_all_actions(
        self,
        label_data_path: str,
        dataset_path: str,
        actions_json: str,
        augment_config: Dict[str, Any],
        dataset_id: str,
        two_operator_mode: bool = False,
        frame_drop_iterations: int = 1,
    ) -> None:
        """Process augmentation stages and return success status for each.

        When *two_operator_mode* is True, spatial localization (5th stage)
        and frame drop post-processing are added to the pipeline.
        """

        # This coroutine runs as a fire-and-forget asyncio task, so ANY exception
        # that escapes is silently dropped ("Task exception was never retrieved")
        # and leaves the augmentation row stuck at "running" forever. The outer
        # handler below guarantees every failure path — validate, stage build /
        # register, or a stage subprocess — resolves the row to a terminal FAILED
        # state and never lets the exception leak.
        try:
            self._validate_video_folders(label_data_path, augment_config)

            # Inject two_operator_mode into augment_config so downstream stages use it
            augment_config.setdefault("gqas", {})["two_operator_mode"] = str(two_operator_mode).lower()

            # Auto-enable spatial_localization and frame_drop when two-operator mode
            # is on. Single-operator users opt in via augment_config.yaml.
            if two_operator_mode:
                augment_config.setdefault("spatial_localization", {})["enable"] = True
                augment_config.setdefault("frame_drop", {})["enable"] = True

            actions = self._build_stage_config(augment_config, two_operator_mode)
            await self._register_stages(actions, augment_config, dataset_id, frame_drop_iterations)

            # Execute each action
            for action_name, action_config in actions.items():
                app_logger.info(f"Processing action: {action_name}")
                try:
                    await self._run_stage(
                        action_name, action_config, label_data_path,
                        dataset_path, actions_json, augment_config, dataset_id,
                    )
                except Exception as e:
                    error_msg = scrub_secrets(str(e))
                    app_logger.error(f"Error processing action {action_name}: {error_msg}")
                    app_logger.error(scrub_secrets(traceback.format_exc()))
                    # Marks each stage row FAILED for status granularity, then
                    # raises to stop processing further actions; the raise is
                    # caught by the outer handler below.
                    await self._fail_all_stages(
                        actions, augment_config, dataset_id, dataset_path, error_msg,
                        frame_drop_iterations,
                    )

            # Frame drop post-processing (auto-enabled in two-operator mode; opt-in via
            # augment_config.yaml for single-operator)
            await self._handle_frame_drop(
                actions, augment_config, dataset_path,
                dataset_id, frame_drop_iterations,
            )

            # Update augmentation status to completed
            await postgres_db.update_data(
                id=dataset_id,
                status=const.COMPLETED_STATUS,
                updated_at=datetime.now(),
            )
        except Exception as e:
            # Backstop: never let an exception escape the task. The row is forced
            # to FAILED here even for failures that occur before any stage row is
            # registered (e.g. validate). _fail_all_stages may already have set
            # FAILED + cleaned up; repeating it is idempotent.
            error_msg = scrub_secrets(str(e))
            app_logger.error(f"Augmentation {dataset_id} failed: {error_msg}")
            app_logger.error(scrub_secrets(traceback.format_exc()))
            shutil.rmtree(dataset_path, ignore_errors=True)
            await postgres_db.update_data(
                id=dataset_id,
                status=const.FAILED_STATUS,
                updated_at=datetime.now(),
            )


# Create service instance
vlm_service = VLMAugmentationService()


@app.post("/api/v1/augment")
async def augment(
    label_data_id: str,
    two_operator_mode: bool = Query(False, description="Enable spatial localization and frame drop"),
    frame_drop_iterations: int = Query(1, ge=0, le=10, description="Frame drop iterations (0 to skip)"),
) -> AugResponse:
    """VLM data augmentation endpoint.

    When *two_operator_mode* is True the pipeline adds spatial localization
    (5th stage) and frame drop post-processing.
    """

    dataset_id = None
    try:
        augment_config = load_config_yaml(
            os.path.join(const.CONFIG_PATH, const.AUGMENTATION_CONFIG_NAME)
        )
        app_logger.info(f"Augment config: {pformat(augment_config)}")
        app_logger.info(f"two_operator_mode={two_operator_mode}, frame_drop_iterations={frame_drop_iterations}")

        # Generate dataset_id
        augmeted_datasets = await postgres_db.list_data(
            schema=Augmentation, condition={"dataset_id": label_data_id}
        )

        replication_count = 0 + len(augmeted_datasets)
        dataset_id = f"{label_data_id}{const.ID_SUFFIX}_{replication_count}"

        # Setup paths — reject traversal in caller-supplied IDs before any FS op.
        try:
            label_data_path = safe_dataset_path(const.DATASET_ROOT, label_data_id)
            dataset_path = safe_dataset_path(const.DATASET_ROOT, dataset_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        actions_json = os.path.join(label_data_path, const.SOP_ACTIONS_JSON_NAME)

        # Verify input paths exist
        if not os.path.exists(label_data_path):
            raise HTTPException(
                status_code=400, detail=f"Label data path not found: {label_data_path}"
            )

        if not os.path.exists(actions_json):
            raise HTTPException(
                status_code=400,
                detail=f"{const.SOP_ACTIONS_JSON_NAME} not found: {actions_json}",
            )

        # Pre-flight: validate video folders synchronously so input errors are
        # reported as a 4xx on THIS request, before any DB row or output dir is
        # created and before the async task is spawned.
        # Without this, validation only ran inside the fire-and-forget task,
        # where a raised exception left the status stuck at "running" forever.
        vlm_service._validate_video_folders(label_data_path, augment_config)

        # Clean and create dataset directory
        clean_and_create_dir(dataset_path)

        app_logger.info(f"Processing Label data: {label_data_id}")
        app_logger.info(f"Label data path: {label_data_path}")
        app_logger.info(f"Output dataset: {dataset_id}")
        app_logger.info(f"Output dataset path: {dataset_path}")

        # Insert into database
        await postgres_db.insert_data(
            id=dataset_id,
            dataset_id=label_data_id,
            parameters=augment_config,
            status=const.PENDING_STATUS,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Mark RUNNING BEFORE spawning the background task. Once the task is
        # spawned it becomes the sole writer of this row and only ever writes a
        # terminal state (COMPLETED/FAILED), so the status can never reverse. If
        # this update ran after create_task, a fast-failing task could write
        # FAILED first and then be overwritten back to RUNNING.
        await postgres_db.update_data(
            id=dataset_id,
            status=const.RUNNING_STATUS,
            updated_at=datetime.now(),
        )

        # Process all actions asynchronously
        asyncio.create_task(
            vlm_service.process_all_actions(
                label_data_path,
                dataset_path,
                actions_json,
                augment_config,
                dataset_id,
                two_operator_mode=two_operator_mode,
                frame_drop_iterations=frame_drop_iterations,
            )
        )

        app_logger.info(f"Augmentation submitted. Dataset: {dataset_id}")

        return AugResponse(dataset_id=dataset_id)
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error: {scrub_secrets(str(e))}")
        app_logger.error(scrub_secrets(traceback.format_exc()))

        if dataset_id:
            # Update database
            await postgres_db.update_data(
                id=dataset_id,
                status=const.FAILED_STATUS,
                updated_at=datetime.now(),
            )

        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        )


@app.get("/api/v1/augmented_datasets")
async def get_all_augmented_datasets() -> Dict:
    """Get augmented datasets"""
    # response: {augmented_data_id: {"status": "completed", "video_count": 10, "total_clips": 100}}
    all_datasets_info = {}
    augmented_datasets = await postgres_db.list_data(
        schema=Augmentation, condition={"status": const.COMPLETED_STATUS}
    )

    try:
        for dataset in augmented_datasets:
            videos = await postgres_db.list_data(
                schema=Video, condition={"dataset_id": dataset.dataset_id}
            )
            all_chunks = [
                await postgres_db.list_data(schema=Chunk, condition={"video_id": video.id})
                for video in videos
            ]
            total_clips = sum([len(chunks) for chunks in all_chunks])
            all_datasets_info[dataset.id] = {
                "status": dataset.status,
                "video_count": len(videos),
                "total_clips": total_clips,
            }
    except Exception as e:
        app_logger.error(f"Error: {scrub_secrets(str(e))}")
        app_logger.error(scrub_secrets(traceback.format_exc()))
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        )

    return all_datasets_info


@app.get("/api/v1/augmentation_status/{dataset_id}")
async def get_augmentation_status(dataset_id: str) -> AugmentationStatusResponse:
    """Get detailed status of augmentation stages for a specific dataset"""
    try:
        # Get the main augmentation record
        augmentation = await postgres_db.get_data(schema=Augmentation, id=dataset_id)

        if not augmentation:
            raise HTTPException(
                status_code=404, detail=f"Augmentation with ID {dataset_id} not found"
            )
        # Get all stages for this augmentation
        stages = await postgres_db.list_data(
            schema=AugmentationStage, condition={"augmentation_id": dataset_id}
        )
        # Convert to response format
        completed_stages = 0
        total_stages = len(stages)

        for stage in stages:
            if stage.status in [const.COMPLETED_STATUS, const.FAILED_STATUS]:
                completed_stages += 1

        # Calculate progress percentage
        progress_percentage = (
            (completed_stages / total_stages * 100) if total_stages > 0 else 100
        )

        return AugmentationStatusResponse(
            dataset_id=dataset_id,
            status=augmentation.status,
            progress=round(progress_percentage, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error getting augmentation status: {scrub_secrets(str(e))}")
        app_logger.error(scrub_secrets(traceback.format_exc()))
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        )


@app.get("/health", tags=["status"])
async def health():
    """Health check endpoint"""
    return {"message": "VLM Data Augmentation API is running", "status": "healthy"}
