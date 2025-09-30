######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
######################################################################################################


import asyncio
import os
import shutil
import subprocess
import traceback
from datetime import datetime
from pprint import pformat
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import utils.constant as const
from components.postgres_db import postgres_db
from utils.logger import app_logger
from utils.utils import clean_and_create_dir, load_config_yaml
from validation.postgres_validation import Augmentation, AugmentationStage, Chunk, Video
from validation.response_validation import (
    AugmentationStatusResponse,
    AugResponse,
)

app = FastAPI(
    title="VLM Data Augmentation API",
    description="FastAPI service for VLM data augmentation - processes all four actions automatically",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VLMAugmentationService:
    def __init__(self):
        pass

    async def _run_cmd(self, cmd):
        """Run asynchronous subprocess with basic environment setup"""
        app_logger.info(f"Command: {' '.join(cmd)}")

        # Create async subprocess
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        async def read_stream(stream, prefix):
            """Read and log stream output in real-time"""
            while True:
                line_bytes = await stream.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8").strip()
                if line:
                    app_logger.info(f"{prefix}: {line}")

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

        # Check request first, then environment variables as fallback
        ngc_personal_key = augment_config["gqas"].get(
            "ngc_personal_key", os.getenv("NGC_PERSONAL_KEY", "")
        )
        llm_type = augment_config["gqas"].get("llm_type", "nvidia")
        local_llm_url = augment_config["gqas"].get("local_llm_url", "")
        llm = augment_config["gqas"].get("llm", const.DEFAULT_LLM)
        num_qa_llm = augment_config["gqas"].get("num_qa_llm", "8")
        num_qa_per_chunk = augment_config["gqas"].get("num_qa_per_chunk", "2")
        ext = augment_config.get("video_extention", const.DEFAULT_VIDEO_EXTENSION)
        exclude_action = augment_config["gqas"].get("exclude_action", "")

        cmd = [
            "python",
            "-m",
            "vlm_aug.gqa_to_gqas",
            "--llm-type",
            llm_type,
            "--llm",
            llm,
            "--api-key",
            ngc_personal_key,
            "--local-llm-url",
            local_llm_url,
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
            "--output-root",
            output_root,
            "--output-name",
            output_name,
        ]

        # run command
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

    async def process_all_actions(
        self,
        label_data_path: str,
        dataset_path: str,
        actions_json: str,
        augment_config: Dict[str, Any],
        dataset_id: str,
    ) -> None:
        """Process all four actions and return success status for each"""

        # Find all video folders in the label_data_id directory
        # Make sure there are videos in label_data_path
        video_folders = self.find_video_folders(
            label_data_path,
            augment_config.get("video_extention", const.DEFAULT_VIDEO_EXTENSION),
        )
        if not video_folders:
            raise HTTPException(
                status_code=400, detail=f"No video folders found in {label_data_path}"
            )

        # Define action configurations
        # TODO: This should come from external config defined by user which augmentation they want to run
        actions = {
            const.STAGE_CONFIG_TO_BCQ: {
                "method": self._config_to_bcq,
                "output_folder": "bcq",
            },
            const.STAGE_CONFIG_TO_MCQ: {
                "method": self._config_to_mcq,
                "output_folder": "mcq",
            },
            const.STAGE_GOLDEN_GQA_TO_GQA: {
                "method": self._golden_gqa_to_gqa,
                "output_folder": "golden_gqa",
            },
            const.STAGE_GQA_TO_GQAS: {
                "method": self._gqa_to_gqas,
                "output_folder": "gqas",
            },
        }

        # Insert action configurations to database
        for action_name, action_config in actions.items():
            await postgres_db.insert_data(
                schema=AugmentationStage,
                id=f"{dataset_id}_{action_name}",
                augmentation_id=dataset_id,
                stage_name=action_name,
                status=const.PENDING_STATUS,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

        # Execute each action
        for action_name, action_config in actions.items():
            app_logger.info(f"Processing action: {action_name}")

            try:
                # Update stage status to running
                await postgres_db.update_data(
                    schema=AugmentationStage,
                    id=f"{dataset_id}_{action_name}",
                    condition={"stage_name": action_name},
                    status=const.RUNNING_STATUS,
                    updated_at=datetime.now(),
                )

                # Create output folder for this action
                action_output_path = os.path.join(
                    dataset_path, action_config["output_folder"]
                )

                await action_config["method"](
                    label_data_path,
                    action_output_path,
                    action_config["output_folder"],
                    actions_json,
                    augment_config,
                )

                # Consolidate videos for this action
                self.clean_up(action_output_path)

                # Update stage status to completed
                await postgres_db.update_data(
                    schema=AugmentationStage,
                    id=f"{dataset_id}_{action_name}",
                    condition={"stage_name": action_name},
                    status=const.COMPLETED_STATUS,
                    updated_at=datetime.now(),
                )

                app_logger.info(f"Action {action_name} completed successfully")

            except Exception as e:
                # One fail, all fail
                error_msg = str(e)
                app_logger.error(f"Error processing action {action_name}: {error_msg}")
                app_logger.error(traceback.format_exc())

                # Update stage status to failed
                for action_name, _ in actions.items():
                    await postgres_db.update_data(
                        schema=AugmentationStage,
                        id=f"{dataset_id}_{action_name}",
                        condition={"stage_name": action_name},
                        status=const.FAILED_STATUS,
                        updated_at=datetime.now(),
                        error_message=error_msg,
                    )

                # clean up all generated files
                app_logger.info(f"Cleaning up {dataset_path}")
                shutil.rmtree(dataset_path, ignore_errors=True)

                # Update augmentation status to failed
                await postgres_db.update_data(
                    id=dataset_id,
                    status=const.FAILED_STATUS,
                    updated_at=datetime.now(),
                )

                raise HTTPException(
                    status_code=500,
                    detail=f"Internal server error: {traceback.format_exc()}",
                )

        # Update augmentation status to completed
        await postgres_db.update_data(
            id=dataset_id,
            status=const.COMPLETED_STATUS,
            updated_at=datetime.now(),
        )


# Create service instance
vlm_service = VLMAugmentationService()


@app.post("/api/v1/augment")
async def augment(label_data_id: str) -> AugResponse:
    """VLM data augmentation endpoint - processes all four actions automatically"""

    try:
        augment_config = load_config_yaml(
            os.path.join(const.CONFIG_PATH, const.AUGMENTATION_CONFIG_NAME)
        )
        app_logger.info(f"Augment config: {pformat(augment_config)}")

        # Generate dataset_id
        augmeted_datasets = await postgres_db.list_data(
            schema=Augmentation, condition={"dataset_id": label_data_id}
        )

        replication_count = 0 + len(augmeted_datasets)
        dataset_id = f"{label_data_id}{const.ID_SUFFIX}_{replication_count}"

        # Setup paths
        label_data_path = os.path.join(const.DATASET_ROOT, label_data_id)
        dataset_path = os.path.join(const.DATASET_ROOT, dataset_id)
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

        # Process all actions asynchronously
        asyncio.create_task(
            vlm_service.process_all_actions(
                label_data_path,
                dataset_path,
                actions_json,
                augment_config,
                dataset_id,
            )
        )

        app_logger.info(f"Augmentation submitted. Dataset: {dataset_id}")

        # Update database to running
        await postgres_db.update_data(
            id=dataset_id,
            status=const.RUNNING_STATUS,
            updated_at=datetime.now(),
        )

        return AugResponse(dataset_id=dataset_id)
    except Exception as e:
        app_logger.error(f"Error: {str(e)}")
        app_logger.error(traceback.format_exc())

        # Update database
        await postgres_db.update_data(
            id=dataset_id,
            status=const.FAILED_STATUS,
            updated_at=datetime.now(),
        )

        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(traceback.format_exc())}",
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
        app_logger.error(f"Error: {str(e)}")
        app_logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(traceback.format_exc())}",
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

    except Exception as e:
        app_logger.error(f"Error getting augmentation status: {str(e)}")
        app_logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(traceback.format_exc())}",
        )


@app.get("/health", tags=["status"])
async def health():
    """Health check endpoint"""
    return {"message": "VLM Data Augmentation API is running", "status": "healthy"}
