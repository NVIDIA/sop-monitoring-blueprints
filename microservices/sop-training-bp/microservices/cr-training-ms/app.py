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
import logging
import os
import traceback
import uuid
from datetime import datetime

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import utils.constant as const
from components.cache import TrainingJobCache
from components.postgres_db import postgres_db
from utils.dataset_utils import get_all_json_paths
from utils.utils import create_file, dump_toml, parse_cr_log, read_toml, safe_dataset_path, terminate_process_tree
from validation.request_validation import FineTuningResponse, TrainingStatus


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cosmos-Reason Fine-tuning Microservice",
    description="A microservice for fine-tuning Cosmos-Reason models",
    openapi_tags=[
        {"name": "fine-tuning", "description": "Operations for fine-tuning Cosmos-Reason models"},
        {"name": "status", "description": "Operations for checking training status and results"},
    ],
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

# Global storage for training jobs cache
# This is to reduce the frequency of querying the database for training jobs
# The training job would be deleted from the cache after it's completed, cancelled, or failed
# TODO: Use Redis to replace this in-memory cache
training_jobs_cache = TrainingJobCache()


@app.get("/health", tags=["status"])
async def root():
    """Health check endpoint"""
    return {"message": "Cosmos-Reason Fine-tuning Microservice is running"}


@app.post("/api/v1/fine-tuning/start", response_model=FineTuningResponse, tags=["fine-tuning"])
async def start_fine_tuning(dataset_id: str, background_tasks: BackgroundTasks):
    """
    Start a new fine-tuning job

    This endpoint initiates a new Cosmos-Reason fine-tuning job with the provided configuration.
    The training will run in the background and can be monitored via the status endpoint.
    """
    try:
        # check if there's a training job running
        for job in training_jobs_cache.cache.values():
            if job["status"] == const.RUNNING_STATUS:
                raise HTTPException(
                    status_code=400,
                    detail=f"A job is already running: {job['job_id']}. Please wait for it to finish or cancel it.",
                )

        job_id = str(uuid.uuid4())
        try:
            dataset_path = safe_dataset_path(const.DATASET_ROOT, dataset_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        # create log file if it doesn't exist
        create_file(os.path.join(const.RESULTS_ROOT, job_id, "log.txt"))

        # get custom dataset and train config
        custom_dataset_path = os.path.join(const.TOOL_PATH, const.CUSTOM_DATASET_NAME)
        train_config_path = os.path.join(const.CONFIG_PATH, const.TRAIN_CONFIG_NAME)

        # Create job cache
        training_jobs_cache.set(
            job_id,
            {
                "job_id": job_id,
                "status": const.QUEUE_STATUS,
                "progress": 0.0,
                "current_step": 0,
                "total_steps": 0,
                "loss": None,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "log_file_path": os.path.join(const.RESULTS_ROOT, job_id, "log.txt"),
                "process_pid": None,  # Will be set when process starts
            },
        )

        # insert job record into database
        await postgres_db.insert_training_job(
            id=job_id,
            aug_dataset_id=dataset_id,
            **training_jobs_cache.get(job_id),
        )

        # Start training in background (includes model download if needed)
        background_tasks.add_task(
            run_fine_tuning,
            job_id,
            dataset_path,
            train_config_path,
            custom_dataset_path,
        )

        logger.info(f"Started fine-tuning job {job_id}")

        return FineTuningResponse(
            job_id=job_id,
            status=training_jobs_cache.get(job_id)["status"],
            message="Fine-tuning job has been queued and will start shortly",
            created_at=training_jobs_cache.get(job_id)["created_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting fine-tuning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start fine-tuning: {str(e)}")


@app.get("/api/v1/fine-tuning/status/{job_id}", response_model=TrainingStatus, tags=["status"])
async def get_training_status(job_id: str):
    """
    Get the status of a fine-tuning job

    Returns detailed information about the training progress including current step,
    loss values, and logs.
    """
    try:
        # Try to get job from cache first
        job = training_jobs_cache.get(job_id)

        # if job is not in cache, try to get from database
        if job is None:
            job = await postgres_db.get_training_job(job_id)
            job = job.to_dict() if job else None

        if job is None:
            raise HTTPException(status_code=404, detail="Training job not found")

        return TrainingStatus(
            job_id=job_id,
            status=job["status"],
            progress=job["progress"],
            current_step=job["current_step"],
            total_steps=job["total_steps"],
            loss=job["loss"],
            created_at=job["created_at"],
            updated_at=job["updated_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")


@app.get("/api/v1/fine-tuning/all_jobs", tags=["status"])
async def get_all_jobs():
    """
    Get all fine-tuning jobs
    """
    # format: {job_id: {...}}
    all_jobs = await postgres_db.list_training_jobs()
    return {job.id: job.to_dict() for job in all_jobs}


@app.post("/api/v1/fine-tuning/cancel/{job_id}", tags=["fine-tuning"])
async def cancel_fine_tuning(job_id: str):
    """
    Cancel a fine-tuning job

    This endpoint cancels a fine-tuning job if it is still in the running state.
    It will terminate the entire process tree including torchrun and all worker processes.
    """
    if training_jobs_cache.get(job_id) is None:
        raise HTTPException(status_code=404, detail="The job is not in running state or not found")

    # Check if we have a process PID to terminate
    process_pid = training_jobs_cache.get(job_id).get("process_pid")
    if not process_pid:
        # Currently, we don't support cancelling queued jobs
        # Because the FastAPI BackgroundTasks would pass the job right away
        # TODO: Use message queue mechanism to implement the real queueing
        return {
            "message": f"Training job {job_id} has not been assigned to a worker yet. Please wait a second and try again."
        }

    # Try to terminate the process tree
    logger.info(f"Attempting to cancel running job {job_id} with PID {process_pid}")

    success = terminate_process_tree(process_pid)

    if success:
        training_jobs_cache.update(
            job_id,
            status=const.CANCELLED_STATUS,
            updated_at=datetime.now(),
        )

        # Log the cancellation
        log_file_path = training_jobs_cache.get(job_id).get("log_file_path")
        if log_file_path and os.path.exists(log_file_path):
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"[CANCELLED] Training job {job_id} was cancelled by user\n")

        logger.info(f"Successfully cancelled training job {job_id}")
        return {"message": f"Training job {job_id} has been cancelled successfully"}
    else:
        logger.error(f"Failed to cancel training job {job_id}")
        return {"message": f"Failed to cancel training job {job_id}. Process may still be running."}


async def run_fine_tuning(job_id: str, dataset_path: str, train_config_path: str, custom_dataset_path: str):
    """Background task to run fine-tuning"""
    try:
        # override train config
        json_paths = get_all_json_paths(dataset_path)
        split_names = [file.split("/")[-1].split(".")[0] for file in json_paths]
        train_config = read_toml(train_config_path)
        train_config["train"]["train_policy"]["dataset"]["name"] = str(json_paths)
        train_config["train"]["train_policy"]["dataset"]["split"] = split_names
        train_config["train"]["output_dir"] = os.path.join(const.RESULTS_ROOT, job_id)
        train_config["logging"]["experiment_name"] = job_id
        dump_toml(train_config, os.path.join(const.RESULTS_ROOT, job_id, f"{job_id}.toml"))

        # Update job status
        training_jobs_cache.update(
            job_id,
            status=const.RUNNING_STATUS,
            updated_at=datetime.now(),
        )

        # Update database
        await postgres_db.update_training_job(
            job_id,
            status=training_jobs_cache.get(job_id)["status"],
            updated_at=training_jobs_cache.get(job_id)["updated_at"],
        )

        # Validate model path
        if not os.path.exists(train_config["policy"]["model_name_or_path"]):
            raise FileNotFoundError(f"Model path does not exist: {train_config['policy']['model_name_or_path']}")

        cmd = [
            "cosmos-rl",
            "--config",
            os.path.join(const.RESULTS_ROOT, job_id, f"{job_id}.toml"),
            f"{custom_dataset_path}",
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        # Use asyncio.create_subprocess_exec for non-blocking execution
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        # Store the process PID for cancellation
        training_jobs_cache.update(
            job_id,
            process_pid=process.pid,
        )
        logger.info(f"Started Cosmos-Reason process with PID {process.pid} for job {job_id}")

        # Path to status.json and log file
        log_file_path = training_jobs_cache.get(job_id)["log_file_path"]

        async def read_stream(stream):
            """Reads and logs a stream line by line without blocking."""

            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"Starting training job {job_id}\n")
                while True:
                    line_bytes = await stream.readline()
                    if not line_bytes:
                        break
                    line = line_bytes.decode("utf-8").strip()
                    if line:
                        log_file.write(f"{line}\n")
                        log_file.flush()

                        # parse the log string to extract training progress and information
                        progress_info = parse_cr_log(line)

                        if progress_info:
                            training_jobs_cache.update(
                                job_id,
                                current_step=progress_info["current_step"],
                                total_steps=progress_info["total_steps"],
                                loss=progress_info["loss"],
                                progress=round(
                                    (progress_info["current_step"] / progress_info["total_steps"]) * 100, 2
                                ),
                                updated_at=datetime.now(),
                            )

                        logger.info(f"Job {job_id}: {line}")

        # Run stream readers concurrently
        await asyncio.gather(read_stream(process.stdout), read_stream(process.stderr))

        # Wait for the process to complete
        return_code = await process.wait()

        # Final status update
        if return_code == 0:
            training_jobs_cache.update(
                job_id,
                status=const.COMPLETED_STATUS,
                updated_at=datetime.now(),
                progress=100.0,
            )
        elif training_jobs_cache.get(job_id)["status"] != const.CANCELLED_STATUS:
            training_jobs_cache.update(
                job_id,
                status=const.FAILED_STATUS,
                updated_at=datetime.now(),
            )

        # update database
        await postgres_db.update_training_job(
            **training_jobs_cache.get(job_id),
        )

        logger.info(f"Training job {job_id} finished with status: {training_jobs_cache.get(job_id)['status']}")

        # delete job cache
        training_jobs_cache.delete(job_id)
    except Exception as e:
        training_jobs_cache.update(
            job_id,
            status=const.FAILED_STATUS,
            updated_at=datetime.now(),
        )

        # Also write this critical error to the log file if possible
        with open(training_jobs_cache.get(job_id)["log_file_path"], "a", encoding="utf-8") as log_file:
            log_file.write(f"[ERROR] A critical error occurred in the training runner: {str(e)}\n")

        # update database
        await postgres_db.update_training_job(
            **training_jobs_cache.get(job_id),
        )

        # delete job cache
        training_jobs_cache.delete(job_id)

        logger.error(f"Error in training job {job_id}: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to start fine-tuning")
