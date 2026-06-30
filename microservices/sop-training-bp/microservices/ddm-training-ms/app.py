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

import asyncio
import json
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
from utils.dataset_utils import generate_ddm_annotation, get_all_json_paths
from utils.utils import create_file, dump_toml, parse_ddm_log, read_toml, safe_dataset_path, terminate_process_tree, dump_yaml, read_yaml
from validation.request_validation import FineTuningResponse, TrainingStatus


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DDM-Net Fine-tuning Microservice",
    description="A microservice for fine-tuning DDM-Net models",
    openapi_tags=[
        {"name": "fine-tuning", "description": "Operations for fine-tuning DDM-Net models"},
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


async def cancel_orphaned_jobs_on_startup():
    """
    Marks all running/queued jobs as cancelled on service startup.
    This handles cases where the service was restarted while jobs were running.
    Also terminates any real running processes if their PIDs are still valid.
    """
    logger.info("=" * 80)
    logger.info("DDM-Net TRAINING MICROSERVICE STARTUP - Cleaning up orphaned jobs")
    logger.info("=" * 80)
    logger.info("Starting up DDM-Net training microservice...")
    
    try:
        # Get all jobs from database
        all_jobs = await postgres_db.list_training_jobs()
        logger.info(f"Found {len(all_jobs)} total jobs in database")
        
        cancelled_count = 0
        terminated_count = 0
        for job in all_jobs:
            # If job is in running or queued state, mark it as cancelled
            if job.status in [const.RUNNING_STATUS, const.QUEUE_STATUS]:
                logger.warning(
                    f"Found orphaned job {job.id} in {job.status} state. "
                    "Marking as cancelled due to service restart."
                )

                # Try to terminate the real process if PID is available
                process_pid = getattr(job, 'process_pid', None)
                if process_pid:
                    logger.info(f"Attempting to terminate orphaned process PID {process_pid} for job {job.id}")
                    success = terminate_process_tree(process_pid)
                    if success:
                        logger.info(f"Successfully terminated orphaned process PID {process_pid}")
                        terminated_count += 1
                    else:
                        logger.warning(
                            f"Could not terminate process PID {process_pid} - "
                            "it may have already exited or the PID was reused by another process"
                        )
                else:
                    logger.info(f"No PID stored for orphaned job {job.id}, cannot terminate process")

                logger.info(f"Cancelling orphaned job: {job.id} (status: {job.status})")
                await postgres_db.update_training_job(
                    job.id,
                    status=const.CANCELLED_STATUS,
                    process_pid=None,  # Clear the PID
                    updated_at=datetime.now(),
                )
                cancelled_count += 1
        
        if cancelled_count > 0:
            logger.info(f"Marked {cancelled_count} orphaned jobs as cancelled")
            logger.info(f"Terminated {terminated_count} orphaned processes")
        else:
            logger.info("No orphaned jobs found")
        
        logger.info("=" * 80)
            
    except Exception as e:
        logger.error(f"Error during startup job cleanup: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


@app.on_event("startup")
async def startup_event():
    """Run startup tasks"""
    await cancel_orphaned_jobs_on_startup()


@app.get("/health", tags=["status"])
async def root():
    """Health check endpoint"""
    return {"message": "DDM-Net Fine-tuning Microservice is running"}


@app.post("/api/v1/fine-tuning/start", response_model=FineTuningResponse, tags=["fine-tuning"])
async def start_fine_tuning(dataset_id: str, validation_dataset_id: str = None, background_tasks: BackgroundTasks = None):
    """
    Start a new fine-tuning job

    This endpoint initiates a new DDM-Net fine-tuning job with the provided configuration.
    The training will run in the background and can be monitored via the status endpoint.
    
    Args:
        dataset_id: Training dataset ID
        validation_dataset_id: Validation dataset ID (defaults to same as training dataset if not provided)
        background_tasks: FastAPI background tasks
    """
    try:
        # Default validation dataset to training dataset if not provided
        if validation_dataset_id is None or validation_dataset_id == "":
            validation_dataset_id = dataset_id
            logger.info(f"Validation dataset not specified, using training dataset: {dataset_id}")
        
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
            validation_dataset_path = safe_dataset_path(const.DATASET_ROOT, validation_dataset_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        # Generate training annotation for DDM-Net fine-tuning
        try:
            annotation_file_path = generate_ddm_annotation(dataset_path, const.DDM_TRAIN_ANNOTATION_NAME)
            logger.info(f"Generated DDM-Net training annotation: {annotation_file_path}")
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to generate DDM-Net training annotation: {str(e)}"
            )
        
        # Try to generate validation annotation for DDM-Net fine-tuning
        # If it fails, run_fine_tuning will handle the fallback
        try:
            val_annotation_file_path = generate_ddm_annotation(validation_dataset_path, const.DDM_VAL_ANNOTATION_NAME)
            logger.info(f"Generated DDM-Net validation annotation: {val_annotation_file_path}")
        except (FileNotFoundError, ValueError) as e:
            logger.warning(
                f"Failed to generate DDM-Net validation annotation: {str(e)}. "
                f"Will attempt to use training annotation for validation."
            )

        # create log file if it doesn't exist
        create_file(os.path.join(const.RESULTS_ROOT, job_id, "log.txt"))

        # get custom dataset and train config
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
            validation_dataset_path,
            train_config_path,
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
    It will terminate the entire process tree including all worker processes.
    """
    # First check cache
    job_in_cache = training_jobs_cache.get(job_id)
    
    # If not in cache, check database
    if job_in_cache is None:
        logger.info(f"Job {job_id} not found in cache, checking database...")
        job_in_db = await postgres_db.get_training_job(job_id)
        
        if job_in_db is None:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        # If job is already in a terminal state (completed, cancelled, failed), just return
        if job_in_db.status in [const.COMPLETED_STATUS, const.CANCELLED_STATUS, const.FAILED_STATUS]:
            return {
                "message": f"Training job {job_id} is already in {job_in_db.status} state. No cancellation needed."
            }
        
        # If job is in running/queued state but not in cache (likely after service restart),
        # we can't cancel it because we don't have the process PID
        # Just mark it as cancelled in the database
        logger.warning(f"Job {job_id} is in {job_in_db.status} state but not in cache. Marking as cancelled in database.")
        await postgres_db.update_training_job(
            job_id,
            status=const.CANCELLED_STATUS,
            updated_at=datetime.now(),
        )
        return {
            "message": f"Training job {job_id} has been marked as cancelled. Note: Process may still be running if service was restarted."
        }

    # Check if we have a process PID to terminate
    process_pid = job_in_cache.get("process_pid")
    if not process_pid:
        # Job is queued but hasn't started yet
        # Just mark it as cancelled
        logger.info(f"Job {job_id} is queued but hasn't started. Marking as cancelled.")
        training_jobs_cache.update(
            job_id,
            status=const.CANCELLED_STATUS,
            updated_at=datetime.now(),
        )
        await postgres_db.update_training_job(
            job_id,
            status=const.CANCELLED_STATUS,
            updated_at=datetime.now(),
        )
        return {
            "message": f"Training job {job_id} was queued and has been cancelled successfully."
        }

    # Try to terminate the process tree
    logger.info(f"Attempting to cancel running job {job_id} with PID {process_pid}")

    success = terminate_process_tree(process_pid)

    if success:
        training_jobs_cache.update(
            job_id,
            status=const.CANCELLED_STATUS,
            process_pid=None,
            updated_at=datetime.now(),
        )
        
        # Update database (clear PID since process is terminated)
        await postgres_db.update_training_job(
            job_id,
            status=const.CANCELLED_STATUS,
            process_pid=None,
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
        raise HTTPException(status_code=500, detail=f"Failed to cancel training job {job_id}. Process may still be running.")


# This is the most critical function to modify for DDM
# You need to:
# 1. Update the training configuration preparation logic
# 2. Replace the cosmos-rl command with DDM's training command
# 3. Update the log parsing to match DDM's output format
# 4. Ensure the process management works with DDM's training script
async def run_fine_tuning(job_id: str, dataset_path: str, validation_dataset_path: str, train_config_path: str):
    """Background task to run fine-tuning
    
    Args:
        job_id: Unique identifier for this training job
        dataset_path: Path to the training dataset
        validation_dataset_path: Path to the validation dataset
        train_config_path: Path to the training configuration file
    """
    try:
        # Load and prepare training configuration for DDM
        if train_config_path.endswith(".toml"):
            train_config = read_toml(train_config_path)
        elif train_config_path.endswith(".yaml"):
            train_config = read_yaml(train_config_path)
        else:
            raise ValueError(f"Unsupported training configuration file format: {train_config_path}")
        logger.info(f"Training configuration loaded from: {train_config_path}")

        # Set training annotation path
        train_annotation_path = os.path.join(dataset_path, const.DDM_TRAIN_ANNOTATION_NAME)
        train_config["dataset_config"]["train_config"]["anno_path"] = train_annotation_path
        train_config["dataset_config"]["train_config"]["data_root"] = dataset_path
        
        # Try to use validation annotation, fallback to training annotation if validation fails
        val_annotation_path = os.path.join(validation_dataset_path, const.DDM_VAL_ANNOTATION_NAME)
        use_training_for_validation = False
        
        # Check if validation annotation file exists and is valid
        if os.path.exists(val_annotation_path):
            try:
                # Try to read and validate the JSON file
                with open(val_annotation_path, 'r', encoding='utf-8') as f:
                    val_data = json.load(f)
                
                # Basic validation: check if it's a dict and not empty
                if not isinstance(val_data, dict) or len(val_data) == 0:
                    raise ValueError("Validation annotation is empty or invalid format")
                
                # Validation annotation is good, use it
                train_config["dataset_config"]["val_config"]["anno_path"] = val_annotation_path
                train_config["dataset_config"]["val_config"]["data_root"] = validation_dataset_path
                logger.info(f"Using validation annotation: {val_annotation_path}")
                
            except (ValueError, IOError) as e:
                # Validation annotation exists but is invalid, fallback to training
                logger.warning(
                    f"Validation annotation file exists but is invalid: {str(e)}. "
                    f"Falling back to training annotation."
                )
                use_training_for_validation = True
        else:
            # Validation annotation doesn't exist, fallback to training
            logger.warning(
                f"Validation annotation not found at {val_annotation_path}. "
                f"Falling back to training annotation."
            )
            use_training_for_validation = True
        
        # Apply fallback if needed
        if use_training_for_validation:
            train_config["dataset_config"]["val_config"]["anno_path"] = train_annotation_path
            train_config["dataset_config"]["val_config"]["data_root"] = dataset_path
            logger.info(f"Using training annotation for validation: {train_annotation_path}")
        
        # Update training output paths and experiment name
        train_config["training_config"]["output"] = os.path.join(const.RESULTS_ROOT, job_id)
        train_config["training_config"]["exp_name"] = job_id
        
        # Save the updated config for this training job
        if train_config_path.endswith(".toml"):
            job_config_path = os.path.join(const.RESULTS_ROOT, job_id, f"{job_id}.toml")
            dump_toml(train_config, job_config_path)
        elif train_config_path.endswith(".yaml"):
            job_config_path = os.path.join(const.RESULTS_ROOT, job_id, f"{job_id}.yaml")
            dump_yaml(train_config, job_config_path)
        else:
            raise ValueError(f"Unsupported training configuration file format: {train_config_path}")

         # Get total epochs from config for progress calculation
        total_epochs = train_config.get("training_config", {}).get("epochs", 30)

        # Update job status
        training_jobs_cache.update(
            job_id,
            status=const.RUNNING_STATUS,
            total_epochs=total_epochs,
            updated_at=datetime.now(),
        )

        # Update database
        await postgres_db.update_training_job(
            job_id,
            status=training_jobs_cache.get(job_id)["status"],
            updated_at=training_jobs_cache.get(job_id)["updated_at"],
        )

        # Validate pretrained model path if specified
        pretrained_path = train_config.get("model_config", {}).get("pretrained")
        if isinstance(pretrained_path, bool):
            logger.info(f"Pretrained set to {pretrained_path}. Training script will handle initialization.")
        elif pretrained_path and isinstance(pretrained_path, str) and pretrained_path != "":
            # Check if it's a relative or absolute path
            if not os.path.isabs(pretrained_path):
                # If relative, check from multiple possible locations
                possible_paths = [
                    pretrained_path,  # Current directory
                    os.path.join(const.PRETRAINED_MODEL_ROOT, pretrained_path),
                    os.path.join("/workspace/sop-ddm-ftms", pretrained_path),
                ]
                pretrained_exists = any(os.path.exists(p) for p in possible_paths)
                if not pretrained_exists:
                    logger.warning(
                        f"Pretrained model not found at any of: {possible_paths}. "
                        "Training will start from scratch or use default initialization."
                    )
            elif not os.path.exists(pretrained_path):
                logger.warning(
                    f"Pretrained model not found at: {pretrained_path}. "
                    "Training will start from scratch or use default initialization."
                )

        # Construct training command with the generated config file
        # Note: The training script must be executed from the ddm/ directory
        # because it uses relative imports (e.g., from utils.getter import getModel)
        # Use bash -c to cd into ddm/ and execute the training script
        cmd = [
            "/bin/bash",
            "-c",
            f"cd /workspace/sop-ddm-ftms/ddm && python DDM-Net/train_sop_lightning.py --config {job_config_path}"
        ]
        
        logger.info(f"Running command: {cmd[2]}")
        logger.info(f"Training config loaded from: {job_config_path}")

        # Use asyncio.create_subprocess_exec for non-blocking execution
        process = await asyncio.create_subprocess_exec(
            *cmd, 
            stdout=asyncio.subprocess.PIPE, 
            stderr=asyncio.subprocess.PIPE
        )

        # Store the process PID for cancellation (in both cache and database)
        training_jobs_cache.update(
            job_id,
            process_pid=process.pid,
        )
        # Persist PID to database so it survives service restarts
        await postgres_db.update_training_job(
            job_id,
            process_pid=process.pid,
        )
        logger.info(f"Started DDM-Net process with PID {process.pid} for job {job_id}")

        # Path to status.json and log file
        log_file_path = training_jobs_cache.get(job_id)["log_file_path"]

        async def read_stream(stream):
            """Reads and logs a stream line by line without blocking.

            tqdm progress bars use \\r as a line terminator, so the runner can
            accumulate >64 KB before a \\n arrives and asyncio's StreamReader
            raises LimitOverrunError. Drain on overflow and continue rather
            than letting the exception kill the runner while training is fine.
            """

            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"Starting training job {job_id}\n")
                while True:
                    try:
                        line_bytes = await stream.readuntil(b"\n")
                    except asyncio.LimitOverrunError as e:
                        # tqdm overflow: drain the consumed bytes and continue
                        try:
                            dropped = await stream.readexactly(e.consumed)
                        except asyncio.IncompleteReadError as ie:
                            dropped = ie.partial
                        log_file.write(
                            f"<runner: dropped {len(dropped)} bytes of tqdm overflow>\n"
                        )
                        log_file.flush()
                        continue
                    except asyncio.IncompleteReadError as e:
                        line_bytes = e.partial
                        if not line_bytes:
                            break
                    if not line_bytes:
                        break
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if line:
                        log_file.write(f"{line}\n")
                        log_file.flush()

                        progress_info = parse_ddm_log(line)

                        if progress_info:
                            # Calculate progress: (global_step / total_steps_all_epochs) * 100
                            # parse_ddm_log returns current_step as global_step (epoch * steps_per_epoch + step_in_epoch)
                            total_epochs = training_jobs_cache.get(job_id).get("total_epochs", 30)
                            global_step = progress_info["current_step"]  # Already includes epoch info
                            steps_per_epoch = progress_info["total_steps"]
                            
                            total_steps_all = total_epochs * steps_per_epoch
                            progress = round((global_step / total_steps_all) * 100, 2)
                            
                            training_jobs_cache.update(
                                job_id,
                                current_step=global_step,
                                total_steps=total_steps_all,
                                loss=progress_info["loss"],
                                progress=progress,
                                updated_at=datetime.now(),
                            )

                        logger.info(f"Job {job_id}: {line}")

        # Run stream readers concurrently
        await asyncio.gather(read_stream(process.stdout), read_stream(process.stderr))

        # Wait for the process to complete
        return_code = await process.wait()

        # Final status update (clear PID since process has exited)
        if return_code == 0:
            training_jobs_cache.update(
                job_id,
                status=const.COMPLETED_STATUS,
                process_pid=None,
                updated_at=datetime.now(),
                progress=100.0,
            )
        elif training_jobs_cache.get(job_id)["status"] != const.CANCELLED_STATUS:
            training_jobs_cache.update(
                job_id,
                status=const.FAILED_STATUS,
                process_pid=None,
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
            process_pid=None,  # Clear PID on failure
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

