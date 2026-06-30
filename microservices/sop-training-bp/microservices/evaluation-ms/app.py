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
import json
import logging
import os
from pathlib import Path
import traceback
import tempfile
import uuid
from datetime import datetime
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import utils.constant as const
from components.cache import JobCache
from components.postgres_db import postgres_db
from utils.eval_utils import extract_mcq_data, parse_eval_results, prepare_eval_assets, resolve_checkpoint_path
from utils.e2e_eval_utils import load_ddm_config, resolve_ddm_checkpoint
from utils.utils import create_file, safe_dataset_path, terminate_process_tree
from validation.request_validation import (
    E2eEvaluationRequest, E2eEvaluationResponse, E2eEvaluationStatus,
    EvaluationRequest, EvaluationResponse, EvaluationStatus,
)


# Subprocess scripts resolved relative to this module so the service is
# deployment-root agnostic.
_SOP_EVAL_SCRIPT = str(Path(__file__).parent / "sop" / "sop_eval.py")
_SOP_E2E_EVAL_SCRIPT = str(Path(__file__).parent / "sop" / "sop_e2e_eval.py")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Evaluation Microservice",
    description="A microservice for evaluating SOP VLM models",
    openapi_tags=[
        {"name": "evaluation", "description": "Per-action-chunk VLM evaluation"},
        {"name": "e2e-evaluation", "description": "End-to-end DDM + VLM evaluation"},
        {"name": "status", "description": "Health + job status"},
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

# In-memory job caches to reduce DB load during active runs.
# Entries are evicted on terminal status (completed/cancelled/failed).
# TODO: replace with Redis for multi-replica deployments.
eval_jobs_cache = JobCache()
e2e_eval_jobs_cache = JobCache()


@app.get("/health", tags=["status"])
async def root():
    """Health check endpoint"""
    return {"message": "Evaluation Microservice is running"}


@app.get("/api/v1/gpus", tags=["status"])
async def list_gpus():
    """List the GPUs visible to this container.

    Returned shape: ``{"count": N, "gpus": [{"index": int, "name": str,
    "total_memory_mb": int, "free_memory_mb": int|null}, ...]}``.

    The frontend uses this to populate the per-job GPU selector. Free
    memory comes from pynvml when available (best-effort — set to None
    on any failure so the endpoint never errors out).
    """
    try:
        import torch
    except Exception as e:
        logger.warning("torch import failed in /gpus: %s", e)
        return {"count": 0, "gpus": []}

    if not torch.cuda.is_available():
        return {"count": 0, "gpus": []}

    count = torch.cuda.device_count()
    gpus = []

    # pynvml is best-effort: torch.cuda has no free-memory API.
    nvml_handles = None
    try:
        import pynvml
        pynvml.nvmlInit()
        nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
    except Exception as e:
        logger.debug("pynvml unavailable in /gpus (free-memory will be null): %s", e)
        nvml_handles = None

    for i in range(count):
        try:
            props = torch.cuda.get_device_properties(i)
            name = props.name
            total_mb = int(props.total_memory / (1024 * 1024))
        except Exception:
            name = f"GPU {i}"
            total_mb = None

        free_mb = None
        if nvml_handles is not None:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handles[i])
                free_mb = int(info.free / (1024 * 1024))
            except Exception:
                free_mb = None

        gpus.append({
            "index": i,
            "name": name,
            "total_memory_mb": total_mb,
            "free_memory_mb": free_mb,
        })

    if nvml_handles is not None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return {"count": count, "gpus": gpus}


def _validate_gpu_id(gpu_id: Optional[int]) -> Optional[int]:
    """Validate gpu_id against the visible device count. Returns the validated
    id (or None for "auto"). Raises HTTPException(400) on out-of-range ids.
    Imported lazily to keep the module importable without torch."""
    if gpu_id is None:
        return None
    try:
        import torch
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        count = 0
    if gpu_id < 0 or gpu_id >= count:
        # Special-case count=0 to avoid the nonsense range "0..-1".
        range_msg = f"0..{count - 1}" if count > 0 else "none"
        raise HTTPException(
            status_code=400,
            detail=f"gpu_id={gpu_id} is out of range (visible GPUs: {range_msg})",
        )
    return gpu_id


def _subprocess_env_for_gpu(gpu_id: Optional[int]) -> Optional[dict]:
    """Build the subprocess env dict with CUDA_VISIBLE_DEVICES set to gpu_id.
    Returns None when no override is needed (caller can pass env=None to
    inherit the parent's env)."""
    if gpu_id is None:
        return None
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


@app.post("/api/v1/evaluation/start", response_model=EvaluationResponse, tags=["evaluation"])
async def start_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Start a per-action-chunk evaluation job for a completed training experiment."""
    try:
        validated_gpu_id = _validate_gpu_id(request.gpu_id)

        # Singleton guard: one per-action-chunk eval at a time.
        for job in eval_jobs_cache.cache.values():
            if job["status"] == const.RUNNING_STATUS:
                raise HTTPException(
                    status_code=400,
                    detail=f"An evaluation is already running: {job['eval_job_id']}. Wait for it to finish or cancel it.",
                )

        training_job = await postgres_db.get_training_job(request.training_job_id)
        if training_job is None:
            raise HTTPException(status_code=404, detail=f"Training job not found: {request.training_job_id}")
        if training_job.status != const.COMPLETED_STATUS:
            raise HTTPException(
                status_code=400,
                detail=f"Training job {request.training_job_id} is not completed (status: {training_job.status}). Evaluation requires a completed training job.",
            )

        aug_dataset_id = training_job.aug_dataset_id

        # actions.json lives on the original (pre-augmentation) dataset.
        original_dataset_id = await postgres_db.get_original_dataset_id(aug_dataset_id)
        if not original_dataset_id:
            raise HTTPException(
                status_code=400,
                detail=f"Could not find original dataset for augmented dataset {aug_dataset_id}",
            )
        try:
            original_dataset_path = safe_dataset_path(const.DATASET_ROOT, original_dataset_id)
            # Validate val_dataset_id upfront so traversal attempts return 400
            # immediately rather than failing asynchronously in run_evaluation.
            safe_dataset_path(const.DATASET_ROOT, request.val_dataset_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        actions_json_path = os.path.join(original_dataset_path, "actions.json")
        if not os.path.exists(actions_json_path):
            raise HTTPException(
                status_code=400,
                detail=f"actions.json not found for dataset {original_dataset_id}: {actions_json_path}",
            )

        try:
            checkpoint_path, checkpoint_step = resolve_checkpoint_path(
                const.RESULTS_ROOT, request.training_job_id, step=request.checkpoint_step
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))

        eval_job_id = str(uuid.uuid4())
        log_path = os.path.join(const.RESULTS_ROOT, eval_job_id, const.LOG_FILENAME)
        create_file(log_path)

        now = datetime.now()
        eval_jobs_cache.set(
            eval_job_id,
            {
                "eval_job_id": eval_job_id,
                "training_job_id": request.training_job_id,
                "val_dataset_id": request.val_dataset_id,
                "checkpoint_step": checkpoint_step,
                "status": const.QUEUE_STATUS,
                "overall_accuracy": None,
                "results_json": None,
                "fps": request.fps,
                "temperature": request.temperature,
                "backend": request.backend,
                "created_at": now,
                "updated_at": now,
                "log_file_path": log_path,
                "process_pid": None,
            },
        )

        await postgres_db.insert_evaluation_job(
            id=eval_job_id,
            training_job_id=request.training_job_id,
            val_dataset_id=request.val_dataset_id,
            checkpoint_step=checkpoint_step,
            status=const.QUEUE_STATUS,
            fps=request.fps,
            temperature=request.temperature,
            backend=request.backend,
            created_at=now,
            updated_at=now,
        )

        background_tasks.add_task(
            run_evaluation,
            eval_job_id=eval_job_id,
            training_job_id=request.training_job_id,
            actions_json_path=actions_json_path,
            val_dataset_id=request.val_dataset_id,
            checkpoint_path=checkpoint_path,
            checkpoint_step=checkpoint_step,
            fps=request.fps,
            temperature=request.temperature,
            top_p=request.top_p,
            backend=request.backend,
            resolution_config=(
                request.resolution_config.model_dump(exclude_none=True)
                if request.resolution_config else None
            ),
            gpu_id=validated_gpu_id,
        )

        logger.info(f"Started evaluation job {eval_job_id}")

        return EvaluationResponse(
            eval_job_id=eval_job_id,
            status=const.QUEUE_STATUS,
            message="Evaluation job has been queued and will start shortly",
            created_at=now,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start evaluation: {str(e)}")


@app.get("/api/v1/evaluation/status/{eval_job_id}", response_model=EvaluationStatus, tags=["evaluation"])
async def get_evaluation_status(eval_job_id: str):
    """Get the status of an evaluation job."""
    try:
        job = eval_jobs_cache.get(eval_job_id)
        if job is None:
            db_job = await postgres_db.get_evaluation_job(eval_job_id)
            job = db_job.to_dict() if db_job else None
        if job is None:
            raise HTTPException(status_code=404, detail="Evaluation job not found")
        return EvaluationStatus(
            eval_job_id=eval_job_id,
            training_job_id=job.get("training_job_id", ""),
            val_dataset_id=job.get("val_dataset_id", ""),
            status=job["status"],
            overall_accuracy=job.get("overall_accuracy"),
            checkpoint_step=job.get("checkpoint_step"),
            created_at=job["created_at"],
            updated_at=job["updated_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/evaluation/results/{eval_job_id}", tags=["evaluation"])
async def get_evaluation_results(eval_job_id: str):
    """Get the full per-action accuracy results for a completed evaluation job."""
    try:
        db_job = await postgres_db.get_evaluation_job(eval_job_id)
        if db_job is None:
            raise HTTPException(status_code=404, detail="Evaluation job not found")
        if db_job.status != const.COMPLETED_STATUS:
            raise HTTPException(status_code=400, detail=f"Evaluation not completed (status: {db_job.status})")
        return db_job.results_json
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/evaluation/all_jobs", tags=["evaluation"])
async def get_all_evaluation_jobs():
    """List all evaluation jobs."""
    all_jobs = await postgres_db.list_evaluation_jobs()
    return {job.id: job.to_dict() for job in all_jobs}


@app.post("/api/v1/evaluation/cancel/{eval_job_id}", tags=["evaluation"])
async def cancel_evaluation(eval_job_id: str):
    """Cancel a running evaluation job."""
    job = eval_jobs_cache.get(eval_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Evaluation job not in running state or not found")
    process_pid = job.get("process_pid")
    if not process_pid:
        return {"message": f"Evaluation job {eval_job_id} has not started yet. Try again in a moment."}
    success = terminate_process_tree(process_pid)
    if success:
        updated_at = datetime.now()
        eval_jobs_cache.update(eval_job_id, status=const.CANCELLED_STATUS, updated_at=updated_at)
        await postgres_db.update_evaluation_job(
            eval_job_id,
            status=const.CANCELLED_STATUS,
            updated_at=updated_at,
        )
        return {"message": f"Evaluation job {eval_job_id} cancelled successfully"}
    return {"message": f"Failed to cancel evaluation job {eval_job_id}"}


# ===========================================================================
# E2E Evaluation Endpoints
# ===========================================================================


@app.post("/api/v1/e2e-evaluation/start", response_model=E2eEvaluationResponse, tags=["e2e-evaluation"])
async def start_e2e_evaluation(request: E2eEvaluationRequest, background_tasks: BackgroundTasks):
    """Start an end-to-end evaluation job (DDM temporal segmentation + VLM action recognition)."""
    try:
        validated_gpu_id = _validate_gpu_id(request.gpu_id)

        # Singleton guard: one e2e eval at a time.
        for job in e2e_eval_jobs_cache.cache.values():
            if job["status"] == const.RUNNING_STATUS:
                raise HTTPException(
                    status_code=400,
                    detail=f"An e2e evaluation is already running: {job['eval_job_id']}. Wait for it to finish or cancel it.",
                )

        training_job = await postgres_db.get_training_job(request.training_job_id)
        if training_job is None:
            raise HTTPException(status_code=404, detail=f"VLM training job not found: {request.training_job_id}")
        if training_job.status != const.COMPLETED_STATUS:
            raise HTTPException(
                status_code=400,
                detail=f"VLM training job {request.training_job_id} is not completed (status: {training_job.status}).",
            )

        # DDM job + checkpoint only required when DDM is the chunker; uniform
        # chunking skips the DDM stage entirely.
        if request.chunking_algorithm == "ddm":
            ddm_job = await postgres_db.get_ddm_training_job(request.ddm_training_job_id)
            if ddm_job is None:
                raise HTTPException(status_code=404, detail=f"DDM training job not found: {request.ddm_training_job_id}")
            if ddm_job.status != const.COMPLETED_STATUS:
                raise HTTPException(
                    status_code=400,
                    detail=f"DDM training job {request.ddm_training_job_id} is not completed (status: {ddm_job.status}).",
                )

        try:
            val_dataset_path = safe_dataset_path(const.DATASET_ROOT, request.val_dataset_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        actions_json_path = os.path.join(val_dataset_path, "actions.json")
        if not os.path.exists(actions_json_path):
            raise HTTPException(
                status_code=400,
                detail=f"actions.json not found for dataset {request.val_dataset_id}: {actions_json_path}",
            )

        try:
            checkpoint_path, checkpoint_step = resolve_checkpoint_path(
                const.RESULTS_ROOT, request.training_job_id, step=request.checkpoint_step
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Uniform chunking leaves these None; the subprocess then receives
        # no --ddm-* args.
        ddm_checkpoint_path = None
        ddm_resolution: Optional[int] = None
        ddm_frames_per_side: Optional[int] = None
        if request.chunking_algorithm == "ddm":
            try:
                ddm_checkpoint_path, ddm_config_path = resolve_ddm_checkpoint(
                    const.RESULTS_ROOT, request.ddm_training_job_id, checkpoint_name=request.ddm_checkpoint
                )
                ddm_config = load_ddm_config(ddm_config_path)
                ddm_resolution = ddm_config["resolution"]
                ddm_frames_per_side = ddm_config["frames_per_side"]
            except FileNotFoundError as e:
                raise HTTPException(status_code=400, detail=str(e))

        from sop.sop_e2e_eval import collect_annotations
        anno_json = collect_annotations(val_dataset_path)
        if not anno_json:
            raise HTTPException(
                status_code=400,
                detail=f"No annotation JSONs found in dataset {request.val_dataset_id}",
            )

        eval_job_id = str(uuid.uuid4())
        output_dir = os.path.join(const.RESULTS_ROOT, eval_job_id)
        os.makedirs(output_dir, exist_ok=True)
        anno_json_path = os.path.join(output_dir, "anno.json")
        with open(anno_json_path, "w") as f:
            json.dump(anno_json, f, indent=2)

        # Writes vlm_prompts.txt under <output_dir>/assets/; the subprocess
        # re-derives the same path.
        prompt_text, _ = extract_mcq_data(actions_json_path)
        prepare_eval_assets(eval_job_id, prompt_text)

        log_path = os.path.join(output_dir, const.LOG_FILENAME)
        create_file(log_path)

        now = datetime.now()
        ddm_checkpoint_basename = (
            os.path.basename(ddm_checkpoint_path) if ddm_checkpoint_path else None
        )

        e2e_eval_jobs_cache.set(
            eval_job_id,
            {
                "eval_job_id": eval_job_id,
                "training_job_id": request.training_job_id,
                "ddm_training_job_id": request.ddm_training_job_id,
                "val_dataset_id": request.val_dataset_id,
                "checkpoint_step": checkpoint_step,
                "ddm_checkpoint": ddm_checkpoint_basename,
                "status": const.QUEUE_STATUS,
                "overall_accuracy": None,
                "avg_f1": None,
                "results_json": None,
                "fps": request.fps,
                "temperature": request.temperature,
                "backend": request.backend,
                "score_threshold": request.score_threshold,
                "nms_sec": request.nms_sec,
                "ddm_batch_size": request.ddm_batch_size,
                "chunking_algorithm": request.chunking_algorithm,
                "chunk_length_sec": request.chunk_length_sec,
                "created_at": now,
                "updated_at": now,
                "log_file_path": log_path,
                "process_pid": None,
            },
        )

        await postgres_db.insert_e2e_evaluation_job(
            id=eval_job_id,
            training_job_id=request.training_job_id,
            ddm_training_job_id=request.ddm_training_job_id,
            val_dataset_id=request.val_dataset_id,
            checkpoint_step=checkpoint_step,
            ddm_checkpoint=ddm_checkpoint_basename,
            status=const.QUEUE_STATUS,
            fps=request.fps,
            temperature=request.temperature,
            backend=request.backend,
            score_threshold=request.score_threshold,
            nms_sec=request.nms_sec,
            ddm_batch_size=request.ddm_batch_size,
            chunking_algorithm=request.chunking_algorithm,
            chunk_length_sec=request.chunk_length_sec,
            created_at=now,
            updated_at=now,
        )

        background_tasks.add_task(
            run_e2e_evaluation,
            eval_job_id=eval_job_id,
            actions_json_path=actions_json_path,
            anno_json_path=anno_json_path,
            val_dataset_id=request.val_dataset_id,
            checkpoint_path=checkpoint_path,
            checkpoint_step=checkpoint_step,
            ddm_checkpoint_path=ddm_checkpoint_path,
            ddm_resolution=ddm_resolution,
            ddm_frames_per_side=ddm_frames_per_side,
            fps=request.fps,
            temperature=request.temperature,
            top_p=request.top_p,
            backend=request.backend,
            score_threshold=request.score_threshold,
            nms_sec=request.nms_sec,
            ddm_batch_size=request.ddm_batch_size,
            frames_per_segment_hint=request.frames_per_segment_hint,
            resolution_config=(
                request.resolution_config.model_dump(exclude_none=True)
                if request.resolution_config else None
            ),
            chunking_algorithm=request.chunking_algorithm,
            chunk_length_sec=request.chunk_length_sec,
            gpu_id=validated_gpu_id,
        )

        logger.info(f"Started e2e evaluation job {eval_job_id}")
        return E2eEvaluationResponse(
            eval_job_id=eval_job_id,
            status=const.QUEUE_STATUS,
            message="E2E evaluation job has been queued and will start shortly",
            created_at=now,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting e2e evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start e2e evaluation: {str(e)}")


@app.get("/api/v1/e2e-evaluation/status/{eval_job_id}", response_model=E2eEvaluationStatus, tags=["e2e-evaluation"])
async def get_e2e_evaluation_status(eval_job_id: str):
    """Get the status of an e2e evaluation job."""
    try:
        job = e2e_eval_jobs_cache.get(eval_job_id)
        if job is None:
            db_job = await postgres_db.get_e2e_evaluation_job(eval_job_id)
            job = db_job.to_dict() if db_job else None
        if job is None:
            raise HTTPException(status_code=404, detail="E2E evaluation job not found")
        return E2eEvaluationStatus(
            eval_job_id=eval_job_id,
            training_job_id=job.get("training_job_id", ""),
            ddm_training_job_id=job.get("ddm_training_job_id", ""),
            val_dataset_id=job.get("val_dataset_id", ""),
            status=job["status"],
            overall_accuracy=job.get("overall_accuracy"),
            avg_f1=job.get("avg_f1"),
            checkpoint_step=job.get("checkpoint_step"),
            created_at=job["created_at"],
            updated_at=job["updated_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/e2e-evaluation/results/{eval_job_id}", tags=["e2e-evaluation"])
async def get_e2e_evaluation_results(eval_job_id: str):
    """Get the full results for a completed e2e evaluation job."""
    try:
        db_job = await postgres_db.get_e2e_evaluation_job(eval_job_id)
        if db_job is None:
            raise HTTPException(status_code=404, detail="E2E evaluation job not found")
        if db_job.status != const.COMPLETED_STATUS:
            raise HTTPException(status_code=400, detail=f"E2E evaluation not completed (status: {db_job.status})")
        return db_job.results_json
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/e2e-evaluation/all_jobs", tags=["e2e-evaluation"])
async def get_all_e2e_evaluation_jobs():
    """List all e2e evaluation jobs."""
    all_jobs = await postgres_db.list_e2e_evaluation_jobs()
    return {job.id: job.to_dict() for job in all_jobs}


@app.post("/api/v1/e2e-evaluation/cancel/{eval_job_id}", tags=["e2e-evaluation"])
async def cancel_e2e_evaluation(eval_job_id: str):
    """Cancel a running e2e evaluation job."""
    job = e2e_eval_jobs_cache.get(eval_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="E2E evaluation job not in running state or not found")
    process_pid = job.get("process_pid")
    if not process_pid:
        return {"message": f"E2E evaluation job {eval_job_id} has not started yet. Try again in a moment."}
    success = terminate_process_tree(process_pid)
    if success:
        updated_at = datetime.now()
        e2e_eval_jobs_cache.update(eval_job_id, status=const.CANCELLED_STATUS, updated_at=updated_at)
        await postgres_db.update_e2e_evaluation_job(
            eval_job_id,
            status=const.CANCELLED_STATUS,
            updated_at=updated_at,
        )
        return {"message": f"E2E evaluation job {eval_job_id} cancelled successfully"}
    return {"message": f"Failed to cancel e2e evaluation job {eval_job_id}"}


def _append_to_file(path: str, text: str):
    """Sync file-append helper (extracted to avoid sync open() in async context)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        f.flush()


async def run_evaluation(
    eval_job_id: str,
    training_job_id: str,
    actions_json_path: str,
    val_dataset_id: str,
    checkpoint_path: str,
    checkpoint_step: int,
    fps: int,
    temperature: float,
    backend: str,
    top_p: float = 1.0,
    resolution_config: Optional[dict] = None,
    gpu_id: Optional[int] = None,
):
    """Background task to run per-action-chunk VLM evaluation."""
    cached = eval_jobs_cache.get(eval_job_id) or {}
    log_file_path = cached.get("log_file_path", os.path.join(tempfile.gettempdir(), f"eval_{eval_job_id}.txt"))
    try:
        eval_jobs_cache.update(eval_job_id, status=const.RUNNING_STATUS, updated_at=datetime.now())
        await postgres_db.update_evaluation_job(
            eval_job_id,
            status=const.RUNNING_STATUS,
            updated_at=eval_jobs_cache.get(eval_job_id)["updated_at"],
        )

        # MCQ prompt generation matches the augmentation pipeline.
        prompt_text, choices = extract_mcq_data(actions_json_path)
        asset_root = prepare_eval_assets(eval_job_id, prompt_text)

        val_videos_path = safe_dataset_path(const.DATASET_ROOT, val_dataset_id)
        output_dir = os.path.join(const.RESULTS_ROOT, eval_job_id)
        inference_json_path = os.path.join(output_dir, "inference_results.json")

        cmd = [
            "python",
            _SOP_EVAL_SCRIPT,
            "--model-path", checkpoint_path,
            "--val-videos-path", val_videos_path,
            "--asset-root", asset_root,
            "--output-dir", output_dir,
            "--output-name", "inference_results",
            "--temperature", str(temperature),
            # Default mirrors training config (max_frames=40, 16k vision
            # tokens). Eval-at-training-resolution keeps the VLM
            # in-distribution; the UI advanced panel can still override.
            "--resolution-config", json.dumps(resolution_config) if resolution_config else '{"max_frames": 40, "total_pixels": 16572416}',
            "--fps", str(fps),
            "--top_p", str(top_p),
            "--backend", backend,
            "--use-fps-or-nframes", "fps",
        ]

        env = _subprocess_env_for_gpu(gpu_id)
        if gpu_id is not None:
            logger.info(f"Pinning evaluation subprocess to GPU {gpu_id} via CUDA_VISIBLE_DEVICES")
        logger.info(f"Running eval command: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        eval_jobs_cache.update(eval_job_id, process_pid=process.pid)

        log_file_path = eval_jobs_cache.get(eval_job_id)["log_file_path"]

        # Header written once here (not inside read_stream) so the
        # concurrent stdout+stderr readers don't duplicate it.
        _append_to_file(log_file_path, f"Starting evaluation job {eval_job_id}\n")

        async def read_stream(stream):
            while True:
                line_bytes = await stream.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8").strip()
                if line:
                    _append_to_file(log_file_path, f"{line}\n")
                    logger.info(f"Eval {eval_job_id}: {line}")

        await asyncio.gather(read_stream(process.stdout), read_stream(process.stderr))
        return_code = await process.wait()

        if return_code == 0:
            inference_results = json.loads(Path(inference_json_path).read_text())
            metrics = parse_eval_results(inference_results, choices)
            eval_jobs_cache.update(
                eval_job_id,
                status=const.COMPLETED_STATUS,
                overall_accuracy=metrics["overall_accuracy"],
                results_json=metrics,
                updated_at=datetime.now(),
            )
        elif eval_jobs_cache.get(eval_job_id)["status"] != const.CANCELLED_STATUS:
            eval_jobs_cache.update(eval_job_id, status=const.FAILED_STATUS, updated_at=datetime.now())

        final_cache = eval_jobs_cache.get(eval_job_id)
        await postgres_db.update_evaluation_job(
            eval_job_id,
            status=final_cache["status"],
            overall_accuracy=final_cache.get("overall_accuracy"),
            results_json=final_cache.get("results_json"),
            updated_at=final_cache["updated_at"],
        )

        logger.info(f"Evaluation job {eval_job_id} finished: {final_cache['status']}")
        eval_jobs_cache.delete(eval_job_id)

    except Exception as e:
        logger.error(f"Error in eval job {eval_job_id}: {traceback.format_exc()}")
        eval_jobs_cache.update(eval_job_id, status=const.FAILED_STATUS, updated_at=datetime.now())
        _append_to_file(log_file_path, f"[ERROR] {str(e)}\n")
        await postgres_db.update_evaluation_job(
            eval_job_id,
            status=const.FAILED_STATUS,
            updated_at=eval_jobs_cache.get(eval_job_id)["updated_at"],
        )
        eval_jobs_cache.delete(eval_job_id)


async def run_e2e_evaluation(
    eval_job_id: str,
    actions_json_path: str,
    anno_json_path: str,
    val_dataset_id: str,
    checkpoint_path: str,
    checkpoint_step: int,
    ddm_checkpoint_path: Optional[str],
    ddm_resolution: Optional[int],
    ddm_frames_per_side: Optional[int],
    fps: int,
    temperature: float,
    backend: str,
    score_threshold: float,
    nms_sec: float,
    ddm_batch_size: int,
    frames_per_segment_hint: int,
    top_p: float = 1.0,
    resolution_config: Optional[dict] = None,
    chunking_algorithm: str = "ddm",
    chunk_length_sec: Optional[float] = None,
    gpu_id: Optional[int] = None,
):
    """Background task to run end-to-end (DDM + VLM) evaluation."""
    cached = e2e_eval_jobs_cache.get(eval_job_id) or {}
    log_file_path = cached.get("log_file_path", os.path.join(tempfile.gettempdir(), f"e2e_eval_{eval_job_id}.txt"))
    try:
        e2e_eval_jobs_cache.update(eval_job_id, status=const.RUNNING_STATUS, updated_at=datetime.now())
        await postgres_db.update_e2e_evaluation_job(
            eval_job_id,
            status=const.RUNNING_STATUS,
            updated_at=e2e_eval_jobs_cache.get(eval_job_id)["updated_at"],
        )

        val_videos_path = safe_dataset_path(const.DATASET_ROOT, val_dataset_id)
        output_dir = os.path.join(const.RESULTS_ROOT, eval_job_id)
        asset_root = os.path.join(output_dir, "assets")
        e2e_results_path = os.path.join(output_dir, "e2e_results.json")

        cmd = [
            "python",
            _SOP_E2E_EVAL_SCRIPT,
            "--vlm-model-path", checkpoint_path,
            "--video-dir", val_videos_path,
            "--asset-root", asset_root,
            "--output-dir", output_dir,
            "--anno-json-path", anno_json_path,
            "--actions-json-path", actions_json_path,
            "--chunking-algorithm", chunking_algorithm,
            "--score-threshold", str(score_threshold),
            "--nms-sec", str(nms_sec),
            "--ddm-batch-size", str(ddm_batch_size),
            "--frames-per-segment-hint", str(frames_per_segment_hint),
            "--fps", str(fps),
            "--temperature", str(temperature),
            "--top-p", str(top_p),
            "--backend", backend,
        ]
        # DDM-specific args only when DDM is the chunker. Uniform mode skips
        # them entirely (sop_e2e_eval.py treats ddm-checkpoint-path as optional
        # and won't try to load DDM-Net at all).
        if chunking_algorithm == "ddm" and ddm_checkpoint_path:
            cmd.extend([
                "--ddm-checkpoint-path", ddm_checkpoint_path,
                "--ddm-resolution", str(ddm_resolution),
                "--ddm-frames-per-side", str(ddm_frames_per_side),
            ])
        if chunking_algorithm == "uniform" and chunk_length_sec is not None:
            cmd.extend(["--chunk-length-sec", str(chunk_length_sec)])
        if resolution_config:
            cmd.extend(["--resolution-config", json.dumps(resolution_config)])

        env = _subprocess_env_for_gpu(gpu_id)
        if gpu_id is not None:
            logger.info(f"Pinning e2e evaluation subprocess to GPU {gpu_id} via CUDA_VISIBLE_DEVICES")
        logger.info(f"Running e2e eval command: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        e2e_eval_jobs_cache.update(eval_job_id, process_pid=process.pid)

        # Header written once here so the concurrent stdout+stderr
        # readers don't duplicate it.
        _append_to_file(log_file_path, f"Starting e2e evaluation job {eval_job_id}\n")

        async def read_stream(stream):
            while True:
                line_bytes = await stream.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8").strip()
                if line:
                    _append_to_file(log_file_path, f"{line}\n")
                    logger.info(f"E2E Eval {eval_job_id}: {line}")

        await asyncio.gather(read_stream(process.stdout), read_stream(process.stderr))
        return_code = await process.wait()

        if return_code == 0:
            e2e_results = json.loads(Path(e2e_results_path).read_text())
            overall_accuracy = e2e_results.get("action_recognition", {}).get("overall_accuracy", 0.0)
            avg_f1 = e2e_results.get("temporal_segmentation", {}).get("avg_f1", 0.0)
            e2e_eval_jobs_cache.update(
                eval_job_id,
                status=const.COMPLETED_STATUS,
                overall_accuracy=overall_accuracy,
                avg_f1=avg_f1,
                results_json=e2e_results,
                updated_at=datetime.now(),
            )
        elif e2e_eval_jobs_cache.get(eval_job_id)["status"] != const.CANCELLED_STATUS:
            e2e_eval_jobs_cache.update(eval_job_id, status=const.FAILED_STATUS, updated_at=datetime.now())

        final_cache = e2e_eval_jobs_cache.get(eval_job_id)
        await postgres_db.update_e2e_evaluation_job(
            eval_job_id,
            status=final_cache["status"],
            overall_accuracy=final_cache.get("overall_accuracy"),
            avg_f1=final_cache.get("avg_f1"),
            results_json=final_cache.get("results_json"),
            updated_at=final_cache["updated_at"],
        )

        logger.info(f"E2E evaluation job {eval_job_id} finished: {final_cache['status']}")
        e2e_eval_jobs_cache.delete(eval_job_id)

    except Exception as e:
        logger.error(f"Error in e2e eval job {eval_job_id}: {traceback.format_exc()}")
        e2e_eval_jobs_cache.update(eval_job_id, status=const.FAILED_STATUS, updated_at=datetime.now())
        _append_to_file(log_file_path, f"[ERROR] {str(e)}\n")
        await postgres_db.update_e2e_evaluation_job(
            eval_job_id,
            status=const.FAILED_STATUS,
            updated_at=e2e_eval_jobs_cache.get(eval_job_id)["updated_at"],
        )
        e2e_eval_jobs_cache.delete(eval_job_id)
