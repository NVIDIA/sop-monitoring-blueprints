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


from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import traceback
import uuid
import zipfile
from contextlib import asynccontextmanager
from sqlalchemy import text
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from moviepy.editor import VideoFileClip, concatenate_videoclips

from fastapi import (
    Body,
    FastAPI,
    File,
    HTTPException,
    Path,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

import utils.constant as const
from components.postgres_db import postgres_db
from utils.logger import app_logger
from utils.utils import (
    clean_up_file,
    convert_to_h264,
    create_dir,
    safe_dataset_path,
)
from validations.data_models import (
    ActionsUploadResponse,
    ClearDatasetResponse,
    ResetActionsResponse,
    VideoMetadata,
    VideoUploadResponse,
)
from validations.db_models import Annotation, Chunk, Dataset, Video

import yaml as _yaml  # local alias to avoid colliding with any existing yaml usage

_AUGMENT_CONFIG_PATH = os.getenv(
    "AUGMENT_CONFIG_PATH", "/app/assets/config/augment_config.yaml"
)


def _load_merge_threshold() -> float:
    """Read merge_small_chunks threshold from augment_config.yaml.

    Returns 0.0 (no merge) if disabled or the file is missing.
    """
    try:
        with open(_AUGMENT_CONFIG_PATH) as f:
            cfg = _yaml.safe_load(f) or {}
    except FileNotFoundError:
        app_logger.warning(
            f"augment_config not found at {_AUGMENT_CONFIG_PATH}; skipping chunk merge"
        )
        return 0.0
    msc = cfg.get("merge_small_chunks") or {}
    if not msc.get("enable", True):
        return 0.0
    return float(msc.get("threshold", 0.2))


# Global data_id to track current dataset version
current_data_id = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI app"""
    # Startup
    app_logger.info("Starting annotation backend service")

    # Ensure video directory exists
    create_dir(const.VIDEO_ROOT)

    # Auto-migrate: add two_operator_mode column if it doesn't exist
    try:
        async with postgres_db.engine.begin() as conn:
            await conn.execute(text(
                "ALTER TABLE dataset ADD COLUMN IF NOT EXISTS two_operator_mode BOOLEAN DEFAULT FALSE"
            ))
    except Exception as e:
        app_logger.warning(f"Auto-migration for two_operator_mode skipped: {e}")

    yield

    # Shutdown
    app_logger.info("Shutting down annotation backend service")


# Create FastAPI application
app = FastAPI(
    title="annotation_backend",
    description="Video annotation backend service",
    version="1.0.0",
    lifespan=lifespan,
)

# Reject oversized request bodies up front to mitigate resource exhaustion (T17 / FSR-AVA-1).
# Configurable via MAX_REQUEST_BODY_MB (default 2048 MB, matching the nginx proxy cap so
# legitimate video uploads still pass). Streamed bodies without Content-Length are not
# covered here — operators should also enforce limits at the reverse proxy.
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


@app.post("/api/v1/dataset/{data_id}/set_current")
async def set_current_dataset(data_id: str = Path(..., description="Data ID to set as current")):
    """Set the current global data_id to an existing dataset"""
    global current_data_id

    try:
        # Verify dataset exists
        dataset = await postgres_db.get_data(data_id, Dataset)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {data_id} not found")

        current_data_id = data_id
        app_logger.info(f"Set current_data_id to existing dataset: {data_id}")

        return {"status": "success", "message": f"Current dataset set to {data_id}", "data_id": data_id}
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error setting current dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/dataset/{data_id}/set_two_operator_mode")
async def set_two_operator_mode(
    data_id: str = Path(..., description="Data ID"),
    request_body: dict = None,
):
    """Update the two_operator_mode flag for a dataset"""
    try:
        dataset = await postgres_db.get_data(data_id, Dataset)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {data_id} not found")

        if not request_body or "two_operator_mode" not in request_body:
            raise HTTPException(status_code=400, detail="Missing 'two_operator_mode' in request body")

        new_mode = bool(request_body["two_operator_mode"])
        await postgres_db.update_data(
            data_id,
            Dataset,
            two_operator_mode=new_mode,
            updated_at=datetime.now(),
        )

        app_logger.info(f"Set two_operator_mode={new_mode} for dataset {data_id}")
        return {
            "status": "success",
            "data_id": data_id,
            "two_operator_mode": new_mode,
        }
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error setting two_operator_mode: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/upload")
async def upload_video(file: UploadFile = File(...)) -> VideoUploadResponse:
    """Upload video file"""

    global current_data_id

    if not current_data_id:
        raise HTTPException(
            status_code=400,
            detail="Data ID is not set. Please upload actions.json first.",
        )

    file_data = file.file.read()

    # Extract just the filename from the path (in case it contains path separators from folder upload)
    original_filename = file.filename
    safe_basename = (
        os.path.basename(original_filename)
        if original_filename
        else "unknown_file" + const.DEFAULT_VIDEO_EXTENSION
    )

    try:
        # Ensure video directory exists
        videos_dir = os.path.join(const.VIDEO_ROOT, current_data_id)

        # Ensure the filename always ends with .mp4 for consistency
        if not safe_basename.lower().endswith(const.DEFAULT_VIDEO_EXTENSION):
            # If the file doesn't have .mp4 extension, add it
            name_without_ext = os.path.splitext(safe_basename)[0]
            safe_basename = f"{name_without_ext}" + const.DEFAULT_VIDEO_EXTENSION
            app_logger.info(f"Normalized filename to: {safe_basename}")

        # Check if video with same original filename already exists in this dataset
        # We need to list all videos in this dataset and check their names
        # The names in DB are stored as "{uuid}_{original_name}"
        existing_videos = await postgres_db.list_data(Video, condition={"dataset_id": current_data_id})

        video_id = None
        final_file_name = None

        for v in existing_videos:
            # Check if the stored filename ends with "_" + safe_basename
            if v.name.endswith(const.ID_NAME_SEPARATOR + safe_basename):
                app_logger.info(f"Found existing video for {safe_basename}: {v.id}")
                video_id = v.id
                final_file_name = v.name
                break

        if not video_id:
            # Generate unique video ID if new
            video_id = str(uuid.uuid4())
            # Create safe filename with UUID to prevent conflicts
            final_file_name = f"{video_id}{const.ID_NAME_SEPARATOR}{safe_basename}"
            app_logger.info(f"New video upload: {safe_basename} -> {video_id}")
        else:
            app_logger.info(f"Overwriting existing video: {safe_basename} (ID: {video_id})")

        temp_file_path = os.path.join(videos_dir, f"temp_{final_file_name}")
        final_file_path = os.path.join(videos_dir, final_file_name)

        # Save original file to temporary location first
        with open(temp_file_path, "wb") as f:
            f.write(file_data)

        file_size = len(file_data)

        # Enhanced file size validation
        if file_size < 1024:  # Less than 1KB
            app_logger.error(f"Uploaded file is too small ({file_size} bytes)")
            clean_up_file(temp_file_path)

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is too small to be a valid video",
            )

        # Log file size for monitoring
        file_size_mb = file_size / (1024 * 1024)
        app_logger.info(f"Processing video file: {safe_basename}, size: {file_size_mb:.1f} MB")

        # Warn if file is very large
        if file_size_mb > 1024:  # More than 1GB
            app_logger.warning(f"Large file upload detected: {file_size_mb:.1f} MB. This may take some time to process.")

        # Convert to H264 encoding to ensure compatibility
        try:
            # TODO: Modify video storage to MinIO
            converted_file_path = await convert_to_h264(temp_file_path, final_file_path)

            # Update file size after conversion
            final_file_size = os.path.getsize(converted_file_path)

            clean_up_file(temp_file_path)

        except Exception as e:
            app_logger.error(f"Error converting video to H264: {str(e)}")
            clean_up_file(temp_file_path)
            clean_up_file(final_file_path)

            raise HTTPException(
                status_code=500, detail=f"Video conversion failed: {str(e)}"
            )

        # Save or Update metadata to database
        # If video_id existed, we update; if not, we insert.
        # However, upsert logic is simpler if we just check existence again or use delete-then-insert
        # But delete-then-insert might break foreign keys (cascade).
        # Since we are reusing video_id, we should UPDATE the record or just let it be if attributes haven't changed.
        # But file_size might have changed.

        existing_video_record = await postgres_db.get_data(video_id, Video)

        if existing_video_record:
            await postgres_db.update_data(
                video_id,
                Video,
                file_size=final_file_size,
                updated_at=datetime.now()
            )
            app_logger.info(f"Updated existing video record: {video_id}")
        else:
            await postgres_db.insert_data(
                Video,
                id=video_id,
                dataset_id=current_data_id,
                name=final_file_name,
                mime_type=const.MIME_TYPE,
                file_size=final_file_size,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            app_logger.info(f"Inserted new video record: {video_id}")

        app_logger.info(
            f"File '{final_file_name}' (original: '{original_filename}') successfully converted and saved to {converted_file_path} and recorded in database, ID: {video_id}"
        )

        return VideoUploadResponse(
            message="Video file has been successfully uploaded, converted to H264, and saved",
            file_id=video_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file '{safe_basename}' (original: '{original_filename}'): {str(e)}",
        )


@app.post("/api/v1/actions/upload")
async def upload_actions(file: UploadFile = File(...)) -> ActionsUploadResponse:
    """Upload actions.json file and save it to the videos directory

    Returns:
        ActionsUploadResponse: Upload response with status and message
    """
    global current_data_id

    try:
        app_logger.info(f"Uploading actions file: {file.filename}")

        # Read and validate JSON content
        file_content = await file.read()

        actions_data = json.loads(file_content.decode("utf-8"))
        actions_array = actions_data["actions"]

        if not isinstance(actions_array, list) or len(actions_array) == 0:
            raise HTTPException(
                status_code=400,
                detail="'actions' must be an array and cannot be empty",
            )

        app_logger.info(f"Actions file contains {len(actions_array)} actions")

        # Generate new data_id for this dataset version
        new_data_id = str(uuid.uuid4())
        current_data_id = new_data_id
        app_logger.info(f"Generated new data_id: {new_data_id}")

        # Create data_id specific directory
        data_dir = os.path.join(const.VIDEO_ROOT, new_data_id)
        create_dir(data_dir)
        app_logger.info(f"Created data directory: {data_dir}")

        # Save actions.json to data_id directory
        actions_file_path = os.path.join(data_dir, "actions.json")
        with open(actions_file_path, "w", encoding="utf-8") as f:
            json.dump(actions_data, f, indent=2, ensure_ascii=False)
        app_logger.info(f"Saved actions file with {len(actions_array)} actions")

        # Save actions to database (two_operator_mode defaults to False, set later via toggle)
        await postgres_db.insert_data(
            Dataset,
            id=new_data_id,
            actions=actions_array,
            two_operator_mode=False,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        return ActionsUploadResponse(
            status="success",
            message=f"Actions file uploaded successfully with {len(actions_array)} actions",
            actions_count=len(actions_array),
            actions=actions_array,
            data_id=new_data_id,
        )
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        app_logger.error(f"Invalid JSON format: {str(e)}")
        app_logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=400, detail="Invalid JSON format"
        )
    except Exception as e:
        app_logger.error(f"Error uploading actions file: {str(e)}")
        app_logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Failed to upload actions file",
        )


@app.post("/api/v1/actions/reset")
async def reset_actions() -> ResetActionsResponse:
    """Reset the current data_id to start a new dataset annotation

    Returns:
        ResetActionsResponse: Reset response with status
    """
    global current_data_id

    try:
        if current_data_id is None:
            raise HTTPException(
                status_code=400,
                detail="No data_id is currently set. Nothing to reset."
            )
        old_data_id = current_data_id
        current_data_id = None
        app_logger.info(f"Reset data_id from {old_data_id} to None")

        return ResetActionsResponse(
            status="success",
            message="Data ID reset successfully. Ready for new dataset annotation.",
            previous_data_id=old_data_id,
        )
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive; reset body has no DB/IO that can fail
        app_logger.error(f"Error resetting data_id: {str(e)}")
        app_logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Failed to reset data_id",
        )


@app.get("/api/v1/videos/{video_id}")
async def get_video(video_id: str = Path(..., description="Video ID")) -> VideoMetadata:
    """Get video metadata by ID"""

    try:
        app_logger.info(f"Getting video metadata for ID: {video_id}")

        metadata = await postgres_db.get_data(video_id, Video)
        if not metadata:
            raise HTTPException(status_code=404, detail="Video not found")

        return VideoMetadata(
            id=metadata.id,
            filename=metadata.name,
            file_path=os.path.join(
                const.VIDEO_ROOT, metadata.dataset_id, metadata.name
            ),
            file_size=metadata.file_size,
            upload_time=metadata.created_at.isoformat(),
            mime_type=metadata.mime_type,
        )
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error getting video metadata: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting video metadata: {str(e)}"
        )


@app.get("/api/v1/datasets")
async def get_datasets_info() -> Dict:
    """Get metadata, including actions, annotations and chunks, for current all datasets"""

    try:
        # format: {data_id: {video_id: {original_file_name:.., total_duration:..., processed_at: ..., clips: [...]}}}
        all_datasets_info = {}
        app_logger.info("Getting metadata information for all datasets")

        datasets = await postgres_db.list_data(Dataset)

        # get all videos for each dataset
        for dataset in datasets:
            dataset_info = {}
            videos = await postgres_db.list_data(
                Video, condition={"dataset_id": dataset.id}
            )

            for video in videos:
                all_chunks_annotations = await postgres_db.list_data(
                    Annotation, condition={"video_id": video.id}
                )

                if not all_chunks_annotations:
                    app_logger.warning(f"No annotations found for video: {video.id}")
                    continue

                original_file_name = video.name.replace(
                    video.id + const.ID_NAME_SEPARATOR, ""
                )
                total_duration = sum(
                    [
                        annotation.end_time - annotation.start_time
                        for annotation in all_chunks_annotations
                    ]
                )
                processed_at = max(
                    [annotation.created_at for annotation in all_chunks_annotations]
                )

                async def _format_clips_info(annotation):
                    # get corresponding chunk
                    chunk = await postgres_db.get_data(annotation.chunk_id, Chunk)

                    return {
                        "id": chunk.id,
                        "filename": chunk.name,
                        "start_time": annotation.start_time,
                        "end_time": annotation.end_time,
                        "duration": annotation.end_time - annotation.start_time,
                        "action_description": annotation.action_description,
                        "action_index": annotation.action_index,
                    }

                clips_info = await asyncio.gather(
                    *[
                        _format_clips_info(annotation)
                        for annotation in all_chunks_annotations
                    ]
                )
                dataset_info[video.id] = {
                    "original_file_name": original_file_name,
                    "total_duration": total_duration,
                    "processed_at": processed_at,
                    "clips": clips_info,
                }
            all_datasets_info[dataset.id] = {
                "actions": dataset.actions,
                "two_operator_mode": bool(dataset.two_operator_mode) if dataset.two_operator_mode is not None else False,
                "videos": dataset_info,
            }

        return all_datasets_info

    except Exception as e:
        app_logger.error(f"Error getting datasets info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting datasets info: {str(e)}"
        )


@app.get("/api/v1/videos/{video_id}/download")
async def download_video(
    request: Request, video_id: str = Path(..., description="Video ID")
):
    """Download video file with Range request support for video seeking"""

    try:
        app_logger.info(f"Downloading video with ID: {video_id}")

        metadata = await postgres_db.get_data(video_id, Video)
        if not metadata:
            app_logger.error(f"Video metadata not found in database for ID: {video_id}")
            raise HTTPException(status_code=404, detail="Video not found")

        app_logger.info(f"Video metadata found: dataset_id={metadata.dataset_id}, name={metadata.name}")

        # TODO: This should be replaced with MinIO
        file_path = os.path.join(const.VIDEO_ROOT, metadata.dataset_id, metadata.name)
        app_logger.info(f"Attempting to access video file at: {file_path}")
        if not os.path.exists(file_path):
            app_logger.error(f"Video file not found on disk: {file_path}")
            raise HTTPException(status_code=404, detail="Video file not found on disk")

        # Get file size
        file_size = metadata.file_size
        # Parse Range header if present
        range_header = request.headers.get("range")

        if range_header:
            # Parse range header: "bytes=start-end"
            try:
                range_match = range_header.replace("bytes=", "").split("-")
                start = int(range_match[0]) if range_match[0] else 0
                end = int(range_match[1]) if range_match[1] else file_size - 1

                # Ensure end doesn't exceed file size
                end = min(end, file_size - 1)

                # Calculate content length
                content_length = end - start + 1

                app_logger.info(
                    f"Range request: {start}-{end}/{file_size} (content_length: {content_length})"
                )

                # Read the requested range
                with open(file_path, "rb") as video_file:
                    video_file.seek(start)
                    data = video_file.read(content_length)

                # Return partial content with proper headers
                headers = {
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(content_length),
                    "Content-Type": metadata.mime_type,
                }

                return Response(
                    content=data,
                    status_code=206,  # Partial Content
                    headers=headers,
                )

            except (ValueError, IndexError) as e:
                app_logger.error(f"Invalid range header: {range_header}, error: {e}")
                # Fall back to full file if range parsing fails

        # No range header or invalid range - return full file
        app_logger.info(f"Serving full file: {file_path}")
        return FileResponse(
            path=file_path,
            filename=metadata.name,
            media_type=metadata.mime_type,
            headers={"Accept-Ranges": "bytes"},  # Indicate range support
        )
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error downloading video: {str(e)}")
        app_logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error downloading video: {str(e)}"
        )


@app.get("/api/v1/chunks/{chunk_id}")
async def get_chunk(chunk_id: str = Path(..., description="Chunk ID")) -> Dict:
    """Get chunk metadata by ID"""

    try:
        app_logger.info(f"Getting chunk metadata for ID: {chunk_id}")

        chunk_metadata = await postgres_db.get_data(chunk_id, Chunk)
        if not chunk_metadata:
            app_logger.error(f"Chunk not found: {chunk_id}")
            raise HTTPException(status_code=404, detail="Chunk not found")

        return {
            "id": chunk_metadata.id,
            "video_id": chunk_metadata.video_id,
            "filename": chunk_metadata.name,
            "action": chunk_metadata.action,
            "file_size": chunk_metadata.file_size,
            "mime_type": chunk_metadata.mime_type,
            "created_at": chunk_metadata.created_at.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error getting chunk metadata: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting chunk metadata: {str(e)}"
        )


@app.get("/api/v1/chunks/{chunk_id}/download")
async def download_chunk(
    request: Request, chunk_id: str = Path(..., description="Chunk ID")
):
    """Download a specific video chunk/clip file"""

    try:
        app_logger.info(f"Downloading chunk with ID: {chunk_id}")

        # Get chunk metadata
        chunk_metadata = await postgres_db.get_data(chunk_id, Chunk)
        if not chunk_metadata:
            app_logger.error(f"Chunk metadata not found in database for ID: {chunk_id}")
            raise HTTPException(status_code=404, detail="Chunk not found")

        app_logger.info(f"Chunk metadata found: video_id={chunk_metadata.video_id}, name={chunk_metadata.name}")

        # Get video metadata to find dataset_id
        video_metadata = await postgres_db.get_data(chunk_metadata.video_id, Video)
        if not video_metadata:
            app_logger.error(f"Video metadata not found for video ID: {chunk_metadata.video_id}")
            raise HTTPException(status_code=404, detail="Parent video not found")

        video_name_without_ext = os.path.splitext(video_metadata.name)[0]

        # Construct chunk file path
        chunk_file_path = os.path.join(
            const.VIDEO_ROOT,
            video_metadata.dataset_id,
            video_name_without_ext,
            chunk_metadata.name
        )

        app_logger.info(f"Attempting to access chunk file at: {chunk_file_path}")

        if not os.path.exists(chunk_file_path):
            app_logger.error(f"Chunk file not found on disk: {chunk_file_path}")
            raise HTTPException(status_code=404, detail="Chunk file not found on disk")

        # Return the chunk file
        return FileResponse(
            path=chunk_file_path,
            filename=chunk_metadata.name,
            media_type=chunk_metadata.mime_type or const.MIME_TYPE
        )

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error downloading chunk: {str(e)}")
        app_logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error downloading chunk: {str(e)}"
        )


@app.get("/api/v1/videos/{video_id}/download-all")
async def download_all_video_clips(video_id: str = Path(..., description="Video ID")):
    """Download all clips of a specific video as a ZIP file

    Args:
        video_id: The original video ID (not clip ID)

    Returns:
        FileResponse: ZIP file containing all clips for the video
    """
    app_logger.info(f"Download all clips request for video ID: {video_id}")

    # Get dataset id from video metadata
    video_metadata = await postgres_db.get_data(video_id, Video)
    if not video_metadata:
        app_logger.error(f"Video not found: {video_id}")
        raise HTTPException(status_code=404, detail="Video not found")

    dataset_id = video_metadata.dataset_id

    # Get all chunks for this video from database
    all_chunks = await postgres_db.list_data(Chunk, condition={"video_id": video_id})
    if not all_chunks:
        app_logger.error(f"No chunks found for video ID: {video_id}")
        raise HTTPException(status_code=404, detail="No chunks found for this video")

    # Create temporary ZIP file
    try:
        # Create temporary file for ZIP
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_zip:
            temp_zip_path = temp_zip.name

        app_logger.info(f"Creating ZIP file: {temp_zip_path}")

        with zipfile.ZipFile(temp_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for chunk in all_chunks:
                # video name is in format: {video_id}_{original_video_name}.mp4
                chunk_file_path = os.path.join(
                    const.VIDEO_ROOT,
                    dataset_id,
                    os.path.splitext(video_metadata.name)[0],
                    chunk.name,
                )
                if os.path.exists(chunk_file_path):
                    # Add file to ZIP with its original filename
                    arcname = chunk.name
                    app_logger.info(f"Adding to ZIP: {chunk.name}")
                    zipf.write(chunk_file_path, arcname)
                else:
                    app_logger.warning(
                        f"Clip file not found, skipping: {chunk_file_path}"
                    )

        # Check if ZIP file was created successfully
        if not os.path.exists(temp_zip_path):
            raise HTTPException(status_code=500, detail="Failed to create ZIP file")

        zip_file_size = os.path.getsize(temp_zip_path)
        app_logger.info(f"ZIP file created successfully, size: {zip_file_size} bytes")

        # Generate ZIP filename
        zip_filename = f"{os.path.splitext(video_metadata.name)[0]}_clips.zip"

        # Return the ZIP file. Delete the temp file via a BackgroundTask that
        # runs *after* the response is sent — not at process exit — so temp zips
        # don't accumulate and exhaust disk over the server's lifetime, and no
        # per-request atexit callback is leaked. (T18 / FSR-AVA-1)
        def cleanup_temp_file():  # pragma: no cover - runs in a background task after the response is sent
            """Remove the temporary ZIP once the response has been delivered."""
            try:
                if os.path.exists(temp_zip_path):
                    os.unlink(temp_zip_path)
                    app_logger.info(f"Cleaned up temporary ZIP file: {temp_zip_path}")
            except Exception as e:
                app_logger.warning(f"Failed to clean up temporary file: {e}")

        return FileResponse(
            path=temp_zip_path,
            filename=zip_filename,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
            background=BackgroundTask(cleanup_temp_file),
        )

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error creating ZIP file: {str(e)}")
        # Clean up temporary file if it exists
        if "temp_zip_path" in locals() and os.path.exists(temp_zip_path):
            os.unlink(temp_zip_path)
        raise HTTPException(
            status_code=500, detail=f"Failed to create ZIP file: {str(e)}"
        )


@app.get("/api/v1/videos")
async def list_videos() -> List[VideoMetadata]:
    """Get all uploaded videos list"""

    try:
        app_logger.info("Getting all videos list")

        videos = await postgres_db.list_data(Video)

        result = [
            VideoMetadata(
                id=video.id,
                filename=video.name,
                file_path=os.path.join(const.VIDEO_ROOT, video.dataset_id, video.name),
                file_size=video.file_size,
                upload_time=video.created_at.isoformat(),
                mime_type=video.mime_type,
            )
            for video in videos
        ]

        return result

    except Exception as e:
        app_logger.error(f"Error listing videos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing videos: {str(e)}")


@app.delete("/api/v1/videos/clear-all-datasets")
async def clear_all_videos() -> ClearDatasetResponse:
    """Clear all videos (metadata and files) or videos for specific data_id

    Args:
        data_id: Optional data_id to clear specific dataset. If None, clears all.
    """

    try:
        # Clear all videos (existing functionality)
        app_logger.info("Starting to clear all videos")

        # Get video directory
        videos_dir = const.VIDEO_ROOT

        db_deleted_count = 0
        dataset_deleted_count = 0

        # Get all dataset ids
        dataset_ids = [dataset.id for dataset in await postgres_db.list_data(Dataset)]

        # Clear main videos directory
        if os.path.exists(videos_dir):
            for dataset_id in dataset_ids:
                dataset_dir = os.path.join(videos_dir, dataset_id)
                if os.path.exists(dataset_dir):
                    shutil.rmtree(dataset_dir)
                    app_logger.info(f"Deleted dataset directory: {dataset_dir}")

                    dataset_deleted_count += 1
                else:
                    app_logger.warning(f"Dataset directory not found: {dataset_dir}")

        # Delete database records
        db_deleted_count = await postgres_db.delete_all_data(Dataset)

        app_logger.info(
            f"Cleared {db_deleted_count} database records and {dataset_deleted_count} datasets"
        )

        return ClearDatasetResponse(
            message="Successfully cleared all videos",
            data_id=",".join(dataset_ids),
            deleted_count=db_deleted_count,
            files_deleted=dataset_deleted_count,
        )
    except Exception as e:
        app_logger.error(f"Error clearing videos: {str(e)}")
        app_logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Error clearing all datasets",
        )


@app.delete("/api/v1/videos/clear-dataset/{data_id}")
async def clear_dataset(
    data_id: str = Path(..., description="Data ID to clear"),
) -> ClearDatasetResponse:
    """Clear all videos and files for a specific dataset (data_id)

    Args:
        data_id: The data_id of the dataset to clear
    """
    try:
        app_logger.info(f"Starting to clear videos for data_id: {data_id}")

        try:
            data_dir = safe_dataset_path(const.VIDEO_ROOT, data_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        dataset_deleted_count = 0

        # Clear specific data_id directory
        if os.path.exists(data_dir):
            try:
                # Count files before deletion
                # TODO: This should be replaced with MinIO

                shutil.rmtree(data_dir)
                app_logger.info(f"Deleted data directory: {data_dir}")

                dataset_deleted_count += 1
            except Exception as e:
                app_logger.warning(
                    f"Failed to delete data directory {data_dir}: {str(e)}"
                )

        # Delete database records for this data_id
        db_deleted_count = await postgres_db.delete_data(data_id, Dataset)

        app_logger.info(
            f"Cleared dataset {data_id}: {db_deleted_count} records, {dataset_deleted_count} datasets"
        )

        return ClearDatasetResponse(
            message=f"Successfully cleared dataset {data_id}",
            data_id=data_id,
            deleted_count=db_deleted_count,
            files_deleted=dataset_deleted_count,
        )
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error clearing videos: {str(e)}")
        app_logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing dataset {data_id}",
        )


@app.post("/api/v1/videos/{video_id}/split")
async def split_video(
    video_id: str = Path(..., description="Video ID"),
    request_body: Dict = Body(..., description="Request body containing timestamps"),
):
    """Split video into multiple segments based on timestamps

    Args:
        video_id: The video ID to split
        request_body: Request body with format {
            "timestamps": [{"start": float, "end": float, "actionIndex": int, "actionDescription": str}, ...],
            "twoOperatorMode": bool (optional, default False)
        }

    Returns:
        Dict: Dictionary containing split results
    """
    # TODO: Fix the request_body. Currently, there's no way for user to know the format of the request_body.
    # TODO: Add a way for user to know the format of the request_body.

    app_logger.info(f"Splitting video based on timestamps, ID: {video_id}")
    app_logger.info(f"Request body received: {request_body}")

    # Extract timestamps from request body
    if not request_body or not isinstance(request_body, dict):
        app_logger.error(f"Invalid request body format: {request_body}")
        raise HTTPException(status_code=400, detail="Invalid request body format")

    timestamps = request_body.get("timestamps")
    # Per-request override; if not provided, fall back to dataset-level setting
    two_operator_mode = request_body.get("twoOperatorMode", None)

    if two_operator_mode is None and current_data_id:
        try:
            ds = await postgres_db.get_data(current_data_id, Dataset)
            two_operator_mode = bool(ds.two_operator_mode) if ds and ds.two_operator_mode else False
        except Exception:
            two_operator_mode = False
    elif two_operator_mode is None:
        two_operator_mode = False

    app_logger.info(f"Two-operator mode: {two_operator_mode}")

    if not timestamps or not isinstance(timestamps, list) or len(timestamps) == 0:
        app_logger.error(f"Invalid timestamp data: {timestamps}")
        raise HTTPException(status_code=400, detail="Invalid timestamp data")

    # Get video metadata
    video_metadata = await postgres_db.get_data(video_id, Video)

    if not video_metadata:
        app_logger.error(f"Could not find video with ID {video_id}")
        raise HTTPException(
            status_code=404, detail=f"Could not find video with ID {video_id}"
        )
    app_logger.info(f"Video metadata: {video_metadata.to_dict()}")

    # Validate timestamp
    app_logger.info(f"Number of timestamps received: {len(timestamps)}")
    try:
        valid_timestamps = []
        for i, ts in enumerate(timestamps):
            app_logger.info(f"Timestamp {i}: {ts}")

            if not isinstance(ts, dict) or "start" not in ts or "end" not in ts:
                app_logger.warning(
                    f"Skipping invalid timestamp format: {ts} / {type(ts)}"
                )
                continue

            app_logger.info(f"  - start: {ts.get('start', 'MISSING')}")
            app_logger.info(f"  - end: {ts.get('end', 'MISSING')}")
            app_logger.info(f"  - actionIndex: {ts.get('actionIndex', 'MISSING')}")
            app_logger.info(
                f"  - actionDescription: {ts.get('actionDescription', 'MISSING')}"
            )

            try:
                start = float(ts["start"])
                end = float(ts["end"])

                if end <= start:
                    app_logger.warning(
                        f"Skipping invalid time range (end<=start): {ts}"
                    )
                    continue

                # Preserve actionIndex and actionDescription
                validated_ts = {
                    "start": start,
                    "end": end,
                    "actionIndex": ts.get("actionIndex", const.DEFAULT_ACTION_INDEX),
                    "actionDescription": ts.get(
                        "actionDescription", const.DEFAULT_ACTION_DESCRIPTION
                    ),
                }

                app_logger.info(f"Validated timestamp: {validated_ts}")
                valid_timestamps.append(validated_ts)
            except (ValueError, TypeError) as e:
                app_logger.warning(
                    f"Skipping invalid timestamp value: {ts}, error: {str(e)}"
                )
                continue

        if not valid_timestamps:
            app_logger.error("No valid timestamps")
            raise HTTPException(status_code=400, detail="No valid timestamps")

        app_logger.info(f"Final valid timestamps to process: {valid_timestamps}")

        # Clean up existing chunks and annotations for this video to allow re-annotation
        existing_chunks = await postgres_db.list_data(Chunk, condition={"video_id": video_id})
        if existing_chunks:
            app_logger.info(f"Found {len(existing_chunks)} existing chunks for video {video_id}. Deleting them for re-annotation.")

            # Get video name without extension for folder path
            video_name_without_ext = os.path.splitext(video_metadata.name)[0]
            clips_subfolder = os.path.join(
                const.VIDEO_ROOT,
                video_metadata.dataset_id,
                video_name_without_ext
            )

            # Remove all chunk files
            for chunk in existing_chunks:
                chunk_file_path = os.path.join(clips_subfolder, chunk.name)
                if os.path.exists(chunk_file_path):
                    try:
                        os.remove(chunk_file_path)
                        app_logger.info(f"Deleted old chunk file: {chunk_file_path}")
                    except Exception as e:
                        app_logger.warning(f"Failed to delete chunk file {chunk_file_path}: {e}")

                # Delete from database (Cascade should handle annotations)
                await postgres_db.delete_data(chunk.id, Chunk)

        # Use moviepy to split video
        result_clips = await _split_video_by_timestamps(
            video_metadata, valid_timestamps, two_operator_mode
        )

        # Post-split merge for two-operator mode
        merge_stats = None
        if two_operator_mode:
            merge_threshold = _load_merge_threshold()
            if merge_threshold > 0:
                # Get clips subfolder for merge operations
                video_name_without_ext = os.path.splitext(video_metadata.name)[0]
                clips_subfolder = os.path.join(
                    const.VIDEO_ROOT,
                    video_metadata.dataset_id,
                    video_name_without_ext
                )
                result_clips, merge_stats = _merge_small_chunks(
                    result_clips, clips_subfolder, threshold=merge_threshold
                )

                # Clean up DB records for merged-away chunks
                removed_ids = merge_stats.pop("_removed_chunk_ids", [])
                updated_chunks = merge_stats.pop("_updated_chunks", {})
                for removed_id in removed_ids:
                    try:
                        # Delete annotations linked to the merged-away chunk
                        annotations = await postgres_db.list_data(
                            Annotation, condition={"chunk_id": removed_id}
                        )
                        for ann in annotations:
                            await postgres_db.delete_data(ann.id, Annotation)
                        # Delete the chunk itself
                        await postgres_db.delete_data(removed_id, Chunk)
                        app_logger.info(f"Removed DB records for merged chunk {removed_id}")
                    except Exception as e:
                        app_logger.warning(f"Failed to remove DB records for chunk {removed_id}: {e}")
                for chunk_id, timestamps in updated_chunks.items():
                    try:
                        # Update timestamps for all annotations belonging to the target chunk
                        annotations = await postgres_db.list_data(
                            Annotation, condition={"chunk_id": chunk_id}
                        )
                        for ann in annotations:
                            await postgres_db.update_data(
                                ann.id,
                                Annotation,
                                start_time=timestamps["start_time"],
                                end_time=timestamps["end_time"],
                            )
                    except Exception as e:
                        app_logger.warning(f"Failed to update timestamps for chunk {chunk_id}: {e}")

        return {
            "status": "success",
            "clips": result_clips,
            "twoOperatorMode": two_operator_mode,
            "mergeStats": merge_stats,
        }
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error processing video split request: {str(e)}")
        app_logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail="Video split failed"
        )


def _emit_segment(prev_time: float, time: float, active_actions: set, timestamps: List[Dict]) -> Dict:
    """Build a time segment dict from the active actions at a sweep-line boundary.

    Args:
        prev_time: Start of the interval
        time: End of the interval
        active_actions: Set of indices into *timestamps* that are currently active
        timestamps: The original timestamp list (used to look up action info)

    Returns:
        Dict with start, end, concurrent_actions list, and is_concurrent flag
    """
    concurrent_actions = []
    seen_action_indices = set()
    for action_idx in active_actions:
        ai = timestamps[action_idx]["actionIndex"]
        if ai not in seen_action_indices:
            seen_action_indices.add(ai)
            concurrent_actions.append({
                "actionIndex": ai,
                "actionDescription": timestamps[action_idx]["actionDescription"]
            })

    return {
        "start": prev_time,
        "end": time,
        "concurrent_actions": concurrent_actions,
        "is_concurrent": len(concurrent_actions) > 1
    }


def _find_concurrent_time_segments(timestamps: List[Dict]) -> List[Dict]:
    """Find time segments where actions overlap (concurrent actions).

    For two-operator scenarios, this groups overlapping actions into time segments
    with the set of concurrent actions for each segment.

    Args:
        timestamps: List of timestamp dicts with start, end, actionIndex, actionDescription

    Returns:
        List of time segments, each with start, end, and list of concurrent actions
    """
    if not timestamps:
        return []

    # Collect all time boundaries
    events = []
    for i, ts in enumerate(timestamps):
        events.append((ts["start"], "start", i))
        events.append((ts["end"], "end", i))

    # Sort by time (and by type so "end" comes before "start" at same time)
    events.sort(key=lambda x: (x[0], x[1] == "start"))

    segments = []
    active_actions = set()
    prev_time = None

    for time, event_type, idx in events:
        # Create a segment for the interval if there were active actions
        if prev_time is not None and prev_time < time and active_actions:
            segments.append(_emit_segment(prev_time, time, active_actions, timestamps))

        # Update active actions
        if event_type == "start":
            active_actions.add(idx)
        else:
            active_actions.discard(idx)

        prev_time = time

    return segments


def _load_action_descriptions(video_metadata: Video) -> Dict:
    """Load action descriptions from the saved actions.json file for a video's dataset.

    Args:
        video_metadata: Video metadata object with dataset_id

    Returns:
        Dict mapping action index (int) to action description string
    """
    actions_file_path = os.path.join(
        const.VIDEO_ROOT, video_metadata.dataset_id, "actions.json"
    )

    app_logger.info(f"Loading actions from: {actions_file_path}")
    with open(actions_file_path, "r", encoding="utf-8") as f:
        actions_data = json.load(f)

    action_descriptions = {
        idx: v for idx, v in enumerate(actions_data.get("actions", []))
    }

    app_logger.info(f"Loaded {len(action_descriptions)} action descriptions")
    return action_descriptions


def _validate_video_file(video_path: str) -> float:
    """Validate that a video file exists, is readable, and has valid properties.

    Args:
        video_path: Absolute path to the video file

    Returns:
        float: The video duration in seconds

    Raises:
        HTTPException: If the video file is missing, corrupt, or has invalid properties
    """
    app_logger.info(f"Validating source video file: {video_path}")
    if not os.path.exists(video_path):
        app_logger.error(f"Source video file does not exist: {video_path}")
        raise HTTPException(
            status_code=404,
            detail=f"Source video file does not exist: {os.path.basename(video_path)}",
        )

    # Enhanced video validation - check if it's a valid video file
    app_logger.info("Validating video file format and properties...")

    try:
        # Use moviepy to validate and get video properties
        clip = VideoFileClip(video_path)

        # Check if video has valid dimensions and duration
        if clip.w <= 0 or clip.h <= 0:
            app_logger.error("Invalid video dimensions")
            clip.close()
            raise HTTPException(
                status_code=400,
                detail="Invalid video dimensions",
            )

        # Get video duration
        video_duration = clip.duration

        if video_duration is None or video_duration <= 0:
            app_logger.error("Invalid video duration")
            clip.close()
            raise HTTPException(
                status_code=400,
                detail="Invalid video duration",
            )

        app_logger.info(f"Video validation successful. Duration: {video_duration}s, Resolution: {clip.w}x{clip.h}")

        # Close the clip to free resources
        clip.close()

    except Exception as e:
        app_logger.error(f"Failed to analyze video file: {str(e)}")
        app_logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail="Failed to analyze video file")

    return video_duration


async def _process_concurrent_segment(
    segment: Dict,
    video_path: str,
    video_name_without_ext: str,
    clips_subfolder: str,
    video_metadata: Video,
    video_duration: float,
    action_combo_count: Dict,
    timeline_order: int,
) -> Optional[Tuple[Dict, Dict]]:
    """Process a single concurrent (two-operator) time segment: extract clip, save DB records.

    Args:
        segment: Dict with start, end, concurrent_actions, is_concurrent
        video_path: Path to the source video
        video_name_without_ext: Original video name without extension
        clips_subfolder: Directory to write output clips
        video_metadata: Video ORM object
        video_duration: Duration of the source video in seconds
        action_combo_count: Mutable dict tracking repetition counts per action combo
        timeline_order: Current timeline order (1-based)

    Returns:
        Tuple of (clip_result_dict, annotation_entry_dict), or None if the segment was skipped
    """
    start_time = segment["start"]
    end_time = segment["end"]
    concurrent_actions = segment["concurrent_actions"]
    is_concurrent = segment["is_concurrent"]

    # Validate timestamps against video duration
    if video_duration > 0 and start_time >= video_duration:
        app_logger.warning(f"Start time {start_time}s exceeds video duration, skipping")
        return None
    if video_duration > 0 and end_time > video_duration:
        end_time = video_duration

    if end_time - start_time < 0.1:
        app_logger.warning(f"Segment too short ({end_time - start_time}s), skipping")
        return None

    # Generate filename based on action(s)
    action_indices = sorted([a["actionIndex"] for a in concurrent_actions])
    action_numbers = [str(idx + 1).zfill(2) for idx in action_indices]
    action_prefix = "-".join(action_numbers)

    # Track repetition for this action combination
    combo_key = tuple(action_indices)
    if combo_key not in action_combo_count:
        action_combo_count[combo_key] = 0
    action_combo_count[combo_key] += 1
    repetition_count = action_combo_count[combo_key]

    output_filename = f"{action_prefix}_{video_name_without_ext}_{repetition_count}_{timeline_order}.mp4"
    output_path = os.path.join(clips_subfolder, output_filename)

    app_logger.info(f"Processing concurrent segment: {start_time}s-{end_time}s, actions: {action_prefix}")

    try:
        # Load and extract subclip
        full_clip = VideoFileClip(video_path)
        subclip = full_clip.subclip(start_time, end_time)

        subclip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            preset='fast',
            ffmpeg_params=[
                '-crf', '23',
                '-profile:v', 'high',
                '-level:v', '4.0',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-avoid_negative_ts', 'make_zero'
            ],
            logger=None
        )

        subclip.close()
        full_clip.close()
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

    if not os.path.exists(output_path):
        return None

    file_size = os.path.getsize(output_path)
    if file_size < 10240:
        app_logger.error(f"Video split produced small file ({file_size} bytes)")
        os.remove(output_path)
        return None

    # Create chunk record
    chunk_id = str(uuid.uuid4())
    created_at = datetime.now()

    # For concurrent actions, use combined description
    action_descriptions_list = [a["actionDescription"] for a in concurrent_actions]
    combined_description = " AND ".join(action_descriptions_list) if is_concurrent else action_descriptions_list[0]

    await postgres_db.insert_data(
        Chunk,
        id=chunk_id,
        video_id=video_metadata.id,
        name=output_filename,
        action=combined_description,
        mime_type=const.MIME_TYPE,
        file_size=file_size,
        created_at=created_at,
        updated_at=created_at,
    )

    # Create annotation for each action in this segment
    for action_info in concurrent_actions:
        await postgres_db.insert_data(
            Annotation,
            id=str(uuid.uuid4()),
            video_id=video_metadata.id,
            chunk_id=chunk_id,
            start_time=start_time,
            end_time=end_time,
            action_index=action_info["actionIndex"],
            action_description=action_info["actionDescription"],
            created_at=created_at,
            updated_at=created_at,
        )

    clip_result = {
        "id": chunk_id,
        "filename": output_filename,
        "start_time": start_time,
        "end_time": end_time,
        "duration": end_time - start_time,
        "action_indices": action_indices,
        "action_descriptions": action_descriptions_list,
        "action_description": " AND ".join(action_descriptions_list),
        "action_index": action_indices[0] if len(action_indices) == 1 else None,
        "is_concurrent": is_concurrent,
        "timeline_order": timeline_order,
        "created_at": created_at,
    }

    annotation_entry = {
        "chunk": f"chunk #{timeline_order}",
        "actions": [idx + 1 for idx in action_indices],
        "descriptions": action_descriptions_list,
        "is_concurrent": is_concurrent,
        "start_timestamp": start_time,
        "end_timestamp": end_time,
        "chunk_name": output_filename,
    }

    return clip_result, annotation_entry


async def _process_single_segment(
    ts: Dict,
    video_path: str,
    video_name_without_ext: str,
    clips_subfolder: str,
    video_metadata: Video,
    video_duration: float,
    action_descriptions: Dict,
    action_repetition_count: Dict,
    timeline_order: int,
) -> Optional[Tuple[Dict, Dict]]:
    """Process a single sequential (single-operator) timestamp segment.

    Args:
        ts: Timestamp dict with start, end, actionIndex, actionDescription
        video_path: Path to the source video
        video_name_without_ext: Original video name without extension
        clips_subfolder: Directory to write output clips
        video_metadata: Video ORM object
        video_duration: Duration of the source video in seconds
        action_descriptions: Dict mapping action index to description string
        action_repetition_count: Mutable dict tracking repetition counts per action index
        timeline_order: Current timeline order (1-based)

    Returns:
        Tuple of (clip_result_dict, annotation_entry_dict), or None if the segment was skipped

    Raises:
        HTTPException: On unrecoverable errors (propagated to caller)
    """
    start_time = float(ts["start"])
    end_time = float(ts["end"])
    action_index = ts["actionIndex"]

    # Get action description from loaded actions.json or fallback to timestamp data
    action_description = action_descriptions.get(
        action_index, ts["actionDescription"]
    )
    app_logger.info(
        f"Using action description for index {action_index}: {action_description}"
    )

    # Validate timestamps against video duration
    if video_duration > 0 and start_time >= video_duration:
        app_logger.warning(
            f"Start time {start_time}s exceeds video duration {video_duration}s, skipping"
        )
        return None

    # Adjust end time if it exceeds video duration
    if video_duration > 0 and end_time > video_duration:
        app_logger.warning(
            f"End time {end_time}s exceeds video duration {video_duration}s, adjusting to {video_duration}s"
        )
        end_time = video_duration

    # Track repetition count for this actionIndex
    if action_index not in action_repetition_count:
        action_repetition_count[action_index] = 0
    action_repetition_count[action_index] += 1
    repetition_count = action_repetition_count[action_index]

    # Generate new filename format: 0x_original_video_y_z.mp4
    action_number = str(action_index + 1).zfill(
        2
    )  # Convert to 1-based and pad with leading zeros
    output_filename = f"{action_number}_{video_name_without_ext}_{repetition_count}_{timeline_order}.mp4"

    # Store clips in subfolder named after original video
    output_path = os.path.join(clips_subfolder, output_filename)

    # Use moviepy to extract video segment
    duration = end_time - start_time

    app_logger.info(
        f"Action info - Index: {action_index} (1-based: {action_index + 1}), Description: {action_description}"
    )
    app_logger.info(
        f"Filename components - ActionNumber: {action_number}, Original: {video_name_without_ext}, "
        f"Repetition: {repetition_count}, Timeline: {timeline_order}"
    )

    chunk_id = str(uuid.uuid4())

    try:
        # Load the video clip
        full_clip = VideoFileClip(video_path)

        # Extract the subclip
        subclip = full_clip.subclip(start_time, end_time)

        # Write the subclip with optimized settings
        subclip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            preset='fast',  # Fast preset for better performance
            ffmpeg_params=[
                '-crf', '23',  # Good quality setting
                '-profile:v', 'high',
                '-level:v', '4.0',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',  # Optimize for web playback
                '-avoid_negative_ts', 'make_zero'  # Handle negative timestamps
            ],
            logger=None  # Suppress verbose output
        )

        # Clean up clips to free memory
        subclip.close()
        full_clip.close()

        # Verify output file exists and has reasonable size
        if not os.path.exists(output_path):
            raise FileNotFoundError("Output file was not created")

        file_size = os.path.getsize(output_path)

        # Enhanced file size validation - reject files smaller than 10KB
        if file_size < 10240:  # 10KB threshold
            app_logger.error(
                f"Video split produced small file ({file_size} bytes): {output_path}"
            )
            os.remove(output_path)
            raise ValueError("Video split produced file that is too small")

        # Additional validation: try to open the clip to ensure it's playable
        test_clip = VideoFileClip(output_path)
        test_duration = test_clip.duration
        test_clip.close()

        if test_duration is None or test_duration <= 0:
            app_logger.error("Split video has invalid duration")
            os.remove(output_path)
            raise ValueError("Split video has invalid duration")

        app_logger.info(
            f"Video split successful: {output_path}, size: {file_size} bytes, duration: {test_duration}s"
        )

    except Exception as e:
        app_logger.error(f"Error processing video segment: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)

        # prune database records
        await postgres_db.delete_data(chunk_id, Chunk)
        await postgres_db.delete_data(chunk_id, Annotation)
        raise HTTPException(status_code=500, detail=f"Video split failed: {str(e)}")

    # Create segment metadata and save to DB
    created_at = datetime.now()

    # save chunk to database
    await postgres_db.insert_data(
        Chunk,
        id=chunk_id,
        video_id=video_metadata.id,
        name=output_filename,
        action=action_description,
        mime_type=const.MIME_TYPE,
        file_size=file_size,
        created_at=created_at,
        updated_at=created_at,
    )

    # save annotation to database
    await postgres_db.insert_data(
        Annotation,
        id=str(uuid.uuid4()),
        video_id=video_metadata.id,
        chunk_id=chunk_id,
        start_time=start_time,
        end_time=end_time,
        action_index=action_index,
        action_description=action_description,
        created_at=created_at,
        updated_at=created_at,
    )

    clip_result = {
        "id": chunk_id,
        "filename": output_filename,
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "action_index": action_index,
        "action_number": action_index + 1,
        "repetition_count": repetition_count,
        "timeline_order": timeline_order,
        "action_description": action_description,
        "created_at": created_at,
    }

    annotation_entry = {
        "chunk": f"chunk #{timeline_order}",
        "idx": repetition_count,
        "action": action_index + 1,  # 1-based action index
        "description": f"{action_description}",
        "start_timestamp": start_time,
        "end_timestamp": end_time,
        "chunk_name": output_filename,
    }
    app_logger.info(f"Added annotation entry: {annotation_entry}")

    return clip_result, annotation_entry


async def _split_video_by_timestamps(
    video_metadata: Video, timestamps: List[Dict], two_operator_mode: bool = False
) -> List[Dict]:
    """Use moviepy to split video based on timestamps

    Args:
        video_metadata: Video metadata object
        timestamps: List of timestamp dicts
        two_operator_mode: If True, allows overlapping timestamps for concurrent actions

    Returns:
        List[Dict]: List of split results, each containing segment information
    """

    # Load action descriptions from saved actions.json file
    action_descriptions = _load_action_descriptions(video_metadata)

    # Validate video file
    video_path = os.path.join(
        const.VIDEO_ROOT, video_metadata.dataset_id, video_metadata.name
    )
    video_duration = _validate_video_file(video_path)

    app_logger.info(f"Starting video split, {len(timestamps)} time segments, two_operator_mode={two_operator_mode}")
    result_clips = []
    annotation_data = []  # Store annotation data for JSON file

    # Extract original filename from video path to create subfolder
    video_filename = os.path.basename(video_path)
    video_name_without_ext = os.path.splitext(video_filename)[0]

    # Create subfolder for this video's clips
    clips_subfolder = os.path.join(
        os.path.dirname(video_path), video_name_without_ext
    )
    app_logger.info(f"Creating clips subfolder: {clips_subfolder}")
    create_dir(clips_subfolder)

    # TWO-OPERATOR MODE: Handle concurrent/overlapping actions
    if two_operator_mode:
        app_logger.info("Processing in two-operator mode (concurrent actions allowed)")

        # Find concurrent time segments
        concurrent_segments = _find_concurrent_time_segments(timestamps)
        app_logger.info(f"Found {len(concurrent_segments)} time segments with concurrent action analysis")

        timeline_order = 0
        action_combo_count = {}  # Track repetition for action combinations

        for i, segment in enumerate(concurrent_segments):
            timeline_order += 1
            try:
                pair = await _process_concurrent_segment(
                    segment, video_path, video_name_without_ext, clips_subfolder,
                    video_metadata, video_duration, action_combo_count, timeline_order,
                )
            except Exception as e:
                app_logger.error(f"Error processing concurrent segment {i}: {str(e)}")
                continue

            if pair is None:
                timeline_order -= 1  # revert increment for skipped segments
                continue

            clip_result, annotation_entry = pair
            result_clips.append(clip_result)
            annotation_data.append(annotation_entry)

        # Save annotation JSON and return
        if result_clips:
            annotation_filename = f"{video_name_without_ext}_annotation.json"
            annotation_file_path = os.path.join(clips_subfolder, annotation_filename)

            def _write_annotation():
                with open(annotation_file_path, "w", encoding="utf-8") as f:
                    json.dump(annotation_data, f, indent=2, ensure_ascii=False)

            await asyncio.to_thread(_write_annotation)
            app_logger.info(f"Saved concurrent annotation file with {len(annotation_data)} entries")

        return result_clips

    # SINGLE-OPERATOR MODE: Original sequential processing
    action_repetition_count = {}  # Track how many times each actionIndex appears
    timeline_order = 0  # Track the overall timeline order (z component)

    for i, ts in enumerate(timestamps):
        app_logger.info(f"Processing time segment {i + 1}/{len(timestamps)}: {ts}")

        timeline_order += 1

        pair = await _process_single_segment(
            ts, video_path, video_name_without_ext, clips_subfolder,
            video_metadata, video_duration, action_descriptions,
            action_repetition_count, timeline_order,
        )

        if pair is None:
            timeline_order -= 1  # revert increment for skipped segments
            continue

        clip_result, annotation_entry = pair
        result_clips.append(clip_result)
        annotation_data.append(annotation_entry)

    if not result_clips:
        app_logger.warning("No successful video segments generated")
        raise HTTPException(
            status_code=500, detail="Video split failed, no segments generated"
        )

    # Save annotation JSON file
    try:
        annotation_filename = f"{video_name_without_ext}_annotation.json"
        annotation_file_path = os.path.join(clips_subfolder, annotation_filename)

        app_logger.info(f"Saving annotation file: {annotation_file_path}")
        app_logger.info(f"Annotation data: {annotation_data}")

        def _write_single_op_annotation():
            with open(annotation_file_path, "w", encoding="utf-8") as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)

        await asyncio.to_thread(_write_single_op_annotation)

        app_logger.info(
            f"Successfully saved annotation file with {len(annotation_data)} entries"
        )

    except Exception as e:
        app_logger.error(f"Error saving annotation file: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error saving annotation file: {str(e)}"
        )

    app_logger.info(f"Video split completed, {len(result_clips)} segments generated")
    return result_clips


def _is_small_relative_to_neighbors(chunk_idx: int, chunks: List[Dict],
                                    threshold: float = 0.20):
    """Check if a chunk is small (< threshold) relative to adjacent neighbors with shared actions.

    Returns:
        Tuple of (is_small, target_index, direction)
    """
    current_chunk = chunks[chunk_idx]
    current_duration = current_chunk["end_timestamp"] - current_chunk["start_timestamp"]
    current_actions = set(current_chunk["actions"])

    candidates = []

    # Check previous neighbor
    if chunk_idx > 0:
        prev_chunk = chunks[chunk_idx - 1]
        prev_actions = set(prev_chunk["actions"])
        if current_actions & prev_actions:
            prev_duration = prev_chunk["end_timestamp"] - prev_chunk["start_timestamp"]
            if current_duration < (threshold * prev_duration):
                candidates.append((chunk_idx - 1, prev_duration, "prev"))

    # Check next neighbor
    if chunk_idx < len(chunks) - 1:
        next_chunk = chunks[chunk_idx + 1]
        next_actions = set(next_chunk["actions"])
        if current_actions & next_actions:
            next_duration = next_chunk["end_timestamp"] - next_chunk["start_timestamp"]
            if current_duration < (threshold * next_duration):
                candidates.append((chunk_idx + 1, next_duration, "next"))

    if not candidates:
        return False, -1, ""

    # Choose the SMALLER neighbor
    target_idx, _, direction = min(candidates, key=lambda x: x[1])
    return True, target_idx, direction


def _clips_to_merge_format(chunks_data: List[Dict]) -> List[Dict]:
    """Convert result_clips format to the annotation-style format used by merge logic."""
    return [
        {
            "chunk": clip.get("chunk", f"chunk #{clip.get('timeline_order', 0)}"),
            "actions": clip.get("action_indices", []),
            "descriptions": clip.get("action_descriptions", []),
            "is_concurrent": clip.get("is_concurrent", False),
            "start_timestamp": clip["start_time"],
            "end_timestamp": clip["end_time"],
            "chunk_name": clip["filename"],
            "_clip_data": clip,
        }
        for clip in chunks_data
    ]


def _concatenate_videos(small_video: str, target_video: str,
                        temp_path: str, direction: str) -> bool:
    """Concatenate two video files and write to temp_path. Returns True on success."""
    if direction == "prev":
        clips = [VideoFileClip(target_video), VideoFileClip(small_video)]
    else:
        clips = [VideoFileClip(small_video), VideoFileClip(target_video)]

    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(
        temp_path, codec='libx264', audio_codec='aac', preset='fast',
        logger=None, verbose=False, audio=True,
    )

    for c in clips:
        c.close()
    final_clip.close()

    return os.path.exists(temp_path)


def _apply_merge(small_chunk, target_chunk, small_video, target_video,
                 temp_merged, video_folder, removed_ids, updated_chunks):
    """Update timestamps, swap files, and track DB changes after a successful merge."""
    target_chunk["start_timestamp"] = min(
        target_chunk["start_timestamp"], small_chunk["start_timestamp"]
    )
    target_chunk["end_timestamp"] = max(
        target_chunk["end_timestamp"], small_chunk["end_timestamp"]
    )

    os.remove(target_video)
    os.remove(small_video)
    os.rename(temp_merged, os.path.join(video_folder, target_chunk["chunk_name"]))

    small_clip_data = small_chunk.get("_clip_data", {})
    if small_clip_data.get("id"):
        removed_ids.append(small_clip_data["id"])
    target_clip_data = target_chunk.get("_clip_data", {})
    if target_clip_data.get("id"):
        updated_chunks[target_clip_data["id"]] = {
            "start_time": target_chunk["start_timestamp"],
            "end_time": target_chunk["end_timestamp"],
        }


def _try_merge_one(i, chunks, video_folder, threshold, removed_ids, updated_chunks):
    """Attempt to merge chunk at index *i* into a neighbor. Returns True if merged."""
    is_small, target_idx, direction = _is_small_relative_to_neighbors(i, chunks, threshold)
    if not is_small:
        return False

    small_chunk = chunks[i]
    target_chunk = chunks[target_idx]
    app_logger.info(f"Merging chunk #{i+1} into #{target_idx+1} ({direction})")

    small_video = os.path.join(video_folder, small_chunk["chunk_name"])
    target_video = os.path.join(video_folder, target_chunk["chunk_name"])
    temp_merged = os.path.join(video_folder, f"temp_merged_{i}.mp4")

    if not (os.path.exists(small_video) and os.path.exists(target_video)):
        return False

    try:
        if not _concatenate_videos(small_video, target_video, temp_merged, direction):
            app_logger.warning(f"Temp merged file not created for chunk #{i+1}")
            return False

        _apply_merge(small_chunk, target_chunk, small_video, target_video,
                     temp_merged, video_folder, removed_ids, updated_chunks)
        chunks.pop(i)
        app_logger.info(f"Merged chunk #{i+1} successfully")
        return True
    except Exception as e:
        app_logger.error(f"Video merge failed for chunk #{i+1}: {e}")
        if os.path.exists(temp_merged):
            os.remove(temp_merged)
        return False


def _merge_small_chunks(chunks_data: List[Dict], video_folder: str,
                        threshold: float = 0.20):
    """Merge small chunks into their smaller qualifying neighbor using moviepy.

    Returns:
        (merged_chunks, merge_stats) where merge_stats is a dict with counts.
    """
    chunks = _clips_to_merge_format(chunks_data)
    original_count = len(chunks)
    merge_count = 0
    removed_chunk_ids = []
    updated_chunks = {}

    i = len(chunks) - 1
    while i >= 0:
        if _try_merge_one(i, chunks, video_folder, threshold, removed_chunk_ids, updated_chunks):
            merge_count += 1
        i -= 1

    # Renumber and convert back to result_clips format
    merged_clips = []
    for idx, chunk in enumerate(chunks, 1):
        chunk["chunk"] = f"chunk #{idx}"
        clip_data = chunk.get("_clip_data", {})
        clip_data["start_time"] = chunk["start_timestamp"]
        clip_data["end_time"] = chunk["end_timestamp"]
        clip_data["duration"] = chunk["end_timestamp"] - chunk["start_timestamp"]
        clip_data["timeline_order"] = idx
        merged_clips.append(clip_data)

    merge_stats = {
        "original_count": original_count,
        "final_count": len(merged_clips),
        "merged_count": merge_count,
        "_removed_chunk_ids": removed_chunk_ids,
        "_updated_chunks": updated_chunks,
    }

    app_logger.info(f"Merge complete: {original_count} -> {len(merged_clips)} chunks ({merge_count} merged)")
    return merged_clips, merge_stats


@app.get("/health/live")
async def health_live():
    """Liveness probe"""
    return {"status": "alive"}


@app.get("/health/ready")
async def health_ready():
    """Readiness probe"""
    return {"status": "ready"}


def get_openapi_schema(file_dump=True, pprint=True):
    """Get OpenAPI schema"""
    from fastapi.openapi.utils import get_openapi

    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="annotation_backend",
        version="1.0.0",
        description="Video annotation backend service",
        routes=app.routes,
    )

    app.openapi_schema = openapi_schema

    if file_dump:
        with open("openapi.json", "w") as f:
            json.dump(openapi_schema, f, indent=2)

    if pprint:
        import pprint

        pprint.pprint(openapi_schema)

    return openapi_schema
