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
from datetime import datetime
from typing import Dict, List
from moviepy.editor import VideoFileClip

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
from fastapi.responses import FileResponse

import utils.constant as const
from components.postgres_db import postgres_db
from utils.logger import app_logger
from utils.utils import (
    clean_up_file,
    convert_to_h264,
    create_dir,
)
from validations.data_models import (
    ActionsUploadResponse,
    ClearDatasetResponse,
    ResetActionsResponse,
    VideoMetadata,
    VideoUploadResponse,
)
from validations.db_models import Annotation, Chunk, Dataset, Video

# Global data_id to track current dataset version
current_data_id = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI app"""
    # Startup
    app_logger.info("Starting annotation backend service")

    # Ensure video directory exists
    create_dir(const.VIDEO_ROOT)

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
        # Generate unique video ID
        video_id = str(uuid.uuid4())

        # Ensure video directory exists
        videos_dir = os.path.join(const.VIDEO_ROOT, current_data_id)

        # Ensure the filename always ends with .mp4 for consistency
        if not safe_basename.lower().endswith(const.DEFAULT_VIDEO_EXTENSION):
            # If the file doesn't have .mp4 extension, add it
            name_without_ext = os.path.splitext(safe_basename)[0]
            safe_basename = f"{name_without_ext}" + const.DEFAULT_VIDEO_EXTENSION
            app_logger.info(f"Normalized filename to: {safe_basename}")

        # Create safe filename with UUID to prevent conflicts
        # Use only the basename to avoid path separator issues
        safe_filename = f"{video_id}{const.ID_NAME_SEPARATOR}{safe_basename}"
        temp_file_path = os.path.join(videos_dir, f"temp_{safe_filename}")
        final_file_path = os.path.join(videos_dir, safe_filename)

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

        # Save metadata to database
        await postgres_db.insert_data(
            Video,
            id=video_id,
            dataset_id=current_data_id,
            name=safe_filename,
            mime_type="video/mp4",
            file_size=final_file_size,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        app_logger.info(
            f"File '{safe_filename}' (original: '{original_filename}') successfully converted and saved to {converted_file_path} and recorded in database, ID: {video_id}"
        )

        return VideoUploadResponse(
            message="Video file has been successfully uploaded, converted to H264, and saved",
            file_id=video_id,
        )
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

        # Save actions to database
        await postgres_db.insert_data(
            Dataset,
            id=new_data_id,
            actions=actions_array,
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
    except json.JSONDecodeError as e:
        app_logger.error(f"Invalid JSON format: {str(e)}")
        app_logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON format: {traceback.format_exc()}"
        )
    except Exception as e:
        app_logger.error(f"Error uploading actions file: {str(e)}")
        app_logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload actions file: {traceback.format_exc()}",
        )


@app.post("/api/v1/actions/reset")
async def reset_actions() -> ResetActionsResponse:
    """Reset the current data_id to start a new dataset annotation

    Returns:
        ResetActionsResponse: Reset response with status
    """
    global current_data_id

    try:
        old_data_id = current_data_id
        current_data_id = None
        app_logger.info(f"Reset data_id from {old_data_id} to None")

        return ResetActionsResponse(
            status="success",
            message="Data ID reset successfully. Ready for new dataset annotation.",
            previous_data_id=old_data_id,
        )
    except Exception as e:
        app_logger.error(f"Error resetting data_id: {str(e)}")
        app_logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset data_id: {traceback.format_exc()}",
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
            all_datasets_info[dataset.id] = dataset_info

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
            media_type=chunk_metadata.mime_type or "video/mp4"
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

        # Return the ZIP file
        def cleanup_temp_file():
            """Clean up temporary file after response is sent"""
            try:
                if os.path.exists(temp_zip_path):
                    os.unlink(temp_zip_path)
                    app_logger.info(f"Cleaned up temporary ZIP file: {temp_zip_path}")
            except Exception as e:
                app_logger.warning(f"Failed to clean up temporary file: {e}")

        # Schedule cleanup (this will run after the response is sent)
        import atexit

        atexit.register(cleanup_temp_file)

        return FileResponse(
            path=temp_zip_path,
            filename=zip_filename,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
        )

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
            detail=f"Error clearing all datasets: {traceback.format_exc()}",
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

        # Get base videos directory
        data_dir = os.path.join(const.VIDEO_ROOT, data_id)

        dataset_deleted_count = 0

        # Clear specific data_id directory
        if os.path.exists(data_dir):
            try:
                # Count files before deletion
                # TODO: This should be replaced with MinIO
                dataset_deleted_count += 1

                shutil.rmtree(data_dir)
                app_logger.info(f"Deleted data directory: {data_dir}")
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
    except Exception as e:
        app_logger.error(f"Error clearing videos: {str(e)}")
        app_logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing dataset {data_id}: {traceback.format_exc()}",
        )


@app.post("/api/v1/videos/{video_id}/split")
async def split_video(
    video_id: str = Path(..., description="Video ID"),
    request_body: Dict = Body(..., description="Request body containing timestamps"),
):
    """Split video into multiple segments based on timestamps

    Args:
        video_id: The video ID to split
        request_body: Request body with format {"timestamps": [{"start": float, "end": float, "actionIndex": int, "actionDescription": str}, ...]}

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

        # Use moviepy to split video
        result_clips = await _split_video_by_timestamps(
            video_metadata, valid_timestamps
        )

        return {"status": "success", "clips": result_clips}
    except Exception as e:
        app_logger.error(f"Error processing video split request: {str(e)}")
        app_logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Video split failed: {traceback.format_exc()}"
        )


async def _split_video_by_timestamps(
    video_metadata: Video, timestamps: List[Dict]
) -> List[Dict]:
    """Use moviepy to split video based on timestamps

    Args:
        video_path: Video file path

    Returns:
        List[Dict]: List of split results, each containing segment information
    """

    # Load action descriptions from saved actions.json file
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

    # Validate video file
    video_path = os.path.join(
        const.VIDEO_ROOT, video_metadata.dataset_id, video_metadata.name
    )
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

    app_logger.info(f"Starting video split, {len(timestamps)} time segments")
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

    # Track repetition count for each actionIndex and timeline order
    action_repetition_count = {}  # Track how many times each actionIndex appears
    timeline_order = 0  # Track the overall timeline order (z component)

    for i, ts in enumerate(timestamps):
        app_logger.info(f"Processing time segment {i + 1}/{len(timestamps)}: {ts}")
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
            continue

        # Adjust end time if it exceeds video duration
        if video_duration > 0 and end_time > video_duration:
            app_logger.warning(
                f"End time {end_time}s exceeds video duration {video_duration}s, adjusting to {video_duration}s"
            )
            end_time = video_duration

        # Increment timeline order for each valid segment
        timeline_order += 1

        # Track repetition count for this actionIndex
        if action_index not in action_repetition_count:
            action_repetition_count[action_index] = 0
        action_repetition_count[action_index] += 1
        repetition_count = action_repetition_count[action_index]

        # Generate new filename format: 0x_original_video_y_z.mp4
        # 0x: actionIndex + 1, padded to 2 digits
        # original_video: original video name without extension
        # y: repetition count for this actionIndex
        # z: timeline order
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

        success = False
        file_size = 0

        try:
            # Load the video clip
            full_clip = VideoFileClip(video_path)

            # Extract the subclip
            subclip = full_clip.subclip(start_time, end_time)

            # Write the subclip with optimized settings
            # MoviePy will handle keyframe issues automatically, it has similar parameters as ffmpeg
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
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)

                # Enhanced file size validation - reject files smaller than 10KB
                if file_size < 10240:  # 10KB threshold
                    app_logger.error(
                        f"Video split produced small file ({file_size} bytes): {output_path}"
                    )
                    os.remove(output_path)
                    raise Exception("Video split produced file that is too small")

                # Additional validation: try to open the clip to ensure it's playable
                try:
                    test_clip = VideoFileClip(output_path)
                    test_duration = test_clip.duration
                    test_clip.close()

                    if test_duration is None or test_duration <= 0:
                        app_logger.error("Split video has invalid duration")
                        os.remove(output_path)
                        raise Exception("Split video has invalid duration")

                    app_logger.info(
                        f"Video split successful: {output_path}, size: {file_size} bytes, duration: {test_duration}s"
                    )
                    success = True
                except Exception as e:
                    app_logger.error(f"Split video validation failed: {str(e)}")
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    raise
            else:
                raise Exception("Output file was not created")

            if success:
                # Create segment metadata
                chunk_id = str(uuid.uuid4())
                created_at = datetime.now()

                # save chunk to database
                await postgres_db.insert_data(
                    Chunk,
                    id=chunk_id,
                    video_id=video_metadata.id,
                    name=output_filename,
                    action=action_description,
                    mime_type="video/mp4",
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

                # Add to result list
                result_clips.append(
                    {
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
                )

                # Add to annotation data for JSON file
                annotation_entry = {
                    "chunk": f"chunk #{timeline_order}",
                    "idx": repetition_count,
                    "action": action_index + 1,  # 1-based action index
                    "description": f"{action_description}",
                    "start_timestamp": start_time,
                    "end_timestamp": end_time,
                    "chunk_name": output_filename,
                }
                annotation_data.append(annotation_entry)
                app_logger.info(f"Added annotation entry: {annotation_entry}")
        except Exception as e:
            app_logger.error(f"Error processing video segment {i}: {str(e)}")
            if os.path.exists(output_path):
                os.remove(output_path)

            # prune database records
            await postgres_db.delete_data(chunk_id, Chunk)
            await postgres_db.delete_data(chunk_id, Annotation)
            raise HTTPException(status_code=500, detail=f"Video split failed: {str(e)}")

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

        with open(annotation_file_path, "w", encoding="utf-8") as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)

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
