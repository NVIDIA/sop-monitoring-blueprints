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


import os
import shutil
import traceback
from moviepy.editor import VideoFileClip
import av

from utils.logger import app_logger


def safe_dataset_path(root: str, dataset_id: str) -> str:
    """Join ``dataset_id`` onto ``root`` and verify the result stays under ``root``.

    Raises ``ValueError`` if ``dataset_id`` contains path separators or traversal
    segments, or if the resolved path escapes ``root``. Callers should translate
    that into an HTTP 400.
    """
    if not dataset_id or "/" in dataset_id or "\\" in dataset_id or dataset_id in (".", ".."):
        raise ValueError(f"Invalid dataset id: {dataset_id!r}")
    root_real = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root_real, dataset_id))
    if candidate != root_real and not candidate.startswith(root_real + os.sep):
        raise ValueError(f"Dataset id escapes root: {dataset_id!r}")
    return candidate


def create_dir(dir_path: str) -> bool:
    """Create a directory if it does not exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return True
    return False


def clean_up_file(file_path: str) -> bool:
    """Clean up a file"""
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


def modify_directory_permission(dir_path: str, permission: int = 0o777) -> bool:
    """Modify the permission of a directory"""

    try:
        os.chmod(dir_path, permission)
        return True
    except Exception as e:
        app_logger.error(f"Cannot modify directory permissions: {str(e)}")
        app_logger.error(traceback.format_exc())
        return False


async def convert_to_h264(input_path: str, output_path: str) -> str:
    """Convert video to H264 encoding for better compatibility

    Args:
        input_path: Path to input video file
        output_path: Path for output H264 encoded video file

    Returns:
        str: Path to the converted video file

    Raises:
        Exception: If conversion fails
    """

    try:
        # First, check if the video is already H264 encoded
        is_h264 = await check_if_h264_encoded(input_path)

        if is_h264:
            app_logger.info("Video is already H264 encoded, copying to final location")
            # Just copy the file to final location
            shutil.copy2(input_path, output_path)
            return output_path

        app_logger.info(
            f"Converting video to H264 encoding: {input_path} -> {output_path}"
        )

        # Check if moviepy dependencies are available
        # try:
        #     moviepy_check()
        # except Exception as e:
        #     app_logger.error(f"MoviePy dependency check failed: {str(e)}")
        #     raise Exception("MoviePy dependencies are not properly installed")

        # Load video using moviepy
        clip = VideoFileClip(input_path)

        # Write video with H264 codec
        # MoviePy uses similar parameters as ffmpeg
        clip.write_videofile(
            output_path,
            codec='libx264',  # H264 codec
            audio_codec='aac',  # AAC audio codec
            preset='medium',  # Encoding preset (balance between speed and quality)
            ffmpeg_params=[
                '-profile:v', 'high',
                '-level:v', '4.0',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart'  # Optimize for web playback
            ],
            logger=None  # Suppress moviepy's verbose output
        )

        # Clean up clip to free memory
        clip.close()

        # Verify output file exists and has reasonable size
        if not os.path.exists(output_path):
            raise FileNotFoundError("Conversion completed but output file not found")

        output_size = os.path.getsize(output_path)
        if output_size < 1024:  # Less than 1KB
            raise ValueError(f"Converted file too small ({output_size} bytes)")

        app_logger.info(
            f"Video conversion successful. Output size: {output_size} bytes"
        )
        return output_path
    except Exception as e:
        app_logger.error(f"Error during video conversion: {str(e)}")
        # Clean up output file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
        raise e


async def check_if_h264_encoded(video_path: str) -> bool:
    """Check if video is already H264 encoded

    Args:
        video_path: Path to video file

    Returns:
        bool: True if video is H264 encoded, False otherwise
    """
    container = None
    try:
        # Use av to check video codec
        with av.open(video_path) as container:
            # Check video streams for H264 codec
            is_h264 = False
            for stream in container.streams.video:
                video_codec = stream.codec.name.lower()
                # Check if codec is H264 (h264, avc are common names for H264)
                if video_codec in ['h264', 'avc', 'libx264']:
                    is_h264 = True
                    app_logger.info(f"Video codec detected: {video_codec}, is H264: {is_h264}")
                    break
                else:
                    app_logger.info(f"Video codec detected: {video_codec}, is H264: {is_h264}")

            if not container.streams.video:
                app_logger.warning("No video streams found in the file")
                is_h264 = False

            return is_h264

    except Exception as e:
        app_logger.warning(f"Error checking video codec: {str(e)}")
        return False
