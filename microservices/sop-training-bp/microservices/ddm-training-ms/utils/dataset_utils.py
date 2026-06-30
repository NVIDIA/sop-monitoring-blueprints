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

import json
import logging
import os
from glob import glob
from pathlib import Path
from typing import Dict, List


logger = logging.getLogger(__name__)


# TODO: Update this function based on DDM's dataset format
# - If DDM uses JSON files like CR, this function might work as-is
# - If DDM uses a different format (e.g., .txt, .csv, .parquet), update the file extension
# - If DDM has a different directory structure, update the logic accordingly
def get_all_json_paths(dataset_path: str) -> List[str]:
    """
    Get all JSON paths in a dataset folder.
    """
    json_paths = []
    dataset_path = Path(dataset_path)

    if dataset_path.exists():
        # Find all JSON files recursively
        for json_file in dataset_path.rglob("*.json"):
            json_paths.append(str(json_file))

    return json_paths


def generate_ddm_annotation(dataset_path: str, output_filename: str) -> str:
    """
    Generate DDM training annotation file from video annotations.
    
    This function:
    1. Finds all MP4 videos in the dataset path
    2. Loads the corresponding annotation JSON for each video
    3. Combines all annotations into a single JSON file for DDM training
    
    Args:
        dataset_path: Path to the dataset directory containing videos and annotations
        output_filename: Name of the output annotation file (e.g., "ddm_train_annotation.json")
        
    Returns:
        str: Path to the generated annotation file
        
    Raises:
        FileNotFoundError: If dataset_path doesn't exist or required annotation files are missing
        ValueError: If no valid video/annotation pairs are found
        json.JSONDecodeError: If annotation files contain invalid JSON
    """
    logger.info(f"Generating DDM annotation from dataset: {dataset_path}")
    
    # Validate dataset path exists
    if not os.path.exists(dataset_path):
        error_msg = f"Dataset path does not exist: {dataset_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Find all video files
    video_paths = glob(os.path.join(dataset_path, "*.mp4"))
    
    if not video_paths:
        error_msg = f"No MP4 videos found in {dataset_path}"
        logger.warning(error_msg)
        # Return empty annotation instead of raising error (might be valid scenario)
    
    logger.info(f"Found {len(video_paths)} video(s) in dataset")
    logger.info(" ".join(video_paths))
    
    # Build annotation dictionary
    ddm_annotation_dict = {}
    missing_annotations = []
    
    for video_path in video_paths:
        video_id = os.path.basename(video_path).split(".")[0]
        
        # Check if this video has an annotation directory
        annotation_file = os.path.join(dataset_path, video_id, f"{video_id}_annotation.json")
        
        if not os.path.exists(annotation_file):
            logger.warning(f"Annotation file not found for video {video_id}: {annotation_file}")
            missing_annotations.append(video_id)
            continue
        
        try:
            # Load annotation
            with open(annotation_file, "r", encoding="utf-8") as f:
                action_chunks_result = json.load(f)
            
            ddm_annotation_dict[str(video_id)] = action_chunks_result
            logger.debug(f"Loaded annotation for video: {video_id}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in annotation file {annotation_file}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading annotation for {video_id}: {str(e)}")
            raise
    
    # Check if we have any valid annotations
    if not ddm_annotation_dict:
        error_msg = f"No valid video/annotation pairs found in {dataset_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if missing_annotations:
        logger.warning(f"Missing annotations for {len(missing_annotations)} video(s): {missing_annotations}")
    
    # Write combined annotation file
    output_path = os.path.join(dataset_path, output_filename)
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ddm_annotation_dict, f, indent=4)
        
        logger.info(f"Successfully generated DDM annotation file: {output_path}")
        logger.info(f"Total annotations: {len(ddm_annotation_dict)}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error writing annotation file to {output_path}: {str(e)}")
        raise

