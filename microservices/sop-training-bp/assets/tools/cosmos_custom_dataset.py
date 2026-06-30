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


import argparse
import ast
import copy
import re
import os
import json
import random
import pydantic
import toml
from pathlib import Path

from cosmos_reason2_utils.text import create_conversation
from cosmos_reason2_utils.vision import PIXELS_PER_TOKEN, VisionConfig
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config
from cosmos_rl.utils.logging import logger
from torch.utils.data import Dataset


class CustomConfig(pydantic.BaseModel):
    vision: VisionConfig = pydantic.Field(default=VisionConfig(fps=8, max_pixels=81920, max_frames=8))
    """Vision processor config."""

    system_prompt: str = pydantic.Field(
        default="Answer the questions.",
        description="System prompt to add to conversations"
    )
    """System prompt to add to conversations"""


class CosmosSFTDataset(Dataset):
    def __init__(self, config: Config, custom_config: CustomConfig):
        self.config = config
        self.custom_config = custom_config
        self.vision_kwargs = custom_config.vision.model_dump(exclude_none=True)
        self.system_prompt = custom_config.system_prompt

        # load multiple json files
        self.datasets = []
        self.mm_files_paths = {}

        for _, (annotation_path, split_name) in enumerate(zip(config.train.train_policy.dataset.name, config.train.train_policy.dataset.split)):

            # Load JSON data
            logger.info(f"Loading JSON dataset from {annotation_path}")
            annotations = json.load(open(annotation_path))

            # Get media path
            # Use directory of annotation file as media path
            media_path = os.path.dirname(annotation_path)

            # Build media files mapping for this dataset
            if os.path.exists(media_path):
                for root, _, files in os.walk(media_path):
                    for file in files:
                        if file.lower().endswith((".mp4", ".avi", ".mov", ".jpg", ".jpeg", ".png")): # Common video and image extensions
                            self.mm_files_paths[file] = os.path.join(root, file)

            # Store dataset info
            for item in annotations:
                item['_split'] = split_name
                item['_media_path'] = media_path
            logger.info(f"Loaded {len(annotations)} samples from {annotation_path} for split {split_name}")
            self.datasets.extend(annotations)

        # shuffle datasets
        random.shuffle(self.datasets)
        logger.info(f"Total samples loaded: {len(self.datasets)}")
        logger.info(f"Total video files found: {len(self.mm_files_paths)}")


    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        """
        Return a tuple of (prompt, reference answer)
        """
        payload = self.datasets[idx]
        conversations = copy.deepcopy(payload["conversations"])

        user_prompt = conversations[0]["value"]
        response = conversations[1]["value"]

        if user_prompt is None:
            raise ValueError(f"No user prompt found in sample {idx}")
        if response is None:
            raise ValueError(f"No assistant response found in sample {idx}")

        # Handle images and videos
        images = payload.get("image", None) or payload.get("images", None)
        if images:
            images = [images] if isinstance(images, str) else images

            processed_images = []
            for img in images:
                image_filename = os.path.basename(img)
                processed_images.append(self.mm_files_paths[image_filename])
            images = processed_images

        videos = payload.get("video", None) or payload.get("videos", None)
        if videos:
            videos = [videos] if isinstance(videos, str) else videos

            processed_videos = []
            for vid in videos:
                video_filename = os.path.basename(vid)
                processed_videos.append(self.mm_files_paths[video_filename])
            videos = processed_videos

        # Remove image and video tags from user prompt
        user_prompt = re.sub(r"(\n)?</?(image|video)>(\n)?", "", user_prompt)

        conversations = create_conversation(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response=response,
            images=images,
            videos=videos,
            vision_kwargs=self.vision_kwargs,
        )

        return conversations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]

    # load config
    with open(args.config, "r") as f:
        config_kwargs = toml.load(f)

    config = Config.from_dict(config_kwargs)
    config.train.train_policy.dataset.name = ast.literal_eval(config.train.train_policy.dataset.name)

    # custom config
    custom_config = CustomConfig.model_validate(config_kwargs["custom"])

    # set total_pixels if not defined
    if custom_config.vision.total_pixels is None:
        custom_config.vision.total_pixels = int(
        config.policy.model_max_length * PIXELS_PER_TOKEN * 0.9
    )

    # Log
    role = os.environ.get("COSMOS_ROLE")
    is_controller = role == "Controller"
    if is_controller:
        output_dir = Path(config.train.output_dir).resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_kwargs = config.model_dump()
        config_kwargs["custom"] = custom_config.model_dump()
        config_path = output_dir / f"{config.logging.experiment_name}_config.toml"
        config_path.write_text(toml.dumps(config_kwargs))
        logger.info(f"Saved config to {config_path}")

    launch_worker(
        dataset=CosmosSFTDataset(config=config, custom_config=custom_config),
    )