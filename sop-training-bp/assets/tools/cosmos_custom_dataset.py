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

import argparse
import ast
import copy
import os

import pydantic
import toml
from cosmos_reason1_utils.vision import VisionConfig
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config
from datasets import load_dataset
from torch.utils.data import ConcatDataset, Dataset
from transformers import AutoTokenizer


class CustomConfig(pydantic.BaseModel):
    vision: VisionConfig = pydantic.Field(default=VisionConfig(fps=8, max_pixels=81920, max_frames=8))
    """Vision processor config."""


class CosmosSFTDataset(Dataset):
    def __init__(self, dataset: Dataset, custom_config: CustomConfig):
        self.dataset = dataset
        self.custom_config = custom_config
        self.vision_kwargs = custom_config.vision.model_dump(exclude_none=True)

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        """
        Called by launcher after being mounted
        """
        config.train.train_policy.dataset.name = ast.literal_eval(config.train.train_policy.dataset.name)

        self.config = config
        self.tokenizer = tokenizer

        if config.train.train_policy.dataset.split:
            if isinstance(config.train.train_policy.dataset.split, list):
                dataset_list = []
                for split_name in config.train.train_policy.dataset.split:
                    dataset_list.append(self.dataset[split_name])
                self.dataset = ConcatDataset(dataset_list)
            else:
                assert isinstance(config.train.train_policy.dataset.split, str)
                self.dataset = self.dataset[config.train.train_policy.dataset.split]

        # get multi-modal files paths
        video_clips_paths = [os.path.dirname(name) for name in config.train.train_policy.dataset.name]
        for video_clips_path in video_clips_paths:
            if not os.path.exists(video_clips_path):
                raise FileNotFoundError(
                    f"Dataset directory {video_clips_path} does not exist. Please check the dataset path."
                )
            mm_files_paths = {}
            for root, dirs, files in os.walk(video_clips_path):
                for file in files:
                    if file.endswith((".mp4", ".avi", ".mov")):  # Common video extensions
                        mm_files_paths[file] = os.path.join(root, file)
            if not hasattr(self, "mm_files_paths"):
                self.mm_files_paths = {}
            self.mm_files_paths.update(mm_files_paths)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        """
        Return a tuple of (prompt, reference answer)
        """
        payload = self.dataset[idx]
        conversations = copy.deepcopy(payload["conversations"])

        for conv in conversations:
            # Transform the conversation format to the format required by vllm
            # role/content <- from/value, user/assistant <- human/gpt
            # Transform conversation format: from/value -> role/content, human/gpt -> user/assistant
            if "from" in conv and "value" in conv:
                conv["role"] = conv.pop("from")
                conv["content"] = conv.pop("value")

                if conv["role"] == "human":
                    conv["role"] = "user"
                elif conv["role"] == "gpt":
                    conv["role"] = "assistant"

            if conv["role"] == "user":
                assert isinstance(conv["content"], str), "User message must be string"
                # Rewrite to support image/video tokens
                content = [
                    {
                        "type": "video",
                        "video": self.mm_files_paths[payload["video"].split("/")[-1]],
                        **self.vision_kwargs,
                    },
                    {
                        "type": "text",
                        "text": conv["content"],
                    },
                ]
                conv["content"] = content
        # add a new role: "system", with content: "Answer the questions."
        conversations.insert(0, {"role": "system", "content": "Answer the questions."})
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
    if isinstance(config.train.train_policy.dataset.name, list):
        data_files = {}
        for split, name in zip(config.train.train_policy.dataset.split, config.train.train_policy.dataset.name):
            print(f"Loading json dataset from {name}")
            data_files[split] = name
        dataset = load_dataset("json", data_files=data_files)
    else:
        # Download HF dataset only on launcher worker
        dataset = load_dataset(config.train.train_policy.dataset.name, config.train.train_policy.dataset.subset)

    launch_worker(
        dataset=CosmosSFTDataset(dataset=dataset, custom_config=custom_config),
    )
