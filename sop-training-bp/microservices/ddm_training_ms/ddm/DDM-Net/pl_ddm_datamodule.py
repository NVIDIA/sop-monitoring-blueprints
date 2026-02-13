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

"""DDM Data Module for Lightning 2.x"""

from typing import Optional
from torch.utils.data import DataLoader
import lightning as L
import torchvision.transforms as T
from datasets.ddm_dataset import DDMDataset
from datasets.ddm_val_dataset import DDMValStreamingDataset


class DDMDataModule(L.LightningDataModule):
    """
    Lightning 2.x DataModule for DDM (Dense Temporal Boundary Detection)

    Features:
    - Automatic distributed sampling (no manual DistributedSampler needed!)
    - Support for YAML configuration
    - Clean and simple implementation
    """

    def __init__(self, dataset_config):
        """
        Lightning DataModule Initialization

        Args:
            dataset_config (dict): Configuration for the dataset from YAML file
        """
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = dataset_config["batch_size"]
        self.num_workers = dataset_config["workers"]
        self.resolution = dataset_config["resolution"]
        self.dataset = dataset_config["dataset"]


    def setup(self, stage: Optional[str] = None):
        """
        Setup the dataset

        Args:
            stage (str): Stage of the dataset
        """
        if stage == "fit" or stage is None:
            transform = T.Compose([
                T.Resize(self.resolution),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            if self.dataset == "DDMDataset":
                self.train_dataset = DDMDataset(
                    mode=self.dataset_config["train_config"]["mode"],
                    anno_path=self.dataset_config["train_config"]["anno_path"],
                    data_root=self.dataset_config["train_config"]["data_root"],
                    resolution=self.resolution,
                    num_classes=self.dataset_config["num_classes"],
                    frames_per_side=self.dataset_config["frames_per_side"],
                    downsample=self.dataset_config["downsample"],
                    min_change_dur=self.dataset_config["min_change_dur"],
                    seed=self.dataset_config["seed"],
                    video_backend=self.dataset_config["video_backend"],
                    transform=transform,
                )
                self.val_dataset = DDMValStreamingDataset(
                    annotation_file=self.dataset_config["val_config"]["anno_path"],
                    video_root=self.dataset_config["val_config"]["data_root"],
                    frames_per_side=self.dataset_config["frames_per_side"],
                    downsample=self.dataset_config["downsample"],
                    temporal_stride=self.dataset_config["val_config"]["temporal_stride"],
                    min_change_dur=self.dataset_config["min_change_dur"],
                    chunk_duration=None,
                    resolution=self.resolution,
                    enable_load_balancing=True,
                    transform=transform,
                )
            else:
                raise NotImplementedError(
                    "Wrong dataset name %s (choose one from [DDMDataset,])"
                    % self.dataset
                )


    def train_dataloader(self):
        """Build the dataloader for training.

        Lightning automatically handles DistributedSampler when using DDP/FSDP strategy.
        No need to manually create samplers!

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Lightning will replace with DistributedSampler in DDP mode
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True,
            drop_last=False,
        )
        return train_loader


    def val_dataloader(self):
        """Build the dataloader for validation.

        Lightning automatically handles DistributedSampler for validation too!

        Returns:
            val_loader: PyTorch DataLoader used for validation.
        """
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Lightning will add DistributedSampler (shuffle=False) in DDP mode
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True,
        )
        return val_loader
