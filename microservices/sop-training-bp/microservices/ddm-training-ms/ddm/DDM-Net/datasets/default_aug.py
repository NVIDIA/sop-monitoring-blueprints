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
#
# This file is based on DDM-Net (https://github.com/MCG-NJU/DDM),
# Copyright (c) 2021 Mike Zheng Shou, licensed under the MIT License.
# Modifications Copyright (c) NVIDIA CORPORATION & AFFILIATES.
######################################################################################################

import random
import warnings

import torch
import torch.nn as nn
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]

INTERPOLATION_MAPPING = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
    "nearest_exact": InterpolationMode.NEAREST_EXACT,
}


def resolve_interpolations(interpolation_names):
    """Resolve string names to InterpolationMode enums, deduplicated and order-preserving.

    Args:
        interpolation_names: List of interpolation mode names to resolve.

    Returns:
        List of resolved interpolation modes.
    """
    resolved = []
    seen = set()
    for name in interpolation_names:
        key = str(name).lower()
        if key in INTERPOLATION_MAPPING:
            mode = INTERPOLATION_MAPPING[key]
            # Avoid repeating interpolation modes
            if mode not in seen:
                resolved.append(mode)
                seen.add(mode)
        else:
            warnings.warn(f"Unknown interpolation name '{name}', skipping.")
    return resolved


class RandomResizePolicy(nn.Module):
    """Resize with randomly sampled interpolation mode and antialias setting.

    Bilinear is always included as a fallback. When antialias_prob is 0,
    bilinear is weighted 2x higher and its antialias is coin-flipped.
    """

    def __init__(self, resolution, interp_names, antialias_prob):
        super().__init__()
        self.resolution = resolution
        self.interps = resolve_interpolations(interp_names)
        self.antialias_prob = min(1.0, max(0.0, antialias_prob))
        self.antialias_disabled = self.antialias_prob <= 0
        if InterpolationMode.BILINEAR not in self.interps:
            self.interps.append(InterpolationMode.BILINEAR)

    def forward(self, img):
        interp = self._sample_interpolation()
        antialias = self._sample_antialias(interp)
        return F.resize(
            img,
            self.resolution,
            interpolation=interp,
            antialias=antialias,
        )

    def _sample_interpolation(self):
        if not self.antialias_disabled:
            return random.choice(self.interps)
        # When antialias is disabled, favor bilinear (2x weight) for stability
        weights = [2 if m == InterpolationMode.BILINEAR else 1 for m in self.interps]
        return random.choices(self.interps, weights=weights, k=1)[0]

    def _sample_antialias(self, interp):
        # Bilinear without antialias can cause aliasing, so coin-flip when prob is 0
        if interp == InterpolationMode.BILINEAR and self.antialias_disabled:
            return random.choice([True, False])
        return random.random() < self.antialias_prob


def build_random_resize_transform(random_resize_config, resolution):
    """Build a random-interpolation resize followed by float32 conversion.

    Args:
        random_resize_config: Config dict with 'interpolation' and 'antialias_prob' keys.
        resolution: Target (H, W) resolution.

    Returns:
        List of transforms: [RandomResizePolicy, ToDtype].
    """
    policy = RandomResizePolicy(
        resolution=resolution,
        interp_names=random_resize_config.get("interpolation", []),
        antialias_prob=random_resize_config.get("antialias_prob", 0.5),
    )
    return [policy, T.ToDtype(torch.float32, scale=True)]


def build_color_jitter_transform(color_jitter_config):
    """Build a ColorJitter transform from config.

    Args:
        color_jitter_config: Config dict with brightness/contrast/saturation/hue keys.

    Returns:
        Single-element list containing the ColorJitter transform.
    """
    return [T.ColorJitter(
        brightness=color_jitter_config.get("brightness", 0),
        contrast=color_jitter_config.get("contrast", 0),
        saturation=color_jitter_config.get("saturation", 0),
        hue=color_jitter_config.get("hue", 0),
    )]


def build_random_gaussian_blur_transform(gaussian_blur_config):
    """Build a randomly-applied GaussianBlur transform from config.

    Args:
        gaussian_blur_config: Config dict with 'apply_prob', 'kernel_size', and 'sigma' keys.

    Returns:
        Single-element list containing RandomApply(GaussianBlur).
    """
    p = min(1.0, max(0.0, float(gaussian_blur_config.get("apply_prob", 0.5))))
    kernel_size = gaussian_blur_config.get("kernel_size", 3)
    if kernel_size % 2 == 0:
        raise ValueError(f"GaussianBlur kernel_size must be odd, got {kernel_size}")
    sigma = gaussian_blur_config.get("sigma", [0.1, 0.5])
    return [T.RandomApply([T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)], p=p)]


def compose_default_augmentations(augmentation_config, resolution):
    """Compose the default augmentation pipeline: Resize -> ToDtype -> [augmentations] -> Normalize.

    Args:
        augmentation_config: Augmentation config dict with optional RandomResize/ColorJitter/GaussianBlur sections.
        resolution: Target resolution as int or (H, W) tuple.

    Returns:
        Composed transform pipeline.
    """
    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    transforms = []

    if augmentation_config.get("RandomResize", {}).get("enabled", False):
        transforms.extend(build_random_resize_transform(augmentation_config["RandomResize"], resolution))
    else:
        transforms.extend([T.Resize(resolution), T.ToDtype(torch.float32, scale=True)])

    if augmentation_config.get("ColorJitter", {}).get("enabled", False):
        transforms.extend(build_color_jitter_transform(augmentation_config["ColorJitter"]))
    if augmentation_config.get("GaussianBlur", {}).get("enabled", False):
        transforms.extend(build_random_gaussian_blur_transform(augmentation_config["GaussianBlur"]))

    transforms.append(T.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD))

    return T.Compose(transforms)
