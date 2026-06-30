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

import os
import json
import math
import pathlib
from typing import Callable, Optional
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as T
from tqdm import tqdm
from qwen_vl_utils import fetch_image
from transformers import AutoProcessor
try:
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_AVAILABLE = True
except ImportError:
    TORCHCODEC_AVAILABLE = False
try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False


class PyAVVideoDecoder:
    """Wrapper class for PyAV to match torchcodec's VideoDecoder interface"""
    def __init__(self, video_path):
        self.video_path = video_path
        self.container = av.open(video_path)
        self.stream = self.container.streams.video[0]

        # Cache metadata
        self.metadata = type('obj', (object,), {
            'average_fps': float(self.stream.average_rate),
            'duration_seconds': float(self.stream.duration * self.stream.time_base) if self.stream.duration else self.container.duration / av.time_base
        })

        # Cache frames for efficient random access
        self._frames_cache = None
        self._num_frames = None

    def __len__(self):
        if self._num_frames is None:
            # Count frames - this might be slow for large videos
            self._num_frames = self.stream.frames
            if self._num_frames == 0:
                # If frame count is not available in metadata, count manually
                self._num_frames = sum(1 for _ in self.container.decode(video=0))
                self.container.seek(0)  # Reset to beginning
        return self._num_frames

    def get_frames_at(self, indices, return_pil=False):
        """Get frames at specific indices with smart seeking

        Args:
            indices: numpy array or list of frame indices

        Returns:
            Object with .data attribute containing frames as torch tensor (N, 3, H, W) in uint8
        """
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()

        frames_dict = {}
        pil_dict = {}
        min_idx = min(indices)
        max_idx = max(indices)

        # Smart seek: only seek if min_idx is far from beginning
        if min_idx > 100:  # Threshold for when seeking is worth it
            # Seek to a bit before min_idx
            target_time = max(0, (min_idx - 50) / float(self.stream.average_rate))
            # Use stream time_base for proper PTS calculation
            if hasattr(self.stream, 'time_base'):
                target_pts = int(target_time / float(self.stream.time_base))
            else:
                # Fallback: assume time_base is 1/1000000 (microseconds)
                target_pts = int(target_time * 1000000)

            try:
                # Use stream-specific seeking for better accuracy
                self.container.seek(target_pts, stream=self.stream)
            except:
                try:
                    # Fallback to timestamp-based seeking
                    self.container.seek(int(target_time * av.time_base))
                except:
                    # Final fallback to beginning if all seeking fails
                    self.container.seek(0)
        else:
            # For frames near the beginning, just start from 0
            self.container.seek(0)

        indices_set = set(indices)

        # We don't know exact frame_idx after seeking, so we need to figure it out
        frame_idx = 0
        estimated_start_frame = None

        # If we did seek, estimate where we landed
        if min_idx > 100:
            estimated_start_frame = max(0, min_idx - 50)
            frame_idx = estimated_start_frame

        frames_seen = 0
        for frame in self.container.decode(video=0):
            # On first frame after seek, try to get accurate position
            if frames_seen == 0 and frame.pts is not None and estimated_start_frame is not None:
                # Calculate actual frame index from PTS
                time_in_seconds = float(frame.pts * self.stream.time_base)
                actual_frame = int(time_in_seconds * float(self.stream.average_rate))

                # Only trust PTS-based estimate if it's reasonable
                if abs(actual_frame - estimated_start_frame) < 100:
                    frame_idx = actual_frame
                # If PTS estimate seems wrong, check if we overshot
                elif actual_frame > max_idx:
                    # We overshot, need to seek back to beginning
                    self.container.seek(0)
                    frame_idx = 0
                    frames_seen = -1  # Reset counter
                    continue

            # Only process frames in our range of interest
            if frame_idx >= min_idx - 10 and frame_idx <= max_idx + 10:
                if frame_idx in indices_set:
                    img = frame.to_ndarray(format='rgb24')  # (H, W, 3)
                    img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
                    frames_dict[frame_idx] = img
                    if return_pil:
                        pil_dict[frame_idx] = frame.to_image()
                    indices_set.remove(frame_idx)

                    if not indices_set:
                        break

            frame_idx += 1
            frames_seen += 1

            # Stop if we've passed all our indices
            if frame_idx > max_idx + 10:
                break

        # Handle missing frames - might occur due to seeking inaccuracy
        if indices_set:
            # If we missed some frames, we need to decode from beginning
            self.container.seek(0)
            frame_idx = 0

            for frame in self.container.decode(video=0):
                if frame_idx in indices_set:
                    img = frame.to_ndarray(format='rgb24')
                    img = torch.from_numpy(img).permute(2, 0, 1)
                    frames_dict[frame_idx] = img
                    if return_pil:
                        pil_dict[frame_idx] = frame.to_image()
                    indices_set.remove(frame_idx)

                    if not indices_set:
                        break

                frame_idx += 1

                if frame_idx > max(indices_set) if indices_set else max_idx:
                    break

        # Create ordered output according to original indices (preserve order!)
        ordered_frames = []
        ordered_pil = []
        for idx in indices:
            if idx in frames_dict:
                ordered_frames.append(frames_dict[idx])
                if return_pil:
                    ordered_pil.append(pil_dict[idx])
            else:
                # Create zero frame with correct dimensions
                h = self.stream.height if hasattr(self.stream, 'height') else 224
                w = self.stream.width if hasattr(self.stream, 'width') else 224
                ordered_frames.append(torch.zeros(3, h, w, dtype=torch.uint8))
                if return_pil:
                    ordered_pil.append(None)

        frames_tensor = torch.stack(ordered_frames, dim=0)
        if frames_tensor.dtype != torch.uint8:
            raise ValueError(
                f"Expected uint8 frames from video decoder, got {frames_tensor.dtype}. "
                f"ToDtype(scale=True) requires uint8 input to correctly divide by 255."
            )

        # Return with .data attribute
        if return_pil:
            return type('obj', (object,), {'data': frames_tensor, 'pil': ordered_pil})
        else:
            return type('obj', (object,), {'data': frames_tensor})

    def close(self):
        """Close the video container"""
        if hasattr(self, 'container'):
            self.container.close()

    def __del__(self):
        """Ensure container is closed when object is destroyed"""
        self.close()


class DDMDataset(Dataset):
    """
    DDM Dataset for Training. Supports validation as well, but validation using this dataset is slow.
    Argument Explanation:
    """
    def __init__(
        self,
        mode: str,
        anno_path: str,
        data_root: pathlib.Path,
        transform: Optional[Callable] = None,
        resolution: Optional[int | tuple[int, int]] = 224,  # Can be an int or a tuple (H, W)
        num_classes: int = 2,
        frames_per_side: int = 5,
        downsample: int = 1,
        min_change_dur: float = 0.3,
        seed: int = 666,
        video_backend: str = "pyav", # Currently, only 'pyav' and 'torchcodec' are supported
        use_cache: bool = False,
        processor_name_or_path: Optional[str] = None,
    ):
        assert mode.lower() in ["train", "val", "test"], "Wrong mode for DDM"
        assert video_backend in ["pyav", "torchcodec"], "Currently only support pyav and torchcodec"

        # Backend selection and availability check
        # If a processor is specified, force 'pyav' backend
        if processor_name_or_path and processor_name_or_path != "":
            video_backend = "pyav"
        if video_backend == "torchcodec" and not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available. Please install it or use 'pyav' backend.")
        elif video_backend == "pyav" and not PYAV_AVAILABLE:
            raise ImportError("pyav is not available. Please install it or use 'torchcodec' backend.")

        self.mode = mode.lower()
        self.anno_path = anno_path
        self.data_root = data_root
        self.frames_per_side = frames_per_side
        self.seed = seed
        self.video_backend = video_backend
        self.downsample = downsample
        self.min_change_dur = min_change_dur

        if isinstance(resolution, int): # (H, W)
            self.resolution = (resolution, resolution)
        else:
            self.resolution = resolution

        # Processor is required if you want to use CR's Vision Encoder as DDM-Net's backbone
        if processor_name_or_path and processor_name_or_path != "":
            # use fast version or not is not important here (it will be only used in text tokenizer)
            self.processor = AutoProcessor.from_pretrained(processor_name_or_path, use_fast=True)
        else:
            self.processor = None


        # In training, since training sample is sparse, we don't need to cache the video content
        self.use_cache = use_cache if self.mode != "train" else False

        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize(self.resolution),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.seqs = [] # list of dicts: video_id, label, current_idx, block_idx
        self.video_paths = {} # dict of video_id: video_features
        self.video_info = {} # dict of video_id: fps
        self.process_data(anno_path) # process sequence and video features

        self.labels_set = list(np.arange(num_classes))
        if self.mode == "train":
            self.labels = torch.LongTensor([dta["label"] for dta in self.seqs])
            self.label_to_indices = {
                label: np.where(self.labels.numpy() == label)[0]
                for label in self.labels_set
            }
            self.ratios = [
                len(self.label_to_indices[0]) / len(self.label_to_indices[1]),
                1,
            ]
        else: # val or test
            self.labels = torch.LongTensor([dta["label"] for dta in self.seqs])
            self.label_to_indices = {
                label: np.where(self.labels.numpy() == label)[0]
                for label in self.labels_set
            }


    def process_data(self, anno_path):

        for k, content in tqdm(json.load(open(anno_path, "r")).items()):
            video_path = os.path.join(self.data_root, k + ".mp4")
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file {video_path} not found")

            self.video_paths[k] = video_path # Store the path

            # You can still open the video temporarily to get metadata
            self.video_info[k] = {}
            try:
                if self.video_backend == "torchcodec":
                    vd = VideoDecoder(video_path)
                else: # pyav
                    vd = PyAVVideoDecoder(video_path)

                fps = vd.metadata.average_fps
                duration = vd.metadata.duration_seconds
                vlen = len(vd)

                # Preprocess and cache the video content
                if self.mode != "train" and self.use_cache:
                    if self.video_backend == "torchcodec":
                        video_content = vd.get_frames_in_range(0, vlen)
                    else: # pyav
                        video_content = vd.get_frames_at(np.arange(vlen), return_pil=bool(self.processor))

                    # Normalize and transform video frames
                    self.video_info[k]["inp"] = self.transform(video_content.data)

                    if self.processor:
                        pil_list = [
                            fetch_image({
                                "image": pil,
                                "resized_height": self.resolution[0],
                                "resized_width": self.resolution[1]
                            })
                            for pil in video_content.pil
                            if pil is not None
                        ]
                        inputs = self.processor.image_processor(pil_list)
                        # Store pixel_values and grid_thw for each frame
                        self.video_info[k]["pixel_values"] = inputs["pixel_values"].view(vlen, -1, inputs["pixel_values"].shape[-1])
                        self.video_info[k]["grid_thw"] = inputs["image_grid_thw"]
                    del video_content

                del vd # IMPORTANT: Close the file handle immediately
            except Exception as e:
                print(f"Could not read metadata for {video_path}: {e}")
                continue

            self.video_info[k].update({"fps": fps, "duration": duration, "vlen": vlen})
            boundary_list = []
            # content.pop() # Exclude Final Segment
            content = [ct for ct in content if "final segment" not in ct["description"].lower()]
            for s_sample, e_sample in zip(content[:-1], content[1:]):
                s_time = s_sample["end_timestamp"]
                e_time = e_sample["start_timestamp"]
                boundary_list.append(math.floor((s_time + e_time) / 2 * fps))

            labels = np.zeros(vlen)
            half_dur_2_nframes = self.min_change_dur * fps / 2
            for boundary in boundary_list:
                start_idx = math.ceil(max(0, boundary - half_dur_2_nframes))
                end_idx = math.floor(min(vlen, round(boundary + half_dur_2_nframes) + 1))
                labels[start_idx: end_idx] = 1

            for selected_idx in range(0, vlen, self.downsample):
                if selected_idx == 0 or selected_idx == vlen - 1:
                    continue
                block_idx = selected_idx + np.arange(-self.downsample * self.frames_per_side, self.downsample * (self.frames_per_side + 1), self.downsample)
                block_idx = np.clip(block_idx, 0, vlen - 1)
                sample = {
                    "video_id": k,
                    "label": labels[selected_idx],
                    "current_idx": block_idx[len(block_idx) // 2],
                    "block_idx": block_idx
                }
                self.seqs.append(sample)
        self.seqs = np.array(self.seqs, dtype=object)


    def _get_training_samples(self, index):
        indices = []
        for class_ in self.labels_set:
            real_index = self.label_to_indices[class_][int(index * self.ratios[class_])]
            indices.append(real_index)
        return indices


    def _read_data(self, index):
        item = self.seqs[index]
        video_id = item["video_id"]
        block_idx = item["block_idx"]
        current_idx = block_idx[len(block_idx) // 2]

        video_path = self.video_paths[video_id]
        returned_dict = {}

        try:
            if self.mode != "train" and self.use_cache:
                block_index_tensor = torch.tensor(block_idx)
                img = self.video_info[video_id]["inp"][block_index_tensor]
                if self.processor:
                    returned_dict.update({
                        "pixel_values": self.video_info[video_id]["pixel_values"][block_index_tensor],
                        "grid_thw": self.video_info[video_id]["grid_thw"][block_index_tensor]
                    })
            else:
                # Use the decoder to get frames
                if self.video_backend == "torchcodec":
                    decoder = VideoDecoder(video_path)
                    video_content = decoder.get_frames_at(block_idx)
                else: # pyav
                    decoder = PyAVVideoDecoder(video_path)
                    video_content = decoder.get_frames_at(block_idx, return_pil=bool(self.processor))
                img = self.transform(video_content.data)

                if self.processor:
                    pil_list = [fetch_image({"image": pil, "resized_height": self.resolution[0], "resized_width": self.resolution[1]}) for pil in video_content.pil]
                    inputs = self.processor.image_processor(pil_list)
                    returned_dict.update({
                        "pixel_values": inputs["pixel_values"].view(len(block_idx), -1, inputs["pixel_values"].shape[-1]),
                        "grid_thw": inputs["image_grid_thw"]}
                    )
                    del inputs, pil_list
                del video_content # Clean up the decoder and its file handle

        except Exception as e:
            print(f"Error reading video {video_id} at index {index}: {e}")
            # Return a dummy tensor or raise the error
            # For simplicity, we can raise it
            raise e

        returned_dict.update({
            "inp": img,
            "label": item["label"],
            "video_id": video_id,
            "current_idx": current_idx
        })


        return returned_dict


    def __getitem__(self, index):
        indices = self._get_training_samples(index) if self.mode == "train" else [index]
        samples = [self._read_data(real_index) for real_index in indices]
        returned_dict = {
            "inp": torch.stack([sample["inp"] for sample in samples], dim=0),
            "label": torch.LongTensor([sample["label"] for sample in samples]),
            "path": [sample["video_id"] for sample in samples],
            "current_ids": [sample["current_idx"] for sample in samples],
        }
        if self.processor:
            returned_dict.update({
                "pixel_values": torch.stack([sample["pixel_values"] for sample in samples], dim=0),
                "grid_thw": torch.cat([sample["grid_thw"] for sample in samples], dim=0),
            })

        return returned_dict


    def shuffle(self):
        np.random.seed(self.seed)
        for class_ in self.labels_set:
            np.random.shuffle(self.label_to_indices[class_])


    def __len__(self):
        if self.mode == "train":
            return len(self.label_to_indices[1])

        return sum([len(v) for v in self.label_to_indices.values()])


    def collate_fn(self, batch):
        """
        Custom collate function that handles flattening of path and current_ids.

        This prevents current_ids from being converted to tensor by default_collate,
        and ensures all data has matching dimensions for training/validation steps.
        """
        # Filter out None samples
        batch = list(filter(lambda x: x["inp"] is not None, batch))
        if len(batch) == 0:
            return torch.Tensor()

        # Stack tensors
        inp = torch.stack([item['inp'] for item in batch])  # (B, seq_len, C, H, W)
        label = torch.stack([item['label'] for item in batch])  # (B, seq_len)

        # Flatten current_ids to simple list (not tensor!)
        current_ids_flat = []
        for ids in [item['current_ids'] for item in batch]:
            if isinstance(ids, torch.Tensor):
                current_ids_flat.extend(ids.cpu().tolist())
            elif isinstance(ids, list):
                current_ids_flat.extend(ids)
            else:
                current_ids_flat.append(int(ids))

        # Flatten paths to simple list
        paths_flat = []
        for path_list in [item['path'] for item in batch]:
            if isinstance(path_list, list):
                paths_flat.extend(path_list)
            else:
                paths_flat.append(path_list)

        result = {
            'inp': inp,
            'label': label,
            'path': paths_flat,
            'current_ids': current_ids_flat
        }

        # Handle processor outputs if present
        if 'pixel_values' in batch[0]:
            result['pixel_values'] = torch.stack([item['pixel_values'] for item in batch])
        if 'grid_thw' in batch[0]:
            result['grid_thw'] = torch.cat([item['grid_thw'] for item in batch])

        return result