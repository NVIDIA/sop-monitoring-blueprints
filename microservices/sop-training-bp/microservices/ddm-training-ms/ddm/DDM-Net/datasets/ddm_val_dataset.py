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
import math
import time
import json
import av
import random
import collections

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as data
from torch.utils.data import IterableDataset, default_collate

from torchvision import transforms

from typing import (
    Optional, Callable, Tuple, List, Union, Dict
)

# Import Decord
try:
    from decord import VideoReader, bridge, cpu
    bridge.set_bridge('torch')
except ImportError:
    raise ImportError("Please install decord: pip install decord")

DEFAULT_RESOLUTION = (224, 224)
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


class DecordStreamingReader:
    """
    High-performance Video Reader using Decord.

    Args:
        path: str, path to the video file
        frames_per_side: int, number of frames to read on each side of the current frame
        downsample: int, temporal downsampling rate. Controls frame sampling density along the time axis.
        temporal_stride: int, sliding window stride (in units of downsample). Controls how frequently boundary scores are computed.
        start_time: float, start time of the video
        end_time: float, end time of the video
        resolution: tuple, resolution of the video
        transform: callable, transform to apply to the video

    Returns:
        dict, a dictionary containing the video frames and the current frame index
            "inp": torch.Tensor, the video frames
            "current_ids": int, the current frame index
            "worker_id": int, the worker id
    """
    def __init__(
        self,
        path: str,
        frames_per_side: int = 5,
        downsample: int = 1,
        temporal_stride: int = 1,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        resolution: Union[Tuple[int, int], int] = DEFAULT_RESOLUTION,
        transform: Optional[Callable] = None,
    ):
        self.path = path
        self.frames_per_side = frames_per_side
        self.downsample = downsample
        self.temporal_stride = temporal_stride
        if isinstance(resolution, int):
            self.resolution = (resolution, resolution)
        elif type(resolution) == tuple and len(resolution) == 2:
            self.resolution = resolution
        else:
            self.resolution = DEFAULT_RESOLUTION
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.resolution),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
            ])

        # num_threads=0 is crucial to prevent interference with DataLoader workers
        self.vr = VideoReader(path, ctx=cpu(0), num_threads=0)
        
        self.fps = self.vr.get_avg_fps()
        self.total_frames = len(self.vr)

        start_frame_idx = round(start_time * self.fps)
        if end_time is not None:
            self.end_frame_idx = min(round(end_time * self.fps), self.total_frames)
        else:
            self.end_frame_idx = self.total_frames

        buffer_size = 2 * self.frames_per_side + 1
        self.tensor_buffer = torch.zeros(
            (buffer_size, 3, self.resolution[0], self.resolution[1]),
            dtype=torch.float32
        )
        
        self.current_center_idx = start_frame_idx
        self._fill_buffer()

    def _process_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim == 3:  # single frame: (H, W, C) -> (C, H, W)
            frames = frames.permute(2, 0, 1)
        elif frames.ndim == 4:  # batch: (N, H, W, C) -> (N, C, H, W)
            frames = frames.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unexpected frames shape: {frames.shape}, expected single frame (H,W,C) or batch (N,H,W,C)")
        if frames.dtype != torch.uint8:
            raise ValueError(
                f"Expected uint8 frames from video decoder, got {frames.dtype}. "
                f"ToDtype(scale=True) requires uint8 input to correctly divide by 255."
            )
        return self.transform(frames)

    def _fill_buffer(self, frame_indices=None):
        if frame_indices is None:
            frame_indices = self._get_clipped_indices()
        frames_batch = self.vr.get_batch(frame_indices)  # (N, H, W, 3)
        self.tensor_buffer = self._process_frames(frames_batch)

    def _get_clipped_indices(self):
        # Calculate frame indices with downsample spacing
        # e.g., frames_per_side=1, downsample=2, center=0 -> indices = [-2, 0, 2]
        frame_indices = [
            self.current_center_idx + i * self.downsample
            for i in range(-self.frames_per_side, self.frames_per_side + 1)
        ]
        # Clip to valid range
        frame_indices = np.clip(frame_indices, 0, self.total_frames - 1).tolist()
        return frame_indices

    def __iter__(self):
        step_size = self.downsample * self.temporal_stride
        buffer_size = self.tensor_buffer.shape[0]

        while self.current_center_idx < self.end_frame_idx:
            # 1. return current result
            yield {
                "inp": self.tensor_buffer.clone(),
                "current_ids": self.current_center_idx
            }

            # 2. move to next center index
            self.current_center_idx += step_size
            if self.current_center_idx >= self.end_frame_idx:
                break
            
            # 3. caculate all indices for next sliding window
            frame_indices = self._get_clipped_indices()
            
            # 4. update tensor buffer for next iteration
            if self.temporal_stride < buffer_size:
                # a. buffer left shift
                self.tensor_buffer[:-self.temporal_stride] = self.tensor_buffer[self.temporal_stride:].clone()
                # b. fill in new frame features
                new_frame_indices = frame_indices[-self.temporal_stride:]
                new_frames = self.vr.get_batch(new_frame_indices)
                self.tensor_buffer[-self.temporal_stride:] = self._process_frames(new_frames)
            else:
                self._fill_buffer(frame_indices=frame_indices)


class DDMValStreamingDataset(IterableDataset):
    def __init__(
        self,
        annotation_file: str,
        video_root: str,
        frames_per_side: int = 5,
        downsample: int = 1,
        temporal_stride: int = 1,
        min_change_dur: float = 0.3,
        chunk_duration: Optional[float] = None,
        resolution: Union[Tuple[int, int], int] = DEFAULT_RESOLUTION,
        enable_load_balancing: bool = True,
        transform: Optional = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.frames_per_side = frames_per_side
        self.downsample = downsample
        self.temporal_stride = temporal_stride
        self.chunk_duration = chunk_duration
        self.min_change_dur = min_change_dur
        self.transform = transform
        self.verbose = verbose
        self.video_info = {}
        
        self.video_clip_configs: List[Dict] = [] 
        
        if isinstance(resolution, int):
            self.resolution = (resolution, resolution)
        else:
            self.resolution = resolution
            
        self.enable_load_balancing = enable_load_balancing

        self._process_data(annotation_file, video_root)
    
    def _process_data(self, annotation_file, video_root):
        if self.verbose:
            print(f"📂 Processing annotations from {annotation_file}...")
        try:
            with open(annotation_file, "r") as f:
                tmp_data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load annotation file: {e}")
            return

        for k, content in tmp_data.items():
            video_path = os.path.join(video_root, k + ".mp4")
            if not os.path.exists(video_path):
                continue
                
            try:
                with av.open(video_path) as container:
                    stream = container.streams.video[0]
                    duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
                    fps = float(stream.average_rate)
                    vlen = stream.frames if stream.frames > 0 else int(duration * fps)

                    base_config = {
                        "path": video_path,
                        "video_id": k,
                        "start": 0.0,
                        "end": duration
                    }

                    if self.chunk_duration:
                        clips = self._chunk_videos(base_config, self.chunk_duration)
                    else:
                        clips = [base_config]
                    
                    self.video_clip_configs.extend(clips)

                    # Handle Labels (remains unchanged)
                    boundary_list = []
                    content_ = [ct for ct in content if ct.get("description") != "Final Segment"]
                    for s_sample, e_sample in zip(content_[:-1], content_[1:]):
                        s_time = s_sample.get("end_timestamp", 0)
                        e_time = e_sample.get("start_timestamp", 0)
                        boundary_list.append(math.floor((s_time + e_time) / 2 * fps))

                    labels = torch.LongTensor(np.zeros(vlen))
                    half_dur_2_nframes = self.min_change_dur * fps / 2
                    for boundary in boundary_list:
                        start_idx = math.ceil(max(0, boundary - half_dur_2_nframes))
                        end_idx = math.floor(min(vlen, round(boundary + half_dur_2_nframes) + 1))
                        if end_idx > start_idx:
                            labels[start_idx: end_idx] = 1

                    self.video_info[k] = {
                        "duration": duration,
                        "fps": fps,
                        "labels": labels,
                    }

            except Exception as e:
                print(f"⚠️ Error processing video {k}: {e}")
                continue
        
        if self.verbose:
            print(f"✅ Processed {len(self.video_clip_configs)} clips from {len(self.video_info)} videos.")

    def _chunk_videos(
        self, 
        config: Dict, 
        chunk_duration: float,
    ) -> List[Dict]:
        
        start = config['start']
        end = config['end']
        vid_duration = end - start
        chunked = []
        
        if vid_duration > 0:
            num_chunks = int(math.ceil(vid_duration / chunk_duration))
            for i in range(num_chunks):
                cs = start + i * chunk_duration
                ce = min(start + (i + 1) * chunk_duration, end)
                
                # Copy original config and update time
                new_config = config.copy()
                new_config['start'] = cs
                new_config['end'] = ce
                chunked.append(new_config)
        else:
            chunked.append(config)
        
        return chunked

    def _balance_load_by_duration(self, configs: List[Dict], num_partitions: int):
        """
        Revised Load Balancing (Dict version)
        """
        if not configs: return [[] for _ in range(num_partitions)]
        
        # Calculate total duration
        total_dur = sum((c['end'] - c['start']) for c in configs)
        target = total_dur / num_partitions if num_partitions > 0 else 0
        
        SOFT_THRESHOLD = 2.0 
        
        partitions = [[] for _ in range(num_partitions)]
        part_loads = [0.0] * num_partitions
        
        curr_part = 0
        cfg_idx = 0
        offset = 0.0
        
        while cfg_idx < len(configs) and curr_part < num_partitions:
            config = configs[cfg_idx]
            
            # Get original parameters
            o_start = config['start']
            o_end = config['end']
            
            clip_dur = o_end - o_start
            remaining = clip_dur - offset
            
            current_load = part_loads[curr_part]
            space = target - current_load
            
            if curr_part == num_partitions - 1: space = float('inf')
            
            # Case 1: Can fit in current partition
            if remaining <= space + SOFT_THRESHOLD:
                c_start = o_start + offset
                c_end = c_start + remaining # == o_end
                
                new_config = config.copy()
                new_config['start'] = c_start
                new_config['end'] = c_end
                
                partitions[curr_part].append(new_config)
                part_loads[curr_part] += remaining
                
                cfg_idx += 1
                offset = 0.0
                
                if part_loads[curr_part] >= target:
                    curr_part += 1
            
            # Case 2: Need to split
            else:
                split_duration = space
                c_start = o_start + offset
                c_end = c_start + split_duration
                
                new_config = config.copy()
                new_config['start'] = c_start
                new_config['end'] = c_end
                
                partitions[curr_part].append(new_config)
                part_loads[curr_part] += split_duration
                
                offset += split_duration
                curr_part += 1
                
        return partitions

    def __iter__(self):
        # 1. DDP & Worker Info
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
            
        worker_info = data.get_worker_info()
        if worker_info:
            num_workers = worker_info.num_workers
            wid = worker_info.id
        else:
            num_workers = 1
            wid = 0
            
        total_partitions = world_size * num_workers
        my_global_id = rank * num_workers + wid
        
        # 2. Partition Data
        if self.enable_load_balancing:
            all_loads = self._balance_load_by_duration(self.video_clip_configs, total_partitions)
            my_configs = all_loads[my_global_id]
        else:
            my_configs = self.video_clip_configs[my_global_id::total_partitions]
        
        # 3. Shuffle
        if self.chunk_duration is not None:
            random.shuffle(my_configs)
        
        # --- 4. Logging (Dict Access) ---
        if self.verbose:
            total_assigned_duration = 0.0
            log_lines = [f"\n👷 [Rank {rank} | Worker {wid} (Global ID: {my_global_id})]"]
            log_lines.append(f"   {'Video Name':<20} | {'Start':<8} | {'End':<8} | {'Dur':<6}")
            log_lines.append(f"   {'-'*20} | {'-'*8} | {'-'*8} | {'-'*6}")
            
            for cfg in my_configs:
                video_name = cfg['video_id'][:20]
                start = cfg['start']
                end = cfg['end']
                dur_val = end - start
                total_assigned_duration += dur_val
                log_lines.append(f"   {video_name:<20} | {start:<8.1f} | {end:<8.1f} | {dur_val:<6.1f}")
                
            log_lines.append(f"   Total Load: {total_assigned_duration:.2f} seconds\n")
            
            if len(my_configs) > 0:
                print("\n".join(log_lines))

        # 5. Iterate
        for cfg in my_configs:
            try:
                reader = DecordStreamingReader(
                    path=cfg['path'],
                    frames_per_side=self.frames_per_side,
                    downsample=self.downsample,
                    temporal_stride=self.temporal_stride,
                    start_time=cfg['start'],
                    end_time=cfg['end'],
                    resolution=self.resolution,
                    transform=self.transform
                )
                
                for buffer_state in reader:
                    video_id = cfg['video_id']
                    current_ids = buffer_state["current_ids"]
                    yield {
                        "inp": buffer_state["inp"],
                        "label": self.video_info[video_id]["labels"][current_ids],
                        "path": video_id,
                        "current_ids": current_ids,
                        "worker_id": my_global_id
                    }
            except Exception as e:
                print(f"⚠️ Error [Worker {my_global_id}] reading {cfg['path']}: {e}")
                continue
    
    def collate_fn(self, batch):
        return default_collate(batch)