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
#
# This file is based on DDM-Net (https://github.com/MCG-NJU/DDM),
# Copyright (c) 2021 Mike Zheng Shou, licensed under the MIT License.
# Modifications Copyright (c) NVIDIA CORPORATION & AFFILIATES.
######################################################################################################

import matplotlib.pyplot as plt
from typing import Dict, List, Union


plt.switch_backend('Agg')
def visualize_scores_with_boundaries(
    video_name: str,
    golden_bdy: list[float],
    pred_bdy: list[float],
    scores_dict: Union[List[float], Dict[int, float]],
    fps: Union[float, int],
    output_path: str
    ):
    """
    Visualize scores over time with golden and predicted boundaries overlaid as vertical lines.

    Args:
        video_name: Name of the video
        golden_bdy: List of golden boundary timestamps
        pred_bdy: List of predicted boundary timestamps
        scores: List of scores or Dict mapping frame_idx to score
        fps: Frames per second of the video
        output_path: Path to save the visualization
    """
    if not scores_dict:
        print("No scores to visualize")
        return

    # Create time axis
    frame_times = [i / fps for i in scores_dict.keys()]
    scores = scores_dict.values()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot scores as area plot
    ax.fill_between(frame_times, scores, alpha=0.6, color='skyblue', label='Scores')
    ax.plot(frame_times, scores, color='blue', linewidth=1.5, alpha=0.8)

    # Draw vertical lines for golden boundaries
    for i, bdy in enumerate(golden_bdy):
        if 0 <= bdy <= max(frame_times):  # Only draw if boundary is within time range
            ax.axvline(x=bdy, color='red', linewidth=2, linestyle='--', alpha=0.8,
                      label='Golden Boundaries' if i == 0 else "")

            # Add boundary index label (starting from 1)
            ax.text(bdy, max(scores) * 0.9, str(i + 1),
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))

    # Draw vertical lines for predicted boundaries
    for i, bdy in enumerate(pred_bdy):
        if 0 <= bdy <= max(frame_times):  # Only draw if boundary is within time range
            ax.axvline(x=bdy, color='green', linewidth=2, linestyle='-', alpha=0.7,
                      label='Predicted Boundaries' if i == 0 else "")

            # Add boundary index label (starting from 1)
            ax.text(bdy, max(scores) * 0.8, str(i + 1),
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', edgecolor='green', alpha=0.9))

    # Set axis properties
    ax.set_xlim(0, max(frame_times) if frame_times else 1)
    ax.set_ylim(0, max(scores) * 1.1 if scores else 1)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc='upper right', fontsize=10)

    # Add title with video info
    ax.set_title(f'{video_name}\nScores over Time with Golden and Predicted Boundaries',
                fontsize=14, fontweight='bold')

    # Add text annotation with stats
    avg_score = sum(scores) / len(scores) if scores else 0
    max_score = max(scores) if scores else 0
    min_score = min(scores) if scores else 0

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()  # Close the figure to free memory