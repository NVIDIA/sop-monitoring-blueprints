#! /bin/bash
######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
######################################################################################################
set -e
# add default value for input_file and output_dir

function usage() {
    echo "Usage: $0 [input_file] [output_dir]"
    echo "Example: $0 ./test_video_whole_sop_h264.mp4 ./streams/simulation"
}

if [ -z "$1" ]; then
    input_file="./test_video_whole_sop_h264.mp4"
elif [ "$1" == "--help" || "$1" == "-h" ]; then
    usage
    exit 0
else
    input_file=$1
fi

if [ -z "$2" ]; then
    output_dir="./streams/simulation"
else
    output_dir=$2
fi

echo "Preparing camera simulation..."
echo "Input file: $input_file"
echo "Output dir: $output_dir"
echo "--------------------------------"
echo "Starting camera simulation..."

if [ ! -f "$input_file" ]; then
    echo "Error: Input file not found: $input_file"
    usage
    exit 1
fi

mkdir -p $output_dir

if [ ! -d "$output_dir" ]; then
    echo "Error: Output directory not created: $output_dir"
    usage
    exit 1
fi

gst-launch-1.0 -e filesrc location=$input_file ! decodebin ! nvvideoconvert ! pngenc ! multifilesink sync=false location=$output_dir/sop_sample_frame_%04d.png

if [ $? -ne 0 ]; then
    echo "Error: Failed to prepare camera simulation"
    exit 1
fi

echo "Camera simulation prepared successfully"
