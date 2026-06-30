#! /bin/bash
######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
set -e
# add default value for input_file and output_dir

function usage() {
    echo "Usage: $0 [input_file] [output_dir]"
    echo "Example: $0 ./test_video_whole_sop_h264.mp4 ./streams/simulation"
    return 0
}

if [[ -z "$1" ]]; then
    input_file="./test_video_whole_sop_h264.mp4"
elif [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
else
    input_file=$1
fi

if [[ -z "$2" ]]; then
    output_dir="./streams/simulation"
else
    output_dir=$2
fi

echo "Preparing camera simulation..."
echo "Input file: $input_file"
echo "Output dir: $output_dir"
echo "--------------------------------"
echo "Starting camera simulation..."

if [[ ! -f "$input_file" ]]; then
    echo "Error: Input file not found: $input_file" >&2
    usage
    exit 1
fi

mkdir -p $output_dir

if [[ ! -d "$output_dir" ]]; then
    echo "Error: Output directory not created: $output_dir" >&2
    usage
    exit 1
fi

if ! gst-launch-1.0 -e \
    filesrc location=$input_file ! decodebin ! nvvideoconvert ! pngenc ! \
    multifilesink sync=false location=$output_dir/sop_sample_frame_%04d.png; then
    echo "Error: Failed to prepare camera simulation" >&2
    exit 1
fi

echo "Camera simulation prepared successfully"
