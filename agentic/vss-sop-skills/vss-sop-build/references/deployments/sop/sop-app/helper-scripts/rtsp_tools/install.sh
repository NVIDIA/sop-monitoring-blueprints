#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

sudo apt-get update && sudo apt-get install -y gir1.2-gst-rtsp-server-1.0 libgstrtspserver-1.0-0
sudo apt-get update && sudo apt-get install -y gstreamer1.0-libav
sudo apt-get install python3-gi python3-gi-cairo libgirepository-2.0-dev
sudo apt-get install -y gobject-introspection
sudo apt-get install libcairo2-dev pkg-config python3-dev
pkg-config --cflags --libs girepository-1.0
pip install PyGObject
sudo apt update && sudo apt install -y ffmpeg
