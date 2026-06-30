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

# ds-sop Post-Build Verification — runs the 8 in-container checks against
# the built ds-sop:1.0.0 image. Stops at the first failing check.
#
# NOTE: RTSP streaming output is an OPT-IN ds-sop-skills feature (ds-sop-skills § 18 —
# skill_18_rtsp_streaming_output.md); the generation prompt must request it
# ("with rtsp streaming output feature"). vss-sop-build does not patch the source;
# these checks verify that the GENERATED + built image shipped RTSP correctly. If a
# check fails, the fix lives in the ds-sop-skills generation (ensure RTSP was requested)
# — regenerate ds_sop_microservice rather than patching here.
#
# Run modes:
#   1. Inside the container (after `docker run --rm -it --gpus all ds-sop:1.0.0`):
#      ./post_build_checks.sh
#   2. From the host (auto-execs in a fresh container):
#      ./post_build_checks.sh --container
set -uo pipefail

if [[ "${1:-}" = "--container" ]]; then
  SCRIPT="$(realpath "$0")"
  exec docker run --rm --gpus all \
    -v "$SCRIPT:/tmp/post_build_checks.sh:ro" \
    --entrypoint bash ds-sop:1.0.0 /tmp/post_build_checks.sh
fi

echo "=== DS-SOP Post-Build Checks ==="
FAIL=0

run_check() {
  local n="$1"; shift
  local name="$1"; shift
  echo
  echo "--- Check $n: $name ---"
  if "$@"; then
    echo "Check $n: PASS"
  else
    echo "Check $n: FAIL"
    FAIL=$((FAIL+1))
  fi
  return 0
}

# Check 1 — GstRtspServer importable
run_check 1 "GstRtspServer importable" python3 -c "
import gi
gi.require_version('GstRtspServer','1.0')
from gi.repository import GstRtspServer
print('OK: GstRtspServer', GstRtspServer)
"

# Check 2 — x264enc available
run_check 2 "x264enc encoder available" python3 -c "
import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; Gst.init(None)
enc = Gst.ElementFactory.find('x264enc')
print('OK: x264enc found' if enc else 'FAIL: x264enc not found')
import sys; sys.exit(0 if enc else 1)
"

# Check 3 — Protobuf stubs compiled
run_check 3 "Protobuf stubs compiled" python3 -c "
from nvds_action_detector.protos import nv_pb2, ext_pb2
print('OK: nv_pb2 descriptors:',  len(nv_pb2.DESCRIPTOR.message_types_by_name))
print('OK: ext_pb2 descriptors:', len(ext_pb2.DESCRIPTOR.message_types_by_name))
"

# Check 4 — GStreamer registry has all required plugins
run_check 4 "GStreamer registry plugins" python3 -c "
import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; Gst.init(None)
required = ['x264enc', 'rtph264pay', 'udpsink', 'nvvideoconvert', 'jpegenc', 'rtpjpegpay']
missing = [e for e in required if not Gst.ElementFactory.find(e)]
print('FAIL: missing', missing) if missing else print('OK: all required GStreamer elements found')
import sys; sys.exit(1 if missing else 0)
"

# Check 5 — Shared library deps satisfied
run_check 5 "Shared library deps" python3 -c "
import subprocess, sys
r = subprocess.run(['bash','-c','ldd /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgst*.so 2>/dev/null | grep \"not found\"'],
                   capture_output=True, text=True)
print('FAIL: missing libs:\n'+r.stdout) if r.stdout.strip() else print('OK: all GStreamer plugin shared libraries satisfied')
sys.exit(1 if r.stdout.strip() else 0)
"

# Check 6 — RTSPStreamingServer instantiable
run_check 6 "RTSPStreamingServer instantiable" python3 -c "
import gi, socket
gi.require_version('Gst','1.0'); gi.require_version('GstRtspServer','1.0')
from gi.repository import Gst, GstRtspServer
Gst.init(None)
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.bind(('', 0)); port = s.getsockname()[1]
srv = GstRtspServer.RTSPServer.new()
srv.set_service(str(port))
factory = GstRtspServer.RTSPMediaFactory.new()
factory.set_launch(f'( udpsrc name=pay0 port={port+1} caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96\" )')
factory.set_shared(True)
srv.get_mount_points().add_factory('/test', factory)
srv.attach(None)
print(f'OK: RTSP server bound on port {port}, path /test')
"

# Check 7 — Qwen3-VL detection wired
run_check 7 "Qwen3-VL detection wired" python3 -c "
import inspect
from nvds_action_detector.vllm_inference import VLLMInference
src = inspect.getsource(VLLMInference)
assert '_is_qwen3vl' in src, 'FAIL: _is_qwen3vl missing'
assert 'return_video_metadata' in src, 'FAIL: return_video_metadata not wired'
print('OK: Qwen3-VL detection wired into VLLMInference')
"

# Check 8 — Chunking is DDM-Net only
run_check 8 "ChunkingOptions = DdmNetChunkingOptions only" python3 -c "
from nvds_action_detector.api_types import ChunkingOptions
import typing
inner = typing.get_args(typing.get_args(ChunkingOptions)[0])
names = [t.__name__ for t in inner]
assert 'UniformChunkingOptions' not in names, f'FAIL: still present in {names}'
print('OK: ChunkingOptions =', names)
"

echo
echo "=== Summary: $FAIL failure(s) ==="
exit "$FAIL"

