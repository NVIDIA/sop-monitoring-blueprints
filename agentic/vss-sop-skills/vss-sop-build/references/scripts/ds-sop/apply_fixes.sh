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

# ds-sop Auto-Fix Recipes — applies remediation for the four post-build
# checks that have automatable fixes (Checks 2, 3, 4, 5).
#
# Run INSIDE the ds-sop container. After the fixes apply cleanly, port
# the changes back into docker/Docker.build and rebuild the image.
#
# Usage:
#   ./apply_fixes.sh            # apply ALL fixes
#   ./apply_fixes.sh check2     # only Check 2 fix (codec plugins)
#   ./apply_fixes.sh check3     # only Check 3 fix (recompile protos)
#   ./apply_fixes.sh check4     # only Check 4 fix (rebuild GStreamer registry)
#   ./apply_fixes.sh check5     # only Check 5 fix (reinstall codec libs)
set -uo pipefail

WHICH="${1:-all}"

fix_check2() {
  echo "=== Check 2 fix: install codec plugins + rebuild registry ==="
  apt update
  apt install -y gstreamer1.0-plugins-ugly gstreamer1.0-plugins-bad gstreamer1.0-libav
  rm -rf ~/.cache/gstreamer-1.0/registry.*.bin
  echo "Done. Add the above packages to docker/Docker.build, then rebuild."
  return 0
}

fix_check3() {
  echo "=== Check 3 fix: recompile protobuf stubs ==="
  cd /opt/nvidia/nvds_sop/nvds_action_detector/protos/
  protoc -I. --python_out=. nv.proto ext.proto
  sed -i 's/^import nv_pb2 as/from . import nv_pb2 as/' ext_pb2.py
  echo "Done. Update the Dockerfile RUN step to match, then rebuild."
  return 0
}

fix_check4() {
  echo "=== Check 4 fix: rebuild GStreamer registry cache ==="
  rm -rf ~/.cache/gstreamer-1.0/registry.*.bin
  gst-inspect-1.0 > /dev/null 2>&1 || true
  cat <<'EOF'
Element -> package map (install matching packages to satisfy missing elements):
  x264enc                                     -> gstreamer1.0-plugins-ugly
  jpegenc, rtph264pay, rtpjpegpay, udpsink   -> gstreamer1.0-plugins-good
  nvvideoconvert                              -> deepstream (base image)
EOF
  return 0
}

fix_check5() {
  echo "=== Check 5 fix: reinstall codec libraries ==="
  apt install -y --reinstall \
    libvpx9 libzvbi0t64 libmp3lame0 libx265-199 libunibreak5 libmpg123-0t64
  echo "Done. Add any new 'not found' libs (and -dev variants) to docker/Docker.build."
  return 0
}

case "$WHICH" in
  all)    fix_check2; echo; fix_check3; echo; fix_check4; echo; fix_check5 ;;
  check2) fix_check2 ;;
  check3) fix_check3 ;;
  check4) fix_check4 ;;
  check5) fix_check5 ;;
  *) echo "Unknown target: $WHICH (expected: all|check2|check3|check4|check5)"; exit 2 ;;
esac

