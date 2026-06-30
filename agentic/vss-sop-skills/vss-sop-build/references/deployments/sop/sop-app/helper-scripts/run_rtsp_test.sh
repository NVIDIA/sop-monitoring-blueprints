#!/bin/bash

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

# End-to-end wiring for VSS-SOP livestream/recording.
#
# Pipeline:
#   rtsp_server.py (:8552)  →  mdx-ds-sop-1 DeepStream pipeline  →  Kafka / ELK
#   rtsp_server.py (:8554 /ds-out/<sensor>, standalone passthrough)  →  VST
#       rtspserver-ms  →  livestream-ms / recorder-ms
#
# The ds-sop DeepStream pipeline analyses the :8552 input and publishes SOP
# chunk results to Kafka (-> Elasticsearch). It does NOT itself serve the :8554
# RTSP output: in-pipeline RTSP is disabled by default (DS_INPIPELINE_RTSP=0)
# because creating a GstRtspServer inside the pyservicemaker processor thread
# (no serviced GLib main loop) hangs pipeline construction and starves the DDM
# inference branch on no-NVENC GPUs (H100/A100). Instead a standalone
# rtsp_server.py passthrough re-streams the source on :8554/ds-out/<sensor>,
# fully decoupled from inference, which VST records and livestreams. Because the
# SOP output carries no overlays (no bboxes), a plain passthrough re-stream of
# the source H.264 is identical in content to what ds-sop would have emitted.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SENSOR_ID="sensor_0"
# Prefer the routable default-route source IP (avoids picking a link-local/secondary
# interface like 169.254.x.x, which VST/recorder containers cannot reach).
HOST_IP="${HOST_IP:-$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9.]+' | head -1)}"
HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"
VST_ENDPOINT="${VST_ENDPOINT:-http://localhost:30888/vst}"
RTSP_OUT_PORT="${RTSP_OUT_PORT:-8554}"
export TEST_RTSP_VIDEO_URL="rtsp://127.0.0.1:8552/$SENSOR_ID"

# Resolve the source video served on :8552 and reuse it for the :8554 passthrough.
discover_video_file() {
    local vf
    vf=$(pgrep -af "rtsp_server.py" 2>/dev/null \
        | grep -oE -- '--filename[= ][^ ]+' | head -1 | sed -E 's/^--filename[= ]//')
    if [[ -z "$vf" || ! -f "$vf" ]]; then
        # Fall back to the deploy default path (WORKSPACE = CWD of bp repo).
        local ws
        ws="$(cd "$SCRIPT_DIR/../../../../" 2>/dev/null && pwd)"
        vf="$ws/sop-resources/sop-server-fan-installation-data_v1.0-260213/server_fan/raw/Install_1_h264_30fps.mp4"
    fi
    echo "$vf"
    return 0
}
VIDEO_FILE="$(discover_video_file)"

cleanup() {
    if [[ -n "${DS_PID:-}" ]] && kill -0 "$DS_PID" 2>/dev/null; then
        echo "Stopping ds-sop client (pid $DS_PID)..."
        kill "$DS_PID" 2>/dev/null || true
        wait "$DS_PID" 2>/dev/null || true
    fi
    if [[ -n "${RELAY_PID:-}" ]] && kill -0 "$RELAY_PID" 2>/dev/null; then
        echo "Stopping :$RTSP_OUT_PORT passthrough relay (pid $RELAY_PID)..."
        kill "$RELAY_PID" 2>/dev/null || true
        wait "$RELAY_PID" 2>/dev/null || true
    fi
    return 0
}
trap cleanup EXIT

remove_existing_sensor() {
    # Best-effort: drop any prior sensor_0 registration so we start cold.
    local ids
    ids=$(curl -sS "${VST_ENDPOINT}/api/v1/sensor/list" 2>/dev/null \
        | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin)
except Exception:
    sys.exit(0)
for s in d:
    if s.get("name") == "'"$SENSOR_ID"'":
        print(s.get("sensorId",""))' 2>/dev/null || true)
    for sid in $ids; do
        [[ -n "$sid" ]] || continue
        echo "Removing stale VST sensor $sid..."
        curl -sS -X DELETE "${VST_ENDPOINT}/api/v1/sensor/$sid" >/dev/null 2>&1 || true
    done
    return 0
}

kill_stale_api_client() {
    # A prior interrupted run may have left api_client_test.py running. Its
    # HTTP session to ds-sop is effectively dead but the Python process can
    # keep retrying, so without this we'd end up with duplicate clients after
    # a few runs. pkill is a no-op if none exist.
    if pgrep -f "python3 -u api_client_test.py" >/dev/null 2>&1; then
        echo "Killing stale api_client_test.py processes..."
        pkill -f "python3 -u api_client_test.py" 2>/dev/null || true
        sleep 1
        pkill -9 -f "python3 -u api_client_test.py" 2>/dev/null || true
    fi
    return 0
}

kill_stale_relay() {
    # Drop any prior :8554 passthrough relay so we start cold on the output port.
    if pgrep -f "rtsp_server.py.*--port[= ]$RTSP_OUT_PORT" >/dev/null 2>&1; then
        echo "Killing stale :$RTSP_OUT_PORT passthrough relay..."
        pkill -f "rtsp_server.py.*--port[= ]$RTSP_OUT_PORT" 2>/dev/null || true
        sleep 1
    fi
    return 0
}

kill_stale_api_client
kill_stale_relay
remove_existing_sensor

echo "Source video for SOP streaming: $VIDEO_FILE"

echo "Starting ds-sop inference pipeline in background (drives SOP -> Kafka -> ELK)..."
python3 -u api_client_test.py >/tmp/ds_sop_client.log 2>&1 &
DS_PID=$!

echo "Starting standalone :$RTSP_OUT_PORT passthrough relay (/ds-out/$SENSOR_ID)..."
python3 -u rtsp_tools/rtsp_server.py \
    --filename "$VIDEO_FILE" \
    --port "$RTSP_OUT_PORT" \
    --mount "/ds-out/$SENSOR_ID" \
    --mode overlay >/tmp/ds_sop_relay.log 2>&1 &
RELAY_PID=$!

echo "Waiting for output RTSP (port $RTSP_OUT_PORT) to be ready..."
for i in $(seq 1 90); do
    if ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "(:|\\.)$RTSP_OUT_PORT$"; then
        echo "Output RTSP is live (took ${i}s)."
        break
    fi
    if ! kill -0 "$RELAY_PID" 2>/dev/null; then
        echo "ERROR: passthrough relay exited before RTSP was ready. Tail of log:" >&2
        tail -n 20 /tmp/ds_sop_relay.log >&2
        exit 1
    fi
    sleep 1
done

if ! ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "(:|\\.)$RTSP_OUT_PORT$"; then
    echo "ERROR: passthrough relay did not bind port $RTSP_OUT_PORT within timeout." >&2
    tail -n 20 /tmp/ds_sop_relay.log >&2
    exit 1
fi

# Give the relay a couple of extra seconds to start producing encoded frames
# before VST subscribes; otherwise the first SETUP can race the first keyframe
# and clients fall into the onDataTimeout loop.
sleep 3

echo "Registering sensor with VST and starting recording..."
python3 -u add-ds-sop-to-vst.py \
    --rtsp_url "rtsp://${HOST_IP}:$RTSP_OUT_PORT/ds-out/$SENSOR_ID" \
    --camera_id "$SENSOR_ID" \
    --vst_endpoint "$VST_ENDPOINT" \
    --record

# The :8554 relay is the long-lived stream VST records/livestreams from, so keep
# it (and this script) alive. The inference client (DS_PID) keeps feeding ELK
# in the background until its own stream timeout.
echo "Streaming test running in foreground now (Ctrl-C to stop). Serving rtsp://${HOST_IP}:$RTSP_OUT_PORT/ds-out/$SENSOR_ID"
wait "$RELAY_PID"

