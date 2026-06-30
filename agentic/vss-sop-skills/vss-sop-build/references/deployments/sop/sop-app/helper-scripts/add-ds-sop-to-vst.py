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

import requests
import argparse
import json
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

REQUEST_TIMEOUT_SECONDS = 10
STREAM_ID_RETRY_ATTEMPTS = 10
STREAM_ID_RETRY_DELAY_SECONDS = 1
RECORD_START_RETRY_ATTEMPTS = 20
RECORD_START_RETRY_DELAY_SECONDS = 2
PROXIED_URL_RETRY_ATTEMPTS = 15
PROXIED_URL_RETRY_DELAY_SECONDS = 2


def _extract_stream_id_by_name(streams_payload, camera_name):
    """Find VST stream UUID by camera/display name."""
    for entry in streams_payload:
        if not isinstance(entry, dict):
            continue
        for stream_id, stream_list in entry.items():
            if not isinstance(stream_list, list):
                continue
            for stream_info in stream_list:
                if stream_info.get("name") == camera_name:
                    return stream_info.get("streamId") or stream_id
    return None


def _extract_rtsp_host(rtsp_url):
    try:
        return urlparse(rtsp_url).hostname or ""
    except Exception:
        return ""


def _is_loopback_host(host):
    return host in {"127.0.0.1", "localhost", "0.0.0.0", "::1", ""}


def _find_sensor_by_name(sensors_payload, camera_name):
    for sensor in sensors_payload:
        if isinstance(sensor, dict) and sensor.get("name") == camera_name:
            return sensor
    return None


def get_sensor_details(vst_endpoint, camera_id):
    list_url = f"{vst_endpoint}/api/v1/sensor/list"
    try:
        response = requests.get(list_url, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code != 200:
            print(f"Failed to fetch sensor list. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        sensor = _find_sensor_by_name(response.json(), camera_id)
        if sensor:
            print(
                f"Found sensor '{camera_id}' with id '{sensor.get('sensorId')}', "
                f"ip '{sensor.get('sensorIp')}', state '{sensor.get('state')}'."
            )
        return sensor
    except Exception as e:
        print(f"Error fetching sensor details for {camera_id}: {e}")
        return None


def remove_sensor(vst_endpoint, sensor_id):
    remove_url = f"{vst_endpoint}/api/v1/sensor/{sensor_id}"
    try:
        response = requests.delete(remove_url, timeout=REQUEST_TIMEOUT_SECONDS)
        print(f"Remove sensor status code: {response.status_code}")
        if response.status_code != 200:
            print(f"Remove sensor response: {response.text}")
            return False
        print(f"Removed sensor with id {sensor_id}.")
        return True
    except Exception as e:
        print(f"Error removing sensor {sensor_id}: {e}")
        return False


def _post_add_sensor(add_device_url, json_to_send):
    response = requests.post(add_device_url, json=json_to_send, timeout=REQUEST_TIMEOUT_SECONDS)
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text}")
    return response


def _get_proxied_stream_url(vst_endpoint, camera_id):
    """Return the proxied RTSP URL (rtspserver-ms side) for the named camera, or None."""
    streams_url = f"{vst_endpoint}/api/v1/sensor/streams"
    try:
        response = requests.get(streams_url, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code != 200:
            return None
        for entry in response.json():
            if not isinstance(entry, dict):
                continue
            for stream_id, stream_list in entry.items():
                if not isinstance(stream_list, list):
                    continue
                for stream_info in stream_list:
                    if stream_info.get("name") == camera_id:
                        url = stream_info.get("url", "")
                        # Only return the proxied URL (not the source), which is
                        # served by rtspserver-ms on port 30554-30556 range.
                        if url and not _is_loopback_host(_extract_rtsp_host(url)):
                            return url, stream_id
        return None
    except Exception as e:
        print(f"Error fetching proxied stream URL for {camera_id}: {e}")
        return None


def publish_camera_streaming_event(redis_host, redis_port, camera_id, camera_name, proxied_url):
    """Publish a camera_streaming event to Redis vst.event so recorder/livestream SDRs provision the stream."""
    try:
        import redis as redis_lib
        r = redis_lib.Redis(host=redis_host, port=redis_port, decode_responses=True)
        payload = json.dumps({
            "alert_type": "camera_status_change",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event": {
                "camera_id": camera_id,
                "camera_name": camera_name,
                "camera_url": proxied_url,
                "change": "camera_streaming",
                "metadata": {"codec": "h264", "framerate": "", "resolution": ""},
                "tags": "",
            },
            "source": "vst",
        })
        result = r.xadd("vst.event", {"sensor.id": payload})
        print(f"Published camera_streaming event to Redis vst.event (id={result}).")
        return True
    except ImportError:
        print("redis-py not installed; falling back to redis-cli via docker exec.")
        return _publish_via_redis_cli(camera_id, camera_name, proxied_url)
    except Exception as e:
        print(f"Error publishing camera_streaming to Redis: {e}")
        return False


def _publish_via_redis_cli(camera_id, camera_name, proxied_url):
    import subprocess
    payload = json.dumps({
        "alert_type": "camera_status_change",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event": {
            "camera_id": camera_id,
            "camera_name": camera_name,
            "camera_url": proxied_url,
            "change": "camera_streaming",
            "metadata": {"codec": "h264", "framerate": "", "resolution": ""},
            "tags": "",
        },
        "source": "vst",
    })
    cmd = [
        "docker", "exec", "mdx-redis", "redis-cli",
        "XADD", "vst.event", "*", "sensor.id", payload,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"Published camera_streaming via redis-cli (id={result.stdout.strip()}).")
            return True
        print(f"redis-cli XADD failed: {result.stderr.strip()}")
        return False
    except Exception as e:
        print(f"Error running redis-cli: {e}")
        return False


def get_proxied_url_with_retry(vst_endpoint, camera_id):
    """Wait for rtspserver-ms to establish the proxy and return (proxied_url, stream_uuid)."""
    for attempt in range(1, PROXIED_URL_RETRY_ATTEMPTS + 1):
        result = _get_proxied_stream_url(vst_endpoint, camera_id)
        if result:
            url, stream_id = result
            print(f"Got proxied URL for '{camera_id}': {url} (stream_id={stream_id})")
            return url, stream_id
        if attempt < PROXIED_URL_RETRY_ATTEMPTS:
            print(
                f"Waiting for proxied RTSP URL for '{camera_id}' "
                f"({attempt}/{PROXIED_URL_RETRY_ATTEMPTS})..."
            )
            time.sleep(PROXIED_URL_RETRY_DELAY_SECONDS)
    print(f"Proxied URL never appeared for '{camera_id}' after {PROXIED_URL_RETRY_ATTEMPTS} attempts.")
    return None, None


def resolve_stream_id(vst_endpoint, camera_id):
    streams_url = f"{vst_endpoint}/api/v1/sensor/streams"
    try:
        response = requests.get(streams_url, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code != 200:
            print(f"Failed to fetch streams. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None

        stream_id = _extract_stream_id_by_name(response.json(), camera_id)
        if stream_id:
            print(f"Resolved camera name '{camera_id}' to stream UUID '{stream_id}'.")
        else:
            print(f"Could not resolve stream UUID for camera name '{camera_id}'.")
        return stream_id
    except Exception as e:
        print(f"Error resolving stream UUID for {camera_id}: {e}")
        return None


def add_rtsp_stream_to_vst(rtsp_url, camera_id, vst_endpoint):
    json_to_send = {
        "location": "", 
        "name": camera_id, 
        "password": "", 
        "sensorUrl": rtsp_url, 
        "username": ""
    }
    print(f"Adding RTSP stream to VST: {json_to_send}")
    add_device_url = f"{vst_endpoint}/api/v1/sensor/add"
    
    try:
        response = _post_add_sensor(add_device_url, json_to_send)
        
        if response.status_code == 200:
            print(f"Successfully added stream {camera_id} to VST.")
            return True
        if response.status_code == 400 and "already exists" in response.text.lower():
            # Existing entries can become stale (offline/loopback URL) after restarts.
            # Recreate the sensor when needed so recorder/livestream modules re-register it.
            print(f"Stream {camera_id} already exists in VST. Checking if refresh is needed.")
            sensor = get_sensor_details(vst_endpoint, camera_id)
            desired_host = _extract_rtsp_host(rtsp_url)
            sensor_ip = (sensor or {}).get("sensorIp", "")
            sensor_state = (sensor or {}).get("state", "")

            # Only recreate when the sensor was registered with a loopback IP while
            # we want a real host IP. Deleting an otherwise-healthy sensor will
            # disconnect rtspserver-ms from the upstream ds-sop producer, and
            # ds-sop's on-demand RTSP session does NOT automatically restart —
            # subsequent re-adds then fail to establish SDP and no downstream
            # workload (recorder/livestream) ever gets provisioned. See:
            # sdr-http-recorder waits for `camera_streaming`, which is only
            # emitted after rtspserver-ms receives SDP from the upstream proxy.
            needs_recreate = (
                bool(sensor)
                and _is_loopback_host(sensor_ip)
                and not _is_loopback_host(desired_host)
            )

            if needs_recreate:
                sensor_id = sensor.get("sensorId")
                print(
                    f"Existing sensor '{camera_id}' is stale (state={sensor_state}, ip={sensor_ip}). "
                    "Recreating sensor."
                )
                if sensor_id and remove_sensor(vst_endpoint, sensor_id):
                    time.sleep(2)
                    recreate_response = _post_add_sensor(add_device_url, json_to_send)
                    if recreate_response.status_code == 200:
                        print(f"Successfully recreated stream {camera_id} in VST.")
                        return True
                    print(f"Failed to recreate stream. Error: {recreate_response.text}")
                    return False
                print(f"Failed to remove stale sensor '{camera_id}' before recreate.")
                return False

            print(f"Stream {camera_id} already exists and looks healthy. Continuing.")
            return True
        else:
            print(f"Failed to add stream. Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error connecting to VST: {e}")
        return False

_ACTIVE_RECORD_STATES = {"on", "recording", "started", "start", "running", "active"}


def _parse_record_state(payload):
    """Normalize different shapes of /record/status responses to a status string."""
    if isinstance(payload, str):
        return payload.strip().lower()
    if not isinstance(payload, dict):
        return ""
    for key in ("recording_status", "recordingStatus", "status", "state"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value.strip().lower()
    return ""


def is_recording_active(vst_endpoint, stream_id):
    """Return True if VST reports recording is already active for the stream."""
    status_url = f"{vst_endpoint}/api/v1/record/status"
    try:
        response = requests.get(
            status_url,
            params={"sensorId": stream_id},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            return False
        try:
            payload = response.json()
        except ValueError:
            return False

        candidates = []
        if isinstance(payload, list):
            candidates = payload
        elif isinstance(payload, dict):
            # VST returns either a flat status dict or a map keyed by stream UUID.
            if stream_id in payload and isinstance(payload[stream_id], dict):
                candidates = [payload[stream_id]]
            else:
                candidates = [payload]

        for entry in candidates:
            if isinstance(entry, dict):
                sensor_key = entry.get("sensorId") or entry.get("id")
                if sensor_key not in (None, stream_id):
                    continue
            if _parse_record_state(entry) in _ACTIVE_RECORD_STATES:
                return True
        return False
    except Exception as e:
        print(f"Error checking recording status: {e}")
        return False


def set_record_on(vst_endpoint, stream_id, start=True):
    action = "start" if start else "stop"
    record_url = f"{vst_endpoint}/api/v1/record/{stream_id}/{action}"
    print(f"Setting recording to {action} for stream UUID {stream_id}...")

    try:
        response = requests.post(record_url, timeout=REQUEST_TIMEOUT_SECONDS)
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            return True
        print(f"Response: {response.text}")
        # VST auto-starts recording via always_recording once sensor-ms pushes the
        # stream into recorder-ms. Treat that as success so the caller stops retrying.
        if start and is_recording_active(vst_endpoint, stream_id):
            print(
                f"Recording already active for stream UUID {stream_id} "
                "(auto-started by VST always_recording); treating as success."
            )
            return True
        return False
    except Exception as e:
        print(f"Error setting recording state: {e}")
        return False


def resolve_stream_id_with_retry(vst_endpoint, camera_id):
    """Retry stream UUID lookup to handle eventual consistency in VST."""
    for attempt in range(1, STREAM_ID_RETRY_ATTEMPTS + 1):
        stream_id = resolve_stream_id(vst_endpoint, camera_id)
        if stream_id:
            return stream_id
        if attempt < STREAM_ID_RETRY_ATTEMPTS:
            print(
                f"Retrying stream UUID lookup for '{camera_id}' "
                f"({attempt}/{STREAM_ID_RETRY_ATTEMPTS})..."
            )
            time.sleep(STREAM_ID_RETRY_DELAY_SECONDS)
    return None


def set_record_on_with_retry(vst_endpoint, stream_id, start=True):
    """Retry record start/stop for transient VST backend readiness issues."""
    for attempt in range(1, RECORD_START_RETRY_ATTEMPTS + 1):
        success = set_record_on(vst_endpoint, stream_id, start=start)
        if success:
            return True
        if attempt < RECORD_START_RETRY_ATTEMPTS:
            print(
                f"Retrying record {'start' if start else 'stop'} for stream UUID "
                f"{stream_id} ({attempt}/{RECORD_START_RETRY_ATTEMPTS})..."
            )
            time.sleep(RECORD_START_RETRY_DELAY_SECONDS)
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add DS-SOP RTSP stream to VST")
    parser.add_argument("--rtsp_url", required=True, help="RTSP URL from DS-SOP (e.g., rtsp://<ds-sop-ip>:8554/ds-out/camera1)")
    parser.add_argument("--camera_id", required=True, help="Unique ID for the camera in VST")
    parser.add_argument("--vst_endpoint", default="http://localhost:30888/vst", help="VST endpoint URL")
    parser.add_argument("--record", action="store_true", help="Enable recording for this stream")
    parser.add_argument("--redis_host", default="localhost", help="Redis host (default: localhost)")
    parser.add_argument("--redis_port", type=int, default=6379, help="Redis port (default: 6379)")

    args = parser.parse_args()

    success = add_rtsp_stream_to_vst(args.rtsp_url, args.camera_id, args.vst_endpoint)

    if success:
        # After the sensor is added, rtspserver-SDR processes the camera_proxy event and
        # sets up the RTSP proxy. We then publish camera_streaming to vst.event so that
        # recorder/replaystream/livestream SDRs provision the stream with the proxied URL.
        # (In the VST microservices split, camera_streaming is never published automatically.)
        proxied_url, stream_uuid = get_proxied_url_with_retry(args.vst_endpoint, args.camera_id)
        if proxied_url and stream_uuid:
            publish_camera_streaming_event(
                args.redis_host, args.redis_port,
                stream_uuid, args.camera_id, proxied_url,
            )
        else:
            print(
                f"Warning: could not obtain proxied URL for '{args.camera_id}'; "
                "recorder/livestream SDRs may not provision the stream."
            )

    if success and args.record:
        stream_id = resolve_stream_id_with_retry(args.vst_endpoint, args.camera_id)
        if stream_id:
            record_started = set_record_on_with_retry(args.vst_endpoint, stream_id, True)
            if not record_started:
                print(
                    "Unable to start recording after retries for stream UUID "
                    f"'{stream_id}'."
                )
        else:
            print(
                "Unable to start recording because stream UUID could not be resolved "
                f"for camera name '{args.camera_id}' after retries."
            )

