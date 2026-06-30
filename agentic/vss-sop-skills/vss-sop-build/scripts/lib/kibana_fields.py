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

"""Canonical Kibana / Elasticsearch field-naming rules for the SOP flat-JSON schema.

Single source of truth shared by:
  - vss-sop-build/scripts/verify_build.py   (static sop-kibana-objects.ndjson check)
  - vss-sop-test/scripts/vss_sop_test.py    (live ES mapping + Kibana saved-object check)

SOP emits flat-JSON messages (SOP_MESSAGING_SCHEMA=JSON), so the Kibana data view and
the ES mapping must use flat field names (response.keyword, sensor_id.keyword, @timestamp,
cv_execute_time, ...) instead of the upstream protobuf-style dotted names (llm.queries.response,
sensor.id.keyword, info.cv_execute_time, ...). Keep these rules here only; do not re-hardcode
field lists in the verifier or the test harness.
"""

TIME_FIELD = "@timestamp"
BAD_TIME_FIELD = "timestamp"

# Flat-JSON base field names that MUST exist in the live ES mapping.
FLAT_FIELDS = {
    "response",
    "sensor_id",
    "cv_execute_time",
    "vlm_execute_time",
    "chunk_idx",
    "frame_number",
    "@timestamp",
}

# Protobuf-style nested roots that MUST NOT appear in the live ES mapping.
PROTOBUF_NESTED_ROOTS = {"llm", "sensor"}

# Protobuf-style tokens that MUST NOT appear in the static ndjson data view.
BAD_NDJSON_TOKENS = [
    ("llm.queries.response", "protobuf-style 'llm.queries.response'"),
    ("sensor.id.keyword", "protobuf-style 'sensor.id.keyword'"),
    ('"info.cv_execute_time"', "protobuf-style 'info.cv_execute_time'"),
    ('"info.vlm_execute_time"', "protobuf-style 'info.vlm_execute_time'"),
    ('"info.chunk_idx"', "protobuf-style 'info.chunk_idx'"),
    ('"info.frame_number"', "protobuf-style 'info.frame_number'"),
]

# Flat-JSON tokens that MUST be present in the static ndjson data view.
GOOD_NDJSON_TOKENS = [
    ("response.keyword", "flat JSON field 'response.keyword'"),
    ("sensor_id.keyword", "flat JSON field 'sensor_id.keyword'"),
    ("@timestamp", "time field '@timestamp'"),
]


def scan_ndjson_text(content):
    """Validate static sop-kibana-objects.ndjson content. Returns a list of error strings."""
    errors = []
    for token, desc in BAD_NDJSON_TOKENS:
        if token in content:
            errors.append(f"found incorrect {desc}")
    if f'"timeFieldName": "{BAD_TIME_FIELD}"' in content:
        errors.append(f"timeFieldName should be '{TIME_FIELD}', not '{BAD_TIME_FIELD}'")
    for token, desc in GOOD_NDJSON_TOKENS:
        if token not in content:
            errors.append(f"expected {desc} not found")
    return errors


def scan_runtime_field_map(runtime_map, time_field):
    """Validate a live Kibana index-pattern runtimeFieldMap + timeFieldName.

    `runtime_map` is the raw runtimeFieldMap string from the saved object.
    Returns a list of error strings.
    """
    errors = []
    for token, desc in BAD_NDJSON_TOKENS:
        # The quoted info.* tokens are ndjson-specific; match on the bare dotted name here.
        bare = token.strip('"')
        if bare in runtime_map:
            errors.append(f"runtime fields reference {desc}")
    if time_field == BAD_TIME_FIELD:
        errors.append(f"timeFieldName is '{BAD_TIME_FIELD}' instead of '{TIME_FIELD}'")
    return errors


def scan_mapping_fields(mapping_fields):
    """Validate the live ES mapping field set. Returns (present, missing, bad_present)."""
    present = FLAT_FIELDS & mapping_fields
    missing = FLAT_FIELDS - mapping_fields
    bad_present = PROTOBUF_NESTED_ROOTS & mapping_fields
    return present, missing, bad_present

