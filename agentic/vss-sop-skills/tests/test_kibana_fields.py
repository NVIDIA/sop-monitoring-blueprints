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
######################################################################################################

"""Unit tests for kibana_fields.py."""
import pytest


GOOD_NDJSON = """
{"attributes":{"timeFieldName":"@timestamp","fieldAttrs":"response.keyword sensor_id.keyword"}}
"""


@pytest.mark.unit
class TestScanNdjsonText:
    def test_valid_flat_json_content_has_no_errors(self, kibana_fields):
        assert kibana_fields.scan_ndjson_text(GOOD_NDJSON) == []

    def test_detects_protobuf_bad_tokens(self, kibana_fields):
        content = GOOD_NDJSON + "llm.queries.response sensor.id.keyword"
        errors = kibana_fields.scan_ndjson_text(content)
        assert any("llm.queries.response" in e for e in errors)
        assert any("sensor.id.keyword" in e for e in errors)

    def test_detects_info_dotted_bad_tokens(self, kibana_fields):
        content = GOOD_NDJSON + '"info.cv_execute_time" "info.vlm_execute_time"'
        errors = kibana_fields.scan_ndjson_text(content)
        assert any("info.cv_execute_time" in e for e in errors)

    def test_detects_wrong_time_field_name(self, kibana_fields):
        content = GOOD_NDJSON.replace("@timestamp", "x") + '"timeFieldName": "timestamp"'
        errors = kibana_fields.scan_ndjson_text(content)
        assert any("timeFieldName" in e for e in errors)

    def test_detects_missing_good_tokens(self, kibana_fields):
        errors = kibana_fields.scan_ndjson_text('{"attributes":{"timeFieldName":"@timestamp"}}')
        assert any("response.keyword" in e for e in errors)
        assert any("sensor_id.keyword" in e for e in errors)


@pytest.mark.unit
class TestScanRuntimeFieldMap:
    def test_clean_runtime_map(self, kibana_fields):
        assert kibana_fields.scan_runtime_field_map("response.keyword", "@timestamp") == []

    def test_detects_protobuf_in_runtime_map(self, kibana_fields):
        errors = kibana_fields.scan_runtime_field_map("llm.queries.response", "@timestamp")
        assert len(errors) >= 1

    def test_detects_bad_time_field(self, kibana_fields):
        errors = kibana_fields.scan_runtime_field_map("", "timestamp")
        assert any("timestamp" in e for e in errors)


@pytest.mark.unit
class TestScanMappingFields:
    def test_all_flat_fields_present(self, kibana_fields):
        fields = set(kibana_fields.FLAT_FIELDS)
        present, missing, bad = kibana_fields.scan_mapping_fields(fields)
        assert present == kibana_fields.FLAT_FIELDS
        assert missing == set()
        assert bad == set()

    def test_missing_and_bad_protobuf_roots(self, kibana_fields):
        fields = {"response", "llm", "sensor", "@timestamp"}
        present, missing, bad = kibana_fields.scan_mapping_fields(fields)
        assert "response" in present
        assert "llm" in bad
        assert "sensor" in bad
        assert "cv_execute_time" in missing
