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

"""Unit tests for the request body-size guard middleware (T17 / FSR-AVA-1).

Drives the async middleware via asyncio.run() from sync tests so the suite does
not depend on pytest-asyncio / asyncio_mode being configured in CI.
"""

import asyncio

from inference import limit_request_body_size


class _Req:
    """Minimal stand-in for a Starlette Request (only .headers is used)."""

    def __init__(self, content_length=None):
        self.headers = {} if content_length is None else {"content-length": content_length}


async def _passthrough(request):
    return "passed-through"


def test_oversized_body_rejected_413():
    resp = asyncio.run(limit_request_body_size(_Req("9999999999999"), _passthrough))
    assert resp.status_code == 413


def test_invalid_content_length_rejected_400():
    resp = asyncio.run(limit_request_body_size(_Req("not-a-number"), _passthrough))
    assert resp.status_code == 400


def test_within_limit_passes_through():
    assert asyncio.run(limit_request_body_size(_Req("100"), _passthrough)) == "passed-through"


def test_no_content_length_passes_through():
    assert asyncio.run(limit_request_body_size(_Req(None), _passthrough)) == "passed-through"
