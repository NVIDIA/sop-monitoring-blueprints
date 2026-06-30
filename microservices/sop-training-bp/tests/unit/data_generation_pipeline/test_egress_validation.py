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

"""Unit tests for the LLM egress SSRF guard (FSR-NET-2 / T10).

Covers ``validate_egress_url``, ``llm_url_env_allows_private`` and
``resolve_llm_base_url`` in ``vlm_aug.utils.helper``. The data_generation_pipeline
conftest mocks the heavy deps (cv2/numpy/moviepy) and puts the vlm_aug source on
sys.path, so the real helper module imports here. IP-literal hosts avoid live DNS;
the unresolvable-host case monkeypatches ``socket.getaddrinfo``.
"""

import socket

import pytest

from vlm_aug.utils.helper import (
    llm_url_env_allows_private,
    resolve_llm_base_url,
    validate_egress_url,
)

API_BASE_URL = "https://integrate.api.nvidia.com/v1"


class TestValidateEgressUrl:
    @pytest.mark.unit
    def test_public_ip_allowed(self):
        url = "https://8.8.8.8/v1"
        assert validate_egress_url(url, allow_private=False) == url

    @pytest.mark.unit
    @pytest.mark.parametrize("url", ["ftp://example.com", "file:///etc/passwd", "not-a-url", "http:///v1"])
    def test_bad_scheme_or_missing_host_rejected(self, url):
        with pytest.raises(ValueError):
            validate_egress_url(url)

    @pytest.mark.unit
    @pytest.mark.parametrize("allow_private", [True, False])
    def test_metadata_link_local_always_blocked(self, allow_private):
        # 169.254.169.254 (cloud metadata) must be refused even when private is allowed.
        with pytest.raises(ValueError):
            validate_egress_url("http://169.254.169.254/latest/meta-data", allow_private=allow_private)

    @pytest.mark.unit
    @pytest.mark.parametrize("host", ["10.0.0.5", "192.168.1.10", "127.0.0.1"])
    def test_private_loopback_blocked_by_default(self, host):
        with pytest.raises(ValueError):
            validate_egress_url(f"http://{host}:9000/v1", allow_private=False)

    @pytest.mark.unit
    @pytest.mark.parametrize("host", ["10.0.0.5", "192.168.1.10", "127.0.0.1"])
    def test_private_loopback_allowed_when_opted_in(self, host):
        url = f"http://{host}:9000/v1"
        assert validate_egress_url(url, allow_private=True) == url

    @pytest.mark.unit
    def test_unresolvable_host_rejected(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise socket.gaierror("name resolution failed")

        monkeypatch.setattr(socket, "getaddrinfo", _boom)
        with pytest.raises(ValueError):
            validate_egress_url("https://does-not-exist.invalid/v1")


class TestLlmUrlEnvAllowsPrivate:
    @pytest.mark.unit
    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv("ALLOW_PRIVATE_LLM_URL", val)
        assert llm_url_env_allows_private() is True

    @pytest.mark.unit
    @pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "maybe"])
    def test_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv("ALLOW_PRIVATE_LLM_URL", val)
        assert llm_url_env_allows_private() is False

    @pytest.mark.unit
    def test_unset_is_false(self, monkeypatch):
        monkeypatch.delenv("ALLOW_PRIVATE_LLM_URL", raising=False)
        assert llm_url_env_allows_private() is False


class TestResolveLlmBaseUrl:
    @pytest.mark.unit
    def test_nvidia_returns_fixed_endpoint_without_validation(self):
        # NVIDIA path ignores local_llm_url entirely and never SSRF-validates.
        assert resolve_llm_base_url("nvidia", "http://169.254.169.254/v1", API_BASE_URL) == API_BASE_URL

    @pytest.mark.unit
    def test_local_allows_private(self):
        url = "http://11.20.18.86:9000/v1"
        assert resolve_llm_base_url("local", url, API_BASE_URL) == url

    @pytest.mark.unit
    def test_local_still_blocks_metadata(self):
        with pytest.raises(ValueError):
            resolve_llm_base_url("local", "http://169.254.169.254/v1", API_BASE_URL)

    @pytest.mark.unit
    def test_non_local_private_blocked_without_env(self, monkeypatch):
        monkeypatch.delenv("ALLOW_PRIVATE_LLM_URL", raising=False)
        with pytest.raises(ValueError):
            resolve_llm_base_url("custom", "http://10.0.0.5:9000/v1", API_BASE_URL)

    @pytest.mark.unit
    def test_non_local_private_allowed_with_env(self, monkeypatch):
        monkeypatch.setenv("ALLOW_PRIVATE_LLM_URL", "true")
        url = "http://10.0.0.5:9000/v1"
        assert resolve_llm_base_url("custom", url, API_BASE_URL) == url


class TestDnsRebindingPinning:
    """validate_egress_url pins the validated IP for http DNS-name hosts (T10 TOCTOU)."""

    def _patch_dns(self, monkeypatch, ip):
        def fake_getaddrinfo(host, port, *args, **kwargs):
            fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
            return [(fam, socket.SOCK_STREAM, 6, "", (ip, port or 0))]
        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    @pytest.mark.unit
    def test_http_hostname_pinned_to_resolved_ip(self, monkeypatch):
        self._patch_dns(monkeypatch, "8.8.8.8")
        # public IP, allow_private not needed; hostname must be replaced by the IP
        assert validate_egress_url("http://llm.example.com:9000/v1") == "http://8.8.8.8:9000/v1"

    @pytest.mark.unit
    def test_http_hostname_private_pinned_when_allowed(self, monkeypatch):
        self._patch_dns(monkeypatch, "10.0.0.5")
        assert validate_egress_url("http://vllm.local:9000/v1", allow_private=True) == "http://10.0.0.5:9000/v1"

    @pytest.mark.unit
    def test_https_hostname_left_unchanged(self, monkeypatch):
        # https must keep the hostname so TLS SNI/cert validation still works
        self._patch_dns(monkeypatch, "8.8.8.8")
        url = "https://llm.example.com:9000/v1"
        assert validate_egress_url(url) == url

    @pytest.mark.unit
    def test_ip_literal_returned_unchanged(self, monkeypatch):
        # already an IP — nothing to re-resolve, returned verbatim
        self._patch_dns(monkeypatch, "10.0.0.5")  # should not even be consulted for the substitution
        url = "http://10.0.0.5:9000/v1"
        assert validate_egress_url(url, allow_private=True) == url

    @pytest.mark.unit
    def test_hostname_resolving_to_metadata_blocked(self, monkeypatch):
        self._patch_dns(monkeypatch, "169.254.169.254")
        with pytest.raises(ValueError):
            validate_egress_url("http://innocent.example.com/v1", allow_private=True)

    @pytest.mark.unit
    def test_http_hostname_ipv6_pinned_bracketed(self, monkeypatch):
        # IPv6 resolution must be pinned and bracketed in the URL
        self._patch_dns(monkeypatch, "2001:4860:4860::8888")
        assert validate_egress_url("http://llm.example.com:9000/v1") == "http://[2001:4860:4860::8888]:9000/v1"

    @pytest.mark.unit
    def test_http_hostname_pinned_preserves_userinfo(self, monkeypatch):
        # user:pass@ must survive IP pinning
        self._patch_dns(monkeypatch, "10.0.0.5")
        assert (
            validate_egress_url("http://user:pass@vllm.local:9000/v1", allow_private=True)
            == "http://user:pass@10.0.0.5:9000/v1"
        )
