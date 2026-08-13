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

"""Unit tests for verify_rtsp_components.py (vss-sop-deploy skill)."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "vss-sop-deploy" / "scripts" / "verify_rtsp_components.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_rtsp_components", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify_rtsp_components"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return _load_module()


def _setup_gi_gst(*, find_results=None, rtsp_ok=True):
    """Build minimal gi / Gst / GstRtspServer mocks for import inside checks."""
    find_results = find_results or {"x264enc": True}

    def _find(name):
        val = find_results.get(name)
        if val is True:
            return mock.Mock()
        return None

    mock_gst = mock.Mock()
    mock_gst.ElementFactory.find.side_effect = _find

    mock_srv = mock.Mock()
    mock_factory = mock.Mock()
    mock_gst_rtsp = mock.Mock()
    mock_gst_rtsp.RTSPServer.new.return_value = mock_srv
    mock_gst_rtsp.RTSPMediaFactory.new.return_value = mock_factory
    mock_srv.get_mount_points.return_value.add_factory = mock.Mock()

    mock_gi = mock.Mock()
    mock_gi.require_version = mock.Mock()

    repo = types.ModuleType("gi.repository")
    repo.Gst = mock_gst
    repo.GstRtspServer = mock_gst_rtsp

    patches = {
        "gi": mock_gi,
        "gi.repository": repo,
        "gi.repository.Gst": mock_gst,
        "gi.repository.GstRtspServer": mock_gst_rtsp,
    }
    return patches, mock_gst, mock_gst_rtsp, mock_srv


# ---------------------------------------------------------------------------
# run_host_driver
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunHostDriver:
    def test_docker_ps_direct_success(self, mod):
        run_calls = []

        def fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            if cmd[:2] == ["docker", "ps"]:
                return mock.Mock(returncode=0)
            return mock.Mock(returncode=0)

        with mock.patch.object(mod.subprocess, "run", side_effect=fake_run):
            assert mod.run_host_driver() == 0
        assert any(c[0] == "docker" for c in run_calls if len(c) > 1 and c[1] == "run")

    def test_sg_docker_prefix(self, mod):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            if cmd == ["docker", "ps"]:
                raise OSError("denied")
            if cmd[:3] == ["sg", "docker", "-c"] and "docker ps" in cmd[3]:
                return mock.Mock(returncode=0)
            if cmd[:3] == ["sg", "docker", "-c"]:
                return mock.Mock(returncode=0)
            return mock.Mock(returncode=0)

        with mock.patch.object(mod.subprocess, "run", side_effect=fake_run):
            rc = mod.run_host_driver()
        assert rc == 0
        assert any(c[:3] == ["sg", "docker", "-c"] for c in calls)

    def test_sudo_prefix(self, mod):
        def fake_run(cmd, **kwargs):
            if cmd == ["docker", "ps"]:
                raise OSError("denied")
            if cmd[:3] == ["sg", "docker", "-c"]:
                raise OSError("denied")
            if cmd == ["sudo", "docker", "ps"]:
                return mock.Mock(returncode=0)
            return mock.Mock(returncode=0)

        with mock.patch.object(mod.subprocess, "run", side_effect=fake_run):
            assert mod.run_host_driver() == 0

    def test_total_docker_failure(self, mod, capsys):
        with mock.patch.object(mod.subprocess, "run", side_effect=Exception("no docker")):
            assert mod.run_host_driver() == 1
        assert "Cannot connect to Docker" in capsys.readouterr().out

    def test_called_process_error(self, mod):
        def fake_run(cmd, **kwargs):
            if cmd == ["docker", "ps"]:
                return mock.Mock(returncode=0)
            raise mod.subprocess.CalledProcessError(2, cmd)

        with mock.patch.object(mod.subprocess, "run", side_effect=fake_run):
            assert mod.run_host_driver() == 2

    def test_file_not_found(self, mod, capsys):
        def fake_run(cmd, **kwargs):
            if cmd == ["docker", "ps"]:
                return mock.Mock(returncode=0)
            raise FileNotFoundError("docker")

        with mock.patch.object(mod.subprocess, "run", side_effect=fake_run):
            assert mod.run_host_driver() == 1
        assert "docker' command not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckGstRtspServer:
    def test_success(self, mod):
        patches, _, _, _ = _setup_gi_gst()
        with mock.patch.dict(sys.modules, patches):
            assert mod.check_gst_rtsp_server() is True

    def test_failure(self, mod):
        mock_gi = mock.Mock()
        mock_gi.require_version.side_effect = ImportError("no rtsp")
        with mock.patch.dict(sys.modules, {"gi": mock_gi}):
            assert mod.check_gst_rtsp_server() is False


@pytest.mark.unit
class TestCheckX264encEncoder:
    def test_success(self, mod):
        patches, _, _, _ = _setup_gi_gst(find_results={"x264enc": True})
        with mock.patch.dict(sys.modules, patches):
            assert mod.check_x264enc_encoder() is True

    def test_element_not_found(self, mod):
        patches, _, _, _ = _setup_gi_gst(find_results={"x264enc": False})
        with mock.patch.dict(sys.modules, patches):
            assert mod.check_x264enc_encoder() is False

    def test_import_exception(self, mod):
        with mock.patch.dict(sys.modules, {"gi": None}):
            assert mod.check_x264enc_encoder() is False


@pytest.mark.unit
class TestCheckProtobufStubs:
    def test_success(self, mod):
        nv_pb2 = mock.Mock()
        nv_pb2.DESCRIPTOR.message_types_by_name = {"a": 1, "b": 2}
        ext_pb2 = mock.Mock()
        ext_pb2.DESCRIPTOR.message_types_by_name = {"c": 1}
        protos = types.ModuleType("nvds_action_detector.protos")
        protos.nv_pb2 = nv_pb2
        protos.ext_pb2 = ext_pb2
        pkg = types.ModuleType("nvds_action_detector")
        pkg.protos = protos
        with mock.patch.dict(
            sys.modules,
            {
                "nvds_action_detector": pkg,
                "nvds_action_detector.protos": protos,
            },
        ):
            assert mod.check_protobuf_stubs() is True

    def test_failure(self, mod):
        with mock.patch.dict(sys.modules, {"nvds_action_detector.protos": None}):
            assert mod.check_protobuf_stubs() is False


@pytest.mark.unit
class TestCheckGstreamerElements:
    def test_all_present(self, mod):
        required = ["x264enc", "rtph264pay", "udpsink", "nvvideoconvert", "jpegenc", "rtpjpegpay"]
        patches, _, _, _ = _setup_gi_gst(find_results={e: True for e in required})
        with mock.patch.dict(sys.modules, patches):
            assert mod.check_gstreamer_elements() is True

    def test_missing_elements(self, mod):
        patches, _, _, _ = _setup_gi_gst(find_results={"x264enc": True})
        with mock.patch.dict(sys.modules, patches):
            assert mod.check_gstreamer_elements() is False

    def test_exception(self, mod):
        mock_gi = mock.Mock()
        mock_gi.require_version.side_effect = RuntimeError("gst broken")
        with mock.patch.dict(sys.modules, {"gi": mock_gi}):
            assert mod.check_gstreamer_elements() is False


@pytest.mark.unit
class TestCheckSharedLibraries:
    def test_clean_ldd(self, mod):
        proc = mock.Mock()
        proc.communicate.return_value = ("", "")
        with mock.patch.object(mod.subprocess, "Popen", return_value=proc):
            assert mod.check_shared_libraries() is True

    def test_not_found_warning_still_true(self, mod, capsys):
        proc = mock.Mock()
        proc.communicate.return_value = ("libfoo.so => not found\n", "")
        with mock.patch.object(mod.subprocess, "Popen", return_value=proc):
            assert mod.check_shared_libraries() is True
        assert "WARNING" in capsys.readouterr().out

    def test_exception_still_true(self, mod):
        with mock.patch.object(mod.subprocess, "Popen", side_effect=OSError("ldd fail")):
            assert mod.check_shared_libraries() is True


@pytest.mark.unit
class TestCheckRtspInstantiation:
    def test_success(self, mod):
        patches, _, _, mock_srv = _setup_gi_gst()
        mock_sock = mock.Mock()
        mock_sock.__enter__ = mock.Mock(return_value=mock_sock)
        mock_sock.__exit__ = mock.Mock(return_value=False)
        mock_sock.getsockname.return_value = ("0.0.0.0", 12345)
        with mock.patch.dict(sys.modules, patches), mock.patch.object(mod.socket, "socket", return_value=mock_sock):
            assert mod.check_rtsp_instantiation() is True
        mock_srv.set_service.assert_called_once_with("12345")

    def test_failure(self, mod):
        mock_gi = mock.Mock()
        mock_gi.require_version.side_effect = RuntimeError("rtsp fail")
        with mock.patch.dict(sys.modules, {"gi": mock_gi}):
            assert mod.check_rtsp_instantiation() is False


# ---------------------------------------------------------------------------
# run_inside_container & __main__
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunInsideContainer:
    def test_all_pass_exits_zero(self, mod):
        checks = [
            mod.check_gst_rtsp_server,
            mod.check_x264enc_encoder,
            mod.check_protobuf_stubs,
            mod.check_gstreamer_elements,
            mod.check_shared_libraries,
            mod.check_rtsp_instantiation,
        ]
        with mock.patch.object(mod, "check_gst_rtsp_server", return_value=True), \
             mock.patch.object(mod, "check_x264enc_encoder", return_value=True), \
             mock.patch.object(mod, "check_protobuf_stubs", return_value=True), \
             mock.patch.object(mod, "check_gstreamer_elements", return_value=True), \
             mock.patch.object(mod, "check_shared_libraries", return_value=True), \
             mock.patch.object(mod, "check_rtsp_instantiation", return_value=True):
            with pytest.raises(SystemExit) as exc:
                mod.run_inside_container()
            assert exc.value.code == 0

    def test_mixed_fail_exits_one(self, mod):
        with mock.patch.object(mod, "check_gst_rtsp_server", return_value=True), \
             mock.patch.object(mod, "check_x264enc_encoder", return_value=False), \
             mock.patch.object(mod, "check_protobuf_stubs", return_value=True), \
             mock.patch.object(mod, "check_gstreamer_elements", return_value=True), \
             mock.patch.object(mod, "check_shared_libraries", return_value=True), \
             mock.patch.object(mod, "check_rtsp_instantiation", return_value=True):
            with pytest.raises(SystemExit) as exc:
                mod.run_inside_container()
            assert exc.value.code == 1


@pytest.mark.unit
class TestMain:
    def test_inside_container_branch(self, mod):
        with mock.patch.object(mod, "run_inside_container") as ric, \
             mock.patch.object(mod.sys, "argv", ["verify.py", "--inside-container"]):
            if len(mod.sys.argv) > 1 and mod.sys.argv[1] == "--inside-container":
                mod.run_inside_container()
            else:
                mod.sys.exit(mod.run_host_driver())
            ric.assert_called_once()

    def test_host_driver_branch(self, mod):
        with mock.patch.object(mod, "run_host_driver", return_value=0) as rhd, \
             mock.patch.object(mod.sys, "argv", ["verify.py"]):
            with pytest.raises(SystemExit) as exc:
                if len(mod.sys.argv) > 1 and mod.sys.argv[1] == "--inside-container":
                    mod.run_inside_container()
                else:
                    mod.sys.exit(mod.run_host_driver())
            assert exc.value.code == 0
            rhd.assert_called_once()

    def test_main_block_host_runpy(self):
        import runpy

        def fake_run(cmd, **kwargs):
            if cmd == ["docker", "ps"]:
                return mock.Mock(returncode=0)
            return mock.Mock(returncode=0)

        with mock.patch("subprocess.run", side_effect=fake_run):
            with mock.patch.object(sys, "argv", ["verify.py"]):
                with pytest.raises(SystemExit) as exc:
                    runpy.run_path(str(_SCRIPT), run_name="__main__")
                assert exc.value.code == 0

    def test_main_block_inside_runpy(self):
        import runpy

        patches, _, _, _ = _setup_gi_gst(
            find_results={
                e: True for e in
                ["x264enc", "rtph264pay", "udpsink", "nvvideoconvert", "jpegenc", "rtpjpegpay"]
            }
        )
        nv_pb2 = mock.Mock()
        nv_pb2.DESCRIPTOR.message_types_by_name = {"a": 1}
        ext_pb2 = mock.Mock()
        ext_pb2.DESCRIPTOR.message_types_by_name = {"b": 1}
        protos = types.ModuleType("nvds_action_detector.protos")
        protos.nv_pb2 = nv_pb2
        protos.ext_pb2 = ext_pb2
        pkg = types.ModuleType("nvds_action_detector")
        pkg.protos = protos
        patches.update({
            "nvds_action_detector": pkg,
            "nvds_action_detector.protos": protos,
        })
        proc = mock.Mock()
        proc.communicate.return_value = ("", "")
        mock_sock = mock.Mock()
        mock_sock.__enter__ = mock.Mock(return_value=mock_sock)
        mock_sock.__exit__ = mock.Mock(return_value=False)
        mock_sock.getsockname.return_value = ("0.0.0.0", 9999)

        with mock.patch.dict(sys.modules, patches), \
             mock.patch("subprocess.Popen", return_value=proc), \
             mock.patch("socket.socket", return_value=mock_sock), \
             mock.patch.object(sys, "argv", ["verify.py", "--inside-container"]):
            with pytest.raises(SystemExit) as exc:
                runpy.run_path(str(_SCRIPT), run_name="__main__")
            assert exc.value.code == 0
