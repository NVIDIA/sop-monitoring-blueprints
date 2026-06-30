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

"""Unit tests for evaluation-ms ``utils/utils.py``.

Covers the filesystem helpers (``create_dir``, ``create_file``), the
path-traversal guard (``safe_dataset_path``), and the process-tree teardown
(``terminate_process_tree``) which is exercised with mocked ``psutil`` so the
tests never touch real processes.
"""
import os
import sys
from pathlib import Path
from unittest import mock

import psutil
import pytest


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MS_PATH = str(PROJECT_ROOT / "microservices" / "evaluation-ms")
if MS_PATH not in sys.path:
    sys.path.insert(0, MS_PATH)


@pytest.mark.unit
class TestCreateDir:
    def test_creates_missing_directory(self, tmp_path):
        from utils.utils import create_dir

        target = tmp_path / "new_dir"
        assert create_dir(str(target)) is True
        assert target.is_dir()

    def test_creates_nested_directories(self, tmp_path):
        from utils.utils import create_dir

        target = tmp_path / "a" / "b" / "c"
        assert create_dir(str(target)) is True
        assert target.is_dir()

    def test_returns_false_when_directory_exists(self, tmp_path):
        from utils.utils import create_dir

        assert create_dir(str(tmp_path)) is False


@pytest.mark.unit
class TestCreateFile:
    def test_creates_missing_file_and_parents(self, tmp_path):
        from utils.utils import create_file

        target = tmp_path / "sub" / "dir" / "out.txt"
        assert create_file(str(target)) is True
        assert target.is_file()
        assert target.read_text() == ""

    def test_returns_false_when_file_exists(self, tmp_path):
        from utils.utils import create_file

        existing = tmp_path / "exists.txt"
        existing.write_text("keep me")
        assert create_file(str(existing)) is False
        # Existing content must be untouched.
        assert existing.read_text() == "keep me"


@pytest.mark.unit
class TestSafeDatasetPath:
    def test_valid_id_resolves_under_root(self, tmp_path):
        from utils.utils import safe_dataset_path

        root = str(tmp_path)
        result = safe_dataset_path(root, "dataset-123")
        assert result == os.path.realpath(os.path.join(root, "dataset-123"))
        assert result.startswith(os.path.realpath(root) + os.sep)

    @pytest.mark.parametrize(
        "bad_id",
        ["", "..", ".", "a/b", "a\\b", "../escape", "/abs"],
    )
    def test_rejects_separators_and_traversal(self, tmp_path, bad_id):
        from utils.utils import safe_dataset_path

        with pytest.raises(ValueError):
            safe_dataset_path(str(tmp_path), bad_id)

    def test_rejects_symlink_escape(self, tmp_path):
        from utils.utils import safe_dataset_path

        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        # A symlink inside root that resolves outside root must be rejected.
        link = root / "evil"
        link.symlink_to(outside)
        with pytest.raises(ValueError):
            safe_dataset_path(str(root), "evil")


@pytest.mark.unit
class TestTerminateProcessTree:
    def _make_proc(self, pid, running_after=False):
        proc = mock.MagicMock()
        proc.pid = pid
        proc.is_running.return_value = running_after
        return proc

    def test_graceful_termination_success(self):
        from utils.utils import terminate_process_tree

        parent = self._make_proc(100)
        child = self._make_proc(101)
        parent.children.return_value = [child]

        with mock.patch("utils.utils.psutil.Process", return_value=parent), \
                mock.patch("utils.utils.psutil.wait_procs", return_value=([parent, child], [])):
            assert terminate_process_tree(100) is True

        child.terminate.assert_called_once()
        parent.terminate.assert_called_once()
        child.kill.assert_not_called()
        parent.kill.assert_not_called()

    def test_force_kills_survivors(self):
        from utils.utils import terminate_process_tree

        parent = self._make_proc(200)
        survivor = self._make_proc(201)
        parent.children.return_value = [survivor]

        # First wait_procs (graceful) reports survivor still alive; second
        # (after kill) is the cleanup wait.
        with mock.patch("utils.utils.psutil.Process", return_value=parent), \
                mock.patch(
                    "utils.utils.psutil.wait_procs",
                    side_effect=[([parent], [survivor]), ([survivor], [])],
                ):
            assert terminate_process_tree(200) is True

        survivor.kill.assert_called_once()

    def test_returns_false_when_process_still_running(self):
        from utils.utils import terminate_process_tree

        # Parent reports it is still running even after termination attempt.
        parent = self._make_proc(300, running_after=True)
        parent.children.return_value = []

        with mock.patch("utils.utils.psutil.Process", return_value=parent), \
                mock.patch("utils.utils.psutil.wait_procs", return_value=([], [])):
            assert terminate_process_tree(300) is False

    def test_already_gone_parent_returns_true(self):
        from utils.utils import terminate_process_tree

        with mock.patch(
            "utils.utils.psutil.Process",
            side_effect=psutil.NoSuchProcess(pid=404),
        ):
            assert terminate_process_tree(404) is True

    def test_child_disappearing_during_terminate_is_handled(self):
        from utils.utils import terminate_process_tree

        parent = self._make_proc(500)
        child = self._make_proc(501)
        child.terminate.side_effect = psutil.NoSuchProcess(pid=501)
        parent.children.return_value = [child]

        with mock.patch("utils.utils.psutil.Process", return_value=parent), \
                mock.patch("utils.utils.psutil.wait_procs", return_value=([parent, child], [])):
            assert terminate_process_tree(500) is True

    def test_unexpected_error_returns_false(self):
        from utils.utils import terminate_process_tree

        with mock.patch(
            "utils.utils.psutil.Process",
            side_effect=RuntimeError("boom"),
        ):
            assert terminate_process_tree(600) is False

    def test_parent_disappearing_during_terminate_is_handled(self):
        from utils.utils import terminate_process_tree

        parent = self._make_proc(700)
        parent.children.return_value = []
        parent.terminate.side_effect = psutil.NoSuchProcess(pid=700)

        with mock.patch("utils.utils.psutil.Process", return_value=parent), \
                mock.patch("utils.utils.psutil.wait_procs", return_value=([parent], [])):
            assert terminate_process_tree(700) is True

    def test_survivor_disappearing_during_kill_is_handled(self):
        from utils.utils import terminate_process_tree

        parent = self._make_proc(800)
        survivor = self._make_proc(801)
        survivor.kill.side_effect = psutil.NoSuchProcess(pid=801)
        parent.children.return_value = [survivor]

        with mock.patch("utils.utils.psutil.Process", return_value=parent), \
                mock.patch(
                    "utils.utils.psutil.wait_procs",
                    side_effect=[([parent], [survivor]), ([survivor], [])],
                ):
            assert terminate_process_tree(800) is True

    def test_is_running_check_handles_disappearing_process(self):
        from utils.utils import terminate_process_tree

        # is_running() raising NoSuchProcess means the process is gone -> success.
        parent = self._make_proc(900)
        parent.is_running.side_effect = psutil.NoSuchProcess(pid=900)
        parent.children.return_value = []

        with mock.patch("utils.utils.psutil.Process", return_value=parent), \
                mock.patch("utils.utils.psutil.wait_procs", return_value=([parent], [])):
            assert terminate_process_tree(900) is True
