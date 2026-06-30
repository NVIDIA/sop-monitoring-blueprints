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

"""Coverage for ddm_inference.load_ddm_model and _model_window_scores.

These functions import torch / torchvision / the vendored DDM-Net ``modeling``
package lazily (inside the function body), and none of those are installed in
the test environment. We inject lightweight fakes via ``sys.modules`` so the
function logic — checkpoint validation, state_dict prefix-stripping, the
softmax window scoring — runs without a GPU or the real DDM-Net code.

All injection goes through ``patched_modules``, a context manager that records
the prior ``sys.modules`` state and restores it on exit (removing keys that
weren't present, reinstating ones that were) so no fake leaks into other tests
even if the function under test raises.
"""
import contextlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MS_PATH = str(PROJECT_ROOT / "microservices" / "evaluation-ms")
if MS_PATH not in sys.path:
    sys.path.insert(0, MS_PATH)


@contextlib.contextmanager
def patched_modules(modules):
    """Install ``modules`` into sys.modules, restoring prior state on exit."""
    previous = {k: sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for k, old in previous.items():
            if old is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = old


# ── Fake builders (pure construction; they do NOT touch sys.modules) ──────────

def make_fake_torch(load_return=None):
    """Minimal fake ``torch`` exposing what load_ddm_model touches."""
    torch = types.ModuleType("torch")

    @contextlib.contextmanager
    def _safe_globals(_allowed):
        yield

    torch.serialization = types.SimpleNamespace(safe_globals=_safe_globals)
    torch.load = MagicMock(return_value=load_return)
    return torch


def build_fake_modeling(missing=None, unexpected=None):
    """Build a fake ``modeling.resnetGEBD`` package. Returns (model, modules)."""
    model = MagicMock(name="resnetGEBD_instance")
    model.load_state_dict.return_value = (missing or [], unexpected or [])
    model.to.return_value = model
    model.eval.return_value = model

    pkg = types.ModuleType("modeling")
    sub = types.ModuleType("modeling.resnetGEBD")
    sub.resnetGEBD = MagicMock(return_value=model)
    return model, {"modeling": pkg, "modeling.resnetGEBD": sub}


class _FakeTensor:
    """Stands in for the tensor returned by torch.nn.functional.softmax."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def __getitem__(self, _idx):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


def build_fake_torch_functional(softmax_values):
    """Build fake torch / torch.nn / torch.nn.functional. Returns modules dict."""
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    func = types.ModuleType("torch.nn.functional")
    func.softmax = lambda output, dim: _FakeTensor(softmax_values)
    nn.functional = func
    torch.nn = nn
    return {"torch": torch, "torch.nn": nn, "torch.nn.functional": func}


# ── load_ddm_model ──────────────────────────────────────────────────────────

@pytest.mark.unit
class TestLoadDdmModel:
    def test_missing_checkpoint_raises(self, tmp_path):
        from utils.ddm_inference import load_ddm_model

        with patched_modules({"torch": make_fake_torch()}):
            with pytest.raises(FileNotFoundError):
                load_ddm_model(str(tmp_path / "nope.pth"), frames_per_side=5, device="cpu")

    def test_loads_and_strips_model_and_module_prefixes(self, tmp_path):
        from utils.ddm_inference import load_ddm_model

        ckpt = tmp_path / "ddm.pth"
        ckpt.write_bytes(b"stub")
        state = {"state_dict": {"model.backbone.w": 1, "module.head.b": 2, "plain": 3}}

        model, modeling_mods = build_fake_modeling(missing=["m1"], unexpected=["u1"])
        mods = {"torch": make_fake_torch(load_return=state), **modeling_mods}
        with patched_modules(mods):
            returned = load_ddm_model(str(ckpt), frames_per_side=5, device="cpu")

        # The prefix-stripped dict is what gets loaded.
        loaded = model.load_state_dict.call_args.args[0]
        assert loaded == {"backbone.w": 1, "head.b": 2, "plain": 3}
        # to(device).eval() chain returns the model we hand back.
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once()
        assert returned is model

    def test_raw_state_dict_without_state_dict_key(self, tmp_path):
        from utils.ddm_inference import load_ddm_model

        ckpt = tmp_path / "ddm.pth"
        ckpt.write_bytes(b"stub")
        # No "state_dict" wrapper -> the dict is used as-is (else branch).
        raw = {"backbone.w": 1}

        model, modeling_mods = build_fake_modeling()
        mods = {"torch": make_fake_torch(load_return=raw), **modeling_mods}
        with patched_modules(mods):
            load_ddm_model(str(ckpt), frames_per_side=3, device="cpu")

        assert model.load_state_dict.call_args.args[0] == raw


# ── _model_window_scores ──────────────────────────────────────────────────────

@pytest.mark.unit
class TestModelWindowScores:
    def test_tuple_result_with_list_outputs(self):
        from utils.ddm_inference import _model_window_scores

        # model returns (outputs, _, _) where outputs is a list -> use last.
        last = object()
        model = MagicMock(return_value=([object(), last], None, None))

        with patched_modules(build_fake_torch_functional([0.2, 0.8])):
            scores = _model_window_scores(model, batch="ignored")

        assert isinstance(scores, np.ndarray)
        assert scores.tolist() == pytest.approx([0.2, 0.8])

    def test_plain_tensor_result(self):
        from utils.ddm_inference import _model_window_scores

        # model returns a bare tensor (not a tuple/list) -> used directly.
        model = MagicMock(return_value=object())

        with patched_modules(build_fake_torch_functional([0.5, 0.5])):
            scores = _model_window_scores(model, batch="ignored")

        assert scores.tolist() == pytest.approx([0.5, 0.5])
