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

"""Coverage for ddm_inference.run_ddm_inference window/padding logic.

torch + torchvision are imported lazily inside the function and aren't
installed in the test environment, so we inject fakes via ``sys.modules`` and
patch ``_decode_video_pyav`` / ``_model_window_scores``. The real logic under
test — window batching, the final-partial-batch rule, and edge zero-padding so
``len(scores) == total_frames`` — runs unchanged.
"""
import contextlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MS_PATH = str(PROJECT_ROOT / "microservices" / "evaluation-ms")
if MS_PATH not in sys.path:
    sys.path.insert(0, MS_PATH)


@contextlib.contextmanager
def fake_torch_and_torchvision(cuda_available=False):
    """Install minimal fake torch + torchvision.transforms.v2 for the duration."""
    torch = types.ModuleType("torch")
    torch.float32 = "float32"

    @contextlib.contextmanager
    def _no_grad():
        yield

    torch.no_grad = _no_grad
    torch.stack = MagicMock(return_value=MagicMock(name="batch_tensor"))
    torch.cuda = types.SimpleNamespace(
        is_available=MagicMock(return_value=cuda_available),
        empty_cache=MagicMock(),
    )

    tv = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")
    v2 = types.ModuleType("torchvision.transforms.v2")
    v2.Compose = MagicMock(return_value=MagicMock(name="transform"))
    v2.ToDtype = MagicMock()
    v2.Normalize = MagicMock()
    transforms.v2 = v2
    tv.transforms = transforms

    injected = {
        "torch": torch,
        "torchvision": tv,
        "torchvision.transforms": transforms,
        "torchvision.transforms.v2": v2,
    }
    # Save any preexisting entries so cleanup restores them rather than
    # unconditionally deleting. torch/torchvision aren't installed in this
    # test env, but this keeps the helper correct if they ever become test
    # deps and avoids clobbering a fake left by another test in the session.
    previous = {k: sys.modules.get(k) for k in injected}
    sys.modules.update(injected)
    try:
        yield torch
    finally:
        for k, old in previous.items():
            if old is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = old


def make_meta(total_frames, fps=30.0):
    from utils.ddm_inference import VideoMetadata

    return VideoMetadata(fps=fps, duration_sec=total_frames / fps, total_frames=total_frames)


@pytest.mark.unit
class TestRunDdmInference:
    def test_returns_all_zeros_when_too_few_frames(self):
        from utils.ddm_inference import run_ddm_inference

        meta = make_meta(total_frames=4)  # < window_size = 2*5+1 = 11
        with fake_torch_and_torchvision():
            with patch("utils.ddm_inference._decode_video_pyav",
                       return_value=(MagicMock(), meta)):
                scores, returned_meta = run_ddm_inference(
                    MagicMock(), "v.mp4",
                    resolution=224, frames_per_side=5, batch_size=4, device="cpu",
                )
        assert scores == [0.0] * 4
        assert returned_meta is meta

    def test_even_batches_pad_to_total_frames(self):
        from utils.ddm_inference import run_ddm_inference

        # frames_per_side=2 -> window_size=5; total=12 -> 8 windows; batch_size=4
        # -> two full batches, no final partial.
        meta = make_meta(total_frames=12)
        side = [np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.5, 0.6, 0.7, 0.8])]
        with fake_torch_and_torchvision():
            with patch("utils.ddm_inference._decode_video_pyav",
                       return_value=(MagicMock(), meta)), \
                 patch("utils.ddm_inference._model_window_scores", side_effect=side):
                scores, _ = run_ddm_inference(
                    MagicMock(), "v.mp4",
                    resolution=224, frames_per_side=2, batch_size=4, device="cpu",
                )
        assert len(scores) == 12
        assert scores[:2] == [0.0, 0.0]          # pad_start
        assert scores[-2:] == [0.0, 0.0]         # pad_end
        assert scores[2:10] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    def test_final_partial_batch_of_two_is_scored(self):
        from utils.ddm_inference import run_ddm_inference

        # window_size=5; total=11 -> 7 windows; batch_size=4 -> one batch of 4
        # then a final partial batch of 3 (>= 2, so it IS scored).
        meta = make_meta(total_frames=11)
        side = [np.array([0.1, 0.1, 0.1, 0.1]), np.array([0.2, 0.2, 0.2])]
        with fake_torch_and_torchvision():
            with patch("utils.ddm_inference._decode_video_pyav",
                       return_value=(MagicMock(), meta)), \
                 patch("utils.ddm_inference._model_window_scores", side_effect=side):
                scores, _ = run_ddm_inference(
                    MagicMock(), "v.mp4",
                    resolution=224, frames_per_side=2, batch_size=4, device="cpu",
                )
        assert len(scores) == 11
        # 7 scored windows between the 2-frame pads.
        assert scores[2:9] == pytest.approx([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2])

    def test_final_single_window_is_skipped(self):
        from utils.ddm_inference import run_ddm_inference

        # window_size=5; total=9 -> 5 windows; batch_size=4 -> one batch of 4,
        # then a leftover of 1 window (< 2) which is skipped (matches upstream).
        meta = make_meta(total_frames=9)
        side = [np.array([0.3, 0.3, 0.3, 0.3])]
        with fake_torch_and_torchvision():
            with patch("utils.ddm_inference._decode_video_pyav",
                       return_value=(MagicMock(), meta)), \
                 patch("utils.ddm_inference._model_window_scores", side_effect=side) as mws:
                scores, _ = run_ddm_inference(
                    MagicMock(), "v.mp4",
                    resolution=224, frames_per_side=2, batch_size=4, device="cpu",
                )
        assert mws.call_count == 1  # leftover single window not scored
        assert len(scores) == 9
        # 2-frame start pad, the 4 scored windows, then 3 trailing zeros
        # (the skipped single window contributes no score).
        assert scores[2:6] == pytest.approx([0.3, 0.3, 0.3, 0.3])
        assert scores[6:] == [0.0, 0.0, 0.0]

    def test_cuda_device_triggers_empty_cache(self):
        from utils.ddm_inference import run_ddm_inference

        meta = make_meta(total_frames=12)
        side = [np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.5, 0.6, 0.7, 0.8])]
        with fake_torch_and_torchvision(cuda_available=True) as torch:
            with patch("utils.ddm_inference._decode_video_pyav",
                       return_value=(MagicMock(), meta)), \
                 patch("utils.ddm_inference._model_window_scores", side_effect=side):
                run_ddm_inference(
                    MagicMock(), "v.mp4",
                    resolution=224, frames_per_side=2, batch_size=4, device="cuda:0",
                )
        torch.cuda.empty_cache.assert_called_once()
