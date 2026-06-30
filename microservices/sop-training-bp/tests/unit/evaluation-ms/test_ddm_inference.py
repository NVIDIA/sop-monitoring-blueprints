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

"""Unit tests for utils.ddm_inference.

Covers the pure-Python helpers (detect_boundaries, calculate_chunk_boundaries)
and the PyAV decode path (_decode_video_pyav) — including the resize-at-decode
optimisation that prevents CUDA OOM on long native-resolution videos.

The PyAV-dependent tests synthesise small in-memory mp4 files via PyAV's own
muxer, so they don't need any test fixtures on disk.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MS_PATH = str(PROJECT_ROOT / "microservices" / "evaluation-ms")
if MS_PATH not in sys.path:
    sys.path.insert(0, MS_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_synthetic_mp4(path: str, num_frames: int, width: int, height: int, fps: int = 30):
    """Write a deterministic test mp4 with `num_frames` solid-colour frames.

    Frame i has channel values (i*5, i*7, i*11) mod 256 so each frame is
    distinct but the result is small (mostly compresses well in libx264).
    """
    import av

    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for i in range(num_frames):
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[:, :, 0] = (i * 5) % 256
        arr[:, :, 1] = (i * 7) % 256
        arr[:, :, 2] = (i * 11) % 256
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _write_empty_mp4(path: str, width: int = 64, height: int = 64, fps: int = 30):
    """Write an mp4 container with a video stream but zero encoded frames."""
    import av

    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    # Don't encode any frames; flush the encoder so the stream has trailing
    # packets but no decodable frames.
    for packet in stream.encode():
        container.mux(packet)
    container.close()


# ---------------------------------------------------------------------------
# detect_boundaries / calculate_chunk_boundaries — pure Python
# ---------------------------------------------------------------------------

class TestDetectBoundaries:
    def test_local_max_only(self):
        """A single peak above threshold returns one boundary at the peak index."""
        from utils.ddm_inference import detect_boundaries
        scores = [0.1, 0.6, 0.9, 0.6, 0.1]
        assert detect_boundaries(scores, threshold=0.5, nms_size=2) == [2]

    def test_threshold_filter(self):
        """No score above threshold returns an empty list."""
        from utils.ddm_inference import detect_boundaries
        scores = [0.1, 0.2, 0.4, 0.3, 0.0]
        assert detect_boundaries(scores, threshold=0.5, nms_size=2) == []

    def test_two_peaks_outside_nms(self):
        """Two well-separated peaks both pass NMS."""
        from utils.ddm_inference import detect_boundaries
        scores = [0.1, 0.9, 0.1, 0.1, 0.1, 0.9, 0.1]
        # nms_size=1 keeps both peaks since they're 4 apart.
        assert detect_boundaries(scores, threshold=0.5, nms_size=1) == [1, 5]

    def test_nms_suppresses_neighbouring_peak(self):
        """When two peaks fall within nms_size, only the larger survives."""
        from utils.ddm_inference import detect_boundaries
        scores = [0.0, 0.7, 0.9, 0.0]
        # i=1: window [0,3) argmax=2 (0.9), 2!=1 → drop.
        # i=2: window [0,4) argmax=2 → keep.
        assert detect_boundaries(scores, threshold=0.5, nms_size=2) == [2]

    def test_empty_scores(self):
        """Empty input produces empty output."""
        from utils.ddm_inference import detect_boundaries
        assert detect_boundaries([], threshold=0.5, nms_size=2) == []


class TestCalculateChunkBoundaries:
    def test_full_video_with_boundaries(self):
        """Two interior boundaries split the video into three chunks."""
        from utils.ddm_inference import calculate_chunk_boundaries
        starts, ends = calculate_chunk_boundaries(
            boundaries=[60, 120], fps=30.0, duration_sec=5.0, total_frames=150,
        )
        assert starts == [0.0, 2.0, 4.0]
        assert ends == [2.0, 4.0, 5.0]

    def test_no_boundaries(self):
        """No boundaries means one chunk spanning the whole duration."""
        from utils.ddm_inference import calculate_chunk_boundaries
        starts, ends = calculate_chunk_boundaries(
            boundaries=[], fps=30.0, duration_sec=5.0, total_frames=150,
        )
        assert starts == [0.0]
        assert ends == [5.0]

    def test_fps_fractional(self):
        """Fractional fps still produces correct second-domain boundaries."""
        from utils.ddm_inference import calculate_chunk_boundaries
        starts, ends = calculate_chunk_boundaries(
            boundaries=[24, 48], fps=23.976, duration_sec=4.0, total_frames=96,
        )
        assert starts[0] == 0.0
        assert ends[-1] == 4.0
        assert starts[1] == pytest.approx(24 / 23.976, rel=1e-6)
        assert ends[0] == pytest.approx(24 / 23.976, rel=1e-6)


# ---------------------------------------------------------------------------
# _decode_video_pyav — needs PyAV + numpy + torch
# ---------------------------------------------------------------------------

@pytest.fixture
def small_mp4(tmp_path):
    """5-frame 320x240 30fps mp4 for decode tests."""
    p = tmp_path / "small.mp4"
    _write_synthetic_mp4(str(p), num_frames=5, width=320, height=240, fps=30)
    assert p.exists() and p.stat().st_size > 0
    return str(p)


class TestDecodeVideoPyAV:
    def test_native_resolution_shape(self, small_mp4):
        """Default decode preserves source resolution."""
        import torch
        from utils.ddm_inference import _decode_video_pyav
        tensor, meta = _decode_video_pyav(small_mp4)

        assert tensor.shape[0] >= 1  # at least one frame decoded
        assert tensor.shape[1] == 3
        assert tensor.shape[2] == 240
        assert tensor.shape[3] == 320
        assert tensor.dtype == torch.uint8
        assert meta.fps == pytest.approx(30.0, rel=1e-3)
        assert meta.total_frames >= 1

    def test_target_resolution_resizes(self, small_mp4):
        """When target_resolution is set, frames come back at that square size."""
        import torch
        from utils.ddm_inference import _decode_video_pyav
        tensor, meta = _decode_video_pyav(small_mp4, target_resolution=224)

        assert tensor.shape[1] == 3
        assert tensor.shape[2] == 224
        assert tensor.shape[3] == 224
        assert tensor.dtype == torch.uint8
        assert meta.fps == pytest.approx(30.0, rel=1e-3)

    def test_target_resolution_smaller_than_native(self, small_mp4):
        """A typical DDM 224x224 input is much smaller than the source 320x240."""
        from utils.ddm_inference import _decode_video_pyav
        native, _ = _decode_video_pyav(small_mp4)
        resized, _ = _decode_video_pyav(small_mp4, target_resolution=224)

        # Same frame count but resized has fewer total pixels.
        assert native.shape[0] == resized.shape[0]
        native_pixels = native.shape[2] * native.shape[3]
        resized_pixels = resized.shape[2] * resized.shape[3]
        assert resized_pixels < native_pixels

    def test_target_resolution_upscales(self, small_mp4):
        """target_resolution > native works (libswscale upsamples)."""
        from utils.ddm_inference import _decode_video_pyav
        # 320x240 source → 480x480 target exercises the upsampling code path.
        tensor, _ = _decode_video_pyav(small_mp4, target_resolution=480)
        assert tensor.shape[1] == 3
        assert tensor.shape[2] == 480
        assert tensor.shape[3] == 480

    def test_raises_on_empty_video(self, tmp_path):
        """A container with no decodable frames raises a precise error.

        Two valid outcomes depending on the PyAV/libav build:
          (a) PyAV rejects the malformed empty container at av.open with an
              FFmpegError (an OSError subclass); our code never runs.
          (b) PyAV opens it cleanly but no frames decode; our code raises
              RuntimeError("No frames decoded from ...").
        Both are correct. The test asserts the exception type is one of
        these and, if it's our RuntimeError, that the message is precise.
        """
        import av
        from utils.ddm_inference import _decode_video_pyav

        p = tmp_path / "empty.mp4"
        _write_empty_mp4(str(p))

        with pytest.raises((RuntimeError, av.error.FFmpegError, OSError)) as excinfo:
            _decode_video_pyav(str(p))

        if isinstance(excinfo.value, RuntimeError):
            assert "No frames decoded" in str(excinfo.value)

    def test_raises_on_missing_file(self):
        """Nonexistent path raises an av FFmpegError (OSError-derived)."""
        import av
        from utils.ddm_inference import _decode_video_pyav
        with pytest.raises((av.error.FFmpegError, FileNotFoundError, OSError)):
            _decode_video_pyav("/nonexistent/path/that/does/not/exist.mp4")

    # NOTE: We considered a `test_metadata_handles_missing_average_rate` to
    # exercise the `fps = 0.0` fallback in _decode_video_pyav, but PyAV's
    # `stream.average_rate` is a read-only property — monkeypatching the
    # instance attribute is silently ignored. Cleanly testing that branch
    # would require either mocking the entire `av.open` chain (too brittle)
    # or sourcing a real fixture mp4 with a missing avg_rate (overkill for
    # one if-branch). The branch itself is two lines and obviously correct.


@pytest.fixture
def long_mp4(tmp_path):
    """60-frame 320x240 30fps mp4 = 2.0s duration; used for tail-trim tests."""
    p = tmp_path / "long.mp4"
    _write_synthetic_mp4(str(p), num_frames=60, width=320, height=240, fps=30)
    assert p.exists() and p.stat().st_size > 0
    return str(p)


class TestDecodeVideoPyAVEndTimestampCap:
    """Cover the ``end_timestamp_sec`` early-stop path added for the
    server-side tail-trim feature (replaces the removed host-side
    trim_videos.py preprocessing). Cap-based stopping must:

      (a) be a no-op when the cap is None or beyond the video duration
          (baseline behavior preserved);
      (b) cut decoding when the cap lies mid-video and report
          ``total_frames`` consistent with the returned tensor;
      (c) leave the tensor and metadata internally consistent so the
          downstream window-loop in run_ddm_inference doesn't index past
          the end of the array.
    """

    def test_no_cap_equivalent_to_baseline(self, long_mp4):
        """end_timestamp_sec=None behaves identically to the pre-cap call."""
        from utils.ddm_inference import _decode_video_pyav
        baseline_tensor, baseline_meta = _decode_video_pyav(long_mp4)
        capped_tensor, capped_meta = _decode_video_pyav(
            long_mp4, end_timestamp_sec=None
        )
        assert baseline_tensor.shape == capped_tensor.shape
        assert baseline_meta.total_frames == capped_meta.total_frames
        assert baseline_meta.fps == capped_meta.fps

    def test_cap_beyond_duration_no_trim(self, long_mp4):
        """A cap larger than the video duration yields the full decode.

        Tensor size and total_frames must match the no-cap path. This is
        the common case for well-annotated datasets where the annotation
        end_timestamp already covers the whole mp4 (e.g. server_assemble_test):
        the cap is set, but the loop never breaks.
        """
        from utils.ddm_inference import _decode_video_pyav
        baseline_tensor, baseline_meta = _decode_video_pyav(long_mp4)
        # Cap well past the 2.0s duration.
        capped_tensor, capped_meta = _decode_video_pyav(
            long_mp4, end_timestamp_sec=999.0
        )
        assert capped_tensor.shape[0] == baseline_tensor.shape[0]
        assert capped_meta.total_frames == baseline_tensor.shape[0]

    def test_cap_mid_video_trims_to_cap(self, long_mp4):
        """A cap mid-video stops decoding; tensor matches the decoded count."""
        from utils.ddm_inference import _decode_video_pyav
        # The synthetic mp4 is 60 frames at 30 fps = 2.0s; cap at 1.0s
        # should yield ~30 frames (give or take one for boundary effects).
        capped_tensor, capped_meta = _decode_video_pyav(
            long_mp4, end_timestamp_sec=1.0
        )
        # Tensor and meta must agree on count (downstream invariant).
        assert capped_tensor.shape[0] == capped_meta.total_frames
        # Trim landed in the expected ballpark — definitely less than the
        # full 60 frames, and at least a few decoded.
        assert 10 <= capped_tensor.shape[0] < 60
        # Duration recomputed from the trimmed frame count.
        assert capped_meta.duration_sec == pytest.approx(
            capped_meta.total_frames / capped_meta.fps, rel=1e-3
        )

    def test_cap_zero_decodes_at_least_one_frame(self, long_mp4):
        """A cap of 0.0s still decodes the first frame (pts=0 doesn't exceed
        the cap; the loop breaks on the NEXT frame). Guards against an
        off-by-one that would error with "No frames decoded".
        """
        from utils.ddm_inference import _decode_video_pyav
        tensor, meta = _decode_video_pyav(long_mp4, end_timestamp_sec=0.0)
        assert tensor.shape[0] >= 1
        assert meta.total_frames == tensor.shape[0]

    def test_cap_with_target_resolution(self, long_mp4):
        """Cap composes correctly with the resize-at-decode path."""
        from utils.ddm_inference import _decode_video_pyav
        tensor, meta = _decode_video_pyav(
            long_mp4, target_resolution=224, end_timestamp_sec=1.0
        )
        assert tensor.shape[1] == 3
        assert tensor.shape[2] == 224
        assert tensor.shape[3] == 224
        assert tensor.shape[0] == meta.total_frames
        assert 10 <= tensor.shape[0] < 60
