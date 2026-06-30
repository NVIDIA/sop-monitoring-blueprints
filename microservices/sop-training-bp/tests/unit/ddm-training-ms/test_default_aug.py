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

"""
Unit tests for DDM-Net datasets/default_aug.py.
Covers resolve_interpolations, RandomResizePolicy, and the three
transform builders, as well as compose_default_augmentations.
"""

import sys
import warnings
from unittest.mock import MagicMock

# Must be executed before importing default_aug so heavy ML packages are stubbed out.
_ML_MODULES = [
    "torch",
    "torch.nn",
    "torchvision",
    "torchvision.transforms",
    "torchvision.transforms.v2",
    "torchvision.transforms.functional",
]
for _mod in _ML_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# nn.Module must be a real class so RandomResizePolicy(nn.Module) can set instance attributes.
# `import torch.nn as nn` uses IMPORT_FROM which calls getattr(sys.modules["torch"], "nn"),
# NOT sys.modules["torch.nn"] directly. Set both so either lookup returns the real Module.
_nn_mock = MagicMock()
_nn_mock.Module = type("Module", (), {"__init__": lambda self, *a, **kw: None})
sys.modules["torch.nn"] = _nn_mock
sys.modules["torch"].nn = _nn_mock

# Force re-import so RandomResizePolicy is defined with the real Module base above.
sys.modules.pop("datasets.default_aug", None)

import pytest

from datasets.default_aug import (
    resolve_interpolations,
    RandomResizePolicy,
    build_random_resize_transform,
    build_color_jitter_transform,
    build_random_gaussian_blur_transform,
    compose_default_augmentations,
)


class TestResolveInterpolations:
    """Tests for resolve_interpolations."""

    def test_valid_names_return_correct_modes(self):
        """Test that known interpolation names are resolved to the correct number of modes."""
        result = resolve_interpolations(["bilinear", "bicubic"])
        assert len(result) == 2

    def test_unknown_name_issues_warning(self):
        """Test that an unrecognised interpolation name emits a warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_interpolations(["bilinear", "lanczos"])

        assert len(caught) == 1
        assert "lanczos" in str(caught[0].message)

    def test_all_unknown_returns_empty_list(self):
        """Test that all-unknown names return an empty list."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = resolve_interpolations(["lanczos", "box"])

        assert result == []

    def test_duplicates_are_removed(self):
        """Test that duplicate names are deduplicated."""
        result = resolve_interpolations(["bilinear", "bicubic", "bilinear"])
        assert len(result) == 2

    def test_order_is_preserved(self):
        """Test that insertion order is preserved after deduplication."""
        result_bc_bl = resolve_interpolations(["bicubic", "bilinear"])
        result_bl_bc = resolve_interpolations(["bilinear", "bicubic"])
        assert result_bc_bl[0] != result_bl_bc[0]

    def test_case_insensitive(self):
        """Test that name matching is case-insensitive."""
        result_lower = resolve_interpolations(["bilinear"])
        result_upper = resolve_interpolations(["BILINEAR"])
        result_mixed = resolve_interpolations(["Bilinear"])
        assert len(result_lower) == len(result_upper) == len(result_mixed) == 1

    def test_empty_input_returns_empty_list(self):
        """Test that an empty input list returns an empty list."""
        result = resolve_interpolations([])
        assert result == []

    def test_all_four_supported_modes(self):
        """Test that all four supported interpolation modes are resolved correctly."""
        result = resolve_interpolations(["bilinear", "bicubic", "nearest", "nearest_exact"])
        assert len(result) == 4


class TestRandomResizePolicyInit:
    """Tests for RandomResizePolicy.__init__."""

    def test_bilinear_always_included_as_fallback(self):
        """Test that BILINEAR is automatically added even when not specified."""
        from torchvision.transforms.functional import InterpolationMode
        policy = RandomResizePolicy(
            resolution=(224, 224),
            interp_names=["bicubic", "nearest"],
            antialias_prob=0.5,
        )
        assert InterpolationMode.BILINEAR in policy.interps

    def test_bilinear_not_duplicated_if_already_present(self):
        """Test that BILINEAR is not added twice when already present."""
        from torchvision.transforms.functional import InterpolationMode
        policy = RandomResizePolicy(
            resolution=(224, 224),
            interp_names=["bilinear", "bicubic"],
            antialias_prob=0.5,
        )
        bilinear_count = policy.interps.count(InterpolationMode.BILINEAR)
        assert bilinear_count == 1

    def test_antialias_prob_clamped_at_zero_for_negative(self):
        """Test that a negative antialias_prob is clamped to 0."""
        policy = RandomResizePolicy((224, 224), ["bilinear"], antialias_prob=-5.0)
        assert policy.antialias_prob == pytest.approx(0.0)

    def test_antialias_prob_clamped_at_one_for_over_one(self):
        """Test that antialias_prob greater than 1 is clamped to 1."""
        policy = RandomResizePolicy((224, 224), ["bilinear"], antialias_prob=99.0)
        assert policy.antialias_prob == pytest.approx(1.0)

    def test_antialias_disabled_flag_when_prob_is_zero(self):
        """Test that antialias_disabled is True when prob is 0."""
        policy = RandomResizePolicy((224, 224), ["bilinear"], antialias_prob=0.0)
        assert policy.antialias_disabled is True

    def test_antialias_disabled_flag_when_prob_is_positive(self):
        """Test that antialias_disabled is False when prob is positive."""
        policy = RandomResizePolicy((224, 224), ["bilinear"], antialias_prob=0.5)
        assert policy.antialias_disabled is False


class TestBuildRandomResizeTransform:
    """Tests for build_random_resize_transform."""

    def test_returns_two_transforms(self):
        """Test that the function returns exactly two transforms."""
        result = build_random_resize_transform({}, (224, 224))
        assert len(result) == 2

    def test_first_transform_is_random_resize_policy(self):
        """Test that the first transform is a RandomResizePolicy instance, not a Lambda wrapper."""
        result = build_random_resize_transform(
            {"interpolation": ["bilinear"], "antialias_prob": 0.5},
            (224, 224)
        )

        assert isinstance(result[0], RandomResizePolicy)

    def test_todtype_transform_is_called(self):
        """Test that T.ToDtype is called once for float32 conversion."""
        from torchvision.transforms import v2 as T
        T.ToDtype.reset_mock()

        build_random_resize_transform({}, (224, 224))

        T.ToDtype.assert_called_once()

    def test_empty_interp_names_still_builds(self):
        """Test that an empty interpolation list still builds due to BILINEAR fallback."""
        result = build_random_resize_transform({"interpolation": []}, (224, 224))
        assert len(result) == 2

    def test_non_square_resolution_passes_through(self):
        """Test that a non-square resolution is accepted."""
        result = build_random_resize_transform({}, (128, 256))
        assert len(result) == 2


class TestBuildColorJitterTransform:
    """Tests for build_color_jitter_transform."""

    def test_returns_one_transform(self):
        """Test that the function returns exactly one transform."""
        result = build_color_jitter_transform({})
        assert len(result) == 1

    def test_color_jitter_called_with_config_values(self):
        """Test that config values are forwarded correctly to T.ColorJitter."""
        from torchvision.transforms import v2 as T
        T.ColorJitter.reset_mock()

        config = {"brightness": 0.4, "contrast": 0.3, "saturation": 0.2, "hue": 0.1}
        build_color_jitter_transform(config)

        kw = T.ColorJitter.call_args.kwargs
        assert kw["brightness"] == pytest.approx(0.4)
        assert kw["contrast"] == pytest.approx(0.3)
        assert kw["saturation"] == pytest.approx(0.2)
        assert kw["hue"] == pytest.approx(0.1)

    def test_default_values_are_zero(self):
        """Test that missing config keys default to 0."""
        from torchvision.transforms import v2 as T
        T.ColorJitter.reset_mock()

        build_color_jitter_transform({})

        kw = T.ColorJitter.call_args.kwargs
        assert kw["brightness"] == 0
        assert kw["contrast"] == 0
        assert kw["saturation"] == 0
        assert kw["hue"] == 0

    def test_partial_config_uses_defaults_for_missing_keys(self):
        """Test that unspecified keys fall back to 0 when only some keys are given."""
        from torchvision.transforms import v2 as T
        T.ColorJitter.reset_mock()

        build_color_jitter_transform({"brightness": 0.5})

        kw = T.ColorJitter.call_args.kwargs
        assert kw["brightness"] == pytest.approx(0.5)
        assert kw["hue"] == 0


class TestBuildRandomGaussianBlurTransform:
    """Tests for build_random_gaussian_blur_transform."""

    def test_odd_kernel_size_succeeds(self):
        """Test that an odd kernel size builds successfully."""
        result = build_random_gaussian_blur_transform({"kernel_size": 5})
        assert len(result) == 1

    def test_even_kernel_size_raises_value_error(self):
        """Test that an even kernel size raises ValueError immediately."""
        with pytest.raises(ValueError, match="odd"):
            build_random_gaussian_blur_transform({"kernel_size": 4})

    def test_default_kernel_size_is_valid(self):
        """Test that an empty config does not raise."""
        result = build_random_gaussian_blur_transform({})
        assert len(result) == 1

    def test_apply_prob_above_one_clamped(self):
        """Test that apply_prob greater than 1 is clamped without raising."""
        result = build_random_gaussian_blur_transform({"kernel_size": 3, "apply_prob": 99.0})
        assert len(result) == 1

    def test_apply_prob_negative_clamped(self):
        """Test that a negative apply_prob is clamped without raising."""
        result = build_random_gaussian_blur_transform({"kernel_size": 3, "apply_prob": -1.0})
        assert len(result) == 1


class TestComposeDefaultAugmentations:
    """Tests for compose_default_augmentations."""

    def test_empty_config_does_not_raise(self):
        """Test that an empty config runs without error."""
        compose_default_augmentations({}, (224, 224))

    def test_all_augmentations_enabled_does_not_raise(self):
        """Test that enabling all augmentations simultaneously does not raise."""
        config = {
            "RandomResize": {
                "enabled": True,
                "interpolation": ["bilinear", "bicubic"],
                "antialias_prob": 0.5,
            },
            "ColorJitter": {
                "enabled": True,
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1,
            },
            "GaussianBlur": {
                "enabled": True,
                "kernel_size": 3,
                "sigma": [0.1, 0.5],
                "apply_prob": 0.5,
            },
        }
        compose_default_augmentations(config, (224, 224))

    def test_even_kernel_in_gaussian_blur_raises(self):
        """Test that an even GaussianBlur kernel size raises even when called via compose."""
        config = {
            "GaussianBlur": {
                "enabled": True,
                "kernel_size": 4,
            }
        }
        with pytest.raises(ValueError):
            compose_default_augmentations(config, (224, 224))

    def test_integer_resolution_accepted(self):
        """Test that an integer resolution is accepted."""
        compose_default_augmentations({}, 224)

    def test_non_square_resolution_accepted(self):
        """Test that a non-square resolution tuple is accepted."""
        compose_default_augmentations({}, (128, 256))

    def test_disabled_augmentation_flag_respected(self):
        """Test that enabled=False prevents the augmentation from being added."""
        from torchvision.transforms import v2 as T
        T.ColorJitter.reset_mock()

        config = {
            "ColorJitter": {
                "enabled": False,
                "brightness": 0.5,
            }
        }
        compose_default_augmentations(config, (224, 224))

        T.ColorJitter.assert_not_called()

    def test_random_resize_enabled_uses_policy_not_resize(self):
        """Test that RandomResize enabled inserts a RandomResizePolicy and skips T.Resize."""
        from torchvision.transforms import v2 as T
        T.Resize.reset_mock()
        T.Compose.reset_mock()

        config = {
            "RandomResize": {
                "enabled": True,
                "interpolation": ["bilinear"],
                "antialias_prob": 0.5,
            }
        }
        compose_default_augmentations(config, (224, 224))

        # T.Compose is mocked; inspect the transforms list passed to it instead of result.transforms
        transforms_passed = T.Compose.call_args[0][0]
        T.Resize.assert_not_called()
        assert any(isinstance(t, RandomResizePolicy) for t in transforms_passed)

    def test_random_resize_disabled_uses_regular_resize(self):
        """Test that RandomResize disabled falls back to T.Resize."""
        from torchvision.transforms import v2 as T
        T.Resize.reset_mock()

        compose_default_augmentations({}, (224, 224))

        T.Resize.assert_called_once()

    def test_gaussian_blur_disabled_does_not_call_gaussian_blur(self):
        """Test that GaussianBlur enabled=False does not call T.GaussianBlur."""
        from torchvision.transforms import v2 as T
        T.GaussianBlur.reset_mock()

        config = {
            "GaussianBlur": {
                "enabled": False,
                "kernel_size": 3,
            }
        }
        compose_default_augmentations(config, (224, 224))

        T.GaussianBlur.assert_not_called()

    def test_normalize_always_appended(self):
        """Test that Normalize is always appended to the pipeline regardless of config."""
        from torchvision.transforms import v2 as T
        T.Normalize.reset_mock()

        compose_default_augmentations({}, (224, 224))

        T.Normalize.assert_called_once()
