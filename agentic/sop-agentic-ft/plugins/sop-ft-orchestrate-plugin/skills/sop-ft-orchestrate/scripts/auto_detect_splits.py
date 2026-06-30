#!/usr/bin/env python3
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

"""Auto-detect train/test split subdirectories in a dataset path.

Usage: auto_detect_splits.py <dataset_path>
Output on success:
  TRAIN=<abs_path>
  EVAL=<abs_path>
Exit 1 with diagnostic on ambiguity or no match.
"""
import os
import sys


def stem(name, suffix):
    return name[: -len(suffix)]


def main():
    if len(sys.argv) != 2:
        print("Usage: auto_detect_splits.py <dataset_path>", file=sys.stderr)
        sys.exit(1)

    dataset_path = sys.argv[1]
    subdirs = sorted(os.listdir(dataset_path))
    train_dirs = [d for d in subdirs if d.endswith("_train")]
    test_dirs = [d for d in subdirs if d.endswith("_test")]

    matched_pairs = [
        (t, e)
        for t in train_dirs
        for e in test_dirs
        if stem(t, "_train") == stem(e, "_test")
    ]

    if len(matched_pairs) == 1:
        train_name, eval_name = matched_pairs[0]
        print(f"TRAIN={os.path.join(dataset_path, train_name)}")
        print(f"EVAL={os.path.join(dataset_path, eval_name)}")
    elif len(matched_pairs) == 0:
        print(
            f"Cannot auto-detect train/test pair in {dataset_path}.\n"
            f"  Found *_train: {train_dirs}\n"
            f"  Found *_test:  {test_dirs}\n"
            "No shared stem match. Set eval_dataset_path explicitly in inputs.yaml.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print(
            f"Multiple matching train/test pairs found in {dataset_path}: {matched_pairs}.\n"
            "Set dataset_path and eval_dataset_path explicitly in inputs.yaml to avoid "
            "training on one assembly and evaluating on another.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
