<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
-->

# Proto Generation

This directory contains Protocol Buffer definitions and generated Python code.

## Prerequisites

You need `protoc` (Protocol Buffer Compiler) installed to generate the Python files.

### Install `protoc`

- **Ubuntu/Debian**:
  ```bash
  sudo apt install protobuf-compiler
  ```

- **Manual**:
  Download the release matching your OS from [GitHub releases](https://github.com/protocolbuffers/protobuf/releases), extract it, and add the `bin` directory to your `PATH`.

## Generating Python Files

To generate or update the Python bindings (`*_pb2.py` files) from the `.proto` definitions, run the following command from inside this directory (`nvds_action_detector/protos`):

```bash
protoc -I. --python_out=. nv.proto ext.proto
```

### Command Explanation

- `-I.`: Adds the current directory to the import search path. This allows `ext.proto` to find `nv.proto`.
- `--python_out=.`: Tells the compiler to write the generated Python files to the current directory.
- `nv.proto ext.proto`: The Protocol Buffer definition files to compile.
