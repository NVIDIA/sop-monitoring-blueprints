<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
