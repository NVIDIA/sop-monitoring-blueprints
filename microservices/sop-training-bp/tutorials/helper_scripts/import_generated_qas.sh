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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPORT_SCRIPT="${SCRIPT_DIR}/import_generated_qas.py"

# Find the annotation-backend container (works with any project name)
CONTAINER=$(docker ps --format '{{.Names}}' | grep 'annotation-backend' | head -1)
if [[ -z "${CONTAINER}" ]]; then
    echo "ERROR: No annotation-backend container is running."
    echo "  Start services first: source .env && docker compose up -d"
    exit 1
fi

echo "Using container: ${CONTAINER}"

# Copy and run the import script
docker cp "${IMPORT_SCRIPT}" "${CONTAINER}:/tmp/import_generated_qas.py"
docker exec "${CONTAINER}" python3 /tmp/import_generated_qas.py "$@"
