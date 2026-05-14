######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
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
