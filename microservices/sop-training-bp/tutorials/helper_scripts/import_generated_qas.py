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
import argparse
import asyncio
import asyncpg
import json
import os
import sys
from datetime import datetime
from urllib.parse import quote

# DB connection: prefer an explicit DATABASE_URL; otherwise build it from the
# POSTGRES_* env the services inject — no hard-coded default credentials (T07).
# user/password are URL-encoded so special characters in the password are safe.
DB_URL = os.environ.get("DATABASE_URL") or (
    "postgresql://"
    f"{quote(os.environ.get('POSTGRES_USER', 'sop'), safe='')}:"
    f"{quote(os.environ.get('POSTGRES_PASSWORD', ''), safe='')}@"
    f"{os.environ.get('POSTGRES_HOST', 'metadata_db')}:5432/"
    f"{os.environ.get('POSTGRES_DB', 'sop_db')}"
)
BASE_DIR = os.environ.get("VIDEOS_DIR", "/app/assets/videos")

# Map output folder name (as written to disk by the Augmentation service) ->
# stage_name stored in the augmentation_stages table.
# Source of truth: microservices/data-generation-pipeline/utils/constant.py +
#                  microservices/data-generation-pipeline/app.py: process_all_actions
FOLDER_TO_STAGE = {
    "bcq":         "bcq",
    "mcq":         "sequential_mcq",
    "golden_gqa":  "golden_gqa",
    "gqas":        "gqas",
    "dmcq":        "dynamic_mcq",
    "ds":          "dynamic_shuffling",
    "en":          "extra_negative",
}


def detect_stages(ds_dir: str):
    """Return the list of stage folder names that look like valid augmentation outputs.

    Each valid stage folder contains `<folder>.json` and a `videos/` subdirectory.
    """
    detected = []
    for entry in sorted(os.listdir(ds_dir)):
        stage_dir = os.path.join(ds_dir, entry)
        if not os.path.isdir(stage_dir):
            continue
        if entry not in FOLDER_TO_STAGE:
            print(f"  WARN: unknown stage folder '{entry}' - skipping")
            continue
        if not os.path.isfile(os.path.join(stage_dir, f"{entry}.json")):
            print(f"  WARN: missing {entry}.json in {stage_dir} - skipping")
            continue
        if not os.path.isdir(os.path.join(stage_dir, "videos")):
            print(f"  WARN: missing videos/ in {stage_dir} - skipping")
            continue
        detected.append(entry)
    return detected


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import a pre-generated augmented QA dataset into the SOP Training BP database."
    )
    parser.add_argument(
        "dataset_path",
        help="Path to the augmented dataset folder relative to assets/data/ "
             "(e.g. 'server_fan_train_augmented_0').",
    )
    parser.add_argument(
        "--label-data-id",
        required=True,
        help="Parent label dataset id (must already exist, e.g. imported via import_dataset.sh).",
    )
    parser.add_argument(
        "--augmented-dataset-id",
        default=None,
        help="Override the augmented dataset id (default: basename of dataset_path).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing augmented dataset rows with the same id before importing.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    ds_rel_path = args.dataset_path.strip("/")
    # Confine the resolved dataset directory under BASE_DIR — reject '..'/absolute
    # traversal so a malicious --dataset-path cannot escape the dataset root (T15).
    base_real = os.path.realpath(BASE_DIR)
    ds_dir = os.path.realpath(os.path.join(BASE_DIR, ds_rel_path))
    if ds_dir != base_real and not ds_dir.startswith(base_real + os.sep):
        print(f"ERROR: dataset_path escapes the dataset root: {args.dataset_path!r}")
        sys.exit(1)
    if not os.path.isdir(ds_dir):
        print(f"ERROR: Augmented dataset directory not found: {ds_dir}")
        sys.exit(1)

    augmented_dataset_id = args.augmented_dataset_id or os.path.basename(ds_rel_path)
    if not augmented_dataset_id or "/" in augmented_dataset_id or "\\" in augmented_dataset_id or augmented_dataset_id in (".", ".."):
        print(f"ERROR: invalid augmented dataset id: {augmented_dataset_id!r}")
        sys.exit(1)
    label_data_id = args.label_data_id

    stages = detect_stages(ds_dir)
    if not stages:
        print(f"ERROR: No valid augmentation stage folders found under {ds_dir}")
        sys.exit(1)

    print(f"Augmented dataset: {augmented_dataset_id}")
    print(f"  Parent label data id: {label_data_id}")
    print(f"  Path: {ds_dir}")
    print(f"  Stages: {', '.join(f'{f} ({FOLDER_TO_STAGE[f]})' for f in stages)}")

    conn = await asyncpg.connect(DB_URL)
    try:
        async with conn.transaction():
            parent = await conn.fetchval("SELECT id FROM dataset WHERE id=$1", label_data_id)
            if not parent:
                print(
                    f"ERROR: parent dataset '{label_data_id}' not found in `dataset` table. "
                    "Run import_dataset.sh for it first."
                )
                sys.exit(1)

            existing = await conn.fetchval(
                "SELECT id FROM augmented_data WHERE id=$1", augmented_dataset_id
            )
            if existing:
                if args.force:
                    print(f"  Deleting existing augmented dataset '{augmented_dataset_id}' (--force)")
                    # augmentation_stages has ON DELETE CASCADE -> stage rows removed automatically
                    await conn.execute("DELETE FROM augmented_data WHERE id=$1", augmented_dataset_id)
                else:
                    print(
                        f"ERROR: augmented dataset '{augmented_dataset_id}' already exists. "
                        "Use --force to overwrite."
                    )
                    sys.exit(1)

            now = datetime.now()
            parameters_payload = {
                "imported": True,
                "source_path": ds_rel_path,
                "stages": [FOLDER_TO_STAGE[f] for f in stages],
            }

            await conn.execute(
                "INSERT INTO augmented_data (id, dataset_id, parameters, status, created_at, updated_at)"
                " VALUES ($1, $2, $3::json, $4::status_enum, $5, $6)",
                augmented_dataset_id,
                label_data_id,
                json.dumps(parameters_payload),
                "completed",
                now, now,
            )

            for folder_name in stages:
                stage_name = FOLDER_TO_STAGE[folder_name]
                stage_id = f"{augmented_dataset_id}_{stage_name}"
                await conn.execute(
                    "INSERT INTO augmentation_stages"
                    " (id, augmentation_id, stage_name, status, created_at, updated_at)"
                    " VALUES ($1, $2, $3, $4::status_enum, $5, $6)",
                    stage_id, augmented_dataset_id, stage_name, "completed", now, now,
                )

            print("\nImport complete:")
            print(f"  Augmented dataset id: {augmented_dataset_id}")
            print(f"  Parent dataset id:    {label_data_id}")
            print(f"  Stages imported:      {len(stages)}")
            print("\nUse this id when starting VLM fine-tuning:")
            print(
                f"  curl -X POST 'http://<server_ip>:32080/api/v1/fine-tuning/start"
                f"?dataset_id={augmented_dataset_id}'"
            )
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
