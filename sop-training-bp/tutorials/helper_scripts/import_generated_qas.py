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
import argparse
import asyncio
import asyncpg
import json
import os
import sys
from datetime import datetime

DB_URL = os.environ.get("DATABASE_URL", "postgresql://sop:sop@metadata_db:5432/sop_db")
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
    ds_dir = os.path.join(BASE_DIR, ds_rel_path)
    if not os.path.isdir(ds_dir):
        print(f"ERROR: Augmented dataset directory not found: {ds_dir}")
        sys.exit(1)

    augmented_dataset_id = args.augmented_dataset_id or os.path.basename(ds_rel_path)
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
