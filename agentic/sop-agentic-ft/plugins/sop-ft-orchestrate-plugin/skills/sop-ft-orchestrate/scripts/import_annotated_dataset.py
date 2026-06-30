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

"""
General-purpose script to import a pre-annotated dataset folder into the SOP Training BP database.

This script takes any annotated dataset directory (from the annotation pipeline or prepared
offline) and injects it into the PostgreSQL metadata DB so it appears in the UI and can be
used directly with the Data Augmentation and Training microservices.

Supported dataset formats:
  - UUID-based video dirs, chunk names with 4 parts (action_video_rep_timeline.mp4)
  - Simple video dirs (video-N-pass), chunk names with 3 parts (action_video_rep.MP4)
  Any folder with actions.json + video subdirectories containing *_annotation.json will work.

Prerequisites:
  - The SOP Training BP services must be running (docker compose up)
  - The dataset folder must be inside assets/data/ (mounted into containers)

Usage:
  # Copy script into the annotation-backend container and run it:
  docker cp scripts/import_annotated_dataset.py sop-annotation-backend:/tmp/
  docker exec sop-annotation-backend python3 /tmp/import_annotated_dataset.py <dataset_folder_name>

  # Examples:
  docker exec sop-annotation-backend python3 /tmp/import_annotated_dataset.py dataset/dataset_train_10
  docker exec sop-annotation-backend python3 /tmp/import_annotated_dataset.py dataset

  # With custom dataset ID (default: folder basename):
  docker exec sop-annotation-backend python3 /tmp/import_annotated_dataset.py dataset/dataset_train_10 --dataset-id my-dataset-train

  # Force re-import (deletes existing dataset first):
  docker exec sop-annotation-backend python3 /tmp/import_annotated_dataset.py dataset --force

After import:
  The script prints the dataset_id. Use this ID to launch augmentation:
    curl -X POST 'http://<server_ip>:5487/api/v1/augment?label_data_id=<dataset_id>'
"""
import argparse
import asyncio
import asyncpg
import json
import os
import sys
import uuid
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
VIDEO_EXTENSIONS = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV"}
UUID_RE_PATTERN = r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})_"


def find_source_video(ds_dir: str, folder_name: str) -> str | None:
    """
    Find the source video file that corresponds to a video chunk folder.

    Convention: the source video has the same name as the folder + a video extension.
    E.g., folder "video-25-failed" -> "video-25-failed.mp4"
          folder "08d524fb-..._tp00303_7" -> "08d524fb-..._tp00303_7.mp4"
    """
    for ext in VIDEO_EXTENSIONS:
        candidate = folder_name + ext
        full_path = os.path.join(ds_dir, candidate)
        if os.path.isfile(full_path):
            return candidate
    return None


def extract_uuid_from_filename(filename: str) -> str | None:
    """Extract leading UUID from a filename, or return None."""
    import re
    m = re.match(UUID_RE_PATTERN, filename, re.IGNORECASE)
    return m.group(1) if m else None


def find_chunk_file(video_dir: str, chunk_name: str) -> str | None:
    """
    Locate the actual file on disk that corresponds to chunk_name from the annotation JSON.

    Handles two naming conventions:
      - Exact match: chunk_name directly matches a file
      - Prefix match: chunk_name is a 3-part name and actual file has 4 parts
        e.g., "11_video-1-pass_1.MP4" matches "11_video-1-pass_1_2.mp4"
    """
    try:
        files = os.listdir(video_dir)
    except OSError:
        return None

    # 1. Exact match (case-insensitive)
    for f in files:
        if f.lower() == chunk_name.lower() and os.path.isfile(os.path.join(video_dir, f)):
            return f

    # 2. Prefix match: strip extension, append "_" to find longer filenames
    name_no_ext = os.path.splitext(chunk_name)[0]
    prefix = (name_no_ext + "_").lower()
    for f in files:
        if f.lower().startswith(prefix) and os.path.isfile(os.path.join(video_dir, f)):
            if os.path.splitext(f)[1].lower() in {e.lower() for e in VIDEO_EXTENSIONS}:
                return f

    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import a pre-annotated dataset folder into the SOP Training BP database."
    )
    parser.add_argument(
        "dataset_path",
        help="Path to the dataset folder relative to assets/data/ (e.g., 'dataset/dataset_train_10' or 'dataset')",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Custom dataset ID to use in the DB (default: basename of dataset_path)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing dataset with the same ID before importing",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # Resolve dataset directory
    ds_rel_path = args.dataset_path.strip("/")
    # Confine the resolved dataset directory under BASE_DIR — reject '..'/absolute
    # traversal so a malicious --dataset-path cannot escape the dataset root (T15).
    base_real = os.path.realpath(BASE_DIR)
    ds_dir = os.path.realpath(os.path.join(BASE_DIR, ds_rel_path))
    if ds_dir != base_real and not ds_dir.startswith(base_real + os.sep):
        print(f"ERROR: dataset_path escapes the dataset root: {args.dataset_path!r}")
        sys.exit(1)

    if not os.path.isdir(ds_dir):
        print(f"ERROR: Dataset directory not found: {ds_dir}")
        sys.exit(1)

    # Determine dataset ID (used as a symlink name + DB id — reject path separators / traversal)
    dataset_id = args.dataset_id or os.path.basename(ds_rel_path)
    if not dataset_id or "/" in dataset_id or "\\" in dataset_id or dataset_id in (".", ".."):
        print(f"ERROR: invalid dataset id: {dataset_id!r}")
        sys.exit(1)

    # If the dataset is in a subdirectory (e.g., dataset/dataset_train_10), create a symlink
    # at the top level so the augmentation service can find it by dataset_id
    if "/" in ds_rel_path and dataset_id != ds_rel_path:
        symlink_path = os.path.join(BASE_DIR, dataset_id)
        if not os.path.exists(symlink_path):
            os.symlink(ds_rel_path, symlink_path)
            print(f"  Created symlink: {dataset_id} -> {ds_rel_path}")
        elif os.path.islink(symlink_path):
            print(f"  Symlink already exists: {dataset_id} -> {os.readlink(symlink_path)}")

    # Read actions.json
    actions_file = os.path.join(ds_dir, "actions.json")
    if not os.path.exists(actions_file):
        print(f"ERROR: actions.json not found in {ds_dir}")
        sys.exit(1)

    with open(actions_file) as f:
        actions = json.load(f)["actions"]

    print(f"Dataset: {dataset_id}")
    print(f"  Path: {ds_dir}")
    print(f"  Actions: {len(actions)}")

    # Connect to DB
    conn = await asyncpg.connect(DB_URL)

    try:
        # Check if dataset already exists
        existing = await conn.fetchval("SELECT id FROM dataset WHERE id=$1", dataset_id)
        if existing:
            if args.force:
                print(f"  Deleting existing dataset '{dataset_id}' (--force)")
                await conn.execute("DELETE FROM dataset WHERE id=$1", dataset_id)
            else:
                print(f"ERROR: Dataset '{dataset_id}' already exists in DB. Use --force to overwrite.")
                sys.exit(1)

        now = datetime.now()

        # Insert dataset
        await conn.execute(
            "INSERT INTO dataset (id, actions, created_at, updated_at) VALUES ($1, $2::varchar[], $3, $4)",
            dataset_id, actions, now, now,
        )

        video_count = 0
        chunk_count = 0
        warn_count = 0

        # Walk top-level subdirectories — each is one video pass
        for entry in sorted(os.listdir(ds_dir)):
            video_dir = os.path.join(ds_dir, entry)
            if not os.path.isdir(video_dir):
                continue

            # Skip non-video directories (config_to_bcq, golden_gqa_to_gqas, etc.)
            annotation_file = None
            for fname in sorted(os.listdir(video_dir)):
                if fname.endswith("_annotation.json"):
                    annotation_file = os.path.join(video_dir, fname)
                    break

            if not annotation_file:
                continue

            with open(annotation_file) as f:
                annotations = json.load(f)

            if not annotations:
                print(f"  WARN: {entry} — empty annotation JSON, skipping")
                continue

            # Find source video file and determine video_id / video_name
            source_video = find_source_video(ds_dir, entry)
            if source_video:
                video_name = source_video
                video_size = os.path.getsize(os.path.join(ds_dir, source_video))
                # extract UUID from filename; generate new UUID
                existing_uuid = extract_uuid_from_filename(source_video)
                if existing_uuid:
                    # Check if this UUID is already used by another dataset
                    conflict = await conn.fetchval(
                        "SELECT dataset_id FROM video WHERE id=$1", existing_uuid
                    )
                    if conflict and conflict != dataset_id:
                        print(f"  WARN: video UUID {existing_uuid} already used by dataset '{conflict}', generating new UUID")
                        video_id = str(uuid.uuid4())
                    else:
                        video_id = existing_uuid
                else:
                    video_id = str(uuid.uuid4())
            else:
                # Fallback: no source video found, use sum of chunk sizes
                print(f"  WARN: no source video for folder '{entry}', using synthetic entry")
                chunk_file_names = [
                    fn for fn in os.listdir(video_dir)
                    if os.path.isfile(os.path.join(video_dir, fn))
                    and os.path.splitext(fn)[1] in VIDEO_EXTENSIONS
                ]
                video_size = sum(
                    os.path.getsize(os.path.join(video_dir, fn)) for fn in chunk_file_names
                )
                video_id = str(uuid.uuid4())
                video_name = f"{video_id}_{entry}.mp4"

            await conn.execute(
                "INSERT INTO video (id, dataset_id, name, mime_type, file_size, created_at, updated_at)"
                " VALUES ($1, $2, $3, $4, $5, $6, $7)",
                video_id, dataset_id, video_name, "video/mp4", video_size, now, now,
            )
            video_count += 1

            for ann in annotations:
                action_index = ann["action"]
                description = ann["description"]
                chunk_name_json = ann["chunk_name"]
                start_time = float(ann["start_timestamp"])
                end_time = float(ann["end_timestamp"])

                actual_file = find_chunk_file(video_dir, chunk_name_json)
                if not actual_file:
                    print(f"  WARN: chunk not found: {chunk_name_json} in {video_dir}")
                    warn_count += 1
                    continue

                chunk_id = str(uuid.uuid4())
                chunk_size = os.path.getsize(os.path.join(video_dir, actual_file))

                await conn.execute(
                    "INSERT INTO chunk (id, video_id, name, action, mime_type, file_size, created_at, updated_at)"
                    " VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                    chunk_id, video_id, actual_file, description, "video/mp4", chunk_size, now, now,
                )

                ann_id = str(uuid.uuid4())
                await conn.execute(
                    "INSERT INTO annotation (id, video_id, chunk_id, start_time, end_time, action_index,"
                    " action_description, created_at, updated_at)"
                    " VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
                    ann_id, video_id, chunk_id, start_time, end_time, action_index, description, now, now,
                )
                chunk_count += 1

            print(f"  Video: {entry}  ({len(annotations)} chunks)")

        print(f"\nImport complete:")
        print(f"  Dataset ID: {dataset_id}")
        print(f"  Videos: {video_count}")
        print(f"  Chunks/Annotations: {chunk_count}")
        if warn_count:
            print(f"  Warnings: {warn_count}")
        print(f"\nTo run augmentation:")
        print(f"  curl -X POST 'http://<server_ip>:5487/api/v1/augment?label_data_id={dataset_id}'")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
