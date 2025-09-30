# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
Redis Database utilities for blob storage and retrieval between microservices
"""

import logging

import redis
import redis.asyncio

from .msg_types import STR_ENCODING

_LOGGER = logging.getLogger(__name__)


class RadisDatabase:
    """Simple Redis database interface for blob storage"""

    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client

    def store_blob(self, key: str, blob_data: bytes) -> bool:
        """
        Store binary data (blob) with a key

        Args:
            key: Storage key for the blob
            blob_data: Binary data to store

        Returns:
            True if successful, False otherwise
        """

        result = self.redis_client.set(key, blob_data)
        _LOGGER.debug("Stored blob with key '%s' (size: %d bytes)", key, len(blob_data))
        return bool(result)

    def store_blobs_batch(self, key_blob_pairs: dict[str, bytes]) -> bool:
        """
        Store multiple binary data blobs in a single batch operation

        Args:
            key_blob_pairs: Dictionary mapping keys to blob data

        Returns:
            True if all operations successful, False otherwise
        """
        if not key_blob_pairs:
            return True

        try:
            # Use mset for batch setting multiple key-value pairs
            result = self.redis_client.mset(key_blob_pairs)
            total_size = sum(len(blob) for blob in key_blob_pairs.values())
            _LOGGER.debug("Stored %d blobs in batch (total size: %d bytes)",
                         len(key_blob_pairs), total_size)
            return bool(result)
        except Exception as e:
            _LOGGER.error("Failed to store blobs in batch: %s", e)
            return False

    def get_blob(self, key: str) -> bytes | None:
        """
        Retrieve binary data by key

        Args:
            key: Storage key for the blob

        Returns:
            Binary data if found, None otherwise
        """
        key_bytes = key.encode(STR_ENCODING)
        blob_data = self.redis_client.get(key_bytes)
        if blob_data:
            _LOGGER.debug("Retrieved blob with key '%s' (size: %d bytes)", key, len(blob_data))
        else:
            _LOGGER.debug("No blob found with key '%s'", key)
        return blob_data

    def delete_blob(self, key: str) -> bool:
        """
        Delete a blob by key

        Args:
            key: Storage key to delete

        Returns:
            True if key was deleted, False if key didn't exist or error occurred
        """

        key_bytes = key.encode(STR_ENCODING)
        result = self.redis_client.delete(key_bytes)
        if result:
            _LOGGER.debug("Deleted blob with key '%s'", key)
        else:
            _LOGGER.debug("No blob found to delete with key '%s'", key)
        return bool(result)

    def list_keys(self, pattern: str = "*") -> list[str]:
        """
        List all keys matching a pattern

        Args:
            pattern: Redis pattern to match keys (default "*" for all keys)

        Returns:
            List of matching keys
        """

        keys = self.redis_client.scan_iter(pattern)
        # Decode bytes to strings
        decoded_keys = [key.decode(STR_ENCODING) if isinstance(key, bytes) else key for key in keys]
        _LOGGER.debug("Found %d keys matching pattern '%s'", len(decoded_keys), pattern)
        return decoded_keys

    def get_blobs_batch(self, keys: list[str]) -> dict[str, bytes | None]:
        """
        Retrieve multiple blobs in a single batch operation

        Args:
            keys: List of keys to retrieve

        Returns:
            Dictionary mapping keys to blob data (None if key doesn't exist)
        """
        if not keys:
            return {}

        try:
            # Use mget for batch getting multiple keys
            results = self.redis_client.mget(keys)
            result_dict = dict(zip(keys, results))
            found_count = sum(1 for v in results if v is not None)
            _LOGGER.debug("Retrieved %d/%d blobs in batch", found_count, len(keys))
            return result_dict
        except Exception as e:
            _LOGGER.error("Failed to retrieve blobs in batch: %s", e)
            return {key: None for key in keys}

    def delete_blobs_batch(self, keys: list[str]) -> int:
        """
        Delete multiple blobs in a single batch operation

        Args:
            keys: List of keys to delete

        Returns:
            Number of keys that were actually deleted
        """
        if not keys:
            return 0

        try:
            # Encode keys to bytes for consistency
            key_bytes = [key.encode(STR_ENCODING) for key in keys]
            result = self.redis_client.delete(*key_bytes)
            _LOGGER.debug("Deleted %d/%d blobs in batch", result, len(keys))
            return int(result)
        except Exception as e:
            _LOGGER.error("Failed to delete blobs in batch: %s", e)
            return 0


class AsyncRadisDatabase:
    """Async Redis database interface for blob storage"""

    def __init__(self, redis_client: redis.asyncio.Redis):
        self.redis_client = redis_client

    async def store_blob(self, key: str, blob_data: bytes) -> bool:
        """
        Store binary data (blob) with a key

        Args:
            key: Storage key for the blob
            blob_data: Binary data to store

        Returns:
            True if successful, False otherwise
        """

        result = await self.redis_client.set(key, blob_data)
        _LOGGER.debug("Stored blob with key '%s' (size: %d bytes)", key, len(blob_data))
        return bool(result)

    async def store_blobs_batch(self, key_blob_pairs: dict[str, bytes]) -> bool:
        """
        Store multiple binary data blobs in a single batch operation

        Args:
            key_blob_pairs: Dictionary mapping keys to blob data

        Returns:
            True if all operations successful, False otherwise
        """
        if not key_blob_pairs:
            return True

        try:
            # Use mset for batch setting multiple key-value pairs
            result = await self.redis_client.mset(key_blob_pairs)
            total_size = sum(len(blob) for blob in key_blob_pairs.values())
            _LOGGER.debug("Stored %d blobs in batch (total size: %d bytes)",
                         len(key_blob_pairs), total_size)
            return bool(result)
        except Exception as e:
            _LOGGER.error("Failed to store blobs in batch: %s", e)
            return False

    async def get_blob(self, key: str) -> bytes | None:
        """
        Retrieve binary data by key

        Args:
            key: Storage key for the blob

        Returns:
            Binary data if found, None otherwise
        """
        key_bytes = key.encode(STR_ENCODING)
        blob_data = await self.redis_client.get(key_bytes)
        if blob_data:
            _LOGGER.debug("Retrieved blob with key '%s' (size: %d bytes)", key, len(blob_data))
        else:
            _LOGGER.debug("No blob found with key '%s'", key)
        return blob_data

    async def delete_blob(self, key: str) -> bool:
        """
        Delete a blob by key

        Args:
            key: Storage key to delete

        Returns:
            True if key was deleted, False if key didn't exist or error occurred
        """

        key_bytes = key.encode(STR_ENCODING)
        result = await self.redis_client.delete(key_bytes)
        if result:
            _LOGGER.debug("Deleted blob with key '%s'", key)
        else:
            _LOGGER.debug("No blob found to delete with key '%s'", key)
        return bool(result)

    async def list_keys(self, pattern: str = "*") -> list[str]:
        """
        List all keys matching a pattern

        Args:
            pattern: Redis pattern to match keys (default "*" for all keys)

        Returns:
            List of matching keys
        """

        decoded_keys = []
        async for key in self.redis_client.scan_iter(pattern):
            # Decode bytes to strings
            decoded_key = key.decode(STR_ENCODING) if isinstance(key, bytes) else key
            decoded_keys.append(decoded_key)

        _LOGGER.debug("Found %d keys matching pattern '%s'", len(decoded_keys), pattern)
        return decoded_keys

    async def get_blobs_batch(self, keys: list[str]) -> dict[str, bytes | None]:
        """
        Retrieve multiple blobs in a single batch operation

        Args:
            keys: List of keys to retrieve

        Returns:
            Dictionary mapping keys to blob data (None if key doesn't exist)
        """
        if not keys:
            return {}

        try:
            # Use mget for batch getting multiple keys
            results = await self.redis_client.mget(keys)
            result_dict = dict(zip(keys, results))
            found_count = sum(1 for v in results if v is not None)
            _LOGGER.debug("Retrieved %d/%d blobs in batch", found_count, len(keys))
            return result_dict
        except Exception as e:
            _LOGGER.error("Failed to retrieve blobs in batch: %s", e)
            return {key: None for key in keys}

    async def delete_blobs_batch(self, keys: list[str]) -> int:
        """
        Delete multiple blobs in a single batch operation

        Args:
            keys: List of keys to delete

        Returns:
            Number of keys that were actually deleted
        """
        if not keys:
            return 0

        try:
            # Encode keys to bytes for consistency
            key_bytes = [key.encode(STR_ENCODING) for key in keys]
            result = await self.redis_client.delete(*key_bytes)
            _LOGGER.debug("Deleted %d/%d blobs in batch", result, len(keys))
            return int(result)
        except Exception as e:
            _LOGGER.error("Failed to delete blobs in batch: %s", e)
            return 0
