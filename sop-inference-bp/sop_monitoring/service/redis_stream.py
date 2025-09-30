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
Redis Streams utilities for microservice communication with binary data support
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager, asynccontextmanager
from typing import (
    Generator,
    TypeVar,
    AsyncGenerator,
)

import redis
import redis.asyncio

from .msg_types import (
    RedisStreamMessage,
    RedisStreamRequest,
    RedisStreamResponse,
    STR_ENCODING
)

_LOGGER = logging.getLogger(__name__)

# Type variable for message types inheriting from RedisStreamMessage
_MsgType = TypeVar('MsgType', bound=RedisStreamMessage)
# Type variable for response types inheriting from RedisStreamResponse
_ResponseType = TypeVar('ResponseType', bound=RedisStreamResponse)

def _parse_redis_messages(redis_messages: list[tuple[bytes, list[tuple[bytes, bytes]]]], msg_type: type[_MsgType]) -> list[tuple[str, _MsgType]]:
    """
    Parse Redis messages
    """

    parsed_messages = []
    for stream_data in redis_messages:
        source_stream_name, stream_messages = stream_data
        source_stream_name = source_stream_name.decode(STR_ENCODING)
        for message_id_bytes, message_data in stream_messages:
            message_id = message_id_bytes.decode(STR_ENCODING)
            message = msg_type.from_redis_stream_message(message_data)
            parsed_messages.append((message_id, message))
            _LOGGER.debug("Read message from stream '%s' with ID '%s'", source_stream_name, message_id)

    return parsed_messages


@contextmanager
def auto_delete_stream(redis_stream: RedisStream, stream_name: str) -> Generator[None, None, None]:
    """
    Context manager to automatically delete a Redis stream when exiting the context.

    Args:
        redis_stream: RedisStream instance
        stream_name: Name of the stream to delete

    Example:
        with auto_delete_stream(redis_stream, reply_stream_name):
            # do work with the stream
            pass
        # stream is automatically deleted here
    """
    try:
        yield
    finally:
        redis_stream.delete_stream(stream_name)


def send_request_and_wait_for_response(redis_stream: RedisStream,
                                       request_stream_name: str,
                                       request_message: RedisStreamRequest,
                                       response_msg_type: type[_ResponseType]) -> _ResponseType | RedisStreamResponse:
    """
    Send a request and wait for a response
    """
    block = 0 # must be blocking call
    count = 10 # max number of responses to wait for

    request_id = redis_stream.send_message(request_stream_name, request_message)
    _LOGGER.debug("Sent request %s to stream %s, with reply_stream_name %s", request_id, request_stream_name, request_message.reply_stream_name)

    with auto_delete_stream(redis_stream, request_message.reply_stream_name):
        responses = redis_stream.xread(request_message.reply_stream_name, response_msg_type, count, block, "0")
        if not responses:
            _LOGGER.error("No response received from stream '%s' after %d milliseconds", request_message.reply_stream_name, block)
            return RedisStreamResponse(request_id=request_id, error_message="No response received")

        if len(responses) > 1:
            _LOGGER.error("Received multiple responses from stream '%s' for request %s.", request_message.reply_stream_name, request_id)
            return RedisStreamResponse(request_id=request_id, error_message="Multiple responses received")

        response_id, response = responses[0]

        if response.request_id != request_id:
            _LOGGER.error("Received response %s for request %s, but the waiting request ID is %s", response_id, response.request_id, request_id)
            return RedisStreamResponse(request_id="error", error_message="Response/Request ID mismatch. This should not happen.")

    _LOGGER.debug("Received response %s for request %s", response_id, request_id)
    return response


def send_request_and_get_response_generator(redis_stream: RedisStream,
                                            request_stream_name: str,
                                            request_message: RedisStreamRequest,
                                            response_msg_type: type[_ResponseType]) -> Generator[list[_ResponseType], bool, None]:
    """
    Send a request and get a generator of responses
    """
    count = 10 # max number of responses to wait for
    block = 0 # must be blocking call

    request_id = redis_stream.send_message(request_stream_name, request_message)
    _LOGGER.debug("Sent request %s to stream %s, with reply_stream_name %s",
                  request_id,
                  request_stream_name,
                  request_message.reply_stream_name)

    last_id = "0"
    should_continue = True
    with auto_delete_stream(redis_stream, request_message.reply_stream_name):
        while should_continue:
            responses = redis_stream.xread(request_message.reply_stream_name, response_msg_type, count, block, last_id)
            if not responses:
                _LOGGER.error("No response received from stream '%s'. This should not happen. Something wrong.", request_message.reply_stream_name)
                break

            response_list = []
            for response_id, response in responses:
                _LOGGER.debug("Collecting response %s", response_id)
                response_list.append(response)
                last_id = response_id

            _LOGGER.debug("Yielding response list %s", response_list)
            should_continue = yield response_list
            _LOGGER.debug("Yielded response list %s, should_continue: %s", response_list, should_continue)


def send_response(redis_stream: RedisStream,
                  received_request: RedisStreamRequest,
                  response: RedisStreamResponse) -> None:
    """
    Send a response to the request through the reply stream
    """
    response_id = redis_stream.send_message(received_request.reply_stream_name, response)
    _LOGGER.debug("Sent %s %s to stream %s", type(response).__name__, response_id, received_request.reply_stream_name)



@asynccontextmanager
async def async_auto_delete_stream(redis_stream: AsyncRedisStream, stream_name: str) -> AsyncGenerator[None, None]:
    """
    Async context manager to automatically delete a Redis stream when exiting the context.

    Args:
        redis_stream: AsyncRedisStream instance
        stream_name: Name of the stream to delete

    Example:
        async with async_auto_delete_stream(redis_stream, reply_stream_name):
            # do work with the stream
            pass
        # stream is automatically deleted here
    """
    try:
        yield
    finally:
        await redis_stream.delete_stream(stream_name)


async def async_send_request_and_wait_for_response(redis_stream: AsyncRedisStream,
                                                   request_stream_name: str,
                                                   request_message: RedisStreamRequest,
                                                   response_msg_type: type[_ResponseType]) -> _ResponseType | RedisStreamResponse:
    """
    Send a request and wait for a response (async version)
    """
    block = 0 # must be blocking call
    count = 10 # max number of responses to wait for

    request_id = await redis_stream.send_message(request_stream_name, request_message)
    _LOGGER.debug("Sent request %s to stream %s, with reply_stream_name %s", request_id, request_stream_name, request_message.reply_stream_name)

    async with async_auto_delete_stream(redis_stream, request_message.reply_stream_name):
        responses = await redis_stream.xread(request_message.reply_stream_name, response_msg_type, count, block, "0")
        if not responses:
            _LOGGER.error("No response received from stream '%s' after %d milliseconds", request_message.reply_stream_name, block)
            return RedisStreamResponse(request_id=request_id, error_message="No response received")

        if len(responses) > 1:
            _LOGGER.error("Received multiple responses from stream '%s' for request %s.", request_message.reply_stream_name, request_id)
            return RedisStreamResponse(request_id=request_id, error_message="Multiple responses received")

        response_id, response = responses[0]

        if response.request_id != request_id:
            _LOGGER.error("Received response %s for request %s, but the waiting request ID is %s", response_id, response.request_id, request_id)
            return RedisStreamResponse(request_id="error", error_message="Response/Request ID mismatch. This should not happen.")

    _LOGGER.debug("Received response %s for request %s", response_id, request_id)
    return response


async def async_send_request_and_get_response_generator(redis_stream: AsyncRedisStream,
                                                        request_stream_name: str,
                                                        request_message: RedisStreamRequest,
                                                        response_msg_type: type[_ResponseType]) -> AsyncGenerator[list[_ResponseType], bool]:
    """
    Send a request and get an async generator of responses
    """
    count = 10 # max number of responses to wait for
    block = 0 # must be blocking call

    request_id = await redis_stream.send_message(request_stream_name, request_message)
    _LOGGER.debug("Sent request %s to stream %s, with reply_stream_name %s",
                  request_id,
                  request_stream_name,
                  request_message.reply_stream_name)

    last_id = "0"
    should_continue = True
    async with async_auto_delete_stream(redis_stream, request_message.reply_stream_name):
        while should_continue:
            responses = await redis_stream.xread(request_message.reply_stream_name, response_msg_type, count, block, last_id)
            if not responses:
                _LOGGER.error("No response received from stream '%s'. This should not happen. Something wrong.", request_message.reply_stream_name)
                break

            response_list = []
            for response_id, response in responses:
                _LOGGER.debug("Collecting response %s", response_id)
                response_list.append(response)
                last_id = response_id

            _LOGGER.debug("Yielding response list %s, should_continue: %s", response_list, should_continue)
            should_continue = yield response_list


async def async_send_response(redis_stream: AsyncRedisStream,
                              received_request: RedisStreamRequest,
                              response: RedisStreamResponse) -> None:
    """
    Send a response to the request through the reply stream (async version)
    """
    response_id = await redis_stream.send_message(received_request.reply_stream_name, response)
    _LOGGER.debug("Sent %s %s to stream %s", type(response).__name__, response_id, received_request.reply_stream_name)


class RedisStream:

    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client

    def send_message(self, stream_name: str, message: _MsgType, message_id: str = "*") -> str:
        """
        Send a message to a Redis Stream

        Args:
            stream_name: Name of the stream
            message: RedisStreamMessage object
            message_id: Message ID (default "*" for auto-generated)

        Returns:
            Message ID that was assigned
        """

        returned_message_id_bytes = self.redis_client.xadd(
                stream_name,
                message.to_redis_stream_message(),
                id=message_id,
        )
        returned_message_id = returned_message_id_bytes.decode(STR_ENCODING)
        _LOGGER.debug("Sent message to stream %s with ID: %s", stream_name, returned_message_id)
        return returned_message_id

    def create_consumer_group(self, stream_name: str, group_name: str, start_id: str = "$") -> None:
        """
        Create a consumer group for a stream

        Args:
            stream_name: Name of the stream
            group_name: Name of the consumer group
            start_id: Starting message ID for the group. Default is "$" which means start from the latest message.
        """
        try:
            self.redis_client.xgroup_create(stream_name, group_name, start_id, mkstream=True)
            _LOGGER.info("Created consumer group '%s' for stream '%s'", group_name, stream_name)
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                _LOGGER.info("Consumer group '%s' already exists for stream '%s'", group_name, stream_name)
            else:
                raise

    def delete_stream(self, stream_name: str) -> None:
        """
        Delete a Redis Stream
        """
        self.redis_client.delete(stream_name)
        _LOGGER.debug("Deleted stream '%s'", stream_name)

    def xread(self, stream_name: str, msg_type: type[_MsgType], count: int, block: int, read_id: str) -> list[tuple[str, _MsgType]]:
        """
        Read messages from a Redis Stream

        Args:
            stream_name: Name of the stream to read from
            msg_type: Type of the message to read. It will be used to decode the message.
            count: Maximum number of messages to read
            block: Block time in milliseconds (0 for non-blocking, >0 for blocking)
            read_id: ID to read from.

        Returns:
            List of tuples (message_id, message) where message_id is a string and message is a RedisStreamMessage object
        """
        redis_xread_messages = self.redis_client.xread({stream_name: read_id}, count=count, block=block)
        if not redis_xread_messages:
            return []

        parsed_messages = _parse_redis_messages(redis_xread_messages, msg_type)
        _LOGGER.debug("Read %d messages from stream '%s'", len(parsed_messages), stream_name)
        return parsed_messages

    def xreadgroup(self, stream_name: str, group_name: str, consumer_name: str, msg_type: type[_MsgType],
                   count: int = 1, block: int = 0) -> list[tuple[str, _MsgType]]:
        """
        Read messages from a consumer group

        Args:
            stream_name: Name of the stream
            group_name: Name of the consumer group
            consumer_name: Name of the consumer
            msg_type: Type of the message to read. It will be used to decode the message.
            count: Maximum number of messages to read (default 1)
            block: Block time in milliseconds (0 for non-blocking)

        Returns:
            List of tuples (message_id, message)
        """

        redis_xreadgroup_messages = self.redis_client.xreadgroup(
            group_name,
            consumer_name,
            {stream_name: ">"},
            count=count,
            block=block
        )

        if not redis_xreadgroup_messages:
            return []

        parsed_messages = _parse_redis_messages(redis_xreadgroup_messages, msg_type)
        _LOGGER.debug("Read %d messages from group '%s'", len(parsed_messages), group_name)
        return parsed_messages

    def ack(self, stream_name: str, group_name: str, message_id: str) -> None:
        """
        Acknowledge a message in a consumer group

        Args:
            stream_name: Name of the stream
            group_name: Name of the consumer group
            message_id: Message ID to acknowledge
        """

        self.redis_client.xack(stream_name, group_name, message_id)
        _LOGGER.debug("Acknowledged message %s in group '%s'", message_id, group_name)


class AsyncRedisStream:
    """Async version of RedisStream using async Redis client"""

    def __init__(self, redis_client: redis.asyncio.Redis):
        self.redis_client = redis_client

    async def send_message(self, stream_name: str, message: _MsgType, message_id: str = "*") -> str:
        """
        Send a message to a Redis Stream

        Args:
            stream_name: Name of the stream
            message: RedisStreamMessage object
            message_id: Message ID (default "*" for auto-generated)

        Returns:
            Message ID that was assigned
        """

        returned_message_id_bytes = await self.redis_client.xadd(
                stream_name,
                message.to_redis_stream_message(),
                id=message_id,
        )
        returned_message_id = returned_message_id_bytes.decode(STR_ENCODING)
        _LOGGER.debug("Sent message to stream %s with ID: %s", stream_name, returned_message_id)
        return returned_message_id

    async def create_consumer_group(self, stream_name: str, group_name: str, start_id: str = "$") -> None:
        """
        Create a consumer group for a stream

        Args:
            stream_name: Name of the stream
            group_name: Name of the consumer group
            start_id: Starting message ID for the group. Default is "$" which means start from the latest message.
        """
        try:
            await self.redis_client.xgroup_create(stream_name, group_name, start_id, mkstream=True)
            _LOGGER.info("Created consumer group '%s' for stream '%s'", group_name, stream_name)
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                _LOGGER.info("Consumer group '%s' already exists for stream '%s'", group_name, stream_name)
            else:
                raise

    async def delete_stream(self, stream_name: str) -> None:
        """
        Delete a Redis Stream
        """
        await self.redis_client.delete(stream_name)
        _LOGGER.debug("Deleted stream '%s'", stream_name)

    async def xread(self, stream_name: str, msg_type: type[_MsgType], count: int, block: int, read_id: str) -> list[tuple[str, _MsgType]]:
        """
        Read messages from a Redis Stream

        Args:
            stream_name: Name of the stream to read from
            msg_type: Type of the message to read. It will be used to decode the message.
            count: Maximum number of messages to read
            block: Block time in milliseconds (0 for non-blocking, >0 for blocking)
            read_id: ID to read from.

        Returns:
            List of tuples (message_id, message) where message_id is a string and message is a RedisStreamMessage object
        """
        redis_xread_messages = await self.redis_client.xread({stream_name: read_id}, count=count, block=block)
        if not redis_xread_messages:
            return []

        parsed_messages = await asyncio.to_thread(_parse_redis_messages, redis_xread_messages, msg_type)
        _LOGGER.debug("Read %d messages from stream '%s'", len(parsed_messages), stream_name)
        return parsed_messages

    async def xreadgroup(self, stream_name: str, group_name: str, consumer_name: str, msg_type: type[_MsgType],
                   count: int = 1, block: int = 0) -> list[tuple[str, _MsgType]]:
        """
        Read messages from a consumer group

        Args:
            stream_name: Name of the stream
            group_name: Name of the consumer group
            consumer_name: Name of the consumer
            msg_type: Type of the message to read. It will be used to decode the message.
            count: Maximum number of messages to read (default 1)
            block: Block time in milliseconds (0 for non-blocking)

        Returns:
            List of tuples (message_id, message)
        """

        redis_xreadgroup_messages = await self.redis_client.xreadgroup(
            group_name,
            consumer_name,
            {stream_name: ">"},
            count=count,
            block=block
        )

        if not redis_xreadgroup_messages:
            return []

        parsed_messages = await asyncio.to_thread(_parse_redis_messages, redis_xreadgroup_messages, msg_type)
        _LOGGER.debug("Read %d messages from group '%s'", len(parsed_messages), group_name)
        return parsed_messages

    async def ack(self, stream_name: str, group_name: str, message_id: str) -> None:
        """
        Acknowledge a message in a consumer group

        Args:
            stream_name: Name of the stream
            group_name: Name of the consumer group
            message_id: Message ID to acknowledge
        """

        await self.redis_client.xack(stream_name, group_name, message_id)
        _LOGGER.debug("Acknowledged message %s in group '%s'", message_id, group_name)
