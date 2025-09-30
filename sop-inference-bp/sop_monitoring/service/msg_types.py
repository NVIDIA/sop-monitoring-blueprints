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
Modules for message types among microservices

TODO: This module should be refactored. Lots of literals like '<i', '<d', etc. are repeated.
TODO: Or even better, just let's use flatbuffer for all the message types.
      By that we can also handle complex types like Messages in OpenAI format.
"""

import abc
import io
import logging
import struct
import typing

from dataclasses import dataclass

_LOGGER = logging.getLogger(__name__)
STR_ENCODING = 'utf-8'

# only used for list encoding/decoding
_TYPE_BYTE_LENGTH = 1
_TYPE_BYTE_MAP = {
    int: b'i',
    float: b'f',
    bool: b'?',
    str: b's',
}
_TYPE_BYTE_MAP_REVERSE = {v: k for k, v in _TYPE_BYTE_MAP.items()}
_ITEM_BYTES_NUM_LENGTH = 4
_ITEM_BYTES_NUM_FORMAT = '<I'

def _encode_bytes_num(num: int) -> bytes:
    """
    Encode the number for bytes length
    """
    if num > 2**31 - 1:
        raise ValueError(f"Number {num} is too large to be encoded in 4 bytes.")
    return struct.pack(_ITEM_BYTES_NUM_FORMAT, num)

def _decode_bytes_num(bytes_num: bytes) -> int:
    """
    Decode the number for bytes length
    """
    return struct.unpack(_ITEM_BYTES_NUM_FORMAT, bytes_num)[0]

def encode_value(value: bytes | int | float | bool | str | dict | list) -> bytes:
    """
    Efficiently encode values to bytes for Redis Streams
    Uses struct for numbers, custom format for list[int | float | bool | str], UTF-8 for strings

    FIXME: If any complex types are added or the size of data becomes large, we may consider FlatBuffer.
    """
    if isinstance(value, bytes):
        return value

    # Note that bool is a subclass of int in Python.
    # So we need to check bool before int.
    elif isinstance(value, bool):
        # Use struct for booleans (1 byte)
        return struct.pack('?', value)

    elif isinstance(value, int):
        # Check if the integer can fit in 4 bytes (32-bit signed integer range)
        if not (-2**31 <= value <= 2**31 - 1):
            raise ValueError(f"Integer {value} cannot be encoded in 4 bytes (32-bit signed integer)")
        # Use struct for signed, little-endian, 32-bits integers
        return struct.pack('<i', value)

    elif isinstance(value, float):
        # Use struct for 64 bits, little-endian float
        return struct.pack('<d', value)

    elif isinstance(value, str):
        return value.encode(STR_ENCODING)

    elif isinstance(value, dict):
        raise ValueError(f"Only int, float, bool, str, and list of these types are supported, but found dict with keys {value.keys()} "
                          "Try to flatten the dict to the message types. Our message types should just work like a flat dict.")

    elif isinstance(value, list):
        parts = []
        for i, item in enumerate(value):
            if not isinstance(item, (int, float, bool, str)):
                raise ValueError(f"Only int, float, bool, and str are supported. Found item at index {i} with value {item} of type {type(item).__name__}")

            item_bytes = encode_value(item)
            item_bytes_num_in_bytes = _encode_bytes_num(len(item_bytes))

            # The format is <type_byte:1><item_bytes_num_in_bytes:4><item_bytes:item_bytes_num>
            parts.append(_TYPE_BYTE_MAP[type(item)])
            parts.append(item_bytes_num_in_bytes)
            parts.append(item_bytes)

        return b''.join(parts)

    raise ValueError(f"Unsupported value_type: {type(value).__name__}. "
                     "Only bytes, int, float, bool, str, flat dict, and flat list are supported.")


def decode_value(value_bytes: bytes, value_type: type) -> bytes | int | float | bool | str | dict | list:
    """
    Decode bytes back to original type
    """

    if not isinstance(value_bytes, bytes):
        return value_bytes

    if value_type == int:
        return struct.unpack('<i', value_bytes)[0]
    elif value_type == float:
        return struct.unpack('<d', value_bytes)[0]
    elif value_type == bool:
        return struct.unpack('?', value_bytes)[0]
    elif value_type == str:
        return value_bytes.decode(STR_ENCODING)
    elif value_type == list:
        results = []
        stream = io.BytesIO(value_bytes)
        while stream.tell() < len(value_bytes):
            type_byte = stream.read(_TYPE_BYTE_LENGTH)
            item_bytes_num_in_bytes = stream.read(_ITEM_BYTES_NUM_LENGTH)
            item_bytes_num = _decode_bytes_num(item_bytes_num_in_bytes)
            item_bytes = stream.read(item_bytes_num)
            item = decode_value(item_bytes, _TYPE_BYTE_MAP_REVERSE[type_byte])
            results.append(item)
        return results

    elif value_type == bytes:
        return value_bytes

    raise ValueError(f"Unsupported value_type: {value_type.__name__}. "
                     "Only bytes, int, float, bool, str, flat dict, and flat list are supported.")


class RedisStreamMessage(abc.ABC):
    """
    Base class for Redis stream messages
    """

    def to_redis_stream_message(self) -> dict[bytes, bytes]:
        """
        Convert to Redis stream message format
        """
        annotations = typing.get_type_hints(self.__class__.__init__)
        # This can raise exception if "return" is not in annotations.
        # But __init__ should have "return" as a parameter. So if it raises, something is wrong.
        del annotations["return"]
        return {k.encode(STR_ENCODING): encode_value(getattr(self, k)) for k in annotations}

    @classmethod
    def from_redis_stream_message(cls, message: dict[bytes, bytes]) -> "RedisStreamMessage":
        """
        Create from Redis stream message
        """
        annotations = typing.get_type_hints(cls.__init__)
        del annotations["return"]
        kwargs = {}
        for k, v in message.items():
            key = k.decode(STR_ENCODING)
            if key not in annotations:
                raise ValueError(f"Unknown key {key} for {cls.__name__}")
            origin_type = typing.get_origin(annotations[key])
            if origin_type is None:
                origin_type = annotations[key]
            kwargs[key] = decode_value(v, origin_type)

        return cls(**kwargs)


@dataclass
class RedisStreamRequest(RedisStreamMessage):
    # unique reply steam for the request.
    # You can generate on by utils.generate_reply_stream_name()
    reply_stream_name: str

@dataclass
class RedisStreamResponse(RedisStreamMessage):
    request_id: str
    error_message: str

@dataclass
class VlmInferenceRequest(RedisStreamRequest):
    # FIXME: To avoid complex nested types, we only support 1 system prompt, 1 user prompt and 1 video.
    system_prompt: str
    prompt: str
    video_id: str
    chunk_start_seconds: list[float]
    chunk_end_seconds: list[float]
    # TODO: chunk_keys are a bit legacy.
    # We should use chunk_start_seconds and chunk_end_seconds instead, which allows VLM service to do efficient chunking.
    chunk_keys: list[str]
    stream_response: bool

@dataclass
class VlmInferenceResponse(RedisStreamResponse):
    contents: list[str]
    # if True, this is the final response. Only useful for streaming response.
    final_response: bool

@dataclass
class ActionSegmentRequest(RedisStreamRequest):
    # This video_id is the MinIO object name.
    video_id: str
    # This is a JSON string of the options.
    # The JSON string is generated by the model_dump_json() method of the pydantic BaseModel object.
    options_json: str

@dataclass
class ActionSegmentResponse(RedisStreamResponse):
    chunk_start_seconds: list[float]
    chunk_end_seconds: list[float]
    # TODO: chunk_keys are a bit legacy.
    # We should use chunk_start_seconds and chunk_end_seconds instead, which allows VLM service to do efficient chunking.
    chunk_keys: list[str]

@dataclass
class SopCheckerRequest(RedisStreamRequest):
    # The content of the actions.json
    action_json: str
    # the output of the VLM inference.
    vlm_output: str
    # If this checker should be kept and used for the next request.
    # If true, the response would contain a valid id for users to send the next request.
    # If false, the checker would be cleared.
    keep_alive: bool
    # If this ID is not empty the service would try to find the corresponding checker and continue from there.
    # Use special value "*" for the very first request.
    # Raise error if the checker is not found.
    checker_id: str
    # Options for the SOP detection.
    cycle_completion_threshold: float
    cycle_boundary_threshold_low: float
    cycle_boundary_threshold_high: float

@dataclass
class SopCheckerResponse(RedisStreamResponse):
    # If keep_alive is true, reply a valid checker_id for the next request.
    checker_id: str
    cycle: int
    missing_detected: list[int]
    misordered_detected: list[int]
    final_missing_detected: list[int]
    final_misordered_detected: list[int]
    cycle_completed: bool
    summary_cycles_detected: list[str]
    summary_cycle_analysis: list[str]
