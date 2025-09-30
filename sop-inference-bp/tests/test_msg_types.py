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
Unit tests for msg_types.py
"""

import struct
import os
import sys

import pytest

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sop_monitoring.service.msg_types import (
    encode_value,
    decode_value,
    STR_ENCODING
)

class TestEncodeValue:
    """Test the encode_value function"""

    def test_encode_bytes(self):
        """Test encoding bytes values"""
        test_bytes = b"test data"
        result = encode_value(test_bytes)
        assert result == test_bytes

    def test_encode_bool(self):
        """Test encoding boolean values"""
        assert encode_value(True) == struct.pack('?', True)
        assert encode_value(False) == struct.pack('?', False)

    def test_encode_int_valid(self):
        """Test encoding valid integer values"""
        # Test positive integers
        assert encode_value(0) == struct.pack('<i', 0)
        assert encode_value(100) == struct.pack('<i', 100)
        assert encode_value(2147483647) == struct.pack('<i', 2147483647)  # Max 32-bit signed int

        # Test negative integers
        assert encode_value(-1) == struct.pack('<i', -1)
        assert encode_value(-100) == struct.pack('<i', -100)
        assert encode_value(-2147483648) == struct.pack('<i', -2147483648)  # Min 32-bit signed int

    def test_encode_int_invalid(self):
        """Test encoding invalid integer values"""
        # Test values that exceed 32-bit signed integer range
        with pytest.raises(ValueError, match="cannot be encoded in 4 bytes"):
            encode_value(2147483648)  # Max + 1

        with pytest.raises(ValueError, match="cannot be encoded in 4 bytes"):
            encode_value(-2147483649)  # Min - 1

    def test_encode_float(self):
        """Test encoding float values"""
        test_float = 3.14159
        result = encode_value(test_float)
        assert result == struct.pack('<d', test_float)

        # Test negative float
        test_float_neg = -2.718
        result_neg = encode_value(test_float_neg)
        assert result_neg == struct.pack('<d', test_float_neg)

    def test_encode_str(self):
        """Test encoding string values"""
        test_str = "Hello, World!"
        result = encode_value(test_str)
        assert result == test_str.encode(STR_ENCODING)

        # Test empty string
        empty_str = ""
        result_empty = encode_value(empty_str)
        assert result_empty == empty_str.encode(STR_ENCODING)

        # Test unicode string
        unicode_str = "Hello 世界"
        result_unicode = encode_value(unicode_str)
        assert result_unicode == unicode_str.encode(STR_ENCODING)

    def test_encode_list_invalid(self):
        """Test encoding invalid list values"""
        # Test list with unsupported item type
        invalid_list = ["item1", {"nested": "nested_value"}]
        with pytest.raises(ValueError, match="Only int, float, bool, and str are supported"):
            encode_value(invalid_list)

    def test_encode_unsupported_type(self):
        """Test encoding unsupported types"""
        # Test with None
        with pytest.raises(ValueError, match="Unsupported value_type"):
            encode_value(None)

        # Test with complex number
        with pytest.raises(ValueError, match="Unsupported value_type"):
            encode_value(1 + 2j)

        # Test with tuple
        with pytest.raises(ValueError, match="Unsupported value_type"):
            encode_value((1, 2, 3))


class TestDecodeValue:
    """Test the decode_value function"""

    def test_decode_int(self):
        """Test decoding integer values"""
        test_int = 42
        encoded = struct.pack('<i', test_int)
        result = decode_value(encoded, int)
        assert result == test_int

        # Test negative integer
        test_int_neg = -100
        encoded_neg = struct.pack('<i', test_int_neg)
        result_neg = decode_value(encoded_neg, int)
        assert result_neg == test_int_neg

    def test_decode_float(self):
        """Test decoding float values"""
        test_float = 3.14159
        encoded = struct.pack('d', test_float)
        result = decode_value(encoded, float)
        assert result == pytest.approx(test_float, rel=1e-6)

    def test_decode_bool(self):
        """Test decoding boolean values"""
        assert decode_value(struct.pack('?', True), bool) is True
        assert decode_value(struct.pack('?', False), bool) is False

    def test_decode_str(self):
        """Test decoding string values"""
        test_str = "Hello, World!"
        encoded = test_str.encode(STR_ENCODING)
        result = decode_value(encoded, str)
        assert result == test_str

    def test_decode_list(self):
        """Test decoding list values"""
        test_list = ["item1", 42, True]
        encoded = encode_value(test_list)
        result = decode_value(encoded, list)
        assert result == test_list

    def test_decode_non_bytes_input(self):
        """Test decoding non-bytes input"""
        # Should return the input as-is if it's not bytes
        test_str = "already decoded"
        result = decode_value(test_str, str)
        assert result == test_str

    def test_decode_unsupported_type(self):
        """Test decoding unsupported types"""
        test_bytes = b"test"
        with pytest.raises(ValueError, match="Unsupported value_type"):
            decode_value(test_bytes, tuple)

class TestRoundTrip:
    """Test round trip encoding/decoding"""

    def test_encode_decode_empty_values(self):
        """Test encoding/decoding empty values"""
        empty_values = {
            "str": "",
            "list": [],
        }

        for value_type, empty_value in empty_values.items():
            encoded = encode_value(empty_value)
            decoded = decode_value(encoded, type(empty_value))
            assert decoded == empty_value

    def test_encode_decode_special_chars(self):
        """Test encoding/decoding special characters"""
        special_str = "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        encoded = encode_value(special_str)
        decoded = decode_value(encoded, str)
        assert decoded == special_str

    def test_encode_decode_large_numbers(self):
        """Test encoding/decoding large numbers within limits"""
        # Test maximum valid 32-bit signed integer
        max_int = 2147483647
        encoded = encode_value(max_int)
        decoded = decode_value(encoded, int)
        assert decoded == max_int

        # Test minimum valid 32-bit signed integer
        min_int = -2147483648
        encoded = encode_value(min_int)
        decoded = decode_value(encoded, int)
        assert decoded == min_int

    def test_encode_decode_precision_float(self):
        """Test encoding/decoding float precision"""
        test_float = 1.23456789
        encoded = encode_value(test_float)
        decoded = decode_value(encoded, float)
        # Float precision should be maintained within reasonable limits
        assert abs(decoded - test_float) < 1e-6

    def test_encode_decode_list_mixed_types(self):
        """Test encoding/decoding list with mixed types"""
        mixed_list = [42, 3.14, True, "hello", False, -100, 0.0, ""]
        encoded = encode_value(mixed_list)
        decoded = decode_value(encoded, list)

        # Compare each element, handling floats with approximate equality
        assert len(decoded) == len(mixed_list)
        for decoded_val, original_val in zip(decoded, mixed_list):
            if isinstance(original_val, float):
                assert decoded_val == pytest.approx(original_val, rel=1e-6)
            else:
                assert decoded_val == original_val

    def test_encode_decode_list_integers_only(self):
        """Test encoding/decoding list with integers only"""
        int_list = [1, -5, 0, 2147483647, -2147483648, 999]
        encoded = encode_value(int_list)
        decoded = decode_value(encoded, list)
        assert decoded == int_list

    def test_encode_decode_list_strings_only(self):
        """Test encoding/decoding list with strings only"""
        str_list = ["hello", "world", "", "test 世界", "special!@#$%"]
        encoded = encode_value(str_list)
        decoded = decode_value(encoded, list)
        assert decoded == str_list

    def test_encode_decode_list_booleans_only(self):
        """Test encoding/decoding list with booleans only"""
        bool_list = [True, False, True, True, False]
        encoded = encode_value(bool_list)
        decoded = decode_value(encoded, list)
        assert decoded == bool_list

    def test_encode_decode_list_floats_only(self):
        """Test encoding/decoding list with floats only"""
        float_list = [3.14, -2.718, 0.0, 1.414, -999.999]
        encoded = encode_value(float_list)
        decoded = decode_value(encoded, list)
        # Compare floats with approximate equality
        assert len(decoded) == len(float_list)
        for decoded_val, original_val in zip(decoded, float_list):
            assert decoded_val == pytest.approx(original_val, rel=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
