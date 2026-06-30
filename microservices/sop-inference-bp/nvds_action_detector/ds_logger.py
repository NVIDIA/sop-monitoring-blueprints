################################################################################
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
################################################################################

import logging
import os
import sys
import threading
from enum import Enum
from typing import Any, Dict, Optional

# Global lock for thread-safe logger configuration
_logger_lock = threading.Lock()
_configured_loggers: Dict[str, bool] = {}

# Environment variables
kDebug = int(os.getenv("DS_LOG_DEBUG", "0"))
PACKAGE_NAME = "DS_ACTION_DETECTOR"


class FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that automatically flushes after each log message."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record and flush the stream immediately."""
        super().emit(record)
        self.flush()


class LogLevel(Enum):
    """Enumeration of supported log levels"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def _get_log_level_from_env() -> int:
    """Get log level from environment variable with proper validation"""
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()

    # Override with DEBUG if debug flag is set
    if kDebug:
        return logging.DEBUG

    # Map string levels to logging constants
    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return level_mapping.get(log_level_str, logging.INFO)


def _configure_stderr_encoding() -> None:
    """Configure stderr encoding with error handling"""
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError) as e:
        # Fallback for older Python versions or encoding issues
        logging.warning(f"Could not reconfigure stderr encoding: {e}")


def _setup_root_logger(log_level: int, log_format: str) -> None:
    """Setup root logger configuration in a thread-safe manner"""
    with _logger_lock:
        if not _configured_loggers.get("root", False):
            # Clear any existing handlers to avoid duplicates
            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            # Configure basic logging
            logging.basicConfig(
                level=log_level,
                format=log_format,
                handlers=[FlushingStreamHandler(sys.stderr)],
                force=True,  # Override any existing configuration
            )
            _configured_loggers["root"] = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get component logger with improved configuration and error handling.

    Args:
        name: Optional module name. If None, uses the package name.

    Returns:
        A configured Logger instance.

    Raises:
        ValueError: If the logger name is invalid.
    """
    if name is None:
        name = PACKAGE_NAME
    elif not isinstance(name, str) or not name.strip():
        raise ValueError("Logger name must be a non-empty string")

    # Create full logger name
    full_name = f"{PACKAGE_NAME}.{name.strip()}"

    # Get log level and format
    log_level = _get_log_level_from_env()
    log_format = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"

    # Configure stderr encoding
    _configure_stderr_encoding()

    # Setup root logger if not already configured
    _setup_root_logger(log_level, log_format)

    # Get the specific logger
    logger = logging.getLogger(full_name)
    logger.propagate = True

    # Log the logger creation (only once per logger)
    with _logger_lock:
        if not _configured_loggers.get(full_name, False):
            print(f"Logger '{full_name}' created with level: {logging.getLevelName(log_level)}")
            _configured_loggers[full_name] = True

    return logger


def flush(logger: logging.Logger) -> None:
    """Flush all handlers of the given logger.

    Args:
        logger: The logger instance to flush.
    """
    if not isinstance(logger, logging.Logger):
        raise TypeError("logger must be a logging.Logger instance")

    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception as e:
            # Log the error but don't raise to avoid breaking the application
            print(f"Warning: Failed to flush handler {handler}: {e}", file=sys.stderr)


def set_log_level(logger: logging.Logger, level: str) -> None:
    """Set the log level for a specific logger.

    Args:
        logger: The logger instance to configure.
        level: Log level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    if not isinstance(logger, logging.Logger):
        raise TypeError("logger must be a logging.Logger instance")

    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    log_level = level_mapping.get(level.upper())
    if log_level is None:
        raise ValueError(f"Invalid log level: {level}. Must be one of: {list(level_mapping.keys())}")

    logger.setLevel(log_level)


def get_logger_info(logger: logging.Logger) -> Dict[str, Any]:
    """Get information about a logger instance.

    Args:
        logger: The logger instance to inspect.

    Returns:
        Dictionary containing logger information.
    """
    if not isinstance(logger, logging.Logger):
        raise TypeError("logger must be a logging.Logger instance")

    return {
        "name": logger.name,
        "level": logging.getLevelName(logger.level),
        "effective_level": logging.getLevelName(logger.getEffectiveLevel()),
        "handlers_count": len(logger.handlers),
        "propagate": logger.propagate,
        "disabled": logger.disabled,
    }
