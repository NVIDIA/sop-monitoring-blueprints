######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import logging
import os
import sys
from pathlib import Path
from typing import Optional

import utils.constant as const


class LoggerConfig:
    """Configuration for the logging system"""

    # Log file path - will be created in the same directory as the application
    LOG_FILE = os.path.join(const.LOG_FILE_ROOT, const.LOG_FILE_NAME)

    # Log format with timestamp, logger name, level, and message
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Date format for timestamps
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    # Default log level
    DEFAULT_LEVEL = logging.INFO

    # Maximum log file size in bytes (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024

    # Number of backup files to keep
    BACKUP_COUNT = 5


def setup_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Set up a logger with both file and console handlers

    Args:
        name: The name of the logger (usually __name__)
        level: Logging level (optional, defaults to INFO)

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Set log level
    logger.setLevel(level or LoggerConfig.DEFAULT_LEVEL)

    # Create formatter
    formatter = logging.Formatter(
        LoggerConfig.LOG_FORMAT, datefmt=LoggerConfig.DATE_FORMAT
    )

    # Create file handler with rotation
    file_handler = create_file_handler(formatter)
    logger.addHandler(file_handler)

    # Create console handler
    console_handler = create_console_handler(formatter)
    logger.addHandler(console_handler)

    return logger


def create_file_handler(formatter: logging.Formatter) -> logging.Handler:
    """
    Create a file handler with log rotation

    Args:
        formatter: The log formatter to use

    Returns:
        logging.Handler: Configured file handler
    """
    try:
        # Try to create log file in the application root directory
        log_path = Path(LoggerConfig.LOG_FILE)

        # Ensure the directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create RotatingFileHandler for log rotation
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=LoggerConfig.MAX_FILE_SIZE,
            backupCount=LoggerConfig.BACKUP_COUNT,
            encoding="utf-8",
        )

    except OSError:
        # Fallback to temporary directory if we can't write to the intended location
        import tempfile

        temp_dir = tempfile.gettempdir()
        log_path = Path(temp_dir) / LoggerConfig.LOG_FILE

        try:
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=LoggerConfig.MAX_FILE_SIZE,
                backupCount=LoggerConfig.BACKUP_COUNT,
                encoding="utf-8",
            )
        except Exception:
            # Last resort: use basic FileHandler without rotation
            file_handler = logging.FileHandler(log_path, encoding="utf-8")

    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_handler.setFormatter(formatter)

    return file_handler


def create_console_handler(formatter: logging.Formatter) -> logging.Handler:
    """
    Create a console handler for terminal output

    Args:
        formatter: The log formatter to use

    Returns:
        logging.Handler: Configured console handler
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Only show INFO and above in console
    console_handler.setFormatter(formatter)

    return console_handler


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger instance with the specified name

    Args:
        name: The name of the logger (usually __name__)
        level: Logging level (optional, defaults to INFO)

    Returns:
        logging.Logger: Configured logger instance
    """
    return setup_logger(name, level)


# Convenience functions for common loggers
def get_app_logger() -> logging.Logger:
    """Get the main application logger"""
    return get_logger("data_augmentation")


app_logger = get_app_logger()
app_logger.info("Logging system initialized")