######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
######################################################################################################

import logging
import os
import re
from typing import Any, Dict

import psutil
import toml


logger = logging.getLogger(__name__)


def create_dir(dir_path: str) -> bool:
    """Create a directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return True
    return False


def create_file(file_path: str) -> bool:
    """Create a file if it doesn't exist"""
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        open(file_path, "w").close()
        return True
    return False


def read_toml(file_path: str) -> Dict[str, Any]:
    """Read a TOML file and return a dictionary"""
    with open(file_path, "r") as f:
        return toml.load(f)


def dump_toml(toml_dict: Dict[str, Any], file_path: str) -> bool:
    """Dump a dictionary to a TOML file"""
    with open(file_path, "w") as f:
        toml.dump(toml_dict, f)
    return True


def parse_cr_log(line: str) -> Dict[str, Any]:
    """Parse the log string to extract training progress and information"""
    try:
        # We parse the step (current/total), loss, grad norm, learning rate, and iteration time
        # It's not a good practice to parse the log string, but CR currently doesn't dump training progress json
        # TODO: Remove this once CR dumps training progress json
        pattern = r"Step: (\d+)/(\d+), Loss: ([\d.e+-]+), Grad norm: ([\d.e+-]+), Learning rate: ([\d.e+-]+), Iteration time: ([\d.]+)s\.?"

        match = re.search(pattern, line)

        if not match:
            return {}

        return {
            "current_step": int(match.group(1)),
            "total_steps": int(match.group(2)),
            "loss": float(match.group(3)),
            "grad_norm": float(match.group(4)),
            "learning_rate": float(match.group(5)),
            "iteration_time": float(match.group(6)),
        }

    except Exception as e:
        logger.error(f"Error parsing Cosmos-Reason1 log: {str(e)}")
        return {}


def terminate_process_tree(process_pid: int, timeout: int = 30) -> bool:
    """
    Terminate a process and all its children recursively.

    Args:
        process_pid: PID of the process to terminate
        timeout: Timeout in seconds for graceful termination

    Returns:
        True if all processes were terminated successfully, False otherwise
    """
    try:
        # Get the process
        parent_process = psutil.Process(process_pid)

        # Get all children recursively
        children = parent_process.children(recursive=True)

        logger.info(f"Terminating process tree: parent PID {process_pid}, {len(children)} children")

        # First, try graceful termination (SIGTERM)
        for child in children:
            try:
                child.terminate()
                logger.info(f"Sent SIGTERM to child process {child.pid}")
            except psutil.NoSuchProcess:
                pass  # Process already dead

        # Also terminate the parent
        try:
            parent_process.terminate()
            logger.info(f"Sent SIGTERM to parent process {process_pid}")
        except psutil.NoSuchProcess:
            pass

        # Wait for processes to terminate gracefully
        _, alive = psutil.wait_procs([parent_process] + children, timeout=timeout)

        # If any processes are still alive, force kill them
        if alive:
            logger.warning(f"Force killing {len(alive)} processes that didn't terminate gracefully")
            for process in alive:
                try:
                    process.kill()
                    logger.info(f"Force killed process {process.pid}")
                except psutil.NoSuchProcess:
                    pass

            # Wait a bit more for force-killed processes
            psutil.wait_procs(alive, timeout=5)

        # Final check - make sure all processes are gone
        for process in [parent_process] + children:
            try:
                if process.is_running():
                    logger.error(f"Process {process.pid} is still running after termination attempt")
                    return False
            except psutil.NoSuchProcess:
                pass  # Process is already dead

        logger.info(f"Successfully terminated process tree for PID {process_pid}")
        return True

    except psutil.NoSuchProcess:
        logger.info(f"Process {process_pid} was already terminated")
        return True
    except Exception as e:
        logger.error(f"Error terminating process tree for PID {process_pid}: {str(e)}")
        return False
