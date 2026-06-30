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
import re
from typing import Any, Dict

import psutil
import toml
import yaml


logger = logging.getLogger(__name__)


def create_dir(dir_path: str) -> bool:
    """Create a directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return True
    return False


def safe_dataset_path(root: str, dataset_id: str) -> str:
    """Join ``dataset_id`` onto ``root`` and verify the result stays under ``root``.

    Raises ``ValueError`` if ``dataset_id`` contains path separators or traversal
    segments, or if the resolved path escapes ``root``. Callers should translate
    that into an HTTP 400.
    """
    if not dataset_id or "/" in dataset_id or "\\" in dataset_id or dataset_id in (".", ".."):
        raise ValueError(f"Invalid dataset id: {dataset_id!r}")
    root_real = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root_real, dataset_id))
    if candidate != root_real and not candidate.startswith(root_real + os.sep):
        raise ValueError(f"Dataset id escapes root: {dataset_id!r}")
    return candidate


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

def dump_yaml(data: Dict[str, Any], file_path: str) -> bool:
    """Dump a dictionary to a YAML file"""
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    return True

def read_yaml(file_path: str) -> Dict[str, Any]:
    """Read a YAML file and return a dictionary"""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def parse_ddm_log(line: str) -> Dict[str, Any]:
    """Parse the log string to extract training progress and information
    
    Parses PyTorch Lightning log format:
    Epoch 0:   1%|          | 1/115 [00:02<05:40,  0.33it/s, v_num=0, train/loss_step=12.90]
    """
    try:
        # Pattern for PyTorch Lightning logs
        # Captures: Epoch, current_step/total_steps, and train/loss_step or train/loss_epoch
        epoch_pattern = r"Epoch (\d+):\s+\d+%\|.*?\|\s+(\d+)/(\d+)\s+\[.*?\].*?train/loss_(?:step|epoch)=([\d.]+|nan)"
        
        match = re.search(epoch_pattern, line)
        
        if match:
            epoch = int(match.group(1))
            current_step = int(match.group(2))
            total_steps = int(match.group(3))
            loss_str = match.group(4)
            
            # Handle nan loss
            try:
                loss = float(loss_str)
                if loss != loss:  # Check if NaN
                    loss = None
            except ValueError:
                loss = None
            
            # Calculate global step (epoch * steps_per_epoch + current_step)
            global_step = epoch * total_steps + current_step
            # Total steps for entire training
            max_steps = total_steps  # Per epoch
            
            return {
                "epoch": epoch,
                "current_step": global_step,
                "total_steps": max_steps,
                "loss": loss,
            }
        
        return {}

    except Exception as e:
        logger.error(f"Error parsing DDM log: {str(e)}")
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

