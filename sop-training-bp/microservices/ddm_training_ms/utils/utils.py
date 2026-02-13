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
import yaml


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

