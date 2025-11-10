#!/usr/bin/env python
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
A script to check the environment variables.
"""

import argparse
import os
import socket
import subprocess

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def get_running_container_names() -> list[str]:
    """
    Gets the names of all running Docker containers by docker CLI.
    """

    command = ['docker', 'ps', '-a', '--format', '{{.Names}}']

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        names = result.stdout.strip().split('\n')
        running_names = [name for name in names if name]
        return running_names

    except FileNotFoundError:
        print("Error: The 'docker' command was not found.")
        print("Please ensure Docker is installed and in your system's PATH.")
        raise
    except subprocess.CalledProcessError as exc:
        print(f"Error executing 'docker ps':")
        print(exc.stderr)
        raise

def parse_env_file(env_file: str) -> dict[str, str]:
    """
    Parses the environment file and returns a dictionary of environment variables.
    """
    with open(env_file, "r") as f:
        lines = f.readlines()
    env_vars = {}
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            continue
        if not line:
            continue
        # remove words after the first '#'
        # FIXME: this will fail on a line like MY_NAME="home#name"
        key_value_part, _, _ = line.partition('#')
        key_value_part = key_value_part.strip()
        try:
            key, value = key_value_part.split("=", 1)
        except Exception as e:
            print(f"[Error] Invalid environment variable line: {line}: {e}")
            continue
        key = key.strip()
        value = value.strip().strip('"')
        env_vars[key] = value
    return env_vars

def is_bp_name_unique(env_vars: dict[str, str]) -> bool:
    """
    BP_NAME should be unique among all running Docker containers.
    """
    print(f"Checking BP_NAME: {env_vars['BP_NAME']}")
    print("It shuold be unique among all running Docker containers.")

    running_docker_container_names = get_running_container_names()
    error_msg = ""
    for name in running_docker_container_names:
        if name.startswith(env_vars["BP_NAME"]):
            error_msg += f"Container {name} is using the same BP_NAME: {env_vars['BP_NAME']}\n"

    if error_msg:
        print(f"[Error] BP_NAME {env_vars['BP_NAME']} is not unique.")
        print(error_msg)

    return not bool(error_msg)

def is_port_valid(env_vars: dict[str, str]) -> bool:

    error_msg = ""

    try:
        nginx_port = int(env_vars["NGINX_INGRESS_PORT"])
    except ValueError as exc:
        error_msg += f"[Error] NGINX_INGRESS_PORT {env_vars['NGINX_INGRESS_PORT']} is not a valid port: {exc}\n"

    try:
        minio_console_port = int(env_vars["MINIO_CONSOLE_PORT"])
    except ValueError as exc:
        print(f"[Error] MINIO_CONSOLE_PORT {env_vars['MINIO_CONSOLE_PORT']} is not a valid port: {exc}")

    if error_msg:
        print(error_msg)
        return False

    for port in [nginx_port, minio_console_port]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
            except socket.error as err:
                if err.errno == socket.errno.EADDRINUSE:
                    error_msg += f"Port {port} is occupied.\n"
                else:
                    error_msg += f"Error binding to port {port}: {err}\n"

    if error_msg:
        print(error_msg)
        return False

    return True


def is_vlm_model_path_valid(env_vars: dict[str, str]) -> bool:
    """
    VLM_INFERENCE_MODEL_PATH_ON_HOST should be a valid path.
    """
    print(f"Checking VLM_INFERENCE_MODEL_PATH_ON_HOST: {env_vars['VLM_INFERENCE_MODEL_PATH_ON_HOST']}")
    is_dir = os.path.isdir(env_vars['VLM_INFERENCE_MODEL_PATH_ON_HOST'])
    if not is_dir:
        print(f"[Error] VLM_INFERENCE_MODEL_PATH_ON_HOST {env_vars['VLM_INFERENCE_MODEL_PATH_ON_HOST']} is not a valid path.")
        return False

    has_safetensor = False
    for entry in os.scandir(env_vars['VLM_INFERENCE_MODEL_PATH_ON_HOST']):
        if entry.is_file() and entry.name.endswith(".safetensors"):
            # This is not a perfect way to check if the model is valid, but it's a good enough for now.
            has_safetensor = True
            break

    if not has_safetensor:
        print(f"[Error] VLM_INFERENCE_MODEL_PATH_ON_HOST {env_vars['VLM_INFERENCE_MODEL_PATH_ON_HOST']} does not contain a .safetensors file.")
        return False

    return True


def is_ddm_net_valid(env_vars: dict[str, str]) -> bool:
    """
    ACTION_SEGMENT_DDM_NET_CHECKPOINT_PATH_ON_HOST should be a valid path.
    """
    print(f"Checking ACTION_SEGMENT_DDM_NET_CHECKPOINT_PATH_ON_HOST: {env_vars['ACTION_SEGMENT_DDM_NET_CHECKPOINT_PATH_ON_HOST']}")
    is_file = os.path.isfile(env_vars['ACTION_SEGMENT_DDM_NET_CHECKPOINT_PATH_ON_HOST'])

    error_msg = ""
    if not is_file:
        error_msg += f"[Error] ACTION_SEGMENT_DDM_NET_CHECKPOINT_PATH_ON_HOST {env_vars['ACTION_SEGMENT_DDM_NET_CHECKPOINT_PATH_ON_HOST']} is not a valid path.\n"

    print(f"Checking ACTION_SEGMENT_DDM_NET_RESOLUTION: {env_vars['ACTION_SEGMENT_DDM_NET_RESOLUTION']}")
    try:
        ddm_resolution = int(env_vars["ACTION_SEGMENT_DDM_NET_RESOLUTION"])
    except ValueError as exc:
        ddm_resolution = None
        error_msg += f"[Error] ACTION_SEGMENT_DDM_NET_RESOLUTION {env_vars['ACTION_SEGMENT_DDM_NET_RESOLUTION']} must be an integer. Exception: {exc}\n"

    if error_msg:
        print(error_msg)
        return False

    return True


def is_uboco_model_path_valid(env_vars: dict[str, str]) -> bool:
    """
    ACTION_SEGMENT_UBOCO_*_ON_HOST should be a valid path.
    """
    env_names = [
        "ACTION_SEGMENT_UBOCO_SLOWFAST_PATH_ON_HOST",
        "ACTION_SEGMENT_UBOCO_BPE_PATH_ON_HOST",
        "ACTION_SEGMENT_UBOCO_VICLIP_PATH_ON_HOST",
    ]

    error_msg = ""
    for env_name in env_names:
        model_path = env_vars[env_name]
        is_file = os.path.isfile(model_path)
        if not is_file:
            error_msg += f"[Error] {env_name} {model_path} is not a valid path.\n"

    if error_msg:
        print(error_msg)
        return False

    return True

def print_separator():

    print("\n" + "=" * 80)

def main() -> None:

    default_env_file = os.path.join(_THIS_DIR, "..", "deployment", "docker_compose", ".env")
    default_env_file = os.path.abspath(default_env_file)

    parser = argparse.ArgumentParser(
        description="A script to check the environment variables."
    )

    parser.add_argument("-f", "--env_file",
                        type=str,
                        required=False,
                        default=default_env_file,
                        help=f"Path to the environment file. Default is {default_env_file}")

    args = parser.parse_args()
    env_file = args.env_file
    if not os.path.isfile(env_file):
        raise FileNotFoundError(f"Environment file not found: {env_file}")

    print("Begin to check environment variables.")
    print("Note that KeyError will be raised if necessary environment variables are not set.")

    env_vars = parse_env_file(env_file)

    print_separator()
    if not is_bp_name_unique(env_vars):
        print("Please set a unique BP_NAME.")
    else:
        print("[OK] BP_NAME is unique.")
    print_separator()
    if not is_port_valid(env_vars):
        print("Please set a valid port for NGINX_INGRESS_PORT and MINIO_CONSOLE_PORT.")
    else:
        print("[OK] NGINX_INGRESS_PORT and MINIO_CONSOLE_PORT are valid.")
    print_separator()
    if not is_vlm_model_path_valid(env_vars):
        print("Please set a valid path for VLM_INFERENCE_MODEL_PATH_ON_HOST.")
    else:
        print("[OK] VLM_INFERENCE_MODEL_PATH_ON_HOST seems valid.")
    print_separator()
    if not is_ddm_net_valid(env_vars):
        print("If you want to use DDM-Net, please set a valid path for ACTION_SEGMENT_DDM_NET_CHECKPOINT_PATH_ON_HOST.")
    else:
        print("[OK] DDM-Net seems valid.")
    print_separator()
    if not is_uboco_model_path_valid(env_vars):
        print("If you want to use UBOCO, please set a valid path for ACTION_SEGMENT_UBOCO_*_ON_HOST.")
    else:
        print("[OK] ACTION_SEGMENT_UBOCO_*_ON_HOST seems valid.")
    print_separator()



if __name__ == "__main__":
    main()