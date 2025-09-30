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
Module for checker service.

This service is responsible for checking SOP (Standard Operating Procedure) compliance.

It will:
- Get a request from the API server
- Receive action JSON and VLM outputs
- Process the data to check SOP compliance
- Send a message to the API server to indicate the completion of the SOP checking and the results
"""

import os
import logging
import time
import json
import uuid
import re
import traceback

from .constants import (
    REDIS_STREAM_P_API_SERVER_C_SOP_CHECKER_STREAM_NAME,
    REDIS_STREAM_C_SOP_CHECKER_P_API_SERVER_GROUP_NAME,
    REDIS_STREAM_DB_INDEX,
)
from .msg_types import (
    SopCheckerRequest,
    SopCheckerResponse,
)
from .redis_stream import (
    RedisStream,
    send_response,
)
from .utils import create_redis_client

from ..missing_number_detector import MissingNumberDetector

_LOGGER = logging.getLogger(__name__)

_TMP_DIR = "/dev/shm"

_CHECKER_CACHE = {}

def read_sop_steps(content: str) -> list[str]:
    """Parse SOP steps from content string and return them as a list.

    Args:
        content (str): The content string containing SOP steps

    Returns:
        list[str]: List of SOP steps in order of appearance
    """
    # Initialize list to store SOP steps
    sop_steps = []

    # Normalize newlines in content
    content = content.replace('\r\n', '\n').replace('\r', '\n')

    # Process content directly
    step_start = 0
    while True:
        # Find next opening parenthesis
        open_paren = content.find('(', step_start)
        if open_paren == -1:
            break

        # Find closing parenthesis
        close_paren = content.find(')', open_paren)
        if close_paren == -1:
            break

        # Extract step number and content
        step_num = content[open_paren + 1:close_paren]

        # Find start of next step or end of content
        next_step = content.find('(', close_paren + 1)
        if next_step == -1:
            step_content = content[close_paren + 1:]
        else:
            step_content = content[close_paren + 1:next_step]

        # Replace any newlines with spaces in the step content
        step_content = step_content.replace('\n', ' ')
        # Strip the trailing space
        step_content = step_content.rstrip()

        # Add the full step to our list
        sop_steps.append(f"({step_num}){step_content}")

        # Move to next potential step
        step_start = close_paren + 1

    return sop_steps

def load_checker(checker_id: str) -> tuple[MissingNumberDetector, list[int]]:
    """
    Load the checker from the cache.
    """
    if checker_id in _CHECKER_CACHE:
        return _CHECKER_CACHE[checker_id]
    raise ValueError(f"Checker {checker_id} not found. If this is the first request, please use '*' as the checker_id.")


def save_checker(checker: MissingNumberDetector, actions_can_be_skipped: list[int]) -> str:
    """
    Save the checker to the cache.
    """
    checker_id = f"sopchecker-{str(uuid.uuid4())}"
    _CHECKER_CACHE[checker_id] = (checker, actions_can_be_skipped)
    _LOGGER.debug("Checker is saved to to cache with id: %s", checker_id)
    return checker_id

def try_delete_checker(checker_id: str):
    """
    Try to delete the checker from the cache.
    """
    if checker_id in _CHECKER_CACHE:
        del _CHECKER_CACHE[checker_id]
        _LOGGER.debug("Checker is deleted from cache with id: %s", checker_id)


def process_sop_check(request_id: str, sop_checker_request: SopCheckerRequest) -> SopCheckerResponse:
    """
    Process SOP checking logic.

    Args:
        action_json: The content of the actions.json
        vlm_outputs: The outputs of the VLM inference
        draw_fsm: Whether to draw the FSM image

    Returns:
        tuple: (num_sop_detected, sop_final_state, fsm_image)
    """

    action_json = sop_checker_request.action_json
    vlm_output = sop_checker_request.vlm_output
    keep_alive = sop_checker_request.keep_alive
    checker_id = sop_checker_request.checker_id
    cycle_completion_threshold = sop_checker_request.cycle_completion_threshold
    cycle_boundary_threshold_low = sop_checker_request.cycle_boundary_threshold_low
    cycle_boundary_threshold_high = sop_checker_request.cycle_boundary_threshold_high

    reobj_capture_number = re.compile(r"^\((\d+)\).+")

    if checker_id == "*":
        sop_data = json.loads(action_json)

        actions = sop_data["actions"]
        action_numbers = [int(reobj_capture_number.match(action).group(1)) for action in actions]

        actions_can_be_skipped = sop_data.get("actions_can_be_skipped", [])
        actions_can_be_skipped_numbers = [int(reobj_capture_number.match(action).group(1)) for action in actions_can_be_skipped]

        action_numbers = [num for num in action_numbers if num not in actions_can_be_skipped_numbers]
        index_to_action_number = {i+1: action_numbers[i] for i in range(len(action_numbers))}

        _LOGGER.debug("Creating checker with options: "
                      "cycle_completion_threshold=%s, "
                      "cycle_boundary_threshold_low=%s, "
                      "cycle_boundary_threshold_high=%s",
                      cycle_completion_threshold,
                      cycle_boundary_threshold_low,
                      cycle_boundary_threshold_high)
        checker = MissingNumberDetector(len(action_numbers),
                                        index_to_action_number,
                                        cycle_completion_threshold=cycle_completion_threshold,
                                        cycle_boundary_threshold_low=cycle_boundary_threshold_low,
                                        cycle_boundary_threshold_high=cycle_boundary_threshold_high)

    else:
        _LOGGER.debug("Loading checker with id: %s", checker_id)
        checker, actions_can_be_skipped_numbers = load_checker(checker_id)

    sop_steps = read_sop_steps(vlm_output)
    _LOGGER.info("SOP steps: %s", "\n".join(sop_steps))

    response_missing_detected = []
    response_misordered_detected = []
    response_final_missing_detected = []
    response_final_misordered_detected = []
    response_cycle_completed = False
    response_cycle = checker.current_cycle
    response_summary_cycles_detected = []
    response_summary_cycle_analysis = []

    for sop_step in sop_steps:

        sop_action_number = reobj_capture_number.match(sop_step)
        if sop_action_number is None:
            _LOGGER.warning("Cannot capture action number for SOP step: %s", sop_step)
            continue

        sop_action_number = int(sop_action_number.group(1))

        if sop_action_number in actions_can_be_skipped_numbers:
            _LOGGER.info("This action can be skipped: '%s'", sop_step)
            continue

        detection_result = checker.process_number(sop_action_number)

        if not detection_result:
            _LOGGER.error("Detection result is empty for sop_step: %s", sop_step)
            continue

        response_missing_detected.extend(detection_result["missing_detected"])
        response_misordered_detected.extend(detection_result["misordered_detected"])
        response_cycle_completed = detection_result["cycle_completed"]
        response_cycle = detection_result["cycle"]
    if keep_alive and checker_id == "*":
        checker_id = save_checker(checker, actions_can_be_skipped_numbers)
    elif not keep_alive:
        detection_result = checker.finalize_processing()
        response_final_missing_detected = detection_result["final_missing_detected"]
        response_final_misordered_detected = detection_result["final_misordered_detected"]
        response_cycle_completed = detection_result["final_cycle_completed"]
        print_summary = checker.print_summary()
        response_summary_cycles_detected = print_summary["cycles_detected"]
        response_summary_cycle_analysis = print_summary["cycle_analysis"]

        try_delete_checker(checker_id)

    return SopCheckerResponse(
        request_id=request_id,
        error_message="",
        checker_id=checker_id,
        cycle=response_cycle,
        missing_detected=response_missing_detected,
        misordered_detected=response_misordered_detected,
        final_missing_detected=response_final_missing_detected,
        final_misordered_detected=response_final_misordered_detected,
        cycle_completed=response_cycle_completed,
        summary_cycles_detected=response_summary_cycles_detected,
        summary_cycle_analysis=response_summary_cycle_analysis,
    )

def main():
    log_level_name = os.environ.get("CHECKER_SERVICE_LOG_LEVEL", "INFO")
    try:
        log_level = getattr(logging, log_level_name.upper())
    except AttributeError:
        _LOGGER.error("Invalid log level: %s. Using INFO instead.", log_level_name)
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s"
    )

    redis_client_for_stream = create_redis_client(REDIS_STREAM_DB_INDEX)
    _LOGGER.info("Testing Redis client at %s:%s",
                 redis_client_for_stream.connection_pool.connection_kwargs['host'],
                 redis_client_for_stream.connection_pool.connection_kwargs['port'])
    # this will raise exception if connection fails
    redis_client_for_stream.ping()
    redis_stream = RedisStream(redis_client_for_stream)
    redis_stream.create_consumer_group(REDIS_STREAM_P_API_SERVER_C_SOP_CHECKER_STREAM_NAME,
                                       REDIS_STREAM_C_SOP_CHECKER_P_API_SERVER_GROUP_NAME)

    _LOGGER.info("Checker service started.")
    # main event loop.
    while True:

        try:
            messages = redis_stream.xreadgroup(
                REDIS_STREAM_P_API_SERVER_C_SOP_CHECKER_STREAM_NAME,
                REDIS_STREAM_C_SOP_CHECKER_P_API_SERVER_GROUP_NAME,
                "checker_worker",
                SopCheckerRequest,
                count=10,
                block=0,
            )
        except Exception as e:
            _LOGGER.error("Error reading from Redis stream: %s", e)
            time.sleep(5)  # Wait before retrying to avoid tight error loop
            continue

        # TODO: can be parallelized.
        for message_id, sop_checker_request in messages:
            try:
                _LOGGER.info("Received SOP checker request: %s", message_id)

                # Process the SOP checking
                sop_checker_response = process_sop_check(message_id, sop_checker_request)

            except Exception as e:
                error_message = traceback.format_exc()
                _LOGGER.error("request_id: %s, Error in checker service: %s", message_id, error_message)
                sop_checker_response = SopCheckerResponse(
                    request_id=message_id,
                    checker_id="",
                    cycle=0,
                    missing_detected=[],
                    misordered_detected=[],
                    final_missing_detected=[],
                    final_misordered_detected=[],
                    cycle_completed=False,
                    summary_cycles_detected=[],
                    summary_cycle_analysis=[],
                    error_message=error_message,
                )
            finally:
                _LOGGER.debug("Acknowledged message %s", message_id)
                send_response(redis_stream, sop_checker_request, sop_checker_response)
                redis_stream.ack(REDIS_STREAM_P_API_SERVER_C_SOP_CHECKER_STREAM_NAME,
                                 REDIS_STREAM_C_SOP_CHECKER_P_API_SERVER_GROUP_NAME,
                                 message_id)

if __name__ == "__main__":
    if not os.path.isdir(_TMP_DIR):
        _LOGGER.error("Temporary directory %s does not exist. Please check if the directory is mounted.", _TMP_DIR)
        raise RuntimeError(f"Temporary directory {_TMP_DIR} does not exist. Please check if the directory is mounted.")
    main()
