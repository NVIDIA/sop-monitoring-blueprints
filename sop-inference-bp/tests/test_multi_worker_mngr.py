# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import os
import sys
import time

import pytest

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sop_monitoring.multi_gpu_utils import MultiWorkerManager


class SimpleTestWorker(MultiWorkerManager.Worker):
    """A simple test worker that performs basic arithmetic operations."""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.initialized = False

    def get_name(self) -> str:
        return f"TestWorker-{self.worker_id}"

    def initialize(self) -> None:
        """Initialize the worker."""
        self.initialized = True
        time.sleep(0.1)  # Simulate some initialization time

    def process_request(self, request: dict) -> dict:
        """Process a request and return a response."""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")

        operation = request.get("operation")
        operands = request.get("operands", [])

        if operation == "add":
            result = sum(operands)
        elif operation == "multiply":
            result = 1
            for num in operands:
                result *= num
        elif operation == "echo":
            result = request.get("message", "")
        elif operation == "error":
            raise ValueError("Intentional test error")
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return {
            "worker_id": self.worker_id,
            "operation": operation,
            "result": result
        }


class TestMultiWorkerManager:
    """Test cases for MultiWorkerManager."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.workers = [SimpleTestWorker(i) for i in range(2)]
        self.manager = None

    def teardown_method(self):
        """Clean up after each test method."""
        if self.manager:
            self.manager.shutdown()
            time.sleep(0.1)  # Give processes time to shut down

    def test_manager_initialization(self):
        """Test that the manager initializes correctly with workers."""
        self.manager = MultiWorkerManager(self.workers, max_queue_size=10, timeout_sec=30)

        # Give some time for worker processes to start
        time.sleep(2)

        # Manager should be created successfully
        assert self.manager is not None
        assert len(self.manager._worker_processes) == 2

        # All worker processes should be alive
        for worker_process in self.manager._worker_processes:
            assert worker_process.process.is_alive()

    def test_simple_request_response(self):
        """Test basic request-response functionality."""
        self.manager = MultiWorkerManager(self.workers, max_queue_size=10, timeout_sec=30)
        time.sleep(2)  # Wait for worker initialization

        # Submit a simple addition request
        request = {"operation": "add", "operands": [1, 2, 3]}
        future = self.manager.submit_request(request)

        # Get the result
        response = future.result(timeout=10)

        assert response["operation"] == "add"
        assert response["result"] == 6
        assert "worker_id" in response

    def test_multiple_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        self.manager = MultiWorkerManager(self.workers, max_queue_size=10, timeout_sec=30)
        time.sleep(2)  # Wait for worker initialization

        # Submit multiple requests
        futures = []
        for i in range(5):
            request = {"operation": "multiply", "operands": [i + 1, 2]}
            future = self.manager.submit_request(request)
            futures.append(future)

        # Collect all results
        results = []
        for future in futures:
            response = future.result(timeout=10)
            results.append(response)

        # Verify all requests were processed
        assert len(results) == 5

        # Verify results are correct
        for i, response in enumerate(results):
            assert response["operation"] == "multiply"
            assert response["result"] == (i + 1) * 2

    def test_echo_operation(self):
        """Test echo operation to verify request data integrity."""
        self.manager = MultiWorkerManager(self.workers, max_queue_size=10, timeout_sec=30)
        time.sleep(2)  # Wait for worker initialization

        message = "Hello from test!"
        request = {"operation": "echo", "message": message}
        future = self.manager.submit_request(request)

        response = future.result(timeout=10)

        assert response["operation"] == "echo"
        assert response["result"] == message

    def test_error_handling(self):
        """Test that worker errors are properly propagated."""
        self.manager = MultiWorkerManager(self.workers, max_queue_size=10, timeout_sec=30)
        time.sleep(2)  # Wait for worker initialization

        # Submit a request that will cause an error
        request = {"operation": "error"}
        future = self.manager.submit_request(request)

        # The future should raise an exception
        with pytest.raises(Exception) as excinfo:
            future.result(timeout=10)

        assert "Intentional test error" in str(excinfo.value)

    def test_invalid_operation(self):
        """Test handling of invalid operations."""
        self.manager = MultiWorkerManager(self.workers, max_queue_size=10, timeout_sec=30)
        time.sleep(2)  # Wait for worker initialization

        # Submit a request with invalid operation
        request = {"operation": "invalid_op", "operands": [1, 2]}
        future = self.manager.submit_request(request)

        # The future should raise an exception
        with pytest.raises(Exception) as excinfo:
            future.result(timeout=10)

        assert "Unknown operation" in str(excinfo.value)

    def test_shutdown(self):
        """Test proper shutdown of the manager."""
        self.manager = MultiWorkerManager(self.workers, max_queue_size=10, timeout_sec=30)
        time.sleep(2)  # Wait for worker initialization

        # Verify processes are alive before shutdown
        for worker_process in self.manager._worker_processes:
            assert worker_process.process.is_alive()

        # Shutdown the manager
        self.manager.shutdown()
        time.sleep(1)  # Give processes time to terminate

        # Verify processes are terminated after shutdown
        for worker_process in self.manager._worker_processes:
            assert not worker_process.process.is_alive()
