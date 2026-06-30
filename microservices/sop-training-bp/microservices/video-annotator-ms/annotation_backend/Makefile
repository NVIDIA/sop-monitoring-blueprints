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

# SOP annotation backend Makefile

# Variables
DOCKER_IMAGE_NAME ?= sop-annotation-backend
DOCKER_TAG ?= latest
CONTAINER_NAME = sop-annotation-backend-dev
HOST_PORT = 8100
CONTAINER_PORT = 8100

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build          - Build Docker image (tag: $(DOCKER_TAG))"
	@echo "  push           - Push Docker image to registry"
	@echo "  run            - Run container with GPU support"
	@echo "  run-interactive- Run container with GPU support (interactive)"
	@echo "  stop           - Stop and remove container"
	@echo "  clean          - Remove container and image"
	@echo ""
	@echo "Usage examples:"
	@echo "  make build                       # Build with default tag 'latest'"
	@echo "  make build DOCKER_TAG=v2.1.0     # Build with custom tag"
	@echo "  make push                        # Push with default tag 'latest'"
	@echo "  make push DOCKER_TAG=v2.1.0      # Push with custom tag"
	@echo "  make run                         # Run container with GPU support"
	@echo "  make run DOCKER_TAG=v2.1.0       # Run container with custom tag"
	@echo "  make run-interactive             # Run container with GPU support (interactive)"
	@echo "  make stop                        # Stop and remove container"
	@echo "  make clean                       # Remove container and image"

# Build Docker image
.PHONY: build
build:
	@echo "Building Docker image with tag: $(DOCKER_TAG)"
	docker build --no-cache --progress=plain \
		-t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		--network=host \
		-f Dockerfile \
		. \
		2>&1 | tee docker_build.log
	@echo "Build completed successfully!"

# Run container with GPU support
.PHONY: run
run:
	@echo "Starting container with GPU support..."
	docker run --name $(CONTAINER_NAME) \
		--gpus all \
		--ipc=host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-p $(HOST_PORT):$(CONTAINER_PORT) \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)
	@echo "Container started successfully!"
	@echo "API available at: http://localhost:$(HOST_PORT)"

# Stop and remove container
.PHONY: stop
stop:
	@echo "Stopping and removing container..."
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)
	@echo "Container stopped and removed!"

# Clean up container and image
.PHONY: clean
clean: stop
	@echo "Removing Docker images..."
	-docker rmi $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)
	@echo "Cleanup completed!"


# Interactive run (like the original run_docker.sh)
.PHONY: run-interactive
run-interactive:
	@echo "Starting interactive container with GPU support..."
	docker run -it \
		--rm \
		--gpus all \
		--ipc=host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-v "$(PWD)/assets:/app/assets" \
		-v "$(PWD)/utils:/app/utils" \
		-v "$(PWD)/components:/app/components" \
		-v "$(PWD)/exceptions:/app/exceptions" \
		-v "$(PWD)/validations:/app/validations" \
		-v "$(PWD)/inference.py:/app/inference.py" \
		-v "$(PWD)/run.sh:/app/run.sh" \
		-p $(HOST_PORT):$(CONTAINER_PORT) \
		--name $(CONTAINER_NAME) \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		/bin/bash

# Push Docker image to registry
.PHONY: push
push:
	@echo "Pushing Docker image to registry..."
	docker push $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)
	@echo "Push completed!"