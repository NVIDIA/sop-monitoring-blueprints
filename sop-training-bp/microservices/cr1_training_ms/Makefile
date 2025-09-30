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

# VLM Training Microservice Makefile

# Variables
DOCKER_IMAGE_NAME ?= cr1_ftms
DOCKER_TAG ?= latest
CONTAINER_NAME = cr1-ms-test
HOST_PORT = 32080
CONTAINER_PORT = 32080

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build          - Build Docker image (tag: $(DOCKER_TAG))"
	@echo "  push           - Push Docker image to registry"
	@echo "  run            - Run container"
	@echo "  run-interactive- Run container in interactive mode"
	@echo "  stop           - Stop and remove container"
	@echo "  clean          - Remove container and image"
	@echo ""
	@echo "Usage examples:"
	@echo "  make build                                   # Build with default tag 'latest'"
	@echo "  make build DOCKER_TAG=v2.1.0                 # Build with custom tag"
	@echo "  make push                                    # Push with default tag 'latest'"
	@echo "  make push DOCKER_TAG=v2.1.0                  # Push with custom tag"
	@echo "  make run                                     # Run with default tag 'latest'"	
	@echo "  make run DOCKER_TAG=v2.1.0                   # Run with custom tag"
	@echo "  make run-interactive                         # Run with default tag 'latest' (interactive)"
	@echo "  make run-interactive DOCKER_TAG=v2.1.0       # Run with custom tag (interactive)"
	@echo "  make stop                                    # Stop and remove container"
	@echo "  make clean                                   # Remove container and image"


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

# Run container with GPU support
.PHONY: run
run:
	@echo "Starting container with GPU support..."
	@echo "API would beavailable at: http://localhost:$(HOST_PORT)"
	docker run --name $(CONTAINER_NAME) \
		--gpus all \
		--ipc=host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-v "$(PWD)/assets/config:/workspace/sop-cr1-ftms/assets/config" \
		-v "$(PWD)/assets/data:/workspace/sop-cr1-ftms/assets/data" \
		-v "$(PWD)/assets/weights:/workspace/sop-cr1-ftms/assets/weights" \
		-v "$(PWD)/assets/results:/workspace/sop-cr1-ftms/assets/results" \
		-v "$(PWD)/assets/tools:/workspace/sop-cr1-ftms/assets/tools" \
		-p $(HOST_PORT):$(CONTAINER_PORT) \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

# Stop and remove container
.PHONY: stop
stop:
	@echo "Stopping and removing container..."
	@echo "Container name: $(CONTAINER_NAME)"
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

# Clean up container and image
.PHONY: clean
clean: stop
	@echo "Removing Docker images..."
	@echo "Image name: $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)"
	-docker rmi $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

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
		-v "$(PWD)/assets:/workspace/sop-cr1-ftms/assets" \
		-v "$(PWD)/utils:/workspace/sop-cr1-ftms/utils" \
		-v "$(PWD)/validation:/workspace/sop-cr1-ftms/validation" \
		-v "$(PWD)/components:/workspace/sop-cr1-ftms/components" \
		-v "$(PWD)/app.py:/workspace/sop-cr1-ftms/app.py" \
		-v "$(PWD)/run.sh:/workspace/sop-cr1-ftms/run.sh" \
		-p $(HOST_PORT):$(CONTAINER_PORT) \
		--name $(CONTAINER_NAME) \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		/bin/bash

# Run dev mode
.PHONY: run-dev
run-dev:
	@echo "Starting container with GPU support..."
	@echo "API would be available at: http://localhost:$(HOST_PORT)"
	docker run --name $(CONTAINER_NAME) \
		--gpus all \
		--ipc=host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-e RELOAD_FLAG=true \
		-v "$(PWD)/assets:/workspace/sop-cr1-ftms/assets" \
		-v "$(PWD)/utils:/workspace/sop-cr1-ftms/utils" \
		-v "$(PWD)/validation:/workspace/sop-cr1-ftms/validation" \
		-v "$(PWD)/components:/workspace/sop-cr1-ftms/components" \
		-v "$(PWD)/app.py:/workspace/sop-cr1-ftms/app.py" \
		-v "$(PWD)/run.sh:/workspace/sop-cr1-ftms/run.sh" \
		-p $(HOST_PORT):$(CONTAINER_PORT) \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

# Push Docker image to registry
.PHONY: push
push:
	@echo "Pushing Docker image to registry..."
	docker push $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)
	@echo "Push completed!"



