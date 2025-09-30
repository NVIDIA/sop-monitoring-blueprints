# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
MAKEFILE_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
PRJ_ROOT := $(realpath $(MAKEFILE_DIR)/..)

# Version for image tagging
VERSION := 1.0.0

API_SERVER_IMG_NAME := sop_inference_bp:api_server_$(VERSION)
VLM_INFERENCE_SERVICE_IMG_NAME := sop_inference_bp:vlm_inference_service_$(VERSION)
ACTION_SEGMENT_IMG_NAME := sop_inference_bp:action_segment_$(VERSION)
ACTION_SEGMENT_CR_IMG_NAME := sop_inference_bp:action_segment_cr_$(VERSION)
CHECKER_SERVICE_IMG_NAME := sop_inference_bp:checker_service_$(VERSION)
DEMO_WEB_APP_IMG_NAME := sop_inference_bp:demo_web_app_$(VERSION)

.PHONY: build_services
build_services: build_api_server \
	            build_action_segment \
	            build_action_segment_cr \
	            build_vlm_inference_service \
	            build_checker_service \
	            build_demo_web_app
	@echo "All services built."

build_api_server:
	docker build -f Dockerfile.api_server \
	-t $(API_SERVER_IMG_NAME) \
	--progress=plain \
	$(PRJ_ROOT) 2>&1

build_action_segment:
	docker build -f Dockerfile.action_segment \
	-t $(ACTION_SEGMENT_IMG_NAME) \
	--progress=plain \
	$(PRJ_ROOT) 2>&1

build_action_segment_cr: build_vlm_inference_service
	docker build -f Dockerfile.action_segment_cr \
	--build-arg BASE_IMAGE=$(VLM_INFERENCE_SERVICE_IMG_NAME) \
	-t $(ACTION_SEGMENT_CR_IMG_NAME) \
	--progress=plain \
	$(PRJ_ROOT) 2>&1

build_vlm_inference_service:
	docker build -f Dockerfile.vlm_service \
	-t $(VLM_INFERENCE_SERVICE_IMG_NAME) \
	--progress=plain \
	$(PRJ_ROOT) 2>&1

build_checker_service:
	docker build -f Dockerfile.checker_service \
	-t $(CHECKER_SERVICE_IMG_NAME) \
	--progress=plain \
	$(PRJ_ROOT) 2>&1

build_demo_web_app:
	docker build -f Dockerfile.demo_web_app \
	-t $(DEMO_WEB_APP_IMG_NAME) \
	--progress=plain \
	$(PRJ_ROOT) 2>&1
