# SOP Monitoring Blueprints (Training + Inference)

### Table of Contents
- [Overview](#overview)
- [Blueprints Introduction](#blueprints-introduction)
- [When To Use Which Blueprint](#when-to-use-which-blueprint)
- [End-to-End Workflow](#end-to-end-workflow)
- [Usage](#usage)
- [License](#license)


## Overview

Build, train, and deploy a complete SOP (Standard Operating Procedure) monitoring system with two complementary blueprints:
- Training Blueprint ([sop-training-bp/README.md](sop-training-bp/README.md)): prepares a SOP-specific Vision-Language Model (VLM).
- Inference Blueprint ([sop-inference-bp/README.md](sop-inference-bp/README.md)): serves the trained model behind a scalable, microservices-based API and UI.


## Blueprints Introduction

- SOP Training Blueprint
  - A reference UI to annotate videos into action-aligned segments
  - Automated generation of QA pairs (GQA, BCQ, MCQ) using NVIDIA LLM NIM or local LLM
  - Fine-tuning pipeline for SOP-aware VLMs (currently supports Cosmos-Reason1)
  - Storage and metadata management with Postgres and Adminer

- SOP Inference Blueprint
  - OpenAI-compatible REST API for VLM inference and file management
  - Action segmentation service for automatic chunking
  - SOP detection API to analyze compliance, detect cycles, and summarize
  - Scalable, GPU-aware VLM inference service with load balancing
  - Reference web UI for uploads, visualization, and results


## When To Use Which Blueprint

- Use the Training Blueprint if:
  - You need to create or refine a SOP-aware model from your own annotated videos.
  - You want to programmatically generate QA training data for fine-tuning.

- Use the Inference Blueprint if:
  - You already have a trained SOP model (e.g., from the Training Blueprint).
  - You want to deploy SOP monitoring as an API and web application.

These blueprints are designed to work together: train a model with the Training Blueprint, then serve it with the Inference Blueprint.


## End-to-End Workflow

1. Annotate videos by marking action start/end timestamps (Training).
2. Generate QA pairs (GQA/BCQ/MCQ) from annotations (Training).
3. Fine-tune a VLM (e.g., Cosmos-Reason1) on generated data (Training).
4. Deploy the trained model into the Inference Blueprint (Inference).
5. Upload videos, run inference with optional chunking, and analyze SOP compliance (Inference).


## Usage
Please refer to:
- Training: [sop-training-bp/README.md](sop-training-bp/README.md)
- Inference: [sop-inference-bp/README.md](sop-inference-bp/README.md)

## License
The software and materials in this repository are governed by the [NVIDIA Software and Model Evaluation License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-and-model-evaluation-license/)


This project will download and install additional third-party open-source software projects. Review the license terms of these open-source projects before use.