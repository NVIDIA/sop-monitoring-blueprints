# SOP Monitoring Blueprints (Training + Inference)

### Table of Contents
- [Overview](#overview)
- [When To Use Which Blueprint](#when-to-use-which-blueprint)
- [End-to-End Workflow](#end-to-end-workflow)
- [Usage](#usage)
- [Sample Data](#sample-data)
- [License](#license)


## Overview

Build, train, and deploy a complete SOP (Standard Operating Procedure) monitoring system with two complementary blueprints:
- Training Blueprint ([sop-training-bp/README.md](sop-training-bp/README.md)): prepares a SOP-specific Vision-Language Model (VLM) and Temporal Segmentation Model.
- Inference Blueprint ([sop-inference-bp/README.md](sop-inference-bp/README.md)): serves the trained model behind a scalable DeepStream-SOP.


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
4. Fine-tune a Temporal Segment Model on the annotated data (Training).
5. Deploy the trained model into the Inference Blueprint (Inference).
6. Conduct end-to-end SOP monitoring and analyze SOP compliance (Inference).


## Usage
Please refer to:
- Training: [sop-training-bp/README.md](sop-training-bp/README.md)
- Inference: [sop-inference-bp/README.md](sop-inference-bp/README.md)

## Sample Data
We provide [sample data](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/sop-server-fan-installation-data?version=1.0-260213) which can be used for testing the SOP Training and Inference BP.
The sample data is about installing server fan and power.

## License
The software and materials in this repository are governed by the [NVIDIA Software and Model Evaluation License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-and-model-evaluation-license/)


This project will download and install additional third-party open-source software projects. Review the license terms of these open-source projects before use.