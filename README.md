# SOP Monitoring Services (Training + Inference)

### Table of Contents

- [Overview](#overview)
- [When To Use Which Service](#when-to-use-which-service)
- [End-to-End Workflow](#end-to-end-workflow)
- [Agentic Quick Start](#agentic-quick-start)
- [VSS Example Application](#vss-example-application)
- [Usage](#usage)
- [Sample Data](#sample-data)
- [License](#license)


## Overview

Build, train, and deploy a complete SOP (Standard Operating Procedure) monitoring system. It comprises **two complementary microservices** and **three agentic skills packages**.

**Microservices**

- Training Microservices ([microservices/sop-training-bp/README.md](microservices/sop-training-bp/README.md)): prepares a SOP-specific Vision-Language Model (VLM) and Temporal Segmentation Model.
- Inference Microservices ([microservices/sop-inference-bp/README.md](microservices/sop-inference-bp/README.md)): a low-latency, real-time DeepStream SOP inference microservice, provided as the **reference implementation** that the DeepStream SOP skill generates against, debugs, evaluates, and benchmarks.

**Agentic Skills**

- SOP Fine-tuning Agentic Skills ([agentic/sop-agentic-ft/README.md](agentic/sop-agentic-ft/README.md)): AI coding assistant skills for fine-tuning SOP monitoring models — data augmentation, DDM + Cosmos-Reason fine-tuning, evaluation, RCA, and orchestration.
- DeepStream SOP Agentic Skills ([agentic/ds-sop-skills/README.md](agentic/ds-sop-skills/README.md)): AI coding assistant skills for the DeepStream SOP inference microservice — auto code generation, customization, evaluation, camera/live-stream setup, and performance benchmarking.
- VSS SOP Skills ([agentic/vss-sop-skills/README.md](agentic/vss-sop-skills/README.md)): AI coding assistant skill to integrate VSS with the DeepStream SOP microservice.


## When To Use Which Service

- Use the Training Service if:

  - You need to create or refine a SOP-aware model from your own annotated videos.
  - You want to programmatically generate QA training data for fine-tuning.
- Use the Inference Service if:

  - You already have a trained SOP model (e.g., from the Training Service).
  - You want to deploy SOP monitoring as an API and web application.

These services are designed to work together: train a model with the Training Service, then serve it with the Inference Service.


## End-to-End Workflow

![SOP Fine-Tuning + Inference Agentic Workflow](assets/SOP-FT-Inference-Agentic-Workflow.png)

1. Annotate videos by marking action start/end timestamps (Training).
2. Generate QA pairs (GQA/BCQ/MCQ) from annotations (Training).
3. Fine-tune a VLM (e.g., Cosmos-Reason1) on generated data (Training).
4. Fine-tune a Temporal Segment Model on the annotated data (Training).
5. Deploy the trained model into the Inference Service (Inference).
6. Conduct end-to-end SOP monitoring and analyze SOP compliance (Inference).


## Agentic Quick Start

Two agent-driven paths you can run independently: fine-tune the DDM and Cosmos-Reason models, and generate / evaluate a DeepStream SOP inference microservice — each driven from natural-language prompts to an AI coding agent.

### Fine-tune the models (SOP Fine-tuning Agentic Skills)

1. **Install the fine-tuning skills** — follow [microservices/sop-training-bp/AGENTIC_README.md](microservices/sop-training-bp/AGENTIC_README.md) to register the `sop-agentic-ft` marketplace and install **all six plugins**.
2. **Drive the fine-tuning loop** with a prompt such as:

   ```
   I want to fine-tune SOP monitoring on <training_data> and evaluate on <testing_data>
   targeting seq_accuracy >= <KPI> and max iterations set to <X>.
   Report status every 10 minutes.
   ```

   This produces the fine-tuned **DDM** (temporal action detection) and **Cosmos-Reason** (VLM) model checkpoints.

### Generate & deploy the inference microservice (DeepStream SOP Agentic Skills)

With the DDM and Cosmos-Reason model checkpoints ready (fine-tuned above, or your own):

1. **Install the DeepStream SOP skill** — follow [agentic/ds-sop-skills/README.md](agentic/ds-sop-skills/README.md).
2. **Generate the source code & microservice:**

   ```
   Please follow instructions in agentic/ds-sop-skills/example_sop_prompt.md to generate a
   DeepStream SOP Inference source code and microservice in folder @ds_sop_microservice
   ```
3. **Evaluate, bug-fix & benchmark** — after the generated code is ready and dependency checks pass:

   ```
   Follow agentic/ds-sop-skills/eval_sop_prompt.md to evaluate microservice @ds_sop_microservice,
   fix any issues found during evaluation, and measure the latency benchmark for file input,
   with env settings MODEL_ROOT_DIR=..., DDM_MODEL_PATH=..., VLLM_MODEL_PATH=..., VLM_FPS=..., ...
   ```

   Make sure `MODEL_ROOT_DIR`, `DDM_MODEL_PATH`, `VLLM_MODEL_PATH`, `VLM_FPS`, `VLM_MAX_PIXELS`, `ACTION_CONFIG_PATH`, `VLM_PROMPT_PATH`, and `TEST_VIDEO_PATH` are set correctly according to your dataset and model config. See [DeepStream SOP Microservice Evaluation](agentic/ds-sop-skills/README.md#2-deepstream-sop-microservice-evaluation) for the full list and details.
4. **(Optional) Basler camera evaluation & benchmark** — for a physical GigE camera:

   ```
   Please evaluate @ds_sop_microservice with physical Basler camera serial number 12345678,
   using max_length_sec=2.0. Follow the deepstream-sop skill for rules.
   ```

   Replace the serial number with your local physical GigE Basler camera (tested on a2A2048-37gcPRO).

## VSS Example Application

![VSS SOP Architecture](agentic/vss-sop-skills/vss-sop-build/references/diagrams/VSS%20SOP%20Blueprint%20Architecture.png)

The VSS SOP application is built, deployed, and validated using modular lifecycle skills:

1. **Install the VSS SOP skills** — follow [agentic/vss-sop-skills/README.md](agentic/vss-sop-skills/README.md) to install the `vss-sop` skills.
2. **Install the DS SOP skills** — follow [agentic/ds-sop-skills/README.md](agentic/ds-sop-skills/README.md) to install the `ds-sop` skills.
3. **Build, deploy & test the full SOP pipeline** — drive it end-to-end with a single prompt:

   ```
   Run the full SOP pipeline using the sop-build skill.
   ```

   This runs the full lifecycle:

   - **Build DeepStream SOP microservice**: the [`deepstream-sop` (ds sop) skill](agentic/ds-sop-skills/deepstream-sop/SKILL.md) generates and evaluates the core DeepStream SOP microservice source code.
   - **Build VSS SOP app**: the [`vss-sop-build` skill](agentic/vss-sop-skills/vss-sop-build/SKILL.md) builds the VSS SOP application on top of standard VSS components.
   - **Deploy VSS SOP app**: the [`vss-sop-deploy` skill](agentic/vss-sop-skills/vss-sop-deploy/SKILL.md) performs preflight checks, verifies models, downloads sample assets, and launches all containerized microservices.
   - **Test VSS SOP app**: the [`vss-sop-test` skill](agentic/vss-sop-skills/vss-sop-test/SKILL.md) executes the post-deployment validation suite and verifies end-to-end functionality.

   > **Alternatively**, instead of running the full 4-stage pipeline as described above, you can [pre-build the ds-sop image](microservices/sop-inference-bp/docker/Docker.build). After that, simply run the VSS SOP pipeline using the `vss-sop-build` skill. The pipeline will then consist of only 3 stages: **Build VSS SOP**, **Deploy VSS SOP**, and **Test VSS SOP**.
   >
   > Command to call the skill:
   > ```
   > /vss-sop-build
   > ```

## Usage

Please refer to:

- Training: [microservices/sop-training-bp/README.md](microservices/sop-training-bp/README.md)
- Inference: [microservices/sop-inference-bp/README.md](microservices/sop-inference-bp/README.md)
- SOP Fine-tuning Agentic Skills: [microservices/sop-training-bp/AGENTIC_README.md](microservices/sop-training-bp/AGENTIC_README.md)
- DeepStream SOP Agentic Skills: [agentic/ds-sop-skills/README.md](agentic/ds-sop-skills/README.md)
- VSS SOP Skills: [agentic/vss-sop-skills/README.md](agentic/vss-sop-skills/README.md)

## Sample Data

We provide [sample data](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/sop-server-fan-installation-data?version=1.0-260213) which can be used for testing the SOP Training and Inference BP.
The sample data is about installing server fan and power.

## License

This project is dual-licensed: source code under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) and documentation under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/), per the `CC-BY-4.0 AND Apache-2.0` terms in the top-level [`LICENSE`](./LICENSE).

This project bundles and/or downloads third-party open-source software, each under its own license. See [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md) for the third-party components distributed in this repository, and review the license terms of any additionally downloaded open-source projects before use.