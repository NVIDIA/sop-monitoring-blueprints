# SOP Monitoring Services (Training + Inference)

### Table of Contents
- [Overview](#overview)
- [When To Use Which Service](#when-to-use-which-service)
- [End-to-End Workflow](#end-to-end-workflow)
- [VSS Example Application](#vss-example-application)
- [Usage](#usage)
- [Sample Data](#sample-data)
- [License](#license)


## Overview

Build, train, and deploy a complete SOP (Standard Operating Procedure) monitoring system with two complementary services and two agentic skills packages:
- Training Service ([microservices/sop-training-bp/README.md](microservices/sop-training-bp/README.md)): prepares a SOP-specific Vision-Language Model (VLM) and Temporal Segmentation Model.
- Inference Service ([microservices/sop-inference-bp/README.md](microservices/sop-inference-bp/README.md)): serves the trained model behind a scalable DeepStream-SOP.
- Agentic Fine-tuning Skills ([agentic/sop-agentic-ft/README.md](agentic/sop-agentic-ft/README.md)): AI coding assistant skill and reference materials for fine-tuning SOP monitoring models.
- Agentic Skills ([agentic/ds-sop-skills/README.md](agentic/ds-sop-skills/README.md)): AI coding assistant skill and reference materials for building and extending the DeepStream SOP microservice.
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


## VSS Example Application

![VSS SOP Architecture](agentic/vss-sop-skills/vss-sop-build/references/diagrams/VSS%20SOP%20Blueprint%20Architecture.png)

The VSS SOP application is built, deployed, and validated in four key phases using modular lifecycle skills:

1. **Build the DeepStream SOP microservice**: Use the [`deepstream-sop` (ds sop) skill](agentic/ds-sop-skills/deepstream-sop/SKILL.md) to generate and evaluate the core DeepStream SOP microservice source code.
2. **Build VSS SOP app**: Use the [`vss-sop-build` skill](agentic/vss-sop-skills/vss-sop-build/SKILL.md) to build the VSS SOP application on top of standard VSS components.
3. **Deploy VSS SOP app**: Use the [`vss-sop-deploy` skill](agentic/vss-sop-skills/vss-sop-deploy/SKILL.md) to perform preflight checks, verify models, download sample assets, and launch all containerized microservices.
4. **Test VSS SOP app**: Use the [`vss-sop-test` skill](agentic/vss-sop-skills/vss-sop-test/SKILL.md) to execute the post-deployment validation suite and verify end-to-end functionality.

## Usage
Please refer to:
- Training: [microservices/sop-training-bp/README.md](microservices/sop-training-bp/README.md)
- Inference: [microservices/sop-inference-bp/README.md](microservices/sop-inference-bp/README.md)
- Agentic Skills: [agentic/ds-sop-skills/README.md](agentic/ds-sop-skills/README.md)
- VSS SOP Skills: [agentic/vss-sop-skills/README.md](agentic/vss-sop-skills/README.md)

## Sample Data
We provide [sample data](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/sop-server-fan-installation-data?version=1.0-260213) which can be used for testing the SOP Training and Inference BP.
The sample data is about installing server fan and power.

## License
This project is dual-licensed: source code under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) and documentation under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/), per the `CC-BY-4.0 AND Apache-2.0` terms in the top-level [`LICENSE`](./LICENSE).

This project bundles and/or downloads third-party open-source software, each under its own license. See [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md) for the third-party components distributed in this repository, and review the license terms of any additionally downloaded open-source projects before use.
