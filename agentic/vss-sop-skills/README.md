# VSS SOP Integration and Claude Code Setup Guide

Welcome to the **VSS SOP (Video Search and Summarization — Standard Operating Procedure)** blueprint repository. This guide provides step-by-step instructions to set up **Claude Code**, run the **VSS SOP skills**, understand the purpose and function of each skill type, and ensure development compliance with NVIDIA's software standards.

---

## 🚀 Part 1: Installing & Configuring Claude Code

Claude Code is a fast, terminal-based coding assistant. Follow these commands to upgrade your system package indexes, install Claude Code, and configure your shell environment so that the `claude` command is universally accessible in your terminal.

### 1. Upgrade System and Install Claude Code
Run the following combined command to ensure your host packages are up to date and then fetch and execute the official installer script for Claude Code:

```bash
sudo apt upgrade -y; curl -fsSL https://claude.ai/install.sh | bash
```

### 2. Add Claude to your PATH
By default, the installer places the `claude` binary in `$HOME/.local/bin`. To ensure your shell can locate this binary, append it to your system PATH in your user's `.profile` file:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.profile
```

### 3. Apply the Environment Changes
Source the updated `.profile` file to load the new PATH into your active terminal session immediately:

```bash
source ~/.profile
```

Once completed, you can run the `claude` command in your terminal to start using the assistant!

---

## 🧩 Part 1b: Installing the Claude Plugins & Running the Pipeline

The VSS SOP system consists of two Claude Code plugins that must be installed in order. **You must install the `ds-sop-skills` plugin first**, as the end-to-end orchestration skill (`sop-build`) depends on it to generate the DeepStream SOP microservice source code.

---

### Step 1: Install the DeepStream SOP Skill Plugin (`ds-sop-skills`) — INSTALL FIRST

The **`ds-sop-skills`** plugin provides the capabilities to generate, evaluate, and troubleshoot the DeepStream SOP Inference Microservice.

To install it, simply copy the `deepstream-sop` skill folder directly into your project's `.claude/skills/` directory. This uses the existing, pre-configured `plugin.json` manifest located at `deepstream-sop/.claude-plugin/plugin.json`:

```bash
mkdir -p .claude/skills
cp -r /path/to/sop-monitoring-blueprints/agentic/ds-sop-skills/deepstream-sop .claude/skills/
```

---

### Step 2: Install the VSS SOP life-cycle Skills Plugin (`vss-sop-skills`) — INSTALL NEXT

The **`vss-sop-skills`** plugin provides the four life-cycle skills (`sop-build`, `vss-sop-build`, `vss-sop-deploy`, `vss-sop-test`) to manage the complete blueprint life-cycle.

#### Option A — Plugin install via Marketplace (recommended)
Register the local `vss-sop-skills` folder as a marketplace, then install the `vss-sop` plugin from it:

```bash
# 1. Register the marketplace (local clone path, or a git URL / GitLab shorthand)
/plugin marketplace add /path/to/sop-monitoring-blueprints/agentic/vss-sop-skills

# 2. Install the vss-sop plugin from the "vss-sop-skills" marketplace
/plugin install vss-sop@vss-sop-skills
```

> After install, the skills are available under the `vss-sop` namespace — e.g. `/vss-sop:sop-build`, `/vss-sop:vss-sop-build`, `/vss-sop:vss-sop-deploy`, `/vss-sop:vss-sop-test`.

To update to the latest version later, run `/plugin marketplace update vss-sop-skills`.

#### Option B — Manual copy
If the plugin system is unavailable or you need an offline install, copy the skill folders directly into your project's `.claude/skills/`:

```bash
mkdir -p .claude/skills
cp -r /path/to/sop-monitoring-blueprints/agentic/vss-sop-skills/{sop-build,vss-sop-build,vss-sop-deploy,vss-sop-test} .claude/skills/
```

---

### Step 3: How to Run the `sop-build` Skill (End-to-End Orchestrator)

Once both plugins are installed, you can run the **`sop-build`** skill to orchestrate the entire end-to-end pipeline from scratch.

#### Option A: Run from your terminal CLI
Pass your request to Claude Code as an argument:
```bash
claude "Follow the sop-build skill to set up the complete VSS SOP pipeline."
```

#### Option B: Run within the Claude Code interactive session
1. Start Claude Code in your terminal:
   ```bash
   claude
   ```
2. Once the interactive session starts, describe the task to load the `sop-build` orchestrator:
   ```text
   Run the full SOP pipeline using the sop-build skill.
   ```

---

## 🔑 Part 2: Setting Up the `.secret` Folder (API Keys)

Before running any SOP skill, you must create a **`.secret/`** directory in the **active working folder where you execute Claude Code** (this is the root of your blueprint repository / workspace). The deploy and test skills read the key files directly from this folder in your active working directory, and it is **git-ignored** so your secrets never get committed.

The folder holds up to two key files:

| File | Required? | Purpose |
|---|---|---|
| `.secret/ngc_api_key.txt` | **Yes** | NGC API key — used to log into `nvcr.io`, pull NIM images, and download the SOP sample dataset/assets from NGC. (Trained models are not downloaded — they must be retrained via the SOP Training Blueprint and placed under `/opt/models` and `/opt/sop`.) |
| `.secret/nvidia_build_api_key.txt` | Optional | NVIDIA Build API key (`build.nvidia.com`) — required only when using **remote** LLM/VLM NIM endpoints. |

### 1. Create the `.secret` directory
Navigate to the working directory where you run Claude Code (the repository root) and create the folder with locked-down permissions so only your user can read it:

```bash
mkdir -p .secret
chmod 700 .secret
```

### 2. Obtain & store your NGC API key (required)
1. Go to [https://ngc.nvidia.com](https://ngc.nvidia.com) and sign in.
2. Top-right → **Setup** → **API Keys** → **Generate Personal Key**.
3. Set permissions to include **NGC Catalog**, then copy the key immediately — it is shown only once.
4. Save it into the secret file (replace `<your_ngc_key>`):

```bash
printf '%s' '<your_ngc_key>' > .secret/ngc_api_key.txt
chmod 600 .secret/ngc_api_key.txt
```

> The SOP skills configure the NGC CLI for no organization (`no-org`) and no team (`no-team`).

### 3. (Optional) Store your NVIDIA Build API key — for remote NIM endpoints
If you plan to use remote LLM/VLM endpoints hosted on [build.nvidia.com](https://build.nvidia.com) (instead of local NIMs), generate an API key there and save it:

```bash
printf '%s' '<your_nvidia_build_key>' > .secret/nvidia_build_api_key.txt
chmod 600 .secret/nvidia_build_api_key.txt
```

`deploy.sh` automatically exports this as `NVIDIA_API_KEY` and writes it into `deployments/sop/.env` on each run. If it is missing, calls to remote LLM endpoints will fail with `[401] Unauthorized`.

### 4. Verify the layout
Your `.secret/` folder should now look like this:

```
.secret/
├── ngc_api_key.txt              # required
└── nvidia_build_api_key.txt     # optional (remote NIM only)
```

You can confirm the NGC key is picked up by running the deploy preflight check, which validates the secret files (and with `--fix` will prompt to create a missing NGC key):

```bash
./agentic/vss-sop-skills/vss-sop-deploy/scripts/preflight_check.sh --bp-repo .
```

> ⚠️ **Never** commit these keys or paste them into any tracked file (e.g. `Readme.md`, `TOOLS.md`, or `.env` files under version control). The `.secret/` directory is already listed in `.gitignore`.

---

## 🛠️ Part 3: Meaning, Function, and Description of the SOP Skill Types

The **VSS SOP** system utilizes a modular suite of automated **skills** to manage the blueprint life-cycle. A single top-level orchestrator (`sop-build`) drives the complete pipeline, which is composed of the focused life-cycle skills (`vss-sop-build`, `vss-sop-deploy`, and `vss-sop-test`) plus the `ds-sop-skills` source generator. Below is the detailed breakdown of each skill.

```
                         ┌──────────────────────────────────────────────────────────┐
                         │                       sop-build                          │
                         │             (end-to-end pipeline orchestrator)           │
                         └──────────────────────────────────────────────────────────┘
                                                    │
                         ┌──────────────────────────────────────────────────────────┐
                         │  Phase Pre — Prerequisites Check & Auto-Install (run 1st) │
                         │  vss-sop-deploy/scripts/preflight_check.sh --fix          │
                         └──────────────────────────────────────────────────────────┘
                                                    │
      ┌───────────────┬───────────────┬────────────┴───┬───────────────┬───────────────┐
      ▼               ▼               ▼                ▼               ▼               ▼
┌───────────┐  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌──────────────┐ ┌──────────────┐
│ Verify    │  │ ds-sop-skills │ │ ds-sop-skills │ │ vss-sop-build │ │vss-sop-deploy│ │ vss-sop-test │
│ Models &  │  │  generate     │ │  evaluate     │ │  scaffold &   │ │ build image  │ │  verify      │
│ Assets    │  │  DS-SOP src   │ │  DS-SOP       │ │  patch        │ │  & start     │ │  end-to-end  │
└───────────┘  └───────────────┘ └───────────────┘ └───────────────┘ └──────────────┘ └──────────────┘
   Phase 0         Phase 1          Phase 2          Phase 3          Phase 4          Phase 5
```

### 0. `sop-build` (Full Pipeline Orchestrator)
* **Meaning**: The **End-to-End SOP Pipeline Driver**. This is the single entry point that runs the entire SOP life-cycle from a fresh machine to a fully validated deployment.
* **Function**: Runs a prerequisites check followed by six phases, each of which must succeed before the next begins:
  * **Phase Pre — Prerequisites Check & Auto-Install**: Runs **`vss-sop-deploy`**'s `preflight_check.sh --fix` first to validate (and optionally auto-install) secret key files, GPU drivers, Docker, the NVIDIA Container Toolkit, and NGC configuration before any other phase begins.
  * **Phase 0 — Verify Models & Download Assets**: Verifies trained model and config files exist under `/opt/models` and `/opt/sop` (requiring model retraining via the SOP Training Blueprint for optimal accuracy), downloads the sample video, and prepares cache/datalog directories (delegates to `vss-sop-deploy`'s `download_assets.sh`).
  * **Phase 1 — Generate SOP Microservice**: Generates the DeepStream `ds_sop_microservice` source via the **`ds-sop-skills`** skill, including the required RTSP streaming output feature (ds-sop-skills § 18).
  * **Phase 2 — Evaluate SOP Microservice**: Builds and evaluates the generated microservice (static checks, image build, API/file-video tests) to catch issues before the full blueprint build.
  * **Phase 3 — Build Blueprint**: Runs **`vss-sop-build`** to scaffold and patch the SOP blueprint on top of VSS 3.1 and build the `ds-sop:1.0.0` image.
  * **Phase 4 — Deploy Blueprint**: Runs **`vss-sop-deploy`** to check prerequisites, configure endpoints, and start all microservices.
  * **Phase 5 — Test Blueprint**: Runs **`vss-sop-test`** for the full post-deployment validation suite (must reach a 100% pass rate).
* **Description**: Use this skill when you want to set up the entire SOP pipeline from scratch on a new machine, or re-run the complete build → evaluate → deploy → test cycle after upstream/source changes. It enforces strict pass criteria — every phase must pass with zero failures before completion.
* **How to Run**: Ask Claude Code to run the full pipeline, e.g.:
  ```text
  Follow the sop-build skill to set up the complete VSS SOP pipeline from scratch.
  ```

---

### 1. `vss-sop-build`
* **Meaning**: The **Blueprint Scaffolder and Code Customizer**. This skill transforms the generic NVIDIA VSS 3.1 base reference code into a tailored VSS SOP-compliant structure.
* **Function**:
  * Clones the upstream VSS 3.1.0 `video-search-and-summarization` repository (read-only source) and builds the top-level `compose.yml`, stripping unused upstream services (`vlm-as-verifier`, `lvs`, `developer-workflow`, `proxy`) and adding the `./sop/compose.yml` include with the custom `bp_sop_2d` profile.
  * Copies all reference files verbatim (tuned `.env` files, compose includes, templates, and the entire new `sop/` app layer).
  * Copies and modifies **foundational** services for SOP (stock Elasticsearch image, SOP Kafka topics, `mdx-vlm-captions` ES templates/init scripts, `bp_sop_2d` profiles).
  * Copies and modifies **VIOS/VST** for SOP — splitting the upstream monolith into 5 microservices and adding MinIO.
  * Copies **agents** from upstream and applies the SOP profile, bind-mounting the Python patches (`es_client.py`, `tools.py`, `utils.py`) into `vss-va-mcp`/`vss-agent` for SOP compliance audits.
  * Copies **NIM** model services with per-GPU profiles and SOP env conventions.
  * Builds the local `ds-sop:1.0.0` image from the `ds-sop-skills`-generated `ds_sop_microservice` source.
* **Description**: Prepares the source tree, YAML profiles, and configurations in preparation for deployment. It ensures correct directories and file structures coexist before the runtime container build begins.
* **How to Run**: Ask Claude Code to run the build process, e.g.:
  ```text
  Follow the vss-sop-build skill to scaffold and patch the SOP blueprint.
  ```

---

### 2. `vss-sop-deploy`
* **Meaning**: The **Deployment Orchestrator and Asset Provisioner**. This skill builds the core DeepStream-based `ds-sop` Docker container image, verifies locally installed trained models/configs, downloads the SOP sample dataset/assets, registers configurations, and starts the containerized platform.
* **Function**:
  * **Phase 0 — Prerequisites Check & Auto-Install**: Validates secret key files, GPU capabilities, driver compatibility (Driver 580, CUDA 13), Docker runtime, NVIDIA Container Toolkit (1.18.1), and NGC configuration — and with `--fix` can automatically install/configure any missing components.
  * **Phase 1 — Verify Models & Download Assets**: Checks that required trained model and config files exist under `/opt/models` and `/opt/sop` (requiring model retraining via the SOP Training Blueprint for optimal accuracy), fetches the sample stream video, transcodes it to H.264 at 30 FPS using `ffmpeg`, and pre-creates bind-mounted data log directories with universal permissions.
  * **Phase 2 — Start RTSP Server Simulation**: Starts the RTSP server simulator (ahead of build/deploy) to serve loopback sample footage over `rtsp://localhost:8552/sensor_0` for real-time video perception ingestion in later phases.
  * **Phase 3 — Build DS-SOP Image & Verify Components**: Builds the `ds-sop:1.0.0` container from the `ds-sop-skills`-generated source using BuildKit (no patch step — the generated source already includes RTSP output), then runs the 6 pre-flight RTSP component checks via `verify_rtsp_components.py`.
  * **Phase 4 — Configure & Deploy**: Auto-detects the host's external/internal IPs, configures `deployments/sop/.env` with remote/local LLM and VLM endpoints (customizable, enforcing `use_base64: true` and `SOP_MESSAGING_SCHEMA=JSON`), logs into NGC, clears stale Elasticsearch/Kafka data, and starts all microservices under the `bp_sop_2d` profile.
  * **Phase 5 — Test with RTSP Stream**: Polls `mdx-ds-sop-1` until the API server is up, triggers the test client to register streams to the VIOS VMS, and runs a self-healing integration loop that restarts components if the connection drops (avoiding the stale RTSP factory bug).
* **Description**: Executes the entire infrastructure deployment flow to turn a fresh Linux host into a fully active VSS SOP intelligence pipeline.
* **How to Run**: Ask Claude Code to run the deployment process, e.g.:
  ```text
  Follow the vss-sop-deploy skill to deploy the SOP blueprint.
  ```
  * **Tearing Down**: Ask Claude Code to stop and cleanly erase all containers and datalogs, e.g.:
    ```text
    Run the vss-sop-deploy teardown to stop and clean up all containers and datalogs.
    ```

---

### 3. `vss-sop-test`
* **Meaning**: The **Post-Deployment Quality Assurance and Diagnostic Suite**. This skill comprehensively tests and validates every layer of the deployed blueprint stack, auto-collecting container logs and diagnosing failures.
* **Function**:
  * **Phase 1 — Service Health**: Inspects all required containers (`mdx-kafka`, `mdx-redis`, `mdx-elastic`, `mdx-logstash`, `mdx-kibana`, `vss-agent`, `vss-va-mcp`, `mdx-ds-sop-1`, and all VST microservices: `sensor-ms-sop`, `recorder-ms-1-sop`, `rtspserver-ms-1-sop`, `storage-ms-sop`, `sdr-http-recorder-sop`, `sdr-http-rtspserver-sop`) to verify they are running cleanly.
  * **Phase 2 — ELK Pipeline Validation**: Confirms Elasticsearch cluster connectivity, verifies the creation of `mdx-vlm-captions-*` indexes, checks data ingestion counts, and validates flat JSON field structures on Elasticsearch and Kibana dashboards (ensuring no nested/protobuf-style fields are present, and that `timeFieldName` is `@timestamp`).
  * **Phase 3 — VIOS VMS Recording**: Queries the Video Storage Toolkit (VST) API to guarantee that camera sensors are correctly registered, that recordings are active, and that live streams are functioning.
  * **Phase 4 — End-to-End VSS Agent Auditing**: Directly tests the VSS Agent's communication loop (verifying MCP health, LLM/VLM endpoints, Agent health, invoking a physical snapshot request, and executing actual queries for compliance reports and status logs).
* **Strict Pass Criteria (MANDATORY)**:
  * **100% Pass Rate**: Every single test check across all phases must pass. There is zero tolerance for failures.
  * **No Excuses**: Do not classify failures as "environmental", "infrastructure", "test harness", or "intermittent". Every failure is a bug until fixed.
  * **Re-run After Every Fix**: A fix is not verified until the failing phase is re-run and passes.
* **Service Endpoints Reference**:
  | Service | Endpoint |
  |---|---|
  | Elasticsearch | `http://localhost:9200` |
  | Kibana | `http://<EXTERNAL_IP>:5601` |
  | VST (VIOS) | `http://localhost:30888/vst` |
  | VSS Agent | `http://localhost:8000` |
  | VSS VA MCP | `http://localhost:9901` |
  | VSS UI | `http://<EXTERNAL_IP>:3000` |
  | Phoenix | `http://<EXTERNAL_IP>:6006` |
  | Grafana | `http://<EXTERNAL_IP>:35000` |
* **Common Troubleshooting & Fixes**:
  * **Phase 4.7 "no streams found" on fresh deployment**: If the test runs immediately after deploy, the report time window may precede the first recording. Wait at least 1 minute after the RTSP stream registers before running Phase 4.
  * **Report generation fails with VST clip HTTP 500**: With `overlay_config: true`, VST's OSD/bbox transcode pipeline fails to multiplex short clips. Since SOP incidents carry no object bounding boxes, set `overlay_config: false` for the `vst_video_url` tool in `sop/vss-agent/configs/config.yml` to ensure reliable plain no-overlay clip muxing.
  * **Kibana "No field found for [...]"**: Ensure `SOP_MESSAGING_SCHEMA=JSON` and `ENABLE_MESSAGING=1` are set in `deployments/ds/ds-sop/.env` (automatically enforced by `configure_blueprint.sh`). If needed, re-import the corrected ndjson via:
    ```bash
    curl -X POST localhost:5601/api/saved_objects/_import?overwrite=true -H 'kbn-xsrf: true' --form file=@deployments/sop/sop-app/kibana-dashboard/sop-kibana-objects.ndjson
    ```
* **How to Run**: Ask Claude Code to run the post-deployment tests, e.g.:
  ```text
  Follow the vss-sop-test skill to validate the current deployment.
  ```
  You can also ask Claude to run specific validation phases individually (e.g. to test after applying a fix):
  ```text
  Use the vss-sop-test skill to run Phase 1 (Service health) only.
  ```

---

## 💡 Part 4: Invoking the Skills with Claude Code

Each life-cycle skill is defined under `agentic/vss-sop-skills/<skill-name>/SKILL.md`. Application-level skills are stored in reference directories to be copied into place during blueprint scaffolding. Claude Code automatically discovers the active life-cycle skills and invokes the right one based on your request.

| Skill | Use it to… |
|---|---|
| **`sop-build`** | Run the complete end-to-end pipeline: verify local models/configs and download sample assets → generate & evaluate the DS-SOP microservice → scaffold/patch the blueprint → deploy → validate. |
| **`vss-sop-build`** | Scaffold and patch the SOP blueprint on top of VSS 3.1 and build the `ds-sop:1.0.0` image. Also bundles reference VSS skills. |
| **`vss-sop-deploy`** | Validate prerequisites, verify local models/configs, download sample assets from NGC, build the DS-SOP container, configure parameters, and start all microservices. (Tear down with the deploy skill's `teardown.sh`.) |
| **`vss-sop-test`** | Execute the comprehensive post-deployment diagnostic and validation suite (Phase 1–4) with auto-debugging and container log retrieval. |

### How to Run

You can invoke a skill in two ways:

#### Option A: Run from your terminal CLI
Pass your request to Claude Code as an argument:
```bash
claude "Follow the sop-build skill to set up the complete VSS SOP pipeline."
```

#### Option B: Run within the Claude Code interactive session
1. Start Claude Code in your terminal:
   ```bash
   claude
   ```
2. Once the interactive session starts, describe the task (or name the skill) and Claude will load the matching skill:
   ```text
   Run the full SOP pipeline using the sop-build skill.
   ```
   ```text
   Use vss-sop-test to validate the current deployment.
   ```

---

## 📂 Directory Structure

These skills live under `sop-monitoring-blueprints/agentic/` as a self-contained Claude Code plugin (`vss-sop`). The plugin source tree holds the marketplace/plugin manifests plus the four life-cycle skills and their scripts, references, and evals. When you later run `vss-sop-build` inside a target blueprint working directory, a new `deployments/` folder is scaffolded there to contain the complete, containerized VSS SOP runtime stack.

#### 1. Repository Root Structure (Initial Clone)
This is the layout you get when you clone the `agentic` skills repo. `vss-sop-skills/` is the installable plugin; `ds-sop-skills/` is its sibling DeepStream SOP source generator (referenced by the build skill as `../ds-sop-skills/`).
```
agentic/
├── ds-sop-skills/                      # Sibling plugin: DeepStream SOP microservice source generator
└── vss-sop-skills/                     # This plugin — VSS SOP blueprint life-cycle skills
    ├── .claude-plugin/                 # Claude Code plugin metadata
    │   ├── plugin.json                 # Plugin "vss-sop" (declares the four skills below)
    │   └── marketplace.json            # Marketplace "vss-sop-skills" (source: ./)
    ├── README.md                       # This integration and setup guide
    ├── sop-build/                      # Full pipeline orchestrator (assets → generate → eval → build → deploy → test)
    │   ├── SKILL.md
    │   └── evals/                      # Skill evaluation cases (evals.json)
    ├── vss-sop-build/                  # Build the custom SOP blueprint from VSS 3.1
    │   ├── SKILL.md
    │   ├── scripts/                    # Clone, copy, modify, patch, build, and verify scripts (+ lib/)
    │   ├── references/                 # Golden reference assets and per-area build guides
    │   │   ├── configs/                # foundational/, vios/, vss-agent/ reference configs
    │   │   ├── deployments/            # Reference deployed-tree (compose.yml, ds/, sop/, vst/)
    │   │   ├── diagrams/               # Architecture & build-flow diagrams (PNG)
    │   │   ├── scripts/                # Per-service reference scripts (ds-sop, foundational, nim, vios, vss-agent)
    │   │   ├── vss/                    # Reference VSS application-level skills:
    │   │   │   ├── vss-query-analytics/       # Analytics querying skill (ex-video-analytics)
    │   │   │   ├── vss-search-archive/        # Visual search skill (ex-video-search)
    │   │   │   ├── vss-call-vios-api/         # VMS/VIOS management skill (ex-sensor-ops)
    │   │   │   ├── vss-summarize-video/       # Video summarization skill (ex-video-summarization)
    │   │   │   └── vss-generate-video-report/ # Compliance report generation skill (ex-incident-report)
    │   │   └── *-building.md           # ds-sop / foundational / nim / sop-app / vios / vss-agent build guides
    │   └── evals/
    ├── vss-sop-deploy/                 # Preflight check, NGC downloads, image builder & deployer
    │   ├── SKILL.md
    │   ├── scripts/                    # preflight_check, download_assets, start_rtsp_server, configure,
    │   │                               #   deploy, test_rtsp, teardown, verify_rtsp_components (+ lib/)
    │   ├── references/                 # build_ds_sop_image, ngc, nvidia_driver, prerequisites guides
    │   └── evals/
    └── vss-sop-test/                   # Service health and end-to-end integration tests
        ├── SKILL.md
        ├── scripts/                    # vss_sop_test.py (Phase 1–4 validation suite)
        └── evals/
```

#### 2. Deployed Blueprint Structure (Post-Build)
After running `vss-sop-build` in your target blueprint working directory, a `deployments/` folder is scaffolded there, bringing together the customized microservices and the `sop/` application layer (mirroring `vss-sop-build/references/deployments/`, plus `foundational/`, `agents/`, `nim/`, and `monitoring/` copied from upstream during the build). Crucially, all models, configuration resources, and persistent datalogs reside directly inside your current working directory (CWD):
```
<blueprint-working-dir>/
├── deployments/                    # Populated during the build phase (Stage 0–3)
│   ├── compose.yml                 # Main, top-level Unified Compose entrypoint
│   ├── cleanup_all_datalog.sh      # Utility to clear persistent video storage and DB logs
│   ├── foundational/               # Core services (Kafka, Elasticsearch, Redis, Kibana)
│   ├── vst/                        # Video Storage Toolkit microservices split for storage and recording
│   ├── agents/                     # AI Agents (VSS Agent, Video Analytics MCP Server)
│   ├── nim/                        # Profiles and configurations for NVIDIA NIM model endpoints
│   ├── monitoring/                 # Monitoring stack (Prometheus, Grafana, node-exporter)
│   ├── ds/                         # DS Pipeline configs
│   │   └── ds-sop/                 # Local DeepStream action detector microservice configuration
│   └── sop/                        # Custom SOP Application Layer
│       ├── compose.yml             # Includes app-level services
│       ├── .env                    # Deployment environment variables (Host IP, model settings, keys)
│       ├── vss-agent/              # Customized Agent configurations, overlays, and templates
│       │   ├── configs/            # config.yml, va_mcp_server_config.yml
│       │   ├── patches/            # es_client.py, tools.py, utils.py
│       │   └── templates/          # sop_compliance_report_template.md
│       └── sop-app/                # Video analytics API and Kibana dashboard setup
│           ├── Dockerfiles/        # Service-specific build steps (e.g. kibana-dashboard.Dockerfile)
│           ├── helper-scripts/     # Stream simulation tools and stream-to-VMS scripts
│           ├── kibana-dashboard/   # Dashboard objects configuration (.ndjson)
│           └── vss-video-analytics-api/
├── sop-resources/                  # Resources downloaded in CWD (sample video, configs)
└── sop-data/                       # Datalog folders created in CWD (ES data, logs, Kafka, etc.)
```

---

## License

Use of this skill is governed by the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en) and the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).