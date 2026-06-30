# Agentic Fine-Tuning for SOP Monitoring

This guide teaches you how to drive the SOP Training Service with an agentic flow instead of running each microservice by hand. The repo ships a set of **skills** — portable, plain-text instructions an agent reads to operate this BP — packaged as installable **plugins** under [`sop-agentic-ft/`](../../agentic/sop-agentic-ft/). You install them into your agent, then hand the work off in natural language.

> Audience: you already have the Training BP running locally (`docker compose up`) and want to skip the manual REST calls to each microservice. If you're new to the BP itself, start with [README.md](README.md).

---

## What is a "skill"?

A skill is a folder containing a `SKILL.md` file. The file has YAML frontmatter (`name`, `description`, and an optional `argument-hint`) followed by Markdown instructions the agent loads on demand. That's it — no compiled code, no plugin runtime for the instructions themselves.

The seven skills fall into three groups:

- **Single-stage skills** — one skill per BP microservice, each invokable on its own
- **Diagnostic skill** — reads run artifacts and recommends config changes
- **Orchestrator** — composes all of the above into a closed-loop fine-tuning run

You can use them independently (`/sop-data-augmentation` to regenerate QA pairs, `/sop-cr-finetuning` to launch a VLM run, etc.) or hand the whole loop off to `/sop-ft-orchestrate`.

### Why they ship as plugins

The skills are distributed as a small **plugin marketplace** under [`sop-agentic-ft/`](../../agentic/sop-agentic-ft/) rather than as loose `SKILL.md` folders. Two reasons:

1. **The orchestrator plugin ships a hook** (`sop-ft-orchestrate-plugin/hooks/hooks.json`) that enforces per-iteration bookkeeping — it physically blocks the orchestrator from advancing a phase or ending a run while the previous iteration's artifacts (eval rows, progress files, RCA reports) are incomplete. Hooks are only loaded when a skill is installed *as a plugin*; copying a loose `SKILL.md` into the skills path silently drops this enforcement.
2. A plugin keeps each skill's helper scripts versioned together with the skill, so they can't drift apart.

---

## Skills in this repo

| Skill | Purpose | BP service it talks to | Provided by plugin |
|---|---|---|---|
| `/sop-data-augmentation` | Generates BCQ / MCQ / GQA / DMCQ / DSQA / ENQA training samples from annotated video | `data-generation-pipeline` (port 5487) | `sop-data-augmentation-plugin` |
| `/sop-ddm-finetuning` | Fine-tunes DDM-Net temporal boundary detector | `ddm-training-ms` (port 32100) | `sop-ddm-finetuning-plugin` |
| `/sop-cr-finetuning` | Fine-tunes Cosmos-Reason VLM | `cr-training-ms` (port 32080) | `sop-cr-finetuning-plugin` |
| `/sop-by-action-eval` | VLM accuracy on perfectly-segmented clips | `evaluation-ms` | `sop-evaluation-plugin` |
| `/sop-e2e-inference` | Full DDM + VLM pipeline accuracy | `evaluation-ms` | `sop-evaluation-plugin` |
| `/sop-rca` | Reads eval logs + configs, classifies failure mode, proposes config fixes | (no service — reads run artifacts) | `sop-rca-plugin` |
| `/sop-ft-orchestrate` | Drives the full Augment → DDM → VLM → Eval → RCA loop autonomously | (composes the six skills above) | `sop-ft-orchestrate-plugin` |

The browsable sources live under [`sop-agentic-ft/plugins/`](../../agentic/sop-agentic-ft/plugins/) — each plugin contains the full `SKILL.md` plus any references/scripts it ships with.

Skill ↔ BP mapping at a glance:

```
                   ┌────────────────────────────────────────────┐
                   │              Agent (Claude / …)            │
                   └─────────────────────┬──────────────────────┘
                                         │ invokes
   ┌──────────────────┬──────────────────┼──────────────────┬─────────────────┐
   ▼                  ▼                  ▼                  ▼                 ▼
 sop-data-       sop-ddm-           sop-cr-            sop-by-action-eval sop-rca
 augmentation    finetuning         finetuning         sop-e2e-inference  (artifacts)
   │                  │                  │                  │                 │
   ▼                  ▼                  ▼                  ▼                 │
 data-           ddm-training-ms    cr-training-ms      evaluation-ms         │
 generation-     (32100)            (32080)                                   │
 pipeline                                                                     │
 (5487)                                                                       │
                                         ▲                                    │
                                         └────────── /sop-ft-orchestrate ─────┘
                                                     composes everything
```

---

## Installing the plugins

The [`sop-agentic-ft/`](../../agentic/sop-agentic-ft/) folder is a self-contained plugin marketplace (named `sop-agentic-ft`). Installing is two steps: register the marketplace, then install the plugins from it.

These instructions use **Claude Code** as an example. (`sop-agentic-ft/` is a standard plugin marketplace, so other plugin-compatible agents can consume the same bundle.)

```
# from the repo root, in your agent session:

# 1. register the local marketplace
/plugin marketplace add ./agentic/sop-agentic-ft

# 2. install the orchestrator + every stage plugin it delegates to
/plugin install sop-ft-orchestrate-plugin@sop-agentic-ft
/plugin install sop-data-augmentation-plugin@sop-agentic-ft
/plugin install sop-ddm-finetuning-plugin@sop-agentic-ft
/plugin install sop-cr-finetuning-plugin@sop-agentic-ft
/plugin install sop-evaluation-plugin@sop-agentic-ft
/plugin install sop-rca-plugin@sop-agentic-ft
```

> **Install all six.** `/sop-ft-orchestrate` delegates each phase to the stage skills via the Skill tool, so their plugins must be present — installing the orchestrator alone is not enough. If you only ever drive a single stage by hand, you can install just that one plugin.

### Verifying installation

Open the agent from the repo root and ask:

```
List the SOP fine-tuning skills you have access to.
```

You should see all seven by name. If a skill is missing, re-check that its plugin installed cleanly (`/plugin` to list installed plugins) and that you ran `/plugin marketplace add ./agentic/sop-agentic-ft` from the repo root.

### Plugin trust & provenance (read before installing)

These plugins are **executable code that runs in your environment**, not passive config. The skills drive shell/`docker` commands against your deployment, and the orchestrator plugin ships a **hook** (`sop-ft-orchestrate-plugin/hooks/hooks.json`) that the agent harness runs **automatically** at certain points (e.g. before tool calls and on stop). Installing a plugin is therefore equivalent to running trusted code — treat it like any dependency you `pip install` or `git clone`.

There is **no built-in signing or integrity verification** of plugin code: neither this marketplace nor the agent harness cryptographically checks that the skills, hooks, and scripts are authentic and unmodified before loading them. A tampered or untrusted copy could execute arbitrary commands in your environment via the agent. **You are responsible for plugin provenance and for the privileges the agent runs with.** Specifically:

- **Install only from the official NVIDIA source.** Use the `sop-agentic-ft/` marketplace from the repository you cloned from NVIDIA; do not install modified forks or third-party copies you don't trust.
- **Verify integrity before installing.** Confirm the repo is at a known-good commit/tag and the working tree is unmodified (e.g. `git status`, `git verify-commit`/tag where signed) before `/plugin marketplace add`.
- **Restrict write access** to the installed plugin files so a local process can't tamper with skills/hooks between sessions.
- **Run the agent with least privilege.** It can execute shell and `docker` commands and reach the BP services — run it as a non-privileged user, scoped to this deployment, ideally in an isolated/sandboxed environment, not on a shared or production host.
- **Review hooks/scripts** (`hooks/hooks.json`, `skills/*/scripts/`) if you have any doubt about a copy's origin — the hook commands and scripts are plain text.

The bundled iteration gate (`sop_iter_gate.py`) enforces run *bookkeeping*, not provenance — it does not verify plugin authenticity.

---

## Using a single skill at a time

Each single-stage skill wraps a microservice call — pick one when you only need that stage. Examples:

**Regenerate augmented data with a new config:**

```
/sop-data-augmentation regenerate the augmented dataset for dataset_id=server_fan_agentic_train
with BCQ negative_ratio=1.
```

**Train DDM-Net on an already-augmented dataset:**

```
/sop-ddm-finetuning launch a DDM-Net run on dataset_id=server_fan_agentic_train and val_dataset_id=server_fan_agentic_test
with bilinear interpolation.
```

**Fine-tune the VLM with a lower learning rate:**

```
/sop-cr-finetuning train Cosmos-Reason2-2B on qa_augmented_dataset_id=server_fan_agentic_train_augmented_0
with optm_lr=[5e-6, 5e-6, 5e-6] and report when the job finishes.
```

**Run an evaluation against a trained checkpoint:**

```
/sop-e2e-inference evaluate <vlm_checkpoint_path> + <ddm_checkpoint_path>
on the test data server_fan_agentic_test.
```

---

## The agentic fine-tuning flow

Fine-tuning is rarely "augment once, train once, done." A typical run requires retries with adjusted configs: the LR was too high, the confusion map missed a key pair, DDM under-segmented, etc. The `sop-ft-orchestrate` skill encodes the closed-loop version of that workflow.

### What it does

Given a training dataset, an eval split, and a target metric, the orchestrator:

1. **Imports** the dataset into the BP database (once per run)
2. **Augments** the data via `sop-data-augmentation`
3. **Trains DDM** via `sop-ddm-finetuning`
4. **Trains the VLM** via `sop-cr-finetuning`
5. **Evaluates** with both `sop-by-action-eval` and `sop-e2e-inference`
6. **Diagnoses** with `sop-rca` and reads its recommended config change
7. **Routes** to the right retry step based on the diagnosis:
   - augmentation issue → back to step 2 (then retrain VLM)
   - VLM issue → back to step 4 (retrain VLM only)
   - DDM issue → back to step 3 (retrain DDM only)
   - eval-only fix → re-run step 5
8. **Stops** when the target metric is met *or* the iteration budget is exhausted

Two practical invariants worth knowing:

- **DDM and VLM are independent retry axes.** Tuning augmentation never re-runs DDM training; a DDM retrain never re-runs augmentation or VLM training.

### Required inputs

| Input | Required? | Notes |
|---|---|---|
| Training dataset path | yes | Path under `assets/data/<your_dataset>/` |
| Eval split path | recommended | Defaults to a `*_test` sibling of the training dir |
| Target KPI | yes | e.g. `seq_accuracy >= 0.90` |
| Max iterations | yes | Hard upper bound on retry loops |
| Status report cadence | optional | e.g. `every 10 minutes` |

### Success criteria

The orchestrator declares **SUCCESS** when every enabled criterion is met simultaneously:

| Metric | What it measures | Sample target |
|---|---|---|
| `e2e_sequence_accuracy` | Test videos with zero prediction errors (the primary goal) | 0.75 – 1.00 |
| `e2e_action_accuracy` | DDM-segmented chunks classified correctly | ≥ 0.90 |
| `by_action_accuracy` | VLM accuracy on perfect clips | ≥ 0.90 |
| `ddm_f1` | DDM boundary detection F1 | ≥ 0.90 |

`e2e_sequence_accuracy` is usually the only target worth pinning down.

---

## Prompt examples

The orchestrator takes natural language — no YAML required for common cases. These three patterns cover most needs.

### A.: Single dataset, Single target

```
I want to fine-tune SOP monitoring on <training_data> and evaluate on <testing_data>
targeting seq_accuracy >= <KPI> and max iterations set to <X>.
Report status every 10 minutes.
```

### B. Conservative: hold extra training data in reserve

Use when you have additional training sets but want to hit the target with as little data as possible — the orchestrator only pulls in the extras after RCA-driven config changes are exhausted.

```
I want to fine-tune SOP monitoring on <training_data>
(there are two additional training sets <training_data_2> and <training_data_3>)
and evaluate on <testing_data> targeting seq_accuracy >= <KPI>
and max iterations set to <X>.
We want to reach the KPI with as little training data as possible —
start with <training_data> and iterate following RCA suggestions.
Only pull in the extra training sets if you have nothing else to try.
Report status every 10 minutes.
```

### C. Strict: no early stop on partial wins

Use to prevent the orchestrator from declaring partial success on an intermediate plateau (e.g. by-action looks promising but E2E hasn't caught up yet).

```
I want to fine-tune SOP monitoring on <train_data> and evaluate on <validation_data>
targeting seq_accuracy >= <KPI> and max iterations set to <X>.
Report status every 10 minutes.
Do not stop until criteria are met or max iterations are reached.
```

---

## What the orchestrator produces

Each run writes to a timestamped output directory:

```
<output_dir>/                              # from inputs.yaml; default ./sop_fine_tune
└── run_<YYYYMMDD_HHMMSS>/                 # = <run_dir>; created in Step 2
    ├── run_state.yaml                     # full run state — incl. eval_outputs pointers
    ├── progress.md                        # phase logs
    ├── orchestrator_report.md             # final SUCCESS / PARTIAL summary
    ├── inputs_by_action_iter<N>.yaml      # generated per-iter from template
    ├── inputs_e2e_iter<N>.yaml            # generated per-iter from template
    ├── overrides/                         # code overrides per Override Policy
    ├── watchdog_*.log                     # DDM/VLM training watchdog tails
    └── iter<N>/
        ├── augment_config.yaml            # snapshot of augment config used this iter
        ├── ddm_train_config.yaml          # snapshot of DDM training config used
        ├── train_config.toml              # snapshot of VLM training config used
        ├── training.log                   # copy of VLM training log
        ├── rca_analysis/                  # /sop-rca helper JSONs
        └── rca_report.md                  # /sop-rca formal report
```

If the orchestrator declares PARTIAL, the report includes the highest metrics seen, the remaining gap, and which axis (DDM, VLM, or augmentation) was most recently moving — useful starting points for a follow-up manual iteration.

---

## Scope & current limitations

The **agentic fine-tuning flow** in this release targets **single-operator SOPs** — one operator performing a sequence of actions per video. **Multi-operator / concurrent-action fine-tuning is not supported through the agentic flow in this release**: the stage skills and `/sop-ft-orchestrate` all assume single-operator data. (The underlying augment endpoint does expose a `two_operator_mode` parameter, but driving multi-operator augmentation is outside the scope of the agentic skills documented here.)

---

## Tips for first-time runs

- Bring up the BP first (`docker compose up`) and confirm the training and augmentation services respond at their ports — the agent will surface a clear error if not, but it saves an iteration.


---

## Related

- [README.md](README.md) — Training BP overview, microservice details, manual workflow
- [`sop-agentic-ft/`](../../agentic/sop-agentic-ft/) — the plugin marketplace: every plugin, with full `SKILL.md` and any references/scripts it ships with
- [`tutorials/sop_monitoring_agentic_ft.ipynb`](tutorials/sop_monitoring_agentic_ft.ipynb) — runnable data-prep walkthrough for the public Server Fan sample, ending in a ready-to-paste orchestrator prompt
- [`tutorials/sop_monitoring_training_sample_data.ipynb`](tutorials/sop_monitoring_training_sample_data.ipynb), [`tutorials/sop_monitoring_training_flow.ipynb`](tutorials/sop_monitoring_training_flow.ipynb) — non-agentic notebook walkthroughs (manual stage-by-stage)
