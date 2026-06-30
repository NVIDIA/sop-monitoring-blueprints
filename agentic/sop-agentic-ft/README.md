# sop-agentic-ft

Plugin bundle for the **agentic fine-tuning flow** of the SOP Training Service.

This folder is a self-contained plugin marketplace (`.claude-plugin/marketplace.json` + `plugins/`).
You don't browse it directly — you install it into your agent and drive the flow from there.

👉 **See [`AGENTIC_README.md`](../../microservices/sop-training-bp/AGENTIC_README.md) for install instructions and usage.**

## What's here

| Plugin | Skills it provides |
|---|---|
| `plugins/sop-data-augmentation-plugin/` | `/sop-data-augmentation` |
| `plugins/sop-ddm-finetuning-plugin/` | `/sop-ddm-finetuning` |
| `plugins/sop-cr-finetuning-plugin/` | `/sop-cr-finetuning` |
| `plugins/sop-evaluation-plugin/` | `/sop-by-action-eval`, `/sop-e2e-inference` |
| `plugins/sop-rca-plugin/` | `/sop-rca` |
| `plugins/sop-ft-orchestrate-plugin/` | `/sop-ft-orchestrate` (+ a bookkeeping-enforcement hook) |

## License
This project is dual-licensed under the `CC-BY-4.0 AND Apache-2.0` terms in the [`LICENSE`](./LICENSE) file: code-only files (e.g. `scripts/`, `helpers/`) are licensed under Apache-2.0, and mixed documentation files (e.g. `SKILL.md`, `references/`) under the dual CC-BY-4.0 AND Apache-2.0 terms.
