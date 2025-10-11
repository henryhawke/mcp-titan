# Titan Memory MCP Server

Stdio-first Model Context Protocol (MCP) server that pairs a TensorFlow.js-backed memory model with optional online learning and workflow orchestration services. This README reflects the project state as of October 10, 2025 and aligns with the current `src/index.ts` implementation.

## Requirements
- Node.js **22.0.0+** (enforced by `package.json` `engines` field)
- npm (bundled with Node) or pnpm/yarn if you prefer to adapt the scripts
- Python build chain / node-gyp prerequisites suitable for installing `@tensorflow/tfjs-node`
- Optional: network access for dataset downloads invoked by the training scripts in `scripts/`

## Quick Start
```bash
git clone https://github.com/henryhawke/mcp-titan.git
cd mcp-titan
npm install
npm run build
npm start   # launches the MCP server over stdio
```

The runtime creates `~/.titan_memory/` (override with `--memoryPath` when instantiating `TitanMemoryServer`) and auto-saves the active memory state every 60 seconds.

### CLI Entrypoint
The published package exposes a `titan-memory` binary (see `bin` in `package.json`). Use it directly or via `npx`:

```bash
npx titan-memory
```

## MCP Client Integration
The server only speaks MCP over stdio. Point your client at the executable instead of an HTTP URL.

### Cursor Example (`~/.cursor/settings.json`)
```json
{
  "mcp": {
    "servers": {
      "titan-memory": {
        "command": "titan-memory",
        "env": {
          "NODE_ENV": "production"
        },
        "workingDirectory": "~/.titan_memory"
      }
    }
  }
}
```

### Claude Desktop
Add a custom MCP server, choose “Run a binary on this machine,” and point it at the same `titan-memory` command. No port configuration is required.

## LLM System Prompt
Use the prompt below (or the version tracked in `docs/llm-system-prompt.md`) when wiring Titan into Cursor, Claude, or other MCP-compatible agents.

```markdown
You are connected to the @henryhawke/mcp-titan MCP server. Use the tools exactly as documented in docs/api/README.md. For a comprehensive overview of the system architecture, see docs/architecture-overview.md.

- Always call `init_model` before other memory operations unless the model was auto-loaded.
- Use `help` to retrieve tool schemas.
- Persist state with `save_checkpoint` / `load_checkpoint` as needed.
- Reach for `prune_memory` when capacity exceeds 70% utilization.
- Control the online learner with `init_learner`, `pause_learner`, `resume_learner`, and `get_learner_stats`.
- Treat tool responses as authoritative; handle any returned error text gracefully.
```

## Feature Highlights
- **Stdio MCP Server:** Implemented with `McpServer` + `StdioServerTransport`; no HTTP layer is active in this release.
- **Auto-Initialization:** Loads existing checkpoints from `~/.titan_memory` or bootstraps a default config (`inputDim=768`, `memorySlots=5000`, `transformerLayers=6`).
- **Memory Management:** Vector processing, TF-IDF bootstrap helper, gradient reset, and information-gain pruning when supported by the model.
- **Online Learner Loop:** `LearnerService` exposes MCP hooks for replay-buffer based updates. A mock tokenizer is installed by default—swap in `AdvancedTokenizer` for real encodings.
- **Workflow Skeleton:** `src/workflows/` contains orchestrators for release automation, linting, and feedback analysis. These rely on `WorkflowConfig` feature flags defined in `src/types.ts`.
- **Training Pipeline:** `scripts/train-model.ts` pipes into `src/training/trainer.ts`, providing synthetic data generation and tokenizer training when no dataset is supplied.

## Tooling Overview
The server currently registers **17** MCP tools:

`help`, `bootstrap_memory`, `init_model`, `memory_stats`, `forward_pass`, `train_step`, `get_memory_state`, `get_token_flow_metrics`, `reset_gradients`, `prune_memory`, `save_checkpoint`, `load_checkpoint`, `init_learner`, `pause_learner`, `resume_learner`, `get_learner_stats`, `add_training_sample`.

Use `help` at runtime to confirm the latest list (the string output is being updated to match this registry). See [docs/api/README.md](docs/api/README.md) for parameter schemas and defaults. The `manifold_step` operation remains on the roadmap—track progress in [ROADMAP_ANALYSIS.md](ROADMAP_ANALYSIS.md).

## Persistence & Files
- Memory state is serialized to `~/.titan_memory/memory_state.json`.
- Saved models live under `~/.titan_memory/model/`.
- MCP checkpoints written via `save_checkpoint` are user-specified JSON files that include tensor shapes for validation on load.
- Auto-save interval: 60 seconds with exponential back-off retry on transient errors.

## Development Scripts
- `npm run test` uses Jest (make sure to enable experimental modules for Node 22).
- `npm run train-model`, `npm run train-quick`, `npm run train-production` call into the training pipeline described above.
- `npm run download-data` fetches example corpora (WikiText-2, TinyStories, OpenWebText sample) and supports synthetic data generation when offline.

## Known Gaps & Next Steps
- Help text still references `manifold_step`; update server messaging once the roadmap decision (implement vs remove) finalizes.
- `manifold_step` and several advanced memory operations are referenced in strings but lack MCP tool bindings.
- Additional roadmap tools (`encode_text`, `get_surprise_metrics`, `analyze_memory`, `predict_next`) have schemas drafted but no handlers yet—track status in SYSTEM_AUDIT.md.
- `src/tests/` folder is absent; existing automated coverage resides under `test/` and should be expanded/migrated.
- Tensor summarization in `bootstrap_memory` uses heuristics; integrate an actual summarizer or reuse the tokenizer pipeline for higher fidelity.
- Review workflow managers (`src/workflows/`) for real credential handling before production use.

## Further Reading
- [docs/api/README.md](docs/api/README.md) — complete tool and parameter reference.
- [docs/architecture-overview.md](docs/architecture-overview.md) — component relationships and data flow.
- [SYSTEM_AUDIT.md](SYSTEM_AUDIT.md) — consolidated audit and status tracker.
- [IMPLEMENTATION_PACKAGE.md](IMPLEMENTATION_PACKAGE.md) — navigation guide for research and production tasks.
- [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) — current delivery scope and outstanding tasks.
- [ROADMAP_ANALYSIS.md](ROADMAP_ANALYSIS.md) — strategic outlook and feature roadmap.
