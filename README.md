# HOPE Memory MCP Server

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

The runtime creates `~/.hope_memory/` (override with `--memoryPath` when instantiating `HopeMemoryServer`) and auto-saves the active memory state every 60 seconds.

### CLI Entrypoint
The published package exposes a `titan-memory` binary (see `bin` in `package.json`). Use it directly or via `npx`:

```bash
npx @henryhawke/mcp-titan
```

## MCP Client Integration
The server only speaks MCP over stdio. Point your client at the executable instead of an HTTP URL.

### Cursor Example (`~/.cursor/settings.json`)
```json
{
  "mcp": {
    "servers": {
      "mcp-titan": {
        "command": "npx",
        "args": [
          "-y",
          "@henryhawke/mcp-titan"
        ],
        "workingDirectory": "~/WHEREVER YOU WANT YOUR MEMORY"
      }
    }
  }
}
```

### Claude Desktop
Add a custom MCP server, choose “Run a binary on this machine,” and point it at the same `mcp-titan` command. No port configuration is required.

## LLM System Prompt
Use the prompt below (or the version tracked in `docs/llm-system-prompt.md`) when wiring HOPE into Cursor, Claude, or other MCP-compatible agents.

```markdown
You are operating the @henryhawke/mcp-titan MCP memory server. Follow this checklist on every session:

1. **Discover** — Call `help` to confirm the active tool registry and read parameter hints.
2. **Initialize** — If the model was not auto-loaded, invoke `init_model` with any config overrides. For momentum tuning, adjust:
   - `momentumLearningRate` (base θ),
   - `momentumScoreGain` / `momentumScoreToDecay` (attention-weighted scaling),
   - `momentumSurpriseGain` (surprise-driven boost), and
   - `momentumScoreFloor` (stability floor).
3. **Prime Memory** — Use `bootstrap_memory` or `train_step` pairs before running `forward_pass`. Inspect `memory_stats` / `get_memory_state` to verify momentum and forgetting gates are active.
4. **Maintain** — When utilization exceeds ~70%, run `prune_memory`; persist progress with `save_checkpoint` and restore via `load_checkpoint` after restarts.
5. **Learn Online** — Manage the learner loop through `init_learner`, `pause_learner`, `resume_learner`, `get_learner_stats`, and feed data with `add_training_sample`.
6. **Observe** — Pull telemetry using `get_token_flow_metrics`, `health_check`, and (when hierarchical memory is enabled) `get_hierarchical_metrics`.

Always treat tool responses as authoritative and surface any error text back to the user with context.
```

## Feature Highlights
- **Stdio MCP Server:** Implemented with `McpServer` + `StdioServerTransport`; no HTTP layer is active in this release.
- **Auto-Initialization:** Loads existing checkpoints from `~/.titan_memory` or bootstraps a default config (`inputDim=768`, `memorySlots=5000`, `transformerLayers=6`).
- **Memory Management:** Vector processing, TF-IDF bootstrap helper, gradient reset, and information-gain pruning when supported by the model.
- **Online Learner Loop:** `LearnerService` exposes MCP hooks for replay-buffer based updates. A mock tokenizer is installed by default—swap in `AdvancedTokenizer` for real encodings.
- **Workflow Skeleton:** `src/workflows/` contains orchestrators for release automation, linting, and feedback analysis. These rely on `WorkflowConfig` feature flags defined in `src/types.ts`.
- **Training Pipeline:** `scripts/train-model.ts` pipes into `src/training/trainer.ts`, providing synthetic data generation and tokenizer training when no dataset is supplied.

## Tooling Overview
The server currently registers **19** MCP tools:

`help`, `bootstrap_memory`, `init_model`, `memory_stats`, `forward_pass`, `train_step`, `get_memory_state`, `get_token_flow_metrics`, `get_hierarchical_metrics`, `reset_gradients`, `health_check`, `prune_memory`, `save_checkpoint`, `load_checkpoint`, `init_learner`, `pause_learner`, `resume_learner`, `get_learner_stats`, `add_training_sample`.

The `help` tool now derives its listing dynamically from the active registry, so invoke it at runtime to confirm newly added operations. See [docs/api/README.md](docs/api/README.md) for parameter schemas and defaults. The `manifold_step` operation remains on the roadmap—track progress in [ROADMAP_ANALYSIS.md](ROADMAP_ANALYSIS.md).

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
- [docs/setup-and-tools-guide.md](docs/setup-and-tools-guide.md) — exhaustive setup checklist and tool walkthrough.
- [SYSTEM_AUDIT.md](SYSTEM_AUDIT.md) — consolidated audit and status tracker.
- [IMPLEMENTATION_PACKAGE.md](IMPLEMENTATION_PACKAGE.md) — navigation guide for research and production tasks.
- [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) — current delivery scope and outstanding tasks.
- [ROADMAP_ANALYSIS.md](ROADMAP_ANALYSIS.md) — strategic outlook and feature roadmap.
