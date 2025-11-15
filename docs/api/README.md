# HOPE Memory MCP Server API

Updated October 10, 2025 to reflect the current stdio-based implementation that ships with version 3.0.0 of `@henryhawke/mcp-titan` (server advertises `HOPE Memory` v1.2.0 with Titan aliases for compatibility).

## Overview
- **Transport:** Model Context Protocol over stdio (`StdioServerTransport`). No HTTP endpoints are exposed in this build.
- **Persistence:** Memory state and model artifacts live under `~/.hope_memory/` by default. The server auto-saves every 60 seconds and reloads on startup when files exist.
- **Model stack:** `HopeMemoryModel` backed by TensorFlow.js (Node backend) with optional learner loop and tokenizer services.
- **Source of truth:** All schema definitions are in `src/index.ts`, `src/types.ts`, and `src/learner.ts`. See [docs/architecture-overview.md](../architecture-overview.md) for a component map.

## Requirements & Installation
- Node.js **22.0.0 or newer** (per `package.json` engines field).
- Native dependencies from `@tensorflow/tfjs-node`; install prerequisites for your platform before running `npm install`.
- Optional: network access for dataset/scripts in `scripts/` when using the trainer pipeline.

```bash
npm install -g @henryhawke/mcp-titan
# or run locally
npm install
```

## Running the Server
- CLI entry point: `titan-memory` (see the package `bin` field).
- Local development: `npm start` (loads `dist/index.js`, which invokes the stdio server).
- The process communicates over stdio. Keep it attached to the MCP client or wrap it with a supervisor that forwards stdio.

```bash
# one-off execution
npx titan-memory

# development rebuild + run
npm run build && npm start
```

### Cursor MCP Configuration Example

```json
{
  "mcp": {
    "servers": {
      "titan-memory": {
        "command": "titan-memory",
        "env": {
          "NODE_ENV": "production"
        },
        "workingDirectory": "~/.hope_memory"
      }
    }
  }
}
```

Adjust `workingDirectory` if you need the checkpoints elsewhere. See [README.md](../../README.md) for additional client notes.

## Tool Registry Snapshot
Seventeen tools are registered in `src/index.ts` via `this.server.tool`. The help output will be refreshed to match this table; use `help` for live confirmation.

| Tool | Category | Summary | Key Parameters (defaults) |
| --- | --- | --- | --- |
| `help` | discovery | Lists tools and optional details | `tool?: string`, `category?: string`, `showExamples?: boolean`, `verbose?: boolean` |
| `bootstrap_memory` | onboarding | Seeds memory from URL or raw corpus, builds TF-IDF fallback vectors | `source: string (URL or plain text)` |
| `init_model` | lifecycle | Creates/initializes `HopeMemoryModel` | See [`HopeMemoryConfig` defaults](#hopememoryconfig-defaults) |
| `memory_stats` | observability | Returns raw memory state snapshot via `model.getMemoryState()` | none |
| `forward_pass` | inference | Runs `model.forward` with automatic memory update | `x: string | number[]`, `memoryState?: IMemoryState` |
| `train_step` | training | Executes supervised step between current/next tensors | `x_t`, `x_next` (string or numeric array) |
| `get_memory_state` | observability | Provides formatted stats + quick health check | none |
| `get_token_flow_metrics` | observability | Summarizes token flow window, weights, and variance when enabled | none |
| `reset_gradients` | training | Calls `model.resetGradients()` | none |
| `prune_memory` | maintenance | Invokes `model.pruneMemoryByInformationGain` when available | `threshold?: number (0–1)`, `force?: boolean (default false)` |
| `save_checkpoint` | persistence | Serializes tensors + config to disk | `path: string` (resolved/sanitized) |
| `load_checkpoint` | persistence | Loads checkpoint file, validates dimensionality | `path: string` |
| `init_learner` | learner loop | Creates `LearnerService` with mock tokenizer unless one exists | see [`LearnerConfig` defaults](#learnerservice-defaults) |
| `pause_learner` | learner loop | Stops the training interval | none |
| `resume_learner` | learner loop | Restarts the training interval | none |
| `get_learner_stats` | learner loop | Dumps buffer size, loss metrics, and running status | none |
| `add_training_sample` | learner loop | Pushes sample(s) into replay buffer | `input`, `target`, optional `positive`, `negative` (string or numeric array) |

> **Note:** `manifold_step`, `encode_text`, `get_surprise_metrics`, `analyze_memory`, and `predict_next` are roadmap items. They are referenced in strategy docs but have no MCP handlers today.

### `HopeMemoryConfig` Defaults
The `init_model` schema ultimately uses the `HopeMemoryConfigSchema` from `src/types.ts` with these defaults:

| Field | Default |
| --- | --- |
| `inputDim` | `256`
| `hiddenDim` | `192`
| `memoryDim` | `256`
| `shortTermSlots` | `64`
| `longTermSlots` | `256`
| `archiveSlots` | `512`
| `learningRate` | `0.001`
| `dropoutRate` | `0.1`
| `promotionThreshold` | `0.05`
| `surpriseRetention` | `0.85`
| `routerTopK` | `2`

### `LearnerService` Defaults
When `init_learner` runs without overrides, it instantiates `LearnerService` with:

| Field | Default |
| --- | --- |
| `bufferSize` | `10000`
| `batchSize` | `32`
| `updateInterval` | `1000` ms
| `gradientClipValue` | `1.0`
| `contrastiveWeight` | `0.2`
| `nextTokenWeight` | `0.4`
| `mlmWeight` | `0.4`
| `accumulationSteps` | `4`
| `learningRate` | `0.0001`
| `nanGuardThreshold` | `1e-6`

By default, `init_learner` injects a lightweight mock tokenizer that produces random vectors. Replace `this.tokenizer` on the `TitanMemoryServer` instance if you want deterministic encodings before adding training samples.

## Persistence & Auto-Initialization
- On launch, the server ensures `~/.hope_memory/` exists, then looks for `model/model.json` and `memory_state.json` under that directory.
- If a trained model is present, it loads it; otherwise it initializes a fresh model using the defaults above and immediately saves the artifact for future sessions.
- Checkpoints written via `save_checkpoint` include tensor shapes for validation during `load_checkpoint` and store the current model config alongside the flattened memory arrays.

## Error Handling Notes
- All tool handlers bubble readable error messages through MCP responses; failed operations return a `text` payload with context.
- `prune_memory` and learner tools first check that the underlying optional capabilities exist (`pruneMemoryByInformationGain`, `LearnerService` respectively) before acting.

## Related Documentation
- [docs/architecture-overview.md](../architecture-overview.md) — component relationships and data flow.
- [ROADMAP_ANALYSIS.md](../../ROADMAP_ANALYSIS.md) — planned enhancements, including outstanding tool gaps and workflow integration.
- [SYSTEM_AUDIT.md](../../SYSTEM_AUDIT.md) — consolidated audit findings and action tracker.
- [IMPLEMENTATION_COMPLETE.md](../../IMPLEMENTATION_COMPLETE.md) — current delivery scope and next actions.
- [IMPLEMENTATION_PACKAGE.md](../../IMPLEMENTATION_PACKAGE.md) — navigation guide for documentation and feature work.
