# Titan Memory MCP Server — Setup & Tooling Guide

This guide is the canonical reference for installing, configuring, and operating the Titan Memory MCP server. It assumes no prior context beyond a fresh clone of the repository and walks through day-one setup all the way to issuing MCP tool calls.

## 1. Environment Checklist

| Requirement | Version / Notes |
| --- | --- |
| Node.js | 22.0.0 or newer (enforced by `package.json` engines) |
| npm | Bundled with Node 22; pnpm/yarn also work if you adapt the scripts |
| Python toolchain | Required for native deps used by `@tensorflow/tfjs-node` |
| Disk space | ~2 GB for node_modules + default memory artifacts |
| Optional GPU | Titan uses CPU kernels by default; install `@tensorflow/tfjs-node-gpu` if you have CUDA |

> **Tip:** Run `node -v` and `npm -v` before continuing. If you are upgrading Node, clear previous `node_modules` to avoid ABI mismatches.

## 2. Repository Bootstrap

```bash
# Clone and enter the project
git clone https://github.com/henryhawke/mcp-titan.git
cd mcp-titan

# Install dependencies
npm install

# Build TypeScript sources into dist/
npm run build

# Optional: run the unit tests (Jest + TensorFlow.js)
npm test
```

The build step transpiles the TypeScript sources under `src/` into the `dist/` directory. The CLI entrypoint (`titan-memory`) and the MCP stdio server both run from the compiled output.

## 3. Launching the MCP Server

### 3.1 Local CLI

```bash
# Start the stdio server (development build)
npm start

# or use the installed binary after npm install -g / npx
npx titan-memory
```

The server writes runtime artifacts under `~/.titan_memory/` by default:

- `model/` — serialized encoder/decoder weights and configuration
- `memory_state.json` — current short-term/long-term tensors
- `logs/` — structured JSON logs emitted by `StructuredLogger`

Override the storage directory when embedding Titan inside another process:

```bash
npx titan-memory --memoryPath /custom/path
```

### 3.2 Auto-Initialization

On first launch Titan will:

1. Compile the encoder/decoder graph.
2. Initialize memory tensors (short-term, long-term, metadata, surprise history).
3. Create optional research features if enabled in the config (momentum buffers, token-flow queues, forgetting gate variable, hierarchical tiers).
4. Persist the freshly initialized state to `~/.titan_memory/` for reuse.

Subsequent launches automatically load any saved checkpoint before registering MCP tools.

## 4. Configuring the Model (`init_model`)

The `init_model` tool is the only entrypoint that mutates the model configuration at runtime. Pass a subset of [`TitanMemoryConfig`](docs/api/README.md#titanmemoryconfig-defaults) properties; unspecified fields fall back to defaults.

Key momentum-related knobs enabled by this release:

| Field | Description | Default |
| --- | --- | --- |
| `momentumLearningRate` | Base θ value for the momentum term (scales the gradient contribution) | `0.001` |
| `momentumScoreGain` | Extra θ boost proportional to attention scores (strengthens updates for highly attended slots) | `0.5` |
| `momentumScoreToDecay` | Adjusts η per slot using attention scores (higher attention → slower forgetting) | `0.2` |
| `momentumSurpriseGain` | Scales θ based on the surprise scalar tracked by the model | `0.25` |
| `momentumScoreFloor` | Floor applied to attention weights to keep gradients numerically stable | `0.001` |

Example payload sent from an MCP client:

```json
{
  "method": "init_model",
  "params": {
    "config": {
      "inputDim": 768,
      "memoryDim": 1024,
      "memorySlots": 4096,
      "enableMomentum": true,
      "momentumLearningRate": 0.0025,
      "momentumScoreGain": 0.75,
      "momentumScoreToDecay": 0.15,
      "momentumSurpriseGain": 0.4
    }
  }
}
```

Re-run `init_model` to change parameters mid-session; Titan reuses existing checkpoints when shapes remain compatible.

## 5. Integrating with MCP Clients

### Cursor (desktop or cloud)

Add the following entry to `~/.cursor/settings.json`:

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

1. Open **Settings → MCP Servers → Add Server**.
2. Choose **Run a binary on this machine**.
3. Point to the `titan-memory` executable (global install or the `dist/bin` path inside this repo).
4. Launch Titan prior to connecting to reduce start-up latency.

### Manual JSON-RPC (for debugging)

Use the included test harness to issue tool calls without an MCP client:

```bash
node test_mcp_client.js --tool forward_pass --payload '{"x": [0.1, 0.2, 0.3]}'
```

`test_mcp_client.js` wraps stdio with a simple REPL loop; see `--help` for batching options and payload validation.

## 6. Tool Reference

Titan registers 19 MCP tools. The table below summarizes purpose, prerequisites, and the quickest way to see them in action. For detailed schemas visit [docs/api/README.md](docs/api/README.md).

| Tool | Category | When to Use | Quick Start |
| --- | --- | --- | --- |
| `help` | discovery | Inspect tool list or specific schemas | `node test_mcp_client.js --tool help` |
| `bootstrap_memory` | onboarding | Seed memory with TF-IDF summaries from a URL or blob | Provide `source` as text or HTTPS URL |
| `init_model` | lifecycle | Initialize or reconfigure the model | Pass `config` overrides (see §4) |
| `memory_stats` | observability | Raw tensor statistics & capacity | No params |
| `forward_pass` | inference | Run a single input through encoder → memory → decoder | `x` accepts text or vector |
| `train_step` | supervised learning | Update memory + weights with current/next pair | Provide `x_t` and `x_next` |
| `get_memory_state` | observability | High-level summary of slots, promotions, momentum | No params |
| `get_token_flow_metrics` | observability | Inspect rolling token-flow window and weights | Requires `enableTokenFlow` |
| `reset_gradients` | training | Zero optimizer accumulators | No params |
| `prune_memory` | maintenance | Run information-gain pruning | Optional `threshold` override |
| `save_checkpoint` | persistence | Snapshot model + memory | Provide destination path |
| `load_checkpoint` | persistence | Restore a prior snapshot | Provide source path |
| `init_learner` | learner loop | Start online learner with optional tokenizer swap | Optional `config` block |
| `pause_learner` | learner loop | Temporarily halt learner updates | No params |
| `resume_learner` | learner loop | Resume the learner loop | No params |
| `get_learner_stats` | learner loop | Buffer depth, loss curves, runtime flags | No params |
| `add_training_sample` | learner loop | Push sample(s) into replay buffer | Accepts string or numeric arrays |
| `health_check` | diagnostics | Structured health + dependency check | No params |
| `get_hierarchical_metrics` | observability | Promotion/demotion counters (requires hierarchical memory) | Enable `enableHierarchicalMemory` before calling |

### Sample Tool Flow

A minimal warm-up sequence you can replay during smoke tests:

1. `init_model` with any configuration overrides.
2. `bootstrap_memory` using a short text snippet.
3. `forward_pass` with a query similar to the bootstrap corpus.
4. `train_step` with `(x_t, x_next)` pairs to trigger the new momentum update.
5. `memory_stats` and `get_memory_state` to verify the short-term tensor and momentum buffers change.
6. `save_checkpoint` → restart server → `load_checkpoint` to confirm persistence.

## 7. Observability & Logs

- Logs stream to `~/.titan_memory/logs/*.jsonl`. Tail them with `npm run tail-logs` (see `package.json` scripts).
- Structured telemetry (per-operation latency, error counters) is available through `ModelTelemetry`. Access it via `health_check` or by reading the `telemetry` section of saved checkpoints.
- The new momentum pipeline surfaces row-wise norms in `get_memory_state`. Look for the `momentumNorm` field under `meta` to spot unhealthy growth.

## 8. Troubleshooting Checklist

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `Cannot find module '@tensorflow/tfjs-node'` | Node headers missing during install | Install build tools (`apt-get install build-essential python3`) and rerun `npm install` |
| MCP client exits immediately | Server not on `$PATH` or lacking execute bit | Run via `npm start` or `npx titan-memory` to verify |
| Momentum stays zero after training | `enableMomentum` disabled or surprise is zero | Re-run `init_model` with momentum enabled; feed non-trivial `train_step` pairs |
| `load_checkpoint` fails validation | Tensor shapes changed between sessions | Re-initialize model with matching `memoryDim` / `memorySlots` |
| High memory usage | Token flow window too large or hierarchical tiers active | Reduce `tokenFlowWindow` or disable hierarchical mode |

## 9. Next Steps

- Read [`docs/architecture-overview.md`](docs/architecture-overview.md) for component diagrams.
- Explore [`docs/api/README.md`](docs/api/README.md) for full schemas and optional fields.
- Review [`IMPLEMENTATION_PROGRESS.md`](../IMPLEMENTATION_PROGRESS.md) to understand roadmap status.
- When extending the server, update this guide alongside README so new operators have a single source of truth.

Happy hacking! Titan now ships with adaptive momentum math—experiment with the new coefficients and share telemetry in pull requests.
