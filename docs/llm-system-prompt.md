# Titan Memory MCP Server – System Prompt & Tool Reference

Last updated: October 10, 2025  
Server version: `Titan Memory` v1.2.0  
Tool registry: 17 tools (see below)

---

## Canonical Prompt Snippet

```markdown
You are connected to the @henryhawke/mcp-titan MCP server over stdio. Follow the official tool schemas in docs/api/README.md and prefer the latest help output at runtime.

- Call `init_model` before other memory operations unless the server confirms an active model.
- Use `bootstrap_memory` to seed context before heavy reads.
- Persist state with `save_checkpoint` / `load_checkpoint`; checkpoints must reside in approved directories.
- Monitor capacity via `get_memory_state` and `get_token_flow_metrics`. Prune with `prune_memory` before capacity exceeds 70%.
- Control the learner loop with `init_learner`, `pause_learner`, `resume_learner`, `get_learner_stats`, and `add_training_sample`.
- Treat textual error responses as guidance and adjust parameters accordingly.
```

---

## Tool Catalog

### Discovery & Lifecycle

| Tool | Purpose | Parameters |
| --- | --- | --- |
| `help` | Lists tools, categories, and optional examples. | `{ tool?: string; category?: string; showExamples?: boolean; verbose?: boolean; }` |
| `bootstrap_memory` | Fetches documents from a URL or raw corpus, seeds TF-IDF fallbacks, and stores summaries in memory. | `{ source: string; } // URL or plain text` |
| `init_model` | Instantiates `TitanMemoryModel` with configurable dimensions and flags. Defaults follow `TitanMemoryConfig`. | `{ inputDim?: number; hiddenDim?: number; memoryDim?: number; transformerLayers?: number; numHeads?: number; ffDimension?: number; dropoutRate?: number; maxSequenceLength?: number; memorySlots?: number; similarityThreshold?: number; surpriseDecay?: number; pruningInterval?: number; gradientClip?: number; }` |

### Inference & Training

| Tool | Purpose | Parameters |
| --- | --- | --- |
| `forward_pass` | Runs a forward pass with optional existing memory state; updates internal memory automatically. | `{ x: string | number[]; memoryState?: IMemoryState; }` |
| `train_step` | Performs a supervised update between `x_t` and `x_next`. Validates matching dimensions. | `{ x_t: string | number[]; x_next: string | number[]; }` |
| `reset_gradients` | Clears accumulated gradients to recover from divergence. | `{} (no params)` |
| `add_training_sample` | Pushes samples into the learner replay buffer with optional contrastive pairs. | `{ input: string | number[]; target: string | number[]; positive?: string | number[]; negative?: string | number[]; }` |

### Observability & Maintenance

| Tool | Purpose | Parameters |
| --- | --- | --- |
| `memory_stats` | Returns raw memory tensors/stats. Intended for debugging. | `{} (no params)` |
| `get_memory_state` | Summarizes capacity, surprise score, pattern diversity, and quick health check. | `{} (no params)` |
| `get_token_flow_metrics` | Reports token flow window size, weight statistics, and variance when `enableTokenFlow` is active. | `{} (no params)` |
| `prune_memory` | Runs information-gain pruning with optional threshold override. | `{ threshold?: number; force?: boolean; }` |

### Persistence

| Tool | Purpose | Parameters |
| --- | --- | --- |
| `save_checkpoint` | Serializes memory tensors, shapes, and config to a file inside approved directories. | `{ path: string; }` |
| `load_checkpoint` | Loads checkpoint data, validating tensor shapes and `inputDim`. | `{ path: string; }` |

### Learner Control

| Tool | Purpose | Parameters |
| --- | --- | --- |
| `init_learner` | Starts `LearnerService` with configurable buffer/training hyperparameters. Injects mock tokenizer if none present. | `{ bufferSize?: number; batchSize?: number; updateInterval?: number; gradientClipValue?: number; contrastiveWeight?: number; nextTokenWeight?: number; mlmWeight?: number; accumulationSteps?: number; learningRate?: number; nanGuardThreshold?: number; }` |
| `pause_learner` | Pauses the learner update interval. | `{} (no params)` |
| `resume_learner` | Resumes the learner update interval. | `{} (no params)` |
| `get_learner_stats` | Returns learner buffer size, step counts, and recent loss metrics. | `{} (no params)` |

---

## Usage Notes

- `manifold_step`, `encode_text`, `get_surprise_metrics`, `analyze_memory`, and `predict_next` are roadmap items. Avoid calling them until new handlers ship.
- `bootstrap_memory` may fetch external resources; ensure the environment allows network access if required.
- `save_checkpoint` and `load_checkpoint` enforce a path allowlist; use absolute paths under `~/.titan_memory` or the working directory.
- `init_learner` installs a random-vector tokenizer by default. Replace `server.tokenizer` with `AdvancedTokenizer` if deterministic embeddings are needed before enqueuing samples.
- For long-running sessions, periodically call `get_memory_state` and `prune_memory` to prevent unchecked growth.

---

### Example Workflow

```typescript
await callTool("init_model", { memorySlots: 8000, enableMomentum: true });
await callTool("bootstrap_memory", { source: "https://example.org/notes.txt" });
await callTool("forward_pass", { x: "Summarize the previous meeting notes." });
await callTool("prune_memory", { threshold: 0.75 });
await callTool("save_checkpoint", { path: "~/.titan_memory/checkpoints/session-001.json" });
```

---

For schema updates, consult `docs/api/README.md`. For implementation details and roadmap status see `SYSTEM_AUDIT.md`, `IMPLEMENTATION_PACKAGE.md`, and `ROADMAP_ANALYSIS.md`.
