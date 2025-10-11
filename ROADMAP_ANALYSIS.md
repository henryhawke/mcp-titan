# MCP-Titan Architecture Analysis & Roadmap

Status date: October 10, 2025. This document synchronizes the architecture analysis with the audited codebase (Phase 0) and the refreshed documentation set (Phase 1).

## Executive Summary
- The stdio MCP server is stable and exposes **17** tools spanning discovery, onboarding, inference, observability, maintenance, persistence, and learner management.
- Auto-initialization, checkpointing, and tensor validation are implemented; advanced memory maneuvers (`manifold_step`, hierarchical promotions) remain stubs.
- Online learning infrastructure exists (replay buffer, gradient accumulation) but ships with a mock tokenizer by default; plugging in `AdvancedTokenizer` is the next milestone.
- Workflow automation modules are scaffolded and require production hardening (auth, rate limiting, observability).

## Current Implementation Snapshot

### MCP & Transport
- `TitanMemoryServer` loads `McpServer` with `StdioServerTransport` and publishes server metadata (`name: "Titan Memory"`, `version: "1.2.0"`).
- Tool registry aligns with [docs/api/README.md](docs/api/README.md); `manifold_step` is mentioned in help text but not registered.
- Auto-save cadence: 60 seconds with retry logic for transient errors; graceful shutdown flushes state and disposes TF resources.

### Memory Model & Configuration
- `TitanMemoryModel` leverages TensorFlow.js (Node backend) and supports telemetry capture, pruning statistics, and memory snapshot exports.
- `TitanMemoryConfigSchema` enables feature flags for momentum updates, forgetting gate, token flow tracking, and hierarchical memory (disabled by default).
- Default config: `inputDim=768`, `memoryDim=1024`, `memorySlots=5000`, `transformerLayers=6`, with optional flags described in [docs/api/README.md](docs/api/README.md#titanmemoryconfig-defaults).

### Learner & Training
- `LearnerService` provides replay buffer, gradient accumulation, and loss mixing (contrastive, next-token, MLM). Defaults documented in [docs/api/README.md](docs/api/README.md#learnerservice-defaults).
- CLI scripts (`scripts/train-model.ts`, `scripts/download-data.ts`) wire into `src/training/trainer.ts`, which can generate synthetic corpora when datasets are unavailable.
- Mock tokenizer in `init_learner` should be replaced before serious training; plug in `AdvancedTokenizer` from `src/tokenizer/` to align vector spaces.

### Workflow Integrations
- `WorkflowOrchestrator` coordinates GitHub automation, linting, and feedback ingestion using `WorkflowConfig` feature toggles.
- Memory-backed logging stores workflow execution results via `TitanMemoryModel.storeWorkflowMemory`.
- Productionization todo: secure credential storage, API rate limiting, external dependency resilience.

## Gaps & Risks
1. **Missing Tool Exposure:** `manifold_step` (and other advanced operations referenced in strings) lacks a registered MCP handler.
2. **Summarization Heuristic:** `bootstrap_memory` relies on simple sentence heuristics; integrate tokenizer-based summarization or external LLM once available.
3. **Tokenizer Consistency:** Learner mock tokenizer produces random tensors, leading to unstable training until replaced.
4. **Test Coverage:** No `src/tests/` tree; existing tests live under `test/` but do not exercise new learner/workflow code.
5. **HTTP Documentation Drift:** Legacy references to HTTP endpoints were removed; ensure external clients adopt stdio configuration.

## Roadmap

### Near Term (Sprint-ready)
- Register `manifold_step` MCP tool wired to `TitanMemoryModel.manifoldStep` (if the method is production-ready) or remove mention from help text.
- Swap the learner mock tokenizer with `AdvancedTokenizer` and add deterministic encoding tests.
- Expand auto-save telemetry and surface metrics via `memory_stats` or a new `get_telemetry` tool.

### Mid Term (1–2 months)
- Harden workflow managers: support GitHub App auth, configurable rate limiting, and persistent queueing for retries.
- Add integration tests that spin up the server over stdio and exercise the MCP tool set.
- Extend checkpoint schema for hierarchical memory tiers if `enableHierarchicalMemory` becomes active by default.

### Longer Term
- Research `manifold_step` equivalents once the manifold math inside `TitanMemoryModel` stabilizes.
- Evaluate replacing heuristic summarization with the learner once tokenizer embeddings are stable.
- Consider alternative backends (e.g., ONNX Runtime) for improved inference throughput.

## Integration Considerations
- **Clients:** Cursor, Claude Desktop, and other MCP-compatible tools must run the `titan-memory` binary; no HTTP endpoint exists in this release.
- **Persistence:** Default `memoryPath` is `~/.titan_memory`; ensure hosting environments grant read/write access to that directory.
- **Node Version:** Require Node 22+ everywhere (enforced via `package.json` and documented in [README.md](README.md)).

## References
- [README.md](README.md)
- [docs/api/README.md](docs/api/README.md)
- [docs/architecture-overview.md](docs/architecture-overview.md)
- [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)
