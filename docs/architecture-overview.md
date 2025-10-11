# Architecture Overview

This document captures the relationships between the MCP transport, memory model, learner service, and workflow utilities that make up the Titan Memory server (state as of October 10, 2025).

## Topology

```mermaid
graph TD
    subgraph MCP Server
        A[TitanMemoryServer]
        B[McpServer API]
        C[Tool Handlers]
    end

    subgraph Model Layer
        D[TitanMemoryModel]
        E[VectorProcessor]
        F[TfIdfVectorizer]
        G[AdvancedTokenizer]
    end

    subgraph Persistence
        H[.titan_memory/model]
        I[.titan_memory/memory_state.json]
        J[User Checkpoints]
    end

    subgraph Learner
        K[LearnerService]
        L[Replay Buffer]
        M[Gradient Accumulator]
    end

    subgraph Workflows
        N[WorkflowOrchestrator]
        O[GitHubWorkflowManager]
        P[LintingManager]
        Q[FeedbackProcessor]
    end

    A --> B --> C --> D
    C --> K
    D --> H
    A --> I
    C --> J
    K --> D
    N --> D
    N --> O
    N --> P
    N --> Q
    F --> D
    E --> D
    G --> D
```

## MCP Transport & Lifecycle
- **Entry point:** `TitanMemoryServer` (`src/index.ts`) instantiates `McpServer` with `StdioServerTransport`.
- **Tool registration:** All MCP tools are bound inside the constructor, grouped into discovery, inference, training, persistence, and learner control.
- **Auto-initialization:** On first request, `autoInitialize()` loads or creates model and memory files beneath `memoryPath` (defaults to `~/.titan_memory`). After initialization the server schedules an auto-save loop (60 second cadence, retry-once with 5 second delay).
- **Shutdown:** Signal handlers flush memory state, dispose of TensorFlow resources, and tear down the learner loop.

## Model Stack
- **`TitanMemoryModel` (`src/model.ts`):** Transformer-inspired memory system with telemetry instrumentation, surprise metrics, and optional hierarchical/quantized state hooks.
- **Memory State Representation:** `IMemoryState` (`src/types.ts`) wraps tensors for short-term, long-term, metadata, timestamps, access counts, and surprise history, plus optional momentum/flow fields.
- **Vector Utilities:** `VectorProcessor` and `SafeTensorOps` enforce tensor validation and safe operations across the API surface.
- **Memory Pruning:** `MemoryPruner` coordinates information-gain scoring and pruning thresholds consumed by the `prune_memory` MCP tool.
- **TF-IDF Bootstrap:** `TfIdfVectorizer` seeds sparse fallbacks for `bootstrap_memory` to leverage when the neural model lacks context.

## Learner Loop
- **`LearnerService` (`src/learner.ts`):** Maintains a ring-buffer replay set, gradient accumulation, and configurable loss weighting (contrastive, next-token, MLM).
- **Tokenizer Injection:** `init_learner` installs a mock tokenizer (random tensors) unless you swap in `AdvancedTokenizer`. Replace `server.tokenizer` before calling learner tools for deterministic embeddings.
- **Control Surface:** Tools `init_learner`, `pause_learner`, `resume_learner`, `get_learner_stats`, and `add_training_sample` manage the learner state.

## Workflow Orchestration
- `WorkflowOrchestrator` wires memory-backed analytics into GitHub automation, linting enforcement, and feedback processing. It depends on `WorkflowConfig` feature flags (`src/types.ts`) and on `TitanMemoryModel.storeWorkflowMemory` hooks.
- `GitHubWorkflowManager`, `LintingManager`, and `FeedbackProcessor` (under `src/workflows/`) represent discrete workflow adapters. `WorkflowUtils` supplies shared helpers for credentials, retry policies, and telemetry scaffolding.
- These modules are currently **experimental**—they are not invoked from the MCP server and require productionization (secure credential storage, rate limiting, retries, centralized logging) before deployment. Decide in Phase 5 whether to integrate or archive.

## Persistence Contract
- **Model artifacts:** Saved to `memoryPath/model/` using TensorFlow.js format. Auto-init writes once to ensure subsequent runs can load without reinitializing weights.
- **Memory state:** Stored as JSON arrays of tensor data with shape metadata. `save_checkpoint`/`load_checkpoint` allow arbitrary paths within `memoryPath` or the current working directory (whitelisted by `validateFilePath`).
- **Checkpoints include:** flattened tensor values, shape tuples, model config, and timestamp—enabling safe reloads after restarts.

## Data Flows
1. **Tool Call:** MCP client invokes tool over stdio.
2. **Validation:** Zod schemas ensure parameter correctness; invalid inputs return textual error messages.
3. **Tensor Processing:** `VectorProcessor` and `TitanMemoryModel` convert inputs, manage memory state, and run inference/training.
4. **Persistence:** Memory updates are written to in-memory tensors, optionally flushed to disk via auto-save or explicit checkpoint.
5. **Learner Feedback:** When active, `LearnerService` polls the replay buffer on a fixed interval (`updateInterval`) and applies gradient updates back to the model.

## Related Documentation
- [docs/api/README.md](api/README.md) — detailed tool reference and schema defaults.
- [README.md](../README.md) — quick start, integration notes, and feature summary.
- [ROADMAP_ANALYSIS.md](../ROADMAP_ANALYSIS.md) — strategic roadmap and open gaps.
- [IMPLEMENTATION_COMPLETE.md](../IMPLEMENTATION_COMPLETE.md) — delivery checklist and next steps.
