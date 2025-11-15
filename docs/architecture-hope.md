# HOPE Memory Architecture

The HOPE (Hierarchical Online Persistent Encoding) architecture replaces the Titan memory stack with a continuum memory system that combines retentive sequence modeling, adaptive routing, and transmission-aware training hooks. The implementation lives under `src/hope_model/` and is composed of the following modules:

| Module | Responsibility |
| --- | --- |
| `continuum_memory.ts` | Hierarchical short-, long-, and archival memory levels with promotion, pruning, and statistics helpers. |
| `mamba_filters.ts` | Mamba-inspired selective state-space filter that provides content-aware retention gates. |
| `memory_router.ts` | Sparse Mixture-of-Experts router that selects memory experts per token and produces surprise metrics. |
| `optimizer_hooks.ts` | Gradient compression, layer scheduling, and update buffering utilities for efficient training. |
| `retention_core.ts` | Chunkwise retentive core that fuses token embeddings with memory context and maintains recurrent state. |
| `index.ts` | The `HopeMemoryModel` integration layer that wires the components together and exposes the MCP-facing API. |

## Sequence Processing

`HopeMemoryModel` relies on `RetentiveCore` to process token batches with linear complexity. The core receives concatenated token embeddings and memory readouts, applies a gated recurrence, and feeds the result through the selective state-space filter. Outputs are projected to logits, while intermediate gates expose retention diagnostics.

## Continuum Memory

The continuum memory maintains short-term detail, long-term summaries, and an archival tier. Writes normalise incoming embeddings, append them to the short-term buffer, and automatically promote overflow entries. Pruning removes low-magnitude items and statistics report utilisation and average surprise. Serialization captures all tiers (`shortTerm`, `longTerm`, `archive`, `levelIndex`, `surpriseBuffer`) for persistence.

## Routing and Surprise

`MemoryRouter` routes both writes and reads to a small set of experts. Softmax weights are sparsified to the configured `topK` experts, and entropy-based surprise scores drive memory prioritisation. Router decisions are embedded in the `IMemoryUpdateResult` attention block so downstream tools can inspect gating behaviour.

## Optimizer Hooks

Training pipelines can use:

- **DeltaCompressionHook** – stores deltas relative to previous gradients to minimise communication payloads.
- **LayerScheduler** – selects the highest-magnitude gradients each step.
- **UpdateBuffer** – aggregates layer updates before application or checkpoint logging.

`HopeMemoryModel.trainStep` integrates these hooks before calling the Adam optimizer.

## Server Integration

`HopeMemoryServer` (see `src/index.ts`) loads `HopeMemoryModel`, exposes MCP tools, and persists state via `RobustPersistenceManager`. The server serialises the expanded HOPE memory state, hydrates the model on load, and keeps Titan compatibility aliases (`TitanMemoryModel`, `TITAN_MEMORY_VERSION`) for external code.

## Testing

New Jest suites under `test/test_hope_model/` cover the HOPE modules:

- Retentive core forward/step parity and gate shapes.
- Continuum memory promotion, pruning, and statistics.
- Memory router gating semantics.
- Optimizer hook compression and buffering.

End-to-end smoke tests continue to exercise the MCP server using the HOPE implementation.
