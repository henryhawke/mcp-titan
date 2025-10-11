# MCP-Titan Code Inventory (Baseline Verification)

Date: October 10, 2025  
Reviewer: Codex  
Scope: `src/index.ts`, `src/model.ts`, `src/types.ts`, `src/learner.ts`, `src/workflows/`, test infrastructure

---

## 1. `src/index.ts` – TitanMemoryServer

- **Tool Registry (17 tools):**
  - `help(tool?, category?, showExamples?, verbose?)`
  - `bootstrap_memory(source: string | URL)` – fetches documents, seeds TF-IDF summaries, stores via `model.storeMemory`.
  - `init_model(config overrides)` – initializes `TitanMemoryModel` with schema-enforced defaults.
  - `memory_stats()` – returns raw `model.getMemoryState()` output.
  - `forward_pass(x: string | number[], memoryState?)` – runs `model.forward`; auto-updates internal state.
  - `train_step(x_t, x_next)` – validates matching tensor length before calling `model.trainStep`.
  - `get_memory_state()` – returns formatted stats and `performHealthCheck('quick')`.
  - `get_token_flow_metrics()` – surfaces tensor metrics when token flow enabled.
  - `reset_gradients()` – calls `model.resetGradients()`.
  - `prune_memory(threshold?, force?)` – guards for `pruneMemoryByInformationGain`.
  - `save_checkpoint(path)` – validates path, writes shapes + tensors + config + timestamp.
  - `load_checkpoint(path)` – validates path, checks `inputDim`, reconstructs tensors.
  - `init_learner(bufferSize?, batchSize?, updateInterval?, gradientClipValue?, contrastiveWeight?, nextTokenWeight?, mlmWeight?, accumulationSteps?, learningRate?, nanGuardThreshold?)` – initializes `LearnerService` with mock tokenizer if absent.
  - `pause_learner()`, `resume_learner()`, `get_learner_stats()`, `add_training_sample(...)` – learner management suite.

- **Notable Helpers:**
  - `processInput` sanitizes strings/arrays (length checks, NaN guards).
  - `validateFilePath` enforces directory allowlist, strips traversal attempts.
  - `wrapWithMemoryManagementAsync` uses tf engine scopes to minimize leaks.

- **Gaps / Action Items:**
  - Help text now synced but consider generating dynamically from registry to avoid drift.
  - `this.tokenizer?: any` remains loosely typed; replace with concrete interface when AdvancedTokenizer lands.
  - `performHealthCheck` currently stubbed (`quick` check only); expand during production hardening.

---

## 2. `src/model.ts` – TitanMemoryModel

- **Core Capabilities:**
  - Transformer-style forward pass with telemetry instrumentation.
  - Surprise tracking, information-gain pruning, snapshot export helpers.
  - Hooks for momentum, forgetting gate, token flow, hierarchical memory, quantization, contrastive loss.

- **Major Stubs:**
  - Momentum integration functions exist (`computeMomentumUpdate`, `applyMomentumToMemory`) but not invoked.
  - Token flow history (`tokenFlowHistory`, `flowWeights`) populated only when feature flag set; forward/train missing updates.
  - Hierarchical promotion/demotion rules defined but unused.
  - Quantization methods implemented but disabled (`config.enableQuantization` never true).
  - Contrastive learning references negative buffer; buffer management absent.
  - `manifoldStep` placeholder remains unimplemented.

- **Telemetry:**
  - Internal timers/metrics recorded; no public surfacing beyond `get_token_flow_metrics`.

---

## 3. `src/types.ts`

- **Key Interfaces:**
  - `IMemoryState` – includes shortTerm/longTerm/meta tensors, plus optional `momentumState`, `tokenFlowHistory`, `flowWeights`, `forgettingGate`, etc.
  - `TitanMemoryConfig` schema – defaults align with docs (Node 22+).
  - `WorkflowConfig` feature flags to gate workflow integrations.

- **Gaps:**
  - Momentum/forgetting/tokens fields optional but never persisted to disk yet.
  - Research toggles exist; ensure config plumbing once features implemented.

---

## 4. `src/learner.ts`

- Replay buffer with gradient accumulation and configurable weights (contrastive, next token, MLM).
- Mock tokenizer injected by server; expects real tokenizer to follow same interface (`encode`, `decode`, `getSpecialTokens`).
- No deterministic tests; `add_training_sample` accepts strings or numeric arrays and converts to tensors on the fly.
- Dispose pattern relies on caller to manage created tensors—`add_training_sample` cleans up only when arrays provided.

---

## 5. `src/workflows/`

| File | Purpose | Status |
| --- | --- | --- |
| `WorkflowOrchestrator.ts` | Coordinates GitHub, linting, feedback managers | Scaffold; not invoked |
| `GitHubWorkflowManager.ts` | GitHub automation hooks | Placeholder; missing auth/retries |
| `LintingManager.ts` | Linting orchestration | Placeholder logic only |
| `FeedbackProcessor.ts` | Feedback ingestion, memory storage | Stubbed pipelines |
| `WorkflowUtils.ts` | Shared utilities (credential helpers, retries) | Utilities exist; require security review |

These modules are disconnected from the MCP server; decision pending (integrate vs archive).

---

## 6. Test Infrastructure

- Unit tests located under `test/` (legacy). No `src/__tests__/integration.test.ts` despite references in historic documentation.
- No stdio integration harness. Scripts (`test_mcp_client.js`, etc.) rely on manual invocation.
- Jest configured via `jest.config.js`; coverage thresholds unspecified.

---

## 7. Baseline Verification Notes

- **Tool list:** Confirmed 17 active tools; roadmap items explicitly absent.
- **Autosave:** Interval set to 60 s with retry-once mechanism; final failure silent—log metrics in production sprint.
- **Security:** Checkpoint path validation in place; no authentication or rate limiting yet.
- **Performance:** No recent profiling; forward pass and pruning operate synchronously on TF.js backend.
- **Next Steps:** Use this inventory during Phase 1 to validate documentation parity, generate schema definitions, and prioritize tests for uncovered paths.

---

For updates, edit this inventory alongside `SYSTEM_AUDIT.md` and `IMPLEMENTATION_PROGRESS.md`.
