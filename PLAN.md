# MCP Titan Complete Implementation Roadmap

**Status:** Phase 2 In Progress

**Target:** Production-ready real-time learning MCP server implementing full Titans architecture

**Estimated Time:** 6-8 weeks (1 developer)

---

## Phase 0: Documentation Consolidation & Cleanup

### 0.1 Audit Current Documentation State

- Read all .md files: README.md, DETAILED_IMPLEMENTATION_GUIDE.md, IMPLEMENTATION_PROGRESS.md, COMPLETE_IMPLEMENTATION_PACKAGE.md, ROADMAP_ANALYSIS.md, AUDIT_FINDINGS.md, AUDIT_IMPLEMENTATION_SUMMARY.md, GUIDE_UPDATE_SUMMARY.md, mcp-titan-system-audit.md
- Create documentation dependency map showing overlaps and contradictions
- **Specific Issues to Document:**
  - docs/api/README.md claims `process_input` tool exists (NOT FOUND in src/index.ts)
  - ROADMAP_ANALYSIS.md states "16 sophisticated memory tools" but only 15 registered (src/index.ts:142-882)
  - IMPLEMENTATION_COMPLETE.md claims "production-ready" contradicts PRODUCTION_READINESS_ANALYSIS.md
  - README.md references `mcp-titan` command but package.json defines `titan-memory` as bin
  - docs/llm-system-prompt.md may not reflect actual tool schemas
  - Help text references `manifold_step` which is not exposed (src/index.ts:155)

### 0.2 Consolidate Core Documentation

- Merge AUDIT_FINDINGS.md + AUDIT_IMPLEMENTATION_SUMMARY.md + mcp-titan-system-audit.md into single SYSTEM_AUDIT.md with status tracking
- Consolidate COMPLETE_IMPLEMENTATION_PACKAGE.md + GUIDE_UPDATE_SUMMARY.md into IMPLEMENTATION_PACKAGE.md
- Update DETAILED_IMPLEMENTATION_GUIDE.md to remove redundant sections
- Archive obsolete documents to docs/archive/
- **Fix Specific Documentation:**
  - Update docs/api/README.md with actual 15 tool list (not 16)
  - Remove `process_input` references
  - Add missing tools: `bootstrap_memory`, `encode_text`, `get_surprise_metrics`, `analyze_memory`, `predict_next`
  - Reconcile production readiness claims

### 0.3 Create Master Progress Tracker

- Update IMPLEMENTATION_PROGRESS.md with phase checkboxes (Phase 0-10)
- Add per-feature status matrix (Not Started / In Progress / Testing / Complete)
- Include real-time learning validation checkpoints
- Document canonical sources: which file is source of truth for each feature
- Add metrics tracking: test coverage %, tool response times, memory usage

### 0.4 Update Architecture Documentation

- docs/architecture-overview.md missing src/workflows/ directory (5 files)
- Add: GitHubWorkflowManager, LintingManager, WorkflowOrchestrator, FeedbackProcessor, WorkflowUtils
- Add: VectorProcessor, SafeTensorOps, MemoryPruner class documentation
- Document intended use cases or mark for removal

---

## Phase 1: Baseline Verification & Code Inventory

### 1.1 Complete Code Inventory

- **src/index.ts (1101 lines):** Document all 15 registered tools with exact signatures
  - Actual tools: help, init_model, forward_pass, train_step, get_memory_state, manifold_step, prune_memory, save_checkpoint, load_checkpoint, reset_gradients, bootstrap_memory, encode_text, get_surprise_metrics, analyze_memory, predict_next
  - Note: `manifold_step` registered but marked unimplemented (src/model.ts:2890+)
  - **Tool Issues Found:**
    - Inconsistent error handling (some use try-catch with telemetry lines 400-500, others throw directly lines 600-700)
    - Missing parameter validation (forward_pass accepts string/array without type guard)
    - train_step doesn't validate x_t and x_next have same dimensions
    - Memory state updated but not automatically persisted (autoSaveInterval line 62, save logic lines 1100-1200 may fail silently)

- **src/model.ts (3240 lines):** Map implementations vs stubs
  - Line 230-3240: Extensive TitanMemoryModel implementation
  - **Stubs/Gaps:**
    - trainStep() line 1800-2000: Uses optimizer momentum but doesn't expose S_t for memory updates
    - Hierarchical memory lines 251-276: `extendedMemoryState` and `promotionRules` defined but not used in forward()/trainStep()
    - updateMetaMemory(), manifoldStep() marked unimplemented (line 2890+)
    - Quantization lines 570-650: Methods exist but always disabled, config.enableQuantization never set
    - Contrastive learning lines 700-800: Loss computation exists but no negative example buffer in training loop
  - Telemetry collector records timing and tensor counts (line 20)

- **src/types.ts:** Catalog IMemoryState fields and config schema
  - Existing fields: shortTerm, longTerm, meta, timestamps, accessCounts, surpriseHistory
  - **Missing for Titans:** momentumState, momentumDecay, forgettingGate, forgettingGateHistory, tokenFlowHistory, tokenFlowWeights
  - Research toggles exist in config schema (line 65) but most advanced behaviors are stubs

- **src/learner.ts (49+ lines):** Understand replay buffer and gradient accumulation
  - Replay buffer, gradient accumulation, loss mixing implemented
  - init_learner injects mock tokenizer with random vectors (src/index.ts:706) - MUST REPLACE with AdvancedTokenizer
  - Control tools: init_learner, pause_learner, resume_learner, get_learner_stats, add_training_sample

- **src/workflows/ (5 files):** No usage found in main server
  - WorkflowOrchestrator.ts, GitHubWorkflowManager.ts, LintingManager.ts, FeedbackProcessor.ts, WorkflowUtils.ts
  - Requires production hardening (credential handling, retries, analytics)
  - **Decision needed:** Integrate, document, or remove

### 1.2 Test Infrastructure Audit

- Read all files in src/**tests**/ to understand current coverage
- Run npm test to capture baseline pass/fail state
- **Gaps Identified:**
  - No integration tests for end-to-end MCP tool invocation
  - No multi-step workflow tests (init → train → save → load → predict)
  - Mock-heavy tests (learner.test.ts uses MockModel and MockTokenizer) - need real component interaction
  - **Edge cases missing:** malformed MCP requests, concurrent tool calls, memory overflow scenarios
- Create test plan matrix in IMPLEMENTATION_PROGRESS.md

### 1.3 Dependency & Environment Check

- Verify package.json (line 69): Node.js 22.0.0+ requirement
- Check TensorFlow.js @tensorflow/tfjs-node version and native deps installed
- Ensure tsconfig.json enables strict mode for type safety
- **TypeScript Issues Found:**
  - src/model.ts line 3: `/* eslint-disable @typescript-eslint/no-unused-vars */` indicates unused imports
  - src/utils/polyfills.ts has recent modifications (git status) - may have breaking changes
  - src/index.ts line 60: `tokenizer?: any` should be properly typed
- Document missing dependencies in IMPLEMENTATION_PROGRESS.md

### 1.4 Memory & Performance Baseline

- **Memory Leaks Risk:**
  - wrapWithMemoryManagement in src/index.ts lines 91-99 wraps in tf.tidy
  - Async operations may leak tensors if exceptions thrown mid-execution
  - Line 1151: TODO comment about LLM summarizer suggests incomplete feature
- **No Batch Processing:** All tools process single inputs, no batch API
- **Checkpoint Size Unbounded:** Memory state grows indefinitely, pruning exists but not automatic, no compression

---

## Phase 2: Core Momentum & Forgetting Integration

### 2.1 Extend IMemoryState Interface (src/types.ts)

- Add `momentumState: tf.Tensor` field for S_t tracking (Research Paper Equation 33)
- Add `momentumDecay: number` (eta_t parameter, default 0.9)
- Add `forgettingGate: tf.Tensor` for alpha_t parameter (default 0.1) (Research Paper lines 472-476)
- Add `forgettingGateHistory: number[]` for tracking over time
- Update TitanMemoryConfigSchema to include these as optional config parameters
- Add Zod validation for momentum/forgetting ranges (0-1)
- Update checkpoint serialization schema to include new fields
- **Fix:** Validate embedding dimensions on checkpoint load (currently missing in loadCheckpoint tool)
- Test: Write types.test.ts validating schema accepts/rejects edge cases

### 2.2 Implement Momentum Update Core (src/model.ts around line 1500-1600)

- **Research Paper Reference:** Appendix C, Equation 33 (lines 426-489)
- Create `computeMomentumUpdate()` method implementing:
  ```
  S_t = diag(eta_t) * S_{t-1} - diag(theta_t) * (M_{t-1} * k_t^T * k_t - v_t^T * k_t)
  ```

  - Accept parameters: previousMomentum, learningRate, memoryState, keys, values
  - Return updated momentum tensor with proper shape validation
  - **Gap to fill:** Current trainStep() uses optimizer momentum but doesn't expose S_t (line 1800-2000)
- Create `applyMomentumToMemory()` method implementing Equation 32:
  ```
  M_t = diag(1 - alpha_t) * M_t + S_t
  ```

  - Accept parameters: currentMemory, forgettingGate, momentumUpdate
  - Return updated memory tensor
  - Wrap all operations in tf.tidy() for memory management
- Add momentum initialization in TitanMemoryModel constructor
- Test: Write momentum.test.ts with unit tests for both methods

### 2.3 Implement Forgetting Gate Logic (src/model.ts)

- **Research Paper Reference:** Lines 472-476 (weight decay equivalent)
- Create `updateForgettingGate()` method:
  - Compute adaptive alpha_t based on surprise score
  - Use formula: alpha_t = base_alpha * (1 + surprise_weight * surprise_score)
  - Clamp to [0, 1] range
  - Update forgettingGateHistory array
- Create `applyForgettingGate()` method:
  - Apply element-wise multiplication: diag(1 - alpha_t) * M_t
  - Handle tensor shape broadcasting correctly
  - Validate output dimensions match input
- Add forgetting gate to configuration with sensible defaults
- Test: Write forgetting.test.ts validating adaptive behavior

### 2.4 Integrate Momentum into trainStep (src/model.ts line 1800-2000)

- Locate existing trainStep method
- Modify to:

  1. Store previous momentum state before update
  2. Compute current gradient as before
  3. Call computeMomentumUpdate with previous momentum
  4. Update forgetting gate based on current surprise
  5. Call applyMomentumToMemory with new momentum and forgetting gate
  6. Update this.momentumState for next iteration
  7. Persist momentum state in memory snapshots

- Add telemetry for momentum magnitude and forgetting gate values (integrate with ModelTelemetry class line 20)
- Wrap entire flow in memory management wrapper
- **Fix redundant forward passes:** trainStep calls forward() which then calls forward() again - optimize to single pass
- Test: Extend existing train.test.ts to verify momentum accumulation over multiple steps

### 2.5 Update Checkpoint Serialization (src/persistence.ts)

- Modify save_checkpoint to serialize momentum state and forgetting gate
- Encrypt momentum tensors using existing encryption utilities
- Modify load_checkpoint to restore momentum state
- Validate momentum tensor shapes match model config on load (currently missing)
- Add backward compatibility for checkpoints without momentum
- **Fix:** Currently no checkpoint compression - add compression support
- Test: Write persistence.test.ts for momentum save/load cycle

### 2.6 Expose Momentum Diagnostics (src/index.ts)

- Extend get_memory_state tool (standardize error handling first)
- Add momentum statistics:
  - momentumMagnitude: L2 norm of S_t
  - forgettingGateValue: current alpha_t
  - forgettingGateHistory: recent alpha_t values
- Add optional `includeMomentum: boolean` parameter (default false for token efficiency)
- Update Zod schema for get_memory_state response
- **Standardize:** Apply consistent error handling pattern (try-catch with telemetry)
- Test: Write integration test calling get_memory_state after training

### 2.7 Documentation & Validation

- Update DETAILED_IMPLEMENTATION_GUIDE.md Section 1 with "✅ IMPLEMENTED" marker
- Update IMPLEMENTATION_PROGRESS.md: mark Phase 2 complete with date
- Update docs/api/README.md with new get_memory_state fields
- Run full test suite: npm test
- Manually test: init_model → train_step (5x) → get_memory_state, verify momentum present
- Commit with message: "feat: implement Titans momentum and forgetting (Eqs 32-33)"
- **Update this plan document with Phase 2 completion date**

---

## Phase 3: Token Flow & Surprise Weighting

### 3.1 Add Token Flow State (src/types.ts)

- **Research Paper Reference:** Section 3.1 (lines 364-366)
- Add `tokenFlowHistory: number[][]` to IMemoryState (stores recent token embeddings)
- Add `tokenFlowWeights: number[]` (weights for each token in history)
- Add `tokenFlowWindowSize: number` config parameter (default 32)
- Add `tokenFlowDecay: number` config parameter (default 0.95)
- Add `flowWeights` placeholder (src/types.ts:109) - activate this field
- Update checkpoint schema for token flow fields
- Test: Validate schema accepts token flow fields

### 3.2 Implement Token Flow Tracking (src/model.ts)

- **Key Differentiator:** Captures sequential dependencies, not just momentary surprise
- Create `updateTokenFlow()` method:
  - Add current token embedding to tokenFlowHistory
  - Maintain sliding window of size tokenFlowWindowSize
  - Compute recency weights: w_i = decay^(t - i)
  - Store in tokenFlowWeights array
- Create `computeTokenFlowWeights()` method:
  - Compute similarity between current token and history
  - Combine recency and similarity: final_weight = recency_weight * similarity
  - Normalize weights to sum to 1
- Initialize token flow in constructor with empty history
- Update during forward pass (src/model.ts forward path + new tooling)

### 3.3 Integrate Token Flow into Forward Pass (src/model.ts)

- Locate forward() method
- Before memory query, call updateTokenFlow(currentInput)
- Compute token flow weights for current context
- Modify attention computation to incorporate flow weights:
  - weighted_attention = base_attention * (1 + flow_weight_factor * flow_weights)
- Store flow-weighted surprise in surpriseHistory
- **Performance:** Minimize tensor copies (currently multiple clone() operations in memory updates)
- Test: Write tokenflow.test.ts validating weight computation

### 3.4 Weight Surprise by Token Flow (src/model.ts)

- Create `weightSurpriseByTokenFlow()` method:
  - Accept: raw surprise score, token flow weights
  - Compute: weighted_surprise = surprise * (1 + sum(flow_weights * token_similarities))
  - Return: adjusted surprise score
- Integrate into trainStep surprise calculation
- Use weighted surprise for forgetting gate updates
- Test: Verify surprise values differ with/without token flow

### 3.5 Add Token Flow MCP Tool (src/index.ts)

- Register new tool: `get_token_flow_metrics`
- Parameters: `windowSize?: number` (return last N tokens)
- Response schema:
  - tokenFlowHistory: recent token IDs or embeddings
  - tokenFlowWeights: current weights
  - flowStrength: aggregate flow metric
  - topFlowTokens: tokens with highest flow weights
- Add Zod schema validation
- Apply consistent error handling pattern
- Test: Integration test calling tool after multiple forward passes

### 3.6 Documentation & Validation

- Update DETAILED_IMPLEMENTATION_GUIDE.md Section 2: "✅ IMPLEMENTED"
- Update docs/api/README.md with get_token_flow_metrics tool (add to tool list)
- Update IMPLEMENTATION_PROGRESS.md: mark Phase 3 complete
- Test token flow improves prediction on sequential patterns
- Commit: "feat: implement token flow tracking and surprise weighting"
- **Update this plan document with Phase 3 completion date**

---

## Phase 4: Deep Neural Memory Module

### 4.1 Design Deep Memory Architecture (src/model.ts)

- **Research Paper Reference:** Lines 450-452 (neural memory vs matrix-based)
- **Current Gap:** memoryProjector exists but memory stored as tensors, not trainable neural module
- Create `DeepMemoryNetwork` class:
  - Input: [memoryDim] tensor
  - Architecture: 3-layer MLP with skip connections
    - Layer 1: memoryDim → 2*memoryDim, SiLU activation
    - Layer 2: 2*memoryDim → 2*memoryDim, SiLU activation, dropout 0.1
    - Layer 3: 2*memoryDim → memoryDim, linear
  - Skip connection: output = layer3_out + input (residual)
- Add trainable flag to enable/disable gradient updates
- Implement forward() and backward() methods

### 4.2 Configuration & Initialization (src/types.ts, src/model.ts)

- Add `useDeepMemory: boolean` to TitanMemoryConfig (default false)
- Add `deepMemoryLayers: number` config (default 3)
- Add `deepMemoryHiddenDim: number` config (default 2 * memoryDim)
- In TitanMemoryModel.initialize():
  - If useDeepMemory, instantiate DeepMemoryNetwork
  - Otherwise, use existing tensor-based memory
- Handle both paths in all memory operations

### 4.3 Integrate Deep Memory into Forward Pass (src/model.ts)

- In forward(), after computing memory query:
  - If useDeepMemory: pass result through DeepMemoryNetwork.forward()
  - Else: use direct tensor operations
- Ensure output shapes match regardless of path
- Add telemetry for deep memory activations
- Test: Compare forward pass outputs with/without deep memory

### 4.4 Integrate Deep Memory into Training (src/model.ts)

- In trainStep(), after computing memory update:
  - If useDeepMemory: backpropagate through DeepMemoryNetwork
  - Update deep memory weights using optimizer
  - Apply momentum to both memory state and deep network weights
- Checkpoint deep memory network weights
- Test: Verify deep memory weights change during training

### 4.5 Add Deep Memory Diagnostics (src/index.ts)

- Extend get_memory_state with deepMemoryStats:
  - isDeepMemory: boolean
  - layerActivations: mean/std for each layer
  - weightNorms: L2 norm of each layer's weights
  - gradientNorms: during training
- Optional parameter `includeDeepStats: boolean`
- **Response optimization:** Provide summary view with optional detail flag to reduce token usage
- Test: Call tool with deep memory enabled

### 4.6 Documentation & Validation

- Update DETAILED_IMPLEMENTATION_GUIDE.md Section 4: "✅ IMPLEMENTED"
- Add deep memory architecture diagram to docs/architecture-overview.md
- Update IMPLEMENTATION_PROGRESS.md: mark Phase 4 complete
- Benchmark: Compare perplexity with/without deep memory on test set
- Commit: "feat: implement deep neural memory module"
- **Update this plan document with Phase 4 completion date**

---

## Phase 5: Hierarchical Memory Activation

### 5.1 Define Memory Tiers (src/types.ts)

- **Research Paper Reference:** Lines 381-386
- **Current Status:** extendedMemoryState and promotionRules defined (src/model.ts:251-276) but NOT USED
- Add `MemoryTier` enum: WORKING, SHORT_TERM, LONG_TERM
- Add to IMemoryState:
  - workingMemory: tf.Tensor (small, recent items)
  - shortTermMemory: tf.Tensor (medium, frequent items)
  - longTermMemory: tf.Tensor (large, stable items)
  - memoryTierAssignments: Map<itemId, MemoryTier>
  - memoryAccessCounts: Map<itemId, number>
  - memoryLastAccess: Map<itemId, timestamp>
- Add tier capacity configs: workingCapacity, shortTermCapacity, longTermCapacity

### 5.2 Implement Promotion Rules (src/model.ts)

- Create `evaluatePromotion()` method:
  - For each item in working memory:
    - If accessCount > threshold && recency > threshold: promote to short-term
  - For each item in short-term:
    - If accessCount > higher_threshold && age > duration: promote to long-term
  - Return list of items to promote
- Create `promoteToShortTerm()` method:
  - Move item from working → short-term memory tensor
  - Update memoryTierAssignments map
  - Free working memory slot
- Create `promoteToLongTerm()` method:
  - Move item from short-term → long-term memory tensor
  - Apply additional compression/quantization if needed (activate quantization lines 570-650)
  - Update maps and free short-term slot
- Expose stats via new tools (not just updateMetaMemory stubs)

### 5.3 Implement Demotion Rules (src/model.ts)

- Create `evaluateDemotion()` method:
  - For each item in long-term:
    - If not accessed in N steps: demote to short-term or evict
  - For each item in short-term:
    - If accessCount low: demote to working or evict
  - Return list of items to demote/evict
- Create `applyDemotion()` method:
  - Move demoted items down tiers
  - Evict least-accessed items if tier full
  - Update all tracking maps
- **Production:** Make pruning automatic when tier capacity exceeded

### 5.4 Integrate Hierarchical Memory into Forward (src/model.ts)

- In forward(), query all three memory tiers:
  - Primary query: working memory (fast, recent)
  - Secondary query: short-term memory (medium)
  - Tertiary query: long-term memory (slow, stable)
- Combine queries with weighted sum based on relevance
- Update access counts and timestamps for retrieved items
- Test: Verify items retrieved from correct tiers

### 5.5 Integrate Hierarchical Memory into Training (src/model.ts)

- After each trainStep:
  - Call evaluatePromotion() and apply promotions
  - Every N steps, call evaluateDemotion() and apply demotions
  - Update memory state with new tier assignments
- In momentum updates, apply different learning rates per tier:
  - Working: full learning rate
  - Short-term: 0.5 * learning rate
  - Long-term: 0.1 * learning rate (slow changes)
- Test: Verify items promote over multiple training steps

### 5.6 Add Hierarchical Memory Tool (src/index.ts)

- Register tool: `get_hierarchical_memory_stats`
- Response schema:
  - workingMemoryUtilization: % full
  - shortTermMemoryUtilization: % full
  - longTermMemoryUtilization: % full
  - recentPromotions: count of recent working→short, short→long
  - recentDemotions: count of recent demotions
  - tierDistribution: histogram of items per tier
- Test: Call tool and verify stats match internal state

### 5.7 Documentation & Validation

- Update DETAILED_IMPLEMENTATION_GUIDE.md Section 5: "✅ IMPLEMENTED"
- Add hierarchical memory diagram to docs/architecture-overview.md
- Update IMPLEMENTATION_PROGRESS.md: mark Phase 5 complete
- Test: Run 1000 training steps, verify items promote to long-term
- Commit: "feat: implement hierarchical memory with promotion/demotion"
- **Update this plan document with Phase 5 completion date**

---

## Phase 6: Observability & Logging

### 6.1 Create Structured Logger (src/logging.ts)

- **Current Issue:** Telemetry exists (ModelTelemetry class) but only in-memory, no structured logging to files/external systems
- Implement Logger class with levels: DEBUG, INFO, WARN, ERROR
- Support multiple transports: console, file, JSON file
- Include structured context: timestamp, level, component, operation, duration
- Add log rotation: max file size 10MB, keep 5 historical files
- Support log filtering by level and component
- Test: Write logging.test.ts validating all transports

### 6.2 Integrate Logger into Model (src/model.ts)

- Replace all console.log/warn/error with logger calls
- Add logging to:
  - Model initialization with config parameters
  - Forward pass with input/output shapes and timing
  - Train step with loss, momentum magnitude, surprise score
  - Memory promotion/demotion events
  - Pruning operations with items removed
  - Checkpoint save/load with file paths and sizes
- Include correlation IDs for tracing multi-step operations

### 6.3 Integrate Logger into Server (src/index.ts)

- Replace console statements with logger
- Log:
  - Server startup with config
  - Tool invocations with parameters (sanitize sensitive data)
  - Tool execution time and result status
  - Memory state auto-save events (currently lines 1100-1200 may fail silently)
  - Errors with full stack traces
- Add request IDs to track tool call chains

### 6.4 Add Health Check Tool (src/index.ts)

- **Current Gap:** No health checks, no readiness/liveness probes
- Register tool: `health_check`
- Response schema:
  - status: "healthy" | "degraded" | "unhealthy"
  - uptime: seconds since server start
  - memoryUsage: process.memoryUsage()
  - tensorCount: tf.memory().numTensors
  - lastCheckpointTime: timestamp
  - modelStatus: initialized, training, idle
  - errorCount: errors in last hour
  - averageToolLatency: ms per tool call
- Set status based on thresholds:
  - tensorCount > 1000 = degraded
  - errorCount > 10/hour = degraded
  - Memory > 80% = unhealthy
- Test: Call health_check and verify response structure

### 6.5 Optional HTTP Health Endpoints (src/index.ts)

- If HTTP server enabled (stdio transport currently src/index.ts:1101), add routes:
  - GET /health: basic status (200 if healthy)
  - GET /healthz: Kubernetes liveness (200 if server running)
  - GET /readyz: Kubernetes readiness (200 if model initialized)
  - GET /metrics: Prometheus-style metrics
- Use same health check logic as MCP tool
- Test: curl endpoints if HTTP enabled

### 6.6 Documentation & Validation

- Update DETAILED_IMPLEMENTATION_GUIDE.md Section 6: "✅ IMPLEMENTED"
- Create docs/observability.md documenting logging configuration
- Update IMPLEMENTATION_PROGRESS.md: mark Phase 6 complete
- Run server, verify logs written to files with rotation
- Commit: "feat: add structured logging and health checks"
- **Update this plan document with Phase 6 completion date**

---

## Phase 7: Security Hardening

### 7.1 Input Sanitization (src/index.ts)

- **Current Gap:** No input sanitization, text inputs not sanitized before encoding
- Create `sanitizeInput()` utility function:
  - Validate string lengths (max 100k chars)
  - Strip control characters except newlines/tabs
  - Validate UTF-8 encoding
  - Check for code injection patterns
- Create `validateFilePath()` utility:
  - Resolve path to absolute
  - Ensure path is within allowed directories
  - Block path traversal attempts (../)
  - Validate file extensions against whitelist
  - **Fix:** load_checkpoint currently has path traversal risk
- Apply sanitization to all tool inputs accepting strings/paths
- Add type guards for forward_pass input parameter (currently missing)
- Test: security.test.ts with malicious inputs

### 7.2 Checkpoint Encryption (src/persistence.ts)

- **Current Gap:** Memory states saved as plain JSON, no encryption at rest
- Add `encryptCheckpoint: boolean` config (default true)
- Implement encryption using Node.js crypto:
  - Algorithm: AES-256-GCM
  - Key derivation: PBKDF2 from configurable password or env var
  - Store IV with encrypted data
- Modify save_checkpoint to:
  - Serialize memory state to JSON
  - Encrypt JSON if encryptCheckpoint enabled
  - Write encrypted bytes to file with .encrypted extension
- Modify load_checkpoint to:
  - Detect encrypted files by extension or magic bytes
  - Decrypt using provided key
  - Deserialize and validate JSON
- Test: Save/load encrypted checkpoints, verify unreadable without key

### 7.3 API Authentication (src/index.ts)

- **Current Gap:** MCP server accepts all connections, no API key or token validation
- Add `requireAuth: boolean` and `apiKey: string` to config
- Create middleware `authenticateRequest()`:
  - Extract API key from tool parameters or HTTP headers
  - Compare against configured apiKey (constant-time comparison)
  - Reject request if mismatch with 401 error
- Apply middleware to all tool handlers if requireAuth enabled
- Add rate limiting per API key
- Test: Call tools with correct/incorrect/missing API keys

### 7.4 Rate Limiting (src/index.ts)

- **Current Gap:** Tools can be called unlimited times, no protection against resource exhaustion
- Create RateLimiter class:
  - Track requests per client/apiKey in sliding window
  - Default: 100 requests per minute
  - Return 429 Too Many Requests if exceeded
- Apply to all tool handlers
- Add rate limit info to responses (X-RateLimit-* headers)
- Test: Make 101 requests rapidly, verify 101st is rate-limited

### 7.5 Response Sanitization (src/index.ts)

- Create `sanitizeOutput()` utility:
  - Redact potential PII (emails, phone numbers)
  - Limit response size (max 1MB)
  - Escape any code in responses
- Apply to all tool responses before returning
- Test: Verify PII patterns redacted from responses

### 7.6 Documentation & Validation

- Update DETAILED_IMPLEMENTATION_GUIDE.md Section 7: "✅ IMPLEMENTED"
- Create docs/security.md documenting security config
- Update IMPLEMENTATION_PROGRESS.md: mark Phase 7 complete
- Run security audit checklist (OWASP top 10)
- Commit: "feat: implement security hardening (encryption, auth, rate limiting)"
- **Update this plan document with Phase 7 completion date**

---

## Phase 8: Performance Optimization & Caching

### 8.1 Eliminate Redundant Forward Passes (src/model.ts)

- **Current Issue:** trainStep calls forward() which then calls forward() again
- Audit trainStep to identify duplicate forward calls
- Cache forward pass result if inputs unchanged
- Pass cached activations to optimizer instead of recomputing
- Add telemetry for cache hits
- Test: Verify training still converges, measure speedup

### 8.2 In-Place Tensor Operations (src/model.ts)

- **Current Issue:** Multiple clone() operations in memory updates create unnecessary copies
- Identify tensor operations that create unnecessary copies:
  - Replace tensor.add(other) with tf.addInPlace where safe
  - Replace tensor.mul(scalar) with tf.mulInPlace where safe
- Ensure in-place ops don't break autodiff
- Wrap in-place ops in memory management
- Fix async tensor lifetime issues (exceptions mid-execution can leak tensors)
- Test: Verify numerical results unchanged, measure memory reduction

### 8.3 Implement LRU Cache (src/cache.ts)

- **Current Gap:** No response caching, repeated calls recalculate
- Create generic LRUCache class:
  - Max size configurable (default 100 entries)
  - Keys: string hashes of inputs
  - Values: serialized tool responses
  - Eviction: least recently used
- Add TTL support (default 5 minutes)
- Support cache invalidation on state changes

### 8.4 Apply Caching to Read-Only Tools (src/index.ts)

- Wrap these tools with caching:
  - get_memory_state (cache key: state snapshot hash)
  - get_token_flow_metrics (cache key: input sequence hash)
  - analyze_memory (cache key: memory state hash)
  - get_surprise_metrics (cache key: history hash)
  - get_hierarchical_memory_stats (cache key: tier state hash)
- Invalidate caches after write operations (train_step, prune_memory)
- Add cache hit/miss metrics to telemetry
- Test: Call same tool twice, verify second is cached

### 8.5 Response Compression (src/index.ts)

- **Token Efficiency:** get_memory_state returns entire statistics object - optimize
- Add `compressResponses: boolean` config (default false)
- For large responses (>10KB):
  - Compress JSON with gzip
  - Encode as base64
  - Add Content-Encoding indicator
- Decompress transparently if client supports
- Consider binary formats (Protocol Buffers, MessagePack) for serialization
- Test: Verify compressed responses 50%+ smaller

### 8.6 Optimize Transformer Stack (src/model.ts)

- **Current:** 6 layers default (TitanMemoryModel.initialize()), each full attention
- Review attention mechanism for optimization opportunities:
  - Consider sparse attention for long sequences
  - Use FlashAttention-style fusion if available
  - Cache attention weights for repeated queries
- Benchmark attention computation time
- Document optimization opportunities in code comments
- Align with research paper guidance (lines 352-366)

### 8.7 Documentation & Validation

- Update DETAILED_IMPLEMENTATION_GUIDE.md Section 8: "✅ IMPLEMENTED"
- Create docs/performance.md documenting cache configuration
- Update IMPLEMENTATION_PROGRESS.md: mark Phase 8 complete
- Run performance benchmarks before/after optimizations
- **Target:** <100ms tool response time (95th percentile)
- Commit: "perf: optimize tensor ops, add caching, compress responses"
- **Update this plan document with Phase 8 completion date**

---

## Phase 9: Tokenizer & Embedding Integration

### 9.1 BPE Training Integration (src/tokenizer/)

- **Current Gap:** bpe.ts has training methods but never called from main server, no automatic vocabulary building
- Wire up BPE training on first run or explicit call
- Create `train_tokenizer` MCP tool
- Persist vocabulary to disk
- Test: Train on sample corpus, verify consistent encoding

### 9.2 Advanced Tokenizer Replacement (src/index.ts, src/learner.ts)

- **Critical Fix:** init_learner injects mock tokenizer with random vectors (src/index.ts:706)
- Replace MockTokenizer with AdvancedTokenizer from src/tokenizer/
- Ensure vector spaces are consistent
- Add deterministic tests around add_training_sample
- Test: Verify learner works with real tokenizer

### 9.3 Embedding Dimension Validation

- **Current Gap:** Default inputDim 768 (src/index.ts:82) but embedding layer may use different dimension if loaded from checkpoint
- Add validation in loadCheckpoint tool
- Handle dimension mismatches gracefully (resize or reject)
- Log warnings if dimensions don't match
- Test: Load checkpoint with mismatched dimensions

---

## Phase 10: Real-Time Learning Validation

### 10.1 Create Real-Time Learning Test Suite (src/**tests**/realtime.test.ts)

- Test 1: Surprise-based memory update
  - Initialize model
  - Run forward pass with novel input
  - Verify high surprise score
  - Run train_step
  - Verify memory updated with momentum
  - Run forward pass with same input
  - Verify surprise score decreased (learning occurred)
- Test 2: Token flow learning
  - Feed sequence: A → B → C repeatedly
  - Verify token flow weights increase for sequence
  - Test prediction: Given A → B, predict C with high confidence
- Test 3: Hierarchical memory promotion
  - Access same memory item 100 times
  - Verify promotion from working → short-term → long-term
  - Verify retrieval speed improves over time

### 10.2 Integration Test: End-to-End Learning (src/**tests**/integration.test.ts)

- Scenario: Learn code patterns in real-time
- Initialize model
- Feed 10 code snippets sequentially
- For each snippet:
  - Forward pass (measure surprise)
  - Train step (update memory with momentum)
  - Verify memory state changes
  - Verify token flow tracks sequence
- Re-feed same 10 snippets
- Verify surprise scores decreased (learning occurred)
- Verify forward pass faster (caching + hierarchical memory)

### 10.3 Surprise-Based Adaptation Test (src/**tests**/adaptation.test.ts)

- Test adaptive forgetting gate:
  - Feed high-surprise input → verify alpha_t increases (more forgetting)
  - Feed low-surprise input → verify alpha_t decreases (less forgetting)
  - Verify memory stability for familiar patterns
  - Verify memory updates for novel patterns
- Test momentum accumulation:
  - Feed consistent pattern 10 times
  - Verify momentum magnitude grows
  - Introduce novel pattern
  - Verify momentum resets/adjusts

### 10.4 Edge Case & Robustness Tests

- Test malformed MCP requests
- Test concurrent tool calls
- Test memory overflow scenarios
- Test long sequence handling (context window limits)
- Test checkpoint corruption recovery
- Test out-of-distribution inputs

### 10.5 Documentation & Validation

- Document real-time learning capabilities in README.md
- Add "Real-Time Learning" section to docs/architecture-overview.md
- Explain surprise-based adaptation in DETAILED_IMPLEMENTATION_GUIDE.md
- Update IMPLEMENTATION_PROGRESS.md: mark Phase 10 complete
- Record demo video showing real-time learning in action
- Commit: "test: comprehensive real-time learning validation"
- **Update this plan document with Phase 10 completion date**

---

## Phase 11: Workflow & Integration Finalization

### 11.1 Workflow Components Decision (src/workflows/)

- **Files:** WorkflowOrchestrator.ts, GitHubWorkflowManager.ts, LintingManager.ts, FeedbackProcessor.ts, WorkflowUtils.ts
- **Current Status:** No usage found in main server, requires credential handling and reliability work
- **Decision Tree:**
  - If integrating: Add credential storage, retry logic, observability
  - If documenting: Create docs/workflows.md explaining intended use cases
  - If removing: Archive to docs/archive/ and update documentation
- Document decision in IMPLEMENTATION_PROGRESS.md

### 11.2 MCP Protocol Compliance Validation

- Verify JSON-RPC 2.0 error codes match spec
- Test with MCP SDK test suite if available
- Validate tool schema format against MCP specification
- Ensure progress notifications for long-running operations
- Test stdio transport compliance
- Document compliance in docs/mcp-compliance.md

### 11.3 Contrastive Learning Completion (Optional)

- **Current Status:** Contrastive loss computation exists (src/model.ts:700-800) but no negative example buffer
- Add negative example buffer management
- Integrate into training loop
- Test contrastive learning improves embeddings
- Document in research extensions

---

## Phase 12: Final Documentation & Release

### 12.1 Update All Documentation Files

- **README.md:**
  - Accurate feature list reflecting all implementations
  - Correct installation (titan-memory not mcp-titan)
  - Usage instructions with real-time learning examples
  - Version to 4.0.0
- **docs/api/README.md:**
  - Complete tool list (15+ tools with all additions)
  - Remove process_input references
  - Add all new tools from phases 2-8
  - Parameter schemas for all tools
  - Response examples
- **DETAILED_IMPLEMENTATION_GUIDE.md:**
  - Mark all sections ✅ IMPLEMENTED
  - Add "Implementation Complete" header
  - Include lessons learned section
  - Reference actual line numbers from implementations
- **IMPLEMENTATION_PROGRESS.md:**
  - Final status: All phases complete
  - Completion date and metrics
  - Performance benchmarks
  - Test coverage report (target >80%)

### 12.2 Create Migration Guide (docs/MIGRATION.md)

- Document breaking changes from pre-Titans version
- Provide checkpoint migration scripts if needed
- Explain new configuration parameters
- Include upgrade checklist
- Backward compatibility notes

### 12.3 Update Architecture Diagrams (docs/architecture-overview.md)

- Create diagram showing momentum flow (Equations 32-33)
- Create diagram showing token flow integration
- Create diagram showing hierarchical memory tiers (working/short/long)
- Create diagram showing real-time learning loop (surprise → update → adapt)
- Add workflow components if integrated
- Use Mermaid or similar for maintainability

### 12.4 Create Release Notes (RELEASE_NOTES.md)

- Version 4.0.0 (Titans Implementation)
- **New features:**
  - Momentum-based memory updates (Equations 32-33)
  - Token flow tracking and surprise weighting
  - Forgetting gate with adaptive decay
  - Deep neural memory module
  - Hierarchical memory (working/short-term/long-term)
  - Structured logging and health checks
  - Security hardening (encryption, auth, rate limiting)
  - Performance optimizations and caching
  - Real-time learning validated
- Breaking changes
- Performance improvements (benchmark numbers)
- Bug fixes (list specific issues resolved)
- Migration instructions
- Thank contributors

### 12.5 Prepare for NPM Publish

- Update package.json version to 4.0.0
- Verify package.json metadata (author, license, keywords)
- Update .npmignore to exclude test files and docs
- Test package locally: npm pack, install from tarball
- Verify binary works: npx titan-memory --help
- Document publish steps in CONTRIBUTING.md
- **Memory:** Publish to @henryhawke/mcp-titan (not mcp-titan alone)

### 12.6 Final Validation Checklist

- [ ] All test suites pass (unit + integration)
- [ ] Test coverage >80%
- [ ] No TypeScript errors (fix src/model.ts unused vars, type tokenizer)
- [ ] No linter warnings
- [ ] Documentation 100% current (no tool count discrepancies)
- [ ] Real-time learning validated
- [ ] Performance benchmarks meet targets (<100ms 95th percentile)
- [ ] Security audit passed (OWASP top 10)
- [ ] Backward compatibility maintained (or documented)
- [ ] npm publish dry-run succeeds
- [ ] All phases marked complete in IMPLEMENTATION_PROGRESS.md
- [ ] This plan document updated with all completion dates

### 12.7 Commit and Tag

- Final commit: "docs: complete Titans implementation documentation"
- Create git tag: v4.0.0
- Push tag to origin
- Create GitHub release with release notes
- Publish to npm: npm publish --scope @henryhawke
- Announce release

---

## Success Criteria (Complete Checklist)

After completing all phases, the MCP server MUST have:

### Research Paper Alignment

- [x] Momentum-based updates (Equations 32-33, lines 426-489)
- [x] Token flow tracking (Section 3.1, lines 364-366)
- [x] Forgetting gate (lines 472-476, weight decay equivalent)
- [x] Deep neural memory module (lines 450-452)
- [x] Non-linear recurrence (inter-chunk non-linear, intra-chunk linear)
- [x] Hierarchical memory (lines 381-386)

### Production Features

- [x] Health check endpoint with diagnostics
- [x] Structured logging with rotation (file + JSON)
- [x] Input validation and sanitization
- [x] Path traversal protection
- [x] Checkpoint encryption at rest (AES-256-GCM)
- [x] API authentication and rate limiting

### Performance

- [x] <100ms tool response time (95th percentile)
- [x] No redundant forward passes
- [x] In-place tensor operations where safe
- [x] LRU cache for read-only operations
- [x] Response compression for large payloads
- [x] Efficient tensor memory management

### Code Quality

- [x] >80% test coverage
- [x] No TypeScript compilation errors
- [x] No linter warnings
- [x] Consistent error handling across all tools
- [x] Type safety (no `any` types)
- [x] Memory leak protection

### Documentation

- [x] 100% accurate (no outdated tool lists)
- [x] All tools documented with examples
- [x] Migration guide from v3 to v4
- [x] Architecture diagrams current
- [x] Release notes comprehensive

### Testing

- [x] Unit tests for all new features
- [x] Integration tests (init → forward → train → save → load)
- [x] Real-time learning validated
- [x] Edge case coverage (malformed requests, overflow, concurrent calls)
- [x] Security tests (injection, traversal, auth)

### Specific Bugs Fixed

- [x] manifold_step implemented or removed from help text
- [x] MockTokenizer replaced with AdvancedTokenizer
- [x] Tool count discrepancy resolved (15 not 16)
- [x] forward_pass type guard added
- [x] Checkpoint dimension validation added
- [x] AutoSave silent failure fixed
- [x] Hierarchical memory promotion wired (not just placeholders)
- [x] Quantization enabled or documented as future work

### Compliance

- [x] MCP protocol JSON-RPC 2.0 compliant
- [x] Stdio transport validated
- [x] Tool schemas match specification
- [x] OWASP Top 10 security checklist passed

---

## Metrics & Targets

| Metric | Current Baseline | Target | Phase |

|--------|------------------|--------|-------|

| Test Coverage | Unknown | >80% | 10, 12 |

| Tool Response Time (95th) | Unknown | <100ms | 8 |

| Tensor Count (degraded threshold) | Variable | <1000 | 6, 8 |

| Memory Usage | Variable | <500MB | 8 |

| Checkpoint Size (uncompressed) | Variable | Document | 8 |

| Surprise Score Decrease (after learning) | N/A | >30% reduction | 10 |

| Documentation Accuracy | ~70% | 100% | 0, 12 |

| Security Issues | Multiple | 0 critical | 7 |

---

## Sequential To-Do List (200+ items)

### Phase 0: Documentation (6 tasks)

- [ ] phase0-1: Read all .md files, create dependency map, identify contradictions
- [ ] phase0-2: Merge AUDIT_FINDINGS + AUDIT_IMPLEMENTATION_SUMMARY + mcp-titan-system-audit.md → SYSTEM_AUDIT.md
- [ ] phase0-3: Consolidate COMPLETE_IMPLEMENTATION_PACKAGE + GUIDE_UPDATE_SUMMARY → IMPLEMENTATION_PACKAGE.md
- [ ] phase0-4: Archive obsolete docs to docs/archive/
- [ ] phase0-5: Update IMPLEMENTATION_PROGRESS.md with phase 0-12 checkboxes and metrics
- [ ] phase0-6: Update this plan with Phase 0 completion date

### Phase 1: Baseline (8 tasks)

- [ ] phase1-1: Document all 15 tools from src/index.ts with signatures and issues
- [ ] phase1-2: Map src/model.ts implementations vs stubs (note lines 1800-2000, 251-276, 2890+)
- [ ] phase1-3: Catalog src/types.ts IMemoryState fields and missing Titans fields
- [ ] phase1-4: Understand src/learner.ts replay buffer, note mock tokenizer at src/index.ts:706
- [ ] phase1-5: Run npm test, capture baseline pass/fail, identify gaps
- [ ] phase1-6: Document test gaps and edge cases in IMPLEMENTATION_PROGRESS.md
- [ ] phase1-7: Verify package.json dependencies, check TypeScript config
- [ ] phase1-8: Update this plan with Phase 1 completion date

### Phase 2: Momentum & Forgetting (21 tasks)

- [ ] phase2-1: Add momentumState, momentumDecay, forgettingGate, forgettingGateHistory to IMemoryState (src/types.ts)
- [ ] phase2-2: Add Zod validation for momentum/forgetting in TitanMemoryConfigSchema (0-1 range)
- [ ] phase2-3: Write types.test.ts validating schema edge cases
- [ ] phase2-4: Implement computeMomentumUpdate() in src/model.ts per Equation 33
- [ ] phase2-5: Implement applyMomentumToMemory() per Equation 32
- [ ] phase2-6: Write momentum.test.ts for both methods
- [ ] phase2-7: Implement updateForgettingGate() with adaptive alpha_t
- [ ] phase2-8: Implement applyForgettingGate() with broadcasting
- [ ] phase2-9: Write forgetting.test.ts for adaptive behavior
- [ ] phase2-10: Integrate momentum into trainStep (src/model.ts:1800-2000)
- [ ] phase2-11: Update src/persistence.ts checkpoint serialization for momentum
- [ ] phase2-12: Write persistence.test.ts for momentum save/load
- [ ] phase2-13: Extend get_memory_state tool with momentum stats
- [ ] phase2-14: Write integration test for get_memory_state after training
- [ ] phase2-15: Update docs/api/README.md with new get_memory_state fields
- [ ] phase2-16: Run full test suite, verify all pass
- [ ] phase2-17: Manual test: init_model → train_step(5x) → get_memory_state
- [ ] phase2-18: Update DETAILED_IMPLEMENTATION_GUIDE.md Section 1 with ✅
- [ ] phase2-19: Update IMPLEMENTATION_PROGRESS.md Phase 2 complete with metrics
- [ ] phase2-20: Commit: "feat: implement Titans momentum and forgetting (Eqs 32-33)"
- [ ] phase2-21: Update this plan with Phase 2 completion date

### Phase 3: Token Flow (20 tasks)

- [ ] phase3-1: Add tokenFlowHistory, tokenFlowWeights, windowSize, decay to IMemoryState
- [ ] phase3-2: Update checkpoint schema for token flow
- [ ] phase3-3: Validate schema accepts token flow fields
- [ ] phase3-4: Implement updateTokenFlow() in src/model.ts
- [ ] phase3-5: Implement computeTokenFlowWeights() with recency + similarity
- [ ] phase3-6: Initialize token flow in constructor
- [ ] phase3-7: Integrate into forward() method before memory query
- [ ] phase3-8: Write tokenflow.test.ts
- [ ] phase3-9: Implement weightSurpriseByTokenFlow()
- [ ] phase3-10: Integrate weighted surprise into trainStep
- [ ] phase3-11: Test surprise differs with/without token flow
- [ ] phase3-12: Register get_token_flow_metrics tool in src/index.ts
- [ ] phase3-13: Add Zod schema for get_token_flow_metrics
- [ ] phase3-14: Write integration test for token flow tool
- [ ] phase3-15: Update docs/api/README.md with get_token_flow_metrics
- [ ] phase3-16: Test token flow improves sequential prediction
- [ ] phase3-17: Update DETAILED_IMPLEMENTATION_GUIDE.md Section 2 with ✅
- [ ] phase3-18: Update IMPLEMENTATION_PROGRESS.md Phase 3 complete
- [ ] phase3-19: Commit: "feat: implement token flow tracking"
- [ ] phase3-20: Update this plan with Phase 3 completion date

### Phase 4: Deep Memory (16 tasks)

- [ ] phase4-1: Create DeepMemoryNetwork class in src/model.ts
- [ ] phase4-2: Implement 3-layer MLP with skip connections (SiLU, dropout)
- [ ] phase4-3: Add useDeepMemory, deepMemoryLayers, deepMemoryHiddenDim to config
- [ ] phase4-4: Initialize DeepMemoryNetwork in TitanMemoryModel.initialize()
- [ ] phase4-5: Integrate deep memory into forward() (conditional path)
- [ ] phase4-6: Test forward with/without deep memory (same outputs)
- [ ] phase4-7: Integrate deep memory into trainStep() (backprop through network)
- [ ] phase4-8: Test deep memory weights change during training
- [ ] phase4-9: Extend get_memory_state with deepMemoryStats
- [ ] phase4-10: Test tool with deep memory enabled
- [ ] phase4-11: Benchmark perplexity with/without deep memory
- [ ] phase4-12: Add deep memory diagram to docs/architecture-overview.md
- [ ] phase4-13: Update DETAILED_IMPLEMENTATION_GUIDE.md Section 4 with ✅
- [ ] phase4-14: Update IMPLEMENTATION_PROGRESS.md Phase 4 complete
- [ ] phase4-15: Commit: "feat: implement deep neural memory module"
- [ ] phase4-16: Update this plan with Phase 4 completion date

### Phase 5: Hierarchical Memory (21 tasks)

- [ ] phase5-1: Add MemoryTier enum to src/types.ts
- [ ] phase5-2: Add working/shortTerm/longTerm Memory tensors and tracking maps to IMemoryState
- [ ] phase5-3: Add workingCapacity, shortTermCapacity, longTermCapacity to config
- [ ] phase5-4: Implement evaluatePromotion() in src/model.ts
- [ ] phase5-5: Implement promoteToShortTerm()
- [ ] phase5-6: Implement promoteToLongTerm() with optional quantization
- [ ] phase5-7: Implement evaluateDemotion()
- [ ] phase5-8: Implement applyDemotion() with eviction
- [ ] phase5-9: Integrate hierarchical memory into forward() (query all tiers)
- [ ] phase5-10: Test items retrieved from correct tiers
- [ ] phase5-11: Integrate promotions into trainStep (after each step)
- [ ] phase5-12: Apply tier-specific learning rates (1.0, 0.5, 0.1)
- [ ] phase5-13: Test items promote over multiple training steps
- [ ] phase5-14: Register get_hierarchical_memory_stats tool
- [ ] phase5-15: Test tool returns correct utilization stats
- [ ] phase5-16: Run 1000 training steps, verify long-term promotion
- [ ] phase5-17: Add hierarchical memory diagram to docs/architecture-overview.md
- [ ] phase5-18: Update DETAILED_IMPLEMENTATION_GUIDE.md Section 5 with ✅
- [ ] phase5-19: Update IMPLEMENTATION_PROGRESS.md Phase 5 complete
- [ ] phase5-20: Commit: "feat: implement hierarchical memory"
- [ ] phase5-21: Update this plan with Phase 5 completion date

### Phase 6: Observability (18 tasks)

- [ ] phase6-1: Create Logger class in src/logging.ts (DEBUG, INFO, WARN, ERROR)
- [ ] phase6-2: Implement console, file, JSON transports
- [ ] phase6-3: Add log rotation (10MB max, 5 files)
- [ ] phase6-4: Write logging.test.ts
- [ ] phase6-5: Replace console.* in src/model.ts with logger
- [ ] phase6-6: Add logging to init, forward, train, promotion, pruning, checkpoint
- [ ] phase6-7: Replace console.* in src/index.ts with logger
- [ ] phase6-8: Add logging to tool invocations with sanitization
- [ ] phase6-9: Register health_check tool
- [ ] phase6-10: Implement health check schema (status, uptime, memory, tensors, errors, latency)
- [ ] phase6-11: Test health_check tool
- [ ] phase6-12: Add optional HTTP endpoints (/health, /healthz, /readyz, /metrics)
- [ ] phase6-13: Test HTTP endpoints if enabled
- [ ] phase6-14: Create docs/observability.md
- [ ] phase6-15: Update DETAILED_IMPLEMENTATION_GUIDE.md Section 6 with ✅
- [ ] phase6-16: Update IMPLEMENTATION_PROGRESS.md Phase 6 complete
- [ ] phase6-17: Commit: "feat: add structured logging and health checks"
- [ ] phase6-18: Update this plan with Phase 6 completion date

### Phase 7: Security (22 tasks)

- [ ] phase7-1: Create sanitizeInput() utility (length, control chars, UTF-8, injection)
- [ ] phase7-2: Create validateFilePath() utility (traversal, whitelist)
- [ ] phase7-3: Apply sanitization to all tool inputs
- [ ] phase7-4: Write security.test.ts with malicious inputs
- [ ] phase7-5: Add encryptCheckpoint boolean to config (default true)
- [ ] phase7-6: Implement AES-256-GCM encryption with PBKDF2 key derivation
- [ ] phase7-7: Modify save_checkpoint to encrypt if enabled
- [ ] phase7-8: Modify load_checkpoint to detect and decrypt
- [ ] phase7-9: Test encrypted checkpoint save/load cycle
- [ ] phase7-10: Add requireAuth and apiKey to config
- [ ] phase7-11: Create authenticateRequest() middleware (constant-time comparison)
- [ ] phase7-12: Test authentication with correct/incorrect/missing keys
- [ ] phase7-13: Create RateLimiter class (sliding window, 100/min default)
- [ ] phase7-14: Apply rate limiting to all tool handlers
- [ ] phase7-15: Test rate limiting (verify 101st request blocked)
- [ ] phase7-16: Create sanitizeOutput() utility (PII redaction, size limit)
- [ ] phase7-17: Test PII redaction in responses
- [ ] phase7-18: Create docs/security.md
- [ ] phase7-19: Update DETAILED_IMPLEMENTATION_GUIDE.md Section 7 with ✅
- [ ] phase7-20: Update IMPLEMENTATION_PROGRESS.md Phase 7 complete
- [ ] phase7-21: Commit: "feat: implement security hardening"
- [ ] phase7-22: Update this plan with Phase 7 completion date

### Phase 8: Performance (21 tasks)

- [ ] phase8-1: Audit trainStep for duplicate forward calls
- [ ] phase8-2: Implement forward pass result caching
- [ ] phase8-3: Test training converges with caching
- [ ] phase8-4: Replace tensor ops with in-place variants (addInPlace, mulInPlace)
- [ ] phase8-5: Test numerical results unchanged, measure memory reduction
- [ ] phase8-6: Create LRUCache class in src/cache.ts (max 100, eviction)
- [ ] phase8-7: Add TTL support (5 min default)
- [ ] phase8-8: Wrap get_memory_state with caching
- [ ] phase8-9: Wrap get_token_flow_metrics with caching
- [ ] phase8-10: Wrap get_hierarchical_memory_stats, analyze_memory, get_surprise_metrics
- [ ] phase8-11: Implement cache invalidation on train_step, prune_memory
- [ ] phase8-12: Test cache hit/miss behavior
- [ ] phase8-13: Add response compression (gzip + base64) for >10KB responses
- [ ] phase8-14: Test compressed responses 50%+ smaller
- [ ] phase8-15: Review transformer attention (6 layers), document optimization opportunities
- [ ] phase8-16: Run performance benchmarks before/after (target <100ms 95th percentile)
- [ ] phase8-17: Create docs/performance.md
- [ ] phase8-18: Update DETAILED_IMPLEMENTATION_GUIDE.md Section 8 with ✅
- [ ] phase8-19: Update IMPLEMENTATION_PROGRESS.md Phase 8 complete with benchmark numbers
- [ ] phase8-20: Commit: "perf: optimize tensor ops, add caching"
- [ ] phase8-21: Update this plan with Phase 8 completion date

### Phase 9: Tokenizer (3 tasks)

- [ ] phase9-1: Wire up BPE training, create train_tokenizer tool, persist vocab
- [ ] phase9-2: Replace mock tokenizer with AdvancedTokenizer (src/index.ts:706, src/learner.ts)
- [ ] phase9-3: Add embedding dimension validation in loadCheckpoint

### Phase 10: Real-Time Learning (15 tasks)

- [ ] phase10-1: Create src/**tests**/realtime.test.ts
- [ ] phase10-2: Write Test 1: Surprise-based memory update (novel → high surprise → train → same → low surprise)
- [ ] phase10-3: Write Test 2: Token flow learning (A→B→C sequence, predict C given A→B)
- [ ] phase10-4: Write Test 3: Hierarchical promotion (100 accesses → promotion to long-term)
- [ ] phase10-5: Create src/**tests**/integration.test.ts
- [ ] phase10-6: Write end-to-end learning scenario (10 snippets, re-feed, verify learning)
- [ ] phase10-7: Create src/**tests**/adaptation.test.ts
- [ ] phase10-8: Test adaptive forgetting gate (high surprise → increase alpha, low → decrease)
- [ ] phase10-9: Test momentum accumulation (consistent pattern → momentum grows)
- [ ] phase10-10: Add real-time learning section to README.md
- [ ] phase10-11: Add real-time learning to docs/architecture-overview.md
- [ ] phase10-12: Explain surprise-based adaptation in DETAILED_IMPLEMENTATION_GUIDE.md
- [ ] phase10-13: Update IMPLEMENTATION_PROGRESS.md Phase 10 complete
- [ ] phase10-14: Commit: "test: comprehensive real-time learning validation"
- [ ] phase10-15: Update this plan with Phase 10 completion date

### Phase 11: Workflow (2 tasks)

- [ ] phase11-1: Decide fate of src/workflows/ (integrate, document, or remove)
- [ ] phase11-2: MCP protocol compliance validation (JSON-RPC 2.0, schemas, stdio)

### Phase 12: Release (21 tasks)

- [ ] phase12-1: Update README.md (features, installation titan-memory, v4.0.0)
- [ ] phase12-2: Update docs/api/README.md (15+ tools, remove process_input, add new tools)
- [ ] phase12-3: Mark all DETAILED_IMPLEMENTATION_GUIDE.md sections ✅, add lessons learned
- [ ] phase12-4: Update IMPLEMENTATION_PROGRESS.md final status with all metrics
- [ ] phase12-5: Create docs/MIGRATION.md (v3→v4, breaking changes, checkpoint migration)
- [ ] phase12-6: Update docs/architecture-overview.md with all diagrams (momentum, token flow, hierarchical, learning loop)
- [ ] phase12-7: Create RELEASE_NOTES.md v4.0.0 (features, breaking changes, benchmarks)
- [ ] phase12-8: Update package.json version to 4.0.0
- [ ] phase12-9: Update .npmignore (exclude tests, docs)
- [ ] phase12-10: Test npm pack, install from tarball, verify npx titan-memory --help
- [ ] phase12-11: Verify all test suites pass (unit + integration + real-time)
- [ ] phase12-12: Verify test coverage >80% (run coverage report)
- [ ] phase12-13: Verify no TypeScript errors (fix unused vars, type tokenizer)
- [ ] phase12-14: Verify no linter warnings
- [ ] phase12-15: Verify documentation 100% current (no tool count discrepancies)
- [ ] phase12-16: Commit: "docs: complete Titans implementation"
- [ ] phase12-17: Create git tag v4.0.0
- [ ] phase12-18: Push tag to origin
- [ ] phase12-19: Create GitHub release with RELEASE_NOTES.md
- [ ] phase12-20: Publish: npm publish --scope @henryhawke
- [ ] phase12-21: Update this plan with Phase 12 completion and final retrospective

**Total Tasks: 215**

---

**Next Step:** Begin Phase 0 when ready to execute.