# TypeScript Error Resolution Guide

## Context

You are working on the MCP-Titan HOPE Memory Server. Recent refactoring reduced TypeScript compilation errors from 165 to 42 (75% improvement). The remaining 42 errors are primarily **TensorFlow.js type system strictness issues** where `tf.tidy()` expects `TensorContainer` but custom memory types are returned.

**Current State:**
- ✅ HOPE/TITAN config unification complete
- ✅ IMemoryModel interface fully implemented
- ✅ All missing methods added
- ⚠️ 42 errors remain, categorized as:
  - 20× Property missing errors (TS2339)
  - 17× Type assignment errors (TS2345)
  - 16× Type compatibility errors (TS2322)
  - 2× Wrong argument count (TS2554)
  - 2× Incompatible signatures (TS2416)

**Error Distribution by File:**
```
src/hope_model/continuum_memory.ts  - 10 errors (tf.tidy return types)
src/hope_model/index.ts             - 8 errors (tf.tidy, type conversions)
src/hope_model/retention_core.ts    - 6 errors (tf.tidy return types)
src/hope_model/mamba_filters.ts     - 4 errors (interface mismatches)
src/hope_model/memory_router.ts     - 2 errors (array type issues)
src/index.ts                        - 6 errors (tensor rank, property access)
src/training/trainer.ts             - 4 errors (missing methods)
src/persistence.ts                  - 2 errors (already addressed)
```

## Objective

Achieve **100% TypeScript compilation success** while:
1. Maintaining runtime functionality (all tests must pass)
2. Preserving type safety benefits
3. Avoiding widespread use of `any` or unsafe type assertions
4. Keeping `tf.tidy()` memory management benefits

---

## Phase 0: Analysis & Strategy (2 hours)

**Goal:** Understand the root causes and design a type-safe solution strategy.

### Subtask 0.1: Audit Remaining Errors
**Action:**
```bash
npm run build 2>&1 | grep "error TS" > errors.log
cat errors.log | cut -d'(' -f1 | sort | uniq -c | sort -rn
```

**Deliverable:** Create `docs/typescript-errors-analysis.md` with:
- Error categorization by type code (TS2322, TS2339, etc.)
- File-by-file breakdown with line numbers
- Pattern identification (e.g., "all tf.tidy returns in continuum_memory.ts")

**Acceptance Criteria:**
- [ ] Document lists all 42 errors with file:line references
- [ ] Errors grouped by root cause (tf.tidy, tensor ranks, missing properties, etc.)
- [ ] Each group has a proposed solution strategy

### Subtask 0.2: Design Type Adapter Pattern
**Action:** Create type-safe wrappers for `tf.tidy()` that properly handle custom return types.

**Deliverable:** Create `src/hope_model/type_utils.ts`:
```typescript
import * as tf from '@tensorflow/tfjs-node';

/**
 * Type-safe wrapper for tf.tidy that allows custom return types
 * while preserving memory management benefits
 */
export function tidyMemoryState<T extends Record<string, tf.Tensor | tf.Tensor[] | number>>(
  fn: () => T
): T {
  // Implementation that satisfies TypeScript while maintaining tf.tidy semantics
  return tf.tidy(() => {
    const result = fn();
    // Keep all tensor properties
    Object.entries(result).forEach(([key, value]) => {
      if (value instanceof tf.Tensor || Array.isArray(value)) {
        tf.keep(value as any);
      }
    });
    return result;
  }) as unknown as T;
}

export function tidyTensor2D(fn: () => tf.Tensor): tf.Tensor2D {
  const result = tf.tidy(fn);
  if (result.rank !== 2) {
    throw new Error(`Expected 2D tensor, got rank ${result.rank}`);
  }
  return result as tf.Tensor2D;
}
```

**Acceptance Criteria:**
- [ ] Type utilities compile without errors
- [ ] Utilities have JSDoc explaining purpose and usage
- [ ] Include unit tests showing proper tensor disposal

### Subtask 0.3: Review Existing Tests
**Action:**
```bash
npm test 2>&1 | tee test-results.log
```

**Deliverable:** Document baseline test status in `docs/test-baseline.md`:
- Number of passing tests
- Any failing tests (and reasons)
- Test coverage percentage

**Acceptance Criteria:**
- [ ] All existing tests pass (or failures documented as pre-existing)
- [ ] Test coverage report generated
- [ ] Identified which tests cover tf.tidy memory management

---

## Phase 1: Fix ContinuumMemory (6-8 hours)

**Goal:** Resolve all 10 errors in `src/hope_model/continuum_memory.ts`

### Subtask 1.1: Fix write() Method Return Type
**Target:** `continuum_memory.ts:70,124` (tf.tidy return type errors)

**Action:**
```typescript
// BEFORE (causes TS2322, TS2345):
public write(state: HopeMemoryState, embedding: tf.Tensor2D, metadata: MemoryWriteMetadata): HopeMemoryState {
  return tf.tidy(() => {
    // ... returns HopeMemoryState
  });
}

// AFTER:
public write(state: HopeMemoryState, embedding: tf.Tensor2D, metadata: MemoryWriteMetadata): HopeMemoryState {
  return tidyMemoryState(() => {
    const normalized = this.normalize(embedding);
    const newShort = tf.concat([state.shortTerm, normalized], 0);
    // ... build complete state
    return updated;
  });
}
```

**Test:**
```typescript
// Add to test_hope_model/continuum_memory.test.ts
it('write() properly manages memory and returns correct type', () => {
  const initialTensorCount = tf.memory().numTensors;
  const state = memory.initialize();
  const embedding = tf.randomNormal([1, 256]);

  const newState = memory.write(state, embedding, {
    surprise: 0.5,
    timestamp: Date.now(),
    routeWeights: tf.ones([1, 3])
  });

  expect(newState.shortTerm.rank).toBe(2);
  expect(newState.shortTerm.shape[1]).toBe(256);

  // Verify no memory leak
  embedding.dispose();
  state.shortTerm.dispose();
  // ... dispose all
  const finalTensorCount = tf.memory().numTensors;
  expect(finalTensorCount).toBeLessThanOrEqual(initialTensorCount + 5); // Allow for new state tensors
});
```

**Acceptance Criteria:**
- [ ] `continuum_memory.ts:70` compiles without errors
- [ ] `continuum_memory.ts:124` compiles without errors
- [ ] Test passes showing proper memory management
- [ ] No new tensors leaked (verified with `tf.memory()`)

### Subtask 1.2: Fix read() Method Return Type
**Target:** `continuum_memory.ts:97,103,104` (Tensor2D assignment errors)

**Action:**
```typescript
// Fix the return type and internal operations
public read(state: HopeMemoryState, query: tf.Tensor2D, weights: tf.Tensor2D): tf.Tensor2D {
  return tidyTensor2D(() => {
    const normalizedQuery = this.normalize(query);
    const reads: tf.Tensor2D[] = []; // Explicitly type as Tensor2D[]

    if (state.shortTerm.shape[0] > 0) {
      const shortRead = this.readTier(state.shortTerm, normalizedQuery, weights);
      reads.push(shortRead as tf.Tensor2D); // Type assertion after validation
    }
    // ... similar for longTerm, archive

    if (reads.length === 0) {
      return tf.zeros([1, this.config.memoryDim]) as tf.Tensor2D;
    }

    const combined = tf.concat(reads, 0);
    const result = tf.mean(combined, 0, true);
    return result as tf.Tensor2D; // Safe after mean with keepDims=true
  });
}
```

**Test:**
```typescript
it('read() returns properly typed Tensor2D', () => {
  const state = memory.initialize();
  const query = tf.randomNormal([1, 256]);
  const weights = tf.ones([1, 3]);

  const result = memory.read(state, query, weights);

  expect(result.rank).toBe(2);
  expect(result.shape).toEqual([1, 256]);
  expect(result instanceof tf.Tensor).toBe(true);

  result.dispose();
  query.dispose();
  weights.dispose();
});
```

**Acceptance Criteria:**
- [ ] All read() related errors (97, 103, 104) resolved
- [ ] Return type is explicitly `tf.Tensor2D`
- [ ] Test verifies correct tensor rank and shape
- [ ] No memory leaks

### Subtask 1.3: Fix prune() Method
**Target:** `continuum_memory.ts:162,166` (tf.tidy return, wrong arg count)

**Action:**
```typescript
public prune(state: HopeMemoryState, threshold: number): HopeMemoryState {
  return tidyMemoryState(() => {
    // Get indices of memories to keep
    const surpriseValues = Array.from(state.surpriseHistory.dataSync());
    const keepMask = surpriseValues.map(s => s > threshold);

    // Filter each tier - fix the slice call that expects 3 args
    const keptIndices = keepMask
      .map((keep, idx) => keep ? idx : -1)
      .filter(idx => idx >= 0);

    if (keptIndices.length === 0) {
      return this.initialize(); // Empty state
    }

    // Rebuild tensors with only kept memories
    const newShortTerm = tf.stack(
      keptIndices.map(i => state.shortTerm.slice([i, 0], [1, -1]).squeeze([0]))
    );

    return {
      shortTerm: newShortTerm,
      longTerm: state.longTerm.clone(),
      // ... rest of state
    };
  });
}
```

**Test:**
```typescript
it('prune() correctly filters based on threshold', async () => {
  let state = memory.initialize();

  // Add some memories with varying surprise
  for (let i = 0; i < 10; i++) {
    state = memory.write(state, tf.randomNormal([1, 256]), {
      surprise: i * 0.1, // 0.0, 0.1, 0.2, ..., 0.9
      timestamp: Date.now(),
      routeWeights: tf.ones([1, 3])
    });
  }

  const beforeCount = state.shortTerm.shape[0];
  const pruned = memory.prune(state, 0.5); // Keep only surprise > 0.5
  const afterCount = pruned.shortTerm.shape[0];

  expect(beforeCount).toBe(10);
  expect(afterCount).toBe(5); // Memories with surprise 0.6-0.9

  // Cleanup
  state.shortTerm.dispose();
  pruned.shortTerm.dispose();
});
```

**Acceptance Criteria:**
- [ ] Errors 162 and 166 resolved
- [ ] Pruning logic works correctly (test passes)
- [ ] No slice() argument count errors
- [ ] Memory properly managed

---

## Phase 2: Fix Hope Model Components (6-8 hours)

**Goal:** Resolve errors in `retention_core.ts`, `mamba_filters.ts`, `memory_router.ts`

### Subtask 2.1: Fix RetentionCore tf.tidy Issues
**Target:** `retention_core.ts:61,69,101` (3 tf.tidy return type errors)

**Action:**
```typescript
import { tidyMemoryState } from './type_utils.js';

export interface RetentionState {
  hidden: tf.Tensor2D;
  filter: FilterState;
  steps: number;
}

public initState(batchSize: number): RetentionState {
  return tidyMemoryState(() => ({
    hidden: tf.zeros([batchSize, this.config.hiddenDim]) as tf.Tensor2D,
    filter: this.selectiveFilter.initState(batchSize),
    steps: 0
  }));
}

public forwardSequence(input: tf.Tensor2D, state: RetentionState): SequenceResult {
  return tidyMemoryState(() => {
    // ... implementation
    return {
      outputs: allOutputs as tf.Tensor2D,
      state: finalState,
      gates: allGates as tf.Tensor2D
    };
  });
}
```

**Test:**
```typescript
describe('RetentionCore', () => {
  it('initState creates properly typed state', () => {
    const core = new RetentiveCore(config, filter);
    const state = core.initState(2);

    expect(state.hidden.rank).toBe(2);
    expect(state.hidden.shape).toEqual([2, 192]);
    expect(state.steps).toBe(0);

    state.hidden.dispose();
  });

  it('forwardSequence maintains type safety', () => {
    const input = tf.randomNormal([4, 256]);
    const state = core.initState(1);

    const result = core.forwardSequence(input, state);

    expect(result.outputs.rank).toBe(2);
    expect(result.state.hidden.rank).toBe(2);
    expect(typeof result.state.steps).toBe('number');

    // Cleanup
    input.dispose();
    result.outputs.dispose();
  });
});
```

**Acceptance Criteria:**
- [ ] All 3 retention_core.ts errors resolved
- [ ] Tests verify correct tensor types and shapes
- [ ] Memory properly managed with tf.tidy wrappers

### Subtask 2.2: Fix MambaFilters Interface Mismatches
**Target:** `mamba_filters.ts:51,58,91` (FilterState and FilterStepResult type errors)

**Action:**
```typescript
// Ensure interface matches implementation
export interface FilterState {
  carry: tf.Tensor2D;      // Explicitly Tensor2D, not Tensor<Rank>
  bandwidth: tf.Tensor2D;
}

export interface FilterStepResult {
  state: FilterState;
  output: tf.Tensor2D;     // Explicitly Tensor2D
  retentionGate: tf.Tensor2D;
}

public initState(batchSize: number): FilterState {
  return tidyMemoryState(() => ({
    carry: tf.zeros([batchSize, this.config.hiddenDim]) as tf.Tensor2D,
    bandwidth: tf.ones([batchSize, this.config.hiddenDim]) as tf.Tensor2D
  }));
}

public forwardStep(input: tf.Tensor2D, state: FilterState): FilterStepResult {
  return tidyMemoryState(() => {
    // ... compute new state
    return {
      state: {
        carry: newCarry as tf.Tensor2D,
        bandwidth: newBandwidth as tf.Tensor2D
      },
      output: output as tf.Tensor2D,
      retentionGate: gate as tf.Tensor2D
    };
  });
}
```

**Test:**
```typescript
it('filter state has correct tensor ranks', () => {
  const filter = new SelectiveStateSpace(config);
  const state = filter.initState(2);

  expect(state.carry.rank).toBe(2);
  expect(state.bandwidth.rank).toBe(2);
  expect(state.carry.shape).toEqual([2, config.hiddenDim]);

  state.carry.dispose();
  state.bandwidth.dispose();
});
```

**Acceptance Criteria:**
- [ ] All mamba_filters.ts errors resolved
- [ ] Interface types match implementation
- [ ] Tests verify tensor ranks are enforced

### Subtask 2.3: Fix MemoryRouter Array Type Issue
**Target:** `memory_router.ts:87` (slice property doesn't exist)

**Action:**
```typescript
// The issue is likely with typed arrays
public route(query: tf.Tensor2D, memoryState: HopeMemoryState): RoutingDecision {
  return tf.tidy(() => {
    // ... routing logic

    // Fix: Ensure surpriseData is typed correctly
    const surpriseValues = await state.surpriseHistory.data();
    const surpriseArray = Array.from(surpriseValues); // Convert to regular array
    const recentSurprise = surpriseArray.slice(-10); // Now slice() works

    // Or use TensorFlow operations instead:
    const recentSurpriseTensor = state.surpriseHistory.slice(
      [Math.max(0, state.surpriseHistory.shape[0] - 10)],  // begin
      [-1]                                                   // size (-1 = to end)
    );

    // ...
  });
}
```

**Test:**
```typescript
it('routing handles empty memory gracefully', () => {
  const router = new MemoryRouter(config);
  const emptyState = memory.initialize();
  const query = tf.randomNormal([1, 192]);

  const decision = router.route(query, emptyState);

  expect(decision.weights.rank).toBe(2);
  expect(decision.surprise).toBeGreaterThanOrEqual(0);

  decision.weights.dispose();
  query.dispose();
});
```

**Acceptance Criteria:**
- [ ] Error 87 resolved
- [ ] Array/tensor operations properly typed
- [ ] Test covers edge cases (empty memory)

---

## Phase 3: Fix HopeMemoryModel Index (4-6 hours)

**Goal:** Resolve 8 errors in `src/hope_model/index.ts`

### Subtask 3.1: Fix trainStep Gradient Handling
**Target:** `index.ts:168,197` (NamedVariableMap type errors)

**Action:**
```typescript
public trainStep(x_t: tf.Tensor2D, x_next: tf.Tensor2D, memoryState: IMemoryState): {
  loss: tf.Tensor;
  gradients: IModelGradients;
  memoryUpdate: IMemoryUpdateResult;
} {
  const variableList = this.getTrainableVariables();

  const { value: loss, grads } = tf.variableGrads(() => {
    const { predicted } = this.forward(x_t, memoryState);
    return tf.losses.meanSquaredError(x_next, predicted);
  });

  // Fix: Convert grads to proper format
  const namedGrads: tf.NamedTensorMap = {};
  variableList.forEach((variable, index) => {
    if (grads[variable.name]) {
      namedGrads[variable.name] = grads[variable.name];
    }
  });

  this.optimizer.applyGradients(namedGrads);

  // ...
}
```

**Test:**
```typescript
it('trainStep updates weights correctly', () => {
  const model = new HopeMemoryModel(config);
  const state = model.createInitialState();

  const x_t = tf.randomNormal([1, 256]);
  const x_next = tf.randomNormal([1, 256]);

  const weightsBefore = model.getTrainableVariables()[0].read().arraySync();

  const result = model.trainStep(x_t, x_next, state);

  const weightsAfter = model.getTrainableVariables()[0].read().arraySync();

  expect(result.loss.rank).toBe(0); // Scalar
  expect(weightsBefore).not.toEqual(weightsAfter); // Weights changed

  result.loss.dispose();
  x_t.dispose();
  x_next.dispose();
});
```

**Acceptance Criteria:**
- [ ] Errors 168 and 197 resolved
- [ ] Gradients properly typed for TensorFlow optimizer
- [ ] Test verifies weights actually update

### Subtask 3.2: Fix computeForward Return Type
**Target:** `index.ts:366` (ForwardArtifacts vs TensorContainer)

**Action:**
```typescript
private computeForward(input: tf.Tensor2D, memoryState: HopeMemoryState, updateState: boolean): ForwardArtifacts {
  // Don't use tf.tidy for complex return types - manage manually
  const normalizedInput = this.ensure2d(input);
  const readWeights = this.memoryRouter.route(
    this.retentionState?.hidden ?? normalizedInput,
    memoryState
  );

  const memoryRead = this.continuumMemory.read(memoryState, normalizedInput, readWeights.weights);
  const coreInput = tf.concat([normalizedInput, memoryRead], 1);
  const retentionState = this.retentionState ?? this.retentiveCore.initState(1);
  const { outputs, state } = this.retentiveCore.forwardSequence(coreInput, retentionState);
  const logits = tf.add(tf.matMul(outputs, this.outputKernel), this.outputBias) as tf.Tensor2D;

  let updatedState = memoryState;
  if (updateState) {
    const lastOutput = outputs.slice([outputs.shape[0] - 1, 0], [1, -1]);
    updatedState = this.continuumMemory.write(memoryState, lastOutput as tf.Tensor2D, {
      surprise: readWeights.surprise,
      timestamp: Date.now(),
      routeWeights: readWeights.weights
    });
    this.retentionState = state;
    this.latestMemoryState = updatedState;
  }

  // Keep tensors we're returning
  tf.keep(logits);
  tf.keep(updatedState.shortTerm);
  // ... keep other state tensors

  // Dispose intermediate tensors
  memoryRead.dispose();
  coreInput.dispose();

  return {
    logits,
    memoryState: updatedState,
    retentionState: state,
    decision: readWeights
  };
}
```

**Test:**
```typescript
it('forward pass returns all required artifacts', () => {
  const model = new HopeMemoryModel(config);
  const input = tf.randomNormal([1, 256]);
  const state = model.createInitialState();

  const result = model.forward(input, state);

  expect(result.predicted).toBeDefined();
  expect(result.predicted.rank).toBe(2);
  expect(result.memoryUpdate.newState).toBeDefined();
  expect(result.memoryUpdate.attention).toBeDefined();

  result.predicted.dispose();
  input.dispose();
});
```

**Acceptance Criteria:**
- [ ] Error 366 resolved
- [ ] Manual memory management works correctly
- [ ] No memory leaks (verify with tf.memory())
- [ ] Test passes

---

## Phase 4: Fix Remaining Index.ts Issues (3-4 hours)

**Goal:** Fix 6 errors in `src/index.ts` related to tensor operations

### Subtask 4.1: Fix Tensor Rank Issues
**Target:** `index.ts:530,595,1596` (Tensor to Tensor2D conversions)

**Action:**
```typescript
// Example fix for forward_pass result handling
const result = await this.model.forward(inputTensor as tf.Tensor2D, currentState);

// Ensure result is Tensor2D before using
if (result.predicted.rank !== 2) {
  result.predicted = result.predicted.expandDims(0) as tf.Tensor2D;
}

// For tensor creation from arrays
const inputTensor = tf.tensor2d([normalized]); // Explicitly create Tensor2D
```

**Test:**
```typescript
it('handles various input shapes correctly', async () => {
  // Test 1D array input
  const result1 = await server.forwardPass([1, 2, 3, /* ... 256 values */]);
  expect(result1).toBeDefined();

  // Test 2D array input
  const result2 = await server.forwardPass([[1, 2, 3, /* ... */]]);
  expect(result2).toBeDefined();

  // Results should be consistent
  expect(result1.length).toBe(result2.length);
});
```

**Acceptance Criteria:**
- [ ] All tensor rank errors resolved
- [ ] Input handling is robust (1D, 2D inputs both work)
- [ ] Tests verify various input formats

### Subtask 4.2: Fix Memory Info Property
**Target:** `index.ts:1540` (numBytesInGPU doesn't exist)

**Action:**
```typescript
// The TensorFlow.js MemoryInfo type doesn't have numBytesInGPU in Node backend
const memInfo = tf.memory();

// Only use properties that exist
const memoryUsage = {
  numTensors: memInfo.numTensors,
  numDataBuffers: memInfo.numDataBuffers,
  numBytes: memInfo.numBytes,
  // Don't use numBytesInGPU - it's only available in WebGL backend
  // numBytesInGPU: memInfo.numBytesInGPU || 0  // Remove this line
};
```

**Test:**
```typescript
it('health_check returns valid memory info', async () => {
  const health = await server.healthCheck();

  expect(health.memory).toBeDefined();
  expect(health.memory.numTensors).toBeGreaterThanOrEqual(0);
  expect(health.memory.numBytes).toBeGreaterThanOrEqual(0);
  expect(health.memory).not.toHaveProperty('numBytesInGPU'); // Not in Node backend
});
```

**Acceptance Criteria:**
- [ ] Error 1540 resolved
- [ ] Health check works correctly
- [ ] Only uses properties available in tfjs-node

---

## Phase 5: Fix Training & Persistence (2-3 hours)

**Goal:** Resolve remaining errors in `training/trainer.ts` and `persistence.ts`

### Subtask 5.1: Update Trainer for HopeMemoryModel
**Target:** `trainer.ts:366,439,504,530` (missing methods on HopeMemoryModel)

**Action:**
```typescript
// These methods were added in Phase 0, but trainer may be using old signatures

// Update trainer.ts imports
import type { HopeMemoryModel } from '../hope_model/index.js';

// Fix method calls
const memoryState = this.model.getMemoryState(); // Now exists
await this.model.saveModel(modelPath);           // Now exists (alias for save)

// If trainer expects different return types, adapt:
const memState = this.model.getMemoryState() as IMemoryState;
```

**Test:**
```typescript
describe('Trainer with HopeModel', () => {
  it('can train model for multiple epochs', async () => {
    const trainer = new Trainer(model, config);
    const dataset = generateSyntheticData(100);

    const initialLoss = await trainer.evaluate(dataset.slice(0, 10));
    await trainer.train(dataset, { epochs: 5 });
    const finalLoss = await trainer.evaluate(dataset.slice(0, 10));

    expect(finalLoss).toBeLessThan(initialLoss); // Model improved
  });

  it('can save and restore model state', async () => {
    await trainer.saveCheckpoint('/tmp/test-checkpoint');

    const newModel = new HopeMemoryModel(config);
    await newModel.loadModel('/tmp/test-checkpoint');

    // Verify model works after loading
    const result = newModel.forward(testInput, testState);
    expect(result.predicted).toBeDefined();
  });
});
```

**Acceptance Criteria:**
- [ ] All trainer.ts errors resolved
- [ ] Training loop works end-to-end
- [ ] Model can save and load successfully

### Subtask 5.2: Verify Persistence Already Fixed
**Target:** `persistence.ts:472,541,542` (should be resolved from Phase 0 config updates)

**Action:**
```bash
# Verify these errors are gone after config updates
npm run build 2>&1 | grep "persistence.ts"

# If still present, they're using old config properties
# Update to use new compatibility fields added in Phase 0
```

**Acceptance Criteria:**
- [ ] Confirm persistence.ts errors no longer appear
- [ ] If they do, update to use `memorySlots` and `transformerLayers` from updated config

---

## Phase 6: Verification & Testing (3-4 hours)

**Goal:** Achieve 100% TypeScript compilation and verify all functionality

### Subtask 6.1: Clean Build Verification
**Action:**
```bash
npm run clean
npm install
npm run build
```

**Expected Output:**
```
> @henryhawke/mcp-titan@3.0.0 build
> tsc

# Should complete with ZERO errors
```

**Acceptance Criteria:**
- [ ] TypeScript compilation completes with **0 errors**
- [ ] No warnings related to fixed code (other warnings acceptable)
- [ ] Build artifacts created in `dist/`

### Subtask 6.2: Full Test Suite
**Action:**
```bash
npm test -- --coverage
```

**Acceptance Criteria:**
- [ ] All existing tests pass
- [ ] All new tests added during fixes pass
- [ ] Test coverage > 80% for modified files
- [ ] No memory leaks detected in tests

### Subtask 6.3: Runtime Verification
**Action:** Create integration test script:
```typescript
// test/integration/full-workflow.test.ts
describe('Full HOPE Workflow', () => {
  it('completes full memory lifecycle', async () => {
    const server = new HopeMemoryServer();
    await server.start();

    // 1. Initialize
    const initResult = await server.initModel({ inputDim: 256 });
    expect(initResult.status).toBe('success');

    // 2. Store memories
    await server.bootstrapMemory({ text: 'Test memory content' });

    // 3. Forward pass
    const output = await server.forwardPass({ input: [/* ... */] });
    expect(output).toBeDefined();

    // 4. Train
    const trainResult = await server.trainStep({
      x_t: [/* ... */],
      x_next: [/* ... */]
    });
    expect(trainResult.loss).toBeGreaterThan(0);

    // 5. Prune
    await server.pruneMemory({ threshold: 0.1 });

    // 6. Save & Load
    await server.saveCheckpoint({ path: '/tmp/test-checkpoint.json' });
    await server.loadCheckpoint({ path: '/tmp/test-checkpoint.json' });

    // 7. Verify memory persisted
    const state = await server.getMemoryState();
    expect(state.shortTermSize).toBeGreaterThan(0);

    await server.stop();
  });
});
```

**Acceptance Criteria:**
- [ ] Integration test passes
- [ ] All MCP tools work correctly
- [ ] Memory is properly managed (no leaks)
- [ ] State can be saved and restored

### Subtask 6.4: Performance Baseline
**Action:**
```typescript
// benchmark/hope-performance.ts
const iterations = 1000;
const start = Date.now();

for (let i = 0; i < iterations; i++) {
  const result = model.forward(testInput, state);
  result.predicted.dispose();
}

const elapsed = Date.now() - start;
console.log(`Forward pass: ${(elapsed / iterations).toFixed(2)}ms per iteration`);
console.log(`Memory: ${JSON.stringify(tf.memory())}`);
```

**Acceptance Criteria:**
- [ ] Baseline performance metrics documented
- [ ] No performance regression vs previous working version
- [ ] Memory usage stable (no gradual increase over iterations)

---

## Phase 7: Documentation & Cleanup (2-3 hours)

### Subtask 7.1: Update Documentation
**Action:** Update the following files:

**`docs/typescript-fixes.md`** (new file):
```markdown
# TypeScript Error Resolution Summary

## Overview
This document summarizes the resolution of 42 TypeScript compilation errors in the HOPE architecture implementation.

## Changes Made

### Type System Enhancements
- Created `type_utils.ts` with type-safe tf.tidy wrappers
- Added explicit Tensor2D types throughout
- Unified HopeMemoryConfig across all files

### Fixed Files
1. `continuum_memory.ts` - 10 errors (tf.tidy wrappers)
2. `retention_core.ts` - 6 errors (type-safe state management)
3. `mamba_filters.ts` - 4 errors (interface alignment)
...

### Testing
- Added 15 new tests for type safety
- All tests pass with 0 memory leaks
- Coverage increased from X% to Y%
```

**`README.md`** - Update build status:
```markdown
## Build Status
✅ TypeScript compilation: **0 errors**
✅ Tests: **X/X passing**
✅ Coverage: **XX%**
```

**Acceptance Criteria:**
- [ ] All documentation updated
- [ ] Changelog entry added
- [ ] README reflects current status

### Subtask 7.2: Code Cleanup
**Action:**
- Remove any commented-out code added during debugging
- Remove `errors.log` and `test-results.log`
- Update `.gitignore` if needed

**Acceptance Criteria:**
- [ ] No debug console.logs in production code
- [ ] No commented-out code blocks
- [ ] Clean git status

### Subtask 7.3: Final Commit
**Action:**
```bash
git add -A
git commit -m "Fix: Achieve 100% TypeScript compilation (0 errors)

Resolved all 42 remaining type errors through systematic fixes:

- Created type_utils.ts for type-safe tf.tidy wrappers
- Fixed all continuum_memory.ts tf.tidy return types (10 errors)
- Fixed retention_core.ts state management (6 errors)
- Fixed mamba_filters.ts interface mismatches (4 errors)
- Fixed memory_router.ts array operations (2 errors)
- Fixed index.ts tensor rank issues (6 errors)
- Fixed trainer.ts method calls (4 errors)
- Updated persistence.ts config usage (2 errors)

All tests passing with 0 memory leaks.
Test coverage: XX%

BREAKING: None - all changes are internal type improvements
TESTED: Full integration test suite + new type safety tests"

git push
```

**Acceptance Criteria:**
- [ ] Commit message follows conventional commits format
- [ ] All changes properly staged
- [ ] Push successful

---

## Success Criteria (Must All Pass)

- [ ] **Zero TypeScript compilation errors** (`npm run build` succeeds)
- [ ] **All tests pass** (`npm test` shows 100% pass rate)
- [ ] **No memory leaks** (verified with `tf.memory()` before/after tests)
- [ ] **No performance regression** (benchmark shows similar or better performance)
- [ ] **Type safety maintained** (no `any` types except where absolutely necessary)
- [ ] **Documentation complete** (all fixes documented, README updated)
- [ ] **Git history clean** (clear commit messages, no WIP commits in final PR)

---

## Rollback Plan

If any phase fails catastrophically:

```bash
# Rollback to last known good state
git reset --hard 72485a3  # Last successful commit
npm install
npm run build

# Document the issue
echo "Phase X failed due to: [reason]" >> rollback-log.txt

# Re-evaluate approach for that phase
```

---

## Time Estimate

- **Phase 0:** 2 hours (Analysis)
- **Phase 1:** 8 hours (ContinuumMemory fixes)
- **Phase 2:** 8 hours (Component fixes)
- **Phase 3:** 6 hours (HopeModel index)
- **Phase 4:** 4 hours (Remaining index.ts)
- **Phase 5:** 3 hours (Training & persistence)
- **Phase 6:** 4 hours (Verification)
- **Phase 7:** 3 hours (Documentation)

**Total: 38-40 hours (~5-6 working days)**

---

## Notes for Agent

- **Maintain backward compatibility** - all existing code should continue to work
- **Prioritize correctness over speed** - these are type system fundamentals
- **Test after each file** - don't accumulate errors
- **Use `tf.tidy()` alternatives** - type-safe wrappers are essential
- **Document as you go** - future you (and others) will thank you

Good luck! You've got this. 🚀
