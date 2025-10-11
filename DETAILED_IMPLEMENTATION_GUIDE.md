# MCP-Titan Detailed Implementation Guide
**Version:** 3.0.0  
**Date:** October 2025  
**Status:** Remaining To-Dos with Full Context

---

## 🎯 Research Paper Foundation

This guide implements the complete **Titans (Training at Test Time with Attention for Neural Memory Systems)** architecture as described in the research paper. All implementations are backed by specific equations, theorems, and architectural decisions from the paper.

### Core Titans Concepts (from research_paper_source.md)

**What Makes Titans Unique (lines 381-386, 447-456):**

1. **Neural Memory Module (not matrix-valued)** - Expressive power to compress complex information
2. **Forgetting Mechanism** - Prevents fast memory overflow  
3. **Token Flow Tracking** - Beyond momentary surprise to capture information flow
4. **Momentum-Based Updates** - Combines past and momentary surprise
5. **Deep Memory** - Multi-layer neural memory vs shallow linear memory
6. **Non-Linear Recurrence** - Inter-chunk non-linear, intra-chunk linear

**Key Differentiators from Prior Work (lines 381-404):**
- vs RMT: Neural memory module instead of vector-valued small size memory
- vs DeltaNet/Gated DeltaNet: Momentum-based rule + deep memory + non-linear recurrence + forget gate
- vs Longhorn: Forgetting gate for better memory management
- vs TTT Layers: Forgetting mechanism + momentum-based updates

---

## Table of Contents

1. [Momentum Integration (Priority 1)](#1-momentum-integration) - **Research Paper: Appendix C, Equations 32-33**
2. [Token Flow Integration (Priority 1)](#2-token-flow-integration) - **Research Paper: Section 3.1, Lines 364-366**
3. [Forgetting Gate (Priority 1)](#3-forgetting-gate-implementation) - **Research Paper: Lines 472-476**
4. [Deep Neural Memory (Priority 2)](#4-deep-neural-memory-module) - **Research Paper: Lines 22-26, 450-452**
5. [Hierarchical Memory (Priority 2)](#5-hierarchical-memory-activation) - **Research Paper: Lines 381-386**
6. [Health Check Endpoint (Priority 1)](#6-health-check-endpoint)
7. [Structured Logging System (Priority 1)](#7-structured-logging-system)
8. [Performance Optimization (Priority 2)](#8-performance-optimization) - **Research Paper: Lines 256-270**
9. [Workflow Components Cleanup (Priority 2)](#9-workflow-components-cleanup)
10. [Response Caching (Priority 3)](#10-response-caching)
11. [Advanced Security Features (Priority 3)](#11-advanced-security-features)

---

## 1. Momentum Integration (Priority 1)

### Research Paper Reference
**File:** `research_paper_source.md` lines 426-489  
**Key Equations:**
```
M_t = diag(1-α_t)M_t + S_t                    (Equation 32)
S_t = diag(η_t)S_{t-1} - diag(θ_t)(M_{t-1}k_t^T k_t - v_t^T k_t)    (Equation 33)
```

**Paper Context (lines 447-450):**
> "Momentum-based Rule: The Delta Rule is based on momentary surprise, meaning that the flow of tokens cannot affect the memory update rule. LMM, however, is based on a momentum rule, which considers both past and momentary surprise."

### Current Status
**File:** `src/types.ts` lines 95-99  
**Status:** ✅ Infrastructure ready, ⏳ Integration pending

```typescript
// Already defined in IMemoryState:
momentumState?: ITensor;      // S_t
momentumDecay?: number;        // η_t
forgettingGate?: ITensor;      // α_t
```

### Implementation Steps

#### Step 1.1: Add Momentum Update to trainStep()

**File to Modify:** `src/model.ts`  
**Method:** `trainStep()` (search for "public trainStep" or "trainStep(")  
**Current Location:** Lines ~2000-2100 (approximate)

**Implementation:**

```typescript
public trainStep(x_t: ITensor, x_next: ITensor, state: IMemoryState): {
  loss: ITensor;
  gradients: IModelGradients;
  memoryUpdate: IMemoryUpdateResult;
} {
  return this.withErrorHandling('trainStep', () => {
    return tf.tidy(() => {
      const currentInput = unwrapTensor(x_t);
      const nextInput = unwrapTensor(x_next);

      // Current forward pass
      const { predicted, memoryUpdate } = this.forward(x_t, state);

      // Compute loss
      const predictionLoss = tf.mean(
        tf.squaredDifference(unwrapTensor(predicted), nextInput)
      );

      // NEW: Momentum-based memory update (Equation 32-33)
      let updatedMemoryState = memoryUpdate.newState;
      
      if (this.config.enableMomentum && state.momentumState) {
        const momentum = this.computeMomentumUpdate(
          state,
          currentInput,
          nextInput,
          memoryUpdate
        );
        
        updatedMemoryState = this.applyMomentumToMemory(
          memoryUpdate.newState,
          momentum
        );
      }

      // NEW: Apply forgetting gate if enabled
      if (this.config.enableForgettingGate && state.forgettingGate) {
        updatedMemoryState = this.applyForgettingGate(
          updatedMemoryState,
          state.forgettingGate
        );
      }

      // Continue with existing training logic...
      const optimizer = this.optimizer;
      const gradients = tf.variableGrads(() => {
        return predictionLoss;
      });

      // Apply gradients
      optimizer.applyGradients(gradients.grads);

      return {
        loss: wrapTensor(predictionLoss),
        gradients: {
          shortTerm: wrapTensor(tf.zeros([1])), // Placeholder
          longTerm: wrapTensor(tf.zeros([1])),
          meta: wrapTensor(tf.zeros([1]))
        },
        memoryUpdate: {
          ...memoryUpdate,
          newState: updatedMemoryState
        }
      };
    });
  });
}
```

#### Step 1.2: Implement computeMomentumUpdate()

**Add to:** `src/model.ts` (private method section)  
**Reference:** `research_paper_source.md` lines 432-434

```typescript
/**
 * Computes momentum update term S_t according to Equation 33
 * S_t = diag(η_t)S_{t-1} - diag(θ_t)(M_{t-1}k_t^T k_t - v_t^T k_t)
 */
private computeMomentumUpdate(
  state: IMemoryState,
  currentInput: tf.Tensor,
  nextInput: tf.Tensor,
  memoryUpdate: IMemoryUpdateResult
): tf.Tensor {
  return tf.tidy(() => {
    // Get momentum state S_{t-1}
    const S_prev = state.momentumState ? unwrapTensor(state.momentumState) : 
                   tf.zeros(unwrapTensor(state.shortTerm).shape);
    
    // Get momentum decay η_t
    const eta = state.momentumDecay ?? this.config.momentumDecayRate;
    
    // Get learning rate θ_t
    const theta = this.config.learningRate || 0.001;
    
    // Compute keys and values from attention
    const keys = memoryUpdate.attention.keys;
    const values = memoryUpdate.attention.values;
    
    // Compute M_{t-1}k_t^T k_t
    const M_prev = unwrapTensor(state.shortTerm);
    const k_t = unwrapTensor(keys);
    const v_t = unwrapTensor(values);
    
    // k_t^T k_t (outer product approximation)
    const k_outer = tf.mul(k_t, k_t);
    const M_k = tf.matMul(M_prev, k_outer.reshape([k_outer.shape[0], 1]));
    
    // v_t^T k_t (dot product)
    const v_k = tf.mul(v_t, k_t);
    
    // Gradient term: M_{t-1}k_t^T k_t - v_t^T k_t
    const gradient = tf.sub(M_k, v_k.reshape(M_k.shape));
    
    // Momentum update: η_t * S_{t-1} - θ_t * gradient
    const momentum_decay = tf.mul(S_prev, eta);
    const gradient_term = tf.mul(gradient, -theta);
    const S_t = tf.add(momentum_decay, gradient_term);
    
    return S_t as tf.Tensor2D;
  });
}
```

#### Step 1.3: Implement applyMomentumToMemory()

**Add to:** `src/model.ts` (private method section)  
**Reference:** `research_paper_source.md` line 432

```typescript
/**
 * Applies momentum update to memory state according to Equation 32
 * M_t = diag(1-α_t)M_t + S_t
 */
private applyMomentumToMemory(
  memoryState: IMemoryState,
  momentum: tf.Tensor
): IMemoryState {
  return tf.tidy(() => {
    const M_current = unwrapTensor(memoryState.shortTerm);
    const S_t = momentum;
    
    // Default forgetting: no forgetting (α_t = 0)
    // If forgetting gate enabled, it's applied separately
    const M_t = tf.add(M_current, S_t);
    
    return {
      ...memoryState,
      shortTerm: wrapTensor(M_t),
      momentumState: wrapTensor(S_t) // Store S_t for next iteration
    };
  });
}
```

#### Step 1.4: Implement applyForgettingGate()

**Add to:** `src/model.ts` (private method section)  
**Reference:** `research_paper_source.md` lines 472-476

```typescript
/**
 * Applies learnable forgetting gate to memory
 * M_t = diag(1-α_t)M_t
 */
private applyForgettingGate(
  memoryState: IMemoryState,
  forgettingGate: ITensor
): IMemoryState {
  return tf.tidy(() => {
    const M = unwrapTensor(memoryState.shortTerm);
    const alpha = unwrapTensor(forgettingGate);
    
    // Expand alpha to match memory shape if needed
    let alphaExpanded = alpha;
    if (alpha.rank === 1 && M.rank === 2) {
      alphaExpanded = alpha.reshape([alpha.shape[0], 1]);
      alphaExpanded = tf.tile(alphaExpanded, [1, M.shape[1]]);
    }
    
    // M_t = (1 - α_t) * M_t
    const oneMinusAlpha = tf.sub(1, alphaExpanded);
    const M_forgotten = tf.mul(M, oneMinusAlpha);
    
    return {
      ...memoryState,
      shortTerm: wrapTensor(M_forgotten),
      longTerm: memoryState.longTerm // Long-term memory not affected by forgetting
    };
  });
}
```

#### Step 1.5: Make Forgetting Gate Trainable (Optional Advanced)

**File:** `src/model.ts` constructor or initialize method  
**Reference:** `research_paper_source.md` lines 472-476

```typescript
// In initialize() or initializeMemoryState()
if (this.config.enableForgettingGate) {
  // Make forgetting gate a trainable variable
  const alphaInit = tf.fill([this.config.memorySlots], this.config.forgettingGateInit);
  this.forgettingGateVariable = tf.variable(alphaInit, true, 'forgetting_gate');
  
  // Update memory state to use trainable variable
  this.memoryState.forgettingGate = wrapTensor(this.forgettingGateVariable);
}
```

#### Step 1.6: Testing Momentum Integration

**File:** `src/__tests__/momentum.test.ts` (create new file)

```typescript
import * as tf from '@tensorflow/tfjs-node';
import { TitanMemoryModel } from '../model.js';
import { wrapTensor } from '../types.js';

describe('Momentum-Based Memory Updates', () => {
  let model: TitanMemoryModel;

  beforeEach(async () => {
    model = new TitanMemoryModel();
    await model.initialize({
      inputDim: 128,
      memorySlots: 100,
      enableMomentum: true,
      momentumDecayRate: 0.9,
      enableForgettingGate: false
    });
  });

  test('should initialize momentum state', () => {
    const state = model.getMemoryState();
    expect(state.momentumState).toBeDefined();
    expect(state.momentumDecay).toBe(0.9);
  });

  test('should update momentum during training', async () => {
    const x_t = wrapTensor(tf.randomNormal([128]));
    const x_next = wrapTensor(tf.randomNormal([128]));
    const state = model.getMemoryState();

    const result = model.trainStep(x_t, x_next, state);
    
    expect(result.memoryUpdate.newState.momentumState).toBeDefined();
    
    // Momentum should change after training
    const newMomentum = result.memoryUpdate.newState.momentumState;
    expect(tf.equal(newMomentum, state.momentumState).all().dataSync()[0]).toBe(0);
  });

  test('should apply momentum decay correctly', () => {
    // Test η_t parameter affects momentum
    // Implementation specific to your computeMomentumUpdate logic
  });
});
```

---

## 2. Token Flow Integration (Priority 1)

### Research Paper Reference
**File:** `research_paper_source.md` lines 477-479  
**Key Quote:**
> "Momentum-based Update Rule: TTT layers are based on momentary surprise, meaning that the flow of tokens cannot affect the memory update rule. LMM, however, is based on a momentum rule, which consider both past and momentary surprise."

### Audit Reference
**File:** `mcp-titan-system-audit.plan.md` lines 16-20

### Current Status
**File:** `src/types.ts` lines 100-103  
**Status:** ✅ Infrastructure ready, ⏳ Integration pending

```typescript
// Already defined:
tokenFlowHistory?: ITensor;    // Sequential token tracking
flowWeights?: ITensor;         // Flow contribution weights
```

### Implementation Steps

#### Step 2.1: Update Token Flow in Forward Pass

**File to Modify:** `src/model.ts`  
**Method:** `forward()` (search for "public forward")

```typescript
public forward(input: ITensor, state?: IMemoryState): {
  predicted: ITensor;
  memoryUpdate: IMemoryUpdateResult;
} {
  return this.withErrorHandling('forward', () => {
    return tf.tidy(() => {
      const memoryState = state ?? this.memoryState;
      const inputTensor = unwrapTensor(input);
      
      // NEW: Update token flow history
      let updatedFlowHistory = memoryState.tokenFlowHistory;
      let updatedFlowWeights = memoryState.flowWeights;
      
      if (this.config.enableTokenFlow && memoryState.tokenFlowHistory) {
        const flowUpdate = this.updateTokenFlow(
          memoryState.tokenFlowHistory,
          memoryState.flowWeights,
          inputTensor
        );
        updatedFlowHistory = flowUpdate.history;
        updatedFlowWeights = flowUpdate.weights;
      }
      
      // Existing forward pass logic...
      const inputVector = inputTensor.reshape([1, inputTensor.shape[0]]);
      const encodedInput = this.encoder.predict(inputVector) as tf.Tensor2D;
      
      // Compute attention
      const attention = this.computeMemoryAttention(encodedInput);
      
      // NEW: Weight surprise by token flow
      let surprise = this.computeSurprise(encodedInput, attention);
      
      if (this.config.enableTokenFlow && updatedFlowWeights) {
        surprise = this.weightSurpriseByTokenFlow(
          surprise,
          updatedFlowWeights
        );
      }
      
      // Continue with existing logic...
      const predicted = this.decoder.predict(attention.values);
      
      return {
        predicted: wrapTensor(predicted),
        memoryUpdate: {
          newState: {
            ...memoryState,
            shortTerm: wrapTensor(attention.values),
            surpriseHistory: wrapTensor(surprise.accumulated),
            tokenFlowHistory: updatedFlowHistory,
            flowWeights: updatedFlowWeights
          },
          attention,
          surprise
        }
      };
    });
  });
}
```

#### Step 2.2: Implement updateTokenFlow()

**Add to:** `src/model.ts` (private method section)

```typescript
/**
 * Updates token flow history with new input
 * Implements sliding window of recent tokens
 */
private updateTokenFlow(
  currentHistory: ITensor | undefined,
  currentWeights: ITensor | undefined,
  newToken: tf.Tensor
): { history: ITensor; weights: ITensor } {
  return tf.tidy(() => {
    const history = currentHistory ? unwrapTensor(currentHistory) : 
                    tf.zeros([this.config.tokenFlowWindow, newToken.shape[0]]);
    const weights = currentWeights ? unwrapTensor(currentWeights) :
                   tf.zeros([this.config.tokenFlowWindow]);
    
    // Roll history (shift all tokens by one position)
    const historyArray = history.arraySync() as number[][];
    historyArray.shift(); // Remove oldest
    historyArray.push(Array.from(newToken.dataSync())); // Add newest
    
    const newHistory = tf.tensor2d(historyArray);
    
    // Compute new weights based on recency and similarity
    const newWeights = this.computeTokenFlowWeights(newHistory, newToken);
    
    return {
      history: wrapTensor(newHistory),
      weights: wrapTensor(newWeights)
    };
  });
}
```

#### Step 2.3: Implement computeTokenFlowWeights()

**Add to:** `src/model.ts` (private method section)

```typescript
/**
 * Computes weights for token flow contribution
 * Recent tokens get higher weight, with decay
 */
private computeTokenFlowWeights(
  flowHistory: tf.Tensor,
  currentToken: tf.Tensor
): tf.Tensor {
  return tf.tidy(() => {
    const windowSize = this.config.tokenFlowWindow;
    
    // Recency weights: exponential decay
    const recencyWeights = tf.range(0, windowSize, 1)
      .div(windowSize)
      .sub(1)
      .abs(); // [1.0, 0.9, 0.8, ..., 0.1]
    
    // Similarity weights: cosine similarity with current token
    const currentExpanded = currentToken.reshape([1, currentToken.shape[0]]);
    const similarities = tf.matMul(flowHistory, currentExpanded, false, true)
      .squeeze();
    const normalizedSim = tf.sigmoid(similarities); // Normalize to [0, 1]
    
    // Combined weights: 50% recency, 50% similarity
    const combinedWeights = tf.add(
      tf.mul(recencyWeights, 0.5),
      tf.mul(normalizedSim, 0.5)
    );
    
    // Normalize to sum to 1
    const sumWeights = tf.sum(combinedWeights);
    const normalizedWeights = tf.div(combinedWeights, sumWeights);
    
    return normalizedWeights as tf.Tensor1D;
  });
}
```

#### Step 2.4: Implement weightSurpriseByTokenFlow()

**Add to:** `src/model.ts` (private method section)

```typescript
/**
 * Weights surprise metric by token flow contribution
 * Combines momentary surprise with flow-based surprise
 */
private weightSurpriseByTokenFlow(
  surprise: ISurpriseMetrics,
  flowWeights: ITensor
): ISurpriseMetrics {
  return tf.tidy(() => {
    const immediateValue = unwrapTensor(surprise.immediate);
    const weights = unwrapTensor(flowWeights);
    
    // Flow-weighted surprise: weight recent surprise by flow
    const flowContribution = tf.sum(tf.mul(weights, immediateValue));
    
    // Combined surprise: 50% momentary, 50% flow
    const totalSurprise = tf.add(
      tf.mul(immediateValue, 0.5),
      tf.mul(flowContribution, 0.5)
    );
    
    return {
      immediate: wrapTensor(immediateValue),
      accumulated: wrapTensor(totalSurprise),
      totalSurprise: wrapTensor(totalSurprise)
    };
  });
}
```

#### Step 2.5: Add Token Flow Metrics Tool

**File to Modify:** `src/index.ts`  
**Add new MCP tool:**

```typescript
// Add after get_memory_state tool
this.server.tool(
  'get_token_flow_metrics',
  "Get token flow analysis and statistics",
  {},
  async () => {
    await this.ensureInitialized();
    
    try {
      if (!this.memoryState.tokenFlowHistory || !this.memoryState.flowWeights) {
        return {
          content: [{
            type: "text",
            text: "Token flow tracking not enabled. Initialize with enableTokenFlow: true"
          }]
        };
      }
      
      const history = unwrapTensor(this.memoryState.tokenFlowHistory);
      const weights = unwrapTensor(this.memoryState.flowWeights);
      
      const metrics = {
        windowSize: history.shape[0],
        averageWeight: tf.mean(weights).dataSync()[0],
        maxWeight: tf.max(weights).dataSync()[0],
        minWeight: tf.min(weights).dataSync()[0],
        flowStrength: tf.sum(weights).dataSync()[0],
        historySize: history.shape[0]
      };
      
      return {
        content: [{
          type: "text",
          text: `Token Flow Metrics:\n${JSON.stringify(metrics, null, 2)}`
        }]
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        content: [{
          type: "text",
          text: `Failed to get token flow metrics: ${message}`
        }]
      };
    }
  }
);
```

---

## 3. Hierarchical Memory Activation (Priority 2)

### Research Paper Reference
**File:** `research_paper_source.md` lines 376-392  
**Key Concepts:**
- Working memory → Short-term → Long-term promotion
- Access-based promotion rules
- Time-based demotion

### Audit Reference
**File:** `mcp-titan-system-audit.plan.md` lines 85-89

### Current Status
**File:** `src/model.ts` lines 254-276, 894-929  
**Status:** ⚠️ Defined but NOT activated

```typescript
// Already defined in src/model.ts:
private promotionRules: IMemoryPromotionRules = {
  workingToShortTerm: { accessThreshold: 3, timeThreshold: 30000 },
  shortTermToLongTerm: { accessThreshold: 5, timeThreshold: 300000 },
  // ... fully configured
};
```

### Implementation Steps

#### Step 3.1: Activate Hierarchical Memory in Forward Pass

**File to Modify:** `src/model.ts`  
**Method:** `forward()` - add after memory update

```typescript
// In forward() method, after computing memory update:
if (this.config.useHierarchicalMemory || this.config.enableHierarchicalMemory) {
  const promoted = this.applyMemoryPromotion(memoryUpdate.newState);
  memoryUpdate.newState = promoted;
  
  // Track promotion statistics
  this.updatePromotionStats();
}
```

#### Step 3.2: Implement applyMemoryPromotion()

**Add to:** `src/model.ts` (private method section)

```typescript
/**
 * Applies hierarchical memory promotion/demotion rules
 * Working → Short-term → Long-term based on access patterns
 */
private applyMemoryPromotion(state: IMemoryState): IMemoryState {
  return tf.tidy(() => {
    const currentTime = Date.now();
    const timestamps = unwrapTensor(state.timestamps).arraySync() as number[];
    const accessCounts = unwrapTensor(state.accessCounts).arraySync() as number[];
    
    // Identify memories to promote from working to short-term
    const toPromote = timestamps.map((ts, idx) => {
      const age = currentTime - ts;
      const accesses = accessCounts[idx];
      const rules = this.promotionRules.workingToShortTerm;
      
      return accesses >= rules.accessThreshold && age >= rules.timeThreshold;
    });
    
    // Identify memories to promote from short-term to long-term
    const toLongTerm = timestamps.map((ts, idx) => {
      const age = currentTime - ts;
      const accesses = accessCounts[idx];
      const rules = this.promotionRules.shortTermToLongTerm;
      
      return accesses >= rules.accessThreshold && 
             age >= rules.timeThreshold &&
             toPromote[idx]; // Must already be in short-term
    });
    
    // Apply promotions
    let newState = { ...state };
    
    // Move qualifying memories to long-term
    if (toLongTerm.some(v => v)) {
      newState = this.promoteToLongTerm(newState, toLongTerm);
      this.memoryStats.promotions.total += toLongTerm.filter(v => v).length;
      this.memoryStats.promotions.recent += toLongTerm.filter(v => v).length;
    }
    
    // Apply age-based demotion
    newState = this.applyAgeDemotion(newState, currentTime);
    
    return newState;
  });
}
```

#### Step 3.3: Implement promoteToLongTerm()

**Add to:** `src/model.ts`

```typescript
/**
 * Promotes memories from short-term to long-term storage
 */
private promoteToLongTerm(
  state: IMemoryState,
  promoteFlags: boolean[]
): IMemoryState {
  return tf.tidy(() => {
    const shortTerm = unwrapTensor(state.shortTerm).arraySync() as number[][];
    const longTerm = unwrapTensor(state.longTerm).arraySync() as number[][];
    
    // Extract memories to promote
    const toPromote = shortTerm.filter((_, idx) => promoteFlags[idx]);
    
    if (toPromote.length === 0) {
      return state;
    }
    
    // Add to long-term (with capacity management)
    const maxLongTerm = Math.floor(this.config.memorySlots / 2);
    const updatedLongTerm = [...toPromote, ...longTerm].slice(0, maxLongTerm);
    
    // Remove promoted memories from short-term
    const updatedShortTerm = shortTerm.filter((_, idx) => !promoteFlags[idx]);
    
    return {
      ...state,
      shortTerm: wrapTensor(tf.tensor2d(updatedShortTerm)),
      longTerm: wrapTensor(tf.tensor2d(updatedLongTerm))
    };
  });
}
```

#### Step 3.4: Implement applyAgeDemotion()

**Add to:** `src/model.ts`

```typescript
/**
 * Demotes or removes old, low-access memories
 */
private applyAgeDemotion(
  state: IMemoryState,
  currentTime: number
): IMemoryState {
  return tf.tidy(() => {
    const timestamps = unwrapTensor(state.timestamps).arraySync() as number[];
    const accessCounts = unwrapTensor(state.accessCounts).arraySync() as number[];
    const demotionRules = this.promotionRules.demotionRules;
    
    // Calculate memory scores (higher = keep)
    const scores = timestamps.map((ts, idx) => {
      const age = currentTime - ts;
      const ageDecay = Math.pow(demotionRules.ageDecayRate, age / 1000); // Per second
      const accessBonus = accessCounts[idx] * (1 - demotionRules.lowAccessPenalty);
      return ageDecay * accessBonus;
    });
    
    // Keep memories above forgetting threshold
    const keepFlags = scores.map(score => score > demotionRules.forgettingThreshold);
    
    // Apply filtering
    const shortTerm = unwrapTensor(state.shortTerm).arraySync() as number[][];
    const filteredShortTerm = shortTerm.filter((_, idx) => keepFlags[idx]);
    const filteredTimestamps = timestamps.filter((_, idx) => keepFlags[idx]);
    const filteredAccessCounts = accessCounts.filter((_, idx) => keepFlags[idx]);
    
    const demotedCount = keepFlags.filter(v => !v).length;
    if (demotedCount > 0) {
      this.memoryStats.demotions.total += demotedCount;
      this.memoryStats.demotions.recent += demotedCount;
    }
    
    return {
      ...state,
      shortTerm: wrapTensor(tf.tensor2d(filteredShortTerm.length > 0 ? 
        filteredShortTerm : [[0]])), // Prevent empty tensor
      timestamps: wrapTensor(tf.tensor1d(filteredTimestamps)),
      accessCounts: wrapTensor(tf.tensor1d(filteredAccessCounts))
    };
  });
}
```

#### Step 3.5: Add Hierarchical Memory Metrics Tool

**File to Modify:** `src/index.ts`

```typescript
this.server.tool(
  'get_hierarchical_metrics',
  "Get hierarchical memory promotion/demotion statistics",
  {},
  async () => {
    await this.ensureInitialized();
    
    try {
      const stats = (this.model as any).memoryStats;
      const config = this.model.getConfig();
      
      if (!config.useHierarchicalMemory && !config.enableHierarchicalMemory) {
        return {
          content: [{
            type: "text",
            text: "Hierarchical memory not enabled. Initialize with enableHierarchicalMemory: true"
          }]
        };
      }
      
      const metrics = {
        promotions: stats.promotions,
        demotions: stats.demotions,
        lastUpdate: new Date(stats.lastStatsUpdate).toISOString(),
        shortTermSize: unwrapTensor(this.memoryState.shortTerm).shape[0],
        longTermSize: unwrapTensor(this.memoryState.longTerm).shape[0]
      };
      
      return {
        content: [{
          type: "text",
          text: `Hierarchical Memory Metrics:\n${JSON.stringify(metrics, null, 2)}`
        }]
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        content: [{
          type: "text",
          text: `Failed to get hierarchical metrics: ${message}`
        }]
      };
    }
  }
);
```

---

## 4. Health Check Endpoint (Priority 1)

### Audit Reference
**File:** `mcp-titan-system-audit.plan.md` lines 195-198

### Implementation Steps

#### Step 4.1: Add Health Check Endpoint

**File to Modify:** `src/index.ts`  
**Add after other tool registrations:**

```typescript
this.server.tool(
  'health_check',
  "Get system health status and diagnostics",
  {
    detailed: z.boolean().optional().describe("Include detailed diagnostics")
  },
  async (params) => {
    const detailed = params.detailed ?? false;
    
    try {
      const health = await this.performHealthCheck(detailed ? 'detailed' : 'quick');
      
      return {
        content: [{
          type: "text",
          text: JSON.stringify(health, null, 2)
        }]
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        content: [{
          type: "text",
          text: `Health check failed: ${message}`
        }]
      };
    }
  }
);
```

#### Step 4.2: Implement performHealthCheck() Method

**Add to:** `src/index.ts` (private method section)

```typescript
private async performHealthCheck(level: 'quick' | 'detailed' = 'quick'): Promise<any> {
  const startTime = Date.now();
  
  const health: any = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    version: '3.0.0'
  };
  
  try {
    // Check model initialization
    health.modelInitialized = this.isInitialized;
    if (!this.isInitialized) {
      health.status = 'degraded';
      health.warnings = ['Model not initialized'];
    }
    
    // Check TensorFlow.js memory
    const tfMemory = tf.memory();
    health.tensorflow = {
      numTensors: tfMemory.numTensors,
      numBytes: tfMemory.numBytes,
      numBytesInGPU: tfMemory.numBytesInGPU || 0
    };
    
    if (tfMemory.numTensors > 1000) {
      health.status = 'degraded';
      health.warnings = health.warnings || [];
      health.warnings.push('High tensor count - possible memory leak');
    }
    
    // Check Node.js memory
    const processMemory = process.memoryUsage();
    health.process = {
      heapUsed: Math.round(processMemory.heapUsed / 1024 / 1024) + ' MB',
      heapTotal: Math.round(processMemory.heapTotal / 1024 / 1024) + ' MB',
      external: Math.round(processMemory.external / 1024 / 1024) + ' MB',
      rss: Math.round(processMemory.rss / 1024 / 1024) + ' MB'
    };
    
    if (processMemory.heapUsed / processMemory.heapTotal > 0.9) {
      health.status = 'unhealthy';
      health.errors = health.errors || [];
      health.errors.push('Heap memory usage > 90%');
    }
    
    // Check memory state
    if (this.isInitialized) {
      const memStats = this.getMemoryStats();
      health.memory = {
        capacity: `${(memStats.capacity * 100).toFixed(1)}%`,
        surpriseScore: memStats.surpriseScore.toFixed(4),
        shortTermMean: memStats.shortTermMean.toFixed(4),
        longTermMean: memStats.longTermMean.toFixed(4)
      };
    }
    
    if (level === 'detailed') {
      // Add detailed diagnostics
      health.config = this.model?.getConfig();
      health.features = {
        momentum: this.model?.getConfig().enableMomentum,
        tokenFlow: this.model?.getConfig().enableTokenFlow,
        forgettingGate: this.model?.getConfig().enableForgettingGate,
        hierarchical: this.model?.getConfig().enableHierarchicalMemory
      };
      
      // Test operations
      try {
        const testInput = tf.randomNormal([128]);
        const testResult = this.model?.forward(wrapTensor(testInput), this.memoryState);
        testInput.dispose();
        health.operations = { forward: 'ok' };
      } catch (error) {
        health.operations = { forward: 'failed' };
        health.status = 'unhealthy';
      }
    }
    
    // Calculate response time
    health.responseTimeMs = Date.now() - startTime;
    
  } catch (error) {
    health.status = 'unhealthy';
    health.errors = health.errors || [];
    health.errors.push((error as Error).message);
  }
  
  return health;
}
```

#### Step 4.3: Add HTTP Health Endpoint (if HTTP server exists)

**File:** Search for HTTP server setup in `src/index.ts` or `src/server.ts`  
**Add route:**

```typescript
// If using Express or similar:
app.get('/health', async (req, res) => {
  const detailed = req.query.detailed === 'true';
  const health = await server.performHealthCheck(detailed ? 'detailed' : 'quick');
  
  const statusCode = health.status === 'healthy' ? 200 :
                    health.status === 'degraded' ? 200 :
                    503;
  
  res.status(statusCode).json(health);
});

// Kubernetes-style endpoints
app.get('/healthz', async (req, res) => {
  const health = await server.performHealthCheck('quick');
  res.status(health.status === 'healthy' ? 200 : 503).send(health.status);
});

app.get('/readyz', async (req, res) => {
  const ready = server.isInitialized && server.model !== null;
  res.status(ready ? 200 : 503).send(ready ? 'ready' : 'not ready');
});
```

---

## 5. Structured Logging System (Priority 1)

### Audit Reference
**File:** `mcp-titan-system-audit.plan.md` lines 200-204

### Implementation Steps

#### Step 5.1: Create Logging Infrastructure

**File to Create:** `src/logging.ts`

```typescript
import * as fs from 'fs/promises';
import * as path from 'path';

export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3
}

export interface LogEntry {
  timestamp: string;
  level: string;
  operation: string;
  message: string;
  metadata?: Record<string, any>;
  error?: {
    name: string;
    message: string;
    stack?: string;
  };
}

export class StructuredLogger {
  private static instance: StructuredLogger;
  private logBuffer: LogEntry[] = [];
  private flushInterval?: NodeJS.Timeout;
  private logLevel: LogLevel = LogLevel.INFO;
  private logDir: string;
  private maxFileSize = 10 * 1024 * 1024; // 10MB
  private maxFiles = 5;

  private constructor(logDir: string) {
    this.logDir = logDir;
    this.startFlushInterval();
  }

  public static getInstance(logDir?: string): StructuredLogger {
    if (!StructuredLogger.instance) {
      StructuredLogger.instance = new StructuredLogger(
        logDir || path.join(process.cwd(), '.titan_memory', 'logs')
      );
    }
    return StructuredLogger.instance;
  }

  public setLogLevel(level: LogLevel): void {
    this.logLevel = level;
  }

  public debug(operation: string, message: string, metadata?: Record<string, any>): void {
    if (this.logLevel <= LogLevel.DEBUG) {
      this.log('DEBUG', operation, message, metadata);
    }
  }

  public info(operation: string, message: string, metadata?: Record<string, any>): void {
    if (this.logLevel <= LogLevel.INFO) {
      this.log('INFO', operation, message, metadata);
    }
  }

  public warn(operation: string, message: string, metadata?: Record<string, any>): void {
    if (this.logLevel <= LogLevel.WARN) {
      this.log('WARN', operation, message, metadata);
    }
  }

  public error(operation: string, message: string, error?: Error, metadata?: Record<string, any>): void {
    if (this.logLevel <= LogLevel.ERROR) {
      const errorData = error ? {
        name: error.name,
        message: error.message,
        stack: error.stack
      } : undefined;
      
      this.log('ERROR', operation, message, metadata, errorData);
    }
  }

  private log(
    level: string,
    operation: string,
    message: string,
    metadata?: Record<string, any>,
    error?: { name: string; message: string; stack?: string }
  ): void {
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      operation,
      message,
      metadata,
      error
    };

    // Console output for immediate visibility
    const consoleMsg = `[${entry.timestamp}] ${level} [${operation}]: ${message}`;
    switch (level) {
      case 'ERROR':
        console.error(consoleMsg, metadata, error);
        break;
      case 'WARN':
        console.warn(consoleMsg, metadata);
        break;
      case 'DEBUG':
        console.debug(consoleMsg, metadata);
        break;
      default:
        console.log(consoleMsg, metadata);
    }

    // Buffer for file writing
    this.logBuffer.push(entry);

    // Flush if buffer is large
    if (this.logBuffer.length >= 100) {
      this.flush().catch(err => console.error('Failed to flush logs:', err));
    }
  }

  private startFlushInterval(): void {
    // Flush logs every 10 seconds
    this.flushInterval = setInterval(() => {
      this.flush().catch(err => console.error('Failed to flush logs:', err));
    }, 10000);
  }

  public async flush(): Promise<void> {
    if (this.logBuffer.length === 0) {
      return;
    }

    try {
      await fs.mkdir(this.logDir, { recursive: true });

      const today = new Date().toISOString().split('T')[0];
      const logFile = path.join(this.logDir, `titan-${today}.log`);

      // Check file size and rotate if needed
      await this.rotateLogsIfNeeded(logFile);

      // Write buffered logs
      const logLines = this.logBuffer.map(entry => JSON.stringify(entry)).join('\n') + '\n';
      await fs.appendFile(logFile, logLines, 'utf-8');

      // Clear buffer
      this.logBuffer = [];
    } catch (error) {
      console.error('Failed to write logs:', error);
    }
  }

  private async rotateLogsIfNeeded(logFile: string): Promise<void> {
    try {
      const stats = await fs.stat(logFile);
      
      if (stats.size >= this.maxFileSize) {
        // Rotate: file.log -> file.1.log -> file.2.log -> ...
        for (let i = this.maxFiles - 1; i > 0; i--) {
          const oldFile = logFile.replace('.log', `.${i}.log`);
          const newFile = logFile.replace('.log', `.${i + 1}.log`);
          
          try {
            await fs.rename(oldFile, newFile);
          } catch {
            // File doesn't exist, skip
          }
        }
        
        // Rotate current to .1
        await fs.rename(logFile, logFile.replace('.log', '.1.log'));
      }
    } catch (error) {
      if ((error as any).code !== 'ENOENT') {
        console.error('Failed to rotate logs:', error);
      }
    }
  }

  public async dispose(): Promise<void> {
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
    }
    await this.flush();
  }
}
```

#### Step 5.2: Integrate Logging into Server

**File to Modify:** `src/index.ts`  
**Add at top:**

```typescript
import { StructuredLogger, LogLevel } from './logging.js';
```

**In constructor:**

```typescript
constructor(options: { memoryPath?: string } = {}) {
  this.server = new McpServer({...});
  this.vectorProcessor = VectorProcessor.getInstance();
  this.memoryPath = options.memoryPath ?? path.join(process.cwd(), '.titan_memory');
  this.modelDir = path.join(this.memoryPath, 'model');
  this.memoryState = this.initializeEmptyState();
  
  // Initialize structured logging
  this.logger = StructuredLogger.getInstance(path.join(this.memoryPath, 'logs'));
  this.logger.setLogLevel(process.env.LOG_LEVEL === 'DEBUG' ? LogLevel.DEBUG : LogLevel.INFO);
  this.logger.info('server', 'TitanMemoryServer initialized', {
    memoryPath: this.memoryPath,
    version: '3.0.0'
  });

  this.registerTools();
}
```

**In shutdown:**

```typescript
private async shutdown(): Promise<void> {
  try {
    this.logger.info('server', 'Shutting down TitanMemoryServer');
    
    // ... existing shutdown logic ...
    
    // Flush logs before exit
    await this.logger.dispose();
    
    process.exit(0);
  } catch (error) {
    this.logger.error('server', 'Shutdown failed', error as Error);
    process.exit(1);
  }
}
```

#### Step 5.3: Replace Console.log with Structured Logging

**Throughout `src/index.ts` and `src/model.ts`, replace:**

```typescript
// Old:
console.log(`Memory initialized with ${memorySlots} slots`);
console.error('Error during training:', error);

// New:
this.logger.info('memory', `Memory initialized with ${memorySlots} slots`, {
  memorySlots,
  embeddingSize
});
this.logger.error('training', 'Training failed', error, { step: this.stepCount });
```

---

## 6. Performance Optimization (Priority 2)

### Audit Reference
**File:** `mcp-titan-system-audit.plan.md` lines 256-270

### Implementation Tasks

#### Step 6.1: Eliminate Redundant Forward Passes

**Problem:** `train_step` calls `forward()` which may call `forward()` again  
**File:** `src/model.ts`  
**Solution:** Cache forward pass results in training

```typescript
// In trainStep(), reuse forward pass results:
public trainStep(x_t: ITensor, x_next: ITensor, state: IMemoryState): {
  loss: ITensor;
  gradients: IModelGradients;
  memoryUpdate: IMemoryUpdateResult;
} {
  return this.withErrorHandling('trainStep', () => {
    return tf.tidy(() => {
      // Do forward pass ONCE
      const forwardResult = this.forward(x_t, state);
      
      // Use cached results for loss computation
      const predicted = unwrapTensor(forwardResult.predicted);
      const nextInput = unwrapTensor(x_next);
      
      const loss = tf.mean(tf.squaredDifference(predicted, nextInput));
      
      // Don't call forward() again - reuse forwardResult
      
      // Continue with gradient computation using cached results...
      
      return {
        loss: wrapTensor(loss),
        gradients: {...},
        memoryUpdate: forwardResult.memoryUpdate  // Reuse!
      };
    });
  });
}
```

#### Step 6.2: Implement LRU Cache for get_memory_state

**File to Create:** `src/cache.ts`

```typescript
export class LRUCache<K, V> {
  private cache: Map<K, { value: V; timestamp: number }>;
  private maxSize: number;
  private ttl: number; // Time to live in milliseconds

  constructor(maxSize: number = 100, ttl: number = 60000) {
    this.cache = new Map();
    this.maxSize = maxSize;
    this.ttl = ttl;
  }

  get(key: K): V | undefined {
    const entry = this.cache.get(key);
    
    if (!entry) {
      return undefined;
    }
    
    // Check if expired
    if (Date.now() - entry.timestamp > this.ttl) {
      this.cache.delete(key);
      return undefined;
    }
    
    // Move to end (most recently used)
    this.cache.delete(key);
    this.cache.set(key, entry);
    
    return entry.value;
  }

  set(key: K, value: V): void {
    // Delete if exists (to update position)
    this.cache.delete(key);
    
    // Add to end
    this.cache.set(key, {
      value,
      timestamp: Date.now()
    });
    
    // Evict oldest if over capacity
    if (this.cache.size > this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
  }

  clear(): void {
    this.cache.clear();
  }

  invalidate(key: K): void {
    this.cache.delete(key);
  }
}
```

**Integrate into `src/index.ts`:**

```typescript
import { LRUCache } from './cache.js';

export class TitanMemoryServer {
  private memoryStateCache: LRUCache<string, any>;
  
  constructor(options: { memoryPath?: string } = {}) {
    // ... existing initialization ...
    
    // Initialize cache
    this.memoryStateCache = new LRUCache(50, 30000); // 50 entries, 30s TTL
  }
  
  // In get_memory_state tool:
  this.server.tool(
    'get_memory_state',
    "Get current memory state statistics and information",
    {
      useCache: z.boolean().optional().describe("Use cached result if available")
    },
    async (params) => {
      await this.ensureInitialized();
      
      const useCache = params.useCache ?? true;
      const cacheKey = 'memory_state';
      
      // Check cache first
      if (useCache) {
        const cached = this.memoryStateCache.get(cacheKey);
        if (cached) {
          return {
            content: [{
              type: "text",
              text: cached + "\n(cached)"
            }]
          };
        }
      }
      
      try {
        const stats = this.getMemoryStats();
        const health = await this.performHealthCheck('quick');
        
        const result = `Memory State:\n- Short-term mean: ${stats.shortTermMean.toFixed(4)}\n...`;
        
        // Cache result
        this.memoryStateCache.set(cacheKey, result);
        
        return {
          content: [{
            type: "text",
            text: result
          }]
        };
      } catch (error) {
        // ... error handling ...
      }
    }
  );
  
  // Invalidate cache on memory updates
  private invalidateMemoryCache(): void {
    this.memoryStateCache.clear();
  }
  
  // Call invalidateMemoryCache() after train_step, forward_pass, etc.
}
```

#### Step 6.3: Use In-Place Tensor Operations

**File:** `src/model.ts`  
**Search for:** `.clone()` operations and replace where safe

```typescript
// Old (creates copy):
const updated = currentTensor.add(delta).clone();

// New (in-place where possible):
const updated = currentTensor.add(delta);  // Don't clone unless needed

// Or use tf.keep() for tensors that need to persist:
const kept = tf.keep(currentTensor.add(delta));
```

**Specific locations to optimize:**
- Memory update operations in `forward()`
- Gradient computations in `trainStep()`
- Attention calculations in `computeMemoryAttention()`

---

## 7. Workflow Components Cleanup (Priority 2)

### Audit Reference
**File:** `mcp-titan-system-audit.plan.md` lines 274-284

### Investigation Required

#### Step 7.1: Analyze Workflow Files

**Files to Review:**
1. `src/workflows/WorkflowOrchestrator.ts`
2. `src/workflows/GitHubWorkflowManager.ts`
3. `src/workflows/LintingManager.ts`
4. `src/workflows/FeedbackProcessor.ts`
5. `src/workflows/WorkflowUtils.ts`

**For Each File:**
1. Check if imported anywhere: `grep -r "from.*workflows" src/`
2. Check if used in tools: Search `src/index.ts` for references
3. Document purpose if unclear
4. Decide: Keep, Document, or Remove

#### Step 7.2: Create Documentation or Remove

**Option A: If Workflows Are Intended Features**

Create `docs/workflows.md`:

```markdown
# Workflow Components

## WorkflowOrchestrator
- **Purpose**: Coordinates multi-step memory operations
- **Status**: Experimental, not yet integrated
- **Usage**: (document how to enable/use)

## GitHubWorkflowManager  
- **Purpose**: GitHub integration for memory persistence
- **Status**: Experimental
- **Usage**: (document)

## LintingManager
- **Purpose**: Code quality checks for memory patterns
- **Status**: Placeholder
- **Usage**: (document)

## FeedbackProcessor
- **Purpose**: User feedback loop for memory quality
- **Status**: Not implemented
- **Usage**: (document)
```

**Option B: If Workflows Are Obsolete**

Remove files and update project structure documentation.

---

## 8. Response Caching (Priority 3)

### Audit Reference
**File:** `mcp-titan-system-audit.plan.md` lines 245-248

### Implementation (builds on Step 6.2)

#### Step 8.1: Expand Caching to More Tools

**Tools to Cache:**
- `get_memory_state` (already done in Step 6.2)
- `get_surprise_metrics`
- `analyze_memory`
- `get_token_flow_metrics`
- `get_hierarchical_metrics`

**Pattern for each tool:**

```typescript
this.server.tool(
  'cacheable_tool',
  "Description",
  {
    useCache: z.boolean().optional()
  },
  async (params) => {
    const useCache = params.useCache ?? true;
    const cacheKey = `tool_name_${JSON.stringify(params)}`;
    
    if (useCache) {
      const cached = this.cache.get(cacheKey);
      if (cached) return cached;
    }
    
    const result = await computeExpensiveOperation();
    this.cache.set(cacheKey, result);
    return result;
  }
);
```

#### Step 8.2: Add Response Compression

**File:** `src/index.ts`  
**Install:** `npm install zlib`

```typescript
import * as zlib from 'zlib';
import { promisify } from 'util';

const gzip = promisify(zlib.gzip);
const gunzip = promisify(zlib.gunzip);

// Add compression utility
private async compressResponse(data: any): Promise<Buffer> {
  const json = JSON.stringify(data);
  return await gzip(json);
}

private async decompressResponse(buffer: Buffer): Promise<any> {
  const json = await gunzip(buffer);
  return JSON.parse(json.toString());
}

// Use in large response tools
this.server.tool(
  'get_full_memory_dump',
  "Get complete memory dump (large response)",
  {
    compress: z.boolean().optional().describe("Compress response")
  },
  async (params) => {
    const memoryDump = await this.getFullMemoryDump();
    
    if (params.compress) {
      const compressed = await this.compressResponse(memoryDump);
      return {
        content: [{
          type: "text",
          text: `Compressed data (${compressed.length} bytes). Use decompress tool to extract.`,
          data: compressed.toString('base64')
        }]
      };
    }
    
    return {
      content: [{
        type: "text",
        text: JSON.stringify(memoryDump, null, 2)
      }]
    };
  }
);
```

---

## 9. Advanced Security Features (Priority 3)

### Audit Reference
**File:** `mcp-titan-system-audit.plan.md` lines 218-232

### Implementation Steps

#### Step 9.1: Checkpoint Encryption

**File:** `src/index.ts`  
**Add encryption utilities:**

```typescript
import * as crypto from 'crypto';

private readonly ENCRYPTION_KEY = process.env.TITAN_ENCRYPTION_KEY || 
  crypto.randomBytes(32).toString('hex');
private readonly ENCRYPTION_IV_LENGTH = 16;

private encryptData(data: string): { encrypted: string; iv: string } {
  const iv = crypto.randomBytes(this.ENCRYPTION_IV_LENGTH);
  const cipher = crypto.createCipheriv(
    'aes-256-cbc',
    Buffer.from(this.ENCRYPTION_KEY, 'hex'),
    iv
  );
  
  let encrypted = cipher.update(data, 'utf8', 'hex');
  encrypted += cipher.final('hex');
  
  return {
    encrypted,
    iv: iv.toString('hex')
  };
}

private decryptData(encrypted: string, iv: string): string {
  const decipher = crypto.createDecipheriv(
    'aes-256-cbc',
    Buffer.from(this.ENCRYPTION_KEY, 'hex'),
    Buffer.from(iv, 'hex')
  );
  
  let decrypted = decipher.update(encrypted, 'hex', 'utf8');
  decrypted += decipher.final('utf8');
  
  return decrypted;
}

// Update save_checkpoint tool:
this.server.tool(
  'save_checkpoint',
  "Save current memory state to a checkpoint file",
  {
    path: z.string().describe("Path to save the checkpoint"),
    encrypt: z.boolean().optional().describe("Encrypt checkpoint data")
  },
  async (params) => {
    await this.ensureInitialized();

    try {
      const validatedPath = this.validateFilePath(params.path);
      
      const checkpointData = {
        // ... existing checkpoint data ...
      };
      
      let dataToWrite = JSON.stringify(checkpointData, null, 2);
      let metadata: any = {};
      
      if (params.encrypt) {
        const { encrypted, iv } = this.encryptData(dataToWrite);
        metadata = { encrypted: true, iv };
        dataToWrite = encrypted;
      }
      
      const finalData = {
        metadata,
        data: dataToWrite
      };
      
      await fs.writeFile(validatedPath, JSON.stringify(finalData, null, 2));
      
      return {
        content: [{
          type: "text",
          text: `Checkpoint saved to ${validatedPath}${params.encrypt ? ' (encrypted)' : ''}`
        }]
      };
    } catch (error) {
      // ... error handling ...
    }
  }
);
```

#### Step 9.2: API Authentication

**File:** `src/index.ts`  
**Add authentication middleware:**

```typescript
private readonly API_KEYS = new Set(
  (process.env.TITAN_API_KEYS || '').split(',').filter(k => k.length > 0)
);

private validateApiKey(key?: string): boolean {
  if (this.API_KEYS.size === 0) {
    // No keys configured = no auth required (development mode)
    return true;
  }
  
  return key ? this.API_KEYS.has(key) : false;
}

// Wrap tool handler with auth:
private registerToolWithAuth(
  name: string,
  description: string,
  schema: any,
  handler: Function
): void {
  this.server.tool(name, description, {
    ...schema,
    apiKey: z.string().optional().describe("API authentication key")
  }, async (params) => {
    // Check authentication
    if (!this.validateApiKey(params.apiKey)) {
      return {
        content: [{
          type: "error",
          text: "Authentication required. Provide valid API key."
        }]
      };
    }
    
    // Call original handler
    return await handler(params);
  });
}
```

#### Step 9.3: Rate Limiting

**File:** `src/index.ts`  
**Add rate limiting:**

```typescript
private rateLimiter = new Map<string, { count: number; resetTime: number }>();
private readonly RATE_LIMIT = parseInt(process.env.TITAN_RATE_LIMIT || '100'); // requests per minute
private readonly RATE_WINDOW = 60000; // 1 minute

private checkRateLimit(identifier: string): { allowed: boolean; remaining: number } {
  const now = Date.now();
  const record = this.rateLimiter.get(identifier);
  
  if (!record || now > record.resetTime) {
    // New window
    this.rateLimiter.set(identifier, {
      count: 1,
      resetTime: now + this.RATE_WINDOW
    });
    return { allowed: true, remaining: this.RATE_LIMIT - 1 };
  }
  
  if (record.count >= this.RATE_LIMIT) {
    return { allowed: false, remaining: 0 };
  }
  
  record.count++;
  return { allowed: true, remaining: this.RATE_LIMIT - record.count };
}

// Add to tool wrapper:
private async handleToolCall(toolName: string, params: any): Promise<any> {
  const identifier = params.apiKey || 'anonymous';
  const rateCheck = this.checkRateLimit(identifier);
  
  if (!rateCheck.allowed) {
    return {
      content: [{
        type: "error",
        text: "Rate limit exceeded. Try again later."
      }]
    };
  }
  
  // Continue with tool execution...
}
```

---

## Implementation Tracking

Use `IMPLEMENTATION_PACKAGE.md` for the unified checklist and navigation matrix. `SYSTEM_AUDIT.md` contains the authoritative status tracker, while `IMPLEMENTATION_PROGRESS.md` should be updated after each milestone.

For testing, pair the per-section strategies in this guide with:
1. **Unit coverage** under `test/` (add new files as needed).
2. **Integration harness** (planned) that exercises MCP stdio workflows end-to-end.
3. **Manual MCP validation** using Cursor/Claude until automated coverage lands.

---

## Reference Quick Links

- **Research Paper**: `research_paper_source.md`
- **System Audit**: `SYSTEM_AUDIT.md`
- **Implementation Package**: `IMPLEMENTATION_PACKAGE.md`
- **Current Progress Log**: `IMPLEMENTATION_PROGRESS.md`

---

**End of Detailed Implementation Guide**

This guide provides the context necessary to implement the remaining Titans research features and production-hardening tasks. Each section captures:
- Research paper references with line numbers
- Current code locations
- Implementation steps
- Testing ideas and integration considerations

Follow the priority ordering defined in `IMPLEMENTATION_PACKAGE.md`, and update both the audit and progress tracker as you complete work.
