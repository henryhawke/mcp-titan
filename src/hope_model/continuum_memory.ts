import * as tf from '@tensorflow/tfjs-node';
import type { IMemoryState } from '../types.js';
import { tidyMemoryState, tidyTensor2D } from './type_utils.js';

export interface ContinuumMemoryConfig {
  memoryDim: number;
  shortTermSlots: number;
  longTermSlots: number;
  archiveSlots: number;
  promotionThreshold: number;
  surpriseRetention: number;
  momentumDecay?: number;          // eta_t in HOPE paper (default: 0.9)
  enableMomentum?: boolean;        // Enable momentum-based updates
  enableForgettingGate?: boolean;  // Enable adaptive forgetting
  baseForgettingRate?: number;     // Base alpha_t (default: 0.1)
  surpriseForgettingWeight?: number; // How much surprise affects forgetting (default: 0.3)
}

export interface MemoryWriteMetadata {
  surprise: number;
  timestamp: number;
  routeWeights: tf.Tensor2D;
}

export interface HierarchicalStats {
  shortTerm: number;
  longTerm: number;
  archive: number;
  averageSurprise: number;
}

type BaseStateOverrides = 'shortTerm' | 'longTerm' | 'meta' | 'timestamps' | 'accessCounts' | 'surpriseHistory';

export type HopeMemoryState = Omit<IMemoryState, BaseStateOverrides> & {
  shortTerm: tf.Tensor2D;
  longTerm: tf.Tensor2D;
  meta: tf.Tensor2D;
  timestamps: tf.Tensor1D;
  accessCounts: tf.Tensor1D;
  surpriseHistory: tf.Tensor1D;
  archive: tf.Tensor2D;
  levelIndex: tf.Tensor1D;
  surpriseBuffer: tf.Tensor1D;
  // HOPE paper: Momentum-based memory updates (Equations 32-33)
  momentumState?: tf.Tensor2D;      // S_t in the paper
  forgettingGate?: number;          // alpha_t in the paper
};

const EPSILON = 1e-6;

export class ContinuumMemory {
  private readonly config: ContinuumMemoryConfig;

  constructor(config: ContinuumMemoryConfig) {
    this.config = config;
  }

  public initialize(): HopeMemoryState {
    return tidyMemoryState(() => ({
      shortTerm: tf.tensor2d([], [0, this.config.memoryDim]),
      longTerm: tf.tensor2d([], [0, this.config.memoryDim]),
      archive: tf.tensor2d([], [0, this.config.memoryDim]),
      meta: tf.tensor2d([], [0, 4]),
      timestamps: tf.tensor1d([]),
      accessCounts: tf.tensor1d([]),
      surpriseHistory: tf.tensor1d([]),
      levelIndex: tf.tensor1d([]),
      surpriseBuffer: tf.tensor1d([]),
      // Initialize momentum state if enabled
      momentumState: this.config.enableMomentum ? tf.tensor2d([], [0, this.config.memoryDim]) : undefined,
      forgettingGate: 0
    }));
  }

  public clone(state: HopeMemoryState): HopeMemoryState {
    return tidyMemoryState(() => ({
      shortTerm: state.shortTerm.clone(),
      longTerm: state.longTerm.clone(),
      archive: state.archive.clone(),
      meta: state.meta.clone(),
      timestamps: state.timestamps.clone(),
      accessCounts: state.accessCounts.clone(),
      surpriseHistory: state.surpriseHistory.clone(),
      levelIndex: state.levelIndex.clone(),
      surpriseBuffer: state.surpriseBuffer.clone(),
      momentumState: state.momentumState?.clone(),
      forgettingGate: state.forgettingGate
    }));
  }

  /**
   * HOPE Paper: Compute adaptive forgetting gate (alpha_t)
   * Higher surprise = lower forgetting (retain more)
   * Lower surprise = higher forgetting (forget more redundant info)
   */
  public updateForgettingGate(surprise: number): number {
    if (!this.config.enableForgettingGate) {
      return 0; // No forgetting if disabled
    }

    const baseAlpha = this.config.baseForgettingRate ?? 0.1;
    const surpriseWeight = this.config.surpriseForgettingWeight ?? 0.3;

    // Inverse relationship: high surprise = low forgetting
    // Clamp to [0, 0.5] to prevent excessive forgetting
    const alpha_t = Math.min(0.5, baseAlpha * (1 - surpriseWeight * Math.min(surprise, 1.0)));
    return alpha_t;
  }

  /**
   * HOPE Paper Equation 33: Momentum update
   * S_t = diag(eta_t) * S_{t-1} - diag(theta_t) * (M_{t-1} * k_t^T * k_t - v_t^T * k_t)
   *
   * This implements momentum-based memory updates to prevent catastrophic forgetting
   */
  public computeMomentumUpdate(
    prevMomentum: tf.Tensor2D,
    currentMemory: tf.Tensor2D,
    keys: tf.Tensor2D,
    values: tf.Tensor2D,
    learningRate: number
  ): tf.Tensor2D {
    return tidyTensor2D(() => {
      const eta = this.config.momentumDecay ?? 0.9;

      // Handle empty previous momentum
      if (prevMomentum.shape[0] === 0) {
        return tf.zerosLike(currentMemory);
      }

      // Ensure dimensions match
      const momentumSize = Math.min(prevMomentum.shape[0], currentMemory.shape[0]);
      const momentum = prevMomentum.shape[0] > momentumSize
        ? prevMomentum.slice([0, 0], [momentumSize, -1])
        : prevMomentum;
      const memory = currentMemory.shape[0] > momentumSize
        ? currentMemory.slice([0, 0], [momentumSize, -1])
        : currentMemory;

      // S_t = eta * S_{t-1} (decay previous momentum)
      const decayed = momentum.mul(eta);

      // Compute gradient term: (M * k^T * k - v^T * k)
      // For continuum memory, we approximate with simplified computation
      const keysT = keys.transpose();
      const memoryGrad = memory.matMul(keysT).matMul(keys);
      const valueGrad = values.transpose().matMul(keys);
      const gradient = memoryGrad.sub(valueGrad);

      // S_t = eta * S_{t-1} - theta * gradient
      const update = decayed.sub(gradient.mul(learningRate));

      return update as tf.Tensor2D;
    });
  }

  /**
   * HOPE Paper Equation 32: Apply momentum to memory
   * M_t = diag(1 - alpha_t) * M_t + S_t
   *
   * Combines forgetting gate with momentum update
   */
  public applyMomentumToMemory(
    memory: tf.Tensor2D,
    momentum: tf.Tensor2D,
    forgettingGate: number
  ): tf.Tensor2D {
    return tidyTensor2D(() => {
      if (momentum.shape[0] === 0 || memory.shape[0] === 0) {
        return memory;
      }

      // Ensure dimensions match
      const size = Math.min(memory.shape[0], momentum.shape[0]);
      const mem = memory.shape[0] > size ? memory.slice([0, 0], [size, -1]) : memory;
      const mom = momentum.shape[0] > size ? momentum.slice([0, 0], [size, -1]) : momentum;

      // M_t = (1 - alpha_t) * M_t + S_t
      const retained = mem.mul(1 - forgettingGate);
      const updated = retained.add(mom);

      // If original memory was larger, preserve the extra rows
      if (memory.shape[0] > size) {
        const remainder = memory.slice([size, 0], [-1, -1]);
        return tf.concat([updated, remainder], 0) as tf.Tensor2D;
      }

      return updated as tf.Tensor2D;
    });
  }

  public write(state: HopeMemoryState, embedding: tf.Tensor2D, metadata: MemoryWriteMetadata): HopeMemoryState {
    return tidyMemoryState(() => {
      const normalized = this.normalize(embedding);

      // HOPE Paper: Compute adaptive forgetting gate
      const alpha_t = this.updateForgettingGate(metadata.surprise);

      // Apply momentum-based updates if enabled
      let processedShortTerm = state.shortTerm;
      let newMomentumState = state.momentumState;

      if (this.config.enableMomentum && state.momentumState && state.shortTerm.shape[0] > 0) {
        // Compute momentum update (Equation 33)
        const momentum = this.computeMomentumUpdate(
          state.momentumState,
          state.shortTerm,
          normalized,
          normalized,
          this.config.surpriseRetention
        );

        // Apply momentum with forgetting gate (Equation 32)
        processedShortTerm = this.applyMomentumToMemory(
          state.shortTerm,
          momentum,
          alpha_t
        );

        // Update momentum state for next iteration
        newMomentumState = momentum;
      } else if (this.config.enableForgettingGate && state.shortTerm.shape[0] > 0) {
        // Apply forgetting gate without momentum
        processedShortTerm = state.shortTerm.mul(1 - alpha_t) as tf.Tensor2D;
      }

      // Add new memory to processed short-term
      const newShort = tf.concat<tf.Tensor2D>([processedShortTerm, normalized], 0);

      const newMeta = tf.concat<tf.Tensor2D>([
        state.meta,
        tf.tensor2d([[metadata.surprise, metadata.timestamp, alpha_t, 0]], [1, 4])
      ], 0);
      const newSurpriseHistory = tf.concat<tf.Tensor1D>([state.surpriseHistory, tf.tensor1d([metadata.surprise])], 0);
      const newLevelIndex = tf.concat<tf.Tensor1D>([state.levelIndex, tf.tensor1d([0])], 0);

      // Update momentum state size to match short-term (if enabled)
      if (this.config.enableMomentum && newMomentumState) {
        const currentSize = newShort.shape[0];
        const momentumSize = newMomentumState.shape[0];
        if (momentumSize < currentSize) {
          // Expand momentum with zeros for new entries
          const padding = tf.zeros([currentSize - momentumSize, this.config.memoryDim]);
          newMomentumState = tf.concat([newMomentumState, padding], 0) as tf.Tensor2D;
        }
      }

      let updated: HopeMemoryState = {
        ...state,
        shortTerm: newShort,
        meta: newMeta,
        surpriseHistory: newSurpriseHistory,
        levelIndex: newLevelIndex,
        accessCounts: tf.concat<tf.Tensor1D>([state.accessCounts, tf.tensor1d([0])], 0),
        timestamps: tf.concat<tf.Tensor1D>([state.timestamps, tf.tensor1d([metadata.timestamp])], 0),
        surpriseBuffer: tf.concat<tf.Tensor1D>([state.surpriseBuffer, tf.tensor1d([metadata.surprise])], 0),
        momentumState: newMomentumState,
        forgettingGate: alpha_t
      };

      updated = this.ensureCapacity(updated);
      return updated;
    });
  }

  public read(state: HopeMemoryState, query: tf.Tensor2D, weights: tf.Tensor2D): tf.Tensor2D {
    return tidyTensor2D(() => {
      const normalizedQuery = this.normalize(query);

      const reads: tf.Tensor2D[] = [];
      const segments: Array<{ tensor: tf.Tensor2D; weight: tf.Tensor }>
        = [
          { tensor: state.shortTerm, weight: weights.slice([0, 0], [1, 1]) },
          { tensor: state.longTerm, weight: weights.slice([0, 1], [1, 1]) },
          { tensor: state.archive, weight: weights.slice([0, 2], [1, 1]) }
        ];

      segments.forEach(({ tensor, weight }) => {
        if (tensor.shape[0] === 0) {
          reads.push(tf.zerosLike(normalizedQuery));
          return;
        }
        const similarities = tf.matMul(normalizedQuery, tensor, false, true);
        const attention = tf.softmax(similarities, 1);
        const context = tf.matMul(attention, tensor);
        reads.push(tf.mul(context, weight));
      });

      return reads.reduce((acc, current) => tf.add(acc, current));
    });
  }

  public promote(state: HopeMemoryState): HopeMemoryState {
    return tidyMemoryState(() => {
      let updated = state;
      const overflow = state.shortTerm.shape[0] - this.config.shortTermSlots;
      if (overflow > 0) {
        const promoteSlice = state.shortTerm.slice([0, 0], [overflow, -1]);
        const remainingShort = state.shortTerm.slice([overflow, 0], [-1, -1]);

        updated = {
          ...updated,
          shortTerm: remainingShort,
          longTerm: tf.concat<tf.Tensor2D>([state.longTerm, promoteSlice], 0),
          meta: tf.concat<tf.Tensor2D>([
            state.meta.slice([overflow, 0], [-1, -1]),
            tf.tensor2d(
              Array.from({ length: overflow }, () => [0, 0, 0, 0]),
              [overflow, 4]
            )
          ], 0),
          levelIndex: tf.concat<tf.Tensor1D>([
            state.levelIndex.slice([overflow], [-1]),
            tf.tensor1d(Array(overflow).fill(1))
          ], 0)
        } as HopeMemoryState;
      }

      const longOverflow = updated.longTerm.shape[0] - this.config.longTermSlots;
      if (longOverflow > 0) {
        const archiveSlice = updated.longTerm.slice([0, 0], [longOverflow, -1]);
        const remainingLong = updated.longTerm.slice([longOverflow, 0], [-1, -1]);
        updated = {
          ...updated,
          longTerm: remainingLong,
          archive: tf.concat<tf.Tensor2D>([updated.archive, archiveSlice], 0)
        } as HopeMemoryState;
      }

      const archiveOverflow = updated.archive.shape[0] - this.config.archiveSlots;
      if (archiveOverflow > 0) {
        const trimmedArchive = updated.archive.slice([archiveOverflow, 0], [-1, -1]);
        updated = { ...updated, archive: trimmedArchive } as HopeMemoryState;
      }

      return updated;
    });
  }

  public prune(state: HopeMemoryState, threshold: number): HopeMemoryState {
    return tidyMemoryState(() => {
      if (state.longTerm.shape[0] === 0) { return state; }
      const magnitudes = tf.mean(tf.abs(state.longTerm), 1);
      const mask = tf.greaterEqual(magnitudes, tf.scalar(threshold));
      const maskValues = Array.from(mask.dataSync());
      const indicesList: number[] = [];
      maskValues.forEach((value, index) => {
        if (value > 0) {
          indicesList.push(index);
        }
      });
      if (indicesList.length === 0) {
        return state;
      }
      const indices = tf.tensor1d(indicesList, 'int32');
      const pruned = tf.gather(state.longTerm, indices);
      return {
        ...state,
        longTerm: pruned
      } as HopeMemoryState;
    });
  }

  public getStats(state: HopeMemoryState): HierarchicalStats {
    return tf.tidy(() => ({
      shortTerm: state.shortTerm.shape[0],
      longTerm: state.longTerm.shape[0],
      archive: state.archive.shape[0],
      averageSurprise: state.surpriseHistory.shape[0] === 0
        ? 0
        : tf.mean(state.surpriseHistory).arraySync() as number
    }));
  }

  public serialize(state: HopeMemoryState): Record<string, number[] | number> {
    const serializeTensor = (tensor: tf.Tensor) => Array.from(tensor.dataSync());
    return {
      shortTerm: serializeTensor(state.shortTerm),
      longTerm: serializeTensor(state.longTerm),
      archive: serializeTensor(state.archive),
      meta: serializeTensor(state.meta),
      timestamps: serializeTensor(state.timestamps),
      accessCounts: serializeTensor(state.accessCounts),
      surpriseHistory: serializeTensor(state.surpriseHistory),
      levelIndex: serializeTensor(state.levelIndex),
      surpriseBuffer: serializeTensor(state.surpriseBuffer),
      momentumState: state.momentumState ? serializeTensor(state.momentumState) : [],
      forgettingGate: state.forgettingGate ?? 0,
      memoryDim: this.config.memoryDim,
      shortTermSlots: this.config.shortTermSlots,
      longTermSlots: this.config.longTermSlots,
      archiveSlots: this.config.archiveSlots
    };
  }

  public deserialize(data: Record<string, number[] | number>): HopeMemoryState {
    const toTensor2d = (values: number[], dim: number) => {
      const rows = values.length / dim;
      return tf.tensor2d(values, [rows, dim]);
    };
    const toTensor1d = (values: number[]) => tf.tensor1d(values);
    const dim = (data.memoryDim as number) ?? this.config.memoryDim;

    const momentumData = data.momentumState as number[] ?? [];
    const momentumState = momentumData.length > 0 ? toTensor2d(momentumData, dim) : undefined;

    return {
      shortTerm: toTensor2d(data.shortTerm as number[] ?? [], dim),
      longTerm: toTensor2d(data.longTerm as number[] ?? [], dim),
      archive: toTensor2d(data.archive as number[] ?? [], dim),
      meta: tf.tensor2d(data.meta as number[] ?? [], [(data.meta as number[] ?? []).length / 4 || 0, 4]),
      timestamps: toTensor1d(data.timestamps as number[] ?? []),
      accessCounts: toTensor1d(data.accessCounts as number[] ?? []),
      surpriseHistory: toTensor1d(data.surpriseHistory as number[] ?? []),
      levelIndex: toTensor1d(data.levelIndex as number[] ?? []),
      surpriseBuffer: toTensor1d(data.surpriseBuffer as number[] ?? []),
      momentumState,
      forgettingGate: (data.forgettingGate as number) ?? 0
    } as HopeMemoryState;
  }

  private ensureCapacity(state: HopeMemoryState): HopeMemoryState {
    let updated = this.promote(state);
    const maxShort = this.config.shortTermSlots;
    if (updated.shortTerm.shape[0] > maxShort) {
      updated = this.promote(updated);
    }
    return updated;
  }

  private normalize(tensor: tf.Tensor2D): tf.Tensor2D {
    const norms = tf.maximum(tf.norm(tensor, 'euclidean', 1), tf.scalar(EPSILON));
    return tf.div(tensor, norms.expandDims(1));
  }
}
