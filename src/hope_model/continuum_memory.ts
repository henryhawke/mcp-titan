import * as tf from '@tensorflow/tfjs-node';
import type { IMemoryState } from '../types.js';

export interface ContinuumMemoryConfig {
  memoryDim: number;
  shortTermSlots: number;
  longTermSlots: number;
  archiveSlots: number;
  promotionThreshold: number;
  surpriseRetention: number;
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

export type HopeMemoryState = IMemoryState & {
  archive: tf.Tensor2D;
  levelIndex: tf.Tensor1D;
  surpriseBuffer: tf.Tensor1D;
};

const EPSILON = 1e-6;

export class ContinuumMemory {
  private readonly config: ContinuumMemoryConfig;

  constructor(config: ContinuumMemoryConfig) {
    this.config = config;
  }

  public initialize(): HopeMemoryState {
    return tf.tidy(() => ({
      shortTerm: tf.tensor2d([], [0, this.config.memoryDim]),
      longTerm: tf.tensor2d([], [0, this.config.memoryDim]),
      archive: tf.tensor2d([], [0, this.config.memoryDim]),
      meta: tf.tensor2d([], [0, 4]),
      timestamps: tf.tensor1d([]),
      accessCounts: tf.tensor1d([]),
      surpriseHistory: tf.tensor1d([]),
      levelIndex: tf.tensor1d([]),
      surpriseBuffer: tf.tensor1d([])
    }));
  }

  public clone(state: HopeMemoryState): HopeMemoryState {
    return tf.tidy(() => ({
      shortTerm: state.shortTerm.clone(),
      longTerm: state.longTerm.clone(),
      archive: state.archive.clone(),
      meta: state.meta.clone(),
      timestamps: state.timestamps.clone(),
      accessCounts: state.accessCounts.clone(),
      surpriseHistory: state.surpriseHistory.clone(),
      levelIndex: state.levelIndex.clone(),
      surpriseBuffer: state.surpriseBuffer.clone()
    }));
  }

  public write(state: HopeMemoryState, embedding: tf.Tensor2D, metadata: MemoryWriteMetadata): HopeMemoryState {
    return tf.tidy(() => {
      const normalized = this.normalize(embedding);
      const newShort = tf.concat([state.shortTerm, normalized], 0);
      const newMeta = tf.concat([
        state.meta,
        tf.tensor2d([[metadata.surprise, metadata.timestamp, 0, 0]])
      ], 0);
      const newSurpriseHistory = tf.concat([state.surpriseHistory, tf.tensor1d([metadata.surprise])], 0);
      const newLevelIndex = tf.concat([state.levelIndex, tf.tensor1d([0])], 0);

      let updated: HopeMemoryState = {
        ...state,
        shortTerm: newShort,
        meta: newMeta,
        surpriseHistory: newSurpriseHistory,
        levelIndex: newLevelIndex,
        accessCounts: tf.concat([state.accessCounts, tf.tensor1d([0])], 0),
        timestamps: tf.concat([state.timestamps, tf.tensor1d([metadata.timestamp])], 0),
        surpriseBuffer: tf.concat([state.surpriseBuffer, tf.tensor1d([metadata.surprise])], 0)
      };

      updated = this.ensureCapacity(updated);
      return updated;
    });
  }

  public read(state: HopeMemoryState, query: tf.Tensor2D, weights: tf.Tensor2D): tf.Tensor2D {
    return tf.tidy(() => {
      const normalizedQuery = this.normalize(query);

      const reads: tf.Tensor[] = [];
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
    return tf.tidy(() => {
      let updated = state;
      const overflow = state.shortTerm.shape[0] - this.config.shortTermSlots;
      if (overflow > 0) {
        const promoteSlice = state.shortTerm.slice([0, 0], [overflow, -1]);
        const remainingShort = state.shortTerm.slice([overflow, 0], [-1, -1]);

        updated = {
          ...updated,
          shortTerm: remainingShort,
          longTerm: tf.concat([state.longTerm, promoteSlice], 0),
          meta: tf.concat([state.meta.slice([overflow, 0], [-1, -1]), tf.tensor2d(Array(overflow).fill([0, 0, 0, 0]))], 0),
          levelIndex: tf.concat([state.levelIndex.slice([overflow], [-1]), tf.tensor1d(Array(overflow).fill(1))], 0)
        } as HopeMemoryState;
      }

      const longOverflow = updated.longTerm.shape[0] - this.config.longTermSlots;
      if (longOverflow > 0) {
        const archiveSlice = updated.longTerm.slice([0, 0], [longOverflow, -1]);
        const remainingLong = updated.longTerm.slice([longOverflow, 0], [-1, -1]);
        updated = {
          ...updated,
          longTerm: remainingLong,
          archive: tf.concat([updated.archive, archiveSlice], 0)
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
    return tf.tidy(() => {
      if (state.longTerm.shape[0] === 0) { return state; }
      const magnitudes = tf.mean(tf.abs(state.longTerm), 1);
      const mask = tf.greaterEqual(magnitudes, tf.scalar(threshold));
      const indices = tf.where(mask).reshape([-1]);
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
    return {
      shortTerm: toTensor2d(data.shortTerm as number[] ?? [], dim),
      longTerm: toTensor2d(data.longTerm as number[] ?? [], dim),
      archive: toTensor2d(data.archive as number[] ?? [], dim),
      meta: tf.tensor2d(data.meta as number[] ?? [], [(data.meta as number[] ?? []).length / 4 || 0, 4]),
      timestamps: toTensor1d(data.timestamps as number[] ?? []),
      accessCounts: toTensor1d(data.accessCounts as number[] ?? []),
      surpriseHistory: toTensor1d(data.surpriseHistory as number[] ?? []),
      levelIndex: toTensor1d(data.levelIndex as number[] ?? []),
      surpriseBuffer: toTensor1d(data.surpriseBuffer as number[] ?? [])
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
