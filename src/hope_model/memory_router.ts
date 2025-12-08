import * as tf from '@tensorflow/tfjs-node';
import type { HopeMemoryState } from './continuum_memory.js';

export interface MemoryRouterConfig {
  hiddenDim: number;
  numExperts: number;
  topK?: number;
}

export interface RoutingDecision {
  weights: tf.Tensor2D;
  surprise: number;
  selectedExperts: number[];
}

const DEFAULT_TOPK = 2;

export class MemoryRouter {
  private readonly config: Required<MemoryRouterConfig>;
  private readonly routerKernel: tf.Variable<tf.Rank.R2>;
  private readonly routerBias: tf.Variable<tf.Rank.R1>;

  constructor(config: MemoryRouterConfig) {
    this.config = {
      ...config,
      topK: config.topK ?? DEFAULT_TOPK
    } as Required<MemoryRouterConfig>;

    this.routerKernel = tf.variable(tf.randomNormal([config.hiddenDim, config.numExperts]));
    this.routerBias = tf.variable(tf.zeros([config.numExperts]));
  }

  public route(hidden: tf.Tensor2D, memory: HopeMemoryState): RoutingDecision {
    return tf.tidy(() => {
      const logits = tf.add(tf.matMul(hidden, this.routerKernel), this.routerBias);
      const weights = tf.softmax(logits, 1);

      const values = Array.from(weights.dataSync());
      const surprise = this.computeSurprise(values, memory);

      const sorted = values
        .map((value, index) => ({ value, index }))
        .sort((a, b) => b.value - a.value)
        .slice(0, this.config.topK);

      const selectedExperts = sorted.map(item => item.index);

      // Build a sparse weight tensor retaining only top-k experts
      const maskValues = values.map(() => 0);
      sorted.forEach(item => {
        maskValues[item.index] = item.value;
      });
      const maskedWeights = tf.tensor2d(maskValues, [1, this.config.numExperts]);

      return {
        weights: maskedWeights,
        surprise,
        selectedExperts
      };
    });
  }

  public combineReads(reads: tf.Tensor2D[], decision: RoutingDecision): tf.Tensor2D {
    return tf.tidy(() => {
      const paddedReads = reads.slice(0, this.config.numExperts);
      while (paddedReads.length < this.config.numExperts) {
        paddedReads.push(tf.zerosLike(paddedReads[0]));
      }
      const stacked = tf.stack(paddedReads, 0).reshape([this.config.numExperts, -1]);
      const weights = decision.weights.reshape([1, this.config.numExperts]);
      const weighted = tf.matMul(weights, stacked);
      return weighted.reshape([1, -1]);
    });
  }

  public getTrainableVariables(): tf.Variable[] {
    return [this.routerKernel, this.routerBias];
  }

  private computeSurprise(weights: number[], memory: HopeMemoryState): number {
    if (weights.length === 0) { return 0; }
    const entropy = -weights
      .filter(value => value > 0)
      .reduce((sum, value) => sum + value * Math.log(value), 0);
    const entropyNormalized = weights.length > 1 ? entropy / Math.log(weights.length) : entropy;
    const mean = weights.reduce((a, b) => a + b, 0) / weights.length;
    const variance = weights.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / weights.length;
    const surpriseValues = Array.from(memory.surpriseHistory.dataSync());
    const lastSurprise = surpriseValues[surpriseValues.length - 1] ?? 0;
    const novelty = memory.surpriseHistory.shape[0] === 0
      ? 1
      : Math.max(0.5, 1 + (weights[0] - lastSurprise));
    return (entropyNormalized * novelty) + (0.1 * variance);
  }
}
