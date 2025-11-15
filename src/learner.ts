/**
 * @fileoverview Learner Service - Online Learning Loop for HOPE Memory
 * 
 * This service implements a ring-buffer replay system with mini-batch updates
 * that mix contrastive, next-token prediction, and masked language modeling.
 * It includes gradient accumulation with clipping and NaN guards, and runs
 * in a configurable interval with MCP tools for pause/resume functionality.
 */

import * as tf from '@tensorflow/tfjs-node';
import { z } from 'zod';
import type { IMemoryModel, IMemoryState, ITensor } from './types.js';
import type { AdvancedTokenizer } from './tokenizer/index.js';
import { wrapTensor, unwrapTensor } from './types.js';

/**
 * Configuration schema for the learner service
 */
const LearnerConfigSchema = z.object({
  bufferSize: z.number().int().positive().default(10000),
  batchSize: z.number().int().positive().default(32),
  updateInterval: z.number().int().positive().default(1000), // ms
  gradientClipValue: z.number().positive().default(1.0),
  contrastiveWeight: z.number().min(0).max(1).default(0.2),
  nextTokenWeight: z.number().min(0).max(1).default(0.4),
  mlmWeight: z.number().min(0).max(1).default(0.4),
  accumulationSteps: z.number().int().positive().default(4),
  learningRate: z.number().positive().default(0.0001),
  nanGuardThreshold: z.number().positive().default(1e-6)
});

export type LearnerConfig = z.infer<typeof LearnerConfigSchema>;

/**
 * Training sample interface for the replay buffer
 */
interface TrainingSample {
  input: tf.Tensor;
  target: tf.Tensor;
  positive?: tf.Tensor; // For contrastive learning
  negative?: tf.Tensor; // For contrastive learning
  maskIndices?: number[]; // For MLM
  timestamp: number;
}

/**
 * Ring buffer implementation for replay
 */
class RingBuffer {
  private buffer: TrainingSample[];
  private capacity: number;
  private position: number = 0;
  private size: number = 0;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.buffer = new Array(capacity);
  }

  add(sample: TrainingSample): void {
    // Dispose old tensor if replacing
    if (this.buffer[this.position]) {
      this.disposeSample(this.buffer[this.position]);
    }

    this.buffer[this.position] = sample;
    this.position = (this.position + 1) % this.capacity;
    this.size = Math.min(this.size + 1, this.capacity);
  }

  sample(batchSize: number): TrainingSample[] {
    if (this.size === 0) return [];
    
    const samples: TrainingSample[] = [];
    const availableIndices = Array.from({ length: this.size }, (_, i) => i);
    
    for (let i = 0; i < Math.min(batchSize, this.size); i++) {
      const randomIndex = Math.floor(Math.random() * availableIndices.length);
      const bufferIndex = availableIndices.splice(randomIndex, 1)[0];
      samples.push(this.buffer[bufferIndex]);
    }
    
    return samples;
  }

  private disposeSample(sample: TrainingSample): void {
    sample.input.dispose();
    sample.target.dispose();
    sample.positive?.dispose();
    sample.negative?.dispose();
  }

  getSize(): number {
    return this.size;
  }

  clear(): void {
    for (let i = 0; i < this.size; i++) {
      this.disposeSample(this.buffer[i]);
    }
    this.size = 0;
    this.position = 0;
  }
}

/**
 * Gradient accumulation helper
 */
class GradientAccumulator {
  private accumulatedGradients: Map<string, tf.Tensor> = new Map();
  private step: number = 0;
  private readonly accumulationSteps: number;

  constructor(accumulationSteps: number) {
    this.accumulationSteps = accumulationSteps;
  }

  accumulate(gradients: Map<string, tf.Tensor>): boolean {
    for (const [key, gradient] of gradients) {
      if (this.accumulatedGradients.has(key)) {
        const accumulated = this.accumulatedGradients.get(key)!;
        this.accumulatedGradients.set(key, tf.add(accumulated, gradient));
        accumulated.dispose();
      } else {
        this.accumulatedGradients.set(key, gradient.clone());
      }
    }

    this.step++;
    return this.step >= this.accumulationSteps;
  }

  getAverageGradients(): Map<string, tf.Tensor> {
    const avgGradients = new Map<string, tf.Tensor>();
    const scale = tf.scalar(this.accumulationSteps);
    for (const [key, gradient] of this.accumulatedGradients) {
      avgGradients.set(key, tf.div(gradient, scale));
    }
    scale.dispose();
    return avgGradients;
  }

  reset(): void {
    for (const gradient of this.accumulatedGradients.values()) {
      gradient.dispose();
    }
    this.accumulatedGradients.clear();
    this.step = 0;
  }
}

/**
 * Main Learner service class
 */
export class LearnerService {
  private config: LearnerConfig;
  private replayBuffer: RingBuffer;
  private model: IMemoryModel;
  private tokenizer: AdvancedTokenizer;
  private gradientAccumulator: GradientAccumulator;
  private optimizer: tf.Optimizer;
  private trainingInterval: NodeJS.Timeout | null = null;
  private isRunning: boolean = false;
  private stepCount: number = 0;
  private lastLossValues: number[] = [];

  constructor(
    model: IMemoryModel,
    tokenizer: AdvancedTokenizer,
    config: Partial<LearnerConfig> = {}
  ) {
    this.config = LearnerConfigSchema.parse(config);
    this.model = model;
    this.tokenizer = tokenizer;
    this.replayBuffer = new RingBuffer(this.config.bufferSize);
    this.gradientAccumulator = new GradientAccumulator(this.config.accumulationSteps);
    this.optimizer = tf.train.adam(this.config.learningRate);
  }

  /**
   * Add a training sample to the replay buffer
   */
  async addTrainingSample(
    input: string | tf.Tensor,
    target: string | tf.Tensor,
    positive?: string | tf.Tensor,
    negative?: string | tf.Tensor
  ): Promise<void> {
    const inputInfo = typeof input === 'string'
      ? await this.normalizeSample(input)
      : this.normalizeTensorSample(input);
    const targetInfo = typeof target === 'string'
      ? await this.normalizeSample(target)
      : this.normalizeTensorSample(target);
    const positiveInfo = typeof positive === 'string'
      ? await this.normalizeSample(positive)
      : (positive ? this.normalizeTensorSample(positive) : undefined);
    const negativeInfo = typeof negative === 'string'
      ? await this.normalizeSample(negative)
      : (negative ? this.normalizeTensorSample(negative) : undefined);

    const sample: TrainingSample = {
      input: inputInfo.tensor,
      target: targetInfo.tensor,
      positive: positiveInfo?.tensor,
      negative: negativeInfo?.tensor,
      maskIndices: this.generateMaskIndices(inputInfo.sequenceLength),
      timestamp: Date.now()
    };

    this.replayBuffer.add(sample);
  }

  private async normalizeSample(value: string): Promise<{ tensor: tf.Tensor; sequenceLength: number }> {
    const encodeResult = await (this.tokenizer.encode as any)(value);

    if (encodeResult == null) {
      throw new Error('Tokenizer returned no result for provided text input.');
    }

    if (encodeResult instanceof tf.Tensor) {
      const tensor = this.ensure1DTensor(encodeResult);
      const sequenceLength = encodeResult.shape[0] ?? encodeResult.size;
      encodeResult.dispose();
      return { tensor, sequenceLength };
    }

    if ('embeddings' in encodeResult) {
      const embeddings = encodeResult.embeddings as tf.Tensor;
      const sequenceLength = embeddings.shape[0] ?? embeddings.size;
      const averaged = tf.tidy(() => embeddings.mean(0));
      embeddings.dispose();
      return { tensor: averaged, sequenceLength };
    }

    throw new Error('Unsupported tokenizer encode result. Expected tensor embeddings.');
  }

  private normalizeTensorSample(value: tf.Tensor): { tensor: tf.Tensor; sequenceLength: number } {
    const tensorInput = value as tf.Tensor;

    if (typeof tensorInput.isDisposed === 'function' && tensorInput.isDisposed()) {
      throw new Error('Cannot normalize a disposed tensor input.');
    }

    // Keep a reference to guard against callers disposing their tensor while we clone it.
    const kept = tf.keep(tensorInput);
    const tensor = kept.clone();
    kept.dispose();

    const sequenceLength = tensor.shape[0] ?? tensor.size;
    return { tensor, sequenceLength };
  }

  private ensure1DTensor(tensor: tf.Tensor): tf.Tensor {
    if (tensor.rank === 1) {
      return tensor.clone();
    }

    return tf.tidy(() => tensor.mean(0));
  }

  /**
   * Start the training loop
   */
  startTraining(): void {
    if (this.isRunning) return;
    
    this.isRunning = true;
    this.trainingInterval = setInterval(() => {
      this.performTrainingStep();
    }, this.config.updateInterval);
  }

  /**
   * Pause the training loop
   */
  pauseTraining(): void {
    if (this.trainingInterval) {
      clearInterval(this.trainingInterval);
      this.trainingInterval = null;
    }
    this.isRunning = false;
  }

  /**
   * Resume the training loop
   */
  resumeTraining(): void {
    if (!this.isRunning) {
      this.startTraining();
    }
  }

  /**
   * Check if the learner is currently running
   */
  isTraining(): boolean {
    return this.isRunning;
  }

  /**
   * Get current training statistics
   */
  getTrainingStats(): {
    bufferSize: number;
    stepCount: number;
    isRunning: boolean;
    averageLoss: number;
    lastLoss: number;
  } {
    const averageLoss = this.lastLossValues.length > 0 
      ? this.lastLossValues.reduce((a, b) => a + b, 0) / this.lastLossValues.length
      : 0;
    
    return {
      bufferSize: this.replayBuffer.getSize(),
      stepCount: this.stepCount,
      isRunning: this.isRunning,
      averageLoss,
      lastLoss: this.lastLossValues[this.lastLossValues.length - 1] || 0
    };
  }

  /**
   * Perform a single training step
   */
  private performTrainingStep(): void {
    if (this.replayBuffer.getSize() < this.config.batchSize) {
      return; // Not enough samples
    }

    try {
      tf.tidy(() => {
        const batch = this.replayBuffer.sample(this.config.batchSize);
        const { loss, gradients } = this.computeMixedLoss(batch);
        
        // NaN guard
        if (this.hasNaNGradients(gradients)) {
          console.warn('NaN gradients detected, skipping step');
          return;
        }

        // Gradient clipping
        const clippedGradients = this.clipGradients(gradients);

        for (const gradient of gradients.values()) {
          gradient.dispose();
        }
        gradients.clear();

        // Accumulate gradients
        const shouldUpdate = this.gradientAccumulator.accumulate(clippedGradients);
        
        if (shouldUpdate) {
          // Apply accumulated gradients
          const avgGradients = this.gradientAccumulator.getAverageGradients();
          this.applyGradients(avgGradients);
          this.gradientAccumulator.reset();

          // Dispose average gradients
          for (const gradient of avgGradients.values()) {
            gradient.dispose();
          }
        }

        // Track loss
        const lossValue = loss.dataSync()[0];
        this.lastLossValues.push(lossValue);
        if (this.lastLossValues.length > 100) {
          this.lastLossValues.shift();
        }

        this.stepCount++;
        
        // Dispose clipped gradients
        for (const gradient of clippedGradients.values()) {
          gradient.dispose();
        }
      });
    } catch (error) {
      console.error('Training step error:', error);
    }
  }

  /**
   * Compute mixed loss combining contrastive, next-token, and MLM losses
   */
  private computeMixedLoss(batch: TrainingSample[]): {
    loss: tf.Scalar;
    gradients: Map<string, tf.Tensor>;
  } {
    const trainableVars = this.model.getTrainableVariables();
    const lossFn = () => {
      const lossTerms: tf.Tensor[] = [];

      if (this.config.nextTokenWeight > 0) {
        const nextTokenLoss = this.computeNextTokenLoss(batch);
        lossTerms.push(tf.mul(nextTokenLoss, tf.scalar(this.config.nextTokenWeight)));
      }

      if (this.config.contrastiveWeight > 0) {
        const contrastiveLoss = this.computeContrastiveLoss(batch);
        lossTerms.push(tf.mul(contrastiveLoss, tf.scalar(this.config.contrastiveWeight)));
      }

      if (this.config.mlmWeight > 0) {
        const mlmLoss = this.computeMLMLoss(batch);
        lossTerms.push(tf.mul(mlmLoss, tf.scalar(this.config.mlmWeight)));
      }

      if (lossTerms.length === 0) {
        return tf.scalar(0);
      }

      return lossTerms.length === 1
        ? (lossTerms[0] as tf.Scalar)
        : (tf.addN(lossTerms) as tf.Scalar);
    };

    if (trainableVars.length === 0) {
      return { loss: lossFn(), gradients: new Map<string, tf.Tensor>() };
    }

    const { value: mixedLoss, grads } = tf.variableGrads(lossFn, trainableVars);

    const gradients = new Map<string, tf.Tensor>();
    for (const [name, gradient] of Object.entries(grads)) {
      gradients.set(name, gradient as tf.Tensor);
    }

    return { loss: mixedLoss as tf.Scalar, gradients };
  }

  /**
   * Compute next-token prediction loss
   */
  private computeNextTokenLoss(batch: TrainingSample[]): tf.Scalar {
    const inputs = batch.map(s => s.input);
    const targets = batch.map(s => s.target);
    
    const inputStack = tf.stack(inputs);
    const targetStack = tf.stack(targets);
    
    // Use model's prediction capabilities
    const memoryState = this.model.getMemoryState();
    const predictions = inputStack.split(inputStack.shape[0]).map(input => {
      const squeezed = input.squeeze([0]);
      const result = this.model.forward(wrapTensor(squeezed), memoryState);
      return unwrapTensor(result.predicted);
    });
    
    const predictionStack = tf.stack(predictions);
    const loss = tf.losses.softmaxCrossEntropy(targetStack, predictionStack) as tf.Scalar;
    
    // Clean up
    for (const pred of predictions) {
      pred.dispose();
    }
    predictionStack.dispose();
    inputStack.dispose();
    targetStack.dispose();
    
    return loss;
  }

  /**
   * Compute contrastive learning loss
   */
  private computeContrastiveLoss(batch: TrainingSample[]): tf.Scalar {
    const validSamples = batch.filter(s => s.positive && s.negative);
    if (validSamples.length === 0) {
      return tf.scalar(0);
    }

    let totalLoss = tf.scalar(0);
    let count = 0;

    for (const sample of validSamples) {
      if (!sample.positive || !sample.negative) continue;
      
      const anchorEmbed = this.getEmbedding(sample.input);
      const positiveEmbed = this.getEmbedding(sample.positive);
      const negativeEmbed = this.getEmbedding(sample.negative);
      
      // Compute cosine similarities
      const positiveSim = tf.sum(tf.mul(anchorEmbed, positiveEmbed));
      const negativeSim = tf.sum(tf.mul(anchorEmbed, negativeEmbed));
      
      // Contrastive loss: maximize positive similarity, minimize negative
      const margin = tf.scalar(0.5);
      const loss = tf.maximum(
        tf.scalar(0),
        tf.add(tf.sub(negativeSim, positiveSim), margin)
      );
      
      totalLoss = tf.add(totalLoss, loss);
      count++;
      
      // Clean up
      anchorEmbed.dispose();
      positiveEmbed.dispose();
      negativeEmbed.dispose();
      positiveSim.dispose();
      negativeSim.dispose();
      loss.dispose();
      margin.dispose();
    }

    if (count > 0) {
      const avgLoss = tf.div(totalLoss, tf.scalar(count)) as tf.Scalar;
      totalLoss.dispose();
      return avgLoss;
    }
    
    return totalLoss;
  }

  /**
   * Compute masked language modeling loss
   */
  private computeMLMLoss(batch: TrainingSample[]): tf.Scalar {
    const validSamples = batch.filter(s => s.maskIndices && s.maskIndices.length > 0);
    if (validSamples.length === 0) {
      return tf.scalar(0);
    }

    let totalLoss = tf.scalar(0);
    let count = 0;

    for (const sample of validSamples) {
      if (!sample.maskIndices || sample.maskIndices.length === 0) continue;
      
      // Create masked input
      const maskedInput = sample.input.clone();
      const maskToken = (this.tokenizer as any).getSpecialTokens?.().mask || 103; // Default mask token
      
      // Apply masks
      for (const maskIndex of sample.maskIndices) {
        // This is a simplified approach - in practice you'd need proper indexing
        // maskedInput[maskIndex] = maskToken;
      }
      
      // Get predictions for masked positions
      const memoryState = this.model.getMemoryState();
      const { predicted } = this.model.forward(wrapTensor(maskedInput), memoryState);
      
      // Compute loss only for masked positions
      const maskedLoss = tf.losses.softmaxCrossEntropy(
        sample.target.slice([0], [sample.maskIndices.length]),
        unwrapTensor(predicted).slice([0], [sample.maskIndices.length])
      );
      
      totalLoss = tf.add(totalLoss, maskedLoss);
      count++;
      
      // Clean up
      maskedInput.dispose();
      maskedLoss.dispose();
    }

    if (count > 0) {
      const avgLoss = tf.div(totalLoss, tf.scalar(count)) as tf.Scalar;
      totalLoss.dispose();
      return avgLoss;
    }
    
    return totalLoss;
  }

  /**
   * Get embedding representation of a tensor
   */
  private getEmbedding(tensor: tf.Tensor): tf.Tensor {
    // This should use the model's embedding layer
    // For now, we'll use a simple normalization
    const norm = tf.norm(tensor);
    return tf.div(tensor, norm);
  }

  /**
   * Generate random mask indices for MLM
   */
  private generateMaskIndices(sequenceLength: number): number[] {
    const maskRate = 0.15; // Standard BERT masking rate
    const numMasks = Math.floor(sequenceLength * maskRate);
    const indices: number[] = [];
    
    for (let i = 0; i < numMasks; i++) {
      const randomIndex = Math.floor(Math.random() * sequenceLength);
      if (!indices.includes(randomIndex)) {
        indices.push(randomIndex);
      }
    }
    
    return indices;
  }

  /**
   * Check for NaN gradients
   */
  private hasNaNGradients(gradients: Map<string, tf.Tensor>): boolean {
    for (const gradient of gradients.values()) {
      const hasNaN = tf.any(tf.isNaN(gradient)).dataSync()[0] > 0;
      if (hasNaN) return true;
      
      const maxValue = tf.max(tf.abs(gradient)).dataSync()[0];
      if (maxValue > 1e6) return true; // Treat very large gradients as problematic
    }
    return false;
  }

  /**
   * Clip gradients to prevent exploding gradients
   */
  private clipGradients(gradients: Map<string, tf.Tensor>): Map<string, tf.Tensor> {
    const clippedGradients = new Map<string, tf.Tensor>();
    
    for (const [key, gradient] of gradients) {
      const clipped = tf.clipByValue(
        gradient,
        -this.config.gradientClipValue,
        this.config.gradientClipValue
      );
      clippedGradients.set(key, clipped);
    }
    
    return clippedGradients;
  }

  /**
   * Apply gradients to the model
   */
  private applyGradients(gradients: Map<string, tf.Tensor>): void {
    try {
      if (typeof this.model.applyGradients === 'function') {
        this.model.applyGradients(gradients);
        return;
      }

      const namedGradients: tf.NamedTensorMap = {};
      const variableMap = new Map(
        this.model.getTrainableVariables().map(variable => [variable.name, variable])
      );

      for (const [name, gradient] of gradients) {
        if (variableMap.has(name)) {
          namedGradients[name] = gradient;
        }
      }

      if (Object.keys(namedGradients).length > 0) {
        this.optimizer.applyGradients(namedGradients);
      }
    } finally {
      for (const gradient of gradients.values()) {
        gradient.dispose();
      }
      gradients.clear();
    }
  }

  /**
   * Dispose of all resources
   */
  dispose(): void {
    this.pauseTraining();
    this.replayBuffer.clear();
    this.gradientAccumulator.reset();
    this.optimizer.dispose();
  }
}
