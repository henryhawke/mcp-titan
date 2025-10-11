import './utils/polyfills.js';

/* eslint-disable @typescript-eslint/no-unused-vars */
/**
 * @fileovertitle Titan Memory Model 2.0 - Neural Memory Architecture with Transformer-XL Inspired Mechanisms
 */

import * as tf from '@tensorflow/tfjs-node';
import type { ITensor, IMemoryState, ISurpriseMetrics, IAttentionBlock, IMemoryUpdateResult, IModelGradients, IMemoryModel, ITelemetryData, IHierarchicalMemoryStateInternal, IQuantizedMemoryStateInternal, IMemoryPromotionRules, IRetrievalWeights } from './types.js';
import { unwrapTensor, wrapTensor, TensorError, MemoryError, type IHierarchicalMemoryState, type IExtendedMemoryState, type IQuantizedMemoryState } from './types.js';
import * as fs from 'fs/promises';
import { z } from 'zod';
import { VectorProcessor, SafeTensorOps } from './utils.js';
import { AdvancedTokenizer, type TokenizerConfig } from './tokenizer/index.js';
import { MemoryPruner, type PruningConfig, type PruningResult, createDefaultPruningConfig } from './pruning.js';
import type { TfIdfVectorizer } from './tfidf.js';
import { StructuredLogger, LogLevel } from './logging.js';

// Add telemetry implementation
class ModelTelemetry {
  private static instance: ModelTelemetry;
  private telemetryData: ITelemetryData[] = [];
  private maxEntries = 1000;
  private enabled = true;

  private constructor() {
    // Constructor initialization
  }

  public static getInstance(): ModelTelemetry {
    if (!ModelTelemetry.instance) {
      ModelTelemetry.instance = new ModelTelemetry();
    }
    return ModelTelemetry.instance;
  }

  public recordOperation(operation: string, metrics?: Record<string, number>): () => void {
    if (!this.enabled) { return () => { }; }

    const startTime = performance.now();
    const startMemory = tf.memory();

    return () => {
      const endTime = performance.now();
      const endMemory = tf.memory();

      const telemetryEntry: ITelemetryData = {
        timestamp: Date.now(),
        operation,
        durationMs: endTime - startTime,
        memoryUsage: {
          numTensors: endMemory.numTensors,
          numBytes: endMemory.numBytes,
          unreliable: !!endMemory.unreliable
        },
        metrics
      };

      this.telemetryData.push(telemetryEntry);

      // Trim if needed
      if (this.telemetryData.length > this.maxEntries) {
        this.telemetryData = this.telemetryData.slice(-this.maxEntries);
      }
    };
  }

  public recordError(operation: string, error: Error): void {
    if (!this.enabled) { return; }

    const telemetryEntry: ITelemetryData = {
      timestamp: Date.now(),
      operation,
      durationMs: 0,
      memoryUsage: {
        numTensors: tf.memory().numTensors,
        numBytes: tf.memory().numBytes,
        unreliable: !!tf.memory().unreliable
      },
      error: {
        name: error.name,
        message: error.message,
        stack: error.stack
      }
    };

    this.telemetryData.push(telemetryEntry);

    // Trim if needed
    if (this.telemetryData.length > this.maxEntries) {
      this.telemetryData = this.telemetryData.slice(-this.maxEntries);
    }
  }

  public getMetrics(): ITelemetryData[] {
    return [...this.telemetryData];
  }

  public getAverageMetrics(operation: string, lastN = 10): Record<string, number> {
    const relevantEntries = this.telemetryData
      .filter(entry => entry.operation === operation)
      .slice(-lastN);

    if (relevantEntries.length === 0) {
      return {};
    }

    const avgDuration = relevantEntries.reduce((sum, entry) => sum + entry.durationMs, 0) / relevantEntries.length;
    const avgTensors = relevantEntries.reduce((sum, entry) => sum + entry.memoryUsage.numTensors, 0) / relevantEntries.length;
    const avgBytes = relevantEntries.reduce((sum, entry) => sum + entry.memoryUsage.numBytes, 0) / relevantEntries.length;

    const result: Record<string, number> = {
      avgDurationMs: avgDuration,
      avgTensors: avgTensors,
      avgBytes: avgBytes
    };

    // Add custom metrics if they exist
    if (relevantEntries[0].metrics) {
      Object.keys(relevantEntries[0].metrics).forEach(metricKey => {
        result[`avg${metricKey}`] = relevantEntries.reduce(
          (sum, entry) => sum + (entry.metrics?.[metricKey] ?? 0),
          0
        ) / relevantEntries.length;
      });
    }

    return result;
  }

  public enable(): void {
    this.enabled = true;
  }

  public disable(): void {
    this.enabled = false;
  }

  public clear(): void {
    this.telemetryData = [];
  }
}

// Add polyfill for isNullOrUndefined
const isNullOrUndefined = (value: any): value is null | undefined => value === null || value === undefined;

// Patch TensorFlow.js Node backend
const originalCreateTensorsTypeOpAttr = (tf as any).backend().createTensorsTypeOpAttr;
if (originalCreateTensorsTypeOpAttr) {
  (tf as any).backend().createTensorsTypeOpAttr = function (...args: any[]) {
    // Replace any usage of isNullOrUndefined with our polyfill
    const patchedArgs = args.map(arg => {
      if (typeof arg === 'function' && arg.name === 'isNullOrUndefined') {
        return isNullOrUndefined;
      }
      return arg;
    });
    return originalCreateTensorsTypeOpAttr.apply(this, patchedArgs);
  };
}

// Enhanced configuration schema
const ModelConfigSchema = z.object({
  inputDim: z.number().int().positive().default(768),
  hiddenDim: z.number().int().positive().default(512),
  memoryDim: z.number().int().positive().default(1024),
  transformerLayers: z.number().int().positive().max(12).default(6),
  numHeads: z.number().int().positive().default(8),
  ffDimension: z.number().int().positive().default(2048),
  dropoutRate: z.number().min(0).max(0.9).default(0.1),
  maxSequenceLength: z.number().int().positive().default(512),
  memorySlots: z.number().int().positive().default(5000),
  similarityThreshold: z.number().min(0).max(1).default(0.65),
  surpriseDecay: z.number().min(0).max(1).default(0.9),
  pruningInterval: z.number().int().positive().default(1000),
  gradientClip: z.number().positive().default(1.0),
  learningRate: z.number().positive().default(0.001),
  vocabSize: z.number().int().positive().default(50000),
  decayRate: z.number().min(0).max(1).default(0.9),
  useRotaryEmbeddings: z.boolean().default(false),
  useMultiQueryAttention: z.boolean().default(false),
  useHierarchicalMemory: z.boolean().default(false),
  useSubwordTokenization: z.boolean().default(false),
  useApproximateNearestNeighbors: z.boolean().default(false),
  useGatedLinearUnits: z.boolean().default(false),
  useSwiGLU: z.boolean().default(false),
  useMemoryDistillation: z.boolean().default(false),
  enableQuantization: z.boolean().default(false),
  enableContrastiveLearning: z.boolean().default(false),
  enableAdaptiveComputationTime: z.boolean().default(false),
  enableInformationGainPruning: z.boolean().default(false),
  enableEpisodicSemanticDistinction: z.boolean().default(false),
  enableJITCompilation: z.boolean().default(false),
  enableSparseAttention: z.boolean().default(false),
  sparsityRate: z.number().min(0).max(0.99).default(0.8),
  enableTelemetry: z.boolean().default(true),
  actConfig: z.object({
    maxPonderSteps: z.number().int().positive().default(10),
    ponderCost: z.number().min(0).max(1).default(0.01)
  }).optional().default({}),
  contrastiveWeight: z.number().min(0).max(1).default(0.1),
  forgettingGateInit: z.number().min(0).max(1).default(0.1)
});

export type TitanMemoryConfig = z.infer<typeof ModelConfigSchema>;

interface WeightInfo {
  shape: number[];
  dtype: string;
}

// Add this near the top of the file, after imports but before class definitions
/**
 * Safe logging function that won't interfere with MCP communication
 * @param message The message to log
 */
const safeLog = (message: string, metadata?: Record<string, any>) => {
  StructuredLogger.getInstance().info('model', message, metadata);
};

// Helper to deeply flatten and filter to number[]
function flattenToNumberArray(arr: any): number[] {
  return (arr as any[]).flat(Infinity).filter((v): v is number => typeof v === 'number');
}

export class TitanMemoryModel implements IMemoryModel {
  private config: TitanMemoryConfig = ModelConfigSchema.parse({});
  private transformerStack: tf.LayersModel[] = [];
  private memoryProjector!: tf.LayersModel;
  private similarityNetwork!: tf.LayersModel;
  private optimizer!: tf.Optimizer;
  private stepCount = 0;
  private vocabulary = new Map<string, number>();
  private reverseVocabulary = new Map<number, string>();

  // Enhanced memory state with temporal dynamics
  private memoryState: IMemoryState = {
    shortTerm: tf.zeros([0]),
    longTerm: tf.zeros([0]),
    meta: tf.zeros([0]),
    timestamps: tf.zeros([0]),
    accessCounts: tf.zeros([0]),
    surpriseHistory: tf.zeros([0])
  };

  // Extended memory state with hierarchical tiers and episodic/semantic distinction
  private extendedMemoryState: IExtendedMemoryState | null = null;

  // Memory promotion/demotion rules
  private promotionRules: IMemoryPromotionRules = {
    workingToShortTerm: {
      accessThreshold: 3,
      timeThreshold: 30000, // 30 seconds
      importanceThreshold: 0.6
    },
    shortTermToLongTerm: {
      accessThreshold: 5,
      timeThreshold: 300000, // 5 minutes
      importanceThreshold: 0.8,
      reinforcementCount: 3
    },
    episodicToSemantic: {
      generalityThreshold: 0.7,
      confidenceThreshold: 0.85,
      abstractionLevel: 0.6
    },
    demotionRules: {
      lowAccessPenalty: 0.1,
      ageDecayRate: 0.99,
      forgettingThreshold: 0.1
    }
  };

  // Retrieval weights for different memory types
  private retrievalWeights: IRetrievalWeights = {
    episodic: {
      recencyWeight: 0.6,
      contextWeight: 0.3,
      emotionalWeight: 0.1
    },
    semantic: {
      similarityWeight: 0.5,
      confidenceWeight: 0.3,
      generalityWeight: 0.2
    },
    combined: {
      episodicBias: 0.4,
      semanticBias: 0.6,
      tierPreference: [0.8, 0.6, 0.4] // working, short-term, long-term
    }
  };

  // Memory statistics tracking
  private memoryStats: {
    promotions: { recent: number; total: number };
    demotions: { recent: number; total: number };
    lastStatsUpdate: number;
  } = {
      promotions: { recent: 0, total: 0 },
      demotions: { recent: 0, total: 0 },
      lastStatsUpdate: Date.now()
    };

  // Add hierarchical memory properties
  private hierarchicalLevels = 3;
  private hierarchicalMemory: IHierarchicalMemoryStateInternal | null = null;

  // Add quantization properties
  private quantizedMemory: IQuantizedMemoryStateInternal | null = null;
  private quantizationBits = 8;
  private quantizationRanges: Array<{ min: number; max: number }> = [];

  // Momentum buffer for LMM updates (Equation 33)
  private momentumBuffer: tf.Tensor2D | null = null;
  private forgettingGateVariable: tf.Variable | null = null;

  // Add contrastive learning properties
  private contrastiveBuffer: tf.Tensor[] = [];
  private contrastiveBufferSize = 128;
  private contrastiveTemperature = 0.07;

  // Add encoder and decoder properties
  private encoder!: tf.LayersModel;
  private decoder!: tf.LayersModel;
  private tokenizer: any = null;
  private advancedTokenizer: AdvancedTokenizer | null = null;
  private vocabSize = 10000;
  private useLegacyCharEncoding = false;

  // HNSW index for approximate nearest neighbors
  private hnswIndex: any = null;

  private vectorProcessor: VectorProcessor = VectorProcessor.getInstance();

  // Memory pruning system
  private memoryPruner: MemoryPruner;

  // TF-IDF fallback for untrained encoder
  private tfidfVectorizer: TfIdfVectorizer | null = null;
  private fallbackDocuments: string[] = [];

  private logger: StructuredLogger = StructuredLogger.getInstance();

  // Add error handling wrapper
  private withErrorHandling<T>(operation: string, fn: () => T): T {
    const telemetry = ModelTelemetry.getInstance();
    const endTelemetry = telemetry.recordOperation(operation);

    try {
      const result = fn(); // Execute function first
      endTelemetry(); // Record telemetry on success
      return result; // Return result
    } catch (error) {
      const err = error as Error; // Cast error to Error
      this.logger.error('operation', `Error in operation ${operation}`, err);

      // Log to telemetry, passing the casted Error object
      telemetry.recordError(operation, err);

      // Attempt recovery based on error type
      if (err instanceof TensorError) {
        this.resetGradients();
        this.logger.info('operation', `Recovered from tensor error in ${operation} by resetting gradients`);
      } else if (err instanceof MemoryError) {
        this.initializeMemoryState();
        this.logger.info('operation', `Recovered from memory error in ${operation} by reinitializing memory state`);
      }

      // Always end telemetry, even on error
      endTelemetry();

      throw err; // Re-throw the original error
    }
    // Removed finally block
  }

  // Add async error handling wrapper
  private async withErrorHandlingAsync<T>(operation: string, fn: () => Promise<T>): Promise<T> {
    const telemetry = ModelTelemetry.getInstance();
    const endTelemetry = telemetry.recordOperation(operation);

    try {
      const result = await fn(); // Execute async function
      endTelemetry(); // Record telemetry on success
      return result; // Return result
    } catch (error) {
      const err = error as Error; // Cast error to Error
      this.logger.error('operation', `Error in operation ${operation}`, err);

      // Log to telemetry, passing the casted Error object
      telemetry.recordError(operation, err);

      // Attempt recovery based on error type
      if (err instanceof TensorError) {
        this.resetGradients();
        this.logger.info('operation', `Recovered from tensor error in ${operation} by resetting gradients`);
      } else if (err instanceof MemoryError) {
        this.initializeMemoryState();
        this.logger.info('operation', `Recovered from memory error in ${operation} by reinitializing memory state`);
      }

      // Always end telemetry, even on error
      endTelemetry();

      throw err; // Re-throw the original error
    }
  }

  constructor(config?: Partial<TitanMemoryConfig>) {
    // Initialize with empty config first
    this.config = ModelConfigSchema.parse(config || {});

    // Initialize memory pruner with configuration
    this.memoryPruner = new MemoryPruner({
      keepPercentage: 0.7,
      minMemoriesToKeep: 100,
      maxCapacity: this.config.memorySlots,
      entropyWeight: 1.0,
      surpriseWeight: 1.2,
      redundancyWeight: 0.8,
      enableDistillation: true
    });

    // Initialize tokenizer based on configuration (async, will be handled during initialize())
    this.initializeTokenizer().catch(error => {
      this.logger.warn('tokenizer', 'Advanced tokenizer failed, falling back to legacy mode', { error });
      this.useLegacyCharEncoding = true;
    });
  }

  /**
   * Initialize tokenizer (advanced BPE or legacy character-based)
   */
  private async initializeTokenizer(): Promise<void> {
    const tokenizerConfig: TokenizerConfig = {
      vocabSize: this.config.vocabSize,
      embeddingDim: Math.min(512, Math.max(256, this.config.inputDim)),
      hiddenSize: this.config.hiddenDim,
      maxSequenceLength: this.config.maxSequenceLength,
      useLegacyCharMode: this.useLegacyCharEncoding,
      enableBootstrapping: true,
      useCharFallback: true
    };

    this.advancedTokenizer = new AdvancedTokenizer(tokenizerConfig);

    // Initialize the advanced tokenizer
    try {
      await this.advancedTokenizer.initialize();
      this.logger.info('tokenizer', 'Advanced tokenizer initialized successfully');
    } catch (error) {
      this.logger.warn('tokenizer', 'Failed to initialize advanced tokenizer, falling back to legacy mode', { error });
      this.useLegacyCharEncoding = true;
      this.advancedTokenizer.setLegacyMode(true);
    }

    // Keep legacy tokenizer for backward compatibility
    this.tokenizer = {
      encode: (text: string) => Array.from(text).map(c => c.charCodeAt(0) % this.vocabSize),
      decode: (tokens: number[]) => tokens.map(t => String.fromCharCode(t)).join('')
    };
  }

  private async initializeBackend(): Promise<void> {
    try {
      // Ensure TensorFlow.js is properly initialized
      await tf.ready();

      // Set the backend explicitly
      await tf.setBackend('tensorflow');

      // Double check backend is set and ready
      const backend = tf.getBackend();
      if (!backend) {
        throw new Error('TensorFlow backend not initialized');
      }

      // Initialize components after backend is ready
      this.initializeComponents();
      this.initializeMemoryState();

      const message = `TensorFlow backend initialized: ${backend}`;
      this.logger.info('backend.initialize', message);
    } catch (error) {
      this.logger.error('backend.initialize', 'Error initializing TensorFlow backend', error as Error);
      throw error;
    }
  }

  private initializeVocabulary(): void {
    // Initialize with special tokens
    this.vocabulary.clear();
    this.reverseVocabulary.clear();

    this.vocabulary.set('[PAD]', 0);
    this.vocabulary.set('[UNK]', 1);
    this.vocabulary.set('[CLS]', 2);
    this.vocabulary.set('[SEP]', 3);

    // Add basic characters and common tokens
    const basicChars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-_\'"`()[]{}:;/\\+=<>'.split('');
    basicChars.forEach((char, i) => {
      this.vocabulary.set(char, i + 4);
    });

    // Create reverse mapping
    this.vocabulary.forEach((value, key) => {
      this.reverseVocabulary.set(value, key);
    });
  }

  /**
   * Creates encoder model for processing inputs
   */
  private createEncoder(): tf.LayersModel {
    return this.withErrorHandling('createEncoder', () => {
      const inputShape = [this.config.inputDim];
      const embeddingSize = this.config.memoryDim;

      // Create sequential model
      const model = tf.sequential();

      // Add layers
      model.add(tf.layers.dense({
        inputShape,
        units: embeddingSize * 2,
        activation: 'relu',
        kernelInitializer: 'glorotNormal'
      }));

      // Add dropout for regularization
      model.add(tf.layers.dropout({ rate: 0.2 }));

      // Add final embedding layer
      model.add(tf.layers.dense({
        units: embeddingSize,
        activation: 'tanh',
        kernelInitializer: 'glorotNormal'
      }));

      return model;
    });
  }

  /**
   * Creates decoder model for generating outputs
   */
  private createDecoder(): tf.LayersModel {
    return this.withErrorHandling('createDecoder', () => {
      // Input is concatenated embedding and memory
      const inputShape = [this.config.memoryDim * 2];
      const outputDim = this.config.inputDim;

      // Create sequential model
      const model = tf.sequential();

      // Add layers
      model.add(tf.layers.dense({
        inputShape,
        units: this.config.memoryDim * 2,
        activation: 'relu',
        kernelInitializer: 'glorotNormal'
      }));

      // Add dropout for regularization
      model.add(tf.layers.dropout({ rate: 0.2 }));

      // Add final output layer
      model.add(tf.layers.dense({
        units: outputDim,
        activation: 'linear',
        kernelInitializer: 'glorotNormal'
      }));

      return model;
    });
  }

  /**
   * Encodes text input to tensor using advanced BPE tokenizer
   * @param text The text to encode
   * @returns The encoded tensor
   */
  public async encodeText(text: string): Promise<tf.Tensor1D> {
    // Use advanced tokenizer if available, otherwise fall back to legacy mode
    if (this.advancedTokenizer && !this.useLegacyCharEncoding) {
      try {
        const tokenizationResult = await this.advancedTokenizer.encode(text, {
          maxLength: this.config.maxSequenceLength,
          padding: true,
          truncation: true
        });

        // Convert 2D embeddings to 1D by taking mean across sequence dimension
        const meanResult = tf.mean(tokenizationResult.embeddings, 0);
        const flattened = tf.reshape(meanResult, [-1]) as tf.Tensor1D;

        // Dispose of intermediate tensors
        tokenizationResult.embeddings.dispose();
        tokenizationResult.attentionMask.dispose();

        return this.normalizeToInputDim(flattened);
      } catch (error) {
        this.logger.warn('tokenizer', 'Advanced tokenizer failed, falling back to legacy mode', { error });
        this.useLegacyCharEncoding = true;
        if (this.advancedTokenizer) {
          this.advancedTokenizer.setLegacyMode(true);
        }
      }
    }

    // TF-IDF fallback if encoder is untrained or vectorProcessor unavailable
    if (this.tfidfVectorizer && this.fallbackDocuments.length > 0) {
      try {
        const tfidfVector = this.tfidfVectorizer.transform([text]);
        const vectorArray = await tfidfVector.data();

        // Pad or truncate to match config.maxSequenceLength
        const targetLength = this.config.maxSequenceLength;
        const resultArray = new Float32Array(targetLength);
        const copyLength = Math.min(vectorArray.length, targetLength);

        for (let i = 0; i < copyLength; i++) {
          resultArray[i] = vectorArray[i];
        }

        tfidfVector.dispose();
        return this.normalizeToInputDim(tf.tensor1d(resultArray));
      } catch (error) {
        this.logger.warn('tfidf', 'TF-IDF fallback failed', { error });
      }
    }

    // Fallback to VectorProcessor or legacy character encoding
    if (this.vectorProcessor) {
      const tokenData = this.vectorProcessor.encodeText(text);
      let result: tf.Tensor1D;

      if (tokenData instanceof Promise) {
        result = (await tokenData) as tf.Tensor1D;
      } else {
        result = tokenData as tf.Tensor1D;
      }

      if (result.shape.length !== 1) {
        throw new Error('Encoded tensor has unexpected shape');
      }
      return this.normalizeToInputDim(result);
    } else {
      // Final fallback: simple character encoding
      const tokens = this.tokenizer.encode(text);
      const paddedTokens = tokens.slice(0, this.config.maxSequenceLength);
      while (paddedTokens.length < this.config.maxSequenceLength) {
        paddedTokens.push(0);
      }
      const legacyVector = tf.tensor1d(paddedTokens.map((t: number) => t / this.vocabSize));
      return this.normalizeToInputDim(legacyVector);
    }
  }

  /**
   * Ensure encoded vectors match Titan's input dimensionality (d_in).
   * This mirrors the fixed-width assumption from Equation 32.
   */
  private normalizeToInputDim(vector: tf.Tensor1D): tf.Tensor1D {
    const target = this.config.inputDim;
    const source = vector.dataSync();
    const adjusted = new Float32Array(target);
    const copyLength = Math.min(source.length, target);

    for (let i = 0; i < copyLength; i++) {
      adjusted[i] = source[i];
    }

    vector.dispose();
    return tf.tensor1d(adjusted);
  }

  /**
   * Initializes the model
   * @param config Configuration options
   */
  public async initialize(config?: Partial<TitanMemoryConfig>): Promise<void> {
    return this.withErrorHandlingAsync('initialize', async () => {
      if (config) {
        this.config = { ...this.config, ...config };
      }
      this.encoder = this.createEncoder();
      this.decoder = this.createDecoder();
      const learningRate = this.config.learningRate || 0.001;
      this.optimizer = tf.train.adam(learningRate);
      this.initializeMemoryState();
      // Remove custom tokenizer, rely on VectorProcessor
      if ((this.config as any).enableTextProcessing) {
        if (!this.vectorProcessor) {
          throw new Error('VectorProcessor not initialized');
        }
      }
    });
  }

  /**
   * Retrieves memory based on query
   */
  private retrieveFromMemory(query: ITensor, type: 'episodic' | 'semantic' = 'episodic'): ITensor {
    return this.withErrorHandling('retrieveFromMemory', () => {
      const extendedState = this.extendedMemoryState;
      if (!extendedState) {
        const queryTensor = unwrapTensor(query) as tf.Tensor;
        const queryMatrix = queryTensor.rank === 1 ? queryTensor.reshape([1, queryTensor.shape[0]]) : queryTensor as tf.Tensor2D;
        const attention = this.computeMemoryAttention(queryMatrix);
        return wrapTensor(attention.values);
      }

      const memorySource = type === 'episodic' ? extendedState.episodicMemory : extendedState.semanticMemory;
      // Calculate similarity or recency for retrieval
      let weights: tf.Tensor;

      if (type === 'episodic') {
        const weightsConfig = this.retrievalWeights.episodic;
        const similarities = tf.matMul(memorySource, unwrapTensor(query).reshape([-1, 1]), false, true);
        const recencyScores = tf.sub(tf.scalar(Date.now()), extendedState.episodicTimestamps);
        const weightedSum = tf.add(
          tf.mul(similarities, weightsConfig.contextWeight),
          tf.mul(recencyScores, weightsConfig.recencyWeight)
        );
        weights = tf.softmax(weightedSum);
      } else {
        const weightsConfig = this.retrievalWeights.semantic;
        const similarities = tf.matMul(memorySource, unwrapTensor(query).reshape([-1, 1]), false, true);
        const confidenceScores = extendedState.semanticConfidence;
        const weightedSum = tf.add(
          tf.mul(similarities, weightsConfig.similarityWeight),
          tf.mul(confidenceScores, weightsConfig.confidenceWeight)
        );
        weights = tf.softmax(weightedSum);
      }

      // Retrieve and weigh memories
      const weightedMemory = tf.matMul(weights, memorySource, true, false);

      // Update access counts for memory management
      if (type === 'episodic') {
        const addResult = extendedState.episodicAccessCounts.add(weights);
        const newAccessCounts = tf.reshape(addResult, [-1]) as tf.Tensor1D;
        tf.dispose(extendedState.episodicAccessCounts);
        tf.dispose(addResult);
        extendedState.episodicAccessCounts = newAccessCounts;
      } else {
        const addResult = extendedState.semanticAccessCounts.add(weights);
        const newAccessCounts = tf.reshape(addResult, [-1]) as tf.Tensor1D;
        tf.dispose(extendedState.semanticAccessCounts);
        tf.dispose(addResult);
        extendedState.semanticAccessCounts = newAccessCounts;
      }

      // Clean up
      tf.dispose(weights);

      return wrapTensor(weightedMemory);
    });
  }

  private initializeComponents(): void {
    // Initialize transformer stack
    this.transformerStack = [];
    for (let i = 0; i < this.config.transformerLayers; i++) {
      const layer = tf.sequential({
        layers: [
          tf.layers.dense({
            units: this.config.hiddenDim,
            inputShape: [this.config.inputDim],
            activation: 'linear',
            useBias: true,
            kernelInitializer: 'glorotNormal',
            biasInitializer: 'zeros'
          }),
          tf.layers.layerNormalization(),
          tf.layers.dense({
            units: this.config.ffDimension,
            activation: 'elu',
            kernelInitializer: 'glorotNormal',
            biasInitializer: 'zeros'
          }),
          tf.layers.dropout({ rate: this.config.dropoutRate }),
          tf.layers.dense({
            units: this.config.hiddenDim,
            kernelInitializer: 'glorotNormal',
            biasInitializer: 'zeros'
          }),
          tf.layers.layerNormalization()
        ]
      });
      this.transformerStack.push(layer);
    }

    // Initialize memory projector
    this.memoryProjector = tf.sequential({
      layers: [
        tf.layers.dense({
          units: this.config.memoryDim,
          inputShape: [this.config.hiddenDim],
          activation: 'tanh',
          kernelInitializer: 'glorotNormal',
          biasInitializer: 'zeros'
        }),
        tf.layers.layerNormalization()
      ]
    });

    // Initialize similarity network
    this.similarityNetwork = tf.sequential({
      layers: [
        tf.layers.dense({
          units: this.config.hiddenDim,
          inputShape: [this.config.memoryDim],
          activation: 'relu',
          kernelInitializer: 'glorotNormal',
          biasInitializer: 'zeros'
        }),
        tf.layers.dense({
          units: 1,
          activation: 'sigmoid',
          kernelInitializer: 'glorotNormal',
          biasInitializer: 'zeros'
        })
      ]
    });

    // Initialize optimizer
    this.optimizer = tf.train.adam(this.config.learningRate);
  }

  private initializeMemoryState(): void {
    tf.tidy(() => {
      const memorySlots = this.config.memorySlots;
      const embeddingSize = this.config.memoryDim;
      const currentTime = Date.now();

      // Prepare forgetting gate variable if enabled
      let forgettingGateTensor: tf.Tensor | undefined;
      if (this.config.enableForgettingGate) {
        if (this.forgettingGateVariable) {
          this.forgettingGateVariable.dispose();
        }
        const forgettingInit = Number.isFinite(this.config.forgettingGateInit)
          ? this.config.forgettingGateInit
          : 0.1;
        const alphaInit = tf.fill([memorySlots], forgettingInit);
        this.forgettingGateVariable = tf.variable(alphaInit, true, 'forgetting_gate');
        forgettingGateTensor = this.forgettingGateVariable;
      } else if (this.forgettingGateVariable) {
        this.forgettingGateVariable.dispose();
        this.forgettingGateVariable = null;
      }

      // Initialize standard memory components with research paper extensions
      this.memoryState = {
        shortTerm: tf.keep(tf.zeros([memorySlots, embeddingSize])),
        longTerm: tf.keep(tf.zeros([Math.floor(memorySlots / 2), embeddingSize])),
        meta: tf.keep(tf.zeros([memorySlots, 5])), // metadata features per memory slot
        timestamps: tf.keep(tf.zeros([memorySlots])),
        accessCounts: tf.keep(tf.zeros([memorySlots])),
        surpriseHistory: tf.keep(tf.zeros([100])), // track last 100 surprise scores

        // Research Paper Extensions (Equations 32-33)
        momentumState: this.config.enableMomentum ? tf.keep(tf.zeros([memorySlots, embeddingSize])) : undefined,
        momentumDecay: this.config.enableMomentum ? this.config.momentumDecayRate : undefined,
        forgettingGate: forgettingGateTensor,

        // Token Flow Tracking (Section 3.1)
        tokenFlowHistory: this.config.enableTokenFlow ? tf.keep(tf.zeros([this.config.tokenFlowWindow, embeddingSize])) : undefined,
        flowWeights: this.config.enableTokenFlow ? tf.keep(tf.zeros([this.config.tokenFlowWindow])) : undefined
      };

      // Keep momentum buffer for backward compatibility and optimization
      const momentumInit = tf.zeros([memorySlots, embeddingSize]);
      if (this.momentumBuffer) {
        this.momentumBuffer.dispose();
      }
      this.momentumBuffer = tf.keep(momentumInit);

      // Initialize extended memory state if episodic/semantic distinction is enabled
      if (this.config.enableEpisodicSemanticDistinction) {
        this.initializeExtendedMemoryState(memorySlots, embeddingSize, currentTime);
      }

      // Initialize hierarchical memory if enabled (research paper feature)
      if (this.config.useHierarchicalMemory || this.config.enableHierarchicalMemory) {
        this.initializeHierarchicalMemory();
      }

      // Initialize quantization if enabled
      if (this.config.enableQuantization) {
        this.initializeQuantization();
      }

      const features = [];
      if (this.config.enableMomentum) { features.push('momentum'); }
      if (this.config.enableTokenFlow) { features.push('token-flow'); }
      if (this.config.enableForgettingGate) { features.push('forgetting-gate'); }
      if (this.config.enableHierarchicalMemory) { features.push('hierarchical'); }

      this.logger.info('memory.initialize', `Memory initialized with ${memorySlots} slots, ${embeddingSize} dimensions`);
      if (features.length > 0) {
        this.logger.info('memory.initialize', `Research paper features enabled: ${features.join(', ')}`);
      }
    });
  }

  private initializeExtendedMemoryState(memorySlots: number, embeddingSize: number, currentTime: number): void {
    // Tier distribution: 40% working, 35% short-term, 25% long-term
    const workingSlots = Math.floor(memorySlots * 0.4);
    const shortTermSlots = Math.floor(memorySlots * 0.35);
    const longTermSlots = memorySlots - workingSlots - shortTermSlots;

    // Type distribution: 60% episodic, 40% semantic
    const episodicSlots = Math.floor(memorySlots * 0.6);
    const semanticSlots = memorySlots - episodicSlots;

    this.extendedMemoryState = {
      ...this.memoryState,

      // Hierarchical memory tiers
      workingMemory: tf.zeros([workingSlots, embeddingSize]),
      shortTermMemory: tf.zeros([shortTermSlots, embeddingSize]),
      longTermMemory: tf.zeros([longTermSlots, embeddingSize]),

      // Episodic vs Semantic distinction
      episodicMemory: tf.zeros([episodicSlots, embeddingSize]),
      semanticMemory: tf.zeros([semanticSlots, embeddingSize]),

      // Temporal information
      workingTimestamps: tf.fill([workingSlots], currentTime),
      shortTermTimestamps: tf.fill([shortTermSlots], currentTime),
      longTermTimestamps: tf.fill([longTermSlots], currentTime),
      episodicTimestamps: tf.fill([episodicSlots], currentTime),
      semanticTimestamps: tf.fill([semanticSlots], currentTime),

      // Access patterns
      workingAccessCounts: tf.zeros([workingSlots]),
      shortTermAccessCounts: tf.zeros([shortTermSlots]),
      longTermAccessCounts: tf.zeros([longTermSlots]),
      episodicAccessCounts: tf.zeros([episodicSlots]),
      semanticAccessCounts: tf.zeros([semanticSlots]),

      // Memory quality metrics
      episodicRecency: tf.ones([episodicSlots]), // Start with neutral recency
      semanticConfidence: tf.ones([semanticSlots]).mul(0.5), // Start with medium confidence
      memoryImportance: tf.ones([memorySlots]).mul(0.5), // Start with medium importance
      surpriseScores: tf.zeros([memorySlots]),

      // Memory classification (0=working, 1=short-term, 2=long-term)
      memoryTiers: tf.concat([
        tf.zeros([workingSlots]),           // Working memory = 0
        tf.ones([shortTermSlots]),          // Short-term memory = 1  
        tf.ones([longTermSlots]).mul(2)     // Long-term memory = 2
      ]) as tf.Tensor1D,

      // Memory types (0=episodic, 1=semantic)
      memoryTypes: tf.concat([
        tf.zeros([episodicSlots]),          // Episodic = 0
        tf.ones([semanticSlots])            // Semantic = 1
      ])
    };
  }

  private validateMemoryState(state: IMemoryState): boolean {
    return tf.tidy(() => {
      try {
        const validations = [
          state.shortTerm && !state.shortTerm.isDisposed,
          state.longTerm && !state.longTerm.isDisposed,
          state.meta && !state.meta.isDisposed,
          state.timestamps && !state.timestamps.isDisposed,
          state.accessCounts && !state.accessCounts.isDisposed,
          state.surpriseHistory && !state.surpriseHistory.isDisposed
        ];

        return validations.every(Boolean);
      } catch (error) {
        this.logger.warn('memory.validate', 'Error validating memory state', { error });
        return false;
      }
    });
  }

  public async storeMemory(text: string): Promise<void> {
    const embedding = await this.encodeText(text);
    const similarity = this.calculateSimilarity(embedding);

    const { values, indices } = tf.topk(similarity, 1);
    if (values.dataSync()[0] < this.config.similarityThreshold) {
      this.addMemoryEntry(embedding);
    }

    this.updateAccessStats(indices);
    this.checkPruning();
  }

  private calculateSimilarity(embedding: tf.Tensor1D): tf.Tensor1D {
    return tf.tidy(() => {
      const expanded = embedding.reshape([1, -1]);
      return tf.matMul(this.memoryState.shortTerm, expanded)
        .div(tf.norm(this.memoryState.shortTerm, 2, 1).mul(tf.norm(expanded)))
        .squeeze();
    });
  }

  private addMemoryEntry(embedding: tf.Tensor1D): void {
    tf.tidy(() => {
      const newMemory = tf.concat([
        this.memoryState.shortTerm,
        embedding.reshape([1, -1])
      ], 0).slice(0, this.config.memorySlots);

      this.memoryState.shortTerm.dispose();
      this.memoryState.shortTerm = newMemory as tf.Tensor2D;
    });
  }

  private updateAccessStats(indices: tf.Tensor1D): void {
    tf.tidy(() => {
      const updates = tf.onesLike(indices);
      this.memoryState.accessCounts = tf.add(
        this.memoryState.accessCounts,
        tf.scatterND(indices.reshape([-1, 1]), updates, [this.config.memorySlots])
      );
    });
  }

  private checkPruning(): void {
    this.stepCount++;
    if (this.stepCount % this.config.pruningInterval === 0) {
      this.pruneMemory(this.memoryState, this.config.similarityThreshold);
    }
  }

  public pruneMemory(memoryState: IMemoryState, threshold: number): IMemoryState {
    return tf.tidy(() => {
      const relevance = this.computeMemoryRelevance();
      const { indices } = tf.topk(relevance, this.config.memorySlots);

      return {
        shortTerm: tf.gather(memoryState.shortTerm, indices) as tf.Tensor2D,
        longTerm: tf.gather(memoryState.longTerm, indices) as tf.Tensor2D,
        meta: tf.gather(memoryState.meta, indices) as tf.Tensor2D,
        timestamps: tf.gather(memoryState.timestamps, indices) as tf.Tensor1D,
        accessCounts: tf.gather(memoryState.accessCounts, indices) as tf.Tensor1D,
        surpriseHistory: tf.gather(memoryState.surpriseHistory, indices) as tf.Tensor1D
      };
    });
  }

  private computeMemoryRelevance(): tf.Tensor1D {
    return tf.tidy(() => {
      const recency = tf.sub(tf.scalar(Date.now()), this.memoryState.timestamps);
      const frequency = tf.log(tf.add(this.memoryState.accessCounts, 1));
      const surprise = tf.mul(
        this.memoryState.surpriseHistory,
        this.config.surpriseDecay
      );

      return tf.addN([recency, frequency, surprise]) as tf.Tensor1D;
    });
  }

  public async recallMemory(query: string, topK = 5): Promise<tf.Tensor2D[]> {
    const queryEmbedding = await this.encodeText(query);
    if (this.config.useApproximateNearestNeighbors && this.memoryState.shortTerm.shape[0] > 2000) {
      return this.annSearch(queryEmbedding, topK);
    }

    const similarities = this.calculateSimilarity(queryEmbedding);
    const { indices } = tf.topk(similarities, topK);
    return indices.arraySync().map(i =>
      this.memoryState.shortTerm.slice([i, 0], [1, -1]) as tf.Tensor2D
    );
  }

  /**
   * Performs approximate nearest neighbor search using HNSW index
   * @param query The query embedding
   * @param topK Number of top results to return
   * @returns Array of similar memory tensors
   */
  private async annSearch(query: tf.Tensor1D, topK: number): Promise<tf.Tensor2D[]> {
    // Import and use the HNSW implementation
    const { HNSW } = await import('./ann.js');

    if (!this.hnswIndex) {
      this.hnswIndex = new HNSW();

      // Extract memory vectors for indexing
      const memoryVectors: tf.Tensor[] = [];
      const numSlots = this.memoryState.shortTerm.shape[0];

      for (let i = 0; i < numSlots; i++) {
        const vector = this.memoryState.shortTerm.slice([i, 0], [1, -1]).squeeze();
        memoryVectors.push(vector);
      }

      // Build the index
      await this.hnswIndex.buildIndex(memoryVectors);
    }

    // Check if index needs rebuilding
    if (this.hnswIndex.needsRebuild(true, this.memoryState.shortTerm.shape[0])) {
      // Rebuild the index
      const memoryVectors: tf.Tensor[] = [];
      const numSlots = this.memoryState.shortTerm.shape[0];

      for (let i = 0; i < numSlots; i++) {
        const vector = this.memoryState.shortTerm.slice([i, 0], [1, -1]).squeeze();
        memoryVectors.push(vector);
      }

      await this.hnswIndex.buildIndex(memoryVectors);
    }

    // Perform the search
    const results = await this.hnswIndex.search(query, topK);

    // Convert 1D results back to 2D tensors for compatibility
    return results.map((tensor: tf.Tensor) => tensor.reshape([1, -1]));
  }

  /**
   * Forward pass with hierarchical memory support
   */
  public forward(input: ITensor, state?: IMemoryState): {
    predicted: ITensor;
    memoryUpdate: IMemoryUpdateResult;
  } {
    let predicted: ITensor;
    let memoryUpdate: IMemoryUpdateResult;

    tf.tidy(() => {
      if (!this.encoder || !this.decoder) {
        this.encoder = this.encoder ?? this.createEncoder();
        this.decoder = this.decoder ?? this.createDecoder();
      }

      const memoryState = state ?? this.memoryState;
      const inputTensor = unwrapTensor(input) as tf.Tensor;
      const inputVector = inputTensor.reshape([1, inputTensor.shape[0]]);

      const encodedInput = this.encoder.predict(inputVector) as tf.Tensor2D;
      const attentionBlock = this.computeMemoryAttention(encodedInput);

      let memoryContextTensor: tf.Tensor;
      if (this.config.useHierarchicalMemory) {
        const flattenedEncodedInput = encodedInput.squeeze() as tf.Tensor1D;
        const retrievedTensor = unwrapTensor(
          this.retrieveFromHierarchicalMemory(wrapTensor(flattenedEncodedInput))
        ) as tf.Tensor;
        flattenedEncodedInput.dispose();

        if (retrievedTensor.shape.length === 1) {
          const expanded = tf.expandDims(retrievedTensor as tf.Tensor1D, 0);
          retrievedTensor.dispose();
          memoryContextTensor = expanded;
        } else {
          memoryContextTensor = retrievedTensor;
        }
      } else {
        memoryContextTensor = attentionBlock.values;
      }

      const combined = tf.concat([encodedInput, memoryContextTensor], 1);
      const decoded = this.decoder.predict(combined) as tf.Tensor2D;

      const surpriseVector = tf.sub(decoded, inputVector);
      const surpriseMagnitude = tf.norm(surpriseVector);

      let updatedState = this.updateMemory(
        encodedInput,
        surpriseMagnitude,
        memoryState,
        attentionBlock
      );

      let tokenFlowHistoryTensor: tf.Tensor2D | undefined;
      let flowWeightsTensor: tf.Tensor1D | undefined;
      if (this.config.enableTokenFlow) {
        const flattenedInput = tf.squeeze(encodedInput);
        const flowUpdate = this.updateTokenFlow(
          memoryState.tokenFlowHistory ? unwrapTensor(memoryState.tokenFlowHistory) as tf.Tensor2D : undefined,
          flattenedInput
        );
        tokenFlowHistoryTensor = flowUpdate.history;
        flowWeightsTensor = flowUpdate.weights;
        flattenedInput.dispose();
      }

      const surpriseHistoryTensor = updatedState.surpriseHistory as tf.Tensor1D;
      const windowSize = this.config.tokenFlowWindow;
      const historySize = surpriseHistoryTensor.size;
      const effectiveWindow = Math.min(windowSize, historySize);
      let immediateWindow: tf.Tensor1D;
      if (effectiveWindow > 0) {
        const sliceStart = historySize - effectiveWindow;
        immediateWindow = tf.slice(
          surpriseHistoryTensor,
          [sliceStart],
          [effectiveWindow]
        );
      } else {
        immediateWindow = tf.zeros([0]);
      }
      if (immediateWindow.shape[0] < windowSize) {
        const padAmount = windowSize - immediateWindow.shape[0];
        const padding = tf.zeros([padAmount]);
        immediateWindow = tf.concat([padding, immediateWindow], 0) as tf.Tensor1D;
        padding.dispose();
      }
      let accumulatedHistory = surpriseHistoryTensor.clone();
      let totalSurprise = tf.mean(immediateWindow);

      if (this.config.enableTokenFlow && tokenFlowHistoryTensor && flowWeightsTensor) {
        const weighted = this.weightSurpriseByTokenFlow(
          immediateWindow,
          accumulatedHistory,
          flowWeightsTensor
        );
        immediateWindow.dispose();
        accumulatedHistory.dispose();
        totalSurprise.dispose();
        immediateWindow = weighted.immediate;
        accumulatedHistory = weighted.accumulated;
        totalSurprise = weighted.total;
      } else {
        tf.keep(immediateWindow);
        tf.keep(accumulatedHistory);
        tf.keep(totalSurprise);
      }

      if (updatedState.surpriseHistory && updatedState.surpriseHistory !== accumulatedHistory) {
        (updatedState.surpriseHistory as tf.Tensor).dispose();
      }
      updatedState.surpriseHistory = accumulatedHistory;

      if (this.config.enableTokenFlow && tokenFlowHistoryTensor && flowWeightsTensor) {
        if (updatedState.tokenFlowHistory && updatedState.tokenFlowHistory !== tokenFlowHistoryTensor) {
          (updatedState.tokenFlowHistory as tf.Tensor).dispose();
        }
        if (updatedState.flowWeights && updatedState.flowWeights !== flowWeightsTensor) {
          (updatedState.flowWeights as tf.Tensor).dispose();
        }
        updatedState.tokenFlowHistory = tokenFlowHistoryTensor;
        updatedState.flowWeights = flowWeightsTensor;
      }

      if (this.config.useHierarchicalMemory) {
        this.updateHierarchicalMemory(encodedInput, surpriseMagnitude);
      }

      // Apply hierarchical memory promotion/demotion if enabled
      if (this.config.enableHierarchicalMemory || this.config.useHierarchicalMemory) {
        updatedState = this.applyMemoryPromotion(updatedState);
      }

      this.memoryState = updatedState;
      this.stepCount++;

      predicted = tf.keep(decoded);
      memoryUpdate = {
        newState: updatedState,
        attention: attentionBlock,
        surprise: {
          immediate: wrapTensor(immediateWindow),
          accumulated: wrapTensor(accumulatedHistory),
          totalSurprise: wrapTensor(totalSurprise)
        }
      };
    });

    return { predicted: predicted!, memoryUpdate: memoryUpdate! };
  }

  private computeMemoryAttention(query: tf.Tensor2D): IAttentionBlock {
    return tf.tidy(() => {
      const memoryKeys = this.memoryState.shortTerm as tf.Tensor2D;
      if (memoryKeys.shape[0] === 0) {
        const emptyKeys = tf.keep(tf.zerosLike(memoryKeys));
        const emptyScores = tf.keep(tf.zeros([query.shape[0], 1]));
        const emptyValues = tf.keep(tf.zeros([query.shape[0], this.config.memoryDim]));
        return { keys: emptyKeys, values: emptyValues, scores: emptyScores };
      }

      const epsilon = tf.scalar(1e-8);
      const normalize = (tensor: tf.Tensor2D) => {
        const norms = tf.add(tf.norm(tensor, 'euclidean', 1, true), epsilon);
        return tf.div(tensor, norms);
      };

      const normalizedQuery = normalize(query);
      const normalizedKeys = normalize(memoryKeys);

      const rawScores = tf.matMul(normalizedQuery, normalizedKeys, false, true);
      const scores = tf.softmax(rawScores);
      const values = tf.matMul(scores, memoryKeys);

      return {
        keys: tf.keep(tf.clone(memoryKeys)),
        values: tf.keep(values),
        scores: tf.keep(scores)
      };
    });
  }

  private computeSurprise(input: tf.Tensor2D, expected: tf.Tensor2D): ISurpriseMetrics {
    return tf.tidy(() => {
      const error = SafeTensorOps.sub(input, expected);
      const immediate = tf.mean(tf.square(error), 1);
      const decayTensor = tf.scalar(this.config.surpriseDecay);
      const accumulated = SafeTensorOps.add(
        SafeTensorOps.mul(this.memoryState.surpriseHistory, decayTensor),
        immediate
      );
      // Example: totalSurprise could be immediate or accumulated
      const totalSurprise = immediate.clone();

      return { immediate, accumulated, totalSurprise }; // Return all required fields
    });
  }

  /**
   * Implements contrastive learning to improve embedding space
   * @param anchor The anchor embedding
   * @param positive The positive example (similar to anchor)
   * @returns The contrastive loss
   */
  private contrastiveLearning(anchor: ITensor, positive: ITensor): ITensor {
    if (!this.config.enableContrastiveLearning) {
      return wrapTensor(tf.scalar(0.0));
    }

    return this.withErrorHandling('contrastiveLearning', () => {
      // Normalize embeddings to unit length
      const anchorNorm = tf.div(
        unwrapTensor(anchor),
        tf.norm(unwrapTensor(anchor))
      );

      const positiveNorm = tf.div(
        unwrapTensor(positive),
        tf.norm(unwrapTensor(positive))
      );

      // Add to contrastive buffer if not full
      if (this.contrastiveBuffer.length < this.contrastiveBufferSize) {
        this.contrastiveBuffer.push(anchorNorm.clone());
      } else {
        // Replace random item in buffer
        const replaceIndex = Math.floor(Math.random() * this.contrastiveBufferSize);
        tf.dispose(this.contrastiveBuffer[replaceIndex]);
        this.contrastiveBuffer[replaceIndex] = anchorNorm.clone();
      }

      // Need at least 8 samples for meaningful contrastive learning
      if (this.contrastiveBuffer.length < 8) {
        return wrapTensor(tf.scalar(0.0));
      }

      // Compute similarity between anchor and positive example
      const positiveSimilarity = tf.sum(tf.mul(anchorNorm, positiveNorm));

      // Compute similarities with negative examples from buffer
      const negativeSimilarities = this.contrastiveBuffer.map(negative => {
        return tf.sum(tf.mul(anchorNorm, negative));
      });

      // Concatenate positive and negative similarities
      const allSimilarities = tf.concat([
        positiveSimilarity.reshape([1]),
        tf.stack(negativeSimilarities)
      ]);

      // Scale by temperature
      const scaledSimilarities = tf.div(
        allSimilarities,
        tf.scalar(this.contrastiveTemperature)
      );

      // Compute softmax
      const softmaxSimilarities = tf.softmax(scaledSimilarities);

      // Contrastive loss is negative log likelihood of positive example
      const loss = tf.neg(tf.log(softmaxSimilarities.gather([0])));

      // Clean up
      tf.dispose([
        anchorNorm,
        positiveNorm,
        positiveSimilarity,
        allSimilarities,
        scaledSimilarities,
        softmaxSimilarities
      ]);

      return wrapTensor(loss);
    });
  }

  /**
   * Enhanced training step with contrastive learning
   */
  public trainStep(
    currentInput: ITensor,
    nextInput: ITensor,
    state: IMemoryState
  ): {
    loss: ITensor;
    gradients: IModelGradients;
    memoryUpdate: IMemoryUpdateResult;
  } {
    return this.withErrorHandling('trainStep', () => {
      return tf.tidy(() => {
        const { predicted, memoryUpdate } = this.forward(currentInput, state);

        const predictedTensor = unwrapTensor(predicted);
        const targetTensor = unwrapTensor(nextInput);
        const reshapedTarget = targetTensor.reshape(predictedTensor.shape);
        let predictionLoss = tf.losses.meanSquaredError(
          reshapedTarget,
          predictedTensor
        );
        if (predictionLoss.rank !== 0) {
          predictionLoss = tf.mean(predictionLoss);
        }
        reshapedTarget.dispose();

        let contrastiveLoss = tf.scalar(0.0);
        if (this.config.enableContrastiveLearning) {
          if (!this.encoder) { this.encoder = this.createEncoder(); }
          const currentEncoded = this.encoder.predict(unwrapTensor(currentInput)) as tf.Tensor;
          const nextEncoded = this.encoder.predict(unwrapTensor(nextInput)) as tf.Tensor;
          let contrastiveLossScalar = unwrapTensor(
            this.contrastiveLearning(
              currentEncoded,
              nextEncoded
            )
          );
          if (contrastiveLossScalar.rank !== 0) {
            safeLog("Warning: Contrastive loss tensor was not rank 0. Taking mean.");
            const meanLoss = contrastiveLossScalar.mean();
            tf.dispose(contrastiveLossScalar);
            contrastiveLossScalar = meanLoss;
          }
          contrastiveLoss = contrastiveLossScalar as tf.Scalar;
        }

        const contrastiveWeight = this.config.contrastiveWeight || 0.1;
        const combinedLoss = tf.add(
          predictionLoss,
          tf.mul(contrastiveLoss, tf.scalar(contrastiveWeight))
        );

        let updatedMemoryState = memoryUpdate.newState;

        let momentumApplied = false;
        if (this.config.enableMomentum && state.momentumState) {
          const momentumTensor = this.computeMomentumUpdate(
            state,
            unwrapTensor(currentInput),
            unwrapTensor(nextInput),
            memoryUpdate
          );
          updatedMemoryState = this.applyMomentumToMemory(
            updatedMemoryState,
            momentumTensor
          );
          momentumApplied = true;
        }

        if (this.config.enableForgettingGate && state.forgettingGate && !momentumApplied) {
          updatedMemoryState = this.applyForgettingGate(
            updatedMemoryState,
            state.forgettingGate
          );
        }

        this.memoryState = updatedMemoryState;

        this.optimizer.applyGradients({});
        this.stepCount++;

        tf.dispose([predictionLoss, contrastiveLoss]);

        return {
          loss: wrapTensor(combinedLoss),
          gradients: {
            shortTerm: wrapTensor(tf.zeros([1])),
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

  /**
   * Computes the momentum term S_t using prior state and new memory update.
   * Implements Equation 33 with practical adaptations for matrix shapes.
   */
  private computeMomentumUpdate(
    state: IMemoryState,
    currentInput: tf.Tensor,
    nextInput: tf.Tensor,
    memoryUpdate: IMemoryUpdateResult
  ): tf.Tensor2D {
    return tf.tidy(() => {
      const previousShortTerm = unwrapTensor(state.shortTerm) as tf.Tensor2D;
      const previousMomentum = state.momentumState
        ? (unwrapTensor(state.momentumState) as tf.Tensor2D)
        : tf.zerosLike(previousShortTerm);

      const etaValue = state.momentumDecay ?? this.config.momentumDecayRate ?? 0.9;
      const thetaValue = (this.config as any).momentumLearningRate ?? this.config.learningRate ?? 0.001;

      const scoresTensor = unwrapTensor(memoryUpdate.attention.scores) as tf.Tensor2D;
      const keysTensor = unwrapTensor(memoryUpdate.attention.keys) as tf.Tensor2D;
      const valuesTensor = unwrapTensor(memoryUpdate.attention.values) as tf.Tensor2D;

      if (scoresTensor.size === 0 || keysTensor.size === 0 || valuesTensor.size === 0) {
        return tf.keep(previousMomentum);
      }

      const aggregatedKey = tf.matMul(scoresTensor, keysTensor); // [1, memoryDim]
      const keyMatrix = tf.tile(aggregatedKey, [previousShortTerm.shape[0], 1]);

      const valueMatrix = tf.tile(valuesTensor, [previousShortTerm.shape[0], 1]);

      const scoresColumn = tf.transpose(scoresTensor); // [memorySlots, 1]
      const scoresMatrix = tf.tile(scoresColumn, [1, previousShortTerm.shape[1]]);

      const weightedDifference = tf.mul(
        scoresMatrix,
        tf.sub(tf.mul(previousShortTerm, keyMatrix), valueMatrix)
      );

      const momentumDecayTerm = tf.mul(previousMomentum, etaValue);
      const gradientTerm = tf.mul(weightedDifference, thetaValue);

      const momentumUpdate = tf.sub(momentumDecayTerm, gradientTerm);

      return tf.keep(momentumUpdate) as tf.Tensor2D;
    });
  }

  /**
   * Maintains the rolling token flow window and produces updated flow weights.
   */
  private updateTokenFlow(
    currentHistory: tf.Tensor2D | undefined,
    newToken: tf.Tensor1D
  ): { history: tf.Tensor2D; weights: tf.Tensor1D } {
    return tf.tidy(() => {
      const featureSize = newToken.shape[0];
      const windowSize = this.config.tokenFlowWindow;

      const baseHistory = currentHistory ?? tf.zeros([windowSize, featureSize]);
      let updatedHistory: tf.Tensor2D;

      if (windowSize > 1) {
        const retained = tf.slice(baseHistory, [1, 0], [windowSize - 1, featureSize]);
        const appended = newToken.reshape([1, featureSize]);
        updatedHistory = tf.concat([retained, appended], 0) as tf.Tensor2D;
      } else {
        updatedHistory = newToken.reshape([1, featureSize]);
      }

      const weights = this.computeTokenFlowWeights(updatedHistory, newToken);

      return {
        history: tf.keep(updatedHistory),
        weights: tf.keep(weights)
      };
    });
  }

  /**
   * Computes contribution weights for the rolling token flow window.
   * Combines recency decay with cosine similarity to the current token.
   */
  private computeTokenFlowWeights(
    flowHistory: tf.Tensor2D,
    currentToken: tf.Tensor1D
  ): tf.Tensor1D {
    return tf.tidy(() => {
      const windowSize = flowHistory.shape[0];
      const indices = tf.range(0, windowSize, 1, 'float32');
      const windowScalar = tf.scalar(windowSize, 'float32');
      const minWeight = tf.scalar(0.05, 'float32');

      // Recency weighting: newest token receives weight close to 1.0
      const recencyWeights = tf.maximum(
        tf.div(tf.sub(windowScalar, indices), windowScalar),
        minWeight
      );

      // Similarity weighting: cosine similarity between history and current token
      const epsilon = tf.scalar(1e-8);
      const normalizedToken = tf.div(currentToken, tf.add(tf.norm(currentToken), epsilon));
      const historyNorms = tf.add(tf.norm(flowHistory, 'euclidean', 1, true), epsilon);
      const normalizedHistory = tf.div(flowHistory, historyNorms);
      const similarities = tf.matMul(
        normalizedHistory,
        normalizedToken.reshape([-1, 1])
      ).squeeze();
      const similarityWeights = tf.sigmoid(similarities);

      const combined = tf.add(
        tf.mul(recencyWeights, tf.scalar(0.5)),
        tf.mul(similarityWeights, tf.scalar(0.5))
      );
      const weightSum = tf.maximum(tf.sum(combined), epsilon);
      return tf.div(combined, weightSum);
    });
  }

  /**
   * Blends momentary surprise with flow-informed context to produce a weighted surprise signal.
   */
  private weightSurpriseByTokenFlow(
    immediateSurprise: tf.Tensor1D,
    accumulatedSurprise: tf.Tensor1D,
    flowWeights: tf.Tensor1D
  ): { immediate: tf.Tensor1D; accumulated: tf.Tensor1D; total: tf.Tensor } {
    return tf.tidy(() => {
      const epsilon = tf.scalar(1e-8);
      const normalizedWeights = tf.div(
        flowWeights,
        tf.maximum(tf.sum(flowWeights), epsilon)
      );

      const flowContribution = tf.sum(tf.mul(normalizedWeights, immediateSurprise));

      const historyLength = accumulatedSurprise.shape[0];
      let updatedHistory: tf.Tensor1D;
      if (historyLength > 1) {
        const historyPrefix = tf.slice(accumulatedSurprise, [0], [historyLength - 1]);
        updatedHistory = tf.concat([historyPrefix, flowContribution.reshape([1])], 0) as tf.Tensor1D;
      } else {
        updatedHistory = flowContribution.reshape([1]);
      }

      const blendedImmediate = tf.add(
        tf.mul(immediateSurprise, tf.scalar(0.3)),
        tf.mul(flowContribution, tf.scalar(0.7))
      );

      return {
        immediate: tf.keep(blendedImmediate),
        accumulated: tf.keep(updatedHistory),
        total: tf.keep(blendedImmediate.mean())
      };
    });
  }

  /**
   * Applies the computed momentum to memory while preserving previous base activations.
   * Implements Equation 32 by replacing old momentum contribution with the new S_t.
   */
  private applyMomentumToMemory(
    memoryState: IMemoryState,
    momentum: tf.Tensor2D
  ): IMemoryState {
    return tf.tidy(() => {
      const currentShortTerm = unwrapTensor(memoryState.shortTerm) as tf.Tensor2D;
      const previousMomentum = memoryState.momentumState
        ? (unwrapTensor(memoryState.momentumState) as tf.Tensor2D)
        : tf.zerosLike(momentum);

      const baseShortTerm = tf.sub(currentShortTerm, previousMomentum);

      let gateTensor: tf.Tensor2D | null = null;
      if (memoryState.forgettingGate) {
        const rawGate = unwrapTensor(memoryState.forgettingGate);
        if (rawGate.size === 0 || rawGate.dataSync().some((v: number) => !Number.isFinite(v))) {
          rawGate.dispose();
        } else if (rawGate.rank === 0) {
          gateTensor = tf.fill(momentum.shape, rawGate) as tf.Tensor2D;
        } else if (rawGate.rank === 1) {
          if (rawGate.shape[0] === momentum.shape[0]) {
            gateTensor = tf.tile(rawGate.reshape([rawGate.shape[0], 1]), [1, momentum.shape[1]]);
          } else if (rawGate.shape[0] === momentum.shape[1]) {
            gateTensor = tf.tile(rawGate.reshape([1, rawGate.shape[0]]), [momentum.shape[0], 1]);
          }
        } else if (rawGate.rank === 2) {
          gateTensor = rawGate as tf.Tensor2D;
        }
      }

      const forgettingMask = gateTensor ? tf.clipByValue(gateTensor, 0, 1) : tf.zerosLike(momentum);
      const retainedMomentum = tf.mul(tf.sub(tf.onesLike(forgettingMask), forgettingMask), momentum);
      const forgottenMemory = gateTensor ? tf.mul(baseShortTerm, tf.sub(tf.onesLike(forgettingMask), forgettingMask)) : baseShortTerm;

      const updatedShortTerm = tf.add(forgottenMemory, retainedMomentum);

      return {
        ...memoryState,
        shortTerm: wrapTensor(tf.keep(updatedShortTerm)),
        momentumState: wrapTensor(tf.keep(momentum)),
        momentumDecay: memoryState.momentumDecay ?? this.config.momentumDecayRate
      };
    });
  }

  /**
   * Applies a learnable forgetting gate to the short-term memory.
   */
  private applyForgettingGate(
    memoryState: IMemoryState,
    forgettingGate: ITensor
  ): IMemoryState {
    return tf.tidy(() => {
      const shortTerm = unwrapTensor(memoryState.shortTerm) as tf.Tensor2D;
      const gateTensor = unwrapTensor(forgettingGate);

      let alphaExpanded: tf.Tensor;
      if (gateTensor.rank === 0) {
        alphaExpanded = tf.fill(shortTerm.shape, gateTensor);
      } else if (gateTensor.rank === 1) {
        if (gateTensor.shape[0] === shortTerm.shape[0]) {
          alphaExpanded = tf.tile(gateTensor.reshape([gateTensor.shape[0], 1]), [1, shortTerm.shape[1]]);
        } else if (gateTensor.shape[0] === shortTerm.shape[1]) {
          alphaExpanded = tf.tile(gateTensor.reshape([1, gateTensor.shape[0]]), [shortTerm.shape[0], 1]);
        } else {
          const meanAlpha = tf.mean(gateTensor);
          alphaExpanded = tf.fill(shortTerm.shape, meanAlpha);
        }
      } else {
        alphaExpanded = gateTensor as tf.Tensor;
        if (
          (alphaExpanded.rank === 2 &&
            (alphaExpanded.shape[0] !== shortTerm.shape[0] ||
              alphaExpanded.shape[1] !== shortTerm.shape[1]))
        ) {
          const rowMin = Math.min(alphaExpanded.shape[0], shortTerm.shape[0]);
          const colMin = Math.min(alphaExpanded.shape[1], shortTerm.shape[1]);
          let resized = alphaExpanded.slice([0, 0], [rowMin, colMin]);
          if (rowMin !== shortTerm.shape[0] || colMin !== shortTerm.shape[1]) {
            const rowPad = shortTerm.shape[0] - rowMin;
            const colPad = shortTerm.shape[1] - colMin;
            resized = tf.pad(resized, [[0, rowPad], [0, colPad]]);
          }
          alphaExpanded = resized;
        }
      }

      const clippedAlpha = tf.clipByValue(alphaExpanded, 0, 1);
      const retained = tf.mul(
        shortTerm,
        tf.sub(tf.onesLike(clippedAlpha), clippedAlpha)
      );

      return {
        ...memoryState,
        shortTerm: wrapTensor(tf.keep(retained)),
        forgettingGate: forgettingGate
      };
    });
  }

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
        shortTerm: wrapTensor(tf.tensor2d(updatedShortTerm.length > 0 ? updatedShortTerm : [[0]])),
        longTerm: wrapTensor(tf.tensor2d(updatedLongTerm))
      };
    });
  }

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
        timestamps: wrapTensor(tf.tensor1d(filteredTimestamps.length > 0 ? filteredTimestamps : [0])),
        accessCounts: wrapTensor(tf.tensor1d(filteredAccessCounts.length > 0 ? filteredAccessCounts : [0]))
      };
    });
  }

  public updateMetaMemory(surprise: ISurpriseMetrics, context: ITensor): ITensor {
    return tf.tidy(() => {
      const surpriseGate = tf.sigmoid(surprise.immediate);
      return tf.add(
        tf.mul(this.memoryState.meta, tf.sub(1, surpriseGate)),
        tf.mul(context, surpriseGate)
      );
    });
  }

  public manifoldStep(base: ITensor, velocity: ITensor): ITensor {
    return tf.tidy(() => {
      const norm = tf.norm(velocity);
      const normalized = tf.div(velocity, norm);
      return tf.add(base, tf.mul(normalized, this.config.learningRate));
    });
  }

  public getConfig(): TitanMemoryConfig {
    return { ...this.config };
  }

  /**
   * Saves the model to disk with proper versioning and error handling
   * @param path The path to save the model to (legacy format)
   * @deprecated Use saveCheckpoint() with RobustPersistenceManager instead
   */
  public async save(path: string): Promise<void> {
    return this.withErrorHandlingAsync('save', async () => {
      try {
        const dir = path.split('/').slice(0, -1).join('/');
        await fs.mkdir(dir, { recursive: true });

        const modelMetadata = {
          version: "1.0",
          format: "titan-memory-v1",
          created: new Date().toISOString(),
          config: this.config
        };

        const encoderPath = `${path}/encoder`;
        const decoderPath = `${path}/decoder`;
        await this.encoder.save(`file://${encoderPath}`);
        await this.decoder.save(`file://${decoderPath}`);
        this.logger.info('encoder.save', 'Saved encoder and decoder models');

        const memoryData = {
          shortTerm: await this.memoryState.shortTerm.array(),
          longTerm: await this.memoryState.longTerm.array(),
          meta: await this.memoryState.meta.array(),
          timestamps: Array.from(await this.memoryState.timestamps.data()),
          accessCounts: Array.from(await this.memoryState.accessCounts.data()),
          surpriseHistory: Array.from(await this.memoryState.surpriseHistory.data())
        };

        let hierarchicalData = null;
        if (this.config.useHierarchicalMemory && this.hierarchicalMemory) {
          // Cast to Internal type to access tensor arrays
          const internalHierarchicalMemory = this.hierarchicalMemory;
          hierarchicalData = {
            levels: await Promise.all(internalHierarchicalMemory.levels.map(async (level: tf.Tensor) => await level.array())) as tf.TensorLike[], // levels can be multi-dimensional
            // Ensure each promise resolves to number[] before Promise.all creates number[][]
            timestamps: await Promise.all(internalHierarchicalMemory.timestamps.map(async (ts: tf.Tensor): Promise<number[]> => Array.from(await ts.data()))),
            accessCounts: await Promise.all(internalHierarchicalMemory.accessCounts.map(async (ac: tf.Tensor): Promise<number[]> => Array.from(await ac.data()))),
            surpriseScores: await Promise.all(internalHierarchicalMemory.surpriseScores.map(async (ss: tf.Tensor): Promise<number[]> => Array.from(await ss.data())))
          };
        }

        let quantizationData = null;
        if (this.config.enableQuantization && this.quantizedMemory) {
          // Cast to Internal type
          const internalQuantizedMemory = this.quantizedMemory;
          quantizationData = {
            // Convert Uint8Array to number[] for JSON
            shortTerm: Array.from(internalQuantizedMemory.shortTerm),
            longTerm: Array.from(internalQuantizedMemory.longTerm),
            meta: Array.from(internalQuantizedMemory.meta),
            ranges: internalQuantizedMemory.quantizationRanges,
            bits: this.quantizationBits
          };
        }

        const telemetry = ModelTelemetry.getInstance();
        const telemetryData = telemetry.getMetrics(); // Correct method name

        const modelData = {
          ...modelMetadata,
          encoderPath,
          decoderPath,
          memoryState: memoryData,
          hierarchicalMemory: hierarchicalData,
          quantization: quantizationData,
          telemetry: telemetryData
        };

        const modelPath = `${path}/model.json`;
        await fs.writeFile(modelPath, JSON.stringify(modelData, null, 2));
        this.logger.info('model.save', `Model saved to ${path}`);
      } catch (error) {
        const err = error as Error; // Cast error
        this.logger.error('model.save', 'Error saving model', err);
        throw new MemoryError(`Failed to save model: ${err.message}`); // Use casted error
      }
    });
  }

  /**
   * Save a robust checkpoint using the new persistence manager
   * @param persistenceManager The persistence manager instance
   * @param tokenizer Optional tokenizer to include in checkpoint
   * @param annIndex Optional ANN index to include in checkpoint
   * @param metadata Optional additional metadata
   */
  public async saveCheckpoint(
    persistenceManager: any, // Will be properly typed when persistence is imported
    tokenizer?: any,
    annIndex?: any,
    metadata?: any
  ): Promise<string> {
    return this.withErrorHandlingAsync('saveCheckpoint', async () => {
      return await persistenceManager.saveCheckpoint(this, tokenizer, annIndex, metadata);
    });
  }

  /**
   * Loads the model from disk with proper error handling
   * @param path The path to load the model from (legacy format)
   * @deprecated Use loadCheckpoint() with RobustPersistenceManager instead
   */
  public async load(path: string): Promise<void> {
    return this.withErrorHandlingAsync('load', async () => {
      try {
        // Check if model.json exists (new format)
        const modelPath = `${path}/model.json`;
        let modelData;

        try {
          const modelJson = await fs.readFile(modelPath, 'utf-8');
          modelData = JSON.parse(modelJson);
          safeLog('Found model.json, loading in new format');
        } catch (error) {
          safeLog('No model.json found, trying legacy format');
          await this.loadLegacyFormat(path);
          return;
        }

        // Validate model format
        if (!modelData.format || modelData.format !== 'titan-memory-v1') {
          const isMcpContextValue = process.env.MCP_CONTEXT === 'true';
          if (!isMcpContextValue) {
            this.logger.warn('model.load', `Unknown model format: ${modelData.format || 'undefined'}, attempting to load anyway`);
          }
        }

        // Load configuration
        if (modelData.config) {
          this.config = { ...this.config, ...modelData.config };
          safeLog('Loaded model configuration');
        }

        // Load encoder and decoder models
        try {
          if (modelData.encoderPath && modelData.decoderPath) {
            this.encoder = await tf.loadLayersModel(`file://${modelData.encoderPath}/model.json`);
            this.decoder = await tf.loadLayersModel(`file://${modelData.decoderPath}/model.json`);
            safeLog('Loaded encoder and decoder models');
          } else {
            throw new Error('Missing encoder or decoder paths in model data');
          }
        } catch (error) {
          const err = error as Error; // Cast error
          const isMcpContextValue = process.env.MCP_CONTEXT === 'true';
          if (!isMcpContextValue) {
            this.logger.error('model.load', 'Error loading encoder/decoder models', err);
          }
          throw new MemoryError(`Failed to load encoder/decoder models: ${err.message}`); // Use casted error
        }

        // Initialize optimizer
        const learningRate = this.config.learningRate || 0.001;
        this.optimizer = tf.train.adam(learningRate);

        // Load memory state
        if (modelData.memoryState) {
          try {
            // Dispose existing memory state
            if (this.memoryState) {
              Object.values(this.memoryState).forEach(tensor => {
                if (tensor && !tensor.isDisposed) {
                  tensor.dispose();
                }
              });
            }

            // Create new memory state from saved data
            this.memoryState = {
              shortTerm: tf.tensor(modelData.memoryState.shortTerm),
              longTerm: tf.tensor(modelData.memoryState.longTerm),
              meta: tf.tensor(modelData.memoryState.meta),
              timestamps: tf.tensor1d(modelData.memoryState.timestamps),
              accessCounts: tf.tensor1d(modelData.memoryState.accessCounts),
              surpriseHistory: tf.tensor1d(modelData.memoryState.surpriseHistory)
            };
            if (this.momentumBuffer) {
              this.momentumBuffer.dispose();
            }
            this.momentumBuffer = tf.keep(tf.zerosLike(this.memoryState.shortTerm as tf.Tensor2D));
            safeLog('Loaded memory state');
          } catch (error) {
            const err = error as Error; // Cast error
            const isMcpContextValue = process.env.MCP_CONTEXT === 'true';
            if (!isMcpContextValue) {
              this.logger.error('model.load', 'Error loading memory state', err);
            }
            throw new MemoryError(`Failed to load memory state: ${err.message}`); // Use casted error
          }
        } else {
          const isMcpContextValue = process.env.MCP_CONTEXT === 'true';
          if (!isMcpContextValue) {
            this.logger.warn('model.load', 'No memory state found in model data, initializing new memory state');
          }
          this.initializeMemoryState();
        }

        // Load hierarchical memory if available
        if (modelData.hierarchicalMemory && this.config.useHierarchicalMemory) {
          try {
            const hierarchicalData = modelData.hierarchicalMemory;

            // Initialize the property with the correct internal type structure
            this.hierarchicalMemory = {
              levels: hierarchicalData.levels.map((level: number[][]) => tf.tensor(level)),
              timestamps: hierarchicalData.timestamps.map((ts: number[]) => tf.tensor1d(ts)),
              accessCounts: hierarchicalData.accessCounts.map((ac: number[]) => tf.tensor1d(ac)),
              surpriseScores: hierarchicalData.surpriseScores.map((ss: number[]) => tf.tensor1d(ss))
            };
            safeLog('Loaded hierarchical memory');
          } catch (error) {
            const err = error as Error; // Cast error
            const isMcpContextValue = process.env.MCP_CONTEXT === 'true';
            if (!isMcpContextValue) {
              this.logger.error('model.load', 'Error loading hierarchical memory', err);
            }
            this.hierarchicalMemory = null; // Set to null on error
            if (this.config.useHierarchicalMemory) {
              this.initializeHierarchicalMemory(); // Re-initialize if load failed
            }
          }
        } else if (this.config.useHierarchicalMemory) {
          const isMcpContextValue = process.env.MCP_CONTEXT === 'true';
          if (!isMcpContextValue) {
            this.logger.warn('model.load', 'No hierarchical memory found but enabled in config, initializing new hierarchical memory');
          }
          this.initializeHierarchicalMemory();
        }

        // Load quantization data if available
        if (modelData.quantization && this.config.enableQuantization) {
          try {
            // Initialize the property with the correct internal type structure
            this.quantizedMemory = {
              // Convert number[][] back to Uint8Array[]
              shortTerm: modelData.quantization.shortTerm.map((arr: number[]) => new Uint8Array(arr)),
              longTerm: modelData.quantization.longTerm.map((arr: number[]) => new Uint8Array(arr)),
              meta: modelData.quantization.meta.map((arr: number[]) => new Uint8Array(arr)),
              quantizationRanges: modelData.quantization.ranges
            };
            this.quantizationBits = modelData.quantization.bits;
            safeLog('Loaded quantization data');
          } catch (error) {
            const err = error as Error; // Cast error
            const isMcpContextValue = process.env.MCP_CONTEXT === 'true';
            if (!isMcpContextValue) {
              this.logger.error('model.load', 'Error loading quantization data', err);
            }
            this.quantizedMemory = null; // Set to null on error
            if (this.config.enableQuantization) {
              this.initializeQuantization(); // Re-initialize if load failed
            }
          }
        } else if (this.config.enableQuantization) {
          const isMcpContextValue = process.env.MCP_CONTEXT === 'true';
          if (!isMcpContextValue) {
            this.logger.warn('model.load', 'No quantization data found but enabled in config, initializing new quantization');
          }
          this.initializeQuantization();
        }

        // Initialize tokenizer if text processing is enabled
        if ((this.config as any).enableTextProcessing) {
          if (!this.vectorProcessor) {
            throw new Error('VectorProcessor not initialized');
          }
        }

        safeLog(`Model loaded from ${path}`);
      } catch (error) {
        const err = error as Error; // Cast error
        const isMcpContextValue = process.env.MCP_CONTEXT === 'true';
        if (!isMcpContextValue) {
          this.logger.error('model.load', 'Error loading model', err);
        }
        throw new MemoryError(`Failed to load model: ${err.message}`); // Use casted error
      }
    });
  }

  /**
   * Load a robust checkpoint using the new persistence manager
   * @param persistenceManager The persistence manager instance
   * @param checkpointPath Path to the checkpoint
   * @param options Optional loading options
   */
  public async loadCheckpoint(
    persistenceManager: any, // Will be properly typed when persistence is imported
    checkpointPath: string,
    options: any = {}
  ): Promise<{ model: TitanMemoryModel; tokenizer?: any; annIndex?: any }> {
    return this.withErrorHandlingAsync('loadCheckpoint', async () => {
      const { model, tokenizer, annIndex } = await persistenceManager.loadCheckpoint(checkpointPath, options);
      return { model, tokenizer, annIndex };
    });
  }

  /**
   * Loads the model using the legacy format
   * @param path The path to load the model from
   */
  private async loadLegacyFormat(path: string): Promise<void> {
    const isMcpContext = process.env.MCP_CONTEXT === 'true';

    try {
      safeLog('Attempting to load model in legacy format');

      // Try to load configuration
      try {
        const configPath = `${path}/config.json`;
        const configData = await fs.readFile(configPath, 'utf-8');
        this.config = JSON.parse(configData);
        safeLog('Loaded configuration from legacy format');
      } catch (error) {
        if (!isMcpContext) {
          this.logger.warn('model.load', 'No config.json found in legacy format, using default configuration');
        }
      }

      // Initialize components based on config
      this.encoder = this.createEncoder();
      this.decoder = this.createDecoder();

      // Initialize optimizer
      const learningRate = this.config.learningRate || 0.001;
      this.optimizer = tf.train.adam(learningRate);

      // Try to load memory state
      try {
        const memoryPath = `${path}/memory.json`;
        const memoryData = JSON.parse(await fs.readFile(memoryPath, 'utf-8'));

        // Dispose existing memory state
        if (this.memoryState) {
          Object.values(this.memoryState).forEach(tensor => {
            if (tensor && !tensor.isDisposed) {
              tensor.dispose();
            }
          });
        }

        // Create new memory state from saved data
        this.memoryState = {
          shortTerm: tf.tensor(memoryData.shortTerm),
          longTerm: tf.tensor(memoryData.longTerm),
          meta: tf.tensor(memoryData.meta),
          timestamps: tf.tensor1d(memoryData.timestamps),
          accessCounts: tf.tensor1d(memoryData.accessCounts),
          surpriseHistory: tf.tensor1d(memoryData.surpriseHistory)
        };
        if (this.momentumBuffer) {
          this.momentumBuffer.dispose();
        }
        this.momentumBuffer = tf.keep(tf.zerosLike(this.memoryState.shortTerm as tf.Tensor2D));
        safeLog('Loaded memory state from legacy format');
      } catch (error) {
        if (!isMcpContext) {
          this.logger.warn('model.load', 'No memory.json found in legacy format, initializing new memory state');
        }
        this.initializeMemoryState();
      }

      // Try to load hierarchical memory
      if (this.config.useHierarchicalMemory) {
        try {
          const hierarchicalPath = `${path}/hierarchical.json`;
          const hierarchicalData = JSON.parse(await fs.readFile(hierarchicalPath, 'utf-8'));

          this.hierarchicalMemory = {
            levels: hierarchicalData.levels.map((level: number[][]) => tf.tensor(level)),
            timestamps: hierarchicalData.timestamps.map((ts: number[]) => tf.tensor1d(ts)),
            accessCounts: hierarchicalData.accessCounts.map((ac: number[]) => tf.tensor1d(ac)),
            surpriseScores: hierarchicalData.surpriseScores.map((ss: number[]) => tf.tensor1d(ss))
          };
          safeLog('Loaded hierarchical memory from legacy format');
        } catch (error) {
          if (!isMcpContext) {
            this.logger.warn('model.load', 'No hierarchical.json found in legacy format, initializing new hierarchical memory');
          }
          this.initializeHierarchicalMemory();
        }
      }

      // Try to load quantization data
      if (this.config.enableQuantization) {
        try {
          const quantizationPath = `${path}/quantization.json`;
          const quantizationData = JSON.parse(await fs.readFile(quantizationPath, 'utf-8')) as {
            ranges: number[][];
            bits: number;
          };

          this.quantizationRanges = quantizationData.ranges as unknown as Array<{ min: number; max: number }>;
          this.quantizationBits = quantizationData.bits;

          // Initialize quantized memory
          this.initializeQuantization();
        } catch (error) {
          if (!isMcpContext) {
            this.logger.warn('model.load', 'No quantization.json found in legacy format, initializing new quantization');
          }
          this.initializeQuantization();
        }
      }

      // Initialize tokenizer if text processing is enabled
      if ((this.config as any).enableTextProcessing) {
        if (!this.vectorProcessor) {
          throw new Error('VectorProcessor not initialized');
        }
      }

      safeLog(`Model loaded from ${path} using legacy format`);
    } catch (error) {
      if (!isMcpContext) {
        this.logger.error('model.load', 'Error loading model in legacy format', error as Error);
      }
      throw new MemoryError(`Failed to load model in legacy format: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  public getMemorySnapshot(): Record<string, tf.Tensor> {
    return {
      shortTerm: this.memoryState.shortTerm.clone(),
      longTerm: this.memoryState.longTerm.clone(),
      meta: this.memoryState.meta.clone(),
      timestamps: this.memoryState.timestamps.clone(),
      accessCounts: this.memoryState.accessCounts.clone(),
      surpriseHistory: this.memoryState.surpriseHistory.clone()
    };
  }

  public restoreMemoryState(memoryData: any): void {
    // Dispose existing memory state
    if (this.memoryState) {
      Object.values(this.memoryState).forEach((tensor: any) => {
        if (tensor && typeof tensor.dispose === 'function' && !tensor.isDisposed) {
          tensor.dispose();
        }
      });
    }

    // Restore memory state from data
    this.memoryState = {
      shortTerm: tf.tensor(memoryData.shortTerm),
      longTerm: tf.tensor(memoryData.longTerm),
      meta: tf.tensor(memoryData.meta),
      timestamps: tf.tensor1d(memoryData.timestamps),
      accessCounts: tf.tensor1d(memoryData.accessCounts),
      surpriseHistory: tf.tensor1d(memoryData.surpriseHistory)
    };
    if (this.momentumBuffer) {
      this.momentumBuffer.dispose();
    }
    this.momentumBuffer = tf.keep(tf.zerosLike(this.memoryState.shortTerm as tf.Tensor2D));
  }

  /**
   * Cleans up resources used by the model
   */
  public dispose(): void {
    return this.withErrorHandling('dispose', () => {
      // Dispose of encoder and decoder
      if (this.encoder) {
        this.encoder.dispose();
      }

      if (this.decoder) {
        this.decoder.dispose();
      }

      // Dispose of memory state
      if (this.memoryState) {
        Object.values(this.memoryState).forEach(value => {
          if (value instanceof tf.Tensor && !value.isDisposed) {
            value.dispose();
          }
        });
      }

      // Dispose of hierarchical memory
      if (this.hierarchicalMemory) {
        // Cast to internal type to iterate over tensor arrays
        const internalHierarchicalMemory = this.hierarchicalMemory;
        internalHierarchicalMemory.levels.forEach((tensor: tf.Tensor) => { if (tensor && !tensor.isDisposed) { tensor.dispose(); } });
        internalHierarchicalMemory.timestamps.forEach((tensor: tf.Tensor) => { if (tensor && !tensor.isDisposed) { tensor.dispose(); } });
        internalHierarchicalMemory.accessCounts.forEach((tensor: tf.Tensor) => { if (tensor && !tensor.isDisposed) { tensor.dispose(); } });
        internalHierarchicalMemory.surpriseScores.forEach((tensor: tf.Tensor) => { if (tensor && !tensor.isDisposed) { tensor.dispose(); } });
      }

      // Dispose of contrastive buffer
      this.contrastiveBuffer.forEach(tensor => tensor.dispose());
      this.contrastiveBuffer = [];

      if (this.momentumBuffer) {
        this.momentumBuffer.dispose();
        this.momentumBuffer = null;
      }

      this.logger.info('model', 'Model resources disposed');
    });
  }

  private async getWeightData(): Promise<Record<string, number[]>> {
    return tf.tidy(() => {
      const weights: Record<string, number[]> = {};

      // Save transformer stack weights with proper naming
      this.transformerStack.forEach((layer, layerIndex) => {
        layer.getWeights().forEach((w, weightIndex) => {
          if (!w.isDisposed) {
            const weightName = `transformer_${layerIndex}_${weightIndex}`;
            weights[weightName] = Array.from(w.dataSync());
          }
        });
      });

      // Save projector weights with proper naming
      if (this.memoryProjector) {
        this.memoryProjector.getWeights().forEach((w, weightIndex) => {
          if (!w.isDisposed) {
            const weightName = `projector_layer_${weightIndex}`;
            weights[weightName] = Array.from(w.dataSync());
          }
        });
      }

      // Save similarity network weights with proper naming
      if (this.similarityNetwork) {
        this.similarityNetwork.getWeights().forEach((w, weightIndex) => {
          if (!w.isDisposed) {
            const weightName = `similarity_layer_${weightIndex}`;
            weights[weightName] = Array.from(w.dataSync());
          }
        });
      }

      return weights;
    });
  }

  /**
   * Loads weights from a buffer with proper error handling and version checking
   * @param weightsBuffer The buffer containing the weights
   */
  private async loadWeights(weightsBuffer: Buffer): Promise<void> {
    const isMcpContext = process.env.MCP_CONTEXT === 'true';

    return this.withErrorHandlingAsync('loadWeights', async () => {
      try {
        safeLog('Loading weights from buffer');

        // First try to parse as JSON
        try {
          const jsonData = JSON.parse(weightsBuffer.toString('utf8'));
          safeLog('Found JSON format weights');
          await this.loadWeightsFromJson(jsonData);
          return;
        } catch (jsonError) {
          // Not JSON format, try binary format
          safeLog('Not JSON format, trying binary format');
        }

        // Try binary format
        try {
          await this.loadWeightsFromBinary(weightsBuffer);
        } catch (binaryError) {
          const bError = binaryError as Error; // Cast binaryError
          const isMcpContextValue = process.env.MCP_CONTEXT === 'true'; // Define and check
          if (!isMcpContextValue) {
            this.logger.error('model.load', 'Failed to load weights in binary format', bError);
          }
          throw new MemoryError(`Failed to load weights: ${bError.message}`); // Use casted error message
        }
      } catch (error) {
        if (!isMcpContext) {
          this.logger.error('model.load', 'Error loading weights:', error);
        }
        throw new MemoryError(`Failed to load weights: ${(error instanceof Error ? error.message : String(error))}`);
      }
    });
  }

  private async loadWeightsFromJson(weightData: any): Promise<void> {
    const isMcpContext = process.env.MCP_CONTEXT === 'true';

    try {
      // Check version compatibility
      if (weightData.version && weightData.version !== '1.0') {
        if (!isMcpContext) {
          this.logger.warn('model.load', `Weight version mismatch. Expected 1.0, got ${weightData.version}. Attempting to load anyway.`);
        }
      }

      // Load weights into a map
      const weightMap = new Map<string, tf.Tensor>();

      // Process each weight entry
      for (const [name, data] of Object.entries(weightData.weights)) {
        try {
          const { values, shape, dtype } = data as { values: number[], shape: number[], dtype: string };
          // Explicitly cast dtype to the expected type for tf.tensor
          const tensor = tf.tensor(values, shape, dtype as tf.DataType);
          weightMap.set(name, tensor);
        } catch (error) {
          const err = error as Error; // Cast error
          const isMcpContextValue = process.env.MCP_CONTEXT === 'true'; // Define and check
          if (!isMcpContextValue) {
            this.logger.warn('model.load', `Error loading weight ${name}:`, err);
          }
        }
      }

      // Apply weights to model
      this.applyLoadedWeights(weightMap);

      // Load memory state if available
      if (weightData.memoryState) {
        try {
          // Dispose existing memory state
          if (this.memoryState) {
            Object.values(this.memoryState).forEach(tensor => {
              if (tensor && !tensor.isDisposed) {
                tensor.dispose();
              }
            });
          }

          // Create new memory state from saved data
          this.memoryState = {
            shortTerm: tf.tensor(weightData.memoryState.shortTerm.values, weightData.memoryState.shortTerm.shape),
            longTerm: tf.tensor(weightData.memoryState.longTerm.values, weightData.memoryState.longTerm.shape),
            meta: tf.tensor(weightData.memoryState.meta.values, weightData.memoryState.meta.shape),
            timestamps: tf.tensor1d(weightData.memoryState.timestamps.values),
            accessCounts: tf.tensor1d(weightData.memoryState.accessCounts.values),
            surpriseHistory: tf.tensor1d(weightData.memoryState.surpriseHistory.values)
          };
          if (this.momentumBuffer) {
            this.momentumBuffer.dispose();
          }
          this.momentumBuffer = tf.keep(tf.zerosLike(this.memoryState.shortTerm as tf.Tensor2D));
          safeLog('Loaded memory state from weights');
        } catch (error) {
          const err = error as Error; // Cast error
          const isMcpContextValue = process.env.MCP_CONTEXT === 'true'; // Define and check
          if (!isMcpContextValue) {
            this.logger.warn('model.load', 'Error loading memory state from weights:', err);
          }
          this.initializeMemoryState(); // Initialize if load fails
        }
      }

      // Load hierarchical memory if available
      if (weightData.hierarchicalMemory && this.config.useHierarchicalMemory) {
        try {
          const hierarchicalData = weightData.hierarchicalMemory;

          this.hierarchicalMemory = {
            levels: hierarchicalData.levels.map((level: any) =>
              tf.tensor(level.values, level.shape)),
            timestamps: hierarchicalData.timestamps.map((ts: any) =>
              tf.tensor1d(ts.values)),
            accessCounts: hierarchicalData.accessCounts.map((ac: any) =>
              tf.tensor1d(ac.values)),
            surpriseScores: hierarchicalData.surpriseScores.map((ss: any) =>
              tf.tensor1d(ss.values))
          };
          safeLog('Loaded hierarchical memory from weights');
        } catch (error) {
          const err = error as Error; // Cast error
          const isMcpContextValue = process.env.MCP_CONTEXT === 'true'; // Define and check
          if (!isMcpContextValue) {
            this.logger.warn('model.load', 'Error loading hierarchical memory from weights:', err);
          }
          if (this.config.useHierarchicalMemory) {
            this.initializeHierarchicalMemory();
          }
        }
      }

      // Load quantization data if available
      if (weightData.quantization && this.config.enableQuantization) {
        try {
          this.quantizationRanges = weightData.quantization.ranges;
          this.quantizationBits = weightData.quantization.bits;

          // Initialize quantized memory
          this.initializeQuantization();
          safeLog('Loaded quantization data from weights');
        } catch (error) {
          const err = error as Error; // Cast error
          const isMcpContextValue = process.env.MCP_CONTEXT === 'true'; // Define and check
          if (!isMcpContextValue) {
            this.logger.warn('model.load', 'Error loading quantization data from weights:', err);
          }
          if (this.config.enableQuantization) {
            this.initializeQuantization();
          }
        }
      }

      safeLog('Successfully loaded weights from JSON format');
    } catch (error) {
      const err = error as Error; // Cast error
      const isMcpContextValue = process.env.MCP_CONTEXT === 'true'; // Define and check
      if (!isMcpContextValue) {
        this.logger.error('model.load', 'Error loading weights from JSON:', err);
      }
      throw new MemoryError(`Failed to load weights from JSON: ${err.message}`); // Use casted error
    }
  }

  private async loadWeightsFromBinary(weightsBuffer: Buffer): Promise<void> {
    const isMcpContextValue = process.env.MCP_CONTEXT === 'true';

    try {
      safeLog('Loading weights from binary format');

      // Parse header
      const headerSize = weightsBuffer.readUInt32LE(0);
      const headerJson = weightsBuffer.toString('utf8', 4, 4 + headerSize);
      const header = JSON.parse(headerJson);

      // Validate header
      if (!header.format || header.format !== 'titan-memory') {
        if (!isMcpContextValue) {
          this.logger.warn('model.load', `Unknown weight format: ${header.format || 'undefined'}, attempting to load anyway`);
        }
      }

      // Load weights into a map
      const weightMap = new Map<string, tf.Tensor>();
      const offset = 4 + headerSize;

      for (const [name, info] of Object.entries(header.weights)) {
        const { shape, dtype, byteOffset, byteLength } = info as WeightInfo & { byteOffset: number, byteLength: number };

        try {
          // Read tensor data
          const dataBuffer = weightsBuffer.slice(offset + byteOffset, offset + byteOffset + byteLength);

          // Create tensor based on dtype
          let tensor: tf.Tensor;
          const dtypeStr = dtype;
          // Explicitly check allowed dtypes
          if (dtypeStr === 'float32') {
            const values = new Float32Array(dataBuffer.buffer, dataBuffer.byteOffset, byteLength / 4);
            tensor = tf.tensor(Array.from(values), shape, 'float32');
          } else if (dtypeStr === 'int32') {
            const values = new Int32Array(dataBuffer.buffer, dataBuffer.byteOffset, byteLength / 4);
            tensor = tf.tensor(Array.from(values), shape, 'int32');
          } else if (dtypeStr === 'bool') {
            const values = new Uint8Array(dataBuffer.buffer, dataBuffer.byteOffset, byteLength);
            tensor = tf.tensor(Array.from(values), shape, 'bool');
          } else if (dtypeStr === 'uint8') {
            const values = new Uint8Array(dataBuffer.buffer, dataBuffer.byteOffset, byteLength);
            tensor = tf.tensor(Array.from(values), shape, 'int32');
          } else {
            if (!isMcpContextValue) {
              this.logger.warn('model.load', `Unsupported dtype: ${dtype} for weight ${name}, skipping`);
            }
            continue;
          }
          weightMap.set(name, tensor);
        } catch (error) {
          const err = error as Error; // Cast error
          const isMcpContextValue = process.env.MCP_CONTEXT === 'true'; // Define and check
          if (!isMcpContextValue) {
            this.logger.warn('model.load', `Error loading weight ${name}:`, err);
          }
        }
      }

      // Apply weights to model
      this.applyLoadedWeights(weightMap);

      // Load memory state if available in header
      if (header.memoryState) {
        try {
          // Initialize memory state
          this.initializeMemoryState();
          safeLog('Initialized memory state from binary weights');
        } catch (error) {
          const err = error as Error; // Cast error
          const isMcpContextValue = process.env.MCP_CONTEXT === 'true'; // Define and check
          if (!isMcpContextValue) {
            this.logger.warn('model.load', 'Error initializing memory state from binary weights:', err);
          }
        }
      }

      safeLog('Successfully loaded weights from binary format');
    } catch (error) {
      const err = error as Error; // Cast error
      const isMcpContextValue = process.env.MCP_CONTEXT === 'true'; // Define and check
      if (!isMcpContextValue) {
        this.logger.error('model.load', 'Error loading weights from binary format:', err);
      }
      throw new MemoryError(`Failed to load weights from binary format: ${err.message}`); // Use casted error
    }
  }

  /**
   * Applies loaded weights to model components
   * @param weightMap Map of weight names to tensors
   */
  private applyLoadedWeights(weightMap: Map<string, tf.Tensor>): void {
    const isMcpContext = process.env.MCP_CONTEXT === 'true';

    try {
      // Apply transformer weights
      for (let i = 0; i < this.transformerStack.length; i++) {
        for (let j = 0; j < 10; j++) { // Assuming 10 layers per transformer
          const weightName = `transformer_${i}_${j}`;
          if (weightMap.has(weightName)) {
            const weight = weightMap.get(weightName)!;
            try {
              // Apply weight to transformer layer
              const layer = this.transformerStack[i].getLayer(j);
              if (layer) {
                const weights = layer.getWeights();
                if (weights.length > 0 && weight.shape.every((dim, idx) => dim === weights[0].shape[idx])) {
                  layer.setWeights([weight]);
                  weightMap.delete(weightName); // Remove from map to mark as used
                } else {
                  if (!isMcpContext) {
                    this.logger.warn('model.load', `Shape mismatch for ${weightName}, expected ${weights[0]?.shape}, got ${weight.shape}`);
                  }
                }
              }
            } catch (error) {
              if (!isMcpContext) {
                this.logger.warn('model.load', `Missing weight tensor: ${weightName}, keeping original weights`);
              }
            }
          } else {
            if (!isMcpContext) {
              this.logger.warn('model.load', `Missing weight tensor: ${weightName}, keeping original weights`);
            }
          }
        }
      }

      // Apply memory projector weights
      for (let i = 0; i < 4; i++) {
        const weightName = `projector_layer_${i}`;
        if (weightMap.has(weightName)) {
          const weight = weightMap.get(weightName)!;
          try {
            // Apply weight to projector layer
            const layer = this.memoryProjector.getLayer(i);
            if (layer) {
              const weights = layer.getWeights();
              if (weights.length > 0 && weight.shape.every((dim, idx) => dim === weights[0].shape[idx])) {
                layer.setWeights([weight]);
                weightMap.delete(weightName); // Remove from map to mark as used
              } else {
                if (!isMcpContext) {
                  this.logger.warn('model.load', `Shape mismatch for ${weightName}, expected ${weights[0]?.shape}, got ${weight.shape}`);
                }
              }
            }
          } catch (error) {
            if (!isMcpContext) {
              this.logger.warn('model.load', `Missing weight tensor: ${weightName}, keeping original weights`);
            }
          }
        } else {
          if (!isMcpContext) {
            this.logger.warn('model.load', `Missing weight tensor: ${weightName}, keeping original weights`);
          }
        }
      }

      // Apply similarity network weights
      for (let i = 0; i < 4; i++) {
        const weightName = `similarity_layer_${i}`;
        if (weightMap.has(weightName)) {
          const weight = weightMap.get(weightName)!;
          try {
            // Apply weight to similarity layer
            const layer = this.similarityNetwork.getLayer(i);
            if (layer) {
              const weights = layer.getWeights();
              if (weights.length > 0 && weight.shape.every((dim, idx) => dim === weights[0].shape[idx])) {
                layer.setWeights([weight]);
                weightMap.delete(weightName); // Remove from map to mark as used
              } else {
                if (!isMcpContext) {
                  this.logger.warn('model.load', `Shape mismatch for ${weightName}, expected ${weights[0]?.shape}, got ${weight.shape}`);
                }
              }
            }
          } catch (error) {
            if (!isMcpContext) {
              this.logger.warn('model.load', `Missing weight tensor: ${weightName}, keeping original weights`);
            }
          }
        } else {
          if (!isMcpContext) {
            this.logger.warn('model.load', `Missing weight tensor: ${weightName}, keeping original weights`);
          }
        }
      }

      // Apply encoder/decoder weights if available
      if (weightMap.has('encoder') && this.encoder) {
        try {
          const encoderWeights = weightMap.get('encoder')!;
          // Apply encoder weights
          weightMap.delete('encoder');
        } catch (error) {
          if (!isMcpContext) {
            this.logger.warn('model.load', 'Error applying encoder weights:', error);
          }
        }
      }

      if (weightMap.has('decoder') && this.decoder) {
        try {
          const decoderWeights = weightMap.get('decoder')!;
          // Apply decoder weights
          weightMap.delete('decoder');
        } catch (error) {
          if (!isMcpContext) {
            this.logger.warn('model.load', 'Error applying decoder weights:', error);
          }
        }
      }

      // Clean up any unused tensors
      weightMap.forEach((tensor, name) => {
        if (!tensor.isDisposed) {
          if (!isMcpContext) {
            this.logger.warn('model.load', `Unused weight tensor: ${name}, disposing`);
          }
          tensor.dispose();
        }
      });

      safeLog('Applied loaded weights to model');
    } catch (error) {
      if (!isMcpContext) {
        this.logger.error('model.load', 'Error applying weights to model:', error);
      }
      throw new MemoryError(`Failed to apply weights: ${(error instanceof Error ? error.message : String(error))}`);
    }
  }

  private updateMemoryState(input: tf.Tensor2D, surprise: ISurpriseMetrics): IMemoryUpdateResult {
    // Create tensors outside tidy to ensure they're not disposed
    const shortTermUpdate = tf.tidy(() => {
      return SafeTensorOps.add(
        SafeTensorOps.mul(this.memoryState.shortTerm, tf.scalar(this.config.decayRate)),
        input
      );
    });

    const longTermUpdate = tf.tidy(() => {
      return SafeTensorOps.add(
        this.memoryState.longTerm,
        SafeTensorOps.mul(input, tf.expandDims(surprise.accumulated, -1))
      );
    });

    const metaUpdate = this.updateMetaMemory(surprise, input);
    const currentTime = Date.now();
    const newTimestamps = tf.fill(this.memoryState.timestamps.shape, currentTime);
    const newAccessCounts = SafeTensorOps.add(this.memoryState.accessCounts, tf.ones(this.memoryState.accessCounts.shape));
    const attention = this.computeMemoryAttention(input);

    const newState: IMemoryState = {
      shortTerm: shortTermUpdate,
      longTerm: longTermUpdate,
      meta: metaUpdate,
      timestamps: newTimestamps,
      accessCounts: newAccessCounts,
      surpriseHistory: surprise.accumulated
    };

    return {
      newState,
      attention,
      surprise
    };
  }

  private computeGradients(input: tf.Tensor2D, target: tf.Tensor2D): IModelGradients {
    const error = tf.tidy(() => {
      const { values: attended } = this.computeMemoryAttention(input);
      const prediction = SafeTensorOps.add(attended, input);
      return SafeTensorOps.sub(prediction, target);
    });

    const { value: loss } = tf.variableGrads(() => {
      const [keyWeight, valueWeight] = this.similarityNetwork.getWeights() as [tf.Tensor2D, tf.Tensor2D];
      const keys = SafeTensorOps.matMul(this.memoryState.shortTerm, keyWeight);
      const values = SafeTensorOps.matMul(this.memoryState.shortTerm, valueWeight);
      const scores = tf.softmax(SafeTensorOps.matMul(input, keys.transpose()));
      const attended = SafeTensorOps.matMul(scores, values);
      const prediction = SafeTensorOps.add(attended, input);
      return tf.mean(tf.square(SafeTensorOps.sub(prediction, target)));
    });

    return {
      shortTerm: error,
      longTerm: error,
      meta: tf.keep(loss) as tf.Tensor
    };
  }

  /**
   * Resets accumulated gradients and optimizer state
   * This is useful when encountering gradient explosion or NaN values
   */
  public resetGradients(): void {
    tf.tidy(() => {
      // Recreate optimizer with the same learning rate
      const learningRate = this.config.learningRate || 0.001;
      this.optimizer = tf.train.adam(learningRate);

      // Reset step count
      this.stepCount = 0;

      this.logger.info('gradients', 'Gradients and optimizer state reset successfully');
    });
  }

  // Add MCP server compatibility methods
  public async init_model(config: Partial<TitanMemoryConfig>): Promise<{ status: 'success' } | { status: 'error'; message: string }> {
    try {
      await this.initialize(config);
      return { status: 'success' };
    } catch (error) {
      const err = error as Error; // Cast error
      return { status: 'error', message: err.message }; // Use casted error
    }
  }

  public async forward_pass(x: string | number[], memoryState?: IMemoryState): Promise<{
    predicted: number[];
    memoryUpdate: {
      shortTerm: number[][];
      timestamps: number[];
      accessCounts: number[];
      surpriseHistory: number[];
    };
  }> {
    try {
      // Process input
      let input: tf.Tensor1D;
      if (typeof x === 'string') {
        input = await this.encodeText(x);
      } else {
        input = tf.tensor1d(x);
      }

      // Use provided memory state or current state
      const state = memoryState || this.memoryState;

      // Forward pass
      const result = this.forward(input, state);

      // Convert tensors to arrays for JSON serialization
      const predicted = Array.from(await result.predicted.data());

      // Get memory update as arrays
      const shortTermArray = await result.memoryUpdate.newState.shortTerm.array() as number[][];
      const timestampsArray = Array.from(await result.memoryUpdate.newState.timestamps.data());
      const accessCountsArray = Array.from(await result.memoryUpdate.newState.accessCounts.data());
      const surpriseHistoryArray = Array.from(await result.memoryUpdate.newState.surpriseHistory.data());

      // Clean up tensors
      input.dispose();
      result.predicted.dispose(); // Dispose the tensor directly

      return {
        predicted,
        memoryUpdate: {
          shortTerm: shortTermArray,
          timestamps: timestampsArray,
          accessCounts: accessCountsArray,
          surpriseHistory: surpriseHistoryArray
        }
      };
    } catch (error: unknown) {
      // Return a properly formatted error response
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(JSON.stringify({ error: errorMessage }));
    }
  }

  public async train_step(x_t: string | number[], x_next: string | number[]): Promise<{
    loss: number;
  }> {
    try {
      // Process inputs
      let current: tf.Tensor1D;
      let next: tf.Tensor1D;

      if (typeof x_t === 'string') {
        current = await this.encodeText(x_t);
      } else {
        current = tf.tensor1d(x_t);
      }

      if (typeof x_next === 'string') {
        next = await this.encodeText(x_next);
      } else {
        next = tf.tensor1d(x_next);
      }

      // Train step - Pass tensors directly without object wrapping
      const result = this.trainStep(
        current,
        next,
        this.memoryState
      );

      // Get loss as number, access data directly after casting
      const lossData = await (result.loss as tf.Scalar).data();
      const lossValue = lossData[0];

      // Clean up tensors
      current.dispose();
      next.dispose();
      result.loss.dispose(); // Dispose the tensor directly

      return { loss: lossValue };
    } catch (error: unknown) {
      // Return a properly formatted error response
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(JSON.stringify({ error: errorMessage }));
    }
  }

  public get_memory_state(): {
    stats: {
      meanActivation: number;
      patternDiversity: number;
      surpriseScore: number;
    };
    capacity: number;
    status: string;
  } {
    try {
      return tf.tidy(() => { // Wrap calculations in tidy
        // Calculate memory statistics
        const shortTermMean = this.memoryState.shortTerm.mean().dataSync()[0];
        const longTermMean = this.memoryState.longTerm.mean().dataSync()[0];
        const metaMean = this.memoryState.meta.mean().dataSync()[0];

        // Calculate pattern diversity (standard deviation across memory)
        // Use tf.moments(...).variance.sqrt()
        const shortTermStd = tf.moments(this.memoryState.shortTerm).variance.sqrt().dataSync()[0];
        const longTermStd = tf.moments(this.memoryState.longTerm).variance.sqrt().dataSync()[0];

        // Get surprise score from history
        const surpriseScore = this.memoryState.surpriseHistory.mean().dataSync()[0];

        // Calculate memory capacity
        const memorySize = this.config.memorySlots || 5000;
        const usedSlots = this.memoryState.accessCounts.greater(tf.scalar(0)).sum().dataSync()[0];
        const capacity = 1 - (usedSlots / memorySize);

        // Determine status
        let status = 'active';
        if (capacity < 0.1) {
          status = 'critical';
        } else if (capacity < 0.3) {
          status = 'warning';
        }

        // Return formatted stats
        return {
          stats: {
            meanActivation: (shortTermMean + longTermMean + metaMean) / 3,
            patternDiversity: (shortTermStd + longTermStd) / 2,
            surpriseScore
          },
          capacity,
          status
        };
      });
    } catch (error: unknown) {
      // Return a properly formatted error response
      const errorMessage = error instanceof Error ? error.message : String(error);
      return {
        stats: {
          meanActivation: 0,
          patternDiversity: 0,
          surpriseScore: 0
        },
        capacity: 0,
        status: 'error'
      };
    }
  }

  // Add required interface methods
  public getMemoryState(): IMemoryState {
    const state: IMemoryState = {
      shortTerm: this.memoryState.shortTerm.clone(),
      longTerm: this.memoryState.longTerm.clone(),
      meta: this.memoryState.meta.clone(),
      timestamps: this.memoryState.timestamps.clone(),
      accessCounts: this.memoryState.accessCounts.clone(),
      surpriseHistory: this.memoryState.surpriseHistory.clone()
    };

    if (this.memoryState.momentumState) {
      state.momentumState = this.memoryState.momentumState.clone();
    }

    if (typeof this.memoryState.momentumDecay === 'number') {
      state.momentumDecay = this.memoryState.momentumDecay;
    }

    if (this.memoryState.forgettingGate) {
      state.forgettingGate = wrapTensor(unwrapTensor(this.memoryState.forgettingGate).clone());
    }

    if (this.memoryState.tokenFlowHistory) {
      state.tokenFlowHistory = this.memoryState.tokenFlowHistory.clone();
    }

    if (this.memoryState.flowWeights) {
      state.flowWeights = this.memoryState.flowWeights.clone();
    }

    return state;
  }

  public resetMemory(): void {
    this.initializeMemoryState();
  }

  /**
   * Enable or disable legacy character encoding mode for backward compatibility
   * @param useLegacyMode Whether to use legacy character-based encoding
   */
  public setLegacyCharEncoding(useLegacyMode: boolean): void {
    this.useLegacyCharEncoding = useLegacyMode;
    if (this.advancedTokenizer) {
      this.advancedTokenizer.setLegacyMode(useLegacyMode);
    }
    this.logger.info('encoding', `Text encoding mode set to: ${useLegacyMode ? 'legacy character' : 'advanced BPE'}`);
  }

  /**
   * Get current text encoding mode
   * @returns Whether legacy character encoding is enabled
   */
  public isLegacyCharEncoding(): boolean {
    return this.useLegacyCharEncoding;
  }

  /**
   * Get tokenizer statistics
   * @returns Statistics about the tokenizer usage
   */
  public getTokenizerStats(): {
    mode: 'BPE' | 'Legacy';
    vocabSize?: number;
    mergesCount?: number;
    embeddingDim?: number;
    bootstrapCount?: number;
  } {
    if (this.advancedTokenizer && !this.useLegacyCharEncoding) {
      const stats = this.advancedTokenizer.getStats();
      return {
        mode: 'BPE',
        vocabSize: stats.bpe.vocabSize,
        mergesCount: stats.bpe.mergesCount,
        embeddingDim: stats.embedding.embeddingDim,
        bootstrapCount: stats.bootstrapCount
      };
    } else {
      return {
        mode: 'Legacy',
        vocabSize: this.vocabSize
      };
    }
  }

  /**
   * Initializes hierarchical memory structure if enabled in config
   */
  private initializeHierarchicalMemory(): void {
    if (!this.config.useHierarchicalMemory) {
      this.hierarchicalMemory = null;
      return;
    }

    return this.withErrorHandling('initializeHierarchicalMemory', () => {
      if (this.hierarchicalMemory) {
        this.hierarchicalMemory.levels.forEach(level => level.dispose());
        this.hierarchicalMemory.timestamps.forEach(ts => ts.dispose());
        this.hierarchicalMemory.accessCounts.forEach(count => count.dispose());
        this.hierarchicalMemory.surpriseScores.forEach(score => score.dispose());
      }

      // Create multi-level memory structure
      const levels = this.hierarchicalLevels;
      const slotsPerLevel = Math.floor(this.config.memorySlots / levels);
      const embeddingSize = this.config.memoryDim;

      // Initialize memory levels with decreasing resolution and increasing time spans
      const shortTermLevels = Array(levels).fill(0).map((_, i) => {
        // Each level has fewer slots but covers longer time spans
        const levelSlots = Math.max(1, Math.floor(slotsPerLevel / (i + 1)));
        return tf.keep(tf.zeros([levelSlots, embeddingSize]));
      });

      // Initialize corresponding metadata for each level
      const timestampLevels = Array(levels).fill(0).map((_, i) => {
        const levelSlots = Math.max(1, Math.floor(slotsPerLevel / (i + 1)));
        return tf.keep(tf.zeros([levelSlots]));
      });

      const accessCountLevels = Array(levels).fill(0).map((_, i) => {
        const levelSlots = Math.max(1, Math.floor(slotsPerLevel / (i + 1)));
        return tf.keep(tf.zeros([levelSlots]));
      });

      // Initialize surprise scores for each level
      const surpriseLevels = Array(levels).fill(0).map((_, i) => {
        return tf.keep(tf.zeros([Math.max(10, Math.floor(100 / (i + 1)))]));
      });

      this.hierarchicalMemory = {
        levels: shortTermLevels,
        timestamps: timestampLevels,
        accessCounts: accessCountLevels,
        surpriseScores: surpriseLevels
      };

      this.logger.info('hierarchical', `Initialized hierarchical memory with ${levels} levels`);
    });
  }

  /**
   * Updates hierarchical memory with new information
   * @param input The input tensor to store in memory
   * @param surprise The surprise score for this input
   */
  private updateHierarchicalMemory(input: ITensor, surprise: ITensor): void {
    if (!this.hierarchicalMemory || !this.config.useHierarchicalMemory) {
      return;
    }

    return this.withErrorHandling('updateHierarchicalMemory', () => {
      const hierarchicalMemory = this.hierarchicalMemory as {
        levels: tf.Tensor[];
        timestamps: tf.Tensor[];
        accessCounts: tf.Tensor[];
        surpriseScores: tf.Tensor[];
      };
      const { levels, accessCounts } = hierarchicalMemory;

      // Update each level with different time scales
      levels.forEach((levelMemory, levelIndex) => {
        // Higher levels update less frequently
        const shouldUpdateLevel = (this.stepCount % Math.pow(2, levelIndex)) === 0;
        if (!shouldUpdateLevel && levelIndex > 0) {
          return;
        }

        // Find least recently used slot for this level
        const levelTimestamps = hierarchicalMemory.timestamps[levelIndex];
        const oldestSlotIndex = tf.argMin(levelTimestamps).dataSync()[0];

        // Update memory at the selected slot
        const inputArray = unwrapTensor(input).arraySync();
        const newMemory = levelMemory.arraySync();
        // Ensure newMemory is an array before indexing
        if (Array.isArray(newMemory)) {
          // Ensure the target index exists and is an array if inputArray is an array
          if (oldestSlotIndex < newMemory.length) {
            if (Array.isArray(inputArray) && Array.isArray(newMemory[oldestSlotIndex])) {
              // Deeply flatten and filter to number[]
              const flatInput: number[] = flattenToNumberArray(inputArray);
              newMemory[oldestSlotIndex] = flatInput;
            } else if (!Array.isArray(inputArray) && !Array.isArray(newMemory[oldestSlotIndex])) {
              newMemory[oldestSlotIndex] = inputArray;
            } else {
              safeLog(`Type mismatch at index ${oldestSlotIndex} in updateHierarchicalMemory`);
            }
          }
        }

        // Update metadata
        const newTimestamps = levelTimestamps.arraySync();
        // Ensure newTimestamps is an array before indexing
        if (Array.isArray(newTimestamps) && oldestSlotIndex < newTimestamps.length) {
          newTimestamps[oldestSlotIndex] = Date.now();
        }

        const newAccessCountsArray = accessCounts[levelIndex].arraySync();
        // Ensure newAccessCountsArray is an array before indexing
        if (Array.isArray(newAccessCountsArray) && oldestSlotIndex < newAccessCountsArray.length) {
          newAccessCountsArray[oldestSlotIndex] = 1; // Reset access count for new memory
        }

        // Update surprise history with exponential decay
        const rawSurpriseScores = hierarchicalMemory.surpriseScores[levelIndex].arraySync();
        let newSurpriseScores: number[];
        if (Array.isArray(rawSurpriseScores)) {
          // Deeply flatten and filter to number[]
          newSurpriseScores = flattenToNumberArray(rawSurpriseScores);
          if (newSurpriseScores.length > 0) {
            newSurpriseScores.shift();
          }
          newSurpriseScores.push(unwrapTensor(surprise).dataSync()[0]);
        } else {
          // Handle scalar case correctly
          newSurpriseScores = [unwrapTensor(surprise).dataSync()[0]];
          safeLog("Warning: rawSurpriseScores was scalar or unexpected type. Reinitialized.");
        }

        // Update tensors
        tf.dispose(levels[levelIndex]);
        tf.dispose(hierarchicalMemory.timestamps[levelIndex]);
        tf.dispose(accessCounts[levelIndex]);
        tf.dispose(hierarchicalMemory.surpriseScores[levelIndex]);

        // Ensure newMemory, newTimestamps, newAccessCountsArray are arrays before creating tensors
        const finalMemory: tf.TensorLike = Array.isArray(newMemory) && Array.isArray(newMemory[0]) ? newMemory : [[newMemory]];
        const finalTimestamps: number[] = Array.isArray(newTimestamps) && typeof newTimestamps[0] === 'number' ? newTimestamps as number[] : [Number(newTimestamps)];
        const finalAccessCounts: number[] = Array.isArray(newAccessCountsArray) && typeof newAccessCountsArray[0] === 'number' ? newAccessCountsArray as number[] : [Number(newAccessCountsArray)];
        const finalSurpriseScores: number[] = Array.isArray(newSurpriseScores) && typeof newSurpriseScores[0] === 'number' ? newSurpriseScores : [Number(newSurpriseScores)];

        // Use tf.tensor for potentially multi-dimensional, tf.tensor1d for known 1D
        levels[levelIndex] = tf.tensor(finalMemory);
        hierarchicalMemory.timestamps[levelIndex] = tf.tensor1d(finalTimestamps);
        accessCounts[levelIndex] = tf.tensor1d(finalAccessCounts);
        hierarchicalMemory.surpriseScores[levelIndex] = tf.tensor1d(finalSurpriseScores);
      });
    });
  }

  /**
   * Retrieves memories from hierarchical structure based on query
   * @param query The query tensor to match against memories
   * @returns The retrieved memory tensor
   */
  private retrieveFromHierarchicalMemory(query: ITensor): ITensor {
    if (!this.hierarchicalMemory || !this.config.useHierarchicalMemory) {
      // Fall back to standard memory retrieval
      return this.retrieveFromMemory(query);
    }

    return this.withErrorHandling('retrieveFromHierarchicalMemory', () => {
      const hierarchicalMemory = this.hierarchicalMemory as {
        levels: tf.Tensor[];
        timestamps: tf.Tensor[];
        accessCounts: tf.Tensor[];
        surpriseScores: tf.Tensor[];
      };
      const { levels, accessCounts } = hierarchicalMemory;

      const rawQuery = unwrapTensor(query) as tf.Tensor;
      let queryTensor = rawQuery;

      if (queryTensor.shape.length === 2) {
        if (queryTensor.shape[0] === 1) {
          queryTensor = queryTensor.squeeze() as tf.Tensor;
        } else if (queryTensor.shape[1] === 1) {
          queryTensor = queryTensor.reshape([queryTensor.shape[0]]);
        } else {
          const lastDim = queryTensor.shape[queryTensor.shape.length - 1];
          queryTensor = queryTensor.reshape([lastDim]);
        }
      }

      if (queryTensor.shape.length !== 1) {
        queryTensor = queryTensor.reshape([queryTensor.size]);
      }

      const queryVector = queryTensor as tf.Tensor1D;
      const queryData = queryVector.arraySync() as number[];
      const queryColumn = tf.tensor2d(queryData, [queryData.length, 1]);

      const attentionResults = levels.map((levelMemory, levelIndex) => {
        const similarities = tf.matMul(levelMemory, queryColumn) as tf.Tensor2D;
        const temperature = 1.0 / (levelIndex + 1);
        const scaledSimilarities = tf.div(similarities, temperature);
        similarities.dispose();
        const flattenedSimilarities = tf.squeeze(scaledSimilarities, [1]);
        scaledSimilarities.dispose();
        const attentionWeights = tf.softmax(flattenedSimilarities) as tf.Tensor1D;
        flattenedSimilarities.dispose();

        const newAccessCounts = accessCounts[levelIndex].add(attentionWeights);
        tf.dispose(accessCounts[levelIndex]);
        accessCounts[levelIndex] = newAccessCounts;

        const attentionWeightsRow = attentionWeights.reshape([1, attentionWeights.shape[0]]);
        const weightedMemories = tf.matMul(attentionWeightsRow, levelMemory);
        attentionWeightsRow.dispose();
        const weightedVector = tf.squeeze(weightedMemories) as tf.Tensor1D;
        weightedMemories.dispose();

        const levelResult = tf.mul(weightedVector, Math.pow(0.8, levelIndex)) as tf.Tensor1D;
        weightedVector.dispose();
        attentionWeights.dispose();
        return levelResult;
      });

      if (attentionResults.length === 0) {
        throw new MemoryError('No attention results to combine');
      }

      let combinedMemory = attentionResults[0].clone();
      for (let i = 1; i < attentionResults.length; i++) {
        const sum = tf.add(combinedMemory, attentionResults[i]) as tf.Tensor1D;
        combinedMemory.dispose();
        combinedMemory = sum;
      }

      const norm = tf.norm(combinedMemory);
      const safeNorm = tf.maximum(norm, tf.scalar(1e-12));
      const normalizedMemory = tf.div(combinedMemory, safeNorm) as tf.Tensor1D;

      attentionResults.forEach(tensor => tensor.dispose());
      combinedMemory.dispose();
      norm.dispose();
      safeNorm.dispose();
      queryColumn.dispose();

      return wrapTensor(normalizedMemory);
    });
  }

  /**
   * Initializes quantization if enabled in config
   */
  private initializeQuantization(): void {
    if (!this.config.enableQuantization) {
      this.quantizedMemory = null;
      return;
    }

    return this.withErrorHandling('initializeQuantization', () => {
      const memorySlots = this.config.memorySlots;
      const embeddingSize = this.config.memoryDim;

      // Initialize quantization ranges for each dimension
      this.quantizationRanges = Array(embeddingSize).fill(0).map(() => ({
        min: -1.0,
        max: 1.0
      }));

      // Initialize quantized memory
      this.quantizedMemory = {
        shortTerm: new Uint8Array(memorySlots * embeddingSize),
        longTerm: new Uint8Array(Math.floor(memorySlots / 2) * embeddingSize),
        meta: new Uint8Array(memorySlots * 5),
        quantizationRanges: this.quantizationRanges
      };

      this.logger.info('quantization', `Initialized quantized memory with ${this.quantizationBits} bits precision`);
    });
  }

  /**
   * Quantizes a tensor to lower precision
   * @param tensor The tensor to quantize
   * @returns The quantized data as Uint8Array
   */
  private quantizeTensor(tensor: tf.Tensor): Uint8Array {
    return this.withErrorHandling('quantizeTensor', () => {
      const data = tensor.dataSync();
      const shape = tensor.shape;
      const totalElements = shape.reduce((a, b) => a * b, 1);

      // Create quantized array
      const quantized = new Uint8Array(totalElements);

      // Determine quantization range
      const maxValue = 2 ** this.quantizationBits - 1;

      // Update ranges if needed
      if (tensor.rank === 2 && shape[1] === this.config.memoryDim) {
        // For embedding tensors, track per-dimension ranges
        const values = tensor.arraySync() as number[][];

        for (let dim = 0; dim < shape[1]; dim++) {
          let min = Infinity;
          let max = -Infinity;

          // Find min/max for this dimension
          for (let i = 0; i < shape[0]; i++) {
            const val = values[i][dim];
            if (val < min) { min = val; }
            if (val > max) { max = val; }
          }

          // Update range with exponential moving average
          const alpha = 0.1; // Smoothing factor
          this.quantizationRanges[dim].min = (1 - alpha) * this.quantizationRanges[dim].min + alpha * min;
          this.quantizationRanges[dim].max = (1 - alpha) * this.quantizationRanges[dim].max + alpha * max;

          // Quantize values for this dimension
          for (let i = 0; i < shape[0]; i++) {
            const val = values[i][dim];
            const normalized = (val - this.quantizationRanges[dim].min) /
              (this.quantizationRanges[dim].max - this.quantizationRanges[dim].min);
            const quantizedVal = Math.min(maxValue, Math.max(0, Math.round(normalized * maxValue)));
            quantized[i * shape[1] + dim] = quantizedVal;
          }
        }
      } else {
        // For other tensors, use global min/max
        const min = tf.min(tensor).dataSync()[0];
        const max = tf.max(tensor).dataSync()[0];

        // Quantize all values
        for (let i = 0; i < totalElements; i++) {
          const normalized = (data[i] - min) / (max - min);
          const quantizedVal = Math.min(maxValue, Math.max(0, Math.round(normalized * maxValue)));
          quantized[i] = quantizedVal;
        }
      }

      return quantized;
    });
  }

  /**
   * Dequantizes data back to full precision tensor
   * @param quantized The quantized data
   * @param shape The tensor shape
   * @param ranges Optional quantization ranges for per-dimension dequantization
   * @returns The dequantized tensor
   */
  private dequantizeTensor(quantized: Uint8Array, shape: number[], ranges?: Array<{ min: number; max: number }>): tf.Tensor {
    return this.withErrorHandling('dequantizeTensor', () => {
      const totalElements = shape.reduce((a, b) => a * b, 1);
      const dequantized = new Float32Array(totalElements);

      // Determine dequantization parameters
      const maxValue = 2 ** this.quantizationBits - 1;

      if (ranges && shape.length === 2 && shape[1] === this.config.memoryDim) {
        // For embedding tensors, use per-dimension ranges
        for (let i = 0; i < shape[0]; i++) {
          for (let dim = 0; dim < shape[1]; dim++) {
            const quantizedVal = quantized[i * shape[1] + dim];
            const normalized = quantizedVal / maxValue;
            const range = ranges[dim];
            dequantized[i * shape[1] + dim] = normalized * (range.max - range.min) + range.min;
          }
        }
      } else {
        // For other tensors, use global min/max
        const min = -1.0;
        const max = 1.0;

        for (let i = 0; i < totalElements; i++) {
          const normalized = quantized[i] / maxValue;
          dequantized[i] = normalized * (max - min) + min;
        }
      }

      return tf.tensor(dequantized, shape);
    });
  }

  /**
   * Updates quantized memory with new tensor data
   * @param tensor The tensor to store in quantized form
   * @param memoryType The type of memory to update ('shortTerm', 'longTerm', or 'meta')
   */
  private updateQuantizedMemory(tensor: tf.Tensor, memoryType: 'shortTerm' | 'longTerm' | 'meta'): void {
    if (!this.quantizedMemory || !this.config.enableQuantization) {
      return;
    }

    return this.withErrorHandling('updateQuantizedMemory', () => {
      // Quantize the tensor
      const quantized = this.quantizeTensor(tensor);

      // Update the appropriate memory
      // No longer needs indexing as the type is now Uint8Array, not Uint8Array[]
      this.quantizedMemory![memoryType] = quantized;

      // Update quantization ranges
      if (memoryType === 'shortTerm' || memoryType === 'longTerm') {
        this.quantizedMemory!.quantizationRanges = this.quantizationRanges;
      }
    });
  }

  /**
   * Retrieves tensor from quantized memory
   * @param memoryType The type of memory to retrieve ('shortTerm', 'longTerm', or 'meta')
   * @param shape The shape of the tensor to reconstruct
   * @returns The dequantized tensor
   */
  private retrieveQuantizedMemory(memoryType: 'shortTerm' | 'longTerm' | 'meta', shape: number[]): tf.Tensor {
    if (!this.quantizedMemory || !this.config.enableQuantization) {
      throw new MemoryError('Quantized memory not initialized');
    }

    return this.withErrorHandling('retrieveQuantizedMemory', () => {
      // Get the quantized data
      const quantized = this.quantizedMemory![memoryType];

      // Dequantize based on memory type
      if (memoryType === 'shortTerm' || memoryType === 'longTerm') {
        return this.dequantizeTensor(
          quantized, // Pass the Uint8Array directly
          shape,
          this.quantizedMemory!.quantizationRanges
        );
      } else {
        return this.dequantizeTensor(quantized, shape); // Pass the Uint8Array directly
      }
    });
  }

  /**
   * Updates memory with quantization support
   */
  private updateMemory(
    input: tf.Tensor2D,
    surprise: tf.Tensor,
    state: IMemoryState,
    attention?: IAttentionBlock
  ): IMemoryState {
    const result = tf.tidy(() => {
      const shortTerm = state.shortTerm as tf.Tensor2D;
      const momentumMatrix = this.momentumBuffer ?? tf.zerosLike(shortTerm);
      const surpriseScalar = tf.squeeze(surprise);
      const alpha = tf.clipByValue(tf.sigmoid(surpriseScalar), 0.05, 0.95);
      const forgetGate = tf.sub(tf.scalar(1), alpha);

      const inputNorms = tf.add(tf.norm(input, 'euclidean', 1, true), tf.scalar(1e-8));
      const normalizedInput = tf.div(input, inputNorms);
      const keyTranspose = normalizedInput.transpose();
      const projected = tf.matMul(shortTerm, keyTranspose); // [slots, 1]
      const keyBroadcast = tf.tile(normalizedInput, [shortTerm.shape[0], 1]);
      const valueBroadcast = tf.tile(input, [shortTerm.shape[0], 1]);
      const delta = tf.sub(tf.mul(projected, keyBroadcast), valueBroadcast);

      const theta = tf.scalar(this.config.learningRate);
      const eta = tf.scalar(this.config.decayRate);
      const newMomentumMatrix = tf.add(tf.mul(momentumMatrix, eta), tf.mul(delta, tf.neg(theta)));

      // Select slot to update either via attention weights or recency fallback
      let targetIndex = 0;
      if (attention && attention.scores.size > 0) {
        const scoreData = attention.scores.dataSync();
        let best = 0;
        for (let i = 1; i < scoreData.length; i++) {
          if (scoreData[i] > scoreData[best]) { best = i; }
        }
        targetIndex = best;
      } else {
        targetIndex = tf.argMin(state.timestamps).dataSync()[0];
      }

      const indices = tf.tensor2d([[targetIndex]], [1, 1], 'int32');
      const updatedMemorySlice = tf.mul(
        tf.gather(shortTerm, targetIndex),
        forgetGate
      ).reshape([1, -1]);

      const momentumSlice = tf.gather(newMomentumMatrix, targetIndex).reshape([1, -1]);
      const mergedSlice = tf.add(updatedMemorySlice, momentumSlice);

      const updatedShortTerm = tf.tensorScatterUpdate(shortTerm, indices, mergedSlice);
      const updatedMomentum = tf.tensorScatterUpdate(newMomentumMatrix, indices, momentumSlice);

      const currentTime = Date.now();
      const timestampUpdate = tf.tensor1d([currentTime]);
      const updatedTimestamps = tf.tensorScatterUpdate(state.timestamps as tf.Tensor1D, indices, timestampUpdate);

      const accessSlice = tf.add(tf.gather(state.accessCounts as tf.Tensor1D, targetIndex), tf.scalar(1));
      const updatedAccessCounts = tf.tensorScatterUpdate(
        state.accessCounts as tf.Tensor1D,
        indices,
        accessSlice.reshape([1])
      );

      const historyLength = state.surpriseHistory.shape[0];
      const surpriseHistoryTail = historyLength > 1
        ? tf.slice(state.surpriseHistory as tf.Tensor1D, [1], [historyLength - 1])
        : tf.tensor1d([]);
      const newSurpriseEntry = surpriseScalar.reshape([1]);
      const updatedHistory = historyLength > 1
        ? tf.concat([surpriseHistoryTail, newSurpriseEntry], 0)
        : newSurpriseEntry;

      const momentumNorm = tf.norm(momentumSlice);
      const metaRow = tf.stack([
        surpriseScalar,
        forgetGate,
        momentumNorm,
        tf.scalar(currentTime),
        tf.scalar(this.stepCount + 1)
      ]).reshape([1, 5]);
      const updatedMeta = tf.tensorScatterUpdate(state.meta as tf.Tensor2D, indices, metaRow);

      const momentumToStore = this.config.enableMomentum ? tf.keep(updatedMomentum) : null;

      return {
        newState: {
          shortTerm: tf.keep(updatedShortTerm),
          longTerm: tf.keep(state.longTerm.clone()),
          meta: tf.keep(updatedMeta),
          timestamps: tf.keep(updatedTimestamps),
          accessCounts: tf.keep(updatedAccessCounts),
          surpriseHistory: tf.keep(updatedHistory),
          momentumState: this.config.enableMomentum
            ? wrapTensor(momentumToStore as tf.Tensor2D)
            : state.momentumState,
          momentumDecay: state.momentumDecay ?? (this.config.enableMomentum ? this.config.momentumDecayRate : undefined),
          forgettingGate: state.forgettingGate,
          tokenFlowHistory: state.tokenFlowHistory,
          flowWeights: state.flowWeights
        },
        newMomentum: momentumToStore ?? tf.keep(updatedMomentum)
      };
    });

    const { newState, newMomentum } = result;

    if (this.config.enableQuantization && this.quantizedMemory) {
      this.updateQuantizedMemory(newState.shortTerm, 'shortTerm');
      this.updateQuantizedMemory(newState.longTerm, 'longTerm');
      this.updateQuantizedMemory(newState.meta, 'meta');
    }

    if (this.momentumBuffer) {
      this.momentumBuffer.dispose();
    }
    this.momentumBuffer = tf.keep(newMomentum);

    return newState;
  }

  /**
   * Prune memory using information-gain scoring
   * @param threshold Optional threshold for keeping memories (0.0 to 1.0)
   */
  public async pruneMemoryByInformationGain(threshold?: number): Promise<PruningResult> {
    return this.withErrorHandlingAsync('pruneMemoryByInformationGain', async () => {
      // Update pruner configuration if threshold is provided
      if (threshold !== undefined) {
        this.memoryPruner.updateConfig({ keepPercentage: threshold });
      }

      // Check if pruning is needed
      if (!this.memoryPruner.shouldPrune(this.memoryState)) {
        return {
          originalCount: this.getMemorySize(),
          finalCount: this.getMemorySize(),
          distilledCount: 0,
          averageScore: 0,
          reductionRatio: 0,
          newMemoryState: this.memoryState
        };
      }

      // Perform pruning
      const result = await this.memoryPruner.pruneMemory(this.memoryState);

      // Validate the pruned state
      if (!this.memoryPruner.validatePrunedState(result.newMemoryState)) {
        throw new MemoryError('Pruned memory state failed validation');
      }

      // Update the model's memory state
      this.memoryState = result.newMemoryState;

      // Log pruning statistics
      const stats = this.memoryPruner.getPruningStats();
      this.logger.info('pruning', `Memory pruned: ${result.originalCount} -> ${result.finalCount} slots (${(result.reductionRatio * 100).toFixed(1)}% reduction)`);
      this.logger.info('pruning', `Distilled ${result.distilledCount} memories into long-term storage`);
      this.logger.info('pruning', `Average score of kept memories: ${result.averageScore.toFixed(4)}`);
      this.logger.info('pruning', `Total pruning operations: ${stats.totalPrunings}`);

      return result;
    });
  }

  /**
   * Get the current number of active memories
   */
  private getMemorySize(): number {
    return tf.tidy(() => {
      // Count non-zero entries in timestamps as active memories
      const nonZeroMask = tf.greater(this.memoryState.timestamps, 0);
      return tf.sum(tf.cast(nonZeroMask, 'int32')).dataSync()[0];
    });
  }

  /**
   * Get memory pruning statistics
   */
  public getPruningStats(): {
    totalPrunings: number;
    averageReduction: number;
    lastPruningTime: number;
    timeSinceLastPruning: number;
    shouldPrune: boolean;
    currentMemorySize: number;
    maxCapacity: number;
  } {
    const prunerStats = this.memoryPruner.getPruningStats();
    const currentSize = this.getMemorySize();

    return {
      ...prunerStats,
      shouldPrune: this.memoryPruner.shouldPrune(this.memoryState),
      currentMemorySize: currentSize,
      maxCapacity: this.config.memorySlots
    };
  }

  // 1. Implement saveModel and loadModel
  public async saveModel(path: string): Promise<void> {
    // Corrected: Call save with only one argument as per its definition
    await this.save(path);
  }
  public async loadModel(path: string): Promise<void> {
    await this.load(path);
  }
}

// Export alias for workflow compatibility
export const TitanMemorySystem = TitanMemoryModel;
