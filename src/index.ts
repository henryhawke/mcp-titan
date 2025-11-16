// Import polyfills first
import './utils/polyfills.js';

// HOPE Memory Server entry point
import { z } from "zod";
import * as tf from '@tensorflow/tfjs-node';
import type { TensorContainer } from '@tensorflow/tfjs-core';
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const packageJson = require('../package.json') as { version?: string };

import type { IMemoryState, SerializedAuxiliaryMemoryState, ITokenizer } from './types.js';
import { wrapTensor, unwrapTensor } from './types.js';
import { HopeMemoryModel } from './model.js';
export { HopeMemoryModel } from './model.js';
export { AdvancedTokenizer, BPETokenizer, TokenEmbedding, MaskedLanguageModelHead } from './tokenizer/index.js';
export type { TokenizerConfig, TokenizationResult, BPEConfig, EmbeddingConfig, MLMConfig } from './tokenizer/index.js';
import { AdvancedTokenizer } from './tokenizer/index.js';
export { LearnerService, type LearnerConfig } from './learner.js';
import { LearnerService, type LearnerConfig } from './learner.js';
import { VectorProcessor } from './utils.js';
import { TfIdfVectorizer } from './tfidf.js';
import * as path from 'path';
import { promises as fs } from 'fs';
import * as crypto from 'crypto';
import { StructuredLogger, LogLevel } from './logging.js';

/**
 * Represents a serialized memory state that can be stored and loaded.
 */
interface SerializedMemoryState {
  shapes: Record<string, number[]>;
  shortTerm: number[];
  longTerm: number[];
  archive?: number[];
  meta: number[];
  timestamps: number[];
  accessCounts: number[];
  surpriseHistory: number[];
  momentumState?: number[];
  momentumDecay?: number;
  forgettingGate?: number[];
  tokenFlowHistory?: number[];
  flowWeights?: number[];
  levelIndex?: number[];
  surpriseBuffer?: number[];
  auxiliary?: SerializedAuxiliaryMemoryState;
  timestamp?: number;
}

/**
 * Statistics about the memory state.
 */
interface MemoryStats {
  shortTermMean: number;
  shortTermStd: number;
  longTermMean: number;
  longTermStd: number;
  capacity: number;
  surpriseScore: number;
  patternDiversity: number;
}

const versionFromPackage = typeof packageJson.version === 'string' ? packageJson.version : '0.0.0';
export const HOPE_MEMORY_VERSION = versionFromPackage;
export const TITAN_MEMORY_VERSION = HOPE_MEMORY_VERSION;

/**
 * HOPE Memory Server - Continuum memory system that can learn and predict sequences
 * while maintaining state through a retentive continuum memory vector.
 */
export class HopeMemoryServer {
  private server: McpServer;
  private model!: HopeMemoryModel;
  private vectorProcessor: VectorProcessor;
  private memoryState: IMemoryState;
  private learnerService?: LearnerService;
  private tokenizer?: ITokenizer;
  private isInitialized = false;
  private autoSaveInterval?: NodeJS.Timeout;
  private readonly memoryPath: string;
  private readonly modelDir: string;
  private logger: StructuredLogger;
  private readonly toolDescriptions = new Map<string, string>();

  constructor(options: { memoryPath?: string } = {}) {
    this.server = new McpServer({
      name: "HOPE Memory",
      version: HOPE_MEMORY_VERSION
    });
    this.vectorProcessor = VectorProcessor.getInstance();
    this.memoryPath = options.memoryPath ?? path.join(process.cwd(), '.hope_memory');
    this.modelDir = path.join(this.memoryPath, 'model');
    this.memoryState = this.initializeEmptyState();
    this.syncModelState();

    this.logger = StructuredLogger.getInstance(path.join(this.memoryPath, 'logs'));
    this.logger.setLogLevel(process.env.LOG_LEVEL === 'DEBUG' ? LogLevel.DEBUG : LogLevel.INFO);
    this.logger.info('server', 'HopeMemoryServer initialized', {
      memoryPath: this.memoryPath,
      version: HOPE_MEMORY_VERSION
    });

    this.registerTools();
  }

  private syncModelState(): void {
    if (this.model && typeof (this.model as any).hydrateMemoryState === 'function') {
      (this.model as any).hydrateMemoryState(this.memoryState);
    }
  }

  private registerToolDefinition(
    name: string,
    description: string,
    schema: any,
    handler: (params: any) => Promise<any> | any
  ): void {
    this.toolDescriptions.set(name, description);
    this.server.tool(name, description, schema, handler);
  }

  private initializeEmptyState(): IMemoryState {
    if (this.model && typeof (this.model as any).createInitialState === 'function') {
      return (this.model as any).createInitialState();
    }

    return tf.tidy(() => {
      const memoryDim = 256;
      const state: IMemoryState = {
        shortTerm: wrapTensor(tf.tensor2d([], [0, memoryDim])),
        longTerm: wrapTensor(tf.tensor2d([], [0, memoryDim])),
        meta: wrapTensor(tf.tensor2d([], [0, 4])),
        timestamps: wrapTensor(tf.tensor1d([])),
        accessCounts: wrapTensor(tf.tensor1d([])),
        surpriseHistory: wrapTensor(tf.tensor1d([]))
      };
      return state;
    });
  }

  private serializeMemoryState(): SerializedMemoryState {
    const shapes: Record<string, number[]> = {
      shortTerm: unwrapTensor(this.memoryState.shortTerm).shape,
      longTerm: unwrapTensor(this.memoryState.longTerm).shape,
      archive: (this.memoryState as any).archive ? unwrapTensor((this.memoryState as any).archive).shape : [0, 0],
      meta: unwrapTensor(this.memoryState.meta).shape,
      timestamps: unwrapTensor(this.memoryState.timestamps).shape,
      accessCounts: unwrapTensor(this.memoryState.accessCounts).shape,
      surpriseHistory: unwrapTensor(this.memoryState.surpriseHistory).shape
    };

    if (this.memoryState.momentumState) {
      shapes.momentumState = unwrapTensor(this.memoryState.momentumState).shape;
    }
    if (this.memoryState.forgettingGate) {
      shapes.forgettingGate = unwrapTensor(this.memoryState.forgettingGate).shape;
    }
    if (this.memoryState.tokenFlowHistory) {
      shapes.tokenFlowHistory = unwrapTensor(this.memoryState.tokenFlowHistory).shape;
    }
    if (this.memoryState.flowWeights) {
      shapes.flowWeights = unwrapTensor(this.memoryState.flowWeights).shape;
    }

    const serialized: SerializedMemoryState = {
      shapes,
      shortTerm: Array.from(unwrapTensor(this.memoryState.shortTerm).dataSync()),
      longTerm: Array.from(unwrapTensor(this.memoryState.longTerm).dataSync()),
      archive: (this.memoryState as any).archive
        ? Array.from(unwrapTensor((this.memoryState as any).archive).dataSync())
        : undefined,
      meta: Array.from(unwrapTensor(this.memoryState.meta).dataSync()),
      timestamps: Array.from(unwrapTensor(this.memoryState.timestamps).dataSync()),
      accessCounts: Array.from(unwrapTensor(this.memoryState.accessCounts).dataSync()),
      surpriseHistory: Array.from(unwrapTensor(this.memoryState.surpriseHistory).dataSync()),
      levelIndex: (this.memoryState as any).levelIndex
        ? Array.from(unwrapTensor((this.memoryState as any).levelIndex).dataSync())
        : undefined,
      surpriseBuffer: (this.memoryState as any).surpriseBuffer
        ? Array.from(unwrapTensor((this.memoryState as any).surpriseBuffer).dataSync())
        : undefined
    };

    if (this.memoryState.momentumState) {
      serialized.momentumState = Array.from(unwrapTensor(this.memoryState.momentumState).dataSync());
    }
    if (typeof this.memoryState.momentumDecay === 'number') {
      serialized.momentumDecay = this.memoryState.momentumDecay;
    }
    if (this.memoryState.forgettingGate) {
      serialized.forgettingGate = Array.from(unwrapTensor(this.memoryState.forgettingGate).dataSync());
    }
    if (this.memoryState.tokenFlowHistory) {
      serialized.tokenFlowHistory = Array.from(unwrapTensor(this.memoryState.tokenFlowHistory).dataSync());
    }
    if (this.memoryState.flowWeights) {
      serialized.flowWeights = Array.from(unwrapTensor(this.memoryState.flowWeights).dataSync());
    }

    const auxiliary = this.model?.exportAuxiliaryState();
    if (auxiliary && (auxiliary.hierarchicalMemory || auxiliary.extendedMemory)) {
      serialized.auxiliary = auxiliary;
    }

    return serialized;
  }

  private restoreSerializedMemoryState(state: SerializedMemoryState): void {
    const shapes = state.shapes ?? {};

    this.memoryState = tf.tidy(() => {
      const restored: IMemoryState & { archive?: TensorContainer; levelIndex?: TensorContainer; surpriseBuffer?: TensorContainer } = {
        shortTerm: wrapTensor(tf.tensor2d(state.shortTerm, shapes.shortTerm as [number, number] ?? [state.shortTerm.length, 1])),
        longTerm: wrapTensor(tf.tensor2d(state.longTerm, shapes.longTerm as [number, number] ?? [state.longTerm.length, 1])),
        meta: wrapTensor(tf.tensor2d(state.meta, shapes.meta as [number, number] ?? [state.meta.length, 1])),
        timestamps: wrapTensor(tf.tensor1d(state.timestamps)),
        accessCounts: wrapTensor(tf.tensor1d(state.accessCounts)),
        surpriseHistory: wrapTensor(tf.tensor1d(state.surpriseHistory))
      };

      if (state.archive && shapes.archive) {
        restored.archive = wrapTensor(tf.tensor2d(state.archive, shapes.archive as [number, number]));
      }
      if (state.levelIndex) {
        restored.levelIndex = wrapTensor(tf.tensor1d(state.levelIndex));
      }
      if (state.surpriseBuffer) {
        restored.surpriseBuffer = wrapTensor(tf.tensor1d(state.surpriseBuffer));
      }

      if (state.momentumState && shapes.momentumState) {
        restored.momentumState = wrapTensor(tf.tensor2d(state.momentumState, shapes.momentumState as [number, number]));
      }
      if (typeof state.momentumDecay === 'number') {
        restored.momentumDecay = state.momentumDecay;
      }
      if (state.forgettingGate && shapes.forgettingGate) {
        restored.forgettingGate = wrapTensor(tf.tensor(state.forgettingGate, shapes.forgettingGate));
      }
      if (state.tokenFlowHistory && shapes.tokenFlowHistory) {
        restored.tokenFlowHistory = wrapTensor(tf.tensor2d(state.tokenFlowHistory, shapes.tokenFlowHistory as [number, number]));
      }
      if (state.flowWeights && shapes.flowWeights) {
        restored.flowWeights = wrapTensor(tf.tensor(state.flowWeights, shapes.flowWeights));
      }
      return restored;
    });

    this.syncModelState();

    if (state.auxiliary) {
      this.model?.restoreAuxiliaryState(state.auxiliary);
    }
  }

  private wrapWithMemoryManagement<T extends TensorContainer>(fn: () => T): T {
    return tf.tidy(fn);
  }

  private async wrapWithMemoryManagementAsync<T extends TensorContainer>(fn: () => Promise<T>): Promise<T> {
    tf.engine().startScope();
    try {
      return await fn();
    } finally {
      tf.engine().endScope();
    }
  }

  private encryptTensor(tensor: tf.Tensor): Uint8Array {
    const data = tensor.dataSync();
    const key = crypto.randomBytes(32);
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv('aes-256-cbc', key, iv);
    const encrypted = Buffer.concat([cipher.update(Buffer.from(data.buffer)), cipher.final()]);
    return new Uint8Array(Buffer.concat([iv, key, encrypted]));
  }

  private validateMemoryState(state: IMemoryState): boolean {
    return tf.tidy(() => {
      try {
        const validations = [
          state.shortTerm && !unwrapTensor(state.shortTerm).isDisposed,
          state.longTerm && !unwrapTensor(state.longTerm).isDisposed,
          state.meta && !unwrapTensor(state.meta).isDisposed,
          state.timestamps && !unwrapTensor(state.timestamps).isDisposed,
          state.accessCounts && !unwrapTensor(state.accessCounts).isDisposed,
          state.surpriseHistory && !unwrapTensor(state.surpriseHistory).isDisposed
        ];

        return validations.every(Boolean);
      } catch (error) {
        // Silent validation failure
        return false;
      }
    });
  }

  private async ensureInitialized(): Promise<void> {
    if (!this.isInitialized) {
      await this.autoInitialize();
      this.isInitialized = true;
    }
  }

  private registerTools(): void {
    // Help tool
    this.registerToolDefinition(
      'help',
      "Get help about available tools",
      z.object({
        tool: z.string().optional().describe("Specific tool name to get help for"),
        category: z.string().optional().describe("Category of tools to explore"),
        showExamples: z.boolean().optional().describe("Include usage examples"),
        verbose: z.boolean().optional().describe("Include detailed descriptions")
      }),
      async (params: { tool?: string }) => {
        await this.ensureInitialized();

        const entries = Array.from(this.toolDescriptions.entries()).sort((a, b) =>
          a[0].localeCompare(b[0])
        );

        let helpText: string;
        if (params?.tool) {
          const match = entries.find(([name]) => name === params.tool);
          helpText = match
            ? `${match[0]}: ${match[1]}`
            : `Tool "${params.tool}" is not registered.`;
        } else {
          const lines = [
            "Available tools:",
            ...entries.map(([name, description]) => `- ${name}: ${description}`)
          ];
          helpText = lines.join('\n');
        }

        return {
          content: [{
            type: "text",
            text: helpText
          }]
        };
      }
    );

    // Bootstrap memory tool
    this.registerToolDefinition(
      'bootstrap_memory',
      "Initialize memory and train tokenizer based on a given URL or text corpus",
      z.object({
        source: z.union([z.string().url(), z.string()]).describe("URL or text corpus")
      }),
      async (params) => {
        try {
          // Example logic to fetch data and initialize memory
          const documents = await this.fetchDocuments(params.source);

          // Initialize TF-IDF Vectorizer
          const tfidfVectorizer = new TfIdfVectorizer();
          tfidfVectorizer.fit(documents);

          // Store in model instance variables if available
          if (this.model && typeof this.model === 'object') {
            (this.model as any).tfidfVectorizer = tfidfVectorizer;
            (this.model as any).fallbackDocuments = documents;
          }

          // Generate seed summaries for memory initialization
          const seedSummaries: string[] = [];
          for (const doc of documents.slice(0, 50)) { // Limit to first 50 documents
            const summary = await this.summarizeText(doc);
            seedSummaries.push(summary);
          }

          // Populate memory with summarized documents
          await this.ensureInitialized();
          let memoriesAdded = 0;

          for (const summary of seedSummaries) {
            try {
              // Store each summary in the model's memory
              await this.model.storeMemory(summary);
              memoriesAdded++;
            } catch (error) {
              this.logger.warn('bootstrap_memory', 'Failed to store summary in memory', { error: error instanceof Error ? error.message : 'Unknown error' });
            }
          }

          // Train the tokenizer with documents if advanced tokenizer is available
          if (this.model && (this.model as any).advancedTokenizer) {
            try {
              const tokenizer = (this.model as any).advancedTokenizer;
              // Bootstrap the tokenizer with some of the documents
              for (const doc of documents.slice(0, 20)) {
                await tokenizer.encode(doc);
              }
            } catch (error) {
              this.logger.warn('bootstrap_memory', 'Failed to train tokenizer', { error: error instanceof Error ? error.message : 'Unknown error' });
            }
          }

          return {
            content: [{
              type: "text",
              text: `Memory bootstrap completed successfully. Added ${memoriesAdded} seed memories from ${documents.length} documents. TF-IDF vectorizer initialized for sparse fallback.`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to bootstrap memory: ${message}`
            }]
          };
        }
      }
    );

    // Init model tool
    this.registerToolDefinition(
      'init_model',
      "Initialize the HOPE Memory model",
      z.object({
        inputDim: z.number().int().positive().default(768).describe("Input dimension size"),
        hiddenDim: z.number().int().positive().default(512).describe("Hidden dimension size"),
        memoryDim: z.number().int().positive().default(1024).describe("Memory dimension size"),
        transformerLayers: z.number().int().positive().default(6).describe("Number of transformer layers"),
        numHeads: z.number().int().positive().default(8).describe("Number of attention heads"),
        ffDimension: z.number().int().positive().default(2048).describe("Feed-forward dimension"),
        dropoutRate: z.number().min(0).max(0.9).default(0.1).describe("Dropout rate"),
        maxSequenceLength: z.number().int().positive().default(512).describe("Maximum sequence length"),
        memorySlots: z.number().int().positive().default(5000).describe("Number of memory slots"),
        similarityThreshold: z.number().min(0).max(1).default(0.65).describe("Similarity threshold"),
        surpriseDecay: z.number().min(0).max(1).default(0.9).describe("Surprise decay rate"),
        pruningInterval: z.number().int().positive().default(1000).describe("Pruning interval"),
        gradientClip: z.number().positive().default(1.0).describe("Gradient clipping value")
      }),
      async (params) => {
        try {
          this.model = new HopeMemoryModel();
          const config = {
            inputDim: params.inputDim,
            hiddenDim: params.hiddenDim ?? 512,
            memoryDim: params.memoryDim ?? 1024,
            transformerLayers: params.transformerLayers,
            numHeads: params.numHeads ?? 8,
            ffDimension: params.ffDimension ?? 2048,
            dropoutRate: params.dropoutRate ?? 0.1,
            maxSequenceLength: params.maxSequenceLength ?? 512,
            memorySlots: params.memorySlots,
            similarityThreshold: params.similarityThreshold ?? 0.65,
            surpriseDecay: params.surpriseDecay ?? 0.9,
            pruningInterval: params.pruningInterval ?? 1000,
            gradientClip: params.gradientClip ?? 1.0,
            enableHierarchicalMemory: true,
            useHierarchicalMemory: true,
            enableEpisodicSemanticDistinction: true
          };

          await this.model.initialize(config);
          this.memoryState = this.initializeEmptyState();
          this.syncModelState();
          this.isInitialized = true;

          return {
            content: [{
              type: "text",
              text: `HOPE Memory Model initialized successfully with configuration: ${JSON.stringify(config, null, 2)}`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to initialize model: ${message}`
            }]
          };
        }
      }
    );

    // Memory stats tool
    this.registerToolDefinition(
      'memory_stats',
      "Dump raw memory tensors and derived statistics",
      z.object({}),
      async () => {
        await this.ensureInitialized();
        const stats = this.getMemoryStats();
        const shapes = {
          shortTerm: unwrapTensor(this.memoryState.shortTerm).shape,
          longTerm: unwrapTensor(this.memoryState.longTerm).shape,
          meta: unwrapTensor(this.memoryState.meta).shape
        };

        const summary = {
          stats,
          shapes
        };

        return {
          content: [{
            type: "text",
            text: JSON.stringify(summary, null, 2)
          }]
        };
      }
    );

    // Forward pass tool
    this.registerToolDefinition(
      'forward_pass',
      "Perform a forward pass through the model with given input",
      z.object({
        x: z.union([z.string(), z.array(z.number())]).describe("Input data (text or number array)"),
        memoryState: z.any().optional().describe("Optional memory state")
      }),
      async (params) => {
        await this.ensureInitialized();

        try {
          const input = await this.processInput(params.x);
          const result = this.model.forward(wrapTensor(input), this.memoryState);

          const predicted = Array.from(unwrapTensor(result.predicted).dataSync());
          const memoryUpdate = {
            shortTerm: Array.from(unwrapTensor(result.memoryUpdate.newState.shortTerm).dataSync()),
            longTerm: Array.from(unwrapTensor(result.memoryUpdate.newState.longTerm).dataSync()),
            meta: Array.from(unwrapTensor(result.memoryUpdate.newState.meta).dataSync()),
            timestamps: Array.from(unwrapTensor(result.memoryUpdate.newState.timestamps).dataSync()),
            accessCounts: Array.from(unwrapTensor(result.memoryUpdate.newState.accessCounts).dataSync()),
            surpriseHistory: Array.from(unwrapTensor(result.memoryUpdate.newState.surpriseHistory).dataSync())
          };

          // Update memory state
          this.memoryState = result.memoryUpdate.newState;
          this.syncModelState();
          this.syncModelState();

          input.dispose();

          return {
            content: [{
              type: "text",
              text: `Forward pass completed. Predicted: [${predicted.slice(0, 10).map(x => x.toFixed(4)).join(', ')}${predicted.length > 10 ? '...' : ''}]`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Forward pass failed: ${message}`
            }]
          };
        }
      }
    );

    // Train step tool
    this.registerToolDefinition(
      'train_step',
      "Execute a training step with current and next inputs",
      z.object({
        x_t: z.union([z.string(), z.array(z.number())]).describe("Current input"),
        x_next: z.union([z.string(), z.array(z.number())]).describe("Next expected input")
      }),
      async (params) => {
        await this.ensureInitialized();

        try {
          const currentInput = await this.processInput(params.x_t);
          const nextInput = await this.processInput(params.x_next);

          // Validate dimensions match
          if (currentInput.shape[0] !== nextInput.shape[0]) {
            currentInput.dispose();
            nextInput.dispose();
            return {
              content: [{
                type: "text",
                text: `Training step failed: Input dimensions don't match. x_t has ${currentInput.shape[0]} elements, x_next has ${nextInput.shape[0]} elements.`
              }]
            };
          }

          const result = this.model.trainStep(
            wrapTensor(currentInput),
            wrapTensor(nextInput),
            this.memoryState
          );

          this.memoryState = result.memoryUpdate.newState;

          const loss = unwrapTensor(result.loss).dataSync()[0];

          currentInput.dispose();
          nextInput.dispose();

          return {
            content: [{
              type: "text",
              text: `Training step completed. Loss: ${loss.toFixed(6)}`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Training step failed: ${message}`
            }]
          };
        }
      }
    );

    // Get memory state tool
    this.registerToolDefinition(
      'get_memory_state',
      "Get current memory state statistics and information",
      z.object({}),
      async () => {
        await this.ensureInitialized();

        try {
          const stats = this.getMemoryStats();
          const health = await this.performHealthCheck('quick');

          return {
            content: [{
              type: "text",
              text: `Memory State:
- Short-term mean: ${stats.shortTermMean.toFixed(4)}
- Long-term mean: ${stats.longTermMean.toFixed(4)}
- Capacity: ${(stats.capacity * 100).toFixed(1)}%
- Surprise score: ${stats.surpriseScore.toFixed(4)}
- Pattern diversity: ${stats.patternDiversity.toFixed(4)}
- Health status: ${health.status || 'unknown'}`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to get memory state: ${message}`
            }]
          };
        }
      }
    );

    // Token flow diagnostics
    this.registerToolDefinition(
      'get_token_flow_metrics',
      "Get token flow analysis and statistics",
      z.object({}),
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

          const historyTensor = unwrapTensor(this.memoryState.tokenFlowHistory) as tf.Tensor2D;
          const weightsTensor = unwrapTensor(this.memoryState.flowWeights) as tf.Tensor1D;

          const metrics = tf.tidy(() => {
            const averageWeight = tf.mean(weightsTensor).dataSync()[0];
            const maxWeight = tf.max(weightsTensor).dataSync()[0];
            const minWeight = tf.min(weightsTensor).dataSync()[0];
            const flowStrength = tf.sum(weightsTensor).dataSync()[0];
            const variance = tf.moments(weightsTensor).variance.dataSync()[0];
            return {
              windowSize: historyTensor.shape[0],
              featureSize: historyTensor.shape[1] ?? 0,
              averageWeight,
              maxWeight,
              minWeight,
              flowStrength,
              weightVariance: variance
            };
          });

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

    // Hierarchical memory metrics tool
    this.registerToolDefinition(
      'get_hierarchical_metrics',
      "Get hierarchical memory promotion/demotion statistics",
      z.object({}),
      async () => {
        await this.ensureInitialized();

        try {
          const config = this.model.getConfig();

          if (!config.useHierarchicalMemory && !config.enableHierarchicalMemory) {
            return {
              content: [{
                type: "text",
                text: "Hierarchical memory not enabled. Initialize with enableHierarchicalMemory: true"
              }]
            };
          }

          const stats = (this.model as any).memoryStats;
          const shortTermSize = unwrapTensor(this.memoryState.shortTerm).shape[0];
          const longTermSize = unwrapTensor(this.memoryState.longTerm).shape[0];

          const metrics = {
            promotions: stats.promotions,
            demotions: stats.demotions,
            lastUpdate: new Date(stats.lastStatsUpdate).toISOString(),
            shortTermSize,
            longTermSize,
            totalMemories: shortTermSize + longTermSize,
            promotionRate: stats.promotions.total > 0 ?
              `${(stats.promotions.recent / stats.promotions.total * 100).toFixed(1)}%` : '0%',
            demotionRate: stats.demotions.total > 0 ?
              `${(stats.demotions.recent / stats.demotions.total * 100).toFixed(1)}%` : '0%'
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

    // Reset gradients tool
    this.registerToolDefinition(
      'reset_gradients',
      "Reset accumulated gradients in the model",
      z.object({}),
      async () => {
        await this.ensureInitialized();

        try {
          this.model.resetGradients();
          return {
            content: [{
              type: "text",
              text: "Gradients reset successfully"
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to reset gradients: ${message}`
            }]
          };
        }
      }
    );

    // Health check tool
    this.registerToolDefinition(
      'health_check',
      "Get system health status and diagnostics",
      z.object({
        detailed: z.boolean().optional().describe("Include detailed diagnostics")
      }),
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

    // Prune memory tool with information-gain scoring
    this.registerToolDefinition(
      'prune_memory',
      "Prune memory using information-gain scoring to remove less relevant memories",
      z.object({
        threshold: z.number().min(0).max(1).optional().describe("Percentage of memories to keep (0.0 to 1.0)"),
        force: z.boolean().optional().default(false).describe("Force pruning even if not needed")
      }),
      async (params) => {
        await this.ensureInitialized();

        try {
          // Check if the model supports the pruning method
          if (!this.model.pruneMemoryByInformationGain) {
            return {
              content: [{
                type: "text",
                text: "Memory pruning is not supported by this model version"
              }]
            };
          }

          // Get current memory stats before pruning
          const beforeStats = this.model.getPruningStats();

          // Perform pruning
          await this.model.pruneMemoryByInformationGain(params.threshold);

          // Get stats after pruning
          const afterStats = this.model.getPruningStats();

          const originalCount = beforeStats.shortTerm + beforeStats.longTerm + beforeStats.archive;
          const finalCount = afterStats.shortTerm + afterStats.longTerm + afterStats.archive;
          const reduction = originalCount > 0 ? ((originalCount - finalCount) / originalCount * 100) : 0;

          const message = [
            `Memory pruning completed successfully:`,
            `• Original count: ${originalCount} memories`,
            `• Final count: ${finalCount} memories`,
            `• Memories pruned: ${originalCount - finalCount}`,
            `• Reduction: ${reduction.toFixed(1)}%`,
            `• Average surprise: ${afterStats.averageSurprise.toFixed(4)}`,
            `• Short-term: ${afterStats.shortTerm}, Long-term: ${afterStats.longTerm}, Archive: ${afterStats.archive}`
          ].join('\n');

          return {
            content: [{
              type: "text",
              text: message
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to prune memory: ${message}`
            }]
          };
        }
      }
    );

    // Save checkpoint tool
    this.registerToolDefinition(
      'save_checkpoint',
      "Save current memory state to a checkpoint file",
      z.object({
        path: z.string().describe("Path to save the checkpoint")
      }),
      async (params) => {
        await this.ensureInitialized();

        try {
          // Validate and sanitize the file path
          const validatedPath = this.validateFilePath(params.path);

          const memoryState = this.serializeMemoryState();
          memoryState.timestamp = Date.now();

          const checkpointData = {
            memoryState,
            inputDim: this.model.getConfig().inputDim, // Add for validation on load
            config: this.model.getConfig(),
            version: HOPE_MEMORY_VERSION,
            timestamp: Date.now()
          };

          await fs.mkdir(path.dirname(validatedPath), { recursive: true });
          await fs.writeFile(validatedPath, JSON.stringify(checkpointData, null, 2));

          return {
            content: [{
              type: "text",
              text: `Checkpoint saved to ${validatedPath}`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to save checkpoint: ${message}`
            }]
          };
        }
      }
    );

    // Load checkpoint tool
    this.registerToolDefinition(
      'load_checkpoint',
      "Load memory state from a checkpoint file",
      z.object({
        path: z.string().describe("Path to the checkpoint file")
      }),
      async (params) => {
        try {
          // Validate and sanitize the file path
          const validatedPath = this.validateFilePath(params.path);

          const data = await fs.readFile(validatedPath, 'utf-8');
          const checkpointData = JSON.parse(data) as {
            memoryState?: SerializedMemoryState | {
              shortTerm: number[];
              longTerm: number[];
              meta: number[];
              timestamps: number[];
              accessCounts: number[];
              surpriseHistory: number[];
            };
            shapes?: Record<string, number[]>;
            inputDim?: number;
          };

          // Validate embedding dimensions match if specified
          if (checkpointData.inputDim && this.model) {
            const currentInputDim = this.model.getConfig().inputDim;
            if (checkpointData.inputDim !== currentInputDim) {
              return {
                content: [{
                  type: "text",
                  text: `Checkpoint dimension mismatch: checkpoint has inputDim=${checkpointData.inputDim}, but model has inputDim=${currentInputDim}. Please reinitialize model with matching dimensions.`
                }]
              };
            }
          }

          let serializedState: SerializedMemoryState | undefined;
          if (checkpointData.memoryState && 'shapes' in (checkpointData.memoryState as SerializedMemoryState)) {
            serializedState = checkpointData.memoryState as SerializedMemoryState;
          } else if (checkpointData.memoryState) {
            const legacy = checkpointData.memoryState as {
              shortTerm: number[];
              longTerm: number[];
              meta: number[];
              timestamps: number[];
              accessCounts: number[];
              surpriseHistory: number[];
            };
            serializedState = {
              shapes: checkpointData.shapes ?? {},
              shortTerm: legacy.shortTerm,
              longTerm: legacy.longTerm,
              meta: legacy.meta,
              timestamps: legacy.timestamps,
              accessCounts: legacy.accessCounts,
              surpriseHistory: legacy.surpriseHistory
            };
          }

          if (serializedState) {
            this.restoreSerializedMemoryState(serializedState);
          }

          return {
            content: [{
              type: "text",
              text: `Checkpoint loaded from ${validatedPath}`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to load checkpoint: ${message}`
            }]
          };
        }
      }
    );

    // Initialize learner service tool
    this.registerToolDefinition(
      'init_learner',
      "Initialize the online learning service with specified configuration",
      z.object({
        bufferSize: z.number().int().positive().default(10000).describe("Replay buffer size"),
        batchSize: z.number().int().positive().default(32).describe("Training batch size"),
        updateInterval: z.number().int().positive().default(1000).describe("Update interval in milliseconds"),
        gradientClipValue: z.number().positive().default(1.0).describe("Gradient clipping value"),
        contrastiveWeight: z.number().min(0).max(1).default(0.2).describe("Contrastive learning weight"),
        nextTokenWeight: z.number().min(0).max(1).default(0.4).describe("Next token prediction weight"),
        mlmWeight: z.number().min(0).max(1).default(0.4).describe("Masked language modeling weight"),
        accumulationSteps: z.number().int().positive().default(4).describe("Gradient accumulation steps"),
        learningRate: z.number().positive().default(0.0001).describe("Learning rate"),
        nanGuardThreshold: z.number().positive().default(1e-6).describe("NaN guard threshold")
      }),
      async (params) => {
        await this.ensureInitialized();

        try {
          // Initialize tokenizer if not already done
          if (!this.tokenizer) {
            // TODO: Replace with proper AdvancedTokenizer integration
            // Current limitation: AdvancedTokenizer.encode() is async, but ITokenizer requires sync
            // For production use, refactor learner service to be async or pre-initialize tokenizer
            this.logger.warn('Using fallback tokenizer. For production, integrate AdvancedTokenizer properly.');

            throw new Error(
              'Tokenizer not initialized. The learner service requires a properly initialized tokenizer. ' +
              'This is a known limitation - AdvancedTokenizer is async but the current interface is sync. ' +
              'Please initialize the tokenizer before calling start_learning, or refactor to async.'
            );
          }

          const learnerConfig: Partial<LearnerConfig> = {
            bufferSize: params.bufferSize,
            batchSize: params.batchSize,
            updateInterval: params.updateInterval,
            gradientClipValue: params.gradientClipValue,
            contrastiveWeight: params.contrastiveWeight,
            nextTokenWeight: params.nextTokenWeight,
            mlmWeight: params.mlmWeight,
            accumulationSteps: params.accumulationSteps,
            learningRate: params.learningRate,
            nanGuardThreshold: params.nanGuardThreshold
          };

          this.learnerService = new LearnerService(this.model, this.tokenizer, learnerConfig);

          return {
            content: [{
              type: "text",
              text: `Learner service initialized successfully with configuration: ${JSON.stringify(learnerConfig, null, 2)}`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to initialize learner service: ${message}`
            }]
          };
        }
      }
    );

    // Pause learner tool
    this.registerToolDefinition(
      'pause_learner',
      "Pause the online learning loop",
      z.object({}),
      async () => {
        try {
          if (!this.learnerService) {
            return {
              content: [{
                type: "text",
                text: "Learner service not initialized. Please run init_learner first."
              }]
            };
          }

          this.learnerService.pauseTraining();

          return {
            content: [{
              type: "text",
              text: "Online learning loop paused successfully"
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to pause learner: ${message}`
            }]
          };
        }
      }
    );

    // Resume learner tool
    this.registerToolDefinition(
      'resume_learner',
      "Resume the online learning loop",
      z.object({}),
      async () => {
        try {
          if (!this.learnerService) {
            return {
              content: [{
                type: "text",
                text: "Learner service not initialized. Please run init_learner first."
              }]
            };
          }

          this.learnerService.resumeTraining();

          return {
            content: [{
              type: "text",
              text: "Online learning loop resumed successfully"
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to resume learner: ${message}`
            }]
          };
        }
      }
    );

    // Get learner stats tool
    this.registerToolDefinition(
      'get_learner_stats',
      "Get statistics about the online learning service",
      z.object({}),
      async () => {
        try {
          if (!this.learnerService) {
            return {
              content: [{
                type: "text",
                text: "Learner service not initialized. Please run init_learner first."
              }]
            };
          }

          const stats = this.learnerService.getTrainingStats();

          return {
            content: [{
              type: "text",
              text: `Learner Statistics:
- Buffer size: ${stats.bufferSize}
- Step count: ${stats.stepCount}
- Is running: ${stats.isRunning}
- Average loss: ${stats.averageLoss.toFixed(6)}
- Last loss: ${stats.lastLoss.toFixed(6)}`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to get learner stats: ${message}`
            }]
          };
        }
      }
    );

    // Add training sample tool
    this.registerToolDefinition(
      'add_training_sample',
      "Add a training sample to the replay buffer",
      z.object({
        input: z.union([z.string(), z.array(z.number())]).describe("Input data (text or number array)"),
        target: z.union([z.string(), z.array(z.number())]).describe("Target data (text or number array)"),
        positive: z.union([z.string(), z.array(z.number())]).optional().describe("Positive sample for contrastive learning"),
        negative: z.union([z.string(), z.array(z.number())]).optional().describe("Negative sample for contrastive learning")
      }),
      async (params) => {
        const createdTensors: tf.Tensor[] = [];

        const input = Array.isArray(params.input)
          ? (() => { const tensor = tf.tensor1d(params.input); createdTensors.push(tensor); return tensor; })()
          : params.input;

        const target = Array.isArray(params.target)
          ? (() => { const tensor = tf.tensor1d(params.target); createdTensors.push(tensor); return tensor; })()
          : params.target;

        const positive = params.positive === undefined
          ? undefined
          : Array.isArray(params.positive)
            ? (() => { const tensor = tf.tensor1d(params.positive!); createdTensors.push(tensor); return tensor; })()
            : params.positive;

        const negative = params.negative === undefined
          ? undefined
          : Array.isArray(params.negative)
            ? (() => { const tensor = tf.tensor1d(params.negative!); createdTensors.push(tensor); return tensor; })()
            : params.negative;

        try {
          if (!this.learnerService) {
            return {
              content: [{
                type: "text",
                text: "Learner service not initialized. Please run init_learner first."
              }]
            };
          }

          await this.learnerService.addTrainingSample(input, target, positive, negative);

          const stats = this.learnerService.getTrainingStats();

          return {
            content: [{
              type: "text",
              text: `Training sample added successfully. Buffer size: ${stats.bufferSize}`
            }]
          };
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          return {
            content: [{
              type: "text",
              text: `Failed to add training sample: ${message}`
            }]
          };
        } finally {
          for (const tensor of createdTensors) {
            tensor.dispose();
          }
        }
      }
    );
  }

  /**
   * Validate and sanitize file paths to prevent path traversal attacks
   */
  private validateFilePath(filePath: string): string {
    // Remove any null bytes
    const sanitized = filePath.replace(/\0/g, '');

    // Resolve to absolute path
    const resolved = path.resolve(sanitized);

    // Check for path traversal attempts
    if (resolved.includes('..')) {
      throw new Error('Path traversal detected: .. not allowed in paths');
    }

    // Ensure path is within allowed directories (memory path or current working directory)
    const memoryPathResolved = path.resolve(this.memoryPath);
    const cwdResolved = path.resolve(process.cwd());

    if (!resolved.startsWith(memoryPathResolved) && !resolved.startsWith(cwdResolved)) {
      throw new Error(`Access denied: path must be within ${this.memoryPath} or ${process.cwd()}`);
    }

    return resolved;
  }

  private async processInput(input: string | number[]): Promise<tf.Tensor1D> {
    // Type guard and validation
    if (typeof input === 'string') {
      // Validate string input
      if (input.length === 0) {
        throw new Error('Input string cannot be empty');
      }
      if (input.length > 10000) {
        throw new Error('Input string exceeds maximum length of 10000 characters');
      }
      // Sanitize input by removing control characters except newlines and tabs
      const sanitized = input.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');
      return await this.model.encodeText(sanitized);
    } else if (Array.isArray(input)) {
      // Validate array input
      if (input.length === 0) {
        throw new Error('Input array cannot be empty');
      }
      if (input.length > this.model.getConfig().inputDim * 10) {
        throw new Error(`Input array exceeds maximum length of ${this.model.getConfig().inputDim * 10}`);
      }
      // Validate all elements are numbers
      if (!input.every(x => typeof x === 'number' && !isNaN(x) && isFinite(x))) {
        throw new Error('Input array must contain only valid finite numbers');
      }
      return tf.tensor1d(input);
    } else {
      throw new Error(`Invalid input type: expected string or number array, got ${typeof input}`);
    }
  }

  private async autoInitialize(): Promise<void> {
    try {
      await fs.mkdir(this.memoryPath, { recursive: true });
      const modelMetadataPath = path.join(this.modelDir, 'model.json');
      // Try to load existing model
      const modelExists = await fs.access(modelMetadataPath).then(() => true).catch(() => false);

      if (modelExists) {
        this.model = new HopeMemoryModel();
        await this.model.loadModel(this.modelDir);
      } else {
        // Initialize with default config
        this.model = new HopeMemoryModel();
        await this.model.initialize({
          inputDim: 768,
          memorySlots: 5000,
          transformerLayers: 6,
          enableHierarchicalMemory: true,
          useHierarchicalMemory: true,
          enableEpisodicSemanticDistinction: true
        });

        // Ensure directory exists and save
        await fs.mkdir(this.modelDir, { recursive: true });
        await this.model.save(this.modelDir);
      }

      this.memoryState = this.initializeEmptyState();
      this.syncModelState();
      this.syncModelState();

      // Try to load existing memory state
      const memoryStateExists = await fs.access(path.join(this.memoryPath, 'memory_state.json')).then(() => true).catch(() => false);
      if (memoryStateExists) {
        await this.loadMemoryState();
      }

      // Setup auto-save with proper error logging
      this.autoSaveInterval = setInterval(async () => {
        try {
          await this.saveMemoryState();
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          this.logger.error('autosave', 'Failed to save memory state', error instanceof Error ? error : new Error(message));
          // Attempt retry after delay if it's not a critical error
          if (!message.includes('ENOSPC') && !message.includes('EACCES')) {
            setTimeout(async () => {
              try {
                await this.saveMemoryState();
                this.logger.info('autosave', 'Retry successful');
              } catch (retryError) {
                this.logger.error('autosave', 'Retry also failed');
              }
            }, 5000); // Retry after 5 seconds
          }
        }
      }, 60000); // Save every minute

    } catch (error) {
      // Silent auto-initialization failure
      // Continue with basic initialization
      this.model = new HopeMemoryModel();
      await this.model.initialize({
        inputDim: 768,
        memorySlots: 5000,
        transformerLayers: 6,
        enableHierarchicalMemory: true,
        useHierarchicalMemory: true,
        enableEpisodicSemanticDistinction: true
      });
      this.memoryState = this.initializeEmptyState();
      this.syncModelState();
    }
  }

  private async saveMemoryState(): Promise<void> {
    try {
      await fs.mkdir(this.memoryPath, { recursive: true });
      const serialized = this.serializeMemoryState();
      serialized.timestamp = Date.now();

      await fs.writeFile(
        path.join(this.memoryPath, 'memory_state.json'),
        JSON.stringify(serialized, null, 2)
      );
    } catch (error) {
      // Silent failure
    }
  }

  private async loadMemoryState(): Promise<void> {
    try {
      const data = await fs.readFile(path.join(this.memoryPath, 'memory_state.json'), 'utf-8');
      const state = JSON.parse(data) as SerializedMemoryState;
      this.restoreSerializedMemoryState(state);
    } catch (error) {
      // Silent failure - continue with default state
    }
  }

  public async run(): Promise<void> {
    try {
      const transport = new StdioServerTransport();
      await this.server.connect(transport);

      // Setup graceful shutdown
      process.on('SIGINT', () => this.shutdown());
      process.on('SIGTERM', () => this.shutdown());

      // Server running on stdio
    } catch (error) {
      // Failed to start server
      process.exit(1);
    }
  }

  private async shutdown(): Promise<void> {
    try {
      // Stop learner service if running
      if (this.learnerService) {
        this.learnerService.dispose();
      }

      // Clear auto-save interval
      if (this.autoSaveInterval) {
        clearInterval(this.autoSaveInterval);
      }

      // Save final state
      await this.saveMemoryState();

      // Dispose model
      if (this.model) {
        this.model.dispose();
      }

      process.exit(0);
    } catch (error) {
      process.exit(1);
    }
  }

  private getMemoryStats(): MemoryStats {
    return tf.tidy(() => {
      const shortTermTensor = unwrapTensor(this.memoryState.shortTerm) as tf.Tensor;
      const longTermTensor = unwrapTensor(this.memoryState.longTerm) as tf.Tensor;
      const surpriseTensor = unwrapTensor(this.memoryState.surpriseHistory) as tf.Tensor;

      let shortTermMean = 0;
      let shortTermStd = 0;
      if (shortTermTensor.size > 0) {
        const { mean, variance } = tf.moments(shortTermTensor);
        shortTermMean = mean.dataSync()[0];
        shortTermStd = tf.sqrt(variance).dataSync()[0];
      }

      let longTermMean = 0;
      let longTermStd = 0;
      if (longTermTensor.size > 0) {
        const { mean, variance } = tf.moments(longTermTensor);
        longTermMean = mean.dataSync()[0];
        longTermStd = tf.sqrt(variance).dataSync()[0];
      }

      const surpriseScore = surpriseTensor.size > 0
        ? tf.mean(surpriseTensor).dataSync()[0]
        : 0;

      const shortTermRows = shortTermTensor.shape[0] ?? 0;
      const longTermRows = longTermTensor.shape[0] ?? 0;
      const totalRows = shortTermRows + longTermRows;
      const memorySlots = this.model?.getConfig()?.memorySlots ?? Math.max(totalRows, 1);
      const capacity = memorySlots > 0 ? Math.min(totalRows / memorySlots, 1) : 0;

      const patternDiversity = (shortTermStd + longTermStd) / 2;

      return {
        shortTermMean,
        shortTermStd,
        longTermMean,
        longTermStd,
        capacity,
        surpriseScore,
        patternDiversity
      };
    });
  }

  private async performHealthCheck(checkType: string): Promise<any> {
    const startTime = Date.now();

    const health: any = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      version: HOPE_MEMORY_VERSION,
      checkType
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
        numBytesInGPU: tfMemory.numBytesInGPU || 0,
        numDataBuffers: tfMemory.numDataBuffers
      };

      if (tfMemory.numTensors > 1000) {
        health.status = 'degraded';
        health.warnings = health.warnings || [];
        health.warnings.push('High tensor count - possible memory leak');
      }

      // Check Node.js memory
      const processMemory = process.memoryUsage();
      health.process = {
        heapUsed: `${Math.round(processMemory.heapUsed / 1024 / 1024)} MB`,
        heapTotal: `${Math.round(processMemory.heapTotal / 1024 / 1024)} MB`,
        external: `${Math.round(processMemory.external / 1024 / 1024)} MB`,
        rss: `${Math.round(processMemory.rss / 1024 / 1024)} MB`
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
          longTermMean: memStats.longTermMean.toFixed(4),
          patternDiversity: memStats.patternDiversity.toFixed(4)
        };

        if (memStats.capacity > 0.9) {
          health.warnings = health.warnings || [];
          health.warnings.push('Memory capacity > 90% - consider pruning');
        }
      }

      if (checkType === 'detailed') {
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
          const testInput = tf.randomNormal([this.model?.getConfig().inputDim || 128]);
          const testResult = this.model?.forward(wrapTensor(testInput), this.memoryState);
          testInput.dispose();
          if (testResult) {
            unwrapTensor(testResult.predicted).dispose();
          }
          health.operations = { forward: 'ok' };
        } catch (error) {
          health.operations = { forward: 'failed' };
          health.status = 'unhealthy';
          health.errors = health.errors || [];
          health.errors.push(`Operation test failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
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

  private calculateHealthScore(healthData: any): number {
    // Simple health score calculation
    let score = 1.0;

    if (healthData.tensors > 1000) { score -= 0.3; }
    if (healthData.capacity > 0.8) { score -= 0.2; }

    return Math.max(0, score);
  }

  private generateHealthRecommendations(healthData: any, healthScore: number): string[] {
    const recommendations = [];

    if (healthData.tensors > 1000) {
      recommendations.push("Consider running tensor cleanup - high tensor count detected");
    }

    if (healthData.capacity > 0.8) {
      recommendations.push("Memory capacity is high - consider pruning old memories");
    }

    if (healthScore < 0.7) {
      recommendations.push("Overall health is low - consider running optimization");
    }

    return recommendations;
  }

  /**
   * Fetches documents from a URL or processes text corpus for memory bootstrap
   */
  private async fetchDocuments(source: string): Promise<string[]> {
    try {
      // Check if source is a URL
      if (source.startsWith('http://') || source.startsWith('https://')) {
        // Fetch content from URL
        const response = await fetch(source);
        if (!response.ok) {
          throw new Error(`Failed to fetch URL: ${response.status} ${response.statusText}`);
        }

        const content = await response.text();

        // Simple text processing: split into sentences and paragraphs
        const sentences = content
          .replace(/\n{2,}/g, '\n') // Normalize line breaks
          .split(/[.!?]+/) // Split on sentence endings
          .map(s => s.trim())
          .filter(s => s.length > 10) // Filter out very short sentences
          .slice(0, 1000); // Limit to first 1000 sentences

        return sentences;
      } else {
        // Treat as text corpus
        const sentences = source
          .replace(/\n{2,}/g, '\n')
          .split(/[.!?]+/)
          .map(s => s.trim())
          .filter(s => s.length > 10)
          .slice(0, 1000);

        return sentences;
      }
    } catch (error) {
      throw new Error(`Failed to process source: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Summarizes text using a simple heuristic approach
   * TODO: Replace with actual LLM summarizer when available
   */
  private async summarizeText(text: string): Promise<string> {
    // Simple heuristic summarization:
    // 1. Take first and last sentences
    // 2. Find sentences with keywords
    // 3. Limit to reasonable length

    const sentences = text.split(/[.!?]+/).map(s => s.trim()).filter(s => s.length > 10);

    if (sentences.length <= 3) {
      return text;
    }

    const keywords = ['important', 'key', 'main', 'primary', 'essential', 'critical', 'significant'];
    const keywordSentences = sentences.filter(s =>
      keywords.some(kw => s.toLowerCase().includes(kw))
    );

    const summary = [
      sentences[0], // First sentence
      ...keywordSentences.slice(0, 2), // Up to 2 keyword sentences
      sentences[sentences.length - 1] // Last sentence
    ].join('. ');

    return summary.slice(0, 500); // Limit to 500 characters
  }
}

// Create and run server
const server = new HopeMemoryServer();
server.run().catch(() => process.exit(1));
