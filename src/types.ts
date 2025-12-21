/**
 * @fileoverview Core type definitions for Titan Memory Architecture
 * This file defines the interfaces and types used throughout the implementation
 * of the Titans memory model, including tensor operations, memory states, and
 * model interactions.
 */

import type * as tf from '@tensorflow/tfjs-node';
import { z } from 'zod';

// Core Tensor Operations
export type ITensor = tf.Tensor;
export interface TensorContainer { [key: string]: tf.Tensor | TensorContainer }

/**
 * Creates a wrapped tensor from a TensorFlow.js tensor.
 * @param tensor TensorFlow.js tensor to wrap
 * @returns Wrapped tensor
 */
export const wrapTensor = (t: tf.Tensor) => t;

/**
 * Unwraps a tensor to get the underlying TensorFlow.js tensor.
 * @param tensor Tensor to unwrap
 * @returns Underlying TensorFlow.js tensor
 */
export const unwrapTensor = (t: ITensor) => t;

/**
 * Interface defining the core tensor operations available in the system.
 * Provides a subset of TensorFlow.js operations needed for the Titans implementation.
 */
export interface ITensorOps {
  tensor(data: number[], shape?: number[]): ITensor;
  tensor1d(data: number[]): ITensor;
  scalar(value: number): ITensor;
  zeros(shape: number[]): ITensor;
  randomNormal(shape: number[]): ITensor;
  variable(tensor: ITensor): ITensor;
  tidy<T extends tf.TensorContainer>(fn: () => T): T;
  train: {
    adam: (learningRate: number) => {
      minimize: (lossFn: () => tf.Scalar) => ITensor;
    };
  };
  concat(tensors: ITensor[], axis?: number): ITensor;
  matMul(a: ITensor, b: ITensor): ITensor;
  sub(a: ITensor, b: ITensor): ITensor;
  add(a: ITensor, b: ITensor): ITensor;
  mul(a: ITensor, b: ITensor): ITensor;
  div(a: ITensor, b: ITensor): ITensor;
  relu(x: ITensor): ITensor;
  sigmoid(x: ITensor): ITensor;
  tanh(x: ITensor): ITensor;
  mean(x: ITensor, axis?: number): ITensor;
  sum(x: ITensor, axis?: number): ITensor;
  sqrt(x: ITensor): ITensor;
  exp(x: ITensor): ITensor;
  log(x: ITensor): ITensor;
  dispose(): void;
  memory(): { numTensors: number; numDataBuffers: number; numBytes: number };
}

/**
 * Interface for tokenizer implementations.
 * Defines the contract for text encoding/decoding with tensor embeddings.
 */
export interface ITokenizer {
  /**
   * Encodes text into a tensor representation
   * @param text Input text to encode
   * @returns Tensor embedding of the text
   */
  encode(text: string): ITensor;

  /**
   * Decodes a tensor back into text
   * @param tensor Tensor to decode
   * @returns Decoded text string
   */
  decode(tensor: ITensor): string;

  /**
   * Returns special token IDs used by the tokenizer
   * @returns Map of special token names to their IDs
   */
  getSpecialTokens(): Record<string, number>;
}

// Memory Configuration Schema
export const TitanMemoryConfigSchema = z.object({
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

  // Research Paper Extensions
  enableMomentum: z.boolean().default(true).describe("Enable momentum-based memory updates"),
  momentumDecayRate: z.number().min(0).max(1).default(0.9).describe("η_t: Momentum decay parameter"),
  momentumLearningRate: z.number().positive().default(0.001).describe("θ_t: Momentum learning rate"),
  momentumScoreGain: z.number().min(0).max(10).default(0.5).describe("Attention-weighted boost applied to θ_t"),
  momentumScoreToDecay: z.number().min(0).max(1).default(0.2).describe("Attention-weighted boost applied to η_t"),
  momentumSurpriseGain: z.number().min(0).max(10).default(0.25).describe("Scales θ_t based on surprise magnitude"),
  momentumScoreFloor: z.number().min(0).max(1).default(1e-3).describe("Lower bound for attention weights in momentum updates"),
  enableForgettingGate: z.boolean().default(false).describe("Enable learnable forgetting gate"),
  forgettingGateInit: z.number().min(0).max(1).default(0.1).describe("α_t initial value"),
  enableTokenFlow: z.boolean().default(true).describe("Enable token flow tracking"),
  tokenFlowWindow: z.number().int().positive().default(10).describe("Token flow history window size"),
  enableHierarchicalMemory: z.boolean().default(false).describe("Enable hierarchical memory tiers"),
});

export type TitanMemoryConfig = z.infer<typeof TitanMemoryConfigSchema>;

export const HopeMemoryConfigSchema = z.object({
  inputDim: z.number().int().positive().default(256),
  hiddenDim: z.number().int().positive().default(192),
  memoryDim: z.number().int().positive().default(256),
  shortTermSlots: z.number().int().positive().default(64),
  longTermSlots: z.number().int().positive().default(256),
  archiveSlots: z.number().int().positive().default(512),
  learningRate: z.number().positive().default(0.001),
  dropoutRate: z.number().min(0).max(1).default(0.1),
  promotionThreshold: z.number().min(0).max(1).default(0.05),
  surpriseRetention: z.number().min(0).max(1).default(0.85),
  routerTopK: z.number().int().positive().max(3).default(2),

  // Compatibility and feature flags (for backward compatibility with TITAN-era code)
  maxSequenceLength: z.number().int().positive().default(512),
  memorySlots: z.number().int().positive().default(256).describe("Total memory capacity (computed from tiers)"),
  transformerLayers: z.number().int().positive().default(6).describe("Legacy config (not used in HOPE)"),
  enableMomentum: z.boolean().default(true).describe("Enable momentum-based memory updates"),
  enableTokenFlow: z.boolean().default(true).describe("Enable token flow tracking"),
  enableForgettingGate: z.boolean().default(false).describe("Enable learnable forgetting gate"),
  baseForgettingRate: z.number().min(0).max(1).default(0.1).describe("Base forgetting rate"),
  surpriseForgettingWeight: z.number().min(0).max(1).default(0.3).describe("Surprise influence on forgetting"),
  consolidationInterval: z.number().int().positive().default(100).describe("Interval between consolidation sweeps"),
  enableHierarchicalMemory: z.boolean().default(true).describe("Enable hierarchical memory tiers (always true for HOPE)"),
  useHierarchicalMemory: z.boolean().default(true).describe("Alias for enableHierarchicalMemory")
});

export type HopeMemoryConfig = z.infer<typeof HopeMemoryConfigSchema>;

/**
 * Interface for memory state in the Titans architecture.
 * Extended to support research paper concepts: momentum-based updates and token flow tracking
 */
export interface IMemoryState {
  shortTerm: ITensor;
  longTerm: ITensor;
  meta: ITensor;
  timestamps: ITensor;
  accessCounts: ITensor;
  surpriseHistory: ITensor;

  // Research Paper Extensions (Equations 32-33)
  momentumState?: ITensor;      // S_t: Momentum term for memory updates
  momentumDecay?: number;       // η_t: Momentum decay parameter
  forgettingGate?: ITensor | number;     // α_t: Learnable forgetting gate parameter (scalar)

  // Token Flow Tracking (Section 3.1)
  tokenFlowHistory?: ITensor;   // Sequential token dependency tracking
  flowWeights?: ITensor;        // Weights for token flow contribution
}

/**
 * Interface for attention block in transformer architecture.
 */
export interface IAttentionBlock {
  keys: ITensor;
  values: ITensor;
  scores: ITensor;
}

/**
 * Interface for surprise metrics in memory updates.
 */
export interface ISurpriseMetrics {
  immediate: ITensor;
  accumulated: ITensor;
  totalSurprise: ITensor;
}

/**
 * Interface for memory update results.
 */
export interface IMemoryUpdateResult {
  newState: IMemoryState;
  attention: IAttentionBlock;
  surprise: ISurpriseMetrics;
}

/**
 * Interface for model gradients.
 */
export interface IModelGradients {
  shortTerm: ITensor;
  longTerm: ITensor;
  meta: ITensor;
}

/**
 * Interface for memory manager operations.
 */
export interface IMemoryManager {
  validateVectorShape(tensor: tf.Tensor, expectedShape: number[]): boolean;
  encryptTensor(tensor: tf.Tensor): Buffer;
  decryptTensor(encrypted: Buffer, shape: number[]): tf.Tensor;
  wrapWithMemoryManagement<T extends tf.TensorContainer>(fn: () => T): T;
  wrapWithMemoryManagementAsync<T>(fn: () => Promise<T>): Promise<T>;
  dispose(): void;
}

/**
 * Interface for vector processing operations.
 */
export interface IVectorProcessor {
  processInput(input: number | number[] | string | tf.Tensor): tf.Tensor;
  validateAndNormalize(tensor: tf.Tensor, expectedShape: number[]): tf.Tensor;
  encodeText(text: string, maxLength?: number): Promise<tf.Tensor>;
}

/**
 * Interface for automatic memory maintenance operations.
 */
export interface IMemoryMaintenance {
  dispose(): void;
}

/**
 * Interface for the memory model.
 */
export interface IMemoryModel {
  forward(x: ITensor, memoryState: IMemoryState): {
    predicted: ITensor;
    memoryUpdate: IMemoryUpdateResult;
  };

  trainStep(x_t: ITensor, x_next: ITensor, memoryState: IMemoryState): {
    loss: ITensor;
    gradients: IModelGradients;
    memoryUpdate: IMemoryUpdateResult;
  };

  getTrainableVariables(): tf.Variable[];
  applyGradients?(gradients: Map<string, tf.Tensor>): void;

  updateMetaMemory(surprise: ISurpriseMetrics, context: ITensor): ITensor;
  pruneMemory(memoryState: IMemoryState, threshold: number): IMemoryState;
  manifoldStep(base: ITensor, velocity: ITensor): ITensor;
  saveModel(path: string): Promise<void>;
  loadModel(path: string): Promise<void>;
  getConfig(): any;
  save(modelPath: string, weightsPath: string): Promise<void>;
  getMemorySnapshot(): Record<string, tf.Tensor>;
  dispose(): void;
  resetGradients(): void;
  initialize(config?: any): Promise<void>;

  getMemoryState(): IMemoryState;
  resetMemory(): void;
  getMemoryState(): any;
  encodeText(text: string): Promise<tf.Tensor2D>;
  recallMemory(query: string, topK?: number): Promise<tf.Tensor2D[]>;
  storeMemory(text: string): Promise<void>;
  distillMemories?(similarMemories: tf.Tensor2D[]): tf.Tensor2D;
  quantizeMemory?(): IQuantizedMemoryState;
  dequantizeMemory?(quantizedState: IQuantizedMemoryState): IMemoryState;
  contrastiveLoss?(anchor: tf.Tensor2D, positive: tf.Tensor2D, negative: tf.Tensor2D, margin?: number): tf.Scalar;
  trainWithContrastiveLearning?(anchorText: string, positiveText: string, negativeText: string): Promise<number>;
  pruneMemoryByInformationGain?(threshold?: number): void;
  storeMemoryWithType?(text: string, isEpisodic?: boolean): Promise<void>;
  recallMemoryByType?(query: string, type?: 'episodic' | 'semantic' | 'both', topK?: number): Promise<tf.Tensor2D[]>;
  recallAndDistill?(query: string, topK?: number): Promise<tf.Tensor2D>;
}

/**
 * Interface for memory promotion rules between different memory tiers.
 */
export interface IMemoryPromotionRules {
  workingToShortTerm: {
    accessThreshold: number;
    timeThreshold: number;
    importanceThreshold: number;
  };
  shortTermToLongTerm: {
    accessThreshold: number;
    timeThreshold: number;
    importanceThreshold: number;
    reinforcementCount: number;
  };
  episodicToSemantic: {
    generalityThreshold: number;
    confidenceThreshold: number;
    abstractionLevel: number;
  };
  demotionRules: {
    lowAccessPenalty: number;
    ageDecayRate: number;
    forgettingThreshold: number;
  };
}

/**
 * Interface for retrieval weights in different memory types.
 */
export interface IRetrievalWeights {
  episodic: {
    recencyWeight: number;
    contextWeight: number;
    emotionalWeight: number;
  };
  semantic: {
    similarityWeight: number;
    confidenceWeight: number;
    generalityWeight: number;
  };
  combined: {
    episodicBias: number;
    semanticBias: number;
    tierPreference: number[];
  };
}

/**
 * Interface for server capabilities.
 */
export interface ServerCapabilities {
  name: string;
  version: string;
  description?: string;
  transport: string;
  tools: Record<string, {
    description: string;
    parameters: Record<string, unknown>;
  }>;
}

/**
 * Interface for tool call requests.
 */
export interface CallToolRequest {
  name: string;
  parameters: Record<string, unknown>;
}

/**
 * Interface for tool call results.
 */
export interface CallToolResult {
  content: Array<{
    type: string;
    text: string;
  }>;
}

/**
 * Interface for transport layer.
 */
export interface Transport {
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  onRequest(handler: (request: CallToolRequest) => Promise<CallToolResult>): void;
  send?(message: unknown): void;
}

/**
 * Interface for MCP server.
 */
export interface McpServer {
  tool(name: string, schema: z.ZodRawShape | string, handler: Function): void;
  connect(transport: Transport): Promise<void>;
}

// Memory Operation Schemas
export const StoreMemoryInput = z.object({
  subject: z.string(),
  relationship: z.string(),
  object: z.string()
});

export const RecallMemoryInput = z.object({
  query: z.string()
});

export interface IHierarchicalMemoryState extends IMemoryState {
  workingMemory: tf.Tensor2D;
  shortTermMemory: tf.Tensor2D;
  longTermMemory: tf.Tensor2D;
  workingAccessCounts: tf.Tensor1D;
  shortTermAccessCounts: tf.Tensor1D;
  longTermAccessCounts: tf.Tensor1D;
}

export interface IExtendedMemoryState extends IMemoryState {
  // Hierarchical memory tiers
  workingMemory: tf.Tensor2D;      // Immediate, high-capacity buffer
  shortTermMemory: tf.Tensor2D;    // Temporary storage for recent items
  longTermMemory: tf.Tensor2D;     // Persistent storage for important items

  // Episodic vs Semantic distinction
  episodicMemory: tf.Tensor2D;     // Time-bound, context-specific memories
  semanticMemory: tf.Tensor2D;     // Abstract, generalized knowledge

  // Temporal information
  workingTimestamps: tf.Tensor1D;   // When items entered working memory
  shortTermTimestamps: tf.Tensor1D; // When items entered short-term memory
  longTermTimestamps: tf.Tensor1D;  // When items entered long-term memory
  episodicTimestamps: tf.Tensor1D;  // When episodic memories were formed
  semanticTimestamps: tf.Tensor1D;  // When semantic knowledge was consolidated

  // Access patterns and confidence
  workingAccessCounts: tf.Tensor1D;
  shortTermAccessCounts: tf.Tensor1D;
  longTermAccessCounts: tf.Tensor1D;
  episodicAccessCounts: tf.Tensor1D;
  semanticAccessCounts: tf.Tensor1D;

  // Memory quality metrics
  episodicRecency: tf.Tensor1D;     // Recency scores for episodic memories
  semanticConfidence: tf.Tensor1D;  // Confidence scores for semantic knowledge
  memoryImportance: tf.Tensor1D;    // Importance scores for promotion/demotion
  surpriseScores: tf.Tensor1D;      // Surprise scores for memory consolidation

  // Memory type flags (0 = working, 1 = short-term, 2 = long-term, 3 = episodic, 4 = semantic)
  memoryTiers: tf.Tensor1D;
  memoryTypes: tf.Tensor1D;
}

/**
 * Memory statistics for monitoring and debugging
 */
export interface IMemoryStats {
  // Tier counts
  workingCount: number;
  shortTermCount: number;
  longTermCount: number;

  // Type counts
  episodicCount: number;
  semanticCount: number;

  // Memory utilization
  totalMemoryUsed: number;
  memoryUtilization: number; // percentage

  // Quality metrics
  averageImportance: number;
  averageConfidence: number;
  averageRecency: number;

  // Promotion/Demotion activity
  recentPromotions: number;
  recentDemotions: number;

  // Temporal distribution
  oldestMemoryAge: number;
  newestMemoryAge: number;
  averageMemoryAge: number;
}

/**
 * Promotion and demotion rules for memory tier management
 */
export interface IMemoryPromotionRules {
  // Working memory → Short-term memory
  workingToShortTerm: {
    accessThreshold: number;     // minimum access count
    timeThreshold: number;       // minimum time in working memory (ms)
    importanceThreshold: number; // minimum importance score
  };

  // Short-term memory → Long-term memory
  shortTermToLongTerm: {
    accessThreshold: number;
    timeThreshold: number;
    importanceThreshold: number;
    reinforcementCount: number;  // number of reinforcements needed
  };

  // Episodic → Semantic consolidation
  episodicToSemantic: {
    generalityThreshold: number; // how general/abstract the memory is
    confidenceThreshold: number; // confidence in the knowledge
    abstractionLevel: number;    // level of abstraction achieved
  };

  // Demotion thresholds
  demotionRules: {
    lowAccessPenalty: number;    // penalty for infrequent access
    ageDecayRate: number;        // how much importance decays over time
    forgettingThreshold: number; // when to demote/forget memories
  };
}

/**
 * Retrieval weighting strategies for different memory types
 */
export interface IRetrievalWeights {
  episodic: {
    recencyWeight: number;      // how much to weight recent memories
    contextWeight: number;      // how much to weight contextual similarity
    emotionalWeight: number;    // how much to weight emotional significance
  };

  semantic: {
    similarityWeight: number;   // how much to weight conceptual similarity
    confidenceWeight: number;   // how much to weight confidence scores
    generalityWeight: number;   // how much to weight general applicability
  };

  // Combined retrieval weights
  combined: {
    episodicBias: number;       // bias toward episodic memories
    semanticBias: number;       // bias toward semantic knowledge
    tierPreference: number[];   // preference weights for each memory tier
  };
}

export interface IQuantizedMemoryState {
  shortTermQuantized: Int8Array;
  longTermQuantized: Int8Array;
  metaQuantized: Int8Array;
  shortTermShape: number[];
  longTermShape: number[];
  metaShape: number[];
  quantizer: any;
  timestamps: number[];
  accessCounts: number[];
  surpriseHistory: number[];
}

export interface ITelemetryData {
  timestamp: number;
  operation: string;
  durationMs: number;
  memoryUsage: {
    numTensors: number;
    numBytes: number;
    unreliable: boolean;
  };
  metrics?: Record<string, number>;
  error?: {
    name: string;
    message: string;
    stack?: string;
  };
}

// Add custom error classes
export class TensorError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TensorError';
  }
}

export class MemoryError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'MemoryError';
  }
}

export interface IMemoryModel {
  // Existing methods
  forward(x: ITensor, memoryState: IMemoryState): { predicted: ITensor; memoryUpdate: IMemoryUpdateResult };
  trainStep(x_t: ITensor, x_next: ITensor, memoryState: IMemoryState): {
    loss: ITensor;
    gradients: IModelGradients;
    memoryUpdate: IMemoryUpdateResult;
  };
  pruneMemory(memoryState: IMemoryState, threshold: number): IMemoryState;
  manifoldStep(base: ITensor, velocity: ITensor): ITensor;
  getMemorySnapshot(): Record<string, tf.Tensor>;
  dispose(): void;

  // New methods
  getTrainableVariables(): tf.Variable[];
  applyGradients?(gradients: Map<string, tf.Tensor>): void;
  getMemoryState(): IMemoryState;
  resetMemory(): void;
  resetGradients(): void;

  // MCP server compatibility methods
  init_model(config: any): Promise<{ status: string }>;
  forward_pass(x: string | number[], memoryState?: IMemoryState): Promise<any>;
  train_step(x_t: string | number[], x_next: string | number[]): Promise<{ loss: number }>;
  get_memory_state(): any;

  // Enhanced functionality
  encodeText(text: string): Promise<tf.Tensor2D>;
  recallMemory(query: string, topK?: number): Promise<tf.Tensor2D[]>;
  storeMemory(text: string): Promise<void>;

  // Optional enhanced methods
  distillMemories?(similarMemories: tf.Tensor2D[]): tf.Tensor2D;
  quantizeMemory?(): IQuantizedMemoryState;
  dequantizeMemory?(quantizedState: IQuantizedMemoryState): IMemoryState;
  contrastiveLoss?(anchor: tf.Tensor2D, positive: tf.Tensor2D, negative: tf.Tensor2D, margin?: number): tf.Scalar;
  trainWithContrastiveLearning?(anchorText: string, positiveText: string, negativeText: string): Promise<number>;
  pruneMemoryByInformationGain?(threshold?: number): void;
  storeMemoryWithType?(text: string, isEpisodic?: boolean): Promise<void>;
  recallMemoryByType?(query: string, type?: 'episodic' | 'semantic' | 'both', topK?: number): Promise<tf.Tensor2D[]>;
  recallAndDistill?(query: string, topK?: number): Promise<tf.Tensor2D>;

  // Workflow-specific methods
  storeWorkflowMemory?(type: string, data: any): Promise<void>;
  getRelevantContext?(query: string): Promise<any>;
  findSimilarContent?(content: string): Promise<Array<{ score: number; content: string }>>;
  getWorkflowHistory?(type: string, limit: number): Promise<any[]>;
  shutdown?(): Promise<void>;
  getHealthStatus?(): Promise<any>;
}

// --- Internal/Specific State Types ---

/**
 * Internal representation for hierarchical memory state, using arrays of tensors.
 */
export interface IHierarchicalMemoryStateInternal {
  levels: tf.Tensor[];
  timestamps: tf.Tensor[];
  accessCounts: tf.Tensor[];
  surpriseScores: tf.Tensor[];
}

export interface SerializedTensor {
  data: number[];
  shape: number[];
}

export interface SerializedHierarchicalMemoryState {
  levels: SerializedTensor[];
  timestamps: SerializedTensor[];
  accessCounts: SerializedTensor[];
  surpriseScores: SerializedTensor[];
}

export interface SerializedExtendedMemoryState {
  tensors: Record<string, SerializedTensor>;
}

export interface SerializedAuxiliaryMemoryState {
  hierarchicalMemory?: SerializedHierarchicalMemoryState;
  extendedMemory?: SerializedExtendedMemoryState;
}

/**
 * Internal representation for quantized memory state, using Uint8Arrays.
 */
export interface IQuantizedMemoryStateInternal {
  shortTerm: Uint8Array;
  longTerm: Uint8Array;
  meta: Uint8Array;
  quantizationRanges: Array<{ min: number; max: number }>;
}

// --- Utility Types ---

/**
 * Maps string representations of data types to TensorFlow.js DataType enum strings.
 */
export interface DataTypeMap {
  float32: 'float32';
  int32: 'int32';
  bool: 'bool';
  string: 'string';
  complex64: 'complex64';
  uint8?: 'uint8'; // Optional, used for boolean storage sometimes
}

// ====== WORKFLOW TYPES ======

export interface WorkflowConfig {
  repository: {
    owner: string;
    name: string;
    branch: string;
  };
  features: {
    autoRelease: ReleaseConfig;
    issueManagement: IssueConfig;
    feedback: FeedbackConfig;
    labeling: LabelConfig;
    linting: LintConfig;
  };
  integrations: {
    github: GitHubIntegration;
    notifications: NotificationSystem;
    analytics?: AnalyticsConfig;
  };
  memory: {
    titanConfig: TitanMemoryConfig;
    hopeConfig?: HopeMemoryConfig;
    persistence?: PersistenceConfig;
  };
}

export interface ReleaseConfig {
  versionBump: "patch" | "minor" | "major";
  triggerConditions: {
    commitCount: number;
    timeThreshold: string;
    featureFlags: string[];
  };
  channels: {
    stable: string;
    beta: string;
    alpha: string;
  };
}

export interface IssueConfig {
  autoLabel: boolean;
  autoAssign: boolean;
  duplicateDetection: boolean;
  templates: string[];
}

export interface FeedbackConfig {
  channels: {
    github: {
      issues: boolean;
      discussions: boolean;
      pullRequests: boolean;
    };
    external: {
      slack: boolean;
      discord: boolean;
      email: boolean;
      surveys: boolean;
    };
    analytics: {
      errorTracking: boolean;
      usageMetrics: boolean;
      performanceMetrics: boolean;
    };
  };
}

export interface LabelConfig {
  autoLabeling: boolean;
  taxonomy: LabelTaxonomy;
  rules: LabelingRules;
}

export interface LintConfig {
  levels: {
    syntax: {
      enabled: boolean;
      tools: string[];
      failOnError: boolean;
    };
    style: {
      enabled: boolean;
      config: string;
      autoFix: boolean;
    };
    security: {
      enabled: boolean;
      tools: string[];
      severity: "error" | "warning";
    };
    performance: {
      enabled: boolean;
      thresholds: Record<string, number>;
    };
  };
  integrations: {
    preCommit: boolean;
    prChecks: boolean;
    cicd: boolean;
  };
}

export interface GitHubIntegration {
  authentication: {
    token: string;
    app?: GitHubApp;
  };
  permissions: {
    repositories: string[];
    scopes: string[];
  };
  webhooks: {
    events: string[];
    secret: string;
    url: string;
  };
}

export interface GitHubApp {
  id: string;
  privateKey: string;
  installationId?: string;
}

export interface NotificationSystem {
  channels: {
    slack?: SlackConfig;
    email?: EmailConfig;
    webhook?: WebhookConfig;
  };
  templates: {
    success: string;
    failure: string;
    warning: string;
  };
  routing: {
    rules: RoutingRule[];
    fallback: string;
  };
}

export interface SlackConfig {
  webhook: string;
  channel: string;
  botToken?: string;
}

export interface EmailConfig {
  smtp: {
    host: string;
    port: number;
    secure: boolean;
    auth: {
      user: string;
      pass: string;
    };
  };
  from: string;
  to: string[];
}

export interface WebhookConfig {
  url: string;
  secret?: string;
  headers?: Record<string, string>;
}

export interface RoutingRule {
  condition: string;
  target: string;
}

export interface AnalyticsConfig {
  provider: string;
  apiKey?: string;
  endpoint?: string;
}

export interface PersistenceConfig {
  type: 'file' | 'database' | 'redis';
  connection?: string;
  options?: Record<string, any>;
}

export interface LabelTaxonomy {
  type: Record<string, string>;
  priority: Record<string, string>;
  component: Record<string, string>;
  status: Record<string, string>;
}

export interface LabelingRules {
  textPatterns: Array<{
    pattern: RegExp;
    labels: string[];
    confidence: number;
  }>;
  filePatterns: Array<{
    pattern: string;
    labels: string[];
  }>;
  userRoles: Array<{
    role: string;
    defaultLabels: string[];
  }>;
  contextual: Array<{
    condition: string;
    labels: string[];
  }>;
}

export interface IssueClassification {
  type: "bug" | "feature" | "enhancement" | "question" | "documentation";
  priority: "critical" | "high" | "medium" | "low";
  complexity: "trivial" | "simple" | "moderate" | "complex";
  component: string[];
  estimatedHours?: number;
  dependencies?: string[];
}

export interface ReleasePR {
  title: string;
  body: string;
  labels: string[];
  assignees: string[];
  reviewers: string[];
  milestone?: string;
  metadata: {
    changeType: "breaking" | "feature" | "fix" | "docs";
    affectedComponents: string[];
    testCoverage: number;
    performanceImpact?: string;
  };
}

export interface FeedbackItem {
  id: string;
  source: string;
  timestamp: Date;
  content: string;
  sentiment: "positive" | "negative" | "neutral";
  topics: string[];
  priority: number;
  actionItems: string[];
  metadata: Record<string, any>;
}

export interface WorkflowStatus {
  state: 'initializing' | 'ready' | 'running' | 'error' | 'stopped';
  lastUpdate: Date;
  activeWorkflows: Array<{
    id: string;
    name: string;
    startTime: Date;
    status: string;
  }>;
  health: 'healthy' | 'unhealthy' | 'unknown';
}

export interface WorkflowEvent {
  id: string;
  name: string;
  type: string;
  params?: Record<string, any>;
  result?: any;
  error?: Error;
  executionTime?: number;
}

export interface WorkflowMetrics {
  totalWorkflows: number;
  successfulWorkflows: number;
  failedWorkflows: number;
  averageExecutionTime: number;
  memoryUsage: number;
  lastMetricsUpdate: Date;
}

/**
 * Alias for TitanMemoryModel for workflow compatibility
 */
export type HopeMemorySystem = IMemoryModel;
export type TitanMemorySystem = IMemoryModel;

// ====== END WORKFLOW TYPES ======
