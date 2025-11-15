import * as tf from '@tensorflow/tfjs-node';
import { promises as fs } from 'fs';
import * as path from 'path';

import type {
  IMemoryModel,
  IMemoryState,
  IMemoryUpdateResult,
  IAttentionBlock,
  ISurpriseMetrics,
  IModelGradients,
  HopeMemoryConfig
} from '../types.js';
import { ContinuumMemory, type ContinuumMemoryConfig, type HopeMemoryState, type HierarchicalStats } from './continuum_memory.js';
import { RetentiveCore, type RetentiveCoreConfig, type RetentionState } from './retention_core.js';
import { SelectiveStateSpace } from './mamba_filters.js';
import { MemoryRouter, type RoutingDecision } from './memory_router.js';
import { DeltaCompressionHook, LayerScheduler, UpdateBuffer } from './optimizer_hooks.js';

const DEFAULT_CONFIG: HopeMemoryConfig = {
  inputDim: 256,
  hiddenDim: 192,
  memoryDim: 256,
  shortTermSlots: 64,
  longTermSlots: 256,
  archiveSlots: 512,
  learningRate: 1e-3,
  dropoutRate: 0.1,
  promotionThreshold: 0.05,
  surpriseRetention: 0.85,
  routerTopK: 2,
  // Backward compatibility fields
  maxSequenceLength: 512,
  memorySlots: 256,
  transformerLayers: 6,
  enableMomentum: true,
  enableTokenFlow: true,
  enableForgettingGate: false,
  enableHierarchicalMemory: true,
  useHierarchicalMemory: true
};

interface ForwardArtifacts {
  logits: tf.Tensor2D;
  memoryState: HopeMemoryState;
  retentionState: RetentionState;
  decision: RoutingDecision;
}

export class HopeMemoryModel implements IMemoryModel {
  private config: HopeMemoryConfig;
  private readonly continuumMemory: ContinuumMemory;
  private readonly selectiveFilter: SelectiveStateSpace;
  private readonly retentiveCore: RetentiveCore;
  private readonly memoryRouter: MemoryRouter;
  private readonly compressionHook: DeltaCompressionHook;
  private readonly layerScheduler: LayerScheduler;
  private readonly updateBuffer: UpdateBuffer;
  private readonly outputKernel: tf.Variable<tf.Rank.R2>;
  private readonly outputBias: tf.Variable<tf.Rank.R1>;
  private optimizer: tf.AdamOptimizer;
  private retentionState?: RetentionState;
  private latestMemoryState: HopeMemoryState;

  constructor(config: Partial<HopeMemoryConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    const memoryConfig: ContinuumMemoryConfig = {
      memoryDim: this.config.memoryDim,
      shortTermSlots: this.config.shortTermSlots,
      longTermSlots: this.config.longTermSlots,
      archiveSlots: this.config.archiveSlots,
      promotionThreshold: this.config.promotionThreshold,
      surpriseRetention: this.config.surpriseRetention
    };
    this.continuumMemory = new ContinuumMemory(memoryConfig);

    const filter = new SelectiveStateSpace({
      hiddenDim: this.config.hiddenDim,
      contextDim: this.config.hiddenDim,
      dropoutRate: this.config.dropoutRate
    });
    this.selectiveFilter = filter;
    const coreConfig: RetentiveCoreConfig = {
      inputDim: this.config.inputDim + this.config.memoryDim,
      hiddenDim: this.config.hiddenDim,
      dropoutRate: this.config.dropoutRate,
      chunkSize: 64
    };
    this.retentiveCore = new RetentiveCore(coreConfig, filter);
    this.memoryRouter = new MemoryRouter({
      hiddenDim: this.config.hiddenDim,
      numExperts: 3,
      topK: this.config.routerTopK
    });
    this.compressionHook = new DeltaCompressionHook();
    this.layerScheduler = new LayerScheduler({ maxActiveLayers: 4 });
    this.updateBuffer = new UpdateBuffer();

    this.outputKernel = tf.variable(tf.randomNormal([this.config.hiddenDim, this.config.inputDim]));
    this.outputBias = tf.variable(tf.zeros([this.config.inputDim]));
    this.optimizer = tf.train.adam(this.config.learningRate);
    this.retentionState = this.retentiveCore.initState(1);
    this.latestMemoryState = this.continuumMemory.initialize();
  }

  public async initialize(config?: Partial<HopeMemoryConfig>): Promise<void> {
    if (config) {
      this.config = { ...this.config, ...config };
    }
    this.optimizer = tf.train.adam(this.config.learningRate);
    this.retentionState = this.retentiveCore.initState(1);
    this.latestMemoryState = this.continuumMemory.initialize();
  }

  public createInitialState(): HopeMemoryState {
    this.retentionState = this.retentiveCore.initState(1);
    this.latestMemoryState = this.continuumMemory.initialize();
    return this.latestMemoryState;
  }

  public forward(x: tf.Tensor2D, memoryState: IMemoryState): {
    predicted: tf.Tensor2D;
    memoryUpdate: IMemoryUpdateResult;
  } {
    const hopeState = memoryState as HopeMemoryState;
    const result = this.computeForward(x, hopeState, true);
    this.latestMemoryState = result.memoryState;
    return this.buildForwardResult(result, hopeState);
  }

  public trainStep(x_t: tf.Tensor2D, x_next: tf.Tensor2D, memoryState: IMemoryState): {
    loss: tf.Tensor;
    gradients: IModelGradients;
    memoryUpdate: IMemoryUpdateResult;
  } {
    const hopeState = memoryState as HopeMemoryState;
    const clonedState = this.continuumMemory.clone(hopeState);

    const { value: loss, grads } = tf.variableGrads(() => {
      const forwardResult = this.computeForward(x_t, clonedState, false);
      const target = this.ensure2d(x_next);
      const prediction = forwardResult.logits;
      const mse = tf.losses.meanSquaredError(target, prediction);
      return mse.mean();
    });

    const gradientEntries = Object.entries(grads);
    const gradientTensors = gradientEntries.map(([, tensor]) => tensor);
    const payload = this.compressionHook.compress(gradientTensors);
    const decompressed = this.compressionHook.decompress(payload);
    const activeLayers = this.layerScheduler.selectActiveLayers(decompressed);
    const applyGradients: {[key: string]: tf.Tensor} = {};

    gradientEntries.forEach(([name, tensor], index) => {
      if (activeLayers.includes(index)) {
        applyGradients[name] = tensor;
        this.updateBuffer.push(name, tensor.clone());
      }
    });

    if (Object.keys(applyGradients).length === 0 && gradientEntries.length > 0) {
      const [fallbackName, fallbackTensor] = gradientEntries[0];
      applyGradients[fallbackName] = fallbackTensor;
    }

    this.optimizer.applyGradients(applyGradients);

    const forwardResult = this.computeForward(x_t, hopeState, true);
    this.latestMemoryState = forwardResult.memoryState;
    const memoryUpdate = this.buildForwardResult(forwardResult, hopeState).memoryUpdate;

    return {
      loss,
      gradients: {
        shortTerm: tf.zerosLike(hopeState.shortTerm),
        longTerm: tf.zerosLike(hopeState.longTerm),
        meta: tf.zerosLike(hopeState.meta)
      },
      memoryUpdate
    };
  }

  public getTrainableVariables(): tf.Variable[] {
    return [
      ...this.retentiveCore.getTrainableVariables(),
      ...this.memoryRouter.getTrainableVariables(),
      this.outputKernel,
      this.outputBias
    ];
  }

  public applyGradients?(gradients: Map<string, tf.Tensor>): void {
    const grads: {[key: string]: tf.Tensor} = {};
    gradients.forEach((tensor, key) => { grads[key] = tensor; });
    this.optimizer.applyGradients(grads);
  }

  public getConfig(): HopeMemoryConfig {
    return { ...this.config };
  }

  public resetGradients(): void {
    this.compressionHook.reset();
    this.updateBuffer.clear();
  }

  public hydrateMemoryState(state: IMemoryState): void {
    this.latestMemoryState = state as HopeMemoryState;
  }

  public async pruneMemoryByInformationGain(threshold: number): Promise<HopeMemoryState> {
    this.latestMemoryState = this.continuumMemory.prune(this.latestMemoryState, threshold);
    return this.latestMemoryState;
  }

  public getPruningStats(): HierarchicalStats {
    return this.continuumMemory.getStats(this.latestMemoryState);
  }

  public async encodeText(text: string): Promise<tf.Tensor2D> {
    const tokens = Array.from(text).map(char => char.codePointAt(0) ?? 0);
    const normalized = new Array(this.config.inputDim).fill(0);
    for (let i = 0; i < Math.min(tokens.length, this.config.inputDim); i += 1) {
      normalized[i] = (tokens[i] % 1024) / 1024;
    }
    return tf.tensor2d([normalized]);
  }

  public async storeMemory(text: string): Promise<void> {
    const embedding = await this.encodeText(text);
    const baseState = this.latestMemoryState;
    const decision = this.memoryRouter.route(this.retentionState?.hidden ?? embedding, baseState);
    this.latestMemoryState = this.continuumMemory.write(baseState, embedding, {
      surprise: decision.surprise,
      timestamp: Date.now(),
      routeWeights: decision.weights
    });
  }

  public exportAuxiliaryState(): Record<string, unknown> {
    const snapshot = this.updateBuffer.flush();
    const pending = Array.from(snapshot.entries()).map(([name, tensor]) => {
      const norm = tensor.norm().arraySync();
      tensor.dispose();
      return { name, norm };
    });
    return {
      config: this.config,
      pendingUpdates: pending
    };
  }

  public restoreAuxiliaryState(_: Record<string, unknown>): void {
    // No-op for now – HOPE recomputes auxiliary state during initialization.
  }

  // Alias for backward compatibility
  public async load(directory: string): Promise<void> {
    await this.loadModel(directory);
  }

  public async loadModel(directory: string): Promise<void> {
    const filePath = path.join(directory, 'hope_model.json');
    const exists = await fs.access(filePath).then(() => true).catch(() => false);
    if (!exists) { return; }
    const raw = await fs.readFile(filePath, 'utf8');
    const payload = JSON.parse(raw) as {
      config: HopeMemoryConfig;
      weights: number[][];
      shapes: number[][];
    };
    this.config = { ...this.config, ...payload.config };
    const variables = this.getTrainableVariables();
    payload.weights.forEach((values, index) => {
      const shape = payload.shapes[index];
      if (!variables[index]) { return; }
      const tensor = tf.tensor(values, shape);
      variables[index].assign(tensor as tf.Tensor);
    });
  }

  public async save(directory: string): Promise<void> {
    await fs.mkdir(directory, { recursive: true });
    const variables = this.getTrainableVariables();
    const weights = await Promise.all(variables.map(async variable => Array.from(await variable.data())));
    const shapes = variables.map(variable => variable.shape as number[]);
    const payload = {
      config: this.config,
      weights,
      shapes
    };
    await fs.writeFile(path.join(directory, 'hope_model.json'), JSON.stringify(payload));
  }

  // Alias for IMemoryModel compatibility
  public async saveModel(path: string): Promise<void> {
    await this.save(path);
  }

  // IMemoryModel required methods
  public getMemoryState(): HopeMemoryState {
    return this.latestMemoryState;
  }

  public resetMemory(): void {
    this.latestMemoryState = this.createInitialState();
    this.retentionState = undefined;
  }

  public updateMetaMemory(surprise: ISurpriseMetrics, context: tf.Tensor): tf.Tensor {
    // For HOPE, meta memory is managed within ContinuumMemory
    // Return the context unchanged as this is handled internally
    return context;
  }

  public pruneMemory(memoryState: IMemoryState, threshold: number): IMemoryState {
    // Convert IMemoryState to HopeMemoryState and prune
    const hopeState = memoryState as unknown as HopeMemoryState;
    const pruned = this.continuumMemory.prune(hopeState, threshold);
    return pruned as unknown as IMemoryState;
  }

  public manifoldStep(base: tf.Tensor, velocity: tf.Tensor): tf.Tensor {
    // Simple Euler step on the manifold (base + velocity)
    // In future, this could implement geodesic stepping
    return tf.add(base, velocity);
  }

  public getMemorySnapshot(): Record<string, tf.Tensor> {
    const state = this.latestMemoryState;
    return {
      shortTerm: state.shortTerm,
      longTerm: state.longTerm,
      archive: state.archive || tf.zeros([1, this.config.memoryDim]),
      surpriseHistory: state.surpriseHistory,
      accessCounts: state.accessCounts
    };
  }

  public restoreMemoryState(state: IMemoryState): void {
    this.latestMemoryState = state as unknown as HopeMemoryState;
  }

  public async recallMemory(query: string, topK: number = 5): Promise<tf.Tensor2D[]> {
    const queryTensor = await this.encodeText(query);
    const queryTensor2d = queryTensor.expandDims(0) as tf.Tensor2D;

    // Read from memory using the router
    const decision = this.memoryRouter.route(queryTensor2d, this.latestMemoryState);
    const memoryRead = this.continuumMemory.read(this.latestMemoryState, queryTensor2d, decision.weights);

    // Return the memory read as a single-element array (simplified recall)
    return [memoryRead as tf.Tensor2D];
  }

  public dispose(): void {
    this.getTrainableVariables().forEach(variable => variable.dispose());
    this.retentionState?.hidden.dispose();
    this.retentionState?.filter.carry.dispose();
    this.retentionState?.filter.bandwidth.dispose();
  }

  private computeForward(input: tf.Tensor2D, memoryState: HopeMemoryState, updateState: boolean): ForwardArtifacts {
    return tf.tidy(() => {
      const normalizedInput = this.ensure2d(input);
      const readWeights = this.memoryRouter.route(this.retentionState?.hidden ?? normalizedInput, memoryState);
      const memoryRead = this.continuumMemory.read(memoryState, normalizedInput, readWeights.weights);
      const coreInput = tf.concat([normalizedInput, memoryRead], 1);
      const retentionState = this.retentionState ?? this.retentiveCore.initState(1);
      const { outputs, state } = this.retentiveCore.forwardSequence(coreInput, retentionState);
      const logits = tf.add(tf.matMul(outputs, this.outputKernel), this.outputBias);

      let updatedState = memoryState;
      if (updateState) {
        updatedState = this.continuumMemory.write(memoryState, outputs.slice([outputs.shape[0] - 1, 0], [1, -1]), {
          surprise: readWeights.surprise,
          timestamp: Date.now(),
          routeWeights: readWeights.weights
        });
        this.retentionState = state;
        this.latestMemoryState = updatedState;
      }

      return {
        logits,
        memoryState: updatedState,
        retentionState: state,
        decision: readWeights
      };
    });
  }

  private buildForwardResult(artifacts: ForwardArtifacts, originalState: HopeMemoryState): {
    predicted: tf.Tensor2D;
    memoryUpdate: IMemoryUpdateResult;
  } {
    const attention: IAttentionBlock = {
      keys: tf.zeros([1, this.config.memoryDim]),
      values: tf.zeros([1, this.config.memoryDim]),
      scores: artifacts.decision.weights
    };

    const surprise: ISurpriseMetrics = {
      immediate: tf.tensor1d([artifacts.decision.surprise]),
      accumulated: tf.tensor1d([originalState.surpriseHistory.shape[0]]),
      totalSurprise: tf.tensor1d([artifacts.decision.surprise])
    };

    return {
      predicted: artifacts.logits,
      memoryUpdate: {
        newState: artifacts.memoryState,
        attention,
        surprise
      }
    };
  }

  private ensure2d(tensor: tf.Tensor2D): tf.Tensor2D {
    if (tensor.rank === 2) {
      return tensor;
    }
    return tensor.reshape([tensor.shape[0] ?? 1, this.config.inputDim]);
  }
}
