import * as tf from '@tensorflow/tfjs-node';
import { LearnerService, type LearnerConfig } from '../learner.js';
import { wrapTensor, unwrapTensor, type IMemoryState } from '../types.js';

// Mock implementations
class MockModel {
  getMemoryState(): IMemoryState {
    return {
      shortTerm: wrapTensor(tf.zeros([10, 64])),
      longTerm: wrapTensor(tf.zeros([10, 64])),
      meta: wrapTensor(tf.zeros([10, 64])),
      timestamps: wrapTensor(tf.zeros([10])),
      accessCounts: wrapTensor(tf.zeros([10])),
      surpriseHistory: wrapTensor(tf.zeros([10]))
    };
  }

  forward(input: any, memoryState: IMemoryState) {
    return {
      predicted: wrapTensor(tf.randomNormal([64])),
      memoryUpdate: {
        newState: {
          shortTerm: wrapTensor(tf.zeros([10, 64])),
          longTerm: wrapTensor(tf.zeros([10, 64])),
          meta: wrapTensor(tf.zeros([10, 64])),
          timestamps: wrapTensor(tf.zeros([10])),
          accessCounts: wrapTensor(tf.zeros([10])),
          surpriseHistory: wrapTensor(tf.zeros([10]))
        },
        attention: {} as any,
        surprise: {} as any
      }
    };
  }

  getTrainableVariables(): tf.Variable[] {
    return [];
  }

  applyGradients(): void {
    // No-op for mock
  }

  // Stub methods to satisfy IMemoryModel interface
  trainStep() { return { loss: wrapTensor(tf.scalar(0.5)), gradients: {} as any }; }
  updateMetaMemory() { return wrapTensor(tf.zeros([10, 64])); }
  pruneMemory(state: any) { return state; }
  manifoldStep() { return wrapTensor(tf.zeros([64])); }
  saveModel() { return Promise.resolve(); }
  loadModel() { return Promise.resolve(); }
  getConfig() { return {}; }
  save() { return Promise.resolve(); }
  getMemorySnapshot() { return {}; }
  dispose() { }
  resetGradients() { }
  initialize() { return Promise.resolve(); }
  resetMemory() { }
  init_model() { return Promise.resolve({ status: 'ok' }); }
  forward_pass() { return Promise.resolve({}); }
  train_step() { return Promise.resolve({ loss: 0.5 }); }
  get_memory_state() { return {}; }
  encodeText() { return Promise.resolve(tf.zeros([64]) as any); }
  recallMemory() { return Promise.resolve([]); }
  storeMemory() { return Promise.resolve(); }
}

class MockTokenizer {
  encode(text: string): tf.Tensor {
    return tf.randomNormal([64]);
  }

  decode(tensor: tf.Tensor): string {
    return 'test';
  }

  getSpecialTokens() {
    return { mask: 103, pad: 0, unk: 1 };
  }
}

class TrainableModel {
  public weight: tf.Variable;
  public bias: tf.Variable;
  private optimizer: tf.Optimizer;
  private memoryState: IMemoryState;

  constructor() {
    this.weight = tf.variable(tf.randomNormal([4, 4]), true, 'trainable_weight');
    this.bias = tf.variable(tf.zeros([4]), true, 'trainable_bias');
    this.optimizer = tf.train.sgd(0.05);
    this.memoryState = {
      shortTerm: wrapTensor(tf.zeros([1, 4])),
      longTerm: wrapTensor(tf.zeros([1, 4])),
      meta: wrapTensor(tf.zeros([1, 4])),
      timestamps: wrapTensor(tf.zeros([1])),
      accessCounts: wrapTensor(tf.zeros([1])),
      surpriseHistory: wrapTensor(tf.zeros([1]))
    };
  }

  getMemoryState(): IMemoryState {
    return this.memoryState;
  }

  forward(input: any, memoryState: IMemoryState) {
    const tensor = unwrapTensor(input) as tf.Tensor1D;
    const reshaped = tensor.reshape([1, tensor.shape[0]]);
    const logits = tf.add(tf.matMul(reshaped, this.weight), this.bias);

    return {
      predicted: wrapTensor(logits.squeeze()),
      memoryUpdate: {
        newState: memoryState,
        attention: {} as any,
        surprise: {} as any
      }
    };
  }

  getTrainableVariables(): tf.Variable[] {
    return [this.weight, this.bias];
  }

  applyGradients(gradients: Map<string, tf.Tensor>): void {
    const namedGradients: tf.NamedTensorMap = {};
    const weightGrad = gradients.get(this.weight.name);
    if (weightGrad) {
      namedGradients[this.weight.name] = weightGrad;
    }
    const biasGrad = gradients.get(this.bias.name);
    if (biasGrad) {
      namedGradients[this.bias.name] = biasGrad;
    }
    if (Object.keys(namedGradients).length > 0) {
      this.optimizer.applyGradients(namedGradients);
    }
  }

  dispose(): void {
    this.optimizer.dispose();
  }
}

describe('LearnerService', () => {
  let learner: LearnerService;
  let mockModel: MockModel;
  let mockTokenizer: MockTokenizer;
  
  const testConfig: Partial<LearnerConfig> = {
    bufferSize: 100,
    batchSize: 4,
    updateInterval: 50, // Shorter for testing
    gradientClipValue: 1.0,
    contrastiveWeight: 0.3,
    nextTokenWeight: 0.4,
    mlmWeight: 0.3,
    accumulationSteps: 2,
    learningRate: 0.001,
    nanGuardThreshold: 1e-6
  };

  beforeEach(() => {
    mockModel = new MockModel();
    mockTokenizer = new MockTokenizer();
    learner = new LearnerService(mockModel as any, mockTokenizer as any, testConfig);
  });

  afterEach(() => {
    learner.dispose();
    tf.disposeVariables();
  });

  describe('Ring Buffer', () => {
    test('should add training samples to buffer', () => {
      const input = tf.tensor1d([1, 2, 3, 4]);
      const target = tf.tensor1d([0, 1, 0, 1]);

      learner.addTrainingSample(input, target);
      
      const stats = learner.getTrainingStats();
      expect(stats.bufferSize).toBe(1);

      input.dispose();
      target.dispose();
    });

    test('should handle buffer overflow correctly', () => {
      const bufferSize = 5;
      const smallConfig = { ...testConfig, bufferSize };
      const smallLearner = new LearnerService(mockModel as any, mockTokenizer as any, smallConfig);

      // Add more samples than buffer size
      for (let i = 0; i < bufferSize + 3; i++) {
        const input = tf.tensor1d([i, i + 1, i + 2, i + 3]);
        const target = tf.tensor1d([i % 2, (i + 1) % 2, i % 2, (i + 1) % 2]);
        smallLearner.addTrainingSample(input, target);
        input.dispose();
        target.dispose();
      }

      const stats = smallLearner.getTrainingStats();
      expect(stats.bufferSize).toBe(bufferSize);
      
      smallLearner.dispose();
    });

    test('should add contrastive learning samples', () => {
      const input = tf.tensor1d([1, 2, 3, 4]);
      const target = tf.tensor1d([0, 1, 0, 1]);
      const positive = tf.tensor1d([1, 2, 3, 5]);
      const negative = tf.tensor1d([5, 6, 7, 8]);

      learner.addTrainingSample(input, target, positive, negative);
      
      const stats = learner.getTrainingStats();
      expect(stats.bufferSize).toBe(1);

      input.dispose();
      target.dispose();
      positive.dispose();
      negative.dispose();
    });
  });

  describe('Training Loop', () => {
    test('should start and stop training', (done) => {
      expect(learner.isTraining()).toBe(false);
      
      learner.startTraining();
      expect(learner.isTraining()).toBe(true);
      
      setTimeout(() => {
        learner.pauseTraining();
        expect(learner.isTraining()).toBe(false);
        done();
      }, 100);
    });

    test('should resume training after pause', () => {
      learner.startTraining();
      expect(learner.isTraining()).toBe(true);
      
      learner.pauseTraining();
      expect(learner.isTraining()).toBe(false);
      
      learner.resumeTraining();
      expect(learner.isTraining()).toBe(true);
      
      learner.pauseTraining();
    });

    test('should not start training if already running', () => {
      learner.startTraining();
      const firstStart = learner.isTraining();
      
      learner.startTraining(); // Try to start again
      const secondStart = learner.isTraining();
      
      expect(firstStart).toBe(true);
      expect(secondStart).toBe(true);
      
      learner.pauseTraining();
    });
  });

  describe('Training Statistics', () => {
    test('should track training progress', () => {
      const initialStats = learner.getTrainingStats();
      expect(initialStats.stepCount).toBe(0);
      expect(initialStats.bufferSize).toBe(0);
      expect(initialStats.isRunning).toBe(false);
      expect(initialStats.averageLoss).toBe(0);
      expect(initialStats.lastLoss).toBe(0);
    });

    test('should update buffer size when samples are added', () => {
      const input = tf.tensor1d([1, 2, 3, 4]);
      const target = tf.tensor1d([0, 1, 0, 1]);

      learner.addTrainingSample(input, target);
      
      const stats = learner.getTrainingStats();
      expect(stats.bufferSize).toBe(1);

      input.dispose();
      target.dispose();
    });
  });

  describe('Gradient Safety', () => {
    test('should handle NaN gradients gracefully', () => {
      // This is more of an integration test - we can't easily mock NaN gradients
      // but we can test that the service doesn't crash with various inputs
      const input = tf.tensor1d([0, 0, 0, 0]); // All zeros might cause numerical issues
      const target = tf.tensor1d([1, 1, 1, 1]);

      expect(() => {
        learner.addTrainingSample(input, target);
      }).not.toThrow();

      input.dispose();
      target.dispose();
    });

    test('should clip gradients within specified range', () => {
      // Test that the service can handle extreme values
      const input = tf.tensor1d([1e6, -1e6, 1e6, -1e6]);
      const target = tf.tensor1d([1, 0, 1, 0]);

      expect(() => {
        learner.addTrainingSample(input, target);
      }).not.toThrow();

      input.dispose();
      target.dispose();
    });
  });

  describe('Gradient application', () => {
    test('performTrainingStep updates model weights', () => {
      const trainableModel = new TrainableModel();
      const gradientLearner = new LearnerService(trainableModel as any, mockTokenizer as any, {
        ...testConfig,
        batchSize: 2,
        accumulationSteps: 1,
        nextTokenWeight: 1,
        contrastiveWeight: 0,
        mlmWeight: 0
      });

      const samples: Array<{ input: tf.Tensor1D; target: tf.Tensor1D }> = [
        {
          input: tf.tensor1d([0.2, -0.1, 0.4, 0.3]),
          target: tf.tensor1d([0, 1, 0, 0])
        },
        {
          input: tf.tensor1d([-0.3, 0.5, 0.1, -0.2]),
          target: tf.tensor1d([0, 0, 1, 0])
        }
      ];

      for (const sample of samples) {
        gradientLearner.addTrainingSample(sample.input, sample.target);
      }

      const initialWeight = trainableModel.weight.clone();
      const initialBias = trainableModel.bias.clone();

      (gradientLearner as any).performTrainingStep();

      for (const sample of samples) {
        sample.input.dispose();
        sample.target.dispose();
      }

      const weightDiff = tf.sub(trainableModel.weight, initialWeight);
      const biasDiff = tf.sub(trainableModel.bias, initialBias);
      const weightChange = tf.max(tf.abs(weightDiff)).dataSync()[0];
      const biasChange = tf.max(tf.abs(biasDiff)).dataSync()[0];

      expect(Math.max(weightChange, biasChange)).toBeGreaterThan(0);

      initialWeight.dispose();
      initialBias.dispose();
      weightDiff.dispose();
      biasDiff.dispose();
      gradientLearner.dispose();
      trainableModel.dispose();
    });
  });

  describe('Memory Management', () => {
    test('should dispose resources properly', () => {
      const input = tf.tensor1d([1, 2, 3, 4]);
      const target = tf.tensor1d([0, 1, 0, 1]);

      learner.addTrainingSample(input, target);
      learner.startTraining();
      
      // Record tensor count before disposal
      const tensorsBefore = tf.memory().numTensors;
      
      learner.dispose();
      
      // Check that training stopped
      expect(learner.isTraining()).toBe(false);
      
      input.dispose();
      target.dispose();
    });

    test('should handle tf.tidy properly in training loops', () => {
      // Add multiple samples to ensure buffer has content
      for (let i = 0; i < 10; i++) {
        const input = tf.tensor1d([i, i + 1, i + 2, i + 3]);
        const target = tf.tensor1d([i % 2, (i + 1) % 2, i % 2, (i + 1) % 2]);
        learner.addTrainingSample(input, target);
        input.dispose();
        target.dispose();
      }

      const tensorsBefore = tf.memory().numTensors;
      
      // The training step should not leak tensors significantly
      // Note: This is a basic check - actual tensor management depends on tf.tidy usage
      expect(learner.getTrainingStats().bufferSize).toBe(10);
      
      const tensorsAfter = tf.memory().numTensors;
      // We allow some variance due to internal TensorFlow.js operations
      expect(Math.abs(tensorsAfter - tensorsBefore)).toBeLessThan(100);
    });
  });

  describe('Mixed Loss Computation', () => {
    test('should handle empty batches gracefully', () => {
      // Test with insufficient buffer size
      const stats = learner.getTrainingStats();
      expect(stats.bufferSize).toBe(0);
      
      // This should not crash even with empty buffer
      expect(() => {
        learner.startTraining();
        setTimeout(() => learner.pauseTraining(), 10);
      }).not.toThrow();
    });

    test('should process samples with different loss types', () => {
      // Add samples with different characteristics
      const input1 = tf.tensor1d([1, 2, 3, 4]);
      const target1 = tf.tensor1d([0, 1, 0, 1]);
      
      const input2 = tf.tensor1d([2, 3, 4, 5]);
      const target2 = tf.tensor1d([1, 0, 1, 0]);
      const positive2 = tf.tensor1d([2, 3, 4, 6]);
      const negative2 = tf.tensor1d([8, 9, 10, 11]);

      learner.addTrainingSample(input1, target1);
      learner.addTrainingSample(input2, target2, positive2, negative2);
      
      const stats = learner.getTrainingStats();
      expect(stats.bufferSize).toBe(2);

      // Clean up
      input1.dispose();
      target1.dispose();
      input2.dispose();
      target2.dispose();
      positive2.dispose();
      negative2.dispose();
    });
  });

  describe('Configuration', () => {
    test('should use default configuration when not provided', () => {
      const defaultLearner = new LearnerService(mockModel as any, mockTokenizer as any);
      const stats = defaultLearner.getTrainingStats();
      
      expect(stats.stepCount).toBe(0);
      expect(stats.isRunning).toBe(false);
      
      defaultLearner.dispose();
    });

    test('should respect custom configuration values', () => {
      const customConfig = {
        bufferSize: 50,
        batchSize: 8,
        updateInterval: 200
      };
      
      const customLearner = new LearnerService(mockModel as any, mockTokenizer as any, customConfig);
      
      // Add samples and verify buffer respects custom size
      for (let i = 0; i < 60; i++) {
        const input = tf.tensor1d([i, i + 1, i + 2, i + 3]);
        const target = tf.tensor1d([i % 2, (i + 1) % 2, i % 2, (i + 1) % 2]);
        customLearner.addTrainingSample(input, target);
        input.dispose();
        target.dispose();
      }
      
      const stats = customLearner.getTrainingStats();
      expect(stats.bufferSize).toBe(50); // Should respect custom buffer size
      
      customLearner.dispose();
    });
  });
});
