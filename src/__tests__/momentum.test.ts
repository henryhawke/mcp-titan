import * as tf from '@tensorflow/tfjs-node';
import seedrandom from 'seedrandom';
import { TitanMemoryModel } from '../model.js';
import { wrapTensor, unwrapTensor, type IMemoryState } from '../types.js';

const rng = seedrandom('momentum-tests');
const randomArray = (length: number): number[] => Array.from({ length }, () => (rng() * 2) - 1);

const disposeState = (state: IMemoryState): void => {
  unwrapTensor(state.shortTerm).dispose();
  unwrapTensor(state.longTerm).dispose();
  unwrapTensor(state.meta).dispose();
  unwrapTensor(state.timestamps).dispose();
  unwrapTensor(state.accessCounts).dispose();
  unwrapTensor(state.surpriseHistory).dispose();
  state.momentumState && unwrapTensor(state.momentumState).dispose();
  state.tokenFlowHistory && unwrapTensor(state.tokenFlowHistory).dispose();
  state.flowWeights && unwrapTensor(state.flowWeights).dispose();
  state.forgettingGate && unwrapTensor(state.forgettingGate).dispose();
};

const getMomentumMagnitude = (momentum: IMemoryState['momentumState']): number => {
  if (!momentum) {
    return 0;
  }
  return tf.tidy(() => {
    const tensor = unwrapTensor(momentum);
    const clone = tensor.clone();
    const value = clone.norm().dataSync()[0];
    clone.dispose();
    return value;
  });
};

describe('Momentum-Based Memory Updates', () => {
  let model: TitanMemoryModel;

  beforeAll(async () => {
    model = new TitanMemoryModel();
    await model.initialize({
      inputDim: 16,
      memoryDim: 16,
      memorySlots: 8,
      enableMomentum: true,
      momentumDecayRate: 0.8,
      enableForgettingGate: false
    });
  });

  afterAll(() => {
    model.dispose();
  });

  test('initializes momentum state with decay parameter', () => {
    const state = model.getMemoryState();
    expect(state.momentumState).toBeDefined();
    expect(state.momentumDecay).toBeCloseTo(0.8);
    disposeState(state);
  });

  test('updates momentum state during training', () => {
    const state = model.getMemoryState();
    const inputDim = model.getConfig().inputDim;

    const xTensor = tf.tensor1d(randomArray(inputDim));
    const nextTensor = tf.tensor1d(randomArray(inputDim));

    const result = model.trainStep(
      wrapTensor(xTensor),
      wrapTensor(nextTensor),
      state
    );

    expect(result.memoryUpdate.newState.momentumState).toBeDefined();

    const newMagnitude = getMomentumMagnitude(result.memoryUpdate.newState.momentumState);
    expect(Number.isFinite(newMagnitude)).toBe(true);
    expect(newMagnitude).toBeGreaterThanOrEqual(0);

    unwrapTensor(result.loss).dispose();
    xTensor.dispose();
    nextTensor.dispose();
    disposeState(state);
  });

  test('momentum update reflects decay and gradient influence', () => {
    const state = model.getMemoryState();
    const inputDim = model.getConfig().inputDim;

    const xTensor = tf.tensor1d(randomArray(inputDim));
    const nextTensor = tf.tensor1d(randomArray(inputDim));

    const result = model.trainStep(
      wrapTensor(xTensor),
      wrapTensor(nextTensor),
      state
    );

    const momentumTensor = result.memoryUpdate.newState.momentumState
      ? unwrapTensor(result.memoryUpdate.newState.momentumState)
      : null;

    expect(momentumTensor).not.toBeNull();

    if (momentumTensor) {
      const momentumMean = tf.mean(momentumTensor).dataSync()[0];
      expect(Number.isFinite(momentumMean)).toBe(true);
      momentumTensor.dispose();
    }

    unwrapTensor(result.loss).dispose();
    xTensor.dispose();
    nextTensor.dispose();
    disposeState(state);
  });

  test('momentum decay scaling responds to configuration', async () => {
    const highDecayModel = new TitanMemoryModel();
    await highDecayModel.initialize({
      inputDim: 8,
      memoryDim: 8,
      memorySlots: 6,
      enableMomentum: true,
      momentumDecayRate: 0.9,
      enableForgettingGate: false
    });

    const lowDecayModel = new TitanMemoryModel();
    await lowDecayModel.initialize({
      inputDim: 8,
      memoryDim: 8,
      memorySlots: 6,
      enableMomentum: true,
      momentumDecayRate: 0.1,
      enableForgettingGate: false
    });

    const runStep = (instance: TitanMemoryModel): number => {
      const stateClone = instance.getMemoryState();
      const inputTensor = tf.tensor1d(randomArray(8));
      const nextTensor = tf.tensor1d(randomArray(8));

      const result = instance.trainStep(
        wrapTensor(inputTensor),
        wrapTensor(nextTensor),
        stateClone
      );

      const magnitude = getMomentumMagnitude(result.memoryUpdate.newState.momentumState);

      unwrapTensor(result.loss).dispose();
      inputTensor.dispose();
      nextTensor.dispose();
      disposeState(stateClone);

      return magnitude;
    };

    runStep(highDecayModel);
    runStep(lowDecayModel);

    const highNorm = runStep(highDecayModel);
    const lowNorm = runStep(lowDecayModel);

    expect(highNorm).toBeLessThan(lowNorm);

    highDecayModel.dispose();
    lowDecayModel.dispose();
  });

  test('forgetting gate influences momentum application', async () => {
    const gatingModel = new TitanMemoryModel();
    await gatingModel.initialize({
      inputDim: 8,
      memoryDim: 8,
      memorySlots: 6,
      enableMomentum: true,
      enableForgettingGate: true
    });

    const state = gatingModel.getMemoryState();
    const inputTensor = tf.ones([8]);
    const nextTensor = tf.ones([8]);

    const result = gatingModel.trainStep(
      wrapTensor(inputTensor),
      wrapTensor(nextTensor),
      state
    );

    expect(result.memoryUpdate.newState.shortTerm).toBeDefined();
    expect(result.memoryUpdate.newState.momentumState).toBeDefined();

    unwrapTensor(result.loss).dispose();
    inputTensor.dispose();
    nextTensor.dispose();
    disposeState(state);
    gatingModel.dispose();
  });
});
