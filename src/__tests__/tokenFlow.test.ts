import * as tf from '@tensorflow/tfjs-node';
import { TitanMemoryModel } from '../model.js';
import { wrapTensor, unwrapTensor, type IMemoryState } from '../types.js';
import seedrandom from 'seedrandom';

const rng = seedrandom('token-flow-tests');
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

describe('Token Flow Integration', () => {
  let model: TitanMemoryModel;

  beforeAll(async () => {
    model = new TitanMemoryModel();
    await model.initialize({
      inputDim: 16,
      memoryDim: 16,
      memorySlots: 8,
      enableMomentum: false,
      enableTokenFlow: true,
      tokenFlowWindow: 4
    });
  });

  afterAll(() => {
    model.dispose();
  });

  test('initializes token flow buffers when enabled', () => {
    const state = model.getMemoryState();
    expect(state.tokenFlowHistory).toBeDefined();
    expect(state.flowWeights).toBeDefined();

    const historyShape = unwrapTensor(state.tokenFlowHistory!).shape;
    const weightsShape = unwrapTensor(state.flowWeights!).shape;
    expect(historyShape[0]).toBe(4);
    expect(weightsShape[0]).toBe(4);

    disposeState(state);
  });

  test('forward pass updates token flow history and weights', () => {
    const state = model.getMemoryState();
    const inputValues = tf.tensor1d(randomArray(model.getConfig().inputDim));
    const input = wrapTensor(inputValues);

    const result = model.forward(input, state);

    expect(result.memoryUpdate.newState.tokenFlowHistory).toBeDefined();
    expect(result.memoryUpdate.newState.flowWeights).toBeDefined();

    const history = unwrapTensor(result.memoryUpdate.newState.tokenFlowHistory!) as tf.Tensor2D;
    const weights = unwrapTensor(result.memoryUpdate.newState.flowWeights!) as tf.Tensor1D;

    const lastRow = history.slice([history.shape[0] - 1, 0], [1, history.shape[1]]);
    const lastRowNorm = lastRow.norm().dataSync()[0];
    const weightsSum = tf.sum(weights).dataSync()[0];

    expect(lastRowNorm).toBeGreaterThan(0);
    expect(Math.abs(weightsSum - 1)).toBeLessThan(1e-5);

    const surpriseScalar = unwrapTensor(result.memoryUpdate.surprise.totalSurprise);
    expect(surpriseScalar.size).toBe(1);

    const immediateTensor = unwrapTensor(result.memoryUpdate.surprise.immediate);
    const baselineImmediate = immediateTensor.norm().dataSync()[0];
    const blendedValue = surpriseScalar.dataSync()[0];
    expect(Math.abs(blendedValue - baselineImmediate)).toBeGreaterThan(1e-4);

    surpriseScalar.dispose();
    immediateTensor.dispose();

    lastRow.dispose();
    unwrapTensor(result.predicted).dispose();
    unwrapTensor(input).dispose();
    inputValues.dispose();
    disposeState(state);
  });
});
