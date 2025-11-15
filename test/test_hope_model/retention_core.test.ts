import * as tf from '@tensorflow/tfjs-node';
import { SelectiveStateSpace } from '../../src/hope_model/mamba_filters.js';
import { RetentiveCore } from '../../src/hope_model/retention_core.js';

describe('RetentiveCore', () => {
  afterEach(() => {
    tf.disposeVariables();
  });

  it('matches forwardStep results when processed sequentially', () => {
    const filter = new SelectiveStateSpace({ hiddenDim: 8, contextDim: 8 });
    const core = new RetentiveCore({ inputDim: 16, hiddenDim: 8 }, filter);

    const input = tf.tensor2d([
      Array(16).fill(0.1),
      Array(16).fill(0.2),
      Array(16).fill(0.3)
    ]);

    const initialState = core.initState(1);
    const sequenceResult = core.forwardSequence(input, initialState);

    let state = initialState;
    const stepOutputs: tf.Tensor[] = [];
    for (let i = 0; i < input.shape[0]; i += 1) {
      const slice = input.slice([i, 0], [1, input.shape[1]]);
      const { outputs, state: newState } = core.forwardStep(slice, state);
      stepOutputs.push(outputs);
      state = newState;
    }

    const stackedSteps = tf.concat(stepOutputs, 0);
    expect(stackedSteps.shape).toEqual(sequenceResult.outputs.shape);
    const difference = tf.abs(tf.sub(stackedSteps, sequenceResult.outputs)).max().arraySync() as number;
    expect(difference).toBeLessThan(1e-4);
  });

  it('returns gate activations for each timestep', () => {
    const filter = new SelectiveStateSpace({ hiddenDim: 4, contextDim: 4 });
    const core = new RetentiveCore({ inputDim: 8, hiddenDim: 4 }, filter);
    const input = tf.tensor2d([
      Array(8).fill(0.05),
      Array(8).fill(0.4)
    ]);
    const result = core.forwardSequence(input);
    expect(result.gates.shape).toEqual([2, 4]);
    const gateValues = result.gates.arraySync() as number[][];
    gateValues.flat().forEach(value => {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    });
  });
});
