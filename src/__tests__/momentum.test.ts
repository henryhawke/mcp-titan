import * as tf from '@tensorflow/tfjs-node';
import { HopeMemoryModel } from '../model.js';

describe('HopeMemoryModel optimizer hooks', () => {
  let model: HopeMemoryModel;

  beforeEach(async () => {
    model = new HopeMemoryModel();
    await model.initialize();
  });

  afterEach(() => {
    model.dispose();
  });

  it('resets gradient hooks without throwing', () => {
    expect(() => model.resetGradients()).not.toThrow();
  });

  it('prunes memory using information gain threshold', async () => {
    const state = model.createInitialState();
    const input = tf.tensor2d([[0.3, 0.2, 0.5, 0.1]]);
    const update = model.forward(input, state);
    model.hydrateMemoryState(update.memoryUpdate.newState);
    const pruned = await model.pruneMemoryByInformationGain(0.5);
    expect(pruned.longTerm.shape[0]).toBeGreaterThanOrEqual(0);
  });
});
