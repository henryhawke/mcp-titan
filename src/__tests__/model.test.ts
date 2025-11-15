import * as tf from '@tensorflow/tfjs-node';
import { HopeMemoryModel } from '../model.js';

describe('HopeMemoryModel', () => {
  let model: HopeMemoryModel;

  beforeEach(async () => {
    model = new HopeMemoryModel();
    await model.initialize();
  });

  afterEach(() => {
    model.dispose();
  });

  it('initializes a continuum memory state', () => {
    const state = model.createInitialState();
    expect(state.shortTerm.shape[0]).toBe(0);
    expect(state.longTerm.shape[0]).toBe(0);
  });

  it('produces predictions for a tensor input', () => {
    const input = tf.tensor2d([[0.1, 0.2, 0.3, 0.4]]);
    const state = model.createInitialState();
    const { predicted, memoryUpdate } = model.forward(input, state);
    expect(predicted.shape[0]).toBeGreaterThan(0);
    expect(memoryUpdate.newState.shortTerm.shape[0]).toBeGreaterThanOrEqual(0);
  });

  it('performs a training step and returns loss', () => {
    const input = tf.tensor2d([[0.1, 0.2, 0.3, 0.4]]);
    const target = tf.tensor2d([[0.2, 0.1, 0.4, 0.3]]);
    const state = model.createInitialState();
    const result = model.trainStep(input, target, state);
    expect(result.loss.arraySync()).toBeDefined();
    expect(result.memoryUpdate.newState).toBeDefined();
  });
});
