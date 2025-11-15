import * as tf from '@tensorflow/tfjs-node';
import { HopeMemoryModel } from '../model.js';

describe('HopeMemoryModel Tests', () => {
  let model: HopeMemoryModel;

  beforeEach(async () => {
    model = new HopeMemoryModel();
    await model.initialize();
  });

  afterEach(() => {
    model.dispose();
  });

  it('supports encoding text into tensors', async () => {
    const tensor = await model.encodeText('Testing HOPE memory');
    expect(tensor.shape[1]).toBeGreaterThan(0);
  });

  it('applies trainStep without throwing', () => {
    const input = tf.tensor2d([[0.1, 0.2, 0.3, 0.4]]);
    const target = tf.tensor2d([[0.4, 0.3, 0.2, 0.1]]);
    const state = model.createInitialState();
    const result = model.trainStep(input, target, state);
    expect(result.loss.arraySync()).toBeDefined();
  });
});
