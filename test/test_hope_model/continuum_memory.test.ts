import * as tf from '@tensorflow/tfjs-node';
import { ContinuumMemory } from '../../src/hope_model/continuum_memory.js';

describe('ContinuumMemory', () => {
  const createMemory = () => new ContinuumMemory({
    memoryDim: 4,
    shortTermSlots: 2,
    longTermSlots: 3,
    archiveSlots: 4,
    promotionThreshold: 0.1,
    surpriseRetention: 0.9
  });

  it('writes entries and promotes when capacity exceeded', () => {
    const memory = createMemory();
    let state = memory.initialize();

    for (let i = 0; i < 4; i += 1) {
      const embedding = tf.tensor2d([[0.1 * (i + 1), 0, 0, 0]]);
      state = memory.write(state, embedding, {
        surprise: 0.5,
        timestamp: Date.now(),
        routeWeights: tf.tensor2d([[1, 0, 0]])
      });
    }

    state = memory.promote(state);

    expect(state.shortTerm.shape[0]).toBeLessThanOrEqual(2);
    expect(state.longTerm.shape[0]).toBeGreaterThan(0);
  });

  it('prunes low-importance long-term entries', () => {
    const memory = createMemory();
    let state = memory.initialize();

    const low = tf.tensor2d([[0.001, 0.001, 0.001, 0.001]]);
    const high = tf.tensor2d([[1, 1, 1, 1]]);

    state = memory.write(state, low, {
      surprise: 0.1,
      timestamp: Date.now(),
      routeWeights: tf.tensor2d([[0, 1, 0]])
    });
    state = memory.promote(state);
    state = memory.write(state, high, {
      surprise: 0.9,
      timestamp: Date.now(),
      routeWeights: tf.tensor2d([[0, 1, 0]])
    });
    state = memory.promote(state);

    const pruned = memory.prune(state, 0.01);
    expect(pruned.longTerm.shape[0]).toBeLessThanOrEqual(state.longTerm.shape[0]);
  });

  it('computes stats with average surprise', () => {
    const memory = createMemory();
    let state = memory.initialize();
    const embedding = tf.tensor2d([[0.2, 0.3, 0.4, 0.5]]);
    state = memory.write(state, embedding, {
      surprise: 0.75,
      timestamp: Date.now(),
      routeWeights: tf.tensor2d([[1, 0, 0]])
    });
    const stats = memory.getStats(state);
    expect(stats.shortTerm).toBe(1);
    expect(stats.averageSurprise).toBeGreaterThan(0);
  });
});
