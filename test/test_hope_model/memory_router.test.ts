import * as tf from '@tensorflow/tfjs-node';
import { MemoryRouter } from '../../src/hope_model/memory_router.js';
import { ContinuumMemory } from '../../src/hope_model/continuum_memory.js';

describe('MemoryRouter', () => {
  const memory = new ContinuumMemory({
    memoryDim: 4,
    shortTermSlots: 2,
    longTermSlots: 2,
    archiveSlots: 2,
    promotionThreshold: 0.1,
    surpriseRetention: 0.9
  });

  it('produces weights that sum to <= 1', () => {
    const router = new MemoryRouter({ hiddenDim: 4, numExperts: 3 });
    const state = memory.initialize();
    const hidden = tf.tensor2d([[0.2, 0.1, 0.3, 0.4]]);
    const decision = router.route(hidden, state as any);
    const sum = tf.sum(decision.weights).arraySync() as number;
    expect(sum).toBeGreaterThan(0);
    expect(sum).toBeLessThanOrEqual(1.0001);
  });

  it('selects at most topK experts', () => {
    const router = new MemoryRouter({ hiddenDim: 4, numExperts: 5, topK: 2 });
    const state = memory.initialize();
    const hidden = tf.tensor2d([[0.5, 0.2, 0.1, 0.3]]);
    const decision = router.route(hidden, state as any);
    expect(decision.selectedExperts.length).toBeLessThanOrEqual(2);
  });
});
