import * as tf from '@tensorflow/tfjs-node';
import { DeltaCompressionHook, LayerScheduler, UpdateBuffer } from '../../src/hope_model/optimizer_hooks.js';

describe('Optimizer hooks', () => {
  it('compresses and decompresses gradients accurately', () => {
    const hook = new DeltaCompressionHook();
    const grads = [tf.tensor1d([1, 2, 3]), tf.tensor1d([4, 5, 6])];
    const payload = hook.compress(grads);
    const restored = hook.decompress(payload);

    restored.forEach((tensor, index) => {
      expect(tensor.shape).toEqual(grads[index].shape);
      const diff = tf.sub(tensor, grads[index]).abs().max().arraySync() as number;
      expect(diff).toBeLessThan(1e-6);
    });
  });

  it('selects layers with largest magnitude', () => {
    const scheduler = new LayerScheduler({ maxActiveLayers: 1 });
    const grads = [tf.tensor1d([0.1, 0.1]), tf.tensor1d([5, 5])];
    const selected = scheduler.selectActiveLayers(grads);
    expect(selected).toEqual([1]);
  });

  it('aggregates updates before flushing', () => {
    const buffer = new UpdateBuffer();
    buffer.push('layer1', tf.tensor1d([1, 1]));
    buffer.push('layer1', tf.tensor1d([2, 3]));
    const flushed = buffer.flush();
    expect(flushed.size).toBe(1);
    const combined = flushed.get('layer1');
    expect(combined).toBeDefined();
    if (combined) {
      const values = combined.arraySync() as number[];
      expect(values).toEqual([3, 4]);
      combined.dispose();
    }
  });
});
