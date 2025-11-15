import * as tf from '@tensorflow/tfjs-node';

export interface CompressedGradientPayload {
  deltas: Float32Array[];
  shapes: number[][];
}

export class DeltaCompressionHook {
  private previous: tf.Tensor[] = [];

  public compress(gradients: tf.Tensor[]): CompressedGradientPayload {
    const deltas: Float32Array[] = [];
    const shapes: number[][] = [];

    gradients.forEach((gradient, index) => {
      const previous = this.previous[index];
      const previousValues = previous ? previous.dataSync() : new Float32Array(gradient.size).fill(0);
      const currentValues = gradient.dataSync();
      const delta = new Float32Array(currentValues.length);
      for (let i = 0; i < currentValues.length; i += 1) {
        delta[i] = currentValues[i] - previousValues[i];
      }
      deltas.push(delta);
      shapes.push(gradient.shape);
    });

    this.previous = gradients.map(tensor => tensor.clone());
    return { deltas, shapes };
  }

  public decompress(payload: CompressedGradientPayload): tf.Tensor[] {
    return payload.deltas.map((delta, index) =>
      tf.tensor(delta, payload.shapes[index])
    );
  }

  public reset(): void {
    this.previous.forEach(tensor => tensor.dispose());
    this.previous = [];
  }
}

export interface LayerScheduleConfig {
  maxActiveLayers: number;
}

export class LayerScheduler {
  private readonly config: LayerScheduleConfig;

  constructor(config: LayerScheduleConfig) {
    this.config = config;
  }

  public selectActiveLayers(gradients: tf.Tensor[]): number[] {
    const scored = gradients.map((tensor, index) => ({
      index,
      score: Math.abs(tensor.sum().arraySync() as number)
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, this.config.maxActiveLayers).map(item => item.index);
  }
}

export class UpdateBuffer {
  private buffer = new Map<string, tf.Tensor>();

  public push(layerName: string, update: tf.Tensor): void {
    const existing = this.buffer.get(layerName);
    if (existing) {
      const combined = tf.add(existing, update);
      existing.dispose();
      update.dispose();
      this.buffer.set(layerName, combined);
    } else {
      this.buffer.set(layerName, update.clone());
    }
  }

  public flush(): Map<string, tf.Tensor> {
    const snapshot = new Map<string, tf.Tensor>(this.buffer);
    this.buffer.forEach(tensor => tensor.dispose());
    this.buffer.clear();
    return snapshot;
  }

  public clear(): void {
    this.buffer.forEach(tensor => tensor.dispose());
    this.buffer.clear();
  }
}
