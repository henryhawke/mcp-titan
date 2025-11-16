import * as tf from '@tensorflow/tfjs-node';
import { tidyMemoryState } from './type_utils.js';

export interface FilterState {
  carry: tf.Tensor2D;
  bandwidth: tf.Tensor2D;
}

export interface SelectiveFilterConfig {
  hiddenDim: number;
  contextDim: number;
  dropoutRate?: number;
}

export interface FilterStepResult {
  state: FilterState;
  output: tf.Tensor2D;
  retentionGate: tf.Tensor2D;
}

const DEFAULT_DROPOUT = 0.0;

/**
 * Implements a light-weight, Mamba-inspired selective state space module.
 * The filter keeps a content-dependent carry state that allows the model to
 * dynamically retain or forget information during long sequences without the
 * quadratic costs of attention.
 */
export class SelectiveStateSpace {
  private readonly config: Required<SelectiveFilterConfig>;
  private readonly inputKernel: tf.Variable<tf.Rank.R2>;
  private readonly carryKernel: tf.Variable<tf.Rank.R2>;
  private readonly bandwidthKernel: tf.Variable<tf.Rank.R2>;
  private readonly bias: tf.Variable<tf.Rank.R1>;
  private readonly bandwidthBias: tf.Variable<tf.Rank.R1>;

  constructor(config: SelectiveFilterConfig) {
    this.config = {
      ...config,
      dropoutRate: config.dropoutRate ?? DEFAULT_DROPOUT
    } as Required<SelectiveFilterConfig>;

    const { contextDim, hiddenDim } = this.config;
    this.inputKernel = tf.variable(tf.randomNormal([contextDim, hiddenDim], 0, Math.sqrt(2 / (contextDim + hiddenDim))));
    this.carryKernel = tf.variable(tf.randomNormal([hiddenDim, hiddenDim], 0, Math.sqrt(2 / (2 * hiddenDim))));
    this.bandwidthKernel = tf.variable(tf.randomNormal([contextDim + hiddenDim, hiddenDim]));
    this.bias = tf.variable(tf.zeros([hiddenDim]));
    this.bandwidthBias = tf.variable(tf.zeros([hiddenDim]));
  }

  public initState(batchSize: number): FilterState {
    return tidyMemoryState(() => ({
      carry: tf.zeros([batchSize, this.config.hiddenDim]) as tf.Tensor2D,
      bandwidth: tf.zeros([batchSize, this.config.hiddenDim]) as tf.Tensor2D
    }));
  }

  public step(input: tf.Tensor2D, prevState: FilterState): FilterStepResult {
    return tidyMemoryState(() => {
      const projectedInput = tf.matMul(input, this.inputKernel);
      const carryContribution = tf.matMul(prevState.carry, this.carryKernel);
      const candidate = tf.tanh(tf.add(projectedInput, tf.add(carryContribution, this.bias))) as tf.Tensor2D;

      const gateInput = tf.concat([input, prevState.carry], 1);
      const rawBandwidth = tf.add(tf.matMul(gateInput, this.bandwidthKernel), this.bandwidthBias);
      const retentionGate = tf.sigmoid(rawBandwidth) as tf.Tensor2D;

      const forgetGate = tf.sub(tf.onesLike(retentionGate), retentionGate);
      const newCarry = tf.add(tf.mul(retentionGate, prevState.carry), tf.mul(forgetGate, candidate)) as tf.Tensor2D;

      let output = newCarry;
      if (this.config.dropoutRate > 0) {
        output = tf.dropout(output, this.config.dropoutRate) as tf.Tensor2D;
      }

      return {
        state: {
          carry: newCarry,
          bandwidth: retentionGate
        },
        output,
        retentionGate
      };
    });
  }

  public processSequence(inputs: tf.Tensor2D, initialState: FilterState): {
    outputs: tf.Tensor2D;
    state: FilterState;
    gates: tf.Tensor2D;
  } {
    return tidyMemoryState(() => {
      let state = { ...initialState };
      const outputs: tf.Tensor2D[] = [];
      const gates: tf.Tensor2D[] = [];

      const timeSteps = inputs.shape[0];
      for (let i = 0; i < timeSteps; i += 1) {
        const stepInput = inputs.slice([i, 0], [1, inputs.shape[1]]) as tf.Tensor2D;
        const { state: newState, output, retentionGate } = this.step(stepInput, state);
        outputs.push(output);
        gates.push(retentionGate);
        state = newState;
      }

      return {
        outputs: tf.concat(outputs, 0) as tf.Tensor2D,
        state,
        gates: tf.concat(gates, 0) as tf.Tensor2D
      };
    });
  }

  public getTrainableVariables(): tf.Variable[] {
    return [
      this.inputKernel,
      this.carryKernel,
      this.bandwidthKernel,
      this.bias,
      this.bandwidthBias
    ];
  }
}
