import * as tf from '@tensorflow/tfjs-node';
import type { FilterState, SelectiveStateSpace } from './mamba_filters.js';
import { tidyMemoryState } from './type_utils.js';

export interface RetentionState {
  hidden: tf.Tensor2D;
  filter: FilterState;
  steps: number;
}

export interface RetentiveCoreConfig {
  inputDim: number;
  hiddenDim: number;
  dropoutRate?: number;
  chunkSize?: number;
}

export interface SequenceResult {
  outputs: tf.Tensor2D;
  state: RetentionState;
  gates: tf.Tensor2D;
}

const DEFAULT_CHUNK = 64;
const DEFAULT_DROPOUT = 0.0;

/**
 * RetentiveCore implements a lightweight recurrent block that works hand-in-hand
 * with the selective state space filter. It behaves similarly to a gated RNN but
 * keeps the operations parallel-friendly via chunked processing.
 */
export class RetentiveCore {
  private readonly config: Required<RetentiveCoreConfig>;
  private readonly inputKernel: tf.Variable<tf.Rank.R2>;
  private readonly hiddenKernel: tf.Variable<tf.Rank.R2>;
  private readonly bias: tf.Variable<tf.Rank.R1>;
  private readonly gateKernel: tf.Variable<tf.Rank.R2>;
  private readonly gateBias: tf.Variable<tf.Rank.R1>;
  private readonly outputKernel: tf.Variable<tf.Rank.R2>;
  private readonly outputBias: tf.Variable<tf.Rank.R1>;
  private readonly selectiveFilter: SelectiveStateSpace;

  constructor(config: RetentiveCoreConfig, selectiveFilter: SelectiveStateSpace) {
    this.config = {
      ...config,
      dropoutRate: config.dropoutRate ?? DEFAULT_DROPOUT,
      chunkSize: config.chunkSize ?? DEFAULT_CHUNK
    } as Required<RetentiveCoreConfig>;

    const { inputDim, hiddenDim } = this.config;
    this.inputKernel = tf.variable(tf.randomNormal([inputDim, hiddenDim], 0, Math.sqrt(2 / (inputDim + hiddenDim))));
    this.hiddenKernel = tf.variable(tf.randomNormal([hiddenDim, hiddenDim], 0, Math.sqrt(2 / (2 * hiddenDim))));
    this.bias = tf.variable(tf.zeros([hiddenDim]));
    this.gateKernel = tf.variable(tf.randomNormal([inputDim + hiddenDim, hiddenDim], 0, Math.sqrt(2 / (inputDim + hiddenDim))));
    this.gateBias = tf.variable(tf.zeros([hiddenDim]));
    this.outputKernel = tf.variable(tf.randomNormal([hiddenDim, hiddenDim]));
    this.outputBias = tf.variable(tf.zeros([hiddenDim]));
    this.selectiveFilter = selectiveFilter;
  }

  public initState(batchSize: number): RetentionState {
    return tidyMemoryState(() => ({
      hidden: tf.zeros([batchSize, this.config.hiddenDim]) as tf.Tensor2D,
      filter: this.selectiveFilter.initState(batchSize),
      steps: 0
    }));
  }

  public forwardStep(input: tf.Tensor2D, prevState: RetentionState): SequenceResult {
    return tidyMemoryState(() => {
      const concatenated = tf.concat([input, prevState.hidden], 1);
      const retentionGate = tf.sigmoid(tf.add(tf.matMul(concatenated, this.gateKernel), this.gateBias)) as tf.Tensor2D;

      const projected = tf.add(
        tf.add(tf.matMul(input, this.inputKernel), tf.matMul(prevState.hidden, this.hiddenKernel)),
        this.bias
      );
      const candidate = tf.tanh(projected);

      let hidden = tf.add(tf.mul(retentionGate, prevState.hidden), tf.mul(tf.sub(tf.onesLike(retentionGate), retentionGate), candidate)) as tf.Tensor2D;
      if (this.config.dropoutRate > 0) {
        hidden = tf.dropout(hidden, this.config.dropoutRate) as tf.Tensor2D;
      }

      const filterResult = this.selectiveFilter.step(hidden, prevState.filter);
      const filteredHidden = filterResult.output;
      const output = tf.add(tf.matMul(filteredHidden, this.outputKernel), this.outputBias) as tf.Tensor2D;

      return {
        outputs: output,
        state: {
          hidden: filteredHidden,
          filter: filterResult.state,
          steps: prevState.steps + 1
        },
        gates: filterResult.retentionGate
      };
    });
  }

  public forwardSequence(inputs: tf.Tensor2D, prevState?: RetentionState): SequenceResult {
    return tidyMemoryState(() => {
      const batchSize = inputs.shape[1] ? 1 : inputs.shape[0];
      let state = prevState ?? this.initState(batchSize);
      const outputs: tf.Tensor2D[] = [];
      const gates: tf.Tensor2D[] = [];

      const timeSteps = inputs.shape[0];
      for (let i = 0; i < timeSteps; i += 1) {
        const stepInput = inputs.slice([i, 0], [1, inputs.shape[1]]) as tf.Tensor2D;
        const { outputs: stepOutput, state: newState, gates: stepGate } = this.forwardStep(stepInput, state);
        outputs.push(stepOutput);
        gates.push(stepGate);
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
      this.hiddenKernel,
      this.bias,
      this.gateKernel,
      this.gateBias,
      this.outputKernel,
      this.outputBias,
      ...this.selectiveFilter.getTrainableVariables()
    ];
  }
}
