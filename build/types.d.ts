/**
 * @fileoverview Core type definitions for the Titans memory architecture.
 * This file defines the interfaces and types used throughout the implementation
 * of the Titans memory model, including tensor operations, memory states, and
 * model interactions.
 */
import * as tf from '@tensorflow/tfjs';
import { z } from "zod";
/**
 * Basic interface for an in-house tensor object that wraps TensorFlow.js tensors.
 * Provides essential operations while maintaining compatibility with tf.TensorContainerObject.
 */
export interface ITensor extends tf.TensorContainerObject {
    /** Returns the raw data */
    dataSync(): Float32Array | Int32Array | Uint8Array;
    /** Releases the memory associated with this tensor */
    dispose(): void;
    /** The shape of the tensor as an array of dimensions */
    shape: number[];
    /** Additional properties required for TensorContainerObject compatibility */
    [key: string]: any;
}
/**
 * Interface defining the core tensor operations available in the system.
 * Provides a subset of TensorFlow.js operations needed for the Titans implementation.
 */
export interface ITensorOps {
    /** Creates a new tensor with the specified data and optional shape */
    tensor(data: number[], shape?: number[]): ITensor;
    /** Creates a 1-dimensional tensor */
    tensor1d(data: number[]): ITensor;
    /** Creates a scalar tensor */
    scalar(value: number): ITensor;
    /** Creates a tensor filled with zeros */
    zeros(shape: number[]): ITensor;
    /** Creates a tensor with random normal values */
    randomNormal(shape: number[]): ITensor;
    /** Creates a trainable variable */
    variable(tensor: ITensor): ITensor;
    /** Executes a function while cleaning up intermediate tensors */
    tidy<T extends tf.TensorContainer>(fn: () => T): T;
    /** Training operations */
    train: {
        adam: (learningRate: number) => {
            minimize: (lossFn: () => tf.Scalar) => ITensor;
        };
    };
    /** Concatenates tensors along an axis */
    concat(tensors: ITensor[], axis?: number): ITensor;
    /** Matrix multiplication */
    matMul(a: ITensor, b: ITensor): ITensor;
    /** Element-wise subtraction */
    sub(a: ITensor, b: ITensor): ITensor;
    /** Element-wise addition */
    add(a: ITensor, b: ITensor): ITensor;
    /** Element-wise multiplication */
    mul(a: ITensor, b: ITensor): ITensor;
    /** Element-wise division */
    div(a: ITensor, b: ITensor): ITensor;
    /** ReLU activation function */
    relu(x: ITensor): ITensor;
    /** Sigmoid activation function */
    sigmoid(x: ITensor): ITensor;
    /** Tanh activation function */
    tanh(x: ITensor): ITensor;
    /** Computes mean along specified axis */
    mean(x: ITensor, axis?: number): ITensor;
    /** Computes sum along specified axis */
    sum(x: ITensor, axis?: number): ITensor;
    /** Element-wise square root */
    sqrt(x: ITensor): ITensor;
    /** Element-wise exponential */
    exp(x: ITensor): ITensor;
    /** Element-wise natural logarithm */
    log(x: ITensor): ITensor;
    /** Releases memory for all tensors */
    dispose(): void;
    /** Returns current memory usage statistics */
    memory(): {
        numTensors: number;
        numDataBuffers: number;
        numBytes: number;
    };
}
/**
 * Represents an attention block in the Titans architecture.
 * Contains the key-value pairs and attention scores used in the memory mechanism.
 */
export interface IAttentionBlock extends tf.TensorContainerObject {
    /** Keys used for attention computation */
    keys: ITensor;
    /** Values to be attended over */
    values: ITensor;
    /** Computed attention scores */
    scores: ITensor;
    [key: string]: any;
}
/**
 * Tracks both immediate and accumulated surprise metrics.
 * Used to guide memory updates and test-time learning.
 */
export interface ISurpriseMetrics extends tf.TensorContainerObject {
    /** Immediate surprise from current prediction */
    immediate: ITensor;
    /** Accumulated surprise over time */
    accumulated: ITensor;
    [key: string]: any;
}
/**
 * Represents the three-tier memory state in the Titans architecture.
 * Combines short-term, long-term, and meta-memory components.
 */
export interface IMemoryState extends tf.TensorContainerObject {
    /** Short-term memory for immediate context */
    shortTerm: ITensor;
    /** Long-term memory for persistent information */
    longTerm: ITensor;
    /** Meta-memory for learning to memorize */
    meta: ITensor;
    [key: string]: any;
}
/**
 * Result of a memory update operation, containing the new memory state
 * and associated attention and surprise metrics.
 */
export interface IMemoryUpdateResult extends tf.TensorContainerObject {
    /** Updated memory state after forward pass */
    newState: IMemoryState;
    /** Attention computations used in the update */
    attention: IAttentionBlock;
    /** Computed surprise metrics */
    surprise: ISurpriseMetrics;
    [key: string]: any;
}
/**
 * Gradients computed for each memory component during training.
 * Used to update the model's parameters.
 */
export interface IModelGradients extends tf.TensorContainerObject {
    /** Gradients for short-term memory */
    shortTerm: ITensor;
    /** Gradients for long-term memory */
    longTerm: ITensor;
    /** Gradients for meta-memory */
    meta: ITensor;
    [key: string]: any;
}
/**
 * Core interface for the Titans memory model.
 * Defines the essential methods for forward passes, training, and memory management.
 */
export interface IMemoryModel {
    /**
     * Performs a forward pass through the model.
     * @param x Input tensor
     * @param memoryState Current memory state
     * @returns Predicted output and memory updates
     */
    forward(x: ITensor, memoryState: IMemoryState): {
        predicted: ITensor;
        memoryUpdate: IMemoryUpdateResult;
    };
    /**
     * Performs a training step.
     * @param x_t Current input
     * @param x_next Next input (target)
     * @param memoryState Current memory state
     * @returns Loss and computed gradients
     */
    trainStep(x_t: ITensor, x_next: ITensor, memoryState: IMemoryState): {
        loss: ITensor;
        gradients: {
            shortTerm: ITensor;
            longTerm: ITensor;
            meta: ITensor;
        };
    };
    /**
     * Updates meta-memory based on surprise metrics.
     * @param surprise Current surprise metrics
     * @param context Context tensor
     * @returns Updated meta-memory
     */
    updateMetaMemory(surprise: ISurpriseMetrics, context: ITensor): ITensor;
    /**
     * Prunes memory based on importance scores.
     * @param memoryState Current memory state
     * @param threshold Pruning threshold
     * @returns Pruned memory state
     */
    pruneMemory(memoryState: IMemoryState, threshold: number): IMemoryState;
    /**
     * Performs a manifold optimization step.
     * @param base Base point on manifold
     * @param velocity Update direction
     * @returns Updated point on manifold
     */
    manifoldStep(base: ITensor, velocity: ITensor): ITensor;
    /**
     * Saves model weights to disk.
     * @param path File path
     */
    saveModel(path: string): Promise<void>;
    /**
     * Loads model weights from disk.
     * @param path File path
     */
    loadModel(path: string): Promise<void>;
    /**
     * Returns current model configuration.
     */
    getConfig(): any;
}
/**
 * Wrapper class for TensorFlow.js tensors.
 * Provides a consistent interface for tensor operations while managing underlying TF.js tensors.
 */
export declare class TensorWrapper implements ITensor {
    private tensor;
    constructor(tensor: tf.Tensor);
    [key: string]: any;
    static fromTensor(tensor: tf.Tensor): TensorWrapper;
    get shape(): number[];
    dataSync(): Float32Array | Int32Array | Uint8Array;
    dispose(): void;
    toJSON(): any;
}
/**
 * Creates a wrapped tensor from a TensorFlow.js tensor.
 * @param tensor TensorFlow.js tensor to wrap
 * @returns Wrapped tensor
 */
export declare function wrapTensor(tensor: tf.Tensor): ITensor;
/**
 * Unwraps a tensor to get the underlying TensorFlow.js tensor.
 * @param tensor Wrapped tensor
 * @returns Underlying TensorFlow.js tensor
 * @throws Error if tensor is not a TensorWrapper
 */
export declare function unwrapTensor(tensor: ITensor): tf.Tensor;
/**
 * Zod schema for memory storage input validation.
 */
export declare const StoreMemoryInput: z.ZodObject<{
    subject: z.ZodString;
    relationship: z.ZodString;
    object: z.ZodString;
}, "strip", z.ZodTypeAny, {
    object: string;
    subject: string;
    relationship: string;
}, {
    object: string;
    subject: string;
    relationship: string;
}>;
/**
 * Zod schema for memory recall input validation.
 */
export declare const RecallMemoryInput: z.ZodObject<{
    query: z.ZodString;
}, "strip", z.ZodTypeAny, {
    query: string;
}, {
    query: string;
}>;
