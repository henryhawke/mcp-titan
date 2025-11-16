import * as tf from '@tensorflow/tfjs-node';

/**
 * Type-safe wrapper for tf.tidy that allows custom return types
 * while preserving memory management benefits.
 *
 * This wrapper solves the issue where tf.tidy expects TensorContainer
 * but we need to return complex objects containing tensors.
 *
 * @param fn Function that returns an object containing tensors
 * @returns The result with all tensors properly managed
 */
export function tidyMemoryState<T extends Record<string, any>>(
  fn: () => T
): T {
  return tf.tidy(() => {
    const result = fn();

    // Keep all tensor properties so they're not disposed
    for (const key in result) {
      const value = result[key];
      if (value instanceof tf.Tensor) {
        tf.keep(value);
      } else if (Array.isArray(value)) {
        value.forEach(item => {
          if (item instanceof tf.Tensor) {
            tf.keep(item);
          }
        });
      }
    }

    return result;
  }) as T;
}

/**
 * Type-safe wrapper for operations that should return Tensor2D
 *
 * @param fn Function that returns a tensor
 * @returns Tensor2D with runtime validation
 */
export function tidyTensor2D(fn: () => tf.Tensor): tf.Tensor2D {
  const result = tf.tidy(fn);
  if (result.rank !== 2) {
    throw new Error(`Expected 2D tensor, got rank ${result.rank}`);
  }
  return result as tf.Tensor2D;
}

/**
 * Type-safe wrapper for operations that should return Tensor1D
 *
 * @param fn Function that returns a tensor
 * @returns Tensor1D with runtime validation
 */
export function tidyTensor1D(fn: () => tf.Tensor): tf.Tensor1D {
  const result = tf.tidy(fn);
  if (result.rank !== 1) {
    throw new Error(`Expected 1D tensor, got rank ${result.rank}`);
  }
  return result as tf.Tensor1D;
}

/**
 * Ensures a tensor is 2D, adding a batch dimension if necessary
 *
 * @param tensor Input tensor
 * @returns Tensor2D
 */
export function ensure2D(tensor: tf.Tensor): tf.Tensor2D {
  if (tensor.rank === 2) {
    return tensor as tf.Tensor2D;
  } else if (tensor.rank === 1) {
    return tensor.expandDims(0) as tf.Tensor2D;
  } else if (tensor.rank === 0) {
    return tensor.expandDims(0).expandDims(0) as tf.Tensor2D;
  } else {
    // Flatten to 2D
    const size = tensor.size;
    return tensor.reshape([1, size]) as tf.Tensor2D;
  }
}

/**
 * Safely dispose of tensors in an object
 *
 * @param obj Object containing tensors
 */
export function disposeTensors(obj: Record<string, any>): void {
  for (const key in obj) {
    const value = obj[key];
    if (value instanceof tf.Tensor) {
      value.dispose();
    } else if (Array.isArray(value)) {
      value.forEach(item => {
        if (item instanceof tf.Tensor) {
          item.dispose();
        }
      });
    }
  }
}

/**
 * Clone all tensors in an object
 *
 * @param obj Object containing tensors
 * @returns New object with cloned tensors
 */
export function cloneTensors<T extends Record<string, any>>(obj: T): T {
  const result: any = {};
  for (const key in obj) {
    const value = obj[key];
    if (value instanceof tf.Tensor) {
      result[key] = value.clone();
    } else if (Array.isArray(value)) {
      result[key] = value.map(item =>
        item instanceof tf.Tensor ? item.clone() : item
      );
    } else {
      result[key] = value;
    }
  }
  return result as T;
}
