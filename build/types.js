import { z } from "zod";
// Simple wrapper
export class TensorWrapper {
    constructor(tensor) {
        this.tensor = tensor;
    }
    static fromTensor(tensor) {
        return new TensorWrapper(tensor);
    }
    get shape() {
        return this.tensor.shape;
    }
    dataSync() {
        return Array.from(this.tensor.dataSync());
    }
    dispose() {
        this.tensor.dispose();
    }
    toJSON() {
        return {
            dataSync: this.dataSync(),
            shape: this.shape
        };
    }
}
export function wrapTensor(tensor) {
    return TensorWrapper.fromTensor(tensor);
}
export function unwrapTensor(tensor) {
    if (tensor instanceof TensorWrapper) {
        return tensor.tensor;
    }
    throw new Error('Cannot unwrap non-TensorWrapper object');
}
export const StoreMemoryInput = z.object({
    subject: z.string(),
    relationship: z.string(),
    object: z.string()
});
export const RecallMemoryInput = z.object({
    query: z.string()
});
//# sourceMappingURL=types.js.map