// Polyfill for util.isNullOrUndefined which was removed in newer Node.js versions
export function isNullOrUndefined(value: any): boolean {
    return value === null || value === undefined;
}

// Make it available globally for TensorFlow.js
(global as any).isNullOrUndefined = isNullOrUndefined;

// Minimal performance.now polyfill for environments missing it
if (!(globalThis as any).performance) {
    (globalThis as any).performance = { now: () => Date.now() } as any;
} 