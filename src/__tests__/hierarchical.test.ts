import { jest } from '@jest/globals';
import * as tf from '@tensorflow/tfjs-node';
import { TitanMemoryModel } from '../model.js';
import { wrapTensor, unwrapTensor, type IMemoryState } from '../types.js';

const disposeState = (state: IMemoryState): void => {
    unwrapTensor(state.shortTerm).dispose();
    unwrapTensor(state.longTerm).dispose();
    unwrapTensor(state.meta).dispose();
    unwrapTensor(state.timestamps).dispose();
    unwrapTensor(state.accessCounts).dispose();
    unwrapTensor(state.surpriseHistory).dispose();
    state.momentumState && unwrapTensor(state.momentumState).dispose();
    state.tokenFlowHistory && unwrapTensor(state.tokenFlowHistory).dispose();
    state.flowWeights && unwrapTensor(state.flowWeights).dispose();
    state.forgettingGate && unwrapTensor(state.forgettingGate).dispose();
};

describe('Hierarchical Memory Promotion/Demotion', () => {
    let model: TitanMemoryModel;

    beforeAll(async () => {
        model = new TitanMemoryModel();
        await model.initialize({
            inputDim: 16,
            memoryDim: 16,
            memorySlots: 10,
            useHierarchicalMemory: true,
            enableHierarchicalMemory: true,
            enableMomentum: false,
            enableForgettingGate: false
        });
    });

    afterAll(() => {
        model.dispose();
    });

    test('should apply memory promotion during forward pass', () => {
        const state = model.getMemoryState();
        const inputDim = model.getConfig().inputDim;

        const input = tf.randomNormal([inputDim]);

        // Perform forward pass which should trigger promotion logic
        const result = model.forward(wrapTensor(input), state);

        expect(result.memoryUpdate.newState).toBeDefined();
        expect(result.memoryUpdate.newState.shortTerm).toBeDefined();
        expect(result.memoryUpdate.newState.longTerm).toBeDefined();

        input.dispose();
        unwrapTensor(result.predicted).dispose();
        disposeState(state);
    });

    test('should track promotion/demotion statistics', () => {
        const stats = (model as any).memoryStats;

        expect(stats).toBeDefined();
        expect(stats.promotions).toBeDefined();
        expect(stats.demotions).toBeDefined();
        expect(stats.promotions.total).toBeGreaterThanOrEqual(0);
        expect(stats.demotions.total).toBeGreaterThanOrEqual(0);
    });

    test('should respect promotion rules', () => {
        const rules = (model as any).promotionRules;

        expect(rules).toBeDefined();
        expect(rules.workingToShortTerm).toBeDefined();
        expect(rules.shortTermToLongTerm).toBeDefined();
        expect(rules.demotionRules).toBeDefined();

        expect(rules.workingToShortTerm.accessThreshold).toBeGreaterThan(0);
        expect(rules.shortTermToLongTerm.accessThreshold).toBeGreaterThan(0);
    });

    test('should handle hierarchical memory configuration', () => {
        const config = model.getConfig();

        // Verify hierarchical memory is enabled
        expect(config.enableHierarchicalMemory).toBe(true);

        // Verify promotion rules exist
        const rules = (model as any).promotionRules;
        expect(rules.workingToShortTerm.accessThreshold).toBeDefined();
        expect(rules.shortTermToLongTerm.accessThreshold).toBeDefined();
    });

    test('should have appropriate memory tier sizes', () => {
        const config = model.getConfig();

        // Memory slots should be allocated appropriately
        expect(config.memorySlots).toBe(10);
        expect(config.memoryDim).toBe(16);
    });

    test('should retrieve normalized vector without dimension errors', async () => {
        const testModel = new TitanMemoryModel();
        await testModel.initialize({
            inputDim: 16,
            memoryDim: 16,
            memorySlots: 10,
            useHierarchicalMemory: true,
            enableHierarchicalMemory: true,
            enableMomentum: false,
            enableForgettingGate: false
        });

        const inputDim = testModel.getConfig().inputDim;
        const input = tf.randomNormal([inputDim]);

        const originalRetrieve = (testModel as any).retrieveFromHierarchicalMemory;
        const retrievedVectors: tf.Tensor[] = [];

        const retrieveSpy = jest
            .spyOn(testModel as any, 'retrieveFromHierarchicalMemory')
            .mockImplementation(function (this: any, query: any) {
                const result = originalRetrieve.call(this, query);
                const tensor = unwrapTensor(result) as tf.Tensor;
                retrievedVectors.push(tf.keep(tensor.clone()));
                return result;
            });

        let forwardResult: ReturnType<TitanMemoryModel['forward']> | undefined;
        try {
            expect(() => {
                forwardResult = testModel.forward(wrapTensor(input));
            }).not.toThrow();

            expect(retrievedVectors.length).toBeGreaterThan(0);
            const retrieved = retrievedVectors[0];
            const normTensor = retrieved.norm();
            const normValue = normTensor.arraySync() as number;

            expect(retrieved.shape.length).toBe(1);
            expect(normValue).toBeGreaterThanOrEqual(0);
            expect(normValue).toBeLessThanOrEqual(1 + 1e-5);

            normTensor.dispose();
        } finally {
            retrieveSpy.mockRestore();
            retrievedVectors.forEach(tensor => tensor.dispose());

            if (forwardResult) {
                unwrapTensor(forwardResult.predicted).dispose();
            }

            input.dispose();
            testModel.dispose();
        }

        expect(forwardResult).toBeDefined();
    });
});

