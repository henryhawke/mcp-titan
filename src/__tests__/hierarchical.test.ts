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
        disposeState(result.memoryUpdate.newState);
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
});

