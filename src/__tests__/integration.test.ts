/**
 * Integration Tests for MCP-Titan Memory Server
 * Tests end-to-end workflows through all 11 MCP tools
 */

import * as tf from '@tensorflow/tfjs-node';
import { TitanMemoryServer } from '../index.js';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';

describe('MCP Integration Tests', () => {
    let server: TitanMemoryServer;
    let testDir: string;

    beforeAll(async () => {
        // Create temporary directory for test files
        testDir = await fs.mkdtemp(path.join(os.tmpdir(), 'mcp-titan-test-'));
    });

    afterAll(async () => {
        // Cleanup test directory
        try {
            await fs.rm(testDir, { recursive: true, force: true });
        } catch (error) {
            console.warn('Failed to cleanup test directory:', error);
        }
    });

    beforeEach(() => {
        server = new TitanMemoryServer({ memoryPath: testDir });
    });

    afterEach(() => {
        // Dispose tensors
        tf.disposeVariables();
    });

    describe('Complete Workflow: Init → Forward → Train → Save → Load', () => {
        test('should complete full lifecycle without errors', async () => {
            // Step 1: Initialize model
            const initResult = await (server as any).server.callTool('init_model', {
                inputDim: 128,
                memorySlots: 100,
                transformerLayers: 2
            });

            expect(initResult).toBeDefined();
            expect(initResult.content).toBeDefined();
            expect(initResult.content[0].text).toContain('initialized successfully');

            // Step 2: Forward pass with text
            const forwardResult = await (server as any).server.callTool('forward_pass', {
                x: 'test input text'
            });

            expect(forwardResult).toBeDefined();
            expect(forwardResult.content[0].text).toContain('Forward pass completed');

            // Step 3: Training step
            const trainResult = await (server as any).server.callTool('train_step', {
                x_t: 'current input',
                x_next: 'next input'
            });

            expect(trainResult).toBeDefined();
            expect(trainResult.content[0].text).toContain('Training step completed');
            expect(trainResult.content[0].text).toContain('Loss:');

            // Step 4: Get memory state
            const stateResult = await (server as any).server.callTool('get_memory_state', {});

            expect(stateResult).toBeDefined();
            expect(stateResult.content[0].text).toContain('Memory State');
            expect(stateResult.content[0].text).toContain('Capacity');

            // Step 5: Save checkpoint
            const checkpointPath = path.join(testDir, 'test-checkpoint.json');
            const saveResult = await (server as any).server.callTool('save_checkpoint', {
                path: checkpointPath
            });

            expect(saveResult).toBeDefined();
            expect(saveResult.content[0].text).toContain('Checkpoint saved');

            // Verify file exists
            const fileExists = await fs.access(checkpointPath).then(() => true).catch(() => false);
            expect(fileExists).toBe(true);

            // Step 6: Load checkpoint
            const loadResult = await (server as any).server.callTool('load_checkpoint', {
                path: checkpointPath
            });

            expect(loadResult).toBeDefined();
            expect(loadResult.content[0].text).toContain('Checkpoint loaded');
        });
    });

    describe('Input Validation', () => {
        beforeEach(async () => {
            await (server as any).server.callTool('init_model', {
                inputDim: 128,
                memorySlots: 100
            });
        });

        test('should reject empty string input', async () => {
            const result = await (server as any).server.callTool('forward_pass', {
                x: ''
            });

            expect(result.content[0].text).toContain('cannot be empty');
        });

        test('should reject invalid array input with NaN', async () => {
            const result = await (server as any).server.callTool('forward_pass', {
                x: [1, 2, NaN, 4]
            });

            expect(result.content[0].text).toContain('valid finite numbers');
        });

        test('should reject mismatched dimensions in train_step', async () => {
            const result = await (server as any).server.callTool('train_step', {
                x_t: [1, 2, 3],
                x_next: [1, 2, 3, 4, 5]  // Different length
            });

            expect(result.content[0].text).toContain("dimensions don't match");
        });

        test('should sanitize control characters in text input', async () => {
            const result = await (server as any).server.callTool('forward_pass', {
                x: 'test\x00\x01\x02text'  // Control characters
            });

            // Should succeed but sanitize the input
            expect(result.content[0].text).toContain('Forward pass completed');
        });
    });

    describe('Path Security', () => {
        beforeEach(async () => {
            await (server as any).server.callTool('init_model', {
                inputDim: 128,
                memorySlots: 100
            });
        });

        test('should prevent path traversal in save_checkpoint', async () => {
            const result = await (server as any).server.callTool('save_checkpoint', {
                path: '../../../etc/passwd'
            });

            expect(result.content[0].text).toContain('Path traversal detected');
        });

        test('should prevent path traversal in load_checkpoint', async () => {
            const result = await (server as any).server.callTool('load_checkpoint', {
                path: '../../../etc/passwd'
            });

            expect(result.content[0].text).toContain('Path traversal detected');
        });

        test('should allow valid paths within allowed directories', async () => {
            const validPath = path.join(testDir, 'valid-checkpoint.json');

            // Save first
            const saveResult = await (server as any).server.callTool('save_checkpoint', {
                path: validPath
            });

            expect(saveResult.content[0].text).toContain('Checkpoint saved');

            // Then load
            const loadResult = await (server as any).server.callTool('load_checkpoint', {
                path: validPath
            });

            expect(loadResult.content[0].text).toContain('Checkpoint loaded');
        });
    });

    describe('Dimension Validation', () => {
        test('should detect dimension mismatch on checkpoint load', async () => {
            // Initialize with inputDim=128
            await (server as any).server.callTool('init_model', {
                inputDim: 128,
                memorySlots: 100
            });

            const checkpointPath = path.join(testDir, 'dim-test.json');

            // Save checkpoint
            await (server as any).server.callTool('save_checkpoint', {
                path: checkpointPath
            });

            // Reinitialize with different inputDim
            await (server as any).server.callTool('init_model', {
                inputDim: 256,  // Different dimension
                memorySlots: 100
            });

            // Try to load checkpoint with mismatched dimensions
            const loadResult = await (server as any).server.callTool('load_checkpoint', {
                path: checkpointPath
            });

            expect(loadResult.content[0].text).toContain('dimension mismatch');
        });
    });

    describe('Memory Operations', () => {
        beforeEach(async () => {
            await (server as any).server.callTool('init_model', {
                inputDim: 128,
                memorySlots: 100,
                transformerLayers: 2
            });
        });

        test('should handle manifold_step', async () => {
            // Create base and velocity arrays
            const base = new Array(128).fill(0).map(() => Math.random());
            const velocity = new Array(128).fill(0).map(() => Math.random() * 0.1);

            const result = await (server as any).server.callTool('manifold_step', {
                base,
                velocity
            });

            expect(result).toBeDefined();
            expect(result.content[0].text).toContain('Manifold step completed');
        });

        test('should handle prune_memory with custom threshold', async () => {
            // Add some memories first
            await (server as any).server.callTool('forward_pass', {
                x: 'memory 1'
            });
            await (server as any).server.callTool('forward_pass', {
                x: 'memory 2'
            });

            const result = await (server as any).server.callTool('prune_memory', {
                threshold: 0.7
            });

            expect(result).toBeDefined();
            expect(result.content[0].text).toContain('pruned');
        });

        test('should reset gradients', async () => {
            const result = await (server as any).server.callTool('reset_gradients', {});

            expect(result).toBeDefined();
            expect(result.content[0].text).toContain('Gradients reset');
        });
    });

    describe('Help Tool', () => {
        test('should provide help for all tools', async () => {
            const result = await (server as any).server.callTool('help', {});

            expect(result).toBeDefined();
            expect(result.content[0].text).toContain('Available tools');
            expect(result.content[0].text).toContain('init_model');
            expect(result.content[0].text).toContain('forward_pass');
            expect(result.content[0].text).toContain('train_step');
        });

        test('should provide detailed help with verbose flag', async () => {
            const result = await (server as any).server.callTool('help', {
                verbose: true
            });

            expect(result).toBeDefined();
            expect(result.content[0].text).toContain('Available tools');
        });
    });

    describe('Error Recovery', () => {
        beforeEach(async () => {
            await (server as any).server.callTool('init_model', {
                inputDim: 128,
                memorySlots: 100
            });
        });

        test('should handle graceful failure on invalid operations', async () => {
            // Try to load non-existent checkpoint
            const result = await (server as any).server.callTool('load_checkpoint', {
                path: path.join(testDir, 'nonexistent.json')
            });

            expect(result).toBeDefined();
            expect(result.content[0].text).toContain('Failed');
        });

        test('should continue operating after failed operations', async () => {
            // Cause an error
            await (server as any).server.callTool('load_checkpoint', {
                path: path.join(testDir, 'nonexistent.json')
            });

            // Should still be able to perform operations
            const result = await (server as any).server.callTool('forward_pass', {
                x: 'test after error'
            });

            expect(result.content[0].text).toContain('Forward pass completed');
        });
    });

    describe('Concurrent Operations', () => {
        beforeEach(async () => {
            await (server as any).server.callTool('init_model', {
                inputDim: 128,
                memorySlots: 100
            });
        });

        test('should handle multiple forward passes in sequence', async () => {
            const results = [];

            for (let i = 0; i < 5; i++) {
                const result = await (server as any).server.callTool('forward_pass', {
                    x: `input ${i}`
                });
                results.push(result);
            }

            expect(results).toHaveLength(5);
            results.forEach(result => {
                expect(result.content[0].text).toContain('Forward pass completed');
            });
        });

        test('should handle alternating forward and train operations', async () => {
            for (let i = 0; i < 3; i++) {
                const forwardResult = await (server as any).server.callTool('forward_pass', {
                    x: `input ${i}`
                });
                expect(forwardResult.content[0].text).toContain('Forward pass completed');

                const trainResult = await (server as any).server.callTool('train_step', {
                    x_t: `input ${i}`,
                    x_next: `input ${i + 1}`
                });
                expect(trainResult.content[0].text).toContain('Training step completed');
            }
        });
    });

    describe('Memory State Persistence', () => {
        test('should persist and restore memory across server instances', async () => {
            // First instance
            const server1 = new TitanMemoryServer({ memoryPath: testDir });

            await (server1 as any).server.callTool('init_model', {
                inputDim: 128,
                memorySlots: 100
            });

            await (server1 as any).server.callTool('forward_pass', {
                x: 'persistent memory test'
            });

            const checkpointPath = path.join(testDir, 'persist-test.json');
            await (server1 as any).server.callTool('save_checkpoint', {
                path: checkpointPath
            });

            // Second instance
            const server2 = new TitanMemoryServer({ memoryPath: testDir });

            await (server2 as any).server.callTool('init_model', {
                inputDim: 128,
                memorySlots: 100
            });

            const loadResult = await (server2 as any).server.callTool('load_checkpoint', {
                path: checkpointPath
            });

            expect(loadResult.content[0].text).toContain('Checkpoint loaded');

            // Should be able to continue operations
            const forwardResult = await (server2 as any).server.callTool('forward_pass', {
                x: 'after restore'
            });

            expect(forwardResult.content[0].text).toContain('Forward pass completed');
        });
    });
});


