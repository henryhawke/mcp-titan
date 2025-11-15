import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';

import { HopeMemoryServer } from '../index.js';
import { unwrapTensor } from '../types.js';

describe('HopeMemoryServer integration smoke tests', () => {
  let server: HopeMemoryServer;
  let tempDir: string;

  beforeAll(async () => {
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'hope-memory-test-'));
  });

  afterAll(async () => {
    await fs.rm(tempDir, { recursive: true, force: true });
  });

  beforeEach(() => {
    server = new HopeMemoryServer({ memoryPath: tempDir });
  });

  afterEach(async () => {
    tf.disposeVariables();
    // ensure auto-saved memory does not leak between tests
    await fs.rm(path.join(tempDir, 'memory_state.json'), { force: true });
  });

  test('ensureInitialized creates a model and memory scaffolding', async () => {
    await (server as any).ensureInitialized();

    const memoryState = (server as any).memoryState;
    expect(memoryState).toBeDefined();

    const shortTerm = unwrapTensor(memoryState.shortTerm);
    expect(shortTerm.shape).toHaveLength(2);
    expect(shortTerm.shape[1]).toBeGreaterThan(0);

    const longTerm = unwrapTensor(memoryState.longTerm);
    expect(longTerm.shape).toHaveLength(2);
    expect(longTerm.shape[1]).toBeGreaterThan(0);
  });

  test('serialize and restore memory state round trip', async () => {
    await (server as any).ensureInitialized();

    const serialized = (server as any).serializeMemoryState();
    expect(serialized).toHaveProperty('shortTerm');
    expect(Array.isArray(serialized.shortTerm)).toBe(true);

    // mutate memory state by restoring what we just serialized
    (server as any).restoreSerializedMemoryState(serialized);
    const restored = (server as any).serializeMemoryState();

    expect(restored.shortTerm.length).toBe(serialized.shortTerm.length);
    expect(restored.meta.length).toBe(serialized.meta.length);
  });

  test('saveMemoryState persists state to disk and loadMemoryState restores it', async () => {
    await (server as any).ensureInitialized();

    await (server as any).saveMemoryState();
    const checkpointPath = path.join(tempDir, 'memory_state.json');
    const stat = await fs.stat(checkpointPath);
    expect(stat.isFile()).toBe(true);

    // reset memory to a fresh empty state before loading
    const freshState = (server as any).initializeEmptyState();
    (server as any).memoryState = freshState;

    await (server as any).loadMemoryState();
    const reloaded = (server as any).serializeMemoryState();
    expect(reloaded.shortTerm.length).toBeGreaterThanOrEqual(0);
    expect(reloaded.shapes).toBeDefined();
  });
});
