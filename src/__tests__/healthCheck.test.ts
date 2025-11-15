import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import * as path from 'path';

import { HopeMemoryServer, HOPE_MEMORY_VERSION } from '../index.js';

describe('HopeMemoryServer health check', () => {
  it('reports a version consistent with the server metadata', async () => {
    const tempDir = mkdtempSync(path.join(tmpdir(), 'hope-memory-test-'));

    try {
      const server = new HopeMemoryServer({ memoryPath: tempDir });
      const health = await (server as unknown as { performHealthCheck: (type: string) => Promise<any> }).performHealthCheck('basic');

      expect(health.version).toBe(HOPE_MEMORY_VERSION);
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });
});
