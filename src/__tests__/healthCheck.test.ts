import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import * as path from 'path';

import { TitanMemoryServer, TITAN_MEMORY_VERSION } from '../index.js';

describe('TitanMemoryServer health check', () => {
  it('reports a version consistent with the server metadata', async () => {
    const tempDir = mkdtempSync(path.join(tmpdir(), 'titan-memory-test-'));

    try {
      const server = new TitanMemoryServer({ memoryPath: tempDir });
      const health = await (server as unknown as { performHealthCheck: (type: string) => Promise<any> }).performHealthCheck('basic');

      expect(health.version).toBe(TITAN_MEMORY_VERSION);
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });
});
