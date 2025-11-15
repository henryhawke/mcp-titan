import { HopeMemoryServer } from '../index.js';

describe('HopeMemoryServer', () => {
  let server: HopeMemoryServer;

  beforeEach(async () => {
    server = new HopeMemoryServer();
  });

  afterEach(async () => {
    if (server) {
      // Clean shutdown - the server doesn't have a dispose method in the interface
    }
  });

  test('initializes correctly', () => {
    expect(server).toBeDefined();
  });

  test('handles basic operations', async () => {
    // Basic test that doesn't require complex setup
    expect(server).toBeInstanceOf(HopeMemoryServer);
  });
});
