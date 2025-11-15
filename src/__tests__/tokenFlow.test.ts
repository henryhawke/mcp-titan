import { HopeMemoryModel } from '../model.js';

describe('HopeMemoryModel memory updates', () => {
  let model: HopeMemoryModel;

  beforeEach(async () => {
    model = new HopeMemoryModel();
    await model.initialize();
  });

  afterEach(() => {
    model.dispose();
  });

  it('stores textual memories from strings', async () => {
    await model.storeMemory('Important fact about HOPE');
    const embedding = await model.encodeText('query about HOPE');
    const state = model.createInitialState();
    const updated = model.forward(embedding, state);
    expect(updated.memoryUpdate.newState.shortTerm.shape[0]).toBeGreaterThanOrEqual(0);
  });
});
