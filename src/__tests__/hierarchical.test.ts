import { HopeMemoryModel } from '../model.js';

describe('HopeMemoryModel auxiliary state', () => {
  let model: HopeMemoryModel;

  beforeEach(async () => {
    model = new HopeMemoryModel();
    await model.initialize();
  });

  afterEach(() => {
    model.dispose();
  });

  it('exports auxiliary state with pending updates array', () => {
    const auxiliary = model.exportAuxiliaryState();
    expect(auxiliary).toHaveProperty('config');
    expect(auxiliary).toHaveProperty('pendingUpdates');
  });

  it('hydrates memory state for serialization round-trip', () => {
    const state = model.createInitialState();
    model.hydrateMemoryState(state);
    const updated = model.createInitialState();
    expect(updated.shortTerm.shape[0]).toBe(0);
  });
});
