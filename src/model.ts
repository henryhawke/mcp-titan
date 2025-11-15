import { HopeMemoryModel, type HopeMemoryConfig } from './hope_model/index.js';

export { HopeMemoryModel } from './hope_model/index.js';
export type { HopeMemoryConfig } from './hope_model/index.js';

/**
 * Backwards-compatible alias: legacy code that instantiates TitanMemoryModel
 * will now receive the HOPE implementation transparently.
 */
export class TitanMemoryModel extends HopeMemoryModel {}
export type TitanMemoryConfig = HopeMemoryConfig;
export const TitanMemorySystem = HopeMemoryModel;
