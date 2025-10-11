/** @type {import('jest').Config} */
export default {
  preset: 'ts-jest/presets/default-esm',
  testEnvironment: 'node',
  extensionsToTreatAsEsm: ['.ts'],
  moduleNameMapper: {
    '^(\\.{1,2}/.*)\\.js$': '$1',
    '^util$': '<rootDir>/test/shims/util.cjs',
  },
  transform: {
    '^.+\\.tsx?$': [
      'ts-jest',
      {
        useESM: true,
      },
    ],
  },
  transformIgnorePatterns: ['node_modules/(?!(@tensorflow/tfjs|@modelcontextprotocol)/)'],
  testPathIgnorePatterns: ['/node_modules/', '/build/', '/dist/'],
  maxWorkers: 1,
  forceExit: true,
  testTimeout: 30000,
  collectCoverageFrom: ['src/**/*.{ts,tsx}', '!src/**/*.d.ts'],
};
