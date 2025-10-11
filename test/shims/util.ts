import * as realUtil from 'node:util';

export * from 'node:util';

export const isNullOrUndefined = (value: any): boolean => value === null || value === undefined;

export default { ...realUtil, isNullOrUndefined } as any;

