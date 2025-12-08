import { readdirSync, readFileSync } from 'fs';
import { join } from 'path';

const CONFLICT_PATTERNS = ['<<<<<<<', '=======', '>>>>>>>'];
const IGNORED_DIRS = new Set(['.git', 'node_modules', 'dist', '.hope_memory']);

function walk(dir: string): string[] {
  const entries = readdirSync(dir, { withFileTypes: true });
  const files: string[] = [];

  for (const entry of entries) {
    if (IGNORED_DIRS.has(entry.name)) continue;
    const fullPath = join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...walk(fullPath));
    } else {
      files.push(fullPath);
    }
  }

  return files;
}

describe('Conflict marker guard', () => {
  it('fails if any merge conflict markers exist', () => {
    const repoRoot = join(__dirname, '..', '..');
    const files = walk(repoRoot).filter((file) =>
      /\.(ts|js|json|md|yaml|yml|tsx|jsx)$/i.test(file)
    );

    const offenders: string[] = [];
    for (const file of files) {
      const contents = readFileSync(file, 'utf8');
      if (CONFLICT_PATTERNS.some((marker) => contents.includes(marker))) {
        offenders.push(file);
      }
    }

    if (offenders.length > 0) {
      const message = `Merge conflict markers found in:\n${offenders.join('\n')}`;
      throw new Error(message);
    }
  });
});
