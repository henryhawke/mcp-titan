# MCP Titan / HOPE Memory - Operations Manual

This manual explains how to stand up, verify, and operate the HOPE Memory MCP server from a fresh clone of the repository.

## 1. Prerequisites
- Node.js **22+**
- [Bun](https://bun.sh) **1.2+**
- CPU with AVX2 support (recommended for TensorFlow.js). The binary will still run without AVX2, but TensorFlow will log a warning.

## 2. Install & Build
```bash
cd /workspace/mcp-titan
bun install            # install dependencies
bun run build          # transpile TypeScript into dist/
```

> Re-run `bun run build` after making TypeScript changes so `index.js` can import `dist/index.js`.

## 3. Launch the MCP Server (stdio)
```bash
# from the repo root after building
bun start
```

- The server starts on stdio and registers all MCP tools immediately.
- State, checkpoints, and logs are stored in `./.hope_memory/` by default. Run the server from the directory where you want this folder created.
- To change the storage location, start the server from the target working directory (the constructor reads `process.cwd()`).

### MCP client configuration example
Create `~/.mcp/servers/hope-memory.json` (or add to your client’s config) with:
```json
{
  "command": "bun",
  "args": ["index.js"],
  "cwd": "/workspace/mcp-titan"
}
```
Then restart your MCP client (Cursor/Claude Desktop) and select **hope-memory** from the available servers.

## 4. Quick Verification
Run a fast health check and model smoke test:
```bash
# TypeScript soundness
bun run typecheck

# Minimal TensorFlow + model pipeline test
bun test src/__tests__/healthCheck.test.ts
bun test src/__tests__/model.test.ts
```

TensorFlow emits an informational log about CPU optimizations on first run; this is expected.

## 5. Memory persistence & checkpoints
- Checkpoints live under `./.hope_memory/checkpoints/`.
- Use the MCP tools `save_checkpoint`, `export_checkpoint`, and `load_checkpoint` to manage snapshots. The helpers include checksums and config validation to prevent mismatched dimensions.

## 6. Optional: Data workflows
The repository includes lightweight training utilities:
```bash
bun run download-data   # fetch sample data (if configured)
bun run train-quick     # quick demo training loop
bun run train-model     # fuller training loop
```
These commands assume `bun run build` has already produced `dist/` artifacts.

## 7. Logs
- Structured logs are written to `./.hope_memory/logs/`.
- Increase verbosity with `LOG_LEVEL=DEBUG bun start`.
