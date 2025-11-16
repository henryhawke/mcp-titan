# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **mcp-titan**, a sophisticated neural memory system for LLMs that provides persistent memory capabilities through an MCP (Model Context Protocol) server. The system uses TensorFlow.js for neural computations and implements transformer-based memory mechanisms.

## Development Commands

### Core Commands
- `bun run build` - Compile TypeScript to JavaScript
- `bun run typecheck` - Type check without emitting files  
- `bun run lint` - Run ESLint with zero warnings policy
- `bun run lint:fix` - Auto-fix ESLint issues
- `bun test` - Run Jest tests with experimental VM modules
- `bun run clean` - Remove build artifacts

### Model Training Commands
- `bun run train-model` - Train the model with default settings
- `bun run train-quick` - Quick training for development
- `bun run train-production` - Full production training
- `bun run download-data` - Download training datasets
- `bun run test-model` - Test model functionality

### Server Commands
- `bun start` - Start the MCP server (production)
- `bun --watch src/index.ts` - Development mode with auto-restart

## Architecture Overview

### Core Components

**TitanMemoryServer** (`src/index.ts`) - Main MCP server that registers 16 tools for memory operations including:
- Model initialization and configuration
- Forward passes and training steps
- Memory state management and pruning
- Checkpoint saving/loading
- Bootstrap memory from URLs/text
- Online learning service

**TitanMemoryModel** (`src/model.ts`) - Neural memory implementation with:
- Transformer-XL inspired architecture
- Hierarchical memory (short-term, long-term, meta)
- Information-gain based pruning
- Surprise-based learning
- Tensor memory management

**Key Services:**
- **LearnerService** (`src/learner.ts`) - Online learning with replay buffer
- **AdvancedTokenizer** (`src/tokenizer/`) - BPE tokenization and embeddings  
- **MemoryPruner** (`src/pruning.ts`) - Intelligent memory pruning
- **VectorProcessor** (`src/utils.ts`) - Text encoding and vector operations

### Memory Architecture

The system implements a three-tier memory hierarchy:
1. **Short-term memory** - Recent activations and working memory
2. **Long-term memory** - Consolidated patterns and knowledge
3. **Meta memory** - Statistics and metadata about memory usage

Memory is automatically pruned using information-gain scoring to maintain performance and prevent memory bloat.

## MCP Integration

The server is designed to work with Cursor and other MCP-compatible clients. Configuration example:

```json
{
  "titan-memory": {
    "command": "bun",
    "args": ["index.js"], 
    "cwd": "/path/to/mcp-titan"
  }
}
```

## Dependencies

- **@tensorflow/tfjs-node** - Neural network computations
- **@modelcontextprotocol/sdk** - MCP protocol implementation
- **zod** - Runtime type validation for tool parameters
- **ws** - WebSocket transport support

## Testing

Tests are in `src/__tests__/` and cover:
- Model functionality and training
- Memory operations and pruning  
- Server tool integration
- Tokenizer capabilities

Run `bun test` to execute the full test suite.

## Important Notes

- The system requires Node.js >=22.0.0 for optimal performance
- Memory state is automatically saved every minute and on shutdown
- Use `reset_gradients` tool if training becomes unstable
- Pruning is automatically triggered when memory capacity exceeds limits
- The server includes comprehensive error handling and telemetry