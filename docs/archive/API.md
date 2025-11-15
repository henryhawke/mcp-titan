# Titan Memory MCP Server API Documentation

## Overview

The Titan Memory MCP Server provides a neural memory system that can learn and predict sequences while maintaining state through a memory vector. This document details the available tools and their usage.

## Connection

### Cursor Integration

To use the server with Cursor IDE:

1. Install the server:

```bash
npm install -g @henryhawke/mcp-titan
```

2. Add to Cursor's MCP configuration (`~/.cursor/settings.json`):

```json
{
  "mcp": {
    "servers": {
      "titan-memory": {
        "command": "mcp-titan",
        "env": {
          "NODE_ENV": "production"
        }
      }
    }
  }
}
```

3. Restart Cursor IDE
4. Use `Cmd/Ctrl + Shift + P` and type "MCP: Restart Servers" to initialize

## Available Tools

### process_input

Process text input and update the memory state.

#### Parameters

```typescript
{
  text: string;      // Required: Input text to process
  context?: string;  // Optional: Additional context
}
```

#### Response

```typescript
{
  surprise: number; // Surprise level (0-1)
  memoryUpdated: boolean; // Whether memory was updated
  insight: string; // Human-readable insight
}
```

#### Example

```typescript
const result = await callTool("process_input", {
  text: "function calculateSum(a, b) { return a + b; }",
  context: "JavaScript arithmetic function",
});
```

### get_memory_state

Retrieve the current memory state and statistics.

#### Parameters

None required.

#### Response

```typescript
{
  memoryStats: {
    mean: number; // Mean of memory vector
    std: number; // Standard deviation
  }
  memorySize: number; // Size of memory vector
  status: string; // Current status
}
```

#### Example

```typescript
const state = await callTool("get_memory_state", {});
```

## HTTP API

The server also exposes HTTP endpoints for non-MCP integrations.

### GET /

Returns server information and status.

#### Response

```json
{
  "name": "Titan Memory MCP Server",
  "version": "0.1.0",
  "description": "Automatic memory-augmented learning for Cursor",
  "status": "active",
  "memoryPath": "~/.cursor/titan-memory"
}
```

### POST /mcp

MCP protocol endpoint for tool execution.

#### Request

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "process_input",
    "arguments": {
      "text": "Sample input text"
    }
  },
  "id": 1
}
```

## Error Handling

The server uses standard MCP error codes:

- `INVALID_REQUEST`: Malformed request
- `METHOD_NOT_FOUND`: Unknown tool requested
- `INVALID_PARAMS`: Invalid tool parameters
- `INTERNAL_ERROR`: Server-side error

Example error response:

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32602,
    "message": "Invalid params: text is required"
  },
  "id": 1
}
```

## Memory Management

The server automatically:

- Saves memory state every 5 minutes
- Persists memory to `~/.cursor/titan-memory/memory.json`
- Cleans up tensors to prevent memory leaks
- Handles process termination gracefully

## Performance Considerations

- Memory vector size: 768 dimensions
- Auto-save interval: 5 minutes
- Uses TensorFlow.js for efficient tensor operations
- Dynamic port allocation for HTTP server

## Security Notes

- Server runs locally only
- File access restricted to `~/.cursor/titan-memory`
- No external network calls
- Environment variables sanitized

## Debugging

Enable debug logging:

```bash
DEBUG=mcp-titan* mcp-titan
```

Log files location:

- Server logs: `~/.cursor/titan-memory/logs/server.log`
- Memory state: `~/.cursor/titan-memory/memory.json`
