# Complete Setup Guide for HOPE Memory (MCP-Titan)

**Your exhaustive guide to installing, training, and using the HOPE memory system**

---

## Table of Contents

1. [Prerequisites & System Requirements](#prerequisites--system-requirements)
2. [Installation](#installation)
3. [Quick Start (Using Pre-built)](#quick-start-using-pre-built)
4. [Configuration for MCP Clients](#configuration-for-mcp-clients)
5. [Training from Scratch](#training-from-scratch)
6. [Advanced Training Options](#advanced-training-options)
7. [Using the MCP Server](#using-the-mcp-server)
8. [Development Setup](#development-setup)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Configuration](#advanced-configuration)

---

## Prerequisites & System Requirements

### Minimum Requirements

- **Node.js:** 22.0.0 or higher (REQUIRED)
  - TensorFlow.js requires Node 22+ for optimal performance
  - Check version: `node --version`
- **RAM:** 4GB minimum, 8GB recommended
- **Disk Space:** 2GB for installation + data
- **CPU:** Any modern multi-core processor
- **GPU:** Optional (CPU-only works fine)

### Recommended Requirements

- **Node.js:** 22.x LTS
- **RAM:** 16GB for production training
- **Disk Space:** 10GB+ for large datasets
- **CPU:** 8+ cores for faster training
- **GPU:** CUDA-compatible for GPU acceleration (optional)

### Package Manager

Choose one:
- **npm** (comes with Node.js)
- **bun** (faster, recommended for development)

### Supported Platforms

- ✅ **Linux** (Ubuntu 20.04+, Debian 11+, etc.)
- ✅ **macOS** (10.15+ Catalina or later)
- ✅ **Windows** (10/11 with WSL2 recommended)
- ⚠️ **Windows Native** (works but not recommended)

---

## Installation

### Option 1: Install from npm (Recommended for Users)

```bash
# Using npm
npm install -g @henryhawke/mcp-titan

# Using bun (faster)
bun add -g @henryhawke/mcp-titan

# Verify installation
npx @henryhawke/mcp-titan --version
```

### Option 2: Install from Source (Recommended for Developers)

```bash
# 1. Clone the repository
git clone https://github.com/henryhawke/mcp-titan.git
cd mcp-titan

# 2. Install dependencies
bun install
# OR
npm install

# 3. Build the project
bun run build
# OR
npm run build

# 4. Verify build
ls -lh dist/

# 5. Link for local development (optional)
npm link
```

### Verify Installation

```bash
# Check if all dependencies are installed
node -e "require('@tensorflow/tfjs-node')" && echo "✅ TensorFlow.js OK"
node -e "require('@modelcontextprotocol/sdk')" && echo "✅ MCP SDK OK"
node -e "require('zod')" && echo "✅ Zod OK"

# Check TypeScript compilation
cd mcp-titan
bun run typecheck  # Should output nothing (0 errors)
```

---

## Quick Start (Using Pre-built)

### 1. Initialize Memory System

```bash
# Create a directory for your HOPE memory
mkdir ~/.hope_memory
cd ~/.hope_memory

# Start the HOPE server (first-time initialization)
npx @henryhawke/mcp-titan
```

**What happens:**
- Creates empty memory banks (short-term, long-term, archive)
- Initializes the MCP server
- Starts listening on stdio for MCP tool calls

### 2. Test with Sample Data

```bash
# In a new terminal, test the server
echo '{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "init_model",
    "arguments": {}
  }
}' | npx @henryhawke/mcp-titan
```

Expected output: `{"jsonrpc":"2.0","id":1,"result":{"status":"initialized"}}`

### 3. Add to Your MCP Client

See [Configuration for MCP Clients](#configuration-for-mcp-clients) section below.

---

## Configuration for MCP Clients

### Cursor

**Location:** `~/.cursor/mcp_settings.json` (or through Cursor Settings UI)

```json
{
  "mcpServers": {
    "hope-memory": {
      "command": "npx",
      "args": [
        "-y",
        "@henryhawke/mcp-titan"
      ],
      "env": {
        "MEMORY_DIR": "/Users/yourname/.hope_memory",
        "MODEL_PATH": "/Users/yourname/.hope_memory/model"
      },
      "disabled": false,
      "alwaysAllow": []
    }
  }
}
```

**Steps:**
1. Open Cursor
2. Go to Settings → Features → MCP Servers
3. Click "Add Server"
4. Paste the configuration above
5. Replace `/Users/yourname/` with your actual home directory
6. Click "Save" and restart Cursor

### Claude Desktop

**Location:** `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
**Location:** `%APPDATA%\Claude\claude_desktop_config.json` (Windows)

```json
{
  "mcpServers": {
    "hope-memory": {
      "command": "npx",
      "args": [
        "-y",
        "@henryhawke/mcp-titan"
      ],
      "env": {
        "MEMORY_DIR": "/Users/yourname/.hope_memory"
      }
    }
  }
}
```

**Steps:**
1. Close Claude Desktop completely
2. Open the config file in a text editor
3. Add the `hope-memory` server configuration
4. Save the file
5. Reopen Claude Desktop
6. Look for the 🔌 icon in the chat indicating MCP servers are connected

### Continue.dev (VS Code Extension)

**Location:** `~/.continue/config.json`

```json
{
  "models": [...],
  "mcpServers": [
    {
      "name": "hope-memory",
      "command": "npx",
      "args": ["-y", "@henryhawke/mcp-titan"],
      "env": {
        "MEMORY_DIR": "/path/to/your/memory"
      }
    }
  ]
}
```

### Generic MCP Client

For any MCP-compatible client:

```bash
# Command
npx @henryhawke/mcp-titan

# Or if installed globally
mcp-titan

# Or from source
node /path/to/mcp-titan/dist/index.js
```

---

## Training from Scratch

HOPE can be trained on your own data to learn domain-specific patterns.

### Step 1: Prepare Training Data

#### Option A: Use Synthetic Data (Quick Start)

```bash
# Navigate to your mcp-titan directory
cd mcp-titan

# Generate synthetic training data
bun run download-data --synthetic

# This creates: data/synthetic_training.txt with 10,000 samples
```

#### Option B: Download Real Datasets

```bash
# WikiText-2 (small, good for testing)
bun run download-data --wikitext

# TinyStories (medium, 2.1GB)
bun run download-data --tinystories

# OpenWebText sample (large, 1.2GB)
bun run download-data --openwebtext

# Download all datasets
bun run download-data --all
```

#### Option C: Use Your Own Data

**Format 1: Plain Text (recommended)**

```text
data/my_training.txt:

This is the first training sample.
This is the second training sample.
Each line is a separate training example.
HOPE will learn to predict sequences from this data.
```

**Format 2: JSON Array**

```json
data/my_training.json:

[
  "First training sample",
  "Second training sample",
  "Third training sample"
]
```

**Format 3: JSONL (JSON Lines)**

```jsonl
data/my_training.jsonl:

{"text": "First sample"}
{"text": "Second sample"}
{"text": "Third sample"}
```

### Step 2: Configure Training

Create a configuration file or use environment variables:

**Method 1: Environment Variables**

```bash
export TRAINING_DATA_PATH="data/my_training.txt"
export OUTPUT_DIR="trained_models"
export BATCH_SIZE="16"
export LEARNING_RATE="0.001"
export EPOCHS="5"
export MEMORY_SLOTS="2000"
```

**Method 2: Custom Script**

```typescript
// custom_train.ts
import { HopeTrainer, TrainingConfig } from '@henryhawke/mcp-titan/training';

const config: TrainingConfig = {
  dataPath: 'data/my_training.txt',
  outputDir: 'my_model',
  batchSize: 32,
  learningRate: 0.0005,
  epochs: 10,
  validationSplit: 0.1,
  sequenceLength: 512,
  vocabSize: 16000,
  embeddingDim: 256,
  modelConfig: {
    inputDim: 256,
    hiddenDim: 512,
    memoryDim: 768,
    transformerLayers: 6,
    memorySlots: 5000,
    learningRate: 0.0005
  }
};

const trainer = new HopeTrainer(config);
await trainer.train();
```

### Step 3: Train the Model

#### Quick Training (For Testing)

```bash
# 3 epochs, smaller model, ~15 minutes on CPU
bun run train-quick

# With custom data
TRAINING_DATA_PATH=data/my_data.txt bun run train-quick
```

**What happens:**
1. Loads training data
2. Trains BPE tokenizer on your data (learns vocabulary)
3. Initializes HOPE model with smaller config
4. Trains for 3 epochs
5. Saves checkpoint and final model to `trained_models/`

#### Production Training (Full Model)

```bash
# 10 epochs, full model, ~2-4 hours on CPU
bun run train-production

# With custom parameters
TRAINING_DATA_PATH=data/large_dataset.txt \
EPOCHS=20 \
BATCH_SIZE=64 \
TRANSFORMER_LAYERS=8 \
MEMORY_SLOTS=10000 \
bun run train-production
```

#### Full Control Training

```bash
# Custom configuration for every parameter
TRAINING_DATA_PATH=data/my_data.txt \
OUTPUT_DIR=my_custom_model \
BATCH_SIZE=32 \
LEARNING_RATE=0.0005 \
EPOCHS=15 \
VALIDATION_SPLIT=0.15 \
SEQUENCE_LENGTH=512 \
VOCAB_SIZE=32000 \
EMBEDDING_DIM=512 \
HIDDEN_DIM=1024 \
MEMORY_DIM=1536 \
TRANSFORMER_LAYERS=8 \
MEMORY_SLOTS=8000 \
bun run train-model
```

### Step 4: Monitor Training

During training, you'll see output like:

```
🚀 Starting HOPE Memory Model training pipeline...

📥 Preparing training data...
📊 Training samples: 9000
📊 Validation samples: 1000

🔤 Training tokenizer...
📝 Training on 1500000 characters...
📈 Processed 100 samples for tokenizer training
💾 Tokenizer saved to trained_models/tokenizer

🧠 Initializing model...
✅ Model initialized

🏋️ Training model...

📅 Epoch 1/5
  🔄 Batch 0/563, Loss: 0.4523
  🔄 Batch 10/563, Loss: 0.3891
  ...
📊 Loss: 0.2341, Accuracy: 0.8123, Perplexity: 1.26
📊 Validation Loss: 0.2567, Validation Accuracy: 0.7891

📅 Epoch 2/5
  ...

💾 Checkpoint saved to trained_models/checkpoint_epoch_5

🎉 Training completed successfully in 45.23 minutes!
```

### Step 5: Use Your Trained Model

```bash
# Start MCP server with your trained model
MODEL_PATH=trained_models/final_model/model.json \
TOKENIZER_PATH=trained_models/final_model/tokenizer \
npx @henryhawke/mcp-titan
```

Or update your MCP client configuration:

```json
{
  "hope-memory": {
    "command": "npx",
    "args": ["-y", "@henryhawke/mcp-titan"],
    "env": {
      "MODEL_PATH": "/absolute/path/to/trained_models/final_model/model.json",
      "TOKENIZER_PATH": "/absolute/path/to/trained_models/final_model/tokenizer"
    }
  }
}
```

---

## Advanced Training Options

### Training on Code

```bash
# Prepare code dataset
find ./my_project -name "*.ts" -o -name "*.js" | \
  xargs cat > data/code_training.txt

# Train with code-specific config
TRAINING_DATA_PATH=data/code_training.txt \
VOCAB_SIZE=32000 \
SEQUENCE_LENGTH=1024 \
bun run train-production
```

### Training on Documentation

```bash
# Combine markdown documentation
find ./docs -name "*.md" | xargs cat > data/docs_training.txt

# Train to learn your project's documentation style
TRAINING_DATA_PATH=data/docs_training.txt \
EPOCHS=10 \
bun run train-model
```

### Multi-Domain Training

```typescript
// combined_training.ts
import { HopeTrainer } from '@henryhawke/mcp-titan/training';
import * as fs from 'fs/promises';

// Combine multiple data sources
const codeData = await fs.readFile('data/code.txt', 'utf-8');
const docsData = await fs.readFile('data/docs.txt', 'utf-8');
const conversationData = await fs.readFile('data/conversations.txt', 'utf-8');

const combined = [
  ...codeData.split('\n'),
  ...docsData.split('\n'),
  ...conversationData.split('\n')
];

// Shuffle
for (let i = combined.length - 1; i > 0; i--) {
  const j = Math.floor(Math.random() * (i + 1));
  [combined[i], combined[j]] = [combined[j], combined[i]];
}

await fs.writeFile('data/combined.txt', combined.join('\n'));

// Now train on combined data
const trainer = new HopeTrainer({
  dataPath: 'data/combined.txt',
  outputDir: 'multi_domain_model',
  epochs: 20,
  batchSize: 32
});

await trainer.train();
```

### Resume Training from Checkpoint

```bash
# If training was interrupted, you can resume
MODEL_PATH=trained_models/checkpoint_epoch_5/model.json \
TRAINING_DATA_PATH=data/my_data.txt \
EPOCHS=10 \  # Will train 5 more epochs
bun run train-model
```

### Training with Custom Loss Functions

```typescript
// advanced_training.ts
import { HopeMemoryModel } from '@henryhawke/mcp-titan';
import * as tf from '@tensorflow/tfjs-node';

const model = new HopeMemoryModel();
await model.initialize({
  inputDim: 256,
  hiddenDim: 512,
  memoryDim: 768,
  learningRate: 0.001
});

// Custom training loop
for (let epoch = 0; epoch < 10; epoch++) {
  for (const batch of batches) {
    const result = model.trainStep(
      batch.input,
      batch.target,
      model.getMemoryState()
    );

    // Custom loss weighting
    const customLoss = tf.tidy(() => {
      const baseLoss = result.loss;
      const surprisePenalty = result.memoryUpdate.surprise.totalSurprise.mul(0.1);
      return baseLoss.add(surprisePenalty);
    });

    // Apply gradients with custom loss
    // ... (advanced usage)
  }
}
```

---

## Using the MCP Server

### Available Tools

HOPE exposes 19 tools through MCP. Here's how to use each one:

### 1. Model Initialization

#### `init_model` - Initialize or reconfigure the model

```json
{
  "name": "init_model",
  "arguments": {
    "config": {
      "inputDim": 256,
      "hiddenDim": 512,
      "memoryDim": 768,
      "shortTermSlots": 64,
      "longTermSlots": 256,
      "archiveSlots": 512,
      "learningRate": 0.001,
      "enableMomentum": true,
      "enableTokenFlow": true,
      "enableForgettingGate": true
    }
  }
}
```

**Use case:** First-time setup or changing model configuration

### 2. Memory Operations

#### `bootstrap_memory` - Quick-load context from text or URL

```json
{
  "name": "bootstrap_memory",
  "arguments": {
    "source": "text",
    "content": "I'm working on a Python web application using FastAPI and PostgreSQL. The project uses Docker for containerization and has a React frontend."
  }
}
```

```json
{
  "name": "bootstrap_memory",
  "arguments": {
    "source": "url",
    "url": "https://github.com/myusername/myproject/blob/main/README.md"
  }
}
```

**Use case:** Provide initial context to HOPE before starting work

#### `get_memory_state` - Inspect current memory contents

```json
{
  "name": "get_memory_state",
  "arguments": {}
}
```

**Returns:**
```json
{
  "shortTerm": {
    "size": 12,
    "capacity": 64,
    "averageSurprise": 0.45
  },
  "longTerm": {
    "size": 87,
    "capacity": 256,
    "averageSurprise": 0.23
  },
  "archive": {
    "size": 203,
    "capacity": 512,
    "averageSurprise": 0.12
  },
  "totalMemories": 302,
  "momentumActive": true,
  "tokenFlowActive": true
}
```

#### `memory_stats` - Get statistical summary

```json
{
  "name": "memory_stats",
  "arguments": {}
}
```

**Returns:** Detailed metrics about memory usage, access patterns, and performance

#### `prune_memory` - Clean up low-value memories

```json
{
  "name": "prune_memory",
  "arguments": {
    "threshold": 0.1
  }
}
```

**Use case:** When memory is full or performance degrades. Removes memories with information gain below threshold.

### 3. Learning Operations

#### `forward_pass` - Query memory and get prediction

```json
{
  "name": "forward_pass",
  "arguments": {
    "input": "def calculate_fibonacci(n):"
  }
}
```

**Returns:**
```json
{
  "prediction": "    if n <= 1:\n        return n",
  "surprise": 0.23,
  "memoryState": {...}
}
```

#### `train_step` - Explicit learning from example pair

```json
{
  "name": "train_step",
  "arguments": {
    "x_t": "User: How do I install dependencies?",
    "x_next": "Assistant: Run `npm install` in the project directory."
  }
}
```

**Use case:** Teach HOPE specific patterns or corrections

#### `reset_gradients` - Clear training state

```json
{
  "name": "reset_gradients",
  "arguments": {}
}
```

**Use case:** If training becomes unstable or you want to start fresh

### 4. Persistence

#### `save_checkpoint` - Save memory to file

```json
{
  "name": "save_checkpoint",
  "arguments": {
    "path": "/path/to/checkpoints/session_2024-11-16.json"
  }
}
```

#### `load_checkpoint` - Restore from checkpoint

```json
{
  "name": "load_checkpoint",
  "arguments": {
    "path": "/path/to/checkpoints/session_2024-11-16.json"
  }
}
```

**Use case:** Save session state and restore later

### 5. Online Learning Service

#### `init_learner` - Start background learning

```json
{
  "name": "init_learner",
  "arguments": {
    "config": {
      "bufferSize": 1000,
      "batchSize": 32,
      "learningRate": 0.0001
    }
  }
}
```

#### `add_training_sample` - Feed training data to learner

```json
{
  "name": "add_training_sample",
  "arguments": {
    "data": "function add(a, b) { return a + b; }"
  }
}
```

#### `pause_learner` / `resume_learner` - Control learning

```json
{
  "name": "pause_learner",
  "arguments": {}
}
```

#### `get_learner_stats` - Monitor learning progress

```json
{
  "name": "get_learner_stats",
  "arguments": {}
}
```

**Returns:**
```json
{
  "bufferSize": 347,
  "totalSamplesSeen": 1523,
  "averageLoss": 0.234,
  "learningRate": 0.0001,
  "status": "active"
}
```

### 6. Advanced Metrics

#### `get_token_flow_metrics` - Analyze sequence patterns

```json
{
  "name": "get_token_flow_metrics",
  "arguments": {}
}
```

**Returns:**
```json
{
  "historySize": 32,
  "averageWeight": 0.67,
  "flowStrength": 0.45,
  "sequencePatternsLearned": 127
}
```

#### `get_hierarchical_metrics` - Memory tier distribution

```json
{
  "name": "get_hierarchical_metrics",
  "arguments": {}
}
```

**Returns tier utilization, promotion rates, and access patterns**

#### `health_check` - System health & performance

```json
{
  "name": "health_check",
  "arguments": {}
}
```

**Returns:**
```json
{
  "status": "healthy",
  "memoryUsage": "487 MB",
  "tensorCount": 142,
  "averageLatency": "42ms",
  "uptime": "3h 24m"
}
```

### 7. Utility

#### `help` - List all available tools

```json
{
  "name": "help",
  "arguments": {}
}
```

---

## Development Setup

### For Contributors

```bash
# 1. Fork and clone
git fork https://github.com/henryhawke/mcp-titan.git
cd mcp-titan

# 2. Install dependencies
bun install

# 3. Build in watch mode
bun run dev

# 4. Run tests
bun test

# 5. Run tests with coverage
bun test --coverage

# 6. Type checking
bun run typecheck

# 7. Linting
bun run lint

# 8. Fix lint issues
bun run lint:fix

# 9. Format code
bun run format
```

### Project Structure

```
mcp-titan/
├── src/
│   ├── hope_model/          # Core HOPE architecture
│   │   ├── index.ts         # Main HopeMemoryModel
│   │   ├── continuum_memory.ts  # 3-tier memory system
│   │   ├── retention_core.ts    # Sequence processing
│   │   ├── memory_router.ts     # Surprise-based routing
│   │   ├── mamba_filters.ts     # Selective state-space
│   │   └── optimizer_hooks.ts   # Training optimizations
│   ├── tokenizer/           # BPE tokenization
│   ├── training/            # Training pipeline
│   │   └── trainer.ts       # HopeTrainer class
│   ├── index.ts             # MCP server entry point
│   └── types.ts             # TypeScript interfaces
├── scripts/
│   ├── train-model.ts       # Training script
│   └── download-data.ts     # Data preparation
├── docs/                    # Documentation
├── test/                    # Test suites
└── dist/                    # Compiled output
```

### Running Tests

```bash
# All tests
bun test

# Specific test file
bun test src/__tests__/hope_model.test.ts

# Tests matching pattern
bun test --grep "momentum"

# Watch mode
bun test --watch

# With debugging
DEBUG=* bun test
```

### Debugging

```bash
# Debug MCP server
DEBUG=mcp:* npx @henryhawke/mcp-titan

# Debug TensorFlow operations
TF_CPP_MIN_LOG_LEVEL=0 npx @henryhawke/mcp-titan

# Memory leak detection
node --trace-gc dist/index.js

# Profiling
node --prof dist/index.js
```

---

## Performance Tuning

### Memory Optimization

#### Reduce Memory Footprint

```json
{
  "name": "init_model",
  "arguments": {
    "config": {
      "shortTermSlots": 32,    // Default: 64
      "longTermSlots": 128,    // Default: 256
      "archiveSlots": 256,     // Default: 512
      "memoryDim": 192,        // Default: 256
      "hiddenDim": 384         // Default: 512
    }
  }
}
```

**Impact:** ~50% memory reduction, slight accuracy loss

#### Aggressive Pruning

```json
{
  "name": "prune_memory",
  "arguments": {
    "threshold": 0.2  // Higher = more aggressive (default: 0.1)
  }
}
```

Set up auto-pruning via environment variable:

```bash
AUTO_PRUNE_THRESHOLD=0.15 npx @henryhawke/mcp-titan
```

### Speed Optimization

#### Reduce Computation

```json
{
  "config": {
    "transformerLayers": 4,  // Default: 6
    "memorySlots": 1000,     // Default: 2000
    "dropoutRate": 0         // Disable dropout in inference
  }
}
```

#### Batch Processing

```typescript
// Instead of:
for (const item of items) {
  await model.forward(item, state);
}

// Do:
const batch = tf.stack(items.map(i => tf.tensor2d([i])));
const result = model.forward(batch, state);
```

#### Disable Features for Speed

```bash
# Disable token flow tracking (5-10% speedup)
ENABLE_TOKEN_FLOW=false npx @henryhawke/mcp-titan

# Disable momentum (10-15% speedup, affects learning)
ENABLE_MOMENTUM=false npx @henryhawke/mcp-titan
```

### Training Performance

#### Faster Training

```bash
# Smaller batches for faster iteration
BATCH_SIZE=8 bun run train-model

# Reduce sequence length
SEQUENCE_LENGTH=128 bun run train-model

# Fewer validation checks
VALIDATION_SPLIT=0.05 bun run train-model
```

#### Better Convergence

```bash
# Larger batches for stability
BATCH_SIZE=64 bun run train-model

# Lower learning rate
LEARNING_RATE=0.0001 bun run train-model

# More epochs
EPOCHS=20 bun run train-model
```

### GPU Acceleration (Optional)

```bash
# Install CUDA backend (Linux/Windows with NVIDIA GPU)
npm install @tensorflow/tfjs-node-gpu

# Use GPU version
TF_FORCE_GPU_ALLOW_GROWTH=true \
CUDA_VISIBLE_DEVICES=0 \
npx @henryhawke/mcp-titan
```

**Expected speedup:** 3-10x for training, 2-5x for inference

---

## Troubleshooting

### Common Issues

#### 1. "Module not found" Error

**Symptom:**
```
Error: Cannot find module '@tensorflow/tfjs-node'
```

**Solution:**
```bash
# Ensure Node.js 22+
node --version

# Clean install
rm -rf node_modules package-lock.json
npm install

# Or rebuild native modules
npm rebuild @tensorflow/tfjs-node
```

#### 2. Memory Leaks During Long Sessions

**Symptom:** Process memory grows over time, eventually crashes

**Solution:**
```typescript
// Manually cleanup periodically
{
  "name": "reset_gradients",
  "arguments": {}
}

{
  "name": "prune_memory",
  "arguments": {"threshold": 0.1}
}
```

Or set auto-cleanup:
```bash
AUTO_CLEANUP_INTERVAL=3600 npx @henryhawke/mcp-titan  # Every hour
```

#### 3. Training Loss Not Decreasing

**Symptoms:**
```
Epoch 1: Loss: 2.4513
Epoch 2: Loss: 2.4489
Epoch 3: Loss: 2.4512  # Not improving
```

**Solutions:**

**A. Lower learning rate:**
```bash
LEARNING_RATE=0.0001 bun run train-model
```

**B. Check data quality:**
```bash
# Ensure data has variation
head -100 data/training.txt | sort | uniq | wc -l
# Should be close to 100
```

**C. Increase model capacity:**
```bash
TRANSFORMER_LAYERS=8 MEMORY_SLOTS=5000 bun run train-model
```

**D. Reset and retry:**
```bash
rm -rf trained_models/*
bun run train-model
```

#### 4. "Out of Memory" During Training

**Symptom:**
```
FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory
```

**Solutions:**

**A. Increase Node.js heap:**
```bash
NODE_OPTIONS="--max-old-space-size=8192" bun run train-model
```

**B. Reduce batch size:**
```bash
BATCH_SIZE=8 bun run train-model
```

**C. Reduce model size:**
```bash
MEMORY_SLOTS=1000 TRANSFORMER_LAYERS=4 bun run train-model
```

**D. Enable garbage collection:**
```bash
NODE_OPTIONS="--max-old-space-size=4096 --expose-gc" \
bun run train-model
```

#### 5. Slow Forward Pass (>500ms)

**Symptom:** MCP tool calls take too long

**Diagnosis:**
```json
{
  "name": "health_check",
  "arguments": {}
}
// Check "averageLatency" field
```

**Solutions:**

**A. Reduce memory size:**
```json
{
  "name": "prune_memory",
  "arguments": {"threshold": 0.15}
}
```

**B. Use smaller model:**
```json
{
  "name": "init_model",
  "arguments": {
    "config": {
      "transformerLayers": 4,
      "memorySlots": 1000
    }
  }
}
```

**C. Disable features:**
```bash
ENABLE_TOKEN_FLOW=false \
ENABLE_MOMENTUM=false \
npx @henryhawke/mcp-titan
```

#### 6. MCP Client Not Connecting

**Symptom:** Tools don't appear in Cursor/Claude

**Diagnosis:**
```bash
# Test server manually
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | \
  npx @henryhawke/mcp-titan
```

**Solutions:**

**A. Check config file location:**
- Cursor: `~/.cursor/mcp_settings.json`
- Claude: `~/Library/Application Support/Claude/claude_desktop_config.json`

**B. Verify JSON syntax:**
```bash
# Validate JSON
cat ~/.cursor/mcp_settings.json | python -m json.tool
```

**C. Check permissions:**
```bash
chmod +x $(which npx)
npx @henryhawke/mcp-titan --version
```

**D. Use absolute paths:**
```json
{
  "command": "/usr/local/bin/node",
  "args": [
    "/usr/local/bin/npx",
    "-y",
    "@henryhawke/mcp-titan"
  ]
}
```

**E. Check logs:**
- Cursor: `~/.cursor/logs/`
- Claude: `~/Library/Logs/Claude/`

#### 7. TypeScript Compilation Errors

**Symptom:**
```
error TS2345: Argument of type 'Tensor' is not assignable to parameter of type 'Tensor2D'
```

**Status:** As of v3.1, all TypeScript errors are fixed!

**Verification:**
```bash
cd mcp-titan
bun run typecheck
# Should output nothing (0 errors)
```

If you still see errors:
```bash
# Pull latest version
git pull origin main
bun install
bun run build
```

---

## Advanced Configuration

### Environment Variables Reference

```bash
# Model Configuration
MODEL_PATH=/path/to/model.json
TOKENIZER_PATH=/path/to/tokenizer
MEMORY_DIR=/path/to/memory/storage

# Memory Capacity
SHORT_TERM_SLOTS=64
LONG_TERM_SLOTS=256
ARCHIVE_SLOTS=512

# HOPE Features
ENABLE_MOMENTUM=true
ENABLE_TOKEN_FLOW=true
ENABLE_FORGETTING_GATE=true

# Performance
AUTO_PRUNE_THRESHOLD=0.1
AUTO_CLEANUP_INTERVAL=3600
MAX_SEQUENCE_LENGTH=512

# Training
BATCH_SIZE=32
LEARNING_RATE=0.001
EPOCHS=10
VALIDATION_SPLIT=0.1

# Logging
DEBUG=mcp:*
LOG_LEVEL=info
LOG_FILE=/path/to/hope.log

# Node.js
NODE_OPTIONS="--max-old-space-size=8192"
```

### Programmatic Usage

#### Using HOPE as a Library

```typescript
import { HopeMemoryModel } from '@henryhawke/mcp-titan';

// Initialize
const model = new HopeMemoryModel();
await model.initialize({
  inputDim: 256,
  hiddenDim: 512,
  memoryDim: 768,
  enableMomentum: true,
  enableTokenFlow: true,
  enableForgettingGate: true
});

// Store memory
await model.storeMemory("I'm working on a React project");

// Query
const result = model.forward(
  await model.encodeText("How do I add a component?"),
  model.getMemoryState()
);

console.log('Prediction:', result.predicted);
console.log('Surprise:', result.memoryUpdate.surprise);

// Save
await model.saveModel('./my_model');
```

#### Custom Training Loop

```typescript
import { HopeMemoryModel } from '@henryhawke/mcp-titan';
import * as tf from '@tensorflow/tfjs-node';

const model = new HopeMemoryModel();
await model.initialize({...});

const trainingData = [
  ["input1", "output1"],
  ["input2", "output2"],
  // ...
];

for (let epoch = 0; epoch < 10; epoch++) {
  let totalLoss = 0;

  for (const [input, target] of trainingData) {
    const inputTensor = await model.encodeText(input);
    const targetTensor = await model.encodeText(target);

    const result = model.trainStep(
      inputTensor,
      targetTensor,
      model.getMemoryState()
    );

    totalLoss += result.loss.dataSync()[0];
  }

  console.log(`Epoch ${epoch}: Loss = ${totalLoss / trainingData.length}`);
}

await model.saveModel('./trained_model');
```

#### Integration with Express API

```typescript
import express from 'express';
import { HopeMemoryModel } from '@henryhawke/mcp-titan';

const app = express();
app.use(express.json());

const model = new HopeMemoryModel();
await model.initialize({...});

app.post('/api/store', async (req, res) => {
  await model.storeMemory(req.body.text);
  res.json({ status: 'stored' });
});

app.post('/api/query', async (req, res) => {
  const input = await model.encodeText(req.body.query);
  const result = model.forward(input, model.getMemoryState());
  res.json({
    prediction: result.predicted.arraySync(),
    surprise: result.memoryUpdate.surprise
  });
});

app.listen(3000, () => {
  console.log('HOPE API running on port 3000');
});
```

---

## Next Steps

### After Setup

1. **Test basic functionality:**
   ```bash
   # Initialize and test
   echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"init_model","arguments":{}}}' | \
     npx @henryhawke/mcp-titan
   ```

2. **Try in your MCP client:**
   - Open Cursor/Claude with HOPE configured
   - Try: "Store in memory: I'm working on a Python project"
   - Try: "What am I working on?"

3. **Train on your data:**
   ```bash
   # Prepare your data
   cat > data/my_data.txt << EOF
   Your training data here...
   EOF

   # Train
   TRAINING_DATA_PATH=data/my_data.txt bun run train-quick
   ```

4. **Explore the API:**
   - Read [docs/api/README.md](docs/api/README.md)
   - Try all 19 MCP tools
   - Experiment with different configurations

5. **Join the community:**
   - Star the repo: https://github.com/henryhawke/mcp-titan
   - Open issues for bugs/features
   - Contribute improvements

---

## Additional Resources

- **Documentation:** [docs/](docs/)
- **API Reference:** [docs/api/README.md](docs/api/README.md)
- **Architecture:** [docs/architecture-hope.md](docs/architecture-hope.md)
- **Research Paper:** [HOPE.md](HOPE.md)
- **Implementation Plan:** [PLAN.md](PLAN.md)
- **Changelog:** [docs/changelog.md](docs/changelog.md)

---

## Getting Help

### Issues & Bugs

Open an issue: https://github.com/henryhawke/mcp-titan/issues

Include:
- System info (OS, Node version)
- Error messages
- Steps to reproduce
- What you expected vs. what happened

### Discussions

Ask questions: https://github.com/henryhawke/mcp-titan/discussions

### Community

- GitHub Discussions for Q&A
- Issues for bug reports
- Pull Requests for contributions

---

**🎉 You're all set! Enjoy using HOPE Memory!**

*Transform your AI from a forgetful assistant to a learning companion.*
