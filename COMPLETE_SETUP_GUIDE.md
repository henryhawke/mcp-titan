# MCP Titan Memory Server - Complete Setup Guide

## 🎯 Overview

This guide will take you from zero to a fully trained, production-ready MCP Titan Memory Server. The server provides neural memory capabilities to LLMs through the Model Context Protocol (MCP).

## 🚀 Quick Start (5 minutes)

For immediate testing with a basic model:

```bash
# 1. Clone and setup
git clone <repository-url>
cd mcp-titan
npm install

# 2. Generate training data and train a basic model
npm run download-data --synthetic
npm run train-quick

# 3. Test the model
npm run test-model

# 4. Start the MCP server
npm start
```

## 📋 Prerequisites

### Required
- **Node.js**: 22.0.0 or higher
- **NPM**: Latest version
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space

### For GPU Training (Recommended)
- **NVIDIA GPU**: RTX 3060 or better
- **VRAM**: 8GB minimum for training
- **CUDA**: Compatible with TensorFlow.js

### For CPU-only Training
- **CPU**: Multi-core processor recommended
- **RAM**: 16GB minimum
- Training will be significantly slower

## 🔧 Installation & Setup

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd mcp-titan

# Install dependencies
npm install

# Build project
npm run build

# Verify installation
npm test
```

### 2. Check System Requirements

```bash
# Check Node.js version
node --version  # Should be 22.0.0+

# Check available memory
free -h  # Linux
top     # Check available RAM

# Check GPU (if available)
nvidia-smi  # Should show GPU info
```

## 📊 Training the Model

### Option 1: Quick Training (Recommended for Testing)

Perfect for testing and development:

```bash
# Generate synthetic training data
npm run download-data --synthetic

# Train a small model (3 epochs, 2 transformer layers)
npm run train-quick

# Test the trained model
npm run test-model
```

**Time**: 10-30 minutes  
**Quality**: Basic functionality, good for testing  
**Use case**: Development, testing, proof of concept

### Option 2: Production Training

For production use with better quality:

```bash
# Download real training data
npm run download-data --wikitext

# Train production model (10 epochs, 6 transformer layers)
npm run train-production

# Test the trained model
npm run test-model
```

**Time**: 2-8 hours (depending on hardware)  
**Quality**: Production-ready  
**Use case**: Real-world deployment

### Option 3: Custom Training

For advanced users who want full control:

```bash
# Download specific datasets
npm run download-data --tinystories  # 2.1GB dataset
npm run download-data --openwebtext   # Large Reddit dataset

# Set custom training parameters
export TRAINING_DATA_PATH=data/tinystories.txt
export EPOCHS=15
export TRANSFORMER_LAYERS=8
export MEMORY_SLOTS=10000
export BATCH_SIZE=64
export LEARNING_RATE=0.0003

# Train with custom settings
npm run train-model

# Test the model
npm run test-model
```

## 🎛️ Training Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAINING_DATA_PATH` | `data/synthetic_training.txt` | Path to training data |
| `OUTPUT_DIR` | `trained_models` | Output directory for model |
| `BATCH_SIZE` | `16` | Training batch size |
| `LEARNING_RATE` | `0.001` | Learning rate |
| `EPOCHS` | `5` | Number of training epochs |
| `TRANSFORMER_LAYERS` | `4` | Number of transformer layers |
| `MEMORY_SLOTS` | `2000` | Number of memory slots |
| `EMBEDDING_DIM` | `256` | Embedding dimension |
| `HIDDEN_DIM` | `512` | Hidden layer dimension |
| `MEMORY_DIM` | `768` | Memory dimension |

### Model Size Presets

#### Small Model (Testing)
```bash
TRANSFORMER_LAYERS=2 MEMORY_SLOTS=1000 EMBEDDING_DIM=128 npm run train-model
```
- **Parameters**: ~50M
- **Training time**: 15-30 minutes
- **VRAM needed**: 2GB
- **Use case**: Testing, development

#### Medium Model (Development)
```bash
TRANSFORMER_LAYERS=4 MEMORY_SLOTS=2000 EMBEDDING_DIM=256 npm run train-model
```
- **Parameters**: ~125M
- **Training time**: 1-2 hours
- **VRAM needed**: 4GB
- **Use case**: Development, small deployments

#### Large Model (Production)
```bash
TRANSFORMER_LAYERS=6 MEMORY_SLOTS=5000 EMBEDDING_DIM=512 npm run train-model
```
- **Parameters**: ~350M
- **Training time**: 4-8 hours
- **VRAM needed**: 8GB+
- **Use case**: Production deployments

## 📁 Training Data Options

### 1. Synthetic Data (Default)
- **Size**: ~5MB for 10,000 samples
- **Quality**: Basic, good for testing
- **Time to download**: Instant (generated locally)
- **Command**: `npm run download-data --synthetic`

### 2. WikiText-2
- **Size**: ~12.7MB
- **Quality**: High-quality Wikipedia text
- **Time to download**: 1-2 minutes
- **Command**: `npm run download-data --wikitext`

### 3. TinyStories
- **Size**: ~2.1GB
- **Quality**: Synthetic stories, good for language modeling
- **Time to download**: 10-30 minutes
- **Command**: `npm run download-data --tinystories`

### 4. OpenWebText Sample
- **Size**: ~1.2GB
- **Quality**: Real-world text from Reddit
- **Time to download**: 5-20 minutes
- **Command**: `npm run download-data --openwebtext`

### 5. Custom Data
```bash
# Use your own training data
export TRAINING_DATA_PATH=/path/to/your/data.txt
npm run train-model
```

**Format**: Plain text file with one document per line, or JSON array of strings.

## 🧪 Testing & Validation

### Basic Functionality Test
```bash
npm run test-model
```

This tests:
- ✅ Model loading/saving
- ✅ Tokenizer functionality  
- ✅ Memory operations
- ✅ Forward pass
- ✅ Training step
- ✅ MCP integration

### MCP Server Test
```bash
# Start the server
npm start

# In another terminal, test MCP tools
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"help","arguments":{}}}' | npm start
```

### Integration Test with Cursor

1. **Start the MCP server**:
```bash
npm start
```

2. **Add to Cursor configuration** (`cursor-mcp-config.json`):
```json
{
  "mcpServers": {
    "titan-memory": {
      "command": "node",
      "args": ["index.js"],
      "cwd": "/path/to/mcp-titan"
    }
  }
}
```

3. **Test in Cursor**:
   - Open Cursor
   - Try MCP commands like "help", "init_model", "get_memory_state"

## 🏭 Production Deployment

### 1. Full Production Training
```bash
# Download high-quality training data
npm run download-data --tinystories

# Train production model
export TRAINING_DATA_PATH=data/tinystories.txt
export EPOCHS=15
export TRANSFORMER_LAYERS=6
export MEMORY_SLOTS=5000
export BATCH_SIZE=32
npm run train-model

# Validate the model
npm run test-model
```

### 2. Performance Optimization

#### Memory Usage
```bash
# For limited memory environments
export MEMORY_SLOTS=1000
export BATCH_SIZE=8
export TRANSFORMER_LAYERS=4
```

#### Speed Optimization
```bash
# For faster training
export BATCH_SIZE=64        # If you have enough VRAM
export SEQUENCE_LENGTH=128  # Shorter sequences
```

#### Quality Optimization
```bash
# For best quality (slow training)
export EPOCHS=20
export LEARNING_RATE=0.0001
export TRANSFORMER_LAYERS=8
export MEMORY_SLOTS=10000
```

### 3. Production Server Setup

```bash
# Build for production
npm run build

# Start with process manager (pm2)
npm install -g pm2
pm2 start index.js --name "titan-memory-mcp"

# Or use systemd service
sudo cp titan-memory.service /etc/systemd/system/
sudo systemctl enable titan-memory
sudo systemctl start titan-memory
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "Out of Memory" during training
```bash
# Reduce batch size and memory slots
export BATCH_SIZE=4
export MEMORY_SLOTS=500
npm run train-model
```

#### 2. "TensorFlow not found"
```bash
# Reinstall TensorFlow.js
npm uninstall @tensorflow/tfjs-node
npm install @tensorflow/tfjs-node
```

#### 3. "Training very slow"
- Use GPU if available
- Reduce model size (fewer transformer layers)
- Use smaller dataset
- Increase batch size (if memory allows)

#### 4. "Model performance poor"
- Train for more epochs
- Use larger, higher-quality dataset
- Increase model size
- Lower learning rate

### Performance Benchmarks

| Model Size | Training Time (RTX 4090) | VRAM Usage | Quality |
|------------|--------------------------|------------|---------|
| Small (2 layers) | 30 minutes | 2GB | Basic |
| Medium (4 layers) | 2 hours | 4GB | Good |
| Large (6 layers) | 6 hours | 8GB | Excellent |
| XL (8 layers) | 12 hours | 12GB | Best |

### Monitoring Training

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor CPU and memory
htop

# Check training logs
tail -f trained_models/training_report.json
```

## 📚 Advanced Usage

### Custom Model Architecture
```bash
# Experimental: larger model with custom settings
export TRANSFORMER_LAYERS=12
export MEMORY_SLOTS=20000
export EMBEDDING_DIM=1024
export HIDDEN_DIM=2048
export MEMORY_DIM=1536
export EPOCHS=25
npm run train-model
```

### Multi-Stage Training
```bash
# Stage 1: Train on synthetic data
npm run download-data --synthetic
EPOCHS=5 npm run train-model

# Stage 2: Fine-tune on real data
npm run download-data --wikitext
TRAINING_DATA_PATH=data/wikitext-2.txt EPOCHS=10 npm run train-model
```

### Custom Training Script
```typescript
import { TitanTrainer } from './src/training/trainer.js';

const trainer = new TitanTrainer({
  dataPath: 'my_custom_data.txt',
  outputDir: 'my_model',
  epochs: 20,
  batchSize: 16,
  learningRate: 0.0005,
  modelConfig: {
    transformerLayers: 8,
    memorySlots: 8000
  }
});

await trainer.train();
```

## 🔄 Updating and Maintenance

### Model Updates
```bash
# Retrain with new data
npm run download-data --tinystories
npm run train-production

# Test updated model
npm run test-model

# Deploy updated model
pm2 restart titan-memory-mcp
```

### Performance Monitoring
```bash
# Check memory usage
npm run memory-stats

# Monitor training metrics
cat trained_models/training_report.json | jq '.metrics[-1]'
```

## 📞 Support

### Getting Help

1. **Check logs**: `cat ~/.titan-memory/logs/error.log`
2. **Run diagnostics**: `npm run test-model`
3. **Check GPU**: `nvidia-smi`
4. **Memory usage**: `free -h`

### Common Commands Quick Reference

```bash
# Quick setup and test
npm install && npm run download-data --synthetic && npm run train-quick && npm run test-model

# Production setup
npm install && npm run download-data --tinystories && npm run train-production && npm run test-model

# Start server
npm start

# Monitor training
tail -f trained_models/*/metrics.json

# Test MCP integration
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_memory_state","arguments":{}}}' | npm start
```

## 🎉 Success Criteria

Your MCP Titan Memory Server is production-ready when:

- ✅ `npm run test-model` passes all tests
- ✅ Training perplexity < 50 (lower is better)
- ✅ Memory recall accuracy > 80%
- ✅ MCP server responds to all tool calls
- ✅ Integration with Cursor works smoothly
- ✅ Model generates coherent text
- ✅ Memory persists across sessions

**Congratulations! Your MCP Titan Memory Server is now ready for production use! 🚀**