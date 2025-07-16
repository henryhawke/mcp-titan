#!/usr/bin/env node

import * as path from 'path';
import { TitanTrainer, TrainingConfig } from '../src/training/trainer.js';

async function main() {
  console.log('🚀 Starting MCP Titan Memory Model Training');
  console.log('==========================================\n');

  // Configuration for training
  const config: TrainingConfig = {
    dataPath: process.env.TRAINING_DATA_PATH || 'data/training.txt',
    outputDir: process.env.OUTPUT_DIR || 'trained_models',
    batchSize: parseInt(process.env.BATCH_SIZE || '16'),
    learningRate: parseFloat(process.env.LEARNING_RATE || '0.001'),
    epochs: parseInt(process.env.EPOCHS || '5'),
    validationSplit: parseFloat(process.env.VALIDATION_SPLIT || '0.1'),
    sequenceLength: parseInt(process.env.SEQUENCE_LENGTH || '256'),
    vocabSize: parseInt(process.env.VOCAB_SIZE || '16000'),
    embeddingDim: parseInt(process.env.EMBEDDING_DIM || '256'),
    modelConfig: {
      inputDim: parseInt(process.env.EMBEDDING_DIM || '256'),
      hiddenDim: parseInt(process.env.HIDDEN_DIM || '512'),
      memoryDim: parseInt(process.env.MEMORY_DIM || '768'),
      transformerLayers: parseInt(process.env.TRANSFORMER_LAYERS || '4'),
      memorySlots: parseInt(process.env.MEMORY_SLOTS || '2000'),
      learningRate: parseFloat(process.env.LEARNING_RATE || '0.001')
    }
  };

  console.log('📋 Training Configuration:');
  console.log('  Data Path:', config.dataPath);
  console.log('  Output Directory:', config.outputDir);
  console.log('  Batch Size:', config.batchSize);
  console.log('  Learning Rate:', config.learningRate);
  console.log('  Epochs:', config.epochs);
  console.log('  Vocabulary Size:', config.vocabSize);
  console.log('  Embedding Dimension:', config.embeddingDim);
  console.log('  Transformer Layers:', config.modelConfig.transformerLayers);
  console.log('  Memory Slots:', config.modelConfig.memorySlots);
  console.log('');

  try {
    // Initialize trainer
    const trainer = new TitanTrainer(config);

    // Start training
    const startTime = Date.now();
    await trainer.train();
    const endTime = Date.now();

    const trainingTimeMinutes = (endTime - startTime) / 1000 / 60;
    console.log(`\n🎉 Training completed successfully in ${trainingTimeMinutes.toFixed(2)} minutes!`);
    console.log(`📁 Model saved to: ${path.resolve(config.outputDir)}`);
    
    console.log('\n📖 Next steps:');
    console.log('1. Test the model with: npm run test-model');
    console.log('2. Start the MCP server with the trained model');
    console.log('3. Add the server to your Cursor configuration');

  } catch (error) {
    console.error('\n❌ Training failed:', error);
    process.exit(1);
  }
}

// Handle command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  
  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
MCP Titan Memory Model Training Script

Usage: npm run train-model [options]

Environment Variables:
  TRAINING_DATA_PATH    Path to training data file (default: data/training.txt)
  OUTPUT_DIR           Output directory for trained model (default: trained_models)
  BATCH_SIZE           Training batch size (default: 16)
  LEARNING_RATE        Learning rate (default: 0.001)
  EPOCHS               Number of training epochs (default: 5)
  VALIDATION_SPLIT     Validation data split ratio (default: 0.1)
  SEQUENCE_LENGTH      Maximum sequence length (default: 256)
  VOCAB_SIZE           Vocabulary size (default: 16000)
  EMBEDDING_DIM        Embedding dimension (default: 256)
  HIDDEN_DIM           Hidden dimension (default: 512)
  MEMORY_DIM           Memory dimension (default: 768)
  TRANSFORMER_LAYERS   Number of transformer layers (default: 4)
  MEMORY_SLOTS         Number of memory slots (default: 2000)

Examples:
  # Quick training with small model
  EPOCHS=3 TRANSFORMER_LAYERS=2 MEMORY_SLOTS=1000 npm run train-model
  
  # Production training with larger model  
  EPOCHS=10 TRANSFORMER_LAYERS=6 MEMORY_SLOTS=5000 BATCH_SIZE=32 npm run train-model
  
  # Training with custom data
  TRAINING_DATA_PATH=./my_data.txt OUTPUT_DIR=./my_model npm run train-model
`);
    process.exit(0);
  }

  if (args.includes('--quick')) {
    // Set quick training defaults
    process.env.EPOCHS = process.env.EPOCHS || '3';
    process.env.TRANSFORMER_LAYERS = process.env.TRANSFORMER_LAYERS || '2';
    process.env.MEMORY_SLOTS = process.env.MEMORY_SLOTS || '1000';
    process.env.BATCH_SIZE = process.env.BATCH_SIZE || '8';
    console.log('🚀 Quick training mode enabled');
  }

  if (args.includes('--production')) {
    // Set production training defaults
    process.env.EPOCHS = process.env.EPOCHS || '10';
    process.env.TRANSFORMER_LAYERS = process.env.TRANSFORMER_LAYERS || '6';
    process.env.MEMORY_SLOTS = process.env.MEMORY_SLOTS || '5000';
    process.env.BATCH_SIZE = process.env.BATCH_SIZE || '32';
    process.env.LEARNING_RATE = process.env.LEARNING_RATE || '0.0005';
    console.log('🏭 Production training mode enabled');
  }
}

// Parse command line arguments
parseArgs();

// Run main function
main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});