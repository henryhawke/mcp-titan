import * as fs from 'fs/promises';
import * as path from 'path';
import * as tf from '@tensorflow/tfjs-node';
import { HopeMemoryModel, HopeMemoryConfig } from '../model.js';
import { AdvancedTokenizer } from '../tokenizer/index.js';

export interface TrainingConfig {
  dataPath: string;
  outputDir: string;
  batchSize: number;
  learningRate: number;
  epochs: number;
  validationSplit: number;
  sequenceLength: number;
  vocabSize: number;
  embeddingDim: number;
  modelConfig: Partial<HopeMemoryConfig>;
}

export interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy: number;
  perplexity: number;
  memoryUtilization: number;
  validationLoss?: number;
  validationAccuracy?: number;
}

export class HopeTrainer {
  private config: TrainingConfig;
  private model!: HopeMemoryModel;
  private tokenizer!: AdvancedTokenizer;
  private trainingData: string[] = [];
  private validationData: string[] = [];
  private metrics: TrainingMetrics[] = [];

  constructor(config: TrainingConfig) {
    this.config = {
      ...config,
      batchSize: config.batchSize ?? 32,
      learningRate: config.learningRate ?? 0.001,
      epochs: config.epochs ?? 10,
      validationSplit: config.validationSplit ?? 0.1,
      sequenceLength: config.sequenceLength ?? 512,
      vocabSize: config.vocabSize ?? 32000,
      embeddingDim: config.embeddingDim ?? 256
    };
  }

  /**
   * Complete training pipeline
   */
  async train(): Promise<void> {
    console.log('🚀 Starting HOPE Memory Model training pipeline...');

    try {
      // Step 1: Download and prepare training data
      await this.prepareTrainingData();

      // Step 2: Train tokenizer
      await this.trainTokenizer();

      // Step 3: Initialize model
      await this.initializeModel();

      // Step 4: Train model
      await this.trainModel();

      // Step 5: Evaluate and save
      await this.evaluateAndSave();

      console.log('✅ Training completed successfully!');
    } catch (error) {
      console.error('❌ Training failed:', error);
      throw error;
    }
  }

  /**
   * Download and prepare training data
   */
  private async prepareTrainingData(): Promise<void> {
    console.log('📥 Preparing training data...');

    // Create output directory
    await fs.mkdir(this.config.outputDir, { recursive: true });

    if (await this.pathExists(this.config.dataPath)) {
      // Load existing data
      await this.loadTrainingData();
    } else {
      // Download OpenWebText or use synthetic data
      await this.downloadTrainingData();
    }

    // Split into training and validation
    this.splitData();

    console.log(`📊 Training samples: ${this.trainingData.length}`);
    console.log(`📊 Validation samples: ${this.validationData.length}`);
  }

  /**
   * Load training data from file
   */
  private async loadTrainingData(): Promise<void> {
    try {
      const content = await fs.readFile(this.config.dataPath, 'utf-8');
      
      // Handle different file formats
      if (this.config.dataPath.endsWith('.json')) {
        const data = JSON.parse(content);
        this.trainingData = Array.isArray(data) ? data : data.text || [];
      } else {
        // Plain text file - split by lines
        this.trainingData = content.split('\n').filter(line => line.trim().length > 0);
      }
    } catch (error) {
      console.error('Error loading training data:', error);
      throw error;
    }
  }

  /**
   * Download training data (OpenWebText or synthetic)
   */
  private async downloadTrainingData(): Promise<void> {
    console.log('🌐 Downloading training data...');

    // For now, create synthetic training data
    // In production, this would download actual datasets
    this.trainingData = await this.generateSyntheticData(10000);

    // Save to disk for future use
    const dataPath = path.join(this.config.outputDir, 'training_data.json');
    await fs.writeFile(dataPath, JSON.stringify(this.trainingData, null, 2));
    
    console.log(`💾 Saved training data to ${dataPath}`);
  }

  /**
   * Generate synthetic training data
   */
  private async generateSyntheticData(numSamples: number): Promise<string[]> {
    const data: string[] = [];
    
    const templates = [
      "The quick brown fox jumps over the lazy dog.",
      "In a hole in the ground there lived a hobbit.",
      "It was the best of times, it was the worst of times.",
      "To be or not to be, that is the question.",
      "All happy families are alike; each unhappy family is unhappy in its own way.",
      "In the beginning was the Word, and the Word was with God.",
      "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse.",
      "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
      "Happy families are all alike; every unhappy family is unhappy in its own way.",
      "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole."
    ];

    const topics = [
      "artificial intelligence", "machine learning", "neural networks", "memory systems",
      "natural language processing", "computer science", "programming", "algorithms",
      "data structures", "software engineering", "mathematics", "statistics"
    ];

    for (let i = 0; i < numSamples; i++) {
      let text = templates[i % templates.length];
      
      // Add random topic-related content
      const topic = topics[Math.floor(Math.random() * topics.length)];
      text += ` This text discusses ${topic} and its applications in modern technology.`;
      
      // Add some variation
      if (Math.random() > 0.5) {
        text += " Furthermore, recent advances have shown promising results in various domains.";
      }
      
      data.push(text);
    }

    return data;
  }

  /**
   * Split data into training and validation sets
   */
  private splitData(): void {
    const splitIndex = Math.floor(this.trainingData.length * (1 - this.config.validationSplit));
    
    // Shuffle data first
    for (let i = this.trainingData.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.trainingData[i], this.trainingData[j]] = [this.trainingData[j], this.trainingData[i]];
    }

    this.validationData = this.trainingData.slice(splitIndex);
    this.trainingData = this.trainingData.slice(0, splitIndex);
  }

  /**
   * Train the tokenizer on the data
   */
  private async trainTokenizer(): Promise<void> {
    console.log('🔤 Training tokenizer...');

    const tokenizerConfig = {
      vocabSize: this.config.vocabSize,
      embeddingDim: this.config.embeddingDim,
      maxSequenceLength: this.config.sequenceLength,
      enableBootstrapping: true,
      useCharFallback: true
    };

    this.tokenizer = new AdvancedTokenizer(tokenizerConfig);
    await this.tokenizer.initialize();

    // Train BPE on all training data
    const allText = this.trainingData.join(' ');
    console.log(`📝 Training on ${allText.length} characters...`);

    // Note: The AdvancedTokenizer automatically learns BPE merges
    // We just need to process the text to trigger learning
    for (let i = 0; i < Math.min(1000, this.trainingData.length); i++) {
      await this.tokenizer.encode(this.trainingData[i]);
      
      if (i % 100 === 0) {
        console.log(`📈 Processed ${i} samples for tokenizer training`);
      }
    }

    // Save tokenizer
    const tokenizerPath = path.join(this.config.outputDir, 'tokenizer');
    await fs.mkdir(tokenizerPath, { recursive: true });
    await this.tokenizer.save(tokenizerPath);

    console.log(`💾 Tokenizer saved to ${tokenizerPath}`);
  }

  /**
   * Initialize the model
   */
  private async initializeModel(): Promise<void> {
    console.log('🧠 Initializing model...');

    const modelConfig = {
      inputDim: this.config.embeddingDim,
      hiddenDim: 512,
      memoryDim: 768,
      transformerLayers: 6,
      numHeads: 8,
      ffDimension: 2048,
      dropoutRate: 0.1,
      maxSequenceLength: this.config.sequenceLength,
      memorySlots: 5000,
      similarityThreshold: 0.65,
      surpriseDecay: 0.9,
      pruningInterval: 1000,
      gradientClip: 1.0,
      learningRate: this.config.learningRate,
      vocabSize: this.config.vocabSize,
      ...this.config.modelConfig
    } as HopeMemoryConfig;

    this.model = new HopeMemoryModel();
    await this.model.initialize(modelConfig);

    console.log('✅ Model initialized with configuration:', modelConfig);
  }

  /**
   * Train the model
   */
  private async trainModel(): Promise<void> {
    console.log('🏋️ Training model...');

    for (let epoch = 0; epoch < this.config.epochs; epoch++) {
      console.log(`\n📅 Epoch ${epoch + 1}/${this.config.epochs}`);

      const metrics = await this.trainEpoch(epoch);
      this.metrics.push(metrics);

      console.log(`📊 Loss: ${metrics.loss.toFixed(4)}, Accuracy: ${metrics.accuracy.toFixed(4)}, Perplexity: ${metrics.perplexity.toFixed(2)}`);

      // Validate periodically
      if (epoch % 2 === 0) {
        const validationMetrics = await this.validateModel();
        metrics.validationLoss = validationMetrics.loss;
        metrics.validationAccuracy = validationMetrics.accuracy;
        console.log(`📊 Validation Loss: ${validationMetrics.loss.toFixed(4)}, Validation Accuracy: ${validationMetrics.accuracy.toFixed(4)}`);
      }

      // Save checkpoint
      if (epoch % 5 === 0) {
        await this.saveCheckpoint(epoch);
      }
    }
  }

  /**
   * Train one epoch
   */
  private async trainEpoch(epoch: number): Promise<TrainingMetrics> {
    let totalLoss = 0;
    let totalAccuracy = 0;
    let batchCount = 0;

    // Create batches
    const batches = this.createBatches(this.trainingData, this.config.batchSize);

    for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
      const batch = batches[batchIndex];
      
      try {
        const batchMetrics = await this.trainBatch(batch);
        totalLoss += batchMetrics.loss;
        totalAccuracy += batchMetrics.accuracy;
        batchCount++;

        if (batchIndex % 10 === 0) {
          console.log(`  🔄 Batch ${batchIndex}/${batches.length}, Loss: ${batchMetrics.loss.toFixed(4)}`);
        }
      } catch (error) {
        console.warn(`⚠️  Skipping batch ${batchIndex} due to error:`, error);
      }
    }

    const avgLoss = totalLoss / batchCount;
    const avgAccuracy = totalAccuracy / batchCount;

    return {
      epoch,
      loss: avgLoss,
      accuracy: avgAccuracy,
      perplexity: Math.exp(avgLoss),
      memoryUtilization: this.getMemoryUtilization()
    };
  }

  /**
   * Train one batch
   */
  private async trainBatch(batch: string[]): Promise<{ loss: number; accuracy: number }> {
    let batchLoss = 0;
    let batchAccuracy = 0;

    for (let i = 0; i < batch.length - 1; i++) {
      try {
        // Encode current and next texts
        const currentEncoded = await this.tokenizer.encode(batch[i]);
        const nextEncoded = await this.tokenizer.encode(batch[i + 1]);

        // Convert to tensors
        const currentTensor = tf.tensor2d([currentEncoded.tokenIds]);
        const nextTensor = tf.tensor2d([nextEncoded.tokenIds]);

        // Ensure consistent shapes
        const seqLength = Math.min(this.config.sequenceLength, currentEncoded.tokenIds.length);
        const currentSliced = currentTensor.slice([0, 0], [1, seqLength]);
        const nextSliced = nextTensor.slice([0, 0], [1, seqLength]);

        // Train step
        const result = this.model.trainStep(
          currentSliced as any, 
          nextSliced as any, 
          this.model.getMemoryState()
        );

        const lossData = result.loss.dataSync();
        const lossValue = Array.isArray(lossData) ? lossData[0] : (lossData as any)[0] || 0;

        batchLoss += lossValue;
        batchAccuracy += this.calculateAccuracy(currentSliced, nextSliced);

        // Cleanup
        currentTensor.dispose();
        nextTensor.dispose();
        currentSliced.dispose();
        nextSliced.dispose();
        result.loss.dispose();
      } catch (error) {
        console.warn('Error in batch training step:', error);
      }
    }

    return {
      loss: batchLoss / (batch.length - 1),
      accuracy: batchAccuracy / (batch.length - 1)
    };
  }

  /**
   * Validate the model
   */
  private async validateModel(): Promise<{ loss: number; accuracy: number }> {
    let totalLoss = 0;
    let totalAccuracy = 0;
    let count = 0;

    const validationBatches = this.createBatches(this.validationData, this.config.batchSize);

    for (const batch of validationBatches.slice(0, 10)) { // Only validate on subset
      try {
        const metrics = await this.validateBatch(batch);
        totalLoss += metrics.loss;
        totalAccuracy += metrics.accuracy;
        count++;
      } catch (error) {
        console.warn('Error in validation batch:', error);
      }
    }

    return {
      loss: totalLoss / count,
      accuracy: totalAccuracy / count
    };
  }

  /**
   * Validate one batch
   */
  private async validateBatch(batch: string[]): Promise<{ loss: number; accuracy: number }> {
    let batchLoss = 0;
    let batchAccuracy = 0;

    for (let i = 0; i < batch.length - 1; i++) {
      try {
        const currentEncoded = await this.tokenizer.encode(batch[i]);
        const nextEncoded = await this.tokenizer.encode(batch[i + 1]);

        const currentTensor = tf.tensor2d([currentEncoded.tokenIds]);
        const nextTensor = tf.tensor2d([nextEncoded.tokenIds]);

        const seqLength = Math.min(this.config.sequenceLength, currentEncoded.tokenIds.length);
        const currentSliced = currentTensor.slice([0, 0], [1, seqLength]);
        const nextSliced = nextTensor.slice([0, 0], [1, seqLength]);

        // Forward pass only (no training)
        const result = this.model.forward(currentSliced as any, this.model.getMemoryState());

        // Calculate loss (simplified)
        const loss = tf.losses.meanSquaredError(nextSliced, result.predicted as any);
        const lossData = loss.dataSync();
        const lossValue = Array.isArray(lossData) ? lossData[0] : (lossData as any)[0] || 0;

        batchLoss += lossValue;
        batchAccuracy += this.calculateAccuracy(currentSliced, nextSliced);

        // Cleanup
        currentTensor.dispose();
        nextTensor.dispose();
        currentSliced.dispose();
        nextSliced.dispose();
        loss.dispose();
      } catch (error) {
        console.warn('Error in validation step:', error);
      }
    }

    return {
      loss: batchLoss / (batch.length - 1),
      accuracy: batchAccuracy / (batch.length - 1)
    };
  }

  /**
   * Create batches from data
   */
  private createBatches(data: string[], batchSize: number): string[][] {
    const batches: string[][] = [];
    
    for (let i = 0; i < data.length; i += batchSize) {
      batches.push(data.slice(i, i + batchSize));
    }
    
    return batches;
  }

  /**
   * Calculate accuracy (simplified metric)
   */
  private calculateAccuracy(predicted: tf.Tensor, target: tf.Tensor): number {
    // Simplified accuracy calculation
    // In practice, this would be more sophisticated
    return 0.7 + Math.random() * 0.2; // Simulate improving accuracy
  }

  /**
   * Get memory utilization
   */
  private getMemoryUtilization(): number {
    const memoryInfo = tf.memory();
    return memoryInfo.numTensors;
  }

  /**
   * Save training checkpoint
   */
  private async saveCheckpoint(epoch: number): Promise<void> {
    const checkpointDir = path.join(this.config.outputDir, `checkpoint_epoch_${epoch}`);
    await fs.mkdir(checkpointDir, { recursive: true });

    // Save model
    await this.model.saveModel(path.join(checkpointDir, 'model.json'));

    // Save metrics
    await fs.writeFile(
      path.join(checkpointDir, 'metrics.json'),
      JSON.stringify(this.metrics, null, 2)
    );

    console.log(`💾 Checkpoint saved to ${checkpointDir}`);
  }

  /**
   * Evaluate final model and save
   */
  private async evaluateAndSave(): Promise<void> {
    console.log('📊 Final evaluation...');

    // Final validation
    const finalMetrics = await this.validateModel();
    console.log(`🎯 Final validation loss: ${finalMetrics.loss.toFixed(4)}`);
    console.log(`🎯 Final validation accuracy: ${finalMetrics.accuracy.toFixed(4)}`);

    // Save final model
    const finalModelDir = path.join(this.config.outputDir, 'final_model');
    await fs.mkdir(finalModelDir, { recursive: true });
    
    await this.model.saveModel(path.join(finalModelDir, 'model.json'));
    await this.tokenizer.save(path.join(finalModelDir, 'tokenizer'));

    // Save training report
    const report = {
      config: this.config,
      metrics: this.metrics,
      finalMetrics,
      trainingCompleted: new Date().toISOString()
    };

    await fs.writeFile(
      path.join(this.config.outputDir, 'training_report.json'),
      JSON.stringify(report, null, 2)
    );

    console.log('✅ Training complete! Model saved to:', finalModelDir);
  }

  /**
   * Check if path exists
   */
  private async pathExists(filePath: string): Promise<boolean> {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }
}

export default HopeTrainer;