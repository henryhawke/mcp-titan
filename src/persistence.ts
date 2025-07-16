import * as fs from 'fs/promises';
import * as path from 'path';
import * as crypto from 'crypto';
import * as tf from '@tensorflow/tfjs-node';
import { TitanMemoryModel, TitanMemoryConfig } from './model.js';

export interface CheckpointMetadata {
  version: string;
  format: string;
  created: string;
  modelHash: string;
  snapshotId: string;
  config: TitanMemoryConfig;
  files: {
    modelConfig: string;
    weights: string;
    memoryState?: string;
  };
  size: {
    total: number;
  };
  integrity: {
    checksums: Record<string, string>;
    verified: boolean;
  };
}

export interface PersistenceOptions {
  baseDir: string;
  verification: boolean;
}

export class RobustPersistenceManager {
  private baseDir: string;
  private options: PersistenceOptions;

  constructor(options: Partial<PersistenceOptions> = {}) {
    this.options = {
      baseDir: 'checkpoints',
      verification: true,
      ...options
    };
    this.baseDir = this.options.baseDir;
  }

  /**
   * Save a complete model checkpoint
   */
  async saveCheckpoint(
    model: TitanMemoryModel,
    metadata?: Partial<CheckpointMetadata>
  ): Promise<string> {
    try {
      const modelHash = this.generateModelHash(model.getConfig());
      const snapshotId = this.generateSnapshotId();
      const snapshotDir = this.getSnapshotPath(modelHash, snapshotId);

      // Create snapshot directory
      await fs.mkdir(snapshotDir, { recursive: true });

      // Save all components
      const files = await this.saveAllComponents(snapshotDir, model);

      // Create checkpoint metadata
      const checkpointMetadata: CheckpointMetadata = {
        version: '2.0',
        format: 'titan-memory-v2',
        created: new Date().toISOString(),
        modelHash,
        snapshotId,
        config: model.getConfig(),
        files,
        size: await this.calculateDirectorySize(snapshotDir),
        integrity: await this.generateIntegrityData(snapshotDir, files),
        ...metadata
      };

      // Save metadata
      await fs.writeFile(
        path.join(snapshotDir, 'checkpoint.json'),
        JSON.stringify(checkpointMetadata, null, 2)
      );

      console.log(`✅ Checkpoint saved: ${snapshotDir}`);
      return snapshotDir;
    } catch (error) {
      console.error('Failed to save checkpoint:', error);
      throw new Error(`Checkpoint save failed: ${error}`);
    }
  }

  /**
   * Load a model checkpoint
   */
  async loadCheckpoint(checkpointPath: string): Promise<{
    model: TitanMemoryModel;
    metadata: CheckpointMetadata;
  }> {
    try {
      // Load and validate metadata
      const metadata = await this.loadMetadata(checkpointPath);
      
      // Load model
      const model = new TitanMemoryModel();
      await this.loadModelComponent(path.dirname(checkpointPath), model, metadata);

      console.log(`✅ Checkpoint loaded: ${checkpointPath}`);
      return { model, metadata };
    } catch (error) {
      console.error('Failed to load checkpoint:', error);
      throw new Error(`Checkpoint load failed: ${error}`);
    }
  }

  /**
   * Save all model components
   */
  private async saveAllComponents(
    snapshotDir: string,
    model: TitanMemoryModel
  ): Promise<CheckpointMetadata['files']> {
    const files: CheckpointMetadata['files'] = {
      modelConfig: 'modelConfig.json',
      weights: 'weights.json'
    };

    // Save model configuration
    await fs.writeFile(
      path.join(snapshotDir, files.modelConfig),
      JSON.stringify(model.getConfig(), null, 2)
    );

    // Save model weights using model's built-in save
    await model.saveModel(path.join(snapshotDir, files.weights));

    return files;
  }

  /**
   * Load model component
   */
  private async loadModelComponent(
    snapshotDir: string, 
    model: TitanMemoryModel, 
    metadata: CheckpointMetadata
  ): Promise<void> {
    // Load configuration
    const configPath = path.join(snapshotDir, metadata.files.modelConfig);
    const config = JSON.parse(await fs.readFile(configPath, 'utf-8'));
    
    // Initialize model with config
    await model.initialize(config);
    
    // Load weights
    const weightsPath = path.join(snapshotDir, metadata.files.weights);
    await model.loadModel(weightsPath);
  }

  /**
   * Generate model hash based on configuration
   */
  private generateModelHash(config: TitanMemoryConfig): string {
    const hashableConfig = {
      inputDim: config.inputDim,
      hiddenDim: config.hiddenDim,
      memorySlots: config.memorySlots,
      transformerLayers: config.transformerLayers
    };
    
    const hash = crypto.createHash('sha256');
    hash.update(JSON.stringify(hashableConfig));
    return hash.digest('hex').substring(0, 16);
  }

  /**
   * Generate snapshot ID with timestamp
   */
  private generateSnapshotId(): string {
    const now = new Date();
    const datePart = now.toISOString().split('T')[0].replace(/-/g, '');
    const timePart = now.toTimeString().split(' ')[0].replace(/:/g, '');
    return `snapshot-${datePart}-${timePart}`;
  }

  /**
   * Get snapshot directory path
   */
  private getSnapshotPath(modelHash: string, snapshotId: string): string {
    return path.join(this.baseDir, modelHash, snapshotId);
  }

  /**
   * Calculate directory size
   */
  private async calculateDirectorySize(dirPath: string): Promise<{ total: number }> {
    let total = 0;
    
    const walk = async (currentPath: string): Promise<void> => {
      const items = await fs.readdir(currentPath);
      
      for (const item of items) {
        const itemPath = path.join(currentPath, item);
        const stats = await fs.stat(itemPath);
        
        if (stats.isDirectory()) {
          await walk(itemPath);
        } else {
          total += stats.size;
        }
      }
    };
    
    await walk(dirPath);
    return { total };
  }

  /**
   * Generate integrity data for checkpoint
   */
  private async generateIntegrityData(
    snapshotDir: string, 
    files: CheckpointMetadata['files']
  ): Promise<CheckpointMetadata['integrity']> {
    const checksums: Record<string, string> = {};
    
    for (const [component, filename] of Object.entries(files)) {
      if (filename) {
        const filePath = path.join(snapshotDir, filename);
        if (await this.pathExists(filePath)) {
          checksums[filename] = await this.calculateFileChecksum(filePath);
        }
      }
    }
    
    return {
      checksums,
      verified: true
    };
  }

  /**
   * Calculate file checksum
   */
  private async calculateFileChecksum(filePath: string): Promise<string> {
    const content = await fs.readFile(filePath);
    return crypto.createHash('sha256').update(content).digest('hex');
  }

  /**
   * Load checkpoint metadata
   */
  private async loadMetadata(checkpointPath: string): Promise<CheckpointMetadata> {
    const metadataContent = await fs.readFile(checkpointPath, 'utf-8');
    return JSON.parse(metadataContent);
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

export default RobustPersistenceManager;