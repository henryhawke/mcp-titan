import * as fs from 'fs/promises';
import * as path from 'path';
import * as crypto from 'crypto';
import * as tf from '@tensorflow/tfjs-node';
import { HopeMemoryModel } from './model.js';
import type { HopeMemoryConfig } from './model.js';
import { AdvancedTokenizer } from './tokenizer/index.js';
import { HNSW } from './ann.js';

export interface CheckpointMetadata {
  version: string;
  format: string;
  created: string;
  modelHash: string;
  snapshotId: string;
  config: HopeMemoryConfig;
  files: {
    modelConfig: string;
    weights: string;
    annIndex: string;
    tokenizerMerges: string;
    memoryState?: string;
    telemetry?: string;
  };
  size: {
    total: number;
    compressed?: number;
  };
  integrity: {
    checksums: Record<string, string>;
    verified: boolean;
  };
}

export interface SnapshotRotationConfig {
  maxSnapshots: number;
  retentionPolicy: 'time' | 'count' | 'size';
  retentionValue: number; // days for time, count for count, MB for size
  autoCleanup: boolean;
}

export interface PersistenceOptions {
  baseDir: string;
  compression: boolean;
  verification: boolean;
  rotation: SnapshotRotationConfig;
  backup: boolean;
}

export class RobustPersistenceManager {
  private baseDir: string;
  private options: PersistenceOptions;

  constructor(options: Partial<PersistenceOptions> = {}) {
    this.options = {
      baseDir: 'checkpoints',
      compression: false,
      verification: true,
      rotation: {
        maxSnapshots: 10,
        retentionPolicy: 'count',
        retentionValue: 10,
        autoCleanup: true
      },
      backup: false,
      ...options
    };
    this.baseDir = this.options.baseDir;
  }

  /**
   * Save a complete model checkpoint with all components
   */
  async saveCheckpoint(
    model: HopeMemoryModel,
    tokenizer?: AdvancedTokenizer,
    annIndex?: HNSW,
    metadata?: Partial<CheckpointMetadata>
  ): Promise<string> {
    try {
      const modelHash = this.generateModelHash(model.getConfig());
      const snapshotId = this.generateSnapshotId();
      const snapshotDir = this.getSnapshotPath(modelHash, snapshotId);

      // Create snapshot directory
      await fs.mkdir(snapshotDir, { recursive: true });

      // Save all components
      const files = await this.saveAllComponents(
        snapshotDir,
        model,
        tokenizer,
        annIndex
      );

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

      // Update latest symlink
      await this.updateLatestSymlink(modelHash, snapshotId);

      // Perform rotation cleanup
      if (this.options.rotation.autoCleanup) {
        await this.rotateSnapshots(modelHash);
      }

      console.log(`✅ Checkpoint saved: ${snapshotDir}`);
      return snapshotDir;
    } catch (error) {
      console.error('Failed to save checkpoint:', error);
      throw new Error(`Checkpoint save failed: ${error}`);
    }
  }

  /**
   * Load a model checkpoint with all components
   */
  async loadCheckpoint(
    checkpointPath: string,
    options: {
      verifyIntegrity?: boolean;
      loadComponents?: Array<'model' | 'tokenizer' | 'annIndex'>;
    } = {}
  ): Promise<{
    model: HopeMemoryModel;
    tokenizer?: AdvancedTokenizer;
    annIndex?: HNSW;
    metadata: CheckpointMetadata;
  }> {
    try {
      const { verifyIntegrity = true, loadComponents = ['model'] } = options;

      // Load and validate metadata
      const metadata = await this.loadMetadata(checkpointPath);

      if (verifyIntegrity) {
        await this.verifyCheckpointIntegrity(checkpointPath, metadata);
      }

      // Load model
      const model = new HopeMemoryModel();
      await this.loadModelComponent(path.dirname(checkpointPath), model, metadata);

      // Load optional components
      let tokenizer: AdvancedTokenizer | undefined;
      let annIndex: HNSW | undefined;

      if (loadComponents.includes('tokenizer')) {
        tokenizer = await this.loadTokenizerComponent(path.dirname(checkpointPath), metadata);
      }

      if (loadComponents.includes('annIndex')) {
        annIndex = await this.loadAnnIndexComponent(path.dirname(checkpointPath), metadata);
      }

      console.log(`✅ Checkpoint loaded: ${checkpointPath}`);
      return { model, tokenizer, annIndex, metadata };
    } catch (error) {
      console.error('Failed to load checkpoint:', error);
      throw new Error(`Checkpoint load failed: ${error}`);
    }
  }

  /**
   * List available checkpoints
   */
  async listCheckpoints(modelHash?: string): Promise<CheckpointMetadata[]> {
    const checkpoints: CheckpointMetadata[] = [];

    try {
      const baseExists = await this.pathExists(this.baseDir);
      if (!baseExists) {
        return checkpoints;
      }

      const modelDirs = modelHash
        ? [modelHash]
        : await fs.readdir(this.baseDir);

      for (const hash of modelDirs) {
        const modelDir = path.join(this.baseDir, hash);
        const modelDirStats = await fs.stat(modelDir).catch(() => null);

        if (!modelDirStats?.isDirectory()) { continue; }

        const snapshots = await fs.readdir(modelDir);
        for (const snapshot of snapshots) {
          if (snapshot === 'latest') { continue; } // Skip symlink

          const snapshotDir = path.join(modelDir, snapshot);
          const checkpointFile = path.join(snapshotDir, 'checkpoint.json');

          try {
            const metadata = await this.loadMetadata(checkpointFile);
            checkpoints.push(metadata);
          } catch (error) {
            console.warn(`Skipping invalid checkpoint: ${snapshotDir}`);
          }
        }
      }

      // Sort by creation date (newest first)
      return checkpoints.sort((a, b) =>
        new Date(b.created).getTime() - new Date(a.created).getTime()
      );
    } catch (error) {
      console.error('Failed to list checkpoints:', error);
      return checkpoints;
    }
  }

  /**
   * Get the latest checkpoint for a model
   */
  async getLatestCheckpoint(modelHash: string): Promise<string | null> {
    const latestPath = path.join(this.baseDir, modelHash, 'latest');

    try {
      const stats = await fs.lstat(latestPath);
      if (stats.isSymbolicLink()) {
        const target = await fs.readlink(latestPath);
        const fullPath = path.resolve(path.dirname(latestPath), target);
        return path.join(fullPath, 'checkpoint.json');
      }
    } catch (error) {
      // No latest symlink exists
    }

    return null;
  }

  /**
   * Clean up old snapshots based on rotation policy
   */
  async rotateSnapshots(modelHash: string): Promise<void> {
    const modelDir = path.join(this.baseDir, modelHash);
    const { maxSnapshots, retentionPolicy, retentionValue } = this.options.rotation;

    try {
      const snapshots = await fs.readdir(modelDir);
      const snapshotDirs = snapshots
        .filter(name => name !== 'latest')
        .map(name => ({
          name,
          path: path.join(modelDir, name),
          fullPath: path.join(modelDir, name, 'checkpoint.json')
        }));

      // Load metadata for all snapshots
      const snapshotMetadata = await Promise.all(
        snapshotDirs.map(async (snapshot) => {
          try {
            const metadata = await this.loadMetadata(snapshot.fullPath);
            return { ...snapshot, metadata, created: new Date(metadata.created) };
          } catch {
            return null;
          }
        })
      );

      const validSnapshots = snapshotMetadata
        .filter(snapshot => snapshot !== null)
        .sort((a, b) => b.created.getTime() - a.created.getTime());

      let snapshotsToDelete: typeof validSnapshots = [];

      switch (retentionPolicy) {
        case 'count':
          if (validSnapshots.length > maxSnapshots) {
            snapshotsToDelete = validSnapshots.slice(maxSnapshots);
          }
          break;

        case 'time':
          const cutoffDate = new Date(Date.now() - retentionValue * 24 * 60 * 60 * 1000);
          snapshotsToDelete = validSnapshots.filter(s => s.created < cutoffDate);
          break;

        case 'size':
          let totalSize = 0;
          const maxSizeBytes = retentionValue * 1024 * 1024; // Convert MB to bytes
          for (const snapshot of validSnapshots) {
            totalSize += snapshot.metadata.size.total;
            if (totalSize > maxSizeBytes && validSnapshots.indexOf(snapshot) > 0) {
              snapshotsToDelete = validSnapshots.slice(validSnapshots.indexOf(snapshot));
              break;
            }
          }
          break;
      }

      // Delete old snapshots
      for (const snapshot of snapshotsToDelete) {
        if (snapshot) {
          await this.deleteDirectory(snapshot.path);
          console.log(`🗑️  Deleted old snapshot: ${snapshot.name}`);
        }
      }

      if (snapshotsToDelete.length > 0) {
        console.log(`♻️  Cleaned up ${snapshotsToDelete.length} old snapshots`);
      }
    } catch (error) {
      console.error('Failed to rotate snapshots:', error);
    }
  }

  /**
   * Verify checkpoint integrity
   */
  async verifyCheckpointIntegrity(checkpointPath: string, metadata: CheckpointMetadata): Promise<boolean> {
    if (!this.options.verification) {
      return true;
    }

    try {
      const snapshotDir = path.dirname(checkpointPath);
      const { files, integrity } = metadata;

      // Verify all expected files exist
      for (const [component, filename] of Object.entries(files)) {
        if (filename) {
          const filePath = path.join(snapshotDir, filename);
          const exists = await this.pathExists(filePath);
          if (!exists) {
            throw new Error(`Missing component file: ${component} (${filename})`);
          }
        }
      }

      // Verify checksums
      for (const [filename, expectedChecksum] of Object.entries(integrity.checksums)) {
        const filePath = path.join(snapshotDir, filename);
        const actualChecksum = await this.calculateFileChecksum(filePath);
        if (actualChecksum !== expectedChecksum) {
          throw new Error(`Checksum mismatch for ${filename}`);
        }
      }

      return true;
    } catch (error) {
      console.error('Integrity verification failed:', error);
      return false;
    }
  }

  /**
   * Save all model components
   */
  private async saveAllComponents(
    snapshotDir: string,
    model: HopeMemoryModel,
    tokenizer?: AdvancedTokenizer,
    annIndex?: HNSW
  ): Promise<CheckpointMetadata['files']> {
    const files: CheckpointMetadata['files'] = {
      modelConfig: 'modelConfig.json',
      weights: 'weights',
      annIndex: 'annIndex.json',
      tokenizerMerges: 'tokenizerMerges.json'
    };

    // Save model configuration
    await fs.writeFile(
      path.join(snapshotDir, files.modelConfig),
      JSON.stringify(model.getConfig(), null, 2)
    );

    // Save model weights
    const weightsDir = path.join(snapshotDir, files.weights);
    await model.save(`file://${weightsDir}`);

    // Save memory state
    const memorySnapshot = model.getMemorySnapshot();
    if (memorySnapshot) {
      files.memoryState = 'memoryState.json';
      const memoryData = {
        shortTerm: await memorySnapshot.shortTerm.array(),
        longTerm: await memorySnapshot.longTerm.array(),
        meta: await memorySnapshot.meta.array(),
        timestamps: Array.from(await memorySnapshot.timestamps.data()),
        accessCounts: Array.from(await memorySnapshot.accessCounts.data()),
        surpriseHistory: Array.from(await memorySnapshot.surpriseHistory.data())
      };
      await fs.writeFile(
        path.join(snapshotDir, files.memoryState),
        JSON.stringify(memoryData, null, 2)
      );
    }

    // Save tokenizer data
    if (tokenizer) {
      const tokenizerData = {
        merges: tokenizer.getBPETokenizer().getMerges() || [],
        vocab: tokenizer.getBPETokenizer().getVocab() || {},
        config: tokenizer.getConfig() || {},
        stats: tokenizer.getStats()
      };
      await fs.writeFile(
        path.join(snapshotDir, files.tokenizerMerges),
        JSON.stringify(tokenizerData, null, 2)
      );

      // Save tokenizer weights if embedding is available
      try {
        const embeddingDir = path.join(snapshotDir, 'tokenizer_weights');
        await tokenizer.save(embeddingDir);
      } catch (error) {
        console.warn('Could not save tokenizer weights:', error);
      }
    } else {
      // Create empty tokenizer file
      await fs.writeFile(
        path.join(snapshotDir, files.tokenizerMerges),
        JSON.stringify({ merges: [], vocab: {}, metadata: { placeholder: true } }, null, 2)
      );
    }

    // Save ANN index
    if (annIndex) {
      const annData = {
        type: 'HNSW',
        built: annIndex.isIndexBuilt || false,
        nodes: await this.serializeHNSWNodes(annIndex),
        parameters: annIndex.getParameters()
      };
      await fs.writeFile(
        path.join(snapshotDir, files.annIndex),
        JSON.stringify(annData, null, 2)
      );
    } else {
      // Create empty ANN index file
      await fs.writeFile(
        path.join(snapshotDir, files.annIndex),
        JSON.stringify({ type: 'HNSW', built: false, nodes: [], metadata: { placeholder: true } }, null, 2)
      );
    }

    return files;
  }

  /**
   * Load model component
   */
  private async loadModelComponent(snapshotDir: string, model: HopeMemoryModel, metadata: CheckpointMetadata): Promise<void> {
    // Load configuration
    const configPath = path.join(snapshotDir, metadata.files.modelConfig);
    const config = JSON.parse(await fs.readFile(configPath, 'utf-8'));

    // Initialize model with config
    await model.initialize(config);

    // Load weights
    const weightsPath = path.join(snapshotDir, metadata.files.weights);
    await model.load(`file://${weightsPath}`);

    // Load memory state if available
    if (metadata.files.memoryState) {
      const memoryStatePath = path.join(snapshotDir, metadata.files.memoryState);
      const memoryData = JSON.parse(await fs.readFile(memoryStatePath, 'utf-8'));

      // Restore memory state using public method
      model.restoreMemoryState(memoryData);
    }
  }

  /**
   * Load tokenizer component
   */
  private async loadTokenizerComponent(snapshotDir: string, metadata: CheckpointMetadata): Promise<AdvancedTokenizer | undefined> {
    try {
      const tokenizerPath = path.join(snapshotDir, metadata.files.tokenizerMerges);
      const tokenizerData = JSON.parse(await fs.readFile(tokenizerPath, 'utf-8'));

      if (tokenizerData.metadata?.placeholder) {
        return undefined;
      }

      const tokenizer = new AdvancedTokenizer(tokenizerData.config || {});
      await tokenizer.initialize();

      // Load tokenizer weights if available
      const weightsDir = path.join(snapshotDir, 'tokenizer_weights');
      if (await this.pathExists(weightsDir)) {
        await tokenizer.load(weightsDir);
      }

      return tokenizer;
    } catch (error) {
      console.warn('Could not load tokenizer component:', error);
      return undefined;
    }
  }

  /**
   * Load ANN index component
   */
  private async loadAnnIndexComponent(snapshotDir: string, metadata: CheckpointMetadata): Promise<HNSW | undefined> {
    try {
      const annPath = path.join(snapshotDir, metadata.files.annIndex);
      const annData = JSON.parse(await fs.readFile(annPath, 'utf-8'));

      if (annData.metadata?.placeholder || !annData.built) {
        return undefined;
      }

      const annIndex = new HNSW();
      await this.deserializeHNSWNodes(annIndex, annData);

      return annIndex;
    } catch (error) {
      console.warn('Could not load ANN index component:', error);
      return undefined;
    }
  }

  /**
   * Generate model hash based on configuration
   */
  private generateModelHash(config: HopeMemoryConfig): string {
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
   * Update latest symlink
   */
  private async updateLatestSymlink(modelHash: string, snapshotId: string): Promise<void> {
    const modelDir = path.join(this.baseDir, modelHash);
    const latestPath = path.join(modelDir, 'latest');

    // Remove existing symlink
    try {
      await fs.unlink(latestPath);
    } catch {
      // Symlink doesn't exist, that's ok
    }

    // Create new symlink
    await fs.symlink(snapshotId, latestPath);
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
  private async generateIntegrityData(snapshotDir: string, files: CheckpointMetadata['files']): Promise<CheckpointMetadata['integrity']> {
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
    const stats = await fs.stat(filePath);

    if (stats.isDirectory()) {
      // For directories, calculate checksum of all contained files
      const hash = crypto.createHash('sha256');
      const items = await fs.readdir(filePath);

      for (const item of items.sort()) {
        const itemPath = path.join(filePath, item);
        const itemStats = await fs.stat(itemPath);

        if (itemStats.isFile()) {
          const content = await fs.readFile(itemPath);
          hash.update(content);
        }
      }

      return hash.digest('hex');
    } else {
      const content = await fs.readFile(filePath);
      return crypto.createHash('sha256').update(content).digest('hex');
    }
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

  /**
   * Delete directory recursively
   */
  private async deleteDirectory(dirPath: string): Promise<void> {
    await fs.rm(dirPath, { recursive: true, force: true });
  }

  /**
   * Serialize HNSW nodes for storage
   */
  private async serializeHNSWNodes(annIndex: HNSW): Promise<any[]> {
    // This is a simplified serialization - in practice you'd need to handle tensor serialization
    const nodes = annIndex.hnswNodes || new Map();
    const serialized = [];

    for (const [id, node] of nodes.entries()) {
      serialized.push({
        id,
        vector: await node.vector.array(),
        connections: Object.fromEntries(node.connections)
      });
    }

    return serialized;
  }

  /**
   * Deserialize HNSW nodes from storage
   */
  private async deserializeHNSWNodes(annIndex: HNSW, data: any): Promise<void> {
    // This is a simplified deserialization - in practice you'd need to handle tensor deserialization
    const nodes = new Map();

    for (const nodeData of data.nodes) {
      const connections = new Map();
      for (const [level, connectionSet] of Object.entries(nodeData.connections)) {
        connections.set(parseInt(level), new Set(connectionSet as number[]));
      }

      nodes.set(nodeData.id, {
        id: nodeData.id,
        vector: tf.tensor(nodeData.vector),
        connections
      });
    }

    annIndex.restoreState(nodes, data.built);

    if (data.parameters) {
      annIndex.setParameters(data.parameters);
    }
  }
}

export default RobustPersistenceManager;
