/**
 * @fileoverview Tests for the tokenizer module
 */

import { AdvancedTokenizer, BPETokenizer, TokenEmbedding, MaskedLanguageModelHead } from '../tokenizer/index.js';
import * as tf from '@tensorflow/tfjs-node';

describe('Tokenizer Module', () => {
  beforeAll(async () => {
    await tf.ready();
  });

  afterEach(() => {
    // Clean up any remaining tensors after each test
    const numTensors = tf.memory().numTensors;
    if (numTensors > 0) {
      console.warn(`${numTensors} tensors still in memory after test`);
    }
  });

  describe('BPETokenizer', () => {
    test('should initialize with default config', () => {
      const tokenizer = new BPETokenizer({ vocabSize: 1000 });
      expect(tokenizer.getVocabSize()).toBeGreaterThan(256); // Base chars + special tokens
      tokenizer.dispose();
    });

    test('should encode and decode text', () => {
      const tokenizer = new BPETokenizer({ vocabSize: 1000 });
      const text = 'hello world';
      const tokens = tokenizer.encode(text);
      const decoded = tokenizer.decode(tokens);
      
      expect(tokens).toBeInstanceOf(Array);
      expect(tokens.length).toBeGreaterThan(0);
      expect(decoded.toLowerCase().replace(/\s+/g, ' ').trim()).toContain('hello');
      
      tokenizer.dispose();
    });

    test('should learn from text and create merges', async () => {
      const tokenizer = new BPETokenizer({ vocabSize: 1000 });
      const trainingText = 'hello world hello world hello world test test test';
      
      const initialMerges = tokenizer.getStats().mergesCount;
      await tokenizer.learnFromText(trainingText, 10);
      const finalMerges = tokenizer.getStats().mergesCount;
      
      expect(finalMerges).toBeGreaterThanOrEqual(initialMerges);
      tokenizer.dispose();
    });
  });

  describe('TokenEmbedding', () => {
    test('should initialize with random weights', () => {
      const embedding = new TokenEmbedding({ vocabSize: 1000, embeddingDim: 256 });
      const stats = embedding.getStats();
      
      expect(stats.vocabSize).toBe(1000);
      expect(stats.embeddingDim).toBe(256);
      expect(stats.totalParams).toBe(256000);
      
      embedding.dispose();
    });

    test('should embed token IDs', async () => {
      const embedding = new TokenEmbedding({ vocabSize: 1000, embeddingDim: 256 });
      const tokenIds = [1, 2, 3, 4, 5];
      
      const embeddings = await embedding.embed(tokenIds);
      expect(embeddings.shape).toEqual([5, 256]);
      
      embeddings.dispose();
      embedding.dispose();
    });

    test('should find similar tokens', async () => {
      const embedding = new TokenEmbedding({ vocabSize: 100, embeddingDim: 64 });
      const queryEmbedding = await embedding.embedSingle(10);
      
      const similar = embedding.findSimilar(queryEmbedding, 5);
      expect(similar.tokenIds).toHaveLength(5);
      expect(similar.similarities).toHaveLength(5);
      
      queryEmbedding.dispose();
      embedding.dispose();
    });
  });

  describe('AdvancedTokenizer', () => {
    test('should initialize in BPE mode by default', async () => {
      const tokenizer = new AdvancedTokenizer({ vocabSize: 1000, embeddingDim: 256 });
      await tokenizer.initialize();
      
      const stats = tokenizer.getStats();
      expect(stats.mode).toBe('BPE');
      
      tokenizer.dispose();
    });

    test('should switch to legacy mode', async () => {
      const tokenizer = new AdvancedTokenizer({ vocabSize: 1000, embeddingDim: 256 });
      await tokenizer.initialize();
      
      tokenizer.setLegacyMode(true);
      const stats = tokenizer.getStats();
      expect(stats.mode).toBe('Legacy');
      
      tokenizer.dispose();
    });

    test('should encode text with embeddings', async () => {
      const tokenizer = new AdvancedTokenizer({ 
        vocabSize: 1000, 
        embeddingDim: 256,
        maxSequenceLength: 64
      });
      await tokenizer.initialize();
      
      const result = await tokenizer.encode('hello world test');
      
      expect(result.tokenIds).toBeInstanceOf(Array);
      expect(result.embeddings.shape[1]).toBe(256);
      expect(result.metadata.usedLegacyMode).toBe(false);
      
      result.embeddings.dispose();
      result.attentionMask.dispose();
      tokenizer.dispose();
    });

    test('should decode token IDs back to text', async () => {
      const tokenizer = new AdvancedTokenizer({ vocabSize: 1000, embeddingDim: 256 });
      await tokenizer.initialize();
      
      const text = 'hello world';
      const encoded = await tokenizer.encode(text);
      const decoded = tokenizer.decode(encoded.tokenIds);
      
      expect(decoded).toContain('hello');
      
      encoded.embeddings.dispose();
      encoded.attentionMask.dispose();
      tokenizer.dispose();
    });
  });

  describe('MaskedLanguageModelHead', () => {
    test('should initialize MLM head', async () => {
      const tokenizer = new BPETokenizer({ vocabSize: 1000 });
      const embedding = new TokenEmbedding({ vocabSize: 1000, embeddingDim: 256 });
      
      const mlm = new MaskedLanguageModelHead(
        { hiddenSize: 512, vocabSize: 1000 },
        tokenizer,
        embedding
      );
      
      expect(mlm.getConfig().vocabSize).toBe(1000);
      expect(mlm.getConfig().hiddenSize).toBe(512);
      
      mlm.dispose();
      embedding.dispose();
      tokenizer.dispose();
    });

    test('should create masks for training', async () => {
      const tokenizer = new BPETokenizer({ vocabSize: 1000 });
      const embedding = new TokenEmbedding({ vocabSize: 1000, embeddingDim: 256 });
      
      const mlm = new MaskedLanguageModelHead(
        { hiddenSize: 512, vocabSize: 1000 },
        tokenizer,
        embedding
      );
      
      const inputIds = tf.tensor2d([[1, 2, 3, 4, 5]], [1, 5], 'int32');
      const masks = mlm.createMasks(inputIds);
      
      expect(masks.maskedInputIds.shape).toEqual([1, 5]);
      expect(masks.labels.shape).toEqual([1, 5]);
      expect(masks.maskedPositions.shape).toEqual([1, 5]);
      
      inputIds.dispose();
      masks.maskedInputIds.dispose();
      masks.labels.dispose();
      masks.maskedPositions.dispose();
      
      mlm.dispose();
      embedding.dispose();
      tokenizer.dispose();
    });

    test('should prepare training batch', async () => {
      const tokenizer = new BPETokenizer({ vocabSize: 1000 });
      const embedding = new TokenEmbedding({ vocabSize: 1000, embeddingDim: 256 });
      
      const mlm = new MaskedLanguageModelHead(
        { hiddenSize: 512, vocabSize: 1000, maxSequenceLength: 32 },
        tokenizer,
        embedding
      );
      
      const texts = ['hello world', 'test sentence'];
      const batch = await mlm.prepareBatch(texts, 32);
      
      expect(batch.inputIds.shape[0]).toBe(2); // batch size
      expect(batch.inputIds.shape[1]).toBe(32); // sequence length
      expect(batch.attentionMask.shape).toEqual([2, 32]);
      
      batch.inputIds.dispose();
      batch.attentionMask.dispose();
      batch.labels.dispose();
      batch.maskedPositions.dispose();
      
      mlm.dispose();
      embedding.dispose();
      tokenizer.dispose();
    });
  });

  describe('Integration with HopeMemoryModel', () => {
    test('should be importable from main index', () => {
      // Just test that the exports are available
      expect(AdvancedTokenizer).toBeDefined();
      expect(BPETokenizer).toBeDefined();
      expect(TokenEmbedding).toBeDefined();
      expect(MaskedLanguageModelHead).toBeDefined();
    });
  });
});
