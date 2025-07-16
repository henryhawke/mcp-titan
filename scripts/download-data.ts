#!/usr/bin/env node

import * as fs from 'fs/promises';
import * as path from 'path';
import * as https from 'https';

interface DataSource {
  name: string;
  url: string;
  description: string;
  size: string;
  outputFile: string;
  format: 'json' | 'text' | 'jsonl';
}

const DATA_SOURCES: DataSource[] = [
  {
    name: 'WikiText-2',
    url: 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
    description: 'Collection of over 100 million tokens extracted from Wikipedia articles',
    size: '12.7 MB',
    outputFile: 'wikitext-2.txt',
    format: 'text'
  },
  {
    name: 'TinyStories',
    url: 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt',
    description: 'Synthetic stories generated for language model training',
    size: '2.1 GB',
    outputFile: 'tinystories.txt',
    format: 'text'
  },
  {
    name: 'OpenWebText Sample',
    url: 'https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/subsets/urlsf_subset00-0_data.jsonl',
    description: 'Sample from OpenWebText dataset (Reddit submissions)',
    size: '1.2 GB',
    outputFile: 'openwebtext-sample.jsonl',
    format: 'jsonl'
  }
];

async function downloadFile(url: string, outputPath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = require('fs').createWriteStream(outputPath);
    
    https.get(url, (response) => {
      if (response.statusCode === 302 || response.statusCode === 301) {
        // Handle redirects
        file.close();
        require('fs').unlinkSync(outputPath);
        return downloadFile(response.headers.location!, outputPath).then(resolve).catch(reject);
      }
      
      if (response.statusCode !== 200) {
        reject(new Error(`HTTP ${response.statusCode}: ${response.statusMessage}`));
        return;
      }

      const totalSize = parseInt(response.headers['content-length'] || '0');
      let downloadedSize = 0;

      response.on('data', (chunk) => {
        downloadedSize += chunk.length;
        if (totalSize > 0) {
          const percentage = (downloadedSize / totalSize * 100).toFixed(1);
          process.stdout.write(`\r📥 Downloading: ${percentage}% (${(downloadedSize / 1024 / 1024).toFixed(1)} MB)`);
        }
      });

      response.pipe(file);

      file.on('finish', () => {
        file.close();
        console.log(`\n✅ Download completed: ${outputPath}`);
        resolve();
      });

      file.on('error', (err) => {
        require('fs').unlinkSync(outputPath);
        reject(err);
      });
    }).on('error', reject);
  });
}

async function processJsonl(inputFile: string, outputFile: string): Promise<void> {
  console.log(`📝 Processing JSONL file: ${inputFile}`);
  
  const content = await fs.readFile(inputFile, 'utf-8');
  const lines = content.split('\n').filter(line => line.trim());
  
  const texts: string[] = [];
  
  for (const line of lines) {
    try {
      const data = JSON.parse(line);
      if (data.text && typeof data.text === 'string' && data.text.length > 50) {
        texts.push(data.text.trim());
      }
    } catch (error) {
      // Skip invalid JSON lines
      continue;
    }
  }
  
  await fs.writeFile(outputFile, texts.join('\n'));
  console.log(`✅ Processed ${texts.length} texts to ${outputFile}`);
}

async function generateSyntheticData(outputFile: string, numSamples = 10000): Promise<void> {
  console.log(`🤖 Generating synthetic training data: ${numSamples} samples`);

  const templates = [
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "It was the best of times, it was the worst of times.",
    "To be or not to be, that is the question.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "In the beginning was the Word, and the Word was with God.",
    "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "Science is a way of thinking much more than it is a body of knowledge.",
    "The important thing is not to stop questioning. Curiosity has its own reason for existing."
  ];

  const topics = [
    "artificial intelligence", "machine learning", "neural networks", "memory systems",
    "natural language processing", "computer science", "programming", "algorithms",
    "data structures", "software engineering", "mathematics", "statistics",
    "deep learning", "transformers", "attention mechanisms", "language models",
    "cognitive science", "neuroscience", "philosophy", "linguistics"
  ];

  const connectors = [
    "Furthermore,", "Additionally,", "Moreover,", "In contrast,", "However,",
    "Nevertheless,", "Subsequently,", "Consequently,", "Therefore,", "Thus,"
  ];

  const elaborations = [
    "recent advances have shown promising results in various domains",
    "researchers continue to explore new methodologies and approaches",
    "this field has experienced rapid growth and development",
    "the implications of these findings extend far beyond the initial scope",
    "interdisciplinary collaboration has led to breakthrough discoveries",
    "practical applications are being developed across multiple industries",
    "theoretical foundations continue to evolve and strengthen",
    "experimental validation supports these theoretical frameworks"
  ];

  const texts: string[] = [];

  for (let i = 0; i < numSamples; i++) {
    let text = templates[i % templates.length];
    
    // Add topic-specific content
    const topic = topics[Math.floor(Math.random() * topics.length)];
    text += ` This text discusses ${topic} and its applications in modern technology.`;
    
    // Add random elaboration
    if (Math.random() > 0.4) {
      const connector = connectors[Math.floor(Math.random() * connectors.length)];
      const elaboration = elaborations[Math.floor(Math.random() * elaborations.length)];
      text += ` ${connector} ${elaboration}.`;
    }

    // Add some variation in length
    if (Math.random() > 0.6) {
      const secondTopic = topics[Math.floor(Math.random() * topics.length)];
      text += ` The relationship between ${topic} and ${secondTopic} presents interesting challenges for future research.`;
    }

    texts.push(text);

    if (i % 1000 === 0) {
      process.stdout.write(`\r📝 Generated: ${i}/${numSamples} samples`);
    }
  }

  await fs.writeFile(outputFile, texts.join('\n'));
  console.log(`\n✅ Generated ${texts.length} synthetic samples to ${outputFile}`);
}

async function main() {
  console.log('📚 MCP Titan Training Data Downloader');
  console.log('===================================\n');

  const args = process.argv.slice(2);
  const dataDir = process.env.DATA_DIR || 'data';

  // Create data directory
  await fs.mkdir(dataDir, { recursive: true });

  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
Usage: npm run download-data [options]

Options:
  --help, -h          Show this help message
  --synthetic         Generate synthetic training data only
  --wikitext          Download WikiText-2 dataset
  --tinystories       Download TinyStories dataset  
  --openwebtext       Download OpenWebText sample
  --all               Download all available datasets

Environment Variables:
  DATA_DIR            Directory to save data (default: data)

Examples:
  npm run download-data --synthetic
  npm run download-data --wikitext
  npm run download-data --all
  DATA_DIR=./my_data npm run download-data --synthetic
`);
    return;
  }

  try {
    if (args.includes('--synthetic') || args.length === 0) {
      await generateSyntheticData(path.join(dataDir, 'synthetic_training.txt'));
    }

    if (args.includes('--wikitext') || args.includes('--all')) {
      console.log('\n📥 Downloading WikiText-2...');
      const source = DATA_SOURCES.find(s => s.name === 'WikiText-2')!;
      await downloadFile(source.url, path.join(dataDir, 'wikitext-2.zip'));
      console.log('⚠️  Note: You\'ll need to extract the ZIP file manually');
    }

    if (args.includes('--tinystories') || args.includes('--all')) {
      console.log('\n📥 Downloading TinyStories (this may take a while)...');
      const source = DATA_SOURCES.find(s => s.name === 'TinyStories')!;
      await downloadFile(source.url, path.join(dataDir, source.outputFile));
    }

    if (args.includes('--openwebtext') || args.includes('--all')) {
      console.log('\n📥 Downloading OpenWebText sample...');
      const source = DATA_SOURCES.find(s => s.name === 'OpenWebText Sample')!;
      const tempFile = path.join(dataDir, 'temp_' + source.outputFile);
      await downloadFile(source.url, tempFile);
      await processJsonl(tempFile, path.join(dataDir, 'openwebtext_processed.txt'));
      await fs.unlink(tempFile);
    }

    console.log('\n🎉 Data preparation completed!');
    console.log('\n📁 Available data files:');
    
    const files = await fs.readdir(dataDir);
    for (const file of files) {
      const stats = await fs.stat(path.join(dataDir, file));
      const sizeMB = (stats.size / 1024 / 1024).toFixed(1);
      console.log(`  📄 ${file} (${sizeMB} MB)`);
    }

    console.log('\n📖 Next steps:');
    console.log('1. Choose a data file for training');
    console.log('2. Run: TRAINING_DATA_PATH=data/synthetic_training.txt npm run train-quick');
    console.log('3. Or run: npm run train-model for full training');

  } catch (error) {
    console.error('\n❌ Download failed:', error);
    process.exit(1);
  }
}

main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});