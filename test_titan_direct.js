import { TitanMemoryServer } from './dist/index.js';

async function testTitanMemory() {
  console.log('🚀 Starting Titan Memory Server test...');

  try {
    // Create server instance
    const server = new TitanMemoryServer();
    console.log('✅ Server instance created');

    // Test model initialization by calling the init method directly
    const model = server.model || new (await import('./dist/model.js')).TitanMemoryModel();
    console.log('✅ Model imported');

    // Initialize with basic config
    const config = {
      inputDim: 768,
      hiddenDim: 512,
      memoryDim: 1024,
      transformerLayers: 4,
      numHeads: 8,
      ffDimension: 2048,
      dropoutRate: 0.1,
      maxSequenceLength: 256,
      memorySlots: 1000,
      similarityThreshold: 0.65,
      surpriseDecay: 0.9,
      pruningInterval: 1000,
      gradientClip: 1.0,
    };

    console.log('⏳ Initializing model...');
    await model.initialize(config);
    console.log('✅ Model initialized successfully');

    // Test text encoding
    console.log('⏳ Testing text encoding...');
    const testText = 'Hello, this is a test of the Titan Memory System.';
    const encoded = await model.encodeText(testText);
    console.log('✅ Text encoded successfully, shape:', encoded.shape);

    // Test memory operations
    console.log('⏳ Testing memory storage...');
    const memoryResult = await model.storeMemory(testText);
    console.log('✅ Memory stored successfully');

    // Test memory retrieval
    console.log('⏳ Testing memory retrieval...');
    const retrievedMemories = await model.retrieveMemories(encoded, 3);
    console.log('✅ Memories retrieved:', retrievedMemories.length, 'items');

    // Test get memory state
    console.log('⏳ Testing memory state...');
    const memoryState = await model.getMemoryState();
    console.log('✅ Memory state retrieved:', {
      capacity: memoryState.capacity,
      status: memoryState.status,
    });

    console.log('🎉 All tests passed! Titan Memory Server is working correctly.');

    // Cleanup
    encoded.dispose();
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error('Stack:', error.stack);
  }
}

testTitanMemory().catch(console.error);
