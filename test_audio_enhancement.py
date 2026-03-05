#!/usr/bin/env python3
"""Test script to verify enhanced audio library support"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.INFO)

def test_audio_model_instantiation():
    """Test that Audio model can be instantiated"""
    print("Testing Audio model instantiation...")
    try:
        from core.models.audio.unified_audio_model import UnifiedAudioModel
        
        config = {
            "sample_rate": 16000,
            "from_scratch": True
        }
        
        model = UnifiedAudioModel(config=config)
        print(f"✓ Audio model instantiated successfully")
        print(f"  Model ID: {model.model_id}")
        print(f"  Model type: {model.model_type}")
        
        # Test audio library support
        if hasattr(model, 'librosa_integration'):
            print(f"✓ Librosa integration: {model.librosa_integration}")
        else:
            print("✗ Librosa integration attribute missing")
            
        if hasattr(model, 'torchaudio_integration'):
            print(f"✓ Torchaudio integration: {model.torchaudio_integration}")
        else:
            print("✗ Torchaudio integration attribute missing")
            
        return True
        
    except Exception as e:
        print(f"✗ Audio model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_feature_extraction():
    """Test audio feature extraction functionality"""
    print("\nTesting audio feature extraction...")
    try:
        import numpy as np
        
        # Create test audio data
        test_audio = np.random.randn(16000)  # 1 second at 16kHz
        
        from core.models.audio.unified_audio_model import UnifiedAudioModel
        
        config = {"sample_rate": 16000, "from_scratch": True}
        model = UnifiedAudioModel(config=config)
        
        # Test MFCC extractor
        if hasattr(model, 'mfcc_extractor') and model.mfcc_extractor is not None:
            mfcc = model.mfcc_extractor(test_audio, 16000)
            if mfcc is not None:
                print(f"✓ MFCC extraction successful, shape: {mfcc.shape}")
            else:
                print("✗ MFCC extraction returned None")
        else:
            print("✗ MFCC extractor not available")
        
        # Test other audio processors
        audio_attributes = ['spectrogram_processor', 'mel_spectrogram', 'stft_processor']
        for attr in audio_attributes:
            if hasattr(model, attr) and getattr(model, attr) is not None:
                print(f"✓ {attr} available")
            else:
                print(f"✗ {attr} not available")
                
        return True
        
    except Exception as e:
        print(f"✗ Audio feature extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_model_enhancements():
    """Test enhanced image processing functionality"""
    print("\nTesting enhanced image processing...")
    try:
        from core.models.vision.unified_vision_model import (
            SimpleImagePreprocessor, SimpleAugmentationPipeline,
            SimpleTransformPipeline, SimpleNormalization,
            SimpleResizeOperation, SimpleCropOperation
        )
        
        import numpy as np
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test SimpleImagePreprocessor
        preprocessor = SimpleImagePreprocessor()
        processed = preprocessor.preprocess(test_image)
        print(f"✓ Image preprocessing successful")
        print(f"  Input shape: {test_image.shape}, Output shape: {processed.shape}")
        
        # Test SimpleAugmentationPipeline
        augmenter = SimpleAugmentationPipeline()
        augmented = augmenter.augment(test_image)
        print(f"✓ Image augmentation successful")
        
        # Test SimpleTransformPipeline
        transformer = SimpleTransformPipeline()
        transformed = transformer.transform(test_image)
        print(f"✓ Image transformation successful")
        
        # Test SimpleNormalization
        normalizer = SimpleNormalization()
        normalized = normalizer.normalize(test_image)
        print(f"✓ Image normalization successful")
        
        # Test SimpleResizeOperation
        resizer = SimpleResizeOperation()
        resized = resizer.resize(test_image, (224, 224))
        print(f"✓ Image resizing successful")
        print(f"  Resized shape: {resized.shape}")
        
        # Test SimpleCropOperation
        cropper = SimpleCropOperation()
        cropped = cropper.crop(test_image, [10, 10, 50, 50])
        print(f"✓ Image cropping successful")
        print(f"  Cropped shape: {cropped.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Image processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_manager_coordination():
    """Test enhanced manager coordination capabilities"""
    print("\nTesting enhanced manager coordination...")
    try:
        from core.models.manager.unified_manager_model import (
            SimpleMultimodalProcessor, SimpleVisionProcessor,
            SimpleAudioProcessor, SimpleTextProcessor,
            SimpleCoordinator
        )
        
        # Test SimpleMultimodalProcessor
        multimodal = SimpleMultimodalProcessor()
        modalities = {
            'vision': [[1, 2, 3], [4, 5, 6]],
            'audio': [0.1, 0.2, 0.3, 0.4],
            'text': 'This is a test text for multimodal processing',
            'sensor': {'temperature': 25.5, 'humidity': 60}
        }
        result = multimodal.process(modalities)
        print(f"✓ Multimodal processing successful")
        print(f"  Modalities processed: {result.get('modalities_processed', [])}")
        
        # Test SimpleVisionProcessor
        import numpy as np
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        vision = SimpleVisionProcessor()
        vision_result = vision.process(test_image)
        print(f"✓ Vision processing successful")
        print(f"  Features extracted: {len(vision_result.get('features', []))}")
        
        # Test SimpleAudioProcessor
        test_audio = np.random.randn(16000)
        audio = SimpleAudioProcessor()
        audio_result = audio.process(test_audio)
        print(f"✓ Audio processing successful")
        print(f"  Audio length: {audio_result.get('audio_length')}")
        
        # Test SimpleTextProcessor
        text = SimpleTextProcessor()
        text_result = text.process('This is a positive and excellent test!')
        print(f"✓ Text processing successful")
        print(f"  Sentiment: {text_result.get('sentiment')}")
        print(f"  Keywords: {text_result.get('keywords')}")
        
        # Test SimpleCoordinator
        coordinator = SimpleCoordinator()
        
        # Test coordinate_models
        coordination_result = coordinator.coordinate_models(
            ['vision_model', 'audio_model', 'text_model'],
            'Process image and audio for multimodal analysis'
        )
        print(f"✓ Model coordination successful")
        print(f"  Task sequence: {coordination_result.get('task_sequence', [])}")
        
        # Test allocate_tasks
        allocation_result = coordinator.allocate_tasks(
            ['task1', 'task2', 'task3', 'task4'],
            ['model_a', 'model_b']
        )
        print(f"✓ Task allocation successful")
        print(f"  Assignments: {allocation_result.get('assignments', {})}")
        
        # Test manage_resources
        resource_result = coordinator.manage_resources({
            'cpu': 30,
            'memory': 512,
            'gpu': 50
        })
        print(f"✓ Resource management successful")
        print(f"  Allocated resources: {resource_result.get('allocated_resources', {})}")
        
        # Test monitor_performance
        performance_result = coordinator.monitor_performance(['model_x', 'model_y', 'model_z'])
        print(f"✓ Performance monitoring successful")
        print(f"  Recommendations: {performance_result.get('recommendations', [])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Manager coordination test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Enhanced Functionalities")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Audio Model Instantiation", test_audio_model_instantiation()))
    results.append(("Audio Feature Extraction", test_audio_feature_extraction()))
    results.append(("Image Processing Enhancements", test_image_model_enhancements()))
    results.append(("Manager Coordination", test_manager_coordination()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    for test_name, success in results:
        total += 1
        if success:
            passed += 1
            print(f"✓ {test_name}: PASSED")
        else:
            print(f"✗ {test_name}: FAILED")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\nAll tests passed successfully! ✓")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())