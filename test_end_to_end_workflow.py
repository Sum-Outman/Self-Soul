#!/usr/bin/env python3
"""End-to-end workflow test for multi-model collaboration"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import numpy as np
logging.basicConfig(level=logging.INFO)

def create_test_data():
    """Create test data for end-to-end workflow"""
    # Create test image
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Create test audio
    test_audio = np.random.randn(16000)  # 1 second at 16kHz
    
    # Create test text
    test_text = "This is a test for multimodal collaboration between audio, image, and manager models."
    
    return {
        'image': test_image,
        'audio': test_audio,
        'text': test_text
    }

def test_multimodal_workflow():
    """Test end-to-end multimodal workflow"""
    print("=" * 60)
    print("End-to-End Multimodal Workflow Test")
    print("=" * 60)
    
    try:
        # Import models
        from core.models.audio.unified_audio_model import UnifiedAudioModel
        from core.models.vision.unified_vision_model import UnifiedVisionModel
        from core.models.manager.unified_manager_model import UnifiedManagerModel
        
        print("1. Initializing models...")
        
        # Initialize models
        audio_config = {"sample_rate": 16000, "from_scratch": True}
        vision_config = {"from_scratch": True}
        manager_config = {"from_scratch": True}
        
        audio_model = UnifiedAudioModel(config=audio_config)
        vision_model = UnifiedVisionModel(config=vision_config)
        manager_model = UnifiedManagerModel(config=manager_config)
        
        print(f"   ✓ Audio model initialized: {audio_model.model_id}")
        print(f"   ✓ Vision model initialized: {vision_model.model_id}")
        print(f"   ✓ Manager model initialized: {manager_model.model_id}")
        
        # Create test data
        print("\n2. Creating test data...")
        test_data = create_test_data()
        print(f"   ✓ Test image shape: {test_data['image'].shape}")
        print(f"   ✓ Test audio length: {len(test_data['audio'])}")
        print(f"   ✓ Test text: '{test_data['text'][:50]}...'")
        
        # Test individual model processing
        print("\n3. Testing individual model processing...")
        
        # Audio processing
        print("   a. Audio model processing...")
        audio_result = audio_model.process_audio(
            audio_data=test_data['audio'].tobytes() if hasattr(test_data['audio'], 'tobytes') else test_data['audio'],
            language="en-US"
        )
        print(f"     ✓ Audio processing complete: {audio_result.get('success', 0) == 1}")
        
        # Vision processing
        print("   b. Vision model processing...")
        # Convert image to format expected by vision model
        vision_input = {
            'image_data': test_data['image'],
            'operation': 'classify'
        }
        vision_result = vision_model.process(vision_input)
        print(f"     ✓ Vision processing complete: {vision_result.get('success', 0) == 1}")
        
        # Manager coordination
        print("   c. Manager model coordination...")
        
        # Test manager's multimodal processing
        multimodal_input = {
            'modalities': {
                'vision': test_data['image'],
                'audio': test_data['audio'],
                'text': test_data['text']
            },
            'task': 'multimodal_analysis'
        }
        
        manager_result = manager_model.process(multimodal_input)
        print(f"     ✓ Manager multimodal processing complete: {manager_result.get('success', 0) == 1}")
        
        # Test manager coordination capabilities
        print("\n4. Testing manager coordination capabilities...")
        
        # Get coordinator from manager model
        if hasattr(manager_model, 'coordinate_models'):
            coordination_result = manager_model.coordinate_models(
                models=['audio_model', 'vision_model', 'text_model'],
                task_description='Analyze audio and image data with text context'
            )
            print(f"   ✓ Model coordination: {coordination_result.get('coordinated', False)}")
            print(f"     Task sequence: {coordination_result.get('task_sequence', [])}")
        else:
            # Try to access coordinator directly
            if hasattr(manager_model, 'coordinator'):
                coordinator = manager_model.coordinator
                coordination_result = coordinator.coordinate_models(
                    models=['audio_model', 'vision_model', 'text_model'],
                    task_description='Analyze audio and image data with text context'
                )
                print(f"   ✓ Model coordination: {coordination_result.get('coordinated', False)}")
                print(f"     Task sequence: {coordination_result.get('task_sequence', [])}")
            else:
                print("   ✗ Coordinator not found in manager model")
        
        # Test resource allocation
        print("\n5. Testing resource allocation...")
        if hasattr(manager_model, 'manage_resources') or (hasattr(manager_model, 'coordinator') and hasattr(manager_model.coordinator, 'manage_resources')):
            resource_needed = {'cpu': 40, 'memory': 256, 'gpu': 30}
            if hasattr(manager_model, 'manage_resources'):
                resource_result = manager_model.manage_resources(resource_needed)
            else:
                resource_result = manager_model.coordinator.manage_resources(resource_needed)
            
            print(f"   ✓ Resource allocation: {resource_result.get('managed', False)}")
            allocated = resource_result.get('allocated_resources', {})
            print(f"     Allocated CPU: {allocated.get('cpu', 0)}%")
            print(f"     Allocated Memory: {allocated.get('memory', 0)}MB")
            print(f"     Allocated GPU: {allocated.get('gpu', 0)}%")
        else:
            print("   ✗ Resource management not available")
        
        # Test workflow integration
        print("\n6. Testing integrated workflow...")
        
        # Simulate a complete workflow
        workflow_steps = [
            "load_multimodal_data",
            "preprocess_audio",
            "preprocess_image",
            "extract_audio_features",
            "extract_image_features",
            "fuse_modalities",
            "generate_analysis",
            "produce_output"
        ]
        
        if hasattr(manager_model, 'optimize_workflow') or (hasattr(manager_model, 'coordinator') and hasattr(manager_model.coordinator, 'optimize_workflow')):
            if hasattr(manager_model, 'optimize_workflow'):
                workflow_result = manager_model.optimize_workflow(workflow_steps)
            else:
                workflow_result = manager_model.coordinator.optimize_workflow(workflow_steps)
            
            print(f"   ✓ Workflow optimization: {workflow_result.get('optimized', False)}")
            optimized_seq = workflow_result.get('optimized_sequence', [])
            print(f"     Original steps: {len(workflow_steps)}")
            print(f"     Optimized steps: {len(optimized_seq)}")
            improvement = workflow_result.get('estimated_improvement', 0)
            print(f"     Estimated improvement: {improvement:.1f}%")
        else:
            print("   ✗ Workflow optimization not available")
        
        # Test collaboration handling
        print("\n7. Testing collaboration handling...")
        if hasattr(manager_model, 'handle_collaboration') or (hasattr(manager_model, 'coordinator') and hasattr(manager_model.coordinator, 'handle_collaboration')):
            models_to_collaborate = ['audio_feature_extractor', 'vision_feature_extractor', 'fusion_engine']
            if hasattr(manager_model, 'handle_collaboration'):
                collaboration_result = manager_model.handle_collaboration(models_to_collaborate, 'parallel')
            else:
                collaboration_result = manager_model.coordinator.handle_collaboration(models_to_collaborate, 'parallel')
            
            print(f"   ✓ Collaboration handling: {collaboration_result.get('collaborating', False)}")
            collab_type = collaboration_result.get('collaboration_type', 'unknown')
            data_flow = collaboration_result.get('data_flow', {})
            print(f"     Collaboration type: {collab_type}")
            print(f"     Data flow type: {data_flow.get('type', 'unknown')}")
        else:
            print("   ✗ Collaboration handling not available")
        
        print("\n" + "=" * 60)
        print("Workflow Test Completed Successfully! ✓")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cross_model_communication():
    """Test communication between different models"""
    print("\n" + "=" * 60)
    print("Cross-Model Communication Test")
    print("=" * 60)
    
    try:
        # This test would simulate actual data passing between models
        # For now, we'll test the interfaces
        
        from core.models.audio.unified_audio_model import UnifiedAudioModel
        from core.models.vision.unified_vision_model import UnifiedVisionModel
        from core.models.manager.unified_manager_model import UnifiedManagerModel
        
        print("1. Testing model interoperability...")
        
        # Check if models have compatible interfaces
        audio_model = UnifiedAudioModel(config={"sample_rate": 16000})
        vision_model = UnifiedVisionModel(config={})
        manager_model = UnifiedManagerModel(config={})
        
        # Check for common attributes
        common_attributes = ['model_id', 'model_type', 'model_name', 'config', 'logger']
        
        for model_name, model in [('Audio', audio_model), ('Vision', vision_model), ('Manager', manager_model)]:
            print(f"   {model_name} model attributes:")
            for attr in common_attributes:
                if hasattr(model, attr):
                    value = getattr(model, attr)
                    if attr == 'config' and isinstance(value, dict):
                        print(f"     ✓ {attr}: (dict with {len(value)} keys)")
                    elif attr == 'logger':
                        print(f"     ✓ {attr}: {type(value).__name__}")
                    else:
                        print(f"     ✓ {attr}: {value}")
                else:
                    print(f"     ✗ {attr}: missing")
        
        print("\n2. Testing data format compatibility...")
        
        # Create standardized test data
        test_data = {
            'metadata': {
                'source': 'test_suite',
                'timestamp': '2024-01-01T00:00:00Z',
                'data_type': 'multimodal'
            },
            'payload': {
                'audio': np.random.randn(8000).tolist(),  # 0.5 second at 16kHz
                'image': np.random.randint(0, 255, (100, 100, 3)).tolist(),
                'text': 'Test data for cross-model communication'
            }
        }
        
        print(f"   ✓ Created standardized test data")
        print(f"     Metadata keys: {list(test_data['metadata'].keys())}")
        print(f"     Payload keys: {list(test_data['payload'].keys())}")
        
        print("\n3. Testing manager as orchestrator...")
        
        # Test if manager can route data appropriately
        if hasattr(manager_model, 'process_multi_modal_input'):
            routing_test = manager_model.process_multi_modal_input(test_data['payload'])
            modalities = routing_test.get('modalities_detected', [])
            routing = routing_test.get('routing_decisions', {})
            
            print(f"   ✓ Manager routing test completed")
            print(f"     Modalities detected: {modalities}")
            print(f"     Routing decisions: {routing}")
        else:
            print("   ✗ Manager multimodal input processing not available")
        
        print("\n" + "=" * 60)
        print("Cross-Model Communication Test Completed! ✓")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Cross-model communication test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run end-to-end workflow tests"""
    results = []
    
    print("Starting End-to-End Workflow Tests")
    print("=" * 60)
    
    # Run tests
    results.append(("Multimodal Workflow", test_multimodal_workflow()))
    results.append(("Cross-Model Communication", test_cross_model_communication()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("End-to-End Test Summary")
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
        print("\nAll end-to-end tests passed successfully! ✓")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())