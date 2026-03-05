#!/usr/bin/env python3
"""Test script to check Manager model instantiation"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.INFO)

def test_manager_instantiation():
    """Test that Manager model can be instantiated"""
    print("Testing Manager model instantiation...")
    try:
        from core.models.manager.unified_manager_model import UnifiedManagerModel
        
        config = {"from_scratch": True}
        
        print("Attempting to instantiate UnifiedManagerModel...")
        model = UnifiedManagerModel(config=config)
        print(f"✓ Manager model instantiated successfully")
        print(f"  Model ID: {model.model_id}")
        print(f"  Model type: {model.model_type}")
        
        # Check if _create_stream_processor method exists
        if hasattr(model, '_create_stream_processor'):
            print(f"✓ _create_stream_processor method exists")
            
            # Try to create stream processor
            try:
                stream_processor = model._create_stream_processor()
                print(f"✓ Stream processor created successfully")
                print(f"  Processor type: {type(stream_processor).__name__}")
                
                # Check if it has required methods
                required_methods = ['_initialize_pipeline', 'process_frame']
                for method in required_methods:
                    if hasattr(stream_processor, method):
                        print(f"✓ Stream processor has {method} method")
                    else:
                        print(f"✗ Stream processor missing {method} method")
                        
            except Exception as e:
                print(f"✗ Failed to create stream processor: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("✗ _create_stream_processor method not found")
        
        return True
        
    except Exception as e:
        print(f"✗ Manager model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_abstract_methods():
    """Test abstract methods of UnifiedManagerModel"""
    print("\nTesting abstract methods...")
    try:
        import abc
        from core.models.manager.unified_manager_model import UnifiedManagerModel
        
        # Get all abstract methods
        abstract_methods = set()
        for cls in UnifiedManagerModel.__mro__:
            if hasattr(cls, '__abstractmethods__'):
                abstract_methods.update(cls.__abstractmethods__)
        
        print(f"Abstract methods found: {abstract_methods}")
        
        # Check which methods are implemented
        for method in abstract_methods:
            if hasattr(UnifiedManagerModel, method):
                print(f"✓ {method} is implemented")
            else:
                print(f"✗ {method} is NOT implemented")
        
        return True
        
    except Exception as e:
        print(f"✗ Abstract methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run tests"""
    print("=" * 60)
    print("Manager Model Instantiation Test")
    print("=" * 60)
    
    results = []
    results.append(("Manager Instantiation", test_manager_instantiation()))
    results.append(("Abstract Methods", test_abstract_methods()))
    
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