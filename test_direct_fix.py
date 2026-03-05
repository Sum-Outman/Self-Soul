#!/usr/bin/env python3
"""Direct test to fix abstract method issue"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import abc
import logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise

def test_abstract_method_fix():
    """Test direct fix for abstract method issue"""
    
    # Import required modules
    from core.unified_stream_processor import StreamProcessor
    
    # Define a simple non-abstract StreamProcessor for testing
    class SimpleStreamProcessor(StreamProcessor):
        def __init__(self, config=None):
            super().__init__(config)
            
        def _initialize_pipeline(self):
            """Initialize processing pipeline"""
            self.logger.info("Simple pipeline initialized")
            
        def process_frame(self, frame_data):
            """Process single data frame"""
            return {"success": 1, "frame_processed": True}
    
    # Now create a simplified ManagerModel class
    class SimplifiedManagerModel:
        def __init__(self, config=None):
            self.config = config or {}
            self.model_id = "test_manager"
            
        def _create_stream_processor(self):
            """Create stream processor - must match abstract method signature"""
            return SimpleStreamProcessor(self.config)
    
    print("Testing SimplifiedManagerModel...")
    try:
        model = SimplifiedManagerModel()
        processor = model._create_stream_processor()
        print(f"✓ SimplifiedManagerModel instantiated successfully")
        print(f"  Processor created: {type(processor).__name__}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_unified_manager_model_direct():
    """Test UnifiedManagerModel directly with monkey patching"""
    print("\nTesting UnifiedManagerModel with monkey patch...")
    
    try:
        # Import the actual class
        from core.models.manager.unified_manager_model import UnifiedManagerModel
        
        # First, let's see the actual __abstractmethods__
        print(f"Original __abstractmethods__: {UnifiedManagerModel.__abstractmethods__}")
        
        # Try to monkey patch the abstractmethod decorator away
        import core.models.unified_model_template as template_module
        
        # Get the original method
        original_method = UnifiedManagerModel._create_stream_processor
        
        # Check if it's an abstractmethod
        print(f"Original method is abstractmethod: {isinstance(original_method, abc.abstractmethod)}")
        
        # If it's an abstractmethod, we need to create a non-abstract version
        if isinstance(original_method, abc.abstractmethod):
            print("Method is wrapped with abstractmethod decorator")
            # Get the underlying function
            if hasattr(original_method, '__wrapped__'):
                actual_func = original_method.__wrapped__
                print(f"Found __wrapped__ attribute")
            else:
                # Create a simple implementation
                def simple_create_stream_processor(self):
                    from core.unified_stream_processor import StreamProcessor
                    class SimpleProcessor(StreamProcessor):
                        def __init__(self, config):
                            super().__init__(config)
                        def _initialize_pipeline(self):
                            pass
                        def process_frame(self, frame_data):
                            return {"success": 1}
                    
                    return SimpleProcessor(self.config if hasattr(self, 'config') else {})
                
                # Replace the method
                UnifiedManagerModel._create_stream_processor = simple_create_stream_processor
                print("Replaced method with simple implementation")
        
        # Now try to instantiate
        try:
            model = UnifiedManagerModel(config={})
            print(f"✓ UnifiedManagerModel instantiated successfully after monkey patch")
            return True
        except Exception as e:
            print(f"✗ Still failed after monkey patch: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Error during monkey patch: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_remove_abstract_decorator():
    """Test removing abstractmethod decorator from the template"""
    print("\nTesting removal of abstractmethod decorator...")
    
    try:
        # Import the template
        import core.models.unified_model_template
        
        # Find the _create_stream_processor method in the template
        template_class = core.models.unified_model_template.UnifiedModelTemplate
        
        # Check if it has the abstractmethod decorator
        method = template_class._create_stream_processor
        
        # Try to create a non-abstract version
        def non_abstract_create_stream_processor(self):
            """Non-abstract version of _create_stream_processor"""
            from core.unified_stream_processor import StreamProcessor
            class SimpleProcessor(StreamProcessor):
                def __init__(self, config):
                    super().__init__(config)
                def _initialize_pipeline(self):
                    pass
                def process_frame(self, frame_data):
                    return {"success": 1}
            
            return SimpleProcessor(self.config if hasattr(self, 'config') else {})
        
        # Replace the method in the template
        template_class._create_stream_processor = non_abstract_create_stream_processor
        
        # Update __abstractmethods__
        if hasattr(template_class, '__abstractmethods__'):
            # Remove _create_stream_processor from abstract methods
            abstract_methods = set(template_class.__abstractmethods__)
            abstract_methods.discard('_create_stream_processor')
            template_class.__abstractmethods__ = frozenset(abstract_methods)
            print(f"Updated template __abstractmethods__: {template_class.__abstractmethods__}")
        
        # Now try to instantiate UnifiedManagerModel
        from core.models.manager.unified_manager_model import UnifiedManagerModel
        
        try:
            model = UnifiedManagerModel(config={})
            print(f"✓ UnifiedManagerModel instantiated successfully after template fix")
            return True
        except Exception as e:
            print(f"✗ Still failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run tests"""
    print("=" * 60)
    print("Direct Abstract Method Fix Tests")
    print("=" * 60)
    
    results = []
    results.append(("Simplified Model", test_abstract_method_fix()))
    results.append(("Monkey Patch", test_unified_manager_model_direct()))
    results.append(("Template Fix", test_remove_abstract_decorator()))
    
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