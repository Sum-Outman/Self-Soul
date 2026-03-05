#!/usr/bin/env python3
"""Test method resolution issue"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.WARNING)

def test_method_exists():
    """Test if _initialize_minimal_components method exists in UnifiedManagerModel"""
    print("Testing method existence...")
    
    try:
        from core.models.manager.unified_manager_model import UnifiedManagerModel
        
        # Check if method exists in class
        print(f"_initialize_minimal_components in class: {hasattr(UnifiedManagerModel, '_initialize_minimal_components')}")
        
        # Check if method exists in class dict (directly defined)
        print(f"_initialize_minimal_components in __dict__: {'_initialize_minimal_components' in UnifiedManagerModel.__dict__}")
        
        # Check method signature
        if hasattr(UnifiedManagerModel, '_initialize_minimal_components'):
            method = UnifiedManagerModel._initialize_minimal_components
            print(f"Method: {method}")
            print(f"Method source file: {method.__code__.co_filename if hasattr(method, '__code__') else 'N/A'}")
            print(f"Method line number: {method.__code__.co_firstlineno if hasattr(method, '__code__') else 'N/A'}")
        
        # Check inheritance chain
        print(f"\nMRO (Method Resolution Order):")
        for i, cls in enumerate(UnifiedManagerModel.__mro__):
            print(f"  {i}: {cls.__name__}")
            if hasattr(cls, '_initialize_minimal_components'):
                print(f"     -> has _initialize_minimal_components")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_instantiation_with_fix():
    """Test instantiation with potential fix"""
    print("\nTesting instantiation with fix...")
    
    try:
        # First patch the template to remove abstract method issue
        import core.models.unified_model_template as template_module
        from core.models.unified_model_template import UnifiedModelTemplate
        
        # Remove _create_stream_processor from abstract methods
        if hasattr(UnifiedModelTemplate, '__abstractmethods__'):
            abstract_methods = set(UnifiedModelTemplate.__abstractmethods__)
            abstract_methods.discard('_create_stream_processor')
            UnifiedModelTemplate.__abstractmethods__ = frozenset(abstract_methods)
            print(f"Removed _create_stream_processor from template abstract methods")
        
        # Now test UnifiedManagerModel
        from core.models.manager.unified_manager_model import UnifiedManagerModel
        
        # Ensure UnifiedManagerModel is not abstract
        UnifiedManagerModel.__abstractmethods__ = frozenset()
        
        # Try to create instance
        config = {"from_scratch": True}
        print("Attempting to instantiate...")
        model = UnifiedManagerModel(config=config)
        print(f"✓ Success! Model ID: {model.model_id}")
        
        # Test the method
        print(f"Testing _initialize_minimal_components...")
        model._initialize_minimal_components(config)
        print(f"✓ Method works!")
        
        return True
        
    except Exception as e:
        print(f"✗ Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run tests"""
    print("=" * 60)
    print("Method Resolution Test")
    print("=" * 60)
    
    # Test method existence
    test1 = test_method_exists()
    
    # Test instantiation
    test2 = test_instantiation_with_fix()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    if test1 and test2:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())