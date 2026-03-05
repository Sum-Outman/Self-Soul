#!/usr/bin/env python3
"""Test script to fix abstract method issue"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import abc
import logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise

def test_fix():
    """Test different fixes for abstract method issue"""
    from core.models.manager.unified_manager_model import UnifiedManagerModel
    
    print("Original __abstractmethods__:", UnifiedManagerModel.__abstractmethods__)
    
    # Try fix 1: Clear __abstractmethods__
    print("\n1. Clearing __abstractmethods__...")
    try:
        UnifiedManagerModel.__abstractmethods__ = frozenset()
        print("   ✓ __abstractmethods__ cleared")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Try fix 2: Use ABCMeta to register
    print("\n2. Using ABCMeta.register...")
    try:
        # Create a dummy class that implements _create_stream_processor
        class DummyManagerModel(UnifiedManagerModel):
            pass
        
        # Register with ABC
        abc.ABC.register(DummyManagerModel)
        print("   ✓ Registered with ABC")
        
        # Try to instantiate
        try:
            instance = DummyManagerModel(config={})
            print("   ✓ DummyManagerModel instantiated successfully")
        except Exception as e:
            print(f"   ✗ Instantiation failed: {e}")
            
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Try fix 3: Check method resolution
    print("\n3. Checking method resolution...")
    print(f"   _create_stream_processor in UnifiedManagerModel: {hasattr(UnifiedManagerModel, '_create_stream_processor')}")
    print(f"   _create_stream_processor in UnifiedManagerModel.__dict__: {'_create_stream_processor' in UnifiedManagerModel.__dict__}")
    
    # Get the method
    method = UnifiedManagerModel._create_stream_processor
    print(f"   Method type: {type(method)}")
    print(f"   Is abstractmethod: {isinstance(method, abc.abstractmethod)}")
    
    # Try fix 4: Create a non-abstract subclass
    print("\n4. Creating non-abstract subclass...")
    try:
        class ConcreteManagerModel(UnifiedManagerModel):
            __abstractmethods__ = frozenset()
        
        # Try to instantiate
        try:
            instance = ConcreteManagerModel(config={})
            print("   ✓ ConcreteManagerModel instantiated successfully")
        except Exception as e:
            print(f"   ✗ Instantiation failed: {e}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Try fix 5: Direct instantiation after fixes
    print("\n5. Direct instantiation attempt...")
    try:
        instance = UnifiedManagerModel(config={})
        print("   ✓ UnifiedManagerModel instantiated successfully")
    except Exception as e:
        print(f"   ✗ Instantiation failed: {e}")

if __name__ == "__main__":
    test_fix()