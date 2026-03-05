#!/usr/bin/env python3
"""Test Metacognition model after fix"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Metacognition model after fix...")
try:
    from core.models.metacognition.unified_metacognition_model import UnifiedMetacognitionModel
    
    # Initialize model
    config = {"from_scratch": True}
    model = UnifiedMetacognitionModel(config=config)
    print(f"✓ Model initialized: {model.model_id}")
    
    # Test strategy_selection with the exact parameters from validation
    print(f"\nTesting strategy_selection method with validation parameters...")
    try:
        result = model.strategy_selection(
            task_description='solve a complex problem',
            available_strategies=['divide_and_conquer', 'brute_force', 'heuristic']
        )
        print(f"✓ strategy_selection method call successful")
        print(f"  Result keys: {list(result.keys())}")
        print(f"  Success: {result.get('success', 'N/A')}")
        print(f"  Selection method: {result.get('selection_method', 'N/A')}")
    except Exception as e:
        print(f"✗ strategy_selection method call failed: {e}")
        import traceback
        traceback.print_exc()
        
    # Test apply_metacognition with the exact parameters from validation
    print(f"\nTesting apply_metacognition method with validation parameters...")
    try:
        result = model.apply_metacognition(
            cognitive_state={
                'task': 'learning new concept',
                'confidence': 0.7,
                'focus': 0.8
            }
        )
        print(f"✓ apply_metacognition method call successful")
        print(f"  Result keys: {list(result.keys())}")
        print(f"  Success: {result.get('success', 'N/A')}")
        print(f"  Failure message: {result.get('failure_message', 'N/A')}")
    except Exception as e:
        print(f"✗ apply_metacognition method call failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()