#!/usr/bin/env python3
"""Test Metacognition model API directly"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Metacognition model API directly...")
try:
    from core.models.metacognition.unified_metacognition_model import UnifiedMetacognitionModel
    
    # Initialize model
    config = {"from_scratch": True}
    model = UnifiedMetacognitionModel(config=config)
    print(f"✓ Model initialized: {model.model_id}")
    
    # Check if methods exist
    print(f"\nChecking API methods:")
    methods_to_check = [
        'apply_metacognition',
        'strategy_selection',
        'self_monitoring',
        'error_detection',
        'learning_optimization',
        'reflection'
    ]
    
    for method in methods_to_check:
        print(f"  hasattr(model, '{method}'): {hasattr(model, method)}")
    
    print(f"  hasattr(model, '_process_operation'): {hasattr(model, '_process_operation')}")
    
    # Try to call apply_metacognition method
    if hasattr(model, 'apply_metacognition'):
        print(f"\nTesting apply_metacognition method...")
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
            print(f"  Success/Status: {result.get('success', result.get('status', 'N/A'))}")
            print(f"  Adjusted state: {result.get('adjusted_state', 'N/A')}")
        except Exception as e:
            print(f"✗ apply_metacognition method call failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n✗ apply_metacognition method not found on model instance")
        
    # Try to call strategy_selection method
    if hasattr(model, 'strategy_selection'):
        print(f"\nTesting strategy_selection method...")
        try:
            result = model.strategy_selection(
                task_description='solve a complex problem',
                available_strategies=['divide_and_conquer', 'brute_force', 'heuristic']
            )
            print(f"✓ strategy_selection method call successful")
            print(f"  Result keys: {list(result.keys())}")
            print(f"  Success/Status: {result.get('success', result.get('status', 'N/A'))}")
            print(f"  Selected strategy: {result.get('selected_strategy', 'N/A')}")
        except Exception as e:
            print(f"✗ strategy_selection method call failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n✗ strategy_selection method not found on model instance")
        
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()