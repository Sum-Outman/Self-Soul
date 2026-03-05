#!/usr/bin/env python3
"""Test Planning model API directly"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Planning model API directly...")
try:
    from core.models.planning.unified_planning_model import UnifiedPlanningModel
    
    # Initialize model
    config = {"from_scratch": True}
    model = UnifiedPlanningModel(config=config)
    print(f"✓ Model initialized: {model.model_id}")
    
    # Check if methods exist
    print(f"\nChecking API methods:")
    methods_to_check = [
        'create_plan',
        'analyze_goal_complexity',
        'generate_and_plan',
        'monitor_execution',
        'adjust_plan',
        'execute_autonomous_plan'
    ]
    
    for method in methods_to_check:
        print(f"  hasattr(model, '{method}'): {hasattr(model, method)}")
    
    print(f"  hasattr(model, '_process_operation'): {hasattr(model, '_process_operation')}")
    
    # Try to call create_plan method
    if hasattr(model, 'create_plan'):
        print(f"\nTesting create_plan method...")
        try:
            result = model.create_plan(
                goal='organize a meeting',
                available_models=['manager', 'language', 'knowledge']
            )
            print(f"✓ create_plan method call successful")
            print(f"  Result keys: {list(result.keys())}")
            print(f"  Success/Status: {result.get('success', result.get('status', 'N/A'))}")
            print(f"  Plan: {result.get('plan', 'N/A')}")
        except Exception as e:
            print(f"✗ create_plan method call failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n✗ create_plan method not found on model instance")
        
    # Try to call analyze_goal_complexity method
    if hasattr(model, 'analyze_goal_complexity'):
        print(f"\nTesting analyze_goal_complexity method...")
        try:
            result = model.analyze_goal_complexity(
                goal='implement a new feature'
            )
            print(f"✓ analyze_goal_complexity method call successful")
            print(f"  Result keys: {list(result.keys())}")
            print(f"  Success/Status: {result.get('success', result.get('status', 'N/A'))}")
            print(f"  Complexity: {result.get('complexity', 'N/A')}")
        except Exception as e:
            print(f"✗ analyze_goal_complexity method call failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n✗ analyze_goal_complexity method not found on model instance")
        
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()