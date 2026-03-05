#!/usr/bin/env python3
"""Test Math model API directly"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Math model API directly...")
try:
    from core.models.mathematics.unified_mathematics_model import UnifiedMathematicsModel
    
    # Initialize model
    config = {"from_scratch": True}
    model = UnifiedMathematicsModel(config=config)
    print(f"✓ Model initialized: {model.model_id}")
    
    # Check if methods exist
    print(f"\nChecking API methods:")
    print(f"  hasattr(model, 'evaluate_expression'): {hasattr(model, 'evaluate_expression')}")
    print(f"  hasattr(model, 'solve_equation'): {hasattr(model, 'solve_equation')}")
    print(f"  hasattr(model, '_process_operation'): {hasattr(model, '_process_operation')}")
    
    # Try to call evaluate_expression method
    if hasattr(model, 'evaluate_expression'):
        print(f"\nTesting evaluate_expression method...")
        try:
            result = model.evaluate_expression(
                expression="2 + 3 * 5",
                variables=None
            )
            print(f"✓ evaluate_expression method call successful")
            print(f"  Result keys: {list(result.keys())}")
            print(f"  Success: {result.get('success', 'N/A')}")
            print(f"  Result: {result.get('result', 'N/A')}")
        except Exception as e:
            print(f"✗ evaluate_expression method call failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n✗ evaluate_expression method not found on model instance")
        
    # Try to call solve_equation method
    if hasattr(model, 'solve_equation'):
        print(f"\nTesting solve_equation method...")
        try:
            result = model.solve_equation(
                equation="2*x + 5 = 15",
                variable="x"
            )
            print(f"✓ solve_equation method call successful")
            print(f"  Result keys: {list(result.keys())}")
            print(f"  Success: {result.get('success', 'N/A')}")
            print(f"  Solutions: {result.get('solutions', 'N/A')}")
        except Exception as e:
            print(f"✗ solve_equation method call failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n✗ solve_equation method not found on model instance")
        
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()