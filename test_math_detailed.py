#!/usr/bin/env python3
"""Detailed test for Math Model to diagnose validation issues"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_math_model():
    """Test Math Model with detailed error reporting"""
    print("Testing Math Model...")
    
    try:
        from core.models.mathematics.unified_mathematics_model import UnifiedMathematicsModel
        
        # Initialize model
        config = {"from_scratch": True}
        model = UnifiedMathematicsModel(config=config)
        print(f"✓ Model initialized: {model.model_id}")
        
        # Test 1: evaluate_expression
        print("\nTest 1: evaluate_expression")
        print("  Input: expression='2 + 3 * 5', variables=None")
        try:
            result = model.evaluate_expression(expression='2 + 3 * 5', variables=None)
            print(f"  Result: {result}")
            print(f"  Result type: {type(result)}")
            print(f"  Keys in result: {list(result.keys())}")
            
            # Check for expected keys
            expected_keys = ['status', 'result']
            missing_keys = [key for key in expected_keys if key not in result]
            if missing_keys:
                print(f"  ✗ Missing keys: {missing_keys}")
            else:
                print(f"  ✓ All expected keys present")
            
            # Check success
            if 'success' in result:
                success_val = result['success']
                print(f"  Success key present: {success_val} (type: {type(success_val)})")
                if isinstance(success_val, int) and success_val == 0:
                    print(f"  ⚠ Success is 0 (may indicate failure)")
                elif isinstance(success_val, bool) and not success_val:
                    print(f"  ⚠ Success is False")
            else:
                print(f"  ⚠ No success key in result")
            
            # Check status
            if 'status' in result:
                status_val = result['status']
                print(f"  Status: {status_val}")
                if isinstance(status_val, str) and status_val.lower() in ['error', 'fail']:
                    print(f"  ⚠ Status indicates failure: {status_val}")
            else:
                print(f"  ⚠ No status key in result")
                
        except Exception as e:
            print(f"  ✗ Test execution error: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 2: solve_equation
        print("\nTest 2: solve_equation")
        print("  Input: equation='2*x + 5 = 15', variable='x'")
        try:
            result = model.solve_equation(equation='2*x + 5 = 15', variable='x')
            print(f"  Result: {result}")
            print(f"  Result type: {type(result)}")
            print(f"  Keys in result: {list(result.keys())}")
            
            # Check for expected keys (changed from ['status', 'message'] to ['status', 'solutions'])
            expected_keys = ['status', 'solutions']
            missing_keys = [key for key in expected_keys if key not in result]
            if missing_keys:
                print(f"  ✗ Missing keys: {missing_keys}")
                print(f"  Actual keys: {list(result.keys())}")
            else:
                print(f"  ✓ All expected keys present")
            
            # Check success
            if 'success' in result:
                success_val = result['success']
                print(f"  Success key present: {success_val} (type: {type(success_val)})")
                if isinstance(success_val, int) and success_val == 0:
                    print(f"  ⚠ Success is 0 (may indicate failure)")
                elif isinstance(success_val, bool) and not success_val:
                    print(f"  ⚠ Success is False")
            else:
                print(f"  ⚠ No success key in result")
            
            # Check status
            if 'status' in result:
                status_val = result['status']
                print(f"  Status: {status_val}")
                if isinstance(status_val, str) and status_val.lower() in ['error', 'fail']:
                    print(f"  ⚠ Status indicates failure: {status_val}")
            else:
                print(f"  ⚠ No status key in result")
                
            # Check solutions if present
            if 'solutions' in result:
                solutions = result['solutions']
                print(f"  Solutions: {solutions} (type: {type(solutions)})")
                
        except Exception as e:
            print(f"  ✗ Test execution error: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"✗ Math Model test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_math_model()