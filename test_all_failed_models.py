#!/usr/bin/env python3
"""Test all failed models to identify validation issues"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_validation(model_name, module_path, class_name, test_cases):
    """Test a model and compare with validation expectations"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    try:
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)
        
        # Initialize model
        config = {"from_scratch": True}
        model = model_class(config=config)
        print(f"✓ Model initialized: {model.model_id}")
        
        # Test each test case
        for i, test_case in enumerate(test_cases):
            print(f"\n  Test {i+1}: {test_case['name']}")
            print(f"    Operation: {test_case['operation']}")
            print(f"    Expected keys: {test_case['expected_keys']}")
            
            try:
                # Get method
                if hasattr(model, test_case['operation']):
                    method = getattr(model, test_case['operation'])
                    # Call method
                    result = method(**test_case['data'])
                    
                    print(f"    Result keys: {list(result.keys())}")
                    print(f"    Result type: {type(result)}")
                    
                    # Check for success/status
                    if 'success' in result:
                        print(f"    Success value: {result['success']} (type: {type(result['success'])})")
                    if 'status' in result:
                        print(f"    Status value: {result['status']} (type: {type(result['status'])})")
                    
                    # Check expected keys
                    missing_keys = [key for key in test_case['expected_keys'] if key not in result]
                    if missing_keys:
                        print(f"    ✗ Missing keys: {missing_keys}")
                    else:
                        print(f"    ✓ All expected keys present")
                        
                    # Check if operation actually succeeded
                    has_failure = False
                    if 'success' in result:
                        success_val = result['success']
                        if isinstance(success_val, bool) and not success_val:
                            has_failure = True
                            print(f"    ⚠ Success is False (boolean)")
                        elif isinstance(success_val, int) and success_val == 0:
                            has_failure = True
                            print(f"    ⚠ Success is 0 (integer)")
                        elif isinstance(success_val, str) and success_val.lower() in ['false', 'error', 'fail']:
                            has_failure = True
                            print(f"    ⚠ Success indicates failure: {success_val}")
                    
                    if 'status' in result and isinstance(result['status'], str):
                        status_lower = result['status'].lower()
                        if status_lower in ['error', 'fail', 'failure', 'failed']:
                            has_failure = True
                            print(f"    ⚠ Status indicates failure: {result['status']}")
                    
                    if has_failure:
                        print(f"    ⚠ Operation may have failed despite having expected keys")
                        print(f"    Failure message: {result.get('error', result.get('failure_message', 'No error message'))}")
                    
                else:
                    print(f"    ✗ Method not found: {test_case['operation']}")
                    
            except Exception as e:
                print(f"    ✗ Test execution error: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Testing failed models to identify validation issues...")
    
    # Math Model test cases
    math_test_cases = [
        {
            'name': 'Expression evaluation',
            'operation': 'evaluate_expression',
            'data': {
                'expression': '2 + 3 * 5',
                'variables': None
            },
            'expected_keys': ['status', 'result']
        },
        {
            'name': 'Equation solving',
            'operation': 'solve_equation',
            'data': {
                'equation': '2*x + 5 = 15',
                'variable': 'x'
            },
            'expected_keys': ['status', 'message']
        }
    ]
    
    # Planning Model test cases
    planning_test_cases = [
        {
            'name': 'Create simple plan',
            'operation': 'create_plan',
            'data': {
                'goal': 'organize a meeting',
                'available_models': ['manager', 'language', 'knowledge']
            },
            'expected_keys': ['status']
        },
        {
            'name': 'Analyze goal complexity',
            'operation': 'analyze_goal_complexity',
            'data': {
                'goal': 'implement a new feature'
            },
            'expected_keys': ['score', 'level']
        }
    ]
    
    # Metacognition Model test cases
    metacognition_test_cases = [
        {
            'name': 'Apply metacognition',
            'operation': 'apply_metacognition',
            'data': {
                'cognitive_state': {
                    'task': 'learning new concept',
                    'confidence': 0.7,
                    'focus': 0.8
                }
            },
            'expected_keys': ['success', 'failure_message']
        },
        {
            'name': 'Strategy selection',
            'operation': 'strategy_selection',
            'data': {
                'task_description': 'solve a complex problem',
                'available_strategies': ['divide_and_conquer', 'brute_force', 'heuristic']
            },
            'expected_keys': ['success', 'selection_method', 'selection_timestamp']
        }
    ]
    
    # Test each model
    test_model_validation(
        "Math Model",
        "core.models.mathematics.unified_mathematics_model",
        "UnifiedMathematicsModel",
        math_test_cases
    )
    
    test_model_validation(
        "Planning Model",
        "core.models.planning.unified_planning_model",
        "UnifiedPlanningModel",
        planning_test_cases
    )
    
    test_model_validation(
        "Metacognition Model",
        "core.models.metacognition.unified_metacognition_model",
        "UnifiedMetacognitionModel",
        metacognition_test_cases
    )
    
    print(f"\n{'='*60}")
    print("Testing complete")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()