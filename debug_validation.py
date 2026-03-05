#!/usr/bin/env python3
"""Debug validation failures for Math, Planning, and Metacognition models"""

import sys
import os
import traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robust_model_validation import ModelValidator, create_math_test_cases, create_planning_test_cases, create_metacognition_test_cases

def debug_model_validation(model_name, module_path, class_name, test_cases_func):
    """Debug validation for a specific model"""
    print(f"\n{'='*60}")
    print(f"Debugging: {model_name}")
    print(f"{'='*60}")
    
    # Create validator
    config = {
        'max_retries': 2,
        'timeout_seconds': 30,
        'skip_on_error': False,
        'detailed_logging': True
    }
    validator = ModelValidator(config)
    
    # Create model spec
    model_spec = {
        'name': model_name,
        'module_path': module_path,
        'class_name': class_name,
        'test_cases': test_cases_func()
    }
    
    # Run validation
    result = validator.run_comprehensive_validation(model_spec)
    
    # Print detailed results
    print(f"\nImport Result: {result.get('import_result', {}).get('success', 'N/A')}")
    if result.get('import_result', {}).get('errors'):
        print(f"  Import Errors: {result['import_result']['errors']}")
    
    print(f"\nInstantiation Result: {result.get('instantiation_result', {}).get('success', 'N/A')}")
    if result.get('instantiation_result', {}).get('errors'):
        print(f"  Instantiation Errors: {result['instantiation_result']['errors']}")
    
    if result.get('functionality_result'):
        func_result = result['functionality_result']
        print(f"\nFunctionality Result:")
        print(f"  Success: {func_result.get('success', 'N/A')}")
        print(f"  Total Tests: {func_result.get('total_tests', 'N/A')}")
        print(f"  Passed Tests: {func_result.get('passed_tests', 'N/A')}")
        print(f"  Failed Tests: {func_result.get('failed_tests', 'N/A')}")
        
        if func_result.get('failed_tests', 0) > 0 and func_result.get('test_details'):
            print(f"\n  Failed Test Details:")
            for test_detail in func_result['test_details']:
                if not test_detail.get('success', True):
                    print(f"    Test: {test_detail.get('name', 'Unnamed')}")
                    print(f"      Operation: {test_detail.get('operation', 'N/A')}")
                    print(f"      Error: {test_detail.get('error', 'N/A')}")
                    print(f"      Expected Keys: {test_detail.get('expected_keys', 'N/A')}")
                    print(f"      Actual Keys: {test_detail.get('actual_keys', 'N/A')}")
    
    print(f"\nOverall Success: {result.get('overall_success', 'N/A')}")
    if result.get('errors'):
        print(f"\nValidation Errors: {result['errors']}")
    
    return result

def main():
    print("Debugging validation failures...")
    
    # Math Model
    debug_model_validation(
        "Math Model",
        "core.models.mathematics.unified_mathematics_model",
        "UnifiedMathematicsModel",
        create_math_test_cases
    )
    
    # Planning Model
    debug_model_validation(
        "Planning Model",
        "core.models.planning.unified_planning_model",
        "UnifiedPlanningModel",
        create_planning_test_cases
    )
    
    # Metacognition Model
    debug_model_validation(
        "Metacognition Model",
        "core.models.metacognition.unified_metacognition_model",
        "UnifiedMetacognitionModel",
        create_metacognition_test_cases
    )

if __name__ == "__main__":
    main()