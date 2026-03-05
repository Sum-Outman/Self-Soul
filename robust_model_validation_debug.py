#!/usr/bin/env python3
"""Debug version of robust_model_validation.py with detailed logging"""

import sys
import os
import time
import json
import traceback
import importlib
import importlib.util
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more details
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('model_validation_debug.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import validation framework components
try:
    from core.models.base.unified_model_base import UnifiedModelBase
    logger.info("Successfully imported UnifiedModelBase")
except ImportError as e:
    logger.warning(f"Could not import UnifiedModelBase: {e}")

def create_math_test_cases():
    """Create test cases for math model"""
    return [
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
            'expected_keys': ['status', 'solutions']
        }
    ]

def test_math_model_debug():
    """Debug Math Model validation"""
    print("=== DEBUG: Testing Math Model ===")
    
    try:
        from core.models.mathematics.unified_mathematics_model import UnifiedMathematicsModel
        
        # Initialize model
        config = {"from_scratch": True}
        model = UnifiedMathematicsModel(config=config)
        print(f"✓ Model initialized: {model.model_id}")
        
        test_cases = create_math_test_cases()
        
        for i, test_case in enumerate(test_cases):
            print(f"\nTest {i+1}: {test_case['name']}")
            print(f"  Operation: {test_case['operation']}")
            print(f"  Data: {test_case['data']}")
            print(f"  Expected keys: {test_case['expected_keys']}")
            
            try:
                operation = test_case['operation']
                data = test_case['data']
                
                if not hasattr(model, operation):
                    print(f"  ✗ Operation '{operation}' not available on model")
                    continue
                
                operation_method = getattr(model, operation)
                print(f"  ✓ Method found: {operation_method}")
                
                # Call method
                output = operation_method(**data)
                print(f"  Output: {output}")
                print(f"  Output type: {type(output)}")
                print(f"  Output keys: {list(output.keys())}")
                
                # Validate output
                if isinstance(output, dict):
                    # Check for success/status
                    has_success_failure = False
                    
                    if 'success' in output:
                        success_value = output['success']
                        print(f"  Success value: {success_value} (type: {type(success_value)})")
                        if isinstance(success_value, bool) and not success_value:
                            has_success_failure = True
                            print(f"  ⚠ Success is False (boolean)")
                        elif isinstance(success_value, int) and success_value == 0:
                            has_success_failure = True
                            print(f"  ⚠ Success is 0 (integer)")
                        elif isinstance(success_value, str) and success_value.lower() in ['false', 'error', 'fail']:
                            has_success_failure = True
                            print(f"  ⚠ Success indicates failure: {success_value}")
                    
                    if 'status' in output and isinstance(output['status'], str):
                        status_lower = output['status'].lower()
                        print(f"  Status value: {output['status']} (type: {type(output['status'])})")
                        if status_lower in ['error', 'fail', 'failure', 'failed']:
                            has_success_failure = True
                            print(f"  ⚠ Status indicates failure: {output['status']}")
                    
                    if has_success_failure:
                        print(f"  ⚠ Operation may have failed")
                        error_msg = output.get('error', output.get('failure_message', 'No error message'))
                        print(f"  Error message: {error_msg}")
                    
                    # Check expected keys
                    missing_keys = []
                    for key in test_case['expected_keys']:
                        if key not in output:
                            missing_keys.append(key)
                    
                    if missing_keys:
                        print(f"  ✗ Missing keys: {missing_keys}")
                        print(f"  Actual keys: {list(output.keys())}")
                    else:
                        print(f"  ✓ All expected keys present")
                        
                else:
                    print(f"  ✗ Output is not a dict: {type(output)}")
                    
            except Exception as e:
                print(f"  ✗ Test execution error: {e}")
                traceback.print_exc()
                
    except Exception as e:
        print(f"✗ Math Model test failed: {e}")
        traceback.print_exc()

def create_planning_test_cases():
    """Create test cases for planning model"""
    return [
        {
            'name': 'Create simple plan',
            'operation': 'create_plan',
            'data': {
                'goal': 'organize a meeting',
                'available_models': ['manager', 'language', 'knowledge']
            },
            'expected_keys': ['status', 'success']
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

def test_planning_model_debug():
    """Debug Planning Model validation"""
    print("\n=== DEBUG: Testing Planning Model ===")
    
    try:
        from core.models.planning.unified_planning_model import UnifiedPlanningModel
        
        # Initialize model
        config = {"from_scratch": True}
        model = UnifiedPlanningModel(config=config)
        print(f"✓ Model initialized: {model.model_id}")
        
        test_cases = create_planning_test_cases()
        
        for i, test_case in enumerate(test_cases):
            print(f"\nTest {i+1}: {test_case['name']}")
            print(f"  Operation: {test_case['operation']}")
            print(f"  Data: {test_case['data']}")
            print(f"  Expected keys: {test_case['expected_keys']}")
            
            try:
                operation = test_case['operation']
                data = test_case['data']
                
                if not hasattr(model, operation):
                    print(f"  ✗ Operation '{operation}' not available on model")
                    continue
                
                operation_method = getattr(model, operation)
                print(f"  ✓ Method found: {operation_method}")
                
                # Call method
                output = operation_method(**data)
                print(f"  Output: {output}")
                print(f"  Output type: {type(output)}")
                print(f"  Output keys: {list(output.keys())}")
                
                # Validate output
                if isinstance(output, dict):
                    # Check for success/status
                    has_success_failure = False
                    
                    if 'success' in output:
                        success_value = output['success']
                        print(f"  Success value: {success_value} (type: {type(success_value)})")
                        if isinstance(success_value, bool) and not success_value:
                            has_success_failure = True
                            print(f"  ⚠ Success is False (boolean)")
                        elif isinstance(success_value, int) and success_value == 0:
                            has_success_failure = True
                            print(f"  ⚠ Success is 0 (integer)")
                        elif isinstance(success_value, str) and success_value.lower() in ['false', 'error', 'fail']:
                            has_success_failure = True
                            print(f"  ⚠ Success indicates failure: {success_value}")
                    
                    if 'status' in output and isinstance(output['status'], str):
                        status_lower = output['status'].lower()
                        print(f"  Status value: {output['status']} (type: {type(output['status'])})")
                        if status_lower in ['error', 'fail', 'failure', 'failed']:
                            has_success_failure = True
                            print(f"  ⚠ Status indicates failure: {output['status']}")
                    
                    if has_success_failure:
                        print(f"  ⚠ Operation may have failed")
                        error_msg = output.get('error', output.get('failure_message', 'No error message'))
                        print(f"  Error message: {error_msg}")
                    
                    # Check expected keys
                    missing_keys = []
                    for key in test_case['expected_keys']:
                        if key not in output:
                            missing_keys.append(key)
                    
                    if missing_keys:
                        print(f"  ✗ Missing keys: {missing_keys}")
                        print(f"  Actual keys: {list(output.keys())}")
                    else:
                        print(f"  ✓ All expected keys present")
                        
                else:
                    print(f"  ✗ Output is not a dict: {type(output)}")
                    
            except Exception as e:
                print(f"  ✗ Test execution error: {e}")
                traceback.print_exc()
                
    except Exception as e:
        print(f"✗ Planning Model test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("=== DEBUG MODE: Testing Math and Planning Models ===")
    test_math_model_debug()
    test_planning_model_debug()
    print("\n=== DEBUG COMPLETE ===")