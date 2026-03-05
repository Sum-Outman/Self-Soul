#!/usr/bin/env python3
"""Debug failed models with detailed error information"""

import sys
import os
import traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_math_model():
    """Test Math model directly"""
    print("\n" + "="*60)
    print("Testing Math Model")
    print("="*60)
    
    try:
        from core.models.mathematics.unified_mathematics_model import UnifiedMathematicsModel
        
        # Initialize model
        config = {"from_scratch": True}
        model = UnifiedMathematicsModel(config=config)
        print(f"✓ Model initialized: {model.model_id}")
        
        # Test evaluate_expression
        print(f"\n1. Testing evaluate_expression...")
        try:
            result = model.evaluate_expression(
                expression="2 + 3 * 5",
                variables=None
            )
            print(f"   Result: {result}")
            print(f"   Keys: {list(result.keys())}")
            print(f"   Has 'status': {'status' in result}")
            print(f"   Has 'result': {'result' in result}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            traceback.print_exc()
        
        # Test solve_equation
        print(f"\n2. Testing solve_equation...")
        try:
            result = model.solve_equation(
                equation="2*x + 5 = 15",
                variable="x"
            )
            print(f"   Result: {result}")
            print(f"   Keys: {list(result.keys())}")
            print(f"   Has 'status': {'status' in result}")
            print(f"   Has 'message': {'message' in result}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        traceback.print_exc()

def test_planning_model():
    """Test Planning model directly"""
    print("\n" + "="*60)
    print("Testing Planning Model")
    print("="*60)
    
    try:
        from core.models.planning.unified_planning_model import UnifiedPlanningModel
        
        # Initialize model
        config = {"from_scratch": True}
        model = UnifiedPlanningModel(config=config)
        print(f"✓ Model initialized: {model.model_id}")
        
        # Test create_plan
        print(f"\n1. Testing create_plan...")
        try:
            result = model.create_plan(
                goal='organize a meeting',
                available_models=['manager', 'language', 'knowledge']
            )
            print(f"   Result: {result}")
            print(f"   Keys: {list(result.keys())}")
            print(f"   Has 'status': {'status' in result}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            traceback.print_exc()
        
        # Test analyze_goal_complexity
        print(f"\n2. Testing analyze_goal_complexity...")
        try:
            result = model.analyze_goal_complexity(
                goal='implement a new feature'
            )
            print(f"   Result: {result}")
            print(f"   Keys: {list(result.keys())}")
            print(f"   Has 'score': {'score' in result}")
            print(f"   Has 'level': {'level' in result}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        traceback.print_exc()

def test_metacognition_model():
    """Test Metacognition model directly"""
    print("\n" + "="*60)
    print("Testing Metacognition Model")
    print("="*60)
    
    try:
        from core.models.metacognition.unified_metacognition_model import UnifiedMetacognitionModel
        
        # Initialize model
        config = {"from_scratch": True}
        model = UnifiedMetacognitionModel(config=config)
        print(f"✓ Model initialized: {model.model_id}")
        
        # Test apply_metacognition
        print(f"\n1. Testing apply_metacognition...")
        try:
            result = model.apply_metacognition(
                cognitive_state={
                    'task': 'learning new concept',
                    'confidence': 0.7,
                    'focus': 0.8
                }
            )
            print(f"   Result: {result}")
            print(f"   Keys: {list(result.keys())}")
            print(f"   Has 'success': {'success' in result}")
            print(f"   Has 'failure_message': {'failure_message' in result}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            traceback.print_exc()
        
        # Test strategy_selection
        print(f"\n2. Testing strategy_selection...")
        try:
            result = model.strategy_selection(
                task_description='solve a complex problem',
                available_strategies=['divide_and_conquer', 'brute_force', 'heuristic']
            )
            print(f"   Result: {result}")
            print(f"   Keys: {list(result.keys())}")
            print(f"   Has 'success': {'success' in result}")
            print(f"   Has 'selection_method': {'selection_method' in result}")
            print(f"   Has 'selection_timestamp': {'selection_timestamp' in result}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        traceback.print_exc()

def main():
    print("Debugging failed models with detailed information...")
    
    test_math_model()
    test_planning_model()
    test_metacognition_model()
    
    print("\n" + "="*60)
    print("Debugging complete")
    print("="*60)

if __name__ == "__main__":
    main()