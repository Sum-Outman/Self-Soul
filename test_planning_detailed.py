#!/usr/bin/env python3
"""Detailed test for Planning Model to diagnose validation issues"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_planning_model():
    """Test Planning Model with detailed error reporting"""
    print("Testing Planning Model...")
    
    try:
        from core.models.planning.unified_planning_model import UnifiedPlanningModel
        
        # Initialize model
        config = {"from_scratch": True}
        model = UnifiedPlanningModel(config=config)
        print(f"✓ Model initialized: {model.model_id}")
        
        # Test 1: create_plan
        print("\nTest 1: create_plan")
        print("  Input: goal='organize a meeting', available_models=['manager', 'language', 'knowledge']")
        try:
            result = model.create_plan(
                goal='organize a meeting',
                available_models=['manager', 'language', 'knowledge']
            )
            print(f"  Result: {result}")
            print(f"  Result type: {type(result)}")
            print(f"  Keys in result: {list(result.keys())}")
            
            # Check for expected keys (validation expects ['status', 'success'])
            expected_keys = ['status', 'success']
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
                if isinstance(status_val, str) and status_val.lower() in ['error', 'fail', 'failed']:
                    print(f"  ⚠ Status indicates failure: {status_val}")
            else:
                print(f"  ⚠ No status key in result")
                
        except Exception as e:
            print(f"  ✗ Test execution error: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 2: analyze_goal_complexity
        print("\nTest 2: analyze_goal_complexity")
        print("  Input: goal='implement a new feature'")
        try:
            result = model.analyze_goal_complexity(goal='implement a new feature')
            print(f"  Result: {result}")
            print(f"  Result type: {type(result)}")
            print(f"  Keys in result: {list(result.keys())}")
            
            # Check for expected keys
            expected_keys = ['score', 'level']
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
            
            # Check score and level
            if 'score' in result:
                score_val = result['score']
                print(f"  Score: {score_val} (type: {type(score_val)})")
            if 'level' in result:
                level_val = result['level']
                print(f"  Level: {level_val} (type: {type(level_val)})")
                
        except Exception as e:
            print(f"  ✗ Test execution error: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"✗ Planning Model test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_planning_model()