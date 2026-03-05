#!/usr/bin/env python3
"""Test Advanced Reasoning model API directly"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Advanced Reasoning model API directly...")
try:
    from core.models.advanced_reasoning.unified_advanced_reasoning_model import UnifiedAdvancedReasoningModel
    
    # Initialize model
    config = {"from_scratch": True}
    model = UnifiedAdvancedReasoningModel(config=config)
    print(f"✓ Model initialized: {model.model_id}")
    
    # Check if methods exist
    print(f"\nChecking API methods:")
    methods_to_check = [
        'logical_reasoning',
        'causal_inference',
        'reason',
        'symbolic_manipulation',
        'probabilistic_reasoning',
        'counterfactual_analysis'
    ]
    
    for method in methods_to_check:
        print(f"  hasattr(model, '{method}'): {hasattr(model, method)}")
    
    print(f"  hasattr(model, '_process_operation'): {hasattr(model, '_process_operation')}")
    
    # Try to call logical_reasoning method
    if hasattr(model, 'logical_reasoning'):
        print(f"\nTesting logical_reasoning method...")
        try:
            result = model.logical_reasoning(
                premise='All humans are mortal. Socrates is a human.',
                query='Is Socrates mortal?'
            )
            print(f"✓ logical_reasoning method call successful")
            print(f"  Result keys: {list(result.keys())}")
            print(f"  Success/Status: {result.get('success', result.get('status', 'N/A'))}")
            print(f"  Conclusion: {result.get('conclusion', 'N/A')}")
        except Exception as e:
            print(f"✗ logical_reasoning method call failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n✗ logical_reasoning method not found on model instance")
        
    # Try to call causal_inference method
    if hasattr(model, 'causal_inference'):
        print(f"\nTesting causal_inference method...")
        try:
            result = model.causal_inference(
                cause='raining',
                effect='wet ground'
            )
            print(f"✓ causal_inference method call successful")
            print(f"  Result keys: {list(result.keys())}")
            print(f"  Success/Status: {result.get('success', result.get('status', 'N/A'))}")
            print(f"  Causal strength: {result.get('causal_strength', 'N/A')}")
        except Exception as e:
            print(f"✗ causal_inference method call failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n✗ causal_inference method not found on model instance")
        
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()