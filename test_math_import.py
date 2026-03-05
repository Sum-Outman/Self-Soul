#!/usr/bin/env python3
"""Test import of UnifiedMathematicsModel"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing import of UnifiedMathematicsModel...")
try:
    from core.models.mathematics.unified_mathematics_model import UnifiedMathematicsModel
    print("✓ Import successful")
    print(f"  Class: {UnifiedMathematicsModel}")
    
    # Try to instantiate
    print("\nAttempting to instantiate...")
    try:
        model = UnifiedMathematicsModel(config={"from_scratch": True})
        print(f"✓ Instantiation successful")
        print(f"  Model ID: {model.model_id}")
        print(f"  Model type: {model.model_type}")
    except Exception as e:
        print(f"✗ Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"✗ Other error: {e}")
    import traceback
    traceback.print_exc()