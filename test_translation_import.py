#!/usr/bin/env python3
"""Test import of UnifiedTranslationModel"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing import of UnifiedTranslationModel...")
try:
    from core.models.translation.unified_translation_model import UnifiedTranslationModel
    print("✓ Import successful")
    print(f"  Class: {UnifiedTranslationModel}")
    
    # Try to instantiate
    print("\nAttempting to instantiate...")
    try:
        model = UnifiedTranslationModel(config={"from_scratch": True})
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