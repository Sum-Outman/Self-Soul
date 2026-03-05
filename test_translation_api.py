#!/usr/bin/env python3
"""Test Translation model API directly"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Translation model API directly...")
try:
    from core.models.translation.unified_translation_model import UnifiedTranslationModel
    
    # Initialize model
    config = {"from_scratch": True}
    model = UnifiedTranslationModel(config=config)
    print(f"✓ Model initialized: {model.model_id}")
    
    # Check if translate method exists
    print(f"\nChecking API methods:")
    print(f"  hasattr(model, 'translate'): {hasattr(model, 'translate')}")
    print(f"  hasattr(model, 'batch_translate'): {hasattr(model, 'batch_translate')}")
    print(f"  hasattr(model, '_process_operation'): {hasattr(model, '_process_operation')}")
    
    # Try to call translate method
    if hasattr(model, 'translate'):
        print(f"\nTesting translate method...")
        try:
            result = model.translate(
                text="hello world",
                source_lang="en",
                target_lang="zh",
                lang="en"
            )
            print(f"✓ translate method call successful")
            print(f"  Result keys: {list(result.keys())}")
            print(f"  Success: {result.get('success', 'N/A')}")
            print(f"  Translated text: {result.get('translated_text', 'N/A')}")
        except Exception as e:
            print(f"✗ translate method call failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n✗ translate method not found on model instance")
        
        # Check class attribute
        print(f"Checking class attributes:")
        print(f"  'translate' in UnifiedTranslationModel.__dict__: {'translate' in UnifiedTranslationModel.__dict__}")
        print(f"  hasattr(UnifiedTranslationModel, 'translate'): {hasattr(UnifiedTranslationModel, 'translate')}")
        
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()