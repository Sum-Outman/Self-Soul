#!/usr/bin/env python3
"""Check if process_audio method exists on UnifiedAudioModel"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_method():
    print("Checking UnifiedAudioModel for process_audio method")
    print("=" * 60)
    
    from core.models.audio.unified_audio_model import UnifiedAudioModel
    
    # Check class attribute
    print("1. Checking class attribute:")
    print(f"   'process_audio' in UnifiedAudioModel.__dict__: {'process_audio' in UnifiedAudioModel.__dict__}")
    print(f"   hasattr(UnifiedAudioModel, 'process_audio'): {hasattr(UnifiedAudioModel, 'process_audio')}")
    
    if hasattr(UnifiedAudioModel, 'process_audio'):
        method = getattr(UnifiedAudioModel, 'process_audio')
        print(f"   Method type: {type(method)}")
        print(f"   Is function: {callable(method)}")
    
    # Check MRO
    print("\n2. Method Resolution Order (MRO):")
    for i, cls in enumerate(UnifiedAudioModel.__mro__):
        print(f"   {i}: {cls.__name__}")
        if hasattr(cls, 'process_audio'):
            print(f"        -> has 'process_audio'")
    
    # Check instance
    print("\n3. Checking instance attribute:")
    try:
        audio_model = UnifiedAudioModel(config={"from_scratch": True})
        print(f"   Instance created: {audio_model.model_id}")
        print(f"   hasattr(instance, 'process_audio'): {hasattr(audio_model, 'process_audio')}")
        
        # Try to get attribute
        if hasattr(audio_model, 'process_audio'):
            method = getattr(audio_model, 'process_audio')
            print(f"   Instance method type: {type(method)}")
            print(f"   Instance method callable: {callable(method)}")
        else:
            print("   'process_audio' not found on instance")
            print("   Checking __dict__:")
            print(f"     'process_audio' in instance.__dict__: {'process_audio' in audio_model.__dict__}")
            
    except Exception as e:
        print(f"   Error creating instance: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Method check completed")

if __name__ == "__main__":
    check_method()