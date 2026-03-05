#!/usr/bin/env python3
"""Simple test for audio model processing"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

def test_audio_processing():
    """Test audio model processing with different input formats"""
    print("Testing Audio Model Processing")
    print("=" * 60)
    
    try:
        from core.models.audio.unified_audio_model import UnifiedAudioModel
        
        # Initialize audio model
        config = {"sample_rate": 16000, "from_scratch": True}
        audio_model = UnifiedAudioModel(config=config)
        print(f"✓ Audio model initialized: {audio_model.model_id}")
        
        # Test 1: Raw bytes input
        print("\n1. Testing raw bytes input...")
        # Create random audio bytes (simulating 1 second of 16kHz 16-bit audio)
        audio_bytes = np.random.randint(-32768, 32767, 16000, dtype=np.int16).tobytes()
        result1 = audio_model.process_audio(audio_data=audio_bytes, language="en-US")
        print(f"   Result success: {result1.get('text', '') != ''}")
        print(f"   Error: {result1.get('error', 'None')}")
        
        # Test 2: Numpy array input
        print("\n2. Testing numpy array input...")
        audio_array = np.random.randn(16000).astype(np.float32)
        result2 = audio_model.process_audio(audio_data=audio_array, language="en-US")
        print(f"   Result success: {result2.get('text', '') != ''}")
        print(f"   Error: {result2.get('error', 'None')}")
        
        # Test 3: List input
        print("\n3. Testing list input...")
        audio_list = list(np.random.randn(8000))  # 0.5 second at 16kHz
        result3 = audio_model.process_audio(audio_data=audio_list, language="en-US")
        print(f"   Result success: {result3.get('text', '') != ''}")
        print(f"   Error: {result3.get('error', 'None')}")
        
        # Test 4: Test with tobytes() as in end-to-end test
        print("\n4. Testing with tobytes() as in end-to-end test...")
        test_audio = np.random.randn(16000)
        audio_bytes_tobytes = test_audio.tobytes()
        result4 = audio_model.process_audio(audio_data=audio_bytes_tobytes, language="en-US")
        print(f"   Result success: {result4.get('text', '') != ''}")
        print(f"   Error: {result4.get('error', 'None')}")
        
        print("\n" + "=" * 60)
        print("Audio Processing Test Completed")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Audio processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_audio_processing()
    sys.exit(0 if success else 1)