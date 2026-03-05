"""
Test current model initialization performance
"""

import time
import sys
import os

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_language_model():
    """Test Language Model initialization"""
    print("Testing Language Model...")
    
    # Measure total time from import to instance
    total_start = time.time()
    
    try:
        from core.models.language.unified_language_model import UnifiedLanguageModel
        
        import_time = time.time() - total_start
        print(f"  Import time: {import_time:.2f}s")
        
        # Create instance
        init_start = time.time()
        model = UnifiedLanguageModel({'test_mode': True, 'from_scratch': True})
        init_time = time.time() - init_start
        
        total_time = time.time() - total_start
        
        print(f"  Instance time: {init_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        
        return {
            'success': True,
            'import_time': import_time,
            'init_time': init_time,
            'total_time': total_time
        }
        
    except Exception as e:
        print(f"  Failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def test_vision_model():
    """Test Vision Model initialization"""
    print("Testing Vision Model...")
    
    total_start = time.time()
    
    try:
        from core.models.vision.unified_vision_model import UnifiedVisionModel
        
        import_time = time.time() - total_start
        print(f"  Import time: {import_time:.2f}s")
        
        init_start = time.time()
        model = UnifiedVisionModel({'test_mode': True})
        init_time = time.time() - init_start
        
        total_time = time.time() - total_start
        
        print(f"  Instance time: {init_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        
        return {
            'success': True,
            'import_time': import_time,
            'init_time': init_time,
            'total_time': total_time
        }
        
    except Exception as e:
        print(f"  Failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def test_audio_model():
    """Test Audio Model initialization"""
    print("Testing Audio Model...")
    
    total_start = time.time()
    
    try:
        from core.models.audio.unified_audio_model import UnifiedAudioModel
        
        import_time = time.time() - total_start
        print(f"  Import time: {import_time:.2f}s")
        
        init_start = time.time()
        model = UnifiedAudioModel({'test_mode': True})
        init_time = time.time() - init_start
        
        total_time = time.time() - total_start
        
        print(f"  Instance time: {init_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        
        return {
            'success': True,
            'import_time': import_time,
            'init_time': init_time,
            'total_time': total_time
        }
        
    except Exception as e:
        print(f"  Failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """Main test function"""
    print("=" * 60)
    print("Current Model Initialization Performance")
    print("=" * 60)
    
    results = {}
    
    # Run tests
    results['language'] = test_language_model()
    print()
    
    results['vision'] = test_vision_model()
    print()
    
    results['audio'] = test_audio_model()
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results.values() if r['success']]
    
    if successful:
        total_import = sum(r['import_time'] for r in successful)
        total_init = sum(r['init_time'] for r in successful)
        total_time = sum(r['total_time'] for r in successful)
        
        print(f"Successful tests: {len(successful)}/3")
        print(f"Total import time: {total_import:.2f}s")
        print(f"Total instance time: {total_init:.2f}s")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average per model: {total_time/len(successful):.2f}s")
        
        # Analysis
        print("\nANALYSIS:")
        print(f"- Import overhead: {total_import/total_time*100:.1f}% of total time")
        print(f"- Instance creation: {total_init/total_time*100:.1f}% of total time")
        
        if total_import > total_init * 2:
            print("- MAJOR BOTTLENECK: Import time dominates (>2x instance time)")
            print("  RECOMMENDATION: Implement lazy imports and caching")
        elif total_import > total_init:
            print("- BOTTLENECK: Import time is higher than instance time")
            print("  RECOMMENDATION: Optimize import dependencies")
        else:
            print("- Instance creation is the primary cost")
            print("  RECOMMENDATION: Focus on lazy initialization within models")
    
    else:
        print("No successful tests!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())