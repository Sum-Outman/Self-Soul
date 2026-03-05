"""
Simple initialization performance test
"""

import time
import sys
import os
import logging

# Suppress all logging
logging.getLogger().setLevel(logging.CRITICAL)

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_init(model_name, model_class, config):
    """Test initialization of a single model"""
    print(f"Testing {model_name}...")
    
    try:
        # Time the import
        import_start = time.time()
        exec(f"from {model_class} import {model_name.split('_')[0].title() + model_name.split('_')[1]}")
        import_time = time.time() - import_start
        
        # Time the initialization
        init_start = time.time()
        model_class_obj = eval(model_name.split('_')[0].title() + model_name.split('_')[1])
        model = model_class_obj(config)
        init_time = time.time() - init_start
        
        total_time = import_time + init_time
        
        print(f"  Import: {import_time:.2f}s")
        print(f"  Init: {init_time:.2f}s")
        print(f"  Total: {total_time:.2f}s")
        
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
    print("=" * 50)
    print("Simple Initialization Performance Test")
    print("=" * 50)
    
    # Test configurations
    tests = [
        {
            'name': 'language_model',
            'class': 'core.models.language.unified_language_model',
            'config': {'test_mode': True, 'from_scratch': True}
        },
        {
            'name': 'vision_model', 
            'class': 'core.models.vision.unified_vision_model',
            'config': {'test_mode': True}
        },
        {
            'name': 'audio_model',
            'class': 'core.models.audio.unified_audio_model',
            'config': {'test_mode': True}
        }
    ]
    
    results = {}
    total_time = 0
    
    for test in tests:
        result = test_model_init(test['name'], test['class'], test['config'])
        results[test['name']] = result
        
        if result['success']:
            total_time += result['total_time']
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    successful = [r for r in results.values() if r['success']]
    
    if successful:
        avg_time = sum(r['total_time'] for r in successful) / len(successful)
        
        print(f"Successful tests: {len(successful)}/{len(tests)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per model: {avg_time:.2f}s")
        
        # Breakdown
        print("\nBreakdown:")
        for test_name, result in results.items():
            if result['success']:
                print(f"  {test_name}: {result['total_time']:.2f}s "
                      f"(import: {result['import_time']:.2f}s, "
                      f"init: {result['init_time']:.2f}s)")
    else:
        print("No successful tests!")
    
    return 0 if len(successful) >= 2 else 1

if __name__ == "__main__":
    sys.exit(main())