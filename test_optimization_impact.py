"""
Test the impact of initialization optimizations
"""

import time
import sys
import os

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_standard_template():
    """Test standard template initialization"""
    print("Testing Standard UnifiedModelTemplate...")
    
    total_start = time.time()
    
    try:
        from core.models.unified_model_template import UnifiedModelTemplate
        
        import_time = time.time() - total_start
        print(f"  Import time: {import_time:.2f}s")
        
        # Create multiple instances
        instance_times = []
        for i in range(3):
            init_start = time.time()
            model = UnifiedModelTemplate({'test_mode': True})
            instance_time = time.time() - init_start
            instance_times.append(instance_time)
            
            if i == 0:
                print(f"  First instance: {instance_time:.2f}s")
        
        avg_instance = sum(instance_times) / len(instance_times)
        print(f"  Average instance (3x): {avg_instance:.2f}s")
        
        total_time = time.time() - total_start
        print(f"  Total time: {total_time:.2f}s")
        
        return {
            'success': True,
            'import_time': import_time,
            'first_instance': instance_times[0],
            'avg_instance': avg_instance,
            'total_time': total_time
        }
        
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def test_optimized_template():
    """Test optimized template initialization"""
    print("Testing OptimizedUnifiedModelTemplate...")
    
    total_start = time.time()
    
    try:
        from core.models.optimized_model_template import OptimizedUnifiedModelTemplate
        
        import_time = time.time() - total_start
        print(f"  Import time: {import_time:.2f}s")
        
        # Create multiple instances
        instance_times = []
        for i in range(3):
            init_start = time.time()
            model = OptimizedUnifiedModelTemplate({'test_mode': True, 'optimized_initialization': True})
            instance_time = time.time() - init_start
            instance_times.append(instance_time)
            
            if i == 0:
                print(f"  First instance: {instance_time:.2f}s")
        
        avg_instance = sum(instance_times) / len(instance_times)
        print(f"  Average instance (3x): {avg_instance:.2f}s")
        
        total_time = time.time() - total_start
        print(f"  Total time: {total_time:.2f}s")
        
        # Test lazy component access
        if instance_times[0] < 1.0:  # If fast enough, test lazy access
            print("  Testing lazy component access...")
            lazy_start = time.time()
            
            # Access a lazy component
            try:
                # Try to access a lazy component
                if hasattr(model, 'external_api_service'):
                    _ = model.external_api_service
                    lazy_time = time.time() - lazy_start
                    print(f"  Lazy component access: {lazy_time:.2f}s")
            except:
                pass
        
        return {
            'success': True,
            'import_time': import_time,
            'first_instance': instance_times[0],
            'avg_instance': avg_instance,
            'total_time': total_time
        }
        
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def test_lazy_component_performance():
    """Test lazy component performance"""
    print("\nTesting Lazy Component Performance...")
    
    try:
        from core.initialization_optimizer import LazyComponent
        
        # Test with simulated expensive initialization
        init_count = [0]
        
        def expensive_init():
            init_count[0] += 1
            time.sleep(0.5)  # Simulate expensive initialization
            return {"data": "expensive_result"}
        
        # Create lazy component
        lazy = LazyComponent(expensive_init, "expensive_component")
        
        # First access should be slow
        print("  First access...")
        start = time.time()
        result1 = lazy.get()
        first_time = time.time() - start
        
        print(f"    Time: {first_time:.2f}s")
        print(f"    Initialized: {lazy.is_initialized()}")
        print(f"    Init count: {init_count[0]}")
        
        # Second access should be fast (cached)
        print("  Second access (cached)...")
        start = time.time()
        result2 = lazy.get()
        second_time = time.time() - start
        
        print(f"    Time: {second_time:.2f}s")
        print(f"    Initialized: {lazy.is_initialized()}")
        print(f"    Init count: {init_count[0]} (should be same)")
        
        # Test with multiple threads
        print("  Testing thread safety...")
        import threading
        
        results = []
        threads = []
        
        def access_lazy(idx):
            start = time.time()
            result = lazy.get()
            access_time = time.time() - start
            results.append((idx, access_time))
        
        for i in range(5):
            t = threading.Thread(target=access_lazy, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        avg_thread_time = sum(t for _, t in results) / len(results)
        print(f"    Average thread access time: {avg_thread_time:.2f}s")
        
        return {
            'success': True,
            'first_access': first_time,
            'cached_access': second_time,
            'avg_thread_access': avg_thread_time,
            'init_count': init_count[0]
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
    print("Initialization Optimization Impact Test")
    print("=" * 60)
    
    results = {}
    
    # Test standard template
    results['standard'] = test_standard_template()
    print()
    
    # Test optimized template
    results['optimized'] = test_optimized_template()
    print()
    
    # Test lazy component
    results['lazy'] = test_lazy_component_performance()
    print()
    
    # Comparison
    print("=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    if results['standard']['success'] and results['optimized']['success']:
        std = results['standard']
        opt = results['optimized']
        
        # Import time comparison
        import_improvement = (std['import_time'] - opt['import_time']) / std['import_time'] * 100
        print(f"Import time:")
        print(f"  Standard: {std['import_time']:.2f}s")
        print(f"  Optimized: {opt['import_time']:.2f}s")
        print(f"  Improvement: {import_improvement:.1f}%")
        print()
        
        # First instance comparison
        instance_improvement = (std['first_instance'] - opt['first_instance']) / std['first_instance'] * 100
        print(f"First instance time:")
        print(f"  Standard: {std['first_instance']:.2f}s")
        print(f"  Optimized: {opt['first_instance']:.2f}s")
        print(f"  Improvement: {instance_improvement:.1f}%")
        print()
        
        # Average instance comparison
        avg_improvement = (std['avg_instance'] - opt['avg_instance']) / std['avg_instance'] * 100
        print(f"Average instance time (3x):")
        print(f"  Standard: {std['avg_instance']:.2f}s")
        print(f"  Optimized: {opt['avg_instance']:.2f}s")
        print(f"  Improvement: {avg_improvement:.1f}%")
        print()
        
        # Total time comparison
        total_improvement = (std['total_time'] - opt['total_time']) / std['total_time'] * 100
        print(f"Total test time:")
        print(f"  Standard: {std['total_time']:.2f}s")
        print(f"  Optimized: {opt['total_time']:.2f}s")
        print(f"  Improvement: {total_improvement:.1f}%")
        print()
        
        # Lazy component results
        if results['lazy']['success']:
            lazy = results['lazy']
            print(f"Lazy component performance:")
            print(f"  First access: {lazy['first_access']:.2f}s (simulated 0.5s init)")
            print(f"  Cached access: {lazy['cached_access']:.6f}s")
            print(f"  Cache speedup: {lazy['first_access']/lazy['cached_access']:.0f}x")
            print(f"  Thread-safe average: {lazy['avg_thread_access']:.6f}s")
            print()
        
        # Summary
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        if instance_improvement > 50:
            print("✅ EXCELLENT: More than 50% improvement in instance creation")
        elif instance_improvement > 30:
            print("✅ GOOD: More than 30% improvement in instance creation")
        elif instance_improvement > 10:
            print("✅ MODEST: More than 10% improvement in instance creation")
        elif instance_improvement > 0:
            print("⚠️  MINIMAL: Some improvement, but less than 10%")
        else:
            print("❌ REGRESSION: Optimization made things worse")
        
        if total_improvement > 50:
            print("✅ EXCELLENT: More than 50% total time improvement")
        
        # Recommendations
        print("\nRECOMMENDATIONS:")
        
        if import_improvement < 10:
            print("- Focus on reducing import time (module-level lazy loading)")
        
        if instance_improvement < 30:
            print("- Improve lazy initialization of expensive components")
        
        if results['lazy']['success'] and lazy['cached_access'] < 0.001:
            print("- Lazy caching is working well (microsecond access time)")
    
    else:
        print("One or both tests failed. Cannot compare.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())