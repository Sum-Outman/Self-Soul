"""
Test script for model initialization optimization
"""

import time
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_standard_initialization():
    """Test standard model initialization"""
    logger.info("Testing standard model initialization...")
    
    results = {}
    
    # Test Language Model
    try:
        from core.models.language.unified_language_model import UnifiedLanguageModel
        
        start_time = time.time()
        config = {'test_mode': True, 'from_scratch': True}  # Use from_scratch to avoid BERT loading
        model = UnifiedLanguageModel(config)
        init_time = time.time() - start_time
        
        results['language_model'] = {
            'time': init_time,
            'success': True,
            'model_id': getattr(model, 'model_id', 'unknown')
        }
        
        logger.info(f"Language Model: {init_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Language Model failed: {e}")
        results['language_model'] = {
            'time': 0,
            'success': False,
            'error': str(e)
        }
    
    # Test Vision Model
    try:
        from core.models.vision.unified_vision_model import UnifiedVisionModel
        
        start_time = time.time()
        config = {'test_mode': True}
        model = UnifiedVisionModel(config)
        init_time = time.time() - start_time
        
        results['vision_model'] = {
            'time': init_time,
            'success': True,
            'model_id': getattr(model, 'model_id', 'unknown')
        }
        
        logger.info(f"Vision Model: {init_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Vision Model failed: {e}")
        results['vision_model'] = {
            'time': 0,
            'success': False,
            'error': str(e)
        }
    
    # Test Audio Model
    try:
        from core.models.audio.unified_audio_model import UnifiedAudioModel
        
        start_time = time.time()
        config = {'test_mode': True}
        model = UnifiedAudioModel(config)
        init_time = time.time() - start_time
        
        results['audio_model'] = {
            'time': init_time,
            'success': True,
            'model_id': getattr(model, 'model_id', 'unknown')
        }
        
        logger.info(f"Audio Model: {init_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Audio Model failed: {e}")
        results['audio_model'] = {
            'time': 0,
            'success': False,
            'error': str(e)
        }
    
    # Calculate statistics
    successful_tests = [r for r in results.values() if r['success']]
    if successful_tests:
        total_time = sum(r['time'] for r in successful_tests)
        avg_time = total_time / len(successful_tests)
        
        results['statistics'] = {
            'total_models': len(results),
            'successful_models': len(successful_tests),
            'total_time': total_time,
            'average_time': avg_time
        }
        
        logger.info(f"Standard initialization: {len(successful_tests)}/{len(results)} models, "
                   f"total: {total_time:.2f}s, average: {avg_time:.2f}s")
    
    return results

def test_optimized_initialization():
    """Test optimized model initialization"""
    logger.info("Testing optimized model initialization...")
    
    results = {}
    
    # Test with initialization optimizer
    try:
        from core.initialization_optimizer import create_optimized_model
        
        # Test Language Model
        try:
            from core.models.language.unified_language_model import UnifiedLanguageModel
            
            start_time = time.time()
            config = {'test_mode': True, 'from_scratch': True}
            model = create_optimized_model(UnifiedLanguageModel, config)
            init_time = time.time() - start_time
            
            results['language_model'] = {
                'time': init_time,
                'success': True,
                'model_id': getattr(model, 'model_id', 'unknown')
            }
            
            logger.info(f"Optimized Language Model: {init_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Optimized Language Model failed: {e}")
            results['language_model'] = {
                'time': 0,
                'success': False,
                'error': str(e)
            }
        
        # Test Vision Model
        try:
            from core.models.vision.unified_vision_model import UnifiedVisionModel
            
            start_time = time.time()
            config = {'test_mode': True}
            model = create_optimized_model(UnifiedVisionModel, config)
            init_time = time.time() - start_time
            
            results['vision_model'] = {
                'time': init_time,
                'success': True,
                'model_id': getattr(model, 'model_id', 'unknown')
            }
            
            logger.info(f"Optimized Vision Model: {init_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Optimized Vision Model failed: {e}")
            results['vision_model'] = {
                'time': 0,
                'success': False,
                'error': str(e)
            }
        
        # Test Audio Model
        try:
            from core.models.audio.unified_audio_model import UnifiedAudioModel
            
            start_time = time.time()
            config = {'test_mode': True}
            model = create_optimized_model(UnifiedAudioModel, config)
            init_time = time.time() - start_time
            
            results['audio_model'] = {
                'time': init_time,
                'success': True,
                'model_id': getattr(model, 'model_id', 'unknown')
            }
            
            logger.info(f"Optimized Audio Model: {init_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Optimized Audio Model failed: {e}")
            results['audio_model'] = {
                'time': 0,
                'success': False,
                'error': str(e)
            }
    
    except Exception as e:
        logger.error(f"Optimized initialization framework failed: {e}")
        # Fallback to testing optimized template directly
        logger.info("Falling back to direct optimized template test...")
        
        try:
            from core.models.optimized_model_template import OptimizedUnifiedModelTemplate
            
            start_time = time.time()
            config = {'test_mode': True}
            model = OptimizedUnifiedModelTemplate(config)
            init_time = time.time() - start_time
            
            results['optimized_template'] = {
                'time': init_time,
                'success': True,
                'model_id': getattr(model, 'model_id', 'unknown')
            }
            
            logger.info(f"Optimized Template: {init_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Optimized Template failed: {e}")
            results['optimized_template'] = {
                'time': 0,
                'success': False,
                'error': str(e)
            }
    
    # Calculate statistics
    successful_tests = [r for r in results.values() if r['success']]
    if successful_tests:
        total_time = sum(r['time'] for r in successful_tests)
        avg_time = total_time / len(successful_tests)
        
        results['statistics'] = {
            'total_models': len(results),
            'successful_models': len(successful_tests),
            'total_time': total_time,
            'average_time': avg_time
        }
        
        logger.info(f"Optimized initialization: {len(successful_tests)}/{len(results)} models, "
                   f"total: {total_time:.2f}s, average: {avg_time:.2f}s")
    
    return results

def test_lazy_component():
    """Test lazy component functionality"""
    logger.info("Testing lazy component...")
    
    try:
        from core.initialization_optimizer import LazyComponent
        
        # Track initialization
        init_called = []
        
        def init_func():
            init_called.append(time.time())
            time.sleep(0.1)  # Simulate slow initialization
            return "initialized_value"
        
        # Create lazy component
        lazy = LazyComponent(init_func, name="test_component")
        
        # Should not be initialized yet
        assert not lazy.is_initialized()
        assert len(init_called) == 0
        
        # First access should trigger initialization
        start_time = time.time()
        value = lazy.get()
        init_time = time.time() - start_time
        
        assert lazy.is_initialized()
        assert len(init_called) == 1
        assert value == "initialized_value"
        assert init_time >= 0.1  # Should have taken at least 0.1s
        
        # Second access should use cached value
        start_time = time.time()
        value2 = lazy.get()
        cache_time = time.time() - start_time
        
        assert value2 == "initialized_value"
        assert len(init_called) == 1  # No additional calls
        assert cache_time < 0.01  # Should be very fast
        
        # Reset and test again
        lazy.reset()
        assert not lazy.is_initialized()
        
        value3 = lazy.get()
        assert lazy.is_initialized()
        assert len(init_called) == 2
        
        logger.info(f"Lazy component test passed: "
                   f"first init: {init_time:.3f}s, "
                   f"cache access: {cache_time:.3f}s")
        
        return {
            'success': True,
            'first_init_time': init_time,
            'cache_access_time': cache_time,
            'init_call_count': len(init_called)
        }
        
    except Exception as e:
        logger.error(f"Lazy component test failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def test_parallel_initializer():
    """Test parallel initializer functionality"""
    logger.info("Testing parallel initializer...")
    
    try:
        from core.initialization_optimizer import ParallelInitializer, ComponentProfile, InitializationPriority
        
        # Create initializer
        initializer = ParallelInitializer(max_workers=2)
        
        # Track initialization order
        init_order = []
        init_times = {}
        
        def create_init_func(name, delay=0.2):
            def init_func():
                start = time.time()
                init_order.append(name)
                time.sleep(delay)  # Simulate initialization delay
                init_times[name] = time.time() - start
                return f"{name}_value"
            return init_func
        
        # Create component profiles
        components = {
            'comp1': ComponentProfile(
                name='comp1',
                init_func=create_init_func('comp1', 0.2),
                priority=InitializationPriority.ESSENTIAL
            ),
            'comp2': ComponentProfile(
                name='comp2',
                init_func=create_init_func('comp2', 0.3),
                priority=InitializationPriority.ESSENTIAL,
                dependencies=['comp1']  # comp2 depends on comp1
            ),
            'comp3': ComponentProfile(
                name='comp3',
                init_func=create_init_func('comp3', 0.1),
                priority=InitializationPriority.ESSENTIAL
            )
        }
        
        # Initialize in parallel
        start_time = time.time()
        results = initializer.initialize_graph(components)
        total_time = time.time() - start_time
        
        # Verify results
        assert 'comp1' in results
        assert 'comp2' in results
        assert 'comp3' in results
        assert results['comp1'] == 'comp1_value'
        assert results['comp2'] == 'comp2_value'
        assert results['comp3'] == 'comp3_value'
        
        # Verify dependency order (comp1 before comp2)
        comp1_index = init_order.index('comp1')
        comp2_index = init_order.index('comp2')
        assert comp1_index < comp2_index, "Dependency order not respected"
        
        # Verify parallel execution (comp1 and comp3 should overlap)
        # Since we have 2 workers, comp1 and comp3 should start together
        # comp2 waits for comp1, so total time should be ~comp1 + comp2 (not comp1 + comp2 + comp3)
        expected_max_time = 0.2 + 0.3  # comp1 + comp2 (sequential due to dependency)
        # comp3 runs in parallel with comp1
        assert total_time < expected_max_time + 0.1, f"Expected < {expected_max_time + 0.1}s, got {total_time:.2f}s"
        
        logger.info(f"Parallel initializer test passed: "
                   f"total time: {total_time:.2f}s, "
                   f"init order: {init_order}")
        
        # Cleanup
        initializer.shutdown()
        
        return {
            'success': True,
            'total_time': total_time,
            'init_order': init_order,
            'init_times': init_times
        }
        
    except Exception as e:
        logger.error(f"Parallel initializer test failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def compare_performance(standard_results, optimized_results):
    """Compare performance between standard and optimized initialization"""
    logger.info("Comparing performance...")
    
    comparison = {}
    
    # Compare individual models
    for model_name in ['language_model', 'vision_model', 'audio_model']:
        if (model_name in standard_results and standard_results[model_name]['success'] and
            model_name in optimized_results and optimized_results[model_name]['success']):
            
            std_time = standard_results[model_name]['time']
            opt_time = optimized_results[model_name]['time']
            
            if std_time > 0:
                improvement = (std_time - opt_time) / std_time * 100
                comparison[model_name] = {
                    'standard_time': std_time,
                    'optimized_time': opt_time,
                    'improvement_percent': improvement,
                    'time_saved': std_time - opt_time
                }
                
                logger.info(f"{model_name}: "
                           f"standard={std_time:.2f}s, "
                           f"optimized={opt_time:.2f}s, "
                           f"improvement={improvement:.1f}%")
    
    # Compare total times
    if ('statistics' in standard_results and 'statistics' in optimized_results and
        standard_results['statistics']['successful_models'] > 0 and
        optimized_results['statistics']['successful_models'] > 0):
        
        std_total = standard_results['statistics']['total_time']
        opt_total = optimized_results['statistics']['total_time']
        
        if std_total > 0:
            total_improvement = (std_total - opt_total) / std_total * 100
            
            comparison['total'] = {
                'standard_total': std_total,
                'optimized_total': opt_total,
                'improvement_percent': total_improvement,
                'time_saved': std_total - opt_total,
                'standard_models': standard_results['statistics']['successful_models'],
                'optimized_models': optimized_results['statistics']['successful_models']
            }
            
            logger.info(f"Total: "
                       f"standard={std_total:.2f}s ({standard_results['statistics']['successful_models']} models), "
                       f"optimized={opt_total:.2f}s ({optimized_results['statistics']['successful_models']} models), "
                       f"improvement={total_improvement:.1f}%")
    
    return comparison

def run_comprehensive_test():
    """Run comprehensive initialization optimization tests"""
    logger.info("=" * 60)
    logger.info("Starting Comprehensive Initialization Optimization Tests")
    logger.info("=" * 60)
    
    all_results = {}
    
    # Test lazy component
    logger.info("\n1. Testing Lazy Component...")
    lazy_result = test_lazy_component()
    all_results['lazy_component'] = lazy_result
    
    # Test parallel initializer
    logger.info("\n2. Testing Parallel Initializer...")
    parallel_result = test_parallel_initializer()
    all_results['parallel_initializer'] = parallel_result
    
    # Test standard initialization
    logger.info("\n3. Testing Standard Initialization...")
    standard_results = test_standard_initialization()
    all_results['standard_initialization'] = standard_results
    
    # Test optimized initialization
    logger.info("\n4. Testing Optimized Initialization...")
    optimized_results = test_optimized_initialization()
    all_results['optimized_initialization'] = optimized_results
    
    # Compare performance
    logger.info("\n5. Comparing Performance...")
    comparison = compare_performance(standard_results, optimized_results)
    all_results['comparison'] = comparison
    
    # Generate summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    # Component tests
    logger.info("Component Tests:")
    logger.info(f"  Lazy Component: {'PASS' if lazy_result.get('success') else 'FAIL'}")
    logger.info(f"  Parallel Initializer: {'PASS' if parallel_result.get('success') else 'FAIL'}")
    
    # Performance comparison
    if comparison:
        logger.info("\nPerformance Comparison:")
        
        for model_name, data in comparison.items():
            if model_name != 'total':
                logger.info(f"  {model_name}:")
                logger.info(f"    Standard: {data['standard_time']:.2f}s")
                logger.info(f"    Optimized: {data['optimized_time']:.2f}s")
                logger.info(f"    Improvement: {data['improvement_percent']:.1f}%")
        
        if 'total' in comparison:
            total = comparison['total']
            logger.info(f"\n  Total ({total['standard_models']} models):")
            logger.info(f"    Standard: {total['standard_total']:.2f}s")
            logger.info(f"    Optimized: {total['optimized_total']:.2f}s")
            logger.info(f"    Improvement: {total['improvement_percent']:.1f}%")
            logger.info(f"    Time Saved: {total['time_saved']:.2f}s")
    
    # Success criteria
    success_criteria = [
        lazy_result.get('success', False),
        parallel_result.get('success', False),
        standard_results.get('statistics', {}).get('successful_models', 0) >= 2,
        optimized_results.get('statistics', {}).get('successful_models', 0) >= 2
    ]
    
    all_passed = all(success_criteria)
    
    logger.info(f"\nOverall Result: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed, all_results

def main():
    """Main test function"""
    try:
        success, results = run_comprehensive_test()
        
        # Save results to file
        import json
        with open('initialization_optimization_results.json', 'w') as f:
            # Convert to serializable format
            def serialize(obj):
                if isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                elif isinstance(obj, dict):
                    return {k: serialize(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [serialize(item) for item in obj]
                else:
                    return str(obj)
            
            json.dump(serialize(results), f, indent=2)
        
        logger.info("\nResults saved to initialization_optimization_results.json")
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)