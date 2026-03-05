"""
Test optimization of real models
"""

import time
import sys
import os

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_language_model():
    """Test Language Model optimization"""
    print("Testing Language Model...")
    
    results = {}
    
    try:
        from core.models.language.unified_language_model import UnifiedLanguageModel
        from core.models.optimized_model_template import create_optimized_model
        
        # Test standard model
        print("  Testing standard model...")
        std_times = []
        for i in range(3):
            start = time.time()
            model = UnifiedLanguageModel({
                'test_mode': True,
                'from_scratch': True  # Avoid BERT loading
            })
            std_times.append(time.time() - start)
            
            if i == 0:
                print(f"    First instance: {std_times[0]:.2f}s")
        
        std_avg = sum(std_times) / len(std_times)
        print(f"    Average (3x): {std_avg:.2f}s")
        
        results['standard'] = {
            'success': True,
            'first': std_times[0],
            'avg': std_avg,
            'all': std_times
        }
        
        # Test optimized model using factory
        print("  Testing optimized model (factory)...")
        opt_times = []
        for i in range(3):
            start = time.time()
            model = create_optimized_model(
                UnifiedLanguageModel,
                {
                    'test_mode': True,
                    'from_scratch': True,
                    'optimized_initialization': True
                },
                use_optimized_template=True
            )
            opt_times.append(time.time() - start)
            
            if i == 0:
                print(f"    First instance: {opt_times[0]:.2f}s")
        
        opt_avg = sum(opt_times) / len(opt_times)
        print(f"    Average (3x): {opt_avg:.2f}s")
        
        results['optimized'] = {
            'success': True,
            'first': opt_times[0],
            'avg': opt_avg,
            'all': opt_times
        }
        
        # Compare
        if std_avg > 0:
            improvement = (std_avg - opt_avg) / std_avg * 100
            print(f"  Improvement: {improvement:.1f}%")
            results['improvement'] = improvement
        
        # Test component access
        print("  Testing component access...")
        if opt_times[0] < 5.0:  # If reasonable time
            # Get last optimized model
            model = create_optimized_model(
                UnifiedLanguageModel,
                {
                    'test_mode': True,
                    'from_scratch': True,
                    'optimized_initialization': True
                }
            )
            
            # Access lazy components
            components = ['external_api_service', 'stream_manager', 'data_processor']
            access_times = {}
            
            for comp in components:
                if hasattr(model, comp):
                    start = time.time()
                    try:
                        getattr(model, comp)
                        access_time = time.time() - start
                        access_times[comp] = access_time
                        print(f"    {comp}: {access_time:.3f}s")
                    except Exception as e:
                        print(f"    {comp} failed: {e}")
            
            results['access_times'] = access_times
        
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
    
    return results

def test_vision_model():
    """Test Vision Model optimization"""
    print("\nTesting Vision Model...")
    
    results = {}
    
    try:
        from core.models.vision.unified_vision_model import UnifiedVisionModel
        from core.models.optimized_model_template import create_optimized_model
        
        # Test standard model
        print("  Testing standard model...")
        std_times = []
        for i in range(3):
            start = time.time()
            model = UnifiedVisionModel({
                'test_mode': True
            })
            std_times.append(time.time() - start)
            
            if i == 0:
                print(f"    First instance: {std_times[0]:.2f}s")
        
        std_avg = sum(std_times) / len(std_times)
        print(f"    Average (3x): {std_avg:.2f}s")
        
        results['standard'] = {
            'success': True,
            'first': std_times[0],
            'avg': std_avg,
            'all': std_times
        }
        
        # Test optimized model
        print("  Testing optimized model...")
        opt_times = []
        for i in range(3):
            start = time.time()
            model = create_optimized_model(
                UnifiedVisionModel,
                {
                    'test_mode': True,
                    'optimized_initialization': True
                }
            )
            opt_times.append(time.time() - start)
            
            if i == 0:
                print(f"    First instance: {opt_times[0]:.2f}s")
        
        opt_avg = sum(opt_times) / len(opt_times)
        print(f"    Average (3x): {opt_avg:.2f}s")
        
        results['optimized'] = {
            'success': True,
            'first': opt_times[0],
            'avg': opt_avg,
            'all': opt_times
        }
        
        # Compare
        if std_avg > 0:
            improvement = (std_avg - opt_avg) / std_avg * 100
            print(f"  Improvement: {improvement:.1f}%")
            results['improvement'] = improvement
        
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
    
    return results

def test_audio_model():
    """Test Audio Model optimization"""
    print("\nTesting Audio Model...")
    
    results = {}
    
    try:
        from core.models.audio.unified_audio_model import UnifiedAudioModel
        from core.models.optimized_model_template import create_optimized_model
        
        # Test standard model
        print("  Testing standard model...")
        std_times = []
        for i in range(3):
            start = time.time()
            model = UnifiedAudioModel({
                'test_mode': True
            })
            std_times.append(time.time() - start)
            
            if i == 0:
                print(f"    First instance: {std_times[0]:.2f}s")
        
        std_avg = sum(std_times) / len(std_times)
        print(f"    Average (3x): {std_avg:.2f}s")
        
        results['standard'] = {
            'success': True,
            'first': std_times[0],
            'avg': std_avg,
            'all': std_times
        }
        
        # Test optimized model
        print("  Testing optimized model...")
        opt_times = []
        for i in range(3):
            start = time.time()
            model = create_optimized_model(
                UnifiedAudioModel,
                {
                    'test_mode': True,
                    'optimized_initialization': True
                }
            )
            opt_times.append(time.time() - start)
            
            if i == 0:
                print(f"    First instance: {opt_times[0]:.2f}s")
        
        opt_avg = sum(opt_times) / len(opt_times)
        print(f"    Average (3x): {opt_avg:.2f}s")
        
        results['optimized'] = {
            'success': True,
            'first': opt_times[0],
            'avg': opt_avg,
            'all': opt_times
        }
        
        # Compare
        if std_avg > 0:
            improvement = (std_avg - opt_avg) / std_avg * 100
            print(f"  Improvement: {improvement:.1f}%")
            results['improvement'] = improvement
        
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
    
    return results

def main():
    """Main test function"""
    print("=" * 60)
    print("Real Model Optimization Test")
    print("=" * 60)
    
    all_results = {}
    
    # Test Language Model
    print("\n1. Language Model")
    all_results['language'] = test_language_model()
    
    # Test Vision Model
    print("\n2. Vision Model")
    all_results['vision'] = test_vision_model()
    
    # Test Audio Model
    print("\n3. Audio Model")
    all_results['audio'] = test_audio_model()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful_models = 0
    total_improvement = 0
    improvement_count = 0
    
    for model_name, results in all_results.items():
        if 'standard' in results and 'optimized' in results:
            if results['standard']['success'] and results['optimized']['success']:
                successful_models += 1
                
                std_avg = results['standard']['avg']
                opt_avg = results['optimized']['avg']
                
                print(f"\n{model_name.upper()} MODEL:")
                print(f"  Standard: {std_avg:.2f}s")
                print(f"  Optimized: {opt_avg:.2f}s")
                
                if 'improvement' in results:
                    improvement = results['improvement']
                    print(f"  Improvement: {improvement:.1f}%")
                    
                    if improvement > 0:
                        total_improvement += improvement
                        improvement_count += 1
                    
                    # Grade the improvement
                    if improvement > 50:
                        print(f"  ✅ EXCELLENT (>50% improvement)")
                    elif improvement > 30:
                        print(f"  ✅ GOOD (>30% improvement)")
                    elif improvement > 10:
                        print(f"  👍 MODEST (>10% improvement)")
                    elif improvement > 0:
                        print(f"  ⚠️  MINIMAL (>0% improvement)")
                    else:
                        print(f"  ❌ REGRESSION (no improvement)")
        
        elif 'error' in results:
            print(f"\n{model_name.upper()} MODEL:")
            print(f"  ❌ ERROR: {results['error']}")
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    
    print(f"Successful models tested: {successful_models}/3")
    
    if improvement_count > 0:
        avg_improvement = total_improvement / improvement_count
        print(f"Average improvement: {avg_improvement:.1f}%")
        
        if avg_improvement > 30:
            print("🎉 EXCELLENT: Average improvement >30%")
        elif avg_improvement > 15:
            print("👍 GOOD: Average improvement >15%")
        elif avg_improvement > 5:
            print("👌 ACCEPTABLE: Average improvement >5%")
        elif avg_improvement > 0:
            print("⚠️  MINIMAL: Some improvement")
        else:
            print("❌ POOR: No average improvement")
    else:
        print("No measurable improvements")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    # Analyze each model's performance
    for model_name, results in all_results.items():
        if 'standard' in results and 'optimized' in results:
            if results['standard']['success'] and results['optimized']['success']:
                std_avg = results['standard']['avg']
                
                if std_avg > 2.0:
                    print(f"- {model_name}: Very slow (>2s). Focus on major optimizations.")
                elif std_avg > 1.0:
                    print(f"- {model_name}: Slow (1-2s). Good candidate for optimization.")
                elif std_avg > 0.5:
                    print(f"- {model_name}: Moderate (0.5-1s). Some optimization possible.")
                else:
                    print(f"- {model_name}: Fast (<0.5s). Already well-optimized.")
    
    print("\nGeneral recommendations:")
    print("- Enable lazy loading for all non-critical components")
    print("- Implement parallel initialization for independent components")
    print("- Use background loading for expensive resources (BERT, CV libs)")
    print("- Add warm-up scripts for production deployment")
    
    return 0 if successful_models >= 2 else 1

if __name__ == "__main__":
    sys.exit(main())