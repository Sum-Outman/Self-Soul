"""
Test optimized model implementation
"""

import time
import sys
import os

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestOptimizedModel:
    """Test model using optimized template"""
    
    def _get_model_id(self):
        return "test_optimized_model"
    
    def _get_model_type(self):
        return "test"
    
    def _get_supported_operations(self):
        return ["test_operation"]
    
    def _initialize_model_specific_components(self):
        pass
    
    def _process_operation(self, operation, data):
        return {"result": "test"}

def create_concrete_optimized_model():
    """Create a concrete model class using optimized template"""
    from core.models.optimized_model_template import OptimizedUnifiedModelTemplate
    
    class ConcreteOptimizedModel(OptimizedUnifiedModelTemplate, TestOptimizedModel):
        def __init__(self, config=None):
            # Combine the two parent classes
            TestOptimizedModel.__init__(self)
            OptimizedUnifiedModelTemplate.__init__(self, config)
    
    return ConcreteOptimizedModel

def test_optimized_vs_standard():
    """Compare optimized vs standard model initialization"""
    print("=" * 60)
    print("Optimized vs Standard Model Initialization")
    print("=" * 60)
    
    # First, test importing the modules separately
    print("\n1. Testing module import times...")
    
    # Test importing standard template module
    std_import_start = time.time()
    from core.models.unified_model_template import UnifiedModelTemplate
    std_import_time = time.time() - std_import_start
    print(f"  Standard template import: {std_import_time:.2f}s")
    
    # Test importing optimized template module
    opt_import_start = time.time()
    from core.models.optimized_model_template import OptimizedUnifiedModelTemplate
    opt_import_time = time.time() - opt_import_start
    print(f"  Optimized template import: {opt_import_time:.2f}s")
    
    import_improvement = (std_import_time - opt_import_time) / std_import_time * 100
    print(f"  Import improvement: {import_improvement:.1f}%")
    
    # Now test creating a concrete model
    print("\n2. Testing concrete model creation...")
    
    # Create a simple concrete model class for testing
    class SimpleStandardModel(UnifiedModelTemplate):
        def _get_model_id(self):
            return "simple_standard"
        
        def _get_model_type(self):
            return "test"
        
        def _get_supported_operations(self):
            return ["test"]
        
        def _initialize_model_specific_components(self):
            pass
        
        def _process_operation(self, operation, data):
            return {"result": "test"}
    
    # Create optimized model class
    ConcreteOptimizedModel = create_concrete_optimized_model()
    
    # Test standard model instantiation
    std_instances = []
    print("  Creating standard models (3 instances)...")
    for i in range(3):
        start = time.time()
        model = SimpleStandardModel({'test_mode': True})
        instance_time = time.time() - start
        std_instances.append(instance_time)
        
        if i == 0:
            print(f"    First instance: {instance_time:.2f}s")
    
    std_avg = sum(std_instances) / len(std_instances)
    print(f"    Average instance: {std_avg:.2f}s")
    
    # Test optimized model instantiation
    opt_instances = []
    print("  Creating optimized models (3 instances)...")
    for i in range(3):
        start = time.time()
        model = ConcreteOptimizedModel({'test_mode': True, 'optimized_initialization': True})
        instance_time = time.time() - start
        opt_instances.append(instance_time)
        
        if i == 0:
            print(f"    First instance: {instance_time:.2f}s")
    
    opt_avg = sum(opt_instances) / len(opt_instances)
    print(f"    Average instance: {opt_avg:.2f}s")
    
    # Compare
    instance_improvement = (std_avg - opt_avg) / std_avg * 100
    print(f"  Instance creation improvement: {instance_improvement:.1f}%")
    
    # Test lazy component access
    print("\n3. Testing lazy component access...")
    
    # Get the last optimized model
    opt_model = ConcreteOptimizedModel({'test_mode': True, 'optimized_initialization': True})
    
    # Access lazy components
    lazy_components = ['external_api_service', 'stream_manager', 'data_processor']
    
    for comp in lazy_components:
        if hasattr(opt_model, comp):
            print(f"  Accessing {comp}...")
            start = time.time()
            try:
                component = getattr(opt_model, comp)
                access_time = time.time() - start
                print(f"    Access time: {access_time:.3f}s")
                print(f"    Component type: {type(component).__name__}")
            except Exception as e:
                print(f"    Failed: {e}")
    
    # Test performance with component access
    print("\n4. Testing performance with component access...")
    
    # Standard model with component access
    std_model = SimpleStandardModel({'test_mode': True})
    std_access_times = []
    
    for comp in ['external_api_service', 'stream_manager']:
        if hasattr(std_model, comp):
            start = time.time()
            try:
                getattr(std_model, comp)
                std_access_times.append(time.time() - start)
            except:
                pass
    
    # Optimized model with component access
    opt_model = ConcreteOptimizedModel({'test_mode': True, 'optimized_initialization': True})
    opt_access_times = []
    
    for comp in ['external_api_service', 'stream_manager']:
        if hasattr(opt_model, comp):
            start = time.time()
            try:
                getattr(opt_model, comp)
                opt_access_times.append(time.time() - start)
            except:
                pass
    
    if std_access_times and opt_access_times:
        std_avg_access = sum(std_access_times) / len(std_access_times)
        opt_avg_access = sum(opt_access_times) / len(opt_access_times)
        
        print(f"  Standard model component access: {std_avg_access:.3f}s avg")
        print(f"  Optimized model component access: {opt_avg_access:.3f}s avg")
        
        if opt_avg_access < std_avg_access:
            access_improvement = (std_avg_access - opt_avg_access) / std_avg_access * 100
            print(f"  Component access improvement: {access_improvement:.1f}%")
        else:
            print(f"  Note: Component access may be deferred (lazy)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"Import time improvement: {import_improvement:.1f}%")
    print(f"Instance creation improvement: {instance_improvement:.1f}%")
    
    total_std_time = std_import_time + std_avg * 3
    total_opt_time = opt_import_time + opt_avg * 3
    
    print(f"\nTotal time for 3 models (including import):")
    print(f"  Standard: {total_std_time:.2f}s")
    print(f"  Optimized: {total_opt_time:.2f}s")
    
    total_improvement = (total_std_time - total_opt_time) / total_std_time * 100
    print(f"  Total improvement: {total_improvement:.1f}%")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    
    if import_improvement > 90:
        print("- ✅ Excellent import time reduction")
    elif import_improvement > 50:
        print("- ✅ Good import time reduction")
    else:
        print("- ⚠️  Import time needs more optimization")
    
    if instance_improvement > 50:
        print("- ✅ Excellent instance creation improvement")
    elif instance_improvement > 30:
        print("- ✅ Good instance creation improvement")
    elif instance_improvement > 0:
        print("- ⚠️  Modest instance creation improvement")
    else:
        print("- ❌ Instance creation regression")
    
    if total_improvement > 50:
        print("- 🎉 OVERALL: Excellent optimization (>50% improvement)")
    elif total_improvement > 30:
        print("- 👍 OVERALL: Good optimization (>30% improvement)")
    elif total_improvement > 10:
        print("- 👌 OVERALL: Modest optimization (>10% improvement)")
    elif total_improvement > 0:
        print("- ⚠️  OVERALL: Minimal optimization")
    else:
        print("- ❌ OVERALL: Optimization regression")
    
    return {
        'import_improvement': import_improvement,
        'instance_improvement': instance_improvement,
        'total_improvement': total_improvement,
        'std_import_time': std_import_time,
        'opt_import_time': opt_import_time,
        'std_avg_instance': std_avg,
        'opt_avg_instance': opt_avg
    }

def main():
    """Main test function"""
    try:
        results = test_optimized_vs_standard()
        return 0
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())