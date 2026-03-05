"""
Test the fixed OptimizedUnifiedModelTemplate
"""

import time
import sys
import os

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_optimized_template():
    """Test that OptimizedUnifiedModelTemplate can be instantiated"""
    print("Testing OptimizedUnifiedModelTemplate instantiation...")
    
    try:
        from core.models.optimized_model_template import OptimizedUnifiedModelTemplate
        
        # Test with optimized initialization
        print("  Testing with optimized initialization...")
        start = time.time()
        model = OptimizedUnifiedModelTemplate({
            'test_mode': True, 
            'optimized_initialization': True
        })
        opt_time = time.time() - start
        
        print(f"    Success! Time: {opt_time:.2f}s")
        print(f"    Model ID: {model.model_id}")
        print(f"    Model type: {model.model_type}")
        
        # Test lazy component
        print("  Testing lazy component access...")
        if hasattr(model, 'external_api_service'):
            start = time.time()
            service = model.external_api_service
            access_time = time.time() - start
            print(f"    External API service access: {access_time:.3f}s")
            print(f"    Service type: {type(service).__name__}")
        
        # Test process_operation
        print("  Testing process_operation...")
        result = model._process_operation('test', {'input': 'test'})
        print(f"    Result: {result}")
        
        return {
            'success': True,
            'opt_time': opt_time,
            'model_id': model.model_id
        }
        
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def test_standard_template():
    """Test that we can create a concrete standard model"""
    print("\nTesting concrete standard model...")
    
    try:
        from core.models.unified_model_template import UnifiedModelTemplate
        
        # Create a concrete implementation
        class ConcreteStandardModel(UnifiedModelTemplate):
            def _get_model_id(self):
                return "concrete_standard"
            
            def _get_model_type(self):
                return "test"
            
            def _get_supported_operations(self):
                return ["test"]
            
            def _initialize_model_specific_components(self, config):
                # Initialize any model-specific components here
                pass
            
            def _process_operation(self, operation, input_data):
                return {
                    "operation": operation,
                    "status": "success",
                    "result": f"Processed {operation}"
                }
        
        start = time.time()
        model = ConcreteStandardModel({'test_mode': True})
        std_time = time.time() - start
        
        print(f"  Success! Time: {std_time:.2f}s")
        print(f"  Model ID: {model.model_id}")
        
        return {
            'success': True,
            'std_time': std_time
        }
        
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def test_optimized_concrete_model():
    """Test creating a concrete model using optimized template"""
    print("\nTesting concrete optimized model...")
    
    try:
        from core.models.optimized_model_template import OptimizedUnifiedModelTemplate
        
        # Create concrete implementation
        class ConcreteOptimizedModel(OptimizedUnifiedModelTemplate):
            def _get_model_id(self):
                return "concrete_optimized"
            
            def _get_model_type(self):
                return "test"
            
            def _get_supported_operations(self):
                return ["test", "optimized_test"]
            
            # _initialize_model_specific_components and _process_operation
            # are inherited from OptimizedUnifiedModelTemplate
        
        # Test with optimized initialization
        start = time.time()
        model = ConcreteOptimizedModel({
            'test_mode': True,
            'optimized_initialization': True
        })
        opt_time = time.time() - start
        
        print(f"  Success! Time: {opt_time:.2f}s")
        print(f"  Model ID: {model.model_id}")
        print(f"  Supported operations: {model._get_supported_operations()}")
        
        # Test operation
        result = model._process_operation('test', {'data': 'test'})
        print(f"  Operation result: {result.get('status', 'unknown')}")
        
        return {
            'success': True,
            'opt_time': opt_time
        }
        
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """Main test function"""
    print("=" * 60)
    print("Testing Fixed Optimized Template")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Optimized template
    results['optimized_template'] = test_optimized_template()
    
    # Test 2: Standard concrete model
    results['standard_concrete'] = test_standard_template()
    
    # Test 3: Optimized concrete model
    results['optimized_concrete'] = test_optimized_concrete_model()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results.values() if r['success']]
    
    if successful:
        print(f"Successful tests: {len(successful)}/{len(results)}")
        
        # Compare times if available
        if (results['standard_concrete']['success'] and 
            results['optimized_concrete']['success']):
            
            std_time = results['standard_concrete'].get('std_time', 0)
            opt_time = results['optimized_concrete'].get('opt_time', 0)
            
            if std_time > 0 and opt_time > 0:
                improvement = (std_time - opt_time) / std_time * 100
                print(f"\nPerformance comparison:")
                print(f"  Standard concrete: {std_time:.2f}s")
                print(f"  Optimized concrete: {opt_time:.2f}s")
                print(f"  Improvement: {improvement:.1f}%")
                
                if improvement > 0:
                    print(f"✅ Optimization is working!")
                else:
                    print(f"⚠️  No improvement or regression")
    else:
        print("All tests failed!")
    
    return 0 if len(successful) >= 2 else 1

if __name__ == "__main__":
    sys.exit(main())