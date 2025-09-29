"""
模型完整性测试脚本
Test script to verify all models implement the 6 required abstract methods
"""

import sys
import os
import importlib
import inspect
from typing import List, Dict, Any

# Add core directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

# List of all model files to test
MODEL_FILES = [
    "core.models.audio.unified_audio_model",
    "core.models.autonomous.unified_autonomous_model", 
    "core.models.collaboration.unified_collaboration_model",
    "core.models.computer.unified_computer_model",
    "core.models.emotion.unified_emotion_model",
    "core.models.finance.unified_finance_model",
    "core.models.knowledge.unified_knowledge_model",
    "core.models.language.unified_language_model",
    "core.models.manager.unified_manager_model",
    "core.models.medical.unified_medical_model",
    "core.models.motion.unified_motion_model",
    "core.models.optimization.unified_optimization_model",
    "core.models.planning.unified_planning_model",
    "core.models.prediction.unified_prediction_model",
    "core.models.programming.unified_programming_model",
    "core.models.sensor.unified_sensor_model",
    "core.models.spatial.unified_spatial_model",
    "core.models.video.unified_video_model",
    "core.models.vision.unified_vision_model"
]

# Required abstract methods from unified_model_template
REQUIRED_METHODS = [
    "_get_model_id",
    "_get_model_type", 
    "_get_supported_operations",
    "_initialize_model_specific_components",
    "_process_operation",
    "_create_stream_processor"
]

def test_model_class(model_module_path: str) -> Dict[str, Any]:
    """Test if a model class implements all required methods"""
    try:
        module = importlib.import_module(model_module_path)
        
        # Find the main model class (usually ends with 'Model')
        model_classes = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name.endswith('Model') and not name.startswith('_'):
                model_classes.append((name, obj))
        
        if not model_classes:
            return {"success": False, "error": f"No model class found in {model_module_path}"}
        
        results = {}
        for class_name, model_class in model_classes:
            missing_methods = []
            implemented_methods = []
            
            # Check each required method
            for method_name in REQUIRED_METHODS:
                if hasattr(model_class, method_name):
                    method = getattr(model_class, method_name)
                    if not inspect.isabstract(method):
                        implemented_methods.append(method_name)
                    else:
                        missing_methods.append(method_name)
                else:
                    missing_methods.append(method_name)
            
            # Try to instantiate the class
            can_instantiate = False
            instantiation_error = None
            try:
                instance = model_class()
                can_instantiate = True
            except Exception as e:
                instantiation_error = str(e)
            
            results[class_name] = {
                "implemented_methods": implemented_methods,
                "missing_methods": missing_methods,
                "all_methods_implemented": len(missing_methods) == 0,
                "can_instantiate": can_instantiate,
                "instantiation_error": instantiation_error
            }
        
        return {"success": True, "results": results}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    """Main testing function"""
    print("=" * 80)
    print("MODEL INTEGRITY TEST")
    print("=" * 80)
    print(f"Testing {len(MODEL_FILES)} model files...")
    print()
    
    all_passed = True
    detailed_results = {}
    
    for model_file in MODEL_FILES:
        print(f"Testing: {model_file}")
        result = test_model_class(model_file)
        
        if result["success"]:
            detailed_results[model_file] = result["results"]
            
            for class_name, class_result in result["results"].items():
                status = "PASS" if class_result["all_methods_implemented"] and class_result["can_instantiate"] else "FAIL"
                print(f"  {class_name}: {status}")
                
                if not class_result["all_methods_implemented"]:
                    print(f"    Missing methods: {', '.join(class_result['missing_methods'])}")
                    all_passed = False
                
                if not class_result["can_instantiate"]:
                    print(f"    Instantiation error: {class_result['instantiation_error']}")
                    all_passed = False
        else:
            print(f"  ERROR: {result['error']}")
            all_passed = False
            detailed_results[model_file] = {"error": result["error"]}
        
        print()
    
    # Summary report
    print("=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    if all_passed:
        print("✅ ALL MODELS PASSED INTEGRITY TEST!")
        print("All 18 models correctly implement the 6 required abstract methods.")
        print("All models can be successfully instantiated.")
    else:
        print("❌ SOME MODELS FAILED INTEGRITY TEST")
        print("Please check the detailed report above for missing methods or instantiation errors.")
    
    # Detailed statistics
    total_models = 0
    passed_models = 0
    
    for model_file, results in detailed_results.items():
        if "error" not in results:
            for class_name, class_result in results.items():
                total_models += 1
                if class_result["all_methods_implemented"] and class_result["can_instantiate"]:
                    passed_models += 1
    
    print(f"\nModels tested: {total_models}")
    print(f"Models passed: {passed_models}")
    print(f"Models failed: {total_models - passed_models}")
    
    # Fix division by zero error
    if total_models > 0:
        print(f"Success rate: {passed_models/total_models*100:.1f}%")
    else:
        print("Success rate: 0.0% (no models tested)")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
