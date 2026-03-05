#!/usr/bin/env python3
"""
Robust Model Validation Framework

A comprehensive validation framework for testing model functionality,
handling import issues, and providing detailed diagnostics.
"""
import sys
import os
import importlib
import logging
import traceback
from typing import Dict, List, Any, Optional, Callable
import json

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelValidationError(Exception):
    """Custom exception for model validation errors"""
    pass

class ModelValidator:
    """Robust model validation framework"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize validator with configuration"""
        self.config = config or {}
        self.results = {}
        self.test_cases = {}
        
        # Default configuration
        self.config.setdefault('max_retries', 2)
        self.config.setdefault('timeout_seconds', 30)
        self.config.setdefault('skip_on_error', True)
        self.config.setdefault('detailed_logging', True)
        
    def validate_model_import(self, module_path: str) -> Dict[str, Any]:
        """Validate model import with comprehensive error handling"""
        result = {
            'module_path': module_path,
            'success': False,
            'errors': [],
            'warnings': [],
            'import_time': 0,
            'module': None
        }
        
        import time
        start_time = time.time()
        
        try:
            # Try to import the module
            if self.config['detailed_logging']:
                logger.info(f"Attempting to import module: {module_path}")
            
            # Check if module path is valid
            parts = module_path.split('.')
            if len(parts) < 2:
                result['errors'].append(f"Invalid module path: {module_path}")
                return result
            
            # Try multiple import strategies
            import_strategies = [
                # Strategy 1: Standard import using importlib
                lambda: importlib.import_module(module_path),
                # Strategy 2: Direct import
                lambda: __import__(module_path),
                # Strategy 3: Import parent and get module
                lambda: self._import_with_parent(module_path),
                # Strategy 4: Try to find and load module file
                lambda: self._find_and_load_module(module_path)
            ]
            
            imported_module = None
            last_error = None
            
            for i, strategy in enumerate(import_strategies):
                try:
                    imported_module = strategy()
                    if imported_module:
                        break
                except Exception as e:
                    last_error = e
                    if self.config['detailed_logging']:
                        logger.warning(f"Import strategy {i+1} failed for {module_path}: {e}")
            
            if imported_module is None:
                result['errors'].append(f"All import strategies failed: {last_error}")
                return result
            
            # Get the actual module
            if isinstance(imported_module, str):
                # It's already a module name
                module = sys.modules.get(module_path)
            else:
                # It's a module object
                module = imported_module
            
            if not module:
                result['errors'].append(f"Failed to retrieve module from sys.modules")
                return result
            
            result['module'] = module
            result['success'] = True
            
            if self.config['detailed_logging']:
                logger.info(f"Successfully imported module: {module_path}")
            
        except ImportError as e:
            error_msg = f"ImportError: {e}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
            
            # Provide helpful suggestions
            if "No module named" in str(e):
                missing_module = str(e).split("'")[1] if "'" in str(e) else "unknown"
                result['warnings'].append(f"Missing module: {missing_module}. Try: pip install {missing_module}")
            
        except ModuleNotFoundError as e:
            error_msg = f"ModuleNotFoundError: {e}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error importing {module_path}: {e}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
            if self.config['detailed_logging']:
                logger.error(traceback.format_exc())
        
        finally:
            result['import_time'] = time.time() - start_time
        
        return result
    
    def _import_with_parent(self, module_path: str):
        """Import module by importing parent and accessing child"""
        parts = module_path.split('.')
        parent_path = '.'.join(parts[:-1])
        child_name = parts[-1]
        
        try:
            parent_module = __import__(parent_path, fromlist=[child_name])
            return getattr(parent_module, child_name, None)
        except Exception:
            # Try alternative: import the full module
            return __import__(module_path)
    
    def _find_and_load_module(self, module_path: str):
        """Find module file and load it dynamically"""
        # Convert module path to file path
        parts = module_path.split('.')
        relative_path = os.path.join(*parts) + '.py'
        full_path = os.path.join(project_root, relative_path)
        
        if os.path.exists(full_path):
            # Use importlib to load the module
            spec = importlib.util.spec_from_file_location(module_path, full_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
        
        return None
    
    def validate_model_instantiation(self, module_path: str, class_name: str = None) -> Dict[str, Any]:
        """Validate model instantiation with error handling"""
        result = {
            'module_path': module_path,
            'class_name': class_name,
            'success': False,
            'errors': [],
            'warnings': [],
            'instantiation_time': 0,
            'instance': None
        }
        
        import time
        start_time = time.time()
        
        try:
            # First validate import
            import_result = self.validate_model_import(module_path)
            if not import_result['success']:
                result['errors'].extend(import_result['errors'])
                result['warnings'].extend(import_result['warnings'])
                return result
            
            module = import_result['module']
            
            # Determine class name
            if class_name is None:
                # Try to infer class name from module
                class_name = self._infer_class_name(module, module_path)
                result['class_name'] = class_name
            
            if not class_name:
                result['errors'].append("Could not determine model class name")
                return result
            
            # Try to get the class
            model_class = getattr(module, class_name, None)
            if model_class is None:
                result['errors'].append(f"Class '{class_name}' not found in module")
                return result
            
            # Try to instantiate with different configurations
            configs_to_try = [
                {},  # Empty config
                {'device': 'cpu'},  # Basic config
                {'from_scratch': True, 'device': 'cpu'},  # From scratch config
                {'from_scratch': False, 'device': 'cpu'}  # Pre-trained config
            ]
            
            instance = None
            last_error = None
            
            for i, config in enumerate(configs_to_try):
                try:
                    if self.config['detailed_logging']:
                        logger.info(f"Attempting instantiation {i+1} with config: {config}")
                    
                    instance = model_class(config)
                    
                    # Check if instance has required attributes
                    required_attrs = ['model_id', 'model_type', 'model_name']
                    missing_attrs = []
                    for attr in required_attrs:
                        if not hasattr(instance, attr):
                            missing_attrs.append(attr)
                    
                    if missing_attrs:
                        result['warnings'].append(f"Instance missing attributes: {missing_attrs}")
                    else:
                        # Success!
                        break
                        
                except TypeError as e:
                    # Try without arguments
                    try:
                        instance = model_class()
                        break
                    except Exception as e2:
                        last_error = e2
                        if self.config['detailed_logging']:
                            logger.warning(f"Instantiation {i+1} failed: {e2}")
                except Exception as e:
                    last_error = e
                    if self.config['detailed_logging']:
                        logger.warning(f"Instantiation {i+1} failed: {e}")
            
            if instance is None:
                result['errors'].append(f"All instantiation attempts failed: {last_error}")
                return result
            
            result['instance'] = instance
            result['success'] = True
            
            # Get basic instance info
            instance_info = {}
            for attr in ['model_id', 'model_type', 'model_name', 'supported_operations']:
                if hasattr(instance, attr):
                    value = getattr(instance, attr)
                    instance_info[attr] = value
            
            result['instance_info'] = instance_info
            
            if self.config['detailed_logging']:
                logger.info(f"Successfully instantiated {module_path}.{class_name}")
            
        except Exception as e:
            error_msg = f"Error instantiating model: {e}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
            if self.config['detailed_logging']:
                logger.error(traceback.format_exc())
        
        finally:
            result['instantiation_time'] = time.time() - start_time
        
        return result
    
    def _infer_class_name(self, module, module_path: str) -> str:
        """Infer model class name from module"""
        # Common patterns
        possible_class_names = [
            # Try module name based patterns
            module_path.split('.')[-1].replace('_', ' ').title().replace(' ', '') + 'Model',
            'Unified' + module_path.split('.')[-1].replace('_', ' ').title().replace(' ', '') + 'Model',
            # Generic patterns
            'UnifiedModel',
            'Model',
            'MainModel'
        ]
        
        # Check module attributes
        for attr_name in dir(module):
            if not attr_name.startswith('_'):
                attr = getattr(module, attr_name)
                if isinstance(attr, type):  # It's a class
                    # Check if it looks like a model class
                    if 'Model' in attr_name or 'model' in attr_name.lower():
                        return attr_name
        
        # Try the possible class names
        for class_name in possible_class_names:
            if hasattr(module, class_name):
                return class_name
        
        return None
    
    def validate_model_functionality(self, instance, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate model functionality with test cases"""
        result = {
            'success': False,
            'test_results': [],
            'passed_tests': 0,
            'failed_tests': 0,
            'total_tests': len(test_cases),
            'errors': []
        }
        
        if not instance:
            result['errors'].append("No instance provided for functionality testing")
            return result
        
        for i, test_case in enumerate(test_cases):
            test_result = {
                'test_id': i + 1,
                'name': test_case.get('name', f'Test {i+1}'),
                'success': False,
                'error': None,
                'output': None,
                'execution_time': 0
            }
            
            import time
            start_time = time.time()
            
            try:
                operation = test_case.get('operation')
                data = test_case.get('data', {})
                expected_keys = test_case.get('expected_keys', [])
                
                if not operation:
                    test_result['error'] = "No operation specified in test case"
                    result['test_results'].append(test_result)
                    continue
                
                # Check if instance has the method
                if not hasattr(instance, operation):
                    # Try _process_operation as fallback
                    if hasattr(instance, '_process_operation'):
                        operation_method = getattr(instance, '_process_operation')
                        output = operation_method(operation, data)
                    else:
                        test_result['error'] = f"Operation '{operation}' not available"
                        result['test_results'].append(test_result)
                        continue
                else:
                    operation_method = getattr(instance, operation)
                    output = operation_method(**data)
                
                # Validate output
                if isinstance(output, dict):
                    # Check for success/status - more flexible handling
                    # Some models use 'success' (boolean or 0/1), others use 'status' string
                    has_success_failure = False
                    
                    # Check if operation failed based on success key
                    if 'success' in output:
                        # Handle different success representations: True/False, 1/0, 'success'/'error'
                        success_value = output['success']
                        if isinstance(success_value, bool) and not success_value:
                            has_success_failure = True
                        elif isinstance(success_value, int) and success_value == 0:
                            has_success_failure = True
                        elif isinstance(success_value, str) and success_value.lower() in ['false', 'error', 'fail', 'failure']:
                            has_success_failure = True
                    
                    # Also check status key for failure indication
                    if 'status' in output and isinstance(output['status'], str):
                        status_lower = output['status'].lower()
                        if status_lower in ['error', 'fail', 'failure', 'failed']:
                            has_success_failure = True
                    
                    if has_success_failure:
                        test_result['error'] = output.get('error', output.get('failure_message', 'Operation returned success=False or status=error'))
                        test_result['output'] = output
                    else:
                        # Check expected keys
                        missing_keys = []
                        for key in expected_keys:
                            if key not in output:
                                missing_keys.append(key)
                        
                        if missing_keys:
                            test_result['error'] = f"Output missing keys: {missing_keys}"
                            test_result['output'] = output
                        else:
                            test_result['success'] = True
                            test_result['output'] = output
                else:
                    # Non-dict output, just check if it's not None
                    if output is not None:
                        test_result['success'] = True
                        test_result['output'] = output
                    else:
                        test_result['error'] = "Operation returned None"
                
            except Exception as e:
                test_result['error'] = f"Test execution error: {e}"
                if self.config['detailed_logging']:
                    logger.error(f"Test {i+1} failed: {e}")
                    logger.error(traceback.format_exc())
            
            finally:
                test_result['execution_time'] = time.time() - start_time
            
            # Update counters
            if test_result['success']:
                result['passed_tests'] += 1
            else:
                result['failed_tests'] += 1
            
            result['test_results'].append(test_result)
        
        # Overall success if all tests passed
        result['success'] = result['passed_tests'] == result['total_tests']
        
        return result
    
    def run_comprehensive_validation(self, model_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive validation for a model"""
        result = {
            'model_spec': model_spec,
            'import_result': None,
            'instantiation_result': None,
            'functionality_result': None,
            'overall_success': False,
            'timestamp': None,
            'errors': []
        }
        
        import datetime
        result['timestamp'] = datetime.datetime.now().isoformat()
        
        try:
            module_path = model_spec['module_path']
            class_name = model_spec.get('class_name')
            test_cases = model_spec.get('test_cases', [])
            
            logger.info(f"Starting comprehensive validation for: {module_path}")
            
            # Step 1: Validate import
            result['import_result'] = self.validate_model_import(module_path)
            if not result['import_result']['success']:
                result['errors'].extend(result['import_result']['errors'])
                if self.config['skip_on_error']:
                    logger.warning(f"Skipping further validation due to import error")
                    return result
            
            # Step 2: Validate instantiation
            result['instantiation_result'] = self.validate_model_instantiation(module_path, class_name)
            if not result['instantiation_result']['success']:
                result['errors'].extend(result['instantiation_result']['errors'])
                if self.config['skip_on_error']:
                    logger.warning(f"Skipping functionality validation due to instantiation error")
                    return result
            
            # Step 3: Validate functionality
            instance = result['instantiation_result']['instance']
            if instance and test_cases:
                result['functionality_result'] = self.validate_model_functionality(instance, test_cases)
                if not result['functionality_result']['success']:
                    result['errors'].append(f"Functionality test failed: {result['functionality_result']['failed_tests']}/{result['functionality_result']['total_tests']} tests failed")
            
            # Determine overall success
            import_success = result['import_result']['success']
            instantiation_success = result['instantiation_result']['success']
            functionality_success = not test_cases or result['functionality_result']['success']
            
            result['overall_success'] = import_success and instantiation_success and functionality_success
            
            if result['overall_success']:
                logger.info(f"Validation SUCCESS for: {module_path}")
            else:
                logger.warning(f"Validation FAILED for: {module_path}")
            
        except Exception as e:
            error_msg = f"Validation process error: {e}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
            if self.config['detailed_logging']:
                logger.error(traceback.format_exc())
        
        return result
    
    def save_results(self, filename: str = 'model_validation_results.json'):
        """Save validation results to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Convert results to serializable format
                serializable_results = self._make_serializable(self.results)
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _make_serializable(self, obj, depth=0, max_depth=10):
        """Convert object to serializable format with recursion depth limit"""
        if depth > max_depth:
            return f"<Max recursion depth {max_depth} reached>"
            
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, dict):
            # Fix: Get items list first to avoid "dictionary changed size during iteration"
            items = list(obj.items())
            return {k: self._make_serializable(v, depth+1, max_depth) for k, v in items}
        elif isinstance(obj, list):
            return [self._make_serializable(item, depth+1, max_depth) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item, depth+1, max_depth) for item in obj)
        elif hasattr(obj, '__dict__'):
            # Fix: Get __dict__ copy to avoid modification during iteration
            obj_dict = obj.__dict__.copy()
            return self._make_serializable(obj_dict, depth+1, max_depth)
        else:
            return str(obj)

def create_translation_test_cases():
    """Create test cases for translation model"""
    return [
        {
            'name': 'English to Chinese translation',
            'operation': 'translate',
            'data': {
                'text': 'hello world',
                'source_lang': 'en',
                'target_lang': 'zh',
                'lang': 'en'
            },
            'expected_keys': ['success', 'translation', 'source_text']
        },
        {
            'name': 'Chinese to English translation',
            'operation': 'translate',
            'data': {
                'text': '你好世界',
                'source_lang': 'zh',
                'target_lang': 'en',
                'lang': 'en'
            },
            'expected_keys': ['success', 'translation', 'source_text']
        },
        {
            'name': 'Batch translation',
            'operation': 'batch_translate',
            'data': {
                'texts': ['hello', 'thank you', 'good morning'],
                'source_lang': 'en',
                'target_lang': 'zh',
                'lang': 'en'
            },
            'expected_keys': ['success']
        }
    ]

def create_math_test_cases():
    """Create test cases for math model"""
    return [
        {
            'name': 'Expression evaluation',
            'operation': 'evaluate_expression',
            'data': {
                'expression': '2 + 3 * 5',
                'variables': None
            },
            'expected_keys': ['status', 'result']
        },
        {
            'name': 'Equation solving',
            'operation': 'solve_equation',
            'data': {
                'equation': '2*x + 5 = 15',
                'variable': 'x'
            },
            'expected_keys': ['status', 'solutions']
        }
    ]

def create_planning_test_cases():
    """Create test cases for planning model"""
    return [
        {
            'name': 'Create simple plan',
            'operation': 'create_plan',
            'data': {
                'goal': 'organize a meeting',
                'available_models': ['manager', 'language', 'knowledge']
            },
            'expected_keys': ['status', 'success']
        },
        {
            'name': 'Analyze goal complexity',
            'operation': 'analyze_goal_complexity',
            'data': {
                'goal': 'implement a new feature'
            },
            'expected_keys': ['score', 'level']
        }
    ]

def create_metacognition_test_cases():
    """Create test cases for metacognition model"""
    return [
        {
            'name': 'Apply metacognition',
            'operation': 'apply_metacognition',
            'data': {
                'cognitive_state': {
                    'task': 'learning new concept',
                    'confidence': 0.7,
                    'focus': 0.8
                }
            },
            'expected_keys': ['success', 'failure_message']
        },
        {
            'name': 'Strategy selection',
            'operation': 'strategy_selection',
            'data': {
                'task_description': 'solve a complex problem',
                'available_strategies': ['divide_and_conquer', 'brute_force', 'heuristic']
            },
            'expected_keys': ['success', 'selection_method', 'selection_timestamp']
        }
    ]

def create_advanced_reasoning_test_cases():
    """Create test cases for advanced reasoning model"""
    return [
        {
            'name': 'Logical reasoning',
            'operation': 'logical_reasoning',
            'data': {
                'premise': 'All humans are mortal. Socrates is a human.',
                'query': 'Is Socrates mortal?'
            },
            'expected_keys': ['status', 'result']
        },
        {
            'name': 'Causal inference',
            'operation': 'causal_inference',
            'data': {
                'cause': 'raining',
                'effect': 'wet ground'
            },
            'expected_keys': ['status', 'result']
        }
    ]

def main():
    """Main validation routine"""
    print("=" * 70)
    print("Robust Model Validation Framework")
    print("=" * 70)
    
    # Configuration
    config = {
        'max_retries': 2,
        'timeout_seconds': 30,
        'skip_on_error': False,
        'detailed_logging': True
    }
    
    # Create validator
    validator = ModelValidator(config)
    
    # Define models to validate
    models_to_validate = [
        {
            'name': 'Translation Model',
            'module_path': 'core.models.translation.unified_translation_model',
            'class_name': 'UnifiedTranslationModel',
            'test_cases': create_translation_test_cases()
        },
        {
            'name': 'Math Model',
            'module_path': 'core.models.mathematics.unified_mathematics_model',
            'class_name': 'UnifiedMathematicsModel',
            'test_cases': create_math_test_cases()
        },
        {
            'name': 'Planning Model',
            'module_path': 'core.models.planning.unified_planning_model',
            'class_name': 'UnifiedPlanningModel',
            'test_cases': create_planning_test_cases()
        },
        {
            'name': 'Metacognition Model',
            'module_path': 'core.models.metacognition.unified_metacognition_model',
            'class_name': 'UnifiedMetacognitionModel',
            'test_cases': create_metacognition_test_cases()
        },
        {
            'name': 'Advanced Reasoning Model',
            'module_path': 'core.models.advanced_reasoning.unified_advanced_reasoning_model',
            'class_name': 'UnifiedAdvancedReasoningModel',
            'test_cases': create_advanced_reasoning_test_cases()
        }
    ]
    
    # Run validation for each model
    all_results = {}
    
    for model_spec in models_to_validate:
        print(f"\nValidating: {model_spec['name']}")
        print("-" * 40)
        
        result = validator.run_comprehensive_validation(model_spec)
        all_results[model_spec['name']] = result
        
        # Print summary
        if result['overall_success']:
            print(f"✅ SUCCESS: {model_spec['name']} passed all validation checks")
        else:
            print(f"❌ FAILED: {model_spec['name']} validation failed")
            
            # Print errors
            if result['errors']:
                print("  Errors:")
                for error in result['errors']:
                    print(f"    - {error}")
            
            # Print import errors
            if result['import_result'] and result['import_result']['errors']:
                print("  Import errors:")
                for error in result['import_result']['errors']:
                    print(f"    - {error}")
            
            # Print instantiation errors
            if result['instantiation_result'] and result['instantiation_result']['errors']:
                print("  Instantiation errors:")
                for error in result['instantiation_result']['errors']:
                    print(f"    - {error}")
            
            # Print functionality errors
            if result['functionality_result']:
                failed_tests = result['functionality_result']['failed_tests']
                total_tests = result['functionality_result']['total_tests']
                if failed_tests > 0:
                    print(f"  Functionality: {failed_tests}/{total_tests} tests failed")
    
    # Save results
    validator.results = all_results
    validator.save_results()
    
    # Generate final summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)
    
    successful_models = []
    failed_models = []
    
    for model_name, result in all_results.items():
        if result['overall_success']:
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
    
    print(f"✅ Successful models ({len(successful_models)}):")
    for model_name in successful_models:
        print(f"  - {model_name}")
    
    if failed_models:
        print(f"\n❌ Failed models ({len(failed_models)}):")
        for model_name in failed_models:
            print(f"  - {model_name}")
    
    print(f"\nOverall: {len(successful_models)}/{len(models_to_validate)} models validated successfully")
    
    if failed_models:
        print("\nRecommendations:")
        print("1. Check import paths and module names")
        print("2. Verify model class implementations")
        print("3. Check for missing dependencies")
        print("4. Review error logs in model_validation.log")

if __name__ == "__main__":
    main()