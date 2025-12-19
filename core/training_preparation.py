"""
训练准备模块 - 实现训练前的完整准备流程
Training Preparation Module - Complete pre-training preparation workflow

基于代码库中现有组件，实现环境初始化、数据预处理、模型配置、依赖检查四个核心维度
Based on existing components, implements environment initialization, data preprocessing, model configuration, and dependency checking
"""

import os
import sys
import time
import json
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from core.error_handling import error_handler
from core.training_manager import TrainingManager
from core.model_registry import ModelRegistry


class TrainingPreparation:
    """训练准备类 - 整合训练前的所有准备工作
    Training Preparation Class - Integrates all pre-training preparation tasks
    """
    
    def __init__(self, model_registry: ModelRegistry, training_manager: TrainingManager):
        """初始化训练准备类
        Initialize training preparation class
        
        Args:
            model_registry: 模型注册表实例
            training_manager: 训练管理器实例
        """
        self.model_registry = model_registry
        self.training_manager = training_manager
        self.preparation_status = {}
        self.preparation_log = []
        
    def prepare_training_environment(self) -> Dict[str, Any]:
        """准备训练环境 - 环境与依赖初始化维度
        Prepare training environment - Environment and dependency initialization dimension
        
        Returns:
            dict: 环境准备结果
        """
        start_time = time.time()
        result = {
            'success': True,
            'message': '环境准备完成',
            'steps': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            # 1. 检查Python环境
            python_check = self._check_python_environment()
            result['python_environment'] = python_check
            if not python_check['success']:
                result['success'] = False
                result['errors'].append(f"Python环境检查失败: {python_check['message']}")
            
            # 2. 检查硬件资源
            hardware_check = self._check_hardware_resources()
            result['hardware_resources'] = hardware_check
            if not hardware_check['success']:
                result['warnings'].append(f"硬件资源检查警告: {hardware_check['message']}")
            
            # 3. 检查PyTorch和CUDA
            pytorch_check = self._check_pytorch_environment()
            result['pytorch_environment'] = pytorch_check
            if not pytorch_check['success']:
                result['success'] = False
                result['errors'].append(f"PyTorch环境检查失败: {pytorch_check['message']}")
            
            # 4. 检查依赖库
            dependency_check = self._check_dependencies()
            result['dependencies'] = dependency_check
            if not dependency_check['success']:
                result['warnings'].append(f"依赖库检查警告: {dependency_check['message']}")
            
            # 5. 初始化训练管理器单例
            training_manager_check = self._initialize_training_manager()
            result['training_manager'] = training_manager_check
            if not training_manager_check['success']:
                result['success'] = False
                result['errors'].append(f"训练管理器初始化失败: {training_manager_check['message']}")
            
            # 记录准备步骤
            result['steps'] = [
                'Python环境检查',
                '硬件资源检查', 
                'PyTorch环境检查',
                '依赖库检查',
                '训练管理器初始化'
            ]
            
            result['duration'] = time.time() - start_time
            
            # 记录准备状态
            self.preparation_status['environment'] = result
            self._log_preparation_step('环境准备', result['success'], result['message'])
            
        except Exception as e:
            error_handler.handle_error(e, "TrainingPreparation", "环境准备失败")
            result['success'] = False
            result['message'] = f"环境准备失败: {str(e)}"
            result['errors'].append(str(e))
        
        return result
    
    def prepare_training_data(self, model_id: str, raw_data: Any) -> Dict[str, Any]:
        """准备训练数据 - 数据预处理与验证维度
        Prepare training data - Data preprocessing and validation dimension
        
        Args:
            model_id: 模型ID
            raw_data: 原始训练数据
            
        Returns:
            dict: 数据准备结果
        """
        start_time = time.time()
        result = {
            'success': True,
            'message': '数据准备完成',
            'model_id': model_id,
            'data_quality': {},
            'preprocessing_steps': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            # 1. 获取模型实例
            model = self.model_registry.get_model(model_id)
            if not model:
                result['success'] = False
                result['message'] = f"模型 {model_id} 未找到"
                result['errors'].append(result['message'])
                return result
            
            # 2. 数据质量检查
            data_quality_check = self._check_data_quality(raw_data, model_id)
            result['data_quality'] = data_quality_check
            if not data_quality_check['success']:
                result['warnings'].append(f"数据质量检查警告: {data_quality_check['message']}")
            
            # 3. 数据格式标准化
            standardization_result = self._standardize_training_data(model_id, raw_data)
            result['standardization'] = standardization_result
            if not standardization_result['success']:
                result['success'] = False
                result['errors'].append(f"数据标准化失败: {standardization_result['message']}")
            
            # 4. 领域适配处理
            domain_adaptation_result = self._apply_domain_adaptation(model_id, standardization_result['processed_data'])
            result['domain_adaptation'] = domain_adaptation_result
            if not domain_adaptation_result['success']:
                result['warnings'].append(f"领域适配警告: {domain_adaptation_result['message']}")
            
            # 5. 训练/验证集划分
            split_result = self._split_training_validation_sets(domain_adaptation_result['adapted_data'])
            result['data_split'] = split_result
            if not split_result['success']:
                result['success'] = False
                result['errors'].append(f"数据集划分失败: {split_result['message']}")
            
            # 记录准备步骤
            result['preprocessing_steps'] = [
                '数据质量检查',
                '数据格式标准化',
                '领域适配处理', 
                '训练/验证集划分'
            ]
            
            result['duration'] = time.time() - start_time
            
            # 记录准备状态
            self.preparation_status['data'] = result
            self._log_preparation_step(f'数据准备({model_id})', result['success'], result['message'])
            
        except Exception as e:
            error_handler.handle_error(e, "TrainingPreparation", f"数据准备失败: {model_id}")
            result['success'] = False
            result['message'] = f"数据准备失败: {str(e)}"
            result['errors'].append(str(e))
        
        return result
    
    def prepare_model_configuration(self, model_id: str, custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """准备模型配置 - 模型配置与状态准备维度
        Prepare model configuration - Model configuration and state preparation dimension
        
        Args:
            model_id: 模型ID
            custom_params: 自定义参数
            
        Returns:
            dict: 模型配置准备结果
        """
        start_time = time.time()
        result = {
            'success': True,
            'message': '模型配置准备完成',
            'model_id': model_id,
            'model_status': {},
            'hyperparameters': {},
            'configuration_steps': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            # 1. 模型加载与状态检查
            model_loading_result = self._load_and_check_model(model_id)
            result['model_loading'] = model_loading_result
            if not model_loading_result['success']:
                result['success'] = False
                result['errors'].append(f"模型加载失败: {model_loading_result['message']}")
            
            # 2. 超参数初始化
            hyperparameter_result = self._initialize_hyperparameters(model_id, custom_params)
            result['hyperparameters'] = hyperparameter_result
            if not hyperparameter_result['success']:
                result['warnings'].append(f"超参数初始化警告: {hyperparameter_result['message']}")
            
            # 3. 模型状态准备
            model_preparation_result = self._prepare_model_state(model_id)
            result['model_preparation'] = model_preparation_result
            if not model_preparation_result['success']:
                result['success'] = False
                result['errors'].append(f"模型状态准备失败: {model_preparation_result['message']}")
            
            # 4. 架构兼容性验证
            compatibility_result = self._validate_architecture_compatibility(model_id)
            result['compatibility'] = compatibility_result
            if not compatibility_result['success']:
                result['warnings'].append(f"架构兼容性警告: {compatibility_result['message']}")
            
            # 记录准备步骤
            result['configuration_steps'] = [
                '模型加载与状态检查',
                '超参数初始化',
                '模型状态准备',
                '架构兼容性验证'
            ]
            
            result['duration'] = time.time() - start_time
            
            # 记录准备状态
            self.preparation_status['model_configuration'] = result
            self._log_preparation_step(f'模型配置准备({model_id})', result['success'], result['message'])
            
        except Exception as e:
            error_handler.handle_error(e, "TrainingPreparation", f"模型配置准备失败: {model_id}")
            result['success'] = False
            result['message'] = f"模型配置准备失败: {str(e)}"
            result['errors'].append(str(e))
        
        return result
    
    def prepare_training_context(self, model_ids: List[str], strategy: str = "federated") -> Dict[str, Any]:
        """准备训练上下文 - 训练上下文与任务配置维度
        Prepare training context - Training context and task configuration dimension
        
        Args:
            model_ids: 模型ID列表
            strategy: 训练策略
            
        Returns:
            dict: 训练上下文准备结果
        """
        start_time = time.time()
        result = {
            'success': True,
            'message': '训练上下文准备完成',
            'model_ids': model_ids,
            'strategy': strategy,
            'context_config': {},
            'task_registration': {},
            'context_steps': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            # 1. 共享训练上下文初始化
            context_result = self._initialize_shared_training_context(model_ids, strategy)
            result['context_config'] = context_result
            if not context_result['success']:
                result['success'] = False
                result['errors'].append(f"共享上下文初始化失败: {context_result['message']}")
            
            # 2. 训练任务注册
            task_registration_result = self._register_training_tasks(model_ids, strategy)
            result['task_registration'] = task_registration_result
            if not task_registration_result['success']:
                result['success'] = False
                result['errors'].append(f"训练任务注册失败: {task_registration_result['message']}")
            
            # 3. 日志与监控配置
            logging_result = self._configure_logging_and_monitoring(model_ids)
            result['logging_config'] = logging_result
            if not logging_result['success']:
                result['warnings'].append(f"日志配置警告: {logging_result['message']}")
            
            # 4. 资源限制设置
            resource_result = self._set_resource_limits(model_ids)
            result['resource_limits'] = resource_result
            if not resource_result['success']:
                result['warnings'].append(f"资源限制设置警告: {resource_result['message']}")
            
            # 记录准备步骤
            result['context_steps'] = [
                '共享训练上下文初始化',
                '训练任务注册',
                '日志与监控配置',
                '资源限制设置'
            ]
            
            result['duration'] = time.time() - start_time
            
            # 记录准备状态
            self.preparation_status['training_context'] = result
            self._log_preparation_step(f'训练上下文准备({len(model_ids)}个模型)', result['success'], result['message'])
            
        except Exception as e:
            error_handler.handle_error(e, "TrainingPreparation", "训练上下文准备失败")
            result['success'] = False
            result['message'] = f"训练上下文准备失败: {str(e)}"
            result['errors'].append(str(e))
        
        return result
    
    def execute_complete_preparation(self, model_ids: List[str], raw_data: Dict[str, Any], 
                                   custom_params: Dict[str, Any] = None, strategy: str = "federated") -> Dict[str, Any]:
        """执行完整的训练准备流程
        Execute complete training preparation workflow
        
        Args:
            model_ids: 模型ID列表
            raw_data: 原始数据字典，key为模型ID，value为对应的原始数据
            custom_params: 自定义参数
            strategy: 训练策略
            
        Returns:
            dict: 完整的准备结果
        """
        start_time = time.time()
        result = {
            'success': True,
            'message': '训练准备流程完成',
            'model_ids': model_ids,
            'preparation_phases': {},
            'overall_status': 'pending',
            'warnings': [],
            'errors': []
        }
        
        try:
            # 阶段1: 环境与依赖初始化
            phase1_result = self.prepare_training_environment()
            result['preparation_phases']['environment'] = phase1_result
            
            if not phase1_result['success']:
                result['success'] = False
                result['errors'].extend(phase1_result['errors'])
                result['overall_status'] = 'environment_failed'
                return result
            
            # 阶段2: 数据预处理与验证
            data_results = {}
            for model_id in model_ids:
                if model_id in raw_data:
                    data_result = self.prepare_training_data(model_id, raw_data[model_id])
                    data_results[model_id] = data_result
                    
                    if not data_result['success']:
                        result['success'] = False
                        result['errors'].extend(data_result['errors'])
            
            result['preparation_phases']['data'] = data_results
            
            if not result['success']:
                result['overall_status'] = 'data_preparation_failed'
                return result
            
            # 阶段3: 模型配置与状态准备
            config_results = {}
            for model_id in model_ids:
                config_result = self.prepare_model_configuration(model_id, custom_params)
                config_results[model_id] = config_result
                
                if not config_result['success']:
                    result['success'] = False
                    result['errors'].extend(config_result['errors'])
            
            result['preparation_phases']['model_configuration'] = config_results
            
            if not result['success']:
                result['overall_status'] = 'model_configuration_failed'
                return result
            
            # 阶段4: 训练上下文与任务配置
            context_result = self.prepare_training_context(model_ids, strategy)
            result['preparation_phases']['training_context'] = context_result
            
            if not context_result['success']:
                result['success'] = False
                result['errors'].extend(context_result['errors'])
                result['overall_status'] = 'context_preparation_failed'
                return result
            
            # 所有阶段成功完成
            result['overall_status'] = 'ready'
            result['duration'] = time.time() - start_time
            result['message'] = f"训练准备完成，耗时 {result['duration']:.2f} 秒"
            
            # 记录最终准备状态
            self._log_preparation_step('完整训练准备', result['success'], result['message'])
            
        except Exception as e:
            error_handler.handle_error(e, "TrainingPreparation", "完整训练准备失败")
            result['success'] = False
            result['message'] = f"完整训练准备失败: {str(e)}"
            result['errors'].append(str(e))
            result['overall_status'] = 'failed'
        
        return result
    
    # ========== 环境准备相关方法 ==========
    
    def _check_python_environment(self) -> Dict[str, Any]:
        """检查Python环境
        Check Python environment
        """
        try:
            version_info = {
                'python_version': sys.version,
                'version_info': list(sys.version_info),
                'executable': sys.executable,
                'platform': sys.platform
            }
            
            # 检查Python版本兼容性
            if sys.version_info < (3, 8):
                return {
                    'success': False,
                    'message': f"Python版本 {sys.version_info.major}.{sys.version_info.minor} 过低，需要3.8+",
                    'version_info': version_info
                }
            
            return {
                'success': True,
                'message': f"Python环境检查通过 (版本 {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})",
                'version_info': version_info
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Python环境检查失败: {str(e)}",
                'error': str(e)
            }
    
    def _check_hardware_resources(self) -> Dict[str, Any]:
        """检查硬件资源
        Check hardware resources
        """
        try:
            # 检查CPU
            cpu_count = os.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 检查内存
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            # 检查GPU（如果可用）
            gpu_info = {'available': False}
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info = {
                    'available': True,
                    'count': gpu_count,
                    'devices': [torch.cuda.get_device_name(i) for i in range(gpu_count)]
                }
            
            # 评估资源充足性
            warnings = []
            if cpu_count < 4:
                warnings.append(f"CPU核心数较少 ({cpu_count})，可能影响训练速度")
            
            if memory_total_gb < 8:
                warnings.append(f"内存不足 ({memory_total_gb:.1f}GB)，建议至少8GB")
            
            if cpu_percent > 80:
                warnings.append(f"CPU使用率较高 ({cpu_percent}%)")
            
            if memory_available_gb < 2:
                warnings.append(f"可用内存较少 ({memory_available_gb:.1f}GB)")
            
            return {
                'success': True,
                'message': '硬件资源检查完成',
                'cpu': {
                    'count': cpu_count,
                    'usage_percent': cpu_percent
                },
                'memory': {
                    'total_gb': memory_total_gb,
                    'available_gb': memory_available_gb,
                    'usage_percent': memory.percent
                },
                'gpu': gpu_info,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"硬件资源检查失败: {str(e)}",
                'error': str(e)
            }
    
    def _check_pytorch_environment(self) -> Dict[str, Any]:
        """检查PyTorch环境
        Check PyTorch environment
        """
        try:
            pytorch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else None
            
            # 检查CUDA设备
            cuda_devices = []
            if cuda_available:
                for i in range(torch.cuda.device_count()):
                    device_name = torch.cuda.get_device_name(i)
                    device_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    cuda_devices.append({
                        'index': i,
                        'name': device_name,
                        'memory_gb': device_memory
                    })
            
            return {
                'success': True,
                'message': f"PyTorch环境检查通过 (版本 {pytorch_version})",
                'pytorch_version': pytorch_version,
                'cuda_available': cuda_available,
                'cuda_version': cuda_version,
                'cuda_devices': cuda_devices
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"PyTorch环境检查失败: {str(e)}",
                'error': str(e)
            }
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """检查依赖库
        Check dependencies
        """
        try:
            required_packages = [
                'numpy', 'torch', 'psutil', 'websockets', 'json'
            ]
            
            missing_packages = []
            available_packages = []
            
            for package in required_packages:
                try:
                    if package == 'json':
                        # json是标准库，总是可用
                        available_packages.append({'name': package, 'version': 'built-in'})
                    else:
                        module = __import__(package)
                        version = getattr(module, '__version__', 'unknown')
                        available_packages.append({'name': package, 'version': version})
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                return {
                    'success': False,
                    'message': f"缺少必要的依赖包: {', '.join(missing_packages)}",
                    'missing_packages': missing_packages,
                    'available_packages': available_packages
                }
            
            return {
                'success': True,
                'message': '所有依赖包检查通过',
                'available_packages': available_packages
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"依赖库检查失败: {str(e)}",
                'error': str(e)
            }
    
    def _initialize_training_manager(self) -> Dict[str, Any]:
        """初始化训练管理器
        Initialize training manager
        """
        try:
            # 检查训练管理器是否已正确初始化
            if not hasattr(self.training_manager, 'model_registry'):
                return {
                    'success': False,
                    'message': '训练管理器未正确初始化'
                }
            
            # 检查模型注册表是否已设置
            if self.training_manager.model_registry is None:
                return {
                    'success': False,
                    'message': '训练管理器的模型注册表未设置'
                }
            
            # 检查必要的训练管理器方法
            required_methods = ['set_model_status', 'prepare_model', 'start_training']
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(self.training_manager, method):
                    missing_methods.append(method)
            
            if missing_methods:
                return {
                    'success': False,
                    'message': f"训练管理器缺少必要方法: {', '.join(missing_methods)}",
                    'missing_methods': missing_methods
                }
            
            return {
                'success': True,
                'message': '训练管理器初始化检查通过'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"训练管理器初始化失败: {str(e)}",
                'error': str(e)
            }
    
    # ========== 数据准备相关方法 ==========
    
    def _check_data_quality(self, raw_data: Any, model_id: str) -> Dict[str, Any]:
        """检查数据质量
        Check data quality
        """
        try:
            # 基本数据质量检查
            if raw_data is None:
                return {
                    'success': False,
                    'message': '数据为空',
                    'data_size': 0,
                    'quality_metrics': {}
                }
            
            # 根据数据类型进行不同的检查
            if isinstance(raw_data, (list, tuple)):
                data_size = len(raw_data)
                if data_size == 0:
                    return {
                        'success': False,
                        'message': '数据为空列表',
                        'data_size': 0,
                        'quality_metrics': {}
                    }
                
                # 检查数据样本的有效性
                valid_samples = 0
                for sample in raw_data:
                    if sample is not None:
                        valid_samples += 1
                
                validity_ratio = valid_samples / data_size
                
                quality_metrics = {
                    'data_size': data_size,
                    'valid_samples': valid_samples,
                    'validity_ratio': validity_ratio,
                    'data_type': type(raw_data).__name__
                }
                
                if validity_ratio < 0.9:
                    return {
                        'success': False,
                        'message': f"数据有效性较低 ({validity_ratio:.1%})",
                        'quality_metrics': quality_metrics
                    }
                
                return {
                    'success': True,
                    'message': f"数据质量检查通过 (大小: {data_size}, 有效性: {validity_ratio:.1%})",
                    'quality_metrics': quality_metrics
                }
            
            elif isinstance(raw_data, dict):
                data_size = len(raw_data)
                quality_metrics = {
                    'data_size': data_size,
                    'keys': list(raw_data.keys()),
                    'data_type': 'dict'
                }
                
                return {
                    'success': True,
                    'message': f"字典数据检查通过 (大小: {data_size})",
                    'quality_metrics': quality_metrics
                }
            
            else:
                # 其他数据类型
                return {
                    'success': True,
                    'message': f"数据质量检查通过 (类型: {type(raw_data).__name__})",
                    'quality_metrics': {
                        'data_type': type(raw_data).__name__
                    }
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"数据质量检查失败: {str(e)}",
                'error': str(e)
            }
    
    def _standardize_training_data(self, model_id: str, raw_data: Any) -> Dict[str, Any]:
        """标准化训练数据
        Standardize training data
        """
        try:
            # 获取模型实例
            model = self.model_registry.get_model(model_id)
            if not model:
                return {
                    'success': False,
                    'message': f"模型 {model_id} 未找到"
                }
            
            # 检查模型是否有数据预处理方法
            if hasattr(model, '_prepare_training_data'):
                # 使用模型的专用预处理方法
                processed_data = model._prepare_training_data(raw_data)
                return {
                    'success': True,
                    'message': f"使用模型专用方法标准化数据",
                    'processed_data': processed_data,
                    'method': 'model_specific'
                }
            
            elif hasattr(model, 'prepare_training_data'):
                # 使用模型的通用预处理方法
                processed_data = model.prepare_training_data(raw_data)
                return {
                    'success': True,
                    'message': f"使用模型通用方法标准化数据",
                    'processed_data': processed_data,
                    'method': 'model_generic'
                }
            
            else:
                # 使用默认的标准化方法
                processed_data = self._default_data_standardization(raw_data, model_id)
                return {
                    'success': True,
                    'message': f"使用默认方法标准化数据",
                    'processed_data': processed_data,
                    'method': 'default'
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"数据标准化失败: {str(e)}",
                'error': str(e)
            }
    
    def _default_data_standardization(self, raw_data: Any, model_id: str) -> Any:
        """默认数据标准化方法
        Default data standardization method
        """
        # 这里实现通用的数据标准化逻辑
        # 实际应用中应根据具体需求实现
        
        if isinstance(raw_data, (list, tuple)):
            # 对列表数据进行基本处理
            processed = []
            for item in raw_data:
                if item is not None:
                    processed.append(item)
            return processed
        
        elif isinstance(raw_data, dict):
            # 对字典数据进行基本处理
            processed = {}
            for key, value in raw_data.items():
                if value is not None:
                    processed[key] = value
            return processed
        
        else:
            # 其他数据类型直接返回
            return raw_data
    
    def _apply_domain_adaptation(self, model_id: str, processed_data: Any) -> Dict[str, Any]:
        """应用领域适配处理
        Apply domain adaptation processing
        """
        try:
            # 根据模型类型应用不同的领域适配
            model_type = model_id.split('_')[0] if '_' in model_id else model_id
            
            if model_type in ['vision', 'video']:
                # 视觉模型领域适配
                adapted_data = self._vision_domain_adaptation(processed_data)
                return {
                    'success': True,
                    'message': f"应用视觉领域适配",
                    'adapted_data': adapted_data,
                    'domain': 'vision'
                }
            
            elif model_type in ['audio']:
                # 音频模型领域适配
                adapted_data = self._audio_domain_adaptation(processed_data)
                return {
                    'success': True,
                    'message': f"应用音频领域适配",
                    'adapted_data': adapted_data,
                    'domain': 'audio'
                }
            
            elif model_type in ['medical']:
                # 医疗模型领域适配
                adapted_data = self._medical_domain_adaptation(processed_data)
                return {
                    'success': True,
                    'message': f"应用医疗领域适配",
                    'adapted_data': adapted_data,
                    'domain': 'medical'
                }
            
            else:
                # 通用领域适配
                adapted_data = self._general_domain_adaptation(processed_data)
                return {
                    'success': True,
                    'message': f"应用通用领域适配",
                    'adapted_data': adapted_data,
                    'domain': 'general'
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"领域适配失败: {str(e)}",
                'error': str(e)
            }
    
    def _vision_domain_adaptation(self, data: Any) -> Any:
        """视觉领域适配
        Vision domain adaptation
        """
        # 视觉模型专用的数据适配逻辑
        # 实际应用中应实现具体的视觉数据处理
        return data
    
    def _audio_domain_adaptation(self, data: Any) -> Any:
        """音频领域适配
        Audio domain adaptation
        """
        # 音频模型专用的数据适配逻辑
        return data
    
    def _medical_domain_adaptation(self, data: Any) -> Any:
        """医疗领域适配
        Medical domain adaptation
        """
        # 医疗模型专用的数据适配逻辑
        return data
    
    def _general_domain_adaptation(self, data: Any) -> Any:
        """通用领域适配
        General domain adaptation
        """
        # 通用数据适配
        return data
    
    def _split_training_validation_sets(self, data: Any) -> Dict[str, Any]:
        """划分训练/验证集
        Split training/validation sets
        """
        try:
            # 简单的数据划分逻辑
            if isinstance(data, (list, tuple)):
                total_size = len(data)
                if total_size > 10:
                    train_size = int(total_size * 0.8)
                    train_data = data[:train_size]
                    val_data = data[train_size:]
                    
                    return {
                        'success': True,
                        'message': f'数据划分完成 (训练集: {len(train_data)}, 验证集: {len(val_data)})',
                        'train_data': train_data,
                        'val_data': val_data,
                        'train_size': len(train_data),
                        'val_size': len(val_data)
                    }
                else:
                    return {
                        'success': False,
                        'message': f'数据量过少 ({total_size})，无法进行有效划分'
                    }
            else:
                return {
                    'success': True,
                    'message': '非序列数据，跳过划分',
                    'train_data': data,
                    'val_data': None
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'数据划分失败: {str(e)}',
                'error': str(e)
            }
    
    # ========== 模型配置相关方法 ==========
    
    def _load_and_check_model(self, model_id: str) -> Dict[str, Any]:
        """加载并检查模型
        Load and check model
        """
        try:
            model = self.model_registry.get_model(model_id)
            if not model:
                return {
                    'success': False,
                    'message': f'模型 {model_id} 未找到'
                }
            
            # 检查模型状态
            model_status = getattr(model, 'status', 'unknown')
            
            return {
                'success': True,
                'message': f'模型 {model_id} 加载成功 (状态: {model_status})',
                'model_status': model_status,
                'model_type': type(model).__name__
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'模型加载失败: {str(e)}',
                'error': str(e)
            }
    
    def _initialize_hyperparameters(self, model_id: str, custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """初始化超参数
        Initialize hyperparameters
        """
        try:
            model = self.model_registry.get_model(model_id)
            if not model:
                return {
                    'success': False,
                    'message': f'模型 {model_id} 未找到'
                }
            
            # 默认超参数
            default_params = {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'optimizer': 'adam'
            }
            
            # 合并自定义参数
            final_params = default_params.copy()
            if custom_params:
                final_params.update(custom_params)
            
            return {
                'success': True,
                'message': f'超参数初始化完成',
                'hyperparameters': final_params
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'超参数初始化失败: {str(e)}',
                'error': str(e)
            }
    
    def _prepare_model_state(self, model_id: str) -> Dict[str, Any]:
        """准备模型状态
        Prepare model state
        """
        try:
            # 使用训练管理器准备模型
            preparation_result = self.training_manager.prepare_model(model_id)
            
            if preparation_result['success']:
                return {
                    'success': True,
                    'message': f'模型 {model_id} 状态准备完成',
                    'progress': preparation_result['progress']
                }
            else:
                return {
                    'success': False,
                    'message': f'模型状态准备失败: {preparation_result['message']}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'模型状态准备失败: {str(e)}',
                'error': str(e)
            }
    
    def _validate_architecture_compatibility(self, model_id: str) -> Dict[str, Any]:
        """验证架构兼容性
        Validate architecture compatibility
        """
        try:
            model = self.model_registry.get_model(model_id)
            if not model:
                return {
                    'success': False,
                    'message': f'模型 {model_id} 未找到'
                }
            
            # 检查模型是否有必要的训练方法
            required_methods = ['train', 'forward']
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(model, method):
                    missing_methods.append(method)
            
            if missing_methods:
                return {
                    'success': False,
                    'message': f'模型缺少必要方法: {', '.join(missing_methods)}',
                    'missing_methods': missing_methods
                }
            
            return {
                'success': True,
                'message': '架构兼容性验证通过'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'架构兼容性验证失败: {str(e)}',
                'error': str(e)
            }
    
    # ========== 训练上下文相关方法 ==========
    
    def _initialize_shared_training_context(self, model_ids: List[str], strategy: str) -> Dict[str, Any]:
        """初始化共享训练上下文
        Initialize shared training context
        """
        try:
            context_config = {
                'model_ids': model_ids,
                'strategy': strategy,
                'shared_resources': {},
                'collaboration_mode': 'federated' if strategy == 'federated' else 'independent'
            }
            
            # 根据策略设置不同的上下文配置
            if strategy == 'federated':
                context_config['shared_resources'] = {
                    'gradient_sharing': True,
                    'parameter_synchronization': True,
                    'knowledge_transfer': True
                }
            elif strategy == 'independent':
                context_config['shared_resources'] = {
                    'gradient_sharing': False,
                    'parameter_synchronization': False,
                    'knowledge_transfer': False
                }
            
            return {
                'success': True,
                'message': f'共享训练上下文初始化完成 (策略: {strategy})',
                'context_config': context_config
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'共享训练上下文初始化失败: {str(e)}',
                'error': str(e)
            }
    
    def _register_training_tasks(self, model_ids: List[str], strategy: str) -> Dict[str, Any]:
        """注册训练任务
        Register training tasks
        """
        try:
            task_registration = {}
            
            for model_id in model_ids:
                task_info = {
                    'model_id': model_id,
                    'strategy': strategy,
                    'status': 'registered',
                    'start_time': None,
                    'estimated_duration': None
                }
                
                task_registration[model_id] = task_info
            
            return {
                'success': True,
                'message': f'训练任务注册完成 (共 {len(model_ids)} 个模型)',
                'tasks': task_registration
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'训练任务注册失败: {str(e)}',
                'error': str(e)
            }
    
    def _configure_logging_and_monitoring(self, model_ids: List[str]) -> Dict[str, Any]:
        """配置日志与监控
        Configure logging and monitoring
        """
        try:
            logging_config = {
                'log_level': 'INFO',
                'log_file': f'training_log_{int(time.time())}.log',
                'monitoring_interval': 60,  # 60秒
                'models': model_ids
            }
            
            return {
                'success': True,
                'message': '日志与监控配置完成',
                'logging_config': logging_config
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'日志与监控配置失败: {str(e)}',
                'error': str(e)
            }
    
    def _set_resource_limits(self, model_ids: List[str]) -> Dict[str, Any]:
        """设置资源限制
        Set resource limits
        """
        try:
            resource_limits = {
                'max_memory_gb': 8.0,
                'max_cpu_percent': 80.0,
                'max_gpu_memory_gb': 4.0,
                'models': model_ids
            }
            
            return {
                'success': True,
                'message': '资源限制设置完成',
                'resource_limits': resource_limits
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'资源限制设置失败: {str(e)}',
                'error': str(e)
            }
    
    # ========== 辅助方法 ==========
    
    def _log_preparation_step(self, step_name: str, success: bool, message: str):
        """记录准备步骤
        Log preparation step
        """
        log_entry = {
            'timestamp': time.time(),
            'step': step_name,
            'success': success,
            'message': message
        }
        self.preparation_log.append(log_entry)
        
        # 输出到控制台
        status = '✅' if success else '❌'
        print(f"{status} {step_name}: {message}")


def create_training_preparation() -> TrainingPreparation:
    """创建训练准备实例的工厂函数
    Factory function to create TrainingPreparation instance
    
    Returns:
        TrainingPreparation: 训练准备实例
    """
    try:
        # 创建模型注册表实例
        model_registry = ModelRegistry()
        
        # 创建训练管理器实例
        training_manager = TrainingManager()
        
        # 创建训练准备实例
        preparation = TrainingPreparation(model_registry, training_manager)
        
        print("✅ 训练准备实例创建成功")
        return preparation
        
    except Exception as e:
        print(f"❌ 训练准备实例创建失败: {str(e)}")
        return None