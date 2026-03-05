import zlib
"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""
import time
import json
import os
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
# 延迟导入以避免循环依赖
# ModelRegistry 将在 __init__ 方法中延迟导入
from core.error_handling import error_handler
from core.fusion.multimodal import MultimodalFusion
from core.training_manager import TrainingManager
from core.self_learning import AGISelfLearningSystem as UnifiedSelfLearningSystem
from core.unified_cognitive_architecture import UnifiedCognitiveArchitecture
from core.enhanced_meta_cognition import EnhancedMetaCognition
from core.models.knowledge import KnowledgeModel
from core.intrinsic_motivation_system import IntrinsicMotivationSystem
from core.explainable_ai import ExplainableAI
from core.value_alignment import ValueAlignment
from core.config_manager import UnifiedConfigManager


"""
AGICoordinator类 - AGI系统核心协调器
AGICoordinator Class - Core AGI System Coordinator
"""
class AGICoordinator:
    """Self Soul  AGI中央协调器，管理和协调所有认知组件"""
    
    # AGI水平计算配置 - 动态可调整，避免硬编码
    AGI_LEVEL_CONFIG = {
        'model_coverage_weight': 0.25,
        'performance_weight': 0.35,
        'learning_weight': 0.20,
        'meta_cognition_weight': 0.10,
        'knowledge_weight': 0.10,
        'smoothing_factor': 0.7,  # 平滑因子，用于新旧值之间的平滑
        'min_agi_level': 0.1,
        'max_agi_level': 1.0,
        'adaptive_weights': True,  # 启用自适应权重调整
        'weight_adjustment_rate': 0.01  # 权重调整速率
    }

    # 系统配置 - 增加可配置性和灵活性
    SYSTEM_CONFIG = {
        'max_cpu_percent': 75,          # CPU使用率阈值，降低以避免过载
        'max_memory_percent': 75,       # 内存使用率阈值，与performance.yml保持一致
        'learning_loop_sleep_min': 3.0, # 学习循环最小休眠时间（秒）
        'learning_loop_sleep_max': 45.0,# 学习循环最大休眠时间（秒）
        'max_consecutive_failures': 5,  # 增加最大连续失败次数，增加容错性
        'performance_history_limit': 2000,  # 增加性能历史记录限制
        'max_experiences': 2000,        # 增加最大经验数
        'max_patterns': 1000,           # 增加最大模式数
        'max_task_types': 20,           # 增加最大任务类型数
        'long_term_memory_path': 'data/long_term_memory.json',  # 长期记忆存储路径
        'backup_memory_path': 'data/long_term_memory_backup.json',  # 备份路径
        'config_file': 'config/agi_coordinator_config.json',  # 外部配置文件路径
        'enable_adaptive_resource_management': True,  # 启用自适应资源管理
        'resource_check_interval': 5.0  # 资源检查间隔（秒）
    }
    def __init__(self, from_scratch: bool = False):
        # 延迟导入以避免循环依赖
        from core.model_registry import ModelRegistry
        
        # 使用ComponentFactory获取所有核心组件的单例实例，防止重复加载
        from core.memory_optimization import ComponentFactory
        
        # 获取统一认知架构单例
        self.cognitive_architecture = ComponentFactory.get_component('unified_cognitive_architecture', UnifiedCognitiveArchitecture)
        
        # 获取模型注册表单例
        self.model_registry = ComponentFactory.get_component('model_registry', ModelRegistry)
        
        # 获取多模态融合模块单例
        self.fusion_engine = ComponentFactory.get_component('multimodal_fusion', MultimodalFusion)
        
        # 获取训练管理器单例
        self.training_manager = ComponentFactory.get_component('training_manager', TrainingManager, self.model_registry, from_scratch=from_scratch)
        
        # 获取自主学习系统单例
        self.self_learning = ComponentFactory.get_component('self_learning', UnifiedSelfLearningSystem, from_scratch=True)
        
        # 初始化自我学习系统
        if not self.self_learning.initialized:
            self.self_learning.initialize()
        
        self.from_scratch = from_scratch
        
        if from_scratch:
            error_handler.log_info("AGICoordinator initialized in from-scratch mode", "AGI System")
        
        # 获取其他AGI组件的单例实例
        self.enhanced_meta_cognition = ComponentFactory.get_component('enhanced_meta_cognition', EnhancedMetaCognition)
        self.intrinsic_motivation = ComponentFactory.get_component('intrinsic_motivation', IntrinsicMotivationSystem)
        self.explainable_ai = ComponentFactory.get_component('explainable_ai', ExplainableAI)
        self.value_alignment = ComponentFactory.get_component('value_alignment', ValueAlignment)
        
        # 通过模型注册表获取知识模型
        self.structured_knowledge = self.model_registry.get_model('knowledge')
        
        # 初始化配置管理器并加载模型权重配置
        self.config_manager = UnifiedConfigManager()
        self.model_weights = self._load_model_weights_config()
        
        # 系统状态 - 动态评估而非硬编码
        self.system_state = {
            'status': 'initializing',
            'start_time': time.time(),
            'active_models': [],
            'performance_metrics': {},
            'agi_level': self._calculate_initial_agi_level(),  # 动态计算AGI水平
            'autonomy_level': 0.1,  # 初始自主性水平
            'generalization_score': 0.2,  # 初始泛化能力
            'learning_progress': 0.0,
            'task_success_rate': 0.0,
            'knowledge_coverage': 0.0
        }
        
        # 长期记忆存储
        self.long_term_memory = {
            'experiences': [],
            'learned_patterns': [],
            'optimization_history': [],
            'performance_history': []
        }
        
        # 元认知状态
        self.meta_cognition = {
            'confidence_level': 0.3,
            'knowledge_gaps': [],
            'learning_strategies': [],
            'self_assessment': {}
        }
        
        # 初始化AGI能力提升框架
        self._init_agi_enhancement_frameworks()
        
        # 更新系统状态
        self._update_system_state()
        
        # 加载长期记忆
        self.load_long_term_memory()
        
        # 启动自主学习循环
        self._start_autonomous_learning_loop()
        
        error_handler.log_info("AGI协调器初始化完成，统一认知架构已就绪", "AGICoordinator")
    
    def _load_model_weights_config(self):
        """加载模型权重配置，支持动态更新和自适应调整"""
        try:
            # 默认模型权重配置（作为备选）
            default_model_weights = {
                'language': 0.20, 'audio': 0.08, 'vision_image': 0.15,
                'vision_video': 0.10, 'sensor': 0.05, 'manager': 0.25,
                'planning': 0.08, 'reasoning': 0.12, 'learning': 0.07,
                'knowledge': 0.12, 'autonomous': 0.12, 'programming': 0.10,
                'emotion': 0.06, 'spatial': 0.09, 'computer_vision': 0.11,
                'motion': 0.13, 'prediction': 0.09, 'advanced_reasoning': 0.14,
                'data_fusion': 0.10, 'creative_problem_solving': 0.11,
                'meta_cognition': 0.15, 'value_alignment': 0.12,
                'finance': 0.08, 'medical': 0.10, 'collaboration': 0.10,
                'optimization': 0.09, 'computer': 0.08, 'mathematics': 0.10
            }
            
            # 尝试从配置文件加载模型权重
            config_file = self.SYSTEM_CONFIG.get('config_file', 'config/agi_coordinator_config.json')
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    
                    # 从配置中获取模型权重，如果不存在则使用默认值
                    config_model_weights = config_data.get('model_weights', {})
                    
                    # 合并配置权重和默认权重（配置权重优先）
                    merged_weights = default_model_weights.copy()
                    merged_weights.update(config_model_weights)
                    
                    error_handler.log_info(f"已从配置文件 {config_file} 加载模型权重", "AGICoordinator")
                    return merged_weights
                    
                except Exception as e:
                    error_handler.log_warning(f"加载模型权重配置文件失败: {e}, 使用默认权重", "AGICoordinator")
                    return default_model_weights
            else:
                # 配置文件不存在，创建默认配置文件
                try:
                    os.makedirs(os.path.dirname(config_file), exist_ok=True)
                    config_data = {
                        'model_weights': default_model_weights,
                        'description': 'AGI协调器配置，包含模型权重和系统参数',
                        'version': '1.0.0',
                        'last_updated': datetime.now().isoformat()
                    }
                    with open(config_file, 'w', encoding='utf-8') as f:
                        json.dump(config_data, f, indent=2, ensure_ascii=False)
                    
                    error_handler.log_info(f"已创建默认配置文件 {config_file}", "AGICoordinator")
                    return default_model_weights
                    
                except Exception as e:
                    error_handler.log_warning(f"创建配置文件失败: {e}, 使用默认权重", "AGICoordinator")
                    return default_model_weights
                    
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "加载模型权重配置失败")
            # 返回安全的默认权重
            return {
                 'language': 0.20, 'audio': 0.08, 'vision_image': 0.15,
                 'vision_video': 0.10, 'sensor': 0.05, 'manager': 0.25,
                 'planning': 0.08, 'reasoning': 0.12, 'learning': 0.07
             }
    
    def _calculate_dynamic_weights(self, performance_metrics):
        """基于历史性能数据动态计算权重
        
        Args:
            performance_metrics: 当前性能指标
            
        Returns:
            包含动态权重的字典
        """
        try:
            # 获取历史性能数据
            performance_history = self.long_term_memory.get('performance_history', [])
            
            if len(performance_history) < 10:
                # 数据不足，使用默认权重
                return {
                    'success_weight': 0.6,
                    'error_weight': 0.25,
                    'speed_weight': 0.15,
                    'base_score_factor': 0.5,
                    'learning_factor': 1.0,
                    'meta_cognition_factor': 1.0,
                    'knowledge_factor': 1.0
                }
            
            # 分析历史性能趋势
            recent_history = performance_history[-50:]  # 最近50条记录
            success_rates = [h.get('success_rate', 0.0) for h in recent_history if 'success_rate' in h]
            error_rates = [h.get('error_rate', 0.0) for h in recent_history if 'error_rate' in h]
            processing_times = [h.get('processing_time', 0.0) for h in recent_history if 'processing_time' in h]
            
            # 计算性能稳定性
            if success_rates:
                avg_success = np.mean(success_rates)
                std_success = np.std(success_rates) if len(success_rates) > 1 else 0.0
                success_stability = 1.0 - min(std_success / max(avg_success, 0.1), 1.0)
            else:
                avg_success = 0.5
                success_stability = 0.5
            
            if error_rates:
                avg_error = np.mean(error_rates)
                std_error = np.std(error_rates) if len(error_rates) > 1 else 0.0
                error_stability = 1.0 - min(std_error / max(avg_error, 0.1), 1.0)
            else:
                avg_error = 0.2
                error_stability = 0.5
            
            if processing_times:
                avg_time = np.mean(processing_times)
                std_time = np.std(processing_times) if len(processing_times) > 1 else 0.0
                time_stability = 1.0 - min(std_time / max(avg_time, 0.1), 1.0)
                
                # 基于处理时间动态调整速度权重
                # 处理时间越短，速度权重越高
                if avg_time > 0:
                    normalized_time = min(avg_time / 5.0, 1.0)  # 5秒为参考基准
                    speed_weight_factor = 1.0 - normalized_time
                else:
                    speed_weight_factor = 0.5
            else:
                time_stability = 0.5
                speed_weight_factor = 0.5
            
            # 动态权重计算
            # 成功稳定性越高，成功权重越高
            success_weight = 0.4 + (success_stability * 0.4)  # 范围: 0.4-0.8
            
            # 错误稳定性越高，错误权重越低（系统更可靠）
            error_weight = 0.3 - (error_stability * 0.15)  # 范围: 0.15-0.3
            
            # 速度权重基于处理时间和稳定性
            speed_weight = 0.1 + (speed_weight_factor * 0.2) + (time_stability * 0.1)  # 范围: 0.1-0.4
            
            # 归一化权重，确保总和为1.0
            total_weight = success_weight + error_weight + speed_weight
            success_weight /= total_weight
            error_weight /= total_weight
            speed_weight /= total_weight
            
            # 基础分数因子：基于系统整体稳定性
            overall_stability = (success_stability + error_stability + time_stability) / 3.0
            base_score_factor = 0.3 + (overall_stability * 0.4)  # 范围: 0.3-0.7
            
            # 其他因子的动态调整
            # 学习进度越快，学习因子越高
            learning_progress = self.self_learning.get_learning_progress().get('progress', 0.0)
            learning_factor = 0.5 + (learning_progress * 0.5)  # 范围: 0.5-1.0
            
            # 元认知信心越高，元认知因子越高
            meta_confidence = self.meta_cognition.get('confidence_level', 0.5)
            meta_cognition_factor = 0.5 + (meta_confidence * 0.5)  # 范围: 0.5-1.0
            
            # 知识覆盖率越高，知识因子越高
            knowledge_coverage = self._calculate_knowledge_coverage()
            knowledge_factor = 0.5 + (knowledge_coverage * 0.5)  # 范围: 0.5-1.0
            
            return {
                'success_weight': success_weight,
                'error_weight': error_weight,
                'speed_weight': speed_weight,
                'base_score_factor': base_score_factor,
                'learning_factor': learning_factor,
                'meta_cognition_factor': meta_cognition_factor,
                'knowledge_factor': knowledge_factor
            }
            
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "计算动态权重失败")
            # 返回默认权重
            return {
                'success_weight': 0.6,
                'error_weight': 0.25,
                'speed_weight': 0.15,
                'base_score_factor': 0.5,
                'learning_factor': 1.0,
                'meta_cognition_factor': 1.0,
                'knowledge_factor': 1.0
            }
    
    def _execute_with_recovery(self, operation_name, operation_func, max_retries=2, 
                              recovery_func=None, fallback_value=None, **kwargs):
        """执行操作并提供错误恢复机制
        
        Args:
            operation_name: 操作名称，用于日志记录
            operation_func: 要执行的操作函数
            max_retries: 最大重试次数（不包括初始尝试）
            recovery_func: 恢复函数，在重试失败后调用
            fallback_value: 完全失败时的后备返回值
            **kwargs: 传递给操作函数的参数
            
        Returns:
            操作结果或后备值
        """
        retries = 0
        last_exception = None
        
        while retries <= max_retries:
            try:
                if retries > 0:
                    error_handler.log_info(f"重试 {operation_name}，第{retries}次尝试", "AGICoordinator")
                    # 指数退避：每次重试等待更长时间
                    wait_time = min(1.0 * (2 ** (retries - 1)), 10.0)
                    time.sleep(wait_time)
                
                result = operation_func(**kwargs)
                if retries > 0:
                    error_handler.log_info(f"{operation_name} 在第{retries}次重试后成功", "AGICoordinator")
                return result
                
            except Exception as e:
                last_exception = e
                retries += 1
                error_handler.log_warning(
                    f"{operation_name} 失败，第{retries-1}次尝试: {str(e)[:100]}",
                    "AGICoordinator"
                )
        
        # 所有重试都失败了
        error_message = f"{operation_name} 在{max_retries}次重试后仍然失败"
        if last_exception:
            error_message += f": {last_exception}"
        
        error_handler.handle_error(Exception(error_message), "AGICoordinator", f"{operation_name}失败")
        
        # 尝试恢复函数
        if recovery_func:
            try:
                error_handler.log_info(f"尝试执行{operation_name}的恢复函数", "AGICoordinator")
                recovery_result = recovery_func()
                if recovery_result is not None:
                    return recovery_result
            except Exception as recovery_error:
                error_handler.handle_error(recovery_error, "AGICoordinator", f"{operation_name}恢复失败")
        
        # 返回后备值
        if fallback_value is not None:
            error_handler.log_warning(f"{operation_name} 使用后备值", "AGICoordinator")
            return fallback_value
        else:
            # 如果没有后备值，重新抛出最后一个异常
            raise last_exception if last_exception else Exception(f"{operation_name}失败")
    
    def _calculate_initial_agi_level(self):
        """基于可用模型和能力动态计算初始AGI水平"""
        try:
            # 获取所有模型状态
            models_status = self.model_registry.get_all_models_status()
            if not models_status:
                return 0.1  # 最低水平，如果没有模型
            
            # 动态计算模型类型总数，避免硬编码
            try:
                # 尝试通过模型注册表获取模型类型数量
                from core.model_registry import model_types
                total_model_types = len(model_types)
            except (ImportError, AttributeError):
                # 如果无法获取，使用实际加载的模型类型数量
                total_model_types = len(set([status.get('type', 'unknown') for status in models_status.values()]))
            
            # 避免除零错误
            if total_model_types == 0:
                total_model_types = 1
            
            # 计算模型覆盖率和能力分数
            model_coverage = len(models_status) / total_model_types
            capability_score = 0.0
            
            # 使用动态加载的模型权重配置
            base_model_weights = self.model_weights
            
            # 根据模型状态动态调整权重
            for model_id, status in models_status.items():
                # 获取模型类型
                model_type = status.get('type', model_id)
                weight = base_model_weights.get(model_type, 0.05)
                
                # 基于模型状态调整权重（如果模型已加载且可用）
                if status.get('loaded', False) and status.get('status') == 'active':
                    # 根据性能指标微调权重
                    success_rate = status.get('success_rate', 0.5)
                    weight_adjustment = 0.1 * success_rate
                    capability_score += weight * (1.0 + weight_adjustment)
                else:
                    # 模型未加载或不可用，减少权重
                    capability_score += weight * 0.3
            
            # 归一化能力分数
            if capability_score > 1.0:
                capability_score = 1.0
            
            # 综合评分：使用配置的权重
            model_coverage_weight = self.AGI_LEVEL_CONFIG.get('model_coverage_weight', 0.25)
            performance_weight = self.AGI_LEVEL_CONFIG.get('performance_weight', 0.35)
            base_score = 0.2  # 基础分
            
            agi_level = (model_coverage * model_coverage_weight) + (capability_score * performance_weight) + base_score
            return min(max(agi_level, 0.1), 1.0)  # 保持在0.1-1.0范围内
            
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "计算AGI水平失败")
            return 0.1  # 失败时返回最低水平
    
    def _start_autonomous_learning_loop(self):
        """启动自主学习的后台循环，带有资源监控和限制"""
        import threading
        import psutil
        
        def learning_loop():
            iteration_count = 0
            consecutive_failures = 0
            max_consecutive_failures = self.SYSTEM_CONFIG.get('max_consecutive_failures', 5)
            resource_check_interval = self.SYSTEM_CONFIG.get('resource_check_interval', 5.0)
            
            while self.system_state['status'] != 'shutdown':
                try:
                    # 检查系统资源使用情况
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory_percent = psutil.virtual_memory().percent
                    
                    # 自适应资源管理
                    max_cpu = self.SYSTEM_CONFIG.get('max_cpu_percent', 75)
                    max_memory = self.SYSTEM_CONFIG.get('max_memory_percent', 80)
                    enable_adaptive = self.SYSTEM_CONFIG.get('enable_adaptive_resource_management', True)
                    
                    # 使用自适应学习强度计算
                    learning_intensity = self._calculate_adaptive_learning_intensity(
                        cpu_percent, memory_percent, iteration_count
                    )
                    
                    # 如果资源使用过高，记录警告
                    if cpu_percent > max_cpu or memory_percent > max_memory:
                        error_handler.log_warning(
                            f"资源使用过高，已自适应调整学习强度 - CPU: {cpu_percent}%, Memory: {memory_percent}%, 学习强度: {learning_intensity:.2f}",
                            "AGICoordinator"
                        )
                    
                    # 根据资源使用情况动态调整休眠时间
                    base_sleep_time = 10.0
                    # 资源使用率阈值
                    cpu_critical = max_cpu * 0.9
                    memory_critical = max_memory * 0.9
                    cpu_low = max_cpu * 0.5
                    memory_low = max_memory * 0.5
                    
                    if cpu_percent > cpu_critical or memory_percent > memory_critical:
                        # 资源紧张，增加休眠时间
                        base_sleep_time *= 2.0
                    elif cpu_percent < cpu_low and memory_percent < memory_low:
                        # 资源充足，减少休眠时间（但保持最小限制）
                        base_sleep_time = max(5.0, base_sleep_time * 0.7)
                    
                    # 基于学习强度进一步调整休眠时间：学习强度越高，休眠时间越长（避免过载）
                    intensity_adjustment = 1.0 + (learning_intensity * 0.5)  # 学习强度0.1-1.0对应调整1.05-1.5
                    base_sleep_time *= intensity_adjustment
                    
                    # 运行自主学习，带有超时限制
                    try:
                        import threading as th
                        result = None
                        exception = None
                        
                        def _run_learning():
                            nonlocal result, exception
                            try:
                                result = self.self_learning.autonomous_learn(learning_intensity)
                            except Exception as e:
                                exception = e
                        
                        # 创建学习线程并设置超时
                        learning_thread = th.Thread(target=_run_learning, daemon=True)
                        learning_thread.start()
                        learning_thread.join(timeout=60)  # 60秒超时
                        
                        if exception:
                            raise exception
                        
                        if not result:
                            error_handler.log_warning("自主学习返回空结果", "AGICoordinator")
                    
                    except TimeoutError:
                        error_handler.log_error("自主学习超时", "AGICoordinator")
                        consecutive_failures += 1
                    except Exception as e:
                        error_handler.handle_error(e, "AGICoordinator", "自主学习执行异常")
                        consecutive_failures += 1
                    else:
                        consecutive_failures = 0  # 重置连续失败计数
                    
                    # 如果连续失败过多，延长休眠时间并尝试恢复
                    if consecutive_failures >= max_consecutive_failures:
                        error_handler.log_error(
                            f"自主学习连续失败{consecutive_failures}次，暂停学习循环",
                            "AGICoordinator"
                        )
                        # 指数退避策略
                        sleep_time = 60 * (2 ** min(consecutive_failures - max_consecutive_failures, 4))
                        time.sleep(sleep_time)
                        consecutive_failures = 0
                        continue
                    
                    iteration_count += 1
                    
                    # 定期记录学习进度和资源使用
                    if iteration_count % 10 == 0:
                        # 计算资源使用率指标
                        cpu_usage_ratio = cpu_percent / max_cpu if max_cpu > 0 else 0.0
                        memory_usage_ratio = memory_percent / max_memory if max_memory > 0 else 0.0
                        avg_resource_usage = (cpu_usage_ratio + memory_usage_ratio) / 2.0
                        
                        error_handler.log_info(
                            f"自主学习循环第{iteration_count}次迭代完成 - "
                            f"CPU: {cpu_percent}% ({cpu_usage_ratio:.2f}x), Memory: {memory_percent}% ({memory_usage_ratio:.2f}x), "
                            f"学习强度: {learning_intensity:.2f}, 平均资源使用率: {avg_resource_usage:.2f}x",
                            "AGICoordinator"
                        )
                    
                    # 动态调整休眠时间，避免过载
                    sleep_time = self._calculate_adaptive_sleep_time(
                        cpu_percent, memory_percent, iteration_count
                    )
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    error_handler.handle_error(e, "AGICoordinator", "自主学习循环异常")
                    time.sleep(30)  # 异常时等待更长时间
        
        # 启动后台线程
        learning_thread = threading.Thread(target=learning_loop, daemon=True)
        learning_thread.start()
        error_handler.log_info("自主学习循环已启动（带资源监控）", "AGICoordinator")
    
    def _calculate_adaptive_sleep_time(self, cpu_percent, memory_percent, iteration_count):
        """根据系统资源使用情况动态计算休眠时间"""
        base_sleep_time = 10.0
        
        # 根据CPU使用率调整
        if cpu_percent > 90:
            adjustment = 3.0
        elif cpu_percent > 70:
            adjustment = 1.5
        elif cpu_percent < 30:
            adjustment = 0.7  # 资源充足时可以更频繁运行
        else:
            adjustment = 1.0
        
        # 根据内存使用率调整
        if memory_percent > 90:
            adjustment *= 2.0
        elif memory_percent > 80:
            adjustment *= 1.5
        
        # 根据迭代次数调整（防止长时间运行导致的资源积累）
        if iteration_count > 100:
            adjustment *= 1.2
        
        sleep_time = base_sleep_time * adjustment
        return max(5.0, min(60.0, sleep_time))  # 限制在5-60秒之间
    
    def _calculate_adaptive_learning_intensity(self, cpu_percent, memory_percent, iteration_count):
        """计算自适应学习强度，基于系统资源和历史性能"""
        try:
            # 基础学习强度：基于AGI水平
            base_intensity = max(0.1, self.system_state['agi_level'])
            
            # 资源使用惩罚因子
            max_cpu = self.SYSTEM_CONFIG.get('max_cpu_percent', 75)
            max_memory = self.SYSTEM_CONFIG.get('max_memory_percent', 80)
            
            # 计算资源使用率
            cpu_usage_ratio = min(cpu_percent / max_cpu, 2.0)  # 上限200%
            memory_usage_ratio = min(memory_percent / max_memory, 2.0)
            
            # 资源惩罚因子：资源使用越高，惩罚越重
            cpu_penalty = max(0.1, 1.5 - cpu_usage_ratio)  # 范围: 0.1-1.5
            memory_penalty = max(0.1, 1.5 - memory_usage_ratio)
            
            # 综合惩罚因子（取最小值，因为最稀缺的资源决定上限）
            resource_penalty = min(cpu_penalty, memory_penalty)
            
            # 历史性能因子：基于最近的学习成功率
            performance_history = self.long_term_memory.get('performance_history', [])
            if len(performance_history) >= 10:
                recent_success = [h.get('success', False) for h in performance_history[-10:]]
                success_rate = sum(recent_success) / len(recent_success)
                # 成功率越高，性能因子越高（范围: 0.5-1.5）
                performance_factor = 0.5 + success_rate
            else:
                performance_factor = 1.0  # 默认中性因子
            
            # 系统稳定性因子：基于连续失败次数
            learning_loop_data = self.system_state.get('learning_loop_stats', {'consecutive_failures': 0})
            consecutive_failures = learning_loop_data.get('consecutive_failures', 0)
            stability_factor = max(0.3, 1.0 - (consecutive_failures * 0.2))  # 每失败一次减少20%
            
            # 时间衰减因子：避免长时间运行导致的性能下降
            time_factor = 1.0
            if iteration_count > 50:
                # 每50次迭代轻微衰减
                time_factor = max(0.7, 1.0 - ((iteration_count - 50) // 50) * 0.05)
            
            # 综合计算学习强度
            learning_intensity = base_intensity * resource_penalty * performance_factor * stability_factor * time_factor
            
            # 限制范围
            min_intensity = 0.1
            max_intensity = 1.0
            
            return min(max(learning_intensity, min_intensity), max_intensity)
            
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "计算自适应学习强度失败")
            return 0.5  # 默认强度
    
    
    """
    _update_system_state函数 - 中文函数描述
    _update_system_state Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _init_agi_enhancement_frameworks(self):
        """Initialize AGI capability enhancement frameworks"""
        try:
            from core.agi_capability_enhancement_framework import AGICapabilityEnhancementFramework
            from core.agi_performance_evaluation_framework import AGIPerformanceFramework
            from core.agi_self_learning_evolution import AGISelfLearningEvolutionFramework
            
            self.agi_enhancement = AGICapabilityEnhancementFramework()
            self.agi_performance = AGIPerformanceFramework()
            self.agi_evolution = AGISelfLearningEvolutionFramework()
            
            self.agi_frameworks_enabled = True
            error_handler.log_info("AGI enhancement frameworks initialized successfully", "AGICoordinator")
        except Exception as e:
            error_handler.log_warning(f"AGI enhancement frameworks initialization failed: {e}", "AGICoordinator")
            self.agi_enhancement = None
            self.agi_performance = None
            self.agi_evolution = None
            self.agi_frameworks_enabled = False
    
    def _update_system_state(self):
        """更新系统状态，包括动态计算AGI水平、自主性和泛化能力"""
        active_models = list(self.model_registry.get_all_models().keys())
        
        # 计算性能指标
        performance_metrics = self._calculate_performance_metrics()
        
        # 动态更新AGI水平
        current_agi_level = self._calculate_current_agi_level(performance_metrics)
        
        # 更新系统状态
        self.system_state.update({
            'status': 'running' if active_models else 'idle',
            'active_models': active_models,
            'last_updated': time.time(),
            'agi_level': current_agi_level,
            'autonomy_level': self._calculate_autonomy_level(performance_metrics),
            'generalization_score': self._calculate_generalization_score(performance_metrics),
            'learning_progress': self.self_learning.get_learning_progress(),
            'task_success_rate': performance_metrics.get('success_rate', 0.0),
            'knowledge_coverage': self._calculate_knowledge_coverage()
        })
    
    def _calculate_performance_metrics(self):
        """计算系统性能指标"""
        metrics = {
            'success_rate': 0.0,
            'avg_processing_time': 0.0,
            'throughput': 0.0,
            'error_rate': 0.0
        }
        
        # 从长期记忆中获取历史性能数据
        if self.long_term_memory.get('performance_history'):
            history = self.long_term_memory['performance_history']
            success_count = sum(1 for entry in history if entry.get('success', False))
            total_count = len(history)
            metrics['success_rate'] = success_count / total_count if total_count > 0 else 0.0
            
            processing_times = [entry.get('processing_time', 0) for entry in history]
            metrics['avg_processing_time'] = np.mean(processing_times) if processing_times else 0.0
            
            metrics['throughput'] = total_count / (time.time() - self.system_state['start_time']) if total_count > 0 else 0.0
            metrics['error_rate'] = 1 - metrics['success_rate']
        
        return metrics
    
    def _calculate_current_agi_level(self, performance_metrics):
        """基于当前性能动态计算AGI水平"""
        # 使用错误恢复机制包装计算
        def calculate_agi_level():
            # 获取模型状态
            models_status = self.model_registry.get_all_models_status()
            if not models_status:
                return 0.1
            
            # 模型覆盖率（基于已加载和活跃的模型数量）
            active_models = len([m for m in models_status.values() if m.get('status') == 'active'])
            
            # 安全地获取模型类型总数
            try:
                # 尝试通过模型注册表获取模型类型数量
                from core.model_registry import model_types
                total_model_types = len(model_types)
            except (ImportError, AttributeError):
                # 如果无法获取，使用实际加载的模型类型数量
                total_model_types = len(set([status.get('type', 'unknown') for status in models_status.values()]))
            
            # 避免除零错误
            if total_model_types == 0:
                total_model_types = 1
            
            model_coverage = min(active_models / max(total_model_types, 1), 1.0)
            
            # 计算动态权重
            dynamic_weights = self._calculate_dynamic_weights(performance_metrics)
            
            # 基础能力分数（基于模型覆盖率和动态因子）
            base_score_factor = dynamic_weights.get('base_score_factor', 0.5)
            base_score = model_coverage * base_score_factor
            
            # 使用动态权重计算性能因子
            success_weight = self.AGI_LEVEL_CONFIG.get('performance_weight', 0.35) * dynamic_weights.get('success_weight', 0.6)
            error_weight = self.AGI_LEVEL_CONFIG.get('performance_weight', 0.35) * dynamic_weights.get('error_weight', 0.25)
            speed_weight = self.AGI_LEVEL_CONFIG.get('performance_weight', 0.35) * dynamic_weights.get('speed_weight', 0.15)
            
            # 自适应处理时间范围：基于历史数据动态调整
            # 从性能历史中计算处理时间范围
            performance_history = self.long_term_memory.get('performance_history', [])
            processing_times = [h.get('processing_time', 0.0) for h in performance_history if 'processing_time' in h]
            
            if processing_times:
                min_processing_time = max(0.01, np.percentile(processing_times, 10))  # 10%分位数
                max_processing_time = min(30.0, np.percentile(processing_times, 90))  # 90%分位数，上限30秒
                # 确保有合理的范围
                if max_processing_time <= min_processing_time:
                    max_processing_time = min_processing_time + 1.0
            else:
                # 默认范围
                min_processing_time = 0.05
                max_processing_time = 15.0
            
            avg_processing_time = performance_metrics.get('avg_processing_time', 0.0)
            normalized_processing_time = min(max(avg_processing_time, min_processing_time), max_processing_time)
            speed_score = 1.0 - ((normalized_processing_time - min_processing_time) / (max_processing_time - min_processing_time))
            
            # 确保性能指标存在
            success_rate = performance_metrics.get('success_rate', 0.0)
            error_rate = performance_metrics.get('error_rate', 0.0)
            
            performance_factor = success_rate * success_weight + \
                                (1 - error_rate) * error_weight + \
                                speed_score * speed_weight
            
            # 学习进度因子（使用动态因子）
            learning_progress_dict = self.self_learning.get_learning_progress()
            learning_progress = learning_progress_dict.get('progress', 0.0)
            learning_weight = self.AGI_LEVEL_CONFIG.get('learning_weight', 0.2) * dynamic_weights.get('learning_factor', 1.0)
            learning_factor = learning_progress * learning_weight
            
            # 元认知信心因子（使用动态因子）
            meta_cognition_factor = self.meta_cognition.get('confidence_level', 0.5) * \
                                   self.AGI_LEVEL_CONFIG.get('meta_cognition_weight', 0.1) * \
                                   dynamic_weights.get('meta_cognition_factor', 1.0)
            
            # 知识覆盖率因子（使用动态因子）
            knowledge_coverage = self._calculate_knowledge_coverage()
            knowledge_weight = self.AGI_LEVEL_CONFIG.get('knowledge_weight', 0.1) * dynamic_weights.get('knowledge_factor', 1.0)
            knowledge_factor = knowledge_coverage * knowledge_weight
            
            # 综合计算新AGI水平（使用配置权重和动态权重）
            new_agi_level = base_score + performance_factor + learning_factor + meta_cognition_factor + knowledge_factor
            
            # 平滑过渡，避免剧烈变化
            smoothing_factor = self.AGI_LEVEL_CONFIG.get('smoothing_factor', 0.7)
            if 'agi_level' in self.system_state and self.system_state['agi_level'] > 0.1:
                smoothed_agi_level = smoothing_factor * self.system_state['agi_level'] + (1 - smoothing_factor) * new_agi_level
            else:
                smoothed_agi_level = new_agi_level  # 初始状态直接使用新值
            
            # 限制范围并返回
            min_agi = self.AGI_LEVEL_CONFIG.get('min_agi_level', 0.1)
            max_agi = self.AGI_LEVEL_CONFIG.get('max_agi_level', 1.0)
            return min(max(smoothed_agi_level, min_agi), max_agi)
        
        # 恢复函数：返回当前的AGI水平或默认值
        def recovery_func():
            return self.system_state.get('agi_level', 0.1)
        
        # 使用错误恢复机制执行计算
        return self._execute_with_recovery(
            operation_name="计算当前AGI水平",
            operation_func=calculate_agi_level,
            max_retries=1,
            recovery_func=recovery_func,
            fallback_value=self.system_state.get('agi_level', 0.1)
        )
    
    def _calculate_autonomy_level(self, performance_metrics):
        """计算自主性水平"""
        try:
            # 基于任务成功率和学习进度，使用动态权重
            success_rate = performance_metrics.get('success_rate', 0.0)
            learning_progress = self.system_state.get('learning_progress', 0.0)
            
            # 动态权重计算：基于历史性能
            performance_history = self.long_term_memory.get('performance_history', [])
            if len(performance_history) >= 20:
                recent_success = [h.get('success_rate', 0.0) for h in performance_history[-20:] if 'success_rate' in h]
                if recent_success:
                    avg_success = np.mean(recent_success)
                    # 成功率越高，成功权重越低（系统更可靠，可以更注重学习）
                    success_weight = 0.7 - (min(avg_success, 0.9) * 0.3)  # 范围: 0.4-0.7
                    learning_weight = 1.0 - success_weight
                else:
                    success_weight = 0.6
                    learning_weight = 0.4
            else:
                success_weight = 0.6
                learning_weight = 0.4
            
            autonomy = success_rate * success_weight + learning_progress * learning_weight
            
            return min(max(autonomy, 0.1), 1.0)
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "计算自主性水平失败")
            return 0.1
    
    def _calculate_generalization_score(self, performance_metrics):
        """计算泛化能力评分"""
        try:
            # 基于知识覆盖率和任务多样性，使用动态权重
            knowledge_coverage = self._calculate_knowledge_coverage()
            task_diversity = self._calculate_task_diversity()
            
            # 动态权重计算：基于系统成熟度
            # 成熟度越高，知识覆盖率权重越高（系统更依赖已有知识）
            system_maturity = self.system_state.get('agi_level', 0.1)
            
            # 成熟度在0-1之间，知识权重从0.3到0.7线性增加
            knowledge_weight = 0.3 + (system_maturity * 0.4)
            task_diversity_weight = 1.0 - knowledge_weight
            
            generalization = knowledge_coverage * knowledge_weight + task_diversity * task_diversity_weight
            return min(max(generalization, 0.1), 1.0)
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "计算泛化能力失败")
            return 0.1
    
    def _calculate_knowledge_coverage(self):
        """计算知识覆盖率"""
        try:
            # 基于长期记忆中的经验数量和模式
            experience_count = len(self.long_term_memory.get('experiences', []))
            pattern_count = len(self.long_term_memory.get('learned_patterns', []))
            
            # 动态计算最大经验数和模式数，基于系统配置
            max_experiences = self.SYSTEM_CONFIG.get('max_experiences', 2000)
            max_patterns = self.SYSTEM_CONFIG.get('max_patterns', 1000)
            
            # 计算经验覆盖率
            if max_experiences > 0:
                exp_coverage = min(experience_count / max_experiences, 1.0)
            else:
                exp_coverage = 0.0
                
            # 计算模式覆盖率
            if max_patterns > 0:
                pat_coverage = min(pattern_count / max_patterns, 1.0)
            else:
                pat_coverage = 0.0
            
            # 动态权重：经验越多，经验权重越高（系统更成熟）
            total_items = experience_count + pattern_count
            if total_items > 0:
                exp_weight = 0.4 + (experience_count / total_items * 0.4)  # 范围: 0.4-0.8
                pat_weight = 1.0 - exp_weight
            else:
                exp_weight = 0.6
                pat_weight = 0.4
            
            coverage = (exp_coverage * exp_weight + pat_coverage * pat_weight)
            
            return min(max(coverage, 0.0), 1.0)
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "计算知识覆盖率失败")
            return 0.0
    
    def _calculate_task_diversity(self):
        """计算任务多样性"""
        try:
            # 基于性能历史中的任务类型多样性
            history = self.long_term_memory.get('performance_history', [])
            if not history:
                return 0.1
            
            # 统计不同任务类型的数量
            task_types = set(entry.get('task_type', 'unknown') for entry in history)
            diversity = len(task_types) / 10  # 假设最多10种任务类型
            
            return min(max(diversity, 0.1), 1.0)
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "计算任务多样性失败")
            return 0.1
    
    
    """
    process_user_input函数 - 中文函数描述
    process_user_input Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def process_user_input(self, input_data, input_type='text', lang='zh'):
        """处理用户输入，支持文本、音频、图像等多种输入类型"""
        # 使用错误恢复机制处理用户输入
        def process_input_logic():
            start_time = time.time()
            error_handler.log_info(f"接收到用户输入，类型: {input_type}", "AGICoordinator")
            
            # 根据输入类型选择合适的模型进行处理
            if input_type == 'text':
                result = self._process_text_input(input_data, lang)
            elif input_type == 'audio':
                result = self._process_audio_input(input_data, lang)
            elif input_type == 'image':
                result = self._process_image_input(input_data)
            elif input_type == 'video':
                result = self._process_video_input(input_data)
            elif input_type == 'sensor':
                result = self._process_sensor_input(input_data)
            else:
                raise ValueError(f"不支持的输入类型: {input_type}")
            
            # 记录处理时间
            processing_time = time.time() - start_time
            self.system_state['performance_metrics'][input_type] = {
                'processing_time': processing_time,
                'timestamp': time.time()
            }
            
            # 更新自主学习系统的性能指标
            if input_type in ['text', 'audio', 'image']:
                model_id = input_type
                if input_type == 'image' and 'video' in result:
                    model_id = 'video'
                
                # 提取性能指标
                performance_metrics = {
                    'processing_time': processing_time,
                    'input_length': len(str(input_data)),
                    'output_length': len(str(result))
                }
                
                # 更新自主学习系统
                self.self_learning.update_performance(model_id, performance_metrics)
            
            error_handler.log_info(f"输入处理完成，耗时: {processing_time:.2f}秒", "AGICoordinator")
            return result
        
        # 恢复函数：尝试简化处理或返回基本响应
        def recovery_func():
            try:
                error_handler.log_warning("用户输入处理失败，尝试简化处理", "AGICoordinator")
                # 尝试使用文本作为后备，无论输入类型
                if input_type != 'text' and isinstance(input_data, (str, bytes)):
                    # 尝试将输入转换为文本描述
                    simplified_input = f"[{input_type}输入处理失败，原始数据长度: {len(str(input_data))}]"
                    return self._process_text_input(simplified_input, lang)
                else:
                    # 返回基本错误响应
                    return {"error": "用户输入处理失败，请重试"}
            except Exception as recovery_error:
                error_handler.handle_error(recovery_error, "AGICoordinator", "用户输入恢复失败")
                return {"error": "系统暂时无法处理您的输入"}
        
        # 使用错误恢复机制执行处理
        return self._execute_with_recovery(
            operation_name="处理用户输入",
            operation_func=process_input_logic,
            max_retries=1,
            recovery_func=recovery_func,
            fallback_value={"error": "系统暂时无法处理您的输入"}
        )
    
    
    """
    _process_text_input函数 - 中文函数描述
    _process_text_input Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _process_text_input(self, text, lang):
        """处理文本输入"""
        language_model = self.model_registry.get_model('language')
        if not language_model:
            raise RuntimeError("语言模型未加载")
        
        # 使用语言模型处理文本，构建正确的输入格式
        input_data = {
            "text": text,
            "context": {"language": lang}
        }
        return language_model.process(input_data)
    
    
    """
    _process_audio_input函数 - 中文函数描述
    _process_audio_input Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _process_audio_input(self, audio_data, lang):
        """Process audio input"""
        audio_model = self.model_registry.get_model('audio')
        if not audio_model:
            raise RuntimeError("Audio model not loaded")
        
        # Use audio model to process audio data
        return audio_model.process_audio(audio_data, lang)
    
    
    """
    _process_image_input函数 - 中文函数描述
    _process_image_input Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _process_image_input(self, image_data):
        """Process image input"""
        image_model = self.model_registry.get_model('vision_image')
        if not image_model:
            raise RuntimeError("Image model not loaded")
        
        # Use image model to process image data
        return image_model.process_image(image_data)
    
    
    """
    _process_video_input函数 - 中文函数描述
    _process_video_input Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _process_video_input(self, video_data):
        """处理视频输入"""
        video_model = self.model_registry.get_model('vision_video')
        if not video_model:
            raise RuntimeError("视频模型未加载")
        
        # 使用视频模型处理视频数据
        return video_model.process_video(video_data)
    
    
    """
    _process_sensor_input函数 - 中文函数描述
    _process_sensor_input Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _process_sensor_input(self, sensor_data):
        """处理传感器输入"""
        sensor_model = self.model_registry.get_model('sensor')
        if not sensor_model:
            raise RuntimeError("传感器模型未加载")
        
        # 使用传感器模型处理传感器数据
        return sensor_model.process_sensor_data(sensor_data)
    
    
    """
    coordinate_task函数 - 中文函数描述
    coordinate_task Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def coordinate_task(self, task_description, context=None):
        """协调多模型完成复杂任务，基于系统状态智能决策"""
        # 使用错误恢复机制协调任务
        def coordinate_task_logic():
            start_time = time.time()
            error_handler.log_info(f"开始协调任务: {task_description[:50]}...", "AGICoordinator")
            
            # 记录任务开始信息
            task_id = f"task_{int(time.time() * 1000)}_{(zlib.adler32(str(task_description).encode('utf-8')) & 0xffffffff) % 10000}"
            task_context = context or {}
            task_context['task_id'] = task_id
            
            # 获取管理模型
            manager_model = self.model_registry.get_model('manager')
            if not manager_model:
                raise RuntimeError("管理模型未加载")
            
            # 基于系统状态选择最优协调策略
            coordination_strategy = self._select_coordination_strategy(task_description)
            
            # 使用管理模型进行任务协调，传递策略信息
            result = manager_model.coordinate_task(task_description, {
                **task_context,
                'strategy': coordination_strategy,
                'agi_level': self.system_state['agi_level'],
                'available_models': self.system_state['active_models']
            })
            
            # 记录处理时间
            processing_time = time.time() - start_time
            
            # 记录性能历史
            self._record_performance_history({
                'task_id': task_id,
                'task_type': coordination_strategy.get('type', 'general'),
                'task_description': task_description,
                'processing_time': processing_time,
                'success': 'error' not in result,
                'timestamp': time.time(),
                'agi_level': self.system_state['agi_level'],
                'strategy_used': coordination_strategy
            })
            
            error_handler.log_info(f"任务协调完成，耗时: {processing_time:.2f}秒，成功率: {self.system_state['task_success_rate']:.2%}", "AGICoordinator")
            
            # 基于性能智能决定是否进行优化
            self._intelligent_optimization_decision(processing_time, result)
            
            # 如果任务失败，尝试使用备用策略
            if 'error' in result and coordination_strategy.get('fallback_strategy'):
                error_handler.log_warning(f"主策略失败，尝试备用策略", "AGICoordinator")
                fallback_result = self._execute_fallback_strategy(task_description, task_context, coordination_strategy)
                if 'error' not in fallback_result:
                    result = fallback_result
            
            return result
        
        # 恢复函数：尝试简化协调策略或使用基本模型直接处理
        def recovery_func():
            try:
                error_handler.log_warning("任务协调失败，尝试简化处理", "AGICoordinator")
                # 尝试使用最简单的策略：直接使用语言模型处理
                language_model = self.model_registry.get_model('language')
                if language_model:
                    simplified_input = f"简化任务: {task_description[:100]}"
                    simplified_result = language_model.process({"text": simplified_input})
                    return {"result": simplified_result, "note": "使用简化策略处理"}
                else:
                    # 返回基本响应
                    return {"error": "任务协调失败，请简化任务描述或重试"}
            except Exception as recovery_error:
                error_handler.handle_error(recovery_error, "AGICoordinator", "任务协调恢复失败")
                return {"error": "系统暂时无法协调此任务"}
        
        # 使用错误恢复机制执行协调
        return self._execute_with_recovery(
            operation_name="协调任务",
            operation_func=coordinate_task_logic,
            max_retries=1,
            recovery_func=recovery_func,
            fallback_value={"error": "系统暂时无法协调此任务"}
        )
    
    def _select_coordination_strategy(self, task_description):
        """基于任务描述和系统状态选择最优协调策略"""
        strategy = {
            'type': 'general',
            'priority': 'normal',
            'model_selection': 'auto',
            'fallback_strategy': None
        }
        
        # 基于AGI水平调整策略复杂度
        if self.system_state['agi_level'] > 0.7:
            strategy['complexity'] = 'high'
            strategy['parallel_processing'] = True
        elif self.system_state['agi_level'] > 0.4:
            strategy['complexity'] = 'medium'
            strategy['parallel_processing'] = False
        else:
            strategy['complexity'] = 'low'
            strategy['parallel_processing'] = False
        
        # 基于任务关键词识别任务类型
        task_lower = task_description.lower()
        if any(keyword in task_lower for keyword in ['学习', '训练', '教育']):
            strategy['type'] = 'learning'
            strategy['priority'] = 'medium'
            strategy['fallback_strategy'] = 'basic_learning'
        elif any(keyword in task_lower for keyword in ['分析', '处理', '计算']):
            strategy['type'] = 'analysis'
            strategy['priority'] = 'high'
            strategy['fallback_strategy'] = 'simple_analysis'
        elif any(keyword in task_lower for keyword in ['创作', '生成', '写作']):
            strategy['type'] = 'creative'
            strategy['priority'] = 'normal'
            strategy['fallback_strategy'] = 'template_based'
        
        # 基于系统负载调整优先级
        if len(self.system_state['performance_metrics']) > 5:
            strategy['priority'] = 'low'
        
        return strategy
    
    def _intelligent_optimization_decision(self, processing_time, result):
        """基于性能指标智能决定是否进行系统优化"""
        try:
            # 计算性能指标
            success = 'error' not in result
            recent_success_rate = self.system_state['task_success_rate']
            
            # 决定是否优化的条件
            needs_optimization = (
                processing_time > 2.0 or  # 处理时间过长
                not success or  # 任务失败
                recent_success_rate < 0.6 or  # 近期成功率低
                self.system_state['agi_level'] < 0.5  # AGI水平较低
            )
            
            if needs_optimization:
                # 基于问题的严重程度决定优化强度
                optimization_intensity = min(1.0, max(0.1, 
                    (2.0 - min(processing_time, 2.0)) / 2.0 * 0.5 +  # 时间因子
                    (0.0 if success else 0.3) +  # 失败因子
                    (0.6 - min(recent_success_rate, 0.6)) / 0.6 * 0.2  # 成功率因子
                ))
                
                error_handler.log_info(f"触发系统优化，强度: {optimization_intensity:.2f}", "AGICoordinator")
                self.self_learning.run_optimization(optimization_intensity)
                
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "优化决策失败")
    
    def _execute_fallback_strategy(self, task_description, context, original_strategy):
        """执行备用协调策略"""
        try:
            fallback_strategy = original_strategy.get('fallback_strategy')
            if not fallback_strategy:
                return {"error": "无备用策略可用"}
            
            error_handler.log_info(f"执行备用策略: {fallback_strategy}", "AGICoordinator")
            
            # 根据备用策略类型选择不同的处理方法
            if fallback_strategy == 'basic_learning':
                # 简化学习任务
                simplified_task = f"基础学习: {task_description}"
                return self.coordinate_task(simplified_task, {**context, 'fallback': True})
            
            elif fallback_strategy == 'simple_analysis':
                # 简化分析任务
                simplified_task = f"简单分析: {task_description}"
                return self.coordinate_task(simplified_task, {**context, 'fallback': True})
            
            elif fallback_strategy == 'template_based':
                # 使用模板基础的创作
                template_task = f"使用模板: {task_description}"
                return self.coordinate_task(template_task, {**context, 'fallback': True})
            
            return {"error": f"未知备用策略: {fallback_strategy}"}
            
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "执行备用策略失败")
            return {"error": str(e)}
    
    def _record_performance_history(self, performance_data):
        """记录任务性能历史"""
        try:
            # 添加到长期记忆
            self.long_term_memory['performance_history'].append(performance_data)
            
            # 保持历史记录数量合理（最近1000条）
            if len(self.long_term_memory['performance_history']) > 1000:
                self.long_term_memory['performance_history'] = self.long_term_memory['performance_history'][-1000:]
                
            # 定期保存记忆
            if len(self.long_term_memory['performance_history']) % 10 == 0:
                self.save_long_term_memory()
                
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "记录性能历史失败")
    
    
    """
    train_models函数 - 中文函数描述
    train_models Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def train_models(self, model_ids=None, parameters=None):
        """训练指定的模型"""
        try:
            start_time = time.time()
            error_handler.log_info(f"开始模型训练", "AGICoordinator")
            
            # 如果未指定模型ID，则训练所有模型
            if model_ids is None:
                model_ids = list(self.model_registry.get_all_models().keys())
            
            # 使用训练管理器启动训练任务
            job_id = self.training_manager.start_training(model_ids, parameters or {})
            
            return {"job_id": job_id, "message": "训练任务已启动"}
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "启动训练任务失败")
            return {"error": str(e)}
    
    
    """
    get_system_status函数 - 中文函数描述
    get_system_status Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def get_system_status(self):
        """获取系统状态"""
        self._update_system_state()
        # 获取所有模型的状态
        models_status = self.model_registry.get_all_models_status()
        # 获取自主学习系统状态
        learning_status = self.self_learning.get_learning_status()
        
        status = {
            **self.system_state,
            'models_status': models_status,
            'meta_cognition': self.meta_cognition,
            'memory_size': len(self.long_term_memory),
            'learning_status': learning_status
        }
        
        if self.agi_frameworks_enabled:
            status['agi_enhancement'] = {
                'enabled': True,
                'frameworks': ['PDAC Loop', 'Performance Evaluation', 'Self-Learning Evolution']
            }
        
        return status
    
    def process_with_agi_framework(self, input_data, perception_type='textual'):
        """Process input using AGI enhancement framework"""
        if not self.agi_frameworks_enabled or not self.agi_enhancement:
            return {'error': 'AGI enhancement framework not enabled'}
        
        try:
            from core.agi_capability_enhancement_framework import PerceptionType
            
            type_map = {
                'textual': PerceptionType.TEXTUAL,
                'visual': PerceptionType.VISUAL,
                'auditory': PerceptionType.AUDITORY,
                'sensor': PerceptionType.SENSOR,
                'internal': PerceptionType.INTERNAL
            }
            
            p_type = type_map.get(perception_type.lower(), PerceptionType.TEXTUAL)
            result = self.agi_enhancement.process_input(input_data, p_type)
            
            if self.agi_evolution:
                reward = result.get('assessment', {}).get('overall_score', 0.5)
                self.agi_evolution.process_experience(input_data, result, reward)
            
            return result
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "AGI framework processing failed")
            return {'error': str(e)}
    
    def get_agi_capability_assessment(self):
        """Get AGI capability assessment report"""
        if not self.agi_frameworks_enabled or not self.agi_performance:
            return {'error': 'AGI performance framework not enabled'}
        
        try:
            models = self.model_registry.get_all_models()
            evaluation = self.agi_performance.evaluate_all_models(models)
            return evaluation
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "AGI assessment failed")
            return {'error': str(e)}
    
    def run_agi_self_improvement(self, target_capability=None):
        """Run AGI self-improvement cycle"""
        if not self.agi_frameworks_enabled or not self.agi_evolution:
            return {'error': 'AGI evolution framework not enabled'}
        
        try:
            improvement = self.agi_evolution.run_self_improvement(target_capability)
            return improvement
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "AGI self-improvement failed")
            return {'error': str(e)}
    
    def get_agi_maturity_progress(self):
        """Get AGI maturity progress"""
        if not self.agi_frameworks_enabled:
            return {'error': 'AGI frameworks not enabled'}
        
        try:
            progress = {}
            
            if self.agi_performance:
                progress['performance_report'] = self.agi_performance.get_comprehensive_report()
            
            if self.agi_evolution:
                progress['evolution_status'] = self.agi_evolution.get_comprehensive_status()
            
            return progress
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "AGI maturity progress failed")
            return {'error': str(e)}
    
    
    """
    shutdown函数 - 中文函数描述
    shutdown Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def shutdown(self):
        """关闭系统"""
        error_handler.log_info("开始关闭系统", "AGICoordinator")
        # 保存长期记忆
        self.save_long_term_memory()
        
        # 卸载所有模型
        for model_id in list(self.model_registry.get_all_models().keys()):
            self.model_registry.unload_model(model_id)
        
        self.system_state['status'] = 'shutdown'
        self.system_state['shutdown_time'] = time.time()
        
        error_handler.log_info("系统已成功关闭", "AGICoordinator")
    
    # 新增：自主规划能力
    
    """
    autonomous_planning函数 - 中文函数描述
    autonomous_planning Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def autonomous_planning(self, goal, constraints=None):
        """根据目标自主规划任务序列"""
        try:
            manager_model = self.model_registry.get_model('manager')
            if not manager_model:
                raise RuntimeError("管理模型未加载")
                
            # 获取相关模型和能力信息
            available_models = self.get_system_status()['models_status']
            
            # 制定计划
            plan = manager_model.create_plan(goal, available_models, constraints)
            
            # 执行计划
            results = self.execute_plan(plan)
            
            # 评估结果并更新元认知
            self._evaluate_performance(goal, results)
            
            return results
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "自主规划执行失败")
            return {"error": str(e)}
    
    # 新增：执行计划
    
    """
    execute_plan函数 - 中文函数描述
    execute_plan Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def execute_plan(self, plan):
        """执行自主规划的任务序列"""
        results = {}
        for step in plan.get('steps', []):
            if step.get('action') == 'process_input':
                result = self.process_user_input(
                    step['input_data'],
                    step.get('input_type', 'text'),
                    step.get('lang', 'zh')
                )
            elif step.get('action') == 'coordinate_task':
                result = self.coordinate_task(step['task_description'], step.get('context', {}))
            elif step.get('action') == 'train_model':
                result = self.train_models([step['model_id']], step.get('parameters', {}))
            else:
                result = {"error": f"未知操作: {step.get('action')}"}
            
            results[step['id']] = result
            # 如果步骤失败且计划中指定了停止条件，则终止执行
            if 'error' in result and plan.get('stop_on_error', True):
                break
        
        return results
    
    # 新增：性能评估和自我改进
    
    """
    _evaluate_performance函数 - 中文函数描述
    _evaluate_performance Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _evaluate_performance(self, goal, results):
        """评估系统性能并更新元认知状态"""
        # 简单实现：统计成功率
        success_count = sum(1 for r in results.values() if 'error' not in r) if results else 0
        total_count = len(results) if results else 0
        
        if total_count > 0:
            success_rate = success_count / total_count
            
            # 更新元认知
            self.meta_cognition['confidence_level'] = min(max(success_rate, 0), 1)
            
            # 记录知识缺口
            if success_rate < 0.7:
                self.meta_cognition['knowledge_gaps'].append({
                    'goal': goal,
                    'success_rate': success_rate,
                    'timestamp': time.time()
                })
    
    # 新增：保存和加载长期记忆
    
    """
    save_long_term_memory函数 - 中文函数描述
    save_long_term_memory Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def save_long_term_memory(self):
        """保存长期记忆到持久化存储"""
        try:
            # 使用配置的长期记忆路径
            memory_path = self.SYSTEM_CONFIG.get('long_term_memory_path', 'data/long_term_memory.json')
            backup_path = self.SYSTEM_CONFIG.get('backup_memory_path', 'data/long_term_memory_backup.json')
            
            # 解析路径：如果是相对路径，转换为绝对路径
            if not os.path.isabs(memory_path):
                memory_file = os.path.join(os.path.dirname(__file__), '..', memory_path)
                memory_file = os.path.normpath(os.path.abspath(memory_file))
            else:
                memory_file = memory_path
            
            if not os.path.isabs(backup_path):
                backup_file = os.path.join(os.path.dirname(__file__), '..', backup_path)
                backup_file = os.path.normpath(os.path.abspath(backup_file))
            else:
                backup_file = backup_path
            
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(memory_file), exist_ok=True)
            os.makedirs(os.path.dirname(backup_file), exist_ok=True)
            
            # 先备份现有文件（如果存在）
            if os.path.exists(memory_file):
                import shutil
                try:
                    shutil.copy2(memory_file, backup_file)
                    error_handler.log_info(f"长期记忆已备份到: {backup_file}", "AGICoordinator")
                except Exception as backup_error:
                    error_handler.log_warning(f"长期记忆备份失败: {backup_error}", "AGICoordinator")
            
            # 保存新的长期记忆
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.long_term_memory, f, ensure_ascii=False, indent=2)
            
            error_handler.log_info(f"长期记忆已保存到: {memory_file}", "AGICoordinator")
            return True
            
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "保存长期记忆失败")
            return False
    
    
    """
    load_long_term_memory函数 - 中文函数描述
    load_long_term_memory Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def load_long_term_memory(self):
        """从持久化存储加载长期记忆"""
        try:
            # 使用配置的长期记忆路径
            memory_path = self.SYSTEM_CONFIG.get('long_term_memory_path', 'data/long_term_memory.json')
            backup_path = self.SYSTEM_CONFIG.get('backup_memory_path', 'data/long_term_memory_backup.json')
            
            # 解析路径：如果是相对路径，转换为绝对路径
            if not os.path.isabs(memory_path):
                memory_file = os.path.join(os.path.dirname(__file__), '..', memory_path)
                memory_file = os.path.normpath(os.path.abspath(memory_file))
            else:
                memory_file = memory_path
            
            if not os.path.isabs(backup_path):
                backup_file = os.path.join(os.path.dirname(__file__), '..', backup_path)
                backup_file = os.path.normpath(os.path.abspath(backup_file))
            else:
                backup_file = backup_path
            
            # 尝试从主文件加载
            if os.path.exists(memory_file):
                try:
                    with open(memory_file, 'r', encoding='utf-8') as f:
                        self.long_term_memory = json.load(f)
                    error_handler.log_info(f"长期记忆已从主文件加载: {memory_file}", "AGICoordinator")
                    return True
                except Exception as main_error:
                    error_handler.log_warning(f"从主文件加载长期记忆失败: {main_error}，尝试从备份文件加载", "AGICoordinator")
            
            # 如果主文件加载失败或不存在，尝试从备份文件加载
            if os.path.exists(backup_file):
                try:
                    with open(backup_file, 'r', encoding='utf-8') as f:
                        self.long_term_memory = json.load(f)
                    error_handler.log_info(f"长期记忆已从备份文件加载: {backup_file}", "AGICoordinator")
                    
                    # 尝试修复主文件
                    try:
                        os.makedirs(os.path.dirname(memory_file), exist_ok=True)
                        with open(memory_file, 'w', encoding='utf-8') as f:
                            json.dump(self.long_term_memory, f, ensure_ascii=False, indent=2)
                        error_handler.log_info(f"已从备份恢复主文件: {memory_file}", "AGICoordinator")
                    except Exception as restore_error:
                        error_handler.log_warning(f"恢复主文件失败: {restore_error}", "AGICoordinator")
                    
                    return True
                except Exception as backup_error:
                    error_handler.log_warning(f"从备份文件加载长期记忆失败: {backup_error}", "AGICoordinator")
            
            # 两个文件都不存在或都加载失败，使用空记忆
            error_handler.log_info("未找到长期记忆文件，使用空记忆", "AGICoordinator")
            self.long_term_memory = {
                'experiences': [],
                'learned_patterns': [],
                'optimization_history': [],
                'performance_history': []
            }
            return False
            
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "加载长期记忆失败")
            # 使用空记忆作为后备
            self.long_term_memory = {
                'experiences': [],
                'learned_patterns': [],
                'optimization_history': [],
                'performance_history': []
            }
            return False
    
    def resolve_conflicts(self, model_data: Dict[str, Any]) -> Any:
        """使用torch.distributions.Categorical进行概率共识解决冲突
        
        Args:
            model_data: 包含模型响应和置信度的字典，格式为:
                {
                    'model_id1': {'response': ..., 'confidence': 0.8},
                    'model_id2': {'response': ..., 'confidence': 0.6},
                    ...
                }
                
        Returns:
            共识结果，基于概率分布选择的响应
        """
        try:
            import torch
            from torch.distributions import Categorical
            
            if not model_data:
                return None
            
            # 确定设备：优先使用CUDA如果可用，否则使用CPU
            # 确保所有张量在同一设备上
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 提取模型ID和置信度
            model_ids = list(model_data.keys())
            confidences = []
            responses = []
            
            for model_id in model_ids:
                data = model_data[model_id]
                confidence = data.get('confidence', 0.5)
                response = data.get('response', None)
                
                # 确保置信度在合理范围内
                confidence = max(0.0, min(1.0, confidence))
                confidences.append(confidence)
                responses.append(response)
            
            # 创建概率分布（使用softmax将置信度转换为概率）
            # 确保张量在正确的设备上
            confidences_tensor = torch.tensor(confidences, dtype=torch.float32, device=device)
            probabilities = torch.softmax(confidences_tensor, dim=0)
            
            # 使用Categorical分布进行概率采样，确保分布在同一设备上
            distribution = Categorical(probs=probabilities)
            selected_idx = distribution.sample().item()
            
            # 记录选择过程
            selected_model = model_ids[selected_idx]
            selected_prob = probabilities[selected_idx].item()
            
            error_handler.log_info(
                f"概率共识: 选择模型 {selected_model} (概率: {selected_prob:.3f}, 设备: {device})",
                "AGICoordinator"
            )
            
            # 返回选择的响应
            selected_response = responses[selected_idx]
            
            # 如果响应是数值，可以返回加权平均作为备选
            if all(isinstance(r, (int, float)) for r in responses if r is not None):
                # 计算加权平均作为更稳定的结果
                weighted_sum = 0.0
                total_weight = 0.0
                for i, response in enumerate(responses):
                    if response is not None:
                        weight = probabilities[i].item()
                        weighted_sum += response * weight
                        total_weight += weight
                
                if total_weight > 0:
                    weighted_avg = weighted_sum / total_weight
                    return {
                        'selected_response': selected_response,
                        'selected_model': selected_model,
                        'selection_probability': selected_prob,
                        'weighted_average': weighted_avg,
                        'method': 'categorical_probabilistic_consensus',
                        'device': str(device)
                    }
            
            return {
                'selected_response': selected_response,
                'selected_model': selected_model,
                'selection_probability': selected_prob,
                'method': 'categorical_probabilistic_consensus'
            }
            
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "概率共识解决冲突失败")
            # 回退到简单的加权平均
            return self._fallback_weighted_consensus(model_data)
    
    def _fallback_weighted_consensus(self, model_data: Dict[str, Any]) -> Any:
        """回退加权共识算法（当概率共识失败时使用）"""
        try:
            if not model_data:
                return None
            
            model_ids = list(model_data.keys())
            weights = []
            responses = []
            
            for model_id in model_ids:
                data = model_data[model_id]
                confidence = data.get('confidence', 0.5)
                response = data.get('response', None)
                
                weights.append(confidence)
                responses.append(response)
            
            # 归一化权重
            total_weight = sum(weights)
            if total_weight == 0:
                # 避免除零错误
                normalized_weights = [1.0 / len(weights)] * len(weights)
            else:
                normalized_weights = [w / total_weight for w in weights]
            
            # 对于数值响应，返回加权平均
            if all(isinstance(r, (int, float)) for r in responses if r is not None):
                weighted_sum = 0.0
                for i, response in enumerate(responses):
                    if response is not None:
                        weighted_sum += response * normalized_weights[i]
                return weighted_sum
            
            # 对于其他类型，选择权重最高的响应
            max_weight_idx = max(range(len(normalized_weights)), key=lambda i: normalized_weights[i])
            return responses[max_weight_idx]
            
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "回退加权共识失败")
            # 最终回退：返回第一个响应
            first_key = next(iter(model_data.keys())) if model_data else None
            if first_key:
                return model_data[first_key].get('response', None)
            return None
