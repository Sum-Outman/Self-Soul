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

"""
自主学习和自我优化管理器：集成所有模块实现完整的自主学习和优化流程
Autonomous Learning and Self-Optimization Manager: Integrates all modules for complete autonomous learning and optimization
"""
import time
import threading
import json
import math
import statistics
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from core.error_handling import error_handler
# 延迟导入以避免循环依赖
# from core.model_registry import ModelRegistry  # 模型注册表
from core.joint_training_coordinator import JointTrainingCoordinator
from core.models.prediction import PredictionModel
from core.models.planning import PlanningModel
from core.models.language import AdvancedLanguageModel
from core.models.knowledge import KnowledgeModel
from collections import defaultdict

"""
AutonomousConfig类 - 中文类描述
AutonomousConfig Class - English class description
"""
@dataclass
class AutonomousConfig:
    """自主学习配置"""
    training_interval: int = 3600  # 训练间隔（秒）
    optimization_interval: int = 1800  # 优化间隔（秒）
    monitoring_interval: int = 300  # 监控间隔（秒）
    learning_interval: int = 300  # 学习循环间隔（秒）
    min_improvement_threshold: float = 0.1  # 最小改进阈值
    max_training_iterations: int = 10  # 最大训练迭代次数
    enable_continuous_learning: bool = True  # 启用持续学习


"""
AutonomousLearningManager类 - 中文类描述
AutonomousLearningManager Class - English class description
"""
class AutonomousLearningManager:
    """自主学习管理器，负责协调AGI系统的自主学习过程
    Autonomous Learning Manager, responsible for coordinating the autonomous learning process of the AGI system
    """
    
    def __init__(self, model_registry):
        """初始化自主学习管理器
        Initialize the autonomous learning manager
        
        Args:
            model_registry: 模型注册表实例
            model_registry: Model registry instance
        """
        self.model_registry = model_registry
        self.config = AutonomousConfig()
        self.running = False
        self.learning_thread = None
        self.performance_history = defaultdict(list)
        self.improvement_suggestions = []
        self.model_references = {}
        self.knowledge_model = None
        self.language_model = None
        
        # 添加学习进度和状态跟踪
        self.learning_progress = 0.0
        self.current_learning_status = 'idle'  # idle, running, paused, completed
        self.learning_domains = []
        self.learning_priority = 'balanced'
        self.learning_logs = []
        self.max_logs = 50
        
        # 优先级队列用于存储需要处理的改进建议
        self.priority_queue = []
        # 已实施的建议记录
        self.implemented_suggestions = []
        
        # 初始化模型引用
        self._initialize_model_references()
        # 设置学习基础设施
        self._setup_learning_infrastructure()
        
    def _initialize_model_references(self):
        """初始化对其他模型的引用
        Initialize references to other models
        """
        # 获取关键模型的引用
        self.knowledge_model = self.model_registry.get_model('knowledge')
        self.language_model = self.model_registry.get_model('language')
        
        # 获取所有模型引用
        all_models = self.model_registry.get_all_models()
        for model_id, model in all_models.items():
            self.model_references[model_id] = model
    
    def _setup_learning_infrastructure(self):
        """设置学习基础设施
        Set up learning infrastructure
        """
        # 创建必要的数据结构和组件
        # Create necessary data structures and components
        self.performance_evaluator = self._create_performance_evaluator()
        self.learning_strategy = self._create_learning_strategy()
        
        # 初始化模型状态跟踪
        # Initialize model status tracking
        self.model_status_tracking = defaultdict(lambda: {
            'last_trained': None,
            'performance_score': 0.0,
            'improvement_rate': 0.0,
            'training_priority': 0
        })
    
    def _create_performance_evaluator(self):
        """创建性能评估器
        Create performance evaluator
        """
        def evaluate_model(model, task):
            """评估模型性能
            Evaluate model performance
            """
            try:
                # 获取模型的实际性能指标
                if hasattr(model, 'get_performance_metrics'):
                    metrics = model.get_performance_metrics()
                    # 计算综合性能分数
                    if metrics:
                        score = 0
                        weight_sum = 0
                        
                        # 为不同指标分配权重
                        weights = {
                            'accuracy': 0.3,
                            'precision': 0.2,
                            'recall': 0.2,
                            'f1_score': 0.2,
                            'speed': 0.1
                        }
                        
                        for metric, value in metrics.items():
                            if metric in weights and isinstance(value, (int, float)):
                                score += value * weights[metric]
                                weight_sum += weights[metric]
                        
                        return score / weight_sum if weight_sum > 0 else 0.5
                
                # 如果没有具体性能指标，使用基础评估
                if hasattr(model, 'evaluate'):
                    result = model.evaluate(task)
                    if isinstance(result, (int, float)):
                        return min(max(result, 0), 1)  # 确保在0-1范围内
                
                # 默认返回中间值
                return 0.5
            except Exception as e:
                error_handler.handle_error(e, "AutonomousLearningManager", "性能评估失败")
                return 0.3  # 出错时返回较低分数
        
        # 返回性能评估器实例
        return {'evaluate': evaluate_model}
    
    def _create_learning_strategy(self):
        """创建学习策略
        Create learning strategy
        """
        def select_next_task(model, performance):
            """根据模型性能和类型选择下一个学习任务
            Select next learning task based on model performance and type
            """
            try:
                model_id = model.model_id if hasattr(model, 'model_id') else 'unknown'
                
                # 根据模型类型和性能选择不同的学习任务
                # 性能较低的模型应该优先进行基础训练
                if performance < 0.4:
                    if model_id == 'knowledge':
                        return 'foundational_knowledge_acquisition'
                    elif model_id in ['language', 'vision_image', 'vision_video']:
                        return 'basic_skill_training'
                    else:
                        return 'fundamental_concept_learning'
                elif performance < 0.7:
                    # 中等性能模型进行知识增强和实践训练
                    if model_id == 'knowledge':
                        return 'knowledge_integration'
                    elif model_id == 'language':
                        return 'contextual_understanding_training'
                    elif model_id in ['vision_image', 'vision_video']:
                        return 'complex_pattern_recognition'
                    elif model_id in ['planning', 'prediction']:
                        return 'scenario_based_training'
                    else:
                        return 'task_specific_enhancement'
                else:
                    # 高性能模型进行高级学习和创新
                    if model_id == 'knowledge':
                        return 'knowledge_creation'
                    elif model_id == 'autonomous':
                        return 'meta_learning_optimization'
                    elif model_id == 'programming':
                        return 'advanced_algorithm_exploration'
                    else:
                        return 'cross_domain_knowledge_transfer'
            except Exception as e:
                error_handler.handle_error(e, "AutonomousLearningManager", "学习任务选择失败")
                return 'knowledge_enhancement'  # 默认返回知识增强任务
        
        # 返回学习策略实例
        return {'select_next_task': select_next_task}
    
    def start_autonomous_learning_cycle(self, domains=None, priority='balanced'):
        """启动自主学习循环
        Start autonomous learning cycle
        
        Args:
            domains: 要关注的知识领域列表
            priority: 学习优先级 (balanced, exploration, exploitation)
        
        Returns:
            bool: 是否成功启动
        """
        if self.running:
            error_handler.log_info("自主学习循环已在运行中", "AutonomousLearningManager")
            return False
        
        # 设置学习参数
        self.learning_domains = domains or []
        self.learning_priority = priority
        
        # 重置进度和状态
        self.learning_progress = 0.0
        self.current_learning_status = 'running'
        self.learning_logs = []
        
        # 添加启动日志
        self._add_learning_log(f"开始自主学习，领域: {self.learning_domains}, 优先级: {self.learning_priority}")
        
        self.running = True
        self.learning_thread = threading.Thread(target=self._learning_cycle)
        self.learning_thread.daemon = True
        self.learning_thread.start()
        
        error_handler.log_info("自主学习循环已启动", "AutonomousLearningManager")
        return True
        
    def stop_autonomous_learning_cycle(self):
        """停止自主学习循环
        Stop autonomous learning cycle
        """
        self.running = False
        self.current_learning_status = 'idle'
        self._add_learning_log("自主学习已停止")
        
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=5.0)
            
        error_handler.log_info("自主学习循环已停止", "AutonomousLearningManager")
        return True
        
    def _learning_cycle(self):
        """自主学习循环的内部实现
        Internal implementation of autonomous learning cycle
        """
        try:
            total_cycles = self.config.max_training_iterations
            current_cycle = 0
            
            while self.running and current_cycle < total_cycles:
                try:
                    current_cycle += 1
                    
                    # 更新进度
                    self.learning_progress = min((current_cycle / total_cycles) * 100, 100)
                    
                    # 检查每个模型的状态和性能
                    # Check status and performance of each model
                    self._evaluate_all_models()
                    
                    # 选择最需要改进的模型和任务
                    # Select the model and task that needs improvement the most
                    model_id, task = self._select_next_improvement_target()
                    
                    if model_id and task:
                        # 执行改进任务
                        # Execute improvement task
                        self._execute_improvement_task(model_id, task)
                        
                        # 添加任务执行日志
                        self._add_learning_log(f"已完成 {task} 任务对模型 {model_id}")
                    
                    # 生成学习报告
                    # Generate learning report
                    self._generate_learning_report()
                    
                    # 等待下一个学习周期
                    # Wait for the next learning cycle
                    wait_time = self.config.learning_interval
                    for i in range(wait_time // 1000):
                        if not self.running:
                            break
                        time.sleep(1)
                        # 每1秒小幅度更新进度
                        self.learning_progress = min(self.learning_progress + (100/(total_cycles*wait_time)), 100)
                except Exception as e:
                    error_handler.handle_error(e, "AutonomousLearningManager", "自主学习循环出错")
                    self._add_learning_log(f"学习循环出错: {str(e)}")
                    # 继续运行，即使发生错误
                    # Continue running even if an error occurs
                    time.sleep(5)
            
            # 学习完成
            if self.running:
                self.learning_progress = 100
                self.current_learning_status = 'completed'
                self._add_learning_log("自主学习完成")
                self.running = False
        except Exception as e:
            error_handler.handle_error(e, "AutonomousLearningManager", "自主学习循环严重错误")
            self._add_learning_log(f"学习循环严重错误: {str(e)}")
            self.learning_progress = 0
            self.current_learning_status = 'idle'
            self.running = False
        
    def _evaluate_all_models(self):
        """评估所有模型的性能
        Evaluate the performance of all models
        """
        for model_id, model in self.model_references.items():
            try:
                # 评估模型性能
                # Evaluate model performance
                performance = self._evaluate_model(model_id)
                
                # 更新性能历史
                # Update performance history
                self.performance_history[model_id].append({
                    'timestamp': datetime.now(),
                    'score': performance
                })
                
                # 计算改进率
                # Calculate improvement rate
                improvement_rate = self._calculate_improvement_rate(model_id)
                
                # 更新模型状态跟踪
                # Update model status tracking
                self.model_status_tracking[model_id] = {
                    'last_trained': datetime.now(),
                    'performance_score': performance,
                    'improvement_rate': improvement_rate,
                    'training_priority': self._calculate_training_priority(model_id)
                }
            except Exception as e:
                error_handler.log_warning(f"评估模型 {model_id} 性能时出错: {str(e)}", "AutonomousLearningManager")
        
    def _evaluate_model(self, model_id):
        """评估单个模型的性能
        Evaluate the performance of a single model
        
        Args:
            model_id: 模型ID
            model_id: Model ID
            
        Returns:
            float: 性能分数
            float: Performance score
        """
        try:
            model = self.model_references.get(model_id)
            if not model:
                return 0.3  # 无法找到模型，返回低分
                
            # 使用实际的性能评估方法
            if hasattr(model, 'get_performance_metrics'):
                metrics = model.get_performance_metrics()
                if metrics:
                    # 计算综合性能分数
                    score = 0
                    weight_sum = 0
                    weights = {
                        'accuracy': 0.3,
                        'precision': 0.2,
                        'recall': 0.2,
                        'f1_score': 0.2,
                        'speed': 0.1
                    }
                    
                    for metric, value in metrics.items():
                        if metric in weights and isinstance(value, (int, float)):
                            score += value * weights[metric]
                            weight_sum += weights[metric]
                    
                    return score / weight_sum if weight_sum > 0 else 0.5
            elif hasattr(model, 'evaluate'):
                # 如果模型有evaluate方法，调用它
                result = model.evaluate()
                if isinstance(result, (int, float)):
                    return min(max(result, 0), 1)  # 确保在0-1范围内
            
            # 对于知识模型，我们可以特殊处理
            if model_id == 'knowledge' and hasattr(model, 'get_knowledge_coverage'):
                coverage = model.get_knowledge_coverage()
                if isinstance(coverage, (int, float)):
                    return min(max(coverage/100, 0), 1)
            
            # 如果没有具体性能指标，返回基础分
            return 0.5
        except Exception as e:
            error_handler.log_error(f"评估模型 {model_id} 性能时出错: {str(e)}", "AutonomousLearningManager")
            return 0.3
        
    def _calculate_improvement_rate(self, model_id):
        """计算模型性能的改进率
        Calculate the improvement rate of model performance
        
        Args:
            model_id: 模型ID
            model_id: Model ID
            
        Returns:
            float: 改进率
            float: Improvement rate
        """
        history = self.performance_history.get(model_id, [])
        if len(history) < 2:
            return 0.0
        
        # 计算最近几次性能的平均改进率
        # Calculate the average improvement rate of recent performances
        recent_history = history[-5:]  # 获取最近5次评估
        if len(recent_history) < 2:
            return 0.0
        
        # 计算改进率
        # Calculate improvement rate
        improvements = []
        for i in range(1, len(recent_history)):
            prev_score = recent_history[i-1]['score']
            curr_score = recent_history[i]['score']
            if prev_score > 0:
                improvement = (curr_score - prev_score) / prev_score
                improvements.append(improvement)
        
        return sum(improvements) / len(improvements) if improvements else 0.0
        
    def _calculate_training_priority(self, model_id):
        """计算模型的训练优先级
        Calculate the training priority of a model
        
        Args:
            model_id: 模型ID
            model_id: Model ID
            
        Returns:
            float: 优先级分数
            float: Priority score
        """
        status = self.model_status_tracking.get(model_id, {})
        performance = status.get('performance_score', 0.0)
        improvement_rate = status.get('improvement_rate', 0.0)
        
        # 性能越低，优先级越高
        # The lower the performance, the higher the priority
        # 改进率越低，优先级越高
        # The lower the improvement rate, the higher the priority
        priority = (1.0 - performance) * 0.7 + (1.0 - max(improvement_rate, 0.0)) * 0.3
        
        return priority
        
    def _select_next_improvement_target(self):
        """选择下一个需要改进的目标模型和任务
        Select the next improvement target model and task
        
        Returns:
            tuple: (模型ID, 任务类型) 或 (None, None)
            tuple: (Model ID, Task type) or (None, None)
        """
        # 按优先级排序模型
        # Sort models by priority
        prioritized_models = sorted(
            self.model_status_tracking.items(),
            key=lambda x: x[1]['training_priority'],
            reverse=True
        )
        
        # 选择优先级最高的模型
        # Select the model with the highest priority
        if prioritized_models:
            model_id, _ = prioritized_models[0]
            # 选择适合该模型的任务
            # Select a task suitable for this model
            task = self._select_task_for_model(model_id)
            return model_id, task
        
        return None, None
        
    def _select_task_for_model(self, model_id):
        """为指定模型选择合适的任务
        Select a suitable task for the specified model
        
        Args:
            model_id: 模型ID
            model_id: Model ID
            
        Returns:
            str: 任务类型
            str: Task type
        """
        # 根据模型类型选择任务
        # Select task based on model type
        # 这里应该有更复杂的逻辑来选择最适合的任务
        # There should be more complex logic here to select the most suitable task
        task_map = {
            'language': 'language_enhancement',
            'knowledge': 'knowledge_enhancement',
            'audio': 'audio_enhancement',
            'vision_image': 'vision_enhancement',
            'vision_video': 'vision_enhancement',
            'spatial': 'spatial_enhancement',
            'stereo_spatial': 'spatial_enhancement',
            'sensor': 'sensor_enhancement',
            'computer': 'computer_enhancement',
            'motion': 'motion_enhancement',
            'programming': 'programming_enhancement'
        }
        
        return task_map.get(model_id, 'general_enhancement')
        
    def _execute_improvement_task(self, model_id, task):
        """执行模型改进任务
        Execute model improvement task
        
        Args:
            model_id: 模型ID
            model_id: Model ID
            task: 任务类型
            task: Task type
        """
        try:
            error_handler.log_info(f"开始执行改进任务: {task} 对模型: {model_id}", "AutonomousLearningManager")
            
            # 获取模型引用
            # Get model reference
            model = self.model_references.get(model_id)
            if not model:
                error_handler.log_warning(f"无法找到模型: {model_id}", "AutonomousLearningManager")
                return
            
            # 执行具体任务
            # Execute specific task
            if hasattr(model, 'improve'):
                # 如果模型有improve方法，调用它
                # If the model has an improve method, call it
                model.improve(task, self.knowledge_model)
            else:
                # 否则使用通用改进方法
                # Otherwise, use the general improvement method
                self._general_improvement(model_id, task)
            
            error_handler.log_info(f"完成改进任务: {task} 对模型: {model_id}", "AutonomousLearningManager")
        except Exception as e:
            error_handler.handle_error(e, "AutonomousLearningManager", f"执行改进任务时出错: {task} 对模型: {model_id}")
        
    def _general_improvement(self, model_id, task):
        """通用模型改进方法
        General model improvement method
        
        Args:
            model_id: 模型ID
            model_id: Model ID
            task: 任务类型
            task: Task type
        """
        try:
            error_handler.log_info(f"对模型 {model_id} 应用通用改进: {task}", "AutonomousLearningManager")
            
            model = self.model_references.get(model_id)
            if not model:
                return
            
            # 实际的改进逻辑
            # 1. 从知识库获取相关知识
            relevant_knowledge = []
            if self.knowledge_model and hasattr(self.knowledge_model, 'search_knowledge'):
                # 根据模型ID和任务类型搜索相关知识
                search_query = f"{model_id} {task}"
                relevant_knowledge = self.knowledge_model.search_knowledge(search_query, limit=5)
                
            # 2. 根据不同的任务类型执行不同的改进
            if task.endswith('enhancement'):
                # 基础增强任务
                if hasattr(model, 'update_parameters'):
                    # 为模型提供相关知识作为更新参数
                    model.update_parameters(relevant_knowledge)
            elif task == 'foundational_knowledge_acquisition':
                # 基础知识获取
                if hasattr(model, 'acquire_knowledge'):
                    model.acquire_knowledge(relevant_knowledge)
            elif task == 'cross_domain_knowledge_transfer':
                # 跨领域知识迁移
                if hasattr(model, 'transfer_knowledge'):
                    model.transfer_knowledge(relevant_knowledge)
            
            # 3. 对于知识模型的特殊处理
            if model_id == 'knowledge' and hasattr(model, 'consolidate_knowledge'):
                model.consolidate_knowledge()
                
        except Exception as e:
            error_handler.handle_error(e, "AutonomousLearningManager", f"执行通用改进任务时出错: {task} 对模型: {model_id}")
        
    def _generate_learning_report(self):
        """生成学习报告
        Generate learning report
        """
        # 创建学习报告
        # Create learning report
        report = {
            'timestamp': datetime.now(),
            'models_evaluated': len(self.model_references),
            'improvement_suggestions': self.improvement_suggestions,
            'model_performances': {}
        }
        
        # 添加各模型性能
        # Add performances of each model
        for model_id, status in self.model_status_tracking.items():
            report['model_performances'][model_id] = {
                'performance_score': status.get('performance_score', 0.0),
                'improvement_rate': status.get('improvement_rate', 0.0),
                'training_priority': status.get('training_priority', 0)
            }
        
        # 这里可以添加报告存储逻辑
        # Report storage logic can be added here
        
        # 清空改进建议，为下一轮做准备
        # Clear improvement suggestions for next round
        self.improvement_suggestions = []
        
    def evaluate_suggestion_impact(self, suggestion):
        """评估建议的影响程度
        Evaluate the impact of a suggestion
        
        Args:
            suggestion: 改进建议字典，包含type, target, content等字段
            suggestion: Improvement suggestion dictionary with type, target, content fields
            
        Returns:
            float: 影响评分（0-1之间）
            float: Impact score (between 0-1)
        """
        try:
            impact = 0.0
            
            # 1. 根据建议类型评估影响
            suggestion_type = suggestion.get('type', 'general')
            type_weights = {
                'core': 0.8,
                'model': 0.7,
                'optimization': 0.6,
                'knowledge': 0.5,
                'interface': 0.4,
                'general': 0.3
            }
            impact += type_weights.get(suggestion_type, 0.3) * 0.4
            
            # 2. 根据目标范围评估影响
            target = suggestion.get('target', 'system')
            if target == 'system':
                impact += 0.3  # 系统级影响
            elif target in self.model_references:
                impact += 0.2  # 模型级影响
            else:
                impact += 0.1  # 组件级影响
            
            # 3. 根据预期收益评估影响
            expected_benefit = suggestion.get('expected_benefit', 0.5)
            impact += expected_benefit * 0.2
            
            # 4. 根据实施难度反向评估影响（难度低，影响相对更大）
            implementation_difficulty = suggestion.get('implementation_difficulty', 0.5)
            impact += (1 - implementation_difficulty) * 0.1
            
            # 确保影响分数在0-1范围内
            return min(max(impact, 0), 1)
        except Exception as e:
            error_handler.log_error(f"评估建议影响时出错: {str(e)}", "AutonomousLearningManager")
            return 0.5  # 默认中间值
    
    def process_improvement(self, suggestion):
        """处理改进建议，包括影响评估、优先级排序和调度
        Process improvement suggestion, including impact evaluation, priority ranking and scheduling
        
        Args:
            suggestion: 改进建议
            suggestion: Improvement suggestion
        """
        try:
            # 评估建议影响
            impact = self.evaluate_suggestion_impact(suggestion)
            suggestion['impact_score'] = impact
            
            # 添加时间戳
            suggestion['timestamp'] = datetime.now()
            
            # 如果是高影响建议（>0.6），加入优先级队列
            if impact > 0.6:
                self.priority_queue.append(suggestion)
                # 按影响分数排序优先级队列
                self.priority_queue.sort(key=lambda x: x['impact_score'], reverse=True)
                self._add_learning_log(f"高影响建议已加入优先级队列: {suggestion.get('content', '未命名建议')} (影响分数: {impact:.2f})")
                
                # 立即调度实施
                self.schedule_implementation(suggestion)
            else:
                # 低影响建议加入普通改进列表
                self.improvement_suggestions.append(suggestion)
                self._add_learning_log(f"普通影响建议已记录: {suggestion.get('content', '未命名建议')} (影响分数: {impact:.2f})")
        except Exception as e:
            error_handler.handle_error(e, "AutonomousLearningManager", "处理改进建议时出错")
    
    def schedule_implementation(self, suggestion):
        """调度建议的实施
        Schedule the implementation of a suggestion
        
        Args:
            suggestion: 改进建议
            suggestion: Improvement suggestion
        """
        try:
            self._add_learning_log(f"开始实施建议: {suggestion.get('content', '未命名建议')}")
            
            # 获取建议目标
            target = suggestion.get('target', 'system')
            suggestion_type = suggestion.get('type', 'general')
            content = suggestion.get('content', '')
            
            # 根据建议类型和目标执行不同的实施策略
            if target == 'system':
                # 系统级建议处理
                self._implement_system_suggestion(suggestion)
            elif target in self.model_references:
                # 模型级建议处理
                self._implement_model_suggestion(target, suggestion)
            else:
                # 通用建议处理
                self._implement_general_suggestion(suggestion)
            
            # 记录已实施的建议
            suggestion['implemented'] = True
            suggestion['implementation_time'] = datetime.now()
            self.implemented_suggestions.append(suggestion)
            
            # 如果建议在优先级队列中，移除它
            if suggestion in self.priority_queue:
                self.priority_queue.remove(suggestion)
            
            self._add_learning_log(f"建议实施完成: {suggestion.get('content', '未命名建议')}")
        except Exception as e:
            error_handler.handle_error(e, "AutonomousLearningManager", "调度实施建议时出错")
    
    def _implement_system_suggestion(self, suggestion):
        """实施系统级建议
        Implement system-level suggestion
        
        Args:
            suggestion: 系统级建议
        """
        try:
            content = suggestion.get('content', '')
            if 'training_interval' in content or '优化间隔' in content:
                # 调整训练间隔
                if hasattr(self.config, 'training_interval'):
                    self.config.training_interval = suggestion.get('parameters', {}).get('training_interval', 3600)
            elif 'learning_interval' in content or '学习间隔' in content:
                # 调整学习间隔
                if hasattr(self.config, 'learning_interval'):
                    self.config.learning_interval = suggestion.get('parameters', {}).get('learning_interval', 300)
            # 其他系统级建议处理逻辑...
        except Exception as e:
            error_handler.log_error(f"实施系统级建议时出错: {str(e)}", "AutonomousLearningManager")
    
    def _implement_model_suggestion(self, model_id, suggestion):
        """实施模型级建议
        Implement model-level suggestion
        
        Args:
            model_id: 模型ID
            suggestion: 模型级建议
        """
        try:
            model = self.model_references.get(model_id)
            if not model:
                error_handler.log_warning(f"无法找到模型: {model_id}", "AutonomousLearningManager")
                return
            
            # 调用模型的improve方法
            if hasattr(model, 'improve'):
                model.improve(suggestion.get('content', ''), self.knowledge_model)
            else:
                error_handler.log_warning(f"模型 {model_id} 没有improve方法", "AutonomousLearningManager")
        except Exception as e:
            error_handler.log_error(f"实施模型级建议时出错: {str(e)}", "AutonomousLearningManager")
    
    def _implement_general_suggestion(self, suggestion):
        """实施通用建议
        Implement general suggestion
        
        Args:
            suggestion: 通用建议
        """
        try:
            content = suggestion.get('content', '')
            error_handler.log_info(f"实施通用建议: {content}", "AutonomousLearningManager")
            # 通用建议处理逻辑...
        except Exception as e:
            error_handler.log_error(f"实施通用建议时出错: {str(e)}", "AutonomousLearningManager")
    
    def suggest_improvement(self, suggestion):
        """添加改进建议
        Add improvement suggestion
        
        Args:
            suggestion: 改进建议
            suggestion: Improvement suggestion
        """
        # 直接调用process_improvement处理建议
        self.process_improvement(suggestion)
        
    def get_performance_metrics(self):
        """获取性能指标
        Get performance metrics
        
        Returns:
            dict: 性能指标字典
            dict: Performance metrics dictionary
        """
        metrics = {
            'overall_performance': self._calculate_overall_performance(),
            'model_performances': {}
        }
        
        # 添加各模型性能
        # Add performances of each model
        for model_id, status in self.model_status_tracking.items():
            metrics['model_performances'][model_id] = status
        
        return metrics
        
    def _calculate_overall_performance(self):
        """计算系统整体性能
        Calculate overall system performance
        
        Returns:
            float: 整体性能分数
            float: Overall performance score
        """
        # 计算所有模型的平均性能
        # Calculate the average performance of all models
        performances = [status.get('performance_score', 0.0) for status in self.model_status_tracking.values()]
        if not performances:
            return 0.0
        
        return sum(performances) / len(performances)
        
    def update_config(self, config):
        """更新配置
        Update configuration
        
        Args:
            config: 新的配置字典
            config: New configuration dictionary
        """
        # 更新配置
        # Update configuration
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        error_handler.log_info(f"更新自主学习配置: {config}", "AutonomousLearningManager")
        
    def get_status(self):
        """获取自主学习管理器的状态
        Get the status of the autonomous learning manager
        
        Returns:
            dict: 状态字典
            dict: Status dictionary
        """
        return {
            'running': self.running,
            'config': self.config.__dict__,
            'models_managed': len(self.model_references),
            'last_evaluation_time': self._get_last_evaluation_time()
        }
        
    def _get_last_evaluation_time(self):
        """获取最后一次评估时间
        Get the last evaluation time
        
        Returns:
            datetime or None: 最后一次评估时间或None
            datetime or None: Last evaluation time or None
        """
        # 找出最近的评估时间
        # Find the most recent evaluation time
        latest_time = None
        for model_id, history in self.performance_history.items():
            if history:
                model_latest = max(history, key=lambda x: x['timestamp'])
                if not latest_time or model_latest['timestamp'] > latest_time:
                    latest_time = model_latest['timestamp']
        
        return latest_time
        
    def get_improvement_suggestions(self):
        """获取改进建议列表
        Get list of improvement suggestions
        
        Returns:
            list: 改进建议列表
            list: List of improvement suggestions
        """
        return self.improvement_suggestions.copy()
        
    def reset_learning(self):
        """重置学习过程
        Reset learning process
        """
        # 重置学习相关的数据结构
        # Reset learning-related data structures
        self.performance_history = defaultdict(list)
        self.improvement_suggestions = []
        self.model_status_tracking = defaultdict(lambda: {
            'last_trained': None,
            'performance_score': 0.0,
            'improvement_rate': 0.0,
            'training_priority': 0
        })
        
        # 重置进度和状态
        self.learning_progress = 0.0
        self.current_learning_status = 'idle'
        self.learning_domains = []
        self.learning_priority = 'balanced'
        self.learning_logs = []
        
        error_handler.log_info("重置自主学习过程", "AutonomousLearningManager")
    
    def get_learning_progress(self):
        """获取自主学习进度
        Get autonomous learning progress
        
        Returns:
            dict: 包含进度、状态和日志的字典
        """
        return {
            "progress": round(self.learning_progress, 2),
            "status": self.current_learning_status,
            "logs": self.learning_logs.copy(),
            "domains": self.learning_domains,
            "priority": self.learning_priority
        }
    
    def _add_learning_log(self, message):
        """添加学习日志
        Add learning log
        
        Args:
            message: 日志消息
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message
        }
        
        self.learning_logs.append(log_entry)
        
        # 限制日志数量
        if len(self.learning_logs) > self.max_logs:
            self.learning_logs = self.learning_logs[-self.max_logs:]
