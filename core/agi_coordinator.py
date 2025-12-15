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


"""
AGICoordinator类 - AGI系统核心协调器
AGICoordinator Class - Core AGI System Coordinator
"""
class AGICoordinator:
    """Self Brain  AGI中央协调器，管理和协调所有认知组件"""
    
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
        
        # 更新系统状态
        self._update_system_state()
        
        # 加载长期记忆
        self.load_long_term_memory()
        
        # 启动自主学习循环
        self._start_autonomous_learning_loop()
        
        error_handler.log_info("AGI协调器初始化完成，统一认知架构已就绪", "AGICoordinator")
    
    def _calculate_initial_agi_level(self):
        """基于可用模型和能力动态计算初始AGI水平"""
        try:
            # 获取所有模型状态
            models_status = self.model_registry.get_all_models_status()
            if not models_status:
                return 0.1  # 最低水平，如果没有模型
            
            # 计算模型覆盖率和能力分数
            model_coverage = len(models_status) / 20  # 假设最多20种模型类型
            capability_score = 0.0
            
            # 根据模型类型和能力加权计算
            model_weights = {
                'language': 0.15, 'audio': 0.10, 'vision_image': 0.12,
                'vision_video': 0.08, 'sensor': 0.05, 'manager': 0.20,
                'planning': 0.10, 'reasoning': 0.15, 'learning': 0.05
            }
            
            for model_id, status in models_status.items():
                weight = model_weights.get(model_id, 0.02)
                # 基于模型状态调整权重（如果模型已加载且可用）
                if status.get('loaded', False) and status.get('status') == 'active':
                    capability_score += weight
            
            # 综合评分：模型覆盖率(30%) + 能力分数(50%) + 基础分(20%)
            agi_level = (model_coverage * 0.3) + (capability_score * 0.5) + 0.2
            return min(max(agi_level, 0.1), 1.0)  # 保持在0.1-1.0范围内
            
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "计算AGI水平失败")
            return 0.1  # 失败时返回最低水平
    
    def _start_autonomous_learning_loop(self):
        """启动自主学习的后台循环"""
        import threading
        
        def learning_loop():
            while self.system_state['status'] != 'shutdown':
                try:
                    # 基于系统状态决定学习强度
                    learning_intensity = max(0.1, self.system_state['agi_level'])
                    
                    # 运行自主学习
                    self.self_learning.autonomous_learn(learning_intensity)
                    
                    # 基于性能动态调整学习间隔
                    avg_processing_time = np.mean([
                        metrics.get('processing_time', 1.0) 
                        for metrics in self.system_state['performance_metrics'].values()
                    ]) if self.system_state['performance_metrics'] else 1.0
                    
                    sleep_time = max(5.0, min(30.0, avg_processing_time * 2))
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    error_handler.handle_error(e, "AGICoordinator", "自主学习循环异常")
                    time.sleep(10)  # 异常时等待10秒
        
        # 启动后台线程
        learning_thread = threading.Thread(target=learning_loop, daemon=True)
        learning_thread.start()
        error_handler.log_info("自主学习循环已启动", "AGICoordinator")
    
    
    """
    _update_system_state函数 - 中文函数描述
    _update_system_state Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
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
        try:
            # 获取模型状态
            models_status = self.model_registry.get_all_models_status()
            if not models_status:
                return 0.1
            
            # 基础能力分数（基于模型覆盖率和性能）
            base_score = self.system_state['agi_level']  # 初始值
            
            # 性能加权因子
            performance_factor = performance_metrics['success_rate'] * 0.4 + \
                                (1 - performance_metrics['error_rate']) * 0.3 + \
                                (1 / (1 + performance_metrics['avg_processing_time'])) * 0.3
            
            # 学习进度因子
            learning_factor = self.system_state['learning_progress'] * 0.2
            
            # 元认知信心因子
            meta_cognition_factor = self.meta_cognition['confidence_level'] * 0.1
            
            # 综合计算新AGI水平
            new_agi_level = base_score * 0.6 + performance_factor * 0.2 + learning_factor * 0.1 + meta_cognition_factor * 0.1
            
            # 平滑过渡，避免剧烈变化
            smoothed_agi_level = 0.8 * self.system_state['agi_level'] + 0.2 * new_agi_level
            
            return min(max(smoothed_agi_level, 0.1), 1.0)
            
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "计算当前AGI水平失败")
            return self.system_state['agi_level']  # 保持原值
    
    def _calculate_autonomy_level(self, performance_metrics):
        """计算自主性水平"""
        try:
            # 基于任务成功率和学习进度
            autonomy = performance_metrics['success_rate'] * 0.6 + \
                      self.system_state['learning_progress'] * 0.4
            
            return min(max(autonomy, 0.1), 1.0)
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "计算自主性水平失败")
            return 0.1
    
    def _calculate_generalization_score(self, performance_metrics):
        """计算泛化能力评分"""
        try:
            # 基于知识覆盖率和任务多样性
            knowledge_coverage = self._calculate_knowledge_coverage()
            task_diversity = self._calculate_task_diversity()
            
            generalization = knowledge_coverage * 0.5 + task_diversity * 0.5
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
            
            # 归一化处理
            max_experiences = 1000  # 假设最大经验数
            max_patterns = 500     # 假设最大模式数
            
            coverage = (min(experience_count / max_experiences, 1.0) * 0.6 + 
                       min(pattern_count / max_patterns, 1.0) * 0.4)
            
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
        try:
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
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "处理用户输入失败")
            return {"error": str(e)}
    
    
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
        """处理音频输入"""
        audio_model = self.model_registry.get_model('audio')
        if not audio_model:
            raise RuntimeError("音频模型未加载")
        
        # 使用音频模型处理音频数据
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
        """处理图像输入"""
        image_model = self.model_registry.get_model('vision_image')
        if not image_model:
            raise RuntimeError("图像模型未加载")
        
        # 使用图像模型处理图像数据
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
        try:
            start_time = time.time()
            error_handler.log_info(f"开始协调任务: {task_description[:50]}...", "AGICoordinator")
            
            # 记录任务开始信息
            task_id = f"task_{int(time.time() * 1000)}_{hash(task_description) % 10000}"
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
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "任务协调失败")
            return {"error": str(e)}
    
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
        
        return {
            **self.system_state,
            'models_status': models_status,
            'meta_cognition': self.meta_cognition,
            'memory_size': len(self.long_term_memory),
            'learning_status': learning_status
        }
    
    
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
        memory_file = os.path.join(os.path.dirname(__file__), 'data', 'long_term_memory.json')
        try:
            os.makedirs(os.path.dirname(memory_file), exist_ok=True)
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.long_term_memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "保存长期记忆失败")
    
    
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
        memory_file = os.path.join(os.path.dirname(__file__), 'data', 'long_term_memory.json')
        try:
            if os.path.exists(memory_file):
                with open(memory_file, 'r', encoding='utf-8') as f:
                    self.long_term_memory = json.load(f)
        except Exception as e:
            error_handler.handle_error(e, "AGICoordinator", "加载长期记忆失败")
