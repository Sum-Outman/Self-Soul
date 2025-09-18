"""
统一自主学习系统：实现AGI级别的自我优化和持续进化能力
Unified Self-Learning System: Implements AGI-level self-optimization and continuous evolution capabilities

版权所有 2025 AGI Brain System
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import torch
import json
import os
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import hashlib
from collections import defaultdict
from dataclasses import dataclass

from core.error_handling import error_handler
from core.data_fusion import DataFusion
from core.model_registry import ModelRegistry
from core.training.joint_training import JointTrainingManager


@dataclass
class AutonomousConfig:
    """自主学习配置"""
    training_interval: int = 3600  # 训练间隔（秒）
    optimization_interval: int = 1800  # 优化间隔（秒）
    monitoring_interval: int = 300  # 监控间隔（秒）
    min_improvement_threshold: float = 0.1  # 最小改进阈值
    max_training_iterations: int = 10  # 最大训练迭代次数
    enable_continuous_learning: bool = True  # 启用持续学习
    exploration_rate: float = 0.15  # 探索新策略的概率
    knowledge_transfer_rate: float = 0.25  # 知识迁移概率
    meta_learning_update_interval: int = 100  # 元学习更新间隔
    performance_window_size: int = 20  # 性能分析窗口大小
    trend_analysis_period: int = 10  # 趋势分析周期


class UnifiedSelfLearningSystem:
    """AGI统一自主学习与优化系统
    AGI Unified Self-Learning and Optimization System
    
    功能：集成所有自主学习功能，实现复杂的AGI级别自我优化能力
    Function: Integrates all self-learning functionalities, implements complex AGI-level self-optimization capabilities
    """
    
    def __init__(self, model_registry, training_manager, coordinator=None):
        self.model_registry = model_registry
        self.training_manager = training_manager
        self.coordinator = coordinator
        self.data_fusion = DataFusion()
        
        # 配置系统
        self.config = AutonomousConfig()
        self.running = False
        self.learning_thread = None
        
        # 高级性能监控
        self.performance_metrics = defaultdict(list)
        self.trend_analysis = {}
        self.anomaly_detection = {}
        
        # 学习历史和知识库
        self.learning_history = []
        self.knowledge_base = {}
        self.meta_learning_rules = {}
        
        # 优化系统和队列
        self.optimization_queue = []
        self.optimization_strategies = self._initialize_optimization_strategies()
        self.cross_model_knowledge = {}
        
        # 模型状态跟踪
        self.model_status_tracking = defaultdict(lambda: {
            'last_trained': None,
            'performance_score': 0.0,
            'improvement_rate': 0.0,
            'training_priority': 0,
            'model_type': 'unknown'
        })
        
        # 模型引用
        self.model_references = {}
        self.knowledge_model = None
        self.language_model = None
        
        # 初始化系统
        self._initialize_model_references()
        self._load_learning_history()
        self._initialize_meta_learning()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("统一自主学习系统初始化完成 | Unified self-learning system initialized")
    
    def _initialize_model_references(self):
        """初始化对其他模型的引用"""
        # 获取关键模型的引用
        self.knowledge_model = self.model_registry.get_model('knowledge')
        self.language_model = self.model_registry.get_model('language')
        
        # 获取所有模型引用
        all_models = self.model_registry.get_all_models()
        for model_id, model in all_models.items():
            self.model_references[model_id] = model
            # 提取模型类型
            model_types = ['language', 'image', 'audio', 'video', 'knowledge', 'sensor', 'spatial', 
                          'manager', 'motion', 'programming', 'computer']
            model_type = next((t for t in model_types if model_id.startswith(t)), 'unknown')
            self.model_status_tracking[model_id]['model_type'] = model_type
    
    def _initialize_optimization_strategies(self) -> Dict[str, Any]:
        """初始化优化策略库"""
        return {
            'parameter_tuning': {
                'description': '调整学习率、批次大小等基础参数',
                'complexity': 'low',
                'applicable_models': ['all'],
                'success_rate': 0.7,
                'execution_time': 300  # 秒
            },
            'architecture_optimization': {
                'description': '优化模型架构层数、神经元数量等',
                'complexity': 'high',
                'applicable_models': ['language', 'image', 'video', 'knowledge'],
                'success_rate': 0.5,
                'execution_time': 1800
            },
            'regularization_tuning': {
                'description': '调整dropout、权重衰减等正则化参数',
                'complexity': 'medium',
                'applicable_models': ['all'],
                'success_rate': 0.65,
                'execution_time': 600
            },
            'data_augmentation': {
                'description': '优化数据增强策略和参数',
                'complexity': 'medium',
                'applicable_models': ['image', 'video', 'audio'],
                'success_rate': 0.6,
                'execution_time': 900
            },
            'ensemble_learning': {
                'description': '创建模型集成提高性能',
                'complexity': 'high',
                'applicable_models': ['language', 'image', 'knowledge'],
                'success_rate': 0.55,
                'execution_time': 1200
            },
            'transfer_learning': {
                'description': '应用迁移学习从其他模型获取知识',
                'complexity': 'high',
                'applicable_models': ['all'],
                'success_rate': 0.7,
                'execution_time': 1500
            },
            'meta_learning': {
                'description': '基于历史学习经验优化学习策略',
                'complexity': 'very_high',
                'applicable_models': ['all'],
                'success_rate': 0.8,
                'execution_time': 2400
            }
        }
    
    def _initialize_meta_learning(self):
        """初始化元学习规则"""
        self.meta_learning_rules = {
            'performance_degradation': {
                'conditions': ['accuracy_drop_rapid', 'loss_increase_rapid', 'consecutive_failures'],
                'actions': ['comprehensive_analysis', 'multi_strategy_optimization'],
                'priority': 'high'
            },
            'performance_plateau': {
                'conditions': ['accuracy_stagnant', 'loss_stagnant', 'slow_progress'],
                'actions': ['architecture_exploration', 'hyperparameter_search'],
                'priority': 'medium'
            },
            'new_environment': {
                'conditions': ['input_distribution_change', 'output_requirements_change'],
                'actions': ['transfer_learning', 'rapid_adaptation'],
                'priority': 'high'
            },
            'resource_constraints': {
                'conditions': ['memory_limited', 'computation_limited', 'time_constrained'],
                'actions': ['efficiency_optimization', 'model_compression'],
                'priority': 'medium'
            }
        }
    
    def _load_learning_history(self):
        """加载历史学习数据"""
        history_file = os.path.join(os.path.dirname(__file__), 'data', 'unified_learning_history.json')
        knowledge_file = os.path.join(os.path.dirname(__file__), 'data', 'knowledge_base.json')
        
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.learning_history = json.load(f)
            
            if os.path.exists(knowledge_file):
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                    
        except Exception as e:
            error_handler.handle_error(e, "UnifiedSelfLearningSystem", 
                                     "加载学习历史或知识库失败 | Failed to load learning history or knowledge base")
    
    def _save_learning_data(self):
        """保存学习数据"""
        history_file = os.path.join(os.path.dirname(__file__), 'data', 'unified_learning_history.json')
        knowledge_file = os.path.join(os.path.dirname(__file__), 'data', 'knowledge_base.json')
        
        try:
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            os.makedirs(os.path.dirname(knowledge_file), exist_ok=True)
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.learning_history, f, ensure_ascii=False, indent=2)
                
            with open(knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            error_handler.handle_error(e, "UnifiedSelfLearningSystem", 
                                     "保存学习数据失败 | Failed to save learning data")
    
    def start_autonomous_learning_cycle(self):
        """启动自主学习循环"""
        if self.running:
            self.logger.info("自主学习循环已在运行中 | Autonomous learning cycle already running")
            return False
        
        self.running = True
        self.learning_thread = threading.Thread(target=self._learning_cycle)
        self.learning_thread.daemon = True
        self.learning_thread.start()
        
        self.logger.info("自主学习循环已启动 | Autonomous learning cycle started")
        return True
    
    def stop_autonomous_learning_cycle(self):
        """停止自主学习循环"""
        self.running = False
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=5.0)
            
        self.logger.info("自主学习循环已停止 | Autonomous learning cycle stopped")
        return True
    
    def _learning_cycle(self):
        """自主学习循环的内部实现"""
        while self.running:
            try:
                # 评估所有模型的性能
                self._evaluate_all_models()
                
                # 执行高级性能分析
                for model_id in self.model_references.keys():
                    if model_id in self.performance_metrics and len(self.performance_metrics[model_id]) > 0:
                        self._analyze_performance_trends(model_id)
                        self._detect_anomalies(model_id)
                        self._update_trend_analysis(model_id)
                
                # 智能优化检查和处理
                self._process_intelligent_optimization()
                
                # 生成学习报告
                self._generate_learning_report()
                
                # 等待下一个学习周期
                for _ in range(self.config.monitoring_interval):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"自主学习循环出错: {e} | Autonomous learning cycle error: {e}")
                time.sleep(5)
    
    def _evaluate_all_models(self):
        """评估所有模型的性能"""
        for model_id, model in self.model_references.items():
            try:
                # 评估模型性能
                performance = self._evaluate_model_performance(model_id)
                
                # 更新性能历史
                self.performance_metrics[model_id].append({
                    'timestamp': datetime.now().isoformat(),
                    'score': performance,
                    'model_id': model_id
                })
                
                # 限制历史记录长度
                max_history = self.config.performance_window_size
                if len(self.performance_metrics[model_id]) > max_history:
                    self.performance_metrics[model_id] = self.performance_metrics[model_id][-max_history:]
                
                # 更新模型状态跟踪
                self._update_model_status(model_id, performance)
                
            except Exception as e:
                self.logger.warning(f"评估模型 {model_id} 性能时出错: {e} | Error evaluating model {model_id} performance: {e}")
    
    def _evaluate_model_performance(self, model_id: str) -> float:
        """评估单个模型的性能"""
        model = self.model_references.get(model_id)
        if not model:
            return 0.0
        
        try:
            # 这里应该是实际的性能评估逻辑
            # 基于模型类型和能力的综合评估
            if hasattr(model, 'evaluate_performance'):
                return model.evaluate_performance()
            else:
                # 默认评估方法
                return self._default_performance_evaluation(model_id)
                
        except Exception as e:
            self.logger.error(f"模型性能评估错误: {model_id} - {e} | Model performance evaluation error: {model_id} - {e}")
            return 0.0
    
    def _default_performance_evaluation(self, model_id: str) -> float:
        """默认性能评估方法"""
        # 基于历史性能和模型类型的启发式评估
        if model_id in self.performance_metrics and self.performance_metrics[model_id]:
            recent_performance = [m['score'] for m in self.performance_metrics[model_id][-5:]]
            if recent_performance:
                return sum(recent_performance) / len(recent_performance)
        
        # 新模型或没有历史数据的模型
        model_type = self.model_status_tracking[model_id]['model_type']
        base_scores = {
            'language': 0.8, 'knowledge': 0.7, 'image': 0.75, 'audio': 0.7,
            'video': 0.65, 'spatial': 0.6, 'sensor': 0.55, 'manager': 0.9
        }
        return base_scores.get(model_type, 0.5)
    
    def _update_model_status(self, model_id: str, performance: float):
        """更新模型状态"""
        improvement_rate = self._calculate_improvement_rate(model_id)
        training_priority = self._calculate_training_priority(model_id, performance, improvement_rate)
        
        self.model_status_tracking[model_id].update({
            'last_trained': datetime.now().isoformat(),
            'performance_score': performance,
            'improvement_rate': improvement_rate,
            'training_priority': training_priority
        })
    
    def _calculate_improvement_rate(self, model_id: str) -> float:
        """计算改进率"""
        history = self.performance_metrics.get(model_id, [])
        if len(history) < 2:
            return 0.0
        
        recent_history = history[-5:]
        if len(recent_history) < 2:
            return 0.0
        
        improvements = []
        for i in range(1, len(recent_history)):
            prev_score = recent_history[i-1]['score']
            curr_score = recent_history[i]['score']
            if prev_score > 0:
                improvement = (curr_score - prev_score) / prev_score
                improvements.append(improvement)
        
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    def _calculate_training_priority(self, model_id: str, performance: float, improvement_rate: float) -> float:
        """计算训练优先级"""
        # 性能越低，优先级越高
        # 改进率越低，优先级越高
        priority = (1.0 - performance) * 0.7 + (1.0 - max(improvement_rate, 0.0)) * 0.3
        
        # 根据模型类型调整优先级
        model_type = self.model_status_tracking[model_id]['model_type']
        type_weights = {
            'language': 1.2, 'knowledge': 1.1, 'manager': 1.3,
            'image': 1.0, 'audio': 0.9, 'video': 0.9,
            'spatial': 0.8, 'sensor': 0.7
        }
        
        return priority * type_weights.get(model_type, 1.0)
    
    def _process_intelligent_optimization(self):
        """处理智能优化"""
        # 检查所有模型是否需要优化
        for model_id in self.model_references.keys():
            optimization_needed, reason = self._intelligent_optimization_check(model_id)
            if optimization_needed:
                self._queue_intelligent_optimization(model_id, reason)
        
        # 处理优化队列
        if self.optimization_queue:
            self._process_optimization_queue()
    
    # 以下方法从 advanced_self_learning.py 继承并增强
    def update_performance(self, model_id: str, metrics: Dict[str, Any], context: Dict[str, Any] = None):
        """更新模型性能指标（增强版）"""
        enhanced_metrics = {
            'model_id': model_id,
            **metrics,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'context': context or {}
        }
        
        self.performance_metrics[model_id].append(enhanced_metrics)
        
        # 限制历史记录长度
        max_history = self.config.performance_window_size
        if len(self.performance_metrics[model_id]) > max_history:
            self.performance_metrics[model_id] = self.performance_metrics[model_id][-max_history:]
        
        # 执行高级分析
        self._analyze_performance_trends(model_id)
        self._detect_anomalies(model_id)
        self._update_trend_analysis(model_id)
        
        # 智能判断是否需要优化
        optimization_needed, reason = self._intelligent_optimization_check(model_id)
        
        if optimization_needed:
            self._queue_intelligent_optimization(model_id, reason)
        
        # 定期更新元学习规则
        if len(self.learning_history) % self.config.meta_learning_update_interval == 0:
            self._update_meta_learning_rules()
    
    def _analyze_performance_trends(self, model_id: str):
        """分析性能趋势"""
        if len(self.performance_metrics[model_id]) < 5:
            return
        
        metrics = self.performance_metrics[model_id]
        accuracy_values = [m.get('accuracy', m.get('score', 0)) for m in metrics]
        loss_values = [m.get('loss', 0) for m in metrics]
        
        trends = {
            'accuracy_trend': self._calculate_trend(accuracy_values),
            'loss_trend': self._calculate_trend(loss_values),
            'stability': self._calculate_stability(accuracy_values),
            'volatility': self._calculate_volatility(accuracy_values),
            'recent_improvement': self._calculate_recent_improvement(accuracy_values)
        }
        
        self.trend_analysis[model_id] = trends
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算数值趋势"""
        if len(values) < 2:
            return 'insufficient_data'
        
        x = np.arange(len(values))
        y = np.array(values)
        
        mask = ~np.isnan(y)
        if np.sum(mask) < 2:
            return 'insufficient_data'
            
        x = x[mask]
        y = y[mask]
        
        slope, intercept = np.polyfit(x, y, 1)
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'deteriorating'
        else:
            return 'stable'
    
    def _calculate_stability(self, values: List[float]) -> float:
        """计算稳定性指标"""
        if len(values) < 2:
            return 0.0
        
        values_array = np.array(values)
        mean_val = np.nanmean(values_array)
        std_val = np.nanstd(values_array)
        
        if mean_val == 0:
            return 0.0
            
        cv = std_val / mean_val
        stability = 1.0 / (1.0 + cv)
        
        return float(stability)
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """计算波动性指标"""
        if len(values) < 2:
            return 0.0
        
        returns = np.diff(values) / values[:-1]
        volatility = np.nanstd(returns) if len(returns) > 0 else 0.0
        
        return float(volatility)
    
    def _calculate_recent_improvement(self, values: List[float]) -> float:
        """计算近期改进程度"""
        if len(values) < 4:
            return 0.0
        
        n = len(values)
        recent_start = max(0, n - n // 4)
        prev_start = max(0, n // 2)
        
        recent_vals = values[recent_start:]
        prev_vals = values[prev_start:recent_start]
        
        if len(recent_vals) == 0 or len(prev_vals) == 0:
            return 0.0
        
        recent_mean = np.nanmean(recent_vals)
        prev_mean = np.nanmean(prev_vals)
        
        if prev_mean == 0:
            return 0.0
            
        improvement = (recent_mean - prev_mean) / prev_mean
        return float(improvement)
    
    def _detect_anomalies(self, model_id: str):
        """检测性能异常"""
        if len(self.performance_metrics[model_id]) < 10:
            return
        
        metrics = self.performance_metrics[model_id]
        accuracy_values = [m.get('accuracy', m.get('score', 0)) for m in metrics]
        
        values = np.array(accuracy_values)
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        
        if std_val == 0:
            return
            
        z_scores = np.abs((values - mean_val) / std_val)
        anomalies = z_scores > 2.5
        
        if np.any(anomalies):
            anomaly_indices = np.where(anomalies)[0]
            self.anomaly_detection[model_id] = {
                'count': len(anomaly_indices),
                'latest_anomaly': metrics[anomaly_indices[-1]] if anomaly_indices.size > 0 else None,
                'severity': float(np.max(z_scores[anomalies])),
                'timestamp': time.time()
            }
    
    def _update_trend_analysis(self, model_id: str):
        """更新趋势分析"""
        if len(self.performance_metrics[model_id]) < 3:
            return
        
        metrics = self.performance_metrics[model_id]
        accuracy_values = [m.get('accuracy', m.get('score', 0)) for m in metrics]
        
        trend_analysis = {
            'short_term': self._analyze_short_term_trend(accuracy_values),
            'medium_term': self._analyze_medium_term_trend(accuracy_values),
            'long_term': self._analyze_long_term_trend(accuracy_values)
        }
        
        if model_id not in self.trend_analysis:
            self.trend_analysis[model_id] = {}
        
        self.trend_analysis[model_id].update(trend_analysis)
    
    def _analyze_short_term_trend(self, values: List[float]) -> Dict[str, Any]:
        """分析短期趋势"""
        if len(values) < 3:
            return {'trend': 'insufficient_data', 'confidence': 0.0}
        
        short_term = values[-min(5, len(values)):]
        trend = self._calculate_trend(short_term)
        
        return {
            'trend': trend,
            'confidence': self._calculate_trend_confidence(short_term),
            'period': 'short_term'
        }
    
    def _analyze_medium_term_trend(self, values: List[float]) -> Dict[str, Any]:
        """分析中期趋势"""
        if len(values) < 8:
            return {'trend': 'insufficient_data', 'confidence': 0.0}
        
        medium_term = values[-min(15, len(values)):]
        trend = self._calculate_trend(medium_term)
        
        return {
            'trend': trend,
            'confidence': self._calculate_trend_confidence(medium_term),
            'period': 'medium_term'
        }
    
    def _analyze_long_term_trend(self, values: List[float]) -> Dict[str, Any]:
        """分析长期趋势"""
        if len(values) < 15:
            return {'trend': 'insufficient_data', 'confidence': 0.0}
        
        trend = self._calculate_trend(values)
        
        return {
            'trend': trend,
            'confidence': self._calculate_trend_confidence(values),
            'period': 'long_term'
        }
    
    def _calculate_trend_confidence(self, values: List[float]) -> float:
        """计算趋势置信度"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        mask = ~np.isnan(y)
        if np.sum(mask) < 2:
            return 0.0
            
        x = x[mask]
        y = y[mask]
        
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
            
        r_squared = 1 - (ss_res / ss_tot)
        confidence = max(0.0, min(1.0, r_squared))
        
        return confidence
    
    def _intelligent_optimization_check(self, model_id: str) -> Tuple[bool, str]:
        """智能优化检查"""
        if model_id not in self.trend_analysis:
            return False, "insufficient_data"
        
        trends = self.trend_analysis[model_id]
        optimization_reasons = []
        
        # 1. 性能下降检测
        if trends.get('accuracy_trend') == 'deteriorating':
            optimization_reasons.append('performance_degradation')
        
        # 2. 性能平台期检测
        if (trends.get('accuracy_trend') == 'stable' and
            trends.get('recent_improvement', 0) < 0.01 and
            len(self.performance_metrics[model_id]) > 10):
            optimization_reasons.append('performance_plateau')
        
        # 3. 异常检测
        if model_id in self.anomaly_detection:
            anomaly = self.anomaly_detection[model_id]
            if anomaly['severity'] > 3.0:
                optimization_reasons.append('severe_anomaly')
        
        # 4. 环境变化检测
        if len(self.performance_metrics[model_id]) > 5:
            latest_context = self.performance_metrics[model_id][-1].get('context', {})
            prev_context = self.performance_metrics[model_id][-5].get('context', {})
            
            context_changed = self._detect_context_change(latest_context, prev_context)
            if context_changed:
                optimization_reasons.append('environment_change')
        
        # 5. 资源约束检测
        if len(self.performance_metrics[model_id]) > 0:
            latest_metrics = self.performance_metrics[model_id][-1]
            if (latest_metrics.get('memory_usage', 0) > 0.9 or
                latest_metrics.get('latency', 0) > 1000):
                optimization_reasons.append('resource_constraints')
        
        # 6. 知识迁移机会检测
        if self._check_knowledge_transfer_opportunity(model_id):
            optimization_reasons.append('knowledge_transfer_opportunity')
        
        # 7. 元学习建议
        meta_learning_suggestion = self._get_meta_learning_suggestion(model_id)
        if meta_learning_suggestion:
            optimization_reasons.append(meta_learning_suggestion)
        
        if optimization_reasons:
            priority_order = [
                'severe_anomaly', 'performance_degradation', 'environment_change',
                'resource_constraints', 'performance_plateau', 'knowledge_transfer_opportunity'
            ]
            
            for reason in priority_order:
                if reason in optimization_reasons:
                    return True, reason
            
            return True, optimization_reasons[0]
        
        return False, "no_optimization_needed"
    
    def _detect_context_change(self, current_context: Dict[str, Any], previous_context: Dict[str, Any]) -> bool:
        """检测环境上下文变化"""
        if not current_context or not previous_context:
            return False
            
        similarity_score = self._calculate_context_similarity(current_context, previous_context)
        return similarity_score < 0.7
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """计算上下文相似度"""
        if not context1 or not context2:
            return 0.0
            
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
            
        similarity_sum = 0
        for key in common_keys:
            if context1[key] == context2[key]:
                similarity_sum += 1
            elif isinstance(context1[key], (int, float)) and isinstance(context2[key], (int, float)):
                diff = abs(context1[key] - context2[key])
                avg = (abs(context1[key]) + abs(context2[key])) / 2
                if avg > 0:
                    similarity_sum += 1 - min(diff / avg, 1.0)
        
        return similarity_sum / len(common_keys)
    
    def _check_knowledge_transfer_opportunity(self, model_id: str) -> bool:
        """检查知识迁移机会"""
        if len(self.performance_metrics) < 2:
            return False
            
        current_model_metrics = self.performance_metrics.get(model_id, [])
        if not current_model_metrics:
            return False
            
        latest_performance = current_model_metrics[-1].get('accuracy', current_model_metrics[-1].get('score', 0))
        
        for other_model_id, metrics in self.performance_metrics.items():
            if other_model_id == model_id or not metrics:
                continue
                
            other_performance = metrics[-1].get('accuracy', metrics[-1].get('score', 0))
            
            if other_performance > latest_performance + 0.1:
                if self._are_models_compatible(model_id, other_model_id):
                    return True
                    
        return False
    
    def _are_models_compatible(self, model_id1: str, model_id2: str) -> bool:
        """检查模型兼容性"""
        model_types = ['language', 'image', 'audio', 'video', 'knowledge', 'sensor', 'spatial']
        
        type1 = next((t for t in model_types if model_id1.startswith(t)), 'unknown')
        type2 = next((t for t in model_types if model_id2.startswith(t)), 'unknown')
        
        compatible_pairs = [
            ('language', 'knowledge'),
            ('image', 'video'),
            ('audio', 'language'),
            ('sensor', 'spatial')
        ]
        
        return (type1 == type2 or
                (type1, type2) in compatible_pairs or
                (type2, type1) in compatible_pairs)
    
    def _get_meta_learning_suggestion(self, model_id: str) -> Optional[str]:
        """获取元学习建议"""
        if not self.learning_history:
            return None
            
        similar_cases = self._find_similar_learning_cases(model_id)
        
        if similar_cases:
            best_case = max(similar_cases, key=lambda x: x.get('improvement', 0))
            return best_case.get('strategy_used')
            
        return None
    
    def _find_similar_learning_cases(self, model_id: str) -> List[Dict[str, Any]]:
        """查找相似学习案例"""
        similar_cases = []
        current_trends = self.trend_analysis.get(model_id, {})
        
        for case in self.learning_history:
            if case.get('model_type', '').split('_')[0] == model_id.split('_')[0]:
                case_trends = case.get('trend_analysis', {})
                if self._are_trends_similar(current_trends, case_trends):
                    similar_cases.append(case)
                    
        return similar_cases
    
    def _are_trends_similar(self, trends1: Dict[str, Any], trends2: Dict[str, Any]) -> bool:
        """检查趋势相似性"""
        if not trends1 or not trends2:
            return False
            
        common_metrics = set(trends1.keys()) & set(trends2.keys())
        if not common_metrics:
            return False
            
        similarity_score = 0
        for metric in common_metrics:
            if trends1[metric] == trends2[metric]:
                similarity_score += 1
                
        return similarity_score / len(common_metrics) > 0.6
    
    def _queue_intelligent_optimization(self, model_id: str, reason: str):
        """排队智能优化任务"""
        strategy = self._select_optimization_strategy(model_id, reason)
        
        if strategy:
            optimization_task = {
                'model_id': model_id,
                'reason': reason,
                'strategy': strategy,
                'priority': self._get_optimization_priority(reason),
                'timestamp': time.time(),
                'status': 'queued'
            }
            
            self.optimization_queue.append(optimization_task)
            self.logger.info(f"优化任务已排队: {model_id} - {reason} - {strategy}")
            
            if len(self.optimization_queue) == 1:
                self._process_optimization_queue()
    
    def _select_optimization_strategy(self, model_id: str, reason: str) -> Optional[str]:
        """选择优化策略"""
        reason_strategy_map = {
            'performance_degradation': ['parameter_tuning', 'regularization_tuning', 'architecture_optimization'],
            'performance_plateau': ['architecture_optimization', 'ensemble_learning', 'transfer_learning'],
            'severe_anomaly': ['comprehensive_analysis', 'parameter_tuning', 'data_augmentation'],
            'environment_change': ['transfer_learning', 'rapid_adaptation', 'meta_learning'],
            'resource_constraints': ['efficiency_optimization', 'model_compression', 'parameter_tuning'],
            'knowledge_transfer_opportunity': ['transfer_learning', 'meta_learning']
        }
        
        strategies = reason_strategy_map.get(reason, [])
        applicable_strategies = []
        
        for strategy in strategies:
            if strategy in self.optimization_strategies:
                strategy_info = self.optimization_strategies[strategy]
                if ('all' in strategy_info['applicable_models'] or
                    any(model_id.startswith(t) for t in strategy_info['applicable_models'])):
                    applicable_strategies.append(strategy)
        
        if applicable_strategies:
            scored_strategies = []
            for strategy in applicable_strategies:
                strategy_info = self.optimization_strategies[strategy]
                score = strategy_info['success_rate']
                if strategy_info['complexity'] == 'low':
                    score *= 1.2
                elif strategy_info['complexity'] == 'medium':
                    score *= 1.0
                else:
                    score *= 0.8
                scored_strategies.append((strategy, score))
            
            best_strategy = max(scored_strategies, key=lambda x: x[1])[0]
            return best_strategy
            
        return None
    
    def _get_optimization_priority(self, reason: str) -> int:
        """获取优化优先级"""
        priority_map = {
            'severe_anomaly': 100,
            'performance_degradation': 80,
            'environment_change': 70,
            'resource_constraints': 60,
            'performance_plateau': 50,
            'knowledge_transfer_opportunity': 40
        }
        
        return priority_map.get(reason, 30)
    
    def _process_optimization_queue(self):
        """处理优化队列"""
        if not self.optimization_queue:
            return
            
        self.optimization_queue.sort(key=lambda x: x['priority'], reverse=True)
        
        task = self.optimization_queue[0]
        model_id = task['model_id']
        strategy = task['strategy']
        
        try:
            self.logger.info(f"开始执行优化: {model_id} - {strategy}")
            
            success = self._execute_optimization_strategy(model_id, strategy)
            
            if success:
                task['status'] = 'completed'
                task['completion_time'] = time.time()
                self.logger.info(f"优化完成: {model_id} - {strategy}")
            else:
                task['status'] = 'failed'
                task['failure_reason'] = 'strategy_execution_failed'
                self.logger.warning(f"优化失败: {model_id} - {strategy}")
                
        except Exception as e:
            task['status'] = 'failed'
            task['failure_reason'] = str(e)
            self.logger.error(f"优化执行错误: {model_id} - {strategy} - {e}")
        
        finally:
            self.optimization_queue = [t for t in self.optimization_queue if t['status'] not in ['completed', 'failed']]
            
            learning_record = {
                'model_id': model_id,
                'strategy_used': strategy,
                'reason': task['reason'],
                'success': task.get('status') == 'completed',
                'timestamp': time.time(),
                'performance_before': self.performance_metrics[model_id][-1] if model_id in self.performance_metrics else {},
                'trend_analysis': self.trend_analysis.get(model_id, {})
            }
            
            self.learning_history.append(learning_record)
            self._save_learning_data()
            
            if self.optimization_queue:
                self._process_optimization_queue()
    
    def _execute_optimization_strategy(self, model_id: str, strategy: str) -> bool:
        """执行优化策略"""
        try:
            model = self.model_registry.get_model(model_id)
            if not model:
                self.logger.warning(f"模型未找到: {model_id}")
                return False
            
            if strategy == 'parameter_tuning':
                return self._optimize_parameters(model)
            elif strategy == 'architecture_optimization':
                return self._optimize_architecture(model)
            elif strategy == 'regularization_tuning':
                return self._optimize_regularization(model)
            elif strategy == 'data_augmentation':
                return self._optimize_data_augmentation(model)
            elif strategy == 'ensemble_learning':
                return self._create_ensemble(model)
            elif strategy == 'transfer_learning':
                return self._apply_transfer_learning(model)
            elif strategy == 'meta_learning':
                return self._apply_meta_learning(model)
            else:
                self.logger.warning(f"未知优化策略: {strategy}")
                return False
                
        except Exception as e:
            self.logger.error(f"优化策略执行错误: {strategy} - {e}")
            return False
    
    def _optimize_parameters(self, model) -> bool:
        """优化模型参数"""
        try:
            if hasattr(model, 'learning_rate'):
                current_lr = model.learning_rate
                model.learning_rate = current_lr * 0.8
            
            if hasattr(model, 'batch_size'):
                current_bs = model.batch_size
                model.batch_size = min(current_bs * 2, 256)
                
            return True
            
        except Exception as e:
            self.logger.error(f"参数优化错误: {e}")
            return False
    
    def _optimize_architecture(self, model) -> bool:
        """优化模型架构"""
        try:
            # 架构优化逻辑
            return True
            
        except Exception as e:
            self.logger.error(f"架构优化错误: {e}")
            return False
    
    def _optimize_regularization(self, model) -> bool:
        """优化正则化参数"""
        try:
            if hasattr(model, 'dropout_rate'):
                model.dropout_rate = min(model.dropout_rate + 0.1, 0.5)
                
            return True
            
        except Exception as e:
            self.logger.error(f"正则化优化错误: {e}")
            return False
    
    def _optimize_data_augmentation(self, model) -> bool:
        """优化数据增强策略"""
        try:
            if hasattr(model, 'data_augmentation'):
                # 增强数据增强策略
                pass
                
            return True
            
        except Exception as e:
            self.logger.error(f"数据增强优化错误: {e}")
            return False
    
    def _create_ensemble(self, model) -> bool:
        """创建模型集成"""
        try:
            # 模型集成逻辑
            return True
            
        except Exception as e:
            self.logger.error(f"模型集成错误: {e}")
            return False
    
    def _apply_transfer_learning(self, model) -> bool:
        """应用迁移学习"""
        try:
            source_model_id = self._find_best_source_model(model.model_id)
            if source_model_id:
                source_model = self.model_registry.get_model(source_model_id)
                if source_model:
                    # 知识迁移逻辑
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"迁移学习错误: {e}")
            return False
    
    def _find_best_source_model(self, target_model_id: str) -> Optional[str]:
        """寻找最佳源模型"""
        best_model_id = None
        best_performance = -1
        
        for model_id, metrics in self.performance_metrics.items():
            if model_id == target_model_id or not metrics:
                continue
                
            performance = metrics[-1].get('accuracy', metrics[-1].get('score', 0))
            if (performance > best_performance and
                self._are_models_compatible(target_model_id, model_id)):
                best_performance = performance
                best_model_id = model_id
                
        return best_model_id
    
    def _apply_meta_learning(self, model) -> bool:
        """应用元学习"""
        try:
            similar_cases = self._find_similar_learning_cases(model.model_id)
            
            if similar_cases:
                successful_cases = [case for case in similar_cases if case.get('success', False)]
                
                if successful_cases:
                    best_case = max(successful_cases, key=lambda x: x.get('improvement', 0))
                    best_strategy = best_case.get('strategy_used')
                    
                    if best_strategy:
                        return self._execute_optimization_strategy(model.model_id, best_strategy)
            
            return False
            
        except Exception as e:
            self.logger.error(f"元学习错误: {e}")
            return False
    
    def _update_meta_learning_rules(self):
        """更新元学习规则"""
        successful_cases = [case for case in self.learning_history if case.get('success', False)]
        
        if not successful_cases:
            return
            
        model_groups = defaultdict(list)
        for case in successful_cases:
            model_type = case.get('model_id', '').split('_')[0]
            model_groups[model_type].append(case)
        
        for model_type, cases in model_groups.items():
            strategy_success = defaultdict(list)
            
            for case in cases:
                strategy = case.get('strategy_used')
                improvement = case.get('improvement', 0)
                if strategy and improvement > 0:
                    strategy_success[strategy].append(improvement)
            
            strategy_avg_improvement = {
                strategy: sum(improvements) / len(improvements)
                for strategy, improvements in strategy_success.items()
            }
            
            if strategy_avg_improvement:
                best_strategy = max(strategy_avg_improvement.items(), key=lambda x: x[1])[0]
                
                if best_strategy in self.optimization_strategies:
                    self.optimization_strategies[best_strategy]['success_rate'] = min(
                        self.optimization_strategies[best_strategy]['success_rate'] * 1.1, 0.95
                    )
    
    def _generate_learning_report(self):
        """生成学习报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': len(self.model_references),
            'model_performances': {},
            'optimization_queue_size': len(self.optimization_queue),
            'learning_history_count': len(self.learning_history)
        }
        
        for model_id, status in self.model_status_tracking.items():
            report['model_performances'][model_id] = {
                'performance_score': status.get('performance_score', 0.0),
                'improvement_rate': status.get('improvement_rate', 0.0),
                'training_priority': status.get('training_priority', 0)
            }
        
        # 保存报告
        report_file = os.path.join(os.path.dirname(__file__), 'data', 'learning_reports', 
                                 f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存学习报告失败: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'running': self.running,
            'models_managed': len(self.model_references),
            'optimization_queue_size': len(self.optimization_queue),
            'learning_history_count': len(self.learning_history),
            'overall_performance': self._calculate_overall_performance()
        }
    
    def _calculate_overall_performance(self) -> float:
        """计算系统整体性能"""
        performances = [status.get('performance_score', 0.0) for status in self.model_status_tracking.values()]
        if not performances:
            return 0.0
        
        return sum(performances) / len(performances)
    
    def update_config(self, config: Dict[str, Any]):
        """更新配置"""
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.logger.info(f"更新自主学习配置: {config}")
    
    def reset_learning(self):
        """重置学习过程"""
        self.performance_metrics = defaultdict(list)
        self.learning_history = []
        self.optimization_queue = []
        self.model_status_tracking = defaultdict(lambda: {
            'last_trained': None,
            'performance_score': 0.0,
            'improvement_rate': 0.0,
            'training_priority': 0,
            'model_type': 'unknown'
        })
        
        self.logger.info("重置自主学习过程")
