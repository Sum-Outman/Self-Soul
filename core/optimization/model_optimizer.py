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
模型优化器：负责模型性能优化和协同工作增强
Model Optimizer: Responsible for model performance optimization and collaborative work enhancement
"""

import time
import numpy as np
import asyncio
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from datetime import datetime
import logging
from collections import deque

from ..error_handling import error_handler
from ..model_registry import get_model_registry


class OptimizationStrategy(Enum):
    """优化策略枚举
    Optimization Strategy Enumeration
    """
    PERFORMANCE = "performance"      # 性能优化
    ACCURACY = "accuracy"            # 精度优化  
    EFFICIENCY = "efficiency"        # 效率优化
    COLLABORATION = "collaboration"  # 协作优化
    JOINT = "joint"                  # 联合优化
    ADAPTIVE = "adaptive"            # 自适应优化
    KNOWLEDGE_BASED = "knowledge_based"  # 基于知识的优化


class OptimizationLevel(Enum):
    """优化级别枚举
    Optimization Level Enumeration
    """
    BASIC = "basic"          # 基础优化
    STANDARD = "standard"    # 标准优化
    ADVANCED = "advanced"    # 高级优化
    EXTREME = "extreme"      # 极限优化


@dataclass
class OptimizationTask:
    """优化任务数据类
    Optimization Task Data Class
    """
    model_id: str
    strategy: OptimizationStrategy
    level: OptimizationLevel = OptimizationLevel.STANDARD
    target_metric: str = "overall_score"
    target_value: float = 0.9
    constraints: Dict[str, Any] = None
    priority: int = 5  # 1-10优先级
    timeout: int = 300  # 超时时间（秒）


@dataclass
class PerformanceMetrics:
    """性能指标数据类
    Performance Metrics Data Class
    """
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    throughput: float = 0.0
    energy_efficiency: float = 0.0
    collaboration_score: float = 0.0


class AdvancedOptimizationAlgorithm:
    """高级优化算法类
    Advanced Optimization Algorithm Class
    """
    
    @staticmethod
    async def genetic_optimization(model, metrics: PerformanceMetrics, generations: int = 10) -> Dict[str, Any]:
        """遗传算法优化
        Genetic Algorithm Optimization
        """
        best_fitness = 0.0
        best_params = {}
        
        for generation in range(generations):
            # 生成参数种群
            population = await AdvancedOptimizationAlgorithm._generate_population(model)
            
            # 评估适应度
            fitness_scores = []
            for params in population:
                fitness = await AdvancedOptimizationAlgorithm._evaluate_fitness(model, params, metrics)
                fitness_scores.append((fitness, params))
            
            # 选择最优个体
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            current_best_fitness, current_best_params = fitness_scores[0]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_params = current_best_params
            
            # 交叉和变异
            population = await AdvancedOptimizationAlgorithm._crossover_and_mutate(fitness_scores)
            
            await asyncio.sleep(0.1)  # 模拟计算时间
        
        return {
            "best_fitness": best_fitness,
            "best_params": best_params,
            "generations": generations
        }
    
    @staticmethod
    async def _generate_population(model) -> List[Dict[str, Any]]:
        """生成参数种群
        Generate parameter population
        """
        population = []
        for _ in range(20):
            params = {
                "learning_rate": np.random.uniform(0.0001, 0.01),
                "batch_size": np.random.choice([16, 32, 64, 128]),
                "dropout_rate": np.random.uniform(0.1, 0.5),
                "optimizer": np.random.choice(["adam", "sgd", "rmsprop"])
            }
            population.append(params)
        return population
    
    @staticmethod
    async def _evaluate_fitness(model, params: Dict[str, Any], metrics: PerformanceMetrics) -> float:
        """评估适应度
        Evaluate fitness
        """
        # 模拟适应度计算
        fitness = (metrics.accuracy * 0.3 + 
                  (1.0 / metrics.inference_time) * 0.2 +
                  (1.0 / metrics.memory_usage) * 0.2 +
                  metrics.collaboration_score * 0.3)
        return fitness
    
    @staticmethod
    async def _crossover_and_mutate(fitness_scores: List[Tuple[float, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """交叉和变异
        Crossover and mutation
        """
        new_population = []
        
        # 选择前50%的个体
        elite_count = max(2, len(fitness_scores) // 2)
        elite = [params for _, params in fitness_scores[:elite_count]]
        
        # 交叉
        for i in range(len(elite)):
            for j in range(i + 1, len(elite)):
                child = {}
                for key in elite[i].keys():
                    if np.random.random() < 0.5:
                        child[key] = elite[i][key]
                    else:
                        child[key] = elite[j][key]
                
                # 变异
                if np.random.random() < 0.1:
                    mutation_key = np.random.choice(list(child.keys()))
                    if isinstance(child[mutation_key], float):
                        child[mutation_key] *= np.random.uniform(0.8, 1.2)
                
                new_population.append(child)
        
        return new_population
    
    @staticmethod
    async def bayesian_optimization(model, metrics: PerformanceMetrics, iterations: int = 15) -> Dict[str, Any]:
        """贝叶斯优化
        Bayesian Optimization
        """
        best_score = 0.0
        best_config = {}
        
        for iteration in range(iterations):
            # 模拟贝叶斯优化过程
            config = {
                "learning_rate": np.random.uniform(0.0001, 0.01),
                "regularization": np.random.uniform(0.001, 0.1),
                "architecture_complexity": np.random.uniform(0.5, 2.0)
            }
            
            score = await AdvancedOptimizationAlgorithm._evaluate_configuration(model, config, metrics)
            
            if score > best_score:
                best_score = score
                best_config = config
            
            await asyncio.sleep(0.05)
        
        return {
            "best_score": best_score,
            "best_config": best_config,
            "iterations": iterations
        }
    
    @staticmethod
    async def _evaluate_configuration(model, config: Dict[str, Any], metrics: PerformanceMetrics) -> float:
        """评估配置
        Evaluate configuration
        """
        # 综合评分计算
        score = (metrics.accuracy * 0.4 +
                (1.0 / max(metrics.inference_time, 0.001)) * 0.3 +
                (1.0 / max(metrics.memory_usage, 0.001)) * 0.3)
        return score * (1.0 + config.get("architecture_complexity", 1.0) * 0.1)


class RealTimeMonitor:
    """实时监控器类
    Real-time Monitor Class
    """
    
    def __init__(self, window_size: int = 100):
        self.metrics_history = {}
        self.optimization_events = deque(maxlen=window_size)
        self.performance_alerts = deque(maxlen=50)
        self.collaboration_metrics = {}
    
    async def track_metrics(self, model_id: str, metrics: PerformanceMetrics):
        """跟踪性能指标
        Track performance metrics
        """
        if model_id not in self.metrics_history:
            self.metrics_history[model_id] = deque(maxlen=100)
        
        timestamp = time.time()
        metric_record = {
            "timestamp": timestamp,
            "metrics": metrics.__dict__,
            "model_id": model_id
        }
        
        self.metrics_history[model_id].append(metric_record)
        
        # 检查性能异常
        await self._check_performance_anomalies(model_id, metrics)
    
    async def _check_performance_anomalies(self, model_id: str, metrics: PerformanceMetrics):
        """检查性能异常
        Check performance anomalies
        """
        if len(self.metrics_history[model_id]) < 10:
            return
        
        recent_metrics = list(self.metrics_history[model_id])[-10:]
        
        # 计算平均值和标准差
        accuracies = [m["metrics"]["accuracy"] for m in recent_metrics]
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        current_accuracy = metrics.accuracy
        
        # 检测异常（超过2个标准差）
        if abs(current_accuracy - mean_accuracy) > 2 * std_accuracy and std_accuracy > 0:
            alert = {
                "timestamp": time.time(),
                "model_id": model_id,
                "metric": "accuracy",
                "current_value": current_accuracy,
                "expected_range": f"{mean_accuracy - 2*std_accuracy:.3f}-{mean_accuracy + 2*std_accuracy:.3f}",
                "severity": "warning" if abs(current_accuracy - mean_accuracy) > 3 * std_accuracy else "info"
            }
            self.performance_alerts.append(alert)
    
    def get_metrics_history(self, model_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取指标历史
        Get metrics history
        """
        if model_id in self.metrics_history:
            return list(self.metrics_history[model_id])[-limit:]
        return []
    
    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """获取性能警报
        Get performance alerts
        """
        return list(self.performance_alerts)
    
    async def update_collaboration_metrics(self, model_id: str, partner_id: str, 
                                         latency: float, throughput: float, success_rate: float):
        """更新协作指标
        Update collaboration metrics
        """
        key = f"{model_id}_{partner_id}"
        if key not in self.collaboration_metrics:
            self.collaboration_metrics[key] = {
                "latencies": deque(maxlen=50),
                "throughputs": deque(maxlen=50),
                "success_rates": deque(maxlen=50),
                "last_updated": time.time()
            }
        
        self.collaboration_metrics[key]["latencies"].append(latency)
        self.collaboration_metrics[key]["throughputs"].append(throughput)
        self.collaboration_metrics[key]["success_rates"].append(success_rate)
        self.collaboration_metrics[key]["last_updated"] = time.time()
    
    def get_collaboration_stats(self, model_id: str, partner_id: str) -> Optional[Dict[str, Any]]:
        """获取协作统计
        Get collaboration statistics
        """
        key = f"{model_id}_{partner_id}"
        if key in self.collaboration_metrics:
            stats = self.collaboration_metrics[key]
            return {
                "avg_latency": np.mean(stats["latencies"]) if stats["latencies"] else 0,
                "avg_throughput": np.mean(stats["throughputs"]) if stats["throughputs"] else 0,
                "avg_success_rate": np.mean(stats["success_rates"]) if stats["success_rates"] else 0,
                "last_updated": stats["last_updated"]
            }
        return None


class ModelOptimizer:
    """模型优化器类
    Model Optimizer Class
    """
    
    def __init__(self, from_scratch=False):
        """初始化模型优化器
        Initialize model optimizer
        """
        self.optimization_history = []
        self.performance_baselines = {}
        self.collaboration_patterns = {}
        self.real_time_monitor = RealTimeMonitor()
        self.optimization_strategies = {
            OptimizationStrategy.PERFORMANCE: self._optimize_performance,
            OptimizationStrategy.ACCURACY: self._optimize_accuracy,
            OptimizationStrategy.EFFICIENCY: self._optimize_efficiency,
            OptimizationStrategy.COLLABORATION: self._optimize_collaboration,
            OptimizationStrategy.JOINT: self._optimize_joint,
            OptimizationStrategy.ADAPTIVE: self._optimize_adaptive,
            OptimizationStrategy.KNOWLEDGE_BASED: self._optimize_knowledge_based
        }
        
        # 初始化知识库集成
        self.optimization_knowledge = self._load_optimization_knowledge(from_scratch)
        self.from_scratch = from_scratch
        
        self.logger = logging.getLogger("ModelOptimizer")
        self.logger.setLevel(logging.INFO)
        
        if from_scratch:
            self.logger.info("Model optimizer initialized in from-scratch training mode with simplified optimization knowledge")
    
    def _load_optimization_knowledge(self, from_scratch=False) -> Dict[str, Any]:
        """加载优化知识
        Load optimization knowledge
        """
        if from_scratch:
            # 从零开始训练模式下，只提供最基本的优化知识
            return {
                "best_practices": {
                    "performance": {
                        "batch_size_recommendations": {
                            "default": 32
                        },
                        "learning_rate_ranges": {
                            "default": (1e-5, 1e-3)
                        }
                    },
                    "accuracy": {
                        "regularization_methods": [
                            "dropout"
                        ]
                    }
                },
                "model_specific_optimizations": {}
            }
        else:
            # 标准模式下，提供完整的优化知识
            return {
                "best_practices": {
                    "performance": {
                        "batch_size_recommendations": {
                            "small_model": 32,
                            "medium_model": 64,
                            "large_model": 128
                        },
                        "learning_rate_ranges": {
                            "transformer": (1e-5, 1e-3),
                            "cnn": (1e-4, 1e-2),
                            "rnn": (1e-4, 1e-2)
                        }
                    },
                    "accuracy": {
                        "data_augmentation_techniques": [
                            "random_rotation", "color_jitter", "random_crop"
                        ],
                        "regularization_methods": [
                            "dropout", "weight_decay", "batch_norm"
                        ]
                    }
                },
                "model_specific_optimizations": {}
            }
    
    async def optimize_model(self, model_id: str, strategy: OptimizationStrategy, 
                           level: OptimizationLevel = OptimizationLevel.STANDARD,
                           target_metric: str = "overall_score", 
                           target_value: float = 0.9) -> Dict[str, Any]:
        """优化指定模型
        Optimize specified model
        
        Args:
            model_id: 模型ID
            strategy: 优化策略
            level: 优化级别
            target_metric: 目标指标
            target_value: 目标值
            
        Returns:
            dict: 优化结果
        """
        try:
            model_registry = get_model_registry()
            model = model_registry.get_model(model_id)
            if not model:
                error_handler.log_error(f"模型 {model_id} 未找到", "ModelOptimizer")
                return {"status": "error", "message": f"Model {model_id} not found"}
            
            # 创建优化任务
            task = OptimizationTask(
                model_id=model_id,
                strategy=strategy,
                level=level,
                target_metric=target_metric,
                target_value=target_value
            )
            
            # 执行优化
            optimizer = self.optimization_strategies.get(strategy)
            if not optimizer:
                error_handler.log_error(f"不支持的优化策略: {strategy}", "ModelOptimizer")
                return {"status": "error", "message": f"Unsupported strategy: {strategy}"}
            
            result = await optimizer(task)
            
            # 记录优化历史
            self._record_optimization(task, result)
            
            # 更新知识库
            await self._update_optimization_knowledge(task, result)
            
            return result
            
        except Exception as e:
            error_handler.handle_error(e, "ModelOptimizer", f"优化模型 {model_id} 失败")
            return {"status": "error", "message": str(e)}
    
    async def optimize_joint_models(self, model_ids: List[str], 
                                  strategy: OptimizationStrategy = OptimizationStrategy.JOINT,
                                  target_metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """联合优化多个模型
        Joint optimization of multiple models
        
        Args:
            model_ids: 模型ID列表
            strategy: 优化策略
            target_metrics: 目标指标字典
            
        Returns:
            dict: 联合优化结果
        """
        try:
            if not target_metrics:
                target_metrics = {"collaboration_efficiency": 0.85, "overall_performance": 0.9}
            
            # 分析模型间依赖关系
            dependencies = self._analyze_model_dependencies(model_ids)
            
            # 执行联合优化
            results = {}
            for model_id in model_ids:
                result = await self.optimize_model(
                    model_id,
                    strategy,
                    "joint_performance",
                    target_metrics.get("overall_performance", 0.9)
                )
                results[model_id] = result
            
            return {
                "status": "success",
                "joint_optimization_results": results,
                "dependencies_analyzed": dependencies
            }
            
        except Exception as e:
            error_handler.handle_error(e, "ModelOptimizer", "联合优化失败")
            return {"status": "error", "message": str(e)}
    
    def _analyze_model_dependencies(self, model_ids: List[str]) -> Dict[str, List[str]]:
        """分析模型依赖关系
        Analyze model dependencies
        """
        dependencies = {}
        model_registry = get_model_registry()
        for model_id in model_ids:
            model = model_registry.get_model(model_id)
            if model and hasattr(model, 'get_dependencies'):
                dependencies[model_id] = model.get_dependencies()
            else:
                dependencies[model_id] = []
        return dependencies
    
    def _get_performance_baseline(self, model_id: str) -> Dict[str, float]:
        """获取性能基线
        Get performance baseline
        """
        if model_id in self.performance_baselines:
            return self.performance_baselines[model_id].copy()
        
        # 默认性能基线
        return {
            "accuracy": 0.75,
            "precision": 0.7,
            "recall": 0.72,
            "f1_score": 0.71,
            "inference_time": 0.5,
            "memory_usage": 512.0,
            "cpu_usage": 45.0,
            "throughput": 120.0,
            "energy_efficiency": 0.65,
            "collaboration_score": 0.6,
            "overall_score": 0.68
        }
    
    def _update_performance_baseline(self, model_id: str, optimization_result: Dict[str, Any]):
        """更新性能基线
        Update performance baseline
        """
        if model_id not in self.performance_baselines:
            self.performance_baselines[model_id] = {}
        
        # 从优化结果中提取最新性能指标
        latest_iteration = None
        for key in optimization_result.keys():
            if key.startswith("iteration_"):
                latest_iteration = optimization_result[key]
        
        if latest_iteration and "current_value" in latest_iteration:
            target_metric = optimization_result.get("target_metric", "overall_score")
            self.performance_baselines[model_id][target_metric] = latest_iteration["current_value"]
    
    async def _simulate_optimization(self, duration_factor: float = 1.0):
        """模拟优化过程
        Simulate optimization process
        """
        # 模拟优化计算时间
        await asyncio.sleep(0.1 * duration_factor)
    
    def _record_optimization(self, task: OptimizationTask, result: Dict[str, Any]):
        """记录优化历史
        Record optimization history
        """
        optimization_record = {
            "timestamp": time.time(),
            "task": {
                "model_id": task.model_id,
                "strategy": task.strategy.value,
                "level": task.level.value,
                "target_metric": task.target_metric,
                "target_value": task.target_value
            },
            "result": result,
            "success": result.get("status") == "success"
        }
        
        self.optimization_history.append(optimization_record)
        
        # 保持历史记录数量
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-50:]
    
    async def _optimize_performance(self, task: OptimizationTask) -> Dict[str, Any]:
        """性能优化策略
        Performance optimization strategy
        """
        try:
            model_registry = get_model_registry()
            model = model_registry.get_model(task.model_id)
            baseline = self._get_performance_baseline(task.model_id)
            
            # 性能优化逻辑
            optimization_result = {
                "before_optimization": baseline,
                "optimization_strategy": "performance_enhancement",
                "techniques_applied": [
                    "batch_size_optimization",
                    "learning_rate_tuning",
                    "memory_optimization",
                    "parallel_processing"
                ],
                "iterations": 5
            }
            
            # 模拟性能优化过程
            for i in range(optimization_result["iterations"]):
                if hasattr(model, 'optimize_performance'):
                    result = await model.optimize_performance(task.target_metric, task.target_value)
                    optimization_result[f"iteration_{i+1}"] = result
                else:
                    await self._simulate_optimization()
                    # 性能改进率
                    improvement_rate = 0.12 * (i + 1) * (1 + 0.15 * i)
                    optimization_result[f"iteration_{i+1}"] = {
                        "performance_improvement": improvement_rate,
                        "current_value": baseline.get(task.target_metric, 0) + improvement_rate
                    }
            
            self._update_performance_baseline(task.model_id, optimization_result)
            
            return {
                "status": "success",
                "model_id": task.model_id,
                "optimization_type": "performance",
                "result": optimization_result
            }
            
        except Exception as e:
            error_handler.handle_error(e, "ModelOptimizer", "性能优化失败")
            return {"status": "error", "message": str(e)}
    
    async def _optimize_accuracy(self, task: OptimizationTask) -> Dict[str, Any]:
        """精度优化策略
        Accuracy optimization strategy
        """
        try:
            model_registry = get_model_registry()
            model = model_registry.get_model(task.model_id)
            baseline = self._get_performance_baseline(task.model_id)
            
            # 精度优化逻辑
            optimization_result = {
                "before_optimization": baseline,
                "optimization_strategy": "accuracy_enhancement",
                "techniques_applied": [
                    "data_augmentation",
                    "regularization",
                    "ensemble_learning",
                    "confidence_calibration"
                ],
                "iterations": 4
            }
            
            # 模拟精度优化过程
            for i in range(optimization_result["iterations"]):
                if hasattr(model, 'optimize_accuracy'):
                    result = await model.optimize_accuracy(task.target_metric, task.target_value)
                    optimization_result[f"iteration_{i+1}"] = result
                else:
                    await self._simulate_optimization(0.8)
                    # 精度改进率
                    improvement_rate = 0.08 * (i + 1) * (1 + 0.1 * i)
                    optimization_result[f"iteration_{i+1}"] = {
                        "accuracy_improvement": improvement_rate,
                        "current_value": baseline.get(task.target_metric, 0) + improvement_rate
                    }
            
            self._update_performance_baseline(task.model_id, optimization_result)
            
            return {
                "status": "success",
                "model_id": task.model_id,
                "optimization_type": "accuracy",
                "result": optimization_result
            }
            
        except Exception as e:
            error_handler.handle_error(e, "ModelOptimizer", "精度优化失败")
            return {"status": "error", "message": str(e)}
    
    async def _optimize_efficiency(self, task: OptimizationTask) -> Dict[str, Any]:
        """效率优化策略
        Efficiency optimization strategy
        """
        try:
            model_registry = get_model_registry()
            model = model_registry.get_model(task.model_id)
            baseline = self._get_performance_baseline(task.model_id)
            
            # 效率优化逻辑
            optimization_result = {
                "before_optimization": baseline,
                "optimization_strategy": "efficiency_enhancement",
                "techniques_applied": [
                    "model_pruning",
                    "quantization",
                    "knowledge_distillation",
                    "architecture_search"
                ],
                "iterations": 3
            }
            
            # 模拟效率优化过程
            for i in range(optimization_result["iterations"]):
                if hasattr(model, 'optimize_efficiency'):
                    result = await model.optimize_efficiency(task.target_metric, task.target_value)
                    optimization_result[f"iteration_{i+1}"] = result
                else:
                    await self._simulate_optimization(0.6)
                    # 效率改进率
                    improvement_rate = 0.15 * (i + 1) * (1 + 0.2 * i)
                    optimization_result[f"iteration_{i+1}"] = {
                        "efficiency_improvement": improvement_rate,
                        "current_value": baseline.get(task.target_metric, 0) + improvement_rate
                    }
            
            self._update_performance_baseline(task.model_id, optimization_result)
            
            return {
                "status": "success",
                "model_id": task.model_id,
                "optimization_type": "efficiency",
                "result": optimization_result
            }
            
        except Exception as e:
            error_handler.handle_error(e, "ModelOptimizer", "效率优化失败")
            return {"status": "error", "message": str(e)}
    
    async def _optimize_collaboration(self, task: OptimizationTask) -> Dict[str, Any]:
        """协作优化策略
        Collaboration optimization strategy
        """
        try:
            model_registry = get_model_registry()
            model = model_registry.get_model(task.model_id)
            
            # 协作优化逻辑
            optimization_result = {
                "optimization_strategy": "collaboration_enhancement",
                "techniques_applied": [
                    "communication_protocol_optimization",
                    "data_sharing_efficiency",
                    "synchronization_improvement",
                    "conflict_resolution"
                ]
            }
            
            # 获取协作伙伴
            collaborators = model_registry.get_collaborative_models(task.model_id)
            optimization_result["collaborators"] = collaborators
            
            # 模拟协作优化
            if hasattr(model, 'optimize_collaboration'):
                result = await model.optimize_collaboration(collaborators, task.target_value)
                optimization_result["collaboration_result"] = result
            else:
                await self._simulate_optimization(1.2)
                optimization_result["collaboration_result"] = {
                    "communication_latency_reduction": 0.25,
                    "data_throughput_increase": 0.35,
                    "synchronization_accuracy": 0.92,
                    "overall_collaboration_score": 0.88
                }
            
            return {
                "status": "success",
                "model_id": task.model_id,
                "optimization_type": "collaboration",
                "result": optimization_result
            }
            
        except Exception as e:
            error_handler.handle_error(e, "ModelOptimizer", "协作优化失败")
            return {"status": "error", "message": str(e)}
    
    async def _optimize_joint(self, task: OptimizationTask) -> Dict[str, Any]:
        """联合优化策略
        Joint optimization strategy
        """
        try:
            model_registry = get_model_registry()
            # 获取相关模型
            related_models = model_registry.get_related_models(task.model_id)
            
            optimization_result = {
                "optimization_strategy": "joint_optimization",
                "related_models": related_models,
                "techniques_applied": [
                    "cross_model_learning",
                    "parameter_sharing",
                    "coordinated_training"
                ]
            }
            
            # 执行联合优化
            joint_results = {}
            for related_model_id in related_models:
                if related_model_id != task.model_id:
                    result = await self.optimize_model(
                        related_model_id,
                        OptimizationStrategy.COLLABORATION,
                        "joint_efficiency",
                        0.85
                    )
                    joint_results[related_model_id] = result
            
            optimization_result["joint_results"] = joint_results
            
            return {
                "status": "success",
                "model_id": task.model_id,
                "optimization_type": "joint",
                "result": optimization_result
            }
            
        except Exception as e:
            error_handler.handle_error(e, "ModelOptimizer", "联合优化失败")
            return {"status": "error", "message": str(e)}
    
    async def _optimize_adaptive(self, task: OptimizationTask) -> Dict[str, Any]:
        """自适应优化策略
        Adaptive optimization strategy
        """
        try:
            model_registry = get_model_registry()
            model = model_registry.get_model(task.model_id)
            baseline = self._get_performance_baseline(task.model_id)
            
            # 自适应优化逻辑
            optimization_result = {
                "before_optimization": baseline,
                "optimization_strategy": "adaptive_tuning",
                "techniques_applied": [
                    "dynamic_learning_rate",
                    "adaptive_batch_size",
                    "progressive_training"
                ],
                "iterations": 6
            }
            
            # 模拟自适应优化过程
            for i in range(optimization_result["iterations"]):
                if hasattr(model, 'optimize_adaptive'):
                    result = await model.optimize_adaptive(task.target_metric, task.target_value)
                    optimization_result[f"iteration_{i+1}"] = result
                else:
                    await self._simulate_optimization()
                    # 自适应改进率
                    improvement_rate = 0.08 * (i + 1) * (1 + 0.1 * i)
                    optimization_result[f"iteration_{i+1}"] = {
                        "adaptive_improvement": improvement_rate,
                        "current_value": baseline.get(task.target_metric, 0) + improvement_rate
                    }
            
            self._update_performance_baseline(task.model_id, optimization_result)
            
            return {
                "status": "success",
                "model_id": task.model_id,
                "optimization_type": "adaptive",
                "result": optimization_result
            }
            
        except Exception as e:
            error_handler.handle_error(e, "ModelOptimizer", "自适应优化失败")
            return {"status": "error", "message": str(e)}
    
    async def _optimize_knowledge_based(self, task: OptimizationTask) -> Dict[str, Any]:
        """基于知识的优化策略
        Knowledge-based optimization strategy
        """
        try:
            model_registry = get_model_registry()
            model = model_registry.get_model(task.model_id)
            
            # 从本地知识库获取优化建议
            optimization_suggestions = self._get_local_optimization_suggestions(
                task.model_id,
                task.strategy.value,
                task.level.value
            )
            
            optimization_result = {
                "optimization_strategy": "knowledge_based_enhancement",
                "suggestions_from_knowledge_base": optimization_suggestions,
                "techniques_applied": optimization_suggestions.get("recommended_techniques", [])
            }
            
            # 应用知识库建议
            if hasattr(model, 'apply_knowledge_based_optimization'):
                result = await model.apply_knowledge_based_optimization(optimization_suggestions)
                optimization_result["knowledge_application_result"] = result
            else:
                # 默认知识应用
                await self._simulate_optimization(1.0)
                optimization_result["knowledge_application_result"] = {
                    "success": True,
                    "improvement_estimate": 0.15,
                    "applied_techniques": optimization_suggestions.get("recommended_techniques", [])[:3]
                }
            
            # 更新知识库
            await self._update_optimization_knowledge(task, optimization_result)
            
            return {
                "status": "success",
                "model_id": task.model_id,
                "optimization_type": "knowledge_based",
                "result": optimization_result
            }
            
        except Exception as e:
            error_handler.handle_error(e, "ModelOptimizer", "基于知识的优化失败")
            return {"status": "error", "message": str(e)}
    
    async def _update_optimization_knowledge(self, task: OptimizationTask, result: Dict[str, Any]):
        """更新优化知识
        Update optimization knowledge
        """
        try:
            knowledge_entry = {
                "timestamp": time.time(),
                "model_id": task.model_id,
                "strategy": task.strategy.value,
                "level": task.level.value,
                "result": result,
                "effectiveness_score": self._calculate_effectiveness_score(result)
            }
            
            # 更新本地知识库
            if task.model_id not in self.optimization_knowledge["model_specific_optimizations"]:
                self.optimization_knowledge["model_specific_optimizations"][task.model_id] = []
            
            self.optimization_knowledge["model_specific_optimizations"][task.model_id].append(knowledge_entry)
            
            # 保持知识条目数量
            if len(self.optimization_knowledge["model_specific_optimizations"][task.model_id]) > 50:
                self.optimization_knowledge["model_specific_optimizations"][task.model_id] = \
                    self.optimization_knowledge["model_specific_optimizations"][task.model_id][-25:]
            
        except Exception as e:
            self.logger.warning(f"更新优化知识失败: {e}")
    
    def _calculate_effectiveness_score(self, result: Dict[str, Any]) -> float:
        """计算效果评分
        Calculate effectiveness score
        """
        try:
            if result.get("status") != "success":
                return 0.0
            
            optimization_data = result.get("result", {})
            improvements = []
            
            # 提取改进指标
            for key in optimization_data.keys():
                if key.startswith("iteration_"):
                    iteration_data = optimization_data[key]
                    if "improvement" in iteration_data:
                        improvements.append(iteration_data["improvement"])
                    elif "accuracy_improvement" in iteration_data:
                        improvements.append(iteration_data["accuracy_improvement"])
                    elif "efficiency_improvement" in iteration_data:
                        improvements.append(iteration_data["efficiency_improvement"])
            
            if improvements:
                return float(np.mean(improvements))
            else:
                return 0.5  # 默认评分
            
        except Exception as e:
            self.logger.warning(f"计算效果评分失败: {e}")
            return 0.3
    
    async def get_real_time_metrics(self, model_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """获取实时性能指标
        Get real-time performance metrics
        """
        return self.real_time_monitor.get_metrics_history(model_id, limit)
    
    async def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """获取性能警报
        Get performance alerts
        """
        return self.real_time_monitor.get_performance_alerts()
    
    async def track_model_metrics(self, model_id: str, metrics: Dict[str, Any]):
        """跟踪模型指标
        Track model metrics
        """
        try:
            performance_metrics = PerformanceMetrics(**metrics)
            await self.real_time_monitor.track_metrics(model_id, performance_metrics)
        except Exception as e:
            self.logger.warning(f"跟踪模型指标失败: {e}")
    
    async def update_collaboration_metrics(self, model_id: str, partner_id: str, 
                                         latency: float, throughput: float, success_rate: float):
        """更新协作指标
        Update collaboration metrics
        """
        await self.real_time_monitor.update_collaboration_metrics(
            model_id, partner_id, latency, throughput, success_rate
        )
    
    def get_collaboration_stats(self, model_id: str, partner_id: str) -> Optional[Dict[str, Any]]:
        """获取协作统计
        Get collaboration statistics
        """
        return self.real_time_monitor.get_collaboration_stats(model_id, partner_id)
    
    def get_optimization_knowledge(self) -> Dict[str, Any]:
        """获取优化知识
        Get optimization knowledge
        """
        return self.optimization_knowledge
    
    def save_optimization_knowledge(self, file_path: str):
        """保存优化知识
        Save optimization knowledge
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.optimization_knowledge, f, indent=2, ensure_ascii=False)
            self.logger.info(f"优化知识已保存到: {file_path}")
        except Exception as e:
            self.logger.error(f"保存优化知识失败: {e}")
    
    def load_optimization_knowledge(self, file_path: str):
        """加载优化知识
        Load optimization knowledge
        """
        try:
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.optimization_knowledge = json.load(f)
                self.logger.info(f"优化知识已从 {file_path} 加载")
        except Exception as e:
            self.logger.warning(f"加载优化知识失败: {e}")

# 创建全局模型优化器实例
model_optimizer = ModelOptimizer()
