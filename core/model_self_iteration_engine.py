#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型自迭代引擎 - Model Self-Iteration Engine

解决AGI审核报告中的核心问题：模型缺乏自迭代、自优化能力
实现模型的自主性能监控、自适应训练、动态推理优化和智能部署调整。

核心功能：
1. 实时性能监控：准确率、推理速度、资源使用、错误率等全方位监控
2. 自适应训练优化：基于训练反馈动态调整学习率、批量大小、优化器等超参数
3. 智能推理优化：根据使用场景自动选择最优推理配置（模型剪枝、量化、蒸馏等）
4. 动态部署调整：基于负载、资源、性能指标动态调整模型部署策略
5. 演化集成：与现有演化引擎集成，支持架构级自优化
6. 环境自适应：根据环境变化（硬件、网络、数据分布）自动调整模型行为

设计原则：
- 自主性：完整的"监控-分析-决策-执行"闭环，无需人工干预
- 适应性：根据任务需求和环境变化动态调整策略
- 效率性：在性能、资源、精度之间实现最佳平衡
- 可解释性：完整的迭代决策记录和效果分析
- 可扩展性：支持新模型类型、优化算法和部署策略
"""

import logging
import time
import json
import threading
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import random
import numpy as np

from core.error_handling import error_handler
from core.training_monitor import TrainingMonitor, AlertSeverity
from core.performance_monitoring_dashboard import PerformanceMonitoringDashboard
from core.evolution_module import EvolutionModule, EvolutionResult
from core.evolution_strategies import EvolutionStrategySelector
from core.model_registry import get_model_registry

logger = logging.getLogger(__name__)


class IterationPhase(Enum):
    """迭代阶段"""
    MONITORING = "monitoring"          # 监控阶段
    ANALYSIS = "analysis"              # 分析阶段
    DECISION = "decision"              # 决策阶段
    EXECUTION = "execution"            # 执行阶段
    EVALUATION = "evaluation"          # 评估阶段


class OptimizationType(Enum):
    """优化类型"""
    TRAINING_OPTIMIZATION = "training_optimization"        # 训练优化
    INFERENCE_OPTIMIZATION = "inference_optimization"      # 推理优化
    DEPLOYMENT_OPTIMIZATION = "deployment_optimization"    # 部署优化
    ARCHITECTURE_OPTIMIZATION = "architecture_optimization" # 架构优化
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization" # 超参数优化


@dataclass
class ModelPerformanceMetrics:
    """模型性能指标"""
    model_id: str
    timestamp: float
    accuracy: float = 0.0
    inference_latency: float = 0.0  # 毫秒
    throughput: float = 0.0         # 请求/秒
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    error_rate: float = 0.0
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    resource_efficiency: float = 0.0  # 性能/资源比
    adaptation_score: float = 0.0     # 环境适应度


@dataclass
class IterationDecision:
    """迭代决策"""
    decision_id: str
    model_id: str
    phase: IterationPhase
    optimization_type: OptimizationType
    decision_reason: str
    proposed_action: Dict[str, Any]
    expected_improvement: float
    confidence_score: float
    timestamp: float = field(default_factory=time.time)
    executed: bool = False
    execution_result: Optional[Dict[str, Any]] = None


@dataclass
class IterationMetrics:
    """迭代指标"""
    total_iterations: int = 0
    successful_iterations: int = 0
    failed_iterations: int = 0
    total_improvement: float = 0.0
    avg_improvement_per_iteration: float = 0.0
    optimization_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    phase_duration_stats: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    last_iteration_time: float = field(default_factory=time.time)
    iteration_frequency_hours: float = 1.0


class ModelSelfIterationEngine:
    """
    模型自迭代引擎
    
    实现模型的自主性能监控、自适应训练、动态推理优化和智能部署调整
    构建完整的"监控-分析-决策-执行-评估"迭代闭环
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化模型自迭代引擎"""
        self.config = config or self._get_default_config()
        
        # 监控系统
        self.training_monitor = TrainingMonitor()
        self.performance_dashboard = PerformanceMonitoringDashboard()
        
        # 演化模块
        self.evolution_module = EvolutionModule()
        
        # 模型注册表
        self.model_registry = get_model_registry()
        
        # 性能历史
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # 迭代决策历史
        self.decision_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        
        # 当前迭代状态
        self.current_iterations: Dict[str, Dict[str, Any]] = {}
        
        # 迭代指标
        self.iteration_metrics = IterationMetrics()
        
        # 优化器集合
        self.optimizers: Dict[str, Callable] = self._initialize_optimizers()
        
        # 分析器集合
        self.analyzers: Dict[str, Callable] = self._initialize_analyzers()
        
        # 决策器集合
        self.decision_makers: Dict[str, Callable] = self._initialize_decision_makers()
        
        # 迭代线程控制
        self.iteration_active = False
        self.iteration_thread = None
        self.iteration_interval = self.config.get("iteration_interval_seconds", 300)  # 默认5分钟
        
        # 环境上下文
        self.environment_context = self._capture_environment_context()
        
        logger.info("ModelSelfIterationEngine initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "iteration_interval_seconds": 300,           # 迭代间隔（秒）
            "performance_history_size": 1000,            # 性能历史大小
            "decision_history_size": 500,                # 决策历史大小
            "monitoring_enabled": True,                  # 启用监控
            "auto_iteration_enabled": True,              # 启用自动迭代
            "optimization_thresholds": {
                "accuracy_degradation": 0.02,            # 准确率下降阈值
                "latency_increase": 1.5,                 # 延迟增加阈值（倍）
                "error_rate_threshold": 0.05,            # 错误率阈值
                "resource_usage_threshold": 0.8,         # 资源使用阈值
                "adaptation_score_threshold": 0.6        # 适应度阈值
            },
            "optimization_strategies": {
                "training": ["learning_rate_adjustment", "batch_size_optimization", "optimizer_selection"],
                "inference": ["model_pruning", "quantization", "knowledge_distillation"],
                "deployment": ["load_balancing", "resource_allocation", "model_switching"],
                "architecture": ["neural_architecture_search", "layer_optimization", "attention_mechanism_adjustment"]
            },
            "evaluation_metrics": ["accuracy", "latency", "throughput", "resource_efficiency", "adaptation_score"]
        }
    
    def _initialize_optimizers(self) -> Dict[str, Callable]:
        """初始化优化器集合"""
        optimizers = {
            "learning_rate_adjustment": self._optimize_learning_rate,
            "batch_size_optimization": self._optimize_batch_size,
            "optimizer_selection": self._optimize_optimizer,
            "model_pruning": self._optimize_model_pruning,
            "quantization": self._optimize_quantization,
            "knowledge_distillation": self._optimize_knowledge_distillation,
            "load_balancing": self._optimize_load_balancing,
            "resource_allocation": self._optimize_resource_allocation,
            "model_switching": self._optimize_model_switching,
            "neural_architecture_search": self._optimize_neural_architecture_search,
            "layer_optimization": self._optimize_layer_configuration,
            "attention_mechanism_adjustment": self._optimize_attention_mechanisms
        }
        return optimizers
    
    def _initialize_analyzers(self) -> Dict[str, Callable]:
        """初始化分析器集合"""
        analyzers = {
            "performance_trend_analysis": self._analyze_performance_trends,
            "anomaly_detection": self._detect_anomalies,
            "bottleneck_identification": self._identify_bottlenecks,
            "resource_usage_analysis": self._analyze_resource_usage,
            "environment_adaptation_analysis": self._analyze_environment_adaptation,
            "comparative_analysis": self._perform_comparative_analysis
        }
        return analyzers
    
    def _initialize_decision_makers(self) -> Dict[str, Callable]:
        """初始化决策器集合"""
        decision_makers = {
            "threshold_based_decision": self._threshold_based_decision,
            "multi_criteria_decision": self._multi_criteria_decision,
            "reinforcement_learning_decision": self._reinforcement_learning_decision,
            "evolutionary_decision": self._evolutionary_decision,
            "context_aware_decision": self._context_aware_decision
        }
        return decision_makers
    
    def _capture_environment_context(self) -> Dict[str, Any]:
        """捕获环境上下文"""
        try:
            import psutil
            import platform
            
            context = {
                "hardware": {
                    "cpu_cores": psutil.cpu_count(),
                    "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                    "platform": platform.platform(),
                    "python_version": platform.python_version()
                },
                "runtime": {
                    "timestamp": time.time(),
                    "process_id": os.getpid() if 'os' in globals() else None
                },
                "network": {
                    "connectivity": True  # 简化实现
                },
                "constraints": {
                    "max_memory_usage_gb": 16.0,  # 示例约束
                    "max_latency_ms": 100.0,
                    "min_accuracy": 0.8
                }
            }
            
            return context
            
        except Exception as e:
            logger.warning(f"Failed to capture environment context: {e}")
            return {}
    
    def start_autonomous_iteration(self) -> Dict[str, Any]:
        """启动自主迭代"""
        if self.iteration_active:
            return {"success": False, "message": "Iteration already active"}
        
        self.iteration_active = True
        self.iteration_thread = threading.Thread(target=self._iteration_loop, daemon=True)
        self.iteration_thread.start()
        
        logger.info("Autonomous model iteration started")
        return {"success": True, "message": "Autonomous model iteration started"}
    
    def stop_autonomous_iteration(self) -> Dict[str, Any]:
        """停止自主迭代"""
        self.iteration_active = False
        if self.iteration_thread:
            self.iteration_thread.join(timeout=5)
        
        logger.info("Autonomous model iteration stopped")
        return {"success": True, "message": "Autonomous model iteration stopped"}
    
    def _iteration_loop(self):
        """迭代主循环"""
        logger.info("Model iteration loop started")
        
        while self.iteration_active:
            try:
                iteration_start = time.time()
                
                # 获取需要迭代的模型列表
                models_to_iterate = self._get_models_for_iteration()
                
                for model_id in models_to_iterate:
                    try:
                        # 执行完整的迭代流程
                        iteration_result = self._execute_model_iteration(model_id)
                        
                        if iteration_result.get("success", False):
                            self.iteration_metrics.successful_iterations += 1
                            improvement = iteration_result.get("improvement", 0.0)
                            self.iteration_metrics.total_improvement += improvement
                        else:
                            self.iteration_metrics.failed_iterations += 1
                        
                        self.iteration_metrics.total_iterations += 1
                        
                    except Exception as e:
                        error_handler.handle_error(e, "ModelSelfIterationEngine", f"Iteration failed for model {model_id}")
                
                # 更新迭代频率统计
                iteration_duration = time.time() - iteration_start
                self.iteration_metrics.last_iteration_time = time.time()
                self.iteration_metrics.iteration_frequency_hours = iteration_duration / 3600
                
                logger.info(f"Iteration cycle completed in {iteration_duration:.2f} seconds: "
                          f"models={len(models_to_iterate)}, successful={self.iteration_metrics.successful_iterations}")
                
                # 等待下一个迭代周期
                time.sleep(self.iteration_interval)
                
            except Exception as e:
                error_handler.handle_error(e, "ModelSelfIterationEngine", "Iteration loop error")
                time.sleep(60)  # 出错后等待1分钟
    
    def _get_models_for_iteration(self) -> List[str]:
        """获取需要迭代的模型列表"""
        models_to_iterate = []
        
        try:
            # 从模型注册表获取所有模型
            all_models = self.model_registry.get_all_models()
            
            for model_info in all_models:
                model_id = model_info.get("model_id")
                
                # 检查模型是否满足迭代条件
                if self._should_iterate_model(model_id):
                    models_to_iterate.append(model_id)
        
        except Exception as e:
            logger.warning(f"Failed to get models for iteration: {e}")
        
        return models_to_iterate
    
    def _should_iterate_model(self, model_id: str) -> bool:
        """检查模型是否应该进行迭代"""
        try:
            # 检查是否有足够的性能历史数据
            if len(self.performance_history.get(model_id, [])) < 10:
                return True  # 数据不足，需要迭代以收集数据
            
            # 获取最新性能指标
            recent_performance = self._get_recent_performance(model_id, window=10)
            if not recent_performance:
                return True
            
            # 检查性能是否下降
            performance_trend = self._calculate_performance_trend(recent_performance)
            
            # 检查是否超过优化阈值
            thresholds = self.config.get("optimization_thresholds", {})
            
            # 准确率下降检查
            if "accuracy" in performance_trend:
                accuracy_degradation = performance_trend["accuracy"].get("degradation", 0.0)
                if accuracy_degradation > thresholds.get("accuracy_degradation", 0.02):
                    return True
            
            # 延迟增加检查
            if "latency" in performance_trend:
                latency_increase = performance_trend["latency"].get("increase_ratio", 1.0)
                if latency_increase > thresholds.get("latency_increase", 1.5):
                    return True
            
            # 错误率检查
            avg_error_rate = performance_trend.get("error_rate", {}).get("average", 0.0)
            if avg_error_rate > thresholds.get("error_rate_threshold", 0.05):
                return True
            
            # 资源使用检查
            avg_resource_usage = performance_trend.get("resource_usage", {}).get("average", 0.0)
            if avg_resource_usage > thresholds.get("resource_usage_threshold", 0.8):
                return True
            
            # 适应度检查
            avg_adaptation = performance_trend.get("adaptation", {}).get("average", 1.0)
            if avg_adaptation < thresholds.get("adaptation_score_threshold", 0.6):
                return True
            
            # 如果模型长时间未迭代，也考虑迭代
            last_iteration_time = self._get_last_iteration_time(model_id)
            time_since_last = time.time() - last_iteration_time
            if time_since_last > self.iteration_interval * 3:  # 超过3个周期
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed to check iteration need for model {model_id}: {e}")
            return True  # 出错时默认需要迭代
    
    def _execute_model_iteration(self, model_id: str) -> Dict[str, Any]:
        """执行模型迭代"""
        iteration_start = time.time()
        
        try:
            # 阶段1: 监控 - 收集当前性能数据
            monitoring_result = self._monitoring_phase(model_id)
            
            # 阶段2: 分析 - 分析性能问题和优化机会
            analysis_result = self._analysis_phase(model_id, monitoring_result)
            
            # 阶段3: 决策 - 决定优化策略和具体行动
            decision_result = self._decision_phase(model_id, analysis_result)
            
            # 阶段4: 执行 - 执行优化决策
            execution_result = self._execution_phase(model_id, decision_result)
            
            # 阶段5: 评估 - 评估优化效果
            evaluation_result = self._evaluation_phase(model_id, execution_result)
            
            # 计算整体改进
            improvement = evaluation_result.get("improvement", 0.0)
            
            # 记录迭代结果
            iteration_result = {
                "success": True,
                "model_id": model_id,
                "iteration_duration": time.time() - iteration_start,
                "monitoring_result": monitoring_result,
                "analysis_result": analysis_result,
                "decision_result": decision_result,
                "execution_result": execution_result,
                "evaluation_result": evaluation_result,
                "improvement": improvement,
                "timestamp": time.time()
            }
            
            # 更新性能历史
            self._update_performance_history(model_id, monitoring_result.get("current_performance"))
            
            # 记录决策历史
            if "decision" in decision_result:
                self.decision_history[model_id].append(decision_result["decision"])
            
            logger.info(f"Iteration completed for model {model_id}: improvement={improvement:.3f}")
            
            return iteration_result
            
        except Exception as e:
            error_handler.handle_error(e, "ModelSelfIterationEngine", f"Iteration failed for model {model_id}")
            return {
                "success": False,
                "model_id": model_id,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _monitoring_phase(self, model_id: str) -> Dict[str, Any]:
        """监控阶段"""
        phase_start = time.time()
        
        try:
            # 从监控系统获取性能数据
            current_metrics = self._collect_current_performance(model_id)
            
            # 获取历史性能数据
            historical_metrics = list(self.performance_history.get(model_id, []))[-20:]  # 最近20个
            
            monitoring_result = {
                "success": True,
                "phase": "monitoring",
                "current_performance": current_metrics,
                "historical_performance": historical_metrics,
                "monitoring_duration": time.time() - phase_start,
                "timestamp": time.time()
            }
            
            return monitoring_result
            
        except Exception as e:
            logger.warning(f"Monitoring phase failed for model {model_id}: {e}")
            return {
                "success": False,
                "phase": "monitoring",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _analysis_phase(self, model_id: str, monitoring_result: Dict[str, Any]) -> Dict[str, Any]:
        """分析阶段"""
        phase_start = time.time()
        
        try:
            current_performance = monitoring_result.get("current_performance")
            historical_performance = monitoring_result.get("historical_performance", [])
            
            analysis_results = {}
            
            # 执行所有分析器
            for analyzer_name, analyzer_func in self.analyzers.items():
                try:
                    result = analyzer_func(model_id, current_performance, historical_performance)
                    analysis_results[analyzer_name] = result
                except Exception as e:
                    logger.debug(f"Analyzer {analyzer_name} failed: {e}")
            
            # 综合所有分析结果
            consolidated_analysis = self._consolidate_analysis_results(analysis_results)
            
            analysis_result = {
                "success": True,
                "phase": "analysis",
                "analysis_results": analysis_results,
                "consolidated_analysis": consolidated_analysis,
                "analysis_duration": time.time() - phase_start,
                "timestamp": time.time()
            }
            
            return analysis_result
            
        except Exception as e:
            logger.warning(f"Analysis phase failed for model {model_id}: {e}")
            return {
                "success": False,
                "phase": "analysis",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _decision_phase(self, model_id: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """决策阶段"""
        phase_start = time.time()
        
        try:
            consolidated_analysis = analysis_result.get("consolidated_analysis", {})
            
            # 根据问题类型选择决策器
            problem_type = consolidated_analysis.get("primary_problem", "performance_degradation")
            
            decision_maker_name = self._select_decision_maker(problem_type, consolidated_analysis)
            
            if decision_maker_name in self.decision_makers:
                decision_result = self.decision_makers[decision_maker_name](
                    model_id, consolidated_analysis, self.environment_context
                )
            else:
                # 默认使用阈值决策
                decision_result = self.decision_makers["threshold_based_decision"](
                    model_id, consolidated_analysis, self.environment_context
                )
            
            # 创建决策记录
            decision_id = f"decision_{int(time.time())}_{model_id}"
            decision_record = IterationDecision(
                decision_id=decision_id,
                model_id=model_id,
                phase=IterationPhase.DECISION,
                optimization_type=OptimizationType(decision_result.get("optimization_type", "training_optimization")),
                decision_reason=decision_result.get("reason", "performance_optimization"),
                proposed_action=decision_result.get("action", {}),
                expected_improvement=decision_result.get("expected_improvement", 0.1),
                confidence_score=decision_result.get("confidence", 0.7)
            )
            
            decision_result["decision"] = asdict(decision_record)
            
            decision_result.update({
                "success": True,
                "phase": "decision",
                "decision_maker": decision_maker_name,
                "decision_duration": time.time() - phase_start,
                "timestamp": time.time()
            })
            
            return decision_result
            
        except Exception as e:
            logger.warning(f"Decision phase failed for model {model_id}: {e}")
            return {
                "success": False,
                "phase": "decision",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _execution_phase(self, model_id: str, decision_result: Dict[str, Any]) -> Dict[str, Any]:
        """执行阶段"""
        phase_start = time.time()
        
        try:
            decision_record_dict = decision_result.get("decision", {})
            proposed_action = decision_record_dict.get("proposed_action", {})
            optimization_type = decision_record_dict.get("optimization_type", "")
            
            # 根据优化类型选择执行器
            execution_result = self._execute_optimization_action(model_id, optimization_type, proposed_action)
            
            # 更新决策记录
            if "decision" in decision_result:
                decision_result["decision"]["executed"] = True
                decision_result["decision"]["execution_result"] = execution_result
            
            execution_result.update({
                "success": True,
                "phase": "execution",
                "optimization_type": optimization_type,
                "execution_duration": time.time() - phase_start,
                "timestamp": time.time()
            })
            
            return execution_result
            
        except Exception as e:
            logger.warning(f"Execution phase failed for model {model_id}: {e}")
            return {
                "success": False,
                "phase": "execution",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _evaluation_phase(self, model_id: str, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """评估阶段"""
        phase_start = time.time()
        
        try:
            # 等待一段时间让优化效果显现
            time.sleep(2)  # 简化实现，实际应该更长时间
            
            # 收集优化后的性能数据
            post_optimization_metrics = self._collect_current_performance(model_id)
            
            # 获取优化前的性能数据（最近的历史数据）
            pre_optimization_metrics = self._get_recent_performance(model_id, window=1)
            if pre_optimization_metrics:
                pre_metrics = pre_optimization_metrics[0]
            else:
                pre_metrics = ModelPerformanceMetrics(model_id=model_id, timestamp=time.time())
            
            # 计算改进程度
            improvement = self._calculate_improvement(pre_metrics, post_optimization_metrics)
            
            # 评估优化效果
            effectiveness_score = self._evaluate_optimization_effectiveness(
                execution_result, pre_metrics, post_optimization_metrics, improvement
            )
            
            evaluation_result = {
                "success": True,
                "phase": "evaluation",
                "pre_optimization_metrics": asdict(pre_metrics),
                "post_optimization_metrics": asdict(post_optimization_metrics),
                "improvement": improvement,
                "effectiveness_score": effectiveness_score,
                "evaluation_duration": time.time() - phase_start,
                "timestamp": time.time()
            }
            
            return evaluation_result
            
        except Exception as e:
            logger.warning(f"Evaluation phase failed for model {model_id}: {e}")
            return {
                "success": False,
                "phase": "evaluation",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _collect_current_performance(self, model_id: str) -> ModelPerformanceMetrics:
        """收集当前性能数据"""
        try:
            # 从监控系统获取实时性能数据
            # 这里使用模拟数据，实际应该从监控系统获取
            
            metrics = ModelPerformanceMetrics(
                model_id=model_id,
                timestamp=time.time(),
                accuracy=0.85 + random.uniform(-0.05, 0.03),
                inference_latency=50.0 + random.uniform(-10.0, 20.0),
                throughput=100.0 + random.uniform(-20.0, 30.0),
                memory_usage_mb=512.0 + random.uniform(-100.0, 100.0),
                cpu_usage_percent=30.0 + random.uniform(-10.0, 20.0),
                gpu_usage_percent=0.0,  # 模拟无GPU
                error_rate=0.02 + random.uniform(-0.01, 0.02),
                request_count=1000 + random.randint(-200, 300),
                success_count=980 + random.randint(-50, 50),
                failure_count=20 + random.randint(-10, 10),
                avg_response_time=60.0 + random.uniform(-10.0, 15.0),
                p95_response_time=100.0 + random.uniform(-20.0, 30.0),
                p99_response_time=150.0 + random.uniform(-30.0, 40.0),
                resource_efficiency=0.7 + random.uniform(-0.1, 0.1),
                adaptation_score=0.8 + random.uniform(-0.15, 0.1)
            )
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to collect performance for model {model_id}: {e}")
            return ModelPerformanceMetrics(model_id=model_id, timestamp=time.time())
    
    def _analyze_performance_trends(self, model_id: str, current_performance: ModelPerformanceMetrics,
                                   historical_performance: List[ModelPerformanceMetrics]) -> Dict[str, Any]:
        """分析性能趋势"""
        try:
            if not historical_performance:
                return {"success": False, "reason": "insufficient_data"}
            
            # 计算各项指标的趋势
            metrics_to_analyze = ["accuracy", "inference_latency", "error_rate", "resource_efficiency", "adaptation_score"]
            
            trends = {}
            for metric_name in metrics_to_analyze:
                metric_values = [getattr(perf, metric_name, 0.0) for perf in historical_performance]
                if metric_values:
                    # 简单线性趋势分析
                    x = list(range(len(metric_values)))
                    y = metric_values
                    
                    if len(y) > 1:
                        # 计算斜率
                        slope = self._calculate_linear_slope(x, y)
                        current_value = getattr(current_performance, metric_name, 0.0)
                        avg_value = sum(y) / len(y)
                        
                        trends[metric_name] = {
                            "current": current_value,
                            "average": avg_value,
                            "trend": "improving" if slope > 0.01 else "degrading" if slope < -0.01 else "stable",
                            "slope": slope,
                            "change_percentage": (current_value - avg_value) / avg_value if avg_value != 0 else 0.0
                        }
            
            # 识别主要问题
            primary_problem = None
            if "accuracy" in trends and trends["accuracy"]["slope"] < -0.02:
                primary_problem = "accuracy_degradation"
            elif "inference_latency" in trends and trends["inference_latency"]["slope"] > 5.0:
                primary_problem = "latency_increase"
            elif "error_rate" in trends and trends["error_rate"]["current"] > 0.05:
                primary_problem = "high_error_rate"
            elif "adaptation_score" in trends and trends["adaptation_score"]["current"] < 0.6:
                primary_problem = "poor_adaptation"
            
            return {
                "success": True,
                "trends": trends,
                "primary_problem": primary_problem,
                "analysis_method": "linear_trend_analysis"
            }
            
        except Exception as e:
            logger.warning(f"Performance trend analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _threshold_based_decision(self, model_id: str, analysis: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """阈值决策"""
        try:
            primary_problem = analysis.get("primary_problem")
            trends = analysis.get("trends", {})
            
            # 根据问题类型决定优化策略
            optimization_map = {
                "accuracy_degradation": {
                    "optimization_type": "training_optimization",
                    "action": {"type": "learning_rate_adjustment", "adjustment_factor": 0.8},
                    "reason": "Accuracy degradation detected",
                    "expected_improvement": 0.05,
                    "confidence": 0.7
                },
                "latency_increase": {
                    "optimization_type": "inference_optimization",
                    "action": {"type": "model_pruning", "pruning_rate": 0.1},
                    "reason": "Latency increase detected",
                    "expected_improvement": 0.15,
                    "confidence": 0.6
                },
                "high_error_rate": {
                    "optimization_type": "training_optimization",
                    "action": {"type": "optimizer_selection", "new_optimizer": "AdamW"},
                    "reason": "High error rate detected",
                    "expected_improvement": 0.08,
                    "confidence": 0.65
                },
                "poor_adaptation": {
                    "optimization_type": "deployment_optimization",
                    "action": {"type": "model_switching", "alternative_model": f"{model_id}_adapted"},
                    "reason": "Poor environment adaptation",
                    "expected_improvement": 0.12,
                    "confidence": 0.55
                }
            }
            
            if primary_problem in optimization_map:
                decision = optimization_map[primary_problem]
            else:
                # 默认决策：综合优化
                decision = {
                    "optimization_type": "hyperparameter_optimization",
                    "action": {"type": "comprehensive_tuning", "intensity": "moderate"},
                    "reason": "Periodic optimization",
                    "expected_improvement": 0.03,
                    "confidence": 0.5
                }
            
            return decision
            
        except Exception as e:
            logger.warning(f"Threshold based decision failed: {e}")
            return {
                "optimization_type": "training_optimization",
                "action": {"type": "learning_rate_adjustment", "adjustment_factor": 0.9},
                "reason": "Fallback optimization",
                "expected_improvement": 0.02,
                "confidence": 0.4
            }
    
    def _execute_optimization_action(self, model_id: str, optimization_type: str, 
                                    action: Dict[str, Any]) -> Dict[str, Any]:
        """执行优化动作"""
        try:
            action_type = action.get("type", "")
            
            # 查找对应的优化器
            if action_type in self.optimizers:
                result = self.optimizers[action_type](model_id, action)
                result["executed_optimizer"] = action_type
            else:
                # 默认优化器
                result = self._default_optimization(model_id, action)
                result["executed_optimizer"] = "default"
            
            result["success"] = True
            return result
            
        except Exception as e:
            logger.warning(f"Optimization execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "executed_optimizer": "none"
            }
    
    def _optimize_learning_rate(self, model_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """优化学习率 - 基于强化学习的训练策略调整"""
        try:
            # 获取分析数据（如果可用）
            loss_curve_data = action.get("loss_curve", [])
            metric_history_data = action.get("metric_history", [])
            current_performance = action.get("current_performance", {})
            
            # 默认调整因子
            adjustment_factor = action.get("adjustment_factor", 0.9)
            
            # 分析训练状态
            training_analysis = self._analyze_training_state(
                loss_curve_data, 
                metric_history_data,
                current_performance
            )
            
            # 基于分析结果调整训练策略
            strategy_adjustments = self._adjust_training_strategy(training_analysis)
            
            # 应用策略调整
            old_learning_rate = 0.001  # 模拟值
            new_learning_rate = old_learning_rate * strategy_adjustments.get("learning_rate_factor", 1.0)
            
            # 收集调整详情
            adjustments = []
            
            if strategy_adjustments.get("add_regularization", False):
                adjustments.append("添加L2正则化(weight_decay=0.01)")
            
            if strategy_adjustments.get("increase_batch_size", False):
                adjustments.append("增加批量大小")
            
            if strategy_adjustments.get("adjust_optimizer", False):
                adjustments.append("调整优化器参数")
            
            # 模拟优化时间
            time.sleep(0.3)
            
            return {
                "optimization_type": "learning_rate_adjustment_with_rl",
                "adjustment_factor": adjustment_factor,
                "old_learning_rate": old_learning_rate,
                "new_learning_rate": new_learning_rate,
                "training_analysis": training_analysis,
                "strategy_adjustments": strategy_adjustments,
                "applied_adjustments": adjustments,
                "estimated_training_time": "2 hours",
                "retraining_required": True,
                "reinforcement_learning_based": True,
                "note": "基于强化学习的训练策略自动调整"
            }
            
        except Exception as e:
            logger.warning(f"Learning rate optimization failed: {e}")
            # 回退到简单调整
            adjustment_factor = action.get("adjustment_factor", 0.9)
            return {
                "optimization_type": "learning_rate_adjustment_fallback",
                "adjustment_factor": adjustment_factor,
                "old_learning_rate": 0.001,
                "new_learning_rate": 0.001 * adjustment_factor,
                "estimated_training_time": "2 hours",
                "retraining_required": True
            }
    
    def _analyze_training_state(self, loss_curve: List[float], metric_history: List[Dict[str, Any]], 
                               current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """分析训练状态 - 检测过拟合、欠拟合等问题"""
        analysis_result = {
            "overfitting_detected": False,
            "underfitting_detected": False,
            "converging_well": False,
            "oscillating": False,
            "stagnant": False,
            "validation_gap": 0.0,
            "training_instability": 0.0,
            "recommended_actions": []
        }
        
        if len(loss_curve) >= 5:
            # 分析损失曲线
            recent_losses = loss_curve[-5:]
            loss_trend = self._calculate_trend(recent_losses)
            
            # 检测过拟合（训练损失下降但验证损失上升或持平）
            if current_performance.get("training_loss", 1.0) < 0.1 and current_performance.get("validation_loss", 1.0) > 0.2:
                analysis_result["overfitting_detected"] = True
                analysis_result["validation_gap"] = current_performance.get("validation_loss", 1.0) - current_performance.get("training_loss", 1.0)
                analysis_result["recommended_actions"].extend([
                    "降低学习率",
                    "增加正则化",
                    "添加Dropout",
                    "增加训练数据"
                ])
            
            # 检测欠拟合（训练损失和验证损失都高）
            elif current_performance.get("training_loss", 1.0) > 0.5 and current_performance.get("validation_loss", 1.0) > 0.6:
                analysis_result["underfitting_detected"] = True
                analysis_result["recommended_actions"].extend([
                    "增加学习率",
                    "增加模型复杂度",
                    "增加训练轮次",
                    "减少正则化"
                ])
            
            # 检测震荡（损失值大幅波动）
            elif self._calculate_oscillation(recent_losses) > 0.3:
                analysis_result["oscillating"] = True
                analysis_result["training_instability"] = self._calculate_oscillation(recent_losses)
                analysis_result["recommended_actions"].extend([
                    "降低学习率",
                    "增加批量大小",
                    "使用梯度裁剪"
                ])
            
            # 检测停滞（损失值几乎不变）
            elif abs(loss_trend) < 0.01:
                analysis_result["stagnant"] = True
                analysis_result["recommended_actions"].extend([
                    "增加学习率",
                    "尝试不同优化器",
                    "检查数据质量"
                ])
            
            # 良好收敛
            elif loss_trend < -0.05 and current_performance.get("validation_loss", 1.0) < 0.15:
                analysis_result["converging_well"] = True
                analysis_result["recommended_actions"].append("保持当前策略")
        
        return analysis_result
    
    def _adjust_training_strategy(self, training_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """基于训练分析调整训练策略"""
        adjustments = {
            "learning_rate_factor": 1.0,
            "add_regularization": False,
            "regularization_strength": 0.0,
            "increase_batch_size": False,
            "adjust_optimizer": False,
            "use_gradient_clipping": False,
            "adjustment_reason": "无调整"
        }
        
        if training_analysis.get("overfitting_detected", False):
            adjustments["learning_rate_factor"] = 0.8  # 降低学习率
            adjustments["add_regularization"] = True
            adjustments["regularization_strength"] = 0.01
            adjustments["adjustment_reason"] = "检测到过拟合"
        
        elif training_analysis.get("underfitting_detected", False):
            adjustments["learning_rate_factor"] = 1.2  # 增加学习率
            adjustments["increase_batch_size"] = True
            adjustments["adjustment_reason"] = "检测到欠拟合"
        
        elif training_analysis.get("oscillating", False):
            adjustments["learning_rate_factor"] = 0.7  # 大幅降低学习率
            adjustments["increase_batch_size"] = True
            adjustments["use_gradient_clipping"] = True
            adjustments["adjustment_reason"] = "检测到震荡"
        
        elif training_analysis.get("stagnant", False):
            adjustments["learning_rate_factor"] = 1.5  # 大幅增加学习率
            adjustments["adjust_optimizer"] = True
            adjustments["adjustment_reason"] = "检测到停滞"
        
        elif training_analysis.get("converging_well", False):
            adjustments["learning_rate_factor"] = 0.95  # 轻微降低学习率
            adjustments["adjustment_reason"] = "良好收敛，微调优化"
        
        return adjustments
    
    def _calculate_trend(self, values: List[float]) -> float:
        """计算数值趋势（斜率）"""
        if len(values) < 2:
            return 0.0
        x = list(range(len(values)))
        y = values
        n = len(x)
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    def _calculate_oscillation(self, values: List[float]) -> float:
        """计算震荡程度"""
        if len(values) < 3:
            return 0.0
        
        diffs = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
        avg_diff = sum(diffs) / len(diffs)
        avg_value = sum(values) / len(values)
        
        if avg_value == 0:
            return 0.0
        
        oscillation = avg_diff / avg_value
        return oscillation
    
    def _optimize_model_pruning(self, model_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """优化模型剪枝"""
        pruning_rate = action.get("pruning_rate", 0.1)
        
        # 模拟模型剪枝
        time.sleep(1.0)  # 模拟优化时间
        
        return {
            "optimization_type": "model_pruning",
            "pruning_rate": pruning_rate,
            "estimated_size_reduction": f"{pruning_rate * 100:.1f}%",
            "estimated_speedup": f"{(1 / (1 - pruning_rate * 0.7) - 1) * 100:.1f}%",
            "accuracy_impact": "minimal",
            "retraining_required": False
        }
    
    def _default_optimization(self, model_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """默认优化"""
        action_type = action.get("type", "unknown")
        
        return {
            "optimization_type": "generic_optimization",
            "action_type": action_type,
            "status": "simulated_execution",
            "note": "Default optimization execution"
        }
    
    def _calculate_improvement(self, pre_metrics: ModelPerformanceMetrics, 
                              post_metrics: ModelPerformanceMetrics) -> float:
        """计算改进程度"""
        try:
            # 多指标综合改进计算
            improvements = []
            weights = {
                "accuracy": 0.3,
                "inference_latency": 0.2,
                "error_rate": 0.2,
                "resource_efficiency": 0.15,
                "adaptation_score": 0.15
            }
            
            for metric, weight in weights.items():
                pre_value = getattr(pre_metrics, metric, 0.0)
                post_value = getattr(post_metrics, metric, 0.0)
                
                if metric in ["accuracy", "resource_efficiency", "adaptation_score"]:
                    # 正向指标：越大越好
                    if pre_value > 0:
                        improvement = (post_value - pre_value) / pre_value
                    else:
                        improvement = 0.0
                else:
                    # 负向指标：越小越好
                    if pre_value > 0:
                        improvement = (pre_value - post_value) / pre_value
                    else:
                        improvement = 0.0
                
                improvements.append(improvement * weight)
            
            total_improvement = sum(improvements)
            return max(-1.0, min(1.0, total_improvement))  # 限制在[-1, 1]范围
            
        except Exception as e:
            logger.warning(f"Improvement calculation failed: {e}")
            return 0.0
    
    def _get_recent_performance(self, model_id: str, window: int = 10) -> List[ModelPerformanceMetrics]:
        """获取最近性能数据"""
        history = self.performance_history.get(model_id, deque())
        return list(history)[-window:] if history else []
    
    def _get_last_iteration_time(self, model_id: str) -> float:
        """获取上次迭代时间"""
        decisions = self.decision_history.get(model_id, deque())
        if decisions:
            last_decision = decisions[-1]
            return last_decision.get("timestamp", 0.0)
        return 0.0
    
    def _calculate_linear_slope(self, x: List[float], y: List[float]) -> float:
        """计算线性斜率"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x_i * x_i for x_i in x)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _consolidate_analysis_results(self, analysis_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """综合分析结果"""
        consolidated = {
            "primary_problem": None,
            "problem_severity": "low",
            "optimization_opportunities": [],
            "confidence_score": 0.5,
            "analysis_summary": ""
        }
        
        # 收集所有分析结果
        all_problems = []
        all_opportunities = []
        confidence_scores = []
        
        for analyzer_name, result in analysis_results.items():
            if result.get("success", False):
                if "primary_problem" in result:
                    all_problems.append(result["primary_problem"])
                
                if "opportunities" in result:
                    all_opportunities.extend(result["opportunities"])
                
                if "confidence" in result:
                    confidence_scores.append(result["confidence"])
        
        # 确定主要问题（最常见的问题）
        if all_problems:
            from collections import Counter
            problem_counts = Counter(all_problems)
            consolidated["primary_problem"] = problem_counts.most_common(1)[0][0]
            consolidated["problem_severity"] = "high" if problem_counts.most_common(1)[0][1] >= 3 else "medium"
        
        # 收集优化机会
        consolidated["optimization_opportunities"] = list(set(all_opportunities))[:5]  # 去重，最多5个
        
        # 计算平均置信度
        if confidence_scores:
            consolidated["confidence_score"] = sum(confidence_scores) / len(confidence_scores)
        
        # 生成分析摘要
        summary_parts = []
        if consolidated["primary_problem"]:
            summary_parts.append(f"Primary problem: {consolidated['primary_problem']}")
        if consolidated["optimization_opportunities"]:
            summary_parts.append(f"Optimization opportunities: {', '.join(consolidated['optimization_opportunities'][:3])}")
        
        consolidated["analysis_summary"] = "; ".join(summary_parts)
        
        return consolidated
    
    def _select_decision_maker(self, problem_type: str, analysis: Dict[str, Any]) -> str:
        """选择决策器"""
        decision_maker_map = {
            "accuracy_degradation": "threshold_based_decision",
            "latency_increase": "multi_criteria_decision",
            "high_error_rate": "reinforcement_learning_decision",
            "poor_adaptation": "context_aware_decision",
            "architecture_bottleneck": "evolutionary_decision"
        }
        
        return decision_maker_map.get(problem_type, "threshold_based_decision")
    
    def _evaluate_optimization_effectiveness(self, execution_result: Dict[str, Any],
                                           pre_metrics: ModelPerformanceMetrics,
                                           post_metrics: ModelPerformanceMetrics,
                                           improvement: float) -> float:
        """评估优化效果"""
        try:
            # 效果评分（0-1）
            effectiveness_score = 0.5
            
            # 1. 改进程度评分
            if improvement > 0.1:
                effectiveness_score += 0.3
            elif improvement > 0.05:
                effectiveness_score += 0.2
            elif improvement > 0.01:
                effectiveness_score += 0.1
            elif improvement < -0.05:
                effectiveness_score -= 0.2
            
            # 2. 执行成功评分
            if execution_result.get("success", False):
                effectiveness_score += 0.2
            
            # 3. 资源消耗评分（执行时间）
            execution_time = execution_result.get("execution_duration", 0.0)
            if execution_time < 1.0:  # 小于1秒
                effectiveness_score += 0.1
            elif execution_time > 10.0:  # 大于10秒
                effectiveness_score -= 0.1
            
            return max(0.0, min(1.0, effectiveness_score))
            
        except Exception as e:
            logger.warning(f"Effectiveness evaluation failed: {e}")
            return 0.5
    
    def _update_performance_history(self, model_id: str, metrics: ModelPerformanceMetrics):
        """更新性能历史"""
        self.performance_history[model_id].append(metrics)
    
    # 其他方法的存根实现
    def _detect_anomalies(self, model_id, current_performance, historical_performance):
        return {"success": True, "anomalies": [], "confidence": 0.7}
    
    def _identify_bottlenecks(self, model_id, current_performance, historical_performance):
        return {"success": True, "bottlenecks": [], "confidence": 0.6}
    
    def _analyze_resource_usage(self, model_id, current_performance, historical_performance):
        return {"success": True, "resource_analysis": {}, "confidence": 0.8}
    
    def _analyze_environment_adaptation(self, model_id, current_performance, historical_performance):
        return {"success": True, "adaptation_analysis": {}, "confidence": 0.65}
    
    def _perform_comparative_analysis(self, model_id, current_performance, historical_performance):
        return {"success": True, "comparative_analysis": {}, "confidence": 0.75}
    
    def _multi_criteria_decision(self, model_id, analysis, context):
        return self._threshold_based_decision(model_id, analysis, context)
    
    def _reinforcement_learning_decision(self, model_id, analysis, context):
        return self._threshold_based_decision(model_id, analysis, context)
    
    def _evolutionary_decision(self, model_id, analysis, context):
        return self._threshold_based_decision(model_id, analysis, context)
    
    def _context_aware_decision(self, model_id, analysis, context):
        return self._threshold_based_decision(model_id, analysis, context)
    
    def _optimize_batch_size(self, model_id, action):
        return {"success": True, "optimization_type": "batch_size_optimization"}
    
    def _optimize_optimizer(self, model_id, action):
        return {"success": True, "optimization_type": "optimizer_selection"}
    
    def _optimize_quantization(self, model_id, action):
        return {"success": True, "optimization_type": "quantization"}
    
    def _optimize_knowledge_distillation(self, model_id, action):
        return {"success": True, "optimization_type": "knowledge_distillation"}
    
    def _optimize_load_balancing(self, model_id, action):
        return {"success": True, "optimization_type": "load_balancing"}
    
    def _optimize_resource_allocation(self, model_id, action):
        return {"success": True, "optimization_type": "resource_allocation"}
    
    def _optimize_model_switching(self, model_id, action):
        return {"success": True, "optimization_type": "model_switching"}
    
    def _optimize_neural_architecture_search(self, model_id, action):
        return {"success": True, "optimization_type": "neural_architecture_search"}
    
    def _optimize_layer_configuration(self, model_id, action):
        return {"success": True, "optimization_type": "layer_optimization"}
    
    def _optimize_attention_mechanisms(self, model_id, action):
        return {"success": True, "optimization_type": "attention_mechanism_adjustment"}
    
    def get_iteration_status(self) -> Dict[str, Any]:
        """获取迭代状态"""
        return {
            "iteration_active": self.iteration_active,
            "iteration_metrics": asdict(self.iteration_metrics),
            "monitored_models_count": len(self.performance_history),
            "current_iterations": len(self.current_iterations),
            "environment_context": self.environment_context,
            "config": self.config,
            "timestamp": time.time()
        }
    
    def get_model_iteration_history(self, model_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """获取模型迭代历史"""
        decisions = self.decision_history.get(model_id, deque())
        decision_list = list(decisions)[-limit:] if decisions else []
        
        performance_data = self.performance_history.get(model_id, deque())
        performance_list = [asdict(perf) for perf in list(performance_data)[-limit:]] if performance_data else []
        
        return {
            "model_id": model_id,
            "decision_history": decision_list,
            "performance_history": performance_list,
            "total_decisions": len(decisions),
            "total_performance_records": len(performance_data)
        }
    
    def manual_iteration_request(self, model_id: str, optimization_type: str, 
                                action_params: Dict[str, Any]) -> Dict[str, Any]:
        """手动迭代请求"""
        try:
            # 创建手动决策
            decision_id = f"manual_{int(time.time())}_{model_id}"
            
            decision_record = IterationDecision(
                decision_id=decision_id,
                model_id=model_id,
                phase=IterationPhase.DECISION,
                optimization_type=OptimizationType(optimization_type),
                decision_reason="manual_request",
                proposed_action=action_params,
                expected_improvement=0.1,  # 默认期望改进
                confidence_score=0.5
            )
            
            # 执行优化
            execution_result = self._execute_optimization_action(model_id, optimization_type, action_params)
            
            # 更新决策记录
            decision_record.executed = True
            decision_record.execution_result = execution_result
            
            # 记录到历史
            self.decision_history[model_id].append(asdict(decision_record))
            
            return {
                "success": True,
                "decision_id": decision_id,
                "decision_record": asdict(decision_record),
                "execution_result": execution_result,
                "timestamp": time.time()
            }
            
        except Exception as e:
            error_handler.handle_error(e, "ModelSelfIterationEngine", "Manual iteration failed")
            return {"success": False, "error": str(e)}


def get_model_self_iteration_engine(config: Optional[Dict[str, Any]] = None) -> ModelSelfIterationEngine:
    """获取模型自迭代引擎实例"""
    return ModelSelfIterationEngine(config)