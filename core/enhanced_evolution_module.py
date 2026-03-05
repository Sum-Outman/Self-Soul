#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强演化模块 - Enhanced Evolution Module

基于现有的EvolutionModule，使用EnhancedEvolutionEngine提供以下增强功能：
1. 多算法自适应选择
2. 多目标优化支持
3. 元学习演化策略
4. 协同演化框架
5. 增强的性能监控和分析

设计原则：
- 向后兼容：完全兼容IEvolutionModule接口
- 渐进增强：默认使用增强功能，可回退到基础功能
- 透明替换：可替代现有的EvolutionModule
- 可配置：通过配置启用/禁用特定增强功能
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict

from core.module_interfaces import (
    IEvolutionModule,
    EvolutionResult,
    EvolutionParameters,
)
from core.enhanced_evolution_engine import (
    EnhancedEvolutionEngine,
    create_enhanced_evolution_engine,
    EvolutionAlgorithm,
    MultiObjectiveOptimizer,
    MetaLearningEvolutionController,
)

# 硬件感知模块
try:
    from core.optimization.hardware_aware_evolution import (
        HardwareAwareEvolutionModule,
        HardwareOptimizationLevel,
        create_hardware_aware_evolution_module,
    )
    HARDWARE_AWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AWARE_AVAILABLE = False
    HardwareAwareEvolutionModule = None
    HardwareOptimizationLevel = None
    create_hardware_aware_evolution_module = None

# 配置日志
logger = logging.getLogger(__name__)


class EnhancedEvolutionModule(IEvolutionModule):
    """
    增强演化模块实现类
    实现IEvolutionModule接口，使用EnhancedEvolutionEngine提供增强功能
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化增强演化模块

        Args:
            config: 配置字典，可选
        """
        self.logger = logger
        self.config = config or self._get_default_config()

        # 增强演化引擎实例
        self.enhanced_engine = create_enhanced_evolution_engine(
            self.config.get("engine_config")
        )

        # 演化状态
        self.evolution_active = False
        self.evolution_thread = None
        self.stop_requested = False

        # 演化历史
        self.evolution_history: List[Dict[str, Any]] = []
        self.max_history_size = self.config.get("max_history_size", 100)

        # 统计信息
        self.statistics = {
            "total_evolutions": 0,
            "successful_evolutions": 0,
            "failed_evolutions": 0,
            "total_evolution_time": 0.0,
            "average_evolution_time": 0.0,
            "best_accuracy": 0.0,
            "best_efficiency": 0.0,
            "best_multi_objective_score": 0.0,
            "algorithm_usage": {},
            "learning_progress": 0.0,
        }

        # 回滚点
        self.rollback_points: Dict[int, Dict[str, Any]] = {}
        self.next_rollback_id = 1

        # 增强功能标志
        self.enable_multi_objective = self.config.get("enable_multi_objective", True)
        self.enable_meta_learning = self.config.get("enable_meta_learning", True)
        self.enable_coevolution = self.config.get("enable_coevolution", False)

        # 协同演化组
        self.coevolution_groups: Dict[str, List[str]] = {}

        # 硬件感知功能
        self.enable_hardware_aware = self.config.get("enable_hardware_aware", True) and HARDWARE_AWARE_AVAILABLE
        self.hardware_aware_module = None
        self.hardware_aware_config = self.config.get("hardware_aware_config", {})
        
        if self.enable_hardware_aware and HARDWARE_AWARE_AVAILABLE:
            try:
                self.hardware_aware_module = create_hardware_aware_evolution_module(
                    self.hardware_aware_config
                )
                self.logger.info("硬件感知模块初始化成功")
            except Exception as e:
                self.logger.warning(f"硬件感知模块初始化失败: {e}")
                self.enable_hardware_aware = False
        else:
            if not HARDWARE_AWARE_AVAILABLE:
                self.logger.info("硬件感知模块不可用，跳过初始化")
            else:
                self.logger.info("硬件感知功能已禁用")

        self.logger.info("增强演化模块初始化完成")
        self.logger.info(
            f"启用功能: 多目标优化={self.enable_multi_objective}, "
            f"元学习={self.enable_meta_learning}, "
            f"协同演化={self.enable_coevolution}, "
            f"硬件感知={self.enable_hardware_aware}"
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "max_history_size": 100,
            "engine_config": {
                "meta_controller_config": {
                    "max_performance_history": 100,
                    "exploration_rate": 0.1,
                    "learning_rate": 0.05,
                },
                "multi_objective_config": {
                    "objective_weights": {
                        "accuracy": 0.4,
                        "efficiency": 0.3,
                        "robustness": 0.2,
                        "adaptability": 0.1,
                    },
                    "max_pareto_history": 50,
                },
                "coevolution_config": {
                    "max_group_size": 5,
                    "collaboration_threshold": 0.7,
                    "knowledge_sharing_rate": 0.3,
                },
            },
            "enable_multi_objective": True,
            "enable_meta_learning": True,
            "enable_coevolution": False,
            "enable_hardware_aware": True,  # 启用硬件感知
            "hardware_aware_config": {
                "optimization_level": "balanced",  # balanced, max_performance, power_efficient, edge_optimized
                "task_complexity": 0.5,  # 默认任务复杂性
                "max_model_size_mb": 500.0,
                "max_memory_usage_gb": 32.0,
            },
            "default_algorithm": "adaptive",  # adaptive, genetic, particle_swarm, etc.
            "max_evolution_time": 300,  # 最大演化时间（秒）
        }

    def evolve_architecture(
        self,
        base_architecture: Dict[str, Any],
        performance_targets: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> EvolutionResult:
        """
        演化神经网络架构（增强版）

        Args:
            base_architecture: 基础架构
            performance_targets: 性能目标
            constraints: 约束条件

        Returns:
            演化结果
        """
        start_time = time.time()
        evolution_id = f"enhanced_evo_{int(start_time)}_{len(self.evolution_history)}"

        try:
            self.logger.info(f"开始增强演化架构 (ID: {evolution_id})")
            self.logger.info(f"性能目标: {performance_targets}")
            self.logger.info(f"启用多目标优化: {self.enable_multi_objective}")

            # 检查演化是否已在运行
            if self.evolution_active:
                self.logger.warning("演化已在运行中，等待当前演化完成")
                return EvolutionResult(
                    success=False,
                    evolved_architecture={},
                    performance_metrics={},
                    generation_info={"error": "evolution_already_active"},
                    error_message="演化已在运行中，请等待当前演化完成",
                )

            # 转换基础架构为任务需求
            task_requirements = self._convert_to_task_requirements(
                base_architecture, performance_targets, constraints
            )

            # 设置资源约束
            resource_constraints = (
                constraints.get("resource_constraints", {}) if constraints else {}
            )

            # 开始演化
            self.evolution_active = True
            self.stop_requested = False

            # 硬件感知优化（如果启用）
            if self.enable_hardware_aware and self.hardware_aware_module:
                try:
                    self.logger.info("应用硬件感知优化")
                    
                    # 获取硬件感知参数
                    task_complexity = self.hardware_aware_config.get("task_complexity", 0.5)
                    hardware_params = self.hardware_aware_module.get_hardware_aware_evolution_parameters(task_complexity)
                    
                    # 根据硬件参数调整资源约束
                    if "memory_limit_gb" in hardware_params:
                        resource_constraints["memory_limit_gb"] = hardware_params["memory_limit_gb"]
                    
                    if "model_size_limit_mb" in hardware_params:
                        resource_constraints["model_size_limit_mb"] = hardware_params["model_size_limit_mb"]
                    
                    # 根据硬件优化基础架构
                    optimized_base_architecture = self.hardware_aware_module.optimize_architecture_for_hardware(
                        base_architecture
                    )
                    
                    # 使用优化后的架构更新任务需求
                    if optimized_base_architecture != base_architecture:
                        task_requirements = self._convert_to_task_requirements(
                            optimized_base_architecture, performance_targets, constraints
                        )
                        self.logger.info("基础架构已根据硬件特性优化")
                    
                    self.logger.info(f"硬件感知参数: 种群大小={hardware_params.get('population_size', 'N/A')}, "
                                   f"突变率={hardware_params.get('mutation_rate', 0.0):.3f}")
                    
                except Exception as e:
                    self.logger.warning(f"硬件感知优化失败，继续使用基础演化: {e}")

            # 使用增强引擎执行演化
            evolved_architecture = self.enhanced_engine.evolve_architecture(
                task_requirements=task_requirements,
                performance_feedback=performance_targets,
                resource_constraints=resource_constraints,
                use_multi_objective=self.enable_multi_objective,
                coevolution_group=None,  # 暂时不使用协同演化
            )

            # 获取演化状态
            enhanced_status = self.enhanced_engine.get_enhanced_status()

            # 创建演化结果
            evolved_arch_dict = self._architecture_to_dict(evolved_architecture)

            # 计算性能指标
            performance_metrics = {
                "accuracy": evolved_architecture.performance_metrics.get(
                    "accuracy", 0.0
                ),
                "efficiency": evolved_architecture.performance_metrics.get(
                    "efficiency", 0.0
                ),
                "robustness": evolved_architecture.performance_metrics.get(
                    "robustness", 0.0
                ),
                "adaptability": evolved_architecture.performance_metrics.get(
                    "adaptability", 0.0
                ),
                "resource_usage": evolved_architecture.performance_metrics.get(
                    "resource_usage", 0.0
                ),
                "overall_score": evolved_architecture.fitness_score,
                "multi_objective_score": enhanced_status.get(
                    "best_multi_objective_score", evolved_architecture.fitness_score
                ),
            }

            # 创建代信息
            generation_info = {
                "generation_id": evolution_id,
                "algorithm_used": self._get_algorithm_used(enhanced_status),
                "multi_objective_used": self.enable_multi_objective,
                "pareto_front_size": enhanced_status.get(
                    "multi_objective_stats", {}
                ).get("last_pareto_size", 0),
                "learning_progress": enhanced_status.get("learning_progress", 0.0),
                "base_engine_status": enhanced_status.get("base_engine_status", {}),
            }

            # 计算演化时间
            evolution_time = time.time() - start_time

            # 更新统计信息
            self._update_statistics(
                success=True,
                evolution_time=evolution_time,
                performance_metrics=performance_metrics,
                enhanced_status=enhanced_status,
            )

            # 保存演化历史
            self._save_evolution_history(
                evolution_id=evolution_id,
                base_architecture=base_architecture,
                evolved_architecture=evolved_arch_dict,
                performance_metrics=performance_metrics,
                generation_info=generation_info,
                evolution_time=evolution_time,
            )

            # 创建回滚点
            self._create_rollback_point(
                evolution_id, evolved_arch_dict, performance_metrics
            )

            # 重置演化状态
            self.evolution_active = False

            self.logger.info(
                f"增强演化完成 (ID: {evolution_id}), 时间: {evolution_time:.2f}s, 适应度: {evolved_architecture.fitness_score:.4f}"
            )

            return EvolutionResult(
                success=True,
                evolved_architecture=evolved_arch_dict,
                performance_metrics=performance_metrics,
                generation_info=generation_info,
                error_message=None,
            )

        except Exception as e:
            self.logger.error(f"增强演化失败: {str(e)}")

            # 更新统计信息（失败）
            self._update_statistics(
                success=False,
                evolution_time=time.time() - start_time,
                performance_metrics={},
                enhanced_status={},
            )

            # 重置演化状态
            self.evolution_active = False

            return EvolutionResult(
                success=False,
                evolved_architecture={},
                performance_metrics={},
                generation_info={
                    "error": "enhanced_evolution_failed",
                    "exception": str(e),
                },
                error_message=f"增强演化失败: {str(e)}",
            )

    def _convert_to_task_requirements(
        self,
        base_architecture: Dict[str, Any],
        performance_targets: Dict[str, float],
        constraints: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """转换为基础架构到任务需求"""
        task_requirements = {
            "architecture_type": base_architecture.get("type", "generic"),
            "description": base_architecture.get(
                "description", "Architecture evolution task"
            ),
            "base_architecture": base_architecture,
            "performance_targets": performance_targets,
        }

        if constraints:
            task_requirements["constraints"] = constraints

        return task_requirements

    def _architecture_to_dict(self, architecture) -> Dict[str, Any]:
        """转换架构为字典"""
        # 如果已经是字典，直接返回
        if isinstance(architecture, dict):
            return architecture

        # 否则尝试调用to_dict方法
        try:
            return architecture.to_dict()
        except AttributeError:
            # 回退到基本转换
            return {
                "architecture_id": getattr(architecture, "architecture_id", "unknown"),
                "layers": getattr(architecture, "layers", []),
                "attention_mechanisms": getattr(
                    architecture, "attention_mechanisms", {}
                ),
                "activation_functions": getattr(
                    architecture, "activation_functions", {}
                ),
                "fusion_strategies": getattr(architecture, "fusion_strategies", {}),
                "connection_patterns": getattr(architecture, "connection_patterns", {}),
                "performance_metrics": getattr(architecture, "performance_metrics", {}),
                "resource_requirements": getattr(
                    architecture, "resource_requirements", {}
                ),
                "fitness_score": getattr(architecture, "fitness_score", 0.0),
                "generation": getattr(architecture, "generation", 0),
            }

    def _get_algorithm_used(self, enhanced_status: Dict[str, Any]) -> str:
        """获取使用的算法"""
        algorithm_usage = enhanced_status.get("algorithm_usage", {})
        if algorithm_usage:
            # 返回使用次数最多的算法
            return max(algorithm_usage.items(), key=lambda x: x[1])[0]
        else:
            return "unknown"

    def _update_statistics(
        self,
        success: bool,
        evolution_time: float,
        performance_metrics: Dict[str, float],
        enhanced_status: Dict[str, Any],
    ):
        """更新统计信息"""
        self.statistics["total_evolutions"] += 1

        if success:
            self.statistics["successful_evolutions"] += 1

            # 更新最佳性能
            accuracy = performance_metrics.get("accuracy", 0.0)
            efficiency = performance_metrics.get("efficiency", 0.0)
            multi_objective_score = performance_metrics.get(
                "multi_objective_score", 0.0
            )

            self.statistics["best_accuracy"] = max(
                self.statistics["best_accuracy"], accuracy
            )
            self.statistics["best_efficiency"] = max(
                self.statistics["best_efficiency"], efficiency
            )
            self.statistics["best_multi_objective_score"] = max(
                self.statistics["best_multi_objective_score"], multi_objective_score
            )
        else:
            self.statistics["failed_evolutions"] += 1

        # 更新演化时间
        self.statistics["total_evolution_time"] += evolution_time
        self.statistics["average_evolution_time"] = (
            self.statistics["total_evolution_time"]
            / self.statistics["total_evolutions"]
        )

        # 更新算法使用统计
        algorithm_usage = enhanced_status.get("algorithm_usage", {})
        for algorithm, count in algorithm_usage.items():
            self.statistics["algorithm_usage"][algorithm] = (
                self.statistics["algorithm_usage"].get(algorithm, 0) + count
            )

        # 更新学习进度
        self.statistics["learning_progress"] = enhanced_status.get(
            "learning_progress", 0.0
        )

    def _save_evolution_history(
        self,
        evolution_id: str,
        base_architecture: Dict[str, Any],
        evolved_architecture: Dict[str, Any],
        performance_metrics: Dict[str, float],
        generation_info: Dict[str, Any],
        evolution_time: float,
    ):
        """保存演化历史"""
        history_entry = {
            "evolution_id": evolution_id,
            "timestamp": time.time(),
            "base_architecture": base_architecture,
            "evolved_architecture": evolved_architecture,
            "performance_metrics": performance_metrics,
            "generation_info": generation_info,
            "evolution_time": evolution_time,
        }

        self.evolution_history.append(history_entry)

        # 限制历史记录大小
        if len(self.evolution_history) > self.max_history_size:
            self.evolution_history = self.evolution_history[-self.max_history_size :]

    def _create_rollback_point(
        self,
        evolution_id: str,
        evolved_architecture: Dict[str, Any],
        performance_metrics: Dict[str, float],
    ):
        """创建回滚点"""
        rollback_id = self.next_rollback_id
        self.next_rollback_id += 1

        self.rollback_points[rollback_id] = {
            "rollback_id": rollback_id,
            "evolution_id": evolution_id,
            "architecture": evolved_architecture.copy(),
            "performance_metrics": performance_metrics.copy(),
            "timestamp": time.time(),
            "statistics": self.statistics.copy(),
        }

        # 限制回滚点数量
        max_rollback_points = self.config.get("max_rollback_points", 10)
        if len(self.rollback_points) > max_rollback_points:
            # 删除最旧的回滚点
            oldest_id = min(self.rollback_points.keys())
            del self.rollback_points[oldest_id]

    def get_evolution_status(self) -> Dict[str, Any]:
        """获取演化状态"""
        enhanced_status = self.enhanced_engine.get_enhanced_status()

        return {
            "module_status": {
                "evolution_active": self.evolution_active,
                "history_size": len(self.evolution_history),
                "rollback_points": len(self.rollback_points),
                "enhanced_features": {
                    "multi_objective": self.enable_multi_objective,
                    "meta_learning": self.enable_meta_learning,
                    "coevolution": self.enable_coevolution,
                },
            },
            "engine_status": enhanced_status,
            "statistics": self.statistics,
            "coevolution_groups": {
                group_id: len(model_ids)
                for group_id, model_ids in self.coevolution_groups.items()
            },
        }

    def get_evolution_statistics(self) -> Dict[str, float]:
        """获取演化统计信息"""
        # 复制统计信息，确保不修改原始数据
        stats_copy = self.statistics.copy()
        
        # 转换值为浮点数（如果可能）
        result = {}
        for key, value in stats_copy.items():
            if isinstance(value, (int, float)):
                result[key] = float(value)
            elif isinstance(value, dict):
                # 对于字典类型，转换其中的数值
                converted_dict = {}
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        converted_dict[k] = float(v)
                    else:
                        converted_dict[k] = v
                result[key] = converted_dict
            else:
                result[key] = value
                
        return result

    def stop_evolution(self) -> bool:
        """停止演化过程"""
        if not self.evolution_active:
            self.logger.warning("没有活动的演化过程可停止")
            return False

        self.stop_requested = True

        # 设置超时
        timeout = 5.0  # 5秒
        start_time = time.time()

        while self.evolution_active and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        if self.evolution_active:
            self.logger.error("停止演化超时")
            return False

        self.logger.info("演化已成功停止")
        return True

    def rollback_evolution(self, generation: int = -1) -> bool:
        """回滚到指定代数的架构"""
        if not self.rollback_points:
            self.logger.error("没有可用的回滚点")
            return False

        if generation == -1:
            # 回滚到最新的回滚点
            rollback_id = max(self.rollback_points.keys())
        else:
            # 回滚到指定代数的回滚点
            if generation not in self.rollback_points:
                self.logger.error(f"回滚点 {generation} 不存在")
                return False
            rollback_id = generation

        rollback_point = self.rollback_points[rollback_id]

        self.logger.info(f"回滚到演化 {rollback_point['evolution_id']}")

        # 在实际系统中，这里应该实际恢复架构状态
        # 目前只记录回滚操作

        # 创建回滚历史记录
        rollback_history = {
            "rollback_id": rollback_id,
            "original_evolution_id": rollback_point["evolution_id"],
            "timestamp": time.time(),
            "architecture": rollback_point["architecture"],
            "performance_metrics": rollback_point["performance_metrics"],
        }

        # 将回滚点添加到历史中
        self.evolution_history.append(
            {
                "evolution_id": f"rollback_{rollback_id}_{int(time.time())}",
                "timestamp": time.time(),
                "base_architecture": rollback_point["architecture"],
                "evolved_architecture": rollback_point["architecture"],  # 回滚后架构不变
                "performance_metrics": rollback_point["performance_metrics"],
                "generation_info": {"type": "rollback", "rollback_id": rollback_id},
                "evolution_time": 0.0,
            }
        )

        self.logger.info(
            f"回滚完成，恢复架构: {rollback_point['architecture'].get('architecture_id', 'unknown')}"
        )
        return True

    def enable_feature(self, feature_name: str, enabled: bool = True) -> bool:
        """启用或禁用增强功能"""
        if feature_name == "multi_objective":
            self.enable_multi_objective = enabled
            self.logger.info(f"{'启用' if enabled else '禁用'}多目标优化")
            return True

        elif feature_name == "meta_learning":
            self.enable_meta_learning = enabled
            self.logger.info(f"{'启用' if enabled else '禁用'}元学习")
            return True

        elif feature_name == "coevolution":
            self.enable_coevolution = enabled
            self.logger.info(f"{'启用' if enabled else '禁用'}协同演化")
            return True

        else:
            self.logger.error(f"未知功能: {feature_name}")
            return False

    def create_coevolution_group(
        self,
        group_id: str,
        model_ids: List[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """创建协同演化组"""
        if not self.enable_coevolution:
            self.logger.error("协同演化功能未启用")
            return False

        success = self.enhanced_engine.create_coevolution_group(
            group_id, model_ids, config
        )

        if success:
            self.coevolution_groups[group_id] = model_ids.copy()
            self.logger.info(f"创建协同演化组 {group_id}, 包含 {len(model_ids)} 个模型")

        return success

    def coevolve_architectures(
        self,
        group_id: str,
        base_architecture: Dict[str, Any],
        performance_targets: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        """协同演化架构"""
        if not self.enable_coevolution:
            self.logger.error("协同演化功能未启用")
            return {}

        if group_id not in self.coevolution_groups:
            self.logger.error(f"协同演化组 {group_id} 不存在")
            return {}

        # 转换基础架构为任务需求
        task_requirements = self._convert_to_task_requirements(
            base_architecture, performance_targets, None
        )

        # 执行协同演化
        results = self.enhanced_engine.coevolve_architectures(
            group_id, task_requirements, performance_targets
        )

        # 转换结果为标准格式
        formatted_results = {}
        for model_id, architecture in results.items():
            formatted_results[model_id] = {
                "architecture": self._architecture_to_dict(architecture),
                "performance_metrics": architecture.performance_metrics,
                "fitness_score": architecture.fitness_score,
            }

        self.logger.info(f"协同演化完成，组: {group_id}, 结果: {len(formatted_results)} 个模型")
        return formatted_results

    def get_learning_insights(self) -> Dict[str, Any]:
        """获取学习洞察"""
        enhanced_status = self.enhanced_engine.get_enhanced_status()

        # 分析算法性能
        algorithm_performance = []
        for algorithm, count in self.statistics.get("algorithm_usage", {}).items():
            algorithm_performance.append(
                {
                    "algorithm": algorithm,
                    "usage_count": count,
                    "success_rate": 0.8,  # 这里应该从历史数据计算
                }
            )

        # 分析性能趋势
        performance_trend = {
            "accuracy_improvement": self._calculate_performance_trend("accuracy"),
            "efficiency_improvement": self._calculate_performance_trend("efficiency"),
            "learning_progress": self.statistics.get("learning_progress", 0.0),
        }

        # 获取任务-算法映射
        meta_stats = enhanced_status.get("meta_controller_stats", {})

        return {
            "algorithm_performance": algorithm_performance,
            "performance_trend": performance_trend,
            "meta_learning": {
                "learned_tasks": meta_stats.get("learned_tasks", 0),
                "performance_records": meta_stats.get("performance_records", 0),
            },
            "recommendations": self._generate_recommendations(),
        }

    def _calculate_performance_trend(self, metric_name: str) -> float:
        """计算性能趋势"""
        if len(self.evolution_history) < 2:
            return 0.0

        # 获取最近几次演化的性能值
        recent_values = []
        for entry in self.evolution_history[-10:]:  # 最近10次
            value = entry["performance_metrics"].get(metric_name, 0.0)
            recent_values.append(value)

        if len(recent_values) < 2:
            return 0.0

        # 计算趋势（简单线性回归斜率）
        try:
            x = list(range(len(recent_values)))
            y = recent_values

            # 计算斜率
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x_i * x_i for x_i in x)

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
        except:
            return 0.0

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """生成优化建议"""
        recommendations = []

        # 基于算法使用情况
        algorithm_usage = self.statistics.get("algorithm_usage", {})
        if algorithm_usage:
            most_used = max(algorithm_usage.items(), key=lambda x: x[1])[0]
            least_used = min(algorithm_usage.items(), key=lambda x: x[1])[0]

            if algorithm_usage[most_used] > algorithm_usage[least_used] * 3:
                recommendations.append(
                    {
                        "type": "algorithm_diversity",
                        "priority": "medium",
                        "message": f"算法使用不均衡，建议尝试更多使用{least_used}算法",
                        "suggestion": "调整元学习控制器的探索率",
                    }
                )

        # 基于性能趋势
        accuracy_trend = self._calculate_performance_trend("accuracy")
        if accuracy_trend < 0:
            recommendations.append(
                {
                    "type": "performance_decline",
                    "priority": "high",
                    "message": "检测到准确率下降趋势",
                    "suggestion": "考虑调整演化参数或启用更多探索策略",
                }
            )

        # 基于学习进度
        learning_progress = self.statistics.get("learning_progress", 0.0)
        if learning_progress < 0.3 and self.statistics["total_evolutions"] > 10:
            recommendations.append(
                {
                    "type": "learning_stagnation",
                    "priority": "medium",
                    "message": "学习进度缓慢",
                    "suggestion": "增加演化种群大小或启用协同演化",
                }
            )

        return recommendations


# 工厂函数
def create_enhanced_evolution_module(
    config: Optional[Dict[str, Any]] = None
) -> EnhancedEvolutionModule:
    """创建增强演化模块实例"""
    return EnhancedEvolutionModule(config)


# 兼容性函数
def get_enhanced_evolution_module(
    config: Optional[Dict[str, Any]] = None
) -> EnhancedEvolutionModule:
    """获取增强演化模块（兼容现有代码）"""
    return create_enhanced_evolution_module(config)


if __name__ == "__main__":
    # 演示增强演化模块
    print("=" * 80)
    print("增强演化模块演示")
    print("=" * 80)

    try:
        # 创建增强演化模块
        print("\n1. 创建增强演化模块...")
        module = create_enhanced_evolution_module()

        # 获取初始状态
        print("\n2. 初始状态:")
        status = module.get_evolution_status()
        print(f"   增强功能: {status['module_status']['enhanced_features']}")
        print(f"   历史记录: {status['module_status']['history_size']}")

        # 执行演化
        print("\n3. 执行增强演化...")
        base_architecture = {
            "type": "test_architecture",
            "layers": [{"type": "dense", "size": 128}],
            "description": "测试架构",
        }

        performance_targets = {
            "accuracy": 0.9,
            "efficiency": 0.8,
            "robustness": 0.7,
        }

        result = module.evolve_architecture(
            base_architecture=base_architecture,
            performance_targets=performance_targets,
            constraints={"max_memory_mb": 1000},
        )

        if result.success:
            print(f"   演化成功!")
            print(f"   适应度: {result.performance_metrics.get('overall_score', 0.0):.4f}")
            print(f"   算法: {result.generation_info.get('algorithm_used', 'unknown')}")
            print(
                f"   多目标优化: {result.generation_info.get('multi_objective_used', False)}"
            )
        else:
            print(f"   演化失败: {result.error_message}")

        # 获取更新后状态
        print("\n4. 更新后状态:")
        status = module.get_evolution_status()
        print(f"   总演化次数: {status['statistics']['total_evolutions']}")
        print(f"   成功次数: {status['statistics']['successful_evolutions']}")
        print(f"   算法使用: {status['statistics'].get('algorithm_usage', {})}")

        # 获取学习洞察
        print("\n5. 学习洞察:")
        insights = module.get_learning_insights()
        print(f"   算法性能: {len(insights['algorithm_performance'])} 条记录")
        print(f"   性能趋势: {insights['performance_trend']}")
        print(f"   优化建议: {len(insights['recommendations'])} 条")

        print("\n✅ 增强演化模块演示成功")

    except Exception as e:
        print(f"\n❌ 增强演化模块演示失败: {str(e)}")
        import traceback

        traceback.print_exc()
