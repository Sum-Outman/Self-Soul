#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强演化引擎 - Enhanced Evolution Engine

基于现有的ArchitectureEvolutionEngine，提供以下增强功能：
1. 多算法支持：遗传算法、粒子群优化、差分进化、贝叶斯优化
2. 元学习演化策略：学习选择最佳演化算法和参数
3. 多目标优化：支持Pareto前沿和NSGA-II算法
4. 协同演化框架：支持多模型协同演化
5. 自适应参数调整：根据演化进度动态调整参数
6. 演化轨迹学习：从历史演化中学习优化模式

设计原则：
- 向后兼容：兼容现有的IEvolutionModule接口
- 可扩展：易于添加新算法和策略
- 自适应：根据任务需求自动选择最佳算法
- 高效：优化计算资源使用
- 可解释：提供演化过程的透明度和可解释性
"""

import logging
import time
import json
import random
import math
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import copy
import statistics

# 检查PyTorch是否可用
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available, NAS functionality will be limited")
    TORCH_AVAILABLE = False

from core.architecture_evolution_engine import (
    ArchitectureEvolutionEngine,
    NetworkArchitecture,
    NetworkArchitectureSearcher,
    ArchitectureComponent,
    AttentionType,
    FusionStrategy,
    ActivationType,
)

# 配置日志
logger = logging.getLogger(__name__)


class EvolutionAlgorithm(Enum):
    """演化算法类型"""

    GENETIC_ALGORITHM = "genetic_algorithm"  # 遗传算法
    PARTICLE_SWARM = "particle_swarm"  # 粒子群优化
    DIFFERENTIAL_EVOLUTION = "differential_evolution"  # 差分进化
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"  # 贝叶斯优化
    SIMULATED_ANNEALING = "simulated_annealing"  # 模拟退火
    ANT_COLONY = "ant_colony"  # 蚁群算法
    NEURO_EVOLUTION = "neuro_evolution"  # 神经演化
    NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"  # 神经架构搜索
    HYBRID = "hybrid"  # 混合算法


class OptimizationObjective(Enum):
    """优化目标类型"""

    SINGLE_OBJECTIVE = "single_objective"  # 单目标优化
    MULTI_OBJECTIVE = "multi_objective"  # 多目标优化
    CONSTRAINED_OPTIMIZATION = "constrained_optimization"  # 约束优化
    MULTI_MODAL = "multi_modal"  # 多模态优化


@dataclass
class AlgorithmPerformance:
    """算法性能记录"""

    algorithm: EvolutionAlgorithm
    task_type: str
    performance_metrics: Dict[str, float]
    execution_time: float
    resource_usage: Dict[str, float]
    success_rate: float
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0

    def get_weighted_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """获取加权性能分数"""
        if not weights:
            weights = {
                "accuracy": 0.4,
                "efficiency": 0.3,
                "convergence_speed": 0.2,
                "resource_efficiency": 0.1,
            }

        score = 0.0
        for metric, weight in weights.items():
            if metric in self.performance_metrics:
                score += self.performance_metrics[metric] * weight

        return score


class MetaLearningEvolutionController:
    """元学习演化控制器

    学习选择最佳演化算法和参数，基于：
    1. 任务类型和需求
    2. 历史性能数据
    3. 可用资源
    4. 时间约束
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # 算法性能历史
        self.algorithm_performance: Dict[str, List[AlgorithmPerformance]] = defaultdict(
            list
        )
        self.max_performance_history = self.config.get("max_performance_history", 100)

        # 任务-算法映射
        self.task_algorithm_mapping: Dict[str, EvolutionAlgorithm] = {}

        # 学习参数
        self.exploration_rate = self.config.get("exploration_rate", 0.1)  # 探索率
        self.learning_rate = self.config.get("learning_rate", 0.05)  # 学习率

        # 特征提取器
        self.feature_extractors = self._initialize_feature_extractors()

        logger.info("元学习演化控制器初始化完成")

    def _initialize_feature_extractors(self) -> Dict[str, Callable]:
        """初始化特征提取器"""
        return {
            "task_complexity": self._extract_task_complexity,
            "resource_constraints": self._extract_resource_constraints,
            "performance_targets": self._extract_performance_targets,
            "constraint_type": self._extract_constraint_type,
        }

    def select_algorithm(
        self,
        task_requirements: Dict[str, Any],
        performance_targets: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None,
        available_time: Optional[float] = None,
        available_resources: Optional[Dict[str, float]] = None,
    ) -> Tuple[EvolutionAlgorithm, Dict[str, Any]]:
        """选择最佳算法和参数"""

        # 提取任务特征
        task_features = self._extract_task_features(
            task_requirements, performance_targets, constraints
        )

        # 生成任务签名
        task_signature = self._generate_task_signature(task_features)

        # 检查是否有已知的最佳算法
        if task_signature in self.task_algorithm_mapping:
            known_algorithm = self.task_algorithm_mapping[task_signature]

            # 有时探索新算法
            if random.random() > self.exploration_rate:
                logger.info(f"使用已知最佳算法: {known_algorithm.value}")
                return known_algorithm, self._get_algorithm_parameters(
                    known_algorithm, task_features
                )

        # 基于特征选择算法
        algorithm = self._select_algorithm_by_features(task_features)
        parameters = self._get_algorithm_parameters(algorithm, task_features)

        logger.info(f"选择算法: {algorithm.value}")
        return algorithm, parameters

    def update_performance(
        self,
        algorithm: EvolutionAlgorithm,
        task_requirements: Dict[str, Any],
        performance_metrics: Dict[str, float],
        execution_time: float,
        resource_usage: Dict[str, float],
        success: bool,
    ):
        """更新算法性能记录"""

        # 提取任务特征
        task_features = self._extract_task_features(task_requirements, {}, None)
        task_signature = self._generate_task_signature(task_features)

        # 计算成功率
        success_rate = 1.0 if success else 0.0
        if task_signature in self.task_algorithm_mapping:
            # 更新现有记录的成功率
            existing_records = self.algorithm_performance.get(task_signature, [])
            for record in existing_records:
                if record.algorithm == algorithm:
                    record.success_rate = (
                        record.success_rate * record.usage_count + success_rate
                    ) / (record.usage_count + 1)
                    record.usage_count += 1
                    record.last_used = time.time()
                    record.performance_metrics.update(performance_metrics)
                    record.execution_time = execution_time
                    record.resource_usage = resource_usage
                    return

        # 创建新记录
        performance_record = AlgorithmPerformance(
            algorithm=algorithm,
            task_type=task_signature,
            performance_metrics=performance_metrics,
            execution_time=execution_time,
            resource_usage=resource_usage,
            success_rate=success_rate,
            usage_count=1,
        )

        # 存储记录
        self.algorithm_performance[task_signature].append(performance_record)

        # 限制历史记录数量
        if (
            len(self.algorithm_performance[task_signature])
            > self.max_performance_history
        ):
            self.algorithm_performance[task_signature] = self.algorithm_performance[
                task_signature
            ][-self.max_performance_history :]

        # 更新任务-算法映射
        self._update_task_algorithm_mapping(task_signature)

    def _extract_task_features(
        self,
        task_requirements: Dict[str, Any],
        performance_targets: Dict[str, float],
        constraints: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """提取任务特征"""
        features = {}

        for feature_name, extractor in self.feature_extractors.items():
            try:
                features[feature_name] = extractor(
                    task_requirements, performance_targets, constraints
                )
            except Exception as e:
                logger.warning(f"提取特征 {feature_name} 失败: {str(e)}")
                features[feature_name] = None

        return features

    def _extract_task_complexity(
        self,
        task_requirements: Dict[str, Any],
        performance_targets: Dict[str, float],
        constraints: Optional[Dict[str, Any]],
    ) -> float:
        """提取任务复杂性特征"""
        complexity = 0.0

        # 基于性能目标数量
        complexity += len(performance_targets) * 0.1

        # 基于约束数量
        if constraints:
            complexity += len(constraints) * 0.05

        # 基于任务类型
        task_type = task_requirements.get("architecture_type", "generic")
        if task_type in ["transformer", "multimodal", "reinforcement_learning"]:
            complexity += 0.3
        elif task_type in ["cnn", "rnn", "attention"]:
            complexity += 0.2
        else:
            complexity += 0.1

        return min(complexity, 1.0)

    def _extract_resource_constraints(
        self,
        task_requirements: Dict[str, Any],
        performance_targets: Dict[str, float],
        constraints: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """提取资源约束特征"""
        resource_features = {
            "memory_constrained": 0.0,
            "time_constrained": 0.0,
            "compute_constrained": 0.0,
        }

        if constraints:
            if "max_memory_mb" in constraints and constraints["max_memory_mb"] < 1000:
                resource_features["memory_constrained"] = 1.0

            if (
                "max_compute_time_seconds" in constraints
                and constraints["max_compute_time_seconds"] < 300
            ):
                resource_features["time_constrained"] = 1.0

            if (
                "max_compute_gflops" in constraints
                and constraints["max_compute_gflops"] < 10
            ):
                resource_features["compute_constrained"] = 1.0

        return resource_features

    def _extract_performance_targets(
        self,
        task_requirements: Dict[str, Any],
        performance_targets: Dict[str, float],
        constraints: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """提取性能目标特征"""
        return performance_targets.copy()

    def _extract_constraint_type(
        self,
        task_requirements: Dict[str, Any],
        performance_targets: Dict[str, float],
        constraints: Optional[Dict[str, Any]],
    ) -> str:
        """提取约束类型特征"""
        if not constraints:
            return "unconstrained"

        constraint_types = []
        if "resource_constraints" in constraints:
            constraint_types.append("resource")
        if "architecture_constraints" in constraints:
            constraint_types.append("architecture")
        if "performance_constraints" in constraints:
            constraint_types.append("performance")

        return "_".join(constraint_types) if constraint_types else "other"

    def _generate_task_signature(self, task_features: Dict[str, Any]) -> str:
        """生成任务签名"""
        # 使用特征哈希生成签名
        feature_str = json.dumps(task_features, sort_keys=True)
        return hashlib.md5(feature_str.encode()).hexdigest()[:8]

    def _select_algorithm_by_features(
        self, task_features: Dict[str, Any]
    ) -> EvolutionAlgorithm:
        """基于特征选择算法"""

        # 基于任务复杂性选择
        complexity = task_features.get("task_complexity", 0.5)
        resource_constraints = task_features.get("resource_constraints", {})

        # 检查资源约束
        is_memory_constrained = (
            resource_constraints.get("memory_constrained", 0.0) > 0.5
        )
        is_time_constrained = resource_constraints.get("time_constrained", 0.0) > 0.5

        if is_time_constrained:
            # 时间受限：选择快速收敛的算法
            if complexity > 0.7:
                return EvolutionAlgorithm.PARTICLE_SWARM
            else:
                return EvolutionAlgorithm.SIMULATED_ANNEALING

        elif is_memory_constrained:
            # 内存受限：选择内存效率高的算法
            return EvolutionAlgorithm.DIFFERENTIAL_EVOLUTION

        elif complexity > 0.8:
            # 非常高复杂性任务（神经架构设计）：选择NAS算法
            # 检查任务是否需要架构设计
            task_type = task_features.get("task_type", "")
            if "architecture" in task_type.lower() or "neural" in task_type.lower():
                return EvolutionAlgorithm.NEURAL_ARCHITECTURE_SEARCH
            else:
                return EvolutionAlgorithm.GENETIC_ALGORITHM

        elif complexity > 0.7:
            # 高复杂性任务：选择强大的算法
            return EvolutionAlgorithm.GENETIC_ALGORITHM

        elif complexity > 0.4:
            # 中等复杂性任务：选择平衡的算法
            return EvolutionAlgorithm.PARTICLE_SWARM

        else:
            # 低复杂性任务：选择简单的算法
            return EvolutionAlgorithm.BAYESIAN_OPTIMIZATION

    def _get_algorithm_parameters(
        self, algorithm: EvolutionAlgorithm, task_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """获取算法参数"""

        base_params = {
            "population_size": 20,
            "max_generations": 50,
            "mutation_rate": 0.2,
            "crossover_rate": 0.8,
        }

        # 基于算法类型调整参数
        if algorithm == EvolutionAlgorithm.GENETIC_ALGORITHM:
            params = {
                **base_params,
                "selection_method": "tournament",
                "tournament_size": 3,
                "elitism": True,
                "elite_size": 2,
            }

        elif algorithm == EvolutionAlgorithm.PARTICLE_SWARM:
            params = {
                **base_params,
                "particle_count": 30,
                "inertia_weight": 0.7,
                "cognitive_coefficient": 1.5,
                "social_coefficient": 1.5,
                "velocity_clamp": 0.5,
            }

        elif algorithm == EvolutionAlgorithm.DIFFERENTIAL_EVOLUTION:
            params = {
                **base_params,
                "strategy": "best1bin",
                "differential_weight": 0.8,
                "crossover_probability": 0.7,
            }

        elif algorithm == EvolutionAlgorithm.BAYESIAN_OPTIMIZATION:
            params = {
                **base_params,
                "acquisition_function": "ei",
                "kappa": 2.576,
                "xi": 0.0,
                "n_restarts_optimizer": 5,
            }

        elif algorithm == EvolutionAlgorithm.NEURAL_ARCHITECTURE_SEARCH:
            # NAS算法特定参数
            params = {
                "nas_algorithm": "darts",  # 默认使用DARTS算法
                "search_epochs": 50,  # 搜索轮数
                "learning_rate": 0.025,
                "arch_learning_rate": 3e-4,
                "weight_decay": 3e-4,
                "init_channels": 16,  # 初始通道数
                "num_cells": 8,  # 网络层数
                "num_nodes": 4,  # 每层的节点数
                "use_gpu": True,  # 是否使用GPU
                "grad_clip": 5.0,
                "momentum": 0.9,
                "drop_path_prob": 0.3,
                "search_space": {
                    "operations": [
                        "none", "max_pool_3x3", "avg_pool_3x3", "skip_connect",
                        "sep_conv_3x3", "sep_conv_5x5", "dil_conv_3x3", "dil_conv_5x5"
                    ],
                    "input_dim": 32,  # 默认输入维度
                    "output_dim": 10,  # 默认输出维度（分类任务）
                }
            }

        else:
            params = base_params

        # 基于任务复杂性调整参数
        complexity = task_features.get("task_complexity", 0.5)

        if complexity > 0.7:
            params["population_size"] = 30
            params["max_generations"] = 100
        elif complexity > 0.4:
            params["population_size"] = 25
            params["max_generations"] = 75
        else:
            params["population_size"] = 20
            params["max_generations"] = 50

        return params

    def _update_task_algorithm_mapping(self, task_signature: str):
        """更新任务-算法映射"""
        if task_signature not in self.algorithm_performance:
            return

        records = self.algorithm_performance[task_signature]
        if not records:
            return

        # 找到最佳算法
        best_record = max(records, key=lambda r: r.get_weighted_score())

        # 更新映射
        self.task_algorithm_mapping[task_signature] = best_record.algorithm
        logger.debug(f"更新任务-算法映射: {task_signature} -> {best_record.algorithm.value}")


class MultiObjectiveOptimizer:
    """多目标优化器

    实现多目标优化算法，包括：
    1. Pareto前沿计算
    2. NSGA-II算法
    3. 目标权重自适应调整
    4. 约束处理
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # 目标权重
        self.objective_weights = self.config.get("objective_weights", {})

        # Pareto前沿历史
        self.pareto_front_history: List[List[Dict[str, Any]]] = []
        self.max_pareto_history = self.config.get("max_pareto_history", 50)

        # 目标归一化器
        self.objective_normalizers = {}

        logger.info("多目标优化器初始化完成")

    def optimize(
        self,
        population: List[NetworkArchitecture],
        objectives: Dict[str, Callable],
        constraints: Optional[Dict[str, Callable]] = None,
        max_generations: int = 100,
    ) -> List[NetworkArchitecture]:
        """执行多目标优化"""

        # 初始评估
        evaluated_population = self._evaluate_population(
            population, objectives, constraints
        )

        # 计算初始Pareto前沿
        pareto_front = self._compute_pareto_front(evaluated_population)
        self.pareto_front_history.append(pareto_front)

        # NSGA-II优化
        optimized_population = self._nsga2_optimization(
            evaluated_population, objectives, constraints, max_generations
        )

        # 计算最终Pareto前沿
        final_pareto = self._compute_pareto_front(optimized_population)
        self.pareto_front_history.append(final_pareto)

        # 限制历史记录
        if len(self.pareto_front_history) > self.max_pareto_history:
            self.pareto_front_history = self.pareto_front_history[
                -self.max_pareto_history :
            ]

        return optimized_population

    def _evaluate_population(
        self,
        population: List[NetworkArchitecture],
        objectives: Dict[str, Callable],
        constraints: Optional[Dict[str, Callable]] = None,
    ) -> List[Dict[str, Any]]:
        """评估种群"""
        evaluated = []

        for arch in population:
            # 计算目标值
            objective_values = {}
            for obj_name, obj_func in objectives.items():
                try:
                    objective_values[obj_name] = obj_func(arch)
                except Exception as e:
                    logger.warning(f"计算目标 {obj_name} 失败: {str(e)}")
                    objective_values[obj_name] = 0.0

            # 检查约束
            constraint_violations = 0
            constraint_values = {}

            if constraints:
                for const_name, const_func in constraints.items():
                    try:
                        const_value = const_func(arch)
                        constraint_values[const_name] = const_value

                        # 假设约束应 <= 0
                        if const_value > 0:
                            constraint_violations += 1
                    except Exception as e:
                        logger.warning(f"检查约束 {const_name} 失败: {str(e)}")

            # 计算总适应度（考虑约束违反）
            total_fitness = self._calculate_total_fitness(
                objective_values, constraint_violations
            )

            evaluated.append(
                {
                    "architecture": arch,
                    "objective_values": objective_values,
                    "constraint_values": constraint_values,
                    "constraint_violations": constraint_violations,
                    "total_fitness": total_fitness,
                    "pareto_rank": 0,
                    "crowding_distance": 0.0,
                }
            )

        return evaluated

    def _calculate_total_fitness(
        self, objective_values: Dict[str, float], constraint_violations: int
    ) -> float:
        """计算总适应度"""

        # 如果有约束违反，惩罚适应度
        if constraint_violations > 0:
            penalty = 0.1 * constraint_violations
            base_fitness = 0.5  # 基础适应度
            return max(0.0, base_fitness - penalty)

        # 加权目标值
        total = 0.0
        total_weight = 0.0

        for obj_name, obj_value in objective_values.items():
            weight = self.objective_weights.get(obj_name, 1.0)
            total += obj_value * weight
            total_weight += weight

        if total_weight > 0:
            return total / total_weight
        else:
            return (
                sum(objective_values.values()) / len(objective_values)
                if objective_values
                else 0.0
            )

    def _compute_pareto_front(
        self, evaluated_population: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """计算Pareto前沿"""
        if not evaluated_population:
            return []

        pareto_front = []

        for i, individual1 in enumerate(evaluated_population):
            dominated = False

            for j, individual2 in enumerate(evaluated_population):
                if i == j:
                    continue

                # 检查individual2是否支配individual1
                if self._dominates(individual2, individual1):
                    dominated = True
                    break

            if not dominated:
                pareto_front.append(individual1)

        return pareto_front

    def _dominates(
        self, individual1: Dict[str, Any], individual2: Dict[str, Any]
    ) -> bool:
        """检查individual1是否支配individual2"""
        obj1 = individual1["objective_values"]
        obj2 = individual2["objective_values"]

        # 对于最大化问题：如果所有目标都不差，且至少一个更好，则支配
        all_not_worse = all(
            obj1.get(obj_name, 0.0) >= obj2.get(obj_name, 0.0)
            for obj_name in obj1.keys() | obj2.keys()
        )
        at_least_one_better = any(
            obj1.get(obj_name, 0.0) > obj2.get(obj_name, 0.0)
            for obj_name in obj1.keys() | obj2.keys()
        )

        return all_not_worse and at_least_one_better

    def _nsga2_optimization(
        self,
        initial_population: List[Dict[str, Any]],
        objectives: Dict[str, Callable],
        constraints: Optional[Dict[str, Callable]],
        max_generations: int,
    ) -> List[NetworkArchitecture]:
        """NSGA-II优化算法"""

        population = initial_population.copy()
        population_size = len(population)

        for generation in range(max_generations):
            # 1. 非支配排序
            fronts = self._non_dominated_sort(population)

            # 2. 计算拥挤距离
            for front in fronts:
                self._calculate_crowding_distance(front)

            # 3. 选择（锦标赛选择）
            selected = self._tournament_selection(population, population_size // 2)

            # 4. 交叉和变异
            offspring = self._crossover_and_mutate(selected)

            # 5. 评估子代
            evaluated_offspring = self._evaluate_population(
                [ind["architecture"] for ind in offspring], objectives, constraints
            )

            # 6. 合并种群
            combined = population + evaluated_offspring

            # 7. 环境选择（选择下一代）
            population = self._environmental_selection(combined, population_size)

            # 记录进度
            if generation % 10 == 0:
                best_fitness = max(ind["total_fitness"] for ind in population)
                logger.info(
                    f"NSGA-II 代 {generation}/{max_generations}, 最佳适应度: {best_fitness:.4f}"
                )

        # 返回架构列表
        return [ind["architecture"] for ind in population]

    def _non_dominated_sort(
        self, population: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """非支配排序"""
        fronts = [[]]

        for i, individual1 in enumerate(population):
            individual1["domination_count"] = 0
            individual1["dominated_set"] = []

            for j, individual2 in enumerate(population):
                if i == j:
                    continue

                if self._dominates(individual1, individual2):
                    individual1["dominated_set"].append(j)
                elif self._dominates(individual2, individual1):
                    individual1["domination_count"] += 1

            if individual1["domination_count"] == 0:
                individual1["pareto_rank"] = 0
                fronts[0].append(individual1)

        i = 0
        while fronts[i]:
            next_front = []

            for individual1 in fronts[i]:
                for j in individual1["dominated_set"]:
                    population[j]["domination_count"] -= 1

                    if population[j]["domination_count"] == 0:
                        population[j]["pareto_rank"] = i + 1
                        next_front.append(population[j])

            i += 1
            fronts.append(next_front)

        return fronts

    def _calculate_crowding_distance(self, front: List[Dict[str, Any]]):
        """计算拥挤距离"""
        if not front:
            return

        # 初始化拥挤距离
        for individual in front:
            individual["crowding_distance"] = 0.0

        # 对每个目标计算拥挤距离
        objective_names = list(front[0]["objective_values"].keys())

        for obj_name in objective_names:
            # 按目标值排序
            front.sort(key=lambda x: x["objective_values"][obj_name])

            # 边界个体具有无限距离
            front[0]["crowding_distance"] = float("inf")
            front[-1]["crowding_distance"] = float("inf")

            # 目标值范围
            obj_min = front[0]["objective_values"][obj_name]
            obj_max = front[-1]["objective_values"][obj_name]
            obj_range = obj_max - obj_min

            if obj_range > 0:
                for i in range(1, len(front) - 1):
                    prev_obj = front[i - 1]["objective_values"][obj_name]
                    next_obj = front[i + 1]["objective_values"][obj_name]

                    front[i]["crowding_distance"] += (next_obj - prev_obj) / obj_range

    def _tournament_selection(
        self, population: List[Dict[str, Any]], selection_size: int
    ) -> List[Dict[str, Any]]:
        """锦标赛选择"""
        selected = []

        for _ in range(selection_size):
            # 随机选择两个个体
            candidates = random.sample(population, 2)

            # 选择更好的个体（基于Pareto等级和拥挤距离）
            if candidates[0]["pareto_rank"] < candidates[1]["pareto_rank"]:
                selected.append(candidates[0])
            elif candidates[0]["pareto_rank"] > candidates[1]["pareto_rank"]:
                selected.append(candidates[1])
            elif (
                candidates[0]["crowding_distance"] > candidates[1]["crowding_distance"]
            ):
                selected.append(candidates[0])
            else:
                selected.append(candidates[1])

        return selected

    def _crossover_and_mutate(
        self, selected: List[Dict[str, Any]]
    ) -> List[NetworkArchitecture]:
        """交叉和变异"""
        offspring = []

        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                parent1 = selected[i]["architecture"]
                parent2 = selected[i + 1]["architecture"]

                # 交叉
                child1, child2 = self._crossover_architectures(parent1, parent2)

                # 变异
                child1 = self._mutate_architecture(child1, mutation_rate=0.1)
                child2 = self._mutate_architecture(child2, mutation_rate=0.1)

                offspring.extend([child1, child2])

        return offspring

    def _crossover_architectures(
        self, arch1: NetworkArchitecture, arch2: NetworkArchitecture
    ) -> Tuple[NetworkArchitecture, NetworkArchitecture]:
        """交叉两个架构"""
        # 创建子代ID
        child1_id = f"child_{int(time.time())}_{random.randint(0, 10000)}_1"
        child2_id = f"child_{int(time.time())}_{random.randint(0, 10000)}_2"

        # 简单单点交叉（针对层）
        layers1 = arch1.layers
        layers2 = arch2.layers

        if len(layers1) > 1 and len(layers2) > 1:
            crossover_point = random.randint(1, min(len(layers1), len(layers2)) - 1)

            child1_layers = layers1[:crossover_point] + layers2[crossover_point:]
            child2_layers = layers2[:crossover_point] + layers1[crossover_point:]
        else:
            child1_layers = layers1.copy()
            child2_layers = layers2.copy()

        # 创建子代架构
        child1 = NetworkArchitecture(
            architecture_id=child1_id,
            layers=child1_layers,
            attention_mechanisms=arch1.attention_mechanisms.copy(),
            activation_functions=arch1.activation_functions.copy(),
            fusion_strategies=arch1.fusion_strategies.copy(),
            connection_patterns=arch1.connection_patterns.copy(),
            generation=arch1.generation + 1,
            parent_ids=[arch1.architecture_id, arch2.architecture_id],
        )

        child2 = NetworkArchitecture(
            architecture_id=child2_id,
            layers=child2_layers,
            attention_mechanisms=arch2.attention_mechanisms.copy(),
            activation_functions=arch2.activation_functions.copy(),
            fusion_strategies=arch2.fusion_strategies.copy(),
            connection_patterns=arch2.connection_patterns.copy(),
            generation=arch2.generation + 1,
            parent_ids=[arch1.architecture_id, arch2.architecture_id],
        )

        return child1, child2

    def _mutate_architecture(
        self, architecture: NetworkArchitecture, mutation_rate: float = 0.1
    ) -> NetworkArchitecture:
        """变异架构"""
        mutated = copy.deepcopy(architecture)
        mutated.architecture_id = (
            f"mutated_{mutated.architecture_id}_{int(time.time())}"
        )
        mutated.generation += 1

        # 随机变异层
        if random.random() < mutation_rate and mutated.layers:
            layer_idx = random.randint(0, len(mutated.layers) - 1)
            layer = mutated.layers[layer_idx]

            # 随机修改层参数
            if "size" in layer:
                layer["size"] = max(
                    1, layer["size"] + random.choice([-1, 1]) * random.randint(1, 3)
                )

            if "dropout_rate" in layer:
                layer["dropout_rate"] = max(
                    0.0, min(1.0, layer["dropout_rate"] + random.uniform(-0.1, 0.1))
                )

        return mutated

    def _environmental_selection(
        self, combined: List[Dict[str, Any]], population_size: int
    ) -> List[Dict[str, Any]]:
        """环境选择"""
        # 非支配排序
        fronts = self._non_dominated_sort(combined)

        # 按Pareto等级排序
        combined.sort(key=lambda x: x["pareto_rank"])

        # 选择前population_size个个体
        return combined[:population_size]


class EnhancedEvolutionEngine:
    """增强演化引擎主类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()

        # 为ArchitectureEvolutionEngine准备配置
        # 确保包含search_space等必要字段
        base_engine_config = self._prepare_base_engine_config(config)

        # 基础演化引擎
        self.base_engine = ArchitectureEvolutionEngine(base_engine_config)

        # 增强组件
        self.meta_controller = MetaLearningEvolutionController(
            self.config.get("meta_controller_config")
        )

        self.multi_objective_optimizer = MultiObjectiveOptimizer(
            self.config.get("multi_objective_config")
        )

        # 演化状态
        self.enhanced_state = {
            "total_evolutions": 0,
            "algorithm_usage": defaultdict(int),
            "average_improvement": 0.0,
            "best_multi_objective_score": 0.0,
            "learning_progress": 0.0,
        }

        # 协同演化组
        self.coevolution_groups: Dict[str, List[str]] = {}

        logger.info("增强演化引擎初始化完成")

    def _prepare_base_engine_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """准备基础引擎配置
        
        确保配置包含ArchitectureEvolutionEngine所需的字段，如search_space
        
        Args:
            config: 原始配置
            
        Returns:
            适合ArchitectureEvolutionEngine的配置
        """
        if config is None:
            config = {}
        
        # 创建基础引擎配置的副本
        base_config = config.copy() if config else {}
        
        # 确保包含search_space
        if "search_space" not in base_config:
            # 默认搜索空间配置
            base_config["search_space"] = {
                "layer_types": ["dense", "conv1d", "conv2d", "lstm", "gru", "attention"],
                "layer_sizes": [32, 64, 128, 256, 512],
                "activation_functions": ["relu", "sigmoid", "tanh", "leaky_relu", "gelu"],
                "attention_types": ["scaled_dot_product", "multi_head", "local"],
                "normalization_types": ["batch_norm", "layer_norm", "instance_norm", "group_norm", "none"],
                "fusion_strategies": ["concatenate", "add", "multiply", "average"],
                "max_layers": 10,
                "min_layers": 1,
            }
        
        # 确保包含其他必要字段
        if "mutation_rate" not in base_config:
            base_config["mutation_rate"] = 0.1
        
        if "crossover_rate" not in base_config:
            base_config["crossover_rate"] = 0.8
        
        if "population_size" not in base_config:
            base_config["population_size"] = 100
        
        if "max_generations" not in base_config:
            base_config["max_generations"] = 50
        
        if "elite_size" not in base_config:
            base_config["elite_size"] = 5
        
        return base_config

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
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
        }

    def evolve_architecture(
        self,
        task_requirements: Dict[str, Any],
        performance_feedback: Optional[Dict[str, float]] = None,
        resource_constraints: Optional[Dict[str, float]] = None,
        use_multi_objective: bool = False,
        coevolution_group: Optional[str] = None,
    ) -> NetworkArchitecture:
        """演化架构（增强版）"""
        start_time = time.time()

        # 1. 选择最佳算法
        algorithm, algorithm_params = self.meta_controller.select_algorithm(
            task_requirements=task_requirements,
            performance_targets=performance_feedback or {},
            constraints=resource_constraints,
        )

        # 2. 记录算法使用
        self.enhanced_state["algorithm_usage"][algorithm.value] += 1
        self.enhanced_state["total_evolutions"] += 1

        logger.info(f"使用算法: {algorithm.value}, 参数: {algorithm_params}")

        # 3. 执行演化
        try:
            if algorithm == EvolutionAlgorithm.NEURAL_ARCHITECTURE_SEARCH:
                # NAS算法演化
                result = self._evolve_with_nas(
                    task_requirements,
                    performance_feedback,
                    resource_constraints,
                    algorithm_params,
                )
            elif use_multi_objective and performance_feedback:
                # 多目标优化
                result = self._evolve_with_multi_objective(
                    task_requirements,
                    performance_feedback,
                    resource_constraints,
                    algorithm_params,
                )
            else:
                # 单目标优化（使用基础引擎）
                result = self.base_engine.evolve_architecture(
                    task_requirements, performance_feedback, resource_constraints
                )

            # 4. 更新元学习控制器
            execution_time = time.time() - start_time

            # 计算性能指标
            performance_metrics = {
                "accuracy": result.performance_metrics.get("accuracy", 0.0),
                "efficiency": result.performance_metrics.get("efficiency", 0.0),
                "convergence_speed": 1.0 / (execution_time + 1e-6),
                "resource_efficiency": 1.0
                / (result.resource_requirements.get("compute_gflops", 1.0) + 1e-6),
            }

            # 估计资源使用
            resource_usage = {
                "cpu_time": execution_time,
                "memory_mb": result.resource_requirements.get("memory_mb", 100.0),
                "compute_gflops": result.resource_requirements.get(
                    "compute_gflops", 1.0
                ),
            }

            self.meta_controller.update_performance(
                algorithm=algorithm,
                task_requirements=task_requirements,
                performance_metrics=performance_metrics,
                execution_time=execution_time,
                resource_usage=resource_usage,
                success=True,
            )

            # 5. 更新增强状态
            fitness_improvement = max(
                0.0,
                result.fitness_score
                - self.enhanced_state.get("best_multi_objective_score", 0.0),
            )

            self.enhanced_state["average_improvement"] = (
                self.enhanced_state["average_improvement"]
                * (self.enhanced_state["total_evolutions"] - 1)
                + fitness_improvement
            ) / self.enhanced_state["total_evolutions"]

            self.enhanced_state["best_multi_objective_score"] = max(
                self.enhanced_state["best_multi_objective_score"],
                result.fitness_score,
            )

            # 6. 学习进度计算
            successful_evolutions = sum(
                1 for usage in self.enhanced_state["algorithm_usage"].values()
            )
            self.enhanced_state["learning_progress"] = min(
                1.0,
                successful_evolutions / max(1, self.enhanced_state["total_evolutions"]),
            )

            logger.info(
                f"增强演化完成，适应度: {result.fitness_score:.4f}, 学习进度: {self.enhanced_state['learning_progress']:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"增强演化失败: {str(e)}")

            # 更新元学习控制器（失败情况）
            execution_time = time.time() - start_time

            self.meta_controller.update_performance(
                algorithm=algorithm,
                task_requirements=task_requirements,
                performance_metrics={"accuracy": 0.0, "efficiency": 0.0},
                execution_time=execution_time,
                resource_usage={
                    "cpu_time": execution_time,
                    "memory_mb": 0.0,
                    "compute_gflops": 0.0,
                },
                success=False,
            )

            # 回退到基础引擎
            logger.warning("回退到基础演化引擎")
            return self.base_engine.evolve_architecture(
                task_requirements, performance_feedback, resource_constraints
            )

    def _evolve_with_multi_objective(
        self,
        task_requirements: Dict[str, Any],
        performance_feedback: Dict[str, float],
        resource_constraints: Optional[Dict[str, float]],
        algorithm_params: Dict[str, Any],
    ) -> NetworkArchitecture:
        """使用多目标优化进行演化"""

        # 获取基础引擎的种群
        base_population = self.base_engine.architecture_searcher.population

        if not base_population:
            # 初始化种群
            self.base_engine.architecture_searcher.initialize_population(
                task_requirements=task_requirements,
                resource_constraints=resource_constraints or {},
            )
            base_population = self.base_engine.architecture_searcher.population

        # 定义目标函数
        objectives = {
            "accuracy": lambda arch: arch.performance_metrics.get("accuracy", 0.0),
            "efficiency": lambda arch: arch.performance_metrics.get("efficiency", 0.0),
            "robustness": lambda arch: arch.performance_metrics.get("robustness", 0.0),
            "adaptability": lambda arch: arch.performance_metrics.get(
                "adaptability", 0.0
            ),
        }

        # 定义约束函数
        constraints = {}
        if resource_constraints:
            if "max_memory_mb" in resource_constraints:
                constraints["memory"] = lambda arch: (
                    arch.resource_requirements.get("memory_mb", 0.0)
                    - resource_constraints["max_memory_mb"]
                )

            if "max_compute_gflops" in resource_constraints:
                constraints["compute"] = lambda arch: (
                    arch.resource_requirements.get("compute_gflops", 0.0)
                    - resource_constraints["max_compute_gflops"]
                )

        # 执行多目标优化
        max_generations = algorithm_params.get("max_generations", 50)

        optimized_population = self.multi_objective_optimizer.optimize(
            population=base_population,
            objectives=objectives,
            constraints=constraints if constraints else None,
            max_generations=max_generations,
        )

        # 选择最佳架构（基于加权目标）
        best_arch = max(
            optimized_population,
            key=lambda arch: self._calculate_weighted_score(arch, performance_feedback),
        )

        # 更新基础引擎的种群
        self.base_engine.architecture_searcher.population = optimized_population
        self.base_engine.architecture_searcher.best_architecture = best_arch

        return best_arch

    def _evolve_with_nas(
        self,
        task_requirements: Dict[str, Any],
        performance_feedback: Optional[Dict[str, float]],
        resource_constraints: Optional[Dict[str, float]],
        algorithm_params: Dict[str, Any],
    ) -> NetworkArchitecture:
        """使用NAS算法进行演化"""
        try:
            # 创建NAS引擎配置
            nas_config = {
                **algorithm_params,
                "task_requirements": task_requirements,
                "performance_feedback": performance_feedback or {},
                "resource_constraints": resource_constraints or {},
            }
            
            # 创建NAS引擎
            from core.optimization.advanced_algorithm_enhancer import create_nas_engine
            nas_engine = create_nas_engine(nas_config)
            
            # 创建代理模型用于NAS搜索
            # 在实际应用中，这里应该根据任务需求创建合适的模型
            proxy_model = self._create_nas_proxy_model(task_requirements)
            
            # 数据集信息（用于NAS搜索）
            dataset_info = {
                "input_dim": algorithm_params.get("search_space", {}).get("input_dim", 32),
                "output_dim": algorithm_params.get("search_space", {}).get("output_dim", 10),
                "num_classes": algorithm_params.get("search_space", {}).get("output_dim", 10),
                "dataset_size": 1000,  # 默认数据集大小
            }
            
            # 约束条件
            constraints = {}
            if resource_constraints:
                constraints = {
                    "max_parameters": resource_constraints.get("max_parameters", 1e6),
                    "max_compute_gflops": resource_constraints.get("max_compute_gflops", 10.0),
                    "max_memory_mb": resource_constraints.get("max_memory_mb", 1024),
                }
            
            # 执行NAS搜索
            nas_result = nas_engine.search_optimal_architecture(
                model=proxy_model,
                dataset_info=dataset_info,
                constraints=constraints,
            )
            
            if not nas_result.success:
                raise ValueError(f"NAS搜索失败: {nas_result.get('error_message', '未知错误')}")
            
            # 将NAS结果转换为NetworkArchitecture格式
            optimal_architecture = self._convert_nas_result_to_network_architecture(
                nas_result, task_requirements, performance_feedback
            )
            
            # 记录NAS搜索统计信息
            self._record_nas_statistics(nas_result, algorithm_params)
            
            logger.info(f"NAS演化完成，架构性能: {optimal_architecture.performance_metrics}")
            
            return optimal_architecture
            
        except Exception as e:
            logger.error(f"NAS演化失败: {str(e)}")
            logger.warning("回退到基础演化引擎进行NAS任务")
            
            # 回退到基础引擎
            return self.base_engine.evolve_architecture(
                task_requirements, performance_feedback, resource_constraints
            )
    
    def _create_nas_proxy_model(self, task_requirements: Dict[str, Any]) -> nn.Module:
        """创建NAS代理模型"""
        if not TORCH_AVAILABLE:
            # PyTorch不可用，创建占位模型
            class ProxyModel:
                def __init__(self, input_dim=32, output_dim=10):
                    self.input_dim = input_dim
                    self.output_dim = output_dim
                    
                def to(self, device):
                    return self
                    
                def parameters(self):
                    return []
                    
                def train(self):
                    pass
                    
                def eval(self):
                    pass
                    
                def __call__(self, x):
                    return x
        
        else:
            # PyTorch可用，创建真正的神经网络模型
            class ProxyModel(nn.Module):
                def __init__(self, input_dim=32, output_dim=10):
                    super().__init__()
                    self.input_dim = input_dim
                    self.output_dim = output_dim
                    self.fc = nn.Linear(input_dim, output_dim)
                    
                def forward(self, x):
                    return self.fc(x)
        
        input_dim = task_requirements.get("input_dim", 32)
        output_dim = task_requirements.get("output_dim", 10)
        
        return ProxyModel(input_dim, output_dim)
    
    def _convert_nas_result_to_network_architecture(
        self,
        nas_result: Dict[str, Any],
        task_requirements: Dict[str, Any],
        performance_feedback: Optional[Dict[str, float]],
    ) -> NetworkArchitecture:
        """将NAS结果转换为NetworkArchitecture"""
        
        # 从NAS结果中提取架构信息
        optimal_arch = nas_result.get("optimal_architecture", {})
        performance_metrics = nas_result.get("performance_metrics", {})
        
        # 创建网络架构描述
        architecture_description = {
            "type": "nas_generated",
            "algorithm": nas_result.get("algorithm_used", "darts"),
            "genotype": optimal_arch.get("genotype", {}),
            "normal_cell": optimal_arch.get("normal_cell", {}),
            "reduce_cell": optimal_arch.get("reduce_cell", {}),
            "num_cells": optimal_arch.get("num_cells", 8),
            "num_nodes": optimal_arch.get("num_nodes", 4),
            "init_channels": optimal_arch.get("init_channels", 16),
        }
        
        # 估算资源需求
        resource_requirements = {
            "compute_gflops": performance_metrics.get("compute_gflops", 1.0),
            "memory_mb": performance_metrics.get("memory_mb", 100.0),
            "parameters": performance_metrics.get("parameters", 100000),
            "inference_latency_ms": performance_metrics.get("inference_latency_ms", 10.0),
        }
        
        # 计算适应度分数
        fitness_score = performance_metrics.get("accuracy", 0.5)
        if performance_feedback:
            # 如果有性能反馈，加权计算适应度
            weighted_score = 0.0
            total_weight = 0.0
            for metric_name, target_value in performance_feedback.items():
                weight = target_value if target_value > 0 else 0.0
                actual_value = performance_metrics.get(metric_name, 0.0)
                satisfaction = min(1.0, actual_value / target_value) if target_value > 0 else 0.0
                weighted_score += satisfaction * weight
                total_weight += weight
            
            if total_weight > 0:
                fitness_score = weighted_score / total_weight
        
        # 创建NetworkArchitecture对象
        architecture = NetworkArchitecture(
            architecture_description=architecture_description,
            performance_metrics=performance_metrics,
            resource_requirements=resource_requirements,
            fitness_score=fitness_score,
            metadata={
                "generated_by": "nas_algorithm",
                "nas_result": nas_result,
                "task_requirements": task_requirements,
                "performance_feedback": performance_feedback,
            }
        )
        
        return architecture
    
    def _record_nas_statistics(self, nas_result: Dict[str, Any], algorithm_params: Dict[str, Any]) -> None:
        """记录NAS统计信息"""
        # 记录NAS特定的统计信息
        nas_stats = {
            "search_duration": nas_result.get("search_duration_seconds", 0.0),
            "optimal_architecture": nas_result.get("optimal_architecture", {}).get("type", "unknown"),
            "algorithm_used": nas_result.get("algorithm_used", "unknown"),
            "performance_metrics": nas_result.get("performance_metrics", {}),
            "constraints_met": nas_result.get("constraints_met", False),
        }
        
        # 可以在这里将统计信息保存到日志或数据库
        logger.info(f"NAS统计: {nas_stats}")

    def _calculate_weighted_score(
        self, architecture: NetworkArchitecture, performance_targets: Dict[str, float]
    ) -> float:
        """计算加权分数"""
        score = 0.0
        total_weight = 0.0

        for target_name, target_value in performance_targets.items():
            # 目标权重（基于目标值的重要性）
            weight = target_value  # 假设目标值越高，重要性越高

            # 实际性能值
            actual_value = architecture.performance_metrics.get(target_name, 0.0)

            # 计算满足程度
            satisfaction = (
                min(1.0, actual_value / target_value) if target_value > 0 else 0.0
            )

            score += satisfaction * weight
            total_weight += weight

        if total_weight > 0:
            return score / total_weight
        else:
            return architecture.fitness_score

    def get_enhanced_status(self) -> Dict[str, Any]:
        """获取增强状态"""
        return {
            **self.enhanced_state,
            "meta_controller_stats": {
                "learned_tasks": len(self.meta_controller.task_algorithm_mapping),
                "performance_records": sum(
                    len(records)
                    for records in self.meta_controller.algorithm_performance.values()
                ),
            },
            "multi_objective_stats": {
                "pareto_fronts": len(
                    self.multi_objective_optimizer.pareto_front_history
                ),
                "last_pareto_size": len(
                    self.multi_objective_optimizer.pareto_front_history[-1]
                )
                if self.multi_objective_optimizer.pareto_front_history
                else 0,
            },
            "base_engine_status": self.base_engine.get_evolution_status(),
        }

    def create_coevolution_group(
        self,
        group_id: str,
        model_ids: List[str],
        collaboration_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """创建协同演化组"""
        if group_id in self.coevolution_groups:
            logger.warning(f"协同演化组 {group_id} 已存在")
            return False

        max_group_size = self.config.get("coevolution_config", {}).get(
            "max_group_size", 5
        )

        if len(model_ids) > max_group_size:
            logger.warning(f"模型数量超过最大组大小 {max_group_size}")
            model_ids = model_ids[:max_group_size]

        self.coevolution_groups[group_id] = model_ids

        logger.info(f"创建协同演化组 {group_id}, 包含 {len(model_ids)} 个模型")
        return True

    def coevolve_architectures(
        self,
        group_id: str,
        task_requirements: Dict[str, Any],
        performance_targets: Dict[str, float],
    ) -> Dict[str, NetworkArchitecture]:
        """协同演化架构"""
        if group_id not in self.coevolution_groups:
            logger.error(f"协同演化组 {group_id} 不存在")
            return {}

        model_ids = self.coevolution_groups[group_id]
        results = {}

        logger.info(f"开始协同演化，组: {group_id}, 模型: {model_ids}")

        # 这里可以实现更复杂的协同演化逻辑
        # 例如：知识共享、协作优化、竞争演化等

        for model_id in model_ids:
            try:
                # 为每个模型执行演化（可以加入协作逻辑）
                result = self.evolve_architecture(
                    task_requirements=task_requirements,
                    performance_feedback=performance_targets,
                    use_multi_objective=True,
                )

                results[model_id] = result

                logger.info(f"模型 {model_id} 协同演化完成，适应度: {result.fitness_score:.4f}")

            except Exception as e:
                logger.error(f"模型 {model_id} 协同演化失败: {str(e)}")

        return results
    
    def perform_neural_architecture_search(
        self,
        model: Any,
        search_space: Dict[str, Any],
        constraints: Dict[str, Any],
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """执行真实的神经架构搜索（NAS）
        
        Args:
            model: 基础模型或搜索空间定义
            search_space: 搜索空间定义
            constraints: 资源约束（参数量、计算量等）
            num_iterations: 搜索迭代次数
            
        Returns:
            NAS搜索结果
        """
        try:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch不可用，无法执行真实NAS")
            
            logger.info(f"开始神经架构搜索，搜索空间: {search_space}")
            
            # 定义可能的层类型和操作
            layer_types = [
                "linear", "conv1d", "conv2d", "conv3d", "lstm", "gru", "attention", "transformer"
            ]
            
            activation_types = ["relu", "sigmoid", "tanh", "gelu", "leaky_relu", "elu", "swish"]
            
            # 初始化最佳架构
            best_architecture = None
            best_performance = float('-inf')
            search_history = []
            
            for iteration in range(num_iterations):
                # 生成随机架构
                architecture = self._generate_random_architecture(
                    search_space, constraints, layer_types, activation_types
                )
                
                # 评估架构性能（使用代理模型或实际训练）
                performance = self._evaluate_architecture_performance(
                    architecture, model, constraints
                )
                
                # 记录搜索结果
                search_result = {
                    "iteration": iteration,
                    "architecture": architecture,
                    "performance": performance,
                    "constraints_satisfied": self._check_constraints(architecture, constraints)
                }
                search_history.append(search_result)
                
                # 更新最佳架构
                if performance > best_performance:
                    best_performance = performance
                    best_architecture = architecture
                    logger.info(f"Iteration {iteration}: 发现新最佳架构，性能: {performance:.4f}")
                
                # 应用演化操作（突变、交叉等）
                if iteration % 10 == 0 and iteration > 0:
                    # 每10次迭代进行一次演化操作
                    architecture = self._apply_evolution_operations(
                        best_architecture, search_space, layer_types, activation_types
                    )
            
            # 返回最佳架构
            return {
                "success": True,
                "best_architecture": best_architecture,
                "best_performance": best_performance,
                "search_history": search_history,
                "total_iterations": num_iterations,
                "constraints": constraints
            }
            
        except Exception as e:
            logger.error(f"神经架构搜索失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "best_architecture": None,
                "best_performance": 0.0
            }
    
    def _generate_random_architecture(
        self,
        search_space: Dict[str, Any],
        constraints: Dict[str, Any],
        layer_types: List[str],
        activation_types: List[str]
    ) -> Dict[str, Any]:
        """生成随机神经网络架构"""
        import random
        
        # 随机确定层数
        min_layers = search_space.get("min_layers", 1)
        max_layers = search_space.get("max_layers", 10)
        num_layers = random.randint(min_layers, max_layers)
        
        layers = []
        total_params = 0
        
        for i in range(num_layers):
            # 随机选择层类型
            layer_type = random.choice(layer_types)
            
            # 生成层配置
            layer_config = {
                "type": layer_type,
                "id": f"layer_{i}",
                "activation": random.choice(activation_types)
            }
            
            # 根据层类型添加特定参数
            if layer_type.startswith("conv"):
                # 卷积层
                layer_config.update({
                    "in_channels": random.choice([1, 3, 16, 32, 64]),
                    "out_channels": random.choice([16, 32, 64, 128, 256]),
                    "kernel_size": random.choice([1, 3, 5, 7]),
                    "stride": random.choice([1, 2]),
                    "padding": random.choice([0, 1, 2])
                })
                # 估算参数量
                params = (layer_config["in_channels"] * layer_config["out_channels"] * 
                         layer_config["kernel_size"] ** (2 if layer_type == "conv2d" else 1))
                
            elif layer_type == "linear":
                # 全连接层
                in_features = random.choice([128, 256, 512, 1024])
                out_features = random.choice([64, 128, 256, 512])
                layer_config.update({
                    "in_features": in_features,
                    "out_features": out_features
                })
                params = in_features * out_features
                
            elif layer_type in ["lstm", "gru"]:
                # 循环神经网络层
                input_size = random.choice([128, 256, 512])
                hidden_size = random.choice([64, 128, 256])
                layer_config.update({
                    "input_size": input_size,
                    "hidden_size": hidden_size,
                    "num_layers": random.choice([1, 2, 3])
                })
                params = (4 if layer_type == "lstm" else 3) * (input_size * hidden_size + hidden_size * hidden_size)
                
            elif layer_type in ["attention", "transformer"]:
                # 注意力或Transformer层
                embed_dim = random.choice([128, 256, 512])
                num_heads = random.choice([4, 8, 16])
                layer_config.update({
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "feedforward_dim": embed_dim * 4
                })
                params = 3 * embed_dim * embed_dim + embed_dim * (embed_dim * 4) * 2
            
            else:
                # 默认层
                params = random.randint(1000, 10000)
            
            # 检查约束
            total_params += params
            if "max_parameters" in constraints and total_params > constraints["max_parameters"]:
                # 如果超过最大参数量，停止添加层
                break
            
            layers.append(layer_config)
        
        # 创建架构定义
        architecture = {
            "layers": layers,
            "total_params": total_params,
            "num_layers": len(layers),
            "input_dim": search_space.get("input_dim", 512),
            "output_dim": search_space.get("output_dim", 10),
            "architecture_id": f"nas_arch_{int(time.time())}_{random.randint(1000, 9999)}"
        }
        
        return architecture
    
    def _evaluate_architecture_performance(
        self,
        architecture: Dict[str, Any],
        base_model: Any,
        constraints: Dict[str, Any]
    ) -> float:
        """评估架构性能
        
        使用代理模型或简单启发式方法评估架构性能
        在实际NAS中，这里应该使用实际的训练和验证
        """
        import random
        
        # 基础性能分数
        performance = 0.0
        
        # 1. 基于层数的分数（适中最好）
        num_layers = len(architecture.get("layers", []))
        optimal_layers = constraints.get("optimal_layers", 5)
        layer_score = 1.0 - min(abs(num_layers - optimal_layers) / optimal_layers, 1.0)
        performance += layer_score * 0.3
        
        # 2. 基于参数量的分数（越少越好，但要有一定量）
        total_params = architecture.get("total_params", 0)
        max_params = constraints.get("max_parameters", 1000000)
        if max_params > 0:
            param_efficiency = 1.0 - min(total_params / max_params, 1.0)
            performance += param_efficiency * 0.3
        
        # 3. 基于层多样性的分数
        layers = architecture.get("layers", [])
        layer_types = [layer.get("type", "unknown") for layer in layers]
        unique_types = len(set(layer_types))
        diversity_score = unique_types / max(len(layer_types), 1)
        performance += diversity_score * 0.2
        
        # 4. 随机噪声（模拟实际评估的不确定性）
        noise = random.uniform(-0.1, 0.1)
        performance += noise
        
        # 确保分数在合理范围内
        performance = max(0.0, min(1.0, performance))
        
        return performance
    
    def _check_constraints(
        self,
        architecture: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> bool:
        """检查架构是否满足约束"""
        # 检查参数量约束
        if "max_parameters" in constraints:
            total_params = architecture.get("total_params", 0)
            if total_params > constraints["max_parameters"]:
                return False
        
        # 检查层数约束
        if "max_layers" in constraints:
            num_layers = len(architecture.get("layers", []))
            if num_layers > constraints["max_layers"]:
                return False
        
        # 检查特定层类型约束
        if "allowed_layer_types" in constraints:
            layers = architecture.get("layers", [])
            for layer in layers:
                layer_type = layer.get("type", "")
                if layer_type not in constraints["allowed_layer_types"]:
                    return False
        
        return True
    
    def _apply_evolution_operations(
        self,
        architecture: Dict[str, Any],
        search_space: Dict[str, Any],
        layer_types: List[str],
        activation_types: List[str]
    ) -> Dict[str, Any]:
        """对架构应用演化操作
        
        包括：添加层、删除层、修改层参数、突变层类型等
        """
        import random
        import copy
        
        # 深拷贝架构以避免修改原架构
        mutated = copy.deepcopy(architecture)
        layers = mutated.get("layers", [])
        
        if not layers:
            return mutated
        
        # 随机选择演化操作
        operation = random.choice(["add_layer", "remove_layer", "mutate_layer", "swap_layers"])
        
        if operation == "add_layer" and len(layers) < search_space.get("max_layers", 20):
            # 添加新层
            new_layer_idx = random.randint(0, len(layers))
            layer_type = random.choice(layer_types)
            
            new_layer = {
                "type": layer_type,
                "id": f"layer_added_{int(time.time())}",
                "activation": random.choice(activation_types)
            }
            
            # 简单配置新层
            if layer_type == "linear":
                new_layer.update({
                    "in_features": 256,
                    "out_features": 128
                })
            elif layer_type.startswith("conv"):
                new_layer.update({
                    "in_channels": 32,
                    "out_channels": 64,
                    "kernel_size": 3
                })
            
            layers.insert(new_layer_idx, new_layer)
            mutated["num_layers"] = len(layers)
            
        elif operation == "remove_layer" and len(layers) > 1:
            # 删除层
            remove_idx = random.randint(0, len(layers) - 1)
            layers.pop(remove_idx)
            mutated["num_layers"] = len(layers)
            
        elif operation == "mutate_layer":
            # 突变现有层
            mutate_idx = random.randint(0, len(layers) - 1)
            layer = layers[mutate_idx]
            
            # 随机改变层的一个属性
            if "out_channels" in layer:
                layer["out_channels"] = max(1, layer["out_channels"] + random.choice([-16, -8, 8, 16]))
            elif "out_features" in layer:
                layer["out_features"] = max(1, layer["out_features"] + random.choice([-32, -16, 16, 32]))
            elif "hidden_size" in layer:
                layer["hidden_size"] = max(1, layer["hidden_size"] + random.choice([-32, -16, 16, 32]))
        
        elif operation == "swap_layers" and len(layers) >= 2:
            # 交换层顺序
            idx1, idx2 = random.sample(range(len(layers)), 2)
            layers[idx1], layers[idx2] = layers[idx2], layers[idx1]
        
        mutated["layers"] = layers
        return mutated
    
    def add_layer_to_model(
        self,
        model: Any,
        layer_type: str,
        layer_config: Dict[str, Any]
    ) -> Any:
        """向模型中添加新层（真实NAS操作）
        
        Args:
            model: PyTorch模型
            layer_type: 层类型（'linear', 'conv2d', 'lstm'等）
            layer_config: 层配置参数
            
        Returns:
            修改后的模型
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch不可用，无法添加层")
            return model
        
        try:
            import torch.nn as nn
            
            # 创建新层
            if layer_type == "linear":
                new_layer = nn.Linear(
                    layer_config.get("in_features", 512),
                    layer_config.get("out_features", 256)
                )
            elif layer_type == "conv2d":
                new_layer = nn.Conv2d(
                    layer_config.get("in_channels", 3),
                    layer_config.get("out_channels", 64),
                    kernel_size=layer_config.get("kernel_size", 3),
                    stride=layer_config.get("stride", 1),
                    padding=layer_config.get("padding", 1)
                )
            elif layer_type == "lstm":
                new_layer = nn.LSTM(
                    input_size=layer_config.get("input_size", 512),
                    hidden_size=layer_config.get("hidden_size", 256),
                    num_layers=layer_config.get("num_layers", 1),
                    batch_first=layer_config.get("batch_first", True)
                )
            elif layer_type == "attention":
                # 简化版注意力层
                class SimpleAttention(nn.Module):
                    def __init__(self, embed_dim, num_heads):
                        super().__init__()
                        self.embed_dim = embed_dim
                        self.num_heads = num_heads
                        self.q_proj = nn.Linear(embed_dim, embed_dim)
                        self.k_proj = nn.Linear(embed_dim, embed_dim)
                        self.v_proj = nn.Linear(embed_dim, embed_dim)
                        self.out_proj = nn.Linear(embed_dim, embed_dim)
                        
                    def forward(self, x):
                        # 简化实现
                        return self.out_proj(x)
                
                new_layer = SimpleAttention(
                    embed_dim=layer_config.get("embed_dim", 512),
                    num_heads=layer_config.get("num_heads", 8)
                )
            else:
                logger.warning(f"不支持的层类型: {layer_type}")
                return model
            
            # 获取模型的模块列表
            if isinstance(model, nn.Module):
                # 如果模型是nn.Module，创建一个新的Sequential模块
                if hasattr(model, 'layers') and isinstance(model.layers, nn.ModuleList):
                    model.layers.append(new_layer)
                elif hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
                    model.features.add_module(f"nas_layer_{int(time.time())}", new_layer)
                else:
                    # 创建一个新的Sequential模型
                    logger.warning("模型结构不支持直接添加层，返回原始模型")
            else:
                logger.warning("模型不是PyTorch nn.Module，无法添加层")
            
            return model
            
        except Exception as e:
            logger.error(f"添加层失败: {str(e)}")
            return model
    
    def remove_layer_from_model(
        self,
        model: Any,
        layer_index: int
    ) -> Any:
        """从模型中移除层（真实NAS操作）
        
        Args:
            model: PyTorch模型
            layer_index: 要移除的层索引
            
        Returns:
            修改后的模型
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch不可用，无法移除层")
            return model
        
        try:
            if isinstance(model, nn.Module):
                if hasattr(model, 'layers') and isinstance(model.layers, nn.ModuleList):
                    if 0 <= layer_index < len(model.layers):
                        model.layers.pop(layer_index)
                        logger.info(f"成功移除层 {layer_index}")
                    else:
                        logger.warning(f"层索引 {layer_index} 超出范围")
                elif hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
                    # 对于Sequential模型，需要重新构建
                    layers = list(model.features.children())
                    if 0 <= layer_index < len(layers):
                        layers.pop(layer_index)
                        model.features = nn.Sequential(*layers)
                        logger.info(f"成功移除层 {layer_index}")
                    else:
                        logger.warning(f"层索引 {layer_index} 超出范围")
                else:
                    logger.warning("模型结构不支持直接移除层")
            else:
                logger.warning("模型不是PyTorch nn.Module，无法移除层")
            
            return model
            
        except Exception as e:
            logger.error(f"移除层失败: {str(e)}")
            return model


# 工厂函数
def create_enhanced_evolution_engine(
    config: Optional[Dict[str, Any]] = None
) -> EnhancedEvolutionEngine:
    """创建增强演化引擎实例"""
    return EnhancedEvolutionEngine(config)


if __name__ == "__main__":
    # 演示增强演化引擎
    print("=" * 80)
    print("增强演化引擎演示")
    print("=" * 80)

    try:
        # 创建增强引擎
        engine = create_enhanced_evolution_engine()

        # 获取初始状态
        status = engine.get_enhanced_status()
        print(f"\n1. 初始状态:")
        print(f"   总演化次数: {status['total_evolutions']}")
        print(f"   学习进度: {status['learning_progress']:.2f}")

        # 执行演化任务
        print("\n2. 执行演化任务...")
        task_requirements = {
            "description": "多模态情感分析",
            "modalities": ["text", "audio", "image"],
            "architecture_type": "multimodal",
        }

        performance_targets = {
            "accuracy": 0.85,
            "efficiency": 0.75,
            "robustness": 0.8,
        }

        result = engine.evolve_architecture(
            task_requirements=task_requirements,
            performance_feedback=performance_targets,
            use_multi_objective=True,
        )

        print(f"   演化完成，最佳架构: {result.architecture_id}")
        print(f"   适应度分数: {result.fitness_score:.4f}")
        print(f"   层数: {len(result.layers)}")
        print(f"   性能指标: {result.performance_metrics}")

        # 获取更新后的状态
        status = engine.get_enhanced_status()
        print(f"\n3. 更新后状态:")
        print(f"   总演化次数: {status['total_evolutions']}")
        print(f"   学习进度: {status['learning_progress']:.2f}")
        print(f"   算法使用: {dict(status['algorithm_usage'])}")

        print("\n✅ 增强演化引擎演示成功")

    except Exception as e:
        print(f"\n❌ 增强演化引擎演示失败: {str(e)}")
        import traceback

        traceback.print_exc()
