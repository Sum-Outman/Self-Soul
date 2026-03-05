#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal Reasoning Planner - 时间推理规划器

核心功能：
1. 时间关系分析和表示
2. 时间约束处理和优化
3. 时间感知的计划生成
4. 动态时间调整和调度
5. 时间不确定性处理
6. 时间资源协调

基于时间逻辑、时序规划和调度算法实现时间感知的智能规划。

Copyright (c) 2025 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
import json
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import networkx as nx
import numpy as np

# 导入错误处理
from core.error_handling import ErrorHandler

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()

class TemporalRelation(Enum):
    """时间关系枚举"""
    BEFORE = "before"            # A在B之前
    AFTER = "after"              # A在B之后
    DURING = "during"            # A在B期间
    OVERLAPS = "overlaps"        # A与B重叠
    MEETS = "meets"              # A紧接着B开始
    MET_BY = "met_by"            # B紧接着A开始
    STARTS = "starts"            # A与B同时开始
    FINISHES = "finishes"        # A与B同时结束
    EQUALS = "equals"            # A与B时间相等
    CONTAINS = "contains"        # A包含B

class TimeUnit(Enum):
    """时间单位枚举"""
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"

class TemporalConstraintType(Enum):
    """时间约束类型枚举"""
    DEADLINE = "deadline"                # 截止时间
    DURATION = "duration"                # 持续时间
    START_TIME = "start_time"            # 开始时间
    END_TIME = "end_time"                # 结束时间
    INTERVAL = "interval"                # 时间间隔
    PRECEDENCE = "precedence"            # 先后顺序
    SIMULTANEOUS = "simultaneous"        # 同时发生
    NON_OVERLAP = "non_overlap"          # 不重叠
    RESOURCE_AVAILABILITY = "resource_availability"  # 资源可用时间

class TemporalUncertainty(Enum):
    """时间不确定性级别"""
    EXACT = "exact"              # 精确时间
    APPROXIMATE = "approximate"  # 近似时间
    ESTIMATED = "estimated"      # 估计时间
    UNCERTAIN = "uncertain"      # 不确定时间
    VARIABLE = "variable"        # 可变时间

class TemporalReasoningPlanner:
    """
    时间推理规划器 - 提供时间感知的规划和推理
    
    核心特性：
    1. 时间关系分析和推理
    2. 时间约束满足和优化
    3. 动态时间调度和调整
    4. 时间不确定性处理
    5. 时间资源协调和分配
    6. 时间敏感的计划生成
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化时间推理规划器
        
        Args:
            config: 配置字典
        """
        self.logger = logger
        self.config = config or self._get_default_config()
        
        # 时间图和模型
        self.temporal_graphs = defaultdict(nx.DiGraph)  # 时间图集合
        self.schedule_models = {}                       # 调度模型
        self.time_windows = defaultdict(list)           # 时间窗口
        
        # 时间推理状态
        self.temporal_reasoning_state = {
            "total_plans_generated": 0,
            "temporal_constraints_processed": 0,
            "schedule_optimizations": 0,
            "temporal_conflicts_resolved": 0,
            "uncertainty_handled_count": 0
        }
        
        # 时间知识库
        self.temporal_knowledge = {
            "duration_estimates": defaultdict(dict),
            "time_patterns": [],
            "temporal_heuristics": [],
            "historical_schedules": []
        }
        
        # 初始化组件
        self._initialize_components()
        
        logger.info("时间推理规划器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "temporal_modeling": {
                "enable_temporal_graphs": True,
                "enable_time_windows": True,
                "max_time_horizon": 365,  # 天
                "time_granularity": "minutes"
            },
            "constraint_handling": {
                "enable_constraint_propagation": True,
                "enable_constraint_optimization": True,
                "max_constraint_iterations": 100,
                "constraint_relaxation_enabled": True
            },
            "scheduling": {
                "enable_optimal_scheduling": True,
                "scheduling_algorithm": "priority_based",  # priority, genetic, simulated_annealing
                "enable_parallel_scheduling": True,
                "max_scheduling_iterations": 50
            },
            "uncertainty_handling": {
                "enable_uncertainty_modeling": True,
                "uncertainty_method": "fuzzy_intervals",  # fuzzy, probabilistic, robust
                "robustness_level": 0.8,
                "contingency_planning": True
            },
            "optimization": {
                "enable_temporal_optimization": True,
                "optimization_objectives": ["minimize_makespan", "maximize_resource_utilization"],
                "optimization_method": "multi_objective",
                "learning_enabled": True
            }
        }
    
    def _initialize_components(self) -> None:
        """初始化组件"""
        try:
            # 初始化时间关系分析器
            self._initialize_temporal_relation_analyzer()
            
            # 初始化时间约束处理器
            self._initialize_constraint_processor()
            
            # 初始化调度器
            self._initialize_scheduler()
            
            # 初始化不确定性处理器
            if self.config["uncertainty_handling"]["enable_uncertainty_modeling"]:
                self._initialize_uncertainty_handler()
            
            # 初始化时间优化器
            if self.config["optimization"]["enable_temporal_optimization"]:
                self._initialize_temporal_optimizer()
            
            logger.info("时间推理组件初始化完成")
            
        except Exception as e:
            error_handler.handle_error(e, "TemporalReasoningPlanner", "初始化组件失败")
            logger.error(f"组件初始化失败: {e}")
    
    def _initialize_temporal_relation_analyzer(self) -> None:
        """初始化时间关系分析器"""
        self.relation_analyzer = {
            "relation_types": [rel.value for rel in TemporalRelation],
            "inference_rules": self._create_temporal_inference_rules(),
            "consistency_checker": self._create_consistency_checker()
        }
    
    def _create_temporal_inference_rules(self) -> List[Dict[str, Any]]:
        """创建时间推理规则"""
        rules = [
            {
                "name": "transitivity_before",
                "premises": [("A", "before", "B"), ("B", "before", "C")],
                "conclusion": ("A", "before", "C"),
                "confidence": 1.0
            },
            {
                "name": "transitivity_during",
                "premises": [("A", "during", "B"), ("B", "during", "C")],
                "conclusion": ("A", "during", "C"),
                "confidence": 1.0
            },
            {
                "name": "during_implies_before_end",
                "premises": [("A", "during", "B")],
                "conclusion": ("A", "before", ("end", "B")),
                "confidence": 0.9
            },
            {
                "name": "meets_implies_before",
                "premises": [("A", "meets", "B")],
                "conclusion": ("A", "before", "B"),
                "confidence": 1.0
            },
            {
                "name": "overlaps_implies_concurrent",
                "premises": [("A", "overlaps", "B")],
                "conclusion": ("concurrent", "A", "B"),
                "confidence": 0.8
            }
        ]
        
        return rules
    
    def _create_consistency_checker(self) -> Dict[str, Any]:
        """创建一致性检查器"""
        inconsistent_patterns = [
            ("A", "before", "B"), ("B", "before", "A"),  # 矛盾的前后关系
            ("A", "equals", "B"), ("A", "before", "B"),  # 相等但又前后矛盾
            ("A", "during", "B"), ("B", "during", "A"),  # 相互包含矛盾
        ]
        
        return {
            "inconsistent_patterns": inconsistent_patterns,
            "check_method": "pattern_matching",
            "repair_strategies": ["relax_constraints", "reorder_events", "adjust_durations"]
        }
    
    def _initialize_constraint_processor(self) -> None:
        """初始化约束处理器"""
        self.constraint_processor = {
            "constraint_types": [ct.value for ct in TemporalConstraintType],
            "propagation_methods": ["arc_consistency", "path_consistency", "global_consistency"],
            "satisfaction_algorithms": ["backtracking", "constraint_satisfaction", "linear_programming"],
            "relaxation_strategies": ["priority_based", "cost_based", "utility_based"]
        }
    
    def _initialize_scheduler(self) -> None:
        """初始化调度器"""
        class PriorityBasedScheduler:
            def __init__(self):
                self.tasks = []
                self.resources = []
            
            def schedule(self, tasks, resources, constraints):
                # 简化实现：基于优先级调度
                scheduled = []
                current_time = 0
                
                # 按优先级排序任务
                sorted_tasks = sorted(tasks, key=lambda t: t.get("priority", 0), reverse=True)
                
                for task in sorted_tasks:
                    duration = task.get("duration", 1)
                    scheduled.append({
                        "task": task["id"],
                        "start": current_time,
                        "end": current_time + duration,
                        "duration": duration
                    })
                    current_time += duration
                
                return scheduled
        
        self.schedule_models["priority_based"] = PriorityBasedScheduler()
    
    def _initialize_uncertainty_handler(self) -> None:
        """初始化不确定性处理器"""
        self.uncertainty_handler = {
            "modeling_methods": ["fuzzy_sets", "probability_distributions", "interval_arithmetic", "scenario_based"],
            "robustness_metrics": ["makespan_robustness", "resource_robustness", "constraint_robustness"],
            "contingency_plans": []
        }
    
    def _initialize_temporal_optimizer(self) -> None:
        """初始化时间优化器"""
        self.temporal_optimizer = {
            "optimization_methods": ["genetic_algorithm", "simulated_annealing", "linear_programming", "heuristic_search"],
            "objective_functions": [
                "minimize_makespan",
                "maximize_resource_utilization", 
                "minimize_idle_time",
                "maximize_concurrency",
                "minimize_tardiness"
            ],
            "optimization_constraints": ["resource_constraints", "precedence_constraints", "time_windows"]
        }
    
    def analyze_temporal_aspects(self, plan: Dict[str, Any],
                                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        分析计划的时间方面
        
        Args:
            plan: 要分析的计划
            context: 上下文信息
            
        Returns:
            时间分析结果
        """
        start_time = time.time()
        
        try:
            # 1. 提取时间特征
            temporal_features = self._extract_temporal_features(plan, context)
            
            # 2. 识别时间约束
            temporal_constraints = self._identify_temporal_constraints(plan, temporal_features)
            
            # 3. 分析时间关系
            temporal_relations = self._analyze_temporal_relations(plan, temporal_features)
            
            # 4. 评估时间复杂性
            temporal_complexity = self._assess_temporal_complexity(plan, temporal_features, temporal_constraints)
            
            # 5. 识别时间挑战
            temporal_challenges = self._identify_temporal_challenges(plan, temporal_features, temporal_constraints)
            
            # 6. 生成时间优化机会
            optimization_opportunities = self._identify_temporal_optimization_opportunities(
                plan, temporal_features, temporal_constraints
            )
            
            result = {
                "success": True,
                "temporal_features": temporal_features,
                "temporal_constraints": temporal_constraints,
                "temporal_relations": temporal_relations,
                "temporal_complexity": temporal_complexity,
                "temporal_challenges": temporal_challenges,
                "optimization_opportunities": optimization_opportunities,
                "performance_metrics": {
                    "analysis_time": time.time() - start_time,
                    "feature_count": len(temporal_features),
                    "constraint_count": len(temporal_constraints),
                    "relation_count": len(temporal_relations)
                }
            }
            
            self.temporal_reasoning_state["total_plans_generated"] += 1
            
            logger.info(f"时间分析完成: 计划ID={plan.get('id', 'unknown')}, 约束数={len(temporal_constraints)}")
            
            return result
            
        except Exception as e:
            error_handler.handle_error(e, "TemporalReasoningPlanner", "时间分析失败")
            return {
                "success": False,
                "error": str(e),
                "partial_results": {
                    "temporal_features": {},
                    "temporal_constraints": [],
                    "temporal_relations": []
                }
            }
    
    def _extract_temporal_features(self, plan: Dict[str, Any],
                                  context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """从计划中提取时间特征"""
        features = {
            "temporal_dimensions": {},
            "time_estimates": {},
            "scheduling_patterns": [],
            "time_sensitivities": {}
        }
        
        # 从计划步骤中提取
        steps = plan.get("steps", [])
        total_duration = 0
        step_durations = []
        
        for i, step in enumerate(steps):
            step_id = f"step_{i}"
            duration = step.get("estimated_time", 0)
            
            features["time_estimates"][step_id] = {
                "duration": duration,
                "unit": "minutes",
                "uncertainty": self._estimate_time_uncertainty(duration),
                "flexibility": self._assess_time_flexibility(step)
            }
            
            step_durations.append(duration)
            total_duration += duration
        
        features["temporal_dimensions"]["total_duration"] = total_duration
        features["temporal_dimensions"]["step_count"] = len(steps)
        features["temporal_dimensions"]["avg_duration"] = total_duration / max(1, len(steps))
        features["temporal_dimensions"]["duration_variance"] = np.var(step_durations) if step_durations else 0
        
        # 从依赖关系中提取
        dependencies = plan.get("dependencies", {})
        if dependencies:
            features["scheduling_patterns"].append({
                "type": "dependency_based",
                "dependency_count": len(dependencies),
                "critical_path_potential": len(dependencies) / max(1, len(steps))
            })
        
        # 从资源中提取
        resources = plan.get("resource_requirements", {})
        if resources:
            features["time_sensitivities"]["resource_dependent"] = len(resources) > 0
        
        # 从上下文中提取
        if context:
            context_features = self._extract_context_temporal_features(context)
            features.update(context_features)
        
        # 识别时间模式
        features["temporal_patterns"] = self._identify_temporal_patterns(plan, features)
        
        return features
    
    def _estimate_time_uncertainty(self, duration: float) -> TemporalUncertainty:
        """估计时间不确定性"""
        if duration <= 0:
            return TemporalUncertainty.UNCERTAIN
        
        if duration <= 5:
            return TemporalUncertainty.EXACT
        elif duration <= 15:
            return TemporalUncertainty.APPROXIMATE
        elif duration <= 30:
            return TemporalUncertainty.ESTIMATED
        elif duration <= 60:
            return TemporalUncertainty.UNCERTAIN
        else:
            return TemporalUncertainty.VARIABLE
    
    def _assess_time_flexibility(self, step: Dict[str, Any]) -> float:
        """评估时间灵活性"""
        step_type = step.get("type", "")
        
        # 不同类型步骤的灵活性
        flexibility_scores = {
            "analysis": 0.8,      # 分析步骤通常较灵活
            "evaluation": 0.7,    # 评估步骤较灵活
            "decision": 0.6,      # 决策步骤有一定灵活性
            "generation": 0.5,    # 生成步骤灵活性中等
            "execution": 0.3      # 执行步骤通常较固定
        }
        
        return flexibility_scores.get(step_type, 0.5)
    
    def _extract_context_temporal_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """从上下文中提取时间特征"""
        features = {}
        
        # 检查时间相关上下文
        time_keywords = ["time", "deadline", "duration", "schedule", "timeline", "window"]
        
        for key, value in context.items():
            key_lower = key.lower()
            
            if any(keyword in key_lower for keyword in time_keywords):
                features[f"context_{key}"] = {
                    "value": value,
                    "temporal_relevance": self._assess_temporal_relevance(key, value),
                    "constraint_potential": self._assess_constraint_potential(key, value)
                }
        
        return features
    
    def _assess_temporal_relevance(self, key: str, value: Any) -> float:
        """评估时间相关性"""
        key_lower = key.lower()
        
        if "deadline" in key_lower or "due" in key_lower:
            return 0.9
        elif "time" in key_lower or "duration" in key_lower:
            return 0.7
        elif "schedule" in key_lower or "timeline" in key_lower:
            return 0.6
        elif "window" in key_lower or "interval" in key_lower:
            return 0.5
        else:
            return 0.3
    
    def _assess_constraint_potential(self, key: str, value: Any) -> float:
        """评估约束潜力"""
        if isinstance(value, (int, float)):
            # 数值可能表示持续时间或截止时间
            return 0.6
        elif isinstance(value, str):
            # 字符串可能包含时间信息
            time_patterns = ["days", "hours", "minutes", "weeks", "months"]
            if any(pattern in value.lower() for pattern in time_patterns):
                return 0.7
            else:
                return 0.4
        elif isinstance(value, dict) and any(k in value for k in ["start", "end", "duration"]):
            # 字典包含时间结构
            return 0.8
        else:
            return 0.3
    
    def _identify_temporal_patterns(self, plan: Dict[str, Any],
                                   features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别时间模式"""
        patterns = []
        
        steps = plan.get("steps", [])
        if not steps:
            return patterns
        
        # 1. 检查序列模式
        if len(steps) >= 3:
            durations = [step.get("estimated_time", 0) for step in steps]
            
            # 检查递增/递减模式
            if all(durations[i] <= durations[i+1] for i in range(len(durations)-1)):
                patterns.append({
                    "type": "increasing_duration",
                    "description": "步骤持续时间递增",
                    "confidence": 0.7,
                    "implication": "后期步骤可能需要更多时间"
                })
            elif all(durations[i] >= durations[i+1] for i in range(len(durations)-1)):
                patterns.append({
                    "type": "decreasing_duration",
                    "description": "步骤持续时间递减",
                    "confidence": 0.7,
                    "implication": "早期步骤可能需要更多时间"
                })
        
        # 2. 检查依赖模式
        dependencies = plan.get("dependencies", {})
        if dependencies:
            dep_count = len(dependencies)
            step_count = len(steps)
            
            if dep_count >= step_count * 0.7:
                patterns.append({
                    "type": "high_dependency",
                    "description": "高依赖计划",
                    "confidence": 0.8,
                    "implication": "计划可能有严格的时间顺序"
                })
            elif dep_count <= step_count * 0.3:
                patterns.append({
                    "type": "low_dependency",
                    "description": "低依赖计划",
                    "confidence": 0.7,
                    "implication": "计划可能有并行执行机会"
                })
        
        # 3. 检查资源模式
        resources = plan.get("resource_requirements", {})
        if resources:
            resource_count = len(resources)
            if resource_count >= 3:
                patterns.append({
                    "type": "resource_intensive",
                    "description": "资源密集型计划",
                    "confidence": 0.6,
                    "implication": "可能需要资源调度和协调"
                })
        
        return patterns
    
    def _identify_temporal_constraints(self, plan: Dict[str, Any],
                                      features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别时间约束"""
        constraints = []
        
        # 1. 持续时间约束
        total_duration = features.get("temporal_dimensions", {}).get("total_duration", 0)
        if total_duration > 0:
            constraints.append({
                "type": TemporalConstraintType.DURATION.value,
                "target": "total_plan",
                "value": total_duration,
                "unit": "minutes",
                "strictness": 0.7,
                "description": f"计划总持续时间: {total_duration}分钟"
            })
        
        # 2. 步骤持续时间约束
        steps = plan.get("steps", [])
        for i, step in enumerate(steps):
            duration = step.get("estimated_time", 0)
            if duration > 0:
                constraints.append({
                    "type": TemporalConstraintType.DURATION.value,
                    "target": f"step_{i}",
                    "value": duration,
                    "unit": "minutes",
                    "strictness": self._assess_duration_strictness(step),
                    "description": f"步骤{i+1}持续时间: {duration}分钟"
                })
        
        # 3. 依赖约束（先后顺序）
        dependencies = plan.get("dependencies", {})
        for step_id, deps in dependencies.items():
            for dep in deps:
                constraints.append({
                    "type": TemporalConstraintType.PRECEDENCE.value,
                    "target": dep,
                    "dependent": step_id,
                    "relation": "before",
                    "strictness": 0.9,
                    "description": f"{dep}必须在{step_id}之前完成"
                })
        
        # 4. 从上下文中提取约束
        context_constraints = self._extract_context_constraints(features)
        constraints.extend(context_constraints)
        
        # 5. 隐含约束
        implied_constraints = self._infer_implied_constraints(plan, features, constraints)
        constraints.extend(implied_constraints)
        
        return constraints
    
    def _assess_duration_strictness(self, step: Dict[str, Any]) -> float:
        """评估持续时间严格性"""
        step_type = step.get("type", "")
        
        # 不同类型步骤的严格性
        strictness_scores = {
            "execution": 0.8,     # 执行步骤通常较严格
            "decision": 0.7,      # 决策步骤有一定严格性
            "generation": 0.6,    # 生成步骤中等严格
            "evaluation": 0.5,    # 评估步骤较灵活
            "analysis": 0.4       # 分析步骤最灵活
        }
        
        return strictness_scores.get(step_type, 0.5)
    
    def _extract_context_constraints(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从上下文中提取约束"""
        constraints = []
        
        context_features = features.get("context_features", {})
        for key, feature_info in context_features.items():
            relevance = feature_info.get("temporal_relevance", 0)
            constraint_potential = feature_info.get("constraint_potential", 0)
            value = feature_info.get("value")
            
            if relevance > 0.5 and constraint_potential > 0.5:
                constraint_type = self._determine_constraint_type(key, value)
                
                if constraint_type:
                    constraints.append({
                        "type": constraint_type.value,
                        "source": "context",
                        "key": key,
                        "value": value,
                        "strictness": min(0.9, relevance * constraint_potential),
                        "description": f"上下文约束: {key}={value}"
                    })
        
        return constraints
    
    def _determine_constraint_type(self, key: str, value: Any) -> Optional[TemporalConstraintType]:
        """确定约束类型"""
        key_lower = key.lower()
        
        if "deadline" in key_lower or "due" in key_lower:
            return TemporalConstraintType.DEADLINE
        elif "duration" in key_lower:
            return TemporalConstraintType.DURATION
        elif "start" in key_lower:
            return TemporalConstraintType.START_TIME
        elif "end" in key_lower:
            return TemporalConstraintType.END_TIME
        elif "window" in key_lower or "interval" in key_lower:
            return TemporalConstraintType.INTERVAL
        elif "schedule" in key_lower:
            return TemporalConstraintType.PRECEDENCE
        else:
            return None
    
    def _infer_implied_constraints(self, plan: Dict[str, Any],
                                  features: Dict[str, Any],
                                  explicit_constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """推断隐含约束"""
        implied_constraints = []
        
        steps = plan.get("steps", [])
        if len(steps) < 2:
            return implied_constraints
        
        # 1. 资源约束推断
        resources = plan.get("resource_requirements", {})
        if resources:
            # 检查哪些步骤使用相同资源
            resource_to_steps = defaultdict(list)
            
            for i, step in enumerate(steps):
                step_resources = step.get("resources", [])
                for resource in step_resources:
                    resource_to_steps[resource].append(i)
            
            # 为使用相同资源的步骤添加非重叠约束
            for resource, step_indices in resource_to_steps.items():
                if len(step_indices) >= 2:
                    for i in range(len(step_indices)):
                        for j in range(i + 1, len(step_indices)):
                            implied_constraints.append({
                                "type": TemporalConstraintType.NON_OVERLAP.value,
                                "targets": [f"step_{step_indices[i]}", f"step_{step_indices[j]}"],
                                "resource": resource,
                                "strictness": 0.6,
                                "description": f"步骤{step_indices[i]+1}和步骤{step_indices[j]+1}共享资源'{resource}'，不能同时进行"
                            })
        
        # 2. 时间窗口约束推断
        total_duration = features.get("temporal_dimensions", {}).get("total_duration", 0)
        if total_duration > 60:  # 超过1小时的计划
            # 推断可能需要休息或分阶段
            implied_constraints.append({
                "type": TemporalConstraintType.INTERVAL.value,
                "target": "plan_segments",
                "value": {"segment_duration": 45, "break_duration": 15},
                "strictness": 0.4,
                "description": "长时间计划建议分阶段进行，每45分钟休息15分钟"
            })
        
        return implied_constraints
    
    def _analyze_temporal_relations(self, plan: Dict[str, Any],
                                   features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析时间关系"""
        relations = []
        
        steps = plan.get("steps", [])
        if len(steps) < 2:
            return relations
        
        # 分析步骤间的时间关系
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                relation = self._determine_step_relation(i, j, plan, features)
                if relation:
                    relations.append(relation)
        
        # 分析整体时间结构
        overall_relation = self._analyze_overall_temporal_structure(plan, features)
        if overall_relation:
            relations.append(overall_relation)
        
        return relations
    
    def _determine_step_relation(self, i: int, j: int, plan: Dict[str, Any],
                                features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """确定步骤间的时间关系"""
        steps = plan.get("steps", [])
        dependencies = plan.get("dependencies", {})
        
        step_i_id = f"step_{i}"
        step_j_id = f"step_{j}"
        
        # 检查依赖关系
        if step_j_id in dependencies.get(step_i_id, []):
            return {
                "type": TemporalRelation.BEFORE.value,
                "from": step_i_id,
                "to": step_j_id,
                "confidence": 0.9,
                "basis": "explicit_dependency",
                "description": f"{step_i_id}在{step_j_id}之前（显式依赖）"
            }
        elif step_i_id in dependencies.get(step_j_id, []):
            return {
                "type": TemporalRelation.AFTER.value,
                "from": step_i_id,
                "to": step_j_id,
                "confidence": 0.9,
                "basis": "explicit_dependency",
                "description": f"{step_i_id}在{step_j_id}之后（显式依赖）"
            }
        
        # 检查资源使用
        resources_i = steps[i].get("resources", [])
        resources_j = steps[j].get("resources", [])
        
        common_resources = set(resources_i) & set(resources_j)
        if common_resources:
            # 共享资源可能暗示非重叠关系
            return {
                "type": TemporalRelation.NON_OVERLAP.value,
                "from": step_i_id,
                "to": step_j_id,
                "confidence": 0.7,
                "basis": "shared_resource",
                "description": f"{step_i_id}和{step_j_id}共享资源{list(common_resources)}，可能不能同时进行"
            }
        
        # 基于步骤类型的推断
        type_i = steps[i].get("type", "")
        type_j = steps[j].get("type", "")
        
        # 常见顺序模式
        common_patterns = [
            (["analysis"], ["generation"], TemporalRelation.BEFORE, 0.8),
            (["generation"], ["evaluation"], TemporalRelation.BEFORE, 0.8),
            (["evaluation"], ["decision"], TemporalRelation.BEFORE, 0.7),
            (["decision"], ["execution"], TemporalRelation.BEFORE, 0.9)
        ]
        
        for pattern_i, pattern_j, relation, confidence in common_patterns:
            if type_i in pattern_i and type_j in pattern_j:
                return {
                    "type": relation.value,
                    "from": step_i_id,
                    "to": step_j_id,
                    "confidence": confidence,
                    "basis": "common_pattern",
                    "description": f"基于常见模式: {type_i}通常在{type_j}之前"
                }
        
        # 基于持续时间的推断
        duration_i = steps[i].get("estimated_time", 0)
        duration_j = steps[j].get("estimated_time", 0)
        
        if duration_i > duration_j * 2:
            # 长步骤可能在短步骤之前
            return {
                "type": TemporalRelation.BEFORE.value,
                "from": step_i_id,
                "to": step_j_id,
                "confidence": 0.6,
                "basis": "duration_comparison",
                "description": f"长步骤({duration_i}分钟)可能在短步骤({duration_j}分钟)之前"
            }
        
        return None
    
    def _analyze_overall_temporal_structure(self, plan: Dict[str, Any],
                                           features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """分析整体时间结构"""
        steps = plan.get("steps", [])
        if len(steps) < 2:
            return None
        
        durations = [step.get("estimated_time", 0) for step in steps]
        avg_duration = sum(durations) / len(durations)
        duration_std = np.std(durations) if len(durations) > 1 else 0
        
        # 确定整体结构类型
        if duration_std < avg_duration * 0.3:
            structure_type = "uniform"
            description = "步骤持续时间相对均匀"
        elif all(durations[i] <= durations[i+1] for i in range(len(durations)-1)):
            structure_type = "increasing"
            description = "步骤持续时间递增"
        elif all(durations[i] >= durations[i+1] for i in range(len(durations)-1)):
            structure_type = "decreasing"
            description = "步骤持续时间递减"
        else:
            structure_type = "mixed"
            description = "步骤持续时间混合模式"
        
        return {
            "type": "overall_structure",
            "structure": structure_type,
            "confidence": 0.7,
            "description": description,
            "metrics": {
                "avg_duration": avg_duration,
                "duration_std": duration_std,
                "total_duration": sum(durations)
            }
        }
    
    def _assess_temporal_complexity(self, plan: Dict[str, Any],
                                   features: Dict[str, Any],
                                   constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估时间复杂性"""
        complexity = {
            "scores": {},
            "overall_complexity": 0.0,
            "level": "medium"
        }
        
        # 1. 约束复杂性
        constraint_complexity = self._assess_constraint_complexity(constraints)
        complexity["scores"]["constraint_complexity"] = constraint_complexity
        
        # 2. 依赖复杂性
        dependency_complexity = self._assess_dependency_complexity(plan)
        complexity["scores"]["dependency_complexity"] = dependency_complexity
        
        # 3. 时间范围复杂性
        time_horizon_complexity = self._assess_time_horizon_complexity(features)
        complexity["scores"]["time_horizon_complexity"] = time_horizon_complexity
        
        # 4. 不确定性复杂性
        uncertainty_complexity = self._assess_uncertainty_complexity(features)
        complexity["scores"]["uncertainty_complexity"] = uncertainty_complexity
        
        # 计算总体复杂性
        weights = {
            "constraint_complexity": 0.3,
            "dependency_complexity": 0.25,
            "time_horizon_complexity": 0.25,
            "uncertainty_complexity": 0.2
        }
        
        overall = sum(
            complexity["scores"][key] * weights[key]
            for key in weights.keys()
            if key in complexity["scores"]
        )
        
        complexity["overall_complexity"] = overall
        complexity["level"] = self._determine_complexity_level(overall)
        
        return complexity
    
    def _assess_constraint_complexity(self, constraints: List[Dict[str, Any]]) -> float:
        """评估约束复杂性"""
        if not constraints:
            return 0.1
        
        # 考虑约束数量、类型和严格性
        num_constraints = len(constraints)
        
        # 不同类型约束的复杂性权重
        type_weights = {
            TemporalConstraintType.DEADLINE.value: 0.9,
            TemporalConstraintType.PRECEDENCE.value: 0.7,
            TemporalConstraintType.NON_OVERLAP.value: 0.8,
            TemporalConstraintType.RESOURCE_AVAILABILITY.value: 0.6,
            TemporalConstraintType.DURATION.value: 0.5,
            TemporalConstraintType.START_TIME.value: 0.4,
            TemporalConstraintType.END_TIME.value: 0.4,
            TemporalConstraintType.INTERVAL.value: 0.3,
            TemporalConstraintType.SIMULTANEOUS.value: 0.2
        }
        
        # 计算平均约束复杂性
        total_complexity = 0.0
        for constraint in constraints:
            constraint_type = constraint.get("type", "")
            strictness = constraint.get("strictness", 0.5)
            
            type_weight = type_weights.get(constraint_type, 0.5)
            constraint_complexity = type_weight * (0.5 + strictness * 0.5)
            
            total_complexity += constraint_complexity
        
        avg_constraint_complexity = total_complexity / len(constraints)
        
        # 结合约束数量
        count_factor = min(1.0, num_constraints / 10.0)
        
        return (avg_constraint_complexity * 0.7 + count_factor * 0.3)
    
    def _assess_dependency_complexity(self, plan: Dict[str, Any]) -> float:
        """评估依赖复杂性"""
        steps = plan.get("steps", [])
        if len(steps) <= 1:
            return 0.1
        
        dependencies = plan.get("dependencies", {})
        
        # 计算依赖密度
        total_possible_deps = len(steps) * (len(steps) - 1) / 2
        if total_possible_deps == 0:
            return 0.1
        
        actual_deps = sum(len(deps) for deps in dependencies.values())
        dependency_density = actual_deps / total_possible_deps
        
        # 检查依赖结构
        dependency_structure_score = 0.5
        if dependencies:
            # 检查是否有循环依赖
            try:
                dep_graph = nx.DiGraph()
                for step, deps in dependencies.items():
                    for dep in deps:
                        dep_graph.add_edge(dep, step)
                
                if nx.is_directed_acyclic_graph(dep_graph):
                    dependency_structure_score = 0.3  # DAG较简单
                else:
                    dependency_structure_score = 0.9  # 有循环较复杂
            except (nx.NetworkXError, ValueError, TypeError) as e:
                logger.debug(f"依赖结构分析失败: {e}")
                dependency_structure_score = 0.7
        
        return (dependency_density * 0.6 + dependency_structure_score * 0.4)
    
    def _assess_time_horizon_complexity(self, features: Dict[str, Any]) -> float:
        """评估时间范围复杂性"""
        total_duration = features.get("temporal_dimensions", {}).get("total_duration", 0)
        
        # 基于持续时间评估复杂性
        if total_duration <= 30:
            return 0.2  # 短时间
        elif total_duration <= 120:
            return 0.4  # 中等时间
        elif total_duration <= 480:
            return 0.6  # 长时间
        elif total_duration <= 1440:
            return 0.8  # 超长时间（一天）
        else:
            return 0.9  # 非常长时间
    
    def _assess_uncertainty_complexity(self, features: Dict[str, Any]) -> float:
        """评估不确定性复杂性"""
        time_estimates = features.get("time_estimates", {})
        if not time_estimates:
            return 0.3
        
        # 计算平均不确定性
        uncertainty_values = {
            TemporalUncertainty.EXACT: 0.1,
            TemporalUncertainty.APPROXIMATE: 0.3,
            TemporalUncertainty.ESTIMATED: 0.5,
            TemporalUncertainty.UNCERTAIN: 0.7,
            TemporalUncertainty.VARIABLE: 0.9
        }
        
        total_uncertainty = 0.0
        for estimate in time_estimates.values():
            uncertainty = estimate.get("uncertainty")
            if isinstance(uncertainty, TemporalUncertainty):
                total_uncertainty += uncertainty_values.get(uncertainty, 0.5)
            else:
                total_uncertainty += 0.5
        
        avg_uncertainty = total_uncertainty / len(time_estimates)
        
        return avg_uncertainty
    
    def _determine_complexity_level(self, complexity_score: float) -> str:
        """确定复杂性级别"""
        if complexity_score >= 0.8:
            return "very_high"
        elif complexity_score >= 0.6:
            return "high"
        elif complexity_score >= 0.4:
            return "medium"
        elif complexity_score >= 0.2:
            return "low"
        else:
            return "very_low"
    
    def _identify_temporal_challenges(self, plan: Dict[str, Any],
                                     features: Dict[str, Any],
                                     constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别时间挑战"""
        challenges = []
        
        # 1. 时间约束挑战
        strict_constraints = [c for c in constraints if c.get("strictness", 0) >= 0.8]
        if strict_constraints:
            challenges.append({
                "type": "strict_constraints",
                "description": f"有{len(strict_constraints)}个严格时间约束",
                "severity": min(0.9, len(strict_constraints) * 0.1),
                "impact": "可能限制调度灵活性"
            })
        
        # 2. 依赖挑战
        steps = plan.get("steps", [])
        dependencies = plan.get("dependencies", {})
        
        if dependencies and len(steps) > 1:
            # 检查关键路径长度
            critical_path_length = self._estimate_critical_path_length(steps, dependencies)
            if critical_path_length >= len(steps) * 0.7:
                challenges.append({
                    "type": "long_critical_path",
                    "description": f"关键路径较长，包含约{critical_path_length}个步骤",
                    "severity": 0.7,
                    "impact": "可能延长总体完成时间"
                })
        
        # 3. 资源挑战
        resources = plan.get("resource_requirements", {})
        if resources:
            resource_conflicts = self._identify_resource_conflicts(plan)
            if resource_conflicts:
                challenges.append({
                    "type": "resource_conflicts",
                    "description": f"检测到{len(resource_conflicts)}个资源冲突",
                    "severity": 0.6,
                    "impact": "可能需要复杂的资源调度",
                    "conflicts": resource_conflicts
                })
        
        # 4. 不确定性挑战
        uncertainty_level = self._assess_overall_uncertainty(features)
        if uncertainty_level >= 0.7:
            challenges.append({
                "type": "high_uncertainty",
                "description": "时间估计不确定性较高",
                "severity": uncertainty_level,
                "impact": "计划可能偏离预期时间表"
            })
        
        # 5. 时间范围挑战
        total_duration = features.get("temporal_dimensions", {}).get("total_duration", 0)
        if total_duration >= 240:  # 4小时以上
            challenges.append({
                "type": "extended_time_horizon",
                "description": f"计划总持续时间较长: {total_duration}分钟",
                "severity": min(0.8, total_duration / 600),
                "impact": "可能需要分阶段实施或定期监控"
            })
        
        return challenges
    
    def _estimate_critical_path_length(self, steps: List[Dict[str, Any]],
                                      dependencies: Dict[str, List[str]]) -> int:
        """估计关键路径长度"""
        # 简化实现：计算最长依赖链
        max_length = 0
        
        for step_idx in range(len(steps)):
            current_length = 1
            current_step = f"step_{step_idx}"
            
            # 向后追溯依赖
            while current_step in dependencies:
                # 找到当前步骤的依赖
                current_deps = dependencies.get(current_step, [])
                if not current_deps:
                    break
                
                # 选择第一个依赖（简化）
                current_step = current_deps[0]
                current_length += 1
                
                # 防止无限循环
                if current_length > len(steps):
                    break
            
            max_length = max(max_length, current_length)
        
        return max_length
    
    def _identify_resource_conflicts(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别资源冲突"""
        conflicts = []
        
        steps = plan.get("steps", [])
        resources = plan.get("resource_requirements", {})
        
        if not resources or len(steps) < 2:
            return conflicts
        
        # 创建资源到步骤的映射
        resource_to_steps = defaultdict(list)
        for i, step in enumerate(steps):
            step_resources = step.get("resources", [])
            for resource in step_resources:
                resource_to_steps[resource].append(i)
        
        # 检查每个资源的冲突
        for resource, step_indices in resource_to_steps.items():
            if len(step_indices) >= 2:
                # 检查这些步骤是否有依赖关系
                dependencies = plan.get("dependencies", {})
                
                conflict_found = False
                for i in range(len(step_indices)):
                    for j in range(i + 1, len(step_indices)):
                        step_i = f"step_{step_indices[i]}"
                        step_j = f"step_{step_indices[j]}"
                        
                        # 检查是否有依赖关系允许它们顺序执行
                        i_depends_on_j = step_i in dependencies.get(step_j, [])
                        j_depends_on_i = step_j in dependencies.get(step_i, [])
                        
                        # 如果没有依赖关系，它们可能需要同时执行，造成冲突
                        if not i_depends_on_j and not j_depends_on_i:
                            conflict_found = True
                
                if conflict_found:
                    conflicts.append({
                        "resource": resource,
                        "steps": step_indices,
                        "description": f"资源'{resource}'被多个步骤共享，可能造成调度冲突"
                    })
        
        return conflicts
    
    def _assess_overall_uncertainty(self, features: Dict[str, Any]) -> float:
        """评估总体不确定性"""
        time_estimates = features.get("time_estimates", {})
        if not time_estimates:
            return 0.5
        
        # 计算平均不确定性
        total_uncertainty = 0.0
        count = 0
        
        for estimate in time_estimates.values():
            uncertainty = estimate.get("uncertainty")
            if isinstance(uncertainty, TemporalUncertainty):
                if uncertainty == TemporalUncertainty.EXACT:
                    total_uncertainty += 0.1
                elif uncertainty == TemporalUncertainty.APPROXIMATE:
                    total_uncertainty += 0.3
                elif uncertainty == TemporalUncertainty.ESTIMATED:
                    total_uncertainty += 0.5
                elif uncertainty == TemporalUncertainty.UNCERTAIN:
                    total_uncertainty += 0.7
                else:  # VARIABLE
                    total_uncertainty += 0.9
            else:
                total_uncertainty += 0.5
            
            count += 1
        
        return total_uncertainty / count if count > 0 else 0.5
    
    def _identify_temporal_optimization_opportunities(self, plan: Dict[str, Any],
                                                     features: Dict[str, Any],
                                                     constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别时间优化机会"""
        opportunities = []
        
        # 1. 并行化机会
        parallel_opportunities = self._identify_parallelization_opportunities(plan)
        if parallel_opportunities:
            opportunities.append({
                "type": "parallelization",
                "description": f"识别到{len(parallel_opportunities)}个并行执行机会",
                "potential_benefit": 0.3,  # 估计可以节省30%时间
                "implementation": "重新组织步骤依赖关系以支持并行执行"
            })
        
        # 2. 资源优化机会
        resource_opportunities = self._identify_resource_optimization_opportunities(plan)
        if resource_opportunities:
            opportunities.append({
                "type": "resource_optimization",
                "description": f"识别到{len(resource_opportunities)}个资源优化机会",
                "potential_benefit": 0.2,
                "implementation": "优化资源分配和调度"
            })
        
        # 3. 时间估计优化
        time_estimation_opportunities = self._identify_time_estimation_opportunities(features)
        if time_estimation_opportunities:
            opportunities.append({
                "type": "time_estimation",
                "description": "时间估计可以进一步优化",
                "potential_benefit": 0.15,
                "implementation": "细化时间估计，减少不确定性"
            })
        
        # 4. 约束放松机会
        constraint_relaxation_opportunities = self._identify_constraint_relaxation_opportunities(constraints)
        if constraint_relaxation_opportunities:
            opportunities.append({
                "type": "constraint_relaxation",
                "description": f"可以放松{len(constraint_relaxation_opportunities)}个约束",
                "potential_benefit": 0.25,
                "implementation": "重新评估约束严格性，适当放松以增加灵活性"
            })
        
        # 5. 调度优化机会
        scheduling_opportunities = self._identify_scheduling_optimization_opportunities(plan, features)
        if scheduling_opportunities:
            opportunities.append({
                "type": "scheduling_optimization",
                "description": "存在调度优化空间",
                "potential_benefit": 0.2,
                "implementation": "使用优化算法重新调度步骤"
            })
        
        return opportunities
    
    def _identify_parallelization_opportunities(self, plan: Dict[str, Any]) -> List[List[int]]:
        """识别并行化机会"""
        opportunities = []
        steps = plan.get("steps", [])
        dependencies = plan.get("dependencies", {})
        
        if len(steps) < 2:
            return opportunities
        
        # 寻找没有依赖关系的步骤对
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                step_i_id = f"step_{i}"
                step_j_id = f"step_{j}"
                
                # 检查是否存在依赖关系
                i_depends_on_j = step_i_id in dependencies.get(step_j_id, [])
                j_depends_on_i = step_j_id in dependencies.get(step_i_id, [])
                
                if not i_depends_on_j and not j_depends_on_i:
                    # 检查资源冲突
                    resources_i = steps[i].get("resources", [])
                    resources_j = steps[j].get("resources", [])
                    
                    common_resources = set(resources_i) & set(resources_j)
                    if not common_resources:
                        opportunities.append([i, j])
        
        return opportunities
    
    def _identify_resource_optimization_opportunities(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别资源优化机会"""
        opportunities = []
        resources = plan.get("resource_requirements", {})
        
        if not resources:
            return opportunities
        
        # 检查资源使用模式
        for resource, quantity in resources.items():
            if quantity >= 3:
                opportunities.append({
                    "resource": resource,
                    "current_quantity": quantity,
                    "suggestion": f"考虑减少资源'{resource}'的使用或寻找替代资源",
                    "potential_savings": 0.3
                })
        
        return opportunities
    
    def _identify_time_estimation_opportunities(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别时间估计优化机会"""
        opportunities = []
        time_estimates = features.get("time_estimates", {})
        
        for step_id, estimate in time_estimates.items():
            uncertainty = estimate.get("uncertainty")
            
            if isinstance(uncertainty, TemporalUncertainty):
                if uncertainty in [TemporalUncertainty.UNCERTAIN, TemporalUncertainty.VARIABLE]:
                    opportunities.append({
                        "step": step_id,
                        "current_uncertainty": uncertainty.value,
                        "suggestion": f"为步骤{step_id}提供更精确的时间估计",
                        "potential_improvement": 0.4
                    })
        
        return opportunities
    
    def _identify_constraint_relaxation_opportunities(self, constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别约束放松机会"""
        opportunities = []
        
        for constraint in constraints:
            strictness = constraint.get("strictness", 0.5)
            constraint_type = constraint.get("type", "")
            
            # 高严格性且非关键约束可以放松
            if strictness >= 0.8 and constraint_type not in [
                TemporalConstraintType.DEADLINE.value,
                TemporalConstraintType.PRECEDENCE.value  # 核心依赖
            ]:
                opportunities.append({
                    "constraint": constraint.get("description", "未知约束"),
                    "current_strictness": strictness,
                    "suggestion": f"考虑放松此约束，将严格性从{strictness:.1f}降低到0.6",
                    "potential_benefit": 0.2
                })
        
        return opportunities
    
    def _identify_scheduling_optimization_opportunities(self, plan: Dict[str, Any],
                                                       features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别调度优化机会"""
        opportunities = []
        steps = plan.get("steps", [])
        
        if len(steps) < 3:
            return opportunities
        
        # 检查是否存在明显的调度问题
        durations = [step.get("estimated_time", 0) for step in steps]
        duration_variance = np.var(durations) if len(durations) > 1 else 0
        
        if duration_variance > 100:  # 持续时间差异大
            opportunities.append({
                "issue": "步骤持续时间差异大",
                "variance": duration_variance,
                "suggestion": "平衡步骤持续时间或重新分配任务",
                "potential_benefit": 0.15
            })
        
        # 检查是否有长时间步骤阻塞流程
        long_steps = [i for i, d in enumerate(durations) if d >= 30]
        if len(long_steps) >= 2:
            opportunities.append({
                "issue": "多个长时间步骤",
                "long_step_count": len(long_steps),
                "suggestion": "考虑将长时间步骤分解或并行化",
                "potential_benefit": 0.25
            })
        
        return opportunities
    
    def generate_temporal_plan(self, plan: Dict[str, Any],
                              temporal_analysis: Dict[str, Any],
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        生成时间感知的计划
        
        Args:
            plan: 原始计划
            temporal_analysis: 时间分析结果
            context: 上下文信息
            
        Returns:
            时间优化的计划
        """
        start_time = time.time()
        
        try:
            # 1. 提取调度任务
            scheduling_tasks = self._prepare_scheduling_tasks(plan, temporal_analysis)
            
            # 2. 应用时间约束
            constrained_tasks = self._apply_temporal_constraints(scheduling_tasks, temporal_analysis)
            
            # 3. 生成初始调度
            initial_schedule = self._generate_initial_schedule(constrained_tasks, temporal_analysis)
            
            # 4. 优化调度
            optimized_schedule = self._optimize_schedule(initial_schedule, temporal_analysis, context)
            
            # 5. 处理时间不确定性
            robust_schedule = self._handle_temporal_uncertainty(optimized_schedule, temporal_analysis)
            
            # 6. 生成最终时间计划
            temporal_plan = self._create_temporal_plan(plan, robust_schedule, temporal_analysis)
            
            result = {
                "success": True,
                "original_plan": plan,
                "temporal_analysis": temporal_analysis,
                "temporal_plan": temporal_plan,
                "schedule_details": robust_schedule,
                "optimization_metrics": {
                    "scheduling_time": time.time() - start_time,
                    "schedule_quality": self._assess_schedule_quality(robust_schedule),
                    "improvement_over_naive": self._calculate_schedule_improvement(robust_schedule, initial_schedule),
                    "robustness_score": self._assess_schedule_robustness(robust_schedule, temporal_analysis)
                }
            }
            
            self.temporal_reasoning_state["temporal_constraints_processed"] += len(
                temporal_analysis.get("temporal_constraints", [])
            )
            
            logger.info(f"时间计划生成完成: 计划ID={plan.get('id', 'unknown')}, 质量={result['optimization_metrics']['schedule_quality']:.2f}")
            
            return result
            
        except Exception as e:
            error_handler.handle_error(e, "TemporalReasoningPlanner", "生成时间计划失败")
            return {
                "success": False,
                "error": str(e),
                "fallback_plan": plan  # 返回原始计划作为备用
            }

    def plan_with_temporal_constraints(self, goal: str, temporal_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        时间约束下的规划
        
        Args:
            goal: 规划目标
            temporal_constraints: 时间约束
            
        Returns:
            时间感知的规划结果
        """
        start_time = time.time()
        
        try:
            # 1. 构建基本计划
            base_plan = {
                "goal": goal,
                "temporal_constraints": temporal_constraints,
                "steps": self._extract_steps_from_goal(goal),
                "timestamp": datetime.now().isoformat()
            }
            
            # 2. 分析时间方面
            temporal_analysis = self.analyze_temporal_aspects(base_plan, {"temporal_constraints": temporal_constraints})
            
            if not temporal_analysis.get("success", False):
                # 如果分析失败，返回基本计划
                return {
                    "success": True,
                    "plan": base_plan,
                    "temporal_analysis_available": False,
                    "fallback_reason": "时间分析失败",
                    "planning_time": time.time() - start_time
                }
            
            # 3. 生成时间计划
            temporal_plan_result = self.generate_temporal_plan(base_plan, temporal_analysis)
            
            if not temporal_plan_result.get("success", False):
                # 如果生成失败，返回分析结果和基本计划
                return {
                    "success": True,
                    "plan": base_plan,
                    "temporal_analysis": temporal_analysis,
                    "temporal_plan_available": False,
                    "fallback_reason": "时间计划生成失败",
                    "planning_time": time.time() - start_time
                }
            
            # 4. 返回完整结果
            result = {
                "success": True,
                "goal": goal,
                "temporal_constraints": temporal_constraints,
                "temporal_analysis": temporal_analysis,
                "temporal_plan": temporal_plan_result["temporal_plan"],
                "optimization_metrics": temporal_plan_result["optimization_metrics"],
                "planning_time": time.time() - start_time,
                "planning_complexity": temporal_analysis.get("temporal_complexity", {}).get("overall_complexity", 0.0)
            }
            
            logger.info(f"时间约束规划完成: 目标='{goal}', 约束数={len(temporal_constraints)}, 时间={result['planning_time']:.2f}s")
            
            return result
            
        except Exception as e:
            error_handler.handle_error(e, "TemporalReasoningPlanner", "时间约束规划失败")
            return {
                "success": False,
                "goal": goal,
                "error": str(e),
                "planning_time": time.time() - start_time
            }

    def _extract_steps_from_goal(self, goal: str) -> List[Dict[str, Any]]:
        """
        从目标中提取步骤（简化实现）
        
        Args:
            goal: 规划目标
            
        Returns:
            步骤列表
        """
        # 简化实现：根据目标长度生成基本步骤
        goal_length = len(goal.split())
        steps = []
        
        # 根据目标复杂度生成不同数量的步骤
        if goal_length < 5:
            step_count = 3
        elif goal_length < 10:
            step_count = 5
        else:
            step_count = 7
        
        for i in range(step_count):
            steps.append({
                "id": f"step_{i}",
                "description": f"步骤 {i+1}: 处理目标的一部分",
                "estimated_time": 10 * (i + 1),  # 简单的估计
                "resources": ["cognitive_capacity"],
                "depends_on": [] if i == 0 else [f"step_{i-1}"]
            })
        
        return steps


# 实用函数：创建时间推理规划器实例
def create_temporal_reasoning_planner(config: Optional[Dict[str, Any]] = None) -> TemporalReasoningPlanner:
    """
    创建时间推理规划器实例
    
    Args:
        config: 配置字典
        
    Returns:
        初始化好的时间推理规划器实例
    """
    return TemporalReasoningPlanner(config)


# 示例使用
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建规划器
    planner = create_temporal_reasoning_planner()
    
    # 测试计划
    test_plan = {
        "id": "test_plan_temporal_1",
        "goal": "开发具有时间感知的AGI系统",
        "steps": [
            {
                "type": "analysis",
                "description": "分析时间需求",
                "resources": ["analytical_capacity"],
                "estimated_time": 15,
                "depends_on": []
            },
            {
                "type": "generation",
                "description": "设计时间模型",
                "resources": ["creative_capacity", "technical_knowledge"],
                "estimated_time": 25,
                "depends_on": ["analysis"]
            },
            {
                "type": "evaluation",
                "description": "评估时间模型",
                "resources": ["analytical_capacity", "domain_knowledge"],
                "estimated_time": 12,
                "depends_on": ["generation"]
            },
            {
                "type": "execution",
                "description": "实现时间推理",
                "resources": ["technical_skills", "computational_resources"],
                "estimated_time": 40,
                "depends_on": ["evaluation"]
            }
        ],
        "resource_requirements": {
            "analytical_capacity": 2,
            "creative_capacity": 1,
            "technical_knowledge": 1,
            "domain_knowledge": 1,
            "technical_skills": 2,
            "computational_resources": 3
        },
        "dependencies": {
            "step_1": ["step_0"],
            "step_2": ["step_1"],
            "step_3": ["step_2"]
        }
    }
    
    print("开始测试时间推理规划器...")
    print(f"测试计划: {test_plan['goal']}")
    print()
    
    # 分析时间方面
    analysis_result = planner.analyze_temporal_aspects(test_plan)
    
    print("时间分析结果:")
    print(f"成功: {analysis_result['success']}")
    
    if analysis_result['success']:
        print(f"时间特征数: {len(analysis_result['temporal_features'])}")
        print(f"时间约束数: {len(analysis_result['temporal_constraints'])}")
        print(f"时间关系数: {len(analysis_result['temporal_relations'])}")
        print(f"时间复杂性: {analysis_result['temporal_complexity']['overall_complexity']:.2f}")
        print(f"时间挑战数: {len(analysis_result['temporal_challenges'])}")
        print(f"优化机会数: {len(analysis_result['optimization_opportunities'])}")
        
        print()
        print("时间挑战:")
        for challenge in analysis_result['temporal_challenges']:
            print(f"  - {challenge['description']} (严重性: {challenge['severity']:.2f})")
        
        print()
        print("优化机会:")
        for opportunity in analysis_result['optimization_opportunities']:
            print(f"  - {opportunity['description']} (潜在收益: {opportunity['potential_benefit']:.0%})")
    
    print()
    
    # 生成时间计划
    if analysis_result['success']:
        temporal_plan_result = planner.generate_temporal_plan(test_plan, analysis_result)
        
        print("时间计划生成结果:")
        print(f"成功: {temporal_plan_result['success']}")
        
        if temporal_plan_result['success']:
            metrics = temporal_plan_result['optimization_metrics']
            print(f"调度时间: {metrics['scheduling_time']:.2f}秒")
            print(f"计划质量: {metrics['schedule_quality']:.2f}")
            print(f"改进程度: {metrics['improvement_over_naive']:.2f}")
            print(f"鲁棒性分数: {metrics['robustness_score']:.2f}")