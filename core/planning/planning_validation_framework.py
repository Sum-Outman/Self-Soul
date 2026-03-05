#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zlib
"""
规划验证与评估框架

功能:
1. 规划验证: 检查规划的完整性、一致性和可行性
2. 规划评估: 评估规划的质量指标和性能
3. 规划优化: 提供改进建议和优化方案
4. 规划模拟: 模拟执行以预测结果和风险
5. 验证报告: 生成详细的验证报告和建议

验证维度:
1. 结构性验证: 规划层次结构、依赖关系、数据完整性
2. 可行性验证: 资源约束、时间约束、能力约束
3. 质量评估: 效率、风险、鲁棒性、适应性
4. 认知评估: 目标合理性、任务粒度、行动可执行性

核心组件:
- PlanningValidator: 规划验证器
- PlanningEvaluator: 规划评估器
- PlanningOptimizer: 规划优化器
- PlanningSimulator: 规划模拟器
- ValidationReport: 验证报告生成器

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import copy
from collections import defaultdict

# 导入网络分析库
import networkx as nx

# 导入项目模块
from .hierarchical_planning_system import (
    HierarchicalPlanningSystem,
    Goal, Task, Action, PlanningContext,
    PlanningLevel, GoalStatus, ActionStatus
)

# 导入错误处理
try:
    from core.error_handling import error_handler
except ImportError:
    error_handler = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """验证严重性级别"""
    CRITICAL = "critical"      # 阻止执行的严重问题
    ERROR = "error"           # 需要修复的问题
    WARNING = "warning"       # 需要注意的问题
    INFO = "info"            # 信息性消息
    SUGGESTION = "suggestion"  # 改进建议


class ValidationType(Enum):
    """验证类型"""
    STRUCTURAL = "structural"      # 结构验证
    FEASIBILITY = "feasibility"    # 可行性验证
    QUALITY = "quality"           # 质量评估
    RISK = "risk"                # 风险评估
    OPTIMIZATION = "optimization"  # 优化建议
    SIMULATION = "simulation"     # 模拟分析


@dataclass
class ValidationResult:
    """验证结果数据类"""
    id: str
    type: ValidationType
    severity: ValidationSeverity
    description: str
    component: str  # 验证的组件：goal, task, action, system
    component_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """后初始化验证"""
        if not self.id:
            self.id = f"validation_{int(time.time())}_{(zlib.adler32(str(self.description).encode('utf-8')) & 0xffffffff) % 10000}"


@dataclass
class ValidationReport:
    """验证报告数据类"""
    planning_system_id: str
    validation_time: datetime
    total_validations: int = 0
    critical_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    suggestion_count: int = 0
    validation_results: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: ValidationResult):
        """添加验证结果"""
        self.validation_results.append(result)
        self.total_validations += 1
        
        # 更新统计
        if result.severity == ValidationSeverity.CRITICAL:
            self.critical_count += 1
        elif result.severity == ValidationSeverity.ERROR:
            self.error_count += 1
        elif result.severity == ValidationSeverity.WARNING:
            self.warning_count += 1
        elif result.severity == ValidationSeverity.INFO:
            self.info_count += 1
        elif result.severity == ValidationSeverity.SUGGESTION:
            self.suggestion_count += 1
    
    def get_overall_status(self) -> str:
        """获取整体验证状态"""
        if self.critical_count > 0:
            return "CRITICAL"
        elif self.error_count > 0:
            return "ERROR"
        elif self.warning_count > 0:
            return "WARNING"
        else:
            return "PASSED"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "planning_system_id": self.planning_system_id,
            "validation_time": self.validation_time.isoformat(),
            "total_validations": self.total_validations,
            "critical_count": self.critical_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "suggestion_count": self.suggestion_count,
            "overall_status": self.get_overall_status(),
            "validation_results": [{
                "id": r.id,
                "type": r.type.value,
                "severity": r.severity.value,
                "description": r.description,
                "component": r.component,
                "component_id": r.component_id,
                "details": r.details,
                "recommendation": r.recommendation,
                "timestamp": r.timestamp.isoformat()
            } for r in self.validation_results],
            "summary": self.summary,
            "recommendations": self.recommendations,
            "metadata": self.metadata
        }
    
    def save_to_file(self, file_path: str):
        """保存到文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"验证报告已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存验证报告失败: {e}")
            if error_handler:
                error_handler.handle_error(e, "ValidationReport", "保存验证报告失败")


class PlanningValidator:
    """
    规划验证器 - 检查规划的完整性、一致性和可行性
    
    验证范围:
    1. 目标验证: 目标定义、优先级、时间约束
    2. 任务验证: 任务分解、依赖关系、资源需求
    3. 行动验证: 行动定义、参数、可执行性
    4. 系统验证: 整体规划结构、资源分配、约束满足
    
    验证方法:
    - 静态分析: 检查数据完整性和一致性
    - 约束检查: 验证约束是否满足
    - 依赖分析: 分析任务和行动的依赖关系
    - 资源模拟: 模拟资源使用情况
    """
    
    def __init__(self, planning_system: HierarchicalPlanningSystem, 
                 validation_config: Optional[Dict[str, Any]] = None):
        """
        初始化规划验证器
        
        Args:
            planning_system: 分层规划系统实例
            validation_config: 验证配置（可选）
        """
        self.planning_system = planning_system
        self.config = validation_config or self._get_default_config()
        
        # 验证统计
        self.validation_stats = {
            "validations_performed": 0,
            "validation_time": 0.0,
            "critical_findings": 0,
            "error_findings": 0,
            "warning_findings": 0
        }
        
        # 验证规则
        self.validation_rules = self._initialize_validation_rules()
        
        logger.info(f"规划验证器初始化完成，配置规则: {len(self.validation_rules)} 个")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "enable_structural_validation": True,
            "enable_feasibility_validation": True,
            "enable_quality_validation": True,
            "enable_risk_validation": True,
            "max_validation_time": 30.0,  # 秒
            "enable_dependency_check": True,
            "enable_resource_check": True,
            "enable_temporal_check": True,
            "enable_circular_dependency_check": True,
            "enable_orphaned_component_check": True,
            "validation_report_level": "detailed"  # detailed, summary, minimal
        }
    
    def _initialize_validation_rules(self) -> List[Dict[str, Any]]:
        """初始化验证规则"""
        return [
            {
                "id": "goal_missing_fields",
                "type": ValidationType.STRUCTURAL,
                "severity": ValidationSeverity.CRITICAL,
                "description": "目标缺少必需字段",
                "check_function": self._check_goal_missing_fields
            },
            {
                "id": "task_missing_fields",
                "type": ValidationType.STRUCTURAL,
                "severity": ValidationSeverity.CRITICAL,
                "description": "任务缺少必需字段",
                "check_function": self._check_task_missing_fields
            },
            {
                "id": "action_missing_fields",
                "type": ValidationType.STRUCTURAL,
                "severity": ValidationSeverity.CRITICAL,
                "description": "行动缺少必需字段",
                "check_function": self._check_action_missing_fields
            },
            {
                "id": "circular_dependency",
                "type": ValidationType.STRUCTURAL,
                "severity": ValidationSeverity.ERROR,
                "description": "检测到循环依赖",
                "check_function": self._check_circular_dependencies
            },
            {
                "id": "orphaned_task",
                "type": ValidationType.STRUCTURAL,
                "severity": ValidationSeverity.WARNING,
                "description": "任务没有关联的目标",
                "check_function": self._check_orphaned_tasks
            },
            {
                "id": "orphaned_action",
                "type": ValidationType.STRUCTURAL,
                "severity": ValidationSeverity.WARNING,
                "description": "行动没有关联的任务",
                "check_function": self._check_orphaned_actions
            },
            {
                "id": "resource_constraint_violation",
                "type": ValidationType.FEASIBILITY,
                "severity": ValidationSeverity.ERROR,
                "description": "资源约束违反",
                "check_function": self._check_resource_constraints
            },
            {
                "id": "temporal_constraint_violation",
                "type": ValidationType.FEASIBILITY,
                "severity": ValidationSeverity.ERROR,
                "description": "时间约束违反",
                "check_function": self._check_temporal_constraints
            },
            {
                "id": "deadline_exceeded",
                "type": ValidationType.FEASIBILITY,
                "severity": ValidationSeverity.WARNING,
                "description": "目标截止时间已过或即将过期",
                "check_function": self._check_deadlines
            },
            {
                "id": "goal_hierarchy_depth",
                "type": ValidationType.QUALITY,
                "severity": ValidationSeverity.INFO,
                "description": "目标层次深度分析",
                "check_function": self._analyze_goal_hierarchy_depth
            },
            {
                "id": "task_granularity",
                "type": ValidationType.QUALITY,
                "severity": ValidationSeverity.SUGGESTION,
                "description": "任务粒度分析",
                "check_function": self._analyze_task_granularity
            },
            {
                "id": "action_granularity",
                "type": ValidationType.QUALITY,
                "severity": ValidationSeverity.SUGGESTION,
                "description": "行动粒度分析",
                "check_function": self._analyze_action_granularity
            },
            {
                "id": "single_point_of_failure",
                "type": ValidationType.RISK,
                "severity": ValidationSeverity.WARNING,
                "description": "检测到单点故障（关键依赖）",
                "check_function": self._check_single_points_of_failure
            },
            {
                "id": "parallelization_opportunity",
                "type": ValidationType.OPTIMIZATION,
                "severity": ValidationSeverity.SUGGESTION,
                "description": "并行执行机会识别",
                "check_function": self._identify_parallelization_opportunities
            },
            {
                "id": "resource_optimization",
                "type": ValidationType.OPTIMIZATION,
                "severity": ValidationSeverity.SUGGESTION,
                "description": "资源使用优化建议",
                "check_function": self._suggest_resource_optimizations
            }
        ]
    
    def validate_planning_system(self, report_level: str = "detailed") -> ValidationReport:
        """
        验证整个规划系统
        
        Args:
            report_level: 报告级别（detailed, summary, minimal）
            
        Returns:
            验证报告
        """
        start_time = time.time()
        
        # 创建验证报告
        report = ValidationReport(
            planning_system_id=f"planning_system_{int(time.time())}",
            validation_time=datetime.now(),
            metadata={
                "report_level": report_level,
                "planning_system_type": self.planning_system.__class__.__name__
            }
        )
        
        logger.info(f"开始验证规划系统，启用验证: {self.config}")
        
        try:
            # 按类别执行验证规则
            for rule in self.validation_rules:
                # 检查是否启用该类型验证
                rule_type = rule["type"].value
                config_key = f"enable_{rule_type}_validation"
                
                if config_key not in self.config or self.config[config_key]:
                    try:
                        # 执行验证规则
                        rule_check_func = rule["check_function"]
                        results = rule_check_func()
                        
                        # 处理验证结果
                        if results:
                            if isinstance(results, list):
                                for result_data in results:
                                    if result_data:
                                        self._add_result_to_report(result_data, rule, report)
                            elif results:
                                self._add_result_to_report(results, rule, report)
                        
                        # 更新验证统计
                        self.validation_stats["validations_performed"] += 1
                        
                    except Exception as e:
                        logger.error(f"执行验证规则 {rule['id']} 失败: {e}")
                        # 添加验证失败的错误结果
                        error_result = {
                            "component": "validator",
                            "component_id": rule["id"],
                            "details": {"error": str(e)},
                            "recommendation": "检查验证规则实现"
                        }
                        self._add_result_to_report(error_result, rule, report)
            
            # 计算验证时间
            validation_time = time.time() - start_time
            self.validation_stats["validation_time"] = validation_time
            
            # 生成验证摘要
            report.summary = self._generate_validation_summary(report, validation_time)
            
            # 添加建议
            report.recommendations = self._generate_recommendations(report)
            
            # 更新统计
            report.critical_count = report.critical_count
            report.error_count = report.error_count
            report.warning_count = report.warning_count
            report.info_count = report.info_count
            report.suggestion_count = report.suggestion_count
            
            logger.info(f"规划系统验证完成: {report.total_validations} 个验证, "
                       f"用时: {validation_time:.2f} 秒, 状态: {report.get_overall_status()}")
            
            return report
            
        except Exception as e:
            logger.error(f"规划系统验证失败: {e}")
            if error_handler:
                error_handler.handle_error(e, "PlanningValidator", "规划系统验证失败")
            
            # 返回包含错误的基本报告
            error_report = ValidationReport(
                planning_system_id="error",
                validation_time=datetime.now(),
                metadata={"error": str(e)}
            )
            
            error_report.add_result(ValidationResult(
                id="validation_error",
                type=ValidationType.STRUCTURAL,
                severity=ValidationSeverity.CRITICAL,
                description=f"验证过程失败: {e}",
                component="validator",
                details={"error_traceback": str(e)}
            ))
            
            return error_report
    
    def _add_result_to_report(self, result_data: Dict[str, Any], rule: Dict[str, Any], report: ValidationReport):
        """添加验证结果到报告"""
        try:
            # 构建验证结果
            validation_result = ValidationResult(
                id=f"{rule['id']}_{int(time.time())}_{len(report.validation_results)}",
                type=rule["type"],
                severity=rule["severity"],
                description=rule["description"],
                component=result_data.get("component", "unknown"),
                component_id=result_data.get("component_id"),
                details=result_data.get("details", {}),
                recommendation=result_data.get("recommendation")
            )
            
            # 添加额外信息
            if "additional_info" in result_data:
                validation_result.details["additional_info"] = result_data["additional_info"]
            
            # 添加到报告
            report.add_result(validation_result)
            
            # 更新验证统计
            severity = rule["severity"]
            if severity == ValidationSeverity.CRITICAL:
                self.validation_stats["critical_findings"] += 1
            elif severity == ValidationSeverity.ERROR:
                self.validation_stats["error_findings"] += 1
            elif severity == ValidationSeverity.WARNING:
                self.validation_stats["warning_findings"] += 1
            
        except Exception as e:
            logger.error(f"添加验证结果到报告失败: {e}")
    
    def _check_goal_missing_fields(self) -> List[Dict[str, Any]]:
        """检查目标缺少必需字段"""
        missing_field_goals = []
        
        for goal_id, goal in self.planning_system.goals.items():
            missing_fields = []
            
            # 检查必需字段
            if not goal.description:
                missing_fields.append("description")
            if not goal.level:
                missing_fields.append("level")
            if goal.priority is None:
                missing_fields.append("priority")
            
            if missing_fields:
                missing_field_goals.append({
                    "component": "goal",
                    "component_id": goal_id,
                    "details": {
                        "missing_fields": missing_fields,
                        "goal_description": goal.description[:50] if goal.description else "N/A"
                    },
                    "recommendation": f"为目标 {goal_id} 添加缺失字段: {', '.join(missing_fields)}"
                })
        
        return missing_field_goals
    
    def _check_task_missing_fields(self) -> List[Dict[str, Any]]:
        """检查任务缺少必需字段"""
        missing_field_tasks = []
        
        for task_id, task in self.planning_system.tasks.items():
            missing_fields = []
            
            # 检查必需字段
            if not task.description:
                missing_fields.append("description")
            if not task.level:
                missing_fields.append("level")
            if not task.goal_id:
                missing_fields.append("goal_id")
            if task.estimated_duration is None:
                missing_fields.append("estimated_duration")
            
            if missing_fields:
                missing_field_tasks.append({
                    "component": "task",
                    "component_id": task_id,
                    "details": {
                        "missing_fields": missing_fields,
                        "task_description": task.description[:50] if task.description else "N/A",
                        "goal_id": task.goal_id
                    },
                    "recommendation": f"为任务 {task_id} 添加缺失字段: {', '.join(missing_fields)}"
                })
        
        return missing_field_tasks
    
    def _check_action_missing_fields(self) -> List[Dict[str, Any]]:
        """检查行动缺少必需字段"""
        missing_field_actions = []
        
        for action_id, action in self.planning_system.actions.items():
            missing_fields = []
            
            # 检查必需字段
            if not action.description:
                missing_fields.append("description")
            if not action.task_id:
                missing_fields.append("task_id")
            if action.estimated_duration is None:
                missing_fields.append("estimated_duration")
            
            if missing_fields:
                missing_field_actions.append({
                    "component": "action",
                    "component_id": action_id,
                    "details": {
                        "missing_fields": missing_fields,
                        "action_description": action.description[:50] if action.description else "N/A",
                        "task_id": action.task_id
                    },
                    "recommendation": f"为行动 {action_id} 添加缺失字段: {', '.join(missing_fields)}"
                })
        
        return missing_field_actions
    
    def _check_circular_dependencies(self) -> List[Dict[str, Any]]:
        """检查循环依赖"""
        circular_deps = []
        
        # 检查任务依赖图中的循环
        task_graph = self.planning_system.task_dependency_graph
        
        try:
            # 查找循环依赖
            cycles = list(nx.simple_cycles(task_graph))
            
            if cycles:
                for i, cycle in enumerate(cycles[:5]):  # 限制前5个循环
                    if len(cycle) > 1:  # 忽略自循环
                        cycle_details = []
                        for task_id in cycle:
                            task = self.planning_system.tasks.get(task_id)
                            if task:
                                cycle_details.append({
                                    "task_id": task_id,
                                    "description": task.description[:30] if task.description else "N/A"
                                })
                        
                        circular_deps.append({
                            "component": "task_dependencies",
                            "component_id": f"cycle_{i}",
                            "details": {
                                "cycle": cycle,
                                "cycle_details": cycle_details,
                                "cycle_length": len(cycle)
                            },
                            "recommendation": f"检测到循环依赖，涉及任务: {', '.join(cycle)}。需要修改依赖关系打破循环。"
                        })
        except Exception as e:
            logger.warning(f"检查循环依赖时出错: {e}")
        
        # 检查行动依赖图中的循环
        action_graph = self.planning_system.action_dependency_graph
        
        try:
            cycles = list(nx.simple_cycles(action_graph))
            
            if cycles:
                for i, cycle in enumerate(cycles[:5]):  # 限制前5个循环
                    if len(cycle) > 1:  # 忽略自循环
                        cycle_details = []
                        for action_id in cycle:
                            action = self.planning_system.actions.get(action_id)
                            if action:
                                cycle_details.append({
                                    "action_id": action_id,
                                    "description": action.description[:30] if action.description else "N/A"
                                })
                        
                        circular_deps.append({
                            "component": "action_dependencies",
                            "component_id": f"action_cycle_{i}",
                            "details": {
                                "cycle": cycle,
                                "cycle_details": cycle_details,
                                "cycle_length": len(cycle)
                            },
                            "recommendation": f"检测到行动循环依赖，涉及行动: {', '.join(cycle)}。需要修改依赖关系打破循环。"
                        })
        except Exception as e:
            logger.warning(f"检查行动循环依赖时出错: {e}")
        
        return circular_deps
    
    def _check_orphaned_tasks(self) -> List[Dict[str, Any]]:
        """检查孤立任务（没有关联的目标）"""
        orphaned_tasks = []
        
        for task_id, task in self.planning_system.tasks.items():
            if task.goal_id not in self.planning_system.goals:
                orphaned_tasks.append({
                    "component": "task",
                    "component_id": task_id,
                    "details": {
                        "task_description": task.description[:50] if task.description else "N/A",
                        "missing_goal_id": task.goal_id
                    },
                    "recommendation": f"任务 {task_id} 关联的目标 {task.goal_id} 不存在。需要修复目标关联或删除此任务。"
                })
        
        return orphaned_tasks
    
    def _check_orphaned_actions(self) -> List[Dict[str, Any]]:
        """检查孤立行动（没有关联的任务）"""
        orphaned_actions = []
        
        for action_id, action in self.planning_system.actions.items():
            if action.task_id not in self.planning_system.tasks:
                orphaned_actions.append({
                    "component": "action",
                    "component_id": action_id,
                    "details": {
                        "action_description": action.description[:50] if action.description else "N/A",
                        "missing_task_id": action.task_id
                    },
                    "recommendation": f"行动 {action_id} 关联的任务 {action.task_id} 不存在。需要修复任务关联或删除此行动。"
                })
        
        return orphaned_actions
    
    def _check_resource_constraints(self) -> List[Dict[str, Any]]:
        """检查资源约束违反"""
        resource_violations = []
        
        # 检查是否有规划上下文
        if not self.planning_system.current_context:
            return [{
                "component": "system",
                "component_id": "planning_context",
                "details": {"missing_context": True},
                "recommendation": "规划系统缺少上下文信息，无法进行资源约束检查。请设置规划上下文。"
            }]
        
        available_resources = self.planning_system.current_context.available_resources
        
        # 计算所有行动的总资源需求
        total_resource_requirements = defaultdict(float)
        
        for action_id, action in self.planning_system.actions.items():
            for resource, amount in action.resources.items():
                total_resource_requirements[resource] += amount
        
        # 检查资源是否足够
        for resource, required in total_resource_requirements.items():
            available = available_resources.get(resource, 0.0)
            if required > available:
                resource_violations.append({
                    "component": "resource_constraints",
                    "component_id": resource,
                    "details": {
                        "resource": resource,
                        "required": required,
                        "available": available,
                        "deficit": required - available
                    },
                    "recommendation": f"资源 {resource} 不足: 需要 {required:.2f}, 可用 {available:.2f}, 缺口 {required-available:.2f}。"
                })
        
        return resource_violations
    
    def _check_temporal_constraints(self) -> List[Dict[str, Any]]:
        """检查时间约束违反"""
        temporal_violations = []
        
        # 检查任务持续时间是否合理
        for task_id, task in self.planning_system.tasks.items():
            if task.estimated_duration <= 0:
                temporal_violations.append({
                    "component": "task",
                    "component_id": task_id,
                    "details": {
                        "task_description": task.description[:50] if task.description else "N/A",
                        "estimated_duration": task.estimated_duration
                    },
                    "recommendation": f"任务 {task_id} 的预计持续时间 ({task.estimated_duration}) 无效。持续时间应为正数。"
                })
        
        # 检查行动持续时间是否合理
        for action_id, action in self.planning_system.actions.items():
            if action.estimated_duration <= 0:
                temporal_violations.append({
                    "component": "action",
                    "component_id": action_id,
                    "details": {
                        "action_description": action.description[:50] if action.description else "N/A",
                        "estimated_duration": action.estimated_duration
                    },
                    "recommendation": f"行动 {action_id} 的预计持续时间 ({action.estimated_duration}) 无效。持续时间应为正数。"
                })
        
        return temporal_violations
    
    def _check_deadlines(self) -> List[Dict[str, Any]]:
        """检查截止时间"""
        deadline_issues = []
        current_time = datetime.now()
        
        for goal_id, goal in self.planning_system.goals.items():
            if goal.deadline:
                time_remaining = goal.deadline - current_time
                
                # 检查截止时间是否已过
                if time_remaining.total_seconds() < 0:
                    deadline_issues.append({
                        "component": "goal",
                        "component_id": goal_id,
                        "details": {
                            "goal_description": goal.description[:50] if goal.description else "N/A",
                            "deadline": goal.deadline.isoformat(),
                            "overdue_by": abs(time_remaining.total_seconds() / 3600)  # 小时
                        },
                        "recommendation": f"目标 {goal_id} 的截止时间已过 {abs(time_remaining.total_seconds()/3600):.1f} 小时。需要重新评估或调整截止时间。"
                    })
                # 检查截止时间是否即将到来（7天内）
                elif time_remaining.total_seconds() < 7 * 24 * 3600:  # 7天
                    deadline_issues.append({
                        "component": "goal",
                        "component_id": goal_id,
                        "details": {
                            "goal_description": goal.description[:50] if goal.description else "N/A",
                            "deadline": goal.deadline.isoformat(),
                            "days_remaining": time_remaining.total_seconds() / (24 * 3600)
                        },
                        "recommendation": f"目标 {goal_id} 的截止时间即将到来 (剩余 {time_remaining.total_seconds()/(24*3600):.1f} 天)。需要加速执行或重新评估可行性。"
                    })
        
        return deadline_issues
    
    def _analyze_goal_hierarchy_depth(self) -> List[Dict[str, Any]]:
        """分析目标层次深度"""
        analysis_results = []
        
        # 获取目标层次深度
        max_depth = self.planning_system._calculate_max_hierarchy_depth()
        
        analysis_results.append({
            "component": "goal_hierarchy",
            "component_id": "hierarchy_depth",
            "details": {
                "max_depth": max_depth,
                "depth_interpretation": self._interpret_hierarchy_depth(max_depth)
            },
            "recommendation": self._get_hierarchy_depth_recommendation(max_depth)
        })
        
        return analysis_results
    
    def _interpret_hierarchy_depth(self, depth: int) -> str:
        """解释层次深度"""
        if depth == 0:
            return "无层次结构（单层目标）"
        elif depth <= 2:
            return "浅层次结构（适合简单规划）"
        elif depth <= 4:
            return "中等层次结构（适合中等复杂度规划）"
        elif depth <= 6:
            return "较深层次结构（适合复杂规划）"
        else:
            return "过深层次结构（可能过度分解，需要简化）"
    
    def _get_hierarchy_depth_recommendation(self, depth: int) -> str:
        """获取层次深度建议"""
        if depth == 0:
            return "考虑为复杂目标添加子目标层次"
        elif depth <= 2:
            return "层次结构合理，适合大多数规划场景"
        elif depth <= 4:
            return "层次结构良好，平衡了复杂性和可管理性"
        elif depth <= 6:
            return "层次结构较深，考虑简化或合并某些子目标"
        else:
            return "层次结构过深，可能导致管理困难。建议简化分解或合并相关子目标"
    
    def _analyze_task_granularity(self) -> List[Dict[str, Any]]:
        """分析任务粒度"""
        analysis_results = []
        
        # 分析任务持续时间分布
        task_durations = []
        for task in self.planning_system.tasks.values():
            if task.estimated_duration > 0:
                task_durations.append(task.estimated_duration)
        
        if task_durations:
            avg_duration = sum(task_durations) / len(task_durations)
            min_duration = min(task_durations)
            max_duration = max(task_durations)
            
            # 分类任务粒度
            fine_grain = sum(1 for d in task_durations if d < 4)  # 小于4小时
            medium_grain = sum(1 for d in task_durations if 4 <= d <= 24)  # 4-24小时
            coarse_grain = sum(1 for d in task_durations if d > 24)  # 大于24小时
            
            analysis_results.append({
                "component": "task_granularity",
                "component_id": "granularity_analysis",
                "details": {
                    "total_tasks": len(task_durations),
                    "average_duration": avg_duration,
                    "min_duration": min_duration,
                    "max_duration": max_duration,
                    "fine_grain_tasks": fine_grain,
                    "medium_grain_tasks": medium_grain,
                    "coarse_grain_tasks": coarse_grain,
                    "granularity_balance": self._assess_granularity_balance(fine_grain, medium_grain, coarse_grain)
                },
                "recommendation": self._get_task_granularity_recommendation(fine_grain, medium_grain, coarse_grain, avg_duration)
            })
        
        return analysis_results
    
    def _assess_granularity_balance(self, fine: int, medium: int, coarse: int) -> str:
        """评估粒度平衡"""
        total = fine + medium + coarse
        if total == 0:
            return "无任务数据"
        
        fine_pct = fine / total * 100
        coarse_pct = coarse / total * 100
        
        if fine_pct > 70:
            return "粒度过细（可能导致过度管理和碎片化）"
        elif coarse_pct > 50:
            return "粒度过粗（可能导致任务过大，难以管理）"
        elif medium >= fine and medium >= coarse:
            return "粒度平衡良好（适中的任务大小）"
        else:
            return "粒度分布不均（建议调整任务大小）"
    
    def _get_task_granularity_recommendation(self, fine: int, medium: int, coarse: int, avg_duration: float) -> str:
        """获取任务粒度建议"""
        total = fine + medium + coarse
        if total == 0:
            return "无任务数据可供分析"
        
        fine_pct = fine / total * 100
        coarse_pct = coarse / total * 100
        
        if fine_pct > 70:
            return f"任务粒度过细 ({fine_pct:.1f}% 的任务小于4小时)。考虑合并相关的小任务以提高效率。"
        elif coarse_pct > 50:
            return f"任务粒度过粗 ({coarse_pct:.1f}% 的任务大于24小时)。考虑将大任务分解为更小的可管理单元。"
        elif avg_duration > 48:
            return f"平均任务持续时间过长 ({avg_duration:.1f} 小时)。建议将任务分解为更小的单元以改善进度跟踪。"
        elif avg_duration < 2:
            return f"平均任务持续时间过短 ({avg_duration:.1f} 小时)。考虑合并小任务以减少管理开销。"
        else:
            return f"任务粒度适中 (平均 {avg_duration:.1f} 小时)。任务大小适合有效的规划和管理。"
    
    def _analyze_action_granularity(self) -> List[Dict[str, Any]]:
        """分析行动粒度"""
        analysis_results = []
        
        # 分析行动持续时间分布
        action_durations = []
        for action in self.planning_system.actions.values():
            if action.estimated_duration > 0:
                action_durations.append(action.estimated_duration)
        
        if action_durations:
            avg_duration = sum(action_durations) / len(action_durations)
            min_duration = min(action_durations)
            max_duration = max(action_durations)
            
            # 分类行动粒度
            micro_actions = sum(1 for d in action_durations if d < 0.5)  # 小于0.5小时
            short_actions = sum(1 for d in action_durations if 0.5 <= d <= 2)  # 0.5-2小时
            medium_actions = sum(1 for d in action_durations if 2 < d <= 8)  # 2-8小时
            long_actions = sum(1 for d in action_durations if d > 8)  # 大于8小时
            
            analysis_results.append({
                "component": "action_granularity",
                "component_id": "action_granularity_analysis",
                "details": {
                    "total_actions": len(action_durations),
                    "average_duration": avg_duration,
                    "min_duration": min_duration,
                    "max_duration": max_duration,
                    "micro_actions": micro_actions,
                    "short_actions": short_actions,
                    "medium_actions": medium_actions,
                    "long_actions": long_actions,
                    "action_granularity_balance": self._assess_action_granularity_balance(micro_actions, short_actions, medium_actions, long_actions)
                },
                "recommendation": self._get_action_granularity_recommendation(micro_actions, short_actions, medium_actions, long_actions, avg_duration)
            })
        
        return analysis_results
    
    def _assess_action_granularity_balance(self, micro: int, short: int, medium: int, long: int) -> str:
        """评估行动粒度平衡"""
        total = micro + short + medium + long
        if total == 0:
            return "无行动数据"
        
        micro_pct = micro / total * 100
        long_pct = long / total * 100
        
        if micro_pct > 60:
            return "行动粒度过细（可能导致执行碎片化和上下文切换开销）"
        elif long_pct > 40:
            return "行动粒度过粗（可能导致难以监控和调整）"
        elif short + medium >= total * 0.6:
            return "行动粒度适中（适合有效执行）"
        else:
            return "行动粒度分布不均"
    
    def _get_action_granularity_recommendation(self, micro: int, short: int, medium: int, long: int, avg_duration: float) -> str:
        """获取行动粒度建议"""
        total = micro + short + medium + long
        if total == 0:
            return "无行动数据可供分析"
        
        micro_pct = micro / total * 100
        long_pct = long / total * 100
        
        if micro_pct > 60:
            return f"行动粒度过细 ({micro_pct:.1f}% 的行动小于0.5小时)。考虑合并微小行动以减少上下文切换。"
        elif long_pct > 40:
            return f"行动粒度过粗 ({long_pct:.1f}% 的行动大于8小时)。考虑将长行动分解为更小的可监控单元。"
        elif avg_duration > 6:
            return f"平均行动持续时间较长 ({avg_duration:.1f} 小时)。考虑分解长行动以改善执行监控。"
        elif avg_duration < 0.5:
            return f"平均行动持续时间过短 ({avg_duration:.1f} 小时)。考虑合并微小行动以提高执行效率。"
        else:
            return f"行动粒度适中 (平均 {avg_duration:.1f} 小时)。行动大小适合有效的执行和监控。"
    
    def _check_single_points_of_failure(self) -> List[Dict[str, Any]]:
        """检查单点故障（关键依赖）"""
        single_points = []
        
        # 分析任务依赖图中的关键节点
        task_graph = self.planning_system.task_dependency_graph
        
        try:
            # 计算节点的度中心性
            if task_graph.number_of_nodes() > 0:
                degree_centrality = nx.degree_centrality(task_graph)
                
                # 查找高度中心性的节点
                for node, centrality in degree_centrality.items():
                    if centrality > 0.5:  # 高度中心性阈值
                        task = self.planning_system.tasks.get(node)
                        if task:
                            single_points.append({
                                "component": "task_dependencies",
                                "component_id": node,
                                "details": {
                                    "task_description": task.description[:50] if task.description else "N/A",
                                    "degree_centrality": centrality,
                                    "in_degree": task_graph.in_degree(node),
                                    "out_degree": task_graph.out_degree(node)
                                },
                                "recommendation": f"任务 {node} 是依赖图的关键节点（中心性 {centrality:.3f}）。考虑添加并行路径或减少依赖以减少单点故障风险。"
                            })
        except Exception as e:
            logger.warning(f"检查单点故障时出错: {e}")
        
        return single_points
    
    def _identify_parallelization_opportunities(self) -> List[Dict[str, Any]]:
        """识别并行执行机会"""
        parallelization_opportunities = []
        
        # 分析任务依赖图
        task_graph = self.planning_system.task_dependency_graph
        
        try:
            # 查找没有依赖关系的任务组（可以并行执行）
            independent_tasks = []
            
            for task_id in task_graph.nodes():
                # 检查是否有前置任务
                predecessors = list(task_graph.predecessors(task_id))
                
                if not predecessors:
                    # 检查是否有后置任务（说明可能有并行机会）
                    successors = list(task_graph.successors(task_id))
                    
                    if successors:
                        independent_tasks.append(task_id)
            
            if independent_tasks and len(independent_tasks) > 1:
                parallelization_opportunities.append({
                    "component": "task_parallelization",
                    "component_id": "parallel_group",
                    "details": {
                        "independent_tasks": independent_tasks,
                        "task_count": len(independent_tasks),
                        "task_descriptions": [
                            self.planning_system.tasks.get(tid, {}).get('description', 'N/A')[:30]
                            for tid in independent_tasks
                        ]
                    },
                    "recommendation": f"发现 {len(independent_tasks)} 个可以并行执行的任务: {', '.join(independent_tasks)}。考虑并行执行以提高效率。"
                })
        
        except Exception as e:
            logger.warning(f"识别并行执行机会时出错: {e}")
        
        return parallelization_opportunities
    
    def _suggest_resource_optimizations(self) -> List[Dict[str, Any]]:
        """建议资源使用优化"""
        optimization_suggestions = []
        
        # 检查是否有规划上下文
        if not self.planning_system.current_context:
            return optimization_suggestions
        
        available_resources = self.planning_system.current_context.available_resources
        
        # 分析资源使用情况
        total_resource_requirements = defaultdict(float)
        resource_by_action = defaultdict(list)
        
        for action_id, action in self.planning_system.actions.items():
            for resource, amount in action.resources.items():
                total_resource_requirements[resource] += amount
                resource_by_action[resource].append({
                    "action_id": action_id,
                    "amount": amount,
                    "description": action.description[:30] if action.description else "N/A"
                })
        
        # 分析每个资源的利用情况
        for resource, required in total_resource_requirements.items():
            available = available_resources.get(resource, 0.0)
            
            if available > 0:
                utilization_rate = required / available * 100
                
                if utilization_rate < 30:
                    optimization_suggestions.append({
                        "component": "resource_optimization",
                        "component_id": resource,
                        "details": {
                            "resource": resource,
                            "required": required,
                            "available": available,
                            "utilization_rate": utilization_rate
                        },
                        "recommendation": f"资源 {resource} 利用率低 ({utilization_rate:.1f}%)。考虑减少此资源分配或增加使用以提高效率。"
                    })
                elif utilization_rate > 90:
                    optimization_suggestions.append({
                        "component": "resource_optimization",
                        "component_id": resource,
                        "details": {
                            "resource": resource,
                            "required": required,
                            "available": available,
                            "utilization_rate": utilization_rate
                        },
                        "recommendation": f"资源 {resource} 利用率过高 ({utilization_rate:.1f}%)。考虑增加此资源分配或优化使用以避免瓶颈。"
                    })
        
        return optimization_suggestions
    
    def _generate_validation_summary(self, report: ValidationReport, validation_time: float) -> Dict[str, Any]:
        """生成验证摘要"""
        total_components = {
            "goals": len(self.planning_system.goals),
            "tasks": len(self.planning_system.tasks),
            "actions": len(self.planning_system.actions)
        }
        
        # 计算验证覆盖率
        validation_coverage = {}
        for component_type, count in total_components.items():
            component_validations = sum(1 for r in report.validation_results if r.component == component_type)
            validation_coverage[component_type] = {
                "total": count,
                "validated": component_validations,
                "coverage_percentage": (component_validations / max(count, 1)) * 100
            }
        
        # 系统状态评估
        system_health_score = self._calculate_system_health_score(report)
        
        summary = {
            "validation_time_seconds": validation_time,
            "total_components": total_components,
            "validation_coverage": validation_coverage,
            "system_health_score": system_health_score,
            "health_assessment": self._assess_system_health(system_health_score),
            "critical_issues_present": report.critical_count > 0,
            "error_issues_present": report.error_count > 0,
            "warning_issues_present": report.warning_count > 0,
            "overall_validation_status": report.get_overall_status(),
            "validator_performance": self.validation_stats.copy()
        }
        
        return summary
    
    def _calculate_system_health_score(self, report: ValidationReport) -> float:
        """计算系统健康分数（0-100）"""
        if report.total_validations == 0:
            return 100.0  # 无验证，假设健康
        
        # 计算扣分
        critical_penalty = report.critical_count * 30
        error_penalty = report.error_count * 15
        warning_penalty = report.warning_count * 5
        
        total_penalty = critical_penalty + error_penalty + warning_penalty
        
        # 计算基础分数
        base_score = 100.0
        
        # 应用惩罚
        health_score = max(0.0, base_score - total_penalty)
        
        # 考虑验证覆盖率
        total_components = len(self.planning_system.goals) + len(self.planning_system.tasks) + len(self.planning_system.actions)
        if total_components > 0:
            coverage_factor = min(1.0, report.total_validations / (total_components * 0.5))
            health_score *= coverage_factor
        
        return min(100.0, health_score)
    
    def _assess_system_health(self, health_score: float) -> str:
        """评估系统健康状况"""
        if health_score >= 90:
            return "EXCELLENT - 系统状态优秀，规划质量高"
        elif health_score >= 75:
            return "GOOD - 系统状态良好，少量问题需要关注"
        elif health_score >= 60:
            return "FAIR - 系统状态一般，需要修复一些重要问题"
        elif health_score >= 40:
            return "POOR - 系统状态较差，需要立即修复关键问题"
        else:
            return "CRITICAL - 系统状态危急，需要紧急修复"
    
    def _generate_recommendations(self, report: ValidationReport) -> List[str]:
        """生成建议列表"""
        recommendations = []
        
        # 基于验证结果生成建议
        if report.critical_count > 0:
            recommendations.append(f"发现 {report.critical_count} 个严重问题。建议立即修复这些问题，否则规划可能无法执行。")
        
        if report.error_count > 0:
            recommendations.append(f"发现 {report.error_count} 个错误问题。建议在继续规划前修复这些问题。")
        
        if report.warning_count > 0:
            recommendations.append(f"发现 {report.warning_count} 个警告问题。建议审查这些问题以改进规划质量。")
        
        # 基于系统状态的建议
        total_components = len(self.planning_system.goals) + len(self.planning_system.tasks) + len(self.planning_system.actions)
        
        if total_components == 0:
            recommendations.append("规划系统为空。建议添加目标和任务以开始规划。")
        elif len(self.planning_system.goals) == 0:
            recommendations.append("规划系统没有目标。建议添加战略、战术或操作目标。")
        elif len(self.planning_system.tasks) == 0:
            recommendations.append("规划系统没有任务。建议为目标生成任务分解。")
        elif len(self.planning_system.actions) == 0:
            recommendations.append("规划系统没有行动。建议为任务生成具体行动。")
        
        # 检查是否有规划上下文
        if not self.planning_system.current_context:
            recommendations.append("规划系统缺少上下文信息。建议设置规划上下文以支持资源约束检查和时间安排。")
        
        return recommendations
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """获取验证统计信息"""
        return self.validation_stats.copy()


class PlanningEvaluator:
    """
    规划评估器 - 评估规划的质量指标和性能
    
    评估维度:
    1. 效率评估: 时间效率、资源效率、执行效率
    2. 质量评估: 规划完整性、一致性、可执行性
    3. 风险评估: 执行风险、依赖风险、资源风险
    4. 适应性评估: 变更适应性、不确定性处理能力
    5. 成本效益评估: 投入产出比、价值实现度
    
    评估方法:
    - 指标计算: 计算各种评估指标
    - 基准比较: 与最佳实践基准比较
    - 模拟评估: 通过模拟执行评估
    - 专家评估: 应用专家规则评估
    """
    
    def __init__(self, planning_system: HierarchicalPlanningSystem,
                 evaluation_config: Optional[Dict[str, Any]] = None):
        """
        初始化规划评估器
        
        Args:
            planning_system: 分层规划系统实例
            evaluation_config: 评估配置（可选）
        """
        self.planning_system = planning_system
        self.config = evaluation_config or self._get_default_config()
        
        # 评估指标定义
        self.metrics_definitions = self._initialize_metrics_definitions()
        
        # 评估基准（最佳实践值）
        self.benchmarks = self._initialize_benchmarks()
        
        # 评估历史
        self.evaluation_history: List[Dict[str, Any]] = []
        self.max_history_size = self.config.get('max_history_size', 100)
        
        # 评估统计
        self.evaluation_stats = {
            "evaluations_performed": 0,
            "total_evaluation_time": 0.0,
            "average_evaluation_time": 0.0,
            "last_evaluation_time": None
        }
        
        logger.info(f"规划评估器初始化完成，配置指标: {len(self.metrics_definitions)} 个")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "enable_efficiency_evaluation": True,
            "enable_quality_evaluation": True,
            "enable_risk_evaluation": True,
            "enable_adaptability_evaluation": True,
            "enable_cost_benefit_evaluation": True,
            "evaluation_time_limit": 60.0,  # 秒
            "use_simulation_for_evaluation": True,
            "simulation_iterations": 100,
            "benchmark_comparison_enabled": True,
            "max_history_size": 100
        }
    
    def _initialize_metrics_definitions(self) -> Dict[str, Dict[str, Any]]:
        """初始化评估指标定义"""
        return {
            # 效率指标
            "time_efficiency": {
                "name": "时间效率",
                "description": "规划的时间利用效率",
                "category": "efficiency",
                "calculation": self._calculate_time_efficiency,
                "unit": "百分比",
                "optimal_range": (70, 95),
                "weight": 0.25
            },
            "resource_efficiency": {
                "name": "资源效率",
                "description": "规划的资源利用效率",
                "category": "efficiency",
                "calculation": self._calculate_resource_efficiency,
                "unit": "百分比",
                "optimal_range": (65, 90),
                "weight": 0.20
            },
            "planning_completeness": {
                "name": "规划完整性",
                "description": "规划的完整性和覆盖度",
                "category": "quality",
                "calculation": self._calculate_planning_completeness,
                "unit": "百分比",
                "optimal_range": (80, 100),
                "weight": 0.15
            },
            "dependency_coherence": {
                "name": "依赖一致性",
                "description": "任务和行动依赖关系的一致性",
                "category": "quality",
                "calculation": self._calculate_dependency_coherence,
                "unit": "分数",
                "optimal_range": (0.7, 1.0),
                "weight": 0.10
            },
            "risk_level": {
                "name": "风险水平",
                "description": "规划的整体风险水平",
                "category": "risk",
                "calculation": self._calculate_risk_level,
                "unit": "分数",
                "optimal_range": (0.0, 0.3),
                "weight": 0.15,
                "inverse": True  # 越低越好
            },
            "adaptability_score": {
                "name": "适应性分数",
                "description": "规划应对变化和不确定性的能力",
                "category": "adaptability",
                "calculation": self._calculate_adaptability_score,
                "unit": "分数",
                "optimal_range": (0.6, 1.0),
                "weight": 0.10
            },
            "cost_benefit_ratio": {
                "name": "成本效益比",
                "description": "规划的成本与预期效益比率",
                "category": "cost_benefit",
                "calculation": self._calculate_cost_benefit_ratio,
                "unit": "比率",
                "optimal_range": (1.5, 5.0),
                "weight": 0.05
            }
        }
    
    def _initialize_benchmarks(self) -> Dict[str, Any]:
        """初始化评估基准"""
        return {
            "time_efficiency": {
                "excellent": 90.0,
                "good": 75.0,
                "fair": 60.0,
                "poor": 40.0
            },
            "resource_efficiency": {
                "excellent": 85.0,
                "good": 70.0,
                "fair": 55.0,
                "poor": 35.0
            },
            "planning_completeness": {
                "excellent": 95.0,
                "good": 85.0,
                "fair": 70.0,
                "poor": 50.0
            },
            "dependency_coherence": {
                "excellent": 0.95,
                "good": 0.85,
                "fair": 0.70,
                "poor": 0.50
            },
            "risk_level": {
                "excellent": 0.10,
                "good": 0.25,
                "fair": 0.40,
                "poor": 0.60
            },
            "adaptability_score": {
                "excellent": 0.90,
                "good": 0.75,
                "fair": 0.60,
                "poor": 0.40
            },
            "cost_benefit_ratio": {
                "excellent": 4.0,
                "good": 2.5,
                "fair": 1.5,
                "poor": 1.0
            }
        }
    
    def evaluate_planning_system(self, 
                                detailed_report: bool = True,
                                include_simulation: bool = True) -> Dict[str, Any]:
        """
        评估规划系统
        
        Args:
            detailed_report: 是否生成详细报告
            include_simulation: 是否包含模拟评估
            
        Returns:
            评估报告
        """
        start_time = time.time()
        
        logger.info("开始评估规划系统...")
        
        try:
            # 初始化评估结果
            evaluation_result = {
                "timestamp": datetime.now().isoformat(),
                "planning_system_id": f"evaluation_{int(time.time())}",
                "metrics": {},
                "category_scores": {},
                "overall_score": 0.0,
                "benchmark_comparison": {},
                "recommendations": [],
                "simulation_results": None,
                "metadata": {
                    "evaluation_config": self.config,
                    "planning_system_summary": self.planning_system.get_planning_summary()
                }
            }
            
            # 计算所有指标
            calculated_metrics = {}
            for metric_id, metric_def in self.metrics_definitions.items():
                try:
                    # 检查是否启用该类别评估
                    category = metric_def["category"]
                    config_key = f"enable_{category}_evaluation"
                    
                    if config_key in self.config and not self.config[config_key]:
                        logger.debug(f"跳过未启用的指标类别: {category}")
                        continue
                    
                    # 计算指标
                    calculation_func = metric_def["calculation"]
                    metric_value = calculation_func()
                    
                    calculated_metrics[metric_id] = {
                        "value": metric_value,
                        "name": metric_def["name"],
                        "description": metric_def["description"],
                        "category": category,
                        "unit": metric_def["unit"],
                        "optimal_range": metric_def["optimal_range"],
                        "weight": metric_def.get("weight", 0.1),
                        "inverse": metric_def.get("inverse", False)
                    }
                    
                    logger.debug(f"计算指标 {metric_id}: {metric_value}")
                    
                except Exception as e:
                    logger.error(f"计算指标 {metric_id} 失败: {e}")
                    calculated_metrics[metric_id] = {
                        "value": None,
                        "error": str(e),
                        "name": metric_def["name"],
                        "category": metric_def["category"]
                    }
            
            # 计算类别分数
            category_scores = self._calculate_category_scores(calculated_metrics)
            
            # 计算整体分数
            overall_score = self._calculate_overall_score(calculated_metrics, category_scores)
            
            # 基准比较
            benchmark_comparison = self._compare_with_benchmarks(calculated_metrics)
            
            # 生成建议
            recommendations = self._generate_evaluation_recommendations(calculated_metrics, benchmark_comparison)
            
            # 模拟评估（如果启用）
            simulation_results = None
            if include_simulation and self.config.get("use_simulation_for_evaluation", True):
                try:
                    simulation_results = self._perform_simulation_evaluation()
                except Exception as e:
                    logger.error(f"模拟评估失败: {e}")
                    simulation_results = {"error": str(e)}
            
            # 构建最终结果
            evaluation_result["metrics"] = calculated_metrics
            evaluation_result["category_scores"] = category_scores
            evaluation_result["overall_score"] = overall_score
            evaluation_result["benchmark_comparison"] = benchmark_comparison
            evaluation_result["recommendations"] = recommendations
            evaluation_result["simulation_results"] = simulation_results
            
            # 评估质量评级
            evaluation_result["quality_rating"] = self._determine_quality_rating(overall_score)
            evaluation_result["strengths_weaknesses"] = self._identify_strengths_weaknesses(calculated_metrics, benchmark_comparison)
            
            # 计算评估时间
            evaluation_time = time.time() - start_time
            evaluation_result["evaluation_time_seconds"] = evaluation_time
            
            # 更新评估统计
            self._update_evaluation_stats(evaluation_time)
            
            # 记录评估历史
            self._record_evaluation_history(evaluation_result)
            
            logger.info(f"规划系统评估完成: 整体分数 {overall_score:.2f}, 用时 {evaluation_time:.2f} 秒")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"规划系统评估失败: {e}")
            if error_handler:
                error_handler.handle_error(e, "PlanningEvaluator", "规划系统评估失败")
            
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "success": False
            }
    
    def _calculate_time_efficiency(self) -> float:
        """计算时间效率指标"""
        try:
            # 计算总计划时间
            total_planned_time = 0.0
            
            for task in self.planning_system.tasks.values():
                if task.estimated_duration > 0:
                    total_planned_time += task.estimated_duration
            
            for action in self.planning_system.actions.values():
                if action.estimated_duration > 0:
                    total_planned_time += action.estimated_duration
            
            # 如果没有计划时间，返回基准值
            if total_planned_time == 0:
                return 50.0  # 中性基准值
            
            # 分析时间分布合理性
            # 这里使用简化逻辑，实际应用中可能需要更复杂的分析
            time_efficiency = 75.0  # 基准效率
            
            # 基于时间分布调整
            task_count = len(self.planning_system.tasks)
            action_count = len(self.planning_system.actions)
            
            if task_count > 0 and action_count > 0:
                # 任务与行动的时间分配比例
                task_time_ratio = sum(t.estimated_duration for t in self.planning_system.tasks.values() if t.estimated_duration > 0) / total_planned_time
                
                if 0.3 <= task_time_ratio <= 0.7:
                    time_efficiency += 10.0  # 合理的时间分配
                else:
                    time_efficiency -= 10.0  # 不合理的时间分配
            
            # 限制在合理范围内
            return max(0.0, min(100.0, time_efficiency))
            
        except Exception as e:
            logger.error(f"计算时间效率失败: {e}")
            return 50.0  # 错误时返回中性值
    
    def _calculate_resource_efficiency(self) -> float:
        """计算资源效率指标"""
        try:
            # 检查是否有规划上下文
            if not self.planning_system.current_context:
                return 50.0  # 中性基准值
            
            available_resources = self.planning_system.current_context.available_resources
            
            if not available_resources:
                return 50.0
            
            # 计算总资源需求和利用率
            total_required = 0.0
            total_available = 0.0
            
            for resource, amount in available_resources.items():
                total_available += amount
            
            # 计算行动的资源需求
            for action in self.planning_system.actions.values():
                for resource, amount in action.resources.items():
                    total_required += amount
            
            # 计算资源效率
            if total_available > 0:
                utilization_rate = (total_required / total_available) * 100
                
                # 理想的资源利用率在60-85%之间
                if 60 <= utilization_rate <= 85:
                    resource_efficiency = 85.0
                elif utilization_rate < 30:
                    resource_efficiency = 40.0  # 利用率过低
                elif utilization_rate > 95:
                    resource_efficiency = 60.0  # 利用率过高，有风险
                else:
                    # 线性插值
                    if utilization_rate < 60:
                        resource_efficiency = 40 + (utilization_rate / 60) * 45
                    else:  # 85 < utilization_rate <= 95
                        resource_efficiency = 85 - ((utilization_rate - 85) / 10) * 25
            else:
                resource_efficiency = 50.0
            
            return max(0.0, min(100.0, resource_efficiency))
            
        except Exception as e:
            logger.error(f"计算资源效率失败: {e}")
            return 50.0
    
    def _calculate_planning_completeness(self) -> float:
        """计算规划完整性指标"""
        try:
            completeness_score = 0.0
            component_weights = {
                "goals": 0.3,
                "tasks": 0.4,
                "actions": 0.3
            }
            
            # 检查目标完整性
            goals_complete = 0
            for goal in self.planning_system.goals.values():
                if (goal.description and goal.level and goal.priority is not None):
                    goals_complete += 1
            
            goals_score = (goals_complete / max(1, len(self.planning_system.goals))) * 100
            
            # 检查任务完整性
            tasks_complete = 0
            for task in self.planning_system.tasks.values():
                if (task.description and task.level and task.goal_id and 
                    task.estimated_duration is not None and task.estimated_duration > 0):
                    tasks_complete += 1
            
            tasks_score = (tasks_complete / max(1, len(self.planning_system.tasks))) * 100
            
            # 检查行动完整性
            actions_complete = 0
            for action in self.planning_system.actions.values():
                if (action.description and action.task_id and 
                    action.estimated_duration is not None and action.estimated_duration > 0):
                    actions_complete += 1
            
            actions_score = (actions_complete / max(1, len(self.planning_system.actions))) * 100
            
            # 计算加权总分
            completeness_score = (
                goals_score * component_weights["goals"] +
                tasks_score * component_weights["tasks"] +
                actions_score * component_weights["actions"]
            )
            
            return max(0.0, min(100.0, completeness_score))
            
        except Exception as e:
            logger.error(f"计算规划完整性失败: {e}")
            return 50.0
    
    def _calculate_dependency_coherence(self) -> float:
        """计算依赖一致性指标"""
        try:
            coherence_score = 1.0  # 起始分数
            
            # 检查任务依赖图
            task_graph = self.planning_system.task_dependency_graph
            
            if task_graph.number_of_nodes() > 0:
                # 检查循环依赖
                try:
                    cycles = list(nx.simple_cycles(task_graph))
                    if cycles:
                        # 每个循环扣0.1分
                        coherence_score -= min(0.5, len(cycles) * 0.1)
                except:
                    pass
                
                # 检查孤立节点（没有依赖关系的任务）
                isolated_nodes = sum(1 for node in task_graph.nodes() 
                                   if task_graph.degree(node) == 0)
                total_nodes = task_graph.number_of_nodes()
                
                if total_nodes > 0:
                    isolated_ratio = isolated_nodes / total_nodes
                    if isolated_ratio > 0.3:  # 超过30%的孤立节点
                        coherence_score -= 0.2
            
            # 检查行动依赖图
            action_graph = self.planning_system.action_dependency_graph
            
            if action_graph.number_of_nodes() > 0:
                # 检查循环依赖
                try:
                    cycles = list(nx.simple_cycles(action_graph))
                    if cycles:
                        coherence_score -= min(0.3, len(cycles) * 0.05)
                except:
                    pass
            
            # 检查任务与行动的一致性
            for action in self.planning_system.actions.values():
                if action.task_id not in self.planning_system.tasks:
                    coherence_score -= 0.01  # 每个不一致的行动扣0.01分
            
            # 检查任务与目标的一致性
            for task in self.planning_system.tasks.values():
                if task.goal_id not in self.planning_system.goals:
                    coherence_score -= 0.02  # 每个不一致的任务扣0.02分
            
            return max(0.0, min(1.0, coherence_score))
            
        except Exception as e:
            logger.error(f"计算依赖一致性失败: {e}")
            return 0.5
    
    def _calculate_risk_level(self) -> float:
        """计算风险水平指标"""
        try:
            risk_score = 0.0
            risk_factors = []
            
            # 1. 资源风险
            if self.planning_system.current_context:
                available_resources = self.planning_system.current_context.available_resources
                total_required = defaultdict(float)
                
                for action in self.planning_system.actions.values():
                    for resource, amount in action.resources.items():
                        total_required[resource] += amount
                
                for resource, required in total_required.items():
                    available = available_resources.get(resource, 0.0)
                    if available > 0:
                        utilization = required / available
                        if utilization > 0.9:
                            risk_factors.append(("资源瓶颈", 0.3))
                        elif utilization > 1.0:
                            risk_factors.append(("资源不足", 0.5))
            
            # 2. 时间风险
            total_duration = 0.0
            for task in self.planning_system.tasks.values():
                if task.estimated_duration > 0:
                    total_duration += task.estimated_duration
            
            for action in self.planning_system.actions.values():
                if action.estimated_duration > 0:
                    total_duration += action.estimated_duration
            
            if total_duration > 168:  # 超过一周（168小时）
                risk_factors.append(("规划时间过长", 0.2))
            
            # 3. 依赖风险
            task_graph = self.planning_system.task_dependency_graph
            
            if task_graph.number_of_nodes() > 0:
                # 检查关键路径长度
                try:
                    # 查找所有源节点（入度为0）
                    source_nodes = [node for node in task_graph.nodes() 
                                   if task_graph.in_degree(node) == 0]
                    
                    # 查找所有汇节点（出度为0）
                    sink_nodes = [node for node in task_graph.nodes() 
                                 if task_graph.out_degree(node) == 0]
                    
                    if source_nodes and sink_nodes:
                        # 计算最长路径作为关键路径长度
                        longest_path = 0
                        for source in source_nodes:
                            for sink in sink_nodes:
                                try:
                                    path = nx.shortest_path_length(task_graph, source, sink)
                                    longest_path = max(longest_path, path)
                                except:
                                    continue
                        
                        if longest_path > 10:  # 关键路径过长
                            risk_factors.append(("关键路径过长", 0.25))
                except:
                    pass
            
            # 4. 复杂性风险
            total_components = (len(self.planning_system.goals) + 
                              len(self.planning_system.tasks) + 
                              len(self.planning_system.actions))
            
            if total_components > 100:
                risk_factors.append(("规划过于复杂", 0.15))
            elif total_components < 3:
                risk_factors.append(("规划过于简单", 0.1))
            
            # 计算总风险分数
            if risk_factors:
                # 取最高风险因子加上其他因子的加权和
                max_risk = max(r[1] for r in risk_factors)
                other_risks = sum(r[1] for r in risk_factors if r[1] < max_risk) * 0.3
                risk_score = min(1.0, max_risk + other_risks)
            
            return risk_score
            
        except Exception as e:
            logger.error(f"计算风险水平失败: {e}")
            return 0.3
    
    def _calculate_adaptability_score(self) -> float:
        """计算适应性分数指标"""
        try:
            adaptability_score = 0.7  # 基准分数
            
            # 1. 并行性分析
            task_graph = self.planning_system.task_dependency_graph
            
            if task_graph.number_of_nodes() > 0:
                # 计算可以并行执行的任务比例
                independent_tasks = []
                for task_id in task_graph.nodes():
                    predecessors = list(task_graph.predecessors(task_id))
                    if not predecessors:
                        independent_tasks.append(task_id)
                
                parallel_ratio = len(independent_tasks) / max(1, task_graph.number_of_nodes())
                
                if parallel_ratio > 0.3:
                    adaptability_score += 0.1
                elif parallel_ratio < 0.1:
                    adaptability_score -= 0.1
            
            # 2. 模块化分析
            total_tasks = len(self.planning_system.tasks)
            total_goals = len(self.planning_system.goals)
            
            if total_goals > 0:
                tasks_per_goal = total_tasks / total_goals
                if 2 <= tasks_per_goal <= 8:  # 适中的任务/目标比例
                    adaptability_score += 0.05
                elif tasks_per_goal > 15:  # 目标分解过细
                    adaptability_score -= 0.05
            
            # 3. 资源灵活性分析
            if self.planning_system.current_context:
                available_resources = self.planning_system.current_context.available_resources
                resource_count = len(available_resources)
                
                if resource_count >= 3:  # 多种资源类型
                    adaptability_score += 0.05
                elif resource_count == 1:  # 单一资源类型
                    adaptability_score -= 0.05
            
            # 4. 时间缓冲分析
            total_duration = 0.0
            for task in self.planning_system.tasks.values():
                if task.estimated_duration > 0:
                    total_duration += task.estimated_duration
            
            if total_duration > 0:
                # 检查是否有时间缓冲（通过对比估计时间和实际可能时间）
                # 这里使用简化逻辑
                estimated_buffer = 0.0
                
                # 假设每个任务有10%的时间缓冲
                for task in self.planning_system.tasks.values():
                    if task.estimated_duration > 0:
                        estimated_buffer += task.estimated_duration * 0.1
                
                buffer_ratio = estimated_buffer / total_duration
                if buffer_ratio > 0.15:
                    adaptability_score += 0.1
                elif buffer_ratio < 0.05:
                    adaptability_score -= 0.05
            
            return max(0.0, min(1.0, adaptability_score))
            
        except Exception as e:
            logger.error(f"计算适应性分数失败: {e}")
            return 0.5
    
    def _calculate_cost_benefit_ratio(self) -> float:
        """计算成本效益比指标"""
        try:
            # 简化的成本效益计算
            # 成本因素
            total_cost = 0.0
            
            # 1. 时间成本（假设每小时成本为1）
            for task in self.planning_system.tasks.values():
                if task.estimated_duration > 0:
                    total_cost += task.estimated_duration * 1.0
            
            for action in self.planning_system.actions.values():
                if action.estimated_duration > 0:
                    total_cost += action.estimated_duration * 1.0
            
            # 2. 资源成本（简化处理）
            if self.planning_system.current_context:
                available_resources = self.planning_system.current_context.available_resources
                resource_cost = sum(available_resources.values()) * 0.1  # 假设资源成本因子
                total_cost += resource_cost
            
            # 效益因素
            total_benefit = 0.0
            
            # 1. 目标优先级加权
            for goal in self.planning_system.goals.values():
                goal_benefit = goal.priority * 10.0  # 优先级转换为效益值
                total_benefit += goal_benefit
            
            # 2. 任务完成度效益
            completed_tasks = sum(1 for task in self.planning_system.tasks.values() 
                                if task.status == GoalStatus.COMPLETED)
            total_tasks = len(self.planning_system.tasks)
            
            if total_tasks > 0:
                completion_ratio = completed_tasks / total_tasks
                total_benefit += completion_ratio * 20.0
            
            # 计算成本效益比
            if total_cost > 0:
                cost_benefit_ratio = total_benefit / total_cost
            else:
                cost_benefit_ratio = 1.0  # 零成本的默认值
            
            # 限制在合理范围内
            return max(0.1, min(10.0, cost_benefit_ratio))
            
        except Exception as e:
            logger.error(f"计算成本效益比失败: {e}")
            return 1.5
    
    def _calculate_category_scores(self, metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """计算类别分数"""
        category_scores = {}
        category_metrics = defaultdict(list)
        
        # 按类别分组指标
        for metric_id, metric_data in metrics.items():
            if metric_data.get("value") is not None:
                category = metric_data.get("category")
                if category:
                    category_metrics[category].append(metric_data)
        
        # 计算每个类别的平均分数
        for category, metric_list in category_metrics.items():
            valid_metrics = [m for m in metric_list if m.get("value") is not None]
            
            if valid_metrics:
                # 计算加权平均
                total_weight = sum(m.get("weight", 0.1) for m in valid_metrics)
                weighted_sum = 0.0
                
                for metric in valid_metrics:
                    value = metric["value"]
                    weight = metric.get("weight", 0.1)
                    
                    # 对于逆向指标（值越低越好），需要反转
                    if metric.get("inverse", False):
                        # 将[0,1]范围内的值转换为[0,100]的分数
                        if 0 <= value <= 1:
                            normalized_value = (1 - value) * 100
                        else:
                            normalized_value = max(0, min(100, 100 - value))
                    else:
                        # 标准化到[0,100]范围
                        if 0 <= value <= 1:
                            normalized_value = value * 100
                        elif 0 <= value <= 100:
                            normalized_value = value
                        else:
                            # 尝试标准化
                            normalized_value = max(0, min(100, value))
                    
                    weighted_sum += normalized_value * weight
                
                category_score = weighted_sum / max(0.001, total_weight)
                category_scores[category] = max(0.0, min(100.0, category_score))
            else:
                category_scores[category] = 50.0  # 中性值
        
        return category_scores
    
    def _calculate_overall_score(self, metrics: Dict[str, Dict[str, Any]], 
                                category_scores: Dict[str, float]) -> float:
        """计算整体分数"""
        try:
            # 方法1：基于指标权重的加权平均
            weighted_sum = 0.0
            total_weight = 0.0
            
            for metric_id, metric_data in metrics.items():
                if metric_data.get("value") is not None:
                    value = metric_data["value"]
                    weight = metric_data.get("weight", 0.1)
                    
                    # 标准化值
                    if metric_data.get("inverse", False):
                        # 逆向指标：值越低越好
                        if 0 <= value <= 1:
                            normalized_value = (1 - value) * 100
                        else:
                            normalized_value = max(0, min(100, 100 - value))
                    else:
                        # 正向指标
                        if 0 <= value <= 1:
                            normalized_value = value * 100
                        elif 0 <= value <= 100:
                            normalized_value = value
                        else:
                            normalized_value = max(0, min(100, value))
                    
                    weighted_sum += normalized_value * weight
                    total_weight += weight
            
            method1_score = weighted_sum / max(0.001, total_weight) if total_weight > 0 else 50.0
            
            # 方法2：基于类别分数的平均
            if category_scores:
                method2_score = sum(category_scores.values()) / len(category_scores)
            else:
                method2_score = 50.0
            
            # 结合两种方法（加权平均）
            overall_score = method1_score * 0.6 + method2_score * 0.4
            
            return max(0.0, min(100.0, overall_score))
            
        except Exception as e:
            logger.error(f"计算整体分数失败: {e}")
            return 50.0
    
    def _compare_with_benchmarks(self, metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """与基准比较"""
        comparison_results = {}
        
        for metric_id, metric_data in metrics.items():
            if metric_data.get("value") is not None and metric_id in self.benchmarks:
                value = metric_data["value"]
                benchmark = self.benchmarks[metric_id]
                
                # 确定评级
                rating = "unknown"
                
                if "excellent" in benchmark and value >= benchmark["excellent"]:
                    rating = "excellent"
                elif "good" in benchmark and value >= benchmark["good"]:
                    rating = "good"
                elif "fair" in benchmark and value >= benchmark["fair"]:
                    rating = "fair"
                else:
                    rating = "poor"
                
                # 计算与基准的差距
                if rating == "excellent":
                    gap_to_next = 0.0
                elif rating == "good":
                    gap_to_next = benchmark.get("excellent", value + 1) - value
                elif rating == "fair":
                    gap_to_next = benchmark.get("good", value + 1) - value
                else:  # poor
                    gap_to_next = benchmark.get("fair", value + 1) - value
                
                comparison_results[metric_id] = {
                    "value": value,
                    "rating": rating,
                    "gap_to_next_level": max(0.0, gap_to_next),
                    "benchmark_levels": benchmark.copy()
                }
        
        return comparison_results
    
    def _generate_evaluation_recommendations(self, metrics: Dict[str, Dict[str, Any]], 
                                           comparison: Dict[str, Any]) -> List[str]:
        """生成评估建议"""
        recommendations = []
        
        # 基于指标评级的建议
        for metric_id, comp_data in comparison.items():
            rating = comp_data.get("rating")
            value = comp_data.get("value")
            gap = comp_data.get("gap_to_next_level", 0.0)
            
            metric_info = metrics.get(metric_id, {})
            metric_name = metric_info.get("name", metric_id)
            
            if rating == "poor":
                recommendations.append(
                    f"指标'{metric_name}'评级为'差'({value:.2f})。需要重点关注和改进。"
                )
            elif rating == "fair" and gap > 0:
                recommendations.append(
                    f"指标'{metric_name}'评级为'一般'({value:.2f})。需要改进以达到'良好'级别(差距: {gap:.2f})。"
                )
            elif rating == "good" and gap > 0:
                recommendations.append(
                    f"指标'{metric_name}'评级为'良好'({value:.2f})。可以进一步优化以达到'优秀'级别(差距: {gap:.2f})。"
                )
        
        # 基于整体评级的建议
        overall_rating = self._determine_quality_rating(
            sum(m.get("value", 50) for m in metrics.values()) / max(1, len(metrics))
        )
        
        if overall_rating == "poor":
            recommendations.append("整体规划质量评级为'差'。需要全面审查和改进规划系统。")
        elif overall_rating == "fair":
            recommendations.append("整体规划质量评级为'一般'。建议针对关键指标进行优化。")
        elif overall_rating == "good":
            recommendations.append("整体规划质量评级为'良好'。规划系统运行正常，可以进一步优化。")
        elif overall_rating == "excellent":
            recommendations.append("整体规划质量评级为'优秀'。规划系统表现优异。")
        
        return recommendations
    
    def _perform_simulation_evaluation(self) -> Dict[str, Any]:
        """执行模拟评估"""
        try:
            simulation_results = {
                "timestamp": datetime.now().isoformat(),
                "iterations": self.config.get("simulation_iterations", 100),
                "success_rate": 0.0,
                "average_completion_time": 0.0,
                "resource_utilization": {},
                "risk_analysis": {},
                "bottlenecks": []
            }
            
            # 简化模拟逻辑
            # 在实际应用中，这里应该实现更复杂的蒙特卡洛模拟
            
            # 模拟成功率和完成时间
            if len(self.planning_system.tasks) > 0:
                # 基于任务复杂度的简单模拟
                total_tasks = len(self.planning_system.tasks)
                completed_tasks = sum(1 for t in self.planning_system.tasks.values() 
                                    if t.status == GoalStatus.COMPLETED)
                
                base_success_rate = completed_tasks / max(1, total_tasks) * 100
                
                # 添加随机性
                import random
                success_rate_variation = random.uniform(-10, 10)
                simulation_results["success_rate"] = max(0, min(100, base_success_rate + success_rate_variation))
                
                # 模拟平均完成时间
                total_duration = sum(t.estimated_duration for t in self.planning_system.tasks.values() 
                                   if t.estimated_duration > 0)
                avg_task_duration = total_duration / max(1, total_tasks)
                
                # 考虑不确定性
                duration_variation = random.uniform(0.8, 1.2)
                simulation_results["average_completion_time"] = avg_task_duration * duration_variation
            else:
                simulation_results["success_rate"] = 100.0
                simulation_results["average_completion_time"] = 0.0
            
            # 模拟资源利用率
            if self.planning_system.current_context:
                available_resources = self.planning_system.current_context.available_resources
                
                for resource, available in available_resources.items():
                    # 简化模拟：随机利用率
                    import random
                    utilization = random.uniform(0.5, 0.9)
                    simulation_results["resource_utilization"][resource] = {
                        "available": available,
                        "simulated_used": available * utilization,
                        "utilization_rate": utilization * 100
                    }
            
            # 风险分析
            risk_factors = []
            
            # 依赖风险
            task_graph = self.planning_system.task_dependency_graph
            if task_graph.number_of_nodes() > 0:
                try:
                    # 检查最长路径
                    longest_path = 0
                    for node in task_graph.nodes():
                        try:
                            path_length = nx.dag_longest_path_length(task_graph)
                            longest_path = max(longest_path, path_length)
                        except:
                            pass
                    
                    if longest_path > 5:
                        risk_factors.append({
                            "type": "dependency_risk",
                            "severity": "medium",
                            "description": f"依赖路径较长({longest_path}个任务)，可能增加延迟风险"
                        })
                except:
                    pass
            
            # 资源风险
            if simulation_results["resource_utilization"]:
                high_utilization_resources = [
                    r for r, data in simulation_results["resource_utilization"].items()
                    if data["utilization_rate"] > 85
                ]
                
                if high_utilization_resources:
                    risk_factors.append({
                        "type": "resource_risk",
                        "severity": "high" if any(data["utilization_rate"] > 95 for data in simulation_results["resource_utilization"].values()) else "medium",
                        "description": f"资源利用率高: {', '.join(high_utilization_resources)}"
                    })
            
            simulation_results["risk_analysis"]["risk_factors"] = risk_factors
            simulation_results["risk_analysis"]["overall_risk_level"] = "low" if not risk_factors else "medium"
            
            # 瓶颈分析
            if risk_factors:
                simulation_results["bottlenecks"] = [
                    f"{rf['description']} (严重性: {rf['severity']})"
                    for rf in risk_factors
                ]
            
            return simulation_results
            
        except Exception as e:
            logger.error(f"执行模拟评估失败: {e}")
            return {"error": str(e)}
    
    def _determine_quality_rating(self, score: float) -> str:
        """确定质量评级"""
        if score >= 85:
            return "excellent"
        elif score >= 70:
            return "good"
        elif score >= 50:
            return "fair"
        else:
            return "poor"
    
    def _identify_strengths_weaknesses(self, metrics: Dict[str, Dict[str, Any]], 
                                     comparison: Dict[str, Any]) -> Dict[str, List[str]]:
        """识别强项和弱项"""
        strengths = []
        weaknesses = []
        
        for metric_id, comp_data in comparison.items():
            rating = comp_data.get("rating")
            metric_info = metrics.get(metric_id, {})
            metric_name = metric_info.get("name", metric_id)
            
            if rating == "excellent":
                strengths.append(f"{metric_name}: 优秀表现")
            elif rating == "good":
                strengths.append(f"{metric_name}: 良好表现")
            elif rating == "poor":
                weaknesses.append(f"{metric_name}: 需要重大改进")
            elif rating == "fair":
                weaknesses.append(f"{metric_name}: 需要改进")
        
        # 如果指标太少，添加通用建议
        if not strengths and not weaknesses:
            if len(metrics) == 0:
                strengths.append("系统简单，易于管理")
                weaknesses.append("缺乏详细规划，可能无法处理复杂场景")
            elif len(metrics) < 3:
                strengths.append("基本规划功能完整")
                weaknesses.append("评估维度有限，建议扩展评估指标")
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses
        }
    
    def _update_evaluation_stats(self, evaluation_time: float):
        """更新评估统计"""
        self.evaluation_stats["evaluations_performed"] += 1
        self.evaluation_stats["total_evaluation_time"] += evaluation_time
        
        n = self.evaluation_stats["evaluations_performed"]
        current_avg = self.evaluation_stats["average_evaluation_time"]
        
        self.evaluation_stats["average_evaluation_time"] = (
            current_avg * (n - 1) + evaluation_time
        ) / n
        
        self.evaluation_stats["last_evaluation_time"] = datetime.now().isoformat()
    
    def _record_evaluation_history(self, evaluation_result: Dict[str, Any]):
        """记录评估历史"""
        history_entry = {
            "timestamp": evaluation_result.get("timestamp", datetime.now().isoformat()),
            "overall_score": evaluation_result.get("overall_score", 0.0),
            "quality_rating": evaluation_result.get("quality_rating", "unknown"),
            "evaluation_time": evaluation_result.get("evaluation_time_seconds", 0.0)
        }
        
        self.evaluation_history.append(history_entry)
        
        # 限制历史记录大小
        if len(self.evaluation_history) > self.max_history_size:
            self.evaluation_history = self.evaluation_history[-self.max_history_size:]
    
    def get_evaluation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取评估历史"""
        if limit:
            return self.evaluation_history[-limit:]
        return self.evaluation_history.copy()
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """获取评估统计信息"""
        return self.evaluation_stats.copy()


def test_planning_validation_framework():
    """测试规划验证与评估框架"""
    logger.info("开始测试规划验证与评估框架")
    
    try:
        # 创建示例规划系统
        from .hierarchical_planning_system import create_example_planning_system
        planning_system = create_example_planning_system()
        
        # 测试规划验证器
        logger.info("测试规划验证器...")
        validator = PlanningValidator(planning_system)
        validation_report = validator.validate_planning_system()
        
        logger.info(f"验证完成: {validation_report.total_validations} 个验证结果")
        logger.info(f"严重问题: {validation_report.critical_count}, 错误: {validation_report.error_count}, 警告: {validation_report.warning_count}")
        logger.info(f"整体状态: {validation_report.get_overall_status()}")
        
        # 测试规划评估器
        logger.info("测试规划评估器...")
        evaluator = PlanningEvaluator(planning_system)
        evaluation_result = evaluator.evaluate_planning_system(detailed_report=True, include_simulation=True)
        
        if "error" not in evaluation_result:
            logger.info(f"评估完成: 整体分数 {evaluation_result.get('overall_score', 0):.2f}")
            logger.info(f"质量评级: {evaluation_result.get('quality_rating', 'unknown')}")
            
            # 显示类别分数
            category_scores = evaluation_result.get('category_scores', {})
            for category, score in category_scores.items():
                logger.info(f"  {category}: {score:.2f}")
            
            # 显示建议
            recommendations = evaluation_result.get('recommendations', [])
            if recommendations:
                logger.info("建议:")
                for rec in recommendations[:3]:  # 只显示前3个建议
                    logger.info(f"  - {rec}")
        
        logger.info("规划验证与评估框架测试完成")
        
        return {
            "validation_report": validation_report,
            "evaluation_result": evaluation_result,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"测试规划验证与评估框架失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_result = test_planning_validation_framework()
    
    if test_result["success"]:
        print("规划验证与评估框架测试成功！")
    else:
        print(f"测试失败: {test_result.get('error', '未知错误')}")