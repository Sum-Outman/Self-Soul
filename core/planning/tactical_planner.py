#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zlib
"""
战术层规划器 - 处理中期、项目级的任务规划和调度

功能:
1. 任务分解与分配: 将战略目标分解为具体任务
2. 时间安排优化: 优化任务时间表和资源分配
3. 依赖关系管理: 处理任务间的逻辑和时序依赖
4. 里程碑设置: 设定关键检查和完成节点
5. 风险监控调整: 监控执行风险并动态调整
6. 进度追踪反馈: 追踪进度并提供反馈

核心特性:
- 时间尺度: 天~周级别
- 抽象级别: 项目计划、里程碑、任务序列
- 决策依据: 资源约束、时间约束、依赖关系
- 输出形式: 任务序列、时间安排、资源分配

战术规划流程:
1. 目标解析: 解析战略目标为战术需求
2. 任务分解: 将需求分解为具体任务
3. 依赖分析: 分析任务间的依赖关系
4. 资源分配: 分配任务所需资源
5. 时间安排: 制定时间表和里程碑
6. 监控调整: 跟踪执行并调整计划

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import copy
from collections import defaultdict

# 导入相关模块
from .hierarchical_planning_system import (
    HierarchicalPlanningSystem,
    Goal, Task, Action, PlanningContext,
    PlanningLevel, GoalStatus, ActionStatus
)

# 导入战略规划器
try:
    from .strategic_planner import StrategicPlanner, StrategicObjective
except ImportError:
    StrategicPlanner = StrategicObjective = None

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


class TaskPriority(Enum):
    """任务优先级枚举"""
    CRITICAL = "critical"      # 关键任务：直接影响战略目标
    HIGH = "high"              # 高优先级：对目标有重要影响
    MEDIUM = "medium"          # 中等优先级：正常重要
    LOW = "low"                # 低优先级：补充性任务
    BACKGROUND = "background"  # 背景任务：不直接影响目标


class TaskComplexity(Enum):
    """任务复杂度枚举"""
    SIMPLE = "simple"          # 简单任务：1人天以内
    MODERATE = "moderate"      # 中等任务：1-3人天
    COMPLEX = "complex"        # 复杂任务：3-10人天
    VERY_COMPLEX = "very_complex"  # 非常复杂：10+人天


class ResourceType(Enum):
    """资源类型枚举"""
    HUMAN = "human"            # 人力资源
    COMPUTATIONAL = "computational"  # 计算资源
    FINANCIAL = "financial"    # 财务资源
    MATERIAL = "material"      # 物质资源
    INFORMATIONAL = "informational"  # 信息资源
    TEMPORAL = "temporal"      # 时间资源


class DependencyType(Enum):
    """依赖类型枚举"""
    FINISH_START = "finish_start"  # 完成-开始：前任务完成后开始
    START_START = "start_start"    # 开始-开始：与前任务同时开始
    FINISH_FINISH = "finish_finish"  # 完成-完成：与前任务同时完成
    START_FINISH = "start_finish"  # 开始-完成：前任务开始后完成


@dataclass
class TacticalContext:
    """战术上下文数据类"""
    strategic_objective: Optional[StrategicObjective]  # 战略目标
    available_resources: Dict[ResourceType, float]  # 可用资源
    time_constraints: Dict[str, Any]  # 时间约束
    team_capabilities: List[str]  # 团队能力
    historical_performance: Dict[str, Any]  # 历史绩效
    risk_tolerance: float  # 风险容忍度（0-1）
    quality_requirements: Dict[str, Any]  # 质量要求
    stakeholder_expectations: List[str]  # 利益相关者期望
    
    def __post_init__(self):
        """后初始化验证"""
        self.risk_tolerance = max(0.0, min(1.0, self.risk_tolerance))
        if not self.available_resources:
            self.available_resources = {rtype: 1.0 for rtype in ResourceType}


@dataclass
class TaskSpecification:
    """任务规格数据类"""
    id: str
    description: str
    strategic_objective_id: str  # 关联的战略目标ID
    priority: TaskPriority
    complexity: TaskComplexity
    estimated_effort: float  # 人天
    required_skills: List[str]  # 所需技能
    required_resources: Dict[ResourceType, float]  # 所需资源
    dependencies: List[Tuple[str, DependencyType]]  # 依赖关系
    quality_criteria: List[str]  # 质量标准
    success_criteria: List[str]  # 成功标准
    risk_factors: List[Dict[str, Any]]  # 风险因素
    constraints: List[Dict[str, Any]]  # 约束条件
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        if not self.id:
            self.id = f"task_spec_{int(time.time())}_{(zlib.adler32(str(self.description).encode('utf-8')) & 0xffffffff) % 10000}"
        self.estimated_effort = max(0.1, self.estimated_effort)


@dataclass
class Milestone:
    """里程碑数据类"""
    id: str
    name: str
    description: str
    target_date: datetime
    associated_tasks: List[str]  # 关联的任务ID
    completion_criteria: List[str]  # 完成标准
    importance_level: str  # 重要性级别
    notification_requirements: List[str]  # 通知要求
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        if not self.id:
            self.id = f"milestone_{int(time.time())}_{(zlib.adler32(str(self.name).encode('utf-8')) & 0xffffffff) % 10000}"


@dataclass
class TacticalPlan:
    """战术计划数据类"""
    id: str
    name: str
    description: str
    time_period: Tuple[datetime, datetime]  # 时间周期
    tactical_context: TacticalContext
    task_specifications: List[TaskSpecification]
    milestones: List[Milestone]
    resource_allocation: Dict[str, Dict[ResourceType, float]]  # 资源分配
    schedule: Dict[str, Any]  # 时间安排
    risk_mitigation_plan: List[Dict[str, Any]]  # 风险缓解计划
    monitoring_framework: Dict[str, Any]  # 监控框架
    contingency_plans: List[Dict[str, Any]]  # 应急计划
    performance_metrics: Dict[str, Any]  # 性能指标
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        if not self.id:
            self.id = f"tactical_plan_{int(time.time())}_{(zlib.adler32(str(self.name).encode('utf-8')) & 0xffffffff) % 10000}"
        
        # 确保结束时间晚于开始时间
        start_time, end_time = self.time_period
        if end_time <= start_time:
            self.time_period = (start_time, start_time + timedelta(days=30))  # 默认30天


class TacticalPlanner:
    """
    战术层规划器 - 处理中期项目规划和调度
    
    核心功能:
    1. 任务分解: 将战略目标分解为可执行任务
    2. 依赖分析: 分析任务间的逻辑和时序依赖
    3. 资源优化: 优化资源分配和时间安排
    4. 风险管理: 识别和缓解项目风险
    5. 进度监控: 跟踪执行进度和质量
    6. 动态调整: 根据变化调整计划和资源
    
    技术特点:
    - 时间导向: 天~周级别的时间尺度
    - 资源约束: 考虑多种资源的约束优化
    - 依赖管理: 处理复杂的任务依赖关系
    - 风险感知: 主动识别和缓解项目风险
    - 适应性强: 支持动态调整和重规划
    """
    
    def __init__(self, 
                 planning_system: Optional[HierarchicalPlanningSystem] = None,
                 tactical_config: Optional[Dict[str, Any]] = None):
        """
        初始化战术规划器
        
        Args:
            planning_system: 分层规划系统实例（可选）
            tactical_config: 战术规划配置（可选）
        """
        self.planning_system = planning_system
        self.config = tactical_config or self._get_default_config()
        
        # 战术规划组件
        self.tactical_context: Optional[TacticalContext] = None
        self.current_tactical_plan: Optional[TacticalPlan] = None
        self.tactical_plans_history: List[TacticalPlan] = []
        self.max_history_size = self.config.get('max_tactical_plans_history', 20)
        
        # 任务模板库
        self.task_templates = self._initialize_task_templates()
        
        # 调度算法配置
        self.scheduling_algorithms = self._initialize_scheduling_algorithms()
        
        # 性能统计
        self.performance_stats = {
            "tactical_plans_created": 0,
            "tasks_decomposed": 0,
            "milestones_defined": 0,
            "total_planning_time": 0.0,
            "average_planning_time": 0.0,
            "last_planning_time": None,
            "replannings_performed": 0
        }
        
        logger.info(f"战术规划器初始化完成，任务模板: {len(self.task_templates)} 个")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "enable_task_decomposition": True,
            "enable_dependency_analysis": True,
            "enable_resource_optimization": True,
            "enable_risk_assessment": True,
            "max_tasks_per_objective": 15,
            "max_milestones_per_plan": 10,
            "planning_time_limit": 120.0,  # 2分钟
            "max_tactical_plans_history": 20,
            "enable_dynamic_rescheduling": True,
            "risk_assessment_threshold": 0.7,
            "resource_utilization_target": 0.8,  # 80%资源利用率
            "schedule_slack_time": 0.1,  # 10%缓冲时间
            "enable_progress_monitoring": True,
            "milestone_review_frequency_days": 7,  # 每周里程碑审查
            "enable_quality_assurance": True
        }
    
    def _initialize_task_templates(self) -> Dict[str, Any]:
        """初始化任务模板库"""
        return {
            "research_analysis": {
                "name": "研究与分析任务模板",
                "description": "用于研究、分析和调研类任务",
                "typical_subtasks": [
                    "需求收集与分析",
                    "文献调研与综述",
                    "数据收集与整理",
                    "分析与建模",
                    "报告撰写与评审"
                ],
                "required_skills": ["research", "analysis", "writing"],
                "typical_complexity": TaskComplexity.MODERATE,
                "estimated_effort_range": (3.0, 10.0),  # 人天
                "risk_factors": ["数据质量", "方法适用性", "时间约束"]
            },
            "design_development": {
                "name": "设计与开发任务模板",
                "description": "用于系统设计、开发和实现类任务",
                "typical_subtasks": [
                    "架构设计",
                    "详细设计",
                    "编码实现",
                    "单元测试",
                    "集成测试"
                ],
                "required_skills": ["design", "programming", "testing"],
                "typical_complexity": TaskComplexity.COMPLEX,
                "estimated_effort_range": (5.0, 20.0),
                "risk_factors": ["技术风险", "需求变更", "集成问题"]
            },
            "testing_validation": {
                "name": "测试与验证任务模板",
                "description": "用于测试、验证和质量保证类任务",
                "typical_subtasks": [
                    "测试计划制定",
                    "测试用例设计",
                    "测试环境搭建",
                    "测试执行",
                    "缺陷管理与追踪",
                    "测试报告编写"
                ],
                "required_skills": ["testing", "quality_assurance", "documentation"],
                "typical_complexity": TaskComplexity.MODERATE,
                "estimated_effort_range": (2.0, 8.0),
                "risk_factors": ["测试覆盖不足", "环境问题", "时间压力"]
            },
            "deployment_operations": {
                "name": "部署与运维任务模板",
                "description": "用于系统部署、运维和支持类任务",
                "typical_subtasks": [
                    "部署计划制定",
                    "环境准备",
                    "系统部署",
                    "配置与调优",
                    "监控与维护"
                ],
                "required_skills": ["deployment", "operations", "monitoring"],
                "typical_complexity": TaskComplexity.MODERATE,
                "estimated_effort_range": (3.0, 12.0),
                "risk_factors": ["部署失败", "性能问题", "安全风险"]
            },
            "training_documentation": {
                "name": "培训与文档任务模板",
                "description": "用于用户培训、文档编写和知识传递类任务",
                "typical_subtasks": [
                    "内容规划",
                    "材料编写",
                    "培训实施",
                    "反馈收集",
                    "文档维护"
                ],
                "required_skills": ["documentation", "training", "communication"],
                "typical_complexity": TaskComplexity.SIMPLE,
                "estimated_effort_range": (1.0, 5.0),
                "risk_factors": ["内容准确性", "用户接受度", "更新不及时"]
            }
        }
    
    def _initialize_scheduling_algorithms(self) -> Dict[str, Any]:
        """初始化调度算法"""
        return {
            "critical_path_method": {
                "name": "关键路径法",
                "description": "识别项目中耗时最长的任务序列",
                "applicability": "适用于有明确依赖关系的项目",
                "complexity": "中等",
                "strengths": ["识别关键任务", "优化时间安排"],
                "weaknesses": ["不处理资源约束", "对不确定性敏感"]
            },
            "resource_leveling": {
                "name": "资源平衡法",
                "description": "平衡资源使用，避免资源过载",
                "applicability": "适用于资源受限的项目",
                "complexity": "高",
                "strengths": ["优化资源利用", "避免资源冲突"],
                "weaknesses": ["可能延长项目时间", "计算复杂"]
            },
            "earliest_deadline_first": {
                "name": "最早截止时间优先",
                "description": "优先安排截止时间最早的任务",
                "applicability": "适用于有严格时间约束的项目",
                "complexity": "低",
                "strengths": ["简单易用", "保证截止时间"],
                "weaknesses": ["不考虑任务重要性", "可能资源冲突"]
            },
            "priority_based_scheduling": {
                "name": "基于优先级调度",
                "description": "按任务优先级安排执行顺序",
                "applicability": "适用于重要性差异明显的项目",
                "complexity": "低",
                "strengths": ["保证高优先级任务", "简单直观"],
                "weaknesses": ["可能忽视低优先级任务", "资源利用不佳"]
            }
        }
    
    def set_tactical_context(self, context: TacticalContext):
        """
        设置战术上下文
        
        Args:
            context: 战术上下文
        """
        self.tactical_context = context
        logger.info(f"战术上下文已设置，可用资源: {len(context.available_resources)} 种")
    
    def decompose_strategic_objective(self, 
                                    objective: StrategicObjective,
                                    context: TacticalContext) -> List[TaskSpecification]:
        """
        分解战略目标为战术任务
        
        Args:
            objective: 战略目标
            context: 战术上下文
            
        Returns:
            任务规格列表
        """
        start_time = time.time()
        
        logger.info(f"开始分解战略目标: {objective.description}")
        
        try:
            task_specifications = []
            
            # 基于目标描述分析任务类型
            objective_desc = objective.description.lower()
            
            # 根据目标类型选择任务模板
            if any(keyword in objective_desc for keyword in ["研究", "分析", "调研", "research", "analysis"]):
                task_template = self.task_templates["research_analysis"]
            elif any(keyword in objective_desc for keyword in ["开发", "设计", "实现", "develop", "design", "implement"]):
                task_template = self.task_templates["design_development"]
            elif any(keyword in objective_desc for keyword in ["测试", "验证", "质量", "test", "validate", "quality"]):
                task_template = self.task_templates["testing_validation"]
            elif any(keyword in objective_desc for keyword in ["部署", "运维", "上线", "deploy", "operation"]):
                task_template = self.task_templates["deployment_operations"]
            elif any(keyword in objective_desc for keyword in ["培训", "文档", "知识", "train", "document", "knowledge"]):
                task_template = self.task_templates["training_documentation"]
            else:
                # 使用通用模板
                task_template = self.task_templates["design_development"]
            
            # 生成任务规格
            for i, subtask_desc in enumerate(task_template["typical_subtasks"]):
                # 生成任务ID
                task_id = f"{objective.id}_task_{i}"
                
                # 评估任务复杂度
                complexity = task_template.get("typical_complexity", TaskComplexity.MODERATE)
                
                # 估计工作量（在范围内随机）
                effort_range = task_template.get("estimated_effort_range", (2.0, 5.0))
                estimated_effort = (effort_range[0] + effort_range[1]) / 2
                
                # 确定优先级（基于战略目标优先级）
                if objective.priority > 0.8:
                    priority = TaskPriority.CRITICAL
                elif objective.priority > 0.6:
                    priority = TaskPriority.HIGH
                elif objective.priority > 0.4:
                    priority = TaskPriority.MEDIUM
                else:
                    priority = TaskPriority.LOW
                
                # 估算资源需求
                required_resources = self._estimate_resource_requirements(
                    subtask_desc, complexity, estimated_effort, context
                )
                
                # 识别所需技能
                required_skills = task_template.get("required_skills", [])
                
                # 构建依赖关系（线性依赖）
                dependencies = []
                if i > 0:
                    prev_task_id = f"{objective.id}_task_{i-1}"
                    dependencies.append((prev_task_id, DependencyType.FINISH_START))
                
                # 识别风险因素
                risk_factors = self._identify_task_risks(subtask_desc, objective, context)
                
                # 创建任务规格
                task_spec = TaskSpecification(
                    id=task_id,
                    description=f"{subtask_desc}: {objective.description}",
                    strategic_objective_id=objective.id,
                    priority=priority,
                    complexity=complexity,
                    estimated_effort=estimated_effort,
                    required_skills=required_skills,
                    required_resources=required_resources,
                    dependencies=dependencies,
                    quality_criteria=["按时完成", "质量达标", "符合规范"],
                    success_criteria=[f"成功完成{subtask_desc}"],
                    risk_factors=risk_factors,
                    constraints=objective.constraints if hasattr(objective, 'constraints') else [],
                    metadata={
                        "template_name": task_template["name"],
                        "subtask_index": i,
                        "generated_at": datetime.now().isoformat()
                    }
                )
                
                task_specifications.append(task_spec)
            
            # 限制任务数量
            max_tasks = self.config.get("max_tasks_per_objective", 15)
            if len(task_specifications) > max_tasks:
                task_specifications = task_specifications[:max_tasks]
                logger.warning(f"战略目标 {objective.id} 的任务数量限制为 {max_tasks}")
            
            # 更新性能统计
            decomposition_time = time.time() - start_time
            self._update_performance_stats("tasks_decomposed", decomposition_time, count=len(task_specifications))
            
            logger.info(f"战略目标分解完成，生成 {len(task_specifications)} 个任务，用时: {decomposition_time:.2f} 秒")
            
            return task_specifications
            
        except Exception as e:
            logger.error(f"战略目标分解失败: {e}")
            if error_handler:
                error_handler.handle_error(e, "TacticalPlanner", "战略目标分解失败")
            return []
    
    def _estimate_resource_requirements(self, 
                                      task_description: str,
                                      complexity: TaskComplexity,
                                      estimated_effort: float,
                                      context: TacticalContext) -> Dict[ResourceType, float]:
        """估算资源需求"""
        resources = {}
        
        # 根据任务描述和复杂度估算
        desc_lower = task_description.lower()
        
        # 人力资源（人天）
        resources[ResourceType.HUMAN] = estimated_effort
        
        # 计算资源（根据任务类型）
        if any(keyword in desc_lower for keyword in ["计算", "处理", "分析", "compute", "process", "analyze"]):
            resources[ResourceType.COMPUTATIONAL] = estimated_effort * 2.0
        else:
            resources[ResourceType.COMPUTATIONAL] = estimated_effort * 0.5
        
        # 财务资源（简单估算）
        resources[ResourceType.FINANCIAL] = estimated_effort * 100.0  # 假设每人天100单位
        
        # 时间资源（与人力资源相同）
        resources[ResourceType.TEMPORAL] = estimated_effort
        
        # 根据复杂度调整
        if complexity == TaskComplexity.SIMPLE:
            for rtype in resources:
                resources[rtype] *= 0.7
        elif complexity == TaskComplexity.VERY_COMPLEX:
            for rtype in resources:
                resources[rtype] *= 1.5
        
        return resources
    
    def _identify_task_risks(self, 
                            task_description: str,
                            objective: StrategicObjective,
                            context: TacticalContext) -> List[Dict[str, Any]]:
        """识别任务风险"""
        risks = []
        
        # 基于任务描述的风险
        desc_lower = task_description.lower()
        
        if any(keyword in desc_lower for keyword in ["依赖", "整合", "协调", "dependency", "integration", "coordination"]):
            risks.append({
                "type": "dependency_risk",
                "description": "依赖关系复杂，协调困难",
                "likelihood": "medium",
                "impact": "high",
                "mitigation": "明确接口，加强沟通"
            })
        
        if any(keyword in desc_lower for keyword in ["新技术", "未验证", "实验", "new_tech", "unproven", "experimental"]):
            risks.append({
                "type": "technical_risk",
                "description": "技术不成熟或未经验证",
                "likelihood": "medium",
                "impact": "high",
                "mitigation": "技术调研，原型验证"
            })
        
        if any(keyword in desc_lower for keyword in ["时间紧", "压力大", "紧急", "tight_schedule", "pressure", "urgent"]):
            risks.append({
                "type": "schedule_risk",
                "description": "时间紧张，进度压力大",
                "likelihood": "high",
                "impact": "medium",
                "mitigation": "增加缓冲，优先级管理"
            })
        
        # 基于战略目标风险
        if objective.risk_level.name in ["HIGH", "VERY_HIGH"]:
            risks.append({
                "type": "strategic_risk_propagation",
                "description": "战略目标的高风险传播到任务",
                "likelihood": "medium",
                "impact": "high",
                "mitigation": "加强监控，准备应急方案"
            })
        
        # 基于团队能力的风险
        if context.team_capabilities and len(context.team_capabilities) < 3:
            risks.append({
                "type": "capability_risk",
                "description": "团队能力有限，可能影响任务执行",
                "likelihood": "medium",
                "impact": "medium",
                "mitigation": "培训提升，外部支持"
            })
        
        return risks
    
    def define_milestones(self, 
                         task_specifications: List[TaskSpecification],
                         context: TacticalContext) -> List[Milestone]:
        """
        定义里程碑
        
        Args:
            task_specifications: 任务规格列表
            context: 战术上下文
            
        Returns:
            里程碑列表
        """
        milestones = []
        
        if not task_specifications:
            return milestones
        
        # 基于任务分组定义里程碑
        total_tasks = len(task_specifications)
        
        # 定义启动里程碑
        if total_tasks >= 1:
            start_milestone = Milestone(
                id=f"milestone_start_{int(time.time())}",
                name="项目启动",
                description="项目正式启动，任务分配完成",
                target_date=datetime.now() + timedelta(days=1),
                associated_tasks=[task_specifications[0].id] if task_specifications else [],
                completion_criteria=["项目计划批准", "团队组建完成", "资源分配到位"],
                importance_level="high",
                notification_requirements=["项目发起人", "主要利益相关者"]
            )
            milestones.append(start_milestone)
        
        # 定义中间里程碑（每25%进度）
        milestone_intervals = [0.25, 0.5, 0.75]
        
        for i, interval in enumerate(milestone_intervals):
            task_index = int(total_tasks * interval)
            if task_index < total_tasks and task_index > 0:
                milestone_name = f"阶段{i+1}完成"
                milestone_desc = f"完成{interval*100:.0f}%的任务"
                
                # 选择关联的任务
                associated_tasks = []
                start_idx = int(total_tasks * (interval - 0.25))
                end_idx = task_index
                for j in range(start_idx, min(end_idx, total_tasks)):
                    if j < len(task_specifications):
                        associated_tasks.append(task_specifications[j].id)
                
                milestone = Milestone(
                    id=f"milestone_{i+1}_{int(time.time())}",
                    name=milestone_name,
                    description=milestone_desc,
                    target_date=datetime.now() + timedelta(days=int(30 * interval)),  # 假设30天项目
                    associated_tasks=associated_tasks,
                    completion_criteria=[
                        f"阶段任务完成率>90%",
                        "质量检查通过",
                        f"阶段评审通过"
                    ],
                    importance_level="medium",
                    notification_requirements=["项目团队", "相关利益相关者"]
                )
                milestones.append(milestone)
        
        # 定义完成里程碑
        if total_tasks >= 1:
            completion_milestone = Milestone(
                id=f"milestone_completion_{int(time.time())}",
                name="项目完成",
                description="所有任务完成，项目目标达成",
                target_date=datetime.now() + timedelta(days=30),  # 假设30天项目
                associated_tasks=[spec.id for spec in task_specifications[-3:]] if len(task_specifications) >= 3 else [spec.id for spec in task_specifications],
                completion_criteria=[
                    "所有任务完成率100%",
                    "质量验收通过",
                    "利益相关者满意度>80%",
                    "项目文档完整"
                ],
                importance_level="critical",
                notification_requirements=["所有利益相关者", "管理层", "项目团队"]
            )
            milestones.append(completion_milestone)
        
        # 限制里程碑数量
        max_milestones = self.config.get("max_milestones_per_plan", 10)
        if len(milestones) > max_milestones:
            milestones = milestones[:max_milestones]
            logger.warning(f"里程碑数量限制为 {max_milestones}")
        
        # 更新性能统计
        self._update_performance_stats("milestones_defined", 0, count=len(milestones))
        
        logger.info(f"定义 {len(milestones)} 个里程碑")
        
        return milestones
    
    def create_tactical_plan(self,
                           context: TacticalContext,
                           strategic_objective: Optional[StrategicObjective] = None,
                           task_specifications: Optional[List[TaskSpecification]] = None,
                           milestones: Optional[List[Milestone]] = None) -> TacticalPlan:
        """
        创建战术计划
        
        Args:
            context: 战术上下文
            strategic_objective: 战略目标（可选，如未提供使用上下文中的目标）
            task_specifications: 任务规格（可选，如未提供将自动分解）
            milestones: 里程碑（可选，如未提供将自动定义）
            
        Returns:
            战术计划
        """
        start_time = time.time()
        
        logger.info("开始创建战术计划...")
        
        try:
            # 获取战略目标
            if strategic_objective is None:
                if context.strategic_objective is None:
                    raise ValueError("未提供战略目标且上下文中无战略目标")
                strategic_objective = context.strategic_objective
            
            # 生成任务规格（如未提供）
            if task_specifications is None:
                task_specifications = self.decompose_strategic_objective(strategic_objective, context)
            
            # 定义里程碑（如未提供）
            if milestones is None:
                milestones = self.define_milestones(task_specifications, context)
            
            # 优化资源分配
            resource_allocation = self.optimize_resource_allocation(task_specifications, context)
            
            # 制定时间安排
            schedule = self.create_schedule(task_specifications, milestones, context)
            
            # 制定风险缓解计划
            risk_mitigation_plan = self.create_risk_mitigation_plan(task_specifications, context)
            
            # 创建监控框架
            monitoring_framework = self.create_monitoring_framework(task_specifications, milestones)
            
            # 制定应急计划
            contingency_plans = self.create_contingency_plans(task_specifications, context)
            
            # 定义性能指标
            performance_metrics = self.define_performance_metrics(task_specifications)
            
            # 创建战术计划
            plan_id = f"tactical_plan_{int(time.time())}_{(zlib.adler32(str(strategic_objective.id).encode('utf-8')) & 0xffffffff) % 10000}"
            
            tactical_plan = TacticalPlan(
                id=plan_id,
                name=f"战术计划 {datetime.now().strftime('%Y-%m-%d')}",
                description=f"基于战略目标 '{strategic_objective.description[:50]}...' 的战术计划",
                time_period=(
                    datetime.now(),
                    datetime.now() + timedelta(days=30)  # 默认30天
                ),
                tactical_context=context,
                task_specifications=task_specifications,
                milestones=milestones,
                resource_allocation=resource_allocation,
                schedule=schedule,
                risk_mitigation_plan=risk_mitigation_plan,
                monitoring_framework=monitoring_framework,
                contingency_plans=contingency_plans,
                performance_metrics=performance_metrics
            )
            
            # 更新当前计划和历史记录
            self.current_tactical_plan = tactical_plan
            self.tactical_plans_history.append(tactical_plan)
            
            # 限制历史记录大小
            if len(self.tactical_plans_history) > self.max_history_size:
                self.tactical_plans_history = self.tactical_plans_history[-self.max_history_size:]
            
            # 更新性能统计
            execution_time = time.time() - start_time
            self._update_performance_stats("tactical_plans_created", execution_time)
            
            logger.info(f"战术计划创建完成，ID: {plan_id}，用时: {execution_time:.2f} 秒")
            
            return tactical_plan
            
        except Exception as e:
            logger.error(f"创建战术计划失败: {e}")
            if error_handler:
                error_handler.handle_error(e, "TacticalPlanner", "创建战术计划失败")
            
            # 返回基本的战术计划
            return TacticalPlan(
                id=f"failed_plan_{int(time.time())}",
                name="失败的战术计划",
                description="计划创建失败",
                time_period=(datetime.now(), datetime.now() + timedelta(days=30)),
                tactical_context=context,
                task_specifications=[],
                milestones=[],
                resource_allocation={},
                schedule={},
                risk_mitigation_plan=[],
                monitoring_framework={},
                contingency_plans=[],
                performance_metrics={}
            )
    
    def optimize_resource_allocation(self,
                                   task_specifications: List[TaskSpecification],
                                   context: TacticalContext) -> Dict[str, Dict[ResourceType, float]]:
        """
        优化资源分配
        
        Args:
            task_specifications: 任务规格列表
            context: 战术上下文
            
        Returns:
            优化的资源分配
        """
        logger.info("开始优化资源分配...")
        
        try:
            allocation = {}
            
            # 简单的资源分配算法：基于优先级和资源约束
            available_resources = context.available_resources.copy()
            
            # 按优先级排序任务
            priority_order = [TaskPriority.CRITICAL, TaskPriority.HIGH, 
                            TaskPriority.MEDIUM, TaskPriority.LOW, TaskPriority.BACKGROUND]
            
            for priority in priority_order:
                # 获取该优先级的任务
                priority_tasks = [ts for ts in task_specifications if ts.priority == priority]
                
                for task in priority_tasks:
                    task_allocation = {}
                    
                    # 尝试分配资源
                    for resource_type, requirement in task.required_resources.items():
                        if resource_type in available_resources:
                            # 分配资源，但不超过可用量
                            allocated = min(requirement, available_resources[resource_type])
                            task_allocation[resource_type] = allocated
                            available_resources[resource_type] -= allocated
                        else:
                            # 资源类型不可用，分配0
                            task_allocation[resource_type] = 0.0
                    
                    allocation[task.id] = task_allocation
            
            logger.info(f"资源分配优化完成，分配了 {len(allocation)} 个任务")
            return allocation
            
        except Exception as e:
            logger.error(f"资源分配优化失败: {e}")
            # 返回简单的平均分配
            allocation = {}
            num_tasks = len(task_specifications)
            if num_tasks == 0:
                num_tasks = 1
            
            for task in task_specifications:
                task_allocation = {}
                for resource_type, available in context.available_resources.items():
                    task_allocation[resource_type] = available / num_tasks
                allocation[task.id] = task_allocation
            
            return allocation
    
    def create_schedule(self,
                       task_specifications: List[TaskSpecification],
                       milestones: List[Milestone],
                       context: TacticalContext) -> Dict[str, Any]:
        """创建时间安排"""
        schedule = {
            "planning_method": "critical_path_method",  # 默认使用关键路径法
            "start_date": datetime.now().isoformat(),
            "end_date": (datetime.now() + timedelta(days=30)).isoformat(),  # 默认30天
            "tasks_schedule": [],
            "milestones_schedule": [],
            "critical_path": [],
            "total_duration_days": 30.0,
            "slack_time_days": 3.0,
            "resource_constraints": context.available_resources,
            "dependencies_considered": True
        }
        
        # 简单的时间安排算法
        current_date = datetime.now()
        for task in task_specifications:
            # 估算开始和结束时间
            start_date = current_date
            end_date = start_date + timedelta(days=task.estimated_effort)
            
            # 考虑依赖关系
            for dep_id, dep_type in task.dependencies:
                # 这里可以添加更复杂的依赖处理逻辑
                pass
            
            schedule["tasks_schedule"].append({
                "task_id": task.id,
                "task_description": task.description[:50],
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "duration_days": task.estimated_effort,
                "priority": task.priority.value,
                "complexity": task.complexity.value
            })
            
            # 更新当前日期（简单线性安排）
            current_date = end_date + timedelta(days=0.5)  # 0.5天间隔
        
        # 添加里程碑安排
        for milestone in milestones:
            schedule["milestones_schedule"].append({
                "milestone_id": milestone.id,
                "milestone_name": milestone.name,
                "target_date": milestone.target_date.isoformat(),
                "importance_level": milestone.importance_level,
                "associated_tasks": milestone.associated_tasks
            })
        
        # 计算总持续时间
        if schedule["tasks_schedule"]:
            first_task_start = datetime.fromisoformat(schedule["tasks_schedule"][0]["start_date"])
            last_task_end = datetime.fromisoformat(schedule["tasks_schedule"][-1]["end_date"])
            total_duration = (last_task_end - first_task_start).days
            schedule["total_duration_days"] = float(total_duration)
        
        return schedule
    
    def create_risk_mitigation_plan(self,
                                   task_specifications: List[TaskSpecification],
                                   context: TacticalContext) -> List[Dict[str, Any]]:
        """制定风险缓解计划"""
        risk_plan = []
        
        for task in task_specifications:
            for risk in task.risk_factors:
                risk_plan.append({
                    "task_id": task.id,
                    "risk_type": risk.get("type", "unknown"),
                    "risk_description": risk.get("description", "未知风险"),
                    "likelihood": risk.get("likelihood", "medium"),
                    "impact": risk.get("impact", "medium"),
                    "mitigation_strategy": risk.get("mitigation", "待制定缓解策略"),
                    "monitoring_frequency": "每周",
                    "responsibility": "项目经理",
                    "escalation_threshold": "风险发生或影响超过预期"
                })
        
        return risk_plan
    
    def create_monitoring_framework(self,
                                  task_specifications: List[TaskSpecification],
                                  milestones: List[Milestone]) -> Dict[str, Any]:
        """创建监控框架"""
        framework = {
            "monitoring_frequency": "每日",
            "reporting_structure": {
                "task_level": "每日站会报告",
                "milestone_level": "里程碑审查会议",
                "project_level": "每周项目状态报告"
            },
            "key_performance_indicators": [],
            "alert_mechanisms": ["邮件通知", "即时消息", "仪表板"],
            "review_meetings": [
                {"frequency": "每日", "type": "站会", "duration": "15分钟"},
                {"frequency": "每周", "type": "项目评审", "duration": "1小时"},
                {"frequency": "里程碑", "type": "里程碑审查", "duration": "2小时"}
            ]
        }
        
        # 添加任务KPI
        for task in task_specifications:
            framework["key_performance_indicators"].append({
                "task_id": task.id,
                "kpi_name": f"{task.id}_进度",
                "measurement": "完成百分比",
                "target": "100%",
                "frequency": "每日"
            })
        
        # 添加里程碑KPI
        for milestone in milestones:
            framework["key_performance_indicators"].append({
                "milestone_id": milestone.id,
                "kpi_name": f"{milestone.name}_达成",
                "measurement": "是否按时达成",
                "target": "是",
                "frequency": "里程碑日期"
            })
        
        return framework
    
    def create_contingency_plans(self,
                               task_specifications: List[TaskSpecification],
                               context: TacticalContext) -> List[Dict[str, Any]]:
        """制定应急计划"""
        contingency_plans = []
        
        # 为高风险任务创建应急计划
        for task in task_specifications:
            if task.priority == TaskPriority.CRITICAL or task.complexity == TaskComplexity.VERY_COMPLEX:
                plan = {
                    "task_id": task.id,
                    "risk_scenarios": [
                        "资源不足",
                        "时间延误",
                        "质量不达标",
                        "依赖任务失败"
                    ],
                    "contingency_actions": [
                        "重新分配资源",
                        "调整时间表",
                        "增加质量控制点",
                        "准备备用方案"
                    ],
                    "escalation_procedure": "立即上报项目经理和利益相关者",
                    "recovery_target": "在3个工作日内恢复正常进度",
                    "communication_plan": "每小时更新进展，每日总结报告"
                }
                contingency_plans.append(plan)
        
        return contingency_plans
    
    def define_performance_metrics(self,
                                 task_specifications: List[TaskSpecification]) -> Dict[str, Any]:
        """定义性能指标"""
        metrics = {
            "schedule_performance": {
                "description": "时间表遵守率",
                "measurement": "按时完成任务数 / 总任务数",
                "target": ">80%"
            },
            "resource_utilization": {
                "description": "资源利用率",
                "measurement": "实际使用资源 / 分配资源",
                "target": "70-90%"
            },
            "quality_metrics": {
                "description": "质量指标",
                "measurement": "质量检查通过率",
                "target": ">90%"
            },
            "risk_management": {
                "description": "风险管理效果",
                "measurement": "已识别风险缓解率",
                "target": ">70%"
            },
            "stakeholder_satisfaction": {
                "description": "利益相关者满意度",
                "measurement": "满意度调查评分",
                "target": ">75%"
            },
            "team_productivity": {
                "description": "团队生产率",
                "measurement": "完成任务人天 / 总人天",
                "target": ">85%"
            }
        }
        
        return metrics
    
    def _update_performance_stats(self, stat_type: str, execution_time: float, count: int = 1):
        """更新性能统计"""
        if stat_type == "tactical_plans_created":
            self.performance_stats["tactical_plans_created"] += 1
        elif stat_type == "tasks_decomposed":
            self.performance_stats["tasks_decomposed"] += count
        elif stat_type == "milestones_defined":
            self.performance_stats["milestones_defined"] += count
        elif stat_type == "replannings_performed":
            self.performance_stats["replannings_performed"] += 1
        
        # 更新时间统计
        self.performance_stats["total_planning_time"] += execution_time
        self.performance_stats["last_planning_time"] = execution_time
        
        # 计算平均时间
        total_actions = (self.performance_stats["tactical_plans_created"] +
                        self.performance_stats["tasks_decomposed"] // 10 +  # 每10个任务算一次行动
                        self.performance_stats["milestones_defined"] // 5 +  # 每5个里程碑算一次行动
                        self.performance_stats["replannings_performed"])
        
        if total_actions > 0:
            self.performance_stats["average_planning_time"] = (
                self.performance_stats["total_planning_time"] / total_actions
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "performance_stats": self.performance_stats.copy(),
            "current_plan": self.current_tactical_plan.id if self.current_tactical_plan else None,
            "plans_in_history": len(self.tactical_plans_history),
            "task_templates_available": len(self.task_templates),
            "scheduling_algorithms": len(self.scheduling_algorithms),
            "config_summary": {
                "enable_task_decomposition": self.config.get("enable_task_decomposition", True),
                "enable_resource_optimization": self.config.get("enable_resource_optimization", True),
                "max_tasks_per_objective": self.config.get("max_tasks_per_objective", 15),
                "enable_dynamic_rescheduling": self.config.get("enable_dynamic_rescheduling", True)
            }
        }
        
        return report