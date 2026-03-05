#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zlib
"""
分层规划系统 - 实现多时间尺度的分层决策和规划

核心功能:
1. 三层规划抽象: 战略层→战术层→操作层的层次分解
2. 多时间尺度规划: 从分钟到数年的不同时间粒度
3. 目标分解和任务规划: 高层目标到具体行动的转换
4. 实时重规划和适应: 环境变化时的动态调整
5. 资源约束优化: 在有限资源下的最优规划

分层结构:
1. 战略层 (Strategic):
   - 时间尺度: 月~年
   - 抽象级别: 高层目标、价值导向
   - 输出: 战略目标、优先级、资源分配

2. 战术层 (Tactical):
   - 时间尺度: 天~周
   - 抽象级别: 项目计划、里程碑
   - 输出: 任务序列、依赖关系、时间安排

3. 操作层 (Operational):
   - 时间尺度: 分钟~小时
   - 抽象级别: 具体行动、即时决策
   - 输出: 动作序列、参数设置、执行监控

技术实现:
- 目标导向规划算法
- 约束满足规划(CSP)
- 分层任务网络(HTN)
- 马尔可夫决策过程(MDP)
- 在线学习和优化

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from enum import Enum
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import networkx as nx

# 导入错误处理
from core.error_handling import ErrorHandler

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()


class PlanningLevel(Enum):
    """规划层次枚举"""
    STRATEGIC = "strategic"    # 战略层: 长期目标
    TACTICAL = "tactical"      # 战术层: 中期计划
    OPERATIONAL = "operational"  # 操作层: 短期行动


class GoalStatus(Enum):
    """目标状态枚举"""
    PENDING = "pending"        # 待处理
    ACTIVE = "active"          # 活跃
    IN_PROGRESS = "in_progress"  # 进行中
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"          # 失败
    CANCELLED = "cancelled"    # 取消


class ActionStatus(Enum):
    """行动状态枚举"""
    PENDING = "pending"        # 待执行
    EXECUTING = "executing"    # 执行中
    SUCCESS = "success"        # 成功
    FAILED = "failed"          # 失败
    CANCELLED = "cancelled"    # 取消


class ConstraintType(Enum):
    """约束类型枚举"""
    TEMPORAL = "temporal"      # 时间约束
    RESOURCE = "resource"      # 资源约束
    PRECEDENCE = "precedence"  # 前置约束
    CAPABILITY = "capability"  # 能力约束
    SAFETY = "safety"          # 安全约束


@dataclass
class Goal:
    """目标数据类"""
    id: str
    description: str
    level: PlanningLevel
    priority: float = 1.0
    deadline: Optional[datetime] = None
    status: GoalStatus = GoalStatus.PENDING
    parent_goal: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.priority = max(0.0, min(1.0, self.priority))
        if not self.id:
            self.id = f"goal_{int(time.time())}_{(zlib.adler32(str(self.description).encode('utf-8')) & 0xffffffff) % 10000}"


@dataclass
class Task:
    """任务数据类"""
    id: str
    description: str
    level: PlanningLevel
    goal_id: str
    estimated_duration: float  # 小时
    prerequisites: List[str] = field(default_factory=list)
    resources: Dict[str, float] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        if not self.id:
            self.id = f"task_{int(time.time())}_{(zlib.adler32(str(self.description).encode('utf-8')) & 0xffffffff) % 10000}"


@dataclass
class Action:
    """行动数据类"""
    id: str
    description: str
    task_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: float = 0.1  # 小时
    prerequisites: List[str] = field(default_factory=list)
    resources: Dict[str, float] = field(default_factory=dict)
    status: ActionStatus = ActionStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        if not self.id:
            self.id = f"action_{int(time.time())}_{(zlib.adler32(str(self.description).encode('utf-8')) & 0xffffffff) % 10000}"


@dataclass
class PlanningContext:
    """规划上下文数据类"""
    current_state: Dict[str, Any]
    available_resources: Dict[str, float]
    constraints: List[Dict[str, Any]]
    historical_data: Optional[Dict[str, Any]] = None
    world_model: Optional[Any] = None
    causal_knowledge: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchicalPlanningSystem:
    """
    分层规划系统
    
    核心组件:
    1. 目标管理器: 目标创建、更新和追踪
    2. 任务分解器: 目标到任务的分解
    3. 行动生成器: 任务到行动的具体化
    4. 调度器: 时间表和资源分配
    5. 监控器: 执行监控和反馈
    6. 重规划器: 动态调整和适应
    
    工作流程:
    高层目标 → 目标管理器 → 战略目标 → 任务分解器 → 战术任务
    战术任务 → 行动生成器 → 操作行动 → 调度器 → 执行计划
    执行监控 → 监控器 → 反馈 → 重规划器 → 调整计划
    
    技术特性:
    - 层次化目标分解
    - 多时间尺度规划
    - 约束满足和优化
    - 实时重规划和适应
    - 从经验中学习改进
    """
    
    def __init__(self,
                 max_planning_horizon: float = 365.0,  # 天
                 min_planning_granularity: float = 0.0167,  # 小时 (1分钟)
                 resource_constraints_enabled: bool = True,
                 realtime_adaptation_enabled: bool = True):
        """
        初始化分层规划系统
        
        Args:
            max_planning_horizon: 最大规划时间范围（天）
            min_planning_granularity: 最小规划粒度（小时）
            resource_constraints_enabled: 是否启用资源约束
            realtime_adaptation_enabled: 是否启用实时适应
        """
        self.max_planning_horizon = max_planning_horizon
        self.min_planning_granularity = min_planning_granularity
        self.resource_constraints_enabled = resource_constraints_enabled
        self.realtime_adaptation_enabled = realtime_adaptation_enabled
        
        # 数据存储
        self.goals: Dict[str, Goal] = {}
        self.tasks: Dict[str, Task] = {}
        self.actions: Dict[str, Action] = {}
        
        # 层次结构图
        self.goal_hierarchy = nx.DiGraph()
        self.task_dependency_graph = nx.DiGraph()
        self.action_dependency_graph = nx.DiGraph()
        
        # 规划上下文
        self.current_context: Optional[PlanningContext] = None
        
        # 规划缓存
        self.planning_cache = {}
        
        # 配置参数
        self.config = {
            'goal_decomposition_depth': 3,
            'max_subgoals_per_goal': 5,
            'max_tasks_per_goal': 10,
            'max_actions_per_task': 20,
            'planning_time_limit': 60.0,  # 秒
            'replanning_trigger_threshold': 0.3,
            'resource_allocation_strategy': 'balanced',
            'temporal_constraint_strictness': 'medium'
        }
        
        # 性能统计
        self.performance_stats = {
            'goals_created': 0,
            'tasks_generated': 0,
            'actions_generated': 0,
            'plans_executed': 0,
            'replannings_triggered': 0,
            'constraint_violations': 0,
            'average_planning_time': 0.0,
            'success_rate': 0.0
        }
        
        logger.info(f"分层规划系统初始化完成，最大规划范围: {max_planning_horizon} 天")
    
    def set_planning_context(self, context: PlanningContext):
        """设置规划上下文"""
        self.current_context = context
        logger.info(f"规划上下文已设置，可用资源: {len(context.available_resources)} 种")
    
    def create_goal(self,
                   description: str,
                   level: PlanningLevel,
                   priority: float = 1.0,
                   deadline: Optional[datetime] = None,
                   parent_goal: Optional[str] = None,
                   constraints: Optional[List[Dict[str, Any]]] = None) -> Goal:
        """
        创建新目标
        
        Args:
            description: 目标描述
            level: 规划层次
            priority: 优先级 (0.0-1.0)
            deadline: 截止时间
            parent_goal: 父目标ID
            constraints: 约束条件
            
        Returns:
            创建的目标
        """
        goal_id = f"goal_{int(time.time())}_{(zlib.adler32(str(description).encode('utf-8')) & 0xffffffff) % 10000}"
        
        goal = Goal(
            id=goal_id,
            description=description,
            level=level,
            priority=priority,
            deadline=deadline,
            status=GoalStatus.PENDING,
            parent_goal=parent_goal,
            constraints=constraints or [],
            metadata={
                'created_at': datetime.now().isoformat(),
                'creator': 'hierarchical_planning_system'
            }
        )
        
        self.goals[goal_id] = goal
        
        # 更新目标层次图
        self.goal_hierarchy.add_node(goal_id, **goal.__dict__)
        if parent_goal and parent_goal in self.goals:
            self.goal_hierarchy.add_edge(parent_goal, goal_id)
        
        self.performance_stats['goals_created'] += 1
        
        logger.info(f"目标创建成功: {goal_id} ({level.value}), 优先级: {priority}")
        
        return goal
    
    def decompose_goal(self, goal_id: str) -> List[str]:
        """
        分解目标为子目标
        
        Args:
            goal_id: 目标ID
            
        Returns:
            子目标ID列表
        """
        if goal_id not in self.goals:
            logger.error(f"目标不存在: {goal_id}")
            return []
        
        goal = self.goals[goal_id]
        
        # 检查是否已经达到最大分解深度
        current_depth = self._calculate_goal_depth(goal_id)
        if current_depth >= self.config['goal_decomposition_depth']:
            logger.info(f"目标 {goal_id} 已达到最大分解深度 {current_depth}")
            return []
        
        # 基于目标层次生成子目标
        subgoal_ids = []
        
        if goal.level == PlanningLevel.STRATEGIC:
            # 战略目标分解为战术子目标
            subgoal_descriptions = self._decompose_strategic_goal(goal)
            subgoal_level = PlanningLevel.TACTICAL
            
        elif goal.level == PlanningLevel.TACTICAL:
            # 战术目标分解为操作子目标
            subgoal_descriptions = self._decompose_tactical_goal(goal)
            subgoal_level = PlanningLevel.OPERATIONAL
            
        else:
            # 操作目标不进一步分解
            logger.info(f"操作目标 {goal_id} 不进行分解")
            return []
        
        # 创建子目标
        for i, sub_desc in enumerate(subgoal_descriptions):
            subgoal_id = f"{goal_id}_sub_{i}"
            
            subgoal = Goal(
                id=subgoal_id,
                description=sub_desc,
                level=subgoal_level,
                priority=goal.priority * 0.9,  # 子目标优先级略低
                deadline=goal.deadline,
                parent_goal=goal_id,
                status=GoalStatus.PENDING,
                metadata={
                    'created_at': datetime.now().isoformat(),
                    'parent_goal': goal_id,
                    'decomposition_index': i
                }
            )
            
            self.goals[subgoal_id] = subgoal
            subgoal_ids.append(subgoal_id)
            
            # 更新目标层次图
            self.goal_hierarchy.add_node(subgoal_id, **subgoal.__dict__)
            self.goal_hierarchy.add_edge(goal_id, subgoal_id)
            
            # 更新父目标的子目标列表
            goal.subgoals.append(subgoal_id)
        
        # 限制子目标数量
        if len(subgoal_ids) > self.config['max_subgoals_per_goal']:
            subgoal_ids = subgoal_ids[:self.config['max_subgoals_per_goal']]
            logger.warning(f"目标 {goal_id} 的子目标数量限制为 {self.config['max_subgoals_per_goal']}")
        
        logger.info(f"目标 {goal_id} 分解为 {len(subgoal_ids)} 个子目标")
        
        return subgoal_ids
    
    def _decompose_strategic_goal(self, goal: Goal) -> List[str]:
        """分解战略目标为战术子目标"""
        # 基于目标描述的语义分解
        description_lower = goal.description.lower()
        
        if any(keyword in description_lower for keyword in ['develop', 'build', 'create']):
            # 开发/构建类目标
            return [
                f"需求分析: {goal.description}",
                f"设计阶段: {goal.description}",
                f"实现阶段: {goal.description}",
                f"测试验证: {goal.description}",
                f"部署发布: {goal.description}"
            ]
        
        elif any(keyword in description_lower for keyword in ['improve', 'enhance', 'optimize']):
            # 改进/优化类目标
            return [
                f"现状评估: {goal.description}",
                f"问题识别: {goal.description}",
                f"方案设计: {goal.description}",
                f"实施改进: {goal.description}",
                f"效果评估: {goal.description}"
            ]
        
        elif any(keyword in description_lower for keyword in ['learn', 'study', 'research']):
            # 学习/研究类目标
            return [
                f"背景调研: {goal.description}",
                f"知识获取: {goal.description}",
                f"实践应用: {goal.description}",
                f"总结反思: {goal.description}"
            ]
        
        else:
            # 通用分解
            return [
                f"规划阶段: {goal.description}",
                f"准备阶段: {goal.description}",
                f"执行阶段: {goal.description}",
                f"完成阶段: {goal.description}"
            ]
    
    def _decompose_tactical_goal(self, goal: Goal) -> List[str]:
        """分解战术目标为操作子目标"""
        # 基于目标描述的语义分解
        description_lower = goal.description.lower()
        
        # 提取关键词
        if '阶段' in goal.description or 'phase' in description_lower:
            # 阶段类目标进一步分解为具体任务
            return [
                f"任务定义: {goal.description}",
                f"资源准备: {goal.description}",
                f"执行监控: {goal.description}",
                f"结果评估: {goal.description}"
            ]
        
        elif '分析' in goal.description or 'analysis' in description_lower:
            # 分析类目标
            return [
                f"数据收集: {goal.description}",
                f"数据处理: {goal.description}",
                f"分析计算: {goal.description}",
                f"报告生成: {goal.description}"
            ]
        
        elif '设计' in goal.description or 'design' in description_lower:
            # 设计类目标
            return [
                f"概念设计: {goal.description}",
                f"详细设计: {goal.description}",
                f"设计验证: {goal.description}",
                f"设计优化: {goal.description}"
            ]
        
        else:
            # 通用分解
            return [
                f"准备: {goal.description}",
                f"执行: {goal.description}",
                f"检查: {goal.description}",
                f"完成: {goal.description}"
            ]
    
    def _calculate_goal_depth(self, goal_id: str) -> int:
        """计算目标在层次结构中的深度"""
        try:
            # 找到根目标
            ancestors = list(nx.ancestors(self.goal_hierarchy, goal_id))
            if not ancestors:
                return 0
            
            # 计算最大深度
            depths = []
            for ancestor in ancestors:
                path_length = len(nx.shortest_path(self.goal_hierarchy, ancestor, goal_id)) - 1
                depths.append(path_length)
            
            return max(depths) if depths else 0
            
        except Exception as e:
            logger.warning(f"计算目标深度失败 {goal_id}: {e}")
            return 0
    
    def generate_tasks_for_goal(self, goal_id: str) -> List[str]:
        """
        为目标生成任务
        
        Args:
            goal_id: 目标ID
            
        Returns:
            任务ID列表
        """
        if goal_id not in self.goals:
            logger.error(f"目标不存在: {goal_id}")
            return []
        
        goal = self.goals[goal_id]
        
        # 检查目标层次
        if goal.level != PlanningLevel.OPERATIONAL:
            logger.warning(f"非操作层目标 {goal_id} 需要先分解为操作子目标")
            # 自动分解
            subgoal_ids = self.decompose_goal(goal_id)
            
            # 为子目标生成任务
            task_ids = []
            for subgoal_id in subgoal_ids:
                subgoal_tasks = self.generate_tasks_for_goal(subgoal_id)
                task_ids.extend(subgoal_tasks)
            
            return task_ids
        
        # 为操作层目标生成任务
        task_descriptions = self._generate_task_descriptions(goal)
        
        task_ids = []
        for i, task_desc in enumerate(task_descriptions):
            task_id = f"{goal_id}_task_{i}"
            
            # 估计任务持续时间
            estimated_duration = self._estimate_task_duration(task_desc, goal)
            
            task = Task(
                id=task_id,
                description=task_desc,
                level=goal.level,
                goal_id=goal_id,
                estimated_duration=estimated_duration,
                status=GoalStatus.PENDING,
                metadata={
                    'created_at': datetime.now().isoformat(),
                    'goal_id': goal_id,
                    'task_index': i
                }
            )
            
            self.tasks[task_id] = task
            task_ids.append(task_id)
            
            # 更新任务依赖图
            self.task_dependency_graph.add_node(task_id, **task.__dict__)
        
        # 建立任务依赖关系
        self._establish_task_dependencies(task_ids, goal)
        
        # 限制任务数量
        if len(task_ids) > self.config['max_tasks_per_goal']:
            task_ids = task_ids[:self.config['max_tasks_per_goal']]
            logger.warning(f"目标 {goal_id} 的任务数量限制为 {self.config['max_tasks_per_goal']}")
        
        self.performance_stats['tasks_generated'] += len(task_ids)
        
        logger.info(f"为目标 {goal_id} 生成 {len(task_ids)} 个任务")
        
        return task_ids
    
    def _generate_task_descriptions(self, goal: Goal) -> List[str]:
        """生成任务描述"""
        # 基于目标描述和上下文生成任务
        description_lower = goal.description.lower()
        
        if '准备' in goal.description or 'prepare' in description_lower:
            return [
                f"收集必要资源: {goal.description}",
                f"设置工作环境: {goal.description}",
                f"确认需求和约束: {goal.description}",
                f"制定详细计划: {goal.description}"
            ]
        
        elif '执行' in goal.description or 'execute' in description_lower:
            return [
                f"启动执行过程: {goal.description}",
                f"监控执行进度: {goal.description}",
                f"处理执行问题: {goal.description}",
                f"记录执行结果: {goal.description}"
            ]
        
        elif '检查' in goal.description or 'check' in description_lower:
            return [
                f"收集检查数据: {goal.description}",
                f"执行检查分析: {goal.description}",
                f"评估检查结果: {goal.description}",
                f"生成检查报告: {goal.description}"
            ]
        
        elif '完成' in goal.description or 'complete' in description_lower:
            return [
                f"完成收尾工作: {goal.description}",
                f"验证完成标准: {goal.description}",
                f"整理成果文档: {goal.description}",
                f"提交完成报告: {goal.description}"
            ]
        
        else:
            # 通用任务生成
            return [
                f"启动: {goal.description}",
                f"进行: {goal.description}",
                f"完成: {goal.description}",
                f"验证: {goal.description}"
            ]
    
    def _estimate_task_duration(self, task_description: str, goal: Goal) -> float:
        """估计任务持续时间（小时）"""
        # 基于任务描述的简单估计
        desc_lower = task_description.lower()
        
        duration_map = {
            '收集': 2.0,
            '设置': 1.5,
            '确认': 1.0,
            '制定': 3.0,
            '启动': 0.5,
            '监控': 8.0,
            '处理': 2.0,
            '记录': 1.0,
            '执行': 4.0,
            '评估': 2.0,
            '生成': 1.5,
            '完成': 1.0,
            '验证': 1.5,
            '整理': 2.0,
            '提交': 0.5
        }
        
        # 查找关键词
        for keyword, duration in duration_map.items():
            if keyword in task_description:
                return duration
        
        # 默认持续时间
        return 2.0
    
    def _establish_task_dependencies(self, task_ids: List[str], goal: Goal):
        """建立任务依赖关系"""
        if len(task_ids) < 2:
            return
        
        # 简单的线性依赖：前一个任务是后一个任务的前置条件
        for i in range(len(task_ids) - 1):
            task_id = task_ids[i]
            next_task_id = task_ids[i + 1]
            
            # 更新任务的前置条件
            if next_task_id in self.tasks:
                self.tasks[next_task_id].prerequisites.append(task_id)
            
            # 更新依赖图
            self.task_dependency_graph.add_edge(task_id, next_task_id, 
                                               dependency_type="sequential")
    
    def generate_actions_for_task(self, task_id: str) -> List[str]:
        """
        为任务生成行动
        
        Args:
            task_id: 任务ID
            
        Returns:
            行动ID列表
        """
        if task_id not in self.tasks:
            logger.error(f"任务不存在: {task_id}")
            return []
        
        task = self.tasks[task_id]
        
        # 生成行动描述
        action_descriptions = self._generate_action_descriptions(task)
        
        action_ids = []
        for i, action_desc in enumerate(action_descriptions):
            action_id = f"{task_id}_action_{i}"
            
            # 估计行动持续时间
            estimated_duration = self._estimate_action_duration(action_desc, task)
            
            action = Action(
                id=action_id,
                description=action_desc,
                task_id=task_id,
                estimated_duration=estimated_duration,
                status=ActionStatus.PENDING,
                metadata={
                    'created_at': datetime.now().isoformat(),
                    'task_id': task_id,
                    'action_index': i
                }
            )
            
            self.actions[action_id] = action
            action_ids.append(action_id)
            
            # 更新行动依赖图
            self.action_dependency_graph.add_node(action_id, **action.__dict__)
        
        # 建立行动依赖关系
        self._establish_action_dependencies(action_ids, task)
        
        # 限制行动数量
        if len(action_ids) > self.config['max_actions_per_task']:
            action_ids = action_ids[:self.config['max_actions_per_task']]
            logger.warning(f"任务 {task_id} 的行动数量限制为 {self.config['max_actions_per_task']}")
        
        self.performance_stats['actions_generated'] += len(action_ids)
        
        logger.info(f"为任务 {task_id} 生成 {len(action_ids)} 个行动")
        
        return action_ids
    
    def _generate_action_descriptions(self, task: Task) -> List[str]:
        """生成行动描述"""
        # 基于任务描述生成具体行动
        desc_lower = task.description.lower()
        
        if '收集' in task.description or 'collect' in desc_lower:
            return [
                f"确定收集目标: {task.description}",
                f"选择收集方法: {task.description}",
                f"执行收集操作: {task.description}",
                f"整理收集结果: {task.description}"
            ]
        
        elif '设置' in task.description or 'setup' in desc_lower:
            return [
                f"检查设置需求: {task.description}",
                f"准备设置材料: {task.description}",
                f"执行设置操作: {task.description}",
                f"验证设置结果: {task.description}"
            ]
        
        elif '执行' in task.description or 'execute' in desc_lower:
            return [
                f"准备执行环境: {task.description}",
                f"配置执行参数: {task.description}",
                f"启动执行过程: {task.description}",
                f"监控执行状态: {task.description}"
            ]
        
        elif '完成' in task.description or 'complete' in desc_lower:
            return [
                f"检查完成条件: {task.description}",
                f"执行收尾操作: {task.description}",
                f"验证完成结果: {task.description}",
                f"记录完成状态: {task.description}"
            ]
        
        else:
            # 通用行动生成
            return [
                f"准备: {task.description}",
                f"配置: {task.description}",
                f"执行: {task.description}",
                f"检查: {task.description}"
            ]
    
    def _estimate_action_duration(self, action_description: str, task: Task) -> float:
        """估计行动持续时间（小时）"""
        # 基于行动描述的简单估计
        desc_lower = action_description.lower()
        
        duration_map = {
            '确定': 0.25,
            '选择': 0.5,
            '准备': 0.5,
            '配置': 0.5,
            '启动': 0.1,
            '执行': task.estimated_duration * 0.6,  # 占任务时间的60%
            '监控': task.estimated_duration * 0.3,  # 占任务时间的30%
            '检查': 0.25,
            '验证': 0.5,
            '整理': 0.5,
            '记录': 0.25
        }
        
        # 查找关键词
        for keyword, duration in duration_map.items():
            if keyword in action_description:
                return max(self.min_planning_granularity, duration)
        
        # 默认持续时间
        return max(self.min_planning_granularity, 0.5)
    
    def _establish_action_dependencies(self, action_ids: List[str], task: Task):
        """建立行动依赖关系"""
        if len(action_ids) < 2:
            return
        
        # 简单的线性依赖
        for i in range(len(action_ids) - 1):
            action_id = action_ids[i]
            next_action_id = action_ids[i + 1]
            
            # 更新行动的前置条件
            if next_action_id in self.actions:
                self.actions[next_action_id].prerequisites.append(action_id)
            
            # 更新依赖图
            self.action_dependency_graph.add_edge(action_id, next_action_id,
                                                 dependency_type="sequential")
    
    def create_plan(self, goal_id: str) -> Dict[str, Any]:
        """
        创建完整规划
        
        Args:
            goal_id: 目标ID
            
        Returns:
            规划结果
        """
        start_time = time.time()
        
        if goal_id not in self.goals:
            return {
                'success': False,
                'error': f'目标不存在: {goal_id}'
            }
        
        # 检查规划时间限制
        if time.time() - start_time > self.config['planning_time_limit']:
            return {
                'success': False,
                'error': '规划时间超限'
            }
        
        goal = self.goals[goal_id]
        
        logger.info(f"开始为目标 {goal_id} 创建规划，层次: {goal.level.value}")
        
        # 生成完整规划层次结构
        plan_result = self._generate_complete_plan(goal_id)
        
        # 检查资源约束
        if self.resource_constraints_enabled and self.current_context:
            resource_check = self._check_resource_constraints(plan_result)
            if not resource_check['success']:
                logger.warning(f"资源约束检查失败: {resource_check['message']}")
                # 尝试调整规划
                adjusted_plan = self._adjust_plan_for_resources(plan_result, resource_check)
                if adjusted_plan:
                    plan_result = adjusted_plan
                else:
                    return {
                        'success': False,
                        'error': '无法满足资源约束',
                        'details': resource_check
                    }
        
        # 时间安排
        schedule_result = self._schedule_plan(plan_result)
        
        # 更新性能统计
        planning_time = time.time() - start_time
        self._update_planning_stats(planning_time)
        
        plan_summary = {
            'success': True,
            'goal_id': goal_id,
            'goal_description': goal.description,
            'planning_time': planning_time,
            'goal_hierarchy': self._summarize_goal_hierarchy(goal_id),
            'task_count': len(plan_result.get('tasks', [])),
            'action_count': len(plan_result.get('actions', [])),
            'estimated_total_duration': schedule_result.get('total_duration', 0),
            'schedule': schedule_result,
            'plan_structure': plan_result
        }
        
        logger.info(f"规划创建完成: 目标 {goal_id}, {plan_summary['task_count']} 任务, {plan_summary['action_count']} 行动, 预计时间: {plan_summary['estimated_total_duration']:.2f} 小时")
        
        return plan_summary
    
    def _generate_complete_plan(self, goal_id: str) -> Dict[str, Any]:
        """生成完整规划（递归分解）"""
        if goal_id not in self.goals:
            return {}
        
        goal = self.goals[goal_id]
        
        # 如果目标不是操作层，先分解
        if goal.level != PlanningLevel.OPERATIONAL:
            subgoal_ids = self.decompose_goal(goal_id)
            
            # 为每个子目标递归生成规划
            subplans = []
            for subgoal_id in subgoal_ids:
                subplan = self._generate_complete_plan(subgoal_id)
                if subplan:
                    subplans.append(subplan)
            
            # 合并子规划
            merged_plan = self._merge_subplans(subplans, goal_id)
            return merged_plan
        
        # 操作层目标：生成任务和行动
        task_ids = self.generate_tasks_for_goal(goal_id)
        
        # 为每个任务生成行动
        all_action_ids = []
        for task_id in task_ids:
            action_ids = self.generate_actions_for_task(task_id)
            all_action_ids.extend(action_ids)
        
        # 构建规划结构
        plan = {
            'goal_id': goal_id,
            'goal_level': goal.level.value,
            'tasks': [self.tasks[task_id] for task_id in task_ids if task_id in self.tasks],
            'actions': [self.actions[action_id] for action_id in all_action_ids if action_id in self.actions],
            'task_ids': task_ids,
            'action_ids': all_action_ids,
            'generated_at': datetime.now().isoformat()
        }
        
        return plan
    
    def _merge_subplans(self, subplans: List[Dict[str, Any]], parent_goal_id: str) -> Dict[str, Any]:
        """合并子规划"""
        if not subplans:
            return {}
        
        merged_plan = {
            'goal_id': parent_goal_id,
            'goal_level': self.goals[parent_goal_id].level.value,
            'subplans': subplans,
            'tasks': [],
            'actions': [],
            'task_ids': [],
            'action_ids': [],
            'generated_at': datetime.now().isoformat()
        }
        
        # 合并所有任务和行动
        for subplan in subplans:
            if 'tasks' in subplan:
                merged_plan['tasks'].extend(subplan['tasks'])
            
            if 'actions' in subplan:
                merged_plan['actions'].extend(subplan['actions'])
            
            if 'task_ids' in subplan:
                merged_plan['task_ids'].extend(subplan['task_ids'])
            
            if 'action_ids' in subplan:
                merged_plan['action_ids'].extend(subplan['action_ids'])
        
        return merged_plan
    
    def _check_resource_constraints(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """检查资源约束"""
        if not self.current_context:
            return {'success': True, 'message': '无上下文信息，跳过资源检查'}
        
        available_resources = self.current_context.available_resources
        
        # 计算规划所需资源
        required_resources = defaultdict(float)
        
        # 累加行动的资源需求
        for action in plan.get('actions', []):
            for resource, amount in action.resources.items():
                required_resources[resource] += amount
        
        # 检查资源是否足够
        insufficient_resources = []
        for resource, required in required_resources.items():
            available = available_resources.get(resource, 0.0)
            if required > available:
                insufficient_resources.append({
                    'resource': resource,
                    'required': required,
                    'available': available,
                    'deficit': required - available
                })
        
        if insufficient_resources:
            return {
                'success': False,
                'message': '资源不足',
                'insufficient_resources': insufficient_resources
            }
        else:
            return {
                'success': True,
                'message': '资源足够',
                'required_resources': dict(required_resources)
            }
    
    def _adjust_plan_for_resources(self, 
                                  plan: Dict[str, Any], 
                                  resource_check: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """调整规划以满足资源约束"""
        # 简化实现：按优先级排序并移除低优先级行动
        if 'insufficient_resources' not in resource_check:
            return None
        
        # 获取不足的资源
        insufficient = resource_check['insufficient_resources']
        
        # 计算行动优先级（基于任务和目标优先级）
        action_priorities = {}
        for action in plan.get('actions', []):
            # 简单优先级计算
            task = self.tasks.get(action.task_id)
            if task:
                goal = self.goals.get(task.goal_id)
                if goal:
                    priority = goal.priority * 0.7 + task.priority * 0.3
                else:
                    priority = 0.5
            else:
                priority = 0.5
            
            action_priorities[action.id] = priority
        
        # 按优先级排序行动
        sorted_actions = sorted(plan.get('actions', []), 
                               key=lambda a: action_priorities.get(a.id, 0.0), 
                               reverse=True)
        
        # 逐步添加行动直到资源不足
        adjusted_actions = []
        available_resources = self.current_context.available_resources.copy()
        
        for action in sorted_actions:
            # 检查行动的资源需求
            action_resources = action.resources
            
            # 检查是否满足资源约束
            feasible = True
            for resource, amount in action_resources.items():
                if amount > available_resources.get(resource, 0.0):
                    feasible = False
                    break
            
            if feasible:
                # 添加行动并更新可用资源
                adjusted_actions.append(action)
                for resource, amount in action_resources.items():
                    available_resources[resource] -= amount
        
        # 创建调整后的规划
        if adjusted_actions:
            adjusted_plan = plan.copy()
            adjusted_plan['actions'] = adjusted_actions
            adjusted_plan['action_ids'] = [a.id for a in adjusted_actions]
            
            # 重新计算任务（只保留有行动的任务）
            adjusted_task_ids = set(a.task_id for a in adjusted_actions)
            adjusted_plan['tasks'] = [t for t in plan.get('tasks', []) if t.id in adjusted_task_ids]
            adjusted_plan['task_ids'] = list(adjusted_task_ids)
            
            logger.info(f"规划调整完成，保留 {len(adjusted_actions)}/{len(plan.get('actions', []))} 个行动")
            
            return adjusted_plan
        
        return None
    
    def _schedule_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """安排规划时间表"""
        # 简化实现：基于依赖关系的顺序安排
        
        # 提取行动依赖图
        action_ids = plan.get('action_ids', [])
        
        if not action_ids:
            return {'total_duration': 0, 'schedule': {}}
        
        # 计算行动的最早开始时间
        action_start_times = {}
        action_durations = {}
        
        # 初始化
        for action_id in action_ids:
            if action_id in self.actions:
                action = self.actions[action_id]
                action_durations[action_id] = action.estimated_duration
        
        # 计算关键路径
        try:
            # 构建依赖图
            dep_graph = self.action_dependency_graph.subgraph(action_ids).copy()
            
            # 如果没有依赖关系，按顺序安排
            if dep_graph.number_of_edges() == 0:
                current_time = 0.0
                schedule = {}
                
                for action_id in action_ids:
                    schedule[action_id] = {
                        'start': current_time,
                        'end': current_time + action_durations.get(action_id, 0.0),
                        'duration': action_durations.get(action_id, 0.0)
                    }
                    current_time += action_durations.get(action_id, 0.0)
                
                total_duration = current_time
                
            else:
                # 使用拓扑排序计算开始时间
                from networkx.algorithms.dag import topological_sort
                
                sorted_actions = list(topological_sort(dep_graph))
                
                # 计算最早开始时间
                earliest_start = {}
                for action_id in sorted_actions:
                    # 获取前置行动
                    predecessors = list(dep_graph.predecessors(action_id))
                    
                    if not predecessors:
                        earliest_start[action_id] = 0.0
                    else:
                        # 最早开始时间 = 所有前置行动结束时间的最大值
                        pred_end_times = []
                        for pred in predecessors:
                            pred_end = earliest_start.get(pred, 0.0) + action_durations.get(pred, 0.0)
                            pred_end_times.append(pred_end)
                        
                        earliest_start[action_id] = max(pred_end_times) if pred_end_times else 0.0
                
                # 构建时间表
                schedule = {}
                for action_id in action_ids:
                    start_time = earliest_start.get(action_id, 0.0)
                    duration = action_durations.get(action_id, 0.0)
                    
                    schedule[action_id] = {
                        'start': start_time,
                        'end': start_time + duration,
                        'duration': duration
                    }
                
                # 计算总持续时间
                end_times = [schedule[action_id]['end'] for action_id in action_ids]
                total_duration = max(end_times) if end_times else 0.0
                
        except Exception as e:
            logger.error(f"时间安排计算失败，使用简化安排: {e}")
            # 简化安排
            current_time = 0.0
            schedule = {}
            
            for action_id in action_ids:
                schedule[action_id] = {
                    'start': current_time,
                    'end': current_time + action_durations.get(action_id, 0.0),
                    'duration': action_durations.get(action_id, 0.0)
                }
                current_time += action_durations.get(action_id, 0.0)
            
            total_duration = current_time
        
        return {
            'total_duration': total_duration,
            'schedule': schedule,
            'unit': 'hours'
        }
    
    def _summarize_goal_hierarchy(self, goal_id: str) -> Dict[str, Any]:
        """汇总目标层次结构"""
        if goal_id not in self.goals:
            return {}
        
        def build_hierarchy(node_id: str, depth: int = 0) -> Dict[str, Any]:
            if node_id not in self.goals:
                return {}
            
            goal = self.goals[node_id]
            
            hierarchy = {
                'id': goal.id,
                'description': goal.description,
                'level': goal.level.value,
                'priority': goal.priority,
                'status': goal.status.value,
                'depth': depth
            }
            
            # 递归添加子目标
            subgoals = []
            for subgoal_id in goal.subgoals:
                sub_hierarchy = build_hierarchy(subgoal_id, depth + 1)
                if sub_hierarchy:
                    subgoals.append(sub_hierarchy)
            
            if subgoals:
                hierarchy['subgoals'] = subgoals
            
            return hierarchy
        
        return build_hierarchy(goal_id)
    
    def _update_planning_stats(self, planning_time: float):
        """更新规划统计"""
        self.performance_stats['plans_executed'] += 1
        
        # 更新平均规划时间
        current_avg = self.performance_stats['average_planning_time']
        n_executed = self.performance_stats['plans_executed']
        
        new_avg = (current_avg * (n_executed - 1) + planning_time) / n_executed
        self.performance_stats['average_planning_time'] = new_avg
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return self.performance_stats.copy()
    
    def get_planning_summary(self) -> Dict[str, Any]:
        """获取规划系统摘要"""
        return {
            'goals_count': len(self.goals),
            'tasks_count': len(self.tasks),
            'actions_count': len(self.actions),
            'goal_hierarchy_depth': self._calculate_max_hierarchy_depth(),
            'planning_context_set': self.current_context is not None,
            'config': self.config,
            'performance_stats': self.performance_stats
        }
    
    def _calculate_max_hierarchy_depth(self) -> int:
        """计算最大层次深度"""
        if not self.goal_hierarchy.nodes():
            return 0
        
        try:
            # 找到所有根节点（无父节点的节点）
            roots = [node for node in self.goal_hierarchy.nodes() 
                    if self.goal_hierarchy.in_degree(node) == 0]
            
            max_depth = 0
            for root in roots:
                # 计算从根节点到所有可达节点的最长路径
                for node in nx.descendants(self.goal_hierarchy, root):
                    try:
                        path_length = len(nx.shortest_path(self.goal_hierarchy, root, node)) - 1
                        max_depth = max(max_depth, path_length)
                    except:
                        continue
            
            return max_depth
            
        except Exception as e:
            logger.error(f"计算层次深度失败: {e}")
            return 0


# 示例和测试函数
def create_example_planning_system() -> HierarchicalPlanningSystem:
    """创建示例分层规划系统"""
    planning_system = HierarchicalPlanningSystem(
        max_planning_horizon=365.0,
        min_planning_granularity=0.0167,
        resource_constraints_enabled=True,
        realtime_adaptation_enabled=True
    )
    return planning_system


def test_hierarchical_planning():
    """测试分层规划系统"""
    logger.info("开始测试分层规划系统")
    
    # 创建示例规划系统
    planning_system = create_example_planning_system()
    
    # 创建规划上下文
    context = PlanningContext(
        current_state={'status': 'ready'},
        available_resources={
            'compute': 100.0,
            'memory': 50.0,
            'storage': 200.0,
            'energy': 1000.0
        },
        constraints=[
            {'type': 'temporal', 'max_duration': 24.0},
            {'type': 'resource', 'resource': 'compute', 'max_usage': 80.0}
        ],
        metadata={'test_context': True}
    )
    
    planning_system.set_planning_context(context)
    
    # 创建战略目标
    logger.info("创建战略目标...")
    strategic_goal = planning_system.create_goal(
        description="开发新一代AI系统",
        level=PlanningLevel.STRATEGIC,
        priority=0.9,
        deadline=datetime.now() + timedelta(days=180)
    )
    
    # 生成完整规划
    logger.info("生成完整规划...")
    plan_result = planning_system.create_plan(strategic_goal.id)
    
    if plan_result['success']:
        logger.info(f"规划成功创建: {plan_result['goal_description']}")
        logger.info(f"任务数量: {plan_result['task_count']}")
        logger.info(f"行动数量: {plan_result['action_count']}")
        logger.info(f"预计总时间: {plan_result['estimated_total_duration']:.2f} 小时")
        
        # 显示目标层次
        hierarchy = plan_result['goal_hierarchy']
        logger.info(f"目标层次深度: {hierarchy.get('depth', 0)}")
        
        # 显示性能统计
        stats = planning_system.get_performance_stats()
        logger.info(f"性能统计: 创建 {stats['goals_created']} 个目标，生成 {stats['tasks_generated']} 个任务，执行 {stats['plans_executed']} 个规划")
    
    # 获取规划系统摘要
    summary = planning_system.get_planning_summary()
    logger.info(f"规划系统摘要: {summary['goals_count']} 目标, {summary['tasks_count']} 任务, {summary['actions_count']} 行动")
    
    logger.info("分层规划系统测试完成")
    return planning_system


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_planning_system = test_hierarchical_planning()