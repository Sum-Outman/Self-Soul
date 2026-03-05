#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技能记忆存储 - 实现程序记忆的技能和动作序列存储

核心功能:
1. 技能表示和存储: 存储技能的定义和参数
2. 动作序列管理: 管理技能中的动作序列
3. 技能执行记录: 记录技能的执行历史
4. 技能改进优化: 基于执行反馈优化技能
5. 技能迁移学习: 在不同情境间迁移技能

技能记忆特点:
- 程序性: 技能是程序性知识
- 序列性: 技能由动作序列组成
- 参数化: 技能可以有参数
- 条件性: 技能执行有条件限制
- 可改进: 技能可以通过练习改进

技能类型:
1. 动作技能 (motor): 身体动作技能
2. 认知技能 (cognitive): 认知处理技能
3. 感知技能 (perceptual): 感知处理技能
4. 社交技能 (social): 社交交互技能
5. 复合技能 (composite): 多种技能的组合

技能改进机制:
1. 练习效应: 通过重复练习提高技能
2. 反馈学习: 基于执行结果调整技能
3. 泛化能力: 将技能应用于新情境
4. 自动化: 技能执行变得自动化
5. 遗忘和衰退: 长期不使用技能会衰退

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
import math
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


class SkillType(Enum):
    """技能类型枚举"""
    MOTOR = "motor"          # 动作技能
    COGNITIVE = "cognitive"  # 认知技能
    PERCEPTUAL = "perceptual"  # 感知技能
    SOCIAL = "social"        # 社交技能
    COMPOSITE = "composite"  # 复合技能


class SkillProficiencyLevel(Enum):
    """技能熟练度级别枚举"""
    NOVICE = "novice"        # 新手: 需要指导，错误率高
    BEGINNER = "beginner"    # 初学者: 基本掌握，仍有错误
    COMPETENT = "competent"  # 胜任: 能独立完成任务
    PROFICIENT = "proficient"  # 熟练: 高效完成任务
    EXPERT = "expert"        # 专家: 自动化，能处理异常


class SkillExecutionStatus(Enum):
    """技能执行状态枚举"""
    NOT_STARTED = "not_started"  # 未开始
    IN_PROGRESS = "in_progress"  # 进行中
    COMPLETED = "completed"      # 完成
    FAILED = "failed"            # 失败
    ABORTED = "aborted"          # 中止


@dataclass
class Action:
    """动作数据类"""
    id: str
    name: str
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0  # 预期持续时间（秒）
    prerequisites: List[str] = field(default_factory=list)  # 前提条件
    effects: List[str] = field(default_factory=list)        # 效果
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.duration = max(0.0, self.duration)


@dataclass
class Skill:
    """技能数据类"""
    id: str
    name: str
    skill_type: SkillType
    description: str = ""
    action_sequence: List[Action] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)  # 执行前提条件
    postconditions: List[str] = field(default_factory=list)  # 执行后条件
    proficiency_level: SkillProficiencyLevel = SkillProficiencyLevel.NOVICE
    proficiency_score: float = 0.0  # 0.0-1.0
    execution_count: int = 0
    success_count: int = 0
    average_execution_time: float = 0.0
    created_time: float = field(default_factory=time.time)
    last_executed_time: Optional[float] = None
    last_improved_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.proficiency_score = max(0.0, min(1.0, self.proficiency_score))
        if self.last_executed_time is None:
            self.last_executed_time = self.created_time
        if self.last_improved_time is None:
            self.last_improved_time = self.created_time
    
    @property
    def success_rate(self) -> float:
        """获取成功率"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count
    
    @property
    def is_automated(self) -> bool:
        """判断是否自动化"""
        return self.proficiency_level in [SkillProficiencyLevel.PROFICIENT, 
                                         SkillProficiencyLevel.EXPERT]
    
    def update_execution(self, success: bool, execution_time: float):
        """更新执行信息"""
        self.execution_count += 1
        
        if success:
            self.success_count += 1
        
        # 更新平均执行时间
        if self.average_execution_time == 0:
            self.average_execution_time = execution_time
        else:
            self.average_execution_time = (
                self.average_execution_time * (self.execution_count - 1) + execution_time
            ) / self.execution_count
        
        self.last_executed_time = time.time()
        
        # 更新熟练度分数
        self._update_proficiency_score(success, execution_time)
    
    def _update_proficiency_score(self, success: bool, execution_time: float):
        """更新熟练度分数"""
        # 成功率因子
        success_factor = 1.0 if success else 0.0
        
        # 执行时间因子（与平均时间比较）
        time_factor = 1.0
        if self.average_execution_time > 0:
            time_factor = min(2.0, self.average_execution_time / max(execution_time, 0.001))
        
        # 执行次数因子（练习效应）
        practice_factor = min(1.0, self.execution_count / 100.0)
        
        # 更新分数
        improvement = (success_factor * 0.5 + time_factor * 0.3 + practice_factor * 0.2) / 3.0
        self.proficiency_score = min(1.0, self.proficiency_score + improvement * 0.1)
        
        # 更新熟练度级别
        self._update_proficiency_level()
        
        # 记录改进时间
        if improvement > 0.05:
            self.last_improved_time = time.time()
    
    def _update_proficiency_level(self):
        """更新熟练度级别"""
        if self.proficiency_score < 0.2:
            self.proficiency_level = SkillProficiencyLevel.NOVICE
        elif self.proficiency_score < 0.4:
            self.proficiency_level = SkillProficiencyLevel.BEGINNER
        elif self.proficiency_score < 0.6:
            self.proficiency_level = SkillProficiencyLevel.COMPETENT
        elif self.proficiency_score < 0.8:
            self.proficiency_level = SkillProficiencyLevel.PROFICIENT
        else:
            self.proficiency_level = SkillProficiencyLevel.EXPERT


@dataclass
class SkillExecution:
    """技能执行记录数据类"""
    id: str
    skill_id: str
    execution_status: SkillExecutionStatus
    start_time: float
    end_time: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    feedback: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        if self.end_time is None:
            self.end_time = self.start_time
    
    @property
    def execution_time(self) -> float:
        """获取执行时间"""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    @property
    def was_successful(self) -> bool:
        """判断是否成功"""
        return self.execution_status == SkillExecutionStatus.COMPLETED and not self.errors


@dataclass
class SkillImprovement:
    """技能改进记录数据类"""
    id: str
    skill_id: str
    improvement_type: str
    improvement_time: float
    before_proficiency: float
    after_proficiency: float
    improvement_details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.improvement_amount = self.after_proficiency - self.before_proficiency


class SkillMemoryStore:
    """
    技能记忆存储 - 管理技能和技能执行
    
    核心组件:
    1. 技能管理器: 技能的增删改查
    2. 动作管理器: 动作的定义和管理
    3. 执行记录器: 记录技能执行
    4. 改进分析器: 分析技能改进
    5. 技能检索器: 基于情境检索技能
    
    工作流程:
    技能定义 → 技能管理器 → 存储技能 → 执行记录器 → 记录执行
    执行反馈 → 改进分析器 → 分析改进 → 技能管理器 → 更新技能
    技能请求 → 技能检索器 → 匹配技能 → 返回可用技能
    
    技术特性:
    - 技能参数化和条件化
    - 执行历史记录和分析
    - 熟练度建模和改进
    - 技能迁移和泛化
    - 动作序列优化
    """
    
    def __init__(self,
                 max_skills: int = 1000,
                 max_executions_per_skill: int = 100,
                 auto_improvement_enabled: bool = True):
        """
        初始化技能记忆存储
        
        Args:
            max_skills: 最大技能数量
            max_executions_per_skill: 每个技能的最大执行记录数
            auto_improvement_enabled: 是否启用自动改进
        """
        self.max_skills = max_skills
        self.max_executions_per_skill = max_executions_per_skill
        self.auto_improvement_enabled = auto_improvement_enabled
        
        # 技能存储
        self.skills: Dict[str, Skill] = {}
        
        # 动作存储
        self.actions: Dict[str, Action] = {}
        
        # 执行记录
        self.skill_executions: Dict[str, List[SkillExecution]] = defaultdict(list)
        
        # 改进记录
        self.skill_improvements: Dict[str, List[SkillImprovement]] = defaultdict(list)
        
        # 索引结构
        self.skill_name_index: Dict[str, str] = {}  # 名称到ID的映射
        self.skills_by_type: Dict[SkillType, Set[str]] = defaultdict(set)
        self.skills_by_proficiency: Dict[SkillProficiencyLevel, Set[str]] = defaultdict(set)
        
        # 配置参数
        self.config = {
            'min_proficiency_for_automation': 0.7,
            'improvement_decay_rate': 0.99,
            'forgetting_rate': 0.95,
            'transfer_learning_threshold': 0.6,
            'max_action_sequence_length': 50,
            'skill_matching_threshold': 0.5
        }
        
        # 性能统计
        self.performance_stats = {
            'skills_stored': 0,
            'actions_stored': 0,
            'skill_executions_recorded': 0,
            'skill_improvements_recorded': 0,
            'skills_retrieved': 0,
            'average_retrieval_time': 0.0
        }
        
        # 初始化基础技能
        self._initialize_basic_skills()
        
        logger.info(f"技能记忆存储初始化完成，最大技能: {max_skills}")
    
    def _initialize_basic_skills(self):
        """初始化基础技能"""
        # 添加一些基础动作
        basic_actions = [
            Action(id="action_move", name="移动", action_type="motor", 
                  parameters={"direction": "forward", "distance": 1.0}),
            Action(id="action_grasp", name="抓取", action_type="motor",
                  parameters={"object": "any", "force": "gentle"}),
            Action(id="action_look", name="观察", action_type="perceptual",
                  parameters={"target": "any", "duration": 1.0}),
            Action(id="action_think", name="思考", action_type="cognitive",
                  parameters={"topic": "any", "depth": "shallow"})
        ]
        
        for action in basic_actions:
            self.actions[action.id] = action
        
        # 添加基础技能
        basic_skills = [
            ("basic_movement", "基础移动", SkillType.MOTOR,
             [self.actions["action_move"]], 0.3),
            ("object_manipulation", "物体操作", SkillType.MOTOR,
             [self.actions["action_grasp"], self.actions["action_move"]], 0.2),
            ("visual_inspection", "视觉检查", SkillType.PERCEPTUAL,
             [self.actions["action_look"], self.actions["action_think"]], 0.4)
        ]
        
        for name, description, skill_type, action_seq, proficiency in basic_skills:
            self.add_skill(name, description, skill_type, action_seq, 
                          proficiency_score=proficiency)
        
        logger.info(f"初始化了 {len(basic_skills)} 个基础技能和 {len(basic_actions)} 个基础动作")
    
    def add_skill(self,
                 name: str,
                 description: str,
                 skill_type: SkillType,
                 action_sequence: List[Action],
                 parameters: Optional[Dict[str, Any]] = None,
                 preconditions: Optional[List[str]] = None,
                 postconditions: Optional[List[str]] = None,
                 proficiency_score: float = 0.0,
                 metadata: Optional[Dict[str, Any]] = None) -> Skill:
        """
        添加技能
        
        Args:
            name: 技能名称
            description: 技能描述
            skill_type: 技能类型
            action_sequence: 动作序列
            parameters: 技能参数
            preconditions: 前提条件
            postconditions: 后置条件
            proficiency_score: 初始熟练度分数
            metadata: 元数据
            
        Returns:
            添加的技能
        """
        # 检查技能是否已存在
        if name in self.skill_name_index:
            skill_id = self.skill_name_index[name]
            skill = self.skills[skill_id]
            
            # 更新现有技能
            skill.description = description or skill.description
            skill.action_sequence = action_sequence or skill.action_sequence
            skill.proficiency_score = max(skill.proficiency_score, proficiency_score)
            skill._update_proficiency_level()
            
            if parameters:
                skill.parameters.update(parameters)
            
            if preconditions:
                skill.preconditions = preconditions
            
            if postconditions:
                skill.postconditions = postconditions
            
            if metadata:
                skill.metadata.update(metadata)
            
            logger.info(f"技能更新: {name} (ID: {skill_id})")
            
            return skill
        
        # 生成技能ID
        skill_id = f"skill_{len(self.skills)}"
        
        # 创建技能
        skill = Skill(
            id=skill_id,
            name=name,
            description=description,
            skill_type=skill_type,
            action_sequence=action_sequence,
            parameters=parameters or {},
            preconditions=preconditions or [],
            postconditions=postconditions or [],
            proficiency_score=proficiency_score,
            metadata=metadata or {}
        )
        
        # 存储技能
        self.skills[skill_id] = skill
        self.skill_name_index[name] = skill_id
        
        # 更新索引
        self.skills_by_type[skill_type].add(skill_id)
        self.skills_by_proficiency[skill.proficiency_level].add(skill_id)
        
        # 检查存储容量
        self._check_skill_capacity()
        
        # 更新统计
        self.performance_stats['skills_stored'] += 1
        
        logger.info(f"技能添加: {name} (类型: {skill_type.value})，熟练度: {skill.proficiency_level.value}")
        
        return skill
    
    def _check_skill_capacity(self):
        """检查技能容量"""
        if len(self.skills) > self.max_skills:
            # 删除最不熟练的技能
            self._remove_least_proficient_skills(len(self.skills) - self.max_skills)
    
    def _remove_least_proficient_skills(self, count: int):
        """删除最不熟练的技能"""
        if not self.skills:
            return
        
        # 按熟练度分数排序
        skills_by_proficiency = sorted(self.skills.values(), 
                                      key=lambda s: s.proficiency_score)
        
        for skill in skills_by_proficiency[:count]:
            self._remove_skill(skill.id)
        
        logger.info(f"删除 {count} 个最不熟练的技能")
    
    def _remove_skill(self, skill_id: str):
        """移除技能"""
        if skill_id not in self.skills:
            return
        
        skill = self.skills[skill_id]
        
        # 从名称索引中移除
        if skill.name in self.skill_name_index:
            del self.skill_name_index[skill.name]
        
        # 从类型索引中移除
        self.skills_by_type[skill.skill_type].discard(skill_id)
        
        # 从熟练度索引中移除
        self.skills_by_proficiency[skill.proficiency_level].discard(skill_id)
        
        # 移除执行记录
        if skill_id in self.skill_executions:
            del self.skill_executions[skill_id]
        
        # 移除改进记录
        if skill_id in self.skill_improvements:
            del self.skill_improvements[skill_id]
        
        # 从存储中删除
        del self.skills[skill_id]
        
        logger.info(f"技能移除: {skill.name} (ID: {skill_id})")
    
    def record_skill_execution(self,
                              skill_name: str,
                              execution_status: SkillExecutionStatus,
                              parameters: Optional[Dict[str, Any]] = None,
                              context: Optional[Dict[str, Any]] = None,
                              results: Optional[Dict[str, Any]] = None,
                              errors: Optional[List[str]] = None,
                              feedback: Optional[Dict[str, Any]] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> SkillExecution:
        """
        记录技能执行
        
        Args:
            skill_name: 技能名称
            execution_status: 执行状态
            parameters: 执行参数
            context: 执行情境
            results: 执行结果
            errors: 错误信息
            feedback: 执行反馈
            metadata: 元数据
            
        Returns:
            执行记录
        """
        # 获取技能ID
        skill_id = self.skill_name_index.get(skill_name)
        if not skill_id or skill_id not in self.skills:
            # 创建新技能（如果不存在）
            skill = self.add_skill(skill_name, f"自动创建的技能: {skill_name}", 
                                  SkillType.COMPOSITE, [])
            skill_id = skill.id
        
        skill = self.skills[skill_id]
        
        # 生成执行ID
        execution_id = f"execution_{skill_id}_{len(self.skill_executions[skill_id])}"
        
        # 计算执行时间
        end_time = time.time()
        start_time = end_time  # 简化：假设执行瞬间完成
        
        # 创建执行记录
        execution = SkillExecution(
            id=execution_id,
            skill_id=skill_id,
            execution_status=execution_status,
            start_time=start_time,
            end_time=end_time,
            parameters=parameters or {},
            context=context or {},
            results=results or {},
            errors=errors or [],
            feedback=feedback or {},
            metadata=metadata or {}
        )
        
        # 存储执行记录
        self.skill_executions[skill_id].append(execution)
        
        # 限制执行记录数量
        self._limit_execution_records(skill_id)
        
        # 更新技能执行信息
        success = execution.was_successful
        execution_time = execution.execution_time
        
        skill.update_execution(success, execution_time)
        
        # 更新索引中的熟练度
        old_level = skill.proficiency_level
        skill._update_proficiency_level()
        new_level = skill.proficiency_level
        
        if old_level != new_level:
            # 从旧级别移除，添加到新级别
            self.skills_by_proficiency[old_level].discard(skill_id)
            self.skills_by_proficiency[new_level].add(skill_id)
        
        # 记录改进（如果熟练度提高）
        if skill.proficiency_score > 0 and self.auto_improvement_enabled:
            self._record_skill_improvement(skill_id, execution)
        
        # 更新统计
        self.performance_stats['skill_executions_recorded'] += 1
        
        logger.info(f"技能执行记录: {skill_name}，状态: {execution_status.value}，"
                   f"成功率: {skill.success_rate:.2f}")
        
        return execution
    
    def _limit_execution_records(self, skill_id: str):
        """限制执行记录数量"""
        executions = self.skill_executions.get(skill_id, [])
        if len(executions) > self.max_executions_per_skill:
            # 删除最旧的记录
            executions_to_remove = len(executions) - self.max_executions_per_skill
            self.skill_executions[skill_id] = executions[executions_to_remove:]
            
            logger.debug(f"技能 {skill_id} 的执行记录限制为 {self.max_executions_per_skill}")
    
    def _record_skill_improvement(self, skill_id: str, execution: SkillExecution):
        """记录技能改进"""
        skill = self.skills[skill_id]
        
        # 检查是否需要记录改进
        executions = self.skill_executions.get(skill_id, [])
        if len(executions) < 2:
            return  # 至少需要两次执行才能检测改进
        
        # 获取最近两次执行
        recent_executions = executions[-2:]
        
        # 计算改进
        if len(recent_executions) == 2:
            prev_exec = recent_executions[0]
            curr_exec = recent_executions[1]
            
            # 检查是否有改进
            improvement = self._detect_improvement(prev_exec, curr_exec, skill)
            
            if improvement['has_improvement']:
                # 生成改进ID
                improvement_id = f"improvement_{skill_id}_{len(self.skill_improvements[skill_id])}"
                
                # 创建改进记录
                skill_improvement = SkillImprovement(
                    id=improvement_id,
                    skill_id=skill_id,
                    improvement_type=improvement['type'],
                    improvement_time=time.time(),
                    before_proficiency=skill.proficiency_score - improvement['amount'],
                    after_proficiency=skill.proficiency_score,
                    improvement_details=improvement['details'],
                    metadata={'execution_id': execution.id}
                )
                
                # 存储改进记录
                self.skill_improvements[skill_id].append(skill_improvement)
                
                # 更新统计
                self.performance_stats['skill_improvements_recorded'] += 1
                
                logger.info(f"技能改进记录: {skill.name}，改进类型: {improvement['type']}，"
                           f"改进量: {improvement['amount']:.3f}")
    
    def _detect_improvement(self, 
                           prev_exec: SkillExecution,
                           curr_exec: SkillExecution,
                           skill: Skill) -> Dict[str, Any]:
        """检测改进"""
        improvement = {
            'has_improvement': False,
            'type': 'none',
            'amount': 0.0,
            'details': {}
        }
        
        # 检查成功率改进
        if (not prev_exec.was_successful and curr_exec.was_successful):
            improvement['has_improvement'] = True
            improvement['type'] = 'success_rate'
            improvement['amount'] = 0.1
            improvement['details'] = {'from_failure_to_success': True}
        
        # 检查执行时间改进
        elif (curr_exec.was_successful and prev_exec.was_successful and
              curr_exec.execution_time < prev_exec.execution_time * 0.8):  # 改进20%以上
            improvement['has_improvement'] = True
            improvement['type'] = 'execution_speed'
            improvement['amount'] = 0.05
            improvement['details'] = {
                'time_reduction': prev_exec.execution_time - curr_exec.execution_time,
                'reduction_percentage': (prev_exec.execution_time - curr_exec.execution_time) / prev_exec.execution_time
            }
        
        # 检查错误减少
        elif (len(curr_exec.errors) < len(prev_exec.errors)):
            improvement['has_improvement'] = True
            improvement['type'] = 'error_reduction'
            improvement['amount'] = 0.03
            improvement['details'] = {
                'previous_errors': len(prev_exec.errors),
                'current_errors': len(curr_exec.errors)
            }
        
        return improvement
    
    def get_skill(self, skill_name: str) -> Optional[Skill]:
        """
        获取技能
        
        Args:
            skill_name: 技能名称
            
        Returns:
            技能，如果不存在则返回None
        """
        skill_id = self.skill_name_index.get(skill_name)
        if skill_id and skill_id in self.skills:
            self.performance_stats['skills_retrieved'] += 1
            return self.skills[skill_id]
        
        return None
    
    def find_skills_by_context(self,
                              context: Dict[str, Any],
                              skill_type: Optional[SkillType] = None,
                              min_proficiency: float = 0.0,
                              max_results: int = 10) -> List[Skill]:
        """
        根据情境查找技能
        
        Args:
            context: 情境信息
            skill_type: 限制技能类型
            min_proficiency: 最小熟练度
            max_results: 最大结果数量
            
        Returns:
            匹配的技能列表
        """
        start_time = time.time()
        
        matched_skills = []
        
        # 候选技能
        candidate_skill_ids = set()
        
        if skill_type:
            # 限制技能类型
            candidate_skill_ids.update(self.skills_by_type.get(skill_type, set()))
        else:
            # 所有技能
            candidate_skill_ids.update(self.skills.keys())
        
        # 过滤和匹配
        for skill_id in candidate_skill_ids:
            if skill_id not in self.skills:
                continue
            
            skill = self.skills[skill_id]
            
            # 熟练度过滤
            if skill.proficiency_score < min_proficiency:
                continue
            
            # 情境匹配
            match_score = self._compute_context_match(skill, context)
            
            if match_score >= self.config['skill_matching_threshold']:
                matched_skills.append((match_score, skill))
        
        # 按匹配分数排序
        matched_skills.sort(key=lambda x: x[0], reverse=True)
        
        # 提取技能
        result_skills = [skill for _, skill in matched_skills[:max_results]]
        
        # 更新统计
        retrieval_time = time.time() - start_time
        self._update_retrieval_stats(retrieval_time)
        
        logger.info(f"情境技能查找: 在 {len(candidate_skill_ids)} 个候选技能中找到 {len(result_skills)} 个匹配技能")
        
        return result_skills
    
    def _compute_context_match(self, skill: Skill, context: Dict[str, Any]) -> float:
        """计算情境匹配度"""
        if not context:
            return 0.5  # 无情境时中等匹配度
        
        match_factors = []
        
        # 前提条件匹配
        if skill.preconditions:
            precondition_match = self._match_conditions(skill.preconditions, context)
            match_factors.append(precondition_match * 0.4)
        
        # 技能参数匹配
        if skill.parameters:
            parameter_match = self._match_parameters(skill.parameters, context)
            match_factors.append(parameter_match * 0.3)
        
        # 技能类型匹配
        skill_type_weight = {
            SkillType.MOTOR: 1.0,
            SkillType.COGNITIVE: 0.9,
            SkillType.PERCEPTUAL: 0.8,
            SkillType.SOCIAL: 0.7,
            SkillType.COMPOSITE: 0.6
        }
        type_factor = skill_type_weight.get(skill.skill_type, 0.5) * 0.2
        
        # 熟练度权重
        proficiency_factor = skill.proficiency_score * 0.1
        
        # 综合匹配度
        if match_factors:
            base_match = sum(match_factors) / len(match_factors)
        else:
            base_match = 0.5
        
        total_match = base_match + type_factor + proficiency_factor
        
        return min(1.0, total_match)
    
    def _match_conditions(self, conditions: List[str], context: Dict[str, Any]) -> float:
        """匹配条件"""
        if not conditions:
            return 1.0
        
        matched = 0
        for condition in conditions:
            # 简化：条件与情境键匹配
            if condition in context:
                matched += 1
        
        return matched / len(conditions) if conditions else 0.0
    
    def _match_parameters(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> float:
        """匹配参数"""
        if not parameters:
            return 1.0
        
        matched = 0
        total = 0
        
        for param_key, param_value in parameters.items():
            total += 1
            if param_key in context:
                context_value = context[param_key]
                
                # 简单值匹配
                if param_value == context_value:
                    matched += 1
                elif isinstance(param_value, str) and isinstance(context_value, str):
                    # 字符串包含匹配
                    if param_value in context_value or context_value in param_value:
                        matched += 0.5
        
        return matched / total if total > 0 else 0.0
    
    def _update_retrieval_stats(self, retrieval_time: float):
        """更新检索统计"""
        # 更新平均检索时间
        current_avg = self.performance_stats['average_retrieval_time']
        n_retrievals = self.performance_stats['skills_retrieved']
        
        if n_retrievals > 0:
            new_avg = (current_avg * (n_retrievals - 1) + retrieval_time) / n_retrievals
            self.performance_stats['average_retrieval_time'] = new_avg
    
    def get_skill_execution_history(self,
                                   skill_name: str,
                                   limit: int = 10) -> List[SkillExecution]:
        """
        获取技能执行历史
        
        Args:
            skill_name: 技能名称
            limit: 限制记录数量
            
        Returns:
            执行历史列表
        """
        skill_id = self.skill_name_index.get(skill_name)
        if not skill_id or skill_id not in self.skill_executions:
            return []
        
        executions = self.skill_executions[skill_id]
        return executions[-limit:]  # 返回最近的记录
    
    def get_skill_improvement_history(self,
                                     skill_name: str,
                                     limit: int = 10) -> List[SkillImprovement]:
        """
        获取技能改进历史
        
        Args:
            skill_name: 技能名称
            limit: 限制记录数量
            
        Returns:
            改进历史列表
        """
        skill_id = self.skill_name_index.get(skill_name)
        if not skill_id or skill_id not in self.skill_improvements:
            return []
        
        improvements = self.skill_improvements[skill_id]
        return improvements[-limit:]  # 返回最近的记录
    
    def transfer_skill(self,
                      source_skill_name: str,
                      target_context: Dict[str, Any],
                      new_skill_name: Optional[str] = None) -> Optional[Skill]:
        """
        技能迁移（将技能应用于新情境）
        
        Args:
            source_skill_name: 源技能名称
            target_context: 目标情境
            new_skill_name: 新技能名称
            
        Returns:
            迁移后的技能，如果失败则返回None
        """
        source_skill = self.get_skill(source_skill_name)
        if not source_skill:
            logger.error(f"技能迁移失败: 源技能不存在 ({source_skill_name})")
            return None
        
        # 检查是否适合迁移
        if source_skill.proficiency_score < self.config['transfer_learning_threshold']:
            logger.warning(f"技能迁移警告: 源技能熟练度不足 ({source_skill.proficiency_score:.2f})")
        
        # 创建新技能名称
        if not new_skill_name:
            new_skill_name = f"{source_skill_name}_transferred_{int(time.time())}"
        
        # 创建迁移后的技能
        transferred_skill = self.add_skill(
            name=new_skill_name,
            description=f"从 {source_skill_name} 迁移的技能，适应新情境",
            skill_type=source_skill.skill_type,
            action_sequence=source_skill.action_sequence.copy(),
            parameters=source_skill.parameters.copy(),
            preconditions=list(target_context.keys()),  # 使用新情境作为前提条件
            postconditions=source_skill.postconditions.copy(),
            proficiency_score=source_skill.proficiency_score * 0.7,  # 迁移后熟练度降低
            metadata={
                'transferred_from': source_skill_name,
                'transfer_context': target_context,
                'transfer_time': time.time()
            }
        )
        
        logger.info(f"技能迁移完成: {source_skill_name} -> {new_skill_name}，"
                   f"新熟练度: {transferred_skill.proficiency_score:.2f}")
        
        return transferred_skill
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 技能统计
        skill_counts_by_type = {t.value: len(ids) for t, ids in self.skills_by_type.items()}
        
        skill_counts_by_proficiency = {l.value: len(ids) for l, ids in self.skills_by_proficiency.items()}
        
        # 执行统计
        total_executions = sum(len(execs) for execs in self.skill_executions.values())
        total_improvements = sum(len(imps) for imps in self.skill_improvements.values())
        
        # 计算平均成功率
        total_success_rate = 0.0
        skills_with_executions = 0
        
        for skill in self.skills.values():
            if skill.execution_count > 0:
                total_success_rate += skill.success_rate
                skills_with_executions += 1
        
        avg_success_rate = total_success_rate / skills_with_executions if skills_with_executions > 0 else 0.0
        
        return {
            'total_skills': len(self.skills),
            'skill_counts_by_type': skill_counts_by_type,
            'skill_counts_by_proficiency': skill_counts_by_proficiency,
            'total_actions': len(self.actions),
            'total_executions': total_executions,
            'total_improvements': total_improvements,
            'average_success_rate': avg_success_rate,
            'performance_stats': self.performance_stats
        }


# 示例和测试函数
def create_example_skill_store() -> SkillMemoryStore:
    """创建示例技能记忆存储"""
    store = SkillMemoryStore(
        max_skills=500,
        max_executions_per_skill=50,
        auto_improvement_enabled=True
    )
    return store


def test_skill_memory_store():
    """测试技能记忆存储"""
    logger.info("开始测试技能记忆存储")
    
    # 创建示例存储
    store = create_example_skill_store()
    
    # 添加自定义技能
    logger.info("添加自定义技能...")
    
    # 创建动作
    approach_action = Action(
        id="approach_object",
        name="接近物体",
        action_type="motor",
        parameters={"speed": "slow", "distance": 1.0}
    )
    
    inspect_action = Action(
        id="inspect_object",
        name="检查物体",
        action_type="perceptual",
        parameters={"detail_level": "high", "duration": 2.0}
    )
    
    # 添加技能
    inspection_skill = store.add_skill(
        name="object_inspection",
        description="接近并检查物体",
        skill_type=SkillType.COMPOSITE,
        action_sequence=[approach_action, inspect_action],
        preconditions=["object_visible", "area_clear"],
        postconditions=["object_identified", "inspection_complete"],
        proficiency_score=0.3
    )
    
    # 记录技能执行
    logger.info("记录技能执行...")
    
    # 第一次执行（失败）
    execution1 = store.record_skill_execution(
        skill_name="object_inspection",
        execution_status=SkillExecutionStatus.FAILED,
        context={"object_visible": True, "area_clear": False},
        errors=["路径被阻挡"],
        feedback={"difficulty": "high"}
    )
    
    # 第二次执行（成功）
    execution2 = store.record_skill_execution(
        skill_name="object_inspection",
        execution_status=SkillExecutionStatus.COMPLETED,
        context={"object_visible": True, "area_clear": True},
        results={"object_type": "box", "color": "red"},
        feedback={"difficulty": "medium"}
    )
    
    # 第三次执行（更快成功）
    execution3 = store.record_skill_execution(
        skill_name="object_inspection",
        execution_status=SkillExecutionStatus.COMPLETED,
        context={"object_visible": True, "area_clear": True, "lighting": "good"},
        results={"object_type": "box", "color": "red", "size": "large"},
        feedback={"difficulty": "low"}
    )
    
    logger.info(f"技能执行记录: {inspection_skill.execution_count} 次执行，"
               f"成功率: {inspection_skill.success_rate:.2f}，"
               f"熟练度: {inspection_skill.proficiency_level.value}")
    
    # 情境技能查找
    logger.info("情境技能查找...")
    context = {"object_visible": True, "area_clear": True, "task": "inspection"}
    
    matched_skills = store.find_skills_by_context(
        context=context,
        min_proficiency=0.2,
        max_results=5
    )
    
    logger.info(f"在情境中找到 {len(matched_skills)} 个匹配技能")
    for skill in matched_skills:
        logger.info(f"  技能: {skill.name}，类型: {skill.skill_type.value}，"
                   f"熟练度: {skill.proficiency_score:.2f}")
    
    # 技能迁移
    logger.info("技能迁移测试...")
    new_context = {"object_visible": True, "area_clear": True, "environment": "indoor"}
    
    transferred_skill = store.transfer_skill(
        source_skill_name="object_inspection",
        target_context=new_context,
        new_skill_name="indoor_object_inspection"
    )
    
    if transferred_skill:
        logger.info(f"技能迁移成功: {transferred_skill.name}，"
                   f"新熟练度: {transferred_skill.proficiency_score:.2f}")
    
    # 获取执行历史
    logger.info("获取执行历史...")
    execution_history = store.get_skill_execution_history(
        skill_name="object_inspection",
        limit=5
    )
    
    logger.info(f"执行历史: {len(execution_history)} 条记录")
    
    # 获取统计信息
    stats = store.get_statistics()
    logger.info(f"统计信息: {stats['total_skills']} 个技能，{stats['total_executions']} 次执行，"
               f"平均成功率: {stats['average_success_rate']:.2f}")
    
    logger.info("技能记忆存储测试完成")
    return store


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_skill_store_instance = test_skill_memory_store()