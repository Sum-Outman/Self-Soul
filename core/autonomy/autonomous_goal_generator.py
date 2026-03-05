#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zlib
"""
自主目标生成系统 - 实现基于内在动机的目标生成和选择

核心功能:
1. 动机激活: 基于当前状态激活相应内在动机
2. 机会识别: 识别可能的目标机会和改善空间
3. 目标生成: 将机会转化为具体、可操作的目标
4. 价值评估: 评估目标的期望价值和成本
5. 优先级排序: 根据价值和紧迫性排序目标
6. 目标选择: 选择最优目标进行执行

内在动机类型:
1. 好奇心驱动 (Curiosity-Driven):
   - 对新奇性、不确定性、信息增益的追求
   - 探索未知领域、尝试新方法、学习新知识
   - 衡量指标: 信息增益、预测误差、新奇度

2. 能力提升驱动 (Competence-Driven):
   - 对技能掌握、效率提升、性能优化的追求
   - 练习已有技能、优化操作流程、提高成功率
   - 衡量指标: 成功率、效率、熟练度

3. 知识完整性驱动 (Knowledge-Completeness):
   - 对知识体系完整性和一致性的追求
   - 填补知识空白、解决认知矛盾、建立知识联系
   - 衡量指标: 知识覆盖率、一致性、解释力

目标生成过程:
1. 监测当前状态和性能
2. 激活相应的内在动机
3. 识别潜在目标机会
4. 生成具体目标候选
5. 评估目标价值和成本
6. 选择最优目标执行

技术特性:
- 多动机并行激活和竞争
- 动态目标价值评估
- 风险和不确定性建模
- 从执行结果中学习优化

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
import math
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


class MotivationType(Enum):
    """动机类型枚举"""
    CURIOSITY = "curiosity"          # 好奇心驱动
    COMPETENCE = "competence"        # 能力提升驱动
    KNOWLEDGE_COMPLETENESS = "knowledge_completeness"  # 知识完整性驱动
    SAFETY = "safety"                # 安全驱动
    EFFICIENCY = "efficiency"        # 效率驱动


class GoalType(Enum):
    """目标类型枚举"""
    EXPLORATION = "exploration"      # 探索目标
    LEARNING = "learning"            # 学习目标
    OPTIMIZATION = "optimization"    # 优化目标
    INTEGRATION = "integration"      # 整合目标
    MAINTENANCE = "maintenance"      # 维护目标


class GoalStatus(Enum):
    """目标状态枚举"""
    PROPOSED = "proposed"            # 已提出
    EVALUATED = "evaluated"          # 已评估
    SELECTED = "selected"            # 已选择
    EXECUTING = "executing"          # 执行中
    COMPLETED = "completed"          # 完成
    FAILED = "failed"                # 失败
    CANCELLED = "cancelled"          # 取消


class GoalPriority(Enum):
    """目标优先级枚举"""
    LOW = "low"                      # 低优先级
    MEDIUM = "medium"                # 中优先级
    HIGH = "high"                    # 高优先级
    CRITICAL = "critical"            # 关键优先级


@dataclass
class MotivationState:
    """动机状态数据类"""
    motivation_type: MotivationType
    activation_level: float = 0.0
    last_activation_time: float = field(default_factory=time.time)
    activation_count: int = 0
    satisfaction_level: float = 0.5
    growth_need: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.activation_level = max(0.0, min(1.0, self.activation_level))
        self.satisfaction_level = max(0.0, min(1.0, self.satisfaction_level))
        self.growth_need = max(0.0, min(1.0, self.growth_need))
    
    def update_activation(self, new_activation: float):
        """更新激活水平"""
        self.activation_level = max(0.0, min(1.0, new_activation))
        self.last_activation_time = time.time()
        self.activation_count += 1
    
    def update_satisfaction(self, new_satisfaction: float):
        """更新满足度"""
        self.satisfaction_level = max(0.0, min(1.0, new_satisfaction))
        # 满足度降低会增加成长需求
        self.growth_need = max(0.0, min(1.0, 1.0 - self.satisfaction_level))


@dataclass
class GoalCandidate:
    """目标候选数据类"""
    id: str
    description: str
    goal_type: GoalType
    motivation_type: MotivationType
    expected_value: float = 0.0
    expected_cost: float = 0.0
    expected_duration: float = 0.0
    success_probability: float = 0.5
    novelty_score: float = 0.0
    competence_gain: float = 0.0
    knowledge_gain: float = 0.0
    urgency: float = 0.0
    prerequisites: List[str] = field(default_factory=list)
    success_conditions: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.expected_value = max(0.0, self.expected_value)
        self.expected_cost = max(0.0, self.expected_cost)
        self.expected_duration = max(0.0, self.expected_duration)
        self.success_probability = max(0.0, min(1.0, self.success_probability))
        self.novelty_score = max(0.0, min(1.0, self.novelty_score))
        self.competence_gain = max(0.0, min(1.0, self.competence_gain))
        self.knowledge_gain = max(0.0, min(1.0, self.knowledge_gain))
        self.urgency = max(0.0, min(1.0, self.urgency))
        
        if not self.id:
            self.id = f"goal_{int(time.time())}_{(zlib.adler32(str(self.description).encode('utf-8')) & 0xffffffff) % 10000}"
    
    @property
    def expected_utility(self) -> float:
        """计算期望效用"""
        # 效用 = 期望价值 * 成功概率 - 期望成本
        return self.expected_value * self.success_probability - self.expected_cost
    
    @property
    def priority_score(self) -> float:
        """计算优先级分数"""
        # 综合效用、紧迫性、新颖性、能力增益和知识增益
        base_score = self.expected_utility * 0.4
        urgency_score = self.urgency * 0.2
        novelty_score = self.novelty_score * 0.15
        competence_score = self.competence_gain * 0.15
        knowledge_score = self.knowledge_gain * 0.1
        
        return (base_score + urgency_score + novelty_score + 
                competence_score + knowledge_score)


@dataclass
class GoalExecution:
    """目标执行记录数据类"""
    goal_id: str
    start_time: float
    end_time: Optional[float] = None
    execution_status: GoalStatus = GoalStatus.EXECUTING
    actual_value: float = 0.0
    actual_cost: float = 0.0
    actual_duration: float = 0.0
    success_metrics: Dict[str, float] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    feedback: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        if self.end_time is None:
            self.end_time = self.start_time
    
    @property
    def was_successful(self) -> bool:
        """判断是否成功"""
        return self.execution_status == GoalStatus.COMPLETED
    
    @property
    def execution_time(self) -> float:
        """获取执行时间"""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    def complete(self, success: bool, actual_value: float = 0.0, 
                actual_cost: float = 0.0, metrics: Optional[Dict[str, float]] = None):
        """完成目标执行"""
        self.end_time = time.time()
        self.execution_status = GoalStatus.COMPLETED if success else GoalStatus.FAILED
        self.actual_value = actual_value
        self.actual_cost = actual_cost
        self.actual_duration = self.execution_time
        
        if metrics:
            self.success_metrics.update(metrics)


@dataclass
class GoalSelectionResult:
    """目标选择结果数据类"""
    selected_goals: List[GoalCandidate]
    rejected_goals: List[GoalCandidate]
    selection_time: float = field(default_factory=time.time)
    selection_criteria: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        # 确保置信度分数在有效范围内
        for goal_id, score in self.confidence_scores.items():
            self.confidence_scores[goal_id] = max(0.0, min(1.0, score))


class AutonomousGoalGenerator:
    """
    自主目标生成系统
    
    核心组件:
    1. 动机管理器: 管理内在动机状态
    2. 机会检测器: 检测潜在目标机会
    3. 目标生成器: 生成具体目标候选
    4. 价值评估器: 评估目标价值和成本
    5. 选择决策器: 选择最优目标组合
    6. 执行监控器: 监控目标执行过程
    7. 学习优化器: 从执行结果中学习
    
    工作流程:
    当前状态 → 动机管理器 → 激活动机 → 机会检测器 → 检测机会
    机会 → 目标生成器 → 生成目标候选 → 价值评估器 → 评估价值
    价值评估 → 选择决策器 → 选择目标 → 执行监控器 → 监控执行
    执行结果 → 学习优化器 → 改进目标生成 → 反馈到动机管理器
    
    技术特性:
    - 多动机并行激活和竞争
    - 动态目标价值评估
    - 风险和不确定性建模
    - 从执行结果中学习优化
    """
    
    def __init__(self,
                 max_goals_per_cycle: int = 10,
                 min_goal_utility: float = 0.1,
                 exploration_rate: float = 0.3,
                 learning_rate: float = 0.1):
        """
        初始化自主目标生成系统
        
        Args:
            max_goals_per_cycle: 每个周期最大目标数量
            min_goal_utility: 最小目标效用阈值
            exploration_rate: 探索率（新奇性目标比例）
            learning_rate: 学习率
        """
        self.max_goals_per_cycle = max_goals_per_cycle
        self.min_goal_utility = min_goal_utility
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        
        # 动机状态管理
        self.motivation_states: Dict[MotivationType, MotivationState] = {}
        self._initialize_motivations()
        
        # 目标管理
        self.goal_candidates: Dict[str, GoalCandidate] = {}
        self.selected_goals: Dict[str, GoalCandidate] = {}
        self.goal_executions: Dict[str, List[GoalExecution]] = defaultdict(list)
        
        # 知识和技能状态（延迟初始化）
        self.knowledge_state: Optional[Any] = None
        self.skill_state: Optional[Any] = None
        
        # 配置参数
        self.config = {
            'curiosity_decay_rate': 0.95,
            'competence_growth_rate': 1.05,
            'knowledge_completeness_weight': 0.3,
            'safety_weight': 0.2,
            'efficiency_weight': 0.15,
            'value_estimation_confidence': 0.7,
            'cost_estimation_confidence': 0.6,
            'novelty_threshold': 0.3,
            'competence_threshold': 0.4
        }
        
        # 性能统计
        self.performance_stats = {
            'goals_generated': 0,
            'goals_selected': 0,
            'goals_completed': 0,
            'goals_failed': 0,
            'average_goal_utility': 0.0,
            'exploration_ratio': 0.0,
            'learning_rate_adjustments': 0
        }
        
        # 状态变量
        self.last_goal_generation_time = time.time()
        self.cycle_count = 0
        
        logger.info(f"自主目标生成系统初始化完成，最大目标数/周期: {max_goals_per_cycle}")
    
    def _initialize_motivations(self):
        """初始化动机状态"""
        for motivation_type in MotivationType:
            self.motivation_states[motivation_type] = MotivationState(
                motivation_type=motivation_type,
                activation_level=0.5,  # 初始中等激活
                satisfaction_level=0.5,
                growth_need=0.5
            )
    
    def update_system_state(self,
                           knowledge_state: Optional[Any] = None,
                           skill_state: Optional[Any] = None,
                           performance_metrics: Optional[Dict[str, float]] = None):
        """
        更新系统状态
        
        Args:
            knowledge_state: 知识状态
            skill_state: 技能状态
            performance_metrics: 性能指标
        """
        if knowledge_state is not None:
            self.knowledge_state = knowledge_state
        
        if skill_state is not None:
            self.skill_state = skill_state
        
        # 基于性能指标更新动机状态
        if performance_metrics:
            self._update_motivations_from_performance(performance_metrics)
        
        logger.debug("系统状态已更新")
    
    def _update_motivations_from_performance(self, metrics: Dict[str, float]):
        """基于性能指标更新动机状态"""
        # 好奇心动机：基于知识增长和新奇性
        if 'knowledge_growth' in metrics:
            knowledge_growth = metrics['knowledge_growth']
            curiosity_satisfaction = min(1.0, knowledge_growth * 2.0)  # 知识增长提高好奇心满足度
            self.motivation_states[MotivationType.CURIOSITY].update_satisfaction(
                curiosity_satisfaction
            )
        
        # 能力提升动机：基于技能熟练度
        if 'average_skill_proficiency' in metrics:
            proficiency = metrics['average_skill_proficiency']
            competence_satisfaction = proficiency
            self.motivation_states[MotivationType.COMPETENCE].update_satisfaction(
                competence_satisfaction
            )
        
        # 知识完整性动机：基于知识覆盖率
        if 'knowledge_coverage' in metrics:
            coverage = metrics['knowledge_coverage']
            completeness_satisfaction = coverage
            self.motivation_states[MotivationType.KNOWLEDGE_COMPLETENESS].update_satisfaction(
                completeness_satisfaction
            )
        
        # 根据满足度更新激活水平
        for motivation_type, state in self.motivation_states.items():
            # 满足度越低，成长需求越高，激活水平越高
            new_activation = state.growth_need  # 成长需求直接作为激活水平
            state.update_activation(new_activation)
    
    def generate_goal_candidates(self) -> List[GoalCandidate]:
        """
        生成目标候选
        
        Returns:
            目标候选列表
        """
        start_time = time.time()
        
        # 清空旧的目标候选
        self.goal_candidates.clear()
        
        # 基于当前动机状态生成目标
        all_candidates = []
        
        # 好奇心驱动目标
        curiosity_activation = self.motivation_states[MotivationType.CURIOSITY].activation_level
        if curiosity_activation > 0.3:
            curiosity_candidates = self._generate_curiosity_goals(curiosity_activation)
            all_candidates.extend(curiosity_candidates)
        
        # 能力提升驱动目标
        competence_activation = self.motivation_states[MotivationType.COMPETENCE].activation_level
        if competence_activation > 0.3:
            competence_candidates = self._generate_competence_goals(competence_activation)
            all_candidates.extend(competence_candidates)
        
        # 知识完整性驱动目标
        completeness_activation = self.motivation_states[MotivationType.KNOWLEDGE_COMPLETENESS].activation_level
        if completeness_activation > 0.3:
            completeness_candidates = self._generate_knowledge_completeness_goals(completeness_activation)
            all_candidates.extend(completeness_candidates)
        
        # 安全驱动目标（始终考虑）
        safety_candidates = self._generate_safety_goals()
        all_candidates.extend(safety_candidates)
        
        # 效率驱动目标
        efficiency_candidates = self._generate_efficiency_goals()
        all_candidates.extend(efficiency_candidates)
        
        # 存储并返回
        for candidate in all_candidates:
            self.goal_candidates[candidate.id] = candidate
        
        # 更新统计
        self.performance_stats['goals_generated'] += len(all_candidates)
        self.last_goal_generation_time = time.time()
        self.cycle_count += 1
        
        generation_time = time.time() - start_time
        logger.info(f"目标生成完成: 生成了 {len(all_candidates)} 个候选目标，耗时 {generation_time:.2f}秒")
        
        return all_candidates
    
    def _generate_curiosity_goals(self, activation_level: float) -> List[GoalCandidate]:
        """生成好奇心驱动目标"""
        candidates = []
        
        # 基于激活水平确定生成数量
        n_goals = max(1, int(activation_level * 5))
        
        for i in range(n_goals):
            goal_id = f"curiosity_goal_{self.cycle_count}_{i}"
            
            # 随机选择探索方向
            exploration_types = [
                ("探索未知概念", "寻找并学习系统中未充分理解的概念"),
                ("尝试新方法", "用不同的方法解决熟悉的问题"),
                ("调查异常现象", "研究系统性能中的异常或意外结果"),
                ("学习新领域", "扩展知识到当前专业领域之外"),
                ("测试边界条件", "探索系统在极端或边界条件下的行为")
            ]
            
            description, details = random.choice(exploration_types)
            
            # 计算目标参数
            novelty_score = random.uniform(0.5, 1.0) * activation_level
            knowledge_gain = random.uniform(0.3, 0.8)
            success_probability = 0.6 + activation_level * 0.2
            
            # 成本和持续时间
            expected_cost = random.uniform(10.0, 50.0)
            expected_duration = random.uniform(1.0, 10.0) * 3600  # 1-10小时
            
            # 期望价值：基于知识增益和新奇性
            expected_value = (knowledge_gain * 50.0 + novelty_score * 30.0) * activation_level
            
            candidate = GoalCandidate(
                id=goal_id,
                description=description,
                goal_type=GoalType.EXPLORATION,
                motivation_type=MotivationType.CURIOSITY,
                expected_value=expected_value,
                expected_cost=expected_cost,
                expected_duration=expected_duration,
                success_probability=success_probability,
                novelty_score=novelty_score,
                knowledge_gain=knowledge_gain,
                urgency=0.2,  # 探索目标通常不紧急
                context={"exploration_type": description, "details": details},
                metadata={"generation_method": "curiosity_driven"}
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def _generate_competence_goals(self, activation_level: float) -> List[GoalCandidate]:
        """生成能力提升驱动目标"""
        candidates = []
        
        # 基于激活水平确定生成数量
        n_goals = max(1, int(activation_level * 4))
        
        for i in range(n_goals):
            goal_id = f"competence_goal_{self.cycle_count}_{i}"
            
            # 技能提升目标类型
            competence_types = [
                ("提高特定技能熟练度", "通过练习提高关键技能的熟练度"),
                ("优化执行流程", "改进任务执行流程以提高效率"),
                ("减少错误率", "通过针对性训练减少常见错误"),
                ("提高任务完成速度", "优化方法以减少任务完成时间"),
                ("扩展技能适用范围", "将已有技能应用到更广泛的情境")
            ]
            
            description, details = random.choice(competence_types)
            
            # 计算目标参数
            competence_gain = random.uniform(0.4, 0.9) * activation_level
            success_probability = 0.7 + activation_level * 0.2
            
            # 成本和持续时间
            expected_cost = random.uniform(5.0, 30.0)
            expected_duration = random.uniform(0.5, 5.0) * 3600  # 0.5-5小时
            
            # 期望价值：基于能力增益
            expected_value = competence_gain * 80.0 * activation_level
            
            candidate = GoalCandidate(
                id=goal_id,
                description=description,
                goal_type=GoalType.OPTIMIZATION,
                motivation_type=MotivationType.COMPETENCE,
                expected_value=expected_value,
                expected_cost=expected_cost,
                expected_duration=expected_duration,
                success_probability=success_probability,
                competence_gain=competence_gain,
                urgency=0.4,  # 能力提升有中等紧迫性
                context={"competence_type": description, "details": details},
                metadata={"generation_method": "competence_driven"}
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def _generate_knowledge_completeness_goals(self, activation_level: float) -> List[GoalCandidate]:
        """生成知识完整性驱动目标"""
        candidates = []
        
        # 基于激活水平确定生成数量
        n_goals = max(1, int(activation_level * 3))
        
        for i in range(n_goals):
            goal_id = f"knowledge_goal_{self.cycle_count}_{i}"
            
            # 知识完整性目标类型
            knowledge_types = [
                ("填补知识空白", "识别并填补知识体系中的空白"),
                ("解决认知矛盾", "解决不同知识之间的不一致或矛盾"),
                ("建立知识联系", "在不同知识领域之间建立联系"),
                ("验证知识准确性", "验证现有知识的准确性和可靠性"),
                ("系统化知识整理", "将零散知识整理成系统化的体系")
            ]
            
            description, details = random.choice(knowledge_types)
            
            # 计算目标参数
            knowledge_gain = random.uniform(0.5, 1.0) * activation_level
            success_probability = 0.65 + activation_level * 0.15
            
            # 成本和持续时间
            expected_cost = random.uniform(15.0, 40.0)
            expected_duration = random.uniform(2.0, 8.0) * 3600  # 2-8小时
            
            # 期望价值：基于知识增益
            expected_value = knowledge_gain * 60.0 * activation_level
            
            candidate = GoalCandidate(
                id=goal_id,
                description=description,
                goal_type=GoalType.LEARNING,
                motivation_type=MotivationType.KNOWLEDGE_COMPLETENESS,
                expected_value=expected_value,
                expected_cost=expected_cost,
                expected_duration=expected_duration,
                success_probability=success_probability,
                knowledge_gain=knowledge_gain,
                urgency=0.3,  # 知识完整性有中等紧迫性
                context={"knowledge_type": description, "details": details},
                metadata={"generation_method": "knowledge_completeness_driven"}
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def _generate_safety_goals(self) -> List[GoalCandidate]:
        """生成安全驱动目标"""
        candidates = []
        
        # 总是至少生成一个安全目标
        goal_id = f"safety_goal_{self.cycle_count}"
        
        candidate = GoalCandidate(
            id=goal_id,
            description="系统安全检查和维护",
            goal_type=GoalType.MAINTENANCE,
            motivation_type=MotivationType.SAFETY,
            expected_value=100.0,  # 安全有很高价值
            expected_cost=20.0,
            expected_duration=1.0 * 3600,  # 1小时
            success_probability=0.9,
            urgency=0.6,  # 安全有较高紧迫性
            context={"safety_check_type": "comprehensive"},
            metadata={"generation_method": "safety_driven", "priority": "high"}
        )
        
        candidates.append(candidate)
        return candidates
    
    def _generate_efficiency_goals(self) -> List[GoalCandidate]:
        """生成效率驱动目标"""
        candidates = []
        
        # 随机决定是否生成效率目标
        if random.random() < 0.5:  # 50%概率
            goal_id = f"efficiency_goal_{self.cycle_count}"
            
            candidate = GoalCandidate(
                id=goal_id,
                description="优化系统资源使用效率",
                goal_type=GoalType.OPTIMIZATION,
                motivation_type=MotivationType.EFFICIENCY,
                expected_value=40.0,
                expected_cost=15.0,
                expected_duration=2.0 * 3600,  # 2小时
                success_probability=0.75,
                urgency=0.3,  # 效率有中等紧迫性
                context={"optimization_area": "resource_usage"},
                metadata={"generation_method": "efficiency_driven"}
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def evaluate_and_select_goals(self, 
                                 candidates: Optional[List[GoalCandidate]] = None,
                                 max_selected: Optional[int] = None) -> GoalSelectionResult:
        """
        评估并选择目标
        
        Args:
            candidates: 要评估的候选目标（如果为None则使用所有候选）
            max_selected: 最大选择数量（如果为None则使用默认值）
            
        Returns:
            目标选择结果
        """
        start_time = time.time()
        
        if candidates is None:
            candidates = list(self.goal_candidates.values())
        
        if max_selected is None:
            max_selected = min(self.max_goals_per_cycle, len(candidates))
        
        # 评估每个候选目标
        evaluated_candidates = []
        for candidate in candidates:
            # 计算优先级分数
            priority_score = candidate.priority_score
            
            # 应用效用阈值过滤
            if candidate.expected_utility < self.min_goal_utility:
                logger.debug(f"目标 {candidate.id} 因效用过低被过滤: {candidate.expected_utility:.2f}")
                continue
            
            # 应用探索率：确保一定比例的新奇性目标
            is_exploratory = candidate.motivation_type == MotivationType.CURIOSITY
            if is_exploratory:
                # 对探索性目标给予额外加分
                priority_score *= (1.0 + self.exploration_rate)
            
            evaluated_candidates.append((priority_score, candidate))
        
        # 按优先级分数排序
        evaluated_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 选择目标
        selected = []
        rejected = []
        
        for i, (score, candidate) in enumerate(evaluated_candidates):
            if i < max_selected:
                selected.append(candidate)
                self.selected_goals[candidate.id] = candidate
            else:
                rejected.append(candidate)
        
        # 计算置信度分数
        confidence_scores = {}
        for candidate in selected:
            # 置信度基于成功概率和效用
            confidence = min(1.0, candidate.success_probability * 0.7 + 
                           candidate.expected_utility * 0.3)
            confidence_scores[candidate.id] = confidence
        
        # 更新统计
        self.performance_stats['goals_selected'] += len(selected)
        
        selection_time = time.time() - start_time
        logger.info(f"目标选择完成: 从 {len(candidates)} 个候选中选择 {len(selected)} 个目标，耗时 {selection_time:.2f}秒")
        
        return GoalSelectionResult(
            selected_goals=selected,
            rejected_goals=rejected,
            selection_criteria={
                'min_utility_threshold': self.min_goal_utility,
                'exploration_rate': self.exploration_rate,
                'max_selected': max_selected
            },
            confidence_scores=confidence_scores
        )
    
    def execute_goal(self, goal_id: str, **kwargs) -> GoalExecution:
        """
        执行目标
        
        Args:
            goal_id: 目标ID
            **kwargs: 执行参数
            
        Returns:
            目标执行记录
        """
        if goal_id not in self.selected_goals:
            raise ValueError(f"目标不存在或未被选择: {goal_id}")
        
        goal = self.selected_goals[goal_id]
        
        # 创建执行记录
        execution = GoalExecution(
            goal_id=goal_id,
            start_time=time.time(),
            execution_status=GoalStatus.EXECUTING
        )
        
        # 存储执行记录
        self.goal_executions[goal_id].append(execution)
        
        logger.info(f"开始执行目标: {goal.description} (ID: {goal_id})")
        
        return execution
    
    def complete_goal_execution(self, 
                               goal_id: str, 
                               success: bool,
                               actual_value: float = 0.0,
                               actual_cost: float = 0.0,
                               metrics: Optional[Dict[str, float]] = None,
                               feedback: Optional[Dict[str, Any]] = None) -> GoalExecution:
        """
        完成目标执行
        
        Args:
            goal_id: 目标ID
            success: 是否成功
            actual_value: 实际价值
            actual_cost: 实际成本
            metrics: 成功指标
            feedback: 执行反馈
            
        Returns:
            更新后的执行记录
        """
        if goal_id not in self.goal_executions or not self.goal_executions[goal_id]:
            raise ValueError(f"目标执行记录不存在: {goal_id}")
        
        # 获取最新的执行记录
        executions = self.goal_executions[goal_id]
        execution = executions[-1]
        
        # 完成执行
        execution.complete(success, actual_value, actual_cost, metrics)
        
        if feedback:
            execution.feedback.update(feedback)
        
        # 更新统计
        if success:
            self.performance_stats['goals_completed'] += 1
        else:
            self.performance_stats['goals_failed'] += 1
        
        # 更新平均目标效用
        if success and actual_value > 0:
            current_avg = self.performance_stats['average_goal_utility']
            n_completed = self.performance_stats['goals_completed']
            
            new_avg = (current_avg * (n_completed - 1) + actual_value) / n_completed
            self.performance_stats['average_goal_utility'] = new_avg
        
        # 学习：基于执行结果调整参数
        self._learn_from_execution(goal_id, execution)
        
        logger.info(f"目标执行完成: {goal_id}，成功: {success}，实际价值: {actual_value:.2f}")
        
        return execution
    
    def _learn_from_execution(self, goal_id: str, execution: GoalExecution):
        """从执行结果中学习"""
        if goal_id not in self.selected_goals:
            return
        
        goal = self.selected_goals[goal_id]
        
        # 计算预测误差
        value_error = abs(goal.expected_value - execution.actual_value)
        cost_error = abs(goal.expected_cost - execution.actual_cost)
        duration_error = abs(goal.expected_duration - execution.actual_duration)
        
        # 调整学习率（基于预测误差）
        if value_error > goal.expected_value * 0.3:  # 价值预测误差超过30%
            self.learning_rate = min(0.5, self.learning_rate * 1.1)
            self.performance_stats['learning_rate_adjustments'] += 1
            logger.debug(f"学习率调整: {self.learning_rate:.3f} (价值预测误差: {value_error:.2f})")
        
        # 调整探索率（基于目标类型和结果）
        if execution.was_successful:
            if goal.motivation_type == MotivationType.CURIOSITY:
                # 探索成功：稍微增加探索率
                self.exploration_rate = min(0.5, self.exploration_rate * 1.05)
            else:
                # 利用成功：稍微降低探索率
                self.exploration_rate = max(0.1, self.exploration_rate * 0.95)
        
        # 更新探索比率统计
        total_selected = self.performance_stats['goals_selected']
        if total_selected > 0:
            exploration_count = sum(1 for g in self.selected_goals.values() 
                                  if g.motivation_type == MotivationType.CURIOSITY)
            self.performance_stats['exploration_ratio'] = exploration_count / total_selected
    
    def get_goal_recommendations(self, 
                                n_recommendations: int = 5,
                                motivation_filter: Optional[MotivationType] = None) -> List[GoalCandidate]:
        """
        获取目标推荐
        
        Args:
            n_recommendations: 推荐数量
            motivation_filter: 动机类型过滤
            
        Returns:
            推荐的目标列表
        """
        # 如果没有目标候选，先生成一些
        if not self.goal_candidates:
            self.generate_goal_candidates()
        
        # 过滤候选
        candidates = list(self.goal_candidates.values())
        
        if motivation_filter:
            candidates = [c for c in candidates if c.motivation_type == motivation_filter]
        
        # 按优先级分数排序
        candidates.sort(key=lambda c: c.priority_score, reverse=True)
        
        # 返回前N个推荐
        return candidates[:n_recommendations]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 动机统计
        motivation_stats = {}
        for motivation_type, state in self.motivation_states.items():
            motivation_stats[motivation_type.value] = {
                'activation_level': state.activation_level,
                'satisfaction_level': state.satisfaction_level,
                'growth_need': state.growth_need,
                'activation_count': state.activation_count
            }
        
        # 目标类型统计
        goal_type_counts = defaultdict(int)
        for candidate in self.goal_candidates.values():
            goal_type_counts[candidate.goal_type.value] += 1
        
        # 动机类型统计
        motivation_counts = defaultdict(int)
        for candidate in self.goal_candidates.values():
            motivation_counts[candidate.motivation_type.value] += 1
        
        return {
            'cycle_count': self.cycle_count,
            'active_goal_candidates': len(self.goal_candidates),
            'selected_goals': len(self.selected_goals),
            'goal_type_distribution': dict(goal_type_counts),
            'motivation_distribution': dict(motivation_counts),
            'motivation_states': motivation_stats,
            'performance_stats': self.performance_stats,
            'learning_parameters': {
                'exploration_rate': self.exploration_rate,
                'learning_rate': self.learning_rate,
                'min_goal_utility': self.min_goal_utility
            }
        }


# 示例和测试函数
def create_example_goal_generator() -> AutonomousGoalGenerator:
    """创建示例目标生成器"""
    generator = AutonomousGoalGenerator(
        max_goals_per_cycle=8,
        min_goal_utility=0.15,
        exploration_rate=0.35,
        learning_rate=0.12
    )
    return generator


def test_autonomous_goal_generator():
    """测试自主目标生成器"""
    logger.info("开始测试自主目标生成器")
    
    # 创建示例生成器
    generator = create_example_goal_generator()
    
    # 更新系统状态（模拟）
    performance_metrics = {
        'knowledge_growth': 0.6,
        'average_skill_proficiency': 0.7,
        'knowledge_coverage': 0.5
    }
    generator.update_system_state(performance_metrics=performance_metrics)
    
    # 生成目标候选
    logger.info("生成目标候选...")
    candidates = generator.generate_goal_candidates()
    logger.info(f"生成了 {len(candidates)} 个目标候选")
    
    # 显示一些候选信息
    for i, candidate in enumerate(candidates[:3]):
        logger.info(f"候选 {i+1}: {candidate.description}，类型: {candidate.goal_type.value}，"
                   f"动机: {candidate.motivation_type.value}，效用: {candidate.expected_utility:.2f}")
    
    # 评估并选择目标
    logger.info("评估并选择目标...")
    selection_result = generator.evaluate_and_select_goals(max_selected=3)
    
    logger.info(f"选择了 {len(selection_result.selected_goals)} 个目标")
    for goal in selection_result.selected_goals:
        logger.info(f"  选择: {goal.description}，优先级分数: {goal.priority_score:.2f}")
    
    # 模拟目标执行
    if selection_result.selected_goals:
        goal = selection_result.selected_goals[0]
        logger.info(f"模拟执行目标: {goal.description}")
        
        # 开始执行
        execution = generator.execute_goal(goal.id)
        
        # 模拟执行过程
        time.sleep(0.1)  # 模拟执行时间
        
        # 完成执行（假设成功）
        generator.complete_goal_execution(
            goal_id=goal.id,
            success=True,
            actual_value=goal.expected_value * 0.9,  # 实际价值略低于预期
            actual_cost=goal.expected_cost * 1.1,    # 实际成本略高于预期
            metrics={'learning_gain': 0.7, 'efficiency_improvement': 0.3},
            feedback={'difficulty': 'medium', 'insights': ['发现新优化机会']}
        )
    
    # 获取目标推荐
    logger.info("获取目标推荐...")
    recommendations = generator.get_goal_recommendations(n_recommendations=3)
    logger.info(f"推荐 {len(recommendations)} 个目标")
    
    for i, rec in enumerate(recommendations):
        logger.info(f"  推荐 {i+1}: {rec.description}，动机: {rec.motivation_type.value}")
    
    # 获取统计信息
    stats = generator.get_statistics()
    logger.info(f"统计信息: {stats['cycle_count']} 个周期，{stats['active_goal_candidates']} 个活跃候选")
    
    # 显示动机状态
    logger.info("动机状态:")
    for motivation, state in stats['motivation_states'].items():
        logger.info(f"  {motivation}: 激活水平 {state['activation_level']:.2f}，满足度 {state['satisfaction_level']:.2f}")
    
    logger.info("自主目标生成器测试完成")
    return generator


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_generator_instance = test_autonomous_goal_generator()