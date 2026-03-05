#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zlib
"""
能力提升目标生成 - 实现基于技能掌握和性能优化的目标生成

核心功能:
1. 技能评估: 评估当前技能熟练度和性能
2. 改进机会识别: 识别技能改进的潜在机会
3. 目标制定: 制定具体可衡量的能力提升目标
4. 练习计划生成: 生成系统化的练习和改进计划
5. 进度监控: 监控能力提升进度和效果

能力维度:
1. 熟练度 (Proficiency): 技能执行的准确性和流畅性
2. 效率 (Efficiency): 技能执行的速度和资源使用
3. 可靠性 (Reliability): 技能在不同条件下的稳定性和一致性
4. 适应性 (Adaptability): 技能在新情境中的应用能力
5. 创新性 (Innovativeness): 技能应用中的创造性和优化能力

改进策略:
1. 刻意练习 (Deliberate Practice): 有针对性的重复练习
2. 变异性练习 (Variable Practice): 在不同条件下练习
3. 反馈循环 (Feedback Loop): 基于结果的即时反馈和调整
4. 渐进挑战 (Progressive Challenge): 逐步增加任务难度
5. 跨技能迁移 (Cross-Skill Transfer): 利用相关技能的迁移学习

目标类型:
1. 熟练度提升目标: 提高特定技能的准确性和流畅性
2. 效率优化目标: 减少执行时间或资源消耗
3. 错误减少目标: 减少特定类型的错误
4. 适应性扩展目标: 扩展技能的应用范围
5. 复合技能目标: 整合多个技能的协同提升

练习机制:
1. 间隔重复: 优化练习间隔以加强记忆
2. 交错练习: 交替练习不同技能以防止遗忘
3. 情境变化: 在不同情境中练习以提高适应性
4. 难度分级: 逐步增加练习难度
5. 自我测试: 通过自我测试评估进步

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


class CompetenceDimension(Enum):
    """能力维度枚举"""
    PROFICIENCY = "proficiency"      # 熟练度
    EFFICIENCY = "efficiency"        # 效率
    RELIABILITY = "reliability"      # 可靠性
    ADAPTABILITY = "adaptability"    # 适应性
    INNOVATIVENESS = "innovativeness"  # 创新性


class ImprovementStrategy(Enum):
    """改进策略枚举"""
    DELIBERATE_PRACTICE = "deliberate_practice"  # 刻意练习
    VARIABLE_PRACTICE = "variable_practice"      # 变异性练习
    FEEDBACK_LOOP = "feedback_loop"              # 反馈循环
    PROGRESSIVE_CHALLENGE = "progressive_challenge"  # 渐进挑战
    CROSS_SKILL_TRANSFER = "cross_skill_transfer"    # 跨技能迁移


class PracticeType(Enum):
    """练习类型枚举"""
    SPACED_REPETITION = "spaced_repetition"    # 间隔重复
    INTERLEAVED_PRACTICE = "interleaved_practice"  # 交错练习
    CONTEXT_VARIATION = "context_variation"    # 情境变化
    DIFFICULTY_GRADATION = "difficulty_gradation"  # 难度分级
    SELF_TESTING = "self_testing"              # 自我测试


@dataclass
class SkillAssessment:
    """技能评估数据类"""
    skill_id: str
    skill_name: str
    assessment_time: float = field(default_factory=time.time)
    proficiency_score: float = 0.5
    efficiency_score: float = 0.5
    reliability_score: float = 0.5
    adaptability_score: float = 0.5
    innovativeness_score: float = 0.5
    execution_count: int = 0
    success_count: int = 0
    average_execution_time: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)
    recent_performance: List[float] = field(default_factory=list)
    improvement_history: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.proficiency_score = max(0.0, min(1.0, self.proficiency_score))
        self.efficiency_score = max(0.0, min(1.0, self.efficiency_score))
        self.reliability_score = max(0.0, min(1.0, self.reliability_score))
        self.adaptability_score = max(0.0, min(1.0, self.adaptability_score))
        self.innovativeness_score = max(0.0, min(1.0, self.innovativeness_score))
    
    @property
    def overall_score(self) -> float:
        """计算总体分数"""
        # 加权平均
        weights = [0.35, 0.25, 0.20, 0.15, 0.05]  # 熟练度权重最高
        scores = [
            self.proficiency_score,
            self.efficiency_score,
            self.reliability_score,
            self.adaptability_score,
            self.innovativeness_score
        ]
        
        return sum(w * s for w, s in zip(weights, scores))
    
    @property
    def success_rate(self) -> float:
        """计算成功率"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count
    
    @property
    def improvement_potential(self) -> float:
        """计算改进潜力"""
        # 基于当前分数和最近表现
        current_score = self.overall_score
        improvement = 1.0 - current_score
        
        # 考虑最近表现趋势
        if self.recent_performance:
            recent_avg = sum(self.recent_performance) / len(self.recent_performance)
            trend = current_score - recent_avg
            if trend > 0:
                # 正在改进，潜力略减
                improvement *= 0.9
            elif trend < 0:
                # 在退步，潜力增加
                improvement *= 1.1
        
        return max(0.0, min(1.0, improvement))
    
    def update_from_execution(self, success: bool, execution_time: float, errors: List[str]):
        """从执行结果更新评估"""
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
        
        # 更新错误类型统计
        for error in errors:
            self.error_types[error] = self.error_types.get(error, 0) + 1
        
        # 更新最近表现（最近10次）
        performance_score = 1.0 if success else 0.5  # 成功1.0，失败0.5
        self.recent_performance.append(performance_score)
        if len(self.recent_performance) > 10:
            self.recent_performance.pop(0)
        
        # 重新计算分数
        self._recalculate_scores()
    
    def _recalculate_scores(self):
        """重新计算分数"""
        # 熟练度：基于成功率
        self.proficiency_score = self.success_rate
        
        # 效率：基于执行时间（假设有基准时间）
        efficiency_base = 1.0
        if self.average_execution_time > 0:
            # 简化：执行时间越短，效率越高
            time_factor = min(1.0, efficiency_base / max(0.1, self.average_execution_time))
            self.efficiency_score = time_factor * 0.7 + self.success_rate * 0.3
        
        # 可靠性：基于错误率和一致性
        error_rate = 1.0 - self.success_rate
        consistency = 1.0 - self._calculate_performance_variance()
        self.reliability_score = (1.0 - error_rate) * 0.6 + consistency * 0.4
        
        # 适应性：基于在不同条件下的表现（简化）
        self.adaptability_score = min(1.0, self.success_rate * 1.2)
        
        # 创新性：基于改进历史和表现变化（简化）
        if self.improvement_history:
            avg_improvement = sum(self.improvement_history) / len(self.improvement_history)
            self.innovativeness_score = min(1.0, avg_improvement * 2.0)
    
    def _calculate_performance_variance(self) -> float:
        """计算表现方差"""
        if len(self.recent_performance) < 2:
            return 0.0
        
        mean = sum(self.recent_performance) / len(self.recent_performance)
        variance = sum((x - mean) ** 2 for x in self.recent_performance) / len(self.recent_performance)
        return min(1.0, variance)  # 归一化


@dataclass
class ImprovementOpportunity:
    """改进机会数据类"""
    skill_id: str
    dimension: CompetenceDimension
    current_level: float
    target_level: float
    improvement_gap: float
    priority: float = 0.5
    estimated_effort: float = 1.0
    estimated_benefit: float = 1.0
    suggested_strategies: List[ImprovementStrategy] = field(default_factory=list)
    detection_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.current_level = max(0.0, min(1.0, self.current_level))
        self.target_level = max(0.0, min(1.0, self.target_level))
        self.improvement_gap = max(0.0, self.target_level - self.current_level)
        self.priority = max(0.0, min(1.0, self.priority))
        self.estimated_effort = max(0.0, self.estimated_effort)
        self.estimated_benefit = max(0.0, self.estimated_benefit)
    
    @property
    def improvement_ratio(self) -> float:
        """计算改进比率"""
        if self.improvement_gap == 0:
            return 0.0
        return self.estimated_benefit / self.estimated_effort


@dataclass
class CompetenceGoal:
    """能力目标数据类"""
    id: str
    skill_id: str
    goal_description: str
    target_dimension: CompetenceDimension
    current_level: float
    target_level: float
    improvement_strategy: ImprovementStrategy
    practice_type: PracticeType
    success_criteria: List[str] = field(default_factory=list)
    practice_plan: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: float = 1.0  # 小时
    estimated_effort: float = 1.0
    expected_improvement: float = 0.1
    priority: float = 0.5
    creation_time: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.current_level = max(0.0, min(1.0, self.current_level))
        self.target_level = max(0.0, min(1.0, self.target_level))
        self.expected_improvement = max(0.0, min(1.0, self.expected_improvement))
        self.priority = max(0.0, min(1.0, self.priority))
        self.estimated_duration = max(0.0, self.estimated_duration)
        self.estimated_effort = max(0.0, self.estimated_effort)
        
        if not self.id:
            self.id = f"competence_goal_{int(time.time())}_{(zlib.adler32(str(self.goal_description).encode('utf-8')) & 0xffffffff) % 10000}"
    
    @property
    def improvement_gap(self) -> float:
        """计算改进差距"""
        return self.target_level - self.current_level
    
    @property
    def urgency(self) -> float:
        """计算紧迫性"""
        if self.deadline is None:
            return 0.3  # 无截止日期时中等紧迫性
        
        time_remaining = self.deadline - time.time()
        if time_remaining <= 0:
            return 1.0  # 已过截止日期
        
        # 根据剩余时间计算紧迫性
        days_remaining = time_remaining / (24 * 3600)
        
        if days_remaining < 1:
            return 0.9  # 少于1天，非常紧急
        elif days_remaining < 3:
            return 0.7  # 1-3天，紧急
        elif days_remaining < 7:
            return 0.5  # 3-7天，中等
        else:
            return 0.3  # 超过7天，不紧急
    
    @property
    def importance_score(self) -> float:
        """计算重要性分数"""
        # 基于优先级、改进差距和紧迫性
        return (self.priority * 0.4 + 
                self.improvement_gap * 0.3 + 
                self.urgency * 0.3)


@dataclass
class PracticeSession:
    """练习会话数据类"""
    id: str
    goal_id: str
    practice_type: PracticeType
    start_time: float
    end_time: Optional[float] = None
    duration: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    errors_made: List[str] = field(default_factory=list)
    insights_gained: List[str] = field(default_factory=list)
    feedback: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    
    def __post_init__(self):
        """后初始化验证"""
        if self.end_time is None:
            self.end_time = self.start_time
    
    @property
    def session_duration(self) -> float:
        """获取会话持续时间"""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    def complete_session(self, 
                        performance_metrics: Dict[str, float],
                        errors: List[str],
                        insights: List[str],
                        feedback: Dict[str, Any]):
        """完成练习会话"""
        self.end_time = time.time()
        self.duration = self.session_duration
        self.performance_metrics = performance_metrics
        self.errors_made = errors
        self.insights_gained = insights
        self.feedback = feedback
        self.completed = True


class CompetenceBasedGoals:
    """
    能力提升目标生成系统
    
    核心组件:
    1. 技能评估器: 评估当前技能状态和性能
    2. 机会检测器: 检测技能改进的潜在机会
    3. 目标生成器: 生成具体可衡量的能力提升目标
    4. 练习计划器: 制定系统化的练习计划
    5. 进度监控器: 监控练习进度和效果
    6. 策略优化器: 优化改进策略和练习方法
    
    工作流程:
    技能执行数据 → 技能评估器 → 评估当前状态 → 机会检测器 → 检测改进机会
    改进机会 → 目标生成器 → 生成能力目标 → 练习计划器 → 制定练习计划
    练习计划 → 进度监控器 → 监控练习进度 → 策略优化器 → 优化改进策略
    
    技术特性:
    - 多维度技能评估
    - 个性化改进机会检测
    - 科学练习计划生成
    - 实时进度监控和反馈
    - 自适应策略优化
    """
    
    def __init__(self,
                 max_skills_tracked: int = 50,
                 min_improvement_threshold: float = 0.05,
                 practice_intensity: float = 0.7,
                 review_interval_days: float = 7.0):
        """
        初始化能力提升目标生成系统
        
        Args:
            max_skills_tracked: 最大跟踪技能数量
            min_improvement_threshold: 最小改进阈值
            practice_intensity: 练习强度 (0.0-1.0)
            review_interval_days: 复查间隔天数
        """
        self.max_skills_tracked = max_skills_tracked
        self.min_improvement_threshold = min_improvement_threshold
        self.practice_intensity = practice_intensity
        self.review_interval_days = review_interval_days
        
        # 技能评估管理
        self.skill_assessments: Dict[str, SkillAssessment] = {}
        
        # 改进机会
        self.improvement_opportunities: List[ImprovementOpportunity] = []
        
        # 能力目标
        self.competence_goals: Dict[str, CompetenceGoal] = {}
        
        # 练习会话
        self.practice_sessions: Dict[str, List[PracticeSession]] = defaultdict(list)
        
        # 配置参数
        self.config = {
            'proficiency_weight': 0.35,
            'efficiency_weight': 0.25,
            'reliability_weight': 0.20,
            'adaptability_weight': 0.15,
            'innovativeness_weight': 0.05,
            'improvement_gap_threshold': 0.1,
            'max_goals_per_skill': 3,
            'max_practice_sessions_per_day': 5,
            'practice_session_duration_range': (0.5, 2.0),  # 小时
            'skill_decay_rate': 0.98,
            'goal_completion_bonus': 0.1
        }
        
        # 性能统计
        self.performance_stats = {
            'skills_assessed': 0,
            'improvement_opportunities_detected': 0,
            'competence_goals_created': 0,
            'competence_goals_completed': 0,
            'practice_sessions_completed': 0,
            'total_practice_time': 0.0,
            'average_improvement_per_goal': 0.0,
            'skill_level_increase': 0.0
        }
        
        # 状态变量
        self.last_review_time = time.time()
        
        logger.info(f"能力提升目标生成系统初始化完成，最大跟踪技能: {max_skills_tracked}")
    
    def assess_skill(self,
                    skill_id: str,
                    skill_name: str,
                    execution_data: Optional[Dict[str, Any]] = None) -> SkillAssessment:
        """
        评估技能
        
        Args:
            skill_id: 技能ID
            skill_name: 技能名称
            execution_data: 执行数据
            
        Returns:
            技能评估结果
        """
        # 检查是否已有评估
        if skill_id in self.skill_assessments:
            assessment = self.skill_assessments[skill_id]
            
            # 如果有执行数据，更新评估
            if execution_data:
                success = execution_data.get('success', False)
                execution_time = execution_data.get('execution_time', 0.0)
                errors = execution_data.get('errors', [])
                
                assessment.update_from_execution(success, execution_time, errors)
                
                # 记录改进历史
                new_score = assessment.overall_score
                if assessment.improvement_history:
                    old_score = assessment.improvement_history[-1]
                    improvement = new_score - old_score
                    assessment.improvement_history.append(improvement)
                else:
                    assessment.improvement_history.append(0.0)
            
            return assessment
        
        # 创建新评估
        assessment = SkillAssessment(
            skill_id=skill_id,
            skill_name=skill_name
        )
        
        # 如果有初始执行数据，使用它
        if execution_data:
            success = execution_data.get('success', False)
            execution_time = execution_data.get('execution_time', 0.0)
            errors = execution_data.get('errors', [])
            
            assessment.update_from_execution(success, execution_time, errors)
        
        # 存储评估
        self.skill_assessments[skill_id] = assessment
        
        # 检查技能数量限制
        self._check_skill_limit()
        
        # 更新统计
        self.performance_stats['skills_assessed'] += 1
        
        logger.info(f"技能评估完成: {skill_name} (ID: {skill_id})，总体分数: {assessment.overall_score:.3f}")
        
        return assessment
    
    def _check_skill_limit(self):
        """检查技能数量限制"""
        if len(self.skill_assessments) > self.max_skills_tracked:
            # 删除总体分数最低的技能
            skills_by_score = sorted(self.skill_assessments.items(),
                                   key=lambda x: x[1].overall_score)
            
            for skill_id, _ in skills_by_score[:len(self.skill_assessments) - self.max_skills_tracked]:
                self._remove_skill(skill_id)
            
            logger.info(f"删除 {len(self.skill_assessments) - self.max_skills_tracked} 个低分技能以保持数量限制")
    
    def _remove_skill(self, skill_id: str):
        """移除技能"""
        if skill_id in self.skill_assessments:
            skill_name = self.skill_assessments[skill_id].skill_name
            del self.skill_assessments[skill_id]
            
            # 移除相关改进机会
            self.improvement_opportunities = [
                opp for opp in self.improvement_opportunities 
                if opp.skill_id != skill_id
            ]
            
            # 移除相关目标
            goals_to_remove = [
                goal_id for goal_id, goal in self.competence_goals.items()
                if goal.skill_id == skill_id
            ]
            
            for goal_id in goals_to_remove:
                del self.competence_goals[goal_id]
            
            logger.debug(f"技能移除: {skill_name} (ID: {skill_id})")
    
    def detect_improvement_opportunities(self, 
                                        skill_filter: Optional[List[str]] = None) -> List[ImprovementOpportunity]:
        """
        检测改进机会
        
        Args:
            skill_filter: 技能ID过滤列表
            
        Returns:
            改进机会列表
        """
        opportunities = []
        
        # 确定要检查的技能
        skills_to_check = list(self.skill_assessments.keys())
        if skill_filter:
            skills_to_check = [sid for sid in skills_to_check if sid in skill_filter]
        
        for skill_id in skills_to_check:
            assessment = self.skill_assessments[skill_id]
            
            # 检查每个能力维度
            dimensions = [
                (CompetenceDimension.PROFICIENCY, assessment.proficiency_score),
                (CompetenceDimension.EFFICIENCY, assessment.efficiency_score),
                (CompetenceDimension.RELIABILITY, assessment.reliability_score),
                (CompetenceDimension.ADAPTABILITY, assessment.adaptability_score),
                (CompetenceDimension.INNOVATIVENESS, assessment.innovativeness_score)
            ]
            
            for dimension, current_level in dimensions:
                # 检查是否有改进空间
                if current_level < 0.9:  # 低于0.9有改进空间
                    improvement_gap = 1.0 - current_level
                    
                    # 设置目标水平
                    target_level = min(1.0, current_level + 0.2)  # 提高0.2
                    
                    # 计算优先级
                    priority = self._calculate_opportunity_priority(
                        skill_id, dimension, current_level, target_level
                    )
                    
                    # 估计努力和收益
                    estimated_effort = self._estimate_improvement_effort(
                        dimension, improvement_gap
                    )
                    
                    estimated_benefit = self._estimate_improvement_benefit(
                        skill_id, dimension, improvement_gap
                    )
                    
                    # 建议改进策略
                    suggested_strategies = self._suggest_improvement_strategies(
                        dimension, current_level
                    )
                    
                    # 创建改进机会
                    opportunity = ImprovementOpportunity(
                        skill_id=skill_id,
                        dimension=dimension,
                        current_level=current_level,
                        target_level=target_level,
                        improvement_gap=improvement_gap,
                        priority=priority,
                        estimated_effort=estimated_effort,
                        estimated_benefit=estimated_benefit,
                        suggested_strategies=suggested_strategies,
                        metadata={
                            'skill_name': assessment.skill_name,
                            'detection_method': 'automated'
                        }
                    )
                    
                    opportunities.append(opportunity)
        
        # 存储机会
        self.improvement_opportunities = opportunities
        
        # 更新统计
        self.performance_stats['improvement_opportunities_detected'] += len(opportunities)
        
        logger.info(f"检测到 {len(opportunities)} 个改进机会")
        
        return opportunities
    
    def _calculate_opportunity_priority(self,
                                      skill_id: str,
                                      dimension: CompetenceDimension,
                                      current_level: float,
                                      target_level: float) -> float:
        """计算改进机会优先级"""
        assessment = self.skill_assessments.get(skill_id)
        if not assessment:
            return 0.5
        
        # 基于多个因素计算优先级
        factors = []
        
        # 1. 改进差距
        improvement_gap = target_level - current_level
        factors.append(improvement_gap * 1.0)
        
        # 2. 技能重要性（基于执行频率）
        execution_frequency_factor = min(1.0, assessment.execution_count / 100.0)
        factors.append(execution_frequency_factor * 0.8)
        
        # 3. 维度权重
        dimension_weights = {
            CompetenceDimension.PROFICIENCY: 1.0,
            CompetenceDimension.EFFICIENCY: 0.9,
            CompetenceDimension.RELIABILITY: 0.8,
            CompetenceDimension.ADAPTABILITY: 0.7,
            CompetenceDimension.INNOVATIVENESS: 0.6
        }
        factors.append(dimension_weights.get(dimension, 0.7) * 0.6)
        
        # 4. 当前水平（水平越低，优先级越高）
        factors.append((1.0 - current_level) * 0.7)
        
        # 加权平均
        weights = [0.3, 0.3, 0.2, 0.2]
        priority = sum(w * f for w, f in zip(weights, factors))
        
        return min(1.0, priority)
    
    def _estimate_improvement_effort(self,
                                   dimension: CompetenceDimension,
                                   improvement_gap: float) -> float:
        """估计改进努力"""
        # 不同维度的改进难度不同
        difficulty_factors = {
            CompetenceDimension.PROFICIENCY: 1.0,
            CompetenceDimension.EFFICIENCY: 1.2,
            CompetenceDimension.RELIABILITY: 1.5,
            CompetenceDimension.ADAPTABILITY: 1.8,
            CompetenceDimension.INNOVATIVENESS: 2.0
        }
        
        difficulty = difficulty_factors.get(dimension, 1.0)
        effort = improvement_gap * difficulty * 10.0  # 基础努力单位
        
        return max(1.0, effort)
    
    def _estimate_improvement_benefit(self,
                                    skill_id: str,
                                    dimension: CompetenceDimension,
                                    improvement_gap: float) -> float:
        """估计改进收益"""
        assessment = self.skill_assessments.get(skill_id)
        if not assessment:
            return improvement_gap * 5.0
        
        # 基于技能使用频率和维度重要性
        usage_factor = min(1.0, assessment.execution_count / 50.0)
        
        dimension_importance = {
            CompetenceDimension.PROFICIENCY: 1.0,
            CompetenceDimension.EFFICIENCY: 0.9,
            CompetenceDimension.RELIABILITY: 0.8,
            CompetenceDimension.ADAPTABILITY: 0.7,
            CompetenceDimension.INNOVATIVENESS: 0.6
        }
        
        importance = dimension_importance.get(dimension, 0.7)
        
        benefit = improvement_gap * importance * usage_factor * 10.0
        
        return max(1.0, benefit)
    
    def _suggest_improvement_strategies(self,
                                       dimension: CompetenceDimension,
                                       current_level: float) -> List[ImprovementStrategy]:
        """建议改进策略"""
        strategies = []
        
        # 基于维度和当前水平选择策略
        if dimension == CompetenceDimension.PROFICIENCY:
            strategies.append(ImprovementStrategy.DELIBERATE_PRACTICE)
            if current_level < 0.7:
                strategies.append(ImprovementStrategy.FEEDBACK_LOOP)
        
        elif dimension == CompetenceDimension.EFFICIENCY:
            strategies.append(ImprovementStrategy.DELIBERATE_PRACTICE)
            strategies.append(ImprovementStrategy.PROGRESSIVE_CHALLENGE)
        
        elif dimension == CompetenceDimension.RELIABILITY:
            strategies.append(ImprovementStrategy.VARIABLE_PRACTICE)
            strategies.append(ImprovementStrategy.FEEDBACK_LOOP)
        
        elif dimension == CompetenceDimension.ADAPTABILITY:
            strategies.append(ImprovementStrategy.VARIABLE_PRACTICE)
            strategies.append(ImprovementStrategy.CROSS_SKILL_TRANSFER)
        
        elif dimension == CompetenceDimension.INNOVATIVENESS:
            strategies.append(ImprovementStrategy.PROGRESSIVE_CHALLENGE)
            strategies.append(ImprovementStrategy.CROSS_SKILL_TRANSFER)
        
        # 确保至少有一个策略
        if not strategies:
            strategies.append(ImprovementStrategy.DELIBERATE_PRACTICE)
        
        return strategies
    
    def create_competence_goals(self,
                               max_goals: int = 5,
                               opportunity_filter: Optional[List[str]] = None) -> List[CompetenceGoal]:
        """
        创建能力目标
        
        Args:
            max_goals: 最大目标数量
            opportunity_filter: 机会ID过滤列表
            
        Returns:
            创建的能力目标列表
        """
        # 如果没有改进机会，先检测
        if not self.improvement_opportunities:
            self.detect_improvement_opportunities()
        
        # 过滤机会
        opportunities = self.improvement_opportunities
        if opportunity_filter:
            opportunities = [opp for opp in opportunities 
                           if opp.skill_id in opportunity_filter]
        
        # 按优先级排序
        opportunities.sort(key=lambda opp: opp.priority, reverse=True)
        
        created_goals = []
        
        for opportunity in opportunities[:max_goals]:
            # 检查是否已为该技能和维度创建了太多目标
            existing_goals_count = sum(
                1 for goal in self.competence_goals.values()
                if goal.skill_id == opportunity.skill_id and 
                goal.target_dimension == opportunity.dimension
            )
            
            if existing_goals_count >= self.config['max_goals_per_skill']:
                continue
            
            # 创建目标
            goal = self._create_goal_from_opportunity(opportunity)
            
            # 存储目标
            self.competence_goals[goal.id] = goal
            created_goals.append(goal)
            
            # 更新统计
            self.performance_stats['competence_goals_created'] += 1
        
        logger.info(f"创建了 {len(created_goals)} 个能力目标")
        
        return created_goals
    
    def _create_goal_from_opportunity(self, opportunity: ImprovementOpportunity) -> CompetenceGoal:
        """从改进机会创建目标"""
        assessment = self.skill_assessments.get(opportunity.skill_id)
        skill_name = assessment.skill_name if assessment else opportunity.skill_id
        
        # 生成目标描述
        goal_description = self._generate_goal_description(
            skill_name, opportunity.dimension, 
            opportunity.current_level, opportunity.target_level
        )
        
        # 选择改进策略（使用建议的第一个策略）
        improvement_strategy = opportunity.suggested_strategies[0] if opportunity.suggested_strategies else ImprovementStrategy.DELIBERATE_PRACTICE
        
        # 选择练习类型
        practice_type = self._select_practice_type(improvement_strategy, opportunity.dimension)
        
        # 生成练习计划
        practice_plan = self._generate_practice_plan(
            opportunity.skill_id, opportunity.dimension, 
            improvement_strategy, practice_type
        )
        
        # 生成成功标准
        success_criteria = self._generate_success_criteria(
            opportunity.dimension, opportunity.current_level, opportunity.target_level
        )
        
        # 设置截止日期（默认2周）
        deadline = time.time() + (14 * 24 * 3600)  # 14天后
        
        goal = CompetenceGoal(
            id=f"competence_goal_{len(self.competence_goals)}",
            skill_id=opportunity.skill_id,
            goal_description=goal_description,
            target_dimension=opportunity.dimension,
            current_level=opportunity.current_level,
            target_level=opportunity.target_level,
            improvement_strategy=improvement_strategy,
            practice_type=practice_type,
            success_criteria=success_criteria,
            practice_plan=practice_plan,
            estimated_duration=opportunity.estimated_effort * 0.5,  # 小时
            estimated_effort=opportunity.estimated_effort,
            expected_improvement=opportunity.improvement_gap,
            priority=opportunity.priority,
            deadline=deadline,
            metadata={
                'opportunity_id': (zlib.adler32(str(str(opportunity).encode('utf-8')) & 0xffffffff)),
                'skill_name': skill_name,
                'creation_method': 'opportunity_based'
            }
        )
        
        return goal
    
    def _generate_goal_description(self,
                                 skill_name: str,
                                 dimension: CompetenceDimension,
                                 current_level: float,
                                 target_level: float) -> str:
        """生成目标描述"""
        dimension_names = {
            CompetenceDimension.PROFICIENCY: "熟练度",
            CompetenceDimension.EFFICIENCY: "效率",
            CompetenceDimension.RELIABILITY: "可靠性",
            CompetenceDimension.ADAPTABILITY: "适应性",
            CompetenceDimension.INNOVATIVENESS: "创新性"
        }
        
        dim_name = dimension_names.get(dimension, dimension.value)
        
        return f"将{skill_name}的{dim_name}从{current_level:.1%}提升到{target_level:.1%}"
    
    def _select_practice_type(self,
                             strategy: ImprovementStrategy,
                             dimension: CompetenceDimension) -> PracticeType:
        """选择练习类型"""
        # 基于策略和维度选择练习类型
        if strategy == ImprovementStrategy.DELIBERATE_PRACTICE:
            return PracticeType.SPACED_REPETITION
        elif strategy == ImprovementStrategy.VARIABLE_PRACTICE:
            return PracticeType.CONTEXT_VARIATION
        elif strategy == ImprovementStrategy.FEEDBACK_LOOP:
            return PracticeType.SELF_TESTING
        elif strategy == ImprovementStrategy.PROGRESSIVE_CHALLENGE:
            return PracticeType.DIFFICULTY_GRADATION
        elif strategy == ImprovementStrategy.CROSS_SKILL_TRANSFER:
            return PracticeType.INTERLEAVED_PRACTICE
        else:
            return PracticeType.SPACED_REPETITION
    
    def _generate_practice_plan(self,
                              skill_id: str,
                              dimension: CompetenceDimension,
                              strategy: ImprovementStrategy,
                              practice_type: PracticeType) -> Dict[str, Any]:
        """生成练习计划"""
        assessment = self.skill_assessments.get(skill_id)
        skill_name = assessment.skill_name if assessment else skill_id
        
        plan = {
            'skill_id': skill_id,
            'skill_name': skill_name,
            'dimension': dimension.value,
            'strategy': strategy.value,
            'practice_type': practice_type.value,
            'session_duration_range': self.config['practice_session_duration_range'],
            'sessions_per_week': int(7 * self.practice_intensity),  # 基于练习强度
            'total_sessions': 10,  # 默认10次会话
            'progress_checkpoints': [3, 6, 9],  # 进度检查点
            'variation_requirements': [],  # 变异性要求
            'feedback_methods': ['self_assessment', 'performance_metrics']
        }
        
        # 根据策略添加具体要求
        if strategy == ImprovementStrategy.DELIBERATE_PRACTICE:
            plan['focus_areas'] = ['accuracy', 'consistency']
            plan['repetition_count'] = 10
        elif strategy == ImprovementStrategy.VARIABLE_PRACTICE:
            plan['variation_requirements'] = ['context_changes', 'parameter_variations']
            plan['context_count'] = 5
        elif strategy == ImprovementStrategy.FEEDBACK_LOOP:
            plan['feedback_methods'].append('immediate_correction')
            plan['error_analysis_required'] = True
        elif strategy == ImprovementStrategy.PROGRESSIVE_CHALLENGE:
            plan['difficulty_levels'] = ['easy', 'medium', 'hard']
            plan['progression_criteria'] = '80%_success_rate'
        elif strategy == ImprovementStrategy.CROSS_SKILL_TRANSFER:
            plan['related_skills'] = self._identify_related_skills(skill_id)
            plan['integration_exercises'] = True
        
        return plan
    
    def _identify_related_skills(self, skill_id: str) -> List[str]:
        """识别相关技能"""
        # 简化实现：随机选择一些其他技能
        all_skill_ids = list(self.skill_assessments.keys())
        related = [sid for sid in all_skill_ids if sid != skill_id]
        
        # 随机选择1-3个相关技能
        if len(related) > 3:
            related = random.sample(related, random.randint(1, 3))
        elif related:
            related = random.sample(related, min(len(related), random.randint(1, 3)))
        
        return related
    
    def _generate_success_criteria(self,
                                 dimension: CompetenceDimension,
                                 current_level: float,
                                 target_level: float) -> List[str]:
        """生成成功标准"""
        criteria = []
        
        # 基础成功标准
        criteria.append(f"达到目标水平 {target_level:.1%}")
        
        # 维度特定标准
        if dimension == CompetenceDimension.PROFICIENCY:
            criteria.append("连续3次执行成功率超过90%")
        elif dimension == CompetenceDimension.EFFICIENCY:
            criteria.append("执行时间减少20%")
        elif dimension == CompetenceDimension.RELIABILITY:
            criteria.append("在不同条件下保持稳定表现")
        elif dimension == CompetenceDimension.ADAPTABILITY:
            criteria.append("成功应用于3个新情境")
        elif dimension == CompetenceDimension.INNOVATIVENESS:
            criteria.append("提出或实施至少1个改进方法")
        
        return criteria
    
    def get_recommended_goals(self, 
                             n_recommendations: int = 3,
                             skill_filter: Optional[List[str]] = None) -> List[CompetenceGoal]:
        """
        获取推荐目标
        
        Args:
            n_recommendations: 推荐数量
            skill_filter: 技能过滤
            
        Returns:
            推荐目标列表
        """
        # 获取所有目标
        all_goals = list(self.competence_goals.values())
        
        # 应用技能过滤
        if skill_filter:
            all_goals = [goal for goal in all_goals if goal.skill_id in skill_filter]
        
        # 按重要性分数排序
        all_goals.sort(key=lambda goal: goal.importance_score, reverse=True)
        
        # 返回前N个推荐
        return all_goals[:n_recommendations]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 技能统计
        skill_stats = {}
        for skill_id, assessment in self.skill_assessments.items():
            skill_stats[skill_id] = {
                'skill_name': assessment.skill_name,
                'overall_score': assessment.overall_score,
                'execution_count': assessment.execution_count,
                'success_rate': assessment.success_rate,
                'improvement_potential': assessment.improvement_potential
            }
        
        # 目标统计
        goal_stats = {
            'total_goals': len(self.competence_goals),
            'goals_by_dimension': defaultdict(int),
            'goals_by_strategy': defaultdict(int)
        }
        
        for goal in self.competence_goals.values():
            goal_stats['goals_by_dimension'][goal.target_dimension.value] += 1
            goal_stats['goals_by_strategy'][goal.improvement_strategy.value] += 1
        
        # 练习统计
        total_practice_time = sum(
            session.duration for sessions in self.practice_sessions.values()
            for session in sessions
        )
        
        return {
            'skills_tracked': len(self.skill_assessments),
            'improvement_opportunities': len(self.improvement_opportunities),
            'active_goals': len(self.competence_goals),
            'total_practice_sessions': sum(len(sessions) for sessions in self.practice_sessions.values()),
            'total_practice_time': total_practice_time,
            'skill_statistics': skill_stats,
            'goal_statistics': goal_stats,
            'performance_stats': self.performance_stats
        }


# 示例和测试函数
def create_example_competence_system() -> CompetenceBasedGoals:
    """创建示例能力提升系统"""
    system = CompetenceBasedGoals(
        max_skills_tracked=20,
        min_improvement_threshold=0.08,
        practice_intensity=0.6,
        review_interval_days=5.0
    )
    return system


def test_competence_based_goals():
    """测试能力提升目标系统"""
    logger.info("开始测试能力提升目标系统")
    
    # 创建示例系统
    system = create_example_competence_system()
    
    # 评估一些技能
    logger.info("评估技能...")
    
    skills = [
        ("skill_1", "数据清洗"),
        ("skill_2", "机器学习建模"),
        ("skill_3", "报告生成"),
        ("skill_4", "API集成"),
        ("skill_5", "数据可视化")
    ]
    
    for skill_id, skill_name in skills:
        # 模拟一些执行数据
        execution_data = {
            'success': random.random() > 0.3,  # 70%成功率
            'execution_time': random.uniform(1.0, 10.0),
            'errors': [] if random.random() > 0.2 else ['error_type_1', 'error_type_2']
        }
        
        assessment = system.assess_skill(skill_id, skill_name, execution_data)
        logger.info(f"技能评估: {skill_name}，总体分数: {assessment.overall_score:.3f}")
    
    # 检测改进机会
    logger.info("检测改进机会...")
    opportunities = system.detect_improvement_opportunities()
    logger.info(f"检测到 {len(opportunities)} 个改进机会")
    
    # 显示一些机会
    for i, opp in enumerate(opportunities[:3]):
        assessment = system.skill_assessments.get(opp.skill_id)
        skill_name = assessment.skill_name if assessment else opp.skill_id
        logger.info(f"机会 {i+1}: {skill_name} - {opp.dimension.value}，当前: {opp.current_level:.3f}，目标: {opp.target_level:.3f}")
    
    # 创建能力目标
    logger.info("创建能力目标...")
    goals = system.create_competence_goals(max_goals=4)
    logger.info(f"创建了 {len(goals)} 个能力目标")
    
    # 显示目标
    for i, goal in enumerate(goals):
        logger.info(f"目标 {i+1}: {goal.goal_description}，优先级: {goal.priority:.3f}，重要性: {goal.importance_score:.3f}")
    
    # 获取推荐目标
    logger.info("获取推荐目标...")
    recommendations = system.get_recommended_goals(n_recommendations=2)
    logger.info(f"推荐 {len(recommendations)} 个目标")
    
    for i, rec in enumerate(recommendations):
        logger.info(f"推荐 {i+1}: {rec.goal_description}")
    
    # 获取统计信息
    stats = system.get_statistics()
    logger.info(f"统计信息: {stats['skills_tracked']} 个技能，{stats['active_goals']} 个活跃目标")
    
    logger.info("能力提升目标系统测试完成")
    return system


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_system_instance = test_competence_based_goals()