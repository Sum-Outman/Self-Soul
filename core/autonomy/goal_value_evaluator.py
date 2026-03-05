#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zlib
"""
目标价值评估系统 - 评估目标的认知价值、实用价值、风险和可行性

核心功能:
1. 多维度价值评估: 从认知、实用、安全等多个维度评估目标价值
2. 成本效益分析: 评估目标的执行成本与预期收益
3. 风险评估: 识别和量化目标执行的风险
4. 可行性分析: 评估目标在当前条件下的可行性
5. 综合评分计算: 综合多个维度计算目标总体价值

价值维度:
1. 认知价值 (Cognitive Value):
   - 知识增益: 目标执行带来的新知识和理解
   - 技能提升: 目标执行带来的技能改进和能力增长
   - 认知效率: 目标执行对认知过程的优化作用
   - 知识整合: 目标执行促进的知识体系整合

2. 实用价值 (Practical Value):
   - 问题解决: 目标执行解决的实际问题数量和质量
   - 效率提升: 目标执行带来的时间和资源效率改进
   - 资源获取: 目标执行带来的新资源获取能力
   - 系统优化: 目标执行对系统整体性能的改善

3. 安全价值 (Safety Value):
   - 风险规避: 目标执行带来的风险降低
   - 安全保障: 目标执行增强的系统安全性
   - 合规性: 目标执行符合安全标准和规范
   - 可靠性: 目标执行提高的系统可靠性

4. 探索价值 (Exploration Value):
   - 新奇性: 目标执行探索的新领域或方法
   - 信息增益: 目标执行带来的不确定性和信息增益
   - 多样性: 目标执行增加的知识和行为多样性
   - 长期潜力: 目标执行的长期潜在价值

成本维度:
1. 时间成本: 目标执行所需的时间
2. 计算资源: 目标执行所需的计算资源
3. 数据需求: 目标执行所需的数据和知识
4. 技能要求: 目标执行所需的技能水平
5. 风险成本: 目标失败可能带来的负面影响

风险评估维度:
1. 执行风险: 目标执行过程中可能遇到的技术困难
2. 结果风险: 目标执行结果可能带来的负面后果
3. 依赖风险: 目标依赖的其他条件或资源的不确定性
4. 机会成本: 执行此目标而放弃其他目标的机会成本

可行性评估维度:
1. 技能匹配度: 当前技能与目标要求的匹配程度
2. 资源可用性: 所需资源的可获得性
3. 环境适应性: 当前环境对目标执行的支持程度
4. 时间可行性: 在可用时间内完成目标的可能性

评估方法:
1. 多属性效用理论 (Multi-Attribute Utility Theory, MAUT)
2. 成本效益分析 (Cost-Benefit Analysis, CBA)
3. 风险矩阵分析 (Risk Matrix Analysis)
4. 可行性分析矩阵 (Feasibility Analysis Matrix)
5. 敏感性分析 (Sensitivity Analysis)

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class ValueDimension(Enum):
    """价值维度枚举"""
    COGNITIVE = "cognitive"      # 认知价值
    PRACTICAL = "practical"      # 实用价值
    SAFETY = "safety"            # 安全价值
    EXPLORATION = "exploration"  # 探索价值


class CostDimension(Enum):
    """成本维度枚举"""
    TIME = "time"                # 时间成本
    COMPUTATIONAL = "computational"  # 计算资源成本
    DATA = "data"                # 数据成本
    SKILL = "skill"              # 技能要求成本
    RISK = "risk"                # 风险成本


class RiskDimension(Enum):
    """风险维度枚举"""
    EXECUTION = "execution"      # 执行风险
    OUTCOME = "outcome"          # 结果风险
    DEPENDENCY = "dependency"    # 依赖风险
    OPPORTUNITY = "opportunity"  # 机会成本风险


class FeasibilityDimension(Enum):
    """可行性维度枚举"""
    SKILL_MATCH = "skill_match"      # 技能匹配度
    RESOURCE_AVAILABILITY = "resource_availability"  # 资源可用性
    ENVIRONMENT_SUPPORT = "environment_support"  # 环境支持度
    TIME_FEASIBILITY = "time_feasibility"  # 时间可行性


class EvaluationMethod(Enum):
    """评估方法枚举"""
    MAUT = "maut"                # 多属性效用理论
    CBA = "cba"                  # 成本效益分析
    RISK_MATRIX = "risk_matrix"  # 风险矩阵分析
    FEASIBILITY_MATRIX = "feasibility_matrix"  # 可行性分析矩阵
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"  # 敏感性分析


@dataclass
class ValueAssessment:
    """价值评估数据类"""
    cognitive_value: float = 0.0    # 认知价值 (0.0-1.0)
    practical_value: float = 0.0    # 实用价值 (0.0-1.0)
    safety_value: float = 0.0       # 安全价值 (0.0-1.0)
    exploration_value: float = 0.0  # 探索价值 (0.0-1.0)
    
    # 详细评估结果
    details: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5         # 评估置信度 (0.0-1.0)
    evaluation_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """后初始化验证"""
        self.cognitive_value = max(0.0, min(1.0, self.cognitive_value))
        self.practical_value = max(0.0, min(1.0, self.practical_value))
        self.safety_value = max(0.0, min(1.0, self.safety_value))
        self.exploration_value = max(0.0, min(1.0, self.exploration_value))
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    @property
    def total_value(self) -> float:
        """计算总价值（加权平均）"""
        # 权重配置
        weights = {
            'cognitive': 0.3,
            'practical': 0.4,
            'safety': 0.2,
            'exploration': 0.1
        }
        
        total = (
            self.cognitive_value * weights['cognitive'] +
            self.practical_value * weights['practical'] +
            self.safety_value * weights['safety'] +
            self.exploration_value * weights['exploration']
        )
        
        return min(1.0, total)
    
    def get_dimension_value(self, dimension: ValueDimension) -> float:
        """获取指定维度的价值"""
        dimension_map = {
            ValueDimension.COGNITIVE: self.cognitive_value,
            ValueDimension.PRACTICAL: self.practical_value,
            ValueDimension.SAFETY: self.safety_value,
            ValueDimension.EXPLORATION: self.exploration_value
        }
        
        return dimension_map.get(dimension, 0.0)


@dataclass
class CostAssessment:
    """成本评估数据类"""
    time_cost: float = 0.0          # 时间成本 (0.0-1.0)
    computational_cost: float = 0.0  # 计算资源成本 (0.0-1.0)
    data_cost: float = 0.0          # 数据成本 (0.0-1.0)
    skill_cost: float = 0.0         # 技能要求成本 (0.0-1.0)
    risk_cost: float = 0.0          # 风险成本 (0.0-1.0)
    
    # 详细评估结果
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5         # 评估置信度 (0.0-1.0)
    evaluation_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """后初始化验证"""
        self.time_cost = max(0.0, min(1.0, self.time_cost))
        self.computational_cost = max(0.0, min(1.0, self.computational_cost))
        self.data_cost = max(0.0, min(1.0, self.data_cost))
        self.skill_cost = max(0.0, min(1.0, self.skill_cost))
        self.risk_cost = max(0.0, min(1.0, self.risk_cost))
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    @property
    def total_cost(self) -> float:
        """计算总成本（加权平均）"""
        # 权重配置
        weights = {
            'time': 0.3,
            'computational': 0.25,
            'data': 0.2,
            'skill': 0.15,
            'risk': 0.1
        }
        
        total = (
            self.time_cost * weights['time'] +
            self.computational_cost * weights['computational'] +
            self.data_cost * weights['data'] +
            self.skill_cost * weights['skill'] +
            self.risk_cost * weights['risk']
        )
        
        return min(1.0, total)
    
    def get_dimension_cost(self, dimension: CostDimension) -> float:
        """获取指定维度的成本"""
        dimension_map = {
            CostDimension.TIME: self.time_cost,
            CostDimension.COMPUTATIONAL: self.computational_cost,
            CostDimension.DATA: self.data_cost,
            CostDimension.SKILL: self.skill_cost,
            CostDimension.RISK: self.risk_cost
        }
        
        return dimension_map.get(dimension, 0.0)


@dataclass
class RiskAssessment:
    """风险评估数据类"""
    execution_risk: float = 0.0     # 执行风险 (0.0-1.0)
    outcome_risk: float = 0.0       # 结果风险 (0.0-1.0)
    dependency_risk: float = 0.0    # 依赖风险 (0.0-1.0)
    opportunity_risk: float = 0.0   # 机会成本风险 (0.0-1.0)
    
    # 风险详情
    risk_details: Dict[str, Any] = field(default_factory=dict)
    mitigation_strategies: List[str] = field(default_factory=list)
    confidence: float = 0.5         # 评估置信度 (0.0-1.0)
    evaluation_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """后初始化验证"""
        self.execution_risk = max(0.0, min(1.0, self.execution_risk))
        self.outcome_risk = max(0.0, min(1.0, self.outcome_risk))
        self.dependency_risk = max(0.0, min(1.0, self.dependency_risk))
        self.opportunity_risk = max(0.0, min(1.0, self.opportunity_risk))
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    @property
    def total_risk(self) -> float:
        """计算总风险（加权平均）"""
        # 权重配置
        weights = {
            'execution': 0.4,
            'outcome': 0.3,
            'dependency': 0.2,
            'opportunity': 0.1
        }
        
        total = (
            self.execution_risk * weights['execution'] +
            self.outcome_risk * weights['outcome'] +
            self.dependency_risk * weights['dependency'] +
            self.opportunity_risk * weights['opportunity']
        )
        
        return min(1.0, total)
    
    @property
    def risk_level(self) -> str:
        """获取风险等级"""
        total = self.total_risk
        
        if total >= 0.8:
            return "极高风险"
        elif total >= 0.6:
            return "高风险"
        elif total >= 0.4:
            return "中等风险"
        elif total >= 0.2:
            return "低风险"
        else:
            return "极低风险"
    
    def get_dimension_risk(self, dimension: RiskDimension) -> float:
        """获取指定维度的风险"""
        dimension_map = {
            RiskDimension.EXECUTION: self.execution_risk,
            RiskDimension.OUTCOME: self.outcome_risk,
            RiskDimension.DEPENDENCY: self.dependency_risk,
            RiskDimension.OPPORTUNITY: self.opportunity_risk
        }
        
        return dimension_map.get(dimension, 0.0)


@dataclass
class FeasibilityAssessment:
    """可行性评估数据类"""
    skill_match: float = 0.0         # 技能匹配度 (0.0-1.0)
    resource_availability: float = 0.0  # 资源可用性 (0.0-1.0)
    environment_support: float = 0.0  # 环境支持度 (0.0-1.0)
    time_feasibility: float = 0.0    # 时间可行性 (0.0-1.0)
    
    # 可行性详情
    feasibility_details: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    enablers: List[str] = field(default_factory=list)
    confidence: float = 0.5          # 评估置信度 (0.0-1.0)
    evaluation_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """后初始化验证"""
        self.skill_match = max(0.0, min(1.0, self.skill_match))
        self.resource_availability = max(0.0, min(1.0, self.resource_availability))
        self.environment_support = max(0.0, min(1.0, self.environment_support))
        self.time_feasibility = max(0.0, min(1.0, self.time_feasibility))
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    @property
    def total_feasibility(self) -> float:
        """计算总可行性（加权平均）"""
        # 权重配置
        weights = {
            'skill_match': 0.3,
            'resource_availability': 0.3,
            'environment_support': 0.2,
            'time_feasibility': 0.2
        }
        
        total = (
            self.skill_match * weights['skill_match'] +
            self.resource_availability * weights['resource_availability'] +
            self.environment_support * weights['environment_support'] +
            self.time_feasibility * weights['time_feasibility']
        )
        
        return min(1.0, total)
    
    @property
    def feasibility_level(self) -> str:
        """获取可行性等级"""
        total = self.total_feasibility
        
        if total >= 0.8:
            return "极高可行性"
        elif total >= 0.6:
            return "高可行性"
        elif total >= 0.4:
            return "中等可行性"
        elif total >= 0.2:
            return "低可行性"
        else:
            return "极低可行性"
    
    def get_dimension_feasibility(self, dimension: FeasibilityDimension) -> float:
        """获取指定维度的可行性"""
        dimension_map = {
            FeasibilityDimension.SKILL_MATCH: self.skill_match,
            FeasibilityDimension.RESOURCE_AVAILABILITY: self.resource_availability,
            FeasibilityDimension.ENVIRONMENT_SUPPORT: self.environment_support,
            FeasibilityDimension.TIME_FEASIBILITY: self.time_feasibility
        }
        
        return dimension_map.get(dimension, 0.0)


@dataclass
class GoalEvaluationResult:
    """目标评估结果数据类"""
    goal_id: str
    value_assessment: ValueAssessment
    cost_assessment: CostAssessment
    risk_assessment: RiskAssessment
    feasibility_assessment: FeasibilityAssessment
    
    # 综合评分
    overall_score: float = 0.0
    evaluation_method: EvaluationMethod = EvaluationMethod.MAUT
    evaluation_time: float = field(default_factory=time.time)
    evaluator_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.overall_score = max(0.0, min(1.0, self.overall_score))
    
    @property
    def cost_benefit_ratio(self) -> float:
        """计算成本效益比（价值/成本）"""
        total_value = self.value_assessment.total_value
        total_cost = self.cost_assessment.total_cost
        
        if total_cost <= 0:
            return float('inf') if total_value > 0 else 1.0
        
        return total_value / total_cost
    
    @property
    def risk_adjusted_value(self) -> float:
        """计算风险调整后价值"""
        total_value = self.value_assessment.total_value
        total_risk = self.risk_assessment.total_risk
        
        # 风险调整：价值 * (1 - 风险)
        return total_value * (1.0 - total_risk)
    
    @property
    def feasibility_adjusted_value(self) -> float:
        """计算可行性调整后价值"""
        total_value = self.value_assessment.total_value
        total_feasibility = self.feasibility_assessment.total_feasibility
        
        # 可行性调整：价值 * 可行性
        return total_value * total_feasibility
    
    @property
    def final_score(self) -> float:
        """计算最终综合评分"""
        # 综合价值、成本、风险、可行性
        value_weight = 0.4
        cost_weight = 0.2
        risk_weight = 0.2
        feasibility_weight = 0.2
        
        # 基础价值
        base_value = self.value_assessment.total_value
        
        # 成本调整（成本越低越好）
        cost_factor = 1.0 - self.cost_assessment.total_cost
        
        # 风险调整（风险越低越好）
        risk_factor = 1.0 - self.risk_assessment.total_risk
        
        # 可行性调整（可行性越高越好）
        feasibility_factor = self.feasibility_assessment.total_feasibility
        
        # 综合评分
        final_score = (
            base_value * value_weight +
            cost_factor * cost_weight +
            risk_factor * risk_weight +
            feasibility_factor * feasibility_weight
        )
        
        return min(1.0, final_score)
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        return {
            'goal_id': self.goal_id,
            'overall_score': self.overall_score,
            'final_score': self.final_score,
            'total_value': self.value_assessment.total_value,
            'total_cost': self.cost_assessment.total_cost,
            'total_risk': self.risk_assessment.total_risk,
            'risk_level': self.risk_assessment.risk_level,
            'total_feasibility': self.feasibility_assessment.total_feasibility,
            'feasibility_level': self.feasibility_assessment.feasibility_level,
            'cost_benefit_ratio': self.cost_benefit_ratio,
            'risk_adjusted_value': self.risk_adjusted_value,
            'feasibility_adjusted_value': self.feasibility_adjusted_value,
            'evaluation_method': self.evaluation_method.value,
            'evaluation_time': self.evaluation_time,
            'recommendation': self.get_recommendation()
        }
    
    def get_recommendation(self) -> str:
        """获取推荐建议"""
        final_score = self.final_score
        
        if final_score >= 0.8:
            return "强烈推荐：高价值、低风险、高可行性的目标"
        elif final_score >= 0.6:
            return "推荐执行：具有良好价值效益比的目标"
        elif final_score >= 0.4:
            return "谨慎考虑：需要进一步优化或风险控制的目标"
        elif final_score >= 0.2:
            return "不推荐：价值较低或风险较高的目标"
        else:
            return "强烈不推荐：价值低、风险高、可行性差的目标"


class GoalValueEvaluator:
    """
    目标价值评估系统
    
    核心组件:
    1. 价值评估器: 评估目标的认知、实用、安全、探索价值
    2. 成本评估器: 评估目标的时间、资源、技能、风险成本
    3. 风险评估器: 识别和量化目标的执行和结果风险
    4. 可行性评估器: 评估目标在当前条件下的可行性
    5. 综合评分器: 综合多个维度计算目标总体评分
    6. 敏感性分析器: 分析评估结果对参数变化的敏感性
    
    评估流程:
    目标描述 → 价值评估器 → 价值维度评分 → 成本评估器 → 成本维度评分
    风险识别 → 风险评估器 → 风险维度评分 → 可行性评估器 → 可行性评分
    综合评分器 → 综合评分 → 敏感性分析器 → 敏感性分析 → 评估报告
    
    技术特性:
    - 多维度综合评估: 考虑价值、成本、风险、可行性多个维度
    - 灵活权重配置: 支持根据不同场景调整评估权重
    - 敏感性分析: 分析评估结果对参数变化的敏感程度
    - 可解释性评估: 提供详细的评估理由和依据
    - 持续改进: 从实际执行结果中学习改进评估准确性
    
    应用场景:
    1. 自主目标选择: 为AGI系统选择最有价值的目标
    2. 资源分配决策: 基于目标价值分配有限资源
    3. 风险评估与管理: 识别和管理目标执行风险
    4. 目标优化: 基于评估结果优化目标设计和执行计划
    """
    
    def __init__(self,
                 evaluation_method: EvaluationMethod = EvaluationMethod.MAUT,
                 enable_sensitivity_analysis: bool = True,
                 learning_enabled: bool = True):
        """
        初始化目标价值评估系统
        
        Args:
            evaluation_method: 主要评估方法
            enable_sensitivity_analysis: 启用敏感性分析
            learning_enabled: 启用从经验中学习
        """
        self.evaluation_method = evaluation_method
        self.enable_sensitivity_analysis = enable_sensitivity_analysis
        self.learning_enabled = learning_enabled
        
        # 权重配置
        self.value_weights = {
            'cognitive': 0.3,
            'practical': 0.4,
            'safety': 0.2,
            'exploration': 0.1
        }
        
        self.cost_weights = {
            'time': 0.3,
            'computational': 0.25,
            'data': 0.2,
            'skill': 0.15,
            'risk': 0.1
        }
        
        self.risk_weights = {
            'execution': 0.4,
            'outcome': 0.3,
            'dependency': 0.2,
            'opportunity': 0.1
        }
        
        self.feasibility_weights = {
            'skill_match': 0.3,
            'resource_availability': 0.3,
            'environment_support': 0.2,
            'time_feasibility': 0.2
        }
        
        # 综合评分权重
        self.final_score_weights = {
            'value': 0.4,
            'cost': 0.2,
            'risk': 0.2,
            'feasibility': 0.2
        }
        
        # 评估模板库
        self.evaluation_templates = self._initialize_templates()
        
        # 学习数据
        self.learning_data: List[Dict[str, Any]] = []
        self.max_learning_data = 1000
        
        # 性能统计
        self.performance_stats = {
            'total_evaluations': 0,
            'average_evaluation_time': 0.0,
            'evaluation_time_total': 0.0,
            'sensitivity_analyses_performed': 0,
            'learning_updates': 0
        }
        
        logger.info(f"目标价值评估系统初始化完成，评估方法: {evaluation_method.value}")
    
    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """初始化评估模板库"""
        templates = {
            'learning_goal': {
                'value_weights': {'cognitive': 0.5, 'practical': 0.2, 'safety': 0.1, 'exploration': 0.2},
                'cost_weights': {'time': 0.4, 'computational': 0.2, 'data': 0.2, 'skill': 0.1, 'risk': 0.1},
                'description': '学习型目标评估模板'
            },
            'exploration_goal': {
                'value_weights': {'cognitive': 0.3, 'practical': 0.1, 'safety': 0.2, 'exploration': 0.4},
                'cost_weights': {'time': 0.3, 'computational': 0.3, 'data': 0.2, 'skill': 0.1, 'risk': 0.1},
                'description': '探索型目标评估模板'
            },
            'optimization_goal': {
                'value_weights': {'cognitive': 0.2, 'practical': 0.6, 'safety': 0.1, 'exploration': 0.1},
                'cost_weights': {'time': 0.3, 'computational': 0.3, 'data': 0.1, 'skill': 0.2, 'risk': 0.1},
                'description': '优化型目标评估模板'
            },
            'safety_goal': {
                'value_weights': {'cognitive': 0.1, 'practical': 0.2, 'safety': 0.6, 'exploration': 0.1},
                'cost_weights': {'time': 0.2, 'computational': 0.2, 'data': 0.2, 'skill': 0.2, 'risk': 0.2},
                'description': '安全型目标评估模板'
            }
        }
        
        return templates
    
    def evaluate_goal(self,
                     goal_description: str,
                     goal_type: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None,
                     detailed: bool = True) -> GoalEvaluationResult:
        """
        评估目标
        
        Args:
            goal_description: 目标描述
            goal_type: 目标类型（用于选择评估模板）
            context: 评估上下文信息
            detailed: 是否进行详细评估
            
        Returns:
            目标评估结果
        """
        start_time = time.time()
        
        # 获取评估模板
        template = self._get_evaluation_template(goal_type)
        
        # 执行价值评估
        value_assessment = self._assess_value(goal_description, template, context)
        
        # 执行成本评估
        cost_assessment = self._assess_cost(goal_description, template, context)
        
        # 执行风险评估
        risk_assessment = self._assess_risk(goal_description, context)
        
        # 执行可行性评估
        feasibility_assessment = self._assess_feasibility(goal_description, context)
        
        # 生成目标ID
        goal_id = f"goal_{int(time.time())}_{(zlib.adler32(str(goal_description).encode('utf-8')) & 0xffffffff) % 10000}"
        
        # 创建评估结果
        result = GoalEvaluationResult(
            goal_id=goal_id,
            value_assessment=value_assessment,
            cost_assessment=cost_assessment,
            risk_assessment=risk_assessment,
            feasibility_assessment=feasibility_assessment,
            evaluation_method=self.evaluation_method,
            evaluator_version="1.0",
            metadata={
                'goal_description': goal_description,
                'goal_type': goal_type,
                'context': context,
                'template_used': template.get('description', 'default')
            }
        )
        
        # 设置综合评分
        result.overall_score = result.final_score
        
        # 执行敏感性分析（如果启用）
        if self.enable_sensitivity_analysis and detailed:
            sensitivity_results = self._perform_sensitivity_analysis(result)
            result.metadata['sensitivity_analysis'] = sensitivity_results
        
        # 记录学习数据（如果启用学习）
        if self.learning_enabled:
            self._record_evaluation_for_learning(result, goal_description, context)
        
        # 更新性能统计
        evaluation_time = time.time() - start_time
        self._update_performance_stats(evaluation_time)
        
        logger.info(f"目标评估完成: {goal_id}，最终评分: {result.final_score:.2f}")
        
        return result
    
    def _get_evaluation_template(self, goal_type: Optional[str]) -> Dict[str, Any]:
        """获取评估模板"""
        if goal_type and goal_type in self.evaluation_templates:
            return self.evaluation_templates[goal_type]
        
        # 返回默认模板
        return {
            'value_weights': self.value_weights,
            'cost_weights': self.cost_weights,
            'description': '默认评估模板'
        }
    
    def _assess_value(self, 
                     goal_description: str,
                     template: Dict[str, Any],
                     context: Optional[Dict[str, Any]]) -> ValueAssessment:
        """评估目标价值"""
        # 简化实现：基于关键词和上下文进行价值评估
        # 实际实现应该使用更复杂的NLP和分析方法
        
        # 提取关键词
        keywords = self._extract_keywords(goal_description)
        
        # 初始化价值分数
        cognitive_value = 0.0
        practical_value = 0.0
        safety_value = 0.0
        exploration_value = 0.0
        
        # 认知价值关键词
        cognitive_keywords = ['学习', '理解', '分析', '研究', '知识', '概念']
        for keyword in cognitive_keywords:
            if keyword in goal_description:
                cognitive_value += 0.2
        
        # 实用价值关键词
        practical_keywords = ['解决', '优化', '提高', '改进', '效率', '性能']
        for keyword in practical_keywords:
            if keyword in goal_description:
                practical_value += 0.2
        
        # 安全价值关键词
        safety_keywords = ['安全', '保护', '可靠', '稳定', '风险', '预防']
        for keyword in safety_keywords:
            if keyword in goal_description:
                safety_value += 0.2
        
        # 探索价值关键词
        exploration_keywords = ['探索', '发现', '尝试', '实验', '新', '未知']
        for keyword in exploration_keywords:
            if keyword in goal_description:
                exploration_value += 0.2
        
        # 考虑上下文影响
        if context:
            # 上下文可能提供额外信息影响价值评估
            if context.get('urgency', False):
                practical_value = min(1.0, practical_value + 0.1)
            
            if context.get('knowledge_gap', False):
                cognitive_value = min(1.0, cognitive_value + 0.15)
        
        # 应用模板权重
        value_weights = template.get('value_weights', self.value_weights)
        
        # 限制在0-1范围内
        cognitive_value = max(0.0, min(1.0, cognitive_value))
        practical_value = max(0.0, min(1.0, practical_value))
        safety_value = max(0.0, min(1.0, safety_value))
        exploration_value = max(0.0, min(1.0, exploration_value))
        
        # 创建价值评估结果
        assessment = ValueAssessment(
            cognitive_value=cognitive_value,
            practical_value=practical_value,
            safety_value=safety_value,
            exploration_value=exploration_value,
            confidence=0.7,  # 中等置信度
            details={
                'keywords_found': keywords,
                'cognitive_keywords_matched': sum(1 for k in cognitive_keywords if k in goal_description),
                'practical_keywords_matched': sum(1 for k in practical_keywords if k in goal_description),
                'safety_keywords_matched': sum(1 for k in safety_keywords if k in goal_description),
                'exploration_keywords_matched': sum(1 for k in exploration_keywords if k in goal_description)
            }
        )
        
        return assessment
    
    def _assess_cost(self,
                    goal_description: str,
                    template: Dict[str, Any],
                    context: Optional[Dict[str, Any]]) -> CostAssessment:
        """评估目标成本"""
        # 简化实现：基于目标复杂性和上下文进行成本评估
        
        # 分析目标复杂性
        complexity_score = self._estimate_complexity(goal_description)
        
        # 基于复杂性的成本估算
        time_cost = complexity_score * 0.8  # 时间成本与复杂性正相关
        computational_cost = complexity_score * 0.6  # 计算成本
        data_cost = complexity_score * 0.4  # 数据需求成本
        skill_cost = complexity_score * 0.5  # 技能要求成本
        risk_cost = complexity_score * 0.3  # 基础风险成本
        
        # 考虑上下文影响
        if context:
            # 资源可用性影响
            resource_availability = context.get('resource_availability', 1.0)
            computational_cost = computational_cost * (1.0 / resource_availability)
            
            # 技能水平影响
            skill_level = context.get('skill_level', 0.5)
            skill_cost = skill_cost * (1.0 - skill_level)
        
        # 应用模板权重
        cost_weights = template.get('cost_weights', self.cost_weights)
        
        # 限制在0-1范围内
        time_cost = max(0.0, min(1.0, time_cost))
        computational_cost = max(0.0, min(1.0, computational_cost))
        data_cost = max(0.0, min(1.0, data_cost))
        skill_cost = max(0.0, min(1.0, skill_cost))
        risk_cost = max(0.0, min(1.0, risk_cost))
        
        # 创建成本评估结果
        assessment = CostAssessment(
            time_cost=time_cost,
            computational_cost=computational_cost,
            data_cost=data_cost,
            skill_cost=skill_cost,
            risk_cost=risk_cost,
            confidence=0.6,  # 中等置信度
            details={
                'complexity_score': complexity_score,
                'estimated_complexity': self._describe_complexity(complexity_score)
            }
        )
        
        return assessment
    
    def _assess_risk(self,
                    goal_description: str,
                    context: Optional[Dict[str, Any]]) -> RiskAssessment:
        """评估目标风险"""
        # 简化实现：基于目标性质和上下文进行风险评估
        
        # 分析风险因素
        execution_risk = 0.0
        outcome_risk = 0.0
        dependency_risk = 0.0
        opportunity_risk = 0.0
        
        # 执行风险：基于目标复杂性
        complexity = self._estimate_complexity(goal_description)
        execution_risk = complexity * 0.7
        
        # 结果风险：基于潜在负面影响
        negative_keywords = ['删除', '破坏', '修改', '改变', '风险', '危险']
        negative_count = sum(1 for keyword in negative_keywords if keyword in goal_description)
        outcome_risk = negative_count * 0.15
        
        # 依赖风险：基于外部依赖
        dependency_keywords = ['依赖', '需要', '要求', '基于', '使用', '借助']
        dependency_count = sum(1 for keyword in dependency_keywords if keyword in goal_description)
        dependency_risk = dependency_count * 0.1
        
        # 机会成本风险：基于时间估计
        time_estimate = self._estimate_time_requirement(goal_description)
        opportunity_risk = min(0.5, time_estimate * 0.2)
        
        # 考虑上下文影响
        if context:
            # 环境稳定性影响
            environment_stability = context.get('environment_stability', 0.5)
            execution_risk = execution_risk * (1.0 - environment_stability * 0.5)
            
            # 经验水平影响
            experience_level = context.get('experience_level', 0.5)
            outcome_risk = outcome_risk * (1.0 - experience_level * 0.3)
        
        # 限制在0-1范围内
        execution_risk = max(0.0, min(1.0, execution_risk))
        outcome_risk = max(0.0, min(1.0, outcome_risk))
        dependency_risk = max(0.0, min(1.0, dependency_risk))
        opportunity_risk = max(0.0, min(1.0, opportunity_risk))
        
        # 创建风险评估结果
        assessment = RiskAssessment(
            execution_risk=execution_risk,
            outcome_risk=outcome_risk,
            dependency_risk=dependency_risk,
            opportunity_risk=opportunity_risk,
            confidence=0.65,  # 中等置信度
            risk_details={
                'negative_keywords_found': negative_count,
                'dependency_keywords_found': dependency_count,
                'complexity_based_execution_risk': complexity * 0.7
            },
            mitigation_strategies=self._suggest_risk_mitigations(goal_description)
        )
        
        return assessment
    
    def _assess_feasibility(self,
                           goal_description: str,
                           context: Optional[Dict[str, Any]]) -> FeasibilityAssessment:
        """评估目标可行性"""
        # 简化实现：基于资源、技能和上下文进行可行性评估
        
        # 初始化可行性分数
        skill_match = 0.5  # 默认中等技能匹配
        resource_availability = 0.5  # 默认中等资源可用性
        environment_support = 0.5  # 默认中等环境支持
        time_feasibility = 0.5  # 默认中等时间可行性
        
        # 分析技能要求
        skill_keywords = ['高级', '复杂', '专业', '专家', '熟练', '精通']
        skill_demand = sum(1 for keyword in skill_keywords if keyword in goal_description)
        
        if skill_demand == 0:
            skill_match = 0.8  # 低技能要求，高匹配度
        elif skill_demand <= 2:
            skill_match = 0.5  # 中等技能要求
        else:
            skill_match = 0.3  # 高技能要求，低匹配度
        
        # 分析资源需求
        resource_keywords = ['大量', '丰富', '充足', '复杂', '多样', '特殊']
        resource_demand = sum(1 for keyword in resource_keywords if keyword in goal_description)
        
        if resource_demand == 0:
            resource_availability = 0.8  # 低资源需求
        elif resource_demand <= 2:
            resource_availability = 0.5  # 中等资源需求
        else:
            resource_availability = 0.3  # 高资源需求
        
        # 时间可行性分析
        time_estimate = self._estimate_time_requirement(goal_description)
        
        if time_estimate < 0.3:
            time_feasibility = 0.8  # 短期目标，高可行性
        elif time_estimate < 0.7:
            time_feasibility = 0.5  # 中期目标，中等可行性
        else:
            time_feasibility = 0.3  # 长期目标，低可行性
        
        # 考虑上下文影响
        if context:
            # 当前技能水平
            current_skill = context.get('current_skill_level', 0.5)
            skill_match = min(1.0, skill_match * (1.0 + current_skill * 0.5))
            
            # 资源状态
            resource_status = context.get('resource_status', 0.5)
            resource_availability = min(1.0, resource_availability * (1.0 + resource_status * 0.5))
            
            # 环境支持度
            environment_support_level = context.get('environment_support', 0.5)
            environment_support = environment_support_level
        
        # 限制在0-1范围内
        skill_match = max(0.0, min(1.0, skill_match))
        resource_availability = max(0.0, min(1.0, resource_availability))
        environment_support = max(0.0, min(1.0, environment_support))
        time_feasibility = max(0.0, min(1.0, time_feasibility))
        
        # 创建可行性评估结果
        assessment = FeasibilityAssessment(
            skill_match=skill_match,
            resource_availability=resource_availability,
            environment_support=environment_support,
            time_feasibility=time_feasibility,
            confidence=0.7,  # 中等置信度
            feasibility_details={
                'skill_demand_level': skill_demand,
                'resource_demand_level': resource_demand,
                'time_estimate': time_estimate
            },
            constraints=self._identify_constraints(goal_description),
            enablers=self._identify_enablers(goal_description)
        )
        
        return assessment
    
    def _perform_sensitivity_analysis(self, result: GoalEvaluationResult) -> Dict[str, Any]:
        """执行敏感性分析"""
        sensitivity_results = {
            'parameter_sensitivity': {},
            'robustness_score': 0.0,
            'critical_parameters': []
        }
        
        # 分析价值权重敏感性
        value_sensitivity = self._analyze_parameter_sensitivity(
            'value_weights',
            result.value_assessment.total_value,
            self.value_weights
        )
        sensitivity_results['parameter_sensitivity']['value_weights'] = value_sensitivity
        
        # 分析成本权重敏感性
        cost_sensitivity = self._analyze_parameter_sensitivity(
            'cost_weights',
            result.cost_assessment.total_cost,
            self.cost_weights
        )
        sensitivity_results['parameter_sensitivity']['cost_weights'] = cost_sensitivity
        
        # 分析风险权重敏感性
        risk_sensitivity = self._analyze_parameter_sensitivity(
            'risk_weights',
            result.risk_assessment.total_risk,
            self.risk_weights
        )
        sensitivity_results['parameter_sensitivity']['risk_weights'] = risk_sensitivity
        
        # 计算鲁棒性分数（对参数变化的敏感程度越低，鲁棒性越高）
        robustness = 1.0 - self._calculate_overall_sensitivity(sensitivity_results['parameter_sensitivity'])
        sensitivity_results['robustness_score'] = robustness
        
        # 识别关键参数（敏感性高的参数）
        critical_params = []
        for param_name, sensitivity in sensitivity_results['parameter_sensitivity'].items():
            if sensitivity.get('max_change', 0) > 0.2:  # 变化超过20%视为关键参数
                critical_params.append({
                    'parameter': param_name,
                    'sensitivity': sensitivity.get('max_change', 0),
                    'impact': 'high'
                })
        
        sensitivity_results['critical_parameters'] = critical_params
        
        # 更新性能统计
        self.performance_stats['sensitivity_analyses_performed'] += 1
        
        return sensitivity_results
    
    def _record_evaluation_for_learning(self,
                                      result: GoalEvaluationResult,
                                      goal_description: str,
                                      context: Optional[Dict[str, Any]]):
        """记录评估数据用于学习"""
        learning_record = {
            'goal_description': goal_description,
            'goal_id': result.goal_id,
            'evaluation_result': {
                'final_score': result.final_score,
                'total_value': result.value_assessment.total_value,
                'total_cost': result.cost_assessment.total_cost,
                'total_risk': result.risk_assessment.total_risk,
                'total_feasibility': result.feasibility_assessment.total_feasibility
            },
            'context': context or {},
            'evaluation_time': result.evaluation_time,
            'evaluator_version': result.evaluator_version
        }
        
        # 添加到学习数据
        self.learning_data.append(learning_record)
        
        # 限制学习数据大小
        if len(self.learning_data) > self.max_learning_data:
            self.learning_data = self.learning_data[-self.max_learning_data:]
        
        self.performance_stats['learning_updates'] += 1
    
    def _update_performance_stats(self, evaluation_time: float):
        """更新性能统计"""
        self.performance_stats['total_evaluations'] += 1
        
        # 更新平均评估时间
        current_avg = self.performance_stats['average_evaluation_time']
        total_evaluations = self.performance_stats['total_evaluations']
        
        new_avg = (current_avg * (total_evaluations - 1) + evaluation_time) / total_evaluations
        self.performance_stats['average_evaluation_time'] = new_avg
        
        # 累计总评估时间
        self.performance_stats['evaluation_time_total'] += evaluation_time
    
    # 辅助方法
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简化实现：按空格分割并过滤常见词汇
        common_words = {'的', '在', '是', '有', '和', '或', '与', '了', '可能', '一些', '各种'}
        words = text.split()
        
        # 过滤短词和常见词
        keywords = [word for word in words if len(word) > 1 and word not in common_words]
        
        return keywords[:10]  # 返回前10个关键词
    
    def _estimate_complexity(self, goal_description: str) -> float:
        """估计目标复杂性"""
        # 基于描述长度和关键词数量
        length_factor = min(1.0, len(goal_description) / 100.0)
        
        # 复杂性关键词
        complexity_keywords = ['复杂', '困难', '挑战', '多步', '集成', '系统', '综合']
        keyword_count = sum(1 for keyword in complexity_keywords if keyword in goal_description)
        keyword_factor = min(1.0, keyword_count * 0.2)
        
        # 综合复杂性分数
        complexity = (length_factor * 0.4 + keyword_factor * 0.6)
        
        return max(0.0, min(1.0, complexity))
    
    def _estimate_time_requirement(self, goal_description: str) -> float:
        """估计时间需求"""
        # 基于描述和关键词
        time_keywords = ['长期', '短期', '快速', '立即', '尽快', '逐步']
        time_indicators = {
            '长期': 0.8,
            '短期': 0.3,
            '快速': 0.2,
            '立即': 0.1,
            '尽快': 0.4,
            '逐步': 0.6
        }
        
        # 检查时间关键词
        time_score = 0.5  # 默认中等时间需求
        
        for keyword, score in time_indicators.items():
            if keyword in goal_description:
                time_score = score
                break
        
        # 考虑复杂性
        complexity = self._estimate_complexity(goal_description)
        time_score = max(time_score, complexity * 0.7)  # 复杂性增加时间需求
        
        return max(0.0, min(1.0, time_score))
    
    def _describe_complexity(self, complexity_score: float) -> str:
        """描述复杂性"""
        if complexity_score >= 0.8:
            return "极高复杂性"
        elif complexity_score >= 0.6:
            return "高复杂性"
        elif complexity_score >= 0.4:
            return "中等复杂性"
        elif complexity_score >= 0.2:
            return "低复杂性"
        else:
            return "极低复杂性"
    
    def _suggest_risk_mitigations(self, goal_description: str) -> List[str]:
        """建议风险缓解策略"""
        mitigations = []
        
        # 基于常见风险类型建议缓解策略
        if '复杂' in goal_description or '困难' in goal_description:
            mitigations.append("将复杂目标分解为多个子目标")
            mitigations.append("增加测试和验证环节")
        
        if '依赖' in goal_description or '需要' in goal_description:
            mitigations.append("确保依赖资源的可用性和可靠性")
            mitigations.append("制定备用方案以应对依赖失败")
        
        if '风险' in goal_description or '危险' in goal_description:
            mitigations.append("实施严格的安全检查和监控")
            mitigations.append("进行小规模试点测试")
        
        # 通用缓解策略
        mitigations.append("定期评估进展并调整执行计划")
        mitigations.append("建立问题上报和解决机制")
        
        return mitigations
    
    def _identify_constraints(self, goal_description: str) -> List[str]:
        """识别约束条件"""
        constraints = []
        
        # 基于关键词识别潜在约束
        if '时间' in goal_description and ('紧张' in goal_description or '有限' in goal_description):
            constraints.append("时间约束：可用时间有限")
        
        if '资源' in goal_description and ('有限' in goal_description or '不足' in goal_description):
            constraints.append("资源约束：资源可用性有限")
        
        if '技能' in goal_description and ('缺乏' in goal_description or '不足' in goal_description):
            constraints.append("技能约束：所需技能可能不足")
        
        if '环境' in goal_description and ('限制' in goal_description or '约束' in goal_description):
            constraints.append("环境约束：执行环境可能有限制")
        
        return constraints
    
    def _identify_enablers(self, goal_description: str) -> List[str]:
        """识别使能因素"""
        enablers = []
        
        # 基于关键词识别潜在使能因素
        if '现有' in goal_description and ('知识' in goal_description or '经验' in goal_description):
            enablers.append("现有知识和经验积累")
        
        if '工具' in goal_description or '技术' in goal_description:
            enablers.append("可用工具和技术支持")
        
        if '团队' in goal_description or '合作' in goal_description:
            enablers.append("团队合作和协同工作")
        
        if '数据' in goal_description and ('可用' in goal_description or '丰富' in goal_description):
            enablers.append("数据可用性和丰富性")
        
        return enablers
    
    def _analyze_parameter_sensitivity(self,
                                     param_name: str,
                                     base_value: float,
                                     weights: Dict[str, float]) -> Dict[str, Any]:
        """分析参数敏感性"""
        sensitivity = {
            'parameter': param_name,
            'base_value': base_value,
            'variations': [],
            'max_change': 0.0,
            'average_change': 0.0
        }
        
        # 测试权重变化的影响
        variations = []
        
        for weight_name, weight_value in weights.items():
            # 测试增加20%
            increased_weight = min(1.0, weight_value * 1.2)
            temp_weights = weights.copy()
            temp_weights[weight_name] = increased_weight
            
            # 重新计算（简化实现）
            # 实际实现应该重新运行完整的评估
            adjusted_value = base_value * (increased_weight / weight_value)
            change_percent = abs(adjusted_value - base_value) / base_value if base_value > 0 else 0.0
            
            variations.append({
                'weight': weight_name,
                'change': '+20%',
                'adjusted_value': adjusted_value,
                'change_percent': change_percent
            })
            
            sensitivity['max_change'] = max(sensitivity['max_change'], change_percent)
        
        sensitivity['variations'] = variations
        
        if variations:
            avg_change = sum(v['change_percent'] for v in variations) / len(variations)
            sensitivity['average_change'] = avg_change
        
        return sensitivity
    
    def _calculate_overall_sensitivity(self, sensitivity_data: Dict[str, Any]) -> float:
        """计算总体敏感性"""
        if not sensitivity_data:
            return 0.0
        
        # 使用最大敏感性作为总体敏感性
        max_sensitivities = [data.get('max_change', 0.0) for data in sensitivity_data.values()]
        
        if max_sensitivities:
            return max(max_sensitivities)
        
        return 0.0
    
    def compare_goals(self,
                     goal_evaluations: List[GoalEvaluationResult],
                     comparison_dimensions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        比较多个目标
        
        Args:
            goal_evaluations: 目标评估结果列表
            comparison_dimensions: 比较维度（可选）
            
        Returns:
            比较结果
        """
        if not goal_evaluations:
            return {'error': '没有可比较的目标评估结果'}
        
        # 默认比较维度
        if comparison_dimensions is None:
            comparison_dimensions = ['final_score', 'total_value', 'total_cost', 
                                   'total_risk', 'total_feasibility', 'cost_benefit_ratio']
        
        comparison_results = {
            'compared_goals': [eval.goal_id for eval in goal_evaluations],
            'comparison_dimensions': comparison_dimensions,
            'rankings': {},
            'summary': {}
        }
        
        # 按各个维度排序
        for dimension in comparison_dimensions:
            # 获取该维度的值
            dimension_values = []
            
            for eval_result in goal_evaluations:
                if dimension == 'final_score':
                    value = eval_result.final_score
                elif dimension == 'total_value':
                    value = eval_result.value_assessment.total_value
                elif dimension == 'total_cost':
                    value = eval_result.cost_assessment.total_cost
                elif dimension == 'total_risk':
                    value = eval_result.risk_assessment.total_risk
                elif dimension == 'total_feasibility':
                    value = eval_result.feasibility_assessment.total_feasibility
                elif dimension == 'cost_benefit_ratio':
                    value = eval_result.cost_benefit_ratio
                else:
                    value = 0.0
                
                dimension_values.append((eval_result.goal_id, value))
            
            # 排序（对于成本和风险，值越低越好）
            if dimension in ['total_cost', 'total_risk']:
                dimension_values.sort(key=lambda x: x[1])  # 升序
            else:
                dimension_values.sort(key=lambda x: x[1], reverse=True)  # 降序
            
            comparison_results['rankings'][dimension] = [
                {'goal_id': goal_id, 'value': value, 'rank': i+1}
                for i, (goal_id, value) in enumerate(dimension_values)
            ]
        
        # 综合排名（基于最终评分）
        final_scores = [(eval.goal_id, eval.final_score) for eval in goal_evaluations]
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        comparison_results['summary']['overall_ranking'] = [
            {'goal_id': goal_id, 'final_score': score, 'rank': i+1}
            for i, (goal_id, score) in enumerate(final_scores)
        ]
        
        # 最佳目标推荐
        if final_scores:
            best_goal_id, best_score = final_scores[0]
            comparison_results['summary']['recommended_goal'] = {
                'goal_id': best_goal_id,
                'final_score': best_score,
                'reason': f"综合评分最高 ({best_score:.2f})"
            }
        
        return comparison_results
    
    def get_evaluator_statistics(self) -> Dict[str, Any]:
        """获取评估器统计信息"""
        return {
            'total_evaluations': self.performance_stats['total_evaluations'],
            'average_evaluation_time': self.performance_stats['average_evaluation_time'],
            'sensitivity_analyses_performed': self.performance_stats['sensitivity_analyses_performed'],
            'learning_updates': self.performance_stats['learning_updates'],
            'learning_data_size': len(self.learning_data),
            'evaluation_method': self.evaluation_method.value,
            'enable_sensitivity_analysis': self.enable_sensitivity_analysis,
            'learning_enabled': self.learning_enabled
        }


# 示例和测试函数
def create_example_evaluator() -> GoalValueEvaluator:
    """创建示例目标价值评估器"""
    evaluator = GoalValueEvaluator(
        evaluation_method=EvaluationMethod.MAUT,
        enable_sensitivity_analysis=True,
        learning_enabled=True
    )
    return evaluator


def test_goal_evaluator():
    """测试目标价值评估器"""
    logger.info("开始测试目标价值评估器")
    
    # 创建示例评估器
    evaluator = create_example_evaluator()
    
    # 测试目标
    test_goals = [
        "学习新的机器学习算法以提高预测准确性",
        "优化系统架构以减少资源消耗和提高性能",
        "探索新的数据可视化技术以增强分析能力",
        "实施安全审计机制以降低系统风险"
    ]
    
    evaluation_results = []
    
    # 评估每个目标
    for i, goal in enumerate(test_goals):
        logger.info(f"评估目标 {i+1}: {goal}")
        
        # 创建评估上下文
        context = {
            'urgency': i == 0,  # 第一个目标设为紧急
            'resource_availability': 0.7,
            'skill_level': 0.6,
            'environment_stability': 0.8
        }
        
        # 执行评估
        result = evaluator.evaluate_goal(
            goal_description=goal,
            goal_type='learning_goal' if '学习' in goal else 'optimization_goal',
            context=context,
            detailed=True
        )
        
        evaluation_results.append(result)
        
        # 显示评估摘要
        summary = result.get_evaluation_summary()
        logger.info(f"  最终评分: {summary['final_score']:.2f}, 推荐: {summary['recommendation']}")
    
    # 比较目标
    logger.info("比较所有目标...")
    comparison = evaluator.compare_goals(evaluation_results)
    
    if 'summary' in comparison and 'recommended_goal' in comparison['summary']:
        rec = comparison['summary']['recommended_goal']
        logger.info(f"推荐目标: {rec['goal_id']}, 理由: {rec['reason']}")
    
    # 获取统计信息
    stats = evaluator.get_evaluator_statistics()
    logger.info(f"评估器统计: {stats['total_evaluations']} 次评估，平均时间: {stats['average_evaluation_time']:.2f}秒")
    
    logger.info("目标价值评估器测试完成")
    return evaluator


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_goal_evaluator()