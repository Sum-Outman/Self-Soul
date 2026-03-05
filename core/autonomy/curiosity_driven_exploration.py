#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
好奇心驱动探索 - 实现基于新奇性、不确定性和信息增益的探索行为

核心功能:
1. 新奇性检测: 识别环境中新奇或意外的元素
2. 不确定性量化: 估计模型对不同情况的预测不确定性
3. 信息增益计算: 计算探索不同选项可能获得的信息量
4. 探索策略优化: 在探索和利用之间找到最优平衡
5. 好奇心衰减建模: 模拟好奇心的饱和和衰减过程

好奇心类型:
1. 感知好奇心 (Perceptual Curiosity):
   - 对新奇感官输入的追求
   - 对意外、异常、不匹配模式的关注
   - 基于预测误差的激活

2. 认知好奇心 (Cognitive Curiosity):
   - 对知识缺口和认知冲突的追求
   - 对解释不一致性、逻辑矛盾的关注
   - 基于知识不完整性的激活

3. 多样性好奇心 (Diversity Seeking):
   - 对多样性和变化的追求
   - 避免重复和单调，寻求新颖体验
   - 基于熵和多样性的激活

好奇心驱动机制:
1. 预测误差信号: 实际观察与预期之间的差异
2. 学习进度信号: 学习新知识的速率和效率
3. 知识缺口信号: 当前知识体系中的空白和不一致
4. 新奇性信号: 输入与已知模式的差异程度

探索策略:
1. 基于不确定性的探索: 优先探索不确定性高的区域
2. 基于新奇性的探索: 优先探索新奇性高的区域
3. 基于信息增益的探索: 优先探索信息增益大的区域
4. 基于多样性的探索: 确保探索的多样性和覆盖度

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


class CuriosityType(Enum):
    """好奇心类型枚举"""
    PERCEPTUAL = "perceptual"      # 感知好奇心
    COGNITIVE = "cognitive"        # 认知好奇心
    DIVERSITY = "diversity"        # 多样性好奇心


class ExplorationStrategy(Enum):
    """探索策略枚举"""
    UNCERTAINTY_BASED = "uncertainty_based"  # 基于不确定性
    NOVELTY_BASED = "novelty_based"          # 基于新奇性
    INFORMATION_GAIN = "information_gain"    # 基于信息增益
    DIVERSITY_DRIVEN = "diversity_driven"    # 基于多样性


@dataclass
class NoveltyDetection:
    """新奇性检测数据类"""
    observation_id: str
    observation_data: Any
    novelty_score: float
    novelty_type: str
    prediction_error: float = 0.0
    familiarity_score: float = 0.0
    unexpectedness_score: float = 0.0
    detection_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.novelty_score = max(0.0, min(1.0, self.novelty_score))
        self.prediction_error = max(0.0, self.prediction_error)
        self.familiarity_score = max(0.0, min(1.0, self.familiarity_score))
        self.unexpectedness_score = max(0.0, min(1.0, self.unexpectedness_score))


@dataclass
class UncertaintyEstimation:
    """不确定性估计数据类"""
    state_id: str
    uncertainty_type: str
    uncertainty_value: float
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    estimation_method: str = "default"
    sample_size: int = 0
    estimation_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.uncertainty_value = max(0.0, min(1.0, self.uncertainty_value))
        # 确保置信区间有效
        lower, upper = self.confidence_interval
        if lower > upper:
            self.confidence_interval = (upper, lower)


@dataclass
class InformationGainEstimation:
    """信息增益估计数据类"""
    action_id: str
    state_id: str
    expected_information_gain: float
    entropy_reduction: float = 0.0
    mutual_information: float = 0.0
    kl_divergence: float = 0.0
    estimation_confidence: float = 0.8
    estimation_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.expected_information_gain = max(0.0, self.expected_information_gain)
        self.entropy_reduction = max(0.0, self.entropy_reduction)
        self.mutual_information = max(0.0, self.mutual_information)
        self.kl_divergence = max(0.0, self.kl_divergence)
        self.estimation_confidence = max(0.0, min(1.0, self.estimation_confidence))


@dataclass
class ExplorationOption:
    """探索选项数据类"""
    id: str
    description: str
    exploration_strategy: ExplorationStrategy
    expected_novelty: float = 0.0
    expected_uncertainty_reduction: float = 0.0
    expected_information_gain: float = 0.0
    diversity_contribution: float = 0.0
    exploration_cost: float = 0.0
    estimated_duration: float = 0.0
    success_probability: float = 0.5
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.expected_novelty = max(0.0, min(1.0, self.expected_novelty))
        self.expected_uncertainty_reduction = max(0.0, min(1.0, self.expected_uncertainty_reduction))
        self.expected_information_gain = max(0.0, self.expected_information_gain)
        self.diversity_contribution = max(0.0, min(1.0, self.diversity_contribution))
        self.exploration_cost = max(0.0, self.exploration_cost)
        self.estimated_duration = max(0.0, self.estimated_duration)
        self.success_probability = max(0.0, min(1.0, self.success_probability))
    
    @property
    def exploration_score(self) -> float:
        """计算探索分数"""
        # 权重可以根据策略调整
        weights = {
            ExplorationStrategy.UNCERTAINTY_BASED: (0.0, 0.6, 0.3, 0.1),
            ExplorationStrategy.NOVELTY_BASED: (0.6, 0.2, 0.1, 0.1),
            ExplorationStrategy.INFORMATION_GAIN: (0.1, 0.1, 0.7, 0.1),
            ExplorationStrategy.DIVERSITY_DRIVEN: (0.2, 0.1, 0.1, 0.6)
        }
        
        w_novelty, w_uncertainty, w_info, w_diversity = weights.get(
            self.exploration_strategy, (0.25, 0.25, 0.25, 0.25)
        )
        
        score = (
            self.expected_novelty * w_novelty +
            self.expected_uncertainty_reduction * w_uncertainty +
            self.expected_information_gain * w_info +
            self.diversity_contribution * w_diversity
        )
        
        # 考虑成本和成功概率
        cost_factor = 1.0 / (1.0 + self.exploration_cost * 0.1)
        success_factor = self.success_probability
        
        return score * cost_factor * success_factor


@dataclass
class CuriosityState:
    """好奇心状态数据类"""
    curiosity_type: CuriosityType
    activation_level: float = 0.5
    saturation_level: float = 0.0
    learning_progress: float = 0.0
    last_activation_time: float = field(default_factory=time.time)
    activation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.activation_level = max(0.0, min(1.0, self.activation_level))
        self.saturation_level = max(0.0, min(1.0, self.saturation_level))
        self.learning_progress = max(0.0, min(1.0, self.learning_progress))
    
    def update_activation(self, stimulation: float):
        """更新激活水平"""
        # 刺激增加激活，但受饱和限制
        effective_stimulation = stimulation * (1.0 - self.saturation_level)
        self.activation_level = max(0.0, min(1.0, 
            self.activation_level * 0.7 + effective_stimulation * 0.3))
        
        # 更新饱和水平（重复激活会增加饱和）
        self.saturation_level = max(0.0, min(1.0,
            self.saturation_level * 0.9 + effective_stimulation * 0.1))
        
        self.last_activation_time = time.time()
        self.activation_count += 1
    
    def decay(self, decay_rate: float = 0.95):
        """好奇心衰减"""
        self.activation_level *= decay_rate
        self.saturation_level *= decay_rate  # 饱和也会衰减


class CuriosityDrivenExploration:
    """
    好奇心驱动探索系统
    
    核心组件:
    1. 新奇性检测器: 检测环境中的新奇元素
    2. 不确定性估计器: 估计模型的不确定性
    3. 信息增益计算器: 计算探索的信息收益
    4. 多样性维护器: 确保探索的多样性
    5. 探索策略选择器: 选择最优探索策略
    6. 好奇心状态管理器: 管理好奇心激活和衰减
    
    工作流程:
    环境观测 → 新奇性检测器 → 检测新奇性 → 不确定性估计器 → 估计不确定性
    不确定性 + 新奇性 → 信息增益计算器 → 计算信息增益 → 探索策略选择器 → 选择策略
    选择策略 → 生成探索选项 → 好奇心状态管理器 → 更新好奇心状态
    
    技术特性:
    - 多维度新奇性检测
    - 贝叶斯不确定性量化
    - 信息论指导的探索
    - 自适应好奇心衰减
    - 探索-利用平衡优化
    """
    
    def __init__(self,
                 max_exploration_options: int = 20,
                 novelty_threshold: float = 0.3,
                 uncertainty_threshold: float = 0.4,
                 exploration_exploitation_ratio: float = 0.3,
                 curiosity_decay_rate: float = 0.97):
        """
        初始化好奇心驱动探索系统
        
        Args:
            max_exploration_options: 最大探索选项数量
            novelty_threshold: 新奇性阈值
            uncertainty_threshold: 不确定性阈值
            exploration_exploitation_ratio: 探索-利用比率
            curiosity_decay_rate: 好奇心衰减率
        """
        self.max_exploration_options = max_exploration_options
        self.novelty_threshold = novelty_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.exploration_exploitation_ratio = exploration_exploitation_ratio
        self.curiosity_decay_rate = curiosity_decay_rate
        
        # 好奇心状态管理
        self.curiosity_states: Dict[CuriosityType, CuriosityState] = {}
        self._initialize_curiosity_states()
        
        # 新奇性检测历史
        self.novelty_detections: List[NoveltyDetection] = []
        self.novelty_history: Dict[str, List[float]] = defaultdict(list)
        
        # 不确定性估计
        self.uncertainty_estimations: Dict[str, UncertaintyEstimation] = {}
        
        # 探索选项
        self.exploration_options: Dict[str, ExplorationOption] = {}
        
        # 配置参数
        self.config = {
            'novelty_decay_factor': 0.95,
            'uncertainty_update_rate': 0.1,
            'information_gain_weight': 0.4,
            'diversity_weight': 0.2,
            'cost_weight': 0.2,
            'success_probability_weight': 0.2,
            'max_novelty_history': 100,
            'exploration_strategy_weights': {
                ExplorationStrategy.UNCERTAINTY_BASED: 0.3,
                ExplorationStrategy.NOVELTY_BASED: 0.3,
                ExplorationStrategy.INFORMATION_GAIN: 0.2,
                ExplorationStrategy.DIVERSITY_DRIVEN: 0.2
            }
        }
        
        # 性能统计
        self.performance_stats = {
            'novelty_detections': 0,
            'uncertainty_estimations': 0,
            'exploration_options_generated': 0,
            'exploration_actions_taken': 0,
            'information_gain_achieved': 0.0,
            'uncertainty_reduction': 0.0,
            'average_novelty_score': 0.0
        }
        
        logger.info(f"好奇心驱动探索系统初始化完成，探索-利用比率: {exploration_exploitation_ratio}")
    
    def _initialize_curiosity_states(self):
        """初始化好奇心状态"""
        for curiosity_type in CuriosityType:
            self.curiosity_states[curiosity_type] = CuriosityState(
                curiosity_type=curiosity_type,
                activation_level=0.5,
                saturation_level=0.0,
                learning_progress=0.0
            )
    
    def detect_novelty(self,
                      observation_id: str,
                      observation_data: Any,
                      expected_value: Optional[Any] = None,
                      context: Optional[Dict[str, Any]] = None) -> NoveltyDetection:
        """
        检测新奇性
        
        Args:
            observation_id: 观测ID
            observation_data: 观测数据
            expected_value: 预期值（如果可用）
            context: 情境信息
            
        Returns:
            新奇性检测结果
        """
        # 计算新奇性分数
        novelty_score = self._compute_novelty_score(observation_data, expected_value, context)
        
        # 计算预测误差（如果有预期值）
        prediction_error = 0.0
        if expected_value is not None:
            prediction_error = self._compute_prediction_error(observation_data, expected_value)
        
        # 计算熟悉度分数（基于历史）
        familiarity_score = self._compute_familiarity_score(observation_id, observation_data)
        
        # 计算意外性分数
        unexpectedness_score = self._compute_unexpectedness_score(novelty_score, prediction_error)
        
        # 确定新奇类型
        novelty_type = self._determine_novelty_type(novelty_score, prediction_error, familiarity_score)
        
        # 创建新奇性检测记录
        detection = NoveltyDetection(
            observation_id=observation_id,
            observation_data=observation_data,
            novelty_score=novelty_score,
            novelty_type=novelty_type,
            prediction_error=prediction_error,
            familiarity_score=familiarity_score,
            unexpectedness_score=unexpectedness_score,
            metadata={
                'context': context or {},
                'detection_time': time.time()
            }
        )
        
        # 存储检测记录
        self.novelty_detections.append(detection)
        
        # 更新历史
        self.novelty_history[observation_id].append(novelty_score)
        if len(self.novelty_history[observation_id]) > self.config['max_novelty_history']:
            self.novelty_history[observation_id].pop(0)
        
        # 更新好奇心状态
        self._update_curiosity_from_novelty(detection)
        
        # 更新统计
        self.performance_stats['novelty_detections'] += 1
        self._update_average_novelty(novelty_score)
        
        logger.debug(f"新奇性检测: {observation_id}，分数: {novelty_score:.3f}，类型: {novelty_type}")
        
        return detection
    
    def _compute_novelty_score(self, 
                              observation_data: Any,
                              expected_value: Optional[Any],
                              context: Optional[Dict[str, Any]]) -> float:
        """计算新奇性分数"""
        # 简化实现：基于随机性和上下文
        base_novelty = random.uniform(0.0, 0.3)  # 基础新奇性
        
        # 如果有预期值，计算差异
        if expected_value is not None:
            difference = self._compute_difference(observation_data, expected_value)
            novelty_from_difference = min(1.0, difference * 2.0)
            base_novelty = max(base_novelty, novelty_from_difference)
        
        # 上下文影响
        if context and 'novelty_context' in context:
            context_factor = context.get('novelty_context', 0.5)
            base_novelty = min(1.0, base_novelty * (1.0 + context_factor))
        
        return min(1.0, base_novelty)
    
    def _compute_prediction_error(self, actual: Any, expected: Any) -> float:
        """计算预测误差"""
        # 简化实现
        try:
            if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                return abs(actual - expected) / max(abs(expected), 1.0)
            else:
                # 对于非数值，使用简单差异度量
                return 0.5 if str(actual) != str(expected) else 0.0
        except Exception:
            return 0.5
    
    def _compute_difference(self, value1: Any, value2: Any) -> float:
        """计算差异"""
        # 简化实现
        if value1 == value2:
            return 0.0
        else:
            return 0.5  # 中等差异
    
    def _compute_familiarity_score(self, observation_id: str, observation_data: Any) -> float:
        """计算熟悉度分数"""
        if observation_id not in self.novelty_history or not self.novelty_history[observation_id]:
            return 0.0
        
        # 基于历史新奇性分数计算熟悉度
        history = self.novelty_history[observation_id]
        avg_novelty = sum(history) / len(history)
        
        # 新奇性越低，熟悉度越高
        familiarity = 1.0 - min(1.0, avg_novelty * 1.5)
        
        # 考虑历史长度
        history_factor = min(1.0, len(history) / 10.0)
        familiarity = min(1.0, familiarity * (1.0 + history_factor * 0.3))
        
        return familiarity
    
    def _compute_unexpectedness_score(self, novelty_score: float, prediction_error: float) -> float:
        """计算意外性分数"""
        # 结合新奇性和预测误差
        unexpectedness = (novelty_score * 0.6 + prediction_error * 0.4)
        return min(1.0, unexpectedness)
    
    def _determine_novelty_type(self, 
                               novelty_score: float, 
                               prediction_error: float,
                               familiarity_score: float) -> str:
        """确定新奇类型"""
        if prediction_error > 0.7:
            return "high_prediction_error"
        elif novelty_score > 0.7:
            return "high_novelty"
        elif familiarity_score < 0.3:
            return "low_familiarity"
        else:
            return "moderate_novelty"
    
    def _update_curiosity_from_novelty(self, detection: NoveltyDetection):
        """从新奇性检测更新好奇心状态"""
        # 感知好奇心：基于新奇性和预测误差
        perceptual_stimulation = (detection.novelty_score * 0.6 + 
                                 detection.prediction_error * 0.4)
        self.curiosity_states[CuriosityType.PERCEPTUAL].update_activation(
            perceptual_stimulation
        )
        
        # 认知好奇心：基于意外性和新奇类型
        cognitive_stimulation = detection.unexpectedness_score * 0.8
        if detection.novelty_type == "high_prediction_error":
            cognitive_stimulation *= 1.2  # 预测错误更刺激认知好奇心
        
        self.curiosity_states[CuriosityType.COGNITIVE].update_activation(
            cognitive_stimulation
        )
        
        # 多样性好奇心：基于新奇性和熟悉度
        diversity_stimulation = (detection.novelty_score * 0.7 + 
                                (1.0 - detection.familiarity_score) * 0.3)
        self.curiosity_states[CuriosityType.DIVERSITY].update_activation(
            diversity_stimulation
        )
    
    def _update_average_novelty(self, new_novelty_score: float):
        """更新平均新奇性分数"""
        current_avg = self.performance_stats['average_novelty_score']
        n_detections = self.performance_stats['novelty_detections']
        
        if n_detections == 0:
            new_avg = new_novelty_score
        else:
            new_avg = (current_avg * (n_detections - 1) + new_novelty_score) / n_detections
        
        self.performance_stats['average_novelty_score'] = new_avg
    
    def estimate_uncertainty(self,
                            state_id: str,
                            uncertainty_type: str = "model_uncertainty",
                            current_estimate: Optional[float] = None,
                            new_evidence: Optional[Any] = None) -> UncertaintyEstimation:
        """
        估计不确定性
        
        Args:
            state_id: 状态ID
            uncertainty_type: 不确定性类型
            current_estimate: 当前估计值
            new_evidence: 新证据
            
        Returns:
            不确定性估计结果
        """
        # 计算不确定性值
        if current_estimate is not None:
            # 基于当前估计和证据更新
            uncertainty_value = self._update_uncertainty_estimate(
                current_estimate, new_evidence
            )
        else:
            # 初始估计
            uncertainty_value = self._compute_initial_uncertainty(state_id, uncertainty_type)
        
        # 计算置信区间
        confidence_interval = self._compute_confidence_interval(uncertainty_value)
        
        # 确定估计方法
        estimation_method = "bayesian_update" if current_estimate is not None else "initial"
        
        # 创建不确定性估计记录
        estimation = UncertaintyEstimation(
            state_id=state_id,
            uncertainty_type=uncertainty_type,
            uncertainty_value=uncertainty_value,
            confidence_interval=confidence_interval,
            estimation_method=estimation_method,
            sample_size=1 if new_evidence is not None else 0,
            metadata={
                'previous_estimate': current_estimate,
                'new_evidence': new_evidence
            }
        )
        
        # 存储估计
        self.uncertainty_estimations[state_id] = estimation
        
        # 更新统计
        self.performance_stats['uncertainty_estimations'] += 1
        
        logger.debug(f"不确定性估计: {state_id}，值: {uncertainty_value:.3f}，类型: {uncertainty_type}")
        
        return estimation
    
    def _update_uncertainty_estimate(self, 
                                   current_estimate: float,
                                   new_evidence: Any) -> float:
        """更新不确定性估计"""
        # 简化实现：基于证据质量降低不确定性
        evidence_quality = self._assess_evidence_quality(new_evidence)
        reduction_factor = 0.1 + evidence_quality * 0.3
        
        new_uncertainty = current_estimate * (1.0 - reduction_factor)
        return max(0.0, min(1.0, new_uncertainty))
    
    def _assess_evidence_quality(self, evidence: Any) -> float:
        """评估证据质量"""
        # 简化实现
        if evidence is None:
            return 0.0
        elif isinstance(evidence, (int, float)):
            # 数值证据：考虑精度
            return 0.7 if abs(evidence) > 0.1 else 0.3
        elif isinstance(evidence, str):
            # 文本证据：考虑长度和内容
            length = len(evidence)
            if length > 50:
                return 0.8
            elif length > 10:
                return 0.5
            else:
                return 0.2
        else:
            return 0.4
    
    def _compute_initial_uncertainty(self, state_id: str, uncertainty_type: str) -> float:
        """计算初始不确定性"""
        # 基于状态ID和类型
        if uncertainty_type == "model_uncertainty":
            # 模型不确定性通常较高
            return random.uniform(0.6, 0.9)
        elif uncertainty_type == "measurement_uncertainty":
            # 测量不确定性中等
            return random.uniform(0.3, 0.7)
        else:
            # 其他类型
            return random.uniform(0.4, 0.8)
    
    def _compute_confidence_interval(self, uncertainty_value: float) -> Tuple[float, float]:
        """计算置信区间"""
        # 不确定性越高，置信区间越宽
        interval_width = uncertainty_value * 0.5
        
        lower = max(0.0, uncertainty_value - interval_width / 2)
        upper = min(1.0, uncertainty_value + interval_width / 2)
        
        return (lower, upper)
    
    def generate_exploration_options(self,
                                   n_options: int = 10,
                                   strategy_weights: Optional[Dict[ExplorationStrategy, float]] = None) -> List[ExplorationOption]:
        """
        生成探索选项
        
        Args:
            n_options: 选项数量
            strategy_weights: 策略权重
            
        Returns:
            探索选项列表
        """
        if n_options > self.max_exploration_options:
            n_options = self.max_exploration_options
        
        options = []
        
        # 使用提供的权重或默认权重
        if strategy_weights is None:
            strategy_weights = self.config['exploration_strategy_weights']
        
        # 计算每种策略的选项数量
        strategy_counts = self._allocate_options_by_strategy(n_options, strategy_weights)
        
        # 为每种策略生成选项
        for strategy, count in strategy_counts.items():
            strategy_options = self._generate_strategy_options(strategy, count)
            options.extend(strategy_options)
        
        # 存储选项
        for option in options:
            self.exploration_options[option.id] = option
        
        # 更新统计
        self.performance_stats['exploration_options_generated'] += len(options)
        
        logger.info(f"生成 {len(options)} 个探索选项，策略分布: {strategy_counts}")
        
        return options
    
    def _allocate_options_by_strategy(self, 
                                     n_options: int,
                                     strategy_weights: Dict[ExplorationStrategy, float]) -> Dict[ExplorationStrategy, int]:
        """按策略分配选项数量"""
        total_weight = sum(strategy_weights.values())
        if total_weight == 0:
            # 平均分配
            strategies = list(strategy_weights.keys())
            weight_per_strategy = 1.0 / len(strategies)
            strategy_weights = {s: weight_per_strategy for s in strategies}
            total_weight = 1.0
        
        counts = {}
        remaining = n_options
        
        # 第一轮：按权重分配整数部分
        for strategy, weight in strategy_weights.items():
            count = int(n_options * weight / total_weight)
            counts[strategy] = count
            remaining -= count
        
        # 第二轮：分配剩余部分
        if remaining > 0:
            # 按权重排序
            sorted_strategies = sorted(strategy_weights.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True)
            
            for strategy, weight in sorted_strategies:
                if remaining <= 0:
                    break
                counts[strategy] += 1
                remaining -= 1
        
        return counts
    
    def _generate_strategy_options(self, 
                                  strategy: ExplorationStrategy,
                                  count: int) -> List[ExplorationOption]:
        """生成特定策略的选项"""
        options = []
        
        for i in range(count):
            option_id = f"exploration_{strategy.value}_{len(self.exploration_options) + i}"
            
            # 基于策略生成参数
            if strategy == ExplorationStrategy.UNCERTAINTY_BASED:
                params = self._generate_uncertainty_based_params()
            elif strategy == ExplorationStrategy.NOVELTY_BASED:
                params = self._generate_novelty_based_params()
            elif strategy == ExplorationStrategy.INFORMATION_GAIN:
                params = self._generate_information_gain_params()
            else:  # DIVERSITY_DRIVEN
                params = self._generate_diversity_driven_params()
            
            # 创建选项
            option = ExplorationOption(
                id=option_id,
                description=params['description'],
                exploration_strategy=strategy,
                expected_novelty=params['expected_novelty'],
                expected_uncertainty_reduction=params['expected_uncertainty_reduction'],
                expected_information_gain=params['expected_information_gain'],
                diversity_contribution=params['diversity_contribution'],
                exploration_cost=params['exploration_cost'],
                estimated_duration=params['estimated_duration'],
                success_probability=params['success_probability'],
                context=params['context'],
                metadata={'generation_strategy': strategy.value}
            )
            
            options.append(option)
        
        return options
    
    def _generate_uncertainty_based_params(self) -> Dict[str, Any]:
        """生成基于不确定性策略的参数"""
        return {
            'description': '探索高不确定性区域以降低模型不确定性',
            'expected_novelty': random.uniform(0.3, 0.7),
            'expected_uncertainty_reduction': random.uniform(0.5, 0.9),
            'expected_information_gain': random.uniform(0.4, 0.8),
            'diversity_contribution': random.uniform(0.2, 0.6),
            'exploration_cost': random.uniform(15.0, 40.0),
            'estimated_duration': random.uniform(1.0, 4.0) * 3600,
            'success_probability': random.uniform(0.6, 0.9),
            'context': {'focus': 'uncertainty_reduction', 'method': 'targeted_exploration'}
        }
    
    def _generate_novelty_based_params(self) -> Dict[str, Any]:
        """生成基于新奇性策略的参数"""
        return {
            'description': '探索新奇区域以发现新知识和模式',
            'expected_novelty': random.uniform(0.6, 1.0),
            'expected_uncertainty_reduction': random.uniform(0.2, 0.6),
            'expected_information_gain': random.uniform(0.5, 0.9),
            'diversity_contribution': random.uniform(0.5, 0.9),
            'exploration_cost': random.uniform(20.0, 50.0),
            'estimated_duration': random.uniform(1.5, 5.0) * 3600,
            'success_probability': random.uniform(0.5, 0.8),
            'context': {'focus': 'novelty_discovery', 'method': 'broad_exploration'}
        }
    
    def _generate_information_gain_params(self) -> Dict[str, Any]:
        """生成基于信息增益策略的参数"""
        return {
            'description': '探索信息增益最大化的区域以加速学习',
            'expected_novelty': random.uniform(0.4, 0.8),
            'expected_uncertainty_reduction': random.uniform(0.3, 0.7),
            'expected_information_gain': random.uniform(0.7, 1.0),
            'diversity_contribution': random.uniform(0.3, 0.7),
            'exploration_cost': random.uniform(10.0, 35.0),
            'estimated_duration': random.uniform(1.0, 3.0) * 3600,
            'success_probability': random.uniform(0.7, 0.95),
            'context': {'focus': 'information_gain', 'method': 'efficient_exploration'}
        }
    
    def _generate_diversity_driven_params(self) -> Dict[str, Any]:
        """生成基于多样性策略的参数"""
        return {
            'description': '探索多样化区域以确保广泛的知识覆盖',
            'expected_novelty': random.uniform(0.5, 0.9),
            'expected_uncertainty_reduction': random.uniform(0.2, 0.5),
            'expected_information_gain': random.uniform(0.3, 0.7),
            'diversity_contribution': random.uniform(0.7, 1.0),
            'exploration_cost': random.uniform(25.0, 60.0),
            'estimated_duration': random.uniform(2.0, 6.0) * 3600,
            'success_probability': random.uniform(0.4, 0.7),
            'context': {'focus': 'diversity', 'method': 'wide_coverage_exploration'}
        }
    
    def select_best_exploration_options(self,
                                       options: Optional[List[ExplorationOption]] = None,
                                       n_selected: int = 3,
                                       strategy_constraint: Optional[ExplorationStrategy] = None) -> List[ExplorationOption]:
        """
        选择最佳探索选项
        
        Args:
            options: 选项列表（如果为None则使用所有选项）
            n_selected: 选择数量
            strategy_constraint: 策略约束
            
        Returns:
            选择的探索选项
        """
        if options is None:
            options = list(self.exploration_options.values())
        
        # 应用策略约束
        if strategy_constraint is not None:
            options = [opt for opt in options if opt.exploration_strategy == strategy_constraint]
        
        # 计算探索分数并排序
        scored_options = []
        for option in options:
            score = option.exploration_score
            scored_options.append((score, option))
        
        scored_options.sort(key=lambda x: x[0], reverse=True)
        
        # 选择前N个
        selected = [option for _, option in scored_options[:n_selected]]
        
        # 更新统计
        self.performance_stats['exploration_actions_taken'] += len(selected)
        
        logger.info(f"选择 {len(selected)} 个探索选项，最高分数: {scored_options[0][0]:.3f}")
        
        return selected
    
    def update_from_exploration_results(self,
                                      option_id: str,
                                      actual_novelty: float,
                                      actual_uncertainty_reduction: float,
                                      actual_information_gain: float,
                                      success: bool):
        """
        从探索结果更新系统
        
        Args:
            option_id: 选项ID
            actual_novelty: 实际新奇性
            actual_uncertainty_reduction: 实际不确定性降低
            actual_information_gain: 实际信息增益
            success: 是否成功
        """
        # 更新性能统计
        self.performance_stats['information_gain_achieved'] += actual_information_gain
        self.performance_stats['uncertainty_reduction'] += actual_uncertainty_reduction
        
        # 更新好奇心状态
        if success:
            # 成功探索：更新学习进度
            learning_progress = (actual_information_gain * 0.5 + 
                                actual_uncertainty_reduction * 0.3 +
                                actual_novelty * 0.2)
            
            for curiosity_state in self.curiosity_states.values():
                curiosity_state.learning_progress = min(1.0, 
                    curiosity_state.learning_progress + learning_progress * 0.1)
        
        # 衰减好奇心
        self._decay_curiosity()
        
        logger.debug(f"探索结果更新: 选项 {option_id}，信息增益: {actual_information_gain:.3f}，"
                    f"不确定性降低: {actual_uncertainty_reduction:.3f}")
    
    def _decay_curiosity(self):
        """衰减好奇心"""
        for curiosity_state in self.curiosity_states.values():
            curiosity_state.decay(self.curiosity_decay_rate)
    
    def get_curiosity_levels(self) -> Dict[CuriosityType, Dict[str, float]]:
        """获取好奇心水平"""
        levels = {}
        for curiosity_type, state in self.curiosity_states.items():
            levels[curiosity_type] = {
                'activation_level': state.activation_level,
                'saturation_level': state.saturation_level,
                'learning_progress': state.learning_progress,
                'activation_count': state.activation_count
            }
        return levels
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'novelty_detection_count': self.performance_stats['novelty_detections'],
            'uncertainty_estimation_count': self.performance_stats['uncertainty_estimations'],
            'exploration_options_generated': self.performance_stats['exploration_options_generated'],
            'exploration_actions_taken': self.performance_stats['exploration_actions_taken'],
            'total_information_gain': self.performance_stats['information_gain_achieved'],
            'total_uncertainty_reduction': self.performance_stats['uncertainty_reduction'],
            'average_novelty_score': self.performance_stats['average_novelty_score'],
            'curiosity_levels': self.get_curiosity_levels(),
            'current_exploration_ratio': self.exploration_exploitation_ratio
        }


# 示例和测试函数
def create_example_curiosity_explorer() -> CuriosityDrivenExploration:
    """创建示例好奇心探索器"""
    explorer = CuriosityDrivenExploration(
        max_exploration_options=15,
        novelty_threshold=0.35,
        uncertainty_threshold=0.45,
        exploration_exploitation_ratio=0.35,
        curiosity_decay_rate=0.96
    )
    return explorer


def test_curiosity_driven_exploration():
    """测试好奇心驱动探索"""
    logger.info("开始测试好奇心驱动探索")
    
    # 创建示例探索器
    explorer = create_example_curiosity_explorer()
    
    # 新奇性检测测试
    logger.info("新奇性检测测试...")
    for i in range(5):
        observation_id = f"obs_{i}"
        observation_data = f"data_{random.randint(1, 100)}"
        expected_value = f"data_{random.randint(1, 100)}" if random.random() > 0.5 else None
        
        detection = explorer.detect_novelty(
            observation_id=observation_id,
            observation_data=observation_data,
            expected_value=expected_value,
            context={'novelty_context': random.uniform(0.0, 1.0)}
        )
        
        logger.debug(f"新奇性检测 {i+1}: {observation_id}，分数: {detection.novelty_score:.3f}")
    
    # 不确定性估计测试
    logger.info("不确定性估计测试...")
    for i in range(3):
        state_id = f"state_{i}"
        estimation = explorer.estimate_uncertainty(
            state_id=state_id,
            uncertainty_type="model_uncertainty"
        )
        
        logger.debug(f"不确定性估计 {i+1}: {state_id}，值: {estimation.uncertainty_value:.3f}")
    
    # 生成探索选项
    logger.info("生成探索选项...")
    options = explorer.generate_exploration_options(n_options=8)
    logger.info(f"生成了 {len(options)} 个探索选项")
    
    # 显示一些选项信息
    for i, option in enumerate(options[:3]):
        logger.info(f"选项 {i+1}: {option.description}，策略: {option.exploration_strategy.value}，"
                   f"分数: {option.exploration_score:.3f}")
    
    # 选择最佳选项
    logger.info("选择最佳探索选项...")
    selected = explorer.select_best_exploration_options(n_selected=2)
    logger.info(f"选择了 {len(selected)} 个最佳选项")
    
    for option in selected:
        logger.info(f"  选择: {option.description}，探索分数: {option.exploration_score:.3f}")
    
    # 模拟探索结果
    if selected:
        option = selected[0]
        logger.info(f"模拟探索结果: {option.description}")
        
        explorer.update_from_exploration_results(
            option_id=option.id,
            actual_novelty=option.expected_novelty * random.uniform(0.8, 1.2),
            actual_uncertainty_reduction=option.expected_uncertainty_reduction * random.uniform(0.7, 1.1),
            actual_information_gain=option.expected_information_gain * random.uniform(0.9, 1.3),
            success=random.random() < option.success_probability
        )
    
    # 获取好奇心水平
    curiosity_levels = explorer.get_curiosity_levels()
    logger.info("好奇心水平:")
    for curiosity_type, levels in curiosity_levels.items():
        logger.info(f"  {curiosity_type.value}: 激活 {levels['activation_level']:.3f}，"
                   f"饱和 {levels['saturation_level']:.3f}")
    
    # 获取统计信息
    stats = explorer.get_statistics()
    logger.info(f"统计信息: {stats['novelty_detection_count']} 次新奇性检测，"
               f"{stats['exploration_actions_taken']} 次探索行动")
    
    logger.info("好奇心驱动探索测试完成")
    return explorer


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_explorer_instance = test_curiosity_driven_exploration()