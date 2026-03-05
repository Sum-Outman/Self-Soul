"""
推理策略评估器

该模块实现AGI系统的推理策略评估功能，包括：
1. 策略性能评估：评估不同推理策略的效果和效率
2. 适应性分析：分析策略对不同情境的适应性
3. 策略比较：比较不同策略的优劣
4. 策略推荐：基于情境推荐最优策略
5. 策略优化：优化现有策略或生成新策略

核心功能：
1. 多维度策略性能度量
2. 情境-策略匹配分析
3. 策略效果对比实验
4. 个性化策略推荐
5. 策略参数优化

技术特性：
- 多维度评估指标体系
- 情境特征提取和匹配
- 贝叶斯策略选择模型
- 强化学习策略优化
- 实时性能监控和调整
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
import random
from scipy import stats

# 配置日志
logger = logging.getLogger(__name__)

class ReasoningStrategy(Enum):
    """推理策略"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    PROBABILISTIC = "probabilistic"
    HEURISTIC = "heuristic"
    SYSTEMATIC = "systematic"
    INTUITIVE = "intuitive"
    HYBRID = "hybrid"

class ProblemType(Enum):
    """问题类型"""
    LOGICAL_REASONING = "logical_reasoning"
    CAUSAL_INFERENCE = "causal_inference"
    PATTERN_RECOGNITION = "pattern_recognition"
    DECISION_MAKING = "decision_making"
    PROBLEM_SOLVING = "problem_solving"
    PLANNING = "planning"
    LEARNING = "learning"
    CREATIVE_THINKING = "creative_thinking"

class EvaluationDimension(Enum):
    """评估维度"""
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    GENERALIZABILITY = "generalizability"
    EXPLAINABILITY = "explainability"
    CREATIVITY = "creativity"
    RESOURCE_USAGE = "resource_usage"
    ADAPTABILITY = "adaptability"

@dataclass
class StrategyPerformance:
    """策略性能"""
    strategy: ReasoningStrategy
    problem_type: ProblemType
    accuracy: float  # 0-1
    efficiency: float  # 0-1 (速度/资源效率)
    robustness: float  # 0-1 (抗干扰能力)
    generalizability: float  # 0-1 (泛化能力)
    explainability: float  # 0-1 (可解释性)
    creativity: float  # 0-1 (创造性)
    resource_usage: float  # 0-1 (资源使用，越低越好)
    adaptability: float  # 0-1 (适应性)
    overall_score: float  # 0-1
    confidence: float  # 0-1
    evaluation_count: int
    last_evaluated: datetime

@dataclass
class ProblemContext:
    """问题上下文"""
    problem_id: str
    problem_type: ProblemType
    description: str
    complexity: float  # 0-1
    uncertainty: float  # 0-1
    time_constraint: float  # 0-1
    resource_constraints: Dict[str, float]
    domain: str
    features: Dict[str, Any]

@dataclass
class StrategyRecommendation:
    """策略推荐"""
    recommendation_id: str
    problem_context: ProblemContext
    recommended_strategies: List[Tuple[ReasoningStrategy, float]]  # (策略, 适用性得分)
    rationale: List[str]
    expected_performance: Dict[ReasoningStrategy, StrategyPerformance]
    confidence: float
    generated_at: datetime

@dataclass
class StrategyComparison:
    """策略比较"""
    comparison_id: str
    problem_type: ProblemType
    strategies: List[ReasoningStrategy]
    performance_comparison: Dict[ReasoningStrategy, StrategyPerformance]
    strengths_and_weaknesses: Dict[ReasoningStrategy, List[str]]
    best_strategy: ReasoningStrategy
    margin_of_superiority: float

class ReasoningStrategyEvaluator:
    """
    推理策略评估器
    
    核心组件:
    1. 性能评估器: 评估策略在不同维度的性能
    2. 情境分析器: 分析问题情境特征
    3. 匹配优化器: 优化策略-情境匹配
    4. 比较分析器: 比较不同策略的优劣
    5. 推荐生成器: 生成策略推荐
    6. 策略优化器: 优化策略参数或生成新策略
    
    工作流程:
    问题情境 → 情境分析器 → 提取特征 → 匹配优化器 → 计算匹配度
    策略性能 → 性能评估器 → 评估性能 → 比较分析器 → 比较策略
    综合分析 → 推荐生成器 → 生成推荐 → 策略优化器 → 优化策略
    
    技术特性:
    - 多维度性能评估
    - 情境特征提取
    - 贝叶斯策略选择
    - 性能对比分析
    - 自适应策略优化
    """
    
    def __init__(self,
                 evaluation_history_size: int = 1000,
                 min_evaluations_for_reliability: int = 10,
                 exploration_rate: float = 0.3,
                 adaptation_rate: float = 0.1,
                 bayesian_optimization_enabled: bool = True):
        """
        初始化推理策略评估器
        
        Args:
            evaluation_history_size: 评估历史大小
            min_evaluations_for_reliability: 可靠性所需的最小评估次数
            exploration_rate: 探索率（尝试新策略的概率）
            adaptation_rate: 适应率（更新权重和参数的速度）
            bayesian_optimization_enabled: 是否启用贝叶斯优化
        """
        self.evaluation_history_size = evaluation_history_size
        self.min_evaluations_for_reliability = min_evaluations_for_reliability
        self.exploration_rate = exploration_rate
        self.adaptation_rate = adaptation_rate
        self.bayesian_optimization_enabled = bayesian_optimization_enabled
        
        # 策略性能数据库
        self.strategy_performance: Dict[Tuple[ReasoningStrategy, ProblemType], StrategyPerformance] = {}
        
        # 评估历史
        self.evaluation_history: List[Dict[str, Any]] = []
        
        # 策略-情境匹配规则
        self.strategy_context_rules: Dict[Tuple[ReasoningStrategy, ProblemType], Dict[str, Any]] = {}
        self._initialize_strategy_context_rules()
        
        # 问题特征提取器
        self.feature_extractors: Dict[ProblemType, List[callable]] = {}
        self._initialize_feature_extractors()
        
        # 配置参数
        self.config = {
            'performance_decay_factor': 0.99,
            'confidence_growth_rate': 0.05,
            'dimension_weights': {
                EvaluationDimension.ACCURACY.value: 0.25,
                EvaluationDimension.EFFICIENCY.value: 0.20,
                EvaluationDimension.ROBUSTNESS.value: 0.15,
                EvaluationDimension.GENERALIZABILITY.value: 0.10,
                EvaluationDimension.EXPLAINABILITY.value: 0.10,
                EvaluationDimension.CREATIVITY.value: 0.10,
                EvaluationDimension.RESOURCE_USAGE.value: 0.05,
                EvaluationDimension.ADAPTABILITY.value: 0.05
            },
            'matching_threshold': 0.6,
            'comparison_significance_level': 0.05,
            'bayesian_prior_strength': 0.1,
            'optimization_iterations': 50
        }
        
        # 贝叶斯优化器（如果启用）
        self.bayesian_optimizer = None
        if self.bayesian_optimization_enabled:
            self._initialize_bayesian_optimizer()
        
        # 性能统计
        self.performance_stats = {
            'evaluations_performed': 0,
            'recommendations_generated': 0,
            'strategy_comparisons': 0,
            'strategy_optimizations': 0,
            'average_recommendation_confidence': 0.0,
            'strategy_improvement_rate': 0.0,
            'matching_accuracy': 0.0
        }
        
        # 状态变量
        self.last_evaluation_time = time.time()
        self.system_start_time = time.time()
        self.recommendation_counter = 0
        
        logger.info(f"推理策略评估器初始化完成，探索率: {exploration_rate}")
    
    def _initialize_strategy_context_rules(self):
        """初始化策略-情境匹配规则"""
        # 演绎推理：适合逻辑明确、规则清晰的问题
        self.strategy_context_rules[(ReasoningStrategy.DEDUCTIVE, ProblemType.LOGICAL_REASONING)] = {
            "ideal_features": {
                "rule_clarity": 0.8,
                "certainty": 0.7,
                "structuredness": 0.9
            },
            "weight_factors": {
                "accuracy": 0.4,
                "explainability": 0.3,
                "efficiency": 0.3
            },
            "strengths": ["逻辑严谨", "可证明性", "确定性高"],
            "weaknesses": ["需要完整前提", "缺乏创造性", "不适用模糊问题"]
        }
        
        # 归纳推理：适合从具体实例中发现一般规律
        self.strategy_context_rules[(ReasoningStrategy.INDUCTIVE, ProblemType.PATTERN_RECOGNITION)] = {
            "ideal_features": {
                "example_availability": 0.8,
                "pattern_regularity": 0.7,
                "generalization_needed": 0.9
            },
            "weight_factors": {
                "generalizability": 0.4,
                "creativity": 0.3,
                "robustness": 0.3
            },
            "strengths": ["从数据中学习", "发现新规律", "适用大数据"],
            "weaknesses": ["结论不一定可靠", "需要足够样本", "可能过度泛化"]
        }
        
        # 溯因推理：适合解释观测现象、提出假设
        self.strategy_context_rules[(ReasoningStrategy.ABDUCTIVE, ProblemType.CAUSAL_INFERENCE)] = {
            "ideal_features": {
                "observation_complexity": 0.8,
                "explanation_needed": 0.9,
                "hypothesis_generation": 0.8
            },
            "weight_factors": {
                "creativity": 0.4,
                "explainability": 0.3,
                "adaptability": 0.3
            },
            "strengths": ["解释复杂现象", "生成创新假设", "处理不确定性"],
            "weaknesses": ["假设不一定正确", "计算复杂度高", "需要领域知识"]
        }
        
        # 类比推理：适合借助相似性解决问题
        self.strategy_context_rules[(ReasoningStrategy.ANALOGICAL, ProblemType.CREATIVE_THINKING)] = {
            "ideal_features": {
                "similarity_detection": 0.8,
                "transfer_learning": 0.7,
                "creative_solution": 0.9
            },
            "weight_factors": {
                "creativity": 0.5,
                "adaptability": 0.3,
                "efficiency": 0.2
            },
            "strengths": ["跨领域借鉴", "创新解决方案", "学习效率高"],
            "weaknesses": ["类比可能不恰当", "需要丰富知识", "可能过度依赖相似性"]
        }
        
        # 因果推理：适合分析因果关系和干预效果
        self.strategy_context_rules[(ReasoningStrategy.CAUSAL, ProblemType.CAUSAL_INFERENCE)] = {
            "ideal_features": {
                "causal_structure": 0.9,
                "intervention_analysis": 0.8,
                "counterfactual_reasoning": 0.7
            },
            "weight_factors": {
                "accuracy": 0.4,
                "robustness": 0.3,
                "explainability": 0.3
            },
            "strengths": ["揭示深层机制", "支持决策干预", "处理混杂因素"],
            "weaknesses": ["需要因果数据", "模型复杂度高", "假设检验严格"]
        }
        
        # 概率推理：适合处理不确定性和风险
        self.strategy_context_rules[(ReasoningStrategy.PROBABILISTIC, ProblemType.DECISION_MAKING)] = {
            "ideal_features": {
                "uncertainty_level": 0.8,
                "risk_analysis": 0.9,
                "probabilistic_data": 0.7
            },
            "weight_factors": {
                "robustness": 0.4,
                "accuracy": 0.3,
                "generalizability": 0.3
            },
            "strengths": ["处理不确定性", "量化风险评估", "支持最优决策"],
            "weaknesses": ["需要概率模型", "计算复杂度高", "对先验敏感"]
        }
    
    def _initialize_feature_extractors(self):
        """初始化特征提取器"""
        # 逻辑推理问题特征提取
        self.feature_extractors[ProblemType.LOGICAL_REASONING] = [
            self._extract_rule_clarity,
            self._extract_structuredness,
            self._extract_completeness
        ]
        
        # 因果推理问题特征提取
        self.feature_extractors[ProblemType.CAUSAL_INFERENCE] = [
            self._extract_causal_structure,
            self._extract_observation_complexity,
            self._extract_intervention_potential
        ]
        
        # 模式识别问题特征提取
        self.feature_extractors[ProblemType.PATTERN_RECOGNITION] = [
            self._extract_pattern_regularity,
            self._extract_example_availability,
            self._extract_feature_dimension
        ]
        
        # 决策问题特征提取
        self.feature_extractors[ProblemType.DECISION_MAKING] = [
            self._extract_uncertainty_level,
            self._extract_risk_implication,
            self._extract_time_constraint
        ]
        
        # 创造性思维问题特征提取
        self.feature_extractors[ProblemType.CREATIVE_THINKING] = [
            self._extract_novelty_requirement,
            self._extract_constraint_flexibility,
            self._extract_inspiration_potential
        ]
    
    def _extract_rule_clarity(self, context: ProblemContext) -> float:
        """提取规则清晰度特征"""
        description = context.description.lower()
        
        clarity_indicators = ["rule", "law", "principle", "theorem", "axiom"]
        ambiguity_indicators = ["maybe", "perhaps", "could", "might", "ambiguous"]
        
        clarity_score = sum(1 for indicator in clarity_indicators if indicator in description)
        ambiguity_score = sum(1 for indicator in ambiguity_indicators if indicator in description)
        
        total = clarity_score + ambiguity_score + 1  # 避免除零
        return clarity_score / total
    
    def _extract_structuredness(self, context: ProblemContext) -> float:
        """提取结构化程度特征"""
        # 基于问题复杂度和领域
        complexity = context.complexity
        domain = context.domain.lower()
        
        # 某些领域更结构化
        structured_domains = ["mathematics", "logic", "programming", "physics"]
        unstructured_domains = ["art", "philosophy", "creative", "subjective"]
        
        structure_bonus = 0.0
        for domain_pattern in structured_domains:
            if domain_pattern in domain:
                structure_bonus = 0.3
                break
        
        for domain_pattern in unstructured_domains:
            if domain_pattern in domain:
                structure_bonus = -0.2
                break
        
        return max(0.0, min(1.0, 0.5 + structure_bonus - complexity * 0.2))
    
    def _extract_completeness(self, context: ProblemContext) -> float:
        """提取信息完整性特征"""
        # 基于问题描述长度和细节
        description = context.description
        word_count = len(description.split())
        
        # 简单启发式：描述越长越完整
        completeness = min(1.0, word_count / 100.0)
        
        # 考虑不确定性
        uncertainty = context.uncertainty
        completeness *= (1.0 - uncertainty * 0.5)
        
        return completeness
    
    def _extract_causal_structure(self, context: ProblemContext) -> float:
        """提取因果结构特征"""
        description = context.description.lower()
        
        causal_keywords = ["cause", "effect", "because", "therefore", "leads to", "results in"]
        causal_count = sum(1 for keyword in causal_keywords if keyword in description)
        
        return min(1.0, causal_count / 3.0)
    
    def _extract_observation_complexity(self, context: ProblemContext) -> float:
        """提取观测复杂性特征"""
        # 基于问题复杂度和不确定性
        complexity = context.complexity
        uncertainty = context.uncertainty
        
        # 复杂且不确定的问题观测更复杂
        observation_complexity = (complexity + uncertainty) / 2.0
        
        # 调整范围
        return min(1.0, observation_complexity * 1.2)
    
    def _extract_intervention_potential(self, context: ProblemContext) -> float:
        """提取干预潜力特征"""
        description = context.description.lower()
        
        intervention_keywords = ["what if", "intervene", "change", "modify", "alter", "control"]
        intervention_count = sum(1 for keyword in intervention_keywords if keyword in description)
        
        # 决策问题通常有干预潜力
        if context.problem_type == ProblemType.DECISION_MAKING:
            intervention_count += 2
        
        return min(1.0, intervention_count / 4.0)
    
    def _extract_pattern_regularity(self, context: ProblemContext) -> float:
        """提取模式规律性特征"""
        description = context.description.lower()
        
        pattern_keywords = ["pattern", "regular", "repeat", "sequence", "trend", "cycle"]
        pattern_count = sum(1 for keyword in pattern_keywords if keyword in description)
        
        irregular_keywords = ["random", "chaotic", "irregular", "unpredictable"]
        irregular_count = sum(1 for keyword in irregular_keywords if keyword in description)
        
        total = pattern_count + irregular_count + 1
        regularity = pattern_count / total
        
        # 调整基于复杂度
        complexity = context.complexity
        regularity *= (1.0 - complexity * 0.3)
        
        return regularity
    
    def _extract_example_availability(self, context: ProblemContext) -> float:
        """提取示例可用性特征"""
        description = context.description
        
        # 检查是否有具体示例
        example_indicators = ["for example", "e.g.", "such as", "instance", "case"]
        example_count = sum(1 for indicator in example_indicators if indicator.lower() in description.lower())
        
        # 数字通常表示具体数据
        import re
        number_count = len(re.findall(r'\d+', description))
        
        availability_score = min(1.0, (example_count * 0.3 + number_count * 0.1))
        
        return availability_score
    
    def _extract_feature_dimension(self, context: ProblemContext) -> float:
        """提取特征维度特征"""
        description = context.description
        
        # 简单词数作为维度代理
        word_count = len(description.split())
        dimension_score = min(1.0, word_count / 200.0)
        
        return dimension_score
    
    def _extract_uncertainty_level(self, context: ProblemContext) -> float:
        """提取不确定性水平特征"""
        # 直接使用上下文中的不确定性
        uncertainty = context.uncertainty
        
        # 调整基于问题类型
        if context.problem_type == ProblemType.DECISION_MAKING:
            uncertainty *= 1.2  # 决策问题通常更不确定
        
        return min(1.0, uncertainty)
    
    def _extract_risk_implication(self, context: ProblemContext) -> float:
        """提取风险影响特征"""
        description = context.description.lower()
        
        risk_keywords = ["risk", "danger", "loss", "fail", "mistake", "error", "consequence"]
        risk_count = sum(1 for keyword in risk_keywords if keyword in description)
        
        # 时间约束增加风险感知
        time_constraint = context.time_constraint
        risk_count += int(time_constraint * 3)
        
        return min(1.0, risk_count / 5.0)
    
    def _extract_time_constraint(self, context: ProblemContext) -> float:
        """提取时间约束特征"""
        # 直接使用上下文中的时间约束
        return context.time_constraint
    
    def _extract_novelty_requirement(self, context: ProblemContext) -> float:
        """提取新颖性需求特征"""
        description = context.description.lower()
        
        novelty_keywords = ["novel", "creative", "innovative", "original", "new", "unique"]
        novelty_count = sum(1 for keyword in novelty_keywords if keyword in description)
        
        constraint_keywords = ["constraint", "limit", "restrict", "boundary"]
        constraint_count = sum(1 for keyword in constraint_keywords if keyword in description)
        
        # 新颖性需求与约束数量负相关
        novelty_score = min(1.0, novelty_count / 3.0)
        constraint_factor = max(0.0, 1.0 - constraint_count / 5.0)
        
        return novelty_score * constraint_factor
    
    def _extract_constraint_flexibility(self, context: ProblemContext) -> float:
        """提取约束灵活性特征"""
        description = context.description.lower()
        
        flexible_keywords = ["flexible", "adaptable", "adjustable", "modifiable"]
        flexible_count = sum(1 for keyword in flexible_keywords if keyword in description)
        
        rigid_keywords = ["fixed", "rigid", "strict", "must", "required", "mandatory"]
        rigid_count = sum(1 for keyword in rigid_keywords if keyword in description)
        
        total = flexible_count + rigid_count + 1
        flexibility = flexible_count / total
        
        return flexibility
    
    def _extract_inspiration_potential(self, context: ProblemContext) -> float:
        """提取灵感潜力特征"""
        description = context.description
        
        # 检查是否有启发性的元素
        inspiration_indicators = ["inspire", "imagine", "what if", "suppose", "analogy", "metaphor"]
        inspiration_count = sum(1 for indicator in inspiration_indicators if indicator.lower() in description.lower())
        
        # 问题复杂度增加灵感潜力
        complexity = context.complexity
        
        inspiration_potential = min(1.0, inspiration_count / 3.0 + complexity * 0.3)
        
        return inspiration_potential
    
    def _initialize_bayesian_optimizer(self):
        """初始化贝叶斯优化器"""
        # 简化版本 - 实际项目中应使用真实的贝叶斯优化库
        logger.info("初始化贝叶斯优化器（简化版）")
        
        self.bayesian_optimizer = {
            "strategy_priors": {},
            "performance_models": {},
            "acquisition_function": "expected_improvement",
            "exploration_weight": 0.1
        }
        
        # 初始化先验
        for strategy in ReasoningStrategy:
            for problem_type in ProblemType:
                key = (strategy, problem_type)
                self.bayesian_optimizer["strategy_priors"][key] = {
                    "mean": 0.5,
                    "variance": 0.25,
                    "observation_count": 0
                }
    
    def extract_problem_features(self, context: ProblemContext) -> Dict[str, float]:
        """提取问题特征"""
        features = {}
        
        # 提取基本特征
        features["complexity"] = context.complexity
        features["uncertainty"] = context.uncertainty
        features["time_constraint"] = context.time_constraint
        
        # 提取类型特定特征
        problem_type = context.problem_type
        if problem_type in self.feature_extractors:
            extractors = self.feature_extractors[problem_type]
            for extractor in extractors:
                try:
                    feature_name = extractor.__name__.replace("_extract_", "")
                    feature_value = extractor(context)
                    features[feature_name] = feature_value
                except Exception as e:
                    logger.warning(f"特征提取失败: {extractor.__name__}, 错误: {e}")
        
        return features
    
    def evaluate_strategy_performance(self,
                                    strategy: ReasoningStrategy,
                                    problem_context: ProblemContext,
                                    performance_metrics: Dict[str, float]) -> StrategyPerformance:
        """评估策略性能"""
        # 提取特征
        features = self.extract_problem_features(problem_context)
        
        # 获取或创建性能记录
        key = (strategy, problem_context.problem_type)
        if key in self.strategy_performance:
            existing = self.strategy_performance[key]
            
            # 更新性能指标（指数移动平均）
            alpha = self.adaptation_rate
            accuracy = alpha * performance_metrics.get("accuracy", 0.5) + (1 - alpha) * existing.accuracy
            efficiency = alpha * performance_metrics.get("efficiency", 0.5) + (1 - alpha) * existing.efficiency
            robustness = alpha * performance_metrics.get("robustness", 0.5) + (1 - alpha) * existing.robustness
            generalizability = alpha * performance_metrics.get("generalizability", 0.5) + (1 - alpha) * existing.generalizability
            explainability = alpha * performance_metrics.get("explainability", 0.5) + (1 - alpha) * existing.explainability
            creativity = alpha * performance_metrics.get("creativity", 0.5) + (1 - alpha) * existing.creativity
            resource_usage = alpha * performance_metrics.get("resource_usage", 0.5) + (1 - alpha) * existing.resource_usage
            adaptability = alpha * performance_metrics.get("adaptability", 0.5) + (1 - alpha) * existing.adaptability
            
            # 更新置信度
            evaluation_count = existing.evaluation_count + 1
            confidence = min(1.0, existing.confidence + self.config['confidence_growth_rate'])
            
        else:
            # 创建新记录
            accuracy = performance_metrics.get("accuracy", 0.5)
            efficiency = performance_metrics.get("efficiency", 0.5)
            robustness = performance_metrics.get("robustness", 0.5)
            generalizability = performance_metrics.get("generalizability", 0.5)
            explainability = performance_metrics.get("explainability", 0.5)
            creativity = performance_metrics.get("creativity", 0.5)
            resource_usage = performance_metrics.get("resource_usage", 0.5)
            adaptability = performance_metrics.get("adaptability", 0.5)
            
            evaluation_count = 1
            confidence = 0.5  # 初始置信度
        
        # 计算总体得分
        dimension_weights = self.config['dimension_weights']
        overall_score = (
            accuracy * dimension_weights[EvaluationDimension.ACCURACY.value] +
            efficiency * dimension_weights[EvaluationDimension.EFFICIENCY.value] +
            robustness * dimension_weights[EvaluationDimension.ROBUSTNESS.value] +
            generalizability * dimension_weights[EvaluationDimension.GENERALIZABILITY.value] +
            explainability * dimension_weights[EvaluationDimension.EXPLAINABILITY.value] +
            creativity * dimension_weights[EvaluationDimension.CREATIVITY.value] +
            (1.0 - resource_usage) * dimension_weights[EvaluationDimension.RESOURCE_USAGE.value] +  # 资源使用越低越好
            adaptability * dimension_weights[EvaluationDimension.ADAPTABILITY.value]
        )
        
        # 创建性能对象
        performance = StrategyPerformance(
            strategy=strategy,
            problem_type=problem_context.problem_type,
            accuracy=accuracy,
            efficiency=efficiency,
            robustness=robustness,
            generalizability=generalizability,
            explainability=explainability,
            creativity=creativity,
            resource_usage=resource_usage,
            adaptability=adaptability,
            overall_score=overall_score,
            confidence=confidence,
            evaluation_count=evaluation_count,
            last_evaluated=datetime.now()
        )
        
        # 保存性能记录
        self.strategy_performance[key] = performance
        
        # 更新评估历史
        history_entry = {
            "timestamp": datetime.now(),
            "strategy": strategy.value,
            "problem_type": problem_context.problem_type.value,
            "performance_metrics": performance_metrics,
            "overall_score": overall_score,
            "confidence": confidence
        }
        self.evaluation_history.append(history_entry)
        if len(self.evaluation_history) > self.evaluation_history_size:
            self.evaluation_history.pop(0)
        
        # 更新性能统计
        self.performance_stats['evaluations_performed'] += 1
        
        # 更新贝叶斯先验（如果启用）
        if self.bayesian_optimization_enabled and self.bayesian_optimizer:
            if key in self.bayesian_optimizer["strategy_priors"]:
                prior = self.bayesian_optimizer["strategy_priors"][key]
                n = prior["observation_count"]
                
                # 更新均值和方差
                old_mean = prior["mean"]
                old_variance = prior["variance"]
                
                new_mean = (old_mean * n + overall_score) / (n + 1)
                if n > 0:
                    # 简化方差更新
                    new_variance = (old_variance * n + (overall_score - old_mean)**2) / (n + 1)
                else:
                    new_variance = 0.25  # 默认方差
                
                self.bayesian_optimizer["strategy_priors"][key] = {
                    "mean": new_mean,
                    "variance": new_variance,
                    "observation_count": n + 1
                }
        
        logger.debug(f"策略性能评估完成: {strategy.value}, 问题类型: {problem_context.problem_type.value}, 得分: {overall_score:.3f}")
        return performance
    
    def recommend_strategies(self, problem_context: ProblemContext) -> StrategyRecommendation:
        """推荐推理策略"""
        self.recommendation_counter += 1
        
        # 提取问题特征
        features = self.extract_problem_features(problem_context)
        
        # 计算策略适用性得分
        strategy_scores = []
        rationale = []
        
        for strategy in ReasoningStrategy:
            # 检查是否有匹配规则
            key = (strategy, problem_context.problem_type)
            if key in self.strategy_context_rules:
                rules = self.strategy_context_rules[key]
                ideal_features = rules["ideal_features"]
                weight_factors = rules["weight_factors"]
                
                # 计算特征匹配度
                feature_match_scores = []
                for feature_name, ideal_value in ideal_features.items():
                    if feature_name in features:
                        actual_value = features[feature_name]
                        match_score = 1.0 - abs(actual_value - ideal_value)
                        feature_match_scores.append(match_score)
                
                if feature_match_scores:
                    feature_match = np.mean(feature_match_scores)
                else:
                    feature_match = 0.5
                
                # 获取性能数据（如果可用）
                performance_score = 0.5
                confidence = 0.5
                if key in self.strategy_performance:
                    performance = self.strategy_performance[key]
                    performance_score = performance.overall_score
                    confidence = performance.confidence
                
                # 计算综合得分
                strategy_score = (
                    feature_match * 0.4 +  # 情境匹配
                    performance_score * 0.4 +  # 历史性能
                    confidence * 0.2  # 置信度
                )
                
                # 应用探索率（随机探索新策略）
                if random.random() < self.exploration_rate:
                    exploration_bonus = random.uniform(0.1, 0.3)
                    strategy_score *= (1.0 + exploration_bonus)
                    rationale.append(f"探索加成应用于 {strategy.value}")
                
                strategy_scores.append((strategy, strategy_score))
                
                # 记录推理依据
                if feature_match > 0.7:
                    rationale.append(f"{strategy.value} 与问题特征匹配度高 ({feature_match:.2f})")
                if performance_score > 0.7:
                    rationale.append(f"{strategy.value} 历史性能优异 ({performance_score:.2f})")
                
            else:
                # 无匹配规则，使用默认得分
                default_score = 0.5
                strategy_scores.append((strategy, default_score))
        
        # 按得分排序
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 获取预期性能
        expected_performance = {}
        for strategy, score in strategy_scores[:3]:  # 前三名
            key = (strategy, problem_context.problem_type)
            if key in self.strategy_performance:
                expected_performance[strategy] = self.strategy_performance[key]
            else:
                # 创建默认性能估计
                expected_performance[strategy] = StrategyPerformance(
                    strategy=strategy,
                    problem_type=problem_context.problem_type,
                    accuracy=0.5,
                    efficiency=0.5,
                    robustness=0.5,
                    generalizability=0.5,
                    explainability=0.5,
                    creativity=0.5,
                    resource_usage=0.5,
                    adaptability=0.5,
                    overall_score=score,
                    confidence=0.3,
                    evaluation_count=0,
                    last_evaluated=datetime.now()
                )
        
        # 计算推荐置信度
        if strategy_scores:
            top_score = strategy_scores[0][1]
            second_score = strategy_scores[1][1] if len(strategy_scores) > 1 else 0.0
            
            if top_score > 0:
                confidence = min(1.0, (top_score - second_score) / top_score * 2.0)
            else:
                confidence = 0.3
        else:
            confidence = 0.3
        
        # 创建推荐
        recommendation = StrategyRecommendation(
            recommendation_id=f"rec_{self.recommendation_counter:06d}",
            problem_context=problem_context,
            recommended_strategies=strategy_scores[:5],  # 前5个策略
            rationale=rationale[:5],  # 前5个推理依据
            expected_performance=expected_performance,
            confidence=confidence,
            generated_at=datetime.now()
        )
        
        # 更新性能统计
        self.performance_stats['recommendations_generated'] += 1
        self.performance_stats['average_recommendation_confidence'] = (
            (self.performance_stats['average_recommendation_confidence'] * 
             (self.performance_stats['recommendations_generated'] - 1) + 
             confidence) / self.performance_stats['recommendations_generated']
        )
        
        logger.info(f"策略推荐生成完成: {recommendation.recommendation_id}, 顶级策略: {strategy_scores[0][0].value}")
        return recommendation
    
    def compare_strategies(self, 
                         problem_type: ProblemType,
                         strategies: List[ReasoningStrategy]) -> StrategyComparison:
        """比较策略"""
        comparison_id = f"comp_{len(self.evaluation_history):06d}"
        
        # 收集性能数据
        performance_comparison = {}
        strengths_and_weaknesses = {}
        
        for strategy in strategies:
            key = (strategy, problem_type)
            if key in self.strategy_performance:
                performance_comparison[strategy] = self.strategy_performance[key]
                
                # 提取优势和劣势
                if key in self.strategy_context_rules:
                    rules = self.strategy_context_rules[key]
                    strengths_and_weaknesses[strategy] = {
                        "strengths": rules.get("strengths", []),
                        "weaknesses": rules.get("weaknesses", [])
                    }
                else:
                    strengths_and_weaknesses[strategy] = {
                        "strengths": ["通用性强", "适用广泛"],
                        "weaknesses": ["缺乏针对性", "可能非最优"]
                    }
            else:
                # 无数据，创建估计
                performance_comparison[strategy] = StrategyPerformance(
                    strategy=strategy,
                    problem_type=problem_type,
                    accuracy=0.5,
                    efficiency=0.5,
                    robustness=0.5,
                    generalizability=0.5,
                    explainability=0.5,
                    creativity=0.5,
                    resource_usage=0.5,
                    adaptability=0.5,
                    overall_score=0.5,
                    confidence=0.3,
                    evaluation_count=0,
                    last_evaluated=datetime.now()
                )
                
                strengths_and_weaknesses[strategy] = {
                    "strengths": ["数据不足，无法评估优势"],
                    "weaknesses": ["数据不足，无法评估劣势"]
                }
        
        # 确定最佳策略
        best_strategy = None
        margin_of_superiority = 0.0
        
        if performance_comparison:
            # 按总体得分排序
            sorted_strategies = sorted(
                performance_comparison.items(),
                key=lambda x: x[1].overall_score,
                reverse=True
            )
            
            best_strategy = sorted_strategies[0][0]
            
            if len(sorted_strategies) > 1:
                best_score = sorted_strategies[0][1].overall_score
                second_score = sorted_strategies[1][1].overall_score
                margin_of_superiority = best_score - second_score
        
        # 创建比较
        comparison = StrategyComparison(
            comparison_id=comparison_id,
            problem_type=problem_type,
            strategies=strategies,
            performance_comparison=performance_comparison,
            strengths_and_weaknesses=strengths_and_weaknesses,
            best_strategy=best_strategy,
            margin_of_superiority=margin_of_superiority
        )
        
        # 更新性能统计
        self.performance_stats['strategy_comparisons'] += 1
        
        return comparison
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        current_time = datetime.now()
        
        # 计算策略改进率
        improvement_rate = 0.0
        if len(self.evaluation_history) >= 10:
            recent_scores = [entry["overall_score"] for entry in self.evaluation_history[-10:]]
            early_scores = [entry["overall_score"] for entry in self.evaluation_history[:10]]
            
            if early_scores and recent_scores:
                avg_early = np.mean(early_scores)
                avg_recent = np.mean(recent_scores)
                
                if avg_early > 0:
                    improvement_rate = (avg_recent - avg_early) / avg_early
        
        # 计算匹配准确率（简化）
        matching_accuracy = 0.7  # 默认值
        if self.performance_stats['recommendations_generated'] > 10:
            matching_accuracy = self.performance_stats['average_recommendation_confidence']
        
        summary = {
            "timestamp": current_time,
            "system_uptime_hours": (time.time() - self.system_start_time) / 3600,
            "evaluation_statistics": {
                "total_evaluations": self.performance_stats['evaluations_performed'],
                "total_recommendations": self.performance_stats['recommendations_generated'],
                "total_comparisons": self.performance_stats['strategy_comparisons'],
                "strategy_coverage": len(self.strategy_performance)
            },
            "performance_metrics": {
                "average_recommendation_confidence": self.performance_stats['average_recommendation_confidence'],
                "strategy_improvement_rate": improvement_rate,
                "matching_accuracy": matching_accuracy
            },
            "top_strategies": self._get_top_strategies(5),
            "recent_recommendations": self._get_recent_recommendations(3)
        }
        
        return summary
    
    def _get_top_strategies(self, n: int) -> List[Dict[str, Any]]:
        """获取顶级策略"""
        if not self.strategy_performance:
            return []
        
        # 按总体得分排序
        sorted_performances = sorted(
            self.strategy_performance.values(),
            key=lambda p: p.overall_score,
            reverse=True
        )
        
        top_strategies = []
        for i, performance in enumerate(sorted_performances[:n]):
            top_strategies.append({
                "rank": i + 1,
                "strategy": performance.strategy.value,
                "problem_type": performance.problem_type.value,
                "overall_score": performance.overall_score,
                "confidence": performance.confidence,
                "evaluation_count": performance.evaluation_count
            })
        
        return top_strategies
    
    def _get_recent_recommendations(self, n: int) -> List[Dict[str, Any]]:
        """获取最近推荐"""
        # 从评估历史中提取推荐信息
        recent_evaluations = self.evaluation_history[-n:] if self.evaluation_history else []
        
        recommendations = []
        for eval_entry in recent_evaluations:
            recommendations.append({
                "strategy": eval_entry["strategy"],
                "problem_type": eval_entry["problem_type"],
                "overall_score": eval_entry["overall_score"],
                "confidence": eval_entry["confidence"],
                "timestamp": eval_entry["timestamp"].isoformat()
            })
        
        return recommendations

# 全局实例
reasoning_strategy_evaluator_instance = ReasoningStrategyEvaluator()

if __name__ == "__main__":
    # 测试推理策略评估器
    print("测试推理策略评估器...")
    
    evaluator = ReasoningStrategyEvaluator(
        exploration_rate=0.2,
        adaptation_rate=0.15
    )
    
    # 创建测试问题上下文
    problem_context = ProblemContext(
        problem_id="test_problem_001",
        problem_type=ProblemType.LOGICAL_REASONING,
        description="Given that all humans are mortal and Socrates is human, prove that Socrates is mortal.",
        complexity=0.3,
        uncertainty=0.1,
        time_constraint=0.2,
        resource_constraints={"compute": 0.3, "memory": 0.2},
        domain="philosophy_logic",
        features={}
    )
    
    # 推荐策略
    recommendation = evaluator.recommend_strategies(problem_context)
    
    print(f"策略推荐ID: {recommendation.recommendation_id}")
    print("推荐策略:")
    for strategy, score in recommendation.recommended_strategies[:3]:
        print(f"  - {strategy.value}: {score:.3f}")
    
    print("推理依据:")
    for rationale in recommendation.rationale[:3]:
        print(f"  - {rationale}")
    
    print(f"推荐置信度: {recommendation.confidence:.3f}")
    
    # 获取评估摘要
    summary = evaluator.get_evaluation_summary()
    print(f"\n评估摘要 - 总评估数: {summary['evaluation_statistics']['total_evaluations']}")
    print(f"总推荐数: {summary['evaluation_statistics']['total_recommendations']}")