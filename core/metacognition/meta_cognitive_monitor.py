"""
元认知监控系统核心

该模块实现AGI系统的元认知监控功能，包括：
1. 思维过程监控：追踪和分析思维活动过程
2. 认知偏差检测：识别思维中的各种认知偏差
3. 推理策略评估：评估推理策略的有效性和效率
4. 认知资源管理：管理和优化认知资源分配
5. 自我调节控制：根据监控结果调节认知过程

核心功能：
1. 实时思维追踪：记录思维活动的时序和内容
2. 偏差模式识别：检测常见认知偏差模式
3. 策略效果分析：分析不同推理策略的效果
4. 资源优化分配：动态调整认知资源分配
5. 反馈调节机制：提供实时反馈和调节建议

技术特性：
- 多层级思维活动建模
- 实时偏差检测算法
- 策略效果量化评估
- 动态资源分配优化
- 闭环反馈控制系统
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
from collections import deque

# 配置日志
logger = logging.getLogger(__name__)

class CognitiveActivityType(Enum):
    """认知活动类型"""
    PERCEPTION = "perception"
    ATTENTION = "attention"
    MEMORY_RETRIEVAL = "memory_retrieval"
    REASONING = "reasoning"
    PROBLEM_SOLVING = "problem_solving"
    DECISION_MAKING = "decision_making"
    LEARNING = "learning"
    PLANNING = "planning"
    METACOGNITION = "metacognition"

class CognitiveBiasType(Enum):
    """认知偏差类型"""
    CONFIRMATION_BIAS = "confirmation_bias"
    ANCHORING_BIAS = "anchoring_bias"
    AVAILABILITY_HEURISTIC = "availability_heuristic"
    REPRESENTATIVENESS_HEURISTIC = "representativeness_heuristic"
    OVERCONFIDENCE_BIAS = "overconfidence_bias"
    HINDSIGHT_BIAS = "hindsight_bias"
    FRAMING_EFFECT = "framing_effect"
    STATUS_QUO_BIAS = "status_quo_bias"
    SUNK_COST_FALLACY = "sunk_cost_fallacy"
    GAMBLERS_FALLACY = "gamblers_fallacy"

class ReasoningStrategyType(Enum):
    """推理策略类型"""
    DEDUCTIVE_REASONING = "deductive_reasoning"
    INDUCTIVE_REASONING = "inductive_reasoning"
    ABDUCTIVE_REASONING = "abductive_reasoning"
    ANALOGICAL_REASONING = "analogical_reasoning"
    CAUSAL_REASONING = "causal_reasoning"
    SYMBOLIC_REASONING = "symbolic_reasoning"
    PROBABILISTIC_REASONING = "probabilistic_reasoning"
    HEURISTIC_REASONING = "heuristic_reasoning"
    SYSTEMATIC_REASONING = "systematic_reasoning"
    INTUITIVE_REASONING = "intuitive_reasoning"

@dataclass
class ThoughtEvent:
    """思维事件"""
    event_id: str
    timestamp: datetime
    activity_type: CognitiveActivityType
    content: str
    context: Dict[str, Any]
    duration_ms: float
    cognitive_load: float  # 0-1
    confidence: float  # 0-1
    emotional_tone: Optional[str] = None
    related_events: List[str] = field(default_factory=list)

@dataclass
class BiasDetection:
    """偏差检测结果"""
    bias_id: str
    bias_type: CognitiveBiasType
    detected_at: datetime
    context: str
    evidence: List[str]
    confidence: float
    severity: float  # 0-1
    impact: float  # 0-1
    suggested_correction: str

@dataclass
class StrategyEvaluation:
    """策略评估结果"""
    strategy_id: str
    strategy_type: ReasoningStrategyType
    context: str
    evaluation_metrics: Dict[str, float]
    effectiveness_score: float  # 0-1
    efficiency_score: float  # 0-1
    robustness_score: float  # 0-1
    overall_score: float  # 0-1
    recommendations: List[str]
    evaluated_at: datetime

@dataclass
class CognitiveResource:
    """认知资源"""
    resource_type: str
    total_capacity: float
    current_usage: float
    allocation: Dict[str, float]  # 活动类型 -> 分配比例
    utilization_history: List[Tuple[datetime, float]] = field(default_factory=list)

@dataclass
class MetacognitiveState:
    """元认知状态"""
    timestamp: datetime
    thought_events_count: int
    active_biases: List[BiasDetection]
    current_strategy: Optional[StrategyEvaluation]
    resource_utilization: Dict[str, float]
    overall_cognitive_efficiency: float
    self_awareness_score: float
    regulation_effectiveness: float

class MetaCognitiveMonitor:
    """
    元认知监控系统核心
    
    核心组件:
    1. 思维追踪器: 记录和追踪思维活动
    2. 偏差检测器: 识别认知偏差和思维错误
    3. 策略评估器: 评估推理策略的效果
    4. 资源管理器: 管理和优化认知资源分配
    5. 调节控制器: 根据监控结果调节认知过程
    
    工作流程:
    思维活动 → 思维追踪器 → 记录事件 → 偏差检测器 → 检测偏差
    推理过程 → 策略评估器 → 评估策略 → 资源管理器 → 分配资源
    监控结果 → 调节控制器 → 生成调节 → 反馈到认知系统
    
    技术特性:
    - 多层级思维活动建模
    - 实时偏差检测算法
    - 策略效果量化评估
    - 动态资源分配优化
    - 闭环反馈控制系统
    """
    
    def __init__(self,
                 monitoring_interval_ms: float = 100.0,
                 max_thought_history: int = 1000,
                 bias_detection_enabled: bool = True,
                 strategy_evaluation_enabled: bool = True,
                 resource_optimization_enabled: bool = True):
        """
        初始化元认知监控系统
        
        Args:
            monitoring_interval_ms: 监控间隔（毫秒）
            max_thought_history: 最大思维历史记录
            bias_detection_enabled: 是否启用偏差检测
            strategy_evaluation_enabled: 是否启用策略评估
            resource_optimization_enabled: 是否启用资源优化
        """
        self.monitoring_interval_ms = monitoring_interval_ms
        self.max_thought_history = max_thought_history
        self.bias_detection_enabled = bias_detection_enabled
        self.strategy_evaluation_enabled = strategy_evaluation_enabled
        self.resource_optimization_enabled = resource_optimization_enabled
        
        # 思维活动记录
        self.thought_history: List[ThoughtEvent] = []
        self.thought_event_graph = nx.MultiDiGraph()
        
        # 偏差检测
        self.bias_detections: List[BiasDetection] = []
        self.bias_patterns: Dict[CognitiveBiasType, Dict[str, Any]] = {}
        self._initialize_bias_patterns()
        
        # 策略评估
        self.strategy_evaluations: List[StrategyEvaluation] = []
        self.current_strategy: Optional[StrategyEvaluation] = None
        
        # 认知资源管理
        self.cognitive_resources: Dict[str, CognitiveResource] = {}
        self._initialize_cognitive_resources()
        
        # 元认知状态
        self.metacognitive_states: List[MetacognitiveState] = []
        
        # 配置参数
        self.config = {
            'thought_sampling_rate': 0.1,
            'bias_detection_threshold': 0.7,
            'strategy_evaluation_interval': 60.0,  # 秒
            'resource_reallocation_threshold': 0.3,
            'cognitive_load_alert_threshold': 0.8,
            'confidence_calibration_factor': 0.8,
            'learning_rate': 0.1,
            'feedback_delay_ms': 50.0
        }
        
        # 性能统计
        self.performance_stats = {
            'thought_events_recorded': 0,
            'biases_detected': 0,
            'strategies_evaluated': 0,
            'resource_reallocations': 0,
            'cognitive_regulations': 0,
            'average_cognitive_efficiency': 0.0,
            'average_bias_severity': 0.0,
            'strategy_improvement_rate': 0.0
        }
        
        # 状态变量
        self.last_monitoring_time = time.time()
        self.system_start_time = time.time()
        self.thought_event_counter = 0
        
        logger.info(f"元认知监控系统初始化完成，监控间隔: {monitoring_interval_ms} 毫秒")
    
    def _initialize_bias_patterns(self):
        """初始化偏差模式"""
        self.bias_patterns = {
            CognitiveBiasType.CONFIRMATION_BIAS: {
                "description": "倾向于寻找、解释、记忆支持已有信念的信息",
                "detection_patterns": [
                    "selective_attention_to_confirming_evidence",
                    "disconfirmation_avoidance",
                    "biased_interpretation"
                ],
                "severity_weight": 0.8,
                "correction_strategy": "主动寻找反证，考虑多种可能性"
            },
            CognitiveBiasType.ANCHORING_BIAS: {
                "description": "过度依赖首次获得的信息（锚点）",
                "detection_patterns": [
                    "initial_value_overemphasis",
                    "insufficient_adjustment",
                    "primacy_effect"
                ],
                "severity_weight": 0.6,
                "correction_strategy": "考虑多个锚点，进行系统化调整"
            },
            CognitiveBiasType.AVAILABILITY_HEURISTIC: {
                "description": "基于容易想起的例子的频率判断概率",
                "detection_patterns": [
                    "recent_example_overweighting",
                    "vivid_example_preference",
                    "memory_based_frequency_estimation"
                ],
                "severity_weight": 0.5,
                "correction_strategy": "使用统计数据而非个人经验，考虑基础概率"
            },
            CognitiveBiasType.OVERCONFIDENCE_BIAS: {
                "description": "高估自己的知识、能力或判断准确性",
                "detection_patterns": [
                    "confidence_exceeds_accuracy",
                    "prediction_interval_too_narrow",
                    "calibration_mismatch"
                ],
                "severity_weight": 0.7,
                "correction_strategy": "进行概率校准，考虑不确定性，寻求反馈"
            }
        }
    
    def _initialize_cognitive_resources(self):
        """初始化认知资源"""
        resources = [
            CognitiveResource(
                resource_type="attention",
                total_capacity=1.0,
                current_usage=0.0,
                allocation={
                    CognitiveActivityType.PERCEPTION.value: 0.3,
                    CognitiveActivityType.REASONING.value: 0.4,
                    CognitiveActivityType.MEMORY_RETRIEVAL.value: 0.2,
                    CognitiveActivityType.METACOGNITION.value: 0.1
                }
            ),
            CognitiveResource(
                resource_type="working_memory",
                total_capacity=1.0,
                current_usage=0.0,
                allocation={
                    CognitiveActivityType.REASONING.value: 0.5,
                    CognitiveActivityType.PROBLEM_SOLVING.value: 0.3,
                    CognitiveActivityType.PLANNING.value: 0.2
                }
            ),
            CognitiveResource(
                resource_type="processing_power",
                total_capacity=1.0,
                current_usage=0.0,
                allocation={
                    CognitiveActivityType.REASONING.value: 0.4,
                    CognitiveActivityType.LEARNING.value: 0.3,
                    CognitiveActivityType.DECISION_MAKING.value: 0.3
                }
            )
        ]
        
        for resource in resources:
            self.cognitive_resources[resource.resource_type] = resource
    
    def record_thought_event(self,
                            activity_type: CognitiveActivityType,
                            content: str,
                            context: Dict[str, Any],
                            duration_ms: float,
                            cognitive_load: float,
                            confidence: float) -> ThoughtEvent:
        """记录思维事件"""
        event_id = f"thought_{self.thought_event_counter:06d}"
        self.thought_event_counter += 1
        
        event = ThoughtEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            activity_type=activity_type,
            content=content,
            context=context,
            duration_ms=duration_ms,
            cognitive_load=cognitive_load,
            confidence=confidence
        )
        
        # 添加到历史
        self.thought_history.append(event)
        if len(self.thought_history) > self.max_thought_history:
            self.thought_history.pop(0)
        
        # 添加到图
        self.thought_event_graph.add_node(event_id, **event.__dict__)
        
        # 添加边（与前一个事件的连接）
        if len(self.thought_history) > 1:
            prev_event = self.thought_history[-2]
            self.thought_event_graph.add_edge(
                prev_event.event_id,
                event_id,
                relation="temporal_sequence",
                time_gap_ms=(event.timestamp - prev_event.timestamp).total_seconds() * 1000
            )
        
        # 更新性能统计
        self.performance_stats['thought_events_recorded'] += 1
        
        # 检测偏差（如果启用）
        if self.bias_detection_enabled:
            biases = self.detect_cognitive_biases(event)
            if biases:
                self.bias_detections.extend(biases)
                self.performance_stats['biases_detected'] += len(biases)
        
        # 评估策略（如果启用且是推理活动）
        if (self.strategy_evaluation_enabled and 
            activity_type == CognitiveActivityType.REASONING):
            self.evaluate_reasoning_strategy(event)
        
        # 更新资源使用
        self.update_resource_usage(event, cognitive_load)
        
        return event
    
    def detect_cognitive_biases(self, thought_event: ThoughtEvent) -> List[BiasDetection]:
        """检测认知偏差"""
        detected_biases = []
        
        # 分析思维内容中的偏差模式
        thought_content = thought_event.content.lower()
        thought_context = thought_event.context
        
        # 检查确认偏差
        if self._check_confirmation_bias(thought_content, thought_context):
            bias = BiasDetection(
                bias_id=f"bias_{len(self.bias_detections) + 1:04d}",
                bias_type=CognitiveBiasType.CONFIRMATION_BIAS,
                detected_at=thought_event.timestamp,
                context=thought_event.content[:100],  # 截断
                evidence=[
                    "选择性关注确认性证据",
                    "忽视反证信息",
                    "有偏见的解释"
                ],
                confidence=0.75,
                severity=0.6,
                impact=0.5,
                suggested_correction="主动寻找反证，考虑多种可能性假设"
            )
            detected_biases.append(bias)
        
        # 检查锚定偏差
        if self._check_anchoring_bias(thought_content, thought_context):
            bias = BiasDetection(
                bias_id=f"bias_{len(self.bias_detections) + 1:04d}",
                bias_type=CognitiveBiasType.ANCHORING_BIAS,
                detected_at=thought_event.timestamp,
                context=thought_event.content[:100],
                evidence=[
                    "过度依赖初始信息",
                    "调整不足",
                    "首因效应"
                ],
                confidence=0.7,
                severity=0.5,
                impact=0.4,
                suggested_correction="考虑多个参考点，进行系统化调整"
            )
            detected_biases.append(bias)
        
        # 检查过度自信偏差
        if self._check_overconfidence_bias(thought_event.confidence, thought_context):
            bias = BiasDetection(
                bias_id=f"bias_{len(self.bias_detections) + 1:04d}",
                bias_type=CognitiveBiasType.OVERCONFIDENCE_BIAS,
                detected_at=thought_event.timestamp,
                context=thought_event.content[:100],
                evidence=[
                    f"置信度({thought_event.confidence:.2f})可能过高",
                    "预测区间过窄",
                    "校准不匹配"
                ],
                confidence=0.8,
                severity=0.7,
                impact=0.6,
                suggested_correction="进行概率校准，考虑不确定性范围"
            )
            detected_biases.append(bias)
        
        return detected_biases
    
    def _check_confirmation_bias(self, content: str, context: Dict[str, Any]) -> bool:
        """检查确认偏差"""
        # 简单启发式检查
        confirmation_keywords = ["confirm", "prove", "support", "evidence for", "consistent with"]
        disconfirmation_keywords = ["disprove", "contradict", "against", "evidence against"]
        
        confirmation_count = sum(1 for keyword in confirmation_keywords if keyword in content)
        disconfirmation_count = sum(1 for keyword in disconfirmation_keywords if keyword in content)
        
        # 如果确认性词语远多于否定性词语，可能存在确认偏差
        bias_ratio = confirmation_count / (disconfirmation_count + 1)  # 避免除零
        
        return bias_ratio > 3.0  # 阈值
    
    def _check_anchoring_bias(self, content: str, context: Dict[str, Any]) -> bool:
        """检查锚定偏差"""
        # 检查是否过度引用初始值
        if "initial" in content or "first" in content or "original" in content:
            # 检查是否有调整不足的迹象
            adjustment_keywords = ["adjust", "revise", "update", "correct"]
            adjustment_count = sum(1 for keyword in adjustment_keywords if keyword in content)
            
            # 如果提到初始值但很少调整，可能存在锚定偏差
            return adjustment_count < 2
        
        return False
    
    def _check_overconfidence_bias(self, confidence: float, context: Dict[str, Any]) -> bool:
        """检查过度自信偏差"""
        # 检查置信度是否过高
        if confidence > 0.95:
            # 检查是否有校准证据
            calibration_evidence = context.get("calibration_evidence", False)
            
            # 如果没有校准证据且置信度极高，可能存在过度自信
            return not calibration_evidence
        
        return False
    
    def evaluate_reasoning_strategy(self, thought_event: ThoughtEvent) -> Optional[StrategyEvaluation]:
        """评估推理策略"""
        if not self.strategy_evaluation_enabled:
            return None
        
        # 从上下文中提取策略信息
        context = thought_event.context
        strategy_type_str = context.get("reasoning_strategy", "deductive_reasoning")
        
        try:
            strategy_type = ReasoningStrategyType(strategy_type_str)
        except ValueError:
            strategy_type = ReasoningStrategyType.DEDUCTIVE_REASONING
        
        # 评估指标
        evaluation_metrics = {
            "accuracy": context.get("accuracy", 0.8),
            "efficiency": thought_event.duration_ms / 1000.0,  # 转换为秒
            "resource_usage": thought_event.cognitive_load,
            "robustness": context.get("robustness", 0.7),
            "generalizability": context.get("generalizability", 0.6)
        }
        
        # 计算得分
        effectiveness_score = evaluation_metrics["accuracy"] * evaluation_metrics["robustness"]
        efficiency_score = 1.0 / (evaluation_metrics["efficiency"] + 0.1)  # 避免除零
        efficiency_score = min(1.0, efficiency_score)  # 标准化
        robustness_score = evaluation_metrics["robustness"] * evaluation_metrics["generalizability"]
        
        overall_score = (
            effectiveness_score * 0.4 +
            efficiency_score * 0.3 +
            robustness_score * 0.3
        )
        
        # 生成建议
        recommendations = []
        if effectiveness_score < 0.7:
            recommendations.append("考虑使用更可靠的推理策略")
        if efficiency_score < 0.5:
            recommendations.append("优化推理过程以提高效率")
        if robustness_score < 0.6:
            recommendations.append("增强推理策略的鲁棒性和泛化能力")
        
        # 创建评估结果
        strategy_id = f"strategy_{len(self.strategy_evaluations) + 1:04d}"
        evaluation = StrategyEvaluation(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            context=thought_event.content[:200],
            evaluation_metrics=evaluation_metrics,
            effectiveness_score=effectiveness_score,
            efficiency_score=efficiency_score,
            robustness_score=robustness_score,
            overall_score=overall_score,
            recommendations=recommendations,
            evaluated_at=thought_event.timestamp
        )
        
        # 保存评估
        self.strategy_evaluations.append(evaluation)
        self.current_strategy = evaluation
        
        # 更新统计
        self.performance_stats['strategies_evaluated'] += 1
        
        # 计算策略改进率
        if len(self.strategy_evaluations) > 1:
            prev_score = self.strategy_evaluations[-2].overall_score
            curr_score = overall_score
            improvement = curr_score - prev_score
            self.performance_stats['strategy_improvement_rate'] = (
                self.performance_stats['strategy_improvement_rate'] * 0.9 + improvement * 0.1
            )
        
        return evaluation
    
    def update_resource_usage(self, thought_event: ThoughtEvent, cognitive_load: float):
        """更新资源使用"""
        activity_type = thought_event.activity_type.value
        
        for resource_name, resource in self.cognitive_resources.items():
            # 获取该活动类型的分配比例
            allocation = resource.allocation.get(activity_type, 0.0)
            
            # 更新使用量
            usage_increase = cognitive_load * allocation
            resource.current_usage += usage_increase
            
            # 记录使用历史
            resource.utilization_history.append((thought_event.timestamp, resource.current_usage))
            if len(resource.utilization_history) > 100:
                resource.utilization_history.pop(0)
            
            # 检查是否需要重新分配资源
            if (self.resource_optimization_enabled and 
                resource.current_usage > resource.total_capacity * self.config['resource_reallocation_threshold']):
                self.optimize_resource_allocation()
                self.performance_stats['resource_reallocations'] += 1
    
    def optimize_resource_allocation(self):
        """优化资源分配"""
        # 分析当前使用模式
        usage_patterns = {}
        for resource_name, resource in self.cognitive_resources.items():
            recent_usage = [u for _, u in resource.utilization_history[-10:]] if resource.utilization_history else [0.0]
            avg_usage = np.mean(recent_usage) if recent_usage else 0.0
            usage_patterns[resource_name] = avg_usage
        
        # 重新分配资源（简化版本）
        for resource_name, resource in self.cognitive_resources.items():
            # 减少低效活动的资源分配，增加高效活动的分配
            current_allocation = resource.allocation.copy()
            
            # 基于策略评估结果调整
            if self.current_strategy:
                strategy_type = self.current_strategy.strategy_type.value
                if strategy_type in current_allocation:
                    # 如果当前策略有效，增加其资源分配
                    strategy_score = self.current_strategy.overall_score
                    adjustment = 0.1 * strategy_score  # 根据得分调整
                    current_allocation[strategy_type] = min(1.0, current_allocation[strategy_type] + adjustment)
            
            # 重新归一化
            total = sum(current_allocation.values())
            if total > 0:
                for key in current_allocation:
                    current_allocation[key] /= total
            
            resource.allocation = current_allocation
    
    def regulate_cognitive_process(self) -> Dict[str, Any]:
        """调节认知过程"""
        regulations = {
            "timestamp": datetime.now(),
            "adjustments": [],
            "warnings": [],
            "recommendations": []
        }
        
        # 检查认知负载
        overall_load = self._calculate_overall_cognitive_load()
        if overall_load > self.config['cognitive_load_alert_threshold']:
            warning = f"认知负载过高: {overall_load:.2f}，建议减少并发任务"
            regulations["warnings"].append(warning)
            regulations["recommendations"].append("优先处理高优先级任务，推迟低优先级任务")
        
        # 检查偏差影响
        recent_biases = [b for b in self.bias_detections[-5:]] if self.bias_detections else []
        if recent_biases:
            avg_severity = np.mean([b.severity for b in recent_biases])
            if avg_severity > 0.5:
                regulation = "检测到显著认知偏差，建议启用偏差校正机制"
                regulations["adjustments"].append({
                    "type": "bias_correction",
                    "description": "启用偏差校正",
                    "intensity": avg_severity
                })
                regulations["recommendations"].append("使用结构化决策框架，寻求外部验证")
        
        # 检查策略效果
        if self.current_strategy and self.current_strategy.overall_score < 0.6:
            recommendation = f"当前推理策略效果不佳(得分: {self.current_strategy.overall_score:.2f})，建议切换策略"
            regulations["recommendations"].append(recommendation)
            
            # 建议替代策略
            alternative_strategies = self._suggest_alternative_strategies()
            if alternative_strategies:
                regulations["recommendations"].append(f"可考虑的策略: {', '.join(alternative_strategies[:3])}")
        
        # 应用调节
        if regulations["adjustments"] or regulations["warnings"]:
            self.performance_stats['cognitive_regulations'] += 1
        
        return regulations
    
    def _calculate_overall_cognitive_load(self) -> float:
        """计算总体认知负载"""
        if not self.cognitive_resources:
            return 0.0
        
        loads = []
        for resource_name, resource in self.cognitive_resources.items():
            if resource.total_capacity > 0:
                load = resource.current_usage / resource.total_capacity
                loads.append(load)
        
        return np.mean(loads) if loads else 0.0
    
    def _suggest_alternative_strategies(self) -> List[str]:
        """建议替代策略"""
        # 基于历史评估建议更好的策略
        if len(self.strategy_evaluations) < 3:
            return ["deductive_reasoning", "probabilistic_reasoning", "systematic_reasoning"]
        
        # 找出历史上最有效的策略
        strategy_scores = {}
        for evaluation in self.strategy_evaluations[-10:]:  # 最近10次评估
            strategy_type = evaluation.strategy_type.value
            if strategy_type not in strategy_scores:
                strategy_scores[strategy_type] = []
            strategy_scores[strategy_type].append(evaluation.overall_score)
        
        # 计算平均得分
        avg_scores = {}
        for strategy_type, scores in strategy_scores.items():
            avg_scores[strategy_type] = np.mean(scores)
        
        # 按得分排序
        sorted_strategies = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 返回前3个策略
        return [strategy for strategy, score in sorted_strategies[:3]]
    
    def get_metacognitive_state(self) -> MetacognitiveState:
        """获取当前元认知状态"""
        # 计算认知效率
        recent_events = self.thought_history[-10:] if self.thought_history else []
        if recent_events:
            avg_confidence = np.mean([e.confidence for e in recent_events])
            avg_load = np.mean([e.cognitive_load for e in recent_events])
            cognitive_efficiency = avg_confidence / (avg_load + 0.1)  # 避免除零
            cognitive_efficiency = min(1.0, cognitive_efficiency)
        else:
            cognitive_efficiency = 0.5
        
        # 计算自我意识得分
        self_awareness = 0.5
        if self.bias_detections:
            # 偏差检测率反映自我意识
            detection_rate = len(self.bias_detections) / max(1, self.performance_stats['thought_events_recorded'])
            self_awareness = min(1.0, detection_rate * 2.0)
        
        # 计算调节效果
        regulation_effectiveness = 0.5
        if self.performance_stats['cognitive_regulations'] > 0:
            # 基于偏差减少评估调节效果
            if len(self.bias_detections) > 10:
                early_biases = self.bias_detections[:5]
                recent_biases = self.bias_detections[-5:]
                early_severity = np.mean([b.severity for b in early_biases])
                recent_severity = np.mean([b.severity for b in recent_biases])
                if early_severity > 0:
                    improvement = (early_severity - recent_severity) / early_severity
                    regulation_effectiveness = max(0.0, min(1.0, 0.5 + improvement))
        
        # 资源利用率
        resource_utilization = {}
        for resource_name, resource in self.cognitive_resources.items():
            if resource.total_capacity > 0:
                resource_utilization[resource_name] = resource.current_usage / resource.total_capacity
        
        state = MetacognitiveState(
            timestamp=datetime.now(),
            thought_events_count=len(self.thought_history),
            active_biases=self.bias_detections[-5:] if self.bias_detections else [],
            current_strategy=self.current_strategy,
            resource_utilization=resource_utilization,
            overall_cognitive_efficiency=cognitive_efficiency,
            self_awareness_score=self_awareness,
            regulation_effectiveness=regulation_effectiveness
        )
        
        # 保存状态历史
        self.metacognitive_states.append(state)
        if len(self.metacognitive_states) > 100:
            self.metacognitive_states.pop(0)
        
        return state
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """获取监控仪表板"""
        current_state = self.get_metacognitive_state()
        
        dashboard = {
            "timestamp": datetime.now(),
            "system_uptime_hours": (time.time() - self.system_start_time) / 3600,
            "current_state": {
                "cognitive_efficiency": current_state.overall_cognitive_efficiency,
                "self_awareness": current_state.self_awareness_score,
                "regulation_effectiveness": current_state.regulation_effectiveness,
                "thought_events_count": current_state.thought_events_count,
                "active_biases_count": len(current_state.active_biases),
                "current_strategy": current_state.current_strategy.strategy_type.value if current_state.current_strategy else None
            },
            "performance_stats": self.performance_stats,
            "resource_utilization": current_state.resource_utilization,
            "recent_biases": [
                {
                    "type": bias.bias_type.value,
                    "severity": bias.severity,
                    "detected_at": bias.detected_at.isoformat()
                }
                for bias in current_state.active_biases
            ],
            "cognitive_trends": self._calculate_cognitive_trends()
        }
        
        return dashboard
    
    def _calculate_cognitive_trends(self) -> Dict[str, float]:
        """计算认知趋势"""
        trends = {}
        
        # 分析最近的状态变化
        if len(self.metacognitive_states) >= 5:
            recent_states = self.metacognitive_states[-5:]
            
            # 认知效率趋势
            efficiency_values = [s.overall_cognitive_efficiency for s in recent_states]
            trends["efficiency_trend"] = self._calculate_trend(efficiency_values)
            
            # 自我意识趋势
            awareness_values = [s.self_awareness_score for s in recent_states]
            trends["awareness_trend"] = self._calculate_trend(awareness_values)
            
            # 调节效果趋势
            regulation_values = [s.regulation_effectiveness for s in recent_states]
            trends["regulation_trend"] = self._calculate_trend(regulation_values)
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> float:
        """计算趋势（斜率）"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        try:
            slope, _ = np.polyfit(x, y, 1)
            return slope
        except:
            return 0.0

# 全局实例
meta_cognitive_monitor_instance = MetaCognitiveMonitor()

if __name__ == "__main__":
    # 测试元认知监控系统
    print("测试元认知监控系统...")
    
    monitor = MetaCognitiveMonitor(
        monitoring_interval_ms=50.0,
        bias_detection_enabled=True,
        strategy_evaluation_enabled=True
    )
    
    # 记录一些思维事件
    monitor.record_thought_event(
        activity_type=CognitiveActivityType.REASONING,
        content="基于已有数据推断因果关系",
        context={
            "reasoning_strategy": "causal_reasoning",
            "accuracy": 0.85,
            "robustness": 0.7,
            "calibration_evidence": True
        },
        duration_ms=150.0,
        cognitive_load=0.6,
        confidence=0.9
    )
    
    monitor.record_thought_event(
        activity_type=CognitiveActivityType.DECISION_MAKING,
        content="选择最优解决方案，这个方案肯定是最好的",
        context={},
        duration_ms=100.0,
        cognitive_load=0.5,
        confidence=0.95  # 可能过度自信
    )
    
    # 获取元认知状态
    state = monitor.get_metacognitive_state()
    print(f"元认知状态 - 认知效率: {state.overall_cognitive_efficiency:.3f}")
    print(f"自我意识得分: {state.self_awareness_score:.3f}")
    
    # 获取监控仪表板
    dashboard = monitor.get_monitoring_dashboard()
    print(f"监控仪表板 - 偏差检测数: {dashboard['performance_stats']['biases_detected']}")
    
    # 调节认知过程
    regulations = monitor.regulate_cognitive_process()
    if regulations["warnings"] or regulations["recommendations"]:
        print("认知调节建议:")
        for rec in regulations["recommendations"]:
            print(f"  - {rec}")