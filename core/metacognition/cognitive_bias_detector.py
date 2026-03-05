"""
认知偏差检测器

该模块实现AGI系统的认知偏差检测功能，包括：
1. 偏差模式识别：识别常见的认知偏差模式
2. 偏差严重性评估：评估偏差的严重程度和影响
3. 偏差根源分析：分析偏差产生的根本原因
4. 偏差校正建议：提供针对性的校正建议
5. 偏差趋势监控：监控偏差发生趋势和变化

核心功能：
1. 多类型偏差检测算法
2. 模式匹配和机器学习检测
3. 严重性量化和风险评估
4. 校正策略生成
5. 趋势分析和预警

技术特性：
- 基于规则的偏差检测
- 机器学习模式识别
- 多层次严重性评估
- 个性化校正建议
- 实时趋势监控
"""

import time
import logging
import re
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
import random

# 配置日志
logger = logging.getLogger(__name__)

class BiasSeverity(Enum):
    """偏差严重性"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class BiasCategory(Enum):
    """偏差类别"""
    INFORMATION_PROCESSING = "information_processing"
    JUDGMENT_DECISION = "judgment_decision"
    MEMORY_RECALL = "memory_recall"
    SOCIAL_COGNITIVE = "social_cognitive"
    MOTIVATIONAL = "motivational"

@dataclass
class BiasPattern:
    """偏差模式"""
    pattern_id: str
    bias_type: str
    description: str
    detection_rules: List[Dict[str, Any]]
    severity_indicators: Dict[str, float]
    correction_strategies: List[str]
    common_triggers: List[str]
    false_positive_filters: List[Dict[str, Any]]

@dataclass
class BiasDetectionResult:
    """偏差检测结果"""
    detection_id: str
    bias_type: str
    detected_at: datetime
    context: str
    evidence: List[str]
    confidence: float
    severity: BiasSeverity
    severity_score: float  # 0-1
    impact_score: float  # 0-1
    root_causes: List[str]
    suggested_corrections: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BiasStatistics:
    """偏差统计"""
    total_detections: int = 0
    detections_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    detections_by_severity: Dict[BiasSeverity, int] = field(default_factory=lambda: defaultdict(int))
    average_confidence: float = 0.0
    average_severity_score: float = 0.0
    detection_rate_per_hour: float = 0.0
    correction_success_rate: float = 0.0

@dataclass
class BiasTrend:
    """偏差趋势"""
    timestamp: datetime
    bias_types: Dict[str, int]
    severity_distribution: Dict[BiasSeverity, float]
    overall_trend: float  # -1到1，负值表示改善
    alert_level: str

class CognitiveBiasDetector:
    """
    认知偏差检测器
    
    核心组件:
    1. 模式匹配器: 匹配已知偏差模式
    2. 机器学习检测器: 使用ML模型检测偏差
    3. 严重性评估器: 评估偏差严重性和影响
    4. 根源分析器: 分析偏差产生的根源
    5. 校正建议器: 提供针对性校正建议
    6. 趋势分析器: 分析偏差趋势和预警
    
    工作流程:
    思维内容 → 模式匹配器 → 匹配模式 → 机器学习检测器 → ML检测
    检测结果 → 严重性评估器 → 评估严重性 → 根源分析器 → 分析根源
    综合分析 → 校正建议器 → 提供建议 → 趋势分析器 → 分析趋势
    
    技术特性:
    - 多层次检测框架
    - 多类型偏差覆盖
    - 严重性量化评估
    - 根源追溯分析
    - 趋势预警系统
    """
    
    def __init__(self,
                 detection_threshold: float = 0.7,
                 min_confidence: float = 0.6,
                 trend_window_hours: float = 24.0,
                 alert_threshold: float = 0.8,
                 ml_detection_enabled: bool = True):
        """
        初始化认知偏差检测器
        
        Args:
            detection_threshold: 检测阈值
            min_confidence: 最小置信度
            trend_window_hours: 趋势分析时间窗口（小时）
            alert_threshold: 预警阈值
            ml_detection_enabled: 是否启用机器学习检测
        """
        self.detection_threshold = detection_threshold
        self.min_confidence = min_confidence
        self.trend_window_hours = trend_window_hours
        self.alert_threshold = alert_threshold
        self.ml_detection_enabled = ml_detection_enabled
        
        # 偏差模式库
        self.bias_patterns: Dict[str, BiasPattern] = {}
        self._initialize_bias_patterns()
        
        # 检测结果
        self.bias_detections: List[BiasDetectionResult] = []
        
        # 校正历史
        self.correction_history: List[Dict[str, Any]] = []
        
        # 趋势数据
        self.trend_history: List[BiasTrend] = []
        
        # 统计信息
        self.statistics = BiasStatistics()
        
        # 配置参数
        self.config = {
            'confidence_decay_rate': 0.95,
            'severity_weight_factors': {
                'frequency': 0.3,
                'impact': 0.4,
                'persistence': 0.2,
                'corrigibility': 0.1
            },
            'pattern_matching_weight': 0.6,
            'ml_detection_weight': 0.4,
            'trend_smoothing_factor': 0.1,
            'alert_cooldown_hours': 1.0,
            'min_evidence_count': 2,
            'max_root_causes': 3
        }
        
        # 机器学习模型（简化版）
        self.ml_models: Dict[str, Any] = {}
        if self.ml_detection_enabled:
            self._initialize_ml_models()
        
        # 性能统计
        self.performance_stats = {
            'detections_attempted': 0,
            'detections_successful': 0,
            'false_positives': 0,
            'corrections_suggested': 0,
            'corrections_applied': 0,
            'alerts_triggered': 0,
            'average_detection_time_ms': 0.0,
            'detection_accuracy': 0.0
        }
        
        # 状态变量
        self.last_detection_time = time.time()
        self.system_start_time = time.time()
        self.detection_counter = 0
        self.last_alert_time = time.time() - 3600  # 1小时前
        
        logger.info(f"认知偏差检测器初始化完成，检测阈值: {detection_threshold}")
    
    def _initialize_bias_patterns(self):
        """初始化偏差模式"""
        patterns = [
            # 确认偏差
            BiasPattern(
                pattern_id="confirmation_bias_001",
                bias_type="confirmation_bias",
                description="倾向于寻找、解释、记忆支持已有信念的信息",
                detection_rules=[
                    {
                        "rule_type": "keyword_pattern",
                        "keywords": ["confirm", "prove", "support", "evidence for", "consistent with"],
                        "weight": 0.3
                    },
                    {
                        "rule_type": "negation_pattern",
                        "keywords": ["disprove", "contradict", "against", "evidence against"],
                        "negation_weight": -0.2
                    },
                    {
                        "rule_type": "selectivity_ratio",
                        "confirming_keywords": ["confirm", "support", "agree"],
                        "disconfirming_keywords": ["disprove", "refute", "contradict"],
                        "threshold": 3.0
                    }
                ],
                severity_indicators={
                    "selectivity_ratio": 0.7,
                    "confidence_discrepancy": 0.6,
                    "evidence_ignored_count": 0.5
                },
                correction_strategies=[
                    "主动寻找反证信息",
                    "考虑多种可能性假设",
                    "使用反事实思考"
                ],
                common_triggers=[
                    "强烈先验信念",
                    "情感投入议题",
                    "群体认同压力"
                ],
                false_positive_filters=[
                    {
                        "filter_type": "context_aware",
                        "context_keywords": ["scientific", "experimental", "testing"],
                        "weight_adjustment": -0.3
                    }
                ]
            ),
            
            # 锚定偏差
            BiasPattern(
                pattern_id="anchoring_bias_001",
                bias_type="anchoring_bias",
                description="过度依赖首次获得的信息（锚点）",
                detection_rules=[
                    {
                        "rule_type": "numeric_anchoring",
                        "pattern": r"initial\s+value.*?\$\d+",
                        "weight": 0.4
                    },
                    {
                        "rule_type": "adjustment_insufficiency",
                        "keywords": ["adjust", "revise", "update"],
                        "adjustment_ratio_threshold": 0.3
                    },
                    {
                        "rule_type": "primacy_effect",
                        "first_mention_weight": 0.5,
                        "subsequent_mentions_weight": 0.2
                    }
                ],
                severity_indicators={
                    "anchoring_strength": 0.8,
                    "adjustment_ratio": 0.4,
                    "range_narrowing": 0.6
                },
                correction_strategies=[
                    "考虑多个参考锚点",
                    "进行系统化调整",
                    "使用德尔菲法或群体估计"
                ],
                common_triggers=[
                    "缺乏领域知识",
                    "时间压力",
                    "信息过载"
                ],
                false_positive_filters=[
                    {
                        "filter_type": "deliberate_anchoring",
                        "context_keywords": ["anchoring", "reference point", "baseline"],
                        "weight_adjustment": -0.4
                    }
                ]
            ),
            
            # 过度自信偏差
            BiasPattern(
                pattern_id="overconfidence_bias_001",
                bias_type="overconfidence_bias",
                description="高估自己的知识、能力或判断准确性",
                detection_rules=[
                    {
                        "rule_type": "confidence_calibration",
                        "confidence_threshold": 0.95,
                        "calibration_evidence_required": True
                    },
                    {
                        "rule_type": "prediction_interval",
                        "interval_narrowness_threshold": 0.3
                    },
                    {
                        "rule_type": "certainty_language",
                        "keywords": ["certain", "sure", "definitely", "absolutely"],
                        "weight": 0.35
                    }
                ],
                severity_indicators={
                    "confidence_accuracy_gap": 0.7,
                    "prediction_interval_width": 0.3,
                    "calibration_mismatch": 0.8
                },
                correction_strategies=[
                    "进行概率校准训练",
                    "考虑不确定性范围",
                    "寻求外部反馈和验证"
                ],
                common_triggers=[
                    "专业知识领域",
                    "近期成功经验",
                    "缺乏质量反馈"
                ],
                false_positive_filters=[
                    {
                        "filter_type": "justified_confidence",
                        "evidence_keywords": ["data", "evidence", "research", "statistics"],
                        "weight_adjustment": -0.2
                    }
                ]
            ),
            
            # 可用性启发式
            BiasPattern(
                pattern_id="availability_heuristic_001",
                bias_type="availability_heuristic",
                description="基于容易想起的例子的频率判断概率",
                detection_rules=[
                    {
                        "rule_type": "vivid_example_preference",
                        "keywords": ["recent", "memorable", "vivid", "dramatic"],
                        "weight": 0.3
                    },
                    {
                        "rule_type": "base_rate_neglect",
                        "base_rate_mention_required": True,
                        "neglect_threshold": 0.7
                    },
                    {
                        "rule_type": "media_exposure_influence",
                        "media_related_keywords": ["news", "media", "reported", "headline"],
                        "weight": 0.25
                    }
                ],
                severity_indicators={
                    "vividness_factor": 0.6,
                    "base_rate_disregard": 0.7,
                    "recency_weight": 0.5
                },
                correction_strategies=[
                    "使用统计数据而非个人经验",
                    "考虑基础概率",
                    "系统化收集信息"
                ],
                common_triggers=[
                    "媒体过度报道",
                    "个人强烈经历",
                    "情感唤起事件"
                ],
                false_positive_filters=[
                    {
                        "filter_type": "statistical_context",
                        "context_keywords": ["statistical", "probability", "frequency"],
                        "weight_adjustment": -0.3
                    }
                ]
            ),
            
            # 沉没成本谬误
            BiasPattern(
                pattern_id="sunk_cost_fallacy_001",
                bias_type="sunk_cost_fallacy",
                description="因已投入成本而继续投入，不顾未来收益",
                detection_rules=[
                    {
                        "rule_type": "past_investment_focus",
                        "keywords": ["already invested", "sunk cost", "too much to lose"],
                        "weight": 0.4
                    },
                    {
                        "rule_type": "future_benefit_neglect",
                        "future_keywords": ["future", "prospective", "going forward"],
                        "neglect_threshold": 0.6
                    },
                    {
                        "rule_type": "emotional_attachment",
                        "keywords": ["attachment", "emotional", "personal"],
                        "weight": 0.3
                    }
                ],
                severity_indicators={
                    "investment_focus_ratio": 0.8,
                    "future_benefit_neglect": 0.7,
                    "emotional_involvement": 0.6
                },
                correction_strategies=[
                    "忽略沉没成本，关注未来收益",
                    "进行边际分析",
                    "寻求外部客观意见"
                ],
                common_triggers=[
                    "重大前期投入",
                    "个人认同项目",
                    "公开承诺压力"
                ],
                false_positive_filters=[
                    {
                        "filter_type": "strategic_persistence",
                        "context_keywords": ["strategic", "long-term", "persistence"],
                        "weight_adjustment": -0.2
                    }
                ]
            )
        ]
        
        for pattern in patterns:
            self.bias_patterns[pattern.pattern_id] = pattern
    
    def _initialize_ml_models(self):
        """初始化机器学习模型"""
        # 简化版ML模型 - 实际项目中应使用真实ML模型
        logger.info("初始化简化ML检测模型")
        
        self.ml_models = {
            "confirmation_bias": {
                "features": ["keyword_ratio", "negation_count", "certainty_score"],
                "weights": [0.4, 0.3, 0.3],
                "threshold": 0.65
            },
            "overconfidence_bias": {
                "features": ["confidence_score", "calibration_gap", "certainty_words"],
                "weights": [0.5, 0.3, 0.2],
                "threshold": 0.7
            },
            "availability_heuristic": {
                "features": ["vivid_word_count", "recency_emphasis", "statistical_reference"],
                "weights": [0.4, 0.4, 0.2],
                "threshold": 0.6
            }
        }
    
    def detect_biases(self, 
                     content: str, 
                     context: Dict[str, Any]) -> List[BiasDetectionResult]:
        """检测认知偏差"""
        start_time = time.time()
        self.performance_stats['detections_attempted'] += 1
        
        all_detections = []
        
        # 1. 基于规则的检测
        rule_based_detections = self._rule_based_detection(content, context)
        all_detections.extend(rule_based_detections)
        
        # 2. 机器学习检测（如果启用）
        if self.ml_detection_enabled:
            ml_detections = self._ml_based_detection(content, context)
            all_detections.extend(ml_detections)
        
        # 3. 合并和过滤重复检测
        filtered_detections = self._filter_and_merge_detections(all_detections)
        
        # 4. 评估严重性
        for detection in filtered_detections:
            self._assess_severity(detection, context)
        
        # 5. 分析根源
        for detection in filtered_detections:
            self._analyze_root_causes(detection, context)
        
        # 6. 生成校正建议
        for detection in filtered_detections:
            self._generate_corrections(detection)
        
        # 保存检测结果
        for detection in filtered_detections:
            self.bias_detections.append(detection)
            
            # 更新统计
            self.statistics.total_detections += 1
            self.statistics.detections_by_type[detection.bias_type] += 1
            self.statistics.detections_by_severity[detection.severity] += 1
            
            # 更新平均置信度和严重性得分
            n = self.statistics.total_detections
            self.statistics.average_confidence = (
                (self.statistics.average_confidence * (n - 1) + detection.confidence) / n
            )
            self.statistics.average_severity_score = (
                (self.statistics.average_severity_score * (n - 1) + detection.severity_score) / n
            )
        
        # 更新性能统计
        detection_count = len(filtered_detections)
        if detection_count > 0:
            self.performance_stats['detections_successful'] += detection_count
            self.performance_stats['corrections_suggested'] += sum(
                len(d.suggested_corrections) for d in filtered_detections
            )
        
        # 计算检测时间
        detection_time_ms = (time.time() - start_time) * 1000
        n_detections = self.performance_stats['detections_attempted']
        self.performance_stats['average_detection_time_ms'] = (
            (self.performance_stats['average_detection_time_ms'] * (n_detections - 1) + detection_time_ms) / n_detections
        )
        
        # 更新检测率
        time_since_start = (time.time() - self.system_start_time) / 3600
        if time_since_start > 0:
            self.statistics.detection_rate_per_hour = self.statistics.total_detections / time_since_start
        
        self.last_detection_time = time.time()
        
        logger.info(f"偏差检测完成: 发现{len(filtered_detections)}个偏差")
        return filtered_detections
    
    def _rule_based_detection(self, content: str, context: Dict[str, Any]) -> List[BiasDetectionResult]:
        """基于规则的检测"""
        detections = []
        
        for pattern_id, pattern in self.bias_patterns.items():
            detection_score = 0.0
            evidence = []
            
            # 应用检测规则
            for rule in pattern.detection_rules:
                rule_score = self._apply_detection_rule(rule, content, context)
                detection_score += rule_score
                
                if rule_score > 0.1:  # 显著贡献
                    evidence.append(f"规则匹配: {rule.get('rule_type', 'unknown')}, 得分: {rule_score:.2f}")
            
            # 应用误报过滤器
            for filter_rule in pattern.false_positive_filters:
                filter_adjustment = self._apply_false_positive_filter(filter_rule, content, context)
                detection_score += filter_adjustment
            
            # 检查是否达到检测阈值
            if detection_score >= self.detection_threshold:
                # 计算置信度
                confidence = min(1.0, detection_score * 1.2)  # 调整置信度
                
                # 创建检测结果
                detection = BiasDetectionResult(
                    detection_id=f"detection_{self.detection_counter:06d}",
                    bias_type=pattern.bias_type,
                    detected_at=datetime.now(),
                    context=content[:200],  # 截断
                    evidence=evidence,
                    confidence=confidence,
                    severity=BiasSeverity.LOW,  # 稍后评估
                    severity_score=0.0,
                    impact_score=0.0,
                    root_causes=[],
                    suggested_corrections=[]
                )
                
                detections.append(detection)
                self.detection_counter += 1
        
        return detections
    
    def _apply_detection_rule(self, rule: Dict[str, Any], content: str, context: Dict[str, Any]) -> float:
        """应用检测规则"""
        rule_type = rule.get("rule_type", "")
        
        if rule_type == "keyword_pattern":
            keywords = rule.get("keywords", [])
            weight = rule.get("weight", 0.3)
            
            score = 0.0
            content_lower = content.lower()
            for keyword in keywords:
                if keyword in content_lower:
                    score += weight
            
            return min(1.0, score)
        
        elif rule_type == "negation_pattern":
            keywords = rule.get("keywords", [])
            negation_weight = rule.get("negation_weight", -0.2)
            
            content_lower = content.lower()
            for keyword in keywords:
                if keyword in content_lower:
                    return negation_weight
            
            return 0.0
        
        elif rule_type == "selectivity_ratio":
            confirming = rule.get("confirming_keywords", [])
            disconfirming = rule.get("disconfirming_keywords", [])
            threshold = rule.get("threshold", 3.0)
            
            content_lower = content.lower()
            confirm_count = sum(1 for kw in confirming if kw in content_lower)
            disconfirm_count = sum(1 for kw in disconfirming if kw in content_lower)
            
            if disconfirm_count > 0:
                ratio = confirm_count / disconfirm_count
                if ratio > threshold:
                    return min(1.0, (ratio - threshold) / threshold)
            
            return 0.0
        
        elif rule_type == "confidence_calibration":
            confidence_threshold = rule.get("confidence_threshold", 0.95)
            calibration_required = rule.get("calibration_evidence_required", True)
            
            confidence = context.get("confidence", 0.5)
            calibration_evidence = context.get("calibration_evidence", False)
            
            if confidence > confidence_threshold:
                if calibration_required and not calibration_evidence:
                    return min(1.0, (confidence - confidence_threshold) / (1.0 - confidence_threshold))
                elif not calibration_required:
                    return min(1.0, (confidence - confidence_threshold) / (1.0 - confidence_threshold))
            
            return 0.0
        
        # 其他规则类型...
        return 0.0
    
    def _apply_false_positive_filter(self, filter_rule: Dict[str, Any], content: str, context: Dict[str, Any]) -> float:
        """应用误报过滤器"""
        filter_type = filter_rule.get("filter_type", "")
        
        if filter_type == "context_aware":
            keywords = filter_rule.get("context_keywords", [])
            weight_adjustment = filter_rule.get("weight_adjustment", -0.3)
            
            content_lower = content.lower()
            for keyword in keywords:
                if keyword in content_lower:
                    return weight_adjustment
        
        elif filter_type == "deliberate_anchoring":
            keywords = filter_rule.get("context_keywords", [])
            weight_adjustment = filter_rule.get("weight_adjustment", -0.4)
            
            content_lower = content.lower()
            for keyword in keywords:
                if keyword in content_lower:
                    return weight_adjustment
        
        return 0.0
    
    def _ml_based_detection(self, content: str, context: Dict[str, Any]) -> List[BiasDetectionResult]:
        """基于机器学习的检测"""
        detections = []
        
        for bias_type, model_config in self.ml_models.items():
            # 提取特征
            features = self._extract_ml_features(content, context, model_config["features"])
            
            # 计算得分（简化版）
            if features:
                scores = []
                for i, feature in enumerate(model_config["features"]):
                    if feature in features:
                        score = features[feature] * model_config["weights"][i]
                        scores.append(score)
                
                if scores:
                    ml_score = sum(scores)
                    
                    if ml_score >= model_config["threshold"]:
                        detection = BiasDetectionResult(
                            detection_id=f"ml_detection_{self.detection_counter:06d}",
                            bias_type=bias_type,
                            detected_at=datetime.now(),
                            context=content[:200],
                            evidence=[f"ML检测得分: {ml_score:.2f}"],
                            confidence=ml_score,
                            severity=BiasSeverity.LOW,
                            severity_score=0.0,
                            impact_score=0.0,
                            root_causes=[],
                            suggested_corrections=[]
                        )
                        
                        detections.append(detection)
                        self.detection_counter += 1
        
        return detections
    
    def _extract_ml_features(self, content: str, context: Dict[str, Any], feature_list: List[str]) -> Dict[str, float]:
        """提取机器学习特征"""
        features = {}
        content_lower = content.lower()
        
        for feature in feature_list:
            if feature == "keyword_ratio":
                # 计算关键词比例
                confirming = ["confirm", "support", "agree"]
                disconfirming = ["disprove", "refute", "contradict"]
                
                confirm_count = sum(1 for kw in confirming if kw in content_lower)
                disconfirm_count = sum(1 for kw in disconfirming if kw in content_lower)
                
                total = confirm_count + disconfirm_count
                if total > 0:
                    features[feature] = confirm_count / total
                else:
                    features[feature] = 0.0
            
            elif feature == "negation_count":
                # 否定词数量
                negation_words = ["not", "no", "never", "cannot", "won't"]
                count = sum(1 for word in negation_words if word in content_lower)
                features[feature] = min(1.0, count / 5.0)
            
            elif feature == "certainty_score":
                # 确定性语言得分
                certainty_words = ["certain", "sure", "definitely", "absolutely", "always"]
                count = sum(1 for word in certainty_words if word in content_lower)
                features[feature] = min(1.0, count / 5.0)
            
            elif feature == "confidence_score":
                # 置信度得分
                confidence = context.get("confidence", 0.5)
                features[feature] = confidence
            
            elif feature == "calibration_gap":
                # 校准差距
                confidence = context.get("confidence", 0.5)
                accuracy = context.get("accuracy", 0.5)
                features[feature] = abs(confidence - accuracy)
            
            elif feature == "vivid_word_count":
                # 生动词汇数量
                vivid_words = ["dramatic", "memorable", "vivid", "shocking", "emotional"]
                count = sum(1 for word in vivid_words if word in content_lower)
                features[feature] = min(1.0, count / 5.0)
            
            elif feature == "recency_emphasis":
                # 近期性强调
                recency_words = ["recent", "latest", "new", "just happened", "yesterday"]
                count = sum(1 for word in recency_words if word in content_lower)
                features[feature] = min(1.0, count / 5.0)
        
        return features
    
    def _filter_and_merge_detections(self, detections: List[BiasDetectionResult]) -> List[BiasDetectionResult]:
        """过滤和合并检测结果"""
        if not detections:
            return []
        
        # 按置信度排序
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        filtered = []
        seen_types = set()
        
        for detection in detections:
            # 检查置信度阈值
            if detection.confidence < self.min_confidence:
                continue
            
            # 检查重复类型（保留置信度最高的）
            if detection.bias_type in seen_types:
                continue
            
            seen_types.add(detection.bias_type)
            filtered.append(detection)
        
        return filtered
    
    def _assess_severity(self, detection: BiasDetectionResult, context: Dict[str, Any]):
        """评估严重性"""
        # 计算严重性得分
        severity_score = 0.0
        
        # 基于置信度
        severity_score += detection.confidence * 0.3
        
        # 基于上下文影响
        impact = context.get("decision_impact", 0.5)
        severity_score += impact * 0.4
        
        # 基于频率（如果有多条证据）
        evidence_count = len(detection.evidence)
        frequency_factor = min(1.0, evidence_count / 5.0)
        severity_score += frequency_factor * 0.3
        
        detection.severity_score = min(1.0, severity_score)
        detection.impact_score = impact
        
        # 确定严重性级别
        if severity_score >= 0.8:
            detection.severity = BiasSeverity.CRITICAL
        elif severity_score >= 0.6:
            detection.severity = BiasSeverity.HIGH
        elif severity_score >= 0.4:
            detection.severity = BiasSeverity.MODERATE
        else:
            detection.severity = BiasSeverity.LOW
    
    def _analyze_root_causes(self, detection: BiasDetectionResult, context: Dict[str, Any]):
        """分析根源"""
        root_causes = []
        
        # 基于偏差类型分析根源
        if detection.bias_type == "confirmation_bias":
            root_causes.extend([
                "先验信念过强",
                "反证信息获取不足",
                "认知闭合需求高"
            ])
        elif detection.bias_type == "overconfidence_bias":
            root_causes.extend([
                "缺乏校准反馈",
                "领域专业知识错觉",
                "成功经验过度泛化"
            ])
        elif detection.bias_type == "anchoring_bias":
            root_causes.extend([
                "初始信息过载",
                "调整启发式不足",
                "认知资源限制"
            ])
        elif detection.bias_type == "availability_heuristic":
            root_causes.extend([
                "易得性信息主导",
                "基础概率忽视",
                "媒体曝光影响"
            ])
        elif detection.bias_type == "sunk_cost_fallacy":
            root_causes.extend([
                "情感投入过深",
                "损失厌恶心理",
                "自我合理化倾向"
            ])
        
        # 添加上下文相关的根源
        decision_pressure = context.get("decision_pressure", 0.0)
        if decision_pressure > 0.7:
            root_causes.append("时间压力或决策压力")
        
        emotional_state = context.get("emotional_state", "neutral")
        if emotional_state in ["stressed", "anxious", "angry"]:
            root_causes.append(f"情绪状态影响: {emotional_state}")
        
        # 限制根源数量
        detection.root_causes = root_causes[:self.config['max_root_causes']]
    
    def _generate_corrections(self, detection: BiasDetectionResult):
        """生成校正建议"""
        corrections = []
        
        # 基于偏差类型的校正建议
        if detection.bias_type == "confirmation_bias":
            corrections.extend([
                "主动寻找反证信息",
                "考虑多种可能性假设",
                "与持不同观点者讨论",
                "使用魔鬼代言人法"
            ])
        elif detection.bias_type == "overconfidence_bias":
            corrections.extend([
                "进行概率校准训练",
                "考虑不确定性范围",
                "寻求外部反馈验证",
                "记录预测与实际结果对比"
            ])
        elif detection.bias_type == "anchoring_bias":
            corrections.extend([
                "考虑多个参考锚点",
                "进行系统化调整",
                "使用德尔菲法或群体估计",
                "延迟初始判断"
            ])
        elif detection.bias_type == "availability_heuristic":
            corrections.extend([
                "使用统计数据而非个人经验",
                "考虑基础概率",
                "系统化收集信息",
                "避免依赖易得性信息"
            ])
        elif detection.bias_type == "sunk_cost_fallacy":
            corrections.extend([
                "忽略沉没成本，关注未来收益",
                "进行边际分析",
                "寻求外部客观意见",
                "考虑机会成本"
            ])
        
        # 基于严重性的额外建议
        if detection.severity in [BiasSeverity.HIGH, BiasSeverity.CRITICAL]:
            corrections.append("启用结构化决策框架")
            corrections.append("寻求外部专家评审")
        
        detection.suggested_corrections = corrections
    
    def analyze_trends(self) -> BiasTrend:
        """分析偏差趋势"""
        current_time = datetime.now()
        
        # 获取时间窗口内的检测
        window_start = current_time - timedelta(hours=self.trend_window_hours)
        recent_detections = [
            d for d in self.bias_detections
            if d.detected_at >= window_start
        ]
        
        # 统计偏差类型分布
        bias_types = defaultdict(int)
        for detection in recent_detections:
            bias_types[detection.bias_type] += 1
        
        # 计算严重性分布
        severity_counts = defaultdict(int)
        for detection in recent_detections:
            severity_counts[detection.severity] += 1
        
        severity_distribution = {}
        total = len(recent_detections)
        if total > 0:
            for severity in BiasSeverity:
                count = severity_counts[severity]
                severity_distribution[severity] = count / total
        else:
            for severity in BiasSeverity:
                severity_distribution[severity] = 0.0
        
        # 计算总体趋势
        overall_trend = 0.0
        if len(self.trend_history) >= 2:
            # 比较最近两个趋势点
            prev_trend = self.trend_history[-1]
            current_total = sum(bias_types.values())
            prev_total = sum(prev_trend.bias_types.values())
            
            if prev_total > 0:
                change_rate = (current_total - prev_total) / prev_total
                overall_trend = max(-1.0, min(1.0, change_rate))
        
        # 确定预警级别
        alert_level = "normal"
        if overall_trend > self.alert_threshold:
            alert_level = "high_alert"
            self.performance_stats['alerts_triggered'] += 1
            self.last_alert_time = time.time()
        elif overall_trend > self.alert_threshold * 0.7:
            alert_level = "warning"
        
        # 创建趋势对象
        trend = BiasTrend(
            timestamp=current_time,
            bias_types=dict(bias_types),
            severity_distribution=severity_distribution,
            overall_trend=overall_trend,
            alert_level=alert_level
        )
        
        # 保存趋势历史
        self.trend_history.append(trend)
        if len(self.trend_history) > 100:
            self.trend_history.pop(0)
        
        return trend
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """获取检测摘要"""
        current_trend = self.analyze_trends()
        
        summary = {
            "timestamp": datetime.now(),
            "system_uptime_hours": (time.time() - self.system_start_time) / 3600,
            "statistics": {
                "total_detections": self.statistics.total_detections,
                "detection_rate_per_hour": self.statistics.detection_rate_per_hour,
                "average_confidence": self.statistics.average_confidence,
                "average_severity_score": self.statistics.average_severity_score
            },
            "recent_trend": {
                "overall_trend": current_trend.overall_trend,
                "alert_level": current_trend.alert_level,
                "top_biases": sorted(
                    current_trend.bias_types.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            },
            "performance_stats": self.performance_stats,
            "correction_status": {
                "suggested": self.performance_stats['corrections_suggested'],
                "applied": self.performance_stats['corrections_applied'],
                "success_rate": (
                    self.performance_stats['corrections_applied'] / 
                    max(1, self.performance_stats['corrections_suggested'])
                )
            }
        }
        
        return summary

# 全局实例
cognitive_bias_detector_instance = CognitiveBiasDetector()

if __name__ == "__main__":
    # 测试认知偏差检测器
    print("测试认知偏差检测器...")
    
    detector = CognitiveBiasDetector(
        detection_threshold=0.65,
        min_confidence=0.6
    )
    
    # 测试内容
    test_content = "这个方案肯定是最好的，我们已经投入了太多资源，现在放弃损失太大了。"
    test_context = {
        "confidence": 0.95,
        "decision_impact": 0.8,
        "decision_pressure": 0.7,
        "emotional_state": "stressed"
    }
    
    # 检测偏差
    detections = detector.detect_biases(test_content, test_context)
    
    print(f"检测到 {len(detections)} 个偏差:")
    for detection in detections:
        print(f"- {detection.bias_type}: 置信度 {detection.confidence:.2f}, 严重性 {detection.severity.value}")
        print(f"  证据: {detection.evidence[0] if detection.evidence else '无'}")
        print(f"  建议: {detection.suggested_corrections[0] if detection.suggested_corrections else '无'}")
    
    # 获取摘要
    summary = detector.get_detection_summary()
    print(f"\n检测摘要 - 总检测数: {summary['statistics']['total_detections']}")
    print(f"检测率: {summary['statistics']['detection_rate_per_hour']:.2f}/小时")
    print(f"趋势: {summary['recent_trend']['overall_trend']:.2f}")