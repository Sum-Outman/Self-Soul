"""
思维过程追踪器

该模块实现AGI系统的思维过程详细追踪功能，包括：
1. 思维活动记录：详细记录思维活动的时序、内容和属性
2. 思维图构建：构建思维活动的图结构表示
3. 思维模式分析：分析思维活动的模式和规律
4. 注意力追踪：追踪注意力的焦点和转移
5. 思维质量评估：评估思维过程的质量和效率

核心功能：
1. 高精度思维事件记录
2. 思维关系图构建和分析
3. 注意力动态追踪
4. 思维模式识别
5. 思维质量量化评估

技术特性：
- 多层次思维活动建模
- 实时思维图分析
- 注意力动态建模
- 模式识别算法
- 质量评估指标
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
from collections import deque, defaultdict
import json
from hashlib import md5

# 配置日志
logger = logging.getLogger(__name__)

class ThoughtGranularity(Enum):
    """思维粒度"""
    MICRO = "micro"  # 毫秒级基本思维单元
    MESO = "meso"   # 秒级思维片段
    MACRO = "macro"  # 分钟级思维阶段

class ThoughtRelationType(Enum):
    """思维关系类型"""
    TEMPORAL_SEQUENCE = "temporal_sequence"
    CAUSAL = "causal"
    LOGICAL = "logical"
    ASSOCIATIVE = "associative"
    HIERARCHICAL = "hierarchical"
    CONTRADICTORY = "contradictory"
    SUPPORTING = "supporting"
    REFUTING = "refuting"

class AttentionState(Enum):
    """注意力状态"""
    FOCUSED = "focused"
    SCANNING = "scanning"
    DIVIDED = "divided"
    SWITCHING = "switching"
    RESTING = "resting"

@dataclass
class ThoughtUnit:
    """思维单元"""
    unit_id: str
    timestamp: datetime
    content: str
    granularity: ThoughtGranularity
    activity_type: str
    cognitive_load: float  # 0-1
    confidence: float  # 0-1
    emotional_valence: float  # -1到1
    arousal: float  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ThoughtRelation:
    """思维关系"""
    relation_id: str
    source_unit_id: str
    target_unit_id: str
    relation_type: ThoughtRelationType
    strength: float  # 0-1
    confidence: float  # 0-1
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AttentionFocus:
    """注意力焦点"""
    focus_id: str
    timestamp: datetime
    topic: str
    intensity: float  # 0-1
    duration_ms: float
    thought_unit_ids: List[str]
    state: AttentionState
    transition_from: Optional[str] = None

@dataclass
class ThinkingPattern:
    """思维模式"""
    pattern_id: str
    pattern_type: str
    description: str
    detected_at: datetime
    frequency: int
    confidence: float
    characteristics: Dict[str, Any]
    thought_unit_ids: List[str]

@dataclass
class ThoughtQualityMetrics:
    """思维质量指标"""
    clarity: float  # 0-1
    coherence: float  # 0-1
    relevance: float  # 0-1
    depth: float  # 0-1
    efficiency: float  # 0-1
    creativity: float  # 0-1
    overall_score: float  # 0-1

class ThinkingProcessTracker:
    """
    思维过程追踪器
    
    核心组件:
    1. 思维记录器: 记录思维单元和属性
    2. 关系分析器: 分析思维单元之间的关系
    3. 注意力追踪器: 追踪注意力焦点和转移
    4. 模式识别器: 识别思维模式
    5. 质量评估器: 评估思维质量
    
    工作流程:
    思维活动 → 思维记录器 → 记录单元 → 关系分析器 → 分析关系
    注意力变化 → 注意力追踪器 → 追踪焦点 → 模式识别器 → 识别模式
    综合分析 → 质量评估器 → 评估质量 → 反馈到认知系统
    
    技术特性:
    - 多层次思维单元记录
    - 实时关系图分析
    - 注意力动态建模
    - 模式识别算法
    - 多维度质量评估
    """
    
    def __init__(self,
                 recording_granularity: ThoughtGranularity = ThoughtGranularity.MESO,
                 max_thought_history: int = 5000,
                 pattern_recognition_enabled: bool = True,
                 attention_tracking_enabled: bool = True,
                 quality_assessment_enabled: bool = True):
        """
        初始化思维过程追踪器
        
        Args:
            recording_granularity: 记录粒度
            max_thought_history: 最大思维历史记录
            pattern_recognition_enabled: 是否启用模式识别
            attention_tracking_enabled: 是否启用注意力追踪
            quality_assessment_enabled: 是否启用量评估
        """
        self.recording_granularity = recording_granularity
        self.max_thought_history = max_thought_history
        self.pattern_recognition_enabled = pattern_recognition_enabled
        self.attention_tracking_enabled = attention_tracking_enabled
        self.quality_assessment_enabled = quality_assessment_enabled
        
        # 思维单元存储
        self.thought_units: Dict[str, ThoughtUnit] = {}
        self.thought_unit_order: List[str] = []
        
        # 思维关系图
        self.thought_graph = nx.MultiDiGraph()
        
        # 注意力追踪
        self.attention_history: List[AttentionFocus] = []
        self.current_focus: Optional[AttentionFocus] = None
        self.attention_transitions: List[Tuple[str, str, float]] = []  # (from, to, duration)
        
        # 模式识别
        self.thinking_patterns: Dict[str, ThinkingPattern] = {}
        self.pattern_history: List[ThinkingPattern] = []
        
        # 质量评估
        self.quality_history: List[Tuple[datetime, ThoughtQualityMetrics]] = []
        
        # 配置参数
        self.config = {
            'micro_threshold_ms': 50.0,
            'meso_threshold_ms': 1000.0,
            'macro_threshold_minutes': 5.0,
            'relation_strength_threshold': 0.3,
            'attention_switch_threshold_ms': 200.0,
            'pattern_detection_window': 10,
            'quality_assessment_interval_s': 30.0,
            'clarity_weight': 0.2,
            'coherence_weight': 0.2,
            'relevance_weight': 0.15,
            'depth_weight': 0.15,
            'efficiency_weight': 0.15,
            'creativity_weight': 0.15
        }
        
        # 性能统计
        self.performance_stats = {
            'thought_units_recorded': 0,
            'thought_relations_identified': 0,
            'attention_foci_tracked': 0,
            'thinking_patterns_detected': 0,
            'quality_assessments_performed': 0,
            'average_thought_quality': 0.0,
            'attention_stability': 0.0,
            'pattern_regularity': 0.0
        }
        
        # 状态变量
        self.last_recording_time = time.time()
        self.system_start_time = time.time()
        self.thought_unit_counter = 0
        
        logger.info(f"思维过程追踪器初始化完成，记录粒度: {recording_granularity.value}")
    
    def record_thought_unit(self,
                           content: str,
                           activity_type: str,
                           cognitive_load: float,
                           confidence: float,
                           emotional_valence: float = 0.0,
                           arousal: float = 0.5,
                           metadata: Optional[Dict[str, Any]] = None) -> ThoughtUnit:
        """记录思维单元"""
        # 生成单元ID
        unit_id = f"thought_{self.thought_unit_counter:08d}"
        self.thought_unit_counter += 1
        
        # 确定粒度
        current_time = datetime.now()
        if self.thought_unit_order:
            last_unit_id = self.thought_unit_order[-1]
            last_unit = self.thought_units[last_unit_id]
            time_gap = (current_time - last_unit.timestamp).total_seconds() * 1000
            
            if time_gap < self.config['micro_threshold_ms']:
                granularity = ThoughtGranularity.MICRO
            elif time_gap < self.config['meso_threshold_ms']:
                granularity = ThoughtGranularity.MESO
            else:
                granularity = ThoughtGranularity.MACRO
        else:
            granularity = self.recording_granularity
        
        # 创建思维单元
        unit = ThoughtUnit(
            unit_id=unit_id,
            timestamp=current_time,
            content=content,
            granularity=granularity,
            activity_type=activity_type,
            cognitive_load=cognitive_load,
            confidence=confidence,
            emotional_valence=emotional_valence,
            arousal=arousal,
            metadata=metadata or {}
        )
        
        # 存储思维单元
        self.thought_units[unit_id] = unit
        self.thought_unit_order.append(unit_id)
        
        # 添加到图
        self.thought_graph.add_node(unit_id, **unit.__dict__)
        
        # 分析与前一个单元的关系
        if len(self.thought_unit_order) > 1:
            prev_unit_id = self.thought_unit_order[-2]
            self._analyze_thought_relation(prev_unit_id, unit_id)
        
        # 更新注意力焦点
        if self.attention_tracking_enabled:
            self._update_attention_focus(unit_id, content, activity_type)
        
        # 更新统计
        self.performance_stats['thought_units_recorded'] += 1
        self.last_recording_time = time.time()
        
        # 检查是否需要进行模式识别
        if (self.pattern_recognition_enabled and 
            len(self.thought_unit_order) % self.config['pattern_detection_window'] == 0):
            self._detect_thinking_patterns()
        
        # 检查是否需要进行质量评估
        if (self.quality_assessment_enabled and 
            time.time() - self.last_recording_time > self.config['quality_assessment_interval_s']):
            self._assess_thought_quality()
        
        logger.debug(f"记录思维单元: {unit_id}, 内容: {content[:50]}...")
        return unit
    
    def _analyze_thought_relation(self, source_id: str, target_id: str):
        """分析思维关系"""
        source_unit = self.thought_units[source_id]
        target_unit = self.thought_units[target_id]
        
        relations = []
        
        # 1. 时间关系
        time_gap = (target_unit.timestamp - source_unit.timestamp).total_seconds()
        if time_gap < 2.0:  # 2秒内
            relation = ThoughtRelation(
                relation_id=f"rel_{source_id}_{target_id}_temporal",
                source_unit_id=source_id,
                target_unit_id=target_id,
                relation_type=ThoughtRelationType.TEMPORAL_SEQUENCE,
                strength=min(1.0, 1.0 / (time_gap + 0.1)),
                confidence=0.8,
                evidence=["时间相邻性", f"时间间隔: {time_gap:.2f}秒"]
            )
            relations.append(relation)
        
        # 2. 内容关系
        content_similarity = self._calculate_content_similarity(
            source_unit.content, target_unit.content
        )
        if content_similarity > self.config['relation_strength_threshold']:
            relation_type = ThoughtRelationType.ASSOCIATIVE
            if content_similarity > 0.7:
                relation_type = ThoughtRelationType.LOGICAL
            
            relation = ThoughtRelation(
                relation_id=f"rel_{source_id}_{target_id}_content",
                source_unit_id=source_id,
                target_unit_id=target_id,
                relation_type=relation_type,
                strength=content_similarity,
                confidence=0.7,
                evidence=["内容相似性", f"相似度: {content_similarity:.2f}"]
            )
            relations.append(relation)
        
        # 3. 认知状态关系
        cognitive_similarity = 1.0 - abs(source_unit.cognitive_load - target_unit.cognitive_load)
        if cognitive_similarity > 0.6:
            relation = ThoughtRelation(
                relation_id=f"rel_{source_id}_{target_id}_cognitive",
                source_unit_id=source_id,
                target_unit_id=target_id,
                relation_type=ThoughtRelationType.HIERARCHICAL,
                strength=cognitive_similarity,
                confidence=0.6,
                evidence=["认知状态连续性", f"认知负载相似度: {cognitive_similarity:.2f}"]
            )
            relations.append(relation)
        
        # 添加到图
        for relation in relations:
            self.thought_graph.add_edge(
                source_id,
                target_id,
                key=relation.relation_id,
                **relation.__dict__
            )
            self.performance_stats['thought_relations_identified'] += 1
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似度"""
        # 简单基于关键词的相似度计算
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _update_attention_focus(self, unit_id: str, content: str, activity_type: str):
        """更新注意力焦点"""
        current_time = datetime.now()
        
        # 提取话题关键词
        topic = self._extract_topic(content, activity_type)
        
        if self.current_focus is None:
            # 创建新的注意力焦点
            self.current_focus = AttentionFocus(
                focus_id=f"focus_{len(self.attention_history) + 1:04d}",
                timestamp=current_time,
                topic=topic,
                intensity=0.7,
                duration_ms=0.0,
                thought_unit_ids=[unit_id],
                state=AttentionState.FOCUSED
            )
            self.attention_history.append(self.current_focus)
            self.performance_stats['attention_foci_tracked'] += 1
            
        else:
            # 检查话题是否变化
            current_topic = self.current_focus.topic
            topic_similarity = self._calculate_topic_similarity(current_topic, topic)
            
            if topic_similarity < 0.4:  # 话题显著变化
                # 结束当前焦点
                duration_ms = (current_time - self.current_focus.timestamp).total_seconds() * 1000
                self.current_focus.duration_ms = duration_ms
                
                # 记录转移
                if len(self.attention_history) > 1:
                    prev_focus = self.attention_history[-2]
                    self.attention_transitions.append((
                        prev_focus.topic,
                        topic,
                        duration_ms
                    ))
                
                # 创建新焦点
                self.current_focus = AttentionFocus(
                    focus_id=f"focus_{len(self.attention_history) + 1:04d}",
                    timestamp=current_time,
                    topic=topic,
                    intensity=0.7,
                    duration_ms=0.0,
                    thought_unit_ids=[unit_id],
                    state=AttentionState.SWITCHING,
                    transition_from=current_topic
                )
                self.attention_history.append(self.current_focus)
                self.performance_stats['attention_foci_tracked'] += 1
                
            else:
                # 继续当前焦点
                self.current_focus.thought_unit_ids.append(unit_id)
                self.current_focus.intensity = min(1.0, self.current_focus.intensity + 0.05)
                
                # 更新状态
                if len(self.current_focus.thought_unit_ids) > 10:
                    self.current_focus.state = AttentionState.FOCUSED
                elif len(self.current_focus.thought_unit_ids) > 5:
                    self.current_focus.state = AttentionState.SCANNING
        
        # 更新注意力稳定性统计
        if len(self.attention_history) >= 5:
            durations = [f.duration_ms for f in self.attention_history[-5:]]
            avg_duration = np.mean(durations)
            self.performance_stats['attention_stability'] = avg_duration / 1000.0  # 转换为秒
    
    def _extract_topic(self, content: str, activity_type: str) -> str:
        """提取话题"""
        # 简单关键词提取
        words = content.lower().split()
        
        # 常见停用词
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        if keywords:
            # 选择出现频率高的词
            word_counts = defaultdict(int)
            for word in keywords:
                word_counts[word] += 1
            
            if word_counts:
                top_word = max(word_counts.items(), key=lambda x: x[1])[0]
                return f"{activity_type}:{top_word}"
        
        return f"{activity_type}:general"
    
    def _calculate_topic_similarity(self, topic1: str, topic2: str) -> float:
        """计算话题相似度"""
        if topic1 == topic2:
            return 1.0
        
        # 提取主要部分
        main1 = topic1.split(":")[-1] if ":" in topic1 else topic1
        main2 = topic2.split(":")[-1] if ":" in topic2 else topic2
        
        # 简单字符串相似度
        if main1 in main2 or main2 in main1:
            return 0.6
        
        # 基于共同字符
        common_chars = set(main1).intersection(set(main2))
        if common_chars:
            return len(common_chars) / max(len(main1), len(main2))
        
        return 0.0
    
    def _detect_thinking_patterns(self):
        """检测思维模式"""
        if not self.thought_unit_order or len(self.thought_unit_order) < 10:
            return
        
        recent_units = self.thought_unit_order[-10:]
        
        # 分析思维序列的模式
        patterns = []
        
        # 1. 检查重复模式
        pattern1 = self._detect_repetition_pattern(recent_units)
        if pattern1:
            patterns.append(pattern1)
        
        # 2. 检查交替模式
        pattern2 = self._detect_alternation_pattern(recent_units)
        if pattern2:
            patterns.append(pattern2)
        
        # 3. 检查渐进模式
        pattern3 = self._detect_progressive_pattern(recent_units)
        if pattern3:
            patterns.append(pattern3)
        
        # 保存模式
        for pattern in patterns:
            pattern_id = f"pattern_{len(self.thinking_patterns) + 1:04d}"
            pattern.pattern_id = pattern_id
            self.thinking_patterns[pattern_id] = pattern
            self.pattern_history.append(pattern)
            self.performance_stats['thinking_patterns_detected'] += 1
        
        # 更新模式规律性统计
        if len(self.pattern_history) >= 3:
            pattern_intervals = []
            for i in range(1, len(self.pattern_history)):
                time_gap = (self.pattern_history[i].detected_at - 
                           self.pattern_history[i-1].detected_at).total_seconds()
                pattern_intervals.append(time_gap)
            
            if pattern_intervals:
                interval_std = np.std(pattern_intervals)
                self.performance_stats['pattern_regularity'] = 1.0 / (interval_std + 1.0)
    
    def _detect_repetition_pattern(self, unit_ids: List[str]) -> Optional[ThinkingPattern]:
        """检测重复模式"""
        # 检查是否在重复相同类型的思维活动
        activity_types = []
        for unit_id in unit_ids:
            if unit_id in self.thought_units:
                unit = self.thought_units[unit_id]
                activity_types.append(unit.activity_type)
        
        # 检查是否有显著重复
        from collections import Counter
        type_counts = Counter(activity_types)
        most_common_type, count = type_counts.most_common(1)[0]
        
        if count >= len(unit_ids) * 0.6:  # 60%以上是相同类型
            return ThinkingPattern(
                pattern_id="",  # 稍后设置
                pattern_type="repetition",
                description=f"重复{most_common_type}思维活动",
                detected_at=datetime.now(),
                frequency=count,
                confidence=count / len(unit_ids),
                characteristics={
                    "dominant_activity_type": most_common_type,
                    "repetition_ratio": count / len(unit_ids),
                    "variety_score": len(set(activity_types)) / len(activity_types)
                },
                thought_unit_ids=unit_ids
            )
        
        return None
    
    def _detect_alternation_pattern(self, unit_ids: List[str]) -> Optional[ThinkingPattern]:
        """检测交替模式"""
        # 检查思维活动是否在两种类型间交替
        if len(unit_ids) < 4:
            return None
        
        activity_types = []
        for unit_id in unit_ids:
            if unit_id in self.thought_units:
                unit = self.thought_units[unit_id]
                activity_types.append(unit.activity_type)
        
        # 检查交替性
        alternation_count = 0
        for i in range(1, len(activity_types) - 1):
            if activity_types[i-1] != activity_types[i] and activity_types[i] != activity_types[i+1]:
                alternation_count += 1
        
        alternation_ratio = alternation_count / (len(activity_types) - 2)
        
        if alternation_ratio > 0.6:  # 高交替性
            return ThinkingPattern(
                pattern_id="",
                pattern_type="alternation",
                description="思维活动类型频繁交替",
                detected_at=datetime.now(),
                frequency=alternation_count,
                confidence=alternation_ratio,
                characteristics={
                    "alternation_ratio": alternation_ratio,
                    "activity_type_count": len(set(activity_types)),
                    "average_alternation_interval": len(activity_types) / alternation_count if alternation_count > 0 else 0
                },
                thought_unit_ids=unit_ids
            )
        
        return None
    
    def _detect_progressive_pattern(self, unit_ids: List[str]) -> Optional[ThinkingPattern]:
        """检测渐进模式"""
        # 检查思维深度是否逐渐增加
        if len(unit_ids) < 3:
            return None
        
        cognitive_loads = []
        confidences = []
        
        for unit_id in unit_ids:
            if unit_id in self.thought_units:
                unit = self.thought_units[unit_id]
                cognitive_loads.append(unit.cognitive_load)
                confidences.append(unit.confidence)
        
        # 检查趋势
        load_trend = self._calculate_trend(cognitive_loads)
        confidence_trend = self._calculate_trend(confidences)
        
        if load_trend > 0.1 and confidence_trend > 0.05:  # 认知负载和置信度都在增加
            return ThinkingPattern(
                pattern_id="",
                pattern_type="progressive_deepening",
                description="思维深度渐进增加",
                detected_at=datetime.now(),
                frequency=1,
                confidence=(load_trend + confidence_trend) / 2,
                characteristics={
                    "cognitive_load_trend": load_trend,
                    "confidence_trend": confidence_trend,
                    "average_cognitive_load": np.mean(cognitive_loads),
                    "average_confidence": np.mean(confidences)
                },
                thought_unit_ids=unit_ids
            )
        
        return None
    
    def _calculate_trend(self, values: List[float]) -> float:
        """计算趋势"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        try:
            slope, _ = np.polyfit(x, y, 1)
            return slope
        except:
            return 0.0
    
    def _assess_thought_quality(self):
        """评估思维质量"""
        if not self.thought_unit_order:
            return
        
        # 获取最近思维单元
        recent_count = min(20, len(self.thought_unit_order))
        recent_unit_ids = self.thought_unit_order[-recent_count:]
        
        if not recent_unit_ids:
            return
        
        # 收集指标数据
        units = [self.thought_units[uid] for uid in recent_unit_ids]
        
        # 1. 清晰度：基于置信度和内容明确性
        clarity_scores = []
        for unit in units:
            score = unit.confidence * (1.0 - unit.cognitive_load * 0.3)  # 高认知负载可能降低清晰度
            clarity_scores.append(score)
        clarity = np.mean(clarity_scores)
        
        # 2. 连贯性：基于思维单元间的关系强度
        coherence_scores = []
        for i in range(len(recent_unit_ids) - 1):
            source_id = recent_unit_ids[i]
            target_id = recent_unit_ids[i + 1]
            
            if self.thought_graph.has_edge(source_id, target_id):
                edges = self.thought_graph.get_edge_data(source_id, target_id)
                if edges:
                    max_strength = max(edge.get('strength', 0.0) for edge in edges.values())
                    coherence_scores.append(max_strength)
            else:
                coherence_scores.append(0.0)
        
        coherence = np.mean(coherence_scores) if coherence_scores else 0.5
        
        # 3. 相关性：基于注意力稳定性
        relevance = 0.7  # 默认值
        if self.attention_history:
            recent_foci = self.attention_history[-3:]
            if recent_foci:
                topic_similarities = []
                for i in range(len(recent_foci) - 1):
                    sim = self._calculate_topic_similarity(
                        recent_foci[i].topic,
                        recent_foci[i + 1].topic
                    )
                    topic_similarities.append(sim)
                
                if topic_similarities:
                    relevance = np.mean(topic_similarities)
        
        # 4. 深度：基于认知负载和思维粒度
        depth_scores = []
        for unit in units:
            # 宏观思维通常更深
            depth_factor = 1.0
            if unit.granularity == ThoughtGranularity.MACRO:
                depth_factor = 1.2
            elif unit.granularity == ThoughtGranularity.MICRO:
                depth_factor = 0.8
            
            depth_scores.append(unit.cognitive_load * depth_factor)
        depth = np.mean(depth_scores)
        
        # 5. 效率：基于思维速度和多样性
        if len(units) >= 2:
            time_span = (units[-1].timestamp - units[0].timestamp).total_seconds()
            if time_span > 0:
                efficiency = len(units) / time_span  # 单位时间思维单元数
                efficiency = min(1.0, efficiency / 5.0)  # 标准化
            else:
                efficiency = 0.5
        else:
            efficiency = 0.5
        
        # 6. 创造性：基于思维模式的新颖性
        creativity = 0.5
        if self.pattern_history:
            recent_patterns = self.pattern_history[-5:]
            if recent_patterns:
                # 模式多样性反映创造性
                pattern_types = set(p.pattern_type for p in recent_patterns)
                creativity = len(pattern_types) / 3.0  # 标准化
        
        # 计算总体得分
        weights = self.config
        overall_score = (
            clarity * weights['clarity_weight'] +
            coherence * weights['coherence_weight'] +
            relevance * weights['relevance_weight'] +
            depth * weights['depth_weight'] +
            efficiency * weights['efficiency_weight'] +
            creativity * weights['creativity_weight']
        )
        
        # 创建质量指标
        quality_metrics = ThoughtQualityMetrics(
            clarity=clarity,
            coherence=coherence,
            relevance=relevance,
            depth=depth,
            efficiency=efficiency,
            creativity=creativity,
            overall_score=overall_score
        )
        
        # 保存质量历史
        self.quality_history.append((datetime.now(), quality_metrics))
        if len(self.quality_history) > 100:
            self.quality_history.pop(0)
        
        # 更新统计
        self.performance_stats['quality_assessments_performed'] += 1
        self.performance_stats['average_thought_quality'] = (
            (self.performance_stats['average_thought_quality'] * 
             (self.performance_stats['quality_assessments_performed'] - 1) + 
             overall_score) / self.performance_stats['quality_assessments_performed']
        )
    
    def get_thought_analysis(self) -> Dict[str, Any]:
        """获取思维分析报告"""
        current_time = datetime.now()
        
        # 获取最近质量评估
        recent_quality = None
        if self.quality_history:
            recent_quality = self.quality_history[-1][1]
        
        # 获取当前注意力状态
        current_attention = None
        if self.current_focus:
            current_attention = {
                "topic": self.current_focus.topic,
                "state": self.current_focus.state.value,
                "intensity": self.current_focus.intensity,
                "duration_ms": self.current_focus.duration_ms,
                "thought_unit_count": len(self.current_focus.thought_unit_ids)
            }
        
        # 获取最近模式
        recent_patterns = []
        if self.pattern_history:
            for pattern in self.pattern_history[-3:]:
                recent_patterns.append({
                    "type": pattern.pattern_type,
                    "description": pattern.description,
                    "confidence": pattern.confidence,
                    "detected_at": pattern.detected_at.isoformat()
                })
        
        analysis = {
            "timestamp": current_time,
            "system_uptime_hours": (time.time() - self.system_start_time) / 3600,
            "thought_statistics": {
                "total_units": len(self.thought_units),
                "total_relations": self.performance_stats['thought_relations_identified'],
                "recent_activity": len(self.thought_unit_order[-10:]) if self.thought_unit_order else 0,
                "average_quality": self.performance_stats['average_thought_quality']
            },
            "current_attention": current_attention,
            "recent_patterns": recent_patterns,
            "recent_quality": {
                "overall_score": recent_quality.overall_score if recent_quality else 0.0,
                "clarity": recent_quality.clarity if recent_quality else 0.0,
                "coherence": recent_quality.coherence if recent_quality else 0.0,
                "efficiency": recent_quality.efficiency if recent_quality else 0.0
            },
            "performance_stats": self.performance_stats,
            "attention_transitions": [
                {
                    "from": trans[0],
                    "to": trans[1],
                    "duration_ms": trans[2]
                }
                for trans in self.attention_transitions[-5:]
            ]
        }
        
        return analysis
    
    def export_thought_data(self, format: str = "json") -> str:
        """导出思维数据"""
        if format == "json":
            data = {
                "thought_units": [
                    {
                        "id": unit.unit_id,
                        "timestamp": unit.timestamp.isoformat(),
                        "content": unit.content,
                        "granularity": unit.granularity.value,
                        "activity_type": unit.activity_type,
                        "cognitive_load": unit.cognitive_load,
                        "confidence": unit.confidence
                    }
                    for unit in self.thought_units.values()
                ],
                "attention_history": [
                    {
                        "topic": focus.topic,
                        "state": focus.state.value,
                        "duration_ms": focus.duration_ms,
                        "timestamp": focus.timestamp.isoformat()
                    }
                    for focus in self.attention_history[-50:]
                ],
                "thinking_patterns": [
                    {
                        "type": pattern.pattern_type,
                        "description": pattern.description,
                        "confidence": pattern.confidence,
                        "detected_at": pattern.detected_at.isoformat()
                    }
                    for pattern in self.pattern_history[-20:]
                ]
            }
            return json.dumps(data, ensure_ascii=False, indent=2)
        
        return ""

# 全局实例
thinking_process_tracker_instance = ThinkingProcessTracker()

if __name__ == "__main__":
    # 测试思维过程追踪器
    print("测试思维过程追踪器...")
    
    tracker = ThinkingProcessTracker(
        recording_granularity=ThoughtGranularity.MESO,
        pattern_recognition_enabled=True,
        attention_tracking_enabled=True
    )
    
    # 记录一些思维单元
    tracker.record_thought_unit(
        content="分析用户请求中的关键信息",
        activity_type="reasoning",
        cognitive_load=0.6,
        confidence=0.8
    )
    
    tracker.record_thought_unit(
        content="检索相关知识和解决方案",
        activity_type="memory_retrieval",
        cognitive_load=0.4,
        confidence=0.9
    )
    
    tracker.record_thought_unit(
        content="比较不同方案的优劣",
        activity_type="decision_making",
        cognitive_load=0.7,
        confidence=0.75
    )
    
    # 获取思维分析
    analysis = tracker.get_thought_analysis()
    print(f"思维分析 - 总单元数: {analysis['thought_statistics']['total_units']}")
    print(f"平均质量: {analysis['thought_statistics']['average_quality']:.3f}")
    
    if analysis['current_attention']:
        print(f"当前注意力: {analysis['current_attention']['topic']}")
    
    # 导出数据
    export_data = tracker.export_thought_data()
    print(f"数据导出完成，长度: {len(export_data)} 字符")