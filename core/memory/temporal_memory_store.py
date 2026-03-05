#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时序记忆存储 - 实现情景记忆的时序组织和检索

核心功能:
1. 事件序列存储: 按照时间顺序存储事件
2. 时序关联发现: 发现事件之间的时序模式
3. 时间线重建: 根据时间戳重建事件时间线
4. 时序检索: 基于时间范围检索事件
5. 事件持续时间建模: 记录和建模事件持续时间

时序记忆特点:
- 时间轴组织: 按照时间顺序组织记忆
- 事件时间戳: 每个事件都有精确的时间戳
- 持续时间: 记录事件的开始和结束时间
- 时间关系: 记录事件之间的时序关系
- 周期性模式: 识别事件的周期性模式

时序关系类型:
1. 同时发生 (concurrent): 事件在同一时间发生
2. 先后发生 (sequential): 一个事件在另一个事件之后发生
3. 包含关系 (containment): 一个事件包含另一个事件
4. 重叠关系 (overlap): 事件在时间上有重叠

时序模式发现:
1. 频繁序列: 频繁出现的时序模式
2. 周期性模式: 按照固定间隔重复的模式
3. 因果时序: 具有因果关系的时序模式
4. 异常时序: 偏离正常模式的时序

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
import bisect
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from heapq import heappush, heappop

logger = logging.getLogger(__name__)


class TemporalRelation(Enum):
    """时序关系枚举"""
    BEFORE = "before"          # A 在 B 之前
    AFTER = "after"            # A 在 B 之后
    DURING = "during"          # A 在 B 期间发生
    CONTAINS = "contains"      # A 包含 B
    OVERLAPS = "overlaps"      # A 与 B 重叠
    MEETS = "meets"            # A 与 B 紧接着
    STARTS = "starts"          # A 与 B 同时开始
    ENDS = "ends"              # A 与 B 同时结束
    EQUALS = "equals"          # A 与 B 同时开始和结束


class TemporalPatternType(Enum):
    """时序模式类型枚举"""
    FREQUENT_SEQUENCE = "frequent_sequence"    # 频繁序列
    PERIODIC = "periodic"                      # 周期性模式
    CAUSAL_SEQUENCE = "causal_sequence"        # 因果序列
    ANOMALY = "anomaly"                        # 异常模式


@dataclass
class TemporalEvent:
    """时序事件数据类"""
    id: str
    content: Any
    start_time: float
    end_time: Optional[float] = None
    event_type: str = "generic"
    importance: float = 0.5
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        if self.end_time is None:
            self.end_time = self.start_time  # 瞬时事件
        
        if self.end_time < self.start_time:
            # 交换开始和结束时间
            self.start_time, self.end_time = self.end_time, self.start_time
        
        self.importance = max(0.0, min(1.0, self.importance))
    
    @property
    def duration(self) -> float:
        """获取事件持续时间"""
        return self.end_time - self.start_time
    
    @property
    def is_instant(self) -> bool:
        """判断是否是瞬时事件"""
        return abs(self.duration) < 1e-6


@dataclass
class TemporalRelationRecord:
    """时序关系记录数据类"""
    source_event_id: str
    target_event_id: str
    relation: TemporalRelation
    confidence: float = 1.0
    evidence_count: int = 1
    discovered_time: float = field(default_factory=time.time)
    last_observed_time: Optional[float] = None
    
    def __post_init__(self):
        """后初始化验证"""
        self.confidence = max(0.0, min(1.0, self.confidence))
        if self.last_observed_time is None:
            self.last_observed_time = self.discovered_time


@dataclass
class TemporalPattern:
    """时序模式数据类"""
    pattern_id: str
    pattern_type: TemporalPatternType
    event_sequence: List[str]  # 事件ID序列
    time_intervals: List[float]  # 事件之间的时间间隔
    frequency: int = 1
    confidence: float = 0.8
    first_observed: float = field(default_factory=time.time)
    last_observed: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.confidence = max(0.0, min(1.0, self.confidence))
        if self.last_observed is None:
            self.last_observed = self.first_observed
        
        # 确保序列和间隔长度匹配
        if len(self.event_sequence) != len(self.time_intervals) + 1:
            # 为最后一个事件添加虚拟间隔
            self.time_intervals.append(0.0)


class TemporalMemoryStore:
    """
    时序记忆存储 - 管理事件的时间和时序关系
    
    核心组件:
    1. 事件管理器: 事件的增删改查
    2. 时序分析器: 分析事件的时间关系
    3. 模式发现器: 发现时序模式
    4. 时间线构建器: 构建事件时间线
    5. 时序查询器: 基于时间的查询
    
    工作流程:
    事件数据 → 事件管理器 → 存储事件 → 时序分析器 → 分析关系
    时序关系 → 模式发现器 → 发现模式 → 时间线构建器 → 构建时间线
    查询请求 → 时序查询器 → 检索事件 → 返回结果
    
    技术特性:
    - 高效的时间索引和检索
    - 复杂的时序关系分析
    - 时序模式发现
    - 时间线重建和可视化
    """
    
    def __init__(self,
                 max_events: int = 5000,
                 time_resolution: float = 1.0,  # 秒
                 pattern_discovery_enabled: bool = True):
        """
        初始化时序记忆存储
        
        Args:
            max_events: 最大事件数量
            time_resolution: 时间分辨率（秒）
            pattern_discovery_enabled: 是否启用模式发现
        """
        self.max_events = max_events
        self.time_resolution = time_resolution
        self.pattern_discovery_enabled = pattern_discovery_enabled
        
        # 事件存储
        self.events: Dict[str, TemporalEvent] = {}
        
        # 时间索引
        self.time_index = []  # 排序的时间戳列表
        self.event_time_index = []  # 对应的事件ID列表
        
        # 类型索引
        self.events_by_type: Dict[str, Set[str]] = defaultdict(set)
        
        # 时序关系
        self.temporal_relations: List[TemporalRelationRecord] = []
        self.relation_index: Dict[Tuple[str, str, TemporalRelation], TemporalRelationRecord] = {}
        
        # 时序模式
        self.temporal_patterns: Dict[str, TemporalPattern] = {}
        self.patterns_by_type: Dict[TemporalPatternType, Set[str]] = defaultdict(set)
        
        # 配置参数
        self.config = {
            'min_pattern_length': 2,
            'max_pattern_length': 10,
            'min_pattern_frequency': 2,
            'max_time_gap': 3600.0,  # 秒
            'pattern_confidence_threshold': 0.6,
            'relation_confidence_threshold': 0.5,
            'time_window_size': 3600.0,  # 秒
            'pattern_update_interval': 300.0  # 秒
        }
        
        # 性能统计
        self.performance_stats = {
            'events_stored': 0,
            'events_retrieved': 0,
            'temporal_relations_discovered': 0,
            'temporal_patterns_discovered': 0,
            'timelines_reconstructed': 0,
            'average_query_time': 0.0
        }
        
        # 模式发现状态
        self.last_pattern_discovery_time = time.time()
        
        logger.info(f"时序记忆存储初始化完成，最大事件: {max_events}，时间分辨率: {time_resolution}秒")
    
    def add_event(self,
                 content: Any,
                 start_time: float,
                 end_time: Optional[float] = None,
                 event_type: str = "generic",
                 importance: float = 0.5,
                 context: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> TemporalEvent:
        """
        添加事件
        
        Args:
            content: 事件内容
            start_time: 开始时间
            end_time: 结束时间（如果为None则是瞬时事件）
            event_type: 事件类型
            importance: 重要性 (0.0-1.0)
            context: 情境信息
            metadata: 元数据
            
        Returns:
            添加的事件
        """
        # 生成事件ID
        event_id = f"event_{int(time.time())}_{len(self.events)}"
        
        # 创建事件
        event = TemporalEvent(
            id=event_id,
            content=content,
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
            importance=importance,
            context=context or {},
            metadata=metadata or {}
        )
        
        # 存储事件
        self.events[event_id] = event
        
        # 更新时间索引
        self._update_time_index(event)
        
        # 更新类型索引
        self.events_by_type[event_type].add(event_id)
        
        # 检查存储容量
        self._check_capacity()
        
        # 更新统计
        self.performance_stats['events_stored'] += 1
        
        logger.info(f"事件添加: {event_id} ({event_type}), 开始时间: {start_time}")
        
        return event
    
    def _update_time_index(self, event: TemporalEvent):
        """更新时间索引"""
        # 为开始时间创建索引条目
        start_pos = bisect.bisect_left(self.time_index, event.start_time)
        self.time_index.insert(start_pos, event.start_time)
        self.event_time_index.insert(start_pos, event.id)
        
        # 如果事件有持续时间，也为结束时间创建索引（可选）
        if not event.is_instant:
            end_pos = bisect.bisect_left(self.time_index, event.end_time)
            self.time_index.insert(end_pos, event.end_time)
            self.event_time_index.insert(end_pos, event.id)
    
    def _check_capacity(self):
        """检查容量"""
        if len(self.events) > self.max_events:
            # 删除最旧的事件
            self._remove_oldest_events(len(self.events) - self.max_events)
    
    def _remove_oldest_events(self, count: int):
        """删除最旧的事件"""
        if not self.events:
            return
        
        # 按开始时间排序
        sorted_events = sorted(self.events.values(), key=lambda e: e.start_time)
        
        for event in sorted_events[:count]:
            self._remove_event(event.id)
        
        logger.info(f"删除 {count} 个最旧的事件")
    
    def _remove_event(self, event_id: str):
        """移除事件"""
        if event_id not in self.events:
            return
        
        event = self.events[event_id]
        
        # 从时间索引中移除
        start_positions = [i for i, ev_id in enumerate(self.event_time_index) 
                          if ev_id == event_id]
        
        for pos in sorted(start_positions, reverse=True):
            del self.time_index[pos]
            del self.event_time_index[pos]
        
        # 从类型索引中移除
        self.events_by_type[event.event_type].discard(event_id)
        
        # 移除相关时序关系
        self._remove_event_relations(event_id)
        
        # 从存储中删除
        del self.events[event_id]
        
        logger.info(f"事件移除: {event_id}")
    
    def _remove_event_relations(self, event_id: str):
        """移除事件的时序关系"""
        relations_to_remove = []
        
        for i, relation in enumerate(self.temporal_relations):
            if relation.source_event_id == event_id or relation.target_event_id == event_id:
                relations_to_remove.append(i)
        
        # 反向移除以避免索引问题
        for i in reversed(relations_to_remove):
            relation = self.temporal_relations[i]
            
            # 从关系索引中移除
            relation_key = (relation.source_event_id, relation.target_event_id, relation.relation)
            if relation_key in self.relation_index:
                del self.relation_index[relation_key]
            
            del self.temporal_relations[i]
    
    def get_event(self, event_id: str) -> Optional[TemporalEvent]:
        """
        获取事件
        
        Args:
            event_id: 事件ID
            
        Returns:
            事件，如果不存在则返回None
        """
        self.performance_stats['events_retrieved'] += 1
        return self.events.get(event_id)
    
    def find_events_by_time_range(self,
                                 start_time: float,
                                 end_time: float,
                                 event_type: Optional[str] = None) -> List[TemporalEvent]:
        """
        根据时间范围查找事件
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            event_type: 事件类型过滤
            
        Returns:
            事件列表
        """
        start_pos = bisect.bisect_left(self.time_index, start_time)
        end_pos = bisect.bisect_right(self.time_index, end_time)
        
        event_ids = set(self.event_time_index[start_pos:end_pos])
        
        # 过滤事件类型
        if event_type:
            type_event_ids = self.events_by_type.get(event_type, set())
            event_ids = event_ids.intersection(type_event_ids)
        
        # 获取事件并确保它们在时间范围内
        events = []
        for event_id in event_ids:
            if event_id in self.events:
                event = self.events[event_id]
                
                # 检查事件是否与时间范围有重叠
                if self._time_range_overlaps(event, start_time, end_time):
                    events.append(event)
        
        # 按开始时间排序
        events.sort(key=lambda e: e.start_time)
        
        return events
    
    def _time_range_overlaps(self, event: TemporalEvent, start_time: float, end_time: float) -> bool:
        """检查事件是否与时间范围重叠"""
        return not (event.end_time < start_time or event.start_time > end_time)
    
    def analyze_temporal_relations(self, event_id1: str, event_id2: str) -> Optional[TemporalRelation]:
        """
        分析两个事件的时序关系
        
        Args:
            event_id1: 第一个事件ID
            event_id2: 第二个事件ID
            
        Returns:
            时序关系，如果无法确定则返回None
        """
        if event_id1 not in self.events or event_id2 not in self.events:
            return None
        
        event1 = self.events[event_id1]
        event2 = self.events[event_id2]
        
        # 计算时序关系
        relation = self._compute_temporal_relation(event1, event2)
        
        if relation:
            # 记录时序关系
            self._record_temporal_relation(event_id1, event_id2, relation)
        
        return relation
    
    def _compute_temporal_relation(self, event1: TemporalEvent, event2: TemporalEvent) -> Optional[TemporalRelation]:
        """计算两个事件的时序关系"""
        # 简化实现，基于Allen的时间区间代数
        tolerance = self.time_resolution * 0.1
        
        # 检查是否是同一事件
        if event1.id == event2.id:
            return TemporalRelation.EQUALS
        
        # 检查时间重叠情况
        if abs(event1.start_time - event2.start_time) < tolerance:
            if abs(event1.end_time - event2.end_time) < tolerance:
                return TemporalRelation.EQUALS
            elif event1.end_time < event2.end_time:
                return TemporalRelation.STARTS
            else:
                # event2在event1期间开始
                return None
        
        if abs(event1.end_time - event2.end_time) < tolerance:
            if event1.start_time < event2.start_time:
                return TemporalRelation.ENDS
            else:
                # event2在event1期间结束
                return None
        
        # 检查包含关系
        if (event1.start_time <= event2.start_time and 
            event1.end_time >= event2.end_time):
            return TemporalRelation.CONTAINS
        
        if (event2.start_time <= event1.start_time and 
            event2.end_time >= event1.end_time):
            return TemporalRelation.DURING
        
        # 检查重叠关系
        if (event1.start_time < event2.end_time and 
            event2.start_time < event1.end_time):
            return TemporalRelation.OVERLAPS
        
        # 检查先后关系
        if event1.end_time <= event2.start_time:
            if abs(event1.end_time - event2.start_time) < tolerance:
                return TemporalRelation.MEETS
            else:
                return TemporalRelation.BEFORE
        
        if event2.end_time <= event1.start_time:
            if abs(event2.end_time - event1.start_time) < tolerance:
                # event2在event1之前紧接着
                return TemporalRelation.AFTER
            else:
                # event2在event1之前
                return TemporalRelation.AFTER  # 注意：这是从event1的角度看event2
        
        return None
    
    def _record_temporal_relation(self,
                                 source_id: str,
                                 target_id: str,
                                 relation: TemporalRelation,
                                 confidence: float = 1.0):
        """记录时序关系"""
        relation_key = (source_id, target_id, relation)
        
        if relation_key in self.relation_index:
            # 更新现有关系
            existing = self.relation_index[relation_key]
            existing.evidence_count += 1
            existing.last_observed_time = time.time()
            
            # 更新置信度（基于证据数量）
            existing.confidence = min(1.0, 0.5 + 0.5 * (existing.evidence_count / (existing.evidence_count + 1)))
        else:
            # 创建新关系
            relation_record = TemporalRelationRecord(
                source_event_id=source_id,
                target_event_id=target_id,
                relation=relation,
                confidence=confidence
            )
            
            self.temporal_relations.append(relation_record)
            self.relation_index[relation_key] = relation_record
            
            self.performance_stats['temporal_relations_discovered'] += 1
        
        logger.debug(f"时序关系记录: {source_id} {relation.value} {target_id}")
    
    def discover_temporal_patterns(self) -> List[TemporalPattern]:
        """
        发现时序模式
        
        Returns:
            发现的时序模式列表
        """
        if not self.pattern_discovery_enabled:
            return []
        
        current_time = time.time()
        
        # 检查是否达到模式发现间隔
        if current_time - self.last_pattern_discovery_time < self.config['pattern_update_interval']:
            return []
        
        discovered_patterns = []
        
        # 按类型分组事件
        for event_type, event_ids in self.events_by_type.items():
            if len(event_ids) < self.config['min_pattern_length']:
                continue
            
            # 获取事件并按时间排序
            events = [self.events[event_id] for event_id in event_ids 
                     if event_id in self.events]
            events.sort(key=lambda e: e.start_time)
            
            # 发现频繁序列
            frequent_sequences = self._discover_frequent_sequences(events)
            
            # 发现周期性模式
            periodic_patterns = self._discover_periodic_patterns(events)
            
            # 发现因果序列（简化实现）
            causal_sequences = self._discover_causal_sequences(events)
            
            # 整合发现的模式
            all_patterns = frequent_sequences + periodic_patterns + causal_sequences
            
            for pattern in all_patterns:
                # 检查模式是否足够频繁
                if pattern.frequency >= self.config['min_pattern_frequency']:
                    discovered_patterns.append(pattern)
                    
                    # 存储模式
                    self._store_temporal_pattern(pattern)
        
        # 更新时间
        self.last_pattern_discovery_time = current_time
        
        # 更新统计
        self.performance_stats['temporal_patterns_discovered'] += len(discovered_patterns)
        
        logger.info(f"时序模式发现: 发现了 {len(discovered_patterns)} 个模式")
        
        return discovered_patterns
    
    def _discover_frequent_sequences(self, events: List[TemporalEvent]) -> List[TemporalPattern]:
        """发现频繁序列"""
        patterns = []
        
        # 简化实现：查找重复的事件序列
        n_events = len(events)
        max_len = min(self.config['max_pattern_length'], n_events)
        
        for pattern_len in range(self.config['min_pattern_length'], max_len + 1):
            sequence_counts = defaultdict(int)
            
            for i in range(n_events - pattern_len + 1):
                sequence = events[i:i + pattern_len]
                
                # 创建序列键（基于事件类型）
                sequence_key = tuple(e.event_type for e in sequence)
                
                # 计算时间间隔
                time_intervals = []
                for j in range(pattern_len - 1):
                    interval = events[i + j + 1].start_time - events[i + j].end_time
                    time_intervals.append(max(0.0, interval))
                
                sequence_counts[sequence_key] += 1
            
            # 检查频繁序列
            for sequence_key, frequency in sequence_counts.items():
                if frequency >= self.config['min_pattern_frequency']:
                    pattern_id = f"pattern_freq_{len(self.temporal_patterns)}"
                    
                    # 创建事件ID序列（使用实际事件）
                    event_sequence = []
                    for i in range(n_events - pattern_len + 1):
                        if tuple(e.event_type for e in events[i:i + pattern_len]) == sequence_key:
                            event_sequence = [e.id for e in events[i:i + pattern_len]]
                            break
                    
                    pattern = TemporalPattern(
                        pattern_id=pattern_id,
                        pattern_type=TemporalPatternType.FREQUENT_SEQUENCE,
                        event_sequence=event_sequence,
                        time_intervals=[1.0] * (pattern_len - 1),  # 简化
                        frequency=frequency,
                        confidence=min(1.0, frequency / 10.0)  # 基于频率的置信度
                    )
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _discover_periodic_patterns(self, events: List[TemporalEvent]) -> List[TemporalPattern]:
        """发现周期性模式"""
        patterns = []
        
        if len(events) < 3:  # 至少需要3个事件来检测周期性
            return patterns
        
        # 按时间间隔分组相似事件
        event_type_groups = defaultdict(list)
        for event in events:
            event_type_groups[event.event_type].append(event)
        
        for event_type, type_events in event_type_groups.items():
            if len(type_events) < 3:
                continue
            
            # 计算事件之间的时间间隔
            type_events.sort(key=lambda e: e.start_time)
            intervals = []
            
            for i in range(1, len(type_events)):
                interval = type_events[i].start_time - type_events[i-1].start_time
                intervals.append(interval)
            
            # 检查间隔的规律性
            if len(intervals) >= 2:
                # 计算间隔的变异系数
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                if mean_interval > 0 and std_interval / mean_interval < 0.3:  # 变异系数小于0.3
                    # 发现周期性模式
                    pattern_id = f"pattern_periodic_{len(self.temporal_patterns)}"
                    
                    event_sequence = [e.id for e in type_events[:2]]  # 前两个事件
                    
                    pattern = TemporalPattern(
                        pattern_id=pattern_id,
                        pattern_type=TemporalPatternType.PERIODIC,
                        event_sequence=event_sequence,
                        time_intervals=[mean_interval],
                        frequency=len(type_events),
                        confidence=min(1.0, 1.0 - (std_interval / mean_interval))
                    )
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _discover_causal_sequences(self, events: List[TemporalEvent]) -> List[TemporalPattern]:
        """发现因果序列"""
        patterns = []
        
        # 简化实现：查找经常按顺序出现的事件对
        n_events = len(events)
        
        for i in range(n_events - 1):
            event1 = events[i]
            event2 = events[i + 1]
            
            # 检查时序关系
            relation = self._compute_temporal_relation(event1, event2)
            
            if relation in [TemporalRelation.BEFORE, TemporalRelation.MEETS]:
                # 记录这种顺序出现的频率
                sequence_key = (event1.event_type, event2.event_type)
                
                # 在实际实现中，这里应该有更复杂的因果发现逻辑
                # 简化：总是创建因果序列模式
                pattern_id = f"pattern_causal_{len(self.temporal_patterns)}"
                
                pattern = TemporalPattern(
                    pattern_id=pattern_id,
                    pattern_type=TemporalPatternType.CAUSAL_SEQUENCE,
                    event_sequence=[event1.id, event2.id],
                    time_intervals=[event2.start_time - event1.end_time],
                    frequency=2,  # 简化
                    confidence=0.7  # 中等置信度
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _store_temporal_pattern(self, pattern: TemporalPattern):
        """存储时序模式"""
        self.temporal_patterns[pattern.pattern_id] = pattern
        self.patterns_by_type[pattern.pattern_type].add(pattern.pattern_id)
    
    def reconstruct_timeline(self,
                            start_time: float,
                            end_time: float,
                            include_events: bool = True,
                            include_patterns: bool = True) -> Dict[str, Any]:
        """
        重建时间线
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            include_events: 是否包含事件
            include_patterns: 是否包含模式
            
        Returns:
            时间线数据
        """
        timeline = {
            'start_time': start_time,
            'end_time': end_time,
            'events': [],
            'patterns': [],
            'relations': []
        }
        
        if include_events:
            # 获取时间范围内的事件
            events = self.find_events_by_time_range(start_time, end_time)
            timeline['events'] = [{
                'id': event.id,
                'content': str(event.content)[:100],  # 截断内容
                'start_time': event.start_time,
                'end_time': event.end_time,
                'event_type': event.event_type,
                'importance': event.importance
            } for event in events]
        
        if include_patterns:
            # 获取相关的时序模式
            patterns = self.get_patterns_in_time_range(start_time, end_time)
            timeline['patterns'] = [{
                'id': pattern.pattern_id,
                'type': pattern.pattern_type.value,
                'event_count': len(pattern.event_sequence),
                'frequency': pattern.frequency,
                'confidence': pattern.confidence
            } for pattern in patterns]
        
        # 获取时序关系
        timeline['relations'] = self.get_relations_in_time_range(start_time, end_time)
        
        self.performance_stats['timelines_reconstructed'] += 1
        
        return timeline
    
    def get_patterns_in_time_range(self, start_time: float, end_time: float) -> List[TemporalPattern]:
        """获取时间范围内的时序模式"""
        patterns_in_range = []
        
        for pattern in self.temporal_patterns.values():
            # 检查模式中的事件是否在时间范围内
            pattern_events_in_range = []
            
            for event_id in pattern.event_sequence:
                if event_id in self.events:
                    event = self.events[event_id]
                    if self._time_range_overlaps(event, start_time, end_time):
                        pattern_events_in_range.append(event_id)
            
            if pattern_events_in_range:
                patterns_in_range.append(pattern)
        
        return patterns_in_range
    
    def get_relations_in_time_range(self, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """获取时间范围内的时序关系"""
        relations_in_range = []
        
        for relation in self.temporal_relations:
            # 检查关系中的事件是否在时间范围内
            source_event = self.events.get(relation.source_event_id)
            target_event = self.events.get(relation.target_event_id)
            
            if source_event and target_event:
                if (self._time_range_overlaps(source_event, start_time, end_time) or
                    self._time_range_overlaps(target_event, start_time, end_time)):
                    
                    relations_in_range.append({
                        'source': relation.source_event_id,
                        'target': relation.target_event_id,
                        'relation': relation.relation.value,
                        'confidence': relation.confidence,
                        'evidence_count': relation.evidence_count
                    })
        
        return relations_in_range
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 事件统计
        event_counts_by_type = {t: len(ids) for t, ids in self.events_by_type.items()}
        
        # 关系统计
        relation_counts_by_type = defaultdict(int)
        for relation in self.temporal_relations:
            relation_counts_by_type[relation.relation.value] += 1
        
        # 模式统计
        pattern_counts_by_type = {t.value: len(ids) for t, ids in self.patterns_by_type.items()}
        
        return {
            'total_events': len(self.events),
            'event_counts_by_type': event_counts_by_type,
            'total_temporal_relations': len(self.temporal_relations),
            'relation_counts_by_type': dict(relation_counts_by_type),
            'total_temporal_patterns': len(self.temporal_patterns),
            'pattern_counts_by_type': pattern_counts_by_type,
            'performance_stats': self.performance_stats
        }


# 示例和测试函数
def create_example_temporal_store() -> TemporalMemoryStore:
    """创建示例时序记忆存储"""
    store = TemporalMemoryStore(
        max_events=1000,
        time_resolution=1.0,
        pattern_discovery_enabled=True
    )
    return store


def test_temporal_memory_store():
    """测试时序记忆存储"""
    logger.info("开始测试时序记忆存储")
    
    # 创建示例存储
    store = create_example_temporal_store()
    
    current_time = time.time()
    
    # 添加事件
    logger.info("添加事件...")
    
    # 事件1：早晨起床
    event1 = store.add_event(
        content="早晨起床",
        start_time=current_time - 3600 * 2,  # 2小时前
        end_time=current_time - 3600 * 2 + 300,  # 持续5分钟
        event_type="daily_activity",
        importance=0.6,
        context={"location": "卧室", "activity": "起床"}
    )
    
    # 事件2：早餐
    event2 = store.add_event(
        content="早餐",
        start_time=current_time - 3600 * 1.8,  # 1.8小时前
        end_time=current_time - 3600 * 1.8 + 1200,  # 持续20分钟
        event_type="meal",
        importance=0.7,
        context={"location": "厨房", "meal": "早餐"}
    )
    
    # 事件3：工作开始
    event3 = store.add_event(
        content="开始工作",
        start_time=current_time - 3600 * 1.5,  # 1.5小时前
        end_time=current_time - 3600 * 1.5 + 600,  # 持续10分钟
        event_type="work",
        importance=0.9,
        context={"location": "办公室", "task": "工作"}
    )
    
    # 分析时序关系
    logger.info("分析时序关系...")
    relation1 = store.analyze_temporal_relations(event1.id, event2.id)
    relation2 = store.analyze_temporal_relations(event2.id, event3.id)
    
    if relation1:
        logger.info(f"事件关系 {event1.id} -> {event2.id}: {relation1.value}")
    if relation2:
        logger.info(f"事件关系 {event2.id} -> {event3.id}: {relation2.value}")
    
    # 查找时间范围内的事件
    logger.info("查找时间范围内的事件...")
    start_time = current_time - 3600 * 3  # 3小时前
    end_time = current_time  # 现在
    
    events_in_range = store.find_events_by_time_range(start_time, end_time)
    logger.info(f"在时间范围内找到 {len(events_in_range)} 个事件")
    
    # 发现时序模式
    logger.info("发现时序模式...")
    patterns = store.discover_temporal_patterns()
    logger.info(f"发现了 {len(patterns)} 个时序模式")
    
    for pattern in patterns[:3]:
        logger.info(f"  模式: {pattern.pattern_type.value}，频率: {pattern.frequency}，置信度: {pattern.confidence:.2f}")
    
    # 重建时间线
    logger.info("重建时间线...")
    timeline = store.reconstruct_timeline(
        start_time=current_time - 3600 * 3,
        end_time=current_time,
        include_events=True,
        include_patterns=True
    )
    
    logger.info(f"时间线包含 {len(timeline['events'])} 个事件，{len(timeline['patterns'])} 个模式")
    
    # 获取统计信息
    stats = store.get_statistics()
    logger.info(f"统计信息: {stats['total_events']} 个事件，{stats['total_temporal_relations']} 个关系")
    
    logger.info("时序记忆存储测试完成")
    return store


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_temporal_store_instance = test_temporal_memory_store()