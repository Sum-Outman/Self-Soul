#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zlib
"""
综合记忆系统 - 实现情景、语义和程序记忆的完整体系

核心功能:
1. 情景记忆: 存储具体经历的时间序列和情境
2. 语义记忆: 存储概念、事实和因果关系
3. 程序记忆: 存储技能、动作序列和操作流程
4. 记忆巩固: 从情景记忆中提取规律形成语义记忆
5. 记忆检索: 基于内容的相似性检索和情境召回
6. 遗忘机制: 基于重要性、频率和时间的智能遗忘

记忆类型详解:
1. 情景记忆 (Episodic Memory):
   - 存储: 具体事件的时间、地点、人物、感受
   - 特点: 时间序列性、情境依赖性、细节丰富
   - 用途: 个人经历回顾、情境学习、模式识别

2. 语义记忆 (Semantic Memory):
   - 存储: 概念、事实、规则、因果关系
   - 特点: 抽象性、通用性、关系网络
   - 用途: 知识推理、概念理解、问题解决

3. 程序记忆 (Procedural Memory):
   - 存储: 技能、动作序列、操作流程
   - 特点: 自动化、序列性、条件触发
   - 用途: 技能执行、习惯形成、动作优化

记忆过程:
1. 编码 (Encoding): 感知信息转化为记忆表示
2. 存储 (Storage): 记忆在长期存储中的保持
3. 巩固 (Consolidation): 短期记忆转化为长期记忆
4. 检索 (Retrieval): 从存储中提取记忆内容
5. 遗忘 (Forgetting): 选择性删除不重要记忆

技术实现:
- 多模态编码和存储
- 基于内容的相似性检索
- 记忆关联网络
- 遗忘曲线建模
- 记忆巩固算法

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from enum import Enum
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import networkx as nx
from heapq import heappush, heappop

# 导入错误处理
from core.error_handling import ErrorHandler

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()


class MemoryType(Enum):
    """记忆类型枚举"""
    EPISODIC = "episodic"      # 情景记忆: 具体事件
    SEMANTIC = "semantic"      # 语义记忆: 概念事实
    PROCEDURAL = "procedural"  # 程序记忆: 技能流程


class MemoryEncodingStrength(Enum):
    """记忆编码强度枚举"""
    WEAK = "weak"        # 弱编码: 单次接触，注意力低
    MODERATE = "moderate"  # 中等编码: 重复接触，中等注意力
    STRONG = "strong"    # 强编码: 深度处理，高注意力，情感关联


class MemoryRetrievalType(Enum):
    """记忆检索类型枚举"""
    RECALL = "recall"          # 回忆: 主动提取记忆内容
    RECOGNITION = "recognition"  # 再认: 识别是否见过
    CUED = "cued"              # 线索提示: 基于线索提取
    ASSOCIATIVE = "associative"  # 关联检索: 通过相关记忆提取


class MemoryConsolidationStage(Enum):
    """记忆巩固阶段枚举"""
    SHORT_TERM = "short_term"      # 短期记忆: 几分钟到几小时
    INTERMEDIATE = "intermediate"  # 中期记忆: 几小时到几天
    LONG_TERM = "long_term"        # 长期记忆: 几天到永久


@dataclass
class MemoryItem:
    """记忆项数据类"""
    id: str
    content: Any
    memory_type: MemoryType
    encoding_strength: MemoryEncodingStrength
    encoding_time: float
    importance: float = 1.0
    emotional_valence: float = 0.0  # 情感效价: -1.0(负面)到1.0(正面)
    emotional_arousal: float = 0.0  # 情感唤醒: 0.0(低)到1.0(高)
    context: Dict[str, Any] = field(default_factory=dict)
    associations: List[str] = field(default_factory=list)
    retrieval_count: int = 0
    last_retrieval_time: Optional[float] = None
    consolidation_stage: MemoryConsolidationStage = MemoryConsolidationStage.SHORT_TERM
    decay_rate: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.importance = max(0.0, min(1.0, self.importance))
        self.emotional_valence = max(-1.0, min(1.0, self.emotional_valence))
        self.emotional_arousal = max(0.0, min(1.0, self.emotional_arousal))
        self.decay_rate = max(0.0, min(1.0, self.decay_rate))
        if not self.id:
            self.id = f"memory_{int(time.time())}_{(zlib.adler32(str(str(self.content).encode('utf-8')) & 0xffffffff)) % 10000}"


@dataclass
class MemoryAssociation:
    """记忆关联数据类"""
    source_id: str
    target_id: str
    association_type: str
    strength: float = 1.0
    created_time: float = field(default_factory=time.time)
    last_activated_time: Optional[float] = None
    activation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.strength = max(0.0, min(1.0, self.strength))
        if self.last_activated_time is None:
            self.last_activated_time = self.created_time


@dataclass
class MemoryRetrievalResult:
    """记忆检索结果数据类"""
    memory_item: MemoryItem
    retrieval_type: MemoryRetrievalType
    retrieval_time: float
    confidence: float = 1.0
    relevance_score: float = 1.0
    retrieved_associations: List[MemoryAssociation] = field(default_factory=list)
    context_matches: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))


@dataclass
class MemoryConsolidationResult:
    """记忆巩固结果数据类"""
    memory_id: str
    from_stage: MemoryConsolidationStage
    to_stage: MemoryConsolidationStage
    consolidation_time: float
    strength_increase: float = 0.0
    decay_rate_decrease: float = 0.0
    new_associations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """后初始化验证"""
        self.strength_increase = max(0.0, self.strength_increase)
        self.decay_rate_decrease = max(0.0, self.decay_rate_decrease)


class EpisodicSemanticMemory:
    """
    综合记忆系统管理器
    
    核心组件:
    1. 记忆编码器: 将信息编码为记忆表示
    2. 记忆存储器: 存储不同类型和强度的记忆
    3. 记忆检索器: 基于内容和情境检索记忆
    4. 记忆巩固器: 将短期记忆转化为长期记忆
    5. 记忆关联器: 建立记忆之间的联系
    6. 遗忘管理器: 管理记忆的遗忘和删除
    
    工作流程:
    输入信息 → 记忆编码器 → 记忆表示 → 记忆存储器 → 存储记忆
    检索请求 → 记忆检索器 → 检索记忆 → 返回结果
    时间推移 → 记忆巩固器 → 巩固记忆 → 记忆关联器 → 增强关联
    定期清理 → 遗忘管理器 → 评估重要性 → 选择性遗忘
    
    技术特性:
    - 多模态记忆编码和存储
    - 基于内容和情境的智能检索
    - 记忆巩固和长期存储
    - 记忆关联网络
    - 自适应遗忘机制
    """
    
    def __init__(self,
                 max_memory_items: int = 10000,
                 short_term_capacity: int = 100,
                 consolidation_interval: float = 3600.0,  # 秒
                 retrieval_threshold: float = 0.3,
                 forgetting_threshold: float = 0.1):
        """
        初始化综合记忆系统
        
        Args:
            max_memory_items: 最大记忆项数量
            short_term_capacity: 短期记忆容量
            consolidation_interval: 巩固间隔时间（秒）
            retrieval_threshold: 检索阈值
            forgetting_threshold: 遗忘阈值
        """
        self.max_memory_items = max_memory_items
        self.short_term_capacity = short_term_capacity
        self.consolidation_interval = consolidation_interval
        self.retrieval_threshold = retrieval_threshold
        self.forgetting_threshold = forgetting_threshold
        
        # 记忆存储
        self.memory_items: Dict[str, MemoryItem] = {}
        self.memory_associations: List[MemoryAssociation] = []
        
        # 索引结构
        self.memory_by_type: Dict[MemoryType, Set[str]] = defaultdict(set)
        self.memory_by_context: Dict[str, Set[str]] = defaultdict(set)
        self.memory_association_graph = nx.MultiGraph()
        
        # 短期记忆队列
        self.short_term_memory: deque = deque(maxlen=short_term_capacity)
        
        # 记忆巩固状态
        self.last_consolidation_time = time.time()
        
        # 配置参数
        self.config = {
            'encoding_strength_weights': {
                'weak': 0.3,
                'moderate': 0.6,
                'strong': 0.9
            },
            'retrieval_decay_factor': 0.95,
            'association_strength_decay': 0.99,
            'emotional_enhancement_factor': 1.5,
            'consolidation_strength_increase': 0.1,
            'consolidation_decay_decrease': 0.05,
            'forgetting_curve_alpha': 0.7,
            'forgetting_curve_beta': 0.2,
            'rehearsal_effect': 0.2
        }
        
        # 性能统计
        self.performance_stats = {
            'memory_items_encoded': 0,
            'memory_items_retrieved': 0,
            'memory_associations_created': 0,
            'memory_consolidations': 0,
            'memory_forgettings': 0,
            'average_encoding_time': 0.0,
            'average_retrieval_time': 0.0,
            'retrieval_success_rate': 0.0
        }
        
        logger.info(f"综合记忆系统初始化完成，最大记忆项: {max_memory_items}，短期记忆容量: {short_term_capacity}")
    
    def encode_memory(self,
                     content: Any,
                     memory_type: MemoryType,
                     encoding_strength: MemoryEncodingStrength = MemoryEncodingStrength.MODERATE,
                     importance: float = 0.5,
                     emotional_valence: float = 0.0,
                     emotional_arousal: float = 0.0,
                     context: Optional[Dict[str, Any]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> MemoryItem:
        """
        编码记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            encoding_strength: 编码强度
            importance: 重要性 (0.0-1.0)
            emotional_valence: 情感效价 (-1.0-1.0)
            emotional_arousal: 情感唤醒 (0.0-1.0)
            context: 情境信息
            metadata: 元数据
            
        Returns:
            编码的记忆项
        """
        start_time = time.time()
        
        # 生成记忆ID
        memory_id = f"memory_{int(time.time())}_{(zlib.adler32(str(str(content).encode('utf-8')) & 0xffffffff)) % 10000}"
        
        # 情感增强编码
        emotional_enhancement = 1.0 + abs(emotional_valence) * self.config['emotional_enhancement_factor']
        effective_importance = min(1.0, importance * emotional_enhancement)
        
        # 计算遗忘率
        base_decay_rate = 0.1
        encoding_weight = self.config['encoding_strength_weights'][encoding_strength.value]
        decay_rate = base_decay_rate * (1.0 - encoding_weight * 0.5)
        
        # 创建记忆项
        memory_item = MemoryItem(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            encoding_strength=encoding_strength,
            encoding_time=time.time(),
            importance=effective_importance,
            emotional_valence=emotional_valence,
            emotional_arousal=emotional_arousal,
            context=context or {},
            associations=[],
            consolidation_stage=MemoryConsolidationStage.SHORT_TERM,
            decay_rate=decay_rate,
            metadata=metadata or {}
        )
        
        # 存储记忆项
        self.memory_items[memory_id] = memory_item
        
        # 更新索引
        self.memory_by_type[memory_type].add(memory_id)
        
        for key, value in (context or {}).items():
            context_key = f"{key}:{value}"
            self.memory_by_context[context_key].add(memory_id)
        
        # 添加到短期记忆
        self.short_term_memory.append(memory_id)
        
        # 更新性能统计
        encoding_time = time.time() - start_time
        self._update_encoding_stats(encoding_time)
        
        logger.info(f"记忆编码完成: {memory_id} ({memory_type.value}, {encoding_strength.value}), 重要性: {effective_importance:.2f}")
        
        return memory_item
    
    def _update_encoding_stats(self, encoding_time: float):
        """更新编码统计"""
        self.performance_stats['memory_items_encoded'] += 1
        
        # 更新平均编码时间
        current_avg = self.performance_stats['average_encoding_time']
        n_encoded = self.performance_stats['memory_items_encoded']
        
        new_avg = (current_avg * (n_encoded - 1) + encoding_time) / n_encoded
        self.performance_stats['average_encoding_time'] = new_avg
    
    def retrieve_memory(self,
                       query: Any,
                       memory_type: Optional[MemoryType] = None,
                       context: Optional[Dict[str, Any]] = None,
                       retrieval_type: MemoryRetrievalType = MemoryRetrievalType.RECALL,
                       max_results: int = 10) -> List[MemoryRetrievalResult]:
        """
        检索记忆
        
        Args:
            query: 检索查询
            memory_type: 限制记忆类型
            context: 情境过滤
            retrieval_type: 检索类型
            max_results: 最大结果数量
            
        Returns:
            检索结果列表
        """
        start_time = time.time()
        
        # 计算查询表示
        query_representation = self._compute_query_representation(query)
        
        # 候选记忆项
        candidate_memory_ids = self._get_candidate_memory_ids(memory_type, context)
        
        # 计算相关性分数
        scored_memories = []
        
        for memory_id in candidate_memory_ids:
            if memory_id not in self.memory_items:
                continue
            
            memory_item = self.memory_items[memory_id]
            
            # 计算相关性
            relevance_score = self._compute_relevance_score(memory_item, query_representation, context)
            
            # 应用遗忘曲线
            time_since_encoding = time.time() - memory_item.encoding_time
            forgetting_factor = self._compute_forgetting_factor(memory_item, time_since_encoding)
            
            # 最终分数
            final_score = relevance_score * forgetting_factor
            
            if final_score >= self.retrieval_threshold:
                scored_memories.append((final_score, memory_item))
        
        # 按分数排序
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        # 获取关联记忆
        retrieval_results = []
        
        for score, memory_item in scored_memories[:max_results]:
            # 检索关联记忆
            associations = self._retrieve_associated_memories(memory_item.id)
            
            # 更新记忆项的检索信息
            memory_item.retrieval_count += 1
            memory_item.last_retrieval_time = time.time()
            
            # 创建检索结果
            result = MemoryRetrievalResult(
                memory_item=memory_item,
                retrieval_type=retrieval_type,
                retrieval_time=time.time(),
                confidence=score,
                relevance_score=relevance_score,
                retrieved_associations=associations,
                context_matches=self._find_context_matches(memory_item, context)
            )
            
            retrieval_results.append(result)
        
        # 更新性能统计
        retrieval_time = time.time() - start_time
        self._update_retrieval_stats(retrieval_time, len(retrieval_results) > 0)
        
        logger.info(f"记忆检索完成: 查询类型 {retrieval_type.value}, 找到 {len(retrieval_results)} 个结果")
        
        return retrieval_results
    
    def _compute_query_representation(self, query: Any) -> Any:
        """计算查询表示"""
        # 简化实现：返回查询本身
        # 实际实现应该使用嵌入模型
        return str(query)
    
    def _get_candidate_memory_ids(self, 
                                 memory_type: Optional[MemoryType],
                                 context: Optional[Dict[str, Any]]) -> Set[str]:
        """获取候选记忆ID"""
        if memory_type and context:
            # 类型和情境都指定
            type_ids = self.memory_by_type.get(memory_type, set())
            context_ids = set()
            
            for key, value in context.items():
                context_key = f"{key}:{value}"
                context_ids.update(self.memory_by_context.get(context_key, set()))
            
            return type_ids.intersection(context_ids)
        
        elif memory_type:
            # 只指定类型
            return self.memory_by_type.get(memory_type, set())
        
        elif context:
            # 只指定情境
            context_ids = set()
            for key, value in context.items():
                context_key = f"{key}:{value}"
                context_ids.update(self.memory_by_context.get(context_key, set()))
            
            return context_ids
        
        else:
            # 无限制，返回所有记忆ID
            return set(self.memory_items.keys())
    
    def _compute_relevance_score(self,
                                memory_item: MemoryItem,
                                query_representation: Any,
                                context: Optional[Dict[str, Any]]) -> float:
        """计算相关性分数"""
        # 简化实现：基于内容相似性
        try:
            # 内容相似性
            content_similarity = self._compute_similarity(memory_item.content, query_representation)
            
            # 情境匹配
            context_match = 1.0
            if context:
                context_overlap = self._compute_context_overlap(memory_item.context, context)
                context_match = 0.3 + 0.7 * context_overlap  # 情境匹配权重
            
            # 编码强度影响
            encoding_weight = self.config['encoding_strength_weights'][memory_item.encoding_strength.value]
            
            # 重要性影响
            importance_weight = 0.5 + 0.5 * memory_item.importance
            
            # 综合分数
            relevance_score = content_similarity * context_match * encoding_weight * importance_weight
            
            return min(1.0, relevance_score)
            
        except Exception as e:
            logger.error(f"相关性计算失败: {e}")
            return 0.5
    
    def _compute_similarity(self, content1: Any, content2: Any) -> float:
        """计算相似性"""
        # 简化实现
        if content1 == content2:
            return 1.0
        
        # 字符串相似性
        if isinstance(content1, str) and isinstance(content2, str):
            # 简单的词重叠
            words1 = set(content1.lower().split())
            words2 = set(content2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
        
        # 数值相似性
        if isinstance(content1, (int, float)) and isinstance(content2, (int, float)):
            diff = abs(content1 - content2)
            max_val = max(abs(content1), abs(content2), 1.0)
            return 1.0 - min(1.0, diff / max_val)
        
        # 默认相似性
        return 0.3
    
    def _compute_context_overlap(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """计算情境重叠"""
        if not context1 or not context2:
            return 0.0
        
        common_keys = set(context1.keys()).intersection(set(context2.keys()))
        if not common_keys:
            return 0.0
        
        matches = 0
        for key in common_keys:
            if context1[key] == context2[key]:
                matches += 1
        
        return matches / len(common_keys)
    
    def _compute_forgetting_factor(self, memory_item: MemoryItem, time_since_encoding: float) -> float:
        """计算遗忘因子"""
        # 艾宾浩斯遗忘曲线模型
        # 公式: R = e^(-(t/S)^β)，其中R是记忆保留率，t是时间，S是记忆强度
        
        # 记忆强度因子
        encoding_weight = self.config['encoding_strength_weights'][memory_item.encoding_strength.value]
        emotional_factor = 1.0 + abs(memory_item.emotional_valence) * 0.5
        importance_factor = 0.5 + 0.5 * memory_item.importance
        
        strength = encoding_weight * emotional_factor * importance_factor
        
        # 检索次数增强
        rehearsal_factor = 1.0 + memory_item.retrieval_count * self.config['rehearsal_effect']
        
        # 时间衰减
        t = time_since_encoding / 3600.0  # 转换为小时
        alpha = self.config['forgetting_curve_alpha']
        beta = self.config['forgetting_curve_beta']
        
        # 遗忘曲线
        if t <= 0:
            retention = 1.0
        else:
            retention = math.exp(-(t / (strength * rehearsal_factor * alpha)) ** beta)
        
        return max(self.forgetting_threshold, retention)
    
    def _retrieve_associated_memories(self, memory_id: str) -> List[MemoryAssociation]:
        """检索关联记忆"""
        associations = []
        
        for assoc in self.memory_associations:
            if assoc.source_id == memory_id or assoc.target_id == memory_id:
                associations.append(assoc)
                
                # 更新关联激活
                assoc.activation_count += 1
                assoc.last_activated_time = time.time()
        
        return associations
    
    def _find_context_matches(self, memory_item: MemoryItem, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """查找情境匹配"""
        if not context:
            return {}
        
        matches = {}
        for key, value in context.items():
            if key in memory_item.context and memory_item.context[key] == value:
                matches[key] = value
        
        return matches
    
    def _update_retrieval_stats(self, retrieval_time: float, success: bool):
        """更新检索统计"""
        self.performance_stats['memory_items_retrieved'] += 1
        
        # 更新平均检索时间
        current_avg = self.performance_stats['average_retrieval_time']
        n_retrieved = self.performance_stats['memory_items_retrieved']
        
        new_avg = (current_avg * (n_retrieved - 1) + retrieval_time) / n_retrieved
        self.performance_stats['average_retrieval_time'] = new_avg
        
        # 更新成功率
        current_success_rate = self.performance_stats['retrieval_success_rate']
        new_success_rate = (current_success_rate * (n_retrieved - 1) + (1.0 if success else 0.0)) / n_retrieved
        self.performance_stats['retrieval_success_rate'] = new_success_rate
    
    def create_association(self,
                          source_id: str,
                          target_id: str,
                          association_type: str,
                          strength: float = 0.8) -> Optional[MemoryAssociation]:
        """
        创建记忆关联
        
        Args:
            source_id: 源记忆ID
            target_id: 目标记忆ID
            association_type: 关联类型
            strength: 关联强度
            
        Returns:
            创建的关联，如果失败则返回None
        """
        if source_id not in self.memory_items or target_id not in self.memory_items:
            logger.error(f"记忆关联创建失败: 源或目标记忆不存在 ({source_id}, {target_id})")
            return None
        
        # 检查是否已存在关联
        existing_assoc = self._find_existing_association(source_id, target_id, association_type)
        if existing_assoc:
            # 增强现有关联
            existing_assoc.strength = min(1.0, existing_assoc.strength + 0.1)
            existing_assoc.last_activated_time = time.time()
            logger.info(f"记忆关联增强: {source_id} -> {target_id} ({association_type}), 新强度: {existing_assoc.strength:.2f}")
            return existing_assoc
        
        # 创建新关联
        association = MemoryAssociation(
            source_id=source_id,
            target_id=target_id,
            association_type=association_type,
            strength=strength
        )
        
        self.memory_associations.append(association)
        
        # 更新记忆项的关联列表
        self.memory_items[source_id].associations.append(association.target_id)
        self.memory_items[target_id].associations.append(association.source_id)
        
        # 更新关联图
        self.memory_association_graph.add_edge(
            source_id, target_id,
            association_type=association_type,
            strength=strength,
            created_time=association.created_time
        )
        
        self.performance_stats['memory_associations_created'] += 1
        
        logger.info(f"记忆关联创建: {source_id} -> {target_id} ({association_type}), 强度: {strength:.2f}")
        
        return association
    
    def _find_existing_association(self, 
                                 source_id: str, 
                                 target_id: str, 
                                 association_type: str) -> Optional[MemoryAssociation]:
        """查找现有关联"""
        for assoc in self.memory_associations:
            if (assoc.source_id == source_id and assoc.target_id == target_id and 
                assoc.association_type == association_type):
                return assoc
        
        return None
    
    def consolidate_memories(self) -> List[MemoryConsolidationResult]:
        """
        巩固记忆（将短期记忆转化为长期记忆）
        
        Returns:
            巩固结果列表
        """
        current_time = time.time()
        
        # 检查是否达到巩固间隔
        if current_time - self.last_consolidation_time < self.consolidation_interval:
            return []
        
        consolidation_results = []
        
        # 检查短期记忆中的记忆项
        for memory_id in list(self.short_term_memory):
            if memory_id not in self.memory_items:
                continue
            
            memory_item = self.memory_items[memory_id]
            
            # 检查是否可以巩固
            if memory_item.consolidation_stage == MemoryConsolidationStage.SHORT_TERM:
                # 计算巩固概率
                consolidation_probability = self._compute_consolidation_probability(memory_item)
                
                if consolidation_probability > 0.7:  # 巩固阈值
                    # 执行巩固
                    result = self._consolidate_memory_item(memory_item)
                    consolidation_results.append(result)
                    
                    # 从短期记忆中移除（如果转化为长期记忆）
                    if result.to_stage == MemoryConsolidationStage.LONG_TERM:
                        try:
                            self.short_term_memory.remove(memory_id)
                        except ValueError:
                            pass
        
        # 更新巩固时间
        self.last_consolidation_time = current_time
        
        # 更新性能统计
        self.performance_stats['memory_consolidations'] += len(consolidation_results)
        
        logger.info(f"记忆巩固完成: {len(consolidation_results)} 个记忆项被巩固")
        
        return consolidation_results
    
    def _compute_consolidation_probability(self, memory_item: MemoryItem) -> float:
        """计算巩固概率"""
        # 基于重要性、情感强度和检索频率
        importance_factor = memory_item.importance
        
        emotional_factor = 0.5 + 0.5 * (abs(memory_item.emotional_valence) + memory_item.emotional_arousal)
        
        retrieval_factor = 1.0 - math.exp(-memory_item.retrieval_count * 0.5)
        
        # 时间因子（编码时间越近，巩固概率越高）
        time_since_encoding = time.time() - memory_item.encoding_time
        time_factor = max(0.0, 1.0 - time_since_encoding / (24 * 3600))  # 24小时内
        
        # 综合概率
        probability = (importance_factor * 0.4 + 
                      emotional_factor * 0.3 + 
                      retrieval_factor * 0.2 + 
                      time_factor * 0.1)
        
        return min(1.0, probability)
    
    def _consolidate_memory_item(self, memory_item: MemoryItem) -> MemoryConsolidationResult:
        """巩固单个记忆项"""
        from_stage = memory_item.consolidation_stage
        
        # 确定下一个阶段
        if from_stage == MemoryConsolidationStage.SHORT_TERM:
            to_stage = MemoryConsolidationStage.INTERMEDIATE
        elif from_stage == MemoryConsolidationStage.INTERMEDIATE:
            to_stage = MemoryConsolidationStage.LONG_TERM
        else:
            to_stage = MemoryConsolidationStage.LONG_TERM
        
        # 更新记忆项
        memory_item.consolidation_stage = to_stage
        
        # 增加记忆强度（降低遗忘率）
        strength_increase = self.config['consolidation_strength_increase']
        decay_decrease = self.config['consolidation_decay_decrease']
        
        memory_item.decay_rate = max(0.01, memory_item.decay_rate - decay_decrease)
        
        # 创建新关联（巩固过程中可能形成新关联）
        new_associations = self._create_consolidation_associations(memory_item)
        
        result = MemoryConsolidationResult(
            memory_id=memory_item.id,
            from_stage=from_stage,
            to_stage=to_stage,
            consolidation_time=time.time(),
            strength_increase=strength_increase,
            decay_rate_decrease=decay_decrease,
            new_associations=new_associations
        )
        
        logger.info(f"记忆巩固: {memory_item.id} {from_stage.value} -> {to_stage.value}")
        
        return result
    
    def _create_consolidation_associations(self, memory_item: MemoryItem) -> List[str]:
        """创建巩固关联"""
        new_associations = []
        
        # 查找相关记忆项（基于内容和情境）
        candidate_memory_ids = self._find_related_memories(memory_item)
        
        for candidate_id in candidate_memory_ids[:3]:  # 最多创建3个新关联
            if candidate_id != memory_item.id and candidate_id in self.memory_items:
                # 创建关联
                association = self.create_association(
                    source_id=memory_item.id,
                    target_id=candidate_id,
                    association_type="consolidation",
                    strength=0.6
                )
                
                if association:
                    new_associations.append(association.target_id)
        
        return new_associations
    
    def _find_related_memories(self, memory_item: MemoryItem) -> List[str]:
        """查找相关记忆"""
        # 基于内容和情境的简单查找
        related_ids = set()
        
        # 相同类型的记忆
        same_type_ids = self.memory_by_type.get(memory_item.memory_type, set())
        related_ids.update(same_type_ids)
        
        # 相似情境的记忆
        for key, value in memory_item.context.items():
            context_key = f"{key}:{value}"
            context_ids = self.memory_by_context.get(context_key, set())
            related_ids.update(context_ids)
        
        # 排除自身
        related_ids.discard(memory_item.id)
        
        return list(related_ids)
    
    def manage_forgetting(self) -> List[str]:
        """
        管理记忆遗忘
        
        Returns:
            被遗忘的记忆ID列表
        """
        forgotten_memory_ids = []
        current_time = time.time()
        
        # 检查所有记忆项
        for memory_id, memory_item in list(self.memory_items.items()):
            # 计算记忆强度
            time_since_encoding = current_time - memory_item.encoding_time
            forgetting_factor = self._compute_forgetting_factor(memory_item, time_since_encoding)
            
            # 如果记忆强度低于遗忘阈值，且不是重要记忆
            if (forgetting_factor < self.forgetting_threshold and 
                memory_item.importance < 0.3 and
                memory_item.consolidation_stage != MemoryConsolidationStage.LONG_TERM):
                
                # 遗忘记忆
                self._forget_memory(memory_id)
                forgotten_memory_ids.append(memory_id)
        
        # 更新性能统计
        self.performance_stats['memory_forgettings'] += len(forgotten_memory_ids)
        
        if forgotten_memory_ids:
            logger.info(f"记忆遗忘完成: {len(forgotten_memory_ids)} 个记忆项被遗忘")
        
        return forgotten_memory_ids
    
    def _forget_memory(self, memory_id: str):
        """遗忘单个记忆"""
        if memory_id not in self.memory_items:
            return
        
        memory_item = self.memory_items[memory_id]
        
        # 从索引中移除
        self.memory_by_type[memory_item.memory_type].discard(memory_id)
        
        for key, value in memory_item.context.items():
            context_key = f"{key}:{value}"
            self.memory_by_context[context_key].discard(memory_id)
        
        # 移除关联
        self._remove_memory_associations(memory_id)
        
        # 从存储中删除
        del self.memory_items[memory_id]
        
        # 从短期记忆中移除
        try:
            self.short_term_memory.remove(memory_id)
        except ValueError:
            pass
        
        logger.info(f"记忆遗忘: {memory_id} ({memory_item.memory_type.value})")
    
    def _remove_memory_associations(self, memory_id: str):
        """移除记忆关联"""
        # 移除关联列表中的条目
        associations_to_remove = []
        
        for i, assoc in enumerate(self.memory_associations):
            if assoc.source_id == memory_id or assoc.target_id == memory_id:
                associations_to_remove.append(i)
        
        # 反向移除以避免索引问题
        for i in reversed(associations_to_remove):
            del self.memory_associations[i]
        
        # 更新关联图
        if memory_id in self.memory_association_graph:
            self.memory_association_graph.remove_node(memory_id)
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        # 按类型统计
        type_counts = {t.value: len(ids) for t, ids in self.memory_by_type.items()}
        
        # 按巩固阶段统计
        stage_counts = defaultdict(int)
        for memory_item in self.memory_items.values():
            stage_counts[memory_item.consolidation_stage.value] += 1
        
        # 记忆项总大小
        total_size = len(self.memory_items)
        
        return {
            'total_memory_items': total_size,
            'memory_type_counts': type_counts,
            'consolidation_stage_counts': dict(stage_counts),
            'memory_associations_count': len(self.memory_associations),
            'short_term_memory_usage': len(self.short_term_memory),
            'short_term_memory_capacity': self.short_term_capacity,
            'performance_stats': self.performance_stats
        }
    
    def search_memory_by_content(self, 
                                query: str,
                                memory_type: Optional[MemoryType] = None,
                                min_relevance: float = 0.5) -> List[MemoryRetrievalResult]:
        """
        基于内容搜索记忆
        
        Args:
            query: 搜索查询
            memory_type: 限制记忆类型
            min_relevance: 最小相关性
            
        Returns:
            搜索结果
        """
        return self.retrieve_memory(
            query=query,
            memory_type=memory_type,
            retrieval_type=MemoryRetrievalType.CUED,
            max_results=20
        )
    
    def get_memory_timeline(self, 
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None) -> List[MemoryItem]:
        """
        获取时间线记忆（情景记忆）
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            时间线记忆列表
        """
        episodic_memories = []
        
        for memory_id in self.memory_by_type.get(MemoryType.EPISODIC, set()):
            if memory_id not in self.memory_items:
                continue
            
            memory_item = self.memory_items[memory_id]
            
            # 时间过滤
            if start_time and memory_item.encoding_time < start_time:
                continue
            
            if end_time and memory_item.encoding_time > end_time:
                continue
            
            episodic_memories.append(memory_item)
        
        # 按时间排序
        episodic_memories.sort(key=lambda m: m.encoding_time)
        
        return episodic_memories
    
    def clear_memory(self, memory_type: Optional[MemoryType] = None):
        """清除记忆"""
        if memory_type:
            # 清除特定类型的记忆
            memory_ids = list(self.memory_by_type.get(memory_type, set()))
            for memory_id in memory_ids:
                if memory_id in self.memory_items:
                    self._forget_memory(memory_id)
            
            logger.info(f"清除 {memory_type.value} 类型记忆: {len(memory_ids)} 个记忆项")
        else:
            # 清除所有记忆
            memory_ids = list(self.memory_items.keys())
            for memory_id in memory_ids:
                self._forget_memory(memory_id)
            
            logger.info(f"清除所有记忆: {len(memory_ids)} 个记忆项")

    def replay_memories(self, n_memories: int = 10, replay_strength: float = 0.1) -> List[str]:
        """
        重放记忆以加强巩固（模拟睡眠中的记忆重放）
        
        Args:
            n_memories: 重放记忆数量
            replay_strength: 重放强度增加
            
        Returns:
            重放的记忆ID列表
        """
        replayed_memory_ids = []
        
        # 选择近期重要记忆进行重放
        recent_memories = []
        for memory_id, memory_item in self.memory_items.items():
            if memory_item.consolidation_stage != MemoryConsolidationStage.LONG_TERM:
                recent_memories.append((memory_item.encoding_time, memory_item.importance, memory_id))
        
        # 按时间和重要性排序
        recent_memories.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        # 选择前n个记忆进行重放
        for _, _, memory_id in recent_memories[:n_memories]:
            if memory_id in self.memory_items:
                memory_item = self.memory_items[memory_id]
                
                # 增强记忆强度（降低遗忘率）
                memory_item.decay_rate = max(0.01, memory_item.decay_rate - replay_strength * 0.05)
                
                # 模拟检索（增加检索次数）
                memory_item.retrieval_count += 1
                memory_item.last_retrieval_time = time.time()
                
                # 创建巩固关联
                self._create_consolidation_associations(memory_item)
                
                replayed_memory_ids.append(memory_id)
                
                logger.debug(f"记忆重放: {memory_id}, 新遗忘率: {memory_item.decay_rate:.3f}")
        
        logger.info(f"记忆重放完成: {len(replayed_memory_ids)} 个记忆项被重放")
        return replayed_memory_ids
    
    def schedule_consolidation(self, schedule_type: str = "nightly") -> Dict[str, Any]:
        """
        安排记忆巩固计划
        
        Args:
            schedule_type: 计划类型 ("nightly", "daily", "weekly")
            
        Returns:
            计划执行结果
        """
        schedule_configs = {
            "nightly": {
                "n_memories": 50,
                "replay_strength": 0.15,
                "forgetting_aggressiveness": 0.3
            },
            "daily": {
                "n_memories": 20,
                "replay_strength": 0.1,
                "forgetting_aggressiveness": 0.2
            },
            "weekly": {
                "n_memories": 100,
                "replay_strength": 0.2,
                "forgetting_aggressiveness": 0.4
            }
        }
        
        config = schedule_configs.get(schedule_type, schedule_configs["nightly"])
        
        # 执行记忆重放
        replayed_memories = self.replay_memories(
            n_memories=config["n_memories"],
            replay_strength=config["replay_strength"]
        )
        
        # 执行主动遗忘
        forgotten_memories = self.active_forgetting(
            aggressiveness=config["forgetting_aggressiveness"]
        )
        
        # 执行标准巩固
        consolidated_memories = self.consolidate_memories()
        
        return {
            "schedule_type": schedule_type,
            "replayed_memories_count": len(replayed_memories),
            "forgotten_memories_count": len(forgotten_memories),
            "consolidated_memories_count": len(consolidated_memories),
            "config": config
        }
    
    def active_forgetting(self, aggressiveness: float = 0.3) -> List[str]:
        """
        主动遗忘机制 - 基于相关性、重要性和时间的智能遗忘
        
        Args:
            aggressiveness: 遗忘积极性 (0.0-1.0)
            
        Returns:
            被遗忘的记忆ID列表
        """
        forgotten_memory_ids = []
        current_time = time.time()
        
        # 计算每个记忆的保留价值
        memory_values = []
        for memory_id, memory_item in self.memory_items.items():
            # 基础价值：重要性 + 情感强度
            base_value = memory_item.importance * 0.6 + \
                        (abs(memory_item.emotional_valence) + memory_item.emotional_arousal) * 0.2
            
            # 检索频率价值
            retrieval_value = min(1.0, memory_item.retrieval_count * 0.1)
            
            # 时间衰减
            time_since_encoding = (current_time - memory_item.encoding_time) / (24 * 3600)  # 天数
            time_decay = math.exp(-time_since_encoding * 0.1)
            
            # 关联度价值（与其他记忆的关联数量）
            association_count = len(memory_item.associations)
            association_value = min(1.0, association_count * 0.2)
            
            # 总保留价值
            retention_value = (base_value * 0.4 + retrieval_value * 0.3 + 
                              time_decay * 0.2 + association_value * 0.1)
            
            memory_values.append((retention_value, memory_id, memory_item))
        
        # 按保留价值排序
        memory_values.sort(key=lambda x: x[0])
        
        # 根据积极性决定遗忘数量
        n_to_forget = int(len(memory_values) * aggressiveness * 0.1)  # 最多10% * 积极性
        
        # 遗忘价值最低的记忆
        for i in range(min(n_to_forget, len(memory_values))):
            retention_value, memory_id, memory_item = memory_values[i]
            
            # 长期记忆和重要记忆不易被遗忘
            if (memory_item.consolidation_stage == MemoryConsolidationStage.LONG_TERM and 
                memory_item.importance > 0.5):
                continue
            
            # 执行遗忘
            self._forget_memory(memory_id)
            forgotten_memory_ids.append(memory_id)
            
            logger.debug(f"主动遗忘: {memory_id}, 保留价值: {retention_value:.3f}")
        
        # 更新性能统计
        self.performance_stats['memory_forgettings'] += len(forgotten_memory_ids)
        
        if forgotten_memory_ids:
            logger.info(f"主动遗忘完成: {len(forgotten_memory_ids)} 个记忆项被遗忘")
        
        return forgotten_memory_ids
    
    def get_consolidation_report(self) -> Dict[str, Any]:
        """
        获取记忆巩固报告
        
        Returns:
            巩固报告
        """
        # 按阶段统计
        stage_counts = defaultdict(int)
        stage_avg_importance = defaultdict(float)
        
        for memory_item in self.memory_items.values():
            stage = memory_item.consolidation_stage
            stage_counts[stage.value] += 1
            stage_avg_importance[stage.value] += memory_item.importance
        
        # 计算平均重要性
        for stage in stage_avg_importance:
            if stage_counts[stage] > 0:
                stage_avg_importance[stage] /= stage_counts[stage]
        
        # 遗忘率分析
        decay_rates = [m.decay_rate for m in self.memory_items.values()]
        avg_decay_rate = sum(decay_rates) / len(decay_rates) if decay_rates else 0.0
        
        # 关联度分析
        avg_associations = sum(len(m.associations) for m in self.memory_items.values()) / \
                          len(self.memory_items) if self.memory_items else 0.0
        
        # 检索频率分析
        avg_retrieval_count = sum(m.retrieval_count for m in self.memory_items.values()) / \
                             len(self.memory_items) if self.memory_items else 0.0
        
        return {
            "total_memories": len(self.memory_items),
            "stage_distribution": dict(stage_counts),
            "avg_importance_by_stage": dict(stage_avg_importance),
            "avg_decay_rate": avg_decay_rate,
            "avg_associations_per_memory": avg_associations,
            "avg_retrieval_count": avg_retrieval_count,
            "short_term_memory_usage": len(self.short_term_memory),
            "short_term_memory_capacity": self.short_term_capacity,
            "consolidation_interval": self.consolidation_interval,
            "last_consolidation_time": self.last_consolidation_time,
            "performance_stats": self.performance_stats
        }
    
    def enhance_consolidation(self, memory_id: str, enhancement_type: str = "rehearsal") -> bool:
        """
        增强特定记忆的巩固
        
        Args:
            memory_id: 记忆ID
            enhancement_type: 增强类型 ("rehearsal", "emotional", "contextual")
            
        Returns:
            是否成功增强
        """
        if memory_id not in self.memory_items:
            logger.error(f"记忆增强失败: 记忆 {memory_id} 不存在")
            return False
        
        memory_item = self.memory_items[memory_id]
        
        enhancement_factors = {
            "rehearsal": 0.15,  # 复述增强
            "emotional": 0.25,  # 情感增强
            "contextual": 0.20,  # 情境增强
            "association": 0.30   # 关联增强
        }
        
        enhancement_factor = enhancement_factors.get(enhancement_type, 0.1)
        
        # 根据增强类型应用不同效果
        if enhancement_type == "rehearsal":
            # 增加检索次数
            memory_item.retrieval_count += 3
            memory_item.last_retrieval_time = time.time()
            
        elif enhancement_type == "emotional":
            # 增强情感强度
            memory_item.emotional_arousal = min(1.0, memory_item.emotional_arousal + 0.2)
            
        elif enhancement_type == "contextual":
            # 增强情境信息
            if not memory_item.context:
                memory_item.context = {"enhanced": True}
            else:
                memory_item.context["enhanced"] = True
                
        elif enhancement_type == "association":
            # 创建新关联
            self._create_consolidation_associations(memory_item)
        
        # 降低遗忘率
        memory_item.decay_rate = max(0.01, memory_item.decay_rate - enhancement_factor * 0.1)
        
        logger.info(f"记忆增强: {memory_id} ({enhancement_type}), 新遗忘率: {memory_item.decay_rate:.3f}")
        
        return True


# 示例和测试函数
def create_example_memory_system() -> EpisodicSemanticMemory:
    """创建示例记忆系统"""
    memory_system = EpisodicSemanticMemory(
        max_memory_items=1000,
        short_term_capacity=50,
        consolidation_interval=1800.0,  # 30分钟
        retrieval_threshold=0.3,
        forgetting_threshold=0.1
    )
    return memory_system


def test_memory_system():
    """测试记忆系统"""
    logger.info("开始测试综合记忆系统")
    
    # 创建示例记忆系统
    memory_system = create_example_memory_system()
    
    # 编码情景记忆
    logger.info("编码情景记忆...")
    episodic_memory = memory_system.encode_memory(
        content="今天在公园遇到老朋友，聊了很久",
        memory_type=MemoryType.EPISODIC,
        encoding_strength=MemoryEncodingStrength.STRONG,
        importance=0.8,
        emotional_valence=0.7,  # 正面情感
        emotional_arousal=0.6,  # 中度唤醒
        context={
            "location": "公园",
            "time": "下午",
            "people": "老朋友"
        },
        metadata={"weather": "晴朗"}
    )
    
    # 编码语义记忆
    logger.info("编码语义记忆...")
    semantic_memory = memory_system.encode_memory(
        content="人工智能是模拟人类智能的计算机系统",
        memory_type=MemoryType.SEMANTIC,
        encoding_strength=MemoryEncodingStrength.MODERATE,
        importance=0.9,
        context={"domain": "计算机科学", "concept": "人工智能"}
    )
    
    # 编码程序记忆
    logger.info("编码程序记忆...")
    procedural_memory = memory_system.encode_memory(
        content="骑自行车的步骤：平衡、蹬踏、转向",
        memory_type=MemoryType.PROCEDURAL,
        encoding_strength=MemoryEncodingStrength.STRONG,
        importance=0.7,
        context={"skill": "运动", "difficulty": "中等"}
    )
    
    # 创建记忆关联
    logger.info("创建记忆关联...")
    association = memory_system.create_association(
        source_id=episodic_memory.id,
        target_id=semantic_memory.id,
        association_type="contextual",
        strength=0.7
    )
    
    # 检索记忆
    logger.info("检索记忆...")
    retrieval_results = memory_system.retrieve_memory(
        query="人工智能",
        memory_type=MemoryType.SEMANTIC,
        retrieval_type=MemoryRetrievalType.RECALL
    )
    
    if retrieval_results:
        logger.info(f"检索到 {len(retrieval_results)} 个相关记忆")
        for result in retrieval_results[:3]:
            logger.info(f"  记忆: {result.memory_item.content[:50]}...，置信度: {result.confidence:.2f}")
    
    # 巩固记忆
    logger.info("巩固记忆...")
    consolidation_results = memory_system.consolidate_memories()
    logger.info(f"巩固了 {len(consolidation_results)} 个记忆")
    
    # 获取统计信息
    stats = memory_system.get_memory_statistics()
    logger.info(f"记忆统计: {stats['total_memory_items']} 个记忆项，{stats['memory_associations_count']} 个关联")
    
    # 获取时间线记忆
    timeline = memory_system.get_memory_timeline()
    logger.info(f"时间线记忆: {len(timeline)} 个情景记忆")
    
    logger.info("综合记忆系统测试完成")
    return memory_system


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_memory_system_instance = test_memory_system()