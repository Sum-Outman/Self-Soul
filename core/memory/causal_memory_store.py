#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因果记忆存储 - 实现语义记忆的因果关系和概念存储

核心功能:
1. 概念表示和存储: 存储概念的定义、属性和关系
2. 因果关系存储: 存储概念之间的因果关系
3. 因果推理: 基于存储的因果关系进行推理
4. 概念层次构建: 构建概念的层次结构
5. 因果证据管理: 管理因果关系的证据和置信度

因果记忆特点:
- 概念中心: 以概念为基本存储单元
- 关系网络: 概念通过关系连接成网络
- 因果方向: 因果关系具有方向性
- 置信度管理: 每个关系都有置信度
- 证据支持: 关系有证据支持

关系类型:
1. 因果关系 (causes): A 导致 B
2. 相关关系 (correlated_with): A 与 B 相关
3. 包含关系 (contains): A 包含 B
4. 继承关系 (is_a): A 是 B 的一种
5. 属性关系 (has_property): A 具有属性 B
6. 部分关系 (part_of): A 是 B 的一部分

因果推理类型:
1. 正向推理: 从原因推理结果
2. 反向推理: 从结果推理原因
3. 中介分析: 分析因果路径中的中介变量
4. 混杂控制: 控制潜在的混杂变量
5. 反事实推理: 如果...会怎样

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """关系类型枚举"""
    CAUSES = "causes"                    # 因果关系: A 导致 B
    CORRELATED_WITH = "correlated_with"  # 相关关系: A 与 B 相关
    CONTAINS = "contains"                # 包含关系: A 包含 B
    IS_A = "is_a"                        # 继承关系: A 是 B 的一种
    HAS_PROPERTY = "has_property"        # 属性关系: A 具有属性 B
    PART_OF = "part_of"                  # 部分关系: A 是 B 的一部分
    PRECEDES = "precedes"                # 时序关系: A 在 B 之前发生
    IMPLIES = "implies"                  # 逻辑关系: A 蕴含 B


class CausalStrength(Enum):
    """因果强度枚举"""
    WEAK = "weak"        # 弱因果关系
    MODERATE = "moderate"  # 中等因果关系
    STRONG = "strong"    # 强因果关系


class InferenceDirection(Enum):
    """推理方向枚举"""
    FORWARD = "forward"      # 正向推理: 从原因到结果
    BACKWARD = "backward"    # 反向推理: 从结果到原因
    BIDIRECTIONAL = "bidirectional"  # 双向推理


@dataclass
class Concept:
    """概念数据类"""
    id: str
    name: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    examples: List[Any] = field(default_factory=list)
    importance: float = 0.5
    abstraction_level: int = 0  # 抽象级别: 0=具体, 越高越抽象
    created_time: float = field(default_factory=time.time)
    last_accessed_time: Optional[float] = None
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.importance = max(0.0, min(1.0, self.importance))
        self.abstraction_level = max(0, self.abstraction_level)
        if self.last_accessed_time is None:
            self.last_accessed_time = self.created_time
    
    def update_access(self):
        """更新访问信息"""
        self.access_count += 1
        self.last_accessed_time = time.time()


@dataclass
class CausalRelation:
    """因果关系数据类"""
    id: str
    source_concept_id: str
    target_concept_id: str
    relation_type: RelationType
    strength: float = 0.5
    confidence: float = 0.8
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    discovered_time: float = field(default_factory=time.time)
    last_verified_time: Optional[float] = None
    verification_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))
        if self.last_verified_time is None:
            self.last_verified_time = self.discovered_time
    
    def add_evidence(self, evidence_data: Dict[str, Any]):
        """添加证据"""
        self.evidence.append(evidence_data)
        self.verification_count += 1
        self.last_verified_time = time.time()
        
        # 更新置信度（基于证据数量）
        self.confidence = min(1.0, 0.5 + 0.5 * (self.verification_count / (self.verification_count + 1)))
    
    @property
    def causal_strength_level(self) -> CausalStrength:
        """获取因果强度级别"""
        if self.strength < 0.3:
            return CausalStrength.WEAK
        elif self.strength < 0.7:
            return CausalStrength.MODERATE
        else:
            return CausalStrength.STRONG


@dataclass
class InferenceResult:
    """推理结果数据类"""
    query_concept_id: str
    direction: InferenceDirection
    inferred_concepts: List[Tuple[str, float]]  # (概念ID, 置信度)
    inference_paths: List[List[str]]  # 推理路径
    inference_time: float = field(default_factory=time.time)
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.confidence = max(0.0, min(1.0, self.confidence))


class CausalMemoryStore:
    """
    因果记忆存储 - 管理概念和因果关系
    
    核心组件:
    1. 概念管理器: 概念的增删改查
    2. 关系管理器: 因果关系的管理
    3. 推理引擎: 基于因果关系的推理
    4. 证据管理器: 管理因果关系的证据
    5. 概念层次构建器: 构建概念层次结构
    
    工作流程:
    概念数据 → 概念管理器 → 存储概念 → 关系管理器 → 建立关系
    推理请求 → 推理引擎 → 使用关系网络 → 执行推理 → 返回结果
    新证据 → 证据管理器 → 更新关系置信度 → 优化推理
    
    技术特性:
    - 概念层次结构和继承
    - 因果网络构建和查询
    - 置信度传播推理
    - 证据权重管理
    - 概念相似性计算
    """
    
    def __init__(self,
                 max_concepts: int = 10000,
                 max_relations: int = 50000,
                 inference_depth_limit: int = 5):
        """
        初始化因果记忆存储
        
        Args:
            max_concepts: 最大概念数量
            max_relations: 最大关系数量
            inference_depth_limit: 推理深度限制
        """
        self.max_concepts = max_concepts
        self.max_relations = max_relations
        self.inference_depth_limit = inference_depth_limit
        
        # 概念存储
        self.concepts: Dict[str, Concept] = {}
        
        # 关系存储
        self.relations: Dict[str, CausalRelation] = {}
        
        # 索引结构
        self.concept_name_index: Dict[str, str] = {}  # 名称到ID的映射
        self.relations_by_concept: Dict[str, Dict[RelationType, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.concept_graph = nx.MultiDiGraph()
        
        # 配置参数
        self.config = {
            'min_relation_confidence': 0.3,
            'max_inference_paths': 10,
            'similarity_threshold': 0.7,
            'inheritance_inference_enabled': True,
            'transitive_closure_enabled': True,
            'evidence_decay_factor': 0.99,
            'concept_importance_decay': 0.95
        }
        
        # 性能统计
        self.performance_stats = {
            'concepts_stored': 0,
            'relations_stored': 0,
            'inferences_performed': 0,
            'evidence_added': 0,
            'concept_accesses': 0,
            'average_inference_time': 0.0,
            'inference_success_rate': 0.0
        }
        
        # 初始化基础概念
        self._initialize_basic_concepts()
        
        logger.info(f"因果记忆存储初始化完成，最大概念: {max_concepts}，最大关系: {max_relations}")
    
    def _initialize_basic_concepts(self):
        """初始化基础概念"""
        # 添加一些基础概念
        basic_concepts = [
            ("entity", "基本实体", 0.9, 0),
            ("object", "物理对象", 0.8, 1),
            ("event", "事件", 0.8, 1),
            ("process", "过程", 0.7, 1),
            ("property", "属性", 0.6, 1),
            ("relation", "关系", 0.6, 1),
            ("cause", "原因", 0.7, 2),
            ("effect", "结果", 0.7, 2),
            ("time", "时间", 0.9, 0),
            ("space", "空间", 0.9, 0)
        ]
        
        for name, description, importance, abstraction_level in basic_concepts:
            self.add_concept(name, description, importance=importance, 
                           abstraction_level=abstraction_level)
        
        # 添加基础关系
        self.add_relation("object", "entity", RelationType.IS_A, strength=0.9)
        self.add_relation("event", "entity", RelationType.IS_A, strength=0.9)
        self.add_relation("process", "entity", RelationType.IS_A, strength=0.8)
        self.add_relation("property", "entity", RelationType.IS_A, strength=0.8)
        self.add_relation("relation", "entity", RelationType.IS_A, strength=0.8)
        
        logger.info(f"初始化了 {len(basic_concepts)} 个基础概念")
    
    def add_concept(self,
                   name: str,
                   description: str = "",
                   properties: Optional[Dict[str, Any]] = None,
                   examples: Optional[List[Any]] = None,
                   importance: float = 0.5,
                   abstraction_level: int = 0,
                   metadata: Optional[Dict[str, Any]] = None) -> Concept:
        """
        添加概念
        
        Args:
            name: 概念名称
            description: 概念描述
            properties: 概念属性
            examples: 概念示例
            importance: 重要性 (0.0-1.0)
            abstraction_level: 抽象级别
            metadata: 元数据
            
        Returns:
            添加的概念
        """
        # 检查概念是否已存在
        if name in self.concept_name_index:
            concept_id = self.concept_name_index[name]
            concept = self.concepts[concept_id]
            
            # 更新现有概念
            concept.description = description or concept.description
            concept.importance = max(concept.importance, importance)
            concept.abstraction_level = max(concept.abstraction_level, abstraction_level)
            
            if properties:
                concept.properties.update(properties)
            
            if examples:
                concept.examples.extend(examples)
            
            if metadata:
                concept.metadata.update(metadata)
            
            logger.info(f"概念更新: {name} (ID: {concept_id})")
            
            return concept
        
        # 生成概念ID
        concept_id = f"concept_{len(self.concepts)}"
        
        # 创建概念
        concept = Concept(
            id=concept_id,
            name=name,
            description=description,
            properties=properties or {},
            examples=examples or [],
            importance=importance,
            abstraction_level=abstraction_level,
            metadata=metadata or {}
        )
        
        # 存储概念
        self.concepts[concept_id] = concept
        self.concept_name_index[name] = concept_id
        
        # 添加到概念图
        self.concept_graph.add_node(concept_id, 
                                   name=name,
                                   importance=importance,
                                   abstraction_level=abstraction_level)
        
        # 检查存储容量
        self._check_concept_capacity()
        
        # 更新统计
        self.performance_stats['concepts_stored'] += 1
        
        logger.info(f"概念添加: {name} (ID: {concept_id})，抽象级别: {abstraction_level}")
        
        return concept
    
    def _check_concept_capacity(self):
        """检查概念容量"""
        if len(self.concepts) > self.max_concepts:
            # 删除最不重要的概念
            self._remove_least_important_concepts(len(self.concepts) - self.max_concepts)
    
    def _remove_least_important_concepts(self, count: int):
        """删除最不重要的概念"""
        if not self.concepts:
            return
        
        # 按重要性排序
        concepts_by_importance = sorted(self.concepts.values(), 
                                       key=lambda c: (c.importance, -c.access_count))
        
        for concept in concepts_by_importance[:count]:
            self._remove_concept(concept.id)
        
        logger.info(f"删除 {count} 个最不重要的概念")
    
    def _remove_concept(self, concept_id: str):
        """移除概念"""
        if concept_id not in self.concepts:
            return
        
        concept = self.concepts[concept_id]
        
        # 从名称索引中移除
        if concept.name in self.concept_name_index:
            del self.concept_name_index[concept.name]
        
        # 移除相关关系
        self._remove_concept_relations(concept_id)
        
        # 从概念图中移除
        if concept_id in self.concept_graph:
            self.concept_graph.remove_node(concept_id)
        
        # 从索引中移除
        if concept_id in self.relations_by_concept:
            del self.relations_by_concept[concept_id]
        
        # 从存储中删除
        del self.concepts[concept_id]
        
        logger.info(f"概念移除: {concept.name} (ID: {concept_id})")
    
    def _remove_concept_relations(self, concept_id: str):
        """移除概念的相关关系"""
        relations_to_remove = []
        
        for relation_id, relation in list(self.relations.items()):
            if (relation.source_concept_id == concept_id or 
                relation.target_concept_id == concept_id):
                relations_to_remove.append(relation_id)
        
        for relation_id in relations_to_remove:
            self._remove_relation(relation_id)
    
    def add_relation(self,
                    source_concept_name: str,
                    target_concept_name: str,
                    relation_type: RelationType,
                    strength: float = 0.5,
                    confidence: float = 0.8,
                    evidence: Optional[List[Dict[str, Any]]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Optional[CausalRelation]:
        """
        添加关系
        
        Args:
            source_concept_name: 源概念名称
            target_concept_name: 目标概念名称
            relation_type: 关系类型
            strength: 关系强度 (0.0-1.0)
            confidence: 置信度 (0.0-1.0)
            evidence: 证据列表
            metadata: 元数据
            
        Returns:
            添加的关系，如果失败则返回None
        """
        # 获取概念ID
        source_id = self.concept_name_index.get(source_concept_name)
        target_id = self.concept_name_index.get(target_concept_name)
        
        if not source_id or not target_id:
            logger.error(f"关系添加失败: 概念不存在 ({source_concept_name}, {target_concept_name})")
            return None
        
        # 检查关系是否已存在
        existing_relation = self._find_existing_relation(source_id, target_id, relation_type)
        if existing_relation:
            # 更新现有关系
            existing_relation.strength = max(existing_relation.strength, strength)
            existing_relation.confidence = max(existing_relation.confidence, confidence)
            
            if evidence:
                for ev in evidence:
                    existing_relation.add_evidence(ev)
            
            if metadata:
                existing_relation.metadata.update(metadata)
            
            logger.info(f"关系更新: {source_concept_name} {relation_type.value} {target_concept_name}")
            
            return existing_relation
        
        # 生成关系ID
        relation_id = f"relation_{len(self.relations)}"
        
        # 创建关系
        relation = CausalRelation(
            id=relation_id,
            source_concept_id=source_id,
            target_concept_id=target_id,
            relation_type=relation_type,
            strength=strength,
            confidence=confidence,
            evidence=evidence or [],
            metadata=metadata or {}
        )
        
        # 存储关系
        self.relations[relation_id] = relation
        
        # 更新索引
        self.relations_by_concept[source_id][relation_type].append(relation_id)
        
        # 添加到概念图
        self.concept_graph.add_edge(
            source_id, target_id,
            relation_id=relation_id,
            relation_type=relation_type.value,
            strength=strength,
            confidence=confidence
        )
        
        # 检查存储容量
        self._check_relation_capacity()
        
        # 更新统计
        self.performance_stats['relations_stored'] += 1
        
        # 如果是继承关系，更新抽象级别
        if relation_type == RelationType.IS_A:
            self._update_inheritance_abstraction(source_id, target_id)
        
        logger.info(f"关系添加: {source_concept_name} {relation_type.value} {target_concept_name}，强度: {strength:.2f}")
        
        return relation
    
    def _find_existing_relation(self, 
                               source_id: str, 
                               target_id: str, 
                               relation_type: RelationType) -> Optional[CausalRelation]:
        """查找现有关系"""
        for relation in self.relations.values():
            if (relation.source_concept_id == source_id and 
                relation.target_concept_id == target_id and 
                relation.relation_type == relation_type):
                return relation
        
        return None
    
    def _update_inheritance_abstraction(self, child_id: str, parent_id: str):
        """更新继承关系的抽象级别"""
        if child_id in self.concepts and parent_id in self.concepts:
            child = self.concepts[child_id]
            parent = self.concepts[parent_id]
            
            # 子概念的抽象级别应该比父概念低（更具体）
            if child.abstraction_level <= parent.abstraction_level:
                child.abstraction_level = parent.abstraction_level + 1
    
    def _check_relation_capacity(self):
        """检查关系容量"""
        if len(self.relations) > self.max_relations:
            # 删除置信度最低的关系
            self._remove_lowest_confidence_relations(len(self.relations) - self.max_relations)
    
    def _remove_lowest_confidence_relations(self, count: int):
        """删除置信度最低的关系"""
        if not self.relations:
            return
        
        # 按置信度排序
        relations_by_confidence = sorted(self.relations.values(), 
                                        key=lambda r: r.confidence)
        
        for relation in relations_by_confidence[:count]:
            self._remove_relation(relation.id)
        
        logger.info(f"删除 {count} 个置信度最低的关系")
    
    def _remove_relation(self, relation_id: str):
        """移除关系"""
        if relation_id not in self.relations:
            return
        
        relation = self.relations[relation_id]
        
        # 从索引中移除
        source_relations = self.relations_by_concept.get(relation.source_concept_id)
        if source_relations and relation.relation_type in source_relations:
            relation_list = source_relations[relation.relation_type]
            if relation_id in relation_list:
                relation_list.remove(relation_id)
        
        # 从概念图中移除边
        edges_to_remove = []
        for u, v, key, data in self.concept_graph.edges(keys=True, data=True):
            if data.get('relation_id') == relation_id:
                edges_to_remove.append((u, v, key))
        
        for u, v, key in edges_to_remove:
            self.concept_graph.remove_edge(u, v, key)
        
        # 从存储中删除
        del self.relations[relation_id]
        
        logger.info(f"关系移除: {relation_id}")
    
    def infer_related_concepts(self,
                              concept_name: str,
                              direction: InferenceDirection = InferenceDirection.FORWARD,
                              relation_type: Optional[RelationType] = None,
                              max_results: int = 10,
                              min_confidence: float = 0.3) -> InferenceResult:
        """
        推理相关概念
        
        Args:
            concept_name: 概念名称
            direction: 推理方向
            relation_type: 限制关系类型
            max_results: 最大结果数量
            min_confidence: 最小置信度
            
        Returns:
            推理结果
        """
        start_time = time.time()
        
        # 获取概念ID
        concept_id = self.concept_name_index.get(concept_name)
        if not concept_id or concept_id not in self.concepts:
            # 概念不存在，返回空结果
            return InferenceResult(
                query_concept_id=concept_name,
                direction=direction,
                inferred_concepts=[],
                inference_paths=[],
                confidence=0.0
            )
        
        # 更新概念访问
        self.concepts[concept_id].update_access()
        self.performance_stats['concept_accesses'] += 1
        
        # 执行推理
        inferred_concepts = []
        inference_paths = []
        
        if direction in [InferenceDirection.FORWARD, InferenceDirection.BIDIRECTIONAL]:
            # 正向推理：从概念出发
            forward_results = self._forward_inference(concept_id, relation_type, 
                                                     max_results, min_confidence)
            inferred_concepts.extend(forward_results['concepts'])
            inference_paths.extend(forward_results['paths'])
        
        if direction in [InferenceDirection.BACKWARD, InferenceDirection.BIDIRECTIONAL]:
            # 反向推理：到达概念
            backward_results = self._backward_inference(concept_id, relation_type,
                                                       max_results, min_confidence)
            inferred_concepts.extend(backward_results['concepts'])
            inference_paths.extend(backward_results['paths'])
        
        # 去重和排序
        inferred_concepts = self._deduplicate_and_sort(inferred_concepts, max_results)
        
        # 计算整体置信度
        overall_confidence = self._compute_overall_confidence(inferred_concepts)
        
        # 创建推理结果
        result = InferenceResult(
            query_concept_id=concept_id,
            direction=direction,
            inferred_concepts=inferred_concepts,
            inference_paths=inference_paths[:max_results],
            confidence=overall_confidence
        )
        
        # 更新性能统计
        inference_time = time.time() - start_time
        self._update_inference_stats(inference_time, len(inferred_concepts) > 0)
        
        logger.info(f"概念推理完成: {concept_name} -> {len(inferred_concepts)} 个相关概念")
        
        return result
    
    def _forward_inference(self,
                          concept_id: str,
                          relation_type: Optional[RelationType],
                          max_results: int,
                          min_confidence: float) -> Dict[str, Any]:
        """正向推理"""
        inferred_concepts = []
        inference_paths = []
        
        # 获取直接关系
        direct_results = self._get_direct_relations(concept_id, relation_type, 
                                                   outgoing=True, 
                                                   min_confidence=min_confidence)
        
        for target_id, relation_id, rel_type, confidence in direct_results:
            if target_id in self.concepts:
                inferred_concepts.append((target_id, confidence))
                inference_paths.append([concept_id, target_id])
        
        # 获取间接关系（多跳推理）
        if len(inferred_concepts) < max_results:
            indirect_results = self._get_indirect_relations(concept_id, relation_type,
                                                           outgoing=True,
                                                           max_depth=self.inference_depth_limit,
                                                           min_confidence=min_confidence)
            
            for path, confidence in indirect_results:
                if len(path) > 1:
                    target_id = path[-1]
                    if target_id in self.concepts:
                        inferred_concepts.append((target_id, confidence))
                        inference_paths.append(path)
        
        return {
            'concepts': inferred_concepts,
            'paths': inference_paths
        }
    
    def _backward_inference(self,
                           concept_id: str,
                           relation_type: Optional[RelationType],
                           max_results: int,
                           min_confidence: float) -> Dict[str, Any]:
        """反向推理"""
        inferred_concepts = []
        inference_paths = []
        
        # 获取直接关系（入边）
        direct_results = self._get_direct_relations(concept_id, relation_type,
                                                   outgoing=False,
                                                   min_confidence=min_confidence)
        
        for source_id, relation_id, rel_type, confidence in direct_results:
            if source_id in self.concepts:
                inferred_concepts.append((source_id, confidence))
                inference_paths.append([source_id, concept_id])
        
        # 获取间接关系
        if len(inferred_concepts) < max_results:
            indirect_results = self._get_indirect_relations(concept_id, relation_type,
                                                           outgoing=False,
                                                           max_depth=self.inference_depth_limit,
                                                           min_confidence=min_confidence)
            
            for path, confidence in indirect_results:
                if len(path) > 1:
                    source_id = path[0]
                    if source_id in self.concepts:
                        inferred_concepts.append((source_id, confidence))
                        inference_paths.append(path)
        
        return {
            'concepts': inferred_concepts,
            'paths': inference_paths
        }
    
    def _get_direct_relations(self,
                            concept_id: str,
                            relation_type: Optional[RelationType],
                            outgoing: bool,
                            min_confidence: float) -> List[Tuple[str, str, RelationType, float]]:
        """获取直接关系"""
        results = []
        
        if outgoing:
            # 出边：概念作为源
            edges = self.concept_graph.out_edges(concept_id, data=True)
        else:
            # 入边：概念作为目标
            edges = self.concept_graph.in_edges(concept_id, data=True)
        
        for u, v, data in edges:
            # 过滤关系类型
            if relation_type and data.get('relation_type') != relation_type.value:
                continue
            
            # 过滤置信度
            confidence = data.get('confidence', 0.0)
            if confidence < min_confidence:
                continue
            
            # 获取关系类型
            rel_type = RelationType(data.get('relation_type', 'causes'))
            
            # 获取相关概念ID
            related_concept_id = v if outgoing else u
            
            results.append((related_concept_id, data.get('relation_id', ''), 
                           rel_type, confidence))
        
        return results
    
    def _get_indirect_relations(self,
                               concept_id: str,
                               relation_type: Optional[RelationType],
                               outgoing: bool,
                               max_depth: int,
                               min_confidence: float) -> List[Tuple[List[str], float]]:
        """获取间接关系（多跳）"""
        if max_depth <= 1:
            return []
        
        results = []
        
        # 使用BFS或DFS查找路径
        visited = set([concept_id])
        queue = deque([(concept_id, [concept_id], 1.0)])  # (当前节点, 路径, 路径置信度)
        
        while queue and len(results) < 10:  # 限制结果数量
            current_node, path, path_confidence = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            # 获取当前节点的直接关系
            if outgoing:
                edges = self.concept_graph.out_edges(current_node, data=True)
            else:
                edges = self.concept_graph.in_edges(current_node, data=True)
            
            for u, v, data in edges:
                # 过滤关系类型
                if relation_type and data.get('relation_type') != relation_type.value:
                    continue
                
                # 获取边置信度
                edge_confidence = data.get('confidence', 0.0)
                if edge_confidence < min_confidence:
                    continue
                
                # 计算路径置信度
                new_path_confidence = path_confidence * edge_confidence
                
                # 获取下一个节点
                next_node = v if outgoing else u
                
                # 避免循环
                if next_node in visited:
                    continue
                
                # 创建新路径
                new_path = path + [next_node]
                
                # 添加到结果
                if len(new_path) > 2:  # 至少是两跳路径
                    results.append((new_path, new_path_confidence))
                
                # 继续探索
                visited.add(next_node)
                queue.append((next_node, new_path, new_path_confidence))
        
        return results
    
    def _deduplicate_and_sort(self, 
                            inferred_concepts: List[Tuple[str, float]], 
                            max_results: int) -> List[Tuple[str, float]]:
        """去重和排序"""
        # 去重
        concept_dict = {}
        for concept_id, confidence in inferred_concepts:
            if concept_id in concept_dict:
                concept_dict[concept_id] = max(concept_dict[concept_id], confidence)
            else:
                concept_dict[concept_id] = confidence
        
        # 排序
        sorted_concepts = sorted(concept_dict.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_concepts[:max_results]
    
    def _compute_overall_confidence(self, inferred_concepts: List[Tuple[str, float]]) -> float:
        """计算整体置信度"""
        if not inferred_concepts:
            return 0.0
        
        confidences = [confidence for _, confidence in inferred_concepts]
        return np.mean(confidences)
    
    def _update_inference_stats(self, inference_time: float, success: bool):
        """更新推理统计"""
        self.performance_stats['inferences_performed'] += 1
        
        # 更新平均推理时间
        current_avg = self.performance_stats['average_inference_time']
        n_inferences = self.performance_stats['inferences_performed']
        
        new_avg = (current_avg * (n_inferences - 1) + inference_time) / n_inferences
        self.performance_stats['average_inference_time'] = new_avg
        
        # 更新成功率
        current_success_rate = self.performance_stats['inference_success_rate']
        new_success_rate = (current_success_rate * (n_inferences - 1) + (1.0 if success else 0.0)) / n_inferences
        self.performance_stats['inference_success_rate'] = new_success_rate
    
    def add_evidence_to_relation(self,
                                relation_id: str,
                                evidence_data: Dict[str, Any]) -> bool:
        """
        为关系添加证据
        
        Args:
            relation_id: 关系ID
            evidence_data: 证据数据
            
        Returns:
            是否成功添加
        """
        if relation_id not in self.relations:
            logger.error(f"证据添加失败: 关系不存在 ({relation_id})")
            return False
        
        relation = self.relations[relation_id]
        relation.add_evidence(evidence_data)
        
        # 更新概念图中的置信度
        self._update_relation_in_graph(relation)
        
        self.performance_stats['evidence_added'] += 1
        
        logger.info(f"证据添加: 关系 {relation_id}，新证据数量: {len(relation.evidence)}")
        
        return True
    
    def _update_relation_in_graph(self, relation: CausalRelation):
        """更新概念图中的关系"""
        # 查找并更新边
        for u, v, key, data in self.concept_graph.edges(keys=True, data=True):
            if data.get('relation_id') == relation.id:
                self.concept_graph[u][v][key]['confidence'] = relation.confidence
                self.concept_graph[u][v][key]['strength'] = relation.strength
                break
    
    def find_concept_by_name(self, name: str) -> Optional[Concept]:
        """根据名称查找概念"""
        concept_id = self.concept_name_index.get(name)
        if concept_id and concept_id in self.concepts:
            concept = self.concepts[concept_id]
            concept.update_access()
            self.performance_stats['concept_accesses'] += 1
            return concept
        
        return None
    
    def get_concept_hierarchy(self, concept_name: str) -> Dict[str, Any]:
        """获取概念层次结构"""
        concept_id = self.concept_name_index.get(concept_name)
        if not concept_id or concept_id not in self.concepts:
            return {}
        
        hierarchy = {
            'concept': self.concepts[concept_id].name,
            'parents': [],
            'children': [],
            'siblings': []
        }
        
        # 获取父概念（IS_A关系的源）
        parent_relations = self._get_relations_for_concept(concept_id, RelationType.IS_A, outgoing=False)
        hierarchy['parents'] = [self.concepts[rel.source_concept_id].name 
                               for rel in parent_relations]
        
        # 获取子概念（IS_A关系的目标）
        child_relations = self._get_relations_for_concept(concept_id, RelationType.IS_A, outgoing=True)
        hierarchy['children'] = [self.concepts[rel.target_concept_id].name 
                                for rel in child_relations]
        
        # 获取兄弟概念（共享父概念）
        sibling_ids = set()
        for parent_name in hierarchy['parents']:
            parent_id = self.concept_name_index.get(parent_name)
            if parent_id:
                parent_children = self._get_relations_for_concept(parent_id, RelationType.IS_A, outgoing=True)
                sibling_ids.update([rel.target_concept_id for rel in parent_children])
        
        # 移除自身
        sibling_ids.discard(concept_id)
        hierarchy['siblings'] = [self.concepts[sib_id].name for sib_id in sibling_ids 
                                if sib_id in self.concepts]
        
        return hierarchy
    
    def _get_relations_for_concept(self,
                                  concept_id: str,
                                  relation_type: RelationType,
                                  outgoing: bool) -> List[CausalRelation]:
        """获取概念的关系"""
        relations = []
        
        for relation in self.relations.values():
            if relation.relation_type != relation_type:
                continue
            
            if outgoing and relation.source_concept_id == concept_id:
                relations.append(relation)
            elif not outgoing and relation.target_concept_id == concept_id:
                relations.append(relation)
        
        return relations
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 概念统计
        concept_counts_by_level = defaultdict(int)
        for concept in self.concepts.values():
            concept_counts_by_level[concept.abstraction_level] += 1
        
        # 关系统计
        relation_counts_by_type = defaultdict(int)
        for relation in self.relations.values():
            relation_counts_by_type[relation.relation_type.value] += 1
        
        return {
            'total_concepts': len(self.concepts),
            'concept_counts_by_abstraction_level': dict(concept_counts_by_level),
            'total_relations': len(self.relations),
            'relation_counts_by_type': dict(relation_counts_by_type),
            'graph_nodes': self.concept_graph.number_of_nodes(),
            'graph_edges': self.concept_graph.number_of_edges(),
            'performance_stats': self.performance_stats
        }


# 示例和测试函数
def create_example_causal_store() -> CausalMemoryStore:
    """创建示例因果记忆存储"""
    store = CausalMemoryStore(
        max_concepts=1000,
        max_relations=5000,
        inference_depth_limit=3
    )
    return store


def test_causal_memory_store():
    """测试因果记忆存储"""
    logger.info("开始测试因果记忆存储")
    
    # 创建示例存储
    store = create_example_causal_store()
    
    # 添加概念
    logger.info("添加概念...")
    
    # 添加一些科学概念
    store.add_concept("gravity", "引力", 
                     properties={"type": "fundamental force", "strength": "weak"},
                     importance=0.9, abstraction_level=2)
    
    store.add_concept("mass", "质量",
                     description="物体的惯性量度",
                     importance=0.8, abstraction_level=2)
    
    store.add_concept("acceleration", "加速度",
                     description="速度的变化率",
                     importance=0.7, abstraction_level=2)
    
    store.add_concept("force", "力",
                     description="改变物体运动状态的原因",
                     importance=0.8, abstraction_level=2)
    
    store.add_concept("newton", "牛顿",
                     description="物理学单位",
                     importance=0.6, abstraction_level=3)
    
    # 添加因果关系
    logger.info("添加因果关系...")
    
    store.add_relation("mass", "gravity", RelationType.CAUSES, 
                      strength=0.8, confidence=0.9)
    
    store.add_relation("gravity", "force", RelationType.CAUSES,
                      strength=0.9, confidence=0.95)
    
    store.add_relation("force", "acceleration", RelationType.CAUSES,
                      strength=0.85, confidence=0.9)
    
    # 添加继承关系
    store.add_relation("gravity", "force", RelationType.IS_A,
                      strength=0.7, confidence=0.8)
    
    # 推理测试
    logger.info("推理测试...")
    
    # 正向推理：从质量推理
    forward_result = store.infer_related_concepts(
        concept_name="mass",
        direction=InferenceDirection.FORWARD,
        relation_type=RelationType.CAUSES,
        max_results=5
    )
    
    logger.info(f"从 'mass' 正向推理到 {len(forward_result.inferred_concepts)} 个概念")
    for concept_id, confidence in forward_result.inferred_concepts[:3]:
        concept = store.concepts.get(concept_id)
        if concept:
            logger.info(f"  概念: {concept.name}，置信度: {confidence:.2f}")
    
    # 反向推理：从加速度推理
    backward_result = store.infer_related_concepts(
        concept_name="acceleration",
        direction=InferenceDirection.BACKWARD,
        relation_type=RelationType.CAUSES,
        max_results=5
    )
    
    logger.info(f"从 'acceleration' 反向推理到 {len(backward_result.inferred_concepts)} 个概念")
    for concept_id, confidence in backward_result.inferred_concepts[:3]:
        concept = store.concepts.get(concept_id)
        if concept:
            logger.info(f"  概念: {concept.name}，置信度: {confidence:.2f}")
    
    # 查找概念层次
    logger.info("查找概念层次...")
    hierarchy = store.get_concept_hierarchy("gravity")
    logger.info(f"'gravity' 的层次结构: 父概念: {hierarchy.get('parents', [])}，子概念: {hierarchy.get('children', [])}")
    
    # 获取统计信息
    stats = store.get_statistics()
    logger.info(f"统计信息: {stats['total_concepts']} 个概念，{stats['total_relations']} 个关系")
    
    logger.info("因果记忆存储测试完成")
    return store


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_causal_store_instance = test_causal_memory_store()