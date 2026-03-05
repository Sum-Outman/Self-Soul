#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因果知识图谱 - 实现因果关系的结构化存储和查询

核心功能:
1. 因果关系的图结构存储
2. 因果强度和证据管理
3. 因果路径查询和推理
4. 潜在混杂变量检测
5. 因果知识验证和更新

图谱特性:
- 有向边表示因果关系方向
- 边属性包含因果强度、证据类型、置信度
- 支持时间戳和版本管理
- 支持因果路径的发现和验证

技术实现:
- 基于networkx的有向图
- 因果证据的标准化表示
- 因果查询语言接口
- 增量式知识更新

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import networkx as nx
from datetime import datetime

# 导入错误处理
from core.error_handling import ErrorHandler

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()


class EvidenceType(Enum):
    """证据类型枚举"""
    RANDOMIZED_TRIAL = "randomized_trial"      # 随机试验
    OBSERVATIONAL_STUDY = "observational_study"  # 观察性研究
    EXPERT_KNOWLEDGE = "expert_knowledge"      # 专家知识
    MECHANISTIC_REASONING = "mechanistic_reasoning"  # 机制推理
    COMPUTATIONAL_MODEL = "computational_model"  # 计算模型
    META_ANALYSIS = "meta_analysis"            # 元分析


class CausalStrength(Enum):
    """因果强度等级枚举"""
    STRONG = "strong"          # 强因果: p < 0.01, effect size > 0.8
    MODERATE = "moderate"      # 中等因果: p < 0.05, effect size > 0.5
    WEAK = "weak"              # 弱因果: p < 0.1, effect size > 0.2
    VERY_WEAK = "very_weak"    # 很弱: p < 0.2, effect size > 0.1
    UNCERTAIN = "uncertain"    # 不确定: 证据不足


class CausalKnowledgeGraph:
    """
    因果知识图谱 - 因果关系的有向图表示
    
    核心特性:
    1. 节点: 变量、概念、实体
    2. 边: 因果关系，包含方向、强度、证据
    3. 属性: 因果强度、置信度、时间戳
    4. 元数据: 证据来源、研究设计、样本大小
    
    查询功能:
    1. 因果路径发现
    2. 中介变量识别
    3. 混杂变量检测
    4. 因果效应传播
    5. 反事实查询
    
    技术实现:
    - 基于networkx的DiGraph
    - 因果证据的标准编码
    - 高效的图算法
    - 增量式更新和验证
    """
    
    def __init__(self, name: str = "Causal Knowledge Graph"):
        """
        初始化因果知识图谱
        
        Args:
            name: 图谱名称
        """
        self.name = name
        self.graph = nx.DiGraph()
        
        # 元数据
        self.metadata = {
            "creation_time": time.time(),
            "last_updated": time.time(),
            "version": "1.0.0",
            "description": "Causal knowledge graph for AGI reasoning"
        }
        
        # 索引和缓存
        self.node_index = {}           # 节点ID到节点数据的映射
        self.evidence_index = {}       # 证据ID到证据的映射
        self.query_cache = {}          # 查询缓存
        
        # 统计信息
        self.stats = {
            "nodes": 0,
            "edges": 0,
            "evidence_count": 0,
            "queries_processed": 0,
            "updates_applied": 0
        }
        
        # 配置
        self.min_confidence_threshold = 0.3
        self.max_path_length = 10
        
        logger.info(f"因果知识图谱初始化完成: {name}")
    
    def add_node(self,
                 node_id: str,
                 node_type: str = "variable",
                 properties: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        添加节点到图谱
        
        Args:
            node_id: 节点ID
            node_type: 节点类型 ("variable", "concept", "entity", "event")
            properties: 节点属性
            metadata: 节点元数据
            
        Returns:
            是否成功添加
        """
        try:
            if node_id in self.graph.nodes():
                logger.warning(f"节点已存在: {node_id}")
                return False
            
            # 创建节点数据
            node_data = {
                "id": node_id,
                "type": node_type,
                "properties": properties or {},
                "metadata": metadata or {},
                "creation_time": time.time(),
                "last_updated": time.time()
            }
            
            # 添加到图
            self.graph.add_node(node_id, **node_data)
            
            # 更新索引
            self.node_index[node_id] = node_data
            
            # 更新统计
            self.stats["nodes"] += 1
            
            logger.debug(f"添加节点: {node_id} (类型: {node_type})")
            return True
            
        except Exception as e:
            logger.error(f"添加节点失败: {node_id}, 错误: {e}")
            return False
    
    def add_causal_relation(self,
                           cause: str,
                           effect: str,
                           strength: Union[CausalStrength, str] = CausalStrength.MODERATE,
                           confidence: float = 0.7,
                           evidence: Optional[List[Dict[str, Any]]] = None,
                           properties: Optional[Dict[str, Any]] = None) -> bool:
        """
        添加因果关系
        
        Args:
            cause: 原因节点ID
            effect: 结果节点ID
            strength: 因果强度
            confidence: 置信度
            evidence: 证据列表
            properties: 边属性
            
        Returns:
            是否成功添加
        """
        try:
            # 确保节点存在
            if cause not in self.graph.nodes():
                self.add_node(cause, "variable")
            
            if effect not in self.graph.nodes():
                self.add_node(effect, "variable")
            
            # 转换强度为字符串
            if isinstance(strength, CausalStrength):
                strength_str = strength.value
            else:
                strength_str = str(strength)
            
            # 创建边数据
            edge_id = f"{cause}->{effect}"
            edge_data = {
                "id": edge_id,
                "cause": cause,
                "effect": effect,
                "strength": strength_str,
                "confidence": max(0.0, min(1.0, confidence)),
                "evidence": evidence or [],
                "properties": properties or {},
                "creation_time": time.time(),
                "last_updated": time.time()
            }
            
            # 检查是否已存在边
            if self.graph.has_edge(cause, effect):
                # 更新现有边
                existing_data = self.graph[cause][effect]
                existing_data.update(edge_data)
                logger.debug(f"更新因果关系: {cause} -> {effect}")
            else:
                # 添加新边
                self.graph.add_edge(cause, effect, **edge_data)
                self.stats["edges"] += 1
                logger.debug(f"添加因果关系: {cause} -> {effect} (强度: {strength_str}, 置信度: {confidence})")
            
            # 添加证据到索引
            if evidence:
                for ev in evidence:
                    ev_id = ev.get("id")
                    if ev_id:
                        self.evidence_index[ev_id] = ev
                        self.stats["evidence_count"] += 1
            
            # 更新元数据
            self.metadata["last_updated"] = time.time()
            self.stats["updates_applied"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"添加因果关系失败: {cause} -> {effect}, 错误: {e}")
            return False
    
    def add_evidence(self,
                     edge_id: str,
                     evidence: Dict[str, Any]) -> bool:
        """
        为因果关系添加证据
        
        Args:
            edge_id: 边ID (格式: "cause->effect")
            evidence: 证据数据
            
        Returns:
            是否成功添加
        """
        try:
            # 解析边ID
            if "->" not in edge_id:
                logger.error(f"无效的边ID格式: {edge_id}, 应为 'cause->effect'")
                return False
            
            cause, effect = edge_id.split("->", 1)
            
            if not self.graph.has_edge(cause, effect):
                logger.error(f"边不存在: {cause} -> {effect}")
                return False
            
            # 确保证据有ID
            if "id" not in evidence:
                evidence["id"] = f"ev_{len(self.evidence_index) + 1}"
            
            ev_id = evidence["id"]
            
            # 添加到证据索引
            self.evidence_index[ev_id] = evidence
            
            # 添加到边
            edge_data = self.graph[cause][effect]
            if "evidence" not in edge_data:
                edge_data["evidence"] = []
            
            edge_data["evidence"].append(evidence)
            edge_data["last_updated"] = time.time()
            
            # 更新置信度（基于证据数量和质量）
            self._update_edge_confidence(cause, effect)
            
            self.stats["evidence_count"] += 1
            self.metadata["last_updated"] = time.time()
            
            logger.debug(f"添加证据: {ev_id} 到边 {edge_id}")
            return True
            
        except Exception as e:
            logger.error(f"添加证据失败: {edge_id}, 错误: {e}")
            return False
    
    def _update_edge_confidence(self, cause: str, effect: str):
        """更新边的置信度（基于证据）"""
        if not self.graph.has_edge(cause, effect):
            return
        
        edge_data = self.graph[cause][effect]
        evidence_list = edge_data.get("evidence", [])
        
        if not evidence_list:
            return
        
        # 计算基于证据的置信度
        total_weight = 0.0
        weighted_confidence = 0.0
        
        for ev in evidence_list:
            # 证据质量权重
            ev_type = ev.get("type", "observational_study")
            ev_quality = self._evidence_quality_weight(ev_type)
            ev_confidence = ev.get("confidence", 0.5)
            
            weight = ev_quality * ev_confidence
            weighted_confidence += weight
            total_weight += ev_quality
        
        if total_weight > 0:
            new_confidence = weighted_confidence / total_weight
            # 与现有置信度加权平均
            old_confidence = edge_data.get("confidence", 0.5)
            updated_confidence = 0.7 * new_confidence + 0.3 * old_confidence
            edge_data["confidence"] = min(1.0, updated_confidence)
    
    def _evidence_quality_weight(self, evidence_type: str) -> float:
        """证据质量权重"""
        weights = {
            "randomized_trial": 1.0,
            "meta_analysis": 0.9,
            "mechanistic_reasoning": 0.8,
            "computational_model": 0.7,
            "observational_study": 0.6,
            "expert_knowledge": 0.5
        }
        return weights.get(evidence_type, 0.5)
    
    def query_causal_path(self,
                         source: str,
                         target: str,
                         max_length: Optional[int] = None,
                         min_confidence: Optional[float] = None) -> Dict[str, Any]:
        """
        查询因果路径
        
        Args:
            source: 源节点
            target: 目标节点
            max_length: 最大路径长度
            min_confidence: 最小置信度阈值
            
        Returns:
            路径查询结果
        """
        start_time = time.time()
        
        # 检查缓存
        cache_key = f"path_{source}_{target}_{max_length}_{min_confidence}"
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            cached_result["cached"] = True
            return cached_result
        
        # 参数设置
        max_len = max_length or self.max_path_length
        min_conf = min_confidence or self.min_confidence_threshold
        
        # 检查节点是否存在
        if source not in self.graph.nodes():
            return {
                "success": False,
                "error": f"源节点不存在: {source}",
                "source": source,
                "target": target
            }
        
        if target not in self.graph.nodes():
            return {
                "success": False,
                "error": f"目标节点不存在: {target}",
                "source": source,
                "target": target
            }
        
        # 查找所有简单路径
        try:
            all_paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_len))
        except nx.NodeNotFound:
            return {
                "success": False,
                "error": "节点不存在于图中",
                "source": source,
                "target": target
            }
        
        # 过滤和评估路径
        valid_paths = []
        for path in all_paths:
            path_info = self._evaluate_path(path, min_conf)
            if path_info["valid"]:
                valid_paths.append(path_info)
        
        # 排序路径（按总置信度降序）
        valid_paths.sort(key=lambda x: x["total_confidence"], reverse=True)
        
        # 构建结果
        result = {
            "success": True,
            "source": source,
            "target": target,
            "total_paths_found": len(all_paths),
            "valid_paths": len(valid_paths),
            "paths": valid_paths[:10],  # 限制返回数量
            "best_path": valid_paths[0] if valid_paths else None,
            "query_parameters": {
                "max_length": max_len,
                "min_confidence": min_conf
            },
            "performance": {
                "query_time": time.time() - start_time,
                "graph_size": (self.stats["nodes"], self.stats["edges"])
            }
        }
        
        # 缓存结果
        self.query_cache[cache_key] = result.copy()
        if len(self.query_cache) > 1000:
            # 清理旧缓存
            old_keys = list(self.query_cache.keys())[:100]
            for key in old_keys:
                del self.query_cache[key]
        
        self.stats["queries_processed"] += 1
        logger.debug(f"因果路径查询: {source} -> {target}, 找到{len(valid_paths)}条有效路径")
        
        return result
    
    def _evaluate_path(self, path: List[str], min_confidence: float) -> Dict[str, Any]:
        """评估路径质量和有效性"""
        if len(path) < 2:
            return {
                "valid": False,
                "error": "路径长度不足"
            }
        
        edges = []
        total_confidence = 1.0
        min_edge_confidence = 1.0
        
        for i in range(len(path) - 1):
            cause = path[i]
            effect = path[i + 1]
            
            if not self.graph.has_edge(cause, effect):
                return {
                    "valid": False,
                    "error": f"边不存在: {cause} -> {effect}"
                }
            
            edge_data = self.graph[cause][effect]
            edge_confidence = edge_data.get("confidence", 0.0)
            
            # 检查置信度阈值
            if edge_confidence < min_confidence:
                return {
                    "valid": False,
                    "error": f"边置信度不足: {cause} -> {effect} ({edge_confidence} < {min_confidence})"
                }
            
            edges.append({
                "cause": cause,
                "effect": effect,
                "strength": edge_data.get("strength", "unknown"),
                "confidence": edge_confidence,
                "evidence_count": len(edge_data.get("evidence", []))
            })
            
            total_confidence *= edge_confidence
            min_edge_confidence = min(min_edge_confidence, edge_confidence)
        
        # 计算路径分数
        path_length = len(path) - 1
        avg_confidence = total_confidence ** (1.0 / path_length) if path_length > 0 else 0.0
        
        # 路径质量分数（考虑长度和置信度）
        # 偏好较短的路径和较高的置信度
        length_penalty = math.exp(-0.5 * (path_length - 1))  # 长度惩罚
        quality_score = avg_confidence * length_penalty
        
        return {
            "valid": True,
            "path": path,
            "length": path_length,
            "edges": edges,
            "total_confidence": total_confidence,
            "avg_confidence": avg_confidence,
            "min_confidence": min_edge_confidence,
            "quality_score": quality_score
        }
    
    def find_mediators(self,
                       cause: str,
                       effect: str,
                       max_depth: int = 3) -> Dict[str, Any]:
        """
        查找中介变量
        
        Args:
            cause: 原因变量
            effect: 结果变量
            max_depth: 最大搜索深度
            
        Returns:
            中介变量分析结果
        """
        start_time = time.time()
        
        # 检查直接路径是否存在
        if not self.graph.has_edge(cause, effect):
            return {
                "success": False,
                "error": f"直接因果关系不存在: {cause} -> {effect}",
                "cause": cause,
                "effect": effect
            }
        
        # 查找所有从cause到effect的长度为2或3的路径
        mediators = set()
        mediation_paths = []
        
        # 查找长度2的路径: cause -> M -> effect
        for node in self.graph.successors(cause):
            if node != effect and self.graph.has_edge(node, effect):
                path = [cause, node, effect]
                path_info = self._evaluate_path(path, self.min_confidence_threshold)
                if path_info["valid"]:
                    mediators.add(node)
                    mediation_paths.append({
                        "mediator": node,
                        "path": path,
                        "path_info": path_info
                    })
        
        # 查找长度3的路径: cause -> M1 -> M2 -> effect
        if max_depth >= 3:
            for node1 in self.graph.successors(cause):
                for node2 in self.graph.successors(node1):
                    if node2 != effect and self.graph.has_edge(node2, effect):
                        path = [cause, node1, node2, effect]
                        path_info = self._evaluate_path(path, self.min_confidence_threshold)
                        if path_info["valid"]:
                            mediators.add(node1)
                            mediators.add(node2)
                            mediation_paths.append({
                                "mediators": [node1, node2],
                                "path": path,
                                "path_info": path_info
                            })
        
        # 分析每个中介变量
        mediator_analysis = []
        for mediator in mediators:
            analysis = self._analyze_mediator(cause, mediator, effect)
            mediator_analysis.append(analysis)
        
        # 按中介强度排序
        mediator_analysis.sort(key=lambda x: x.get("mediation_strength", 0), reverse=True)
        
        elapsed_time = time.time() - start_time
        result = {
            "success": True,
            "cause": cause,
            "effect": effect,
            "mediators_found": len(mediators),
            "mediators": list(mediators),
            "mediation_paths": mediation_paths,
            "mediator_analysis": mediator_analysis,
            "performance": {
                "analysis_time": elapsed_time,
                "max_depth": max_depth
            }
        }
        
        return result
    
    def _analyze_mediator(self, cause: str, mediator: str, effect: str) -> Dict[str, Any]:
        """分析中介变量"""
        # 检查路径存在性
        has_cause_to_mediator = self.graph.has_edge(cause, mediator)
        has_mediator_to_effect = self.graph.has_edge(mediator, effect)
        
        if not (has_cause_to_mediator and has_mediator_to_effect):
            return {
                "mediator": mediator,
                "valid_path": False,
                "mediation_strength": 0.0
            }
        
        # 获取边数据
        edge1 = self.graph[cause][mediator]
        edge2 = self.graph[mediator][effect]
        
        # 计算中介强度（边置信度的几何平均）
        conf1 = edge1.get("confidence", 0.0)
        conf2 = edge2.get("confidence", 0.0)
        mediation_strength = math.sqrt(conf1 * conf2)
        
        # 计算总效应和直接效应
        total_effect = 0.0
        if self.graph.has_edge(cause, effect):
            direct_edge = self.graph[cause][effect]
            total_effect = direct_edge.get("confidence", 0.0)
        
        # 中介比例（简化）
        mediation_proportion = mediation_strength / (total_effect + mediation_strength) if (total_effect + mediation_strength) > 0 else 0.0
        
        return {
            "mediator": mediator,
            "valid_path": True,
            "cause_to_mediator": {
                "strength": edge1.get("strength", "unknown"),
                "confidence": conf1,
                "evidence_count": len(edge1.get("evidence", []))
            },
            "mediator_to_effect": {
                "strength": edge2.get("strength", "unknown"),
                "confidence": conf2,
                "evidence_count": len(edge2.get("evidence", []))
            },
            "mediation_strength": mediation_strength,
            "total_effect": total_effect,
            "mediation_proportion": mediation_proportion,
            "interpretation": f"{mediator} mediates {mediation_proportion:.1%} of the effect from {cause} to {effect}"
        }
    
    def detect_confounders(self,
                           cause: str,
                           effect: str,
                           max_depth: int = 2) -> Dict[str, Any]:
        """
        检测潜在混杂变量
        
        Args:
            cause: 原因变量
            effect: 结果变量
            max_depth: 最大搜索深度
            
        Returns:
            混杂变量检测结果
        """
        start_time = time.time()
        
        # 查找共同原因
        confounders = set()
        confounding_paths = []
        
        # 方法1: 查找同时指向cause和effect的变量
        cause_predecessors = set(self.graph.predecessors(cause))
        effect_predecessors = set(self.graph.predecessors(effect))
        common_causes = cause_predecessors.intersection(effect_predecessors)
        
        for confounder in common_causes:
            # 检查路径有效性
            path1 = [confounder, cause]
            path2 = [confounder, effect]
            
            path1_info = self._evaluate_path(path1, self.min_confidence_threshold)
            path2_info = self._evaluate_path(path2, self.min_confidence_threshold)
            
            if path1_info["valid"] and path2_info["valid"]:
                confounders.add(confounder)
                confounding_paths.append({
                    "confounder": confounder,
                    "path_to_cause": path1,
                    "path_to_effect": path2,
                    "path1_info": path1_info,
                    "path2_info": path2_info
                })
        
        # 方法2: 查找间接混杂（通过中间变量）
        if max_depth >= 2:
            for node in self.graph.nodes():
                if node == cause or node == effect or node in common_causes:
                    continue
                
                # 检查 node -> X -> cause 和 node -> Y -> effect 模式
                for x in self.graph.successors(node):
                    if x != cause and self.graph.has_edge(x, cause):
                        for y in self.graph.successors(node):
                            if y != effect and self.graph.has_edge(y, effect):
                                path1 = [node, x, cause]
                                path2 = [node, y, effect]
                                
                                path1_info = self._evaluate_path(path1, self.min_confidence_threshold)
                                path2_info = self._evaluate_path(path2, self.min_confidence_threshold)
                                
                                if path1_info["valid"] and path2_info["valid"]:
                                    confounders.add(node)
                                    confounding_paths.append({
                                        "confounder": node,
                                        "path_to_cause": path1,
                                        "path_to_effect": path2,
                                        "path1_info": path1_info,
                                        "path2_info": path2_info,
                                        "indirect": True,
                                        "intermediate_nodes": [x, y]
                                    })
        
        # 分析每个混杂变量
        confounder_analysis = []
        for confounder in confounders:
            analysis = self._analyze_confounder(confounder, cause, effect)
            confounder_analysis.append(analysis)
        
        # 按混杂风险排序
        confounder_analysis.sort(key=lambda x: x.get("confounding_risk", 0), reverse=True)
        
        elapsed_time = time.time() - start_time
        result = {
            "success": True,
            "cause": cause,
            "effect": effect,
            "confounders_found": len(confounders),
            "confounders": list(confounders),
            "confounding_paths": confounding_paths,
            "confounder_analysis": confounder_analysis,
            "performance": {
                "detection_time": elapsed_time,
                "max_depth": max_depth
            }
        }
        
        return result
    
    def _analyze_confounder(self, confounder: str, cause: str, effect: str) -> Dict[str, Any]:
        """分析混杂变量"""
        # 获取路径信息
        has_confounder_to_cause = self.graph.has_edge(confounder, cause)
        has_confounder_to_effect = self.graph.has_edge(confounder, effect)
        
        if not (has_confounder_to_cause and has_confounder_to_effect):
            return {
                "confounder": confounder,
                "valid_confounding": False,
                "confounding_risk": 0.0
            }
        
        # 获取边数据
        edge1 = self.graph[confounder][cause]
        edge2 = self.graph[confounder][effect]
        
        # 计算混杂风险（边置信度的几何平均）
        conf1 = edge1.get("confidence", 0.0)
        conf2 = edge2.get("confidence", 0.0)
        confounding_risk = math.sqrt(conf1 * conf2)
        
        # 获取直接因果关系的置信度（如果有）
        direct_effect_conf = 0.0
        if self.graph.has_edge(cause, effect):
            direct_edge = self.graph[cause][effect]
            direct_effect_conf = direct_edge.get("confidence", 0.0)
        
        # 混杂比例（简化）
        confounding_ratio = confounding_risk / (direct_effect_conf + confounding_risk) if (direct_effect_conf + confounding_risk) > 0 else 0.0
        
        return {
            "confounder": confounder,
            "valid_confounding": True,
            "confounder_to_cause": {
                "strength": edge1.get("strength", "unknown"),
                "confidence": conf1,
                "evidence_count": len(edge1.get("evidence", []))
            },
            "confounder_to_effect": {
                "strength": edge2.get("strength", "unknown"),
                "confidence": conf2,
                "evidence_count": len(edge2.get("evidence", []))
            },
            "confounding_risk": confounding_risk,
            "direct_effect_confidence": direct_effect_conf,
            "confounding_ratio": confounding_ratio,
            "adjustment_required": confounding_ratio > 0.3,  # 阈值
            "interpretation": f"{confounder} confounds {confounding_ratio:.1%} of the observed relationship between {cause} and {effect}"
        }
    
    def causal_query_language(self, query: str) -> Dict[str, Any]:
        """
        因果查询语言接口
        
        支持查询类型:
        1. PATH <source> TO <target>
        2. MEDIATORS OF <cause> ON <effect>
        3. CONFOUNDERS OF <cause> AND <effect>
        4. DIRECT_EFFECT <cause> ON <effect>
        5. TOTAL_EFFECT <cause> ON <effect>
        
        Args:
            query: 查询字符串
            
        Returns:
            查询结果
        """
        start_time = time.time()
        
        # 解析查询
        query_lower = query.lower().strip()
        tokens = query_lower.split()
        
        if len(tokens) < 3:
            return {
                "success": False,
                "error": "查询太短",
                "query": query
            }
        
        result = None
        
        # 路径查询
        if tokens[0] == "path" and "to" in tokens:
            to_index = tokens.index("to")
            if to_index >= 1 and to_index + 1 < len(tokens):
                source = tokens[1]
                target = tokens[to_index + 1]
                result = self.query_causal_path(source, target)
        
        # 中介变量查询
        elif tokens[0] == "mediators" and "of" in tokens and "on" in tokens:
            of_index = tokens.index("of")
            on_index = tokens.index("on")
            if of_index + 1 < on_index and on_index + 1 < len(tokens):
                cause = tokens[of_index + 1]
                effect = tokens[on_index + 1]
                result = self.find_mediators(cause, effect)
        
        # 混杂变量查询
        elif tokens[0] == "confounders" and "of" in tokens and "and" in tokens:
            of_index = tokens.index("of")
            and_index = tokens.index("and")
            if of_index + 1 < and_index and and_index + 1 < len(tokens):
                cause = tokens[of_index + 1]
                effect = tokens[and_index + 1]
                result = self.detect_confounders(cause, effect)
        
        # 直接效应查询
        elif tokens[0] == "direct_effect" and "on" in tokens:
            on_index = tokens.index("on")
            if on_index >= 2 and on_index + 1 < len(tokens):
                cause = tokens[1]
                effect = tokens[on_index + 1]
                result = self._query_direct_effect(cause, effect)
        
        # 总效应查询
        elif tokens[0] == "total_effect" and "on" in tokens:
            on_index = tokens.index("on")
            if on_index >= 2 and on_index + 1 < len(tokens):
                cause = tokens[1]
                effect = tokens[on_index + 1]
                result = self._query_total_effect(cause, effect)
        
        else:
            result = {
                "success": False,
                "error": "不支持的查询类型",
                "query": query,
                "supported_queries": [
                    "PATH <source> TO <target>",
                    "MEDIATORS OF <cause> ON <effect>",
                    "CONFOUNDERS OF <cause> AND <effect>",
                    "DIRECT_EFFECT <cause> ON <effect>",
                    "TOTAL_EFFECT <cause> ON <effect>"
                ]
            }
        
        if result:
            elapsed_time = time.time() - start_time
            if "performance" not in result:
                result["performance"] = {}
            result["performance"]["query_time"] = elapsed_time
            result["original_query"] = query
        
        return result if result else {
            "success": False,
            "error": "查询解析失败",
            "query": query
        }
    
    def _query_direct_effect(self, cause: str, effect: str) -> Dict[str, Any]:
        """查询直接效应"""
        if not self.graph.has_edge(cause, effect):
            return {
                "success": False,
                "error": f"直接因果关系不存在: {cause} -> {effect}",
                "cause": cause,
                "effect": effect
            }
        
        edge_data = self.graph[cause][effect]
        
        return {
            "success": True,
            "query_type": "direct_effect",
            "cause": cause,
            "effect": effect,
            "direct_effect": {
                "strength": edge_data.get("strength", "unknown"),
                "confidence": edge_data.get("confidence", 0.0),
                "evidence_count": len(edge_data.get("evidence", [])),
                "evidence_types": list(set([ev.get("type", "unknown") for ev in edge_data.get("evidence", [])]))
            }
        }
    
    def _query_total_effect(self, cause: str, effect: str) -> Dict[str, Any]:
        """查询总效应（直接+间接）"""
        # 查找所有路径
        path_result = self.query_causal_path(cause, effect)
        
        if not path_result["success"] or path_result["valid_paths"] == 0:
            return {
                "success": False,
                "error": f"未找到从 {cause} 到 {effect} 的因果路径",
                "cause": cause,
                "effect": effect
            }
        
        # 计算总效应（基于最佳路径的置信度）
        best_path = path_result.get("best_path")
        total_effect_confidence = best_path["total_confidence"] if best_path else 0.0
        
        # 收集所有路径的效应
        all_path_effects = []
        for path_info in path_result.get("paths", []):
            all_path_effects.append({
                "path": path_info["path"],
                "confidence": path_info["total_confidence"],
                "length": path_info["length"],
                "quality": path_info["quality_score"]
            })
        
        return {
            "success": True,
            "query_type": "total_effect",
            "cause": cause,
            "effect": effect,
            "total_effect_confidence": total_effect_confidence,
            "direct_effect_exists": self.graph.has_edge(cause, effect),
            "paths_found": path_result["valid_paths"],
            "all_path_effects": all_path_effects,
            "interpretation": f"Total effect from {cause} to {effect} has confidence {total_effect_confidence:.3f} across {path_result['valid_paths']} paths"
        }
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        # 基本统计
        nodes = list(self.graph.nodes())
        edges = list(self.graph.edges())
        
        # 节点类型统计
        node_types = defaultdict(int)
        for node in nodes:
            node_data = self.graph.nodes[node]
            node_type = node_data.get("type", "unknown")
            node_types[node_type] += 1
        
        # 边强度统计
        edge_strengths = defaultdict(int)
        edge_confidences = []
        
        for u, v in edges:
            edge_data = self.graph[u][v]
            strength = edge_data.get("strength", "unknown")
            edge_strengths[strength] += 1
            
            confidence = edge_data.get("confidence", 0.0)
            edge_confidences.append(confidence)
        
        # 计算置信度统计
        if edge_confidences:
            avg_confidence = np.mean(edge_confidences)
            std_confidence = np.std(edge_confidences)
            min_confidence = np.min(edge_confidences)
            max_confidence = np.max(edge_confidences)
        else:
            avg_confidence = std_confidence = min_confidence = max_confidence = 0.0
        
        # 图属性
        is_dag = nx.is_directed_acyclic_graph(self.graph)
        
        # 连通性
        try:
            weakly_connected = nx.number_weakly_connected_components(self.graph)
            strongly_connected = nx.number_strongly_connected_components(self.graph)
        except:
            weakly_connected = strongly_connected = 1
        
        return {
            "basic_stats": {
                "nodes": len(nodes),
                "edges": len(edges),
                "evidence_count": self.stats["evidence_count"],
                "is_dag": is_dag
            },
            "node_stats": {
                "types": dict(node_types),
                "type_count": len(node_types)
            },
            "edge_stats": {
                "strengths": dict(edge_strengths),
                "confidence": {
                    "average": float(avg_confidence),
                    "std": float(std_confidence),
                    "min": float(min_confidence),
                    "max": float(max_confidence)
                }
            },
            "connectivity": {
                "weakly_connected_components": weakly_connected,
                "strongly_connected_components": strongly_connected
            },
            "performance_stats": self.stats.copy()
        }
    
    def save_graph(self, filepath: str, format: str = "gexf") -> bool:
        """
        保存图谱到文件
        
        Args:
            filepath: 文件路径
            format: 格式 ("gexf", "graphml", "pickle")
            
        Returns:
            是否成功保存
        """
        try:
            if format == "gexf":
                nx.write_gexf(self.graph, filepath)
            elif format == "graphml":
                nx.write_graphml(self.graph, filepath)
            elif format == "pickle":
                import pickle
                graph_data = {
                    "graph": self.graph,
                    "metadata": self.metadata,
                    "stats": self.stats,
                    "node_index": self.node_index,
                    "evidence_index": self.evidence_index
                }
                with open(filepath, 'wb') as f:
                    pickle.dump(graph_data, f)
            else:
                logger.error(f"不支持的格式: {format}")
                return False
            
            logger.info(f"图谱保存到: {filepath} (格式: {format})")
            return True
            
        except Exception as e:
            logger.error(f"保存图谱失败: {e}")
            return False
    
    def load_graph(self, filepath: str, format: str = "gexf") -> bool:
        """
        从文件加载图谱
        
        Args:
            filepath: 文件路径
            format: 格式 ("gexf", "graphml", "pickle")
            
        Returns:
            是否成功加载
        """
        try:
            if format == "gexf":
                self.graph = nx.read_gexf(filepath)
            elif format == "graphml":
                self.graph = nx.read_graphml(filepath)
            elif format == "pickle":
                import pickle
                with open(filepath, 'rb') as f:
                    graph_data = pickle.load(f)
                self.graph = graph_data.get("graph", nx.DiGraph())
                self.metadata = graph_data.get("metadata", self.metadata)
                self.stats = graph_data.get("stats", self.stats.copy())
                self.node_index = graph_data.get("node_index", {})
                self.evidence_index = graph_data.get("evidence_index", {})
            else:
                logger.error(f"不支持的格式: {format}")
                return False
            
            # 更新统计
            self.stats["nodes"] = self.graph.number_of_nodes()
            self.stats["edges"] = self.graph.number_of_edges()
            
            logger.info(f"图谱从 {filepath} 加载 (格式: {format})")
            return True
            
        except Exception as e:
            logger.error(f"加载图谱失败: {e}")
            return False


# 示例和测试函数
def create_example_causal_graph() -> CausalKnowledgeGraph:
    """创建示例因果知识图谱"""
    graph = CausalKnowledgeGraph(name="Example Causal Graph")
    
    # 添加节点
    nodes = [
        ("smoking", "behavior"),
        ("lung_cancer", "disease"),
        ("genetics", "factor"),
        ("age", "demographic"),
        ("exercise", "behavior"),
        ("health", "outcome")
    ]
    
    for node_id, node_type in nodes:
        graph.add_node(node_id, node_type)
    
    # 添加因果关系
    causal_relations = [
        ("smoking", "lung_cancer", CausalStrength.STRONG, 0.9),
        ("genetics", "lung_cancer", CausalStrength.MODERATE, 0.7),
        ("age", "lung_cancer", CausalStrength.WEAK, 0.5),
        ("smoking", "health", CausalStrength.MODERATE, 0.6),
        ("exercise", "health", CausalStrength.STRONG, 0.8),
        ("genetics", "health", CausalStrength.WEAK, 0.4)
    ]
    
    for cause, effect, strength, confidence in causal_relations:
        graph.add_causal_relation(cause, effect, strength, confidence)
    
    # 添加证据
    evidence = {
        "id": "ev_smoking_cancer",
        "type": EvidenceType.META_ANALYSIS.value,
        "description": "Meta-analysis of 50 studies on smoking and lung cancer",
        "sample_size": 1000000,
        "effect_size": 0.85,
        "p_value": 0.001,
        "confidence": 0.95
    }
    
    graph.add_evidence("smoking->lung_cancer", evidence)
    
    return graph


def test_causal_knowledge_graph():
    """测试因果知识图谱"""
    logger.info("开始测试因果知识图谱")
    
    # 创建示例图谱
    graph = create_example_causal_graph()
    
    # 测试路径查询
    logger.info("测试路径查询...")
    path_result = graph.query_causal_path("smoking", "health")
    if path_result["success"]:
        logger.info(f"路径查询: 找到{path_result['valid_paths']}条有效路径")
        if path_result["best_path"]:
            best_path = path_result["best_path"]["path"]
            logger.info(f"最佳路径: {' -> '.join(best_path)}")
    
    # 测试中介变量检测
    logger.info("测试中介变量检测...")
    mediator_result = graph.find_mediators("smoking", "health")
    if mediator_result["success"]:
        logger.info(f"中介变量检测: 找到{mediator_result['mediators_found']}个中介变量")
        for mediator in mediator_result["mediators"]:
            logger.info(f"  中介变量: {mediator}")
    
    # 测试混杂变量检测
    logger.info("测试混杂变量检测...")
    confounder_result = graph.detect_confounders("smoking", "lung_cancer")
    if confounder_result["success"]:
        logger.info(f"混杂变量检测: 找到{confounder_result['confounders_found']}个混杂变量")
        for confounder in confounder_result["confounders"]:
            logger.info(f"  混杂变量: {confounder}")
    
    # 测试查询语言
    logger.info("测试因果查询语言...")
    queries = [
        "PATH smoking TO health",
        "MEDIATORS OF smoking ON health",
        "CONFOUNDERS OF smoking AND lung_cancer",
        "DIRECT_EFFECT smoking ON lung_cancer"
    ]
    
    for query in queries:
        result = graph.causal_query_language(query)
        logger.info(f"查询: {query} -> 成功: {result.get('success', False)}")
    
    # 显示统计信息
    stats = graph.get_graph_statistics()
    logger.info(f"图谱统计: {stats['basic_stats']}")
    
    logger.info("因果知识图谱测试完成")
    return graph


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_graph = test_causal_knowledge_graph()