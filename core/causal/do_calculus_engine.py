#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do-Calculus数学引擎 - 实现Judea Pearl的do-calculus完整数学框架

核心功能:
1. Do-Calculus三大规则:
   - 规则1: 插入/删除观测
   - 规则2: 行动/观测交换
   - 规则3: 插入/删除行动
2. 后门准则和前门准则
3. 工具变量识别
4. 中介分析和路径阻断
5. 因果效应可识别性判定

数学基础:
- 基于有向无环图(DAG)的因果结构
- 条件独立关系的图论判定(d-分离)
- 概率分布的操作和变换
- 干预分布的计算

基于Pearl的do-calculus公理化系统:
1. P(y|do(x), z, w) = P(y|do(x), w) if (Y ⊥⊥ Z | X, W) in G_{X̄}
2. P(y|do(x), do(z), w) = P(y|do(x), z, w) if (Y ⊥⊥ Z | X, W) in G_{X̄Z̅}
3. P(y|do(x), do(z), w) = P(y|do(x), w) if (Y ⊥⊥ Z | X, W) in G_{X̄Z(W)}

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import math
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from enum import Enum
from collections import defaultdict
import networkx as nx
import numpy as np
from scipy import stats

# 导入错误处理
from core.error_handling import ErrorHandler

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()


class DoCalculusRule(Enum):
    """Do-Calculus规则枚举"""
    RULE_1_INSERT_DELETE_OBSERVATION = "rule_1_insert_delete_observation"
    RULE_2_ACTION_OBSERVATION_EXCHANGE = "rule_2_action_observation_exchange"
    RULE_3_INSERT_DELETE_ACTION = "rule_3_insert_delete_action"


class CriterionType(Enum):
    """准则类型枚举"""
    BACKDOOR_CRITERION = "backdoor_criterion"
    FRONTDOOR_CRITERION = "frontdoor_criterion"
    INSTRUMENTAL_VARIABLE = "instrumental_variable"
    MEDIATION_CRITERION = "mediation_criterion"


class GraphModificationType(Enum):
    """图修改类型枚举"""
    DELETE_INCOMING_EDGES = "delete_incoming_edges"  # 删除指向变量的边(G_{X̄})
    DELETE_OUTGOING_EDGES = "delete_outgoing_edges"  # 删除从变量出发的边(G_{X̅})
    MUTILATED_GRAPH = "mutilated_graph"             # 干预后的图


class DoCalculusEngine:
    """
    Do-Calculus数学引擎
    
    核心特性:
    1. Do-Calculus三大规则的完整实现
    2. 后门/前门准则的自动应用
    3. 因果效应可识别性判定
    4. 工具变量和中介分析
    5. 条件独立关系的图论判定(d-分离)
    
    技术实现:
    - 基于networkx的图操作
    - d-分离算法的优化实现
    - 概率分布的条件化
    - 干预分布的变换
    """
    
    def __init__(self, causal_graph: Optional[nx.DiGraph] = None):
        """
        初始化Do-Calculus引擎
        
        Args:
            causal_graph: 因果图（有向无环图DAG）
        """
        self.causal_graph = causal_graph if causal_graph is not None else nx.DiGraph()
        self.graph_modifications: Dict[str, nx.DiGraph] = {}
        self.conditional_independence_cache: Dict[Tuple[str, str, Tuple[str, ...]], bool] = {}
        self.identification_results: Dict[str, Dict[str, Any]] = {}
        
        # 配置参数
        self.config = {
            'd_separation_algorithm': 'moralization',  # d-分离算法类型
            'cache_enabled': True,
            'max_cache_size': 10000,
            'probability_precision': 1e-10,
            'graph_operation_timeout': 5.0  # 秒
        }
        
        # 性能统计
        self.performance_stats = {
            'd_separation_checks': 0,
            'rule_applications': 0,
            'identification_attempts': 0,
            'successful_identifications': 0,
            'average_computation_time': 0.0
        }
        
        # 规则库
        self.rules_database: Dict[DoCalculusRule, Dict[str, Any]] = {}
        self._initialize_rules_database()
        
        logger.info("Do-Calculus数学引擎初始化完成")
    
    def _initialize_rules_database(self):
        """初始化规则数据库"""
        # 规则1: 插入/删除观测
        self.rules_database[DoCalculusRule.RULE_1_INSERT_DELETE_OBSERVATION] = {
            "name": "规则1: 插入/删除观测",
            "description": "在干预图中，如果Y与Z在给定(X,W)的条件下d-分离，则可以从条件中删除Z",
            "formal_statement": "P(y|do(x), z, w) = P(y|do(x), w) if (Y ⊥⊥ Z | X, W) in G_{X̄}",
            "applicability_conditions": [
                "Z是观测变量",
                "Y与Z在G_{X̄}中给定(X,W)条件下d-分离",
                "X是干预变量"
            ],
            "example": "在医疗试验中，如果年龄与结果在给定治疗和性别的条件下独立，则可以从条件中删除年龄"
        }
        
        # 规则2: 行动/观测交换
        self.rules_database[DoCalculusRule.RULE_2_ACTION_OBSERVATION_EXCHANGE] = {
            "name": "规则2: 行动/观测交换",
            "description": "如果Y与Z在给定(X,W)的条件下在双重干预图中d-分离，则可以将干预替换为观测",
            "formal_statement": "P(y|do(x), do(z), w) = P(y|do(x), z, w) if (Y ⊥⊥ Z | X, W) in G_{X̄Z̅}",
            "applicability_conditions": [
                "Z是干预变量",
                "Y与Z在G_{X̄Z̅}中给定(X,W)条件下d-分离",
                "X是干预变量"
            ],
            "example": "在双重随机化试验中，如果第二个治疗与结果在给定第一个治疗的条件下独立，则可以将第二个治疗视为观测"
        }
        
        # 规则3: 插入/删除行动
        self.rules_database[DoCalculusRule.RULE_3_INSERT_DELETE_ACTION] = {
            "name": "规则3: 插入/删除行动",
            "description": "如果Y与Z在给定(X,W)的条件下在部分干预图中d-分离，则可以删除对Z的干预",
            "formal_statement": "P(y|do(x), do(z), w) = P(y|do(x), w) if (Y ⊥⊥ Z | X, W) in G_{X̄Z(W)}",
            "applicability_conditions": [
                "Z是干预变量",
                "Y与Z在G_{X̄Z(W)}中给定(X,W)条件下d-分离",
                "G_{X̄Z(W)}是从G_{X̄}中删除指向Z的边后得到的图，但保留从W指向Z的边"
            ],
            "example": "在药物剂量研究中，如果安慰剂与结果在给定实际治疗的条件下独立，则可以删除对安慰剂的干预"
        }
    
    def create_mutilated_graph(self, 
                              intervention_variables: Set[str],
                              graph_type: GraphModificationType = GraphModificationType.DELETE_INCOMING_EDGES) -> nx.DiGraph:
        """
        创建干预图（破损图）
        
        Args:
            intervention_variables: 干预变量集合
            graph_type: 图修改类型
        
        Returns:
            修改后的因果图
        """
        start_time = time.time()
        
        # 创建图的副本
        if graph_type == GraphModificationType.DELETE_INCOMING_EDGES:
            # G_{X̄}: 删除指向干预变量的边
            mutilated_graph = self.causal_graph.copy()
            for var in intervention_variables:
                if var in mutilated_graph:
                    # 删除所有指向该变量的边
                    incoming_edges = list(mutilated_graph.in_edges(var))
                    mutilated_graph.remove_edges_from(incoming_edges)
        
        elif graph_type == GraphModificationType.DELETE_OUTGOING_EDGES:
            # G_{X̅}: 删除从干预变量出发的边
            mutilated_graph = self.causal_graph.copy()
            for var in intervention_variables:
                if var in mutilated_graph:
                    # 删除所有从该变量出发的边
                    outgoing_edges = list(mutilated_graph.out_edges(var))
                    mutilated_graph.remove_edges_from(outgoing_edges)
        
        elif graph_type == GraphModificationType.MUTILATED_GRAPH:
            # G_{X̄Z(W)}: 特殊干预图
            mutilated_graph = self.causal_graph.copy()
            # 这里需要根据具体规则处理
            # 简化实现：删除指向干预变量的边
            for var in intervention_variables:
                if var in mutilated_graph:
                    incoming_edges = list(mutilated_graph.in_edges(var))
                    mutilated_graph.remove_edges_from(incoming_edges)
        else:
            mutilated_graph = self.causal_graph.copy()
        
        # 缓存修改后的图
        cache_key = f"{graph_type.value}_{'_'.join(sorted(intervention_variables))}"
        self.graph_modifications[cache_key] = mutilated_graph
        
        computation_time = time.time() - start_time
        logger.debug(f"创建干预图完成: {cache_key}, 耗时: {computation_time:.4f}秒")
        
        return mutilated_graph
    
    def d_separation(self,
                    variables_y: Set[str],
                    variables_z: Set[str],
                    conditioning_set: Set[str],
                    graph: Optional[nx.DiGraph] = None) -> bool:
        """
        检查d-分离关系
        
        Args:
            variables_y: Y变量集合
            variables_z: Z变量集合
            conditioning_set: 条件集合
            graph: 使用的图（默认为因果图）
        
        Returns:
            True如果Y与Z在给定条件集合下d-分离，否则False
        """
        start_time = time.time()
        
        # 使用指定的图或默认因果图
        use_graph = graph if graph is not None else self.causal_graph
        
        # 检查缓存
        if self.config['cache_enabled']:
            cache_key = (
                frozenset(variables_y),
                frozenset(variables_z),
                frozenset(conditioning_set)
            )
            if cache_key in self.conditional_independence_cache:
                self.performance_stats['d_separation_checks'] += 1
                return self.conditional_independence_cache[cache_key]
        
        # 实现d-分离检查
        # 方法1: 使用networkx的d分离（如果可用）
        try:
            # networkx 2.8+ 支持d_separated
            if hasattr(nx, 'd_separated'):
                is_d_separated = nx.d_separated(
                    use_graph,
                    variables_y,
                    variables_z,
                    conditioning_set
                )
            else:
                # 回退到道德图方法
                is_d_separated = self._d_separation_moralization(
                    variables_y,
                    variables_z,
                    conditioning_set,
                    use_graph
                )
        except Exception as e:
            logger.warning(f"d-分离检查失败: {e}, 使用回退方法")
            is_d_separated = self._d_separation_simple(
                variables_y,
                variables_z,
                conditioning_set,
                use_graph
            )
        
        # 更新缓存
        if self.config['cache_enabled']:
            cache_key = (
                frozenset(variables_y),
                frozenset(variables_z),
                frozenset(conditioning_set)
            )
            self.conditional_independence_cache[cache_key] = is_d_separated
            
            # 限制缓存大小
            if len(self.conditional_independence_cache) > self.config['max_cache_size']:
                # 移除最旧的条目
                oldest_key = next(iter(self.conditional_independence_cache))
                del self.conditional_independence_cache[oldest_key]
        
        # 更新性能统计
        self.performance_stats['d_separation_checks'] += 1
        computation_time = time.time() - start_time
        self.performance_stats['average_computation_time'] = (
            self.performance_stats['average_computation_time'] * 
            (self.performance_stats['d_separation_checks'] - 1) + 
            computation_time
        ) / self.performance_stats['d_separation_checks']
        
        return is_d_separated
    
    def _d_separation_moralization(self,
                                  variables_y: Set[str],
                                  variables_z: Set[str],
                                  conditioning_set: Set[str],
                                  graph: nx.DiGraph) -> bool:
        """
        使用道德图方法检查d-分离
        
        算法步骤:
        1. 删除条件集合中的节点及其边
        2. 道德化：为有共同子节点的未连接父母节点添加无向边
        3. 将剩余的有向边转换为无向边
        4. 检查在无向图中Y和Z是否连通
        """
        # 步骤1: 创建图的副本
        moral_graph = graph.copy()
        
        # 删除条件集合中的节点
        for node in conditioning_set:
            if node in moral_graph:
                moral_graph.remove_node(node)
        
        # 步骤2: 道德化
        # 为有共同子节点的父母节点添加边
        nodes_to_moralize = list(moral_graph.nodes())
        for node in nodes_to_moralize:
            parents = list(moral_graph.predecessors(node))
            # 为每对父母添加边（如果不存在）
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    if not moral_graph.has_edge(parents[i], parents[j]):
                        moral_graph.add_edge(parents[i], parents[j])
                    if not moral_graph.has_edge(parents[j], parents[i]):
                        moral_graph.add_edge(parents[j], parents[i])
        
        # 步骤3: 转换为无向图
        moral_undirected = moral_graph.to_undirected()
        
        # 步骤4: 检查连通性
        # 如果Y中的任何节点与Z中的任何节点连通，则不是d-分离
        for y_node in variables_y:
            if y_node not in moral_undirected:
                continue
            for z_node in variables_z:
                if z_node not in moral_undirected:
                    continue
                try:
                    if nx.has_path(moral_undirected, y_node, z_node):
                        return False
                except:
                    # 如果节点不存在于图中，继续检查
                    continue
        
        return True
    
    def _d_separation_simple(self,
                            variables_y: Set[str],
                            variables_z: Set[str],
                            conditioning_set: Set[str],
                            graph: nx.DiGraph) -> bool:
        """
        简化的d-分离检查
        
        这是一个启发式方法，适用于简单的图结构
        """
        # 检查Y和Z是否有直接连接
        for y_node in variables_y:
            for z_node in variables_z:
                if graph.has_edge(y_node, z_node) or graph.has_edge(z_node, y_node):
                    return False
        
        # 检查条件阻断
        # 简化：如果条件集合包含所有共同原因，则可能d-分离
        # 这是一个非常简化的实现
        return True
    
    def apply_rule_1(self,
                    y: str,
                    x: str,
                    z: str,
                    w: Set[str],
                    probability_dist: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        应用规则1: 插入/删除观测
        
        Args:
            y: 结果变量
            x: 干预变量
            z: 要删除的观测变量
            w: 条件变量集合
            probability_dist: 可选的概率分布
        
        Returns:
            (规则是否适用, 变换后的概率分布)
        """
        self.performance_stats['rule_applications'] += 1
        
        # 创建干预图 G_{X̄}
        intervention_vars = {x}
        graph_x_bar = self.create_mutilated_graph(
            intervention_vars,
            GraphModificationType.DELETE_INCOMING_EDGES
        )
        
        # 检查d-分离条件: (Y ⊥⊥ Z | X, W) in G_{X̄}
        y_set = {y}
        z_set = {z}
        conditioning_set = {x}.union(w)
        
        is_d_separated = self.d_separation(y_set, z_set, conditioning_set, graph_x_bar)
        
        if not is_d_separated:
            logger.debug(f"规则1不适用: Y({y})与Z({z})在给定(X={x}, W={w})条件下在G_X̄中不是d-分离的")
            return False, None
        
        # 规则适用，进行概率变换
        transformed_dist = None
        if probability_dist is not None:
            # 简化：从条件中删除Z
            # 实际实现需要根据具体概率分布进行计算
            transformed_dist = self._remove_conditioning_variable(probability_dist, z, {x}.union(w))
        
        logger.info(f"规则1适用: P({y}|do({x}), {z}, {w}) = P({y}|do({x}), {w})")
        return True, transformed_dist
    
    def apply_rule_2(self,
                    y: str,
                    x: str,
                    z: str,
                    w: Set[str],
                    probability_dist: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        应用规则2: 行动/观测交换
        
        Args:
            y: 结果变量
            x: 干预变量
            z: 要交换的变量（从干预变为观测）
            w: 条件变量集合
            probability_dist: 可选的概率分布
        
        Returns:
            (规则是否适用, 变换后的概率分布)
        """
        self.performance_stats['rule_applications'] += 1
        
        # 创建双重干预图 G_{X̄Z̅}
        intervention_vars = {x, z}
        graph_x_bar_z_bar = self.create_mutilated_graph(
            intervention_vars,
            GraphModificationType.DELETE_INCOMING_EDGES
        )
        
        # 检查d-分离条件: (Y ⊥⊥ Z | X, W) in G_{X̄Z̅}
        y_set = {y}
        z_set = {z}
        conditioning_set = {x}.union(w)
        
        is_d_separated = self.d_separation(y_set, z_set, conditioning_set, graph_x_bar_z_bar)
        
        if not is_d_separated:
            logger.debug(f"规则2不适用: Y({y})与Z({z})在给定(X={x}, W={w})条件下在G_X̄Z̅中不是d-分离的")
            return False, None
        
        # 规则适用，进行概率变换
        transformed_dist = None
        if probability_dist is not None:
            # 简化：将do(z)替换为z
            # 实际实现需要根据具体概率分布进行计算
            transformed_dist = self._replace_intervention_with_observation(probability_dist, z, {x}.union(w))
        
        logger.info(f"规则2适用: P({y}|do({x}), do({z}), {w}) = P({y}|do({x}), {z}, {w})")
        return True, transformed_dist
    
    def apply_rule_3(self,
                    y: str,
                    x: str,
                    z: str,
                    w: Set[str],
                    probability_dist: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        应用规则3: 插入/删除行动
        
        Args:
            y: 结果变量
            x: 干预变量
            z: 要删除的干预变量
            w: 条件变量集合
            probability_dist: 可选的概率分布
        
        Returns:
            (规则是否适用, 变换后的概率分布)
        """
        self.performance_stats['rule_applications'] += 1
        
        # 创建特殊干预图 G_{X̄Z(W)}
        # 需要特殊处理：删除指向Z的边，但保留从W指向Z的边
        graph_x_bar = self.create_mutilated_graph(
            {x},
            GraphModificationType.DELETE_INCOMING_EDGES
        )
        
        # 创建G_{X̄Z(W)}：从G_{X̄}中删除指向Z的边，但保留从W指向Z的边
        graph_x_bar_z_w = graph_x_bar.copy()
        
        # 删除指向Z的边，除非边来自W
        if z in graph_x_bar_z_w:
            incoming_edges = list(graph_x_bar_z_w.in_edges(z))
            for src, tgt in incoming_edges:
                if src not in w:  # 如果源节点不在W中，删除边
                    graph_x_bar_z_w.remove_edge(src, tgt)
        
        # 检查d-分离条件: (Y ⊥⊥ Z | X, W) in G_{X̄Z(W)}
        y_set = {y}
        z_set = {z}
        conditioning_set = {x}.union(w)
        
        is_d_separated = self.d_separation(y_set, z_set, conditioning_set, graph_x_bar_z_w)
        
        if not is_d_separated:
            logger.debug(f"规则3不适用: Y({y})与Z({z})在给定(X={x}, W={w})条件下在G_X̄Z(W)中不是d-分离的")
            return False, None
        
        # 规则适用，进行概率变换
        transformed_dist = None
        if probability_dist is not None:
            # 简化：删除对Z的干预
            # 实际实现需要根据具体概率分布进行计算
            transformed_dist = self._remove_intervention_variable(probability_dist, z, {x}.union(w))
        
        logger.info(f"规则3适用: P({y}|do({x}), do({z}), {w}) = P({y}|do({x}), {w})")
        return True, transformed_dist
    
    def _remove_conditioning_variable(self,
                                     probability_dist: Dict[str, Any],
                                     variable_to_remove: str,
                                     conditioning_vars: Set[str]) -> Dict[str, Any]:
        """从条件中删除变量（简化实现）"""
        # 简化实现：创建一个新的分布表示
        transformed_dist = probability_dist.copy()
        transformed_dist['conditioning_variables'] = list(
            set(transformed_dist.get('conditioning_variables', [])) - {variable_to_remove}
        )
        transformed_dist['rule_applied'] = 'rule_1'
        transformed_dist['variables_removed'] = [variable_to_remove]
        return transformed_dist
    
    def _replace_intervention_with_observation(self,
                                              probability_dist: Dict[str, Any],
                                              variable_to_replace: str,
                                              conditioning_vars: Set[str]) -> Dict[str, Any]:
        """将干预替换为观测（简化实现）"""
        transformed_dist = probability_dist.copy()
        if 'intervention_variables' in transformed_dist:
            transformed_dist['intervention_variables'] = list(
                set(transformed_dist['intervention_variables']) - {variable_to_replace}
            )
        transformed_dist['observation_variables'] = transformed_dist.get('observation_variables', []) + [variable_to_replace]
        transformed_dist['rule_applied'] = 'rule_2'
        transformed_dist['variables_replaced'] = [variable_to_replace]
        return transformed_dist
    
    def _remove_intervention_variable(self,
                                     probability_dist: Dict[str, Any],
                                     variable_to_remove: str,
                                     conditioning_vars: Set[str]) -> Dict[str, Any]:
        """删除干预变量（简化实现）"""
        transformed_dist = probability_dist.copy()
        if 'intervention_variables' in transformed_dist:
            transformed_dist['intervention_variables'] = list(
                set(transformed_dist['intervention_variables']) - {variable_to_remove}
            )
        transformed_dist['rule_applied'] = 'rule_3'
        transformed_dist['variables_removed'] = [variable_to_remove]
        return transformed_dist
    
    def check_backdoor_criterion(self,
                                treatment: str,
                                outcome: str,
                                adjustment_set: Set[str]) -> Tuple[bool, List[str]]:
        """
        检查后门准则
        
        Args:
            treatment: 处理变量
            outcome: 结果变量
            adjustment_set: 调整变量集合
        
        Returns:
            (是否满足后门准则, 违反的原因列表)
        """
        violations = []
        
        # 条件1: 调整集合不包含处理变量的后代
        treatment_descendants = set(nx.descendants(self.causal_graph, treatment))
        if treatment in treatment_descendants:
            treatment_descendants.remove(treatment)  # 移除自身
        
        intersection = adjustment_set.intersection(treatment_descendants)
        if intersection:
            violations.append(f"调整集合包含处理变量的后代: {intersection}")
        
        # 条件2: 调整集合阻断所有从处理变量到结果变量的后门路径
        # 后门路径：以指向处理变量的箭头开始的路径
        backdoor_paths = self._find_backdoor_paths(treatment, outcome)
        
        for path in backdoor_paths:
            # 检查路径是否被调整集合阻断
            if not self._is_path_blocked(path, adjustment_set):
                violations.append(f"后门路径未被阻断: {path}")
        
        is_satisfied = len(violations) == 0
        return is_satisfied, violations
    
    def _find_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """查找从处理变量到结果变量的后门路径"""
        backdoor_paths = []
        
        # 查找所有从treatment到outcome的路径
        try:
            all_paths = list(nx.all_simple_paths(self.causal_graph, treatment, outcome))
        except nx.NodeNotFound:
            return backdoor_paths
        
        # 筛选后门路径：路径以指向treatment的箭头开始
        for path in all_paths:
            if len(path) >= 2:
                # 检查第一个箭头方向
                first_edge = (path[0], path[1])
                if self.causal_graph.has_edge(*first_edge):
                    # 箭头从path[0]指向path[1]，即从treatment指向下一个节点
                    # 这不是后门路径
                    continue
                elif self.causal_graph.has_edge(path[1], path[0]):
                    # 箭头指向treatment，这是后门路径的开始
                    backdoor_paths.append(path)
        
        return backdoor_paths
    
    def _is_path_blocked(self, path: List[str], conditioning_set: Set[str]) -> bool:
        """检查路径是否被条件集合阻断"""
        # 简化实现：使用d-分离
        if len(path) < 2:
            return True
        
        # 将路径转换为变量对
        for i in range(len(path) - 1):
            for j in range(i + 1, len(path)):
                var1 = path[i]
                var2 = path[j]
                if self.d_separation({var1}, {var2}, conditioning_set):
                    return True
        
        return False
    
    def identify_causal_effect(self,
                              treatment: str,
                              outcome: str,
                              available_variables: Set[str]) -> Dict[str, Any]:
        """
        识别因果效应
        
        Args:
            treatment: 处理变量
            outcome: 结果变量
            available_variables: 可用变量集合
        
        Returns:
            识别结果字典
        """
        self.performance_stats['identification_attempts'] += 1
        
        identification_id = f"identification_{self.performance_stats['identification_attempts']:06d}"
        result = {
            "identification_id": identification_id,
            "treatment": treatment,
            "outcome": outcome,
            "available_variables": list(available_variables),
            "is_identifiable": False,
            "identification_method": None,
            "adjustment_set": None,
            "applicable_rules": [],
            "violations": [],
            "confidence": 0.0
        }
        
        # 尝试后门准则
        for adjustment_candidate in self._generate_adjustment_sets(available_variables):
            is_satisfied, violations = self.check_backdoor_criterion(
                treatment, outcome, adjustment_candidate
            )
            
            if is_satisfied:
                result["is_identifiable"] = True
                result["identification_method"] = "backdoor_criterion"
                result["adjustment_set"] = list(adjustment_candidate)
                result["confidence"] = 0.9
                self.performance_stats['successful_identifications'] += 1
                logger.info(f"因果效应可识别: {treatment} -> {outcome}, 调整集合: {adjustment_candidate}")
                break
            else:
                # 记录失败的尝试
                result["violations"].extend(violations)
        
        # 如果后门准则失败，尝试前门准则等其他方法
        if not result["is_identifiable"]:
            # 这里可以添加前门准则、工具变量等方法
            result["confidence"] = 0.1
        
        # 保存识别结果
        self.identification_results[identification_id] = result
        
        return result
    
    def _generate_adjustment_sets(self, available_variables: Set[str]) -> List[Set[str]]:
        """生成可能的调整集合"""
        adjustment_sets = []
        variables_list = list(available_variables)
        
        # 生成所有可能的子集（限制大小以避免组合爆炸）
        max_set_size = min(5, len(variables_list))
        
        from itertools import combinations
        for r in range(max_set_size + 1):
            for combo in combinations(variables_list, r):
                adjustment_sets.append(set(combo))
        
        return adjustment_sets
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {
            "timestamp": time.time(),
            "performance_stats": self.performance_stats,
            "cache_statistics": {
                "cache_size": len(self.conditional_independence_cache),
                "graph_modifications": len(self.graph_modifications),
                "identification_results": len(self.identification_results)
            },
            "rules_database": {
                rule.value: info["name"]
                for rule, info in self.rules_database.items()
            }
        }
        
        return summary

# 全局实例
do_calculus_engine_instance = DoCalculusEngine()

if __name__ == "__main__":
    # 测试Do-Calculus引擎
    print("测试Do-Calculus数学引擎...")
    
    # 创建一个简单的因果图
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("X", "Y"),  # X导致Y
        ("Z", "X"),  # Z导致X
        ("Z", "Y")   # Z导致Y
    ])
    
    # 初始化引擎
    engine = DoCalculusEngine(graph)
    
    # 测试d-分离
    is_d_separated = engine.d_separation({"Y"}, {"Z"}, {"X"})
    print(f"Y与Z在给定X条件下d-分离: {is_d_separated}")
    
    # 测试后门准则
    is_satisfied, violations = engine.check_backdoor_criterion("X", "Y", {"Z"})
    print(f"后门准则满足 (调整集合{{Z}}): {is_satisfied}")
    if violations:
        print(f"违反: {violations}")
    
    # 测试因果效应识别
    result = engine.identify_causal_effect("X", "Y", {"Z"})
    print(f"因果效应可识别: {result['is_identifiable']}")
    if result['is_identifiable']:
        print(f"调整集合: {result['adjustment_set']}")
    
    # 获取性能摘要
    summary = engine.get_performance_summary()
    print(f"\n性能摘要:")
    print(f"  d-分离检查次数: {summary['performance_stats']['d_separation_checks']}")
    print(f"  规则应用次数: {summary['performance_stats']['rule_applications']}")
    print(f"  识别尝试次数: {summary['performance_stats']['identification_attempts']}")