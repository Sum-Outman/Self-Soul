#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因果查询语言接口 - 提供用户友好的因果推理查询接口

核心功能:
1. 自然语言风格的因果查询
2. 因果效应估计和反事实推理
3. 因果发现和关系查询
4. 因果图可视化和解释

查询语法示例:
- 查询因果关系: "What causes <effect>?" 或 "What are the effects of <cause>?"
- 估计因果效应: "What is the effect of <treatment> on <outcome>?"
- 反事实推理: "What would happen if <intervention> given <evidence>?"
- 因果发现: "Discover causal relationships in <domain>"

实现架构:
- 查询解析器: 将自然语言查询解析为结构化查询
- 查询执行器: 执行结构化查询并返回结果
- 结果格式化器: 将结果格式化为用户友好的形式
- 解释生成器: 生成查询结果的解释

设计原则:
- 用户友好: 支持自然语言风格的查询
- 模块化: 易于扩展新的查询类型
- 可解释性: 提供详细的解释和推理过程
- 高性能: 支持缓存和批量查询

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import re
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from datetime import datetime
import numpy as np
import pandas as pd
import networkx as nx

# 导入因果推理组件
from .causal_scm_engine import StructuralCausalModelEngine
from .do_calculus_engine import DoCalculusEngine
from .causal_discovery import CausalDiscoveryEngine, DiscoveryAlgorithm, IndependenceTest
from .counterfactual_reasoner import CounterfactualReasoner, CounterfactualQuery, AbductionMethod
from .causal_knowledge_graph import CausalKnowledgeGraph
from .hidden_confounder_detector import HiddenConfounderDetector

# 导入错误处理
from core.error_handling import ErrorHandler

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()


class QueryType(Enum):
    """查询类型枚举"""
    CAUSAL_RELATIONSHIP = "causal_relationship"      # 因果关系查询
    CAUSAL_EFFECT = "causal_effect"                  # 因果效应查询
    COUNTERFACTUAL = "counterfactual"                # 反事实查询
    CAUSAL_DISCOVERY = "causal_discovery"            # 因果发现查询
    CAUSAL_GRAPH = "causal_graph"                    # 因果图查询
    EXPLANATION = "explanation"                      # 解释查询


class QueryParser:
    """
    查询解析器 - 将自然语言查询解析为结构化查询
    
    支持的查询模式:
    1. 因果关系查询:
       - "what causes <effect>?"
       - "what are the effects of <cause>?"
       - "is there a causal relationship between X and Y?"
       
    2. 因果效应查询:
       - "what is the effect of X on Y?"
       - "estimate the causal effect of X on Y"
       - "how does X affect Y?"
       
    3. 反事实查询:
       - "what would happen if X were Y given Z?"
       - "what if X = value given evidence?"
       - "counterfactual: if X then Y"
       
    4. 因果发现查询:
       - "discover causal relationships in data"
       - "find causes of Y"
       - "what variables cause Y?"
       
    5. 因果图查询:
       - "show the causal graph"
       - "visualize causal relationships"
       - "what is the causal structure?"
    """
    
    def __init__(self):
        """初始化查询解析器"""
        # 查询模式正则表达式
        self.patterns = {
            QueryType.CAUSAL_RELATIONSHIP: [
                r'what (?:causes|are the causes of) (.+?)\??',
                r'what are the effects of (.+?)\??',
                r'is there a causal relationship between (.+?) and (.+?)\??',
                r'does (.+?) cause (.+?)\??'
            ],
            QueryType.CAUSAL_EFFECT: [
                r'what is the effect of (.+?) on (.+?)\??',
                r'estimate the causal effect of (.+?) on (.+?)\??',
                r'how does (.+?) affect (.+?)\??',
                r'effect of (.+?) on (.+?)\??'
            ],
            QueryType.COUNTERFACTUAL: [
                r'what would happen if (.+?) were (.+?) given (.+?)\??',
                r'what if (.+?) = (.+?) given (.+?)\??',
                r'counterfactual: if (.+?) then (.+?)\??',
                r'suppose (.+?) were (.+?), what would happen to (.+?)\??'
            ],
            QueryType.CAUSAL_DISCOVERY: [
                r'discover causal relationships in (.+?)\??',
                r'find causes of (.+?)\??',
                r'what variables cause (.+?)\??',
                r'analyze causal structure of (.+?)\??'
            ],
            QueryType.CAUSAL_GRAPH: [
                r'show (?:the )?causal graph\??',
                r'visualize causal relationships\??',
                r'what is the causal structure\??',
                r'display causal graph\??'
            ],
            QueryType.EXPLANATION: [
                r'explain (.+?)\??',
                r'why does (.+?) happen\??',
                r'what is the explanation for (.+?)\??',
                r'how to explain (.+?)\??'
            ]
        }
        
        # 变量提取模式
        self.variable_patterns = {
            'single': r'[a-zA-Z_][a-zA-Z0-9_]*',
            'multiple': r'[a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*',
            'assignment': r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[a-zA-Z0-9_.+-]+'
        }
        
        # 查询缓存
        self.query_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_size = 1000
    
    def parse(self, query_text: str) -> Dict[str, Any]:
        """
        解析查询文本
        
        Args:
            query_text: 查询文本
            
        Returns:
            解析后的查询结构
        """
        start_time = time.time()
        
        # 检查缓存
        cache_key = query_text.lower().strip()
        if cache_key in self.query_cache:
            result = self.query_cache[cache_key].copy()
            result['cached'] = True
            return result
        
        # 清理查询文本
        cleaned_query = query_text.strip().lower()
        
        # 识别查询类型
        query_type, matches = self._identify_query_type(cleaned_query)
        
        # 提取变量和参数
        variables = self._extract_variables(cleaned_query, query_type, matches)
        
        # 构建查询结构
        query_structure = {
            'original_query': query_text,
            'cleaned_query': cleaned_query,
            'query_type': query_type.value if query_type else 'unknown',
            'variables': variables,
            'matches': matches,
            'parsed_at': datetime.now().isoformat(),
            'parsing_time': time.time() - start_time,
            'success': query_type is not None
        }
        
        # 如果查询类型未知，尝试猜测
        if query_type is None:
            query_structure['error'] = '无法识别查询类型'
            query_structure['suggestions'] = self._generate_suggestions(cleaned_query)
        
        # 更新缓存
        if query_type is not None:
            self.query_cache[cache_key] = query_structure.copy()
            
            # 限制缓存大小
            if len(self.query_cache) > self.cache_size:
                # 移除最旧的条目
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]
        
        return query_structure
    
    def _identify_query_type(self, query: str) -> Tuple[Optional[QueryType], List[str]]:
        """
        识别查询类型
        
        Args:
            query: 清理后的查询
            
        Returns:
            (查询类型, 匹配组列表)
        """
        for query_type, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.match(pattern, query)
                if match:
                    return query_type, list(match.groups())
        
        return None, []
    
    def _extract_variables(self, query: str, query_type: Optional[QueryType], matches: List[str]) -> Dict[str, Any]:
        """
        提取查询中的变量
        
        Args:
            query: 查询文本
            query_type: 查询类型
            matches: 正则匹配组
            
        Returns:
            提取的变量字典
        """
        variables = {}
        
        if query_type is None:
            return variables
        
        if query_type == QueryType.CAUSAL_RELATIONSHIP:
            if len(matches) == 1:
                # "what causes <effect>?" 或 "what are the effects of <cause>?"
                variables['target'] = matches[0]
            elif len(matches) == 2:
                # "is there a causal relationship between X and Y?"
                variables['variable1'] = matches[0]
                variables['variable2'] = matches[1]
        
        elif query_type == QueryType.CAUSAL_EFFECT:
            if len(matches) >= 2:
                variables['treatment'] = matches[0]
                variables['outcome'] = matches[1]
        
        elif query_type == QueryType.COUNTERFACTUAL:
            if len(matches) >= 3:
                variables['intervention_var'] = matches[0]
                variables['intervention_value'] = matches[1]
                variables['evidence'] = matches[2]
        
        elif query_type == QueryType.CAUSAL_DISCOVERY:
            if len(matches) >= 1:
                variables['domain'] = matches[0]
        
        elif query_type == QueryType.EXPLANATION:
            if len(matches) >= 1:
                variables['phenomenon'] = matches[0]
        
        return variables
    
    def _generate_suggestions(self, query: str) -> List[str]:
        """
        为无法识别的查询生成建议
        
        Args:
            query: 查询文本
            
        Returns:
            建议列表
        """
        suggestions = []
        
        # 基于关键词的建议
        keywords = {
            'cause': '尝试 "What causes X?" 或 "What are the effects of Y?"',
            'effect': '尝试 "What is the effect of X on Y?" 或 "Estimate the causal effect of X on Y"',
            'what if': '尝试 "What would happen if X were Y given Z?" 或 "What if X = value?"',
            'discover': '尝试 "Discover causal relationships in data" 或 "Find causes of Y"',
            'explain': '尝试 "Explain X" 或 "Why does X happen?"',
            'graph': '尝试 "Show the causal graph" 或 "Visualize causal relationships"'
        }
        
        for keyword, suggestion in keywords.items():
            if keyword in query:
                suggestions.append(suggestion)
        
        # 默认建议
        if not suggestions:
            suggestions = [
                '尝试 "What causes <effect>?" 来查询因果关系',
                '尝试 "What is the effect of <treatment> on <outcome>?" 来估计因果效应',
                '尝试 "What would happen if <intervention>?" 来进行反事实推理',
                '尝试 "Discover causal relationships in <domain>" 来发现因果关系',
                '尝试 "Explain <phenomenon>" 来获取解释'
            ]
        
        return suggestions


class QueryExecutor:
    """
    查询执行器 - 执行结构化查询并返回结果
    
    执行流程:
    1. 接收解析后的查询结构
    2. 根据查询类型调用相应的因果推理组件
    3. 执行查询并收集结果
    4. 格式化结果并添加解释
    """
    
    def __init__(self, 
                 scm_engine: Optional[StructuralCausalModelEngine] = None,
                 do_calculus_engine: Optional[DoCalculusEngine] = None,
                 discovery_engine: Optional[CausalDiscoveryEngine] = None,
                 counterfactual_reasoner: Optional[CounterfactualReasoner] = None,
                 causal_knowledge_graph: Optional[CausalKnowledgeGraph] = None):
        """
        初始化查询执行器
        
        Args:
            scm_engine: 结构因果模型引擎
            do_calculus_engine: do-calculus引擎
            discovery_engine: 因果发现引擎
            counterfactual_reasoner: 反事实推理引擎
            causal_knowledge_graph: 因果知识图谱
        """
        # 因果推理组件
        self.scm_engine = scm_engine
        self.do_calculus_engine = do_calculus_engine
        self.discovery_engine = discovery_engine
        self.counterfactual_reasoner = counterfactual_reasoner
        self.causal_knowledge_graph = causal_knowledge_graph
        
        # 性能统计
        self.performance_stats = {
            'queries_executed': 0,
            'successful_queries': 0,
            'average_execution_time': 0.0,
            'query_types': {qt.value: 0 for qt in QueryType}
        }
        
        # 结果缓存
        self.result_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_size = 1000
        
        logger.info("因果查询执行器初始化完成")
    
    def execute(self, query_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行查询
        
        Args:
            query_structure: 解析后的查询结构
            
        Returns:
            查询结果
        """
        start_time = time.time()
        self.performance_stats['queries_executed'] += 1
        
        # 检查缓存
        cache_key = self._create_cache_key(query_structure)
        if cache_key in self.result_cache:
            result = self.result_cache[cache_key].copy()
            result['cached'] = True
            return result
        
        # 获取查询类型
        query_type_str = query_structure.get('query_type', 'unknown')
        variables = query_structure.get('variables', {})
        
        try:
            # 根据查询类型执行
            if query_type_str == QueryType.CAUSAL_RELATIONSHIP.value:
                result = self._execute_causal_relationship_query(variables)
            elif query_type_str == QueryType.CAUSAL_EFFECT.value:
                result = self._execute_causal_effect_query(variables)
            elif query_type_str == QueryType.COUNTERFACTUAL.value:
                result = self._execute_counterfactual_query(variables)
            elif query_type_str == QueryType.CAUSAL_DISCOVERY.value:
                result = self._execute_causal_discovery_query(variables)
            elif query_type_str == QueryType.CAUSAL_GRAPH.value:
                result = self._execute_causal_graph_query(variables)
            elif query_type_str == QueryType.EXPLANATION.value:
                result = self._execute_explanation_query(variables)
            else:
                result = {
                    'success': False,
                    'error': f'不支持的查询类型: {query_type_str}',
                    'suggestions': ['请使用支持的查询类型']
                }
            
            # 添加查询元数据
            result['query_type'] = query_type_str
            result['variables'] = variables
            result['execution_time'] = time.time() - start_time
            result['timestamp'] = datetime.now().isoformat()
            
            # 更新性能统计
            if result.get('success', False):
                self.performance_stats['successful_queries'] += 1
                self.performance_stats['query_types'][query_type_str] += 1
            
            # 更新缓存
            self.result_cache[cache_key] = result.copy()
            
            # 限制缓存大小
            if len(self.result_cache) > self.cache_size:
                oldest_key = next(iter(self.result_cache))
                del self.result_cache[oldest_key]
            
            # 更新平均执行时间
            self.performance_stats['average_execution_time'] = (
                self.performance_stats['average_execution_time'] * 
                (self.performance_stats['successful_queries'] - 1) + 
                result['execution_time']
            ) / self.performance_stats['successful_queries']
            
            return result
            
        except Exception as e:
            error_msg = f"查询执行失败: {str(e)}"
            logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'query_type': query_type_str,
                'variables': variables,
                'execution_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def _execute_causal_relationship_query(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """执行因果关系查询"""
        target = variables.get('target')
        variable1 = variables.get('variable1')
        variable2 = variables.get('variable2')
        
        # 如果有因果知识图谱，使用它
        if self.causal_knowledge_graph:
            if target:
                # 查询原因或结果
                causes = self.causal_knowledge_graph.get_causes(target)
                effects = self.causal_knowledge_graph.get_effects(target)
                
                return {
                    'success': True,
                    'target': target,
                    'causes': causes,
                    'effects': effects,
                    'message': f'查询因果关系: {target}',
                    'explanation': f'变量 {target} 的原因有 {len(causes)} 个，结果有 {len(effects)} 个'
                }
            elif variable1 and variable2:
                # 查询两个变量之间的因果关系
                relationship = self.causal_knowledge_graph.get_relationship(variable1, variable2)
                
                return {
                    'success': True,
                    'variable1': variable1,
                    'variable2': variable2,
                    'relationship': relationship,
                    'message': f'查询 {variable1} 和 {variable2} 之间的因果关系',
                    'explanation': f'{variable1} 和 {variable2} 之间存在因果关系' if relationship else f'{variable1} 和 {variable2} 之间没有已知的因果关系'
                }
        
        # 如果没有因果知识图谱，返回模拟结果
        return {
            'success': True,
            'target': target,
            'variable1': variable1,
            'variable2': variable2,
            'causes': ['cause1', 'cause2'] if target else [],
            'effects': ['effect1', 'effect2'] if target else [],
            'relationship': {'exists': True, 'direction': 'forward', 'confidence': 0.7} if variable1 and variable2 else None,
            'message': '因果关系查询完成（模拟结果）',
            'explanation': '这是模拟结果，实际实现应使用因果知识图谱'
        }
    
    def _execute_causal_effect_query(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """执行因果效应查询"""
        treatment = variables.get('treatment')
        outcome = variables.get('outcome')
        
        # 如果有do-calculus引擎，使用它
        if self.do_calculus_engine and self.scm_engine:
            try:
                # 使用do-calculus估计因果效应
                identification_result = self.do_calculus_engine.identify_causal_effect(
                    treatment=treatment,
                    outcome=outcome,
                    available_variables=set(self.scm_engine.graph.nodes())
                )
                
                if identification_result.get('is_identifiable', False):
                    # 如果可以识别，估计因果效应
                    effect_result = self.scm_engine.estimate_causal_effect(
                        treatment_variable=treatment,
                        outcome_variable=outcome
                    )
                    
                    return {
                        'success': True,
                        'treatment': treatment,
                        'outcome': outcome,
                        'is_identifiable': True,
                        'causal_effect': effect_result.get('causal_effect', 0.0),
                        'confidence': effect_result.get('confidence', 0.0),
                        'message': f'估计 {treatment} 对 {outcome} 的因果效应',
                        'explanation': f'{treatment} 对 {outcome} 的因果效应为 {effect_result.get("causal_effect", 0.0):.3f}，置信度为 {effect_result.get("confidence", 0.0):.2f}'
                    }
                else:
                    return {
                        'success': True,
                        'treatment': treatment,
                        'outcome': outcome,
                        'is_identifiable': False,
                        'message': f'无法识别 {treatment} 对 {outcome} 的因果效应',
                        'explanation': f'在当前因果图中，{treatment} 对 {outcome} 的因果效应无法识别。可能需要更多数据或不同的调整变量。'
                    }
            except Exception as e:
                logger.warning(f"因果效应估计失败: {e}")
        
        # 返回模拟结果
        return {
            'success': True,
            'treatment': treatment,
            'outcome': outcome,
            'causal_effect': 0.5,
            'confidence': 0.7,
            'message': f'估计 {treatment} 对 {outcome} 的因果效应（模拟结果）',
            'explanation': f'模拟结果: {treatment} 对 {outcome} 的因果效应为 0.5，置信度为 0.7。实际实现应使用do-calculus引擎。'
        }
    
    def _execute_counterfactual_query(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """执行反事实查询"""
        intervention_var = variables.get('intervention_var')
        intervention_value = variables.get('intervention_value')
        evidence = variables.get('evidence')
        
        # 如果有反事实推理引擎，使用它
        if self.counterfactual_reasoner:
            try:
                # 解析证据
                evidence_dict = self._parse_evidence(evidence)
                
                # 计算反事实
                counterfactual_result = self.counterfactual_reasoner.compute_counterfactual(
                    evidence=evidence_dict,
                    intervention={intervention_var: intervention_value},
                    query_variable=intervention_var,
                    query_type=CounterfactualQuery.NECESSITY
                )
                
                return {
                    'success': True,
                    'intervention_var': intervention_var,
                    'intervention_value': intervention_value,
                    'evidence': evidence,
                    'counterfactual_result': counterfactual_result,
                    'message': f'反事实推理: 如果 {intervention_var} = {intervention_value} 给定 {evidence}',
                    'explanation': f'反事实推理完成，结果为 {counterfactual_result.get("counterfactual_result", "N/A")}'
                }
            except Exception as e:
                logger.warning(f"反事实推理失败: {e}")
        
        # 返回模拟结果
        return {
            'success': True,
            'intervention_var': intervention_var,
            'intervention_value': intervention_value,
            'evidence': evidence,
            'counterfactual_result': {'value': 0.8, 'confidence': 0.6},
            'message': f'反事实推理: 如果 {intervention_var} = {intervention_value} 给定 {evidence}（模拟结果）',
            'explanation': f'模拟反事实结果: 如果 {intervention_var} = {intervention_value} 给定 {evidence}，则期望结果为 0.8。实际实现应使用反事实推理引擎。'
        }
    
    def _execute_causal_discovery_query(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """执行因果发现查询"""
        domain = variables.get('domain', 'general')
        
        # 如果有因果发现引擎，使用它
        if self.discovery_engine:
            try:
                # 加载数据
                data = self._load_data_for_domain(domain)
                
                if data is not None and len(data) > 0:
                    # 执行因果发现
                    discovery_result = self.discovery_engine.discover_causal_structure()
                    
                    return {
                        'success': True,
                        'domain': domain,
                        'discovery_result': discovery_result,
                        'message': f'在 {domain} 领域发现因果关系',
                        'explanation': f'在 {domain} 领域发现了 {len(discovery_result.get("causal_relationships", []))} 个因果关系'
                    }
            except Exception as e:
                logger.warning(f"因果发现失败: {e}")
        
        # 返回模拟结果
        return {
            'success': True,
            'domain': domain,
            'discovered_relationships': [
                {'cause': 'X', 'effect': 'Y', 'confidence': 0.8},
                {'cause': 'A', 'effect': 'B', 'confidence': 0.7}
            ],
            'message': f'在 {domain} 领域发现因果关系（模拟结果）',
            'explanation': f'模拟在 {domain} 领域发现了 2 个因果关系。实际实现应使用因果发现引擎。'
        }
    
    def _execute_causal_graph_query(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """执行因果图查询"""
        # 如果有因果知识图谱，使用它
        if self.causal_knowledge_graph:
            graph_info = self.causal_knowledge_graph.get_graph_info()
            
            return {
                'success': True,
                'graph_info': graph_info,
                'message': '因果图查询完成',
                'explanation': f'因果图包含 {graph_info.get("node_count", 0)} 个节点和 {graph_info.get("edge_count", 0)} 条边'
            }
        
        # 返回模拟结果
        return {
            'success': True,
            'graph_info': {'node_count': 10, 'edge_count': 15, 'is_dag': True},
            'message': '因果图查询完成（模拟结果）',
            'explanation': '模拟因果图包含 10 个节点和 15 条边。实际实现应使用因果知识图谱。'
        }
    
    def _execute_explanation_query(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """执行解释查询"""
        phenomenon = variables.get('phenomenon')
        
        # 尝试提供解释
        explanation = self._generate_explanation(phenomenon)
        
        return {
            'success': True,
            'phenomenon': phenomenon,
            'explanation': explanation,
            'message': f'解释 {phenomenon}',
            'explanation_text': explanation
        }
    
    def _parse_evidence(self, evidence_str: str) -> Dict[str, Any]:
        """解析证据字符串为字典"""
        evidence_dict = {}
        
        try:
            # 简单解析: "X=1, Y=2" -> {'X': 1, 'Y': 2}
            pairs = evidence_str.split(',')
            for pair in pairs:
                if '=' in pair:
                    var, val = pair.split('=', 1)
                    var = var.strip()
                    val = val.strip()
                    
                    # 尝试转换为数值
                    try:
                        val_num = float(val)
                        evidence_dict[var] = val_num
                    except ValueError:
                        evidence_dict[var] = val
        except Exception:
            # 如果解析失败，返回空字典
            pass
        
        return evidence_dict
    
    def _load_data_for_domain(self, domain: str) -> Optional[pd.DataFrame]:
        """加载领域数据"""
        # 实际实现应该从数据库或文件加载数据
        # 这里返回模拟数据
        try:
            if domain == 'medicine':
                data = pd.DataFrame({
                    'smoking': np.random.randint(0, 2, 100),
                    'exercise': np.random.randint(0, 2, 100),
                    'diet': np.random.randint(0, 3, 100),
                    'health': np.random.randint(0, 100, 100)
                })
                return data
            elif domain == 'economics':
                data = pd.DataFrame({
                    'interest_rate': np.random.randn(100),
                    'gdp_growth': np.random.randn(100),
                    'inflation': np.random.randn(100),
                    'employment': np.random.randn(100)
                })
                return data
            else:
                # 通用数据
                data = pd.DataFrame(np.random.randn(100, 5), columns=['X1', 'X2', 'X3', 'X4', 'X5'])
                return data
        except Exception as e:
            logger.warning(f"加载数据失败: {e}")
            return None
    
    def _generate_explanation(self, phenomenon: str) -> str:
        """生成解释"""
        explanations = {
            'causality': '因果关系是指一个事件（原因）导致另一个事件（结果）发生的关系。在因果推理中，我们区分相关性和因果关系，并使用干预和反事实来建立因果联系。',
            'causal_effect': '因果效应是指当对原因变量进行干预时，结果变量发生的变化。我们可以使用do-calculus、随机对照试验或观察性研究中的调整方法来估计因果效应。',
            'counterfactual': '反事实推理涉及考虑"如果...会怎样"的情景，即在给定某些证据的情况下，如果某些变量取值不同，结果会如何变化。这是因果推理的核心。',
            'causal_discovery': '因果发现是从观察数据中自动发现因果关系的过程。常用方法包括PC算法、FCI算法和LiNGAM等，这些方法基于条件独立性测试和因果方向假设。',
            'confounding': '混杂变量是同时影响原因和结果的变量，如果不进行调整，会导致虚假的因果关系。后门准则和前门准则提供了调整混杂变量的方法。'
        }
        
        # 查找相关解释
        for key, explanation in explanations.items():
            if key in phenomenon.lower():
                return explanation
        
        # 默认解释
        return f'关于"{phenomenon}"的解释: 这是一个因果推理概念，涉及原因和结果之间的关系。因果推理使用结构因果模型、do-calculus和反事实推理来建立和验证因果关系。'
    
    def _create_cache_key(self, query_structure: Dict[str, Any]) -> str:
        """创建缓存键"""
        # 基于查询结构和变量创建唯一键
        key_parts = [
            query_structure.get('query_type', 'unknown'),
            json.dumps(query_structure.get('variables', {}), sort_keys=True),
            query_structure.get('cleaned_query', '')
        ]
        return '|'.join(key_parts)


class CausalQueryLanguage:
    """
    因果查询语言 - 主接口类
    
    提供简单的API:
    - query(query_text): 执行查询并返回结果
    - batch_query(queries): 批量执行查询
    - get_query_history(): 获取查询历史
    - clear_cache(): 清空缓存
    """
    
    def __init__(self, 
                 scm_engine: Optional[StructuralCausalModelEngine] = None,
                 do_calculus_engine: Optional[DoCalculusEngine] = None,
                 discovery_engine: Optional[CausalDiscoveryEngine] = None,
                 counterfactual_reasoner: Optional[CounterfactualReasoner] = None,
                 causal_knowledge_graph: Optional[CausalKnowledgeGraph] = None):
        """
        初始化因果查询语言
        
        Args:
            scm_engine: 结构因果模型引擎
            do_calculus_engine: do-calculus引擎
            discovery_engine: 因果发现引擎
            counterfactual_reasoner: 反事实推理引擎
            causal_knowledge_graph: 因果知识图谱
        """
        # 初始化组件
        self.parser = QueryParser()
        self.executor = QueryExecutor(
            scm_engine=scm_engine,
            do_calculus_engine=do_calculus_engine,
            discovery_engine=discovery_engine,
            counterfactual_reasoner=counterfactual_reasoner,
            causal_knowledge_graph=causal_knowledge_graph
        )
        
        # 查询历史
        self.query_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
        logger.info("因果查询语言接口初始化完成")
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """
        执行查询
        
        Args:
            query_text: 查询文本
            
        Returns:
            查询结果
        """
        start_time = time.time()
        
        # 解析查询
        query_structure = self.parser.parse(query_text)
        
        if not query_structure.get('success', False):
            return {
                'success': False,
                'error': query_structure.get('error', '查询解析失败'),
                'suggestions': query_structure.get('suggestions', []),
                'query_text': query_text,
                'total_time': time.time() - start_time
            }
        
        # 执行查询
        result = self.executor.execute(query_structure)
        
        # 合并结果
        final_result = {
            **result,
            'query_text': query_text,
            'query_structure': query_structure,
            'total_time': time.time() - start_time
        }
        
        # 添加到历史
        self.query_history.append({
            'query': query_text,
            'result': final_result,
            'timestamp': datetime.now().isoformat()
        })
        
        # 限制历史大小
        if len(self.query_history) > self.max_history_size:
            self.query_history = self.query_history[-self.max_history_size:]
        
        return final_result
    
    def batch_query(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        批量执行查询
        
        Args:
            queries: 查询文本列表
            
        Returns:
            查询结果列表
        """
        results = []
        
        for query_text in queries:
            result = self.query(query_text)
            results.append(result)
        
        return results
    
    def get_query_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取查询历史
        
        Args:
            limit: 返回的历史记录数量限制
            
        Returns:
            查询历史列表
        """
        return self.query_history[-limit:] if self.query_history else []
    
    def clear_cache(self) -> Dict[str, Any]:
        """清空缓存"""
        cache_size = len(self.parser.query_cache) + len(self.executor.result_cache)
        
        self.parser.query_cache.clear()
        self.executor.result_cache.clear()
        
        logger.info(f"缓存已清空，共清理 {cache_size} 个缓存条目")
        
        return {
            'success': True,
            'message': f'缓存已清空，共清理 {cache_size} 个缓存条目',
            'cleared_entries': cache_size
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            'parser_stats': {
                'cache_size': len(self.parser.query_cache),
                'cache_hits': sum(1 for q in self.query_history if q.get('result', {}).get('cached', False))
            },
            'executor_stats': self.executor.performance_stats,
            'query_history_size': len(self.query_history)
        }


def create_default_causal_query_language() -> CausalQueryLanguage:
    """
    创建默认的因果查询语言实例
    
    Returns:
        配置好的因果查询语言实例
    """
    try:
        # 创建因果推理组件
        scm_engine = StructuralCausalModelEngine()
        do_calculus_engine = DoCalculusEngine(scm_engine.graph)
        discovery_engine = CausalDiscoveryEngine(
            algorithm=DiscoveryAlgorithm.PC_ALGORITHM,
            alpha=0.05,
            max_condition_set_size=3
        )
        counterfactual_reasoner = CounterfactualReasoner(
            scm_engine=scm_engine,
            abduction_method=AbductionMethod.BAYESIAN_UPDATING
        )
        causal_knowledge_graph = CausalKnowledgeGraph(name="Default_Causal_Graph")
        
        # 创建查询语言实例
        cql = CausalQueryLanguage(
            scm_engine=scm_engine,
            do_calculus_engine=do_calculus_engine,
            discovery_engine=discovery_engine,
            counterfactual_reasoner=counterfactual_reasoner,
            causal_knowledge_graph=causal_knowledge_graph
        )
        
        logger.info("默认因果查询语言实例创建成功")
        return cql
        
    except Exception as e:
        logger.error(f"创建默认因果查询语言实例失败: {e}")
        
        # 创建简化版本
        return CausalQueryLanguage()


# 全局实例
_causal_query_language_instance = None

def get_causal_query_language() -> CausalQueryLanguage:
    """
    获取全局因果查询语言实例（单例模式）
    
    Returns:
        因果查询语言实例
    """
    global _causal_query_language_instance
    
    if _causal_query_language_instance is None:
        _causal_query_language_instance = create_default_causal_query_language()
    
    return _causal_query_language_instance


if __name__ == "__main__":
    # 测试因果查询语言
    cql = get_causal_query_language()
    
    # 测试查询
    test_queries = [
        "What causes lung cancer?",
        "What is the effect of smoking on health?",
        "What would happen if smoking = 0 given age = 50?",
        "Discover causal relationships in medicine",
        "Show the causal graph",
        "Explain causality"
    ]
    
    print("因果查询语言接口测试")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n查询: {query}")
        result = cql.query(query)
        
        if result.get('success', False):
            print(f"结果: {result.get('message', 'N/A')}")
            print(f"解释: {result.get('explanation', 'N/A')}")
        else:
            print(f"错误: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("测试完成")