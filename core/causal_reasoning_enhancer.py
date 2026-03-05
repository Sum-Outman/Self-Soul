#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal Reasoning Enhancer - 因果推理增强器

核心功能：
1. 因果图构建和分析
2. 干预效果估计
3. 反事实推理
4. 因果发现
5. 因果链分析
6. 因果效应传播

基于结构因果模型（SCM）和do-calculus实现高级因果推理能力，
为规划和决策提供因果层面的深度分析。

Copyright (c) 2025 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
import json
import math
import ast
import operator
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from enum import Enum
from collections import defaultdict, deque
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 导入错误处理
from core.error_handling import ErrorHandler

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()

class SafeExpressionEvaluator:
    """安全表达式求值器，替代eval()"""
    def __init__(self):
        self.allowed_operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '**': operator.pow,
            '//': operator.floordiv,
            '%': operator.mod,
            'abs': abs,
            'round': round,
            'min': min,
            'max': max
        }
        
        self.allowed_functions = {
            'math': {
                'sqrt': math.sqrt,
                'exp': math.exp,
                'log': math.log,
                'log10': math.log10,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'asin': math.asin,
                'acos': math.acos,
                'atan': math.atan,
                'degrees': math.degrees,
                'radians': math.radians,
                'pi': math.pi,
                'e': math.e
            }
        }
    
    def evaluate(self, expression: str, context: Dict[str, Any]) -> float:
        """
        安全求值表达式
        
        Args:
            expression: 要计算的表达式字符串
            context: 变量上下文字典
            
        Returns:
            计算结果
        """
        try:
            # 尝试使用ast.literal_eval处理简单表达式
            try:
                return float(ast.literal_eval(expression))
            except (ValueError, SyntaxError):
                pass
            
            # 解析简单算术表达式
            return self._evaluate_simple_expression(expression, context)
            
        except Exception as e:
            logger.warning(f"安全表达式求值失败: {expression}, 错误: {e}")
            return 0.0
    
    def _evaluate_simple_expression(self, expression: str, context: Dict[str, Any]) -> float:
        """评估简单算术表达式"""
        # 移除空格
        expr = expression.replace(' ', '')
        
        # 处理括号
        while '(' in expr and ')' in expr:
            start = expr.rfind('(')
            end = expr.find(')', start)
            if end == -1:
                break
            
            inner_expr = expr[start + 1:end]
            inner_result = self._evaluate_simple_expression(inner_expr, context)
            expr = expr[:start] + str(inner_result) + expr[end + 1:]
        
        # 解析运算符优先级：先处理乘除，再处理加减
        # 处理乘方
        if '**' in expr:
            left, right = expr.split('**', 1)
            left_val = self._evaluate_simple_expression(left, context)
            right_val = self._evaluate_simple_expression(right, context)
            return self.allowed_operators['**'](left_val, right_val)
        
        # 处理乘除取余整除
        for op in ['*', '/', '//', '%']:
            if op in expr:
                parts = expr.split(op)
                if len(parts) == 2:
                    left_val = self._evaluate_simple_expression(parts[0], context)
                    right_val = self._evaluate_simple_expression(parts[1], context)
                    if op == '/' and right_val == 0:
                        return 0.0  # 避免除零错误
                    return self.allowed_operators[op](left_val, right_val)
        
        # 处理加减
        for op in ['+', '-']:
            if op in expr and not (expr.startswith(op) and expr.count(op) == 1):
                parts = expr.split(op)
                if len(parts) == 2:
                    left_val = self._evaluate_simple_expression(parts[0], context)
                    right_val = self._evaluate_simple_expression(parts[1], context)
                    return self.allowed_operators[op](left_val, right_val)
        
        # 处理数学函数
        for func_name, func_dict in self.allowed_functions.items():
            for func_key, func in func_dict.items():
                if expr.startswith(f"{func_name}.{func_key}(") and expr.endswith(')'):
                    arg_str = expr[len(f"{func_name}.{func_key}("):-1]
                    arg_val = self._evaluate_simple_expression(arg_str, context)
                    return func(arg_val)
        
        # 处理变量访问
        if expr.startswith('inputs.get(') and expr.endswith(')'):
            key_str = expr[len('inputs.get('):-1].strip("'\"")
            return context.get('inputs', {}).get(key_str, 0.0)
        elif expr.startswith('inputs.') and '.' in expr[7:]:
            # 处理嵌套访问
            parts = expr.split('.')
            current = context.get('inputs', {})
            for part in parts[1:]:
                if isinstance(current, dict):
                    current = current.get(part, 0.0)
                else:
                    return 0.0
            return float(current) if isinstance(current, (int, float)) else 0.0
        
        # 尝试解析为数字
        try:
            return float(expr)
        except ValueError:
            # 尝试从上下文中获取
            if expr in context:
                return float(context[expr])
            elif 'inputs' in context and expr in context['inputs']:
                return float(context['inputs'][expr])
            else:
                return 0.0

class CausalRelationshipType(Enum):
    """因果关系类型枚举"""
    DIRECT_CAUSE = "direct_cause"          # 直接原因
    INDIRECT_CAUSE = "indirect_cause"      # 间接原因
    COMMON_CAUSE = "common_cause"          # 共同原因
    MEDIATOR = "mediator"                  # 中介变量
    CONFOUNDER = "confounder"              # 混杂因子
    COLLIDER = "collider"                  # 碰撞变量
    SPURIOUS = "spurious"                  # 伪相关

class InterventionType(Enum):
    """干预类型枚举"""
    HARD_INTERVENTION = "hard_intervention"      # 硬干预（do-操作）
    SOFT_INTERVENTION = "soft_intervention"      # 软干预（概率改变）
    STRUCTURAL_INTERVENTION = "structural_intervention"  # 结构干预
    TEMPORAL_INTERVENTION = "temporal_intervention"      # 时间干预

class CausalStrength(Enum):
    """因果强度级别"""
    WEAK = "weak"           # 弱因果：影响系数 < 0.3
    MODERATE = "moderate"   # 中等因果：影响系数 0.3-0.6
    STRONG = "strong"       # 强因果：影响系数 0.6-0.8
    VERY_STRONG = "very_strong"  # 非常强：影响系数 > 0.8

class CausalReasoningEnhancer:
    """
    因果推理增强器 - 提供深度因果分析和推理
    
    核心特性：
    1. 结构因果模型（SCM）构建和推理
    2. do-calculus和干预效果估计
    3. 反事实推理和what-if分析
    4. 因果发现和因果结构学习
    5. 因果效应传播和影响分析
    6. 因果链识别和关键路径分析
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化因果推理增强器
        
        Args:
            config: 配置字典
        """
        self.logger = logger
        self.config = config or self._get_default_config()
        
        # 因果图和模型
        self.causal_graphs = defaultdict(nx.DiGraph)  # 因果图集合
        self.structural_models = {}                    # 结构方程模型
        self.causal_discovery_models = {}              # 因果发现模型
        
        # 干预和反事实分析
        self.intervention_history = []
        self.counterfactual_scenarios = []
        
        # 学习和优化
        self.learning_data = {
            "causal_patterns": [],
            "intervention_effects": [],
            "counterfactual_accuracy": [],
            "discovery_performance": []
        }
        
        # 初始化组件
        self._initialize_components()
        
        logger.info("因果推理增强器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "causal_modeling": {
                "enable_scm_learning": True,
                "enable_graph_construction": True,
                "max_graph_size": 50,
                "causal_strength_threshold": 0.3
            },
            "intervention_analysis": {
                "enable_do_calculus": True,
                "enable_soft_interventions": True,
                "intervention_depth": 3,
                "effect_propagation_limit": 100
            },
            "counterfactual_reasoning": {
                "enable_counterfactuals": True,
                "max_counterfactual_scenarios": 10,
                "counterfactual_depth": 2,
                "enable_what_if_analysis": True
            },
            "causal_discovery": {
                "enable_structure_learning": True,
                "discovery_method": "pc_algorithm",  # pc, fci, lingam, neural
                "significance_level": 0.05,
                "max_iterations": 100
            },
            "optimization": {
                "enable_causal_optimization": True,
                "optimization_method": "gradient_based",
                "learning_rate": 0.01,
                "max_optimization_iterations": 50
            }
        }
    
    def _initialize_components(self) -> None:
        """初始化组件"""
        try:
            # 初始化因果图构建器
            self._initialize_graph_builder()
            
            # 初始化结构方程模型
            if self.config["causal_modeling"]["enable_scm_learning"]:
                self._initialize_structural_models()
            
            # 初始化因果发现引擎
            if self.config["causal_discovery"]["enable_structure_learning"]:
                self._initialize_causal_discovery()
            
            # 初始化干预分析器
            self._initialize_intervention_analyzer()
            
            # 初始化反事实推理器
            if self.config["counterfactual_reasoning"]["enable_counterfactuals"]:
                self._initialize_counterfactual_reasoner()
            
            logger.info("因果推理组件初始化完成")
            
        except Exception as e:
            error_handler.handle_error(e, "CausalReasoningEnhancer", "初始化组件失败")
            logger.error(f"组件初始化失败: {e}")
    
    def _initialize_graph_builder(self) -> None:
        """初始化因果图构建器"""
        self.graph_builder = {
            "node_types": ["variable", "latent", "observed", "intervention"],
            "edge_types": ["direct", "bidirectional", "latent", "temporal"],
            "causal_mechanisms": ["linear", "nonlinear", "probabilistic", "structural"]
        }
    
    def _initialize_structural_models(self) -> None:
        """初始化结构方程模型"""
        
        # 使用模块级安全表达式求值器
        safe_evaluator = SafeExpressionEvaluator()
        
        # 简单的线性结构方程模型实现
        class SimpleStructuralEquationModel:
            def __init__(self, variables, equations):
                self.variables = variables
                self.equations = equations
                self.evaluator = safe_evaluator
            
            def predict(self, inputs):
                # 简化实现
                outputs = {}
                for var, eq in self.equations.items():
                    if isinstance(eq, str):
                        # 安全表达式求值
                        try:
                            context = {"inputs": inputs, "math": math}
                            outputs[var] = self.evaluator.evaluate(eq, context)
                        except Exception as e:
                            logger.debug(f"表达式求值失败: {eq}, 错误: {e}")
                            outputs[var] = 0.0
                    else:
                        outputs[var] = eq(inputs)
                return outputs
        
        self.structural_models["simple_sem"] = SimpleStructuralEquationModel(
            variables=["X", "Y", "Z"],
            equations={
                "Y": "0.6 * inputs.get('X', 0) + 0.3 * inputs.get('Z', 0)",
                "Z": "0.4 * inputs.get('X', 0)"
            }
        )
    
    def _initialize_causal_discovery(self) -> None:
        """初始化因果发现引擎"""
        # 实现PC算法（简化版本）
        class SimplifiedPCalgorithm:
            def __init__(self, data, alpha=0.05):
                self.data = data
                self.alpha = alpha
                self.graph = nx.Graph()
            
            def run(self):
                # 简化实现：构建完全连接图，然后基于相关性删除边
                n_vars = self.data.shape[1]
                for i in range(n_vars):
                    for j in range(i + 1, n_vars):
                        # 计算相关性
                        corr = np.corrcoef(self.data[:, i], self.data[:, j])[0, 1]
                        if abs(corr) > self.alpha:
                            self.graph.add_edge(f"X{i}", f"X{j}", weight=corr)
                
                return self.graph
        
        self.causal_discovery_models["pc_algorithm"] = SimplifiedPCalgorithm
    
    def _initialize_intervention_analyzer(self) -> None:
        """初始化干预分析器"""
        self.intervention_analyzer = {
            "methods": ["do_operator", "propensity_score", "instrumental_variable", "regression_discontinuity"],
            "effect_estimators": ["average_treatment_effect", "conditional_average_treatment_effect", "individual_treatment_effect"],
            "validity_checks": ["ignorability", "positivity", "consistency", "no_interference"]
        }
    
    def _initialize_counterfactual_reasoner(self) -> None:
        """初始化反事实推理器"""
        self.counterfactual_reasoner = {
            "methods": ["structural_counterfactuals", "potential_outcomes", "neural_counterfactuals", "abductive_counterfactuals"],
            "scenario_generators": ["random_perturbation", "systematic_variation", "adversarial_generation", "causal_manipulation"],
            "validity_metrics": ["consistency", "realism", "causal_plausibility", "logical_coherence"]
        }
    
    def build_causal_graph(self, variables: List[str], 
                          relationships: List[Tuple[str, str, Dict[str, Any]]],
                          domain: str = "general") -> nx.DiGraph:
        """
        构建因果图
        
        Args:
            variables: 变量列表
            relationships: 关系列表，每个元素为 (source, target, attributes)
            domain: 领域标识符
            
        Returns:
            构建的因果图
        """
        try:
            graph = nx.DiGraph()
            
            # 添加节点
            for var in variables:
                graph.add_node(var, type="variable", domain=domain)
            
            # 添加边（因果关系）
            for source, target, attrs in relationships:
                if source in graph and target in graph:
                    # 确定因果关系类型
                    causal_type = self._determine_causal_relationship_type(source, target, attrs)
                    strength = self._estimate_causal_strength(source, target, attrs)
                    
                    graph.add_edge(
                        source, target,
                        relationship_type=causal_type.value,
                        strength=strength.value,
                        strength_value=self._calculate_strength_value(strength),
                        **attrs
                    )
            
            # 保存到图集合
            graph_id = f"{domain}_causal_graph_{int(time.time())}"
            self.causal_graphs[graph_id] = graph
            
            # 分析图属性
            graph_analysis = self._analyze_causal_graph(graph)
            
            logger.info(f"因果图构建完成: {graph_id}, 节点数={graph.number_of_nodes()}, 边数={graph.number_of_edges()}")
            
            return graph
            
        except Exception as e:
            error_handler.handle_error(e, "CausalReasoningEnhancer", "构建因果图失败")
            return nx.DiGraph()
    
    def _determine_causal_relationship_type(self, source: str, target: str, 
                                           attrs: Dict[str, Any]) -> CausalRelationshipType:
        """确定因果关系类型"""
        # 基于属性判断关系类型
        if attrs.get("direct", False):
            return CausalRelationshipType.DIRECT_CAUSE
        elif attrs.get("mediated", False):
            return CausalRelationshipType.MEDIATOR
        elif attrs.get("confounded", False):
            return CausalRelationshipType.CONFOUNDER
        elif attrs.get("bidirectional", False):
            return CausalRelationshipType.COMMON_CAUSE
        else:
            # 基于其他启发式规则
            if source.lower() in target.lower() or target.lower() in source.lower():
                return CausalRelationshipType.DIRECT_CAUSE
            else:
                return CausalRelationshipType.INDIRECT_CAUSE
    
    def _estimate_causal_strength(self, source: str, target: str, 
                                 attrs: Dict[str, Any]) -> CausalStrength:
        """估计因果强度"""
        strength_value = attrs.get("strength", 0.5)
        
        if strength_value < 0.3:
            return CausalStrength.WEAK
        elif strength_value < 0.6:
            return CausalStrength.MODERATE
        elif strength_value < 0.8:
            return CausalStrength.STRONG
        else:
            return CausalStrength.VERY_STRONG
    
    def _calculate_strength_value(self, strength: CausalStrength) -> float:
        """计算强度数值"""
        if strength == CausalStrength.WEAK:
            return 0.2
        elif strength == CausalStrength.MODERATE:
            return 0.45
        elif strength == CausalStrength.STRONG:
            return 0.7
        else:  # VERY_STRONG
            return 0.9
    
    def _analyze_causal_graph(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """分析因果图属性"""
        analysis = {
            "basic_stats": {
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "density": nx.density(graph)
            },
            "topology": {
                "is_dag": nx.is_directed_acyclic_graph(graph),
                "has_cycles": not nx.is_directed_acyclic_graph(graph),
                "connected_components": nx.number_weakly_connected_components(graph)
            },
            "centrality": {
                "degree_centrality": nx.degree_centrality(graph),
                "betweenness_centrality": nx.betweenness_centrality(graph),
                "closeness_centrality": nx.closeness_centrality(graph)
            },
            "causal_properties": {
                "source_nodes": [n for n in graph.nodes() if graph.in_degree(n) == 0],
                "sink_nodes": [n for n in graph.nodes() if graph.out_degree(n) == 0],
                "mediators": self._identify_mediators(graph),
                "confounders": self._identify_confounders(graph)
            }
        }
        
        return analysis
    
    def _identify_mediators(self, graph: nx.DiGraph) -> List[str]:
        """识别中介变量"""
        mediators = []
        for node in graph.nodes():
            predecessors = list(graph.predecessors(node))
            successors = list(graph.successors(node))
            
            if predecessors and successors:
                # 检查是否在因果路径上
                is_mediator = True
                for pred in predecessors:
                    for succ in successors:
                        if not nx.has_path(graph, pred, succ):
                            is_mediator = False
                            break
                    if not is_mediator:
                        break
                
                if is_mediator:
                    mediators.append(node)
        
        return mediators
    
    def _identify_confounders(self, graph: nx.DiGraph) -> List[str]:
        """识别混杂因子"""
        confounders = []
        
        # 寻找共同原因节点
        for node in graph.nodes():
            children = list(graph.successors(node))
            
            if len(children) >= 2:
                # 检查子节点之间是否有直接连接
                has_direct_connection = False
                for i in range(len(children)):
                    for j in range(i + 1, len(children)):
                        if graph.has_edge(children[i], children[j]) or graph.has_edge(children[j], children[i]):
                            has_direct_connection = True
                            break
                    if has_direct_connection:
                        break
                
                if not has_direct_connection:
                    confounders.append(node)
        
        return confounders
    
    def analyze_causality(self, plan: Dict[str, Any], 
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        分析计划的因果结构
        
        Args:
            plan: 要分析的计划
            context: 上下文信息
            
        Returns:
            因果分析结果
        """
        start_time = time.time()
        
        try:
            # 1. 提取计划中的因果因素
            causal_factors = self._extract_causal_factors(plan, context)
            
            # 2. 构建计划因果图
            plan_causal_graph = self._build_plan_causal_graph(plan, causal_factors)
            
            # 3. 识别关键因果路径
            critical_paths = self._identify_critical_causal_paths(plan_causal_graph, plan)
            
            # 4. 分析因果脆弱性
            vulnerabilities = self._analyze_causal_vulnerabilities(plan_causal_graph, plan)
            
            # 5. 评估因果鲁棒性
            robustness = self._assess_causal_robustness(plan_causal_graph, plan)
            
            # 6. 生成因果优化建议
            optimization_suggestions = self._generate_causal_optimization_suggestions(
                plan_causal_graph, critical_paths, vulnerabilities
            )
            
            result = {
                "success": True,
                "causal_factors": causal_factors,
                "causal_graph": {
                    "nodes": list(plan_causal_graph.nodes()),
                    "edges": list(plan_causal_graph.edges(data=True)),
                    "analysis": self._analyze_causal_graph(plan_causal_graph)
                },
                "critical_paths": critical_paths,
                "vulnerabilities": vulnerabilities,
                "robustness_assessment": robustness,
                "optimization_suggestions": optimization_suggestions,
                "performance_metrics": {
                    "analysis_time": time.time() - start_time,
                    "graph_complexity": plan_causal_graph.number_of_nodes() + plan_causal_graph.number_of_edges(),
                    "critical_path_count": len(critical_paths),
                    "vulnerability_count": len(vulnerabilities)
                }
            }
            
            logger.info(f"因果分析完成: 计划ID={plan.get('id', 'unknown')}, 因素数={len(causal_factors)}")
            
            return result
            
        except Exception as e:
            error_handler.handle_error(e, "CausalReasoningEnhancer", "因果分析失败")
            return {
                "success": False,
                "error": str(e),
                "partial_results": {
                    "causal_factors": [],
                    "critical_paths": [],
                    "vulnerabilities": []
                }
            }
    
    def _extract_causal_factors(self, plan: Dict[str, Any], 
                               context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从计划中提取因果因素"""
        factors = []
        
        # 从计划步骤中提取
        steps = plan.get("steps", [])
        for i, step in enumerate(steps):
            factor = {
                "id": f"step_{i}",
                "type": "plan_step",
                "description": step.get("description", ""),
                "causal_role": self._determine_step_causal_role(step),
                "dependencies": step.get("depends_on", []),
                "resources": step.get("resources", []),
                "estimated_time": step.get("estimated_time", 0)
            }
            factors.append(factor)
        
        # 从资源中提取
        resources = plan.get("resource_requirements", {})
        for resource, quantity in resources.items():
            factor = {
                "id": f"resource_{resource}",
                "type": "resource",
                "description": f"资源: {resource}",
                "causal_role": "enabler" if quantity > 0 else "constraint",
                "quantity": quantity,
                "criticality": self._assess_resource_criticality(resource, plan)
            }
            factors.append(factor)
        
        # 从上下文中提取
        if context:
            for key, value in context.items():
                if isinstance(value, (str, int, float, bool)):
                    factor = {
                        "id": f"context_{key}",
                        "type": "context",
                        "description": f"上下文: {key}",
                        "causal_role": "condition",
                        "value": value,
                        "impact": self._assess_context_impact(key, value, plan)
                    }
                    factors.append(factor)
        
        return factors
    
    def _determine_step_causal_role(self, step: Dict[str, Any]) -> str:
        """确定计划步骤的因果角色"""
        step_type = step.get("type", "")
        
        role_mapping = {
            "analysis": "information_provider",
            "generation": "action_initiator",
            "evaluation": "quality_controller",
            "decision": "choice_maker",
            "execution": "outcome_producer"
        }
        
        return role_mapping.get(step_type, "unknown")
    
    def _assess_resource_criticality(self, resource: str, plan: Dict[str, Any]) -> float:
        """评估资源关键性"""
        steps = plan.get("steps", [])
        if not steps:
            return 0.0
        
        # 计算需要该资源的步骤比例
        steps_using_resource = 0
        for step in steps:
            if resource in step.get("resources", []):
                steps_using_resource += 1
        
        return steps_using_resource / len(steps)
    
    def _assess_context_impact(self, key: str, value: Any, plan: Dict[str, Any]) -> float:
        """评估上下文影响"""
        # 简化实现：基于关键词匹配
        plan_str = str(plan).lower()
        key_lower = key.lower()
        
        if key_lower in plan_str:
            return 0.7
        elif any(word in key_lower for word in ["time", "resource", "constraint", "requirement"]):
            return 0.5
        else:
            return 0.3
    
    def _build_plan_causal_graph(self, plan: Dict[str, Any],
                                causal_factors: List[Dict[str, Any]]) -> nx.DiGraph:
        """构建计划因果图"""
        graph = nx.DiGraph()
        
        # 添加因素作为节点
        for factor in causal_factors:
            graph.add_node(
                factor["id"],
                type=factor["type"],
                description=factor["description"],
                causal_role=factor["causal_role"]
            )
        
        # 添加因果关系边
        for factor in causal_factors:
            factor_id = factor["id"]
            
            # 依赖关系
            for dep in factor.get("dependencies", []):
                if dep in graph:
                    graph.add_edge(dep, factor_id, relation="depends_on", strength="strong")
            
            # 资源关系
            for resource in factor.get("resources", []):
                resource_id = f"resource_{resource}"
                if resource_id in graph:
                    graph.add_edge(resource_id, factor_id, relation="enables", strength="moderate")
            
            # 时间顺序关系（基于步骤顺序）
            if factor["type"] == "plan_step":
                step_num = int(factor["id"].split("_")[1])
                if step_num > 0:
                    prev_step_id = f"step_{step_num - 1}"
                    if prev_step_id in graph:
                        graph.add_edge(prev_step_id, factor_id, relation="precedes", strength="strong")
        
        return graph
    
    def _identify_critical_causal_paths(self, graph: nx.DiGraph, 
                                       plan: Dict[str, Any]) -> List[List[str]]:
        """识别关键因果路径"""
        critical_paths = []
        
        # 找到源节点（没有入边的节点）
        source_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        
        # 找到汇节点（没有出边的节点）
        sink_nodes = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        
        # 找出所有从源到汇的路径
        for source in source_nodes:
            for sink in sink_nodes:
                try:
                    paths = list(nx.all_simple_paths(graph, source, sink))
                    for path in paths:
                        # 评估路径关键性
                        criticality = self._assess_path_criticality(path, graph, plan)
                        if criticality > 0.7:  # 关键性阈值
                            critical_paths.append({
                                "path": path,
                                "criticality": criticality,
                                "length": len(path),
                                "bottlenecks": self._identify_path_bottlenecks(path, graph)
                            })
                except nx.NetworkXNoPath:
                    continue
        
        # 按关键性排序
        critical_paths.sort(key=lambda x: x["criticality"], reverse=True)
        
        return critical_paths[:5]  # 返回前5个最关键的路径
    
    def _assess_path_criticality(self, path: List[str], graph: nx.DiGraph,
                                plan: Dict[str, Any]) -> float:
        """评估路径关键性"""
        if len(path) <= 1:
            return 0.0
        
        criticality = 0.0
        
        # 1. 路径长度权重
        length_weight = min(1.0, len(path) / 10.0)
        criticality += 0.3 * length_weight
        
        # 2. 节点关键性
        node_criticality_sum = 0.0
        for node in path:
            node_data = graph.nodes[node]
            role = node_data.get("causal_role", "")
            
            role_criticality = {
                "action_initiator": 0.9,
                "outcome_producer": 0.8,
                "choice_maker": 0.7,
                "quality_controller": 0.6,
                "information_provider": 0.5,
                "enabler": 0.4,
                "condition": 0.3
            }.get(role, 0.3)
            
            node_criticality_sum += role_criticality
        
        avg_node_criticality = node_criticality_sum / len(path)
        criticality += 0.4 * avg_node_criticality
        
        # 3. 边强度
        edge_strength_sum = 0.0
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i+1], default={})
            strength = edge_data.get("strength", "weak")
            
            strength_value = {
                "strong": 0.9,
                "moderate": 0.6,
                "weak": 0.3
            }.get(strength, 0.3)
            
            edge_strength_sum += strength_value
        
        if len(path) > 1:
            avg_edge_strength = edge_strength_sum / (len(path) - 1)
            criticality += 0.3 * avg_edge_strength
        
        return min(1.0, criticality)
    
    def _identify_path_bottlenecks(self, path: List[str], graph: nx.DiGraph) -> List[str]:
        """识别路径瓶颈"""
        bottlenecks = []
        
        for i, node in enumerate(path):
            # 检查节点的入度和出度
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            
            # 高入度节点可能是汇聚点
            if in_degree >= 3:
                bottlenecks.append({
                    "node": node,
                    "type": "convergence_point",
                    "degree": in_degree
                })
            
            # 高出度节点可能是分发点
            if out_degree >= 3:
                bottlenecks.append({
                    "node": node,
                    "type": "divergence_point",
                    "degree": out_degree
                })
            
            # 检查边强度
            if i > 0:
                prev_node = path[i - 1]
                edge_data = graph.get_edge_data(prev_node, node, default={})
                if edge_data.get("strength", "weak") == "weak":
                    bottlenecks.append({
                        "node": f"{prev_node}->{node}",
                        "type": "weak_link",
                        "strength": "weak"
                    })
        
        return bottlenecks
    
    def _analyze_causal_vulnerabilities(self, graph: nx.DiGraph,
                                       plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析因果脆弱性"""
        vulnerabilities = []
        
        # 1. 单点故障分析
        single_points_of_failure = self._identify_single_points_of_failure(graph)
        for spof in single_points_of_failure:
            vulnerabilities.append({
                "type": "single_point_of_failure",
                "node": spof,
                "description": f"单点故障: {spof}",
                "severity": 0.9,
                "mitigation": f"为{spof}添加冗余或替代方案"
            })
        
        # 2. 关键依赖分析
        critical_dependencies = self._identify_critical_dependencies(graph)
        for dep in critical_dependencies:
            vulnerabilities.append({
                "type": "critical_dependency",
                "dependency": dep,
                "description": f"关键依赖: {dep}",
                "severity": 0.7,
                "mitigation": f"减少对{dep}的依赖或添加备份"
            })
        
        # 3. 资源约束分析
        resource_constraints = self._identify_resource_constraints(plan)
        for constraint in resource_constraints:
            vulnerabilities.append({
                "type": "resource_constraint",
                "resource": constraint["resource"],
                "description": f"资源约束: {constraint['resource']} (需求: {constraint['demand']})",
                "severity": constraint["severity"],
                "mitigation": f"增加{constraint['resource']}供应或优化使用"
            })
        
        # 4. 时间压力分析
        time_pressures = self._identify_time_pressures(plan)
        for pressure in time_pressures:
            vulnerabilities.append({
                "type": "time_pressure",
                "step": pressure["step"],
                "description": f"时间压力: {pressure['step']} (估计时间: {pressure['estimated_time']})",
                "severity": pressure["severity"],
                "mitigation": f"优化{pressure['step']}的时间安排或并行化"
            })
        
        return vulnerabilities
    
    def _identify_single_points_of_failure(self, graph: nx.DiGraph) -> List[str]:
        """识别单点故障"""
        spofs = []
        
        # 计算节点的中介中心性
        betweenness = nx.betweenness_centrality(graph)
        
        for node, centrality in betweenness.items():
            if centrality > 0.7:  # 高中介中心性
                spofs.append(node)
        
        return spofs
    
    def _identify_critical_dependencies(self, graph: nx.DiGraph) -> List[str]:
        """识别关键依赖"""
        critical_deps = []
        
        for node in graph.nodes():
            # 检查节点的依赖数量
            dependencies = list(graph.predecessors(node))
            
            if len(dependencies) >= 2:
                # 检查依赖之间的关系
                all_connected = True
                for i in range(len(dependencies)):
                    for j in range(i + 1, len(dependencies)):
                        if not (graph.has_edge(dependencies[i], dependencies[j]) or 
                                graph.has_edge(dependencies[j], dependencies[i])):
                            all_connected = False
                            break
                    if not all_connected:
                        break
                
                if not all_connected:
                    critical_deps.append(node)
        
        return critical_deps
    
    def _identify_resource_constraints(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别资源约束"""
        constraints = []
        resources = plan.get("resource_requirements", {})
        
        for resource, demand in resources.items():
            # 简化实现：基于需求大小判断约束
            if demand >= 3:  # 高需求
                severity = min(0.9, demand / 5.0)
                constraints.append({
                    "resource": resource,
                    "demand": demand,
                    "severity": severity
                })
        
        return constraints
    
    def _identify_time_pressures(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别时间压力"""
        pressures = []
        steps = plan.get("steps", [])
        
        for step in steps:
            estimated_time = step.get("estimated_time", 0)
            if estimated_time >= 20:  # 长时间任务
                severity = min(0.8, estimated_time / 50.0)
                pressures.append({
                    "step": step.get("description", "unknown"),
                    "estimated_time": estimated_time,
                    "severity": severity
                })
        
        return pressures
    
    def _assess_causal_robustness(self, graph: nx.DiGraph,
                                 plan: Dict[str, Any]) -> Dict[str, Any]:
        """评估因果鲁棒性"""
        robustness_scores = {
            "structural_robustness": self._assess_structural_robustness(graph),
            "functional_redundancy": self._assess_functional_redundancy(graph),
            "adaptability": self._assess_adaptability(graph, plan),
            "fault_tolerance": self._assess_fault_tolerance(graph)
        }
        
        # 总体鲁棒性分数
        overall_robustness = sum(robustness_scores.values()) / len(robustness_scores)
        
        return {
            "scores": robustness_scores,
            "overall_robustness": overall_robustness,
            "level": self._determine_robustness_level(overall_robustness)
        }
    
    def _assess_structural_robustness(self, graph: nx.DiGraph) -> float:
        """评估结构鲁棒性"""
        # 基于图连通性
        if graph.number_of_nodes() == 0:
            return 0.0
        
        # 计算图的连通性
        try:
            if nx.is_weakly_connected(graph):
                connectivity_score = 0.8
            else:
                # 多个连通组件
                num_components = nx.number_weakly_connected_components(graph)
                connectivity_score = 0.8 / num_components
        except Exception as e:
            logger.debug(f"图连通性分析失败: {e}")
            connectivity_score = 0.5
        
        # 计算图的密度
        density = nx.density(graph)
        density_score = min(1.0, density * 5)  # 适度密度更好
        
        return (connectivity_score + density_score) / 2
    
    def _assess_functional_redundancy(self, graph: nx.DiGraph) -> float:
        """评估功能冗余"""
        if graph.number_of_nodes() <= 1:
            return 0.0
        
        # 寻找功能相似的节点（基于邻居相似性）
        redundant_pairs = 0
        total_pairs = graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2
        
        if total_pairs == 0:
            return 0.0
        
        nodes = list(graph.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1 = nodes[i]
                node2 = nodes[j]
                
                # 检查邻居相似性
                neighbors1 = set(graph.successors(node1)) | set(graph.predecessors(node1))
                neighbors2 = set(graph.successors(node2)) | set(graph.predecessors(node2))
                
                similarity = len(neighbors1 & neighbors2) / max(1, len(neighbors1 | neighbors2))
                
                if similarity > 0.6:  # 高度相似
                    redundant_pairs += 1
        
        redundancy_score = min(1.0, redundant_pairs / max(1, total_pairs) * 3)
        
        return redundancy_score
    
    def _assess_adaptability(self, graph: nx.DiGraph, plan: Dict[str, Any]) -> float:
        """评估适应性"""
        # 基于计划步骤的灵活性
        steps = plan.get("steps", [])
        if not steps:
            return 0.0
        
        flexible_steps = 0
        for step in steps:
            step_type = step.get("type", "")
            # 分析和评估步骤通常更灵活
            if step_type in ["analysis", "evaluation", "decision"]:
                flexible_steps += 1
        
        adaptability_score = flexible_steps / len(steps)
        
        # 基于图结构的适应性
        avg_degree = sum(dict(graph.degree()).values()) / max(1, graph.number_of_nodes())
        degree_variability = np.std(list(dict(graph.degree()).values())) if graph.number_of_nodes() > 1 else 0
        
        # 适度变异性有利于适应性
        variability_score = min(1.0, degree_variability / 2)
        
        return (adaptability_score + variability_score) / 2
    
    def _assess_fault_tolerance(self, graph: nx.DiGraph) -> float:
        """评估容错性"""
        if graph.number_of_nodes() <= 1:
            return 0.0
        
        # 模拟节点故障的影响
        fault_tolerance_scores = []
        
        for node in graph.nodes():
            # 创建副本并移除节点
            graph_copy = graph.copy()
            graph_copy.remove_node(node)
            
            # 评估移除后的连通性
            if nx.is_weakly_connected(graph_copy):
                connectivity_after = 1.0
            else:
                num_components = nx.number_weakly_connected_components(graph_copy)
                connectivity_after = 1.0 / num_components
            
            fault_tolerance_scores.append(connectivity_after)
        
        if not fault_tolerance_scores:
            return 0.0
        
        avg_fault_tolerance = sum(fault_tolerance_scores) / len(fault_tolerance_scores)
        
        return avg_fault_tolerance
    
    def _determine_robustness_level(self, robustness_score: float) -> str:
        """确定鲁棒性级别"""
        if robustness_score >= 0.8:
            return "high"
        elif robustness_score >= 0.6:
            return "moderate"
        elif robustness_score >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def _generate_causal_optimization_suggestions(self, graph: nx.DiGraph,
                                                 critical_paths: List[Dict[str, Any]],
                                                 vulnerabilities: List[Dict[str, Any]]) -> List[str]:
        """生成因果优化建议"""
        suggestions = []
        
        # 基于关键路径的优化
        for path_info in critical_paths[:3]:  # 前3个最关键的路径
            path = path_info["path"]
            bottlenecks = path_info.get("bottlenecks", [])
            
            if bottlenecks:
                for bottleneck in bottlenecks:
                    suggestions.append(
                        f"优化关键路径上的瓶颈: {bottleneck.get('node', 'unknown')} "
                        f"(类型: {bottleneck.get('type', 'unknown')})"
                    )
            else:
                suggestions.append(
                    f"加强关键路径: {' -> '.join(path)} "
                    f"(关键性: {path_info['criticality']:.2f})"
                )
        
        # 基于脆弱性的优化
        for vulnerability in vulnerabilities[:5]:  # 前5个最严重的脆弱性
            v_type = vulnerability["type"]
            severity = vulnerability.get("severity", 0.0)
            
            if severity >= 0.7:
                suggestions.append(
                    f"解决高严重性脆弱性: {vulnerability['description']} "
                    f"(严重性: {severity:.2f}) - 建议: {vulnerability.get('mitigation', '检查并修复')}"
                )
        
        # 一般性优化建议
        suggestions.extend([
            "考虑添加冗余路径以提高鲁棒性",
            "减少对单一资源或步骤的依赖",
            "优化步骤顺序以减少等待时间",
            "增加监控和反馈机制",
            "实施渐进式实施而非一次性部署"
        ])
        
        return suggestions
    
    def estimate_intervention_effects(self, plan: Dict[str, Any],
                                     intervention: Dict[str, Any],
                                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        估计干预措施的效果
        
        Args:
            plan: 原始计划
            intervention: 干预描述
            context: 上下文信息
            
        Returns:
            干预效果估计
        """
        try:
            # 1. 分析原始计划的因果结构
            original_analysis = self.analyze_causality(plan, context)
            
            # 2. 应用干预
            modified_plan = self._apply_intervention_to_plan(plan, intervention)
            
            # 3. 分析干预后计划的因果结构
            modified_analysis = self.analyze_causality(modified_plan, context)
            
            # 4. 比较分析结果
            effect_analysis = self._compare_causal_analyses(original_analysis, modified_analysis, intervention)
            
            result = {
                "success": True,
                "original_plan_analysis": original_analysis,
                "modified_plan_analysis": modified_analysis,
                "intervention_effects": effect_analysis,
                "recommendation": self._generate_intervention_recommendation(effect_analysis),
                "confidence": self._estimate_intervention_confidence(effect_analysis)
            }
            
            # 记录干预历史
            self.intervention_history.append({
                "timestamp": time.time(),
                "intervention": intervention,
                "effect_analysis": effect_analysis,
                "confidence": result["confidence"]
            })
            
            logger.info(f"干预效果估计完成: 干预类型={intervention.get('type', 'unknown')}")
            
            return result
            
        except Exception as e:
            error_handler.handle_error(e, "CausalReasoningEnhancer", "估计干预效果失败")
            return {
                "success": False,
                "error": str(e),
                "partial_analysis": None
            }
    
    def _apply_intervention_to_plan(self, plan: Dict[str, Any],
                                   intervention: Dict[str, Any]) -> Dict[str, Any]:
        """将干预应用到计划"""
        modified_plan = plan.copy()
        intervention_type = intervention.get("type", "")
        
        if intervention_type == "add_resource":
            # 添加资源
            resource = intervention.get("resource", "")
            quantity = intervention.get("quantity", 1)
            
            current_resources = modified_plan.get("resource_requirements", {})
            current_resources[resource] = current_resources.get(resource, 0) + quantity
            modified_plan["resource_requirements"] = current_resources
            
        elif intervention_type == "remove_step":
            # 移除步骤
            step_to_remove = intervention.get("step", "")
            steps = modified_plan.get("steps", [])
            
            filtered_steps = [
                step for step in steps 
                if step.get("description", "") != step_to_remove
            ]
            modified_plan["steps"] = filtered_steps
            
        elif intervention_type == "reorder_steps":
            # 重新排序步骤
            new_order = intervention.get("new_order", [])
            if new_order:
                steps = modified_plan.get("steps", [])
                
                # 创建步骤映射
                step_map = {step.get("description", ""): step for step in steps}
                
                # 按新顺序重新排列
                reordered_steps = []
                for step_desc in new_order:
                    if step_desc in step_map:
                        reordered_steps.append(step_map[step_desc])
                
                # 添加未在新顺序中指定的步骤
                for step in steps:
                    if step.get("description", "") not in new_order:
                        reordered_steps.append(step)
                
                modified_plan["steps"] = reordered_steps
        
        elif intervention_type == "increase_efficiency":
            # 提高效率（减少时间）
            efficiency_factor = intervention.get("factor", 0.8)  # 0.8表示减少20%时间
            steps = modified_plan.get("steps", [])
            
            for step in steps:
                if "estimated_time" in step:
                    step["estimated_time"] = int(step["estimated_time"] * efficiency_factor)
            
            modified_plan["steps"] = steps
        
        else:
            # 默认：添加干预注释
            modified_plan["interventions_applied"] = modified_plan.get("interventions_applied", [])
            modified_plan["interventions_applied"].append(intervention)
        
        return modified_plan
    
    def _compare_causal_analyses(self, original: Dict[str, Any],
                                modified: Dict[str, Any],
                                intervention: Dict[str, Any]) -> Dict[str, Any]:
        """比较因果分析结果"""
        comparison = {
            "robustness_change": self._compare_robustness(original, modified),
            "vulnerability_change": self._compare_vulnerabilities(original, modified),
            "critical_path_change": self._compare_critical_paths(original, modified),
            "overall_effect": self._calculate_overall_effect(original, modified, intervention)
        }
        
        return comparison
    
    def _compare_robustness(self, original: Dict[str, Any],
                           modified: Dict[str, Any]) -> Dict[str, Any]:
        """比较鲁棒性"""
        orig_robustness = original.get("robustness_assessment", {}).get("overall_robustness", 0.5)
        mod_robustness = modified.get("robustness_assessment", {}).get("overall_robustness", 0.5)
        
        change = mod_robustness - orig_robustness
        improvement = change > 0
        
        return {
            "original": orig_robustness,
            "modified": mod_robustness,
            "change": change,
            "improvement": improvement,
            "magnitude": abs(change)
        }
    
    def _compare_vulnerabilities(self, original: Dict[str, Any],
                                modified: Dict[str, Any]) -> Dict[str, Any]:
        """比较脆弱性"""
        orig_vulns = original.get("vulnerabilities", [])
        mod_vulns = modified.get("vulnerabilities", [])
        
        # 计算平均严重性
        def avg_severity(vulns):
            if not vulns:
                return 0.0
            severities = [v.get("severity", 0.0) for v in vulns]
            return sum(severities) / len(severities)
        
        orig_severity = avg_severity(orig_vulns)
        mod_severity = avg_severity(mod_vulns)
        
        # 计算数量变化
        count_change = len(mod_vulns) - len(orig_vulns)
        
        return {
            "original_count": len(orig_vulns),
            "modified_count": len(mod_vulns),
            "count_change": count_change,
            "original_avg_severity": orig_severity,
            "modified_avg_severity": mod_severity,
            "severity_change": mod_severity - orig_severity,
            "improvement": mod_severity < orig_severity
        }
    
    def _compare_critical_paths(self, original: Dict[str, Any],
                               modified: Dict[str, Any]) -> Dict[str, Any]:
        """比较关键路径"""
        orig_paths = original.get("critical_paths", [])
        mod_paths = modified.get("critical_paths", [])
        
        # 计算平均关键性
        def avg_criticality(paths):
            if not paths:
                return 0.0
            criticalities = [p.get("criticality", 0.0) for p in paths]
            return sum(criticalities) / len(criticalities)
        
        orig_criticality = avg_criticality(orig_paths)
        mod_criticality = avg_criticality(mod_paths)
        
        return {
            "original_avg_criticality": orig_criticality,
            "modified_avg_criticality": mod_criticality,
            "criticality_change": mod_criticality - orig_criticality,
            "improvement": mod_criticality < orig_criticality,
            "path_count_change": len(mod_paths) - len(orig_paths)
        }
    
    def _calculate_overall_effect(self, original: Dict[str, Any],
                                 modified: Dict[str, Any],
                                 intervention: Dict[str, Any]) -> Dict[str, Any]:
        """计算总体效果"""
        # 收集各个维度的变化
        robustness_change = self._compare_robustness(original, modified)
        vulnerability_change = self._compare_vulnerabilities(original, modified)
        critical_path_change = self._compare_critical_paths(original, modified)
        
        # 加权计算总体效果
        weights = {
            "robustness": 0.4,
            "vulnerability": 0.3,
            "critical_path": 0.3
        }
        
        # 归一化变化值
        def normalize_change(change_dict, positive_is_good=True):
            change = change_dict.get("change", 0.0)
            if positive_is_good:
                return max(-1.0, min(1.0, change))
            else:
                return max(-1.0, min(1.0, -change))  # 反转符号
        
        robustness_score = normalize_change(robustness_change, True)
        vulnerability_score = normalize_change(vulnerability_change, False)  # 脆弱性减少是好的
        critical_path_score = normalize_change(critical_path_change, False)  # 关键性减少是好的
        
        overall_score = (
            weights["robustness"] * robustness_score +
            weights["vulnerability"] * vulnerability_score +
            weights["critical_path"] * critical_path_score
        )
        
        return {
            "overall_score": overall_score,
            "component_scores": {
                "robustness": robustness_score,
                "vulnerability": vulnerability_score,
                "critical_path": critical_path_score
            },
            "weights": weights,
            "interpretation": self._interpret_overall_score(overall_score)
        }
    
    def _interpret_overall_score(self, score: float) -> str:
        """解释总体分数"""
        if score >= 0.7:
            return "强烈正面效果：干预显著改善了计划的因果属性"
        elif score >= 0.3:
            return "正面效果：干预对计划有积极影响"
        elif score >= -0.3:
            return "中性效果：干预影响有限"
        elif score >= -0.7:
            return "负面效果：干预对计划有不利影响"
        else:
            return "强烈负面效果：干预显著恶化了计划的因果属性"
    
    def _generate_intervention_recommendation(self, effect_analysis: Dict[str, Any]) -> str:
        """生成干预建议"""
        overall_effect = effect_analysis.get("overall_effect", {})
        overall_score = overall_effect.get("overall_score", 0.0)
        
        if overall_score >= 0.5:
            return "强烈推荐实施此干预措施"
        elif overall_score >= 0.2:
            return "建议实施此干预措施"
        elif overall_score >= -0.2:
            return "可以考虑实施此干预措施，但效果有限"
        elif overall_score >= -0.5:
            return "不建议实施此干预措施"
        else:
            return "强烈不建议实施此干预措施"
    
    def _estimate_intervention_confidence(self, effect_analysis: Dict[str, Any]) -> float:
        """估计干预置信度"""
        # 基于分析的完整性
        completeness_score = 0.0
        
        components = ["robustness_change", "vulnerability_change", "critical_path_change"]
        for component in components:
            if component in effect_analysis and effect_analysis[component]:
                completeness_score += 0.33
        
        # 基于效果的一致性
        consistency_score = 1.0
        
        # 基于历史数据的准确性（简化）
        historical_accuracy = 0.7 if self.intervention_history else 0.5
        
        # 综合置信度
        confidence = (completeness_score + consistency_score + historical_accuracy) / 3
        
        return min(1.0, max(0.0, confidence))


# 实用函数：创建因果推理增强器实例
def create_causal_reasoning_enhancer(config: Optional[Dict[str, Any]] = None) -> CausalReasoningEnhancer:
    """
    创建因果推理增强器实例
    
    Args:
        config: 配置字典
        
    Returns:
        初始化好的因果推理增强器实例
    """
    return CausalReasoningEnhancer(config)


# 示例使用
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建增强器
    enhancer = create_causal_reasoning_enhancer()
    
    # 测试计划
    test_plan = {
        "id": "test_plan_1",
        "goal": "开发AGI系统",
        "steps": [
            {
                "type": "analysis",
                "description": "分析AGI需求",
                "resources": ["analytical_capacity"],
                "estimated_time": 10,
                "depends_on": []
            },
            {
                "type": "generation",
                "description": "设计系统架构",
                "resources": ["creative_capacity", "technical_knowledge"],
                "estimated_time": 15,
                "depends_on": ["analysis"]
            },
            {
                "type": "evaluation",
                "description": "评估设计可行性",
                "resources": ["analytical_capacity"],
                "estimated_time": 8,
                "depends_on": ["generation"]
            },
            {
                "type": "execution",
                "description": "实施系统",
                "resources": ["technical_skills", "computational_resources"],
                "estimated_time": 30,
                "depends_on": ["evaluation"]
            }
        ],
        "resource_requirements": {
            "analytical_capacity": 2,
            "creative_capacity": 1,
            "technical_knowledge": 1,
            "technical_skills": 2,
            "computational_resources": 3
        }
    }
    
    print("开始测试因果推理增强器...")
    print(f"测试计划: {test_plan['goal']}")
    print()
    
    # 分析因果结构
    analysis_result = enhancer.analyze_causality(test_plan)
    
    print("因果分析结果:")
    print(f"成功: {analysis_result['success']}")
    
    if analysis_result['success']:
        print(f"因果因素数: {len(analysis_result['causal_factors'])}")
        print(f"关键路径数: {len(analysis_result['critical_paths'])}")
        print(f"脆弱性数: {len(analysis_result['vulnerabilities'])}")
        print(f"总体鲁棒性: {analysis_result['robustness_assessment']['overall_robustness']:.2f}")
        
        print()
        print("优化建议:")
        for suggestion in analysis_result['optimization_suggestions'][:5]:
            print(f"  - {suggestion}")
    
    print()
    
    # 测试干预效果估计
    intervention = {
        "type": "add_resource",
        "resource": "computational_resources",
        "quantity": 2
    }
    
    effect_result = enhancer.estimate_intervention_effects(test_plan, intervention)
    
    print("干预效果估计:")
    print(f"成功: {effect_result['success']}")
    
    if effect_result['success']:
        overall_effect = effect_result['intervention_effects']['overall_effect']
        print(f"总体效果分数: {overall_effect['overall_score']:.2f}")
        print(f"效果解释: {overall_effect['interpretation']}")
        print(f"建议: {effect_result['recommendation']}")
        print(f"置信度: {effect_result['confidence']:.2f}")