#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
潜在混杂变量检测器 - 检测因果分析中的潜在混杂变量

核心功能:
1. 基于图结构的混杂变量检测
2. 统计依赖性分析
3. 因果结构学习中的混杂识别
4. 敏感性分析
5. 混杂风险量化

检测方法:
1. 后门准则验证
2. 工具变量检验
3. 双重稳健估计
4. 敏感性分析
5. 基于机器学习的混杂检测

技术实现:
- 因果图遍历算法
- 统计独立性检验
- 机器学习模型
- 敏感性分析框架

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
from scipy import stats
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# 导入错误处理
from core.error_handling import ErrorHandler

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()


class ConfounderDetectionMethod(Enum):
    """混杂变量检测方法枚举"""
    BACKDOOR_CRITERION = "backdoor_criterion"          # 后门准则
    INSTRUMENTAL_VARIABLE = "instrumental_variable"    # 工具变量
    DOUBLE_ROBUST = "double_robust"                    # 双重稳健
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"      # 敏感性分析
    MACHINE_LEARNING = "machine_learning"              # 机器学习
    GRAPH_BASED = "graph_based"                        # 基于图结构


class ConfounderRiskLevel(Enum):
    """混杂风险等级枚举"""
    NONE = "none"              # 无风险: 无显著混杂
    LOW = "low"                # 低风险: 轻微混杂，影响小
    MODERATE = "moderate"      # 中等风险: 适度混杂，需要调整
    HIGH = "high"              # 高风险: 显著混杂，严重影响估计
    SEVERE = "severe"          # 严重风险: 严重混杂，因果推断不可靠


class HiddenConfounderDetector:
    """
    潜在混杂变量检测器
    
    核心功能:
    1. 基于因果图的混杂变量检测
    2. 统计依赖性分析
    3. 混杂风险量化
    4. 敏感性分析
    5. 调整建议生成
    
    检测流程:
    1. 输入因果图和目标变量
    2. 应用多种检测方法
    3. 综合评估混杂风险
    4. 提供调整建议
    5. 验证调整效果
    
    技术特性:
    - 多方法集成检测
    - 混杂风险量化
    - 敏感性分析框架
    - 调整策略推荐
    - 性能优化和缓存
    """
    
    def __init__(self,
                 min_dependence_threshold: float = 0.1,
                 min_risk_threshold: float = 0.3,
                 max_confounders: int = 10):
        """
        初始化潜在混杂变量检测器
        
        Args:
            min_dependence_threshold: 最小依赖性阈值
            min_risk_threshold: 最小风险阈值
            max_confounders: 最大混杂变量数量
        """
        self.min_dependence_threshold = min_dependence_threshold
        self.min_risk_threshold = min_risk_threshold
        self.max_confounders = max_confounders
        
        # 检测缓存
        self.detection_cache = {}
        
        # 性能统计
        self.performance_stats = {
            "detections_performed": 0,
            "confounders_detected": 0,
            "statistical_tests": 0,
            "graph_traversals": 0,
            "adjustments_recommended": 0
        }
        
        # 初始化机器学习模型（按需创建）
        self.ml_models = {}
        
        logger.info("潜在混杂变量检测器初始化完成")
    
    def detect_confounders(self,
                          causal_graph: nx.DiGraph,
                          treatment: str,
                          outcome: str,
                          observed_data: Optional[Dict[str, np.ndarray]] = None,
                          methods: Optional[List[ConfounderDetectionMethod]] = None) -> Dict[str, Any]:
        """
        检测潜在混杂变量
        
        Args:
            causal_graph: 因果图（有向无环图）
            treatment: 处理变量
            outcome: 结果变量
            observed_data: 观测数据字典 {variable: values}
            methods: 检测方法列表（如果为None则使用所有方法）
            
        Returns:
            混杂检测结果
        """
        start_time = time.time()
        
        # 检查输入有效性
        validation_result = self._validate_inputs(causal_graph, treatment, outcome, observed_data)
        if not validation_result["valid"]:
            return validation_result
        
        # 默认使用所有方法
        if methods is None:
            methods = list(ConfounderDetectionMethod)
        
        # 执行各种检测方法
        detection_results = {}
        
        for method in methods:
            method_start = time.time()
            
            if method == ConfounderDetectionMethod.BACKDOOR_CRITERION:
                result = self._detect_by_backdoor_criterion(causal_graph, treatment, outcome)
            
            elif method == ConfounderDetectionMethod.GRAPH_BASED:
                result = self._detect_by_graph_structure(causal_graph, treatment, outcome)
            
            elif method == ConfounderDetectionMethod.INSTRUMENTAL_VARIABLE:
                result = self._detect_by_instrumental_variable(causal_graph, treatment, outcome)
            
            elif method == ConfounderDetectionMethod.DOUBLE_ROBUST:
                result = self._detect_by_double_robust(causal_graph, treatment, outcome, observed_data)
            
            elif method == ConfounderDetectionMethod.SENSITIVITY_ANALYSIS:
                result = self._detect_by_sensitivity_analysis(causal_graph, treatment, outcome, observed_data)
            
            elif method == ConfounderDetectionMethod.MACHINE_LEARNING:
                result = self._detect_by_machine_learning(causal_graph, treatment, outcome, observed_data)
            
            else:
                result = {
                    "success": False,
                    "error": f"不支持的检测方法: {method}"
                }
            
            method_time = time.time() - method_start
            result["method_time"] = method_time
            detection_results[method.value] = result
        
        # 综合评估结果
        combined_result = self._combine_detection_results(detection_results, treatment, outcome)
        
        # 添加总体性能信息
        elapsed_time = time.time() - start_time
        combined_result["performance"] = {
            "total_time": elapsed_time,
            "methods_applied": len(methods),
            "cache_hits": 0  # 简化，实际应统计缓存命中
        }
        
        self.performance_stats["detections_performed"] += 1
        
        logger.info(f"混杂变量检测完成: {treatment} -> {outcome}, 检测方法: {len(methods)}, 发现混杂变量: {len(combined_result.get('confounders', []))}")
        
        return combined_result
    
    def _validate_inputs(self,
                        causal_graph: nx.DiGraph,
                        treatment: str,
                        outcome: str,
                        observed_data: Optional[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """验证输入参数"""
        # 检查图是否为有向无环图
        if not nx.is_directed_acyclic_graph(causal_graph):
            return {
                "success": False,
                "valid": False,
                "error": "因果图必须是有向无环图(DAG)"
            }
        
        # 检查变量是否存在
        if treatment not in causal_graph.nodes():
            return {
                "success": False,
                "valid": False,
                "error": f"处理变量不存在于图中: {treatment}"
            }
        
        if outcome not in causal_graph.nodes():
            return {
                "success": False,
                "valid": False,
                "error": f"结果变量不存在于图中: {outcome}"
            }
        
        # 检查观测数据（如果提供）
        if observed_data is not None:
            # 检查必要变量
            required_vars = [treatment, outcome]
            missing_vars = [var for var in required_vars if var not in observed_data]
            
            if missing_vars:
                return {
                    "success": False,
                    "valid": False,
                    "error": f"观测数据缺少变量: {missing_vars}"
                }
            
            # 检查数据长度一致性
            data_lengths = [len(observed_data[var]) for var in observed_data]
            if len(set(data_lengths)) > 1:
                return {
                    "success": False,
                    "valid": False,
                    "error": "观测数据长度不一致"
                }
        
        return {
            "success": True,
            "valid": True,
            "message": "输入验证通过"
        }
    
    def _detect_by_backdoor_criterion(self,
                                     causal_graph: nx.DiGraph,
                                     treatment: str,
                                     outcome: str) -> Dict[str, Any]:
        """基于后门准则检测混杂变量"""
        self.performance_stats["graph_traversals"] += 1
        
        # 查找满足后门准则的变量集
        backdoor_sets = self._find_backdoor_adjustment_sets(causal_graph, treatment, outcome)
        
        # 识别潜在混杂变量（后门集中的变量）
        potential_confounders = []
        for adj_set in backdoor_sets:
            for var in adj_set:
                if var not in potential_confounders:
                    potential_confounders.append(var)
        
        # 评估每个混杂变量
        confounder_assessments = []
        for confounder in potential_confounders[:self.max_confounders]:  # 限制数量
            assessment = self._assess_confounder_by_graph(confounder, causal_graph, treatment, outcome)
            confounder_assessments.append(assessment)
        
        # 按风险排序
        confounder_assessments.sort(key=lambda x: x.get("risk_score", 0), reverse=True)
        
        return {
            "success": True,
            "method": "backdoor_criterion",
            "potential_confounders": potential_confounders[:self.max_confounders],
            "confounder_assessments": confounder_assessments,
            "backdoor_sets_found": len(backdoor_sets),
            "interpretation": "基于后门准则的混杂变量检测"
        }
    
    def _find_backdoor_adjustment_sets(self,
                                      causal_graph: nx.DiGraph,
                                      treatment: str,
                                      outcome: str,
                                      max_set_size: int = 3) -> List[List[str]]:
        """查找满足后门准则的调整变量集"""
        backdoor_sets = []
        
        # 获取所有变量（排除处理和结果变量）
        all_vars = list(causal_graph.nodes())
        candidate_vars = [var for var in all_vars if var != treatment and var != outcome]
        
        # 生成所有可能的子集（限制大小）
        from itertools import combinations
        
        for set_size in range(1, min(max_set_size + 1, len(candidate_vars) + 1)):
            for var_set in combinations(candidate_vars, set_size):
                var_list = list(var_set)
                
                # 检查是否满足后门准则
                if self._check_backdoor_criterion(var_list, causal_graph, treatment, outcome):
                    backdoor_sets.append(var_list)
        
        return backdoor_sets
    
    def _check_backdoor_criterion(self,
                                 adjustment_set: List[str],
                                 causal_graph: nx.DiGraph,
                                 treatment: str,
                                 outcome: str) -> bool:
        """检查调整集是否满足后门准则"""
        # 简化实现：检查三个条件
        
        # 条件1: 调整集不包含处理变量的后代
        for var in adjustment_set:
            if nx.has_path(causal_graph, treatment, var):
                return False
        
        # 条件2: 调整集阻塞所有从处理到结果的后门路径
        # 后门路径：从处理到结果的路径，且第一个边指向处理变量
        
        # 获取所有从处理到结果的路径
        try:
            all_paths = list(nx.all_simple_paths(causal_graph, treatment, outcome))
        except:
            all_paths = []
        
        for path in all_paths:
            # 检查是否为后门路径（第一个边指向处理变量）
            if len(path) >= 2:
                # 检查路径中第一个边是否指向处理变量
                # 在DAG中，如果第二个节点是第一个节点的父节点，则是后门路径
                if causal_graph.has_edge(path[1], path[0]):  # 反向边
                    # 这是后门路径，检查是否被调整集阻塞
                    if not self._is_path_blocked(path, adjustment_set, causal_graph):
                        return False
        
        # 条件3: 简化：假设满足
        return True
    
    def _is_path_blocked(self,
                        path: List[str],
                        adjustment_set: List[str],
                        causal_graph: nx.DiGraph) -> bool:
        """检查路径是否被调整集阻塞"""
        # 简化实现：检查调整集中的变量是否在路径上
        for var in adjustment_set:
            if var in path:
                return True
        
        return False
    
    def _assess_confounder_by_graph(self,
                                   confounder: str,
                                   causal_graph: nx.DiGraph,
                                   treatment: str,
                                   outcome: str) -> Dict[str, Any]:
        """基于图结构评估混杂变量"""
        # 检查是否存在从混杂变量到处理和结果的路径
        has_to_treatment = nx.has_path(causal_graph, confounder, treatment)
        has_to_outcome = nx.has_path(causal_graph, confounder, outcome)
        
        # 检查是否为共同原因
        is_common_cause = has_to_treatment and has_to_outcome
        
        # 计算混杂风险分数
        risk_score = 0.0
        
        if is_common_cause:
            # 基础风险分数
            risk_score = 0.5
            
            # 路径强度增强（如果有直接边）
            if causal_graph.has_edge(confounder, treatment):
                risk_score += 0.2
            if causal_graph.has_edge(confounder, outcome):
                risk_score += 0.2
            
            # 路径长度惩罚（较长的路径风险较低）
            if has_to_treatment:
                try:
                    treatment_path_length = len(nx.shortest_path(causal_graph, confounder, treatment)) - 1
                    risk_score *= max(0.5, 1.0 / treatment_path_length)
                except:
                    pass
            
            if has_to_outcome:
                try:
                    outcome_path_length = len(nx.shortest_path(causal_graph, confounder, outcome)) - 1
                    risk_score *= max(0.5, 1.0 / outcome_path_length)
                except:
                    pass
        
        # 确定风险等级
        risk_level = self._determine_risk_level(risk_score)
        
        return {
            "confounder": confounder,
            "is_common_cause": is_common_cause,
            "has_path_to_treatment": has_to_treatment,
            "has_path_to_outcome": has_to_outcome,
            "risk_score": risk_score,
            "risk_level": risk_level.value if hasattr(risk_level, 'value') else str(risk_level),
            "adjustment_required": risk_level in [ConfounderRiskLevel.MODERATE, ConfounderRiskLevel.HIGH, ConfounderRiskLevel.SEVERE]
        }
    
    def _detect_by_graph_structure(self,
                                  causal_graph: nx.DiGraph,
                                  treatment: str,
                                  outcome: str) -> Dict[str, Any]:
        """基于图结构检测混杂变量"""
        self.performance_stats["graph_traversals"] += 1
        
        # 查找共同原因
        common_causes = []
        
        # 获取所有节点
        all_nodes = list(causal_graph.nodes())
        
        for node in all_nodes:
            if node == treatment or node == outcome:
                continue
            
            # 检查是否为处理和结果的共同原因
            has_to_treatment = nx.has_path(causal_graph, node, treatment)
            has_to_outcome = nx.has_path(causal_graph, node, outcome)
            
            if has_to_treatment and has_to_outcome:
                common_causes.append(node)
        
        # 评估每个共同原因
        confounder_assessments = []
        for confounder in common_causes[:self.max_confounders]:
            assessment = self._assess_confounder_by_graph(confounder, causal_graph, treatment, outcome)
            confounder_assessments.append(assessment)
        
        # 按风险排序
        confounder_assessments.sort(key=lambda x: x.get("risk_score", 0), reverse=True)
        
        return {
            "success": True,
            "method": "graph_based",
            "common_causes_found": len(common_causes),
            "common_causes": common_causes[:self.max_confounders],
            "confounder_assessments": confounder_assessments,
            "interpretation": "基于图结构的共同原因检测"
        }
    
    def _detect_by_instrumental_variable(self,
                                        causal_graph: nx.DiGraph,
                                        treatment: str,
                                        outcome: str) -> Dict[str, Any]:
        """基于工具变量检测混杂变量"""
        # 工具变量检测：寻找满足工具变量条件的变量
        # 工具变量条件：
        # 1. 与处理变量相关
        # 2. 仅通过处理变量影响结果
        # 3. 与混杂变量独立
        
        potential_instruments = []
        
        # 获取所有节点
        all_nodes = list(causal_graph.nodes())
        
        for node in all_nodes:
            if node == treatment or node == outcome:
                continue
            
            # 条件1: 与处理变量相关（有路径）
            has_to_treatment = nx.has_path(causal_graph, node, treatment)
            
            if has_to_treatment:
                # 条件2: 仅通过处理变量影响结果
                # 检查是否存在不经过处理变量到结果的路径
                paths_to_outcome = []
                try:
                    # 获取所有从node到outcome的路径
                    for path in nx.all_simple_paths(causal_graph, node, outcome):
                        paths_to_outcome.append(path)
                except:
                    pass
                
                # 检查是否有路径不包含treatment
                has_direct_path = False
                for path in paths_to_outcome:
                    if treatment not in path:
                        has_direct_path = True
                        break
                
                # 如果没有直接路径（只通过treatment），可能是工具变量
                if not has_direct_path:
                    # 条件3: 与混杂变量独立（简化：假设满足）
                    potential_instruments.append(node)
        
        # 如果有工具变量，说明可能存在未观测的混杂
        has_hidden_confounding = len(potential_instruments) > 0
        
        return {
            "success": True,
            "method": "instrumental_variable",
            "has_hidden_confounding": has_hidden_confounding,
            "potential_instruments": potential_instruments[:self.max_confounders],
            "instruments_found": len(potential_instruments),
            "interpretation": f"基于工具变量检测: {'存在' if has_hidden_confounding else '未发现'}未观测混杂的证据"
        }
    
    def _detect_by_double_robust(self,
                                causal_graph: nx.DiGraph,
                                treatment: str,
                                outcome: str,
                                observed_data: Optional[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """基于双重稳健估计检测混杂变量"""
        if observed_data is None:
            return {
                "success": False,
                "method": "double_robust",
                "error": "双重稳健估计需要观测数据",
                "interpretation": "无法执行双重稳健估计（缺少数据）"
            }
        
        self.performance_stats["statistical_tests"] += 1
        
        # 简化实现：比较不同调整方法的估计结果
        
        # 获取候选调整变量
        candidate_vars = list(observed_data.keys())
        candidate_vars = [var for var in candidate_vars if var != treatment and var != outcome]
        
        if len(candidate_vars) == 0:
            return {
                "success": False,
                "method": "double_robust",
                "error": "没有可用的调整变量",
                "interpretation": "无法执行双重稳健估计（缺少调整变量）"
            }
        
        # 尝试不同的调整集
        adjustment_results = []
        
        # 无调整
        unadjusted_effect = self._estimate_naive_effect(observed_data, treatment, outcome)
        adjustment_results.append({
            "adjustment_set": [],
            "effect_estimate": unadjusted_effect,
            "method": "unadjusted"
        })
        
        # 使用所有变量调整
        all_adjusted_effect = self._estimate_adjusted_effect(observed_data, treatment, outcome, candidate_vars)
        adjustment_results.append({
            "adjustment_set": candidate_vars,
            "effect_estimate": all_adjusted_effect,
            "method": "all_variables"
        })
        
        # 比较估计结果
        effect_difference = abs(unadjusted_effect - all_adjusted_effect)
        has_confounding = effect_difference > self.min_dependence_threshold
        
        # 识别可能的混杂变量（导致估计变化的变量）
        suspected_confounders = []
        if has_confounding:
            # 测试每个变量的单独影响
            for var in candidate_vars[:min(5, len(candidate_vars))]:  # 测试前5个
                single_adjusted = self._estimate_adjusted_effect(observed_data, treatment, outcome, [var])
                single_difference = abs(unadjusted_effect - single_adjusted)
                
                if single_difference > self.min_dependence_threshold:
                    suspected_confounders.append({
                        "variable": var,
                        "effect_change": single_difference,
                        "adjusted_estimate": single_adjusted
                    })
            
            # 按影响大小排序
            suspected_confounders.sort(key=lambda x: x["effect_change"], reverse=True)
        
        return {
            "success": True,
            "method": "double_robust",
            "has_confounding": has_confounding,
            "effect_difference": effect_difference,
            "unadjusted_estimate": unadjusted_effect,
            "adjusted_estimate": all_adjusted_effect,
            "suspected_confounders": suspected_confounders,
            "adjustment_results": adjustment_results,
            "interpretation": f"双重稳健估计: {'检测到' if has_confounding else '未检测到'}显著混杂"
        }
    
    def _estimate_naive_effect(self,
                              observed_data: Dict[str, np.ndarray],
                              treatment: str,
                              outcome: str) -> float:
        """估计朴素因果效应（无调整）"""
        # 简化：计算相关性
        treatment_data = observed_data[treatment]
        outcome_data = observed_data[outcome]
        
        if len(treatment_data) < 2 or len(outcome_data) < 2:
            return 0.0
        
        # 计算相关系数
        try:
            correlation = np.corrcoef(treatment_data, outcome_data)[0, 1]
            if np.isnan(correlation):
                return 0.0
            return correlation
        except:
            return 0.0
    
    def _estimate_adjusted_effect(self,
                                 observed_data: Dict[str, np.ndarray],
                                 treatment: str,
                                 outcome: str,
                                 adjustment_set: List[str]) -> float:
        """估计调整后的因果效应"""
        if not adjustment_set:
            return self._estimate_naive_effect(observed_data, treatment, outcome)
        
        # 简化：使用线性回归调整
        try:
            # 准备数据
            n_samples = len(observed_data[treatment])
            
            # 构建设计矩阵
            X = np.ones((n_samples, 1 + len(adjustment_set)))  # 截距 + 调整变量
            X[:, 0] = observed_data[treatment]  # 处理变量
            
            for i, var in enumerate(adjustment_set):
                if var in observed_data and len(observed_data[var]) == n_samples:
                    X[:, 1 + i] = observed_data[var]
                else:
                    X[:, 1 + i] = 0.0
            
            y = observed_data[outcome]
            
            # 线性回归
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            # 处理变量的系数作为因果效应估计
            effect = model.coef_[0] if len(model.coef_) > 0 else 0.0
            return effect
            
        except Exception as e:
            logger.warning(f"调整效应估计失败: {e}")
            return self._estimate_naive_effect(observed_data, treatment, outcome)
    
    def _detect_by_sensitivity_analysis(self,
                                       causal_graph: nx.DiGraph,
                                       treatment: str,
                                       outcome: str,
                                       observed_data: Optional[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """基于敏感性分析检测混杂变量"""
        if observed_data is None:
            return {
                "success": False,
                "method": "sensitivity_analysis",
                "error": "敏感性分析需要观测数据",
                "interpretation": "无法执行敏感性分析（缺少数据）"
            }
        
        self.performance_stats["statistical_tests"] += 1
        
        # 基础效应估计
        base_effect = self._estimate_naive_effect(observed_data, treatment, outcome)
        
        # 敏感性分析：模拟未观测混杂的影响
        sensitivity_results = []
        
        # 模拟不同强度的未观测混杂
        confounding_strengths = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for strength in confounding_strengths:
            # 模拟混杂变量
            n_samples = len(observed_data[treatment])
            simulated_confounder = np.random.normal(0, 1, n_samples)
            
            # 模拟混杂对处理和结果的影响
            # treatment = f(confounder) + error
            treatment_bias = strength * simulated_confounder + np.random.normal(0, 0.1, n_samples)
            outcome_bias = strength * simulated_confounder + np.random.normal(0, 0.1, n_samples)
            
            # 调整后的数据
            adjusted_treatment = observed_data[treatment] - treatment_bias
            adjusted_outcome = observed_data[outcome] - outcome_bias
            
            # 估计调整后效应
            adjusted_data = observed_data.copy()
            adjusted_data[treatment] = adjusted_treatment
            adjusted_data[outcome] = adjusted_outcome
            
            adjusted_effect = self._estimate_naive_effect(adjusted_data, treatment, outcome)
            
            # 效应变化
            effect_change = abs(base_effect - adjusted_effect)
            
            sensitivity_results.append({
                "confounding_strength": strength,
                "adjusted_effect": adjusted_effect,
                "effect_change": effect_change,
                "relative_change": effect_change / abs(base_effect) if abs(base_effect) > 0 else 0.0
            })
        
        # 评估敏感性
        max_change = max([r["effect_change"] for r in sensitivity_results])
        is_sensitive = max_change > self.min_dependence_threshold
        
        return {
            "success": True,
            "method": "sensitivity_analysis",
            "is_sensitive_to_confounding": is_sensitive,
            "base_effect": base_effect,
            "max_effect_change": max_change,
            "sensitivity_results": sensitivity_results,
            "interpretation": f"敏感性分析: 效应估计{'对混杂敏感' if is_sensitive else '相对稳健'}"
        }
    
    def _detect_by_machine_learning(self,
                                   causal_graph: nx.DiGraph,
                                   treatment: str,
                                   outcome: str,
                                   observed_data: Optional[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """基于机器学习检测混杂变量"""
        if observed_data is None:
            return {
                "success": False,
                "method": "machine_learning",
                "error": "机器学习检测需要观测数据",
                "interpretation": "无法执行机器学习检测（缺少数据）"
            }
        
        # 获取候选变量
        candidate_vars = list(observed_data.keys())
        candidate_vars = [var for var in candidate_vars if var != treatment and var != outcome]
        
        if len(candidate_vars) < 2:
            return {
                "success": False,
                "method": "machine_learning",
                "error": "候选变量不足",
                "interpretation": "无法执行机器学习检测（变量不足）"
            }
        
        # 使用随机森林检测混杂变量
        try:
            # 准备数据
            n_samples = len(observed_data[treatment])
            
            # 特征矩阵（候选变量）
            X = np.zeros((n_samples, len(candidate_vars)))
            for i, var in enumerate(candidate_vars):
                if var in observed_data and len(observed_data[var]) == n_samples:
                    X[:, i] = observed_data[var]
            
            # 目标：预测处理变量
            y_treatment = observed_data[treatment]
            
            # 训练随机森林
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            
            # 二值化处理变量（简化）
            y_binary = (y_treatment > np.median(y_treatment)).astype(int)
            
            rf.fit(X, y_binary)
            
            # 获取特征重要性
            importances = rf.feature_importances_
            
            # 识别重要变量
            important_vars = []
            for i, var in enumerate(candidate_vars):
                if importances[i] > self.min_dependence_threshold:
                    important_vars.append({
                        "variable": var,
                        "importance": float(importances[i]),
                        "rank": i
                    })
            
            # 按重要性排序
            important_vars.sort(key=lambda x: x["importance"], reverse=True)
            
            # 评估混杂风险
            has_potential_confounding = len(important_vars) > 0
            
            return {
                "success": True,
                "method": "machine_learning",
                "has_potential_confounding": has_potential_confounding,
                "important_variables": important_vars[:self.max_confounders],
                "total_candidates": len(candidate_vars),
                "important_count": len(important_vars),
                "interpretation": f"机器学习检测: 发现{len(important_vars)}个可能混杂变量"
            }
            
        except Exception as e:
            logger.error(f"机器学习检测失败: {e}")
            return {
                "success": False,
                "method": "machine_learning",
                "error": str(e),
                "interpretation": "机器学习检测失败"
            }
    
    def _combine_detection_results(self,
                                  detection_results: Dict[str, Dict[str, Any]],
                                  treatment: str,
                                  outcome: str) -> Dict[str, Any]:
        """综合各检测方法的结果"""
        # 收集所有检测到的混杂变量
        all_confounders = set()
        confounder_scores = defaultdict(float)
        method_counts = defaultdict(int)
        
        for method_name, result in detection_results.items():
            if result.get("success", False):
                # 从不同方法中提取混杂变量
                confounders_from_method = []
                
                if "potential_confounders" in result:
                    confounders_from_method.extend(result["potential_confounders"])
                
                if "common_causes" in result:
                    confounders_from_method.extend(result["common_causes"])
                
                if "suspected_confounders" in result:
                    for item in result["suspected_confounders"]:
                        if isinstance(item, dict) and "variable" in item:
                            confounders_from_method.append(item["variable"])
                
                if "important_variables" in result:
                    for item in result["important_variables"]:
                        if isinstance(item, dict) and "variable" in item:
                            confounders_from_method.append(item["variable"])
                
                # 更新分数和计数
                for confounder in confounders_from_method:
                    all_confounders.add(confounder)
                    confounder_scores[confounder] += 1.0  # 每个方法贡献1分
                    method_counts[confounder] += 1
        
        # 计算综合风险分数
        confounder_assessments = []
        for confounder in all_confounders:
            # 基础分数（被多少方法检测到）
            base_score = confounder_scores[confounder] / len(detection_results)
            
            # 调整分数（基于方法一致性）
            consistency_score = method_counts[confounder] / len(detection_results)
            
            # 综合风险分数
            risk_score = 0.7 * base_score + 0.3 * consistency_score
            
            # 确定风险等级
            risk_level = self._determine_risk_level(risk_score)
            
            confounder_assessments.append({
                "confounder": confounder,
                "risk_score": risk_score,
                "risk_level": risk_level.value if hasattr(risk_level, 'value') else str(risk_level),
                "detected_by_methods": method_counts[confounder],
                "adjustment_required": risk_level in [ConfounderRiskLevel.MODERATE, ConfounderRiskLevel.HIGH, ConfounderRiskLevel.SEVERE]
            })
        
        # 按风险分数排序
        confounder_assessments.sort(key=lambda x: x["risk_score"], reverse=True)
        
        # 确定总体混杂风险
        overall_risk_score = 0.0
        if confounder_assessments:
            overall_risk_score = confounder_assessments[0]["risk_score"]
        
        overall_risk_level = self._determine_risk_level(overall_risk_score)
        
        # 生成调整建议
        adjustment_recommendations = []
        for assessment in confounder_assessments[:3]:  # 前3个高风险混杂
            if assessment["adjustment_required"]:
                adjustment_recommendations.append({
                    "confounder": assessment["confounder"],
                    "risk_level": assessment["risk_level"],
                    "adjustment_methods": ["regression_adjustment", "propensity_score", "matching"],
                    "priority": "high" if assessment["risk_score"] > 0.7 else "medium"
                })
        
        if adjustment_recommendations:
            self.performance_stats["adjustments_recommended"] += len(adjustment_recommendations)
        
        return {
            "success": True,
            "treatment": treatment,
            "outcome": outcome,
            "detection_methods_applied": list(detection_results.keys()),
            "confounders_detected": len(confounder_assessments),
            "confounder_assessments": confounder_assessments[:self.max_confounders],
            "overall_risk": {
                "risk_score": overall_risk_score,
                "risk_level": overall_risk_level.value if hasattr(overall_risk_level, 'value') else str(overall_risk_level),
                "interpretation": self._get_risk_interpretation(overall_risk_level)
            },
            "adjustment_recommendations": adjustment_recommendations,
            "summary": f"检测到{len(confounder_assessments)}个潜在混杂变量，总体风险等级: {overall_risk_level.value}"
        }
    
    def _determine_risk_level(self, risk_score: float) -> ConfounderRiskLevel:
        """根据风险分数确定风险等级"""
        if risk_score < 0.2:
            return ConfounderRiskLevel.NONE
        elif risk_score < 0.4:
            return ConfounderRiskLevel.LOW
        elif risk_score < 0.6:
            return ConfounderRiskLevel.MODERATE
        elif risk_score < 0.8:
            return ConfounderRiskLevel.HIGH
        else:
            return ConfounderRiskLevel.SEVERE
    
    def _get_risk_interpretation(self, risk_level: ConfounderRiskLevel) -> str:
        """获取风险等级的解释"""
        interpretations = {
            ConfounderRiskLevel.NONE: "无显著混杂风险，因果推断相对可靠",
            ConfounderRiskLevel.LOW: "低混杂风险，影响较小，可选调整",
            ConfounderRiskLevel.MODERATE: "中等混杂风险，建议进行调整以提高估计准确性",
            ConfounderRiskLevel.HIGH: "高混杂风险，强烈建议调整，否则估计可能严重偏误",
            ConfounderRiskLevel.SEVERE: "严重混杂风险，必须进行调整，否则因果推断不可靠"
        }
        return interpretations.get(risk_level, "未知风险等级")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return self.performance_stats.copy()
    
    def clear_cache(self):
        """清除检测缓存"""
        self.detection_cache.clear()
        logger.info("检测缓存已清除")


# 示例和测试函数
def create_example_detector() -> HiddenConfounderDetector:
    """创建示例混杂变量检测器"""
    detector = HiddenConfounderDetector(
        min_dependence_threshold=0.15,
        min_risk_threshold=0.25,
        max_confounders=5
    )
    return detector


def test_confounder_detection():
    """测试混杂变量检测器"""
    logger.info("开始测试潜在混杂变量检测器")
    
    # 创建示例检测器
    detector = create_example_detector()
    
    # 创建示例因果图
    graph = nx.DiGraph()
    
    # 添加节点
    nodes = ["smoking", "lung_cancer", "genetics", "age", "exercise", "health"]
    for node in nodes:
        graph.add_node(node)
    
    # 添加边（因果关系）
    edges = [
        ("genetics", "smoking"),
        ("genetics", "lung_cancer"),
        ("age", "smoking"),
        ("age", "lung_cancer"),
        ("smoking", "lung_cancer"),
        ("exercise", "health"),
        ("smoking", "health")
    ]
    
    for u, v in edges:
        graph.add_edge(u, v)
    
    # 创建示例观测数据
    np.random.seed(42)
    n_samples = 100
    
    observed_data = {
        "smoking": np.random.binomial(1, 0.3, n_samples),
        "lung_cancer": np.zeros(n_samples),
        "genetics": np.random.normal(0, 1, n_samples),
        "age": np.random.normal(50, 10, n_samples),
        "exercise": np.random.binomial(1, 0.5, n_samples),
        "health": np.random.normal(70, 15, n_samples)
    }
    
    # 模拟因果关系
    observed_data["lung_cancer"] = (
        0.5 * observed_data["smoking"] +
        0.3 * observed_data["genetics"] +
        0.1 * (observed_data["age"] - 50) / 10 +
        np.random.normal(0, 0.2, n_samples)
    )
    
    observed_data["health"] = (
        70 -
        5 * observed_data["smoking"] +
        10 * observed_data["exercise"] +
        np.random.normal(0, 5, n_samples)
    )
    
    # 测试混杂变量检测
    logger.info("测试混杂变量检测...")
    result = detector.detect_confounders(
        causal_graph=graph,
        treatment="smoking",
        outcome="lung_cancer",
        observed_data=observed_data,
        methods=[
            ConfounderDetectionMethod.BACKDOOR_CRITERION,
            ConfounderDetectionMethod.GRAPH_BASED,
            ConfounderDetectionMethod.DOUBLE_ROBUST
        ]
    )
    
    if result["success"]:
        logger.info(f"检测结果: 发现{result['confounders_detected']}个潜在混杂变量")
        logger.info(f"总体风险等级: {result['overall_risk']['risk_level']}")
        
        for i, assessment in enumerate(result["confounder_assessments"][:3]):
            logger.info(f"  混杂变量{i+1}: {assessment['confounder']}, 风险分数: {assessment['risk_score']:.3f}, 风险等级: {assessment['risk_level']}")
    
    # 显示性能统计
    stats = detector.get_performance_stats()
    logger.info(f"性能统计: {stats}")
    
    logger.info("潜在混杂变量检测器测试完成")
    return detector


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_detector = test_confounder_detection()