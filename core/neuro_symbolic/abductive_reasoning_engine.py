#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
溯因推理引擎 - 实现最佳解释推理(Abductive Reasoning)

核心功能:
1. 从观察结果溯因推导最佳解释
2. 假设生成和评估
3. 解释一致性和完整性检查
4. 多假设排序和选择
5. 解释与背景知识的整合

溯因推理公式:
给定观察O和背景知识K，寻找假设H使得：
1. H与K一致
2. H能够解释O (K ∪ H ⊨ O)
3. H是简单、合理、经济的

算法实现:
- 基于溯因逻辑编程(ALP)
- 假设空间搜索和剪枝
- 解释质量评估
- 增量假设优化

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
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

# 导入错误处理
from core.error_handling import ErrorHandler

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()


class ExplanationQuality(Enum):
    """解释质量等级枚举"""
    EXCELLENT = "excellent"      # 优秀: 完全解释，高度一致，非常简单
    GOOD = "good"                # 良好: 大部分解释，一致，较简单
    FAIR = "fair"                # 一般: 部分解释，基本一致，复杂度中等
    POOR = "poor"                # 较差: 有限解释，一致性有问题，复杂
    REJECTED = "rejected"        # 拒绝: 不一致或无法解释


class HypothesisSource(Enum):
    """假设来源枚举"""
    BACKGROUND_KNOWLEDGE = "background_knowledge"  # 背景知识
    ANALOGICAL_REASONING = "analogical_reasoning"  # 类比推理
    CREATIVE_GENERATION = "creative_generation"    # 创造性生成
    EXTERNAL_SOURCE = "external_source"            # 外部来源
    COMBINATION = "combination"                    # 组合生成


class AbductiveReasoningEngine:
    """
    溯因推理引擎 - 实现最佳解释推理
    
    核心组件:
    1. ObservationAnalyzer: 观察分析器
    2. HypothesisGenerator: 假设生成器
    3. ConsistencyChecker: 一致性检查器
    4. ExplanationEvaluator: 解释评估器
    5. HypothesisRanker: 假设排序器
    
    工作流程:
    观察 → ObservationAnalyzer → 观察特征 → HypothesisGenerator → 假设集合
    假设集合 → ConsistencyChecker → 一致假设 → ExplanationEvaluator → 评估结果
    评估结果 → HypothesisRanker → 排序假设 → 最佳解释
    
    技术特性:
    - 多假设生成和评估
    - 背景知识整合
    - 解释质量量化
    - 增量推理优化
    - 不确定性和概率处理
    """
    
    def __init__(self,
                 max_hypotheses: int = 10,
                 consistency_threshold: float = 0.7,
                 simplicity_weight: float = 0.3,
                 consistency_weight: float = 0.4,
                 completeness_weight: float = 0.3):
        """
        初始化溯因推理引擎
        
        Args:
            max_hypotheses: 最大假设数量
            consistency_threshold: 一致性阈值
            simplicity_weight: 简洁性权重
            consistency_weight: 一致性权重
            completeness_weight: 完备性权重
        """
        self.max_hypotheses = max_hypotheses
        self.consistency_threshold = consistency_threshold
        self.simplicity_weight = simplicity_weight
        self.consistency_weight = consistency_weight
        self.completeness_weight = completeness_weight
        
        # 知识库
        self.knowledge_base = {
            "facts": set(),          # 事实集合
            "rules": [],             # 规则列表
            "patterns": [],          # 模式列表
            "analogies": []          # 类比关系
        }
        
        # 假设缓存
        self.hypothesis_cache = {}
        
        # 性能统计
        self.performance_stats = {
            "abductions_performed": 0,
            "hypotheses_generated": 0,
            "explanations_evaluated": 0,
            "inconsistencies_detected": 0,
            "best_explanations_found": 0
        }
        
        # 初始化基础知识
        self._initialize_basic_knowledge()
        
        logger.info("溯因推理引擎初始化完成")
    
    def _initialize_basic_knowledge(self):
        """初始化基础知识"""
        # 基础溯因模式
        basic_patterns = [
            # 因果模式: 如果效果E发生，可能原因C
            {"pattern": "effect(E) -> possible_cause(C)", "confidence": 0.8},
            
            # 意图模式: 如果行动A发生，可能意图I
            {"pattern": "action(A) -> possible_intent(I)", "confidence": 0.7},
            
            # 故障模式: 如果系统异常，可能故障F
            {"pattern": "system_abnormal(S) -> possible_fault(F)", "confidence": 0.9},
            
            # 目标模式: 如果行为B发生，可能目标G
            {"pattern": "behavior(B) -> possible_goal(G)", "confidence": 0.6}
        ]
        
        for pattern in basic_patterns:
            self.knowledge_base["patterns"].append(pattern)
        
        # 基础类比关系
        basic_analogies = [
            {"source": "bird", "target": "airplane", "relation": "can_fly", "confidence": 0.7},
            {"source": "car", "target": "train", "relation": "transportation", "confidence": 0.8},
            {"source": "computer", "target": "brain", "relation": "information_processing", "confidence": 0.6}
        ]
        
        for analogy in basic_analogies:
            self.knowledge_base["analogies"].append(analogy)
        
        logger.info(f"初始化基础知识: {len(basic_patterns)}模式, {len(basic_analogies)}类比")
    
    def add_knowledge(self,
                     knowledge_type: str,
                     content: Any,
                     confidence: float = 1.0) -> bool:
        """
        添加知识到知识库
        
        Args:
            knowledge_type: 知识类型 ("fact", "rule", "pattern", "analogy")
            content: 知识内容
            confidence: 置信度
            
        Returns:
            是否成功添加
        """
        try:
            if knowledge_type == "fact":
                self.knowledge_base["facts"].add((str(content), confidence))
            elif knowledge_type == "rule":
                self.knowledge_base["rules"].append({"rule": content, "confidence": confidence})
            elif knowledge_type == "pattern":
                self.knowledge_base["patterns"].append({"pattern": content, "confidence": confidence})
            elif knowledge_type == "analogy":
                self.knowledge_base["analogies"].append({"analogy": content, "confidence": confidence})
            else:
                logger.warning(f"未知知识类型: {knowledge_type}")
                return False
            
            logger.debug(f"添加知识: {knowledge_type}, 置信度: {confidence}")
            return True
            
        except Exception as e:
            logger.error(f"添加知识失败: {knowledge_type}, 错误: {e}")
            return False
    
    def abduce(self,
               observations: List[str],
               context: Optional[Dict[str, Any]] = None,
               max_depth: int = 3) -> Dict[str, Any]:
        """
        执行溯因推理
        
        Args:
            observations: 观察列表
            context: 上下文信息
            max_depth: 最大推理深度
            
        Returns:
            溯因推理结果
        """
        start_time = time.time()
        
        # 分析观察
        observation_features = self._analyze_observations(observations, context)
        
        # 生成假设
        hypotheses = self._generate_hypotheses(observation_features, max_depth)
        
        # 评估假设
        evaluated_hypotheses = []
        for hypothesis in hypotheses:
            evaluation = self._evaluate_hypothesis(hypothesis, observations, context)
            evaluated_hypotheses.append({
                "hypothesis": hypothesis,
                "evaluation": evaluation
            })
        
        # 排序假设
        ranked_hypotheses = self._rank_hypotheses(evaluated_hypotheses)
        
        # 选择最佳解释
        best_explanation = None
        if ranked_hypotheses:
            best_hypothesis = ranked_hypotheses[0]
            best_explanation = self._construct_explanation(best_hypothesis, observations)
        
        elapsed_time = time.time() - start_time
        result = {
            "observations": observations,
            "context": context,
            "hypotheses_generated": len(hypotheses),
            "hypotheses_evaluated": len(evaluated_hypotheses),
            "ranked_hypotheses": ranked_hypotheses,
            "best_explanation": best_explanation,
            "performance": {
                "reasoning_time": elapsed_time,
                "max_depth": max_depth,
                "cache_hits": 0  # 简化，实际应统计缓存命中
            }
        }
        
        self.performance_stats["abductions_performed"] += 1
        self.performance_stats["hypotheses_generated"] += len(hypotheses)
        self.performance_stats["explanations_evaluated"] += len(evaluated_hypotheses)
        
        if best_explanation:
            self.performance_stats["best_explanations_found"] += 1
        
        logger.info(f"溯因推理完成: {len(observations)}观察 → {len(hypotheses)}假设 → {len(ranked_hypotheses)}排序")
        
        return result
    
    def _analyze_observations(self,
                             observations: List[str],
                             context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析观察
        
        Args:
            observations: 观察列表
            context: 上下文信息
            
        Returns:
            观察特征
        """
        features = {
            "observation_count": len(observations),
            "observation_types": defaultdict(int),
            "keywords": [],
            "temporal_pattern": None,
            "spatial_pattern": None,
            "causal_indicators": []
        }
        
        # 分析每个观察
        for obs in observations:
            # 提取关键词
            words = obs.lower().split()
            features["keywords"].extend(words)
            
            # 分类观察类型（简化）
            if any(word in obs.lower() for word in ["break", "damage", "fail"]):
                features["observation_types"]["failure"] += 1
                features["causal_indicators"].append("failure_event")
            elif any(word in obs.lower() for word in ["move", "go", "run"]):
                features["observation_types"]["action"] += 1
            elif any(word in obs.lower() for word in ["want", "need", "desire"]):
                features["observation_types"]["desire"] += 1
            elif any(word in obs.lower() for word in ["know", "think", "believe"]):
                features["observation_types"]["belief"] += 1
            else:
                features["observation_types"]["other"] += 1
        
        # 去除重复关键词
        features["keywords"] = list(set(features["keywords"]))
        
        # 添加上下文特征
        if context:
            features["context_present"] = True
            features["context_keys"] = list(context.keys())
        else:
            features["context_present"] = False
        
        return features
    
    def _generate_hypotheses(self,
                            observation_features: Dict[str, Any],
                            max_depth: int) -> List[Dict[str, Any]]:
        """
        生成假设
        
        Args:
            observation_features: 观察特征
            max_depth: 最大深度
            
        Returns:
            假设列表
        """
        hypotheses = []
        
        # 方法1: 基于模式的假设生成
        pattern_based = self._generate_pattern_based_hypotheses(observation_features)
        hypotheses.extend(pattern_based)
        
        # 方法2: 基于类比的假设生成
        analogy_based = self._generate_analogy_based_hypotheses(observation_features)
        hypotheses.extend(analogy_based)
        
        # 方法3: 基于规则的假设生成
        rule_based = self._generate_rule_based_hypotheses(observation_features)
        hypotheses.extend(rule_based)
        
        # 方法4: 组合假设生成
        if len(hypotheses) > 1 and max_depth > 1:
            combined = self._generate_combined_hypotheses(hypotheses, max_depth - 1)
            hypotheses.extend(combined)
        
        # 去重和限制数量
        unique_hypotheses = []
        seen_contents = set()
        
        for hyp in hypotheses:
            content_key = str(hyp.get("content", ""))
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                unique_hypotheses.append(hyp)
        
        # 限制数量
        hypotheses = unique_hypotheses[:self.max_hypotheses]
        
        logger.debug(f"生成假设: {len(hypotheses)}个唯一假设")
        return hypotheses
    
    def _generate_pattern_based_hypotheses(self,
                                          observation_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于模式生成假设"""
        hypotheses = []
        
        for pattern_info in self.knowledge_base["patterns"]:
            pattern = pattern_info.get("pattern", "")
            confidence = pattern_info.get("confidence", 0.5)
            
            # 简化：检查模式是否匹配观察特征
            if self._pattern_matches_observations(pattern, observation_features):
                hypothesis = {
                    "content": f"Pattern: {pattern}",
                    "source": HypothesisSource.BACKGROUND_KNOWLEDGE.value,
                    "generation_method": "pattern_matching",
                    "confidence": confidence * 0.8,  # 模式匹配的折扣
                    "simplicity_score": 0.7,  # 模式通常较简单
                    "explanation_power": 0.6  # 解释能力中等
                }
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _pattern_matches_observations(self,
                                     pattern: str,
                                     observation_features: Dict[str, Any]) -> bool:
        """检查模式是否匹配观察"""
        # 简化实现：检查关键词匹配
        keywords = observation_features.get("keywords", [])
        
        # 提取模式中的关键词
        pattern_keywords = []
        for word in pattern.lower().split():
            if len(word) > 3 and word not in ["->", "possible", "effect", "cause", "action", "intent"]:
                pattern_keywords.append(word)
        
        # 检查是否有匹配的关键词
        if pattern_keywords:
            for keyword in keywords:
                if any(pk in keyword or keyword in pk for pk in pattern_keywords):
                    return True
        
        # 检查观察类型匹配
        obs_types = observation_features.get("observation_types", {})
        if "failure" in obs_types and "failure" in pattern.lower():
            return True
        if "action" in obs_types and "action" in pattern.lower():
            return True
        
        return False
    
    def _generate_analogy_based_hypotheses(self,
                                          observation_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于类比生成假设"""
        hypotheses = []
        
        for analogy_info in self.knowledge_base["analogies"]:
            # 简化：使用预定义的类比
            if isinstance(analogy_info, dict) and "source" in analogy_info:
                source = analogy_info.get("source", "")
                target = analogy_info.get("target", "")
                relation = analogy_info.get("relation", "")
                confidence = analogy_info.get("confidence", 0.5)
                
                # 检查观察是否与源域相关
                keywords = observation_features.get("keywords", [])
                if any(source_kw in ' '.join(keywords) for source_kw in [source, relation]):
                    hypothesis = {
                        "content": f"Analogy: {source} → {target} ({relation})",
                        "source": HypothesisSource.ANALOGICAL_REASONING.value,
                        "generation_method": "analogical_transfer",
                        "confidence": confidence * 0.7,  # 类比推理的折扣
                        "simplicity_score": 0.5,  # 类比可能较复杂
                        "explanation_power": 0.7  # 类比解释能力较强
                    }
                    hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_rule_based_hypotheses(self,
                                       observation_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于规则生成假设"""
        hypotheses = []
        
        for rule_info in self.knowledge_base["rules"]:
            rule = rule_info.get("rule", "")
            confidence = rule_info.get("confidence", 0.5)
            
            # 简化：检查规则前提是否与观察相关
            if self._rule_applicable(rule, observation_features):
                hypothesis = {
                    "content": f"Rule: {rule}",
                    "source": HypothesisSource.BACKGROUND_KNOWLEDGE.value,
                    "generation_method": "rule_application",
                    "confidence": confidence * 0.9,  # 规则应用的折扣较小
                    "simplicity_score": 0.6,  # 规则复杂度中等
                    "explanation_power": 0.8  # 规则解释能力较强
                }
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _rule_applicable(self, rule: str, observation_features: Dict[str, Any]) -> bool:
        """检查规则是否适用"""
        # 简化实现：检查规则关键词是否出现在观察中
        keywords = observation_features.get("keywords", [])
        
        # 提取规则中的关键词（忽略逻辑连接词）
        rule_keywords = []
        for word in rule.lower().split():
            if len(word) > 2 and word not in ["if", "then", "and", "or", "not", "forall", "exists"]:
                rule_keywords.append(word)
        
        # 检查关键词匹配
        if rule_keywords:
            for keyword in keywords:
                if any(rk in keyword or keyword in rk for rk in rule_keywords):
                    return True
        
        return False
    
    def _generate_combined_hypotheses(self,
                                     base_hypotheses: List[Dict[str, Any]],
                                     depth: int) -> List[Dict[str, Any]]:
        """生成组合假设"""
        if depth <= 0 or len(base_hypotheses) < 2:
            return []
        
        combined_hypotheses = []
        
        # 生成两两组合
        for i in range(len(base_hypotheses)):
            for j in range(i + 1, len(base_hypotheses)):
                hyp1 = base_hypotheses[i]
                hyp2 = base_hypotheses[j]
                
                # 组合假设内容
                content1 = hyp1.get("content", "")
                content2 = hyp2.get("content", "")
                combined_content = f"Combination: ({content1}) AND ({content2})"
                
                # 计算组合置信度（几何平均）
                conf1 = hyp1.get("confidence", 0.5)
                conf2 = hyp2.get("confidence", 0.5)
                combined_confidence = math.sqrt(conf1 * conf2) * 0.8  # 组合折扣
                
                # 计算简洁性（组合通常更复杂）
                simplicity1 = hyp1.get("simplicity_score", 0.5)
                simplicity2 = hyp2.get("simplicity_score", 0.5)
                combined_simplicity = (simplicity1 + simplicity2) / 2 * 0.7
                
                # 计算解释能力（组合可能更强）
                power1 = hyp1.get("explanation_power", 0.5)
                power2 = hyp2.get("explanation_power", 0.5)
                combined_power = min(1.0, (power1 + power2) * 0.8)
                
                hypothesis = {
                    "content": combined_content,
                    "source": HypothesisSource.COMBINATION.value,
                    "generation_method": "hypothesis_combination",
                    "confidence": combined_confidence,
                    "simplicity_score": combined_simplicity,
                    "explanation_power": combined_power
                }
                combined_hypotheses.append(hypothesis)
        
        return combined_hypotheses
    
    def _evaluate_hypothesis(self,
                            hypothesis: Dict[str, Any],
                            observations: List[str],
                            context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估假设
        
        Args:
            hypothesis: 假设
            observations: 观察列表
            context: 上下文信息
            
        Returns:
            评估结果
        """
        # 计算一致性
        consistency_score = self._calculate_consistency(hypothesis, context)
        
        # 计算完备性（解释观察的能力）
        completeness_score = self._calculate_completeness(hypothesis, observations)
        
        # 计算简洁性
        simplicity_score = hypothesis.get("simplicity_score", 0.5)
        
        # 计算总体质量分数
        quality_score = (
            self.simplicity_weight * simplicity_score +
            self.consistency_weight * consistency_score +
            self.completeness_weight * completeness_score
        )
        
        # 确定质量等级
        quality_level = self._determine_quality_level(quality_score, consistency_score)
        
        # 检查是否与知识一致
        consistent_with_knowledge = consistency_score >= self.consistency_threshold
        
        evaluation = {
            "consistency_score": consistency_score,
            "completeness_score": completeness_score,
            "simplicity_score": simplicity_score,
            "quality_score": quality_score,
            "quality_level": quality_level.value if hasattr(quality_level, 'value') else str(quality_level),
            "consistent_with_knowledge": consistent_with_knowledge,
            "explanation_power": hypothesis.get("explanation_power", 0.5),
            "confidence": hypothesis.get("confidence", 0.5)
        }
        
        if not consistent_with_knowledge:
            self.performance_stats["inconsistencies_detected"] += 1
        
        return evaluation
    
    def _calculate_consistency(self,
                              hypothesis: Dict[str, Any],
                              context: Optional[Dict[str, Any]]) -> float:
        """计算假设一致性"""
        # 简化实现：基于假设来源和置信度
        
        source = hypothesis.get("source", "")
        confidence = hypothesis.get("confidence", 0.5)
        
        # 不同来源的一致性基准
        consistency_baseline = {
            HypothesisSource.BACKGROUND_KNOWLEDGE.value: 0.8,
            HypothesisSource.ANALOGICAL_REASONING.value: 0.6,
            HypothesisSource.CREATIVE_GENERATION.value: 0.4,
            HypothesisSource.EXTERNAL_SOURCE.value: 0.5,
            HypothesisSource.COMBINATION.value: 0.7
        }
        
        baseline = consistency_baseline.get(source, 0.5)
        
        # 结合置信度
        consistency = baseline * confidence
        
        # 上下文一致性（简化）
        if context:
            # 检查假设是否与上下文冲突
            hypothesis_content = str(hypothesis.get("content", "")).lower()
            context_str = str(context).lower()
            
            # 简单关键词冲突检查
            conflict_keywords = ["not", "never", "cannot", "impossible"]
            for keyword in conflict_keywords:
                if keyword in hypothesis_content and keyword in context_str:
                    consistency *= 0.5  # 冲突惩罚
        
        return min(1.0, consistency)
    
    def _calculate_completeness(self,
                               hypothesis: Dict[str, Any],
                               observations: List[str]) -> float:
        """计算假设完备性（解释观察的能力）"""
        if not observations:
            return 0.0
        
        hypothesis_content = str(hypothesis.get("content", "")).lower()
        explained_count = 0
        
        for obs in observations:
            obs_lower = obs.lower()
            
            # 简化：检查假设内容是否包含观察关键词
            obs_words = set(obs_lower.split())
            hyp_words = set(hypothesis_content.split())
            
            # 计算重叠度
            overlap = len(obs_words.intersection(hyp_words))
            if overlap > 0:
                explained_count += min(1.0, overlap / max(1, len(obs_words)))
        
        completeness = explained_count / len(observations)
        
        # 考虑假设的解释能力
        explanation_power = hypothesis.get("explanation_power", 0.5)
        completeness = min(1.0, completeness * (0.5 + 0.5 * explanation_power))
        
        return completeness
    
    def _determine_quality_level(self,
                                quality_score: float,
                                consistency_score: float) -> ExplanationQuality:
        """确定解释质量等级"""
        if consistency_score < self.consistency_threshold:
            return ExplanationQuality.REJECTED
        elif quality_score >= 0.8:
            return ExplanationQuality.EXCELLENT
        elif quality_score >= 0.6:
            return ExplanationQuality.GOOD
        elif quality_score >= 0.4:
            return ExplanationQuality.FAIR
        else:
            return ExplanationQuality.POOR
    
    def _rank_hypotheses(self,
                        evaluated_hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """排序假设"""
        if not evaluated_hypotheses:
            return []
        
        # 计算排序分数
        ranked = []
        for item in evaluated_hypotheses:
            hypothesis = item["hypothesis"]
            evaluation = item["evaluation"]
            
            # 跳过不一致的假设
            if not evaluation.get("consistent_with_knowledge", True):
                continue
            
            # 计算排序分数（加权质量分数）
            quality_score = evaluation.get("quality_score", 0)
            confidence = hypothesis.get("confidence", 0.5)
            explanation_power = hypothesis.get("explanation_power", 0.5)
            
            # 排序分数 = 质量分数 * 置信度 * 解释能力
            rank_score = quality_score * confidence * explanation_power
            
            ranked.append({
                "hypothesis": hypothesis,
                "evaluation": evaluation,
                "rank_score": rank_score
            })
        
        # 按排序分数降序排序
        ranked.sort(key=lambda x: x["rank_score"], reverse=True)
        
        return ranked
    
    def _construct_explanation(self,
                              best_hypothesis_item: Dict[str, Any],
                              observations: List[str]) -> Dict[str, Any]:
        """构建最佳解释"""
        hypothesis = best_hypothesis_item["hypothesis"]
        evaluation = best_hypothesis_item["evaluation"]
        
        explanation = {
            "hypothesis_content": hypothesis.get("content", ""),
            "hypothesis_source": hypothesis.get("source", ""),
            "confidence": hypothesis.get("confidence", 0.5),
            "quality_level": evaluation.get("quality_level", ""),
            "quality_score": evaluation.get("quality_score", 0),
            "consistency_score": evaluation.get("consistency_score", 0),
            "completeness_score": evaluation.get("completeness_score", 0),
            "simplicity_score": evaluation.get("simplicity_score", 0),
            "explanation_power": hypothesis.get("explanation_power", 0.5),
            "observations_explained": observations,
            "explanation_text": self._generate_explanation_text(hypothesis, observations, evaluation)
        }
        
        return explanation
    
    def _generate_explanation_text(self,
                                  hypothesis: Dict[str, Any],
                                  observations: List[str],
                                  evaluation: Dict[str, Any]) -> str:
        """生成解释文本"""
        hypothesis_content = hypothesis.get("content", "")
        quality_level = evaluation.get("quality_level", "")
        confidence = hypothesis.get("confidence", 0.5)
        
        # 根据质量等级生成不同风格的文本
        if quality_level == ExplanationQuality.EXCELLENT.value:
            confidence_text = "highly confident"
        elif quality_level == ExplanationQuality.GOOD.value:
            confidence_text = "confident"
        elif quality_level == ExplanationQuality.FAIR.value:
            confidence_text = "somewhat confident"
        else:
            confidence_text = "tentatively"
        
        explanation = f"Based on the observations, the best explanation is: {hypothesis_content}. "
        explanation += f"This explanation is {confidence_text} (confidence: {confidence:.2f}) "
        explanation += f"and provides a {quality_level} level of explanation."
        
        return explanation
    
    def incremental_abduction(self,
                             initial_observations: List[str],
                             new_observations: List[str],
                             previous_explanation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        增量溯因推理
        
        Args:
            initial_observations: 初始观察
            new_observations: 新观察
            previous_explanation: 先前解释
            
        Returns:
            增量推理结果
        """
        start_time = time.time()
        
        # 合并观察
        all_observations = initial_observations + new_observations
        
        # 如果有先前解释，检查是否仍然有效
        if previous_explanation:
            # 评估先前解释对新观察的完备性
            previous_hypothesis = {
                "content": previous_explanation.get("hypothesis_content", ""),
                "confidence": previous_explanation.get("confidence", 0.5)
            }
            
            new_completeness = self._calculate_completeness(previous_hypothesis, new_observations)
            
            if new_completeness >= 0.7:  # 阈值
                # 先前解释仍然有效，更新置信度
                updated_confidence = previous_explanation.get("confidence", 0.5) * (0.9 + 0.1 * new_completeness)
                updated_explanation = previous_explanation.copy()
                updated_explanation["confidence"] = min(1.0, updated_confidence)
                updated_explanation["incremental_update"] = True
                
                elapsed_time = time.time() - start_time
                return {
                    "success": True,
                    "explanation": updated_explanation,
                    "previous_explanation_reused": True,
                    "new_completeness": new_completeness,
                    "performance": {
                        "reasoning_time": elapsed_time,
                        "method": "incremental_reuse"
                    }
                }
        
        # 需要重新推理
        context = {
            "previous_observations": initial_observations,
            "previous_explanation": previous_explanation
        }
        
        result = self.abduce(all_observations, context)
        result["incremental_update"] = True
        result["previous_explanation_reused"] = False
        
        return result
    
    def explain_contradiction(self,
                             observation1: str,
                             observation2: str) -> Dict[str, Any]:
        """
        解释矛盾
        
        Args:
            observation1: 第一个观察
            observation2: 第二个观察
            
        Returns:
            矛盾解释结果
        """
        start_time = time.time()
        
        # 检查是否确实矛盾
        is_contradiction = self._check_contradiction(observation1, observation2)
        
        if not is_contradiction:
            return {
                "success": False,
                "is_contradiction": False,
                "message": "The observations do not contradict each other"
            }
        
        # 生成矛盾解释假设
        contradiction_hypotheses = [
            {
                "content": "One observation is mistaken or inaccurate",
                "source": HypothesisSource.BACKGROUND_KNOWLEDGE.value,
                "confidence": 0.6,
                "simplicity_score": 0.8,
                "explanation_power": 0.7
            },
            {
                "content": "Observations are from different time points",
                "source": HypothesisSource.BACKGROUND_KNOWLEDGE.value,
                "confidence": 0.5,
                "simplicity_score": 0.7,
                "explanation_power": 0.6
            },
            {
                "content": "Different perspectives or contexts",
                "source": HypothesisSource.CREATIVE_GENERATION.value,
                "confidence": 0.4,
                "simplicity_score": 0.6,
                "explanation_power": 0.8
            }
        ]
        
        # 评估假设
        observations = [observation1, observation2]
        evaluated_hypotheses = []
        
        for hypothesis in contradiction_hypotheses:
            evaluation = self._evaluate_hypothesis(hypothesis, observations, None)
            evaluated_hypotheses.append({
                "hypothesis": hypothesis,
                "evaluation": evaluation
            })
        
        # 排序假设
        ranked_hypotheses = self._rank_hypotheses(evaluated_hypotheses)
        
        # 最佳解释
        best_explanation = None
        if ranked_hypotheses:
            best_hypothesis = ranked_hypotheses[0]
            best_explanation = self._construct_explanation(best_hypothesis, observations)
        
        elapsed_time = time.time() - start_time
        result = {
            "success": True,
            "is_contradiction": True,
            "observation1": observation1,
            "observation2": observation2,
            "contradiction_type": "direct",  # 简化
            "ranked_explanations": ranked_hypotheses,
            "best_explanation": best_explanation,
            "performance": {
                "reasoning_time": elapsed_time
            }
        }
        
        return result
    
    def _check_contradiction(self, observation1: str, observation2: str) -> bool:
        """检查两个观察是否矛盾"""
        # 简化实现：检查否定词
        
        obs1_lower = observation1.lower()
        obs2_lower = observation2.lower()
        
        # 否定词列表
        negation_words = ["not", "no", "never", "cannot", "won't", "don't"]
        
        # 检查是否有相反的描述
        for neg_word in negation_words:
            if (neg_word in obs1_lower and neg_word not in obs2_lower) or \
               (neg_word in obs2_lower and neg_word not in obs1_lower):
                # 检查是否描述同一事物
                # 提取关键词（忽略否定词）
                words1 = [w for w in obs1_lower.split() if w not in negation_words and len(w) > 2]
                words2 = [w for w in obs2_lower.split() if w not in negation_words and len(w) > 2]
                
                # 检查是否有重叠关键词
                overlap = len(set(words1).intersection(set(words2)))
                if overlap > 0:
                    return True
        
        return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = self.performance_stats.copy()
        stats["knowledge_base_stats"] = {
            "facts_count": len(self.knowledge_base["facts"]),
            "rules_count": len(self.knowledge_base["rules"]),
            "patterns_count": len(self.knowledge_base["patterns"]),
            "analogies_count": len(self.knowledge_base["analogies"]),
            "hypothesis_cache_size": len(self.hypothesis_cache)
        }
        return stats
    
    def clear_cache(self):
        """清除假设缓存"""
        self.hypothesis_cache.clear()
        logger.info("假设缓存已清除")
    
    def save_knowledge_base(self, filepath: str) -> bool:
        """保存知识库到文件"""
        try:
            import pickle
            kb_data = {
                "knowledge_base": self.knowledge_base,
                "performance_stats": self.performance_stats,
                "hypothesis_cache": self.hypothesis_cache
            }
            with open(filepath, 'wb') as f:
                pickle.dump(kb_data, f)
            logger.info(f"知识库保存到: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存知识库失败: {e}")
            return False
    
    def load_knowledge_base(self, filepath: str) -> bool:
        """从文件加载知识库"""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                kb_data = pickle.load(f)
            
            self.knowledge_base = kb_data.get("knowledge_base", self.knowledge_base)
            self.performance_stats = kb_data.get("performance_stats", self.performance_stats.copy())
            self.hypothesis_cache = kb_data.get("hypothesis_cache", {})
            
            logger.info(f"知识库从 {filepath} 加载")
            return True
        except Exception as e:
            logger.error(f"加载知识库失败: {e}")
            return False


# 示例和测试函数
def create_example_abduction_engine() -> AbductiveReasoningEngine:
    """创建示例溯因推理引擎"""
    engine = AbductiveReasoningEngine(
        max_hypotheses=5,
        consistency_threshold=0.6,
        simplicity_weight=0.3,
        consistency_weight=0.4,
        completeness_weight=0.3
    )
    
    # 添加示例知识
    example_facts = [
        "The car won't start",
        "The battery is old",
        "Batteries degrade over time",
        "Old batteries may not hold charge"
    ]
    
    for fact in example_facts:
        engine.add_knowledge("fact", fact, 0.8)
    
    example_rules = [
        "If battery is old and car won't start, then possible cause is dead battery",
        "If lights are dim and car won't start, then possible cause is weak battery",
        "If engine cranks slowly and car won't start, then possible cause is bad starter"
    ]
    
    for rule in example_rules:
        engine.add_knowledge("rule", rule, 0.7)
    
    return engine


def test_abductive_reasoning():
    """测试溯因推理引擎"""
    logger.info("开始测试溯因推理引擎")
    
    # 创建示例引擎
    engine = create_example_abduction_engine()
    
    # 测试溯因推理
    observations = ["The car won't start", "The battery is old"]
    logger.info(f"观察: {observations}")
    
    result = engine.abduce(observations)
    
    if result.get("best_explanation"):
        best_explanation = result["best_explanation"]
        logger.info(f"最佳解释: {best_explanation['hypothesis_content']}")
        logger.info(f"质量等级: {best_explanation['quality_level']}, 置信度: {best_explanation['confidence']:.2f}")
    else:
        logger.info("未找到满意解释")
    
    # 测试增量溯因
    logger.info("测试增量溯因...")
    initial_observations = ["The car won't start"]
    new_observations = ["The lights are dim"]
    
    incremental_result = engine.incremental_abduction(initial_observations, new_observations, result.get("best_explanation"))
    logger.info(f"增量推理: 重用先前解释={incremental_result.get('previous_explanation_reused', False)}")
    
    # 测试矛盾解释
    logger.info("测试矛盾解释...")
    contradiction_result = engine.explain_contradiction(
        "The car is working perfectly",
        "The car won't start"
    )
    
    if contradiction_result.get("success") and contradiction_result.get("is_contradiction"):
        if contradiction_result.get("best_explanation"):
            logger.info(f"矛盾解释: {contradiction_result['best_explanation']['hypothesis_content']}")
    
    # 显示性能统计
    stats = engine.get_performance_stats()
    logger.info(f"性能统计: {stats}")
    
    logger.info("溯因推理引擎测试完成")
    return engine


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_engine = test_abductive_reasoning()