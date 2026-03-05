#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zlib
"""
Cross-Domain Planner - 跨领域规划器

核心功能：
1. 领域知识整合和迁移
2. 跨领域策略学习和应用
3. 多领域约束处理和优化
4. 元策略学习和自适应

实现多领域知识整合和策略迁移的高级规划能力，
支持复杂跨领域问题的解决。

Copyright (c) 2025 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
import json
import math
import random
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from enum import Enum
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field, asdict
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# 导入错误处理
from core.error_handling import ErrorHandler

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()

class DomainCategory(Enum):
    """领域分类枚举"""
    GENERAL_KNOWLEDGE = "general_knowledge"        # 通用知识
    SCIENCE_TECHNOLOGY = "science_technology"      # 科学技术
    BUSINESS_ECONOMICS = "business_economics"      # 商业经济
    SOCIAL_SCIENCES = "social_sciences"            # 社会科学
    ARTS_HUMANITIES = "arts_humanities"            # 艺术人文
    ENGINEERING = "engineering"                    # 工程
    MEDICINE_HEALTH = "medicine_health"            # 医学健康
    LAW_GOVERNMENT = "law_government"              # 法律政府
    EDUCATION = "education"                        # 教育
    ENVIRONMENT_ENERGY = "environment_energy"      # 环境能源

class TransferLearningType(Enum):
    """迁移学习类型枚举"""
    INSTANCE_TRANSFER = "instance_transfer"        # 实例迁移
    FEATURE_TRANSFER = "feature_transfer"          # 特征迁移
    PARAMETER_TRANSFER = "parameter_transfer"      # 参数迁移
    RELATIONAL_TRANSFER = "relational_transfer"    # 关系迁移
    MULTI_TASK_LEARNING = "multi_task_learning"    # 多任务学习

@dataclass
class DomainKnowledge:
    """领域知识表示"""
    domain_id: str
    domain_name: str
    category: DomainCategory
    knowledge_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    key_concepts: List[str] = field(default_factory=list)
    key_relations: List[Tuple[str, str, str]] = field(default_factory=list)
    strategies: List[Dict[str, Any]] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    resources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class StrategyPattern:
    """策略模式表示"""
    pattern_id: str
    source_domain: str
    target_domain: str
    pattern_type: str
    effectiveness: float
    applicability_score: float
    transfer_cost: float
    conditions: Dict[str, Any]
    implementation: Dict[str, Any]
    examples: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class CrossDomainPlan:
    """跨领域计划表示"""
    plan_id: str
    goal: Any
    source_domains: List[str]
    target_domain: str
    integrated_strategies: List[StrategyPattern]
    cross_domain_constraints: Dict[str, Any]
    adaptation_rules: List[Dict[str, Any]]
    feasibility_score: float
    estimated_resources: Dict[str, Any]
    temporal_aspects: Dict[str, Any] = field(default_factory=dict)

class CrossDomainPlanner:
    """
    跨领域规划器 - 整合多领域知识进行高级规划
    
    核心特性：
    1. 领域知识整合和融合
    2. 跨领域策略迁移学习
    3. 多领域约束处理和优化
    4. 自适应跨领域适配
    5. 元策略学习和优化
    6. 动态领域边界扩展
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化跨领域规划器"""
        self.logger = logger
        self.config = config or self._get_default_config()
        
        # 初始化组件
        self._initialize_components()
        
        # 领域知识库
        self.domain_knowledge_base: Dict[str, DomainKnowledge] = {}
        
        # 策略模式库
        self.strategy_pattern_library: Dict[str, StrategyPattern] = {}
        
        # 跨领域映射
        self.domain_similarity_matrix: Optional[np.ndarray] = None
        self.domain_mapping_rules: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        
        # 状态跟踪
        self.state = {
            "total_cross_domain_plans": 0,
            "successful_cross_domain_plans": 0,
            "average_cross_domain_transfer_time": 0,
            "domain_knowledge_count": 0,
            "strategy_pattern_count": 0,
            "successful_transfers": 0,
            "failed_transfers": 0,
            "adaptation_attempts": 0,
            "adaptation_successes": 0
        }
        
        # 缓存
        self.domain_knowledge_cache: Dict[str, Dict[str, Any]] = {}
        self.strategy_transfer_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        
        logger.info("跨领域规划器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "domain_integration": {
                "enable_automatic_domain_discovery": True,
                "domain_similarity_threshold": 0.7,
                "max_domains_to_integrate": 3,
                "enable_dynamic_domain_boundary": True,
                "domain_expansion_rate": 0.1
            },
            "strategy_transfer": {
                "enable_strategy_transfer": True,
                "transfer_learning_type": TransferLearningType.FEATURE_TRANSFER,
                "minimum_effectiveness_threshold": 0.6,
                "maximum_transfer_cost": 0.3,
                "enable_adaptive_transfer": True
            },
            "constraint_handling": {
                "enable_constraint_propagation": True,
                "constraint_relaxation_factor": 0.1,
                "max_constraint_iterations": 10,
                "enable_constraint_learning": True
            },
            "adaptation": {
                "enable_automatic_adaptation": True,
                "adaptation_iterations": 3,
                "adaptation_success_threshold": 0.8,
                "enable_meta_learning": True
            },
            "optimization": {
                "enable_multi_objective_optimization": True,
                "optimization_iterations": 5,
                "pareto_front_size": 10,
                "enable_evolutionary_optimization": False
            }
        }
    
    def _initialize_components(self) -> None:
        """初始化所有组件"""
        try:
            # 1. 领域知识整合器
            self.domain_knowledge_integrator = DomainKnowledgeIntegrator()
            
            # 2. 策略迁移器
            self.strategy_transferor = StrategyTransferor()
            
            # 3. 跨领域适配器
            self.cross_domain_adapter = CrossDomainAdapter()
            
            # 4. 元策略学习器
            self.meta_strategy_learner = MetaStrategyLearner()
            
            logger.info("跨领域规划器组件初始化完成")
            
        except Exception as e:
            error_handler.handle_error(e, "CrossDomainPlanner", "初始化组件失败")
            logger.error(f"组件初始化失败: {e}")
    
    def plan_across_domains(self, goal: Any, 
                           target_domain: str,
                           context: Optional[Dict[str, Any]] = None,
                           available_domains: Optional[List[str]] = None,
                           constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        跨领域规划 - 整合多领域知识进行规划
        
        Args:
            goal: 规划目标
            target_domain: 目标领域
            context: 上下文信息
            available_domains: 可用的源领域列表
            constraints: 跨领域约束
            
        Returns:
            跨领域规划结果
        """
        start_time = time.time()
        self.state["total_cross_domain_plans"] += 1
        
        try:
            # 1. 分析目标和上下文
            goal_analysis = self._analyze_cross_domain_goal(goal, target_domain, context)
            
            # 2. 确定相关的源领域
            relevant_domains = self._identify_relevant_domains(goal_analysis, target_domain, available_domains)
            
            # 3. 提取和整合领域知识
            integrated_knowledge = self._integrate_domain_knowledge(relevant_domains, goal_analysis)
            
            # 4. 发现和迁移策略
            transferable_strategies = self._discover_and_transfer_strategies(
                relevant_domains, target_domain, goal_analysis
            )
            
            # 5. 处理跨领域约束
            processed_constraints = self._process_cross_domain_constraints(
                constraints, relevant_domains, target_domain
            )
            
            # 6. 生成跨领域候选计划
            candidate_plans = self._generate_cross_domain_candidate_plans(
                goal_analysis, integrated_knowledge, transferable_strategies, processed_constraints
            )
            
            # 7. 评估和优化候选计划
            optimized_plan = self._evaluate_and_optimize_cross_domain_plans(
                candidate_plans, goal_analysis, integrated_knowledge
            )
            
            # 8. 适配到目标领域
            adapted_plan = self._adapt_plan_to_target_domain(
                optimized_plan, target_domain, relevant_domains
            )
            
            # 9. 生成最终结果
            final_result = self._generate_cross_domain_final_result(
                adapted_plan, goal_analysis, relevant_domains, transferable_strategies
            )
            
            # 10. 记录学习经验
            self._record_cross_domain_learning_experience(
                goal_analysis, relevant_domains, final_result
            )
            
            # 更新状态
            planning_time = time.time() - start_time
            self.state["average_cross_domain_transfer_time"] = (
                self.state["average_cross_domain_transfer_time"] * 
                (self.state["total_cross_domain_plans"] - 1) + planning_time
            ) / self.state["total_cross_domain_plans"]
            
            if final_result.get("success", False):
                self.state["successful_cross_domain_plans"] += 1
            
            logger.info(
                f"跨领域规划完成: 目标='{goal}', "
                f"目标领域='{target_domain}', "
                f"源领域数={len(relevant_domains)}, "
                f"时间={planning_time:.2f}s, "
                f"成功={final_result.get('success', False)}"
            )
            
            return final_result
            
        except Exception as e:
            error_handler.handle_error(e, "CrossDomainPlanner", "跨领域规划过程失败")
            return self._generate_cross_domain_error_result(goal, target_domain, str(e))
    
    def integrate_domain_knowledge(self, domain_data: Dict[str, Any], 
                                  domain_name: str,
                                  category: DomainCategory) -> Dict[str, Any]:
        """
        整合领域知识
        
        Args:
            domain_data: 领域数据
            domain_name: 领域名称
            category: 领域分类
            
        Returns:
            知识整合结果
        """
        try:
            result = self.domain_knowledge_integrator.integrate(
                domain_data, domain_name, category
            )
            
            # 存储到知识库
            domain_knowledge = DomainKnowledge(
                domain_id=f"domain_{int(time.time())}_{(zlib.adler32(str(domain_name).encode('utf-8')) & 0xffffffff)}",
                domain_name=domain_name,
                category=category,
                knowledge_graph=result.get("knowledge_graph", nx.DiGraph()),
                key_concepts=result.get("key_concepts", []),
                key_relations=result.get("key_relations", []),
                strategies=result.get("strategies", []),
                constraints=result.get("constraints", {}),
                resources=result.get("resources", []),
                metadata={
                    "integration_time": time.time(),
                    "data_size": len(str(domain_data)),
                    "concept_count": len(result.get("key_concepts", [])),
                    "relation_count": len(result.get("key_relations", []))
                }
            )
            
            self.domain_knowledge_base[domain_knowledge.domain_id] = domain_knowledge
            self.state["domain_knowledge_count"] = len(self.domain_knowledge_base)
            
            logger.info(f"领域知识整合完成: 领域='{domain_name}', 概念数={len(domain_knowledge.key_concepts)}")
            
            return {
                "success": True,
                "domain_id": domain_knowledge.domain_id,
                "domain_name": domain_name,
                "category": category.value,
                "concept_count": len(domain_knowledge.key_concepts),
                "relation_count": len(domain_knowledge.key_relations),
                "strategy_count": len(domain_knowledge.strategies),
                "knowledge_graph_size": domain_knowledge.knowledge_graph.number_of_nodes()
            }
            
        except Exception as e:
            error_handler.handle_error(e, "CrossDomainPlanner", "领域知识整合失败")
            return {
                "success": False,
                "error": f"领域知识整合失败: {str(e)}",
                "domain_name": domain_name
            }
    
    def transfer_strategy(self, source_domain: str, 
                         target_domain: str,
                         strategy_pattern: Dict[str, Any],
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        迁移策略
        
        Args:
            source_domain: 源领域
            target_domain: 目标领域
            strategy_pattern: 策略模式
            context: 上下文信息
            
        Returns:
            策略迁移结果
        """
        start_time = time.time()
        
        try:
            # 检查缓存
            cache_key = (source_domain, target_domain, str(strategy_pattern.get("pattern_id", "")))
            if cache_key in self.strategy_transfer_cache:
                cached_result = self.strategy_transfer_cache[cache_key]
                cached_result["cached"] = True
                return cached_result
            
            # 执行策略迁移
            result = self.strategy_transferor.transfer(
                source_domain, target_domain, strategy_pattern, context
            )
            
            # 存储策略模式
            if result.get("success", False):
                pattern = StrategyPattern(
                    pattern_id=f"pattern_{int(time.time())}_{(zlib.adler32(str(str(strategy_pattern).encode('utf-8')) & 0xffffffff))}",
                    source_domain=source_domain,
                    target_domain=target_domain,
                    pattern_type=strategy_pattern.get("type", "unknown"),
                    effectiveness=result.get("effectiveness", 0.5),
                    applicability_score=result.get("applicability_score", 0.5),
                    transfer_cost=result.get("transfer_cost", 0.1),
                    conditions=result.get("conditions", {}),
                    implementation=result.get("implementation", {}),
                    examples=result.get("examples", [])
                )
                
                self.strategy_pattern_library[pattern.pattern_id] = pattern
                self.state["strategy_pattern_count"] = len(self.strategy_pattern_library)
                
                # 更新统计
                self.state["successful_transfers"] += 1
                
                # 缓存结果
                result["pattern_id"] = pattern.pattern_id
                self.strategy_transfer_cache[cache_key] = result
            else:
                self.state["failed_transfers"] += 1
            
            # 添加性能指标
            result["transfer_time"] = time.time() - start_time
            result["source_domain"] = source_domain
            result["target_domain"] = target_domain
            
            logger.info(
                f"策略迁移完成: 源领域='{source_domain}', "
                f"目标领域='{target_domain}', "
                f"成功率={result.get('success', False)}, "
                f"时间={result['transfer_time']:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.state["failed_transfers"] += 1
            error_handler.handle_error(e, "CrossDomainPlanner", "策略迁移失败")
            return {
                "success": False,
                "error": f"策略迁移失败: {str(e)}",
                "source_domain": source_domain,
                "target_domain": target_domain,
                "transfer_time": time.time() - start_time
            }
    
    def adapt_to_domain(self, plan: Dict[str, Any],
                       target_domain: str,
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        适配计划到目标领域
        
        Args:
            plan: 要适配的计划
            target_domain: 目标领域
            context: 上下文信息
            
        Returns:
            适配结果
        """
        start_time = time.time()
        self.state["adaptation_attempts"] += 1
        
        try:
            result = self.cross_domain_adapter.adapt(plan, target_domain, context)
            
            if result.get("success", False):
                self.state["adaptation_successes"] += 1
                
                # 学习适配规则
                self._learn_adaptation_rules(plan, result, target_domain)
            
            # 添加性能指标
            result["adaptation_time"] = time.time() - start_time
            result["target_domain"] = target_domain
            result["original_plan_id"] = plan.get("id", "unknown")
            
            logger.info(
                f"领域适配完成: 目标领域='{target_domain}', "
                f"成功率={result.get('success', False)}, "
                f"时间={result['adaptation_time']:.2f}s"
            )
            
            return result
            
        except Exception as e:
            error_handler.handle_error(e, "CrossDomainPlanner", "领域适配失败")
            return {
                "success": False,
                "error": f"领域适配失败: {str(e)}",
                "target_domain": target_domain,
                "adaptation_time": time.time() - start_time
            }
    
    def learn_meta_strategy(self, planning_experiences: List[Dict[str, Any]],
                           domain_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        学习元策略
        
        Args:
            planning_experiences: 规划经验列表
            domain_pairs: 领域对列表
            
        Returns:
            元策略学习结果
        """
        try:
            result = self.meta_strategy_learner.learn(planning_experiences, domain_pairs)
            
            logger.info(
                f"元策略学习完成: 经验数={len(planning_experiences)}, "
                f"领域对数={len(domain_pairs)}, "
                f"成功={result.get('success', False)}"
            )
            
            return result
            
        except Exception as e:
            error_handler.handle_error(e, "CrossDomainPlanner", "元策略学习失败")
            return {
                "success": False,
                "error": f"元策略学习失败: {str(e)}",
                "experience_count": len(planning_experiences)
            }
    
    def _analyze_cross_domain_goal(self, goal: Any, 
                                  target_domain: str,
                                  context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """分析跨领域目标"""
        try:
            # 提取目标中的领域相关特征
            goal_text = str(goal).lower() if not isinstance(goal, dict) else str(goal).lower()
            
            # 识别领域关键词
            domain_keywords = self._extract_domain_keywords(goal_text, target_domain)
            
            # 分析目标复杂度
            complexity_score = self._assess_cross_domain_complexity(goal_text, target_domain, context)
            
            # 识别跨领域需求
            cross_domain_needs = self._identify_cross_domain_needs(goal_text, target_domain, context)
            
            analysis = {
                "goal_representation": goal_text[:100] + "..." if len(goal_text) > 100 else goal_text,
                "target_domain": target_domain,
                "domain_keywords": domain_keywords,
                "complexity_score": complexity_score,
                "cross_domain_needs": cross_domain_needs,
                "requires_multiple_domains": len(domain_keywords.get("other_domains", [])) > 0,
                "estimated_domain_count": 1 + len(domain_keywords.get("other_domains", [])),
                "analysis_time": time.time()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"跨领域目标分析失败: {e}")
            return {
                "goal_representation": str(goal)[:50],
                "target_domain": target_domain,
                "domain_keywords": {},
                "complexity_score": 0.5,
                "cross_domain_needs": [],
                "requires_multiple_domains": False,
                "estimated_domain_count": 1,
                "analysis_error": str(e)
            }
    
    def _identify_relevant_domains(self, goal_analysis: Dict[str, Any],
                                  target_domain: str,
                                  available_domains: Optional[List[str]]) -> List[str]:
        """识别相关领域"""
        try:
            relevant_domains = [target_domain]
            
            # 如果指定了可用领域，使用它们
            if available_domains:
                relevant_domains.extend(available_domains)
            else:
                # 自动发现相关领域
                other_domains = goal_analysis.get("domain_keywords", {}).get("other_domains", [])
                relevant_domains.extend(other_domains)
                
                # 从知识库中查找相关领域
                knowledge_based_domains = self._find_related_domains_from_knowledge(
                    target_domain, goal_analysis
                )
                relevant_domains.extend(knowledge_based_domains)
            
            # 去重和限制数量
            relevant_domains = list(OrderedDict.fromkeys(relevant_domains))
            max_domains = self.config["domain_integration"]["max_domains_to_integrate"]
            relevant_domains = relevant_domains[:max_domains]
            
            return relevant_domains
            
        except Exception as e:
            logger.error(f"识别相关领域失败: {e}")
            return [target_domain]
    
    def _integrate_domain_knowledge(self, domains: List[str],
                                   goal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """整合领域知识"""
        try:
            # 收集各领域知识
            domain_knowledge_list = []
            
            for domain in domains:
                if domain in self.domain_knowledge_base:
                    knowledge = self.domain_knowledge_base[domain]
                    domain_knowledge_list.append(asdict(knowledge))
                else:
                    # 如果没有现成知识，尝试从缓存或其他来源获取
                    cached = self.domain_knowledge_cache.get(domain)
                    if cached:
                        domain_knowledge_list.append(cached)
            
            # 如果领域知识不足，创建基础知识
            if len(domain_knowledge_list) < len(domains):
                for domain in domains:
                    if not any(k.get("domain_name") == domain for k in domain_knowledge_list):
                        base_knowledge = self._create_base_domain_knowledge(domain, goal_analysis)
                        domain_knowledge_list.append(base_knowledge)
            
            # 整合知识
            integrated_knowledge = self.domain_knowledge_integrator.combine(
                domain_knowledge_list, goal_analysis
            )
            
            return integrated_knowledge
            
        except Exception as e:
            logger.error(f"整合领域知识失败: {e}")
            return {
                "domains": domains,
                "combined_concepts": [],
                "combined_relations": [],
                "combined_strategies": [],
                "integration_error": str(e)
            }
    
    def _discover_and_transfer_strategies(self, source_domains: List[str],
                                         target_domain: str,
                                         goal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """发现和迁移策略"""
        try:
            transferable_strategies = []
            
            for source_domain in source_domains:
                if source_domain == target_domain:
                    continue
                
                # 发现可迁移的策略
                strategies = self._discover_strategies_for_transfer(
                    source_domain, target_domain, goal_analysis
                )
                
                # 迁移策略
                for strategy in strategies:
                    transfer_result = self.transfer_strategy(
                        source_domain, target_domain, strategy, goal_analysis
                    )
                    
                    if transfer_result.get("success", False):
                        transferable_strategies.append({
                            "source_domain": source_domain,
                            "target_domain": target_domain,
                            "strategy_pattern": strategy,
                            "transfer_result": transfer_result,
                            "effectiveness": transfer_result.get("effectiveness", 0.5)
                        })
            
            # 按效果排序
            transferable_strategies.sort(key=lambda x: x["effectiveness"], reverse=True)
            
            return transferable_strategies
            
        except Exception as e:
            logger.error(f"发现和迁移策略失败: {e}")
            return []
    
    def _process_cross_domain_constraints(self, constraints: Optional[Dict[str, Any]],
                                         domains: List[str],
                                         target_domain: str) -> Dict[str, Any]:
        """处理跨领域约束"""
        try:
            if not constraints:
                constraints = {}
            
            # 收集各领域约束
            domain_constraints = []
            
            for domain in domains:
                if domain in self.domain_knowledge_base:
                    domain_knowledge = self.domain_knowledge_base[domain]
                    domain_constraints.append(domain_knowledge.constraints)
                else:
                    domain_constraints.append({})
            
            # 整合约束
            processed_constraints = self.domain_knowledge_integrator.process_constraints(
                constraints, domain_constraints, target_domain
            )
            
            return processed_constraints
            
        except Exception as e:
            logger.error(f"处理跨领域约束失败: {e}")
            return constraints or {}
    
    def _generate_cross_domain_candidate_plans(self, goal_analysis: Dict[str, Any],
                                              integrated_knowledge: Dict[str, Any],
                                              transferable_strategies: List[Dict[str, Any]],
                                              constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成跨领域候选计划"""
        try:
            candidate_plans = []
            
            # 生成多个候选计划
            for i in range(3):  # 生成3个候选计划
                plan = self._create_cross_domain_plan(
                    goal_analysis, integrated_knowledge, transferable_strategies, constraints, i
                )
                
                if plan:
                    candidate_plans.append(plan)
            
            return candidate_plans
            
        except Exception as e:
            logger.error(f"生成跨领域候选计划失败: {e}")
            return []
    
    def _evaluate_and_optimize_cross_domain_plans(self, candidate_plans: List[Dict[str, Any]],
                                                 goal_analysis: Dict[str, Any],
                                                 integrated_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """评估和优化跨领域候选计划"""
        try:
            if not candidate_plans:
                return self._create_empty_plan(goal_analysis)
            
            # 评估计划
            evaluated_plans = []
            for plan in candidate_plans:
                evaluation = self._evaluate_cross_domain_plan(plan, goal_analysis, integrated_knowledge)
                plan["evaluation"] = evaluation
                evaluated_plans.append(plan)
            
            # 选择最佳计划
            best_plan = max(evaluated_plans, key=lambda x: x["evaluation"].get("overall_score", 0))
            
            # 优化计划
            optimized_plan = self._optimize_cross_domain_plan(best_plan, goal_analysis, integrated_knowledge)
            
            return optimized_plan
            
        except Exception as e:
            logger.error(f"评估和优化跨领域计划失败: {e}")
            return self._create_empty_plan(goal_analysis)
    
    def _adapt_plan_to_target_domain(self, plan: Dict[str, Any],
                                    target_domain: str,
                                    source_domains: List[str]) -> Dict[str, Any]:
        """适配计划到目标领域"""
        try:
            adaptation_result = self.adapt_to_domain(plan, target_domain, {
                "source_domains": source_domains,
                "plan_context": plan.get("context", {})
            })
            
            if adaptation_result.get("success", False):
                adapted_plan = adaptation_result.get("adapted_plan", plan)
                adapted_plan["adaptation_result"] = adaptation_result
                return adapted_plan
            else:
                # 适配失败，返回原始计划
                plan["adaptation_failed"] = True
                plan["adaptation_error"] = adaptation_result.get("error", "未知错误")
                return plan
                
        except Exception as e:
            logger.error(f"适配计划到目标领域失败: {e}")
            plan["adaptation_failed"] = True
            plan["adaptation_error"] = str(e)
            return plan
    
    def _generate_cross_domain_final_result(self, plan: Dict[str, Any],
                                           goal_analysis: Dict[str, Any],
                                           relevant_domains: List[str],
                                           transferable_strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成跨领域最终结果"""
        try:
            # 计算跨领域指标
            cross_domain_metrics = self._calculate_cross_domain_metrics(
                plan, goal_analysis, relevant_domains, transferable_strategies
            )
            
            # 生成最终结果
            final_result = {
                "success": True,
                "plan": plan,
                "goal_analysis": goal_analysis,
                "relevant_domains": relevant_domains,
                "transferable_strategies_count": len(transferable_strategies),
                "cross_domain_metrics": cross_domain_metrics,
                "performance_metrics": {
                    "plan_complexity": goal_analysis.get("complexity_score", 0.5),
                    "domain_integration_score": cross_domain_metrics.get("integration_score", 0.5),
                    "strategy_transfer_score": cross_domain_metrics.get("transfer_score", 0.5),
                    "adaptation_success_score": 1.0 if not plan.get("adaptation_failed") else 0.0,
                    "overall_cross_domain_score": (
                        goal_analysis.get("complexity_score", 0.5) * 0.3 +
                        cross_domain_metrics.get("integration_score", 0.5) * 0.3 +
                        cross_domain_metrics.get("transfer_score", 0.5) * 0.2 +
                        (1.0 if not plan.get("adaptation_failed") else 0.0) * 0.2
                    )
                },
                "recommendations": self._generate_cross_domain_recommendations(
                    plan, goal_analysis, relevant_domains
                )
            }
            
            return final_result
            
        except Exception as e:
            logger.error(f"生成跨领域最终结果失败: {e}")
            return {
                "success": False,
                "error": f"生成最终结果失败: {str(e)}",
                "goal_analysis": goal_analysis,
                "relevant_domains": relevant_domains
            }
    
    def _record_cross_domain_learning_experience(self, goal_analysis: Dict[str, Any],
                                                relevant_domains: List[str],
                                                final_result: Dict[str, Any]) -> None:
        """记录跨领域学习经验"""
        try:
            experience = {
                "timestamp": time.time(),
                "goal": goal_analysis.get("goal_representation", ""),
                "target_domain": goal_analysis.get("target_domain", ""),
                "relevant_domains": relevant_domains,
                "success": final_result.get("success", False),
                "cross_domain_metrics": final_result.get("cross_domain_metrics", {}),
                "performance_metrics": final_result.get("performance_metrics", {}),
                "plan_id": final_result.get("plan", {}).get("id", "unknown")
            }
            
            # 这里可以将经验存储到数据库或文件中
            # 暂时只记录到日志
            logger.debug(f"跨领域学习经验记录: {experience}")
            
        except Exception as e:
            logger.error(f"记录跨领域学习经验失败: {e}")
    
    def _generate_cross_domain_error_result(self, goal: Any, 
                                           target_domain: str,
                                           error_message: str) -> Dict[str, Any]:
        """生成跨领域错误结果"""
        return {
            "success": False,
            "error": error_message,
            "goal": str(goal)[:100],
            "target_domain": target_domain,
            "partial_results": {
                "goal_analysis": None,
                "relevant_domains": [],
                "transferable_strategies": [],
                "plan": None
            },
            "recommendations": [
                "检查领域知识库是否包含目标领域",
                "确保有足够的源领域知识可用",
                "简化目标或减少涉及的领域数量"
            ]
        }
    
    # 辅助方法（简化实现）
    def _extract_domain_keywords(self, goal_text: str, target_domain: str) -> Dict[str, Any]:
        """提取领域关键词"""
        # 简化实现 - 实际应该使用NLP技术
        keywords = {
            "target_domain_keywords": [target_domain],
            "other_domains": [],
            "domain_specific_terms": []
        }
        
        # 简单关键词匹配
        domain_keyword_map = {
            "science": ["research", "experiment", "theory", "discovery"],
            "business": ["profit", "market", "investment", "strategy"],
            "technology": ["software", "hardware", "system", "development"],
            "education": ["learning", "teaching", "curriculum", "student"]
        }
        
        for domain, terms in domain_keyword_map.items():
            if any(term in goal_text for term in terms) and domain != target_domain:
                keywords["other_domains"].append(domain)
                keywords["domain_specific_terms"].extend(terms)
        
        return keywords
    
    def _assess_cross_domain_complexity(self, goal_text: str, 
                                       target_domain: str,
                                       context: Optional[Dict[str, Any]]) -> float:
        """评估跨领域复杂度"""
        # 简化实现
        complexity_factors = []
        
        # 文本长度
        text_length = len(goal_text)
        complexity_factors.append(min(text_length / 500, 1.0))
        
        # 领域数量估计
        domain_count = 1 + goal_text.count(" and ") + goal_text.count(" with ")
        complexity_factors.append(min(domain_count / 5, 1.0))
        
        # 返回平均复杂度
        return sum(complexity_factors) / len(complexity_factors) if complexity_factors else 0.5
    
    def _identify_cross_domain_needs(self, goal_text: str,
                                    target_domain: str,
                                    context: Optional[Dict[str, Any]]) -> List[str]:
        """识别跨领域需求"""
        # 简化实现
        needs = []
        
        if "data" in goal_text and "analysis" in goal_text:
            needs.append("data_science")
        
        if "system" in goal_text and "design" in goal_text:
            needs.append("systems_engineering")
        
        if "user" in goal_text and "interface" in goal_text:
            needs.append("human_computer_interaction")
        
        return needs
    
    def _find_related_domains_from_knowledge(self, target_domain: str,
                                            goal_analysis: Dict[str, Any]) -> List[str]:
        """从知识库中查找相关领域"""
        # 简化实现
        related_domains = []
        
        # 这里应该实现基于知识图谱的领域相似性计算
        # 暂时返回空列表
        return related_domains
    
    def _create_base_domain_knowledge(self, domain: str,
                                     goal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """创建基础领域知识"""
        return {
            "domain_name": domain,
            "category": "unknown",
            "key_concepts": [domain],
            "key_relations": [],
            "strategies": [],
            "constraints": {},
            "resources": [],
            "metadata": {
                "created_time": time.time(),
                "is_base_knowledge": True
            }
        }
    
    def _discover_strategies_for_transfer(self, source_domain: str,
                                         target_domain: str,
                                         goal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """发现可迁移的策略"""
        # 简化实现
        strategies = []
        
        # 创建一些示例策略
        example_strategies = [
            {
                "type": "problem_decomposition",
                "description": "问题分解策略",
                "effectiveness": 0.7,
                "conditions": {"requires_structured_thinking": True}
            },
            {
                "type": "resource_allocation",
                "description": "资源分配策略",
                "effectiveness": 0.6,
                "conditions": {"has_multiple_resources": True}
            },
            {
                "type": "constraint_satisfaction",
                "description": "约束满足策略",
                "effectiveness": 0.8,
                "conditions": {"has_constraints": True}
            }
        ]
        
        return example_strategies[:2]  # 返回前2个策略
    
    def _create_cross_domain_plan(self, goal_analysis: Dict[str, Any],
                                 integrated_knowledge: Dict[str, Any],
                                 transferable_strategies: List[Dict[str, Any]],
                                 constraints: Dict[str, Any],
                                 plan_index: int) -> Dict[str, Any]:
        """创建跨领域计划"""
        plan_id = f"cross_domain_plan_{int(time.time())}_{plan_index}"
        
        plan = {
            "id": plan_id,
            "goal": goal_analysis.get("goal_representation", ""),
            "target_domain": goal_analysis.get("target_domain", ""),
            "source_domains": integrated_knowledge.get("domains", []),
            "steps": self._generate_plan_steps(goal_analysis, integrated_knowledge, plan_index),
            "strategies": [s["strategy_pattern"] for s in transferable_strategies],
            "constraints": constraints,
            "resources": integrated_knowledge.get("combined_resources", []),
            "estimated_duration": 60 * (plan_index + 1),  # 分钟
            "complexity": goal_analysis.get("complexity_score", 0.5),
            "created_time": time.time()
        }
        
        return plan
    
    def _generate_plan_steps(self, goal_analysis: Dict[str, Any],
                            integrated_knowledge: Dict[str, Any],
                            plan_index: int) -> List[Dict[str, Any]]:
        """生成计划步骤"""
        steps = []
        
        # 基础步骤模板
        base_steps = [
            {"id": 1, "description": "分析目标和需求", "duration": 10, "domain": "general"},
            {"id": 2, "description": "收集相关领域知识", "duration": 20, "domain": "knowledge_acquisition"},
            {"id": 3, "description": "设计解决方案框架", "duration": 30, "domain": "design"},
            {"id": 4, "description": "实施核心功能", "duration": 40, "domain": "implementation"},
            {"id": 5, "description": "测试和验证结果", "duration": 20, "domain": "testing"}
        ]
        
        # 根据计划索引调整
        for step in base_steps:
            adjusted_step = step.copy()
            adjusted_step["duration"] = step["duration"] * (plan_index + 1) // 2
            adjusted_step["domain_specific"] = plan_index > 0
            steps.append(adjusted_step)
        
        return steps
    
    def _evaluate_cross_domain_plan(self, plan: Dict[str, Any],
                                   goal_analysis: Dict[str, Any],
                                   integrated_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """评估跨领域计划"""
        evaluation = {
            "completeness_score": 0.7 + random.random() * 0.2,
            "feasibility_score": 0.6 + random.random() * 0.3,
            "efficiency_score": 0.5 + random.random() * 0.3,
            "cross_domain_integration_score": 0.4 + random.random() * 0.4,
            "strategy_application_score": 0.3 + random.random() * 0.4
        }
        
        evaluation["overall_score"] = (
            evaluation["completeness_score"] * 0.3 +
            evaluation["feasibility_score"] * 0.3 +
            evaluation["efficiency_score"] * 0.2 +
            evaluation["cross_domain_integration_score"] * 0.1 +
            evaluation["strategy_application_score"] * 0.1
        )
        
        return evaluation
    
    def _optimize_cross_domain_plan(self, plan: Dict[str, Any],
                                   goal_analysis: Dict[str, Any],
                                   integrated_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """优化跨领域计划"""
        optimized_plan = plan.copy()
        
        # 简化优化：调整步骤顺序和持续时间
        if "steps" in optimized_plan:
            for step in optimized_plan["steps"]:
                # 减少持续时间10%
                step["duration"] = max(1, int(step["duration"] * 0.9))
                
                # 添加优化标记
                step["optimized"] = True
        
        optimized_plan["optimized"] = True
        optimized_plan["optimization_time"] = time.time()
        
        return optimized_plan
    
    def _calculate_cross_domain_metrics(self, plan: Dict[str, Any],
                                       goal_analysis: Dict[str, Any],
                                       relevant_domains: List[str],
                                       transferable_strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算跨领域指标"""
        metrics = {
            "domain_count": len(relevant_domains),
            "strategy_count": len(transferable_strategies),
            "integration_score": min(len(relevant_domains) / 5, 1.0),
            "transfer_score": min(len(transferable_strategies) / 3, 1.0),
            "adaptation_required": any("adapt" in step.get("description", "").lower() 
                                      for step in plan.get("steps", [])),
            "cross_domain_complexity": goal_analysis.get("complexity_score", 0.5)
        }
        
        return metrics
    
    def _generate_cross_domain_recommendations(self, plan: Dict[str, Any],
                                              goal_analysis: Dict[str, Any],
                                              relevant_domains: List[str]) -> List[str]:
        """生成跨领域建议"""
        recommendations = []
        
        if len(relevant_domains) > 3:
            recommendations.append("考虑减少涉及的领域数量以简化实施")
        
        if goal_analysis.get("complexity_score", 0.5) > 0.7:
            recommendations.append("建议将目标分解为多个阶段实施")
        
        if not plan.get("strategies"):
            recommendations.append("考虑添加更多跨领域策略以提高成功率")
        
        return recommendations or ["计划看起来合理，可以开始实施"]
    
    def _create_empty_plan(self, goal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """创建空计划"""
        return {
            "id": f"empty_plan_{int(time.time())}",
            "goal": goal_analysis.get("goal_representation", ""),
            "target_domain": goal_analysis.get("target_domain", ""),
            "source_domains": [],
            "steps": [],
            "strategies": [],
            "constraints": {},
            "resources": [],
            "estimated_duration": 0,
            "complexity": goal_analysis.get("complexity_score", 0.5),
            "created_time": time.time(),
            "is_empty_plan": True
        }
    
    def _learn_adaptation_rules(self, original_plan: Dict[str, Any],
                               adaptation_result: Dict[str, Any],
                               target_domain: str) -> None:
        """学习适配规则"""
        # 简化实现
        logger.debug(f"学习适配规则: 目标领域={target_domain}, 计划ID={original_plan.get('id')}")


# 组件类定义
class DomainKnowledgeIntegrator:
    """领域知识整合器"""
    
    def integrate(self, domain_data: Dict[str, Any], 
                 domain_name: str,
                 category: DomainCategory) -> Dict[str, Any]:
        """整合领域数据为知识表示"""
        # 简化实现
        return {
            "knowledge_graph": nx.DiGraph(),
            "key_concepts": list(set(domain_data.get("concepts", []))),
            "key_relations": domain_data.get("relations", []),
            "strategies": domain_data.get("strategies", []),
            "constraints": domain_data.get("constraints", {}),
            "resources": domain_data.get("resources", []),
            "integration_success": True
        }
    
    def combine(self, domain_knowledge_list: List[Dict[str, Any]],
               goal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """组合多个领域知识"""
        combined = {
            "domains": [k.get("domain_name", "unknown") for k in domain_knowledge_list],
            "combined_concepts": [],
            "combined_relations": [],
            "combined_strategies": [],
            "combined_resources": [],
            "combined_constraints": {}
        }
        
        for knowledge in domain_knowledge_list:
            combined["combined_concepts"].extend(knowledge.get("key_concepts", []))
            combined["combined_relations"].extend(knowledge.get("key_relations", []))
            combined["combined_strategies"].extend(knowledge.get("strategies", []))
            combined["combined_resources"].extend(knowledge.get("resources", []))
            
            # 合并约束
            for key, value in knowledge.get("constraints", {}).items():
                if key not in combined["combined_constraints"]:
                    combined["combined_constraints"][key] = value
        
        # 去重
        combined["combined_concepts"] = list(set(combined["combined_concepts"]))
        
        return combined
    
    def process_constraints(self, constraints: Dict[str, Any],
                           domain_constraints: List[Dict[str, Any]],
                           target_domain: str) -> Dict[str, Any]:
        """处理跨领域约束"""
        processed = constraints.copy()
        
        for domain_constraint in domain_constraints:
            processed.update(domain_constraint)
        
        return processed

class StrategyTransferor:
    """策略迁移器"""
    
    def transfer(self, source_domain: str,
                target_domain: str,
                strategy_pattern: Dict[str, Any],
                context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """迁移策略"""
        # 简化实现
        similarity_score = self._calculate_domain_similarity(source_domain, target_domain)
        
        if similarity_score < 0.3:
            return {
                "success": False,
                "error": f"领域相似度过低: {similarity_score:.2f}",
                "effectiveness": 0.0
            }
        
        # 模拟迁移过程
        effectiveness = 0.5 + similarity_score * 0.3
        
        return {
            "success": True,
            "effectiveness": effectiveness,
            "applicability_score": effectiveness * 0.8,
            "transfer_cost": 0.1 + (1 - similarity_score) * 0.2,
            "conditions": strategy_pattern.get("conditions", {}),
            "implementation": {
                "adapted_from": source_domain,
                "adapted_to": target_domain,
                "adaptation_changes": ["参数调整", "术语转换", "约束适配"]
            },
            "examples": [
                {"source_example": "在源领域的应用示例", "target_example": "在目标领域的应用示例"}
            ]
        }
    
    def _calculate_domain_similarity(self, domain1: str, domain2: str) -> float:
        """计算领域相似度"""
        # 简化实现
        if domain1 == domain2:
            return 1.0
        
        # 简单字符串相似度
        common_chars = set(domain1.lower()) & set(domain2.lower())
        return len(common_chars) / max(len(domain1), len(domain2), 1)

class CrossDomainAdapter:
    """跨领域适配器"""
    
    def adapt(self, plan: Dict[str, Any],
             target_domain: str,
             context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """适配计划到目标领域"""
        # 简化实现
        adapted_plan = plan.copy()
        
        # 添加适配标记
        adapted_plan["adapted_to"] = target_domain
        adapted_plan["adaptation_time"] = time.time()
        
        # 修改步骤描述以反映领域适配
        if "steps" in adapted_plan:
            for step in adapted_plan["steps"]:
                if "domain" in step:
                    step["description"] = f"[{target_domain}] {step.get('description', '')}"
        
        return {
            "success": True,
            "adapted_plan": adapted_plan,
            "adaptation_changes": [
                "术语本地化",
                "约束调整",
                "资源重映射"
            ],
            "adaptation_quality": 0.7 + random.random() * 0.2
        }

class MetaStrategyLearner:
    """元策略学习器"""
    
    def learn(self, planning_experiences: List[Dict[str, Any]],
             domain_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """学习元策略"""
        # 简化实现
        if not planning_experiences:
            return {
                "success": False,
                "error": "没有规划经验可供学习",
                "meta_strategies": []
            }
        
        # 分析成功模式
        successful_experiences = [e for e in planning_experiences if e.get("success", False)]
        success_rate = len(successful_experiences) / len(planning_experiences) if planning_experiences else 0
        
        # 提取元策略
        meta_strategies = [
            {
                "strategy_type": "domain_similarity_based_transfer",
                "description": "基于领域相似度的策略迁移",
                "applicability_conditions": {
                    "domain_similarity_threshold": 0.5,
                    "requires_common_concepts": True
                },
                "expected_success_rate": success_rate
            },
            {
                "strategy_type": "constraint_relaxation",
                "description": "约束放松策略",
                "applicability_conditions": {
                    "has_strict_constraints": True,
                    "allows_flexibility": True
                },
                "expected_success_rate": success_rate * 0.8
            }
        ]
        
        return {
            "success": True,
            "meta_strategies": meta_strategies,
            "learning_metrics": {
                "experience_count": len(planning_experiences),
                "success_rate": success_rate,
                "domain_pair_count": len(domain_pairs)
            }
        }


# 工厂函数
def create_cross_domain_planner(config: Optional[Dict[str, Any]] = None) -> CrossDomainPlanner:
    """
    创建跨领域规划器实例
    
    Args:
        config: 可选配置字典
        
    Returns:
        跨领域规划器实例
    """
    try:
        planner = CrossDomainPlanner(config)
        logger.info("跨领域规划器实例创建成功")
        return planner
    except Exception as e:
        logger.error(f"创建跨领域规划器失败: {e}")
        raise


# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("跨领域规划器测试")
    print("=" * 60)
    
    try:
        # 创建规划器
        planner = create_cross_domain_planner()
        print("✓ 跨领域规划器创建成功")
        
        # 测试领域知识整合
        science_data = {
            "concepts": ["hypothesis", "experiment", "data", "analysis", "conclusion"],
            "relations": [("hypothesis", "leads_to", "experiment"), ("data", "supports", "conclusion")],
            "strategies": [{"type": "scientific_method", "description": "科学方法"}],
            "constraints": {"requires_evidence": True, "reproducible": True},
            "resources": ["lab_equipment", "research_funding"]
        }
        
        integration_result = planner.integrate_domain_knowledge(
            science_data, "science_research", DomainCategory.SCIENCE_TECHNOLOGY
        )
        
        if integration_result["success"]:
            print(f"✓ 领域知识整合成功: 领域={integration_result['domain_name']}")
        else:
            print(f"✗ 领域知识整合失败: {integration_result.get('error', '未知错误')}")
        
        # 测试跨领域规划
        goal = "开发一个基于科学研究的商业产品"
        target_domain = "business_product_development"
        
        plan_result = planner.plan_across_domains(
            goal=goal,
            target_domain=target_domain,
            context={"budget": 100000, "timeline": "6 months"},
            available_domains=["science_research", "technology_development"]
        )
        
        if plan_result["success"]:
            plan = plan_result["plan"]
            print(f"✓ 跨领域规划成功: 目标='{goal}'")
            print(f"  计划ID: {plan.get('id', 'unknown')}")
            print(f"  相关领域数: {len(plan.get('source_domains', [])) + 1}")
            print(f"  步骤数: {len(plan.get('steps', []))}")
        else:
            print(f"✗ 跨领域规划失败: {plan_result.get('error', '未知错误')}")
        
        print("=" * 60)
        print("测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()