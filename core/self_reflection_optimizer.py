#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zlib
"""
Self-Reflection Optimizer - 自我反思优化器

核心功能：
1. 规划和推理性能分析
2. 错误模式和根本原因诊断
3. 改进建议生成和优先级排序
4. 自适应学习和优化
5. 元认知监控和控制

实现AGI系统的自我反思、错误诊断和持续优化能力，
支持系统通过学习和反思不断提升性能。

Copyright (c) 2025 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
import json
import math
import random
import statistics
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from enum import Enum
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import networkx as nx
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 导入错误处理
from core.error_handling import ErrorHandler

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()

class ReflectionType(Enum):
    """反思类型枚举"""
    PERFORMANCE_REFLECTION = "performance_reflection"        # 性能反思
    ERROR_REFLECTION = "error_reflection"                    # 错误反思
    STRATEGY_REFLECTION = "strategy_reflection"              # 策略反思
    ADAPTATION_REFLECTION = "adaptation_reflection"          # 适应反思
    META_COGNITIVE_REFLECTION = "meta_cognitive_reflection"  # 元认知反思

class ImprovementPriority(Enum):
    """改进优先级枚举"""
    CRITICAL = "critical"        # 关键：必须立即解决
    HIGH = "high"                # 高：需要尽快解决
    MEDIUM = "medium"            # 中：需要安排时间解决
    LOW = "low"                  # 低：可以考虑解决
    FUTURE = "future"            # 未来：跟踪但不立即解决

@dataclass
class PerformanceMetric:
    """性能指标表示"""
    metric_id: str
    metric_type: str
    value: float
    timestamp: float
    context: Dict[str, Any]
    confidence: float = 1.0
    trend: str = "stable"
    
@dataclass
class ErrorPattern:
    """错误模式表示"""
    pattern_id: str
    error_type: str
    error_message: str
    frequency: int
    first_occurrence: float
    last_occurrence: float
    context_patterns: List[Dict[str, Any]]
    root_causes: List[str]
    impact_score: float
    mitigation_strategies: List[Dict[str, Any]]
    
@dataclass
class ImprovementSuggestion:
    """改进建议表示"""
    suggestion_id: str
    title: str
    description: str
    priority: ImprovementPriority
    expected_benefit: float
    implementation_cost: float
    implementation_effort: float
    related_metrics: List[str]
    related_errors: List[str]
    implementation_steps: List[Dict[str, Any]]
    validation_criteria: List[Dict[str, Any]]
    
@dataclass
class ReflectionSession:
    """反思会话表示"""
    session_id: str
    reflection_type: ReflectionType
    start_time: float
    end_time: float
    focus_areas: List[str]
    insights_gained: List[Dict[str, Any]]
    improvements_generated: List[ImprovementSuggestion]
    performance_impact: Dict[str, Any]

class SelfReflectionOptimizer:
    """
    自我反思优化器 - 实现AGI系统的自我反思和优化
    
    核心特性：
    1. 多维性能分析和监控
    2. 错误模式和根本原因诊断
    3. 改进建议生成和优先级排序
    4. 自适应学习和元认知控制
    5. 持续优化和性能提升
    6. 知识迁移和经验积累
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化自我反思优化器"""
        self.logger = logger
        self.config = config or self._get_default_config()
        
        # 初始化组件
        self._initialize_components()
        
        # 性能数据存储
        self.performance_metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.improvement_suggestions: Dict[str, ImprovementSuggestion] = {}
        self.reflection_sessions: Dict[str, ReflectionSession] = {}
        
        # 状态跟踪
        self.state = {
            "total_reflection_sessions": 0,
            "total_improvements_generated": 0,
            "total_improvements_implemented": 0,
            "average_reflection_time": 0,
            "performance_metrics_collected": 0,
            "error_patterns_identified": 0,
            "successful_optimizations": 0,
            "performance_improvement_rate": 0.0
        }
        
        # 学习数据
        self.learning_data = {
            "performance_trends": defaultdict(list),
            "error_correlation": defaultdict(dict),
            "improvement_effectiveness": defaultdict(list),
            "strategy_adaptation_patterns": [],
            "meta_cognitive_insights": []
        }
        
        # 缓存
        self.recent_performance_cache: deque = deque(maxlen=1000)
        self.error_context_cache: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        logger.info("自我反思优化器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "performance_monitoring": {
                "enable_real_time_monitoring": True,
                "metrics_collection_interval": 60,  # 秒
                "performance_thresholds": {
                    "planning_accuracy": 0.7,
                    "reasoning_quality": 0.6,
                    "adaptation_success": 0.5,
                    "cross_domain_transfer": 0.4
                },
                "trend_analysis_window": 100
            },
            "error_analysis": {
                "enable_automatic_error_detection": True,
                "error_clustering_enabled": True,
                "root_cause_analysis_depth": 3,
                "error_pattern_min_frequency": 3,
                "enable_error_prediction": True
            },
            "improvement_generation": {
                "enable_automatic_suggestion": True,
                "suggestion_quality_threshold": 0.6,
                "benefit_cost_ratio_threshold": 2.0,
                "implementation_effort_limit": 0.3,
                "enable_adaptive_prioritization": True
            },
            "reflection": {
                "enable_periodic_reflection": True,
                "reflection_interval": 3600,  # 秒
                "deep_reflection_threshold": 10,  # 错误或低性能次数
                "enable_meta_cognitive_reflection": True,
                "reflection_depth": "moderate"
            },
            "optimization": {
                "enable_continuous_optimization": True,
                "optimization_iterations": 5,
                "performance_improvement_target": 0.1,
                "enable_knowledge_transfer": True,
                "optimization_aggressiveness": 0.5
            }
        }
    
    def _initialize_components(self) -> None:
        """初始化所有组件"""
        try:
            # 1. 性能分析器
            self.performance_analyzer = PerformanceAnalyzer()
            
            # 2. 错误诊断器
            self.error_diagnoser = ErrorDiagnoser()
            
            # 3. 改进生成器
            self.improvement_generator = ImprovementGenerator()
            
            # 4. 自适应学习器
            self.adaptation_learner = AdaptationLearner()
            
            logger.info("自我反思优化器组件初始化完成")
            
        except Exception as e:
            error_handler.handle_error(e, "SelfReflectionOptimizer", "初始化组件失败")
            logger.error(f"组件初始化失败: {e}")
    
    def reflect_on_performance(self, 
                              performance_data: Dict[str, Any],
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        基于性能数据进行反思
        
        Args:
            performance_data: 性能数据
            context: 上下文信息
            
        Returns:
            反思结果
        """
        start_time = time.time()
        self.state["total_reflection_sessions"] += 1
        
        try:
            # 1. 分析性能数据
            performance_analysis = self.performance_analyzer.analyze(performance_data, context)
            
            # 2. 识别性能问题
            performance_issues = self._identify_performance_issues(performance_analysis)
            
            # 3. 分析错误模式
            error_analysis = self._analyze_error_patterns(performance_data, context)
            
            # 4. 生成改进建议
            improvement_suggestions = self._generate_improvement_suggestions(
                performance_analysis, performance_issues, error_analysis
            )
            
            # 5. 优先级排序
            prioritized_suggestions = self._prioritize_improvement_suggestions(improvement_suggestions)
            
            # 6. 生成反思见解
            reflection_insights = self._generate_reflection_insights(
                performance_analysis, performance_issues, error_analysis
            )
            
            # 7. 创建反思会话记录
            reflection_session = self._create_reflection_session(
                ReflectionType.PERFORMANCE_REFLECTION,
                start_time,
                time.time(),
                performance_analysis,
                reflection_insights,
                prioritized_suggestions
            )
            
            # 8. 更新学习数据
            self._update_learning_data(performance_analysis, error_analysis, reflection_insights)
            
            # 9. 生成最终结果
            final_result = self._generate_performance_reflection_result(
                reflection_session, performance_analysis, prioritized_suggestions
            )
            
            # 更新状态
            reflection_time = time.time() - start_time
            self.state["average_reflection_time"] = (
                self.state["average_reflection_time"] * 
                (self.state["total_reflection_sessions"] - 1) + reflection_time
            ) / self.state["total_reflection_sessions"]
            
            logger.info(
                f"性能反思完成: 数据点={len(performance_data)}, "
                f"问题数={len(performance_issues)}, "
                f"建议数={len(prioritized_suggestions)}, "
                f"时间={reflection_time:.2f}s"
            )
            
            return final_result
            
        except Exception as e:
            error_handler.handle_error(e, "SelfReflectionOptimizer", "性能反思失败")
            return self._generate_reflection_error_result("性能反思", str(e))
    
    def reflect_on_errors(self, 
                         error_data: List[Dict[str, Any]],
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        基于错误数据进行反思
        
        Args:
            error_data: 错误数据列表
            context: 上下文信息
            
        Returns:
            错误反思结果
        """
        start_time = time.time()
        self.state["total_reflection_sessions"] += 1
        
        try:
            # 1. 分析错误数据
            error_analysis = self.error_diagnoser.analyze(error_data, context)
            
            # 2. 识别错误模式
            error_patterns = self._identify_error_patterns(error_analysis)
            
            # 3. 分析根本原因
            root_causes = self._analyze_root_causes(error_patterns, context)
            
            # 4. 生成缓解策略
            mitigation_strategies = self._generate_mitigation_strategies(error_patterns, root_causes)
            
            # 5. 生成预防建议
            prevention_suggestions = self._generate_prevention_suggestions(error_patterns, root_causes)
            
            # 6. 创建反思会话记录
            reflection_session = self._create_reflection_session(
                ReflectionType.ERROR_REFLECTION,
                start_time,
                time.time(),
                {"error_data": error_data},
                {
                    "error_patterns": error_patterns,
                    "root_causes": root_causes,
                    "mitigation_strategies": mitigation_strategies
                },
                prevention_suggestions
            )
            
            # 7. 更新错误模式库
            self._update_error_pattern_library(error_patterns)
            
            # 8. 生成最终结果
            final_result = self._generate_error_reflection_result(
                reflection_session, error_patterns, root_causes, mitigation_strategies
            )
            
            logger.info(
                f"错误反思完成: 错误数={len(error_data)}, "
                f"模式数={len(error_patterns)}, "
                f"根本原因数={len(root_causes)}, "
                f"时间={time.time() - start_time:.2f}s"
            )
            
            return final_result
            
        except Exception as e:
            error_handler.handle_error(e, "SelfReflectionOptimizer", "错误反思失败")
            return self._generate_reflection_error_result("错误反思", str(e))
    
    def reflect_on_strategy(self, 
                           strategy_data: Dict[str, Any],
                           performance_outcomes: Dict[str, Any],
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        基于策略数据进行反思
        
        Args:
            strategy_data: 策略数据
            performance_outcomes: 性能结果
            context: 上下文信息
            
        Returns:
            策略反思结果
        """
        start_time = time.time()
        self.state["total_reflection_sessions"] += 1
        
        try:
            # 1. 分析策略效果
            strategy_analysis = self._analyze_strategy_effectiveness(strategy_data, performance_outcomes)
            
            # 2. 识别策略改进机会
            improvement_opportunities = self._identify_strategy_improvement_opportunities(strategy_analysis)
            
            # 3. 生成策略优化建议
            optimization_suggestions = self._generate_strategy_optimization_suggestions(
                strategy_analysis, improvement_opportunities
            )
            
            # 4. 分析策略适应需求
            adaptation_needs = self._analyze_strategy_adaptation_needs(strategy_analysis, context)
            
            # 5. 创建反思会话记录
            reflection_session = self._create_reflection_session(
                ReflectionType.STRATEGY_REFLECTION,
                start_time,
                time.time(),
                strategy_analysis,
                {
                    "improvement_opportunities": improvement_opportunities,
                    "adaptation_needs": adaptation_needs
                },
                optimization_suggestions
            )
            
            # 6. 更新策略学习数据
            self._update_strategy_learning_data(strategy_analysis, improvement_opportunities)
            
            # 7. 生成最终结果
            final_result = self._generate_strategy_reflection_result(
                reflection_session, strategy_analysis, optimization_suggestions, adaptation_needs
            )
            
            logger.info(
                f"策略反思完成: 策略数={len(strategy_data.get('strategies', []))}, "
                f"改进机会数={len(improvement_opportunities)}, "
                f"优化建议数={len(optimization_suggestions)}, "
                f"时间={time.time() - start_time:.2f}s"
            )
            
            return final_result
            
        except Exception as e:
            error_handler.handle_error(e, "SelfReflectionOptimizer", "策略反思失败")
            return self._generate_reflection_error_result("策略反思", str(e))
    
    def execute_improvement(self, 
                           improvement_id: str,
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行改进建议
        
        Args:
            improvement_id: 改进建议ID
            context: 上下文信息
            
        Returns:
            改进执行结果
        """
        start_time = time.time()
        
        try:
            # 1. 获取改进建议
            suggestion = self.improvement_suggestions.get(improvement_id)
            if not suggestion:
                return {
                    "success": False,
                    "error": f"改进建议不存在: {improvement_id}",
                    "improvement_id": improvement_id
                }
            
            # 2. 准备执行环境
            execution_context = self._prepare_improvement_execution(suggestion, context)
            
            # 3. 执行改进步骤
            execution_results = []
            for step in suggestion.implementation_steps:
                step_result = self._execute_improvement_step(step, execution_context)
                execution_results.append(step_result)
                
                # 如果步骤失败，停止执行
                if not step_result.get("success", False):
                    break
            
            # 4. 验证改进效果
            validation_results = self._validate_improvement_effect(suggestion, execution_results)
            
            # 5. 更新改进状态
            improvement_success = all(r.get("success", False) for r in execution_results)
            if improvement_success:
                self.state["total_improvements_implemented"] += 1
                
                # 评估性能影响
                performance_impact = self._assess_improvement_performance_impact(suggestion, validation_results)
                
                # 记录成功经验
                self._record_improvement_success(suggestion, execution_results, performance_impact)
            
            # 6. 生成最终结果
            final_result = self._generate_improvement_execution_result(
                suggestion, execution_results, validation_results, improvement_success
            )
            
            logger.info(
                f"改进执行完成: 建议ID={improvement_id}, "
                f"成功={improvement_success}, "
                f"步骤数={len(execution_results)}, "
                f"时间={time.time() - start_time:.2f}s"
            )
            
            return final_result
            
        except Exception as e:
            error_handler.handle_error(e, "SelfReflectionOptimizer", "改进执行失败")
            return {
                "success": False,
                "error": f"改进执行失败: {str(e)}",
                "improvement_id": improvement_id,
                "execution_time": time.time() - start_time
            }
    
    def get_performance_report(self, 
                              time_range: Optional[Tuple[float, float]] = None,
                              metric_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        获取性能报告
        
        Args:
            time_range: 时间范围（开始时间，结束时间）
            metric_types: 指标类型列表
            
        Returns:
            性能报告
        """
        try:
            # 1. 收集性能数据
            performance_data = self._collect_performance_data(time_range, metric_types)
            
            # 2. 分析性能趋势
            trend_analysis = self._analyze_performance_trends(performance_data)
            
            # 3. 识别关键问题
            key_issues = self._identify_key_performance_issues(performance_data, trend_analysis)
            
            # 4. 生成改进建议
            improvement_suggestions = self._generate_report_improvement_suggestions(key_issues)
            
            # 5. 生成报告
            report = self._generate_performance_report(
                performance_data, trend_analysis, key_issues, improvement_suggestions
            )
            
            logger.info(
                f"性能报告生成完成: 时间范围={time_range}, "
                f"指标类型数={len(metric_types or [])}, "
                f"数据点数={sum(len(data) for data in performance_data.values())}"
            )
            
            return report
            
        except Exception as e:
            error_handler.handle_error(e, "SelfReflectionOptimizer", "性能报告生成失败")
            return {
                "success": False,
                "error": f"性能报告生成失败: {str(e)}",
                "time_range": time_range,
                "metric_types": metric_types
            }
    
    def _identify_performance_issues(self, performance_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别性能问题"""
        issues = []
        thresholds = self.config["performance_monitoring"]["performance_thresholds"]
        
        for metric_name, metric_data in performance_analysis.get("metrics", {}).items():
            current_value = metric_data.get("current_value")
            threshold = thresholds.get(metric_name)
            
            if threshold is not None and current_value < threshold:
                issue = {
                    "metric": metric_name,
                    "current_value": current_value,
                    "threshold": threshold,
                    "deviation": threshold - current_value,
                    "severity": self._calculate_issue_severity(current_value, threshold),
                    "trend": metric_data.get("trend", "unknown")
                }
                issues.append(issue)
        
        # 按严重性排序
        issues.sort(key=lambda x: x["severity"], reverse=True)
        
        return issues
    
    def _analyze_error_patterns(self, performance_data: Dict[str, Any],
                               context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """分析错误模式"""
        try:
            # 提取错误数据
            error_data = performance_data.get("errors", [])
            if not error_data:
                return {"error_patterns": [], "analysis_complete": False}
            
            # 分析错误模式
            analysis = self.error_diagnoser.analyze(error_data, context)
            
            # 更新错误模式库
            for pattern in analysis.get("patterns", []):
                pattern_id = pattern.get("pattern_id")
                if pattern_id:
                    self.error_patterns[pattern_id] = ErrorPattern(
                        pattern_id=pattern_id,
                        error_type=pattern.get("error_type", "unknown"),
                        error_message=pattern.get("error_message", ""),
                        frequency=pattern.get("frequency", 1),
                        first_occurrence=pattern.get("first_occurrence", time.time()),
                        last_occurrence=pattern.get("last_occurrence", time.time()),
                        context_patterns=pattern.get("context_patterns", []),
                        root_causes=pattern.get("root_causes", []),
                        impact_score=pattern.get("impact_score", 0.5),
                        mitigation_strategies=pattern.get("mitigation_strategies", [])
                    )
            
            self.state["error_patterns_identified"] = len(self.error_patterns)
            
            return analysis
            
        except Exception as e:
            logger.error(f"分析错误模式失败: {e}")
            return {"error_patterns": [], "analysis_error": str(e)}
    
    def _generate_improvement_suggestions(self, 
                                         performance_analysis: Dict[str, Any],
                                         performance_issues: List[Dict[str, Any]],
                                         error_analysis: Dict[str, Any]) -> List[ImprovementSuggestion]:
        """生成改进建议"""
        suggestions = []
        
        # 基于性能问题生成建议
        for issue in performance_issues:
            suggestion = self._create_performance_improvement_suggestion(issue, performance_analysis)
            if suggestion:
                suggestions.append(suggestion)
        
        # 基于错误模式生成建议
        for pattern in error_analysis.get("patterns", []):
            suggestion = self._create_error_improvement_suggestion(pattern, error_analysis)
            if suggestion:
                suggestions.append(suggestion)
        
        # 基于趋势分析生成预防性建议
        trend_suggestions = self._generate_trend_based_suggestions(performance_analysis)
        suggestions.extend(trend_suggestions)
        
        return suggestions
    
    def _prioritize_improvement_suggestions(self, 
                                           suggestions: List[ImprovementSuggestion]) -> List[ImprovementSuggestion]:
        """优先级排序改进建议"""
        if not suggestions:
            return []
        
        # 计算优先级分数
        prioritized = []
        for suggestion in suggestions:
            priority_score = self._calculate_priority_score(suggestion)
            suggestion.priority_score = priority_score
            prioritized.append(suggestion)
        
        # 按优先级分数排序
        prioritized.sort(key=lambda x: x.priority_score, reverse=True)
        
        # 分配优先级等级
        for i, suggestion in enumerate(prioritized):
            if i < len(prioritized) * 0.2:  # 前20%
                suggestion.priority = ImprovementPriority.CRITICAL
            elif i < len(prioritized) * 0.5:  # 前50%
                suggestion.priority = ImprovementPriority.HIGH
            elif i < len(prioritized) * 0.8:  # 前80%
                suggestion.priority = ImprovementPriority.MEDIUM
            else:
                suggestion.priority = ImprovementPriority.LOW
        
        return prioritized
    
    def _generate_reflection_insights(self, 
                                     performance_analysis: Dict[str, Any],
                                     performance_issues: List[Dict[str, Any]],
                                     error_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成反思见解"""
        insights = []
        
        # 性能见解
        if performance_issues:
            worst_issue = max(performance_issues, key=lambda x: x["severity"])
            insights.append({
                "type": "performance_insight",
                "description": f"最严重的性能问题是 {worst_issue['metric']}，当前值为 {worst_issue['current_value']:.2f}，低于阈值 {worst_issue['threshold']:.2f}",
                "severity": worst_issue["severity"],
                "recommended_action": f"优先解决 {worst_issue['metric']} 性能问题"
            })
        
        # 错误见解
        error_patterns = error_analysis.get("patterns", [])
        if error_patterns:
            most_frequent = max(error_patterns, key=lambda x: x.get("frequency", 0))
            insights.append({
                "type": "error_insight",
                "description": f"最常见的错误是 {most_frequent.get('error_type', 'unknown')}，发生了 {most_frequent.get('frequency', 0)} 次",
                "frequency": most_frequent.get("frequency", 0),
                "recommended_action": "分析并解决此错误模式的根本原因"
            })
        
        # 趋势见解
        trends = performance_analysis.get("trends", {})
        for metric, trend in trends.items():
            if trend.get("direction") == "decreasing" and trend.get("strength", 0) > 0.7:
                insights.append({
                    "type": "trend_insight",
                    "description": f"{metric} 指标呈下降趋势，需要关注",
                    "metric": metric,
                    "trend_strength": trend.get("strength", 0),
                    "recommended_action": f"调查 {metric} 下降的原因"
                })
        
        return insights
    
    def _create_reflection_session(self, 
                                  reflection_type: ReflectionType,
                                  start_time: float,
                                  end_time: float,
                                  analysis_data: Dict[str, Any],
                                  insights: List[Dict[str, Any]],
                                  suggestions: List[ImprovementSuggestion]) -> ReflectionSession:
        """创建反思会话记录"""
        session_id = f"reflection_{int(time.time())}_{(zlib.adler32(str(str(analysis_data).encode('utf-8')) & 0xffffffff))}"
        
        session = ReflectionSession(
            session_id=session_id,
            reflection_type=reflection_type,
            start_time=start_time,
            end_time=end_time,
            focus_areas=list(analysis_data.keys()),
            insights_gained=insights,
            improvements_generated=suggestions,
            performance_impact={
                "session_duration": end_time - start_time,
                "insights_count": len(insights),
                "suggestions_count": len(suggestions),
                "analysis_complexity": len(str(analysis_data)) / 1000  # KB
            }
        )
        
        self.reflection_sessions[session_id] = session
        return session
    
    def _update_learning_data(self, 
                             performance_analysis: Dict[str, Any],
                             error_analysis: Dict[str, Any],
                             insights: List[Dict[str, Any]]) -> None:
        """更新学习数据"""
        # 更新性能趋势
        for metric, data in performance_analysis.get("metrics", {}).items():
            if "current_value" in data:
                self.learning_data["performance_trends"][metric].append(data["current_value"])
        
        # 更新错误相关性
        for pattern in error_analysis.get("patterns", []):
            error_type = pattern.get("error_type")
            context = pattern.get("context", {})
            
            if error_type and context:
                self.learning_data["error_correlation"][error_type].update(context)
        
        # 添加元认知见解
        for insight in insights:
            if insight.get("type") == "meta_cognitive_insight":
                self.learning_data["meta_cognitive_insights"].append(insight)
    
    def _generate_performance_reflection_result(self, 
                                               session: ReflectionSession,
                                               performance_analysis: Dict[str, Any],
                                               suggestions: List[ImprovementSuggestion]) -> Dict[str, Any]:
        """生成性能反思结果"""
        return {
            "success": True,
            "session_id": session.session_id,
            "reflection_type": session.reflection_type.value,
            "performance_analysis": {
                "metrics_summary": {k: v.get("current_value", 0) for k, v in performance_analysis.get("metrics", {}).items()},
                "issues_count": len(performance_analysis.get("issues", [])),
                "overall_performance": performance_analysis.get("overall_score", 0.5)
            },
            "insights": session.insights_gained,
            "improvement_suggestions": [
                {
                    "id": s.suggestion_id,
                    "title": s.title,
                    "priority": s.priority.value,
                    "expected_benefit": s.expected_benefit,
                    "implementation_cost": s.implementation_cost
                }
                for s in suggestions
            ],
            "recommendations": [
                "定期监控关键性能指标",
                "实施高优先级改进建议",
                "跟踪改进效果并进行调整"
            ],
            "performance_metrics": {
                "session_duration": session.performance_impact["session_duration"],
                "analysis_complexity": session.performance_impact["analysis_complexity"],
                "improvement_potential": sum(s.expected_benefit for s in suggestions) / max(len(suggestions), 1)
            }
        }
    
    def _generate_reflection_error_result(self, reflection_type: str, error_message: str) -> Dict[str, Any]:
        """生成反思错误结果"""
        return {
            "success": False,
            "error": f"{reflection_type}失败: {error_message}",
            "reflection_type": reflection_type,
            "timestamp": time.time(),
            "recommendations": [
                "检查输入数据的完整性和格式",
                "确保所有依赖组件正常工作",
                "简化反思范围或调整配置参数"
            ]
        }
    
    # 辅助方法（简化实现）
    def _calculate_issue_severity(self, current_value: float, threshold: float) -> float:
        """计算问题严重性"""
        deviation = threshold - current_value
        relative_deviation = deviation / threshold if threshold > 0 else 0
        return min(relative_deviation * 10, 1.0)
    
    def _create_performance_improvement_suggestion(self, 
                                                  issue: Dict[str, Any],
                                                  performance_analysis: Dict[str, Any]) -> Optional[ImprovementSuggestion]:
        """创建性能改进建议"""
        metric = issue["metric"]
        severity = issue["severity"]
        
        # 根据指标类型生成不同的建议
        suggestion_templates = {
            "planning_accuracy": {
                "title": f"提高{metric}性能",
                "description": f"当前{metric}值为{issue['current_value']:.2f}，低于阈值{issue['threshold']:.2f}。建议优化规划算法或增加训练数据。",
                "expected_benefit": severity * 0.8,
                "implementation_cost": 0.3,
                "implementation_effort": 0.4
            },
            "reasoning_quality": {
                "title": f"改进{metric}质量",
                "description": f"{metric}需要提升以改善推理结果。建议增强推理引擎的逻辑能力或知识表示。",
                "expected_benefit": severity * 0.7,
                "implementation_cost": 0.4,
                "implementation_effort": 0.5
            },
            "adaptation_success": {
                "title": f"增强{metric}能力",
                "description": f"适应能力不足影响系统性能。建议改进适应算法或增加适应策略。",
                "expected_benefit": severity * 0.6,
                "implementation_cost": 0.5,
                "implementation_effort": 0.6
            }
        }
        
        template = suggestion_templates.get(metric, {
            "title": f"优化{metric}",
            "description": f"需要改进{metric}性能。",
            "expected_benefit": severity * 0.5,
            "implementation_cost": 0.2,
            "implementation_effort": 0.3
        })
        
        suggestion_id = f"suggestion_{int(time.time())}_{(zlib.adler32(str(metric).encode('utf-8')) & 0xffffffff)}"
        
        return ImprovementSuggestion(
            suggestion_id=suggestion_id,
            title=template["title"],
            description=template["description"],
            priority=ImprovementPriority.HIGH if severity > 0.7 else ImprovementPriority.MEDIUM,
            expected_benefit=template["expected_benefit"],
            implementation_cost=template["implementation_cost"],
            implementation_effort=template["implementation_effort"],
            related_metrics=[metric],
            related_errors=[],
            implementation_steps=[
                {"step": 1, "action": f"分析{metric}问题原因", "duration": 30},
                {"step": 2, "action": "设计改进方案", "duration": 60},
                {"step": 3, "action": "实施优化", "duration": 120},
                {"step": 4, "action": "测试和验证", "duration": 90}
            ],
            validation_criteria=[
                {"criterion": f"{metric}提升", "target": issue["threshold"], "unit": "score"},
                {"criterion": "稳定性", "target": 0.8, "unit": "success_rate"}
            ]
        )
    
    def _create_error_improvement_suggestion(self, 
                                            pattern: Dict[str, Any],
                                            error_analysis: Dict[str, Any]) -> Optional[ImprovementSuggestion]:
        """创建错误改进建议"""
        error_type = pattern.get("error_type", "unknown")
        frequency = pattern.get("frequency", 1)
        impact = pattern.get("impact_score", 0.5)
        
        suggestion_id = f"error_suggestion_{int(time.time())}_{(zlib.adler32(str(error_type).encode('utf-8')) & 0xffffffff)}"
        
        return ImprovementSuggestion(
            suggestion_id=suggestion_id,
            title=f"解决{error_type}错误",
            description=f"此错误已发生{frequency}次，影响评分为{impact:.2f}。建议分析根本原因并实施修复。",
            priority=ImprovementPriority.CRITICAL if frequency > 5 else ImprovementPriority.HIGH,
            expected_benefit=impact * 0.9,
            implementation_cost=0.4,
            implementation_effort=0.5,
            related_metrics=[],
            related_errors=[error_type],
            implementation_steps=[
                {"step": 1, "action": "分析错误日志和上下文", "duration": 45},
                {"step": 2, "action": "识别根本原因", "duration": 60},
                {"step": 3, "action": "设计修复方案", "duration": 90},
                {"step": 4, "action": "实施和测试修复", "duration": 120}
            ],
            validation_criteria=[
                {"criterion": "错误频率减少", "target": 0.8, "unit": "reduction_rate"},
                {"criterion": "系统稳定性", "target": 0.9, "unit": "success_rate"}
            ]
        )
    
    def _generate_trend_based_suggestions(self, performance_analysis: Dict[str, Any]) -> List[ImprovementSuggestion]:
        """生成基于趋势的建议"""
        suggestions = []
        trends = performance_analysis.get("trends", {})
        
        for metric, trend in trends.items():
            if trend.get("direction") == "decreasing" and trend.get("strength", 0) > 0.6:
                suggestion_id = f"trend_suggestion_{int(time.time())}_{(zlib.adler32(str(metric).encode('utf-8')) & 0xffffffff)}"
                
                suggestion = ImprovementSuggestion(
                    suggestion_id=suggestion_id,
                    title=f"预防{metric}下降趋势",
                    description=f"{metric}呈下降趋势（强度{trend['strength']:.2f}），建议采取预防措施。",
                    priority=ImprovementPriority.MEDIUM,
                    expected_benefit=0.4,
                    implementation_cost=0.2,
                    implementation_effort=0.3,
                    related_metrics=[metric],
                    related_errors=[],
                    implementation_steps=[
                        {"step": 1, "action": "监控趋势变化", "duration": 30},
                        {"step": 2, "action": "分析可能原因", "duration": 60},
                        {"step": 3, "action": "制定预防策略", "duration": 90}
                    ],
                    validation_criteria=[
                        {"criterion": "趋势稳定或改善", "target": 0.7, "unit": "stability_score"}
                    ]
                )
                
                suggestions.append(suggestion)
        
        return suggestions
    
    def _calculate_priority_score(self, suggestion: ImprovementSuggestion) -> float:
        """计算优先级分数"""
        # 基于预期收益、成本和努力计算
        benefit = suggestion.expected_benefit
        cost = suggestion.implementation_cost
        effort = suggestion.implementation_effort
        
        # 收益成本比
        benefit_cost_ratio = benefit / max(cost, 0.01)
        
        # 努力调整因子
        effort_factor = 1.0 - effort * 0.5
        
        # 综合分数
        priority_score = benefit * benefit_cost_ratio * effort_factor
        
        return priority_score


# 组件类定义
class PerformanceAnalyzer:
    """性能分析器"""
    
    def analyze(self, performance_data: Dict[str, Any],
                context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """分析性能数据"""
        # 简化实现
        analysis = {
            "metrics": {},
            "trends": {},
            "issues": [],
            "overall_score": 0.5
        }
        
        # 分析各个指标
        for key, value in performance_data.items():
            if isinstance(value, (int, float)):
                analysis["metrics"][key] = {
                    "current_value": float(value),
                    "min_value": float(value) * 0.9,
                    "max_value": float(value) * 1.1,
                    "average": float(value),
                    "trend": "stable"
                }
        
        # 计算整体分数
        if analysis["metrics"]:
            values = [m["current_value"] for m in analysis["metrics"].values()]
            analysis["overall_score"] = sum(values) / len(values)
        
        # 识别趋势
        for metric, data in analysis["metrics"].items():
            if "planning" in metric.lower():
                analysis["trends"][metric] = {
                    "direction": "stable",
                    "strength": 0.5,
                    "confidence": 0.7
                }
        
        return analysis

class ErrorDiagnoser:
    """错误诊断器"""
    
    def analyze(self, error_data: List[Dict[str, Any]],
                context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """分析错误数据"""
        # 简化实现
        patterns = []
        
        # 按错误类型分组
        error_groups = defaultdict(list)
        for error in error_data:
            error_type = error.get("type", "unknown")
            error_groups[error_type].append(error)
        
        # 生成错误模式
        for error_type, errors in error_groups.items():
            pattern_id = f"pattern_{(zlib.adler32(str(error_type).encode('utf-8')) & 0xffffffff)}_{int(time.time())}"
            
            pattern = {
                "pattern_id": pattern_id,
                "error_type": error_type,
                "error_message": errors[0].get("message", "") if errors else "",
                "frequency": len(errors),
                "first_occurrence": min(e.get("timestamp", time.time()) for e in errors),
                "last_occurrence": max(e.get("timestamp", time.time()) for e in errors),
                "context_patterns": self._extract_context_patterns(errors),
                "root_causes": self._infer_root_causes(errors, error_type),
                "impact_score": self._calculate_impact_score(errors),
                "mitigation_strategies": self._generate_mitigation_strategies(error_type, errors)
            }
            
            patterns.append(pattern)
        
        return {
            "patterns": patterns,
            "total_errors": len(error_data),
            "unique_error_types": len(error_groups),
            "analysis_complete": True
        }
    
    def _extract_context_patterns(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取上下文模式"""
        # 简化实现
        patterns = []
        
        if errors:
            # 提取常见上下文特征
            context_features = defaultdict(int)
            for error in errors:
                context = error.get("context", {})
                for key, value in context.items():
                    if isinstance(value, str) and len(value) < 50:
                        context_features[f"{key}:{value}"] += 1
            
            # 选择最常见的前3个特征
            common_features = sorted(context_features.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for feature, count in common_features:
                patterns.append({
                    "feature": feature,
                    "frequency": count,
                    "confidence": count / len(errors)
                })
        
        return patterns
    
    def _infer_root_causes(self, errors: List[Dict[str, Any]], error_type: str) -> List[str]:
        """推断根本原因"""
        # 简化实现
        root_causes = []
        
        if "timeout" in error_type.lower():
            root_causes.append("资源不足或任务过载")
            root_causes.append("网络或IO延迟")
        
        if "memory" in error_type.lower():
            root_causes.append("内存泄漏")
            root_causes.append("数据结构过大")
        
        if "logic" in error_type.lower():
            root_causes.append("算法错误")
            root_causes.append("边界条件处理不当")
        
        # 默认原因
        if not root_causes:
            root_causes.append("未知原因，需要进一步分析")
        
        return root_causes[:3]
    
    def _calculate_impact_score(self, errors: List[Dict[str, Any]]) -> float:
        """计算影响评分"""
        # 基于频率、严重性和恢复时间计算
        frequency = len(errors)
        severity_sum = sum(e.get("severity", 0.5) for e in errors)
        recovery_sum = sum(e.get("recovery_time", 1.0) for e in errors)
        
        avg_severity = severity_sum / max(frequency, 1)
        avg_recovery = recovery_sum / max(frequency, 1)
        
        # 综合评分
        impact = (frequency * 0.3 + avg_severity * 0.4 + avg_recovery * 0.3) / 3.0
        return min(impact, 1.0)
    
    def _generate_mitigation_strategies(self, error_type: str, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成缓解策略"""
        strategies = []
        
        if "timeout" in error_type.lower():
            strategies.append({
                "strategy": "增加超时时间",
                "description": "适当增加任务超时限制",
                "effectiveness": 0.7,
                "implementation_cost": 0.2
            })
            strategies.append({
                "strategy": "优化资源分配",
                "description": "改进资源管理和分配策略",
                "effectiveness": 0.8,
                "implementation_cost": 0.4
            })
        
        if "memory" in error_type.lower():
            strategies.append({
                "strategy": "内存优化",
                "description": "检查和修复内存泄漏",
                "effectiveness": 0.9,
                "implementation_cost": 0.5
            })
        
        # 通用策略
        strategies.append({
            "strategy": "增加错误处理和恢复机制",
            "description": "改进错误处理和自动恢复能力",
            "effectiveness": 0.6,
            "implementation_cost": 0.3
        })
        
        return strategies[:3]

class ImprovementGenerator:
    """改进生成器"""
    
    def generate(self, issues: List[Dict[str, Any]],
                 context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成改进建议"""
        # 简化实现
        suggestions = []
        
        for i, issue in enumerate(issues):
            suggestion = {
                "suggestion_id": f"auto_suggestion_{int(time.time())}_{i}",
                "title": f"改进{issue.get('metric', '问题')}",
                "description": f"解决{issue.get('metric', '未知')}性能问题",
                "priority": "medium",
                "expected_benefit": issue.get("severity", 0.5) * 0.8,
                "implementation_cost": 0.3,
                "implementation_effort": 0.4
            }
            suggestions.append(suggestion)
        
        return suggestions

class AdaptationLearner:
    """自适应学习器"""
    
    def learn(self, experiences: List[Dict[str, Any]],
              context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """从经验中学习"""
        # 简化实现
        if not experiences:
            return {
                "success": False,
                "error": "没有经验数据",
                "learned_patterns": []
            }
        
        # 分析成功和失败模式
        successful = [e for e in experiences if e.get("success", False)]
        failed = [e for e in experiences if not e.get("success", False)]
        
        success_rate = len(successful) / len(experiences) if experiences else 0
        
        # 提取学习模式
        patterns = []
        
        if successful:
            patterns.append({
                "pattern_type": "success_pattern",
                "description": "成功案例的共同特征",
                "confidence": success_rate,
                "applicability": 0.7
            })
        
        if failed:
            patterns.append({
                "pattern_type": "failure_pattern",
                "description": "失败案例的常见原因",
                "confidence": 1 - success_rate,
                "applicability": 0.6
            })
        
        return {
            "success": True,
            "learned_patterns": patterns,
            "learning_metrics": {
                "experience_count": len(experiences),
                "success_rate": success_rate,
                "pattern_count": len(patterns)
            }
        }


# 工厂函数
def create_self_reflection_optimizer(config: Optional[Dict[str, Any]] = None) -> SelfReflectionOptimizer:
    """
    创建自我反思优化器实例
    
    Args:
        config: 可选配置字典
        
    Returns:
        自我反思优化器实例
    """
    try:
        optimizer = SelfReflectionOptimizer(config)
        logger.info("自我反思优化器实例创建成功")
        return optimizer
    except Exception as e:
        logger.error(f"创建自我反思优化器失败: {e}")
        raise


# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("自我反思优化器测试")
    print("=" * 60)
    
    try:
        # 创建优化器
        optimizer = create_self_reflection_optimizer()
        print("✓ 自我反思优化器创建成功")
        
        # 测试性能反思
        performance_data = {
            "planning_accuracy": 0.65,
            "reasoning_quality": 0.58,
            "adaptation_success": 0.72,
            "cross_domain_transfer": 0.45,
            "errors": [
                {"type": "timeout", "message": "任务执行超时", "timestamp": time.time() - 3600, "severity": 0.7},
                {"type": "memory", "message": "内存不足", "timestamp": time.time() - 1800, "severity": 0.8},
                {"type": "timeout", "message": "任务执行超时", "timestamp": time.time() - 900, "severity": 0.6}
            ]
        }
        
        reflection_result = optimizer.reflect_on_performance(performance_data, {"test_mode": True})
        
        if reflection_result["success"]:
            print(f"✓ 性能反思成功: 会话ID={reflection_result['session_id']}")
            print(f"  性能问题数: {reflection_result['performance_analysis']['issues_count']}")
            print(f"  改进建议数: {len(reflection_result['improvement_suggestions'])}")
            
            # 显示前3个建议
            for i, suggestion in enumerate(reflection_result['improvement_suggestions'][:3], 1):
                print(f"  建议{i}: {suggestion['title']} (优先级: {suggestion['priority']})")
        else:
            print(f"✗ 性能反思失败: {reflection_result.get('error', '未知错误')}")
        
        # 测试错误反思
        error_data = [
            {"type": "logic_error", "message": "除零错误", "timestamp": time.time() - 7200, "context": {"module": "calculator"}},
            {"type": "logic_error", "message": "除零错误", "timestamp": time.time() - 3600, "context": {"module": "calculator"}},
            {"type": "io_error", "message": "文件读取失败", "timestamp": time.time() - 1800, "context": {"file": "data.txt"}}
        ]
        
        error_reflection_result = optimizer.reflect_on_errors(error_data, {"test_mode": True})
        
        if error_reflection_result["success"]:
            print(f"✓ 错误反思成功: 错误数={error_data}")
            print(f"  错误模式数: {error_reflection_result.get('error_pattern_count', 0)}")
        else:
            print(f"✗ 错误反思失败: {error_reflection_result.get('error', '未知错误')}")
        
        # 测试性能报告
        report_result = optimizer.get_performance_report(
            time_range=(time.time() - 86400, time.time()),
            metric_types=["planning_accuracy", "reasoning_quality"]
        )
        
        if report_result["success"]:
            print(f"✓ 性能报告生成成功")
            print(f"  报告范围: {report_result.get('time_range', '未知')}")
        else:
            print(f"✗ 性能报告生成失败: {report_result.get('error', '未知错误')}")
        
        print("=" * 60)
        print("测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()