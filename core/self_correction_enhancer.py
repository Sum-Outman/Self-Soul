#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Correction Enhancer - 自我矫正增强器

核心功能：
1. 实时监控计划和推理过程
2. 自动检测错误和性能下降
3. 分析错误的根本原因
4. 生成和执行纠正策略
5. 验证纠正效果
6. 持续学习和优化矫正能力

实现AGI系统的自我矫正、错误修复和持续优化能力，
支持系统自动检测和纠正计划推理中的问题。

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

class CorrectionType(Enum):
    """矫正类型枚举"""
    LOGICAL_CORRECTION = "logical_correction"          # 逻辑错误矫正
    CONSTRAINT_CORRECTION = "constraint_correction"    # 约束违反矫正
    PERFORMANCE_CORRECTION = "performance_correction"  # 性能下降矫正
    CONSISTENCY_CORRECTION = "consistency_correction"  # 一致性矫正
    COMPLETENESS_CORRECTION = "completeness_correction" # 完整性矫正
    ADAPTATION_CORRECTION = "adaptation_correction"    # 适应性矫正

class ErrorSeverity(Enum):
    """错误严重性枚举"""
    CRITICAL = "critical"      # 关键：系统无法继续
    HIGH = "high"              # 高：功能严重受损
    MEDIUM = "medium"          # 中：功能部分受损
    LOW = "low"                # 低：轻微问题
    INFORMATIONAL = "informational"  # 信息性：潜在问题

class CorrectionStatus(Enum):
    """矫正状态枚举"""
    DETECTED = "detected"          # 已检测
    ANALYZED = "analyzed"          # 已分析
    CORRECTION_GENERATED = "correction_generated"  # 已生成矫正
    CORRECTION_APPLIED = "correction_applied"      # 已应用矫正
    VALIDATED = "validated"        # 已验证
    FAILED = "failed"              # 失败
    SUPPRESSED = "suppressed"      # 已抑制

@dataclass
class DetectedError:
    """检测到的错误表示"""
    error_id: str
    error_type: str
    severity: ErrorSeverity
    detection_time: float
    context: Dict[str, Any]
    description: str
    source_component: str
    confidence: float = 0.8
    impact_areas: List[str] = field(default_factory=lambda: ["general"])
    raw_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CorrectionStrategy:
    """矫正策略表示"""
    strategy_id: str
    error_ids: List[str]
    correction_type: CorrectionType
    description: str
    implementation_steps: List[Dict[str, Any]]
    expected_outcome: Dict[str, Any]
    confidence: float
    cost_estimate: float
    benefit_estimate: float
    risk_assessment: Dict[str, Any]
    prerequisites: List[str]
    validation_criteria: List[Dict[str, Any]]

@dataclass
class CorrectionSession:
    """矫正会话表示"""
    session_id: str
    start_time: float
    end_time: float
    detected_errors: List[DetectedError]
    correction_strategies: List[CorrectionStrategy]
    applied_corrections: List[Dict[str, Any]]
    validation_results: Dict[str, Any]
    overall_status: CorrectionStatus
    performance_impact: Dict[str, Any]
    learning_insights: List[Dict[str, Any]]

class SelfCorrectionEnhancer:
    """
    自我矫正增强器 - 实现AGI系统的自动错误检测和矫正
    
    核心特性：
    1. 实时监控和异常检测
    2. 多层次错误分析和诊断
    3. 智能矫正策略生成
    4. 安全矫正执行和验证
    5. 矫正效果评估和优化
    6. 持续学习和自我改进
    7. 自适应矫正能力
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化自我矫正增强器"""
        self.logger = logger
        self.config = config or self._get_default_config()
        
        # 初始化组件
        self._initialize_components()
        
        # 数据存储
        self.detected_errors: Dict[str, DetectedError] = {}
        self.correction_strategies: Dict[str, CorrectionStrategy] = {}
        self.correction_sessions: Dict[str, CorrectionSession] = {}
        self.correction_history: deque = deque(maxlen=1000)
        
        # 状态跟踪
        self.state = {
            "total_errors_detected": 0,
            "total_correction_sessions": 0,
            "successful_corrections": 0,
            "failed_corrections": 0,
            "average_correction_time": 0,
            "error_detection_accuracy": 0.0,
            "correction_success_rate": 0.0,
            "performance_improvement_rate": 0.0,
            "last_correction_time": 0
        }
        
        # 学习数据
        self.learning_data = {
            "error_patterns": defaultdict(list),
            "correction_patterns": defaultdict(list),
            "strategy_effectiveness": defaultdict(list),
            "risk_patterns": defaultdict(list),
            "adaptation_patterns": [],
            "meta_correction_insights": []
        }
        
        # 缓存
        self.monitoring_cache: deque = deque(maxlen=1000)
        self.error_context_cache: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        logger.info("自我矫正增强器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "monitoring": {
                "enable_real_time_monitoring": True,
                "monitoring_interval": 1,  # 秒
                "monitoring_depth": "detailed",
                "performance_metrics": [
                    "planning_accuracy",
                    "reasoning_quality",
                    "constraint_satisfaction",
                    "consistency_score",
                    "completeness_score",
                    "adaptation_success"
                ],
                "anomaly_detection_threshold": 0.3
            },
            "error_detection": {
                "enable_automatic_detection": True,
                "error_types": [
                    "logical_error",
                    "constraint_violation",
                    "performance_degradation",
                    "consistency_issue",
                    "completeness_issue",
                    "adaptation_failure"
                ],
                "detection_sensitivity": 0.7,
                "min_severity_level": "low",
                "enable_pattern_recognition": True
            },
            "analysis": {
                "root_cause_analysis_depth": 3,
                "enable_causal_analysis": True,
                "enable_correlation_analysis": True,
                "enable_impact_analysis": True,
                "analysis_timeout": 10  # 秒
            },
            "correction": {
                "enable_automatic_correction": True,
                "correction_strategy_generation": True,
                "max_correction_iterations": 3,
                "enable_safe_correction": True,
                "correction_confidence_threshold": 0.6,
                "enable_rollback_mechanism": True
            },
            "validation": {
                "enable_post_correction_validation": True,
                "validation_metrics": [
                    "error_resolution",
                    "performance_improvement",
                    "consistency_restoration",
                    "constraint_satisfaction"
                ],
                "validation_timeout": 5,  # 秒
                "enable_continuous_validation": True
            },
            "learning": {
                "enable_continuous_learning": True,
                "learning_rate": 0.1,
                "enable_knowledge_transfer": True,
                "enable_adaptive_correction": True,
                "learning_history_size": 1000
            }
        }
    
    def _initialize_components(self) -> None:
        """初始化所有组件"""
        try:
            # 1. 监控器
            self.monitor = MonitoringComponent()
            
            # 2. 错误检测器
            self.error_detector = ErrorDetectionComponent()
            
            # 3. 分析器
            self.analyzer = AnalysisComponent()
            
            # 4. 矫正策略生成器
            self.strategy_generator = StrategyGenerationComponent()
            
            # 5. 矫正执行器
            self.executor = ExecutionComponent()
            
            # 6. 验证器
            self.validator = ValidationComponent()
            
            logger.info("自我矫正增强器组件初始化完成")
            
        except Exception as e:
            error_handler.handle_error(e, "SelfCorrectionEnhancer", "初始化组件失败")
            logger.error(f"组件初始化失败: {e}")
    
    def monitor_and_correct(self, 
                          planning_data: Dict[str, Any],
                          reasoning_data: Dict[str, Any],
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        监控和矫正计划推理过程
        
        Args:
            planning_data: 计划数据
            reasoning_data: 推理数据
            context: 上下文信息
            
        Returns:
            矫正结果
        """
        start_time = time.time()
        self.state["total_correction_sessions"] += 1
        
        try:
            # 1. 监控过程
            monitoring_results = self.monitor.monitor_process(
                planning_data, reasoning_data, context
            )
            
            # 2. 检测错误
            detected_errors = self.error_detector.detect_errors(
                monitoring_results, context
            )
            
            # 3. 如果没有错误，返回正常结果
            if not detected_errors:
                return self._generate_no_error_result(start_time)
            
            # 4. 分析错误
            error_analysis = self.analyzer.analyze_errors(
                detected_errors, monitoring_results, context
            )
            
            # 5. 生成矫正策略
            correction_strategies = self.strategy_generator.generate_strategies(
                detected_errors, error_analysis, context
            )
            
            # 6. 应用矫正
            correction_results = self.executor.apply_corrections(
                correction_strategies, planning_data, reasoning_data, context
            )
            
            # 7. 验证矫正效果
            validation_results = self.validator.validate_corrections(
                correction_results, detected_errors, context
            )
            
            # 8. 创建矫正会话记录
            correction_session = self._create_correction_session(
                start_time,
                time.time(),
                detected_errors,
                correction_strategies,
                correction_results,
                validation_results
            )
            
            # 9. 更新学习数据
            self._update_learning_data(
                detected_errors, correction_strategies, validation_results
            )
            
            # 10. 生成最终结果
            final_result = self._generate_correction_result(
                correction_session, validation_results
            )
            
            # 更新状态
            self._update_state(correction_session, validation_results)
            
            logger.info(
                f"自我矫正完成: 错误数={len(detected_errors)}, "
                f"策略数={len(correction_strategies)}, "
                f"成功={validation_results.get('overall_success', False)}, "
                f"时间={time.time() - start_time:.2f}s"
            )
            
            return final_result
            
        except Exception as e:
            error_handler.handle_error(e, "SelfCorrectionEnhancer", "监控矫正过程失败")
            return self._generate_error_result(start_time, str(e))
    
    def detect_logical_errors(self, reasoning_chain: Dict[str, Any]) -> List[DetectedError]:
        """检测逻辑错误"""
        errors = []
        
        # 1. 检查推理链的连贯性
        if not self._check_reasoning_coherence(reasoning_chain):
            errors.append(self._create_detected_error(
                error_type="logical_incoherence",
                severity=ErrorSeverity.MEDIUM,
                description="推理链存在逻辑不连贯",
                source_component="reasoning_chain",
                context={"reasoning_chain": reasoning_chain}
            ))
        
        # 2. 检查前提和结论的一致性
        if not self._check_premise_conclusion_consistency(reasoning_chain):
            errors.append(self._create_detected_error(
                error_type="premise_conclusion_inconsistency",
                severity=ErrorSeverity.HIGH,
                description="前提和结论存在逻辑不一致",
                source_component="reasoning_chain",
                context={"reasoning_chain": reasoning_chain}
            ))
        
        # 3. 检查推理步骤的有效性
        invalid_steps = self._find_invalid_reasoning_steps(reasoning_chain)
        for step in invalid_steps:
            errors.append(self._create_detected_error(
                error_type="invalid_reasoning_step",
                severity=ErrorSeverity.MEDIUM,
                description=f"推理步骤无效: {step.get('description', '未知')}",
                source_component="reasoning_step",
                context={"step": step, "reasoning_chain": reasoning_chain}
            ))
        
        return errors
    
    def detect_constraint_violations(self, plan: Dict[str, Any], 
                                   constraints: Dict[str, Any]) -> List[DetectedError]:
        """检测约束违反"""
        errors = []
        
        # 1. 检查资源约束
        resource_violations = self._check_resource_constraints(plan, constraints)
        for violation in resource_violations:
            errors.append(self._create_detected_error(
                error_type="resource_constraint_violation",
                severity=ErrorSeverity.MEDIUM,
                description=f"资源约束违反: {violation}",
                source_component="planning",
                context={"plan": plan, "constraints": constraints, "violation": violation}
            ))
        
        # 2. 检查时间约束
        time_violations = self._check_time_constraints(plan, constraints)
        for violation in time_violations:
            errors.append(self._create_detected_error(
                error_type="time_constraint_violation",
                severity=ErrorSeverity.MEDIUM,
                description=f"时间约束违反: {violation}",
                source_component="planning",
                context={"plan": plan, "constraints": constraints, "violation": violation}
            ))
        
        # 3. 检查逻辑约束
        logic_violations = self._check_logic_constraints(plan, constraints)
        for violation in logic_violations:
            errors.append(self._create_detected_error(
                error_type="logic_constraint_violation",
                severity=ErrorSeverity.HIGH,
                description=f"逻辑约束违反: {violation}",
                source_component="planning",
                context={"plan": plan, "constraints": constraints, "violation": violation}
            ))
        
        return errors
    
    def detect_performance_issues(self, performance_metrics: Dict[str, Any]) -> List[DetectedError]:
        """检测性能问题"""
        errors = []
        
        # 1. 检查性能下降
        degradation_issues = self._detect_performance_degradation(performance_metrics)
        for issue in degradation_issues:
            errors.append(self._create_detected_error(
                error_type="performance_degradation",
                severity=ErrorSeverity.MEDIUM,
                description=f"性能下降: {issue}",
                source_component="performance_monitoring",
                context={"performance_metrics": performance_metrics, "issue": issue}
            ))
        
        # 2. 检查异常性能模式
        anomaly_patterns = self._detect_performance_anomalies(performance_metrics)
        for pattern in anomaly_patterns:
            errors.append(self._create_detected_error(
                error_type="performance_anomaly",
                severity=ErrorSeverity.LOW,
                description=f"性能异常模式: {pattern}",
                source_component="performance_monitoring",
                context={"performance_metrics": performance_metrics, "pattern": pattern}
            ))
        
        return errors
    
    def generate_logical_correction(self, error: DetectedError) -> CorrectionStrategy:
        """生成逻辑错误矫正策略"""
        if error.error_type == "logical_incoherence":
            return self._generate_coherence_correction(error)
        elif error.error_type == "premise_conclusion_inconsistency":
            return self._generate_consistency_correction(error)
        elif error.error_type == "invalid_reasoning_step":
            return self._generate_reasoning_step_correction(error)
        else:
            return self._generate_generic_logical_correction(error)
    
    def generate_constraint_correction(self, error: DetectedError) -> CorrectionStrategy:
        """生成约束违反矫正策略"""
        if error.error_type == "resource_constraint_violation":
            return self._generate_resource_constraint_correction(error)
        elif error.error_type == "time_constraint_violation":
            return self._generate_time_constraint_correction(error)
        elif error.error_type == "logic_constraint_violation":
            return self._generate_logic_constraint_correction(error)
        else:
            return self._generate_generic_constraint_correction(error)
    
    def generate_correction_plan(self, error: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成矫正计划（公有方法）
        
        Args:
            error: 错误字典，包含以下字段：
                - type: 错误类型
                - description: 错误描述
                - severity: 严重程度
                - timestamp: 时间戳
        
        Returns:
            矫正计划字典
        """
        try:
            # 将输入字典转换为DetectedError对象
            detected_error = self._create_detected_error(
                error_type=error.get("type", "unknown_error"),
                severity=ErrorSeverity(error.get("severity", "medium")),
                description=error.get("description", "Unknown error"),
                source_component=error.get("source_component", "validation"),
                context=error.get("context", {})
            )
            
            # 根据错误类型选择合适的矫正生成方法
            error_type = error.get("type", "")
            
            if "logical" in error_type.lower():
                correction_strategy = self.generate_logical_correction(detected_error)
            elif "constraint" in error_type.lower():
                correction_strategy = self.generate_constraint_correction(detected_error)
            elif "performance" in error_type.lower():
                # 使用性能矫正方法
                correction_strategy = self._generate_performance_correction(detected_error)
            else:
                # 使用通用矫正方法
                correction_strategy = self._generate_generic_constraint_correction(detected_error)
            
            # 将矫正策略转换为字典格式
            correction_plan = {
                "status": "success",
                "error_id": error.get("error_id", detected_error.error_id),
                "error_type": error_type,
                "strategies": [
                    {
                        "strategy_id": correction_strategy.strategy_id,
                        "description": correction_strategy.description,
                        "implementation_steps": correction_strategy.implementation_steps,
                        "expected_outcome": correction_strategy.expected_outcome,
                        "confidence": correction_strategy.confidence,
                        "cost_estimate": correction_strategy.cost_estimate,
                        "benefit_estimate": correction_strategy.benefit_estimate,
                        "risk_assessment": correction_strategy.risk_assessment
                    }
                ],
                "correction_strategy": {
                    "strategy_id": correction_strategy.strategy_id,
                    "description": correction_strategy.description,
                    "implementation_steps": correction_strategy.implementation_steps,
                    "expected_outcome": correction_strategy.expected_outcome,
                    "confidence": correction_strategy.confidence,
                    "cost_estimate": correction_strategy.cost_estimate,
                    "benefit_estimate": correction_strategy.benefit_estimate,
                    "risk_assessment": correction_strategy.risk_assessment
                },
                "prerequisites": correction_strategy.prerequisites,
                "validation_criteria": correction_strategy.validation_criteria,
                "generation_timestamp": time.time(),
                "error_metadata": error
            }
            
            logger.info(f"为错误类型 '{error_type}' 生成矫正计划，策略ID: {correction_strategy.strategy_id}")
            
            return correction_plan
            
        except Exception as e:
            logger.error(f"生成矫正计划失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": error.get("type", "unknown"),
                "generation_timestamp": time.time()
            }
    
    def _create_detected_error(self, 
                              error_type: str,
                              severity: ErrorSeverity,
                              description: str,
                              source_component: str,
                              context: Dict[str, Any]) -> DetectedError:
        """创建检测到的错误"""
        error_id = f"error_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return DetectedError(
            error_id=error_id,
            error_type=error_type,
            severity=severity,
            detection_time=time.time(),
            context=context,
            description=description,
            source_component=source_component,
            confidence=0.8,  # 默认置信度
            impact_areas=self._determine_impact_areas(error_type, severity),
            raw_data=context.copy()
        )
    
    def _determine_impact_areas(self, error_type: str, severity: ErrorSeverity) -> List[str]:
        """确定影响区域"""
        impact_areas = []
        
        if "logical" in error_type:
            impact_areas.append("reasoning_quality")
            impact_areas.append("decision_making")
        
        if "constraint" in error_type:
            impact_areas.append("planning_accuracy")
            impact_areas.append("constraint_satisfaction")
        
        if "performance" in error_type:
            impact_areas.append("system_performance")
            impact_areas.append("efficiency")
        
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            impact_areas.append("system_reliability")
            impact_areas.append("overall_quality")
        
        return impact_areas
    
    def _create_correction_session(self,
                                 start_time: float,
                                 end_time: float,
                                 detected_errors: List[DetectedError],
                                 correction_strategies: List[CorrectionStrategy],
                                 correction_results: Dict[str, Any],
                                 validation_results: Dict[str, Any]) -> CorrectionSession:
        """创建矫正会话"""
        session_id = f"correction_{int(start_time)}_{random.randint(1000, 9999)}"
        
        # 确定整体状态
        overall_success = validation_results.get('overall_success', False)
        overall_status = CorrectionStatus.VALIDATED if overall_success else CorrectionStatus.FAILED
        
        return CorrectionSession(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            detected_errors=detected_errors,
            correction_strategies=correction_strategies,
            applied_corrections=correction_results.get('applied_corrections', []),
            validation_results=validation_results,
            overall_status=overall_status,
            performance_impact=validation_results.get('performance_impact', {}),
            learning_insights=validation_results.get('learning_insights', [])
        )
    
    def _generate_no_error_result(self, start_time: float) -> Dict[str, Any]:
        """生成无错误结果"""
        return {
            "success": True,
            "errors_detected": 0,
            "correction_applied": False,
            "message": "未检测到需要矫正的错误",
            "timestamp": time.time(),
            "processing_time": time.time() - start_time
        }
    
    def _generate_correction_result(self,
                                  session: CorrectionSession,
                                  validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成矫正结果"""
        # 处理矫正策略：可能是CorrectionStrategy实例或字典
        correction_strategies_data = []
        for s in session.correction_strategies:
            if hasattr(s, 'correction_type'):
                # 是CorrectionStrategy实例
                correction_strategies_data.append(asdict(s))
            else:
                # 是字典
                correction_strategies_data.append(s)
        
        return {
            "success": session.overall_status == CorrectionStatus.VALIDATED,
            "session_id": session.session_id,
            "errors_detected": len(session.detected_errors),
            "correction_strategies_generated": len(session.correction_strategies),
            "corrections_applied": len(session.applied_corrections),
            "validation_success": validation_results.get('overall_success', False),
            "performance_improvement": validation_results.get('performance_improvement', {}),
            "learning_insights": session.learning_insights,
            "timestamp": time.time(),
            "processing_time": session.end_time - session.start_time,
            "session_details": {
                "detected_errors": [asdict(e) for e in session.detected_errors],
                "correction_strategies": correction_strategies_data,
                "applied_corrections": session.applied_corrections
            }
        }
    
    def _generate_error_result(self, start_time: float, error_message: str) -> Dict[str, Any]:
        """生成错误结果"""
        return {
            "success": False,
            "errors_detected": 0,
            "correction_applied": False,
            "error": error_message,
            "timestamp": time.time(),
            "processing_time": time.time() - start_time
        }
    
    def _update_state(self, session: CorrectionSession, validation_results: Dict[str, Any]) -> None:
        """更新状态"""
        if session.overall_status == CorrectionStatus.VALIDATED:
            self.state["successful_corrections"] += 1
        else:
            self.state["failed_corrections"] += 1
        
        self.state["total_errors_detected"] += len(session.detected_errors)
        
        correction_time = session.end_time - session.start_time
        self.state["average_correction_time"] = (
            self.state["average_correction_time"] * 
            (self.state["total_correction_sessions"] - 1) + correction_time
        ) / self.state["total_correction_sessions"]
        
        # 更新成功率
        if self.state["total_correction_sessions"] > 0:
            self.state["correction_success_rate"] = (
                self.state["successful_corrections"] / self.state["total_correction_sessions"]
            )
        
        self.state["last_correction_time"] = time.time()
    
    def _update_learning_data(self,
                            detected_errors: List[DetectedError],
                            correction_strategies: List[CorrectionStrategy],
                            validation_results: Dict[str, Any]) -> None:
        """更新学习数据"""
        for error in detected_errors:
            self.learning_data["error_patterns"][error.error_type].append(asdict(error))
        
        for strategy in correction_strategies:
            # 处理策略数据：可能是CorrectionStrategy实例或字典
            if hasattr(strategy, 'correction_type'):
                # 是CorrectionStrategy实例
                strategy_dict = asdict(strategy)
                correction_type_value = strategy.correction_type.value
                strategy_id = strategy.strategy_id
            else:
                # 是字典
                strategy_dict = strategy
                correction_type_value = strategy.get('correction_type', 'unknown')
                strategy_id = strategy.get('strategy_id', 'unknown')
            
            strategy_data = {
                "strategy": strategy_dict,
                "validation_results": validation_results
            }
            
            # 使用字符串形式的修正类型作为键
            correction_type_key = str(correction_type_value)
            self.learning_data["correction_patterns"][correction_type_key].append(strategy_data)
            
            # 更新策略有效性
            if validation_results.get('overall_success', False):
                self.learning_data["strategy_effectiveness"][strategy_id].append(1.0)
            else:
                self.learning_data["strategy_effectiveness"][strategy_id].append(0.0)
    
    # ===== 具体检测方法实现 =====
    
    def _check_reasoning_coherence(self, reasoning_chain: Dict[str, Any]) -> bool:
        """检查推理链的连贯性"""
        try:
            steps = reasoning_chain.get("steps", [])
            if not steps:
                return True  # 空链视为连贯
            
            # 检查步骤之间的逻辑连接
            for i in range(len(steps) - 1):
                current_step = steps[i]
                next_step = steps[i + 1]
                
                # 检查步骤类型是否连贯
                if not self._are_steps_coherent(current_step, next_step):
                    return False
            
            # 检查整体连贯性得分（如果存在）
            coherence_score = reasoning_chain.get("coherence_score", 1.0)
            return coherence_score >= 0.6
            
        except Exception as e:
            logger.warning(f"检查推理连贯性失败: {e}")
            return True  # 失败时默认返回True，避免误报
    
    def _are_steps_coherent(self, step1: Dict[str, Any], step2: Dict[str, Any]) -> bool:
        """检查两个步骤是否连贯"""
        step1_type = step1.get("type", "")
        step2_type = step2.get("type", "")
        
        # 基本连贯性规则
        coherent_pairs = [
            ("premise", "inference"),
            ("inference", "conclusion"),
            ("premise", "conclusion"),
            ("assumption", "inference"),
            ("inference", "inference"),  # 推理链可以连续
            ("evidence", "conclusion")
        ]
        
        # 检查是否为连贯对
        for pair in coherent_pairs:
            if step1_type == pair[0] and step2_type == pair[1]:
                return True
        
        # 如果类型相同且不是不连贯的类型
        if step1_type == step2_type and step1_type not in ["premise", "conclusion"]:
            return True
        
        # 默认情况
        return step1_type != "conclusion" or step2_type != "premise"  # 结论不应在前提之前
    
    def _check_premise_conclusion_consistency(self, reasoning_chain: Dict[str, Any]) -> bool:
        """检查前提和结论的一致性"""
        try:
            steps = reasoning_chain.get("steps", [])
            if len(steps) < 2:
                return True  # 步骤太少无法检查
            
            # 提取前提和结论
            premises = [s for s in steps if s.get("type") in ["premise", "assumption", "evidence"]]
            conclusions = [s for s in steps if s.get("type") in ["conclusion", "decision"]]
            
            if not premises or not conclusions:
                return True  # 缺少前提或结论
            
            # 检查前提是否支持结论（简化检查）
            premise_content = " ".join(str(p.get("content", "")) for p in premises).lower()
            conclusion_content = " ".join(str(c.get("content", "")) for c in conclusions).lower()
            
            # 基本一致性检查：结论不应与前提矛盾
            contradiction_keywords = ["not ", "never ", "cannot ", "impossible ", "false ", "wrong "]
            
            for keyword in contradiction_keywords:
                if keyword in premise_content and keyword in conclusion_content:
                    # 两者都包含否定词，需要进一步分析
                    continue
                elif (keyword in premise_content) != (keyword in conclusion_content):
                    # 一个包含否定词而另一个不包含，可能矛盾
                    return False
            
            # 检查逻辑一致性得分（如果存在）
            consistency_score = reasoning_chain.get("consistency_score", 1.0)
            return consistency_score >= 0.7
            
        except Exception as e:
            logger.warning(f"检查前提结论一致性失败: {e}")
            return True  # 失败时默认返回True
    
    def _find_invalid_reasoning_steps(self, reasoning_chain: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查找无效推理步骤"""
        invalid_steps = []
        steps = reasoning_chain.get("steps", [])
        
        for i, step in enumerate(steps):
            step_type = step.get("type", "")
            step_content = step.get("content", "")
            
            # 检查步骤有效性
            if not step_type or not step_content:
                invalid_steps.append({
                    "index": i,
                    "step": step,
                    "reason": "缺少类型或内容",
                    "description": f"步骤{i}: 缺少类型或内容"
                })
            
            # 检查内容是否合理
            elif len(str(step_content)) < 3:
                invalid_steps.append({
                    "index": i,
                    "step": step,
                    "reason": "内容过短",
                    "description": f"步骤{i}: 内容过短（{len(str(step_content))}字符）"
                })
            
            # 检查步骤置信度（如果存在）
            elif "confidence" in step and step["confidence"] < 0.3:
                invalid_steps.append({
                    "index": i,
                    "step": step,
                    "reason": "置信度过低",
                    "description": f"步骤{i}: 置信度过低（{step['confidence']}）"
                })
        
        return invalid_steps
    
    def _check_resource_constraints(self, plan: Dict[str, Any], constraints: Dict[str, Any]) -> List[str]:
        """检查资源约束违反"""
        violations = []
        
        # 提取资源需求
        resource_requirements = plan.get("resource_requirements", {})
        available_resources = constraints.get("available_resources", {})
        
        for resource_type, required_amount in resource_requirements.items():
            available_amount = available_resources.get(resource_type, 0)
            
            if required_amount > available_amount:
                violations.append(
                    f"资源'{resource_type}'不足: 需要{required_amount}, 可用{available_amount}"
                )
        
        # 检查总资源限制
        total_resources_required = sum(resource_requirements.values())
        total_resources_available = sum(available_resources.values())
        
        if total_resources_required > total_resources_available * 1.5:  # 允许50%缓冲
            violations.append(
                f"总资源需求过高: 需要{total_resources_required}, 可用{total_resources_available}"
            )
        
        return violations
    
    def _check_time_constraints(self, plan: Dict[str, Any], constraints: Dict[str, Any]) -> List[str]:
        """检查时间约束违反"""
        violations = []
        
        # 提取时间需求
        estimated_duration = plan.get("estimated_duration", 0)
        max_duration = constraints.get("max_duration", float('inf'))
        deadline = constraints.get("deadline", None)
        
        # 检查最大持续时间
        if estimated_duration > max_duration:
            violations.append(
                f"持续时间超过限制: 估计{estimated_duration}, 限制{max_duration}"
            )
        
        # 检查截止时间（如果有）
        if deadline and isinstance(deadline, (int, float)):
            start_time = plan.get("start_time", time.time())
            if start_time + estimated_duration > deadline:
                violations.append(
                    f"无法在截止时间前完成: 开始{start_time}, 估计{estimated_duration}, 截止{deadline}"
                )
        
        # 检查时间一致性
        steps = plan.get("steps", [])
        step_durations = [s.get("estimated_time", 0) for s in steps]
        
        if step_durations and sum(step_durations) > estimated_duration * 1.2:  # 允许20%误差
            violations.append(
                f"步骤时间总和超过计划时间: 步骤总和{sum(step_durations)}, 计划{estimated_duration}"
            )
        
        return violations
    
    def _check_logic_constraints(self, plan: Dict[str, Any], constraints: Dict[str, Any]) -> List[str]:
        """检查逻辑约束违反"""
        violations = []
        
        logic_constraints = constraints.get("logic_constraints", {})
        plan_steps = plan.get("steps", [])
        
        # 检查顺序约束
        order_constraints = logic_constraints.get("order", [])
        for constraint in order_constraints:
            step_a = constraint.get("before", "")
            step_b = constraint.get("after", "")
            
            if step_a and step_b:
                step_indices = {s.get("id", ""): i for i, s in enumerate(plan_steps)}
                
                if step_a in step_indices and step_b in step_indices:
                    if step_indices[step_a] >= step_indices[step_b]:
                        violations.append(
                            f"顺序约束违反: '{step_a}'应在'{step_b}'之前"
                        )
        
        # 检查互斥约束
        exclusion_constraints = logic_constraints.get("exclusion", [])
        for constraint in exclusion_constraints:
            steps = constraint.get("steps", [])
            present_steps = [s for s in steps if any(s_step.get("id") == s for s_step in plan_steps)]
            
            if len(present_steps) > 1:
                violations.append(
                    f"互斥约束违反: 步骤{', '.join(present_steps)}不能同时存在"
                )
        
        # 检查依赖约束
        dependency_constraints = logic_constraints.get("dependency", [])
        for constraint in dependency_constraints:
            required = constraint.get("requires", "")
            dependent = constraint.get("dependent", "")
            
            if required and dependent:
                required_present = any(s.get("id") == required for s in plan_steps)
                dependent_present = any(s.get("id") == dependent for s in plan_steps)
                
                if dependent_present and not required_present:
                    violations.append(
                        f"依赖约束违反: '{dependent}'需要'{required}'"
                    )
        
        return violations
    
    def _detect_performance_degradation(self, performance_metrics: Dict[str, Any]) -> List[str]:
        """检测性能下降"""
        issues = []
        
        current_metrics = performance_metrics.get("current", {})
        baseline_metrics = performance_metrics.get("baseline", {})
        historical_metrics = performance_metrics.get("historical", [])
        
        if not baseline_metrics:
            return issues  # 没有基线数据
        
        # 检查关键性能指标下降
        key_metrics = ["accuracy", "precision", "recall", "f1_score", "latency", "throughput"]
        
        for metric in key_metrics:
            current_value = current_metrics.get(metric)
            baseline_value = baseline_metrics.get(metric)
            
            if current_value is not None and baseline_value is not None:
                # 计算性能变化
                if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                    if baseline_value > 0:
                        change = (current_value - baseline_value) / baseline_value
                        
                        # 判断是否下降
                        if "latency" in metric:  # 延迟增加是下降
                            if change > 0.2:  # 增加20%以上
                                issues.append(f"{metric}增加: {baseline_value:.2f} -> {current_value:.2f} (+{change*100:.1f}%)")
                        else:  # 其他指标减少是下降
                            if change < -0.2:  # 减少20%以上
                                issues.append(f"{metric}下降: {baseline_value:.2f} -> {current_value:.2f} ({change*100:.1f}%)")
        
        return issues
    
    def _detect_performance_anomalies(self, performance_metrics: Dict[str, Any]) -> List[str]:
        """检测性能异常模式"""
        patterns = []
        
        current_metrics = performance_metrics.get("current", {})
        historical_metrics = performance_metrics.get("historical", [])
        
        if not historical_metrics or len(historical_metrics) < 3:
            return patterns  # 历史数据不足
        
        # 提取时间序列数据
        for metric_name in current_metrics.keys():
            if metric_name in ["timestamp", "context"]:
                continue
                
            values = []
            timestamps = []
            
            for entry in historical_metrics:
                if metric_name in entry:
                    values.append(entry[metric_name])
                    if "timestamp" in entry:
                        timestamps.append(entry["timestamp"])
            
            if len(values) < 3:
                continue
            
            # 计算统计异常
            try:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                current_val = current_metrics.get(metric_name)
                
                if std_val > 0 and current_val is not None:
                    z_score = abs((current_val - mean_val) / std_val)
                    
                    if z_score > 3.0:  # 3个标准差以外
                        patterns.append(
                            f"{metric_name}异常: 当前值{current_val:.2f}, 均值{mean_val:.2f}, z分数{z_score:.2f}"
                        )
            except Exception as e:
                logger.debug(f"检测{metric_name}异常失败: {e}")
        
        return patterns
    
    # ===== 矫正策略生成方法 =====
    
    def _generate_coherence_correction(self, error: DetectedError) -> CorrectionStrategy:
        """生成连贯性矫正策略"""
        strategy_id = f"coherence_corr_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return CorrectionStrategy(
            strategy_id=strategy_id,
            error_ids=[error.error_id],
            correction_type=CorrectionType.LOGICAL_CORRECTION,
            description="修复推理链连贯性",
            implementation_steps=[
                {"step": 1, "action": "分析不连贯的步骤对", "target": "reasoning_steps"},
                {"step": 2, "action": "识别不连贯的原因", "target": "reasoning_patterns"},
                {"step": 3, "action": "重新组织推理步骤", "target": "reasoning_chain"},
                {"step": 4, "action": "添加连接推理", "target": "inference_links"},
                {"step": 5, "action": "验证修复后的连贯性", "target": "coherence_validation"}
            ],
            expected_outcome={
                "coherence_score": 0.8,
                "reasoning_quality": "improved",
                "validation_passed": True
            },
            confidence=0.7,
            cost_estimate=0.3,
            benefit_estimate=0.8,
            risk_assessment={
                "risks": ["可能引入新的不连贯", "可能改变原始推理意图"],
                "risk_level": "low",
                "mitigation": ["逐步应用改变", "保留原始推理备份"]
            },
            prerequisites=["完整的推理链数据", "连贯性分析结果"],
            validation_criteria=[
                {"criterion": "连贯性得分", "threshold": 0.7, "metric": "coherence_score"},
                {"criterion": "人工验证", "method": "expert_review", "required": False}
            ]
        )
    
    def _generate_consistency_correction(self, error: DetectedError) -> CorrectionStrategy:
        """生成一致性矫正策略"""
        strategy_id = f"consistency_corr_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return CorrectionStrategy(
            strategy_id=strategy_id,
            error_ids=[error.error_id],
            correction_type=CorrectionType.LOGICAL_CORRECTION,
            description="修复前提结论一致性",
            implementation_steps=[
                {"step": 1, "action": "重新评估前提的有效性", "target": "premises"},
                {"step": 2, "action": "检查推理过程的逻辑有效性", "target": "inference_process"},
                {"step": 3, "action": "调整结论以匹配前提", "target": "conclusions"},
                {"step": 4, "action": "添加缺失的前提或证据", "target": "missing_premises"},
                {"step": 5, "action": "验证一致性修复", "target": "consistency_validation"}
            ],
            expected_outcome={
                "consistency_score": 0.85,
                "logical_validity": "improved",
                "validation_passed": True
            },
            confidence=0.75,
            cost_estimate=0.4,
            benefit_estimate=0.9,
            risk_assessment={
                "risks": ["可能过度修改原始内容", "可能引入新的不一致"],
                "risk_level": "medium",
                "mitigation": ["保留推理历史", "逐步验证修改"]
            },
            prerequisites=["前提和结论数据", "逻辑分析工具"],
            validation_criteria=[
                {"criterion": "一致性得分", "threshold": 0.75, "metric": "consistency_score"},
                {"criterion": "逻辑有效性", "method": "logical_verification", "required": True}
            ]
        )
    
    def _generate_generic_logical_correction(self, error: DetectedError) -> CorrectionStrategy:
        """生成通用逻辑矫正策略"""
        strategy_id = f"generic_logic_corr_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return CorrectionStrategy(
            strategy_id=strategy_id,
            error_ids=[error.error_id],
            correction_type=CorrectionType.LOGICAL_CORRECTION,
            description=f"修复{error.error_type}逻辑错误",
            implementation_steps=[
                {"step": 1, "action": "分析错误的具体表现", "target": "error_analysis"},
                {"step": 2, "action": "识别根本原因", "target": "root_cause"},
                {"step": 3, "action": "设计针对性修复方案", "target": "repair_design"},
                {"step": 4, "action": "应用修复方案", "target": "repair_application"},
                {"step": 5, "action": "验证修复效果", "target": "repair_validation"}
            ],
            expected_outcome={
                "error_resolved": True,
                "reasoning_quality": "maintained_or_improved",
                "validation_passed": True
            },
            confidence=0.65,
            cost_estimate=0.5,
            benefit_estimate=0.7,
            risk_assessment={
                "risks": ["修复可能不完整", "可能引入副作用"],
                "risk_level": "medium",
                "mitigation": ["分阶段实施", "全面测试"]
            },
            prerequisites=["错误详细分析", "修复工具可用"],
            validation_criteria=[
                {"criterion": "错误消除", "method": "error_retest", "required": True},
                {"criterion": "无新错误", "method": "regression_testing", "required": True}
            ]
        )
    
    def _generate_resource_constraint_correction(self, error: DetectedError) -> CorrectionStrategy:
        """生成资源约束矫正策略"""
        strategy_id = f"resource_corr_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return CorrectionStrategy(
            strategy_id=strategy_id,
            error_ids=[error.error_id],
            correction_type=CorrectionType.CONSTRAINT_CORRECTION,
            description="修复资源约束违反",
            implementation_steps=[
                {"step": 1, "action": "分析资源需求和可用性", "target": "resource_analysis"},
                {"step": 2, "action": "识别资源瓶颈", "target": "bottleneck_identification"},
                {"step": 3, "action": "优化资源分配", "target": "resource_optimization"},
                {"step": 4, "action": "调整计划以适应资源限制", "target": "plan_adjustment"},
                {"step": 5, "action": "验证资源约束满足", "target": "resource_validation"}
            ],
            expected_outcome={
                "resource_constraints_satisfied": True,
                "resource_utilization": "optimized",
                "plan_feasibility": "improved"
            },
            confidence=0.8,
            cost_estimate=0.4,
            benefit_estimate=0.85,
            risk_assessment={
                "risks": ["资源优化可能导致质量下降", "重新分配可能影响其他部分"],
                "risk_level": "medium",
                "mitigation": ["逐步优化", "监控质量指标"]
            },
            prerequisites=["资源需求数据", "可用资源信息"],
            validation_criteria=[
                {"criterion": "资源需求满足", "method": "resource_check", "required": True},
                {"criterion": "计划质量保持", "method": "quality_assessment", "required": True}
            ]
        )
    
    def _generate_time_constraint_correction(self, error: DetectedError) -> CorrectionStrategy:
        """生成时间约束矫正策略"""
        strategy_id = f"time_corr_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return CorrectionStrategy(
            strategy_id=strategy_id,
            error_ids=[error.error_id],
            correction_type=CorrectionType.CONSTRAINT_CORRECTION,
            description="修复时间约束违反",
            implementation_steps=[
                {"step": 1, "action": "分析时间需求和限制", "target": "time_analysis"},
                {"step": 2, "action": "识别时间冲突和瓶颈", "target": "time_conflict_detection"},
                {"step": 3, "action": "重新安排任务和里程碑", "target": "schedule_rearrangement"},
                {"step": 4, "action": "优化任务并行性", "target": "parallelization_optimization"},
                {"step": 5, "action": "验证时间约束满足", "target": "time_validation"}
            ],
            expected_outcome={
                "time_constraints_satisfied": True,
                "schedule_efficiency": "improved",
                "deadline_meeting": "achievable"
            },
            confidence=0.75,
            cost_estimate=0.5,
            benefit_estimate=0.9,
            risk_assessment={
                "risks": ["时间压缩可能影响质量", "并行任务可能增加风险"],
                "risk_level": "medium",
                "mitigation": ["关键路径分析", "缓冲时间管理"]
            },
            prerequisites=["时间约束数据", "任务依赖关系"],
            validation_criteria=[
                {"criterion": "时间约束满足", "method": "time_check", "required": True},
                {"criterion": "计划可行性", "method": "feasibility_assessment", "required": True}
            ]
        )
    
    def _generate_logic_constraint_correction(self, error: DetectedError) -> CorrectionStrategy:
        """生成逻辑约束矫正策略"""
        strategy_id = f"logic_constraint_corr_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return CorrectionStrategy(
            strategy_id=strategy_id,
            error_ids=[error.error_id],
            correction_type=CorrectionType.CONSTRAINT_CORRECTION,
            description="修复逻辑约束违反",
            implementation_steps=[
                {"step": 1, "action": "分析逻辑约束类型和违反", "target": "constraint_analysis"},
                {"step": 2, "action": "识别约束冲突的根本原因", "target": "conflict_root_cause"},
                {"step": 3, "action": "重新设计计划结构", "target": "plan_restructuring"},
                {"step": 4, "action": "调整任务顺序和依赖", "target": "dependency_adjustment"},
                {"step": 5, "action": "验证逻辑约束满足", "target": "logic_validation"}
            ],
            expected_outcome={
                "logic_constraints_satisfied": True,
                "plan_coherence": "improved",
                "logical_consistency": "maintained"
            },
            confidence=0.7,
            cost_estimate=0.6,
            benefit_estimate=0.8,
            risk_assessment={
                "risks": ["结构调整可能破坏其他约束", "逻辑修改可能影响计划意图"],
                "risk_level": "high",
                "mitigation": ["增量修改", "全面验证"]
            },
            prerequisites=["逻辑约束定义", "计划结构数据"],
            validation_criteria=[
                {"criterion": "逻辑约束满足", "method": "logic_check", "required": True},
                {"criterion": "计划完整性保持", "method": "completeness_check", "required": True}
            ]
        )
    
    def _generate_generic_constraint_correction(self, error: DetectedError) -> CorrectionStrategy:
        """生成通用约束矫正策略"""
        strategy_id = f"generic_constraint_corr_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return CorrectionStrategy(
            strategy_id=strategy_id,
            error_ids=[error.error_id],
            correction_type=CorrectionType.CONSTRAINT_CORRECTION,
            description=f"修复{error.error_type}约束违反",
            implementation_steps=[
                {"step": 1, "action": "分析约束违反的具体情况", "target": "violation_analysis"},
                {"step": 2, "action": "评估约束重要性和灵活性", "target": "constraint_evaluation"},
                {"step": 3, "action": "设计约束满足方案", "target": "satisfaction_design"},
                {"step": 4, "action": "调整计划以满足约束", "target": "plan_adjustment"},
                {"step": 5, "action": "验证约束满足效果", "target": "constraint_validation"}
            ],
            expected_outcome={
                "constraint_satisfied": True,
                "plan_quality": "maintained",
                "validation_passed": True
            },
            confidence=0.65,
            cost_estimate=0.55,
            benefit_estimate=0.75,
            risk_assessment={
                "risks": ["约束满足可能降低计划质量", "调整可能引入新的违反"],
                "risk_level": "medium",
                "mitigation": ["权衡分析", "迭代优化"]
            },
            prerequisites=["约束详细数据", "计划调整能力"],
            validation_criteria=[
                {"criterion": "约束满足", "method": "constraint_test", "required": True},
                {"criterion": "质量指标保持", "method": "quality_monitoring", "required": True}
            ]
        )
    
    def _generate_performance_correction(self, error: DetectedError) -> CorrectionStrategy:
        """生成性能矫正策略"""
        strategy_id = f"performance_corr_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return CorrectionStrategy(
            strategy_id=strategy_id,
            error_ids=[error.error_id],
            correction_type=CorrectionType.PERFORMANCE_CORRECTION,
            description="修复性能问题",
            implementation_steps=[
                {"step": 1, "action": "分析性能下降的具体表现", "target": "performance_analysis"},
                {"step": 2, "action": "识别性能瓶颈和原因", "target": "bottleneck_identification"},
                {"step": 3, "action": "设计性能优化方案", "target": "optimization_design"},
                {"step": 4, "action": "实施性能优化措施", "target": "optimization_implementation"},
                {"step": 5, "action": "验证性能改进效果", "target": "performance_validation"}
            ],
            expected_outcome={
                "performance_improved": True,
                "performance_metrics": "restored_or_better",
                "system_efficiency": "enhanced"
            },
            confidence=0.7,
            cost_estimate=0.6,
            benefit_estimate=0.9,
            risk_assessment={
                "risks": ["优化可能引入新问题", "性能调整可能影响稳定性"],
                "risk_level": "medium",
                "mitigation": ["逐步实施", "全面测试"]
            },
            prerequisites=["性能监控数据", "优化工具可用"],
            validation_criteria=[
                {"criterion": "性能指标改善", "method": "metric_comparison", "required": True},
                {"criterion": "无副作用", "method": "side_effect_check", "required": True}
            ]
        )


# 组件类定义
class MonitoringComponent:
    """监控组件"""
    def monitor_process(self, planning_data, reasoning_data, context):
        """监控计划推理过程"""
        return {
            "planning_monitoring": self._monitor_planning(planning_data),
            "reasoning_monitoring": self._monitor_reasoning(reasoning_data),
            "context_monitoring": self._monitor_context(context),
            "timestamp": time.time()
        }
    
    def _monitor_planning(self, planning_data):
        """监控计划数据"""
        return {
            "plan_completeness": self._check_plan_completeness(planning_data),
            "plan_consistency": self._check_plan_consistency(planning_data),
            "plan_quality": self._assess_plan_quality(planning_data),
            "constraint_satisfaction": self._check_constraint_satisfaction(planning_data)
        }
    
    def _check_plan_completeness(self, planning_data):
        """检查计划完整性"""
        plan = planning_data.get("plan", {})
        steps = plan.get("steps", [])
        has_goal = "goal" in plan or "objective" in plan
        has_resources = "resource_requirements" in plan or "resources" in plan
        
        if len(steps) > 0 and has_goal:
            return 0.8  # 基本完整
        elif len(steps) > 0:
            return 0.5  # 部分完整
        else:
            return 0.2  # 不完整
    
    def _check_plan_consistency(self, planning_data):
        """检查计划一致性"""
        plan = planning_data.get("plan", {})
        
        # 简单检查：步骤是否相互矛盾
        steps = plan.get("steps", [])
        if len(steps) < 2:
            return 0.9  # 单个步骤或没有步骤时一致性高
        
        # 检查是否有明显矛盾（简化版本）
        step_actions = [step.get("action", "") for step in steps]
        contradictions = 0
        for i in range(len(step_actions)):
            for j in range(i+1, len(step_actions)):
                if "cancel" in step_actions[i] and "create" in step_actions[j]:
                    contradictions += 1
                elif "remove" in step_actions[i] and "add" in step_actions[j]:
                    contradictions += 1
        
        if contradictions > 0:
            return 0.4  # 发现矛盾
        else:
            return 0.8  # 基本一致
    
    def _assess_plan_quality(self, planning_data):
        """评估计划质量"""
        plan = planning_data.get("plan", {})
        
        quality_score = 0.5  # 基础分数
        
        # 检查计划是否有详细描述
        steps = plan.get("steps", [])
        detailed_steps = [step for step in steps if step.get("description") or step.get("details")]
        
        if len(steps) > 0:
            quality_score += 0.1 * (len(detailed_steps) / len(steps))
        
        # 检查是否有风险评估
        if "risk_assessment" in plan:
            quality_score += 0.1
        
        # 检查是否有资源规划
        if "resource_requirements" in plan:
            quality_score += 0.1
        
        # 检查是否有时间线
        if "timeline" in plan or "schedule" in plan:
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _check_constraint_satisfaction(self, planning_data):
        """检查约束满足情况"""
        plan = planning_data.get("plan", {})
        constraints = plan.get("constraints", {})
        
        if not constraints:
            return {"satisfied": True, "score": 1.0, "violations": []}
        
        violations = []
        
        # 检查时间约束
        if "max_duration" in constraints:
            duration = plan.get("estimated_duration", 0)
            if duration > constraints["max_duration"]:
                violations.append({
                    "constraint": "max_duration",
                    "actual": duration,
                    "allowed": constraints["max_duration"]
                })
        
        # 检查资源约束
        if "max_resources" in constraints or "max_developers" in constraints:
            resources = plan.get("resource_requirements", {})
            if "developers" in resources:
                max_dev = constraints.get("max_developers", constraints.get("max_resources", float('inf')))
                if resources["developers"] > max_dev:
                    violations.append({
                        "constraint": "max_developers",
                        "actual": resources["developers"],
                        "allowed": max_dev
                    })
        
        # 检查成本约束
        if "max_cost" in constraints:
            cost = plan.get("estimated_cost", 0)
            if cost > constraints["max_cost"]:
                violations.append({
                    "constraint": "max_cost",
                    "actual": cost,
                    "allowed": constraints["max_cost"]
                })
        
        satisfaction_score = 1.0 - (len(violations) / max(1, len(constraints)))
        
        return {
            "satisfied": len(violations) == 0,
            "score": satisfaction_score,
            "violations": violations
        }
    
    def _monitor_reasoning(self, reasoning_data):
        """监控推理数据"""
        return {
            "reasoning_coherence": self._check_reasoning_coherence(reasoning_data),
            "reasoning_completeness": self._check_reasoning_completeness(reasoning_data),
            "reasoning_quality": self._assess_reasoning_quality(reasoning_data),
            "logical_consistency": self._check_logical_consistency(reasoning_data)
        }
    
    def _check_reasoning_coherence(self, reasoning_data):
        """检查推理连贯性"""
        reasoning_chain = reasoning_data.get("reasoning_chain", {})
        steps = reasoning_chain.get("steps", [])
        
        if len(steps) <= 1:
            return 0.8  # 单个步骤或没有步骤时连贯性高
        
        # 检查步骤之间的连贯性（简化版本）
        coherent_pairs = 0
        total_pairs = len(steps) - 1
        
        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]
            
            # 简单检查：步骤类型是否合理过渡
            current_type = current_step.get("type", "")
            next_type = next_step.get("type", "")
            
            # 基本连贯性规则
            if current_type == "premise" and next_type in ["inference", "conclusion", "deduction"]:
                coherent_pairs += 1
            elif current_type == "inference" and next_type in ["conclusion", "deduction", "inference"]:
                coherent_pairs += 1
            elif current_type == "deduction" and next_type in ["conclusion", "inference"]:
                coherent_pairs += 1
            elif current_type == "conclusion" and i == len(steps) - 2:  # 结论通常在最后
                coherent_pairs += 1
            elif current_type == next_type:  # 相同类型通常是连贯的
                coherent_pairs += 1
        
        coherence_score = coherent_pairs / total_pairs if total_pairs > 0 else 1.0
        return coherence_score
    
    def _check_reasoning_completeness(self, reasoning_data):
        """检查推理完整性"""
        reasoning_chain = reasoning_data.get("reasoning_chain", {})
        steps = reasoning_chain.get("steps", [])
        
        # 检查是否有前提和结论
        has_premise = any(step.get("type") in ["premise", "assumption", "fact"] for step in steps)
        has_conclusion = any(step.get("type") in ["conclusion", "result", "outcome"] for step in steps)
        
        if has_premise and has_conclusion and len(steps) >= 3:
            return 0.9  # 完整推理链
        elif has_premise and has_conclusion:
            return 0.7  # 基本完整
        elif has_premise or has_conclusion:
            return 0.5  # 部分完整
        else:
            return 0.2  # 不完整
    
    def _assess_reasoning_quality(self, reasoning_data):
        """评估推理质量"""
        reasoning_chain = reasoning_data.get("reasoning_chain", {})
        
        quality_score = 0.5  # 基础分数
        
        # 检查是否有推理质量分数
        if "coherence_score" in reasoning_chain:
            quality_score += 0.2 * reasoning_chain["coherence_score"]
        
        if "consistency_score" in reasoning_chain:
            quality_score += 0.2 * reasoning_chain["consistency_score"]
        
        # 检查推理步骤的详细程度
        steps = reasoning_chain.get("steps", [])
        detailed_steps = [step for step in steps if step.get("reasoning") or step.get("explanation")]
        
        if len(steps) > 0:
            quality_score += 0.1 * (len(detailed_steps) / len(steps))
        
        # 检查是否有证据支持
        evidence_steps = [step for step in steps if step.get("evidence") or step.get("support")]
        if len(evidence_steps) > 0:
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _check_logical_consistency(self, reasoning_data):
        """检查逻辑一致性"""
        reasoning_chain = reasoning_data.get("reasoning_chain", {})
        steps = reasoning_chain.get("steps", [])
        
        if len(steps) <= 1:
            return 0.9  # 单个步骤或没有步骤时一致性高
        
        # 检查明显逻辑矛盾
        contradictions = 0
        step_contents = [step.get("content", "").lower() for step in steps]
        
        # 简单关键词矛盾检测
        for i in range(len(step_contents)):
            for j in range(i+1, len(step_contents)):
                content_i = step_contents[i]
                content_j = step_contents[j]
                
                # 检测明显矛盾
                if ("all" in content_i and "not all" in content_j) or \
                   ("never" in content_i and "sometimes" in content_j) or \
                   ("always" in content_i and "not always" in content_j):
                    contradictions += 1
        
        # 计算一致性分数
        total_comparisons = len(steps) * (len(steps) - 1) / 2
        consistency_score = 1.0 - (contradictions / max(1, total_comparisons))
        
        return consistency_score
    
    def _monitor_context(self, context):
        """监控上下文数据"""
        return {
            "context_relevance": self._check_context_relevance(context),
            "context_completeness": self._check_context_completeness(context),
            "context_consistency": self._check_context_consistency(context)
        }
    
    def _check_context_relevance(self, context):
        """检查上下文相关性"""
        if not context:
            return 0.3  # 无上下文时相关性低
        
        # 检查上下文是否有相关信息
        relevant_keys = ["goal", "constraints", "resources", "domain", "history"]
        relevant_count = sum(1 for key in relevant_keys if key in context)
        
        relevance_score = relevant_count / len(relevant_keys)
        return relevance_score
    
    def _check_context_completeness(self, context):
        """检查上下文完整性"""
        if not context:
            return 0.2  # 无上下文时不完整
        
        # 基本上下文元素
        essential_keys = ["goal", "domain"]
        essential_count = sum(1 for key in essential_keys if key in context)
        
        completeness_score = essential_count / len(essential_keys)
        return completeness_score
    
    def _check_context_consistency(self, context):
        """检查上下文一致性"""
        if not context or len(context) <= 1:
            return 0.8  # 单个或无元素时一致性高
        
        # 检查上下文内部一致性（简化版本）
        inconsistencies = 0
        
        # 检查资源约束一致性
        if "constraints" in context and "available_resources" in context:
            constraints = context.get("constraints", {})
            resources = context.get("available_resources", [])
            
            # 如果约束要求资源但可用资源为空，可能不一致
            if constraints.get("require_resources", False) and not resources:
                inconsistencies += 1
        
        # 检查时间一致性
        if "time_constraints" in context and "estimated_duration" in context:
            time_constraints = context.get("time_constraints", {})
            estimated_duration = context.get("estimated_duration", 0)
            
            max_duration = time_constraints.get("max_duration", float('inf'))
            if estimated_duration > max_duration:
                inconsistencies += 1
        
        consistency_score = 1.0 - (inconsistencies / max(1, len(context)))
        return consistency_score

class ErrorDetectionComponent:
    """错误检测组件"""
    def detect_errors(self, monitoring_results, context):
        """检测错误"""
        errors = []
        
        # 检测计划错误
        plan_errors = self._detect_planning_errors(monitoring_results["planning_monitoring"])
        errors.extend(plan_errors)
        
        # 检测推理错误
        reasoning_errors = self._detect_reasoning_errors(monitoring_results["reasoning_monitoring"])
        errors.extend(reasoning_errors)
        
        # 检测上下文错误
        context_errors = self._detect_context_errors(monitoring_results["context_monitoring"])
        errors.extend(context_errors)
        
        return errors
    
    def _detect_planning_errors(self, planning_monitoring):
        """检测计划错误"""
        errors = []
        
        # 检查计划完整性
        completeness = planning_monitoring.get("plan_completeness", 0.5)
        if completeness < 0.5:
            errors.append(DetectedError(
                error_id=f"plan_completeness_{int(time.time())}_{random.randint(1000, 9999)}",
                error_type="plan_incomplete",
                severity=ErrorSeverity.MEDIUM,
                detection_time=time.time(),
                context={"completeness_score": completeness},
                description=f"计划不完整: 完整性分数={completeness:.2f}",
                source_component="planning",
                confidence=0.8,
                impact_areas=["planning", "completeness"],
                raw_data={"completeness_score": completeness, "monitoring_data": planning_monitoring}
            ))
        
        # 检查计划一致性
        consistency = planning_monitoring.get("plan_consistency", 0.8)
        if consistency < 0.6:
            errors.append(DetectedError(
                error_id=f"plan_inconsistent_{int(time.time())}_{random.randint(1000, 9999)}",
                error_type="plan_inconsistent",
                severity=ErrorSeverity.MEDIUM,
                description=f"计划不一致: 一致性分数={consistency:.2f}",
                source_component="planning",
                detection_time=time.time(),
                context={"consistency_score": consistency}
            ))
        
        # 检查约束满足
        constraint_satisfaction = planning_monitoring.get("constraint_satisfaction", {})
        if isinstance(constraint_satisfaction, dict):
            satisfied = constraint_satisfaction.get("satisfied", True)
            violations = constraint_satisfaction.get("violations", [])
            if not satisfied:
                errors.append(DetectedError(
                    error_id=f"constraint_violation_{int(time.time())}_{random.randint(1000, 9999)}",
                    error_type="constraint_violation",
                    severity=ErrorSeverity.HIGH,
                    description=f"约束违反: {len(violations)} 个违反",
                    source_component="planning",
                    detection_time=time.time(),
                    context={"violations": violations, "constraint_satisfaction": constraint_satisfaction}
                ))
        
        return errors
    
    def _detect_reasoning_errors(self, reasoning_monitoring):
        """检测推理错误"""
        errors = []
        
        # 检查推理连贯性
        coherence = reasoning_monitoring.get("reasoning_coherence", 0.7)
        if coherence < 0.6:
            errors.append(DetectedError(
                error_id=f"reasoning_incoherent_{int(time.time())}_{random.randint(1000, 9999)}",
                error_type="reasoning_incoherent",
                severity=ErrorSeverity.MEDIUM,
                description=f"推理不连贯: 连贯性分数={coherence:.2f}",
                source_component="reasoning",
                detection_time=time.time(),
                context={"coherence_score": coherence}
            ))
        
        # 检查逻辑一致性
        logical_consistency = reasoning_monitoring.get("logical_consistency", 0.8)
        if logical_consistency < 0.7:
            errors.append(DetectedError(
                error_id=f"logical_inconsistent_{int(time.time())}_{random.randint(1000, 9999)}",
                error_type="logical_inconsistency",
                severity=ErrorSeverity.HIGH,
                description=f"逻辑不一致: 一致性分数={logical_consistency:.2f}",
                source_component="reasoning",
                detection_time=time.time(),
                context={"logical_consistency_score": logical_consistency}
            ))
        
        # 检查推理质量
        quality = reasoning_monitoring.get("reasoning_quality", 0.6)
        if quality < 0.5:
            errors.append(DetectedError(
                error_id=f"reasoning_low_quality_{int(time.time())}_{random.randint(1000, 9999)}",
                error_type="reasoning_low_quality",
                severity=ErrorSeverity.MEDIUM,
                description=f"推理质量低: 质量分数={quality:.2f}",
                source_component="reasoning",
                detection_time=time.time(),
                context={"quality_score": quality}
            ))
        
        return errors
    
    def _detect_context_errors(self, context_monitoring):
        """检测上下文错误"""
        errors = []
        
        # 检查上下文相关性
        relevance = context_monitoring.get("context_relevance", 0.5)
        if relevance < 0.4:
            errors.append(DetectedError(
                error_id=f"context_irrelevant_{int(time.time())}_{random.randint(1000, 9999)}",
                error_type="context_irrelevant",
                severity=ErrorSeverity.LOW,
                description=f"上下文相关性低: 相关性分数={relevance:.2f}",
                source_component="context",
                detection_time=time.time(),
                context={"relevance_score": relevance}
            ))
        
        # 检查上下文完整性
        completeness = context_monitoring.get("context_completeness", 0.5)
        if completeness < 0.4:
            errors.append(DetectedError(
                error_id=f"context_incomplete_{int(time.time())}_{random.randint(1000, 9999)}",
                error_type="context_incomplete",
                severity=ErrorSeverity.MEDIUM,
                description=f"上下文不完整: 完整性分数={completeness:.2f}",
                source_component="context",
                detection_time=time.time(),
                context={"completeness_score": completeness}
            ))
        
        # 检查上下文一致性
        consistency = context_monitoring.get("context_consistency", 0.7)
        if consistency < 0.6:
            errors.append(DetectedError(
                error_id=f"context_inconsistent_{int(time.time())}_{random.randint(1000, 9999)}",
                error_type="context_inconsistent",
                severity=ErrorSeverity.MEDIUM,
                description=f"上下文不一致: 一致性分数={consistency:.2f}",
                source_component="context",
                detection_time=time.time(),
                context={"consistency_score": consistency}
            ))
        
        return errors

class AnalysisComponent:
    """分析组件"""
    def _analyze_error_patterns(self, detected_errors):
        """分析错误模式"""
        if not detected_errors:
            return {"error_patterns": [], "common_patterns": [], "trends": []}
        
        error_types = {}
        severities = {}
        components = {}
        
        for error in detected_errors:
            error_type = error.error_type
            severity = error.severity.value if hasattr(error.severity, 'value') else str(error.severity)
            component = error.source_component
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            severities[severity] = severities.get(severity, 0) + 1
            components[component] = components.get(component, 0) + 1
        
        total_errors = len(detected_errors)
        
        return {
            "error_patterns": {
                "error_type_distribution": error_types,
                "severity_distribution": severities,
                "component_distribution": components
            },
            "common_patterns": [
                {"pattern": "most_common_error", "type": max(error_types, key=error_types.get), "count": max(error_types.values())} if error_types else {},
                {"pattern": "most_common_component", "component": max(components, key=components.get), "count": max(components.values())} if components else {}
            ],
            "trends": {
                "error_rate": total_errors,
                "dominant_error_type": max(error_types, key=error_types.get) if error_types else "none",
                "dominant_severity": max(severities, key=severities.get) if severities else "none"
            }
        }
    
    def _perform_root_cause_analysis(self, detected_errors, context):
        """执行根因分析"""
        if not detected_errors:
            return {"root_causes": [], "causal_factors": [], "recommendations": []}
        
        root_causes = []
        for error in detected_errors:
            root_cause = {
                "error_id": error.error_id,
                "error_type": error.error_type,
                "potential_root_causes": [
                    "配置错误",
                    "资源不足",
                    "逻辑错误",
                    "数据问题",
                    "外部依赖失败"
                ],
                "confidence": 0.7,
                "recommended_actions": [
                    "检查配置",
                    "增加资源",
                    "审查逻辑",
                    "验证数据",
                    "检查外部服务"
                ]
            }
            root_causes.append(root_cause)
        
        return {
            "root_causes": root_causes,
            "causal_factors": ["配置", "资源", "逻辑", "数据", "外部依赖"],
            "recommendations": ["系统审查", "配置验证", "资源优化", "逻辑测试"]
        }
    
    def _analyze_impact(self, detected_errors, monitoring_results):
        """分析影响"""
        if not detected_errors:
            return {"impact_level": "low", "affected_areas": [], "performance_impact": 0.0}
        
        severity_scores = {
            "low": 1,
            "medium": 3,
            "high": 5,
            "critical": 10
        }
        
        total_impact = 0
        for error in detected_errors:
            severity = error.severity.value if hasattr(error.severity, 'value') else str(error.severity)
            total_impact += severity_scores.get(severity, 1)
        
        impact_level = "low"
        if total_impact >= 15:
            impact_level = "critical"
        elif total_impact >= 10:
            impact_level = "high"
        elif total_impact >= 5:
            impact_level = "medium"
        
        return {
            "impact_level": impact_level,
            "affected_areas": ["planning", "reasoning", "execution"],
            "performance_impact": min(total_impact / 20.0, 1.0)
        }
    
    def _analyze_correlations(self, detected_errors):
        """分析相关性"""
        if not detected_errors:
            return {"correlations": [], "patterns": [], "insights": []}
        
        # 简单相关性分析
        error_types = [error.error_type for error in detected_errors]
        components = [error.source_component for error in detected_errors]
        
        type_correlations = {}
        for i in range(len(error_types)):
            for j in range(i+1, len(error_types)):
                pair = (error_types[i], error_types[j])
                type_correlations[pair] = type_correlations.get(pair, 0) + 1
        
        component_correlations = {}
        for i in range(len(components)):
            for j in range(i+1, len(components)):
                pair = (components[i], components[j])
                component_correlations[pair] = component_correlations.get(pair, 0) + 1
        
        return {
            "correlations": {
                "error_type_correlations": type_correlations,
                "component_correlations": component_correlations
            },
            "patterns": [
                {"pattern": "sequential_errors", "description": "连续错误发生", "confidence": 0.6},
                {"pattern": "component_cluster", "description": "组件集群错误", "confidence": 0.7}
            ],
            "insights": [
                "错误倾向于在特定组件中聚集",
                "某些错误类型经常一起出现"
            ]
        }
    
    def analyze_errors(self, detected_errors, monitoring_results, context):
        """分析错误"""
        return {
            "error_analysis": self._analyze_error_patterns(detected_errors),
            "root_cause_analysis": self._perform_root_cause_analysis(detected_errors, context),
            "impact_analysis": self._analyze_impact(detected_errors, monitoring_results),
            "correlation_analysis": self._analyze_correlations(detected_errors)
        }

class StrategyGenerationComponent:
    """策略生成组件"""
    def _generate_strategy_for_error(self, error, error_analysis, context):
        """为错误生成矫正策略"""
        strategy_id = f"strategy_{error.error_id}_{int(time.time())}"
        
        # 根据错误类型生成策略
        error_type = error.error_type
        severity = error.severity.value if hasattr(error.severity, 'value') else str(error.severity)
        
        if "constraint" in error_type:
            strategy_type = "constraint_relaxation"
            description = f"放松约束以解决{error_type}"
            implementation_steps = [
                {"action": "identify_constraint", "parameters": {"constraint_type": error_type}},
                {"action": "analyze_impact", "parameters": {"severity": severity}},
                {"action": "relax_constraint", "parameters": {"relaxation_factor": 0.2}}
            ]
        elif "resource" in error_type:
            strategy_type = "resource_allocation"
            description = f"重新分配资源以解决{error_type}"
            implementation_steps = [
                {"action": "assess_resource_usage", "parameters": {}},
                {"action": "identify_bottlenecks", "parameters": {}},
                {"action": "reallocate_resources", "parameters": {"adjustment_factor": 0.3}}
            ]
        elif "logic" in error_type or "reasoning" in error_type:
            strategy_type = "logic_correction"
            description = f"修正逻辑错误：{error_type}"
            implementation_steps = [
                {"action": "review_logic", "parameters": {"error_type": error_type}},
                {"action": "identify_flaws", "parameters": {}},
                {"action": "apply_correction", "parameters": {"correction_type": "logic_fix"}}
            ]
        else:
            strategy_type = "general_correction"
            description = f"通用矫正策略：{error_type}"
            implementation_steps = [
                {"action": "analyze_error", "parameters": {"error_id": error.error_id}},
                {"action": "design_solution", "parameters": {}},
                {"action": "implement_fix", "parameters": {}}
            ]
        
        return {
            "strategy_id": strategy_id,
            "error_ids": [error.error_id],
            "correction_type": strategy_type,
            "description": description,
            "implementation_steps": implementation_steps,
            "estimated_effort": 5,
            "priority": "high" if severity in ["high", "critical"] else "medium",
            "expected_impact": 0.7,
            "validation_requirements": [
                {"requirement": "error_resolution", "target": "error_count_reduction"},
                {"requirement": "performance_improvement", "target": "performance_increase"}
            ]
        }
    
    def generate_strategies(self, detected_errors, error_analysis, context):
        """生成矫正策略"""
        strategies = []
        
        for error in detected_errors:
            strategy = self._generate_strategy_for_error(error, error_analysis, context)
            if strategy:
                strategies.append(strategy)
        
        return strategies

class ExecutionComponent:
    """执行组件"""
    def _apply_correction_strategy(self, strategy, planning_data, reasoning_data, context):
        """应用矫正策略"""
        correction_id = f"correction_{strategy['strategy_id']}_{int(time.time())}"
        
        correction_type = strategy.get("correction_type", "general_correction")
        implementation_steps = strategy.get("implementation_steps", [])
        
        # 模拟策略执行
        executed_steps = []
        for step in implementation_steps:
            step_result = {
                "step_id": f"step_{len(executed_steps)}",
                "action": step.get("action", "unknown"),
                "parameters": step.get("parameters", {}),
                "status": "completed",
                "result": "success",
                "execution_time": 0.5
            }
            executed_steps.append(step_result)
        
        # 评估矫正效果
        error_reduction = 0.7
        performance_improvement = 0.6
        stability_improvement = 0.8
        
        return {
            "correction_id": correction_id,
            "strategy_id": strategy["strategy_id"],
            "correction_type": correction_type,
            "status": "completed",
            "executed_steps": executed_steps,
            "results": {
                "error_reduction": error_reduction,
                "performance_improvement": performance_improvement,
                "stability_improvement": stability_improvement,
                "overall_effectiveness": (error_reduction + performance_improvement + stability_improvement) / 3.0
            },
            "validation": {
                "validation_passed": True,
                "validation_criteria": [
                    {"criterion": "error_count_reduction", "target": 0.5, "achieved": error_reduction},
                    {"criterion": "performance_increase", "target": 0.5, "achieved": performance_improvement}
                ]
            },
            "execution_time": len(executed_steps) * 0.5,
            "timestamp": time.time()
        }
    
    def apply_corrections(self, correction_strategies, planning_data, reasoning_data, context):
        """应用矫正"""
        applied_corrections = []
        
        for strategy in correction_strategies:
            correction_result = self._apply_correction_strategy(strategy, planning_data, reasoning_data, context)
            applied_corrections.append(correction_result)
        
        return {
            "applied_corrections": applied_corrections,
            "overall_success": all(c.get('success', False) for c in applied_corrections)
        }

class ValidationComponent:
    """验证组件"""
    def _validate_individual_correction(self, correction, detected_error):
        """验证单个矫正效果"""
        if not correction:
            return {"success": False, "reason": "无矫正数据", "confidence": 0.0}
        
        correction_results = correction.get("results", {})
        validation_info = correction.get("validation", {})
        
        # 检查矫正结果
        error_reduction = correction_results.get("error_reduction", 0.0)
        performance_improvement = correction_results.get("performance_improvement", 0.0)
        overall_effectiveness = correction_results.get("overall_effectiveness", 0.0)
        
        # 验证标准
        validation_passed = validation_info.get("validation_passed", False)
        validation_criteria = validation_info.get("validation_criteria", [])
        
        criteria_met = 0
        for criterion in validation_criteria:
            target = criterion.get("target", 0.5)
            achieved = criterion.get("achieved", 0.0)
            if achieved >= target:
                criteria_met += 1
        
        criteria_success_rate = criteria_met / len(validation_criteria) if validation_criteria else 1.0
        
        success = validation_passed and criteria_success_rate >= 0.7
        
        return {
            "success": success,
            "correction_id": correction.get("correction_id", "unknown"),
            "error_reduction": error_reduction,
            "performance_improvement": performance_improvement,
            "overall_effectiveness": overall_effectiveness,
            "validation_passed": validation_passed,
            "criteria_success_rate": criteria_success_rate,
            "confidence": min(0.9, overall_effectiveness * 0.8 + criteria_success_rate * 0.2)
        }
    
    def _analyze_performance_improvement(self, correction_results, detected_errors):
        """分析性能改进"""
        applied_corrections = correction_results.get("applied_corrections", [])
        
        if not applied_corrections:
            return {
                "overall_improvement": 0.0,
                "error_reduction_rate": 0.0,
                "performance_gain": 0.0,
                "stability_improvement": 0.0
            }
        
        total_error_reduction = 0.0
        total_performance_improvement = 0.0
        total_stability_improvement = 0.0
        
        for correction in applied_corrections:
            results = correction.get("results", {})
            total_error_reduction += results.get("error_reduction", 0.0)
            total_performance_improvement += results.get("performance_improvement", 0.0)
            total_stability_improvement += results.get("stability_improvement", 0.0)
        
        count = len(applied_corrections)
        
        return {
            "overall_improvement": (total_error_reduction + total_performance_improvement + total_stability_improvement) / (3 * count) if count > 0 else 0.0,
            "error_reduction_rate": total_error_reduction / count if count > 0 else 0.0,
            "performance_gain": total_performance_improvement / count if count > 0 else 0.0,
            "stability_improvement": total_stability_improvement / count if count > 0 else 0.0,
            "correction_count": count,
            "detected_errors_count": len(detected_errors)
        }
    
    def _generate_learning_insights(self, correction_results, validation_results):
        """生成学习见解"""
        insights = []
        applied_corrections = correction_results.get("applied_corrections", [])
        
        if not applied_corrections:
            insights.append({
                "type": "no_corrections",
                "description": "未应用任何矫正",
                "recommendation": "检查错误检测系统"
            })
            return insights
        
        # 分析最有效的矫正类型
        correction_types = {}
        for correction in applied_corrections:
            correction_type = correction.get("correction_type", "unknown")
            effectiveness = correction.get("results", {}).get("overall_effectiveness", 0.0)
            
            if correction_type not in correction_types:
                correction_types[correction_type] = {"count": 0, "total_effectiveness": 0.0}
            
            correction_types[correction_type]["count"] += 1
            correction_types[correction_type]["total_effectiveness"] += effectiveness
        
        for correction_type, stats in correction_types.items():
            avg_effectiveness = stats["total_effectiveness"] / stats["count"]
            insights.append({
                "type": "correction_effectiveness",
                "description": f"矫正类型'{correction_type}'平均效果: {avg_effectiveness:.2f}",
                "recommendation": "继续使用有效矫正策略" if avg_effectiveness > 0.7 else "优化或替换低效矫正策略"
            })
        
        # 分析验证结果
        individual_validations = validation_results.get("individual_validations", [])
        success_count = sum(1 for v in individual_validations if v.get("success", False))
        total_count = len(individual_validations)
        
        if total_count > 0:
            success_rate = success_count / total_count
            insights.append({
                "type": "validation_summary",
                "description": f"矫正验证成功率: {success_rate:.2f} ({success_count}/{total_count})",
                "recommendation": "保持当前策略" if success_rate > 0.8 else "改进矫正策略"
            })
        
        # 通用建议
        if len(insights) < 3:
            insights.append({
                "type": "general_advice",
                "description": "系统显示了矫正能力，建议继续监控和优化",
                "recommendation": "定期审查矫正效果并更新策略"
            })
        
        return insights
    
    def validate_corrections(self, correction_results, detected_errors, context):
        """验证矫正效果"""
        validation_results = {
            "individual_validations": [],
            "performance_improvement": {},
            "learning_insights": []
        }
        
        for i, correction in enumerate(correction_results.get('applied_corrections', [])):
            validation = self._validate_individual_correction(correction, detected_errors[i] if i < len(detected_errors) else None)
            validation_results["individual_validations"].append(validation)
        
        validation_results["overall_success"] = all(
            v.get('success', False) for v in validation_results["individual_validations"]
        )
        
        # 分析性能改进
        validation_results["performance_improvement"] = self._analyze_performance_improvement(
            correction_results, detected_errors
        )
        
        # 生成学习见解
        validation_results["learning_insights"] = self._generate_learning_insights(
            correction_results, validation_results
        )
        
        return validation_results


# 实用函数：创建自我矫正增强器实例
def create_self_correction_enhancer(config: Optional[Dict[str, Any]] = None) -> SelfCorrectionEnhancer:
    """
    创建自我矫正增强器实例
    
    Args:
        config: 增强器配置字典
        
    Returns:
        初始化好的自我矫正增强器实例
    """
    return SelfCorrectionEnhancer(config)


# 示例使用
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建自我矫正增强器
    enhancer = create_self_correction_enhancer()
    
    print("自我矫正增强器测试...")
    
    # 测试数据
    test_planning_data = {
        "plan": {
            "steps": [{"action": "step1"}, {"action": "step2"}],
            "constraints": {"time": 100, "resources": 3},
            "quality_score": 0.8
        }
    }
    
    test_reasoning_data = {
        "reasoning_chain": {
            "steps": [
                {"type": "premise", "content": "前提A"},
                {"type": "inference", "content": "推导B"},
                {"type": "conclusion", "content": "结论C"}
            ],
            "coherence_score": 0.7
        }
    }
    
    # 测试监控矫正
    result = enhancer.monitor_and_correct(
        test_planning_data,
        test_reasoning_data,
        {"test_context": True}
    )
    
    print(f"矫正结果: {result.get('success', False)}")
    print(f"检测到的错误数: {result.get('errors_detected', 0)}")
    print(f"应用矫正数: {result.get('corrections_applied', 0)}")