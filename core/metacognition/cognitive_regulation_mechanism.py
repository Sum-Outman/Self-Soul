#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知调节机制模块 - 实现统一的认知过程调节和优化

核心功能:
1. 多层级认知状态监控：实时监控注意、记忆、推理、决策等认知状态
2. 动态参数调整：基于监控结果动态调整认知过程的参数
3. 资源分配优化：优化认知资源（注意力、工作记忆等）的分配
4. 策略切换管理：根据任务需求切换认知策略
5. 反馈闭环控制：基于执行效果调整调节策略

调节维度:
1. 注意力调节: 聚焦、分配、切换、恢复
2. 记忆调节: 编码、存储、检索、遗忘
3. 推理调节: 策略选择、深度控制、广度控制
4. 决策调节: 风险偏好、时间偏好、信息处理
5. 情绪调节: 情感强度、情感方向、情感影响

设计原则:
- 实时性: 低延迟的调节响应
- 自适应性: 基于反馈的自适应调节
- 个性化: 考虑个体认知特点和偏好
- 可解释性: 调节过程和结果可解释
- 安全性: 防止过调节和负面效应

系统架构:
监控层 → 分析层 → 决策层 → 执行层 → 评估层 → 反馈层

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import time
import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import random

# 导入相关模块
from .thinking_process_tracker import ThinkingProcessTracker
from .cognitive_bias_detector import CognitiveBiasDetector
from .reasoning_strategy_evaluator import ReasoningStrategyEvaluator

# 配置日志
logger = logging.getLogger(__name__)


class RegulationGoal(Enum):
    """调节目标"""
    IMPROVE_PERFORMANCE = "improve_performance"
    REDUCE_COGNITIVE_LOAD = "reduce_cognitive_load"
    ENHANCE_EFFICIENCY = "enhance_efficiency"
    INCREASE_ACCURACY = "increase_accuracy"
    BOOST_CREATIVITY = "boost_creativity"
    PREVENT_BIAS = "prevent_bias"
    MANAGE_EMOTION = "manage_emotion"
    OPTIMIZE_RESOURCE = "optimize_resource"


class RegulationType(Enum):
    """调节类型"""
    ATTENTION_REGULATION = "attention_regulation"
    MEMORY_REGULATION = "memory_regulation"
    REASONING_REGULATION = "reasoning_regulation"
    DECISION_REGULATION = "decision_regulation"
    EMOTION_REGULATION = "emotion_regulation"
    STRATEGY_REGULATION = "strategy_regulation"
    RESOURCE_REGULATION = "resource_regulation"


class RegulationIntensity(Enum):
    """调节强度"""
    MINIMAL = "minimal"        # 微调
    MODERATE = "moderate"      # 适中调整
    SUBSTANTIAL = "substantial"  # 显著调整
    DRAMATIC = "dramatic"      # 剧烈调整


@dataclass
class CognitiveState:
    """认知状态"""
    timestamp: datetime
    # 注意力状态
    attention_focus: float  # 0-1
    attention_distribution: Dict[str, float]  # 各任务注意力分布
    attention_stability: float  # 0-1
    
    # 记忆状态
    working_memory_load: float  # 0-1
    long_term_memory_access: float  # 0-1
    memory_retrieval_speed: float  # 0-1
    
    # 推理状态
    reasoning_depth: float  # 0-1
    reasoning_speed: float  # 0-1
    reasoning_accuracy: float  # 0-1
    
    # 决策状态
    decision_confidence: float  # 0-1
    decision_speed: float  # 0-1
    risk_tolerance: float  # 0-1
    
    # 情绪状态
    emotional_valence: float  # -1到1
    emotional_arousal: float  # 0-1
    emotion_cognition_influence: float  # 0-1
    
    # 资源状态
    cognitive_resource_available: float  # 0-1
    resource_allocation_efficiency: float  # 0-1
    
    # 元认知状态
    self_awareness: float  # 0-1
    metacognitive_control: float  # 0-1
    
    # 性能指标
    performance_score: float  # 0-1
    efficiency_score: float  # 0-1
    
    # 上下文信息
    current_task: Optional[str] = None
    task_difficulty: float = 0.5  # 0-1
    time_pressure: float = 0.5  # 0-1
    importance: float = 0.5  # 0-1
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegulationTarget:
    """调节目标"""
    target_id: str
    regulation_type: RegulationType
    goal: RegulationGoal
    target_variable: str
    current_value: float
    target_value: float
    acceptable_range: Tuple[float, float]
    priority: int  # 1-10
    time_constraint: Optional[timedelta] = None
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegulationStrategy:
    """调节策略"""
    strategy_id: str
    regulation_targets: List[RegulationTarget]
    strategy_type: str
    intensity: RegulationIntensity
    implementation_plan: List[Dict[str, Any]]
    expected_effects: Dict[str, float]
    potential_risks: List[str]
    risk_mitigation: List[str]
    confidence: float  # 0-1
    cost_estimate: float  # 0-1
    time_estimate: timedelta
    learning_opportunities: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegulationResult:
    """调节结果"""
    result_id: str
    regulation_strategy: RegulationStrategy
    timestamp: datetime
    execution_duration: timedelta
    actual_changes: Dict[str, Dict[str, float]]  # 变量: {old: 新值, new: 新值}
    effectiveness_score: float  # 0-1
    efficiency_score: float  # 0-1
    side_effects: List[str]
    learning_outcomes: List[str]
    recommendations_for_future: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegulationHistory:
    """调节历史"""
    regulation_history: List[RegulationResult]
    success_rate: float  # 0-1
    average_effectiveness: float  # 0-1
    preferred_strategies: Dict[str, int]  # 策略类型: 使用次数
    learning_accumulated: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class CognitiveRegulationMechanism:
    """
    认知调节机制 - 主类
    
    工作流程:
    1. 认知状态监控: 收集和分析当前认知状态
    2. 调节需求分析: 识别需要调节的认知维度
    3. 策略生成: 生成针对性的调节策略
    4. 策略执行: 执行调节策略
    5. 效果评估: 评估调节效果
    6. 学习更新: 从调节经验中学习
    """
    
    def __init__(self,
                 thinking_tracker: Optional[ThinkingProcessTracker] = None,
                 bias_detector: Optional[CognitiveBiasDetector] = None,
                 strategy_evaluator: Optional[ReasoningStrategyEvaluator] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化认知调节机制
        
        Args:
            thinking_tracker: 思维过程追踪器
            bias_detector: 认知偏差检测器
            strategy_evaluator: 推理策略评估器
            config: 配置参数
        """
        # 依赖组件
        self.thinking_tracker = thinking_tracker
        self.bias_detector = bias_detector
        self.strategy_evaluator = strategy_evaluator
        
        # 配置参数
        self.config = config or self._get_default_config()
        
        # 认知状态历史
        self.cognitive_state_history: List[CognitiveState] = []
        self.max_history_size = self.config.get('max_history_size', 1000)
        
        # 调节历史
        self.regulation_history: RegulationHistory = RegulationHistory(
            regulation_history=[],
            success_rate=0.5,
            average_effectiveness=0.5,
            preferred_strategies={},
            learning_accumulated=[],
            metadata={'created_at': datetime.now()}
        )
        
        # 当前状态
        self.current_cognitive_state: Optional[CognitiveState] = None
        self.active_regulation_targets: List[RegulationTarget] = []
        self.current_regulation_strategy: Optional[RegulationStrategy] = None
        
        # 性能统计
        self.performance_stats = {
            'regulations_initiated': 0,
            'regulations_completed': 0,
            'successful_regulations': 0,
            'average_regulation_time': 0.0,
            'regulations_by_type': defaultdict(int),
            'regulations_by_goal': defaultdict(int)
        }
        
        # 学习知识
        self.regulation_knowledge_base = self._initialize_knowledge_base()
        
        logger.info("认知调节机制初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'max_history_size': 1000,
            'regulation_threshold': 0.3,  # 状态差异阈值
            'min_confidence_for_regulation': 0.6,
            'regulation_frequency_limit': timedelta(seconds=10),
            'performance_improvement_threshold': 0.1,
            'cognitive_load_threshold': 0.8,
            'attention_focus_threshold': 0.4,
            'memory_load_threshold': 0.7,
            'monitoring_interval': timedelta(seconds=5),
            'regulation_effectiveness_window': timedelta(minutes=30),
            'learning_rate': 0.1,
            'exploration_rate': 0.1,
            'safety_margin': 0.2,
            'emergency_regulation_threshold': 0.8,
            'adaptive_parameter_update_frequency': 100
        }
    
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """初始化知识库"""
        return {
            'regulation_strategies': {
                'attention_regulation': [
                    {
                        'name': 'selective_focusing',
                        'description': '选择性聚焦注意力',
                        'applicable_when': ['attention_focus < 0.5', 'task_requires_deep_work'],
                        'implementation': ['设定单一任务焦点', '消除干扰源', '启用专注定时器'],
                        'expected_effects': {'attention_focus': 0.3, 'performance_score': 0.2},
                        'risks': ['可能错过重要信息', '任务切换困难']
                    },
                    {
                        'name': 'attention_switching',
                        'description': '注意力的有效切换',
                        'applicable_when': ['multiple_tasks', 'time_pressure > 0.6'],
                        'implementation': ['设置任务切换计划', '保留上下文', '切换缓冲时间'],
                        'expected_effects': {'resource_allocation_efficiency': 0.2, 'time_efficiency': 0.15},
                        'risks': ['认知疲劳', '任务混淆']
                    }
                ],
                'memory_regulation': [
                    {
                        'name': 'working_memory_optimization',
                        'description': '优化工作记忆使用',
                        'applicable_when': ['working_memory_load > 0.7', 'complex_task'],
                        'implementation': ['信息分块', '外部化存储', '简化表示'],
                        'expected_effects': {'working_memory_load': -0.3, 'reasoning_accuracy': 0.1},
                        'risks': ['信息丢失', '过度简化']
                    },
                    {
                        'name': 'memory_retrieval_enhancement',
                        'description': '增强记忆检索',
                        'applicable_when': ['memory_retrieval_speed < 0.5', 'knowledge_intensive_task'],
                        'implementation': ['建立知识关联', '使用记忆线索', '加强重复检索'],
                        'expected_effects': {'memory_retrieval_speed': 0.25, 'decision_confidence': 0.1},
                        'risks': ['错误关联', '记忆污染']
                    }
                ],
                'reasoning_regulation': [
                    {
                        'name': 'depth_vs_breadth_balance',
                        'description': '深度与广度的平衡',
                        'applicable_when': ['reasoning_depth < 0.4', 'task_requires_depth'],
                        'implementation': ['增加推理步骤', '加强假设检验', '深入探索分支'],
                        'expected_effects': {'reasoning_depth': 0.25, 'reasoning_accuracy': 0.15},
                        'risks': ['过度分析', '时间消耗']
                    },
                    {
                        'name': 'strategic_abstraction',
                        'description': '策略性抽象',
                        'applicable_when': ['overly_concrete', 'need_pattern_recognition'],
                        'implementation': ['识别模式', '提取本质特征', '建立抽象模型'],
                        'expected_effects': {'reasoning_speed': 0.2, 'creativity_score': 0.15},
                        'risks': ['过度抽象', '失去具体细节']
                    }
                ],
                'emotion_regulation': [
                    {
                        'name': 'emotional_cognitive_reappraisal',
                        'description': '情绪认知重评',
                        'applicable_when': ['emotional_arousal > 0.7', 'emotion_impairs_cognition'],
                        'implementation': ['重新解释情境', '调整期望', '关注可控因素'],
                        'expected_effects': {'emotional_arousal': -0.3, 'reasoning_accuracy': 0.15},
                        'risks': ['情绪抑制', '认知回避']
                    }
                ]
            },
            'context_adaptation_rules': [
                {
                    'context_pattern': 'high_time_pressure',
                    'adaptations': ['prefer_faster_strategies', 'reduce_perfectionism', 'increase_risk_tolerance']
                },
                {
                    'context_pattern': 'high_importance_task',
                    'adaptations': ['increase_caution', 'use_more_strategies', 'enable_verification']
                },
                {
                    'context_pattern': 'novel_problem',
                    'adaptations': ['increase_exploration', 'allow_more_errors', 'use_analogical_reasoning']
                }
            ],
            'individual_differences': {
                'cognitive_style': 'balanced',  # 可调整
                'risk_preference': 'moderate',
                'learning_preference': 'experiential',
                'attention_style': 'flexible',
                'memory_capacity': 'average'
            }
        }
    
    def monitor_cognitive_state(self, context: Optional[Dict[str, Any]] = None) -> CognitiveState:
        """
        监控认知状态
        
        Args:
            context: 上下文信息
            
        Returns:
            当前认知状态
        """
        try:
            timestamp = datetime.now()
            
            # 收集各维度的状态信息
            attention_state = self._assess_attention_state(context)
            memory_state = self._assess_memory_state(context)
            reasoning_state = self._assess_reasoning_state(context)
            decision_state = self._assess_decision_state(context)
            emotion_state = self._assess_emotion_state(context)
            resource_state = self._assess_resource_state(context)
            
            # 从元认知组件获取信息
            metacognitive_state = self._get_metacognitive_state()
            
            # 计算性能指标
            performance_metrics = self._calculate_performance_metrics()
            
            # 构建认知状态对象
            cognitive_state = CognitiveState(
                timestamp=timestamp,
                # 注意力状态
                attention_focus=attention_state.get('focus_level', 0.5),
                attention_distribution=attention_state.get('distribution', {}),
                attention_stability=attention_state.get('stability', 0.5),
                
                # 记忆状态
                working_memory_load=memory_state.get('working_memory_load', 0.5),
                long_term_memory_access=memory_state.get('long_term_access', 0.5),
                memory_retrieval_speed=memory_state.get('retrieval_speed', 0.5),
                
                # 推理状态
                reasoning_depth=reasoning_state.get('depth', 0.5),
                reasoning_speed=reasoning_state.get('speed', 0.5),
                reasoning_accuracy=reasoning_state.get('accuracy', 0.5),
                
                # 决策状态
                decision_confidence=decision_state.get('confidence', 0.5),
                decision_speed=decision_state.get('speed', 0.5),
                risk_tolerance=decision_state.get('risk_tolerance', 0.5),
                
                # 情绪状态
                emotional_valence=emotion_state.get('valence', 0.0),
                emotional_arousal=emotion_state.get('arousal', 0.5),
                emotion_cognition_influence=emotion_state.get('influence', 0.5),
                
                # 资源状态
                cognitive_resource_available=resource_state.get('available', 0.5),
                resource_allocation_efficiency=resource_state.get('efficiency', 0.5),
                
                # 元认知状态
                self_awareness=metacognitive_state.get('self_awareness', 0.5),
                metacognitive_control=metacognitive_state.get('control', 0.5),
                
                # 性能指标
                performance_score=performance_metrics.get('performance', 0.5),
                efficiency_score=performance_metrics.get('efficiency', 0.5),
                
                # 上下文信息
                current_task=context.get('current_task') if context else None,
                task_difficulty=context.get('task_difficulty', 0.5) if context else 0.5,
                time_pressure=context.get('time_pressure', 0.5) if context else 0.5,
                importance=context.get('importance', 0.5) if context else 0.5,
                
                metadata={
                    'source': 'cognitive_regulation_mechanism',
                    'monitoring_method': 'multi_dimensional_assessment',
                    'confidence': 0.7
                }
            )
            
            # 更新当前状态和历史
            self.current_cognitive_state = cognitive_state
            self.cognitive_state_history.append(cognitive_state)
            
            # 限制历史大小
            if len(self.cognitive_state_history) > self.max_history_size:
                self.cognitive_state_history = self.cognitive_state_history[-self.max_history_size:]
            
            logger.debug(f"认知状态监控完成: {cognitive_state.timestamp}")
            return cognitive_state
            
        except Exception as e:
            logger.error(f"认知状态监控失败: {e}")
            # 返回默认状态
            return CognitiveState(
                timestamp=datetime.now(),
                attention_focus=0.5,
                attention_distribution={},
                attention_stability=0.5,
                working_memory_load=0.5,
                long_term_memory_access=0.5,
                memory_retrieval_speed=0.5,
                reasoning_depth=0.5,
                reasoning_speed=0.5,
                reasoning_accuracy=0.5,
                decision_confidence=0.5,
                decision_speed=0.5,
                risk_tolerance=0.5,
                emotional_valence=0.0,
                emotional_arousal=0.5,
                emotion_cognition_influence=0.5,
                cognitive_resource_available=0.5,
                resource_allocation_efficiency=0.5,
                self_awareness=0.5,
                metacognitive_control=0.5,
                performance_score=0.5,
                efficiency_score=0.5
            )
    
    def analyze_regulation_needs(self, 
                                cognitive_state: Optional[CognitiveState] = None,
                                context: Optional[Dict[str, Any]] = None) -> List[RegulationTarget]:
        """
        分析调节需求
        
        Args:
            cognitive_state: 当前认知状态
            context: 上下文信息
            
        Returns:
            调节目标列表
        """
        try:
            # 使用提供的状态或当前状态
            if cognitive_state is None:
                if self.current_cognitive_state is None:
                    cognitive_state = self.monitor_cognitive_state(context)
                else:
                    cognitive_state = self.current_cognitive_state
            
            # 分析各维度的调节需求
            regulation_targets = []
            target_id_counter = 1
            
            # 1. 分析注意力调节需求
            attention_targets = self._analyze_attention_needs(cognitive_state, context)
            regulation_targets.extend(attention_targets)
            
            # 2. 分析记忆调节需求
            memory_targets = self._analyze_memory_needs(cognitive_state, context)
            regulation_targets.extend(memory_targets)
            
            # 3. 分析推理调节需求
            reasoning_targets = self._analyze_reasoning_needs(cognitive_state, context)
            regulation_targets.extend(reasoning_targets)
            
            # 4. 分析决策调节需求
            decision_targets = self._analyze_decision_needs(cognitive_state, context)
            regulation_targets.extend(decision_targets)
            
            # 5. 分析情绪调节需求
            emotion_targets = self._analyze_emotion_needs(cognitive_state, context)
            regulation_targets.extend(emotion_targets)
            
            # 6. 分析资源调节需求
            resource_targets = self._analyze_resource_needs(cognitive_state, context)
            regulation_targets.extend(resource_targets)
            
            # 为每个目标分配ID
            for i, target in enumerate(regulation_targets):
                if not hasattr(target, 'target_id') or not target.target_id:
                    target.target_id = f"regulation_target_{target_id_counter}"
                    target_id_counter += 1
            
            # 设置活动调节目标
            self.active_regulation_targets = regulation_targets
            
            logger.info(f"调节需求分析完成，发现 {len(regulation_targets)} 个调节目标")
            return regulation_targets
            
        except Exception as e:
            logger.error(f"调节需求分析失败: {e}")
            return []
    
    def generate_regulation_strategy(self,
                                   regulation_targets: List[RegulationTarget],
                                   context: Optional[Dict[str, Any]] = None) -> RegulationStrategy:
        """
        生成调节策略
        
        Args:
            regulation_targets: 调节目标列表
            context: 上下文信息
            
        Returns:
            调节策略
        """
        try:
            if not regulation_targets:
                raise ValueError("没有提供调节目标")
            
            # 分析调节目标的优先级和相互关系
            prioritized_targets = self._prioritize_targets(regulation_targets, context)
            
            # 根据目标类型选择合适的调节类型
            primary_regulation_type = self._identify_primary_regulation_type(prioritized_targets)
            
            # 确定调节强度
            regulation_intensity = self._determine_regulation_intensity(prioritized_targets, context)
            
            # 生成实施计划
            implementation_plan = self._generate_implementation_plan(prioritized_targets, context)
            
            # 估计预期效果
            expected_effects = self._estimate_expected_effects(prioritized_targets, implementation_plan)
            
            # 识别潜在风险
            potential_risks = self._identify_potential_risks(implementation_plan, context)
            
            # 制定风险缓解措施
            risk_mitigation = self._develop_risk_mitigation(potential_risks)
            
            # 生成策略ID
            strategy_id = f"regulation_strategy_{int(time.time())}_{random.randint(1000, 9999)}"
            
            # 构建调节策略
            regulation_strategy = RegulationStrategy(
                strategy_id=strategy_id,
                regulation_targets=prioritized_targets,
                strategy_type=primary_regulation_type.value if primary_regulation_type else "balanced",
                intensity=regulation_intensity,
                implementation_plan=implementation_plan,
                expected_effects=expected_effects,
                potential_risks=potential_risks,
                risk_mitigation=risk_mitigation,
                confidence=self._calculate_strategy_confidence(prioritized_targets, implementation_plan),
                cost_estimate=self._estimate_regulation_cost(implementation_plan),
                time_estimate=self._estimate_regulation_time(implementation_plan),
                learning_opportunities=self._identify_learning_opportunities(prioritized_targets),
                metadata={
                    'generated_at': datetime.now().isoformat(),
                    'context': context or {},
                    'target_count': len(regulation_targets)
                }
            )
            
            # 更新当前调节策略
            self.current_regulation_strategy = regulation_strategy
            
            logger.info(f"调节策略生成完成: {strategy_id}")
            return regulation_strategy
            
        except Exception as e:
            logger.error(f"调节策略生成失败: {e}")
            # 返回默认策略
            return RegulationStrategy(
                strategy_id=f"fallback_strategy_{int(time.time())}",
                regulation_targets=regulation_targets,
                strategy_type="balanced",
                intensity=RegulationIntensity.MODERATE,
                implementation_plan=[],
                expected_effects={},
                potential_risks=[],
                risk_mitigation=[],
                confidence=0.5,
                cost_estimate=0.5,
                time_estimate=timedelta(seconds=30),
                learning_opportunities=[]
            )
    
    def execute_regulation_strategy(self,
                                  regulation_strategy: RegulationStrategy,
                                  context: Optional[Dict[str, Any]] = None) -> RegulationResult:
        """
        执行调节策略
        
        Args:
            regulation_strategy: 调节策略
            context: 上下文信息
            
        Returns:
            调节结果
        """
        start_time = datetime.now()
        
        try:
            self.performance_stats['regulations_initiated'] += 1
            
            # 记录执行前的状态
            state_before = self.current_cognitive_state or self.monitor_cognitive_state(context)
            
            # 执行调节计划
            actual_changes = {}
            execution_log = []
            side_effects = []
            
            for step in regulation_strategy.implementation_plan:
                step_start = datetime.now()
                
                try:
                    # 执行调节步骤
                    step_result = self._execute_regulation_step(step, context)
                    
                    # 记录步骤结果
                    execution_log.append({
                        'step': step,
                        'result': step_result,
                        'duration': (datetime.now() - step_start).total_seconds()
                    })
                    
                    # 收集实际变化
                    if 'changes' in step_result:
                        for variable, change in step_result['changes'].items():
                            if variable not in actual_changes:
                                actual_changes[variable] = {'old': None, 'new': None}
                            actual_changes[variable]['new'] = change['new_value']
                            
                            if actual_changes[variable]['old'] is None:
                                # 查找旧值
                                old_value = self._get_state_value(state_before, variable)
                                actual_changes[variable]['old'] = old_value
                    
                    # 检查副作用
                    if 'side_effects' in step_result:
                        side_effects.extend(step_result['side_effects'])
                        
                except Exception as step_error:
                    logger.warning(f"调节步骤执行失败: {step_error}")
                    execution_log.append({
                        'step': step,
                        'error': str(step_error),
                        'duration': (datetime.now() - step_start).total_seconds()
                    })
            
            # 记录执行后的状态
            state_after = self.monitor_cognitive_state(context)
            
            # 计算执行持续时间
            execution_duration = datetime.now() - start_time
            
            # 评估调节效果
            effectiveness_score = self._evaluate_regulation_effectiveness(
                regulation_strategy, 
                state_before, 
                state_after,
                actual_changes
            )
            
            # 计算调节效率
            efficiency_score = self._calculate_regulation_efficiency(
                regulation_strategy,
                execution_duration,
                effectiveness_score
            )
            
            # 识别学习机会
            learning_outcomes = self._extract_learning_outcomes(
                regulation_strategy,
                execution_log,
                effectiveness_score
            )
            
            # 生成未来建议
            recommendations_for_future = self._generate_recommendations_for_future(
                regulation_strategy,
                effectiveness_score,
                side_effects
            )
            
            # 构建调节结果
            regulation_result = RegulationResult(
                result_id=f"regulation_result_{int(time.time())}_{random.randint(1000, 9999)}",
                regulation_strategy=regulation_strategy,
                timestamp=datetime.now(),
                execution_duration=execution_duration,
                actual_changes=actual_changes,
                effectiveness_score=effectiveness_score,
                efficiency_score=efficiency_score,
                side_effects=side_effects,
                learning_outcomes=learning_outcomes,
                recommendations_for_future=recommendations_for_future,
                metadata={
                    'execution_log': execution_log,
                    'state_before': self._state_to_dict(state_before),
                    'state_after': self._state_to_dict(state_after),
                    'context': context or {}
                }
            )
            
            # 更新调节历史
            self.regulation_history.regulation_history.append(regulation_result)
            
            # 更新性能统计
            self.performance_stats['regulations_completed'] += 1
            if effectiveness_score > 0.6:
                self.performance_stats['successful_regulations'] += 1
            
            # 更新策略类型统计
            strategy_type = regulation_strategy.strategy_type
            self.performance_stats['regulations_by_type'][strategy_type] += 1
            
            # 更新目标类型统计
            for target in regulation_strategy.regulation_targets:
                goal_value = target.goal.value if hasattr(target.goal, 'value') else str(target.goal)
                self.performance_stats['regulations_by_goal'][goal_value] += 1
            
            # 更新平均执行时间
            total_time = self.performance_stats.get('total_regulation_time', 0.0)
            total_time += execution_duration.total_seconds()
            self.performance_stats['total_regulation_time'] = total_time
            self.performance_stats['average_regulation_time'] = (
                total_time / self.performance_stats['regulations_completed']
            )
            
            # 学习和更新知识库
            self._learn_from_regulation_result(regulation_result)
            
            logger.info(f"调节策略执行完成: {regulation_result.result_id}, 效果: {effectiveness_score:.2f}")
            return regulation_result
            
        except Exception as e:
            logger.error(f"调节策略执行失败: {e}")
            
            # 返回失败结果
            return RegulationResult(
                result_id=f"regulation_failure_{int(time.time())}",
                regulation_strategy=regulation_strategy,
                timestamp=datetime.now(),
                execution_duration=datetime.now() - start_time,
                actual_changes={},
                effectiveness_score=0.0,
                efficiency_score=0.0,
                side_effects=["执行失败: " + str(e)],
                learning_outcomes=["执行过程中发生错误"],
                recommendations_for_future=["检查错误原因", "简化调节策略"],
                metadata={'error': str(e)}
            )
    
    def regulate_cognitive_process(self, 
                                  context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        完整的认知调节过程（一站式接口）
        
        Args:
            context: 上下文信息
            
        Returns:
            调节过程结果
        """
        try:
            # 1. 监控认知状态
            cognitive_state = self.monitor_cognitive_state(context)
            
            # 2. 分析调节需求
            regulation_targets = self.analyze_regulation_needs(cognitive_state, context)
            
            # 3. 如果没有调节需求，返回当前状态
            if not regulation_targets:
                return {
                    'success': True,
                    'regulation_needed': False,
                    'message': '当前认知状态良好，无需调节',
                    'cognitive_state': self._state_to_dict(cognitive_state),
                    'timestamp': datetime.now().isoformat()
                }
            
            # 4. 生成调节策略
            regulation_strategy = self.generate_regulation_strategy(regulation_targets, context)
            
            # 5. 执行调节策略
            regulation_result = self.execute_regulation_strategy(regulation_strategy, context)
            
            # 6. 返回完整调节结果
            return {
                'success': True,
                'regulation_needed': True,
                'cognitive_state_before': self._state_to_dict(cognitive_state),
                'regulation_targets': [self._target_to_dict(t) for t in regulation_targets],
                'regulation_strategy': self._strategy_to_dict(regulation_strategy),
                'regulation_result': self._result_to_dict(regulation_result),
                'effectiveness': regulation_result.effectiveness_score,
                'efficiency': regulation_result.efficiency_score,
                'timestamp': datetime.now().isoformat(),
                'message': f'认知调节完成，效果评分: {regulation_result.effectiveness_score:.2f}'
            }
            
        except Exception as e:
            logger.error(f"认知调节过程失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _assess_attention_state(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """评估注意力状态"""
        # 简化实现，实际应基于思维追踪器和其他监控数据
        try:
            if self.thinking_tracker:
                # 使用思维追踪器的注意力数据
                attention_data = self.thinking_tracker.get_attention_state()
                return {
                    'focus_level': attention_data.get('focus_level', 0.5),
                    'distribution': attention_data.get('distribution', {'default': 1.0}),
                    'stability': attention_data.get('stability', 0.5),
                    'metadata': {'source': 'thinking_tracker'}
                }
        except Exception:
            pass
        
        # 默认状态
        return {
            'focus_level': 0.5 + random.uniform(-0.2, 0.2),
            'distribution': {'current_task': 0.7, 'background': 0.3},
            'stability': 0.6 + random.uniform(-0.1, 0.1),
            'metadata': {'source': 'default_estimation'}
        }
    
    def _assess_memory_state(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """评估记忆状态"""
        # 简化实现
        return {
            'working_memory_load': 0.5 + random.uniform(-0.2, 0.2),
            'long_term_access': 0.6 + random.uniform(-0.1, 0.1),
            'retrieval_speed': 0.7 + random.uniform(-0.15, 0.15),
            'metadata': {'source': 'default_estimation'}
        }
    
    def _assess_reasoning_state(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """评估推理状态"""
        # 简化实现
        return {
            'depth': 0.5 + random.uniform(-0.2, 0.2),
            'speed': 0.6 + random.uniform(-0.15, 0.15),
            'accuracy': 0.7 + random.uniform(-0.1, 0.1),
            'metadata': {'source': 'default_estimation'}
        }
    
    def _assess_decision_state(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """评估决策状态"""
        # 简化实现
        return {
            'confidence': 0.6 + random.uniform(-0.15, 0.15),
            'speed': 0.5 + random.uniform(-0.2, 0.2),
            'risk_tolerance': 0.5 + random.uniform(-0.1, 0.1),
            'metadata': {'source': 'default_estimation'}
        }
    
    def _assess_emotion_state(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """评估情绪状态"""
        # 简化实现
        return {
            'valence': random.uniform(-0.5, 0.5),
            'arousal': 0.5 + random.uniform(-0.2, 0.2),
            'influence': 0.3 + random.uniform(-0.1, 0.1),
            'metadata': {'source': 'default_estimation'}
        }
    
    def _assess_resource_state(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """评估资源状态"""
        # 简化实现
        return {
            'available': 0.7 + random.uniform(-0.2, 0.2),
            'efficiency': 0.6 + random.uniform(-0.15, 0.15),
            'metadata': {'source': 'default_estimation'}
        }
    
    def _get_metacognitive_state(self) -> Dict[str, Any]:
        """获取元认知状态"""
        # 简化实现
        return {
            'self_awareness': 0.6 + random.uniform(-0.1, 0.1),
            'control': 0.5 + random.uniform(-0.15, 0.15),
            'metadata': {'source': 'default_estimation'}
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """计算性能指标"""
        # 简化实现
        return {
            'performance': 0.7 + random.uniform(-0.15, 0.15),
            'efficiency': 0.65 + random.uniform(-0.1, 0.1),
            'metadata': {'source': 'default_estimation'}
        }
    
    def _analyze_attention_needs(self, 
                                cognitive_state: CognitiveState,
                                context: Optional[Dict[str, Any]] = None) -> List[RegulationTarget]:
        """分析注意力调节需求"""
        targets = []
        
        # 检查注意力聚焦度
        if cognitive_state.attention_focus < self.config.get('attention_focus_threshold', 0.4):
            target = RegulationTarget(
                target_id="",
                regulation_type=RegulationType.ATTENTION_REGULATION,
                goal=RegulationGoal.IMPROVE_PERFORMANCE,
                target_variable="attention_focus",
                current_value=cognitive_state.attention_focus,
                target_value=min(1.0, cognitive_state.attention_focus + 0.3),
                acceptable_range=(0.6, 1.0),
                priority=8,
                time_constraint=timedelta(minutes=5),
                constraints=["保持任务切换能力", "避免过度聚焦"],
                metadata={"reason": "注意力聚焦度不足", "threshold": self.config.get('attention_focus_threshold', 0.4)}
            )
            targets.append(target)
        
        # 检查认知负载
        if cognitive_state.working_memory_load > self.config.get('cognitive_load_threshold', 0.8):
            target = RegulationTarget(
                target_id="",
                regulation_type=RegulationType.ATTENTION_REGULATION,
                goal=RegulationGoal.REDUCE_COGNITIVE_LOAD,
                target_variable="working_memory_load",
                current_value=cognitive_state.working_memory_load,
                target_value=max(0.0, cognitive_state.working_memory_load - 0.3),
                acceptable_range=(0.0, 0.7),
                priority=9,
                time_constraint=timedelta(minutes=3),
                constraints=["保持必要的信息处理", "避免信息丢失"],
                metadata={"reason": "工作记忆负载过高", "threshold": self.config.get('cognitive_load_threshold', 0.8)}
            )
            targets.append(target)
        
        return targets
    
    def _analyze_memory_needs(self, 
                             cognitive_state: CognitiveState,
                             context: Optional[Dict[str, Any]] = None) -> List[RegulationTarget]:
        """分析记忆调节需求"""
        targets = []
        
        # 检查记忆负载
        if cognitive_state.working_memory_load > self.config.get('memory_load_threshold', 0.7):
            target = RegulationTarget(
                target_id="",
                regulation_type=RegulationType.MEMORY_REGULATION,
                goal=RegulationGoal.REDUCE_COGNITIVE_LOAD,
                target_variable="working_memory_load",
                current_value=cognitive_state.working_memory_load,
                target_value=max(0.0, cognitive_state.working_memory_load - 0.25),
                acceptable_range=(0.0, 0.6),
                priority=7,
                time_constraint=timedelta(minutes=5),
                constraints=["保持关键信息", "优化信息组织"],
                metadata={"reason": "记忆负载过高", "threshold": self.config.get('memory_load_threshold', 0.7)}
            )
            targets.append(target)
        
        return targets
    
    def _analyze_reasoning_needs(self, 
                                cognitive_state: CognitiveState,
                                context: Optional[Dict[str, Any]] = None) -> List[RegulationTarget]:
        """分析推理调节需求"""
        targets = []
        
        # 检查推理准确度
        if cognitive_state.reasoning_accuracy < 0.6:
            target = RegulationTarget(
                target_id="",
                regulation_type=RegulationType.REASONING_REGULATION,
                goal=RegulationGoal.INCREASE_ACCURACY,
                target_variable="reasoning_accuracy",
                current_value=cognitive_state.reasoning_accuracy,
                target_value=min(1.0, cognitive_state.reasoning_accuracy + 0.25),
                acceptable_range=(0.7, 1.0),
                priority=6,
                time_constraint=timedelta(minutes=10),
                constraints=["保持推理速度", "避免过度分析"],
                metadata={"reason": "推理准确度不足", "threshold": 0.6}
            )
            targets.append(target)
        
        return targets
    
    def _analyze_decision_needs(self, 
                               cognitive_state: CognitiveState,
                               context: Optional[Dict[str, Any]] = None) -> List[RegulationTarget]:
        """分析决策调节需求"""
        targets = []
        
        # 检查决策置信度
        if cognitive_state.decision_confidence < 0.5:
            target = RegulationTarget(
                target_id="",
                regulation_type=RegulationType.DECISION_REGULATION,
                goal=RegulationGoal.IMPROVE_PERFORMANCE,
                target_variable="decision_confidence",
                current_value=cognitive_state.decision_confidence,
                target_value=min(1.0, cognitive_state.decision_confidence + 0.3),
                acceptable_range=(0.6, 1.0),
                priority=5,
                time_constraint=timedelta(minutes=8),
                constraints=["保持理性分析", "避免过度自信"],
                metadata={"reason": "决策置信度不足", "threshold": 0.5}
            )
            targets.append(target)
        
        return targets
    
    def _analyze_emotion_needs(self, 
                              cognitive_state: CognitiveState,
                              context: Optional[Dict[str, Any]] = None) -> List[RegulationTarget]:
        """分析情绪调节需求"""
        targets = []
        
        # 检查情绪唤醒度
        if cognitive_state.emotional_arousal > 0.8:
            target = RegulationTarget(
                target_id="",
                regulation_type=RegulationType.EMOTION_REGULATION,
                goal=RegulationGoal.MANAGE_EMOTION,
                target_variable="emotional_arousal",
                current_value=cognitive_state.emotional_arousal,
                target_value=max(0.0, cognitive_state.emotional_arousal - 0.3),
                acceptable_range=(0.2, 0.7),
                priority=8,
                time_constraint=timedelta(minutes=3),
                constraints=["保持适当情绪", "避免情绪抑制"],
                metadata={"reason": "情绪唤醒度过高", "threshold": 0.8}
            )
            targets.append(target)
        
        return targets
    
    def _analyze_resource_needs(self, 
                               cognitive_state: CognitiveState,
                               context: Optional[Dict[str, Any]] = None) -> List[RegulationTarget]:
        """分析资源调节需求"""
        targets = []
        
        # 检查可用认知资源
        if cognitive_state.cognitive_resource_available < 0.3:
            target = RegulationTarget(
                target_id="",
                regulation_type=RegulationType.RESOURCE_REGULATION,
                goal=RegulationGoal.OPTIMIZE_RESOURCE,
                target_variable="cognitive_resource_available",
                current_value=cognitive_state.cognitive_resource_available,
                target_value=min(1.0, cognitive_state.cognitive_resource_available + 0.4),
                acceptable_range=(0.4, 1.0),
                priority=9,
                time_constraint=timedelta(minutes=15),
                constraints=["逐步恢复", "避免过度消耗"],
                metadata={"reason": "认知资源不足", "threshold": 0.3}
            )
            targets.append(target)
        
        return targets
    
    def _prioritize_targets(self, 
                           regulation_targets: List[RegulationTarget],
                           context: Optional[Dict[str, Any]] = None) -> List[RegulationTarget]:
        """对调节目标进行优先级排序"""
        if not regulation_targets:
            return []
        
        # 根据优先级排序
        sorted_targets = sorted(regulation_targets, key=lambda t: t.priority, reverse=True)
        
        # 限制同时处理的目标数量
        max_targets = self.config.get('max_concurrent_targets', 3)
        return sorted_targets[:max_targets]
    
    def _identify_primary_regulation_type(self, 
                                         regulation_targets: List[RegulationTarget]) -> Optional[RegulationType]:
        """识别主要调节类型"""
        if not regulation_targets:
            return None
        
        # 统计各调节类型的出现频率
        type_counts = defaultdict(int)
        for target in regulation_targets:
            type_counts[target.regulation_type] += 1
        
        # 返回最常见的调节类型
        if type_counts:
            most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
            return most_common_type
        
        return None
    
    def _determine_regulation_intensity(self, 
                                       regulation_targets: List[RegulationTarget],
                                       context: Optional[Dict[str, Any]] = None) -> RegulationIntensity:
        """确定调节强度"""
        if not regulation_targets:
            return RegulationIntensity.MINIMAL
        
        # 计算平均优先级和调整幅度
        avg_priority = sum(t.priority for t in regulation_targets) / len(regulation_targets)
        avg_adjustment = sum(abs(t.target_value - t.current_value) for t in regulation_targets) / len(regulation_targets)
        
        # 根据优先级和调整幅度确定强度
        if avg_priority >= 9 or avg_adjustment >= 0.4:
            return RegulationIntensity.DRAMATIC
        elif avg_priority >= 7 or avg_adjustment >= 0.25:
            return RegulationIntensity.SUBSTANTIAL
        elif avg_priority >= 5 or avg_adjustment >= 0.15:
            return RegulationIntensity.MODERATE
        else:
            return RegulationIntensity.MINIMAL
    
    def _generate_implementation_plan(self, 
                                     regulation_targets: List[RegulationTarget],
                                     context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """生成实施计划"""
        plan = []
        
        for i, target in enumerate(regulation_targets):
            # 根据目标类型生成具体步骤
            if target.regulation_type == RegulationType.ATTENTION_REGULATION:
                step = {
                    'step_id': f"attention_step_{i+1}",
                    'action': '调整注意力配置',
                    'target_variable': target.target_variable,
                    'adjustment_amount': target.target_value - target.current_value,
                    'method': '注意力重分配技术',
                    'estimated_duration': timedelta(seconds=30),
                    'prerequisites': [],
                    'expected_outcome': f"{target.target_variable} 从 {target.current_value:.2f} 调整到 {target.target_value:.2f}"
                }
                plan.append(step)
            
            elif target.regulation_type == RegulationType.MEMORY_REGULATION:
                step = {
                    'step_id': f"memory_step_{i+1}",
                    'action': '优化记忆管理',
                    'target_variable': target.target_variable,
                    'adjustment_amount': target.target_value - target.current_value,
                    'method': '记忆负载减轻策略',
                    'estimated_duration': timedelta(seconds=45),
                    'prerequisites': [],
                    'expected_outcome': f"{target.target_variable} 从 {target.current_value:.2f} 调整到 {target.target_value:.2f}"
                }
                plan.append(step)
            
            elif target.regulation_type == RegulationType.REASONING_REGULATION:
                step = {
                    'step_id': f"reasoning_step_{i+1}",
                    'action': '改进推理过程',
                    'target_variable': target.target_variable,
                    'adjustment_amount': target.target_value - target.current_value,
                    'method': '推理策略优化',
                    'estimated_duration': timedelta(minutes=1),
                    'prerequisites': [],
                    'expected_outcome': f"{target.target_variable} 从 {target.current_value:.2f} 调整到 {target.target_value:.2f}"
                }
                plan.append(step)
            
            else:
                # 通用步骤
                step = {
                    'step_id': f"general_step_{i+1}",
                    'action': f'调节{target.regulation_type.value}',
                    'target_variable': target.target_variable,
                    'adjustment_amount': target.target_value - target.current_value,
                    'method': '自适应调节技术',
                    'estimated_duration': timedelta(seconds=40),
                    'prerequisites': [],
                    'expected_outcome': f"{target.target_variable} 从 {target.current_value:.2f} 调整到 {target.target_value:.2f}"
                }
                plan.append(step)
        
        return plan
    
    def _estimate_expected_effects(self, 
                                  regulation_targets: List[RegulationTarget],
                                  implementation_plan: List[Dict[str, Any]]) -> Dict[str, float]:
        """估计预期效果"""
        effects = {}
        
        for target in regulation_targets:
            variable = target.target_variable
            expected_change = target.target_value - target.current_value
            
            # 主要效果
            effects[variable] = expected_change
            
            # 预期副作用
            if variable == "attention_focus":
                effects["attention_stability"] = expected_change * 0.5
                effects["cognitive_resource_available"] = -abs(expected_change) * 0.3
            
            elif variable == "working_memory_load":
                effects["reasoning_speed"] = -expected_change * 0.4
                effects["decision_confidence"] = -expected_change * 0.2
        
        return effects
    
    def _identify_potential_risks(self, 
                                 implementation_plan: List[Dict[str, Any]],
                                 context: Optional[Dict[str, Any]] = None) -> List[str]:
        """识别潜在风险"""
        risks = []
        
        for step in implementation_plan:
            adjustment = step.get('adjustment_amount', 0)
            variable = step.get('target_variable', '')
            
            if abs(adjustment) > 0.3:
                risks.append(f"过度调节风险: {variable} 调整幅度过大 ({adjustment:.2f})")
            
            if "attention" in variable.lower():
                risks.append("注意力调节可能影响任务切换能力")
            
            if "memory" in variable.lower():
                risks.append("记忆调节可能导致信息丢失")
        
        return risks
    
    def _develop_risk_mitigation(self, potential_risks: List[str]) -> List[str]:
        """制定风险缓解措施"""
        mitigation = []
        
        risk_mapping = {
            "过度调节风险": "采用渐进式调整，分阶段实施",
            "注意力调节可能影响任务切换能力": "保留部分注意力资源用于任务监控",
            "记忆调节可能导致信息丢失": "实施记忆备份和验证机制",
            "推理调节可能降低效率": "平衡深度和速度，设置时间限制"
        }
        
        for risk in potential_risks:
            for risk_pattern, mitigation_strategy in risk_mapping.items():
                if risk_pattern in risk:
                    mitigation.append(mitigation_strategy)
                    break
            else:
                mitigation.append("监控调节过程，及时调整策略")
        
        return mitigation
    
    def _calculate_strategy_confidence(self, 
                                      regulation_targets: List[RegulationTarget],
                                      implementation_plan: List[Dict[str, Any]]) -> float:
        """计算策略置信度"""
        if not regulation_targets:
            return 0.5
        
        # 基础置信度
        confidence = 0.6
        
        # 基于目标数量的调整
        target_count = len(regulation_targets)
        if target_count == 1:
            confidence += 0.1
        elif target_count > 3:
            confidence -= 0.15
        
        # 基于计划复杂度的调整
        plan_complexity = len(implementation_plan)
        if plan_complexity <= 2:
            confidence += 0.1
        elif plan_complexity > 5:
            confidence -= 0.2
        
        # 基于历史成功率的调整
        success_rate = self.regulation_history.success_rate
        confidence += (success_rate - 0.5) * 0.3
        
        return max(0.1, min(1.0, confidence))
    
    def _estimate_regulation_cost(self, implementation_plan: List[Dict[str, Any]]) -> float:
        """估计调节成本"""
        if not implementation_plan:
            return 0.1
        
        # 基于步骤数量和复杂度的成本估计
        base_cost = 0.3
        step_cost = len(implementation_plan) * 0.1
        complexity_cost = sum(1 for step in implementation_plan if step.get('estimated_duration', timedelta(0)) > timedelta(seconds=30)) * 0.05
        
        total_cost = base_cost + step_cost + complexity_cost
        return min(1.0, total_cost)
    
    def _estimate_regulation_time(self, implementation_plan: List[Dict[str, Any]]) -> timedelta:
        """估计调节时间"""
        if not implementation_plan:
            return timedelta(seconds=10)
        
        total_time = timedelta()
        for step in implementation_plan:
            step_time = step.get('estimated_duration', timedelta(seconds=20))
            total_time += step_time
        
        # 增加缓冲时间
        buffer_time = total_time * 0.3
        return total_time + buffer_time
    
    def _identify_learning_opportunities(self, regulation_targets: List[RegulationTarget]) -> List[str]:
        """识别学习机会"""
        opportunities = []
        
        for target in regulation_targets:
            opp = f"学习如何有效调节{target.regulation_type.value.replace('_', ' ')}"
            opportunities.append(opp)
            
            if target.priority >= 8:
                opportunities.append(f"学习高优先级认知调节策略")
            
            if abs(target.target_value - target.current_value) > 0.3:
                opportunities.append(f"学习大幅度认知状态调整技术")
        
        return list(set(opportunities))[:5]  # 限制数量
    
    def _execute_regulation_step(self, 
                                step: Dict[str, Any],
                                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行调节步骤"""
        try:
            step_id = step.get('step_id', 'unknown')
            action = step.get('action', '')
            target_variable = step.get('target_variable', '')
            adjustment_amount = step.get('adjustment_amount', 0)
            
            logger.debug(f"执行调节步骤: {step_id} - {action}")
            
            # 模拟调节执行
            time.sleep(0.1)  # 模拟执行时间
            
            # 生成调节结果
            old_value = self._get_state_value(self.current_cognitive_state, target_variable) if self.current_cognitive_state else 0.5
            new_value = old_value + adjustment_amount * random.uniform(0.8, 1.2)  # 带随机性的实际调整
            
            # 限制值范围
            new_value = max(0.0, min(1.0, new_value))
            
            # 更新当前认知状态
            if self.current_cognitive_state:
                self._set_state_value(self.current_cognitive_state, target_variable, new_value)
            
            result = {
                'step_id': step_id,
                'action': action,
                'success': True,
                'changes': {
                    target_variable: {
                        'old_value': old_value,
                        'new_value': new_value,
                        'adjustment_achieved': new_value - old_value,
                        'adjustment_planned': adjustment_amount
                    }
                },
                'side_effects': [],
                'execution_time': timedelta(seconds=0.1),
                'metadata': {'simulated': True}
            }
            
            # 随机生成一些副作用
            if random.random() < 0.2:
                side_effect = f"调节{target_variable}时轻微影响相关认知维度"
                result['side_effects'].append(side_effect)
            
            return result
            
        except Exception as e:
            logger.warning(f"调节步骤执行失败: {e}")
            return {
                'step_id': step.get('step_id', 'unknown'),
                'action': step.get('action', ''),
                'success': False,
                'error': str(e),
                'changes': {},
                'side_effects': [f"执行失败: {str(e)}"],
                'execution_time': timedelta(0),
                'metadata': {'error': True}
            }
    
    def _evaluate_regulation_effectiveness(self,
                                          regulation_strategy: RegulationStrategy,
                                          state_before: CognitiveState,
                                          state_after: CognitiveState,
                                          actual_changes: Dict[str, Dict[str, float]]) -> float:
        """评估调节效果"""
        if not regulation_strategy.regulation_targets:
            return 0.5
        
        total_effectiveness = 0.0
        target_count = len(regulation_strategy.regulation_targets)
        
        for target in regulation_strategy.regulation_targets:
            variable = target.target_variable
            expected_change = target.target_value - target.current_value
            
            if variable in actual_changes:
                actual_change = actual_changes[variable].get('new', 0) - actual_changes[variable].get('old', 0)
                
                # 计算目标达成度
                if abs(expected_change) > 0.001:  # 避免除零
                    achievement_ratio = actual_change / expected_change
                    achievement_score = min(1.0, abs(achievement_ratio))
                else:
                    achievement_score = 0.5
                
                # 考虑方向正确性
                direction_correct = (actual_change * expected_change) >= 0
                direction_score = 1.0 if direction_correct else 0.3
                
                # 综合评分
                target_effectiveness = (achievement_score + direction_score) / 2
                total_effectiveness += target_effectiveness
            else:
                # 没有实际变化数据
                total_effectiveness += 0.3
        
        average_effectiveness = total_effectiveness / target_count if target_count > 0 else 0.5
        
        # 考虑整体性能改进
        performance_improvement = state_after.performance_score - state_before.performance_score
        performance_bonus = max(0.0, performance_improvement * 0.5)
        
        final_effectiveness = min(1.0, average_effectiveness + performance_bonus)
        return final_effectiveness
    
    def _calculate_regulation_efficiency(self,
                                       regulation_strategy: RegulationStrategy,
                                       execution_duration: timedelta,
                                       effectiveness_score: float) -> float:
        """计算调节效率"""
        if execution_duration.total_seconds() <= 0:
            return 0.5
        
        # 效率 = 效果 / 时间
        time_efficiency = 1.0 / (execution_duration.total_seconds() + 1.0)  # 加1避免除零
        
        # 考虑策略复杂度
        strategy_complexity = len(regulation_strategy.implementation_plan)
        complexity_factor = 1.0 / (strategy_complexity + 1.0)
        
        # 综合效率评分
        efficiency = effectiveness_score * time_efficiency * complexity_factor * 2.0  # 缩放
        
        return max(0.0, min(1.0, efficiency))
    
    def _extract_learning_outcomes(self,
                                  regulation_strategy: RegulationStrategy,
                                  execution_log: List[Dict[str, Any]],
                                  effectiveness_score: float) -> List[str]:
        """提取学习成果"""
        outcomes = []
        
        # 基于效果评分的学习
        if effectiveness_score > 0.8:
            outcomes.append("学习到高效的认知调节策略")
        elif effectiveness_score < 0.4:
            outcomes.append("学习到需要避免的调节方法")
        
        # 基于执行日志的学习
        successful_steps = sum(1 for log in execution_log if log.get('result', {}).get('success', False))
        total_steps = len(execution_log)
        
        if total_steps > 0:
            success_rate = successful_steps / total_steps
            if success_rate > 0.9:
                outcomes.append("学习到可靠的调节步骤执行顺序")
            elif success_rate < 0.5:
                outcomes.append("学习到需要改进的调节步骤设计")
        
        # 基于策略类型的学习
        strategy_type = regulation_strategy.strategy_type
        outcomes.append(f"积累{strategy_type}调节经验")
        
        return outcomes[:3]  # 限制数量
    
    def _generate_recommendations_for_future(self,
                                            regulation_strategy: RegulationStrategy,
                                            effectiveness_score: float,
                                            side_effects: List[str]) -> List[str]:
        """生成未来建议"""
        recommendations = []
        
        # 基于效果的建议
        if effectiveness_score > 0.8:
            recommendations.append("未来可继续使用类似调节策略")
            recommendations.append("可尝试更积极的调节强度")
        elif effectiveness_score < 0.4:
            recommendations.append("未来应调整调节方法")
            recommendations.append("考虑更保守的调节策略")
        
        # 基于副作用建议
        if side_effects:
            if len(side_effects) > 2:
                recommendations.append("未来应减少调节步骤以降低副作用风险")
            recommendations.append("实施前应更详细评估潜在副作用")
        
        # 基于策略复杂度的建议
        plan_length = len(regulation_strategy.implementation_plan)
        if plan_length > 5:
            recommendations.append("简化调节计划，减少步骤数量")
        elif plan_length < 2:
            recommendations.append("增加调节步骤的详细度")
        
        return recommendations[:5]  # 限制数量
    
    def _learn_from_regulation_result(self, regulation_result: RegulationResult):
        """从调节结果中学习"""
        try:
            effectiveness = regulation_result.effectiveness_score
            
            # 更新成功率
            successful_regulations = sum(1 for r in self.regulation_history.regulation_history 
                                       if r.effectiveness_score > 0.6)
            total_regulations = len(self.regulation_history.regulation_history)
            
            if total_regulations > 0:
                self.regulation_history.success_rate = successful_regulations / total_regulations
            
            # 更新平均效果
            total_effectiveness = sum(r.effectiveness_score for r in self.regulation_history.regulation_history)
            self.regulation_history.average_effectiveness = total_effectiveness / total_regulations if total_regulations > 0 else 0.5
            
            # 更新偏好策略
            strategy_type = regulation_result.regulation_strategy.strategy_type
            if strategy_type in self.regulation_history.preferred_strategies:
                self.regulation_history.preferred_strategies[strategy_type] += 1
            else:
                self.regulation_history.preferred_strategies[strategy_type] = 1
            
            # 积累学习成果
            for outcome in regulation_result.learning_outcomes:
                if outcome not in self.regulation_history.learning_accumulated:
                    self.regulation_history.learning_accumulated.append(outcome)
            
            # 限制学习积累数量
            max_learning = 50
            if len(self.regulation_history.learning_accumulated) > max_learning:
                self.regulation_history.learning_accumulated = self.regulation_history.learning_accumulated[-max_learning:]
            
            logger.debug(f"从调节结果 {regulation_result.result_id} 中学习，效果: {effectiveness:.2f}")
            
        except Exception as e:
            logger.warning(f"从调节结果学习失败: {e}")
    
    def _get_state_value(self, state: CognitiveState, variable: str) -> float:
        """获取状态变量的值"""
        try:
            return getattr(state, variable)
        except AttributeError:
            logger.warning(f"状态变量 {variable} 不存在")
            return 0.5
    
    def _set_state_value(self, state: CognitiveState, variable: str, value: float):
        """设置状态变量的值"""
        try:
            setattr(state, variable, value)
        except AttributeError:
            logger.warning(f"无法设置状态变量 {variable}")
    
    def _state_to_dict(self, state: CognitiveState) -> Dict[str, Any]:
        """将认知状态转换为字典"""
        if state is None:
            return {}
        
        return {
            'timestamp': state.timestamp.isoformat() if hasattr(state.timestamp, 'isoformat') else str(state.timestamp),
            'attention_focus': state.attention_focus,
            'working_memory_load': state.working_memory_load,
            'reasoning_accuracy': state.reasoning_accuracy,
            'decision_confidence': state.decision_confidence,
            'emotional_arousal': state.emotional_arousal,
            'cognitive_resource_available': state.cognitive_resource_available,
            'performance_score': state.performance_score,
            'efficiency_score': state.efficiency_score,
            'current_task': state.current_task,
            'task_difficulty': state.task_difficulty,
            'time_pressure': state.time_pressure,
            'importance': state.importance
        }
    
    def _target_to_dict(self, target: RegulationTarget) -> Dict[str, Any]:
        """将调节目标转换为字典"""
        return {
            'target_id': target.target_id,
            'regulation_type': target.regulation_type.value if hasattr(target.regulation_type, 'value') else str(target.regulation_type),
            'goal': target.goal.value if hasattr(target.goal, 'value') else str(target.goal),
            'target_variable': target.target_variable,
            'current_value': target.current_value,
            'target_value': target.target_value,
            'priority': target.priority,
            'time_constraint': str(target.time_constraint) if target.time_constraint else None
        }
    
    def _strategy_to_dict(self, strategy: RegulationStrategy) -> Dict[str, Any]:
        """将调节策略转换为字典"""
        return {
            'strategy_id': strategy.strategy_id,
            'strategy_type': strategy.strategy_type,
            'intensity': strategy.intensity.value if hasattr(strategy.intensity, 'value') else str(strategy.intensity),
            'target_count': len(strategy.regulation_targets),
            'confidence': strategy.confidence,
            'cost_estimate': strategy.cost_estimate,
            'time_estimate': str(strategy.time_estimate),
            'potential_risks': strategy.potential_risks[:3]  # 只显示前3个
        }
    
    def _result_to_dict(self, result: RegulationResult) -> Dict[str, Any]:
        """将调节结果转换为字典"""
        return {
            'result_id': result.result_id,
            'effectiveness_score': result.effectiveness_score,
            'efficiency_score': result.efficiency_score,
            'execution_duration': str(result.execution_duration),
            'side_effects_count': len(result.side_effects),
            'learning_outcomes': result.learning_outcomes[:3]  # 只显示前3个
        }


def create_default_cognitive_regulation_mechanism() -> CognitiveRegulationMechanism:
    """创建默认的认知调节机制实例"""
    return CognitiveRegulationMechanism()


# 全局实例
_cognitive_regulation_mechanism_instance = None

def get_cognitive_regulation_mechanism() -> CognitiveRegulationMechanism:
    """获取全局认知调节机制实例（单例模式）"""
    global _cognitive_regulation_mechanism_instance
    
    if _cognitive_regulation_mechanism_instance is None:
        _cognitive_regulation_mechanism_instance = create_default_cognitive_regulation_mechanism()
    
    return _cognitive_regulation_mechanism_instance


if __name__ == "__main__":
    # 测试认知调节机制
    print("认知调节机制测试")
    print("=" * 60)
    
    crm = get_cognitive_regulation_mechanism()
    
    # 测试完整调节过程
    context = {
        'current_task': '复杂问题解决',
        'task_difficulty': 0.8,
        'time_pressure': 0.6,
        'importance': 0.9
    }
    
    result = crm.regulate_cognitive_process(context)
    
    print(f"调节成功: {result.get('success', False)}")
    print(f"是否需要调节: {result.get('regulation_needed', False)}")
    
    if result.get('regulation_needed', False):
        print(f"调节效果: {result.get('effectiveness', 0.0):.2f}")
        print(f"调节效率: {result.get('efficiency', 0.0):.2f}")
        print(f"调节目标数量: {len(result.get('regulation_targets', []))}")
        
        # 显示前两个目标
        targets = result.get('regulation_targets', [])
        for i, target in enumerate(targets[:2]):
            print(f"目标 {i+1}: {target.get('target_variable')} "
                  f"从 {target.get('current_value', 0):.2f} 到 {target.get('target_value', 0):.2f}")
    
    print("\n" + "=" * 60)
    print("测试完成")