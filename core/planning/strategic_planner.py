#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zlib
"""
战略层规划器 - 处理长期、高层、价值导向的目标规划

功能:
1. 长期目标制定: 基于价值观和使命制定长期战略目标
2. 战略优先级设置: 评估目标价值和风险，设置优先级
3. 资源战略分配: 为长期目标分配战略资源
4. 环境趋势分析: 分析外部环境和内部能力趋势
5. 战略风险评估: 评估长期战略风险
6. 战略调整优化: 基于反馈调整战略方向

核心特性:
- 时间尺度: 月~年级别
- 抽象级别: 高层目标、价值导向、使命驱动
- 决策依据: 价值观、长期趋势、核心竞争力
- 输出形式: 战略目标、资源分配、优先级排序

战略规划流程:
1. 环境扫描: 分析内外部环境
2. 使命确认: 明确核心价值观和使命
3. 目标设定: 制定长期战略目标
4. 策略制定: 制定实现目标的战略
5. 资源配置: 分配战略资源
6. 监控调整: 跟踪执行并调整战略

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import copy
from collections import defaultdict

# 导入相关模块
from .hierarchical_planning_system import (
    HierarchicalPlanningSystem,
    Goal, Task, Action, PlanningContext,
    PlanningLevel, GoalStatus, ActionStatus
)

# 导入世界模型
try:
    from core.world_model.belief_state import BeliefState
    from core.world_model.state_transition_model import StateTransitionModel
    from core.world_model.partial_observability_handler import PartialObservabilityHandler
except ImportError:
    BeliefState = StateTransitionModel = PartialObservabilityHandler = None

# 导入因果推理
try:
    from core.causal.causal_knowledge_graph import CausalKnowledgeGraph
    from core.causal.causal_scm_engine import StructuralCausalModelEngine
except ImportError:
    CausalKnowledgeGraph = StructuralCausalModelEngine = None

# 导入错误处理
try:
    from core.error_handling import error_handler
except ImportError:
    error_handler = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StrategicFocusArea(Enum):
    """战略重点领域"""
    INNOVATION = "innovation"          # 创新与研发
    GROWTH = "growth"                  # 增长与扩张
    EFFICIENCY = "efficiency"          # 效率与优化
    STABILITY = "stability"            # 稳定与可持续
    ADAPTABILITY = "adaptability"      # 适应性与韧性
    COMPETITIVENESS = "competitiveness"  # 竞争力提升
    COLLABORATION = "collaboration"    # 合作与生态
    SOCIAL_IMPACT = "social_impact"    # 社会影响


class StrategicTimeHorizon(Enum):
    """战略时间视野"""
    SHORT_TERM = "short_term"      # 1-12个月
    MEDIUM_TERM = "medium_term"    # 1-3年
    LONG_TERM = "long_term"        # 3-5年
    VERY_LONG_TERM = "very_long_term"  # 5+年


class StrategicRiskLevel(Enum):
    """战略风险级别"""
    LOW = "low"         # 低风险：可预测，影响有限
    MEDIUM = "medium"   # 中等风险：部分可预测，中等影响
    HIGH = "high"       # 高风险：不确定性高，潜在重大影响
    VERY_HIGH = "very_high"  # 极高风险：高度不确定，可能颠覆性影响


@dataclass
class StrategicContext:
    """战略上下文数据类"""
    mission_statement: str  # 使命陈述
    core_values: List[str]  # 核心价值观
    vision_statement: str   # 愿景陈述
    strategic_focus_areas: List[StrategicFocusArea]  # 战略重点领域
    time_horizon: StrategicTimeHorizon  # 战略时间视野
    external_environment: Dict[str, Any]  # 外部环境分析
    internal_capabilities: Dict[str, Any]  # 内部能力分析
    competitive_landscape: Dict[str, Any]  # 竞争格局分析
    historical_performance: Dict[str, Any]  # 历史绩效数据
    risk_appetite: float  # 风险偏好（0-1）
    resource_constraints: Dict[str, float]  # 资源约束
    stakeholder_expectations: List[str]  # 利益相关者期望
    regulatory_constraints: List[str]  # 法规约束
    ethical_guidelines: List[str]  # 伦理指导原则
    
    def __post_init__(self):
        """后初始化验证"""
        self.risk_appetite = max(0.0, min(1.0, self.risk_appetite))
        if not self.mission_statement:
            self.mission_statement = "未定义使命"
        if not self.vision_statement:
            self.vision_statement = "未定义愿景"


@dataclass
class StrategicAnalysis:
    """战略分析结果数据类"""
    swot_analysis: Dict[str, List[str]]  # SWOT分析
    pestel_analysis: Dict[str, List[str]]  # PESTEL分析
    five_forces_analysis: Dict[str, Any]  # 五力模型分析
    core_competencies: List[str]  # 核心竞争力
    strategic_gaps: List[str]  # 战略缺口
    opportunity_areas: List[Dict[str, Any]]  # 机会领域
    threat_areas: List[Dict[str, Any]]  # 威胁领域
    trend_analysis: Dict[str, Any]  # 趋势分析
    scenario_analysis: Dict[str, Any]  # 情景分析


@dataclass
class StrategicObjective:
    """战略目标数据类"""
    id: str
    description: str
    strategic_focus_area: StrategicFocusArea
    time_horizon: StrategicTimeHorizon
    alignment_score: float  # 与使命/愿景的匹配度（0-1）
    value_score: float  # 价值评分（0-1）
    feasibility_score: float  # 可行性评分（0-1）
    risk_level: StrategicRiskLevel
    priority: float  # 优先级（0-1）
    resource_requirements: Dict[str, float]  # 资源需求
    key_performance_indicators: Dict[str, Any]  # 关键绩效指标
    success_criteria: List[str]  # 成功标准
    dependencies: List[str]  # 依赖关系
    assumptions: List[str]  # 假设条件
    constraints: List[Dict[str, Any]]  # 约束条件
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.alignment_score = max(0.0, min(1.0, self.alignment_score))
        self.value_score = max(0.0, min(1.0, self.value_score))
        self.feasibility_score = max(0.0, min(1.0, self.feasibility_score))
        self.priority = max(0.0, min(1.0, self.priority))
        
        if not self.id:
            self.id = f"strategic_objective_{int(time.time())}_{(zlib.adler32(str(self.description).encode('utf-8')) & 0xffffffff) % 10000}"


@dataclass
class StrategicPlan:
    """战略计划数据类"""
    id: str
    name: str
    description: str
    time_period: Tuple[datetime, datetime]  # 时间周期（开始，结束）
    strategic_context: StrategicContext
    strategic_analysis: StrategicAnalysis
    strategic_objectives: List[StrategicObjective]
    resource_allocation: Dict[str, float]  # 资源分配
    risk_mitigation_strategies: List[Dict[str, Any]]  # 风险缓解策略
    monitoring_framework: Dict[str, Any]  # 监控框架
    contingency_plans: List[Dict[str, Any]]  # 应急计划
    implementation_roadmap: Dict[str, Any]  # 实施路线图
    success_metrics: Dict[str, Any]  # 成功度量指标
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        if not self.id:
            self.id = f"strategic_plan_{int(time.time())}_{(zlib.adler32(str(self.name).encode('utf-8')) & 0xffffffff) % 10000}"
        
        # 确保结束时间晚于开始时间
        start_time, end_time = self.time_period
        if end_time <= start_time:
            # 默认设置为1年周期
            self.time_period = (start_time, start_time + timedelta(days=365))


class StrategicPlanner:
    """
    战略层规划器 - 处理长期战略规划
    
    核心功能:
    1. 环境分析: 分析内外环境，识别机会和威胁
    2. 使命确认: 明确组织使命、愿景和价值观
    3. 目标制定: 基于分析制定长期战略目标
    4. 策略制定: 制定实现目标的战略策略
    5. 资源配置: 为战略目标分配资源
    6. 风险评估: 评估战略风险并制定缓解措施
    7. 计划制定: 制定完整的战略计划
    8. 监控调整: 跟踪执行并调整战略
    
    技术特点:
    - 长期视野: 考虑月到年的时间跨度
    - 价值导向: 基于核心价值观和使命
    - 系统性分析: 使用SWOT、PESTEL等分析工具
    - 风险感知: 主动识别和缓解战略风险
    - 适应性规划: 基于环境变化动态调整
    - 因果推理: 使用因果模型分析战略影响
    """
    
    def __init__(self, 
                 planning_system: Optional[HierarchicalPlanningSystem] = None,
                 strategic_config: Optional[Dict[str, Any]] = None):
        """
        初始化战略规划器
        
        Args:
            planning_system: 分层规划系统实例（可选）
            strategic_config: 战略规划配置（可选）
        """
        self.planning_system = planning_system
        self.config = strategic_config or self._get_default_config()
        
        # 战略规划组件
        self.strategic_context: Optional[StrategicContext] = None
        self.current_strategic_plan: Optional[StrategicPlan] = None
        self.strategic_plans_history: List[StrategicPlan] = []
        self.max_history_size = self.config.get('max_strategic_plans_history', 10)
        
        # 世界模型集成
        self.world_model_integrated = False
        if BeliefState is not None:
            self.world_model_integrated = True
        
        # 因果推理集成
        self.causal_reasoning_integrated = False
        if CausalKnowledgeGraph is not None and StructuralCausalModelEngine is not None:
            self.causal_reasoning_integrated = True
        
        # 分析工具
        self.analysis_tools = self._initialize_analysis_tools()
        
        # 战略模板库
        self.strategic_templates = self._initialize_strategic_templates()
        
        # 性能统计
        self.performance_stats = {
            "strategic_analyses_performed": 0,
            "strategic_plans_created": 0,
            "strategic_objectives_generated": 0,
            "total_planning_time": 0.0,
            "average_planning_time": 0.0,
            "last_planning_time": None
        }
        
        logger.info(f"战略规划器初始化完成，世界模型集成: {self.world_model_integrated}, 因果推理集成: {self.causal_reasoning_integrated}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "enable_environmental_analysis": True,
            "enable_swot_analysis": True,
            "enable_pestel_analysis": True,
            "enable_scenario_analysis": True,
            "enable_causal_strategic_analysis": True,
            "max_strategic_objectives": 10,
            "min_strategic_alignment_score": 0.7,
            "strategic_planning_time_limit": 300.0,  # 5分钟
            "max_strategic_plans_history": 10,
            "enable_strategic_risk_assessment": True,
            "enable_resource_allocation_optimization": True,
            "enable_strategic_monitoring": True,
            "strategic_plan_validity_period_days": 365,  # 1年
            "strategic_review_frequency_days": 90,  # 每季度审查
            "enable_adaptive_strategic_planning": True
        }
    
    def _initialize_analysis_tools(self) -> Dict[str, Any]:
        """初始化分析工具"""
        return {
            "swot_framework": {
                "strengths": "内部优势",
                "weaknesses": "内部劣势",
                "opportunities": "外部机会",
                "threats": "外部威胁"
            },
            "pestel_framework": {
                "political": "政治因素",
                "economic": "经济因素",
                "social": "社会因素",
                "technological": "技术因素",
                "environmental": "环境因素",
                "legal": "法律因素"
            },
            "five_forces_framework": {
                "competitive_rivalry": "行业竞争强度",
                "supplier_power": "供应商议价能力",
                "buyer_power": "买家议价能力",
                "threat_of_substitutes": "替代品威胁",
                "threat_of_new_entrants": "新进入者威胁"
            },
            "scenario_types": {
                "baseline": "基线情景",
                "optimistic": "乐观情景",
                "pessimistic": "悲观情景",
                "disruptive": "颠覆性情景"
            },
            "risk_assessment_framework": {
                "likelihood_levels": ["很低", "低", "中等", "高", "很高"],
                "impact_levels": ["可忽略", "轻微", "中等", "重大", "灾难性"],
                "risk_matrix": {
                    "很低": {"可忽略": "低", "轻微": "低", "中等": "低", "重大": "中", "灾难性": "中"},
                    "低": {"可忽略": "低", "轻微": "低", "中等": "中", "重大": "中", "灾难性": "高"},
                    "中等": {"可忽略": "低", "轻微": "中", "中等": "中", "重大": "高", "灾难性": "高"},
                    "高": {"可忽略": "中", "轻微": "中", "中等": "高", "重大": "高", "灾难性": "很高"},
                    "很高": {"可忽略": "中", "轻微": "高", "中等": "高", "重大": "很高", "灾难性": "很高"}
                }
            }
        }
    
    def _initialize_strategic_templates(self) -> Dict[str, Any]:
        """初始化战略模板库"""
        return {
            "growth_strategy": {
                "name": "增长战略",
                "description": "专注于业务增长和市场扩张",
                "focus_areas": [StrategicFocusArea.GROWTH, StrategicFocusArea.INNOVATION],
                "typical_objectives": [
                    "扩大市场份额",
                    "进入新市场",
                    "开发新产品/服务",
                    "增加收入来源"
                ],
                "risk_profile": "中等至高",
                "resource_intensity": "高",
                "time_horizon": StrategicTimeHorizon.MEDIUM_TERM
            },
            "efficiency_strategy": {
                "name": "效率战略",
                "description": "专注于优化运营效率和降低成本",
                "focus_areas": [StrategicFocusArea.EFFICIENCY, StrategicFocusArea.STABILITY],
                "typical_objectives": [
                    "降低运营成本",
                    "提高生产效率",
                    "优化资源配置",
                    "改进工作流程"
                ],
                "risk_profile": "低至中等",
                "resource_intensity": "中等",
                "time_horizon": StrategicTimeHorizon.SHORT_TERM
            },
            "innovation_strategy": {
                "name": "创新战略",
                "description": "专注于技术创新和研发",
                "focus_areas": [StrategicFocusArea.INNOVATION, StrategicFocusArea.COMPETITIVENESS],
                "typical_objectives": [
                    "开发突破性技术",
                    "建立研发能力",
                    "申请专利",
                    "建立创新文化"
                ],
                "risk_profile": "高",
                "resource_intensity": "高",
                "time_horizon": StrategicTimeHorizon.LONG_TERM
            },
            "sustainability_strategy": {
                "name": "可持续性战略",
                "description": "专注于长期可持续性和社会责任",
                "focus_areas": [StrategicFocusArea.STABILITY, StrategicFocusArea.SOCIAL_IMPACT],
                "typical_objectives": [
                    "减少环境足迹",
                    "增强社会影响力",
                    "建立可持续供应链",
                    "促进员工福祉"
                ],
                "risk_profile": "低至中等",
                "resource_intensity": "中等",
                "time_horizon": StrategicTimeHorizon.LONG_TERM
            },
            "adaptability_strategy": {
                "name": "适应性战略",
                "description": "专注于增强组织的适应性和韧性",
                "focus_areas": [StrategicFocusArea.ADAPTABILITY, StrategicFocusArea.STABILITY],
                "typical_objectives": [
                    "增强危机应对能力",
                    "建立敏捷组织",
                    "多样化业务组合",
                    "培养学习型文化"
                ],
                "risk_profile": "中等",
                "resource_intensity": "中等",
                "time_horizon": StrategicTimeHorizon.MEDIUM_TERM
            }
        }
    
    def set_strategic_context(self, context: StrategicContext):
        """
        设置战略上下文
        
        Args:
            context: 战略上下文
        """
        self.strategic_context = context
        logger.info(f"战略上下文已设置: {context.mission_statement[:50]}...")
    
    def analyze_strategic_environment(self, 
                                     context: Optional[StrategicContext] = None) -> StrategicAnalysis:
        """
        分析战略环境
        
        Args:
            context: 战略上下文（如果未提供，使用已设置的上下文）
            
        Returns:
            战略分析结果
        """
        start_time = time.time()
        
        if context is None:
            context = self.strategic_context
        
        if context is None:
            raise ValueError("未设置战略上下文")
        
        logger.info("开始战略环境分析...")
        
        try:
            # 执行SWOT分析
            swot_analysis = self._perform_swot_analysis(context)
            
            # 执行PESTEL分析
            pestel_analysis = self._perform_pestel_analysis(context)
            
            # 执行五力模型分析
            five_forces_analysis = self._perform_five_forces_analysis(context)
            
            # 识别核心竞争力
            core_competencies = self._identify_core_competencies(context)
            
            # 识别战略缺口
            strategic_gaps = self._identify_strategic_gaps(context, swot_analysis)
            
            # 识别机会领域
            opportunity_areas = self._identify_opportunity_areas(context, swot_analysis, pestel_analysis)
            
            # 识别威胁领域
            threat_areas = self._identify_threat_areas(context, swot_analysis, pestel_analysis)
            
            # 趋势分析
            trend_analysis = self._analyze_trends(context)
            
            # 情景分析
            scenario_analysis = self._perform_scenario_analysis(context, trend_analysis)
            
            # 构建战略分析结果
            strategic_analysis = StrategicAnalysis(
                swot_analysis=swot_analysis,
                pestel_analysis=pestel_analysis,
                five_forces_analysis=five_forces_analysis,
                core_competencies=core_competencies,
                strategic_gaps=strategic_gaps,
                opportunity_areas=opportunity_areas,
                threat_areas=threat_areas,
                trend_analysis=trend_analysis,
                scenario_analysis=scenario_analysis
            )
            
            # 更新性能统计
            analysis_time = time.time() - start_time
            self._update_performance_stats("strategic_analyses_performed", analysis_time)
            
            logger.info(f"战略环境分析完成，用时: {analysis_time:.2f} 秒")
            
            return strategic_analysis
            
        except Exception as e:
            logger.error(f"战略环境分析失败: {e}")
            if error_handler:
                error_handler.handle_error(e, "StrategicPlanner", "战略环境分析失败")
            
            # 返回基本的分析结果
            return StrategicAnalysis(
                swot_analysis={},
                pestel_analysis={},
                five_forces_analysis={},
                core_competencies=[],
                strategic_gaps=[],
                opportunity_areas=[],
                threat_areas=[],
                trend_analysis={},
                scenario_analysis={}
            )
    
    def _perform_swot_analysis(self, context: StrategicContext) -> Dict[str, List[str]]:
        """执行SWOT分析"""
        swot_results = {
            "strengths": [],
            "weaknesses": [],
            "opportunities": [],
            "threats": []
        }
        
        # 分析内部优势
        if context.internal_capabilities:
            # 基于内部能力识别优势
            capabilities = context.internal_capabilities
            
            # 技术优势
            if capabilities.get("technology_advanced", False):
                swot_results["strengths"].append("先进的技术能力")
            
            # 人才优势
            if capabilities.get("talented_team", False):
                swot_results["strengths"].append("优秀的人才团队")
            
            # 财务优势
            if capabilities.get("strong_financials", False):
                swot_results["strengths"].append("稳健的财务状况")
            
            # 品牌优势
            if capabilities.get("strong_brand", False):
                swot_results["strengths"].append("强大的品牌影响力")
            
            # 运营优势
            if capabilities.get("efficient_operations", False):
                swot_results["strengths"].append("高效的运营体系")
        
        # 分析内部劣势
        # 基于战略缺口识别劣势
        if hasattr(context, 'historical_performance') and context.historical_performance:
            perf = context.historical_performance
            
            # 技术劣势
            if perf.get("technology_gap", False):
                swot_results["weaknesses"].append("技术能力有待提升")
            
            # 市场劣势
            if perf.get("market_share_declining", False):
                swot_results["weaknesses"].append("市场份额下降")
            
            # 运营劣势
            if perf.get("operational_inefficiencies", False):
                swot_results["weaknesses"].append("运营效率不高")
        
        # 分析外部机会
        if context.external_environment:
            env = context.external_environment
            
            # 市场机会
            if env.get("market_growing", False):
                swot_results["opportunities"].append("市场快速增长")
            
            # 技术机会
            if env.get("emerging_technologies", False):
                swot_results["opportunities"].append("新兴技术出现")
            
            # 政策机会
            if env.get("favorable_policies", False):
                swot_results["opportunities"].append("有利的政策环境")
            
            # 合作机会
            if env.get("collaboration_opportunities", False):
                swot_results["opportunities"].append("合作机会增多")
        
        # 分析外部威胁
        if context.competitive_landscape:
            comp = context.competitive_landscape
            
            # 竞争威胁
            if comp.get("intense_competition", False):
                swot_results["threats"].append("竞争激烈")
            
            # 市场威胁
            if comp.get("market_saturation", False):
                swot_results["threats"].append("市场饱和")
            
            # 技术威胁
            if comp.get("technology_disruption", False):
                swot_results["threats"].append("技术颠覆风险")
            
            # 监管威胁
            if comp.get("regulatory_changes", False):
                swot_results["threats"].append("监管政策变化")
        
        return swot_results
    
    def _perform_pestel_analysis(self, context: StrategicContext) -> Dict[str, List[str]]:
        """执行PESTEL分析"""
        pestel_results = {
            "political": [],
            "economic": [],
            "social": [],
            "technological": [],
            "environmental": [],
            "legal": []
        }
        
        if context.external_environment:
            env = context.external_environment
            
            # 政治因素
            if env.get("government_stability", False):
                pestel_results["political"].append("政府稳定")
            if env.get("policy_support", False):
                pestel_results["political"].append("政策支持")
            
            # 经济因素
            if env.get("economic_growth", False):
                pestel_results["economic"].append("经济增长")
            if env.get("inflation_rate", False):
                inflation = env.get("inflation_rate", 0)
                if inflation > 5:
                    pestel_results["economic"].append(f"高通胀率: {inflation}%")
            
            # 社会因素
            if env.get("demographic_trends", False):
                pestel_results["social"].append("有利的人口趋势")
            if env.get("cultural_acceptance", False):
                pestel_results["social"].append("文化接受度高")
            
            # 技术因素
            if env.get("technology_advancement", False):
                pestel_results["technological"].append("技术进步")
            if env.get("digital_transformation", False):
                pestel_results["technological"].append("数字化转型")
            
            # 环境因素
            if env.get("environmental_regulations", False):
                pestel_results["environmental"].append("环保法规")
            if env.get("sustainability_awareness", False):
                pestel_results["environmental"].append("可持续发展意识增强")
            
            # 法律因素
            if env.get("legal_framework", False):
                pestel_results["legal"].append("完善的法律框架")
            if env.get("compliance_requirements", False):
                pestel_results["legal"].append("合规要求")
        
        return pestel_results
    
    def _perform_five_forces_analysis(self, context: StrategicContext) -> Dict[str, Any]:
        """执行五力模型分析"""
        five_forces_results = {
            "competitive_rivalry": {"level": "中等", "description": "行业竞争强度"},
            "supplier_power": {"level": "中等", "description": "供应商议价能力"},
            "buyer_power": {"level": "中等", "description": "买家议价能力"},
            "threat_of_substitutes": {"level": "中等", "description": "替代品威胁"},
            "threat_of_new_entrants": {"level": "中等", "description": "新进入者威胁"}
        }
        
        if context.competitive_landscape:
            comp = context.competitive_landscape
            
            # 行业竞争强度
            if comp.get("competitor_count", 0) > 10:
                five_forces_results["competitive_rivalry"]["level"] = "高"
            elif comp.get("competitor_count", 0) > 5:
                five_forces_results["competitive_rivalry"]["level"] = "中等"
            else:
                five_forces_results["competitive_rivalry"]["level"] = "低"
            
            # 供应商议价能力
            if comp.get("supplier_concentration", False):
                five_forces_results["supplier_power"]["level"] = "高"
            
            # 买家议价能力
            if comp.get("buyer_concentration", False):
                five_forces_results["buyer_power"]["level"] = "高"
            
            # 替代品威胁
            if comp.get("substitute_availability", False):
                five_forces_results["threat_of_substitutes"]["level"] = "高"
            
            # 新进入者威胁
            if comp.get("low_barriers_to_entry", False):
                five_forces_results["threat_of_new_entrants"]["level"] = "高"
            elif comp.get("high_capital_requirements", False):
                five_forces_results["threat_of_new_entrants"]["level"] = "低"
        
        return five_forces_results
    
    def _identify_core_competencies(self, context: StrategicContext) -> List[str]:
        """识别核心竞争力"""
        competencies = []
        
        if context.internal_capabilities:
            capabilities = context.internal_capabilities
            
            # 技术能力
            if capabilities.get("technology_leadership", False):
                competencies.append("技术领先优势")
            if capabilities.get("research_development", False):
                competencies.append("研发能力")
            
            # 运营能力
            if capabilities.get("operational_excellence", False):
                competencies.append("卓越运营")
            if capabilities.get("supply_chain_management", False):
                competencies.append("供应链管理")
            
            # 市场能力
            if capabilities.get("brand_equity", False):
                competencies.append("品牌资产")
            if capabilities.get("customer_relationships", False):
                competencies.append("客户关系管理")
            
            # 组织能力
            if capabilities.get("talent_development", False):
                competencies.append("人才发展")
            if capabilities.get("innovation_culture", False):
                competencies.append("创新文化")
        
        # 如果没有识别到核心竞争力，使用默认值
        if not competencies:
            competencies = ["学习能力", "适应性", "问题解决能力"]
        
        return competencies
    
    def _identify_strategic_gaps(self, context: StrategicContext, swot_analysis: Dict[str, List[str]]) -> List[str]:
        """识别战略缺口"""
        gaps = []
        
        # 基于劣势识别缺口
        weaknesses = swot_analysis.get("weaknesses", [])
        for weakness in weaknesses:
            gaps.append(f"改善{weakness}")
        
        # 基于威胁识别缺口
        threats = swot_analysis.get("threats", [])
        for threat in threats:
            gaps.append(f"应对{threat}")
        
        # 基于历史绩效识别缺口
        if hasattr(context, 'historical_performance') and context.historical_performance:
            perf = context.historical_performance
            
            if perf.get("performance_gaps", []):
                for gap in perf["performance_gaps"]:
                    gaps.append(f"弥补{gap}")
        
        # 去除重复项
        gaps = list(set(gaps))
        
        return gaps
    
    def _identify_opportunity_areas(self, context: StrategicContext, 
                                  swot_analysis: Dict[str, List[str]],
                                  pestel_analysis: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """识别机会领域"""
        opportunities = []
        
        # 基于SWOT机会
        swot_opportunities = swot_analysis.get("opportunities", [])
        for opp in swot_opportunities:
            opportunities.append({
                "name": opp,
                "type": "market_opportunity",
                "priority": "medium",
                "potential_impact": "high" if "快速" in opp or "有利" in opp else "medium"
            })
        
        # 基于PESTEL分析的机会
        for category, items in pestel_analysis.items():
            for item in items:
                if "增长" in item or "有利" in item or "支持" in item:
                    opportunities.append({
                        "name": f"{category}: {item}",
                        "type": f"{category}_opportunity",
                        "priority": "high" if "增长" in item else "medium",
                        "potential_impact": "high"
                    })
        
        # 基于核心竞争力的机会
        core_competencies = self._identify_core_competencies(context)
        for competency in core_competencies:
            opportunities.append({
                "name": f"利用{competency}",
                "type": "capability_based_opportunity",
                "priority": "high",
                "potential_impact": "high"
            })
        
        return opportunities
    
    def _identify_threat_areas(self, context: StrategicContext,
                              swot_analysis: Dict[str, List[str]],
                              pestel_analysis: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """识别威胁领域"""
        threats = []
        
        # 基于SWOT威胁
        swot_threats = swot_analysis.get("threats", [])
        for threat in swot_threats:
            threats.append({
                "name": threat,
                "type": "competitive_threat",
                "severity": "high" if "颠覆" in threat or "激烈" in threat else "medium",
                "likelihood": "high"
            })
        
        # 基于PESTEL分析的威胁
        for category, items in pestel_analysis.items():
            for item in items:
                if "风险" in item or "威胁" in item or "变化" in item:
                    threats.append({
                        "name": f"{category}: {item}",
                        "type": f"{category}_threat",
                        "severity": "medium",
                        "likelihood": "medium"
                    })
        
        # 基于竞争格局的威胁
        if context.competitive_landscape:
            comp = context.competitive_landscape
            if comp.get("new_competitors", False):
                threats.append({
                    "name": "新竞争对手出现",
                    "type": "competitive_threat",
                    "severity": "high",
                    "likelihood": "medium"
                })
        
        return threats
    
    def _analyze_trends(self, context: StrategicContext) -> Dict[str, Any]:
        """分析趋势"""
        trends = {
            "market_trends": [],
            "technology_trends": [],
            "societal_trends": [],
            "regulatory_trends": [],
            "economic_trends": []
        }
        
        if context.external_environment:
            env = context.external_environment
            
            # 市场趋势
            if env.get("market_growing", False):
                trends["market_trends"].append("市场持续增长")
            if env.get("customer_preferences_changing", False):
                trends["market_trends"].append("客户偏好变化")
            
            # 技术趋势
            if env.get("ai_advancement", False):
                trends["technology_trends"].append("人工智能技术快速发展")
            if env.get("digitalization", False):
                trends["technology_trends"].append("数字化转型加速")
            
            # 社会趋势
            if env.get("remote_work_trend", False):
                trends["societal_trends"].append("远程办公普及")
            if env.get("sustainability_focus", False):
                trends["societal_trends"].append("可持续发展关注度提高")
            
            # 法规趋势
            if env.get("privacy_regulations", False):
                trends["regulatory_trends"].append("数据隐私法规加强")
            if env.get("esg_reporting", False):
                trends["regulatory_trends"].append("ESG报告要求增加")
            
            # 经济趋势
            if env.get("global_economic_uncertainty", False):
                trends["economic_trends"].append("全球经济不确定性增加")
            if env.get("interest_rate_changes", False):
                trends["economic_trends"].append("利率变化")
        
        return trends
    
    def _perform_scenario_analysis(self, context: StrategicContext, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """执行情景分析"""
        scenarios = {
            "baseline": {
                "description": "基线情景 - 当前趋势延续",
                "probability": 0.5,
                "impact": "medium",
                "key_drivers": trend_analysis.get("market_trends", [])[:3]
            },
            "optimistic": {
                "description": "乐观情景 - 有利条件出现",
                "probability": 0.3,
                "impact": "high",
                "key_drivers": ["市场快速增长", "技术进步加速", "政策支持增强"]
            },
            "pessimistic": {
                "description": "悲观情景 - 不利条件出现",
                "probability": 0.15,
                "impact": "high",
                "key_drivers": ["经济衰退", "竞争加剧", "监管收紧"]
            },
            "disruptive": {
                "description": "颠覆性情景 - 重大技术或市场变革",
                "probability": 0.05,
                "impact": "very_high",
                "key_drivers": ["技术颠覆", "新商业模式", "市场重构"]
            }
        }
        
        # 根据风险偏好调整概率
        if context.risk_appetite < 0.3:  # 风险厌恶
            scenarios["baseline"]["probability"] = 0.6
            scenarios["optimistic"]["probability"] = 0.2
            scenarios["pessimistic"]["probability"] = 0.15
            scenarios["disruptive"]["probability"] = 0.05
        elif context.risk_appetite > 0.7:  # 风险偏好
            scenarios["baseline"]["probability"] = 0.4
            scenarios["optimistic"]["probability"] = 0.4
            scenarios["pessimistic"]["probability"] = 0.15
            scenarios["disruptive"]["probability"] = 0.05
        
        return scenarios
    
    def _update_performance_stats(self, stat_type: str, execution_time: float):
        """更新性能统计"""
        if stat_type == "strategic_analyses_performed":
            self.performance_stats["strategic_analyses_performed"] += 1
        elif stat_type == "strategic_plans_created":
            self.performance_stats["strategic_plans_created"] += 1
        elif stat_type == "strategic_objectives_generated":
            self.performance_stats["strategic_objectives_generated"] += 1
        
        # 更新时间统计
        self.performance_stats["total_planning_time"] += execution_time
        self.performance_stats["last_planning_time"] = execution_time
        
        # 计算平均时间
        total_actions = (self.performance_stats["strategic_analyses_performed"] +
                        self.performance_stats["strategic_plans_created"] +
                        self.performance_stats["strategic_objectives_generated"])
        
        if total_actions > 0:
            self.performance_stats["average_planning_time"] = (
                self.performance_stats["total_planning_time"] / total_actions
            )
    
    def generate_strategic_objectives(self, 
                                     strategic_analysis: StrategicAnalysis,
                                     context: StrategicContext) -> List[StrategicObjective]:
        """
        生成战略目标
        
        Args:
            strategic_analysis: 战略分析结果
            context: 战略上下文
            
        Returns:
            战略目标列表
        """
        start_time = time.time()
        
        logger.info("开始生成战略目标...")
        
        try:
            objectives = []
            
            # 基于机会领域生成目标
            for opportunity in strategic_analysis.opportunity_areas:
                objective = self._create_objective_from_opportunity(opportunity, context)
                if objective:
                    objectives.append(objective)
            
            # 基于战略缺口生成目标
            for gap in strategic_analysis.strategic_gaps:
                objective = self._create_objective_from_gap(gap, context)
                if objective:
                    objectives.append(objective)
            
            # 基于核心竞争力生成目标
            for competency in strategic_analysis.core_competencies:
                objective = self._create_objective_from_competency(competency, context)
                if objective:
                    objectives.append(objective)
            
            # 基于战略模板生成目标
            for template_name, template in self.strategic_templates.items():
                if any(focus_area in context.strategic_focus_areas 
                      for focus_area in template["focus_areas"]):
                    template_objectives = self._create_objectives_from_template(template, context)
                    objectives.extend(template_objectives)
            
            # 限制目标数量
            max_objectives = self.config.get("max_strategic_objectives", 10)
            if len(objectives) > max_objectives:
                # 根据优先级排序并选择前N个
                objectives.sort(key=lambda x: x.priority, reverse=True)
                objectives = objectives[:max_objectives]
            
            # 更新性能统计
            execution_time = time.time() - start_time
            self._update_performance_stats("strategic_objectives_generated", execution_time)
            
            logger.info(f"生成 {len(objectives)} 个战略目标，用时: {execution_time:.2f} 秒")
            
            return objectives
            
        except Exception as e:
            logger.error(f"生成战略目标失败: {e}")
            if error_handler:
                error_handler.handle_error(e, "StrategicPlanner", "生成战略目标失败")
            return []
    
    def _create_objective_from_opportunity(self, opportunity: Dict[str, Any], 
                                          context: StrategicContext) -> Optional[StrategicObjective]:
        """从机会领域创建战略目标"""
        try:
            # 评估机会的价值和可行性
            value_score = 0.8 if opportunity.get("potential_impact") == "high" else 0.5
            feasibility_score = 0.7 if opportunity.get("priority") == "high" else 0.5
            
            # 计算与使命的匹配度
            alignment_score = self._calculate_alignment_score(opportunity["name"], context)
            
            # 评估风险级别
            risk_level = self._assess_opportunity_risk(opportunity, context)
            
            # 计算优先级
            priority = (value_score * 0.4 + feasibility_score * 0.3 + 
                       alignment_score * 0.3)
            
            # 估算资源需求
            resource_requirements = self._estimate_resource_requirements(opportunity, context)
            
            # 创建战略目标
            objective = StrategicObjective(
                id=f"opportunity_{int(time.time())}_{(zlib.adler32(str(opportunity['name']).encode('utf-8')) & 0xffffffff) % 10000}",
                description=f"把握机会: {opportunity['name']}",
                strategic_focus_area=self._determine_focus_area(opportunity),
                time_horizon=self._determine_time_horizon(opportunity),
                alignment_score=alignment_score,
                value_score=value_score,
                feasibility_score=feasibility_score,
                risk_level=risk_level,
                priority=priority,
                resource_requirements=resource_requirements,
                key_performance_indicators=self._create_kpis_for_opportunity(opportunity),
                success_criteria=[f"成功实现{opportunity['name']}"],
                dependencies=[],
                assumptions=["外部环境保持稳定", "资源供应充足"],
                constraints=context.constraints if hasattr(context, 'constraints') else []
            )
            
            return objective
            
        except Exception as e:
            logger.error(f"从机会创建目标失败: {e}")
            return None
    
    def _create_objective_from_gap(self, gap: str, context: StrategicContext) -> Optional[StrategicObjective]:
        """从战略缺口创建战略目标"""
        try:
            # 评估缺口的重要性和紧迫性
            importance_score = 0.7 if "改善" in gap or "应对" in gap else 0.5
            urgency_score = 0.8 if "威胁" in gap else 0.5
            
            # 计算与使命的匹配度
            alignment_score = self._calculate_alignment_score(gap, context)
            
            # 评估风险级别（弥补缺口通常风险较低）
            risk_level = StrategicRiskLevel.LOW
            
            # 计算优先级
            priority = (importance_score * 0.5 + urgency_score * 0.3 + 
                       alignment_score * 0.2)
            
            # 估算资源需求
            resource_requirements = {
                "time": 3.0,  # 月
                "budget": 5.0,  # 相对预算单位
                "personnel": 2.0  # 人员需求
            }
            
            # 创建战略目标
            objective = StrategicObjective(
                id=f"gap_{int(time.time())}_{(zlib.adler32(str(gap).encode('utf-8')) & 0xffffffff) % 10000}",
                description=f"弥补缺口: {gap}",
                strategic_focus_area=StrategicFocusArea.STABILITY,
                time_horizon=StrategicTimeHorizon.SHORT_TERM,
                alignment_score=alignment_score,
                value_score=importance_score,
                feasibility_score=0.8,  # 弥补缺口通常可行
                risk_level=risk_level,
                priority=priority,
                resource_requirements=resource_requirements,
                key_performance_indicators={
                    "gap_reduction": "缺口减少百分比",
                    "timeline_adherence": "时间表遵守率"
                },
                success_criteria=[f"成功{gap}"],
                dependencies=[],
                assumptions=["组织支持", "资源可用"],
                constraints=context.constraints if hasattr(context, 'constraints') else []
            )
            
            return objective
            
        except Exception as e:
            logger.error(f"从缺口创建目标失败: {e}")
            return None
    
    def _create_objective_from_competency(self, competency: str, 
                                         context: StrategicContext) -> Optional[StrategicObjective]:
        """从核心竞争力创建战略目标"""
        try:
            # 评估能力的战略价值
            strategic_value = 0.8 if "领先" in competency or "卓越" in competency else 0.6
            
            # 计算与使命的匹配度
            alignment_score = self._calculate_alignment_score(competency, context)
            
            # 评估风险级别（基于能力的目标风险中等）
            risk_level = StrategicRiskLevel.MEDIUM
            
            # 计算优先级
            priority = (strategic_value * 0.6 + alignment_score * 0.4)
            
            # 估算资源需求
            resource_requirements = {
                "time": 6.0,  # 月
                "budget": 8.0,  # 相对预算单位
                "personnel": 3.0  # 人员需求
            }
            
            # 创建战略目标
            objective = StrategicObjective(
                id=f"competency_{int(time.time())}_{(zlib.adler32(str(competency).encode('utf-8')) & 0xffffffff) % 10000}",
                description=f"增强核心竞争力: {competency}",
                strategic_focus_area=StrategicFocusArea.COMPETITIVENESS,
                time_horizon=StrategicTimeHorizon.MEDIUM_TERM,
                alignment_score=alignment_score,
                value_score=strategic_value,
                feasibility_score=0.7,
                risk_level=risk_level,
                priority=priority,
                resource_requirements=resource_requirements,
                key_performance_indicators={
                    "competency_level": "能力水平提升",
                    "competitive_advantage": "竞争优势增强"
                },
                success_criteria=[f"显著提升{competency}"],
                dependencies=[],
                assumptions=["持续学习", "组织投入"],
                constraints=context.constraints if hasattr(context, 'constraints') else []
            )
            
            return objective
            
        except Exception as e:
            logger.error(f"从能力创建目标失败: {e}")
            return None
    
    def _create_objectives_from_template(self, template: Dict[str, Any],
                                        context: StrategicContext) -> List[StrategicObjective]:
        """从战略模板创建目标"""
        objectives = []
        
        typical_objectives = template.get("typical_objectives", [])
        for obj_desc in typical_objectives:
            try:
                # 评估目标的价值和可行性
                value_score = 0.7 if "增长" in obj_desc or "创新" in obj_desc else 0.5
                feasibility_score = 0.6
                
                # 计算与使命的匹配度
                alignment_score = self._calculate_alignment_score(obj_desc, context)
                
                # 评估风险级别
                risk_profile = template.get("risk_profile", "medium")
                if risk_profile == "high":
                    risk_level = StrategicRiskLevel.HIGH
                elif risk_profile == "low":
                    risk_level = StrategicRiskLevel.LOW
                else:
                    risk_level = StrategicRiskLevel.MEDIUM
                
                # 计算优先级
                priority = (value_score * 0.4 + feasibility_score * 0.3 + 
                           alignment_score * 0.3)
                
                # 估算资源需求
                resource_intensity = template.get("resource_intensity", "medium")
                if resource_intensity == "high":
                    resource_requirements = {"time": 12.0, "budget": 15.0, "personnel": 5.0}
                elif resource_intensity == "low":
                    resource_requirements = {"time": 3.0, "budget": 3.0, "personnel": 1.0}
                else:
                    resource_requirements = {"time": 6.0, "budget": 8.0, "personnel": 3.0}
                
                # 创建战略目标
                objective = StrategicObjective(
                    id=f"template_{int(time.time())}_{(zlib.adler32(str(obj_desc).encode('utf-8')) & 0xffffffff) % 10000}",
                    description=obj_desc,
                    strategic_focus_area=template["focus_areas"][0],
                    time_horizon=template.get("time_horizon", StrategicTimeHorizon.MEDIUM_TERM),
                    alignment_score=alignment_score,
                    value_score=value_score,
                    feasibility_score=feasibility_score,
                    risk_level=risk_level,
                    priority=priority,
                    resource_requirements=resource_requirements,
                    key_performance_indicators=self._create_generic_kpis(obj_desc),
                    success_criteria=[f"成功实现{obj_desc}"],
                    dependencies=[],
                    assumptions=["模板假设成立", "环境条件满足"],
                    constraints=context.constraints if hasattr(context, 'constraints') else []
                )
                
                objectives.append(objective)
                
            except Exception as e:
                logger.error(f"从模板创建目标失败 {obj_desc}: {e}")
                continue
        
        return objectives
    
    def _calculate_alignment_score(self, objective_description: str, 
                                  context: StrategicContext) -> float:
        """计算目标与使命/愿景的匹配度"""
        # 简单实现：基于关键词匹配
        mission_keywords = ["使命", "价值", "愿景", "目标"]
        objective_lower = objective_description.lower()
        
        score = 0.5  # 基础分
        
        # 检查是否包含使命关键词
        for keyword in mission_keywords:
            if keyword in context.mission_statement.lower() and keyword in objective_lower:
                score += 0.1
        
        # 检查是否包含核心价值观关键词
        for value in context.core_values:
            if value.lower() in objective_lower:
                score += 0.05
        
        # 限制在0-1之间
        return max(0.0, min(1.0, score))
    
    def _assess_opportunity_risk(self, opportunity: Dict[str, Any], 
                                context: StrategicContext) -> StrategicRiskLevel:
        """评估机会风险"""
        opportunity_type = opportunity.get("type", "")
        
        if "disruptive" in opportunity_type or "technology" in opportunity_type:
            return StrategicRiskLevel.HIGH
        elif "market" in opportunity_type:
            return StrategicRiskLevel.MEDIUM
        else:
            return StrategicRiskLevel.LOW
    
    def _determine_focus_area(self, opportunity: Dict[str, Any]) -> StrategicFocusArea:
        """确定战略重点领域"""
        opportunity_type = opportunity.get("type", "")
        
        if "market" in opportunity_type:
            return StrategicFocusArea.GROWTH
        elif "technology" in opportunity_type:
            return StrategicFocusArea.INNOVATION
        elif "capability" in opportunity_type:
            return StrategicFocusArea.COMPETITIVENESS
        else:
            return StrategicFocusArea.GROWTH
    
    def _determine_time_horizon(self, opportunity: Dict[str, Any]) -> StrategicTimeHorizon:
        """确定时间视野"""
        potential_impact = opportunity.get("potential_impact", "medium")
        
        if potential_impact == "high":
            return StrategicTimeHorizon.LONG_TERM
        elif potential_impact == "medium":
            return StrategicTimeHorizon.MEDIUM_TERM
        else:
            return StrategicTimeHorizon.SHORT_TERM
    
    def _estimate_resource_requirements(self, opportunity: Dict[str, Any],
                                       context: StrategicContext) -> Dict[str, float]:
        """估算资源需求"""
        # 简单实现：基于机会类型和影响估算
        potential_impact = opportunity.get("potential_impact", "medium")
        
        if potential_impact == "high":
            return {
                "time": 12.0,  # 月
                "budget": 20.0,  # 相对预算单位
                "personnel": 5.0,  # 人员需求
                "technology": 8.0  # 技术资源
            }
        elif potential_impact == "medium":
            return {
                "time": 6.0,
                "budget": 10.0,
                "personnel": 3.0,
                "technology": 4.0
            }
        else:
            return {
                "time": 3.0,
                "budget": 5.0,
                "personnel": 2.0,
                "technology": 2.0
            }
    
    def _create_kpis_for_opportunity(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """为机会创建关键绩效指标"""
        opportunity_type = opportunity.get("type", "")
        
        if "market" in opportunity_type:
            return {
                "market_share": "市场份额增长率",
                "revenue_growth": "收入增长率",
                "customer_acquisition": "新客户获取数"
            }
        elif "technology" in opportunity_type:
            return {
                "technology_adoption": "技术采纳率",
                "innovation_index": "创新指数",
                "patent_filings": "专利申请数"
            }
        else:
            return {
                "success_rate": "成功率",
                "efficiency_gain": "效率提升",
                "quality_improvement": "质量改进"
            }
    
    def _create_generic_kpis(self, objective_description: str) -> Dict[str, Any]:
        """创建通用的关键绩效指标"""
        return {
            "completion_rate": "完成率",
            "timeline_adherence": "时间表遵守率",
            "quality_score": "质量评分",
            "stakeholder_satisfaction": "利益相关者满意度"
        }
    
    def create_strategic_plan(self, 
                             context: StrategicContext,
                             strategic_analysis: Optional[StrategicAnalysis] = None,
                             strategic_objectives: Optional[List[StrategicObjective]] = None) -> StrategicPlan:
        """
        创建战略计划
        
        Args:
            context: 战略上下文
            strategic_analysis: 战略分析（可选，如未提供将自动执行）
            strategic_objectives: 战略目标（可选，如未提供将自动生成）
            
        Returns:
            战略计划
        """
        start_time = time.time()
        
        logger.info("开始创建战略计划...")
        
        try:
            # 执行战略分析（如未提供）
            if strategic_analysis is None:
                strategic_analysis = self.analyze_strategic_environment(context)
            
            # 生成战略目标（如未提供）
            if strategic_objectives is None:
                strategic_objectives = self.generate_strategic_objectives(strategic_analysis, context)
            
            # 优化资源配置
            resource_allocation = self.optimize_resource_allocation(
                strategic_objectives, 
                context.resource_constraints
            )
            
            # 制定风险缓解策略
            risk_mitigation_strategies = self.develop_risk_mitigation_strategies(
                strategic_objectives, 
                context
            )
            
            # 创建监控框架
            monitoring_framework = self.create_monitoring_framework(strategic_objectives)
            
            # 制定应急计划
            contingency_plans = self.create_contingency_plans(strategic_objectives, context)
            
            # 制定实施路线图
            implementation_roadmap = self.create_implementation_roadmap(strategic_objectives)
            
            # 定义成功度量指标
            success_metrics = self.define_success_metrics(strategic_objectives)
            
            # 创建战略计划
            plan_id = f"strategic_plan_{int(time.time())}_{(zlib.adler32(str(context.mission_statement).encode('utf-8')) & 0xffffffff) % 10000}"
            
            strategic_plan = StrategicPlan(
                id=plan_id,
                name=f"战略计划 {datetime.now().strftime('%Y-%m')}",
                description=f"基于{context.mission_statement[:30]}...的战略计划",
                time_period=(
                    datetime.now(),
                    datetime.now() + timedelta(days=self.config.get("strategic_plan_validity_period_days", 365))
                ),
                strategic_context=context,
                strategic_analysis=strategic_analysis,
                strategic_objectives=strategic_objectives,
                resource_allocation=resource_allocation,
                risk_mitigation_strategies=risk_mitigation_strategies,
                monitoring_framework=monitoring_framework,
                contingency_plans=contingency_plans,
                implementation_roadmap=implementation_roadmap,
                success_metrics=success_metrics
            )
            
            # 更新当前计划和历史记录
            self.current_strategic_plan = strategic_plan
            self.strategic_plans_history.append(strategic_plan)
            
            # 限制历史记录大小
            if len(self.strategic_plans_history) > self.max_history_size:
                self.strategic_plans_history = self.strategic_plans_history[-self.max_history_size:]
            
            # 更新性能统计
            execution_time = time.time() - start_time
            self._update_performance_stats("strategic_plans_created", execution_time)
            
            logger.info(f"战略计划创建完成，ID: {plan_id}，用时: {execution_time:.2f} 秒")
            
            return strategic_plan
            
        except Exception as e:
            logger.error(f"创建战略计划失败: {e}")
            if error_handler:
                error_handler.handle_error(e, "StrategicPlanner", "创建战略计划失败")
            
            # 返回基本的战略计划
            return StrategicPlan(
                id=f"failed_plan_{int(time.time())}",
                name="失败的战略计划",
                description="计划创建失败",
                time_period=(datetime.now(), datetime.now() + timedelta(days=365)),
                strategic_context=context,
                strategic_analysis=StrategicAnalysis(
                    swot_analysis={},
                    pestel_analysis={},
                    five_forces_analysis={},
                    core_competencies=[],
                    strategic_gaps=[],
                    opportunity_areas=[],
                    threat_areas=[],
                    trend_analysis={},
                    scenario_analysis={}
                ),
                strategic_objectives=[],
                resource_allocation={},
                risk_mitigation_strategies=[],
                monitoring_framework={},
                contingency_plans=[],
                implementation_roadmap={},
                success_metrics={}
            )
    
    def optimize_resource_allocation(self,
                                   strategic_objectives: List[StrategicObjective],
                                   resource_constraints: Dict[str, float]) -> Dict[str, float]:
        """
        优化资源配置
        
        Args:
            strategic_objectives: 战略目标列表
            resource_constraints: 资源约束
            
        Returns:
            优化的资源配置
        """
        logger.info("开始优化资源配置...")
        
        try:
            # 简单的资源配置算法：基于优先级分配
            total_priority = sum(obj.priority for obj in strategic_objectives)
            if total_priority == 0:
                total_priority = 1.0  # 避免除零
            
            allocation = {}
            
            # 初始化资源分配
            for resource, constraint in resource_constraints.items():
                allocation[resource] = 0.0
            
            # 按优先级分配资源
            for objective in strategic_objectives:
                weight = objective.priority / total_priority
                
                for resource, constraint in resource_constraints.items():
                    # 获取目标的资源需求
                    objective_requirement = objective.resource_requirements.get(resource, 0.0)
                    
                    # 按权重分配，但不超过约束
                    allocated = min(objective_requirement * weight, constraint - allocation.get(resource, 0.0))
                    allocation[resource] = allocation.get(resource, 0.0) + allocated
            
            logger.info(f"资源配置优化完成，分配了 {len(allocation)} 种资源")
            return allocation
            
        except Exception as e:
            logger.error(f"资源配置优化失败: {e}")
            # 返回平均分配
            allocation = {}
            num_objectives = len(strategic_objectives)
            if num_objectives == 0:
                num_objectives = 1
            
            for resource, constraint in resource_constraints.items():
                allocation[resource] = constraint / num_objectives
            
            return allocation
    
    def develop_risk_mitigation_strategies(self,
                                          strategic_objectives: List[StrategicObjective],
                                          context: StrategicContext) -> List[Dict[str, Any]]:
        """制定风险缓解策略"""
        strategies = []
        
        for objective in strategic_objectives:
            if objective.risk_level == StrategicRiskLevel.HIGH or objective.risk_level == StrategicRiskLevel.VERY_HIGH:
                strategy = {
                    "objective_id": objective.id,
                    "risk_description": f"目标'{objective.description}'的高风险",
                    "mitigation_strategy": "分散风险、建立缓冲、定期监控",
                    "responsibility": "风险管理团队",
                    "timeline": "持续进行",
                    "success_metrics": ["风险发生率降低", "影响程度减小"]
                }
                strategies.append(strategy)
        
        return strategies
    
    def create_monitoring_framework(self, strategic_objectives: List[StrategicObjective]) -> Dict[str, Any]:
        """创建监控框架"""
        framework = {
            "monitoring_frequency": "季度",
            "reporting_structure": {
                "operational": "月度报告",
                "tactical": "季度报告", 
                "strategic": "年度报告"
            },
            "key_metrics": [],
            "alert_thresholds": {},
            "review_process": "季度战略审查"
        }
        
        # 添加关键指标
        for objective in strategic_objectives:
            for kpi_name, kpi_desc in objective.key_performance_indicators.items():
                framework["key_metrics"].append({
                    "name": kpi_name,
                    "description": kpi_desc,
                    "objective_id": objective.id,
                    "target": "待设定",
                    "current_value": "待收集"
                })
        
        return framework
    
    def create_contingency_plans(self, 
                                strategic_objectives: List[StrategicObjective],
                                context: StrategicContext) -> List[Dict[str, Any]]:
        """制定应急计划"""
        contingency_plans = []
        
        # 为高风险目标创建应急计划
        for objective in strategic_objectives:
            if objective.risk_level == StrategicRiskLevel.HIGH or objective.risk_level == StrategicRiskLevel.VERY_HIGH:
                plan = {
                    "trigger_condition": f"目标'{objective.description}'进展受阻或风险发生",
                    "contingency_actions": [
                        "重新评估目标可行性",
                        "调整资源分配",
                        "实施替代方案",
                        "启动危机管理流程"
                    ],
                    "escalation_procedure": "立即上报战略委员会",
                    "recovery_target": "在3个月内恢复正常进展",
                    "communication_plan": "向所有利益相关者通报情况"
                }
                contingency_plans.append(plan)
        
        return contingency_plans
    
    def create_implementation_roadmap(self, strategic_objectives: List[StrategicObjective]) -> Dict[str, Any]:
        """制定实施路线图"""
        # 按时间视野分组目标
        objectives_by_horizon = {
            StrategicTimeHorizon.SHORT_TERM: [],
            StrategicTimeHorizon.MEDIUM_TERM: [],
            StrategicTimeHorizon.LONG_TERM: [],
            StrategicTimeHorizon.VERY_LONG_TERM: []
        }
        
        for objective in strategic_objectives:
            objectives_by_horizon[objective.time_horizon].append(objective)
        
        # 创建路线图阶段
        phases = []
        
        # 短期阶段（0-12个月）
        if objectives_by_horizon[StrategicTimeHorizon.SHORT_TERM]:
            phases.append({
                "name": "短期实施阶段",
                "duration": "0-12个月",
                "objectives": [obj.id for obj in objectives_by_horizon[StrategicTimeHorizon.SHORT_TERM]],
                "key_milestones": ["启动项目", "建立基线", "完成初步成果"],
                "success_criteria": ["短期目标完成率>80%", "利益相关者满意度>75%"]
            })
        
        # 中期阶段（1-3年）
        if objectives_by_horizon[StrategicTimeHorizon.MEDIUM_TERM]:
            phases.append({
                "name": "中期实施阶段",
                "duration": "1-3年",
                "objectives": [obj.id for obj in objectives_by_horizon[StrategicTimeHorizon.MEDIUM_TERM]],
                "key_milestones": ["扩大规模", "优化流程", "建立能力"],
                "success_criteria": ["市场地位提升", "运营效率改善", "能力建设完成"]
            })
        
        # 长期阶段（3-5年）
        if objectives_by_horizon[StrategicTimeHorizon.LONG_TERM]:
            phases.append({
                "name": "长期实施阶段",
                "duration": "3-5年",
                "objectives": [obj.id for obj in objectives_by_horizon[StrategicTimeHorizon.LONG_TERM]],
                "key_milestones": ["实现战略转型", "建立可持续优势", "创造新价值"],
                "success_criteria": ["战略目标全面实现", "可持续竞争优势", "价值创造显著"]
            })
        
        roadmap = {
            "phases": phases,
            "timeline": f"{len(phases)}个阶段，{sum(len(phase['objectives']) for phase in phases)}个目标",
            "dependencies": "阶段间存在依赖关系，前一阶段成果为后一阶段基础",
            "flexibility": "允许根据环境变化调整路线图",
            "review_points": ["每季度进度审查", "年度战略重新评估"]
        }
        
        return roadmap
    
    def define_success_metrics(self, strategic_objectives: List[StrategicObjective]) -> Dict[str, Any]:
        """定义成功度量指标"""
        metrics = {
            "objective_completion_rate": {
                "description": "目标完成率",
                "target": ">80%",
                "measurement": "已完成目标数 / 总目标数"
            },
            "timeline_adherence": {
                "description": "时间表遵守率",
                "target": ">70%",
                "measurement": "按时完成目标数 / 总目标数"
            },
            "resource_efficiency": {
                "description": "资源效率",
                "target": "<预算的110%",
                "measurement": "实际资源消耗 / 计划资源"
            },
            "stakeholder_satisfaction": {
                "description": "利益相关者满意度",
                "target": ">75%",
                "measurement": "满意度调查评分"
            },
            "strategic_alignment": {
                "description": "战略对齐度",
                "target": ">70%",
                "measurement": "与使命/愿景对齐的目标数 / 总目标数"
            },
            "risk_mitigation_effectiveness": {
                "description": "风险缓解效果",
                "target": "风险发生率降低>50%",
                "measurement": "实际风险发生数 / 预计风险发生数"
            }
        }
        
        return metrics
    
    def review_strategic_plan(self, plan: StrategicPlan, 
                             new_context: Optional[StrategicContext] = None) -> Tuple[bool, List[str]]:
        """
        审查战略计划
        
        Args:
            plan: 要审查的战略计划
            new_context: 新的战略上下文（可选）
            
        Returns:
            (是否需要更新, 审查发现)
        """
        logger.info(f"开始审查战略计划: {plan.id}")
        
        findings = []
        needs_update = False
        
        try:
            # 检查计划是否过期
            _, plan_end = plan.time_period
            if datetime.now() > plan_end:
                findings.append("计划已过期")
                needs_update = True
            
            # 检查环境变化
            if new_context is not None:
                # 简单比较：如果外部环境有重大变化
                old_env = plan.strategic_context.external_environment
                new_env = new_context.external_environment
                
                if old_env != new_env:
                    findings.append("外部环境发生重大变化")
                    needs_update = True
            
            # 检查目标进展
            # 这里可以添加更复杂的进展检查逻辑
            
            # 检查资源约束变化
            # 这里可以添加资源约束检查逻辑
            
            if not findings:
                findings.append("计划审查通过，无需更新")
            
            logger.info(f"战略计划审查完成: 需要更新={needs_update}, 发现={len(findings)}条")
            
            return needs_update, findings
            
        except Exception as e:
            logger.error(f"战略计划审查失败: {e}")
            return True, [f"审查失败: {str(e)}"]
    
    def adapt_strategic_plan(self, plan: StrategicPlan,
                           new_context: StrategicContext,
                           findings: List[str]) -> StrategicPlan:
        """
        调整战略计划以适应变化
        
        Args:
            plan: 原始战略计划
            new_context: 新的战略上下文
            findings: 审查发现
            
        Returns:
            调整后的战略计划
        """
        logger.info(f"开始调整战略计划: {plan.id}")
        
        try:
            # 重新分析环境
            new_analysis = self.analyze_strategic_environment(new_context)
            
            # 重新生成目标（基于新分析和原有目标）
            new_objectives = []
            
            # 保留仍然相关的原有目标
            for objective in plan.strategic_objectives:
                # 检查目标是否仍然相关
                if self._is_objective_still_relevant(objective, new_context, new_analysis):
                    new_objectives.append(objective)
            
            # 添加基于新分析的目标
            additional_objectives = self.generate_strategic_objectives(new_analysis, new_context)
            
            # 合并目标，避免重复
            existing_descriptions = [obj.description for obj in new_objectives]
            for obj in additional_objectives:
                if obj.description not in existing_descriptions:
                    new_objectives.append(obj)
                    existing_descriptions.append(obj.description)
            
            # 创建调整后的计划
            adapted_plan = self.create_strategic_plan(
                context=new_context,
                strategic_analysis=new_analysis,
                strategic_objectives=new_objectives
            )
            
            # 添加调整元数据
            adapted_plan.metadata["adapted_from"] = plan.id
            adapted_plan.metadata["adaptation_reason"] = "; ".join(findings)
            adapted_plan.metadata["adaptation_date"] = datetime.now().isoformat()
            
            logger.info(f"战略计划调整完成，新计划ID: {adapted_plan.id}")
            
            return adapted_plan
            
        except Exception as e:
            logger.error(f"战略计划调整失败: {e}")
            # 返回原始计划（标记为需要手动调整）
            plan.metadata["adaptation_failed"] = True
            plan.metadata["adaptation_error"] = str(e)
            return plan
    
    def _is_objective_still_relevant(self, objective: StrategicObjective,
                                   context: StrategicContext,
                                   analysis: StrategicAnalysis) -> bool:
        """检查目标是否仍然相关"""
        # 简单实现：检查目标是否与当前使命对齐
        alignment_score = self._calculate_alignment_score(objective.description, context)
        return alignment_score >= self.config.get("min_strategic_alignment_score", 0.7)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "performance_stats": self.performance_stats.copy(),
            "current_plan": self.current_strategic_plan.id if self.current_strategic_plan else None,
            "plans_in_history": len(self.strategic_plans_history),
            "world_model_integrated": self.world_model_integrated,
            "causal_reasoning_integrated": self.causal_reasoning_integrated,
            "config_summary": {
                "enable_environmental_analysis": self.config.get("enable_environmental_analysis", True),
                "enable_strategic_risk_assessment": self.config.get("enable_strategic_risk_assessment", True),
                "max_strategic_objectives": self.config.get("max_strategic_objectives", 10)
            }
        }
        
        return report