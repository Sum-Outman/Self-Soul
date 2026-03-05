"""
自我改进规划器

该模块实现AGI系统的改进计划生成和优化，包括：
1. 计划生成：基于弱点分析生成具体的改进计划
2. 资源规划：分配计算、内存、时间等资源
3. 风险评估：评估改进计划的风险和影响
4. 优先级排序：优化改进计划的执行顺序
5. 进度跟踪：监控改进计划的执行进度

规划过程：
1. 输入分析：接收性能评估和弱点分析结果
2. 目标设定：基于弱点设定具体的改进目标
3. 方案设计：设计具体的改进方案和步骤
4. 资源分配：分配执行改进计划所需的资源
5. 风险控制：识别和控制改进过程中的风险
6. 计划优化：优化改进计划的效率和效果

技术特性：
- 多目标优化算法
- 资源约束规划
- 风险评估模型
- 动态优先级调整
- 进度预测和监控
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
from scipy import optimize

# 配置日志
logger = logging.getLogger(__name__)

class ImprovementPlanStatus(Enum):
    """改进计划状态"""
    DRAFT = "draft"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"

class ResourceType(Enum):
    """资源类型"""
    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    HUMAN = "human"
    TIME = "time"

class RiskLevel(Enum):
    """风险级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ResourceRequirement:
    """资源需求"""
    resource_type: ResourceType
    amount: float
    unit: str
    duration_hours: float
    timing_constraint: Optional[str] = None
    flexibility: float = 0.5  # 0-1，灵活性越高越容易调整

@dataclass
class ImprovementAction:
    """改进行动"""
    action_id: str
    description: str
    target_weakness_id: str
    expected_outcome: str
    implementation_steps: List[str]
    prerequisites: List[str]
    estimated_duration_hours: float
    resource_requirements: List[ResourceRequirement]
    success_criteria: List[str]
    risk_assessment: Dict[str, Any]
    priority: float = 0.5

@dataclass
class RiskAssessment:
    """风险评估"""
    risk_id: str
    description: str
    risk_level: RiskLevel
    probability: float  # 0-1
    impact: float  # 0-1
    mitigation_strategies: List[str]
    contingency_plan: List[str]
    monitoring_indicators: List[str]

@dataclass
class ImprovementPlan:
    """改进计划"""
    plan_id: str
    plan_name: str
    description: str
    target_weaknesses: List[str]
    improvement_actions: List[ImprovementAction]
    overall_objectives: List[str]
    success_metrics: Dict[str, Any]
    
    # 规划信息
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    estimated_total_hours: float = 0.0
    resource_allocation: Dict[ResourceType, float] = field(default_factory=dict)
    
    # 状态信息
    status: ImprovementPlanStatus = ImprovementPlanStatus.DRAFT
    progress_percentage: float = 0.0
    completed_actions: List[str] = field(default_factory=list)
    current_actions: List[str] = field(default_factory=list)
    pending_actions: List[str] = field(default_factory=list)
    
    # 风险管理
    risks: List[RiskAssessment] = field(default_factory=list)
    risk_score: float = 0.0
    
    # 性能指标
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    target_metrics: Dict[str, float] = field(default_factory=dict)
    current_metrics: Dict[str, float] = field(default_factory=dict)
    
    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    version: str = "1.0"

@dataclass
class PlanningContext:
    """规划上下文"""
    weaknesses: List[Any]  # CognitiveWeakness列表
    performance_data: Dict[str, Any]
    available_resources: Dict[ResourceType, float]
    constraints: Dict[str, Any]
    preferences: Dict[str, Any]

class SelfImprovementPlanner:
    """
    自我改进规划器
    
    核心组件:
    1. 目标设定器: 基于弱点设定具体的改进目标
    2. 方案设计器: 设计具体的改进方案和步骤
    3. 资源分配器: 分配执行改进计划所需的资源
    4. 风险评估器: 识别和控制改进过程中的风险
    5. 进度优化器: 优化改进计划的执行顺序和进度
    6. 监控协调器: 监控计划执行并协调资源
    
    工作流程:
    弱点分析 → 目标设定器 → 设定目标 → 方案设计器 → 设计方案
    设计方案 → 资源分配器 → 分配资源 → 风险评估器 → 评估风险
    综合评估 → 进度优化器 → 优化计划 → 监控协调器 → 执行监控
    
    技术特性:
    - 多目标优化算法
    - 资源约束规划
    - 风险评估模型
    - 动态优先级调整
    - 进度预测和监控
    """
    
    def __init__(self,
                 planning_horizon_days: float = 30.0,
                 max_concurrent_actions: int = 5,
                 risk_tolerance: float = 0.3,
                 resource_optimization_enabled: bool = True,
                 dynamic_replanning_enabled: bool = True):
        """
        初始化自我改进规划器
        
        Args:
            planning_horizon_days: 规划时间范围（天）
            max_concurrent_actions: 最大并发行动数量
            risk_tolerance: 风险容忍度（0-1）
            resource_optimization_enabled: 是否启用资源优化
            dynamic_replanning_enabled: 是否启用动态重规划
        """
        self.planning_horizon_days = planning_horizon_days
        self.max_concurrent_actions = max_concurrent_actions
        self.risk_tolerance = risk_tolerance
        self.resource_optimization_enabled = resource_optimization_enabled
        self.dynamic_replanning_enabled = dynamic_replanning_enabled
        
        # 改进计划管理
        self.improvement_plans: Dict[str, ImprovementPlan] = {}
        self.active_plans: Set[str] = set()
        
        # 规划上下文
        self.current_context: Optional[PlanningContext] = None
        
        # 资源池
        self.resource_pool: Dict[ResourceType, Dict[str, Any]] = {}
        self._initialize_resource_pool()
        
        # 配置参数
        self.config = {
            'action_duration_estimation_factor': 1.2,
            'resource_safety_margin': 0.2,
            'risk_weight_factors': {
                'probability': 0.4,
                'impact': 0.6
            },
            'priority_weight_factors': {
                'severity': 0.3,
                'impact': 0.25,
                'feasibility': 0.2,
                'urgency': 0.15,
                'resource_efficiency': 0.1
            },
            'optimization_iterations': 100,
            'replanning_trigger_threshold': 0.3,
            'progress_monitoring_interval_hours': 6.0
        }
        
        # 性能统计
        self.performance_stats = {
            'plans_created': 0,
            'actions_executed': 0,
            'actions_completed': 0,
            'actions_failed': 0,
            'replannings_triggered': 0,
            'resource_utilization_rate': 0.0,
            'average_plan_success_rate': 0.0,
            'average_risk_score': 0.0
        }
        
        # 状态变量
        self.last_planning_time = time.time()
        self.system_start_time = time.time()
        
        logger.info(f"自我改进规划器初始化完成，规划时间范围: {planning_horizon_days} 天")
    
    def _initialize_resource_pool(self):
        """初始化资源池"""
        self.resource_pool = {
            ResourceType.COMPUTE: {
                "total": 100.0,  # 计算单位
                "allocated": 0.0,
                "unit": "compute_units",
                "cost_per_unit": 0.1,
                "renewal_rate": 10.0  # 每小时恢复
            },
            ResourceType.MEMORY: {
                "total": 64.0,  # GB
                "allocated": 0.0,
                "unit": "GB",
                "cost_per_unit": 0.05,
                "renewal_rate": 0.0  # 不自动恢复
            },
            ResourceType.STORAGE: {
                "total": 1000.0,  # GB
                "allocated": 0.0,
                "unit": "GB",
                "cost_per_unit": 0.01,
                "renewal_rate": 0.0
            },
            ResourceType.NETWORK: {
                "total": 1000.0,  # Mbps
                "allocated": 0.0,
                "unit": "Mbps",
                "cost_per_unit": 0.02,
                "renewal_rate": 100.0
            },
            ResourceType.TIME: {
                "total": 24.0 * 30,  # 30天的小时数
                "allocated": 0.0,
                "unit": "hours",
                "cost_per_unit": 1.0,
                "renewal_rate": 1.0  # 每小时1小时
            }
        }
    
    def create_improvement_plan(self, 
                                context: PlanningContext) -> ImprovementPlan:
        """创建改进计划"""
        logger.info("开始创建改进计划")
        
        start_time = time.time()
        self.current_context = context
        
        # 生成计划ID
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 设定改进目标
        objectives = self._set_improvement_objectives(context.weaknesses)
        
        # 设计改进行动
        improvement_actions = self._design_improvement_actions(context.weaknesses)
        
        # 分配资源
        resource_allocation = self._allocate_resources(improvement_actions)
        
        # 评估风险
        risks = self._assess_risks(improvement_actions, context)
        
        # 优化行动顺序
        optimized_actions = self._optimize_action_sequence(improvement_actions)
        
        # 计算总时长
        total_hours = sum(action.estimated_duration_hours for action in optimized_actions)
        
        # 设置成功指标
        success_metrics = self._define_success_metrics(context.weaknesses)
        
        # 创建改进计划
        plan = ImprovementPlan(
            plan_id=plan_id,
            plan_name=f"改进计划_{plan_id}",
            description=f"基于{len(context.weaknesses)}个弱点的改进计划",
            target_weaknesses=[w.weakness_id for w in context.weaknesses],
            improvement_actions=optimized_actions,
            overall_objectives=objectives,
            success_metrics=success_metrics,
            estimated_total_hours=total_hours,
            resource_allocation=resource_allocation,
            risks=risks,
            risk_score=self._calculate_overall_risk_score(risks),
            baseline_metrics=self._extract_baseline_metrics(context.performance_data),
            target_metrics=self._calculate_target_metrics(context.weaknesses, context.performance_data),
            status=ImprovementPlanStatus.DRAFT
        )
        
        # 保存计划
        self.improvement_plans[plan_id] = plan
        
        # 更新统计
        self.performance_stats['plans_created'] += 1
        self.performance_stats['average_risk_score'] = (
            (self.performance_stats['average_risk_score'] * 
             (self.performance_stats['plans_created'] - 1) + 
             plan.risk_score) / self.performance_stats['plans_created']
        )
        
        # 更新最后规划时间
        self.last_planning_time = time.time()
        
        planning_duration = time.time() - start_time
        logger.info(f"改进计划创建完成: {plan_id}, 行动数: {len(optimized_actions)}, 总时长: {total_hours:.1f}小时")
        
        return plan
    
    def _set_improvement_objectives(self, weaknesses: List[Any]) -> List[str]:
        """设定改进目标"""
        objectives = []
        
        for weakness in weaknesses:
            objective = f"改进{weakness.description}，目标提升{(weakness.impact_assessment.get('overall', 0.5) * 100):.1f}%"
            objectives.append(objective)
        
        # 添加总体目标
        if weaknesses:
            objectives.append(f"总体性能提升{np.mean([w.impact_assessment.get('overall', 0.5) for w in weaknesses]) * 100:.1f}%")
            objectives.append(f"解决{len([w for w in weaknesses if w.severity.value in ['high', 'critical']])}个高优先级弱点")
        
        return objectives
    
    def _design_improvement_actions(self, weaknesses: List[Any]) -> List[ImprovementAction]:
        """设计改进行动"""
        improvement_actions = []
        
        for i, weakness in enumerate(weaknesses):
            # 为每个弱点设计1-3个改进行动
            num_actions = min(3, int(weakness.impact_assessment.get('overall', 0.5) * 3) + 1)
            
            for j in range(num_actions):
                action = self._create_improvement_action(weakness, j + 1, i * 10 + j)
                improvement_actions.append(action)
        
        return improvement_actions
    
    def _create_improvement_action(self, weakness: Any, action_num: int, index: int) -> ImprovementAction:
        """创建改进行动"""
        action_id = f"action_{weakness.weakness_id}_{action_num}"
        
        # 基于弱点类别设计行动
        if weakness.category.value == "cognitive":
            description = f"优化{weakness.description}的认知算法"
            steps = [
                f"分析{weakness.weakness_id}的当前算法实现",
                f"设计改进算法方案",
                f"实现算法优化",
                f"测试优化效果",
                f"部署优化后的算法"
            ]
            resource_reqs = [
                ResourceRequirement(
                    resource_type=ResourceType.COMPUTE,
                    amount=20.0 + index * 2.0,
                    unit="compute_units",
                    duration_hours=4.0 + index * 0.5
                ),
                ResourceRequirement(
                    resource_type=ResourceType.MEMORY,
                    amount=8.0,
                    unit="GB",
                    duration_hours=4.0 + index * 0.5
                )
            ]
        
        elif weakness.category.value == "execution":
            description = f"改进{weakness.description}的执行效率"
            steps = [
                f"分析{weakness.weakness_id}的执行瓶颈",
                f"设计执行优化方案",
                f"实现执行优化",
                f"测试执行效率提升",
                f"部署优化方案"
            ]
            resource_reqs = [
                ResourceRequirement(
                    resource_type=ResourceType.COMPUTE,
                    amount=15.0 + index * 1.5,
                    unit="compute_units",
                    duration_hours=3.0 + index * 0.4
                ),
                ResourceRequirement(
                    resource_type=ResourceType.NETWORK,
                    amount=100.0,
                    unit="Mbps",
                    duration_hours=2.0 + index * 0.3
                )
            ]
        
        elif weakness.category.value == "social":
            description = f"增强{weakness.description}的社会能力"
            steps = [
                f"分析{weakness.weakness_id}的社会交互问题",
                f"设计社会能力增强方案",
                f"实现社会能力优化",
                f"测试社会交互改进",
                f"部署社会能力增强"
            ]
            resource_reqs = [
                ResourceRequirement(
                    resource_type=ResourceType.COMPUTE,
                    amount=25.0 + index * 2.5,
                    unit="compute_units",
                    duration_hours=5.0 + index * 0.6
                ),
                ResourceRequirement(
                    resource_type=ResourceType.STORAGE,
                    amount=50.0,
                    unit="GB",
                    duration_hours=5.0 + index * 0.6
                )
            ]
        
        else:  # technical or systemic
            description = f"修复{weakness.description}的技术问题"
            steps = [
                f"诊断{weakness.weakness_id}的技术问题",
                f"设计技术修复方案",
                f"实现技术修复",
                f"测试修复效果",
                f"部署修复方案"
            ]
            resource_reqs = [
                ResourceRequirement(
                    resource_type=ResourceType.COMPUTE,
                    amount=10.0 + index * 1.0,
                    unit="compute_units",
                    duration_hours=2.0 + index * 0.3
                ),
                ResourceRequirement(
                    resource_type=ResourceType.TIME,
                    amount=2.0 + index * 0.5,
                    unit="hours",
                    duration_hours=2.0 + index * 0.3
                )
            ]
        
        # 估计持续时间
        duration_hours = 2.0 + index * 0.5
        
        # 风险评估
        risk_assessment = {
            "technical_risk": 0.3 + index * 0.02,
            "schedule_risk": 0.2 + index * 0.01,
            "resource_risk": 0.25 + index * 0.015,
            "quality_risk": 0.35 + index * 0.025
        }
        
        # 成功标准
        success_criteria = [
            f"{weakness.description}的性能提升≥20%",
            f"执行过程中无重大错误",
            f"资源使用在预算范围内",
            f"按时完成改进行动"
        ]
        
        # 优先级计算
        priority = weakness.priority_score * (1.0 - index * 0.01)
        
        return ImprovementAction(
            action_id=action_id,
            description=description,
            target_weakness_id=weakness.weakness_id,
            expected_outcome=f"改进{weakness.description}，预计提升效果{weakness.impact_assessment.get('overall', 0.5) * 100:.1f}%",
            implementation_steps=steps,
            prerequisites=[],
            estimated_duration_hours=duration_hours,
            resource_requirements=resource_reqs,
            success_criteria=success_criteria,
            risk_assessment=risk_assessment,
            priority=priority
        )
    
    def _allocate_resources(self, improvement_actions: List[ImprovementAction]) -> Dict[ResourceType, float]:
        """分配资源"""
        resource_allocation = {}
        
        # 计算每种资源的总需求
        resource_totals = {}
        for action in improvement_actions:
            for req in action.resource_requirements:
                resource_type = req.resource_type
                amount = req.amount * req.duration_hours
                
                if resource_type not in resource_totals:
                    resource_totals[resource_type] = 0.0
                resource_totals[resource_type] += amount
        
        # 应用安全边际
        for resource_type, total in resource_totals.items():
            safe_total = total * (1.0 + self.config['resource_safety_margin'])
            resource_allocation[resource_type] = safe_total
        
        return resource_allocation
    
    def _assess_risks(self, 
                     improvement_actions: List[ImprovementAction],
                     context: PlanningContext) -> List[RiskAssessment]:
        """评估风险"""
        risks = []
        
        # 技术风险
        tech_risk = RiskAssessment(
            risk_id="risk_technical_001",
            description="技术实现复杂度超出预期",
            risk_level=RiskLevel.MEDIUM,
            probability=0.4,
            impact=0.6,
            mitigation_strategies=[
                "分阶段实施技术改进",
                "准备备选技术方案",
                "增加技术评审环节"
            ],
            contingency_plan=[
                "降低技术改进目标",
                "延长实施时间",
                "寻求外部技术支持"
            ],
            monitoring_indicators=[
                "技术进展速度",
                "技术问题数量",
                "代码复杂度变化"
            ]
        )
        risks.append(tech_risk)
        
        # 进度风险
        schedule_risk = RiskAssessment(
            risk_id="risk_schedule_001",
            description="改进计划进度延迟",
            risk_level=RiskLevel.HIGH,
            probability=0.5,
            impact=0.7,
            mitigation_strategies=[
                "设置进度缓冲时间",
                "优先实施关键改进",
                "定期进度审查"
            ],
            contingency_plan=[
                "调整改进优先级",
                "增加资源投入",
                "延后非关键改进"
            ],
            monitoring_indicators=[
                "实际进度 vs 计划进度",
                "行动完成率",
                "资源消耗率"
            ]
        )
        risks.append(schedule_risk)
        
        # 资源风险
        resource_risk = RiskAssessment(
            risk_id="risk_resource_001",
            description="资源不足或分配不当",
            risk_level=RiskLevel.MEDIUM,
            probability=0.3,
            impact=0.5,
            mitigation_strategies=[
                "动态调整资源分配",
                "优化资源使用效率",
                "建立资源优先级"
            ],
            contingency_plan=[
                "缩减非关键行动资源",
                "延迟低优先级行动",
                "增加资源预算"
            ],
            monitoring_indicators=[
                "资源利用率",
                "资源等待时间",
                "资源冲突次数"
            ]
        )
        risks.append(resource_risk)
        
        # 质量风险
        quality_risk = RiskAssessment(
            risk_id="risk_quality_001",
            description="改进质量不达标",
            risk_level=RiskLevel.HIGH,
            probability=0.35,
            impact=0.8,
            mitigation_strategies=[
                "建立严格的质量标准",
                "增加测试覆盖",
                "实施质量审查"
            ],
            contingency_plan=[
                "重新设计改进方案",
                "增加质量保证投入",
                "接受部分质量妥协"
            ],
            monitoring_indicators=[
                "测试通过率",
                "缺陷密度",
                "用户满意度"
            ]
        )
        risks.append(quality_risk)
        
        return risks
    
    def _calculate_overall_risk_score(self, risks: List[RiskAssessment]) -> float:
        """计算总体风险得分"""
        if not risks:
            return 0.0
        
        risk_scores = []
        for risk in risks:
            # 风险级别权重
            level_weights = {
                RiskLevel.LOW: 0.25,
                RiskLevel.MEDIUM: 0.5,
                RiskLevel.HIGH: 0.75,
                RiskLevel.CRITICAL: 1.0
            }
            level_weight = level_weights.get(risk.risk_level, 0.5)
            
            # 风险得分 = 概率 × 影响 × 级别权重
            risk_score = risk.probability * risk.impact * level_weight
            risk_scores.append(risk_score)
        
        return np.mean(risk_scores)
    
    def _optimize_action_sequence(self, improvement_actions: List[ImprovementAction]) -> List[ImprovementAction]:
        """优化行动顺序"""
        if not improvement_actions:
            return []
        
        # 按优先级排序
        sorted_actions = sorted(improvement_actions, key=lambda a: a.priority, reverse=True)
        
        # 考虑依赖关系（如果有）
        # 这里简化处理，实际应该考虑行动间的依赖关系
        
        # 限制并发行动数量
        if len(sorted_actions) > self.max_concurrent_actions:
            # 将行动分组为批次
            batched_actions = []
            for i in range(0, len(sorted_actions), self.max_concurrent_actions):
                batch = sorted_actions[i:i + self.max_concurrent_actions]
                # 在批次内按持续时间排序（短任务优先）
                batch.sort(key=lambda a: a.estimated_duration_hours)
                batched_actions.extend(batch)
            sorted_actions = batched_actions
        
        return sorted_actions
    
    def _define_success_metrics(self, weaknesses: List[Any]) -> Dict[str, Any]:
        """定义成功指标"""
        success_metrics = {
            "primary_metrics": {},
            "secondary_metrics": {},
            "process_metrics": {}
        }
        
        # 主要指标：弱点改进
        for weakness in weaknesses:
            metric_key = f"weakness_{weakness.weakness_id}_improvement"
            target_improvement = weakness.impact_assessment.get('overall', 0.5) * 100
            success_metrics["primary_metrics"][metric_key] = {
                "description": f"{weakness.description}改进程度",
                "target": target_improvement,
                "unit": "百分比",
                "weight": weakness.priority_score
            }
        
        # 次要指标：过程指标
        success_metrics["secondary_metrics"] = {
            "plan_completion_rate": {
                "description": "计划完成率",
                "target": 90.0,
                "unit": "百分比",
                "weight": 0.3
            },
            "resource_efficiency": {
                "description": "资源效率",
                "target": 85.0,
                "unit": "百分比",
                "weight": 0.2
            },
            "schedule_adherence": {
                "description": "进度符合度",
                "target": 80.0,
                "unit": "百分比",
                "weight": 0.25
            },
            "quality_score": {
                "description": "质量得分",
                "target": 90.0,
                "unit": "得分",
                "weight": 0.25
            }
        }
        
        # 过程指标
        success_metrics["process_metrics"] = {
            "actions_completed": {
                "description": "完成行动数",
                "target": len(weaknesses) * 2,  # 平均每个弱点2个行动
                "unit": "数量",
                "weight": 0.1
            },
            "risk_mitigation": {
                "description": "风险缓解率",
                "target": 70.0,
                "unit": "百分比",
                "weight": 0.15
            },
            "stakeholder_satisfaction": {
                "description": "利益相关者满意度",
                "target": 85.0,
                "unit": "得分",
                "weight": 0.1
            }
        }
        
        return success_metrics
    
    def _extract_baseline_metrics(self, performance_data: Dict[str, Any]) -> Dict[str, float]:
        """提取基线指标"""
        baseline_metrics = {}
        
        if "performance_dimensions" in performance_data:
            for dim_id, dim_data in performance_data["performance_dimensions"].items():
                baseline_metrics[dim_id] = dim_data.get("current_value", 0.0)
        
        return baseline_metrics
    
    def _calculate_target_metrics(self, 
                                 weaknesses: List[Any], 
                                 performance_data: Dict[str, Any]) -> Dict[str, float]:
        """计算目标指标"""
        target_metrics = self._extract_baseline_metrics(performance_data).copy()
        
        # 基于弱点改进目标调整目标值
        for weakness in weaknesses:
            # 找到弱点对应的性能指标
            for metric_id in weakness.performance_metrics_affected:
                if metric_id in target_metrics:
                    current_value = target_metrics[metric_id]
                    # 改进目标：提升20-50%
                    improvement_factor = 1.0 + weakness.impact_assessment.get('overall', 0.5) * 0.5
                    target_metrics[metric_id] = min(1.0, current_value * improvement_factor)
        
        return target_metrics
    
    def execute_improvement_plan(self, plan_id: str) -> Dict[str, Any]:
        """执行改进计划"""
        if plan_id not in self.improvement_plans:
            logger.error(f"改进计划不存在: {plan_id}")
            return {"status": "error", "message": "Plan not found"}
        
        plan = self.improvement_plans[plan_id]
        
        # 检查资源可用性
        resource_check = self._check_resource_availability(plan)
        if not resource_check["available"]:
            logger.error(f"资源不足，无法执行计划: {plan_id}")
            return {"status": "error", "message": "Insufficient resources", "details": resource_check}
        
        # 分配资源
        self._allocate_plan_resources(plan)
        
        # 更新计划状态
        plan.status = ImprovementPlanStatus.IN_PROGRESS
        plan.start_date = datetime.now()
        plan.end_date = plan.start_date + timedelta(hours=plan.estimated_total_hours)
        
        # 设置当前行动
        if plan.improvement_actions:
            first_actions = plan.improvement_actions[:self.max_concurrent_actions]
            plan.current_actions = [action.action_id for action in first_actions]
            plan.pending_actions = [action.action_id for action in plan.improvement_actions[self.max_concurrent_actions:]]
        
        # 添加到活动计划
        self.active_plans.add(plan_id)
        
        logger.info(f"开始执行改进计划: {plan_id}")
        
        return {
            "status": "started",
            "plan_id": plan_id,
            "start_time": plan.start_date,
            "estimated_end_time": plan.end_date,
            "current_actions": plan.current_actions,
            "resource_allocation": plan.resource_allocation
        }
    
    def _check_resource_availability(self, plan: ImprovementPlan) -> Dict[str, Any]:
        """检查资源可用性"""
        availability_check = {
            "available": True,
            "missing_resources": {},
            "suggestions": []
        }
        
        for resource_type, required_amount in plan.resource_allocation.items():
            if resource_type in self.resource_pool:
                resource_info = self.resource_pool[resource_type]
                available_amount = resource_info["total"] - resource_info["allocated"]
                
                if available_amount < required_amount:
                    availability_check["available"] = False
                    availability_check["missing_resources"][resource_type.value] = {
                        "required": required_amount,
                        "available": available_amount,
                        "shortage": required_amount - available_amount
                    }
                    
                    suggestion = f"增加{resource_type.value}资源或减少相关行动的{resource_type.value}需求"
                    availability_check["suggestions"].append(suggestion)
        
        return availability_check
    
    def _allocate_plan_resources(self, plan: ImprovementPlan):
        """分配计划资源"""
        for resource_type, amount in plan.resource_allocation.items():
            if resource_type in self.resource_pool:
                self.resource_pool[resource_type]["allocated"] += amount
                logger.debug(f"分配{resource_type.value}资源: {amount}{self.resource_pool[resource_type]['unit']}")
    
    def monitor_plan_progress(self, plan_id: str) -> Dict[str, Any]:
        """监控计划进度"""
        if plan_id not in self.improvement_plans:
            return {"status": "error", "message": "Plan not found"}
        
        plan = self.improvement_plans[plan_id]
        
        # 计算进度
        total_actions = len(plan.improvement_actions)
        completed_actions = len(plan.completed_actions)
        
        if total_actions > 0:
            progress_percentage = (completed_actions / total_actions) * 100
            plan.progress_percentage = progress_percentage
        else:
            progress_percentage = 0.0
        
        # 检查是否需要重规划
        needs_replanning = self._check_replanning_needed(plan)
        
        # 更新性能统计
        self.performance_stats['actions_completed'] = completed_actions
        if total_actions > 0:
            self.performance_stats['average_plan_success_rate'] = progress_percentage / 100.0
        
        progress_report = {
            "plan_id": plan_id,
            "status": plan.status.value,
            "progress_percentage": progress_percentage,
            "completed_actions": completed_actions,
            "total_actions": total_actions,
            "current_actions": plan.current_actions,
            "pending_actions": len(plan.pending_actions),
            "start_time": plan.start_date.isoformat() if plan.start_date else None,
            "estimated_end_time": plan.end_date.isoformat() if plan.end_date else None,
            "elapsed_time_hours": (datetime.now() - plan.start_date).total_seconds() / 3600 if plan.start_date else 0.0,
            "remaining_time_hours": (plan.end_date - datetime.now()).total_seconds() / 3600 if plan.start_date and plan.end_date else 0.0,
            "needs_replanning": needs_replanning,
            "risk_score": plan.risk_score,
            "resource_utilization": self._calculate_resource_utilization()
        }
        
        # 如果进度完成，更新状态
        if progress_percentage >= 99.9 and plan.status == ImprovementPlanStatus.IN_PROGRESS:
            plan.status = ImprovementPlanStatus.COMPLETED
            plan.progress_percentage = 100.0
            self._release_plan_resources(plan)
            self.active_plans.discard(plan_id)
            progress_report["status"] = "completed"
            
            logger.info(f"改进计划完成: {plan_id}")
        
        return progress_report
    
    def _check_replanning_needed(self, plan: ImprovementPlan) -> bool:
        """检查是否需要重规划"""
        if not self.dynamic_replanning_enabled:
            return False
        
        # 检查进度延迟
        if plan.start_date and plan.end_date:
            elapsed_time = (datetime.now() - plan.start_date).total_seconds() / 3600
            expected_progress = elapsed_time / plan.estimated_total_hours
            
            if expected_progress > 0:  # 避免除零
                actual_progress = plan.progress_percentage / 100.0
                progress_delay = (expected_progress - actual_progress) / expected_progress
                
                if progress_delay > self.config['replanning_trigger_threshold']:
                    logger.warning(f"计划{plan.plan_id}进度延迟{progress_delay*100:.1f}%，需要重规划")
                    return True
        
        # 检查资源问题
        resource_utilization = self._calculate_resource_utilization()
        for resource_type, utilization in resource_utilization.items():
            if utilization > 0.9:  # 资源利用率超过90%
                logger.warning(f"资源{resource_type}利用率过高: {utilization*100:.1f}%，需要重规划")
                return True
        
        # 检查风险变化
        if plan.risk_score > self.risk_tolerance * 1.5:  # 风险超过容忍度的150%
            logger.warning(f"计划{plan.plan_id}风险过高: {plan.risk_score:.3f}，需要重规划")
            return True
        
        return False
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """计算资源利用率"""
        utilization = {}
        
        for resource_type, resource_info in self.resource_pool.items():
            total = resource_info["total"]
            allocated = resource_info["allocated"]
            
            if total > 0:
                utilization[resource_type.value] = allocated / total
            else:
                utilization[resource_type.value] = 0.0
        
        return utilization
    
    def _release_plan_resources(self, plan: ImprovementPlan):
        """释放计划资源"""
        for resource_type, amount in plan.resource_allocation.items():
            if resource_type in self.resource_pool:
                self.resource_pool[resource_type]["allocated"] -= amount
                self.resource_pool[resource_type]["allocated"] = max(0.0, self.resource_pool[resource_type]["allocated"])
                logger.debug(f"释放{resource_type.value}资源: {amount}{self.resource_pool[resource_type]['unit']}")
    
    def get_planning_dashboard(self) -> Dict[str, Any]:
        """获取规划仪表板"""
        active_plans_count = len(self.active_plans)
        total_plans_count = len(self.improvement_plans)
        
        # 计算总体进度
        total_progress = 0.0
        active_plans_progress = []
        
        for plan_id in self.active_plans:
            if plan_id in self.improvement_plans:
                plan = self.improvement_plans[plan_id]
                total_progress += plan.progress_percentage
                active_plans_progress.append({
                    "plan_id": plan_id,
                    "name": plan.plan_name,
                    "progress": plan.progress_percentage,
                    "status": plan.status.value,
                    "risk_score": plan.risk_score
                })
        
        average_progress = total_progress / active_plans_count if active_plans_count > 0 else 0.0
        
        dashboard = {
            "timestamp": datetime.now(),
            "system_uptime_hours": (time.time() - self.system_start_time) / 3600,
            "planning_stats": {
                "total_plans": total_plans_count,
                "active_plans": active_plans_count,
                "completed_plans": sum(1 for p in self.improvement_plans.values() 
                                      if p.status == ImprovementPlanStatus.COMPLETED),
                "average_progress": average_progress,
                "resource_utilization": self._calculate_resource_utilization(),
                "average_risk_score": self.performance_stats['average_risk_score']
            },
            "performance_stats": self.performance_stats,
            "active_plans": active_plans_progress,
            "resource_status": self.resource_pool
        }
        
        return dashboard

# 全局实例
self_improvement_planner_instance = SelfImprovementPlanner()

if __name__ == "__main__":
    # 测试自我改进规划器
    print("测试自我改进规划器...")
    
    planner = SelfImprovementPlanner(
        planning_horizon_days=7.0,
        max_concurrent_actions=3,
        risk_tolerance=0.4
    )
    
    # 模拟规划上下文
    class MockWeakness:
        def __init__(self, weakness_id, description, category, priority_score, impact_assessment):
            self.weakness_id = weakness_id
            self.description = description
            self.category = type('Enum', (), {'value': category})()
            self.priority_score = priority_score
            self.impact_assessment = impact_assessment
            self.performance_metrics_affected = [f"metric_{weakness_id}"]
    
    weaknesses = [
        MockWeakness(
            "weakness_001",
            "逻辑推理准确率不足",
            "cognitive",
            0.85,
            {"overall": 0.7}
        ),
        MockWeakness(
            "weakness_002",
            "执行效率低下",
            "execution",
            0.65,
            {"overall": 0.5}
        )
    ]
    
    context = PlanningContext(
        weaknesses=weaknesses,
        performance_data={
            "performance_dimensions": {
                "metric_weakness_001": {"current_value": 0.75, "target_value": 0.95},
                "metric_weakness_002": {"current_value": 0.65, "target_value": 0.85}
            }
        },
        available_resources={
            "compute": 100.0,
            "memory": 64.0,
            "storage": 1000.0
        },
        constraints={},
        preferences={}
    )
    
    # 创建改进计划
    plan = planner.create_improvement_plan(context)
    print(f"改进计划创建完成: {plan.plan_id}")
    print(f"计划包含 {len(plan.improvement_actions)} 个改进行动")
    print(f"预计总时长: {plan.estimated_total_hours:.1f} 小时")
    print(f"总体风险得分: {plan.risk_score:.3f}")
    
    # 获取仪表板
    dashboard = planner.get_planning_dashboard()
    print(f"规划仪表板: {dashboard['planning_stats']['active_plans']} 个活动计划")