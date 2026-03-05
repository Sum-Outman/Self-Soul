#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zlib
"""
操作层规划器 - 处理短期、具体的行动执行和即时决策

功能:
1. 行动序列生成: 将任务分解为具体行动序列
2. 参数配置优化: 优化行动参数和执行配置
3. 实时决策支持: 提供执行中的即时决策支持
4. 执行监控反馈: 实时监控行动执行并提供反馈
5. 异常处理恢复: 处理执行异常并恢复执行
6. 资源动态调整: 根据执行情况动态调整资源

核心特性:
- 时间尺度: 分钟~小时级别
- 抽象级别: 具体行动、即时决策、参数设置
- 决策依据: 实时状态、环境反馈、资源可用性
- 输出形式: 动作序列、执行参数、监控数据

操作规划流程:
1. 任务解析: 解析战术任务为操作需求
2. 行动生成: 生成具体的行动序列
3. 参数优化: 优化行动参数和执行配置
4. 资源分配: 分配执行所需的实时资源
5. 执行监控: 实时监控执行状态和进度
6. 动态调整: 根据反馈调整执行计划

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

# 导入战术规划器
try:
    from .tactical_planner import TacticalPlanner, TaskSpecification
except ImportError:
    TacticalPlanner = TaskSpecification = None

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


class ActionType(Enum):
    """行动类型枚举"""
    PREPARATION = "preparation"        # 准备行动
    EXECUTION = "execution"            # 执行行动
    MONITORING = "monitoring"          # 监控行动
    VERIFICATION = "verification"      # 验证行动
    ADJUSTMENT = "adjustment"          # 调整行动
    COMPLETION = "completion"          # 完成行动


class ActionPrecision(Enum):
    """行动精度枚举"""
    LOW = "low"            # 低精度：粗略执行
    MEDIUM = "medium"      # 中等精度：标准执行
    HIGH = "high"          # 高精度：精细执行
    VERY_HIGH = "very_high"  # 非常高精度：精确执行


class ExecutionEnvironment(Enum):
    """执行环境枚举"""
    DEVELOPMENT = "development"      # 开发环境
    TESTING = "testing"              # 测试环境
    STAGING = "staging"              # 预发布环境
    PRODUCTION = "production"        # 生产环境
    SIMULATION = "simulation"        # 模拟环境


class ResourceAvailability(Enum):
    """资源可用性枚举"""
    PLENTIFUL = "plentiful"          # 充足：资源丰富
    ADEQUATE = "adequate"            # 充足：满足需求
    LIMITED = "limited"              # 有限：需优化使用
    CRITICAL = "critical"            # 紧缺：需严格管理


@dataclass
class OperationalContext:
    """操作上下文数据类"""
    task_specification: Optional[TaskSpecification]  # 任务规格
    current_state: Dict[str, Any]  # 当前状态
    available_resources: Dict[str, float]  # 可用资源
    environmental_conditions: Dict[str, Any]  # 环境条件
    realtime_constraints: Dict[str, Any]  # 实时约束
    quality_requirements: Dict[str, Any]  # 质量要求
    safety_requirements: List[str]  # 安全要求
    historical_performance: Dict[str, Any]  # 历史性能
    
    def __post_init__(self):
        """后初始化验证"""
        if not self.current_state:
            self.current_state = {"status": "ready", "progress": 0.0}


@dataclass
class ActionSpecification:
    """行动规格数据类"""
    id: str
    description: str
    task_specification_id: str  # 关联的任务规格ID
    action_type: ActionType
    precision: ActionPrecision
    parameters: Dict[str, Any]  # 执行参数
    estimated_duration: float  # 分钟
    prerequisites: List[str]  # 前置行动ID
    required_resources: Dict[str, float]  # 所需资源
    success_criteria: List[str]  # 成功标准
    failure_conditions: List[str]  # 失败条件
    monitoring_points: List[Dict[str, Any]]  # 监控点
    adjustment_rules: List[Dict[str, Any]]  # 调整规则
    safety_checks: List[str]  # 安全检查
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        if not self.id:
            self.id = f"action_spec_{int(time.time())}_{(zlib.adler32(str(self.description).encode('utf-8')) & 0xffffffff) % 10000}"
        self.estimated_duration = max(0.1, self.estimated_duration)


@dataclass
class ExecutionPlan:
    """执行计划数据类"""
    id: str
    name: str
    description: str
    task_specification_id: str
    action_sequence: List[ActionSpecification]  # 行动序列
    resource_allocation: Dict[str, Dict[str, float]]  # 资源分配
    schedule: Dict[str, Any]  # 时间安排
    monitoring_plan: Dict[str, Any]  # 监控计划
    adjustment_strategies: List[Dict[str, Any]]  # 调整策略
    contingency_plans: List[Dict[str, Any]]  # 应急计划
    performance_targets: Dict[str, Any]  # 性能目标
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        if not self.id:
            self.id = f"execution_plan_{int(time.time())}_{(zlib.adler32(str(self.name).encode('utf-8')) & 0xffffffff) % 10000}"


class OperationalPlanner:
    """
    操作层规划器 - 处理短期具体行动的执行规划
    
    核心功能:
    1. 行动分解: 将任务分解为具体的行动序列
    2. 参数优化: 优化行动参数和执行配置
    3. 资源调度: 调度实时资源支持执行
    4. 实时监控: 监控执行状态和进度
    5. 动态调整: 根据反馈动态调整执行
    6. 异常处理: 处理执行异常和恢复
    
    技术特点:
    - 实时性: 分钟~小时级别的时间响应
    - 精确性: 高精度的参数配置和行动控制
    - 适应性: 实时适应环境和状态变化
    - 安全性: 严格的安全检查和保护机制
    - 可靠性: 高可靠性的执行和恢复能力
    """
    
    def __init__(self, 
                 planning_system: Optional[HierarchicalPlanningSystem] = None,
                 operational_config: Optional[Dict[str, Any]] = None):
        """
        初始化操作规划器
        
        Args:
            planning_system: 分层规划系统实例（可选）
            operational_config: 操作规划配置（可选）
        """
        self.planning_system = planning_system
        self.config = operational_config or self._get_default_config()
        
        # 操作规划组件
        self.operational_context: Optional[OperationalContext] = None
        self.current_execution_plan: Optional[ExecutionPlan] = None
        self.execution_plans_history: List[ExecutionPlan] = []
        self.max_history_size = self.config.get('max_execution_plans_history', 50)
        
        # 行动模板库
        self.action_templates = self._initialize_action_templates()
        
        # 参数优化器配置
        self.parameter_optimizers = self._initialize_parameter_optimizers()
        
        # 监控器配置
        self.monitoring_configurations = self._initialize_monitoring_configurations()
        
        # 性能统计
        self.performance_stats = {
            "execution_plans_created": 0,
            "actions_generated": 0,
            "executions_monitored": 0,
            "adjustments_made": 0,
            "total_planning_time": 0.0,
            "average_planning_time": 0.0,
            "last_planning_time": None,
            "success_rate": 0.0
        }
        
        logger.info(f"操作规划器初始化完成，行动模板: {len(self.action_templates)} 个")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "enable_action_decomposition": True,
            "enable_parameter_optimization": True,
            "enable_realtime_monitoring": True,
            "enable_dynamic_adjustment": True,
            "max_actions_per_task": 30,
            "min_action_duration_minutes": 1.0,
            "max_action_duration_minutes": 480.0,  # 8小时
            "planning_time_limit": 30.0,  # 30秒
            "max_execution_plans_history": 50,
            "monitoring_frequency_seconds": 60.0,
            "adjustment_response_time_seconds": 5.0,
            "safety_check_enabled": True,
            "quality_control_enabled": True,
            "resource_optimization_enabled": True,
            "enable_failure_recovery": True,
            "recovery_time_limit_minutes": 30.0,
            "enable_performance_tracking": True
        }
    
    def _initialize_action_templates(self) -> Dict[str, Any]:
        """初始化行动模板库"""
        return {
            "preparation_actions": {
                "name": "准备行动模板",
                "description": "用于执行前的准备工作",
                "typical_actions": [
                    "资源检查与确认",
                    "环境准备与配置",
                    "工具与材料准备",
                    "安全检查与验证",
                    "参数设置与校准"
                ],
                "action_type": ActionType.PREPARATION,
                "typical_precision": ActionPrecision.HIGH,
                "estimated_duration_range": (5.0, 60.0),  # 分钟
                "required_resources": {"human": 1.0, "time": 1.0},
                "success_criteria": ["资源就绪", "环境就绪", "安全检查通过"],
                "safety_checks": ["环境安全", "工具安全", "操作安全"]
            },
            "execution_actions": {
                "name": "执行行动模板",
                "description": "用于主要的执行操作",
                "typical_actions": [
                    "启动执行过程",
                    "执行核心操作",
                    "处理执行数据",
                    "控制执行流程",
                    "记录执行结果"
                ],
                "action_type": ActionType.EXECUTION,
                "typical_precision": ActionPrecision.MEDIUM,
                "estimated_duration_range": (10.0, 240.0),
                "required_resources": {"human": 1.0, "computational": 2.0, "time": 1.0},
                "success_criteria": ["操作完成", "结果符合预期", "质量达标"],
                "safety_checks": ["过程安全", "数据安全", "结果安全"]
            },
            "monitoring_actions": {
                "name": "监控行动模板",
                "description": "用于执行过程的监控",
                "typical_actions": [
                    "状态检查与记录",
                    "进度监控与报告",
                    "质量检查与评估",
                    "风险检测与预警",
                    "性能测量与分析"
                ],
                "action_type": ActionType.MONITORING,
                "typical_precision": ActionPrecision.HIGH,
                "estimated_duration_range": (2.0, 30.0),
                "required_resources": {"monitoring": 1.0, "time": 1.0},
                "success_criteria": ["状态清晰", "数据准确", "预警及时"],
                "safety_checks": ["监控有效", "预警准确", "响应及时"]
            },
            "verification_actions": {
                "name": "验证行动模板",
                "description": "用于执行结果的验证",
                "typical_actions": [
                    "结果检查与验证",
                    "质量评估与确认",
                    "合规性检查",
                    "验收测试执行",
                    "文档验证与确认"
                ],
                "action_type": ActionType.VERIFICATION,
                "typical_precision": ActionPrecision.VERY_HIGH,
                "estimated_duration_range": (5.0, 120.0),
                "required_resources": {"verification": 1.0, "time": 1.0},
                "success_criteria": ["验证通过", "质量合格", "符合要求"],
                "safety_checks": ["验证准确", "标准符合", "记录完整"]
            },
            "adjustment_actions": {
                "name": "调整行动模板",
                "description": "用于执行过程中的调整",
                "typical_actions": [
                    "参数调整与优化",
                    "流程调整与改进",
                    "资源重新分配",
                    "策略调整与变更",
                    "纠偏措施执行"
                ],
                "action_type": ActionType.ADJUSTMENT,
                "typical_precision": ActionPrecision.MEDIUM,
                "estimated_duration_range": (3.0, 60.0),
                "required_resources": {"adjustment": 1.0, "time": 1.0},
                "success_criteria": ["调整有效", "问题解决", "改进明显"],
                "safety_checks": ["调整安全", "过程可控", "结果可预期"]
            },
            "completion_actions": {
                "name": "完成行动模板",
                "description": "用于执行完成后的收尾工作",
                "typical_actions": [
                    "结果整理与归档",
                    "资源清理与释放",
                    "文档完成与提交",
                    "经验总结与记录",
                    "交接与关闭"
                ],
                "action_type": ActionType.COMPLETION,
                "typical_precision": ActionPrecision.HIGH,
                "estimated_duration_range": (5.0, 90.0),
                "required_resources": {"human": 1.0, "time": 1.0},
                "success_criteria": ["工作完成", "资源释放", "文档齐全"],
                "safety_checks": ["收尾安全", "数据安全", "环境恢复"]
            }
        }
    
    def _initialize_parameter_optimizers(self) -> Dict[str, Any]:
        """初始化参数优化器"""
        return {
            "rule_based_optimization": {
                "name": "规则基础优化",
                "description": "基于预定义规则优化参数",
                "applicability": "适用于有明确规则的场景",
                "optimization_focus": ["参数范围", "约束条件", "经验规则"],
                "strengths": ["简单快速", "规则明确", "可解释性强"],
                "weaknesses": ["灵活性差", "依赖规则质量", "不适应新场景"]
            },
            "model_based_optimization": {
                "name": "模型基础优化",
                "description": "基于数学模型优化参数",
                "applicability": "适用于有明确模型的场景",
                "optimization_focus": ["数学模型", "目标函数", "约束条件"],
                "strengths": ["优化效果好", "理论支持强", "可处理复杂问题"],
                "weaknesses": ["模型依赖", "计算复杂", "实施难度大"]
            },
            "learning_based_optimization": {
                "name": "学习基础优化",
                "description": "基于机器学习优化参数",
                "applicability": "适用于数据丰富的场景",
                "optimization_focus": ["历史数据", "模式识别", "预测优化"],
                "strengths": ["适应性强", "可处理非线性", "持续改进"],
                "weaknesses": ["需要大量数据", "可解释性差", "训练时间长"]
            },
            "hybrid_optimization": {
                "name": "混合优化",
                "description": "结合多种优化方法",
                "applicability": "适用于复杂多变的场景",
                "optimization_focus": ["多方法结合", "动态选择", "综合优化"],
                "strengths": ["灵活性高", "适应性强", "效果好"],
                "weaknesses": ["实现复杂", "参数多", "调试困难"]
            }
        }
    
    def _initialize_monitoring_configurations(self) -> Dict[str, Any]:
        """初始化监控配置"""
        return {
            "basic_monitoring": {
                "name": "基础监控",
                "description": "基本的执行状态监控",
                "monitoring_frequency": "60秒",
                "monitored_aspects": ["进度", "状态", "基本质量"],
                "alert_thresholds": ["进度偏差>20%", "状态异常", "质量不合格"],
                "resource_requirements": "低",
                "suitability": "简单任务，低风险场景"
            },
            "advanced_monitoring": {
                "name": "高级监控",
                "description": "全面的执行监控与分析",
                "monitoring_frequency": "30秒",
                "monitored_aspects": ["进度", "质量", "性能", "风险", "资源"],
                "alert_thresholds": ["进度偏差>10%", "质量下降", "性能不达标", "风险升高", "资源紧张"],
                "resource_requirements": "中",
                "suitability": "中等任务，中等风险场景"
            },
            "intensive_monitoring": {
                "name": "密集监控",
                "description": "高频率、全方位的监控",
                "monitoring_frequency": "10秒",
                "monitored_aspects": ["实时进度", "实时质量", "实时性能", "实时风险", "实时资源", "实时安全"],
                "alert_thresholds": ["进度偏差>5%", "任何质量问题", "性能下降", "任何风险", "资源不足", "安全风险"],
                "resource_requirements": "高",
                "suitability": "关键任务，高风险场景"
            },
            "adaptive_monitoring": {
                "name": "自适应监控",
                "description": "根据情况动态调整的监控",
                "monitoring_frequency": "动态调整",
                "monitored_aspects": ["根据风险调整", "根据重要性调整", "根据阶段调整"],
                "alert_thresholds": ["动态调整", "基于上下文", "个性化设置"],
                "resource_requirements": "动态",
                "suitability": "复杂任务，动态变化场景"
            }
        }
    
    def set_operational_context(self, context: OperationalContext):
        """
        设置操作上下文
        
        Args:
            context: 操作上下文
        """
        self.operational_context = context
        logger.info(f"操作上下文已设置，当前状态: {context.current_state.get('status', 'unknown')}")
    
    def decompose_task_to_actions(self,
                                task_specification: TaskSpecification,
                                context: OperationalContext) -> List[ActionSpecification]:
        """
        分解任务为具体行动
        
        Args:
            task_specification: 任务规格
            context: 操作上下文
            
        Returns:
            行动规格列表
        """
        start_time = time.time()
        
        logger.info(f"开始分解任务: {task_specification.description}")
        
        try:
            action_specifications = []
            
            # 根据任务类型选择合适的行动模板
            task_desc = task_specification.description.lower()
            
            # 确定主要行动类型
            if any(keyword in task_desc for keyword in ["准备", "配置", "检查", "prepare", "setup", "check"]):
                primary_template = self.action_templates["preparation_actions"]
            elif any(keyword in task_desc for keyword in ["执行", "运行", "处理", "execute", "run", "process"]):
                primary_template = self.action_templates["execution_actions"]
            elif any(keyword in task_desc for keyword in ["监控", "检查", "观察", "monitor", "check", "observe"]):
                primary_template = self.action_templates["monitoring_actions"]
            elif any(keyword in task_desc for keyword in ["验证", "测试", "确认", "verify", "test", "confirm"]):
                primary_template = self.action_templates["verification_actions"]
            elif any(keyword in task_desc for keyword in ["调整", "优化", "改进", "adjust", "optimize", "improve"]):
                primary_template = self.action_templates["adjustment_actions"]
            else:
                # 默认使用完整执行模板序列
                return self._generate_complete_action_sequence(task_specification, context)
            
            # 生成基于模板的行动
            for i, action_desc in enumerate(primary_template["typical_actions"]):
                # 生成行动ID
                action_id = f"{task_specification.id}_action_{i}"
                
                # 确定行动精度
                precision = primary_template.get("typical_precision", ActionPrecision.MEDIUM)
                
                # 估计持续时间
                duration_range = primary_template.get("estimated_duration_range", (5.0, 30.0))
                estimated_duration = (duration_range[0] + duration_range[1]) / 2
                
                # 根据任务复杂度调整
                if hasattr(task_specification, 'complexity'):
                    if task_specification.complexity.name in ["COMPLEX", "VERY_COMPLEX"]:
                        estimated_duration *= 1.5
                
                # 限制持续时间在配置范围内
                min_duration = self.config.get("min_action_duration_minutes", 1.0)
                max_duration = self.config.get("max_action_duration_minutes", 480.0)
                estimated_duration = max(min_duration, min(max_duration, estimated_duration))
                
                # 估算资源需求
                required_resources = self._estimate_action_resources(
                    action_desc, primary_template, estimated_duration, context
                )
                
                # 构建依赖关系（线性依赖）
                prerequisites = []
                if i > 0:
                    prev_action_id = f"{task_specification.id}_action_{i-1}"
                    prerequisites.append(prev_action_id)
                
                # 定义参数
                parameters = self._define_action_parameters(action_desc, task_specification, context)
                
                # 定义监控点
                monitoring_points = self._define_monitoring_points(action_desc, estimated_duration)
                
                # 定义调整规则
                adjustment_rules = self._define_adjustment_rules(action_desc, task_specification)
                
                # 创建行动规格
                action_spec = ActionSpecification(
                    id=action_id,
                    description=f"{action_desc}: {task_specification.description[:50]}",
                    task_specification_id=task_specification.id,
                    action_type=primary_template["action_type"],
                    precision=precision,
                    parameters=parameters,
                    estimated_duration=estimated_duration,
                    prerequisites=prerequisites,
                    required_resources=required_resources,
                    success_criteria=primary_template.get("success_criteria", []),
                    failure_conditions=self._define_failure_conditions(action_desc, task_specification),
                    monitoring_points=monitoring_points,
                    adjustment_rules=adjustment_rules,
                    safety_checks=primary_template.get("safety_checks", []),
                    metadata={
                        "template_name": primary_template["name"],
                        "action_index": i,
                        "generated_at": datetime.now().isoformat()
                    }
                )
                
                action_specifications.append(action_spec)
            
            # 限制行动数量
            max_actions = self.config.get("max_actions_per_task", 30)
            if len(action_specifications) > max_actions:
                action_specifications = action_specifications[:max_actions]
                logger.warning(f"任务 {task_specification.id} 的行动数量限制为 {max_actions}")
            
            # 更新性能统计
            decomposition_time = time.time() - start_time
            self._update_performance_stats("actions_generated", decomposition_time, count=len(action_specifications))
            
            logger.info(f"任务分解完成，生成 {len(action_specifications)} 个行动，用时: {decomposition_time:.2f} 秒")
            
            return action_specifications
            
        except Exception as e:
            logger.error(f"任务分解失败: {e}")
            if error_handler:
                error_handler.handle_error(e, "OperationalPlanner", "任务分解失败")
            return []
    
    def _generate_complete_action_sequence(self,
                                         task_specification: TaskSpecification,
                                         context: OperationalContext) -> List[ActionSpecification]:
        """生成完整的行动序列（包含所有类型的行动）"""
        action_specifications = []
        action_index = 0
        
        # 生成准备行动
        prep_template = self.action_templates["preparation_actions"]
        for i, action_desc in enumerate(prep_template["typical_actions"][:2]):  # 取前2个准备行动
            action_spec = self._create_action_from_template(
                action_desc, prep_template, task_specification, context, action_index
            )
            if action_spec:
                action_specifications.append(action_spec)
                action_index += 1
        
        # 生成执行行动
        exec_template = self.action_templates["execution_actions"]
        for i, action_desc in enumerate(exec_template["typical_actions"][:3]):  # 取前3个执行行动
            action_spec = self._create_action_from_template(
                action_desc, exec_template, task_specification, context, action_index
            )
            if action_spec:
                action_specifications.append(action_spec)
                action_index += 1
        
        # 生成监控行动
        monitor_template = self.action_templates["monitoring_actions"]
        for i, action_desc in enumerate(monitor_template["typical_actions"][:2]):  # 取前2个监控行动
            action_spec = self._create_action_from_template(
                action_desc, monitor_template, task_specification, context, action_index
            )
            if action_spec:
                action_specifications.append(action_spec)
                action_index += 1
        
        # 生成验证行动
        verify_template = self.action_templates["verification_actions"]
        for i, action_desc in enumerate(verify_template["typical_actions"][:2]):  # 取前2个验证行动
            action_spec = self._create_action_from_template(
                action_desc, verify_template, task_specification, context, action_index
            )
            if action_spec:
                action_specifications.append(action_spec)
                action_index += 1
        
        # 生成完成行动
        completion_template = self.action_templates["completion_actions"]
        for i, action_desc in enumerate(completion_template["typical_actions"][:2]):  # 取前2个完成行动
            action_spec = self._create_action_from_template(
                action_desc, completion_template, task_specification, context, action_index
            )
            if action_spec:
                action_specifications.append(action_spec)
                action_index += 1
        
        return action_specifications
    
    def _create_action_from_template(self,
                                   action_desc: str,
                                   template: Dict[str, Any],
                                   task_specification: TaskSpecification,
                                   context: OperationalContext,
                                   action_index: int) -> Optional[ActionSpecification]:
        """从模板创建行动规格"""
        try:
            action_id = f"{task_specification.id}_action_{action_index}"
            
            # 估计持续时间
            duration_range = template.get("estimated_duration_range", (5.0, 30.0))
            estimated_duration = (duration_range[0] + duration_range[1]) / 2
            
            # 限制持续时间
            min_duration = self.config.get("min_action_duration_minutes", 1.0)
            max_duration = self.config.get("max_action_duration_minutes", 480.0)
            estimated_duration = max(min_duration, min(max_duration, estimated_duration))
            
            # 估算资源需求
            required_resources = {}
            template_resources = template.get("required_resources", {})
            for resource, amount in template_resources.items():
                required_resources[resource] = amount * estimated_duration / 60.0  # 转换为小时
            
            # 构建依赖关系
            prerequisites = []
            if action_index > 0:
                prev_action_id = f"{task_specification.id}_action_{action_index-1}"
                prerequisites.append(prev_action_id)
            
            # 创建行动规格
            action_spec = ActionSpecification(
                id=action_id,
                description=f"{action_desc}: {task_specification.description[:50]}",
                task_specification_id=task_specification.id,
                action_type=template["action_type"],
                precision=template.get("typical_precision", ActionPrecision.MEDIUM),
                parameters={"template_based": True, "action_desc": action_desc},
                estimated_duration=estimated_duration,
                prerequisites=prerequisites,
                required_resources=required_resources,
                success_criteria=template.get("success_criteria", []),
                failure_conditions=["超时", "资源不足", "参数错误"],
                monitoring_points=[{"point": "midpoint", "check": "进度检查"}],
                adjustment_rules=[{"condition": "进度滞后", "action": "加快执行"}],
                safety_checks=template.get("safety_checks", []),
                metadata={
                    "template_name": template["name"],
                    "action_index": action_index,
                    "generated_at": datetime.now().isoformat()
                }
            )
            
            return action_spec
            
        except Exception as e:
            logger.error(f"从模板创建行动失败 {action_desc}: {e}")
            return None
    
    def _estimate_action_resources(self,
                                 action_desc: str,
                                 template: Dict[str, Any],
                                 estimated_duration: float,
                                 context: OperationalContext) -> Dict[str, float]:
        """估算行动资源需求"""
        resources = {}
        
        # 基于模板的基础资源
        template_resources = template.get("required_resources", {})
        for resource, amount in template_resources.items():
            resources[resource] = amount * estimated_duration / 60.0  # 转换为小时
        
        # 根据行动描述添加特定资源
        desc_lower = action_desc.lower()
        
        if any(keyword in desc_lower for keyword in ["计算", "处理", "分析", "compute", "process", "analyze"]):
            resources["computational"] = resources.get("computational", 0.0) + estimated_duration / 30.0
        
        if any(keyword in desc_lower for keyword in ["监控", "检查", "观察", "monitor", "check", "observe"]):
            resources["monitoring"] = resources.get("monitoring", 0.0) + estimated_duration / 60.0
        
        if any(keyword in desc_lower for keyword in ["验证", "测试", "确认", "verify", "test", "confirm"]):
            resources["verification"] = resources.get("verification", 0.0) + estimated_duration / 40.0
        
        # 添加时间资源
        resources["time"] = estimated_duration / 60.0  # 转换为小时
        
        return resources
    
    def _define_action_parameters(self,
                                action_desc: str,
                                task_specification: TaskSpecification,
                                context: OperationalContext) -> Dict[str, Any]:
        """定义行动参数"""
        parameters = {
            "action_description": action_desc,
            "task_id": task_specification.id,
            "task_priority": task_specification.priority.name if hasattr(task_specification, 'priority') else "medium",
            "generated_at": datetime.now().isoformat(),
            "context_state": context.current_state.get("status", "unknown")
        }
        
        # 根据行动类型添加特定参数
        desc_lower = action_desc.lower()
        
        if any(keyword in desc_lower for keyword in ["准备", "配置", "setup", "prepare"]):
            parameters["preparation_type"] = "standard"
            parameters["checklist_required"] = True
        
        elif any(keyword in desc_lower for keyword in ["执行", "运行", "execute", "run"]):
            parameters["execution_mode"] = "normal"
            parameters["quality_control"] = True
            parameters["progress_tracking"] = True
        
        elif any(keyword in desc_lower for keyword in ["监控", "检查", "monitor", "check"]):
            parameters["monitoring_frequency"] = "continuous"
            parameters["alert_enabled"] = True
            parameters["logging_enabled"] = True
        
        return parameters
    
    def _define_monitoring_points(self,
                                action_desc: str,
                                estimated_duration: float) -> List[Dict[str, Any]]:
        """定义监控点"""
        monitoring_points = []
        
        # 基于持续时间定义监控点
        if estimated_duration >= 30.0:  # 30分钟以上
            # 开始点
            monitoring_points.append({
                "point": "start",
                "time_offset": 0.0,
                "check_type": "initial",
                "checks": ["资源就绪", "参数正确", "环境正常"]
            })
            
            # 中间点（如果时间足够）
            if estimated_duration >= 60.0:
                monitoring_points.append({
                    "point": "midpoint",
                    "time_offset": estimated_duration / 2,
                    "check_type": "progress",
                    "checks": ["进度正常", "质量合格", "无异常"]
                })
            
            # 结束点
            monitoring_points.append({
                "point": "end",
                "time_offset": estimated_duration,
                "check_type": "completion",
                "checks": ["任务完成", "结果正确", "质量达标"]
            })
        
        else:  # 短时间行动
            monitoring_points.append({
                "point": "completion",
                "time_offset": estimated_duration,
                "check_type": "final",
                "checks": ["行动完成", "结果符合预期"]
            })
        
        return monitoring_points
    
    def _define_adjustment_rules(self,
                               action_desc: str,
                               task_specification: TaskSpecification) -> List[Dict[str, Any]]:
        """定义调整规则"""
        adjustment_rules = []
        
        # 通用调整规则
        adjustment_rules.append({
            "condition": "进度滞后超过20%",
            "action": "加快执行速度",
            "parameters": {"speed_increase": 1.2},
            "priority": "medium"
        })
        
        adjustment_rules.append({
            "condition": "质量不达标",
            "action": "增加质量控制点",
            "parameters": {"additional_checks": 2},
            "priority": "high"
        })
        
        adjustment_rules.append({
            "condition": "资源不足",
            "action": "重新分配资源",
            "parameters": {"resource_reallocation": True},
            "priority": "high"
        })
        
        # 根据行动类型添加特定规则
        desc_lower = action_desc.lower()
        
        if any(keyword in desc_lower for keyword in ["执行", "运行", "execute", "run"]):
            adjustment_rules.append({
                "condition": "执行错误或异常",
                "action": "执行错误恢复程序",
                "parameters": {"recovery_procedure": "standard"},
                "priority": "critical"
            })
        
        return adjustment_rules
    
    def _define_failure_conditions(self,
                                 action_desc: str,
                                 task_specification: TaskSpecification) -> List[str]:
        """定义失败条件"""
        failure_conditions = [
            "超时未完成",
            "资源耗尽",
            "严重错误发生",
            "安全违规",
            "质量严重不达标"
        ]
        
        # 根据任务优先级添加条件
        if hasattr(task_specification, 'priority'):
            if task_specification.priority.name in ["CRITICAL", "HIGH"]:
                failure_conditions.append("任何进度偏差影响最终期限")
        
        return failure_conditions
    
    def create_execution_plan(self,
                            context: OperationalContext,
                            task_specification: Optional[TaskSpecification] = None,
                            action_specifications: Optional[List[ActionSpecification]] = None) -> ExecutionPlan:
        """
        创建执行计划
        
        Args:
            context: 操作上下文
            task_specification: 任务规格（可选，如未提供使用上下文中的规格）
            action_specifications: 行动规格（可选，如未提供将自动分解）
            
        Returns:
            执行计划
        """
        start_time = time.time()
        
        logger.info("开始创建执行计划...")
        
        try:
            # 获取任务规格
            if task_specification is None:
                if context.task_specification is None:
                    raise ValueError("未提供任务规格且上下文中无任务规格")
                task_specification = context.task_specification
            
            # 生成行动规格（如未提供）
            if action_specifications is None:
                action_specifications = self.decompose_task_to_actions(task_specification, context)
            
            # 优化资源分配
            resource_allocation = self.optimize_resource_allocation(action_specifications, context)
            
            # 制定时间安排
            schedule = self.create_schedule(action_specifications, context)
            
            # 制定监控计划
            monitoring_plan = self.create_monitoring_plan(action_specifications, context)
            
            # 制定调整策略
            adjustment_strategies = self.create_adjustment_strategies(action_specifications, context)
            
            # 制定应急计划
            contingency_plans = self.create_contingency_plans(action_specifications, context)
            
            # 定义性能目标
            performance_targets = self.define_performance_targets(action_specifications)
            
            # 创建执行计划
            plan_id = f"execution_plan_{int(time.time())}_{(zlib.adler32(str(task_specification.id).encode('utf-8')) & 0xffffffff) % 10000}"
            
            execution_plan = ExecutionPlan(
                id=plan_id,
                name=f"执行计划 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                description=f"基于任务 '{task_specification.description[:50]}...' 的执行计划",
                task_specification_id=task_specification.id,
                action_sequence=action_specifications,
                resource_allocation=resource_allocation,
                schedule=schedule,
                monitoring_plan=monitoring_plan,
                adjustment_strategies=adjustment_strategies,
                contingency_plans=contingency_plans,
                performance_targets=performance_targets
            )
            
            # 更新当前计划和历史记录
            self.current_execution_plan = execution_plan
            self.execution_plans_history.append(execution_plan)
            
            # 限制历史记录大小
            if len(self.execution_plans_history) > self.max_history_size:
                self.execution_plans_history = self.execution_plans_history[-self.max_history_size:]
            
            # 更新性能统计
            execution_time = time.time() - start_time
            self._update_performance_stats("execution_plans_created", execution_time)
            
            logger.info(f"执行计划创建完成，ID: {plan_id}，用时: {execution_time:.2f} 秒")
            
            return execution_plan
            
        except Exception as e:
            logger.error(f"创建执行计划失败: {e}")
            if error_handler:
                error_handler.handle_error(e, "OperationalPlanner", "创建执行计划失败")
            
            # 返回基本的执行计划
            return ExecutionPlan(
                id=f"failed_plan_{int(time.time())}",
                name="失败的执行计划",
                description="计划创建失败",
                task_specification_id=context.task_specification.id if context.task_specification else "unknown",
                action_sequence=[],
                resource_allocation={},
                schedule={},
                monitoring_plan={},
                adjustment_strategies=[],
                contingency_plans=[],
                performance_targets={}
            )
    
    def optimize_resource_allocation(self,
                                   action_specifications: List[ActionSpecification],
                                   context: OperationalContext) -> Dict[str, Dict[str, float]]:
        """
        优化资源分配
        
        Args:
            action_specifications: 行动规格列表
            context: 操作上下文
            
        Returns:
            优化的资源分配
        """
        logger.info("开始优化资源分配...")
        
        try:
            allocation = {}
            
            # 简单的资源分配算法：基于行动顺序和可用资源
            available_resources = context.available_resources.copy()
            
            for action in action_specifications:
                action_allocation = {}
                
                # 尝试分配资源
                for resource, requirement in action.required_resources.items():
                    resource_key = resource if isinstance(resource, str) else resource.value
                    
                    if resource_key in available_resources:
                        # 分配资源，但不超过可用量
                        allocated = min(requirement, available_resources[resource_key])
                        action_allocation[resource_key] = allocated
                        available_resources[resource_key] -= allocated
                    else:
                        # 资源类型不可用，分配0
                        action_allocation[resource_key] = 0.0
                
                allocation[action.id] = action_allocation
            
            logger.info(f"资源分配优化完成，分配了 {len(allocation)} 个行动")
            return allocation
            
        except Exception as e:
            logger.error(f"资源分配优化失败: {e}")
            # 返回简单的平均分配
            allocation = {}
            num_actions = len(action_specifications)
            if num_actions == 0:
                num_actions = 1
            
            for action in action_specifications:
                action_allocation = {}
                for resource, available in context.available_resources.items():
                    action_allocation[resource] = available / num_actions
                allocation[action.id] = action_allocation
            
            return allocation
    
    def create_schedule(self,
                       action_specifications: List[ActionSpecification],
                       context: OperationalContext) -> Dict[str, Any]:
        """创建时间安排"""
        schedule = {
            "planning_method": "sequential_scheduling",
            "start_time": datetime.now().isoformat(),
            "total_duration_minutes": 0.0,
            "actions_schedule": [],
            "dependencies_considered": True,
            "resource_constraints_considered": True,
            "buffer_time_included": True
        }
        
        # 简单的时间安排算法（顺序执行）
        current_time = datetime.now()
        total_duration = 0.0
        
        for action in action_specifications:
            # 计算开始时间（考虑前置依赖）
            start_time = current_time
            
            # 计算结束时间
            end_time = start_time + timedelta(minutes=action.estimated_duration)
            
            # 更新总持续时间
            total_duration += action.estimated_duration
            
            schedule["actions_schedule"].append({
                "action_id": action.id,
                "action_description": action.description[:50],
                "action_type": action.action_type.value,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_minutes": action.estimated_duration,
                "prerequisites": action.prerequisites,
                "precision": action.precision.value
            })
            
            # 更新当前时间（顺序执行）
            current_time = end_time + timedelta(minutes=1.0)  # 1分钟间隔
        
        schedule["total_duration_minutes"] = total_duration
        schedule["estimated_completion_time"] = current_time.isoformat()
        
        # 添加缓冲时间
        buffer_time = total_duration * 0.1  # 10%缓冲
        schedule["buffer_time_minutes"] = buffer_time
        schedule["total_with_buffer_minutes"] = total_duration + buffer_time
        
        return schedule
    
    def create_monitoring_plan(self,
                             action_specifications: List[ActionSpecification],
                             context: OperationalContext) -> Dict[str, Any]:
        """创建监控计划"""
        monitoring_plan = {
            "monitoring_strategy": "adaptive_monitoring",
            "monitoring_frequency_seconds": self.config.get("monitoring_frequency_seconds", 60.0),
            "monitored_actions": [],
            "alert_configuration": {
                "enabled": True,
                "channels": ["log", "notification"],
                "thresholds": {
                    "progress_deviation": 0.2,  # 20%
                    "quality_deviation": 0.1,   # 10%
                    "resource_usage": 0.9,      # 90%
                    "time_deviation": 0.15      # 15%
                }
            },
            "logging_configuration": {
                "enabled": True,
                "level": "detailed",
                "retention_days": 30
            }
        }
        
        # 为每个行动添加监控配置
        for action in action_specifications:
            action_monitoring = {
                "action_id": action.id,
                "monitoring_points": action.monitoring_points,
                "check_frequency": "continuous" if action.action_type == ActionType.MONITORING else "periodic",
                "metrics_to_monitor": ["progress", "quality", "resource_usage", "errors"],
                "alert_rules": [
                    {"condition": "progress < 50% at midpoint", "severity": "warning"},
                    {"condition": "quality < acceptable", "severity": "error"},
                    {"condition": "resource_exhausted", "severity": "critical"},
                    {"condition": "safety_violation", "severity": "critical"}
                ]
            }
            monitoring_plan["monitored_actions"].append(action_monitoring)
        
        return monitoring_plan
    
    def create_adjustment_strategies(self,
                                   action_specifications: List[ActionSpecification],
                                   context: OperationalContext) -> List[Dict[str, Any]]:
        """制定调整策略"""
        adjustment_strategies = []
        
        # 通用调整策略
        adjustment_strategies.append({
            "strategy_name": "进度滞后调整",
            "trigger_condition": "实际进度比计划滞后超过20%",
            "adjustment_actions": [
                "增加资源投入",
                "优化执行流程",
                "延长工作时间",
                "调整后续行动优先级"
            ],
            "evaluation_criteria": ["进度恢复", "质量保持", "资源效率"],
            "monitoring_requirements": ["进度监控", "资源监控", "质量监控"]
        })
        
        adjustment_strategies.append({
            "strategy_name": "资源不足调整",
            "trigger_condition": "关键资源使用率超过90%",
            "adjustment_actions": [
                "资源重新分配",
                "寻找替代资源",
                "调整行动顺序",
                "降低非关键行动资源"
            ],
            "evaluation_criteria": ["资源平衡", "进度影响", "质量影响"],
            "monitoring_requirements": ["资源监控", "进度监控"]
        })
        
        adjustment_strategies.append({
            "strategy_name": "质量不达标调整",
            "trigger_condition": "质量指标低于标准10%",
            "adjustment_actions": [
                "增加质量检查点",
                "加强过程控制",
                "培训操作人员",
                "调整执行参数"
            ],
            "evaluation_criteria": ["质量改进", "进度影响", "成本影响"],
            "monitoring_requirements": ["质量监控", "过程监控"]
        })
        
        # 为高风险行动添加特定策略
        for action in action_specifications:
            if action.precision == ActionPrecision.VERY_HIGH or action.action_type == ActionType.EXECUTION:
                adjustment_strategies.append({
                    "strategy_name": f"{action.id}_精细调整",
                    "trigger_condition": f"{action.id}执行偏差超过5%",
                    "adjustment_actions": [
                        "精细参数调整",
                        "增加校准频率",
                        "加强实时监控",
                        "准备备用方案"
                    ],
                    "evaluation_criteria": ["精度恢复", "进度保持", "质量稳定"],
                    "monitoring_requirements": ["实时监控", "精度监控", "质量监控"]
                })
        
        return adjustment_strategies
    
    def create_contingency_plans(self,
                               action_specifications: List[ActionSpecification],
                               context: OperationalContext) -> List[Dict[str, Any]]:
        """制定应急计划"""
        contingency_plans = []
        
        # 为关键行动创建应急计划
        for action in action_specifications:
            if action.action_type == ActionType.EXECUTION or action.precision == ActionPrecision.VERY_HIGH:
                plan = {
                    "action_id": action.id,
                    "failure_scenarios": [
                        "执行失败",
                        "资源故障",
                        "环境异常",
                        "安全事件"
                    ],
                    "recovery_actions": [
                        "执行备用方案",
                        "切换备用资源",
                        "调整执行环境",
                        "启动安全协议"
                    ],
                    "escalation_procedure": "立即上报并启动应急响应",
                    "recovery_time_target": f"{min(30.0, action.estimated_duration * 0.5)}分钟",
                    "communication_protocol": "实时通报，每小时总结"
                }
                contingency_plans.append(plan)
        
        return contingency_plans
    
    def define_performance_targets(self,
                                 action_specifications: List[ActionSpecification]) -> Dict[str, Any]:
        """定义性能目标"""
        performance_targets = {
            "schedule_performance": {
                "description": "时间表遵守率",
                "measurement": "按时完成行动数 / 总行动数",
                "target": ">90%"
            },
            "resource_efficiency": {
                "description": "资源效率",
                "measurement": "实际资源使用 / 计划资源",
                "target": "85-95%"
            },
            "quality_performance": {
                "description": "质量性能",
                "measurement": "质量达标行动数 / 总行动数",
                "target": ">95%"
            },
            "adjustment_effectiveness": {
                "description": "调整效果",
                "measurement": "成功调整次数 / 总调整次数",
                "target": ">80%"
            },
            "recovery_performance": {
                "description": "恢复性能",
                "measurement": "成功恢复次数 / 总失败次数",
                "target": ">90%"
            },
            "safety_performance": {
                "description": "安全性能",
                "measurement": "安全违规次数",
                "target": "0"
            }
        }
        
        return performance_targets
    
    def _update_performance_stats(self, stat_type: str, execution_time: float, count: int = 1):
        """更新性能统计"""
        if stat_type == "execution_plans_created":
            self.performance_stats["execution_plans_created"] += 1
        elif stat_type == "actions_generated":
            self.performance_stats["actions_generated"] += count
        elif stat_type == "executions_monitored":
            self.performance_stats["executions_monitored"] += count
        elif stat_type == "adjustments_made":
            self.performance_stats["adjustments_made"] += 1
        
        # 更新时间统计
        self.performance_stats["total_planning_time"] += execution_time
        self.performance_stats["last_planning_time"] = execution_time
        
        # 计算平均时间
        total_actions = (self.performance_stats["execution_plans_created"] +
                        self.performance_stats["actions_generated"] // 10 +  # 每10个行动算一次行动
                        self.performance_stats["executions_monitored"] // 20 +  # 每20个监控算一次行动
                        self.performance_stats["adjustments_made"])
        
        if total_actions > 0:
            self.performance_stats["average_planning_time"] = (
                self.performance_stats["total_planning_time"] / total_actions
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "performance_stats": self.performance_stats.copy(),
            "current_plan": self.current_execution_plan.id if self.current_execution_plan else None,
            "plans_in_history": len(self.execution_plans_history),
            "action_templates_available": len(self.action_templates),
            "parameter_optimizers": len(self.parameter_optimizers),
            "monitoring_configurations": len(self.monitoring_configurations),
            "config_summary": {
                "enable_action_decomposition": self.config.get("enable_action_decomposition", True),
                "enable_realtime_monitoring": self.config.get("enable_realtime_monitoring", True),
                "enable_dynamic_adjustment": self.config.get("enable_dynamic_adjustment", True),
                "max_actions_per_task": self.config.get("max_actions_per_task", 30),
                "safety_check_enabled": self.config.get("safety_check_enabled", True)
            }
        }
        
        return report