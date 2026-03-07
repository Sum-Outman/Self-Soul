"""
自主性系统

为统一认知架构实现自主意识功能，包括内在动机驱动、自主目标生成和好奇心驱动探索。
基于原有Self-Soul系统的自主性系统功能，提供真正的自主决策和行为驱动能力。
"""

import torch
import torch.nn as nn
import asyncio
import logging
import time
import random
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class IntrinsicMotivationType(Enum):
    """内在动机类型枚举"""
    CURIOSITY = "curiosity"           # 好奇心驱动
    COMPETENCE = "competence"         # 能力提升驱动
    KNOWLEDGE_COMPLETENESS = "knowledge_completeness"  # 知识完整性驱动
    EXPLORATION = "exploration"       # 探索驱动
    MASTERY = "mastery"               # 精通驱动
    NOVELTY = "novelty"               # 新奇性驱动


class GoalType(Enum):
    """目标类型枚举"""
    EXPLORATORY = "exploratory"       # 探索性目标
    EXPLOITATIVE = "exploitative"     # 利用性目标
    LEARNING = "learning"             # 学习目标
    OPTIMIZATION = "optimization"     # 优化目标
    MAINTENANCE = "maintenance"       # 维护目标
    CREATIVE = "creative"             # 创造性目标


@dataclass
class IntrinsicMotivation:
    """内在动机数据结构"""
    motivation_type: IntrinsicMotivationType  # 动机类型
    strength: float                           # 动机强度 (0-1)
    satisfaction: float                       # 满意度 (0-1)
    last_activation: float                    # 最后激活时间
    activation_count: int                     # 激活次数
    parameters: Dict[str, Any] = field(default_factory=dict)  # 动机参数


@dataclass
class AutonomousGoal:
    """自主目标数据结构"""
    goal_id: str                              # 目标ID
    goal_type: GoalType                       # 目标类型
    description: str                          # 目标描述
    motivation_source: IntrinsicMotivationType  # 动机来源
    priority: float                           # 优先级 (0-1)
    feasibility: float                        # 可行性 (0-1)
    expected_value: float                     # 预期价值 (0-1)
    creation_time: float                      # 创建时间
    deadline: Optional[float] = None          # 截止时间
    constraints: Dict[str, Any] = field(default_factory=dict)  # 约束条件
    progress: float = 0.0                     # 进度 (0-1)
    status: str = "pending"                   # 状态


@dataclass
class ExplorationState:
    """探索状态数据结构"""
    novelty_score: float                      # 新奇性评分 (0-1)
    uncertainty: float                        # 不确定性 (0-1)
    information_gain: float                   # 信息增益 (0-1)
    coverage: float                           # 覆盖度 (0-1)
    last_exploration: float                   # 最后探索时间


class AutonomousSystem:
    """自主性系统"""
    
    def __init__(self, communication):
        """
        初始化自主性系统。
        
        参数:
            communication: 神经通信系统
        """
        self.communication = communication
        self.initialized = False
        
        # 内在动机
        self.intrinsic_motivations: Dict[IntrinsicMotivationType, IntrinsicMotivation] = {}
        
        # 自主目标
        self.active_goals: Dict[str, AutonomousGoal] = {}
        self.completed_goals: Dict[str, AutonomousGoal] = {}
        self.failed_goals: Dict[str, AutonomousGoal] = {}
        
        # 探索状态
        self.exploration_state = ExplorationState(
            novelty_score=0.5,
            uncertainty=0.8,
            information_gain=0.0,
            coverage=0.1,
            last_exploration=time.time()
        )
        
        # 配置参数
        self.config = {
            'goal_generation_interval': 60.0,      # 目标生成间隔 (秒)
            'max_active_goals': 5,                 # 最大活跃目标数
            'motivation_decay_rate': 0.95,         # 动机衰减率
            'novelty_threshold': 0.7,              # 新奇性阈值
            'uncertainty_threshold': 0.6,          # 不确定性阈值
            'exploration_exploitation_balance': 0.5,  # 探索-利用平衡
            'goal_success_threshold': 0.8,         # 目标成功阈值
            'learning_rate': 0.1,                  # 学习率
        }
        
        # 知识状态
        self.knowledge_state = {
            'knowledge_gaps': [],                  # 知识缺口
            'competence_levels': {},               # 能力水平
            'familiarity_scores': {},              # 熟悉度评分
            'mastery_areas': []                    # 精通领域
        }
        
        # 性能指标
        self.performance_metrics = {
            'total_goals_generated': 0,
            'total_goals_completed': 0,
            'total_goals_failed': 0,
            'goal_success_rate': 0.0,
            'average_goal_duration': 0.0,
            'motivation_satisfaction': 0.5,
            'autonomy_level': 0.5,
            'exploration_efficiency': 0.0
        }
        
        # 监控任务
        self.monitoring_task = None
        self.monitoring_active = False
        
        logger.info("自主性系统已初始化")
    
    async def initialize(self):
        """初始化自主性系统"""
        if self.initialized:
            return
        
        logger.info("初始化自主性系统...")
        
        # 初始化内在动机
        await self._initialize_intrinsic_motivations()
        
        # 初始化知识状态
        await self._initialize_knowledge_state()
        
        # 在通信系统中注册
        await self.communication.register_component(
            component_name="autonomy",
            component_type="cognitive"
        )
        
        # 启动监控任务
        await self._start_monitoring()
        
        self.initialized = True
        logger.info("自主性系统初始化完成")
    
    async def _initialize_intrinsic_motivations(self):
        """初始化内在动机"""
        try:
            current_time = time.time()
            
            # 好奇心动机
            self.intrinsic_motivations[IntrinsicMotivationType.CURIOSITY] = IntrinsicMotivation(
                motivation_type=IntrinsicMotivationType.CURIOSITY,
                strength=0.8,
                satisfaction=0.3,  # 初始满意度低，驱动探索
                last_activation=current_time,
                activation_count=0,
                parameters={
                    'novelty_weight': 0.7,
                    'uncertainty_weight': 0.3,
                    'information_gain_weight': 0.5
                }
            )
            
            # 能力提升动机
            self.intrinsic_motivations[IntrinsicMotivationType.COMPETENCE] = IntrinsicMotivation(
                motivation_type=IntrinsicMotivationType.COMPETENCE,
                strength=0.6,
                satisfaction=0.5,
                last_activation=current_time,
                activation_count=0,
                parameters={
                    'skill_improvement_weight': 0.8,
                    'efficiency_weight': 0.2,
                    'mastery_threshold': 0.9
                }
            )
            
            # 知识完整性动机
            self.intrinsic_motivations[IntrinsicMotivationType.KNOWLEDGE_COMPLETENESS] = IntrinsicMotivation(
                motivation_type=IntrinsicMotivationType.KNOWLEDGE_COMPLETENESS,
                strength=0.7,
                satisfaction=0.4,
                last_activation=current_time,
                activation_count=0,
                parameters={
                    'coverage_weight': 0.6,
                    'consistency_weight': 0.3,
                    'explanatory_power_weight': 0.1
                }
            )
            
            # 探索动机
            self.intrinsic_motivations[IntrinsicMotivationType.EXPLORATION] = IntrinsicMotivation(
                motivation_type=IntrinsicMotivationType.EXPLORATION,
                strength=0.9,
                satisfaction=0.2,  # 初始满意度低，驱动探索
                last_activation=current_time,
                activation_count=0,
                parameters={
                    'unknown_area_weight': 0.8,
                    'boundary_weight': 0.2,
                    'risk_tolerance': 0.3
                }
            )
            
            logger.info(f"初始化了 {len(self.intrinsic_motivations)} 种内在动机")
            
        except Exception as e:
            logger.error(f"初始化内在动机失败: {e}")
    
    async def _initialize_knowledge_state(self):
        """初始化知识状态"""
        try:
            # 初始知识状态
            self.knowledge_state = {
                'knowledge_gaps': [
                    {'id': 'gap_001', 'description': '未知环境特征', 'importance': 0.8},
                    {'id': 'gap_002', 'description': '未掌握的复杂技能', 'importance': 0.7},
                    {'id': 'gap_003', 'description': '新任务类型', 'importance': 0.6}
                ],
                'competence_levels': {
                    'basic_mobility': 0.6,
                    'object_manipulation': 0.4,
                    'environment_navigation': 0.5,
                    'task_planning': 0.7,
                    'social_interaction': 0.3
                },
                'familiarity_scores': {
                    'home_environment': 0.8,
                    'office_environment': 0.4,
                    'outdoor_environment': 0.2,
                    'social_situations': 0.3
                },
                'mastery_areas': ['basic_mobility', 'task_planning']
            }
            
            logger.info("知识状态初始化完成")
            
        except Exception as e:
            logger.error(f"初始化知识状态失败: {e}")
    
    async def _start_monitoring(self):
        """启动监控任务"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("自主性监控任务已启动")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 更新动机状态
                await self._update_motivation_states()
                
                # 检查目标状态
                await self._check_goal_progress()
                
                # 生成新目标（如果需要）
                await self._generate_autonomous_goals()
                
                # 更新探索状态
                await self._update_exploration_state()
                
                # 更新性能指标
                await self._update_performance_metrics()
                
                # 等待下一个周期
                await asyncio.sleep(self.config['goal_generation_interval'])
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(10.0)
    
    async def _update_motivation_states(self):
        """更新动机状态"""
        current_time = time.time()
        
        for motivation_type, motivation in self.intrinsic_motivations.items():
            try:
                # 计算时间衰减
                time_since_last = current_time - motivation.last_activation
                decay_factor = self.config['motivation_decay_rate'] ** (time_since_last / 3600.0)  # 按小时衰减
                
                # 更新动机强度
                motivation.strength = max(0.1, motivation.strength * decay_factor)
                
                # 如果满意度低，增加动机强度
                if motivation.satisfaction < 0.3:
                    motivation.strength = min(1.0, motivation.strength + 0.1)
                
                # 记录更新
                motivation.last_activation = current_time
                
            except Exception as e:
                logger.error(f"更新动机 {motivation_type.value} 状态失败: {e}")
    
    async def _check_goal_progress(self):
        """检查目标进度"""
        current_time = time.time()
        goals_to_remove = []
        
        for goal_id, goal in self.active_goals.items():
            try:
                # 检查是否超时
                if goal.deadline and current_time > goal.deadline:
                    goal.status = "failed"
                    self.failed_goals[goal_id] = goal
                    goals_to_remove.append(goal_id)
                    logger.warning(f"目标 {goal_id} 超时失败")
                    continue
                
                # 更新进度（简化实现）
                # 实际系统应该根据实际进展更新进度
                if goal.status == "in_progress":
                    # 模拟进度更新
                    time_elapsed = current_time - goal.creation_time
                    expected_duration = 300.0  # 默认5分钟
                    goal.progress = min(1.0, time_elapsed / expected_duration)
                    
                    # 检查是否完成
                    if goal.progress >= 1.0:
                        goal.status = "completed"
                        goal.progress = 1.0
                        self.completed_goals[goal_id] = goal
                        goals_to_remove.append(goal_id)
                        logger.info(f"目标 {goal_id} 完成")
                        
                        # 更新动机满意度
                        await self._update_motivation_satisfaction(goal.motivation_source, True)
                
            except Exception as e:
                logger.error(f"检查目标 {goal_id} 进度失败: {e}")
        
        # 移除已完成或失败的目标
        for goal_id in goals_to_remove:
            if goal_id in self.active_goals:
                del self.active_goals[goal_id]
    
    async def _generate_autonomous_goals(self):
        """生成自主目标"""
        try:
            # 检查是否达到最大活跃目标数
            if len(self.active_goals) >= self.config['max_active_goals']:
                return
            
            # 选择最强烈的动机
            strongest_motivation = max(
                self.intrinsic_motivations.values(),
                key=lambda m: m.strength * (1.0 - m.satisfaction)
            )
            
            # 根据动机类型生成目标
            goal = None
            if strongest_motivation.motivation_type == IntrinsicMotivationType.CURIOSITY:
                goal = await self._generate_curiosity_goal(strongest_motivation)
            elif strongest_motivation.motivation_type == IntrinsicMotivationType.COMPETENCE:
                goal = await self._generate_competence_goal(strongest_motivation)
            elif strongest_motivation.motivation_type == IntrinsicMotivationType.KNOWLEDGE_COMPLETENESS:
                goal = await self._generate_knowledge_goal(strongest_motivation)
            elif strongest_motivation.motivation_type == IntrinsicMotivationType.EXPLORATION:
                goal = await self._generate_exploration_goal(strongest_motivation)
            else:
                goal = await self._generate_general_goal(strongest_motivation)
            
            if goal:
                # 添加目标
                self.active_goals[goal.goal_id] = goal
                self.performance_metrics['total_goals_generated'] += 1
                
                # 激活动机
                strongest_motivation.activation_count += 1
                strongest_motivation.last_activation = time.time()
                
                logger.info(f"生成自主目标: {goal.description} (动机: {strongest_motivation.motivation_type.value})")
                
        except Exception as e:
            logger.error(f"生成自主目标失败: {e}")
    
    async def _generate_curiosity_goal(self, motivation: IntrinsicMotivation) -> AutonomousGoal:
        """生成好奇心驱动目标"""
        # 基于新奇性和不确定性生成目标
        novelty_weight = motivation.parameters.get('novelty_weight', 0.5)
        uncertainty_weight = motivation.parameters.get('uncertainty_weight', 0.3)
        
        # 计算目标优先级
        priority = (
            novelty_weight * self.exploration_state.novelty_score +
            uncertainty_weight * self.exploration_state.uncertainty
        )
        
        goal_id = f"curiosity_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return AutonomousGoal(
            goal_id=goal_id,
            goal_type=GoalType.EXPLORATORY,
            description="探索未知环境或现象以满足好奇心",
            motivation_source=IntrinsicMotivationType.CURIOSITY,
            priority=priority,
            feasibility=0.7,
            expected_value=self.exploration_state.information_gain,
            creation_time=time.time(),
            deadline=time.time() + 600.0,  # 10分钟后截止
            constraints={
                "risk_level": "medium",
                "resource_requirements": "low",
                "time_limit": 600.0
            }
        )
    
    async def _generate_competence_goal(self, motivation: IntrinsicMotivation) -> AutonomousGoal:
        """生成能力提升目标"""
        # 找到能力水平较低的技能
        lowest_competence = min(
            self.knowledge_state['competence_levels'].items(),
            key=lambda x: x[1]
        )
        
        skill_name, competence_level = lowest_competence
        improvement_needed = 1.0 - competence_level
        
        goal_id = f"competence_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return AutonomousGoal(
            goal_id=goal_id,
            goal_type=GoalType.LEARNING,
            description=f"提升 {skill_name} 技能水平 (当前: {competence_level:.2f})",
            motivation_source=IntrinsicMotivationType.COMPETENCE,
            priority=improvement_needed * motivation.strength,
            feasibility=0.8,
            expected_value=improvement_needed,
            creation_time=time.time(),
            deadline=time.time() + 1800.0,  # 30分钟后截止
            constraints={
                "skill_focus": skill_name,
                "target_level": min(1.0, competence_level + 0.3),
                "practice_intensity": "moderate"
            }
        )
    
    async def _generate_knowledge_goal(self, motivation: IntrinsicMotivation) -> AutonomousGoal:
        """生成知识完整性目标"""
        # 找到最重要的知识缺口
        if not self.knowledge_state['knowledge_gaps']:
            # 如果没有知识缺口，创建通用学习目标
            return await self._generate_general_goal(motivation)
        
        most_important_gap = max(
            self.knowledge_state['knowledge_gaps'],
            key=lambda x: x.get('importance', 0.0)
        )
        
        goal_id = f"knowledge_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return AutonomousGoal(
            goal_id=goal_id,
            goal_type=GoalType.LEARNING,
            description=f"填补知识缺口: {most_important_gap.get('description', '未知缺口')}",
            motivation_source=IntrinsicMotivationType.KNOWLEDGE_COMPLETENESS,
            priority=most_important_gap.get('importance', 0.5),
            feasibility=0.6,
            expected_value=most_important_gap.get('importance', 0.5),
            creation_time=time.time(),
            deadline=time.time() + 3600.0,  # 1小时后截止
            constraints={
                "knowledge_domain": "unknown",
                "learning_method": "research_and_experiment",
                "verification_required": True
            }
        )
    
    async def _generate_exploration_goal(self, motivation: IntrinsicMotivation) -> AutonomousGoal:
        """生成探索目标"""
        # 基于探索状态生成目标
        exploration_balance = self.config['exploration_exploitation_balance']
        
        if self.exploration_state.coverage < 0.3:
            # 低覆盖度，优先探索
            exploration_focus = "expand_coverage"
        elif self.exploration_state.novelty_score < self.config['novelty_threshold']:
            # 新奇性低，寻找新区域
            exploration_focus = "find_novel_areas"
        else:
            # 平衡探索和利用
            exploration_focus = "balanced_exploration"
        
        goal_id = f"exploration_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return AutonomousGoal(
            goal_id=goal_id,
            goal_type=GoalType.EXPLORATORY,
            description=f"探索新区域以增加环境覆盖率 (当前: {self.exploration_state.coverage:.2f})",
            motivation_source=IntrinsicMotivationType.EXPLORATION,
            priority=motivation.strength * (1.0 - self.exploration_state.coverage),
            feasibility=0.9,
            expected_value=self.exploration_state.information_gain,
            creation_time=time.time(),
            deadline=time.time() + 1200.0,  # 20分钟后截止
            constraints={
                "exploration_focus": exploration_focus,
                "safety_priority": "high",
                "mapping_required": True
            }
        )
    
    async def _generate_general_goal(self, motivation: IntrinsicMotivation) -> AutonomousGoal:
        """生成通用目标"""
        goal_id = f"general_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return AutonomousGoal(
            goal_id=goal_id,
            goal_type=GoalType.OPTIMIZATION,
            description="优化系统性能或学习新技能",
            motivation_source=motivation.motivation_type,
            priority=motivation.strength * 0.5,
            feasibility=0.7,
            expected_value=0.5,
            creation_time=time.time(),
            deadline=time.time() + 2400.0,  # 40分钟后截止
            constraints={
                "goal_type": "general_improvement",
                "flexibility": "high",
                "learning_opportunity": True
            }
        )
    
    async def _update_exploration_state(self):
        """更新探索状态"""
        try:
            current_time = time.time()
            time_since_last = current_time - self.exploration_state.last_exploration
            
            # 模拟探索状态更新
            # 实际系统应该基于实际探索数据更新
            
            # 新奇性随时间衰减
            novelty_decay = 0.99 ** (time_since_last / 60.0)  # 每分钟衰减1%
            self.exploration_state.novelty_score *= novelty_decay
            
            # 如果不确定性高，信息增益潜力大
            if self.exploration_state.uncertainty > 0.5:
                self.exploration_state.information_gain = self.exploration_state.uncertainty * 0.8
            else:
                self.exploration_state.information_gain = 0.2
            
            # 如果有探索活动，更新覆盖度
            if len(self.active_goals) > 0:
                exploration_goals = [
                    g for g in self.active_goals.values() 
                    if g.goal_type == GoalType.EXPLORATORY
                ]
                
                if exploration_goals:
                    # 假设每次探索增加覆盖度
                    coverage_increase = 0.01 * len(exploration_goals)
                    self.exploration_state.coverage = min(1.0, self.exploration_state.coverage + coverage_increase)
            
            self.exploration_state.last_exploration = current_time
            
        except Exception as e:
            logger.error(f"更新探索状态失败: {e}")
    
    async def _update_motivation_satisfaction(self, motivation_type: IntrinsicMotivationType, success: bool):
        """更新动机满意度"""
        try:
            if motivation_type not in self.intrinsic_motivations:
                return
            
            motivation = self.intrinsic_motivations[motivation_type]
            
            if success:
                # 成功增加满意度
                satisfaction_increase = 0.2
                motivation.satisfaction = min(1.0, motivation.satisfaction + satisfaction_increase)
                
                # 成功降低动机强度（需求得到满足）
                motivation.strength = max(0.1, motivation.strength * 0.8)
            else:
                # 失败降低满意度
                satisfaction_decrease = 0.1
                motivation.satisfaction = max(0.0, motivation.satisfaction - satisfaction_decrease)
                
                # 失败可能增加动机强度（需要再次尝试）
                motivation.strength = min(1.0, motivation.strength * 1.1)
            
        except Exception as e:
            logger.error(f"更新动机满意度失败: {e}")
    
    async def _update_performance_metrics(self):
        """更新性能指标"""
        try:
            # 计算目标成功率
            total_goals = (
                self.performance_metrics['total_goals_completed'] +
                self.performance_metrics['total_goals_failed']
            )
            
            if total_goals > 0:
                self.performance_metrics['goal_success_rate'] = (
                    self.performance_metrics['total_goals_completed'] / total_goals
                )
            
            # 计算动机平均满意度
            total_satisfaction = sum(m.satisfaction for m in self.intrinsic_motivations.values())
            motivation_count = len(self.intrinsic_motivations)
            
            if motivation_count > 0:
                self.performance_metrics['motivation_satisfaction'] = total_satisfaction / motivation_count
            
            # 计算自主性水平（基于目标生成和完成情况）
            autonomy_score = (
                self.performance_metrics['goal_success_rate'] * 0.4 +
                self.performance_metrics['motivation_satisfaction'] * 0.3 +
                (self.exploration_state.coverage * 0.3)
            )
            self.performance_metrics['autonomy_level'] = autonomy_score
            
            # 计算探索效率
            if self.performance_metrics['total_goals_generated'] > 0:
                exploration_goals = sum(
                    1 for g in self.completed_goals.values() 
                    if g.goal_type == GoalType.EXPLORATORY
                )
                self.performance_metrics['exploration_efficiency'] = (
                    exploration_goals / self.performance_metrics['total_goals_generated']
                )
            
        except Exception as e:
            logger.error(f"更新性能指标失败: {e}")
    
    async def get_active_goals(self) -> List[AutonomousGoal]:
        """获取活跃目标列表"""
        return list(self.active_goals.values())
    
    async def get_motivation_states(self) -> Dict[str, Dict[str, Any]]:
        """获取动机状态"""
        return {
            mt.value: {
                'strength': m.strength,
                'satisfaction': m.satisfaction,
                'activation_count': m.activation_count,
                'last_activation': m.last_activation
            }
            for mt, m in self.intrinsic_motivations.items()
        }
    
    async def get_autonomy_report(self) -> Dict[str, Any]:
        """获取自主性报告"""
        return {
            'performance_metrics': self.performance_metrics.copy(),
            'exploration_state': {
                'novelty_score': self.exploration_state.novelty_score,
                'uncertainty': self.exploration_state.uncertainty,
                'information_gain': self.exploration_state.information_gain,
                'coverage': self.exploration_state.coverage,
                'last_exploration': self.exploration_state.last_exploration
            },
            'knowledge_state_summary': {
                'knowledge_gap_count': len(self.knowledge_state['knowledge_gaps']),
                'competence_areas': len(self.knowledge_state['competence_levels']),
                'mastery_areas': len(self.knowledge_state['mastery_areas']),
                'average_competence': sum(self.knowledge_state['competence_levels'].values()) / 
                                      max(1, len(self.knowledge_state['competence_levels']))
            },
            'active_goals_count': len(self.active_goals),
            'completed_goals_count': len(self.completed_goals),
            'timestamp': time.time()
        }
    
    async def shutdown(self):
        """关闭自主性系统"""
        if not self.initialized:
            return
        
        logger.info("关闭自主性系统...")
        
        # 停止监控任务
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # 注销组件
        try:
            await self.communication.unregister_component("autonomy")
        except Exception as e:
            logger.warning(f"注销组件失败: {e}")
        
        # 清理数据
        self.active_goals.clear()
        self.completed_goals.clear()
        self.failed_goals.clear()
        self.intrinsic_motivations.clear()
        
        self.initialized = False
        logger.info("自主性系统已关闭")