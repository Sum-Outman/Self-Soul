"""
统一认知架构

核心实现替换了27模型HTTP协调系统的统一认知架构，
采用单一、连贯的认知系统。

关键组件:
- 感知: 多模态输入处理
- 注意力: 层级注意力机制
- 记忆: 情景和语义记忆系统
- 推理: 通用推理引擎
- 规划: 层级规划系统
- 决策: 价值基础决策系统
- 行动: 自适应行动执行
- 学习: 元学习系统

所有组件通过神经张量通信，而非HTTP。
"""

import torch
import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# 配置日志
logger = logging.getLogger(__name__)

# 导入认知组件
from .representation import UnifiedRepresentationSpace
from .perception import MultimodalPerception
from .attention import HierarchicalAttention
from .memory import EpisodicSemanticMemory
from .reasoning import UniversalReasoningEngine
from .planning import HierarchicalPlanning
from .decision import ValueBasedDecision
from .action import AdaptiveAction
from .learning import MetaLearningSystem

# 导入自主演化系统
try:
    from .evolution import AutonomousEvolutionSystem
    EVOLUTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"演化组件不可用: {e}")
    AutonomousEvolutionSystem = None
    EVOLUTION_AVAILABLE = False

# 导入自主性系统
try:
    from .autonomy import AutonomousSystem
    AUTONOMY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"自主性组件不可用: {e}")
    AutonomousSystem = None
    AUTONOMY_AVAILABLE = False

# 导入人形机器人AGI系统
try:
    from humanoid.humanoid_agi import HumanoidAGISystem
    HUMANOID_AGI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"人形机器人AGI组件不可用: {e}")
    HumanoidAGISystem = None
    HUMANOID_AGI_AVAILABLE = False

from .self_cognition import SelfCognitionSystem

# 导入控制组件
try:
    # 尝试不同的导入方式
    import sys
    import os
    # 添加src目录到路径
    src_dir = os.path.join(os.path.dirname(__file__), '..')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    from control.motion_control import MotionControlSystem
    from control.hardware_interface import HardwareControlSystem
    from control.sensor_integration import SensorIntegrationSystem
    from control.motor_control import MotorControlSystem
    CONTROL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"控制组件不可用: {e}")
    MotionControlSystem = None
    HardwareControlSystem = None
    SensorIntegrationSystem = None
    MotorControlSystem = None
    CONTROL_AVAILABLE = False

# 导入神经通信
try:
    # 首先尝试相对导入
    from ..neural.communication import NeuralCommunication, MessagePriority
except ImportError:
    # 回退到绝对导入
    from neural.communication import NeuralCommunication, MessagePriority


class CognitiveState:
    """认知系统当前状态"""
    
    def __init__(self):
        # 当前焦点和注意力
        self.current_focus = None
        self.attention_weights = {}
        
        # 工作记忆(短期)
        self.working_memory = []
        self.working_memory_capacity = 10
        
        # 长期上下文
        self.long_term_context = {
            'goals': [],            # 目标列表
            'preferences': {},      # 偏好设置
            'knowledge_context': {}, # 知识上下文
            'emotional_state': None # 情感状态
        }
        
        # 目标栈
        self.goal_stack = []
        
        # 认知负荷指标
        self.cognitive_load = 0.0
        self.attention_fatigue = 0.0
        
        # 时间戳
        self.last_update = time.time()
    
    def update_working_memory(self, item: Any):
        """用新项目更新工作记忆"""
        self.working_memory.append({
            'item': item,
            'timestamp': time.time(),
            'importance': 1.0  # 默认重要性
        })
        
        # 限制工作记忆大小
        if len(self.working_memory) > self.working_memory_capacity:
            # 移除最不重要的项目
            self.working_memory.sort(key=lambda x: x['importance'])
            self.working_memory.pop(0)
    
    def update_attention(self, focus: Any, weights: Dict[str, float]):
        """更新注意力焦点和权重"""
        self.current_focus = focus
        self.attention_weights = weights
        self.last_update = time.time()
    
    def add_goal(self, goal, priority: float = 1.0):
        """添加新目标"""
        # 支持字典或字符串
        if isinstance(goal, dict):
            # 如果已经是字典，直接使用
            goal_dict = goal.copy()
            if 'priority' not in goal_dict:
                goal_dict['priority'] = priority
            if 'added_at' not in goal_dict:
                goal_dict['added_at'] = time.time()
            self.goal_stack.append(goal_dict)
        else:
            # 如果是字符串，转换为字典
            self.goal_stack.append({
                'goal': goal,
                'priority': priority,
                'added_at': time.time()
            })
        
        # 按优先级排序
        self.goal_stack.sort(key=lambda x: x.get('priority', 0), reverse=True)
    
    def get_relevant_context(self, query: str = "", max_items: int = 3) -> List[str]:
        """获取相关上下文"""
        # 简单实现：返回工作记忆中的项目
        context = []
        for item in self.working_memory[-max_items:]:
            if isinstance(item, dict) and 'item' in item:
                context.append(str(item['item']))
            else:
                context.append(str(item))
        return context
    
    def get_current_goal(self) -> Optional[Dict[str, Any]]:
        """获取当前最高优先级目标（跳过已完成的）"""
        for goal in self.goal_stack:
            # 跳过状态为'completed'的目标
            if goal.get('status') == 'completed':
                continue
            return goal
        return None
    
    def complete_goal(self, goal_id: str):
        """完成并移除目标"""
        self.goal_stack = [g for g in self.goal_stack if g.get('goal') != goal_id]
    
    def update_cognitive_load(self, load: float):
        """更新认知负荷"""
        self.cognitive_load = load
        self.attention_fatigue = min(1.0, self.attention_fatigue + load * 0.1)
    
    def to_dict(self) -> Dict[str, Any]:
        """将状态转换为字典"""
        return {
            'current_focus': self.current_focus,
            'current_goal': self.get_current_goal(),
            'attention_weights': self.attention_weights,
            'working_memory': self.working_memory,
            'working_memory_size': len(self.working_memory),
            'working_memory_capacity': self.working_memory_capacity,
            'long_term_context': self.long_term_context,
            'goal_stack': self.goal_stack,
            'goal_stack_size': len(self.goal_stack),
            'cognitive_load': self.cognitive_load,
            'attention_fatigue': self.attention_fatigue,
            'last_update': self.last_update
        }
    
    def from_dict(self, state_dict: Dict[str, Any]):
        """从字典恢复状态"""
        self.current_focus = state_dict.get('current_focus')
        self.attention_weights = state_dict.get('attention_weights', {})
        self.working_memory = state_dict.get('working_memory', [])
        # working_memory_capacity 保持原值，不从字典恢复
        self.long_term_context = state_dict.get('long_term_context', {})
        self.goal_stack = state_dict.get('goal_stack', [])
        self.cognitive_load = state_dict.get('cognitive_load', 0.0)
        self.attention_fatigue = state_dict.get('attention_fatigue', 0.0)
        self.last_update = state_dict.get('last_update', time.time())


class UnifiedCognitiveArchitecture:
    """
    统一认知架构 - 替换27模型HTTP协调系统的单一认知架构。
    
    提供完整的认知循环：感知 → 注意力 → 记忆 → 推理 → 规划 → 决策 → 行动 → 学习。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化统一认知架构。
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        
        # 从配置获取参数
        self.embedding_dim = self.config.get('embedding_dim', 1024)
        self.max_shared_memory_mb = self.config.get('max_shared_memory_mb', 1024)
        self.port = self.config.get('port', 9000)
        
        # 创建认知状态
        self.cognitive_state = CognitiveState()
        
        # 创建神经通信系统
        self.neural_comm = NeuralCommunication(
            max_shared_memory_mb=self.max_shared_memory_mb
        )
        
        # 延迟加载的认知组件
        self._repr_space = None
        self._perception = None
        self._attention = None
        self._memory = None
        self._reasoning = None
        self._planning = None
        self._decision = None
        self._action = None
        self._learning = None
        self._evolution = None
        self._self_cognition = None
        self._autonomy = None
        self._humanoid_agi = None
        
        # 延迟加载的控制组件
        self._motion_control = None
        self._hardware_control = None
        self._sensor_integration = None
        self._motor_control = None
        
        # 组件初始化状态
        self.components_initialized = False
        
        # 性能跟踪
        self.performance_metrics = {
            'total_cycles': 0,
            'successful_cycles': 0,
            'failed_cycles': 0,
            'avg_response_time': 0.0,
            'total_processing_time': 0.0
        }
        
        # 注册组件
        self._register_components()
        
        logger.info(f"统一认知架构已初始化，嵌入维度: {self.embedding_dim}")
    
    @property
    def repr_space(self) -> UnifiedRepresentationSpace:
        """获取统一表征空间(延迟加载)"""
        if self._repr_space is None:
            self._repr_space = UnifiedRepresentationSpace(
                embedding_dim=self.embedding_dim,
                enable_cache=True
            )
            logger.info("统一表征空间已初始化")
        return self._repr_space
    
    @property
    def perception(self) -> MultimodalPerception:
        """获取多模态感知组件(延迟加载)"""
        if self._perception is None:
            self._perception = MultimodalPerception(
                communication=self.neural_comm
            )
            logger.info("多模态感知组件已初始化")
        return self._perception
    
    @property
    def attention(self) -> HierarchicalAttention:
        """获取层级注意力组件(延迟加载)"""
        if self._attention is None:
            self._attention = HierarchicalAttention(
                communication=self.neural_comm
            )
            logger.info("层级注意力组件已初始化")
        return self._attention
    
    @property
    def memory(self) -> EpisodicSemanticMemory:
        """获取情景语义记忆组件(延迟加载)"""
        if self._memory is None:
            self._memory = EpisodicSemanticMemory(
                communication=self.neural_comm
            )
            logger.info("情景语义记忆组件已初始化")
        return self._memory
    
    @property
    def reasoning(self) -> UniversalReasoningEngine:
        """获取通用推理引擎(延迟加载)"""
        if self._reasoning is None:
            self._reasoning = UniversalReasoningEngine(
                communication=self.neural_comm
            )
            logger.info("通用推理引擎已初始化")
        return self._reasoning
    
    @property
    def planning(self) -> HierarchicalPlanning:
        """获取层级规划系统(延迟加载)"""
        if self._planning is None:
            self._planning = HierarchicalPlanning(
                communication=self.neural_comm
            )
            logger.info("层级规划系统已初始化")
        return self._planning
    
    @property
    def decision(self) -> ValueBasedDecision:
        """获取价值基础决策系统(延迟加载)"""
        if self._decision is None:
            self._decision = ValueBasedDecision(
                communication=self.neural_comm
            )
            logger.info("价值基础决策系统已初始化")
        return self._decision
    
    @property
    def action(self) -> AdaptiveAction:
        """获取自适应行动执行组件(延迟加载)"""
        if self._action is None:
            self._action = AdaptiveAction(
                communication=self.neural_comm
            )
            logger.info("自适应行动执行组件已初始化")
        return self._action
    
    @property
    def learning(self) -> MetaLearningSystem:
        """获取元学习系统(延迟加载)"""
        if self._learning is None:
            self._learning = MetaLearningSystem(
                communication=self.neural_comm
            )
            logger.info("元学习系统已初始化")
        return self._learning
    
    @property
    def evolution(self) -> AutonomousEvolutionSystem:
        """获取自主演化系统(延迟加载)"""
        if self._evolution is None:
            self._evolution = AutonomousEvolutionSystem(
                communication=self.neural_comm
            )
            logger.info("自主演化系统已初始化")
        return self._evolution
    
    @property
    def self_cognition(self) -> SelfCognitionSystem:
        """获取自我认知系统(延迟加载)"""
        if self._self_cognition is None:
            self._self_cognition = SelfCognitionSystem(
                communication=self.neural_comm
            )
            logger.info("自我认知系统已初始化")
        return self._self_cognition
    
    @property
    def autonomy(self) -> AutonomousSystem:
        """获取自主性系统(延迟加载)"""
        if not AUTONOMY_AVAILABLE:
            logger.warning("自主性系统不可用")
            return None
        
        if self._autonomy is None:
            self._autonomy = AutonomousSystem(
                communication=self.neural_comm
            )
            logger.info("自主性系统已初始化")
        return self._autonomy
    
    @property
    def humanoid_agi(self) -> HumanoidAGISystem:
        """获取人形机器人AGI系统(延迟加载)"""
        if not HUMANOID_AGI_AVAILABLE:
            logger.warning("人形机器人AGI系统不可用")
            return None
        
        if self._humanoid_agi is None:
            self._humanoid_agi = HumanoidAGISystem(
                communication=self.neural_comm
            )
            logger.info("人形机器人AGI系统已初始化")
        return self._humanoid_agi
    
    @property
    def motion_control(self):
        """获取运动控制系统(延迟加载)"""
        if not CONTROL_AVAILABLE:
            logger.warning("运动控制系统不可用")
            return None
        
        if self._motion_control is None:
            self._motion_control = MotionControlSystem(
                communication=self.neural_comm
            )
            logger.info("运动控制系统已初始化")
        return self._motion_control
    
    @property
    def hardware_control(self):
        """获取硬件控制系统(延迟加载)"""
        if not CONTROL_AVAILABLE:
            logger.warning("硬件控制系统不可用")
            return None
        
        if self._hardware_control is None:
            self._hardware_control = HardwareControlSystem(
                communication=self.neural_comm
            )
            logger.info("硬件控制系统已初始化")
        return self._hardware_control
    
    @property
    def sensor_integration(self):
        """获取传感器集成系统(延迟加载)"""
        if not CONTROL_AVAILABLE:
            logger.warning("传感器集成系统不可用")
            return None
        
        if self._sensor_integration is None:
            self._sensor_integration = SensorIntegrationSystem(
                communication=self.neural_comm
            )
            logger.info("传感器集成系统已初始化")
        return self._sensor_integration
    
    @property
    def motor_control(self):
        """获取电机控制系统(延迟加载)"""
        if not CONTROL_AVAILABLE:
            logger.warning("电机控制系统不可用")
            return None
        
        if self._motor_control is None:
            self._motor_control = MotorControlSystem(
                communication=self.neural_comm
            )
            logger.info("电机控制系统已初始化")
        return self._motor_control
    
    def _register_components(self):
        """注册所有认知组件到通信系统"""
        # 注意：实际注册将在组件初始化时进行
        # 这里只设置组件名称和类型映射
        self.component_registry = {
            'perception': 'perception',
            'attention': 'attention',
            'memory': 'memory',
            'reasoning': 'reasoning',
            'planning': 'planning',
            'decision': 'decision',
            'action': 'action',
            'learning': 'learning',
            'evolution': 'evolution',
            'self_cognition': 'self_cognition'
        }
        
        # 注册自主性组件（如果可用）
        if AUTONOMY_AVAILABLE:
            self.component_registry['autonomy'] = 'cognitive'
        
        # 注册人形机器人AGI组件（如果可用）
        if HUMANOID_AGI_AVAILABLE:
            self.component_registry['humanoid_agi'] = 'humanoid'
        
        # 注册控制组件（如果可用）
        if CONTROL_AVAILABLE:
            self.component_registry.update({
                'motion_control': 'control',
                'hardware_control': 'control',
                'sensor_integration': 'control',
                'motor_control': 'control'
            })
    
    def initialize_components(self):
        """初始化所有认知组件"""
        if self.components_initialized:
            return
        
        # 触发所有组件的延迟加载
        _ = self.repr_space
        _ = self.perception
        _ = self.attention
        _ = self.memory
        _ = self.reasoning
        _ = self.planning
        _ = self.decision
        _ = self.action
        _ = self.learning
        _ = self.evolution
        _ = self.self_cognition
        
        # 触发自主性组件的延迟加载（如果可用）
        if AUTONOMY_AVAILABLE:
            _ = self.autonomy
        
        # 触发人形机器人AGI组件的延迟加载（如果可用）
        if HUMANOID_AGI_AVAILABLE:
            _ = self.humanoid_agi
        
        # 触发控制组件的延迟加载（如果可用）
        if CONTROL_AVAILABLE:
            _ = self.motion_control
            _ = self.hardware_control
            _ = self.sensor_integration
            _ = self.motor_control
        
        self.components_initialized = True
        logger.info("所有认知组件已初始化")
    
    async def initialize(self):
        """异步初始化认知架构（API服务器使用）"""
        self.initialize_components()
        logger.info("统一认知架构已通过initialize()方法初始化")
        return True
    
    async def cognitive_cycle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行完整的认知循环。
        
        参数:
            input_data: 输入数据字典
            
        返回:
            认知循环结果
        """
        start_time = time.time()
        
        try:
            # 初始化组件（如果需要）
            self.initialize_components()
            
            # 更新认知状态中的当前输入
            self.cognitive_state.update_working_memory(input_data)
            
            # 步骤1: 多模态感知
            logger.info("认知循环: 多模态感知阶段")
            perception_result = await self._process_perception(input_data)
            
            # 步骤2: 层级注意力
            logger.info("认知循环: 层级注意力阶段")
            attention_result = await self._process_attention(perception_result)
            
            # 步骤3: 情景语义记忆
            logger.info("认知循环: 情景语义记忆阶段")
            memory_result = await self._process_memory(attention_result)
            
            # 步骤4: 通用推理
            logger.info("认知循环: 通用推理阶段")
            reasoning_result = await self._process_reasoning(memory_result)
            
            # 步骤5: 层级规划
            logger.info("认知循环: 层级规划阶段")
            planning_result = await self._process_planning(reasoning_result)
            
            # 步骤6: 价值基础决策
            logger.info("认知循环: 价值基础决策阶段")
            decision_result = await self._process_decision(planning_result)
            
            # 步骤7: 自适应行动
            logger.info("认知循环: 自适应行动阶段")
            action_result = await self._process_action(decision_result)
            
            # 步骤8: 元学习
            logger.info("认知循环: 元学习阶段")
            learning_result = await self._process_learning(action_result)
            
            # 整合所有结果
            final_result = self._integrate_results(
                perception_result, attention_result, memory_result,
                reasoning_result, planning_result, decision_result,
                action_result, learning_result
            )
            
            # 更新性能指标
            self._update_performance_metrics(start_time, success=True)
            
            return final_result
            
        except Exception as e:
            logger.error(f"认知循环失败: {e}", exc_info=True)
            self._update_performance_metrics(start_time, success=False)
            
            return {
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': time.time()
            }
    
    async def _process_perception(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理多模态感知"""
        try:
            # 编码输入为统一表征
            unified_repr = self.repr_space.encode(input_data, use_cache=True)
            
            # 通过感知组件处理
            perception_result = {
                'unified_representation': unified_repr,
                'modality_detected': list(input_data.keys()),
                'confidence': 0.9,  # 模拟置信度
                'timestamp': time.time()
            }
            
            # 更新认知状态
            self.cognitive_state.update_attention(
                focus=unified_repr,
                weights={'visual': 0.6, 'auditory': 0.3, 'textual': 0.1}
            )
            
            return perception_result
            
        except Exception as e:
            logger.error(f"感知处理失败: {e}")
            return {
                'error': f"感知处理失败: {str(e)}",
                'unified_representation': None,
                'timestamp': time.time()
            }
    
    async def _process_attention(self, perception_result: Dict[str, Any]) -> Dict[str, Any]:
        """处理层级注意力"""
        try:
            # 模拟注意力处理
            attention_result = {
                'attention_focus': 'unified_representation',
                'attention_weights': self.cognitive_state.attention_weights,
                'saliency_map': None,  # 实际实现中会有显著性图
                'timestamp': time.time()
            }
            
            return attention_result
            
        except Exception as e:
            logger.error(f"注意力处理失败: {e}")
            return {
                'error': f"注意力处理失败: {str(e)}",
                'attention_focus': None,
                'timestamp': time.time()
            }
    
    async def _process_memory(self, attention_result: Dict[str, Any]) -> Dict[str, Any]:
        """处理情景语义记忆"""
        try:
            # 模拟记忆检索
            memory_result = {
                'episodic_memory': [],
                'semantic_memory': [],
                'retrieval_confidence': 0.8,
                'context_relevance': 0.7,
                'timestamp': time.time()
            }
            
            return memory_result
            
        except Exception as e:
            logger.error(f"记忆处理失败: {e}")
            return {
                'error': f"记忆处理失败: {str(e)}",
                'episodic_memory': None,
                'timestamp': time.time()
            }
    
    async def _process_reasoning(self, memory_result: Dict[str, Any]) -> Dict[str, Any]:
        """处理通用推理"""
        try:
            # 模拟推理过程
            reasoning_result = {
                'logical_conclusion': '基于输入的模拟结论',
                'reasoning_chain': ['前提1', '前提2', '结论'],
                'confidence': 0.85,
                'alternative_hypotheses': [],
                'timestamp': time.time()
            }
            
            return reasoning_result
            
        except Exception as e:
            logger.error(f"推理处理失败: {e}")
            return {
                'error': f"推理处理失败: {str(e)}",
                'logical_conclusion': None,
                'timestamp': time.time()
            }
    
    async def _process_planning(self, reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """处理层级规划"""
        try:
            # 模拟规划过程
            planning_result = {
                'plan': ['步骤1', '步骤2', '步骤3'],
                'subgoals': [],
                'estimated_duration': 10.0,
                'success_probability': 0.9,
                'timestamp': time.time()
            }
            
            return planning_result
            
        except Exception as e:
            logger.error(f"规划处理失败: {e}")
            return {
                'error': f"规划处理失败: {str(e)}",
                'plan': None,
                'timestamp': time.time()
            }
    
    async def _process_decision(self, planning_result: Dict[str, Any]) -> Dict[str, Any]:
        """处理价值基础决策"""
        try:
            # 模拟决策过程
            decision_result = {
                'selected_action': '执行计划',
                'value_estimate': 0.95,
                'risk_assessment': 0.1,
                'expected_reward': 1.0,
                'timestamp': time.time()
            }
            
            return decision_result
            
        except Exception as e:
            logger.error(f"决策处理失败: {e}")
            return {
                'error': f"决策处理失败: {str(e)}",
                'selected_action': None,
                'timestamp': time.time()
            }
    
    async def _process_action(self, decision_result: Dict[str, Any]) -> Dict[str, Any]:
        """处理自适应行动"""
        try:
            # 模拟行动执行
            action_result = {
                'executed_action': decision_result.get('selected_action', '未知行动'),
                'execution_status': 'simulated',
                'feedback': {'quality': 0.5},  # 模拟反馈
                'actual_cost': 1.0,
                'timestamp': time.time()
            }
            
            return action_result
            
        except Exception as e:
            logger.error(f"行动处理失败: {e}")
            return {
                'error': f"行动处理失败: {str(e)}",
                'executed_action': None,
                'timestamp': time.time()
            }
    
    async def _process_learning(self, action_result: Dict[str, Any]) -> Dict[str, Any]:
        """处理元学习"""
        try:
            # 模拟学习过程
            learning_result = {
                'learning_update': True,
                'improvement': 0.01,
                'knowledge_gain': 0.05,
                'skill_improvement': 0.02,
                'timestamp': time.time()
            }
            
            return learning_result
            
        except Exception as e:
            logger.error(f"学习处理失败: {e}")
            return {
                'error': f"学习处理失败: {str(e)}",
                'learning_update': False,
                'timestamp': time.time()
            }
    
    def _convert_tensors(self, data: Any) -> Any:
        """递归转换张量为列表以进行序列化"""
        if isinstance(data, torch.Tensor):
            # 转换为Python列表
            return data.tolist()
        elif isinstance(data, dict):
            # 递归处理字典
            return {key: self._convert_tensors(value) for key, value in data.items()}
        elif isinstance(data, list):
            # 递归处理列表
            return [self._convert_tensors(item) for item in data]
        elif isinstance(data, np.ndarray):
            # 转换numpy数组
            return data.tolist()
        else:
            # 其他类型保持不变
            return data
    
    def _integrate_results(self, *component_results: Dict[str, Any]) -> Dict[str, Any]:
        """整合所有组件结果"""
        integrated_result = {
            'perception': self._convert_tensors(component_results[0]),
            'attention': self._convert_tensors(component_results[1]),
            'memory': self._convert_tensors(component_results[2]),
            'reasoning': self._convert_tensors(component_results[3]),
            'planning': self._convert_tensors(component_results[4]),
            'decision': self._convert_tensors(component_results[5]),
            'action': self._convert_tensors(component_results[6]),
            'learning': self._convert_tensors(component_results[7]),
            'cognitive_state': self._convert_tensors(self.cognitive_state.to_dict()),
            'timestamp': time.time(),
            'cycle_id': f"cycle_{self.performance_metrics['total_cycles']}",
            'success': all('error' not in r or r.get('error') is None for r in component_results)
        }
        
        return integrated_result
    
    def _update_performance_metrics(self, start_time: float, success: bool):
        """更新性能指标"""
        cycle_time = time.time() - start_time
        
        self.performance_metrics['total_cycles'] += 1
        if success:
            self.performance_metrics['successful_cycles'] += 1
        else:
            self.performance_metrics['failed_cycles'] += 1
        
        # 更新平均响应时间（加权平均）
        total_time = self.performance_metrics['total_processing_time'] + cycle_time
        avg_time = total_time / self.performance_metrics['total_cycles']
        
        self.performance_metrics['total_processing_time'] = total_time
        self.performance_metrics['avg_response_time'] = avg_time
        
        logger.info(f"认知循环完成: 时间={cycle_time:.3f}s, 成功={success}")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """获取系统诊断信息"""
        diagnostics = {
            'system_info': {
                'total_cycles': self.performance_metrics['total_cycles'],
                'successful_cycles': self.performance_metrics['successful_cycles'],
                'failed_cycles': self.performance_metrics['failed_cycles'],
                'success_rate': self.performance_metrics['successful_cycles'] / max(1, self.performance_metrics['total_cycles']),
                'avg_response_time': self.performance_metrics['avg_response_time'],
                'total_processing_time': self.performance_metrics['total_processing_time'],
                'components_initialized': self.components_initialized,
                'embedding_dim': self.embedding_dim
            },
            'cognitive_state': self._convert_tensors(self.cognitive_state.to_dict()),
            'representation_cache': self._convert_tensors(self.repr_space.get_cache_stats()),
            'communication_stats': self._convert_tensors(self.neural_comm.get_statistics()),
            'configuration': self.config,
            'timestamp': time.time()
        }
        
        return diagnostics
    
    async def shutdown(self):
        """关闭认知架构"""
        logger.info("关闭认知架构...")
        
        # 清除缓存
        self.repr_space.clear_cache()
        
        # 清理通信系统
        # 注意：实际实现中会有更多清理工作
        
        logger.info("认知架构关闭完成")


# 简单测试
if __name__ == "__main__":
    import asyncio
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    async def test_cognitive_cycle():
        # 创建认知架构
        config = {
            'embedding_dim': 512,
            'max_shared_memory_mb': 100
        }
        
        agi = UnifiedCognitiveArchitecture(config)
        
        # 测试输入
        test_input = {
            'text': "这是什么测试？",
            'context': {'test': 'example'}
        }
        
        # 执行认知循环
        result = await agi.cognitive_cycle(test_input)
        
        print(f"认知循环结果: {result}")
        
        # 获取诊断信息
        diagnostics = agi.get_diagnostics()
        print(f"诊断信息: {diagnostics}")
        
        # 关闭
        await agi.shutdown()
    
    # 运行测试
    asyncio.run(test_cognitive_cycle())