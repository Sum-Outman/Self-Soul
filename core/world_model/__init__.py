"""
世界模型模块 - 实现动态环境建模、状态表示和预测

模块列表:
- world_state_representation: 世界状态表示系统
- belief_state: 贝叶斯信念状态管理
- state_transition_model: 状态转移模型
- partial_observability_handler: 部分可观测性处理器

核心功能:
1. 动态环境状态表示和维护
2. 基于贝叶斯推理的信念状态更新
3. 状态转移预测和不确定性传播
4. 部分可观测环境下的状态估计
5. 世界模型的在线学习和适应

技术特点:
- 概率状态表示: 支持不确定性和概率分布
- 因果整合: 结合因果知识改进状态预测
- 多时间尺度: 支持不同粒度的时间表示
- 模块化设计: 可插拔的状态变量和转移模型

应用场景:
1. 机器人导航和环境理解
2. 复杂系统状态监控和预测
3. 决策支持系统的情境建模
4. 游戏AI的环境理解和预测

版权所有 (c) 2026 AGI Soul Team
"""

from .world_state_representation import WorldStateRepresentation
from .belief_state import BeliefState
from .state_transition_model import StateTransitionModel
from .partial_observability_handler import PartialObservabilityHandler

__all__ = [
    'WorldStateRepresentation',
    'BeliefState',
    'StateTransitionModel',
    'PartialObservabilityHandler'
]