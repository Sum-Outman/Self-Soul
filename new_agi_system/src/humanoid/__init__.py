"""
人形机器人AGI模块

提供人形机器人特有的高级功能，包括：
1. 平衡控制与双足行走算法
2. 社交交互与人机互动
3. 环境适应与复杂地形导航
4. 高级任务执行与规划

将原有Self-Soul系统的人形机器人功能迁移到统一认知架构中。
"""

from .balance_control import BalanceControlSystem
from .walking_gait import WalkingGaitSystem
from .humanoid_agi import HumanoidAGISystem

# 社交交互和环境适应系统将在未来版本中实现
# from .social_interaction import SocialInteractionSystem
# from .environment_adaptation import EnvironmentAdaptationSystem

__all__ = [
    'BalanceControlSystem',
    'WalkingGaitSystem',
    'HumanoidAGISystem'
]