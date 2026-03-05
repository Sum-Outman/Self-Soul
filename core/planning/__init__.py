"""
分层规划系统 - 实现多时间尺度的分层决策和规划

模块列表:
- hierarchical_planning_system: 分层规划系统核心
- strategic_planner: 战略层规划器（长期目标）
- tactical_planner: 战术层规划器（中期计划）
- operational_planner: 操作层规划器（短期行动）

核心功能:
1. 三层规划抽象：战略→战术→操作的层次分解
2. 多时间尺度规划：从分钟到数年的不同时间粒度
3. 目标分解和任务规划：高层目标到具体行动的转换
4. 实时重规划和适应：环境变化时的动态调整
5. 资源约束优化：在有限资源下的最优规划

分层结构:
1. 战略层（Strategic）:
   - 时间尺度：月~年
   - 抽象级别：高层目标、价值导向
   - 输出：战略目标、优先级、资源分配

2. 战术层（Tactical）:
   - 时间尺度：天~周
   - 抽象级别：项目计划、里程碑
   - 输出：任务序列、依赖关系、时间安排

3. 操作层（Operational）:
   - 时间尺度：分钟~小时
   - 抽象级别：具体行动、即时决策
   - 输出：动作序列、参数设置、执行监控

技术特点:
- 目标导向规划：从抽象目标到具体行动的完整链条
- 约束满足规划：处理时间、资源、能力等约束
- 不确定性处理：支持概率规划和风险感知
- 学习优化：从经验中改进规划策略

版权所有 (c) 2026 AGI Soul Team
"""

from .hierarchical_planning_system import HierarchicalPlanningSystem
from .strategic_planner import StrategicPlanner
from .tactical_planner import TacticalPlanner
from .operational_planner import OperationalPlanner

__all__ = [
    'HierarchicalPlanningSystem',
    'StrategicPlanner',
    'TacticalPlanner',
    'OperationalPlanner'
]