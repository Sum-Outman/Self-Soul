"""
自主性系统 - 实现基于内在动机的自主目标生成和行为驱动

模块列表:
- autonomous_goal_generator: 自主目标生成系统
- curiosity_driven_exploration: 好奇心驱动探索
- competence_based_goals: 能力提升目标生成
- knowledge_gap_detector: 知识缺口检测器

核心功能:
1. 内在动机驱动：基于好奇心、能力提升等内在需求生成目标
2. 目标价值评估：评估目标的可行性、价值和风险
3. 知识缺口发现：识别当前知识体系的不足和缺失
4. 探索-利用平衡：在探索新领域和利用已知知识间取得平衡
5. 目标冲突解决：处理多个目标间的冲突和优先级

内在动机类型:
1. 好奇心驱动（Curiosity-Driven）:
   - 对新奇性、不确定性、信息增益的追求
   - 探索未知领域、尝试新方法、学习新知识
   - 衡量指标：信息增益、预测误差、新奇度

2. 能力提升驱动（Competence-Driven）:
   - 对技能掌握、效率提升、性能优化的追求
   - 练习已有技能、优化操作流程、提高成功率
   - 衡量指标：成功率、效率、熟练度

3. 知识完整性驱动（Knowledge-Completeness）:
   - 对知识体系完整性和一致性的追求
   - 填补知识空白、解决认知矛盾、建立知识联系
   - 衡量指标：知识覆盖率、一致性、解释力

目标生成过程:
1. 动机激活：基于当前状态激活相应内在动机
2. 机会识别：识别可能的目标机会和改善空间
3. 目标生成：将机会转化为具体、可操作的目标
4. 价值评估：评估目标的期望价值和成本
5. 优先级排序：根据价值和紧迫性排序目标
6. 目标选择：选择最优目标进行执行

技术特点:
- 多目标优化：同时考虑多个维度的优化目标
- 动态调整：根据经验动态调整动机权重
- 风险感知：考虑目标执行的风险和不确定性
- 学习改进：从目标执行结果中学习改进目标生成策略

应用场景:
1. 自主学习系统的目标设定
2. 研究型AI的探索方向选择
3. 技能学习系统的练习计划生成
4. 问题解决系统的子问题分解

版权所有 (c) 2026 AGI Soul Team
"""

from .autonomous_goal_generator import AutonomousGoalGenerator
from .curiosity_driven_exploration import CuriosityDrivenExploration
from .competence_based_goals import CompetenceBasedGoals
from .knowledge_gap_detector import KnowledgeGapDetector

__all__ = [
    'AutonomousGoalGenerator',
    'CuriosityDrivenExploration',
    'CompetenceBasedGoals',
    'KnowledgeGapDetector'
]