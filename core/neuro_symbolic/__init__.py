"""
神经符号统一框架 - 实现双向神经符号推理系统

模块列表:
- neuro_symbolic_unified: 神经符号统一框架核心
- first_order_logic_reasoner: 一阶逻辑推理引擎
- neural_concept_grounder: 神经概念接地器
- abductive_reasoning_engine: 溯因推理引擎

核心功能:
1. 神经表示到符号命题的自动提取和学习
2. 符号命题到神经执行的约束指导和验证
3. 一阶逻辑推理(与、或、非、蕴含、量词)
4. 溯因推理寻找最佳解释
5. 神经符号双向转换的一致性维护

设计原则:
- 可微符号推理: 符号操作可微分，支持端到端学习
- 双向信息流: 神经⇄符号双向转换和验证
- 分层抽象: 从具体感知到抽象概念的多层表示
- 一致性保证: 确保神经输出符合符号约束

版权所有 (c) 2026 AGI Soul Team
"""

from .neuro_symbolic_unified import NeuralSymbolicUnifiedFramework
from .first_order_logic_reasoner import FirstOrderLogicReasoner
from .neural_concept_grounder import NeuralConceptGrounder
from .abductive_reasoning_engine import AbductiveReasoningEngine

__all__ = [
    'NeuralSymbolicUnifiedFramework',
    'FirstOrderLogicReasoner',
    'NeuralConceptGrounder',
    'AbductiveReasoningEngine'
]