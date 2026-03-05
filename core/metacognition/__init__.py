"""
元认知系统 - 实现思维过程的自我监控、评估和调节

模块列表:
- meta_cognitive_monitor: 元认知监控核心系统
- thinking_process_tracker: 思维过程追踪器
- cognitive_bias_detector: 认知偏差检测器
- reasoning_strategy_evaluator: 推理策略评估器
- cognitive_regulation_mechanism: 认知调节机制

核心功能:
1. 思维过程监控：实时追踪和分析思维活动
2. 认知偏差检测：识别和纠正思维中的系统性偏差
3. 策略效果评估：评估不同推理策略的有效性
4. 认知资源分配：优化注意力、记忆和计算资源的分配
5. 自我反思和改进：基于监控结果进行自我调整

元认知维度:
1. 元认知知识（Metacognitive Knowledge）:
   - 关于认知过程的知识：认知能力、策略、任务需求
   - 关于认知任务的知识：任务类型、难度、要求
   - 关于认知策略的知识：可用策略、适用条件、效果

2. 元认知体验（Metacognitive Experience）:
   - 认知过程的情感体验：困惑、确信、流畅感
   - 任务执行的感受：难度感知、进展感、成就感
   - 学习过程的体验：理解程度、掌握感、进步感

3. 元认知调节（Metacognitive Regulation）:
   - 计划：任务分析、目标设定、策略选择
   - 监控：进程追踪、理解检查、策略评估
   - 评估：结果评价、策略反思、改进计划

监控内容:
1. 思维质量监控：
   - 逻辑一致性、证据充分性、推理严密性
   - 创造性、灵活性、深度和广度
   - 偏见和错误倾向

2. 认知资源监控：
   - 注意力集中度、记忆负荷、计算复杂度
   - 疲劳程度、压力水平、情绪状态
   - 时间和能量消耗

3. 策略效果监控：
   - 问题解决策略的有效性
   - 学习策略的效率
   - 决策策略的合理性

调节机制:
1. 策略调整：根据效果调整认知策略
2. 资源重分配：优化认知资源分配
3. 目标修正：调整任务目标和期望
4. 求助决策：决定是否需要外部帮助

技术特点:
- 实时监控：低延迟的思维过程追踪
- 多维度评估：综合评估认知过程多个方面
- 自适应调节：基于监控结果的动态调整
- 学习改进：从监控经验中学习改进监控策略

应用价值:
1. 提高问题解决的效率和质量
2. 减少认知偏差和错误决策
3. 优化学习过程和知识获取
4. 增强自我意识和自我调节能力

版权所有 (c) 2026 AGI Soul Team
"""

from .meta_cognitive_monitor import MetaCognitiveMonitor
from .thinking_process_tracker import ThinkingProcessTracker
from .cognitive_bias_detector import CognitiveBiasDetector
from .reasoning_strategy_evaluator import ReasoningStrategyEvaluator
from .cognitive_regulation_mechanism import CognitiveRegulationMechanism, RegulationGoal, RegulationType, RegulationIntensity

__all__ = [
    'MetaCognitiveMonitor',
    'ThinkingProcessTracker',
    'CognitiveBiasDetector',
    'ReasoningStrategyEvaluator',
    'CognitiveRegulationMechanism',
    'RegulationGoal',
    'RegulationType',
    'RegulationIntensity'
]