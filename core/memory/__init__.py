"""
增强记忆系统 - 实现情景、语义和程序记忆的完整体系

模块列表:
- episodic_semantic_memory: 综合记忆系统管理器
- temporal_memory_store: 情景记忆存储（时间序列）
- causal_memory_store: 因果语义记忆（概念关系）
- skill_memory_store: 程序技能记忆（动作序列）

核心功能:
1. 情景记忆：存储具体经历的时间序列和情境
2. 语义记忆：存储概念、事实和因果关系
3. 程序记忆：存储技能、动作序列和操作流程
4. 记忆巩固：从情景记忆中提取规律形成语义记忆
5. 记忆检索：基于内容的相似性检索和情境召回
6. 遗忘机制：基于重要性、频率和时间的智能遗忘

记忆类型详解:
1. 情景记忆（Episodic Memory）:
   - 存储：具体事件的时间、地点、人物、感受
   - 特点：时间序列性、情境依赖性、细节丰富
   - 用途：个人经历回顾、情境学习、模式识别

2. 语义记忆（Semantic Memory）:
   - 存储：概念、事实、规则、因果关系
   - 特点：抽象性、通用性、关系网络
   - 用途：知识推理、概念理解、问题解决

3. 程序记忆（Procedural Memory）:
   - 存储：技能、动作序列、操作流程
   - 特点：自动化、序列性、条件触发
   - 用途：技能执行、习惯形成、动作优化

记忆过程:
1. 编码（Encoding）：感知信息转化为记忆表示
2. 存储（Storage）：记忆在长期存储中的保持
3. 巩固（Consolidation）：短期记忆转化为长期记忆
4. 检索（Retrieval）：从存储中提取记忆内容
5. 遗忘（Forgetting）：选择性删除不重要记忆

技术特点:
- 多模态存储：支持文本、图像、声音等多种信息
- 关联检索：基于内容和情境的智能检索
- 记忆重组：记忆的重新组织和整合
- 情感标记：记忆与情感状态的关联

版权所有 (c) 2026 AGI Soul Team
"""

from .episodic_semantic_memory import EpisodicSemanticMemory
from .temporal_memory_store import TemporalMemoryStore
from .causal_memory_store import CausalMemoryStore
from .skill_memory_store import SkillMemoryStore

__all__ = [
    'EpisodicSemanticMemory',
    'TemporalMemoryStore',
    'CausalMemoryStore',
    'SkillMemoryStore'
]