"""
因果推理模块包 - 实现结构因果模型(SCM)、do-calculus和因果发现算法

模块列表:
- causal_scm_engine: 结构因果模型引擎
- do_calculus_engine: do-calculus数学实现  
- causal_discovery: 因果发现算法(PC, FCI等)
- counterfactual_reasoner: 反事实推理引擎
- causal_inference: 因果效应估计和推断

核心功能:
1. 基于Pearl因果框架的结构因果模型构建
2. do-calculus数学运算和干预效果估计
3. 从数据中自动发现因果关系
4. 反事实推理和what-if分析
5. 因果效应传播和影响分析

版权所有 (c) 2026 AGI Soul Team
"""

from .causal_scm_engine import StructuralCausalModelEngine
from .do_calculus_engine import DoCalculusEngine
from .causal_discovery import CausalDiscoveryEngine, DiscoveryAlgorithm, IndependenceTest
from .counterfactual_reasoner import CounterfactualReasoner, CounterfactualQuery, AbductionMethod
from .causal_query_language import CausalQueryLanguage, QueryType, QueryParser, QueryExecutor
from .causal_knowledge_graph import CausalKnowledgeGraph

__all__ = [
    'StructuralCausalModelEngine',
    'DoCalculusEngine', 
    'CausalDiscoveryEngine',
    'DiscoveryAlgorithm',
    'IndependenceTest',
    'CounterfactualReasoner',
    'CounterfactualQuery',
    'AbductionMethod',
    'CausalQueryLanguage',
    'QueryType',
    'QueryParser',
    'QueryExecutor',
    'CausalKnowledgeGraph'
]