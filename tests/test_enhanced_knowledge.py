import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.knowledge_integrator_enhanced import AGIKnowledgeIntegrator, EnhancedKnowledgeRelation
from datetime import datetime
import networkx as nx

def test_causal_reasoning_enhanced():
    """测试增强的因果推理功能"""
    print("=== 测试增强的因果推理功能 ===")
    
    # 创建知识整合器实例
    knowledge_integrator = AGIKnowledgeIntegrator(from_scratch=True)
    
    # 添加测试知识
    knowledge_integrator.add_knowledge(
        subject="吸烟",
        relation="导致",
        object="肺癌",
        confidence=0.9,
        source="medical",
        relation_type="causal",
        domain="medicine"
    )
    
    knowledge_integrator.add_knowledge(
        subject="肺癌",
        relation="导致",
        object="呼吸困难",
        confidence=0.85,
        source="medical",
        relation_type="causal",
        domain="medicine"
    )
    
    knowledge_integrator.add_knowledge(
        subject="吸烟",
        relation="导致",
        object="咳嗽",
        confidence=0.8,
        source="medical",
        relation_type="causal",
        domain="medicine"
    )
    
    knowledge_integrator.add_knowledge(
        subject="咳嗽",
        relation="可能导致",
        object="声音嘶哑",
        confidence=0.7,
        source="medical",
        relation_type="causal",
        domain="medicine"
    )
    
    knowledge_integrator.add_knowledge(
        subject="声音嘶哑",
        relation="可能由",
        object="肺癌",
        confidence=0.6,
        source="medical",
        relation_type="causal",
        domain="medicine"
    )
    
    # 测试因果推理
    print("\n--- 测试因果路径查找 ---")
    cause_results = knowledge_integrator.semantic_query("吸烟", max_results=3)
    effect_results = knowledge_integrator.semantic_query("肺癌", max_results=3)
    
    print(f"原因结果: {cause_results}")
    print(f"效果结果: {effect_results}")
    
    # 查找因果路径
    causal_paths = knowledge_integrator._find_causal_paths(
        cause_results, 
        effect_results, 
        max_depth=5, 
        max_paths=3,
        evolutionary_factor=0.1
    )
    
    print(f"\n找到的因果路径: {causal_paths}")
    
    # 测试因果推理函数
    print("\n--- 测试因果推理函数 ---")
    causal_reasoning_result = knowledge_integrator.causal_reasoning("吸烟", "肺癌")
    print(f"因果推理结果: {causal_reasoning_result}")
    
    # 测试路径评分
    print("\n--- 测试路径评分函数 ---")
    if causal_paths:
        path = causal_paths[0]
        path_score = knowledge_integrator._calculate_path_score(
            path['strength'],
            path['confidence'],
            path['length'],
            path['path'],
            path['consistency'],
            0.5,
            0.1
        )
        print(f"路径评分: {path_score}")
    
    # 测试演化潜力计算
    print("\n--- 测试演化潜力计算 ---")
    if causal_paths and causal_paths[0]['path']:
        evolutionary_potential = knowledge_integrator._calculate_evolutionary_potential(causal_paths[0]['path'])
        print(f"演化潜力: {evolutionary_potential}")
    
    # 测试知识图谱演化
    print("\n--- 测试知识图谱演化 ---")
    knowledge_integrator._evolve_knowledge_graph(causal_paths, evolutionary_factor=0.1)
    print("知识图谱演化完成")
    
    return True

def test_semantic_query_enhanced():
    """测试增强的语义查询功能"""
    print("\n=== 测试增强的语义查询功能 ===")
    
    knowledge_integrator = AGIKnowledgeIntegrator(from_scratch=True)
    
    # 添加测试知识
    knowledge_integrator.add_knowledge(
        subject="人工智能",
        relation="是",
        object="计算机科学分支",
        confidence=0.95,
        source="academic",
        relation_type="hierarchical",
        domain="computer_science"
    )
    
    knowledge_integrator.add_knowledge(
        subject="机器学习",
        relation="是",
        object="人工智能子领域",
        confidence=0.98,
        source="academic",
        relation_type="hierarchical",
        domain="computer_science"
    )
    
    knowledge_integrator.add_knowledge(
        subject="深度学习",
        relation="是",
        object="机器学习方法",
        confidence=0.92,
        source="academic",
        relation_type="hierarchical",
        domain="computer_science"
    )
    
    # 测试语义查询
    results = knowledge_integrator.semantic_query(
        query="人工智能",
        max_results=5,
        similarity_threshold=0.6,
        domain_weights={"computer_science": 1.2}
    )
    
    print(f"语义查询结果: {results}")
    return True

if __name__ == "__main__":
    try:
        test_causal_reasoning_enhanced()
        test_semantic_query_enhanced()
        print("\n=== 所有测试通过！===\n")
    except Exception as e:
        print(f"\n=== 测试失败: {e} ===\n")
        import traceback
        traceback.print_exc()
