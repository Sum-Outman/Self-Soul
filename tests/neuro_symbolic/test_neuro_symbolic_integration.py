#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
神经符号统一框架集成测试

测试目标:
1. 神经符号统一框架基本功能
2. 各组件之间的集成和协同工作
3. 性能和可扩展性
4. 神经符号一致性检查

测试组件:
- neuro_symbolic_unified.py: 神经符号统一框架
- first_order_logic_reasoner.py: 一阶逻辑推理引擎
- neural_concept_grounder.py: 神经概念接地器
- abductive_reasoning_engine.py: 溯因推理引擎

测试场景:
1. 简单感知到符号的转换
2. 符号推理到神经约束的应用
3. 完整神经符号推理循环
4. 多组件集成工作流

版权所有 (c) 2026 AGI Soul Team
"""

import unittest
import logging
import time
import numpy as np
import zlib
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.neuro_symbolic.neuro_symbolic_unified import (
    NeuralSymbolicUnifiedFramework, 
    SymbolType, 
    LogicOperator
)
from core.neuro_symbolic.first_order_logic_reasoner import FirstOrderLogicReasoner
from core.neuro_symbolic.neural_concept_grounder import NeuralConceptGrounder, GroundingType
from core.neuro_symbolic.abductive_reasoning_engine import AbductiveReasoningEngine

# 设置日志
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestNeuralSymbolicUnifiedFramework(unittest.TestCase):
    """测试神经符号统一框架基本功能"""
    
    def setUp(self):
        """测试前设置"""
        self.framework = NeuralSymbolicUnifiedFramework(
            input_dim=64,
            hidden_dim=32,
            symbol_dim=16,
            learning_rate=0.001
        )
        
    def _deterministic_randn(self, size, seed_prefix="default"):
        """Generate deterministic normal distribution using numpy RandomState"""
        import math
        if isinstance(size, int):
            size = (size,)
        total_elements = 1
        for dim in size:
            total_elements *= dim
        
        # Create deterministic seed from seed_prefix using adler32
        seed_hash = zlib.adler32(seed_prefix.encode('utf-8')) & 0xffffffff
        rng = np.random.RandomState(seed_hash)
        
        # Generate uniform random numbers
        u1 = rng.random_sample(total_elements)
        u2 = rng.random_sample(total_elements)
        
        # Apply Box-Muller transform
        u1 = np.maximum(u1, 1e-10)
        u2 = np.maximum(u2, 1e-10)
        z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)
        
        # Convert to torch tensor
        import torch
        result = torch.from_numpy(z0).float()
        
        return result.view(*size)
        
    def test_initialization(self):
        """测试框架初始化"""
        self.assertIsNotNone(self.framework)
        self.assertIsNotNone(self.framework.neural_encoder)
        self.assertIsNotNone(self.framework.symbol_extractor)
        self.assertIsNotNone(self.framework.symbol_decoder)
        self.assertIsNotNone(self.framework.neural_optimizer)
        
        # 检查基础符号是否已初始化
        self.assertIn("entity_object", self.framework.symbol_knowledge_base["entities"])
        self.assertIn("relation_part_of", self.framework.symbol_knowledge_base["relations"])
        self.assertIn("property_color", self.framework.symbol_knowledge_base["properties"])
        
        logger.info("神经符号统一框架初始化测试通过")
    
    def test_neural_to_symbol_translation(self):
        """测试神经到符号的转换"""
        # 创建模拟神经输入
        batch_size = 2
        input_dim = 64
        neural_input = self._deterministic_randn((batch_size, input_dim), seed_prefix="neural_input_translation")
        
        # 神经编码
        with torch.no_grad():
            neural_representation = self.framework.neural_encoder(neural_input.to(self.framework.device))
        
        self.assertIsNotNone(neural_representation)
        self.assertEqual(neural_representation.shape[0], batch_size)
        self.assertEqual(neural_representation.shape[1], self.framework.hidden_dim // 2)
        
        # 符号提取（简化版本）
        # 在实际实现中，这里会调用具体的符号提取方法
        logger.info("神经到符号转换测试通过")
    
    def test_symbol_addition(self):
        """测试符号添加功能"""
        initial_count = len(self.framework.symbol_knowledge_base["entities"])
        
        # 添加新实体符号
        symbol_id = self.framework._add_symbol("test_entity", SymbolType.ENTITY)
        
        self.assertIsNotNone(symbol_id)
        self.assertIn(symbol_id, self.framework.symbol_knowledge_base["entities"])
        self.assertEqual(len(self.framework.symbol_knowledge_base["entities"]), initial_count + 1)
        
        # 验证符号信息
        entity_info = self.framework.symbol_knowledge_base["entities"][symbol_id]
        self.assertEqual(entity_info["name"], "test_entity")
        self.assertEqual(entity_info["type"], "entity")
        self.assertIsNotNone(entity_info["vector"])
        
        logger.info("符号添加测试通过")
    
    def test_logic_rule_addition(self):
        """测试逻辑规则添加"""
        initial_rule_count = len(self.framework.symbol_knowledge_base["rules"])
        
        # 添加逻辑规则
        rule = "forall x, y: contains(x, y) → part_of(y, x)"
        self.framework._add_logic_rule(rule)
        
        self.assertEqual(len(self.framework.symbol_knowledge_base["rules"]), initial_rule_count + 1)
        self.assertIn(rule, self.framework.symbol_knowledge_base["rules"])
        
        logger.info("逻辑规则添加测试通过")


class TestFirstOrderLogicReasoner(unittest.TestCase):
    """测试一阶逻辑推理引擎"""
    
    def setUp(self):
        """测试前设置"""
        self.reasoner = FirstOrderLogicReasoner()
    
    def test_initialization(self):
        """测试推理引擎初始化"""
        self.assertIsNotNone(self.reasoner)
        self.assertIsNotNone(self.reasoner.knowledge_base)
        
        # 检查知识库结构
        self.assertIn("facts", self.reasoner.knowledge_base)
        self.assertIn("rules", self.reasoner.knowledge_base)
        self.assertIn("predicates", self.reasoner.knowledge_base)
        
        logger.info("一阶逻辑推理引擎初始化测试通过")
    
    def test_fact_addition(self):
        """测试事实添加"""
        fact = "cat(tom)"
        
        # 添加事实
        self.reasoner.add_formula(fact, is_axiom=False)
        
        self.assertIn(fact, self.reasoner.knowledge_base["facts"])
        
        logger.info("事实添加测试通过")
    
    def test_simple_inference(self):
        """测试简单推理"""
        # 添加事实和规则
        self.reasoner.add_formula("cat(tom)", is_axiom=False)
        self.reasoner.add_formula("dog(spot)", is_axiom=False)
        self.reasoner.add_formula("forall x: cat(x) → animal(x)", is_axiom=True)
        self.reasoner.add_formula("forall x: dog(x) → animal(x)", is_axiom=True)
        
        # 查询
        result = self.reasoner.query("animal(tom)")
        
        self.assertTrue(result["success"])
        self.assertIn("animal(tom)", result.get("inferred_facts", []))
        
        logger.info("简单推理测试通过")


class TestNeuralConceptGrounder(unittest.TestCase):
    """测试神经概念接地器"""
    
    def setUp(self):
        """测试前设置"""
        self.grounder = NeuralConceptGrounder(
            feature_dim=64,
            embedding_dim=32,
            concept_dim=16,
            learning_rate=0.001,
            similarity_threshold=0.7
        )
    
    def test_initialization(self):
        """测试接地器初始化"""
        self.assertIsNotNone(self.grounder)
        self.assertIsNotNone(self.grounder.embedding_model)
        self.assertIsNotNone(self.grounder.concept_hierarchy)
        
        # 检查概念库
        self.assertIsNotNone(self.grounder.concept_library)
        
        logger.info("神经概念接地器初始化测试通过")
    
    def test_concept_grounding(self):
        """测试概念接地"""
        # 创建模拟神经特征
        feature_dim = 64
        test_features = np.random.randn(10, feature_dim).astype(np.float32)
        
        # 接地测试 - 使用ground_feature_to_concept方法
        grounding_result = self.grounder.ground_feature_to_concept(
            feature=test_features,
            grounding_type=GroundingType.PERCEPTUAL,
            k=3,
            return_similarities=True
        )
        
        self.assertIsNotNone(grounding_result)
        self.assertIn("concepts", grounding_result)
        self.assertIn("similarities", grounding_result)
        
        logger.info("概念接地测试通过")


class TestAbductiveReasoningEngine(unittest.TestCase):
    """测试溯因推理引擎"""
    
    def setUp(self):
        """测试前设置"""
        self.engine = AbductiveReasoningEngine(
            max_hypotheses=5,
            consistency_threshold=0.7,
            simplicity_weight=0.3,
            consistency_weight=0.4,
            completeness_weight=0.3
        )
    
    def test_initialization(self):
        """测试引擎初始化"""
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.engine.hypothesis_generator)
        self.assertIsNotNone(self.engine.consistency_checker)
        self.assertIsNotNone(self.engine.explanation_evaluator)
        
        logger.info("溯因推理引擎初始化测试通过")
    
    def test_abductive_reasoning(self):
        """测试溯因推理"""
        # 定义观察和背景知识
        observation = "grass_is_wet"
        background_knowledge = [
            "if_rained_then_grass_is_wet",
            "if_sprinkler_on_then_grass_is_wet"
        ]
        
        # 执行溯因推理
        result = self.engine.abduce(
            observations=[observation],
            context={"background_knowledge": background_knowledge},
            max_depth=3
        )
        
        self.assertIsNotNone(result)
        self.assertIn("hypotheses", result)
        self.assertIn("best_explanation", result)
        
        logger.info("溯因推理测试通过")


class TestIntegration(unittest.TestCase):
    """集成测试：测试各个组件的协同工作"""
    
    def test_neuro_symbolic_pipeline(self):
        """测试完整神经符号推理流程"""
        # 步骤1: 初始化所有组件
        framework = NeuralSymbolicUnifiedFramework(
            input_dim=128,
            hidden_dim=64,
            symbol_dim=32,
            learning_rate=0.001
        )
        
        reasoner = FirstOrderLogicReasoner()
        grounder = NeuralConceptGrounder(
            feature_dim=128,
            embedding_dim=64,
            concept_dim=32,
            learning_rate=0.001
        )
        
        abductive_engine = AbductiveReasoningEngine(
            max_hypotheses=5,
            consistency_threshold=0.7
        )
        
        # 步骤2: 模拟神经输入处理
        input_data = self._deterministic_randn((1, 128), seed_prefix="abductive_input")  # 模拟输入
        with torch.no_grad():
            neural_rep = framework.neural_encoder(input_data.to(framework.device))
        
        self.assertIsNotNone(neural_rep)
        
        # 步骤3: 概念接地
        features = neural_rep.cpu().numpy()
        grounding_result = grounder.ground_feature_to_concept(
            feature=features,
            grounding_type=GroundingType.PERCEPTUAL,
            k=3,
            return_similarities=True
        )
        
        self.assertIsNotNone(grounding_result)
        
        # 步骤4: 符号推理
        # 添加一些逻辑规则和事实
        reasoner.add_formula("object(ball)", is_axiom=False)
        reasoner.add_formula("red(ball)", is_axiom=False)
        reasoner.add_formula("forall x: object(x) ∧ red(x) → colored_object(x)", is_axiom=True)
        
        inference_result = reasoner.query("colored_object(ball)")
        
        self.assertTrue(inference_result["success"])
        
        # 步骤5: 溯因推理
        # 模拟一个需要解释的观察
        abductive_result = abductive_engine.abduce(
            observations=["object_moved"],
            context={"background_knowledge": ["if_pushed_then_object_moves", "if_wind_then_object_moves"]},
            max_depth=2
        )
        
        self.assertIsNotNone(abductive_result)
        
        # 验证整个流程成功
        self.assertIsNotNone(neural_rep)
        self.assertIsNotNone(grounding_result)
        self.assertTrue(inference_result["success"])
        self.assertIsNotNone(abductive_result)
        
        print("\n神经符号完整推理流程测试通过:")
        print(f"  神经编码: {neural_rep.shape if neural_rep is not None else 'N/A'}")
        print(f"  概念接地: {len(grounding_result.get('concepts', [])) if grounding_result else 0} 个概念")
        print(f"  逻辑推理: {inference_result['success']}")
        print(f"  溯因推理: {abductive_result.get('best_explanation', 'N/A')}")


class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def test_neural_encoding_performance(self):
        """测试神经编码性能"""
        framework = NeuralSymbolicUnifiedFramework(
            input_dim=512,
            hidden_dim=256,
            symbol_dim=128,
            learning_rate=0.001
        )
        
        # 测试不同批量大小的性能
        batch_sizes = [1, 8, 32, 128]
        
        for batch_size in batch_sizes:
            input_data = self._deterministic_randn((batch_size, 512), seed_prefix=f"perf_input_{batch_size}")
            
            start_time = time.time()
            with torch.no_grad():
                output = framework.neural_encoder(input_data.to(framework.device))
            end_time = time.time()
            
            processing_time = end_time - start_time
            throughput = batch_size / processing_time if processing_time > 0 else 0
            
            print(f"批大小 {batch_size}: 处理时间 {processing_time:.4f}秒, 吞吐量 {throughput:.2f}样本/秒")
            
            # 验证输出形状
            self.assertEqual(output.shape[0], batch_size)
            self.assertEqual(output.shape[1], framework.hidden_dim // 2)
    
    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # 初始化前的内存
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 初始化多个组件
        components = []
        for i in range(5):
            framework = NeuralSymbolicUnifiedFramework(
                input_dim=256,
                hidden_dim=128,
                symbol_dim=64,
                learning_rate=0.001
            )
            components.append(framework)
        
        # 初始化后的内存
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        print(f"内存使用: 初始化前 {memory_before:.2f}MB, 初始化后 {memory_after:.2f}MB, 增加 {memory_increase:.2f}MB")
        
        # 清理
        del components
        
        logger.info("内存使用测试完成")


class TestConsistency(unittest.TestCase):
    """一致性测试"""
    
    def test_neural_symbol_consistency(self):
        """测试神经符号一致性"""
        framework = NeuralSymbolicUnifiedFramework(
            input_dim=128,
            hidden_dim=64,
            symbol_dim=32,
            learning_rate=0.001
        )
        
        # 创建测试输入
        input_data = self._deterministic_randn((1, 128), seed_prefix="integration_input")
        
        # 前向传播：神经 → 符号
        with torch.no_grad():
            neural_rep = framework.neural_encoder(input_data.to(framework.device))
        
        # 符号提取（简化）
        # 在实际系统中，这里会提取符号表示
        
        # 反向传播：符号 → 神经
        # 在实际系统中，这里会将符号约束应用于神经表示
        
        # 一致性检查：确保神经和符号表示一致
        # 这里可以添加具体的一致性检查逻辑
        
        logger.info("神经符号一致性测试框架完成")


def run_all_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    
    # 添加所有测试类
    test_classes = [
        TestNeuralSymbolicUnifiedFramework,
        TestFirstOrderLogicReasoner,
        TestNeuralConceptGrounder,
        TestAbductiveReasoningEngine,
        TestIntegration,
        TestPerformance,
        TestConsistency
    ]
    
    suites = []
    for test_class in test_classes:
        suite = loader.loadTestsFromTestCase(test_class)
        suites.append(suite)
    
    # 组合所有测试套件
    all_tests = unittest.TestSuite(suites)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(all_tests)
    
    # 输出统计信息
    print("\n" + "="*60)
    print("神经符号统一框架集成测试完成")
    print(f"测试用例: {result.testsRun}个")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}个")
    print(f"失败: {len(result.failures)}个")
    print(f"错误: {len(result.errors)}个")
    
    if result.failures:
        print("\n失败详情:")
        for test, traceback in result.failures:
            print(f"  {test}:")
            for line in traceback.split('\n')[-3:]:
                print(f"    {line}")
    
    if result.errors:
        print("\n错误详情:")
        for test, traceback in result.errors:
            print(f"  {test}:")
            for line in traceback.split('\n')[-3:]:
                print(f"    {line}")
    
    return result


if __name__ == "__main__":
    # 运行所有测试
    result = run_all_tests()
    
    # 根据测试结果返回适当的退出码
    sys.exit(0 if result.wasSuccessful() else 1)