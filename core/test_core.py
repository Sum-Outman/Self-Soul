#!/usr/bin/env python3
"""
统一认知架构和核心模块测试脚本
测试增强后的AGI核心模块功能
"""

import sys
import os
import json
import logging
from typing import Dict, Any, List

# 添加根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# 导入要测试的模块
from core.unified_cognitive_architecture import (
    UnifiedCognitiveArchitecture,
    EnhancedSelfAwarenessModule,
    EnhancedNeuroSymbolicReasoner,
    ArchitectureAdjuster,
    CognitiveArchitectureMonitor,
    NeuralSymbolicBridge,
    CausalReasoningModule,
    CounterfactualEngine
)
from core.advanced_reasoning import EnhancedAdvancedReasoningEngine as EnhancedReasoningEngine
from core.adaptive_learning_engine import EnhancedAdaptiveLearningEngine
from core.meta_learning_system import EnhancedMetaLearningSystem
from core.creative_problem_solver import EnhancedCreativeProblemSolver
from core.system_monitor import SystemMonitor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoreModuleTester:
    """核心模块测试器"""
    
    def __init__(self):
        self.results = {}
        self.test_count = 0
        self.passed_count = 0
    
    def run_test(self, test_name: str, test_func):
        """运行单个测试"""
        self.test_count += 1
        try:
            result = test_func()
            if result:
                print(f"✓ [{test_name}] 通过")
                self.passed_count += 1
                self.results[test_name] = "通过"
            else:
                print(f"✗ [{test_name}] 失败")
                self.results[test_name] = "失败"
        except Exception as e:
            print(f"✗ [{test_name}] 异常: {e}")
            self.results[test_name] = f"异常: {e}"
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n" + "=" * 60)
        print("测试摘要:")
        print("=" * 60)
        for test_name, result in self.results.items():
            status = "✓" if result == "通过" else "✗"
            print(f"{status} {test_name}: {result}")
        print(f"\n总计: {self.passed_count}/{self.test_count} 通过 ({self.passed_count/self.test_count*100:.1f}%)")

def test_unified_architecture_initialization():
    """测试统一认知架构初始化"""
    print("\n=== 测试统一认知架构初始化 ===")
    try:
        architecture = UnifiedCognitiveArchitecture()
        print("✓ UnifiedCognitiveArchitecture 初始化成功")
        
        # 检查核心组件
        components = [
            ("NeuralEmbeddingSpace", architecture.neural_space),
            ("SymbolicMapper", architecture.symbolic_mapper),
            ("CrossModalReasoner", architecture.cross_modal_reasoner),
            ("NeuroSymbolicReasoner", architecture.neuro_symbolic_reasoner),
            ("EnhancedSelfAwarenessModule", architecture.enhanced_self_awareness),
            ("EnhancedNeuroSymbolicReasoner", architecture.enhanced_neuro_symbolic_reasoner),
            ("ArchitectureAdjuster", architecture.architecture_adjuster),
            ("CognitiveArchitectureMonitor", architecture.cognitive_monitor)
        ]
        
        for name, component in components:
            if component is not None:
                print(f"✓ {name} 已初始化")
            else:
                print(f"✗ {name} 未初始化")
                return False
        
        return True
    except Exception as e:
        print(f"✗ 统一认知架构初始化失败: {e}")
        return False

def test_enhanced_self_awareness():
    """测试增强自我意识模块"""
    print("\n=== 测试增强自我意识模块 ===")
    try:
        module = EnhancedSelfAwarenessModule()
        
        # 测试自我状态评估
        self_state = module.assess_self_state()
        print(f"✓ 自我状态评估完成: {len(self_state)} 个指标")
        
        # 测试元认知监控
        task = "解决数学问题"
        metacognitive_insight = module.monitor_metacognition(task)
        print(f"✓ 元认知监控完成: {metacognitive_insight}")
        
        # 测试内省分析
        reflection = module.perform_introspective_analysis()
        print(f"✓ 内省分析完成: {reflection[:100]}...")
        
        return True
    except Exception as e:
        print(f"✗ 增强自我意识模块测试失败: {e}")
        return False

def test_enhanced_neuro_symbolic_reasoner():
    """测试增强神经符号推理器"""
    print("\n=== 测试增强神经符号推理器 ===")
    try:
        reasoner = EnhancedNeuroSymbolicReasoner()
        
        # 测试神经符号映射
        neural_input = {"text": "猫是动物", "confidence": 0.9}
        symbolic_output = reasoner.neural_to_symbolic(neural_input)
        print(f"✓ 神经符号映射完成: {symbolic_output}")
        
        # 测试因果推理
        cause = "下雨"
        effect = "地面湿"
        causal_strength = reasoner.infer_causal_strength(cause, effect)
        print(f"✓ 因果推理完成: {cause} -> {effect} = {causal_strength}")
        
        # 测试反事实推理
        factual = {"action": "带伞", "outcome": "没淋湿"}
        counterfactual = {"action": "没带伞", "outcome": "淋湿"}
        cf_result = reasoner.evaluate_counterfactual(factual, counterfactual)
        print(f"✓ 反事实推理完成: {cf_result}")
        
        return True
    except Exception as e:
        print(f"✗ 增强神经符号推理器测试失败: {e}")
        return False

def test_architecture_adjuster():
    """测试架构调整器"""
    print("\n=== 测试架构调整器 ===")
    try:
        adjuster = ArchitectureAdjuster()
        
        # 测试架构评估
        current_config = {
            "neural_layers": 10,
            "symbolic_rules": 50,
            "attention_heads": 8
        }
        performance_metrics = {
            "accuracy": 0.85,
            "latency": 120,
            "memory_usage": 1024
        }
        
        adjustment = adjuster.evaluate_and_adjust(current_config, performance_metrics)
        print(f"✓ 架构评估完成: {adjustment}")
        
        # 测试动态调整
        new_config = adjuster.dynamic_adjustment(current_config, "high_load")
        print(f"✓ 动态调整完成: {new_config}")
        
        return True
    except Exception as e:
        print(f"✗ 架构调整器测试失败: {e}")
        return False

def test_enhanced_reasoning_engine():
    """测试增强推理引擎"""
    print("\n=== 测试增强推理引擎 ===")
    try:
        engine = EnhancedReasoningEngine()
        
        # 测试逻辑推导
        premises = ["所有哺乳动物都有脊椎", "狗是哺乳动物"]
        conclusion = engine.logical_deduction(premises, "狗有脊椎")
        print(f"✓ 逻辑推导完成: {conclusion}")
        
        # 测试定理证明
        theorem = "勾股定理: a² + b² = c²"
        proof = engine.theorem_proving(theorem)
        print(f"✓ 定理证明尝试: {proof[:100]}...")
        
        # 测试结构因果模型
        variables = ["X", "Y", "Z"]
        relationships = [("X", "Y"), ("Y", "Z")]
        scm_result = engine.structural_causal_modeling(variables, relationships)
        print(f"✓ 结构因果建模完成: {scm_result}")
        
        return True
    except Exception as e:
        print(f"✗ 增强推理引擎测试失败: {e}")
        return False

def test_enhanced_adaptive_learning_engine():
    """测试增强自适应学习引擎"""
    print("\n=== 测试增强自适应学习引擎 ===")
    try:
        engine = EnhancedAdaptiveLearningEngine()
        
        # 测试强化学习优化
        state = {"loss": 0.5, "accuracy": 0.8, "epoch": 10}
        action = engine.reinforcement_learning_optimizer.select_action(state)
        print(f"✓ 强化学习优化完成: 选择动作 {action}")
        
        # 测试元学习分析
        task_family = "分类任务"
        meta_analysis = engine.meta_learning_analyzer.analyze_task_family(task_family)
        print(f"✓ 元学习分析完成: {meta_analysis}")
        
        # 测试多目标优化
        objectives = ["accuracy", "latency", "memory"]
        optimized_params = engine.multi_objective_optimization(objectives)
        print(f"✓ 多目标优化完成: {optimized_params}")
        
        return True
    except Exception as e:
        print(f"✗ 增强自适应学习引擎测试失败: {e}")
        return False

def test_enhanced_meta_learning_system():
    """测试增强元学习系统"""
    print("\n=== 测试增强元学习系统 ===")
    try:
        system = EnhancedMetaLearningSystem()
        
        # 测试MAML算法
        task = {"train": ["task1", "task2"], "test": ["task3"]}
        maml_result = system.maml_algorithm.adapt_to_task(task)
        print(f"✓ MAML算法完成: {maml_result}")
        
        # 测试任务分布学习
        task_distribution = system.task_distribution_learner.learn_distribution()
        print(f"✓ 任务分布学习完成: {task_distribution}")
        
        # 测试元学习器
        meta_learner_result = system.meta_learner.learn_meta_parameters()
        print(f"✓ 元学习器完成: {meta_learner_result}")
        
        return True
    except Exception as e:
        print(f"✗ 增强元学习系统测试失败: {e}")
        return False

def test_enhanced_creative_problem_solver():
    """测试增强创造性问题解决器"""
    print("\n=== 测试增强创造性问题解决器 ===")
    try:
        solver = EnhancedCreativeProblemSolver()
        
        # 测试创造性生成
        problem = "如何减少城市交通拥堵？"
        creative_solutions = solver.creative_generator.generate_solutions(problem, num_solutions=3)
        print(f"✓ 创造性生成完成: 生成 {len(creative_solutions)} 个解决方案")
        
        # 测试创造性评估
        evaluation = solver.creative_evaluator.evaluate_solutions(creative_solutions)
        print(f"✓ 创造性评估完成: {evaluation}")
        
        # 测试六种创造性方法
        for method in solver.creative_methods:
            result = method.apply(problem)
            print(f"✓ 创造性方法 '{method.name}' 完成: {result[:50]}...")
        
        return True
    except Exception as e:
        print(f"✗ 增强创造性问题解决器测试失败: {e}")
        return False

def test_system_monitor_agi_metrics():
    """测试系统监控器的AGI指标"""
    print("\n=== 测试系统监控器AGI指标 ===")
    try:
        monitor = SystemMonitor()
        
        # 检查AGI指标数据结构
        if hasattr(monitor, 'agi_metrics'):
            print(f"✓ AGI指标数据结构存在: {monitor.agi_metrics}")
            
            # 测试指标更新
            monitor.agi_metrics['cognitive_load'] = 0.7
            monitor.agi_metrics['learning_efficiency'] = 0.85
            monitor.agi_metrics['decision_quality'] = 0.9
            
            print(f"✓ AGI指标更新完成: {monitor.agi_metrics}")
            return True
        else:
            print("✗ AGI指标数据结构不存在")
            return False
    except Exception as e:
        print(f"✗ 系统监控器测试失败: {e}")
        return False

def test_integration():
    """测试模块集成"""
    print("\n=== 测试模块集成 ===")
    try:
        # 创建统一认知架构实例
        architecture = UnifiedCognitiveArchitecture()
        
        # 模拟处理流程
        input_data = {
            "text": "人工智能会超越人类吗？",
            "context": "哲学和技术讨论"
        }
        
        # 神经处理
        neural_embedding = architecture.neural_space.encode(input_data["text"])
        print(f"✓ 神经嵌入完成: 维度 {len(neural_embedding)}")
        
        # 符号映射
        symbolic_representation = architecture.symbolic_mapper.map_to_symbols(neural_embedding)
        print(f"✓ 符号映射完成: {symbolic_representation}")
        
        # 神经符号推理
        reasoning_result = architecture.enhanced_neuro_symbolic_reasoner.integrated_reasoning(
            neural_embedding, symbolic_representation
        )
        print(f"✓ 神经符号推理完成: {reasoning_result}")
        
        # 自我意识评估
        self_state = architecture.enhanced_self_awareness.assess_self_state()
        print(f"✓ 自我意识评估完成: {len(self_state)} 个指标")
        
        return True
    except Exception as e:
        print(f"✗ 模块集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始AGI核心模块综合测试")
    print("=" * 60)
    
    tester = CoreModuleTester()
    
    # 运行所有测试
    tests = [
        ("统一认知架构初始化", test_unified_architecture_initialization),
        ("增强自我意识模块", test_enhanced_self_awareness),
        ("增强神经符号推理器", test_enhanced_neuro_symbolic_reasoner),
        ("架构调整器", test_architecture_adjuster),
        ("增强推理引擎", test_enhanced_reasoning_engine),
        ("增强自适应学习引擎", test_enhanced_adaptive_learning_engine),
        ("增强元学习系统", test_enhanced_meta_learning_system),
        ("增强创造性问题解决器", test_enhanced_creative_problem_solver),
        ("系统监控器AGI指标", test_system_monitor_agi_metrics),
        ("模块集成", test_integration)
    ]
    
    for test_name, test_func in tests:
        tester.run_test(test_name, test_func)
    
    # 打印摘要
    tester.print_summary()
    
    # 保存测试结果
    with open("core/test_results.json", "w") as f:
        json.dump(tester.results, f, indent=2, ensure_ascii=False)
    
    # 返回总体结果
    return tester.passed_count == tester.test_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
