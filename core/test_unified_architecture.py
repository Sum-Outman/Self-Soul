#!/usr/bin/env python3
"""
测试统一认知架构的基本功能
验证所有组件能够正确集成和运行
"""

import sys
import os
import torch
import numpy as np

# 添加父目录到路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_cognitive_architecture import UnifiedCognitiveArchitecture

def test_architecture_initialization():
    """测试架构初始化"""
    print("测试架构初始化...")
    try:
        # 创建架构实例
        architecture = UnifiedCognitiveArchitecture()
        print("✓ 架构初始化成功")
        
        # 检查主要组件是否存在
        assert hasattr(architecture, 'text_encoder'), "文本编码器缺失"
        assert hasattr(architecture, 'vision_processor'), "视觉处理器缺失"
        assert hasattr(architecture, 'audio_processor'), "音频处理器缺失"
        assert hasattr(architecture, 'neural_reasoner'), "神经推理器缺失"
        assert hasattr(architecture, 'symbolic_mapper'), "符号映射器缺失"
        assert hasattr(architecture, 'meta_learner'), "元学习器缺失"
        assert hasattr(architecture, 'self_reflection'), "自我反思模块缺失"
        assert hasattr(architecture, 'training_coordinator'), "训练协调器缺失"
        assert hasattr(architecture, 'communication_bus'), "通信总线缺失"
        assert hasattr(architecture, 'knowledge_sharing'), "知识共享模块缺失"
        
        print("✓ 所有主要组件都存在")
        return architecture
        
    except Exception as e:
        print(f"✗ 架构初始化失败: {e}")
        return None

def test_text_processing(architecture):
    """测试文本处理功能"""
    print("\n测试文本处理功能...")
    try:
        # 测试文本编码
        test_text = "这是一个测试句子，用于验证文本处理功能。"
        encoded_text = architecture.text_encoder.encode(test_text)
        
        print(f"✓ 文本编码成功")
        print(f"  输入文本: {test_text}")
        print(f"  编码形状: {encoded_text.shape}")
        
        # 测试文本解码
        decoded_text = architecture.text_encoder.decode(encoded_text)
        print(f"✓ 文本解码成功")
        print(f"  解码文本: {decoded_text}")
        
        return True
        
    except Exception as e:
        print(f"✗ 文本处理失败: {e}")
        return False

def test_neural_reasoning(architecture):
    """测试神经推理功能"""
    print("\n测试神经推理功能...")
    try:
        # 创建测试输入
        test_input = torch.randn(1, 512)  # 假设的输入向量
        
        # 测试推理
        reasoning_result = architecture.neural_reasoner.reason(test_input)
        
        print(f"✓ 神经推理成功")
        print(f"  推理结果形状: {reasoning_result.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 神经推理失败: {e}")
        return False

def test_symbolic_mapping(architecture):
    """测试符号映射功能"""
    print("\n测试符号映射功能...")
    try:
        # 创建测试输入
        test_vector = torch.randn(1, 512)
        
        # 测试符号映射
        symbolic_result = architecture.symbolic_mapper.map_to_symbols(test_vector)
        
        print(f"✓ 符号映射成功")
        print(f"  符号结果类型: {type(symbolic_result)}")
        
        # 测试概念学习
        new_concept = "测试概念"
        architecture.symbolic_mapper.learn_concept(new_concept, test_vector)
        print(f"✓ 概念学习成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 符号映射失败: {e}")
        return False

def test_meta_learning(architecture):
    """测试元学习功能"""
    print("\n测试元学习功能...")
    try:
        # 测试元学习
        learning_result = architecture.meta_learner.learn_from_experience()
        
        print(f"✓ 元学习成功")
        print(f"  学习结果: {learning_result}")
        
        return True
        
    except Exception as e:
        print(f"✗ 元学习失败: {e}")
        return False

def test_communication_bus(architecture):
    """测试通信总线功能"""
    print("\n测试通信总线功能...")
    try:
        # 测试消息发送
        test_message = {
            'type': 'test_message',
            'content': '测试通信总线功能',
            'sender': 'test_system',
            'timestamp': '2024-01-01 00:00:00'
        }
        
        # 发送消息
        architecture.communication_bus.send_message(test_message)
        print(f"✓ 消息发送成功")
        
        # 测试消息接收（模拟）
        received_messages = architecture.communication_bus.receive_messages('test_system')
        print(f"✓ 消息接收成功")
        print(f"  接收到的消息数量: {len(received_messages)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 通信总线测试失败: {e}")
        return False

def test_training_coordination(architecture):
    """测试训练协调功能"""
    print("\n测试训练协调功能...")
    try:
        # 测试训练会话创建
        session_id = architecture.training_coordinator.create_training_session(
            model_id='test_model',
            training_type='joint_training',
            participants=['model1', 'model2']
        )
        
        print(f"✓ 训练会话创建成功")
        print(f"  会话ID: {session_id}")
        
        # 测试资源分配
        resource_allocation = architecture.training_coordinator.resource_allocator.allocate_resources(
            session_id, 
            requirements={'gpu_memory': 4, 'cpu_cores': 2}
        )
        
        print(f"✓ 资源分配成功")
        print(f"  分配的资源: {resource_allocation}")
        
        return True
        
    except Exception as e:
        print(f"✗ 训练协调测试失败: {e}")
        return False

def test_knowledge_sharing(architecture):
    """测试知识共享功能"""
    print("\n测试知识共享功能...")
    try:
        # 测试知识共享
        knowledge_result = architecture.knowledge_sharing.share_knowledge(
            source_model='model1',
            target_model='model2',
            knowledge_type='concept_mapping'
        )
        
        print(f"✓ 知识共享成功")
        print(f"  共享结果: {knowledge_result}")
        
        return True
        
    except Exception as e:
        print(f"✗ 知识共享测试失败: {e}")
        return False

def test_architecture_integration():
    """测试架构整体集成"""
    print("\n测试架构整体集成...")
    try:
        # 初始化架构
        architecture = UnifiedCognitiveArchitecture()
        
        # 测试输入处理
        test_input = {
            'text': "测试输入文本",
            'vision': None,
            'audio': None
        }
        
        # 处理输入
        processing_result = architecture.process_input(test_input)
        print(f"✓ 输入处理成功")
        print(f"  处理结果类型: {type(processing_result)}")
        
        # 测试问题解决
        problem = "如何解决这个测试问题？"
        solution = architecture.solve_problem(problem)
        print(f"✓ 问题解决成功")
        print(f"  解决方案: {solution}")
        
        # 测试自我反思
        reflection_result = architecture.self_reflect()
        print(f"✓ 自我反思成功")
        print(f"  反思结果: {reflection_result}")
        
        return True
        
    except Exception as e:
        print(f"✗ 架构集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试统一认知架构...")
    print("=" * 50)
    
    # 测试架构初始化
    architecture = test_architecture_initialization()
    if not architecture:
        print("架构初始化失败，无法继续测试")
        return False
    
    # 运行各个功能测试
    tests_passed = 0
    total_tests = 8
    
    if test_text_processing(architecture):
        tests_passed += 1
    
    if test_neural_reasoning(architecture):
        tests_passed += 1
    
    if test_symbolic_mapping(architecture):
        tests_passed += 1
    
    if test_meta_learning(architecture):
        tests_passed += 1
    
    if test_communication_bus(architecture):
        tests_passed += 1
    
    if test_training_coordination(architecture):
        tests_passed += 1
    
    if test_knowledge_sharing(architecture):
        tests_passed += 1
    
    if test_architecture_integration():
        tests_passed += 1
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print(f"测试完成: {tests_passed}/{total_tests} 个测试通过")
    
    if tests_passed == total_tests:
        print("🎉 所有测试都通过了！统一认知架构功能正常。")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步修复。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
