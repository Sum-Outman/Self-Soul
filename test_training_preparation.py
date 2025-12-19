"""
训练准备流程测试脚本
Training Preparation Process Test Script

测试TrainingPreparation类的完整性和可靠性
Test the integrity and reliability of TrainingPreparation class
"""

import sys
import os
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.training_preparation import TrainingPreparation, create_training_preparation
from core.model_registry import ModelRegistry
from core.training_manager import TrainingManager


def test_environment_preparation():
    """测试环境准备功能
    Test environment preparation functionality
    """
    print("=== 测试环境准备功能 ===")
    
    try:
        # 创建训练准备实例
        preparation = create_training_preparation()
        if not preparation:
            print("❌ 创建训练准备实例失败")
            return False
        
        # 测试环境准备
        result = preparation.prepare_training_environment()
        
        print(f"环境准备结果: {result['success']}")
        print(f"消息: {result['message']}")
        
        if result['success']:
            print("✅ 环境准备测试通过")
            return True
        else:
            print("❌ 环境准备测试失败")
            print(f"错误: {result['errors']}")
            return False
            
    except Exception as e:
        print(f"❌ 环境准备测试异常: {str(e)}")
        return False


def test_data_preparation():
    """测试数据准备功能
    Test data preparation functionality
    """
    print("\n=== 测试数据准备功能 ===")
    
    try:
        # 创建训练准备实例
        preparation = create_training_preparation()
        if not preparation:
            print("❌ 创建训练准备实例失败")
            return False
        
        # 创建测试数据
        test_data = [
            {"input": [1, 2, 3], "output": 1},
            {"input": [4, 5, 6], "output": 0},
            {"input": [7, 8, 9], "output": 1},
            {"input": [10, 11, 12], "output": 0}
        ]
        
        # 测试数据准备
        result = preparation.prepare_training_data("test_model", test_data)
        
        print(f"数据准备结果: {result['success']}")
        print(f"消息: {result['message']}")
        
        if result['success']:
            print("✅ 数据准备测试通过")
            return True
        else:
            print("❌ 数据准备测试失败")
            print(f"错误: {result['errors']}")
            return False
            
    except Exception as e:
        print(f"❌ 数据准备测试异常: {str(e)}")
        return False


def test_model_configuration():
    """测试模型配置功能
    Test model configuration functionality
    """
    print("\n=== 测试模型配置功能 ===")
    
    try:
        # 创建训练准备实例
        preparation = create_training_preparation()
        if not preparation:
            print("❌ 创建训练准备实例失败")
            return False
        
        # 测试模型配置
        custom_params = {
            'learning_rate': 0.01,
            'batch_size': 16,
            'epochs': 50
        }
        
        # 这里使用一个不存在的模型ID进行测试
        result = preparation.prepare_model_configuration("nonexistent_model", custom_params)
        
        print(f"模型配置结果: {result['success']}")
        print(f"消息: {result['message']}")
        
        # 对于不存在的模型，配置应该失败，这是预期的
        if not result['success'] and '未找到' in result['message']:
            print("✅ 模型配置测试通过（正确处理了不存在的模型）")
            return True
        else:
            print("❌ 模型配置测试失败")
            print(f"错误: {result['errors']}")
            return False
            
    except Exception as e:
        print(f"❌ 模型配置测试异常: {str(e)}")
        return False


def test_training_context():
    """测试训练上下文功能
    Test training context functionality
    """
    print("\n=== 测试训练上下文功能 ===")
    
    try:
        # 创建训练准备实例
        preparation = create_training_preparation()
        if not preparation:
            print("❌ 创建训练准备实例失败")
            return False
        
        # 测试训练上下文准备
        model_ids = ["model1", "model2", "model3"]
        result = preparation.prepare_training_context(model_ids, "federated")
        
        print(f"训练上下文准备结果: {result['success']}")
        print(f"消息: {result['message']}")
        
        if result['success']:
            print("✅ 训练上下文准备测试通过")
            return True
        else:
            print("❌ 训练上下文准备测试失败")
            print(f"错误: {result['errors']}")
            return False
            
    except Exception as e:
        print(f"❌ 训练上下文准备测试异常: {str(e)}")
        return False


def test_complete_preparation():
    """测试完整训练准备流程
    Test complete training preparation workflow
    """
    print("\n=== 测试完整训练准备流程 ===")
    
    try:
        # 创建训练准备实例
        preparation = create_training_preparation()
        if not preparation:
            print("❌ 创建训练准备实例失败")
            return False
        
        # 准备测试数据
        model_ids = ["test_model1", "test_model2"]
        raw_data = {
            "test_model1": [
                {"input": [1, 2, 3], "output": 1},
                {"input": [4, 5, 6], "output": 0}
            ],
            "test_model2": {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6]
            }
        }
        
        custom_params = {
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        # 执行完整准备流程
        start_time = time.time()
        result = preparation.execute_complete_preparation(
            model_ids, raw_data, custom_params, "federated"
        )
        duration = time.time() - start_time
        
        print(f"完整准备流程结果: {result['success']}")
        print(f"消息: {result['message']}")
        print(f"耗时: {duration:.2f} 秒")
        print(f"总体状态: {result['overall_status']}")
        
        # 输出各阶段结果
        for phase, phase_result in result['preparation_phases'].items():
            if isinstance(phase_result, dict):
                print(f"  {phase}: {phase_result.get('success', 'N/A')}")
            else:
                print(f"  {phase}: 包含 {len(phase_result)} 个模型的结果")
        
        if result['success']:
            print("✅ 完整训练准备流程测试通过")
            return True
        else:
            print("❌ 完整训练准备流程测试失败")
            print(f"错误: {result['errors']}")
            return False
            
    except Exception as e:
        print(f"❌ 完整训练准备流程测试异常: {str(e)}")
        return False


def test_error_handling():
    """测试错误处理功能
    Test error handling functionality
    """
    print("\n=== 测试错误处理功能 ===")
    
    try:
        # 创建训练准备实例
        preparation = create_training_preparation()
        if not preparation:
            print("❌ 创建训练准备实例失败")
            return False
        
        # 测试空数据的情况
        result = preparation.prepare_training_data("test_model", None)
        
        print(f"空数据处理结果: {result['success']}")
        print(f"消息: {result['message']}")
        
        # 空数据应该被正确处理
        if not result['success'] and '数据为空' in result['message']:
            print("✅ 错误处理测试通过（正确处理了空数据）")
            return True
        else:
            print("❌ 错误处理测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 错误处理测试异常: {str(e)}")
        return False


def test_preparation_summary():
    """测试准备总结功能
    Test preparation summary functionality
    """
    print("\n=== 测试准备总结功能 ===")
    
    try:
        # 创建训练准备实例
        preparation = create_training_preparation()
        if not preparation:
            print("❌ 创建训练准备实例失败")
            return False
        
        # 执行一些准备操作
        preparation.prepare_training_environment()
        
        # 获取准备总结
        summary = preparation.get_preparation_summary()
        
        print(f"总步骤数: {summary['total_steps']}")
        print(f"成功步骤数: {summary['successful_steps']}")
        print(f"失败步骤数: {summary['failed_steps']}")
        print(f"成功率: {summary['success_rate']:.1%}")
        
        if summary['total_steps'] > 0:
            print("✅ 准备总结测试通过")
            return True
        else:
            print("❌ 准备总结测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 准备总结测试异常: {str(e)}")
        return False


def main():
    """主测试函数
    Main test function
    """
    print("开始训练准备流程测试...\n")
    
    # 执行所有测试
    tests = [
        ("环境准备", test_environment_preparation),
        ("数据准备", test_data_preparation),
        ("模型配置", test_model_configuration),
        ("训练上下文", test_training_context),
        ("完整流程", test_complete_preparation),
        ("错误处理", test_error_handling),
        ("准备总结", test_preparation_summary)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {str(e)}")
            results.append((test_name, False))
    
    # 输出测试结果总结
    print("\n" + "="*50)
    print("测试结果总结:")
    print("="*50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n🎉 所有测试通过！训练准备流程功能完整可靠。")
        return True
    else:
        print(f"\n⚠️  {failed} 个测试失败，需要进一步调试。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)