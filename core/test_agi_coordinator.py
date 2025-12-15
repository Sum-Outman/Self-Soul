#!/usr/bin/env python3
"""
AGI协调器测试脚本
测试模型加载、注册和协调功能
"""

import sys
import os
# 添加根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from core.agi_coordinator import AGICoordinator
from core.model_registry import ModelRegistry
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_registry():
    """测试模型注册表功能"""
    print("=== 测试模型注册表 ===")
    try:
        registry = ModelRegistry()
        print("✓ 模型注册表初始化成功")
        
        # 测试获取支持的模型类型
        model_types = registry.get_supported_model_types()
        print(f"✓ 支持的模型类型: {len(model_types)} 种")
        for model_type in model_types:
            print(f"  - {model_type}")
        
        # 测试模型加载
        print("\n=== 测试模型加载 ===")
        loaded_models = registry.load_all_models()
        print(f"✓ 成功加载模型: {len(loaded_models)} 个")
        
        for model_name, model_instance in loaded_models.items():
            print(f"  - {model_name}: {type(model_instance).__name__}")
            
        return True
        
    except Exception as e:
        print(f"✗ 模型注册表测试失败: {e}")
        return False

def test_agi_coordinator():
    """测试AGI协调器功能"""
    print("\n=== 测试AGI协调器 ===")
    try:
        coordinator = AGICoordinator()
        print("✓ AGI协调器初始化成功")
        
        # 测试模型管理
        print("\n=== 测试模型管理 ===")
        models = coordinator.get_available_models()
        print(f"✓ 可用模型数量: {len(models)}")
        
        for model_name in models:
            model = coordinator.get_model(model_name)
            if model:
                print(f"  - {model_name}: ✓ 可访问")
            else:
                print(f"  - {model_name}: ✗ 无法访问")
        
        # 测试多模态处理能力
        print("\n=== 测试多模态处理能力 ===")
        capabilities = coordinator.get_capabilities()
        print("✓ 系统能力:")
        for capability, enabled in capabilities.items():
            status = "✓" if enabled else "✗"
            print(f"  - {capability}: {status}")
        
        return True
        
    except Exception as e:
        print(f"✗ AGI协调器测试失败: {e}")
        return False

def test_model_interaction():
    """测试模型交互功能"""
    print("\n=== 测试模型交互 ===")
    try:
        coordinator = AGICoordinator()
        
        # 测试文本处理
        print("测试文本处理...")
        text_result = coordinator.process_text("Hello, this is a test message.")
        print(f"✓ 文本处理结果: {type(text_result)}")
        
        # 测试知识库功能
        print("测试知识库功能...")
        knowledge_status = coordinator.check_knowledge_base()
        print(f"✓ 知识库状态: {knowledge_status}")
        
        # 测试自主学习功能
        print("测试自主学习功能...")
        learning_status = coordinator.check_autonomous_learning()
        print(f"✓ 自主学习状态: {learning_status}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型交互测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始AGI协调器综合测试...")
    print("=" * 50)
    
    tests_passed = 0
    tests_total = 3
    
    # 运行测试
    if test_model_registry():
        tests_passed += 1
    
    if test_agi_coordinator():
        tests_passed += 1
        
    if test_model_interaction():
        tests_passed += 1
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print(f"测试完成: {tests_passed}/{tests_total} 通过")
    
    if tests_passed == tests_total:
        print("🎉 所有测试通过！AGI协调器功能正常")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步调试")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
