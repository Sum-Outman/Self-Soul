#!/usr/bin/env python3
"""
测试协调方法修复
"""

import sys
import os
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_coordinate_task_fix():
    """测试coordinate_task方法修复"""
    print("=== 测试coordinate_task方法修复 ===\n")
    
    try:
        from core.models.manager.unified_manager_model import UnifiedManagerModel
        
        # 创建管理器实例
        config = {"model_id": "test_fix", "test_mode": True}
        manager = UnifiedManagerModel(config)
        print(f"管理器实例创建成功")
        
        # 测试1: 使用子类签名调用 (required_models, priority)
        print("\n测试1: 使用子类签名调用")
        try:
            result1 = manager.coordinate_task(
                task_description="测试视觉-语言协作",
                required_models=["language", "vision"],
                priority=5
            )
            print(f"   调用成功 - 状态: {result1.get('status', 'unknown')}")
            print(f"   消息: {result1.get('message', '无消息')}")
        except Exception as e:
            print(f"   调用失败: {type(e).__name__}: {e}")
        
        # 测试2: 使用父类签名调用 (required_resources)
        print("\n测试2: 使用父类签名调用")
        try:
            result2 = manager.coordinate_task(
                task_description="测试任务",
                required_resources={"models": ["audio", "emotion"]}
            )
            print(f"   调用成功 - 状态: {result2.get('status', 'unknown')}")
            print(f"   消息: {result2.get('message', '无消息')}")
        except Exception as e:
            print(f"   调用失败: {type(e).__name__}: {e}")
        
        # 测试3: 使用位置参数调用
        print("\n测试3: 使用位置参数调用")
        try:
            result3 = manager.coordinate_task(
                "测试位置参数",
                ["knowledge", "programming"],
                3
            )
            print(f"   调用成功 - 状态: {result3.get('status', 'unknown')}")
            print(f"   消息: {result3.get('message', '无消息')}")
        except Exception as e:
            print(f"   调用失败: {type(e).__name__}: {e}")
        
        # 测试4: 测试enhanced_coordinate_task方法
        print("\n测试4: 测试enhanced_coordinate_task方法")
        try:
            if hasattr(manager, 'enhanced_coordinate_task'):
                result4 = manager.enhanced_coordinate_task(
                    task_description="测试增强协调",
                    required_models=["language", "vision", "audio"],
                    priority=5,
                    collaboration_mode="smart"
                )
                print(f"   调用成功 - 状态: {result4.get('status', 'unknown')}")
                print(f"   消息: {result4.get('message', '无消息')}")
            else:
                print("   方法不存在")
        except Exception as e:
            print(f"   调用失败: {type(e).__name__}: {e}")
        
        # 检查子模型状态
        print(f"\n子模型字典: {list(manager.sub_models.keys())}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_method_existence():
    """检查方法是否存在"""
    print("\n=== 检查方法存在性 ===\n")
    
    try:
        from core.models.manager.unified_manager_model import UnifiedManagerModel
        
        # 检查类方法
        print("类方法检查:")
        print(f"  coordinate_task: {'coordinate_task' in UnifiedManagerModel.__dict__}")
        print(f"  enhanced_coordinate_task: {'enhanced_coordinate_task' in UnifiedManagerModel.__dict__}")
        print(f"  _discover_and_register_sub_models: {'_discover_and_register_sub_models' in UnifiedManagerModel.__dict__}")
        
        # 创建实例检查
        config = {"model_id": "test_methods", "test_mode": True}
        manager = UnifiedManagerModel(config)
        
        print("\n实例方法检查:")
        print(f"  coordinate_task: {hasattr(manager, 'coordinate_task')}")
        print(f"  enhanced_coordinate_task: {hasattr(manager, 'enhanced_coordinate_task')}")
        print(f"  _discover_and_register_sub_models: {hasattr(manager, '_discover_and_register_sub_models')}")
        
        # 检查方法来源
        if hasattr(manager, 'coordinate_task'):
            import inspect
            print(f"\ncoordinate_task方法来源:")
            print(f"  模块: {inspect.getmodule(manager.coordinate_task)}")
            print(f"  文件: {inspect.getfile(manager.coordinate_task) if hasattr(manager.coordinate_task, '__code__') else 'N/A'}")
        
    except Exception as e:
        print(f"检查失败: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_coordinate_task_fix()
    check_method_existence()