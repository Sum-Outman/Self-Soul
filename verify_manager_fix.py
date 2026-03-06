#!/usr/bin/env python3
"""
验证管理器修复
"""

import sys
import os
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_fixes():
    """验证修复"""
    print("=== 验证管理器修复 ===\n")
    
    try:
        # 导入管理器
        from core.models.manager.unified_manager_model import UnifiedManagerModel
        
        # 创建实例
        config = {"model_id": "verify_fix", "test_mode": True}
        manager = UnifiedManagerModel(config)
        
        print("1. 基础检查:")
        print(f"   实例创建: 成功")
        print(f"   子模型字典大小: {len(manager.sub_models)}")
        print(f"   子模型键: {list(manager.sub_models.keys())}")
        
        # 检查关键方法
        print("\n2. 方法检查:")
        methods = ['coordinate_task', 'enhanced_coordinate_task', '_discover_and_register_sub_models']
        for method in methods:
            has = hasattr(manager, method)
            print(f"   {method}: {'存在' if has else '不存在'}")
        
        # 测试coordinate_task
        print("\n3. coordinate_task测试:")
        try:
            result = manager.coordinate_task(
                task_description="测试图片分析",
                required_models=["language", "vision"],
                priority=5
            )
            print(f"   调用成功")
            print(f"   状态: {result.get('status', 'unknown')}")
            print(f"   消息: {result.get('message', '无消息')}")
            if result.get('status') == 'success':
                print(f"   参与模型: {result.get('participating_models', 'N/A')}")
        except Exception as e:
            print(f"   调用失败: {type(e).__name__}: {e}")
        
        # 测试enhanced_coordinate_task
        print("\n4. enhanced_coordinate_task测试:")
        try:
            result = manager.enhanced_coordinate_task(
                task_description="测试增强协调",
                required_models=["language", "audio"],
                priority=3,
                collaboration_mode="smart"
            )
            print(f"   调用成功")
            print(f"   状态: {result.get('status', 'unknown')}")
            print(f"   协作模式: {result.get('collaboration_mode', 'N/A')}")
        except Exception as e:
            print(f"   调用失败: {type(e).__name__}: {e}")
        
        # 测试子模型发现
        print("\n5. 子模型发现测试:")
        try:
            if hasattr(manager, '_discover_and_register_sub_models'):
                available = manager._discover_and_register_sub_models()
                print(f"   发现方法执行成功")
                print(f"   返回可用模型: {available}")
                print(f"   当前子模型字典大小: {len(manager.sub_models)}")
            else:
                print("   发现方法不存在")
        except Exception as e:
            print(f"   发现方法执行失败: {type(e).__name__}: {e}")
        
        # 检查模拟模型功能
        print("\n6. 模拟模型功能检查:")
        core_models = ["language", "vision", "audio", "knowledge", "programming", "planning"]
        for model_id in core_models:
            if model_id in manager.sub_models and manager.sub_models[model_id] is not None:
                model = manager.sub_models[model_id]
                print(f"   {model_id}: 存在")
                # 测试初始化
                if hasattr(model, 'initialize'):
                    try:
                        init_result = model.initialize()
                        print(f"     初始化: {init_result.get('success', 'unknown')}")
                    except:
                        print(f"     初始化: 失败")
            else:
                print(f"   {model_id}: 不存在")
        
        print("\n7. 综合协调测试:")
        test_tasks = [
            ("分析一张风景图片", ["vision", "language"]),
            ("将英文文本翻译成中文", ["language", "translation"]),
            ("识别音频中的情感", ["audio", "emotion"])
        ]
        
        for desc, models in test_tasks:
            try:
                result = manager.coordinate_task(
                    task_description=desc,
                    required_models=models,
                    priority=5
                )
                status = result.get('status', 'unknown')
                print(f"   '{desc[:20]}...': {status}")
            except Exception as e:
                print(f"   '{desc[:20]}...': 异常 - {type(e).__name__}")
        
        print("\n=== 验证完成 ===")
        return True
        
    except Exception as e:
        print(f"验证失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_fixes()
    sys.exit(0 if success else 1)