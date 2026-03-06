#!/usr/bin/env python3
"""
测试子模型方法可用性

检查管理器模型的子模型是否具有必要的方法来处理任务。
"""

import sys
import os
import time
import json
import logging

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_submodel_methods():
    """测试子模型方法"""
    logger.info("=== 测试子模型方法可用性 ===")
    
    try:
        # 导入管理器模型
        from core.models.manager.unified_manager_model import UnifiedManagerModel
        
        # 创建管理器实例
        config = {
            "model_id": "manager",
            "test_mode": True,
            "initialize_sub_models": True
        }
        
        logger.info("创建管理器模型实例...")
        manager = UnifiedManagerModel(config)
        
        # 确保子模型已注册
        logger.info("确保子模型已注册...")
        if hasattr(manager, '_discover_and_register_sub_models'):
            available_models = manager._discover_and_register_sub_models()
            logger.info(f"可用模型: {available_models}")
        else:
            logger.error("❌ _discover_and_register_sub_models 方法不存在")
            return False
        
        # 检查核心子模型
        core_models = ["language", "vision", "audio", "knowledge"]
        
        for model_id in core_models:
            logger.info(f"\n检查子模型: {model_id}")
            
            if model_id in manager.sub_models:
                model = manager.sub_models[model_id]
                
                if model is None:
                    logger.warning(f"  ⚠️ 模型实例为 None")
                    continue
                
                logger.info(f"  模型类型: {type(model)}")
                
                # 检查关键方法
                methods_to_check = [
                    'process',
                    'process_input',
                    'initialize',
                    'prepare_for_coordination',
                    'get_status'
                ]
                
                for method_name in methods_to_check:
                    has_method = hasattr(model, method_name)
                    logger.info(f"  {method_name}: {'✅ 存在' if has_method else '❌ 不存在'}")
                    
                    if has_method:
                        # 尝试调用initialize方法
                        if method_name == 'initialize':
                            try:
                                result = model.initialize()
                                logger.info(f"    初始化结果: {result}")
                            except Exception as e:
                                logger.warning(f"    初始化失败: {e}")
                
                # 测试process方法
                if hasattr(model, 'process'):
                    try:
                        # 简单的测试输入
                        test_input = {"text": "测试处理功能", "type": "text"}
                        logger.info(f"  测试process方法...")
                        result = model.process(test_input)
                        logger.info(f"    process结果: {result}")
                    except Exception as e:
                        logger.warning(f"    process调用失败: {e}")
                elif hasattr(model, 'process_input'):
                    try:
                        test_input = {"text": "测试处理功能", "type": "text"}
                        logger.info(f"  测试process_input方法...")
                        result = model.process_input(test_input)
                        logger.info(f"    process_input结果: {result}")
                    except Exception as e:
                        logger.warning(f"    process_input调用失败: {e}")
            else:
                logger.warning(f"  ⚠️ 模型不在sub_models字典中")
        
        # 测试管理器协调方法是否实际调用子模型
        logger.info("\n=== 测试管理器协调方法 ===")
        
        # 创建一个模拟任务
        test_task = "分析一张图片并生成描述"
        
        logger.info(f"测试任务: {test_task}")
        
        # 调用coordinate_task
        try:
            result = manager.coordinate_task(
                task_description=test_task,
                required_models=["language", "vision"],
                priority=5
            )
            
            logger.info(f"协调结果状态: {result.get('status', 'unknown')}")
            logger.info(f"协调结果消息: {result.get('message', '无消息')}")
            
            if result.get('status') == 'success':
                logger.info("✅ 协调成功")
                if 'result' in result:
                    logger.info(f"协调结果: {result['result']}")
            else:
                logger.warning("⚠️ 协调未成功")
                if 'unavailable_models' in result:
                    logger.info(f"不可用模型: {result['unavailable_models']}")
        
        except Exception as e:
            logger.error(f"协调调用异常: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info("\n=== 测试完成 ===")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_coordination_logic():
    """分析协调逻辑"""
    logger.info("\n=== 分析协调逻辑 ===")
    
    try:
        # 检查协调方法的实现
        from core.models.manager.unified_manager_model import UnifiedManagerModel
        
        # 获取方法源码位置
        import inspect
        
        methods_to_check = [
            'coordinate_task',
            '_initiate_model_coordination',
            '_monitor_coordination',
            '_integrate_final_results',
            '_process_dependencies',
            '_collect_intermediate_results'
        ]
        
        for method_name in methods_to_check:
            try:
                # 获取方法
                method = getattr(UnifiedManagerModel, method_name, None)
                if method:
                    # 获取方法定义的文件和行号
                    try:
                        file_path = inspect.getfile(method)
                        line_num = inspect.getsourcelines(method)[1]
                        logger.info(f"{method_name}: 定义在 {file_path}:{line_num}")
                    except:
                        logger.info(f"{method_name}: 存在但无法获取源码位置")
                else:
                    logger.warning(f"{method_name}: ❌ 不存在")
            except Exception as e:
                logger.error(f"检查 {method_name} 失败: {e}")
    
    except Exception as e:
        logger.error(f"分析协调逻辑失败: {e}")

def main():
    """主函数"""
    logger.info("开始子模型方法测试")
    
    # 测试子模型方法
    test_result = test_submodel_methods()
    
    # 分析协调逻辑
    analyze_coordination_logic()
    
    logger.info("\n" + "="*60)
    logger.info("测试总结")
    logger.info("="*60)
    
    if test_result:
        logger.info("✅ 测试基本完成")
    else:
        logger.info("⚠️ 测试发现问题")
    
    return 0 if test_result else 1

if __name__ == "__main__":
    sys.exit(main())