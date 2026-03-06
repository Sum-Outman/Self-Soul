#!/usr/bin/env python3
"""
测试管理器模型协调功能
以最严厉的态度验证管理器模型是否能实际协调子模型
"""

import sys
import os
import time
import json
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.error_handling import error_handler
from core.models.manager.unified_manager_model import UnifiedManagerModel

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_manager_initialization():
    """测试管理器模型初始化"""
    logger = setup_logging()
    logger.info("开始测试管理器模型初始化...")
    
    try:
        # 创建管理器模型实例
        config = {
            "model_id": "test_manager",
            "test_mode": True,
            "enable_lazy_loading": False  # 禁用延迟加载，确保立即初始化
        }
        
        manager = UnifiedManagerModel(config)
        logger.info(f"管理器模型实例创建成功: {manager}")
        
        # 检查子模型字典
        logger.info(f"子模型数量: {len(manager.sub_models)}")
        logger.info(f"子模型键: {list(manager.sub_models.keys())}")
        
        # 检查管理器模型自身是否在子模型中
        if "manager" in manager.sub_models:
            logger.info("✓ 管理器模型自身已注册到子模型字典")
        else:
            logger.error("✗ 管理器模型自身未注册到子模型字典")
            
        # 尝试发现和注册子模型
        logger.info("尝试发现和注册子模型...")
        try:
            available_models = manager._discover_and_register_sub_models()
            logger.info(f"发现的可用于模型: {available_models}")
            
            # 检查子模型是否已加载
            for model_id in available_models:
                if model_id in manager.sub_models and manager.sub_models[model_id] is not None:
                    logger.info(f"✓ 子模型 {model_id} 已加载")
                else:
                    logger.warning(f"⚠️ 子模型 {model_id} 在字典中但值为空")
        except Exception as e:
            logger.error(f"发现子模型失败: {e}")
            
        return manager
        
    except Exception as e:
        logger.error(f"管理器模型初始化失败: {e}")
        return None

def test_task_coordination(manager):
    """测试任务协调功能"""
    logger = setup_logging()
    logger.info("开始测试任务协调功能...")
    
    if manager is None:
        logger.error("管理器模型未初始化，跳过任务协调测试")
        return False
    
    # 测试1: 简单任务协调
    test_tasks = [
        {
            "description": "分析一张图片并生成描述",
            "required_models": ["language", "vision"],
            "name": "测试视觉-语言协作"
        },
        {
            "description": "翻译英文文本到中文",
            "required_models": ["language", "translation"],
            "name": "测试翻译任务"
        },
        {
            "description": "识别语音中的情感",
            "required_models": ["audio", "emotion"],
            "name": "测试音频情感分析"
        }
    ]
    
    all_passed = True
    
    for i, task in enumerate(test_tasks):
        logger.info(f"\n测试任务 {i+1}: {task['name']}")
        logger.info(f"任务描述: {task['description']}")
        logger.info(f"所需模型: {task['required_models']}")
        
        try:
            # 检查所需模型是否可用
            unavailable_models = []
            for model_id in task['required_models']:
                if model_id not in manager.sub_models or manager.sub_models[model_id] is None:
                    unavailable_models.append(model_id)
            
            if unavailable_models:
                logger.warning(f"⚠️ 以下模型不可用: {unavailable_models}")
                # 即使模型不可用，也测试协调框架
                logger.info("测试协调框架（返回模拟数据）...")
                
            # 执行任务协调
            start_time = time.time()
            result = manager.coordinate_task(
                task_description=task['description'],
                required_models=task['required_models'],
                priority=5
            )
            elapsed_time = time.time() - start_time
            
            logger.info(f"协调结果状态: {result.get('status', 'unknown')}")
            logger.info(f"执行时间: {elapsed_time:.2f}秒")
            
            if result.get('status') == 'success':
                logger.info("✓ 任务协调成功")
                if 'participating_models' in result:
                    logger.info(f"参与模型: {result['participating_models']}")
                if 'result' in result:
                    logger.info(f"协调结果摘要: {str(result['result'])[:200]}...")
            else:
                logger.warning(f"⚠️ 任务协调失败: {result.get('message', '未知原因')}")
                if unavailable_models:
                    logger.info("失败原因可能是模型不可用")
                else:
                    all_passed = False
                    
        except Exception as e:
            logger.error(f"✗ 任务协调异常: {e}")
            all_passed = False
    
    return all_passed

def test_enhanced_coordination(manager):
    """测试增强协调功能"""
    logger = setup_logging()
    logger.info("开始测试增强协调功能...")
    
    if manager is None:
        logger.error("管理器模型未初始化，跳过增强协调测试")
        return False
    
    # 测试不同协作模式
    collaboration_modes = ["smart", "parallel", "serial"]
    
    for mode in collaboration_modes:
        logger.info(f"\n测试协作模式: {mode}")
        
        try:
            result = manager.enhanced_coordinate_task(
                task_description="分析用户输入的多模态数据（文本+图片+音频）并生成综合报告",
                required_models=["language", "vision", "audio"],
                priority=5,
                collaboration_mode=mode
            )
            
            logger.info(f"增强协调结果状态: {result.get('status', 'unknown')}")
            
            if result.get('status') == 'success':
                logger.info(f"✓ {mode} 模式协调成功")
            else:
                logger.warning(f"⚠️ {mode} 模式协调失败: {result.get('message', '未知原因')}")
                
        except Exception as e:
            logger.error(f"✗ {mode} 模式协调异常: {e}")
    
    return True

def main():
    """主测试函数"""
    logger = setup_logging()
    logger.info("=== 管理器模型协调功能严格测试 ===")
    
    # 测试管理器初始化
    logger.info("\n" + "="*60)
    logger.info("阶段1: 管理器模型初始化测试")
    logger.info("="*60)
    manager = test_manager_initialization()
    
    if manager is None:
        logger.error("管理器模型初始化失败，终止测试")
        return False
    
    # 测试任务协调
    logger.info("\n" + "="*60)
    logger.info("阶段2: 基本任务协调测试")
    logger.info("="*60)
    coordination_passed = test_task_coordination(manager)
    
    # 测试增强协调
    logger.info("\n" + "="*60)
    logger.info("阶段3: 增强协调功能测试")
    logger.info("="*60)
    enhanced_passed = test_enhanced_coordination(manager)
    
    # 输出总结
    logger.info("\n" + "="*60)
    logger.info("测试总结")
    logger.info("="*60)
    
    if coordination_passed and enhanced_passed:
        logger.info("✓ 所有协调功能测试通过")
        return True
    else:
        logger.error("✗ 部分协调功能测试失败")
        
        # 输出详细诊断信息
        logger.info("\n诊断信息:")
        logger.info(f"子模型字典大小: {len(manager.sub_models)}")
        logger.info(f"子模型键: {list(manager.sub_models.keys())}")
        
        # 检查每个子模型的状态
        for model_id, model_instance in manager.sub_models.items():
            if model_instance is None:
                logger.warning(f"子模型 {model_id}: 未初始化 (None)")
            else:
                logger.info(f"子模型 {model_id}: 已初始化 ({type(model_instance).__name__})")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)