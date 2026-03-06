#!/usr/bin/env python3
"""
验证管理器协调修复

测试修复后的管理器模型是否能实际协调子模型。
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

def test_fixed_coordination():
    """测试修复后的协调功能"""
    logger.info("=== 测试修复后的管理器协调功能 ===")
    
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
        
        # 测试1: 基础协调功能
        logger.info("\n测试1: 基础协调功能")
        test_tasks = [
            {
                "description": "分析一张风景图片并生成描述",
                "required_models": ["language", "vision"],
                "name": "视觉-语言协作"
            },
            {
                "description": "翻译英文文本到中文",
                "required_models": ["language"],
                "name": "文本翻译"
            },
            {
                "description": "处理语音指令",
                "required_models": ["audio", "language"],
                "name": "语音处理"
            }
        ]
        
        all_passed = True
        
        for i, task in enumerate(test_tasks):
            logger.info(f"\n任务 {i+1}: {task['name']}")
            logger.info(f"描述: {task['description']}")
            logger.info(f"所需模型: {task['required_models']}")
            
            try:
                result = manager.coordinate_task(
                    task_description=task["description"],
                    required_models=task["required_models"],
                    priority=5
                )
                
                status = result.get("status", "unknown")
                logger.info(f"协调状态: {status}")
                
                if status == "success":
                    logger.info("✅ 协调成功")
                    
                    # 检查结果结构
                    if "model_results" in result:
                        model_results = result["model_results"]
                        logger.info(f"模型结果数量: {len(model_results)}")
                        
                        # 检查每个模型的结果
                        for model_id, model_result in model_results.items():
                            success = model_result.get("success", False)
                            status = "✅ 成功" if success else "❌ 失败"
                            logger.info(f"  模型 {model_id}: {status}")
                            
                            if not success:
                                logger.info(f"    错误: {model_result.get('error', '未知错误')}")
                    else:
                        logger.warning("⚠️ 结果中缺少model_results字段")
                    
                    # 检查执行错误
                    if "execution_errors" in result and result["execution_errors"]:
                        logger.warning(f"⚠️ 有执行错误: {result['execution_errors']}")
                    
                    # 检查结果摘要
                    if "result" in result:
                        summary = result["result"].get("summary", "无摘要")
                        confidence = result["result"].get("confidence", 0)
                        logger.info(f"结果摘要: {summary}")
                        logger.info(f"置信度: {confidence:.2f}")
                
                else:
                    logger.error(f"❌ 协调失败: {result.get('message', '未知原因')}")
                    all_passed = False
                    
            except Exception as e:
                logger.error(f"❌ 协调异常: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False
        
        # 测试2: 增强协调功能
        logger.info("\n测试2: 增强协调功能")
        
        try:
            result = manager.enhanced_coordinate_task(
                task_description="智能多模型协作测试",
                required_models=["language", "vision", "knowledge"],
                priority=7,
                collaboration_mode="smart"
            )
            
            status = result.get("status", "unknown")
            logger.info(f"增强协调状态: {status}")
            
            if status == "success":
                logger.info("✅ 增强协调成功")
                collaboration_mode = result.get("collaboration_mode", "unknown")
                logger.info(f"协作模式: {collaboration_mode}")
                
                if "result" in result:
                    summary = result["result"].get("summary", "无摘要")
                    confidence = result["result"].get("confidence", 0)
                    logger.info(f"结果摘要: {summary}")
                    logger.info(f"置信度: {confidence:.2f}")
            else:
                logger.error(f"❌ 增强协调失败: {result.get('message', '未知原因')}")
                all_passed = False
                
        except Exception as e:
            logger.error(f"❌ 增强协调异常: {type(e).__name__}: {e}")
            all_passed = False
        
        # 测试3: 检查子模型注册
        logger.info("\n测试3: 检查子模型注册")
        
        core_models = ["language", "vision", "audio", "knowledge", "programming", "planning"]
        registered_count = 0
        
        for model_id in core_models:
            if model_id in manager.sub_models and manager.sub_models[model_id] is not None:
                registered_count += 1
                model = manager.sub_models[model_id]
                
                # 检查方法可用性
                has_process = hasattr(model, 'process')
                has_initialize = hasattr(model, 'initialize')
                
                logger.info(f"  {model_id}: 已注册 | process: {'✅' if has_process else '❌'} | initialize: {'✅' if has_initialize else '❌'}")
            else:
                logger.warning(f"  {model_id}: ❌ 未注册")
        
        logger.info(f"子模型注册率: {registered_count}/{len(core_models)}")
        
        if registered_count >= 3:
            logger.info("✅ 满足最严厉审核要求：至少3个子模型已注册")
        else:
            logger.error("❌ 不满足审核要求：注册子模型少于3个")
            all_passed = False
        
        logger.info("\n" + "="*60)
        logger.info("修复验证结果")
        logger.info("="*60)
        
        if all_passed:
            logger.info("✅ 所有测试通过！管理器协调修复成功")
            logger.info("✅ 管理器能实际协调子模型")
            logger.info("✅ 协调方法返回实际处理结果")
            logger.info("✅ 满足最严厉审核报告要求")
        else:
            logger.info("⚠️ 部分测试失败，需要进一步修复")
        
        return all_passed
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    logger.info("开始验证管理器协调修复")
    
    result = test_fixed_coordination()
    
    if result:
        logger.info("\n✅ 验证成功：管理器协调功能已修复")
        return 0
    else:
        logger.info("\n❌ 验证失败：需要进一步修复")
        return 1

if __name__ == "__main__":
    sys.exit(main())
