#!/usr/bin/env python3
"""
执行训练计划脚本 - Execute Training Plan Script

根据training_plan.md中的训练计划，按照指定的训练顺序和并行策略启动模型训练。
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.model_registry import ModelRegistry, get_model_registry
from core.training_manager import TrainingManager
from core.training_preparation import TrainingPreparation, create_training_preparation

# 设置日志配置
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainingPlanExecutor")

class TrainingPlanExecutor:
    """训练计划执行器"""
    
    def __init__(self):
        """初始化训练计划执行器"""
        self.model_registry = get_model_registry()
        self.training_manager = TrainingManager(self.model_registry)
        self.training_preparation = create_training_preparation()
        
    def execute_training_plan(self):
        """执行完整的训练计划"""
        logger.info("开始执行训练计划...")
        
        # 训练阶段定义 - 只执行语言模型训练
        training_phases = [
            {
                "name": "第一阶段：基础感知与认知模型",
                "parallel_groups": [
                    {"models": ["language"], "description": "语言模型单独训练"}
                ]
            }
        ]
        
        # 训练参数设置
        training_parameters = {
            "from_scratch": True,
            "training_mode": "individual",  # 使用独立训练模式
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "validation_split": 0.2
        }
        
        # 执行每个训练阶段
        for phase in training_phases:
            logger.info(f"\n=== {phase['name']} ===")
            
            # 并行执行每个训练组
            for group in phase['parallel_groups']:
                model_ids = group['models']
                description = group['description']
                
                logger.info(f"开始训练: {description} ({', '.join(model_ids)})")
                
                try:
                    # 检查模型是否已加载
                    for model_id in model_ids:
                        if not self.model_registry.get_model(model_id):
                            logger.info(f"加载模型: {model_id}")
                            self.model_registry.load_model(model_id)
                    
                    # 启动训练
                    job_id = self.training_manager.start_training(model_ids, training_parameters)
                    logger.info(f"训练任务已启动，任务ID: {job_id}")
                    
                    # 监控训练进度，而不是模拟等待
                    logger.info(f"开始监控训练进度，任务ID: {job_id}")
                    max_wait_time = 3600  # 最长等待1小时（60分钟 * 60秒）
                    check_interval = 5    # 每5秒检查一次
                    start_time = time.time()
                    
                    while True:
                        # 检查是否超时
                        elapsed_time = time.time() - start_time
                        if elapsed_time > max_wait_time:
                            logger.warning(f"训练任务 {job_id} 超时（超过 {max_wait_time} 秒）")
                            break
                        
                        # 获取训练状态
                        status = self.training_manager.get_training_status(job_id)
                        
                        if not status.get('success', False):
                            logger.warning(f"无法获取训练任务 {job_id} 的状态: {status.get('message', '未知错误')}")
                            time.sleep(check_interval)
                            continue
                        
                        job_status = status.get('status', 'unknown')
                        progress = status.get('progress', 0)
                        
                        # 记录训练进度
                        if elapsed_time % 30 < check_interval:  # 每30秒记录一次进度
                            logger.info(f"训练任务 {job_id} 状态: {job_status}, 进度: {progress:.2%}")
                        
                        # 检查训练是否完成
                        if job_status in ['completed', 'failed', 'cancelled']:
                            if job_status == 'completed':
                                logger.info(f"训练任务 {job_id} 已完成，进度: {progress:.2%}")
                            elif job_status == 'failed':
                                logger.error(f"训练任务 {job_id} 失败")
                            elif job_status == 'cancelled':
                                logger.warning(f"训练任务 {job_id} 被取消")
                            break
                        
                        # 等待下一次检查
                        time.sleep(check_interval)
                    
                    logger.info(f"训练完成: {description} ({', '.join(model_ids)})")
                    
                except Exception as e:
                    logger.error(f"训练失败: {description} ({', '.join(model_ids)}) - 错误: {str(e)}")
                    
        logger.info("\n=== 训练计划执行完成 ===")

def main():
    """主函数"""
    try:
        # 初始化训练准备
        logger.info("初始化训练环境...")
        training_preparation = create_training_preparation()
        
        # 准备训练环境
        env_result = training_preparation.prepare_training_environment()
        if not env_result['success']:
            logger.error(f"环境准备失败: {env_result['message']}")
            return
        
        logger.info("训练环境准备完成！")
        
        # 执行训练计划
        executor = TrainingPlanExecutor()
        executor.execute_training_plan()
        
    except Exception as e:
        logger.error(f"执行训练计划时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
