# -*- coding: utf-8 -*-
"""
测试模型协作与集成框架
"""

import os
import sys
import asyncio
import time
import logging

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# 导入必要的模块
try:
    from typing import Dict, Any
    from core.collaboration.model_collaborator import (
        collaboration_orchestrator,
        CollaborationMode,
        start_collaboration_orchestrator,
        stop_collaboration_orchestrator
    )
    from core.model_registry import get_model_registry
    from core.meta_learning_system import EnhancedMetaLearningSystem
    from core.models.base_model import BaseModel
    import numpy as np
    import random

except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建测试模型类
class SimpleTestModel(BaseModel):
    """简单的测试模型"""
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        model_id = config.get('model_id', 'simple_test_model')
        config['model_id'] = model_id
        super().__init__(config)
        self.task_type = config.get('task_type', 'classification')
        self.model_id = model_id
    
    def initialize(self) -> Dict[str, Any]:
        """初始化模型资源"""
        self.is_initialized = True
        return {"success": True, "model_id": self.model_id, "initialized": True}
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入数据"""
        # 从input_data中提取参数
        parameters = input_data.get('parameters', {})
        actual_input = input_data.get('input_data', input_data)
        
        return self.execute_task(actual_input, parameters)
    
    def execute_task(self, input_data, parameters=None):
        """执行测试任务"""
        parameters = parameters or {}
        processing_time = parameters.get("processing_time", 0.1)
        time.sleep(processing_time)  # 模拟处理时间
        
        if isinstance(input_data, dict) and "value" in input_data:
            result_value = input_data["value"] * 2
        else:
            result_value = 10  # 默认值
        
        return {
            "model_id": self.model_id,
            "result": result_value,
            "input_data": input_data,
            "parameters": parameters
        }

class ClassificationTestModel(BaseModel):
    """分类测试模型"""
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        model_id = config.get('model_id', 'classification_test_model')
        config['model_id'] = model_id
        config['task_type'] = 'classification'
        super().__init__(config)
        self.model_id = model_id
    
    def initialize(self) -> Dict[str, Any]:
        """初始化模型资源"""
        self.is_initialized = True
        return {"success": True, "model_id": self.model_id, "initialized": True}
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入数据"""
        # 从input_data中提取参数
        parameters = input_data.get('parameters', {})
        actual_input = input_data.get('input_data', input_data)
        
        return self.execute_task(actual_input, parameters)
    
    def execute_task(self, input_data, parameters=None):
        """执行分类任务"""
        parameters = parameters or {}
        processing_time = parameters.get("processing_time", 0.05)
        time.sleep(processing_time)
        
        # 模拟分类结果
        if isinstance(input_data, dict) and "features" in input_data:
            features = input_data["features"]
            prediction = sum(features) % 2  # 简单的分类逻辑
        else:
            prediction = random.randint(0, 1)
        
        return {
            "model_id": self.model_id,
            "prediction": prediction,
            "confidence": 0.7 + random.random() * 0.3,
            "input_data": input_data,
            "parameters": parameters
        }

class ConditionTestModel(BaseModel):
    """条件判断测试模型"""
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        model_id = config.get('model_id', 'condition_test_model')
        config['model_id'] = model_id
        config['task_type'] = 'condition'
        super().__init__(config)
        self.model_id = model_id
    
    def initialize(self) -> Dict[str, Any]:
        """初始化模型资源"""
        self.is_initialized = True
        return {"success": True, "model_id": self.model_id, "initialized": True}
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入数据"""
        # 从input_data中提取参数
        parameters = input_data.get('parameters', {})
        actual_input = input_data.get('input_data', input_data)
        
        return self.execute_task(actual_input, parameters)
    
    def execute_task(self, input_data, parameters=None):
        """执行条件判断任务"""
        parameters = parameters or {}
        time.sleep(0.05)
        
        # 根据输入数据的大小选择分支
        if isinstance(input_data, dict) and "value" in input_data:
            value = input_data["value"]
            if value > 5:
                return {"decision": "large_value", "original_value": value}
            else:
                return {"decision": "small_value", "original_value": value}
        else:
            return {"decision": "default", "original_value": input_data}

async def test_model_collaboration():
    """测试模型协作与集成框架"""
    print("=== 测试模型协作与集成框架 ===")
    
    # 启动协作协调器
    await start_collaboration_orchestrator()
    print("协作协调器启动完成")
    
    # 获取模型注册表
    registry = get_model_registry()
    
    # 注册测试模型
    test_models = [
        SimpleTestModel({"model_id": "test_model_1"}),
        SimpleTestModel({"model_id": "test_model_2"}),
        SimpleTestModel({"model_id": "test_model_3"}),
        ClassificationTestModel({"model_id": "class_model_1"}),
        ClassificationTestModel({"model_id": "class_model_2"}),
        ClassificationTestModel({"model_id": "class_model_3"}),
        ConditionTestModel({"model_id": "condition_model"})
    ]
    
    for model in test_models:
        registry.register_model(model.model_id, type(model), {})
        print(f"注册模型: {model.model_id}")
    
    # 测试1: 注册自定义协作模式
    print("\n=== 测试1: 注册自定义协作模式 ===")
    custom_pattern = {
        "models": ["test_model_1", "test_model_2", "test_model_3"],
        "workflow": [
            {"model": "test_model_1", "task": "process_data", "share_result": True},
            {"model": "test_model_2", "task": "process_data", "share_result": True},
            {"model": "test_model_3", "task": "process_data", "depends_on": ["test_model_1", "test_model_2"]}
        ],
        "mode": CollaborationMode.SERIAL,
        "description": "测试三个模型串行协作"
    }
    
    success = collaboration_orchestrator.register_collaboration_pattern("custom_serial", custom_pattern)
    print(f"自定义协作模式注册: {'成功' if success else '失败'}")
    
    # 测试2: 执行串行协作
    print("\n=== 测试2: 执行串行协作 ===")
    serial_result = await collaboration_orchestrator.initiate_collaboration(
        "custom_serial", 
        {"value": 5},
        {"processing_time": 0.1}
    )
    print(f"串行协作结果: {serial_result}")
    
    # 测试3: 执行并行协作
    print("\n=== 测试3: 执行并行协作 ===")
    parallel_pattern = {
        "models": ["class_model_1", "class_model_2", "class_model_3"],
        "workflow": [
            {"model": "class_model_1", "task": "classify", "share_result": True},
            {"model": "class_model_2", "task": "classify", "share_result": True},
            {"model": "class_model_3", "task": "classify", "share_result": True}
        ],
        "mode": CollaborationMode.PARALLEL,
        "description": "测试三个分类模型并行协作"
    }
    
    success = collaboration_orchestrator.register_collaboration_pattern("custom_parallel", parallel_pattern)
    if success:
        parallel_result = await collaboration_orchestrator.initiate_collaboration(
            "custom_parallel",
            {"features": [1, 2, 3, 4, 5]},
            {"merge_strategy": "vote", "processing_time": 0.1}
        )
        print(f"并行协作结果: {parallel_result}")
    
    # 测试4: 执行混合协作
    print("\n=== 测试4: 执行混合协作 ===")
    hybrid_pattern = {
        "models": ["class_model_1", "class_model_2", "test_model_1"],
        "workflow": [
            {"model": "class_model_1", "task": "classify", "share_result": True},
            {"model": "class_model_2", "task": "classify", "share_result": True},
            {"model": "test_model_1", "task": "process_data", "depends_on": ["class_model_1", "class_model_2"]}
        ],
        "mode": CollaborationMode.HYBRID,
        "description": "测试混合协作模式"
    }
    
    success = collaboration_orchestrator.register_collaboration_pattern("custom_hybrid", hybrid_pattern)
    if success:
        hybrid_result = await collaboration_orchestrator.initiate_collaboration(
            "custom_hybrid",
            {"features": [1, 2, 3, 4, 5]},
            {
                "parallel_models": ["class_model_1", "class_model_2"],
                "serial_models": ["test_model_1"],
                "processing_time": 0.1
            }
        )
        print(f"混合协作结果: {hybrid_result}")
    
    # 测试5: 执行条件协作
    print("\n=== 测试5: 执行条件协作 ===")
    condition_pattern = {
        "models": ["condition_model", "test_model_1", "test_model_2"],
        "workflow": [
            {"model": "condition_model", "task": "evaluate", "share_result": True},
            {"model": "test_model_1", "task": "process_data", "depends_on": "condition_model", "condition": "small_value"},
            {"model": "test_model_2", "task": "process_data", "depends_on": "condition_model", "condition": "large_value"}
        ],
        "mode": CollaborationMode.CONDITIONAL,
        "description": "测试条件协作模式"
    }
    
    success = collaboration_orchestrator.register_collaboration_pattern("custom_conditional", condition_pattern)
    if success:
        # 测试小值情况
        condition_result_small = await collaboration_orchestrator.initiate_collaboration(
            "custom_conditional",
            {"value": 3},
            {
                "condition_model": "condition_model",
                "branches": {
                    "small_value": {"models": ["test_model_1"], "mode": "serial"},
                    "large_value": {"models": ["test_model_2"], "mode": "serial"}
                },
                "condition_mapping": {
                    "small_value": "small_value",
                    "large_value": "large_value"
                },
                "default_branch": "small_value",
                "processing_time": 0.1
            }
        )
        print(f"条件协作结果(小值): {condition_result_small}")
        
        # 测试大值情况
        condition_result_large = await collaboration_orchestrator.initiate_collaboration(
            "custom_conditional",
            {"value": 7},
            {
                "condition_model": "condition_model",
                "branches": {
                    "small_value": {"models": ["test_model_1"], "mode": "serial"},
                    "large_value": {"models": ["test_model_2"], "mode": "serial"}
                },
                "condition_mapping": {
                    "small_value": "small_value",
                    "large_value": "large_value"
                },
                "default_branch": "small_value",
                "processing_time": 0.1
            }
        )
        print(f"条件协作结果(大值): {condition_result_large}")
    
    # 测试6: 获取协作状态和性能报告
    print("\n=== 测试6: 获取协作状态和性能报告 ===")
    status = collaboration_orchestrator.get_collaboration_status()
    print(f"协作状态: {status}")
    
    performance_report = collaboration_orchestrator.get_performance_report()
    print(f"性能报告: {performance_report}")
    
    # 测试7: 与元学习系统集成
    print("\n=== 测试7: 与元学习系统集成 ===")
    try:
        # 初始化元学习系统
        mls = EnhancedMetaLearningSystem(from_scratch=True, device="cpu")
        print("元学习系统初始化完成")
        
        # 生成测试学习 episode
        for i in range(3):
            task_type = random.choice(["classification", "regression", "reinforcement", "planning"])
            strategy = random.choice(["maml", "reptile", "supervised", "transfer"])
            
            # 执行协作任务
            serial_result = await collaboration_orchestrator.initiate_collaboration(
                "custom_serial",
                {"value": random.randint(1, 10)},
                {"processing_time": 0.05}
            )
            
            # 记录学习 episode
            if "error" not in serial_result:
                from core.meta_learning_system import LearningEpisode
                episode = LearningEpisode(
                    task_type=task_type,
                    task_id=f"collab_task_{i}",
                    strategy_used=strategy,
                    success_metric=0.7 + random.random() * 0.3,
                    learning_time=serial_result.get("execution_time", 0.3),
                    resources_used={"cpu": 0.5, "memory": 0.3},
                    insights_gained=[f"协作测试洞察 {i}"],
                    timestamp=time.time(),
                    model_params_snapshot={"param1": [0.1, 0.2], "param2": [0.3, 0.4]},
                    gradient_updates=[{"param": "param1", "gradient": [-0.01, 0.02]}],
                    meta_learning_used=True
                )
                
                mls.record_learning_episode(episode)
                print(f"记录协作学习 episode {i+1}")
        
        # 测试元训练
        print("\n=== 测试元训练 ===")
        avg_loss = mls.meta_train(num_iterations=10, batch_size=2)
        if avg_loss is not None:
            print(f"元训练完成，平均损失: {avg_loss:.6f}")
        else:
            print("元训练因数据不足而跳过")
            
    except Exception as e:
        print(f"元学习集成测试错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试8: 清理共享内存
    print("\n=== 测试8: 清理共享内存 ===")
    success = collaboration_orchestrator.clear_shared_memory()
    print(f"共享内存清理: {'成功' if success else '失败'}")
    
    # 停止协作协调器
    await stop_collaboration_orchestrator()
    print("协作协调器停止完成")
    
    print("\n=== 模型协作与集成框架测试完成 ===")

if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_model_collaboration())
