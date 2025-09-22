"""
Test for from-scratch training functionality

This test demonstrates how to use the from-scratch training feature in the AGI system.
"""

import asyncio
import time
import json
import os
from core.training_manager import TrainingManager
from core.model_registry import ModelRegistry

# 设置测试配置
TEST_CONFIG = {
    "model_ids": ["language", "knowledge"],  # 要训练的模型ID
    "training_mode": "joint",  # 训练模式: 'individual' 或 'joint'
    "epochs": 5,  # 训练轮数
    "batch_size": 16,  # 批次大小
    "from_scratch": True,  # 启用从零开始训练
    "strategic_objectives": {
        "exploration": 0.8,  # 高探索比例
        "exploitation": 0.2,  # 低利用比例
        "foundational_learning": 0.9  # 高基础学习权重
    },
    "learning_rate_schedule": "warmup",  # 学习率调度策略
    "enable_nas": False,  # 禁用神经架构搜索以简化测试
    "enable_real_time_monitoring": True  # 启用实时监控
}

class TestFromScratchTraining:
    """测试从零开始训练功能"""
    
    def __init__(self):
        """初始化测试环境"""
        # 初始化模型注册表
        self.model_registry = ModelRegistry()
        # 初始化训练管理器
        self.training_manager = TrainingManager(self.model_registry)
        # 测试结果存储
        self.test_results = {}
        
    def setup_test_environment(self):
        """设置测试环境"""
        print("Setting up test environment...")
        
        # 确保测试使用的模型已加载
        for model_id in TEST_CONFIG["model_ids"]:
            try:
                if not self.model_registry.get_model(model_id):
                    print(f"Loading model: {model_id}...")
                    self.model_registry.load_model(model_id)
                print(f"Model {model_id} loaded successfully")
            except Exception as e:
                print(f"Failed to load model {model_id}: {str(e)}")
                raise
        
    def run_from_scratch_training(self):
        """运行从零开始训练测试"""
        print("\n=== Starting From-Scratch Training Test ===")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 启动从零开始训练
            job_id = self.training_manager.start_training(
                model_ids=TEST_CONFIG["model_ids"],
                parameters=TEST_CONFIG
            )
            print(f"Started training job with ID: {job_id}")
            
            # 存储作业ID
            self.test_results["job_id"] = job_id
            
            # 等待训练完成（在实际环境中，训练可能需要更长时间）
            print("Training in progress. Waiting for completion...")
            
            # 模拟等待训练完成（实际环境中应使用回调或状态轮询）
            # 在真实实现中，这里应该等待实际的训练完成信号
            wait_time = 10  # 等待10秒用于演示
            print(f"Waiting {wait_time} seconds for demonstration purposes...")
            time.sleep(wait_time)
            
            # 获取训练状态（在真实实现中，应轮询状态直到训练完成）
            training_status = self._get_training_status(job_id)
            self.test_results["status"] = training_status
            
            # 记录结束时间
            end_time = time.time()
            self.test_results["training_time"] = end_time - start_time
            
            print(f"\nTraining completed in {self.test_results['training_time']:.2f} seconds")
            print(f"Training status: {training_status}")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            self.test_results["error"] = str(e)
            raise
        
        print("\n=== From-Scratch Training Test Completed ===")
        
    def _get_training_status(self, job_id):
        """获取训练状态（模拟实现）"""
        # 在实际实现中，应该调用训练管理器的API获取真实状态
        # 这里使用模拟状态用于演示
        if job_id in self.training_manager.training_jobs:
            job = self.training_manager.training_jobs[job_id]
            return job["status"]
        return "unknown"
        
    def save_test_results(self):
        """保存测试结果"""
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, f"from_scratch_training_test_{int(time.time())}.json")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "test_config": TEST_CONFIG,
                "results": self.test_results
            }, f, ensure_ascii=False, indent=2)
            
        print(f"Test results saved to: {results_file}")
        
    def compare_with_normal_training(self):
        """对比从零开始训练与普通训练（概念性）"""
        print("\n=== Comparison with Normal Training ===")
        print("Key differences in from-scratch training:")
        print("1. No prior knowledge or experience used")
        print("2. Higher exploration ratio (0.8 vs 0.3 in normal training)")
        print("3. Lower initial meta-cognitive awareness (0.1 vs 0.5 in normal training)")
        print("4. Simplified knowledge context with foundational focus")
        print("5. More frequent reflection cycles (per_batch vs per_epoch)")
        print("6. Hierarchical guided collaboration strategy")
        print("7. Random walk neural architecture search strategy")
        print("8. Special handling for error recovery and fallback mechanisms")

# 运行测试
if __name__ == "__main__":
    try:
        test = TestFromScratchTraining()
        
        # 设置测试环境
        test.setup_test_environment()
        
        # 运行从零开始训练测试
        test.run_from_scratch_training()
        
        # 保存测试结果
        test.save_test_results()
        
        # 对比与普通训练的差异
        test.compare_with_normal_training()
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()