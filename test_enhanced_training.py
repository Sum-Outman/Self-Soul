# -*- coding: utf-8 -*-
"""
测试增强的元学习训练与优化机制
"""

import os
import sys
import numpy as np
import random

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# 导入必要的模块
try:
    from core.meta_learning_system import EnhancedMetaLearningSystem, LearningEpisode
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

def test_enhanced_meta_learning():
    """测试增强的元学习训练与优化机制"""
    print("=== 测试增强的元学习训练与优化机制 ===")
    
    # 初始化元学习系统
    mls = EnhancedMetaLearningSystem(from_scratch=True, device="cpu")
    print("元学习系统初始化完成")
    
    # 生成一些测试学习 episode
    print("\n=== 生成测试学习 episode ===")
    for i in range(10):
        task_type = random.choice(["classification", "regression", "reinforcement", "planning"])
        strategy = random.choice(["maml", "reptile", "supervised", "transfer"])
        
        episode = LearningEpisode(
            task_type=task_type,
            task_id=f"test_task_{i}",
            strategy_used=strategy,
            success_metric=0.6 + random.random() * 0.35,  # 0.6-0.95
            learning_time=30.0 + random.random() * 60.0,
            resources_used={"cpu": 0.5 + random.random() * 0.4, "memory": 0.7 + random.random() * 0.2},
            insights_gained=[f"测试洞察 {i}"],
            timestamp=1234567890 + i,
            model_params_snapshot={"layer1": [0.1, 0.2, 0.3], "layer2": [0.4, 0.5, 0.6]},
            gradient_updates=[{"param": "layer1", "gradient": [-0.01, 0.02, -0.005]}],
            meta_learning_used=True
        )
        
        mls.record_learning_episode(episode)
        print(f"生成 episode {i+1}/{10}: {task_type}, {strategy}, 成功率: {episode.success_metric:.3f}")
    
    # 测试元训练
    print("\n=== 测试元训练 ===")
    try:
        avg_loss = mls.meta_train(num_iterations=20, batch_size=3)
        if avg_loss is not None:
            print(f"元训练完成，平均损失: {avg_loss:.6f}")
        else:
            print("元训练因数据不足而跳过")
    except Exception as e:
        print(f"元训练错误: {e}")
    
    # 测试快速适应
    print("\n=== 测试快速适应 ===")
    try:
        # 创建测试支持数据
        x_support = np.random.randn(5, 32).astype(np.float32)
        y_support = np.random.randn(5, 1).astype(np.float32)
        support_data = (x_support, y_support)
        
        task_data = {
            "task_type": "classification",
            "dataset_size": 100,
            "num_classes": 10,
            "similar_tasks": ["object recognition", "pattern classification"]
        }
        
        adaptation_result = mls.fast_adapt(task_data, support_data, num_steps=5)
        
        print("适应结果:")
        print(f"  适应步骤: {adaptation_result['adaptation_steps']}")
        print(f"  初始损失: {adaptation_result['initial_loss']:.6f}")
        print(f"  最终损失: {adaptation_result['final_loss']:.6f}")
        print(f"  适应时间: {adaptation_result['adaptation_time']:.3f}秒")
        print(f"  改进率: {adaptation_result['improvement_rate']:.3f}")
        print(f"  任务复杂度: {adaptation_result['task_complexity']:.3f}")
        print(f"  使用的优化器: {adaptation_result['optimizer_used']}")
        print(f"  收敛状态: {'是' if adaptation_result['converged'] else '否'}")
    except Exception as e:
        print(f"快速适应错误: {e}")
    
    # 生成元学习洞察
    print("\n=== 生成元学习洞察 ===")
    insights = mls.generate_meta_insights()
    for insight in insights:
        print(f"\n{insight['type']}:")
        if 'recommendation' in insight:
            print(f"  建议: {insight['recommendation']}")
        if 'best_strategy' in insight:
            print(f"  最佳策略: {insight['best_strategy']} (得分: {insight['best_score']:.3f})")
            print(f"  最差策略: {insight['worst_strategy']} (得分: {insight['worst_score']:.3f})")
        if 'trend' in insight:
            print(f"  趋势: {insight['trend']}")
            if 'trend_strength' in insight:
                print(f"  趋势强度: {insight['trend_strength']:.3f}")
    
    # 显示系统统计信息
    print("\n=== 系统统计信息 ===")
    stats = mls.get_system_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        else:
            print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_enhanced_meta_learning()
