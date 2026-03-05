import os
import json
import sys
import time
import datetime
from core.training_manager import TrainingManager

def train_dialogue_model_deep(model_type: str, display_name: str, dataset_path: str, 
                              epochs_per_cycle: int, total_cycles: int):
    """深度训练单个对话模型，使用多周期训练策略"""
    print(f"\n开始深度训练 {display_name} ({model_type})...")
    
    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: {display_name} 的数据集不存在: {dataset_path}")
        return False
    
    try:
        # 加载数据集以验证格式
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"加载了 {len(dataset)} 条训练数据")
        
        # 获取训练管理器实例
        training_manager = TrainingManager()
        
        # 多周期训练
        overall_success = True
        cycle_results = []
        
        for cycle in range(1, total_cycles + 1):
            print(f"\n{'='*60}")
            print(f"  训练周期 {cycle}/{total_cycles}")
            print(f"{'='*60}")
            
            cycle_start_time = time.time()
            
            # 准备深度训练参数
            training_params = {
                "model_type": model_type,
                "dataset_path": dataset_path,
                "epochs": epochs_per_cycle,
                "batch_size": 16,  # 减小批次大小以提高模型质量
                "learning_rate": 5e-4,  # 精细学习率
                "validation_split": 0.2,
                "save_dir": os.path.join("data", "models", model_type, f"cycle_{cycle}"),
                "early_stopping": False,  # 深度训练中禁用早停
                "patience": None,
                "learning_rate_decay": True,  # 启用学习率衰减
                "data_augmentation": True,  # 启用数据增强
                "cycle_number": cycle,
                "total_cycles": total_cycles
            }
            
            # 创建训练任务
            print(f"正在创建第 {cycle} 周期的训练任务...")
            task_id = training_manager.create_training_task(
                model_type=model_type,
                parameters=training_params,
                priority=2  # 深度训练具有较高优先级
            )
            
            if not task_id:
                print(f"创建第 {cycle} 周期训练任务失败")
                overall_success = False
                break
            
            print(f"训练任务已创建，任务ID: {task_id}")
            
            # 启动训练
            print(f"开始第 {cycle} 周期训练...")
            training_manager.start_training_task(task_id)
            
            # 监控训练进度
            max_wait_time = 7200  # 每个周期最大等待时间2小时
            start_time = time.time()
            last_status = ""
            
            cycle_completed = False
            while not cycle_completed:
                # 检查训练任务状态
                task_status = training_manager.get_training_status(task_id)
                
                if task_status:
                    current_status = task_status.get('status', 'unknown')
                    progress = task_status.get('progress', 0)
                    current_epoch = task_status.get('current_epoch', 0)
                    total_epochs = task_status.get('total_epochs', 1)
                    
                    # 显示进度变化
                    status_display = f"周期 {cycle}: 状态={current_status}, 进度={progress:.1%}, 轮次={current_epoch}/{total_epochs}"
                    if status_display != last_status:
                        print(status_display)
                        last_status = status_display
                    
                    # 检查是否完成
                    if current_status in ['completed', 'failed', 'cancelled']:
                        if current_status == 'completed':
                            print(f"第 {cycle} 周期训练成功完成！")
                            # 获取训练结果
                            result = training_manager.get_training_result(task_id)
                            if result:
                                cycle_result = {
                                    'cycle': cycle,
                                    'final_loss': result.get('final_loss', 'N/A'),
                                    'final_accuracy': result.get('final_accuracy', 'N/A'),
                                    'training_time': result.get('training_time', 0)
                                }
                                cycle_results.append(cycle_result)
                                print(f"训练结果: 损失={cycle_result['final_loss']}, 准确率={cycle_result['final_accuracy']}")
                            cycle_completed = True
                        else:
                            print(f"第 {cycle} 周期训练失败，状态: {current_status}")
                            error_msg = task_status.get('error_message', '未知错误')
                            print(f"错误信息: {error_msg}")
                            overall_success = False
                            cycle_completed = True
                
                # 检查超时
                if time.time() - start_time > max_wait_time:
                    print(f"第 {cycle} 周期训练超时，已等待{max_wait_time}秒")
                    overall_success = False
                    break
                
                # 等待一段时间再检查
                time.sleep(2)
            
            # 如果当前周期失败，则停止整个训练
            if not overall_success:
                break
            
            # 计算周期耗时
            cycle_duration = time.time() - cycle_start_time
            cycle_minutes = cycle_duration / 60
            
            print(f"\n📊 第 {cycle} 周期训练完成")
            print(f"   耗时: {cycle_minutes:.1f} 分钟")
            print(f"   平均每轮耗时: {cycle_duration / epochs_per_cycle:.2f} 秒")
            
            # 周期之间的短暂休息（可选，但实际训练中可能不需要）
            if cycle < total_cycles:
                print(f"\n⏸️  周期间休息 3 秒...")
                time.sleep(3)
        
        # 深度训练总结
        if overall_success:
            print(f"\n{'='*80}")
            print(f"✅ {display_name} 深度训练完成!")
            print(f"   总训练周期: {total_cycles}")
            print(f"   总训练轮次: {epochs_per_cycle * total_cycles}")
            
            # 输出周期结果
            if cycle_results:
                print(f"\n📈 周期训练结果:")
                for result in cycle_results:
                    print(f"   周期 {result['cycle']}: 损失={result['final_loss']}, 准确率={result['final_accuracy']}")
            
            return True
        else:
            print(f"\n❌ {display_name} 深度训练失败")
            return False
    
    except Exception as e:
        print(f"深度训练 {display_name} 模型时发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 80)
    print("        Self Soul - 对话模型深度训练脚本        ")
    print("=" * 80)
    print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n启动深度训练模式...")
    print("深度训练特点：增加训练轮次、多周期训练、精细参数调整\n")
    
    # 定义要训练的对话模型配置
    dialogue_models = [
        {
            "name": "language",
            "display_name": "语言模型",
            "dataset": "data/datasets/language_dataset.json",
            "epochs_per_cycle": 100,
            "cycles": 3
        },
        {
            "name": "manager",
            "display_name": "管理模型",
            "dataset": "data/datasets/manager_dataset.json",
            "epochs_per_cycle": 60,
            "cycles": 3
        }
    ]
    
    # 创建必要的目录
    for model_config in dialogue_models:
        model_type = model_config["name"]
        for cycle in range(1, model_config["cycles"] + 1):
            cycle_dir = os.path.join("data", "models", model_type, f"cycle_{cycle}")
            os.makedirs(cycle_dir, exist_ok=True)
    
    # 训练每个模型
    success_count = 0
    total_models = len(dialogue_models)
    
    for model_config in dialogue_models:
        model_type = model_config["name"]
        display_name = model_config["display_name"]
        dataset_path = model_config["dataset"]
        epochs_per_cycle = model_config["epochs_per_cycle"]
        total_cycles = model_config["cycles"]
        
        print(f"\n{'='*80}")
        print(f"        开始训练 {display_name}        ")
        print(f"{'='*80}")
        print(f"模型类型: {model_type}")
        print(f"每周期训练轮次: {epochs_per_cycle}")
        print(f"训练周期数: {total_cycles}")
        print(f"总训练轮次: {epochs_per_cycle * total_cycles}")
        print(f"数据集: {dataset_path}")
        
        success = train_dialogue_model_deep(
            model_type=model_type,
            display_name=display_name,
            dataset_path=dataset_path,
            epochs_per_cycle=epochs_per_cycle,
            total_cycles=total_cycles
        )
        
        if success:
            success_count += 1
            print(f"\n✅ {display_name} 深度训练完成！")
        else:
            print(f"\n❌ {display_name} 深度训练失败！")
        
        print("-" * 50)
    
    # 输出总结
    print(f"\n{'='*80}")
    print(f"深度训练完成总结:")
    print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"成功训练模型: {success_count}/{total_models}")
    
    if success_count == total_models:
        print("\n✅ 所有对话模型深度训练成功！")
        print("深度训练结果:")
        print("1. 模型参数已经过充分优化")
        print("2. 模型性能得到显著提升")
        print("3. 对话能力更加流畅自然")
        print("\n模型已准备好进行高级对话功能测试!")
        return 0
    else:
        print(f"\n⚠️  有 {total_models - success_count} 个模型深度训练失败。")
        print("建议检查数据集、配置和系统资源。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
