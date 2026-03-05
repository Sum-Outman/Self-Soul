import os
import sys
import time
import json
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.training_manager import TrainingManager

def train_model_efficient(model_type: str, dataset_path: str, training_params: dict) -> bool:
    """高效训练单个模型，使用优化的参数"""
    try:
        print(f"开始训练 {model_type} 模型...")
        
        # 检查数据集是否存在
        if not os.path.exists(dataset_path):
            print(f"警告: {model_type} 模型的数据集不存在，跳过训练: {dataset_path}")
            return False
        
        # 获取训练管理器实例（单例模式）
        training_manager = TrainingManager()
        
        # 创建训练任务
        task_id = training_manager.create_training_task(
            model_type=model_type,
            parameters=training_params,
            priority=2  # 高效训练具有较高优先级
        )
        
        if not task_id:
            print(f"创建训练任务失败")
            return False
        
        print(f"训练任务已创建，任务ID: {task_id}")
        
        # 启动训练（异步）
        training_manager.start_training_task(task_id)
        
        # 快速检查训练是否开始
        time.sleep(2)
        
        # 获取初始状态
        task_status = training_manager.get_training_status(task_id)
        if not task_status:
            print(f"无法获取训练任务状态")
            return False
        
        current_status = task_status.get('status', 'unknown')
        if current_status == 'failed':
            error_msg = task_status.get('error_message', '未知错误')
            print(f"训练任务启动失败: {error_msg}")
            return False
        
        print(f"训练任务已启动，状态: {current_status}")
        return task_id
        
    except Exception as e:
        print(f"训练 {model_type} 模型时出错: {str(e)}")
        return False

def monitor_training_tasks(task_ids: dict, max_wait_time: int = 7200):
    """监控多个训练任务的进度"""
    training_manager = TrainingManager()
    start_time = time.time()
    
    # 初始化状态跟踪
    task_status = {model_type: {'id': task_id, 'completed': False, 'success': False} 
                  for model_type, task_id in task_ids.items() if task_id}
    
    while True:
        all_completed = True
        status_updates = []
        
        for model_type, info in task_status.items():
            if info['completed']:
                continue
                
            task_id = info['id']
            task_info = training_manager.get_training_status(task_id)
            
            if task_info:
                current_status = task_info.get('status', 'unknown')
                progress = task_info.get('progress', 0)
                
                if current_status in ['completed', 'failed', 'cancelled']:
                    info['completed'] = True
                    if current_status == 'completed':
                        info['success'] = True
                        result = training_manager.get_training_result(task_id)
                        status_updates.append(f"✅ {model_type}: 训练完成")
                        if result:
                            status_updates.append(f"   损失={result.get('final_loss', 'N/A')}, 准确率={result.get('final_accuracy', 'N/A')}")
                    else:
                        error_msg = task_info.get('error_message', '未知错误')
                        status_updates.append(f"❌ {model_type}: 训练失败 - {error_msg}")
                else:
                    all_completed = False
                    # 每10%进度更新一次
                    if progress % 10 < 0.1:  # 当进度接近10的倍数时
                        status_updates.append(f"⏳ {model_type}: {progress:.1%}")
            else:
                # 无法获取状态，假设仍在进行中
                all_completed = False
        
        # 显示状态更新
        if status_updates:
            print(f"\n训练状态更新 ({time.strftime('%H:%M:%S')}):")
            for update in status_updates:
                print(f"  {update}")
        
        # 检查是否全部完成
        if all_completed:
            print("\n所有训练任务已完成!")
            break
        
        # 检查超时
        if time.time() - start_time > max_wait_time:
            print(f"\n训练超时，已等待{max_wait_time/3600:.1f}小时")
            break
        
        # 等待一段时间再检查
        time.sleep(10)
        
        # 定期清理内存
        gc.collect()
    
    # 返回训练结果
    results = {}
    for model_type, info in task_status.items():
        results[model_type] = {
            'success': info.get('success', False),
            'completed': info.get('completed', False)
        }
    
    return results

def train_all_models_efficient():
    """高效训练所有模型，使用并行训练策略"""
    # 定义所有支持的模型类型及其对应的数据集路径和训练参数
    model_configs = {
        "manager": {
            "dataset": "data/datasets/manager_dataset.json",
            "epochs": 30,
            "batch_size": 32,
            "learning_rate": 0.001,
            "priority": 1
        },
        "language": {
            "dataset": "data/datasets/language_dataset.json",
            "epochs": 40,  # 比简单模式少一些，但使用更高效的学习率
            "batch_size": 32,
            "learning_rate": 0.0005,
            "priority": 1
        },
        "knowledge": {
            "dataset": "data/datasets/knowledge_dataset.json",
            "epochs": 20,
            "batch_size": 16,
            "learning_rate": 0.0003,
            "priority": 2
        },
        "programming": {
            "dataset": "data/datasets/programming_dataset.json",
            "epochs": 20,
            "batch_size": 16,
            "learning_rate": 0.0003,
            "priority": 2
        },
        "planning": {
            "dataset": "data/datasets/planning_dataset.json",
            "epochs": 20,
            "batch_size": 16,
            "learning_rate": 0.0003,
            "priority": 2
        }
        # 注意：为了高效训练，我们只训练核心模型，其他模型可以后续训练
        # 如果需要训练更多模型，可以在此添加
    }
    
    print("开始高效训练所有模型...")
    print(f"本次将训练 {len(model_configs)} 个核心模型")
    print("=" * 60)
    
    # 创建必要的目录
    for model_type in model_configs:
        os.makedirs(os.path.join("data", "models", model_type), exist_ok=True)
    
    # 启动所有训练任务（并行）
    task_ids = {}
    start_time = time.time()
    
    print(f"正在启动训练任务...")
    for model_type, config in model_configs.items():
        dataset_path = config["dataset"]
        
        # 准备训练参数
        training_params = {
            "model_type": model_type,
            "dataset_path": dataset_path,
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "validation_split": 0.2,
            "save_dir": os.path.join("data", "models", model_type),
            "early_stopping": True,
            "patience": 5,
            "learning_rate_decay": True,
            "optimizer": "adamw",  # 使用AdamW优化器
            "weight_decay": 0.01  # 权重衰减
        }
        
        # 启动训练任务
        task_id = train_model_efficient(model_type, dataset_path, training_params)
        if task_id:
            task_ids[model_type] = task_id
            print(f"  {model_type}: 任务ID {task_id}")
        else:
            print(f"  {model_type}: 启动失败")
        
        # 短暂延迟以避免资源竞争
        time.sleep(1)
    
    print(f"\n所有训练任务已启动，共 {len(task_ids)} 个任务")
    
    # 监控训练进度
    if task_ids:
        print(f"\n开始监控训练进度...")
        results = monitor_training_tasks(task_ids, max_wait_time=10800)  # 3小时最大等待时间
        
        # 统计结果
        successful_count = sum(1 for result in results.values() if result.get('success', False))
        total_count = len(results)
        
        print(f"\n{'='*60}")
        print("高效训练完成总结:")
        print(f"成功训练模型: {successful_count}/{total_count}")
        print(f"总耗时: {(time.time() - start_time)/60:.1f} 分钟")
        
        if successful_count == total_count:
            print("✅ 所有核心模型训练成功！")
            print("系统已准备好进行高效功能测试。")
            return 0
        else:
            failed_count = total_count - successful_count
            print(f"⚠️  有 {failed_count} 个模型训练失败。")
            # 显示失败模型
            for model_type, result in results.items():
                if not result.get('success', False):
                    print(f"  - {model_type}: 训练失败")
            return 1
    else:
        print("❌ 没有成功启动任何训练任务")
        return 1

def train_main_dialogue_models_efficient():
    """高效训练主要对话模型（语言模型和管理器模型）"""
    print("高效训练主要对话模型...")
    
    # 只训练语言模型和管理器模型
    dialogue_models = [
        {
            "type": "language",
            "display_name": "语言模型",
            "dataset": "data/datasets/language_dataset.json",
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.0005
        },
        {
            "type": "manager",
            "display_name": "管理模型",
            "dataset": "data/datasets/manager_dataset.json",
            "epochs": 30,
            "batch_size": 32,
            "learning_rate": 0.001
        }
    ]
    
    task_ids = {}
    start_time = time.time()
    
    for model_info in dialogue_models:
        model_type = model_info["type"]
        dataset_path = model_info["dataset"]
        
        # 准备训练参数
        training_params = {
            "model_type": model_type,
            "dataset_path": dataset_path,
            "epochs": model_info["epochs"],
            "batch_size": model_info["batch_size"],
            "learning_rate": model_info["learning_rate"],
            "validation_split": 0.2,
            "save_dir": os.path.join("data", "models", model_type),
            "early_stopping": True,
            "patience": 5,
            "learning_rate_decay": True
        }
        
        # 启动训练任务
        task_id = train_model_efficient(model_type, dataset_path, training_params)
        if task_id:
            task_ids[model_type] = task_id
            print(f"{model_info['display_name']}: 任务ID {task_id}")
        else:
            print(f"{model_info['display_name']}: 启动失败")
        
        time.sleep(1)
    
    # 监控训练进度
    if task_ids:
        results = monitor_training_tasks(task_ids, max_wait_time=7200)
        
        successful_count = sum(1 for result in results.values() if result.get('success', False))
        total_count = len(results)
        
        print(f"\n对话模型训练完成:")
        print(f"成功训练模型: {successful_count}/{total_count}")
        print(f"总耗时: {(time.time() - start_time)/60:.1f} 分钟")
        
        if successful_count == total_count:
            print("✅ 所有对话模型训练成功！")
            return True
        else:
            print("❌ 部分对话模型训练失败")
            return False
    else:
        print("❌ 没有成功启动任何对话模型训练任务")
        return False

def main():
    print("Self Soul - 高效训练脚本")
    print("=" * 60)
    print("模式选择:")
    print("  1. 训练所有核心模型（高效并行）")
    print("  2. 训练主要对话模型")
    print("  3. 自定义训练")
    print("=" * 60)
    
    try:
        choice = input("请选择训练模式 (1-3): ").strip()
        
        if choice == "1":
            return train_all_models_efficient()
        elif choice == "2":
            success = train_main_dialogue_models_efficient()
            return 0 if success else 1
        elif choice == "3":
            print("自定义训练功能暂未实现")
            return 1
        else:
            print("无效选择，默认使用模式1")
            return train_all_models_efficient()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断。")
        return 130
    except Exception as e:
        print(f"\n训练过程中发生未预期异常: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
