import os
import json
import sys
import time
from core.training_manager import TrainingManager

def train_dialogue_model(model_type: str, dataset_path: str):
    """训练单个对话模型"""
    print(f"\n开始训练 {model_type} 模型...")
    
    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: {model_type} 模型的数据集不存在: {dataset_path}")
        return False
    
    try:
        # 加载数据集以验证格式
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"加载了 {len(dataset)} 条训练数据")
        
        # 获取训练管理器实例
        training_manager = TrainingManager()
        
        # 准备训练参数
        training_params = {
            "model_type": model_type,
            "dataset_path": dataset_path,
            "epochs": 50 if model_type == "language" else 30,
            "batch_size": 32,
            "learning_rate": 0.001,
            "validation_split": 0.2,
            "save_dir": os.path.join("data", "models", model_type),
            "early_stopping": True,
            "patience": 5
        }
        
        # 创建训练任务
        print(f"正在创建 {model_type} 模型的训练任务...")
        task_id = training_manager.create_training_task(
            model_type=model_type,
            parameters=training_params,
            priority=1
        )
        
        if not task_id:
            print(f"创建训练任务失败")
            return False
        
        print(f"训练任务已创建，任务ID: {task_id}")
        
        # 启动训练
        print(f"开始训练 {model_type} 模型...")
        training_manager.start_training_task(task_id)
        
        # 监控训练进度
        max_wait_time = 3600  # 最大等待时间1小时
        start_time = time.time()
        last_status = ""
        
        while True:
            # 检查训练任务状态
            task_status = training_manager.get_training_status(task_id)
            
            if task_status:
                current_status = task_status.get('status', 'unknown')
                progress = task_status.get('progress', 0)
                current_epoch = task_status.get('current_epoch', 0)
                total_epochs = task_status.get('total_epochs', 1)
                
                # 显示进度变化
                status_display = f"训练状态: {current_status}, 进度: {progress:.1%}, 轮次: {current_epoch}/{total_epochs}"
                if status_display != last_status:
                    print(status_display)
                    last_status = status_display
                
                # 检查是否完成
                if current_status in ['completed', 'failed', 'cancelled']:
                    if current_status == 'completed':
                        print(f"训练成功完成！")
                        # 获取训练结果
                        result = training_manager.get_training_result(task_id)
                        if result:
                            print(f"训练结果: 最终损失={result.get('final_loss', 'N/A')}, "
                                  f"最终准确率={result.get('final_accuracy', 'N/A')}")
                        return True
                    else:
                        print(f"训练失败，状态: {current_status}")
                        error_msg = task_status.get('error_message', '未知错误')
                        print(f"错误信息: {error_msg}")
                        return False
            
            # 检查超时
            if time.time() - start_time > max_wait_time:
                print(f"训练超时，已等待{max_wait_time}秒")
                return False
            
            # 等待一段时间再检查
            time.sleep(2)
    
    except Exception as e:
        print(f"训练 {model_type} 模型时发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("开始训练对话模型系统...")
    
    # 定义要训练的对话模型类型
    dialogue_models = [
        {
            "type": "language",
            "display_name": "语言模型",
            "dataset": "data/datasets/language_dataset.json"
        },
        {
            "type": "manager",
            "display_name": "管理模型",
            "dataset": "data/datasets/manager_dataset.json"
        }
    ]
    
    # 创建必要的目录
    os.makedirs("data/models/language", exist_ok=True)
    os.makedirs("data/models/manager", exist_ok=True)
    
    # 训练每个模型
    success_count = 0
    total_models = len(dialogue_models)
    
    for model_info in dialogue_models:
        model_type = model_info["type"]
        dataset_path = model_info["dataset"]
        
        success = train_dialogue_model(model_type, dataset_path)
        
        if success:
            success_count += 1
            print(f"{model_info['display_name']} 训练完成！")
        else:
            print(f"{model_info['display_name']} 训练失败！")
        
        print("-" * 50)
    
    # 输出总结
    print(f"\n训练完成总结:")
    print(f"成功训练模型: {success_count}/{total_models}")
    
    if success_count == total_models:
        print("所有对话模型训练成功！")
        print("模型已准备好进行对话功能测试。")
        return 0
    else:
        print(f"有 {total_models - success_count} 个模型训练失败。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
