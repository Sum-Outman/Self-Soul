import os
import sys
import json
import time
from core.training_manager import TrainingManager

def train_model_simple(model_type: str, dataset_path: str) -> bool:
    """简单训练单个模型"""
    print(f"\n开始训练 {model_type} 模型...")
    
    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"警告: {model_type} 模型的数据集不存在，跳过训练: {dataset_path}")
        return False
    
    try:
        # 加载数据集以验证格式
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"加载了 {len(dataset)} 条训练数据")
        
        # 获取训练管理器实例
        training_manager = TrainingManager()
        
        # 根据模型类型设置不同的训练参数
        if model_type == "language":
            epochs = 50  # 语言模型需要更多训练轮次
            batch_size = 32
            learning_rate = 0.001
        elif model_type == "manager":
            epochs = 30  # 管理器模型次之
            batch_size = 32
            learning_rate = 0.001
        elif model_type in ["knowledge", "programming", "planning"]:
            epochs = 20  # 知识密集型模型
            batch_size = 16
            learning_rate = 0.0005
        else:
            epochs = 10  # 其他模型
            batch_size = 16
            learning_rate = 0.0005
        
        # 准备训练参数
        training_params = {
            "model_type": model_type,
            "dataset_path": dataset_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "validation_split": 0.2,
            "save_dir": os.path.join("data", "models", model_type),
            "early_stopping": True,
            "patience": 3
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

def train_all_models_simple():
    """训练所有模型（简单模式）"""
    # 定义所有支持的模型类型及其对应的数据集路径
    model_types = [
        ("manager", "data/datasets/manager_dataset.json"),
        ("language", "data/datasets/language_dataset.json"),
        ("audio", "data/datasets/audio_dataset.json"),
        ("vision_image", "data/datasets/vision_image_dataset.json"),
        ("vision_video", "data/datasets/vision_video_dataset.json"),
        ("spatial", "data/datasets/spatial_dataset.json"),
        ("sensor", "data/datasets/sensor_dataset.json"),
        ("computer", "data/datasets/computer_dataset.json"),
        ("motion", "data/datasets/motion_dataset.json"),
        ("knowledge", "data/datasets/knowledge_dataset.json"),
        ("programming", "data/datasets/programming_dataset.json"),
        ("planning", "data/datasets/planning_dataset.json"),
        ("autonomous", "data/datasets/autonomous_dataset.json"),
        ("emotion", "data/datasets/emotion_dataset.json"),
        ("prediction", "data/datasets/prediction_dataset.json"),
        ("collaboration", "data/datasets/collaboration_dataset.json"),
        ("optimization", "data/datasets/optimization_dataset.json"),
        ("finance", "data/datasets/finance_dataset.json"),
        ("medical", "data/datasets/medical_dataset.json"),
        ("value_alignment", "data/datasets/value_alignment_dataset.json"),
        ("stereo_vision", "data/datasets/stereo_vision_dataset.json")
    ]
    
    print("开始训练所有模型（简单模式）...")
    print(f"共需训练 {len(model_types)} 种模型")
    
    # 创建必要的目录
    for model_type, _ in model_types:
        os.makedirs(os.path.join("data", "models", model_type), exist_ok=True)
    
    # 为每种模型类型训练模型
    successful_count = 0
    failed_count = 0
    
    for model_type, dataset_path in model_types:
        success = train_model_simple(model_type, dataset_path)
        if success:
            successful_count += 1
            print(f"✅ {model_type} 模型训练完成！")
        else:
            failed_count += 1
            print(f"❌ {model_type} 模型训练失败！")
        
        # 短暂休息以释放内存
        print(f"休息5秒以释放内存...")
        time.sleep(5)
    
    print(f"\n所有模型训练完成！")
    print(f"成功: {successful_count} 个模型")
    print(f"失败: {failed_count} 个模型")
    
    return successful_count, failed_count

def main():
    print("Self Soul - 所有模型简单训练脚本")
    print("此脚本将依次训练所有支持的模型类型")
    print("=" * 60)
    
    try:
        successful_count, failed_count = train_all_models_simple()
        
        if failed_count == 0:
            print("\n✅ 所有模型训练成功！")
            print("系统已准备好进行全方位功能测试。")
            return 0
        else:
            print(f"\n⚠️  有 {failed_count} 个模型训练失败。")
            print("建议检查数据集路径和训练配置。")
            return 1
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
