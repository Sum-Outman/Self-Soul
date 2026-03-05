"""
Training Manager 重构后使用示例

此文件展示了如何使用重构后的TrainingManager及其各个模块。
重构后的架构包括：
1. TrainingManager (协调层) - 高层API接口
2. TrainingScheduler - 训练任务调度
3. ResourceManager - 系统资源管理
4. TrainingMonitor - 训练监控
5. DataPreprocessor - 数据预处理

所有模块都可以独立使用，也可以通过TrainingManager协调使用。
"""

import sys
import os
# 添加项目根目录到Python路径，确保可以导入core模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_1_standalone_modules():
    """示例1：独立使用各个模块"""
    print("=== 示例1：独立使用各个模块 ===")
    
    from core.training_scheduler import TrainingScheduler, TrainingJob
    from core.resource_manager import ResourceManager
    from core.training_monitor import TrainingMonitor
    from core.data_preprocessor import DataPreprocessor, DataPreprocessorConfig, DataType
    
    # 1. 初始化各个模块
    print("1. 初始化各个独立模块...")
    scheduler = TrainingScheduler()
    resource_manager = ResourceManager()
    monitor = TrainingMonitor()
    
    preprocessor_config = DataPreprocessorConfig(
        batch_size=32,
        shuffle=True,
        num_workers=4,
        augmentation_level='basic'
    )
    data_preprocessor = DataPreprocessor(preprocessor_config)
    
    print("   ✓ 所有模块初始化成功")
    
    # 2. 使用TrainingScheduler调度任务
    print("\n2. 使用TrainingScheduler调度任务...")
    job = TrainingJob(
        model_id="language_model",
        data_config={"dataset_path": "data/training", "data_type": "text"},
        training_params={"epochs": 10, "batch_size": 32, "learning_rate": 0.001},
        priority="normal"
    )
    
    schedule_result = scheduler.schedule_job(job)
    print(f"   调度结果: {'成功' if schedule_result['success'] else '失败'}")
    if schedule_result['success']:
        print(f"   任务ID: {schedule_result['job_id']}")
        print(f"   优先级: {schedule_result['priority']}")
        print(f"   预估队列时间: {schedule_result['estimated_queue_time']:.1f}秒")
    
    # 3. 使用ResourceManager分配资源
    print("\n3. 使用ResourceManager分配资源...")
    resource_req = {"cpu_cores": 2, "memory_gb": 4}
    alloc_result = resource_manager.allocate_resources("test_job_1", resource_req)
    print(f"   资源分配: {'成功' if alloc_result['success'] else '失败'}")
    
    if alloc_result['success']:
        # 获取系统状态
        system_status = resource_manager.get_system_status()
        print(f"   CPU使用率: {system_status['cpu']['utilization_percent']:.1f}%")
        print(f"   内存使用率: {system_status['memory']['utilization_percent']:.1f}%")
        
        # 释放资源
        release_result = resource_manager.release_resources("test_job_1")
        print(f"   资源释放: {'成功' if release_result['success'] else '失败'}")
    
    # 4. 使用TrainingMonitor监控
    print("\n4. 使用TrainingMonitor监控...")
    monitor.start_monitoring("monitor_job_1")
    monitor_status = monitor.get_status()
    print(f"   监控状态: {monitor_status['status']}")
    print(f"   活跃监控任务: {monitor_status['active_monitors']}")
    
    # 5. 使用DataPreprocessor
    print("\n5. 使用DataPreprocessor...")
    print(f"   批次大小: {data_preprocessor.config.batch_size}")
    print(f"   数据增强级别: {data_preprocessor.config.augmentation_level}")
    
    return True


def example_2_training_manager_coordination():
    """示例2：使用TrainingManager协调层"""
    print("\n=== 示例2：使用TrainingManager协调层 ===")
    
    from core.training_manager import TrainingManager, get_training_manager
    
    # 1. 获取TrainingManager实例（单例模式）
    print("1. 获取TrainingManager单例实例...")
    tm1 = TrainingManager()
    tm2 = get_training_manager()  # 另一种获取方式
    
    # 验证单例模式
    print(f"   单例验证: tm1 is tm2 = {tm1 is tm2}")
    print(f"   协调层初始化完成")
    
    # 2. 检查各个子模块
    print("\n2. 检查各个子模块...")
    modules = {
        "TrainingScheduler": tm1.scheduler,
        "ResourceManager": tm1.resource_manager,
        "TrainingMonitor": tm1.monitor,
        "DataPreprocessor": tm1.data_preprocessor
    }
    
    for name, module in modules.items():
        status = "✓ 已初始化" if module is not None else "✗ 未初始化"
        print(f"   {name}: {status}")
    
    # 3. 启动训练任务
    print("\n3. 启动训练任务...")
    start_result = tm1.start_training(
        model_id="vision_model",
        data_config={
            "dataset_path": "data/images",
            "data_type": "image",
            "image_size": [224, 224],
            "normalize": True
        },
        training_params={
            "epochs": 5,
            "batch_size": 16,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss_function": "cross_entropy"
        }
    )
    
    print(f"   训练启动: {'成功' if start_result['success'] else '失败'}")
    if start_result['success']:
        job_id = start_result['job_id']
        print(f"   任务ID: {job_id}")
        print(f"   调度信息: {start_result.get('scheduled_info', {})}")
        
        # 4. 获取训练状态
        print("\n4. 获取训练状态...")
        time.sleep(1)  # 等待一下让训练开始
        
        status_result = tm1.get_training_status(job_id)
        if status_result['success']:
            print(f"   任务状态: {status_result['status']}")
            print(f"   模型ID: {status_result['model_id']}")
            print(f"   开始时间: {status_result['start_time']}")
            
            if 'metrics' in status_result:
                print(f"   监控指标: 已收集{len(status_result['metrics'])}个指标")
            
            if 'resource_usage' in status_result:
                print(f"   资源使用: 已分配")
        else:
            print(f"   状态获取失败: {status_result['message']}")
        
        # 5. 获取仪表板数据
        print("\n5. 获取仪表板数据...")
        dashboard_data = tm1.get_dashboard_data()
        
        print(f"   训练进度: {len(dashboard_data.get('training_progress', {}))}个任务")
        print(f"   系统状态: {'已获取' if 'system_status' in dashboard_data else '未获取'}")
        print(f"   AGI指标: {'已获取' if 'agi_metrics' in dashboard_data else '未获取'}")
        
        # 6. 停止训练（演示用）
        print("\n6. 停止训练任务...")
        stop_result = tm1.stop_training(job_id)
        print(f"   停止结果: {'成功' if stop_result['success'] else '失败'}")
    
    return True


def example_3_enhanced_training_manager():
    """示例3：使用EnhancedTrainingManager"""
    print("\n=== 示例3：使用EnhancedTrainingManager ===")
    
    from core.training_manager import TrainingManager
    from core.enhanced_training_manager import EnhancedTrainingManager
    
    try:
        # 1. 初始化
        print("1. 初始化EnhancedTrainingManager...")
        base_tm = TrainingManager()
        etm = EnhancedTrainingManager(base_tm)
        
        print("   ✓ EnhancedTrainingManager初始化成功")
        
        # 2. 设置训练设备
        print("\n2. 设置训练设备...")
        device_result = etm.set_training_device("cpu")
        print(f"   设备设置: {'成功' if device_result['success'] else '失败'}")
        if device_result['success']:
            print(f"   当前设备: {device_result['device']}")
        
        # 3. 获取训练状态
        print("\n3. 获取增强训练状态...")
        etm_status = etm.get_training_status()
        print(f"   当前设备: {etm_status.get('current_device', '未知')}")
        print(f"   可用设备: {list(etm_status.get('available_devices', {}).keys())}")
        print(f"   活跃任务数: {etm_status.get('active_jobs_count', 0)}")
        
        # 4. 优化设备配置
        print("\n4. 优化CPU设备配置...")
        optimize_result = etm.optimize_training_for_device("cpu")
        print(f"   优化结果: {'成功' if optimize_result['success'] else '失败'}")
        if optimize_result['success']:
            config = optimize_result['optimized_config']
            print(f"   批次大小: {config.get('batch_size')}")
            print(f"   混合精度: {config.get('mixed_precision')}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ EnhancedTrainingManager示例失败: {e}")
        return False


def example_4_integration_with_other_components():
    """示例4：与其他组件集成"""
    print("\n=== 示例4：与其他组件集成 ===")
    
    try:
        # 1. 与model_registry集成
        print("1. 与ModelRegistry集成...")
        from core.model_registry import get_model_registry
        from core.training_manager import TrainingManager
        
        model_registry = get_model_registry()
        tm = TrainingManager(model_registry)
        
        print(f"   ModelRegistry: {model_registry is not None}")
        print(f"   TrainingManager模型注册表: {tm.model_registry is not None}")
        
        # 2. 与training_preparation集成
        print("\n2. 与TrainingPreparation集成...")
        from core.training_preparation import TrainingPreparation
        
        preparation = TrainingPreparation(model_registry, tm)
        print(f"   TrainingPreparation初始化: 成功")
        
        # 3. 示例模型准备
        print("\n3. 示例模型准备流程...")
        prepare_result = tm.prepare_model("test_model")
        print(f"   模型准备: {'成功' if prepare_result['success'] else '失败'}")
        
        # 4. 示例设置模型状态
        print("\n4. 示例设置模型状态...")
        status_result = tm.set_model_status("test_model", "training", 0.25)
        print(f"   状态设置: {'成功' if status_result else '失败'}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ 集成示例失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_5_advanced_features():
    """示例5：高级功能演示"""
    print("\n=== 示例5：高级功能演示 ===")
    
    from core.training_manager import TrainingManager
    
    tm = TrainingManager()
    
    # 1. 批量获取训练状态
    print("1. 批量获取训练状态...")
    batch_status = tm.get_training_status()  # 不传job_id获取所有任务
    if batch_status['success']:
        print(f"   总任务数: {batch_status['total_jobs']}")
        print(f"   活跃任务: {batch_status['active_jobs']}")
        print(f"   系统状态: {'已获取' if 'system_status' in batch_status else '未获取'}")
    
    # 2. 仪表板更新回调
    print("\n2. 仪表板更新回调设置...")
    
    def dashboard_callback(data):
        """自定义仪表板更新回调函数"""
        active_jobs = len(data.get('training_progress', {}))
        print(f"   [回调] 活跃训练任务: {active_jobs}")
        if 'system_status' in data:
            cpu_usage = data['system_status']['cpu']['utilization_percent']
            memory_usage = data['system_status']['memory']['utilization_percent']
            print(f"   [回调] CPU使用率: {cpu_usage:.1f}%, 内存使用率: {memory_usage:.1f}%")
    
    tm.set_dashboard_update_callback(dashboard_callback)
    print("   ✓ 仪表板回调已设置")
    
    # 3. 手动触发仪表板更新
    print("\n3. 手动触发仪表板更新...")
    # 在实际应用中，仪表板会自动更新
    # 这里我们手动触发一次作为演示
    tm._update_dashboard()
    
    # 4. 系统关闭
    print("\n4. 系统关闭演示...")
    print("   （在实际应用中，应在程序退出前调用tm.shutdown()）")
    # tm.shutdown()  # 注释掉，因为这是演示
    
    return True


def run_all_examples():
    """运行所有示例"""
    print("=" * 60)
    print("Training Manager 重构后使用示例")
    print("=" * 60)
    
    results = []
    
    # 运行示例1
    results.append(("独立模块使用", example_1_standalone_modules()))
    
    # 运行示例2
    results.append(("协调层使用", example_2_training_manager_coordination()))
    
    # 运行示例3
    results.append(("增强训练管理器", example_3_enhanced_training_manager()))
    
    # 运行示例4
    results.append(("组件集成", example_4_integration_with_other_components()))
    
    # 运行示例5
    results.append(("高级功能", example_5_advanced_features()))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("示例执行总结:")
    print("=" * 60)
    
    success_count = 0
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{name:20} {status}")
        if success:
            success_count += 1
    
    print(f"\n总计: {success_count}/{len(results)} 个示例通过")
    
    return success_count == len(results)


if __name__ == "__main__":
    # 运行所有示例
    success = run_all_examples()
    
    if success:
        print("\n" + "=" * 60)
        print("所有示例执行成功！")
        print("TrainingManager重构验证完成。")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("部分示例执行失败，请检查相关问题。")
        print("=" * 60)
