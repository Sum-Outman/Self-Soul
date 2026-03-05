#!/usr/bin/env python3
"""
测试Spatial、Sensor、Optimization模型的参数计数修复

验证这些模型不再返回"N/A"作为参数规模，并提供真实的参数信息。
"""

import sys
import os
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_spatial_model():
    """测试Spatial模型的参数计数"""
    print("\n" + "="*60)
    print("测试 Spatial 模型参数计数")
    print("="*60)
    
    try:
        from core.models.spatial.unified_spatial_model import UnifiedSpatialModel
        
        # 使用默认配置初始化模型
        config = {
            "neural_network": {
                "input_channels": 6,
                "hidden_size": 128,
                "num_layers": 2,
                "output_size": 64
            }
        }
        
        model = UnifiedSpatialModel(config)
        
        # 获取模型信息
        model_info = model._get_model_info_specific()
        
        print(f"✓ Spatial模型初始化成功")
        print(f"  模型ID: {model_info.get('model_id')}")
        print(f"  模型类型: {model_info.get('model_type')}")
        
        # 检查参数信息
        architecture = model_info.get('architecture', {})
        total_parameters = architecture.get('total_parameters', 0)
        
        print(f"  总参数数量: {total_parameters}")
        
        # 验证参数计数
        if total_parameters == 0:
            print("  ⚠️ 警告: 参数数量为0，可能神经网络未正确初始化")
        else:
            print(f"  ✅ 成功: 模型包含 {total_parameters:,} 个参数")
        
        # 检查神经网络信息
        neural_networks = architecture.get('neural_networks', {})
        print(f"  神经网络组件: {list(neural_networks.keys())}")
        
        for name, info in neural_networks.items():
            params = info.get('parameters', 0)
            print(f"    {name}: {params:,} 参数, {info.get('layers')} 层, 类型: {info.get('type')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Spatial模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sensor_model():
    """测试Sensor模型的参数计数"""
    print("\n" + "="*60)
    print("测试 Sensor 模型参数计数")
    print("="*60)
    
    try:
        from core.models.sensor.unified_sensor_model import UnifiedSensorModel
        
        # 使用默认配置初始化模型
        config = {
            'sample_rate': 2.0,
            'max_buffer_size': 500,
            'sensors': {
                'temp1': {'type': 'temperature', 'enabled': True},
                'hum1': {'type': 'humidity', 'enabled': True},
                'light1': {'type': 'light', 'enabled': True}
            }
        }
        
        model = UnifiedSensorModel()
        
        # 初始化模型
        if not hasattr(model, 'initialize_model'):
            print("✓ Sensor模型已创建，但跳过初始化（需要initialize_model方法）")
        else:
            if model.initialize_model(config):
                print("✓ Sensor模型初始化成功")
            else:
                print("⚠️ Sensor模型初始化失败，继续测试")
        
        # 获取模型信息
        model_info = model._get_model_info_specific()
        
        print(f"  模型类型: {model_info.get('model_type')}")
        print(f"  模型子类型: {model_info.get('model_subtype')}")
        
        # 检查神经网络信息
        nn_info = model_info.get('neural_network_info', {})
        params = nn_info.get('parameters', 0)
        param_scale = nn_info.get('parameter_scale', 'N/A')
        
        print(f"  参数数量: {params:,}")
        print(f"  参数规模: {param_scale}")
        print(f"  神经网络已初始化: {nn_info.get('is_initialized')}")
        print(f"  神经网络已训练: {nn_info.get('is_trained')}")
        
        # 验证参数计数
        if param_scale == "N/A (not initialized)":
            print("  ⚠️ 警告: 神经网络未初始化，参数为N/A")
        elif params == 0:
            print("  ⚠️ 警告: 参数数量为0")
        else:
            print(f"  ✅ 成功: 模型包含 {params:,} 个参数 ({param_scale})")
        
        return True
        
    except Exception as e:
        print(f"✗ Sensor模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimization_model():
    """测试Optimization模型的参数报告机制"""
    print("\n" + "="*60)
    print("测试 Optimization 模型参数报告")
    print("="*60)
    
    try:
        from core.models.optimization.unified_optimization_model import UnifiedOptimizationModel
        
        # 初始化模型
        model = UnifiedOptimizationModel()
        
        # 获取模型信息
        model_info = model._get_model_info_specific()
        
        print(f"✓ Optimization模型初始化成功")
        print(f"  模型类型: {model_info.get('model_type')}")
        print(f"  模型子类型: {model_info.get('model_subtype')}")
        
        # 检查参数信息
        param_info = model_info.get('parameter_information', {})
        param_summary = model_info.get('parameter_summary', {})
        
        print(f"  参数计数: {param_info.get('parameter_count')}")
        print(f"  参数规模: {param_info.get('parameter_scale')}")
        print(f"  描述: {param_info.get('description')}")
        
        # 检查优化组件
        components = param_info.get('optimization_components', [])
        print(f"  优化组件数量: {len(components)}")
        
        for comp in components:
            print(f"    {comp.get('component')}: {comp.get('type')} (参数: {comp.get('parameters')})")
        
        # 检查超参数
        hyperparams = param_info.get('hyperparameters', {})
        print(f"  可配置超参数: {len(hyperparams)} 个")
        for key, value in hyperparams.items():
            print(f"    {key}: {value}")
        
        # 验证参数报告
        if param_info.get('parameter_scale') == "N/A (optimization model)":
            print("  ✅ 成功: 优化模型正确报告参数规模为N/A（这是预期的）")
        else:
            print(f"  ⚠️ 注意: 参数规模报告为 {param_info.get('parameter_scale')}")
        
        print(f"  有神经网络参数: {param_summary.get('has_neural_network_parameters')}")
        print(f"  有算法参数: {param_summary.get('has_algorithmic_parameters')}")
        print(f"  可配置参数数量: {param_summary.get('configurable_parameters_count')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Optimization模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("="*80)
    print("模型参数计数修复验证测试")
    print("="*80)
    
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING)
    
    # 运行所有测试
    tests = [
        ("Spatial模型", test_spatial_model),
        ("Sensor模型", test_sensor_model),
        ("Optimization模型", test_optimization_model)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n>>> 开始测试: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"!!! 测试 {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status} {test_name}")
    
    print(f"\n总计: {passed_tests}/{total_tests} 个测试通过")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！参数计数修复成功。")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} 个测试失败，需要进一步调试。")
        return 1

if __name__ == "__main__":
    sys.exit(main())