#!/usr/bin/env python3
"""
测试控制模块导入
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 全局导入用于测试
try:
    from src.control.motion_control import MotionControlSystem
    from src.control.hardware_interface import HardwareControlSystem
    from src.control.sensor_integration import SensorIntegrationSystem
    from src.control.motor_control import MotorControlSystem
    from src.cognitive.architecture import UnifiedCognitiveArchitecture
    from src.api.control import router as control_router
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"导入错误: {e}")
    import traceback
    traceback.print_exc()
    IMPORT_SUCCESS = False
except Exception as e:
    print(f"其他错误: {e}")
    import traceback
    traceback.print_exc()
    IMPORT_SUCCESS = False

def test_control_imports():
    """测试控制模块导入"""
    print("测试控制模块导入...")
    
    if not IMPORT_SUCCESS:
        print("✗ 导入失败")
        return False
    
    try:
        print("✓ MotionControlSystem 导入成功")
        print("✓ HardwareControlSystem 导入成功")
        print("✓ SensorIntegrationSystem 导入成功")
        print("✓ MotorControlSystem 导入成功")
        print("✓ UnifiedCognitiveArchitecture 导入成功")
        print("✓ 控制API路由导入成功")
        
        print("\n所有控制模块导入成功！")
        return True
        
    except Exception as e:
        print(f"✗ 测试错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_architecture_integration():
    """测试架构集成"""
    print("\n测试架构集成...")
    
    try:
        # 创建架构实例
        config = {
            'embedding_dim': 512,
            'max_shared_memory_mb': 100
        }
        
        agi = UnifiedCognitiveArchitecture(config)
        print("✓ 统一认知架构创建成功")
        
        # 测试控制组件属性
        print("测试控制组件属性...")
        
        # 注意：控制组件可能不可用（CONTROL_AVAILABLE）
        # 所以我们需要检查它们是否可用
        
        if hasattr(agi, 'motion_control'):
            motion = agi.motion_control
            if motion is not None:
                print("✓ 运动控制系统可用")
            else:
                print("⚠ 运动控制系统不可用（预期中）")
        else:
            print("✗ motion_control 属性不存在")
        
        if hasattr(agi, 'hardware_control'):
            hardware = agi.hardware_control
            if hardware is not None:
                print("✓ 硬件控制系统可用")
            else:
                print("⚠ 硬件控制系统不可用（预期中）")
        else:
            print("✗ hardware_control 属性不存在")
        
        if hasattr(agi, 'sensor_integration'):
            sensor = agi.sensor_integration
            if sensor is not None:
                print("✓ 传感器集成系统可用")
            else:
                print("⚠ 传感器集成系统不可用（预期中）")
        else:
            print("✗ sensor_integration 属性不存在")
        
        if hasattr(agi, 'motor_control'):
            motor = agi.motor_control
            if motor is not None:
                print("✓ 电机控制系统可用")
            else:
                print("⚠ 电机控制系统不可用（预期中）")
        else:
            print("✗ motor_control 属性不存在")
        
        # 测试组件注册
        print(f"\n组件注册表: {agi.component_registry}")
        
        # 测试组件初始化
        print("初始化组件...")
        agi.initialize_components()
        print("✓ 组件初始化完成")
        
        return True
        
    except Exception as e:
        print(f"✗ 架构集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("控制模块导入测试")
    print("=" * 60)
    
    import_success = test_control_imports()
    
    if import_success:
        integration_success = test_architecture_integration()
        
        if integration_success:
            print("\n" + "=" * 60)
            print("所有测试通过！控制模块已成功集成到统一认知架构中。")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("架构集成测试失败。")
            print("=" * 60)
            sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("导入测试失败。")
        print("=" * 60)
        sys.exit(1)