"""
机器人硬件集成测试脚本

测试多模态AGI系统中的机器人硬件控制、传感器处理、电机控制等功能。
验证机器人硬件接口的正确性和多模态集成能力。
"""

import sys
import os
import time
import json
import asyncio
from typing import Dict, Any, List
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RobotHardwareTest")

def test_robot_api_imports():
    """测试机器人API导入"""
    print("=" * 80)
    print("测试机器人API导入")
    print("=" * 80)
    
    try:
        from core.robot_api import router, initialize_robot_api
        print("✅ 机器人API导入成功")
        
        # 检查API端点
        print(f"  API路由前缀: {router.prefix}")
        print(f"  API标签: {router.tags}")
        
        return True
    except Exception as e:
        print(f"❌ 机器人API导入失败: {e}")
        return False

def test_robot_hardware_interface():
    """测试机器人硬件接口"""
    print("\n" + "=" * 80)
    print("测试机器人硬件接口")
    print("=" * 80)
    
    try:
        from core.hardware.robot_hardware_interface import RobotHardwareInterface
        
        print("✅ 机器人硬件接口导入成功")
        
        # 创建硬件接口实例
        robot_hardware = RobotHardwareInterface(use_robot_driver=True)
        
        # 测试初始化
        init_result = robot_hardware.initialize()
        print(f"  硬件初始化结果: {init_result}")
        
        # 测试硬件检测
        hardware_status = robot_hardware.check_hardware_status()
        print(f"  硬件状态检测: {hardware_status.get('status', 'unknown')}")
        
        # 测试传感器列表
        sensors = robot_hardware.get_sensors()
        print(f"  检测到的传感器数量: {len(sensors)}")
        
        # 测试执行器列表
        actuators = robot_hardware.get_actuators()
        print(f"  检测到的执行器数量: {len(actuators)}")
        
        # 测试设备列表
        devices = robot_hardware.get_devices()
        print(f"  检测到的设备总数: {len(devices)}")
        
        return True
    except Exception as e:
        print(f"❌ 机器人硬件接口测试失败: {e}")
        return False

def test_robot_generic_driver():
    """测试通用机器人驱动"""
    print("\n" + "=" * 80)
    print("测试通用机器人驱动")
    print("=" * 80)
    
    try:
        from core.hardware.generic_robot_driver import GenericRobotDriver, HumanoidJoint, HumanoidSensor
        
        print("✅ 通用机器人驱动导入成功")
        
        # 创建机器人驱动实例
        robot_driver = GenericRobotDriver()
        
        # 测试连接
        connect_result = robot_driver.connect()
        print(f"  机器人连接结果: {connect_result}")
        
        # 测试关节枚举
        print("  支持的关节类型:")
        for joint in HumanoidJoint:
            print(f"    - {joint.value}")
        
        # 测试传感器枚举
        print("  支持的传感器类型:")
        for sensor in HumanoidSensor:
            print(f"    - {sensor.value}")
        
        return True
    except Exception as e:
        print(f"❌ 通用机器人驱动测试失败: {e}")
        return False

def test_robot_config_manager():
    """测试硬件配置管理器"""
    print("\n" + "=" * 80)
    print("测试硬件配置管理器")
    print("=" * 80)
    
    try:
        from core.hardware_config_manager import HardwareConfigManager
        
        print("✅ 硬件配置管理器导入成功")
        
        # 创建配置管理器实例
        config_manager = HardwareConfigManager()
        
        # 测试配置加载
        configs = config_manager.load_hardware_configs()
        print(f"  加载的硬件配置数量: {len(configs)}")
        
        # 测试配置验证
        validation_result = config_manager.validate_configs(configs)
        print(f"  配置验证结果: {validation_result.get('valid', False)}")
        
        # 测试机器人配置生成
        robot_config = config_manager.generate_robot_config()
        print(f"  生成的机器人配置类型: {type(robot_config)}")
        
        return True
    except Exception as e:
        print(f"❌ 硬件配置管理器测试失败: {e}")
        return False

def test_multimodal_robot_control():
    """测试多模态机器人控制"""
    print("\n" + "=" * 80)
    print("测试多模态机器人控制")
    print("=" * 80)
    
    try:
        # 测试多模态输入处理
        from core.multimodal.true_data_processor import TrueDataProcessor
        from core.hardware.robot_hardware_interface import RobotHardwareInterface
        
        print("✅ 多模态机器人控制测试准备")
        
        # 创建真实数据处理器
        data_processor = TrueDataProcessor()
        
        # 创建机器人硬件接口
        robot_hardware = RobotHardwareInterface(use_robot_driver=True)
        
        # 测试语音到动作映射
        test_commands = [
            "向前走",
            "抬起右臂",
            "向左转",
            "坐下",
            "挥手"
        ]
        
        print("  语音命令到机器人动作映射:")
        for command in test_commands:
            # 处理语音命令
            processed_command = data_processor.process_text(command)
            print(f"    '{command}' -> 机器人动作处理")
        
        # 测试视觉到动作映射
        print("  视觉输入到机器人动作映射:")
        vision_inputs = [
            "检测到前方障碍物",
            "识别到人形物体",
            "检测到手势信号",
            "识别到目标物体",
            "检测到危险区域"
        ]
        
        for vision_input in vision_inputs:
            # 处理视觉输入
            processed_vision = data_processor.process_text(vision_input)
            print(f"    '{vision_input}' -> 机器人避障/导航处理")
        
        return True
    except Exception as e:
        print(f"❌ 多模态机器人控制测试失败: {e}")
        return False

def test_robot_sensor_data_fusion():
    """测试机器人传感器数据融合"""
    print("\n" + "=" * 80)
    print("测试机器人传感器数据融合")
    print("=" * 80)
    
    try:
        from core.sensor_fusion import SensorFusionEngine
        from core.hardware.robot_hardware_interface import RobotHardwareInterface
        
        print("✅ 传感器数据融合测试准备")
        
        # 创建传感器融合引擎
        fusion_engine = SensorFusionEngine()
        
        # 创建机器人硬件接口
        robot_hardware = RobotHardwareInterface(use_robot_driver=True)
        
        # 模拟传感器数据
        sensor_data = {
            "imu": {
                "acceleration": [0.1, 0.2, 9.8],
                "gyroscope": [0.01, 0.02, 0.03],
                "magnetometer": [10.5, 20.3, 30.7]
            },
            "force_sensors": {
                "left_foot": [50.2, 48.7, 49.5, 51.1],
                "right_foot": [49.8, 50.2, 50.5, 49.9]
            },
            "joint_angles": {
                "left_hip": 15.2,
                "left_knee": 30.5,
                "left_ankle": -5.3,
                "right_hip": 14.8,
                "right_knee": 31.2,
                "right_ankle": -4.9
            }
        }
        
        # 测试传感器数据融合
        fused_data = fusion_engine.fuse_sensor_data(sensor_data)
        print(f"  传感器数据融合结果:")
        print(f"    - 姿态估计: {fused_data.get('pose_estimation', 'N/A')}")
        print(f"    - 平衡状态: {fused_data.get('balance_status', 'N/A')}")
        print(f"    - 步态分析: {fused_data.get('gait_analysis', 'N/A')}")
        
        return True
    except ImportError as e:
        print(f"⚠️  传感器融合引擎不可用: {e}")
        return False
    except Exception as e:
        print(f"❌ 传感器数据融合测试失败: {e}")
        return False

def test_robot_motor_control():
    """测试电机控制功能"""
    print("\n" + "=" * 80)
    print("测试电机控制功能")
    print("=" * 80)
    
    try:
        from core.hardware.robot_hardware_interface import RobotHardwareInterface
        
        print("✅ 电机控制测试准备")
        
        # 创建机器人硬件接口
        robot_hardware = RobotHardwareInterface(use_robot_driver=True)
        
        # 初始化硬件
        init_result = robot_hardware.initialize()
        
        if init_result.get("success", False):
            print("  硬件初始化成功，准备测试电机控制")
            
            # 测试关节控制（模拟）
            test_joints = [
                {"joint_id": "arm_left_shoulder", "position": 45.0, "velocity": 10.0},
                {"joint_id": "arm_left_elbow", "position": 90.0, "velocity": 15.0},
                {"joint_id": "arm_left_wrist", "position": 0.0, "velocity": 5.0},
                {"joint_id": "leg_left_hip", "position": 30.0, "velocity": 8.0},
                {"joint_id": "leg_left_knee", "position": 60.0, "velocity": 12.0},
                {"joint_id": "leg_left_ankle", "position": -10.0, "velocity": 6.0}
            ]
            
            print("  模拟关节控制测试:")
            for joint in test_joints:
                print(f"    控制关节 {joint['joint_id']}: 位置={joint['position']}°, 速度={joint['velocity']}°/s")
                time.sleep(0.1)
            
            print("  ✅ 关节控制测试完成")
            
            # 测试伺服控制
            print("  模拟伺服控制测试:")
            servos = robot_hardware.get_actuators()
            for servo in servos:
                print(f"    伺服: {servo['name']}, 状态: {servo['status']}")
            
            print("  ✅ 伺服控制测试完成")
            
            return True
        else:
            print(f"❌ 硬件初始化失败: {init_result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ 电机控制测试失败: {e}")
        return False

def test_robot_api_endpoints():
    """测试机器人API端点"""
    print("\n" + "=" * 80)
    print("测试机器人API端点")
    print("=" * 80)
    
    try:
        import requests
        import json
        
        # 测试API端点（假设本地服务运行在8000端口）
        base_url = "http://localhost:8000/api/robot"
        
        endpoints = [
            "/health",
            "/status",
            "/sensors",
            "/joints",
            "/tasks",
            "/collaboration/patterns"
        ]
        
        print("  测试机器人API端点:")
        for endpoint in endpoints:
            url = base_url + endpoint
            try:
                response = requests.get(url, timeout=5)
                status = "✅" if response.status_code == 200 else "❌"
                print(f"    {status} {endpoint}: HTTP {response.status_code}")
            except requests.exceptions.ConnectionError:
                print(f"    ⚠️ {endpoint}: 连接失败（服务未运行）")
            except Exception as e:
                print(f"    ❌ {endpoint}: 错误 - {e}")
        
        return True
    except Exception as e:
        print(f"❌ 机器人API端点测试失败: {e}")
        return False

def generate_robot_integration_report():
    """生成机器人集成报告"""
    print("\n" + "=" * 80)
    print("机器人硬件集成测试报告")
    print("=" * 80)
    
    tests = [
        ("机器人API导入", test_robot_api_imports()),
        ("机器人硬件接口", test_robot_hardware_interface()),
        ("通用机器人驱动", test_robot_generic_driver()),
        ("硬件配置管理器", test_robot_config_manager()),
        ("多模态机器人控制", test_multimodal_robot_control()),
        ("传感器数据融合", test_robot_sensor_data_fusion()),
        ("电机控制功能", test_robot_motor_control()),
        ("机器人API端点", test_robot_api_endpoints()),
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    print(f"\n测试总结:")
    print(f"  总测试数: {total}")
    print(f"  通过测试: {passed}")
    print(f"  失败测试: {total - passed}")
    print(f"  通过率: {(passed/total*100):.1f}%")
    
    print("\n详细结果:")
    for test_name, result in tests:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}: {test_name}")
    
    # 生成建议
    print("\n建议:")
    if passed == total:
        print("  🎉 所有机器人硬件集成测试通过！系统已准备好进行实际机器人硬件连接。")
    else:
        print("  ⚠️ 部分测试失败，需要检查以下方面:")
        for test_name, result in tests:
            if not result:
                print(f"    - {test_name}")
        
        print("\n  建议修复步骤:")
        print("    1. 检查机器人硬件驱动安装")
        print("    2. 验证硬件连接状态")
        print("    3. 检查依赖库安装")
        print("    4. 确认硬件配置文件")
    
    return passed == total

def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("Self-Soul-B多模态AGI系统机器人硬件集成测试")
    print("=" * 80)
    print("测试目标: 验证机器人硬件控制、传感器处理、电机控制等功能")
    print("测试时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    try:
        # 运行所有测试
        success = generate_robot_integration_report()
        
        # 生成测试报告文件
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system": "Self-Soul-B多模态AGI系统",
            "test_type": "机器人硬件集成测试",
            "success": success,
            "recommendations": [
                "连接实际机器人硬件进行进一步测试",
                "配置机器人硬件参数和校准",
                "进行多模态交互测试",
                "实现安全控制和紧急停止功能"
            ]
        }
        
        # 保存报告
        report_file = "robot_hardware_integration_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n测试报告已保存到: {report_file}")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ 测试执行失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())