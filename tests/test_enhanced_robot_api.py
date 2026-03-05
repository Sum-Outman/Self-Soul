"""
增强的机器人API测试

测试机器人硬件集成相关的API端点，包括：
1. 增强机器人状态查询
2. 传感器融合API
3. 运动控制API
4. 多模态集成测试
"""

import sys
import os
import time
import json
import requests
import asyncio
from typing import Dict, Any, List

# 设置环境变量以启用模拟模式
os.environ['ENVIRONMENT'] = 'development'
os.environ['ALLOW_ROBOT_SIMULATION'] = 'true'
os.environ['ROBOT_HARDWARE_TEST_MODE'] = 'true'

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_enhanced_robot_status(base_url: str = "http://localhost:8000"):
    """测试增强机器人状态端点"""
    print("=" * 80)
    print("测试增强机器人状态端点")
    print("=" * 80)
    
    try:
        url = f"{base_url}/api/robot/enhanced/status"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 状态端点测试成功 (状态码: {response.status_code})")
            print(f"   基础状态: {data.get('base', {}).get('status', 'unknown')}")
            print(f"   增强状态: {data.get('enhanced', {}).get('api_initialized', False)}")
            print(f"   运动控制器可用: {data.get('enhanced', {}).get('modules', {}).get('motion_control', False)}")
            print(f"   传感器融合可用: {data.get('enhanced', {}).get('modules', {}).get('sensor_fusion', False)}")
            return True
        else:
            print(f"❌ 状态端点测试失败 (状态码: {response.status_code})")
            print(f"   响应内容: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ 状态端点测试异常: {e}")
        return False

def test_robot_motion_capabilities(base_url: str = "http://localhost:8000"):
    """测试机器人运动能力端点"""
    print("\n" + "=" * 80)
    print("测试机器人运动能力端点")
    print("=" * 80)
    
    try:
        url = f"{base_url}/api/robot/enhanced/motion/capabilities"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 运动能力端点测试成功 (状态码: {response.status_code})")
            
            if 'motion_types' in data:
                print(f"   支持的运动类型: {len(data['motion_types'])} 种")
                print(f"   示例: {data['motion_types'][:5]}")
            
            if 'control_modes' in data:
                print(f"   控制模式: {data['control_modes']}")
            
            return True
        else:
            print(f"❌ 运动能力端点测试失败 (状态码: {response.status_code})")
            return False
            
    except Exception as e:
        print(f"❌ 运动能力端点测试异常: {e}")
        return False

def test_sensor_fusion_api(base_url: str = "http://localhost:8000"):
    """测试传感器融合API"""
    print("\n" + "=" * 80)
    print("测试传感器融合API")
    print("=" * 80)
    
    try:
        # 测试融合状态
        status_url = f"{base_url}/api/robot/enhanced/fusion/status"
        response = requests.get(status_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 传感器融合状态测试成功 (状态码: {response.status_code})")
            print(f"   融合状态: {data.get('state', 'unknown')}")
            print(f"   融合引擎可用: {data.get('available', False)}")
            print(f"   融合引擎已启动: {data.get('started', False)}")
            
            # 如果未启动，尝试启动
            if data.get('available', False) and not data.get('started', False):
                print("   尝试启动传感器融合引擎...")
                start_url = f"{base_url}/api/robot/enhanced/fusion/start"
                start_response = requests.post(start_url, timeout=10)
                if start_response.status_code == 200:
                    start_data = start_response.json()
                    print(f"   启动结果: {start_data.get('message', 'unknown')}")
            
            # 测试传感器数据处理
            process_url = f"{base_url}/api/robot/enhanced/fusion/process"
            sensor_data = {
                "imu": {
                    "acceleration": [0.05, 0.1, 9.81],
                    "gyroscope": [0.005, 0.01, 0.002]
                },
                "force_sensors": {
                    "left_foot": [25.2, 24.7, 26.5, 25.1],
                    "right_foot": [24.8, 25.2, 24.5, 25.9]
                }
            }
            
            process_response = requests.post(process_url, json=sensor_data, timeout=10)
            if process_response.status_code == 200:
                process_data = process_response.json()
                print(f"✅ 传感器数据处理测试成功")
                print(f"   融合传感器数: {len(process_data.get('fused_sensors', []))}")
                print(f"   融合周期时间: {process_data.get('cycle_time', 0)*1000:.2f}ms")
            
            return True
        else:
            print(f"❌ 传感器融合状态测试失败 (状态码: {response.status_code})")
            return False
            
    except Exception as e:
        print(f"❌ 传感器融合API测试异常: {e}")
        return False

def test_motion_command_api(base_url: str = "http://localhost:8000"):
    """测试运动命令API"""
    print("\n" + "=" * 80)
    print("测试运动命令API")
    print("=" * 80)
    
    try:
        url = f"{base_url}/api/robot/enhanced/motion/command"
        
        # 测试语音命令
        voice_command = {
            "command_type": "voice",
            "command_data": {"text": "前进三步"},
            "priority": 5,
            "async_execution": True
        }
        
        response = requests.post(url, json=voice_command, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 运动命令API测试成功 (状态码: {response.status_code})")
            print(f"   命令执行成功: {data.get('success', False)}")
            print(f"   生成命令数: {data.get('commands_generated', 0)}")
            print(f"   执行命令数: {data.get('commands_executed', 0)}")
            print(f"   异步模式: {data.get('async_mode', False)}")
            return True
        else:
            print(f"❌ 运动命令API测试失败 (状态码: {response.status_code})")
            print(f"   响应内容: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ 运动命令API测试异常: {e}")
        return False

def test_multimodal_integration(base_url: str = "http://localhost:8000"):
    """测试多模态集成"""
    print("\n" + "=" * 80)
    print("测试多模态集成")
    print("=" * 80)
    
    try:
        url = f"{base_url}/api/robot/enhanced/multimodal/test"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 多模态集成测试成功 (状态码: {response.status_code})")
            
            modules = data.get('modules', {})
            print(f"   模块状态:")
            for module_name, available in modules.items():
                status = "✅ 可用" if available else "❌ 不可用"
                print(f"     - {module_name}: {status}")
            
            integration_status = data.get('integration_status', 'unknown')
            print(f"   集成状态: {integration_status}")
            
            return True
        else:
            print(f"❌ 多模态集成测试失败 (状态码: {response.status_code})")
            return False
            
    except Exception as e:
        print(f"❌ 多模态集成测试异常: {e}")
        return False

def test_simulated_hardware_info(base_url: str = "http://localhost:8000"):
    """测试模拟硬件信息"""
    print("\n" + "=" * 80)
    print("测试模拟硬件信息")
    print("=" * 80)
    
    try:
        url = f"{base_url}/api/robot/enhanced/hardware/simulated"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 模拟硬件信息测试成功 (状态码: {response.status_code})")
            
            print(f"   环境: {data.get('environment', 'unknown')}")
            print(f"   模拟允许: {data.get('simulation_allowed', False)}")
            print(f"   测试模式: {data.get('test_mode', False)}")
            print(f"   模拟可用: {data.get('simulation_available', False)}")
            
            simulated_sensors = data.get('simulated_sensors', [])
            print(f"   模拟传感器数: {len(simulated_sensors)}")
            
            simulated_servos = data.get('simulated_servos', [])
            print(f"   模拟伺服数: {len(simulated_servos)}")
            
            return True
        else:
            print(f"❌ 模拟硬件信息测试失败 (状态码: {response.status_code})")
            return False
            
    except Exception as e:
        print(f"❌ 模拟硬件信息测试异常: {e}")
        return False

def test_echo_endpoint(base_url: str = "http://localhost:8000"):
    """测试连通性端点"""
    print("\n" + "=" * 80)
    print("测试连通性端点")
    print("=" * 80)
    
    try:
        url = f"{base_url}/api/robot/enhanced/test/echo?message=机器人API测试"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 连通性端点测试成功 (状态码: {response.status_code})")
            print(f"   消息: {data.get('message', '无消息')}")
            print(f"   端点: {data.get('endpoint', '未知端点')}")
            return True
        else:
            print(f"❌ 连通性端点测试失败 (状态码: {response.status_code})")
            return False
            
    except Exception as e:
        print(f"❌ 连通性端点测试异常: {e}")
        return False

def test_emergency_stop(base_url: str = "http://localhost:8000"):
    """测试紧急停止端点"""
    print("\n" + "=" * 80)
    print("测试紧急停止端点")
    print("=" * 80)
    
    try:
        url = f"{base_url}/api/robot/enhanced/emergency/stop"
        response = requests.post(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 紧急停止端点测试成功 (状态码: {response.status_code})")
            print(f"   停止方法: {data.get('method', 'unknown')}")
            print(f"   停止成功: {data.get('success', False)}")
            return True
        else:
            print(f"❌ 紧急停止端点测试失败 (状态码: {response.status_code})")
            return False
            
    except Exception as e:
        print(f"❌ 紧急停止端点测试异常: {e}")
        return False

def generate_api_test_report(base_url: str = "http://localhost:8000"):
    """生成API测试报告"""
    print("\n" + "=" * 80)
    print("增强的机器人API测试报告")
    print("=" * 80)
    
    tests = [
        ("增强机器人状态", test_enhanced_robot_status, base_url),
        ("运动能力查询", test_robot_motion_capabilities, base_url),
        ("传感器融合API", test_sensor_fusion_api, base_url),
        ("运动命令API", test_motion_command_api, base_url),
        ("多模态集成测试", test_multimodal_integration, base_url),
        ("模拟硬件信息", test_simulated_hardware_info, base_url),
        ("连通性端点", test_echo_endpoint, base_url),
        ("紧急停止", test_emergency_stop, base_url),
    ]
    
    passed = 0
    total = len(tests)
    
    print(f"开始测试 {total} 个API端点...")
    print(f"基础URL: {base_url}")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    for test_name, test_func, url in tests:
        try:
            result = test_func(url)
            if result:
                passed += 1
                print(f"  ✅ {test_name}: 通过")
            else:
                print(f"  ❌ {test_name}: 失败")
        except Exception as e:
            print(f"  ❌ {test_name}: 异常 - {e}")
    
    print("\n" + "=" * 80)
    print("测试总结:")
    print(f"  总测试数: {total}")
    print(f"  通过测试: {passed}")
    print(f"  失败测试: {total - passed}")
    print(f"  通过率: {(passed/total*100):.1f}%")
    
    # 生成建议
    print("\n建议:")
    if passed == total:
        print("  🎉 所有增强的机器人API测试通过！系统已准备好进行多模态机器人控制开发。")
        print("\n  下一步:")
        print("    1. 启动渐进式服务器进行完整测试")
        print("    2. 集成机器人硬件接口（真实硬件或模拟）")
        print("    3. 开发多模态机器人控制应用")
        print("    4. 实现传感器数据实时处理管道")
    elif passed > total * 0.7:
        print("  ⚠️  大部分API测试通过，部分功能需要检查。")
        print("\n  需要关注的方面:")
        print("    1. 确保机器人硬件接口模块正确导入")
        print("    2. 检查环境变量设置（ENVIRONMENT, ALLOW_ROBOT_SIMULATION等）")
        print("    3. 验证传感器融合和运动控制模块的依赖")
    else:
        print("  ❌ API测试通过率较低，需要详细检查。")
        print("\n  排查步骤:")
        print("    1. 检查服务器是否正在运行")
        print("    2. 验证API端点URL是否正确")
        print("    3. 检查模块导入和依赖问题")
        print("    4. 查看服务器日志以了解详细错误")
    
    return passed == total

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Self-Soul-B多模态AGI系统 - 增强的机器人API测试")
    print("=" * 80)
    print("测试目标: 验证增强的机器人API端点功能")
    print("测试模式: API端点测试（需要运行服务器）")
    print("环境设置: 启用模拟模式进行测试")
    print("=" * 80)
    
    # 检查服务器是否可访问
    base_url = "http://localhost:8000"
    
    try:
        # 尝试连接服务器
        print("检查服务器连接...")
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code == 200:
            print(f"✅ 服务器连接成功: {base_url}")
        else:
            print(f"⚠️  服务器响应异常: 状态码 {health_response.status_code}")
            print("   服务器可能未启动或配置不同端口")
            print("   使用默认端口8000，如需更改请修改base_url参数")
            proceed = input("是否继续测试？(y/n): ")
            if proceed.lower() != 'y':
                print("测试中止")
                return 1
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到服务器: {base_url}")
        print("请确保渐进式服务器正在运行：")
        print("  python progressive_server.py")
        print("\n或使用正确的URL：")
        print("  python tests/test_enhanced_robot_api.py --url http://localhost:5175")
        return 1
    
    # 运行测试
    success = generate_api_test_report(base_url)
    
    # 保存测试报告
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": base_url,
        "environment": os.environ.get('ENVIRONMENT', 'unknown'),
        "simulation_allowed": os.environ.get('ALLOW_ROBOT_SIMULATION', 'false'),
        "test_mode": os.environ.get('ROBOT_HARDWARE_TEST_MODE', 'false'),
        "system": "Self-Soul-B多模态AGI系统",
        "test_type": "增强的机器人API测试",
        "success": success,
        "recommendations": [
            "启动渐进式服务器进行完整功能测试",
            "配置机器人硬件接口（真实或模拟）",
            "开发多模态机器人控制界面",
            "集成传感器数据处理管道",
            "实现实时机器人状态监控"
        ]
    }
    
    # 保存报告
    report_file = "enhanced_robot_api_test_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n测试报告已保存到: {report_file}")
    
    return 0 if success else 1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="增强的机器人API测试")
    parser.add_argument("--url", type=str, default="http://localhost:8000", 
                       help="服务器基础URL (默认: http://localhost:8000)")
    parser.add_argument("--no-check", action="store_true", 
                       help="跳过服务器连接检查")
    
    args = parser.parse_args()
    
    if args.no_check:
        # 直接运行测试，不检查服务器连接
        success = generate_api_test_report(args.url)
        sys.exit(0 if success else 1)
    else:
        sys.exit(main())