"""
增强的机器人API简单测试

测试增强的机器人API路由器和集成功能，无需启动服务器。
"""

import os
import sys

# 设置环境变量以启用模拟模式
os.environ['ENVIRONMENT'] = 'development'
os.environ['ALLOW_ROBOT_SIMULATION'] = 'true'
os.environ['ROBOT_HARDWARE_TEST_MODE'] = 'true'

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_enhanced_api_router():
    """测试增强的机器人API路由器"""
    print("=" * 80)
    print("测试增强的机器人API路由器")
    print("=" * 80)
    
    try:
        from core.robot_api_enhanced import router, initialize_enhanced_robot_api
        
        print("✅ 增强的机器人API路由器导入成功")
        
        # 初始化API
        print("初始化增强的机器人API...")
        initialized = initialize_enhanced_robot_api()
        print(f"初始化结果: {initialized}")
        
        # 检查路由器端点
        routes = [route for route in router.routes]
        print(f"路由器包含 {len(routes)} 个端点")
        
        # 显示所有端点
        print("所有端点:")
        for i, route in enumerate(routes):
            if hasattr(route, 'path'):
                print(f"  {i+1:2d}. {route.path}")
        
        # 检查关键端点是否存在
        required_endpoints = [
            '/api/robot/enhanced/status',
            '/api/robot/enhanced/motion/command',
            '/api/robot/enhanced/fusion/status',
            '/api/robot/enhanced/fusion/start',
            '/api/robot/enhanced/fusion/stop',
            '/api/robot/enhanced/fusion/process',
            '/api/robot/enhanced/motion/capabilities',
            '/api/robot/enhanced/emergency/stop',
            '/api/robot/enhanced/multimodal/test',
            '/api/robot/enhanced/hardware/simulated',
            '/api/robot/enhanced/test/echo',
            '/api/robot/enhanced/test/integration'
        ]
        
        endpoint_paths = []
        for route in routes:
            if hasattr(route, 'path'):
                endpoint_paths.append(route.path)
        
        missing_endpoints = []
        for required in required_endpoints:
            if required not in endpoint_paths:
                missing_endpoints.append(required)
        
        if missing_endpoints:
            print(f"❌ 缺少 {len(missing_endpoints)} 个必需的端点:")
            for missing in missing_endpoints:
                print(f"  - {missing}")
            return False
        else:
            print("✅ 所有必需的端点都存在")
            return True
            
    except Exception as e:
        print(f"❌ 增强的机器人API路由器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_motion_controller_availability():
    """测试运动控制器可用性"""
    print("\n" + "=" * 80)
    print("测试运动控制器可用性")
    print("=" * 80)
    
    try:
        from core.robot_motion_control import get_motion_controller
        
        print("✅ 运动控制器导入成功")
        
        # 获取运动控制器实例
        motion_controller = get_motion_controller()
        
        # 检查运动控制器功能
        voice_commands = list(motion_controller.mapping_rules["voice_to_action"].keys())
        print(f"运动控制器支持 {len(voice_commands)} 种语音命令映射: {', '.join(voice_commands[:5])}...")
        
        # 测试简单的语音命令处理
        test_command = "前进三步"
        motion_commands = motion_controller.process_multimodal_input(voice_input=test_command)
        print(f"语音命令 '{test_command}' -> {len(motion_commands)} 个运动命令")
        
        return True
        
    except Exception as e:
        print(f"❌ 运动控制器可用性测试失败: {e}")
        return False

def test_sensor_fusion_availability():
    """测试传感器融合引擎可用性"""
    print("\n" + "=" * 80)
    print("测试传感器融合引擎可用性")
    print("=" * 80)
    
    try:
        from core.sensor_fusion import get_fusion_engine
        
        print("✅ 传感器融合引擎导入成功")
        
        # 获取融合引擎实例
        fusion_engine = get_fusion_engine()
        
        # 检查融合引擎状态
        fusion_state = fusion_engine.get_fusion_state()
        print(f"传感器融合引擎状态: {fusion_state.value}")
        
        # 检查传感器缓冲区
        sensor_buffers = fusion_engine.sensor_buffers
        total_buffered = sum(len(buffer) for buffer in sensor_buffers.values())
        print(f"传感器缓冲区总大小: {total_buffered}")
        
        return True
        
    except Exception as e:
        print(f"❌ 传感器融合引擎可用性测试失败: {e}")
        return False

def test_robot_hardware_interface_availability():
    """测试机器人硬件接口可用性"""
    print("\n" + "=" * 80)
    print("测试机器人硬件接口可用性")
    print("=" * 80)
    
    try:
        from core.hardware.robot_hardware_interface import RobotHardwareInterface
        
        print("✅ 机器人硬件接口导入成功")
        
        # 检查环境变量
        environment = os.environ.get('ENVIRONMENT', 'production')
        allow_simulation = os.environ.get('ALLOW_ROBOT_SIMULATION', 'false').lower() == 'true'
        test_mode = os.environ.get('ROBOT_HARDWARE_TEST_MODE', 'false').lower() == 'true'
        
        print(f"环境配置:")
        print(f"  ENVIRONMENT: {environment}")
        print(f"  ALLOW_ROBOT_SIMULATION: {allow_simulation}")
        print(f"  ROBOT_HARDWARE_TEST_MODE: {test_mode}")
        
        return True
        
    except Exception as e:
        print(f"❌ 机器人硬件接口可用性测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("Self-Soul-B多模态AGI系统 - 增强的机器人API简单测试")
    print("=" * 80)
    print("测试目标: 验证增强的机器人API和相关模块的可用性")
    print("测试模式: 模块级测试（无需启动服务器）")
    print("环境设置: 启用模拟模式进行测试")
    print("=" * 80)
    
    tests = [
        ("增强的机器人API路由器", test_enhanced_api_router),
        ("运动控制器可用性", test_motion_controller_availability),
        ("传感器融合引擎可用性", test_sensor_fusion_availability),
        ("机器人硬件接口可用性", test_robot_hardware_interface_availability),
    ]
    
    passed = 0
    total = len(tests)
    
    print(f"运行 {total} 个测试...")
    print()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
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
    
    if passed == total:
        print("\n✅ 所有增强的机器人API测试通过！")
        print("\n下一步:")
        print("  1. 启动渐进式服务器: python progressive_server.py")
        print("  2. 运行完整的API测试: python tests/test_enhanced_robot_api.py")
        print("  3. 开发多模态机器人控制应用")
        print("  4. 实现传感器数据实时处理管道")
        return 0
    else:
        print("\n⚠️  部分测试失败，需要检查。")
        print("\n建议:")
        print("  1. 检查模块导入路径")
        print("  2. 验证环境变量设置")
        print("  3. 检查依赖模块是否正确安装")
        return 1

if __name__ == "__main__":
    sys.exit(main())