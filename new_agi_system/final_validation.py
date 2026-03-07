"""
最终验证脚本

验证所有功能模块都可以正确导入，并且系统可以正常初始化。
"""

import sys
import os
import asyncio

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


async def validate_all_modules():
    """验证所有模块"""
    print("=" * 60)
    print("最终验证 - 所有功能模块")
    print("=" * 60)
    
    all_valid = True
    
    # 1. 验证核心认知模块
    print("\n1. 验证核心认知模块...")
    try:
        from cognitive.architecture import UnifiedCognitiveArchitecture
        print("  ✓ UnifiedCognitiveArchitecture")
        
        from cognitive.representation import UnifiedRepresentationSpace
        print("  ✓ UnifiedRepresentationSpace")
        
        from neural.communication import NeuralCommunication
        print("  ✓ NeuralCommunication")
        
        # 创建架构实例
        agi = UnifiedCognitiveArchitecture()
        print("  ✓ 统一认知架构实例化成功")
        
    except Exception as e:
        print(f"  ✗ 核心认知模块验证失败: {e}")
        all_valid = False
    
    # 2. 验证训练功能
    print("\n2. 验证训练功能...")
    try:
        from training.trainer import TrainingManager
        print("  ✓ TrainingManager")
        
        # 检查API
        from api.training import router as training_router
        print("  ✓ 训练API")
        
    except Exception as e:
        print(f"  ✗ 训练功能验证失败: {e}")
        all_valid = False
    
    # 3. 验证演化功能
    print("\n3. 验证演化功能...")
    try:
        from cognitive.evolution import AutonomousEvolutionSystem
        print("  ✓ AutonomousEvolutionSystem")
        
        from api.evolution import router as evolution_router
        print("  ✓ 演化API")
        
    except Exception as e:
        print(f"  ✗ 演化功能验证失败: {e}")
        all_valid = False
    
    # 4. 验证自我认知功能
    print("\n4. 验证自我认知功能...")
    try:
        from cognitive.self_cognition import SelfCognitionSystem
        print("  ✓ SelfCognitionSystem")
        
        from api.self_cognition import router as self_cognition_router
        print("  ✓ 自我认知API")
        
    except Exception as e:
        print(f"  ✗ 自我认知功能验证失败: {e}")
        all_valid = False
    
    # 5. 验证控制功能
    print("\n5. 验证控制功能...")
    try:
        from control.motion_control import MotionControlSystem
        print("  ✓ MotionControlSystem")
        
        from control.hardware_interface import HardwareControlSystem
        print("  ✓ HardwareControlSystem")
        
        from control.sensor_integration import SensorIntegrationSystem
        print("  ✓ SensorIntegrationSystem")
        
        from control.motor_control import MotorControlSystem
        print("  ✓ MotorControlSystem")
        
        from api.control import router as control_router
        print("  ✓ 控制API")
        
    except Exception as e:
        print(f"  ✗ 控制功能验证失败: {e}")
        all_valid = False
    
    # 6. 验证自主意识功能
    print("\n6. 验证自主意识功能...")
    try:
        from cognitive.autonomy import AutonomousSystem
        print("  ✓ AutonomousSystem")
        
        from api.autonomy import router as autonomy_router
        print("  ✓ 自主意识API")
        
    except Exception as e:
        print(f"  ✗ 自主意识功能验证失败: {e}")
        all_valid = False
    
    # 7. 验证人形机器人AGI功能
    print("\n7. 验证人形机器人AGI功能...")
    try:
        from humanoid.balance_control import BalanceControlSystem
        print("  ✓ BalanceControlSystem")
        
        from humanoid.walking_gait import WalkingGaitSystem
        print("  ✓ WalkingGaitSystem")
        
        from humanoid.humanoid_agi import HumanoidAGISystem
        print("  ✓ HumanoidAGISystem")
        
        from api.humanoid import router as humanoid_router
        print("  ✓ 人形机器人API")
        
    except Exception as e:
        print(f"  ✗ 人形机器人AGI功能验证失败: {e}")
        all_valid = False
    
    # 8. 验证API服务器
    print("\n8. 验证API服务器...")
    try:
        from api.server import app
        print("  ✓ FastAPI应用")
        
        # 检查路由
        routes = [route.path for route in app.routes]
        print(f"  ✓ 总路由数: {len(routes)}")
        
        # 检查主要API端点
        required_prefixes = [
            '/api/training',
            '/api/evolution',
            '/api/self_cognition',
            '/api/control',
            '/api/autonomy',
            '/api/humanoid'
        ]
        
        for prefix in required_prefixes:
            matching_routes = [r for r in routes if r.startswith(prefix)]
            if matching_routes:
                print(f"  ✓ {prefix} API: {len(matching_routes)}个端点")
            else:
                print(f"  ✗ {prefix} API: 无端点")
                all_valid = False
        
    except Exception as e:
        print(f"  ✗ API服务器验证失败: {e}")
        all_valid = False
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✓ 所有功能模块验证成功！")
        print("系统已准备好接管原有Self-Soul系统的全部功能。")
    else:
        print("✗ 部分功能模块验证失败！")
        print("请检查错误信息并修复问题。")
    
    print("=" * 60)
    return all_valid


async def validate_system_initialization():
    """验证系统初始化"""
    print("\n" + "=" * 60)
    print("系统初始化验证")
    print("=" * 60)
    
    try:
        from cognitive.architecture import UnifiedCognitiveArchitecture
        
        # 创建架构实例
        agi = UnifiedCognitiveArchitecture()
        print("1. 统一认知架构实例化成功")
        
        # 初始化组件
        agi.initialize_components()
        print("2. 所有认知组件初始化成功")
        
        # 检查主要组件
        components_to_check = [
            ('perception', '感知系统'),
            ('memory', '记忆系统'),
            ('reasoning', '推理系统'),
            ('learning', '学习系统'),
            ('evolution', '演化系统'),
            ('self_cognition', '自我认知系统'),
            ('autonomy', '自主意识系统')
        ]
        
        for attr_name, display_name in components_to_check:
            component = getattr(agi, attr_name, None)
            if component is not None:
                print(f"3. {display_name}可用")
            else:
                print(f"3. {display_name}不可用")
        
        # 检查控制组件
        if hasattr(agi, 'motion_control') and agi.motion_control is not None:
            print("4. 控制组件可用")
        else:
            print("4. 控制组件不可用")
        
        # 检查人形机器人组件
        if hasattr(agi, 'humanoid_agi') and agi.humanoid_agi is not None:
            print("5. 人形机器人AGI组件可用")
        else:
            print("5. 人形机器人AGI组件不可用")
        
        print("\n✓ 系统初始化验证成功！")
        return True
        
    except Exception as e:
        print(f"\n✗ 系统初始化验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主验证函数"""
    print("开始最终验证...")
    
    # 验证所有模块
    modules_valid = await validate_all_modules()
    
    # 验证系统初始化
    initialization_valid = await validate_system_initialization()
    
    # 总结
    print("\n" + "=" * 60)
    print("最终验证结果")
    print("=" * 60)
    
    if modules_valid and initialization_valid:
        print("✓ 所有验证通过！")
        print("\n系统状态: 已准备好接管原有Self-Soul系统")
        print("API服务器端口: 9000")
        print("API文档地址: http://127.0.0.1:9000/docs")
        return True
    else:
        print("✗ 验证失败！")
        print("\n系统状态: 需要修复问题")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)