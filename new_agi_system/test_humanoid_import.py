"""
测试人形机器人AGI模块导入

验证人形机器人AGI系统的所有模块可以正确导入，并且与统一认知架构集成。
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_humanoid_imports():
    """测试人形机器人模块导入"""
    print("=" * 60)
    print("人形机器人模块导入测试")
    print("=" * 60)
    
    try:
        # 测试平衡控制模块导入
        from src.humanoid.balance_control import BalanceControlSystem
        print("✓ BalanceControlSystem 导入成功")
        
        # 测试行走步态模块导入
        from src.humanoid.walking_gait import WalkingGaitSystem
        print("✓ WalkingGaitSystem 导入成功")
        
        # 测试人形机器人AGI模块导入
        from src.humanoid.humanoid_agi import HumanoidAGISystem, HumanoidTaskType
        print("✓ HumanoidAGISystem 导入成功")
        print("✓ HumanoidTaskType 导入成功")
        
        # 测试架构导入
        from src.cognitive.architecture import UnifiedCognitiveArchitecture
        print("✓ UnifiedCognitiveArchitecture 导入成功")
        
        # 测试API导入
        from src.api.humanoid import router as humanoid_router
        print("✓ 人形机器人API路由导入成功")
        
        print("\n所有人形机器人模块导入成功！")
        return True
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_architecture_integration():
    """测试架构集成"""
    print("\n" + "=" * 60)
    print("人形机器人架构集成测试")
    print("=" * 60)
    
    try:
        # 创建统一认知架构实例
        from src.cognitive.architecture import UnifiedCognitiveArchitecture
        
        agi = UnifiedCognitiveArchitecture()
        print("✓ UnifiedCognitiveArchitecture 实例化成功")
        
        # 检查人形机器人AGI属性
        humanoid_system = agi.humanoid_agi
        if humanoid_system is None:
            print("✗ 人形机器人AGI系统不可用")
            return False
        
        print("✓ 人形机器人AGI系统可用")
        
        # 检查子系统
        if hasattr(humanoid_system, 'balance_control'):
            print("✓ 平衡控制系统可用")
        
        if hasattr(humanoid_system, 'walking_gait'):
            print("✓ 行走步态系统可用")
        
        # 检查组件注册
        if hasattr(agi, 'component_registry'):
            if 'humanoid_agi' in agi.component_registry:
                print("✓ 人形机器人AGI组件已注册")
        
        print("\n人形机器人AGI系统已成功集成到统一认知架构中！")
        return True
        
    except Exception as e:
        print(f"✗ 架构集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_integration():
    """测试API集成"""
    print("\n" + "=" * 60)
    print("人形机器人API集成测试")
    print("=" * 60)
    
    try:
        # 测试FastAPI应用可以导入
        from src.api.server import app
        
        # 检查应用的路由是否包含人形机器人路由
        routes = [route.path for route in app.routes]
        humanoid_routes = [r for r in routes if '/api/humanoid' in r]
        
        if humanoid_routes:
            print(f"✓ 人形机器人API路由已注册 ({len(humanoid_routes)}个路由)")
            for route in humanoid_routes[:5]:  # 只显示前5个路由
                print(f"  - {route}")
            if len(humanoid_routes) > 5:
                print(f"   ... 还有 {len(humanoid_routes) - 5} 个路由")
        else:
            print("✗ 人形机器人API路由未注册")
            return False
        
        print("\n人形机器人API已成功集成到服务器中！")
        return True
        
    except Exception as e:
        print(f"✗ API集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("开始人形机器人AGI系统导入和集成测试")
    print("=" * 60)
    
    all_passed = True
    
    # 测试导入
    if not test_humanoid_imports():
        all_passed = False
    
    # 测试架构集成
    if not test_architecture_integration():
        all_passed = False
    
    # 测试API集成
    if not test_api_integration():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过！人形机器人AGI系统已成功集成。")
    else:
        print("✗ 部分测试失败！请检查错误信息。")
    
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)