#!/usr/bin/env python3
"""
自主意识模块导入测试
测试自主意识模块是否可以正确导入和集成到统一认知架构中。
"""

import sys
import os
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_autonomy_imports():
    """测试自主意识模块导入"""
    print("=" * 60)
    print("自主意识模块导入测试")
    print("=" * 60)
    
    try:
        # 测试自主性系统导入
        from src.cognitive.autonomy import AutonomousSystem
        print("✓ AutonomousSystem 导入成功")
        
        # 测试架构导入
        from src.cognitive.architecture import UnifiedCognitiveArchitecture
        print("✓ UnifiedCognitiveArchitecture 导入成功")
        
        # 测试API导入
        from src.api.autonomy import router as autonomy_router
        print("✓ 自主意识API路由导入成功")
        
        print("\n所有自主意识模块导入成功！")
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
    print("架构集成测试")
    print("=" * 60)
    
    try:
        # 创建配置
        config = {
            'embedding_dim': 512,
            'max_shared_memory_mb': 100
        }
        
        # 创建统一认知架构
        from src.cognitive.architecture import UnifiedCognitiveArchitecture
        agi = UnifiedCognitiveArchitecture(config)
        
        print("✓ 统一认知架构创建成功")
        
        # 测试自主性组件属性
        print("\n测试自主性组件属性...")
        
        # 检查自主性系统是否可用
        if not hasattr(agi, 'autonomy'):
            print("✗ 架构中没有autonomy属性")
            return False
        
        # 尝试访问自主性系统（延迟加载）
        autonomy_system = agi.autonomy
        if autonomy_system is None:
            print("✗ 自主性系统不可用")
            return False
        
        print("✓ 自主性系统可用")
        
        # 测试组件注册
        print(f"\n组件注册表: {agi.component_registry}")
        
        if 'autonomy' not in agi.component_registry:
            print("✗ 自主性组件未在注册表中")
            return False
        
        print("✓ 自主性组件已注册")
        
        # 初始化组件
        print("\n初始化组件...")
        try:
            # 触发组件初始化
            agi.initialize_components()
            print("✓ 组件初始化完成")
        except Exception as e:
            print(f"✗ 组件初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 架构集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """测试API端点定义"""
    print("\n" + "=" * 60)
    print("API端点测试")
    print("=" * 60)
    
    try:
        from src.api.autonomy import router
        from src.api.autonomy import get_autonomy_status, get_autonomy_report
        
        # 检查路由是否存在
        if not hasattr(router, 'routes'):
            print("✗ 路由没有routes属性")
            return False
        
        # 统计路由数量
        route_count = len(router.routes)
        print(f"✓ 自主意识API路由数量: {route_count}")
        
        # 列出路由
        print("\nAPI端点:")
        for route in router.routes:
            if hasattr(route, 'path'):
                path = route.path
                methods = route.methods if hasattr(route, 'methods') else ['GET']
                print(f"  {list(methods)[0]} {path}")
        
        # 检查关键端点
        required_endpoints = [
            '/api/autonomy/status',
            '/api/autonomy/report',
            '/api/autonomy/motivations',
            '/api/autonomy/goals/active'
        ]
        
        print("\n检查关键端点...")
        all_found = True
        for endpoint in required_endpoints:
            found = False
            for route in router.routes:
                if hasattr(route, 'path') and route.path.endswith(endpoint):
                    found = True
                    break
            
            if found:
                print(f"✓ {endpoint}")
            else:
                print(f"✗ {endpoint} 未找到")
                all_found = False
        
        if not all_found:
            return False
        
        print("\n所有API端点定义正确！")
        return True
        
    except Exception as e:
        print(f"✗ API端点测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("自主意识功能集成测试")
    print("=" * 60)
    
    # 测试导入
    if not test_autonomy_imports():
        print("\n导入测试失败，停止后续测试")
        return False
    
    # 测试架构集成
    if not test_architecture_integration():
        print("\n架构集成测试失败")
        return False
    
    # 测试API端点
    if not test_api_endpoints():
        print("\nAPI端点测试失败")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！自主意识功能已成功集成。")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)