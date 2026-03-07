"""
直接导入测试

测试server.py中失败的自认知API和控制API导入。
"""

import sys
import os

# 模拟server.py的环境
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("测试self_cognition导入...")
try:
    # 使用相对导入，就像server.py中那样
    from src.api.self_cognition import router as self_cognition_router
    print(f"  ✓ 导入成功, router类型: {type(self_cognition_router)}")
    print(f"    路由前缀: {self_cognition_router.prefix}")
    print(f"    标签: {self_cognition_router.tags}")
    print(f"    路由数量: {len(self_cognition_router.routes)}")
except ImportError as e:
    print(f"  ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试control导入...")
try:
    from src.api.control import router as control_router
    print(f"  ✓ 导入成功, router类型: {type(control_router)}")
    print(f"    路由前缀: {control_router.prefix}")
    print(f"    标签: {control_router.tags}")
    print(f"    路由数量: {len(control_router.routes)}")
except ImportError as e:
    print(f"  ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试evolution导入...")
try:
    from src.api.evolution import router as evolution_router
    print(f"  ✓ 导入成功, router类型: {type(evolution_router)}")
    print(f"    路由前缀: {evolution_router.prefix}")
    print(f"    标签: {evolution_router.tags}")
    print(f"    路由数量: {len(evolution_router.routes)}")
except ImportError as e:
    print(f"  ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试training导入...")
try:
    from src.api.training import router as training_router
    print(f"  ✓ 导入成功, router类型: {type(training_router)}")
    print(f"    路由前缀: {training_router.prefix}")
    print(f"    标签: {training_router.tags}")
    print(f"    路由数量: {len(training_router.routes)}")
except ImportError as e:
    print(f"  ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试humanoid导入...")
try:
    from src.api.humanoid import router as humanoid_router
    print(f"  ✓ 导入成功, router类型: {type(humanoid_router)}")
    print(f"    路由前缀: {humanoid_router.prefix}")
    print(f"    标签: {humanoid_router.tags}")
    print(f"    路由数量: {len(humanoid_router.routes)}")
except ImportError as e:
    print(f"  ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试autonomy导入...")
try:
    from src.api.autonomy import router as autonomy_router
    print(f"  ✓ 导入成功, router类型: {type(autonomy_router)}")
    print(f"    路由前缀: {autonomy_router.prefix}")
    print(f"    标签: {autonomy_router.tags}")
    print(f"    路由数量: {len(autonomy_router.routes)}")
except ImportError as e:
    print(f"  ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()