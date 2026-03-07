"""
测试FastAPI应用路由

检查server.py应用中注册的所有路由，特别是控制API路由。
"""

import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入应用
from api.server import app

print("=" * 60)
print("FastAPI应用路由检查")
print("=" * 60)

# 获取所有路由
all_routes = []
for route in app.routes:
    route_info = {
        'path': route.path,
        'name': route.name,
        'methods': list(route.methods) if hasattr(route, 'methods') else []
    }
    all_routes.append(route_info)

print(f"总路由数: {len(all_routes)}")

# 按前缀分组路由
routes_by_prefix = {}
for route in all_routes:
    path = route['path']
    # 提取前缀（第一个路径段）
    if path.startswith('/'):
        parts = path.split('/')
        if len(parts) > 1:
            prefix = f"/{parts[1]}"
        else:
            prefix = '/'
    else:
        prefix = 'unknown'
    
    if prefix not in routes_by_prefix:
        routes_by_prefix[prefix] = []
    routes_by_prefix[prefix].append(route)

print("\n按前缀分组的路由:")
for prefix in sorted(routes_by_prefix.keys()):
    routes = routes_by_prefix[prefix]
    print(f"\n{prefix} ({len(routes)}个路由):")
    for route in routes[:5]:  # 只显示前5个
        print(f"  - {route['path']} ({', '.join(route['methods'])})")
    if len(routes) > 5:
        print(f"    ... 还有 {len(routes) - 5} 个路由")

# 特别检查控制API路由
print("\n" + "=" * 60)
print("控制API路由详细检查")
print("=" * 60)

control_routes = [r for r in all_routes if r['path'].startswith('/api/control')]
print(f"以 '/api/control' 开头的路由数: {len(control_routes)}")

if control_routes:
    print("\n控制API路由:")
    for route in control_routes:
        print(f"  - {route['path']} ({', '.join(route['methods'])})")
else:
    print("\n未找到控制API路由！")
    
    # 检查所有以 /api/ 开头的路由
    api_routes = [r for r in all_routes if r['path'].startswith('/api/')]
    print(f"\n所有API路由 ({len(api_routes)}个):")
    for route in api_routes:
        print(f"  - {route['path']}")

print("\n" + "=" * 60)
print("检查server.py中控制API的导入状态")
print("=" * 60)

# 尝试直接从server.py检查控制API状态
try:
    # 重新导入server模块以获取最新状态
    import importlib
    import api.server as server_module
    importlib.reload(server_module)
    
    # 检查CONTROL_AVAILABLE变量
    if hasattr(server_module, 'CONTROL_AVAILABLE'):
        print(f"CONTROL_AVAILABLE: {server_module.CONTROL_AVAILABLE}")
    else:
        print("CONTROL_AVAILABLE 变量不存在")
        
    # 检查control_router变量
    if hasattr(server_module, 'control_router'):
        router = server_module.control_router
        if router:
            print(f"control_router 类型: {type(router)}")
            print(f"control_router 前缀: {router.prefix}")
            print(f"control_router 路由数: {len(router.routes)}")
        else:
            print("control_router 为 None")
    else:
        print("control_router 变量不存在")
        
except Exception as e:
    print(f"检查server.py状态时出错: {e}")
    import traceback
    traceback.print_exc()