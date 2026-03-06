#!/usr/bin/env python3
"""快速导入测试脚本"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from core.main import app
    print("✓ 主应用导入成功")
    
    # 检查路由
    routes = []
    for route in app.routes:
        if hasattr(route, "path"):
            routes.append(route.path)
    
    print(f"✓ 找到 {len(routes)} 个路由")
    
    # 检查关键路由
    critical_routes = ["/health", "/api/health/detailed", "/api/agi/status", "/api/monitoring/realtime"]
    for route in critical_routes:
        if any(route in r for r in routes):
            print(f"✓ 关键路由存在: {route}")
        else:
            print(f"✗ 关键路由缺失: {route}")
    
    print("\n✓ 基本功能测试通过")
    
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)