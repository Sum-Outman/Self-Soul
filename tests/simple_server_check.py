"""
简单服务器检查 - 验证服务器模块是否可以导入和初始化
"""

import os
import sys

# 设置环境变量
os.environ['ENVIRONMENT'] = 'development'
os.environ['ALLOW_ROBOT_SIMULATION'] = 'true'
os.environ['ROBOT_HARDWARE_TEST_MODE'] = 'true'

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

print("=" * 80)
print("Self-Soul-B多模态系统 - 服务器模块检查")
print("=" * 80)

# 1. 检查progressive_server模块
print("1. 检查progressive_server模块...")
try:
    import progressive_server
    print("✅ progressive_server模块导入成功")
    
    # 检查FastAPI应用
    if hasattr(progressive_server, 'app'):
        print("✅ FastAPI应用对象存在")
        
        # 检查路由
        routes = progressive_server.app.routes if hasattr(progressive_server.app, 'routes') else []
        print(f"   应用包含 {len(routes)} 个路由")
        
        # 检查是否有机器人API增强路由
        robot_enhanced_routes = 0
        for route in routes:
            if hasattr(route, 'path') and '/api/robot/enhanced' in route.path:
                robot_enhanced_routes += 1
        
        print(f"   机器人增强API路由: {robot_enhanced_routes} 个")
    else:
        print("❌ FastAPI应用对象不存在")
        
except Exception as e:
    print(f"❌ progressive_server模块导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n2. 检查增强的机器人API模块...")
try:
    from core.robot_api_enhanced import router, initialize_enhanced_robot_api
    
    print("✅ 增强的机器人API模块导入成功")
    
    # 检查路由器端点
    routes = [route for route in router.routes]
    print(f"   增强的机器人API包含 {len(routes)} 个端点")
    
    # 显示端点
    print("   端点列表:")
    for i, route in enumerate(routes[:6]):  # 只显示前6个
        if hasattr(route, 'path'):
            print(f"     {i+1}. {route.path}")
    if len(routes) > 6:
        print(f"     ... 还有 {len(routes)-6} 个端点")
    
    # 初始化测试
    print("\n   初始化增强的机器人API...")
    initialized = initialize_enhanced_robot_api()
    print(f"   初始化结果: {initialized}")
    
except Exception as e:
    print(f"❌ 增强的机器人API模块导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n3. 检查前端项目结构...")
app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app')
if os.path.exists(app_dir):
    print("✅ 前端应用目录存在")
    
    # 检查关键文件
    required_files = [
        'package.json',
        'index.html',
        'src/main.ts',
        'src/App.vue'
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(app_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  缺少 {len(missing_files)} 个关键文件: {', '.join(missing_files)}")
    else:
        print("✅ 所有关键前端文件都存在")
        
    # 检查package.json脚本
    package_json = os.path.join(app_dir, 'package.json')
    if os.path.exists(package_json):
        import json
        try:
            with open(package_json, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
            
            if 'scripts' in package_data:
                scripts = package_data['scripts']
                print(f"✅ package.json包含 {len(scripts)} 个脚本")
                
                # 检查开发脚本
                if 'dev' in scripts:
                    print(f"   开发脚本: npm run dev")
                else:
                    print("⚠️  缺少 'dev' 脚本")
            else:
                print("⚠️  package.json缺少scripts部分")
                
        except Exception as e:
            print(f"❌ 读取package.json失败: {e}")
else:
    print("❌ 前端应用目录不存在")

print("\n" + "=" * 80)
print("检查完成")
print("=" * 80)
print("\n建议:")
print("1. 如果所有模块导入成功，可以启动服务器进行完整测试")
print("2. 前端项目结构完整，可以启动开发服务器")
print("3. 使用以下命令启动:")
print("   - 后端: python progressive_server.py")
print("   - 前端: cd app && npm run dev")
print("\n测试URL:")
print("   - 前端仪表板: http://localhost:5175")
print("   - API文档: http://localhost:8000/docs")
print("   - 健康检查: http://localhost:8000/health")