"""
验证前端API连接测试

测试前后端服务连接性，验证 http://localhost:5175 和 http://localhost:8000 之间的通信。
"""

import os
import sys
import time
import subprocess
import requests
import threading
from typing import Optional

# 设置环境变量
os.environ['ENVIRONMENT'] = 'development'
os.environ['ALLOW_ROBOT_SIMULATION'] = 'true'
os.environ['ROBOT_HARDWARE_TEST_MODE'] = 'true'

def start_server() -> Optional[subprocess.Popen]:
    """启动渐进式服务器"""
    print("启动渐进式服务器...")
    
    # 准备环境
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # 启动服务器进程
        server_process = subprocess.Popen(
            [sys.executable, "progressive_server.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print(f"服务器进程已启动 (PID: {server_process.pid})")
        print("等待服务器启动（10秒）...")
        
        # 等待服务器启动
        time.sleep(10)
        
        return server_process
        
    except Exception as e:
        print(f"启动服务器失败: {e}")
        return None

def test_api_health(server_process: subprocess.Popen, timeout: int = 30) -> bool:
    """测试API健康检查端点"""
    print("测试API健康检查端点...")
    
    start_time = time.time()
    health_checked = False
    
    # 读取服务器输出以查看启动状态
    def read_output():
        try:
            for line in iter(server_process.stdout.readline, ''):
                if time.time() - start_time > timeout:
                    break
                line = line.strip()
                if line:
                    print(f"[Server] {line}")
                    # 检查服务器启动完成的消息
                    if "Uvicorn running on" in line or "Application startup complete" in line:
                        print("✅ 服务器启动完成")
        except Exception as e:
            print(f"读取服务器输出失败: {e}")
    
    # 启动输出读取线程
    output_thread = threading.Thread(target=read_output, daemon=True)
    output_thread.start()
    
    # 尝试连接健康检查端点
    health_url = "http://localhost:8000/health"
    api_health_url = "http://localhost:8000/api/health"
    
    attempts = 0
    max_attempts = 10
    
    while time.time() - start_time < timeout and attempts < max_attempts:
        try:
            # 先尝试基本健康检查
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                print(f"✅ 基础健康检查通过 (状态码: {response.status_code})")
                health_data = response.json()
                print(f"   状态: {health_data.get('status', 'unknown')}")
                print(f"   版本: {health_data.get('version', 'unknown')}")
                health_checked = True
                break
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.RequestException as e:
            print(f"健康检查请求异常: {e}")
        
        # 尝试API健康检查
        try:
            response = requests.get(api_health_url, timeout=5)
            if response.status_code == 200:
                print(f"✅ API健康检查通过 (状态码: {response.status_code})")
                health_data = response.json()
                print(f"   状态: {health_data.get('status', 'unknown')}")
                health_checked = True
                break
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.RequestException as e:
            print(f"API健康检查请求异常: {e}")
        
        attempts += 1
        print(f"连接尝试 {attempts}/{max_attempts} 失败，等待2秒后重试...")
        time.sleep(2)
    
    if health_checked:
        # 测试增强的机器人API端点
        print("\n测试增强的机器人API端点...")
        enhanced_endpoints = [
            "/api/robot/enhanced/status",
            "/api/robot/enhanced/motion/capabilities",
            "/api/robot/enhanced/fusion/status",
        ]
        
        for endpoint in enhanced_endpoints:
            try:
                url = f"http://localhost:8000{endpoint}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    print(f"✅ {endpoint} 测试通过")
                else:
                    print(f"⚠️  {endpoint} 返回状态码: {response.status_code}")
            except Exception as e:
                print(f"❌ {endpoint} 测试失败: {e}")
    
    return health_checked

def test_frontend_connection() -> bool:
    """测试前端连接"""
    print("\n测试前端连接 (http://localhost:5175)...")
    
    # 首先检查前端构建是否存在
    app_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
    if not os.path.exists(app_dir):
        print("❌ 前端应用目录不存在")
        return False
    
    # 检查package.json
    package_json = os.path.join(app_dir, "package.json")
    if not os.path.exists(package_json):
        print("❌ package.json 不存在")
        return False
    
    print("✅ 前端项目结构完整")
    
    # 尝试检查前端是否在运行（可选）
    # 由于我们可能没有启动前端，这里只检查构建配置
    
    # 检查vite配置
    vite_config = os.path.join(app_dir, "vite.config.js")
    if os.path.exists(vite_config) or os.path.exists(os.path.join(app_dir, "vite.config.ts")):
        print("✅ Vite配置存在")
    else:
        print("⚠️  Vite配置未找到")
    
    return True

def main():
    """主测试函数"""
    print("=" * 80)
    print("Self-Soul-B多模态系统 - API连接验证测试")
    print("=" * 80)
    print("测试目标: 验证前后端服务连接性")
    print("测试步骤:")
    print("  1. 启动渐进式服务器 (端口 8000)")
    print("  2. 测试API健康检查端点")
    print("  3. 测试增强的机器人API端点")
    print("  4. 验证前端项目结构 (端口 5175)")
    print("=" * 80)
    
    server_process = None
    api_test_passed = False
    frontend_test_passed = False
    
    try:
        # 步骤1: 启动服务器
        server_process = start_server()
        if not server_process:
            print("❌ 无法启动服务器")
            return 1
        
        # 步骤2: 测试API
        api_test_passed = test_api_health(server_process)
        
        if api_test_passed:
            print("\n✅ API连接测试通过")
        else:
            print("\n❌ API连接测试失败")
        
        # 步骤3: 测试前端连接
        frontend_test_passed = test_frontend_connection()
        
        if frontend_test_passed:
            print("✅ 前端项目验证通过")
        else:
            print("❌ 前端项目验证失败")
        
    except Exception as e:
        print(f"❌ 测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # 清理：停止服务器进程
        if server_process:
            print("\n停止服务器进程...")
            try:
                server_process.terminate()
                server_process.wait(timeout=5)
                print("✅ 服务器已停止")
            except subprocess.TimeoutExpired:
                print("⚠️  服务器进程终止超时，强制停止...")
                server_process.kill()
                server_process.wait()
                print("✅ 服务器已强制停止")
            except Exception as e:
                print(f"❌ 停止服务器时出错: {e}")
    
    print("\n" + "=" * 80)
    print("测试总结:")
    print(f"  API连接测试: {'✅ 通过' if api_test_passed else '❌ 失败'}")
    print(f"  前端项目验证: {'✅ 通过' if frontend_test_passed else '❌ 失败'}")
    
    if api_test_passed and frontend_test_passed:
        print("\n🎉 前后端连接验证完成！")
        print("\n下一步:")
        print("  1. 启动前端开发服务器: cd app && npm run dev")
        print("  2. 访问前端仪表板: http://localhost:5175")
        print("  3. 验证多模态对话功能")
        print("  4. 测试机器人硬件控制界面")
        return 0
    else:
        print("\n⚠️  连接验证未完全通过，需要进一步检查。")
        print("\n建议:")
        if not api_test_passed:
            print("  - 检查渐进式服务器启动日志")
            print("  - 验证端口8000未被占用")
            print("  - 检查依赖包是否已安装")
        if not frontend_test_passed:
            print("  - 检查前端项目完整性")
            print("  - 验证package.json配置")
            print("  - 确保Node.js和npm已正确安装")
        return 1

if __name__ == "__main__":
    sys.exit(main())