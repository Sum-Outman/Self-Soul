"""
启动服务器进行机器人API测试

启动渐进式服务器，配置为开发模式，启用机器人模拟功能。
"""

import subprocess
import sys
import os
import time
import signal
import atexit

def start_server():
    """启动渐进式服务器"""
    print("启动渐进式服务器进行机器人API测试...")
    print("=" * 60)
    
    # 设置环境变量
    env = os.environ.copy()
    env['ENVIRONMENT'] = 'development'
    env['ALLOW_ROBOT_SIMULATION'] = 'true'
    env['ROBOT_HARDWARE_TEST_MODE'] = 'true'
    
    # 启动服务器
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
    print("等待服务器启动...")
    
    # 等待服务器启动
    server_ready = False
    for line in iter(server_process.stdout.readline, ''):
        print(f"服务器: {line.strip()}")
        if "Server will start with basic functionality" in line:
            server_ready = True
            break
        if "error" in line.lower() or "exception" in line.lower():
            print("服务器启动过程中出现错误")
            break
        time.sleep(0.1)
    
    if server_ready:
        print("✅ 服务器已启动并准备就绪")
        print("=" * 60)
        print("增强的机器人API端点已可用：")
        print("  - GET /api/robot/enhanced/status")
        print("  - GET /api/robot/enhanced/motion/capabilities")
        print("  - POST /api/robot/enhanced/motion/command")
        print("  - GET /api/robot/enhanced/fusion/status")
        print("  - POST /api/robot/enhanced/fusion/start")
        print("  - POST /api/robot/enhanced/fusion/process")
        print("  - GET /api/robot/enhanced/multimodal/test")
        print("  - GET /api/robot/enhanced/hardware/simulated")
        print("  - GET /api/robot/enhanced/test/echo")
        print("  - POST /api/robot/enhanced/emergency/stop")
        print("=" * 60)
        print("现在可以运行测试：")
        print("  python tests/test_enhanced_robot_api.py")
        print("=" * 60)
    else:
        print("❌ 服务器启动失败")
        return None
    
    return server_process

def cleanup(process):
    """清理函数，确保服务器进程被终止"""
    if process and process.poll() is None:
        print("终止服务器进程...")
        process.terminate()
        try:
            process.wait(timeout=5)
            print("服务器进程已终止")
        except subprocess.TimeoutExpired:
            print("强制终止服务器进程...")
            process.kill()
            process.wait()
            print("服务器进程已被强制终止")

if __name__ == "__main__":
    server = None
    
    try:
        server = start_server()
        if server:
            print("按 Ctrl+C 停止服务器并运行测试...")
            
            # 等待用户中断
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n用户中断，停止服务器...")
        
    except Exception as e:
        print(f"启动服务器时发生错误: {e}")
    
    finally:
        cleanup(server)
        print("程序结束")