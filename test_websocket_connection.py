#!/usr/bin/env python3
"""
测试WebSocket连接
"""
import asyncio
import websockets
import json
import sys
import time

async def test_websocket_connection(url):
    """测试WebSocket连接"""
    try:
        print(f"尝试连接到: {url}")
        async with websockets.connect(url) as websocket:
            print("✓ WebSocket连接成功")
            
            # 发送测试消息
            test_message = {"type": "test", "message": "Hello WebSocket"}
            await websocket.send(json.dumps(test_message))
            print(f"发送消息: {test_message}")
            
            # 接收响应
            response = await websocket.recv()
            print(f"收到响应: {response}")
            
            return True
    except Exception as e:
        print(f"✗ WebSocket连接失败: {e}")
        return False

async def test_all_websockets():
    """测试所有WebSocket端点"""
    base_url = "ws://localhost:8000"
    
    endpoints = [
        "/ws/training/test123",
        "/ws/monitoring", 
        "/ws/test-connection",
        "/ws/device-control",
        "/ws/autonomous-learning/status",
        "/ws/audio-stream",
        "/ws/video-stream"
    ]
    
    print("=" * 60)
    print("WebSocket连接测试")
    print("=" * 60)
    
    results = []
    for endpoint in endpoints:
        url = base_url + endpoint
        success = await test_websocket_connection(url)
        results.append((endpoint, success))
        await asyncio.sleep(0.5)  # 避免过载
    
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    
    for endpoint, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {endpoint}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n成功: {successful}/{total} ({successful/total*100:.1f}%)")
    
    return successful == total

def main():
    """主函数"""
    try:
        # 检查websockets库
        import websockets
    except ImportError:
        print("错误: 需要安装websockets库")
        print("运行: pip install websockets")
        return False
    
    # 运行测试
    success = asyncio.run(test_all_websockets())
    
    if success:
        print("\n✅ 所有WebSocket端点连接成功")
    else:
        print("\n⚠️  部分WebSocket端点连接失败")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)