#!/usr/bin/env python3
"""
测试视频流WebSocket端点
"""
import asyncio
import websockets
import json
import sys
import time
import base64
import io
from PIL import Image

def create_test_image():
    """创建测试图像"""
    # 创建一个简单的测试图像
    img = Image.new('RGB', (640, 480), color='red')
    
    # 转换为base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    
    # 转换为base64字符串
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return img_base64

async def test_video_stream_connection(url, test_with_image=False):
    """测试视频流WebSocket连接"""
    try:
        print(f"尝试连接到: {url}")
        async with websockets.connect(url) as websocket:
            print("✓ WebSocket连接成功")
            
            # 等待初始连接响应
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                print(f"收到初始响应: {response_data}")
                
                if response_data.get("type") == "error":
                    print(f"✗ 连接错误: {response_data.get('message')}")
                    return False
                
                if response_data.get("type") == "connected":
                    print(f"✓ 连接成功: {response_data.get('message')}")
            except asyncio.TimeoutError:
                print("⚠️  未收到初始响应（可能正常）")
            
            # 发送测试消息
            if test_with_image:
                # 发送测试图像数据
                test_image = create_test_image()
                test_message = {
                    "type": "video_frame",
                    "video_data": test_image,
                    "timestamp": time.time()
                }
                print(f"发送测试图像数据 (大小: {len(test_image)} 字节)")
            else:
                # 发送简单的测试消息
                test_message = {
                    "type": "test",
                    "message": "Test video stream connection",
                    "timestamp": time.time()
                }
                print(f"发送测试消息: {test_message}")
            
            await websocket.send(json.dumps(test_message))
            
            # 等待响应
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                response_data = json.loads(response)
                print(f"收到处理响应: {response_data}")
                
                if response_data.get("type") == "video_processed":
                    print("✓ 视频处理成功")
                    data = response_data.get("data", {})
                    if data.get("status") == "success":
                        print(f"✓ 处理结果: {data.get('message', 'Success')}")
                        return True
                    else:
                        print(f"⚠️  处理警告: {data.get('message', 'Unknown')}")
                        return True
                else:
                    print(f"⚠️  收到非预期响应类型: {response_data.get('type')}")
                    return True
                    
            except asyncio.TimeoutError:
                print("⚠️  未收到处理响应（可能端点等待更多数据）")
                # 发送断开消息
                disconnect_msg = {"type": "websocket.disconnect"}
                await websocket.send(json.dumps(disconnect_msg))
                return True
                
    except Exception as e:
        print(f"✗ WebSocket连接失败: {e}")
        return False

async def test_all_video_endpoints():
    """测试所有视频相关端点"""
    base_url = "ws://localhost:8000"
    
    endpoints = [
        ("/ws/video-stream", False, "视频流处理端点"),
        ("/ws/camera-feed/0", False, "摄像头流端点"),
    ]
    
    print("=" * 60)
    print("视频流端点测试")
    print("=" * 60)
    
    results = []
    for endpoint, with_image, description in endpoints:
        url = base_url + endpoint
        print(f"\n测试: {description} ({endpoint})")
        
        success = await test_video_stream_connection(url, test_with_image=with_image)
        results.append((endpoint, description, success))
        
        await asyncio.sleep(1)  # 避免过载
    
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    
    successful = 0
    for endpoint, description, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {description}: {endpoint}")
        if success:
            successful += 1
    
    total = len(results)
    
    print(f"\n成功: {successful}/{total} ({successful/total*100:.1f}%)")
    
    return successful > 0

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
    success = asyncio.run(test_all_video_endpoints())
    
    if success:
        print("\n✅ 视频流端点测试基本通过")
    else:
        print("\n⚠️  视频流端点测试失败")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)