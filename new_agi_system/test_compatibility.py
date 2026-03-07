#!/usr/bin/env python3
"""
测试遗留系统兼容性
"""

import asyncio
import aiohttp
import json
import time
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

async def test_legacy_compatibility():
    """测试遗留兼容性API"""
    base_url = "http://localhost:9000/api"
    
    print("测试遗留系统兼容性...")
    print(f"基础URL: {base_url}")
    
    async with aiohttp.ClientSession() as session:
        # 测试1: 模型列表
        print("\n1. 测试模型列表API...")
        try:
            async with session.get(f"{base_url}/models") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   成功: 获取到 {len(data.get('models', []))} 个模型")
                    for model in data.get('models', [])[:5]:  # 只显示前5个
                        print(f"     - {model['name']} -> {model['component']}")
                else:
                    print(f"   失败: 状态码 {response.status}")
        except Exception as e:
            print(f"   错误: {e}")
        
        # 测试2: 处理API
        print("\n2. 测试处理API...")
        test_data = {
            "input": {"text": "你好，世界！"},
            "model": "language_model",
            "parameters": {"temperature": 0.7}
        }
        
        try:
            async with session.post(f"{base_url}/process", 
                                  json=test_data) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   成功: 模型={data.get('model')}, 处理时间={data.get('processing_time')}")
                    print(f"   输出: {json.dumps(data.get('output', {}), indent=2, ensure_ascii=False)[:100]}...")
                else:
                    text = await response.text()
                    print(f"   失败: 状态码 {response.status}, 响应: {text}")
        except Exception as e:
            print(f"   错误: {e}")
        
        # 测试3: 状态检查
        print("\n3. 测试状态检查API...")
        try:
            async with session.get(f"{base_url}/status/language_model") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   成功: {data.get('model')} 状态 = {data.get('status')}")
                    print(f"   系统: {data.get('system')}, 端口: {data.get('port')}")
                else:
                    text = await response.text()
                    print(f"   失败: 状态码 {response.status}, 响应: {text}")
        except Exception as e:
            print(f"   错误: {e}")
        
        # 测试4: 多模态API
        print("\n4. 测试多模态API...")
        multimodal_data = {
            "text": "这是一只猫的照片",
            "image": "base64_encoded_image_data",
            "audio": "base64_encoded_audio_data"
        }
        
        try:
            async with session.post(f"{base_url}/multimodal", 
                                  json=multimodal_data) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   成功: 处理完成={data.get('processing_complete')}")
                    results = data.get('multimodal_results', {})
                    print(f"   处理了 {len(results)} 个模态")
                    for modality, result in results.items():
                        print(f"     - {modality}: {result.get('processed', False)}")
                else:
                    text = await response.text()
                    print(f"   失败: 状态码 {response.status}, 响应: {text}")
        except Exception as e:
            print(f"   错误: {e}")
        
        # 测试5: 训练API
        print("\n5. 测试训练API...")
        training_data = {
            "training_type": "cognitive_model",
            "model_name": "test_model",
            "dataset": {"size": 1000},
            "hyperparameters": {"learning_rate": 0.001}
        }
        
        try:
            async with session.post(f"{base_url}/train", 
                                  json=training_data) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   成功: 训练ID={data.get('training_id')}")
                    print(f"   状态: {data.get('status')}, 模型: {data.get('model')}")
                else:
                    text = await response.text()
                    print(f"   失败: 状态码 {response.status}, 响应: {text}")
        except Exception as e:
            print(f"   错误: {e}")

async def test_training_api():
    """测试训练API"""
    print("\n6. 测试训练API...")
    base_url = "http://localhost:9000/training"
    
    async with aiohttp.ClientSession() as session:
        # 测试开始训练
        training_request = {
            "type": "cognitive_model",
            "model_name": "test_model",
            "dataset": {"name": "test_dataset", "size": 1000},
            "hyperparameters": {"learning_rate": 0.001, "epochs": 10},
            "priority": "normal"
        }
        
        try:
            async with session.post(f"{base_url}/start", 
                                  json=training_request) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   训练开始: 任务ID={data.get('job_id')}")
                    print(f"   状态: {data.get('status')}, 消息: {data.get('message')}")
                    
                    # 如果有任务ID，测试状态检查
                    job_id = data.get('job_id')
                    if job_id:
                        await asyncio.sleep(0.5)
                        async with session.get(f"{base_url}/status/{job_id}") as status_response:
                            if status_response.status == 200:
                                status_data = await status_response.json()
                                print(f"   训练状态: {status_data.get('status')}")
                else:
                    text = await response.text()
                    print(f"   失败: 状态码 {response.status}, 响应: {text}")
        except Exception as e:
            print(f"   错误: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("遗留系统兼容性测试")
    print("=" * 60)
    print("注意: 测试前请确保API服务器正在运行 (端口 9000)")
    print()
    
    # 检查服务器是否运行
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 9000))
        sock.close()
        
        if result != 0:
            print("错误: API服务器未在端口9000上运行")
            print("请先启动服务器: python -m api.server --port 9000")
            return
    except Exception as e:
        print(f"连接检查失败: {e}")
        return
    
    # 运行测试
    loop = asyncio.get_event_loop()
    
    try:
        loop.run_until_complete(test_legacy_compatibility())
        loop.run_until_complete(test_training_api())
        
        print("\n" + "=" * 60)
        print("测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        loop.close()

if __name__ == "__main__":
    main()