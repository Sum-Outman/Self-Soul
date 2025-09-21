import requests
import time

# 配置API基础URL
base_url = "http://127.0.0.1:8000"

def test_training_functionality():
    """测试训练面板功能是否完整"""
    print("开始测试训练功能...")
    
    # 1. 测试健康检查端点
    print("\n1. 测试健康检查端点")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"健康检查响应: {response.json()}")
        if response.status_code != 200:
            print("警告: 健康检查失败")
    except Exception as e:
        print(f"错误: 健康检查失败: {e}")
    
    # 2. 获取模型状态，确认至少有一个模型可用
    print("\n2. 获取模型状态")
    try:
        response = requests.get(f"{base_url}/api/models/status")
        if response.status_code == 200:
            models_status = response.json()
            print(f"已加载的模型数: {len(models_status.get('data', []))}")
        else:
            print(f"警告: 获取模型状态失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"错误: 获取模型状态失败: {e}")
    
    # 3. 启动单个模型训练（使用language模型作为示例）
    print("\n3. 启动单个模型训练")
    training_config = {
        "mode": "individual",
        "models": ["language"],
        "parameters": {
            "epochs": 3,
            "batchSize": 16,
            "learningRate": 0.001,
            "validationSplit": 0.2,
            "dropoutRate": 0.1,
            "weightDecay": 0.0001,
            "momentum": 0.9,
            "optimizer": "adam"
        }
    }
    
    try:
        response = requests.post(f"{base_url}/api/train", json=training_config)
        if response.status_code == 200:
            result = response.json()
            job_id = result.get("job_id")
            print(f"训练任务已启动，任务ID: {job_id}")
            
            # 4. 轮询训练状态
            print("\n4. 轮询训练状态")
            for _ in range(6):  # 轮询6次，每次等待2秒
                time.sleep(2)
                try:
                    status_response = requests.get(f"{base_url}/api/training/status/{job_id}")
                    if status_response.status_code == 200:
                        status = status_response.json()
                        print(f"训练状态: {status.get('data', {}).get('status')}")
                        print(f"训练进度: {status.get('data', {}).get('progress', 0)}%")
                        if status.get('data', {}).get('status') in ['completed', 'failed', 'stopped']:
                            print("训练已完成或停止")
                            break
                except Exception as e:
                    print(f"获取状态失败: {e}")
            
            # 5. 停止训练（如果仍在运行）
            print("\n5. 停止训练任务")
            try:
                stop_response = requests.post(f"{base_url}/api/training/stop/{job_id}")
                if stop_response.status_code == 200:
                    print(f"停止训练响应: {stop_response.json()}")
            except Exception as e:
                print(f"停止训练失败: {e}")
        else:
            print(f"警告: 启动训练失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
    except Exception as e:
        print(f"错误: 启动训练失败: {e}")
    
    print("\n训练功能测试完成")

if __name__ == "__main__":
    test_training_functionality()