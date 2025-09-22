import requests
import time

# API端点URLs
base_url = "http://localhost:8000"
start_endpoint = f"{base_url}/api/knowledge/auto-learning/start"
stop_endpoint = f"{base_url}/api/knowledge/auto-learning/stop"

# 测试启动自主学习
def test_start_auto_learning():
    print("=== 测试启动自主学习 ===")
    try:
        # 准备请求参数
        payload = {
            "domains": ["general_knowledge", "programming"],
            "priority": "high"
        }
        
        # 发送POST请求
        response = requests.post(start_endpoint, json=payload)
        
        # 打印响应结果
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {response.json()}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"请求失败: {str(e)}")
        return False

# 测试停止自主学习
def test_stop_auto_learning():
    print("\n=== 测试停止自主学习 ===")
    try:
        # 发送POST请求
        response = requests.post(stop_endpoint)
        
        # 打印响应结果
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {response.json()}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"请求失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("开始测试自主学习API端点...")
    
    # 先测试启动自主学习
    start_success = test_start_auto_learning()
    
    if start_success:
        print("\n启动自主学习测试成功！等待2秒后测试停止功能...")
        time.sleep(2)
        
        # 测试停止自主学习
        stop_success = test_stop_auto_learning()
        
        if stop_success:
            print("\n停止自主学习测试成功！")
        else:
            print("\n停止自主学习测试失败！")
    else:
        print("\n启动自主学习测试失败！")
        
    print("\n测试完成！")