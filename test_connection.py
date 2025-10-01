import requests
import time

# 测试连接到mock服务器
def test_mock_server():
    url = 'http://localhost:8000/health'
    try:
        print(f"Testing connection to {url}...")
        response = requests.get(url, timeout=5)
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.json()}")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

# 主函数
if __name__ == "__main__":
    print("Mock Server Connection Test")
    print("="*50)
    
    # 测试多次连接
    for i in range(3):
        print(f"\nTest #{i+1}:")
        success = test_mock_server()
        print(f"Test #{i+1} {'PASSED' if success else 'FAILED'}")
        if i < 2:  # 最后一次测试后不需要等待
            time.sleep(2)