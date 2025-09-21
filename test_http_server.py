import urllib.request
import urllib.error
import json
from datetime import datetime

# 测试函数
def test_endpoint(url, expected_status=200):
    try:
        print(f"\nTesting endpoint: {url}")
        # 设置请求头
        headers = {'Content-Type': 'application/json'}
        req = urllib.request.Request(url, headers=headers)
        
        # 发送请求
        start_time = datetime.now()
        with urllib.request.urlopen(req, timeout=5) as response:
            end_time = datetime.now()
            
            # 获取响应状态码
            status_code = response.status
            
            # 读取响应内容
            content = response.read().decode('utf-8')
            
            # 计算响应时间
            response_time = (end_time - start_time).total_seconds() * 1000  # 转换为毫秒
            
            print(f"Status code: {status_code}")
            print(f"Response time: {response_time:.2f} ms")
            
            # 尝试解析JSON响应
            try:
                data = json.loads(content)
                print(f"Response data: {json.dumps(data, indent=2)}")
                return True, data
            except json.JSONDecodeError:
                print(f"Response content: {content}")
                return status_code == expected_status, None
    except urllib.error.URLError as e:
        print(f"URL error: {e}")
        return False, None
    except urllib.error.HTTPError as e:
        print(f"HTTP error: {e.code} - {e.reason}")
        return False, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False, None

# 运行测试
def run_tests():
    print("===== Basic Test Server HTTP Tests =====")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试端点列表
    endpoints = [
        "http://127.0.0.1:8000/",
        "http://127.0.0.1:8000/health",
        "http://127.0.0.1:8000/api/training/start",
        "http://127.0.0.1:8000/api/models/status",
        "http://127.0.0.1:8000/not-found"  # 测试404响应
    ]
    
    # 运行所有测试
    results = {}
    for endpoint in endpoints:
        success, data = test_endpoint(endpoint)
        results[endpoint] = success
    
    # 汇总结果
    print("\n===== Test Results Summary =====")
    total_tests = len(results)
    passed_tests = sum(results.values())
    failed_tests = total_tests - passed_tests
    
    print(f"Total tests: {total_tests}")
    print(f"Passed tests: {passed_tests}")
    print(f"Failed tests: {failed_tests}")
    
    if failed_tests > 0:
        print("Failed endpoints:")
        for endpoint, success in results.items():
            if not success:
                print(f"  - {endpoint}")
    else:
        print("All tests passed successfully!")

if __name__ == "__main__":
    run_tests()