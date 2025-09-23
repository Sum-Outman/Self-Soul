import requests
import json

# 测试 /api/models 端点
print("Testing /api/models endpoint...")
try:
    # 使用简化的请求配置
    response = requests.get('http://localhost:8000/api/models', timeout=10)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        try:
            # 尝试解析JSON
            data = response.json()
            print(f"Response structure: {list(data.keys())}")
            print(f"Total models: {data.get('total')}")
            
            # 检查是'models'键还是'data'键
            models = data.get('models', data.get('data', []))
            print(f"Models count: {len(models)}")
            
            # 只打印前5个模型的名称和端口，避免输出过多
            print("\nFirst 5 models:")
            for i, model in enumerate(models[:5]):
                print(f"{i+1}. {model.get('name', 'Unknown')} (Port: {model.get('port', 'Unknown')}, Status: {model.get('status', 'Unknown')})")
        except json.JSONDecodeError:
            print("Error: Invalid JSON response")
            print(f"Response text: {response.text[:200]}...")
    else:
        print(f"Error: {response.status_code} - {response.text}")
except requests.exceptions.Timeout:
    print("Error: Request timed out")
except Exception as e:
    print(f"Exception: {str(e)}")