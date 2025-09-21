import http.client
import time

print("开始测试HTTP连接...")

# 创建一个简单的HTTP连接测试
try:
    # 连接到本地服务器
    conn = http.client.HTTPConnection('127.0.0.1', 8000, timeout=5)
    print("创建连接对象成功")
    
    # 尝试发送请求
    start_time = time.time()
    print("尝试发送GET请求到/health...")
    conn.request('GET', '/health')
    print(f"发送请求成功，耗时: {time.time() - start_time:.2f}秒")
    
    # 获取响应
    response = conn.getresponse()
    print(f"获取响应成功，状态码: {response.status}")
    
    # 读取响应内容
    data = response.read()
    print(f"响应内容: {data.decode('utf-8')}")
    
    print("\nHTTP连接测试成功！")
    
except Exception as e:
    print(f"\nHTTP连接测试失败: {str(e)}")
    import traceback
    traceback.print_exc()

finally:
    # 关闭连接
    if 'conn' in locals():
        conn.close()