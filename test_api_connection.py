#!/usr/bin/env python3
"""
测试前端API连接
"""

import requests
import sys

def test_backend_connection():
    """测试后端API连接"""
    urls = [
        "http://localhost:8000/health",
        "http://localhost:8000/docs",
        "http://localhost:8000/api/multimodal/status"
    ]
    
    results = []
    
    for url in urls:
        try:
            print(f"测试连接: {url}")
            response = requests.get(url, timeout=5)
            print(f"  状态码: {response.status_code}")
            if response.status_code == 200:
                print(f"  连接成功")
                results.append((url, True, response.status_code))
            else:
                print(f"  连接失败 (状态码: {response.status_code})")
                results.append((url, False, response.status_code))
        except requests.exceptions.ConnectionError:
            print(f"  连接被拒绝 - 服务可能未运行")
            results.append((url, False, "ConnectionError"))
        except requests.exceptions.Timeout:
            print(f"  连接超时")
            results.append((url, False, "Timeout"))
        except Exception as e:
            print(f"  错误: {e}")
            results.append((url, False, str(e)))
    
    return results

def test_frontend_connection():
    """测试前端连接"""
    url = "http://localhost:5175"
    
    try:
        print(f"测试前端连接: {url}")
        response = requests.get(url, timeout=10)
        print(f"  状态码: {response.status_code}")
        
        if response.status_code == 200:
            print(f"  前端连接成功")
            # 检查是否是Vue应用（包含常见Vue标记）
            content = response.text[:500]
            if "vue" in content.lower() or "app" in content.lower():
                print(f"  检测到Vue.js应用")
            return (url, True, response.status_code)
        else:
            print(f"  前端连接失败 (状态码: {response.status_code})")
            return (url, False, response.status_code)
    except requests.exceptions.ConnectionError:
        print(f"  前端连接被拒绝 - 服务可能未运行")
        return (url, False, "ConnectionError")
    except Exception as e:
        print(f"  前端连接错误: {e}")
        return (url, False, str(e))

def main():
    print("=" * 60)
    print("多模态系统API连接测试")
    print("=" * 60)
    
    print("\n1. 测试后端API连接...")
    backend_results = test_backend_connection()
    
    print("\n2. 测试前端连接...")
    frontend_result = test_frontend_connection()
    
    print("\n" + "=" * 60)
    print("测试总结:")
    print("=" * 60)
    
    # 统计后端连接
    backend_success = sum(1 for _, success, _ in backend_results if success)
    backend_total = len(backend_results)
    
    print(f"后端API: {backend_success}/{backend_total} 个端点连接成功")
    for url, success, status in backend_results:
        status_str = "✅ 成功" if success else "❌ 失败"
        print(f"  {url}: {status_str} ({status})")
    
    print(f"\n前端: {'✅ 成功' if frontend_result[1] else '❌ 失败'} ({frontend_result[2]})")
    
    # 整体评估
    overall_success = backend_success > 0 and frontend_result[1]
    
    print("\n" + "=" * 60)
    if overall_success:
        print("✅ 系统连接测试通过")
        print("建议: 前端和后端基本连接正常，可以继续进行功能测试")
    else:
        print("❌ 系统连接测试失败")
        print("建议: ")
        if backend_success == 0:
            print("  - 启动后端服务: python core/main.py")
        if not frontend_result[1]:
            print("  - 启动前端服务: cd app && npm run dev")
        print("  - 检查端口配置: 后端8000, 前端5175")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())