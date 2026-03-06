#!/usr/bin/env python3
"""
测试所有模型的输入输出数据完整性
Test input/output data integrity for all models
"""

import json
import requests
import time
from typing import Dict, Any, List
import sys

BASE_URL = "http://localhost:8000"

def test_api_endpoint(method: str, endpoint: str, data: Dict[str, Any] = None, timeout: int = 10) -> Dict[str, Any]:
    """测试API端点并返回结果"""
    try:
        url = f"{BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=timeout)
        else:
            return {"status": "error", "message": f"Unsupported method: {method}"}
        
        return {
            "status": "success",
            "status_code": response.status_code,
            "data": response.json() if response.status_code == 200 else None,
            "response_time": response.elapsed.total_seconds()
        }
    except requests.exceptions.Timeout:
        return {"status": "timeout", "message": f"Request timed out after {timeout}s"}
    except requests.exceptions.ConnectionError:
        return {"status": "connection_error", "message": "Connection refused"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def test_chat_functionality():
    """测试聊天功能输入输出"""
    print("=" * 80)
    print("测试聊天功能输入输出")
    print("=" * 80)
    
    # 测试通用聊天端点
    test_cases = [
        {
            "name": "通用聊天API",
            "endpoint": "/api/chat",
            "method": "POST",
            "data": {
                "text": "你好，请介绍一下你自己",
                "context": {"user_id": "test_user", "session_id": "test_session"}
            }
        },
        {
            "name": "模型特定聊天API",
            "endpoint": "/api/models/8001/chat",
            "method": "POST",
            "data": {
                "text": "What is artificial intelligence?",
                "context": {"language": "en", "user_id": "test_user"}
            }
        }
    ]
    
    results = []
    for test in test_cases:
        print(f"\n测试: {test['name']}")
        print(f"端点: {test['method']} {test['endpoint']}")
        print(f"输入数据: {json.dumps(test['data'], ensure_ascii=False)}")
        
        result = test_api_endpoint(test['method'], test['endpoint'], test['data'])
        results.append({
            "test": test['name'],
            "endpoint": test['endpoint'],
            **result
        })
        
        if result["status"] == "success":
            print(f"✓ 成功 - 状态码: {result['status_code']}, 响应时间: {result['response_time']:.2f}s")
            if result.get("data"):
                print(f"输出数据格式: {type(result['data'])}")
                print(f"输出数据样本: {json.dumps(result['data'], ensure_ascii=False)[:200]}...")
                
                # 验证输出数据结构
                if isinstance(result['data'], dict):
                    required_fields = ["success", "response"]
                    missing_fields = [field for field in required_fields if field not in result['data']]
                    if missing_fields:
                        print(f"⚠ 警告: 缺少字段: {missing_fields}")
                    else:
                        print(f"✓ 输出数据结构完整")
        else:
            print(f"✗ 失败 - {result['status']}: {result.get('message', 'No message')}")
    
    return results

def test_hardware_functionality():
    """测试硬件功能输入输出"""
    print("\n" + "=" * 80)
    print("测试硬件功能输入输出")
    print("=" * 80)
    
    # 测试硬件端点
    test_cases = [
        {
            "name": "获取摄像头列表",
            "endpoint": "/api/devices/cameras",
            "method": "GET"
        },
        {
            "name": "获取传感器列表",
            "endpoint": "/api/devices/sensors",
            "method": "GET"
        },
        {
            "name": "获取执行器列表",
            "endpoint": "/api/devices/actuators",
            "method": "GET"
        },
        {
            "name": "连接温度传感器",
            "endpoint": "/api/devices/sensors/temperature/connect",
            "method": "POST",
            "data": {"sensor_id": "temperature", "protocol": "simulated"}
        },
        {
            "name": "获取传感器数据",
            "endpoint": "/api/devices/sensors/temperature/data",
            "method": "GET"
        }
    ]
    
    results = []
    for test in test_cases:
        print(f"\n测试: {test['name']}")
        print(f"端点: {test['method']} {test['endpoint']}")
        
        data = test.get("data")
        if data:
            print(f"输入数据: {json.dumps(data, ensure_ascii=False)}")
        
        result = test_api_endpoint(test['method'], test['endpoint'], data)
        results.append({
            "test": test['name'],
            "endpoint": test['endpoint'],
            **result
        })
        
        if result["status"] == "success":
            print(f"✓ 成功 - 状态码: {result['status_code']}, 响应时间: {result['response_time']:.2f}s")
            if result.get("data"):
                print(f"输出数据格式: {type(result['data'])}")
                print(f"输出数据样本: {json.dumps(result['data'], ensure_ascii=False)[:200]}...")
        else:
            print(f"✗ 失败 - {result['status']}: {result.get('message', 'No message')}")
    
    return results

def test_multimodal_functionality():
    """测试多模态处理输入输出"""
    print("\n" + "=" * 80)
    print("测试多模态处理输入输出")
    print("=" * 80)
    
    test_cases = [
        {
            "name": "文本处理",
            "endpoint": "/api/process/text",
            "method": "POST",
            "data": {
                "text": "测试文本处理功能",
                "language": "zh",
                "operation": "analyze"
            }
        },
        {
            "name": "图像处理",
            "endpoint": "/api/process/image",
            "method": "POST",
            "data": {
                "image_url": "https://example.com/test.jpg",
                "operation": "describe"
            }
        },
        {
            "name": "音频处理",
            "endpoint": "/api/process/audio",
            "method": "POST",
            "data": {
                "audio_url": "https://example.com/test.mp3",
                "operation": "transcribe"
            }
        },
        {
            "name": "AGI处理",
            "endpoint": "/api/agi/process",
            "method": "POST",
            "data": {
                "input": "分析当前系统状态",
                "context": {"domain": "system_monitoring"}
            }
        }
    ]
    
    results = []
    for test in test_cases:
        print(f"\n测试: {test['name']}")
        print(f"端点: {test['method']} {test['endpoint']}")
        print(f"输入数据: {json.dumps(test['data'], ensure_ascii=False)}")
        
        result = test_api_endpoint(test['method'], test['endpoint'], test['data'])
        results.append({
            "test": test['name'],
            "endpoint": test['endpoint'],
            **result
        })
        
        if result["status"] == "success":
            print(f"✓ 成功 - 状态码: {result['status_code']}, 响应时间: {result['response_time']:.2f}s")
            if result.get("data"):
                print(f"输出数据格式: {type(result['data'])}")
                print(f"输出数据样本: {json.dumps(result['data'], ensure_ascii=False)[:200]}...")
        else:
            print(f"✗ 失败 - {result['status']}: {result.get('message', 'No message')}")
    
    return results

def test_all_models_status():
    """测试所有模型状态"""
    print("\n" + "=" * 80)
    print("测试所有模型状态和输入输出能力")
    print("=" * 80)
    
    # 获取所有模型信息
    models_result = test_api_endpoint("GET", "/api/models")
    
    if models_result["status"] != "success" or not models_result.get("data"):
        print("无法获取模型列表")
        return []
    
    models_data = models_result["data"]
    print(f"获取到 {len(models_data.get('models', []))} 个模型信息")
    
    results = []
    # 测试关键模型端点
    key_model_endpoints = [
        ("语言模型", "/api/models/8001/chat"),
        ("知识模型", "/api/knowledge/search"),
        ("视觉模型", "/api/process/image"),
        ("规划模型", "/api/agi/plan-with-reasoning"),
        ("推理模型", "/api/agi/process"),
        ("传感器模型", "/api/robot/sensors/data"),
        ("运动模型", "/api/robot/enhanced/motion/command"),
    ]
    
    for model_name, endpoint in key_model_endpoints:
        print(f"\n测试模型: {model_name}")
        print(f"端点: {endpoint}")
        
        # 根据模型类型准备测试数据
        test_data = None
        if "chat" in endpoint:
            test_data = {"text": f"测试{model_name}功能", "context": {}}
        elif "knowledge" in endpoint:
            test_data = {"query": "人工智能", "limit": 5}
        elif "image" in endpoint:
            test_data = {"image_url": "https://example.com/test.jpg", "operation": "describe"}
        elif "plan" in endpoint:
            test_data = {"goal": "测试目标", "constraints": [], "context": {}}
        elif "process" in endpoint:
            test_data = {"input": f"测试{model_name}输入", "context": {}}
        elif "sensors" in endpoint:
            # GET请求，不需要数据
            pass
        elif "motion" in endpoint:
            test_data = {"command": "move", "parameters": {"joint": "shoulder", "angle": 45}}
        
        method = "POST" if test_data else "GET"
        result = test_api_endpoint(method, endpoint, test_data)
        
        results.append({
            "model": model_name,
            "endpoint": endpoint,
            **result
        })
        
        if result["status"] == "success":
            print(f"✓ 成功 - 状态码: {result['status_code']}, 响应时间: {result['response_time']:.2f}s")
            if result.get("data"):
                print(f"输出数据格式: {type(result['data'])}")
        else:
            print(f"✗ 失败 - {result['status']}: {result.get('message', 'No message')}")
    
    return results

def generate_report(all_results):
    """生成测试报告"""
    print("\n" + "=" * 80)
    print("测试报告摘要")
    print("=" * 80)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for category, results in all_results.items():
        print(f"\n{category}:")
        for result in results:
            total_tests += 1
            if result.get("status") == "success":
                passed_tests += 1
                status_symbol = "✓"
            else:
                failed_tests += 1
                status_symbol = "✗"
            
            test_name = result.get("test") or result.get("model") or result.get("endpoint", "Unknown")
            print(f"  {status_symbol} {test_name}: {result.get('status', 'unknown')}")
    
    print(f"\n总计: {total_tests} 个测试")
    print(f"通过: {passed_tests} 个")
    print(f"失败: {failed_tests} 个")
    print(f"通过率: {(passed_tests/total_tests*100 if total_tests > 0 else 0):.1f}%")
    
    # 保存详细报告
    report = {
        "timestamp": time.time(),
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests/total_tests*100 if total_tests > 0 else 0)
        },
        "details": all_results
    }
    
    report_file = f"model_input_output_report_{int(time.time())}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细报告已保存到: {report_file}")
    
    return total_tests, passed_tests, failed_tests

def main():
    """主函数"""
    print("开始测试所有模型的输入输出数据完整性...")
    
    # 检查服务器是否运行
    print("检查服务器连接...")
    health_result = test_api_endpoint("GET", "/health")
    if health_result["status"] != "success":
        print(f"错误: 无法连接到服务器: {health_result.get('message')}")
        print("请确保服务器正在运行 (端口 8000)")
        return 1
    
    print("服务器连接正常")
    
    # 运行各项测试
    all_results = {}
    
    # 测试聊天功能
    chat_results = test_chat_functionality()
    all_results["聊天功能"] = chat_results
    
    # 测试硬件功能
    hardware_results = test_hardware_functionality()
    all_results["硬件功能"] = hardware_results
    
    # 测试多模态功能
    multimodal_results = test_multimodal_functionality()
    all_results["多模态功能"] = multimodal_results
    
    # 测试所有模型状态
    models_results = test_all_models_status()
    all_results["模型状态"] = models_results
    
    # 生成报告
    total, passed, failed = generate_report(all_results)
    
    if failed > 0:
        print(f"\n警告: 有 {failed} 个测试失败，需要检查相关功能")
        return 1
    else:
        print(f"\n✅ 所有测试通过！")
        return 0

if __name__ == "__main__":
    sys.exit(main())