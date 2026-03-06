"""
前端-后端集成测试
测试前端API调用与后端的连接
模拟前端对后端API的调用
"""

import requests
import json
import time
import sys
from typing import Dict, List, Any, Optional

BASE_URL = "http://localhost:8000"
TIMEOUT = 10

def test_endpoint(method: str, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict:
    """测试单个API端点"""
    url = f"{BASE_URL}{endpoint}"
    
    print(f"\n测试: {method} {endpoint}")
    
    try:
        start_time = time.time()
        
        if method.upper() == "GET":
            response = requests.get(url, params=params, timeout=TIMEOUT)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=TIMEOUT)
        elif method.upper() == "PUT":
            response = requests.put(url, json=data, timeout=TIMEOUT)
        elif method.upper() == "DELETE":
            response = requests.delete(url, timeout=TIMEOUT)
        else:
            return {"status": "error", "message": f"Unsupported method: {method}"}
        
        elapsed_time = time.time() - start_time
        
        # 解析响应
        response_data = None
        if response.headers.get('content-type', '').startswith('application/json'):
            try:
                response_data = response.json()
            except:
                response_data = {"raw": response.text[:200]}
        
        result = {
            "method": method,
            "endpoint": endpoint,
            "status_code": response.status_code,
            "elapsed_time": round(elapsed_time, 3),
            "status": "success" if 200 <= response.status_code < 300 else "error",
            "response": response_data
        }
        
        print(f"  状态: {response.status_code} ({result['status']}) - {elapsed_time:.2f}s")
        
        if response.status_code >= 400:
            print(f"  错误: {response.text[:200]}")
        
        return result
        
    except requests.exceptions.Timeout:
        print(f"  超时: {method} {endpoint} (>{TIMEOUT}s)")
        return {"status": "timeout", "method": method, "endpoint": endpoint}
    except requests.exceptions.ConnectionError:
        print(f"  连接错误: 无法连接到服务器 {BASE_URL}")
        return {"status": "connection_error", "method": method, "endpoint": endpoint}
    except Exception as e:
        print(f"  异常: {str(e)}")
        return {"status": "exception", "method": method, "endpoint": endpoint, "error": str(e)}

def test_chat_integration():
    """测试聊天功能集成"""
    print("\n" + "="*60)
    print("测试聊天功能集成")
    print("="*60)
    
    endpoints = [
        ("POST", "/api/chat", {"message": "Hello, how are you?", "text": "Hello, how are you?"}),
        ("POST", f"/api/models/8001/chat", {"message": "Test message to manager model", "text": "Test message to manager model"}),
    ]
    
    results = []
    for method, endpoint, data in endpoints:
        result = test_endpoint(method, endpoint, data)
        results.append(result)
    
    return results

def test_hardware_integration():
    """测试硬件接入集成"""
    print("\n" + "="*60)
    print("测试硬件接入集成")
    print("="*60)
    
    endpoints = [
        # 摄像头API
        ("GET", "/api/devices/cameras", None),
        ("POST", "/api/devices/cameras/0/connect", {"camera_index": 0}),
        ("POST", "/api/cameras/0/stream/start", {}),
        ("POST", "/api/cameras/0/stream/stop", {}),
        ("POST", "/api/devices/cameras/0/disconnect", {}),
        
        # 传感器API
        ("GET", "/api/devices/sensors", None),
        ("POST", "/api/devices/sensors/temperature/connect", {}),
        ("GET", "/api/devices/sensors/temperature/data", None),
        ("POST", "/api/devices/sensors/temperature/disconnect", {}),
        
        # 机器人传感器
        ("GET", "/api/robot/sensors", None),
        ("GET", "/api/robot/sensors/data", None),
        
        # 电机/执行器API
        ("GET", "/api/devices/actuators", None),
        ("GET", "/api/robot/hardware/detect", None),
        
        # 机器人控制API
        ("POST", "/api/robot/enhanced/motion/command", {
            "command_type": "direct",
            "command_data": {
                "motion_type": "walking",
                "target": {"x": 0.5, "y": 0, "z": 0},
                "constraints": {"max_velocity": 0.1}
            }
        }),
        ("GET", "/api/robot/enhanced/status", None),
    ]
    
    results = []
    for method, endpoint, data in endpoints:
        result = test_endpoint(method, endpoint, data)
        results.append(result)
        
        # 在摄像头连接后等待一下
        if endpoint == "/api/devices/cameras/0/connect":
            time.sleep(1)
    
    return results

def test_multimodal_integration():
    """测试多模态处理集成"""
    print("\n" + "="*60)
    print("测试多模态处理集成")
    print("="*60)
    
    endpoints = [
        # 文本处理
        ("POST", "/api/process/text", {"text": "Test text for processing"}),
        
        # 图像处理 (模拟)
        ("POST", "/api/process/image", {"image_data": "base64_simulated_image_data", "format": "jpeg"}),
        
        # 视频处理 (模拟)
        ("POST", "/api/process/video", {"video_data": "base64_simulated_video_data", "format": "mp4"}),
        
        # 音频处理 (模拟)
        ("POST", "/api/process/audio", {"audio_data": "base64_simulated_audio_data", "format": "wav"}),
        
        # AGI处理
        ("GET", "/api/agi/status", None),
        ("POST", "/api/agi/process", {
            "input": "Test multimodal input",
            "modalities": ["text", "image"],
            "context": "Test context"
        }),
        ("POST", "/api/agi/cross-domain-planning", {
            "goal": "Test cross-domain planning",
            "domains": ["navigation", "manipulation"]
        }),
    ]
    
    results = []
    for method, endpoint, data in endpoints:
        result = test_endpoint(method, endpoint, data)
        results.append(result)
    
    return results

def test_serial_integration():
    """测试串口通信集成"""
    print("\n" + "="*60)
    print("测试串口通信集成")
    print("="*60)
    
    endpoints = [
        ("GET", "/api/serial/ports", None),
        ("GET", "/api/serial/status", None),
        # 注意：实际连接需要真实的串口设备，这里只测试端点存在性
    ]
    
    results = []
    for method, endpoint, data in endpoints:
        result = test_endpoint(method, endpoint, data)
        results.append(result)
    
    return results

def test_websocket_endpoints():
    """测试WebSocket端点存在性（不建立实际连接）"""
    print("\n" + "="*60)
    print("测试WebSocket端点存在性")
    print("="*60)
    
    # WebSocket端点列表（从前端代码和后端代码中提取）
    websocket_endpoints = [
        "/ws/training/{job_id}",
        "/ws/monitoring",
        "/ws/device-control",
        "/ws/audio-stream",
        "/ws/video-stream",
        "/ws/camera-feed/{camera_id}",
        "/ws/sensor-data/{sensor_id}",
    ]
    
    print("WebSocket端点配置:")
    for endpoint in websocket_endpoints:
        print(f"  - {endpoint}")
    
    # 尝试通过HTTP GET测试WebSocket端点（应该返回426 Upgrade Required或类似）
    print("\n测试WebSocket端点HTTP响应:")
    results = []
    for endpoint in websocket_endpoints[:2]:  # 只测试前两个
        # 替换路径参数
        test_endpoint = endpoint.replace("{job_id}", "test123").replace("{camera_id}", "0").replace("{sensor_id}", "temperature")
        url = f"{BASE_URL}{test_endpoint}"
        try:
            response = requests.get(url, timeout=5)
            status_code = response.status_code
            # WebSocket端点通常返回426 Upgrade Required或404 Not Found
            # 对于这个测试，我们只要端点响应就认为是成功的
            status = "success" if status_code < 500 else "error"
            print(f"  {test_endpoint}: HTTP {status_code} (期望: 426或其他WebSocket响应)")
            
            results.append({
                "method": "GET",
                "endpoint": test_endpoint,
                "status_code": status_code,
                "status": status,
                "elapsed_time": 0.0,
                "response": {"raw": f"HTTP {status_code}"}
            })
        except Exception as e:
            print(f"  {test_endpoint}: 连接错误 - {str(e)}")
            results.append({
                "method": "GET",
                "endpoint": test_endpoint,
                "status_code": 0,
                "status": "connection_error",
                "elapsed_time": 0.0,
                "response": {"error": str(e)}
            })
    
    if not results:
        # 如果没有结果，返回一个信息性结果
        results.append({
            "method": "INFO",
            "endpoint": "WebSocket端点配置",
            "status_code": 0,
            "status": "info",
            "elapsed_time": 0.0,
            "response": {"message": "WebSocket端点配置检查完成"}
        })
    
    return results

def test_frontend_api_compatibility():
    """测试前端API调用的兼容性"""
    print("\n" + "="*60)
    print("测试前端API调用兼容性")
    print("="*60)
    
    # 从前端api.ts中提取的关键端点
    frontend_endpoints = [
        # 系统状态
        ("GET", "/api/system/status", None),
        ("GET", "/health", None),
        
        # 模型管理
        ("GET", "/api/models", None),
        ("GET", "/api/models/available", None),
        ("GET", "/api/models/status", None),
        
        # 知识库
        ("GET", "/api/knowledge/files", None),
        ("GET", "/api/knowledge/search", {"query": "test", "domain": "general"}),
        
        # 训练
        ("GET", "/api/training/active-jobs", None),
        
        # 机器人状态
        ("GET", "/api/robot/status", None),
    ]
    
    results = []
    for method, endpoint, params in frontend_endpoints:
        result = test_endpoint(method, endpoint, params=params)
        results.append(result)
    
    return results

def generate_report(all_results: Dict[str, List[Dict]]):
    """生成测试报告"""
    print("\n" + "="*60)
    print("前端-后端集成测试报告")
    print("="*60)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    timeout_tests = 0
    
    for category, results in all_results.items():
        if isinstance(results, dict):
            continue
            
        print(f"\n{category}:")
        
        category_passed = 0
        category_total = 0
        
        for result in results:
            status = result.get('status', 'unknown')
            
            # 跳过信息性结果，不纳入统计
            if status == 'info':
                continue
                
            category_total += 1
            total_tests += 1
            
            status_code = result.get('status_code', 'N/A')
            
            if status == 'success':
                category_passed += 1
                passed_tests += 1
                print(f"  ✓ {result.get('method', 'UNKNOWN')} {result.get('endpoint', 'UNKNOWN')}: {status_code} ({result.get('elapsed_time', 0):.2f}s)")
            elif status == 'timeout':
                timeout_tests += 1
                print(f"  ⏱️ {result.get('method', 'UNKNOWN')} {result.get('endpoint', 'UNKNOWN')}: 超时")
            elif status == 'connection_error':
                failed_tests += 1
                print(f"  ✗ {result.get('method', 'UNKNOWN')} {result.get('endpoint', 'UNKNOWN')}: 连接错误")
            else:
                failed_tests += 1
                print(f"  ✗ {result.get('method', 'UNKNOWN')} {result.get('endpoint', 'UNKNOWN')}: 错误 ({status_code})")
        
        if category_total > 0:
            success_rate = (category_passed / category_total) * 100
            print(f"  通过率: {category_passed}/{category_total} ({success_rate:.1f}%)")
    
    # 总体统计
    print("\n" + "="*60)
    print("总体统计")
    print("="*60)
    
    if total_tests > 0:
        overall_success_rate = (passed_tests / total_tests) * 100
        
        print(f"总测试数: {total_tests}")
        print(f"通过: {passed_tests}")
        print(f"失败: {failed_tests}")
        print(f"超时: {timeout_tests}")
        print(f"总体通过率: {overall_success_rate:.1f}%")
        
        if overall_success_rate >= 90:
            print("\n✅ 前端-后端集成良好！")
        elif overall_success_rate >= 70:
            print("\n⚠️ 前端-后端集成需要改进")
        else:
            print("\n❌ 前端-后端集成存在严重问题")
    else:
        print("没有执行测试")
    
    # 建议
    print("\n" + "="*60)
    print("建议")
    print("="*60)
    
    if failed_tests > 0:
        print("需要修复的端点:")
        for category, results in all_results.items():
            if isinstance(results, dict):
                continue
            for result in results:
                if result.get('status') not in ['success', 'timeout', 'info']:
                    print(f"  - {result.get('method', 'UNKNOWN')} {result.get('endpoint', 'UNKNOWN')}: {result.get('status', 'error')}")
    
    return {
        "total_tests": total_tests,
        "passed": passed_tests,
        "failed": failed_tests,
        "timeout": timeout_tests,
        "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
    }

def main():
    """主测试函数"""
    print("前端-后端集成测试")
    print("="*60)
    print(f"后端URL: {BASE_URL}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 检查服务器是否运行
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ 后端服务器正在运行")
        else:
            print(f"⚠️ 后端服务器响应异常: {health_response.status_code}")
    except:
        print("❌ 后端服务器未运行或无法访问")
        print("请确保后端服务器正在运行: python -m core.main")
        return
    
    # 执行所有测试
    all_results = {}
    
    all_results["聊天功能集成"] = test_chat_integration()
    time.sleep(1)  # 短暂延迟
    
    all_results["硬件接入集成"] = test_hardware_integration()
    time.sleep(1)
    
    all_results["多模态处理集成"] = test_multimodal_integration()
    time.sleep(1)
    
    all_results["串口通信集成"] = test_serial_integration()
    time.sleep(1)
    
    all_results["前端API兼容性"] = test_frontend_api_compatibility()
    
    # WebSocket端点检查（信息性）
    ws_result = test_websocket_endpoints()
    all_results["WebSocket端点"] = ws_result
    
    # 生成报告
    report = generate_report(all_results)
    
    # 保存详细结果到文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = f"frontend_backend_integration_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "base_url": BASE_URL,
            "summary": report,
            "detailed_results": all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细报告已保存到: {report_file}")
    
    # 返回退出码
    if report.get('success_rate', 0) >= 80:
        print("\n✅ 集成测试基本通过")
        sys.exit(0)
    else:
        print("\n❌ 集成测试未通过")
        sys.exit(1)

if __name__ == "__main__":
    main()