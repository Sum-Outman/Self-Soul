"""
硬件API端点审核测试脚本

用于测试和验证硬件相关的API端点功能，包括：
1. 摄像头API
2. 传感器API  
3. 电机/关节控制API
4. 多模态处理API
"""

import requests
import json
import time
import logging
from typing import Dict, List, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
TIMEOUT = 10

def test_endpoint(method: str, endpoint: str, data: Dict = None, params: Dict = None) -> Dict:
    """测试API端点"""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params, timeout=TIMEOUT)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=TIMEOUT)
        elif method.upper() == "PUT":
            response = requests.put(url, json=data, timeout=TIMEOUT)
        else:
            return {"status": "error", "message": f"Unsupported method: {method}"}
        
        result = {
            "status": "success" if response.status_code < 400 else "error",
            "status_code": response.status_code,
            "endpoint": endpoint,
            "method": method,
            "response_time": response.elapsed.total_seconds()
        }
        
        try:
            result["data"] = response.json()
        except:
            result["data"] = response.text[:500]
            
        return result
        
    except requests.exceptions.Timeout:
        return {"status": "error", "message": f"Timeout after {TIMEOUT}s", "endpoint": endpoint, "method": method}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Connection refused", "endpoint": endpoint, "method": method}
    except Exception as e:
        return {"status": "error", "message": str(e), "endpoint": endpoint, "method": method}

def test_chat_functionality():
    """测试聊天功能API"""
    logger.info("=== 测试聊天功能API ===")
    
    endpoints = [
        ("POST", "/api/chat", {"message": "测试硬件API审核", "session_id": "test_hardware_audit"}),
        ("POST", "/api/models/manager/chat", {"text": "测试管理器模型聊天", "session_id": "test_manager_chat"}),
        ("POST", "/api/models/manager/chat", {"message": "测试机器人控制命令", "session_id": "test_robot_command"})
    ]
    
    results = []
    for method, endpoint, data in endpoints:
        result = test_endpoint(method, endpoint, data)
        results.append(result)
        logger.info(f"{method} {endpoint}: {result['status']} ({result.get('status_code', 'N/A')})")
        
        if result["status"] == "success" and "data" in result:
            response_data = result["data"]
            if isinstance(response_data, dict):
                if "data" in response_data and "response" in response_data["data"]:
                    logger.info(f"  响应: {response_data['data']['response'][:100]}...")
    
    return results

def test_camera_api():
    """测试摄像头API"""
    logger.info("\n=== 测试摄像头API ===")
    
    endpoints = [
        ("GET", "/api/devices/cameras", None),
        ("POST", "/api/devices/cameras/0/connect", {"camera_index": 0}),
        ("POST", "/api/cameras/0/stream/start", None),
        ("POST", "/api/cameras/0/stream/stop", None),
        ("POST", "/api/devices/cameras/0/disconnect", None)
    ]
    
    results = []
    for method, endpoint, data in endpoints:
        result = test_endpoint(method, endpoint, data)
        results.append(result)
        
        status = result["status"]
        status_code = result.get("status_code", "N/A")
        logger.info(f"{method} {endpoint}: {status} ({status_code})")
        
        if status == "success":
            # 检查返回数据格式
            response_data = result.get("data", {})
            if isinstance(response_data, dict):
                if "data" in response_data:
                    logger.info(f"  返回数据: {json.dumps(response_data['data'], ensure_ascii=False)[:100]}...")
    
    return results

def test_sensor_api():
    """测试传感器API"""
    logger.info("\n=== 测试传感器API ===")
    
    endpoints = [
        ("GET", "/api/robot/sensors", None),
        ("GET", "/api/robot/sensors/data", None),
        ("GET", "/api/devices/sensors", None),
        ("POST", "/api/devices/sensors/temperature/connect", {"sensor_id": "test_temperature"}),
        ("GET", "/api/devices/sensors/temperature/data", None)
    ]
    
    results = []
    for method, endpoint, data in endpoints:
        result = test_endpoint(method, endpoint, data)
        results.append(result)
        
        status = result["status"]
        status_code = result.get("status_code", "N/A")
        logger.info(f"{method} {endpoint}: {status} ({status_code})")
        
        if status == "success" and "data" in result:
            response_data = result["data"]
            if isinstance(response_data, dict):
                if "sensor_data" in response_data:
                    sensor_data = response_data["sensor_data"]
                    logger.info(f"  传感器数据: {json.dumps(sensor_data, ensure_ascii=False)[:100]}...")
    
    return results

def test_motor_control_api():
    """测试电机控制API"""
    logger.info("\n=== 测试电机控制API ===")
    
    endpoints = [
        ("GET", "/api/devices/actuators", None),
        ("GET", "/api/robot/hardware/detect", None),
        ("POST", "/api/robot/enhanced/motion/command", {
            "command_type": "direct",
            "command_data": {
                "motion_type": "walking",
                "target": {"x": 0.5, "y": 0, "z": 0},
                "constraints": {"max_velocity": 0.1},
                "control_mode": "position",
                "duration": 2.0
            },
            "priority": 5,
            "async_execution": False
        }),
        ("GET", "/api/robot/enhanced/status", None)
    ]
    
    results = []
    for method, endpoint, data in endpoints:
        result = test_endpoint(method, endpoint, data)
        results.append(result)
        
        status = result["status"]
        status_code = result.get("status_code", "N/A")
        logger.info(f"{method} {endpoint}: {status} ({status_code})")
        
        if status == "success" and "data" in result:
            response_data = result["data"]
            if isinstance(response_data, dict):
                logger.info(f"  返回数据: {json.dumps(response_data, ensure_ascii=False)[:150]}...")
    
    return results

def test_multimodal_processing():
    """测试多模态处理API"""
    logger.info("\n=== 测试多模态处理API ===")
    
    endpoints = [
        ("GET", "/api/agi/status", None),
        ("POST", "/api/agi/process", {
            "input_type": "multimodal",
            "data": {
                "text": "检测前方的物体",
                "image": "base64_placeholder",
                "audio": "base64_placeholder"
            }
        }),
        ("POST", "/api/agi/cross-domain-planning", {
            "goal": "机器人导航到指定位置",
            "target_domain": "navigation",
            "context": {"environment": "indoor"},
            "available_domains": ["vision", "sensor", "motion"]
        })
    ]
    
    results = []
    for method, endpoint, data in endpoints:
        result = test_endpoint(method, endpoint, data)
        results.append(result)
        
        status = result["status"]
        status_code = result.get("status_code", "N/A")
        logger.info(f"{method} {endpoint}: {status} ({status_code})")
        
        if status == "success" and "data" in result:
            response_data = result["data"]
            if isinstance(response_data, dict):
                logger.info(f"  响应类型: {response_data.get('type', 'unknown')}")
    
    return results

def test_audio_stream_api():
    """测试音频流API"""
    logger.info("\n=== 测试音频流API连接 ===")
    
    # 测试音频相关端点
    endpoints = [
        ("GET", "/api/models/status", None),
        ("POST", "/api/models/audio/chat", {"text": "测试音频处理", "session_id": "test_audio"})
    ]
    
    results = []
    for method, endpoint, data in endpoints:
        result = test_endpoint(method, endpoint, data)
        results.append(result)
        
        status = result["status"]
        status_code = result.get("status_code", "N/A")
        logger.info(f"{method} {endpoint}: {status} ({status_code})")
    
    return results

def analyze_results(all_results: List[List[Dict]]):
    """分析测试结果"""
    logger.info("\n" + "="*60)
    logger.info("硬件API审核结果分析")
    logger.info("="*60)
    
    categories = ["聊天功能", "摄像头API", "传感器API", "电机控制API", "多模态处理", "音频API"]
    
    for i, (category, results) in enumerate(zip(categories, all_results)):
        total = len(results)
        success = sum(1 for r in results if r["status"] == "success")
        error = total - success
        
        logger.info(f"\n{category}:")
        logger.info(f"  总数: {total}, 成功: {success}, 失败: {error}")
        
        for result in results:
            if result["status"] == "error":
                logger.info(f"  ✗ {result['method']} {result['endpoint']}: {result.get('message', 'Unknown error')}")
            else:
                response_time = result.get("response_time", 0)
                logger.info(f"  ✓ {result['method']} {result['endpoint']}: {result['status_code']} ({response_time:.2f}s)")
    
    # 总体统计
    all_tests = [r for category_results in all_results for r in category_results]
    total_tests = len(all_tests)
    total_success = sum(1 for r in all_tests if r["status"] == "success")
    success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
    
    logger.info("\n" + "="*60)
    logger.info(f"总体测试结果: {total_success}/{total_tests} 成功 ({success_rate:.1f}%)")
    logger.info("="*60)
    
    return {
        "total_tests": total_tests,
        "total_success": total_success,
        "success_rate": success_rate,
        "categories": categories,
        "detailed_results": all_results
    }

def main():
    """主函数"""
    logger.info("开始硬件API端点审核...")
    logger.info(f"目标服务器: {BASE_URL}")
    
    # 等待服务器启动
    time.sleep(2)
    
    # 运行所有测试
    chat_results = test_chat_functionality()
    camera_results = test_camera_api()
    sensor_results = test_sensor_api()
    motor_results = test_motor_control_api()
    multimodal_results = test_multimodal_processing()
    audio_results = test_audio_stream_api()
    
    # 分析结果
    all_results = [chat_results, camera_results, sensor_results, motor_results, multimodal_results, audio_results]
    analysis = analyze_results(all_results)
    
    # 生成报告
    report_file = "hardware_api_audit_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n详细报告已保存到: {report_file}")
    
    # 建议和总结
    logger.info("\n=== 审核建议 ===")
    if analysis["success_rate"] >= 80:
        logger.info("✅ 硬件API集成良好，大部分端点工作正常")
    elif analysis["success_rate"] >= 50:
        logger.info("⚠️  硬件API部分工作，需要修复一些端点")
    else:
        logger.info("❌ 硬件API集成存在问题，需要重点修复")
    
    return analysis

if __name__ == "__main__":
    main()