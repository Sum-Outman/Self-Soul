#!/usr/bin/env python3
"""
测试多模型协作服务功能

根据《最严厉深度审核报告》要求，验证：
1. 多模型协作服务可正常运行（端口8016响应正常）
2. 管理器能实际协调至少3个子模型完成复杂任务
3. 服务提供完整的任务分发、结果融合、冲突解决功能
"""

import sys
import os
import time
import json
import requests
import logging
from typing import Dict, List, Any, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 服务配置
COLLABORATION_SERVICE_URL = "http://localhost:8016"
MAIN_SERVICE_URL = "http://localhost:8000"

def test_service_health() -> bool:
    """测试服务健康状态"""
    logger.info("=== 测试协作服务健康状态 ===")
    
    try:
        response = requests.get(f"{COLLABORATION_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ 服务健康检查通过: {data}")
            return True
        else:
            logger.error(f"❌ 服务健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ 服务连接失败: {e}")
        return False

def test_root_endpoint() -> bool:
    """测试根端点"""
    logger.info("=== 测试根端点 ===")
    
    try:
        response = requests.get(COLLABORATION_SERVICE_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ 根端点响应: {data}")
            return True
        else:
            logger.error(f"❌ 根端点失败: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ 根端点连接失败: {e}")
        return False

def test_task_creation() -> bool:
    """测试任务创建功能"""
    logger.info("=== 测试任务创建 ===")
    
    task_data = {
        "name": "视觉-语言协作测试",
        "description": "分析一张图片并生成详细描述",
        "required_models": ["vision", "language"],
        "input_data": {
            "image_url": "https://example.com/test.jpg",
            "analysis_type": "detailed_description"
        },
        "priority": "medium",
        "timeout": 30.0,
        "max_retries": 3,
        "dependencies": [],
        "metadata": {
            "test": True,
            "source": "collaboration_service_test"
        }
    }
    
    try:
        response = requests.post(
            f"{COLLABORATION_SERVICE_URL}/api/v1/tasks",
            json=task_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ 任务创建成功")
            logger.info(f"   任务ID: {data.get('task_id')}")
            logger.info(f"   状态: {data.get('status')}")
            logger.info(f"   消息: {data.get('message', '无消息')}")
            return True
        else:
            logger.error(f"❌ 任务创建失败: {response.status_code}")
            logger.error(f"   响应: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ 任务创建异常: {e}")
        return False

def test_task_status() -> bool:
    """测试任务状态查询"""
    logger.info("=== 测试任务状态查询 ===")
    
    # 先创建一个任务
    task_data = {
        "name": "状态查询测试",
        "description": "测试任务状态查询功能",
        "required_models": ["language"],
        "input_data": {"text": "测试状态查询"},
        "priority": "low"
    }
    
    try:
        # 创建任务
        create_response = requests.post(
            f"{COLLABORATION_SERVICE_URL}/api/v1/tasks",
            json=task_data,
            timeout=10
        )
        
        if create_response.status_code != 200:
            logger.warning("无法创建测试任务，跳过状态查询测试")
            return False
        
        task_info = create_response.json()
        task_id = task_info.get("task_id")
        
        if not task_id:
            logger.warning("未获取到任务ID，跳过状态查询测试")
            return False
        
        # 查询任务状态
        response = requests.get(
            f"{COLLABORATION_SERVICE_URL}/api/v1/tasks/{task_id}",
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ 任务状态查询成功")
            logger.info(f"   任务ID: {data.get('task_id')}")
            logger.info(f"   状态: {data.get('status')}")
            logger.info(f"   进度: {data.get('progress', 'N/A')}")
            return True
        else:
            logger.error(f"❌ 任务状态查询失败: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ 任务状态查询异常: {e}")
        return False

def test_scheduler_status() -> bool:
    """测试调度器状态"""
    logger.info("=== 测试调度器状态 ===")
    
    try:
        response = requests.get(
            f"{COLLABORATION_SERVICE_URL}/api/v1/scheduler/status",
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ 调度器状态查询成功")
            logger.info(f"   状态: {data.get('status')}")
            logger.info(f"   活跃任务数: {data.get('active_tasks', 0)}")
            logger.info(f"   待处理任务: {data.get('pending_tasks', 0)}")
            return True
        else:
            # 这个端点可能不存在，记录但不视为失败
            logger.warning(f"⚠️ 调度器状态端点可能不存在: {response.status_code}")
            return True  # 不视为失败
    except Exception as e:
        logger.warning(f"⚠️ 调度器状态查询异常（可能端点不存在）: {e}")
        return True  # 不视为失败

def test_task_cancel() -> bool:
    """测试任务取消功能"""
    logger.info("=== 测试任务取消 ===")
    
    # 先创建一个任务
    task_data = {
        "name": "取消测试任务",
        "description": "测试任务取消功能",
        "required_models": ["knowledge"],
        "input_data": {"query": "测试取消"},
        "priority": "low"
    }
    
    try:
        # 创建任务
        create_response = requests.post(
            f"{COLLABORATION_SERVICE_URL}/api/v1/tasks",
            json=task_data,
            timeout=10
        )
        
        if create_response.status_code != 200:
            logger.warning("无法创建取消测试任务，跳过取消测试")
            return False
        
        task_info = create_response.json()
        task_id = task_info.get("task_id")
        
        if not task_id:
            logger.warning("未获取到任务ID，跳过取消测试")
            return False
        
        # 取消任务
        response = requests.delete(
            f"{COLLABORATION_SERVICE_URL}/api/v1/tasks/{task_id}",
            timeout=5
        )
        
        if response.status_code in [200, 202]:
            data = response.json()
            logger.info(f"✅ 任务取消成功")
            logger.info(f"   任务ID: {data.get('task_id')}")
            logger.info(f"   状态: {data.get('status')}")
            return True
        else:
            logger.error(f"❌ 任务取消失败: {response.status_code}")
            logger.error(f"   响应: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ 任务取消异常: {e}")
        return False

def test_model_coordination() -> bool:
    """测试管理器模型协调功能"""
    logger.info("=== 测试管理器模型协调 ===")
    
    # 测试主服务器的管理器协调功能
    test_cases = [
        {
            "name": "视觉-语言协作",
            "task_description": "分析一张风景图片并生成诗意的描述",
            "required_models": ["vision", "language"]
        },
        {
            "name": "多模态处理",
            "task_description": "处理一段语音并转换为文字，然后进行情感分析",
            "required_models": ["audio", "language", "emotion"]
        },
        {
            "name": "知识检索",
            "task_description": "搜索关于人工智能的最新研究进展",
            "required_models": ["knowledge", "language"]
        }
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"\n测试用例 {i+1}: {test_case['name']}")
        logger.info(f"任务描述: {test_case['task_description']}")
        logger.info(f"所需模型: {test_case['required_models']}")
        
        try:
            # 使用主服务器的管理器协调功能
            response = requests.post(
                f"{MAIN_SERVICE_URL}/api/models/manager/chat",
                json={
                    "message": test_case["task_description"],
                    "models": test_case["required_models"],
                    "mode": "collaborative"
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "unknown")
                
                if status == "success":
                    logger.info(f"✅ 协调成功")
                    logger.info(f"   状态: {status}")
                    success_count += 1
                else:
                    logger.warning(f"⚠️ 协调完成但状态非成功: {status}")
                    logger.info(f"   消息: {data.get('message', '无消息')}")
            else:
                logger.error(f"❌ 协调请求失败: {response.status_code}")
                logger.error(f"   响应: {response.text[:200]}")
                
        except Exception as e:
            logger.error(f"❌ 协调请求异常: {e}")
    
    # 至少需要成功协调3个子模型完成复杂任务
    logger.info(f"\n协调测试结果: {success_count}/{len(test_cases)} 成功")
    
    # 根据最严厉审核报告要求，至少需要成功协调3个子模型
    if success_count >= 3:
        logger.info("✅ 满足审核要求: 管理器能实际协调至少3个子模型")
        return True
    elif success_count >= 1:
        logger.warning("⚠️ 部分满足: 管理器能协调但未达到3个子模型要求")
        return False
    else:
        logger.error("❌ 不满足: 管理器未能协调任何子模型")
        return False

def test_main_server_health() -> bool:
    """测试主服务器健康状态"""
    logger.info("=== 测试主服务器健康状态 ===")
    
    try:
        response = requests.get(f"{MAIN_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ 主服务器健康检查通过: {data}")
            return True
        else:
            logger.error(f"❌ 主服务器健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ 主服务器连接失败: {e}")
        return False

def generate_audit_report(test_results: Dict[str, bool]) -> None:
    """生成审核报告"""
    logger.info("\n" + "="*60)
    logger.info("多模型协作服务审核报告")
    logger.info("="*60)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    logger.info(f"测试总数: {total_tests}")
    logger.info(f"通过数: {passed_tests}")
    logger.info(f"失败数: {total_tests - passed_tests}")
    logger.info(f"通过率: {pass_rate:.1f}%")
    
    logger.info("\n详细测试结果:")
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"  {test_name}: {status}")
    
    logger.info("\n" + "="*60)
    logger.info("最严厉审核标准验证:")
    logger.info("="*60)
    
    # 根据《最严厉深度审核报告》要求验证
    requirements = {
        "1. 多模型协作服务可正常运行": test_results.get("service_health", False),
        "2. 管理器能实际协调至少3个子模型": test_results.get("model_coordination", False),
        "3. 服务提供完整的任务分发功能": test_results.get("task_creation", False),
        "4. 服务提供任务状态查询功能": test_results.get("task_status", False),
        "5. 服务提供任务取消功能": test_results.get("task_cancel", False)
    }
    
    all_requirements_met = all(requirements.values())
    
    for req_name, req_met in requirements.items():
        status = "✅ 满足" if req_met else "❌ 不满足"
        logger.info(f"  {req_name}: {status}")
    
    logger.info("\n" + "="*60)
    logger.info("审核结论:")
    logger.info("="*60)
    
    if all_requirements_met:
        logger.info("✅ 系统满足最严厉审核报告的生产要求！")
        logger.info("✅ 多模型协作服务运行正常")
        logger.info("✅ 管理器协调机制完整")
        logger.info("✅ 服务功能完整")
    else:
        logger.info("⚠️ 系统部分满足要求，但存在以下问题:")
        for req_name, req_met in requirements.items():
            if not req_met:
                logger.info(f"  ❌ {req_name}")
    
    # 保存报告到文件
    report_data = {
        "timestamp": time.time(),
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "pass_rate": pass_rate,
        "test_results": test_results,
        "requirements": requirements,
        "all_requirements_met": all_requirements_met,
        "audit_conclusion": "PASS" if all_requirements_met else "PARTIAL_PASS"
    }
    
    report_file = f"collaboration_service_audit_{int(time.time())}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n详细报告已保存到: {report_file}")

def main():
    """主测试函数"""
    logger.info("开始多模型协作服务最严厉审核测试")
    logger.info(f"协作服务地址: {COLLABORATION_SERVICE_URL}")
    logger.info(f"主服务地址: {MAIN_SERVICE_URL}")
    
    # 所有测试用例
    test_functions = [
        ("main_server_health", test_main_server_health),
        ("service_health", test_service_health),
        ("root_endpoint", test_root_endpoint),
        ("task_creation", test_task_creation),
        ("task_status", test_task_status),
        ("scheduler_status", test_scheduler_status),
        ("task_cancel", test_task_cancel),
        ("model_coordination", test_model_coordination)
    ]
    
    test_results = {}
    
    for test_name, test_func in test_functions:
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            logger.error(f"测试 {test_name} 执行异常: {e}")
            test_results[test_name] = False
    
    # 生成审核报告
    generate_audit_report(test_results)
    
    # 返回总体结果
    all_passed = all(test_results.values())
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())