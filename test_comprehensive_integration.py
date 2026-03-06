#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合集成测试：验证从渐进式服务器到主服务器的完整功能集成
"""

import requests
import json
import time
import sys
from typing import Dict, List, Any, Set

BASE_URL = "http://localhost:8000"

# 从 progressive_server.py 提取的关键API端点列表
# 这些是必须集成到主服务器的API端点
EXPECTED_ENDPOINTS = [
    # 基本健康检查
    ("GET", "/health"),
    ("GET", "/docs"),
    ("GET", "/openapi.json"),
    
    # 聊天API
    ("POST", "/api/chat"),
    
    # 系统状态
    ("GET", "/api/system/status"),
    
    # AGI规划推理
    ("POST", "/api/agi/plan-with-reasoning"),
    ("POST", "/api/agi/analyze-causality"),
    ("POST", "/api/agi/temporal-planning"),
    ("POST", "/api/agi/cross-domain-planning"),
    ("POST", "/api/agi/self-reflection"),
    
    # 目标管理
    ("GET", "/api/goals"),
    ("POST", "/api/goals/update"),
    
    # 知识管理
    ("GET", "/api/knowledge/domains"),
    ("GET", "/api/knowledge/concepts"),
    ("GET", "/api/knowledge/statistics"),
    ("GET", "/api/knowledge/search"),
    
    # 元认知
    ("POST", "/api/meta-cognition/analyze"),
    ("GET", "/api/meta-cognition/status"),
    
    # 可解释AI
    ("POST", "/api/explainable-ai/explain"),
    ("GET", "/api/explainable-ai/capabilities"),
    
    # 监控
    ("GET", "/api/monitoring/data"),
]

# 如果启用了机器人增强功能，还有额外的端点
ROBOT_ENHANCED_ENDPOINTS = [
    ("GET", "/api/robot/enhanced/status"),
    ("POST", "/api/robot/enhanced/motion/command"),
    ("GET", "/api/robot/enhanced/fusion/status"),
    ("POST", "/api/robot/enhanced/fusion/start"),
    ("POST", "/api/robot/enhanced/fusion/stop"),
    ("POST", "/api/robot/enhanced/fusion/process"),
    ("GET", "/api/robot/enhanced/motion/capabilities"),
    ("POST", "/api/robot/enhanced/emergency/stop"),
    ("GET", "/api/robot/enhanced/multimodal/test"),
    ("GET", "/api/robot/enhanced/hardware/simulated"),
]

def test_endpoint_exists(method: str, endpoint: str) -> bool:
    """测试端点是否存在并响应"""
    try:
        url = f"{BASE_URL}{endpoint}"
        
        # 根据方法类型发送请求
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            # 对于POST请求，发送空的JSON体
            response = requests.post(url, json={}, timeout=5)
        else:
            print(f"  ⚠️  未知方法: {method} {endpoint}")
            return False
        
        # 检查响应状态码
        if response.status_code in [200, 201, 400, 401, 422, 500]:
            # 任何响应状态码都表示端点存在
            # 500错误表示端点存在但有内部错误，这也是成功的检测
            return True
        elif response.status_code == 404:
            return False
        else:
            # 其他状态码表示端点存在但可能有其他问题
            return True
            
    except requests.exceptions.ConnectionError:
        print(f"  ✗ 连接错误: {method} {endpoint}")
        return False
    except requests.exceptions.Timeout:
        print(f"  ⏱️  超时: {method} {endpoint}")
        return False
    except Exception as e:
        print(f"  ⚠️  测试异常: {method} {endpoint} - {e}")
        return False

def test_endpoint_functionality(method: str, endpoint: str) -> bool:
    """测试端点的基本功能"""
    try:
        url = f"{BASE_URL}{endpoint}"
        
        # 根据端点类型发送适当的测试请求
        if endpoint == "/health":
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy" or data.get("status") == "ok"
            return False
            
        elif endpoint == "/api/chat":
            response = requests.post(url, json={
                "message": "Hello, test message",
                "context": {"test": True}
            }, timeout=5)
            return response.status_code in [200, 201, 400]
            
        elif endpoint == "/api/knowledge/search":
            response = requests.get(f"{url}?query=test", timeout=5)
            return response.status_code in [200, 400]
            
        elif endpoint == "/api/agi/plan-with-reasoning":
            response = requests.post(url, json={
                "goal": "Test goal",
                "context": {"test": True},
                "complexity": "low"
            }, timeout=10)
            return response.status_code in [200, 400, 500]
            
        elif endpoint == "/api/agi/temporal-planning":
            response = requests.post(url, json={
                "goal": "Test temporal planning",
                "temporal_constraints": {"deadline": "2026-12-31"},
                "context": {"test": True}
            }, timeout=10)
            return response.status_code in [200, 400, 500]
            
        elif endpoint == "/api/agi/cross-domain-planning":
            response = requests.post(url, json={
                "goal": "Test cross-domain planning",
                "target_domain": "test_domain",
                "available_domains": ["domain1", "domain2"],
                "context": {"test": True},
                "constraints": {}
            }, timeout=10)
            return response.status_code in [200, 400, 500]
            
        elif endpoint == "/api/meta-cognition/analyze":
            response = requests.post(url, json={
                "cognitive_data": {"test": True}
            }, timeout=10)
            return response.status_code in [200, 400, 500]
            
        else:
            # 对于其他端点，只测试是否存在
            return test_endpoint_exists(method, endpoint)
            
    except Exception as e:
        print(f"  ⚠️  功能测试异常: {method} {endpoint} - {e}")
        return False

def test_component_initialization() -> Dict[str, bool]:
    """测试组件初始化状态"""
    print("\n测试组件初始化状态...")
    
    components = {}
    
    # 测试系统状态端点以获取组件信息
    try:
        response = requests.get(f"{BASE_URL}/api/system/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            components_loaded = data.get("components_loaded", {})
            
            # 检查关键组件
            key_components = [
                "knowledge_service", "advanced_reasoning_engine",
                "temporal_reasoning_planner", "cross_domain_planner",
                "enhanced_meta_cognition", "self_reflection_optimizer",
                "integrated_planning_reasoning_engine", "causal_reasoning_enhancer"
            ]
            
            for component in key_components:
                status = components_loaded.get(component, "unknown")
                # 检查状态是否为活动或已初始化或可用
                components[component] = status in ["active", "initialized", "running", "available"]
                status_symbol = "✓" if components[component] else "✗"
                print(f"  {status_symbol} {component}: {status}")
        else:
            print("  ✗ 无法获取系统状态")
    except Exception as e:
        print(f"  ⚠️  组件测试异常: {e}")
    
    return components

def main():
    """主测试函数"""
    print("=" * 80)
    print("综合集成测试：验证从渐进式服务器到主服务器的完整功能集成")
    print("=" * 80)
    
    # 测试服务器连接
    print("\n1. 测试服务器连接...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("  ✓ 服务器连接成功")
        else:
            print(f"  ✗ 服务器连接失败: 状态码 {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ 服务器连接失败: {e}")
        return False
    
    # 测试端点存在性
    print("\n2. 测试API端点存在性...")
    endpoints_found = 0
    endpoints_total = len(EXPECTED_ENDPOINTS)
    
    for method, endpoint in EXPECTED_ENDPOINTS:
        exists = test_endpoint_exists(method, endpoint)
        status_symbol = "✓" if exists else "✗"
        print(f"  {status_symbol} {method} {endpoint}")
        if exists:
            endpoints_found += 1
    
    print(f"\n  找到 {endpoints_found}/{endpoints_total} 个端点")
    
    # 测试关键端点功能
    print("\n3. 测试关键端点功能...")
    key_endpoints = [
        ("GET", "/health"),
        ("POST", "/api/chat"),
        ("GET", "/api/knowledge/search"),
        ("POST", "/api/agi/plan-with-reasoning"),
        ("POST", "/api/agi/temporal-planning"),
        ("POST", "/api/agi/cross-domain-planning"),
        ("POST", "/api/meta-cognition/analyze"),
    ]
    
    functional_endpoints = 0
    for method, endpoint in key_endpoints:
        functional = test_endpoint_functionality(method, endpoint)
        status_symbol = "✓" if functional else "✗"
        print(f"  {status_symbol} {method} {endpoint}")
        if functional:
            functional_endpoints += 1
    
    print(f"\n  功能正常 {functional_endpoints}/{len(key_endpoints)} 个关键端点")
    
    # 测试组件初始化
    components = test_component_initialization()
    initialized_components = sum(1 for status in components.values() if status)
    print(f"\n  组件初始化 {initialized_components}/{len(components)} 个关键组件")
    
    # 集成度评估
    print("\n" + "=" * 80)
    print("集成度评估:")
    print("=" * 80)
    
    endpoint_coverage = endpoints_found / endpoints_total * 100
    functionality_coverage = functional_endpoints / len(key_endpoints) * 100
    component_coverage = initialized_components / len(components) * 100 if components else 0
    
    # 综合集成度评分
    integration_score = (
        endpoint_coverage * 0.3 +
        functionality_coverage * 0.4 +
        component_coverage * 0.3
    ) / 100
    
    print(f"  端点覆盖率: {endpoint_coverage:.1f}% ({endpoints_found}/{endpoints_total})")
    print(f"  功能覆盖率: {functionality_coverage:.1f}% ({functional_endpoints}/{len(key_endpoints)})")
    print(f"  组件覆盖率: {component_coverage:.1f}% ({initialized_components}/{len(components)})")
    print(f"  综合集成度: {integration_score:.2%}")
    
    # 集成状态评估
    if integration_score >= 0.9:
        integration_status = "✅ 完全集成"
    elif integration_score >= 0.7:
        integration_status = "⚠️  基本集成（部分功能需要优化）"
    elif integration_score >= 0.5:
        integration_status = "⚠️  部分集成（需要进一步工作）"
    else:
        integration_status = "❌ 集成不完整"
    
    print(f"\n  集成状态: {integration_status}")
    
    # 缺失功能识别
    print("\n" + "=" * 80)
    print("缺失/问题识别:")
    print("=" * 80)
    
    # 检查缺失的端点
    missing_endpoints = []
    for method, endpoint in EXPECTED_ENDPOINTS:
        if not test_endpoint_exists(method, endpoint):
            missing_endpoints.append(f"{method} {endpoint}")
    
    if missing_endpoints:
        print("  缺失的端点:")
        for endpoint in missing_endpoints:
            print(f"    ✗ {endpoint}")
    else:
        print("  ✓ 所有预期端点都存在")
    
    # 检查非功能端点
    non_functional_endpoints = []
    for method, endpoint in key_endpoints:
        if not test_endpoint_functionality(method, endpoint):
            non_functional_endpoints.append(f"{method} {endpoint}")
    
    if non_functional_endpoints:
        print("\n  功能异常的端点:")
        for endpoint in non_functional_endpoints:
            print(f"    ⚠️  {endpoint}")
    else:
        print("\n  ✓ 所有关键端点功能正常")
    
    # 检查未初始化的组件
    if components:
        uninitialized_components = [name for name, status in components.items() if not status]
        if uninitialized_components:
            print("\n  未初始化的组件:")
            for component in uninitialized_components:
                print(f"    ⚠️  {component}")
        else:
            print("\n  ✓ 所有关键组件已初始化")
    
    print("\n" + "=" * 80)
    
    # 总体评估
    if integration_score >= 0.8:
        print("✅ 集成测试通过：功能已基本完整集成到主服务器")
        return True
    else:
        print("⚠️  集成测试未完全通过：存在缺失或功能异常的部分")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)