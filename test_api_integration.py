#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试主服务器的API集成功能
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_knowledge_search():
    """测试知识搜索API"""
    print("测试知识搜索API...")
    response = requests.get(f"{BASE_URL}/api/knowledge/search?query=engineering")
    if response.status_code == 200:
        result = response.json()
        print(f"✓ 知识搜索成功: {result.get('status', 'unknown')}")
        print(f"  结果数量: {len(result.get('data', {}).get('results', []))}")
        return True
    else:
        print(f"✗ 知识搜索失败: {response.status_code}")
        print(f"  响应: {response.text[:200]}")
        return False

def test_plan_with_reasoning():
    """测试集成规划推理API"""
    print("\n测试集成规划推理API...")
    
    # 创建一个简单的计划请求
    plan_data = {
        "goal": "开发一个简单的Web应用",
        "context": {
            "requirements": ["用户注册", "产品展示", "购物车"],
            "constraints": ["使用Python", "2周内完成"],
            "resources": ["开发人员2名", "基本服务器"]
        },
        "complexity": "medium"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/agi/plan-with-reasoning",
            json=plan_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 集成规划推理成功: {result.get('status', 'unknown')}")
            print(f"  计划ID: {result.get('data', {}).get('plan_id', 'N/A')}")
            print(f"  步骤数量: {len(result.get('data', {}).get('plan', {}).get('steps', []))}")
            return True
        else:
            print(f"✗ 集成规划推理失败: {response.status_code}")
            print(f"  响应: {response.text[:500]}")
            return False
    except Exception as e:
        print(f"✗ 集成规划推理异常: {e}")
        return False

def test_temporal_planning():
    """测试时间规划API"""
    print("\n测试时间规划API...")
    
    # 创建一个时间规划请求
    plan_data = {
        "goal": "完成项目开发",
        "temporal_constraints": {
            "deadline": "2026-03-20",
            "start_date": "2026-03-06",
            "work_hours_per_day": 8,
            "dependencies": [
                {"from": "需求分析", "to": "设计架构", "relation": "before"},
                {"from": "设计架构", "to": "编码实现", "relation": "before"},
                {"from": "编码实现", "to": "测试部署", "relation": "before"}
            ]
        },
        "context": {
            "resources": ["开发人员2名", "测试人员1名"],
            "budget": "limited",
            "priority": "high"
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/agi/temporal-planning",
            json=plan_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 时间规划成功: {result.get('status', 'unknown')}")
            print(f"  计划ID: {result.get('data', {}).get('plan_id', 'N/A')}")
            print(f"  成功状态: {result.get('data', {}).get('success', 'N/A')}")
            return True
        else:
            print(f"✗ 时间规划失败: {response.status_code}")
            print(f"  响应: {response.text[:500]}")
            return False
    except Exception as e:
        print(f"✗ 时间规划异常: {e}")
        return False

def test_meta_cognition_analysis():
    """测试元认知分析API"""
    print("\n测试元认知分析API...")
    
    # 创建认知数据请求
    cognitive_data = {
        "thinking_processes": ["planning", "reasoning", "problem_solving"],
        "cognitive_load": 0.7,
        "attention_focus": "system_analysis",
        "recent_decisions": ["start_api_test", "validate_components"]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/meta-cognition/analyze",
            json={"cognitive_data": cognitive_data},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 元认知分析成功: {result.get('status', 'unknown')}")
            print(f"  分析类型: {result.get('data', {}).get('analysis', 'N/A')}")
            print(f"  健康分数: {result.get('data', {}).get('health_assessment', {}).get('health_score', 'N/A')}")
            return True
        else:
            print(f"✗ 元认知分析失败: {response.status_code}")
            print(f"  响应: {response.text[:500]}")
            return False
    except Exception as e:
        print(f"✗ 元认知分析异常: {e}")
        return False

def test_cross_domain_planning():
    """测试跨域规划API"""
    print("\n测试跨域规划API...")
    
    plan_data = {
        "goal": "创建一个跨平台移动应用",
        "target_domain": "mobile_app_development",
        "available_domains": ["software_development", "ui_design", "mobile_platforms"],
        "constraints": {"budget": "limited", "timeframe": "3_months"},
        "context": {"platform": "cross_platform", "target_users": "general_public"}
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/agi/cross-domain-planning",
            json=plan_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 跨域规划成功: {result.get('status', 'unknown')}")
            print(f"  计划ID: {result.get('data', {}).get('plan_id', 'N/A')}")
            print(f"  成功状态: {result.get('data', {}).get('success', 'N/A')}")
            return True
        else:
            print(f"✗ 跨域规划失败: {response.status_code}")
            print(f"  响应: {response.text[:500]}")
            return False
    except Exception as e:
        print(f"✗ 跨域规划异常: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("主服务器API集成测试")
    print("=" * 60)
    
    tests = [
        ("知识搜索", test_knowledge_search),
        ("集成规划推理", test_plan_with_reasoning),
        ("时间规划", test_temporal_planning),
        ("元认知分析", test_meta_cognition_analysis),
        ("跨域规划", test_cross_domain_planning)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ 测试执行异常: {e}")
            results.append((test_name, False))
        
        time.sleep(0.5)  # 短暂延迟
    
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {test_name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有API集成测试通过！")
    else:
        print("⚠️  部分API集成测试失败")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)