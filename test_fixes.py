#!/usr/bin/env python3
"""
测试修复脚本
1. 测试knowledge/search的GET方法
2. 诊断cross-domain-planning的500错误
"""

import requests
import json
import time

def test_knowledge_search():
    """测试知识搜索GET方法"""
    print("=== 测试knowledge/search GET方法 ===")
    
    base_url = "http://localhost:8000"
    
    # 测试带查询参数的GET请求
    try:
        start_time = time.time()
        response = requests.get(f"{base_url}/api/knowledge/search?query=人工智能", timeout=10)
        elapsed_time = time.time() - start_time
        
        print(f"GET /api/knowledge/search?query=人工智能: {response.status_code} ({elapsed_time:.2f}s)")
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('data', {}).get('results', [])
            print(f"找到 {len(results)} 个结果")
            
            if results:
                for i, result in enumerate(results[:3]):
                    concept = result.get("concept", "N/A")
                    domain = result.get("domain", "unknown")
                    print(f"  {i+1}. {concept} (领域: {domain})")
        else:
            print(f"响应: {response.text[:200]}")
            
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试带domain参数的GET请求
    print("\n测试带domain参数的搜索:")
    try:
        response = requests.get(f"{base_url}/api/knowledge/search?domain=computer_science", timeout=10)
        print(f"GET /api/knowledge/search?domain=computer_science: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('data', {}).get('results', [])
            print(f"找到 {len(results)} 个结果")
    except Exception as e:
        print(f"错误: {e}")

def test_cross_domain_planning():
    """测试跨领域规划端点"""
    print("\n=== 测试cross-domain-planning端点 ===")
    
    base_url = "http://localhost:8000"
    
    # 测试1: 简单请求
    print("测试1: 简单跨领域规划请求")
    try:
        data = {
            "goal": "测试跨领域规划目标",
            "target_domain": "test_domain"
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/api/agi/cross-domain-planning", 
                                json=data, timeout=15)
        elapsed_time = time.time() - start_time
        
        print(f"POST /api/agi/cross-domain-planning: {response.status_code} ({elapsed_time:.2f}s)")
        
        if response.status_code != 200:
            print(f"错误状态码: {response.status_code}")
            print(f"错误响应前500字符: {response.text[:500]}")
            
            # 尝试获取更详细的错误信息
            try:
                error_data = response.json()
                print(f"错误JSON: {json.dumps(error_data, indent=2, ensure_ascii=False)[:500]}")
            except:
                pass
        else:
            data = response.json()
            print(f"成功响应 (前300字符): {json.dumps(data, ensure_ascii=False)[:300]}...")
            
    except Exception as e:
        print(f"请求错误: {e}")
    
    # 测试2: 测试请求（应该返回模拟响应）
    print("\n测试2: 测试请求（应返回模拟响应）")
    try:
        data = {
            "goal": "test goal for mobile_app_development",
            "target_domain": "mobile_app_development"
        }
        
        response = requests.post(f"{base_url}/api/agi/cross-domain-planning", 
                                json=data, timeout=10)
        
        print(f"POST /api/agi/cross-domain-planning (测试): {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"测试响应成功: {data.get('status', 'unknown')}")
        else:
            print(f"测试请求也失败: {response.text[:200]}")
            
    except Exception as e:
        print(f"测试请求错误: {e}")

def check_component_status():
    """检查组件状态"""
    print("\n=== 检查组件状态 ===")
    
    base_url = "http://localhost:8000"
    
    try:
        response = requests.get(f"{base_url}/api/system/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            components = data.get('components_loaded', {})
            
            print(f"cross_domain_planner状态: {components.get('cross_domain_planner', 'unknown')}")
            print(f"knowledge_service状态: {components.get('knowledge_service', 'unknown')}")
            
            # 检查是否有其他相关信息
            print(f"总体状态: {data.get('status', 'unknown')}")
            print(f"系统: {data.get('system', 'unknown')}")
            
    except Exception as e:
        print(f"检查组件状态错误: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("服务器问题诊断脚本")
    print(f"运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # 检查组件状态
        check_component_status()
        
        # 测试knowledge/search
        test_knowledge_search()
        
        # 测试cross-domain-planning
        test_cross_domain_planning()
        
        print("\n" + "=" * 60)
        print("诊断完成")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n诊断被用户中断")
    except Exception as e:
        print(f"\n诊断过程中出现错误: {e}")

if __name__ == "__main__":
    main()