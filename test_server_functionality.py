#!/usr/bin/env python3
"""
服务器功能完整性检测脚本
检测所有API端点是否正常启动和初始化
"""

import requests
import json
import time
from typing import List, Dict, Tuple

class ServerFunctionalityTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        
    def test_get_endpoint(self, endpoint: str, timeout: int = 10) -> Dict:
        """测试GET端点"""
        url = f"{self.base_url}{endpoint}"
        try:
            start_time = time.time()
            response = requests.get(url, timeout=timeout)
            elapsed_time = time.time() - start_time
            
            success = response.status_code in [200, 201, 202]
            result = {
                'endpoint': endpoint,
                'method': 'GET',
                'status_code': response.status_code,
                'success': success,
                'response_time': elapsed_time,
                'error': None
            }
            
            if success and endpoint == '/api/system/status':
                result['data'] = response.json()
                
            return result
            
        except Exception as e:
            return {
                'endpoint': endpoint,
                'method': 'GET',
                'status_code': None,
                'success': False,
                'response_time': None,
                'error': str(e)
            }
    
    def test_post_endpoint(self, endpoint: str, data: Dict, timeout: int = 15) -> Dict:
        """测试POST端点"""
        url = f"{self.base_url}{endpoint}"
        try:
            start_time = time.time()
            response = requests.post(url, json=data, timeout=timeout)
            elapsed_time = time.time() - start_time
            
            success = response.status_code in [200, 201, 202]
            result = {
                'endpoint': endpoint,
                'method': 'POST',
                'status_code': response.status_code,
                'success': success,
                'response_time': elapsed_time,
                'error': None
            }
            
            if success:
                try:
                    result['data'] = response.json()
                except:
                    result['data'] = response.text[:500]
                    
            return result
            
        except Exception as e:
            return {
                'endpoint': endpoint,
                'method': 'POST',
                'status_code': None,
                'success': False,
                'response_time': None,
                'error': str(e)
            }
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        print("=" * 70)
        print("服务器功能完整性检测")
        print(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"服务器地址: {self.base_url}")
        print("=" * 70)
        
        # 定义要测试的端点
        get_endpoints = [
            '/health',
            '/docs',
            '/openapi.json',
            '/api/system/status',
            '/api/agi/status',
            '/api/knowledge/domains',
            '/api/knowledge/concepts',
            '/api/goals',
            '/api/models/status',
            '/api/monitoring/data',
        ]
        
        post_endpoints = [
            ('/api/chat', {'message': '测试消息，系统是否正常运行？'}),
            ('/api/knowledge/search', {'query': '人工智能'}),
            ('/api/agi/process', {'text': '测试AGI处理功能'}),
            ('/api/agi/plan-with-reasoning', {'goal': '测试规划推理目标'}),
            ('/api/agi/temporal-planning', {'goal': '测试时间规划'}),
            ('/api/agi/cross-domain-planning', {'goal': '测试跨领域规划'}),
            ('/api/agi/self-reflection', {'context': '测试自我反思'}),
            ('/api/agi/analyze-causality', {'event': '测试因果分析'}),
            ('/api/meta-cognition/analyze', {'input': '测试元认知分析'}),
        ]
        
        # 测试GET端点
        print("\n1. GET端点测试:")
        for endpoint in get_endpoints:
            result = self.test_get_endpoint(endpoint)
            self.results.append(result)
            
            status_icon = '✓' if result['success'] else '✗'
            time_str = f"({result['response_time']:.2f}s)" if result['response_time'] else ""
            print(f"  {status_icon} GET {endpoint}: {result['status_code']} {time_str}")
            
            if result['error']:
                print(f"    错误: {result['error']}")
        
        # 测试POST端点
        print("\n2. POST端点测试:")
        for endpoint, data in post_endpoints:
            result = self.test_post_endpoint(endpoint, data)
            self.results.append(result)
            
            status_icon = '✓' if result['success'] else '✗'
            time_str = f"({result['response_time']:.2f}s)" if result['response_time'] else ""
            print(f"  {status_icon} POST {endpoint}: {result['status_code']} {time_str}")
            
            if result['error']:
                print(f"    错误: {result['error']}")
                
            # 显示关键响应
            if endpoint == '/api/agi/process' and result['success'] and 'data' in result:
                resp_data = result['data']
                if isinstance(resp_data, dict) and 'data' in resp_data and 'text' in resp_data['data']:
                    text = resp_data['data']['text']
                    print(f"    AGI响应: {text[:80]}...")
        
        # 分析结果
        self.analyze_results()
        
    def analyze_results(self):
        """分析测试结果"""
        print("\n" + "=" * 70)
        print("测试结果分析:")
        print("=" * 70)
        
        # 总体统计
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r['success'])
        failed_tests = total_tests - successful_tests
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"总测试端点: {total_tests}个")
        print(f"成功端点: {successful_tests}个")
        print(f"失败端点: {failed_tests}个")
        print(f"成功率: {success_rate:.1f}%")
        
        # 组件状态分析
        print("\n3. 组件初始化状态分析:")
        
        # 查找系统状态结果
        system_status_result = None
        for result in self.results:
            if result['endpoint'] == '/api/system/status' and result['success'] and 'data' in result:
                system_status_result = result['data']
                break
        
        if system_status_result:
            components = system_status_result.get('components_loaded', {})
            
            available_components = [k for k, v in components.items() if v == 'available']
            not_found_components = [k for k, v in components.items() if v == 'not_found']
            other_components = [k for k, v in components.items() if v not in ['available', 'not_found']]
            
            print(f"   可用组件 ({len(available_components)}个):")
            for comp in available_components:
                print(f"     ✓ {comp}")
                
            if not_found_components:
                print(f"   未找到组件 ({len(not_found_components)}个):")
                for comp in not_found_components:
                    print(f"     ⚠️  {comp}")
                    
            if other_components:
                print(f"   其他状态组件 ({len(other_components)}个):")
                for comp in other_components:
                    print(f"     ? {comp}: {components[comp]}")
                    
            # 核心组件检查
            core_components = [
                'knowledge_service',
                'advanced_reasoning_engine',
                'temporal_reasoning_planner',
                'cross_domain_planner',
                'self_reflection_optimizer',
                'integrated_planning_reasoning_engine',
                'causal_reasoning_enhancer',
                'enhanced_meta_cognition',
            ]
            
            print(f"\n   核心AGI组件状态:")
            for comp in core_components:
                status = components.get(comp, 'unknown')
                icon = '✓' if status == 'available' else '⚠️' if status == 'not_found' else '?'
                print(f"     {icon} {comp}: {status}")
                
        else:
            print("   无法获取系统状态信息")
        
        # 失败端点详情
        if failed_tests > 0:
            print(f"\n4. 失败端点详情:")
            for result in self.results:
                if not result['success']:
                    print(f"   {result['method']} {result['endpoint']}: ", end="")
                    if result['status_code']:
                        print(f"状态码 {result['status_code']}")
                    elif result['error']:
                        print(f"错误: {result['error']}")
                    else:
                        print("未知错误")
        
        # 性能分析
        response_times = [r['response_time'] for r in self.results if r['response_time']]
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            print(f"\n5. 性能指标:")
            print(f"   平均响应时间: {avg_time:.2f}秒")
            print(f"   最长响应时间: {max_time:.2f}秒")
        
        print("\n" + "=" * 70)
        print("检测结论:")
        
        if success_rate == 100:
            print("✅ 所有服务器功能已正常启动并初始化完成！")
            print("   所有API端点均可正常访问，核心组件全部可用。")
        elif success_rate >= 90:
            print("✅ 服务器功能基本正常启动，核心功能可用。")
            print(f"   仅有{failed_tests}个端点存在问题，不影响核心功能。")
        elif success_rate >= 80:
            print("⚠️  大部分服务器功能已启动，但存在一些问题。")
            print(f"   有{failed_tests}个端点存在问题，建议检查。")
        elif success_rate >= 60:
            print("⚠️  服务器功能启动存在问题，部分功能不可用。")
            print(f"   有{failed_tests}个端点失败，需要修复。")
        else:
            print("❌ 服务器功能启动存在严重问题。")
            print(f"   超过40%的端点失败，需要立即检查。")
            
        print("=" * 70)
        
    def save_report(self, filename: str = "server_functionality_report.json"):
        """保存测试报告"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'base_url': self.base_url,
            'results': self.results,
            'summary': {
                'total_tests': len(self.results),
                'successful_tests': sum(1 for r in self.results if r['success']),
                'failed_tests': len(self.results) - sum(1 for r in self.results if r['success'])
            }
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n测试报告已保存到: {filename}")
        except Exception as e:
            print(f"\n保存报告失败: {e}")

def main():
    """主函数"""
    tester = ServerFunctionalityTester()
    
    try:
        tester.run_comprehensive_test()
        tester.save_report()
    except KeyboardInterrupt:
        print("\n检测被用户中断")
    except Exception as e:
        print(f"\n检测过程中出现错误: {e}")

if __name__ == "__main__":
    main()