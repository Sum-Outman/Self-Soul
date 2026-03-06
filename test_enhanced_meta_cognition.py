#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试EnhancedMetaCognition组件
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.WARNING)  # 减少日志输出

def test_enhanced_meta_cognition():
    """测试EnhancedMetaCognition"""
    print("测试EnhancedMetaCognition组件...")
    try:
        from core.enhanced_meta_cognition import EnhancedMetaCognition
        print("✓ 导入成功")
        
        # 创建实例
        meta_cognition = EnhancedMetaCognition()
        print("✓ 实例创建成功")
        
        # 测试基本方法
        print("\n测试基本方法:")
        
        # 1. 测试get_system_status
        try:
            status = meta_cognition.get_system_status()
            print("  ✓ get_system_status成功")
            print(f"    状态: {status.get('status')}")
            print(f"    系统名称: {status.get('system_name')}")
        except Exception as e:
            print(f"  ✗ get_system_status失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 2. 测试analyze_system_state
        try:
            analysis = meta_cognition.analyze_system_state()
            print("  ✓ analyze_system_state成功")
            print(f"    分析类型: {analysis.get('analysis_type')}")
            print(f"    健康分数: {analysis.get('health_assessment', {}).get('health_score', 'N/A')}")
        except Exception as e:
            print(f"  ✗ analyze_system_state失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 3. 测试generate_comprehensive_self_report
        try:
            report = meta_cognition.generate_comprehensive_self_report()
            print("  ✓ generate_comprehensive_self_report成功")
            print(f"    报告时间戳: {report.get('timestamp')}")
        except Exception as e:
            print(f"  ✗ generate_comprehensive_self_report失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 4. 测试monitor_thinking_process
        try:
            thought_process = {
                "content": "测试思考过程",
                "context": "测试上下文",
                "type": "analytical"
            }
            monitoring = meta_cognition.monitor_thinking_process(thought_process)
            print("  ✓ monitor_thinking_process成功")
            print(f"    思维质量: {monitoring.get('thinking_quality', {}).get('logical_coherence', 'N/A')}")
        except Exception as e:
            print(f"  ✗ monitor_thinking_process失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 5. 测试regulate_cognition
        try:
            regulation_needs = {
                "focus_improvement": 0.7,
                "creativity_boost": 0.3
            }
            regulation = meta_cognition.regulate_cognition(regulation_needs)
            print("  ✓ regulate_cognition成功")
            print(f"    调节应用: {regulation.get('regulation_applied', False)}")
        except Exception as e:
            print(f"  ✗ regulate_cognition失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n✅ EnhancedMetaCognition所有测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ EnhancedMetaCognition测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_meta_cognition()
    sys.exit(0 if success else 1)