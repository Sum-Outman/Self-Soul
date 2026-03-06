#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复的方法是否有效
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_integrated_planning_reasoning_engine():
    """测试IntegratedPlanningReasoningEngine的_generate_basic_plan方法"""
    print("=== 测试IntegratedPlanningReasoningEngine ===")
    try:
        from core.integrated_planning_reasoning_engine import IntegratedPlanningReasoningEngine
        engine = IntegratedPlanningReasoningEngine()
        
        # 创建一个简单的目标分析
        goal_analysis = {
            "goal_representation": "测试目标：验证修复的方法",
            "overall_complexity_score": 0.6,
            "constraints_analysis": {},
            "temporal_aspects": {}
        }
        
        # 测试_generate_basic_plan方法
        basic_plan = engine._generate_basic_plan(goal_analysis)
        print(f"✓ _generate_basic_plan方法成功调用")
        print(f"  生成计划ID: {basic_plan.get('id')}")
        print(f"  步骤数: {len(basic_plan.get('steps', []))}")
        print(f"  估计时长: {basic_plan.get('estimated_duration')}")
        
        # 测试plan_with_reasoning方法
        result = engine.plan_with_reasoning("测试目标", {"context": "测试"})
        print(f"✓ plan_with_reasoning方法成功调用")
        print(f"  成功: {result.get('success')}")
        
        return True
    except Exception as e:
        print(f"✗ IntegratedPlanningReasoningEngine测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_temporal_reasoning_planner():
    """测试TemporalReasoningPlanner的_create_temporal_plan方法"""
    print("\n=== 测试TemporalReasoningPlanner ===")
    try:
        from core.temporal_reasoning_planner import TemporalReasoningPlanner
        planner = TemporalReasoningPlanner()
        
        # 创建一个简单的计划
        plan = {
            "id": "test_plan_1",
            "goal": "测试时间计划",
            "steps": [
                {
                    "type": "analysis",
                    "description": "分析需求",
                    "estimated_time": 10,
                    "resources": ["cognitive_capacity"]
                }
            ]
        }
        
        # 创建一个简单的调度
        robust_schedule = {
            "scheduled_tasks": [
                {
                    "start_time": 0,
                    "end_time": 10,
                    "duration": 10,
                    "buffer_time": 2,
                    "flexibility_score": 0.8,
                    "critical_path": True
                }
            ],
            "robustness_score": 0.9,
            "total_duration": 10,
            "efficiency_score": 0.85
        }
        
        # 创建一个简单的时间分析
        temporal_analysis = {
            "temporal_constraints": [],
            "temporal_complexity": {"overall_complexity": 0.3},
            "temporal_challenges": [],
            "optimization_opportunities": []
        }
        
        # 测试_create_temporal_plan方法
        temporal_plan = planner._create_temporal_plan(plan, robust_schedule, temporal_analysis)
        print(f"✓ _create_temporal_plan方法成功调用")
        print(f"  生成时间计划ID: {temporal_plan.get('id')}")
        print(f"  原始计划ID: {temporal_plan.get('original_plan_id')}")
        print(f"  健壮性级别: {temporal_plan.get('robustness_level')}")
        
        # 测试generate_temporal_plan方法
        test_plan = {
            "id": "test_temporal",
            "goal": "测试时间规划",
            "steps": [
                {
                    "type": "test",
                    "description": "测试步骤",
                    "estimated_time": 5,
                    "resources": ["test"]
                }
            ]
        }
        
        result = planner.generate_temporal_plan(test_plan, temporal_analysis)
        print(f"✓ generate_temporal_plan方法成功调用")
        print(f"  成功: {result.get('success')}")
        
        return True
    except Exception as e:
        print(f"✗ TemporalReasoningPlanner测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_meta_cognition():
    """测试EnhancedMetaCognition的analyze_system_state方法"""
    print("\n=== 测试EnhancedMetaCognition ===")
    try:
        from core.enhanced_meta_cognition import EnhancedMetaCognition
        meta_cognition = EnhancedMetaCognition()
        
        # 测试analyze_system_state方法
        analysis = meta_cognition.analyze_system_state()
        print(f"✓ analyze_system_state方法成功调用")
        print(f"  分析类型: {analysis.get('analysis_type')}")
        print(f"  健康分数: {analysis.get('health_assessment', {}).get('health_score', 'N/A')}")
        print(f"  改进建议数: {len(analysis.get('improvement_suggestions', []))}")
        
        # 测试其他方法
        system_status = meta_cognition.get_system_status()
        print(f"✓ get_system_status方法成功调用")
        print(f"  系统名称: {system_status.get('system_name')}")
        print(f"  状态: {system_status.get('status')}")
        
        # 测试生成综合报告
        report = meta_cognition.generate_comprehensive_self_report()
        print(f"✓ generate_comprehensive_self_report方法成功调用")
        print(f"  报告时间戳: {report.get('timestamp')}")
        
        return True
    except Exception as e:
        print(f"✗ EnhancedMetaCognition测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_collaboration():
    """测试组件间协作"""
    print("\n=== 测试组件间协作 ===")
    try:
        # 测试组件是否可以协同工作
        from core.integrated_planning_reasoning_engine import create_integrated_planning_reasoning_engine
        from core.temporal_reasoning_planner import create_temporal_reasoning_planner
        from core.enhanced_meta_cognition import EnhancedMetaCognition
        
        # 创建组件实例
        engine = create_integrated_planning_reasoning_engine()
        planner = create_temporal_reasoning_planner()
        meta_cognition = EnhancedMetaCognition()
        
        print("✓ 所有组件成功实例化")
        
        # 模拟一个简单的协作流程
        goal = "测试组件协作：创建一个时间优化的计划并进行元认知分析"
        
        # 步骤1：使用引擎创建计划
        plan_result = engine.plan_with_reasoning(goal, {"test": True})
        print(f"  1. 集成规划推理引擎生成计划: {plan_result.get('success')}")
        
        if plan_result.get('success'):
            plan = plan_result.get('plan', {})
            
            # 步骤2：使用时间规划器优化计划
            temporal_analysis = planner.analyze_temporal_aspects(plan)
            if temporal_analysis.get('success'):
                temporal_plan_result = planner.generate_temporal_plan(plan, temporal_analysis)
                print(f"  2. 时间推理规划器优化计划: {temporal_plan_result.get('success')}")
            
            # 步骤3：使用元认知系统分析过程
            analysis = meta_cognition.analyze_system_state()
            print(f"  3. 元认知系统分析状态: {analysis.get('status')}")
        
        print("✓ 组件协作测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 组件协作测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试修复的方法和组件协作...")
    
    results = []
    
    # 测试各个组件
    results.append(("IntegratedPlanningReasoningEngine", test_integrated_planning_reasoning_engine()))
    results.append(("TemporalReasoningPlanner", test_temporal_reasoning_planner()))
    results.append(("EnhancedMetaCognition", test_enhanced_meta_cognition()))
    results.append(("组件协作", test_component_collaboration()))
    
    # 输出总结
    print("\n" + "="*60)
    print("测试总结:")
    print("="*60)
    
    all_passed = True
    for component_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {component_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("✅ 所有测试通过！修复的方法有效，组件可以正常协作。")
        return 0
    else:
        print("❌ 部分测试失败，需要进一步检查。")
        return 1

if __name__ == "__main__":
    sys.exit(main())