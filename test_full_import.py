#!/usr/bin/env python3
"""全面导入测试脚本"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("测试所有核心组件导入...")

# 测试所有关键导入
try:
    from core.self_model import GoalModel
    print("✓ GoalModel 导入成功")
except Exception as e:
    print(f"✗ GoalModel 导入失败: {e}")

try:
    from core.knowledge_service import get_knowledge_service
    print("✓ get_knowledge_service 导入成功")
except Exception as e:
    print(f"✗ get_knowledge_service 导入失败: {e}")

try:
    from core.advanced_reasoning import EnhancedAdvancedReasoningEngine
    print("✓ EnhancedAdvancedReasoningEngine 导入成功")
except Exception as e:
    print(f"✗ EnhancedAdvancedReasoningEngine 导入失败: {e}")

try:
    from core.temporal_reasoning_planner import create_temporal_reasoning_planner
    print("✓ create_temporal_reasoning_planner 导入成功")
except Exception as e:
    print(f"✗ create_temporal_reasoning_planner 导入失败: {e}")

try:
    from core.cross_domain_planner import create_cross_domain_planner
    print("✓ create_cross_domain_planner 导入成功")
except Exception as e:
    print(f"✗ create_cross_domain_planner 导入失败: {e}")

try:
    from core.self_reflection_optimizer import create_self_reflection_optimizer
    print("✓ create_self_reflection_optimizer 导入成功")
except Exception as e:
    print(f"✗ create_self_reflection_optimizer 导入失败: {e}")

try:
    from core.integrated_planning_reasoning_engine import create_integrated_planning_reasoning_engine
    print("✓ create_integrated_planning_reasoning_engine 导入成功")
except Exception as e:
    print(f"✗ create_integrated_planning_reasoning_engine 导入失败: {e}")

try:
    from core.causal_reasoning_enhancer import create_causal_reasoning_enhancer
    print("✓ create_causal_reasoning_enhancer 导入成功")
except Exception as e:
    print(f"✗ create_causal_reasoning_enhancer 导入失败: {e}")

# 测试AGI组件
try:
    from core.meta_cognition import EnhancedMetaCognition
    print("✓ EnhancedMetaCognition 导入成功")
except Exception as e:
    print(f"✗ EnhancedMetaCognition 导入失败: {e}")

try:
    from core.explainable_ai import ExplainableAI
    print("✓ ExplainableAI 导入成功")
except Exception as e:
    print(f"✗ ExplainableAI 导入失败: {e}")

print("\n测试完成！")