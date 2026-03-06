#!/usr/bin/env python3
"""集成测试脚本 - 测试所有从渐进式服务器集成的功能"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== 集成完整性测试 ===\n")

# 测试所有从渐进式服务器集成的组件
components_to_test = [
    ("GoalModel", "from core.self_model import GoalModel"),
    ("KnowledgeService", "from core.knowledge_service import get_knowledge_service"),
    ("EnhancedAdvancedReasoningEngine", "from core.advanced_reasoning import EnhancedAdvancedReasoningEngine"),
    ("TemporalReasoningPlanner", "from core.temporal_reasoning_planner import create_temporal_reasoning_planner"),
    ("CrossDomainPlanner", "from core.cross_domain_planner import create_cross_domain_planner"),
    ("SelfReflectionOptimizer", "from core.self_reflection_optimizer import create_self_reflection_optimizer"),
    ("IntegratedPlanningEngine", "from core.integrated_planning_reasoning_engine import create_integrated_planning_reasoning_engine"),
    ("CausalReasoningEnhancer", "from core.causal_reasoning_enhancer import create_causal_reasoning_enhancer"),
    ("EnhancedMetaCognition", "from core.enhanced_meta_cognition import EnhancedMetaCognition"),
    ("ExplainableAI", "from core.explainable_ai import ExplainableAI"),
]

print("1. 测试组件导入:")
all_passed = True
for component_name, import_stmt in components_to_test:
    try:
        exec(import_stmt)
        print(f"  ✓ {component_name} 导入成功")
    except Exception as e:
        print(f"  ✗ {component_name} 导入失败: {e}")
        all_passed = False

print("\n2. 测试API端点路由:")
try:
    from core.main import app
    routes = []
    for route in app.routes:
        if hasattr(route, "path"):
            routes.append(route.path)
    
    print(f"  总路由数: {len(routes)}")
    
    # 检查从渐进式服务器集成的关键API端点
    progressive_endpoints = [
        "/api/agi/plan-with-reasoning",
        "/api/agi/analyze-causality",
        "/api/agi/temporal-planning",
        "/api/agi/cross-domain-planning",
        "/api/agi/self-reflection",
        "/api/goals",
        "/api/goals/critical",
        "/api/knowledge/domains",
        "/api/knowledge/search",
        "/api/knowledge/concept/{domain}/{concept_id}",
        "/api/knowledge/statistics",
        "/api/meta-cognition/analyze",
        "/api/meta-cognition/status",
        "/api/explainable-ai/explain",
        "/api/explainable-ai/capabilities",
        "/api/monitoring/data",
        "/api/system/status"
    ]
    
    missing_endpoints = []
    for endpoint in progressive_endpoints:
        if not any(endpoint in r for r in routes):
            missing_endpoints.append(endpoint)
    
    if missing_endpoints:
        print(f"  ✗ 缺失 {len(missing_endpoints)} 个端点:")
        for endpoint in missing_endpoints:
            print(f"    - {endpoint}")
        all_passed = False
    else:
        print(f"  ✓ 所有 {len(progressive_endpoints)} 个渐进式服务器端点都存在")
        
except Exception as e:
    print(f"  ✗ API端点测试失败: {e}")
    all_passed = False

print("\n3. 测试全局变量定义:")
try:
    # 检查主要全局变量是否在模块中定义
    import core.main as main_module
    
    # 检查这些变量是否在模块中定义（即使值为None）
    required_vars = [
        'goal_model', 'knowledge_service', 'advanced_reasoning_engine',
        'temporal_reasoning_planner', 'cross_domain_planner', 'self_reflection_optimizer',
        'integrated_planning_engine', 'causal_reasoning_enhancer',
        'enhanced_meta_cognition', 'explainable_ai', 'system_monitor'
    ]
    
    missing_vars = []
    for var_name in required_vars:
        if not hasattr(main_module, var_name):
            missing_vars.append(var_name)
    
    if missing_vars:
        print(f"  ✗ 缺失全局变量: {missing_vars}")
        all_passed = False
    else:
        print("  ✓ 所有必需全局变量都已定义")
    
except Exception as e:
    print(f"  ✗ 全局变量测试失败: {e}")
    all_passed = False

print("\n=== 测试结果 ===")
if all_passed:
    print("✅ 所有集成测试通过！")
else:
    print("❌ 某些测试失败，需要修复。")
    sys.exit(1)