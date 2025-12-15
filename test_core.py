# -*- coding: utf-8 -*-
"""
Self Soul AGI系统核心功能测试脚本
简化版本，仅测试核心协调器功能
"""

import os
import sys
import traceback

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# 添加core目录到Python路径
core_path = os.path.join(project_root, 'core')
sys.path.append(core_path)

# 添加models目录到Python路径
models_path = os.path.join(core_path, 'models')
sys.path.append(models_path)

# 首先测试简单的模块导入，验证基本功能
try:
    print("=== 测试AGI系统核心功能 ===")
    print(f"Python版本: {sys.version}")
    print(f"项目根目录: {project_root}")
    
    # 测试错误处理模块
    from core.error_handling import error_handler
    print("✓ 错误处理模块导入成功")
    
    # 测试模型注册表
    from core.model_registry import ModelRegistry
    model_registry = ModelRegistry()
    print("✓ 模型注册表初始化成功")
    print(f"  - 已注册模型数: {len(model_registry.models)}")
    
    # 测试AGI协调器基本功能
    from core.agi_coordinator import AGICoordinator
    
    # 创建AGI系统实例（简化版本，不加载完整模型）
    print("正在初始化AGI协调器...")
    agi_system = AGICoordinator(from_scratch=True)
    print("✓ AGI协调器初始化成功")
    
    # 测试系统状态
    system_status = agi_system.get_system_status()
    print("✓ 系统状态获取成功")
    print(f"  - 系统状态: {system_status.get('status')}")
    print(f"  - AGI水平: {system_status.get('agi_level'):.2f}")
    print(f"  - 已加载模型数: {len(system_status.get('active_models', []))}")
    
    # 测试任务协调
    print("正在测试任务协调...")
    task_result = agi_system.coordinate_task('简单测试任务')
    print("✓ 任务协调测试完成")
    
    print("\n=== 核心功能测试完成 ===")
    print("AGI系统核心组件运行正常!")
    
except Exception as e:
    print(f"\n× 测试失败: {str(e)}")
    print("错误详情:")
    traceback.print_exc()
    sys.exit(1)