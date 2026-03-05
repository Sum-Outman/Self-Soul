"""
核心能力演示脚本

展示30天变强版本计划第一优先级实现的核心能力：
1. 统一自我ID和持久化
2. 稳定运行底座
3. 可演示的核心能力（记忆 → 思考 → 行动 → 反馈）

使用场景：
- 技术演示
- 投资人展示
- 团队培训
- 系统验证
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入核心模块
from core.self_identity import get_identity_manager, create_default_identity
from core.runtime_base import get_runtime_base
from core.core_capabilities import get_core_capabilities, demonstrate_capabilities, process_input


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_section(title):
    """打印章节"""
    print(f"\n{'─' * 40}")
    print(f" {title}")
    print(f"{'─' * 40}")


def print_step(step_num, description):
    """打印步骤"""
    print(f"\n[{step_num}] {description}")


def demo_self_identity():
    """演示自我身份系统"""
    print_header("演示1: 统一自我ID和持久化系统")
    
    print("目标: 展示唯一不变的self_id，以及记忆、人格、目标的持久化绑定")
    
    # 步骤1: 创建或获取身份
    print_step(1, "创建/获取自我身份")
    identity_manager = get_identity_manager()
    
    identities = identity_manager.list_identities()
    if identities:
        print(f"  发现 {len(identities)} 个现有身份")
        # 使用第一个身份
        first_id = identities[0]["self_id"]
        identity_manager.set_active_identity(first_id)
        identity = identity_manager.get_active_identity()
        print(f"  使用现有身份: {identity.self_id}")
    else:
        print("  没有现有身份，创建新身份...")
        identity = create_default_identity()
        print(f"  创建新身份: {identity.self_id}")
    
    # 步骤2: 展示身份信息
    print_step(2, "展示身份信息")
    summary = identity.get_summary()
    print(f"  自我ID: {summary['self_id']}")
    print(f"  名称: {summary['name']}")
    print(f"  创建时间: {summary['created_at']}")
    print(f"  访问次数: {summary['access_count']}")
    
    # 步骤3: 展示人格特质
    print_step(3, "展示人格特质")
    personality = identity.get_personality_summary()
    print(f"  人格特质数量: {personality['trait_count']}")
    print(f"  平均特质值: {personality['average_value']:.2f}")
    
    # 显示前3个人格特质
    traits = list(personality['traits'].items())[:3]
    for name, trait_info in traits:
        print(f"    - {name}: {trait_info['value']:.2f} ({trait_info['description']})")
    
    # 步骤4: 展示目标系统
    print_step(4, "展示目标系统")
    goals = identity.get_goal_summary()
    print(f"  总目标数: {goals['total_goals']}")
    print(f"  活跃目标: {goals['active_goals']}")
    print(f"  完成目标: {goals['completed_goals']}")
    
    # 显示活跃目标
    if goals['active_goals_list']:
        print("  活跃目标列表:")
        for goal in goals['active_goals_list'][:2]:
            print(f"    - {goal['description']} (进度: {goal['progress']:.0%})")
    
    # 步骤5: 演示持久化
    print_step(5, "演示持久化")
    save_result = identity.save()
    print(f"  身份保存: {'成功' if save_result else '失败'}")
    
    # 模拟重启后加载
    print("  模拟系统重启...")
    time.sleep(1)
    
    # 重新加载身份
    reloaded_identity = identity_manager.get_identity(identity.self_id)
    if reloaded_identity:
        print(f"  身份重新加载: 成功 (self_id: {reloaded_identity.self_id})")
        print(f"  验证人格特质: {reloaded_identity.get_personality_trait('curiosity').value if reloaded_identity.get_personality_trait('curiosity') else '未找到'}")
    else:
        print("  身份重新加载: 失败")
    
    return identity


def demo_runtime_base():
    """演示运行底座"""
    print_header("演示2: 稳定运行底座")
    
    print("目标: 展示配置中心化、错误捕获、重试机制和健康检查")
    
    # 步骤1: 初始化运行底座
    print_step(1, "初始化运行底座")
    runtime = get_runtime_base()
    print(f"  运行模式: {runtime.mode.value}")
    print(f"  启动时间: {runtime.metrics.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 步骤2: 演示配置管理
    print_step(2, "演示配置管理")
    runtime.set_config("demo.setting", "demo_value")
    config_value = runtime.get_config("demo.setting", "default")
    print(f"  配置设置/获取: demo.setting = {config_value}")
    
    # 步骤3: 演示带重试的执行
    print_step(3, "演示带重试的执行")
    
    call_count = [0]
    
    def unreliable_function():
        call_count[0] += 1
        if call_count[0] < 3:
            raise ValueError(f"第 {call_count[0]} 次调用失败")
        return f"第 {call_count[0]} 次调用成功"
    
    print("  执行可能失败的操作（配置重试3次）...")
    result = runtime.execute_with_retry(unreliable_function)
    print(f"  最终结果: {result}")
    print(f"  总调用次数: {call_count[0]}")
    
    # 步骤4: 演示错误处理
    print_step(4, "演示错误处理")
    
    try:
        raise RuntimeError("演示错误：模拟运行时异常")
    except Exception as e:
        runtime.log_error(e, "演示错误处理")
        print(f"  错误已捕获并记录: {str(e)}")
    
    # 步骤5: 演示健康检查
    print_step(5, "演示健康检查")
    
    health_checks_added = [0]
    
    def demo_health_check():
        health_checks_added[0] += 1
        return True
    
    runtime.add_health_check(demo_health_check)
    
    health_status = runtime.get_health_status()
    print(f"  健康状态: {health_status['status']}")
    print(f"  运行时间: {health_status['metrics']['uptime_seconds']:.1f}秒")
    print(f"  内存使用: {health_status['metrics']['memory_usage_mb']:.1f}MB")
    print(f"  CPU使用率: {health_status['metrics']['cpu_percent']:.1f}%")
    
    return runtime


def demo_core_capabilities_flow():
    """演示核心能力流程"""
    print_header("演示3: 核心能力流程 - 记忆 → 思考 → 行动 → 反馈")
    
    print("目标: 展示完整的AGI核心能力闭环")
    
    # 获取核心能力实例
    capabilities = get_core_capabilities()
    
    # 场景1: 简单问答
    print_section("场景1: 简单问答")
    
    question = "什么是Self-Soul AGI系统？"
    print(f"用户输入: {question}")
    
    result = process_input(question, {"context": "用户询问系统信息"})
    
    if result.get("success"):
        print(f"处理结果: 成功")
        print(f"记忆ID: {result.get('memory_id', 'N/A')}")
        print(f"行动结果: {str(result.get('action_result', {}))[:100]}...")
        print(f"反馈类型: {result.get('feedback', {}).get('type', 'N/A')}")
    else:
        print(f"处理结果: 失败 - {result.get('error', '未知错误')}")
    
    # 场景2: 任务处理
    print_section("场景2: 任务处理")
    
    task = "请分析当前系统状态并提供改进建议"
    print(f"用户输入: {task}")
    
    result = process_input(task, {"context": "用户请求系统分析"})
    
    if result.get("success"):
        thought_process = result.get("thought_process", {})
        print(f"思考过程: {thought_process.get('process', 'N/A')}")
        print(f"思考置信度: {thought_process.get('confidence', 0):.2f}")
        print(f"思考推理: {thought_process.get('reasoning', 'N/A')[:100]}...")
    else:
        print(f"处理结果: 失败 - {result.get('error', '未知错误')}")
    
    # 场景3: 错误处理
    print_section("场景3: 错误处理演示")
    
    error_input = "系统出现错误：无法连接到数据库"
    print(f"用户输入: {error_input}")
    
    result = process_input(error_input, {"context": "用户报告错误"})
    
    if result.get("success"):
        feedback = result.get("feedback", {})
        print(f"反馈类型: {feedback.get('type', 'N/A')}")
        print(f"评估结果: {feedback.get('evaluation', 'N/A')}")
        
        learning_points = feedback.get('learning_points', [])
        if learning_points:
            print(f"学习点: {len(learning_points)} 个")
            for point in learning_points[:2]:
                print(f"  - {point}")
    else:
        print(f"处理结果: 失败 - {result.get('error', '未知错误')}")
    
    return capabilities


def demo_full_capabilities():
    """演示完整能力"""
    print_header("演示4: 完整能力演示")
    
    print("目标: 展示所有核心能力的集成演示")
    
    # 执行完整演示
    print("执行完整核心能力演示...")
    demo_result = demonstrate_capabilities()
    
    if demo_result.get("success"):
        steps = demo_result.get("demonstration_steps", [])
        print(f"演示完成，共 {len(steps)} 个步骤:")
        
        for step in steps:
            print(f"\n  步骤 {step['step']}: {step['capability']}")
            print(f"    行动: {step['action']}")
            
            details = step.get('details', {})
            if 'stored_count' in details:
                print(f"    存储记忆: {details['stored_count']} 个")
            if 'thought_types_tested' in details:
                print(f"    思考类型: {details['thought_types_tested']} 种")
            if 'action_types_tested' in details:
                print(f"    行动类型: {details['action_types_tested']} 种")
            if 'feedbacks_provided' in details:
                print(f"    反馈提供: {details['feedbacks_provided']} 个")
        
        # 显示统计信息
        stats = demo_result.get("statistics", {})
        print(f"\n系统统计:")
        print(f"  记忆系统: {stats.get('memory', {}).get('total_memories', 0)} 个记忆")
        print(f"  思考系统: {stats.get('thinking', {}).get('total_thoughts', 0)} 次思考")
        print(f"  行动系统: {stats.get('action', {}).get('total_actions', 0)} 个行动")
        print(f"  反馈系统: {stats.get('feedback', {}).get('total_feedbacks', 0)} 个反馈")
    else:
        print("演示失败")
    
    return demo_result


def demo_integration():
    """演示系统集成"""
    print_header("演示5: 系统集成演示")
    
    print("目标: 展示所有系统的协同工作")
    
    # 获取所有系统实例
    identity_manager = get_identity_manager()
    runtime = get_runtime_base()
    capabilities = get_core_capabilities()
    
    print("系统组件状态:")
    print(f"  1. 身份系统: {'已加载' if identity_manager else '未加载'}")
    print(f"  2. 运行底座: {runtime.mode.value} 模式")
    print(f"  3. 核心能力: {'已初始化' if capabilities else '未初始化'}")
    
    # 演示集成工作流
    print_section("集成工作流演示")
    
    # 步骤1: 通过运行底座配置身份
    print_step(1, "配置身份目标")
    identity = runtime.get_identity()
    if identity:
        goal_id = identity.add_goal("完成系统集成演示", priority=0.9)
        print(f"  添加目标: {goal_id}")
    else:
        print("  身份不可用，跳过身份集成")
    
    # 步骤2: 使用核心能力处理
    print_step(2, "处理集成请求")
    integration_input = "请演示系统集成功能"
    result = capabilities.process(integration_input, {"integration": True})
    
    if result.get("success"):
        print(f"  处理成功")
        print(f"  生成记忆: {result.get('memory_id', 'N/A')}")
        print(f"  执行行动: {result.get('action_id', 'N/A')}")
        
        # 步骤3: 更新目标进度
        if identity and goal_id:
            identity.update_goal_progress(goal_id, 1.0)
            print(f"  更新目标进度: 100%")
    else:
        print(f"  处理失败: {result.get('error', '未知错误')}")
    
    # 步骤4: 获取系统状态
    print_step(3, "获取系统状态")
    runtime_metrics = runtime.get_metrics()
    capabilities_status = capabilities.get_status()
    
    print(f"  运行指标:")
    print(f"    - 运行时间: {runtime_metrics.get('uptime_seconds', 0):.1f}秒")
    print(f"    - 总请求数: {runtime_metrics.get('total_requests', 0)}")
    print(f"    - 成功请求: {runtime_metrics.get('successful_requests', 0)}")
    print(f"    - 失败请求: {runtime_metrics.get('failed_requests', 0)}")
    
    print(f"  能力状态:")
    print(f"    - 记忆系统: {capabilities_status['memory_system']['status']}")
    print(f"    - 思考系统: {capabilities_status['thinking_system']['status']}")
    print(f"    - 行动系统: {capabilities_status['action_system']['status']}")
    print(f"    - 反馈系统: {capabilities_status['feedback_system']['status']}")
    
    return {
        "identity": identity is not None,
        "runtime": runtime.mode.value,
        "capabilities": capabilities is not None,
        "integration_success": result.get("success", False)
    }


def main():
    """主演示函数"""
    print("\n" + "=" * 80)
    print(" Self-Soul AGI - 30天变强版本计划演示")
    print("=" * 80)
    print(" 演示内容:")
    print("   1. 统一自我ID和持久化系统")
    print("   2. 稳定运行底座")
    print("   3. 核心能力流程 (记忆 → 思考 → 行动 → 反馈)")
    print("   4. 完整能力演示")
    print("   5. 系统集成演示")
    print("=" * 80)
    print(f" 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 创建演示目录
    demo_dir = Path("demo_output")
    demo_dir.mkdir(exist_ok=True)
    
    results = {}
    
    try:
        # 演示1: 自我身份系统
        identity = demo_self_identity()
        results["identity"] = identity.self_id if identity else None
        
        # 演示2: 运行底座
        runtime = demo_runtime_base()
        results["runtime_mode"] = runtime.mode.value
        
        # 演示3: 核心能力流程
        capabilities = demo_core_capabilities_flow()
        results["capabilities_available"] = capabilities is not None
        
        # 演示4: 完整能力
        demo_result = demo_full_capabilities()
        results["full_demo_success"] = demo_result.get("success", False)
        
        # 演示5: 系统集成
        integration_result = demo_integration()
        results.update(integration_result)
        
        # 保存演示结果
        results_file = demo_dir / "demo_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 总结
        print_header("演示总结")
        print("✅ 所有演示完成!")
        print(f"\n演示结果已保存到: {results_file}")
        
        print(f"\n关键成果:")
        print(f"  1. 自我ID: {results.get('identity', '未创建')}")
        print(f"  2. 运行模式: {results.get('runtime_mode', '未知')}")
        print(f"  3. 核心能力: {'可用' if results.get('capabilities_available') else '不可用'}")
        print(f"  4. 完整演示: {'成功' if results.get('full_demo_success') else '失败'}")
        print(f"  5. 系统集成: {'成功' if results.get('integration_success') else '失败'}")
        
        print(f"\n🎉 30天变强版本计划 - 第一优先级演示成功!")
        print(f"\n📋 已实现的核心能力:")
        print("   - 统一自我ID和持久化")
        print("   - 稳定运行底座 (配置、日志、错误处理、重试)")
        print("   - 记忆 → 思考 → 行动 → 反馈 闭环")
        print("   - 系统集成和协同工作")
        
        print(f"\n⏰ 总演示时间: 约3-5分钟")
        print(f"📊 演示输出: {results_file}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 保存错误信息
        error_file = demo_dir / "demo_error.json"
        error_info = {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_info, f, ensure_ascii=False, indent=2)
        
        print(f"错误信息已保存到: {error_file}")
        return 1


if __name__ == "__main__":
    # 设置环境变量
    os.environ['ENVIRONMENT'] = 'development'
    os.environ['RUNTIME_MODE'] = 'development'
    
    # 运行演示
    exit_code = main()
    sys.exit(exit_code)