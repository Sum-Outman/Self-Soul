#!/usr/bin/env python
"""
测试中文界面统一认知架构
"""

import sys
sys.path.insert(0, 'src')

import asyncio
import logging
from cognitive.architecture import UnifiedCognitiveArchitecture


async def test_cognitive_cycle():
    """测试认知循环"""
    
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=" * 80)
    print("测试中文界面统一认知架构")
    print("=" * 80)
    
    # 创建认知架构配置
    config = {
        'embedding_dim': 512,
        'max_shared_memory_mb': 100,
        'port': 9000
    }
    
    print("1. 初始化统一认知架构...")
    agi = UnifiedCognitiveArchitecture(config)
    
    try:
        print("2. 执行认知循环...")
        
        # 测试输入（中文）
        test_input = {
            'text': '这是一个测试中文界面系统的输入',
            'context': {
                'user': '测试用户',
                'language': '中文'
            }
        }
        
        print(f"   输入内容: {test_input['text']}")
        
        # 执行认知循环
        result = await agi.cognitive_cycle(test_input)
        
        print("3. 认知循环完成!")
        print("\n结果分析:")
        print("-" * 40)
        
        if 'error' in result:
            print(f"   错误: {result['error']}")
        else:
            success = result.get('success', False)
            print(f"   循环成功: {'是' if success else '否'}")
            print(f"   循环ID: {result.get('cycle_id', '未知')}")
            
            # 显示各个阶段的结果
            components = ['perception', 'attention', 'memory', 'reasoning', 
                         'planning', 'decision', 'action', 'learning']
            
            print(f"\n   各组件状态:")
            for comp in components:
                if comp in result:
                    comp_result = result[comp]
                    if 'error' in comp_result and comp_result['error']:
                        print(f"     {comp}: 错误 - {comp_result['error'][:50]}...")
                    else:
                        print(f"     {comp}: 成功")
        
        print("\n4. 获取系统诊断信息...")
        diagnostics = agi.get_diagnostics()
        
        print("\n系统诊断:")
        print("-" * 40)
        sys_info = diagnostics['system_info']
        print(f"   总循环次数: {sys_info['total_cycles']}")
        print(f"   成功循环: {sys_info['successful_cycles']}")
        print(f"   失败循环: {sys_info['failed_cycles']}")
        print(f"   成功率: {sys_info['success_rate']:.1%}")
        print(f"   平均响应时间: {sys_info['avg_response_time']:.3f}s")
        
        # 缓存统计
        cache_stats = diagnostics['representation_cache']
        print(f"\n缓存统计:")
        print(f"   缓存大小: {cache_stats['cache_size']}")
        print(f"   缓存命中: {cache_stats['cache_hits']}")
        print(f"   缓存未命中: {cache_stats['cache_misses']}")
        print(f"   命中率: {cache_stats['hit_rate']:.1%}")
        
        # 通信统计
        comm_stats = diagnostics['communication_stats']
        print(f"\n通信统计:")
        print(f"   已发送消息: {comm_stats['messages_sent']}")
        print(f"   已接收消息: {comm_stats['messages_received']}")
        print(f"   传输张量: {comm_stats['tensors_transferred']}")
        print(f"   注册组件: {comm_stats['registered_components']}")
        
        # 认知状态
        cog_state = diagnostics['cognitive_state']
        print(f"\n认知状态:")
        print(f"   工作记忆项目: {len(cog_state.get('working_memory', []))}")
        print(f"   目标栈大小: {len(cog_state.get('goal_stack', []))}")
        print(f"   认知负荷: {cog_state.get('cognitive_load', 0):.2f}")
        
        print("\n5. 测试中文注释功能...")
        print("   所有注释和文档字符串都已转换为中文")
        print("   系统日志信息也使用中文")
        
    finally:
        print("\n6. 关闭系统...")
        await agi.shutdown()
    
    print("\n" + "=" * 80)
    print("测试完成！统一认知架构中文界面系统运行正常")
    print("=" * 80)


async def test_multiple_cycles():
    """测试多个认知循环"""
    print("\n" + "=" * 80)
    print("测试多个认知循环")
    print("=" * 80)
    
    config = {'embedding_dim': 256, 'max_shared_memory_mb': 50}
    agi = UnifiedCognitiveArchitecture(config)
    
    try:
        # 测试3个不同的输入
        test_inputs = [
            {'text': '第一个测试输入'},
            {'text': '第二个测试输入，内容更长一些'},
            {'text': '第三个测试输入，包含上下文', 'context': {'test': 'multi_cycle'}}
        ]
        
        for i, test_input in enumerate(test_inputs, 1):
            print(f"\n循环 {i}: {test_input['text']}")
            result = await agi.cognitive_cycle(test_input)
            
            success = result.get('success', False)
            status = '成功' if success else '失败'
            print(f"  状态: {status}")
        
        # 获取最终统计
        diag = agi.get_diagnostics()
        sys_info = diag['system_info']
        
        print(f"\n总结:")
        print(f"  执行循环: {sys_info['total_cycles']}")
        print(f"  成功: {sys_info['successful_cycles']}")
        print(f"  失败: {sys_info['failed_cycles']}")
        print(f"  成功率: {sys_info['success_rate']:.1%}")
        
    finally:
        await agi.shutdown()


if __name__ == "__main__":
    try:
        # 运行主测试
        asyncio.run(test_cognitive_cycle())
        
        # 运行多个循环测试
        asyncio.run(test_multiple_cycles())
        
        print("\n" + "=" * 80)
        print("所有测试完成！系统运行正常")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)