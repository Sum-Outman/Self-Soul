#!/usr/bin/env python
"""
统一认知架构演示脚本。

此脚本演示统一认知架构的基本功能，
无需启动完整的API服务器。
"""

import asyncio
import logging
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_cognitive_cycle():
    """演示完整的认知循环"""
    from cognitive.architecture import UnifiedCognitiveArchitecture
    
    print("\n" + "="*80)
    print("演示: 统一认知架构")
    print("="*80)
    
    # 使用最小配置创建架构
    config = {
        'embedding_dim': 512,
        'max_shared_memory_mb': 100
    }
    
    print("1. 正在初始化统一认知架构...")
    agi = UnifiedCognitiveArchitecture(config)
    
    try:
        print("2. 正在运行认知循环...")
        
        # 测试输入
        test_input = {
            'text': "法国的首都是什么？",
            'context': {
                'user_id': 'demo_user',
                'session_id': 'demo_session'
            }
        }
        
        print(f"   输入: {test_input['text']}")
        
        # 执行认知循环
        result = await agi.cognitive_cycle(test_input)
        
        print("3. 认知循环完成！")
        print("\n结果:")
        print("-" * 40)
        
        if 'error' in result:
            print(f"   错误: {result['error']}")
        else:
            print(f"   输出: {result.get('output', {})}")
            print(f"   循环时间: {result.get('performance', {}).get('cycle_time', 0):.3f}秒")
            print(f"   统一表征形状: {result.get('performance', {}).get('unified_representation_shape', 'N/A')}")
        
        print("\n4. 正在获取系统诊断信息...")
        diagnostics = agi.get_diagnostics()
        
        print("\n诊断信息:")
        print("-" * 40)
        print(f"   总循环数: {diagnostics['system_info']['total_cycles']}")
        print(f"   平均响应时间: {diagnostics['system_info']['avg_response_time']:.3f}秒")
        print(f"   成功率: {diagnostics['system_info']['success_rate']:.1%}")
        
        # 缓存统计
        cache_stats = diagnostics['representation_cache']
        print(f"\n缓存统计:")
        print(f"   缓存大小: {cache_stats['cache_size']}")
        print(f"   命中次数: {cache_stats['cache_hits']}")
        print(f"   未命中次数: {cache_stats['cache_misses']}")
        print(f"   命中率: {cache_stats['hit_rate']:.1%}")
        
        # 通信统计
        comm_stats = diagnostics['communication_stats']
        print(f"\n通信统计:")
        print(f"   发送消息数: {comm_stats['messages_sent']}")
        print(f"   接收消息数: {comm_stats['messages_received']}")
        print(f"   传输张量数: {comm_stats['tensors_transferred']}")
        print(f"   注册组件数: {comm_stats['registered_components']}")
        
    finally:
        print("\n5. 正在关闭...")
        await agi.shutdown()
    
    print("\n" + "="*80)
    print("演示完成")
    print("="*80)


async def demo_representation_space():
    """演示统一表征空间"""
    from cognitive.representation import UnifiedRepresentationSpace
    
    print("\n" + "="*80)
    print("演示: 统一表征空间")
    print("="*80)
    
    print("1. 正在创建统一表征空间...")
    repr_space = UnifiedRepresentationSpace(embedding_dim=512)
    
    # 测试多模态编码
    print("2. 正在编码多模态输入...")
    
    # 文本输入
    text_input = {'text': "这是一个文本输入"}
    text_repr = repr_space.encode(text_input, use_cache=False)
    print(f"   文本编码形状: {text_repr.shape}")
    
    # 结构化输入
    struct_input = {'structured': {'value': 0.7, 'category': 'demo'}}
    struct_repr = repr_space.encode(struct_input, use_cache=False)
    print(f"   结构化编码形状: {struct_repr.shape}")
    
    # 多模态输入
    multimodal_input = {
        'text': "多模态测试",
        'structured': {'value': 0.5}
    }
    multimodal_repr = repr_space.encode(multimodal_input, use_cache=False)
    print(f"   多模态编码形状: {multimodal_repr.shape}")
    
    # 相似度计算
    print("3. 正在计算相似度...")
    similarity_same = repr_space.get_similarity(text_repr, text_repr)
    print(f"   相同文本相似度: {similarity_same:.3f}")
    
    similarity_different = repr_space.get_similarity(text_repr, struct_repr)
    print(f"   文本与结构化相似度: {similarity_different:.3f}")
    
    # 缓存演示
    print("4. 演示缓存...")
    repr_space.clear_cache()
    
    # 第一次编码（缓存未命中）
    repr_space.encode(text_input, use_cache=True)
    stats1 = repr_space.get_cache_stats()
    print(f"   第一次编码后: {stats1['cache_misses']} 次未命中, {stats1['cache_hits']} 次命中")
    
    # 相同输入的第二次编码（缓存命中）
    repr_space.encode(text_input, use_cache=True)
    stats2 = repr_space.get_cache_stats()
    print(f"   第二次编码后: {stats2['cache_misses']} 次未命中, {stats2['cache_hits']} 次命中")
    
    print("\n" + "="*80)
    print("演示完成")
    print("="*80)


async def demo_neural_communication():
    """演示神经通信"""
    from neural.communication import NeuralCommunication
    
    print("\n" + "="*80)
    print("演示: 神经通信系统")
    print("="*80)
    
    print("1. 正在创建神经通信系统...")
    comm = NeuralCommunication(max_shared_memory_mb=100)
    
    print("2. 正在注册组件...")
    comm.register_component("perception", "perception")
    comm.register_component("memory", "memory")
    comm.register_component("reasoning", "reasoning")
    
    stats = comm.get_statistics()
    print(f"   注册组件数: {stats['registered_components']}")
    
    print("3. 正在创建共享张量...")
    tensor1 = comm.create_shared_tensor("demo_tensor1", shape=(3, 224, 224))
    tensor2 = comm.create_shared_tensor("demo_tensor2", shape=(1, 512))
    
    stats = comm.get_statistics()
    print(f"   共享张量数: {stats['shared_tensors']}")
    print(f"   内存使用量: {stats['current_memory_usage_mb']:.1f}MB / {stats['max_memory_mb']}MB")
    
    print("4. 正在获取共享张量...")
    retrieved = comm.get_shared_tensor("demo_tensor1")
    if retrieved is not None:
        print(f"   检索到的张量形状: {retrieved.shape}")
    
    print("5. 正在重置统计信息...")
    comm.reset_statistics()
    stats = comm.get_statistics()
    print(f"   重置后发送的消息数: {stats['messages_sent']}")
    
    print("\n" + "="*80)
    print("演示完成")
    print("="*80)


async def main():
    """主演示函数"""
    print("统一认知架构 - 演示")
    print("此演示展示了新统一架构的核心组件。")
    print("")
    
    # 运行演示
    await demo_neural_communication()
    await demo_representation_space()
    await demo_cognitive_cycle()
    
    print("\n" + "="*80)
    print("所有演示成功完成")
    print("="*80)
    print("\n下一步:")
    print("1. 启动API服务器: python -m api.server --port 9000")
    print("2. 运行测试: python tests/run_tests.py")
    print("3. 查看README.md获取更多信息")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n演示被用户中断。")
    except Exception as e:
        print(f"\n\n演示期间发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)