#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量存储系统集成演示

演示向量存储系统的完整功能，包括：
1. 向量存储管理器使用
2. 真实数据处理器集成
3. 记忆系统集成
4. 性能测试和缓存功能
"""

import sys
import os
import time
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['ENVIRONMENT'] = 'development'


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 80)
    print(f"演示: {title}")
    print("=" * 80)


def demo_vector_store_manager():
    """演示向量存储管理器"""
    print_header("向量存储管理器演示")
    
    try:
        from core.vector_store_manager import get_vector_store_manager
        
        # 获取管理器实例
        manager = get_vector_store_manager()
        print("✅ 向量存储管理器初始化成功")
        
        # 获取默认存储
        store = manager.get_store()
        print(f"✅ 获取默认存储成功: {type(store).__name__}")
        
        # 添加测试嵌入向量
        test_embedding = [float(i % 10) * 0.1 for i in range(768)]
        test_metadata = {
            "modality": "text",
            "source": "demo",
            "timestamp": "2026-03-06T00:00:00",
            "content": "向量存储系统集成演示"
        }
        
        embedding_id = manager.add_embedding(
            embedding=test_embedding,
            metadata=test_metadata,
            document="这是一个向量存储系统集成演示的测试文档",
            store_id="default"
        )
        print(f"✅ 添加嵌入向量成功，ID: {embedding_id}")
        
        # 搜索相似向量
        query_embedding = [float(i % 10) * 0.12 for i in range(768)]
        start_time = time.time()
        results = manager.search_similar(
            query_embedding=query_embedding,
            n_results=3,
            store_id="default"
        )
        search_time = time.time() - start_time
        
        print(f"✅ 相似度搜索成功，耗时: {search_time:.3f}秒")
        print(f"   找到 {len(results['ids'])} 个结果")
        
        # 显示结果
        for i, (embedding_id, distance) in enumerate(zip(results['ids'], results['distances'])):
            metadata = results['metadatas'][i]
            content_preview = metadata.get('content', '')[:50]
            print(f"   结果 {i+1}: ID={embedding_id[:8]}..., 距离={distance:.3f}, 内容='{content_preview}...'")
        
        # 获取统计信息
        stats = manager.get_stats()
        print(f"✅ 获取统计信息成功:")
        print(f"   总嵌入数: {stats.get('total_embeddings', 0)}")
        print(f"   查询次数: {stats.get('query_count', 0)}")
        print(f"   插入次数: {stats.get('insert_count', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 向量存储管理器演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_true_data_processor_integration():
    """演示真实数据处理器集成"""
    print_header("真实数据处理器集成演示")
    
    try:
        from core.multimodal.true_data_processor import TrueMultimodalDataProcessor
        
        # 创建启用向量存储的处理器
        processor = TrueMultimodalDataProcessor(enable_vector_store=True)
        print("✅ 真实数据处理器初始化成功")
        
        # 模拟多模态输入
        test_input = {
            "text": "向量存储系统集成演示 - 这是一个测试文本，用于验证向量存储与真实数据处理器的集成功能。",
            "image_data": b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 100,  # 模拟JPEG
        }
        
        # 处理并存储多模态输入
        start_time = time.time()
        result = processor.process_and_store_multimodal_input(
            input_data=test_input,
            metadata={
                "source": "demo",
                "timestamp": "2026-03-06T00:00:00",
                "purpose": "integration_test"
            },
            store_id="default"
        )
        process_time = time.time() - start_time
        
        print(f"✅ 处理并存储多模态输入成功，耗时: {process_time:.3f}秒")
        
        # 显示结果
        embeddings = result.get("embeddings", {})
        storage_results = result.get("storage_results", {})
        
        print(f"   生成的嵌入类型: {list(embeddings.keys())}")
        print(f"   存储结果:")
        for modality, storage_id in storage_results.items():
            if storage_id:
                print(f"     {modality}: 存储成功 (ID: {storage_id[:8]}...)")
            else:
                print(f"     {modality}: 存储失败")
        
        return True
        
    except Exception as e:
        print(f"❌ 真实数据处理器集成演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_memory_system_integration():
    """演示记忆系统集成"""
    print_header("记忆系统集成演示")
    
    try:
        from core.core_capabilities import MemorySystem
        
        # 创建启用向量存储的记忆系统
        memory_system = MemorySystem(max_items=20, enable_vector_store=True)
        print("✅ 记忆系统初始化成功")
        
        # 存储测试记忆
        test_memories = [
            {
                "content": "我喜欢学习人工智能和机器学习，特别是深度学习技术",
                "context": {"topic": "AI", "feeling": "excited", "difficulty": "medium"},
                "importance": 0.9
            },
            {
                "content": "Python是最好的编程语言之一，适合数据科学和机器学习",
                "context": {"topic": "programming", "language": "Python", "rating": 5},
                "importance": 0.8
            },
            {
                "content": "深度学习需要大量的计算资源和数据，但效果非常显著",
                "context": {"topic": "deep learning", "requirement": "resources", "benefit": "performance"},
                "importance": 0.7
            }
        ]
        
        memory_ids = []
        for memory in test_memories:
            memory_id = memory_system.store(
                content=memory["content"],
                context=memory["context"],
                importance=memory["importance"]
            )
            memory_ids.append(memory_id)
        
        print(f"✅ 存储 {len(memory_ids)} 个记忆成功")
        
        # 测试传统关键词检索
        query1 = "人工智能"
        start_time = time.time()
        results1 = memory_system.retrieve(query1, limit=2)
        search_time1 = time.time() - start_time
        
        print(f"✅ 传统关键词检索 '{query1}' 成功，耗时: {search_time1:.3f}秒")
        print(f"   找到 {len(results1)} 个记忆")
        
        # 测试向量相似度检索
        query2 = "机器学习"
        start_time = time.time()
        results2 = memory_system.retrieve_similar(query2, limit=2)
        search_time2 = time.time() - start_time
        
        print(f"✅ 向量相似度检索 '{query2}' 成功，耗时: {search_time2:.3f}秒")
        print(f"   找到 {len(results2)} 个记忆")
        
        # 显示向量检索结果详情
        if results2:
            print("   向量检索结果详情:")
            for i, memory_item in enumerate(results2):
                similarity = memory_item.metadata.get("similarity_score", 0)
                content_preview = str(memory_item.content)[:60]
                print(f"     {i+1}. 相似度: {similarity:.3f}, 内容: '{content_preview}...'")
        
        # 获取记忆统计
        stats = memory_system.get_statistics()
        print(f"✅ 记忆系统统计:")
        print(f"   总记忆数: {stats.get('total_memories', 0)}")
        print(f"   总访问次数: {stats.get('total_accesses', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 记忆系统集成演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_performance_and_caching():
    """演示性能和缓存功能"""
    print_header("性能和缓存功能演示")
    
    try:
        from core.vector_store.memory_vector_store import MemoryVectorStore
        
        # 创建带缓存的向量存储
        vector_store = MemoryVectorStore(
            collection_name="performance_demo",
            embedding_dimension=384
        )
        print("✅ 内存向量存储初始化成功")
        
        # 添加测试数据
        num_test_embeddings = 10
        test_embeddings = []
        test_metadatas = []
        
        for i in range(num_test_embeddings):
            # 生成随机嵌入向量
            embedding = [np.random.normal(0, 0.1) for _ in range(384)]
            test_embeddings.append(embedding)
            test_metadatas.append({
                "index": i,
                "content": f"测试内容 {i}",
                "category": "performance_test"
            })
        
        # 批量添加
        start_time = time.time()
        ids = vector_store.add_embeddings(
            embeddings=test_embeddings,
            metadatas=test_metadatas
        )
        add_time = time.time() - start_time
        
        print(f"✅ 批量添加 {len(ids)} 个嵌入向量成功，耗时: {add_time:.3f}秒")
        print(f"   平均每个嵌入: {add_time/len(ids)*1000:.2f}毫秒")
        
        # 第一次查询（未缓存）
        query_embedding = test_embeddings[0]
        start_time = time.time()
        results1 = vector_store.search_similar(query_embedding, n_results=5)
        search_time1 = time.time() - start_time
        
        print(f"✅ 第一次查询（未缓存）成功，耗时: {search_time1:.3f}秒")
        
        # 第二次相同查询（应命中缓存）
        start_time = time.time()
        results2 = vector_store.search_similar(query_embedding, n_results=5)
        search_time2 = time.time() - start_time
        
        print(f"✅ 第二次查询（缓存命中）成功，耗时: {search_time2:.3f}秒")
        
        # 计算缓存性能提升
        if search_time1 > 0:
            speedup = search_time1 / search_time2 if search_time2 > 0 else 0
            print(f"   缓存性能提升: {speedup:.1f}倍加速")
        
        # 显示缓存统计
        cache_size = vector_store.query_cache.size()
        print(f"   当前缓存大小: {cache_size} 个项目")
        
        # 测试带过滤条件的查询
        start_time = time.time()
        filtered_results = vector_store.search_similar(
            query_embedding=query_embedding,
            n_results=3,
            where={"category": "performance_test", "index": {"$lt": 5}}
        )
        filter_time = time.time() - start_time
        
        print(f"✅ 带过滤条件查询成功，耗时: {filter_time:.3f}秒")
        print(f"   过滤后找到 {len(filtered_results['ids'])} 个结果")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能和缓存演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主演示函数"""
    print("=" * 80)
    print("向量存储系统集成演示")
    print("版本: 1.0.0 | 日期: 2026-03-06")
    print("=" * 80)
    
    print("\n📋 演示内容:")
    print("1. 向量存储管理器基本功能")
    print("2. 真实数据处理器集成")
    print("3. 记忆系统集成")
    print("4. 性能和缓存功能")
    
    total_tests = 4
    passed_tests = 0
    
    # 运行所有演示
    if demo_vector_store_manager():
        passed_tests += 1
    
    if demo_true_data_processor_integration():
        passed_tests += 1
    
    if demo_memory_system_integration():
        passed_tests += 1
    
    if demo_performance_and_caching():
        passed_tests += 1
    
    # 总结
    print("\n" + "=" * 80)
    print("演示总结")
    print("=" * 80)
    
    if passed_tests == total_tests:
        print(f"✅ 所有 {total_tests} 个演示全部成功!")
        print("\n🎉 向量存储系统集成验证完成!")
        print("系统已成功集成以下组件:")
        print("  • 向量存储管理器 (VectorStoreManager)")
        print("  • 真实数据处理器 (TrueMultimodalDataProcessor)")
        print("  • 记忆系统 (MemorySystem)")
        print("  • 内存向量存储 (MemoryVectorStore) 带缓存")
        print("\n📚 相关文档:")
        print("  • VECTOR_STORE_SYSTEM_GUIDE.md - 完整使用指南")
        print("  • 多模态系统技术评估与路线图报告.md - 技术架构")
        print("  • tests/ - 测试框架")
        return 0
    else:
        print(f"⚠️  演示部分成功: {passed_tests}/{total_tests}")
        print("部分功能需要进一步调试")
        return 1


if __name__ == "__main__":
    sys.exit(main())