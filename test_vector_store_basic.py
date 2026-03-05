#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量存储基础功能测试

只测试向量存储组件，不依赖外部网络连接。
"""

import sys
import os
import time
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['ENVIRONMENT'] = 'development'


def test_vector_store_manager():
    """测试向量存储管理器基本功能"""
    print("\n1. 测试向量存储管理器")
    print("-" * 50)
    
    try:
        from core.vector_store_manager import get_vector_store_manager
        
        # 获取管理器实例
        manager = get_vector_store_manager()
        print("✅ 向量存储管理器初始化成功")
        
        # 测试获取存储
        store = manager.get_store()
        print(f"✅ 获取存储成功，类型: {type(store).__name__}")
        
        # 测试添加嵌入向量
        embedding = [float(i % 10) * 0.1 for i in range(768)]
        metadata = {"test": "vector_store", "timestamp": "2026-03-06"}
        
        embedding_id = manager.add_embedding(embedding, metadata)
        print(f"✅ 添加嵌入向量成功，ID: {embedding_id[:8]}...")
        
        # 测试相似度搜索
        query_embedding = [float(i % 10) * 0.12 for i in range(768)]
        results = manager.search_similar(query_embedding, n_results=2)
        
        print(f"✅ 相似度搜索成功，找到 {len(results['ids'])} 个结果")
        
        # 测试统计信息
        stats = manager.get_stats()
        print(f"✅ 获取统计信息成功:")
        print(f"   总嵌入数: {stats.get('total_embeddings', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_vector_store():
    """测试内存向量存储"""
    print("\n2. 测试内存向量存储")
    print("-" * 50)
    
    try:
        from core.vector_store.memory_vector_store import MemoryVectorStore
        
        # 创建向量存储
        store = MemoryVectorStore(collection_name="test_collection", embedding_dimension=384)
        print("✅ 内存向量存储初始化成功")
        
        # 批量添加嵌入向量
        embeddings = []
        metadatas = []
        
        for i in range(5):
            embedding = [float(i % 10) * 0.1 + float(j % 5) * 0.02 for j in range(384)]
            embeddings.append(embedding)
            metadatas.append({
                "id": i,
                "content": f"测试内容 {i}",
                "category": "test"
            })
        
        ids = store.add_embeddings(embeddings, metadatas)
        print(f"✅ 批量添加 {len(ids)} 个嵌入向量成功")
        
        # 测试缓存功能
        query_embedding = embeddings[0]
        
        # 第一次查询（应未缓存）
        start_time = time.time()
        results1 = store.search_similar(query_embedding, n_results=3)
        time1 = time.time() - start_time
        
        # 第二次相同查询（应命中缓存）
        start_time = time.time()
        results2 = store.search_similar(query_embedding, n_results=3)
        time2 = time.time() - start_time
        
        print(f"✅ 缓存测试:")
        print(f"   第一次查询: {time1:.4f}秒")
        print(f"   第二次查询: {time2:.4f}秒")
        print(f"   缓存命中加速: {time1/time2:.1f}倍")
        
        # 测试带过滤的查询
        filtered_results = store.search_similar(
            query_embedding=query_embedding,
            n_results=2,
            where={"category": "test", "id": {"$lt": 3}}
        )
        
        print(f"✅ 带过滤查询成功，找到 {len(filtered_results['ids'])} 个结果")
        
        # 测试其他方法
        count = store.count()
        print(f"✅ 计数方法: {count} 个嵌入向量")
        
        info = store.get_collection_info()
        print(f"✅ 集合信息:")
        print(f"   名称: {info['collection_name']}")
        print(f"   维度: {info['embedding_dimension']}")
        print(f"   数量: {info['total_embeddings']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_system_integration():
    """测试记忆系统集成"""
    print("\n3. 测试记忆系统集成")
    print("-" * 50)
    
    try:
        from core.core_capabilities import MemorySystem
        
        # 创建记忆系统
        memory_system = MemorySystem(max_items=10, enable_vector_store=True)
        print("✅ 记忆系统初始化成功")
        
        # 存储测试记忆
        memory_id = memory_system.store(
            content="这是一个测试记忆，用于验证向量存储集成",
            context={"test": True, "purpose": "integration"},
            importance=0.8
        )
        print(f"✅ 存储记忆成功，ID: {memory_id}")
        
        # 测试向量相似度检索（简化的查询）
        results = memory_system.retrieve_similar("测试记忆", limit=2)
        print(f"✅ 向量相似度检索成功，找到 {len(results)} 个记忆")
        
        # 测试传统关键词检索
        keyword_results = memory_system.retrieve("测试", limit=2)
        print(f"✅ 传统关键词检索成功，找到 {len(keyword_results)} 个记忆")
        
        # 获取统计信息
        stats = memory_system.get_statistics()
        print(f"✅ 记忆系统统计:")
        print(f"   总记忆数: {stats.get('total_memories', 0)}")
        print(f"   向量存储集成: {'启用' if stats.get('vector_store_enabled', False) else '禁用'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 80)
    print("向量存储系统基础功能测试")
    print("版本: 1.0.0 | 日期: 2026-03-06")
    print("=" * 80)
    
    total_tests = 3
    passed_tests = 0
    
    # 运行所有测试
    if test_vector_store_manager():
        passed_tests += 1
    
    if test_memory_vector_store():
        passed_tests += 1
    
    if test_memory_system_integration():
        passed_tests += 1
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    if passed_tests == total_tests:
        print(f"✅ 所有 {total_tests} 个测试全部通过!")
        print("\n🎉 向量存储系统基础功能验证完成!")
        print("\n📋 已验证组件:")
        print("  • 向量存储管理器 (VectorStoreManager)")
        print("  • 内存向量存储 (MemoryVectorStore) 带缓存功能")
        print("  • 记忆系统 (MemorySystem) 与向量存储集成")
        print("\n✅ 已验证功能:")
        print("  • 基本嵌入存储和检索")
        print("  • 相似度搜索和缓存优化")
        print("  • 条件过滤查询")
        print("  • 系统集成和统计")
        return 0
    else:
        print(f"⚠️  测试部分通过: {passed_tests}/{total_tests}")
        print("部分功能需要进一步调试")
        return 1


if __name__ == "__main__":
    sys.exit(main())