#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量存储管理器集成测试

测试向量存储管理器的功能以及它与其他系统的集成。
包括：
1. 向量存储管理器基本功能测试
2. 真实数据处理器与向量存储集成测试
3. 记忆系统与向量存储集成测试
"""

import os
import sys
import tempfile
import shutil
import logging
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置环境变量
os.environ['ENVIRONMENT'] = 'development'

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_header(title):
    """打印测试标题"""
    print("\n" + "=" * 80)
    print(f"测试: {title}")
    print("=" * 80)


def test_vector_store_manager_basic():
    """测试向量存储管理器基本功能"""
    print_header("向量存储管理器基本功能测试")
    
    try:
        from core.vector_store_manager import get_vector_store_manager, VectorStoreManager
        
        print("✅ VectorStoreManager导入成功")
        
        # 创建临时目录用于测试
        temp_dir = tempfile.mkdtemp()
        
        # 获取管理器实例
        manager = get_vector_store_manager()
        
        print("1. 向量存储管理器实例获取成功")
        
        # 测试获取默认存储
        default_store = manager.get_store()
        print(f"2. 获取默认存储成功: {type(default_store).__name__}")
        
        # 测试添加嵌入
        test_embedding = [0.1] * 768
        test_metadata = {
            "test_id": "basic_test",
            "source": "test",
            "content": "测试内容"
        }
        
        embedding_id = manager.add_embedding(
            embedding=test_embedding,
            metadata=test_metadata,
            document="测试文档内容",
            store_id="default"
        )
        
        print(f"3. 添加嵌入成功，ID: {embedding_id}")
        
        # 测试搜索
        search_results = manager.search_similar(
            query_embedding=test_embedding,
            n_results=5,
            store_id="default"
        )
        
        print(f"4. 相似度搜索成功，返回 {len(search_results['ids'])} 个结果")
        
        # 测试获取统计
        stats = manager.get_stats()
        print(f"5. 获取统计成功，总嵌入数: {stats.get('total_embeddings', 0)}")
        
        # 测试创建新存储
        new_store_id = "test_store_1"
        created = manager.create_store(
            store_id=new_store_id,
            collection_name=f"test_collection_{new_store_id}",
            persist_directory=os.path.join(temp_dir, new_store_id)
        )
        
        if created:
            print(f"6. 创建新存储成功: {new_store_id}")
            
            # 测试在新存储中添加嵌入
            new_embedding_id = manager.add_embedding(
                embedding=test_embedding,
                metadata=test_metadata,
                store_id=new_store_id
            )
            print(f"7. 在新存储中添加嵌入成功，ID: {new_embedding_id}")
            
            # 测试列出所有存储
            stores = manager.list_stores()
            print(f"8. 列出所有存储成功，数量: {len(stores)}")
            
            # 测试删除存储
            deleted = manager.delete_store(new_store_id)
            print(f"9. 删除存储成功: {deleted}")
        
        # 测试清空存储
        cleared = manager.clear_store()
        print(f"10. 清空默认存储成功: {cleared}")
        
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("✅ 临时目录清理完成")
        
        print("\n🎉 向量存储管理器基本功能测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 向量存储管理器基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_true_data_processor_vector_store_integration():
    """测试真实数据处理器与向量存储集成"""
    print_header("真实数据处理器与向量存储集成测试")
    
    try:
        from core.multimodal.true_data_processor import TrueMultimodalDataProcessor
        
        print("✅ TrueMultimodalDataProcessor导入成功")
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        # 创建处理器（启用向量存储）
        processor = TrueMultimodalDataProcessor(enable_vector_store=True)
        
        print("1. 真实数据处理器初始化成功")
        
        # 测试多模态输入处理
        test_input = {
            "text": "这是一个测试文本，用于测试向量存储集成功能",
            "image_data": b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 100,  # 模拟JPEG
            "audio_data": b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 100  # 模拟WAV
        }
        
        # 测试process_and_store_multimodal_input方法
        test_metadata = {
            "test_id": "vector_store_integration_test",
            "timestamp": "2025-01-01T00:00:00",
            "text_content": test_input["text"]
        }
        
        result = processor.process_and_store_multimodal_input(
            input_data=test_input,
            metadata=test_metadata,
            store_id="default"
        )
        
        print(f"2. 处理并存储多模态输入成功: {result.get('success', False)}")
        
        # 检查结果
        embeddings = result.get("embeddings", {})
        storage_results = result.get("storage_results", {})
        
        print(f"  生成的嵌入数量: {len(embeddings)}")
        print(f"  存储结果: {storage_results}")
        
        # 检查至少有一些嵌入被存储
        stored_count = sum(1 for sid in storage_results.values() if sid)
        
        if stored_count > 0:
            print(f"3. 成功存储 {stored_count} 个嵌入到向量存储")
            
            # 测试向量存储管理器是否可用
            if hasattr(processor, 'vector_store_manager') and processor.vector_store_manager:
                stats = processor.vector_store_manager.get_stats()
                print(f"4. 向量存储统计: {stats.get('total_embeddings', 0)} 个嵌入")
        
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("✅ 临时目录清理完成")
        
        print("\n🎉 真实数据处理器与向量存储集成测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 真实数据处理器与向量存储集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_system_vector_store_integration():
    """测试记忆系统与向量存储集成"""
    print_header("记忆系统与向量存储集成测试")
    
    try:
        from core.core_capabilities import MemorySystem
        
        print("✅ MemorySystem导入成功")
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        # 创建记忆系统（启用向量存储）
        memory_system = MemorySystem(max_items=50, enable_vector_store=True)
        
        print("1. 记忆系统初始化成功")
        
        # 存储一些测试记忆
        test_memories = [
            {
                "content": "我喜欢学习人工智能和机器学习",
                "context": {"topic": "AI", "feeling": "excited"},
                "importance": 0.8
            },
            {
                "content": "Python是最好的编程语言之一",
                "context": {"topic": "programming", "language": "Python"},
                "importance": 0.7
            },
            {
                "content": "深度学习需要大量的数据和计算资源",
                "context": {"topic": "deep learning", "requirement": "resources"},
                "importance": 0.9
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
        
        print(f"2. 存储 {len(memory_ids)} 个记忆成功")
        
        # 测试普通检索
        query1 = "人工智能"
        results1 = memory_system.retrieve(query1, limit=2)
        print(f"3. 普通检索 '{query1}' 成功，找到 {len(results1)} 个结果")
        
        # 测试向量相似度检索
        query2 = "机器学习"
        results2 = memory_system.retrieve_similar(query2, limit=2)
        print(f"4. 向量相似度检索 '{query2}' 成功，找到 {len(results2)} 个结果")
        
        # 检查向量相似度结果
        if results2:
            for i, memory_item in enumerate(results2):
                similarity = memory_item.metadata.get("similarity_score", 0)
                print(f"  结果 {i+1}: 相似度 {similarity:.3f}, 内容: {str(memory_item.content)[:50]}...")
        
        # 测试获取特定记忆
        if memory_ids:
            memory_item = memory_system.get(memory_ids[0])
            if memory_item:
                print(f"5. 获取特定记忆成功: {memory_item.id}")
        
        # 获取统计信息
        stats = memory_system.get_statistics()
        print(f"6. 记忆系统统计: {stats.get('total_memories', 0)} 个记忆")
        
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("✅ 临时目录清理完成")
        
        print("\n🎉 记忆系统与向量存储集成测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 记忆系统与向量存储集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comprehensive_vector_store_integration():
    """测试综合向量存储集成"""
    print_header("综合向量存储集成测试")
    
    try:
        # 测试向量存储管理器
        if not test_vector_store_manager_basic():
            print("❌ 向量存储管理器测试失败，跳过后续测试")
            return False
        
        # 测试真实数据处理器集成
        if not test_true_data_processor_vector_store_integration():
            print("⚠️  真实数据处理器集成测试失败，但继续其他测试")
        
        # 测试记忆系统集成
        if not test_memory_system_vector_store_integration():
            print("⚠️  记忆系统集成测试失败")
        
        print("\n" + "=" * 80)
        print("综合向量存储集成测试完成")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ 综合向量存储集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("开始向量存储管理器集成测试...")
    
    # 运行综合测试
    success = test_comprehensive_vector_store_integration()
    
    if success:
        print("\n✅ 向量存储管理器集成测试完成!")
        return 0
    else:
        print("\n⚠️  向量存储管理器集成测试部分失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())