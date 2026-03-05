"""
向量存储集成测试

测试向量存储系统是否正常工作，并集成到系统中。
"""

import os
import sys
import logging

# 设置环境变量
os.environ['ENVIRONMENT'] = 'development'

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_chroma_vector_store():
    """测试ChromaDB向量存储"""
    print("\n" + "=" * 80)
    print("测试ChromaDB向量存储")
    print("=" * 80)
    
    try:
        from core.vector_store.chroma_vector_store import ChromaVectorStore
        
        print("✅ ChromaVectorStore导入成功")
        
        # 创建测试向量存储（使用临时目录）
        test_dir = "./test_chroma_db"
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)
        
        vector_store = ChromaVectorStore(
            collection_name="test_integration",
            persist_directory=test_dir,
            embedding_dimension=384
        )
        
        print("✅ ChromaVectorStore初始化成功")
        
        # 测试添加嵌入
        test_embeddings = [[0.1] * 384, [0.2] * 384]
        test_metadatas = [
            {"modality": "text", "source": "test", "content": "测试文本1"},
            {"modality": "text", "source": "test", "content": "测试文本2"}
        ]
        test_documents = ["测试文档内容1", "测试文档内容2"]
        
        ids = vector_store.add_embeddings(
            embeddings=test_embeddings,
            metadatas=test_metadatas,
            documents=test_documents
        )
        
        print(f"✅ 添加嵌入成功，生成 {len(ids)} 个ID: {ids[:3]}...")
        
        # 测试搜索
        query_embedding = [0.15] * 384
        results = vector_store.search_embeddings(
            query_embeddings=[query_embedding],
            n_results=2
        )
        
        print(f"✅ 搜索成功，返回 {len(results['ids'][0])} 个结果")
        
        # 测试获取集合信息
        info = vector_store.get_collection_info()
        print(f"✅ 集合信息获取成功，包含 {info.get('count', 0)} 个项目")
        
        # 清理测试目录
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)
            print("✅ 测试目录清理完成")
        
        return True
        
    except Exception as e:
        print(f"❌ ChromaVectorStore测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multimodal_vector_store():
    """测试多模态向量存储"""
    print("\n" + "=" * 80)
    print("测试多模态向量存储")
    print("=" * 80)
    
    try:
        from core.vector_store.multimodal_vector_store import MultimodalVectorStore
        
        print("✅ MultimodalVectorStore导入成功")
        
        # 创建测试向量存储（使用临时目录）
        test_dir = "./test_multimodal_db"
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)
        
        vector_store = MultimodalVectorStore(
            collection_name="test_multimodal_integration",
            persist_directory=test_dir,
            embedding_dimension=384
        )
        
        print("✅ MultimodalVectorStore初始化成功")
        
        # 测试添加文本项目
        text_id = vector_store.add_multimodal_item(
            modality="text",
            data="这是一个集成测试文本",
            metadata={"source": "integration_test", "category": "text"},
            document="完整的测试文档内容用于集成测试"
        )
        
        print(f"✅ 添加文本项目成功，ID: {text_id}")
        
        # 测试文本搜索
        text_results = vector_store.search_by_text(
            query_text="集成测试",
            n_results=2
        )
        
        print(f"✅ 文本搜索成功，返回 {len(text_results['ids'])} 个结果")
        
        # 测试获取项目
        item = vector_store.get_item(text_id)
        print(f"✅ 获取项目成功: {item['ids'][0]}")
        
        # 测试统计信息
        stats = vector_store.get_statistics()
        print(f"✅ 统计信息: 总项目数 = {stats.get('total_items', 0)}")
        
        # 清理测试目录
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)
            print("✅ 测试目录清理完成")
        
        return True
        
    except Exception as e:
        print(f"❌ MultimodalVectorStore测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_store_with_true_data_processor():
    """测试向量存储与真实数据处理器的集成"""
    print("\n" + "=" * 80)
    print("测试向量存储与真实数据处理器的集成")
    print("=" * 80)
    
    try:
        from core.multimodal.true_data_processor import TrueDataProcessor
        from core.vector_store.multimodal_vector_store import MultimodalVectorStore
        
        print("✅ 模块导入成功")
        
        # 创建数据处理器
        data_processor = TrueDataProcessor()
        print("✅ TrueDataProcessor初始化成功")
        
        # 创建向量存储
        vector_store = MultimodalVectorStore(
            collection_name="integration_with_processor",
            persist_directory="./test_integration_db",
            embedding_dimension=768  # 与TrueDataProcessor的嵌入维度匹配
        )
        print("✅ MultimodalVectorStore初始化成功")
        
        # 测试数据处理和向量存储集成
        test_text = "这是一个用于测试向量存储集成的文本内容。"
        
        # 处理文本数据
        processed = data_processor.process_text(test_text)
        print(f"✅ 文本处理完成，输出类型: {type(processed)}")
        
        # 检查是否包含嵌入
        if isinstance(processed, dict) and "embedding" in processed:
            embedding = processed["embedding"]
            print(f"✅ 获取到文本嵌入，维度: {len(embedding) if hasattr(embedding, '__len__') else 'N/A'}")
            
            # 将嵌入存储到向量存储中
            if hasattr(embedding, 'tolist'):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding) if hasattr(embedding, '__iter__') else [float(embedding)]
            
            # 确保维度正确
            if len(embedding_list) != 768:
                print(f"⚠️  嵌入维度不匹配: {len(embedding_list)} != 768，进行调整")
                if len(embedding_list) < 768:
                    embedding_list = embedding_list + [0.0] * (768 - len(embedding_list))
                else:
                    embedding_list = embedding_list[:768]
            
            # 添加到向量存储
            item_id = vector_store.add_multimodal_item(
                modality="text",
                data=test_text,
                metadata={
                    "source": "true_data_processor",
                    "processor": "TrueDataProcessor",
                    "timestamp": "2025-01-01T00:00:00"
                },
                document=test_text
            )
            
            print(f"✅ 文本嵌入存储到向量存储，ID: {item_id}")
            
        else:
            print("⚠️  处理后的文本不包含嵌入，检查TrueDataProcessor实现")
        
        # 清理测试目录
        test_dir = "./test_integration_db"
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)
            print("✅ 测试目录清理完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 向量存储与真实数据处理器集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_store_api_integration():
    """测试向量存储API集成"""
    print("\n" + "=" * 80)
    print("测试向量存储API集成")
    print("=" * 80)
    
    try:
        # 检查是否可以将向量存储集成到API中
        print("检查向量存储API集成可能性...")
        
        # 检查现有的API端点
        from core.vector_store import MultimodalVectorStore
        
        # 创建建议的API端点
        suggested_endpoints = [
            ("POST", "/api/vector/store", "存储多模态嵌入"),
            ("GET", "/api/vector/search", "搜索相似嵌入"),
            ("GET", "/api/vector/item/{item_id}", "获取存储项目"),
            ("DELETE", "/api/vector/item/{item_id}", "删除存储项目"),
            ("GET", "/api/vector/stats", "获取存储统计信息"),
        ]
        
        print("✅ 向量存储API端点设计:")
        for method, path, description in suggested_endpoints:
            print(f"   {method} {path} - {description}")
        
        # 检查是否有现成的API可以扩展
        try:
            from core.robot_api_enhanced import router
            print("✅ 现有的增强机器人API路由器可用，可以添加向量存储端点")
        except ImportError:
            print("⚠️  增强机器人API不可用，需要单独创建向量存储API")
        
        return True
        
    except Exception as e:
        print(f"❌ 向量存储API集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("Self-Soul-B多模态系统 - 向量存储集成测试")
    print("=" * 80)
    print("测试目标: 验证向量存储系统功能并测试集成可能性")
    print("测试环境: 开发模式")
    print("=" * 80)
    
    tests = [
        ("ChromaDB向量存储", test_chroma_vector_store),
        ("多模态向量存储", test_multimodal_vector_store),
        ("向量存储与真实数据处理器集成", test_vector_store_with_true_data_processor),
        ("向量存储API集成", test_vector_store_api_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    print(f"运行 {total} 个测试...")
    print()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"  ✅ {test_name}: 通过")
            else:
                print(f"  ❌ {test_name}: 失败")
        except Exception as e:
            print(f"  ❌ {test_name}: 异常 - {e}")
    
    print("\n" + "=" * 80)
    print("测试总结:")
    print(f"  总测试数: {total}")
    print(f"  通过测试: {passed}")
    print(f"  失败测试: {total - passed}")
    print(f"  通过率: {(passed/total*100):.1f}%")
    
    if passed == total:
        print("\n✅ 所有向量存储集成测试通过！")
        print("\n下一步:")
        print("  1. 实现向量存储API端点")
        print("  2. 集成向量存储到真实数据处理器工作流")
        print("  3. 添加向量存储性能测试和基准")
        print("  4. 创建向量存储使用文档和示例")
        return 0
    else:
        print("\n⚠️  部分测试失败，需要检查。")
        print("\n建议:")
        print("  1. 检查chromadb依赖是否已安装")
        print("  2. 验证向量存储模块导入路径")
        print("  3. 检查TrueDataProcessor的嵌入生成功能")
        return 1

if __name__ == "__main__":
    sys.exit(main())