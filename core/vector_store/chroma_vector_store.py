"""
ChromaDB向量存储实现

基于ChromaDB的向量存储系统，支持多模态嵌入的存储和检索。
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """ChromaDB向量存储"""
    
    def __init__(self, 
                 collection_name: str = "multimodal_embeddings",
                 persist_directory: str = "./chroma_db",
                 embedding_dimension: int = 768):
        """
        初始化ChromaDB向量存储
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
            embedding_dimension: 嵌入维度
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_dimension = embedding_dimension
        
        # 初始化ChromaDB客户端
        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory,
                anonymized_telemetry=False
            )
        )
        
        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"加载现有集合: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "多模态嵌入存储", "created_at": datetime.now().isoformat()}
            )
            logger.info(f"创建新集合: {collection_name}")
        
        logger.info(f"ChromaVectorStore初始化完成，嵌入维度: {embedding_dimension}")
    
    def add_embeddings(self,
                      embeddings: List[List[float]],
                      metadatas: List[Dict[str, Any]],
                      documents: Optional[List[str]] = None,
                      ids: Optional[List[str]] = None) -> List[str]:
        """
        添加嵌入向量到存储
        
        Args:
            embeddings: 嵌入向量列表
            metadatas: 元数据列表
            documents: 文档内容列表（可选）
            ids: 自定义ID列表（可选）
            
        Returns:
            生成的ID列表
        """
        if not embeddings:
            raise ValueError("嵌入向量列表不能为空")
        
        if not metadatas:
            raise ValueError("元数据列表不能为空")
        
        if len(embeddings) != len(metadatas):
            raise ValueError(f"嵌入向量数量({len(embeddings)})与元数据数量({len(metadatas)})不匹配")
        
        if documents and len(embeddings) != len(documents):
            raise ValueError(f"嵌入向量数量({len(embeddings)})与文档数量({len(documents)})不匹配")
        
        # 生成ID（如果未提供）
        if not ids:
            ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
        
        # 验证嵌入维度
        for i, emb in enumerate(embeddings):
            if len(emb) != self.embedding_dimension:
                raise ValueError(f"嵌入向量{i}的维度({len(emb)})与期望维度({self.embedding_dimension})不匹配")
        
        # 添加嵌入到集合
        try:
            self.collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents if documents else [""] * len(embeddings),
                ids=ids
            )
            logger.info(f"成功添加{len(embeddings)}个嵌入向量")
            return ids
        except Exception as e:
            logger.error(f"添加嵌入向量失败: {e}")
            raise
    
    def search_similar(self,
                      query_embedding: List[float],
                      n_results: int = 10,
                      where: Optional[Dict[str, Any]] = None,
                      where_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        搜索相似的嵌入向量
        
        Args:
            query_embedding: 查询嵌入向量
            n_results: 返回结果数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            
        Returns:
            搜索结果
        """
        if len(query_embedding) != self.embedding_dimension:
            raise ValueError(f"查询嵌入维度({len(query_embedding)})与期望维度({self.embedding_dimension})不匹配")
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            
            # 格式化结果
            formatted_results = {
                "ids": results["ids"][0] if results["ids"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "embeddings": results["embeddings"][0] if results["embeddings"] else []
            }
            
            logger.debug(f"相似度搜索完成，返回{len(formatted_results['ids'])}个结果")
            return formatted_results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            raise
    
    def get_by_id(self, ids: Union[str, List[str]]) -> Dict[str, Any]:
        """
        根据ID获取嵌入向量
        
        Args:
            ids: ID或ID列表
            
        Returns:
            嵌入向量信息
        """
        if isinstance(ids, str):
            ids = [ids]
        
        try:
            results = self.collection.get(ids=ids)
            
            return {
                "ids": results["ids"],
                "metadatas": results["metadatas"],
                "documents": results["documents"],
                "embeddings": results["embeddings"]
            }
        except Exception as e:
            logger.error(f"根据ID获取嵌入向量失败: {e}")
            raise
    
    def update_embedding(self,
                        id: str,
                        embedding: Optional[List[float]] = None,
                        metadata: Optional[Dict[str, Any]] = None,
                        document: Optional[str] = None) -> bool:
        """
        更新嵌入向量
        
        Args:
            id: 要更新的ID
            embedding: 新的嵌入向量（可选）
            metadata: 新的元数据（可选）
            document: 新的文档内容（可选）
            
        Returns:
            是否成功
        """
        try:
            # 首先获取现有数据
            existing = self.get_by_id(id)
            if not existing["ids"]:
                raise ValueError(f"ID {id} 不存在")
            
            # 更新字段
            current_metadata = existing["metadatas"][0] if existing["metadatas"] else {}
            current_document = existing["documents"][0] if existing["documents"] else ""
            current_embedding = existing["embeddings"][0] if existing["embeddings"] else None
            
            # 合并更新
            updated_metadata = {**current_metadata, **(metadata or {})}
            updated_document = document if document is not None else current_document
            updated_embedding = embedding if embedding is not None else current_embedding
            
            if updated_embedding is None:
                raise ValueError("更新需要提供嵌入向量")
            
            # 删除旧记录并添加新记录
            self.collection.delete(ids=[id])
            self.collection.add(
                embeddings=[updated_embedding],
                metadatas=[updated_metadata],
                documents=[updated_document],
                ids=[id]
            )
            
            logger.info(f"成功更新嵌入向量: {id}")
            return True
            
        except Exception as e:
            logger.error(f"更新嵌入向量失败: {e}")
            raise
    
    def delete_embeddings(self, ids: Union[str, List[str]]) -> bool:
        """
        删除嵌入向量
        
        Args:
            ids: ID或ID列表
            
        Returns:
            是否成功
        """
        if isinstance(ids, str):
            ids = [ids]
        
        try:
            self.collection.delete(ids=ids)
            logger.info(f"成功删除{len(ids)}个嵌入向量")
            return True
        except Exception as e:
            logger.error(f"删除嵌入向量失败: {e}")
            raise
    
    def count(self) -> int:
        """获取集合中的嵌入向量数量"""
        try:
            count = self.collection.count()
            return count
        except Exception as e:
            logger.error(f"获取数量失败: {e}")
            raise
    
    def list_all(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """
        列出所有嵌入向量
        
        Args:
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            嵌入向量列表
        """
        try:
            # 获取所有ID
            all_ids = self.collection.get()["ids"]
            
            if not all_ids:
                return {"ids": [], "metadatas": [], "documents": [], "embeddings": []}
            
            # 分页
            start_idx = offset
            end_idx = min(offset + limit, len(all_ids))
            page_ids = all_ids[start_idx:end_idx]
            
            return self.get_by_id(page_ids)
        except Exception as e:
            logger.error(f"列出嵌入向量失败: {e}")
            raise
    
    def clear_collection(self) -> bool:
        """清空集合"""
        try:
            # 删除集合并重新创建
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "多模态嵌入存储", "created_at": datetime.now().isoformat()}
            )
            logger.info(f"成功清空集合: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            count = self.count()
            
            return {
                "collection_name": self.collection_name,
                "embedding_dimension": self.embedding_dimension,
                "persist_directory": self.persist_directory,
                "total_embeddings": count,
                "created_at": self.collection.metadata.get("created_at", "unknown")
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            raise


def test_chroma_vector_store():
    """测试ChromaVectorStore"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== 测试ChromaVectorStore ===")
    
    try:
        # 创建向量存储
        vector_store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory="./test_chroma_db",
            embedding_dimension=384  # 使用较小的维度进行测试
        )
        
        print("1. 向量存储初始化成功")
        
        # 生成测试嵌入向量
        test_embeddings = [
            [0.1 * i for i in range(384)],
            [0.2 * i for i in range(384)],
            [0.3 * i for i in range(384)]
        ]
        
        test_metadatas = [
            {"modality": "text", "source": "test1", "timestamp": "2024-01-01"},
            {"modality": "image", "source": "test2", "timestamp": "2024-01-02"},
            {"modality": "audio", "source": "test3", "timestamp": "2024-01-03"}
        ]
        
        test_documents = [
            "这是第一个测试文档",
            "这是第二个测试文档",
            "这是第三个测试文档"
        ]
        
        # 添加嵌入向量
        ids = vector_store.add_embeddings(
            embeddings=test_embeddings,
            metadatas=test_metadatas,
            documents=test_documents
        )
        
        print(f"2. 添加{len(ids)}个嵌入向量成功，ID: {ids}")
        
        # 搜索相似向量
        query_embedding = [0.15 * i for i in range(384)]
        results = vector_store.search_similar(query_embedding, n_results=2)
        
        print(f"3. 相似度搜索成功，返回{len(results['ids'])}个结果")
        print(f"   最相似ID: {results['ids'][0] if results['ids'] else '无'}")
        
        # 根据ID获取
        retrieved = vector_store.get_by_id(ids[0])
        print(f"4. 根据ID获取成功: {retrieved['ids'][0]}")
        
        # 更新嵌入向量
        updated = vector_store.update_embedding(
            id=ids[0],
            metadata={"modality": "text", "source": "updated", "timestamp": "2024-01-04"}
        )
        print(f"5. 更新嵌入向量成功: {updated}")
        
        # 获取数量
        count = vector_store.count()
        print(f"6. 集合数量: {count}")
        
        # 列出所有
        all_items = vector_store.list_all(limit=2)
        print(f"7. 列出所有成功，获取{len(all_items['ids'])}个项目")
        
        # 获取集合信息
        info = vector_store.get_collection_info()
        print(f"8. 集合信息: {info}")
        
        # 删除嵌入向量
        deleted = vector_store.delete_embeddings(ids[2])
        print(f"9. 删除嵌入向量成功: {deleted}")
        
        # 清空集合
        cleared = vector_store.clear_collection()
        print(f"10. 清空集合成功: {cleared}")
        
        print("✅ 所有测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_chroma_vector_store()